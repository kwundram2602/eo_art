"""Sentinel-2 RGBN acquisition and super-resolution."""

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Without these, a stalled connection to the COG/STAC host (e.g. host
# unreachable mid-download) leaves GDAL's HTTP layer blocking forever instead
# of erroring out, since it has no timeout by default.
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")

import cubo
import mlstac
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 (registers the .rio accessor on xarray objects)
import torch
import torch.nn.functional as F
import xarray as xr
from torch.nn.attention import SDPBackend, sdpa_kernel

from sentinel_sr.clouds import DEFAULT_CLOUD_CLASSES, SCL_BAND, cloud_percent
from sentinel_sr.scenes import dedupe
from sentinel_sr.timeseries import Bins, select_timesteps

BANDS_10 = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


@dataclass(frozen=True)
class Scene:
    path: Path
    cloud_cover: float | None
    # Acquisition date (YYYY-MM-DD).
    datetime: str | None = None
    # Percent of AOI pixels in the SCL cloud classes (None: check disabled).
    aoi_cloud: float | None = None


@dataclass(frozen=True)
class ExportResult:
    scenes: list[Scene]
    # Timeseries runs only: start dates of intervals without any scene.
    skipped_bins: list[str]


def _patch_vae_attention_for_low_vram() -> None:
    """Replace LDSRS2-SEN2SR's VAE attention (opensr_model) with an SDPA-based
    equivalent, forced onto the memory-efficient backend.

    The vendored implementation materializes a full [hw, hw] attention matrix
    by hand (1024 MiB at this model's 128x128 resolution) and PyTorch's default
    SDPA backend selection silently falls back to that same math kernel here
    (Flash Attention refuses head_dim=512 > 256). Forcing EFFICIENT_ATTENTION
    drops that single allocation to ~130 MiB with numerically identical output
    (verified: max abs diff ~5e-7 against the original), which is what makes
    this model's inference fit on a 4 GB GPU at all.
    """
    from opensr_model.autoencoder.utils import AttnBlock

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        v = v.reshape(b, c, h * w).permute(0, 2, 1).contiguous()

        if q.is_cuda:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])
        else:
            out = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])
        out = out[:, 0].permute(0, 2, 1).reshape(b, c, h, w)

        return x + self.proj_out(out)

    AttnBlock.forward = forward


def _patch_swin_for_low_vram(chunk: int = 512) -> None:
    """Chunk Swin2SR's per-window and per-token work (sen2sr) over its batch axis.

    LDSRS2-SEN2SR's two Swin2SR stages run on the whole 512x512 SR image at
    once, so a 128px input gives B_ = 4096 windows / L = 262144 tokens and
    every one of the 64 blocks materializes three tensors far too large for a
    4 GB GPU: the [B_, heads, N, N] attention scores (512 MiB), the [B_, N, 3C]
    qkv projection (360 MiB), and the block MLP's [B, L, 4C] hidden state
    (480 MiB, twice -- linear then GELU).

    Windows attend independently and the MLP is token-wise, so slicing the
    batch/token axis is exact rather than approximate (verified: bit-identical
    output against the original, masked and unmasked path alike) and caps each
    of those tensors at a chunk's worth. Unlike an SDPA rewrite for the
    attention half, this needs no kernel support for the model's odd head_dim
    of 15 (embed_dim 120 / 8 heads), and it covers the qkv and MLP tensors an
    SDPA swap would leave untouched.
    """
    from sen2sr.models.opensr_baseline.swin import Mlp, WindowAttention

    def attention_forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1.0 / 0.01)).to(self.logit_scale.device),
        ).exp()

        bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        bias = bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        bias = 16 * torch.sigmoid(bias.permute(2, 0, 1).contiguous()).unsqueeze(0)

        # The (nW, N, N) mask applies to windows cyclically over B_ = B * nW
        # (what the original expresses as a view/broadcast over both axes), so
        # tiling it to B_ once lets it be sliced in lockstep with the chunks.
        mask_full = (
            mask.repeat(B_ // mask.shape[0], 1, 1).unsqueeze(1)
            if mask is not None
            else None
        )

        out = torch.empty_like(x)
        for start in range(0, B_, chunk):
            stop = min(start + chunk, B_)
            qkv = F.linear(input=x[start:stop], weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(stop - start, N, 3, self.num_heads, -1).permute(
                2, 0, 3, 1, 4
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            attn = attn * logit_scale
            attn = attn + bias
            if mask_full is not None:
                attn = attn + mask_full[start:stop]
            attn = self.softmax(attn)

            out[start:stop] = (attn @ v).transpose(1, 2).reshape(stop - start, N, C)

        return self.proj_drop(self.proj(out))

    # The MLP is applied per token, so it chunks along the token axis. Its
    # chunk is scaled up because a token is far smaller than a whole window.
    token_chunk = chunk * 64

    def mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, tokens, _ = x.shape
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty(
            flat.shape[0], self.fc2.out_features, dtype=x.dtype, device=x.device
        )
        for start in range(0, flat.shape[0], token_chunk):
            stop = min(start + token_chunk, flat.shape[0])
            out[start:stop] = self.fc2(self.act(self.fc1(flat[start:stop])))
        return out.view(*lead, tokens, self.fc2.out_features)

    WindowAttention.forward = attention_forward
    Mlp.forward = mlp_forward


def band_descriptions(da: xr.DataArray, bands: tuple[str, ...]) -> list[str]:
    """GeoTIFF band descriptions from the STAC eo:bands common names.

    stackstac (via cubo) carries each asset's eo:bands ``common_name`` as a
    coordinate on the band axis. Names shared by several bands (Planetary
    Computer tags B05/B06/B07 all as ``rededge``) get the band id appended so
    every description stays unique; bands without a common name keep their id.
    """
    common = (
        [str(name) if name else "" for name in da.coords["common_name"].values]
        if "common_name" in da.coords
        else [""] * len(bands)
    )
    names = []
    for band, name in zip(bands, common, strict=True):
        if not name or name in ("None", "nan"):
            names.append(band)
        elif common.count(name) > 1:
            names.append(f"{name}_{band}")
        else:
            names.append(name)
    return names


def export_superres_tifs(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    model_path: str | Path,
    collection: str = "sentinel-2-l2a",
    bands: tuple[str, ...] = BANDS_10,
    edge_size: int = 128,
    resolution: int = 10,
    stac: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    max_items: int | None = None,
    max_cloud_cover: float | None = None,
    device: torch.device | None = None,
    interval: str | None = None,
    every: int = 1,
    overwrite: bool = False,
    max_aoi_cloud: float | None = None,
    scl_classes: tuple[int, ...] = DEFAULT_CLOUD_CLASSES,
    skip_dates: Sequence[str] = (),
) -> ExportResult:
    """Super-resolve Sentinel-2 acquisitions for an AOI (center point +
    edge_size) and write one GeoTIFF per timestep.

    Without ``interval`` every acquisition is processed. With one (``day``,
    ``week``, ``month``, ``year``) only the least cloudy scene per ``every``
    intervals is, chosen from STAC metadata before any pixels are fetched.
    Scenes are fetched, super-resolved and written one at a time; an existing
    output is reused unless ``overwrite``, so an interrupted run resumes.

    With ``max_aoi_cloud`` (percent) the SCL band is fetched for every
    candidate first: scenes with more AOI pixels in ``scl_classes`` are
    dropped, and a timeseries ranks the rest by that AOI cloud share instead
    of the tile-wide ``eo:cloud_cover``.

    ``skip_dates`` (timeseries only): intervals containing one of these dates
    are already done (e.g. timesteps from an earlier run) and get no scene;
    only the other intervals are searched, so a series can be filled up.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # cubo.create forwards unknown kwargs straight to the STAC item search, so
    # `query` filters scenes server-side on the eo:cloud_cover property before
    # anything is downloaded (this is scene-level metadata, not a pixel mask).
    query = {"eo:cloud_cover": {"lt": max_cloud_cover}} if max_cloud_cover is not None else None

    da = cubo.create(
        lat=lat,
        lon=lon,
        collection=collection,
        # SCL rides along in the same cube, so its items and grid match.
        bands=[*bands, SCL_BAND] if max_aoi_cloud is not None else list(bands),
        start_date=start_date,
        end_date=end_date,
        stac=stac,
        edge_size=edge_size,
        resolution=resolution,
        query=query,
    )
    epsg = da.attrs["epsg"]
    spectral = da.sel(band=list(bands))
    band_names = band_descriptions(spectral, bands)

    # stackstac (via cubo) promotes STAC item properties to per-timestep
    # coordinates, so cloud cover travels with the scene; not every
    # collection carries it (only those using the eo extension).
    n_all = da.sizes["time"]
    cloud_covers: list[float | None] = (
        [None if pd.isna(c) else float(c) for c in da.coords["eo:cloud_cover"].values]
        if "eo:cloud_cover" in da.coords
        else [None] * n_all
    )
    # One item per overflight: Planetary Computer lists the same acquisition
    # once per covering tile and once per processing baseline.
    deduped = dedupe([str(i) for i in da.id.values], cloud_covers, epsg=int(epsg))
    unique = deduped.kept
    print(f"{n_all} scenes, {len(unique)} after removing duplicates")
    if deduped.other_zone:
        print(
            f"Skipped {len(deduped.other_zone)} acquisition(s) only available "
            f"from tiles outside EPSG:{epsg} (reprojecting them leaves seams): "
            + ", ".join(deduped.other_zone)
        )

    if interval is not None and skip_dates:
        # Before the SCL fetch: done intervals don't need their candidates.
        bins = Bins(start_date, end_date, interval, every)
        filled = bins.of(skip_dates)
        unique = [i for i in unique if bins.index(da.time.values[i]) not in filled]
        print(
            f"Timeseries: {len(filled)} interval(s) already filled, "
            f"{len(unique)} candidate scene(s) left for the others"
        )

    aoi_cloud: dict[int, float] = {}
    if max_aoi_cloud is not None and unique:
        # One parallel fetch of the small SCL chips of all candidates.
        scl = da.sel(band=SCL_BAND).isel(time=unique).compute().to_numpy()
        aoi_cloud = {
            i: cloud_percent(chip, scl_classes)
            for i, chip in zip(unique, scl, strict=True)
        }
        cloudy = [i for i in unique if aoi_cloud[i] > max_aoi_cloud]
        unique = [i for i in unique if aoi_cloud[i] <= max_aoi_cloud]
        print(
            f"SCL check (classes {list(scl_classes)}): {len(cloudy)} scene(s) over "
            f"{max_aoi_cloud}% cloud in the AOI dropped, {len(unique)} left"
            + (
                ": "
                + ", ".join(
                    f"{pd.Timestamp(da.time.values[i]).date()} "
                    f"({aoi_cloud[i]:.1f}%)"
                    for i in cloudy
                )
                if cloudy
                else ""
            )
        )

    skipped_bins: list[str] = []
    if interval is None:
        selected = unique
        if max_items is not None:
            selected = selected[:max_items]
    else:
        selection = select_timesteps(
            da.time.values[unique],
            # The AOI's own cloud share when known, else the tile's.
            [aoi_cloud.get(i, cloud_covers[i]) for i in unique],
            start_date,
            end_date,
            interval,
            every,
            filled_dates=skip_dates,
        )
        selected = [unique[i] for i in selection.indices]
        skipped_bins = selection.skipped_bins
        print(
            f"Timeseries: {len(selected)} of {len(unique)} scenes selected "
            f"(one per {every} {interval}(s)), {len(skipped_bins)} empty interval(s)"
            + (f": {', '.join(skipped_bins)}" if skipped_bins else "")
        )
        # The interval decides how many scenes a timeseries has; capping it
        # would silently cut the series short.
        if max_items is not None:
            print(f"Timeseries: ignoring max_items={max_items}")

    _patch_vae_attention_for_low_vram()
    _patch_swin_for_low_vram()
    loaded_model = mlstac.load(str(model_path))
    model = loaded_model.compiled_model(device=device)

    n_time = len(selected)
    in_edge = loaded_model.item.properties["mlm:input"][0]["input"]["shape"][-1]
    out_edge = loaded_model.item.properties["mlm:output"][0]["result"]["shape"][-1]
    scale = out_edge / in_edge
    out_edge = int(edge_size * scale)
    bytes_per_tif = len(bands) * out_edge * out_edge * 4  # float32
    total_bytes = bytes_per_tif * n_time
    warnings.warn(
        f"export_superres_tifs will run inference on {n_time} timesteps, "
        f"writing ~{total_bytes / 1e9:.2f} GB total "
        f"({bytes_per_tif / 1e6:.1f} MB per GeoTIFF, {out_edge}x{out_edge}px, "
        f"{len(bands)} bands, float32).",
        stacklevel=2,
    )

    written_paths = []
    for step, i in enumerate(selected, start=1):
        da_t = spectral[i]
        date = pd.Timestamp(da_t.time.values).date().isoformat()
        # `id` (not `time`) is the unique key: overlapping tiles from the same
        # orbit, or duplicate reprocessing baselines, can share the same timestamp.
        scene_id = str(da_t.id.values)
        name = f"superres_{date}_{scene_id}" if interval else f"superres_{scene_id}"
        out_path = output_dir / f"{name}.tif"
        scene = Scene(
            path=out_path,
            cloud_cover=cloud_covers[i],
            datetime=date,
            aoi_cloud=aoi_cloud.get(i),
        )
        if out_path.exists() and not overwrite:
            written_paths.append(scene)
            print(f"[{step}/{n_time}] Kept existing {out_path}")
            continue

        s2_numpy = (da_t.compute().to_numpy() / 10_000).astype("float32")

        X = torch.from_numpy(s2_numpy).float().to(device)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            superX = model(X[None]).squeeze(0)

        # e.g. SEN2SRLite_RGBN is a x4 model (10 m -> 2.5 m); derived from actual shapes
        # so the wrapper keeps working if a different-scale model is passed in.
        scale = superX.shape[-1] / s2_numpy.shape[-1]
        src_transform = da_t.rio.write_crs(f"EPSG:{epsg}").rio.transform()
        sr_transform = src_transform * rasterio.Affine.scale(1 / scale, 1 / scale)

        sr_da = xr.DataArray(
            superX.detach().cpu().numpy(),
            dims=("band", "y", "x"),
            coords={"band": list(bands)},
        )
        sr_da = sr_da.rio.write_crs(f"EPSG:{epsg}")
        sr_da = sr_da.rio.write_transform(sr_transform)
        # rioxarray writes a list-valued long_name as per-band descriptions.
        sr_da.attrs["long_name"] = band_names

        sr_da.rio.to_raster(out_path)

        written_paths.append(scene)
        print(
            f"[{step}/{n_time}] Wrote {out_path} "
            f"({date}, cloud_cover={scene.cloud_cover})"
        )

    return ExportResult(scenes=written_paths, skipped_bins=skipped_bins)
