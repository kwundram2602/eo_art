"""Sentinel-2 RGBN acquisition and super-resolution."""

import os
import warnings
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
import rasterio
import rioxarray  # noqa: F401 (registers the .rio accessor on xarray objects)
import torch
import torch.nn.functional as F
import xarray as xr
from torch.nn.attention import SDPBackend, sdpa_kernel

BANDS_10 = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


@dataclass(frozen=True)
class Scene:
    path: Path
    cloud_cover: float | None


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
) -> list[Scene]:
    """Super-resolve every Sentinel-2 acquisition for an AOI (center point + edge_size) and write one GeoTIFF per timestep."""
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
        bands=list(bands),
        start_date=start_date,
        end_date=end_date,
        stac=stac,
        edge_size=edge_size,
        resolution=resolution,
        query=query,
    )
    epsg = da.attrs["epsg"]
    _patch_vae_attention_for_low_vram()
    loaded_model = mlstac.load(str(model_path))
    model = loaded_model.compiled_model(device=device)

    n_time = da.sizes["time"]
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
    for i in range(da.sizes["time"]):
        if max_items is not None and i >= max_items:
            break
        da_t = da[i]
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

        # `id` (not `time`) is the unique key: overlapping tiles from the same
        # orbit, or duplicate reprocessing baselines, can share the same timestamp.
        scene_id = str(da_t.id.values)
        out_path = output_dir / f"superres_{scene_id}.tif"
        sr_da.rio.to_raster(out_path)

        # stackstac (via cubo) promotes STAC item properties to per-timestep
        # coordinates, so cloud cover travels with the scene; not every
        # collection carries it (only those using the eo extension).
        cloud_cover = (
            float(da_t.coords["eo:cloud_cover"].item())
            if "eo:cloud_cover" in da_t.coords
            else None
        )
        written_paths.append(Scene(path=out_path, cloud_cover=cloud_cover))
        print(f"Wrote {out_path} (cloud_cover={cloud_cover})")

    return written_paths
