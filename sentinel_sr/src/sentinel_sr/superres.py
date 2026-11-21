"""Super-resolve a whole raster of any size instead of one chip.

``export_superres_tifs`` feeds the model one ``edge_size`` chip at a time, so
a grid of chips (eo_art's zones) puts a model border at every chip edge. The
SEN2SR stack treats everything outside its input as zero (zero-padded convs,
the Fourier hard constraint wraps around, the 20 m bands go through
non-renormalised bilinear resampling), which darkens the outer pixel ring.
Neighbouring chips therefore meet with a visible seam.

Here the raster (e.g. an LR mosaic of all zones) is processed as one image,
following stac2cube's large-image path:

1. NaN pixels get the value of their nearest valid neighbour, so the model
   sees no reflectance cliff at nodata or cloud holes (restored to NaN after).
2. Replicate padding of ``EDGE_PAD_LR`` input pixels on every side moves the
   model's own border artifacts into throwaway pixels.
3. ``stac2cube.super_resolution.predict_large_batched`` runs 128 px patches
   with 32 px overlap and drops every patch border, so there are no
   internal seams either.
4. The padding is cropped away: the output covers exactly the input extent.
"""

from pathlib import Path

import mlstac
import numpy as np
import rasterio
import torch
from scipy.ndimage import distance_transform_edt
from stac2cube.super_resolution import PATCH_SIZE, predict_large_batched

from sentinel_sr.rgbn import _patch_swin_for_low_vram, _patch_vae_attention_for_low_vram

# Replicate padding (input pixels) per side. stac2cube measured the border
# bias converged at 8 px; 16 is margin.
EDGE_PAD_LR = 16
PATCH_OVERLAP = 32


def fill_nan_nearest(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``arr`` (bands, y, x) with the ``mask`` pixels set to their nearest
    unmasked pixel (all bands taken from that same pixel)."""
    if not mask.any() or mask.all():
        return arr
    _, (iy, ix) = distance_transform_edt(mask, return_indices=True)
    return arr[:, iy, ix]


def pad_for_patches(
    arr: np.ndarray, pad: int, patch: int = PATCH_SIZE
) -> tuple[np.ndarray, tuple[int, int]]:
    """Replicate-pad ``arr`` by ``pad`` on every side, plus whatever each axis
    still lacks to hold one full ``patch``. Returns (padded, (top, left))."""
    _, ny, nx = arr.shape
    extra_y = max(patch - (ny + 2 * pad), 0)
    extra_x = max(patch - (nx + 2 * pad), 0)
    pad_y = (pad + extra_y // 2, pad + extra_y - extra_y // 2)
    pad_x = (pad + extra_x // 2, pad + extra_x - extra_x // 2)
    padded = np.pad(arr, ((0, 0), pad_y, pad_x), mode="edge")
    return padded, (pad_y[0], pad_x[0])


def superres_array(
    model,
    arr: np.ndarray,
    device: torch.device,
    batch_size: int = 1,
    edge_pad: int = EDGE_PAD_LR,
) -> np.ndarray:
    """SR of a (bands, y, x) reflectance array; NaN stays NaN (per pixel,
    all bands), everything else is filled with model output."""
    mask = np.isnan(arr).any(axis=0)
    filled = fill_nan_nearest(arr, mask)
    padded, (top, left) = pad_for_patches(filled, edge_pad)

    X = torch.from_numpy(np.ascontiguousarray(padded, dtype=np.float32)).to(device)
    # Only an all-NaN raster (nothing to fill from) still has NaN here.
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    out = predict_large_batched(
        model=model, X=X, overlap=PATCH_OVERLAP, batch_size=batch_size
    ).numpy()

    scale = out.shape[-1] // padded.shape[-1]
    ny, nx = arr.shape[1:]
    out = out[:, top * scale : (top + ny) * scale, left * scale : (left + nx) * scale]
    if mask.any():
        out[:, np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)] = np.nan
    return out


def load_model(model_path: str | Path, device: torch.device):
    """The mlstac model, with the low-VRAM patches applied; checks that it
    takes the 128 px patches the large-image predictor cuts."""
    _patch_vae_attention_for_low_vram()
    _patch_swin_for_low_vram()
    loaded = mlstac.load(str(model_path))
    in_edge = loaded.item.properties["mlm:input"][0]["input"]["shape"][-1]
    if in_edge != PATCH_SIZE:
        raise ValueError(
            f"model {model_path} takes {in_edge} px inputs; the large-image "
            f"predictor needs {PATCH_SIZE} px patches"
        )
    return loaded.compiled_model(device=device)


def superres_raster(
    model,
    src: str | Path,
    dst: str | Path,
    device: torch.device,
    batch_size: int = 1,
) -> Path:
    """Super-resolve the GeoTIFF ``src`` (float32 reflectance, bands in the
    model's input order) into ``dst``; band descriptions are kept."""
    dst = Path(dst)
    with rasterio.open(src) as ds:
        arr = ds.read().astype("float32")
        if ds.nodata is not None and not np.isnan(ds.nodata):
            arr[arr == ds.nodata] = np.nan
        profile = ds.profile
        descriptions = ds.descriptions

    _, ny, nx = arr.shape
    step = PATCH_SIZE - PATCH_OVERLAP
    n_patches = (-(-(ny + 2 * EDGE_PAD_LR) // step)) * (
        -(-(nx + 2 * EDGE_PAD_LR) // step)
    )
    print(f"SR {src}: {nx}x{ny} px, ~{n_patches} patches of {PATCH_SIZE} px")
    out = superres_array(model, arr, device, batch_size=batch_size)
    scale = out.shape[-1] / nx

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=out.shape[0],
        width=out.shape[2],
        height=out.shape[1],
        transform=profile["transform"] * rasterio.Affine.scale(1 / scale, 1 / scale),
        nodata=float("nan"),
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        bigtiff="IF_SAFER",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **profile) as out_ds:
        out_ds.write(out.astype("float32"))
        for band, description in enumerate(descriptions, start=1):
            if description:
                out_ds.set_band_description(band, description)
    print(f"Wrote {dst} ({out.shape[2]}x{out.shape[1]} px)")
    return dst


def superres_jobs(
    jobs: list[dict],
    model_path: str | Path,
    device: torch.device | None = None,
    batch_size: int = 1,
    overwrite: bool = False,
) -> list[Path]:
    """Run ``superres_raster`` for every ``{"src": ..., "dst": ...}`` job with
    one model load; existing outputs are kept unless ``overwrite``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = [j for j in jobs if overwrite or not Path(j["dst"]).exists()]
    for job in jobs:
        if job not in todo:
            print(f"Kept existing {job['dst']}")
    if not todo:
        return [Path(j["dst"]) for j in jobs]
    model = load_model(model_path, device)
    for k, job in enumerate(todo, start=1):
        print(f"[{k}/{len(todo)}]", end=" ")
        superres_raster(model, job["src"], job["dst"], device, batch_size)
    return [Path(j["dst"]) for j in jobs]
