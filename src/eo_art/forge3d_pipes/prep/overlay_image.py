"""Export a prepped raster as a viewer-loadable image.

forge3d's live-viewer ``load_overlay`` reads the overlay through Rust's
``image`` crate, which does not support TIFF at all (confirmed: "The image
format Tiff is not supported"). Every prep op writes GeoTIFF, so this
converts the final prepped raster into a PNG right before it's handed to the
viewer. Georeferencing is not needed at this point — placement already comes
from the normalized extent computed against the GeoTIFF beforehand.
"""

import hashlib
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import rasterio

from eo_art.forge3d_pipes.prep.ops import (
    _apply_stretch,
    _percentile_bounds,
    _rgb_band_indexes,
)

# Matches RgbStretchCfg's defaults (prep/ops.py) so an overlay that already
# went through rgb_stretch and one that didn't get the same stretch policy.
_LOWER_PERCENTILE = 2.0
_UPPER_PERCENTILE = 98.0


def export_overlay_png_cached(
    src: Path,
    cache_dir: Path,
    use_cache: bool = True,
    alpha_src: Path | None = None,
) -> Path:
    """Cache ``export_overlay_png`` by ``src``'s (and ``alpha_src``'s, if
    given) content identity.

    Mirrors ``prep.registry.run_prep_chain``'s caching: the same source file
    always maps to the same destination name, so repeated calls (e.g. once
    per sweep variant) reuse a single PNG instead of re-exporting it.
    """
    src = Path(src)
    stat = src.stat()
    payload = f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    if alpha_src is not None:
        alpha_src = Path(alpha_src)
        alpha_stat = alpha_src.stat()
        payload += (
            f"|{alpha_src.resolve()}|{alpha_stat.st_size}|{alpha_stat.st_mtime_ns}"
        )
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]
    dst = Path(cache_dir) / f"{key}_overlay.png"
    if use_cache and dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    return export_overlay_png(src, dst, alpha_src=alpha_src)


def export_overlay_png(src: Path, dst: Path, alpha_src: Path | None = None) -> Path:
    """Write the first three bands of ``src`` (or one band, replicated) as
    an 8-bit RGBA PNG at ``dst``.

    Alpha normally comes from ``src``'s own validity mask (rasterio's
    ``read_masks``, which folds in the nodata value, a per-dataset mask, or
    an existing alpha band). Fully transparent where invalid, fully opaque
    where valid -- this is what keeps a partially-overlapping overlay from
    painting solid black over the terrain once it's been clipped to the
    DEM's full extent.

    ``alpha_src``, if given, is a single-band uint8 raster (see the
    ``validity_mask`` prep op) whose values are used verbatim as alpha
    *instead of* deriving it from ``src``. Needed for the ``align_to_dem``
    path: ``align_to`` only fills the area outside the source's original
    footprint with nodata if the source declared one to begin with, so an
    overlay with no nodata set would otherwise -- after alignment -- read as
    fully valid everywhere, including the fill. ``alpha_src`` carries the
    correct answer, captured before that information could be lost.

    Value range is dtype-driven: a floating-point source is
    percentile-stretched to [0, 255], an integer source is clipped as
    before. In this pipeline a float raster is never already 8-bit display
    data, so a uint8-range float overlay (e.g. one promoted by
    ``reproject``'s ``dtype: auto``) still round-trips correctly --
    percentile-stretching values already in [0, 255] maps them back onto
    approximately the same range.
    """
    with rasterio.open(src) as source:
        indexes = _rgb_band_indexes(source.count)
        is_float = np.issubdtype(np.dtype(source.dtypes[0]), np.floating)
        bands = source.read(indexes)
        if alpha_src is None:
            # min(), not any()/all(): a pixel counts as valid only where
            # every band read contributes to it, matching how the three
            # channels are about to be merged into one RGB(A) pixel.
            alpha = source.read_masks(indexes).min(axis=0).astype("uint8")

    if alpha_src is not None:
        with rasterio.open(alpha_src) as mask_source:
            alpha = mask_source.read(1)

    if len(indexes) == 1:
        bands = np.repeat(bands, 3, axis=0)

    if is_float:
        bands = bands.astype("float64")
        pooled = bands[:, alpha > 0].ravel()
        lo, hi = _percentile_bounds(pooled, _LOWER_PERCENTILE, _UPPER_PERCENTILE)
        rgb = _apply_stretch(bands, lo, hi)
    else:
        rgb = np.clip(bands, 0, 255).astype("uint8")

    rgb = np.moveaxis(rgb, 0, -1)
    rgba = np.dstack([rgb, alpha])
    iio.imwrite(dst, rgba)
    return Path(dst)
