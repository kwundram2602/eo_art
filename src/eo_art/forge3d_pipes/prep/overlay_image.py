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


def export_overlay_png_cached(
    src: Path, cache_dir: Path, use_cache: bool = True
) -> Path:
    """Cache ``export_overlay_png`` by ``src``'s content identity.

    Mirrors ``prep.registry.run_prep_chain``'s caching: the same source file
    always maps to the same destination name, so repeated calls (e.g. once
    per sweep variant) reuse a single PNG instead of re-exporting it.
    """
    src = Path(src)
    stat = src.stat()
    payload = f"{src.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]
    dst = Path(cache_dir) / f"{key}_overlay.png"
    if use_cache and dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    return export_overlay_png(src, dst)


def export_overlay_png(src: Path, dst: Path) -> Path:
    """Write the first three bands of ``src`` (or one band, replicated) as
    an 8-bit RGB PNG at ``dst``. Values are clipped to [0, 255]."""
    with rasterio.open(src) as source:
        if source.count >= 3:
            bands = source.read([1, 2, 3])
        else:
            band = source.read(1)
            bands = np.stack([band, band, band])

    rgb = np.clip(bands, 0, 255).astype("uint8")
    rgb = np.moveaxis(rgb, 0, -1)
    iio.imwrite(dst, rgb)
    return Path(dst)
