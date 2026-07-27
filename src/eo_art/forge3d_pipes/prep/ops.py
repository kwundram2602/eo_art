"""Prep ops. Each takes (src, dst, cfg) and returns the written path."""

from dataclasses import dataclass
from pathlib import Path

import rasterio
from omegaconf import MISSING
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from rasterio.warp import reproject as _rio_reproject

from eo_art.forge3d_pipes.config.schema import ResamplingName
from eo_art.forge3d_pipes.prep.registry import register_op

_RESAMPLING = {
    ResamplingName.nearest: Resampling.nearest,
    ResamplingName.bilinear: Resampling.bilinear,
    ResamplingName.cubic: Resampling.cubic,
}


@dataclass
class ReprojectCfg:
    crs: str = MISSING
    resampling: ResamplingName = ResamplingName.bilinear


@dataclass
class ScaleToGsdCfg:
    target_gsd: float = MISSING
    resampling: ResamplingName = ResamplingName.bilinear


@register_op("reproject", ReprojectCfg)
def reproject(src: Path, dst: Path, cfg: ReprojectCfg) -> Path:
    """Reproject a raster to ``cfg.crs``, preserving all bands."""
    with rasterio.open(src) as source:
        if source.crs is None:
            raise ValueError(f"{src} has no CRS; cannot reproject")
        transform, width, height = calculate_default_transform(
            source.crs, cfg.crs, source.width, source.height, *source.bounds
        )
        meta = source.meta.copy()
        meta.update(
            {"crs": cfg.crs, "transform": transform, "width": width, "height": height}
        )
        with rasterio.open(dst, "w", **meta) as destination:
            for band in range(1, source.count + 1):
                _rio_reproject(
                    source=rasterio.band(source, band),
                    destination=rasterio.band(destination, band),
                    src_transform=source.transform,
                    dst_transform=transform,
                    src_crs=source.crs,
                    dst_crs=cfg.crs,
                    resampling=_RESAMPLING[cfg.resampling],
                )
    return Path(dst)


@register_op("scale_to_gsd", ScaleToGsdCfg)
def scale_to_gsd(src: Path, dst: Path, cfg: ScaleToGsdCfg) -> Path:
    """Resample to a target ground sample distance via vecraspy."""
    from vecraspy import scale_raster_to_gsd

    scale_raster_to_gsd(src, dst, cfg.target_gsd, resampling=cfg.resampling.value)
    return Path(dst)


@dataclass
class SaturateCfg:
    factor: float = 1.0  # 0 = grayscale, 1 = unchanged, >1 = more vivid


@register_op("saturate", SaturateCfg)
def saturate(src: Path, dst: Path, cfg: SaturateCfg) -> Path:
    """Blend each RGB pixel toward (factor<1) or away from (factor>1) its
    luminance (ITU-R BT.601 weights). Bands beyond the first three, if any,
    pass through unchanged."""
    with rasterio.open(src) as source:
        if source.count < 3:
            raise ValueError(
                f"saturate requires at least 3 bands (RGB), got {source.count}"
            )
        r, g, b = source.read(1), source.read(2), source.read(3)
        extra = [source.read(i) for i in range(4, source.count + 1)]
        meta = source.meta.copy()

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    r2 = (gray + (r - gray) * cfg.factor).astype(meta["dtype"])
    g2 = (gray + (g - gray) * cfg.factor).astype(meta["dtype"])
    b2 = (gray + (b - gray) * cfg.factor).astype(meta["dtype"])

    with rasterio.open(dst, "w", **meta) as destination:
        destination.write(r2, 1)
        destination.write(g2, 2)
        destination.write(b2, 3)
        for index, band in enumerate(extra, start=4):
            destination.write(band, index)
    return Path(dst)
