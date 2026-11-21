"""Prep ops. Each takes (src, dst, cfg) and returns the written path."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
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
    ResamplingName.cubic_spline: Resampling.cubic_spline,
    ResamplingName.lanczos: Resampling.lanczos,
    ResamplingName.average: Resampling.average,
}


# Lowercase member names for the same OmegaConf-matches-by-name reason as
# ResamplingName (config/schema.py).
class DtypeMode(Enum):
    auto = "auto"
    preserve = "preserve"
    float32 = "float32"


def _resolve_dst_dtype(
    source_dtype: str, resampling: ResamplingName, mode: DtypeMode
) -> str:
    if mode is DtypeMode.preserve:
        return source_dtype
    if mode is DtypeMode.float32:
        return "float32"
    # auto: nearest only ever copies an existing sample value, so there is no
    # fractional result to lose by leaving the dtype alone. Any other method
    # interpolates, and an integer destination truncates that interpolation
    # straight back into the terracing this promotion exists to prevent.
    if resampling is not ResamplingName.nearest and np.issubdtype(
        np.dtype(source_dtype), np.integer
    ):
        return "float32"
    return source_dtype


@dataclass
class ReprojectCfg:
    crs: str = MISSING
    resampling: ResamplingName = ResamplingName.bilinear
    # This op is generic (also used on overlay rasters, e.g.
    # configs/overlays/ndvi_drape.yaml), so `auto` promoting a uint8 overlay
    # under `bilinear` is a deliberate, accepted side effect -- not DEM-only
    # behaviour. See DtypeMode for the policy semantics.
    dtype: DtypeMode = DtypeMode.auto


@dataclass
class ScaleToGsdCfg:
    target_gsd: float = MISSING
    resampling: ResamplingName = ResamplingName.bilinear


@dataclass
class AlignToCfg:
    reference: str = MISSING
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
        dst_dtype = _resolve_dst_dtype(meta["dtype"], cfg.resampling, cfg.dtype)
        meta.update(
            {
                "crs": cfg.crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": dst_dtype,
            }
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


@register_op("align_to", AlignToCfg)
def align_to(src: Path, dst: Path, cfg: AlignToCfg) -> Path:
    """Snap ``src`` onto ``cfg.reference``'s exact grid via vecraspy.

    Used to re-couple a DEM and an overlay that were reprojected
    independently: each ``reproject`` call picks its own destination grid
    (rasterio's ``calculate_default_transform`` per raster), so two rasters
    reprojected separately land on different pixel grids even when they
    started perfectly aligned.
    """
    from vecraspy import align_raster_grid

    align_raster_grid(cfg.reference, src, dst, resampling=cfg.resampling.value)
    return Path(dst)


def _rgb_band_indexes(count: int) -> list[int]:
    """1-indexed bands ``export_overlay_png`` treats as R, G, B: the first
    three, or the sole band read three times over. Shared with
    ``validity_mask`` so both agree on which bands' validity counts."""
    return [1, 2, 3] if count >= 3 else [1]


@dataclass
class ValidityMaskCfg:
    pass


@register_op("validity_mask", ValidityMaskCfg)
def validity_mask(src: Path, dst: Path, cfg: ValidityMaskCfg) -> Path:
    """Snapshot ``src``'s current per-pixel validity as a standalone
    single-band uint8 raster (0/255), with no nodata of its own.

    Exists to survive a downstream ``align_to``: ``vecraspy.align_raster_grid``
    only fills the area outside the source's footprint with nodata if the
    source dataset already declares one -- an overlay with none set (a
    plausible plain satellite export, no unusual authoring) instead gets a
    validity-less ``0`` fill that ``read_masks()`` then reports as fully
    valid, i.e. opaque black painted over the terrain. This file has no
    such metadata to lose: aligning it alongside the main raster (see
    ``pipeline._align_overlay_to_dem``) produces a 0/255 fill purely from
    the geometry of what the source did or didn't cover, independent of
    whether the source ever set nodata.
    """
    with rasterio.open(src) as source:
        indexes = _rgb_band_indexes(source.count)
        mask = source.read_masks(indexes).min(axis=0).astype("uint8")
        meta = source.meta.copy()

    meta.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(dst, "w", **meta) as destination:
        destination.write(mask, 1)
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
    dtype = np.dtype(meta["dtype"])
    channels = [gray + (channel - gray) * cfg.factor for channel in (r, g, b)]
    if np.issubdtype(dtype, np.integer):
        # A factor > 1 (or < 0) routinely pushes values outside the dtype's
        # range; astype() on an integer dtype wraps silently instead of
        # clamping (e.g. -24.5 -> 232), which is how a correctly dark pixel
        # turns bright and wrong-colored instead of just clipping to black.
        info = np.iinfo(dtype)
        channels = [np.clip(channel, info.min, info.max) for channel in channels]
    r2, g2, b2 = (channel.astype(dtype) for channel in channels)

    with rasterio.open(dst, "w", **meta) as destination:
        destination.write(r2, 1)
        destination.write(g2, 2)
        destination.write(b2, 3)
        for index, band in enumerate(extra, start=4):
            destination.write(band, index)
    return Path(dst)


def _percentile_bounds(
    valid: np.ndarray, lower: float, upper: float
) -> tuple[float, float]:
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [lower, upper])
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


def _apply_stretch(band: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linearly map ``[lo, hi]`` to ``[0, 255]`` uint8, clipping outside it.

    Shared by ``rgb_stretch`` and ``export_overlay_png`` so the clip-scale
    arithmetic exists in exactly one place.
    """
    scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return scaled.astype("uint8")


@dataclass
class RgbStretchCfg:
    # 1-indexed source band numbers to pull as R, G, B. Default matches a
    # 10-band Sentinel-2 stack ordered B02,B03,B04,...: band 3 (B04, red),
    # band 2 (B03, green), band 1 (B02, blue).
    bands: tuple[int, int, int] = (3, 2, 1)
    lower_percentile: float = 2.0
    upper_percentile: float = 98.0
    # A single (lo, hi) is computed from all three bands' pooled pixels and
    # applied identically to each -- stretching each band to its own min/max
    # independently auto-levels every channel separately, which destroys the
    # relative brightness between channels (e.g. water, dark and flat across
    # all bands, ends up rendering pink instead of dark blue). Set False only
    # for deliberately false-color composites where bands aren't meant to
    # share a natural color balance.
    shared_stretch: bool = True
    # Fixed (lo, hi) applied to all three bands instead of percentiles of
    # this raster -- set both or neither. Lets tiles of one mosaic (SR
    # zones) share a stretch, so neighbouring tiles don't differ in
    # brightness; see stretch_bounds_over.
    lower_value: float | None = None
    upper_value: float | None = None


def stretch_bounds_over(
    paths: Sequence[str | Path], cfg: RgbStretchCfg, max_side: int = 256
) -> tuple[float, float]:
    """``rgb_stretch``'s shared (lo, hi), pooled over all of ``paths``.

    Each raster is read decimated to at most ``max_side`` px per side, which
    is plenty for percentiles and keeps many tiles cheap.
    """
    pooled = []
    for path in paths:
        with rasterio.open(path) as source:
            scale = max(source.width, source.height) / max_side
            shape = (
                max(1, round(source.height / max(scale, 1))),
                max(1, round(source.width / max(scale, 1))),
            )
            nodata = source.nodata
            for band_index in cfg.bands:
                band = source.read(band_index, out_shape=shape).astype("float64")
                valid = band[band != nodata] if nodata is not None else band.ravel()
                pooled.append(valid[np.isfinite(valid)])
    values = np.concatenate(pooled) if pooled else np.empty(0)
    return _percentile_bounds(values, cfg.lower_percentile, cfg.upper_percentile)


@register_op("rgb_stretch", RgbStretchCfg)
def rgb_stretch(src: Path, dst: Path, cfg: RgbStretchCfg) -> Path:
    """Select three bands and percentile-stretch them to a true-color uint8 composite.

    ``export_overlay_png`` reads an overlay's first three bands verbatim as
    R, G, B and clips to [0, 255] -- so a multiband reflectance-scaled raster
    (e.g. Sentinel-2 in B02,B03,B04,... order, values roughly 0-1) needs
    reducing to a true-color composite first, or it renders as solid black
    with swapped channels.
    """
    if len(cfg.bands) != 3:
        raise ValueError(
            f"rgb_stretch.bands must have exactly 3 entries, got {cfg.bands!r}"
        )
    if not (0 <= cfg.lower_percentile < cfg.upper_percentile <= 100):
        raise ValueError(
            "rgb_stretch percentiles must satisfy 0 <= lower_percentile < "
            f"upper_percentile <= 100, got lower={cfg.lower_percentile}, "
            f"upper={cfg.upper_percentile}"
        )

    fixed = (cfg.lower_value, cfg.upper_value)
    if (fixed[0] is None) != (fixed[1] is None):
        raise ValueError(
            "rgb_stretch needs both lower_value and upper_value or neither, "
            f"got lower_value={cfg.lower_value}, upper_value={cfg.upper_value}"
        )
    if fixed[0] is not None and fixed[1] is not None and fixed[0] >= fixed[1]:
        raise ValueError(
            "rgb_stretch.lower_value must be below upper_value, got "
            f"{cfg.lower_value} >= {cfg.upper_value}"
        )

    with rasterio.open(src) as source:
        if source.count < max(cfg.bands):
            raise ValueError(
                f"{src} has {source.count} band(s); rgb_stretch.bands={cfg.bands!r} "
                f"needs at least band {max(cfg.bands)}"
            )
        nodata = source.nodata
        meta = source.meta.copy()

        raw_bands = [
            source.read(band_index).astype("float64") for band_index in cfg.bands
        ]

    def _valid(band):
        valid = band[band != nodata] if nodata is not None else band.ravel()
        return valid[np.isfinite(valid)]

    if cfg.lower_value is not None and cfg.upper_value is not None:
        bounds = [(float(cfg.lower_value), float(cfg.upper_value))] * 3
    elif cfg.shared_stretch:
        pooled = np.concatenate([_valid(band) for band in raw_bands])
        bounds = [
            _percentile_bounds(pooled, cfg.lower_percentile, cfg.upper_percentile)
        ] * 3
    else:
        bounds = [
            _percentile_bounds(_valid(band), cfg.lower_percentile, cfg.upper_percentile)
            for band in raw_bands
        ]

    stretched = [
        _apply_stretch(band, lo, hi) for band, (lo, hi) in zip(raw_bands, bounds)
    ]

    meta.update(count=3, dtype="uint8", nodata=None)
    with rasterio.open(dst, "w", **meta) as destination:
        for index, band in enumerate(stretched, start=1):
            destination.write(band, index)
    return Path(dst)
