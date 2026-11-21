"""Overlay preview for a render config, without a GPU.

Runs the same prep chain as ``eo-art-f3d run`` (``pipeline.prepare_scene``,
same cache folder) and shows the overlay PNG the renderer would get, plus the
overlay laid on a hillshade of the prepared DEM as a stand-in for "draped on
the terrain".
"""

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import rasterio
from rasterio.enums import Resampling

from eo_art.forge3d_pipes.config.loader import load_raw, to_pipeline
from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.pipeline import PREP_CACHE_DIRNAME, prepare_scene

# Longest side of the preview images; the renderer's PNG itself is untouched.
MAX_SIDE = 1024
# Light that reaches fully shadowed slopes, so they stay readable.
AMBIENT = 0.35
# Share of the non-ambient light flat ground gets in the preview; the rest
# is headroom for slopes facing the sun.
FLAT_LEVEL = 0.8
# Metres per degree, for DEMs in a geographic CRS.
_M_PER_DEG = 111_320.0


@dataclass(frozen=True)
class OverlayPreview:
    name: str
    source: str  # overlay path as written in the config
    prepare: list[str]  # op names of the overlay's prepare chain
    opacity: float | None
    png_path: Path  # the exact PNG the renderer loads
    overlay_png: bytes  # that PNG, scaled down for display
    relief_png: bytes  # overlay on the DEM hillshade


def hillshade(
    dem: np.ndarray,
    cell: tuple[float, float],
    azimuth: float,
    elevation: float,
    zscale: float = 1.0,
) -> np.ndarray:
    """Lambertian shading in [0, 1]; ``cell`` = (x, y) pixel size in the DEM's
    height units, ``azimuth`` compass-style (0 = north, 90 = east)."""
    dz_drow, dz_dcol = np.gradient(dem.astype("float64"), cell[1], cell[0])
    # Rows run southwards, so the northward slope is the negated row slope.
    dz_dx, dz_dy = dz_dcol * zscale, -dz_drow * zscale
    az, el = math.radians(azimuth), math.radians(elevation)
    sun = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    shade = (-dz_dx * sun[0] - dz_dy * sun[1] + sun[2]) / np.sqrt(
        dz_dx**2 + dz_dy**2 + 1.0
    )
    return np.clip(np.nan_to_num(shade, nan=sun[2]), 0.0, 1.0)


def _absolute(root: Path, path: str) -> str:
    return path if Path(path).is_absolute() else str(root / path)


def _absolute_chain(root: Path, chain: list[Any]) -> list[Any]:
    """Prep ops may name a ``reference`` raster (``align_to``)."""
    return [
        {**op, "reference": _absolute(root, op["reference"])}
        if isinstance(op, dict) and isinstance(op.get("reference"), str)
        else op
        for op in chain
    ]


def _load(
    project_root: Path, layers: Sequence[Path]
) -> tuple[PipelineConfig, dict[str, str]]:
    """The merged config with its paths made absolute (the pipeline resolves
    them against the project folder, its working directory in a run), and
    each overlay's path as written."""
    cfg = to_pipeline(load_raw(list(layers)))
    written = {o.name: o.path for o in cfg.overlays}
    cfg.input.path = _absolute(project_root, cfg.input.path)
    cfg.prepare = _absolute_chain(project_root, cfg.prepare)
    for overlay in cfg.overlays:
        overlay.path = _absolute(project_root, overlay.path)
        overlay.prepare = _absolute_chain(project_root, overlay.prepare)
    return cfg, written


def _read_dem(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """The DEM scaled to at most MAX_SIDE, and its pixel size in metres."""
    with rasterio.open(path) as src:
        factor = max(1.0, max(src.width, src.height) / MAX_SIDE)
        shape = (max(1, round(src.height / factor)), max(1, round(src.width / factor)))
        dem = src.read(1, out_shape=shape, masked=True, resampling=Resampling.bilinear)
        res_x, res_y = abs(src.res[0]) * factor, abs(src.res[1]) * factor
        if src.crs is not None and src.crs.is_geographic:
            lat = math.radians((src.bounds.top + src.bounds.bottom) / 2)
            res_x *= _M_PER_DEG * math.cos(lat)
            res_y *= _M_PER_DEG
    data = dem.filled(np.nan).astype("float64")
    if np.isnan(data).all():
        data = np.zeros_like(data)
    data = np.where(np.isnan(data), np.nanmean(data), data)
    return data, (res_x, res_y)


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize, good enough for a preview."""
    rows = (np.arange(height) * image.shape[0] / height).astype(int)
    cols = (np.arange(width) * image.shape[1] / width).astype(int)
    return image[rows][:, cols]


def _rgba(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 3:
        image = np.dstack([image, np.full(image.shape[:2], 255, image.dtype)])
    return image.astype("float64")


def _png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    iio.imwrite(buffer, image, extension=".png")
    return buffer.getvalue()


def relief_light(shade: np.ndarray, elevation: float) -> np.ndarray:
    """Brightness factor for the preview: shading relative to flat ground.

    A low sun leaves even flat ground dark (sin 10° ≈ 0.17), which buries the
    relief; the renderer's exposure/tonemap decide overall brightness anyway.
    So flat ground maps to a fixed level and only the relief varies: shadows
    darker, slopes facing the sun brighter.
    """
    flat = max(math.sin(math.radians(elevation)), 0.05)
    relative = np.clip(shade / flat, 0.0, 1.25) * FLAT_LEVEL
    return AMBIENT + (1 - AMBIENT) * relative


def _on_relief(
    overlay: np.ndarray,
    light: np.ndarray,
    extent: tuple[float, float, float, float],
    opacity: float,
) -> np.ndarray:
    """Overlay lit by ``light`` inside its UV ``extent`` (v measured from the
    bottom, as ``compute_normalized_extent`` does), grey relief elsewhere."""
    height, width = light.shape
    u0, v0, u1, v1 = extent
    x0, x1 = round(u0 * width), max(round(u0 * width) + 1, round(u1 * width))
    y0, y1 = (
        round((1 - v1) * height),
        max(round((1 - v1) * height) + 1, round((1 - v0) * height)),
    )
    out = np.stack([light * 255.0] * 3, axis=-1)
    box = _resize(overlay, y1 - y0, x1 - x0)
    alpha = box[..., 3:4] / 255.0 * opacity
    lit = box[..., :3] * light[y0:y1, x0:x1, None]
    out[y0:y1, x0:x1] = alpha * lit + (1 - alpha) * out[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype("uint8")


def build_previews(project_root: Path, layers: Sequence[Path]) -> list[OverlayPreview]:
    """One preview per overlay of the merged ``layers`` (sweeps ignored).

    Raises like a run would: a broken config, a missing raster, an overlay
    that doesn't overlap the DEM.
    """
    project_root = Path(project_root)
    cfg, written = _load(project_root, layers)
    if not Path(cfg.input.path).exists():
        raise FileNotFoundError(f"input raster not found: {cfg.input.path}")
    cache_dir = project_root / cfg.run.out_dir / cfg.run.name / PREP_CACHE_DIRNAME
    prepared, resolved = prepare_scene(cfg, cache_dir, use_cache=cfg.run.cache)

    dem, cell = _read_dem(prepared)
    sun = cfg.render.sun
    light = relief_light(
        hillshade(dem, cell, sun.azimuth, sun.elevation, cfg.render.terrain.zscale),
        sun.elevation,
    )
    by_name = {o.name: o for o in cfg.overlays}
    previews = []
    for overlay in resolved:
        image = _rgba(iio.imread(overlay.path))
        factor = max(1.0, max(image.shape[:2]) / MAX_SIDE)
        small = _resize(
            image, round(image.shape[0] / factor), round(image.shape[1] / factor)
        )
        opacity = 1.0 if overlay.opacity is None else overlay.opacity
        source = by_name[overlay.name]
        previews.append(
            OverlayPreview(
                name=overlay.name,
                source=written[overlay.name],
                prepare=[
                    str(op.get("op")) if isinstance(op, dict) else str(op)
                    for op in source.prepare
                ],
                opacity=overlay.opacity,
                png_path=overlay.path,
                overlay_png=_png(small.astype("uint8")),
                relief_png=_png(_on_relief(image, light, overlay.extent, opacity)),
            )
        )
    return previews
