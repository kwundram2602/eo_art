"""Compute a normalized UV extent for draping an overlay onto a DEM."""

from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


def compute_normalized_extent(
    terrain_path: Path, overlay_path: Path
) -> tuple[float, float, float, float]:
    """Normalize ``overlay_path``'s footprint against ``terrain_path``'s bounds.

    Reprojects the overlay's bounds (not its pixels) into the terrain's CRS
    for comparison. An overlay that only partially overlaps the terrain is
    clamped to the terrain's bounds rather than rejected, since partial
    coverage (e.g. an NDVI tile covering part of a larger DEM) is an ordinary
    case, not an error.

    This fits a bounding box only; it does not correct for rotation/shear
    between differently-oriented CRSes. For pixel-accurate draping, reproject
    the overlay into the terrain's CRS via its own ``prepare`` chain first.
    """
    with rasterio.open(terrain_path) as terrain:
        if terrain.crs is None:
            raise ValueError(f"{terrain_path} has no CRS; cannot align overlay")
        t_left, t_bottom, t_right, t_top = terrain.bounds
        terrain_crs = terrain.crs

    with rasterio.open(overlay_path) as overlay:
        if overlay.crs is None:
            raise ValueError(f"{overlay_path} has no CRS; cannot align overlay")
        o_left, o_bottom, o_right, o_top = transform_bounds(
            overlay.crs, terrain_crs, *overlay.bounds
        )

    left, right = max(o_left, t_left), min(o_right, t_right)
    bottom, top = max(o_bottom, t_bottom), min(o_top, t_top)

    if left >= right or bottom >= top:
        raise ValueError(
            f"overlay {overlay_path} does not overlap terrain {terrain_path}: "
            f"overlay bounds (reprojected)=({o_left}, {o_bottom}, {o_right}, {o_top}), "
            f"terrain bounds=({t_left}, {t_bottom}, {t_right}, {t_top})"
        )

    u0 = (left - t_left) / (t_right - t_left)
    u1 = (right - t_left) / (t_right - t_left)
    v0 = (bottom - t_bottom) / (t_top - t_bottom)
    v1 = (top - t_bottom) / (t_top - t_bottom)
    return (u0, v0, u1, v1)
