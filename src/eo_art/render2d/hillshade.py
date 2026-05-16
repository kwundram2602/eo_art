from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vecraspy import hillshade as _hillshade

if TYPE_CHECKING:
    from ..core.data import EOData

from .result import RenderStep


def compute_hillshade(
    eo: EOData,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> RenderStep:
    """Wrap vecraspy.hillshade as a RenderStep for DEM EOData."""
    if eo.kind != "dem":
        raise ValueError(f"compute_hillshade requires kind='dem', got kind='{eo.kind}'")

    elevation: np.ndarray = eo.ds["data"].isel(band=0).values.astype(np.float32)
    pixels = _hillshade(
        elevation,
        azimuth=azimuth,
        altitude=altitude,
        z_factor=z_factor,
        dx=eo.resolution,
        dy=eo.resolution,
    )
    return RenderStep(pixels=pixels, crs=eo.crs, resolution=eo.resolution)
