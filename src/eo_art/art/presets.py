from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.data import EOData

from ..render2d.result import RenderStep


def ndvi_art(
    eo: EOData, *, nir: int = 4, red: int = 1, cmap: str = "RdYlGn"
) -> RenderStep:
    """NDVI computed, normalised to [0, 1], and colourised with a diverging palette."""
    return eo.composite.ndvi(nir=nir, red=red).normalize().colorize(cmap)


def rgb_art(eo: EOData, *, red: int = 1, green: int = 2, blue: int = 3) -> RenderStep:
    """True-colour RGB composite clipped to [0, 1]."""
    return eo.composite.rgb(red=red, green=green, blue=blue).clip()


def hillshade_blend(
    eo_raster: EOData,
    eo_dem: EOData,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    alpha: float = 0.6,
) -> RenderStep:
    """RGB composite multiplied by hillshade for a terrain-textured look.

    alpha controls the hillshade strength: 0 = flat colour, 1 = full relief.
    """
    rgb = rgb_art(eo_raster)
    shade = eo_dem.hillshade(azimuth=azimuth, altitude=altitude).pixels
    # Lerp: pure RGB at alpha=0, full multiply at alpha=1.
    factor = (1.0 - alpha) + alpha * shade[:, :, np.newaxis]
    return rgb._new((rgb.pixels * factor).astype(np.float32))
