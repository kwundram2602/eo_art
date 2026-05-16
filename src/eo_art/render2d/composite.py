from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vecraspy import ndvi as _ndvi

if TYPE_CHECKING:
    from ..core.data import EOData

from .result import RenderStep


class CompositeAccessor:
    def __init__(self, eo: EOData) -> None:
        self._eo = eo

    def rgb(self, red: int = 1, green: int = 2, blue: int = 3) -> RenderStep:
        """Stack three bands into (H, W, 3) float32, normalised per-channel to [0,1]."""
        channels = []
        for band_idx in (red, green, blue):
            ch = self._eo.ds["data"].sel(band=band_idx).values.astype(np.float32)
            lo, hi = float(ch.min()), float(ch.max())
            if hi == lo:
                channels.append(np.zeros_like(ch))
            else:
                channels.append((ch - lo) / (hi - lo))
        pixels = np.stack(channels, axis=-1).astype(np.float32)
        return RenderStep(
            pixels=pixels, crs=self._eo.crs, resolution=self._eo.resolution
        )

    def ndvi(self, nir: int = 4, red: int = 1) -> RenderStep:
        """Wrap vecraspy.ndvi → (H, W) float32 in [-1, 1]."""
        nir_arr = self._eo.ds["data"].sel(band=nir).values.astype(np.float32)
        red_arr = self._eo.ds["data"].sel(band=red).values.astype(np.float32)
        pixels = _ndvi(red_arr, nir_arr)
        return RenderStep(
            pixels=pixels, crs=self._eo.crs, resolution=self._eo.resolution
        )
