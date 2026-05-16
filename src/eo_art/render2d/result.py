from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RenderStep:
    """Immutable rendered intermediate — float32 pixels, shape (H, W) or (H, W, C)."""

    pixels: np.ndarray
    crs: str
    resolution: float

    def __post_init__(self) -> None:
        if self.pixels.ndim not in (2, 3):
            raise ValueError(
                "pixels must be 2-D (H, W) or 3-D (H, W, C), "
                f"got shape {self.pixels.shape}"
            )

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    def _new(self, pixels: np.ndarray) -> RenderStep:
        return RenderStep(
            pixels=pixels.astype(np.float32), crs=self.crs, resolution=self.resolution
        )

    def normalize(self) -> RenderStep:
        """Rescale pixel values linearly so the full range spans [0, 1]."""
        pmin = float(self.pixels.min())
        pmax = float(self.pixels.max())
        if pmax == pmin:
            return self._new(np.zeros_like(self.pixels, dtype=np.float32))
        return self._new((self.pixels - pmin) / (pmax - pmin))

    def clip(self, low: float = 0.0, high: float = 1.0) -> RenderStep:
        """Clip pixel values to [low, high]."""
        return self._new(np.clip(self.pixels, low, high))

    def colorize(self, cmap: str = "viridis") -> RenderStep:
        """Apply a matplotlib colormap to a single-band image → (H, W, 4) RGBA."""
        from matplotlib import colormaps

        band = self.pixels if self.pixels.ndim == 2 else self.pixels[:, :, 0]
        rgba = colormaps[cmap](band).astype(np.float32)
        return self._new(rgba)

    def to_uint8(self) -> np.ndarray:
        """Return pixels as uint8 in [0, 255], clipping out-of-range values first."""
        return (np.clip(self.pixels, 0.0, 1.0) * 255).round().astype(np.uint8)

    def render(self, path: str | Path | None = None) -> np.ndarray:
        """Return the rendered uint8 array, and optionally save it to an image file."""
        pixels_u8 = self.to_uint8()
        if path is not None:
            import imageio

            imageio.imwrite(str(Path(path)), pixels_u8)
        return pixels_u8

    def style_transfer(
        self,
        style: str | Path | "RenderStep",
        **kwargs: object,
    ) -> "RenderStep":
        """Apply neural style transfer to this RenderStep.

        Args:
            style: Style source — a file path (str or Path) to any image,
                or another RenderStep.
            **kwargs: Additional arguments passed to neural_style_transfer,
                including method, max_size, steps, content_weight, style_weight,
                and device.

        Returns:
            A new RenderStep with stylised pixels and the same metadata.
        """
        from .neural import neural_style_transfer

        return neural_style_transfer(self, style, **kwargs)
