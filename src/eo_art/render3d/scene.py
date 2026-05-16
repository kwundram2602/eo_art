from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.data import EOData
    from ..render2d.result import RenderStep


@dataclass
class Scene3D:
    """A 3-D scene built from a DEM with optional drape texture."""

    dem: EOData
    _texture: RenderStep | None = field(default=None, repr=False)

    def drape(self, texture: RenderStep) -> Scene3D:
        """Return a new Scene3D with a render texture draped over the DEM."""
        return Scene3D(dem=self.dem, _texture=texture)

    def to_mesh(self):
        """Build and return a PyVista StructuredGrid mesh from the DEM.

        Raises ImportError if pyvista is not installed.
        """
        try:
            import pyvista as pv
        except (ImportError, TypeError):
            raise ImportError("Install eo_art[3d] for 3-D features")

        elev = self.dem.ds["data"].isel(band=0).values.astype(np.float32)
        H, W = elev.shape
        x = np.arange(W) * self.dem.resolution
        y = np.arange(H) * self.dem.resolution
        xx, yy = np.meshgrid(x, y)
        grid = pv.StructuredGrid(xx, yy, elev)

        if self._texture is not None:
            pixels = self._texture.pixels
            flat = (
                pixels.reshape(-1, pixels.shape[-1])
                if pixels.ndim == 3
                else pixels.ravel()
            )
            grid.point_data["texture"] = flat

        return grid

    def show(self, **kwargs) -> None:
        """Display the scene interactively using PyVista."""
        self.to_mesh().plot(**kwargs)

    def export(self, path: str | Path) -> Path:
        """Export the mesh to a file (STL, OBJ, VTK). Returns resolved path."""
        resolved = Path(path).resolve()
        self.to_mesh().save(str(resolved))
        return resolved
