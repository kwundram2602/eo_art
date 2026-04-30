from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from ..render2d.composite import CompositeAccessor
    from ..render2d.result import RenderStep
    from ..render3d.scene import Scene3D


def _detect_kind(ds: xr.Dataset) -> Literal["raster", "timeseries", "dem"]:
    if not ds.data_vars:
        from .errors import EODataLoadError
        raise EODataLoadError("Dataset contains no data variables.")
    first_var = next(iter(ds.data_vars.values()))
    if "time" in first_var.dims:
        return "timeseries"
    # Arrays without a band dim (default 0 ≠ 1) are treated as raster, not dem.
    if first_var.sizes.get("band", 0) == 1 and np.issubdtype(first_var.dtype, np.floating):
        return "dem"
    return "raster"


@dataclass(frozen=True)
class EOData:
    ds: xr.Dataset
    crs: str
    resolution: float
    kind: Literal["raster", "timeseries", "dem"]

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, crs: str, resolution: float = 1.0) -> EOData:
        return cls(ds=ds, crs=crs, resolution=resolution, kind=_detect_kind(ds))

    @classmethod
    def from_file(cls, path: str | Path) -> EOData:
        from .io import load_file
        return load_file(Path(path))

    @property
    def composite(self) -> CompositeAccessor:
        from ..render2d.composite import CompositeAccessor
        return CompositeAccessor(self)

    def hillshade(self, azimuth: float = 315, altitude: float = 45) -> RenderStep:
        from ..render2d.hillshade import compute_hillshade
        return compute_hillshade(self, azimuth=azimuth, altitude=altitude)

    def scene3d(self, dem: EOData | None = None) -> Scene3D:
        # Pass self as dem when called on a DEM EOData; use .drape() to add raster texture.
        from ..render3d.scene import Scene3D
        return Scene3D(dem=dem if dem is not None else self)
