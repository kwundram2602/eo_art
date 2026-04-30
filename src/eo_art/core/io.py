from __future__ import annotations

from pathlib import Path

import xarray as xr

from .data import EOData
from .errors import CRSMissingError, EODataLoadError


def load_file(path: str | Path) -> EOData:
    import rioxarray  # noqa: F401 — registers .rio accessor

    path = Path(path)
    try:
        da = xr.open_dataarray(str(path), engine="rasterio")
    except Exception as exc:
        raise EODataLoadError(f"Cannot open {path}: {exc}") from exc

    if da.rio.crs is None:
        raise CRSMissingError(f"{path} has no CRS.")

    crs = da.rio.crs.to_string()
    res_x, _ = da.rio.resolution()
    resolution = abs(float(res_x))

    ds = da.to_dataset(name="data")
    return EOData.from_xarray(ds, crs=crs, resolution=resolution)
