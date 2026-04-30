import numpy as np
import pytest
import xarray as xr

from eo_art.core.data import EOData


def _make_ds(nbands: int = 1, dtype=np.float32, with_time: bool = False) -> xr.Dataset:
    if with_time:
        data = np.random.rand(3, nbands, 10, 10).astype(dtype)
        da = xr.DataArray(data, dims=("time", "band", "y", "x"))
    else:
        data = np.random.rand(nbands, 10, 10).astype(dtype)
        da = xr.DataArray(data, dims=("band", "y", "x"))
    return da.to_dataset(name="data")


def test_from_xarray_raster_kind():
    ds = _make_ds(nbands=4, dtype=np.uint16)
    eo = EOData.from_xarray(ds, crs="EPSG:4326", resolution=10.0)
    assert eo.kind == "raster"
    assert eo.crs == "EPSG:4326"
    assert eo.resolution == 10.0


def test_from_xarray_dem_kind():
    ds = _make_ds(nbands=1, dtype=np.float32)
    eo = EOData.from_xarray(ds, crs="EPSG:4326", resolution=30.0)
    assert eo.kind == "dem"


def test_from_xarray_timeseries_kind():
    ds = _make_ds(nbands=1, dtype=np.float32, with_time=True)
    eo = EOData.from_xarray(ds, crs="EPSG:4326", resolution=10.0)
    assert eo.kind == "timeseries"


def test_eodata_is_immutable():
    ds = _make_ds()
    eo = EOData.from_xarray(ds, crs="EPSG:4326", resolution=10.0)
    with pytest.raises((AttributeError, TypeError)):
        eo.crs = "EPSG:32632"  # type: ignore[misc]
