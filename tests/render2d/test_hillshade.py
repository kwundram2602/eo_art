from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eo_art.core.data import EOData
from eo_art.render2d.hillshade import compute_hillshade
from eo_art.render2d.result import RenderStep


def _dem_eo() -> EOData:
    elevation = np.linspace(100, 500, 100).reshape(10, 10).astype(np.float32)
    data = elevation[np.newaxis, :, :]  # shape (1, 10, 10)
    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": [1]},
    )
    ds = xr.Dataset({"data": da})
    return EOData.from_xarray(ds, crs="EPSG:4326", resolution=30.0)


def test_returns_render_step():
    eo = _dem_eo()
    result = compute_hillshade(eo)
    assert isinstance(result, RenderStep)


def test_output_shape_matches_spatial_dims():
    eo = _dem_eo()
    result = compute_hillshade(eo)
    assert result.pixels.shape == (10, 10)


def test_output_dtype_float32():
    eo = _dem_eo()
    result = compute_hillshade(eo)
    assert result.pixels.dtype == np.float32


def test_output_pixels_in_0_1():
    eo = _dem_eo()
    result = compute_hillshade(eo)
    assert float(result.pixels.min()) >= 0.0
    assert float(result.pixels.max()) <= 1.0


def test_crs_and_resolution_preserved():
    eo = _dem_eo()
    result = compute_hillshade(eo)
    assert result.crs == "EPSG:4326"
    assert result.resolution == 30.0


def test_raises_on_non_dem():
    elevation = np.linspace(100, 500, 300).reshape(3, 10, 10).astype(np.float32)
    da = xr.DataArray(
        elevation,
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3]},
    )
    ds = xr.Dataset({"data": da})
    eo = EOData.from_xarray(ds, crs="EPSG:4326", resolution=30.0)
    assert eo.kind == "raster"
    with pytest.raises(ValueError, match="dem"):
        compute_hillshade(eo)
