from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eo_art.core.data import EOData


def _make_eo(
    rng: np.random.Generator, h: int = 8, w: int = 8, bands: int = 4
) -> EOData:
    data = rng.integers(0, 65535, size=(bands, h, w), dtype=np.uint16).astype(
        np.float32
    )
    da = xr.DataArray(
        data,
        dims=["band", "y", "x"],
        coords={"band": np.arange(1, bands + 1)},
    )
    ds = xr.Dataset({"data": da})
    return EOData(ds=ds, crs="EPSG:4326", resolution=10.0, kind="raster")


@pytest.fixture
def eo() -> EOData:
    rng = np.random.default_rng(0)
    return _make_eo(rng)


def test_rgb_shape(eo: EOData) -> None:
    result = eo.composite.rgb()
    assert result.pixels.shape == (8, 8, 3)


def test_rgb_dtype(eo: EOData) -> None:
    result = eo.composite.rgb()
    assert result.pixels.dtype == np.float32


def test_rgb_normalized(eo: EOData) -> None:
    result = eo.composite.rgb()
    assert float(result.pixels.min()) >= 0.0
    assert float(result.pixels.max()) <= 1.0
    for c in range(3):
        ch = result.pixels[:, :, c]
        assert float(ch.min()) == pytest.approx(0.0)
        assert float(ch.max()) == pytest.approx(1.0)


def test_ndvi_shape(eo: EOData) -> None:
    result = eo.composite.ndvi()
    assert result.pixels.shape == (8, 8)


def test_ndvi_dtype(eo: EOData) -> None:
    result = eo.composite.ndvi()
    assert result.pixels.dtype == np.float32


def test_ndvi_range(eo: EOData) -> None:
    result = eo.composite.ndvi()
    assert float(result.pixels.min()) >= -1.0
    assert float(result.pixels.max()) <= 1.0


def test_ndvi_zero_denominator() -> None:
    data = np.zeros((4, 4, 4), dtype=np.float32)
    da = xr.DataArray(
        data,
        dims=["band", "y", "x"],
        coords={"band": np.arange(1, 5)},
    )
    ds = xr.Dataset({"data": da})
    eo = EOData(ds=ds, crs="EPSG:4326", resolution=10.0, kind="raster")
    result = eo.composite.ndvi()
    assert np.all(result.pixels == 0.0)
    assert not np.any(np.isnan(result.pixels))


def test_rgb_preserves_crs_resolution(eo: EOData) -> None:
    result = eo.composite.rgb()
    assert result.crs == "EPSG:4326"
    assert result.resolution == 10.0


def test_ndvi_preserves_crs_resolution(eo: EOData) -> None:
    result = eo.composite.ndvi()
    assert result.crs == "EPSG:4326"
    assert result.resolution == 10.0
