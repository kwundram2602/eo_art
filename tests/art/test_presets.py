from __future__ import annotations

import numpy as np
import xarray as xr

from eo_art.art.presets import hillshade_blend, ndvi_art, rgb_art
from eo_art.core.data import EOData
from eo_art.render2d.result import RenderStep


def _raster_eo(nbands: int = 4, h: int = 10, w: int = 10) -> EOData:
    rng = np.random.default_rng(1)
    data = rng.random((nbands, h, w)).astype(np.float32)
    da = xr.DataArray(
        data, dims=("band", "y", "x"), coords={"band": list(range(1, nbands + 1))}
    )
    return EOData.from_xarray(
        da.to_dataset(name="data"), crs="EPSG:4326", resolution=10.0
    )


def _dem_eo(h: int = 10, w: int = 10) -> EOData:
    elev = np.linspace(100.0, 500.0, h * w, dtype=np.float32).reshape(h, w)
    da = xr.DataArray(elev[np.newaxis], dims=("band", "y", "x"), coords={"band": [1]})
    return EOData.from_xarray(
        da.to_dataset(name="data"), crs="EPSG:4326", resolution=30.0
    )


def test_ndvi_art_returns_render_step():
    result = ndvi_art(_raster_eo())
    assert isinstance(result, RenderStep)


def test_ndvi_art_shape_is_rgba():
    result = ndvi_art(_raster_eo())
    assert result.pixels.ndim == 3
    assert result.pixels.shape[2] == 4


def test_ndvi_art_pixels_in_0_1():
    result = ndvi_art(_raster_eo())
    assert float(result.pixels.min()) >= 0.0
    assert float(result.pixels.max()) <= 1.0


def test_rgb_art_returns_render_step():
    result = rgb_art(_raster_eo())
    assert isinstance(result, RenderStep)


def test_rgb_art_shape_is_h_w_3():
    result = rgb_art(_raster_eo())
    assert result.pixels.shape == (10, 10, 3)


def test_rgb_art_pixels_in_0_1():
    result = rgb_art(_raster_eo())
    assert float(result.pixels.min()) >= 0.0
    assert float(result.pixels.max()) <= 1.0


def test_hillshade_blend_returns_render_step():
    result = hillshade_blend(_raster_eo(), _dem_eo())
    assert isinstance(result, RenderStep)


def test_hillshade_blend_shape_is_h_w_3():
    result = hillshade_blend(_raster_eo(), _dem_eo())
    assert result.pixels.shape == (10, 10, 3)


def test_hillshade_blend_alpha0_equals_rgb():
    raster = _raster_eo()
    flat = hillshade_blend(raster, _dem_eo(), alpha=0.0)
    plain = rgb_art(raster)
    np.testing.assert_allclose(flat.pixels, plain.pixels, atol=1e-5)


def test_public_api_imports():
    import eo_art

    assert hasattr(eo_art, "EOData")
    assert hasattr(eo_art, "RenderStep")
    assert hasattr(eo_art, "animate")
    assert hasattr(eo_art, "blend")
    assert hasattr(eo_art, "load_preset")
