from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from eo_art.core.data import EOData
from eo_art.render2d.result import RenderStep
from eo_art.render3d.scene import Scene3D


def _make_dem() -> EOData:
    data = np.linspace(100.0, 500.0, 100, dtype=np.float32).reshape(1, 10, 10)
    da = xr.DataArray(data, dims=("band", "y", "x"))
    ds = da.to_dataset(name="data")
    return EOData.from_xarray(ds, crs="EPSG:4326", resolution=30.0)


def _make_texture() -> RenderStep:
    pixels = np.random.default_rng(0).random((10, 10, 3)).astype(np.float32)
    return RenderStep(pixels=pixels, crs="EPSG:4326", resolution=30.0)


@pytest.fixture
def dem() -> EOData:
    return _make_dem()


@pytest.fixture
def texture() -> RenderStep:
    return _make_texture()


def test_scene3d_instantiation(dem: EOData) -> None:
    scene = Scene3D(dem=dem)
    assert scene.dem is dem
    assert scene._texture is None


def test_drape_returns_new_scene(dem: EOData, texture: RenderStep) -> None:
    scene = Scene3D(dem=dem)
    draped = scene.drape(texture)

    assert draped is not scene
    assert draped.dem is dem
    assert draped._texture is texture


def test_drape_leaves_original_unchanged(dem: EOData, texture: RenderStep) -> None:
    scene = Scene3D(dem=dem)
    scene.drape(texture)
    assert scene._texture is None


def test_scene3d_dem_and_texture_accessible(dem: EOData, texture: RenderStep) -> None:
    scene = Scene3D(dem=dem, _texture=texture)
    assert scene.dem is dem
    assert scene._texture is texture


def test_to_mesh_raises_without_pyvista(dem: EOData) -> None:
    scene = Scene3D(dem=dem)
    with patch.dict(sys.modules, {"pyvista": None}):
        with pytest.raises(ImportError, match="eo_art\\[3d\\]"):
            scene.to_mesh()
