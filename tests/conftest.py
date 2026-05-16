import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


@pytest.fixture
def dem_tif(tmp_path):
    data = np.linspace(100.0, 500.0, 100, dtype=np.float32).reshape(10, 10)
    transform = from_bounds(west=0, south=0, east=1, north=1, width=10, height=10)
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def rgb_tif(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 10000, (4, 10, 10), dtype=np.uint16)
    transform = from_bounds(west=0, south=0, east=1, north=1, width=10, height=10)
    path = tmp_path / "rgb.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=4,
        dtype=np.uint16,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path
