import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def synthetic_dem(tmp_path):
    """A small WGS84 DEM with a smooth hill, written as a GeoTIFF."""
    width = height = 32
    rows, cols = np.mgrid[0:height, 0:width]
    data = (
        1000.0 + 500.0 * np.exp(-((rows - 16) ** 2 + (cols - 16) ** 2) / 60.0)
    ).astype("float32")

    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-121.8, 46.9, 0.001, 0.001),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def synthetic_overlay(tmp_path):
    """Partially overlaps ``synthetic_dem``: shifted +0.016 deg east/south so
    it extends past the DEM's east and south edges but overlaps its
    northwest, exercising clamping in both U and V."""
    width = height = 32
    data = np.full((height, width), 0.5, dtype="float32")
    path = tmp_path / "overlay.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-121.784, 46.884, 0.001, 0.001),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def disjoint_overlay(tmp_path):
    """Nowhere near ``synthetic_dem``'s bounds, for the no-overlap case."""
    width = height = 8
    data = np.full((height, width), 0.5, dtype="float32")
    path = tmp_path / "disjoint.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(10.0, 10.0, 0.001, 0.001),
    ) as dst:
        dst.write(data, 1)
    return path
