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
def synthetic_dem_int16(tmp_path):
    """An int16 DEM (SRTM/NASADEM's native dtype) with a smooth elevation
    gradient, so cubic/bilinear resampling has genuine fractional values to
    either preserve (as float32) or truncate away (left as int16)."""
    width = height = 32
    rows, cols = np.mgrid[0:height, 0:width]
    data = (1000 + rows * 3 + cols * 2).astype("int16")

    path = tmp_path / "dem_int16.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        nodata=-32768,
        transform=from_origin(-121.8, 46.9, 0.001, 0.001),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def synthetic_overlay(tmp_path):
    """Partially overlaps ``synthetic_dem``: shifted +0.016 deg east/south so
    it extends past the DEM's east and south edges but overlaps its
    northwest, exercising clamping in both U and V.

    Deliberately carries no ``nodata`` -- a plain satellite export commonly
    doesn't declare one, and that's exactly the case the alignment/export
    path must handle correctly (see ``validity_mask`` in prep/ops.py):
    alpha for a partial overlay must come out right regardless of whether
    the source dataset happens to flag its own invalid pixels.
    """
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


# What sentinel_sr.rgbn.band_descriptions writes for the default 10-band SR
# stack on Planetary Computer (shared common names get the band id appended).
SR_DESCRIPTIONS = (
    "blue",
    "green",
    "red",
    "rededge_B05",
    "rededge_B06",
    "rededge_B07",
    "nir",
    "rededge_B8A",
    "swir16",
    "swir22",
)


@pytest.fixture
def make_sr_tif():
    """Factory for a small float32 10-band reflectance GeoTIFF like the SR
    step's output: band ``i`` (0-based) holds ``0.02 * (i + 1)`` plus a
    left-to-right ramp of up to 0.01, and pixel (0, 0) is NaN in every band."""

    def make(path, descriptions=SR_DESCRIPTIONS):
        width, height = 16, 8
        ramp = np.linspace(0.0, 0.01, width, dtype="float32")
        data = np.stack(
            [np.tile(0.02 * (i + 1) + ramp, (height, 1)) for i in range(10)]
        ).astype("float32")
        data[:, 0, 0] = np.nan
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=10,
            dtype="float32",
            crs="EPSG:32632",
            transform=from_origin(600000.0, 5200000.0, 2.5, 2.5),
        ) as dst:
            dst.write(data)
            for i, description in enumerate(descriptions or (), start=1):
                dst.set_band_description(i, description)
        return path

    return make
