import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from eo_art.forge3d_pipes.acquire.terrain import fetch_local_dem

REF_CRS = "EPSG:4326"
# A plausible CRS for a national 1 m LiDAR DGM (UTM 33N covers 12-18 deg E).
DEM_CRS = "EPSG:32633"


@pytest.fixture
def reference_tif(tmp_path):
    """A small WGS84 raster standing in for the optical scene."""
    width = height = 20
    data = np.zeros((height, width), dtype="float32")
    path = tmp_path / "reference.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=REF_CRS,
        transform=from_origin(12.0, 52.01, 0.0005, 0.0005),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def local_dem_tif(tmp_path, reference_tif):
    """A 1 m local DGM in projected UTM: deliberately larger than, and at a
    different resolution and CRS from, the reference -- so a regression that
    silently reprojects the clip onto the reference's grid fails the
    transform/CRS assertions below."""
    with rasterio.open(reference_tif) as ref:
        ref_bounds = ref.bounds
        ref_crs = ref.crs
    left, bottom, right, top = transform_bounds(ref_crs, DEM_CRS, *ref_bounds)

    margin = 500.0  # metres of padding beyond the reference footprint
    pixel = 1.0
    origin_x, origin_y = left - margin, top + margin
    width = int((right - left) + 2 * margin)
    height = int((top - bottom) + 2 * margin)

    rows, cols = np.mgrid[0:height, 0:width]
    data = (100.0 + rows * 0.01 + cols * 0.01).astype("float32")

    path = tmp_path / "local_dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=DEM_CRS,
        nodata=-9999.0,
        transform=from_origin(origin_x, origin_y, pixel, pixel),
    ) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def disjoint_local_dem_tif(tmp_path):
    """A DEM in the same CRS as ``local_dem_tif`` but nowhere near the
    reference footprint's projected location."""
    width = height = 10
    data = np.zeros((height, width), dtype="float32")
    path = tmp_path / "far_dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=DEM_CRS,
        transform=from_origin(100_000.0, 8_000_000.0, 1.0, 1.0),
    ) as dst:
        dst.write(data, 1)
    return path


def test_fetch_local_dem_clips_preserving_own_crs_and_pixel_size(
    tmp_path, reference_tif, local_dem_tif
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    out_path = fetch_local_dem(reference_tif, out_dir, path=local_dem_tif)

    with rasterio.open(local_dem_tif) as dem_src:
        dem_crs = dem_src.crs
        dem_pixel = (abs(dem_src.transform.a), abs(dem_src.transform.e))
        dem_width, dem_height = dem_src.width, dem_src.height

    with rasterio.open(out_path) as out_src:
        # the defining behaviour: the clip keeps the DEM's own grid, not the
        # reference's -- a regression that reprojects onto the reference
        # grid would fail both of these.
        assert out_src.crs == dem_crs
        got_pixel = (abs(out_src.transform.a), abs(out_src.transform.e))
        assert got_pixel == pytest.approx(dem_pixel)

        # clipped down from the (much larger) source DEM to roughly the
        # reference footprint, not left at the full source extent.
        assert 0 < out_src.width < dem_width
        assert 0 < out_src.height < dem_height

        # the output window covers the reference's own footprint (reprojected
        # into the DEM's CRS), within a pixel or two of rounding slack.
        with rasterio.open(reference_tif) as ref_src:
            expected = transform_bounds(ref_src.crs, out_src.crs, *ref_src.bounds)
        got = out_src.bounds
        tol = 2 * dem_pixel[0]
        assert got.left <= expected[0] + tol
        assert got.bottom <= expected[1] + tol
        assert got.right >= expected[2] - tol
        assert got.top >= expected[3] - tol


def test_fetch_local_dem_disjoint_raises_clearly(
    tmp_path, reference_tif, disjoint_local_dem_tif
):
    with pytest.raises(ValueError, match="does not cover"):
        fetch_local_dem(reference_tif, tmp_path / "out", path=disjoint_local_dem_tif)
