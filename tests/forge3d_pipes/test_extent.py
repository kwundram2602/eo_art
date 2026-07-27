import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from eo_art.forge3d_pipes.prep import ops
from eo_art.forge3d_pipes.prep.extent import compute_normalized_extent


def test_full_overlap_yields_full_unit_extent(synthetic_dem):
    extent = compute_normalized_extent(synthetic_dem, synthetic_dem)
    assert extent == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_partial_overlap_clamps_to_terrain_bounds(synthetic_dem, synthetic_overlay):
    extent = compute_normalized_extent(synthetic_dem, synthetic_overlay)
    assert extent == pytest.approx((0.5, 0.0, 1.0, 0.5))


def test_disjoint_overlay_raises(synthetic_dem, disjoint_overlay):
    with pytest.raises(ValueError, match="does not overlap"):
        compute_normalized_extent(synthetic_dem, disjoint_overlay)


def test_overlay_without_crs_raises(synthetic_dem, tmp_path):
    src = tmp_path / "no_crs.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="float32",
        transform=from_origin(100.0, 200.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.ones((8, 8), dtype="float32"), 1)

    with pytest.raises(ValueError, match="has no CRS"):
        compute_normalized_extent(synthetic_dem, src)


def test_terrain_without_crs_raises(synthetic_overlay, tmp_path):
    src = tmp_path / "no_crs.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="float32",
        transform=from_origin(100.0, 200.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.ones((8, 8), dtype="float32"), 1)

    with pytest.raises(ValueError, match="has no CRS"):
        compute_normalized_extent(src, synthetic_overlay)


def test_overlay_in_different_crs_is_reprojected_for_comparison(
    synthetic_dem, synthetic_overlay, tmp_path
):
    terrain_utm = tmp_path / "terrain_utm.tif"
    ops.reproject(synthetic_dem, terrain_utm, ops.ReprojectCfg(crs="EPSG:32610"))

    extent = compute_normalized_extent(terrain_utm, synthetic_overlay)
    u0, v0, u1, v1 = extent
    assert 0.0 <= u0 < u1 <= 1.0
    assert 0.0 <= v0 < v1 <= 1.0
