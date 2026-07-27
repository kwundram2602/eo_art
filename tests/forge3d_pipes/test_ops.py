import numpy as np
import pytest
import rasterio
from omegaconf.errors import MissingMandatoryValue
from rasterio.transform import from_origin

from eo_art.forge3d_pipes.prep import ops, registry


def test_ops_are_registered():
    assert registry.get_op("reproject").schema is ops.ReprojectCfg
    assert registry.get_op("scale_to_gsd").schema is ops.ScaleToGsdCfg
    assert registry.get_op("saturate").schema is ops.SaturateCfg


def test_reproject_changes_crs(synthetic_dem, tmp_path):
    dst = tmp_path / "out.tif"
    result = ops.reproject(synthetic_dem, dst, ops.ReprojectCfg(crs="EPSG:32610"))
    assert result == dst
    with rasterio.open(result) as src:
        assert src.crs.to_string() == "EPSG:32610"
        assert src.width > 0 and src.height > 0
        assert src.count == 1


def test_reproject_preserves_elevation_range(synthetic_dem, tmp_path):
    dst = tmp_path / "out.tif"
    with rasterio.open(synthetic_dem) as src:
        original_max = float(src.read(1).max())
    ops.reproject(synthetic_dem, dst, ops.ReprojectCfg(crs="EPSG:32610"))
    with rasterio.open(dst) as out:
        assert float(out.read(1).max()) == pytest.approx(original_max, rel=0.05)


def test_reproject_requires_crs():
    with pytest.raises(MissingMandatoryValue):
        registry.validate_entry({"op": "reproject"})


def test_reproject_rejects_unknown_resampling():
    from omegaconf.errors import ValidationError

    with pytest.raises(ValidationError):
        registry.validate_entry(
            {"op": "reproject", "crs": "EPSG:32610", "resampling": "quintic"}
        )


def test_scale_to_gsd_changes_resolution(synthetic_dem, tmp_path):
    projected = tmp_path / "utm.tif"
    ops.reproject(synthetic_dem, projected, ops.ReprojectCfg(crs="EPSG:32610"))
    with rasterio.open(projected) as src:
        original_res = src.res[0]

    dst = tmp_path / "scaled.tif"
    target = original_res * 2.0
    ops.scale_to_gsd(projected, dst, ops.ScaleToGsdCfg(target_gsd=target))
    with rasterio.open(dst) as out:
        assert out.res[0] == pytest.approx(target, rel=0.01)


def test_chain_of_both_ops_runs(synthetic_dem, tmp_path):
    result = registry.run_prep_chain(
        synthetic_dem,
        [
            {"op": "reproject", "crs": "EPSG:32610"},
            {"op": "scale_to_gsd", "target_gsd": 200.0},
        ],
        tmp_path / "cache",
    )
    with rasterio.open(result) as src:
        assert src.crs.to_string() == "EPSG:32610"
        # scale_raster_to_gsd rounds to whole pixel counts, so on a small
        # raster the achieved GSD deviates from the target by a few percent:
        # here 12x18 pixels over a 2477x3621m extent yields ~206m, not 200m.
        assert src.res[0] == pytest.approx(200.0, rel=0.05)


def test_reproject_rejects_a_raster_without_a_crs(tmp_path):
    """A CRS-less DEM is plausible first-contact input; fail clearly, not opaquely."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

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
        ops.reproject(src, tmp_path / "out.tif", ops.ReprojectCfg(crs="EPSG:32610"))


def _write_rgb(path, r, g, b):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=r.shape[1],
        height=r.shape[0],
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)


def test_saturate_factor_zero_produces_grayscale(tmp_path):
    r = np.full((4, 4), 200.0, dtype="float32")
    g = np.full((4, 4), 100.0, dtype="float32")
    b = np.full((4, 4), 50.0, dtype="float32")
    src = tmp_path / "src.tif"
    _write_rgb(src, r, g, b)

    dst = tmp_path / "out.tif"
    ops.saturate(src, dst, ops.SaturateCfg(factor=0.0))

    with rasterio.open(dst) as out:
        out_r, out_g, out_b = out.read(1), out.read(2), out.read(3)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    assert out_r == pytest.approx(gray, abs=1e-3)
    assert out_g == pytest.approx(gray, abs=1e-3)
    assert out_b == pytest.approx(gray, abs=1e-3)


def test_saturate_factor_one_is_unchanged(tmp_path):
    r = np.full((4, 4), 200.0, dtype="float32")
    g = np.full((4, 4), 100.0, dtype="float32")
    b = np.full((4, 4), 50.0, dtype="float32")
    src = tmp_path / "src.tif"
    _write_rgb(src, r, g, b)

    dst = tmp_path / "out.tif"
    ops.saturate(src, dst, ops.SaturateCfg(factor=1.0))

    with rasterio.open(dst) as out:
        assert out.read(1) == pytest.approx(r, abs=1e-3)
        assert out.read(2) == pytest.approx(g, abs=1e-3)
        assert out.read(3) == pytest.approx(b, abs=1e-3)


def test_saturate_factor_above_one_increases_deviation_from_gray(tmp_path):
    r = np.full((4, 4), 200.0, dtype="float32")
    g = np.full((4, 4), 100.0, dtype="float32")
    b = np.full((4, 4), 50.0, dtype="float32")
    src = tmp_path / "src.tif"
    _write_rgb(src, r, g, b)

    dst = tmp_path / "out.tif"
    ops.saturate(src, dst, ops.SaturateCfg(factor=2.0))

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    with rasterio.open(dst) as out:
        out_r = out.read(1)
    original_deviation = np.abs(r - gray)
    new_deviation = np.abs(out_r - gray)
    assert new_deviation == pytest.approx(original_deviation * 2.0, abs=1e-3)


def test_saturate_requires_at_least_three_bands(tmp_path):
    src = tmp_path / "src.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(np.zeros((4, 4), dtype="float32"), 1)

    with pytest.raises(ValueError, match="at least 3 bands"):
        ops.saturate(src, tmp_path / "out.tif", ops.SaturateCfg(factor=0.5))
