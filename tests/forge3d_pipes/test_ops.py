import pytest
import rasterio
from omegaconf.errors import MissingMandatoryValue

from eo_art.forge3d_pipes.prep import ops, registry


def test_ops_are_registered():
    assert registry.get_op("reproject").schema is ops.ReprojectCfg
    assert registry.get_op("scale_to_gsd").schema is ops.ScaleToGsdCfg


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
        assert src.res[0] == pytest.approx(200.0, rel=0.01)
