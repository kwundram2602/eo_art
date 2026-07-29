import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from eo_art.forge3d_pipes.acquire.schema import AcquireConfig, DemSource, DtmMode


def _base():
    return OmegaConf.structured(AcquireConfig)


def _required():
    return {
        "aoi_path": "aoi.gpkg",
        "ee_project": "ee-test",
        "sentinel_sr_dir": "../sentinel_sr",
        "sentinel": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "model_path": "/models/LDSRS2-SEN2SR",
        },
    }


def test_defaults():
    cfg = OmegaConf.to_object(OmegaConf.merge(_base(), _required()))
    assert cfg.dem.source is DemSource.fabdem
    assert cfg.dem.scale == 30.0
    assert cfg.dem.path is None
    assert cfg.dtm.apply_erosion is False
    assert cfg.dtm.radius == 8
    assert cfg.dtm.mode is DtmMode.guided_upsample
    assert cfg.align.resampling.value == "bilinear"
    assert cfg.sentinel.reference_index is None
    assert cfg.sentinel.collection == "sentinel-2-l2a"
    assert cfg.out_dir == "out"


def test_missing_required_field_fails():
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_object(_base())


def test_negative_reference_index_rejected():
    merged = OmegaConf.merge(
        _base(), _required(), {"sentinel": {"reference_index": -1}}
    )
    with pytest.raises(ValueError, match="reference_index"):
        OmegaConf.to_object(merged)


def test_dem_source_is_settable_by_name():
    merged = OmegaConf.merge(_base(), _required(), {"dem": {"source": "fabdem"}})
    cfg = OmegaConf.to_object(merged)
    assert cfg.dem.source is DemSource.fabdem


def test_local_source_without_path_fails():
    merged = OmegaConf.merge(_base(), _required(), {"dem": {"source": "local"}})
    with pytest.raises(ValueError, match="dem.path"):
        OmegaConf.to_object(merged)


def test_non_local_source_with_path_fails():
    merged = OmegaConf.merge(
        _base(), _required(), {"dem": {"source": "fabdem", "path": "dem.tif"}}
    )
    with pytest.raises(ValueError, match="dem.path"):
        OmegaConf.to_object(merged)


def test_ee_project_required_for_earth_engine_source():
    required = _required()
    del required["ee_project"]
    merged = OmegaConf.merge(_base(), required)
    with pytest.raises(ValueError, match="ee_project"):
        OmegaConf.to_object(merged)


def test_ee_project_optional_for_local_source():
    required = _required()
    del required["ee_project"]
    required["dem"] = {"source": "local", "path": "dem.tif"}
    merged = OmegaConf.merge(_base(), required)
    cfg = OmegaConf.to_object(merged)
    assert cfg.ee_project is None
    assert cfg.dem.source is DemSource.local
    assert cfg.dem.path == "dem.tif"
