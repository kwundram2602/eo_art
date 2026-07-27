import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from eo_art.forge3d_pipes.acquire.schema import AcquireConfig, DemSource


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
    assert cfg.dem.source is DemSource.copernicus
    assert cfg.dem.scale == 30.0
    assert cfg.dtm.apply_erosion is False
    assert cfg.dtm.radius == 8
    assert cfg.align.resampling.value == "bilinear"
    assert cfg.sentinel.reference_index is None
    assert cfg.sentinel.collection == "sentinel-2-l2a"


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
