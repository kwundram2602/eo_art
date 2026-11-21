import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from eo_art.forge3d_pipes.acquire.schema import (
    AcquireConfig,
    DemSource,
    DtmMode,
    Interval,
)


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


def test_timeseries_defaults_off():
    cfg = OmegaConf.to_object(OmegaConf.merge(_base(), _required()))
    assert cfg.sentinel.timeseries.enabled is False
    assert cfg.sentinel.timeseries.interval is Interval.month
    assert cfg.sentinel.timeseries.every == 1


def test_timeseries_every_must_be_positive():
    required = _required()
    required["sentinel"]["timeseries"] = {"enabled": True, "every": 0}
    with pytest.raises(ValueError, match="every must be >= 1"):
        OmegaConf.to_object(OmegaConf.merge(_base(), required))


def test_max_aoi_cloud_is_a_percentage():
    required = _required()
    required["sentinel"]["max_aoi_cloud"] = 120
    with pytest.raises(ValueError, match="max_aoi_cloud"):
        OmegaConf.to_object(OmegaConf.merge(_base(), required))


def test_zones_are_off_by_default():
    cfg = OmegaConf.to_object(OmegaConf.merge(_base(), _required()))
    assert cfg.sentinel.zones.enabled is False
    assert cfg.sentinel.zones.clip_to_aoi is True
    assert cfg.sentinel.zones.mosaic is False


def test_zones_accept_any_dtm_mode():
    # The acquire config is shared with eo-art-sr, so zones must not make a
    # config with the default dtm.mode invalid; acquire ignores the mode.
    raw = OmegaConf.merge(
        _base(), _required(), {"sentinel": {"zones": {"enabled": True}}}
    )
    assert OmegaConf.to_object(raw).sentinel.zones.enabled is True


def test_negative_zone_offset_is_rejected():
    raw = OmegaConf.merge(
        _base(),
        _required(),
        {"sentinel": {"timeseries": {"max_zone_offset_days": -1}}},
    )
    with pytest.raises(ValueError, match="max_zone_offset_days"):
        OmegaConf.to_object(raw)


def test_single_date_excludes_a_zone_offset():
    raw = OmegaConf.merge(
        _base(),
        _required(),
        {"sentinel": {"timeseries": {"single_date": True, "max_zone_offset_days": 3}}},
    )
    with pytest.raises(ValueError, match="single_date"):
        OmegaConf.to_object(raw)


def test_coreg_defaults_and_validation():
    cfg = OmegaConf.to_object(OmegaConf.merge(_base(), _required()))
    assert cfg.sentinel.timeseries.single_date is False
    assert (cfg.sentinel.coreg.enabled, cfg.sentinel.coreg.anchor) == (True, "auto")
    assert cfg.sentinel.coreg.grid_size == 7

    raw = OmegaConf.merge(
        _base(), _required(), {"sentinel": {"coreg": {"grid_size": 0}}}
    )
    with pytest.raises(ValueError, match="grid_size"):
        OmegaConf.to_object(raw)
