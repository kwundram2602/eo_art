import pytest
from omegaconf import MISSING, OmegaConf
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, ValidationError

from eo_art.forge3d_pipes.config.schema import (
    Camera,
    PipelineConfig,
    ShadowTechnique,
    TonemapOperator,
)


def _base():
    return OmegaConf.structured(PipelineConfig)


def test_defaults_match_demo1():
    cfg = OmegaConf.to_object(OmegaConf.merge(_base(), {"input": {"path": "dem.tif"}}))
    assert cfg.render.width == 1200
    assert cfg.render.height == 720
    assert cfg.render.snapshot_name == "snapshot.png"
    assert cfg.render.camera.phi == 300.0
    assert cfg.render.camera.theta == 10.5
    assert cfg.render.camera.radius == 26000.0
    assert cfg.render.terrain.zscale == 3.0
    assert cfg.render.sun.azimuth == 305.0
    assert cfg.render.sun.elevation == 24.0
    assert cfg.render.sun.ambient == 0.05
    assert cfg.render.pbr.exposure == 1.35
    assert cfg.render.pbr.msaa == 8
    assert cfg.render.pbr.shadow_map_res == 4096
    assert cfg.render.pbr.shadow_technique is ShadowTechnique.pcss
    assert cfg.render.pbr.tonemap.operator is TonemapOperator.aces
    assert cfg.render.pbr.materials.snow_altitude_min == 3200.0
    assert cfg.render.pbr.sky.turbidity == 2.5


def test_camera_fov_default_is_valid_not_demo1_300():
    assert Camera().fov == 60.0


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigKeyError):
        OmegaConf.merge(_base(), {"render": {"sun": {"azimut": 10.0}}})


def test_invalid_enum_is_rejected():
    with pytest.raises(ValidationError):
        OmegaConf.merge(
            _base(), {"render": {"pbr": {"tonemap": {"operator": "aces2"}}}}
        )


def test_input_path_is_mandatory():
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_object(_base())


@pytest.mark.parametrize(
    ("dotlist", "message"),
    [
        (["render.camera.fov=300"], "camera.fov"),
        (["render.camera.theta=91"], "camera.theta"),
        (["render.camera.radius=0"], "camera.radius"),
        (["render.sun.elevation=120"], "sun.elevation"),
        (["render.sun.azimuth=400"], "sun.azimuth"),
        (["render.pbr.msaa=3"], "pbr.msaa"),
        (["render.pbr.height_ao.resolution_scale=1.5"], "resolution_scale"),
        (["render.width=0"], "render.width"),
        (["animation.fps=0"], "animation.fps"),
    ],
)
def test_range_validation_rejects_bad_values(dotlist, message):
    merged = OmegaConf.merge(
        _base(),
        {"input": {"path": "dem.tif"}},
        OmegaConf.from_dotlist(dotlist),
    )
    with pytest.raises(ValueError, match=message):
        OmegaConf.to_object(merged)


def test_input_path_missing_sentinel_is_omegaconf_missing():
    assert (
        PipelineConfig.__dataclass_fields__["input"].default_factory().path == MISSING
    )
