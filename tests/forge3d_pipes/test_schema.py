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
        (["animation.fps=2", "animation.orbit.duration=0.1"], "frame interval"),
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


@pytest.mark.parametrize(
    ("dotlist", "message"),
    [
        (["animation.orbit.fov_start=300"], "orbit.fov_start"),
        (["animation.orbit.theta_start=120"], "orbit.theta_start"),
        (["animation.orbit.theta_end=120"], "orbit.theta_end"),
        (["animation.orbit.fov_end=0"], "orbit.fov_end"),
        (["animation.orbit.radius_end=0"], "orbit.radius_end"),
    ],
)
def test_orbit_angles_are_validated_like_camera(dotlist, message):
    """Orbit must not bypass the ranges Camera enforces on the same knobs."""
    merged = OmegaConf.merge(
        _base(), {"input": {"path": "dem.tif"}}, OmegaConf.from_dotlist(dotlist)
    )
    with pytest.raises(ValueError, match=message):
        OmegaConf.to_object(merged)


def test_orbit_optional_end_values_may_stay_none():
    cfg = OmegaConf.to_object(OmegaConf.merge(_base(), {"input": {"path": "dem.tif"}}))
    assert cfg.animation.orbit.theta_end is None
    assert cfg.animation.orbit.fov_end is None
    assert cfg.animation.orbit.radius_end is None


def test_video_export_without_animation_is_rejected():
    """A still render produces no frames, so there would be nothing to encode."""
    merged = OmegaConf.merge(
        _base(),
        {"input": {"path": "dem.tif"}},
        OmegaConf.from_dotlist(["export.video.enabled=true", "animation.kind=none"]),
    )
    with pytest.raises(ValueError, match="requires an animation"):
        OmegaConf.to_object(merged)


def test_video_export_with_animation_is_accepted():
    cfg = OmegaConf.to_object(
        OmegaConf.merge(
            _base(),
            {"input": {"path": "dem.tif"}},
            OmegaConf.from_dotlist(
                ["export.video.enabled=true", "animation.kind=orbit"]
            ),
        )
    )
    assert cfg.export.video.enabled is True


def test_render_overlay_preserve_colors_defaults_false():
    cfg = OmegaConf.to_object(OmegaConf.merge(_base(), {"input": {"path": "dem.tif"}}))
    assert cfg.render.overlay_preserve_colors is False
    assert cfg.overlays == []


def test_overlay_name_and_path_are_mandatory():
    merged = OmegaConf.merge(
        _base(), {"input": {"path": "dem.tif"}, "overlays": [{}]}
    )
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_object(merged)


def test_overlay_with_name_and_path_is_accepted():
    merged = OmegaConf.merge(
        _base(),
        {
            "input": {"path": "dem.tif"},
            "overlays": [{"name": "ndvi", "path": "ndvi.tif"}],
        },
    )
    cfg = OmegaConf.to_object(merged)
    assert cfg.overlays[0].name == "ndvi"
    assert cfg.overlays[0].path == "ndvi.tif"
    assert cfg.overlays[0].opacity is None
    assert cfg.overlays[0].z_order is None
    assert cfg.overlays[0].extent is None
    assert cfg.overlays[0].prepare == []


@pytest.mark.parametrize("opacity", [-0.1, 1.1])
def test_overlay_opacity_out_of_range_rejected(opacity):
    merged = OmegaConf.merge(
        _base(),
        {
            "input": {"path": "dem.tif"},
            "overlays": [{"name": "ndvi", "path": "ndvi.tif", "opacity": opacity}],
        },
    )
    with pytest.raises(ValueError, match="overlay.opacity"):
        OmegaConf.to_object(merged)


@pytest.mark.parametrize("extent", [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0]])
def test_overlay_extent_wrong_length_rejected(extent):
    merged = OmegaConf.merge(
        _base(),
        {
            "input": {"path": "dem.tif"},
            "overlays": [{"name": "ndvi", "path": "ndvi.tif", "extent": extent}],
        },
    )
    with pytest.raises(ValueError, match="exactly 4 values"):
        OmegaConf.to_object(merged)


@pytest.mark.parametrize(
    "extent", [[-0.1, 0.0, 1.0, 1.0], [0.0, 0.0, 1.1, 1.0], [0.0, 0.0, 1.0, 1.1]]
)
def test_overlay_extent_out_of_unit_range_rejected(extent):
    merged = OmegaConf.merge(
        _base(),
        {
            "input": {"path": "dem.tif"},
            "overlays": [{"name": "ndvi", "path": "ndvi.tif", "extent": extent}],
        },
    )
    with pytest.raises(ValueError, match="overlay.extent"):
        OmegaConf.to_object(merged)


@pytest.mark.parametrize(
    "extent", [[0.5, 0.0, 0.5, 1.0], [0.0, 0.5, 1.0, 0.5]]
)
def test_overlay_extent_non_positive_area_rejected(extent):
    merged = OmegaConf.merge(
        _base(),
        {
            "input": {"path": "dem.tif"},
            "overlays": [{"name": "ndvi", "path": "ndvi.tif", "extent": extent}],
        },
    )
    with pytest.raises(ValueError, match="positive area"):
        OmegaConf.to_object(merged)


def test_multiple_overlays_each_validated_independently():
    merged = OmegaConf.merge(
        _base(),
        {
            "input": {"path": "dem.tif"},
            "overlays": [
                {"name": "good", "path": "good.tif"},
                {"name": "bad", "path": "bad.tif", "opacity": 5.0},
            ],
        },
    )
    with pytest.raises(ValueError, match="overlay.opacity"):
        OmegaConf.to_object(merged)
