import pytest

from eo_art.forge3d_pipes.config.schema import (
    Animation,
    AnimationKind,
    Orbit,
    PipelineConfig,
)
from eo_art.forge3d_pipes.render import animation as anim


def test_keyframe_count_is_fps_times_duration_plus_one():
    frames = anim.orbit_keyframes(Orbit(duration=2.0), fps=10)
    assert len(frames) == 21


def test_first_and_last_keyframe_times_span_duration():
    frames = anim.orbit_keyframes(Orbit(duration=4.0), fps=5)
    assert frames[0].time == 0.0
    assert frames[-1].time == pytest.approx(4.0)


def test_phi_interpolates_linearly_from_start_to_end():
    frames = anim.orbit_keyframes(
        Orbit(duration=1.0, phi_start=0.0, phi_end=180.0), fps=4
    )
    assert [f.phi for f in frames] == pytest.approx([0.0, 45.0, 90.0, 135.0, 180.0])


def test_optional_end_values_hold_the_start_value():
    frames = anim.orbit_keyframes(
        Orbit(duration=1.0, theta_start=20.0, radius_start=1000.0, fov_start=50.0),
        fps=2,
    )
    assert {f.theta for f in frames} == {20.0}
    assert {f.radius for f in frames} == {1000.0}
    assert {f.fov for f in frames} == {50.0}


def test_end_values_interpolate_when_given():
    frames = anim.orbit_keyframes(
        Orbit(
            duration=1.0,
            theta_start=0.0,
            theta_end=40.0,
            radius_start=100.0,
            radius_end=300.0,
            fov_start=30.0,
            fov_end=70.0,
        ),
        fps=2,
    )
    assert [f.theta for f in frames] == pytest.approx([0.0, 20.0, 40.0])
    assert [f.radius for f in frames] == pytest.approx([100.0, 200.0, 300.0])
    assert [f.fov for f in frames] == pytest.approx([30.0, 50.0, 70.0])


def test_build_returns_none_when_animation_disabled():
    cfg = PipelineConfig()
    assert cfg.animation.kind is AnimationKind.none
    assert anim.build_camera_animation(cfg) is None


def test_build_returns_camera_animation_with_keyframes():
    cfg = PipelineConfig()
    cfg.animation = Animation(
        kind=AnimationKind.orbit, fps=4, orbit=Orbit(duration=1.0)
    )
    result = anim.build_camera_animation(cfg)
    assert result is not None
    assert len(result.get_keyframes()) == 5
