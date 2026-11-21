"""Camera animation.

v1 builds ``CameraAnimation`` keyframes directly rather than using forge3d's
``TerrainOrbitRig``: the rigs require a loaded ``TerrainScatterSource`` and run
clearance refinement that can raise, while linear keyframes are pure math and
testable without a GPU.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from eo_art.forge3d_pipes.config.schema import (
    AnimationKind,
    Camera,
    Orbit,
    PipelineConfig,
)

if TYPE_CHECKING:
    from forge3d.animation import CameraAnimation


@dataclass(frozen=True)
class Keyframe:
    time: float
    phi: float
    theta: float
    radius: float
    fov: float


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


def _or(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def orbit_keyframes(orbit: Orbit, fps: int, camera: Camera) -> list[Keyframe]:
    """Linearly interpolated keyframes, one per rendered frame.

    Unset ``*_start`` values come from ``camera`` (the still image's camera).
    Unset ``*_end`` values hold the start value, so the field stays constant
    across the orbit; ``phi_end`` instead defaults to one full turn.
    """
    phi_start = _or(orbit.phi_start, camera.phi)
    theta_start = _or(orbit.theta_start, camera.theta)
    radius_start = _or(orbit.radius_start, camera.radius)
    fov_start = _or(orbit.fov_start, camera.fov)
    phi_end = _or(orbit.phi_end, phi_start + 360.0)
    theta_end = _or(orbit.theta_end, theta_start)
    radius_end = _or(orbit.radius_end, radius_start)
    fov_end = _or(orbit.fov_end, fov_start)

    # Animation.__post_init__ guarantees count >= 1, so division is safe.
    count = int(round(orbit.duration * fps))
    return [
        Keyframe(
            time=orbit.duration * (step / count),
            phi=_lerp(phi_start, phi_end, step / count),
            theta=_lerp(theta_start, theta_end, step / count),
            radius=_lerp(radius_start, radius_end, step / count),
            fov=_lerp(fov_start, fov_end, step / count),
        )
        for step in range(count + 1)
    ]


def build_camera_animation(cfg: PipelineConfig) -> "CameraAnimation | None":
    """Return a forge3d animation, or ``None`` when animation is disabled.

    Import kept lazy: a module-level import would make forge3d load at
    package-import time (``forge3d_pipes/__init__.py`` -> ``pipeline`` ->
    ``runner`` -> ``animation``), before ``gpu.select_gpu`` gets a chance to
    set the Vulkan ICD environment variables that steer which adapter
    forge3d binds to.
    """
    if cfg.animation.kind is AnimationKind.none:
        return None

    from forge3d.animation import CameraAnimation

    animation = CameraAnimation()
    for frame in orbit_keyframes(
        cfg.animation.orbit, cfg.animation.fps, cfg.render.camera
    ):
        animation.add_keyframe(
            time=frame.time,
            phi=frame.phi,
            theta=frame.theta,
            radius=frame.radius,
            fov=frame.fov,
        )
    return animation
