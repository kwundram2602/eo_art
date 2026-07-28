"""Camera animation.

v1 builds ``CameraAnimation`` keyframes directly rather than using forge3d's
``TerrainOrbitRig``: the rigs require a loaded ``TerrainScatterSource`` and run
clearance refinement that can raise, while linear keyframes are pure math and
testable without a GPU.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from eo_art.forge3d_pipes.config.schema import AnimationKind, Orbit, PipelineConfig

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


def orbit_keyframes(orbit: Orbit, fps: int) -> list[Keyframe]:
    """Linearly interpolated keyframes, one per rendered frame.

    ``*_end`` values default to their ``*_start`` counterpart, so an unset
    field simply holds constant across the orbit.
    """
    theta_end = orbit.theta_start if orbit.theta_end is None else orbit.theta_end
    radius_end = orbit.radius_start if orbit.radius_end is None else orbit.radius_end
    fov_end = orbit.fov_start if orbit.fov_end is None else orbit.fov_end

    # Animation.__post_init__ guarantees count >= 1, so division is safe.
    count = int(round(orbit.duration * fps))
    return [
        Keyframe(
            time=orbit.duration * (step / count),
            phi=_lerp(orbit.phi_start, orbit.phi_end, step / count),
            theta=_lerp(orbit.theta_start, theta_end, step / count),
            radius=_lerp(orbit.radius_start, radius_end, step / count),
            fov=_lerp(orbit.fov_start, fov_end, step / count),
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
    for frame in orbit_keyframes(cfg.animation.orbit, cfg.animation.fps):
        animation.add_keyframe(
            time=frame.time,
            phi=frame.phi,
            theta=frame.theta,
            radius=frame.radius,
            fov=frame.fov,
        )
    return animation
