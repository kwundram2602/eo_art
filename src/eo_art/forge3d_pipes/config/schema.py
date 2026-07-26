"""Typed configuration schema for the forge3d render pipeline.

Field defaults reproduce ``demo1.py``. Range checks live in ``__post_init__``
and fire during ``OmegaConf.to_object``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING

# Enum member names are deliberately lowercase. OmegaConf matches enums by member
# name (not value), so lowercase names allow configs to use lowercase values like
# `operator: reinhard` instead of `operator: REINHARD`.


class ShadowTechnique(Enum):
    pcss = "pcss"
    pcf = "pcf"
    none = "none"


class TonemapOperator(Enum):
    aces = "aces"
    reinhard = "reinhard"
    linear = "linear"


class SunVisMode(Enum):
    soft = "soft"
    hard = "hard"


class ResamplingName(Enum):
    nearest = "nearest"
    bilinear = "bilinear"
    cubic = "cubic"


class SweepMode(Enum):
    product = "product"
    zip = "zip"


class AnimationKind(Enum):
    none = "none"
    orbit = "orbit"


class VideoFormat(Enum):
    mp4 = "mp4"
    gif = "gif"


def _in_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass
class Camera:
    phi: float = 300.0
    theta: float = 10.5
    radius: float = 26000.0
    fov: float = 60.0

    def __post_init__(self) -> None:
        _in_range("camera.fov", self.fov, 1.0, 179.0)
        _in_range("camera.theta", self.theta, 0.0, 90.0)
        _positive("camera.radius", self.radius)


@dataclass
class Sun:
    azimuth: float = 305.0
    elevation: float = 24.0
    intensity: float = 1.0
    ambient: float = 0.05

    def __post_init__(self) -> None:
        _in_range("sun.azimuth", self.azimuth, 0.0, 360.0)
        _in_range("sun.elevation", self.elevation, 0.0, 90.0)


@dataclass
class Terrain:
    zscale: float = 3.0


@dataclass
class HeightAO:
    enabled: bool = True
    directions: int = 6
    steps: int = 16
    max_distance: float = 200.0
    strength: float = 1.2
    resolution_scale: float = 0.5

    def __post_init__(self) -> None:
        _in_range("height_ao.resolution_scale", self.resolution_scale, 0.0, 1.0)


@dataclass
class SunVisibility:
    enabled: bool = True
    mode: SunVisMode = SunVisMode.soft
    samples: int = 4
    steps: int = 24
    max_distance: float = 400.0
    softness: float = 1.0
    bias: float = 0.01
    resolution_scale: float = 0.5

    def __post_init__(self) -> None:
        _in_range("sun_visibility.resolution_scale", self.resolution_scale, 0.0, 1.0)


@dataclass
class Materials:
    snow_enabled: bool = True
    snow_altitude_min: float = 3200.0
    snow_altitude_blend: float = 300.0
    snow_slope_max: float = 50.0
    rock_enabled: bool = True
    rock_slope_min: float = 42.0
    wetness_enabled: bool = False
    wetness_strength: float = 0.3


@dataclass
class Tonemap:
    operator: TonemapOperator = TonemapOperator.aces
    white_point: float = 4.0
    white_balance_enabled: bool = True
    temperature: float = 6000.0
    tint: float = 0.0


@dataclass
class LensEffects:
    enabled: bool = True
    distortion: float = 0.0
    chromatic_aberration: float = 0.0
    vignette_strength: float = 0.25
    vignette_radius: float = 0.7
    vignette_softness: float = 0.3


@dataclass
class Sky:
    enabled: bool = True
    turbidity: float = 2.5
    ground_albedo: float = 0.3
    sun_intensity: float = 1.0
    aerial_perspective: bool = True
    sky_exposure: float = 1.0


@dataclass
class Pbr:
    enabled: bool = True
    shadow_technique: ShadowTechnique = ShadowTechnique.pcss
    shadow_map_res: int = 4096
    exposure: float = 1.35
    msaa: int = 8
    ibl_intensity: float = 1.0
    normal_strength: float = 1.1
    height_ao: HeightAO = field(default_factory=HeightAO)
    sun_visibility: SunVisibility = field(default_factory=SunVisibility)
    materials: Materials = field(default_factory=Materials)
    tonemap: Tonemap = field(default_factory=Tonemap)
    lens_effects: LensEffects = field(default_factory=LensEffects)
    sky: Sky = field(default_factory=Sky)

    def __post_init__(self) -> None:
        if self.msaa not in (1, 2, 4, 8):
            raise ValueError(f"pbr.msaa must be one of 1, 2, 4, 8; got {self.msaa}")


@dataclass
class Render:
    width: int = 1200
    height: int = 720
    snapshot_name: str = "snapshot.png"
    camera: Camera = field(default_factory=Camera)
    sun: Sun = field(default_factory=Sun)
    terrain: Terrain = field(default_factory=Terrain)
    pbr: Pbr = field(default_factory=Pbr)

    def __post_init__(self) -> None:
        _positive("render.width", self.width)
        _positive("render.height", self.height)


@dataclass
class Orbit:
    duration: float = 8.0
    phi_start: float = 0.0
    phi_end: float = 360.0
    theta_start: float = 12.0
    theta_end: float | None = None
    radius_start: float = 26000.0
    radius_end: float | None = None
    fov_start: float = 60.0
    fov_end: float | None = None

    def __post_init__(self) -> None:
        _positive("orbit.duration", self.duration)
        _positive("orbit.radius_start", self.radius_start)


@dataclass
class Animation:
    kind: AnimationKind = AnimationKind.none
    fps: int = 30
    orbit: Orbit = field(default_factory=Orbit)

    def __post_init__(self) -> None:
        _positive("animation.fps", self.fps)
        frames = round(self.orbit.duration * self.fps)
        if frames < 1:
            raise ValueError(
                f"animation.orbit.duration ({self.orbit.duration}) x "
                f"animation.fps ({self.fps}) must yield at least 1 frame interval, "
                f"got {frames}"
            )


@dataclass
class Video:
    enabled: bool = False
    format: VideoFormat = VideoFormat.mp4
    fps: int = 30
    quality: int = 8

    def __post_init__(self) -> None:
        _positive("video.fps", self.fps)
        _in_range("video.quality", self.quality, 1, 10)


@dataclass
class Export:
    video: Video = field(default_factory=Video)


@dataclass
class Sweep:
    mode: SweepMode = SweepMode.product
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    name: str = "run"
    out_dir: str = "out"
    cache: bool = True
    fail_fast: bool = False


@dataclass
class Input:
    path: str = MISSING


@dataclass
class PipelineConfig:
    run: Run = field(default_factory=Run)
    input: Input = field(default_factory=Input)
    prepare: list[Any] = field(default_factory=list)
    render: Render = field(default_factory=Render)
    animation: Animation = field(default_factory=Animation)
    export: Export = field(default_factory=Export)
    sweep: Sweep | None = None
