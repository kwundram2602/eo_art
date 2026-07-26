# forge3d_pipes Config-Driven Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `src/eo_art/forge3d_pipes/demo1.py` with a config-driven, modular forge3d render pipeline: validated OmegaConf schemas, a prep-op registry, camera animation, parameter sweeps, and export.

**Architecture:** Path-in / path-out stages. A merged+validated `PipelineConfig` drives `prep chain → render → export`, run once per sweep variant into its own output directory. Pure functions build the forge3d IPC payloads so everything except the viewer call is testable without a GPU.

**Tech Stack:** Python 3.13, OmegaConf 2.3, forge3d 1.34, rasterio, vecraspy, imageio, pytest.

**Spec:** `docs/superpowers/specs/2026-07-25-forge3d-pipes-design.md`

## Global Constraints

- Python `>=3.13`. Package is `src/`-layout under `src/eo_art/`.
- ruff: `line-length = 88`, `target-version = "py313"`, `quote-style = "double"`. Run `uv run ruff format` and `uv run ruff check --fix` before every commit.
- Type checker is `ty` (`uv run ty check`), configured with `src = ["src"]`.
- No Hydra. OmegaConf only.
- No data-download logic. No STAC. Input is a local raster path.
- **Commit messages must NOT contain any `Co-Authored-By` or `Contributor` trailer naming Claude.** (Project rule in `CLAUDE.md`.)
- All commands run through `uv` (e.g. `uv run pytest`).
- Test files live under `tests/forge3d_pipes/`, mirroring the source tree.

## Verified API Facts

These were confirmed empirically against the installed packages. Do not re-derive them.

- `OmegaConf.merge(structured, {"cam": {"fovv": 1}})` raises `ConfigKeyError` (struct mode is implicit for structured configs).
- **OmegaConf matches enums by member name, not value.** Enum members have deliberately lowercase names (e.g. `TonemapOperator.reinhard`) so config files can use lowercase values like `operator: reinhard` instead of `operator: REINHARD`. Invalid enum values raise `omegaconf.errors.ValidationError`.
- `OmegaConf.to_object(cfg)` returns **real dataclass instances and runs `__post_init__`** — this is where range checks fire.
- A field defaulted to `omegaconf.MISSING` raises `MissingMandatoryValue` from `to_object()`.
- `OmegaConf.from_dotlist(["a.b=33"])` coerces to the schema's type on merge.
- Builtin generics (`list[Any]`, `dict[str, Any]`) and `float | None` all work in OmegaConf 2.3.1.
- `forge3d.open_viewer_async(width, height, title, obj_path, gltf_path, terrain_path, fov_deg, timeout, ipc_host, ipc_port) -> ViewerHandle`; `ViewerHandle` is a context manager.
- `ViewerHandle.snapshot(path, width=None, height=None) -> None`.
- `ViewerHandle.render_animation(animation, output_dir, fps=30, width=None, height=None, progress_callback=None) -> None`, writing frames named **`frame_%04d.png`**.
- `forge3d.animation.CameraAnimation()` takes no constructor args; `add_keyframe(time, phi, theta, radius, fov, target=None)`.
- `vecraspy.scale_raster_to_gsd(input_path, output_path, target_gsd, *, resampling="bilinear") -> Path`.
- **`vecraspy.reproject_raster` does not exist** — `demo1.py:4` is a broken import. eo_art implements `reproject` on `rasterio.warp`.
- `vecraspy.scale_raster_to_gsd` rounds the output width/height to whole pixel counts (`round(src.width * src_gsd / target_gsd)`), so the achieved GSD is only approximate on small rasters — e.g. a 2477x3621m extent with `target_gsd=200.0` rounds to 12x18 pixels, yielding ~206m/~201m, not exactly 200m.

## Deviations From The Spec

1. **Camera animation does not use `TerrainOrbitRig`.** The spec proposed mapping config onto forge3d's rigs. `_BaseTerrainRig.bake()` requires a `TerrainScatterSource` (a loaded heightfield) and performs clearance refinement that raises `ValueError` when constraints cannot be satisfied. v1 instead builds `CameraAnimation` keyframes directly — pure math, no terrain load, no clearance failure mode, unit-testable without a GPU. Rigs remain a later addition behind the same `animation.kind` switch.
2. **`camera.fov` default is `60.0`, not demo1's `300.0`.** demo1's value is invalid; its own comment says the flag was never set. This is stated in the spec.
3. **Render parameter defaults reproduce demo1, but the output filename does not.** `Render.snapshot_name` defaults to a generic `"snapshot.png"`, not a scene-specific value. Scene-specific filenames are set per-variant in the config or at runtime; the library default is generic.

## File Structure

| File | Responsibility |
|---|---|
| `src/eo_art/__init__.py` | Package marker (currently missing) |
| `src/eo_art/forge3d_pipes/__init__.py` | Public API: `run`, `load_config`, `PipelineConfig` |
| `src/eo_art/forge3d_pipes/config/schema.py` | All dataclasses + enums + range validation |
| `src/eo_art/forge3d_pipes/config/loader.py` | Merge files + dotlist, `to_pipeline` validation |
| `src/eo_art/forge3d_pipes/prep/registry.py` | `register_op`, entry validation, chain execution, caching |
| `src/eo_art/forge3d_pipes/prep/ops.py` | `reproject`, `scale_to_gsd` |
| `src/eo_art/forge3d_pipes/render/payloads.py` | Pure config → IPC dicts |
| `src/eo_art/forge3d_pipes/render/animation.py` | Orbit keyframes → `CameraAnimation` |
| `src/eo_art/forge3d_pipes/render/runner.py` | Viewer orchestration |
| `src/eo_art/forge3d_pipes/sweep.py` | Sweep expansion → variants |
| `src/eo_art/forge3d_pipes/export.py` | `resolved.yaml` dump, video encoding |
| `src/eo_art/forge3d_pipes/pipeline.py` | Per-variant orchestration, error tiers |
| `src/eo_art/forge3d_pipes/cli.py` | argparse entry point |
| `configs/base.yaml`, `configs/looks/alpine_dusk.yaml` | Shipped configs |

---

### Task 1: Dependencies, package skeleton, and config schema

**Files:**
- Modify: `pyproject.toml`
- Create: `src/eo_art/__init__.py`, `src/eo_art/forge3d_pipes/__init__.py`, `src/eo_art/forge3d_pipes/config/__init__.py`, `src/eo_art/forge3d_pipes/config/schema.py`
- Create: `tests/__init__.py`, `tests/forge3d_pipes/__init__.py`, `tests/forge3d_pipes/test_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `eo_art.forge3d_pipes.config.schema` exporting enums `ShadowTechnique`, `TonemapOperator`, `SunVisMode`, `ResamplingName`, `SweepMode`, `AnimationKind`, `VideoFormat`; dataclasses `Camera`, `Sun`, `Terrain`, `HeightAO`, `SunVisibility`, `Materials`, `Tonemap`, `LensEffects`, `Sky`, `Pbr`, `Render`, `Orbit`, `Animation`, `Video`, `Export`, `Sweep`, `Run`, `Input`, `PipelineConfig`.

- [ ] **Step 1: Add dependencies and pytest config**

```bash
uv add omegaconf
uv add --optional video imageio-ffmpeg
```

Then append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: requires a GPU and spawns the forge3d viewer subprocess"]
addopts = "-m 'not gpu'"
```

- [ ] **Step 2: Create package markers**

```bash
mkdir -p src/eo_art/forge3d_pipes/config tests/forge3d_pipes
touch src/eo_art/__init__.py src/eo_art/forge3d_pipes/config/__init__.py
touch tests/__init__.py tests/forge3d_pipes/__init__.py
```

Leave `src/eo_art/forge3d_pipes/__init__.py` empty for now (Task 11 fills it).

- [ ] **Step 3: Write the failing tests**

Create `tests/forge3d_pipes/test_schema.py`:

```python
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
    cfg = OmegaConf.to_object(
        OmegaConf.merge(_base(), {"input": {"path": "dem.tif"}})
    )
    assert cfg.render.width == 1200
    assert cfg.render.height == 720
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
        OmegaConf.merge(_base(), {"render": {"pbr": {"tonemap": {"operator": "aces2"}}}})


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
    assert PipelineConfig.__dataclass_fields__["input"].default_factory().path == MISSING
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_art.forge3d_pipes.config'`

- [ ] **Step 5: Write the schema**

Create `src/eo_art/forge3d_pipes/config/schema.py`. Every render parameter default is demo1's literal value (camera, sun, terrain, PBR), except `Camera.fov` (see Deviations). The output filename defaults to a generic `snapshot.png`.

```python
"""Typed configuration schema for the forge3d render pipeline.

Field defaults reproduce ``demo1.py``. Range checks live in ``__post_init__``
and fire during ``OmegaConf.to_object``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING


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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_schema.py -v`
Expected: PASS (15 tests — 6 plus 9 parametrized range cases)

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add pyproject.toml uv.lock src/eo_art tests
git commit -m "feat(forge3d_pipes): add validated config schema"
```

---

### Task 2: Prep op registry

**Files:**
- Create: `src/eo_art/forge3d_pipes/prep/__init__.py`, `src/eo_art/forge3d_pipes/prep/registry.py`
- Create: `tests/forge3d_pipes/test_registry.py`

**Interfaces:**
- Consumes: nothing from Task 1 (schema-independent).
- Produces:
  - `RegisteredOp` — frozen dataclass with `name: str`, `func: Callable[[Path, Path, Any], Path]`, `schema: type`.
  - `register_op(name: str, schema: type) -> Callable[[F], F]` decorator.
  - `get_op(name: str) -> RegisteredOp`, raising `ValueError` listing known ops.
  - `validate_entry(entry: dict) -> tuple[RegisteredOp, Any]`.
  - `validate_chain(entries: list) -> list[tuple[RegisteredOp, Any]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_registry.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import MISSING
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue

from eo_art.forge3d_pipes.prep import registry


@dataclass
class DummyCfg:
    factor: float = 2.0
    label: str = MISSING


@pytest.fixture
def dummy_op(monkeypatch):
    monkeypatch.setattr(registry, "_OPS", {})

    @registry.register_op("dummy", DummyCfg)
    def _dummy(src: Path, dst: Path, cfg: DummyCfg) -> Path:
        dst.write_text(f"{src.name}:{cfg.factor}:{cfg.label}")
        return dst

    return _dummy


def test_get_op_returns_registered(dummy_op):
    op = registry.get_op("dummy")
    assert op.name == "dummy"
    assert op.schema is DummyCfg
    assert op.func is dummy_op


def test_unknown_op_lists_known_ops(dummy_op):
    with pytest.raises(ValueError, match="unknown prep op 'nope'.*known ops: dummy"):
        registry.get_op("nope")


def test_duplicate_registration_rejected(dummy_op):
    with pytest.raises(ValueError, match="already registered"):
        registry.register_op("dummy", DummyCfg)(lambda src, dst, cfg: dst)


def test_validate_entry_returns_typed_config(dummy_op):
    op, cfg = registry.validate_entry({"op": "dummy", "factor": 3.0, "label": "x"})
    assert op.name == "dummy"
    assert isinstance(cfg, DummyCfg)
    assert cfg.factor == 3.0


def test_validate_entry_rejects_unknown_param(dummy_op):
    with pytest.raises(ConfigKeyError):
        registry.validate_entry({"op": "dummy", "factorr": 3.0, "label": "x"})


def test_validate_entry_rejects_missing_mandatory_param(dummy_op):
    with pytest.raises(MissingMandatoryValue):
        registry.validate_entry({"op": "dummy", "factor": 3.0})


def test_validate_entry_requires_op_key(dummy_op):
    with pytest.raises(ValueError, match="missing 'op' key"):
        registry.validate_entry({"factor": 3.0})


def test_validate_chain_validates_every_entry(dummy_op):
    entries = [
        {"op": "dummy", "label": "a"},
        {"op": "dummy", "label": "b", "factor": 9.0},
    ]
    result = registry.validate_chain(entries)
    assert [cfg.label for _, cfg in result] == ["a", "b"]
    assert result[1][1].factor == 9.0


def test_validate_chain_reports_entry_index(dummy_op):
    with pytest.raises(ValueError, match="prepare\\[1\\]"):
        registry.validate_chain([{"op": "dummy", "label": "a"}, {"op": "nope"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_art.forge3d_pipes.prep'`

- [ ] **Step 3: Write the registry**

```bash
mkdir -p src/eo_art/forge3d_pipes/prep && touch src/eo_art/forge3d_pipes/prep/__init__.py
```

Create `src/eo_art/forge3d_pipes/prep/registry.py`:

```python
"""Registry mapping prep-op names to implementations and their schemas.

Entries are validated at config-load time, so an unknown op or a bad
parameter fails before any raster work starts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import OmegaConf

OpFunc = Callable[[Path, Path, Any], Path]
F = TypeVar("F", bound=OpFunc)


@dataclass(frozen=True)
class RegisteredOp:
    name: str
    func: OpFunc
    schema: type


_OPS: dict[str, RegisteredOp] = {}


def register_op(name: str, schema: type) -> Callable[[F], F]:
    """Register a prep op under ``name`` with its parameter dataclass."""

    def decorator(func: F) -> F:
        if name in _OPS:
            raise ValueError(f"prep op {name!r} is already registered")
        _OPS[name] = RegisteredOp(name=name, func=func, schema=schema)
        return func

    return decorator


def get_op(name: str) -> RegisteredOp:
    try:
        return _OPS[name]
    except KeyError:
        known = ", ".join(sorted(_OPS)) or "<none>"
        raise ValueError(f"unknown prep op {name!r}; known ops: {known}") from None


def validate_entry(entry: Any) -> tuple[RegisteredOp, Any]:
    """Validate one ``prepare`` entry against its op's schema."""
    params = dict(entry)
    try:
        name = params.pop("op")
    except KeyError:
        raise ValueError(f"prep entry missing 'op' key: {entry!r}") from None
    op = get_op(name)
    merged = OmegaConf.merge(OmegaConf.structured(op.schema), params)
    return op, OmegaConf.to_object(merged)


def validate_chain(entries: list[Any]) -> list[tuple[RegisteredOp, Any]]:
    """Validate every entry, prefixing failures with their index."""
    validated = []
    for index, entry in enumerate(entries):
        try:
            validated.append(validate_entry(entry))
        except Exception as exc:
            raise type(exc)(f"prepare[{index}]: {exc}") from exc
    return validated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_registry.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/prep tests/forge3d_pipes/test_registry.py
git commit -m "feat(forge3d_pipes): add prep op registry with load-time validation"
```

---

### Task 3: Config loader

**Files:**
- Create: `src/eo_art/forge3d_pipes/config/loader.py`
- Create: `tests/forge3d_pipes/test_loader.py`

**Interfaces:**
- Consumes: `schema.PipelineConfig` (Task 1), `registry.validate_chain` (Task 2).
- Produces:
  - `load_raw(paths: Sequence[str | Path], overrides: Sequence[str] = (), out: str | Path | None = None) -> DictConfig` — merged but not yet object-validated. Sweep operates on this.
  - `to_pipeline(cfg: DictConfig) -> PipelineConfig` — `to_object` plus prep-chain validation.
  - `load_config(paths, overrides=(), out=None) -> PipelineConfig` — convenience composition of both.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_loader.py`:

```python
import pytest
from omegaconf.errors import ConfigKeyError

from eo_art.forge3d_pipes.config import loader
from eo_art.forge3d_pipes.config.schema import TonemapOperator


@pytest.fixture
def cfg_files(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "input:\n  path: dem.tif\n"
        "render:\n  width: 800\n  camera:\n    phi: 100.0\n"
    )
    look = tmp_path / "look.yaml"
    look.write_text(
        "render:\n  width: 1600\n  pbr:\n    tonemap:\n      operator: reinhard\n"
    )
    return base, look


def test_single_file_merges_over_defaults(cfg_files):
    base, _ = cfg_files
    cfg = loader.load_config([base])
    assert cfg.render.width == 800
    assert cfg.render.camera.phi == 100.0
    assert cfg.render.height == 720  # untouched default


def test_later_file_wins(cfg_files):
    base, look = cfg_files
    cfg = loader.load_config([base, look])
    assert cfg.render.width == 1600
    assert cfg.render.camera.phi == 100.0  # still from base
    assert cfg.render.pbr.tonemap.operator is TonemapOperator.reinhard


def test_dotlist_beats_files(cfg_files):
    base, look = cfg_files
    cfg = loader.load_config([base, look], overrides=["render.width=2000"])
    assert cfg.render.width == 2000


def test_out_sets_run_out_dir_and_beats_files(cfg_files, tmp_path):
    base, _ = cfg_files
    cfg = loader.load_config([base], out=tmp_path / "renders")
    assert cfg.run.out_dir == str(tmp_path / "renders")


def test_unknown_key_in_file_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("input:\n  path: dem.tif\nrender:\n  widht: 800\n")
    with pytest.raises(ConfigKeyError):
        loader.load_config([bad])


def test_interpolation_resolves(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "input:\n  path: dem.tif\nrun:\n  out_dir: /data\n  name: ${input.path}\n"
    )
    cfg = loader.load_config([cfg_file])
    assert cfg.run.name == "dem.tif"


def test_unknown_prep_op_fails_at_load(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("input:\n  path: dem.tif\nprepare:\n  - op: bogus\n")
    with pytest.raises(ValueError, match="unknown prep op 'bogus'"):
        loader.load_config([cfg_file])


def test_load_raw_keeps_dictconfig_for_sweeping(cfg_files):
    base, _ = cfg_files
    raw = loader.load_raw([base])
    assert raw.render.width == 800
    # DictConfig, not a dataclass instance
    assert not hasattr(raw, "__dataclass_fields__")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_loader.py -v`
Expected: FAIL — `ImportError: cannot import name 'loader'`

- [ ] **Step 3: Write the loader**

Create `src/eo_art/forge3d_pipes/config/loader.py`:

```python
"""Config loading: merge defaults, files, and CLI overrides, then validate."""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.prep.registry import validate_chain


def load_raw(
    paths: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
) -> DictConfig:
    """Merge schema defaults, config files, and dotlist overrides.

    Merge order, later wins: defaults -> files (in order) -> dotlist.
    ``out`` is applied in the dotlist layer, so it beats the files.
    """
    cfg = OmegaConf.structured(PipelineConfig)
    for path in paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(path)))
    dotlist = list(overrides)
    if out is not None:
        dotlist.append(f"run.out_dir={out}")
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def to_pipeline(cfg: DictConfig) -> PipelineConfig:
    """Convert to dataclasses (running range checks) and validate prep ops."""
    obj: PipelineConfig = OmegaConf.to_object(cfg)
    validate_chain(obj.prepare)
    return obj


def load_config(
    paths: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
) -> PipelineConfig:
    return to_pipeline(load_raw(paths, overrides, out))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_loader.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/config/loader.py tests/forge3d_pipes/test_loader.py
git commit -m "feat(forge3d_pipes): add config loader with merge order and validation"
```

---

### Task 4: Prep chain execution and caching

**Files:**
- Modify: `src/eo_art/forge3d_pipes/prep/registry.py` (append)
- Create: `tests/forge3d_pipes/test_prep_chain.py`

**Interfaces:**
- Consumes: `RegisteredOp`, `validate_entry` (Task 2).
- Produces:
  - `chain_cache_key(src: Path, entries: list[Any]) -> str` — 16-char hex.
  - `run_prep_chain(src: Path, entries: list[Any], cache_dir: Path, use_cache: bool = True) -> Path` — returns `src` unchanged when `entries` is empty.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_prep_chain.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import pytest

from eo_art.forge3d_pipes.prep import registry


@dataclass
class CountCfg:
    tag: str = "x"


@pytest.fixture
def counting_op(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(registry, "_OPS", {})

    @registry.register_op("count", CountCfg)
    def _count(src: Path, dst: Path, cfg: CountCfg) -> Path:
        calls.append(cfg.tag)
        dst.write_text(src.read_text() + f"|{cfg.tag}")
        return dst

    return calls


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src.tif"
    src.write_text("dem")
    return src


def test_empty_chain_returns_source_unchanged(source, tmp_path):
    assert registry.run_prep_chain(source, [], tmp_path / "cache") == source


def test_chain_applies_ops_in_order(counting_op, source, tmp_path):
    out = registry.run_prep_chain(
        source,
        [{"op": "count", "tag": "a"}, {"op": "count", "tag": "b"}],
        tmp_path / "cache",
    )
    assert out.read_text() == "dem|a|b"
    assert counting_op == ["a", "b"]


def test_second_run_hits_cache(counting_op, source, tmp_path):
    entries = [{"op": "count", "tag": "a"}]
    cache = tmp_path / "cache"
    first = registry.run_prep_chain(source, entries, cache)
    second = registry.run_prep_chain(source, entries, cache)
    assert first == second
    assert counting_op == ["a"]  # op ran only once


def test_no_cache_forces_recompute(counting_op, source, tmp_path):
    entries = [{"op": "count", "tag": "a"}]
    cache = tmp_path / "cache"
    registry.run_prep_chain(source, entries, cache)
    registry.run_prep_chain(source, entries, cache, use_cache=False)
    assert counting_op == ["a", "a"]


def test_different_params_use_different_cache_entries(counting_op, source, tmp_path):
    cache = tmp_path / "cache"
    a = registry.run_prep_chain(source, [{"op": "count", "tag": "a"}], cache)
    b = registry.run_prep_chain(source, [{"op": "count", "tag": "b"}], cache)
    assert a != b
    assert counting_op == ["a", "b"]


def test_cache_key_changes_when_source_changes(source, tmp_path):
    entries = [{"op": "count", "tag": "a"}]
    before = registry.chain_cache_key(source, entries)
    source.write_text("different content entirely")
    assert registry.chain_cache_key(source, entries) != before


def test_failed_op_leaves_no_cache_file(monkeypatch, source, tmp_path):
    monkeypatch.setattr(registry, "_OPS", {})

    @registry.register_op("boom", CountCfg)
    def _boom(src: Path, dst: Path, cfg: CountCfg) -> Path:
        raise RuntimeError("op exploded")

    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="op exploded"):
        registry.run_prep_chain(source, [{"op": "boom"}], cache)
    assert list(cache.glob("*.tif")) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_prep_chain.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'run_prep_chain'`

- [ ] **Step 3: Append chain execution to the registry**

Add these imports to the top of `src/eo_art/forge3d_pipes/prep/registry.py`:

```python
import hashlib
import json
import shutil
import tempfile
```

Then append:

```python
def chain_cache_key(src: Path, entries: list[Any]) -> str:
    """Hash the source identity plus the canonical prep chain."""
    stat = src.stat()
    payload = {
        "src": str(src.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "chain": entries,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_prep_chain(
    src: Path,
    entries: list[Any],
    cache_dir: Path,
    use_cache: bool = True,
) -> Path:
    """Run the prep chain, caching the result by chain hash.

    Returns ``src`` unchanged when the chain is empty. Intermediates are
    written to a temporary directory and only the final result is promoted
    into the cache, so a failing op leaves no partial cache entry.
    """
    src = Path(src)
    if not entries:
        return src

    key = chain_cache_key(src, entries)
    final = Path(cache_dir) / f"{key}.tif"
    if use_cache and final.exists():
        return final

    final.parent.mkdir(parents=True, exist_ok=True)
    current = src
    with tempfile.TemporaryDirectory(dir=final.parent) as tmp:
        for index, entry in enumerate(entries):
            op, cfg = validate_entry(entry)
            dst = Path(tmp) / f"{index:02d}_{op.name}.tif"
            current = Path(op.func(current, dst, cfg))
        shutil.move(str(current), final)
    return final
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_prep_chain.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/prep/registry.py tests/forge3d_pipes/test_prep_chain.py
git commit -m "feat(forge3d_pipes): add cached prep chain execution"
```

---

### Task 5: Prep ops — reproject and scale_to_gsd

**Files:**
- Create: `src/eo_art/forge3d_pipes/prep/ops.py`
- Create: `tests/conftest.py`
- Create: `tests/forge3d_pipes/test_ops.py`

**Interfaces:**
- Consumes: `register_op` (Task 2), `ResamplingName` (Task 1).
- Produces:
  - `ReprojectCfg` — `crs: str = MISSING`, `resampling: ResamplingName = BILINEAR`.
  - `ScaleToGsdCfg` — `target_gsd: float = MISSING`, `resampling: ResamplingName = BILINEAR`.
  - Ops registered under names `"reproject"` and `"scale_to_gsd"`.
  - Importing `eo_art.forge3d_pipes.prep.ops` is what populates the registry; `prep/__init__.py` imports it so registration always happens.

- [ ] **Step 1: Write the synthetic-DEM fixture**

Create `tests/conftest.py`:

```python
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def synthetic_dem(tmp_path):
    """A small WGS84 DEM with a smooth hill, written as a GeoTIFF."""
    width = height = 32
    rows, cols = np.mgrid[0:height, 0:width]
    data = (
        1000.0
        + 500.0 * np.exp(-((rows - 16) ** 2 + (cols - 16) ** 2) / 60.0)
    ).astype("float32")

    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-121.8, 46.9, 0.001, 0.001),
    ) as dst:
        dst.write(data, 1)
    return path
```

- [ ] **Step 2: Write the failing tests**

Create `tests/forge3d_pipes/test_ops.py`:

```python
import pytest
import rasterio
from omegaconf.errors import MissingMandatoryValue

from eo_art.forge3d_pipes.prep import ops, registry


def test_ops_are_registered():
    assert registry.get_op("reproject").schema is ops.ReprojectCfg
    assert registry.get_op("scale_to_gsd").schema is ops.ScaleToGsdCfg


def test_reproject_changes_crs(synthetic_dem, tmp_path):
    dst = tmp_path / "out.tif"
    result = ops.reproject(
        synthetic_dem, dst, ops.ReprojectCfg(crs="EPSG:32610")
    )
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
        # scale_raster_to_gsd rounds to whole pixel counts, so on a small
        # raster the achieved GSD deviates from the target by a few percent:
        # here 12x18 pixels over a 2477x3621m extent yields ~206m, not 200m.
        assert src.res[0] == pytest.approx(200.0, rel=0.05)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_ops.py -v`
Expected: FAIL — `ImportError: cannot import name 'ops'`

- [ ] **Step 4: Write the ops**

Create `src/eo_art/forge3d_pipes/prep/ops.py`. The `reproject` body is demo1's working rasterio code, generalised:

```python
"""Prep ops. Each takes (src, dst, cfg) and returns the written path."""

from dataclasses import dataclass
from pathlib import Path

import rasterio
from omegaconf import MISSING
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject as _rio_reproject

from eo_art.forge3d_pipes.config.schema import ResamplingName
from eo_art.forge3d_pipes.prep.registry import register_op

_RESAMPLING = {
    ResamplingName.nearest: Resampling.nearest,
    ResamplingName.bilinear: Resampling.bilinear,
    ResamplingName.cubic: Resampling.cubic,
}


@dataclass
class ReprojectCfg:
    crs: str = MISSING
    resampling: ResamplingName = ResamplingName.bilinear


@dataclass
class ScaleToGsdCfg:
    target_gsd: float = MISSING
    resampling: ResamplingName = ResamplingName.bilinear


@register_op("reproject", ReprojectCfg)
def reproject(src: Path, dst: Path, cfg: ReprojectCfg) -> Path:
    """Reproject a raster to ``cfg.crs``, preserving all bands."""
    with rasterio.open(src) as source:
        transform, width, height = calculate_default_transform(
            source.crs, cfg.crs, source.width, source.height, *source.bounds
        )
        meta = source.meta.copy()
        meta.update(
            {"crs": cfg.crs, "transform": transform, "width": width, "height": height}
        )
        with rasterio.open(dst, "w", **meta) as destination:
            for band in range(1, source.count + 1):
                _rio_reproject(
                    source=rasterio.band(source, band),
                    destination=rasterio.band(destination, band),
                    src_transform=source.transform,
                    dst_transform=transform,
                    src_crs=source.crs,
                    dst_crs=cfg.crs,
                    resampling=_RESAMPLING[cfg.resampling],
                )
    return Path(dst)


@register_op("scale_to_gsd", ScaleToGsdCfg)
def scale_to_gsd(src: Path, dst: Path, cfg: ScaleToGsdCfg) -> Path:
    """Resample to a target ground sample distance via vecraspy."""
    from vecraspy import scale_raster_to_gsd

    scale_raster_to_gsd(
        src, dst, cfg.target_gsd, resampling=cfg.resampling.value
    )
    return Path(dst)
```

Then make registration automatic by writing `src/eo_art/forge3d_pipes/prep/__init__.py`:

```python
"""Prep stage: registry plus the built-in ops (imported for registration)."""

from eo_art.forge3d_pipes.prep import ops  # noqa: F401  (populates the registry)
from eo_art.forge3d_pipes.prep.registry import (
    chain_cache_key,
    get_op,
    register_op,
    run_prep_chain,
    validate_chain,
    validate_entry,
)

__all__ = [
    "chain_cache_key",
    "get_op",
    "register_op",
    "run_prep_chain",
    "validate_chain",
    "validate_entry",
]
```

Note: `config/loader.py` imports from `eo_art.forge3d_pipes.prep.registry` directly, so this `__init__` does not create a circular import.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_ops.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Verify the whole suite still passes**

Run: `uv run pytest -v`
Expected: PASS. If `test_registry.py` or `test_prep_chain.py` now fail because real ops leaked into their `monkeypatch.setattr(registry, "_OPS", {})` fixtures, they should not — those fixtures replace the dict wholesale.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/prep tests/conftest.py tests/forge3d_pipes/test_ops.py
git commit -m "feat(forge3d_pipes): add reproject and scale_to_gsd prep ops"
```

---

### Task 6: Render payloads

**Files:**
- Create: `src/eo_art/forge3d_pipes/render/__init__.py`, `src/eo_art/forge3d_pipes/render/payloads.py`
- Create: `tests/forge3d_pipes/data/golden_payloads.json`
- Create: `tests/forge3d_pipes/test_payloads.py`

**Interfaces:**
- Consumes: `PipelineConfig` (Task 1).
- Produces:
  - `build_set_terrain(cfg: PipelineConfig) -> dict[str, Any]`
  - `build_set_terrain_pbr(cfg: PipelineConfig) -> dict[str, Any]`

- [ ] **Step 1: Write the golden fixture**

Create `tests/forge3d_pipes/data/golden_payloads.json`, transcribed from `demo1.py` (with `fov` corrected to 60.0):

```json
{
  "set_terrain": {
    "cmd": "set_terrain",
    "phi": 300.0,
    "theta": 10.5,
    "radius": 26000.0,
    "fov": 60.0,
    "zscale": 3.0,
    "sun_azimuth": 305.0,
    "sun_elevation": 24.0,
    "sun_intensity": 1.0,
    "ambient": 0.05
  },
  "set_terrain_pbr": {
    "cmd": "set_terrain_pbr",
    "enabled": true,
    "shadow_technique": "pcss",
    "shadow_map_res": 4096,
    "exposure": 1.35,
    "msaa": 8,
    "ibl_intensity": 1.0,
    "normal_strength": 1.1,
    "height_ao": {
      "enabled": true,
      "directions": 6,
      "steps": 16,
      "max_distance": 200.0,
      "strength": 1.2,
      "resolution_scale": 0.5
    },
    "sun_visibility": {
      "enabled": true,
      "mode": "soft",
      "samples": 4,
      "steps": 24,
      "max_distance": 400.0,
      "softness": 1.0,
      "bias": 0.01,
      "resolution_scale": 0.5
    },
    "materials": {
      "snow_enabled": true,
      "snow_altitude_min": 3200.0,
      "snow_altitude_blend": 300.0,
      "snow_slope_max": 50.0,
      "rock_enabled": true,
      "rock_slope_min": 42.0,
      "wetness_enabled": false,
      "wetness_strength": 0.3
    },
    "tonemap": {
      "operator": "aces",
      "white_point": 4.0,
      "white_balance_enabled": true,
      "temperature": 6000.0,
      "tint": 0.0
    },
    "lens_effects": {
      "enabled": true,
      "distortion": 0.0,
      "chromatic_aberration": 0.0,
      "vignette_strength": 0.25,
      "vignette_radius": 0.7,
      "vignette_softness": 0.3
    },
    "sky": {
      "enabled": true,
      "turbidity": 2.5,
      "ground_albedo": 0.3,
      "sun_intensity": 1.0,
      "aerial_perspective": true,
      "sky_exposure": 1.0
    }
  }
}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/forge3d_pipes/test_payloads.py`:

```python
import json
from pathlib import Path

from omegaconf import OmegaConf

from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.render import payloads

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden_payloads.json").read_text())


def _default_cfg(**overrides) -> PipelineConfig:
    merged = OmegaConf.merge(
        OmegaConf.structured(PipelineConfig),
        {"input": {"path": "dem.tif"}},
        OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]),
    )
    return OmegaConf.to_object(merged)


def test_set_terrain_matches_demo1_golden():
    assert payloads.build_set_terrain(_default_cfg()) == GOLDEN["set_terrain"]


def test_set_terrain_pbr_matches_demo1_golden():
    assert payloads.build_set_terrain_pbr(_default_cfg()) == GOLDEN["set_terrain_pbr"]


def test_enums_serialise_to_their_string_values():
    pbr = payloads.build_set_terrain_pbr(_default_cfg())
    assert pbr["shadow_technique"] == "pcss"
    assert pbr["tonemap"]["operator"] == "aces"
    assert pbr["sun_visibility"]["mode"] == "soft"


def test_overrides_reach_the_payload():
    cfg = _default_cfg(**{
        "render.camera.phi": 42.0,
        "render.pbr.exposure": 2.0,
        "render.pbr.sky.turbidity": 9.0,
    })
    assert payloads.build_set_terrain(cfg)["phi"] == 42.0
    pbr = payloads.build_set_terrain_pbr(cfg)
    assert pbr["exposure"] == 2.0
    assert pbr["sky"]["turbidity"] == 9.0


def test_payloads_are_plain_json_serialisable():
    cfg = _default_cfg()
    json.dumps(payloads.build_set_terrain(cfg))
    json.dumps(payloads.build_set_terrain_pbr(cfg))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_payloads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_art.forge3d_pipes.render'`

- [ ] **Step 4: Write the payload builders**

```bash
mkdir -p src/eo_art/forge3d_pipes/render && touch src/eo_art/forge3d_pipes/render/__init__.py
```

Create `src/eo_art/forge3d_pipes/render/payloads.py`:

```python
"""Pure config-to-IPC-payload translation. No forge3d import, no I/O."""

from dataclasses import asdict
from enum import Enum
from typing import Any

from eo_art.forge3d_pipes.config.schema import PipelineConfig


def _plain(value: Any) -> Any:
    """Recursively convert enums to their values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def build_set_terrain(cfg: PipelineConfig) -> dict[str, Any]:
    """Camera, terrain scale, and sun in one command."""
    camera = cfg.render.camera
    sun = cfg.render.sun
    return {
        "cmd": "set_terrain",
        "phi": camera.phi,
        "theta": camera.theta,
        "radius": camera.radius,
        "fov": camera.fov,
        "zscale": cfg.render.terrain.zscale,
        "sun_azimuth": sun.azimuth,
        "sun_elevation": sun.elevation,
        "sun_intensity": sun.intensity,
        "ambient": sun.ambient,
    }


def build_set_terrain_pbr(cfg: PipelineConfig) -> dict[str, Any]:
    """Full PBR block, mirroring the nested schema one-to-one."""
    payload = _plain(asdict(cfg.render.pbr))
    return {"cmd": "set_terrain_pbr", **payload}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_payloads.py -v`
Expected: PASS (5 tests). If the golden comparison fails on key order, note that dict equality ignores order — a real mismatch means a schema field name or default diverges from demo1; fix the schema, not the golden.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/render tests/forge3d_pipes/test_payloads.py tests/forge3d_pipes/data
git commit -m "feat(forge3d_pipes): build forge3d IPC payloads from config"
```

---

### Task 7: Camera animation

**Files:**
- Create: `src/eo_art/forge3d_pipes/render/animation.py`
- Create: `tests/forge3d_pipes/test_animation.py`

**Interfaces:**
- Consumes: `Orbit`, `Animation`, `AnimationKind`, `PipelineConfig` (Task 1).
- Produces:
  - `Keyframe` — frozen dataclass `(time, phi, theta, radius, fov)`, all floats.
  - `orbit_keyframes(orbit: Orbit, fps: int) -> list[Keyframe]` — pure.
  - `build_camera_animation(cfg: PipelineConfig) -> CameraAnimation | None` — `None` when `kind is NONE`.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_animation.py`:

```python
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
    cfg.animation = Animation(kind=AnimationKind.orbit, fps=4, orbit=Orbit(duration=1.0))
    result = anim.build_camera_animation(cfg)
    assert result is not None
    assert len(result.get_keyframes()) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_animation.py -v`
Expected: FAIL — `ImportError: cannot import name 'animation'`

- [ ] **Step 3: Write the animation builder**

Create `src/eo_art/forge3d_pipes/render/animation.py`:

```python
"""Camera animation.

v1 builds ``CameraAnimation`` keyframes directly rather than using forge3d's
``TerrainOrbitRig``: the rigs require a loaded ``TerrainScatterSource`` and run
clearance refinement that can raise, while linear keyframes are pure math and
testable without a GPU.
"""

from dataclasses import dataclass

from forge3d.animation import CameraAnimation

from eo_art.forge3d_pipes.config.schema import AnimationKind, Orbit, PipelineConfig


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


def build_camera_animation(cfg: PipelineConfig) -> CameraAnimation | None:
    """Return a forge3d animation, or ``None`` when animation is disabled."""
    if cfg.animation.kind is AnimationKind.none:
        return None

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_animation.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/render/animation.py tests/forge3d_pipes/test_animation.py
git commit -m "feat(forge3d_pipes): build camera animations from orbit config"
```

---

### Task 8: Sweep expansion

**Files:**
- Create: `src/eo_art/forge3d_pipes/sweep.py`
- Create: `tests/forge3d_pipes/test_sweep.py`

**Interfaces:**
- Consumes: `Sweep`, `SweepMode` (Task 1).
- Produces:
  - `Variant` — frozen dataclass with `name: str` and `overrides: tuple[str, ...]` (dotlist strings).
  - `expand(sweep: Sweep | None) -> list[Variant]` — returns `[Variant("default", ())]` when `sweep is None` or has no params.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_sweep.py`:

```python
import pytest

from eo_art.forge3d_pipes.config.schema import Sweep, SweepMode
from eo_art.forge3d_pipes.sweep import Variant, expand


def test_no_sweep_yields_single_default_variant():
    assert expand(None) == [Variant(name="default", overrides=())]


def test_empty_params_yields_single_default_variant():
    assert expand(Sweep()) == [Variant(name="default", overrides=())]


def test_product_yields_cartesian_grid():
    variants = expand(
        Sweep(
            mode=SweepMode.product,
            params={"render.pbr.exposure": [1.0, 2.0], "render.camera.phi": [10, 20]},
        )
    )
    assert len(variants) == 4
    assert [v.overrides for v in variants] == [
        ("render.pbr.exposure=1.0", "render.camera.phi=10"),
        ("render.pbr.exposure=1.0", "render.camera.phi=20"),
        ("render.pbr.exposure=2.0", "render.camera.phi=10"),
        ("render.pbr.exposure=2.0", "render.camera.phi=20"),
    ]


def test_zip_walks_lists_in_lockstep():
    variants = expand(
        Sweep(
            mode=SweepMode.zip,
            params={"render.pbr.exposure": [1.0, 2.0], "render.camera.phi": [10, 20]},
        )
    )
    assert len(variants) == 2
    assert variants[0].overrides == ("render.pbr.exposure=1.0", "render.camera.phi=10")
    assert variants[1].overrides == ("render.pbr.exposure=2.0", "render.camera.phi=20")


def test_zip_rejects_length_mismatch():
    with pytest.raises(ValueError, match="zip sweep requires equal-length"):
        expand(
            Sweep(
                mode=SweepMode.zip,
                params={"a.b": [1, 2, 3], "c.d": [1, 2]},
            )
        )


def test_variant_names_describe_their_params():
    variants = expand(
        Sweep(params={"render.pbr.tonemap.exposure": [1.35], "render.camera.phi": [280]})
    )
    assert variants[0].name == "exposure=1.35__phi=280"


def test_variant_names_are_filesystem_safe():
    variants = expand(Sweep(params={"input.path": ["/data/a b.tif"]}))
    assert variants[0].name == "path=_data_a_b.tif"


def test_variant_names_are_unique():
    variants = expand(Sweep(params={"a.x": [1, 2], "b.x": [3, 4]}))
    assert len({v.name for v in variants}) == len(variants)


def test_non_list_value_is_rejected():
    with pytest.raises(ValueError, match="must be a list"):
        expand(Sweep(params={"render.camera.phi": 10}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eo_art.forge3d_pipes.sweep'`

- [ ] **Step 3: Write the sweep expander**

Create `src/eo_art/forge3d_pipes/sweep.py`:

```python
"""Expand a sweep specification into independently renderable variants."""

import itertools
import re
from dataclasses import dataclass
from typing import Any

from eo_art.forge3d_pipes.config.schema import Sweep, SweepMode

_UNSAFE = re.compile(r"[^A-Za-z0-9.=_-]")

DEFAULT_VARIANT = "default"


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: tuple[str, ...]


def _slug(path: str, value: Any) -> str:
    leaf = path.rsplit(".", 1)[-1]
    return _UNSAFE.sub("_", f"{leaf}={value}")


def _variant(paths: list[str], values: tuple[Any, ...]) -> Variant:
    return Variant(
        name="__".join(_slug(path, value) for path, value in zip(paths, values)),
        overrides=tuple(f"{path}={value}" for path, value in zip(paths, values)),
    )


def expand(sweep: Sweep | None) -> list[Variant]:
    """Expand ``sweep`` into variants; a single default variant when unset."""
    if sweep is None or not sweep.params:
        return [Variant(name=DEFAULT_VARIANT, overrides=())]

    paths = list(sweep.params)
    for path in paths:
        if not isinstance(sweep.params[path], list):
            raise ValueError(
                f"sweep param {path!r} must be a list of values, "
                f"got {type(sweep.params[path]).__name__}"
            )
    value_lists = [list(sweep.params[path]) for path in paths]

    if sweep.mode is SweepMode.zip:
        lengths = {len(values) for values in value_lists}
        if len(lengths) > 1:
            raise ValueError(
                f"zip sweep requires equal-length value lists, got lengths {sorted(lengths)}"
            )
        combinations = zip(*value_lists)
    else:
        combinations = itertools.product(*value_lists)

    return [_variant(paths, values) for values in combinations]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_sweep.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/sweep.py tests/forge3d_pipes/test_sweep.py
git commit -m "feat(forge3d_pipes): add product and zip sweep expansion"
```

---

### Task 9: Export — resolved config and video

**Files:**
- Create: `src/eo_art/forge3d_pipes/export.py`
- Create: `tests/forge3d_pipes/test_export.py`

**Interfaces:**
- Consumes: `Video`, `VideoFormat` (Task 1).
- Produces:
  - `write_resolved_config(cfg: DictConfig, path: Path) -> Path`
  - `collect_frames(frames_dir: Path) -> list[Path]` — sorted `frame_*.png`.
  - `encode_video(frames_dir: Path, out_path: Path, video: Video) -> Path`, raising `RuntimeError` with an install hint when the encoder backend is missing.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_export.py`:

```python
import numpy as np
import pytest
from omegaconf import OmegaConf

from eo_art.forge3d_pipes import export
from eo_art.forge3d_pipes.config.schema import PipelineConfig, Video, VideoFormat


@pytest.fixture
def frames_dir(tmp_path):
    import imageio.v3 as iio

    directory = tmp_path / "frames"
    directory.mkdir()
    for index in range(4):
        frame = np.full((16, 16, 3), index * 60, dtype=np.uint8)
        iio.imwrite(directory / f"frame_{index:04d}.png", frame)
    return directory


def test_write_resolved_config_roundtrips(tmp_path):
    cfg = OmegaConf.merge(
        OmegaConf.structured(PipelineConfig), {"input": {"path": "dem.tif"}}
    )
    path = export.write_resolved_config(cfg, tmp_path / "resolved.yaml")
    assert path.exists()
    reloaded = OmegaConf.load(path)
    assert reloaded.input.path == "dem.tif"
    assert reloaded.render.width == 1200


def test_write_resolved_config_creates_parent_dirs(tmp_path):
    cfg = OmegaConf.merge(
        OmegaConf.structured(PipelineConfig), {"input": {"path": "dem.tif"}}
    )
    path = export.write_resolved_config(cfg, tmp_path / "a" / "b" / "resolved.yaml")
    assert path.exists()


def test_write_resolved_config_resolves_interpolations(tmp_path):
    """Verify interpolations are resolved to literal values in saved config."""
    cfg = OmegaConf.create(
        {
            "run": {"out_dir": "/output/myrun"},
            "export": {"video": "${run.out_dir}/video.mp4"},
        }
    )
    path = export.write_resolved_config(cfg, tmp_path / "resolved.yaml")

    # Reloaded config should have the resolved literal value
    reloaded = OmegaConf.load(path)
    assert reloaded.export.video == "/output/myrun/video.mp4"

    # Raw file text should not contain the interpolation syntax
    raw_text = path.read_text()
    assert "${" not in raw_text
    assert "/output/myrun/video.mp4" in raw_text


def test_collect_frames_is_sorted(frames_dir):
    frames = export.collect_frames(frames_dir)
    assert [f.name for f in frames] == [
        "frame_0000.png",
        "frame_0001.png",
        "frame_0002.png",
        "frame_0003.png",
    ]


def test_collect_frames_ignores_other_files(frames_dir):
    (frames_dir / "notes.txt").write_text("x")
    (frames_dir / "snapshot.png").write_text("x")
    assert len(export.collect_frames(frames_dir)) == 4


def test_encode_gif_writes_a_file(frames_dir, tmp_path):
    out = tmp_path / "out.gif"
    result = export.encode_video(
        frames_dir, out, Video(enabled=True, format=VideoFormat.gif, fps=4)
    )
    assert result == out
    assert out.stat().st_size > 0


def test_encode_rejects_empty_frames_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no frames"):
        export.encode_video(empty, tmp_path / "out.gif", Video(format=VideoFormat.gif))


def test_missing_ffmpeg_backend_gives_install_hint(frames_dir, tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise ImportError("No module named 'imageio_ffmpeg'")

    monkeypatch.setattr(export, "_write_frames", _boom)
    with pytest.raises(RuntimeError, match="eo-art\\[video\\]"):
        export.encode_video(
            frames_dir, tmp_path / "out.mp4", Video(format=VideoFormat.mp4)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_export.py -v`
Expected: FAIL — `ImportError: cannot import name 'export'`

- [ ] **Step 3: Write the export module**

Create `src/eo_art/forge3d_pipes/export.py`:

```python
"""Export: resolved-config dumps and video encoding from PNG frame sequences."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config.schema import Video, VideoFormat

FRAME_GLOB = "frame_*.png"


def write_resolved_config(cfg: DictConfig, path: Path) -> Path:
    """Save the fully merged config with interpolations resolved.

    This creates a standalone record of the exact literal values that produced
    the render, with all ``${...}`` interpolations replaced by their final
    values. This ensures reproducibility: the file can be loaded and inspected
    independently without relying on external configuration context.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
    return path


def collect_frames(frames_dir: Path) -> list[Path]:
    """Frames written by ``ViewerHandle.render_animation``, in order."""
    return sorted(Path(frames_dir).glob(FRAME_GLOB))


def _write_frames(frames: list[Path], out_path: Path, fps: int, quality: int) -> None:
    """Isolated so tests can simulate a missing encoder backend."""
    import imageio.v3 as iio

    images = [iio.imread(frame) for frame in frames]
    if out_path.suffix == ".gif":
        iio.imwrite(out_path, images, duration=1000 / fps, loop=0)
    else:
        iio.imwrite(out_path, images, fps=fps, quality=quality)


def encode_video(frames_dir: Path, out_path: Path, video: Video) -> Path:
    """Encode a frame sequence into mp4 or gif."""
    frames = collect_frames(frames_dir)
    if not frames:
        raise ValueError(f"no frames matching {FRAME_GLOB!r} in {frames_dir}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_frames(frames, out_path, video.fps, video.quality)
    except ImportError as exc:
        hint = (
            "install the optional video extra: uv add 'eo-art[video]'"
            if video.format is VideoFormat.mp4
            else "install imageio's plugin for this format"
        )
        raise RuntimeError(f"cannot encode {video.format.value}: {exc}. {hint}") from exc
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_export.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/export.py tests/forge3d_pipes/test_export.py
git commit -m "feat(forge3d_pipes): add resolved-config dump and video export"
```

---

### Task 10: Render runner

**Files:**
- Create: `src/eo_art/forge3d_pipes/render/runner.py`
- Create: `tests/forge3d_pipes/test_runner.py`

**Interfaces:**
- Consumes: `build_set_terrain`, `build_set_terrain_pbr` (Task 6), `build_camera_animation` (Task 7).
- Produces:
  - `RenderResult` — frozen dataclass with `snapshot: Path | None` and `frames_dir: Path | None`.
  - `render(cfg: PipelineConfig, terrain_path: Path, out_dir: Path) -> RenderResult`.
  - `_open_viewer(cfg, terrain_path)` — indirection point that tests monkeypatch.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_runner.py`. A fake viewer records IPC traffic, so the orchestration is verified without a GPU; one `gpu`-marked test covers the real thing.

```python
from pathlib import Path

import pytest

from eo_art.forge3d_pipes.config.schema import (
    Animation,
    AnimationKind,
    Orbit,
    PipelineConfig,
)
from eo_art.forge3d_pipes.render import runner


class FakeViewer:
    def __init__(self):
        self.commands = []
        self.snapshots = []
        self.animations = []
        self.closed = False

    def send_ipc(self, payload):
        self.commands.append(payload)

    def snapshot(self, path, width=None, height=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")
        self.snapshots.append((Path(path), width, height))

    def render_animation(self, animation, output_dir, fps=30, width=None, height=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for index in range(len(animation.get_keyframes())):
            (Path(output_dir) / f"frame_{index:04d}.png").write_bytes(b"png")
        self.animations.append((animation, Path(output_dir), fps))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


@pytest.fixture
def fake_viewer(monkeypatch):
    viewer = FakeViewer()
    monkeypatch.setattr(runner, "_open_viewer", lambda cfg, terrain_path: viewer)
    return viewer


def _cfg(**kwargs) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.input.path = "dem.tif"
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_still_render_sends_both_commands_then_snapshots(fake_viewer, tmp_path):
    result = runner.render(_cfg(), Path("dem.tif"), tmp_path)
    assert [c["cmd"] for c in fake_viewer.commands] == ["set_terrain", "set_terrain_pbr"]
    assert result.snapshot == tmp_path / "snapshot.png"
    assert result.frames_dir is None
    assert result.snapshot.exists()


def test_snapshot_uses_configured_dimensions(fake_viewer, tmp_path):
    runner.render(_cfg(), Path("dem.tif"), tmp_path)
    _, width, height = fake_viewer.snapshots[0]
    assert (width, height) == (1200, 720)


def test_snapshot_name_is_configurable(fake_viewer, tmp_path):
    cfg = _cfg()
    cfg.render.snapshot_name = "hero.png"
    result = runner.render(cfg, Path("dem.tif"), tmp_path)
    assert result.snapshot == tmp_path / "hero.png"


def test_animation_render_writes_frames_dir(fake_viewer, tmp_path):
    cfg = _cfg(
        animation=Animation(kind=AnimationKind.orbit, fps=4, orbit=Orbit(duration=1.0))
    )
    result = runner.render(cfg, Path("dem.tif"), tmp_path)
    assert result.frames_dir == tmp_path / "frames"
    assert result.snapshot is None
    assert len(list(result.frames_dir.glob("frame_*.png"))) == 5


def test_animation_render_passes_fps(fake_viewer, tmp_path):
    cfg = _cfg(
        animation=Animation(kind=AnimationKind.orbit, fps=12, orbit=Orbit(duration=1.0))
    )
    runner.render(cfg, Path("dem.tif"), tmp_path)
    assert fake_viewer.animations[0][2] == 12


def test_viewer_is_closed_even_on_failure(fake_viewer, tmp_path, monkeypatch):
    def _boom(payload):
        raise RuntimeError("ipc down")

    monkeypatch.setattr(fake_viewer, "send_ipc", _boom)
    with pytest.raises(RuntimeError, match="ipc down"):
        runner.render(_cfg(), Path("dem.tif"), tmp_path)
    assert fake_viewer.closed


@pytest.mark.gpu
def test_real_render_produces_a_png(synthetic_dem, tmp_path):
    """Opt-in smoke test: runs the actual forge3d viewer."""
    from eo_art.forge3d_pipes.prep import ops

    projected = tmp_path / "utm.tif"
    ops.reproject(synthetic_dem, projected, ops.ReprojectCfg(crs="EPSG:32610"))

    cfg = _cfg()
    cfg.render.width = 320
    cfg.render.height = 240
    cfg.render.camera.radius = 5000.0

    result = runner.render(cfg, projected, tmp_path / "out")
    assert result.snapshot is not None and result.snapshot.exists()
    assert result.snapshot.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'runner'`

- [ ] **Step 3: Write the runner**

Create `src/eo_art/forge3d_pipes/render/runner.py`:

```python
"""Drives the forge3d viewer: open, push payloads, snapshot or animate."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import forge3d as f3d

from eo_art.forge3d_pipes.config.schema import AnimationKind, PipelineConfig
from eo_art.forge3d_pipes.render.animation import build_camera_animation
from eo_art.forge3d_pipes.render.payloads import (
    build_set_terrain,
    build_set_terrain_pbr,
)

FRAMES_DIRNAME = "frames"


@dataclass(frozen=True)
class RenderResult:
    snapshot: Path | None = None
    frames_dir: Path | None = None


def _open_viewer(cfg: PipelineConfig, terrain_path: Path) -> Any:
    """Indirection point so tests can substitute a fake viewer."""
    return f3d.open_viewer_async(
        terrain_path=str(terrain_path),
        width=cfg.render.width,
        height=cfg.render.height,
        fov_deg=cfg.render.camera.fov,
    )


def render(cfg: PipelineConfig, terrain_path: Path, out_dir: Path) -> RenderResult:
    """Render a still or an animation into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    animation = build_camera_animation(cfg)

    with _open_viewer(cfg, terrain_path) as viewer:
        viewer.send_ipc(build_set_terrain(cfg))
        viewer.send_ipc(build_set_terrain_pbr(cfg))

        if cfg.animation.kind is AnimationKind.none or animation is None:
            snapshot = out_dir / cfg.render.snapshot_name
            viewer.snapshot(
                str(snapshot), width=cfg.render.width, height=cfg.render.height
            )
            return RenderResult(snapshot=snapshot)

        frames_dir = out_dir / FRAMES_DIRNAME
        viewer.render_animation(
            animation,
            str(frames_dir),
            fps=cfg.animation.fps,
            width=cfg.render.width,
            height=cfg.render.height,
        )
        return RenderResult(frames_dir=frames_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_runner.py -v`
Expected: PASS (6 tests), 1 deselected (the `gpu` test).

- [ ] **Step 5: Confirm the GPU test is opt-in and actually runs**

Run: `uv run pytest tests/forge3d_pipes/test_runner.py -m gpu -v`
Expected: the smoke test runs and passes on a GPU machine. If the environment has no GPU, record the failure output in the commit message body and move on — the deselected-by-default behaviour is what Task 10 must guarantee.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/render/runner.py tests/forge3d_pipes/test_runner.py
git commit -m "feat(forge3d_pipes): add viewer runner for stills and animations"
```

---

### Task 11: Pipeline orchestration

**Files:**
- Create: `src/eo_art/forge3d_pipes/pipeline.py`
- Modify: `src/eo_art/forge3d_pipes/__init__.py`
- Create: `tests/forge3d_pipes/test_pipeline.py`

**Interfaces:**
- Consumes: `load_raw`/`to_pipeline` (Task 3), `run_prep_chain` (Task 4), `expand` (Task 8), `render` (Task 10), `write_resolved_config`/`encode_video` (Task 9).
- Produces:
  - `VariantResult` — frozen dataclass: `name: str`, `out_dir: Path`, `ok: bool`, `error: str | None`, `snapshot: Path | None`, `frames_dir: Path | None`, `video: Path | None`.
  - `run(configs, overrides=(), out=None, use_cache=True, fail_fast=None) -> list[VariantResult]`.
  - Re-exported from `eo_art.forge3d_pipes` as `run`, plus `load_config` and `PipelineConfig`.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_pipeline.py`:

```python
import pytest
import yaml

from eo_art.forge3d_pipes import pipeline
from eo_art.forge3d_pipes.render.runner import RenderResult


@pytest.fixture
def config_file(tmp_path, synthetic_dem):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "run": {"name": "test", "out_dir": str(tmp_path / "out")},
                "prepare": [{"op": "reproject", "crs": "EPSG:32610"}],
            }
        )
    )
    return path


@pytest.fixture
def fake_render(monkeypatch):
    calls = []

    def _render(cfg, terrain_path, out_dir):
        calls.append((cfg, terrain_path, out_dir))
        snapshot = out_dir / cfg.render.snapshot_name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"png")
        return RenderResult(snapshot=snapshot)

    monkeypatch.setattr(pipeline, "render", _render)
    return calls


def test_single_variant_runs_end_to_end(config_file, fake_render, tmp_path):
    results = pipeline.run([config_file])
    assert len(results) == 1
    assert results[0].ok
    assert results[0].name == "default"
    assert results[0].snapshot.exists()


def test_resolved_config_is_written_per_variant(config_file, fake_render, tmp_path):
    results = pipeline.run([config_file])
    resolved = results[0].out_dir / "resolved.yaml"
    assert resolved.exists()
    assert yaml.safe_load(resolved.read_text())["render"]["width"] == 1200


def test_prep_runs_before_render_and_render_gets_prepared_path(
    config_file, fake_render, synthetic_dem
):
    pipeline.run([config_file])
    _, terrain_path, _ = fake_render[0]
    assert terrain_path != synthetic_dem
    assert "_prep" in str(terrain_path)


def test_sweep_produces_one_directory_per_variant(config_file, fake_render, tmp_path):
    results = pipeline.run(
        [config_file],
        overrides=[
            "sweep.mode=product",
            "sweep.params={render.pbr.exposure: [1.0, 2.0]}",
        ],
    )
    assert len(results) == 2
    assert {r.name for r in results} == {"exposure=1.0", "exposure=2.0"}
    assert all(r.out_dir.exists() for r in results)


def test_prep_is_cached_across_sweep_variants(
    config_file, fake_render, monkeypatch, tmp_path
):
    calls = []
    original = pipeline.run_prep_chain

    def _counting(src, entries, cache_dir, use_cache=True):
        calls.append(src)
        return original(src, entries, cache_dir, use_cache)

    monkeypatch.setattr(pipeline, "run_prep_chain", _counting)
    pipeline.run(
        [config_file],
        overrides=["sweep.params={render.pbr.exposure: [1.0, 2.0, 3.0]}"],
    )
    # Called once per variant, but the underlying reprojection is cached,
    # so only one output file exists.
    assert len(calls) == 3
    cache = tmp_path / "out" / "test" / "_prep"
    assert len(list(cache.glob("*.tif"))) == 1


def test_failing_variant_is_recorded_and_others_continue(
    config_file, monkeypatch, tmp_path
):
    def _render(cfg, terrain_path, out_dir):
        if cfg.render.pbr.exposure == 2.0:
            raise RuntimeError("gpu exploded")
        snapshot = out_dir / "snapshot.png"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"png")
        return RenderResult(snapshot=snapshot)

    monkeypatch.setattr(pipeline, "render", _render)
    results = pipeline.run(
        [config_file],
        overrides=["sweep.params={render.pbr.exposure: [1.0, 2.0, 3.0]}"],
    )
    assert [r.ok for r in results] == [True, False, True]
    assert "gpu exploded" in results[1].error


def test_fail_fast_aborts_on_first_error(config_file, monkeypatch):
    def _render(cfg, terrain_path, out_dir):
        raise RuntimeError("gpu exploded")

    monkeypatch.setattr(pipeline, "render", _render)
    with pytest.raises(RuntimeError, match="gpu exploded"):
        pipeline.run(
            [config_file],
            overrides=["sweep.params={render.pbr.exposure: [1.0, 2.0]}"],
            fail_fast=True,
        )


def test_missing_input_file_fails_before_rendering(tmp_path, monkeypatch):
    def _render(cfg, terrain_path, out_dir):
        raise AssertionError("render must not be reached")

    monkeypatch.setattr(pipeline, "render", _render)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"input": {"path": str(tmp_path / "nope.tif")}}))
    with pytest.raises(FileNotFoundError, match="nope.tif"):
        pipeline.run([path])


def test_bad_config_fails_before_any_variant_runs(tmp_path, synthetic_dem, monkeypatch):
    def _render(cfg, terrain_path, out_dir):
        raise AssertionError("render must not be reached")

    monkeypatch.setattr(pipeline, "render", _render)
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": str(synthetic_dem)},
                "sweep": {"params": {"render.camera.fov": [60.0, 300.0]}},
            }
        )
    )
    with pytest.raises(ValueError, match="camera.fov"):
        pipeline.run([path])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'pipeline'`

- [ ] **Step 3: Write the pipeline**

Create `src/eo_art/forge3d_pipes/pipeline.py`. Note the two validation tiers: every variant is fully validated *before* the first render starts.

```python
"""Orchestrates prep -> render -> export for every sweep variant."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config.loader import load_raw, to_pipeline
from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.export import encode_video, write_resolved_config
from eo_art.forge3d_pipes.prep.registry import run_prep_chain
from eo_art.forge3d_pipes.render.runner import RenderResult, render
from eo_art.forge3d_pipes.sweep import Variant, expand

PREP_CACHE_DIRNAME = "_prep"
RESOLVED_CONFIG_NAME = "resolved.yaml"


@dataclass(frozen=True)
class VariantResult:
    name: str
    out_dir: Path
    ok: bool
    error: str | None = None
    snapshot: Path | None = None
    frames_dir: Path | None = None
    video: Path | None = None


def _plan(
    raw: DictConfig, variants: list[Variant]
) -> list[tuple[Variant, DictConfig, PipelineConfig]]:
    """Validate every variant up front, so nothing runs on a broken plan."""
    plan = []
    for variant in variants:
        merged = (
            OmegaConf.merge(raw, OmegaConf.from_dotlist(list(variant.overrides)))
            if variant.overrides
            else raw
        )
        plan.append((variant, merged, to_pipeline(merged)))
    return plan


def _run_variant(
    cfg: PipelineConfig,
    merged: DictConfig,
    out_dir: Path,
    cache_dir: Path,
    use_cache: bool,
) -> RenderResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(merged, out_dir / RESOLVED_CONFIG_NAME)

    prepared = run_prep_chain(
        Path(cfg.input.path), cfg.prepare, cache_dir, use_cache=use_cache
    )
    return render(cfg, prepared, out_dir)


def run(
    configs: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
    use_cache: bool = True,
    fail_fast: bool | None = None,
) -> list[VariantResult]:
    """Load, validate, and execute the pipeline for every sweep variant."""
    raw = load_raw(configs, overrides, out)
    root = to_pipeline(raw)

    source = Path(root.input.path)
    if not source.exists():
        raise FileNotFoundError(f"input raster not found: {source}")

    plan = _plan(raw, expand(root.sweep))

    run_root = Path(root.run.out_dir) / root.run.name
    cache_dir = run_root / PREP_CACHE_DIRNAME
    abort_on_error = root.run.fail_fast if fail_fast is None else fail_fast

    results: list[VariantResult] = []
    for variant, merged, cfg in plan:
        out_dir = run_root / variant.name
        try:
            rendered = _run_variant(cfg, merged, out_dir, cache_dir, use_cache)
            video = None
            if cfg.export.video.enabled and rendered.frames_dir is not None:
                video = encode_video(
                    rendered.frames_dir,
                    out_dir / f"video.{cfg.export.video.format.value}",
                    cfg.export.video,
                )
            results.append(
                VariantResult(
                    name=variant.name,
                    out_dir=out_dir,
                    ok=True,
                    snapshot=rendered.snapshot,
                    frames_dir=rendered.frames_dir,
                    video=video,
                )
            )
        except Exception as exc:
            if abort_on_error:
                raise
            results.append(
                VariantResult(
                    name=variant.name, out_dir=out_dir, ok=False, error=str(exc)
                )
            )
    return results
```

- [ ] **Step 4: Write the public API**

Replace `src/eo_art/forge3d_pipes/__init__.py`:

```python
"""Config-driven forge3d render pipelines."""

from eo_art.forge3d_pipes.config.loader import load_config
from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.pipeline import VariantResult, run

__all__ = ["PipelineConfig", "VariantResult", "load_config", "run"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_pipeline.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add src/eo_art/forge3d_pipes/pipeline.py src/eo_art/forge3d_pipes/__init__.py tests/forge3d_pipes/test_pipeline.py
git commit -m "feat(forge3d_pipes): orchestrate prep, render, and export per variant"
```

---

### Task 12: CLI, shipped configs, and demo1 removal

**Files:**
- Create: `src/eo_art/forge3d_pipes/cli.py`
- Create: `configs/base.yaml`, `configs/looks/alpine_dusk.yaml`
- Modify: `pyproject.toml` (console script)
- Delete: `src/eo_art/forge3d_pipes/demo1.py`
- Create: `tests/forge3d_pipes/test_cli.py`

**Interfaces:**
- Consumes: `pipeline.run` (Task 11).
- Produces: `main(argv: Sequence[str] | None = None) -> int`, exposed as console script `eo-art-f3d`.

- [ ] **Step 1: Write the failing tests**

Create `tests/forge3d_pipes/test_cli.py`:

```python
import pytest
import yaml

from eo_art.forge3d_pipes import cli
from eo_art.forge3d_pipes.pipeline import VariantResult


@pytest.fixture
def captured_run(monkeypatch, tmp_path):
    calls = {}

    def _run(configs, overrides=(), out=None, use_cache=True, fail_fast=None):
        calls.update(
            configs=list(configs),
            overrides=list(overrides),
            out=out,
            use_cache=use_cache,
            fail_fast=fail_fast,
        )
        return [VariantResult(name="default", out_dir=tmp_path, ok=True)]

    monkeypatch.setattr(cli, "run", _run)
    return calls


@pytest.fixture
def cfg_path(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"input": {"path": "dem.tif"}}))
    return path


def test_run_passes_configs_in_order(captured_run, cfg_path, tmp_path):
    other = tmp_path / "look.yaml"
    other.write_text("render:\n  width: 100\n")
    assert cli.main(["run", str(cfg_path), str(other)]) == 0
    assert captured_run["configs"] == [str(cfg_path), str(other)]


def test_set_flags_become_overrides(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--set", "render.width=99", "--set", "run.name=x"])
    assert captured_run["overrides"] == ["render.width=99", "run.name=x"]


def test_out_flag_is_forwarded(captured_run, cfg_path, tmp_path):
    cli.main(["run", str(cfg_path), "--out", str(tmp_path / "renders")])
    assert captured_run["out"] == str(tmp_path / "renders")


def test_sweep_shorthand_becomes_an_override(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--sweep", "render.pbr.exposure=1.0,1.35,1.8"])
    assert captured_run["overrides"] == [
        "sweep.params={render.pbr.exposure: [1.0, 1.35, 1.8]}"
    ]


def test_no_cache_flag_disables_caching(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--no-cache"])
    assert captured_run["use_cache"] is False


def test_fail_fast_flag(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--fail-fast"])
    assert captured_run["fail_fast"] is True


def test_exit_code_is_nonzero_when_a_variant_failed(monkeypatch, cfg_path, tmp_path):
    monkeypatch.setattr(
        cli,
        "run",
        lambda *a, **k: [
            VariantResult(name="a", out_dir=tmp_path, ok=True),
            VariantResult(name="b", out_dir=tmp_path, ok=False, error="boom"),
        ],
    )
    assert cli.main(["run", str(cfg_path)]) == 1


def test_summary_is_printed(monkeypatch, cfg_path, tmp_path, capsys):
    monkeypatch.setattr(
        cli,
        "run",
        lambda *a, **k: [
            VariantResult(name="a", out_dir=tmp_path, ok=True),
            VariantResult(name="b", out_dir=tmp_path, ok=False, error="boom"),
        ],
    )
    cli.main(["run", str(cfg_path)])
    output = capsys.readouterr().out
    assert "1 succeeded" in output
    assert "1 failed" in output
    assert "boom" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/forge3d_pipes/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'cli'`

- [ ] **Step 3: Write the CLI**

Create `src/eo_art/forge3d_pipes/cli.py`:

```python
"""Thin argparse wrapper over ``pipeline.run``."""

import argparse
from collections.abc import Sequence

from eo_art.forge3d_pipes.pipeline import run


def _sweep_to_override(spec: str) -> str:
    """Turn ``path=a,b,c`` into an OmegaConf dotlist assignment."""
    path, _, values = spec.partition("=")
    if not values:
        raise argparse.ArgumentTypeError(
            f"--sweep expects 'dotted.path=v1,v2', got {spec!r}"
        )
    items = ", ".join(value.strip() for value in values.split(","))
    return f"sweep.params={{{path}: [{items}]}}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eo-art-f3d")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="run a render pipeline")
    run_cmd.add_argument(
        "configs", nargs="+", help="config files, merged left to right"
    )
    run_cmd.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="PATH=VALUE", help="dotlist override (repeatable)",
    )
    run_cmd.add_argument("--out", default=None, help="shorthand for run.out_dir")
    run_cmd.add_argument(
        "--sweep", default=None, metavar="PATH=V1,V2",
        help="shorthand for a single-parameter sweep",
    )
    run_cmd.add_argument(
        "--no-cache", action="store_true", help="recompute the prep chain"
    )
    run_cmd.add_argument(
        "--fail-fast", action="store_true", help="abort on the first failing variant"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    overrides = list(args.overrides)
    if args.sweep:
        overrides.append(_sweep_to_override(args.sweep))

    results = run(
        args.configs,
        overrides=overrides,
        out=args.out,
        use_cache=not args.no_cache,
        fail_fast=args.fail_fast or None,
    )

    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    for result in failed:
        print(f"FAILED {result.name}: {result.error}")
    for result in succeeded:
        print(f"ok     {result.name} -> {result.out_dir}")
    print(f"{len(succeeded)} succeeded, {len(failed)} failed")
    return 1 if failed else 0
```

- [ ] **Step 4: Register the console script**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
eo-art-f3d = "eo_art.forge3d_pipes.cli:main"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/forge3d_pipes/test_cli.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Write the shipped configs**

Create `configs/base.yaml` — reproduces demo1's render, relying on schema defaults for everything unchanged:

```yaml
# Reproduces demo1.py. Every unset value comes from the schema defaults.
run:
  name: rainier
  out_dir: out

input:
  path: rainier.tif   # override with --set input.path=...

prepare:
  - op: reproject
    crs: EPSG:32610
    resampling: nearest

render:
  width: 1200
  height: 720
  snapshot_name: rainier.png
```

Create `configs/looks/alpine_dusk.yaml` — an override demonstrating composition:

```yaml
# Warmer, lower sun with a stronger vignette. Merge on top of base.yaml.
render:
  sun:
    azimuth: 285.0
    elevation: 8.0
  pbr:
    exposure: 1.6
    tonemap:
      temperature: 4800.0
    lens_effects:
      vignette_strength: 0.4
    sky:
      turbidity: 4.0
```

- [ ] **Step 7: Verify the shipped configs load**

Run:
```bash
uv run python -c "
from eo_art.forge3d_pipes import load_config
cfg = load_config(['configs/base.yaml', 'configs/looks/alpine_dusk.yaml'])
print(cfg.render.snapshot_name, cfg.render.pbr.exposure, cfg.render.sun.elevation)
"
```
Expected: `rainier.png 1.6 8.0`

- [ ] **Step 8: Delete demo1.py and confirm nothing references it**

```bash
git rm src/eo_art/forge3d_pipes/demo1.py
grep -rn "demo1" src tests configs docs || echo "no references"
```
Expected: `no references`

- [ ] **Step 9: Run the full suite and the type checker**

Run: `uv run pytest -v && uv run ruff check src tests && uv run ty check`
Expected: all tests pass (97 collected, 1 `gpu` test deselected), no lint errors, no type errors.

- [ ] **Step 10: Commit**

```bash
uv run ruff format src tests && uv run ruff check --fix src tests
git add -A
git commit -m "feat(forge3d_pipes): add CLI and shipped configs, remove demo1"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| OmegaConf, no Hydra | 1, 3 |
| Base + override files + dotlist merge order | 3 |
| Typed schemas mirroring IPC payloads | 1, 6 |
| Unknown-key / enum / range validation | 1, 3 |
| `fov: 300` rejected | 1 |
| Prep op registry, `@register_op` | 2 |
| Prep entries validated at load time | 2, 3 |
| `reproject` (own) + `scale_to_gsd` (vecraspy) | 5 |
| Prep caching by chain hash, `--no-cache` | 4, 12 |
| Pure `payloads.py`, golden fixture | 6 |
| Camera animation | 7 |
| Sweep: product/zip, slug names, length mismatch | 8 |
| Two-tier error handling, `--fail-fast` | 11, 12 |
| `resolved.yaml` written before rendering | 11 |
| Video export, `video` extra, clear error | 9, 1 |
| API + CLI, `--out` → `run.out_dir` | 3, 11, 12 |
| Test layers incl. opt-in `gpu` marker | 1, 10 |
| `demo1.py` deleted | 12 |

**Type consistency:** `RenderResult(snapshot, frames_dir)` is produced in Task 10 and consumed in Task 11's `fake_render` fixture and `_run_variant`. `Variant(name, overrides)` is produced in Task 8 and consumed in Task 11's `_plan`. `run_prep_chain(src, entries, cache_dir, use_cache)` is defined in Task 4 and called with that exact signature in Task 11. `validate_entry` returns `tuple[RegisteredOp, Any]` in Task 2 and is unpacked as such in Task 4. `Video` is consumed by `encode_video` in Task 9 and passed as `cfg.export.video` in Task 11.

**Placeholder scan:** No TBDs. Every code step contains complete implementations and complete test bodies.

**Known ordering constraint:** Task 3 (`loader`) imports `prep.registry` (Task 2), so Task 2 must land first. Task 5's `prep/__init__.py` imports `ops`, which imports `registry` — `loader` imports `prep.registry` directly rather than the package, avoiding a cycle.
