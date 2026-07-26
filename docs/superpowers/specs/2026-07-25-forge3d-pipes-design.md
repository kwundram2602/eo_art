# forge3d_pipes — Config-Driven Render Pipeline

**Date:** 2026-07-25
**Status:** Approved design

## Problem

`src/eo_art/forge3d_pipes/demo1.py` renders a terrain scene with forge3d, but every
value is a module-level constant. Changing a look means editing the script; comparing
two looks means copying it. About eighty knobs sit in literal dicts with no validation,
so a misspelled key is silently ignored.

The script also does not run as written: `from vecraspy import reproject_raster`
(line 4) raises `ImportError` — the installed vecraspy exposes no such function. The
import is unused; the reprojection below it is hand-rolled `rasterio.warp` code.
A second latent bug: `CAM_FOV = 300.0` is passed as the field of view while the
comment claims the flag was never set. A 300-degree FOV is meaningless.

This design replaces the script with a config-driven, modular pipeline.

## Scope

In scope: preparation, rendering, and export — including camera animation and
parameter sweeps.

Out of scope: data acquisition. No STAC, no DEM downloads. Input is a local raster
path. Preparation delegates to existing packages (vecraspy, rasterio) rather than
implementing raster algorithms.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Config engine | OmegaConf, no Hydra | Base + override merging is all that is needed; Hydra wants to own the entry point, which conflicts with library use |
| Config composition | Base config + override files + CLI dotlist | Explicit merge, no config-group machinery to learn |
| Schema abstraction | Typed dataclasses mirroring forge3d IPC payloads | Validation catches silently-ignored keys; payload construction stays mechanical |
| Prep modularity | Declarative op list + registry | Several ops are known to be coming; adding one must not touch the pipeline |
| Render modularity | Concrete function, no backend registry | Only one renderer exists; an interface designed against one implementation is usually wrong when the second arrives |
| Multi-frame | Camera animation *and* sweeps, as independent mechanisms | They compose: a sweep variant may itself render an animation |
| Invocation | Python API plus thin argparse CLI | Dotlist overrides make iteration fast; the API remains the real interface |
| Testing | Pure logic + synthetic-raster integration + opt-in GPU smoke test | The GPU path is the biggest risk but must not block CI |

Rejected: running preparation outside the pipeline (user reprojects by hand, config
takes a prepared path). Simpler, but it moves the DEM's provenance outside the
reproducible config, so `resolved.yaml` would no longer describe how a render was made.

## Architecture

Everything is path-in / path-out. forge3d loads terrain from a file and writes
snapshots to files, so no raster needs to live in eo_art's memory. Each stage takes a
config plus an input path and returns an output path.

```
config files ─┬─► load & merge ──► PipelineConfig (validated)
override files├─                          │
--set dotlist ┘                           ▼
                                    sweep.expand()
                                          │  N variants (N=1 if no sweep)
                                          ▼
                        ┌─────────── per variant ────────────┐
                        │  prep chain    render      export  │
                   DEM ─┼─► op ► op ─► prepared ─► frames ─► mp4/gif
                        │   (cached by prep-config hash)     │
                        └────────────────────────────────────┘
                                          │
                                  out/<run>/<variant>/
                                    ├── frames/*.png  (or snapshot.png)
                                    ├── video.mp4     (if export enabled)
                                    └── resolved.yaml (fully merged config)
```

### Modules

| Module | Responsibility |
|---|---|
| `config/schema.py` | Dataclasses defining every knob and its default |
| `config/loader.py` | Merge base + overrides + dotlist, validate, resolve |
| `prep/registry.py` | `@register_op`, `run_prep_chain`, caching |
| `prep/ops.py` | Op implementations and thin wrappers |
| `render/payloads.py` | Pure functions: typed config → forge3d IPC dicts |
| `render/animation.py` | Config → forge3d `CameraAnimation` / rig |
| `render/runner.py` | Opens viewer, sends payloads, snapshots |
| `sweep.py` | Expand sweep spec → list of `(variant_name, config)` |
| `export.py` | PNG sequence → mp4/gif; dump `resolved.yaml` |
| `pipeline.py` | Orchestrates the three stages per variant |
| `cli.py` | argparse → `run()` |

`payloads.py` is pure and separate from `runner.py`. Payload builders are then tested
exhaustively without a GPU, and `runner.py` shrinks to "open viewer, send these dicts,
snapshot" — the only part needing the opt-in GPU test.

## Config System

### Schema

Dataclasses mirror the IPC payload structure, so payload construction is mechanical:

```python
@dataclass
class Camera:
    phi: float = 300.0
    theta: float = 10.5
    radius: float = 26000.0
    fov: float = 60.0          # validated 1..179

@dataclass
class Sun:
    azimuth: float = 305.0     # 0..360
    elevation: float = 24.0    # 0..90
    intensity: float = 1.0
    ambient: float = 0.05

@dataclass
class Pbr:
    enabled: bool = True
    shadow_technique: ShadowTechnique = ShadowTechnique.PCSS
    shadow_map_res: int = 4096
    exposure: float = 1.35
    msaa: int = 8              # validated in {1, 2, 4, 8}
    normal_strength: float = 1.1
    height_ao: HeightAO = field(default_factory=HeightAO)
    sun_visibility: SunVisibility = field(default_factory=SunVisibility)
    materials: Materials = field(default_factory=Materials)
    tonemap: Tonemap = field(default_factory=Tonemap)
    lens_effects: LensEffects = field(default_factory=LensEffects)
    sky: Sky = field(default_factory=Sky)
```

Every render parameter default is demo1's value — camera, sun, terrain, and PBR settings —
except `Camera.fov`, which becomes a valid default rather than demo1's erroneous 300.
The output filename `Render.snapshot_name` defaults to a generic `snapshot.png`, not a
scene-specific value. An otherwise empty config therefore reproduces the current render;
`base.yaml` carries only what differs.

Nested blocks — `HeightAO`, `SunVisibility`, `Materials`, `Tonemap`, `LensEffects`,
`Sky` — follow the same pattern, one dataclass per nested dict in the
`set_terrain_pbr` payload.

### Merge and validation

Merge order, later wins:

```
schema defaults → config files (N, in CLI order) → --set dotlist
```

`OmegaConf.merge()` against a structured config runs in struct mode, so an unknown key
such as `sun_azimut` raises instead of being silently dropped. Enums reject invalid
values like `tonemap: "aces2"`. `OmegaConf.to_object()` instantiates the real
dataclasses at the end, and `__post_init__` performs range checks there — this is where
`fov: 300` fails, before the GPU spins up.

`${...}` interpolation works throughout, e.g. `export.video: ${run.out_dir}/video.mp4`.

### Invocation

Both paths call the same function:

```bash
eo-art-f3d run configs/base.yaml configs/looks/alpine_dusk.yaml \
    --set render.camera.phi=280 --set render.pbr.exposure=1.6 \
    --out out/
```

```python
from eo_art.forge3d_pipes import run
run(["configs/base.yaml", "configs/looks/alpine_dusk.yaml"],
    overrides=["render.camera.phi=280"], out="out/")
```

Multiple positional configs are the override mechanism — no separate flag, no
`defaults:` list. `--set` values are parsed by `OmegaConf.from_dotlist`. `--out` is
shorthand for setting `run.out_dir` and is applied as part of the dotlist layer, so it
wins over the config files.

Sweeps live in the config, with a CLI shorthand:

```yaml
sweep:
  mode: product          # product | zip
  params:
    render.pbr.exposure: [1.0, 1.35, 1.8]
    render.camera.phi: [280, 300]
```

```bash
--sweep render.pbr.exposure=1.0,1.35,1.8
```

## Execution

### Prep registry

Op signature `(src: Path, dst: Path, cfg: OpCfg) -> Path`, matching vecraspy's
`(input_path, output_path, ..., *, resampling: str) -> Path` convention. Registration
binds a name to both function and schema:

```python
@register_op("reproject", ReprojectCfg)
def reproject(src: Path, dst: Path, cfg: ReprojectCfg) -> Path: ...
```

Config form:

```yaml
prepare:
  - op: reproject
    crs: EPSG:32610
    resampling: bilinear
  - op: scale_to_gsd
    target_gsd: 30.0
```

Prep entries are typed `list[Any]` at the top level, because OmegaConf cannot express a
polymorphic union cleanly. Each entry is validated inside the registry by merging
against the named op's schema. This validation runs at config-load time, not at
execution time: an unknown op name or bad parameter fails before any raster work starts.

v1 ships two ops: `reproject` (own implementation on `rasterio.warp`, lifted from
demo1's working code) and `scale_to_gsd` (wrapping `vecraspy.scale_raster_to_gsd`).
Two ops prove the registry handles both an in-house op and a delegated one. Further
vecraspy wrappers — `clip_tif_by_aoi`, `align_raster_grid`, `merge_tifs` — are each a
function plus a dataclass, with no pipeline changes.

### Caching

Key = sha256 of the resolved input path, its size and mtime, and the canonical JSON of
the prep chain. Result stored at `out/<run>/_prep/<hash>.tif`; a hit skips the chain.
`--no-cache` forces recomputation. Without this, a sweep over six looks would reproject
the same DEM six times.

### Render

`runner.py` opens the viewer via `f3d.open_viewer_async(...)`, sends `set_terrain` then
`set_terrain_pbr` built by `payloads.py`, then branches: `snapshot()` for stills,
`render_animation()` for sequences.

`animation.py` maps a config block onto forge3d's existing rigs. `TerrainOrbitRig`'s
fields (`target_xz`, `duration`, `radius`, `phi_start_deg`, `phi_end_deg`,
`theta_start_deg`, `theta_end_deg`, `fov_start_deg`, …) map to config keys almost 1:1,
so this is a constructor call, not interpolation logic. forge3d owns camera
interpolation; eo_art only drives it.

```yaml
animation:
  kind: orbit            # orbit | none
  fps: 30
  orbit:
    duration: 8.0
    phi_start_deg: 0.0
    phi_end_deg: 360.0
    theta_start_deg: 12.0
    radius: 26000.0
```

v1 supports `kind: orbit` and `kind: none`. `rail` and `keyframes` map onto
`TerrainRailRig` and `CameraAnimation` the same way and can be added later.

### Sweep

`mode: product` produces the cartesian grid; `mode: zip` walks the lists in lockstep and
errors on length mismatch. Variant directory names are slugified from the varying
parameters (`exposure=1.35__phi=280`), making output directories self-describing. Each
variant deep-copies the config, applies its overrides, and runs the full pipeline into
its own directory. With no `sweep` block, there is exactly one variant, named `default`.

### Error handling

Two tiers:

- **Config-time** — schema, ranges, unknown ops, sweep consistency, input file
  existence. All validated up front; nothing runs until the whole plan is valid.
- **Run-time** — a failing variant (GPU error, viewer crash) is recorded and the sweep
  continues, with a success/failure summary at the end and a non-zero exit code if any
  failed. `--fail-fast` aborts on the first error. The viewer stays a context manager,
  so a crashed variant cannot leak a subprocess into the next.

### Export

imageio writes mp4/gif from the PNG sequence. mp4 requires `imageio-ffmpeg`, which is
not currently a dependency; it is added as an optional extra `video`, and its absence
raises a clear message rather than failing deep inside imageio.

`resolved.yaml` is written per variant *before* rendering starts, so a crashed run still
leaves behind the config that caused it.

## Testing

| Layer | Covers |
|---|---|
| Unit, no I/O | Merge order, unknown-key rejection, dotlist parsing, enum validation, range checks (including `fov: 300`), registry dispatch, per-op validation, sweep expansion and naming, zip length mismatch |
| Payload golden test | Config → IPC dicts compared against a checked-in golden fixture transcribed from demo1's literals (with the corrected FOV), pinning payload structure and values |
| Integration, no GPU | Synthetic GeoTIFFs written via rasterio, run through the real prep chain, asserting CRS and shape; cache hit and miss |
| `@pytest.mark.gpu`, opt-in | One tiny end-to-end render, asserting a PNG exists with the expected dimensions |

`pyproject.toml` gains `markers = ["gpu"]` and `addopts = "-m 'not gpu'"`, so `pytest`
skips GPU work by default and `pytest -m gpu` runs it.

## Dependencies

Added: `omegaconf`. Added as optional extra `video`: `imageio-ffmpeg`.
No new dependency on Hydra.

## Deliverables

- The module tree above. `demo1.py` is deleted once `configs/base.yaml` reproduces it;
  its literal values survive as schema defaults and as the golden payload fixture.
- `configs/base.yaml` reproducing demo1's render (modulo the corrected FOV), plus at
  least one look override demonstrating composition.
- `eo-art-f3d` console script.
- Test suite as specified.
