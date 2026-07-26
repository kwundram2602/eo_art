# Earth Observation Art


## Forge3d Pipelines

Config-driven terrain rendering on top of [forge3d](https://pypi.org/project/forge3d/).
A YAML file describes the whole render — input DEM, preparation steps, camera, sun,
PBR look, animation, export — so producing a variant means editing config, not code.

### Install

```bash
uv sync
uv sync --extra video   # only needed for mp4 export
```

### Run

```bash
uv run eo-art-f3d run configs/base.yaml configs/looks/alpine_dusk.yaml --out out/
```

Configs merge left to right, so `base.yaml` carries the scene and each look file
overrides only the mood. A look on its own is not a complete config — it has no
`input.path`.

Point it at your own DEM with `--set`:

```bash
uv run eo-art-f3d run configs/base.yaml --set input.path=/data/my_dem.tif
```

### Flags

| Flag | Meaning |
| --- | --- |
| `--set PATH=VALUE` | Override any config value by dotted path. Repeatable. |
| `--out DIR` | Shorthand for `run.out_dir`. |
| `--sweep PATH=V1,V2` | Render one variant per value. **Adds to** any sweep already in the config rather than replacing it. |
| `--no-cache` | Recompute the prep chain instead of reusing the cached result. |
| `--fail-fast` | Abort on the first failing variant instead of continuing. |

Exit code is `1` if any variant failed, `0` otherwise.

### Sweeps

Render the same scene at several exposures in one command:

```bash
uv run eo-art-f3d run configs/base.yaml --sweep render.pbr.exposure=1.0,1.35,1.8
```

Each variant gets its own directory named after the values that vary
(`out/<run.name>/exposure=1.35/`), containing the rendered image, a `resolved.yaml`
recording the exact fully-merged config that produced it, and `video.mp4`/`video.gif`
if video export is enabled. Reprojection is cached and shared across variants, so the
DEM is prepared once no matter how many variants you render.

For a grid over several parameters, declare the sweep in the config:

```yaml
sweep:
  mode: product          # product = every combination; zip = walk lists in lockstep
  params:
    render.pbr.exposure: [1.0, 1.8]
    render.camera.phi: [280, 300]
```

### Config outline

```yaml
run:
  name: rainier          # output goes to <out_dir>/<name>/<variant>/
  out_dir: out
  cache: true            # reuse cached prep results
  fail_fast: false

input:
  path: rainier.tif      # a local raster; nothing is downloaded for you

prepare:                 # ordered; omit entirely to render the DEM as-is
  - op: reproject
    crs: EPSG:32610
    resampling: nearest  # nearest | bilinear | cubic
  - op: scale_to_gsd
    target_gsd: 30.0

render:
  width: 1200
  height: 720
  snapshot_name: rainier.png
  camera: {phi: 300.0, theta: 10.5, radius: 26000.0, fov: 60.0}
  sun: {azimuth: 305.0, elevation: 24.0, intensity: 1.0, ambient: 0.05}
  terrain: {zscale: 3.0}
  pbr:
    exposure: 1.35
    shadow_technique: pcss     # pcss | pcf | none
    tonemap: {operator: aces}  # aces | reinhard | linear
    sky: {turbidity: 2.5}

animation:
  kind: none             # none | orbit
  fps: 30
  orbit: {duration: 8.0, phi_start: 0.0, phi_end: 360.0}

export:
  video:
    enabled: false       # requires animation.kind: orbit
    format: mp4          # mp4 | gif
```

Enum values are lowercase. Unknown keys, out-of-range values, and unknown prep ops are
all rejected when the config loads, before any GPU work starts — so a typo costs a
second, not a render.

### Python API

The CLI is a thin wrapper; the same thing from Python:

```python
from eo_art.forge3d_pipes import run

results = run(
    ["configs/base.yaml", "configs/looks/alpine_dusk.yaml"],
    overrides=["render.pbr.exposure=1.6"],
    out="out/",
)
for result in results:
    print(result.name, result.ok, result.snapshot)
```

### Adding a prep op

Prep ops are pluggable. An op writes `src` to `dst` and returns the path it wrote:

```python
from dataclasses import dataclass
from pathlib import Path
from eo_art.forge3d_pipes.prep.registry import register_op

@dataclass
class SmoothCfg:
    sigma: float = 1.0

@register_op("smooth", SmoothCfg)
def smooth(src: Path, dst: Path, cfg: SmoothCfg) -> Path:
    ...
    return dst
```

It is then usable as `- op: smooth` in any config, and inherits load-time validation
and caching for free. Renderers are not pluggable in this version — forge3d is the
only backend.

### Tests

```bash
uv run pytest              # 122 tests, no GPU needed
uv run pytest -m gpu       # opt-in: opens a real viewer and renders
```
