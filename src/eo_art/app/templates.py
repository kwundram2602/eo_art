"""Starter configs seeded into a freshly created project.

The render template reads exactly the files the acquire stage writes
(``out/dtm.tif``, ``out/optical_aligned.tif``), so a new project can run
Acquire -> Render without editing any paths.
"""

ACQUIRE_TEMPLATE = """\
# Fetches terrain + optical inputs. Paths are relative to the project folder.
aoi_path: aoi/aoi.gpkg
ee_project: your-ee-project
out_dir: out

# Local checkout of the sentinel_sr project (own torch stack, own venv).
# The project sits at <start folder>/projects/<name>, so ../../ is the
# folder eoart-app was started in.
sentinel_sr_dir: ../../sentinel_sr

sentinel:
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  model_path: /path/to/LDSRS2-SEN2SR
  max_items: 10
  max_cloud_cover: 20
  # Drop scenes with more than this % of cloudy AOI pixels (SCL classes).
  max_aoi_cloud: 10
  scl_cloud_classes: [8, 9]  # + 3 shadow, 10 cirrus, 0/1 no data/defective
  # SR-only runs: one scene per interval (the least cloudy one) instead of a
  # single reference scene. Acquire ignores this block.
  timeseries:
    enabled: false
    interval: month  # day | week | month | year
    every: 1
    fill_gaps: false  # true: keep existing timesteps, only fill empty intervals

dem:
  source: fabdem
  scale: 30

dtm:
  apply_erosion: true
  erosion_kwargs:
    iterations: 30

align:
  resampling: bilinear
"""

RENDER_TEMPLATE = """\
# Renders the acquire outputs. Paths are relative to the project folder.
run:
  name: {name}
  out_dir: out

input:
  path: out/dtm.tif

overlays:
  - name: optical
    path: out/optical_aligned.tif

render:
  width: 2560
  height: 1440
  snapshot_name: {name}.png
"""

ACQUIRE_FILENAME = "acquire.yaml"
RENDER_FILENAME = "render.yaml"


def starter_configs(project_name: str) -> dict[str, str]:
    """Filename -> YAML text for every config a new project starts with."""
    return {
        ACQUIRE_FILENAME: ACQUIRE_TEMPLATE,
        RENDER_FILENAME: RENDER_TEMPLATE.format(name=project_name),
    }
