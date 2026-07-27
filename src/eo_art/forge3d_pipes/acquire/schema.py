"""Typed configuration schema for the acquire stage (terrain + optical inputs)."""

from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING

from eo_art.forge3d_pipes.config.schema import ResamplingName


class DemSource(Enum):
    copernicus = "copernicus"
    fabdem = "fabdem"
    srtm = "srtm"


@dataclass
class Sentinel:
    start_date: str = MISSING
    end_date: str = MISSING
    model_path: str = MISSING
    collection: str = "sentinel-2-l2a"
    bands: tuple[str, ...] = (
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    )
    edge_size: int = 128
    resolution: int = 10
    stac: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    max_items: int | None = None
    max_cloud_cover: float | None = None
    # Which of the (possibly many) super-resolved timesteps to carry forward
    # into the DEM/DTM-SR/align steps. None (default) auto-selects the scene
    # with the lowest eo:cloud_cover; set explicitly to pick by position
    # instead.
    reference_index: int | None = None

    def __post_init__(self) -> None:
        if self.reference_index is not None and self.reference_index < 0:
            raise ValueError(
                f"sentinel.reference_index must be >= 0, got {self.reference_index}"
            )


@dataclass
class Dem:
    source: DemSource = DemSource.copernicus
    scale: float = 30.0


@dataclass
class Dtm:
    apply_erosion: bool = False
    erosion_kwargs: dict = field(default_factory=dict)
    radius: int = 8
    eps: float = 1e-2
    band: int | None = None


@dataclass
class Align:
    resampling: ResamplingName = ResamplingName.bilinear


@dataclass
class AcquireConfig:
    aoi_path: str = MISSING
    ee_project: str = MISSING
    # Where dtm.tif/optical_aligned.tif land (plus the internal _acquire
    # cache). Overridden by --out on the CLI when given.
    out_dir: str = "out"
    # Path to a local checkout of the `sentinel_sr` project (it ships its own
    # torch/cubo/mlstac stack in an isolated venv, so it can't be bundled
    # inside eo-art's own wheel/site-packages install; run_sentinel_sr shells
    # out to it via `uv run --project <sentinel_sr_dir>`).
    sentinel_sr_dir: str = MISSING
    sentinel: Sentinel = field(default_factory=Sentinel)
    dem: Dem = field(default_factory=Dem)
    dtm: Dtm = field(default_factory=Dtm)
    align: Align = field(default_factory=Align)
