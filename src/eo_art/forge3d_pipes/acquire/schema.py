"""Typed configuration schema for the acquire stage (terrain + optical inputs)."""

from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING

from eo_art.forge3d_pipes.config.schema import ResamplingName


class DemSource(Enum):
    copernicus = "copernicus"
    fabdem = "fabdem"
    srtm = "srtm"
    local = "local"


class DtmMode(Enum):
    # Optical is master: super_resolve_dtm warps the DEM onto the optical
    # grid. Correct when the DEM is coarser than the imagery (the 30 m
    # Earth Engine sources), but wastes real resolution on a high-res local
    # DEM by downsampling it to match the optical grid.
    guided_upsample = "guided_upsample"
    # DEM is master: the optical raster is aligned onto the DEM's own grid
    # instead, so a high-res local DEM keeps its real elevation resolution.
    native = "native"


class Interval(Enum):
    day = "day"
    week = "week"
    month = "month"
    year = "year"


@dataclass
class Timeseries:
    """SR-only (``eo-art-sr``): keep one scene per interval instead of a single
    reference scene. ``run_acquire`` ignores this block."""

    enabled: bool = False
    interval: Interval = Interval.month
    # One scene every `every` intervals (e.g. interval=week, every=2: biweekly).
    every: int = 1
    # Continue the latest SR run's <aoi>_sr_timeseries/ (copied into the new
    # run) instead of rebuilding it: intervals that already hold a timestep
    # there are skipped, the open ones are searched
    # again on every run (e.g. after extending end_date or relaxing
    # max_aoi_cloud), and existing timesteps are never removed. Off: the folder
    # is reset to exactly this config's series.
    fill_gaps: bool = False
    # With sentinel.zones: per interval, the zones' acquisition dates may lie
    # at most this many days apart, so each step of the series (and its
    # mosaic) shows the whole scene at about the same time. A zone without a
    # scene in the best window takes its scene nearest to it, and the run
    # reports by how much. None: every zone takes its own least cloudy scene.
    max_zone_offset_days: int | None = None
    # With sentinel.zones, the strict alternative to max_zone_offset_days:
    # every step is one acquisition for the whole AOI. Per interval only days
    # on which every zone has a scene (after the cloud checks) are eligible,
    # the least cloudy of them wins; intervals without one get no step.
    single_date: bool = False

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(
                f"sentinel.timeseries.every must be >= 1, got {self.every}"
            )
        if self.max_zone_offset_days is not None and self.max_zone_offset_days < 0:
            raise ValueError(
                "sentinel.timeseries.max_zone_offset_days must be >= 0, got "
                f"{self.max_zone_offset_days}"
            )
        if self.single_date and self.max_zone_offset_days is not None:
            raise ValueError(
                "sentinel.timeseries.single_date and max_zone_offset_days exclude "
                "each other: single_date allows no offset at all"
            )


@dataclass
class Zones:
    """Cover a large AOI with a grid of SR chips (zones) run one after another.

    Without zones only one ``edge_size`` chip around the AOI's center is
    super-resolved (the SR model runs on a whole chip at once, so the chip
    can't grow with the AOI). With zones every chip of the grid gets its own
    sentinel_sr run and its own scene search; ``max_items`` applies per zone.
    Acquire then fetches one DEM over all zones and always builds the DTM in
    ``native`` mode (``dtm.mode`` is ignored: there is no single optical grid
    to upsample onto, the tiles are aligned to the DTM at render time).
    """

    enabled: bool = False
    # Drop grid cells that don't overlap the AOI geometry (corners of an
    # irregular AOI); off: the full rectangle around the AOI.
    clip_to_aoi: bool = True
    # Besides the tiles and their VRT, write the mosaic as one GeoTIFF.
    mosaic: bool = False


@dataclass
class Coreg:
    """AROSICS co-registration of a zoned timeseries (``eo-art-sr`` with
    ``sentinel.zones`` and ``sentinel.timeseries``).

    Runs on the 10 m step mosaics before SR (via stac2cube's consensus
    matcher: many windows per scene, only where both scenes are cloud-free,
    one cubic warp at the end). Steps it can't place reliably are dropped
    and listed in steps.json.
    """

    enabled: bool = True
    # Reference scene: auto (most textured low-cloud step in the middle of the
    # series), first, composite (median of the first 30 days) or YYYY-MM-DD.
    anchor: str = "auto"
    # Matching windows per scene: up to grid_size^2 + 1.
    grid_size: int = 7

    def __post_init__(self) -> None:
        if self.grid_size < 1:
            raise ValueError(
                f"sentinel.coreg.grid_size must be >= 1, got {self.grid_size}"
            )


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
    # Cap on the scenes to super-resolve; ignored when timeseries.enabled,
    # where the interval decides how many scenes there are.
    max_items: int | None = None
    max_cloud_cover: float | None = None
    # AOI-level cloud check on the L2A scene classification (SCL): scenes with
    # more than this percent of AOI pixels in `scl_cloud_classes` are dropped,
    # and the rest are ranked by it. eo:cloud_cover (above) only describes the
    # whole ~100 km tile. None disables the check.
    max_aoi_cloud: float | None = 10.0
    # 8 = cloud medium probability, 9 = cloud high probability; add 3 (cloud
    # shadow), 10 (thin cirrus) or 0/1 (no data/defective) to be stricter.
    scl_cloud_classes: tuple[int, ...] = (8, 9)
    # Which of the (possibly many) super-resolved timesteps to carry forward
    # into the DEM/DTM-SR/align steps. None (default) auto-selects the scene
    # with the lowest eo:cloud_cover; set explicitly to pick by position
    # instead.
    reference_index: int | None = None
    timeseries: Timeseries = field(default_factory=Timeseries)
    zones: Zones = field(default_factory=Zones)
    coreg: Coreg = field(default_factory=Coreg)

    def __post_init__(self) -> None:
        if self.max_aoi_cloud is not None and not 0 <= self.max_aoi_cloud <= 100:
            raise ValueError(
                "sentinel.max_aoi_cloud must be a percentage in [0, 100], "
                f"got {self.max_aoi_cloud}"
            )
        if self.reference_index is not None and self.reference_index < 0:
            raise ValueError(
                f"sentinel.reference_index must be >= 0, got {self.reference_index}"
            )


@dataclass
class Dem:
    # FABDEM is Copernicus GLO-30 with forest canopy and buildings removed --
    # same 30 m posting, but no tree-height noise stippling the surface.
    source: DemSource = DemSource.fabdem
    scale: float = 30.0
    # Local GeoTIFF path, required when source is DemSource.local (e.g. a
    # downloaded 1 m national LiDAR DGM); meaningless otherwise.
    path: str | None = None

    def __post_init__(self) -> None:
        if self.source is DemSource.local:
            if self.path is None:
                raise ValueError("dem.path is required when dem.source is 'local'")
        elif self.path is not None:
            raise ValueError(
                f"dem.path is only valid when dem.source is 'local', got "
                f"source={self.source.value!r} with path={self.path!r}"
            )


@dataclass
class Dtm:
    apply_erosion: bool = False
    erosion_kwargs: dict = field(default_factory=dict)
    radius: int = 8
    eps: float = 1e-2
    band: int | None = None
    mode: DtmMode = DtmMode.guided_upsample


@dataclass
class Align:
    resampling: ResamplingName = ResamplingName.bilinear


@dataclass
class AcquireConfig:
    aoi_path: str = MISSING
    # Only required for Earth Engine DEM sources -- the Sentinel-2 side runs
    # through STAC/Planetary Computer, not Earth Engine, and dem.source=local
    # reads a DEM straight off disk, so neither needs a project at all.
    ee_project: str | None = None
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

    def __post_init__(self) -> None:
        if self.dem.source is not DemSource.local and self.ee_project is None:
            raise ValueError(
                f"ee_project is required when dem.source={self.dem.source.value!r} "
                "(only dem.source=local can skip Earth Engine)"
            )


@dataclass
class SrConfig:
    """The Sentinel SR slice of `AcquireConfig`, for `eo-art-f3d sr`.

    Carries no dem/dtm/ee_project, so an acquire YAML can be reused as-is for
    an SR-only run without satisfying the Earth Engine requirement.
    """

    aoi_path: str = MISSING
    out_dir: str = "out"
    sentinel_sr_dir: str = MISSING
    sentinel: Sentinel = field(default_factory=Sentinel)
