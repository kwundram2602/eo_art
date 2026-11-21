"""Shells out to the isolated `sentinel_sr` project for Sentinel-2 RGBN SR.

`sentinel_sr` carries a heavy, GPU-specific dependency stack (torch, cubo,
mlstac, opensr-model, sen2sr, stac2cube) that would conflict with the rest of
eo_art's dependencies if merged into one environment, so it lives as its own
uv project with its own venv and is invoked as a subprocess rather than
imported directly.
"""

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from vecraspy import get_aoi_center

from eo_art.forge3d_pipes.acquire.schema import Sentinel

MANIFEST_NAME = "manifest.json"
CANDIDATES_NAME = "candidates.json"
# sentinel_sr writes it next to the jobs file of --coreg-jobs.
COREG_REPORT_NAME = "coreg.json"


@dataclass(frozen=True)
class Scene:
    path: Path
    cloud_cover: float | None
    # Acquisition date (YYYY-MM-DD).
    datetime: str | None = None
    # Percent of AOI pixels in the SCL cloud classes (None: check disabled).
    aoi_cloud: float | None = None
    # lr_only runs: the scene's SCL band (for the co-registration cloud mask).
    scl_path: Path | None = None


@dataclass(frozen=True)
class Candidate:
    """A scene sentinel_sr could super-resolve (``--candidates-only``)."""

    id: str
    # Acquisition date (YYYY-MM-DD).
    datetime: str
    cloud_cover: float | None
    aoi_cloud: float | None = None
    # Start date of its timeseries interval.
    bin: str | None = None

    @property
    def cloud(self) -> float | None:
        """The AOI's own cloud share when known, else the tile's."""
        return self.aoi_cloud if self.aoi_cloud is not None else self.cloud_cover


@dataclass(frozen=True)
class CandidateList:
    scenes: list[Candidate]
    # Start dates of all timeseries intervals.
    bins: list[str]


@dataclass(frozen=True)
class Manifest:
    scenes: list[Scene]
    # Timeseries runs: start dates of intervals that had no scene.
    skipped_bins: list[str]


def read_manifest(out_dir: str | Path) -> Manifest:
    """The scene list ``sentinel_sr`` writes next to its GeoTIFFs."""
    manifest = json.loads((Path(out_dir) / MANIFEST_NAME).read_text())
    return Manifest(
        scenes=[
            Scene(
                path=Path(scene["path"]),
                cloud_cover=scene["cloud_cover"],
                datetime=scene.get("datetime"),
                aoi_cloud=scene.get("aoi_cloud"),
                scl_path=Path(scene["scl_path"]) if scene.get("scl_path") else None,
            )
            for scene in manifest["scenes"]
        ],
        skipped_bins=manifest.get("skipped_bins", []),
    )


def read_candidates(out_dir: str | Path) -> CandidateList:
    """The candidate list ``sentinel_sr --candidates-only`` writes."""
    raw = json.loads((Path(out_dir) / CANDIDATES_NAME).read_text())
    return CandidateList(
        scenes=[
            Candidate(
                id=c["id"],
                datetime=c["datetime"],
                cloud_cover=c["cloud_cover"],
                aoi_cloud=c.get("aoi_cloud"),
                bin=c.get("bin"),
            )
            for c in raw["scenes"]
        ],
        bins=raw.get("bins", []),
    )


def read_aoi_center(aoi_path: str | Path) -> tuple[float, float]:
    """Return (lat, lon) for the center of the AOI vector file."""
    aoi_df = gpd.read_file(aoi_path)
    if aoi_df.crs is None:
        aoi_df.crs = "EPSG:4326"
    return get_aoi_center(aoi_df)


def _base_command(
    cfg: Sentinel, sentinel_sr_dir: str | Path, lat: float, lon: float, out_dir: Path
) -> list[str]:
    command = [
        "uv",
        "run",
        "--project",
        str(sentinel_sr_dir),
        "eo-art-sentinel-sr",
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        "--start-date",
        cfg.start_date,
        "--end-date",
        cfg.end_date,
        "--model-path",
        cfg.model_path,
        "--collection",
        cfg.collection,
        "--bands",
        ",".join(cfg.bands),
        "--edge-size",
        str(cfg.edge_size),
        "--resolution",
        str(cfg.resolution),
        "--stac",
        cfg.stac,
        "--out",
        str(out_dir),
    ]
    if cfg.max_cloud_cover is not None:
        command += ["--max-cloud-cover", str(cfg.max_cloud_cover)]
    if cfg.max_aoi_cloud is not None:
        command += [
            "--max-aoi-cloud",
            str(cfg.max_aoi_cloud),
            "--scl-classes",
            ",".join(str(c) for c in cfg.scl_cloud_classes),
        ]
    if cfg.timeseries.enabled:
        command += [
            "--interval",
            cfg.timeseries.interval.value,
            "--every",
            str(cfg.timeseries.every),
        ]
    return command


def run_sentinel_sr(
    cfg: Sentinel,
    sentinel_sr_dir: str | Path,
    lat: float,
    lon: float,
    out_dir: str | Path,
    overwrite: bool = False,
    skip_dates: Sequence[str] = (),
    dates: Sequence[str] = (),
    lr_only: bool = False,
) -> list[Scene]:
    """Run the sentinel_sr subprocess and return the written GeoTIFF paths.

    sentinel_sr keeps GeoTIFFs already in ``out_dir`` (resuming an interrupted
    run) unless ``overwrite``. ``cfg.timeseries`` switches it to one scene per
    interval; intervals containing one of ``skip_dates`` get none. ``dates``
    instead names the acquisitions to super-resolve directly (chosen from
    ``list_sentinel_candidates``). ``lr_only`` fetches the scenes at native
    resolution (plus SCL) without SR.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = _base_command(cfg, sentinel_sr_dir, lat, lon, out_dir)
    # A timeseries has one scene per interval; max_items would cut it short.
    if cfg.max_items is not None and not cfg.timeseries.enabled and not dates:
        command += ["--max-items", str(cfg.max_items)]
    if skip_dates:
        command += ["--skip-dates", ",".join(skip_dates)]
    if dates:
        command += ["--dates", ",".join(dates)]
    if lr_only:
        command.append("--lr-only")
    if overwrite:
        command.append("--overwrite")

    subprocess.run(command, check=True)
    return read_manifest(out_dir).scenes


def _jobs_command(
    sentinel_sr_dir: str | Path, jobs_flag: str, jobs_path: Path
) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(sentinel_sr_dir),
        "eo-art-sentinel-sr",
        jobs_flag,
        str(jobs_path),
    ]


def run_sentinel_coreg(
    cfg: Sentinel,
    sentinel_sr_dir: str | Path,
    jobs: Sequence[dict],
    jobs_path: str | Path,
) -> list[dict]:
    """Co-register LR rasters (``{"src", "dst", "date", "scl"}`` each) as one
    series in sentinel_sr (AROSICS via stac2cube) and return its report: one
    entry per job with ``dst`` None for dropped scenes and the applied
    ``shift_x_px``/``shift_y_px``."""
    jobs_path = Path(jobs_path)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps({"jobs": list(jobs)}, indent=2))
    command = _jobs_command(sentinel_sr_dir, "--coreg-jobs", jobs_path) + [
        "--anchor",
        cfg.coreg.anchor,
        "--grid-size",
        str(cfg.coreg.grid_size),
        "--scl-classes",
        ",".join(str(c) for c in cfg.scl_cloud_classes),
    ]
    subprocess.run(command, check=True)
    return json.loads((jobs_path.parent / COREG_REPORT_NAME).read_text())["scenes"]


def run_sentinel_superres(
    cfg: Sentinel,
    sentinel_sr_dir: str | Path,
    jobs: Sequence[dict],
    jobs_path: str | Path,
    overwrite: bool = False,
) -> None:
    """Super-resolve whole rasters (``{"src", "dst"}`` each) in sentinel_sr
    with overlapping patches and edge padding, so no chip seams appear.
    Existing ``dst`` files are kept unless ``overwrite``."""
    jobs_path = Path(jobs_path)
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps({"jobs": list(jobs)}, indent=2))
    command = _jobs_command(sentinel_sr_dir, "--superres-jobs", jobs_path) + [
        "--model-path",
        cfg.model_path,
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def list_sentinel_candidates(
    cfg: Sentinel,
    sentinel_sr_dir: str | Path,
    lat: float,
    lon: float,
    out_dir: str | Path,
) -> CandidateList:
    """Every scene sentinel_sr could pick for this chip (after duplicate and
    cloud checks, tagged with its interval), without running SR."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = _base_command(cfg, sentinel_sr_dir, lat, lon, out_dir)
    command.append("--candidates-only")
    subprocess.run(command, check=True)
    return read_candidates(out_dir)
