"""What the pipelines left in a project's ``out/``: rasters, renders, run logs."""

from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

from eo_art.app.projects import LOGS_DIRNAME, Project
from eo_art.app.runner import PipelineKind, log_kind
from eo_art.forge3d_pipes.acquire.pipeline import (
    MOSAIC_SUFFIX,
    TIMESERIES_SUFFIX,
    ZONES_SUFFIX,
)
from eo_art.forge3d_pipes.acquire.runs import (
    LATEST_PREFIX,
    latest_run,
    list_runs,
    runs_dir,
)

MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".mp4"}
# Pipeline-internal folders: prep/acquire caches and per-frame dumps.
_SKIP_DIRS = {"_prep", "_acquire", "frames", LOGS_DIRNAME, "runs"}


def media_files(project: Project) -> list[Path]:
    """Rendered images/videos under ``out/``, newest first."""
    out = project.out_dir
    if not out.exists():
        return []
    files = [
        p
        for p in out.rglob("*")
        if p.suffix.lower() in MEDIA_SUFFIXES
        and not _SKIP_DIRS & set(p.relative_to(out).parts[:-1])
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _tifs(root: Path) -> list[Path]:
    """GeoTIFFs of one result folder (a run, or out/ itself): its own, its SR
    zone folders and SR timeseries folders (also per zone).

    SR results and zone mosaics first (newest first), then zone tiles and
    timeseries steps in path (= zone, date) order, then every other raster
    (newest first).
    """
    if not root.is_dir():
        return []

    def order(p: Path) -> tuple[int, float, str]:
        if p.parent != root:
            return 1, 0.0, str(p)
        top = p.name.endswith("_sr.tif") or p.name.endswith(MOSAIC_SUFFIX)
        return (0 if top else 2), -p.stat().st_mtime, ""

    return sorted(
        [
            *root.glob("*.tif"),
            *root.glob(f"*{ZONES_SUFFIX}/*.tif"),
            *root.glob(f"*{TIMESERIES_SUFFIX}/*.tif"),
            *root.glob(f"*{TIMESERIES_SUFFIX}/*/*.tif"),
        ],
        key=order,
    )


def _timeseries_dirs(root: Path) -> list[Path]:
    tops = sorted(p for p in root.glob(f"*{TIMESERIES_SUFFIX}") if p.is_dir())
    return [
        folder
        for top in tops
        for folder in [top, *sorted(p for p in top.iterdir() if p.is_dir())]
    ]


def _zone_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob(f"*{ZONES_SUFFIX}") if p.is_dir())


@dataclass(frozen=True)
class ResultGroup:
    """What one acquire/SR run (or, for older results, out/ itself) wrote."""

    root: Path
    # "acquire" / "sr" from the run folder's name; None for out/ itself.
    kind: str | None
    # The run out/runs/latest-<kind> points at.
    latest: bool = False
    # The app's log of the run (same name), if it was started from the app.
    log: Path | None = None
    rasters: list[Path] = field(default_factory=list)
    zone_dirs: list[Path] = field(default_factory=list)
    timeseries_dirs: list[Path] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.rasters or self.zone_dirs or self.timeseries_dirs)


def _run_kind(run: Path) -> str | None:
    # <YYYYmmdd>-<HHMMSS>-<kind>[-<n>]; the app's acquire_render runs are
    # acquire runs.
    parts = run.name.split("-")
    if len(parts) < 3:
        return None
    return "acquire" if parts[2].startswith("acquire") else parts[2]


def result_groups(project: Project) -> list[ResultGroup]:
    """One group per run folder (newest first), then out/'s own results
    from before runs had folders (only if there are any)."""
    out = project.out_dir
    groups = []
    for run in list_runs(out):
        kind = _run_kind(run)
        log = project.logs_dir / f"{run.name}.log"
        groups.append(
            ResultGroup(
                root=run,
                kind=kind,
                latest=kind is not None and latest_run(out, kind) == run.resolve(),
                log=log if log.exists() else None,
                rasters=[p for p in _tifs(run) if p.parent == run],
                zone_dirs=_zone_dirs(run),
                timeseries_dirs=_timeseries_dirs(run),
            )
        )
    legacy = ResultGroup(
        root=out,
        kind=None,
        rasters=[p for p in _tifs(out) if p.parent == out],
        zone_dirs=_zone_dirs(out) if out.is_dir() else [],
        timeseries_dirs=_timeseries_dirs(out) if out.is_dir() else [],
    )
    if not legacy.empty:
        groups.append(legacy)
    return groups


def project_tifs(project: Project) -> list[Path]:
    """Every result GeoTIFF: the newest runs' files via their stable
    ``out/runs/latest-<kind>/`` paths first (what render configs should point
    at), then each run folder (newest first), then out/'s own older results.
    """
    out = project.out_dir
    base = runs_dir(out)
    links = (
        sorted(p for p in base.glob(f"{LATEST_PREFIX}*") if p.is_dir())
        if base.is_dir()
        else []
    )
    return [
        *(tif for link in links for tif in _tifs(link)),
        *(tif for run in list_runs(out) for tif in _tifs(run)),
        *_tifs(out),
    ]


def timeseries_dirs(project: Project) -> list[Path]:
    """SR timeseries folders of every run and of out/, and with
    ``sentinel.zones`` their per-zone subfolders."""
    out = project.out_dir
    roots = [*list_runs(out), out] if out.is_dir() else []
    return [folder for root in roots for folder in _timeseries_dirs(root)]


def is_timeseries_folder(folder: Path) -> bool:
    """A timeseries folder, or one zone's folder inside it."""
    return folder.name.endswith(TIMESERIES_SUFFIX) or folder.parent.name.endswith(
        TIMESERIES_SUFFIX
    )


def zone_dirs(project: Project) -> list[Path]:
    """``<aoi>_sr_zones/`` folders of every run and of out/."""
    out = project.out_dir
    roots = [*list_runs(out), out] if out.is_dir() else []
    return [folder for root in roots for folder in _zone_dirs(root)]


def zone_mosaic(folder: Path) -> Path:
    """Where ``Export mosaic`` writes the mosaic of a zone folder."""
    stem = folder.name.removesuffix(ZONES_SUFFIX)
    return folder.parent / f"{stem}{MOSAIC_SUFFIX}"


def run_logs(project: Project, kinds: Collection[PipelineKind]) -> list[Path]:
    """Logs of runs of ``kinds``, newest first."""
    logs = project.logs_dir.glob("*.log")
    return sorted((p for p in logs if log_kind(p) in kinds), reverse=True)
