"""Orchestrates sentinel SR -> DEM fetch -> DTM super-resolution -> align."""

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from vecraspy import align_raster_grid, simulate_thermal_erosion, super_resolve_dtm

from eo_art.forge3d_pipes.acquire.schema import (
    AcquireConfig,
    DemSource,
    DtmMode,
    SrConfig,
    Timeseries,
)
from eo_art.forge3d_pipes.acquire.sentinel_bridge import (
    CANDIDATES_NAME,
    MANIFEST_NAME,
    Scene,
    list_sentinel_candidates,
    read_aoi_center,
    read_candidates,
    read_manifest,
    run_sentinel_coreg,
    run_sentinel_sr,
    run_sentinel_superres,
)
from eo_art.forge3d_pipes.acquire.terrain import (
    fetch_local_dem,
    fetch_local_dem_bounds,
    fetch_terrain_dem,
    fetch_terrain_dem_bounds,
)
from eo_art.forge3d_pipes.acquire.zone_steps import (
    Step,
    coordinate_steps,
    describe_step,
)
from eo_art.forge3d_pipes.acquire.zones import (
    ZoneGrid,
    build_vrt,
    export_mosaic,
    merge_first_valid,
    plan_zones,
)

CACHE_DIRNAME = "_acquire"
DTM_FILENAME = "dtm.tif"
OPTICAL_FILENAME = "optical_aligned.tif"
SR_SUFFIX = "_sr.tif"
TIMESERIES_SUFFIX = "_sr_timeseries"
# Name inside the per-config cache dir (already unique per AOI via the hash);
# unchanged so caches written before outputs were named per AOI stay valid.
CACHED_SR_FILENAME = "optical_sr.tif"
ZONES_SUFFIX = "_sr_zones"
MOSAIC_SUFFIX = "_sr_mosaic.tif"
ZONES_CACHE_DIRNAME = "zones"
STEPS_NAME = "steps.json"
# Zoned timeseries: 10 m tiles, step mosaics and their co-registered versions
# inside <aoi>_sr_timeseries/; SR outputs cached by input content.
LR_DIRNAME = "lr"
SR_CACHE_DIRNAME = "sr_mosaics"


@dataclass(frozen=True)
class SrZonesResult:
    # <out_dir>/<aoi>_sr_zones/ with one tile per zone plus the VRT.
    out_dir: Path
    # <aoi>_sr_<zone>.tif, in zone order (rows north to south, then columns);
    # zones without any scene have none.
    tiles: list[Path]
    vrt: Path
    # <out_dir>/<aoi>_sr_mosaic.tif, only with sentinel.zones.mosaic.
    mosaic: Path | None
    # Ids of zones sentinel_sr found no scene for.
    empty_zones: list[str]


@dataclass(frozen=True)
class ZoneStepResult:
    step: Step
    # Zone id -> its 10 m tile in <aoi>_sr_timeseries/lr/<zone>/.
    tiles: dict[str, Path]
    # The step's 10 m mosaic SR ran on (co-registered with sentinel.coreg);
    # None if co-registration dropped every part of the step.
    lr_mosaic: Path | None
    # <aoi>_sr_timeseries/<aoi>_sr_<date>.tif, date = earliest acquisition;
    # None like lr_mosaic.
    sr: Path | None
    # Acquisition date -> applied (x, y) shift in 10 m pixels, None if the
    # matcher dropped it; empty without co-registration.
    shifts: dict[str, tuple[float, float] | None]


@dataclass(frozen=True)
class ZoneTimeseriesResult:
    # <out_dir>/<aoi>_sr_timeseries/
    out_dir: Path
    # The steps this run wrote, in time order.
    steps: list[ZoneStepResult]
    # Start dates of intervals without a scene in any zone.
    skipped_bins: list[str]


@dataclass(frozen=True)
class AcquireResult:
    dtm_path: Path
    # None for a zones run: the tiles stay on their own grids and are aligned
    # to the DTM at render time (overlays.align_to_dem), see `zones`.
    optical_path: Path | None
    zones: SrZonesResult | None = None


@dataclass(frozen=True)
class TimeseriesResult:
    out_dir: Path
    # Every timestep in out_dir, one GeoTIFF per interval, in time order.
    paths: list[Path]
    # Start dates of intervals without any scene.
    skipped_bins: list[str]
    # The timesteps this run wrote (all of `paths` unless filling gaps).
    added: list[Path] = field(default_factory=list)


def _cache_key(cfg: AcquireConfig | SrConfig) -> str:
    # out_dir says where to store results, not what to compute -- excluded so
    # moving the output location doesn't invalidate an otherwise-identical
    # cached result (or, worse, silently redo an expensive GPU+network run).
    payload = {k: v for k, v in asdict(cfg).items() if k != "out_dir"}
    # Acquire ignores the timeseries block, and a disabled one changes nothing:
    # leaving it out keeps caches written before the field existed valid.
    if isinstance(cfg, AcquireConfig) or not cfg.sentinel.timeseries.enabled:
        del payload["sentinel"]["timeseries"]
    else:
        # Ignored by a timeseries run (see run_sentinel_sr).
        del payload["sentinel"]["max_items"]
        # Only steers which cached candidates/dates are combined; candidates
        # and SR'd dates stay valid when it changes.
        del payload["sentinel"]["timeseries"]["max_zone_offset_days"]
        del payload["sentinel"]["timeseries"]["single_date"]
    # Co-registration runs on the fetched scenes and never changes them.
    del payload["sentinel"]["coreg"]
    # Same for disabled zones, which leave the single-chip run unchanged.
    if not cfg.sentinel.zones.enabled:
        del payload["sentinel"]["zones"]
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _results_dir(out_dir: Path, run_dir: str | Path | None) -> Path:
    """Where a run's results go: its run folder, else ``out_dir`` itself."""
    return Path(run_dir) if run_dir is not None else out_dir


def _select_reference(scenes: list[Scene], reference_index: int | None) -> Path:
    """Pick the scene to carry forward: by position, or lowest cloud cover."""
    if reference_index is not None:
        if not 0 <= reference_index < len(scenes):
            raise ValueError(
                f"sentinel.reference_index={reference_index} out of range for "
                f"{len(scenes)} written scene(s)"
            )
        return scenes[reference_index].path

    # The AOI's own SCL cloud share beats the tile-wide eo:cloud_cover.
    # Scenes with neither (some collections lack the eo extension) sort last
    # rather than winning by default.
    def cloud(scene: Scene) -> float | None:
        return scene.aoi_cloud if scene.aoi_cloud is not None else scene.cloud_cover

    return min(
        scenes,
        key=lambda scene: (cloud(scene) is None, cloud(scene) or 0.0),
    ).path


def run_acquire(
    cfg: AcquireConfig,
    out_dir: str | Path,
    use_cache: bool = True,
    run_dir: str | Path | None = None,
) -> AcquireResult:
    """Run the acquire chain, caching the final DTM/optical pair by config hash.

    The cache lives under ``<out_dir>/_acquire/<hash>`` (so re-running with an
    unchanged config skips recomputation) but the returned paths are always
    the stable ``<out_dir>/dtm.tif`` / ``<out_dir>/optical_aligned.tif`` --
    matching how the render pipeline's own ``_prep`` cache stays internal
    while its final artifacts land directly in the given out_dir.

    With ``run_dir`` (see ``runs``) the results land there instead, while
    the cache stays shared under ``out_dir``.
    """
    if cfg.sentinel.zones.enabled:
        return _run_acquire_zones(cfg, Path(out_dir), use_cache, run_dir)
    out_dir = Path(out_dir)
    dest = _results_dir(out_dir, run_dir)
    cache_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    cached_dtm = cache_dir / DTM_FILENAME
    cached_optical = cache_dir / OPTICAL_FILENAME
    final_dtm = dest / DTM_FILENAME
    final_optical = dest / OPTICAL_FILENAME

    if not (use_cache and cached_dtm.exists() and cached_optical.exists()):
        cache_dir.mkdir(parents=True, exist_ok=True)

        lat, lon = read_aoi_center(cfg.aoi_path)
        # Acquire needs one reference scene, never a timeseries.
        sentinel = replace(cfg.sentinel, timeseries=Timeseries())
        written = run_sentinel_sr(
            sentinel,
            cfg.sentinel_sr_dir,
            lat,
            lon,
            cache_dir / "sentinel",
            overwrite=not use_cache,
        )
        if not written:
            raise RuntimeError(
                f"sentinel_sr wrote no scenes for aoi={cfg.aoi_path!r} "
                f"in [{cfg.sentinel.start_date}, {cfg.sentinel.end_date}]"
            )
        reference_tif = _select_reference(written, cfg.sentinel.reference_index)

        if cfg.dem.source is DemSource.local:
            # Dem.__post_init__ requires path to be set for a local source.
            assert cfg.dem.path is not None
            raw_dem = fetch_local_dem(reference_tif, cache_dir, path=cfg.dem.path)
        else:
            # AcquireConfig.__post_init__ requires ee_project for any
            # non-local source.
            assert cfg.ee_project is not None
            raw_dem = fetch_terrain_dem(
                reference_tif,
                cache_dir,
                source=cfg.dem.source,
                ee_project=cfg.ee_project,
                scale=cfg.dem.scale,
            )

        if cfg.dtm.mode is DtmMode.native:
            # DEM is master: no guided upsampling onto the optical grid, so
            # real elevation resolution is preserved rather than invented.
            if cfg.dtm.apply_erosion:
                simulate_thermal_erosion(
                    raw_dem, cached_dtm, **(cfg.dtm.erosion_kwargs or {})
                )
            else:
                shutil.copy2(raw_dem, cached_dtm)
        else:
            super_resolve_dtm(
                raw_dem,
                reference_tif,
                cached_dtm,
                band=cfg.dtm.band,
                radius=cfg.dtm.radius,
                eps=cfg.dtm.eps,
                apply_erosion=cfg.dtm.apply_erosion,
                erosion_kwargs=cfg.dtm.erosion_kwargs or None,
            )

        # In guided_upsample mode cached_dtm is already on the optical grid,
        # so this is effectively an identity; in native mode it does the
        # real work of aligning the optical raster onto the DEM's grid.
        align_raster_grid(
            cached_dtm,
            reference_tif,
            cached_optical,
            resampling=cfg.align.resampling.value,
        )

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_dtm, final_dtm)
    shutil.copy2(cached_optical, final_optical)
    return AcquireResult(dtm_path=final_dtm, optical_path=final_optical)


def sr_filename(aoi_path: str | Path) -> str:
    """``aoi/valley.gpkg`` -> ``valley_sr.tif``, so SR runs for different
    AOIs sharing one out_dir don't overwrite each other's result."""
    return f"{Path(aoi_path).stem}{SR_SUFFIX}"


def run_sr(
    cfg: SrConfig,
    out_dir: str | Path,
    use_cache: bool = True,
    run_dir: str | Path | None = None,
) -> Path:
    """Run only the Sentinel SR step and return ``<out_dir>/<aoi>_sr.tif``.

    The reference scene (lowest cloud cover, or ``sentinel.reference_index``)
    is cached under ``<out_dir>/_acquire/<hash>`` like ``run_acquire``'s
    outputs; the full set of SR timesteps stays next to it in ``sentinel/``.
    For ``sentinel.timeseries.enabled`` use ``run_sr_timeseries``.
    """
    if cfg.sentinel.timeseries.enabled:
        raise ValueError("sentinel.timeseries is enabled; use run_sr_timeseries")
    if cfg.sentinel.zones.enabled:
        raise ValueError("sentinel.zones is enabled; use run_sr_zones")
    out_dir = Path(out_dir)
    dest = _results_dir(out_dir, run_dir)
    cache_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    cached_sr = cache_dir / CACHED_SR_FILENAME
    final_sr = dest / sr_filename(cfg.aoi_path)

    if not (use_cache and cached_sr.exists()):
        cache_dir.mkdir(parents=True, exist_ok=True)

        lat, lon = read_aoi_center(cfg.aoi_path)
        written = run_sentinel_sr(
            cfg.sentinel,
            cfg.sentinel_sr_dir,
            lat,
            lon,
            cache_dir / "sentinel",
            overwrite=not use_cache,
        )
        if not written:
            raise RuntimeError(
                f"sentinel_sr wrote no scenes for aoi={cfg.aoi_path!r} "
                f"in [{cfg.sentinel.start_date}, {cfg.sentinel.end_date}]"
            )
        reference_tif = _select_reference(written, cfg.sentinel.reference_index)
        shutil.copy2(reference_tif, cached_sr)

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_sr, final_sr)
    return final_sr


def timeseries_dirname(aoi_path: str | Path) -> str:
    """``aoi/valley.gpkg`` -> ``valley_sr_timeseries``."""
    return f"{Path(aoi_path).stem}{TIMESERIES_SUFFIX}"


def timeseries_frames(aoi_path: str | Path, folder: Path) -> dict[str, Path]:
    """Date -> timestep GeoTIFF (``<aoi>_sr_<YYYY-MM-DD>.tif``) in ``folder``.

    Other rasters saved there (e.g. ``..._indices.tif``) aren't timesteps.
    """
    pattern = re.compile(
        rf"{re.escape(Path(aoi_path).stem)}{SR_SUFFIX.removesuffix('.tif')}"
        r"_(\d{4}-\d{2}-\d{2})"
    )
    frames = {}
    for path in folder.glob("*.tif"):
        match = pattern.fullmatch(path.stem)
        if match:
            frames[match.group(1)] = path
    return dict(sorted(frames.items()))


def run_sr_timeseries(
    cfg: SrConfig,
    out_dir: str | Path,
    use_cache: bool = True,
    run_dir: str | Path | None = None,
) -> TimeseriesResult:
    """SR one scene per ``sentinel.timeseries`` interval into
    ``<out_dir>/<aoi>_sr_timeseries/<aoi>_sr_<YYYY-MM-DD>.tif``.

    The SR scenes and sentinel_sr's manifest are cached under
    ``<out_dir>/_acquire/<hash>/sentinel``; a run that died halfway left no
    manifest, so the next run reruns sentinel_sr, which keeps the scenes it
    already wrote and only processes the rest.

    With ``timeseries.fill_gaps`` the output folder is the source of truth:
    sentinel_sr always runs, but skips intervals that already hold a
    timestep there, and new timesteps are added next to the existing ones.
    """
    if not cfg.sentinel.timeseries.enabled:
        raise ValueError("sentinel.timeseries is disabled; use run_sr")
    if cfg.sentinel.zones.enabled:
        raise ValueError("sentinel.zones is enabled; use run_sr_timeseries_zones")
    out_dir = Path(out_dir)
    result = _timeseries(
        cfg,
        lambda: read_aoi_center(cfg.aoi_path),
        out_dir / CACHE_DIRNAME / _cache_key(cfg) / "sentinel",
        _results_dir(out_dir, run_dir) / timeseries_dirname(cfg.aoi_path),
        use_cache,
    )
    if result is None:
        raise RuntimeError(
            f"sentinel_sr wrote no scenes for aoi={cfg.aoi_path!r} "
            f"in [{cfg.sentinel.start_date}, {cfg.sentinel.end_date}]"
        )
    return result


def _timeseries(
    cfg: SrConfig,
    center: Callable[[], tuple[float, float]],
    sentinel_dir: Path,
    target: Path,
    use_cache: bool,
) -> TimeseriesResult | None:
    """One timeseries (whole AOI or one zone) into ``target``; None if
    sentinel_sr found no scene and ``target`` holds none either."""
    timeseries = cfg.sentinel.timeseries
    existing = timeseries_frames(cfg.aoi_path, target) if target.exists() else {}

    # Filling gaps always searches again: an open interval may have a usable
    # scene now (longer date range, relaxed cloud limits, new acquisitions).
    if timeseries.fill_gaps or not (
        use_cache and (sentinel_dir / MANIFEST_NAME).exists()
    ):
        lat, lon = center()
        run_sentinel_sr(
            cfg.sentinel,
            cfg.sentinel_sr_dir,
            lat,
            lon,
            sentinel_dir,
            overwrite=not use_cache,
            skip_dates=list(existing) if timeseries.fill_gaps else (),
        )
    manifest = read_manifest(sentinel_dir)
    if not manifest.scenes and not (timeseries.fill_gaps and existing):
        return None

    stem = Path(cfg.aoi_path).stem
    if not timeseries.fill_gaps:
        # Drop dates from an earlier run with other settings, so the folder
        # holds exactly this timeseries.
        for stale in existing.values():
            stale.unlink()
    target.mkdir(parents=True, exist_ok=True)
    added = []
    for scene in sorted(manifest.scenes, key=lambda s: s.datetime or ""):
        dst = target / f"{stem}_sr_{scene.datetime}.tif"
        shutil.copy2(scene.path, dst)
        added.append(dst)
    return TimeseriesResult(
        out_dir=target,
        paths=list(timeseries_frames(cfg.aoi_path, target).values()),
        skipped_bins=manifest.skipped_bins,
        added=added,
    )


def zones_dirname(aoi_path: str | Path) -> str:
    """``aoi/valley.gpkg`` -> ``valley_sr_zones``."""
    return f"{Path(aoi_path).stem}{ZONES_SUFFIX}"


def zone_tile_name(aoi_path: str | Path, zone_id: str) -> str:
    """``aoi/valley.gpkg``, ``z00_01`` -> ``valley_sr_z00_01.tif``."""
    return f"{Path(aoi_path).stem}{SR_SUFFIX.removesuffix('.tif')}_{zone_id}.tif"


def _plan(cfg: AcquireConfig | SrConfig) -> ZoneGrid:
    sentinel = cfg.sentinel
    grid = plan_zones(
        cfg.aoi_path,
        sentinel.edge_size,
        sentinel.resolution,
        clip_to_aoi=sentinel.zones.clip_to_aoi,
    )
    print(
        f"{len(grid.zones)} zone(s) of {sentinel.edge_size}x{sentinel.edge_size} px "
        f"({grid.tile_m:g} m) over {cfg.aoi_path} in EPSG:{grid.epsg}, "
        "processed one after another"
    )
    return grid


def _zone_references(
    cfg: SrConfig, grid: ZoneGrid, cache_dir: Path, use_cache: bool
) -> tuple[dict[str, Path], list[str]]:
    """Run sentinel_sr zone by zone and cache each zone's reference scene.

    Returns (zone id -> cached reference GeoTIFF, ids of empty zones). A
    zone whose reference is cached is skipped, so an interrupted run resumes
    with the next open zone. Fails only if every zone is empty.
    """
    references: dict[str, Path] = {}
    empty: list[str] = []
    n = len(grid.zones)
    for k, zone in enumerate(grid.zones, start=1):
        zone_dir = cache_dir / ZONES_CACHE_DIRNAME / zone.id
        cached = zone_dir / CACHED_SR_FILENAME
        sentinel_dir = zone_dir / "sentinel"
        print(f"[zone {k}/{n}] {zone.id} (lat {zone.lat:.5f}, lon {zone.lon:.5f})")
        if use_cache and cached.exists():
            references[zone.id] = cached
            continue
        if use_cache and (sentinel_dir / MANIFEST_NAME).exists():
            # Finished before, without a usable scene (or before caching).
            scenes = read_manifest(sentinel_dir).scenes
        else:
            scenes = run_sentinel_sr(
                cfg.sentinel,
                cfg.sentinel_sr_dir,
                zone.lat,
                zone.lon,
                sentinel_dir,
                overwrite=not use_cache,
            )
        if not scenes:
            print(f"[zone {k}/{n}] {zone.id}: no scene, left empty")
            empty.append(zone.id)
            continue
        zone_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_select_reference(scenes, cfg.sentinel.reference_index), cached)
        references[zone.id] = cached

    if not references:
        raise RuntimeError(
            f"sentinel_sr wrote no scenes for any of the {n} zone(s) of "
            f"aoi={cfg.aoi_path!r} in "
            f"[{cfg.sentinel.start_date}, {cfg.sentinel.end_date}]"
        )
    return references, empty


def _write_zone_outputs(
    cfg: SrConfig,
    grid: ZoneGrid,
    references: dict[str, Path],
    empty: list[str],
    out_dir: Path,
) -> SrZonesResult:
    """Copy the zone references to ``<out_dir>/<aoi>_sr_zones/`` and write
    their VRT (plus the mosaic GeoTIFF with ``sentinel.zones.mosaic``)."""
    stem = Path(cfg.aoi_path).stem
    target = out_dir / zones_dirname(cfg.aoi_path)
    if target.exists():
        # Tiles of an earlier run with another grid or other settings.
        for stale in target.glob(f"{stem}{SR_SUFFIX.removesuffix('.tif')}_z*.tif"):
            stale.unlink()
    target.mkdir(parents=True, exist_ok=True)
    tiles = []
    for zone in grid.zones:
        if zone.id in references:
            dst = target / zone_tile_name(cfg.aoi_path, zone.id)
            shutil.copy2(references[zone.id], dst)
            tiles.append(dst)
    vrt = build_vrt(tiles, zones_vrt(cfg.aoi_path, out_dir))
    mosaic = (
        export_mosaic(vrt, zones_mosaic_path(cfg.aoi_path, out_dir))
        if cfg.sentinel.zones.mosaic
        else None
    )
    return SrZonesResult(
        out_dir=target, tiles=tiles, vrt=vrt, mosaic=mosaic, empty_zones=empty
    )


def run_sr_zones(
    cfg: SrConfig,
    out_dir: str | Path,
    use_cache: bool = True,
    run_dir: str | Path | None = None,
) -> SrZonesResult:
    """SR every zone of ``sentinel.zones`` one after another.

    Each zone gets its own sentinel_sr run (``max_items`` per zone) and its
    own reference scene (lowest cloud, or ``sentinel.reference_index``),
    written to ``<out_dir>/<aoi>_sr_zones/<aoi>_sr_<zone>.tif`` next to a VRT
    of all tiles. Zone caches live under ``<out_dir>/_acquire/<hash>/zones``
    and are shared with ``run_acquire`` for the same sentinel settings.
    """
    if not cfg.sentinel.zones.enabled:
        raise ValueError("sentinel.zones is disabled; use run_sr")
    if cfg.sentinel.timeseries.enabled:
        raise ValueError("sentinel.timeseries is enabled; use run_sr_timeseries_zones")
    out_dir = Path(out_dir)
    grid = _plan(cfg)
    cache_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    references, empty = _zone_references(cfg, grid, cache_dir, use_cache)
    return _write_zone_outputs(
        cfg, grid, references, empty, _results_dir(out_dir, run_dir)
    )


def zones_vrt(aoi_path: str | Path, out_dir: str | Path) -> Path:
    """``<out_dir>/<aoi>_sr_zones/<aoi>_sr_zones.vrt``."""
    dirname = zones_dirname(aoi_path)
    return Path(out_dir) / dirname / f"{dirname}.vrt"


def zones_mosaic_path(aoi_path: str | Path, out_dir: str | Path) -> Path:
    """``<out_dir>/<aoi>_sr_mosaic.tif``."""
    return Path(out_dir) / f"{Path(aoi_path).stem}{MOSAIC_SUFFIX}"


def export_zones_mosaic(aoi_path: str | Path, out_dir: str | Path) -> Path:
    """Write the VRT of an earlier zones run as ``<aoi>_sr_mosaic.tif``."""
    vrt = zones_vrt(aoi_path, out_dir)
    if not vrt.exists():
        raise FileNotFoundError(
            f"no zones VRT at {vrt}; run the SR with sentinel.zones first"
        )
    return export_mosaic(vrt, zones_mosaic_path(aoi_path, out_dir))


def _zone_candidates(
    cfg: SrConfig, grid: ZoneGrid, cache_dir: Path, use_cache: bool
) -> tuple[dict[str, list], list[str]]:
    """Zone id -> its candidates, and all interval start dates (pass 1)."""
    candidates: dict[str, list] = {}
    bins: list[str] = []
    n = len(grid.zones)
    for k, zone in enumerate(grid.zones, start=1):
        zone_dir = cache_dir / zone.id
        print(f"[zone {k}/{n}] {zone.id}: candidates")
        # Filling gaps searches again: open intervals may have scenes now.
        if cfg.sentinel.timeseries.fill_gaps or not (
            use_cache and (zone_dir / CANDIDATES_NAME).exists()
        ):
            listed = list_sentinel_candidates(
                cfg.sentinel, cfg.sentinel_sr_dir, zone.lat, zone.lon, zone_dir
            )
        else:
            listed = read_candidates(zone_dir)
        candidates[zone.id] = listed.scenes
        bins = bins or listed.bins
    return candidates, bins


def _reset_zone_timeseries(target: Path, stem: str) -> None:
    """Drop the steps of an earlier run, so the folder holds exactly this one."""
    if not target.exists():
        return
    for folder in [target, *(p for p in target.iterdir() if p.is_dir())]:
        for frame in timeseries_frames(stem, folder).values():
            frame.unlink()
    for vrt in target.glob(f"{stem}{SR_SUFFIX.removesuffix('.tif')}_*.vrt"):
        vrt.unlink()
    shutil.rmtree(target / LR_DIRNAME, ignore_errors=True)
    (target / STEPS_NAME).unlink(missing_ok=True)


@dataclass
class _LrPart:
    """The 10 m mosaic of one acquisition date of a step, on the full grid."""

    date: str
    src: Path
    scl: Path | None
    # Set by co-registration: the warped copy, or None if dropped.
    corrected: Path | None = None
    shift: tuple[float, float] | None = None
    dropped: bool = False

    def record(self) -> dict:
        return {
            "date": self.date,
            "src": self.src.name,
            "scl": self.scl.name if self.scl else None,
        }


@dataclass
class _StepWork:
    bin: str
    date: str
    parts: list[_LrPart]
    # New steps carry their Step and tiles, kept ones (fill_gaps) their record.
    step: Step | None = None
    tiles: dict[str, Path] = field(default_factory=dict)
    record: dict | None = None
    lr_mosaic: Path | None = None
    sr: Path | None = None


def _lr_parts(
    step: Step,
    fetched: dict[str, dict[str, Scene]],
    grid: ZoneGrid,
    lr_dir: Path,
    stem: str,
) -> tuple[list[_LrPart], dict[str, Path]]:
    """10 m mosaics of a step, one per acquisition date, all on the full zone
    grid (zones of other dates are NaN), plus zone id -> its 10 m tile."""
    by_date: dict[str, list[str]] = {}
    for zone_id, pick in step.picks.items():
        by_date.setdefault(pick.date, []).append(zone_id)
    multi = len(by_date) > 1
    parts, tiles = [], {}
    for date, zone_ids in sorted(by_date.items()):
        lr_tiles, scl_tiles = [], []
        for zone_id in zone_ids:
            scene = fetched.get(zone_id, {}).get(date)
            if scene is None:
                print(f"  {step.bin}: {zone_id} has no 10 m scene for {date}")
                continue
            dst = lr_dir / zone_id / f"{stem}_lr_{date}.tif"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scene.path, dst)
            tiles[zone_id] = dst
            lr_tiles.append(dst)
            if scene.scl_path is not None:
                scl_dst = dst.with_name(f"{dst.stem}_scl.tif")
                shutil.copy2(scene.scl_path, scl_dst)
                scl_tiles.append(scl_dst)
        if not lr_tiles:
            continue
        # A date can recur as a relaxed pick of another step, with other zones.
        name = f"{stem}_lr_{step.date}_part_{date}" if multi else f"{stem}_lr_{date}"
        src = export_mosaic(
            build_vrt(lr_tiles, lr_dir / f"{name}.vrt", bounds=grid.bounds),
            lr_dir / f"{name}.tif",
        )
        scl = (
            export_mosaic(
                build_vrt(scl_tiles, lr_dir / f"{name}_scl.vrt", bounds=grid.bounds),
                lr_dir / f"{name}_scl.tif",
            )
            if len(scl_tiles) == len(lr_tiles)
            else None
        )
        parts.append(_LrPart(date=date, src=src, scl=scl))
    return parts, tiles


def _kept_work(kept: list[dict], lr_dir: Path) -> list[_StepWork]:
    """Steps of an earlier run (fill_gaps) whose 10 m parts are still there,
    so they are co-registered and super-resolved together with the new ones."""
    work = []
    for record in kept:
        parts = [
            _LrPart(
                date=part["date"],
                src=lr_dir / part["src"],
                scl=lr_dir / part["scl"] if part.get("scl") else None,
            )
            for part in record.get("lr_parts", [])
        ]
        if parts and all(p.src.exists() for p in parts):
            work.append(_StepWork(record["bin"], record["date"], parts, record=record))
        else:
            print(
                f"  {record['bin']}: no 10 m mosaic from the earlier run, its SR "
                "step is kept as is (not co-registered)"
            )
    return work


def _coregister_parts(cfg: SrConfig, parts: list[_LrPart], lr_dir: Path) -> None:
    """AROSICS over all parts of the series (sentinel_sr, stac2cube engine);
    fills each part's ``corrected``/``shift`` or marks it ``dropped``."""
    jobs = [
        {
            "src": str(part.src),
            "dst": str(part.src.with_name(f"{part.src.stem}_cr.tif")),
            "date": part.date,
            "scl": str(part.scl) if part.scl else None,
        }
        for part in parts
    ]
    print(
        f"Co-registration: {len(parts)} 10 m mosaic(s), "
        f"anchor {cfg.sentinel.coreg.anchor}"
    )
    report = run_sentinel_coreg(
        cfg.sentinel, cfg.sentinel_sr_dir, jobs, lr_dir / "coreg_jobs.json"
    )
    by_src = {entry["src"]: entry for entry in report}
    for part in parts:
        entry = by_src.get(str(part.src))
        if entry is None or entry["dst"] is None:
            part.dropped = True
            print(f"  {part.date}: dropped by co-registration (see the log above)")
            continue
        part.corrected = Path(entry["dst"])
        part.shift = (entry["shift_x_px"], entry["shift_y_px"])


def _step_input(work: _StepWork, lr_dir: Path, stem: str) -> Path | None:
    """The 10 m raster SR runs on: the (co-registered) part, or the parts
    merged first-valid; None if every part was dropped."""
    inputs = [p.corrected or p.src for p in work.parts if not p.dropped]
    if not inputs:
        return None
    if len(inputs) == 1:
        return inputs[0]
    return merge_first_valid(inputs, lr_dir / f"{stem}_lr_{work.date}_merged.tif")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _superres_steps(
    cfg: SrConfig,
    work: list[_StepWork],
    sr_cache: Path,
    target: Path,
    stem: str,
    use_cache: bool,
) -> None:
    """SR of every step's 10 m mosaic as one image (overlapping patches, no
    chip seams). Results are cached by input content, so an unchanged mosaic
    (same scenes, same shift) is never super-resolved twice."""
    jobs, todo = [], []
    for w in work:
        if w.lr_mosaic is None:
            continue
        cached = sr_cache / f"{stem}_sr_{w.date}_{_digest(w.lr_mosaic)}.tif"
        jobs.append({"src": str(w.lr_mosaic), "dst": str(cached)})
        todo.append((w, cached))
    if not jobs:
        return
    run_sentinel_superres(
        cfg.sentinel,
        cfg.sentinel_sr_dir,
        jobs,
        sr_cache / "superres_jobs.json",
        overwrite=not use_cache,
    )
    for w, cached in todo:
        w.sr = target / f"{stem}_sr_{w.date}.tif"
        shutil.copy2(cached, w.sr)


def _work_record(w: _StepWork, lr_dir: Path) -> dict:
    if w.record is not None:
        record = dict(w.record)
    else:
        assert w.step is not None
        step = w.step
        record = {
            "bin": step.bin,
            "date": step.date,
            "window": list(step.window) if step.window else None,
            "spread_days": step.spread_days,
            "zones": {
                zone: {
                    "date": pick.date,
                    "cloud": pick.cloud,
                    "outside_days": pick.outside_days,
                }
                for zone, pick in step.picks.items()
            },
            "empty_zones": step.empty_zones,
        }
    record.pop("vrt", None)
    record.pop("mosaic", None)
    record["lr_parts"] = [p.record() for p in w.parts]
    record["coreg"] = {
        p.date: (list(p.shift) if p.shift else None)
        for p in w.parts
        if p.shift is not None or p.dropped
    }
    record["lr_mosaic"] = (
        w.lr_mosaic.relative_to(lr_dir.parent).as_posix() if w.lr_mosaic else None
    )
    record["sr"] = w.sr.name if w.sr else record.get("sr")
    return record


def run_sr_timeseries_zones(
    cfg: SrConfig,
    out_dir: str | Path,
    use_cache: bool = True,
    run_dir: str | Path | None = None,
) -> ZoneTimeseriesResult:
    """A timeseries over all zones with one shared step per interval.

    1. Every zone's candidate scenes are listed (no pixels). Per interval
       the zones' acquisitions are chosen together: one day for all zones
       (``timeseries.single_date``), within ``max_zone_offset_days`` of each
       other, or each zone its own (see ``zone_steps``).
    2. The chosen dates are fetched at 10 m (plus SCL), zone by zone.
    3. Per step and acquisition date the zone tiles become one 10 m mosaic
       on the full zone grid (``lr/``).
    4. With ``sentinel.coreg`` (and at least two mosaics) all mosaics are
       co-registered as one series with AROSICS, at 10 m, before any SR.
    5. Each step's mosaic is super-resolved as one image with overlapping
       patches, so zone borders leave no seam:
       ``<aoi>_sr_timeseries/<aoi>_sr_<date>.tif`` (date = the step's earliest
       acquisition), a regular timeseries folder.

    ``steps.json`` records dates, windows, offsets, the 10 m parts and the
    applied shifts of every step. ``max_items`` is ignored.
    """
    timeseries = cfg.sentinel.timeseries
    if not timeseries.enabled:
        raise ValueError("sentinel.timeseries is disabled; use run_sr_zones")
    if not cfg.sentinel.zones.enabled:
        raise ValueError("sentinel.zones is disabled; use run_sr_timeseries")
    out_dir = Path(out_dir)
    stem = Path(cfg.aoi_path).stem
    grid = _plan(cfg)
    config_cache = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    cache_dir = config_cache / ZONES_CACHE_DIRNAME
    target = _results_dir(out_dir, run_dir) / timeseries_dirname(cfg.aoi_path)
    lr_dir = target / LR_DIRNAME
    steps_path = target / STEPS_NAME

    candidates, bins = _zone_candidates(cfg, grid, cache_dir, use_cache)
    kept: list[dict] = []
    if timeseries.fill_gaps and steps_path.exists():
        kept = json.loads(steps_path.read_text())["steps"]
        filled = {record["bin"] for record in kept}
        bins = [b for b in bins if b not in filled]
        print(f"Timeseries: {len(filled)} step(s) already done, {len(bins)} open")
    steps, skipped = coordinate_steps(
        candidates,
        bins,
        timeseries.max_zone_offset_days,
        single_date=timeseries.single_date,
    )
    if not steps and not kept:
        raise RuntimeError(
            f"sentinel_sr found no scenes for any of the {len(grid.zones)} "
            f"zone(s) of aoi={cfg.aoi_path!r} in "
            f"[{cfg.sentinel.start_date}, {cfg.sentinel.end_date}]"
            + (" on a day shared by all zones" if timeseries.single_date else "")
        )
    print(f"Timeseries: {len(steps)} step(s), {len(skipped)} empty interval(s)")
    for step in steps:
        print("  " + describe_step(step, timeseries.max_zone_offset_days))

    # Pass 2: fetch exactly the chosen dates at 10 m, zone by zone.
    fetched: dict[str, dict[str, Scene]] = {}
    n = len(grid.zones)
    for k, zone in enumerate(grid.zones, start=1):
        dates = sorted({s.picks[zone.id].date for s in steps if zone.id in s.picks})
        if not dates:
            continue
        print(f"[zone {k}/{n}] {zone.id}: 10 m scenes of {len(dates)} date(s)")
        scenes = run_sentinel_sr(
            cfg.sentinel,
            cfg.sentinel_sr_dir,
            zone.lat,
            zone.lon,
            cache_dir / zone.id / LR_DIRNAME,
            overwrite=not use_cache,
            dates=dates,
            lr_only=True,
        )
        fetched[zone.id] = {s.datetime: s for s in scenes if s.datetime}

    if not timeseries.fill_gaps:
        _reset_zone_timeseries(target, stem)
    target.mkdir(parents=True, exist_ok=True)
    work = _kept_work(kept, lr_dir)
    kept_bins = {w.bin for w in work}
    for step in steps:
        parts, tiles = _lr_parts(step, fetched, grid, lr_dir, stem)
        if parts:
            work.append(_StepWork(step.bin, step.date, parts, step=step, tiles=tiles))
    work.sort(key=lambda w: w.bin)

    all_parts = [p for w in work for p in w.parts]
    if cfg.sentinel.coreg.enabled and len(all_parts) >= 2:
        _coregister_parts(cfg, all_parts, lr_dir)
    for w in work:
        w.lr_mosaic = _step_input(w, lr_dir, stem)
    _superres_steps(cfg, work, config_cache / SR_CACHE_DIRNAME, target, stem, use_cache)

    records = {r["bin"]: r for r in kept}
    records.update({w.bin: _work_record(w, lr_dir) for w in work})
    steps_path.write_text(
        json.dumps(
            {
                "max_zone_offset_days": timeseries.max_zone_offset_days,
                "single_date": timeseries.single_date,
                "coreg": asdict(cfg.sentinel.coreg),
                "steps": [records[b] for b in sorted(records)],
                "skipped_bins": skipped,
            },
            indent=2,
        )
    )
    results = [
        ZoneStepResult(
            step=w.step,
            tiles=w.tiles,
            lr_mosaic=w.lr_mosaic,
            sr=w.sr,
            shifts={
                p.date: p.shift for p in w.parts if p.shift is not None or p.dropped
            },
        )
        for w in work
        if w.step is not None and w.bin not in kept_bins
    ]
    return ZoneTimeseriesResult(out_dir=target, steps=results, skipped_bins=skipped)


def _run_acquire_zones(
    cfg: AcquireConfig, out_dir: Path, use_cache: bool, run_dir: str | Path | None
) -> AcquireResult:
    """Acquire for ``sentinel.zones``: SR per zone, one DEM over all zones.

    Nothing is aligned here: the tiles keep their own grids and are snapped
    onto the DTM at render time (``overlays.align_to_dem``). The zone SR
    cache is keyed by the sentinel settings alone, so it is shared with
    ``run_sr_zones`` and survives changes to the dem/dtm blocks.
    """
    # Acquire needs one reference scene per zone, never a timeseries.
    sr_cfg = SrConfig(
        aoi_path=cfg.aoi_path,
        out_dir=cfg.out_dir,
        sentinel_sr_dir=cfg.sentinel_sr_dir,
        sentinel=replace(cfg.sentinel, timeseries=Timeseries()),
    )
    grid = _plan(cfg)
    references, empty = _zone_references(
        sr_cfg, grid, out_dir / CACHE_DIRNAME / _cache_key(sr_cfg), use_cache
    )

    cache_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    cached_dtm = cache_dir / DTM_FILENAME
    if not (use_cache and cached_dtm.exists()):
        cache_dir.mkdir(parents=True, exist_ok=True)
        crs = f"EPSG:{grid.epsg}"
        if cfg.dem.source is DemSource.local:
            assert cfg.dem.path is not None
            raw_dem = fetch_local_dem_bounds(
                grid.bounds, crs, cache_dir, path=cfg.dem.path, label="the zones"
            )
        else:
            assert cfg.ee_project is not None
            raw_dem = fetch_terrain_dem_bounds(
                grid.bounds,
                crs,
                cache_dir,
                source=cfg.dem.source,
                ee_project=cfg.ee_project,
                scale=cfg.dem.scale,
            )
        # Always native with zones: guided_upsample needs one optical grid
        # to warp the DEM onto, zones have one per tile.
        if cfg.dtm.mode is not DtmMode.native:
            print(
                f"zones: dtm.mode={cfg.dtm.mode.value!r} ignored, the DTM is built "
                "natively over all zones (tiles are aligned to it at render time)"
            )
        if cfg.dtm.apply_erosion:
            simulate_thermal_erosion(
                raw_dem, cached_dtm, **(cfg.dtm.erosion_kwargs or {})
            )
        else:
            shutil.copy2(raw_dem, cached_dtm)

    dest = _results_dir(out_dir, run_dir)
    zones = _write_zone_outputs(sr_cfg, grid, references, empty, dest)
    final_dtm = dest / DTM_FILENAME
    shutil.copy2(cached_dtm, final_dtm)
    return AcquireResult(dtm_path=final_dtm, optical_path=None, zones=zones)
