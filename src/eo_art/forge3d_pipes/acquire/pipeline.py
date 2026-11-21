"""Orchestrates sentinel SR -> DEM fetch -> DTM super-resolution -> align."""

import hashlib
import json
import re
import shutil
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
    MANIFEST_NAME,
    Scene,
    read_aoi_center,
    read_manifest,
    run_sentinel_sr,
)
from eo_art.forge3d_pipes.acquire.terrain import fetch_local_dem, fetch_terrain_dem

CACHE_DIRNAME = "_acquire"
DTM_FILENAME = "dtm.tif"
OPTICAL_FILENAME = "optical_aligned.tif"
SR_SUFFIX = "_sr.tif"
TIMESERIES_SUFFIX = "_sr_timeseries"
# Name inside the per-config cache dir (already unique per AOI via the hash);
# unchanged so caches written before outputs were named per AOI stay valid.
CACHED_SR_FILENAME = "optical_sr.tif"


@dataclass(frozen=True)
class AcquireResult:
    dtm_path: Path
    optical_path: Path


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
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


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
    cfg: AcquireConfig, out_dir: str | Path, use_cache: bool = True
) -> AcquireResult:
    """Run the acquire chain, caching the final DTM/optical pair by config hash.

    The cache lives under ``<out_dir>/_acquire/<hash>`` (so re-running with an
    unchanged config skips recomputation) but the returned paths are always
    the stable ``<out_dir>/dtm.tif`` / ``<out_dir>/optical_aligned.tif`` --
    matching how the render pipeline's own ``_prep`` cache stays internal
    while its final artifacts land directly in the given out_dir.
    """
    out_dir = Path(out_dir)
    cache_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    cached_dtm = cache_dir / DTM_FILENAME
    cached_optical = cache_dir / OPTICAL_FILENAME
    final_dtm = out_dir / DTM_FILENAME
    final_optical = out_dir / OPTICAL_FILENAME

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

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_dtm, final_dtm)
    shutil.copy2(cached_optical, final_optical)
    return AcquireResult(dtm_path=final_dtm, optical_path=final_optical)


def sr_filename(aoi_path: str | Path) -> str:
    """``aoi/valley.gpkg`` -> ``valley_sr.tif``, so SR runs for different
    AOIs sharing one out_dir don't overwrite each other's result."""
    return f"{Path(aoi_path).stem}{SR_SUFFIX}"


def run_sr(cfg: SrConfig, out_dir: str | Path, use_cache: bool = True) -> Path:
    """Run only the Sentinel SR step and return ``<out_dir>/<aoi>_sr.tif``.

    The reference scene (lowest cloud cover, or ``sentinel.reference_index``)
    is cached under ``<out_dir>/_acquire/<hash>`` like ``run_acquire``'s
    outputs; the full set of SR timesteps stays next to it in ``sentinel/``.
    For ``sentinel.timeseries.enabled`` use ``run_sr_timeseries``.
    """
    if cfg.sentinel.timeseries.enabled:
        raise ValueError("sentinel.timeseries is enabled; use run_sr_timeseries")
    out_dir = Path(out_dir)
    cache_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    cached_sr = cache_dir / CACHED_SR_FILENAME
    final_sr = out_dir / sr_filename(cfg.aoi_path)

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

    out_dir.mkdir(parents=True, exist_ok=True)
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
    cfg: SrConfig, out_dir: str | Path, use_cache: bool = True
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
    timeseries = cfg.sentinel.timeseries
    if not timeseries.enabled:
        raise ValueError("sentinel.timeseries is disabled; use run_sr")
    out_dir = Path(out_dir)
    sentinel_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg) / "sentinel"
    target = out_dir / timeseries_dirname(cfg.aoi_path)
    existing = timeseries_frames(cfg.aoi_path, target) if target.exists() else {}

    # Filling gaps always searches again: an open interval may have a usable
    # scene now (longer date range, relaxed cloud limits, new acquisitions).
    if timeseries.fill_gaps or not (
        use_cache and (sentinel_dir / MANIFEST_NAME).exists()
    ):
        lat, lon = read_aoi_center(cfg.aoi_path)
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
        raise RuntimeError(
            f"sentinel_sr wrote no scenes for aoi={cfg.aoi_path!r} "
            f"in [{cfg.sentinel.start_date}, {cfg.sentinel.end_date}]"
        )

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
