"""Orchestrates sentinel SR -> DEM fetch -> DTM super-resolution -> align."""

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from vecraspy import align_raster_grid, simulate_thermal_erosion, super_resolve_dtm

from eo_art.forge3d_pipes.acquire.schema import (
    AcquireConfig,
    DemSource,
    DtmMode,
    SrConfig,
)
from eo_art.forge3d_pipes.acquire.sentinel_bridge import (
    Scene,
    read_aoi_center,
    run_sentinel_sr,
)
from eo_art.forge3d_pipes.acquire.terrain import fetch_local_dem, fetch_terrain_dem

CACHE_DIRNAME = "_acquire"
DTM_FILENAME = "dtm.tif"
OPTICAL_FILENAME = "optical_aligned.tif"
SR_FILENAME = "optical_sr.tif"


@dataclass(frozen=True)
class AcquireResult:
    dtm_path: Path
    optical_path: Path


def _cache_key(cfg: AcquireConfig | SrConfig) -> str:
    # out_dir says where to store results, not what to compute -- excluded so
    # moving the output location doesn't invalidate an otherwise-identical
    # cached result (or, worse, silently redo an expensive GPU+network run).
    payload = {k: v for k, v in asdict(cfg).items() if k != "out_dir"}
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

    # Scenes with no cloud_cover metadata (some collections lack the eo
    # extension) sort last rather than winning by default.
    return min(
        scenes,
        key=lambda scene: (
            scene.cloud_cover is None,
            scene.cloud_cover if scene.cloud_cover is not None else 0.0,
        ),
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
        written = run_sentinel_sr(
            cfg.sentinel, cfg.sentinel_sr_dir, lat, lon, cache_dir / "sentinel"
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


def run_sr(cfg: SrConfig, out_dir: str | Path, use_cache: bool = True) -> Path:
    """Run only the Sentinel SR step and return ``<out_dir>/optical_sr.tif``.

    The reference scene (lowest cloud cover, or ``sentinel.reference_index``)
    is cached under ``<out_dir>/_acquire/<hash>`` like ``run_acquire``'s
    outputs; the full set of SR timesteps stays next to it in ``sentinel/``.
    """
    out_dir = Path(out_dir)
    cache_dir = out_dir / CACHE_DIRNAME / _cache_key(cfg)
    cached_sr = cache_dir / SR_FILENAME
    final_sr = out_dir / SR_FILENAME

    if not (use_cache and cached_sr.exists()):
        cache_dir.mkdir(parents=True, exist_ok=True)

        lat, lon = read_aoi_center(cfg.aoi_path)
        written = run_sentinel_sr(
            cfg.sentinel, cfg.sentinel_sr_dir, lat, lon, cache_dir / "sentinel"
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
