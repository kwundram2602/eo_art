"""Orchestrates sentinel SR -> DEM fetch -> DTM super-resolution -> align."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vecraspy import align_raster_grid, super_resolve_dtm

from eo_art.forge3d_pipes.acquire.schema import AcquireConfig
from eo_art.forge3d_pipes.acquire.sentinel_bridge import (
    Scene,
    read_aoi_center,
    run_sentinel_sr,
)
from eo_art.forge3d_pipes.acquire.terrain import fetch_terrain_dem

CACHE_DIRNAME = "_acquire"
DTM_FILENAME = "dtm.tif"
OPTICAL_FILENAME = "optical_aligned.tif"


@dataclass(frozen=True)
class AcquireResult:
    dtm_path: Path
    optical_path: Path


def _cache_key(cfg: AcquireConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
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
    """Run the acquire chain, caching the final DTM/optical pair by config hash."""
    cache_dir = Path(out_dir) / CACHE_DIRNAME / _cache_key(cfg)
    dtm_path = cache_dir / DTM_FILENAME
    optical_path = cache_dir / OPTICAL_FILENAME

    if use_cache and dtm_path.exists() and optical_path.exists():
        return AcquireResult(dtm_path=dtm_path, optical_path=optical_path)

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

    raw_dem = fetch_terrain_dem(
        reference_tif,
        cache_dir,
        source=cfg.dem.source,
        ee_project=cfg.ee_project,
        scale=cfg.dem.scale,
    )

    super_resolve_dtm(
        raw_dem,
        reference_tif,
        dtm_path,
        band=cfg.dtm.band,
        radius=cfg.dtm.radius,
        eps=cfg.dtm.eps,
        apply_erosion=cfg.dtm.apply_erosion,
        erosion_kwargs=cfg.dtm.erosion_kwargs or None,
    )

    align_raster_grid(
        dtm_path, reference_tif, optical_path, resampling=cfg.align.resampling.value
    )

    return AcquireResult(dtm_path=dtm_path, optical_path=optical_path)
