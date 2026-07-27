"""Shells out to the isolated `sentinel_sr` project for Sentinel-2 RGBN SR.

`sentinel_sr` carries a heavy, GPU-specific dependency stack (torch, cubo,
mlstac, opensr-model, sen2sr, stac2cube) that would conflict with the rest of
eo_art's dependencies if merged into one environment, so it lives as its own
uv project with its own venv and is invoked as a subprocess rather than
imported directly.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from vecraspy import get_aoi_center

from eo_art.forge3d_pipes.acquire.schema import Sentinel

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Scene:
    path: Path
    cloud_cover: float | None


def read_aoi_center(aoi_path: str | Path) -> tuple[float, float]:
    """Return (lat, lon) for the center of the AOI vector file."""
    aoi_df = gpd.read_file(aoi_path)
    if aoi_df.crs is None:
        aoi_df.crs = "EPSG:4326"
    return get_aoi_center(aoi_df)


def run_sentinel_sr(
    cfg: Sentinel,
    sentinel_sr_dir: str | Path,
    lat: float,
    lon: float,
    out_dir: str | Path,
) -> list[Scene]:
    """Run the sentinel_sr subprocess and return the written GeoTIFF paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if cfg.max_items is not None:
        command += ["--max-items", str(cfg.max_items)]
    if cfg.max_cloud_cover is not None:
        command += ["--max-cloud-cover", str(cfg.max_cloud_cover)]

    subprocess.run(command, check=True)

    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    return [
        Scene(path=Path(scene["path"]), cloud_cover=scene["cloud_cover"])
        for scene in manifest["scenes"]
    ]
