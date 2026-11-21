"""AOI files: drawn or uploaded geometries stored as EPSG:4326 GeoPackages."""

import json
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd

AOI_CRS = "EPSG:4326"
UPLOAD_SUFFIXES = (".gpkg", ".geojson", ".json", ".zip")
_M_PER_DEG_LAT = 111_320.0


def _to_aoi_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Same assumption read_aoi_center makes for CRS-less vector files.
    if gdf.crs is None:
        gdf = gdf.set_crs(AOI_CRS)
    return gdf.to_crs(AOI_CRS)


def _write(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    if gdf.empty:
        raise ValueError("AOI has no geometries")
    path.parent.mkdir(parents=True, exist_ok=True)
    # The gpkg driver appends layers to an existing file instead of
    # replacing it, so an overwrite has to start from scratch.
    path.unlink(missing_ok=True)
    gdf[["geometry"]].to_file(path, driver="GPKG", layer="aoi")
    return path


def save_features(features: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write GeoJSON features (as drawn on a Leaflet map, i.e. WGS84) to gpkg."""
    gdf = gpd.GeoDataFrame.from_features(list(features), crs=AOI_CRS)
    return _write(gdf, path)


def import_upload(data: bytes, filename: str, path: Path) -> Path:
    """Read an uploaded gpkg / GeoJSON / zipped shapefile and store it as AOI."""
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise ValueError(
            f"unsupported AOI file {filename!r}; expected one of {UPLOAD_SUFFIXES}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / Path(filename).name
        source.write_bytes(data)
        gdf = gpd.read_file(source)
    return _write(_to_aoi_crs(gdf), path)


def load_geojson(path: Path) -> dict[str, Any]:
    """The AOI as a GeoJSON FeatureCollection in EPSG:4326, for map display."""
    return json.loads(_to_aoi_crs(gpd.read_file(path)).to_json())


def bounds(path: Path) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) in EPSG:4326."""
    minx, miny, maxx, maxy = _to_aoi_crs(gpd.read_file(path)).total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def center(path: Path) -> tuple[float, float]:
    """(lat, lon) exactly as the acquire stage computes it for Sentinel SR."""
    from eo_art.forge3d_pipes.acquire.sentinel_bridge import read_aoi_center

    return read_aoi_center(path)


def chip_bounds(
    lat: float, lon: float, edge_m: float
) -> tuple[float, float, float, float]:
    """Approximate (south, west, north, east) of a square ``edge_m`` chip
    centred on (lat, lon) -- the footprint Sentinel SR actually covers."""
    half = edge_m / 2
    dlat = half / _M_PER_DEG_LAT
    dlon = half / (_M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon
