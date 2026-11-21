"""AOI files: drawn or uploaded geometries stored as EPSG:4326 GeoPackages."""

import json
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd

if TYPE_CHECKING:
    from eo_art.forge3d_pipes.acquire.zones import ZoneGrid

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


LatLon = tuple[float, float]


def zone_grid(
    path: Path, edge_size: int, resolution: int, clip_to_aoi: bool = True
) -> "ZoneGrid":
    """The zones ``sentinel.zones`` splits this AOI into -- the same grid the
    acquire stage runs (raises like it, e.g. across a UTM zone boundary)."""
    from eo_art.forge3d_pipes.acquire.zones import plan_zones

    return plan_zones(path, edge_size, resolution, clip_to_aoi=clip_to_aoi)


def zone_rings(grid: "ZoneGrid") -> list[list[LatLon]]:
    """Each zone's chip as a closed (lat, lon) ring, for drawing on a map."""
    from pyproj import Transformer

    to_latlon = Transformer.from_crs(f"EPSG:{grid.epsg}", "EPSG:4326", always_xy=True)
    rings = []
    for zone in grid.zones:
        minx, miny, maxx, maxy = zone.bounds
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        rings.append(
            [(lat, lon) for lon, lat in (to_latlon.transform(x, y) for x, y in corners)]
        )
    return rings


def polygon_feature(points: Sequence[LatLon]) -> dict[str, Any]:
    """A GeoJSON polygon feature from map clicks given as (lat, lon)."""
    if len(points) < 3:
        raise ValueError(f"a polygon needs at least 3 points, got {len(points)}")
    ring = [[lon, lat] for lat, lon in points]
    ring.append(ring[0])
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def rectangle_feature(corner: LatLon, opposite: LatLon) -> dict[str, Any]:
    """A GeoJSON rectangle feature spanned by two opposite (lat, lon) corners."""
    (lat1, lon1), (lat2, lon2) = corner, opposite
    if lat1 == lat2 or lon1 == lon2:
        raise ValueError("the two corners span no area")
    south, north = sorted((lat1, lat2))
    west, east = sorted((lon1, lon2))
    return polygon_feature([(south, west), (south, east), (north, east), (north, west)])


def outlines(path: Path) -> list[list[LatLon]]:
    """Exterior rings of every polygon in the AOI as (lat, lon), for drawing."""
    gdf = _to_aoi_crs(gpd.read_file(path))
    rings: list[list[LatLon]] = []
    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue
        parts = getattr(geometry, "geoms", [geometry])
        for part in parts:
            exterior = getattr(part, "exterior", None)
            if exterior is not None:
                rings.append([(lat, lon) for lon, lat, *_ in exterior.coords])
    return rings
