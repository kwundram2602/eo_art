"""Split an AOI into chip-sized zones that Sentinel SR processes one by one.

``sentinel_sr`` super-resolves a single square chip of ``edge_size`` pixels
around a center point, and the SR model runs on that whole chip at once, so a
chip can't grow with the AOI without running out of GPU memory. A large AOI
is instead covered by a grid of such chips (zones), each fetched and
super-resolved in its own ``sentinel_sr`` run.

cubo (inside ``sentinel_sr``) builds a chip by snapping the center to the
``resolution`` grid in the center's UTM zone and buffering it by
``round(edge_size * resolution / 2)`` metres. Zone centers placed exactly one
chip apart on that grid therefore yield chips that meet pixel-exactly.
"""

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.coords import BoundingBox
from rasterio.shutil import copy as rasterio_copy  # ty: ignore[unresolved-import]
from shapely.geometry import box

Bounds = tuple[float, float, float, float]

# rasterio dtype -> GDAL data type name, as used in VRT XML.
_GDAL_TYPES = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
}


@dataclass(frozen=True)
class Zone:
    # z<row>_<col>, row 0 in the north, col 0 in the west.
    id: str
    row: int
    col: int
    # Center handed to sentinel_sr, in EPSG:4326.
    lat: float
    lon: float
    # (minx, miny, maxx, maxy) of the chip in the grid's UTM CRS.
    bounds: Bounds


@dataclass(frozen=True)
class _Tile:
    path: Path
    crs: CRS
    res: tuple[float, float]
    count: int
    dtype: str
    nodata: float | None
    bounds: BoundingBox
    width: int
    height: int
    descriptions: tuple[str | None, ...]


@dataclass(frozen=True)
class ZoneGrid:
    epsg: int
    # Chip edge in metres.
    tile_m: float
    zones: list[Zone]
    # Bounding box of all zones in the grid's UTM CRS.
    bounds: Bounds


def utm_epsg(lat: float, lon: float) -> int:
    """The WGS 84 UTM zone cubo picks for a chip centered on (lat, lon)."""
    infos = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(lon, lat, lon, lat),
    )
    return int(infos[0].code)


def _read_aoi(aoi_path: str | Path) -> gpd.GeoDataFrame:
    # Same CRS fallback as sentinel_bridge.read_aoi_center.
    aoi = gpd.read_file(aoi_path)
    if aoi.crs is None:
        aoi = aoi.set_crs("EPSG:4326")
    return aoi


def plan_zones(
    aoi_path: str | Path,
    edge_size: int,
    resolution: int,
    clip_to_aoi: bool = True,
) -> ZoneGrid:
    """Cover the AOI with a grid of ``edge_size`` x ``edge_size`` px chips.

    The grid is centered on the AOI's bounding box, so an AOI smaller than
    one chip gets exactly one zone. With ``clip_to_aoi`` cells that don't
    overlap the AOI geometry (corners of an irregular AOI) are dropped.

    Raises ``ValueError`` if a chip's half edge isn't a multiple of
    ``resolution`` (chips would not meet on the pixel grid) or if the zones
    fall into different UTM zones (cubo would put their chips into different
    CRS, which can't be mosaicked seamlessly).
    """
    buffer = round(edge_size * resolution / 2)
    if buffer % resolution:
        raise ValueError(
            f"sentinel.edge_size={edge_size} at resolution={resolution} gives a "
            f"half chip of {buffer} m, not a multiple of {resolution} m, so zones "
            "would not meet on the pixel grid; use an even edge_size"
        )
    tile = 2 * buffer

    aoi = _read_aoi(aoi_path)
    minx, miny, maxx, maxy = aoi.to_crs("EPSG:4326").total_bounds
    epsg = utm_epsg((miny + maxy) / 2, (minx + maxx) / 2)
    geometry = aoi.to_crs(epsg=epsg).union_all()

    minx, miny, maxx, maxy = geometry.bounds
    ncols = max(1, math.ceil((maxx - minx) / tile))
    nrows = max(1, math.ceil((maxy - miny) / tile))
    # Centered on the AOI, snapped outward to the pixel grid; snapping can
    # leave a sliver of the AOI uncovered, which one more row/column takes.
    x0 = math.floor(((minx + maxx) / 2 - ncols * tile / 2) / resolution) * resolution
    y0 = math.ceil(((miny + maxy) / 2 + nrows * tile / 2) / resolution) * resolution
    if x0 + ncols * tile < maxx:
        ncols += 1
    if y0 - nrows * tile > miny:
        nrows += 1

    to_latlon = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    width = max(2, len(str(max(nrows, ncols) - 1)))
    zones = []
    other_utm = []
    for row in range(nrows):
        for col in range(ncols):
            cell = (
                x0 + col * tile,
                y0 - (row + 1) * tile,
                x0 + (col + 1) * tile,
                y0 - row * tile,
            )
            if clip_to_aoi:
                chip = box(*cell)
                if not chip.intersects(geometry) or chip.touches(geometry):
                    continue
            lon, lat = to_latlon.transform(cell[0] + buffer, cell[1] + buffer)
            zone_id = f"z{row:0{width}d}_{col:0{width}d}"
            if utm_epsg(lat, lon) != epsg:
                other_utm.append(zone_id)
            zones.append(Zone(zone_id, row, col, lat, lon, cell))

    if other_utm:
        raise ValueError(
            f"the AOI {aoi_path} crosses a UTM zone boundary: zone(s) "
            f"{', '.join(other_utm)} lie outside EPSG:{epsg}, so their chips "
            "would come in another CRS; split the AOI at the boundary"
        )
    if not zones:
        raise ValueError(f"no zone of the grid overlaps the AOI {aoi_path}")

    return ZoneGrid(
        epsg=epsg,
        tile_m=float(tile),
        zones=zones,
        bounds=(
            min(z.bounds[0] for z in zones),
            min(z.bounds[1] for z in zones),
            max(z.bounds[2] for z in zones),
            max(z.bounds[3] for z in zones),
        ),
    )


def build_vrt(
    tiles: Sequence[str | Path], dst: str | Path, bounds: Bounds | None = None
) -> Path:
    """Write a VRT mosaic of ``tiles`` at ``dst``.

    The tiles must share CRS, pixel size, band count and data type, as zone
    tiles do; nothing is resampled. Tiles are referenced relative to the VRT,
    so the folder can be moved. Areas without a tile are nodata (NaN for
    float data without a nodata value of its own). ``bounds`` (minx, miny,
    maxx, maxy) fixes the extent, e.g. the whole zone grid, so mosaics of
    different zone subsets share one grid; default: the tiles' union.
    """
    dst = Path(dst)
    if not tiles:
        raise ValueError("build_vrt needs at least one tile")

    metas = []
    for tile in tiles:
        with rasterio.open(tile) as src:
            metas.append(
                _Tile(
                    path=Path(tile),
                    crs=CRS.from_user_input(src.crs),
                    res=src.res,
                    count=src.count,
                    dtype=src.dtypes[0],
                    nodata=src.nodata,
                    bounds=src.bounds,
                    width=src.width,
                    height=src.height,
                    descriptions=tuple(src.descriptions),
                )
            )
    first = metas[0]
    for meta in metas[1:]:
        for key in ("crs", "res", "count", "dtype"):
            if getattr(meta, key) != getattr(first, key):
                raise ValueError(
                    f"tile {meta.path} has {key}={getattr(meta, key)!r}, "
                    f"{first.path} has {getattr(first, key)!r}"
                )

    xres, yres = first.res
    if bounds is not None:
        left, bottom, right, top = bounds
    else:
        left = min(m.bounds.left for m in metas)
        top = max(m.bounds.top for m in metas)
        right = max(m.bounds.right for m in metas)
        bottom = min(m.bounds.bottom for m in metas)
    width = round((right - left) / xres)
    height = round((top - bottom) / yres)

    nodata = first.nodata
    if nodata is None and np.issubdtype(np.dtype(str(first.dtype)), np.floating):
        nodata = float("nan")

    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    ET.SubElement(root, "SRS").text = first.crs.to_wkt()
    ET.SubElement(
        root, "GeoTransform"
    ).text = f"{left!r}, {xres!r}, 0.0, {top!r}, 0.0, {-yres!r}"
    for band in range(1, first.count + 1):
        band_el = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType=_GDAL_TYPES[first.dtype],
            band=str(band),
        )
        description = first.descriptions[band - 1]
        if description:
            ET.SubElement(band_el, "Description").text = description
        if nodata is not None:
            ET.SubElement(band_el, "NoDataValue").text = (
                "nan" if math.isnan(nodata) else repr(nodata)
            )
        for meta in metas:
            source = ET.SubElement(band_el, "SimpleSource")
            filename = ET.SubElement(source, "SourceFilename", relativeToVRT="1")
            filename.text = Path(
                os.path.relpath(meta.path.resolve(), dst.parent.resolve())
            ).as_posix()
            ET.SubElement(source, "SourceBand").text = str(band)
            w, h = meta.width, meta.height
            ET.SubElement(
                source, "SrcRect", xOff="0", yOff="0", xSize=str(w), ySize=str(h)
            )
            ET.SubElement(
                source,
                "DstRect",
                xOff=str(round((meta.bounds.left - left) / xres)),
                yOff=str(round((top - meta.bounds.top) / yres)),
                xSize=str(w),
                ySize=str(h),
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(dst, encoding="unicode")
    return dst


def merge_first_valid(parts: Sequence[str | Path], dst: str | Path) -> Path:
    """Merge rasters on one grid into ``dst``: per pixel the first part with
    data in all bands wins (NaN = no data). Used to put the co-registered
    date groups of a step back together."""
    if not parts:
        raise ValueError("merge_first_valid needs at least one raster")
    dst = Path(dst)
    with rasterio.open(parts[0]) as src:
        profile = src.profile
        descriptions = src.descriptions
        out = src.read().astype("float32")
    for part in parts[1:]:
        with rasterio.open(part) as src:
            if (src.transform, src.width, src.height) != (
                profile["transform"],
                profile["width"],
                profile["height"],
            ):
                raise ValueError(f"{part} is not on the grid of {parts[0]}")
            data = src.read().astype("float32")
        hole = np.isnan(out).any(axis=0)
        out[:, hole] = data[:, hole]
    profile.update(
        driver="GTiff", dtype="float32", nodata=float("nan"), compress="deflate"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **profile) as out_ds:
        out_ds.write(out)
        for band, description in enumerate(descriptions, start=1):
            if description:
                out_ds.set_band_description(band, description)
    return dst


def export_mosaic(vrt: str | Path, dst: str | Path) -> Path:
    """Write the VRT mosaic as one Cloud Optimized GeoTIFF (streamed by
    GDAL, never read into memory as a whole)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rasterio_copy(vrt, dst, driver="COG", compress="deflate", bigtiff="IF_SAFER")
    return dst
