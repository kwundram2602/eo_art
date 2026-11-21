import math

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from eo_art.forge3d_pipes.acquire.zones import (
    build_vrt,
    export_mosaic,
    plan_zones,
)

# Inside UTM zone 32N, well away from its 6/12 degree E boundaries.
EPSG = 32632
X0, Y0 = 652_003.0, 5_207_997.0


def _aoi(tmp_path, geometry, crs=f"EPSG:{EPSG}", name="aoi.gpkg"):
    path = tmp_path / name
    gpd.GeoDataFrame(geometry=[geometry], crs=crs).to_file(path)
    return path


def _cubo_bbox(lat, lon, edge_size, resolution, epsg=EPSG):
    """cubo.utils._central_pixel_bbox, reduced to the UTM bounds."""
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(
        lon, lat
    )
    x, y = (round(c / resolution) * resolution for c in (x, y))
    buffer = round(edge_size * resolution / 2)
    return (x - buffer, y - buffer, x + buffer, y + buffer)


def test_small_aoi_is_one_zone(tmp_path):
    grid = plan_zones(_aoi(tmp_path, box(X0, Y0, X0 + 500, Y0 + 500)), 128, 10)

    assert grid.epsg == EPSG
    assert grid.tile_m == 1280
    assert [z.id for z in grid.zones] == ["z00_00"]
    minx, miny, maxx, maxy = grid.zones[0].bounds
    assert maxx - minx == maxy - miny == 1280
    assert minx <= X0 and maxx >= X0 + 500 and miny <= Y0 and maxy >= Y0 + 500


def test_zones_tile_the_aoi_without_gaps(tmp_path):
    grid = plan_zones(_aoi(tmp_path, box(X0, Y0, X0 + 3000, Y0 + 2000)), 128, 10)

    assert [z.id for z in grid.zones] == [
        "z00_00",
        "z00_01",
        "z00_02",
        "z01_00",
        "z01_01",
        "z01_02",
    ]
    by_id = {z.id: z for z in grid.zones}
    # Row 0 is the northern row; neighbours share an edge exactly.
    assert by_id["z00_00"].bounds[2] == by_id["z00_01"].bounds[0]
    assert by_id["z00_00"].bounds[1] == by_id["z01_00"].bounds[3]
    minx, miny, maxx, maxy = grid.bounds
    assert minx <= X0 and maxx >= X0 + 3000 and miny <= Y0 and maxy >= Y0 + 2000
    assert all(v % 10 == 0 for v in grid.bounds)


def test_zone_center_reproduces_the_cubo_chip(tmp_path):
    grid = plan_zones(_aoi(tmp_path, box(X0, Y0, X0 + 3000, Y0 + 2000)), 128, 10)

    for zone in grid.zones:
        assert _cubo_bbox(zone.lat, zone.lon, 128, 10) == pytest.approx(zone.bounds)


def test_aoi_in_geographic_crs(tmp_path):
    to_ll = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    corners = [
        to_ll.transform(x, y)
        for x, y in box(X0, Y0, X0 + 3000, Y0 + 2000).exterior.coords
    ]
    grid = plan_zones(_aoi(tmp_path, Polygon(corners), crs="EPSG:4326"), 128, 10)

    assert grid.epsg == EPSG
    assert len(grid.zones) == 6


def test_clip_to_aoi_drops_cells_outside_the_geometry(tmp_path):
    # L shape: the north-east cell of a 2x2 grid stays empty.
    shape = Polygon(
        [
            (X0, Y0),
            (X0 + 2500, Y0),
            (X0 + 2500, Y0 + 1000),
            (X0 + 1000, Y0 + 1000),
            (X0 + 1000, Y0 + 2500),
            (X0, Y0 + 2500),
        ]
    )
    path = _aoi(tmp_path, shape)

    clipped = plan_zones(path, 128, 10)
    full = plan_zones(path, 128, 10, clip_to_aoi=False)

    assert len(full.zones) == 4
    assert [z.id for z in clipped.zones] == ["z00_00", "z01_00", "z01_01"]


def test_aoi_across_a_utm_boundary_raises(tmp_path):
    path = _aoi(tmp_path, box(5.9, 47.0, 6.1, 47.05), crs="EPSG:4326")

    with pytest.raises(ValueError, match="UTM"):
        plan_zones(path, 128, 10)


def test_edge_size_off_the_pixel_grid_raises(tmp_path):
    path = _aoi(tmp_path, box(X0, Y0, X0 + 500, Y0 + 500))

    with pytest.raises(ValueError, match="edge_size"):
        plan_zones(path, 127, 10)


def _write_tile(path, left, top, value, size=4, res=2.5):
    data = np.stack([np.full((size, size), value + b, "float32") for b in range(3)])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="float32",
        crs=f"EPSG:{EPSG}",
        transform=from_origin(left, top, res, res),
    ) as dst:
        dst.write(data)
        dst.descriptions = ("blue", "green", "red")
    return path


def test_build_vrt_mosaics_tiles(tmp_path):
    tiles = [
        _write_tile(tmp_path / "a.tif", 0, 20, 10),
        _write_tile(tmp_path / "b.tif", 10, 20, 20),
        _write_tile(tmp_path / "c.tif", 0, 10, 30),
        _write_tile(tmp_path / "d.tif", 10, 10, 40),
    ]
    vrt = build_vrt(tiles, tmp_path / "mosaic.vrt")

    with rasterio.open(vrt) as src:
        assert (src.width, src.height, src.count) == (8, 8, 3)
        assert src.crs.to_epsg() == EPSG
        assert src.bounds == pytest.approx((0, 0, 20, 20))
        assert src.descriptions == ("blue", "green", "red")
        band1 = src.read(1)
        band3 = src.read(3)
    assert band1[0, 0] == 10 and band1[0, 7] == 20
    assert band1[7, 0] == 30 and band1[7, 7] == 40
    assert band3[7, 7] == 42


def test_build_vrt_leaves_missing_tiles_as_nodata(tmp_path):
    tiles = [
        _write_tile(tmp_path / "a.tif", 0, 20, 10),
        _write_tile(tmp_path / "d.tif", 10, 10, 40),
    ]
    vrt = build_vrt(tiles, tmp_path / "mosaic.vrt")

    with rasterio.open(vrt) as src:
        band = src.read(1)
    assert math.isnan(band[0, 7]) and math.isnan(band[7, 0])
    assert band[0, 0] == 10


def test_build_vrt_uses_relative_paths(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    tiles = [_write_tile(tiles_dir / "a.tif", 0, 20, 10)]
    vrt = build_vrt(tiles, tiles_dir / "mosaic.vrt")

    moved = tmp_path / "moved"
    tiles_dir.rename(moved)
    with rasterio.open(moved / "mosaic.vrt") as src:
        assert src.read(1)[0, 0] == 10
    assert str(tmp_path) not in (moved / vrt.name).read_text()


def test_export_mosaic_writes_one_geotiff(tmp_path):
    tiles = [
        _write_tile(tmp_path / "a.tif", 0, 20, 10),
        _write_tile(tmp_path / "b.tif", 10, 20, 20),
    ]
    vrt = build_vrt(tiles, tmp_path / "mosaic.vrt")

    out = export_mosaic(vrt, tmp_path / "mosaic.tif")

    with rasterio.open(out) as src:
        assert src.driver == "GTiff"
        assert (src.width, src.height) == (8, 4)
        assert src.read(1)[0, 7] == 20
