import json
import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import box

from eo_art.app.aoi import (
    bounds,
    chip_bounds,
    import_upload,
    load_geojson,
    save_features,
)

SQUARE = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[10, 47], [11, 47], [11, 48], [10, 48], [10, 47]]],
    },
}


def test_save_features_roundtrip(tmp_path):
    path = save_features([SQUARE], tmp_path / "aoi" / "a.gpkg")

    gdf = gpd.read_file(path)
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == 1
    assert bounds(path) == pytest.approx((10, 47, 11, 48))
    assert load_geojson(path)["features"][0]["geometry"]["type"] == "Polygon"


def test_save_features_overwrites_instead_of_appending(tmp_path):
    path = tmp_path / "a.gpkg"
    save_features([SQUARE], path)
    save_features([SQUARE], path)

    assert len(gpd.read_file(path)) == 1


def test_save_features_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        save_features([], tmp_path / "a.gpkg")


def test_import_geojson_upload(tmp_path):
    data = json.dumps({"type": "FeatureCollection", "features": [SQUARE]}).encode()

    path = import_upload(data, "upload.geojson", tmp_path / "u.gpkg")

    assert bounds(path) == pytest.approx((10, 47, 11, 48))


def test_import_gpkg_upload_reprojects_to_wgs84(tmp_path):
    source = tmp_path / "utm.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(500000, 5200000, 510000, 5210000)], crs="EPSG:32632"
    ).to_file(source)

    path = import_upload(source.read_bytes(), "utm.gpkg", tmp_path / "u.gpkg")

    assert gpd.read_file(path).crs.to_epsg() == 4326
    minx, miny, _, _ = bounds(path)
    assert 8 < minx < 10 and 46 < miny < 48


def test_import_zipped_shapefile(tmp_path):
    shp_dir = tmp_path / "shp"
    shp_dir.mkdir()
    utm = gpd.GeoDataFrame(
        geometry=[box(500000, 5200000, 510000, 5210000)], crs="EPSG:32632"
    )
    utm.to_file(shp_dir / "aoi.shp")
    archive = tmp_path / "aoi.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for part in shp_dir.iterdir():
            zf.write(part, part.name)

    path = import_upload(archive.read_bytes(), "aoi.zip", tmp_path / "z.gpkg")

    assert gpd.read_file(path).crs.to_epsg() == 4326
    minx, miny, _, _ = bounds(path)
    assert 8 < minx < 10 and 46 < miny < 48


def test_import_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError):
        import_upload(b"", "aoi.kml", tmp_path / "x.gpkg")


def test_chip_bounds_is_square_in_metres():
    south, west, north, east = chip_bounds(60.0, 10.0, 1280.0)

    assert (north - south) * 111_320 == pytest.approx(1280)
    # At 60°N a degree of longitude is half as long as one of latitude.
    assert (east - west) == pytest.approx(2 * (north - south))
    assert (south + north) / 2 == pytest.approx(60.0)


def test_rectangle_feature_from_any_two_corners_roundtrips(tmp_path):
    from eo_art.app.aoi import outlines, rectangle_feature

    feature = rectangle_feature((47.6, 11.2), (47.5, 11.0))
    path = save_features([feature], tmp_path / "rect.gpkg")

    (ring,) = outlines(path)
    lats = sorted({round(lat, 6) for lat, _ in ring})
    lons = sorted({round(lon, 6) for _, lon in ring})
    assert lats == [47.5, 47.6]
    assert lons == [11.0, 11.2]


def test_polygon_and_rectangle_reject_degenerate_input():
    from eo_art.app.aoi import polygon_feature, rectangle_feature

    with pytest.raises(ValueError, match="at least 3"):
        polygon_feature([(1.0, 1.0), (2.0, 2.0)])
    with pytest.raises(ValueError, match="no area"):
        rectangle_feature((1.0, 1.0), (1.0, 2.0))


def test_polygon_feature_closes_ring_in_lon_lat_order():
    from eo_art.app.aoi import polygon_feature

    ring = polygon_feature([(10.0, 1.0), (10.0, 2.0), (11.0, 2.0)])["geometry"][
        "coordinates"
    ][0]
    assert ring == [[1.0, 10.0], [2.0, 10.0], [2.0, 11.0], [1.0, 10.0]]


def test_zone_rings_cover_the_aoi_like_the_pipeline(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    from eo_art.app.aoi import zone_grid, zone_rings

    path = tmp_path / "big.gpkg"
    # ~3 x 2 km in UTM 32N.
    gpd.GeoDataFrame(
        geometry=[box(652_000, 5_208_000, 655_000, 5_210_000)], crs="EPSG:32632"
    ).to_file(path)

    grid = zone_grid(path, 128, 10)
    rings = zone_rings(grid)

    assert len(rings) == len(grid.zones) == 6
    for ring, zone in zip(rings, grid.zones, strict=True):
        assert ring[0] == ring[-1] and len(ring) == 5
        lats = [lat for lat, _ in ring]
        assert min(lats) < zone.lat < max(lats)
