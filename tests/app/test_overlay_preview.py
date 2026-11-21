import io
import math

import imageio.v3 as iio
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from eo_art.app.overlay_preview import (
    AMBIENT,
    FLAT_LEVEL,
    build_previews,
    hillshade,
    relief_light,
)

GRID = {
    "crs": "EPSG:32632",
    "transform": from_origin(600000.0, 5200000.0, 2.5, 2.5),
    "width": 16,
    "height": 8,
}


def _write_dem(path):
    ramp = np.tile(np.linspace(100.0, 140.0, 16, dtype="float32"), (8, 1))
    with rasterio.open(
        path, "w", driver="GTiff", count=1, dtype="float32", **GRID
    ) as dst:
        dst.write(ramp, 1)
    return path


def _project(tmp_path, make_sr_tif, prepare="[{op: rgb_stretch, bands: [3, 2, 1]}]"):
    (tmp_path / "out").mkdir()
    _write_dem(tmp_path / "out" / "dtm.tif")
    make_sr_tif(tmp_path / "out" / "optical_aligned.tif")
    config = tmp_path / "configs" / "render.yaml"
    config.parent.mkdir()
    config.write_text(
        "run: {name: demo, out_dir: out}\n"
        "input: {path: out/dtm.tif}\n"
        "overlays:\n"
        "  - name: optical\n"
        "    path: out/optical_aligned.tif\n"
        f"    prepare: {prepare}\n"
        "    opacity: 0.8\n"
    )
    return config


def test_hillshade_of_flat_terrain_is_the_sun_elevation_everywhere():
    shade = hillshade(np.zeros((4, 5)), (10.0, 10.0), azimuth=315.0, elevation=30.0)

    assert shade.shape == (4, 5)
    assert np.allclose(shade, math.sin(math.radians(30.0)))


def test_hillshade_lights_slopes_facing_the_sun():
    # Rises towards the east, so the slope faces west.
    dem = np.tile(np.arange(6, dtype="float64") * 5.0, (6, 1))

    from_west = hillshade(dem, (10.0, 10.0), azimuth=270.0, elevation=30.0)
    from_east = hillshade(dem, (10.0, 10.0), azimuth=90.0, elevation=30.0)

    assert from_west.mean() > from_east.mean()


def test_relief_light_keeps_flat_ground_equally_bright_for_any_sun():
    for elevation in (10.0, 45.0, 80.0):
        flat = hillshade(np.zeros((3, 3)), (1.0, 1.0), 315.0, elevation)
        light = relief_light(flat, elevation)
        assert np.allclose(light, AMBIENT + (1 - AMBIENT) * FLAT_LEVEL)


def test_relief_light_keeps_slopes_apart_under_a_low_sun():
    dem = np.tile(np.arange(6, dtype="float64") * 5.0, (6, 1))
    shade = hillshade(dem, (10.0, 10.0), azimuth=270.0, elevation=10.0)
    away = hillshade(dem, (10.0, 10.0), azimuth=90.0, elevation=10.0)

    assert relief_light(shade, 10.0).mean() - relief_light(away, 10.0).mean() > 0.3


def test_build_previews_resolves_paths_against_the_project(tmp_path, make_sr_tif):
    config = _project(tmp_path, make_sr_tif)

    previews = build_previews(tmp_path, [config])

    assert len(previews) == 1
    preview = previews[0]
    assert preview.name == "optical"
    assert preview.source == "out/optical_aligned.tif"
    assert preview.prepare == ["rgb_stretch"]
    assert preview.opacity == 0.8
    # Same cache the render run uses, so a preview warms it.
    assert preview.png_path.parent == tmp_path / "out" / "demo" / "_prep"
    assert preview.png_path.exists()

    relief = iio.imread(io.BytesIO(preview.relief_png))
    overlay = iio.imread(io.BytesIO(preview.overlay_png))
    assert relief.shape[:2] == (8, 16)
    assert overlay.shape[:2] == (8, 16)


def test_build_previews_uses_the_configured_band_order(tmp_path, make_sr_tif):
    # make_sr_tif's bands rise with their index, so band 3 is the brightest of
    # the first three: as R with [3, 2, 1], blue-first ([1, 2, 3]) puts it in B.
    red_first = build_previews(tmp_path, [_project(tmp_path, make_sr_tif)])[0]
    rgb = iio.imread(red_first.png_path)[..., :3].astype(float)

    assert rgb[..., 0].mean() > rgb[..., 2].mean()


def test_build_previews_reports_a_missing_input(tmp_path, make_sr_tif):
    config = _project(tmp_path, make_sr_tif)
    (tmp_path / "out" / "dtm.tif").unlink()

    with pytest.raises(FileNotFoundError, match="dtm.tif"):
        build_previews(tmp_path, [config])
