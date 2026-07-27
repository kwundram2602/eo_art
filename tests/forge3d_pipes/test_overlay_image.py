import imageio.v3 as iio
import numpy as np
import rasterio
from rasterio.transform import from_origin

from eo_art.forge3d_pipes.prep.overlay_image import (
    export_overlay_png,
    export_overlay_png_cached,
)


def _write_raster(path, bands, crs="EPSG:4326"):
    count, height, width = bands.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=bands.dtype,
        crs=crs,
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(bands)


def test_export_overlay_png_from_three_band_raster(tmp_path):
    src = tmp_path / "src.tif"
    bands = np.stack(
        [
            np.full((4, 4), 10.0, dtype="float32"),
            np.full((4, 4), 128.0, dtype="float32"),
            np.full((4, 4), 250.0, dtype="float32"),
        ]
    )
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    result = export_overlay_png(src, dst)

    assert result == dst
    pixels = iio.imread(dst)
    assert pixels.shape == (4, 4, 3)
    assert tuple(pixels[0, 0]) == (10, 128, 250)


def test_export_overlay_png_from_single_band_raster_is_grayscale_rgb(tmp_path):
    src = tmp_path / "src.tif"
    bands = np.stack([np.full((4, 4), 77.0, dtype="float32")])
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    export_overlay_png(src, dst)

    pixels = iio.imread(dst)
    assert pixels.shape == (4, 4, 3)
    assert tuple(pixels[0, 0]) == (77, 77, 77)


def test_export_overlay_png_clips_out_of_range_values(tmp_path):
    src = tmp_path / "src.tif"
    bands = np.stack(
        [
            np.full((4, 4), -50.0, dtype="float32"),
            np.full((4, 4), 128.0, dtype="float32"),
            np.full((4, 4), 999.0, dtype="float32"),
        ]
    )
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    export_overlay_png(src, dst)

    pixels = iio.imread(dst)
    assert tuple(pixels[0, 0]) == (0, 128, 255)


def test_export_overlay_png_uses_only_first_three_bands(tmp_path):
    src = tmp_path / "src.tif"
    bands = np.stack([np.full((4, 4), float(v), dtype="float32") for v in range(5)])
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    export_overlay_png(src, dst)

    pixels = iio.imread(dst)
    assert pixels.shape == (4, 4, 3)
    assert tuple(pixels[0, 0]) == (0, 1, 2)


def test_export_overlay_png_cached_writes_into_cache_dir(tmp_path):
    src = tmp_path / "src.tif"
    _write_raster(src, np.zeros((3, 4, 4), dtype="float32"))
    cache_dir = tmp_path / "cache"

    result = export_overlay_png_cached(src, cache_dir)

    assert result.parent == cache_dir
    assert result.suffix == ".png"
    assert result.exists()


def test_export_overlay_png_cached_reuses_existing_file(tmp_path):
    src = tmp_path / "src.tif"
    _write_raster(src, np.zeros((3, 4, 4), dtype="float32"))
    cache_dir = tmp_path / "cache"

    first = export_overlay_png_cached(src, cache_dir)
    mtime_before = first.stat().st_mtime_ns

    second = export_overlay_png_cached(src, cache_dir)

    assert second == first
    assert second.stat().st_mtime_ns == mtime_before


def test_export_overlay_png_cached_ignores_cache_when_disabled(tmp_path, monkeypatch):
    import eo_art.forge3d_pipes.prep.overlay_image as overlay_image

    src = tmp_path / "src.tif"
    _write_raster(src, np.zeros((3, 4, 4), dtype="float32"))
    cache_dir = tmp_path / "cache"

    calls = []
    original = overlay_image.export_overlay_png

    def _counting(s, d):
        calls.append(s)
        return original(s, d)

    monkeypatch.setattr(overlay_image, "export_overlay_png", _counting)

    export_overlay_png_cached(src, cache_dir)
    export_overlay_png_cached(src, cache_dir, use_cache=False)

    assert len(calls) == 2
