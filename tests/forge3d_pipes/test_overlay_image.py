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
    """An integer source is unchanged by the new export path: still clipped
    to [0, 255] rather than percentile-stretched, plus a new opaque alpha."""
    src = tmp_path / "src.tif"
    bands = np.stack(
        [
            np.full((4, 4), 10, dtype="uint8"),
            np.full((4, 4), 128, dtype="uint8"),
            np.full((4, 4), 250, dtype="uint8"),
        ]
    )
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    result = export_overlay_png(src, dst)

    assert result == dst
    pixels = iio.imread(dst)
    assert pixels.shape == (4, 4, 4)
    assert tuple(pixels[0, 0]) == (10, 128, 250, 255)


def test_export_overlay_png_from_single_band_raster_is_grayscale_rgb(tmp_path):
    src = tmp_path / "src.tif"
    bands = np.stack([np.full((4, 4), 77, dtype="uint8")])
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    export_overlay_png(src, dst)

    pixels = iio.imread(dst)
    assert pixels.shape == (4, 4, 4)
    assert tuple(pixels[0, 0]) == (77, 77, 77, 255)


def test_export_overlay_png_clips_out_of_range_values(tmp_path):
    """Integer sources are clipped, not stretched -- an int band never
    carries reflectance-scaled data in this pipeline (see
    export_overlay_png's docstring), so out-of-range samples are simply an
    error to clamp, not a signal to rescale the whole image around."""
    src = tmp_path / "src.tif"
    bands = np.stack(
        [
            np.full((4, 4), -50, dtype="int16"),
            np.full((4, 4), 128, dtype="int16"),
            np.full((4, 4), 999, dtype="int16"),
        ]
    )
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    export_overlay_png(src, dst)

    pixels = iio.imread(dst)
    assert tuple(pixels[0, 0]) == (0, 128, 255, 255)


def test_export_overlay_png_uses_only_first_three_bands(tmp_path):
    src = tmp_path / "src.tif"
    bands = np.stack([np.full((4, 4), v, dtype="uint8") for v in range(5)])
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    export_overlay_png(src, dst)

    pixels = iio.imread(dst)
    assert pixels.shape == (4, 4, 4)
    assert tuple(pixels[0, 0]) == (0, 1, 2, 255)


def test_export_overlay_png_stretches_float_reflectance_data_to_full_range(tmp_path):
    """The pre-Task-3 bug: `np.clip(bands, 0, 255)` on 0-1-ranged reflectance
    data collapses everything to 0 or 1 -- solid black. A floating-point
    source must instead be percentile-stretched to actually use [0, 255]."""
    src = tmp_path / "src.tif"
    rows, cols = np.mgrid[0:8, 0:8]
    band = (0.1 + 0.6 * (rows + cols) / 14.0).astype("float32")  # spans ~0.1-0.7
    bands = np.stack([band, band, band])
    _write_raster(src, bands)

    dst = tmp_path / "out.png"
    export_overlay_png(src, dst)

    pixels = iio.imread(dst)
    assert pixels.shape == (8, 8, 4)
    rgb = pixels[..., :3]
    # Today's unconditional clip would produce an all-zero image here (every
    # source value is < 1.0); the fix must spread the real value range
    # across close to the full 0-255 span instead.
    assert rgb.max() >= 240
    assert rgb.min() <= 15
    assert (pixels[..., 3] == 255).all()


def test_export_overlay_png_nodata_does_not_influence_stretch_bounds(tmp_path):
    """A huge nodata sentinel sitting in the pooled percentile computation
    would compress the real data into a narrow sliver of the output range;
    only valid pixels may contribute to the stretch bounds."""
    count, height, width = 1, 4, 4
    band = np.full((height, width), 0.5, dtype="float32")
    band[0, 0] = -9999.0  # nodata corner

    src = tmp_path / "src.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs="EPSG:4326",
        nodata=-9999.0,
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(band, 1)

    out_path = tmp_path / "out.png"
    export_overlay_png(src, out_path)

    pixels = iio.imread(out_path)
    # A uniform valid value stretches to a single flat level (the fallback
    # (lo, hi) in _percentile_bounds), not the near-zero level -9999 would
    # drag it to if it were allowed into the percentile pool.
    valid = pixels[1:, 1:, :3]
    assert (valid == valid[0, 0]).all()
    assert pixels[0, 0, 3] == 0  # nodata corner is transparent
    assert pixels[1, 1, 3] == 255


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
