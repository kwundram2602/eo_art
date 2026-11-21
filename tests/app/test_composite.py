import io
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import rasterio
from PIL import Image

from eo_art.app import composite as cp


def test_resolve_bands_from_sr_descriptions():
    descriptions = ["blue", "green", "red", "rededge_B05", "nir", "nir08", "swir22"]

    bands = cp.resolve_bands(descriptions, len(descriptions))

    assert bands == {
        "B02": 0,
        "B03": 1,
        "B04": 2,
        "B05": 3,
        "B08": 4,
        "B8A": 5,
        "B12": 6,
    }


def test_resolve_bands_accepts_raw_ids():
    assert cp.resolve_bands(["B4", "b8a", "B11"], 3) == {"B04": 0, "B8A": 1, "B11": 2}


def test_resolve_bands_falls_back_to_default_order():
    assert cp.resolve_bands([None] * 10, 10) == {
        band: i for i, band in enumerate(cp.DEFAULT_BANDS)
    }


def test_load_raster_masks_nan_and_keeps_georeference(tmp_path, make_sr_tif):
    raster = cp.load_raster(make_sr_tif(tmp_path / "a_sr.tif"))

    assert raster.data.shape == (10, 8, 16)
    assert set(raster.bands) == set(cp.DEFAULT_BANDS)
    assert np.isnan(raster.band("B02")[0, 0])
    assert raster.band("B04")[1, 0] == pytest.approx(0.06)
    assert raster.crs.to_epsg() == 32632


def test_load_raster_from_bytes(tmp_path, make_sr_tif):
    path = make_sr_tif(tmp_path / "a_sr.tif")

    raster = cp.load_raster(path.read_bytes())

    assert raster.band("B12")[1, 0] == pytest.approx(0.2)


def test_compute_index_ndvi(tmp_path, make_sr_tif):
    raster = cp.load_raster(make_sr_tif(tmp_path / "a_sr.tif"))

    ndvi = cp.compute_index(raster, "NDVI")

    # B08 = 0.14, B04 = 0.06 at the ramp's start.
    assert ndvi[1, 0] == pytest.approx(0.4)
    assert np.isnan(ndvi[0, 0])


def test_compute_index_zero_division_is_nan():
    zeros = np.zeros((2, 1, 1), dtype="float32")
    raster = cp.Raster(zeros, {"B08": 0, "B04": 1}, None, rasterio.Affine.identity())

    assert np.isnan(cp.compute_index(raster, "NDVI")).all()


def test_preset_missing_bands(tmp_path, make_sr_tif):
    raster = cp.load_raster(make_sr_tif(tmp_path / "a_sr.tif"))
    by_label = {p.label: p for p in cp.PRESETS}

    assert cp.missing_bands(raster, by_label["Bathymetric"].channels) == ["B01"]
    assert cp.missing_bands(raster, by_label["Color infrared"].channels) == []
    assert cp.missing_bands(raster, by_label["SWIR-NDVI-MNDWI"].channels) == []
    assert by_label["Color infrared"].notation == "B8-B4-B3"


def _ramp() -> np.ndarray:
    return np.linspace(0.0, 1.0, 101, dtype="float32").reshape(1, -1)


def test_percentile_stretch_clips_to_bounds():
    (unit,) = cp.normalize([_ramp()], cp.Stretch(lower=10, upper=90))

    assert unit[0, 0] == 0.0
    assert unit[0, 50] == pytest.approx(0.5)
    assert unit[0, -1] == 1.0


def test_shared_bounds_pool_all_channels():
    low, high = _ramp() * 0.5, _ramp() * 0.5 + 0.5

    shared = cp.bounds([low, high], cp.Stretch(lower=0, upper=100, shared=True))
    separate = cp.bounds([low, high], cp.Stretch(lower=0, upper=100, shared=False))

    assert shared == [(0.0, 1.0), (0.0, 1.0)]
    assert separate == [(0.0, 0.5), (0.5, 1.0)]


def test_minmax_stretch_uses_manual_range():
    stretch = cp.Stretch(method=cp.Method.MINMAX, manual=((0.25, 0.75),))

    (unit,) = cp.normalize([_ramp()], stretch)

    assert unit[0, 25] == pytest.approx(0.0)
    assert unit[0, 50] == pytest.approx(0.5)
    assert unit[0, 75] == pytest.approx(1.0)


def test_minmax_stretch_rejects_wrong_range_count():
    stretch = cp.Stretch(method=cp.Method.MINMAX, manual=((0, 1), (0, 1)))

    with pytest.raises(ValueError, match="minmax"):
        cp.normalize([_ramp()] * 3, stretch)


def test_equalize_flattens_a_skewed_distribution():
    skewed = (_ramp() ** 4).astype("float32")

    (unit,) = cp.normalize([skewed], cp.Stretch(method=cp.Method.EQUALIZE))

    assert np.all(np.diff(unit[0]) >= 0)
    assert unit[0, 50] == pytest.approx(0.5, abs=0.01)


def test_gamma_brightens_midtones():
    (unit,) = cp.normalize([_ramp()], cp.Stretch(lower=0, upper=100, gamma=2.0))

    assert unit[0, 25] == pytest.approx(0.5)


def test_rgb8_and_png_roundtrip(tmp_path, make_sr_tif):
    raster = cp.load_raster(make_sr_tif(tmp_path / "a_sr.tif"))
    channels = [raster.band(b) for b in ("B04", "B03", "B02")]

    rgb = cp.to_rgb8(channels, cp.Stretch())
    decoded = iio.imread(cp.encode_png(rgb))

    assert rgb.dtype == np.uint8 and rgb.shape == (8, 16, 3)
    assert (rgb[0, 0] == 0).all()  # NaN pixel renders black
    np.testing.assert_array_equal(decoded, rgb)


def test_colormap8_shape(tmp_path, make_sr_tif):
    raster = cp.load_raster(make_sr_tif(tmp_path / "a_sr.tif"))

    rgb = cp.to_colormap8(cp.compute_index(raster, "NDVI"), cp.Stretch(), "RdYlGn")

    assert rgb.dtype == np.uint8 and rgb.shape == (8, 16, 3)


def test_indices_geotiff_keeps_grid_and_names(tmp_path, make_sr_tif):
    src = make_sr_tif(tmp_path / "a_sr.tif")
    raster = cp.load_raster(src)

    out = tmp_path / "indices.tif"
    out.write_bytes(cp.indices_geotiff(raster, ["NDVI", "MNDWI"]))

    with rasterio.open(src) as a, rasterio.open(out) as b:
        assert b.count == 2
        assert b.dtypes == ("float32", "float32")
        assert b.descriptions == ("NDVI", "MNDWI")
        assert b.crs == a.crs
        assert b.transform == a.transform
        assert b.read(1)[1, 0] == pytest.approx(0.4)


def test_pooled_reference_keeps_brightness_differences_between_frames():
    dark = _ramp() * 0.5
    bright = _ramp() * 0.5 + 0.5
    stretch = cp.Stretch(lower=0, upper=100)

    reference = cp.pooled_reference([[dark], [bright]])
    (dark_unit,) = cp.normalize([dark], stretch, reference)
    (bright_unit,) = cp.normalize([bright], stretch, reference)
    (alone,) = cp.normalize([dark], stretch)

    assert dark_unit.max() == pytest.approx(0.5)
    assert bright_unit.min() == pytest.approx(0.5)
    # Stretched on its own, the dark frame would fill the whole range.
    assert alone.max() == pytest.approx(1.0)


def test_pooled_reference_with_equalize_and_shape_checks():
    dark, bright = _ramp() * 0.5, _ramp() * 0.5 + 0.5
    reference = cp.pooled_reference([[dark], [bright]])

    (unit,) = cp.normalize([dark], cp.Stretch(method=cp.Method.EQUALIZE), reference)

    assert unit.max() == pytest.approx(0.5, abs=0.01)
    with pytest.raises(ValueError, match="no frames"):
        cp.pooled_reference([])


def test_reference_none_changes_nothing():
    stretch = cp.Stretch(lower=10, upper=90)

    assert np.array_equal(
        cp.normalize([_ramp()], stretch)[0],
        cp.normalize([_ramp()], stretch, None)[0],
    )


def test_frame_date():
    assert cp.frame_date(Path("out/x/valley_sr_2023-02-14.tif")) == "2023-02-14"
    assert cp.frame_date(Path("out/valley_sr.tif")) is None


@pytest.mark.parametrize("loop", [True, False])
def test_encode_gif_roundtrip(loop):
    frames = [np.full((4, 6, 3), value, dtype="uint8") for value in (0, 120, 255)]

    data = cp.encode_gif(frames, seconds=0.5, loop=loop)

    with Image.open(io.BytesIO(data)) as gif:
        assert gif.n_frames == 3
        assert gif.size == (6, 4)
        assert gif.info["duration"] == 500
        assert ("loop" in gif.info) is loop
        gif.seek(2)
        assert np.array(gif.convert("RGB"))[0, 0].tolist() == [255, 255, 255]


def test_timeseries_frames_only_dated_steps_in_date_order(tmp_path, make_sr_tif):
    folder = tmp_path / "alps_sr_timeseries"
    feb = make_sr_tif(folder / "alps_sr_2023-02-01.tif")
    jan = make_sr_tif(folder / "alps_sr_2023-01-01.tif")
    make_sr_tif(folder / "alps_sr_2023-01-01_indices.tif")

    assert cp.timeseries_frames(folder) == [jan, feb]
    assert cp.frame_label(jan) == "2023-01-01"
    assert cp.slug("Color · NDVI (B8)") == "color-ndvi-b8"


def test_build_gif_skips_frames_on_another_grid(tmp_path, make_sr_tif):
    folder = tmp_path / "ts"
    jan = make_sr_tif(folder / "alps_sr_2023-01-01.tif")
    feb = make_sr_tif(folder / "alps_sr_2023-02-01.tif")
    other = tmp_path / "other" / "alps_sr_2023-03-01.tif"
    make_sr_tif(other)
    with rasterio.open(other) as src:
        profile, data = src.profile, src.read()
    profile.update(width=8)
    with rasterio.open(other, "w", **profile) as dst:
        dst.write(data[:, :, :8])

    steps = []
    result = cp.build_gif(
        [jan, feb, other],
        ("B04", "B03", "B02"),
        cp.Stretch(cp.Method.PERCENTILE, 2.0, 98.0),
        None,
        (8, 16),
        0.5,
        True,
        progress=lambda fraction, text: steps.append(fraction),
    )

    assert result.data is not None
    assert result.skipped == ["2023-03-01 (grid (8, 8))"]
    assert steps == sorted(steps) and steps[-1] < 1.0
    # Both kept frames are identical fixtures, which Pillow merges into one.
    assert Image.open(io.BytesIO(result.data)).format == "GIF"


def test_build_gif_without_matching_frames_returns_none(tmp_path, make_sr_tif):
    frame = make_sr_tif(tmp_path / "alps_sr_2023-01-01.tif", descriptions=None)
    result = cp.build_gif(
        [frame],
        ("NOPE",),
        cp.Stretch(cp.Method.PERCENTILE, 2.0, 98.0),
        "gray",
        (8, 16),
        0.5,
        False,
    )
    assert result.data is None
    assert result.skipped == ["2023-01-01 (no NOPE)"]


def test_histogram_png(tmp_path):
    values = np.linspace(0, 1, 500, dtype="float32").reshape(20, 25)
    png = cp.histogram_png(values, "B04", (0.1, 0.9))
    assert png is not None
    assert Image.open(io.BytesIO(png)).format == "PNG"
    assert cp.histogram_png(np.full((4, 4), np.nan, dtype="float32"), "B04") is None
