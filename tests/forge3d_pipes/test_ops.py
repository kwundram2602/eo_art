import numpy as np
import pytest
import rasterio
from omegaconf.errors import MissingMandatoryValue
from rasterio.transform import from_origin

from eo_art.forge3d_pipes.prep import ops, registry


def test_ops_are_registered():
    assert registry.get_op("reproject").schema is ops.ReprojectCfg
    assert registry.get_op("scale_to_gsd").schema is ops.ScaleToGsdCfg
    assert registry.get_op("saturate").schema is ops.SaturateCfg
    assert registry.get_op("rgb_stretch").schema is ops.RgbStretchCfg
    assert ops.RgbStretchCfg().shared_stretch is True


def test_reproject_changes_crs(synthetic_dem, tmp_path):
    dst = tmp_path / "out.tif"
    result = ops.reproject(synthetic_dem, dst, ops.ReprojectCfg(crs="EPSG:32610"))
    assert result == dst
    with rasterio.open(result) as src:
        assert src.crs.to_string() == "EPSG:32610"
        assert src.width > 0 and src.height > 0
        assert src.count == 1


def test_reproject_preserves_elevation_range(synthetic_dem, tmp_path):
    dst = tmp_path / "out.tif"
    with rasterio.open(synthetic_dem) as src:
        original_max = float(src.read(1).max())
    ops.reproject(synthetic_dem, dst, ops.ReprojectCfg(crs="EPSG:32610"))
    with rasterio.open(dst) as out:
        assert float(out.read(1).max()) == pytest.approx(original_max, rel=0.05)


def test_reproject_requires_crs():
    with pytest.raises(MissingMandatoryValue):
        registry.validate_entry({"op": "reproject"})


def test_reproject_rejects_unknown_resampling():
    from omegaconf.errors import ValidationError

    with pytest.raises(ValidationError):
        registry.validate_entry(
            {"op": "reproject", "crs": "EPSG:32610", "resampling": "quintic"}
        )


def test_scale_to_gsd_changes_resolution(synthetic_dem, tmp_path):
    projected = tmp_path / "utm.tif"
    ops.reproject(synthetic_dem, projected, ops.ReprojectCfg(crs="EPSG:32610"))
    with rasterio.open(projected) as src:
        original_res = src.res[0]

    dst = tmp_path / "scaled.tif"
    target = original_res * 2.0
    ops.scale_to_gsd(projected, dst, ops.ScaleToGsdCfg(target_gsd=target))
    with rasterio.open(dst) as out:
        assert out.res[0] == pytest.approx(target, rel=0.01)


def test_chain_of_both_ops_runs(synthetic_dem, tmp_path):
    result = registry.run_prep_chain(
        synthetic_dem,
        [
            {"op": "reproject", "crs": "EPSG:32610"},
            {"op": "scale_to_gsd", "target_gsd": 200.0},
        ],
        tmp_path / "cache",
    )
    with rasterio.open(result) as src:
        assert src.crs.to_string() == "EPSG:32610"
        # scale_raster_to_gsd rounds to whole pixel counts, so on a small
        # raster the achieved GSD deviates from the target by a few percent:
        # here 12x18 pixels over a 2477x3621m extent yields ~206m, not 200m.
        assert src.res[0] == pytest.approx(200.0, rel=0.05)


def test_reproject_rejects_a_raster_without_a_crs(tmp_path):
    """A CRS-less DEM is plausible first-contact input; fail clearly, not opaquely."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    src = tmp_path / "no_crs.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="float32",
        transform=from_origin(100.0, 200.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.ones((8, 8), dtype="float32"), 1)

    with pytest.raises(ValueError, match="has no CRS"):
        ops.reproject(src, tmp_path / "out.tif", ops.ReprojectCfg(crs="EPSG:32610"))


def _write_rgb(path, r, g, b):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=r.shape[1],
        height=r.shape[0],
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)


def test_saturate_factor_zero_produces_grayscale(tmp_path):
    r = np.full((4, 4), 200.0, dtype="float32")
    g = np.full((4, 4), 100.0, dtype="float32")
    b = np.full((4, 4), 50.0, dtype="float32")
    src = tmp_path / "src.tif"
    _write_rgb(src, r, g, b)

    dst = tmp_path / "out.tif"
    ops.saturate(src, dst, ops.SaturateCfg(factor=0.0))

    with rasterio.open(dst) as out:
        out_r, out_g, out_b = out.read(1), out.read(2), out.read(3)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    assert out_r == pytest.approx(gray, abs=1e-3)
    assert out_g == pytest.approx(gray, abs=1e-3)
    assert out_b == pytest.approx(gray, abs=1e-3)


def test_saturate_factor_one_is_unchanged(tmp_path):
    r = np.full((4, 4), 200.0, dtype="float32")
    g = np.full((4, 4), 100.0, dtype="float32")
    b = np.full((4, 4), 50.0, dtype="float32")
    src = tmp_path / "src.tif"
    _write_rgb(src, r, g, b)

    dst = tmp_path / "out.tif"
    ops.saturate(src, dst, ops.SaturateCfg(factor=1.0))

    with rasterio.open(dst) as out:
        assert out.read(1) == pytest.approx(r, abs=1e-3)
        assert out.read(2) == pytest.approx(g, abs=1e-3)
        assert out.read(3) == pytest.approx(b, abs=1e-3)


def test_saturate_factor_above_one_increases_deviation_from_gray(tmp_path):
    r = np.full((4, 4), 200.0, dtype="float32")
    g = np.full((4, 4), 100.0, dtype="float32")
    b = np.full((4, 4), 50.0, dtype="float32")
    src = tmp_path / "src.tif"
    _write_rgb(src, r, g, b)

    dst = tmp_path / "out.tif"
    ops.saturate(src, dst, ops.SaturateCfg(factor=2.0))

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    with rasterio.open(dst) as out:
        out_r = out.read(1)
    original_deviation = np.abs(r - gray)
    new_deviation = np.abs(out_r - gray)
    assert new_deviation == pytest.approx(original_deviation * 2.0, abs=1e-3)


def test_saturate_requires_at_least_three_bands(tmp_path):
    src = tmp_path / "src.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(np.zeros((4, 4), dtype="float32"), 1)

    with pytest.raises(ValueError, match="at least 3 bands"):
        ops.saturate(src, tmp_path / "out.tif", ops.SaturateCfg(factor=0.5))


def test_saturate_clips_instead_of_wrapping_on_an_integer_dtype(tmp_path):
    """A near-black pixel pushed further from gray by factor>1 must clip to
    0, not wrap around to a bright value -- exactly the bug that turned
    Doubtless Bay's water pink: saturate() ran on rgb_stretch's uint8
    output, (gray + (channel - gray) * factor) went negative, and
    astype("uint8") wrapped -24.5 into 232 instead of clamping to 0."""
    r = np.full((2, 2), 5, dtype="uint8")  # dark, near-black pixel
    g = np.full((2, 2), 6, dtype="uint8")
    b = np.full((2, 2), 250, dtype="uint8")  # near-white, to also test the high end
    src = tmp_path / "src.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)

    dst_path = tmp_path / "out.tif"
    ops.saturate(src, dst_path, ops.SaturateCfg(factor=3.0))

    with rasterio.open(dst_path) as out:
        out_r, out_g, out_b = out.read(1), out.read(2), out.read(3)
    assert (out_r == 0).all()  # clipped to black, not wrapped to a bright value
    assert (out_g == 0).all()
    assert (out_b == 255).all()  # clipped to white on the high end too


def _write_multiband(path, bands, *, nodata=None, dtype="float32"):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands[0].shape[1],
        height=bands[0].shape[0],
        count=len(bands),
        dtype=dtype,
        crs="EPSG:4326",
        nodata=nodata,
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        for index, band in enumerate(bands, start=1):
            dst.write(band.astype(dtype), index)


def test_rgb_stretch_selects_and_reorders_bands(tmp_path):
    # distinguishable spatial patterns per band, so picking "the wrong band"
    # or forgetting to reorder shows up as a different array, not just a
    # different scale.
    band1 = np.array([[0.0, 0.0], [300.0, 300.0]], dtype="float32")  # top/bottom
    band2 = np.array([[0.0, 300.0], [0.0, 300.0]], dtype="float32")  # left/right
    band3 = np.array([[300.0, 0.0], [0.0, 300.0]], dtype="float32")  # anti-diagonal
    src = tmp_path / "src.tif"
    _write_multiband(src, [band1, band2, band3])

    dst = tmp_path / "out.tif"
    ops.rgb_stretch(
        src,
        dst,
        ops.RgbStretchCfg(bands=(3, 1, 2), lower_percentile=0.0, upper_percentile=100.0),
    )

    with rasterio.open(dst) as out:
        assert out.count == 3
        assert out.dtypes[0] == "uint8"
        r, g, b = out.read(1), out.read(2), out.read(3)
    assert r.tolist() == [[255, 0], [0, 255]]  # from band3
    assert g.tolist() == [[0, 0], [255, 255]]  # from band1
    assert b.tolist() == [[0, 255], [0, 255]]  # from band2


def test_rgb_stretch_default_bands_match_sentinel2_b02_b03_b04_order(tmp_path):
    blue = np.full((2, 2), 10.0, dtype="float32")  # band 1 (B02)
    green = np.full((2, 2), 20.0, dtype="float32")  # band 2 (B03)
    red = np.array([[0.0, 100.0], [0.0, 100.0]], dtype="float32")  # band 3 (B04)
    src = tmp_path / "src.tif"
    _write_multiband(src, [blue, green, red])

    dst = tmp_path / "out.tif"
    ops.rgb_stretch(src, dst, ops.RgbStretchCfg(lower_percentile=0.0, upper_percentile=100.0))

    with rasterio.open(dst) as out:
        r = out.read(1)
    assert r.tolist() == [[0, 255], [0, 255]]  # default bands=(3, 2, 1) -> R from band 3


def test_rgb_stretch_clips_outliers_beyond_percentile_range(tmp_path):
    # a 0..99 gradient (99 pixels) plus one extreme outlier: with only ~1% of
    # pixels affected, the 2nd/98th percentiles stay governed by the
    # gradient, not the outlier -- so the outlier clips to 255 instead of
    # dragging the whole stretch's dynamic range down with it.
    band = np.arange(100, dtype="float32").reshape(10, 10)
    band[0, 0] = 10_000.0  # was 0; now the extreme outlier
    src = tmp_path / "src.tif"
    _write_multiband(src, [band, band, band])

    dst = tmp_path / "out.tif"
    ops.rgb_stretch(src, dst, ops.RgbStretchCfg())

    with rasterio.open(dst) as out:
        r = out.read(1)
    assert r[0, 0] == 255  # clipped, not blown out past the uint8 range
    assert r[9, 9] >= 200  # value 99, the brightest non-outlier pixel: near-white
    assert r[0, 1] <= 20  # value 1, the dimmest non-outlier pixel: near-black
    mid_value_pixel = tuple(np.argwhere(band == 50)[0])
    assert 80 <= r[mid_value_pixel] <= 180  # a genuine mid-range stretch, not binary


def test_rgb_stretch_shared_stretch_preserves_relative_band_brightness(tmp_path):
    """Same spatial pattern, different physical magnitude per band (as real
    Sentinel-2 bands do): a shared stretch must keep the brighter band
    brighter, not auto-level each band to the same [0, 255] range -- doing
    the latter is exactly what turned Doubtless Bay's water pink instead of
    dark blue (every channel independently maxed out, erasing the real
    magnitude difference between bands)."""
    ramp = np.arange(16, dtype="float32").reshape(4, 4)
    dim = ramp * 0.5  # half the magnitude of `ramp`, same relative shape
    bright = ramp * 2.0  # double the magnitude

    src = tmp_path / "src.tif"
    _write_multiband(src, [ramp, dim, bright])  # bands 1, 2, 3

    dst = tmp_path / "shared.tif"
    ops.rgb_stretch(
        src,
        dst,
        ops.RgbStretchCfg(
            bands=(1, 2, 3), lower_percentile=0.0, upper_percentile=100.0
        ),
    )
    with rasterio.open(dst) as out:
        r, g, b = out.read(1)[3, 3], out.read(2)[3, 3], out.read(3)[3, 3]
    # `bright` truly has twice the values of `ramp`, and `dim` half -- the
    # shared stretch must preserve that ordering instead of flattening it.
    assert b > r > g
    assert b == 255  # the overall pool's max
    assert 15 < r < 140  # ramp's own max (15) is half the pool's max (30)
    assert g < 70  # dim's own max (7.5) is a quarter of the pool's max

    independent_dst = tmp_path / "independent.tif"
    ops.rgb_stretch(
        src,
        independent_dst,
        ops.RgbStretchCfg(
            bands=(1, 2, 3),
            lower_percentile=0.0,
            upper_percentile=100.0,
            shared_stretch=False,
        ),
    )
    with rasterio.open(independent_dst) as out:
        r2, g2, b2 = out.read(1)[3, 3], out.read(2)[3, 3], out.read(3)[3, 3]
    # independent per-band stretch auto-levels each band to its own max,
    # erasing the real magnitude difference between them.
    assert (r2, g2, b2) == (255, 255, 255)


def test_rgb_stretch_excludes_nodata_from_percentile_computation(tmp_path):
    band = np.array(
        [[0.0, 25.0], [75.0, 100.0]], dtype="float32"
    )  # min/max of the *valid* data
    nodata_band = band.copy()
    nodata_band[0, 0] = -9999.0  # nodata sentinel, must not skew the stretch
    src = tmp_path / "src.tif"
    _write_multiband(src, [nodata_band, nodata_band, nodata_band], nodata=-9999.0)

    dst = tmp_path / "out.tif"
    ops.rgb_stretch(src, dst, ops.RgbStretchCfg(lower_percentile=0.0, upper_percentile=100.0))

    with rasterio.open(dst) as out:
        r = out.read(1)
    # stretched against the valid range [25, 100], not [-9999, 100]
    assert r[0, 1] == 0
    assert r[1, 1] == 255


def test_rgb_stretch_requires_exactly_three_bands_selected(tmp_path):
    band = np.zeros((2, 2), dtype="float32")
    src = tmp_path / "src.tif"
    _write_multiband(src, [band, band, band])

    with pytest.raises(ValueError, match="exactly 3 entries"):
        ops.rgb_stretch(src, tmp_path / "out.tif", ops.RgbStretchCfg(bands=(1, 2)))


def test_rgb_stretch_rejects_invalid_percentile_order(tmp_path):
    band = np.zeros((2, 2), dtype="float32")
    src = tmp_path / "src.tif"
    _write_multiband(src, [band, band, band])

    with pytest.raises(ValueError, match="lower_percentile"):
        ops.rgb_stretch(
            src, tmp_path / "out.tif", ops.RgbStretchCfg(lower_percentile=90, upper_percentile=10)
        )


def test_rgb_stretch_rejects_out_of_range_band_index(tmp_path):
    band = np.zeros((2, 2), dtype="float32")
    src = tmp_path / "src.tif"
    _write_multiband(src, [band, band])  # only 2 bands

    with pytest.raises(ValueError, match="2 band"):
        ops.rgb_stretch(src, tmp_path / "out.tif", ops.RgbStretchCfg())  # default needs band 3
