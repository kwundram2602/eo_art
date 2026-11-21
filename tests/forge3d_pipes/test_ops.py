import numpy as np
import pytest
import rasterio
from omegaconf.errors import MissingMandatoryValue
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from eo_art.forge3d_pipes.prep import ops, registry
from eo_art.forge3d_pipes.prep.overlay_image import export_overlay_png


def test_ops_are_registered():
    assert registry.get_op("reproject").schema is ops.ReprojectCfg
    assert registry.get_op("scale_to_gsd").schema is ops.ScaleToGsdCfg
    assert registry.get_op("saturate").schema is ops.SaturateCfg
    assert registry.get_op("rgb_stretch").schema is ops.RgbStretchCfg
    assert registry.get_op("align_to").schema is ops.AlignToCfg
    assert registry.get_op("validity_mask").schema is ops.ValidityMaskCfg
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


def test_align_to_matches_reference_grid_exactly(
    synthetic_dem, synthetic_overlay, tmp_path
):
    dst = tmp_path / "aligned.tif"
    result = ops.align_to(
        synthetic_overlay, dst, ops.AlignToCfg(reference=str(synthetic_dem))
    )
    assert result == dst
    with rasterio.open(synthetic_dem) as ref, rasterio.open(dst) as out:
        assert out.crs == ref.crs
        assert out.transform == ref.transform
        assert out.width == ref.width
        assert out.height == ref.height


def test_validity_mask_reflects_source_nodata(tmp_path):
    src = tmp_path / "src.tif"
    band = np.full((4, 4), 0.5, dtype="float32")
    band[0, 0] = -9999.0
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        nodata=-9999.0,
        transform=from_origin(0.0, 1.0, 0.01, 0.01),
    ) as dst:
        dst.write(band, 1)

    out_path = tmp_path / "mask.tif"
    result = ops.validity_mask(src, out_path, ops.ValidityMaskCfg())
    assert result == out_path
    with rasterio.open(out_path) as out:
        assert out.count == 1
        assert out.dtypes[0] == "uint8"
        assert out.nodata is None
        mask = out.read(1)
    assert mask[0, 0] == 0
    assert (mask.ravel()[1:] == 255).all()


def test_validity_mask_is_all_valid_when_source_has_no_nodata(tmp_path):
    """The scenario the black-paint regression hinges on: a source with no
    nodata declared has no invalid pixels *of its own* -- the mask is only
    meaningful once it's aligned onto a larger reference grid."""
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
        dst.write(np.full((4, 4), 0.5, dtype="float32"), 1)

    out_path = tmp_path / "mask.tif"
    ops.validity_mask(src, out_path, ops.ValidityMaskCfg())
    with rasterio.open(out_path) as out:
        assert (out.read(1) == 255).all()


def test_align_to_recouples_independently_reprojected_dem_and_overlay(
    synthetic_dem, synthetic_overlay, tmp_path
):
    """The core regression this task fixes: a DEM and a partially-overlapping
    overlay, each reprojected independently, land on different grids (each
    `reproject` call picks its own destination transform) -- `align_to`
    afterwards must recouple them so the overlay is, once again, exactly the
    DEM's grid."""
    from eo_art.forge3d_pipes.prep.extent import compute_normalized_extent

    dem_reprojected = tmp_path / "dem.tif"
    ops.reproject(synthetic_dem, dem_reprojected, ops.ReprojectCfg(crs="EPSG:32610"))
    overlay_reprojected = tmp_path / "overlay.tif"
    ops.reproject(
        synthetic_overlay, overlay_reprojected, ops.ReprojectCfg(crs="EPSG:32610")
    )

    with (
        rasterio.open(dem_reprojected) as dem_out,
        rasterio.open(overlay_reprojected) as overlay_out,
    ):
        # Confirms the two independent reprojections actually diverged --
        # otherwise the rest of the test would pass trivially.
        assert dem_out.transform != overlay_out.transform

    aligned = tmp_path / "aligned.tif"
    ops.align_to(
        overlay_reprojected, aligned, ops.AlignToCfg(reference=str(dem_reprojected))
    )

    extent = compute_normalized_extent(dem_reprojected, aligned)
    assert extent == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_validity_mask_then_align_to_is_transparent_outside_aligned_overlay_coverage(
    synthetic_dem, synthetic_overlay, tmp_path
):
    """Guards against the black-paint regression, for the case that breaks
    `align_to` alone: `synthetic_overlay` has no nodata declared (a
    plausible plain satellite export), so `align_raster_grid` fills the
    area the overlay never covered with an untagged `0` -- nothing marks it
    invalid. `validity_mask`, captured before alignment and aligned
    alongside the main raster (mirroring pipeline._align_overlay_to_dem),
    is what actually carries the correct answer through."""
    with rasterio.open(synthetic_overlay) as src:
        assert src.nodata is None  # the scenario this test exists to cover

    mask = tmp_path / "mask.tif"
    ops.validity_mask(synthetic_overlay, mask, ops.ValidityMaskCfg())

    aligned = tmp_path / "aligned.tif"
    ops.align_to(
        synthetic_overlay, aligned, ops.AlignToCfg(reference=str(synthetic_dem))
    )
    aligned_mask = tmp_path / "aligned_mask.tif"
    ops.align_to(
        mask,
        aligned_mask,
        ops.AlignToCfg(
            reference=str(synthetic_dem), resampling=ops.ResamplingName.nearest
        ),
    )

    png = tmp_path / "aligned.png"
    export_overlay_png(aligned, png, alpha_src=aligned_mask)
    import imageio.v3 as iio

    pixels = iio.imread(png)
    assert pixels.shape == (32, 32, 4)
    alpha = pixels[..., 3]
    # synthetic_overlay only covers synthetic_dem's south-east quadrant
    # (rows >= 16, cols >= 16) once aligned -- see test_extent.py's
    # test_partial_overlap_clamps_to_terrain_bounds for the same geometry.
    assert (alpha[16:, 16:] == 255).all()
    assert (alpha[:16, :] == 0).all()
    assert (alpha[:, :16] == 0).all()


def test_export_overlay_png_without_alpha_src_ignores_untagged_alignment_fill(
    synthetic_dem, synthetic_overlay, tmp_path
):
    """The negative case proving the above test exercises a real bug:
    without `alpha_src`, export_overlay_png has no way to tell the
    untagged fill from real coverage, so the whole raster reads as opaque."""
    aligned = tmp_path / "aligned.tif"
    ops.align_to(
        synthetic_overlay, aligned, ops.AlignToCfg(reference=str(synthetic_dem))
    )

    png = tmp_path / "aligned.png"
    export_overlay_png(aligned, png)
    import imageio.v3 as iio

    pixels = iio.imread(png)
    assert (pixels[..., 3] == 255).all()


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


@pytest.mark.parametrize(
    "name,expected",
    [
        ("lanczos", Resampling.lanczos),
        ("cubic_spline", Resampling.cubic_spline),
        ("average", Resampling.average),
    ],
)
def test_new_resampling_names_map_to_rasterio_and_survive_reprojection(
    name, expected, synthetic_dem, tmp_path
):
    resampling = ops.ResamplingName[name]
    assert ops._RESAMPLING[resampling] is expected

    dst = tmp_path / "out.tif"
    result = ops.reproject(
        synthetic_dem, dst, ops.ReprojectCfg(crs="EPSG:32610", resampling=resampling)
    )
    with rasterio.open(result) as src:
        assert src.crs.to_string() == "EPSG:32610"
        assert src.count == 1


def test_a_config_using_each_new_resampling_name_loads(tmp_path):
    for name in ("lanczos", "cubic_spline", "average"):
        _, cfg = registry.validate_entry(
            {"op": "reproject", "crs": "EPSG:32610", "resampling": name}
        )
        assert cfg.resampling is ops.ResamplingName[name]


def test_auto_promotes_int16_dem_to_float32_under_interpolating_resampling(
    synthetic_dem_int16, tmp_path
):
    """The core regression: cubic-resampling an int16 DEM under `auto` must
    both promote to float32 and actually recover fractional elevation
    values -- writing the interpolated result back to int16 truncates it
    right back into the terracing this policy exists to prevent."""
    dst = tmp_path / "out.tif"
    ops.reproject(
        synthetic_dem_int16,
        dst,
        ops.ReprojectCfg(crs="EPSG:32610", resampling=ops.ResamplingName.cubic),
    )
    with rasterio.open(dst) as out:
        assert out.dtypes[0] == "float32"
        auto_distinct = len(np.unique(out.read(1)))

    preserved = tmp_path / "preserved.tif"
    ops.reproject(
        synthetic_dem_int16,
        preserved,
        ops.ReprojectCfg(
            crs="EPSG:32610",
            resampling=ops.ResamplingName.cubic,
            dtype=ops.DtypeMode.preserve,
        ),
    )
    with rasterio.open(preserved) as out:
        assert out.dtypes[0] == "int16"
        preserved_distinct = len(np.unique(out.read(1)))

    assert auto_distinct > preserved_distinct * 2


def test_auto_does_not_promote_nearest_resampling(synthetic_dem_int16, tmp_path):
    dst = tmp_path / "out.tif"
    ops.reproject(
        synthetic_dem_int16,
        dst,
        ops.ReprojectCfg(crs="EPSG:32610", resampling=ops.ResamplingName.nearest),
    )
    with rasterio.open(dst) as out:
        assert out.dtypes[0] == "int16"


def test_auto_leaves_float_sources_unchanged(synthetic_dem, tmp_path):
    dst = tmp_path / "out.tif"
    ops.reproject(
        synthetic_dem,
        dst,
        ops.ReprojectCfg(crs="EPSG:32610", resampling=ops.ResamplingName.cubic),
    )
    with rasterio.open(dst) as out:
        assert out.dtypes[0] == "float32"


def test_preserve_keeps_source_dtype_even_when_interpolating(
    synthetic_dem_int16, tmp_path
):
    dst = tmp_path / "out.tif"
    ops.reproject(
        synthetic_dem_int16,
        dst,
        ops.ReprojectCfg(
            crs="EPSG:32610",
            resampling=ops.ResamplingName.cubic,
            dtype=ops.DtypeMode.preserve,
        ),
    )
    with rasterio.open(dst) as out:
        assert out.dtypes[0] == "int16"


def test_float32_policy_always_promotes_even_under_nearest(
    synthetic_dem_int16, tmp_path
):
    dst = tmp_path / "out.tif"
    ops.reproject(
        synthetic_dem_int16,
        dst,
        ops.ReprojectCfg(
            crs="EPSG:32610",
            resampling=ops.ResamplingName.nearest,
            dtype=ops.DtypeMode.float32,
        ),
    )
    with rasterio.open(dst) as out:
        assert out.dtypes[0] == "float32"


def test_nodata_survives_promotion(synthetic_dem_int16, tmp_path):
    dst = tmp_path / "out.tif"
    ops.reproject(
        synthetic_dem_int16,
        dst,
        ops.ReprojectCfg(crs="EPSG:32610", resampling=ops.ResamplingName.cubic),
    )
    with rasterio.open(dst) as out:
        assert out.dtypes[0] == "float32"
        assert out.nodata == pytest.approx(-32768.0)


def test_auto_promoted_uint8_overlay_still_round_trips_through_export(tmp_path):
    """The documented overlay side effect of `auto`: reprojecting a uint8
    overlay with an interpolating resampler promotes it to float32.
    export_overlay_png must still read that float32, 0-255-ranged output
    correctly rather than assuming an integer source."""
    r = np.array([[10, 200], [50, 250]], dtype="uint8")
    g = np.array([[20, 190], [60, 240]], dtype="uint8")
    b = np.array([[30, 180], [70, 230]], dtype="uint8")
    src = tmp_path / "overlay.tif"
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

    reprojected = tmp_path / "reprojected.tif"
    ops.reproject(
        src,
        reprojected,
        ops.ReprojectCfg(crs="EPSG:32610", resampling=ops.ResamplingName.bilinear),
    )
    with rasterio.open(reprojected) as out:
        assert out.dtypes[0] == "float32"

    png = tmp_path / "overlay.png"
    export_overlay_png(reprojected, png)
    import imageio.v3 as iio

    pixels = iio.imread(png)
    assert pixels.dtype == np.uint8
    assert pixels.shape[2] == 4
    assert (pixels[..., 3] == 255).all()  # fully valid source -> fully opaque
    # The reprojected float32 values are percentile-stretched (they are
    # display-ranged, not reflectance data, but the dtype-driven rule can't
    # tell the two apart -- see export_overlay_png's docstring), so the
    # output no longer matches the pre-reproject values pixel-for-pixel.
    # What must survive is that the round trip doesn't collapse to
    # near-black or a single flat value -- the uint8 -> float32 -> uint8
    # path stays real, varied color data.
    rgb = pixels[..., :3]
    assert rgb.max() > 200
    assert int(rgb.max()) > int(rgb.min())


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
        ops.RgbStretchCfg(
            bands=(3, 1, 2), lower_percentile=0.0, upper_percentile=100.0
        ),
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
    ops.rgb_stretch(
        src, dst, ops.RgbStretchCfg(lower_percentile=0.0, upper_percentile=100.0)
    )

    with rasterio.open(dst) as out:
        r = out.read(1)
    assert r.tolist() == [
        [0, 255],
        [0, 255],
    ]  # default bands=(3, 2, 1) -> R from band 3


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
    ops.rgb_stretch(
        src, dst, ops.RgbStretchCfg(lower_percentile=0.0, upper_percentile=100.0)
    )

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
            src,
            tmp_path / "out.tif",
            ops.RgbStretchCfg(lower_percentile=90, upper_percentile=10),
        )


def test_rgb_stretch_rejects_out_of_range_band_index(tmp_path):
    band = np.zeros((2, 2), dtype="float32")
    src = tmp_path / "src.tif"
    _write_multiband(src, [band, band])  # only 2 bands

    with pytest.raises(ValueError, match="2 band"):
        ops.rgb_stretch(
            src, tmp_path / "out.tif", ops.RgbStretchCfg()
        )  # default needs band 3


def test_rgb_stretch_fixed_bounds_override_percentiles(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from eo_art.forge3d_pipes.prep.ops import RgbStretchCfg, rgb_stretch

    src = tmp_path / "src.tif"
    data = np.stack([np.linspace(0, 1, 16, dtype="float32").reshape(4, 4)] * 3)
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=3,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(0, 40, 10, 10),
    ) as dst:
        dst.write(data)

    out = rgb_stretch(
        src, tmp_path / "out.tif", RgbStretchCfg(lower_value=0.0, upper_value=2.0)
    )

    with rasterio.open(out) as result:
        band = result.read(1)
    # 1.0 is half of [0, 2], not the top of a percentile stretch.
    assert band.max() == 127


def test_rgb_stretch_needs_both_fixed_bounds(tmp_path):
    from eo_art.forge3d_pipes.prep.ops import RgbStretchCfg, rgb_stretch

    with pytest.raises(ValueError, match="both lower_value and upper_value"):
        rgb_stretch(
            tmp_path / "x.tif", tmp_path / "y.tif", RgbStretchCfg(lower_value=0.1)
        )


def test_stretch_bounds_over_pools_all_rasters(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from eo_art.forge3d_pipes.prep.ops import RgbStretchCfg, stretch_bounds_over

    paths = []
    for i, value in enumerate((0.0, 1.0)):
        path = tmp_path / f"t{i}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=3,
            dtype="float32",
            crs="EPSG:32632",
            transform=from_origin(0, 40, 10, 10),
        ) as dst:
            dst.write(np.full((3, 4, 4), value, "float32"))
        paths.append(path)

    lo, hi = stretch_bounds_over(
        paths, RgbStretchCfg(lower_percentile=0, upper_percentile=100)
    )

    assert (lo, hi) == (0.0, 1.0)
