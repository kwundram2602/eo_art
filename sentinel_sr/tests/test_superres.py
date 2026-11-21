"""Whole-raster SR with overlapping patches, with a stub ×4 model."""

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from sentinel_sr import superres


def upsample(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def test_fill_nan_nearest_takes_all_bands_from_one_pixel():
    arr = np.array([[[1.0, np.nan, 3.0]], [[10.0, np.nan, 30.0]]])
    mask = np.isnan(arr).any(axis=0)
    filled = superres.fill_nan_nearest(arr, mask)
    assert filled[0, 0, 1] in (1.0, 3.0)
    assert filled[1, 0, 1] == filled[0, 0, 1] * 10


def test_pad_reaches_one_patch_and_reports_offset():
    padded, (top, left) = superres.pad_for_patches(np.zeros((2, 50, 300)), pad=16)
    assert padded.shape == (2, 128, 332)
    assert (top, left) == (39, 16)


def test_superres_array_is_seamless_and_keeps_nan():
    rng = np.random.default_rng(0)
    arr = rng.random((3, 200, 300)).astype("float32")
    arr[:, 10:20, 30:40] = np.nan

    out = superres.superres_array(upsample, arr, torch.device("cpu"), batch_size=4)

    expected = np.repeat(np.repeat(arr, 4, axis=1), 4, axis=2)
    assert out.shape == (3, 800, 1200)
    # Nearest upsampling stitched from overlapping patches must equal the
    # direct upsampling everywhere: no patch or edge offsets.
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_allclose(out[~np.isnan(out)], expected[~np.isnan(expected)])


def test_superres_raster_writes_scaled_grid(tmp_path):
    src = tmp_path / "lr.tif"
    data = np.ones((2, 40, 60), dtype="float32")
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        width=60,
        height=40,
        count=2,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000, 5200000, 10, 10),
        nodata=np.nan,
    ) as ds:
        ds.write(data)
        ds.set_band_description(1, "blue")
        ds.set_band_description(2, "green")

    dst = superres.superres_raster(
        upsample, src, tmp_path / "sr.tif", torch.device("cpu")
    )

    with rasterio.open(dst) as ds:
        assert (ds.width, ds.height, ds.res) == (240, 160, (2.5, 2.5))
        assert ds.transform.c == 500000 and ds.transform.f == 5200000
        assert ds.descriptions == ("blue", "green")
        assert ds.read().min() == pytest.approx(1.0)


def test_superres_jobs_keeps_existing_outputs(tmp_path, monkeypatch):
    dst = tmp_path / "sr.tif"
    dst.write_bytes(b"x")
    monkeypatch.setattr(
        superres, "load_model", lambda *a: pytest.fail("nothing to do, no model load")
    )
    assert superres.superres_jobs([{"src": "a.tif", "dst": str(dst)}], "/m") == [dst]
