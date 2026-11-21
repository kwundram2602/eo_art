"""AROSICS co-registration of LR rasters on synthetic texture."""

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, shift

from sentinel_sr import cli, coreg

SIZE = 640
BANDS = ("B02", "B03", "B04", "B08")


def _texture(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # AROSICS needs real high-frequency texture: on noise smoothed with
    # sigma >= 2 it underestimates shifts several-fold (measured).
    base = gaussian_filter(rng.random((SIZE + 40, SIZE + 40)), 0.5)
    return (base - base.min()) / (base.max() - base.min()) * 0.3 + 0.05


def _write(path, arr2d, dx=0.0, dy=0.0, cloud=None):
    moved = shift(arr2d, (dy, dx), order=3, mode="nearest")[20:-20, 20:-20]
    data = np.stack([moved * (1 + 0.1 * b) for b in range(len(BANDS))]).astype(
        "float32"
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=len(BANDS),
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000, 5200000, 10, 10),
        nodata=np.nan,
    ) as ds:
        ds.write(data)
        for i, b in enumerate(BANDS, start=1):
            ds.set_band_description(i, b)
    scl = path.with_name(path.stem + "_scl.tif")
    scl_data = np.full((1, SIZE, SIZE), 4, dtype="uint8")
    if cloud is not None:
        scl_data[0][cloud] = 9
    with rasterio.open(
        scl,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=1,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(500000, 5200000, 10, 10),
        nodata=0,
    ) as ds:
        ds.write(scl_data)
    return path, scl


def test_match_band_prefers_nir_and_maps_band_ids():
    names = coreg._band_names(("B02", "B03", "B04", "B08", "B11"), 5)
    assert names == ["blue", "green", "red", "nir", "B11"]
    assert coreg.match_band(names) == "nir"
    with pytest.raises(ValueError):
        coreg.match_band(["B05", "B11"])


def test_unique_times_split_repeated_dates():
    t = coreg._unique_times(["2023-05-01", "2023-05-01", "2023-06-01"])
    assert len(set(t)) == 3


def test_shifted_scenes_are_aligned_to_the_anchor(tmp_path):
    tex = _texture()
    cloud = np.zeros((SIZE, SIZE), dtype=bool)
    cloud[:100, :100] = True
    shifts = {
        "2023-04-01": (0.0, 0.0),
        "2023-05-01": (1.5, -0.7),
        "2023-06-01": (-1.0, 1.2),
    }
    jobs = []
    for date, (dx, dy) in shifts.items():
        src, scl = _write(
            tmp_path / f"lr_{date}.tif",
            tex,
            dx,
            dy,
            cloud=cloud if date == "2023-06-01" else None,
        )
        jobs.append(
            coreg.CoregJob(str(src), str(tmp_path / f"cr_{date}.tif"), date, str(scl))
        )

    results = coreg.coregister_jobs(jobs, anchor="2023-04-01", grid_size=3)

    by_date = {r.date: r for r in results}
    assert all(r.dst is not None for r in results)
    for date, (dx, dy) in shifts.items():
        r = by_date[date]
        # The correction undoes the synthetic shift (sign: AROSICS moves the
        # target back onto the reference).
        assert abs(abs(r.shift_x_px) - abs(dx)) < 0.2, (date, r)
        assert abs(abs(r.shift_y_px) - abs(dy)) < 0.2, (date, r)
    assert by_date["2023-06-01"].cloud_percent == pytest.approx(100 * 100**2 / 640**2)

    inner = (slice(20, -20), slice(20, -20))
    with rasterio.open(tmp_path / "cr_2023-04-01.tif") as ref:
        b = ref.read(4)[inner]
    with rasterio.open(tmp_path / "lr_2023-05-01.tif") as raw:
        before = np.nanmean(np.abs(raw.read(4)[inner] - b))
    with rasterio.open(tmp_path / "cr_2023-05-01.tif") as ds:
        after = np.nanmean(np.abs(ds.read(4)[inner] - b))
        assert ds.descriptions == BANDS
    # Residual is cubic-interpolation error on near-white noise.
    assert after < 0.5 * before


def test_cli_coreg_jobs_writes_report(tmp_path, monkeypatch):
    jobs_path = tmp_path / "coreg_jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "src": "a.tif",
                        "dst": "a_cr.tif",
                        "date": "2023-01-01",
                        "scl": None,
                    },
                    {"src": "b.tif", "dst": "b_cr.tif", "date": "2023-02-01"},
                ]
            }
        )
    )
    seen = {}

    def fake(jobs, anchor, grid_size, scl_classes):
        seen.update(n=len(jobs), anchor=anchor, grid_size=grid_size)
        return [coreg.CoregResult(j.src, j.dst, j.date, 0.5, -0.5, 1.0) for j in jobs]

    monkeypatch.setattr(coreg, "coregister_jobs", fake)
    assert cli.main(["--coreg-jobs", str(jobs_path), "--anchor", "first"]) == 0
    assert seen == {"n": 2, "anchor": "first", "grid_size": 7}
    report = json.loads((tmp_path / "coreg.json").read_text())
    assert report["scenes"][1]["shift_x_px"] == 0.5


def test_cli_scene_modes_still_require_scene_args(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--lr-only"])
    assert "--lat" in capsys.readouterr().err
