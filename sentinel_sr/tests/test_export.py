"""export_superres_tifs with cubo and the SR model stubbed out."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from sentinel_sr import rgbn

# (STAC id, acquisition time, scene cloud cover)
ITEMS = [
    # Feb: one overflight, two tiles x two baselines.
    ("S2A_MSIL2A_20230214T105141_R051_T32ULE_20240815T232641", "2023-02-14", 2.78),
    ("S2A_MSIL2A_20230214T105141_R051_T32ULE_20230214T221915", "2023-02-14", 2.77),
    ("S2A_MSIL2A_20230214T105141_R051_T31UGV_20240815T232641", "2023-02-14", 2.08),
    ("S2A_MSIL2A_20230214T105141_R051_T31UGV_20230215T004018", "2023-02-14", 2.07),
    # Apr: one overflight, two baselines.
    ("S2A_MSIL2A_20230402T103621_R008_T32ULE_20240829T021638", "2023-04-02", 2.73),
    ("S2A_MSIL2A_20230402T103621_R008_T32ULE_20230402T214913", "2023-04-02", 2.72),
    # Mar: only from a tile in UTM 31 while the grid is UTM 32 -> skipped.
    ("S2A_MSIL2A_20230309T103631_R008_T31UGV_20241105T003939", "2023-03-09", 0.02),
    # May: two overflights.
    ("S2B_MSIL2A_20230505T103629_R008_T32ULE_20240901T000000", "2023-05-05", 9.0),
    ("S2A_MSIL2A_20230520T103631_R008_T32ULE_20240901T000000", "2023-05-20", 1.0),
]
BANDS = ("B02", "B03")
# Cloudy SCL pixels (class 9) out of the 16 in each 4x4 chip; others are 4.
SCL_CLOUDY_PIXELS = {
    "S2A_MSIL2A_20230402T103621_R008_T32ULE_20240829T021638": 16,  # 100 %
    "S2A_MSIL2A_20230520T103631_R008_T32ULE_20240901T000000": 1,  # 6.25 %
}


@pytest.fixture
def stubbed(monkeypatch):
    def create(**kwargs):
        n = len(ITEMS)
        bands = kwargs["bands"]
        data = np.full((n, len(bands), 4, 4), 1000.0, dtype="float32")
        if "SCL" in bands:
            scl = data[:, bands.index("SCL")]
            scl[:] = 4.0
            for t, (item_id, _, _) in enumerate(ITEMS):
                scl[t].flat[: SCL_CLOUDY_PIXELS.get(item_id, 0)] = 9.0
        da = xr.DataArray(
            data,
            dims=("time", "band", "y", "x"),
            coords={
                "time": np.array([t for _, t, _ in ITEMS], dtype="datetime64[ns]"),
                "band": list(bands),
                "x": np.arange(4) * 10.0 + 500000.0,
                "y": 5200000.0 - np.arange(4) * 10.0,
                "id": ("time", [i for i, _, _ in ITEMS]),
                "eo:cloud_cover": ("time", [c for _, _, c in ITEMS]),
            },
            attrs={"epsg": 32632},
        )
        return da

    def upsample(x):
        return torch.nn.functional.interpolate(x, scale_factor=4)

    model = SimpleNamespace(
        item=SimpleNamespace(
            properties={
                "mlm:input": [{"input": {"shape": [1, 2, 4, 4]}}],
                "mlm:output": [{"result": {"shape": [1, 2, 16, 16]}}],
            }
        ),
        compiled_model=lambda device: upsample,
    )
    monkeypatch.setattr(rgbn.cubo, "create", create)
    monkeypatch.setattr(rgbn.mlstac, "load", lambda path: model)
    monkeypatch.setattr(rgbn, "_patch_vae_attention_for_low_vram", lambda: None)
    monkeypatch.setattr(rgbn, "_patch_swin_for_low_vram", lambda: None)


def _export(tmp_path, **kwargs):
    return rgbn.export_superres_tifs(
        lat=50.0,
        lon=8.0,
        start_date="2023-01-01",
        end_date="2023-06-30",
        output_dir=tmp_path,
        model_path="/m",
        bands=BANDS,
        device=torch.device("cpu"),
        **kwargs,
    )


def test_plain_run_skips_duplicate_overflights(stubbed, tmp_path):
    result = _export(tmp_path, max_items=3)

    assert [s.path.name for s in result.scenes] == [
        "superres_S2A_MSIL2A_20230214T105141_R051_T32ULE_20240815T232641.tif",
        "superres_S2A_MSIL2A_20230402T103621_R008_T32ULE_20240829T021638.tif",
        "superres_S2B_MSIL2A_20230505T103629_R008_T32ULE_20240901T000000.tif",
    ]


def test_timeseries_one_scene_per_month_ignoring_max_items(stubbed, tmp_path, capsys):
    result = _export(tmp_path, interval="month", max_items=1)

    assert [(s.datetime, s.cloud_cover) for s in result.scenes] == [
        ("2023-02-14", 2.78),
        ("2023-04-02", 2.73),
        ("2023-05-20", 1.0),
    ]
    assert result.scenes[0].path.name == (
        "superres_2023-02-14_S2A_MSIL2A_20230214T105141_R051_T32ULE_20240815T232641.tif"
    )
    assert result.skipped_bins == ["2023-01-01", "2023-03-01", "2023-06-01"]
    out = capsys.readouterr().out
    assert "9 scenes, 4 after removing duplicates" in out
    assert (
        "Skipped 1 acquisition(s) only available from tiles outside EPSG:32632" in out
    )
    assert "ignoring max_items=1" in out


def test_existing_scene_is_kept_unless_overwrite(stubbed, tmp_path, capsys):
    _export(tmp_path, interval="month")
    capsys.readouterr()

    _export(tmp_path, interval="month")
    assert capsys.readouterr().out.count("Kept existing") == 3

    _export(tmp_path, interval="month", overwrite=True)
    assert "Kept existing" not in capsys.readouterr().out


def test_scl_check_drops_cloudy_aoi_and_ranks_by_aoi_cloud(stubbed, tmp_path, capsys):
    result = _export(tmp_path, interval="month", max_aoi_cloud=10.0)

    # Apr is 100 % cloud in the AOI -> gone. In May the tile-wide cover favours
    # 05-20 (1 % vs 9 %), but in the AOI 05-05 is clear and 05-20 is not.
    assert [(s.datetime, s.aoi_cloud) for s in result.scenes] == [
        ("2023-02-14", 0.0),
        ("2023-05-05", 0.0),
    ]
    assert result.skipped_bins == [
        "2023-01-01",
        "2023-03-01",
        "2023-04-01",
        "2023-06-01",
    ]
    assert "1 scene(s) over 10.0% cloud in the AOI dropped" in capsys.readouterr().out


def test_scl_check_in_plain_mode_filters_only(stubbed, tmp_path):
    result = _export(tmp_path, max_aoi_cloud=5.0)

    assert [s.datetime for s in result.scenes] == ["2023-02-14", "2023-05-05"]
    assert all(s.aoi_cloud == 0.0 for s in result.scenes)


def test_skip_dates_fill_only_the_open_intervals(stubbed, tmp_path, capsys):
    # Feb and May already have timesteps from an earlier run.
    result = _export(
        tmp_path, interval="month", skip_dates=["2023-02-01", "2023-05-31"]
    )

    assert [s.datetime for s in result.scenes] == ["2023-04-02"]
    assert result.skipped_bins == ["2023-01-01", "2023-03-01", "2023-06-01"]
    assert "2 interval(s) already filled" in capsys.readouterr().out
    assert not list(tmp_path.glob("superres_2023-02-14_*.tif"))


def test_skip_dates_are_ignored_without_an_interval(stubbed, tmp_path):
    result = _export(tmp_path, skip_dates=["2023-02-01"])

    assert "2023-02-14" in [s.datetime for s in result.scenes]
