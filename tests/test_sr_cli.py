import pytest
import yaml

from eo_art import sr_cli


@pytest.fixture
def fake_run_sr(monkeypatch, tmp_path):
    calls = {}
    sr_path = tmp_path / "optical_sr.tif"

    def _run_sr(cfg, out_dir, use_cache):
        calls.update(cfg=cfg, out_dir=out_dir, use_cache=use_cache)
        return sr_path

    monkeypatch.setattr(sr_cli, "run_sr", _run_sr)
    calls["sr_path"] = sr_path
    return calls


@pytest.fixture
def full_acquire_cfg(tmp_path):
    """An acquire config with dem/dtm blocks and no ee_project -- the SR-only
    command must ignore the former and not require the latter."""
    path = tmp_path / "acquire.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "aoi_path": "aoi.gpkg",
                "out_dir": str(tmp_path / "cfg_out"),
                "sentinel_sr_dir": "../sentinel_sr",
                "sentinel": {
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "model_path": "/models/m",
                },
                "dem": {"source": "fabdem", "scale": 30},
                "dtm": {"apply_erosion": True},
            }
        )
    )
    return path


def test_sr_command_ignores_dem_and_prints_path(fake_run_sr, full_acquire_cfg, capsys):
    exit_code = sr_cli.main([str(full_acquire_cfg)])

    assert exit_code == 0
    assert fake_run_sr["cfg"].aoi_path == "aoi.gpkg"
    assert fake_run_sr["out_dir"] == fake_run_sr["cfg"].out_dir
    assert fake_run_sr["use_cache"] is True
    assert str(fake_run_sr["sr_path"]) in capsys.readouterr().out


def test_sr_command_set_out_and_no_cache(fake_run_sr, full_acquire_cfg, tmp_path):
    sr_cli.main(
        [
            str(full_acquire_cfg),
            "--set",
            "aoi_path=other.gpkg",
            "--out",
            str(tmp_path / "o"),
            "--no-cache",
        ]
    )

    assert fake_run_sr["cfg"].aoi_path == "other.gpkg"
    assert fake_run_sr["out_dir"] == str(tmp_path / "o")
    assert fake_run_sr["use_cache"] is False


def test_sr_command_timeseries_prints_folder_and_gaps(
    monkeypatch, full_acquire_cfg, tmp_path, capsys
):
    from eo_art.forge3d_pipes.acquire import TimeseriesResult

    seen = {}

    def _run_sr_timeseries(cfg, out_dir, use_cache):
        seen.update(cfg=cfg, use_cache=use_cache)
        folder = tmp_path / "aoi_sr_timeseries"
        return TimeseriesResult(
            out_dir=folder,
            paths=[folder / "aoi_sr_2023-01-05.tif"],
            skipped_bins=["2023-02-01"],
            added=[folder / "aoi_sr_2023-01-05.tif"],
        )

    def _run_sr(*args, **kwargs):
        raise AssertionError("run_sr must not run for a timeseries config")

    monkeypatch.setattr(sr_cli, "run_sr_timeseries", _run_sr_timeseries)
    monkeypatch.setattr(sr_cli, "run_sr", _run_sr)

    sr_cli.main([str(full_acquire_cfg), "--set", "sentinel.timeseries.enabled=true"])

    out = capsys.readouterr().out
    assert seen["cfg"].sentinel.timeseries.enabled is True
    assert "no scene for the interval starting 2023-02-01" in out
    assert f"{tmp_path / 'aoi_sr_timeseries'} (1 new, 1 total)" in out
