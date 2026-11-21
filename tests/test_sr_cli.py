from pathlib import Path

import pytest
import yaml

from eo_art import sr_cli


@pytest.fixture
def fake_run_sr(monkeypatch, tmp_path):
    calls = {}
    sr_path = tmp_path / "optical_sr.tif"

    def _run_sr(cfg, out_dir, use_cache, run_dir=None):
        calls.update(cfg=cfg, out_dir=out_dir, use_cache=use_cache, run_dir=run_dir)
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

    def _run_sr_timeseries(cfg, out_dir, use_cache, run_dir=None):
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


def _with_zones(path, **sentinel):
    raw = yaml.safe_load(path.read_text())
    raw["sentinel"].update(sentinel)
    path.write_text(yaml.safe_dump(raw))
    return path


def test_sr_command_dispatches_zones(monkeypatch, full_acquire_cfg, tmp_path, capsys):
    from eo_art.forge3d_pipes.acquire import SrZonesResult

    seen = {}

    def _run_sr_zones(cfg, out_dir, use_cache, run_dir=None):
        seen.update(cfg=cfg, use_cache=use_cache)
        return SrZonesResult(
            out_dir=tmp_path / "valley_sr_zones",
            tiles=[tmp_path / "a.tif", tmp_path / "b.tif"],
            vrt=tmp_path / "valley_sr_zones.vrt",
            mosaic=None,
            empty_zones=["z00_02"],
        )

    monkeypatch.setattr(sr_cli, "run_sr_zones", _run_sr_zones)
    _with_zones(full_acquire_cfg, zones={"enabled": True})

    assert sr_cli.main([str(full_acquire_cfg)]) == 0

    assert seen["cfg"].sentinel.zones.enabled is True
    out = capsys.readouterr().out
    assert "2 tile(s)" in out and "z00_02" in out


def test_sr_command_mosaic_only(monkeypatch, full_acquire_cfg, tmp_path, capsys):
    from eo_art.forge3d_pipes.acquire.runs import link_latest, new_run_dir

    seen = []
    monkeypatch.setattr(
        sr_cli,
        "export_zones_mosaic",
        lambda aoi, run: seen.append((aoi, run)) or tmp_path / "m.tif",
    )
    monkeypatch.setattr(sr_cli, "run_sr", lambda *a, **k: pytest.fail("no SR"))
    out = tmp_path / "cfg_out"

    with pytest.raises(SystemExit, match="no finished run"):
        sr_cli.main([str(full_acquire_cfg), "--mosaic-only"])

    run = new_run_dir(out, "sr", "20260101-000000-sr")
    link_latest(out, "sr", run)
    assert sr_cli.main([str(full_acquire_cfg), "--mosaic-only"]) == 0
    assert sr_cli.main([str(full_acquire_cfg), "--mosaic-only", "--run", "x"]) == 0

    assert seen == [("aoi.gpkg", run), ("aoi.gpkg", Path("x"))]
    assert "m.tif" in capsys.readouterr().out


def test_sr_command_reports_zone_timeseries_steps(
    monkeypatch, full_acquire_cfg, tmp_path, capsys
):
    from eo_art.forge3d_pipes.acquire import ZoneTimeseriesResult
    from eo_art.forge3d_pipes.acquire.pipeline import ZoneStepResult
    from eo_art.forge3d_pipes.acquire.zone_steps import Step, ZonePick

    step = Step(
        bin="2024-01-01",
        picks={
            "z00_00": ZonePick("2024-01-03", 1.0),
            "z00_01": ZonePick("2024-01-25", 2.0, outside_days=8),
        },
        window=("2024-01-03", "2024-01-17"),
        empty_zones=[],
    )
    monkeypatch.setattr(
        sr_cli,
        "run_sr_timeseries_zones",
        lambda cfg, out, use_cache, run_dir: ZoneTimeseriesResult(
            out_dir=tmp_path,
            steps=[
                ZoneStepResult(
                    step,
                    {},
                    tmp_path / "lr.tif",
                    tmp_path / "sr.tif",
                    {"2024-01-03": (0.5, -0.25), "2024-01-25": None},
                )
            ],
            skipped_bins=["2024-02-01"],
        ),
    )
    _with_zones(
        full_acquire_cfg,
        zones={"enabled": True},
        timeseries={"enabled": True, "max_zone_offset_days": 14},
    )

    assert sr_cli.main([str(full_acquire_cfg)]) == 0

    out = capsys.readouterr().out
    assert "z00_01 2024-01-25 (+8 d outside)" in out
    assert "spread 22 d" in out
    assert "2024-02-01" in out
    assert "shift 2024-01-03 (+0.50, -0.25) px, 2024-01-25 dropped" in out


def test_sr_command_writes_into_a_new_run_folder(
    fake_run_sr, full_acquire_cfg, tmp_path
):
    out = tmp_path / "cfg_out"

    sr_cli.main([str(full_acquire_cfg), "--run-id", "20260101-000000-sr"])
    sr_cli.main([str(full_acquire_cfg)])

    first = out / "runs" / "20260101-000000-sr"
    assert fake_run_sr["run_dir"] != first
    assert (first / "config.yaml").read_text().startswith("aoi_path: aoi.gpkg")
    # The newest finished run is the latest one; the cache root stays out/.
    assert (out / "runs" / "latest-sr").resolve() == fake_run_sr["run_dir"]
    assert fake_run_sr["out_dir"] == str(out)


def test_fill_gaps_continues_the_latest_sr_runs_timeseries(
    monkeypatch, full_acquire_cfg, tmp_path
):
    from eo_art.forge3d_pipes.acquire import TimeseriesResult
    from eo_art.forge3d_pipes.acquire.runs import link_latest, new_run_dir

    out = tmp_path / "cfg_out"
    previous = new_run_dir(out, "sr", "20260101-000000-sr")
    frame = previous / "aoi_sr_timeseries" / "aoi_sr_2023-01-05.tif"
    frame.parent.mkdir()
    frame.write_bytes(b"jan")
    link_latest(out, "sr", previous)
    seen = {}

    def _run_sr_timeseries(cfg, out_dir, use_cache, run_dir=None):
        seen["frames"] = sorted(
            p.name for p in (run_dir / "aoi_sr_timeseries").iterdir()
        )
        return TimeseriesResult(out_dir=run_dir, paths=[], skipped_bins=[])

    monkeypatch.setattr(sr_cli, "run_sr_timeseries", _run_sr_timeseries)
    _with_zones(full_acquire_cfg, timeseries={"enabled": True, "fill_gaps": True})

    sr_cli.main([str(full_acquire_cfg)])

    assert seen["frames"] == ["aoi_sr_2023-01-05.tif"]
    assert frame.exists()
