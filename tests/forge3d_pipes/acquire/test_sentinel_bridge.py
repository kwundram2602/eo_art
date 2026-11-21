import json

from omegaconf import OmegaConf

from eo_art.forge3d_pipes.acquire import sentinel_bridge
from eo_art.forge3d_pipes.acquire.schema import Sentinel


def _sentinel(**overrides) -> Sentinel:
    base = OmegaConf.merge(
        OmegaConf.structured(Sentinel),
        {"start_date": "2023-01-01", "end_date": "2023-12-31", "model_path": "/m"},
        overrides,
    )
    return OmegaConf.to_object(base)


def _fake_subprocess(monkeypatch, manifest):
    commands = []

    def _run(command, check):
        commands.append(command)
        out = command[command.index("--out") + 1]
        (sentinel_bridge.Path(out) / sentinel_bridge.MANIFEST_NAME).write_text(
            json.dumps(manifest)
        )

    monkeypatch.setattr(sentinel_bridge.subprocess, "run", _run)
    return commands


def test_plain_run_passes_no_timeseries_flags(monkeypatch, tmp_path):
    commands = _fake_subprocess(
        monkeypatch, {"scenes": [{"path": "a.tif", "cloud_cover": 3.0}]}
    )

    scenes = sentinel_bridge.run_sentinel_sr(_sentinel(), "sr", 1.0, 2.0, tmp_path)

    assert "--interval" not in commands[0]
    assert "--overwrite" not in commands[0]
    # Manifests from before the timeseries option carry no datetime.
    assert scenes == [sentinel_bridge.Scene(sentinel_bridge.Path("a.tif"), 3.0)]


def test_timeseries_and_overwrite_flags(monkeypatch, tmp_path):
    commands = _fake_subprocess(
        monkeypatch,
        {
            "scenes": [
                {"path": "a.tif", "cloud_cover": None, "datetime": "2023-01-05"}
            ],
            "skipped_bins": ["2023-03-01"],
        },
    )
    cfg = _sentinel(timeseries={"enabled": True, "interval": "week", "every": 2})

    scenes = sentinel_bridge.run_sentinel_sr(
        cfg, "sr", 1.0, 2.0, tmp_path, overwrite=True
    )

    command = commands[0]
    assert command[command.index("--interval") + 1] == "week"
    assert command[command.index("--every") + 1] == "2"
    assert command[-1] == "--overwrite"
    assert scenes[0].datetime == "2023-01-05"
    assert sentinel_bridge.read_manifest(tmp_path).skipped_bins == ["2023-03-01"]


def test_timeseries_does_not_pass_max_items(monkeypatch, tmp_path):
    commands = _fake_subprocess(monkeypatch, {"scenes": []})
    plain = _sentinel(max_items=3)
    series = _sentinel(max_items=3, timeseries={"enabled": True})

    sentinel_bridge.run_sentinel_sr(plain, "sr", 1.0, 2.0, tmp_path)
    sentinel_bridge.run_sentinel_sr(series, "sr", 1.0, 2.0, tmp_path)

    assert "--max-items" in commands[0]
    assert "--max-items" not in commands[1]


def test_scl_check_flags_default_on_and_can_be_disabled(monkeypatch, tmp_path):
    commands = _fake_subprocess(
        monkeypatch,
        {"scenes": [{"path": "a.tif", "cloud_cover": 3.0, "aoi_cloud": 0.5}]},
    )

    scenes = sentinel_bridge.run_sentinel_sr(_sentinel(), "sr", 1.0, 2.0, tmp_path)
    sentinel_bridge.run_sentinel_sr(
        _sentinel(max_aoi_cloud=None), "sr", 1.0, 2.0, tmp_path
    )
    sentinel_bridge.run_sentinel_sr(
        _sentinel(max_aoi_cloud=5, scl_cloud_classes=[3, 8, 9]),
        "sr",
        1.0,
        2.0,
        tmp_path,
    )

    default, disabled, custom = commands
    assert default[default.index("--max-aoi-cloud") + 1] == "10.0"
    assert default[default.index("--scl-classes") + 1] == "8,9"
    assert "--max-aoi-cloud" not in disabled and "--scl-classes" not in disabled
    assert custom[custom.index("--scl-classes") + 1] == "3,8,9"
    assert scenes[0].aoi_cloud == 0.5


def test_skip_dates_flag(monkeypatch, tmp_path):
    commands = _fake_subprocess(monkeypatch, {"scenes": []})
    cfg = _sentinel(timeseries={"enabled": True, "fill_gaps": True})

    sentinel_bridge.run_sentinel_sr(cfg, "sr", 1.0, 2.0, tmp_path)
    sentinel_bridge.run_sentinel_sr(
        cfg, "sr", 1.0, 2.0, tmp_path, skip_dates=["2023-01-05", "2023-02-14"]
    )

    assert "--skip-dates" not in commands[0]
    command = commands[1]
    assert command[command.index("--skip-dates") + 1] == "2023-01-05,2023-02-14"
