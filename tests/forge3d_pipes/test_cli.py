import pytest
import yaml

from eo_art.forge3d_pipes import cli
from eo_art.forge3d_pipes.pipeline import VariantResult


@pytest.fixture
def captured_run(monkeypatch, tmp_path):
    calls = {}

    def _run(configs, overrides=(), out=None, use_cache=True, fail_fast=None):
        calls.update(
            configs=list(configs),
            overrides=list(overrides),
            out=out,
            use_cache=use_cache,
            fail_fast=fail_fast,
        )
        return [VariantResult(name="default", out_dir=tmp_path, ok=True)]

    monkeypatch.setattr(cli, "run", _run)
    return calls


@pytest.fixture
def cfg_path(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"input": {"path": "dem.tif"}}))
    return path


def test_run_passes_configs_in_order(captured_run, cfg_path, tmp_path):
    other = tmp_path / "look.yaml"
    other.write_text("render:\n  width: 100\n")
    assert cli.main(["run", str(cfg_path), str(other)]) == 0
    assert captured_run["configs"] == [str(cfg_path), str(other)]


def test_set_flags_become_overrides(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--set", "render.width=99", "--set", "run.name=x"])
    assert captured_run["overrides"] == ["render.width=99", "run.name=x"]


def test_out_flag_is_forwarded(captured_run, cfg_path, tmp_path):
    cli.main(["run", str(cfg_path), "--out", str(tmp_path / "renders")])
    assert captured_run["out"] == str(tmp_path / "renders")


def test_sweep_shorthand_becomes_an_override(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--sweep", "render.pbr.exposure=1.0,1.35,1.8"])
    assert captured_run["overrides"] == [
        "sweep.params={render.pbr.exposure: [1.0, 1.35, 1.8]}"
    ]


def test_malformed_sweep_is_a_clean_error(cfg_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", str(cfg_path), "--sweep", "badvalue"])
    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err
    assert "--sweep expects 'dotted.path=v1,v2'" in stderr


def test_no_cache_flag_disables_caching(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--no-cache"])
    assert captured_run["use_cache"] is False


def test_fail_fast_flag(captured_run, cfg_path):
    cli.main(["run", str(cfg_path), "--fail-fast"])
    assert captured_run["fail_fast"] is True


def test_exit_code_is_nonzero_when_a_variant_failed(monkeypatch, cfg_path, tmp_path):
    monkeypatch.setattr(
        cli,
        "run",
        lambda *a, **k: [
            VariantResult(name="a", out_dir=tmp_path, ok=True),
            VariantResult(name="b", out_dir=tmp_path, ok=False, error="boom"),
        ],
    )
    assert cli.main(["run", str(cfg_path)]) == 1


def test_summary_is_printed(monkeypatch, cfg_path, tmp_path, capsys):
    monkeypatch.setattr(
        cli,
        "run",
        lambda *a, **k: [
            VariantResult(name="a", out_dir=tmp_path, ok=True),
            VariantResult(name="b", out_dir=tmp_path, ok=False, error="boom"),
        ],
    )
    cli.main(["run", str(cfg_path)])
    output = capsys.readouterr().out
    assert "1 succeeded" in output
    assert "1 failed" in output
    assert "boom" in output


class _FakeAcquireResult:
    def __init__(self, dtm_path, optical_path):
        self.dtm_path = dtm_path
        self.optical_path = optical_path


@pytest.fixture
def fake_acquire_stage(monkeypatch, tmp_path):
    calls = {}
    dtm_path = tmp_path / "dtm.tif"
    optical_path = tmp_path / "optical_aligned.tif"

    def _stage(acquire_config, out_dir, use_cache):
        calls.update(
            acquire_config=acquire_config, out_dir=out_dir, use_cache=use_cache
        )
        return _FakeAcquireResult(dtm_path, optical_path)

    monkeypatch.setattr(cli, "_run_acquire_stage", _stage)
    calls["dtm_path"] = dtm_path
    calls["optical_path"] = optical_path
    return calls


def test_acquire_command_prints_paths(fake_acquire_stage, tmp_path, capsys):
    acquire_cfg = tmp_path / "acquire.yaml"
    acquire_cfg.write_text("aoi_path: aoi.gpkg\n")

    exit_code = cli.main(["acquire", str(acquire_cfg), "--out", str(tmp_path / "out")])

    assert exit_code == 0
    assert fake_acquire_stage["acquire_config"] == str(acquire_cfg)
    assert fake_acquire_stage["out_dir"] == str(tmp_path / "out")
    assert fake_acquire_stage["use_cache"] is True
    output = capsys.readouterr().out
    assert str(fake_acquire_stage["dtm_path"]) in output
    assert str(fake_acquire_stage["optical_path"]) in output


def test_acquire_command_no_cache_flag(fake_acquire_stage, tmp_path):
    acquire_cfg = tmp_path / "acquire.yaml"
    acquire_cfg.write_text("aoi_path: aoi.gpkg\n")

    cli.main(["acquire", str(acquire_cfg), "--no-cache"])

    assert fake_acquire_stage["use_cache"] is False


def test_run_with_acquire_injects_input_and_overlay_overrides(
    captured_run, fake_acquire_stage, cfg_path, tmp_path
):
    acquire_cfg = tmp_path / "acquire.yaml"
    acquire_cfg.write_text("aoi_path: aoi.gpkg\n")

    cli.main(["run", str(cfg_path), "--acquire", str(acquire_cfg)])

    assert captured_run["overrides"] == [
        f"input.path={fake_acquire_stage['dtm_path']}",
        f"overlays.0.path={fake_acquire_stage['optical_path']}",
    ]


def test_run_with_acquire_no_cache_flag(
    captured_run, fake_acquire_stage, cfg_path, tmp_path
):
    acquire_cfg = tmp_path / "acquire.yaml"
    acquire_cfg.write_text("aoi_path: aoi.gpkg\n")

    cli.main(
        ["run", str(cfg_path), "--acquire", str(acquire_cfg), "--acquire-no-cache"]
    )

    assert fake_acquire_stage["use_cache"] is False


def test_run_with_acquire_overlay_selects_named_overlay(
    captured_run, fake_acquire_stage, tmp_path
):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": "dem.tif"},
                "overlays": [
                    {"name": "a", "path": "a.tif"},
                    {"name": "b", "path": "b.tif"},
                ],
            }
        )
    )
    acquire_cfg = tmp_path / "acquire.yaml"
    acquire_cfg.write_text("aoi_path: aoi.gpkg\n")

    cli.main(
        [
            "run",
            str(cfg_path),
            "--acquire",
            str(acquire_cfg),
            "--acquire-overlay",
            "b",
        ]
    )

    assert captured_run["overrides"] == [
        f"input.path={fake_acquire_stage['dtm_path']}",
        f"overlays.1.path={fake_acquire_stage['optical_path']}",
    ]
