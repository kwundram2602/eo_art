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
