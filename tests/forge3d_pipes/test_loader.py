import pytest
from omegaconf.errors import ConfigKeyError

from eo_art.forge3d_pipes.config import loader
from eo_art.forge3d_pipes.config.schema import TonemapOperator


@pytest.fixture
def cfg_files(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "input:\n  path: dem.tif\nrender:\n  width: 800\n  camera:\n    phi: 100.0\n"
    )
    look = tmp_path / "look.yaml"
    look.write_text(
        "render:\n  width: 1600\n  pbr:\n    tonemap:\n      operator: reinhard\n"
    )
    return base, look


def test_single_file_merges_over_defaults(cfg_files):
    base, _ = cfg_files
    cfg = loader.load_config([base])
    assert cfg.render.width == 800
    assert cfg.render.camera.phi == 100.0
    assert cfg.render.height == 720  # untouched default


def test_later_file_wins(cfg_files):
    base, look = cfg_files
    cfg = loader.load_config([base, look])
    assert cfg.render.width == 1600
    assert cfg.render.camera.phi == 100.0  # still from base
    assert cfg.render.pbr.tonemap.operator is TonemapOperator.REINHARD


def test_dotlist_beats_files(cfg_files):
    base, look = cfg_files
    cfg = loader.load_config([base, look], overrides=["render.width=2000"])
    assert cfg.render.width == 2000


def test_out_sets_run_out_dir_and_beats_files(cfg_files, tmp_path):
    base, _ = cfg_files
    cfg = loader.load_config([base], out=tmp_path / "renders")
    assert cfg.run.out_dir == str(tmp_path / "renders")


def test_unknown_key_in_file_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("input:\n  path: dem.tif\nrender:\n  widht: 800\n")
    with pytest.raises(ConfigKeyError):
        loader.load_config([bad])


def test_interpolation_resolves(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "input:\n  path: dem.tif\nrun:\n  out_dir: /data\n  name: ${input.path}\n"
    )
    cfg = loader.load_config([cfg_file])
    assert cfg.run.name == "dem.tif"


def test_unknown_prep_op_fails_at_load(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("input:\n  path: dem.tif\nprepare:\n  - op: bogus\n")
    with pytest.raises(ValueError, match="unknown prep op 'bogus'"):
        loader.load_config([cfg_file])


def test_load_raw_keeps_dictconfig_for_sweeping(cfg_files):
    base, _ = cfg_files
    raw = loader.load_raw([base])
    assert raw.render.width == 800
    # DictConfig, not a dataclass instance
    assert not hasattr(raw, "__dataclass_fields__")
