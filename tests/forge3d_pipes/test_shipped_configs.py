"""The shipped example configs must actually load.

Without this, a typo in a user-facing example only surfaces when someone
copies it — which is exactly how `render.pbr.tonemap.exposure` (a key that
never existed) survived in the design spec.
"""

from pathlib import Path

import pytest

from eo_art.forge3d_pipes import load_config

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


BASE = CONFIG_ROOT / "base.yaml"


def _look_configs() -> list[Path]:
    """Look files are override fragments, not standalone configs."""
    return sorted((CONFIG_ROOT / "looks").rglob("*.yaml"))


def _overlay_configs() -> list[Path]:
    """Overlay files are override fragments, not standalone configs."""
    return sorted((CONFIG_ROOT / "overlays").rglob("*.yaml"))


def test_configs_directory_is_found():
    assert BASE.is_file(), f"expected a base config at {BASE}"
    assert _look_configs(), "no shipped look configs found"
    assert _overlay_configs(), "no shipped overlay configs found"


def test_base_config_loads_standalone():
    cfg = load_config([BASE])
    assert cfg.render.width > 0
    assert cfg.render.height > 0


@pytest.mark.parametrize("look", _look_configs(), ids=lambda p: p.name)
def test_each_look_composes_onto_base(look):
    """Looks carry no `input.path`, so they are only valid merged onto a base."""
    cfg = load_config([BASE, look])
    assert cfg.render.width > 0


@pytest.mark.parametrize("look", _look_configs(), ids=lambda p: p.name)
def test_look_alone_is_not_a_standalone_config(look):
    from omegaconf.errors import MissingMandatoryValue

    with pytest.raises(MissingMandatoryValue):
        load_config([look])


@pytest.mark.parametrize("overlay", _overlay_configs(), ids=lambda p: p.name)
def test_each_overlay_config_composes_onto_base(overlay):
    """Overlay files carry no `input.path`, only valid merged onto a base."""
    cfg = load_config([BASE, overlay])
    assert cfg.overlays, f"{overlay} should define at least one overlay"


@pytest.mark.parametrize("overlay", _overlay_configs(), ids=lambda p: p.name)
def test_overlay_config_alone_is_not_a_standalone_config(overlay):
    from omegaconf.errors import MissingMandatoryValue

    with pytest.raises(MissingMandatoryValue):
        load_config([overlay])


def test_base_and_look_compose():
    cfg = load_config(
        [CONFIG_ROOT / "base.yaml", CONFIG_ROOT / "looks" / "alpine_dusk.yaml"]
    )
    # base.yaml supplies the output name and prep chain...
    assert cfg.render.snapshot_name == "rainier.png"
    assert [entry["op"] for entry in cfg.prepare] == ["reproject"]
    # ...the look overrides the mood, and wins where both touch a field.
    assert cfg.render.pbr.exposure == 1.6
    assert cfg.render.sun.elevation == 8.0
    # A field neither file sets keeps its schema default.
    assert cfg.render.pbr.msaa == 8
