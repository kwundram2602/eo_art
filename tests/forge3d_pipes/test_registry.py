from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import MISSING
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue

from eo_art.forge3d_pipes.prep import registry


@dataclass
class DummyCfg:
    factor: float = 2.0
    label: str = MISSING


@pytest.fixture
def dummy_op(monkeypatch):
    monkeypatch.setattr(registry, "_OPS", {})

    @registry.register_op("dummy", DummyCfg)
    def _dummy(src: Path, dst: Path, cfg: DummyCfg) -> Path:
        dst.write_text(f"{src.name}:{cfg.factor}:{cfg.label}")
        return dst

    return _dummy


def test_get_op_returns_registered(dummy_op):
    op = registry.get_op("dummy")
    assert op.name == "dummy"
    assert op.schema is DummyCfg
    assert op.func is dummy_op


def test_unknown_op_lists_known_ops(dummy_op):
    with pytest.raises(ValueError, match="unknown prep op 'nope'.*known ops: dummy"):
        registry.get_op("nope")


def test_duplicate_registration_rejected(dummy_op):
    with pytest.raises(ValueError, match="already registered"):
        registry.register_op("dummy", DummyCfg)(lambda src, dst, cfg: dst)


def test_validate_entry_returns_typed_config(dummy_op):
    op, cfg = registry.validate_entry({"op": "dummy", "factor": 3.0, "label": "x"})
    assert op.name == "dummy"
    assert isinstance(cfg, DummyCfg)
    assert cfg.factor == 3.0


def test_validate_entry_rejects_unknown_param(dummy_op):
    with pytest.raises(ConfigKeyError):
        registry.validate_entry({"op": "dummy", "factorr": 3.0, "label": "x"})


def test_validate_entry_rejects_missing_mandatory_param(dummy_op):
    with pytest.raises(MissingMandatoryValue):
        registry.validate_entry({"op": "dummy", "factor": 3.0})


def test_validate_entry_requires_op_key(dummy_op):
    with pytest.raises(ValueError, match="missing 'op' key"):
        registry.validate_entry({"factor": 3.0})


def test_validate_chain_validates_every_entry(dummy_op):
    entries = [
        {"op": "dummy", "label": "a"},
        {"op": "dummy", "label": "b", "factor": 9.0},
    ]
    result = registry.validate_chain(entries)
    assert [cfg.label for _, cfg in result] == ["a", "b"]
    assert result[1][1].factor == 9.0


def test_validate_chain_reports_entry_index(dummy_op):
    with pytest.raises(ValueError, match="prepare\\[1\\]"):
        registry.validate_chain([{"op": "dummy", "label": "a"}, {"op": "nope"}])
