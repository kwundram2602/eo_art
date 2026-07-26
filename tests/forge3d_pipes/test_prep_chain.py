from dataclasses import dataclass
from pathlib import Path

import pytest

from eo_art.forge3d_pipes.prep import registry


@dataclass
class CountCfg:
    tag: str = "x"


@pytest.fixture
def counting_op(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(registry, "_OPS", {})

    @registry.register_op("count", CountCfg)
    def _count(src: Path, dst: Path, cfg: CountCfg) -> Path:
        calls.append(cfg.tag)
        dst.write_text(src.read_text() + f"|{cfg.tag}")
        return dst

    return calls


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "src.tif"
    src.write_text("dem")
    return src


def test_empty_chain_returns_source_unchanged(source, tmp_path):
    assert registry.run_prep_chain(source, [], tmp_path / "cache") == source


def test_chain_applies_ops_in_order(counting_op, source, tmp_path):
    out = registry.run_prep_chain(
        source,
        [{"op": "count", "tag": "a"}, {"op": "count", "tag": "b"}],
        tmp_path / "cache",
    )
    assert out.read_text() == "dem|a|b"
    assert counting_op == ["a", "b"]


def test_second_run_hits_cache(counting_op, source, tmp_path):
    entries = [{"op": "count", "tag": "a"}]
    cache = tmp_path / "cache"
    first = registry.run_prep_chain(source, entries, cache)
    second = registry.run_prep_chain(source, entries, cache)
    assert first == second
    assert counting_op == ["a"]  # op ran only once


def test_no_cache_forces_recompute(counting_op, source, tmp_path):
    entries = [{"op": "count", "tag": "a"}]
    cache = tmp_path / "cache"
    registry.run_prep_chain(source, entries, cache)
    registry.run_prep_chain(source, entries, cache, use_cache=False)
    assert counting_op == ["a", "a"]


def test_different_params_use_different_cache_entries(counting_op, source, tmp_path):
    cache = tmp_path / "cache"
    a = registry.run_prep_chain(source, [{"op": "count", "tag": "a"}], cache)
    b = registry.run_prep_chain(source, [{"op": "count", "tag": "b"}], cache)
    assert a != b
    assert counting_op == ["a", "b"]


def test_cache_key_changes_when_source_changes(source, tmp_path):
    entries = [{"op": "count", "tag": "a"}]
    before = registry.chain_cache_key(source, entries)
    source.write_text("different content entirely")
    assert registry.chain_cache_key(source, entries) != before


def test_failed_op_leaves_no_cache_file(monkeypatch, source, tmp_path):
    monkeypatch.setattr(registry, "_OPS", {})

    @registry.register_op("boom", CountCfg)
    def _boom(src: Path, dst: Path, cfg: CountCfg) -> Path:
        raise RuntimeError("op exploded")

    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="op exploded"):
        registry.run_prep_chain(source, [{"op": "boom"}], cache)
    assert list(cache.glob("*.tif")) == []


def test_passthrough_op_returning_src_does_not_move_the_source(
    monkeypatch, source, tmp_path
):
    """An op may return `src` to mean "no-op"; that must never relocate it."""
    monkeypatch.setattr(registry, "_OPS", {})

    @registry.register_op("passthrough", CountCfg)
    def _passthrough(src: Path, dst: Path, cfg: CountCfg) -> Path:
        return src

    cache = tmp_path / "cache"
    result = registry.run_prep_chain(source, [{"op": "passthrough"}], cache)

    assert source.exists(), "source raster must not be moved into the cache"
    assert source.read_text() == "dem"
    assert result.exists() and result != source
    assert result.read_text() == "dem"
