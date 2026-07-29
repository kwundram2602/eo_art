from pathlib import Path

from eo_art.forge3d_pipes.config.schema import PipelineConfig, RenderBackend
from eo_art.forge3d_pipes.render import runner
from eo_art.forge3d_pipes.render.backends import offscreen, viewer
from eo_art.forge3d_pipes.render.results import RenderResult


def _cfg(**kwargs) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.input.path = "dem.tif"
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def _patch_backends(monkeypatch, calls):
    def _viewer_render(cfg, terrain_path, out_dir, overlays=()):
        calls.append("viewer")
        return RenderResult(snapshot=out_dir / "snapshot.png")

    def _offscreen_render(cfg, terrain_path, out_dir, overlays=()):
        calls.append("offscreen")
        return RenderResult(snapshot=out_dir / "snapshot.png")

    monkeypatch.setattr(viewer, "render", _viewer_render)
    monkeypatch.setattr(offscreen, "render", _offscreen_render)


def test_viewer_backend_routes_to_viewer_render(monkeypatch, tmp_path):
    calls: list[str] = []
    _patch_backends(monkeypatch, calls)

    cfg = _cfg()
    assert cfg.render.backend is RenderBackend.viewer
    runner.render(cfg, Path("dem.tif"), tmp_path)
    assert calls == ["viewer"]


def test_offscreen_backend_routes_to_offscreen_render(monkeypatch, tmp_path):
    calls: list[str] = []
    _patch_backends(monkeypatch, calls)

    cfg = _cfg()
    cfg.render.backend = RenderBackend.offscreen
    runner.render(cfg, Path("dem.tif"), tmp_path)
    assert calls == ["offscreen"]
