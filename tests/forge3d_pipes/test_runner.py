from pathlib import Path

import pytest

from eo_art.forge3d_pipes.config.schema import (
    Animation,
    AnimationKind,
    Orbit,
    PipelineConfig,
)
from eo_art.forge3d_pipes.render import runner


class FakeViewer:
    def __init__(self):
        self.commands = []
        self.snapshots = []
        self.animations = []
        self.closed = False

    def send_ipc(self, payload):
        self.commands.append(payload)

    def snapshot(self, path, width=None, height=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")
        self.snapshots.append((Path(path), width, height))

    def render_animation(self, animation, output_dir, fps=30, width=None, height=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for index in range(len(animation.get_keyframes())):
            (Path(output_dir) / f"frame_{index:04d}.png").write_bytes(b"png")
        self.animations.append((animation, Path(output_dir), fps))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


@pytest.fixture
def fake_viewer(monkeypatch):
    viewer = FakeViewer()
    monkeypatch.setattr(runner, "_open_viewer", lambda cfg, terrain_path: viewer)
    return viewer


def _cfg(**kwargs) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.input.path = "dem.tif"
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_still_render_sends_both_commands_then_snapshots(fake_viewer, tmp_path):
    result = runner.render(_cfg(), Path("dem.tif"), tmp_path)
    assert [c["cmd"] for c in fake_viewer.commands] == [
        "set_terrain",
        "set_terrain_pbr",
    ]
    assert result.snapshot == tmp_path / "snapshot.png"
    assert result.frames_dir is None
    assert result.snapshot.exists()


def test_snapshot_uses_configured_dimensions(fake_viewer, tmp_path):
    runner.render(_cfg(), Path("dem.tif"), tmp_path)
    _, width, height = fake_viewer.snapshots[0]
    assert (width, height) == (1200, 720)


def test_snapshot_name_is_configurable(fake_viewer, tmp_path):
    cfg = _cfg()
    cfg.render.snapshot_name = "hero.png"
    result = runner.render(cfg, Path("dem.tif"), tmp_path)
    assert result.snapshot == tmp_path / "hero.png"


def test_animation_render_writes_frames_dir(fake_viewer, tmp_path):
    cfg = _cfg(
        animation=Animation(kind=AnimationKind.orbit, fps=4, orbit=Orbit(duration=1.0))
    )
    result = runner.render(cfg, Path("dem.tif"), tmp_path)
    assert result.frames_dir == tmp_path / "frames"
    assert result.snapshot is None
    assert len(list(result.frames_dir.glob("frame_*.png"))) == 5


def test_animation_render_passes_fps(fake_viewer, tmp_path):
    cfg = _cfg(
        animation=Animation(kind=AnimationKind.orbit, fps=12, orbit=Orbit(duration=1.0))
    )
    runner.render(cfg, Path("dem.tif"), tmp_path)
    assert fake_viewer.animations[0][2] == 12


def test_viewer_is_closed_even_on_failure(fake_viewer, tmp_path, monkeypatch):
    def _boom(payload):
        raise RuntimeError("ipc down")

    monkeypatch.setattr(fake_viewer, "send_ipc", _boom)
    with pytest.raises(RuntimeError, match="ipc down"):
        runner.render(_cfg(), Path("dem.tif"), tmp_path)
    assert fake_viewer.closed


@pytest.mark.gpu
def test_real_render_produces_a_png(synthetic_dem, tmp_path):
    """Opt-in smoke test: runs the actual forge3d viewer."""
    from eo_art.forge3d_pipes.prep import ops

    projected = tmp_path / "utm.tif"
    ops.reproject(synthetic_dem, projected, ops.ReprojectCfg(crs="EPSG:32610"))

    cfg = _cfg()
    cfg.render.width = 320
    cfg.render.height = 240
    cfg.render.camera.radius = 5000.0

    result = runner.render(cfg, projected, tmp_path / "out")
    assert result.snapshot is not None and result.snapshot.exists()
    assert result.snapshot.stat().st_size > 0
