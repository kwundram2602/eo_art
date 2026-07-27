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


def test_overlays_sent_after_terrain_commands(fake_viewer, tmp_path):
    overlay = runner.ResolvedOverlay(
        name="ndvi", path=Path("ndvi.tif"), extent=(0.0, 0.0, 1.0, 1.0)
    )
    runner.render(_cfg(), Path("dem.tif"), tmp_path, overlays=[overlay])
    assert [c["cmd"] for c in fake_viewer.commands] == [
        "set_terrain",
        "set_terrain_pbr",
        "load_overlay",
    ]


def test_multiple_overlays_sent_in_list_order(fake_viewer, tmp_path):
    overlays = [
        runner.ResolvedOverlay(
            name="a", path=Path("a.tif"), extent=(0.0, 0.0, 1.0, 1.0), z_order=1
        ),
        runner.ResolvedOverlay(
            name="b", path=Path("b.tif"), extent=(0.0, 0.0, 1.0, 1.0), z_order=2
        ),
    ]
    runner.render(_cfg(), Path("dem.tif"), tmp_path, overlays=overlays)
    load_overlay_cmds = [c for c in fake_viewer.commands if c["cmd"] == "load_overlay"]
    assert [c["name"] for c in load_overlay_cmds] == ["a", "b"]
    assert [c["z_order"] for c in load_overlay_cmds] == [1, 2]


def test_overlay_preserve_colors_sent_once_after_overlays_when_true(
    fake_viewer, tmp_path
):
    cfg = _cfg()
    cfg.render.overlay_preserve_colors = True
    overlays = [
        runner.ResolvedOverlay(
            name="a", path=Path("a.tif"), extent=(0.0, 0.0, 1.0, 1.0)
        ),
        runner.ResolvedOverlay(
            name="b", path=Path("b.tif"), extent=(0.0, 0.0, 1.0, 1.0)
        ),
    ]
    runner.render(cfg, Path("dem.tif"), tmp_path, overlays=overlays)
    cmds = [c["cmd"] for c in fake_viewer.commands]
    assert cmds.count("set_overlay_preserve_colors") == 1
    assert cmds[-1] == "set_overlay_preserve_colors"


def test_overlay_preserve_colors_not_sent_when_false(fake_viewer, tmp_path):
    overlay = runner.ResolvedOverlay(
        name="a", path=Path("a.tif"), extent=(0.0, 0.0, 1.0, 1.0)
    )
    runner.render(_cfg(), Path("dem.tif"), tmp_path, overlays=[overlay])
    assert "set_overlay_preserve_colors" not in [
        c["cmd"] for c in fake_viewer.commands
    ]


def test_render_with_no_overlays_is_unchanged(fake_viewer, tmp_path):
    runner.render(_cfg(), Path("dem.tif"), tmp_path)
    assert [c["cmd"] for c in fake_viewer.commands] == [
        "set_terrain",
        "set_terrain_pbr",
    ]


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


@pytest.mark.gpu
def test_real_render_with_overlay_drapes_onto_terrain(
    synthetic_dem, synthetic_overlay, tmp_path
):
    """Opt-in smoke test: loads a real overlay through the real forge3d viewer.

    This is the only practical way to settle whether forge3d's overlay V axis
    is north-up (GIS convention, assumed by ``compute_normalized_extent``) or
    south-up (image-space convention): render the same terrain with and
    without the overlay and confirm the snapshot actually changes. If it
    turns out ``compute_normalized_extent`` has v0/v1 backwards, the overlay
    will land on the wrong side of the terrain and this test's premise (a
    visibly different, non-degenerate snapshot) still holds, but a follow-up
    visual inspection of the two PNGs would be needed to catch the flip.
    """
    import imageio.v3 as iio
    import numpy as np

    from eo_art.forge3d_pipes.prep import ops
    from eo_art.forge3d_pipes.prep.extent import compute_normalized_extent

    terrain = tmp_path / "utm.tif"
    ops.reproject(synthetic_dem, terrain, ops.ReprojectCfg(crs="EPSG:32610"))
    overlay = tmp_path / "overlay_utm.tif"
    ops.reproject(synthetic_overlay, overlay, ops.ReprojectCfg(crs="EPSG:32610"))
    extent = compute_normalized_extent(terrain, overlay)

    cfg = _cfg()
    cfg.render.width = 320
    cfg.render.height = 240
    cfg.render.camera.radius = 5000.0

    baseline = runner.render(cfg, terrain, tmp_path / "baseline")
    draped = runner.render(
        cfg,
        terrain,
        tmp_path / "draped",
        overlays=[
            runner.ResolvedOverlay(
                name="ndvi", path=overlay, extent=extent, opacity=1.0
            )
        ],
    )

    baseline_pixels = iio.imread(baseline.snapshot)
    draped_pixels = iio.imread(draped.snapshot)
    assert baseline_pixels.shape == draped_pixels.shape
    assert not np.array_equal(baseline_pixels, draped_pixels)
