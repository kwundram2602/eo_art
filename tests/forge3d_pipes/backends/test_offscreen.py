from pathlib import Path

import pytest

from eo_art.forge3d_pipes.config.schema import PipelineConfig, RenderBackend
from eo_art.forge3d_pipes.render.backends import offscreen
from eo_art.forge3d_pipes.render.results import ResolvedOverlay


class FakeFrame:
    def __init__(self):
        self.saved: list[Path] = []

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"png")
        self.saved.append(Path(path))


@pytest.fixture
def fake_frame(monkeypatch):
    frame = FakeFrame()
    monkeypatch.setattr(
        offscreen, "_render_frame", lambda cfg, terrain_path, overlays: frame
    )
    return frame


def _cfg(**kwargs) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.input.path = "dem.tif"
    cfg.render.backend = RenderBackend.offscreen
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_render_saves_frame_to_snapshot_path(fake_frame, tmp_path):
    result = offscreen.render(_cfg(), Path("dem.tif"), tmp_path)
    assert result.snapshot == tmp_path / "snapshot.png"
    assert result.frames_dir is None
    assert fake_frame.saved == [tmp_path / "snapshot.png"]


def test_render_uses_configured_snapshot_name(fake_frame, tmp_path):
    cfg = _cfg()
    cfg.render.snapshot_name = "hero.png"
    result = offscreen.render(cfg, Path("dem.tif"), tmp_path)
    assert result.snapshot == tmp_path / "hero.png"


def test_render_creates_out_dir(fake_frame, tmp_path):
    out_dir = tmp_path / "nested" / "out"
    offscreen.render(_cfg(), Path("dem.tif"), out_dir)
    assert out_dir.exists()


def test_render_passes_overlays_through_to_render_frame(monkeypatch, tmp_path):
    received = []

    def _fake(cfg, terrain_path, overlays):
        received.append(list(overlays))
        return FakeFrame()

    monkeypatch.setattr(offscreen, "_render_frame", _fake)
    overlay = ResolvedOverlay(name="a", path=Path("a.png"), extent=(0.0, 0.0, 1.0, 1.0))
    offscreen.render(_cfg(), Path("dem.tif"), tmp_path, overlays=[overlay])
    assert received == [[overlay]]


# ---------------------------------------------------------------------------
# Placeholder IBL: verified against forge3d's *real* HDR loader (pure file
# parsing, no GPU/Session touched), not just a hand-checked byte layout.
# ---------------------------------------------------------------------------


def test_placeholder_hdr_is_readable_by_forge3ds_real_loader(tmp_path):
    import forge3d

    path = tmp_path / "placeholder.hdr"
    offscreen._write_placeholder_hdr(str(path))
    ibl = forge3d.IBL.from_hdr(str(path), intensity=0.0)
    assert ibl.dimensions == (8, 4)


def test_placeholder_ibl_has_zero_intensity():
    ibl = offscreen._placeholder_ibl()
    assert ibl.intensity == 0.0


def test_placeholder_ibl_cleans_up_its_temp_file():
    import glob
    import tempfile

    before = set(glob.glob(f"{tempfile.gettempdir()}/eo_art_placeholder_ibl_*"))
    offscreen._placeholder_ibl()
    after = set(glob.glob(f"{tempfile.gettempdir()}/eo_art_placeholder_ibl_*"))
    assert after == before


@pytest.mark.gpu
def test_real_offscreen_render_produces_a_png(synthetic_dem, tmp_path):
    """Opt-in smoke test: runs the actual forge3d offscreen renderer."""
    from eo_art.forge3d_pipes.prep import ops

    projected = tmp_path / "utm.tif"
    ops.reproject(synthetic_dem, projected, ops.ReprojectCfg(crs="EPSG:32610"))

    cfg = _cfg()
    cfg.render.width = 320
    cfg.render.height = 240
    cfg.render.camera.radius = 5000.0

    result = offscreen.render(cfg, projected, tmp_path / "out")
    assert result.snapshot is not None and result.snapshot.exists()
    assert result.snapshot.stat().st_size > 0
