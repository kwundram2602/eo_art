import sys
import types
from pathlib import Path

import pytest

from eo_art.forge3d_pipes.config.schema import (
    GpuPreference,
    PipelineConfig,
    RenderBackend,
)
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


# ---------------------------------------------------------------------------
# _open_session: run.gpu -> whether the Vulkan backend is forced. Exercised
# against a fake ``forge3d`` module (no real Session/GPU touched) so these
# run in the default suite -- only the actual construction of a real
# forge3d.Session needs @pytest.mark.gpu.
# ---------------------------------------------------------------------------


class _FakeForge3dModule(types.ModuleType):
    """Stands in for the real ``forge3d`` module in sys.modules.

    ``_open_session`` does ``import forge3d`` locally (a real Session must
    never be constructed before ``gpu.select_gpu`` runs); Python's import
    statement resolves that against sys.modules first, so swapping the
    entry there redirects the local import without touching a real GPU.
    """

    def __init__(self, session_factory):
        super().__init__("forge3d")
        self._session_factory = session_factory

    def Session(self, window=False, backend=None):
        return self._session_factory(window=window, backend=backend)


def _install_fake_forge3d(monkeypatch, session_factory):
    monkeypatch.setitem(sys.modules, "forge3d", _FakeForge3dModule(session_factory))


def test_open_session_off_preference_does_not_force_a_backend(monkeypatch):
    calls = []

    def factory(*, window, backend):
        calls.append({"window": window, "backend": backend})
        return "fake-session"

    _install_fake_forge3d(monkeypatch, factory)
    cfg = _cfg()
    cfg.run.gpu = GpuPreference.off

    result = offscreen._open_session(cfg)

    assert result == "fake-session"
    assert calls == [{"window": False, "backend": None}]


@pytest.mark.parametrize(
    "preference",
    [GpuPreference.auto, GpuPreference.discrete, GpuPreference.integrated],
)
def test_open_session_forces_vulkan_for_every_non_off_preference(
    preference, monkeypatch
):
    calls = []

    def factory(*, window, backend):
        calls.append({"window": window, "backend": backend})
        return "fake-session"

    _install_fake_forge3d(monkeypatch, factory)
    cfg = _cfg()
    cfg.run.gpu = preference

    result = offscreen._open_session(cfg)

    assert result == "fake-session"
    assert calls == [{"window": False, "backend": "vulkan"}]


def test_open_session_wraps_vulkan_unavailable_error(monkeypatch):
    native_message = (
        "[Device] Device error: No suitable GPU adapter found (backends "
        "tried: Backends(VULKAN); WGPU_BACKENDS/WGPU_BACKEND set to "
        "'vulkan'). Verify GPU drivers are installed... install Mesa's "
        "lavapipe."
    )

    def factory(*, window, backend):
        raise RuntimeError(native_message)

    _install_fake_forge3d(monkeypatch, factory)
    cfg = _cfg()
    cfg.run.gpu = GpuPreference.auto

    with pytest.raises(RuntimeError) as exc_info:
        offscreen._open_session(cfg)

    message = str(exc_info.value)
    # Names the cause (forcing Vulkan for run.gpu steering) and the way out
    # (install a driver, or opt out via run.gpu: off) -- not the raw
    # WGPU_BACKENDS/lavapipe message pointing at nothing in this project.
    assert "Vulkan driver" in message
    assert "run.gpu: off" in message
    assert exc_info.value.__cause__ is not None
    assert "lavapipe" in str(exc_info.value.__cause__)


@pytest.mark.gpu
def test_real_offscreen_render_produces_a_png(synthetic_dem, tmp_path):
    """Opt-in smoke test: runs the actual forge3d offscreen renderer.

    Asserts actual pixel *content*, not just file existence/size: a
    degenerate (e.g. solid-black) frame still writes a small-but-nonzero
    PNG, which is exactly the failure mode a size-only assertion missed
    when ``Session(window=False)`` silently fell back to a non-functional
    GL/integrated-GPU backend instead of the Vulkan/discrete GPU
    ``gpu.select_gpu`` had steered the ICD environment towards.
    """
    import imageio.v3 as iio
    import numpy as np

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

    pixels = iio.imread(result.snapshot)
    assert pixels[..., :3].std() > 1.0, (
        "frame is degenerate/flat -- render likely failed"
    )
    assert not np.array_equal(pixels[..., :3], np.zeros_like(pixels[..., :3]))
