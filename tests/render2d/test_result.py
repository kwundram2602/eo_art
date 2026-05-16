import numpy as np
import pytest

from eo_art.render2d.result import RenderStep


def _step(h: int = 8, w: int = 10, channels: int | None = None) -> RenderStep:
    rng = np.random.default_rng(0)
    shape = (h, w) if channels is None else (h, w, channels)
    pixels = rng.random(shape).astype(np.float32)
    return RenderStep(pixels=pixels, crs="EPSG:4326", resolution=10.0)


def test_height_and_width():
    step = _step(8, 12)
    assert step.height == 8
    assert step.width == 12


def test_frozen():
    step = _step()
    with pytest.raises((AttributeError, TypeError)):
        step.crs = "EPSG:32632"  # type: ignore[misc]


def test_invalid_ndim_raises():
    bad = np.zeros((10,), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2-D"):
        RenderStep(pixels=bad, crs="EPSG:4326", resolution=10.0)


def test_normalize_spans_0_to_1():
    pixels = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    out = RenderStep(pixels=pixels, crs="EPSG:4326", resolution=10.0).normalize()
    assert float(out.pixels.min()) == pytest.approx(0.0)
    assert float(out.pixels.max()) == pytest.approx(1.0)


def test_normalize_constant_returns_zeros():
    pixels = np.full((4, 4), 3.0, dtype=np.float32)
    out = RenderStep(pixels=pixels, crs="EPSG:4326", resolution=10.0).normalize()
    assert (out.pixels == 0.0).all()


def test_clip_bounds():
    pixels = np.linspace(-0.5, 1.5, 20).reshape(4, 5).astype(np.float32)
    out = RenderStep(pixels=pixels, crs="EPSG:4326", resolution=10.0).clip()
    assert float(out.pixels.min()) >= 0.0
    assert float(out.pixels.max()) <= 1.0


def test_colorize_produces_rgba_float32():
    step = _step(4, 4)
    out = step.colorize("viridis")
    assert out.pixels.shape == (4, 4, 4)
    assert out.pixels.dtype == np.float32


def test_to_uint8_dtype_and_range():
    step = _step()
    u8 = step.to_uint8()
    assert u8.dtype == np.uint8
    assert int(u8.min()) >= 0
    assert int(u8.max()) <= 255


def test_render_returns_uint8():
    result = _step().render()
    assert result.dtype == np.uint8


def test_render_saves_file(tmp_path):
    out_path = tmp_path / "out.png"
    _step().render(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_chain_preserves_metadata():
    step = _step()
    chained = step.normalize().clip().colorize()
    assert chained.crs == step.crs
    assert chained.resolution == step.resolution


def test_chain_returns_new_instance():
    step = _step()
    out = step.normalize()
    assert out is not step
