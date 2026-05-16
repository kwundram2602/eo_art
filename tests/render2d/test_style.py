from __future__ import annotations

import numpy as np
import pytest

from eo_art.render2d.result import RenderStep
from eo_art.render2d.style import apply_palette, blend, load_preset


def _step(pixels: np.ndarray) -> RenderStep:
    return RenderStep(
        pixels=pixels.astype(np.float32), crs="EPSG:4326", resolution=10.0
    )


def _gray(h: int = 4, w: int = 4) -> RenderStep:
    rng = np.random.default_rng(0)
    return _step(rng.random((h, w)).astype(np.float32))


def test_apply_palette_shape():
    step = _gray()
    result = apply_palette(step, "viridis")
    assert result.pixels.shape == (4, 4, 4)
    assert result.pixels.dtype == np.float32


def test_apply_palette_rgba_range():
    step = _gray()
    result = apply_palette(step, "plasma")
    assert result.pixels.min() >= 0.0
    assert result.pixels.max() <= 1.0


def test_blend_normal_alpha1_equals_overlay():
    base = _step(np.full((4, 4), 0.2, dtype=np.float32))
    overlay = _step(np.full((4, 4), 0.8, dtype=np.float32))
    result = blend(base, overlay, mode="normal", alpha=1.0)
    np.testing.assert_allclose(result.pixels, overlay.pixels, atol=1e-6)


def test_blend_normal_alpha0_equals_base():
    base = _step(np.full((4, 4), 0.2, dtype=np.float32))
    overlay = _step(np.full((4, 4), 0.8, dtype=np.float32))
    result = blend(base, overlay, mode="normal", alpha=0.0)
    np.testing.assert_allclose(result.pixels, base.pixels, atol=1e-6)


def test_blend_multiply():
    b_val, o_val = 0.4, 0.5
    base = _step(np.full((4, 4), b_val, dtype=np.float32))
    overlay = _step(np.full((4, 4), o_val, dtype=np.float32))
    result = blend(base, overlay, mode="multiply", alpha=1.0)
    expected = b_val * o_val
    np.testing.assert_allclose(result.pixels, expected, atol=1e-6)


def test_blend_screen():
    b_val, o_val = 0.4, 0.5
    base = _step(np.full((4, 4), b_val, dtype=np.float32))
    overlay = _step(np.full((4, 4), o_val, dtype=np.float32))
    result = blend(base, overlay, mode="screen", alpha=1.0)
    expected = 1.0 - (1.0 - b_val) * (1.0 - o_val)
    np.testing.assert_allclose(result.pixels, expected, atol=1e-6)


def test_blend_overlay_dark():
    b_val, o_val = 0.3, 0.4
    base = _step(np.full((4, 4), b_val, dtype=np.float32))
    overlay = _step(np.full((4, 4), o_val, dtype=np.float32))
    result = blend(base, overlay, mode="overlay", alpha=1.0)
    expected = 2.0 * b_val * o_val
    np.testing.assert_allclose(result.pixels, expected, atol=1e-6)


def test_blend_overlay_light():
    b_val, o_val = 0.7, 0.6
    base = _step(np.full((4, 4), b_val, dtype=np.float32))
    overlay = _step(np.full((4, 4), o_val, dtype=np.float32))
    result = blend(base, overlay, mode="overlay", alpha=1.0)
    expected = 1.0 - 2.0 * (1.0 - b_val) * (1.0 - o_val)
    np.testing.assert_allclose(result.pixels, expected, atol=1e-6)


def test_blend_unknown_mode_raises():
    base = _gray()
    overlay = _gray()
    with pytest.raises(ValueError, match="Unknown blend mode"):
        blend(base, overlay, mode="dissolve")


def test_load_preset_ndvi():
    preset = load_preset("ndvi")
    assert isinstance(preset, dict)
    assert "cmap" in preset
    assert preset["cmap"] == "RdYlGn"


def test_load_preset_has_blend_mode():
    preset = load_preset("ndvi")
    assert "blend_mode" in preset
    assert "alpha" in preset
