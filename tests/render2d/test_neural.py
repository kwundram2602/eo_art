from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _has_network() -> bool:
    """Return True only when the AdaIN decoder weights are downloadable."""
    import urllib.error
    import urllib.request

    try:
        socket.create_connection(("github.com", 443), timeout=3)
    except OSError:
        return False

    # Verify the actual weights URL is reachable (not just general internet).
    from eo_art.render2d.neural import _ADAIN_DECODER_URL

    try:
        req = urllib.request.Request(
            _ADAIN_DECODER_URL,
            method="HEAD",
            headers={"User-Agent": "eo_art/tests"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


def _make_step(
    h: int = 64,
    w: int = 64,
    channels: int = 3,
    crs: str = "EPSG:4326",
):
    from eo_art.render2d.result import RenderStep

    rng = np.random.default_rng(42)
    pixels = rng.random((h, w, channels)).astype(np.float32)
    return RenderStep(pixels=pixels, crs=crs, resolution=10.0)


# ── import guard ──────────────────────────────────────────────────────────────


def test_missing_dep_raises_import_error() -> None:
    """_require_torch() raises ImportError with pip hint when torch is absent."""
    from eo_art.render2d.neural import _require_torch

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="pip install eo-art"):
            _require_torch()


# ── helpers ───────────────────────────────────────────────────────────────────


@requires_torch
def test_to_tensor_shape() -> None:
    from eo_art.render2d.neural import _to_tensor

    pixels = np.random.rand(32, 48, 3).astype(np.float32)
    t = _to_tensor(pixels)
    assert t.shape == (1, 3, 32, 48)


@requires_torch
def test_to_pixels_roundtrip() -> None:
    from eo_art.render2d.neural import _to_pixels, _to_tensor

    pixels = np.random.rand(32, 48, 3).astype(np.float32)
    recovered = _to_pixels(_to_tensor(pixels))
    assert recovered.shape == (32, 48, 3)
    assert recovered.dtype == np.float32
    np.testing.assert_allclose(recovered, pixels, atol=1e-5)


@requires_torch
def test_to_pixels_clips() -> None:
    from eo_art.render2d.neural import _to_pixels, _to_tensor

    pixels = np.full((8, 8, 3), 2.0, dtype=np.float32)
    out = _to_pixels(_to_tensor(pixels))
    assert out.max() <= 1.0


@requires_torch
def test_load_style_from_path(tmp_path: Path) -> None:
    import torch
    from PIL import Image

    from eo_art.render2d.neural import _load_style

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    p = tmp_path / "style.png"
    img.save(p)
    t = _load_style(p, torch.device("cpu"))
    assert t.shape == (1, 3, 32, 32)


@requires_torch
def test_load_style_from_renderstep() -> None:
    import torch

    from eo_art.render2d.neural import _load_style

    step = _make_step(32, 32, channels=3)
    t = _load_style(step, torch.device("cpu"))
    assert t.shape == (1, 3, 32, 32)


@requires_torch
def test_resize_for_nst_downscales() -> None:
    import torch

    from eo_art.render2d.neural import _resize_for_nst

    t = torch.rand(1, 3, 256, 256)
    small, orig_hw = _resize_for_nst(t, max_size=64)
    assert max(small.shape[2], small.shape[3]) <= 64
    assert orig_hw == (256, 256)


@requires_torch
def test_resize_for_nst_noop_when_small() -> None:
    import torch

    from eo_art.render2d.neural import _resize_for_nst

    t = torch.rand(1, 3, 32, 32)
    small, orig_hw = _resize_for_nst(t, max_size=512)
    assert small.shape == (1, 3, 32, 32)
    assert orig_hw == (32, 32)


@requires_torch
def test_resize_back_restores_size() -> None:
    import torch

    from eo_art.render2d.neural import _resize_back

    t = torch.rand(1, 3, 64, 64)
    restored = _resize_back(t, (128, 200))
    assert restored.shape == (1, 3, 128, 200)


# ── Gatys ─────────────────────────────────────────────────────────────────────


@requires_torch
def test_gatys_output_shape_matches_content() -> None:
    from eo_art import neural_style_transfer

    content = _make_step(64, 80)
    style = _make_step(32, 32)
    result = neural_style_transfer(content, style, method="gatys", max_size=32, steps=2)
    assert result.pixels.shape == (64, 80, 3)


@requires_torch
def test_gatys_crs_and_resolution_preserved() -> None:
    from eo_art import neural_style_transfer
    from eo_art.render2d.result import RenderStep

    content = RenderStep(
        pixels=np.random.default_rng(1).random((32, 32, 3)).astype(np.float32),
        crs="EPSG:32632",
        resolution=30.0,
    )
    style = _make_step(32, 32)
    result = neural_style_transfer(content, style, method="gatys", max_size=32, steps=2)
    assert result.crs == "EPSG:32632"
    assert result.resolution == 30.0


@requires_torch
def test_gatys_style_from_path(tmp_path: Path) -> None:
    from PIL import Image

    from eo_art import neural_style_transfer

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    style_path = tmp_path / "style.jpg"
    img.save(style_path)

    content = _make_step(32, 32)
    result = neural_style_transfer(
        content, style_path, method="gatys", max_size=32, steps=2
    )
    assert result.pixels.shape == (32, 32, 3)
    assert result.pixels.dtype == np.float32


@requires_torch
def test_gatys_output_pixels_in_range() -> None:
    from eo_art import neural_style_transfer

    content = _make_step(32, 32)
    style = _make_step(32, 32)
    result = neural_style_transfer(content, style, method="gatys", max_size=32, steps=2)
    assert result.pixels.min() >= 0.0
    assert result.pixels.max() <= 1.0


@requires_torch
def test_invalid_method_raises_value_error() -> None:
    from eo_art import neural_style_transfer

    content = _make_step(32, 32)
    style = _make_step(32, 32)
    with pytest.raises(ValueError, match="Unknown method"):
        neural_style_transfer(content, style, method="invalid")  # type: ignore[arg-type]


@requires_torch
def test_non_rgb_content_raises_value_error() -> None:
    from eo_art import neural_style_transfer
    from eo_art.render2d.result import RenderStep

    content = RenderStep(
        pixels=np.random.rand(32, 32).astype(np.float32),
        crs="EPSG:4326",
        resolution=1.0,
    )
    style = _make_step(32, 32)
    with pytest.raises(ValueError, match="3 channels"):
        neural_style_transfer(content, style, method="gatys", max_size=32, steps=2)


# ── AdaIN ─────────────────────────────────────────────────────────────────────


@requires_torch
@pytest.mark.skipif(
    not _has_network(), reason="no network — skipping AdaIN weight download"
)
def test_adain_smoke() -> None:
    from eo_art import neural_style_transfer

    content = _make_step(64, 64)
    style = _make_step(64, 64)
    result = neural_style_transfer(content, style, method="adain", max_size=64)
    assert result.pixels.shape == (64, 64, 3)
    assert result.pixels.dtype == np.float32
    assert result.pixels.min() >= 0.0
    assert result.pixels.max() <= 1.0
    assert result.crs == content.crs
    assert result.resolution == content.resolution


# ── RenderStep.style_transfer wrapper ─────────────────────────────────────────


@requires_torch
def test_renderstep_method_equals_standalone() -> None:
    from eo_art import neural_style_transfer
    from eo_art.render2d.result import RenderStep

    rng = np.random.default_rng(0)
    pixels = rng.random((32, 32, 3)).astype(np.float32)
    content = RenderStep(pixels=pixels, crs="EPSG:4326", resolution=10.0)
    style = _make_step(32, 32)

    result_fn = neural_style_transfer(
        content, style, method="gatys", max_size=32, steps=2
    )
    result_method = content.style_transfer(style, method="gatys", max_size=32, steps=2)
    # Both paths must produce the same shape + metadata
    assert result_fn.pixels.shape == result_method.pixels.shape
    assert result_fn.crs == result_method.crs
    assert result_fn.resolution == result_method.resolution
    np.testing.assert_allclose(result_fn.pixels, result_method.pixels, atol=1e-4)


@requires_torch
def test_neural_style_transfer_importable_from_eo_art() -> None:
    from eo_art import neural_style_transfer

    assert callable(neural_style_transfer)
