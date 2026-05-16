# Neural Style Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `neural_style_transfer(content, style, *, method, ...)` to `eo_art`, supporting both Gatys (optimization) and AdaIN (fast feed-forward) style transfer on `RenderStep` objects, with geo-metadata preserved throughout.

**Architecture:** `RenderStep.pixels` is format-agnostic float32 numpy — this is the bridge between GeoTIFF content and PNG-style sources. The module lives at `render2d/neural.py` and integrates into the existing chain API as both a standalone function and a `RenderStep.style_transfer()` method. Heavy ML deps (`torch`, `torchvision`) are an optional extras group.

**Tech Stack:** PyTorch ≥ 2.0, torchvision ≥ 0.15, VGG19 backbone, PIL for style-image loading, Adam optimizer (Gatys), AdaIN normalization (fast path).

---

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | Add `neural` optional dep group |
| `src/eo_art/render2d/neural.py` | **CREATE** — full NST implementation |
| `src/eo_art/render2d/result.py` | **MODIFY** — add `.style_transfer()` method |
| `src/eo_art/__init__.py` | **MODIFY** — export `neural_style_transfer` |
| `tests/render2d/test_neural.py` | **CREATE** — all tests |

---

## Task 1: Optional dependency in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add neural extras group**

In `pyproject.toml`, add `neural` alongside the existing optional deps (`hillshade`, `stac`, `dem`, `3d`):

```toml
[project.optional-dependencies]
hillshade = ["xarray-spatial>=0.3.0"]
stac      = ["pystac-client>=0.8.0", "planetary-computer>=1.0.0"]
dem       = ["py3dep>=0.17.0"]
3d        = ["pyvista>=0.44.0", "trimesh>=4.4.0"]
neural    = ["torch>=2.0", "torchvision>=0.15"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add neural optional dependency group"
```

---

## Task 2: Import guard test + stub (TDD)

**Files:**
- Create: `tests/render2d/test_neural.py`
- Create: `src/eo_art/render2d/neural.py` (stub only)

- [ ] **Step 1: Write the failing test**

Create `tests/render2d/test_neural.py`:

```python
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
    try:
        socket.create_connection(("github.com", 443), timeout=3)
        return True
    except OSError:
        return False


def _make_step(
    h: int = 64,
    w: int = 64,
    channels: int = 3,
    crs: str = "EPSG:4326",
) -> "RenderStep":
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/render2d/test_neural.py::test_missing_dep_raises_import_error -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'eo_art.render2d.neural'`

- [ ] **Step 3: Create neural.py stub with import guard**

Create `src/eo_art/render2d/neural.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch
    from .result import RenderStep

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_GATYS_CONTENT_LAYER = 21            # conv4_2 in VGG19 features
_GATYS_STYLE_LAYERS = [0, 5, 10, 19, 28]  # conv1_1 … conv5_1 in VGG19 features
_ADAIN_ENCODER_DEPTH = 21            # VGG19 features[:21] → relu4_1 (512 ch)

_ADAIN_CACHE_DIR = Path.home() / ".cache" / "eo_art"
_ADAIN_DECODER_PATH = _ADAIN_CACHE_DIR / "adain_decoder.pth"
# NOTE: host decoder.pth on this repo's GitHub releases and update this URL.
_ADAIN_DECODER_URL = (
    "https://github.com/naoto0804/pytorch-AdaIN/raw/master/models/decoder.pth"
)


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "neural_style_transfer requires PyTorch.\n"
            "Install it with:  pip install eo-art[neural]"
        ) from None
```

- [ ] **Step 4: Run the test again — should pass**

```bash
pytest tests/render2d/test_neural.py::test_missing_dep_raises_import_error -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/eo_art/render2d/neural.py tests/render2d/test_neural.py
git commit -m "feat: add neural.py stub with import guard + test"
```

---

## Task 3: Tensor helpers

**Files:**
- Modify: `src/eo_art/render2d/neural.py`
- Modify: `tests/render2d/test_neural.py`

- [ ] **Step 1: Write failing tests** — append to `test_neural.py`:

```python
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
    from PIL import Image

    from eo_art.render2d.neural import _load_style

    import torch

    img = Image.fromarray(
        (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    )
    p = tmp_path / "style.png"
    img.save(p)
    t = _load_style(p, torch.device("cpu"))
    assert t.shape == (1, 3, 32, 32)


@requires_torch
def test_load_style_from_renderstep() -> None:
    from eo_art.render2d.neural import _load_style

    import torch

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
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/render2d/test_neural.py -k "helper or tensor or style or resize" -v
```

Expected: multiple `FAILED` / `ERROR` (functions not yet defined)

- [ ] **Step 3: Implement helpers** — append to `neural.py` after `_ADAIN_DECODER_URL`:

```python

def _to_tensor(pixels: np.ndarray) -> "torch.Tensor":
    import torch

    t = torch.from_numpy(pixels.transpose(2, 0, 1)).float()
    return t.unsqueeze(0)


def _to_pixels(t: "torch.Tensor") -> np.ndarray:
    arr = t.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _load_style(
    src: "str | Path | RenderStep",
    device: "torch.device",
) -> "torch.Tensor":
    from .result import RenderStep

    if isinstance(src, RenderStep):
        p = src.pixels
        if p.ndim == 2:
            p = np.stack([p] * 3, axis=-1)
        elif p.shape[2] > 3:
            p = p[:, :, :3]
        elif p.shape[2] < 3:
            p = np.stack([p[:, :, 0]] * 3, axis=-1)
    else:
        from PIL import Image

        img = Image.open(src).convert("RGB")
        p = np.array(img, dtype=np.float32) / 255.0
    return _to_tensor(p).to(device)


def _resize_for_nst(
    t: "torch.Tensor", max_size: int
) -> "tuple[torch.Tensor, tuple[int, int]]":
    import torch.nn.functional as F

    _, _, h, w = t.shape
    if max(h, w) <= max_size:
        return t, (h, w)
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = F.interpolate(
        t, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return resized, (h, w)


def _resize_back(t: "torch.Tensor", hw: "tuple[int, int]") -> "torch.Tensor":
    import torch.nn.functional as F

    return F.interpolate(t, size=hw, mode="bicubic", align_corners=False)


def _vgg_normalize(t: "torch.Tensor", device: "torch.device") -> "torch.Tensor":
    import torch

    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std
```

- [ ] **Step 4: Run helper tests — all should pass**

```bash
pytest tests/render2d/test_neural.py -k "tensor or style or resize" -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/eo_art/render2d/neural.py tests/render2d/test_neural.py
git commit -m "feat: add tensor helpers to neural.py"
```

---

## Task 4: Gatys NST (TDD)

**Files:**
- Modify: `src/eo_art/render2d/neural.py`
- Modify: `tests/render2d/test_neural.py`

- [ ] **Step 1: Write failing Gatys tests** — append to `test_neural.py`:

```python
# ── Gatys ─────────────────────────────────────────────────────────────────────

@requires_torch
def test_gatys_output_shape_matches_content() -> None:
    from eo_art import neural_style_transfer

    content = _make_step(64, 80)
    style = _make_step(32, 32)
    result = neural_style_transfer(
        content, style, method="gatys", max_size=32, steps=2
    )
    assert result.pixels.shape == (64, 80, 3)


@requires_torch
def test_gatys_crs_and_resolution_preserved() -> None:
    from eo_art import neural_style_transfer

    content = _make_step(32, 32, crs="EPSG:32632")
    content = content.__class__(
        pixels=content.pixels, crs="EPSG:32632", resolution=30.0
    )
    style = _make_step(32, 32)
    result = neural_style_transfer(
        content, style, method="gatys", max_size=32, steps=2
    )
    assert result.crs == "EPSG:32632"
    assert result.resolution == 30.0


@requires_torch
def test_gatys_style_from_path(tmp_path: Path) -> None:
    from PIL import Image

    from eo_art import neural_style_transfer

    img = Image.fromarray(
        (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    )
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
    result = neural_style_transfer(
        content, style, method="gatys", max_size=32, steps=2
    )
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
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/render2d/test_neural.py -k "gatys" -v
```

Expected: `ERROR` or `FAILED` (function not yet defined)

- [ ] **Step 3: Implement Gatys + main function** — append to `neural.py`:

```python

def _gram(t: "torch.Tensor") -> "torch.Tensor":
    _, c, h, w = t.shape
    f = t.view(c, h * w)
    return (f @ f.t()) / (c * h * w)


def _gatys(
    content: "torch.Tensor",
    style: "torch.Tensor",
    *,
    device: "torch.device",
    steps: int,
    content_weight: float,
    style_weight: float,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F
    from torchvision.models import VGG19_Weights, vgg19

    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    _max_layer = max(_GATYS_STYLE_LAYERS + [_GATYS_CONTENT_LAYER])

    def get_feats(x: "torch.Tensor") -> "dict[int, torch.Tensor]":
        out: dict[int, torch.Tensor] = {}
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in {_GATYS_CONTENT_LAYER, *_GATYS_STYLE_LAYERS}:
                out[i] = x
            if i >= _max_layer:
                break
        return out

    content_feats = get_feats(_vgg_normalize(content, device))
    style_feats = get_feats(_vgg_normalize(style, device))
    style_grams = {i: _gram(style_feats[i]) for i in _GATYS_STYLE_LAYERS}

    target = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.01)

    for _ in range(steps):
        target.data.clamp_(0.0, 1.0)
        optimizer.zero_grad()
        target_feats = get_feats(_vgg_normalize(target, device))
        c_loss = content_weight * F.mse_loss(
            target_feats[_GATYS_CONTENT_LAYER],
            content_feats[_GATYS_CONTENT_LAYER],
        )
        s_loss = style_weight * sum(
            F.mse_loss(_gram(target_feats[i]), style_grams[i])
            for i in _GATYS_STYLE_LAYERS
        )
        (c_loss + s_loss).backward()
        optimizer.step()

    target.data.clamp_(0.0, 1.0)
    return target.detach()


def neural_style_transfer(
    content: "RenderStep",
    style: "str | Path | RenderStep",
    *,
    method: Literal["gatys", "adain"] = "gatys",
    max_size: int = 512,
    steps: int = 300,
    content_weight: float = 1.0,
    style_weight: float = 1e6,
    device: str | None = None,
) -> "RenderStep":
    _require_torch()

    import torch

    from .result import RenderStep

    if content.pixels.ndim != 3 or content.pixels.shape[2] != 3:
        raise ValueError(
            f"content must have 3 channels (H, W, 3), got shape {content.pixels.shape}. "
            "Use .composite.rgb() or .colorize() first."
        )

    dev = torch.device(
        device
        if device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    content_t = _to_tensor(content.pixels).to(dev)
    style_t = _load_style(style, dev)

    content_small, orig_hw = _resize_for_nst(content_t, max_size)
    style_small, _ = _resize_for_nst(style_t, max_size)

    if method == "gatys":
        result_small = _gatys(
            content_small,
            style_small,
            device=dev,
            steps=steps,
            content_weight=content_weight,
            style_weight=style_weight,
        )
    elif method == "adain":
        result_small = _adain(content_small, style_small, device=dev)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'gatys' or 'adain'.")

    result_t = _resize_back(result_small, orig_hw)
    out_pixels = _to_pixels(result_t)
    return RenderStep(pixels=out_pixels, crs=content.crs, resolution=content.resolution)
```

Note: `_adain` is defined in Task 5 — do not call `neural_style_transfer(..., method="adain")` until that task is done.

- [ ] **Step 4: Run Gatys tests — all should pass**

```bash
pytest tests/render2d/test_neural.py -k "gatys or invalid_method or non_rgb" -v
```

Expected: all `PASSED` (may be slow on CPU — each test runs only `steps=2`)

- [ ] **Step 5: Commit**

```bash
git add src/eo_art/render2d/neural.py tests/render2d/test_neural.py
git commit -m "feat: implement Gatys NST + neural_style_transfer function"
```

---

## Task 5: AdaIN NST (TDD)

**Files:**
- Modify: `src/eo_art/render2d/neural.py`
- Modify: `tests/render2d/test_neural.py`

- [ ] **Step 1: Write failing AdaIN tests** — append to `test_neural.py`:

```python
# ── AdaIN ─────────────────────────────────────────────────────────────────────

@requires_torch
@pytest.mark.skipif(not _has_network(), reason="no network — skipping AdaIN weight download")
def test_adain_smoke() -> None:
    from eo_art import neural_style_transfer

    content = _make_step(64, 64)
    style = _make_step(64, 64)
    result = neural_style_transfer(
        content, style, method="adain", max_size=64
    )
    assert result.pixels.shape == (64, 64, 3)
    assert result.pixels.dtype == np.float32
    assert result.pixels.min() >= 0.0
    assert result.pixels.max() <= 1.0
    assert result.crs == content.crs
    assert result.resolution == content.resolution
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/render2d/test_neural.py::test_adain_smoke -v
```

Expected: `FAILED` — `NameError: name '_adain' is not defined`

- [ ] **Step 3: Implement AdaIN** — append to `neural.py` **before** `neural_style_transfer`:

```python

class _AdaINDecoder:
    """Mirror-decoder for VGG19 encoder up to relu4_1 (512 channels out)."""

    def __new__(cls) -> "torch.nn.Module":  # type: ignore[misc]
        import torch.nn as nn

        net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
        )
        return net


def _adain_normalize(
    content_feat: "torch.Tensor", style_feat: "torch.Tensor"
) -> "torch.Tensor":
    eps = 1e-5
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + eps
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + eps
    return (content_feat - c_mean) / c_std * s_std + s_mean


def _load_adain_decoder(device: "torch.device") -> "torch.nn.Module":
    import torch

    decoder = _AdaINDecoder()
    if not _ADAIN_DECODER_PATH.exists():
        _ADAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(
            f"[eo_art] Downloading AdaIN decoder weights (~24 MB) "
            f"to {_ADAIN_DECODER_PATH}"
        )
        try:
            torch.hub.download_url_to_file(_ADAIN_DECODER_URL, str(_ADAIN_DECODER_PATH))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download AdaIN decoder weights: {exc}\n"
                f"Download manually from {_ADAIN_DECODER_URL}\n"
                f"and place at: {_ADAIN_DECODER_PATH}"
            ) from exc
    state = torch.load(str(_ADAIN_DECODER_PATH), map_location="cpu", weights_only=True)
    decoder.load_state_dict(state)
    return decoder.to(device).eval()


def _adain(
    content: "torch.Tensor",
    style: "torch.Tensor",
    *,
    device: "torch.device",
) -> "torch.Tensor":
    import torch
    from torchvision.models import VGG19_Weights, vgg19

    encoder = (
        vgg19(weights=VGG19_Weights.DEFAULT)
        .features[:_ADAIN_ENCODER_DEPTH]
        .to(device)
        .eval()
    )
    for p in encoder.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        content_feat = encoder(_vgg_normalize(content, device))
        style_feat = encoder(_vgg_normalize(style, device))
        t = _adain_normalize(content_feat, style_feat)
        decoder = _load_adain_decoder(device)
        output = decoder(t)

    return output.clamp(0.0, 1.0)
```

> **Important:** `_adain` must be defined **before** `neural_style_transfer` in the file. Insert it directly above `neural_style_transfer`.

- [ ] **Step 4: Verify the _AdaINDecoder URL is reachable**

```bash
python -c "import urllib.request; urllib.request.urlopen('https://github.com/naoto0804/pytorch-AdaIN/raw/master/models/decoder.pth', timeout=5)"
```

If this fails with a 404, you need to find the correct URL for the pre-trained decoder weights. Options:
- Check the [naoto0804/pytorch-AdaIN README](https://github.com/naoto0804/pytorch-AdaIN) for the current download link
- Host `decoder.pth` on this repo's GitHub releases and update `_ADAIN_DECODER_URL` in `neural.py`

- [ ] **Step 5: Run AdaIN smoke test**

```bash
pytest tests/render2d/test_neural.py::test_adain_smoke -v -s
```

Expected: `PASSED` (will print download message on first run; subsequent runs use cache)

- [ ] **Step 6: Commit**

```bash
git add src/eo_art/render2d/neural.py tests/render2d/test_neural.py
git commit -m "feat: implement AdaIN fast style transfer"
```

---

## Task 6: RenderStep method + public export (TDD)

**Files:**
- Modify: `tests/render2d/test_neural.py`
- Modify: `src/eo_art/render2d/result.py`
- Modify: `src/eo_art/__init__.py`

- [ ] **Step 1: Write failing wrapper test** — append to `test_neural.py`:

```python
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
    result_method = content.style_transfer(
        style, method="gatys", max_size=32, steps=2
    )
    # Both paths must produce the same shape + metadata
    assert result_fn.pixels.shape == result_method.pixels.shape
    assert result_fn.crs == result_method.crs
    assert result_fn.resolution == result_method.resolution


@requires_torch
def test_neural_style_transfer_importable_from_eo_art() -> None:
    from eo_art import neural_style_transfer

    assert callable(neural_style_transfer)
```

- [ ] **Step 2: Run to confirm failures**

```bash
pytest tests/render2d/test_neural.py -k "wrapper or importable" -v
```

Expected: `FAILED` — `AttributeError: 'RenderStep' object has no attribute 'style_transfer'`

- [ ] **Step 3: Add `.style_transfer()` to RenderStep**

Open `src/eo_art/render2d/result.py`. Add the method at the end of the `RenderStep` class, after `render()`:

```python
    def style_transfer(
        self,
        style: "str | Path | RenderStep",
        **kwargs: object,
    ) -> RenderStep:
        from ..render2d.neural import neural_style_transfer

        return neural_style_transfer(self, style, **kwargs)
```

Also add `Path` to the imports at the top — it is already imported in `result.py`. Verify `from pathlib import Path` is present (it is, at line 4).

You also need to add the `TYPE_CHECKING` guard for the `style` type annotation. Since `str | Path | RenderStep` uses a forward reference that's already in scope, no additional import is needed.

- [ ] **Step 4: Export from `__init__.py`**

Open `src/eo_art/__init__.py`. Add the import and update `__all__`:

```python
__version__ = "0.1.0"

from .core.data import EOData
from .core.errors import CRSMissingError, EODataLoadError
from .export.animation import animate
from .render2d.neural import neural_style_transfer
from .render2d.result import RenderStep
from .render2d.style import apply_palette, blend, load_preset

__all__ = [
    "__version__",
    "EOData",
    "EODataLoadError",
    "CRSMissingError",
    "RenderStep",
    "apply_palette",
    "blend",
    "load_preset",
    "animate",
    "neural_style_transfer",
]
```

- [ ] **Step 5: Run all neural tests**

```bash
pytest tests/render2d/test_neural.py -v
```

Expected: all tests `PASSED` (AdaIN smoke test skipped if no network)

Summary of tests and expected outcomes:
- `test_missing_dep_raises_import_error` → PASSED
- `test_to_tensor_shape` → PASSED
- `test_to_pixels_roundtrip` → PASSED
- `test_to_pixels_clips` → PASSED
- `test_load_style_from_path` → PASSED
- `test_load_style_from_renderstep` → PASSED
- `test_resize_for_nst_downscales` → PASSED
- `test_resize_for_nst_noop_when_small` → PASSED
- `test_resize_back_restores_size` → PASSED
- `test_gatys_output_shape_matches_content` → PASSED
- `test_gatys_crs_and_resolution_preserved` → PASSED
- `test_gatys_style_from_path` → PASSED
- `test_gatys_output_pixels_in_range` → PASSED
- `test_invalid_method_raises_value_error` → PASSED
- `test_non_rgb_content_raises_value_error` → PASSED
- `test_adain_smoke` → PASSED or SKIPPED
- `test_renderstep_method_equals_standalone` → PASSED
- `test_neural_style_transfer_importable_from_eo_art` → PASSED

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
pytest tests/ -v --tb=short
```

Expected: all pre-existing tests still pass, no new failures

- [ ] **Step 7: Commit**

```bash
git add src/eo_art/render2d/result.py src/eo_art/__init__.py tests/render2d/test_neural.py
git commit -m "feat: add RenderStep.style_transfer() + export neural_style_transfer"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** import guard ✓, standalone function ✓, RenderStep method ✓, Gatys ✓, AdaIN ✓, auto device detection ✓, max_size rescaling ✓, style from path ✓, style from RenderStep ✓, optional dep ✓, pyproject.toml ✓, `__init__.py` export ✓
- [x] **Placeholder scan:** no TBD/TODO except the documented AdaIN URL verification step which is explicit
- [x] **Type consistency:** `RenderStep` used consistently, `neural_style_transfer` spelled identically in all tasks, `_require_torch` / `_to_tensor` / `_to_pixels` / `_load_style` / `_resize_for_nst` / `_resize_back` / `_vgg_normalize` / `_gram` / `_gatys` / `_adain` all defined before use
- [x] **AdaIN URL caveat:** Task 5 Step 4 explicitly instructs implementer to verify the URL before proceeding
