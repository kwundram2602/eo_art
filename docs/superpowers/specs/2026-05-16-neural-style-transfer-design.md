# Neural Style Transfer — Design Spec

**Date:** 2026-05-16  
**Status:** Approved

---

## Overview

Add a Neural Style Transfer (NST) module to `eo_art` that lets users apply the visual style of any reference image (painting, photo, another EO scene) to an `EOData`-derived `RenderStep`. The module lives in `render2d/neural.py` and integrates cleanly with the existing `RenderStep` chain API.

---

## Architecture & Data Flow

The TIF-vs-PNG mismatch is resolved by operating on `RenderStep`, not `EOData` directly.
`RenderStep.pixels` is always a format-agnostic `(H, W, 3)` float32 numpy array in `[0, 1]`.
Geo-metadata (CRS, resolution) lives in the `RenderStep` and is carried through unchanged.

```
EOData  (GeoTIFF, N bands)
  │
  ▼  .composite.rgb()  /  .colorize()  / any render method
RenderStep  (H×W×3, float32, [0,1])          ← content input
  │
  ▼  neural_style_transfer(content, style, ...)
  │
  │  style sources:
  │   • str / Path  → PIL loads PNG / JPG / any PIL-readable format
  │   • RenderStep  → another EO scene used as style source
  │
  │  internal pipeline:
  │    numpy (H,W,3) → torch tensor (1,3,H,W)
  │    ↓ resize to max_size (preserving aspect ratio)
  │    ↓ NST (Gatys or AdaIN)
  │    ↓ resize back to original H×W (bicubic)
  │    ↓ torch tensor → numpy (H,W,3) float32, clipped [0,1]
  ▼
RenderStep  (H×W×3, float32)                 ← crs and resolution from content
```

---

## Public API

### Standalone function

```python
from eo_art import neural_style_transfer

result: RenderStep = neural_style_transfer(
    content: RenderStep,
    style: str | Path | RenderStep,
    *,
    method: Literal["gatys", "adain"] = "gatys",
    max_size: int = 512,          # internal processing size; output rescaled to original
    steps: int = 300,             # Gatys only
    content_weight: float = 1.0,  # Gatys only
    style_weight: float = 1e6,    # Gatys only
    device: str | None = None,    # None → auto-detect: cuda > mps > cpu
) -> RenderStep
```

### RenderStep method

```python
step.style_transfer(
    style: str | Path | RenderStep,
    **kwargs,   # forwarded verbatim to neural_style_transfer
) -> RenderStep
```

`RenderStep.style_transfer` is a thin one-liner wrapper that calls the standalone function.

### Optional dependency

`torch` and `torchvision` are **not** installed by default.
They are declared under `[project.optional-dependencies]` as:

```toml
neural = ["torch>=2.0", "torchvision>=0.15"]
```

Importing `neural_style_transfer` without torch installed raises:

```
ImportError: neural_style_transfer requires PyTorch.
Install it with:  pip install eo-art[neural]
```

---

## Implementation Details

### Module location

`src/eo_art/render2d/neural.py` — consistent with `hillshade.py`, `composite.py`, `style.py`.

### Shared helpers (private)

| Helper | Purpose |
|---|---|
| `_to_tensor(pixels)` | `(H,W,3) float32` → `(1,3,H,W)` torch tensor |
| `_to_pixels(t)` | `(1,3,H,W)` → `(H,W,3) float32`, clipped `[0,1]` |
| `_load_style(src, device)` | Path/str → PIL.open → resize → tensor; RenderStep → `_to_tensor` |
| `_resize_for_nst(t, max_size)` | Returns `(resized_tensor, original_hw)` |
| `_resize_back(t, hw)` | Bicubic upsample to `(H, W)` |

### Gatys (optimization-based)

- Backbone: `torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)` — frozen
- **Content layer:** `features[21]` (conv4_2 — original Gatys paper choice)
- **Style layers:** Gram matrices of conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
- **Optimizer:** `torch.optim.LBFGS` (faster convergence than Adam for NST)
- **Starting image:** Content image (not random noise) — produces more stable results
- **ImageNet normalization** applied internally before VGG forward pass (mean/std baked in)
- `steps` parameter controls LBFGS iteration count

### AdaIN (fast feed-forward)

- **Encoder:** VGG19 up to relu4_1 — from torchvision, same frozen backbone
- **Decoder:** Mirror of encoder; pre-trained weights loaded via `torch.hub` on first call from `naoto0804/pytorch-AdaIN`
- Weights cached at `~/.cache/eo_art/adain_decoder.pth`
- **Normalization:** Adaptive Instance Normalization — content features get mean/std of style features
- Single forward pass, no iteration; `steps` parameter is ignored

**AdaIN first-use message:**
```
[eo_art] Downloading AdaIN decoder weights (~24 MB) to ~/.cache/eo_art/adain_decoder.pth
```

If download fails (no network), raises `RuntimeError` with path hint so user can place weights manually.

---

## Testing Strategy

File: `tests/render2d/test_neural.py`

```
HAS_TORCH = importlib.util.find_spec("torch") is not None
```

| Test | Requires torch | Description |
|---|---|---|
| `test_missing_dep_raises_import_error` | No | Friendly error without torch |
| `test_output_shape_matches_content` | Yes | Output H×W == content H×W |
| `test_crs_and_resolution_preserved` | Yes | Geo-metadata passed through |
| `test_style_from_path` | Yes | Style from 64×64 dummy PNG |
| `test_style_from_renderstep` | Yes | Style from another RenderStep |
| `test_method_gatys_smoke` | Yes | `steps=2`, checks no exception |
| `test_renderstep_method_wrapper` | Yes | `.style_transfer()` == standalone |
| `test_method_adain_smoke` | Yes + network | Skip if no network; checks shape |

No output-quality tests — NST quality is subjective and non-deterministic across devices.

---

## pyproject.toml Changes

```toml
[project.optional-dependencies]
neural = ["torch>=2.0", "torchvision>=0.15"]
```

`neural_style_transfer` exported from `src/eo_art/__init__.py`.

---

## Out of Scope

- Batch processing of multiple frames (see `export/animation.py` — compose separately)
- Fine-tuning or training custom style models
- Style mixing (multiple style images)
- Tiled/patch-based processing for very large images (> 4096px)
