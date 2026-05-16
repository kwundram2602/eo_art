from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .result import RenderStep

_PRESETS_DIR = Path(__file__).parent / "presets"


def apply_palette(step: RenderStep, cmap: str) -> RenderStep:
    """Apply a named matplotlib colormap to step, returning (H, W, 4) RGBA."""
    from matplotlib import colormaps

    band = step.pixels if step.pixels.ndim == 2 else step.pixels[:, :, 0]
    rgba = colormaps[cmap](band).astype(np.float32)
    return step._new(rgba)


def blend(
    base: RenderStep,
    overlay: RenderStep,
    *,
    mode: str = "normal",
    alpha: float = 1.0,
) -> RenderStep:
    """Blend overlay onto base using the given mode at the given opacity.

    Supported modes: "normal", "multiply", "screen", "overlay".
    Raises ValueError for unknown mode.
    """
    b = base.pixels
    o = overlay.pixels

    if mode == "normal":
        blended = o
    elif mode == "multiply":
        blended = b * o
    elif mode == "screen":
        blended = 1.0 - (1.0 - b) * (1.0 - o)
    elif mode == "overlay":
        blended = np.where(b < 0.5, 2.0 * b * o, 1.0 - 2.0 * (1.0 - b) * (1.0 - o))
    else:
        raise ValueError(
            f"Unknown blend mode: {mode!r}. "
            "Choose from: normal, multiply, screen, overlay"
        )

    result = alpha * blended + (1.0 - alpha) * b
    return base._new(result.astype(np.float32))


def load_preset(name_or_path: str) -> dict:
    """Load a style preset dict from a YAML file.

    If name_or_path has no directory component, look it up in the built-in
    presets/ directory alongside this file. Otherwise treat it as a filesystem path.
    """
    p = Path(name_or_path)
    if p.parent == Path("."):
        candidate = _PRESETS_DIR / p.with_suffix(".yaml")
        path = candidate if candidate.exists() else _PRESETS_DIR / p
    else:
        path = p

    with path.open() as fh:
        return yaml.safe_load(fh)
