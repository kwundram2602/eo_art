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
