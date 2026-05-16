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
