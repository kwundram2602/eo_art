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


def _to_tensor(pixels: np.ndarray) -> "torch.Tensor":
    """Convert an HxWxC float32 array to a 1xCxHxW tensor."""
    import torch

    t = torch.from_numpy(pixels.transpose(2, 0, 1)).float()
    return t.unsqueeze(0)


def _to_pixels(t: "torch.Tensor") -> np.ndarray:
    """Convert a 1xCxHxW tensor back to an HxWxC float32 array, clipped to [0, 1]."""
    arr = t.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _load_style(
    src: "str | Path | RenderStep",
    device: "torch.device",
) -> "torch.Tensor":
    """Load a style image from a file path or RenderStep into a 1x3xHxW tensor.

    Args:
        src: A file path (str or Path) to an image, or a RenderStep whose
            pixels array is used directly. Single-channel inputs are broadcast
            to 3 channels; extra channels beyond 3 are dropped.
        device: The torch device to place the output tensor on.

    Returns:
        A float32 tensor of shape (1, 3, H, W) with values in [0, 1].
    """
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
    t: "torch.Tensor",
    max_size: int,
) -> "tuple[torch.Tensor, tuple[int, int]]":
    """Downscale a tensor so its longest side is at most max_size.

    Args:
        t: Input tensor of shape (1, C, H, W).
        max_size: Maximum length for either spatial dimension.

    Returns:
        A (resized_tensor, (original_h, original_w)) tuple. If the tensor is
        already within max_size the original tensor is returned unchanged.
    """
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


def _resize_back(
    t: "torch.Tensor",
    hw: "tuple[int, int]",
) -> "torch.Tensor":
    """Upscale a tensor to the given (H, W) using bicubic interpolation.

    Args:
        t: Input tensor of shape (1, C, h, w).
        hw: Target spatial size (H, W).

    Returns:
        Tensor of shape (1, C, H, W).
    """
    import torch.nn.functional as F

    return F.interpolate(t, size=hw, mode="bicubic", align_corners=False)


def _vgg_normalize(t: "torch.Tensor", device: "torch.device") -> "torch.Tensor":
    """Normalize a tensor with ImageNet mean and std for VGG input.

    Args:
        t: Input tensor of shape (1, 3, H, W) with values in [0, 1].
        device: Device on which to create the normalization constants.

    Returns:
        Normalized tensor of the same shape.
    """
    import torch

    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std
