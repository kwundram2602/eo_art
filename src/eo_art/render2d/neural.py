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
# NOTE: The original naoto0804/pytorch-AdaIN GitHub repo no longer hosts
# decoder.pth at this path (returns 404).  Host the file on this repo's
# GitHub Releases and update this URL, or point to a public HuggingFace
# repo that does not require authentication.
_ADAIN_DECODER_URL = (
    "https://huggingface.co/p1atdev/pytorch-AdaIN/resolve/main/decoder.pth"
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


def _gram(t: "torch.Tensor") -> "torch.Tensor":
    """Compute the normalised Gram matrix of a feature map.

    Args:
        t: Feature tensor of shape (1, C, H, W).

    Returns:
        Gram matrix of shape (C, C), normalised by C * H * W.
    """
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
    """Run the Gatys et al. (2015) neural style transfer optimisation.

    Optimises a copy of ``content`` to minimise a weighted sum of content
    loss (at ``_GATYS_CONTENT_LAYER``) and style loss (Gram matrices at
    ``_GATYS_STYLE_LAYERS``) against the provided style image.

    Args:
        content: Content image tensor of shape (1, 3, H, W) in [0, 1].
        style: Style image tensor of shape (1, 3, H', W') in [0, 1].
        device: Torch device for all computation.
        steps: Number of Adam optimisation steps.
        content_weight: Weight applied to the content loss term.
        style_weight: Weight applied to the style loss term.

    Returns:
        Stylised tensor of shape (1, 3, H, W) clamped to [0, 1].
    """
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


class _AdaINDecoder:
    """Mirror-decoder for VGG19 encoder up to relu4_1 (512 channels out).

    Uses ``__new__`` as a factory to return an ``nn.Sequential`` directly,
    since the decoder is only ever used as a plain module — no subclassing
    or custom forward logic is required.
    """

    def __new__(cls) -> "torch.nn.Module":  # type: ignore[misc]
        import torch.nn as nn

        return nn.Sequential(
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


def _adain_normalize(
    content_feat: "torch.Tensor",
    style_feat: "torch.Tensor",
) -> "torch.Tensor":
    """Apply Adaptive Instance Normalisation to content features using style statistics.

    Normalises ``content_feat`` to zero mean / unit variance, then rescales
    by the per-channel mean and std of ``style_feat``.

    Args:
        content_feat: Content feature map of shape (1, C, H, W).
        style_feat: Style feature map of shape (1, C, H', W').

    Returns:
        AdaIN-normalised tensor of the same shape as ``content_feat``.
    """
    eps = 1e-5
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + eps
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + eps
    return (content_feat - c_mean) / c_std * s_std + s_mean


def _load_adain_decoder(device: "torch.device") -> "torch.nn.Module":
    """Load the AdaIN decoder weights, downloading them on first use.

    The decoder ``~/.cache/eo_art/adain_decoder.pth`` is downloaded from
    ``_ADAIN_DECODER_URL`` if it does not exist locally.

    Args:
        device: Torch device on which to place the decoder.

    Returns:
        The decoder ``nn.Module`` in eval mode on ``device``.

    Raises:
        RuntimeError: If the weight file cannot be downloaded.
    """
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
    """Run AdaIN fast feed-forward style transfer.

    Encodes both images with a frozen VGG19 encoder (up to relu4_1),
    applies Adaptive Instance Normalisation to align the content features
    with the style statistics, then decodes back to RGB.

    Args:
        content: Content image tensor of shape (1, 3, H, W) in [0, 1].
        style: Style image tensor of shape (1, 3, H', W') in [0, 1].
        device: Torch device for all computation.

    Returns:
        Stylised tensor of shape (1, 3, H, W) clamped to [0, 1].
    """
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
        adapted = _adain_normalize(content_feat, style_feat)
        decoder = _load_adain_decoder(device)
        output = decoder(adapted)

    return output.clamp(0.0, 1.0)


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
    """Apply neural style transfer to a geospatial raster render.

    Transfers the visual style of ``style`` onto ``content`` using either
    the Gatys (2015) optimisation-based method or AdaIN fast feed-forward
    transfer.  Geospatial metadata (CRS, resolution) is preserved from
    ``content``.

    Args:
        content: Source RenderStep with RGB pixels (H, W, 3) in [0, 1].
        style: Style source — a file path (str or Path) to any image that
            Pillow can open, or a RenderStep.
        method: ``"gatys"`` for iterative optimisation (high quality, slow)
            or ``"adain"`` for fast feed-forward transfer.
        max_size: Longest spatial dimension used during NST.  Both images are
            downscaled to this size before processing; the result is bicubic-
            upscaled back to the original content dimensions.
        steps: Number of optimisation steps (Gatys only; ignored by AdaIN).
        content_weight: Weight of the content loss (Gatys only).
        style_weight: Weight of the style loss (Gatys only).
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).  Auto-
            detected (CUDA → MPS → CPU) when ``None``.

    Returns:
        A new RenderStep with stylised pixels of the same shape as ``content``
        and the same CRS/resolution metadata.

    Raises:
        ValueError: If ``content`` does not have exactly 3 channels.
        ValueError: If ``method`` is not ``"gatys"`` or ``"adain"``.
        ImportError: If PyTorch is not installed.
    """
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
