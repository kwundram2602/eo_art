"""Export: resolved-config dumps and video encoding from PNG frame sequences."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config.schema import Video, VideoFormat

FRAME_GLOB = "frame_*.png"


def write_resolved_config(cfg: DictConfig, path: Path) -> Path:
    """Save the fully merged config next to its outputs, for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    return path


def collect_frames(frames_dir: Path) -> list[Path]:
    """Frames written by ``ViewerHandle.render_animation``, in order."""
    return sorted(Path(frames_dir).glob(FRAME_GLOB))


def _write_frames(frames: list[Path], out_path: Path, fps: int, quality: int) -> None:
    """Isolated so tests can simulate a missing encoder backend."""
    import imageio.v3 as iio

    images = [iio.imread(frame) for frame in frames]
    if out_path.suffix == ".gif":
        iio.imwrite(out_path, images, duration=1000 / fps, loop=0)
    else:
        iio.imwrite(out_path, images, fps=fps, quality=quality)


def encode_video(frames_dir: Path, out_path: Path, video: Video) -> Path:
    """Encode a frame sequence into mp4 or gif."""
    frames = collect_frames(frames_dir)
    if not frames:
        raise ValueError(f"no frames matching {FRAME_GLOB!r} in {frames_dir}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_frames(frames, out_path, video.fps, video.quality)
    except ImportError as exc:
        hint = (
            "install the optional video extra: uv add 'eo-art[video]'"
            if video.format is VideoFormat.mp4
            else "install imageio's plugin for this format"
        )
        raise RuntimeError(
            f"cannot encode {video.format.value}: {exc}. {hint}"
        ) from exc
    return out_path
