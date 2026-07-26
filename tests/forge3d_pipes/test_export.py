import numpy as np
import pytest
from omegaconf import OmegaConf

from eo_art.forge3d_pipes import export
from eo_art.forge3d_pipes.config.schema import PipelineConfig, Video, VideoFormat


@pytest.fixture
def frames_dir(tmp_path):
    import imageio.v3 as iio

    directory = tmp_path / "frames"
    directory.mkdir()
    for index in range(4):
        frame = np.full((16, 16, 3), index * 60, dtype=np.uint8)
        iio.imwrite(directory / f"frame_{index:04d}.png", frame)
    return directory


def test_write_resolved_config_roundtrips(tmp_path):
    cfg = OmegaConf.merge(
        OmegaConf.structured(PipelineConfig), {"input": {"path": "dem.tif"}}
    )
    path = export.write_resolved_config(cfg, tmp_path / "resolved.yaml")
    assert path.exists()
    reloaded = OmegaConf.load(path)
    assert reloaded.input.path == "dem.tif"
    assert reloaded.render.width == 1200


def test_write_resolved_config_creates_parent_dirs(tmp_path):
    cfg = OmegaConf.merge(
        OmegaConf.structured(PipelineConfig), {"input": {"path": "dem.tif"}}
    )
    path = export.write_resolved_config(cfg, tmp_path / "a" / "b" / "resolved.yaml")
    assert path.exists()


def test_write_resolved_config_resolves_interpolations(tmp_path):
    """Verify interpolations are resolved to literal values in saved config."""
    cfg = OmegaConf.create(
        {
            "run": {"out_dir": "/output/myrun"},
            "export": {"video": "${run.out_dir}/video.mp4"},
        }
    )
    path = export.write_resolved_config(cfg, tmp_path / "resolved.yaml")

    # Reloaded config should have the resolved literal value
    reloaded = OmegaConf.load(path)
    assert reloaded.export.video == "/output/myrun/video.mp4"

    # Raw file text should not contain the interpolation syntax
    raw_text = path.read_text()
    assert "${" not in raw_text
    assert "/output/myrun/video.mp4" in raw_text


def test_collect_frames_is_sorted(frames_dir):
    frames = export.collect_frames(frames_dir)
    assert [f.name for f in frames] == [
        "frame_0000.png",
        "frame_0001.png",
        "frame_0002.png",
        "frame_0003.png",
    ]


def test_collect_frames_ignores_other_files(frames_dir):
    (frames_dir / "notes.txt").write_text("x")
    (frames_dir / "snapshot.png").write_text("x")
    assert len(export.collect_frames(frames_dir)) == 4


def test_encode_gif_writes_a_file(frames_dir, tmp_path):
    out = tmp_path / "out.gif"
    result = export.encode_video(
        frames_dir, out, Video(enabled=True, format=VideoFormat.gif, fps=4)
    )
    assert result == out
    assert out.stat().st_size > 0


def test_encode_rejects_empty_frames_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no frames"):
        export.encode_video(empty, tmp_path / "out.gif", Video(format=VideoFormat.gif))


def test_missing_ffmpeg_backend_gives_install_hint(frames_dir, tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise ImportError("No module named 'imageio_ffmpeg'")

    monkeypatch.setattr(export, "_write_frames", _boom)
    with pytest.raises(RuntimeError, match="eo-art\\[video\\]"):
        export.encode_video(
            frames_dir, tmp_path / "out.mp4", Video(format=VideoFormat.mp4)
        )
