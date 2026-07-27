"""Drives the forge3d viewer: open, push payloads, snapshot or animate."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge3d.viewer import open_viewer_async

from eo_art.forge3d_pipes.config.schema import AnimationKind, PipelineConfig
from eo_art.forge3d_pipes.render.animation import build_camera_animation
from eo_art.forge3d_pipes.render.payloads import (
    build_load_overlay,
    build_set_overlay_preserve_colors,
    build_set_terrain,
    build_set_terrain_pbr,
)

FRAMES_DIRNAME = "frames"


@dataclass(frozen=True)
class RenderResult:
    snapshot: Path | None = None
    frames_dir: Path | None = None


@dataclass(frozen=True)
class ResolvedOverlay:
    name: str
    path: Path
    extent: tuple[float, float, float, float]
    opacity: float | None = None
    z_order: int | None = None


def _open_viewer(cfg: PipelineConfig, terrain_path: Path) -> Any:
    """Indirection point so tests can substitute a fake viewer."""
    return open_viewer_async(
        terrain_path=str(terrain_path),
        width=cfg.render.width,
        height=cfg.render.height,
        fov_deg=cfg.render.camera.fov,
    )


def render(
    cfg: PipelineConfig,
    terrain_path: Path,
    out_dir: Path,
    overlays: Sequence[ResolvedOverlay] = (),
) -> RenderResult:
    """Render a still or an animation into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    animation = build_camera_animation(cfg)

    with _open_viewer(cfg, terrain_path) as viewer:
        viewer.send_ipc(build_set_terrain(cfg))
        viewer.send_ipc(build_set_terrain_pbr(cfg))

        for overlay in overlays:
            viewer.send_ipc(
                build_load_overlay(
                    overlay.name,
                    overlay.path,
                    overlay.extent,
                    overlay.opacity,
                    overlay.z_order,
                )
            )
        if cfg.render.overlay_preserve_colors:
            viewer.send_ipc(build_set_overlay_preserve_colors(True))

        if cfg.animation.kind is AnimationKind.none or animation is None:
            snapshot = out_dir / cfg.render.snapshot_name
            viewer.snapshot(
                str(snapshot), width=cfg.render.width, height=cfg.render.height
            )
            return RenderResult(snapshot=snapshot)

        frames_dir = out_dir / FRAMES_DIRNAME
        viewer.render_animation(
            animation,
            str(frames_dir),
            fps=cfg.animation.fps,
            width=cfg.render.width,
            height=cfg.render.height,
        )
        return RenderResult(frames_dir=frames_dir)
