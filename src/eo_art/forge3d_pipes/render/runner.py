"""Drives the forge3d viewer: open, push payloads, snapshot or animate."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import forge3d as f3d

from eo_art.forge3d_pipes.config.schema import AnimationKind, PipelineConfig
from eo_art.forge3d_pipes.render.animation import build_camera_animation
from eo_art.forge3d_pipes.render.payloads import (
    build_set_terrain,
    build_set_terrain_pbr,
)

FRAMES_DIRNAME = "frames"


@dataclass(frozen=True)
class RenderResult:
    snapshot: Path | None = None
    frames_dir: Path | None = None


def _open_viewer(cfg: PipelineConfig, terrain_path: Path) -> Any:
    """Indirection point so tests can substitute a fake viewer."""
    # forge3d's __init__.pyi re-imports open_viewer_async without an `as`
    # alias or __all__ entry, so per PEP 484 stub rules type checkers treat
    # it as private even though it is documented and works at runtime.
    return f3d.open_viewer_async(  # ty: ignore[unresolved-attribute]
        terrain_path=str(terrain_path),
        width=cfg.render.width,
        height=cfg.render.height,
        fov_deg=cfg.render.camera.fov,
    )


def render(cfg: PipelineConfig, terrain_path: Path, out_dir: Path) -> RenderResult:
    """Render a still or an animation into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    animation = build_camera_animation(cfg)

    with _open_viewer(cfg, terrain_path) as viewer:
        viewer.send_ipc(build_set_terrain(cfg))
        viewer.send_ipc(build_set_terrain_pbr(cfg))

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
