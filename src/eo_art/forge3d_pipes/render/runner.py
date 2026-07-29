"""Dispatches rendering to the viewer or offscreen backend.

Both backends keep forge3d imports lazy (inside their own functions), so
importing this module -- and everything upstream of it, including
``forge3d_pipes/__init__.py`` -- never touches forge3d's native library
before ``gpu.select_gpu`` sets the Vulkan ICD environment variables.
"""

from collections.abc import Sequence
from pathlib import Path

from eo_art.forge3d_pipes.config.schema import PipelineConfig, RenderBackend
from eo_art.forge3d_pipes.render.backends import offscreen, viewer
from eo_art.forge3d_pipes.render.results import RenderResult, ResolvedOverlay

__all__ = ["RenderResult", "ResolvedOverlay", "render"]


def render(
    cfg: PipelineConfig,
    terrain_path: Path,
    out_dir: Path,
    overlays: Sequence[ResolvedOverlay] = (),
) -> RenderResult:
    """Render a still or an animation into ``out_dir``."""
    if cfg.render.backend is RenderBackend.offscreen:
        return offscreen.render(cfg, terrain_path, out_dir, overlays=overlays)
    return viewer.render(cfg, terrain_path, out_dir, overlays=overlays)
