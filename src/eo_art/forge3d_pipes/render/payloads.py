"""Pure config-to-IPC-payload translation. No forge3d import, no I/O."""

from dataclasses import asdict
from enum import Enum
from typing import Any

from eo_art.forge3d_pipes.config.schema import PipelineConfig


def _plain(value: Any) -> Any:
    """Recursively convert enums to their values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def build_set_terrain(cfg: PipelineConfig) -> dict[str, Any]:
    """Camera, terrain scale, and sun in one command."""
    camera = cfg.render.camera
    sun = cfg.render.sun
    return {
        "cmd": "set_terrain",
        "phi": camera.phi,
        "theta": camera.theta,
        "radius": camera.radius,
        "fov": camera.fov,
        "zscale": cfg.render.terrain.zscale,
        "sun_azimuth": sun.azimuth,
        "sun_elevation": sun.elevation,
        "sun_intensity": sun.intensity,
        "ambient": sun.ambient,
    }


def build_set_terrain_pbr(cfg: PipelineConfig) -> dict[str, Any]:
    """Full PBR block, mirroring the nested schema one-to-one."""
    payload = _plain(asdict(cfg.render.pbr))
    return {"cmd": "set_terrain_pbr", **payload}


def build_load_overlay(
    name: str,
    path: Any,
    extent: tuple[float, float, float, float],
    opacity: float | None = None,
    z_order: int | None = None,
) -> dict[str, Any]:
    """Mirrors ``ViewerHandle.load_overlay``'s IPC dict.

    Unlike forge3d's own convenience method, ``extent`` is required here, not
    optional: this pipeline always resolves an extent (auto-computed or a
    manual override) before building this payload.
    """
    cmd: dict[str, Any] = {
        "cmd": "load_overlay",
        "name": str(name),
        "path": str(path),
        "extent": list(extent),
    }
    if opacity is not None:
        cmd["opacity"] = float(opacity)
    if z_order is not None:
        cmd["z_order"] = int(z_order)
    return cmd


def build_set_overlay_preserve_colors(preserve_colors: bool) -> dict[str, Any]:
    return {
        "cmd": "set_overlay_preserve_colors",
        "preserve_colors": bool(preserve_colors),
    }
