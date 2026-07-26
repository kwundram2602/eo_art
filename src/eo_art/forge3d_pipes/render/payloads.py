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
