"""Classify, load and validate project configs without ever raising, so the
UI can show a broken config instead of crashing on it."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from eo_art.app.projects import Project


class ConfigKind(Enum):
    acquire = "acquire"
    render = "render"


# Top-level keys only an acquire config has; anything else is a render layer.
_ACQUIRE_KEYS = {"aoi_path", "ee_project", "sentinel_sr_dir", "sentinel", "dem", "dtm"}


@dataclass(frozen=True)
class Validation:
    resolved: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def load_file(path: Path) -> DictConfig:
    loaded = OmegaConf.load(path)
    if loaded is None or not isinstance(loaded, DictConfig):
        return OmegaConf.create({})
    return loaded


def detect_kind(path: Path) -> ConfigKind:
    try:
        keys = set(load_file(path).keys())
    except Exception:
        return ConfigKind.render
    return ConfigKind.acquire if keys & _ACQUIRE_KEYS else ConfigKind.render


def schema_for(kind: ConfigKind) -> type:
    if kind is ConfigKind.acquire:
        from eo_art.forge3d_pipes.acquire.schema import AcquireConfig

        return AcquireConfig
    from eo_art.forge3d_pipes.config.schema import PipelineConfig

    return PipelineConfig


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def validate_render(
    layers: Sequence[Path], overrides: Sequence[str] = ()
) -> Validation:
    """Merge render layers left to right exactly like ``eo-art-f3d run``."""
    from eo_art.forge3d_pipes.config.loader import load_raw, to_pipeline

    try:
        raw = load_raw(list(layers), overrides)
    except Exception as exc:
        return Validation(None, _error(exc))
    resolved = OmegaConf.to_yaml(raw)
    try:
        to_pipeline(raw)
    except Exception as exc:
        return Validation(resolved, _error(exc))
    return Validation(resolved, None)


def validate_acquire(path: Path) -> Validation:
    from eo_art.forge3d_pipes.acquire.schema import AcquireConfig

    try:
        raw = OmegaConf.merge(OmegaConf.structured(AcquireConfig), load_file(path))
    except Exception as exc:
        return Validation(None, _error(exc))
    resolved = OmegaConf.to_yaml(raw)
    try:
        OmegaConf.to_object(raw)
    except Exception as exc:
        return Validation(resolved, _error(exc))
    return Validation(resolved, None)


def validate_sr(path: Path) -> Validation:
    """Only the SR slice of an acquire config, as ``eo-art-sr`` reads it."""
    from eo_art.sr_cli import load_sr_config

    try:
        cfg = load_sr_config(str(path))
    except Exception as exc:
        return Validation(None, _error(exc))
    return Validation(OmegaConf.to_yaml(OmegaConf.structured(cfg)), None)


def overlay_names(layers: Sequence[Path]) -> list[str]:
    """Overlay names of the merged render config ([] if it doesn't merge)."""
    from eo_art.forge3d_pipes.config.loader import load_raw

    try:
        raw = load_raw(list(layers))
    except Exception:
        return []
    return [str(overlay.name) for overlay in raw.overlays]


def set_top_level_value(path: Path, key: str, value: str) -> None:
    """Set ``key: value`` in a YAML file by editing its line in place.

    Unlike an OmegaConf load/save round trip this keeps the file's comments
    and layout; the key is appended if it isn't there yet.
    """
    text = path.read_text() if path.exists() else ""
    # JSON string quoting is valid YAML and safe for any path.
    line = f"{key}: {json.dumps(value)}"
    # Group 1 keeps a trailing comment on the replaced line.
    pattern = re.compile(rf"^{re.escape(key)}:[^\n#]*?([ \t]+#[^\n]*)?$", re.MULTILINE)
    text, count = pattern.subn(lambda m: line + (m.group(1) or ""), text, count=1)
    if not count:
        text = (text if not text or text.endswith("\n") else text + "\n") + line + "\n"
    path.write_text(text)


def configs_of_kind(project: Project, kind: ConfigKind) -> list[Path]:
    return [p for p in project.config_files() if detect_kind(p) is kind]


@dataclass(frozen=True)
class SrSettings:
    """The chip/zone part of an acquire config's ``sentinel`` block."""

    edge_size: int
    resolution: int
    zones: bool = False
    clip_to_aoi: bool = True

    @property
    def edge_m(self) -> float:
        return float(self.edge_size * self.resolution)


def sr_settings(project: Project) -> SrSettings:
    """SR chip size and ``sentinel.zones`` from the first readable acquire
    config (else the schema defaults)."""
    from eo_art.forge3d_pipes.acquire.schema import Sentinel, Zones

    default = SrSettings(
        Sentinel.edge_size, Sentinel.resolution, Zones.enabled, Zones.clip_to_aoi
    )
    for path in configs_of_kind(project, ConfigKind.acquire):
        try:
            sentinel = load_file(path).get("sentinel") or {}
            zones = sentinel.get("zones") or {}
            return SrSettings(
                edge_size=int(sentinel.get("edge_size", default.edge_size)),
                resolution=int(sentinel.get("resolution", default.resolution)),
                zones=bool(zones.get("enabled", default.zones)),
                clip_to_aoi=bool(zones.get("clip_to_aoi", default.clip_to_aoi)),
            )
        except Exception:
            continue
    return default


def sr_edge_m(project: Project) -> float:
    """SR chip edge in metres, from the first readable acquire config (else the
    schema defaults): ``sentinel.edge_size`` x ``sentinel.resolution``."""
    return sr_settings(project).edge_m
