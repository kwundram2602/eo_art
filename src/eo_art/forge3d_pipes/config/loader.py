"""Config loading: merge defaults, files, and CLI overrides, then validate."""

from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config import schema
from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.prep.registry import validate_chain


def _get_all_enums() -> list[type[Enum]]:
    """Get all Enum classes from the schema module."""
    enums = []
    for name in dir(schema):
        obj = getattr(schema, name)
        if isinstance(obj, type) and issubclass(obj, Enum):
            enums.append(obj)
    return enums


def _convert_enum_values(obj: object) -> object:
    """Recursively convert enum values to enum names for OmegaConf compatibility.

    OmegaConf expects enum member names (uppercase), not values (lowercase).
    This function converts string values that match enum values to their member names.
    """
    if isinstance(obj, dict):
        return {k: _convert_enum_values(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_enum_values(item) for item in obj)
    elif isinstance(obj, str):
        # Try to find any Enum with this value and convert to name
        for enum_cls in _get_all_enums():
            for member in enum_cls:
                if member.value == obj:
                    return member.name
    return obj


def load_raw(
    paths: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
) -> DictConfig:
    """Merge schema defaults, config files, and dotlist overrides.

    Merge order, later wins: defaults -> files (in order) -> dotlist.
    ``out`` is applied in the dotlist layer, so it beats the files.
    """
    cfg = OmegaConf.structured(PipelineConfig)
    for path in paths:
        # Load YAML as plain dict, convert enum values to names, then merge
        with open(Path(path)) as f:
            raw_dict = yaml.safe_load(f) or {}
        converted = _convert_enum_values(raw_dict)
        cfg = OmegaConf.merge(cfg, converted)
    dotlist = list(overrides)
    if out is not None:
        dotlist.append(f"run.out_dir={out}")
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def to_pipeline(cfg: DictConfig) -> PipelineConfig:
    """Convert to dataclasses (running range checks) and validate prep ops."""
    obj: PipelineConfig = OmegaConf.to_object(cfg)
    validate_chain(obj.prepare)
    return obj


def load_config(
    paths: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
) -> PipelineConfig:
    return to_pipeline(load_raw(paths, overrides, out))
