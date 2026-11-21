"""Turn the config dataclasses into form field specs, and form edits back
into a sparse config file.

Only scalar leaves (bool/int/float/str/Enum, optionally ``| None``) become
form fields. Lists, dicts, tuples and optional sub-dataclasses (``prepare``,
``overlays``, ``sweep``, ``erosion_kwargs``, ...) are ``complex`` and stay
editable only as YAML.

A dataclass with an ``enabled: bool`` field gates the rest of its fields
(nested ones included): they only matter, and are only shown, while it's on.
"""

import dataclasses
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from omegaconf import MISSING, DictConfig, OmegaConf

Kind = Literal["bool", "int", "float", "str", "enum", "complex"]


@dataclass(frozen=True)
class FieldSpec:
    path: str
    kind: Kind
    default: Any
    optional: bool = False
    required: bool = False
    choices: tuple[str, ...] = ()
    # Path of the ``enabled`` field this field depends on, if any.
    gate: str | None = None

    @property
    def group(self) -> str:
        """Dotted parent path ("" for top-level fields), used for grouping."""
        return self.path.rpartition(".")[0]

    @property
    def label(self) -> str:
        return self.path.rpartition(".")[2]


_SCALARS: dict[type, Kind] = {bool: "bool", int: "int", float: "float", str: "str"}


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _default_of(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return MISSING


def _plain(value: Any) -> Any:
    return value.name if isinstance(value, Enum) else value


def iter_fields(cls: Any, prefix: str = "", gate: str | None = None) -> list[FieldSpec]:
    """Every leaf field of a config dataclass, depth-first, as a FieldSpec."""
    hints = typing.get_type_hints(cls)
    own_gate = f"{prefix}enabled" if hints.get("enabled") is bool else None
    specs: list[FieldSpec] = []
    for f in dataclasses.fields(cls):
        path = f"{prefix}{f.name}"
        tp, optional = _unwrap_optional(hints[f.name])
        default = _default_of(f)
        required = default is MISSING or default == MISSING
        field_gate = gate if path == own_gate else (own_gate or gate)

        if dataclasses.is_dataclass(tp) and not optional:
            specs.extend(iter_fields(tp, prefix=f"{path}.", gate=field_gate))
        elif isinstance(tp, type) and issubclass(tp, Enum):
            specs.append(
                FieldSpec(
                    path,
                    "enum",
                    None if required else _plain(default),
                    optional=optional,
                    required=required,
                    choices=tuple(member.name for member in tp),
                    gate=field_gate,
                )
            )
        elif tp in _SCALARS:
            specs.append(
                FieldSpec(
                    path,
                    _SCALARS[tp],
                    None if required else default,
                    optional=optional,
                    required=required,
                    gate=field_gate,
                )
            )
        else:
            specs.append(
                FieldSpec(
                    path,
                    "complex",
                    None,
                    optional=optional,
                    required=required,
                    gate=field_gate,
                )
            )
    return specs


def is_active(spec: FieldSpec, values: dict[str, Any], specs: list[FieldSpec]) -> bool:
    """Whether every ``enabled`` gate above ``spec`` is on in ``values``."""
    by_path = {s.path: s for s in specs}
    gate = spec.gate
    while gate is not None:
        if not values.get(gate):
            return False
        gate = by_path[gate].gate
    return True


def merged_with_defaults(cls: type, file_cfg: DictConfig) -> DictConfig:
    """Schema defaults overlaid with one config file (type-checked merge)."""
    return typing.cast(DictConfig, OmegaConf.merge(OmegaConf.structured(cls), file_cfg))


def current_values(merged: DictConfig, specs: list[FieldSpec]) -> dict[str, Any]:
    """Path -> plain value (enum members as their names, MISSING as None)."""
    return {
        spec.path: _plain(OmegaConf.select(merged, spec.path))
        for spec in specs
        if spec.kind != "complex"
    }


def changed(initial: dict[str, Any], edited: dict[str, Any]) -> dict[str, Any]:
    return {path: value for path, value in edited.items() if initial.get(path) != value}


def apply_changes(file_cfg: DictConfig, changes: dict[str, Any]) -> DictConfig:
    """Write only ``changes`` into a copy of the file's own content.

    Untouched keys stay absent, so a sparse layer (a look that only sets the
    sun) is still sparse after saving through the form.
    """
    updated = file_cfg.copy()
    for path, value in changes.items():
        OmegaConf.update(updated, path, _plain(value), merge=True)
    return updated
