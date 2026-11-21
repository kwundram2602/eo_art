"""Turn the config dataclasses into form field specs, and form edits back
into a sparse config file.

Scalar leaves (bool/int/float/str/Enum, optionally ``| None``) and
fixed-length tuples of one scalar type (``rgb_stretch.bands``) become form
fields. Other lists, dicts and optional sub-dataclasses are ``complex``; of
those, the render config's ``prepare`` chain and ``overlays`` list have their
own editors (``list_values`` / ``clean_chain`` / ``clean_overlays`` /
``check_lists``), the rest stays editable only as YAML.

A dataclass with an ``enabled: bool`` field gates the rest of its fields
(nested ones included): they only matter, and are only shown, while it's on.
"""

import dataclasses
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from omegaconf import MISSING, DictConfig, OmegaConf

Kind = Literal["bool", "int", "float", "str", "enum", "tuple", "complex"]


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
    # kind "tuple": the scalar kind of its items and how many there are.
    item: Kind | None = None
    size: int = 0

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


def _fixed_tuple(tp: Any) -> tuple[Kind, int] | None:
    """(item kind, length) for ``tuple[int, int, int]``-like types."""
    if typing.get_origin(tp) is not tuple:
        return None
    args = typing.get_args(tp)
    if not args or Ellipsis in args or len(set(args)) != 1 or args[0] not in _SCALARS:
        return None
    return _SCALARS[args[0]], len(args)


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
        elif (fixed := _fixed_tuple(tp)) is not None:
            specs.append(
                FieldSpec(
                    path,
                    "tuple",
                    None if required or default is None else list(default),
                    optional=optional,
                    required=required,
                    gate=field_gate,
                    item=fixed[0],
                    size=fixed[1],
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


@dataclass(frozen=True)
class GroupNode:
    """One dotted group of the form ("" = top-level fields), with its subgroups.

    A group's own ``enabled`` switch is lifted out of ``fields`` into ``gate``
    so the form can show it in the group's header.
    """

    path: str
    fields: tuple[FieldSpec, ...]
    children: tuple["GroupNode", ...]
    gate: FieldSpec | None = None

    def all_fields(self) -> list[FieldSpec]:
        """Gate and fields of this group and all its subgroups."""
        own = [self.gate] if self.gate is not None else []
        return [*own, *self.fields, *(s for c in self.children for s in c.all_fields())]


def _parent(group: str) -> str:
    return group.rpartition(".")[0]


def group_tree(specs: list[FieldSpec]) -> list[GroupNode]:
    """The scalar fields as a tree of groups, in schema order.

    The first node holds the top-level fields (path ""), if there are any; the
    others are the top-level groups. Groups that only contain subgroups (e.g.
    ``export`` around ``export.video``) are nodes without fields.
    """
    fields: dict[str, list[FieldSpec]] = {}
    gates: dict[str, FieldSpec] = {}

    def touch(group: str) -> None:
        if group in fields:
            return
        if group:
            touch(_parent(group))
        fields[group] = []

    for spec in specs:
        if spec.kind == "complex":
            continue
        touch(spec.group)
        if spec.group and spec.label == "enabled" and spec.kind == "bool":
            gates[spec.group] = spec
        else:
            fields[spec.group].append(spec)

    def build(path: str) -> GroupNode:
        children = [p for p in fields if p and _parent(p) == path]
        return GroupNode(
            path, tuple(fields[path]), tuple(map(build, children)), gates.get(path)
        )

    general = [GroupNode("", tuple(fields[""]), ())] if fields.get("") else []
    return general + [build(p) for p in fields if p and "." not in p]


_ABSENT = object()


def set_in_file(file_cfg: DictConfig, specs: list[FieldSpec]) -> set[str]:
    """Paths of the scalar fields the config file itself sets."""
    return {
        spec.path
        for spec in specs
        if spec.kind != "complex"
        and OmegaConf.select(file_cfg, spec.path, default=_ABSENT) is not _ABSENT
    }


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


def save_edits(
    path: Path, cls: type, initial: dict[str, Any], edited: dict[str, Any]
) -> dict[str, Any]:
    """Write the fields of ``edited`` that differ from ``initial`` into the
    config file at ``path`` and return them ({} if nothing changed).

    The result is type-checked against ``cls`` first; on a schema error the
    file is left untouched and the error is raised.
    """
    changes = changed(initial, edited)
    if not changes:
        return {}
    from eo_art.app.configs import load_file

    updated = apply_changes(load_file(path), changes)
    merged_with_defaults(cls, updated)
    OmegaConf.save(updated, path)
    return changes


# --- list editors: the render config's prepare chain and overlays ----------

CHAIN_KEY = "prepare"
OVERLAYS_KEY = "overlays"
LIST_KEYS = (OVERLAYS_KEY, CHAIN_KEY)
# Ops the pipeline adds itself (overlay alignment), not meant to be picked.
INTERNAL_OPS = {"validity_mask"}


def prep_ops() -> list[str]:
    """Prep ops a chain can use, in registration order."""
    import eo_art.forge3d_pipes.prep.ops  # noqa: F401  (registers the ops)
    from eo_art.forge3d_pipes.prep.registry import op_names

    return [name for name in op_names() if name not in INTERNAL_OPS]


def op_fields(name: str) -> list[FieldSpec]:
    """The parameters of prep op ``name`` as form fields."""
    import eo_art.forge3d_pipes.prep.ops  # noqa: F401
    from eo_art.forge3d_pipes.prep.registry import get_op

    return iter_fields(get_op(name).schema)


def overlay_fields() -> list[FieldSpec]:
    """The fields of one overlay, without its prepare chain."""
    from eo_art.forge3d_pipes.config.schema import OverlayCfg

    return [s for s in iter_fields(OverlayCfg) if s.path != CHAIN_KEY]


def parse_number(kind: str, value: Any) -> Any:
    """Typed text -> int/float; anything else (already a number, or text that
    isn't one) comes back unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return int(value) if kind == "int" else float(value)
    except ValueError:
        return value


def _clean(params: dict[str, Any], specs: list[FieldSpec], where: str) -> dict:
    """Parse typed numbers and drop unset (None) values."""
    by_name = {s.path: s for s in specs}
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        spec = by_name.get(key)
        if spec is not None and spec.kind in ("int", "float"):
            value = parse_number(spec.kind, value)
            if isinstance(value, str):
                raise ValueError(f"{where}.{key}: not a number: {value!r}")
        elif spec is not None and spec.kind == "tuple":
            value = [parse_number(spec.item or "float", v) for v in value]
            if any(isinstance(v, str) for v in value):
                raise ValueError(f"{where}.{key}: not a number: {value!r}")
        out[key] = value
    return out


def clean_chain(chain: list[dict[str, Any]], where: str = CHAIN_KEY) -> list[dict]:
    """A prepare chain as it gets saved (see ``_clean``)."""
    return [
        {
            "op": entry["op"],
            **_clean(
                {k: v for k, v in entry.items() if k != "op"},
                op_fields(entry["op"]),
                f"{where}[{index}]",
            ),
        }
        for index, entry in enumerate(chain)
    ]


def clean_overlays(overlays: list[dict[str, Any]]) -> list[dict]:
    """The overlay list as it gets saved: fields cleaned, chains cleaned, an
    empty chain left out."""
    specs = overlay_fields()
    cleaned = []
    for index, overlay in enumerate(overlays):
        where = f"{OVERLAYS_KEY}[{index}]"
        fields = {k: v for k, v in overlay.items() if k != CHAIN_KEY}
        entry = _clean(fields, specs, where)
        chain = clean_chain(overlay.get(CHAIN_KEY) or [], f"{where}.{CHAIN_KEY}")
        if chain:
            entry[CHAIN_KEY] = chain
        cleaned.append(entry)
    return cleaned


def clean_lists(values: dict[str, list]) -> dict[str, list]:
    """The given lists (keys of ``LIST_KEYS``) as they get saved; raises
    ValueError on a bad number. Keys not given stay out, so a view that only
    edits one list never writes the other."""
    cleaners = {OVERLAYS_KEY: clean_overlays, CHAIN_KEY: clean_chain}
    return {key: cleaners[key](items) for key, items in values.items()}


def list_values(
    file_cfg: DictConfig, keys: Sequence[str] = LIST_KEYS
) -> dict[str, list]:
    """The file's own lists as plain lists ([] if unset)."""
    return {
        key: typing.cast(list, OmegaConf.to_container(file_cfg[key]))
        if key in file_cfg and file_cfg[key] is not None
        else []
        for key in keys
    }


def check_lists(values: dict[str, list]) -> None:
    """Validate cleaned lists like a run would; raises with the list index."""
    from eo_art.forge3d_pipes.config.schema import OverlayCfg
    from eo_art.forge3d_pipes.prep.registry import validate_chain

    validate_chain(values.get(CHAIN_KEY, []))
    for index, overlay in enumerate(values.get(OVERLAYS_KEY, [])):
        where = f"{OVERLAYS_KEY}[{index}]"
        missing = [k for k in ("name", "path") if not overlay.get(k)]
        if missing:
            raise ValueError(f"{where}: {' and '.join(missing)} must be set")
        try:
            validate_chain(overlay.get(CHAIN_KEY, []))
            OmegaConf.to_object(
                OmegaConf.merge(OmegaConf.structured(OverlayCfg), overlay)
            )
        except Exception as exc:
            raise ValueError(f"{where}: {exc}") from exc
