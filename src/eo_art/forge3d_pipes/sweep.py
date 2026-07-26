"""Expand a sweep specification into independently renderable variants."""

import itertools
import re
from dataclasses import dataclass
from typing import Any

from eo_art.forge3d_pipes.config.schema import Sweep, SweepMode

_UNSAFE = re.compile(r"[^A-Za-z0-9.=_-]")

DEFAULT_VARIANT = "default"


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: tuple[str, ...]


def _slug(path: str, value: Any) -> str:
    leaf = path.rsplit(".", 1)[-1]
    return _UNSAFE.sub("_", f"{leaf}={value}")


def _variant(paths: list[str], values: tuple[Any, ...]) -> Variant:
    return Variant(
        name="__".join(_slug(path, value) for path, value in zip(paths, values)),
        overrides=tuple(f"{path}={value}" for path, value in zip(paths, values)),
    )


def expand(sweep: Sweep | None) -> list[Variant]:
    """Expand ``sweep`` into variants; a single default variant when unset."""
    if sweep is None or not sweep.params:
        return [Variant(name=DEFAULT_VARIANT, overrides=())]

    paths = list(sweep.params)
    for path in paths:
        if not isinstance(sweep.params[path], list):
            raise ValueError(
                f"sweep param {path!r} must be a list of values, "
                f"got {type(sweep.params[path]).__name__}"
            )
    value_lists = [list(sweep.params[path]) for path in paths]

    if sweep.mode is SweepMode.zip:
        lengths = {len(values) for values in value_lists}
        if len(lengths) > 1:
            raise ValueError(
                f"zip sweep requires equal-length value lists, "
                f"got lengths {sorted(lengths)}"
            )
        combinations = zip(*value_lists)
    else:
        combinations = itertools.product(*value_lists)

    return _deduplicate([_variant(paths, values) for values in combinations])


def _deduplicate(variants: list[Variant]) -> list[Variant]:
    """Suffix repeated names so no two variants share an output directory.

    Slugging maps every unsafe character to ``_``, so distinct values such as
    ``a b`` and ``a_b`` can produce the same name. Without this, the second
    variant would silently overwrite the first.
    """
    seen: dict[str, int] = {}
    unique = []
    for variant in variants:
        count = seen.get(variant.name, 0)
        seen[variant.name] = count + 1
        if count:
            variant = Variant(f"{variant.name}-{count + 1}", variant.overrides)
        unique.append(variant)
    return unique
