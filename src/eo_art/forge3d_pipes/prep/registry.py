"""Registry mapping prep-op names to implementations and their schemas.

Entries are validated at config-load time, so an unknown op or a bad
parameter fails before any raster work starts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import OmegaConf

OpFunc = Callable[[Path, Path, Any], Path]
F = TypeVar("F", bound=OpFunc)


@dataclass(frozen=True)
class RegisteredOp:
    name: str
    func: OpFunc
    schema: type


_OPS: dict[str, RegisteredOp] = {}


def register_op(name: str, schema: type) -> Callable[[F], F]:
    """Register a prep op under ``name`` with its parameter dataclass."""

    def decorator(func: F) -> F:
        if name in _OPS:
            raise ValueError(f"prep op {name!r} is already registered")
        _OPS[name] = RegisteredOp(name=name, func=func, schema=schema)
        return func

    return decorator


def get_op(name: str) -> RegisteredOp:
    try:
        return _OPS[name]
    except KeyError:
        known = ", ".join(sorted(_OPS)) or "<none>"
        raise ValueError(f"unknown prep op {name!r}; known ops: {known}") from None


def validate_entry(entry: Any) -> tuple[RegisteredOp, Any]:
    """Validate one ``prepare`` entry against its op's schema."""
    params = dict(entry)
    try:
        name = params.pop("op")
    except KeyError:
        raise ValueError(f"prep entry missing 'op' key: {entry!r}") from None
    op = get_op(name)
    merged = OmegaConf.merge(OmegaConf.structured(op.schema), params)
    return op, OmegaConf.to_object(merged)


def validate_chain(entries: list[Any]) -> list[tuple[RegisteredOp, Any]]:
    """Validate every entry, prefixing failures with their index."""
    validated = []
    for index, entry in enumerate(entries):
        try:
            validated.append(validate_entry(entry))
        except Exception as exc:
            raise type(exc)(f"prepare[{index}]: {exc}") from exc
    return validated
