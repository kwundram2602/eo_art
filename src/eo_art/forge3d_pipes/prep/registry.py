"""Registry mapping prep-op names to implementations and their schemas.

Entries are validated at config-load time, so an unknown op or a bad
parameter fails before any raster work starts.
"""

import hashlib
import json
import shutil
import tempfile
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
    """Register a prep op under ``name`` with its parameter dataclass.

    An op has the signature ``(src, dst, cfg) -> Path`` and must return the
    path it wrote, normally ``dst``. Returning ``src`` unchanged is supported
    and means "this op was a no-op"; the chain will copy rather than move in
    that case, so the caller's source raster is never relocated.
    """

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


def chain_cache_key(src: Path, entries: list[Any]) -> str:
    """Hash the source identity plus the canonical prep chain."""
    stat = src.stat()
    payload = {
        "src": str(src.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "chain": entries,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_prep_chain(
    src: Path,
    entries: list[Any],
    cache_dir: Path,
    use_cache: bool = True,
) -> Path:
    """Run the prep chain, caching the result by chain hash.

    Returns ``src`` unchanged when the chain is empty. Intermediates are
    written to a temporary directory and only the final result is promoted
    into the cache, so a failing op leaves no partial cache entry.

    If every op was a no-op (the chain's final path is still the source), the
    source is copied into the cache rather than moved — moving would relocate
    the caller's raster out of its original location.
    """
    src = Path(src)
    if not entries:
        return src

    key = chain_cache_key(src, entries)
    final = Path(cache_dir) / f"{key}.tif"
    if use_cache and final.exists():
        return final

    final.parent.mkdir(parents=True, exist_ok=True)
    current = src
    with tempfile.TemporaryDirectory(dir=final.parent) as tmp:
        for index, entry in enumerate(entries):
            op, cfg = validate_entry(entry)
            dst = Path(tmp) / f"{index:02d}_{op.name}.tif"
            current = Path(op.func(current, dst, cfg))
        if current.resolve() == src.resolve():
            shutil.copy2(current, final)
        else:
            shutil.move(str(current), final)
    return final
