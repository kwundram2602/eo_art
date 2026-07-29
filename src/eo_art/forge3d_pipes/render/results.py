"""Shared return/input types for both render backends. No forge3d import, no I/O."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderResult:
    snapshot: Path | None = None
    frames_dir: Path | None = None


@dataclass(frozen=True)
class ResolvedOverlay:
    name: str
    path: Path
    extent: tuple[float, float, float, float]
    opacity: float | None = None
    z_order: int | None = None
