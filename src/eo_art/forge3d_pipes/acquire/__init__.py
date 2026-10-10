"""Acquire stage: fetches and prepares the terrain/optical inputs the render
pipeline needs, ahead of `eo-art-f3d run`."""

from eo_art.forge3d_pipes.acquire.pipeline import (
    AcquireResult,
    run_acquire,
    run_sr,
)
from eo_art.forge3d_pipes.acquire.schema import AcquireConfig, SrConfig

__all__ = ["AcquireConfig", "AcquireResult", "SrConfig", "run_acquire", "run_sr"]
