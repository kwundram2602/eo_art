"""Acquire stage: fetches and prepares the terrain/optical inputs the render
pipeline needs, ahead of `eo-art-f3d run`."""

from eo_art.forge3d_pipes.acquire.pipeline import AcquireResult, run_acquire
from eo_art.forge3d_pipes.acquire.schema import AcquireConfig

__all__ = ["AcquireConfig", "AcquireResult", "run_acquire"]
