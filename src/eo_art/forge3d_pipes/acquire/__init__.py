"""Acquire stage: fetches and prepares the terrain/optical inputs the render
pipeline needs, ahead of `eo-art-f3d run`."""

from eo_art.forge3d_pipes.acquire.pipeline import (
    AcquireResult,
    SrZonesResult,
    TimeseriesResult,
    ZoneTimeseriesResult,
    run_acquire,
    run_sr,
    run_sr_timeseries,
    run_sr_timeseries_zones,
    run_sr_zones,
)
from eo_art.forge3d_pipes.acquire.schema import AcquireConfig, SrConfig

__all__ = [
    "AcquireConfig",
    "AcquireResult",
    "SrConfig",
    "SrZonesResult",
    "TimeseriesResult",
    "ZoneTimeseriesResult",
    "run_acquire",
    "run_sr",
    "run_sr_timeseries",
    "run_sr_timeseries_zones",
    "run_sr_zones",
]
