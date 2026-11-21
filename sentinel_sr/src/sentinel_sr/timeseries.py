"""Pick one Sentinel-2 scene per time interval for a timeseries SR run.

Runs on STAC metadata only (acquisition time + scene cloud cover), so the
selection happens before any pixels are fetched or super-resolved.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# pandas period frequencies; a week is Monday-Sunday.
FREQUENCIES = {"day": "D", "week": "W-SUN", "month": "M", "year": "Y"}


@dataclass(frozen=True)
class Selection:
    # Cube time indices of the chosen scenes, in time order.
    indices: list[int]
    # ISO start dates of the intervals that had no scene at all.
    skipped_bins: list[str]
    # ISO start dates of the intervals already covered by ``filled_dates``.
    filled_bins: list[str] = field(default_factory=list)


def _period(value, freq: str) -> pd.Period:
    return pd.Timestamp(value).tz_localize(None).to_period(freq)


def _cloud_key(cloud_cover: float | None) -> tuple[bool, float]:
    # Scenes without cloud metadata sort after every scene that has it.
    missing = cloud_cover is None or bool(np.isnan(cloud_cover))
    return missing, 0.0 if missing else float(cloud_cover)


@dataclass(frozen=True)
class Bins:
    """``every`` x ``interval`` wide time bins, counted from ``start_date``."""

    start_date: str
    end_date: str
    interval: str
    every: int = 1

    def __post_init__(self) -> None:
        if self.interval not in FREQUENCIES:
            raise ValueError(
                f"interval must be one of {sorted(FREQUENCIES)}, got {self.interval!r}"
            )
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")

    @property
    def _freq(self) -> str:
        return FREQUENCIES[self.interval]

    def index(self, value) -> int:
        """Bin of a time; negative or above ``last`` when out of range."""
        origin = _period(self.start_date, self._freq).ordinal
        return (_period(value, self._freq).ordinal - origin) // self.every

    @property
    def last(self) -> int:
        return self.index(self.end_date)

    def start(self, b: int) -> str:
        first = _period(self.start_date, self._freq)
        return (first + b * self.every).start_time.date().isoformat()

    def of(self, dates: Sequence) -> set[int]:
        """In-range bins containing any of ``dates``."""
        return {b for b in map(self.index, dates) if 0 <= b <= self.last}


def select_timesteps(
    times: Sequence,
    cloud_covers: Sequence[float | None],
    start_date: str,
    end_date: str,
    interval: str,
    every: int = 1,
    filled_dates: Sequence = (),
) -> Selection:
    """One scene per ``every`` x ``interval``, counted from ``start_date``.

    Within an interval the scene with the lowest cloud cover wins (ties go to
    the earlier acquisition). Intervals already holding one of
    ``filled_dates`` get no scene (``filled_bins``); the remaining intervals
    up to ``end_date`` without any scene are reported in ``skipped_bins``.
    """
    if len(times) != len(cloud_covers):
        raise ValueError(
            f"got {len(times)} times but {len(cloud_covers)} cloud cover values"
        )
    bins = Bins(start_date, end_date, interval, every)
    filled = bins.of(filled_dates)

    best: dict[int, int] = {}
    for i, (time, cloud) in enumerate(zip(times, cloud_covers, strict=True)):
        b = bins.index(time)
        if not 0 <= b <= bins.last or b in filled:
            continue
        current = best.get(b)
        if current is None or (_cloud_key(cloud), pd.Timestamp(time)) < (
            _cloud_key(cloud_covers[current]),
            pd.Timestamp(times[current]),
        ):
            best[b] = i

    empty = [b for b in range(bins.last + 1) if b not in best and b not in filled]
    indices = sorted(best.values(), key=lambda i: pd.Timestamp(times[i]))
    return Selection(
        indices=indices,
        skipped_bins=[bins.start(b) for b in empty],
        filled_bins=[bins.start(b) for b in sorted(filled)],
    )
