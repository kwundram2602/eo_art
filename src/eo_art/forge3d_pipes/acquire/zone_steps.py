"""Pick one acquisition per zone and interval so a zoned timeseries shows the
whole scene at about the same time in every step.

Sentinel-2's ~290 km swath images all zones of an AOI on the same overflight,
so zones only end up on different dates when each picks its own least cloudy
scene. Here the zones' candidates are compared per interval instead: the
window of ``max_offset_days`` that gives the most zones a scene, at the
lowest mean cloud, wins. A zone with no scene in that window takes its scene
nearest to it, and the step records by how many days it lies outside.

``single_date`` is the strict alternative: a step only exists on a day on
which every zone has a scene, so the whole AOI comes from one acquisition
(no date seams between zones, one shift per step for co-registration).
"""

from dataclasses import dataclass
from datetime import date, timedelta

from eo_art.forge3d_pipes.acquire.sentinel_bridge import Candidate

# Mean-cloud stand-in for scenes without any cloud metadata: rank them last.
_MISSING_CLOUD = 100.0


@dataclass(frozen=True)
class ZonePick:
    # Acquisition date (YYYY-MM-DD).
    date: str
    cloud: float | None
    # Days outside the step's window (0: inside, or no window).
    outside_days: int = 0


@dataclass(frozen=True)
class Step:
    # Start date of the interval.
    bin: str
    # Zone id -> chosen acquisition, in zone order.
    picks: dict[str, ZonePick]
    # (first, last) day of the chosen window; None without max_offset_days.
    window: tuple[str, str] | None
    # Zones without any candidate in this interval.
    empty_zones: list[str]

    @property
    def date(self) -> str:
        """Earliest acquisition of the step; names its mosaic."""
        return min(p.date for p in self.picks.values())

    @property
    def spread_days(self) -> int:
        """Days between the earliest and the latest acquisition."""
        days = [date.fromisoformat(p.date) for p in self.picks.values()]
        return (max(days) - min(days)).days

    @property
    def relaxed(self) -> dict[str, int]:
        """Zone id -> days outside the window, for zones outside it."""
        return {z: p.outside_days for z, p in self.picks.items() if p.outside_days}


def _score(candidate: Candidate) -> float:
    cloud = candidate.cloud
    return _MISSING_CLOUD if cloud is None else cloud


def _best(candidates: list[Candidate]) -> Candidate:
    # Least cloudy; scenes without cloud metadata last, ties to the earlier.
    return min(candidates, key=lambda c: (c.cloud is None, _score(c), c.datetime))


def _outside(day: date, lo: date, hi: date) -> int:
    if day < lo:
        return (lo - day).days
    if day > hi:
        return (day - hi).days
    return 0


def _pick(candidate: Candidate, outside_days: int = 0) -> ZonePick:
    return ZonePick(candidate.datetime, candidate.cloud, outside_days)


def _coordinate_bin(
    by_zone: dict[str, list[Candidate]], max_offset_days: int | None
) -> tuple[dict[str, ZonePick], tuple[str, str] | None]:
    if max_offset_days is None:
        return {z: _pick(_best(cands)) for z, cands in by_zone.items()}, None

    offset = timedelta(days=max_offset_days)
    anchors = sorted({c.datetime for cands in by_zone.values() for c in cands})
    best_key = None
    best_window: tuple[date, date] | None = None
    best_inside: dict[str, Candidate] = {}
    # The earliest date of an optimal window is always some candidate's date.
    for anchor in anchors:
        lo = date.fromisoformat(anchor)
        hi = lo + offset
        inside = {
            z: _best(window)
            for z, cands in by_zone.items()
            if (
                window := [
                    c for c in cands if lo <= date.fromisoformat(c.datetime) <= hi
                ]
            )
        }
        mean = sum(_score(c) for c in inside.values()) / len(inside)
        key = (-len(inside), mean, anchor)
        if best_key is None or key < best_key:
            best_key, best_window, best_inside = key, (lo, hi), inside

    assert best_window is not None
    lo, hi = best_window
    picks = {}
    for zone, cands in by_zone.items():
        if zone in best_inside:
            picks[zone] = _pick(best_inside[zone])
            continue
        # Relaxed: the scene nearest to the window, then the least cloudy.
        nearest = min(
            cands,
            key=lambda c: (
                _outside(date.fromisoformat(c.datetime), lo, hi),
                c.cloud is None,
                _score(c),
            ),
        )
        picks[zone] = _pick(
            nearest, _outside(date.fromisoformat(nearest.datetime), lo, hi)
        )
    return picks, (lo.isoformat(), hi.isoformat())


def _common_date(by_zone: dict[str, list[Candidate]]) -> dict[str, ZonePick] | None:
    """Picks on the least cloudy day every zone has a scene on (mean over
    the zones, ties to the earlier day), or None without such a day."""
    per_zone = [{c.datetime: c for c in cands} for cands in by_zone.values()]
    shared = set(per_zone[0]).intersection(*per_zone[1:])
    if not shared:
        return None

    def key(day: str) -> tuple[float, str]:
        return (sum(_score(z[day]) for z in per_zone) / len(per_zone), day)

    day = min(shared, key=key)
    return {
        zone: _pick(_best([c for c in cands if c.datetime == day]))
        for zone, cands in by_zone.items()
    }


def coordinate_steps(
    candidates: dict[str, list[Candidate]],
    bins: list[str],
    max_offset_days: int | None,
    single_date: bool = False,
) -> tuple[list[Step], list[str]]:
    """One step per interval in ``bins`` with a pick for every zone that has
    a candidate in it.

    ``candidates`` maps zone id -> its candidates (tagged with their
    ``bin``), in zone order. Returns (steps in time order, start dates of
    intervals where no zone has any candidate). With ``single_date`` an
    interval only gets a step if every zone has a scene on one shared day;
    other intervals are returned as skipped too.
    """
    if single_date and max_offset_days is not None:
        raise ValueError("single_date and max_offset_days exclude each other")
    steps: list[Step] = []
    skipped: list[str] = []
    for bin_start in bins:
        by_zone = {
            zone: in_bin
            for zone, cands in candidates.items()
            if (in_bin := [c for c in cands if c.bin == bin_start])
        }
        if not by_zone:
            skipped.append(bin_start)
            continue
        if single_date:
            picks = _common_date(by_zone) if len(by_zone) == len(candidates) else None
            if picks is None:
                skipped.append(bin_start)
                continue
            steps.append(Step(bin=bin_start, picks=picks, window=None, empty_zones=[]))
            continue
        picks, window = _coordinate_bin(by_zone, max_offset_days)
        steps.append(
            Step(
                bin=bin_start,
                picks=picks,
                window=window,
                empty_zones=[z for z in candidates if z not in by_zone],
            )
        )
    return steps, skipped


def describe_step(step: Step, max_offset_days: int | None) -> str:
    """One log line: dates, window, relaxed zones and the actual spread."""
    dates = sorted({p.date for p in step.picks.values()})
    parts = [
        f"{step.bin}: {len(step.picks)} zone(s) on "
        + (
            dates[0]
            if len(dates) == 1
            else f"{len(dates)} dates {dates[0]}..{dates[-1]}"
        )
    ]
    if step.window is not None:
        parts.append(f"window {step.window[0]}..{step.window[1]} ({max_offset_days} d)")
    if step.relaxed:
        parts.append(
            "relaxed: "
            + ", ".join(
                f"{z} {step.picks[z].date} (+{days} d outside)"
                for z, days in step.relaxed.items()
            )
        )
    parts.append(f"spread {step.spread_days} d")
    if step.empty_zones:
        parts.append("no scene: " + ", ".join(step.empty_zones))
    return " · ".join(parts)
