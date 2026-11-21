"""Collapse STAC items that are the same Sentinel-2 overflight.

Planetary Computer lists one acquisition several times: once per MGRS tile
covering the AOI (an AOI in a tile overlap sits on two), and once per
processing baseline (e.g. the original product plus a later reprocessing).
They carry the same pixels, so only one per overflight is worth the SR run.

Tiles from another UTM zone than the output grid are dropped: stackstac
(via cubo) reprojects them with nearest-neighbour resampling, and the few
degrees of grid rotation between zones turn into periodic staircase seams
(every ~12 px at 10 m around 54 N) that the SR model sharpens into blocks.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# S2A_MSIL2A_20230214T105141_R051_T32ULE_20240815T232641
# mission, sensing time, relative orbit, tile, generation time
_ID_RE = re.compile(
    r"^(?P<mission>S2[A-D])_MSIL2A_(?P<sensing>\d{8}T\d{6})_(?:N\d{4}_)?"
    r"R(?P<orbit>\d{3})_T(?P<tile>[0-9A-Z]{5})_(?P<generation>\d{8}T\d{6})$"
)


@dataclass(frozen=True)
class Dedupe:
    # Indices of the items to keep, in input order.
    kept: list[int]
    # Ids of overflights only available from tiles in another UTM zone.
    other_zone: list[str]


def tile_epsg(tile: str) -> int:
    """WGS84 / UTM EPSG code of an MGRS tile: ``31UGV`` -> 32631.

    Latitude bands N-X are northern, C-M southern hemisphere.
    """
    zone, band = int(tile[:2]), tile[2].upper()
    return (32600 if band >= "N" else 32700) + zone


def _cloud_key(cloud_cover: float | None) -> tuple[bool, float]:
    missing = cloud_cover is None or bool(np.isnan(cloud_cover))
    return missing, 0.0 if missing else float(cloud_cover)


def dedupe(
    ids: Sequence[str],
    cloud_covers: Sequence[float | None],
    epsg: int | None = None,
) -> Dedupe:
    """One item per overflight.

    With ``epsg`` (the output grid), tiles in another UTM zone are dropped.
    Per tile the newest processing baseline (latest generation time) wins;
    across tiles of the same overflight the lowest cloud cover (missing
    cloud cover last, ties to the alphabetically first tile). Ids that don't
    parse as a Sentinel-2 L2A product id are always kept.
    """
    if len(ids) != len(cloud_covers):
        raise ValueError(f"got {len(ids)} ids but {len(cloud_covers)} cloud covers")

    # (mission, sensing, orbit, tile) -> (generation time, index)
    newest_per_tile: dict[tuple[str, str, str, str], tuple[str, int]] = {}
    # overflight -> one of its ids, for tiles dropped as another UTM zone
    other_zone: dict[tuple[str, str, str], str] = {}
    kept: list[int] = []
    for i, item_id in enumerate(ids):
        match = _ID_RE.match(str(item_id))
        if match is None:
            kept.append(i)
            continue
        overflight = (match["mission"], match["sensing"], match["orbit"])
        if epsg is not None and tile_epsg(match["tile"]) != epsg:
            other_zone.setdefault(overflight, str(item_id))
            continue
        key = (*overflight, match["tile"])
        current = newest_per_tile.get(key)
        if current is None or match["generation"] > current[0]:
            newest_per_tile[key] = (match["generation"], i)

    best_per_overflight: dict[tuple[str, str, str], tuple[int, str]] = {}
    for (mission, sensing, orbit, tile), (_, i) in newest_per_tile.items():
        overflight = (mission, sensing, orbit)
        best = best_per_overflight.get(overflight)
        if best is None or (_cloud_key(cloud_covers[i]), tile) < (
            _cloud_key(cloud_covers[best[0]]),
            best[1],
        ):
            best_per_overflight[overflight] = (i, tile)

    kept.extend(i for i, _ in best_per_overflight.values())
    return Dedupe(
        kept=sorted(kept),
        # Only overflights lost entirely; one with a same-zone tile is fine.
        other_zone=[
            item_id
            for overflight, item_id in other_zone.items()
            if overflight not in best_per_overflight
        ],
    )
