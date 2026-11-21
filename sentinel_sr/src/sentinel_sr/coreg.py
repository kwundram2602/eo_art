"""Co-register LR timesteps (e.g. one zone mosaic per step) with AROSICS.

Thin wrapper over ``stac2cube.coregistration.coregister_cube``, which does
the actual work: per scene up to ``grid_size``² + 1 AROSICS windows placed
where both images are cloud-free, a robust consensus over them (reliability-
weighted median, MAD inliers), chaining from an anchor scene in both
directions in time, and one single cubic warp per scene at the end.

Run this at the native 10 m before super-resolution: the matcher needs the
real high-frequency content of the native 10 m bands, not texture an SR model
invented, and SR then sees already aligned inputs.

All inputs must share one grid (same CRS, transform and size), as eo_art's
step mosaics do. Scenes the matcher can't place reliably are dropped, as in
stac2cube; the report says which and why (in the printed summary).
"""

import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401 (registers the .rio accessor on xarray objects)
import xarray as xr
from stac2cube import coregistration as s2c_coreg

from sentinel_sr.clouds import DEFAULT_CLOUD_CLASSES

# Native 10 m Sentinel-2 bands by id, for rasters whose band descriptions are
# ids rather than common names: stac2cube's auto band choice looks for these
# names (20 m bands match far worse).
_COMMON_NAMES = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
_MATCH_ORDER = ("nir", "red", "green", "blue")


@dataclass(frozen=True)
class CoregJob:
    src: str
    dst: str
    # Acquisition date (YYYY-MM-DD); orders the series.
    date: str
    # SCL GeoTIFF on the same grid (None: no cloud mask for this scene).
    scl: str | None = None


@dataclass(frozen=True)
class CoregResult:
    src: str
    dst: str | None
    date: str
    # Total shift applied, in pixels (None: dropped).
    shift_x_px: float | None
    shift_y_px: float | None
    cloud_percent: float | None


def _band_names(descriptions: Sequence[str | None], count: int) -> list[str]:
    names = []
    for i in range(count):
        d = descriptions[i] if i < len(descriptions) else None
        d = d or f"band{i + 1}"
        names.append(_COMMON_NAMES.get(d, d))
    return names


def match_band(names: Sequence[str]) -> str:
    """The native 10 m band AROSICS should match on."""
    lowered = [n.lower() for n in names]
    for cand in _MATCH_ORDER:
        if cand in lowered:
            return names[lowered.index(cand)]
    raise ValueError(
        f"no native 10 m band ({', '.join(_MATCH_ORDER)}) among {list(names)}; "
        "co-registration needs one of B02/B03/B04/B08"
    )


def _unique_times(dates: Sequence[str]) -> np.ndarray:
    """datetime64 per job; repeated dates get +1 s each so every scene keeps
    its own time key (stac2cube selects scenes by time)."""
    seen: dict[str, int] = {}
    out = []
    for d in dates:
        k = seen.get(d, 0)
        seen[d] = k + 1
        out.append(pd.Timestamp(d) + pd.Timedelta(seconds=k))
    return np.array(out, dtype="datetime64[ns]")


def build_stack(
    jobs: Sequence[CoregJob], scl_classes: Sequence[int] = DEFAULT_CLOUD_CLASSES
) -> tuple[xr.Dataset, xr.DataArray | None]:
    """(Dataset with ``Time_Series(time, band, y, x)`` and ``cloud_percentage``,
    cloud mask ``(time, y, x)`` with 1 = cloud, or None without any SCL)."""
    arrays, masks, grid = [], [], None
    for job in jobs:
        with rasterio.open(job.src) as ds:
            this = (ds.crs, ds.transform, ds.width, ds.height)
            if grid is None:
                grid = this
                names = _band_names(ds.descriptions, ds.count)
            elif this != grid:
                raise ValueError(
                    f"{job.src} is not on the grid of {jobs[0].src}; all "
                    "co-registration inputs must share CRS, transform and size"
                )
            arr = ds.read().astype("float32")
            if ds.nodata is not None and not np.isnan(ds.nodata):
                arr[arr == ds.nodata] = np.nan
        arrays.append(arr)
        if job.scl is not None:
            with rasterio.open(job.scl) as ds:
                masks.append(np.isin(ds.read(1), list(scl_classes)))
        else:
            masks.append(np.zeros(arr.shape[1:], dtype=bool))

    assert grid is not None
    crs, transform, width, height = grid
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    times = _unique_times([j.date for j in jobs])

    data = np.stack(arrays)
    valid = np.isfinite(data).all(axis=1)
    mask = np.stack(masks) & valid
    n_valid = valid.sum(axis=(1, 2))
    cloud = np.where(
        n_valid > 0, mask.sum(axis=(1, 2)) / np.maximum(n_valid, 1) * 100, 100.0
    )

    stack = xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={"time": times, "band": names, "y": ys, "x": xs},
        name="Time_Series",
    )
    stack = stack.assign_coords(cloud_percentage=("time", cloud.astype("float64")))
    stack = stack.rio.write_crs(crs).rio.write_transform(transform)
    mask_da = None
    if any(j.scl is not None for j in jobs):
        mask_da = xr.DataArray(
            mask.astype("uint8"),
            dims=("time", "y", "x"),
            coords={"time": times, "y": ys, "x": xs},
        )
    return xr.Dataset({"Time_Series": stack}), mask_da


@contextmanager
def _record_warps():
    """Record the shift stac2cube applies per scene.

    ``coregister_cube`` returns the warped cube but not the shifts. Every warp
    goes through ``_warp_scene_yxb(scene, shift_x, shift_y, ...)``; the final
    single warp of each kept scene is its last call, so the last recorded
    shift per time is the applied one.
    """
    original = s2c_coreg._warp_scene_yxb
    shifts: dict[np.datetime64, tuple[float, float]] = {}

    def recording(da_yxb, shift_x_px, shift_y_px, *args, **kwargs):
        if "time" in da_yxb.coords:
            shifts[np.datetime64(da_yxb.time.values, "ns")] = (
                float(shift_x_px),
                float(shift_y_px),
            )
        return original(da_yxb, shift_x_px, shift_y_px, *args, **kwargs)

    s2c_coreg._warp_scene_yxb = recording
    try:
        yield shifts
    finally:
        s2c_coreg._warp_scene_yxb = original


def coregister_jobs(
    jobs: Sequence[CoregJob],
    anchor: str = "auto",
    grid_size: int = 7,
    scl_classes: Sequence[int] = DEFAULT_CLOUD_CLASSES,
) -> list[CoregResult]:
    """Co-register all ``jobs`` as one series and write each kept scene to
    its ``dst`` (NaN nodata, band descriptions kept from ``src``).

    ``anchor``: ``"auto"`` (most textured low-cloud scene in the middle half
    of the series), ``"first"``, ``"composite"`` or a date ``YYYY-MM-DD``.
    """
    if len(jobs) < 2:
        raise ValueError("co-registration needs at least two scenes")
    order = sorted(range(len(jobs)), key=lambda i: jobs[i].date)
    jobs = [jobs[i] for i in order]
    ds, mask_da = build_stack(jobs, scl_classes)
    names = list(ds["Time_Series"].band.values)
    with rasterio.open(jobs[0].src) as src:
        descriptions = src.descriptions

    with _record_warps() as shifts:
        out_ds, _ = s2c_coreg.coregister_cube(
            ds,
            output_path=None,
            first_scene_mode=anchor,
            grid_size=grid_size,
            match_band=match_band(names),
            # "auto" judges passes by edge roughness, a second temporal
            # difference that is NaN for fewer than 3 scenes, so it would
            # never stop early; one pass is what it keeps on clear data anyway.
            iteration="auto" if len(jobs) >= 3 else 1,
            cloud_mask=mask_da,
        )

    out = out_ds["Time_Series"]
    kept = {np.datetime64(t, "ns") for t in out.time.values}
    cloud = ds["Time_Series"].cloud_percentage.values
    results = []
    for job, t, cp in zip(jobs, ds["Time_Series"].time.values, cloud, strict=True):
        t = np.datetime64(t, "ns")
        if t not in kept:
            results.append(CoregResult(job.src, None, job.date, None, None, float(cp)))
            continue
        scene = out.sel(time=t).transpose("band", "y", "x")
        scene = scene.drop_vars(
            [c for c in scene.coords if c not in ("band", "y", "x")]
        )
        scene = scene.rio.write_nodata(float("nan"))
        scene.attrs = {
            "long_name": [d or n for d, n in zip(descriptions, names, strict=True)]
        }
        Path(job.dst).parent.mkdir(parents=True, exist_ok=True)
        scene.astype("float32").rio.to_raster(job.dst, compress="deflate")
        sx, sy = shifts.get(t, (0.0, 0.0))
        results.append(CoregResult(job.src, job.dst, job.date, sx, sy, float(cp)))
    return results


def write_report(results: Sequence[CoregResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scenes": [asdict(r) for r in results]}, indent=2))
    return path


def read_jobs(path: str | Path) -> list[CoregJob]:
    """Jobs file: ``{"jobs": [{"src", "dst", "date", "scl"?}, ...]}``."""
    return [CoregJob(**job) for job in json.loads(Path(path).read_text())["jobs"]]
