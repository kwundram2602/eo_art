from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eo_art.core.data import EOData
from eo_art.export.animation import animate
from eo_art.render2d.result import RenderStep


def _timeseries_eo() -> EOData:
    data = np.random.default_rng(0).random((5, 1, 8, 8)).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={"band": [1], "time": range(5)},
    )
    ds = da.to_dataset(name="data")
    return EOData(ds=ds, crs="EPSG:4326", resolution=10.0, kind="timeseries")


def _render_fn(eo: EOData) -> RenderStep:
    arr = eo.ds["data"].values[0].astype(np.float32)
    step = RenderStep(pixels=arr, crs=eo.crs, resolution=eo.resolution)
    return step.normalize()


def test_animate_gif_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "anim.gif"
    result = animate(_timeseries_eo(), _render_fn, out, fps=3)
    assert result.exists()
    assert result.stat().st_size > 0


def test_animate_returns_resolved_path(tmp_path: Path) -> None:
    out = tmp_path / "anim.gif"
    result = animate(_timeseries_eo(), _render_fn, out)
    assert result == out.resolve()
    assert isinstance(result, Path)


def test_animate_raises_for_non_timeseries() -> None:
    data = np.random.default_rng(1).random((1, 8, 8)).astype(np.float32)
    da = xr.DataArray(data, dims=("band", "y", "x"), coords={"band": [1]})
    ds = da.to_dataset(name="data")
    eo = EOData(ds=ds, crs="EPSG:4326", resolution=10.0, kind="raster")
    with pytest.raises(ValueError, match="timeseries"):
        animate(eo, _render_fn, "out.gif")


def test_animate_raises_for_unsupported_suffix(tmp_path: Path) -> None:
    out = tmp_path / "anim.xyz"
    with pytest.raises(ValueError, match="Unsupported"):
        animate(_timeseries_eo(), _render_fn, out)
