from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.data import EOData
    from ..render2d.result import RenderStep

_SUPPORTED = {".gif", ".mp4"}


def animate(
    eo: EOData,
    render_fn: Callable[[EOData], RenderStep],
    path: str | Path,
    *,
    fps: int = 5,
) -> Path:
    """Render each time step of a timeseries EOData and write to a GIF or MP4.

    The output format is inferred from path's suffix (.gif or .mp4).
    Returns the resolved output path.
    Raises ValueError if eo.kind != "timeseries".
    """
    if eo.kind != "timeseries":
        raise ValueError(f"animate() requires kind='timeseries', got kind='{eo.kind}'")

    out = Path(path).resolve()
    suffix = out.suffix.lower()
    if suffix not in _SUPPORTED:
        raise ValueError(
            f"Unsupported output format '{suffix}'. Choose from {sorted(_SUPPORTED)}."
        )

    from ..core.data import EOData

    n_frames = eo.ds["data"].sizes["time"]
    frames: list[np.ndarray] = []
    for t in range(n_frames):
        frame_ds = eo.ds.isel(time=t).drop_vars("time", errors="ignore")
        frame_eo = EOData(
            ds=frame_ds, crs=eo.crs, resolution=eo.resolution, kind="raster"
        )
        step = render_fn(frame_eo)
        frames.append(step.to_uint8())

    import imageio

    if suffix == ".gif":
        imageio.mimwrite(str(out), frames, format="GIF", duration=1000 // fps)
    else:
        with imageio.get_writer(str(out), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

    return out
