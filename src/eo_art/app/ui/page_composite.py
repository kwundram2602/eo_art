"""Load an SR GeoTIFF, build band/index composites, stretch them, export a PNG
(or, for an SR timeseries, an animated GIF of all dates)."""

import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from eo_art.app import composite as cp
from eo_art.app.projects import Project
from eo_art.app.ui.common import current_project, rel
from eo_art.forge3d_pipes.acquire.pipeline import TIMESERIES_SUFFIX

_SOURCES = {
    "project": ":material/folder: Project out/",
    "upload": ":material/upload: Upload",
}
_MODES = {
    "preset": "RGB preset",
    "custom": "RGB custom",
    "single": "Single band",
}
_METHODS = {
    cp.Method.PERCENTILE: "Percentile",
    cp.Method.MINMAX: "Min / max",
    cp.Method.EQUALIZE: "Equalize",
}
_CMAPS = [
    "gray",
    "viridis",
    "magma",
    "inferno",
    "cividis",
    "RdYlGn",
    "RdYlBu",
    "Spectral",
    "BrBG",
    "terrain",
    "twilight",
]


@st.cache_data(max_entries=2, show_spinner="Loading raster…")
def _load_path(path: str, mtime: float) -> cp.Raster:
    del mtime  # cache key only: a rewritten file gets reloaded
    return cp.load_raster(Path(path))


@st.cache_data(max_entries=2, show_spinner="Loading raster…")
def _load_upload(file_id: str, _data: bytes) -> cp.Raster:
    return cp.load_raster(_data)


@st.cache_data(max_entries=16, show_spinner="Computing index…")
def _index(source_key: str, name: str, _raster: cp.Raster) -> np.ndarray:
    return cp.compute_index(_raster, name)


@st.cache_data(max_entries=4, show_spinner=False)
def _indices_tif(source_key: str, names: tuple[str, ...], _raster: cp.Raster) -> bytes:
    return cp.indices_geotiff(_raster, names)


def _project_tifs(project: Project) -> list[Path]:
    out = project.out_dir
    if not out.exists():
        return []

    # SR results first (newest first), then timeseries steps in date order,
    # then every other raster (newest first).
    def order(p: Path) -> tuple[int, float, str]:
        if p.parent != out:
            return 1, 0.0, str(p)
        return (0 if p.name.endswith("_sr.tif") else 2), -p.stat().st_mtime, ""

    return sorted(
        [*out.glob("*.tif"), *out.glob(f"*{TIMESERIES_SUFFIX}/*.tif")], key=order
    )


def _source(project: Project | None) -> tuple[cp.Raster, str, str, Path | None] | None:
    """Loaded raster, its cache key, a file stem and (project files) its path."""
    choice = st.segmented_control(
        "Source",
        list(_SOURCES),
        default="project" if project else "upload",
        format_func=_SOURCES.__getitem__,
        required=True,
        key="composite_source",
    )
    if choice == "project":
        if project is None:
            st.info("Select a project in the sidebar, or upload a file.")
            return None
        tifs = _project_tifs(project)
        if not tifs:
            st.info("No GeoTIFFs in out/ yet — run the SR step or upload a file.")
            return None
        path = st.selectbox(
            "GeoTIFF", tifs, format_func=lambda p: rel(project, p), key="composite_tif"
        )
        mtime = path.stat().st_mtime
        return _load_path(str(path), mtime), f"{path}:{mtime}", path.stem, path

    upload = st.file_uploader("GeoTIFF", type=["tif", "tiff"], key="composite_upload")
    if upload is None:
        return None
    raster = _load_upload(upload.file_id, upload.getvalue())
    return raster, upload.file_id, Path(upload.name).stem, None


def _preset_channels(raster: cp.Raster) -> tuple[tuple[str, ...], str]:
    available = [p for p in cp.PRESETS if not cp.missing_bands(raster, p.channels)]
    unavailable = [p for p in cp.PRESETS if p not in available]
    if not available:
        st.warning("No preset fits the bands of this raster.")
        st.stop()
    preset = st.selectbox(
        "Preset",
        available,
        format_func=lambda p: f"{p.group} · {p.label} ({p.notation})",
        key="composite_preset",
    )
    st.caption(preset.info)
    if unavailable:
        missing = ", ".join(
            f"{p.label} ({p.notation}: no "
            f"{', '.join(cp.missing_bands(raster, p.channels))})"
            for p in unavailable
        )
        st.caption(f":material/block: Not available: {missing}")
    return preset.channels, f"{preset.group}-{preset.label}"


def _custom_channels(options: list[str]) -> tuple[tuple[str, ...], str]:
    defaults = [b for b in ("B04", "B03", "B02") if b in options] or options[:3]
    defaults += options[: 3 - len(defaults)]
    picked = []
    for col, label, default in zip(
        st.columns(3), ("Red", "Green", "Blue"), defaults, strict=False
    ):
        with col:
            picked.append(
                st.selectbox(
                    label,
                    options,
                    index=options.index(default),
                    key=f"composite_{label.lower()}",
                )
            )
    return tuple(picked), "-".join(cp.short(c) for c in picked)


def _stretch(
    names: tuple[str, ...], channels: list[np.ndarray], source_key: str
) -> cp.Stretch:
    method = st.segmented_control(
        "Stretch",
        list(_METHODS),
        default=cp.Method.PERCENTILE,
        format_func=_METHODS.__getitem__,
        required=True,
        key="composite_method",
    )
    shared = False
    if len(channels) > 1:
        # Index channels live on a different scale than reflectance, so a
        # shared stretch only makes sense by default for pure band composites
        # (where it keeps the natural color balance, see RgbStretchCfg).
        default_shared = not any(cp.is_index(n) for n in names)
        shared = st.toggle(
            "Shared stretch",
            value=default_shared,
            help="Use one value range for all three channels instead of one each.",
            key=f"composite_shared_{'-'.join(names)}",
        )

    lower, upper, manual = 2.0, 98.0, ()
    if method == cp.Method.PERCENTILE:
        lower, upper = st.slider(
            "Percentile range", 0.0, 100.0, (2.0, 98.0), step=0.5, key="composite_pct"
        )
    elif method == cp.Method.MINMAX:
        groups = [list(channels)] if shared else [[c] for c in channels]
        labels = ["All channels"] if shared else list(names)
        manual = tuple(
            _range_slider(label, group, f"{source_key}:{label}")
            for label, group in zip(labels, groups, strict=True)
        )

    gamma = st.slider(
        "Gamma",
        0.2,
        3.0,
        1.0,
        step=0.05,
        help="Above 1 brightens dark tones, below 1 darkens them.",
        key="composite_gamma",
    )
    return cp.Stretch(method, lower, upper, manual, shared, gamma)


def _range_slider(
    label: str, channels: list[np.ndarray], key: str
) -> tuple[float, float]:
    lo, hi = (
        min(cp.value_range(c)[0] for c in channels),
        max(cp.value_range(c)[1] for c in channels),
    )
    if hi <= lo:
        hi = lo + 1e-6
    (start, stop), *_ = cp.percentile_bounds(channels, 2.0, 98.0, shared=True)
    return st.slider(
        f"Value range · {label}",
        lo,
        hi,
        (max(lo, start), min(hi, stop)),
        step=(hi - lo) / 500,
        format="%.4f",
        key=f"composite_range_{key}",
    )


def _histograms(
    names: tuple[str, ...],
    channels: list[np.ndarray],
    stretch: cp.Stretch,
) -> None:
    bounds = cp.bounds(channels, stretch)
    cols = st.columns(len(channels))
    for i, (col, name, values) in enumerate(zip(cols, names, channels, strict=True)):
        valid = cp.sample(values)
        if valid.size == 0:
            continue
        counts, edges = np.histogram(valid, bins=96)
        frame = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "pixels": counts})
        x = alt.X("start:Q", title=name, axis=alt.Axis(format=".3~f", tickCount=4))
        chart = (
            alt.Chart(frame)
            .mark_bar(color="#7a8597")
            .encode(
                x=x,
                x2="end:Q",
                y=alt.Y(
                    "pixels:Q", title=None, axis=alt.Axis(labels=False, ticks=False)
                ),
                tooltip=[
                    alt.Tooltip("start:Q", title="from", format=".4f"),
                    alt.Tooltip("end:Q", title="to", format=".4f"),
                    alt.Tooltip("pixels:Q", format=","),
                ],
            )
        )
        if bounds is not None:
            lo, hi = bounds[i]
            rules = (
                alt.Chart(pd.DataFrame({"value": [lo, hi]}))
                .mark_rule(strokeDash=[4, 3], strokeWidth=2)
                .encode(x="value:Q", tooltip=alt.Tooltip("value:Q", format=".4f"))
            )
            chart = chart + rules
        with col:
            st.altair_chart(
                chart.properties(height=120),
                alt=f"Histogram of {name}",
            )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _frame_label(path: Path) -> str:
    return cp.frame_date(path) or path.stem


def _build_gif(
    frames: list[Path],
    names: tuple[str, ...],
    stretch: cp.Stretch,
    cmap: str | None,
    shape: tuple[int, int],
    seconds: float,
    loop: bool,
) -> tuple[bytes | None, list[str]]:
    """GIF bytes (None if no frame fits) and the skipped frames with reasons."""
    progress = st.progress(0.0, text="Loading frames…")
    channels: list[list[np.ndarray]] = []
    skipped: list[str] = []
    for i, frame in enumerate(frames, start=1):
        raster = cp.load_raster(frame)
        missing = cp.missing_bands(raster, names)
        if missing:
            skipped.append(f"{_frame_label(frame)} (no {', '.join(missing)})")
        elif raster.data.shape[1:] != shape:
            skipped.append(f"{_frame_label(frame)} (grid {raster.data.shape[1:]})")
        else:
            # Copies, so the frame's full band stack can be freed right away.
            channels.append([cp.channel(raster, n).copy() for n in names])
        progress.progress(i / len(frames) * 0.8, text=f"Loaded {_frame_label(frame)}")
    if not channels:
        progress.empty()
        return None, skipped

    # One set of stretch statistics for the whole series: no flicker, and a
    # darker date stays darker instead of being auto-levelled on its own.
    reference = cp.pooled_reference(channels)
    progress.progress(0.85, text="Rendering frames…")
    images = [
        cp.to_rgb8(ch, stretch, reference)
        if cmap is None
        else cp.to_colormap8(ch[0], stretch, cmap, reference)
        for ch in channels
    ]
    progress.progress(0.95, text="Encoding GIF…")
    data = cp.encode_gif(images, seconds, loop)
    progress.empty()
    return data, skipped


def _timeseries_gif(
    path: Path,
    names: tuple[str, ...],
    label: str,
    stretch: cp.Stretch,
    cmap: str | None,
    shape: tuple[int, int],
) -> None:
    folder = path.parent
    # Only dated SR steps; other rasters saved here (e.g. indices) aren't frames.
    frames = sorted(
        (p for p in folder.glob("*.tif") if cp.frame_date(p)),
        key=lambda p: (_frame_label(p), p.name),
    )
    with st.container(border=True):
        st.markdown("**:material/animation: Timeseries GIF**")
        st.caption(
            "Every frame gets this composite and stretch; the stretch statistics "
            "are pooled over all selected frames."
        )
        chosen = st.multiselect(
            "Frames",
            frames,
            default=frames,
            format_func=_frame_label,
            key=f"composite_gif_frames_{folder}",
        )
        with st.container(horizontal=True):
            seconds = st.slider(
                "Seconds per frame",
                0.1,
                3.0,
                0.8,
                step=0.1,
                key="composite_gif_seconds",
            )
            loop = st.toggle("Loop", value=True, key="composite_gif_loop")

        # Any change to frames, composite, stretch or timing invalidates a
        # built GIF instead of offering a stale one for download.
        settings = repr(
            (
                [(f.name, f.stat().st_mtime) for f in chosen],
                str(folder),
                names,
                stretch,
                cmap,
                seconds,
                loop,
            )
        )
        if st.button(
            "Build GIF",
            icon=":material/movie:",
            disabled=not chosen,
            key="composite_gif_build",
        ):
            st.session_state["composite_gif"] = (
                settings,
                _build_gif(chosen, names, stretch, cmap, shape, seconds, loop),
            )
        built = st.session_state.get("composite_gif")
        if built is None or built[0] != settings:
            if built is not None:
                st.caption("Settings changed since the last build.")
            return

        data, skipped = built[1]
        if skipped:
            st.warning("Skipped frames: " + "; ".join(skipped))
        if data is None:
            st.error("No frame matches this composite and grid.")
            return
        st.image(data, width="stretch", alt=f"{label} timeseries of {folder.name}")
        gif_name = f"{folder.name}_{_slug(label)}.gif"
        with st.container(horizontal=True):
            st.download_button(
                "Download GIF",
                data=data,
                file_name=gif_name,
                mime="image/gif",
                on_click="ignore",
                type="primary",
                icon=":material/download:",
            )
            if st.button("Save GIF to out/", icon=":material/save:"):
                target = folder.parent / gif_name
                target.write_bytes(data)
                st.toast(f"Saved {target.name}", icon=":material/check:")


def render() -> None:
    st.title("Composite")
    st.caption(
        "Build band and index composites from a super-resolved Sentinel-2 GeoTIFF."
    )
    controls, preview = st.columns([1, 2], gap="large")

    with controls:
        with st.container(border=True):
            loaded = _source(current_project())
        if loaded is None:
            return
        raster, source_key, stem, path = loaded
        if not raster.bands:
            st.error(
                "Couldn't identify Sentinel-2 bands in this raster "
                "(no band descriptions and not the default 10-band layout)."
            )
            return

        bands = sorted(raster.bands, key=lambda b: raster.bands[b])
        possible = [n for n in cp.INDICES if not cp.missing_bands(raster, [n])]
        with st.container(border=True):
            indices = st.multiselect(
                "Indices to compute",
                possible,
                format_func=lambda n: f"{n} — {cp.INDICES[n].info}",
                help="Computed from the raster; usable as channels and exportable "
                "as a GeoTIFF.",
                key="composite_indices",
            )
            mode = st.segmented_control(
                "Display",
                list(_MODES),
                default="preset",
                format_func=_MODES.__getitem__,
                required=True,
                key="composite_mode",
            )
            options = bands + indices
            cmap = None
            if mode == "preset":
                names, label = _preset_channels(raster)
            elif mode == "custom":
                names, label = _custom_channels(options)
            else:
                default = indices[0] if indices else bands[0]
                name = st.selectbox(
                    "Channel",
                    options,
                    index=options.index(default),
                    key="composite_single",
                )
                cmap = st.selectbox(
                    "Colormap",
                    _CMAPS,
                    index=_CMAPS.index("RdYlGn" if cp.is_index(name) else "gray"),
                    key=f"composite_cmap_{cp.is_index(name)}",
                )
                names, label = (name,), f"{name}-{cmap}"

        channels = [
            _index(source_key, n, raster) if cp.is_index(n) else raster.band(n)
            for n in names
        ]
        with st.container(border=True):
            stretch = _stretch(names, channels, source_key)

    with preview:
        image = (
            cp.to_rgb8(channels, stretch)
            if cmap is None
            else cp.to_colormap8(channels[0], stretch, cmap)
        )
        st.image(
            image,
            width="stretch",
            alt=f"{label} composite of {stem}",
        )
        _, height, width = raster.data.shape
        st.caption(f"{stem} · {label} · {width}×{height} px")
        _histograms(names, channels, stretch)

        with st.container(horizontal=True):
            st.download_button(
                "Download PNG",
                data=lambda: cp.encode_png(image),
                file_name=f"{stem}_{_slug(label)}.png",
                mime="image/png",
                on_click="ignore",
                type="primary",
                icon=":material/download:",
            )
            if indices:
                tif_name = f"{stem}_indices.tif"
                tif = _indices_tif(source_key, tuple(indices), raster)
                st.download_button(
                    "Download indices GeoTIFF",
                    data=tif,
                    file_name=tif_name,
                    mime="image/tiff",
                    on_click="ignore",
                    icon=":material/download:",
                )
                if path is not None and st.button(
                    "Save indices to out/", icon=":material/save:"
                ):
                    target = path.parent / tif_name
                    target.write_bytes(tif)
                    st.toast(f"Saved {target.name}", icon=":material/check:")

        if path is not None and path.parent.name.endswith(TIMESERIES_SUFFIX):
            _timeseries_gif(path, names, label, stretch, cmap, (height, width))
