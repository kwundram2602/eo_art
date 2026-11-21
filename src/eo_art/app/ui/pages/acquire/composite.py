"""Acquire › Composite: band and index composites of an SR GeoTIFF, stretched,
exported as PNG (or, for an SR timeseries, an animated GIF of all dates)."""

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import cast

import flet as ft
import numpy as np

from eo_art.app import composite as cp
from eo_art.app.results import is_timeseries_folder, project_tifs
from eo_art.app.ui.widgets import (
    caption,
    card,
    dropdown,
    file_picker,
    heading,
    notice,
    rel,
    save_bytes_as,
    toast,
    use_app,
)

METHODS = {
    cp.Method.PERCENTILE: "Percentile",
    cp.Method.MINMAX: "Min / max",
    cp.Method.EQUALIZE: "Equalize",
}
MODES = {"preset": "Preset", "custom": "Custom RGB", "single": "Single band"}
CMAPS = [
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
CONTROLS_WIDTH = 400


@lru_cache(maxsize=2)
def _raster(path: str, mtime: float) -> cp.Raster:
    del mtime  # cache key only: a rewritten file gets reloaded
    return cp.load_raster(Path(path))


@lru_cache(maxsize=16)
def _index(path: str, mtime: float, name: str) -> np.ndarray:
    return cp.compute_index(_raster(path, mtime), name)


def _channels(path: str, mtime: float, names: tuple[str, ...]) -> list[np.ndarray]:
    raster = _raster(path, mtime)
    return [_index(path, mtime, n) if cp.is_index(n) else raster.band(n) for n in names]


def _segmented(options: dict[str, str], value: str, on_change) -> ft.Control:
    return ft.SegmentedButton(
        segments=[ft.Segment(value=k, label=ft.Text(t)) for k, t in options.items()],
        selected=[value],
        on_change=lambda e: on_change(e.control.selected[0]),
    )


def _labeled(label: str, control: ft.Control) -> ft.Control:
    return ft.Column([ft.Text(label, size=12), control], spacing=0)


@ft.component
def CompositePage() -> ft.Control:
    state = use_app()
    picked, set_picked = ft.use_state(cast(str | None, None))
    external, set_external = ft.use_state(list[str]())
    indices, set_indices = ft.use_state(list[str]())
    mode, set_mode = ft.use_state("preset")
    preset_label, set_preset_label = ft.use_state(cast(str | None, None))
    rgb, set_rgb = ft.use_state(cast(list[str] | None, None))
    single, set_single = ft.use_state(cast(str | None, None))
    cmap, set_cmap = ft.use_state(cast(str | None, None))
    method, set_method = ft.use_state(cp.Method.PERCENTILE.value)
    shared_choice, set_shared = ft.use_state(cast(bool | None, None))
    pct, set_pct = ft.use_state((2.0, 98.0))
    manual, set_manual = ft.use_state(dict[str, tuple[float, float]]())
    gamma, set_gamma = ft.use_state(1.0)
    if state is None:
        return ft.Container()
    project = state.project
    _ = state.revision

    # Source: project rasters plus files opened from disk.
    tifs = [str(p) for p in project_tifs(project)] if project else []
    sources = list(dict.fromkeys([*tifs, *external]))
    wanted = state.composite_path or picked
    source = wanted if wanted in sources else (sources[0] if sources else None)

    def choose(path: str | None) -> None:
        state.composite_path = None
        set_picked(path)
        set_manual({})

    async def open_file() -> None:
        files = await file_picker().pick_files(
            dialog_title="Open GeoTIFF", allowed_extensions=["tif", "tiff"]
        )
        if not files:
            return
        if files[0].path is None:
            toast("Opening files needs the desktop app.", error=True)
            return
        set_external(lambda prev: [*prev, files[0].path])
        choose(files[0].path)

    def label_of(path: str) -> str:
        return rel(project, Path(path)) if project else path

    source_card = card(
        heading("Source"),
        ft.Row(
            [
                dropdown(
                    "GeoTIFF",
                    [(s, label_of(s)) for s in sources],
                    source,
                    choose,
                    hint_text="No GeoTIFFs in out/ yet" if not sources else None,
                    expand=True,
                ),
                ft.IconButton(
                    ft.Icons.FOLDER_OPEN, tooltip="Open a GeoTIFF…", on_click=open_file
                ),
            ]
        ),
    )
    if source is None:
        return ft.Column(
            [
                ft.Container(source_card, width=CONTROLS_WIDTH),
                caption("Run the SR step, or open a GeoTIFF from disk."),
            ]
        )

    mtime = Path(source).stat().st_mtime
    try:
        raster = _raster(source, mtime)
    except Exception as exc:
        return ft.Column(
            [
                ft.Container(source_card, width=CONTROLS_WIDTH),
                notice(f"Can't read {source}: {exc}", "error"),
            ]
        )
    if not raster.bands:
        return ft.Column(
            [
                ft.Container(source_card, width=CONTROLS_WIDTH),
                notice(
                    "Couldn't identify Sentinel-2 bands in this raster (no band "
                    "descriptions and not the default 10-band layout).",
                    "error",
                ),
            ]
        )

    bands = sorted(raster.bands, key=lambda b: raster.bands[b])
    possible = [n for n in cp.INDICES if not cp.missing_bands(raster, [n])]
    chosen_indices = [n for n in indices if n in possible]
    options = bands + chosen_indices

    # What to display.
    display: list[ft.Control] = [_segmented(MODES, mode, set_mode)]
    colormap: str | None = None
    if mode == "preset":
        available = [p for p in cp.PRESETS if not cp.missing_bands(raster, p.channels)]
        if not available:
            return notice("No preset fits the bands of this raster.", "warn")
        by_label = {f"{p.group} · {p.label}": p for p in available}
        key = preset_label if preset_label in by_label else next(iter(by_label))
        preset = by_label[key]
        display += [
            dropdown(
                "Preset",
                [(k, f"{k} ({p.notation})") for k, p in by_label.items()],
                key,
                set_preset_label,
            ),
            caption(preset.info),
        ]
        unavailable = [p for p in cp.PRESETS if p not in available]
        if unavailable:
            display.append(
                caption(
                    "Not available: "
                    + ", ".join(
                        f"{p.label} (no {', '.join(missing)})"
                        for p in unavailable
                        if (missing := cp.missing_bands(raster, p.channels))
                    )
                )
            )
        names, label = preset.channels, f"{preset.group}-{preset.label}"
    elif mode == "custom":
        defaults = [b for b in ("B04", "B03", "B02") if b in options] or options[:3]
        defaults += options[: 3 - len(defaults)]
        picked_rgb = [
            c if c in options else d for c, d in zip(rgb or defaults, defaults)
        ]

        def set_channel(i: int, value: str | None) -> None:
            if value:
                set_rgb([value if j == i else c for j, c in enumerate(picked_rgb)])

        display.append(
            ft.Row(
                [
                    dropdown(
                        color,
                        [(o, o) for o in options],
                        picked_rgb[i],
                        lambda v, i=i: set_channel(i, v),
                        expand=True,
                    )
                    for i, color in enumerate(("Red", "Green", "Blue"))
                ]
            )
        )
        names = cast(tuple[str, str, str], tuple(picked_rgb))
        label = "-".join(cp.short(c) for c in names)
    else:
        name = (
            single
            if single in options
            else (chosen_indices[0] if chosen_indices else bands[0])
        )
        colormap = (
            cmap if cmap in CMAPS else ("RdYlGn" if cp.is_index(name) else "gray")
        )
        display.append(
            ft.Row(
                [
                    dropdown(
                        "Channel",
                        [(o, o) for o in options],
                        name,
                        set_single,
                        expand=True,
                    ),
                    dropdown(
                        "Colormap",
                        [(c, c) for c in CMAPS],
                        colormap,
                        set_cmap,
                        expand=True,
                    ),
                ]
            )
        )
        names, label = (name,), f"{name}-{colormap}"

    channels = _channels(source, mtime, tuple(names))

    # Stretch.
    current_method = cp.Method(method)
    shared = False
    stretch_controls: list[ft.Control] = [
        _segmented({m.value: t for m, t in METHODS.items()}, method, set_method)
    ]
    if len(channels) > 1:
        # Index channels live on another scale than reflectance, so a shared
        # stretch is only the default for pure band composites.
        default_shared = not any(cp.is_index(n) for n in names)
        shared = default_shared if shared_choice is None else shared_choice
        stretch_controls.append(
            ft.Switch(
                label="Shared stretch",
                value=shared,
                tooltip="One value range for all three channels instead of one each.",
                on_change=lambda e: set_shared(e.control.value),
            )
        )
    manual_ranges: tuple[tuple[float, float], ...] = ()
    if current_method is cp.Method.PERCENTILE:
        stretch_controls.append(
            _labeled(
                f"Percentile range {pct[0]:g} – {pct[1]:g}",
                ft.RangeSlider(
                    start_value=pct[0],
                    end_value=pct[1],
                    min=0,
                    max=100,
                    divisions=200,
                    label="{value}",
                    on_change_end=lambda e: set_pct(
                        (e.control.start_value, e.control.end_value)
                    ),
                ),
            )
        )
    elif current_method is cp.Method.MINMAX:
        groups = [list(channels)] if shared else [[c] for c in channels]
        labels = ["All channels"] if shared else list(names)
        ranges = []
        for group_label, group in zip(labels, groups, strict=True):
            lo = min(cp.value_range(c)[0] for c in group)
            hi = max(cp.value_range(c)[1] for c in group)
            hi = hi if hi > lo else lo + 1e-6
            (start, stop), *_ = cp.percentile_bounds(group, 2.0, 98.0, shared=True)
            key = f"{source}:{group_label}"
            value = manual.get(key, (max(lo, start), min(hi, stop)))
            ranges.append(value)
            stretch_controls.append(
                _labeled(
                    f"Value range · {group_label}: {value[0]:.4f} – {value[1]:.4f}",
                    ft.RangeSlider(
                        start_value=value[0],
                        end_value=value[1],
                        min=lo,
                        max=hi,
                        divisions=500,
                        on_change_end=lambda e, k=key: set_manual(
                            lambda prev: {
                                **prev,
                                k: (e.control.start_value, e.control.end_value),
                            }
                        ),
                    ),
                )
            )
        manual_ranges = tuple(ranges)
    stretch_controls.append(
        _labeled(
            f"Gamma {gamma:.2f} (above 1 brightens dark tones)",
            ft.Slider(
                value=gamma,
                min=0.2,
                max=3.0,
                divisions=56,
                on_change_end=lambda e: set_gamma(e.control.value),
            ),
        )
    )
    stretch = cp.Stretch(current_method, pct[0], pct[1], manual_ranges, shared, gamma)

    image = (
        cp.to_rgb8(channels, stretch)
        if colormap is None
        else cp.to_colormap8(channels[0], stretch, colormap)
    )
    png = cp.encode_png(image)
    limits = cp.bounds(channels, stretch)
    histograms = [
        h
        for i, (n, c) in enumerate(zip(names, channels, strict=True))
        if (h := cp.histogram_png(c, n, limits[i] if limits else None)) is not None
    ]

    stem = Path(source).stem
    in_project = project is not None and source in tifs
    exports: list[ft.Control] = [
        ft.FilledButton(
            "Save PNG…",
            icon=ft.Icons.DOWNLOAD,
            on_click=lambda: save_bytes_as(png, f"{stem}_{cp.slug(label)}.png", "png"),
        )
    ]
    if chosen_indices:
        tif_name = f"{stem}_indices.tif"

        def indices_tif() -> bytes:
            return cp.indices_geotiff(raster, chosen_indices)

        async def save_indices_as() -> None:
            await save_bytes_as(indices_tif(), tif_name, "tif")

        def save_indices_to_out() -> None:
            target = Path(source).parent / tif_name
            target.write_bytes(indices_tif())
            state.refresh()
            toast(f"Saved {target.name}")

        exports.append(
            ft.OutlinedButton(
                "Save indices GeoTIFF…",
                icon=ft.Icons.DOWNLOAD,
                on_click=save_indices_as,
            )
        )
        if in_project:
            exports.append(
                ft.OutlinedButton(
                    "Save indices to out/",
                    icon=ft.Icons.SAVE,
                    on_click=save_indices_to_out,
                )
            )

    _, height, width = raster.data.shape
    preview: list[ft.Control] = [
        ft.Image(src=png, fit=ft.BoxFit.CONTAIN, gapless_playback=True, height=560),
        caption(f"{stem} · {label} · {width}×{height} px"),
        ft.Row([ft.Image(src=h, gapless_playback=True) for h in histograms], wrap=True),
        ft.Row(exports, wrap=True),
    ]
    if is_timeseries_folder(Path(source).parent):
        preview.append(
            TimeseriesGif(
                Path(source).parent,
                tuple(names),
                label,
                stretch,
                colormap,
                (height, width),
                in_project,
            )
        )

    def toggle_index(name: str, on: bool) -> None:
        set_indices(
            lambda prev: [*prev, name] if on else [n for n in prev if n != name]
        )

    controls = ft.Column(
        [
            source_card,
            card(
                heading("Indices to compute"),
                caption("Usable as channels and exportable as a GeoTIFF."),
                ft.Row(
                    [
                        ft.Chip(
                            label=ft.Text(n),
                            selected=n in chosen_indices,
                            tooltip=cp.INDICES[n].info,
                            on_select=lambda e, n=n: toggle_index(
                                n, e.control.selected
                            ),
                        )
                        for n in possible
                    ],
                    wrap=True,
                ),
            ),
            card(heading("Display"), *display),
            card(heading("Stretch"), *stretch_controls),
        ],
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
        width=CONTROLS_WIDTH,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )
    return ft.Row(
        [
            controls,
            ft.Column(
                preview,
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        ],
        spacing=24,
        vertical_alignment=ft.CrossAxisAlignment.START,
        expand=True,
    )


@ft.component
def TimeseriesGif(
    folder: Path,
    names: tuple[str, ...],
    label: str,
    stretch: cp.Stretch,
    colormap: str | None,
    shape: tuple[int, int],
    in_project: bool,
) -> ft.Control:
    state = use_app()
    excluded, set_excluded = ft.use_state(list[str]())
    seconds, set_seconds = ft.use_state(0.8)
    loop, set_loop = ft.use_state(True)
    progress, set_progress = ft.use_state(cast(tuple[float, str] | None, None))
    built, set_built = ft.use_state(cast(tuple[str, cp.GifResult] | None, None))

    frames = cp.timeseries_frames(folder)
    chosen = [f for f in frames if f.name not in excluded]
    # Any change to frames, composite, stretch or timing invalidates a built
    # GIF instead of offering a stale one.
    settings = repr(
        (
            [(f.name, f.stat().st_mtime) for f in chosen],
            str(folder),
            names,
            stretch,
            colormap,
            seconds,
            loop,
        )
    )

    async def build() -> None:
        event_loop = asyncio.get_running_loop()

        def report(fraction: float, text: str) -> None:
            event_loop.call_soon_threadsafe(set_progress, (fraction, text))

        set_progress((0.0, "Loading frames…"))
        try:
            result = await asyncio.to_thread(
                cp.build_gif,
                chosen,
                names,
                stretch,
                colormap,
                shape,
                seconds,
                loop,
                report,
            )
        except Exception as exc:
            toast(f"GIF failed: {exc}", error=True)
            result = None
        set_progress(None)
        if result is not None:
            set_built((settings, result))

    def toggle(name: str, on: bool) -> None:
        set_excluded(
            lambda prev: [n for n in prev if n != name] if on else [*prev, name]
        )

    gif_name = f"{folder.name}_{cp.slug(label)}.gif"
    controls: list[ft.Control] = [
        heading("Timeseries GIF"),
        caption(
            "Every frame gets this composite and stretch; the stretch statistics "
            "are pooled over all selected frames."
        ),
        ft.Row(
            [
                ft.Chip(
                    label=ft.Text(cp.frame_label(f)),
                    selected=f.name not in excluded,
                    on_select=lambda e, n=f.name: toggle(n, e.control.selected),
                )
                for f in frames
            ],
            wrap=True,
        ),
        ft.Row(
            [
                _labeled(
                    f"Seconds per frame: {seconds:.1f}",
                    ft.Slider(
                        value=seconds,
                        min=0.1,
                        max=3.0,
                        divisions=29,
                        width=260,
                        on_change_end=lambda e: set_seconds(round(e.control.value, 1)),
                    ),
                ),
                ft.Switch(
                    label="Loop",
                    value=loop,
                    on_change=lambda e: set_loop(e.control.value),
                ),
                ft.FilledButton(
                    "Build GIF",
                    icon=ft.Icons.MOVIE,
                    on_click=build,
                    disabled=not chosen or progress is not None,
                ),
            ],
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
    ]
    if progress is not None:
        controls += [ft.ProgressBar(value=progress[0]), caption(progress[1])]
    elif built is not None and built[0] != settings:
        controls.append(caption("Settings changed since the last build."))
    elif built is not None:
        result = built[1]
        if result.skipped:
            controls.append(
                notice("Skipped frames: " + "; ".join(result.skipped), "warn")
            )
        if result.data is None:
            controls.append(
                notice("No frame matches this composite and grid.", "error")
            )
        else:
            data = result.data

            def save_to_out() -> None:
                target = folder.parent / gif_name
                target.write_bytes(data)
                if state is not None:
                    state.refresh()
                toast(f"Saved {target.name}")

            controls += [
                ft.Image(src=data, fit=ft.BoxFit.CONTAIN, height=480),
                ft.Row(
                    [
                        ft.FilledButton(
                            "Save GIF…",
                            icon=ft.Icons.DOWNLOAD,
                            on_click=lambda: save_bytes_as(data, gif_name, "gif"),
                        ),
                        *(
                            [
                                ft.OutlinedButton(
                                    "Save GIF to out/",
                                    icon=ft.Icons.SAVE,
                                    on_click=save_to_out,
                                )
                            ]
                            if in_project
                            else []
                        ),
                    ]
                ),
            ]
    return card(*controls)
