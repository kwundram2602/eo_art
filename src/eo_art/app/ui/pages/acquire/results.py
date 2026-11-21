"""Acquire › Results: rasters the acquire/SR runs wrote to ``out/``."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import cast

import flet as ft

from eo_art.app import composite as cp
from eo_art.app.results import ResultGroup, result_groups, zone_mosaic
from eo_art.app.ui.state import AppState
from eo_art.app.ui.widgets import (
    Expandable,
    NeedsProject,
    caption,
    card,
    heading,
    keyed,
    rel,
    title,
    toast,
    use_app,
)
from eo_art.forge3d_pipes.acquire.zones import export_mosaic


def _open_composite(state: AppState, path: Path) -> None:
    state.composite_path = str(path)
    ft.context.page.navigate("/acquire/composite")


def _raster_tile(state: AppState, path: Path, subtitle: str) -> ft.Control:
    return ft.ListTile(
        leading=ft.Icon(ft.Icons.GRID_ON),
        title=ft.Text(path.name),
        subtitle=ft.Text(subtitle),
        trailing=ft.TextButton(
            "Composite",
            icon=ft.Icons.PALETTE,
            on_click=lambda: _open_composite(state, path),
        ),
    )


@ft.component
def ZonesFolder(folder: Path) -> ft.Control:
    """One ``<aoi>_sr_zones/`` folder: tile count, VRT, mosaic export."""
    state = use_app()
    exporting, set_exporting = ft.use_state(cast(bool, False))
    if state is None or state.project is None:
        return ft.Container()
    project = state.project
    tiles = sorted(folder.glob("*.tif"))
    vrt = folder / f"{folder.name}.vrt"
    mosaic = zone_mosaic(folder)

    async def export() -> None:
        set_exporting(True)
        try:
            await asyncio.to_thread(export_mosaic, vrt, mosaic)
        except Exception as exc:
            toast(f"Mosaic export failed: {exc}", error=True)
        else:
            toast(f"Mosaic -> {rel(project, mosaic)}")
            state.refresh()
        finally:
            set_exporting(False)

    status = f"mosaic {rel(project, mosaic)}" if mosaic.exists() else "no mosaic yet"
    return ft.ListTile(
        leading=ft.Icon(ft.Icons.GRID_VIEW),
        title=ft.Text(folder.name),
        subtitle=ft.Text(
            f"{rel(project, folder)} · {len(tiles)} tile(s) · "
            + (f"VRT {vrt.name} · {status}" if vrt.exists() else "no VRT")
        ),
        trailing=ft.ProgressRing(width=20, height=20)
        if exporting
        else ft.TextButton(
            "Re-export mosaic" if mosaic.exists() else "Export mosaic",
            icon=ft.Icons.LAYERS,
            disabled=not vrt.exists(),
            tooltip="Write all tiles as one GeoTIFF (COG); forge3d keeps "
            "using the tiles",
            on_click=export,
        ),
    )


def _group_controls(state: AppState, group: ResultGroup) -> list[ft.Control]:
    project = state.project
    assert project is not None
    controls: list[ft.Control] = []
    for path in group.rasters:
        stat = path.stat()
        controls.append(
            _raster_tile(
                state,
                path,
                f"{stat.st_size / 1e6:.1f} MB · "
                f"{datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M}",
            )
        )
    controls += [keyed(ZonesFolder, str(folder), folder) for folder in group.zone_dirs]
    for folder in group.timeseries_dirs:
        frames = cp.timeseries_frames(folder)
        if not frames:
            continue
        first, last = cp.frame_label(frames[0]), cp.frame_label(frames[-1])
        controls.append(
            _raster_tile(
                state,
                frames[0],
                f"{rel(project, folder)} · {len(frames)} frames · {first} – {last}",
            )
        )
    notes = [rel(project, group.root)]
    if (group.root / "config.yaml").exists():
        notes.append("config.yaml")
    if group.log is not None:
        notes.append(f"log {rel(project, group.log)}")
    controls.append(caption(" · ".join(notes)))
    if group.empty:
        controls.insert(0, caption("No results (failed or still running?)."))
    return controls


def _group_title(group: ResultGroup) -> tuple[str, str]:
    if group.kind is None:
        return "Older results in out/", "written before runs had their own folders"
    stamp = group.root.name[:15]
    try:
        when = f"{datetime.strptime(stamp, '%Y%m%d-%H%M%S'):%Y-%m-%d %H:%M:%S}"
    except ValueError:
        when = group.root.name
    counts = [
        f"{n} {label}"
        for n, label in (
            (len(group.rasters), "raster(s)"),
            (len(group.zone_dirs), "zone folder(s)"),
            (len(group.timeseries_dirs), "timeseries folder(s)"),
        )
        if n
    ]
    latest = f" · latest-{group.kind}" if group.latest else ""
    return f"{group.kind} · {when}{latest}", ", ".join(counts) or "empty"


@ft.component
def ResultsPage() -> ft.Control:
    state = use_app()
    if state is None:
        return ft.Container()
    project = state.project
    if project is None:
        return NeedsProject()
    _ = state.revision

    groups = result_groups(project)
    sections: list[ft.Control] = []
    for index, group in enumerate(groups):
        name, summary = _group_title(group)
        sections.append(
            card(
                keyed(
                    Expandable,
                    f"run:{group.root}",
                    ft.Text(name, theme_style=ft.TextThemeStyle.TITLE_SMALL),
                    _group_controls(state, group),
                    index == 0,
                    leading=ft.Icon(
                        ft.Icons.STAR if group.latest else ft.Icons.FOLDER_OUTLINED
                    ),
                    subtitle=ft.Text(summary, size=12),
                )
            )
        )

    return ft.Column(
        [
            title(
                "Results",
                "One section per acquire/SR run (out/runs/<time>-<kind>/), newest "
                "first; ★ marks the run out/runs/latest-<kind> points at, which "
                "render configs read. Composite opens a raster on the Composite tab.",
            ),
            *(
                sections
                or [card(heading("No runs yet"), caption("Start one under Run."))]
            ),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )
