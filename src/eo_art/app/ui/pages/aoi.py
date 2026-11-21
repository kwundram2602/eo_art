"""AOI: draw (rectangle / polygon by clicking), import and manage the
project's areas of interest on a map, and point acquire configs at them."""

from pathlib import Path
from typing import Any, cast

import flet as ft
import flet_map as fm

from eo_art.app import aoi
from eo_art.app.configs import (
    ConfigKind,
    SrSettings,
    configs_of_kind,
    set_top_level_value,
    sr_settings,
)
from eo_art.app.projects import Project, validate_name
from eo_art.app.ui.widgets import (
    NeedsProject,
    caption,
    card,
    dropdown,
    file_picker,
    heading,
    rel,
    toast,
    use_app,
)

DEFAULT_CENTER = (47.5, 11.0)
TILES = {
    "osm": (
        "OpenStreetMap",
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "© OpenStreetMap contributors",
    ),
    "satellite": (
        "Satellite",
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Esri World Imagery",
    ),
}
DRAW_MODES = {"off": "Pan", "rectangle": "Rectangle", "polygon": "Polygon"}
DRAW_HINTS = {
    "off": "Pick Rectangle or Polygon to draw on the map.",
    "rectangle": "Click two opposite corners.",
    "polygon": "Click the corners, then Finish polygon.",
}
SIDE_WIDTH = 360
AOI_COLOR = ft.Colors.BLUE
CHIP_COLOR = "#d9480f"
DRAWN_COLOR = ft.Colors.GREEN

LatLon = tuple[float, float]


def _ll(point: LatLon) -> fm.MapLatitudeLongitude:
    return fm.MapLatitudeLongitude(point[0], point[1])


def _ring(feature: dict[str, Any]) -> list[LatLon]:
    return [(lat, lon) for lon, lat in feature["geometry"]["coordinates"][0]]


def _square(south: float, west: float, north: float, east: float) -> list[LatLon]:
    return [(south, west), (south, east), (north, east), (north, west), (south, west)]


def _read_aois(project: Project | None, sr: SrSettings | None) -> list[dict[str, Any]]:
    """Per AOI file: outlines, SR chip square(s), or the read error.

    With ``sentinel.zones`` the chips are the zone grid; a grid that can't be
    planned (e.g. across a UTM boundary) keeps the outline and reports the
    reason in ``zones_error``.
    """
    if project is None or sr is None:
        return []
    items = []
    for path in project.aoi_files():
        try:
            lat, lon = aoi.center(path)
            item: dict[str, Any] = {
                "path": path,
                "rings": aoi.outlines(path),
                "center": (lat, lon),
                "chips": [_square(*aoi.chip_bounds(lat, lon, sr.edge_m))],
                "zones": None,
                "zones_error": None,
                "error": None,
            }
        except Exception as exc:
            items.append(
                {
                    "path": path,
                    "rings": [],
                    "center": None,
                    "chips": [],
                    "zones": None,
                    "zones_error": None,
                    "error": exc,
                }
            )
            continue
        if sr.zones:
            try:
                grid = aoi.zone_grid(path, sr.edge_size, sr.resolution, sr.clip_to_aoi)
                minx, miny, maxx, maxy = grid.bounds
                item["chips"] = aoi.zone_rings(grid)
                item["zones"] = (len(grid.zones), maxx - minx, maxy - miny)
            except Exception as exc:
                item["chips"] = []
                item["zones_error"] = exc
        items.append(item)
    return items


def _camera(items: list[dict[str, Any]]) -> fm.CameraFit | None:
    points = [p for item in items for ring in item["rings"] for p in ring]
    if not points:
        return None
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    return fm.CameraFit(
        bounds=fm.MapLatitudeLongitudeBounds(
            fm.MapLatitudeLongitude(min(lats), min(lons)),
            fm.MapLatitudeLongitude(max(lats), max(lons)),
        ),
        padding=ft.Padding.all(60),
        max_zoom=14,
    )


@ft.component
def AoiPage() -> ft.Control:
    state = use_app()
    tiles, set_tiles = ft.use_state(cast(str, "osm"))
    mode, set_mode = ft.use_state(cast(str, "off"))
    points, set_points = ft.use_state(list[LatLon]())
    drawn, set_drawn = ft.use_state(list[dict[str, Any]]())
    name, set_name = ft.use_state("")
    overwrite, set_overwrite = ft.use_state(False)
    import_name, set_import_name = ft.use_state("")
    target_cfg, set_target_cfg = ft.use_state(cast(str | None, None))
    project = state.project if state else None
    revision = state.revision if state else 0
    sr = sr_settings(project) if project else None
    edge_m = sr.edge_m if sr else 0.0
    files = (
        tuple((p.name, p.stat().st_mtime) for p in project.aoi_files())
        if project
        else ()
    )
    items = ft.use_memo(lambda: _read_aois(project, sr), [revision, files, sr])
    if state is None:
        return ft.Container()
    if project is None:
        return NeedsProject()

    def on_tap(e: fm.MapTapEvent) -> None:
        if mode == "off" or e.coordinates is None:
            return
        point = (e.coordinates.latitude, e.coordinates.longitude)
        if mode == "rectangle" and points:
            try:
                feature = aoi.rectangle_feature(points[0], point)
            except ValueError as exc:
                toast(str(exc), error=True)
                return
            set_drawn(lambda prev: [*prev, feature])
            set_points([])
        else:
            set_points(lambda prev: [*prev, point])

    def finish_polygon() -> None:
        try:
            feature = aoi.polygon_feature(points)
        except ValueError as exc:
            toast(str(exc), error=True)
            return
        set_drawn(lambda prev: [*prev, feature])
        set_points([])

    def change_mode(value: str) -> None:
        set_mode(value)
        set_points([])

    def save_drawn() -> None:
        try:
            target = project.aoi_dir / f"{validate_name(name)}.gpkg"
            if target.exists() and not overwrite:
                raise FileExistsError(f"{target.name} exists (tick Overwrite)")
            aoi.save_features(drawn, target)
        except (ValueError, FileExistsError) as exc:
            toast(str(exc), error=True)
            return
        set_drawn([])
        set_name("")
        state.refresh()
        toast(f"Saved {rel(project, target)}")

    async def import_file() -> None:
        picked = await file_picker().pick_files(
            dialog_title="Import AOI",
            allowed_extensions=[s.lstrip(".") for s in aoi.UPLOAD_SUFFIXES],
        )
        if not picked:
            return
        source = picked[0].path
        if source is None:
            toast("Importing files needs the desktop app.", error=True)
            return
        try:
            target = (
                project.aoi_dir
                / f"{validate_name(import_name or Path(source).stem)}.gpkg"
            )
            aoi.import_upload(Path(source).read_bytes(), Path(source).name, target)
        except Exception as exc:
            toast(f"Import failed: {exc}", error=True)
            return
        set_import_name("")
        state.refresh()
        toast(f"Imported {rel(project, target)}")

    def delete(path: Path) -> None:
        path.unlink(missing_ok=True)
        state.refresh()
        toast(f"Deleted {path.name}")

    acquire_cfgs = [p.name for p in configs_of_kind(project, ConfigKind.acquire)]
    cfg = (
        target_cfg
        if target_cfg in acquire_cfgs
        else (acquire_cfgs[0] if acquire_cfgs else None)
    )

    def use_as_aoi_path(path: Path) -> None:
        if cfg is None:
            return
        set_top_level_value(project.configs_dir / cfg, "aoi_path", rel(project, path))
        state.refresh()
        toast(f"{cfg}: aoi_path = {rel(project, path)}")

    # Map layers: saved AOIs (blue), SR chips (orange), drawn shapes (green),
    # and the shape being drawn.
    polygons = [
        fm.PolygonMarker(
            coordinates=[_ll(p) for p in ring],
            color=ft.Colors.with_opacity(0.15, AOI_COLOR),
            border_color=AOI_COLOR,
            border_stroke_width=2,
            label=item["path"].stem,
        )
        for item in items
        for ring in item["rings"]
    ] + [
        fm.PolygonMarker(
            coordinates=[_ll(p) for p in _ring(feature)],
            color=ft.Colors.with_opacity(0.25, DRAWN_COLOR),
            border_color=DRAWN_COLOR,
            border_stroke_width=2,
        )
        for feature in drawn
    ]
    lines = [
        fm.PolylineMarker(
            coordinates=[_ll(p) for p in chip],
            color=CHIP_COLOR,
            stroke_width=2 if len(item["chips"]) == 1 else 1,
            stroke_pattern=fm.DashedStrokePattern(segments=[8, 6]),
        )
        for item in items
        for chip in item["chips"]
    ]
    if len(points) > 1:
        lines.append(
            fm.PolylineMarker(
                coordinates=[_ll(p) for p in points], color=DRAWN_COLOR, stroke_width=2
            )
        )
    circles = [
        fm.CircleMarker(coordinates=_ll(p), radius=5, color=DRAWN_COLOR) for p in points
    ] + [
        fm.CircleMarker(coordinates=_ll(item["center"]), radius=3, color=CHIP_COLOR)
        for item in items
        if item["center"]
    ]
    _, url, attribution = TILES[tiles]
    the_map = fm.Map(
        expand=True,
        initial_center=_ll(DEFAULT_CENTER),
        initial_zoom=5,
        initial_camera_fit=_camera(items),
        interaction_configuration=fm.InteractionConfiguration(
            flags=fm.InteractionFlag.ALL & ~fm.InteractionFlag.ROTATE
        ),
        on_tap=on_tap,
        layers=[
            fm.TileLayer(url_template=url, user_agent_package_name="eo-art"),
            fm.PolygonLayer(polygons=polygons),
            fm.PolylineLayer(polylines=lines),
            fm.CircleLayer(circles=circles),
            fm.SimpleAttribution(text=attribution),
        ],
    )

    draw_card = card(
        heading("Draw"),
        ft.SegmentedButton(
            segments=[
                ft.Segment(value=k, label=ft.Text(t)) for k, t in DRAW_MODES.items()
            ],
            selected=[mode],
            on_change=lambda e: change_mode(e.control.selected[0]),
        ),
        caption(DRAW_HINTS[mode]),
        ft.Row(
            [
                *(
                    [
                        ft.FilledTonalButton(
                            "Finish polygon",
                            on_click=finish_polygon,
                            disabled=len(points) < 3,
                        )
                    ]
                    if mode == "polygon"
                    else []
                ),
                ft.TextButton(
                    "Undo point",
                    on_click=lambda: set_points(lambda prev: prev[:-1]),
                    disabled=not points,
                ),
                ft.TextButton(
                    "Clear shapes",
                    on_click=lambda: (set_drawn([]), set_points([])),
                    disabled=not drawn and not points,
                ),
            ],
            wrap=True,
        ),
        caption(f"{len(drawn)} shape(s) drawn, not saved yet."),
        ft.TextField(
            label="AOI name",
            hint_text="e.g. valley",
            value=name,
            dense=True,
            on_change=lambda e: set_name(e.control.value),
        ),
        ft.Checkbox(
            label="Overwrite if it exists",
            value=overwrite,
            on_change=lambda e: set_overwrite(e.control.value),
        ),
        ft.FilledButton(
            "Save AOI",
            icon=ft.Icons.SAVE,
            on_click=save_drawn,
            disabled=not drawn or not name.strip(),
        ),
    )
    import_card = card(
        heading("Import"),
        ft.TextField(
            label="AOI name (default: file name)",
            value=import_name,
            dense=True,
            on_change=lambda e: set_import_name(e.control.value),
        ),
        ft.OutlinedButton(
            "gpkg / GeoJSON / zipped shapefile…",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=import_file,
        ),
    )

    rows: list[ft.Control] = []
    for item in items:
        path: Path = item["path"]
        if item["error"] is not None:
            subtitle = f"Can't read: {item['error']}"
        else:
            lat, lon = item["center"]
            subtitle = f"Center {lat:.5f}, {lon:.5f}"
            if item["zones"] is not None:
                count, width, height = item["zones"]
                subtitle += (
                    f" · {count} zone(s), {width / 1000:.2f} × {height / 1000:.2f} km"
                )
            elif item["zones_error"] is not None:
                subtitle += f" · no zones: {item['zones_error']}"
        rows.append(
            ft.ListTile(
                title=ft.Text(path.stem),
                subtitle=ft.Text(subtitle, size=12),
                dense=True,
                content_padding=0,
                trailing=ft.Row(
                    [
                        ft.IconButton(
                            ft.Icons.LINK,
                            tooltip=f"Set as aoi_path in {cfg}"
                            if cfg
                            else "No acquire config",
                            disabled=cfg is None,
                            on_click=lambda _, p=path: use_as_aoi_path(p),
                        ),
                        ft.IconButton(
                            ft.Icons.DELETE_OUTLINE,
                            tooltip="Delete",
                            on_click=lambda _, p=path: delete(p),
                        ),
                    ],
                    tight=True,
                    spacing=0,
                ),
            )
        )
    aoi_card = card(
        heading("Project AOIs"),
        *(
            [
                dropdown(
                    "Acquire config for aoi_path",
                    [(c, c) for c in acquire_cfgs],
                    cfg,
                    set_target_cfg,
                    dense=True,
                )
            ]
            if acquire_cfgs
            else []
        ),
        *(rows or [caption("None yet.")]),
    )

    legend = caption(
        "Blue: saved AOIs. Dashed orange: "
        + (
            f"the zones Sentinel SR processes one after another ({edge_m / 1000:.2f} "
            "km chips, sentinel.zones in the acquire config)."
            if sr and sr.zones
            else f"the chip Sentinel SR actually processes ({edge_m / 1000:.2f} km, "
            "centred on the AOI's center; edge_size × resolution from the acquire "
            "config; enable sentinel.zones to cover the whole AOI)."
        )
        + " Green: drawn, not saved."
    )
    return ft.Row(
        [
            ft.Column(
                [
                    ft.Container(
                        the_map,
                        expand=True,
                        border_radius=10,
                        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    ),
                    ft.Row(
                        [
                            ft.SegmentedButton(
                                segments=[
                                    ft.Segment(value=k, label=ft.Text(t[0]))
                                    for k, t in TILES.items()
                                ],
                                selected=[tiles],
                                on_change=lambda e: set_tiles(e.control.selected[0]),
                            ),
                            ft.Container(legend, expand=True),
                        ]
                    ),
                ],
                expand=True,
            ),
            ft.Column(
                [draw_card, import_card, aoi_card],
                width=SIDE_WIDTH,
                scroll=ft.ScrollMode.AUTO,
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
        ],
        spacing=16,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )
