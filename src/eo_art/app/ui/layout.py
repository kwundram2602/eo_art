"""App frame: AppBar (project + run status), NavigationRail over the stages,
and a tab bar per stage. Each tab is its own route (``/<stage>/<tab>``)."""

from collections.abc import Callable
from dataclasses import dataclass

import flet as ft

from eo_art.app.launcher import projects_dir
from eo_art.app.projects import list_projects
from eo_art.app.runner import ACQUIRE_KINDS, PipelineKind
from eo_art.app.ui.state import AppCtx, AppState


@dataclass(frozen=True)
class Tab:
    path: str
    label: str
    icon: ft.IconData
    component: Callable[[], ft.Control]


@dataclass(frozen=True)
class Stage:
    path: str
    label: str
    icon: ft.IconData
    selected_icon: ft.IconData
    # A stage without tabs is a single page.
    component: Callable[[], ft.Control] | None = None
    tabs: tuple[Tab, ...] = ()


def _stages() -> list[Stage]:
    # Imported here so this module loads without pulling in every page.
    from eo_art.app.ui.pages import acquire, aoi, forge3d, projects

    return [
        Stage(
            "projects",
            "Projects",
            ft.Icons.FOLDER_OUTLINED,
            ft.Icons.FOLDER,
            component=projects.ProjectsPage,
        ),
        Stage("aoi", "AOI", ft.Icons.MAP_OUTLINED, ft.Icons.MAP, component=aoi.AoiPage),
        Stage(
            "acquire",
            "Acquire",
            ft.Icons.SATELLITE_ALT_OUTLINED,
            ft.Icons.SATELLITE_ALT,
            tabs=(
                Tab("config", "Config", ft.Icons.TUNE, acquire.ConfigPage),
                Tab("run", "Run", ft.Icons.PLAY_ARROW, acquire.RunPage),
                Tab("results", "Results", ft.Icons.LAYERS, acquire.ResultsPage),
                Tab("composite", "Composite", ft.Icons.PALETTE, acquire.CompositePage),
            ),
        ),
        Stage(
            "forge3d",
            "Forge3D",
            ft.Icons.TERRAIN_OUTLINED,
            ft.Icons.TERRAIN,
            tabs=(
                Tab("configs", "Configs", ft.Icons.TUNE, forge3d.ConfigsPage),
                Tab("run", "Run", ft.Icons.PLAY_ARROW, forge3d.RunPage),
                Tab("renders", "Renders", ft.Icons.IMAGE, forge3d.RendersPage),
            ),
        ),
    ]


def run_route(kind: PipelineKind) -> str:
    return "/acquire/run" if kind in ACQUIRE_KINDS else "/forge3d/run"


def _segments(location: str) -> list[str]:
    return [s for s in location.split("?")[0].split("/") if s]


def _stage_layout(stage: Stage) -> Callable[[], ft.Control]:
    @ft.component
    def StageLayout() -> ft.Control:
        segments = _segments(ft.use_route_location())
        current = segments[1] if len(segments) > 1 else stage.tabs[0].path
        paths = [t.path for t in stage.tabs]
        index = paths.index(current) if current in paths else 0

        def on_change(e: ft.Event[ft.Tabs]) -> None:
            target = stage.tabs[e.control.selected_index].path
            ft.context.page.navigate(f"/{stage.path}/{target}")

        return ft.Column(
            [
                ft.Tabs(
                    length=len(stage.tabs),
                    selected_index=index,
                    on_change=on_change,
                    content=ft.TabBar(
                        tabs=[ft.Tab(label=t.label, icon=t.icon) for t in stage.tabs],
                        tab_alignment=ft.TabAlignment.START,
                        scrollable=True,
                    ),
                ),
                ft.Container(ft.use_route_outlet(), expand=True, padding=20),
            ],
            spacing=0,
            expand=True,
        )

    return StageLayout


def routes(stages: list[Stage]) -> list[ft.Route]:
    children: list[ft.Route] = []
    for stage in stages:
        if stage.component is not None:
            children.append(ft.Route(path=stage.path, component=stage.component))
            continue
        children.append(
            ft.Route(
                path=stage.path,
                component=_stage_layout(stage),
                children=[
                    ft.Route(index=True, component=stage.tabs[0].component),
                    *(ft.Route(path=t.path, component=t.component) for t in stage.tabs),
                ],
            )
        )
    first = stages[0].component
    assert first is not None
    return [
        ft.Route(
            component=lambda: Shell(stages),
            children=[ft.Route(index=True, component=first), *children],
        )
    ]


@ft.component
def Shell(stages: list[Stage]) -> ft.Control:
    segments = _segments(ft.use_route_location())
    paths = [s.path for s in stages]
    index = paths.index(segments[0]) if segments and segments[0] in paths else 0
    rail = ft.NavigationRail(
        selected_index=index,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=88,
        destinations=[
            ft.NavigationRailDestination(
                icon=s.icon, selected_icon=s.selected_icon, label=s.label
            )
            for s in stages
        ],
        on_change=lambda e: ft.context.page.navigate(
            f"/{stages[e.control.selected_index].path}"
        ),
    )
    outlet = ft.use_route_outlet()
    # Stage layouts pad below their tab bar themselves.
    single_page = stages[index].component is not None
    return ft.Row(
        [
            rail,
            ft.VerticalDivider(width=1),
            ft.Container(outlet, expand=True, padding=20 if single_page else 0),
        ],
        expand=True,
        spacing=0,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )


def _run_chip(state: AppState) -> ft.Control | None:
    if state.run_kind is None:
        return None
    kind = PipelineKind[state.run_kind]
    if state.run_running:
        icon, color, text = ft.Icons.AUTORENEW, ft.Colors.PRIMARY, "running"
    elif state.run_returncode == 0:
        icon, color, text = ft.Icons.CHECK_CIRCLE, ft.Colors.GREEN, "finished"
    else:
        icon, color, text = (
            ft.Icons.ERROR,
            ft.Colors.ERROR,
            (f"failed ({state.run_returncode})"),
        )
    return ft.Chip(
        label=ft.Text(f"{kind.value}: {text}"),
        leading=ft.Icon(icon, color=color),
        on_click=lambda: ft.context.page.navigate(run_route(kind)),
        tooltip="Open the run",
    )


def _appbar(state: AppState) -> ft.AppBar:
    names = [p.name for p in list_projects(projects_dir())]
    picker = ft.Dropdown(
        leading_icon=ft.Icon(ft.Icons.FOLDER_OUTLINED),
        tooltip="Project",
        value=state.project_name if state.project_name in names else None,
        options=[ft.DropdownOption(key=n, text=n) for n in names],
        hint_text="No projects yet" if not names else "Select a project",
        on_select=lambda e: state.select(e.control.value),
        width=260,
        dense=True,
    )
    chip = _run_chip(state)
    return ft.AppBar(
        leading=ft.Icon(ft.Icons.LANDSCAPE),
        title=ft.Text("eo-art"),
        actions=[
            *([chip] if chip else []),
            ft.Container(picker, padding=ft.Padding.only(left=12, right=16)),
        ],
    )


@ft.component
def App(state: AppState) -> ft.View:
    # Read so a refresh re-lists projects in the AppBar.
    _ = state.revision
    stages = ft.use_memo(_stages, [])
    table = ft.use_memo(lambda: routes(stages), [])

    def view() -> ft.View:
        return ft.View(
            route="/",
            appbar=_appbar(state),
            controls=[ft.Router(table, not_found=NotFound)],
            padding=0,
        )

    return AppCtx(state, view)


@ft.component
def NotFound() -> ft.Control:
    return ft.Column(
        [
            ft.Text("Page not found."),
            ft.TextButton(
                "Projects", on_click=lambda: ft.context.page.navigate("/projects")
            ),
        ]
    )
