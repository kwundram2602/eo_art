"""Small building blocks shared by the pages."""

from collections.abc import Callable
from pathlib import Path

import flet as ft

from eo_art.app.configs import Validation
from eo_art.app.projects import Project
from eo_art.app.ui.state import AppCtx, AppState

MONO = "monospace"


def use_app() -> AppState | None:
    """The app state, or None for a stale component after a page switch."""
    return ft.use_context(AppCtx)


def toast(message: str, error: bool = False) -> None:
    ft.context.page.show_dialog(
        ft.SnackBar(
            ft.Text(message),
            bgcolor=ft.Colors.ERROR_CONTAINER if error else None,
            show_close_icon=error,
        )
    )


def file_picker() -> ft.FilePicker:
    """The page's single FilePicker service, registered on first use."""
    page = ft.context.page
    for service in page.services:
        if isinstance(service, ft.FilePicker):
            return service
    picker = ft.FilePicker()
    page.services.append(picker)
    return picker


async def save_bytes_as(data: bytes, file_name: str, extension: str) -> None:
    """Ask where to save ``data`` and write it there."""
    target = await file_picker().save_file(
        file_name=file_name, allowed_extensions=[extension], src_bytes=data
    )
    if target:
        Path(target).write_bytes(data)
        toast(f"Saved {target}")


def rel(project: Project, path: Path) -> str:
    """Path as the pipelines see it: relative to the project folder."""
    try:
        return str(path.relative_to(project.root))
    except ValueError:
        return str(path)


def title(text: str, subtitle: str | None = None) -> ft.Control:
    controls: list[ft.Control] = [
        ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_LARGE)
    ]
    if subtitle:
        controls.append(
            ft.Text(subtitle, color=ft.Colors.ON_SURFACE_VARIANT, selectable=True)
        )
    return ft.Column(controls, spacing=4)


def heading(text: str) -> ft.Control:
    return ft.Text(text, theme_style=ft.TextThemeStyle.TITLE_SMALL)


def caption(text: str) -> ft.Control:
    return ft.Text(text, size=12, color=ft.Colors.ON_SURFACE_VARIANT, selectable=True)


def card(*controls: ft.Control, expand: bool | int | None = None) -> ft.Control:
    return ft.Container(
        ft.Column(list(controls), spacing=10),
        padding=16,
        border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=10,
        expand=expand,
    )


def code(text: str, height: float | None = None) -> ft.Control:
    """Monospace block spanning the available width."""
    return ft.Row(
        [
            ft.Container(
                ft.Column(
                    [ft.Text(text, font_family=MONO, size=12, selectable=True)],
                    scroll=ft.ScrollMode.AUTO,
                    auto_scroll=height is not None,
                ),
                padding=10,
                height=height,
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                border_radius=8,
                expand=True,
            )
        ]
    )


def notice(text: str, kind: str = "info") -> ft.Control:
    icon, color = {
        "info": (ft.Icons.INFO_OUTLINE, ft.Colors.PRIMARY),
        "ok": (ft.Icons.CHECK_CIRCLE_OUTLINE, ft.Colors.GREEN),
        "warn": (ft.Icons.WARNING_AMBER, ft.Colors.AMBER),
        "error": (ft.Icons.ERROR_OUTLINE, ft.Colors.ERROR),
    }[kind]
    return ft.Container(
        ft.Row(
            [ft.Icon(icon, color=color), ft.Text(text, expand=True, selectable=True)],
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
        padding=12,
        border=ft.Border.all(1, color),
        border_radius=8,
    )


@ft.component
def Expandable(
    title: ft.Control,
    controls: list[ft.Control],
    initially_expanded: bool = False,
    leading: ft.Control | None = None,
    dense: bool | None = None,
    subtitle: ft.Control | None = None,
    tile_padding: ft.Padding | None = None,
) -> ft.Control:
    """ExpansionTile that keeps being open or closed while its parent
    re-renders. A bare tile snaps back to its ``expanded`` prop on every
    re-render, which collapses a section the user just opened while they
    edit a field in it."""
    expanded, set_expanded = ft.use_state(initially_expanded)
    return ft.ExpansionTile(
        title=title,
        subtitle=subtitle,
        leading=leading,
        tile_padding=tile_padding,
        controls=controls,
        expanded=expanded,
        dense=dense,
        maintain_state=True,
        on_change=lambda e: set_expanded(e.data in (True, "true")),
    )


def validation(
    name: str, result: Validation, expanded: bool | None = None
) -> ft.Control:
    """Valid/invalid header with the resolved config underneath."""
    controls: list[ft.Control] = []
    if not result.ok:
        controls.append(notice(str(result.error), "error"))
    if result.resolved:
        controls.append(code(result.resolved))
    return keyed(
        Expandable,
        f"validation:{name}",
        ft.Text(f"{name} — {'valid' if result.ok else 'invalid'}"),
        controls,
        (not result.ok) if expanded is None else expanded,
        leading=ft.Icon(
            ft.Icons.CHECK_CIRCLE if result.ok else ft.Icons.ERROR,
            color=ft.Colors.GREEN if result.ok else ft.Colors.ERROR,
        ),
    )


def dropdown(
    label: str,
    options: list[tuple[str, str]],
    value: str | None,
    on_change: Callable[[str | None], None],
    **kwargs,
) -> ft.Dropdown:
    """Dropdown over (key, text) pairs calling ``on_change(key)``."""
    return ft.Dropdown(
        label=label,
        value=value,
        options=[ft.DropdownOption(key=k, text=t) for k, t in options],
        on_select=lambda e: on_change(e.control.value),
        **kwargs,
    )


@ft.component
def NeedsProject() -> ft.Control:
    return ft.Container(
        ft.Column(
            [
                ft.Icon(ft.Icons.FOLDER_OPEN, size=48, color=ft.Colors.OUTLINE),
                ft.Text("Select or create a project first."),
                ft.FilledButton(
                    "Projects",
                    icon=ft.Icons.FOLDER,
                    on_click=lambda: ft.context.page.navigate("/projects"),
                ),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        alignment=ft.Alignment.CENTER,
        expand=True,
    )


def keyed(
    component: Callable[..., ft.Control], key: str, *args: object, **kwargs: object
) -> ft.Control:
    """``component(*args)`` with a reconciliation key: a new key mounts a fresh
    instance, so its local state starts over (e.g. after a file was saved)."""
    return component(*args, key=key, **kwargs)
