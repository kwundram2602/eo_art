"""Forge3D › Renders: rendered images and videos under ``out/``, newest first."""

from pathlib import Path
from typing import cast

import flet as ft
import flet_video as fv

from eo_art.app.projects import Project
from eo_art.app.results import media_files
from eo_art.app.ui.widgets import (
    Expandable,
    NeedsProject,
    caption,
    code,
    dropdown,
    keyed,
    rel,
    title,
    use_app,
)

LIMITS = {"12": 12, "24": 24, "48": 48, "all": None}
THUMB_HEIGHT = 260


def _media(path: Path) -> ft.Control:
    if path.suffix.lower() == ".mp4":
        return fv.Video(
            playlist=[fv.VideoMedia(resource=str(path))],
            autoplay=False,
            height=THUMB_HEIGHT,
        )
    return ft.Image(
        src=path.read_bytes(),
        fit=ft.BoxFit.CONTAIN,
        height=THUMB_HEIGHT,
        cache_height=THUMB_HEIGHT * 2,
    )


def _item(project: Project, path: Path) -> ft.Control:
    controls: list[ft.Control] = [_media(path), caption(rel(project, path))]
    resolved = path.parent / "resolved.yaml"
    if resolved.exists():
        controls.append(
            keyed(
                Expandable,
                f"resolved:{path}",
                ft.Text("resolved.yaml", size=13),
                [code(resolved.read_text(), height=300)],
                dense=True,
            )
        )
    return ft.Container(
        ft.Column(controls, spacing=6),
        col={"md": 6, "xl": 4},
        padding=8,
        border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=10,
    )


@ft.component
def RendersPage() -> ft.Control:
    state = use_app()
    limit, set_limit = ft.use_state(cast(str, "12"))
    if state is None:
        return ft.Container()
    project = state.project
    if project is None:
        return NeedsProject()
    _ = state.revision

    media = media_files(project)
    shown = media[: LIMITS[limit]]
    return ft.Column(
        [
            ft.Row(
                [
                    ft.Container(
                        title("Renders", f"{len(media)} images/videos under out/."),
                        expand=True,
                    ),
                    dropdown(
                        "Show newest",
                        [(k, k) for k in LIMITS],
                        limit,
                        lambda v: set_limit(v or "12"),
                        width=160,
                    ),
                ]
            ),
            ft.ResponsiveRow([_item(project, p) for p in shown], run_spacing=12)
            if shown
            else caption("No rendered images or videos under out/ yet."),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )
