"""Create, adopt and switch between project folders."""

from typing import cast

import flet as ft

from eo_art.app.launcher import projects_dir
from eo_art.app.projects import (
    adopt_folder,
    candidate_folders,
    copy_project,
    create_project,
    list_projects,
)
from eo_art.app.ui.widgets import (
    caption,
    card,
    dropdown,
    heading,
    title,
    toast,
    use_app,
)


@ft.component
def ProjectsPage() -> ft.Control:
    state = use_app()
    name, set_name = ft.use_state("")
    folder, set_folder = ft.use_state(cast(str | None, None))
    copy_from, set_copy_from = ft.use_state(cast(str | None, None))
    copy_name, set_copy_name = ft.use_state("")
    if state is None:
        return ft.Container()
    _ = state.revision
    base = projects_dir()

    def create() -> None:
        try:
            project = create_project(base, name)
        except (ValueError, FileExistsError) as exc:
            toast(str(exc), error=True)
            return
        set_name("")
        state.select(project.name)
        state.refresh()
        toast(f"Created {project.name}")

    def adopt() -> None:
        if not folder:
            return
        project = adopt_folder(base, folder)
        set_folder(None)
        state.select(project.name)
        state.refresh()
        toast(f"Adopted {project.name}")

    projects = list_projects(base)
    by_name = {p.name: p for p in projects}
    # Default source: the open project.
    source = copy_from if copy_from in by_name else state.project_name
    source = source if source in by_name else None

    def copy() -> None:
        if source is None:
            return
        try:
            project = copy_project(base, by_name[source], copy_name)
        except (ValueError, FileExistsError) as exc:
            toast(str(exc), error=True)
            return
        set_copy_name("")
        state.select(project.name)
        state.refresh()
        toast(f"Copied {source} to {project.name}")

    folders = [p.name for p in candidate_folders(base)]
    new_card = card(
        heading("New project"),
        ft.TextField(
            label="Name",
            hint_text="e.g. alps_dusk",
            value=name,
            on_change=lambda e: set_name(e.control.value),
            on_submit=lambda: create(),
        ),
        ft.FilledButton(
            "Create", icon=ft.Icons.ADD, on_click=create, disabled=not name.strip()
        ),
        expand=True,
    )
    adopt_card = card(
        heading("Use existing folder"),
        dropdown(
            "Folder",
            [(f, f) for f in folders],
            folder,
            set_folder,
            hint_text="No other folders" if not folders else "Pick a folder",
            expand=True,
        ),
        caption("Adds aoi/, configs/, out/ and starter configs; keeps your files."),
        ft.OutlinedButton(
            "Adopt",
            icon=ft.Icons.DRIVE_FOLDER_UPLOAD,
            on_click=adopt,
            disabled=not folder,
        ),
        expand=True,
    )
    copy_card = card(
        heading("Copy project"),
        dropdown(
            "From",
            [(n, n) for n in by_name],
            source,
            set_copy_from,
            hint_text="No projects yet" if not by_name else "Pick a project",
            expand=True,
        ),
        ft.TextField(
            label="New name",
            hint_text=f"e.g. {source}_v2" if source else "e.g. alps_v2",
            value=copy_name,
            on_change=lambda e: set_copy_name(e.control.value),
            on_submit=lambda: copy(),
        ),
        caption("Copies the top-level configs and the AOIs; out/ stays behind."),
        ft.OutlinedButton(
            "Copy",
            icon=ft.Icons.COPY_ALL,
            on_click=copy,
            disabled=source is None or not copy_name.strip(),
        ),
        expand=True,
    )

    rows: list[ft.Control] = []
    for project in projects:
        active = project.name == state.project_name
        rows.append(
            ft.ListTile(
                leading=ft.Icon(
                    ft.Icons.FOLDER_SPECIAL if active else ft.Icons.FOLDER_OUTLINED,
                    color=ft.Colors.PRIMARY if active else None,
                ),
                title=ft.Text(
                    project.name, weight=ft.FontWeight.BOLD if active else None
                ),
                subtitle=ft.Text(
                    f"{len(project.aoi_files())} AOIs · "
                    f"{len(project.config_files())} configs · {project.root}"
                ),
                trailing=None
                if active
                else ft.TextButton(
                    "Open", on_click=lambda _, n=project.name: state.select(n)
                ),
                selected=active,
            )
        )

    return ft.Column(
        [
            title(
                "Projects",
                f"Projects live in {base}, i.e. projects/ inside the folder "
                "eoart-app was started in.",
            ),
            ft.ResponsiveRow(
                [
                    ft.Container(new_card, col={"md": 6, "lg": 4}),
                    ft.Container(adopt_card, col={"md": 6, "lg": 4}),
                    ft.Container(copy_card, col={"md": 6, "lg": 4}),
                ]
            ),
            heading("All projects"),
            *(rows or [caption("No projects yet.")]),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )
