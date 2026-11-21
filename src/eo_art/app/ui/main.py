"""Flet entry point (see ``eo_art.app.launcher``)."""

import asyncio

import flet as ft

from eo_art.app.launcher import projects_dir
from eo_art.app.projects import is_project
from eo_art.app.ui.layout import App
from eo_art.app.ui.state import AppState

LAST_PROJECT_KEY = "eoart.last_project"
POLL_SECONDS = 1.0


async def main(page: ft.Page) -> None:
    page.title = "eo-art"
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.TEAL)
    page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.TEAL)
    page.theme_mode = ft.ThemeMode.SYSTEM

    prefs = ft.SharedPreferences()
    page.services.append(prefs)
    state = AppState()
    last = await prefs.get(LAST_PROJECT_KEY)
    if isinstance(last, str) and is_project(projects_dir() / last):
        state.select(last)

    page.render_views(App, state)

    async def poll() -> None:
        # One poller for the AppBar run status (run tabs tail the log
        # themselves) and for remembering the selected project.
        saved = state.project_name
        while True:
            if state.sync_run():
                state.refresh()
            if state.project_name and state.project_name != saved:
                saved = state.project_name
                await prefs.set(LAST_PROJECT_KEY, saved)
            await asyncio.sleep(POLL_SECONDS)

    page.run_task(poll)
