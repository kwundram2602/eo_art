"""App-wide state: the projects folder, the active project and its run."""

from dataclasses import dataclass
from typing import cast

import flet as ft

from eo_art.app.launcher import projects_dir
from eo_art.app.projects import Project, is_project
from eo_art.app.runner import RunHandle, log_kind

# Project root -> its run. Process-wide, so a run outlives the page that
# started it; one run per project at a time.
RUNS: dict[str, RunHandle] = {}


@ft.observable
@dataclass
class AppState:
    project_name: str | None = None
    # Bumped after the app writes files, so lists read from disk refresh.
    revision: int = 0
    # Snapshot of the active project's run, refreshed by the poller.
    run_kind: str | None = None
    run_running: bool = False
    run_returncode: int | None = None
    # GeoTIFF the Composite tab opens next (set from Acquire › Results).
    composite_path: str | None = None

    @property
    def project(self) -> Project | None:
        if not self.project_name:
            return None
        root = projects_dir() / self.project_name
        return Project(root) if is_project(root) else None

    def select(self, name: str | None) -> None:
        self.project_name = name
        self.composite_path = None
        self.sync_run()

    def refresh(self) -> None:
        self.revision += 1

    def run(self) -> RunHandle | None:
        project = self.project
        return RUNS.get(str(project.root)) if project else None

    def sync_run(self) -> bool:
        """Copy the active run's status into observable fields; True when a
        run has just finished."""
        handle = self.run()
        was_running = self.run_running
        if handle is None:
            self.run_kind, self.run_running, self.run_returncode = None, False, None
            return False
        kind = log_kind(handle.log_path)
        self.run_kind = kind.name if kind else None
        self.run_running = handle.running
        self.run_returncode = handle.returncode
        return was_running and not handle.running


AppCtx = ft.create_context(cast(AppState | None, None))
