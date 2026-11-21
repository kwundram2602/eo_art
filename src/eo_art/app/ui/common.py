"""Shared UI state: projects folder, the active project, and running pipelines."""

from pathlib import Path

import streamlit as st

from eo_art.app.configs import ConfigKind, detect_kind
from eo_art.app.launcher import projects_dir
from eo_art.app.projects import Project, is_project
from eo_art.app.runner import RunHandle

PROJECT_PARAM = "project"


def current_project() -> Project | None:
    # session_state survives page switches (which clear query params); the
    # query param only restores the selection after a browser reload.
    name = st.session_state.get(PROJECT_PARAM) or st.query_params.get(PROJECT_PARAM)
    if not name:
        return None
    root = projects_dir() / name
    if not is_project(root):
        return None
    if st.query_params.get(PROJECT_PARAM) != name:
        st.query_params[PROJECT_PARAM] = name
    return Project(root)


def select_project(name: str) -> None:
    st.session_state[PROJECT_PARAM] = name
    st.query_params[PROJECT_PARAM] = name


def require_project() -> Project:
    project = current_project()
    if project is None:
        st.info("Select or create a project in the sidebar first.")
        st.stop()
    return project


def rel(project: Project, path: Path) -> str:
    """Path as the pipelines see it: relative to the project folder."""
    try:
        return str(path.relative_to(project.root))
    except ValueError:
        return str(path)


def configs_of_kind(project: Project, kind: ConfigKind) -> list[Path]:
    return [p for p in project.config_files() if detect_kind(p) is kind]


@st.cache_resource
def active_runs() -> dict[str, RunHandle]:
    """Project root -> its run. Process-wide, so a run outlives a browser
    reload or a second tab instead of being tied to one Streamlit session."""
    return {}
