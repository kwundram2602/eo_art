"""Entry script for ``streamlit run`` (see ``eo_art.app.launcher``)."""

import streamlit as st

from eo_art.app.launcher import projects_dir
from eo_art.app.projects import list_projects
from eo_art.app.ui import (
    page_aoi,
    page_composite,
    page_configs,
    page_projects,
    page_results,
    page_run,
)
from eo_art.app.ui.common import active_runs, current_project, select_project

st.set_page_config(page_title="eo-art", page_icon="🏔️", layout="wide")


def _sidebar() -> None:
    st.sidebar.caption(f"Projects folder: `{projects_dir()}`")
    names = [p.name for p in list_projects(projects_dir())]
    if not names:
        st.sidebar.info("No projects yet — create one on the Projects page.")
        return
    current = current_project()
    index = names.index(current.name) if current else None
    choice = st.sidebar.selectbox(
        "Project", names, index=index, placeholder="Select a project"
    )
    if choice and (current is None or choice != current.name):
        select_project(choice)
        st.rerun()
    if current is not None:
        handle = active_runs().get(str(current.root))
        if handle is not None and handle.running:
            st.sidebar.warning(f"Pipeline running: `{handle.log_path.name}`")


_sidebar()
st.navigation(
    [
        st.Page(page_projects.render, title="Projects", icon="📁", url_path="projects"),
        st.Page(page_aoi.render, title="AOI", icon="🗺️", url_path="aoi"),
        st.Page(page_configs.render, title="Configs", icon="⚙️", url_path="configs"),
        st.Page(page_run.render, title="Run", icon="▶️", url_path="run"),
        st.Page(page_results.render, title="Results", icon="🖼️", url_path="results"),
        st.Page(
            page_composite.render, title="Composite", icon="🎨", url_path="composite"
        ),
    ]
).run()
