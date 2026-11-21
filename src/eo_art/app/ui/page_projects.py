"""Create, adopt and switch between project folders."""

import streamlit as st

from eo_art.app.launcher import projects_dir
from eo_art.app.projects import (
    adopt_folder,
    candidate_folders,
    create_project,
    list_projects,
)
from eo_art.app.ui.common import current_project, select_project


def render() -> None:
    base = projects_dir()
    st.title("Projects")
    st.caption(
        f"Projects live in `{base}`, i.e. `projects/` inside the folder "
        "`eoart-app` was started in."
    )

    left, right = st.columns(2)
    with left, st.form("create_project", clear_on_submit=True):
        st.subheader("New project")
        name = st.text_input("Name", placeholder="e.g. alps_dusk")
        if st.form_submit_button("Create", type="primary"):
            try:
                project = create_project(base, name)
            except (ValueError, FileExistsError) as exc:
                st.error(str(exc))
            else:
                select_project(project.name)
                st.rerun()

    with right, st.form("adopt_folder"):
        st.subheader("Use existing folder")
        folders = [p.name for p in candidate_folders(base)]
        folder = st.selectbox(
            "Folder", folders, index=None, placeholder="Pick a folder"
        )
        st.caption("Adds aoi/, configs/, out/ and starter configs; keeps your files.")
        if st.form_submit_button("Adopt", disabled=not folders) and folder:
            project = adopt_folder(base, folder)
            select_project(project.name)
            st.rerun()

    st.subheader("All projects")
    projects = list_projects(base)
    if not projects:
        st.info("No projects yet.")
        return
    active = current_project()
    for project in projects:
        cols = st.columns([3, 2, 2, 2])
        marker = "▶ " if active and active.root == project.root else ""
        cols[0].markdown(f"**{marker}{project.name}**  \n`{project.root}`")
        cols[1].metric("AOIs", len(project.aoi_files()))
        cols[2].metric("Configs", len(project.config_files()))
        if cols[3].button("Open", key=f"open_{project.name}"):
            select_project(project.name)
            st.rerun()
