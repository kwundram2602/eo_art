"""Browse rendered images/videos and past run logs of the project."""

from pathlib import Path

import streamlit as st

from eo_art.app.projects import LOGS_DIRNAME, Project
from eo_art.app.runner import tail
from eo_art.app.ui.common import rel, require_project

_MEDIA = {".png", ".jpg", ".jpeg", ".gif", ".mp4"}
# Pipeline-internal folders: prep/acquire caches and per-frame dumps.
_SKIP_DIRS = {"_prep", "_acquire", "frames", LOGS_DIRNAME}


def _media(project: Project) -> list[Path]:
    if not project.out_dir.exists():
        return []
    files = [
        p
        for p in project.out_dir.rglob("*")
        if p.suffix.lower() in _MEDIA
        and not _SKIP_DIRS & set(p.relative_to(project.out_dir).parts[:-1])
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _rasters(project: Project) -> list[Path]:
    if not project.out_dir.exists():
        return []
    return sorted(project.out_dir.glob("*.tif"))


def render() -> None:
    project = require_project()
    st.title(f"Results · {project.name}")

    media = _media(project)
    # A slider needs min < max, so only offer it with more than one item.
    limit = (
        st.slider("Show newest", 1, len(media), min(len(media), 12))
        if len(media) > 1
        else len(media)
    )
    if not media:
        st.info("No rendered images or videos under out/ yet.")
    cols = st.columns(3)
    for i, path in enumerate(media[:limit]):
        with cols[i % 3]:
            if path.suffix.lower() == ".mp4":
                st.video(str(path))
            else:
                st.image(str(path), width="stretch")
            st.caption(f"`{rel(project, path)}`")
            resolved = path.parent / "resolved.yaml"
            if resolved.exists():
                with st.expander("resolved.yaml"):
                    st.code(resolved.read_text(), language="yaml")

    rasters = _rasters(project)
    if rasters:
        st.subheader("Rasters")
        for path in rasters:
            st.markdown(
                f"- `{rel(project, path)}` ({path.stat().st_size / 1e6:.1f} MB)"
            )

    st.subheader("Run logs")
    logs = sorted(project.logs_dir.glob("*.log"), reverse=True)
    if not logs:
        st.caption("No runs yet.")
        return
    log = st.selectbox("Log", logs, format_func=lambda p: p.name)
    st.code(tail(log, 1000), language=None)
