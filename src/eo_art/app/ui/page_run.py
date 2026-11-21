"""Pick a pipeline, review its resolved config, run it and follow the log."""

import shlex
from pathlib import Path

import streamlit as st
from omegaconf import OmegaConf

from eo_art.app.configs import (
    ConfigKind,
    Validation,
    load_file,
    overlay_names,
    validate_acquire,
    validate_render,
    validate_sr,
)
from eo_art.app.projects import Project
from eo_art.app.runner import (
    PipelineKind,
    PipelineSpec,
    build_command,
    log_path_for,
    start,
    stop,
    tail,
)
from eo_art.app.ui.common import active_runs, configs_of_kind, rel, require_project

_DESCRIPTIONS = {
    PipelineKind.acquire: "Sentinel SR → DEM → DTM → align. Writes "
    "`out/dtm.tif` and `out/optical_aligned.tif`.",
    PipelineKind.sr: "Only the Sentinel-2 super-resolution. Writes "
    "`out/<aoi-name>_sr.tif`, or with `sentinel.timeseries.enabled` one GeoTIFF "
    "per interval into `out/<aoi-name>_sr_timeseries/`.",
    PipelineKind.render: "forge3d render of the merged render config(s).",
    PipelineKind.acquire_render: "Acquire first, then render with its DTM as "
    "`input.path` and its optical raster as the overlay's path.",
}


def _pick_acquire(project: Project) -> Path | None:
    configs = configs_of_kind(project, ConfigKind.acquire)
    if not configs:
        st.warning("No acquire config in this project (create one on Configs).")
        return None
    return st.selectbox("Acquire config", configs, format_func=lambda p: p.name)


def _pick_layers(project: Project) -> list[Path]:
    configs = configs_of_kind(project, ConfigKind.render)
    if not configs:
        st.warning("No render config in this project (create one on Configs).")
        return []
    default = [p for p in configs if p.name == "render.yaml"] or configs[:1]
    return st.multiselect(
        "Render configs (merged left to right, in the order picked)",
        configs,
        default=default,
        format_func=lambda p: p.name,
    )


def _show(title: str, result: Validation) -> bool:
    with st.expander(
        f"{'✅' if result.ok else '❌'} {title} — resolved config",
        expanded=not result.ok,
    ):
        if not result.ok:
            st.error(result.error)
        if result.resolved:
            st.code(result.resolved, language="yaml")
    return result.ok


def _form(project: Project) -> PipelineSpec | None:
    kind = st.radio(
        "Pipeline",
        list(PipelineKind),
        format_func=lambda k: k.value,
        horizontal=True,
    )
    st.caption(_DESCRIPTIONS[kind])

    acquire_cfg: Path | None = None
    layers: list[Path] = []
    overrides: list[str] = []
    render_overrides: list[str] = []
    sweep = None
    fail_fast = False
    acquire_overlay = None

    if kind is not PipelineKind.render:
        acquire_cfg = _pick_acquire(project)
    if kind is PipelineKind.sr:
        aois = [rel(project, p) for p in project.aoi_files()]
        aoi_choice = st.selectbox(
            "AOI", ["(from config)", *aois], help="Passed as --set aoi_path=..."
        )
        if aoi_choice != "(from config)":
            overrides.append(f"aoi_path={aoi_choice}")
    if kind in (PipelineKind.render, PipelineKind.acquire_render):
        layers = _pick_layers(project)
        raw_overrides = st.text_area(
            "Overrides (one PATH=VALUE per line)",
            placeholder="render.width=1280\nrender.backend=offscreen",
        )
        render_overrides = [
            line.strip() for line in raw_overrides.splitlines() if line.strip()
        ]
        overrides += render_overrides
        sweep = (
            st.text_input(
                "Sweep (PATH=V1,V2,...)", placeholder="render.pbr.exposure=1.0,1.8"
            )
            or None
        )
        fail_fast = st.checkbox("Fail fast")
    if kind is PipelineKind.acquire_render and layers:
        names = overlay_names(layers)
        if names:
            acquire_overlay = st.selectbox(
                "Overlay to receive the optical raster", names
            )
    no_cache = st.checkbox("No cache (recompute everything)")

    valid = True
    if acquire_cfg is not None:
        if kind is PipelineKind.sr:
            valid &= _show(acquire_cfg.name + " (SR slice)", validate_sr(acquire_cfg))
        else:
            valid &= _show(acquire_cfg.name, validate_acquire(acquire_cfg))
            if OmegaConf.select(load_file(acquire_cfg), "sentinel.timeseries.enabled"):
                st.info(
                    "`sentinel.timeseries` is enabled, but acquire ignores it and "
                    "uses one reference scene. Run **SR only** for the timeseries.",
                    icon=":material/info:",
                )
    if layers:
        valid &= _show(
            " + ".join(p.name for p in layers),
            validate_render(layers, render_overrides),
        )

    try:
        spec = PipelineSpec(
            kind,
            acquire_config=rel(project, acquire_cfg) if acquire_cfg else None,
            render_layers=[rel(project, p) for p in layers],
            overrides=overrides,
            sweep=sweep,
            no_cache=no_cache,
            fail_fast=fail_fast,
            acquire_overlay=acquire_overlay,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return None
    return spec if valid else None


def _live(project: Project) -> None:
    handle = active_runs().get(str(project.root))
    if handle is None:
        return

    @st.fragment(run_every=1.0 if handle.running else None)
    def _log() -> None:
        if handle.running:
            st.info(
                f"Running since {handle.started:%H:%M:%S} · `{handle.log_path.name}`"
            )
            if st.button("Stop", type="secondary"):
                stop(handle)
                st.rerun(scope="app")
        elif handle.returncode == 0:
            st.success(f"Finished OK · `{rel(project, handle.log_path)}`")
        else:
            st.error(
                f"Exited with code {handle.returncode} · "
                f"`{rel(project, handle.log_path)}`"
            )
        st.code(tail(handle.log_path, 400) or "(no output yet)", language=None)
        # Polling stops on the next full rerun once the process is done.
        if not handle.running and st.session_state.get("_was_running"):
            st.session_state["_was_running"] = False
            st.rerun(scope="app")
        st.session_state["_was_running"] = handle.running

    _log()


def render() -> None:
    project = require_project()
    st.title(f"Run · {project.name}")

    spec = _form(project)
    runs = active_runs()
    busy = (handle := runs.get(str(project.root))) is not None and handle.running

    if spec is not None:
        command = build_command(spec)
        st.code(
            f"cd {shlex.quote(str(project.root))}\n{shlex.join(command)}",
            language="bash",
        )
        if st.button("Start", type="primary", disabled=busy):
            log_path = log_path_for(project.logs_dir, spec.kind)
            runs[str(project.root)] = start(command, project.root, log_path)
            st.session_state["_was_running"] = True
            st.rerun()
    else:
        st.button("Start", type="primary", disabled=True)

    _live(project)
