"""Config overview and editing: generated form, raw YAML, resolved view."""

from itertools import groupby
from pathlib import Path
from typing import Any

import streamlit as st
import yaml
from omegaconf import OmegaConf

from eo_art.app.config_forms import (
    FieldSpec,
    apply_changes,
    changed,
    current_values,
    is_active,
    iter_fields,
    merged_with_defaults,
)
from eo_art.app.configs import (
    ConfigKind,
    detect_kind,
    load_file,
    schema_for,
    validate_acquire,
    validate_render,
    validate_sr,
)
from eo_art.app.projects import Project, validate_name
from eo_art.app.templates import starter_configs
from eo_art.app.ui.common import configs_of_kind, rel, require_project

_NEW_KINDS = {
    "Render (starter)": "render.yaml",
    "Acquire (starter)": "acquire.yaml",
    "Empty layer": None,
}


def _version_key(path: Path) -> str:
    return f"cfgver_{path}"


def _bump(path: Path) -> None:
    """Re-key the widgets of ``path`` so they reload from disk after a save."""
    st.session_state[_version_key(path)] = (
        st.session_state.get(_version_key(path), 0) + 1
    )


def _new_config(project: Project) -> None:
    with st.popover("New config"), st.form("new_config", clear_on_submit=True):
        name = st.text_input("File name (without .yaml)", placeholder="dusk_look")
        kind = st.selectbox("Start from", list(_NEW_KINDS))
        if st.form_submit_button("Create"):
            try:
                target = project.configs_dir / f"{validate_name(name)}.yaml"
                if target.exists():
                    raise FileExistsError(f"{target.name} already exists")
            except (ValueError, FileExistsError) as exc:
                st.error(str(exc))
                return
            template = _NEW_KINDS[kind]
            text = (
                starter_configs(project.name)[template]
                if template
                else "# Layer merged on top of a render config.\n"
            )
            target.write_text(text)
            st.session_state["config_file"] = target.name
            st.rerun()


def _widget(spec: FieldSpec, value: Any, key: str, aoi_options: list[str]) -> Any:
    label = spec.label + (" *" if spec.required else "")
    if spec.optional:
        enabled = st.checkbox(
            f"{spec.label}: set", value=value is not None, key=f"{key}__on"
        )
        if value is None:
            value = spec.default
    if spec.path == "aoi_path" and aoi_options:
        options = list(dict.fromkeys([*([value] if value else []), *aoi_options]))
        # No preselection while unset, so an untouched form doesn't save
        # the first AOI as if it had been chosen.
        result = st.selectbox(label, options, index=0 if value else None, key=key)
    elif spec.kind == "bool":
        result = st.checkbox(label, value=bool(value), key=key)
    elif spec.kind == "enum":
        index = spec.choices.index(value) if value in spec.choices else 0
        result = st.selectbox(label, spec.choices, index=index, key=key)
    elif spec.kind == "int":
        result = int(st.number_input(label, value=int(value or 0), step=1, key=key))
    elif spec.kind == "float":
        result = st.number_input(
            label, value=float(value or 0.0), format="%g", step=None, key=key
        )
    else:
        text = st.text_input(label, value="" if value is None else str(value), key=key)
        # An untouched empty required field stays MISSING instead of "".
        result = None if (text == "" and value is None) else text
    if spec.optional and not enabled:
        return None
    return result


def _form_tab(project: Project, path: Path, kind: ConfigKind) -> None:
    schema = schema_for(kind)
    try:
        file_cfg = load_file(path)
        merged = merged_with_defaults(schema, file_cfg)
    except Exception as exc:
        st.error(f"Can't build the form, fix it in the YAML tab first:\n\n{exc}")
        return

    specs = [s for s in iter_fields(schema) if s.kind != "complex"]
    complex_paths = [s.path for s in iter_fields(schema) if s.kind == "complex"]
    initial = current_values(merged, specs)
    version = st.session_state.get(_version_key(path), 0)
    aoi_options = [rel(project, p) for p in project.aoi_files()]

    st.caption(
        "Shows schema defaults overlaid with this file. Saving writes only the "
        "fields you changed (YAML comments are dropped by a form save). "
        "Switching a group's `enabled` off hides the settings that depend on it. "
        f"Lists/dicts ({', '.join(complex_paths)}) are edited in the YAML tab."
    )
    edited: dict[str, Any] = {}

    def _fields(group_specs: list[FieldSpec]) -> None:
        # No st.form: an `enabled` toggle has to rerun to reveal its fields.
        # Hidden fields stay out of `edited`, so they're never saved.
        cols = st.columns(3)
        shown = 0
        for spec in group_specs:
            if not is_active(spec, edited, specs):
                continue
            with cols[shown % 3]:
                edited[spec.path] = _widget(
                    spec,
                    initial[spec.path],
                    f"{path}:{version}:{spec.path}",
                    aoi_options,
                )
            shown += 1

    with st.container(border=True):
        top_level = groupby(specs, key=lambda s: s.path.split(".")[0])
        for section, section_specs in top_level:
            section_specs = list(section_specs)
            if len(section_specs) == 1 and "." not in section_specs[0].path:
                spec = section_specs[0]
                edited[spec.path] = _widget(
                    spec,
                    initial[spec.path],
                    f"{path}:{version}:{spec.path}",
                    aoi_options,
                )
                continue
            with st.expander(section, expanded=section in {"run", "input", "sentinel"}):
                for group, group_specs in groupby(section_specs, key=lambda s: s.group):
                    group_specs = list(group_specs)
                    if not any(is_active(s, edited, specs) for s in group_specs):
                        continue
                    if group != section:
                        st.markdown(f"**{group.removeprefix(section + '.')}**")
                    _fields(group_specs)
        submitted = st.button("Save", type="primary", key=f"save_form_{path}")

    if submitted:
        changes = changed(initial, edited)
        if not changes:
            st.info("Nothing changed.")
            return
        updated = apply_changes(file_cfg, changes)
        try:
            merged_with_defaults(schema, updated)
        except Exception as exc:
            st.error(f"Not saved: {exc}")
            return
        OmegaConf.save(updated, path)
        _bump(path)
        st.success("Saved: " + ", ".join(sorted(changes)))
        st.rerun()


def _yaml_tab(path: Path) -> None:
    version = st.session_state.get(_version_key(path), 0)
    text = st.text_area(
        "YAML", path.read_text(), height=520, key=f"yaml_{path}_{version}"
    )
    if st.button("Save YAML", type="primary", key=f"save_yaml_{path}"):
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            st.error(f"Invalid YAML, not saved:\n\n{exc}")
            return
        path.write_text(text)
        _bump(path)
        st.success("Saved.")
        st.rerun()


def _show_validation(title: str, result) -> None:
    if result.ok:
        st.success(f"{title}: valid")
    else:
        st.error(f"{title}: {result.error}")
    if result.resolved:
        st.code(result.resolved, language="yaml")


def _resolved_tab(project: Project, path: Path, kind: ConfigKind) -> None:
    if kind is ConfigKind.acquire:
        left, right = st.columns(2)
        with left:
            _show_validation("Acquire", validate_acquire(path))
        with right:
            _show_validation("SR only", validate_sr(path))
        return
    others = [p for p in configs_of_kind(project, ConfigKind.render) if p != path]
    default = [p for p in others if p.name == "render.yaml"]
    base = st.multiselect(
        "Merge on top of (left to right)",
        others,
        default=default,
        format_func=lambda p: p.name,
        key=f"base_{path}",
    )
    _show_validation(
        " + ".join(p.name for p in [*base, path]), validate_render([*base, path])
    )


def render() -> None:
    project = require_project()
    st.title(f"Configs · {project.name}")

    files = project.config_files()
    head, new = st.columns([4, 1])
    with new:
        _new_config(project)
    if not files:
        st.info("No configs in this project yet.")
        return
    names = [p.name for p in files]
    with head:
        name = st.selectbox(
            "Config file",
            names,
            index=names.index(st.session_state["config_file"])
            if st.session_state.get("config_file") in names
            else 0,
        )
    st.session_state["config_file"] = name
    path = project.configs_dir / name

    detected = detect_kind(path)
    kind = st.radio(
        "Schema",
        list(ConfigKind),
        index=list(ConfigKind).index(detected),
        format_func=lambda k: k.value,
        horizontal=True,
        key=f"kind_{path}",
    )
    st.caption(f"`{rel(project, path)}`")

    form, raw, resolved = st.tabs(["Form", "YAML", "Resolved"])
    with form:
        _form_tab(project, path, kind)
    with raw:
        _yaml_tab(path)
    with resolved:
        _resolved_tab(project, path, kind)
