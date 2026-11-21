"""Pick, create and edit one kind of config: generated form, raw YAML, and the
resolved (validated) view. Shared by Acquire › Config and Forge3D › Configs."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import flet as ft
import yaml

from eo_art.app.config_forms import (
    CHAIN_KEY,
    LIST_KEYS,
    OVERLAYS_KEY,
    FieldSpec,
    GroupNode,
    changed,
    check_lists,
    clean_lists,
    current_values,
    group_tree,
    is_active,
    iter_fields,
    list_values,
    merged_with_defaults,
    save_edits,
    set_in_file,
)
from eo_art.app.configs import (
    ConfigKind,
    configs_of_kind,
    load_file,
    schema_for,
    validate_acquire,
    validate_render,
    validate_sr,
)
from eo_art.app.field_help import ACQUIRE_HELP, RENDER_HELP
from eo_art.app.overlay_preview import OverlayPreview, build_previews
from eo_art.app.projects import Project, validate_name
from eo_art.app.results import project_tifs
from eo_art.app.templates import starter_configs
from eo_art.app.ui.components.fields import field, number_value
from eo_art.app.ui.components.list_editors import (
    Items,
    chain_editor,
    overlays_editor,
)
from eo_art.app.ui.widgets import (
    MONO,
    Expandable,
    NeedsProject,
    caption,
    card,
    dropdown,
    heading,
    keyed,
    notice,
    rel,
    save_bytes_as,
    toast,
    use_app,
    validation,
)

GENERAL = "general"
# Section order and titles; unlisted sections follow in schema order.
SECTIONS = {
    GENERAL: "General",
    "sentinel": "Sentinel-2 / super-resolution",
    "dem": "DEM",
    "dtm": "DTM super-resolution",
    "align": "Align",
    "run": "Run",
    "input": "Input",
    "render": "Render",
    "animation": "Animation",
    "export": "Export",
    OVERLAYS_KEY: "Overlays",
    CHAIN_KEY: "DEM prepare",
}
# Groups open when the form opens; everything else starts collapsed.
EXPANDED = {
    GENERAL,
    "run",
    "input",
    "render",
    "render.camera",
    "render.sun",
    "sentinel",
}
EMPTY_LAYER = "# Layer merged on top of a render config.\n"


def _initial_lists(file_cfg: Any, keys: tuple[str, ...]) -> dict[str, list]:
    """The file's lists, normalised like a save would write them (so a stray
    ``opacity: null`` doesn't count as an unsaved change); as they are if a
    value can't be parsed."""
    raw = list_values(file_cfg, keys)
    try:
        return clean_lists(raw)
    except ValueError:
        return raw


def _overlay_summary(overlay: dict[str, Any]) -> str:
    chain = " → ".join(str(op.get("op")) for op in overlay.get(CHAIN_KEY) or [])
    return f"{overlay.get('name')} ({overlay.get('path')})" + (
        f": {chain}" if chain else ""
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _badge(fields: list[FieldSpec], in_file: set[str], pending: set[str]) -> ft.Control:
    """Header subtitle: field count, how many the file sets, how many are unsaved."""
    paths = {s.path for s in fields}
    parts = [_plural(len(paths), "field")]
    if n_set := len(paths & in_file):
        parts.append(f"{n_set} set in file")
    if n_unsaved := len(paths & pending):
        parts.append(f"{n_unsaved} unsaved")
    return ft.Text(
        " · ".join(parts),
        size=12,
        color=ft.Colors.PRIMARY if n_unsaved else ft.Colors.ON_SURFACE_VARIANT,
    )


@ft.component
def ConfigForm(
    project: Project,
    path: Path,
    kind: ConfigKind,
    on_saved: Any,
    on_open_overlays: Any = None,
) -> ft.Control:
    schema = schema_for(kind)
    help_texts = RENDER_HELP if kind is ConfigKind.render else ACQUIRE_HELP
    try:
        file_cfg = load_file(path)
        merged = merged_with_defaults(schema, file_cfg)
        error = None
    except Exception as exc:
        file_cfg, merged, error = None, None, exc
    all_specs = iter_fields(schema)
    specs = [s for s in all_specs if s.kind != "complex"]
    initial = ft.use_memo(
        lambda: current_values(merged, specs) if merged is not None else {}, []
    )
    in_file = ft.use_memo(
        lambda: set_in_file(file_cfg, specs) if file_cfg is not None else set(), []
    )
    values, set_values = ft.use_state(lambda: dict(initial))
    # The render config's overlays and DEM prepare chain, as in the file.
    # The render config's DEM prepare chain (overlays: Overlay view).
    lists_initial = ft.use_memo(
        lambda: (
            _initial_lists(file_cfg, (CHAIN_KEY,))
            if file_cfg is not None and kind is ConfigKind.render
            else {}
        ),
        [],
    )
    file_overlays = ft.use_memo(
        lambda: (
            list_values(file_cfg, (OVERLAYS_KEY,))[OVERLAYS_KEY]
            if file_cfg is not None and kind is ConfigKind.render
            else []
        ),
        [],
    )
    lists, set_lists = ft.use_state(lambda: dict(lists_initial))
    # (epoch, open): Expand/Collapse all bumps epoch, which remounts every
    # group tile open or closed; open=None means the EXPANDED defaults.
    fold, set_fold = ft.use_state(cast(tuple[int, bool | None], (0, None)))
    if error is not None:
        return notice(
            f"Can't build the form, fix it in the YAML view first: {error}", "error"
        )

    def setter(field_path: str):
        def set_value(value: Any) -> None:
            set_values(lambda prev: {**prev, field_path: value})

        return set_value

    def list_update(key: str):
        def update(change: Any) -> None:
            set_lists(lambda prev: {**prev, key: change(list(prev[key]))})

        return update

    def cleaned_lists() -> dict[str, list]:
        """The lists as they'd be saved; raises ValueError on a bad number."""
        return clean_lists(lists) if lists else {}

    def edited_values() -> dict[str, Any]:
        """Shown fields as they'd be saved. Hidden (gated-off) fields stay
        out, so they're never saved."""
        return {
            s.path: number_value(s, values[s.path])
            if s.kind in ("int", "float")
            else values[s.path]
            for s in specs
            if is_active(s, values, specs)
        }

    def save() -> None:
        edited = edited_values()
        bad = [
            s.path
            for s in specs
            if s.kind in ("int", "float") and isinstance(edited.get(s.path), str)
        ]
        if bad:
            toast("Not saved, not a number: " + ", ".join(bad), error=True)
            return
        try:
            cleaned = cleaned_lists()
            check_lists(cleaned)
        except Exception as exc:
            toast(f"Not saved: {exc}", error=True)
            return
        try:
            changes = save_edits(
                path, schema, {**initial, **lists_initial}, {**edited, **cleaned}
            )
        except Exception as exc:
            toast(f"Not saved: {exc}", error=True)
            return
        if not changes:
            toast("Nothing changed.")
            return
        toast("Saved: " + ", ".join(sorted(changes)))
        on_saved()

    aoi_options = [rel(project, p) for p in project.aoi_files()]
    pending = set(changed(initial, edited_values()))
    try:
        pending |= set(changed(lists_initial, cleaned_lists()))
    except ValueError:
        # A half-typed number: count every list that differs as it stands.
        pending |= {k for k in lists if lists[k] != lists_initial.get(k)}
    epoch, fold_open = fold

    def tile(node: GroupNode, depth: int, name: str | None = None) -> ft.Control:
        if name is None:
            name = (
                SECTIONS.get(node.path or GENERAL, node.path)
                if depth == 0
                else node.path.rpartition(".")[2]
            )
        # A group that only wraps one subgroup (export › video) is shown as one.
        if not node.fields and node.gate is None and len(node.children) == 1:
            child = node.children[0]
            return tile(child, depth, f"{name} · {child.path.rpartition('.')[2]}")

        gate = node.gate
        on = gate is None or bool(values.get(gate.path))
        body: list[ft.Control] = []
        if not on:
            body.append(caption("Off. Switch it on to edit these settings."))
        else:
            visible = [s for s in node.fields if is_active(s, values, specs)]
            if visible:
                body.append(
                    ft.ResponsiveRow(
                        [
                            field(
                                s,
                                values[s.path],
                                setter(s.path),
                                help_texts.get(s.path),
                                options=aoi_options if s.path == "aoi_path" else None,
                            )
                            for s in visible
                        ],
                        run_spacing=8,
                    )
                )
            body.extend(tile(c, depth + 1) for c in node.children)

        header: ft.Control = ft.Text(
            name, weight=ft.FontWeight.BOLD if depth == 0 else ft.FontWeight.W_500
        )
        if gate is not None:
            set_gate = setter(gate.path)
            header = ft.Row(
                [
                    header,
                    ft.Switch(
                        value=on,
                        tooltip=help_texts.get(gate.path) or gate.path,
                        on_change=lambda e: set_gate(e.control.value),
                    ),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        group = node.path or GENERAL
        nested = depth > 0
        return keyed(
            Expandable,
            f"group:{group}:{epoch}",
            header,
            [
                ft.Container(
                    ft.Column(body, spacing=8),
                    padding=ft.Padding.only(
                        # top: room for the floating labels of the first row.
                        left=16,
                        top=8,
                        right=16 if not nested else 0,
                        bottom=8,
                    ),
                    # A line down the left edge shows which group a field is in.
                    border=ft.Border.only(
                        left=ft.BorderSide(2, ft.Colors.OUTLINE_VARIANT)
                    )
                    if nested
                    else None,
                )
            ],
            group in EXPANDED if fold_open is None else fold_open,
            subtitle=_badge(node.all_fields(), in_file, pending),
            dense=nested,
            tile_padding=ft.Padding.only(left=0, right=8) if nested else None,
        )

    def list_tile(key: str, body: ft.Control, count: str) -> ft.Control:
        unsaved = key in pending
        return keyed(
            Expandable,
            f"group:{key}:{epoch}",
            ft.Text(SECTIONS[key], weight=ft.FontWeight.BOLD),
            [
                ft.Container(
                    body, padding=ft.Padding.only(left=16, top=8, right=16, bottom=8)
                )
            ],
            key in EXPANDED if fold_open is None else fold_open,
            subtitle=ft.Text(
                count + (" · unsaved" if unsaved else ""),
                size=12,
                color=ft.Colors.PRIMARY if unsaved else ft.Colors.ON_SURFACE_VARIANT,
            ),
        )

    tiles = [tile(node, 0) for node in group_tree(specs)]
    if lists:
        chain: Items = lists[CHAIN_KEY]
        tiles.append(
            list_tile(
                CHAIN_KEY,
                chain_editor(chain, list_update(CHAIN_KEY)),
                _plural(len(chain), "op") if chain else "none",
            )
        )
        # Second place: overlays at a glance, edited in the Overlay view.
        tiles.insert(
            1,
            ft.Container(
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text(
                                    SECTIONS[OVERLAYS_KEY], weight=ft.FontWeight.BOLD
                                ),
                                caption(
                                    "; ".join(map(_overlay_summary, file_overlays))
                                    or "No overlays."
                                ),
                            ],
                            spacing=2,
                            expand=True,
                        ),
                        ft.TextButton(
                            "Edit in Overlay view",
                            icon=ft.Icons.LAYERS,
                            on_click=on_open_overlays,
                            disabled=on_open_overlays is None,
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=ft.Padding.symmetric(horizontal=16, vertical=8),
            ),
        )

    complex_paths = [
        s.path for s in all_specs if s.kind == "complex" and s.path not in LIST_KEYS
    ]
    save_bar = ft.Container(
        ft.Row(
            [
                ft.FilledButton("Save", icon=ft.Icons.SAVE, on_click=save),
                ft.Text(
                    _plural(len(pending), "unsaved change")
                    if pending
                    else "No unsaved changes",
                    color=ft.Colors.PRIMARY
                    if pending
                    else ft.Colors.ON_SURFACE_VARIANT,
                ),
                ft.Container(expand=True),
                ft.TextButton(
                    "Expand all",
                    icon=ft.Icons.UNFOLD_MORE,
                    on_click=lambda: set_fold((epoch + 1, True)),
                ),
                ft.TextButton(
                    "Collapse all",
                    icon=ft.Icons.UNFOLD_LESS,
                    on_click=lambda: set_fold((epoch + 1, False)),
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        padding=ft.Padding.only(top=8),
        border=ft.Border.only(top=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)),
    )
    return ft.Column(
        [
            caption(
                "Schema defaults overlaid with this file. Saving writes only the "
                "fields you changed (a form save drops YAML comments). A group's "
                "switch turns it on or off; the settings of a group that is off "
                "are not saved. "
                + (
                    f"Lists/dicts ({', '.join(complex_paths)}) are edited in the "
                    "YAML view."
                    if complex_paths
                    else ""
                )
            ),
            ft.Column(
                tiles,
                spacing=4,
                scroll=ft.ScrollMode.AUTO,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                expand=True,
            ),
            save_bar,
        ],
        spacing=8,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )


@ft.component
def YamlEditor(path: Path, on_saved: Any) -> ft.Control:
    text, set_text = ft.use_state(lambda: path.read_text())

    def save() -> None:
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            toast(f"Invalid YAML, not saved: {exc}", error=True)
            return
        path.write_text(text)
        toast("Saved.")
        on_saved()

    return ft.Column(
        [
            ft.TextField(
                value=text,
                multiline=True,
                min_lines=24,
                text_style=ft.TextStyle(font_family=MONO, size=13),
                on_change=lambda e: set_text(e.control.value),
            ),
            ft.Row([ft.FilledButton("Save YAML", icon=ft.Icons.SAVE, on_click=save)]),
        ]
    )


def _use_layers(
    project: Project, path: Path
) -> tuple[list[Path], list[Path], ft.Control]:
    """Hook: the render configs ticked to merge under ``path`` (render.yaml
    by default), the resulting layer list, and the checkboxes to pick them."""
    others = [p for p in configs_of_kind(project, ConfigKind.render) if p != path]
    base, set_base = ft.use_state(
        lambda: [p.name for p in others if p.name == "render.yaml"]
    )

    def toggle(name: str, on: bool) -> None:
        set_base(lambda prev: [*prev, name] if on else [n for n in prev if n != name])

    picker = ft.Column(
        [
            ft.Text("Merge on top of (in the order ticked):"),
            ft.Row(
                [
                    ft.Checkbox(
                        label=p.name,
                        value=p.name in base,
                        on_change=lambda e, n=p.name: toggle(n, e.control.value),
                    )
                    for p in others
                ]
                or [caption("No other render configs.")],
                wrap=True,
            ),
        ]
    )
    return others, [project.configs_dir / n for n in base] + [path], picker


@ft.component
def Resolved(project: Project, path: Path, kind: ConfigKind) -> ft.Control:
    _, layers, picker = _use_layers(project, path)
    if kind is ConfigKind.acquire:
        return ft.Column(
            [
                validation("Acquire", validate_acquire(path), expanded=True),
                validation("SR only", validate_sr(path), expanded=True),
            ]
        )
    return ft.Column(
        [
            picker,
            validation(
                " + ".join(p.name for p in layers),
                validate_render(layers),
                expanded=True,
            ),
        ]
    )


def _saver(preview: OverlayPreview) -> Callable[[], Any]:
    return lambda: save_bytes_as(
        preview.png_path.read_bytes(), f"{preview.name}_overlay.png", "png"
    )


def _preview_card(preview: OverlayPreview) -> ft.Control:
    chain = " → ".join(preview.prepare) or "none"
    opacity = 1.0 if preview.opacity is None else preview.opacity

    def image(label: str, data: bytes) -> ft.Control:
        return ft.Column(
            [
                caption(label),
                ft.Image(
                    src=data, fit=ft.BoxFit.CONTAIN, height=420, gapless_playback=True
                ),
            ],
            col={"sm": 12, "md": 6},
        )

    return card(
        ft.Row(
            [heading(preview.name), caption(preview.source)],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        caption(f"prepare: {chain} · opacity {opacity:g}"),
        ft.ResponsiveRow(
            [
                image("Overlay as the renderer gets it", preview.overlay_png),
                image("On a hillshade of the DEM (render.sun)", preview.relief_png),
            ]
        ),
        ft.Row(
            [
                ft.TextButton(
                    "Save PNG", icon=ft.Icons.SAVE_ALT, on_click=_saver(preview)
                )
            ]
        ),
    )


def _file_overlays(path: Path) -> list[dict[str, Any]] | str:
    """The file's own overlays (normalised), or why they can't be read."""
    try:
        return _initial_lists(load_file(path), (OVERLAYS_KEY,))[OVERLAYS_KEY]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@ft.component
def OverlayPane(project: Project, path: Path, on_saved: Any) -> ft.Control:
    """Edit this file's overlays (and their prepare chains) next to a preview
    of each overlay as a run would prepare it (same prep chain and cache, no
    GPU), plain and on a hillshade of the prepared DEM. Saving remounts the
    pane, so the preview is rebuilt from the saved file."""
    _, layers, picker = _use_layers(project, path)
    initial = ft.use_memo(lambda: _file_overlays(path), [])
    overlays, set_overlays = ft.use_state(
        lambda: list(initial) if isinstance(initial, list) else []
    )
    result, set_result = ft.use_state(cast(list[OverlayPreview] | str | None, None))
    busy, set_busy = ft.use_state(cast(bool, False))
    refresh, set_refresh = ft.use_state(0)
    request = ft.use_ref(0)

    async def build() -> None:
        # A newer build (other layers ticked, Refresh) wins over a slow one.
        request.current = (request.current or 0) + 1
        mine = request.current
        set_busy(True)
        try:
            outcome: list[OverlayPreview] | str = await asyncio.to_thread(
                build_previews, project.root, layers
            )
        except Exception as exc:
            outcome = f"{type(exc).__name__}: {exc}"
        if mine == request.current:
            set_result(outcome)
            set_busy(False)

    ft.use_effect(build, [tuple(str(p) for p in layers), refresh])

    if isinstance(initial, str):
        return notice(
            f"Can't read the overlays, fix the file in the YAML view first: {initial}",
            "error",
        )

    def update(change: Callable[[Items], Items]) -> None:
        set_overlays(lambda prev: change(list(prev)))

    try:
        unsaved = clean_lists({OVERLAYS_KEY: overlays})[OVERLAYS_KEY] != initial
    except ValueError:
        unsaved = True

    def save() -> None:
        try:
            cleaned = clean_lists({OVERLAYS_KEY: overlays})
            check_lists(cleaned)
            changes = save_edits(
                path, schema_for(ConfigKind.render), {OVERLAYS_KEY: initial}, cleaned
            )
        except Exception as exc:
            toast(f"Not saved: {exc}", error=True)
            return
        if not changes:
            toast("Nothing changed.")
            return
        toast("Saved overlays.")
        on_saved()

    rasters = [rel(project, p) for p in project_tifs(project)]
    editor = ft.Column(
        [
            ft.Row(
                [
                    heading(f"Overlays in {path.name}"),
                    ft.Container(expand=True),
                    ft.Text(
                        "unsaved changes" if unsaved else "saved",
                        size=12,
                        color=ft.Colors.PRIMARY
                        if unsaved
                        else ft.Colors.ON_SURFACE_VARIANT,
                    ),
                    ft.FilledButton("Save", icon=ft.Icons.SAVE, on_click=save),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            overlays_editor(overlays, update, rasters),
        ],
        spacing=12,
        col={"sm": 12, "lg": 6},
    )

    if isinstance(result, str):
        previews: list[ft.Control] = [notice(result, "error")]
    elif result == []:
        previews = [caption("No overlays to preview.")]
    else:
        previews = [_preview_card(p) for p in result or []]
    preview = ft.Column(
        [
            ft.Row(
                [
                    heading("Preview"),
                    ft.Container(expand=True),
                    *([ft.ProgressRing(width=18, height=18)] if busy else []),
                    ft.OutlinedButton(
                        "Refresh",
                        icon=ft.Icons.REFRESH,
                        disabled=busy,
                        on_click=lambda: set_refresh(lambda r: r + 1),
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            caption(
                "Built from the saved files (with the layers ticked above); "
                "saving the overlays rebuilds it."
            ),
            *previews,
        ],
        spacing=12,
        col={"sm": 12, "lg": 6},
    )
    return ft.Column(
        [picker, ft.ResponsiveRow([editor, preview], spacing=24, run_spacing=24)],
        spacing=12,
    )


@ft.component
def ConfigEditor(kind: ConfigKind, templates: dict[str, str | None]) -> ft.Control:
    """``templates``: label -> starter config filename (None = empty layer)."""
    state = use_app()
    selected, set_selected = ft.use_state(cast(str | None, None))
    version, set_version = ft.use_state(0)
    view, set_view = ft.use_state(cast(str, "form"))
    creating, set_creating = ft.use_state(cast(bool, False))
    new_name, set_new_name = ft.use_state("")
    new_template, set_new_template = ft.use_state(next(iter(templates)))
    project = state.project if state else None

    def create() -> None:
        assert project is not None
        try:
            target = project.configs_dir / f"{validate_name(new_name)}.yaml"
            if target.exists():
                raise FileExistsError(f"{target.name} already exists")
        except (ValueError, FileExistsError) as exc:
            toast(str(exc), error=True)
            return
        template = templates[new_template]
        target.write_text(
            starter_configs(project.name)[template] if template else EMPTY_LAYER
        )
        set_creating(False)
        set_new_name("")
        set_selected(target.name)
        toast(f"Created {target.name}")

    ft.use_dialog(
        ft.AlertDialog(
            title=ft.Text("New config"),
            content=ft.Column(
                [
                    ft.TextField(
                        label="File name (without .yaml)",
                        hint_text="e.g. dusk_look",
                        value=new_name,
                        on_change=lambda e: set_new_name(e.control.value),
                        autofocus=True,
                    ),
                    dropdown(
                        "Start from",
                        [(k, k) for k in templates],
                        new_template,
                        lambda v: set_new_template(v or next(iter(templates))),
                    ),
                ],
                tight=True,
                width=380,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda: set_creating(False)),
                ft.FilledButton(
                    "Create", on_click=create, disabled=not new_name.strip()
                ),
            ],
        )
        if creating and project is not None
        else None
    )
    if state is None:
        return ft.Container()
    if project is None:
        return NeedsProject()
    _ = state.revision

    files = configs_of_kind(project, kind)
    names = [p.name for p in files]
    name = selected if selected in names else (names[0] if names else None)
    head = ft.Row(
        [
            dropdown(
                "Config file",
                [(n, n) for n in names],
                name,
                set_selected,
                hint_text="No configs yet" if not names else None,
                width=320,
            ),
            ft.OutlinedButton(
                "New config", icon=ft.Icons.ADD, on_click=lambda: set_creating(True)
            ),
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    if name is None:
        return ft.Column([head, caption("No configs of this kind in the project yet.")])

    path = project.configs_dir / name
    bump = lambda: set_version(lambda v: v + 1)  # noqa: E731
    key = f"{path}:{version}"
    if view == "form":
        body: ft.Control = keyed(
            ConfigForm,
            key,
            project,
            path,
            kind,
            bump,
            (lambda: set_view("overlay")) if kind is ConfigKind.render else None,
        )
    elif view == "yaml":
        body = keyed(YamlEditor, key, path, bump)
    elif view == "overlay" and kind is ConfigKind.render:
        body = keyed(OverlayPane, key, project, path, bump)
    else:
        body = keyed(Resolved, key, project, path, kind)

    return ft.Column(
        [
            head,
            ft.Row(
                [
                    ft.SegmentedButton(
                        segments=[
                            ft.Segment(
                                value="form",
                                label=ft.Text("Form"),
                                icon=ft.Icon(ft.Icons.TUNE),
                            ),
                            *(
                                [
                                    ft.Segment(
                                        value="overlay",
                                        label=ft.Text("Overlay"),
                                        icon=ft.Icon(ft.Icons.LAYERS),
                                    )
                                ]
                                if kind is ConfigKind.render
                                else []
                            ),
                            ft.Segment(
                                value="yaml",
                                label=ft.Text("YAML"),
                                icon=ft.Icon(ft.Icons.CODE),
                            ),
                            ft.Segment(
                                value="resolved",
                                label=ft.Text("Resolved"),
                                icon=ft.Icon(ft.Icons.FACT_CHECK),
                            ),
                        ],
                        selected=[view],
                        on_change=lambda e: set_view(e.control.selected[0]),
                    ),
                    caption(rel(project, path)),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            body,
        ],
        spacing=12,
        # The form scrolls itself so its save bar stays pinned at the bottom.
        scroll=None if view == "form" else ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )
