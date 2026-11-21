"""Forge3D › Run: render merged config layers, optionally acquiring first."""

from typing import cast

import flet as ft

from eo_art.app.configs import (
    ConfigKind,
    configs_of_kind,
    overlay_names,
    validate_acquire,
    validate_render,
)
from eo_art.app.runner import DESCRIPTIONS, FORGE3D_KINDS, PipelineKind, PipelineSpec
from eo_art.app.templates import RENDER_FILENAME
from eo_art.app.ui.components.run_panel import RunPanel
from eo_art.app.ui.widgets import (
    NeedsProject,
    caption,
    card,
    dropdown,
    heading,
    notice,
    rel,
    use_app,
    validation,
)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


@ft.component
def RunPage() -> ft.Control:
    state = use_app()
    layers, set_layers = ft.use_state(cast(list[str] | None, None))
    overrides, set_overrides = ft.use_state("")
    sweep, set_sweep = ft.use_state("")
    fail_fast, set_fail_fast = ft.use_state(False)
    no_cache, set_no_cache = ft.use_state(False)
    acquire_first, set_acquire_first = ft.use_state(False)
    acquire_pick, set_acquire_pick = ft.use_state(cast(str | None, None))
    overlay_pick, set_overlay_pick = ft.use_state(cast(str | None, None))
    if state is None:
        return ft.Container()
    project = state.project
    if project is None:
        return NeedsProject()
    _ = state.revision

    render_names = [p.name for p in configs_of_kind(project, ConfigKind.render)]
    default = [n for n in render_names if n == RENDER_FILENAME] or render_names[:1]
    order = [n for n in (default if layers is None else layers) if n in render_names]

    def toggle(name: str, on: bool) -> None:
        set_layers([*order, name] if on else [n for n in order if n != name])

    def move(i: int, step: int) -> None:
        j = i + step
        if 0 <= j < len(order):
            moved = list(order)
            moved[i], moved[j] = moved[j], moved[i]
            set_layers(moved)

    layer_rows: list[ft.Control] = [
        ft.Row(
            [
                ft.Text(f"{i + 1}.", width=24),
                ft.Text(name, expand=True),
                ft.IconButton(
                    ft.Icons.ARROW_UPWARD,
                    disabled=i == 0,
                    on_click=lambda _, i=i: move(i, -1),
                    tooltip="Merge earlier",
                ),
                ft.IconButton(
                    ft.Icons.ARROW_DOWNWARD,
                    disabled=i == len(order) - 1,
                    on_click=lambda _, i=i: move(i, 1),
                    tooltip="Merge later",
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        for i, name in enumerate(order)
    ]
    layer_card = card(
        heading("Render configs"),
        caption("Merged top to bottom; later layers override earlier ones."),
        ft.Row(
            [
                ft.Checkbox(
                    label=n,
                    value=n in order,
                    on_change=lambda e, n=n: toggle(n, e.control.value),
                )
                for n in render_names
            ]
            or [notice("No render config yet (create one on Configs).", "warn")],
            wrap=True,
        ),
        *layer_rows,
    )

    render_overrides = _lines(overrides)
    options_card = card(
        heading("Options"),
        ft.TextField(
            label="Overrides (one PATH=VALUE per line)",
            hint_text="render.width=1280\nrender.backend=offscreen",
            value=overrides,
            multiline=True,
            min_lines=2,
            on_change=lambda e: set_overrides(e.control.value),
        ),
        ft.TextField(
            label="Sweep (PATH=V1,V2,...)",
            hint_text="render.pbr.exposure=1.0,1.8",
            value=sweep,
            on_change=lambda e: set_sweep(e.control.value),
        ),
        ft.Row(
            [
                ft.Checkbox(
                    label="Fail fast",
                    value=fail_fast,
                    on_change=lambda e: set_fail_fast(e.control.value),
                ),
                ft.Checkbox(
                    label="No cache (recompute everything)",
                    value=no_cache,
                    on_change=lambda e: set_no_cache(e.control.value),
                ),
            ],
            wrap=True,
        ),
    )

    layer_paths = [project.configs_dir / n for n in order]
    kind = PipelineKind.acquire_render if acquire_first else PipelineKind.render
    acquire_controls: list[ft.Control] = [
        ft.Switch(
            label="Run acquire first",
            value=acquire_first,
            on_change=lambda e: set_acquire_first(e.control.value),
        ),
        caption(DESCRIPTIONS[kind]),
    ]
    valid = bool(layer_paths)
    acquire_path = None
    overlay = None
    if acquire_first:
        acquire_names = [p.name for p in configs_of_kind(project, ConfigKind.acquire)]
        acquire_name = (
            acquire_pick
            if acquire_pick in acquire_names
            else (acquire_names[0] if acquire_names else None)
        )
        if acquire_name is None:
            acquire_controls.append(
                notice("No acquire config yet (create one under Acquire).", "warn")
            )
            valid = False
        else:
            acquire_path = project.configs_dir / acquire_name
            acquire_controls.append(
                dropdown(
                    "Acquire config",
                    [(n, n) for n in acquire_names],
                    acquire_name,
                    set_acquire_pick,
                    width=320,
                )
            )
            overlays = overlay_names(layer_paths) if layer_paths else []
            if overlays:
                overlay = overlay_pick if overlay_pick in overlays else overlays[0]
                acquire_controls.append(
                    dropdown(
                        "Overlay to receive the optical raster",
                        [(o, o) for o in overlays],
                        overlay,
                        set_overlay_pick,
                        width=320,
                    )
                )
            result = validate_acquire(acquire_path)
            acquire_controls.append(validation(acquire_name, result))
            valid &= result.ok

    checks: list[ft.Control] = []
    if layer_paths:
        result = validate_render(layer_paths, render_overrides)
        checks.append(validation(" + ".join(order), result))
        valid &= result.ok

    spec = None
    if valid:
        try:
            spec = PipelineSpec(
                kind,
                acquire_config=rel(project, acquire_path) if acquire_path else None,
                render_layers=[rel(project, p) for p in layer_paths],
                overrides=render_overrides,
                sweep=sweep.strip() or None,
                no_cache=no_cache,
                fail_fast=fail_fast,
                acquire_overlay=overlay,
            )
        except ValueError as exc:
            checks.append(notice(str(exc), "warn"))

    return ft.Column(
        [
            ft.ResponsiveRow(
                [
                    ft.Container(layer_card, col={"lg": 6}),
                    ft.Container(options_card, col={"lg": 6}),
                ]
            ),
            card(*acquire_controls),
            *checks,
            RunPanel(project, FORGE3D_KINDS, spec),
        ],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )
