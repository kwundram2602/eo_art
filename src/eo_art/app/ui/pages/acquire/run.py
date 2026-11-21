"""Acquire › Run: full acquire (SR → DEM → DTM → align) or the SR step alone."""

from typing import cast

import flet as ft
from omegaconf import OmegaConf

from eo_art.app.configs import (
    ConfigKind,
    configs_of_kind,
    load_file,
    validate_acquire,
    validate_sr,
)
from eo_art.app.runner import ACQUIRE_KINDS, DESCRIPTIONS, PipelineKind, PipelineSpec
from eo_art.app.ui.components.run_panel import RunPanel
from eo_art.app.ui.widgets import (
    NeedsProject,
    caption,
    card,
    dropdown,
    notice,
    rel,
    use_app,
    validation,
)

FROM_CONFIG = "(from config)"


@ft.component
def RunPage() -> ft.Control:
    state = use_app()
    mode, set_mode = ft.use_state(cast(str, PipelineKind.acquire.name))
    picked, set_picked = ft.use_state(cast(str | None, None))
    aoi_choice, set_aoi_choice = ft.use_state(cast(str, FROM_CONFIG))
    no_cache, set_no_cache = ft.use_state(False)
    if state is None:
        return ft.Container()
    project = state.project
    if project is None:
        return NeedsProject()
    _ = state.revision

    kind = PipelineKind[mode]
    configs = configs_of_kind(project, ConfigKind.acquire)
    names = [p.name for p in configs]
    name = picked if picked in names else (names[0] if names else None)

    settings: list[ft.Control] = [
        ft.SegmentedButton(
            segments=[
                ft.Segment(value=k.name, label=ft.Text(k.value)) for k in ACQUIRE_KINDS
            ],
            selected=[mode],
            on_change=lambda e: set_mode(e.control.selected[0]),
        ),
        caption(DESCRIPTIONS[kind]),
    ]
    spec: PipelineSpec | None = None
    if name is None:
        settings.append(notice("No acquire config yet (create one on Config).", "warn"))
    else:
        path = project.configs_dir / name
        settings.append(
            dropdown(
                "Acquire config", [(n, n) for n in names], name, set_picked, width=320
            )
        )
        overrides: list[str] = []
        if kind is PipelineKind.sr:
            aois = [rel(project, p) for p in project.aoi_files()]
            settings.append(
                dropdown(
                    "AOI",
                    [(a, a) for a in [FROM_CONFIG, *aois]],
                    aoi_choice if aoi_choice in aois else FROM_CONFIG,
                    lambda v: set_aoi_choice(v or FROM_CONFIG),
                    width=320,
                    tooltip="Passed as --set aoi_path=...",
                )
            )
            if aoi_choice in aois:
                overrides.append(f"aoi_path={aoi_choice}")
        settings.append(
            ft.Checkbox(
                label="No cache (recompute everything)",
                value=no_cache,
                on_change=lambda e: set_no_cache(e.control.value),
            )
        )
        if kind is PipelineKind.sr:
            result = validate_sr(path)
            settings.append(validation(f"{name} (SR slice)", result))
        else:
            result = validate_acquire(path)
            settings.append(validation(name, result))
        try:
            timeseries = OmegaConf.select(
                load_file(path), "sentinel.timeseries.enabled"
            )
        except Exception:
            timeseries = False
        if kind is PipelineKind.acquire and timeseries:
            settings.append(
                notice(
                    "sentinel.timeseries is enabled, but acquire ignores it and uses "
                    "one reference scene. Run SR only for the timeseries."
                )
            )
        if result.ok:
            spec = PipelineSpec(
                kind,
                acquire_config=rel(project, path),
                overrides=overrides,
                no_cache=no_cache,
            )

    return ft.Column(
        [card(*settings), RunPanel(project, ACQUIRE_KINDS, spec)],
        spacing=16,
        scroll=ft.ScrollMode.AUTO,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        expand=True,
    )
