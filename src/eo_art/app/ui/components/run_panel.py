"""Start/stop a pipeline run, follow its log, browse earlier runs of a stage."""

import asyncio
import shlex
from typing import cast

import flet as ft

from eo_art.app.projects import Project
from eo_art.app.results import run_logs
from eo_art.app.runner import (
    PipelineKind,
    PipelineSpec,
    build_command,
    log_kind,
    log_path_for,
    start,
    stop,
    tail,
)
from eo_art.app.ui.layout import run_route
from eo_art.app.ui.state import RUNS
from eo_art.app.ui.widgets import (
    caption,
    card,
    code,
    dropdown,
    heading,
    notice,
    rel,
    use_app,
)

POLL_SECONDS = 1.0
LIVE_LINES = 400


@ft.component
def RunPanel(
    project: Project, kinds: tuple[PipelineKind, ...], spec: PipelineSpec | None
) -> ft.Control:
    """``spec`` is None while the form above is incomplete or invalid."""
    state = use_app()
    alive = ft.use_ref(False)
    _, set_tick = ft.use_state(0)
    picked_log, set_picked_log = ft.use_state(cast(str | None, None))
    root = str(project.root)

    async def follow() -> None:
        # Re-render while this stage's run is going, plus once when it ends.
        alive.current = True
        was_running = False
        while alive.current:
            handle = RUNS.get(root)
            running = handle is not None and handle.running
            if running or was_running:
                set_tick(lambda t: t + 1)
            was_running = running
            await asyncio.sleep(POLL_SECONDS)

    def stop_following() -> None:
        alive.current = False

    ft.use_effect(follow, [root], cleanup=stop_following)
    if state is None:
        return ft.Container()

    handle = RUNS.get(root)
    busy = handle is not None and handle.running
    handle_kind = log_kind(handle.log_path) if handle else None
    ours = handle is not None and handle_kind in kinds

    def start_run() -> None:
        if spec is None or busy:
            return
        log_path = log_path_for(project.logs_dir, spec.kind)
        # The run folder under out/runs/ takes the log's name.
        RUNS[root] = start(
            build_command(spec, run_id=log_path.stem), project.root, log_path
        )
        state.sync_run()
        set_tick(lambda t: t + 1)

    def stop_run() -> None:
        if handle is not None:
            stop(handle)
        state.sync_run()
        set_tick(lambda t: t + 1)

    controls: list[ft.Control] = [heading("Command")]
    if spec is None:
        controls.append(caption("Complete the settings above to get a command."))
    else:
        command = build_command(spec)
        controls.append(
            code(f"cd {shlex.quote(str(project.root))}\n{shlex.join(command)}")
        )
    controls.append(
        ft.Row(
            [
                ft.FilledButton(
                    "Start",
                    icon=ft.Icons.PLAY_ARROW,
                    on_click=start_run,
                    disabled=spec is None or busy,
                ),
                *(
                    [ft.OutlinedButton("Stop", icon=ft.Icons.STOP, on_click=stop_run)]
                    if busy and ours
                    else []
                ),
            ]
        )
    )
    if busy and not ours and handle_kind is not None:
        controls.append(
            ft.Row(
                [
                    notice(
                        f"A {handle_kind.value} run is active in this project; "
                        "one run per project at a time.",
                        "warn",
                    ),
                    ft.TextButton(
                        "Open it",
                        on_click=lambda: ft.context.page.navigate(
                            run_route(handle_kind)
                        ),
                    ),
                ]
            )
        )

    if ours and handle is not None:
        if handle.running:
            status = notice(
                f"Running since {handle.started:%H:%M:%S} · {handle.log_path.name}"
            )
        elif handle.returncode == 0:
            status = notice(f"Finished OK · {rel(project, handle.log_path)}", "ok")
        else:
            log = rel(project, handle.log_path)
            status = notice(f"Exited with code {handle.returncode} · {log}", "error")
        controls += [
            status,
            code(tail(handle.log_path, LIVE_LINES) or "(no output yet)", height=360),
        ]

    logs = run_logs(project, kinds)
    names = [p.name for p in logs]
    shown = picked_log if picked_log in names else (names[0] if names else None)
    history: list[ft.Control] = [heading("Earlier runs")]
    if shown is None:
        history.append(caption("No runs yet."))
    else:
        history += [
            dropdown("Log", [(n, n) for n in names], shown, set_picked_log, width=420),
            code(tail(project.logs_dir / shown, 1000), height=320),
        ]

    return ft.Column(
        [card(*controls), card(*history)],
        spacing=16,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
    )
