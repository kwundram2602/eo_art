"""Build and launch the existing pipeline CLIs as subprocesses.

The app never calls the pipelines in-process: a run must outlive the page that
started it, be stoppable, and keep the forge3d viewer out of the UI process.
"""

import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class PipelineKind(Enum):
    acquire = "Acquire"
    sr = "SR only"
    render = "Render"
    acquire_render = "Acquire → Render"


# Which runs belong to which stage of the app.
ACQUIRE_KINDS = (PipelineKind.acquire, PipelineKind.sr)
FORGE3D_KINDS = (PipelineKind.render, PipelineKind.acquire_render)

DESCRIPTIONS = {
    PipelineKind.acquire: "Sentinel SR → DEM → DTM → align. Writes dtm.tif and "
    "optical_aligned.tif (with sentinel.zones: one tile per zone) into its own "
    "run folder out/runs/<time>-acquire/; out/runs/latest-acquire points at "
    "the newest one.",
    PipelineKind.sr: "Only the Sentinel-2 super-resolution, into its own run "
    "folder out/runs/<time>-sr/: <aoi-name>_sr.tif, or with "
    "sentinel.timeseries.enabled one GeoTIFF per interval in "
    "<aoi-name>_sr_timeseries/.",
    PipelineKind.render: "forge3d render of the merged render config(s).",
    PipelineKind.acquire_render: "Acquire first, then render with its DTM as "
    "input.path and its optical raster as the overlay's path.",
}

F3D_SCRIPT = "eo-art-f3d"
SR_SCRIPT = "eo-art-sr"


@dataclass(frozen=True)
class PipelineSpec:
    kind: PipelineKind
    acquire_config: str | None = None
    render_layers: Sequence[str] = ()
    overrides: Sequence[str] = ()
    sweep: str | None = None
    no_cache: bool = False
    fail_fast: bool = False
    acquire_overlay: str | None = None

    def __post_init__(self) -> None:
        needs_acquire = self.kind is not PipelineKind.render
        needs_render = self.kind in (PipelineKind.render, PipelineKind.acquire_render)
        if needs_acquire and not self.acquire_config:
            raise ValueError(f"{self.kind.value} needs an acquire config")
        if needs_render and not self.render_layers:
            raise ValueError(f"{self.kind.value} needs at least one render config")


def _script(bin_dir: Path, name: str) -> str:
    return str(bin_dir / name)


def build_command(
    spec: PipelineSpec, bin_dir: Path | None = None, run_id: str | None = None
) -> list[str]:
    """The exact CLI invocation for ``spec``; flags mirror the CLIs 1:1.

    ``run_id`` names the acquire/SR run folder (``out/runs/<run_id>/``); the
    app passes its log's name, so a run's folder and log share one id.
    """
    bin_dir = Path(sys.executable).parent if bin_dir is None else bin_dir
    sets = [arg for override in spec.overrides for arg in ("--set", override)]
    run_flag = ["--run-id", run_id] if run_id else []

    if spec.kind is PipelineKind.sr:
        assert spec.acquire_config is not None
        cmd = [_script(bin_dir, SR_SCRIPT), spec.acquire_config, *sets]
        return cmd + (["--no-cache"] if spec.no_cache else []) + run_flag

    if spec.kind is PipelineKind.acquire:
        assert spec.acquire_config is not None
        cmd = [_script(bin_dir, F3D_SCRIPT), "acquire", spec.acquire_config]
        return cmd + (["--no-cache"] if spec.no_cache else []) + run_flag

    cmd = [_script(bin_dir, F3D_SCRIPT), "run", *spec.render_layers, *sets]
    if spec.sweep:
        cmd += ["--sweep", spec.sweep]
    if spec.no_cache:
        cmd.append("--no-cache")
    if spec.fail_fast:
        cmd.append("--fail-fast")
    if spec.kind is PipelineKind.acquire_render:
        assert spec.acquire_config is not None
        cmd += ["--acquire", spec.acquire_config]
        if spec.acquire_overlay:
            cmd += ["--acquire-overlay", spec.acquire_overlay]
        if spec.no_cache:
            cmd.append("--acquire-no-cache")
        if run_id:
            cmd += ["--acquire-run-id", run_id]
    return cmd


@dataclass
class RunHandle:
    process: subprocess.Popen
    log_path: Path
    command: list[str]
    started: datetime = field(default_factory=datetime.now)

    @property
    def returncode(self) -> int | None:
        return self.process.poll()

    @property
    def running(self) -> bool:
        return self.returncode is None


def log_path_for(logs_dir: Path, kind: PipelineKind) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return logs_dir / f"{stamp}-{kind.name}.log"


def log_kind(path: Path) -> PipelineKind | None:
    """The run kind a log was written for (see ``log_path_for``)."""
    # <YYYYmmdd>-<HHMMSS>-<kind name>
    parts = path.stem.split("-", 2)
    return PipelineKind.__members__.get(parts[2]) if len(parts) == 3 else None


def start(command: list[str], cwd: Path, log_path: Path) -> RunHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w") as log:
        log.write(f"$ cd {shlex.quote(str(cwd))}\n$ {shlex.join(command)}\n\n")
        log.flush()
        # Own session, so stop() can take down the whole group, including
        # the forge3d viewer or the sentinel_sr `uv run` the CLI spawns.
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    return RunHandle(process=process, log_path=log_path, command=command)


def stop(handle: RunHandle) -> None:
    if not handle.running:
        return
    try:
        os.killpg(handle.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        handle.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(handle.process.pid, signal.SIGKILL)


def tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
