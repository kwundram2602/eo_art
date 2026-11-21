"""One folder per acquire/SR run: ``<out_dir>/runs/<YYYYmmdd-HHMMSS>-<kind>/``.

Results land in the run folder next to the resolved config, so no run
overwrites another. The expensive cache (``<out_dir>/_acquire``) stays
shared, so a rerun with unchanged settings only copies its results.
``<out_dir>/runs/latest-<kind>`` points at the newest finished run of a kind,
giving render configs a stable path (e.g. ``out/runs/latest-acquire/dtm.tif``).
"""

import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

RUNS_DIRNAME = "runs"
CONFIG_NAME = "config.yaml"
LATEST_PREFIX = "latest-"


def runs_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / RUNS_DIRNAME


def new_run_id(kind: str, now: datetime | None = None) -> str:
    """``20261122-101500-sr``; same scheme as the app's run logs."""
    return f"{(now or datetime.now()):%Y%m%d-%H%M%S}-{kind}"


def new_run_dir(out_dir: str | Path, kind: str, run_id: str | None = None) -> Path:
    """Create the folder of a new run; a taken id gets a ``-2``, ``-3``, ...
    suffix instead of mixing two runs."""
    base = runs_dir(out_dir)
    name = run_id or new_run_id(kind)
    candidate, n = base / name, 1
    while candidate.exists():
        n += 1
        candidate = base / f"{name}-{n}"
    candidate.mkdir(parents=True)
    return candidate


def write_run_config(run_dir: Path, cfg: object) -> Path:
    """The resolved config of the run, for telling runs apart later."""
    path = run_dir / CONFIG_NAME
    path.write_text(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    return path


def latest_link(out_dir: str | Path, kind: str) -> Path:
    return runs_dir(out_dir) / f"{LATEST_PREFIX}{kind}"


def link_latest(out_dir: str | Path, kind: str, run_dir: Path) -> Path:
    """Point ``runs/latest-<kind>`` at ``run_dir`` (relative symlink,
    replaced atomically)."""
    link = latest_link(out_dir, kind)
    tmp = link.with_name(f".{link.name}.tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(os.path.relpath(run_dir, link.parent), target_is_directory=True)
    tmp.replace(link)
    return link


def latest_run(out_dir: str | Path, kind: str) -> Path | None:
    """The newest finished run of ``kind``, if any."""
    link = latest_link(out_dir, kind)
    return link.resolve() if link.is_dir() else None


def list_runs(out_dir: str | Path) -> list[Path]:
    """Run folders, newest first (the ``latest-*`` links left out)."""
    base = runs_dir(out_dir)
    if not base.is_dir():
        return []
    return sorted(
        (
            p
            for p in base.iterdir()
            if p.is_dir()
            and not p.is_symlink()
            and not p.name.startswith((".", LATEST_PREFIX))
        ),
        key=lambda p: p.name,
        reverse=True,
    )
