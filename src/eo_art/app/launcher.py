"""``eoart-app``: start the Flet UI with the current directory as base dir."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

BASE_DIR_ENV = "EOART_BASE_DIR"
PROJECTS_DIRNAME = "projects"


def base_dir() -> Path:
    """Where ``eoart-app`` was started; projects go in its ``projects/``."""
    return Path(os.environ.get(BASE_DIR_ENV, os.getcwd())).resolve()


def projects_dir() -> Path:
    """Where project folders live: ``<base>/projects`` (created on demand)."""
    return base_dir() / PROJECTS_DIRNAME


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eoart-app",
        description="eo-art UI over the acquire and forge3d pipelines; projects "
        "live in projects/ inside the current folder",
    )
    parser.add_argument(
        "--web", action="store_true", help="open in the browser instead of a window"
    )
    parser.add_argument(
        "--port", type=int, default=0, help="port for --web (default: a free one)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ[BASE_DIR_ENV] = str(Path.cwd().resolve())
    # The light desktop client (Linux default) lacks flet_video; Renders needs it.
    os.environ.setdefault("FLET_DESKTOP_FLAVOR", "full")

    import flet as ft

    from eo_art.app.ui.main import main as app_main

    ft.run(
        app_main,
        view=ft.AppView.WEB_BROWSER if args.web else ft.AppView.FLET_APP,
        port=args.port,
    )
    return 0
