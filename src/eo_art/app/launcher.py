"""``eoart-app``: start the Streamlit UI with the current directory as base dir."""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

BASE_DIR_ENV = "EOART_BASE_DIR"
PROJECTS_DIRNAME = "projects"
APP_SCRIPT = Path(__file__).parent / "ui" / "main.py"


def base_dir() -> Path:
    """Where ``eoart-app`` was started; projects go in its ``projects/``."""
    return Path(os.environ.get(BASE_DIR_ENV, os.getcwd())).resolve()


def projects_dir() -> Path:
    """Where project folders live: ``<base>/projects`` (created on demand)."""
    return base_dir() / PROJECTS_DIRNAME


def main(argv: Sequence[str] | None = None) -> int:
    from streamlit.web import cli as stcli

    extra = list(sys.argv[1:] if argv is None else argv)
    os.environ[BASE_DIR_ENV] = str(Path.cwd().resolve())
    # streamlit's CLI is click-based and reads sys.argv itself; extra args
    # (e.g. --server.port 8600) pass straight through to `streamlit run`.
    sys.argv = ["streamlit", "run", str(APP_SCRIPT), *extra]
    return stcli.main()
