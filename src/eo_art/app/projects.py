"""Project folders: ``<projects>/<name>/`` holding aoi/, configs/ and out/.

``base`` in this module is the folder that holds the projects -- the app
passes ``<start folder>/projects``; it need not exist until a project is
created.
"""

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from eo_art.app.templates import starter_configs

MARKER = "eoart-project.yaml"
LOGS_DIRNAME = "_logs"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class Project:
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def aoi_dir(self) -> Path:
        return self.root / "aoi"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def logs_dir(self) -> Path:
        return self.out_dir / LOGS_DIRNAME

    def aoi_files(self) -> list[Path]:
        return sorted(self.aoi_dir.glob("*.gpkg"))

    def config_files(self) -> list[Path]:
        return sorted(
            [*self.configs_dir.glob("*.yaml"), *self.configs_dir.glob("*.yml")]
        )


def validate_name(name: str) -> str:
    name = name.strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"project name must only contain letters, digits, '_' or '-', got {name!r}"
        )
    return name


def is_project(path: Path) -> bool:
    return (path / MARKER).is_file()


def list_projects(base: Path) -> list[Project]:
    if not base.is_dir():
        return []
    return [Project(p) for p in sorted(base.iterdir()) if p.is_dir() and is_project(p)]


def candidate_folders(base: Path) -> list[Path]:
    """Subfolders that could be adopted as a project (no marker yet)."""
    if not base.is_dir():
        return []
    return [
        p
        for p in sorted(base.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_")) and not is_project(p)
    ]


def _write_marker(root: Path) -> None:
    (root / MARKER).write_text(
        yaml.safe_dump(
            {"name": root.name, "created": datetime.now().isoformat(timespec="seconds")}
        )
    )


def _ensure_layout(project: Project) -> None:
    for folder in (project.aoi_dir, project.configs_dir, project.logs_dir):
        folder.mkdir(parents=True, exist_ok=True)


def create_project(base: Path, name: str) -> Project:
    name = validate_name(name)
    root = base / name
    if root.exists():
        raise FileExistsError(f"{root} already exists; adopt it instead")
    project = Project(root)
    _ensure_layout(project)
    for filename, text in starter_configs(name).items():
        (project.configs_dir / filename).write_text(text)
    _write_marker(root)
    return project


def adopt_folder(base: Path, name: str) -> Project:
    """Turn an existing folder into a project without touching its files."""
    root = base / name
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder")
    project = Project(root)
    _ensure_layout(project)
    for filename, text in starter_configs(root.name).items():
        target = project.configs_dir / filename
        if not target.exists():
            target.write_text(text)
    if not is_project(root):
        _write_marker(root)
    return project


def copy_project(base: Path, source: Project, name: str) -> Project:
    """A new project ``name`` with ``source``'s top-level configs and AOIs.

    Configs are copied verbatim (relative ``aoi_path``s keep working, as the
    AOIs come along); subfolders of configs/ and everything in out/ stay
    behind.
    """
    name = validate_name(name)
    root = base / name
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    project = Project(root)
    _ensure_layout(project)
    for path in [*source.config_files(), *source.aoi_files()]:
        target_dir = (
            project.configs_dir
            if path.parent == source.configs_dir
            else project.aoi_dir
        )
        shutil.copy2(path, target_dir / path.name)
    _write_marker(root)
    return project
