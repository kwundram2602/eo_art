"""Acquire › Config: the acquire YAML (Sentinel-2 SR + DEM/DTM in one file)."""

import flet as ft

from eo_art.app.configs import ConfigKind
from eo_art.app.templates import ACQUIRE_FILENAME
from eo_art.app.ui.components.config_editor import ConfigEditor


@ft.component
def ConfigPage() -> ft.Control:
    return ConfigEditor(ConfigKind.acquire, {"Acquire (starter)": ACQUIRE_FILENAME})
