"""Forge3D › Configs: render configs and the layers merged on top of them."""

import flet as ft

from eo_art.app.configs import ConfigKind
from eo_art.app.templates import RENDER_FILENAME
from eo_art.app.ui.components.config_editor import ConfigEditor


@ft.component
def ConfigsPage() -> ft.Control:
    return ConfigEditor(
        ConfigKind.render,
        {"Render (starter)": RENDER_FILENAME, "Empty layer": None},
    )
