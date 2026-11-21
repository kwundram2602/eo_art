"""Acquire stage: Sentinel-2 super-resolution and DEM/DTM acquisition, which
share one acquire config."""

from eo_art.app.ui.pages.acquire.composite import CompositePage
from eo_art.app.ui.pages.acquire.config import ConfigPage
from eo_art.app.ui.pages.acquire.results import ResultsPage
from eo_art.app.ui.pages.acquire.run import RunPage

__all__ = ["CompositePage", "ConfigPage", "ResultsPage", "RunPage"]
