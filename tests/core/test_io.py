import pytest

from eo_art.core.data import EOData
from eo_art.core.errors import CRSMissingError, EODataLoadError
from eo_art.core.io import load_file


def test_load_dem_tif(dem_tif):
    eo = load_file(dem_tif)
    assert eo.kind == "dem"
    assert "4326" in eo.crs
    assert eo.resolution > 0


def test_load_rgb_tif(rgb_tif):
    eo = load_file(rgb_tif)
    assert eo.kind == "raster"
    assert eo.ds["data"].sizes["band"] == 4


def test_load_missing_file_raises():
    with pytest.raises(EODataLoadError):
        load_file("/nonexistent/path/file.tif")


def test_from_file_shortcut(dem_tif):
    eo = EOData.from_file(dem_tif)
    assert eo.kind == "dem"
