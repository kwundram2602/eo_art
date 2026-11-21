import numpy as np
import xarray as xr

from sentinel_sr.rgbn import band_descriptions


def _cube(bands, common_names=None):
    coords = {"band": list(bands)}
    if common_names is not None:
        coords["common_name"] = ("band", list(common_names))
    return xr.DataArray(
        np.zeros((len(bands), 2, 2)), dims=("band", "y", "x"), coords=coords
    )


def test_unique_common_names_are_used_as_is():
    bands = ("B02", "B03", "B04", "B08")
    da = _cube(bands, ["blue", "green", "red", "nir"])

    assert band_descriptions(da, bands) == ["blue", "green", "red", "nir"]


def test_shared_common_names_get_the_band_id():
    bands = ("B05", "B06", "B07", "B11")
    da = _cube(bands, ["rededge", "rededge", "rededge", "swir16"])

    assert band_descriptions(da, bands) == [
        "rededge_B05",
        "rededge_B06",
        "rededge_B07",
        "swir16",
    ]


def test_missing_common_name_falls_back_to_band_id():
    bands = ("B02", "B03")

    assert band_descriptions(_cube(bands), bands) == ["B02", "B03"]
    assert band_descriptions(_cube(bands, ["blue", None]), bands) == ["blue", "B03"]
