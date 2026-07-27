"""Fetch a DEM clipped to a reference raster's footprint, via Earth Engine."""

from pathlib import Path

import geemap
import geopandas as gpd
import rasterio

import geets
from geets.terrain import load_copernicus_dem, load_fabdem, load_srtm
from vecraspy import tif_bounds_as_polygon

from eo_art.forge3d_pipes.acquire.schema import DemSource

_LOADERS = {
    DemSource.copernicus: load_copernicus_dem,
    DemSource.fabdem: load_fabdem,
    DemSource.srtm: load_srtm,
}


def fetch_terrain_dem(
    reference_tif: str | Path,
    out_dir: str | Path,
    *,
    source: DemSource,
    ee_project: str,
    scale: float,
    filename: str = "dem",
) -> Path:
    """Download a DEM clipped to ``reference_tif``'s footprint."""
    geets.initialize_ee(project=ee_project)

    reference_tif = Path(reference_tif)
    ref_polygon = tif_bounds_as_polygon(reference_tif, exact=True)
    with rasterio.open(reference_tif) as src:
        ref_crs = src.crs

    aoi_gdf = gpd.GeoDataFrame(geometry=[ref_polygon], crs=ref_crs)
    aoi = geemap.geopandas_to_ee(aoi_gdf).geometry()

    dem_image = _LOADERS[source](aoi, clip=True)

    return geets.l_download_image(
        dem_image, Path(out_dir), filename, scale=scale, region=aoi, crs=str(ref_crs)
    )
