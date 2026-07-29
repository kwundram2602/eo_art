"""Fetch a DEM clipped to a reference raster's footprint, via Earth Engine
or from a local GeoTIFF."""

from pathlib import Path

import geemap
import geets
import geopandas as gpd
import rasterio
from geets.terrain import load_copernicus_dem, load_fabdem, load_srtm
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
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
    """Download a DEM clipped to ``reference_tif``'s footprint via Earth Engine.

    ``geets.l_download_image`` is limited to ~32 MB per file, which at 1 m
    posting is only about a 2.8 x 2.8 km tile -- workable for the ~30 m
    sources in ``_LOADERS``, a dead end for high-resolution DEMs. Use
    ``fetch_local_dem`` with a downloaded GeoTIFF for those instead.
    """
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


def fetch_local_dem(
    reference_tif: str | Path,
    out_dir: str | Path,
    *,
    path: str | Path,
    filename: str = "dem",
) -> Path:
    """Clip a local DEM GeoTIFF to ``reference_tif``'s footprint.

    Unlike ``fetch_terrain_dem``, the output keeps the DEM's own CRS and
    pixel size: reprojecting a high-resolution local DGM onto the reference
    (optical) grid at this stage would throw away the real elevation detail
    this path exists to preserve. ``dtm.mode`` decides which grid the DTM
    and optical raster ultimately share.
    """
    reference_tif = Path(reference_tif)
    dem_path = Path(path)
    out_path = Path(out_dir) / f"{filename}.tif"

    with rasterio.open(reference_tif) as ref_src:
        ref_bounds = ref_src.bounds
        ref_crs = ref_src.crs

    with rasterio.open(dem_path) as dem_src:
        dem_bounds = dem_src.bounds
        window_bounds = transform_bounds(ref_crs, dem_src.crs, *ref_bounds)
        left, bottom, right, top = window_bounds
        if (
            right <= dem_bounds.left
            or left >= dem_bounds.right
            or top <= dem_bounds.bottom
            or bottom >= dem_bounds.top
        ):
            raise ValueError(
                f"local DEM {dem_path} does not cover the footprint of "
                f"{reference_tif} (dem bounds={dem_bounds}, reference "
                f"bounds in dem CRS={window_bounds})"
            )

        # Clip to the DEM's own extent -- a reference that only partially
        # overlaps must not attempt to read past the source raster's edges.
        clipped_bounds = (
            max(left, dem_bounds.left),
            max(bottom, dem_bounds.bottom),
            min(right, dem_bounds.right),
            min(top, dem_bounds.top),
        )
        window = from_bounds(*clipped_bounds, transform=dem_src.transform)
        window = window.round_offsets().round_lengths()

        data = dem_src.read(window=window)
        profile = dem_src.profile.copy()
        profile.update(
            height=window.height,
            width=window.width,
            transform=dem_src.window_transform(window),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)

    return out_path
