"""One-line explanations of config fields, shown as hover help in the forms.

Keys are the dotted field paths of ``iter_fields`` (see ``config_forms``).
"""

RENDER_HELP: dict[str, str] = {
    # run
    "run.name": "Name of this run; output goes to <out_dir>/<name>/.",
    "run.out_dir": "Folder for render output, relative to the project folder.",
    "run.cache": "Reuse cached prepare steps (reproject, scale, ...) from earlier "
    "runs instead of recomputing them.",
    "run.fail_fast": "In a sweep: stop at the first failing variant instead of "
    "rendering the rest.",
    "run.gpu": "Which GPU to render on: auto, discrete, integrated, or off (CPU).",
    # input
    "input.path": "DEM GeoTIFF to render as terrain (acquire writes "
    "out/runs/latest-acquire/dtm.tif).",
    # render
    "render.width": "Image width in pixels.",
    "render.height": "Image height in pixels.",
    "render.snapshot_name": "File name of the rendered image.",
    "render.backend": "viewer: interactive forge3d window (overlays, animation). "
    "offscreen: headless still image, no overlays.",
    "render.overlay_preserve_colors": "Draw overlay colours on top after terrain "
    "lighting, so categorical palettes stay unchanged (viewer only).",
    "render.camera.phi": "Camera azimuth in degrees: the direction you look at the "
    "terrain from.",
    "render.camera.theta": "Camera tilt in degrees: 0 = straight down, towards 90 "
    "= near the horizon.",
    "render.camera.radius": "Camera distance from the terrain centre, in map "
    "units (metres).",
    "render.camera.fov": "Vertical field of view in degrees (1–179). Smaller = "
    "more telephoto.",
    "render.sun.azimuth": "Sun direction in degrees (0–360), compass-style.",
    "render.sun.elevation": "Sun height above the horizon in degrees (0–90). Low "
    "values give long shadows.",
    "render.sun.intensity": "Sun light strength.",
    "render.sun.ambient": "Base light in shadowed areas (viewer only).",
    "render.terrain.zscale": "Vertical exaggeration of the terrain; 1 = true height.",
    # pbr
    "render.pbr.enabled": "Physically based shading with shadows, materials, sky "
    "and tonemapping (viewer only).",
    "render.pbr.shadow_technique": "Shadow method: pcss = soft contact-hardening, "
    "pcf = filtered hard edges, none = no shadows.",
    "render.pbr.shadow_map_res": "Shadow map size in pixels; higher = sharper "
    "shadows, more GPU memory.",
    "render.pbr.exposure": "Overall image brightness before tonemapping.",
    "render.pbr.msaa": "Edge anti-aliasing samples: 1, 2, 4 or 8.",
    "render.pbr.ibl_intensity": "Strength of image-based (environment) lighting.",
    "render.pbr.normal_strength": "How strongly surface normals bring out small "
    "relief details.",
    "render.pbr.height_ao.enabled": "Ambient occlusion from the height field: "
    "darkens valleys and crevices.",
    "render.pbr.height_ao.directions": "Number of directions sampled for AO.",
    "render.pbr.height_ao.steps": "Steps per direction; more = smoother AO, slower.",
    "render.pbr.height_ao.max_distance": "How far AO looks for occluding terrain, "
    "in metres.",
    "render.pbr.height_ao.strength": "How dark the occlusion gets.",
    "render.pbr.height_ao.resolution_scale": "AO buffer size relative to the "
    "image (0.1–1).",
    "render.pbr.sun_visibility.enabled": "Terrain self-shadowing from the sun, "
    "traced over the height field.",
    "render.pbr.sun_visibility.mode": "soft = blurred shadow edges, hard = crisp "
    "edges.",
    "render.pbr.sun_visibility.samples": "Samples per pixel for soft shadows.",
    "render.pbr.sun_visibility.steps": "Ray steps towards the sun; more = fewer "
    "missed occluders, slower.",
    "render.pbr.sun_visibility.max_distance": "How far the shadow ray travels, "
    "in metres.",
    "render.pbr.sun_visibility.softness": "Width of the shadow penumbra.",
    "render.pbr.sun_visibility.bias": "Offset against self-shadowing artefacts (acne).",
    "render.pbr.sun_visibility.resolution_scale": "Shadow buffer size relative "
    "to the image (0.1–1).",
    "render.pbr.materials.snow_enabled": "Paint snow on high, flat enough terrain.",
    "render.pbr.materials.snow_altitude_min": "Snow line in metres.",
    "render.pbr.materials.snow_altitude_blend": "Height range in metres over "
    "which snow fades in.",
    "render.pbr.materials.snow_slope_max": "Steepest slope in degrees that still "
    "holds snow.",
    "render.pbr.materials.rock_enabled": "Show bare rock on steep slopes.",
    "render.pbr.materials.rock_slope_min": "Slope in degrees from which rock shows.",
    "render.pbr.materials.wetness_enabled": "Darker, glossier wet look in low areas.",
    "render.pbr.materials.wetness_strength": "How strong the wet look is.",
    "render.pbr.tonemap.operator": "HDR to screen mapping: aces = filmic, "
    "reinhard = softer, linear = none.",
    "render.pbr.tonemap.white_point": "Brightness that maps to pure white; higher "
    "keeps more highlight detail.",
    "render.pbr.tonemap.white_balance_enabled": "Apply temperature and tint.",
    "render.pbr.tonemap.temperature": "White balance in Kelvin; lower = cooler "
    "(bluer) image, higher = warmer.",
    "render.pbr.tonemap.tint": "Green–magenta shift; 0 = neutral.",
    "render.pbr.lens_effects.enabled": "Camera lens effects: distortion, "
    "chromatic aberration, vignette.",
    "render.pbr.lens_effects.distortion": "Barrel (+) or pincushion (−) "
    "distortion; 0 = none.",
    "render.pbr.lens_effects.chromatic_aberration": "Colour fringes towards the "
    "image edges; 0 = none.",
    "render.pbr.lens_effects.vignette_strength": "How much the corners darken.",
    "render.pbr.lens_effects.vignette_radius": "Where the darkening starts, from "
    "the centre (0–1).",
    "render.pbr.lens_effects.vignette_softness": "How gradual the vignette edge is.",
    "render.pbr.sky.enabled": "Physical sky and atmosphere.",
    "render.pbr.sky.turbidity": "Haze in the air: ~2 = clear, higher = hazier.",
    "render.pbr.sky.ground_albedo": "How much light the ground reflects into the "
    "sky (0–1).",
    "render.pbr.sky.sun_intensity": "Brightness of the sun disc in the sky.",
    "render.pbr.sky.aerial_perspective": "Distant terrain fades into the haze.",
    "render.pbr.sky.sky_exposure": "Sky brightness.",
    # quality (offscreen only)
    "render.quality.render_scale": "Render at this multiple of the image size, "
    "then downsample (0.25–4). Offscreen only.",
    "render.quality.aa_samples": "Accumulated anti-aliasing samples: 1, 16, 64 or "
    "256. Offscreen only.",
    "render.quality.anisotropy": "Texture filtering at grazing angles (1–16). "
    "Offscreen only.",
    "render.quality.mip_filter": "Filter between texture mip levels. Offscreen only.",
    "render.quality.pom_enabled": "Parallax occlusion mapping for fine relief. "
    "Offscreen only.",
    "render.quality.overlay_resolution_scale": "Overlay texture resolution "
    "factor (0.1–2). Offscreen only.",
    # animation / export
    "animation.kind": "none = still image, orbit = camera flies around the "
    "terrain (viewer only).",
    "animation.fps": "Frames per second of the animation.",
    "animation.orbit.duration": "Length of the orbit in seconds.",
    "animation.orbit.phi_start": "Start azimuth in degrees; unset = render.camera.phi.",
    "animation.orbit.phi_end": "End azimuth in degrees; unset = one full turn "
    "(start + 360).",
    "animation.orbit.theta_start": "Start tilt in degrees (0 = straight down); "
    "unset = render.camera.theta.",
    "animation.orbit.theta_end": "End tilt; unset = keep the start tilt.",
    "animation.orbit.radius_start": "Start camera distance in metres; unset = "
    "render.camera.radius.",
    "animation.orbit.radius_end": "End distance; unset = keep the start distance.",
    "animation.orbit.fov_start": "Start field of view in degrees; unset = "
    "render.camera.fov.",
    "animation.orbit.fov_end": "End field of view; unset = keep the start value.",
    "export.video.enabled": "Encode the animation frames into a video (needs an "
    "animation).",
    "export.video.format": "mp4 or gif.",
    "export.video.fps": "Frames per second of the video.",
    "export.video.quality": "Encoding quality, 1 (small) to 10 (best).",
}

# Acquire config (shared by acquire and eo-art-sr).
ACQUIRE_HELP: dict[str, str] = {
    "aoi_path": "AOI vector file (GeoPackage/GeoJSON), relative to the project folder.",
    "ee_project": "Google Earth Engine project for the DEM download; not needed "
    "with dem.source = local.",
    "out_dir": "Output folder; every run writes into out/runs/<time>-<kind>/, "
    "the cache stays in out/_acquire/.",
    "sentinel_sr_dir": "Local checkout of the sentinel_sr project (runs in its "
    "own venv with torch).",
    # sentinel
    "sentinel.start_date": "First acquisition date searched (YYYY-MM-DD).",
    "sentinel.end_date": "Last acquisition date searched (YYYY-MM-DD).",
    "sentinel.model_path": "Folder of the SR model (mlstac, e.g. LDSRS2-SEN2SR).",
    "sentinel.collection": "STAC collection, normally sentinel-2-l2a.",
    "sentinel.edge_size": "Chip size in pixels: without zones the one chip around "
    "the AOI centre, with zones the size of each zone.",
    "sentinel.resolution": "Input pixel size in metres (10 m; SR makes 2.5 m).",
    "sentinel.stac": "STAC API to search (default: Planetary Computer).",
    "sentinel.max_items": "Most scenes to super-resolve (per zone with zones); "
    "ignored for timeseries.",
    "sentinel.max_cloud_cover": "Drop scenes whose whole tile is cloudier than "
    "this percent (eo:cloud_cover, before download).",
    "sentinel.max_aoi_cloud": "Drop scenes with more than this percent of AOI "
    "pixels in the SCL cloud classes; unset = no check.",
    "sentinel.reference_index": "Which scene acquire carries forward (by "
    "position); unset = the least cloudy.",
    # timeseries
    "sentinel.timeseries.enabled": "eo-art-sr only: one scene per interval "
    "instead of a single reference scene.",
    "sentinel.timeseries.interval": "Length of one interval: day, week, month or year.",
    "sentinel.timeseries.every": "One scene every N intervals (e.g. week + 2 = "
    "biweekly).",
    "sentinel.timeseries.fill_gaps": "Continue the latest SR run's series instead "
    "of rebuilding it: done intervals are kept, only empty ones are searched "
    "again (e.g. after a longer date range or looser cloud limits).",
    "sentinel.timeseries.max_zone_offset_days": "With zones: the zones' dates of "
    "one step may lie at most this many days apart; unset = each zone picks "
    "its own least cloudy scene.",
    "sentinel.timeseries.single_date": "With zones: each step is one acquisition "
    "for the whole AOI; intervals without a day shared by all zones are "
    "skipped. Excludes max_zone_offset_days.",
    # zones
    "sentinel.zones.enabled": "Cover a large AOI with a grid of edge_size chips "
    "instead of one chip around its centre.",
    "sentinel.zones.clip_to_aoi": "Skip grid cells that don't touch an irregular "
    "AOI; off = the full rectangle.",
    "sentinel.zones.mosaic": "Without timeseries: also write the zone tiles as one "
    "GeoTIFF (<aoi>_sr_mosaic.tif).",
    # coreg
    "sentinel.coreg.enabled": "Zoned timeseries: align the steps with AROSICS at "
    "10 m before SR; unmatchable steps are dropped.",
    "sentinel.coreg.anchor": "Reference step: auto (most textured, low cloud, mid "
    "series), first, composite or a date YYYY-MM-DD.",
    "sentinel.coreg.grid_size": "Matching windows per step: up to grid_size² + 1; "
    "more = robust but slower.",
    # dem / dtm / align
    "dem.source": "Where the elevation comes from: copernicus, fabdem (no trees "
    "and buildings), srtm (Earth Engine) or local.",
    "dem.scale": "Download resolution in metres for Earth Engine sources.",
    "dem.path": "Local DEM GeoTIFF (e.g. 1 m LiDAR DGM); only with dem.source = local.",
    "dtm.apply_erosion": "Run a thermal erosion simulation over the DTM (softer, "
    "more natural slopes).",
    "dtm.radius": "guided_upsample: filter window radius in optical pixels; larger "
    "= smoother transfer of optical edges.",
    "dtm.eps": "guided_upsample: filter regularisation; larger = fewer optical "
    "details in the DTM.",
    "dtm.band": "guided_upsample: optical band (1-based) used as guide; unset = "
    "mean of all bands.",
    "dtm.mode": "guided_upsample: DEM refined onto the optical grid (coarse DEMs). "
    "native: optical aligned onto the DEM grid (high-res local DEMs).",
    "align.resampling": "Resampling used to put the optical raster onto the DTM grid.",
}

# Fields of one entry in ``overlays:`` (its prepare chain has its own editor).
OVERLAY_HELP: dict[str, str] = {
    "name": "Label of this overlay; only used in logs and the preview.",
    "path": "Raster draped onto the terrain, relative to the project folder "
    "(acquire writes out/runs/latest-acquire/optical_aligned.tif).",
    "align_to_dem": "Snap the overlay onto the DEM's grid after its prepare "
    "chain, so both line up pixel for pixel.",
    "opacity": "0 = invisible, 1 = opaque; unset = opaque.",
    "z_order": "With several overlays: higher draws on top.",
    "extent": "Manual placement as (u0, v0, u1, v1) in 0..1 of the DEM; unset = "
    "computed from the geographic bounds.",
}

# Prep op -> (one-line description, help per parameter).
OP_HELP: dict[str, tuple[str, dict[str, str]]] = {
    "reproject": (
        "warp into another CRS",
        {
            "crs": "Target CRS, e.g. EPSG:32632.",
            "resampling": "nearest keeps exact values (classes); bilinear/cubic "
            "smooth continuous data.",
            "dtype": "auto turns integers into float32 when interpolating, so "
            "heights don't terrace.",
        },
    ),
    "scale_to_gsd": (
        "resample to a pixel size",
        {
            "target_gsd": "Pixel size in CRS units (metres for UTM).",
            "resampling": "How pixels are combined or interpolated.",
        },
    ),
    "align_to": (
        "snap onto another raster's grid",
        {
            "reference": "Raster whose grid (CRS, extent, pixel size) to match, "
            "relative to the project folder.",
            "resampling": "How pixels are interpolated onto that grid.",
        },
    ),
    "saturate": (
        "make colours more or less vivid",
        {"factor": "0 = grayscale, 1 = unchanged, above 1 = more vivid."},
    ),
    "rgb_stretch": (
        "pick 3 bands as R, G, B and stretch them",
        {
            "bands": "Band numbers used as R, G, B (1-based). Acquire output is "
            "B02,B03,B04,...: true colour = 3, 2, 1; false colour NIR = 7, 3, 2.",
            "lower_percentile": "Values below this percentile become black.",
            "upper_percentile": "Values above this percentile become white.",
            "shared_stretch": "One stretch for all three bands keeps natural "
            "colour balance; off = each band stretched on its own.",
            "lower_value": "Fixed value that becomes black, instead of the "
            "percentile (set together with upper_value). Set automatically for "
            "SR zone tiles so they share one stretch.",
            "upper_value": "Fixed value that becomes white, instead of the "
            "percentile (set together with lower_value).",
        },
    ),
}
