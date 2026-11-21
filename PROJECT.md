# eo_art

## Offen
- [ ] Commit the uncommitted SR timeseries + Streamlit app work on `feat/render-quality`

## Journal
### 2026-10-10
- SR timeseries added: one scene per day/week/month/year, AOI-level SCL cloud check, duplicate overflights collapsed (`sentinel_sr/{timeseries,clouds,scenes}.py`) — uncommitted
- Cross-UTM-zone tiles dropped before SR — nearest-neighbour reprojection caused staircase seams that SR sharpened into blocks
- Streamlit front end over the pipelines (projects, AOI, configs, run, results), started via `eoart-app` (`src/eo_art/app/`) — uncommitted
- Sentinel-2 super-resolution wired into acquire as `eo-art-sr` (`src/eo_art/sr_cli.py`, `sentinel_sr/rgbn.py`)

### 2026-07-29
- Offscreen forge3d backend added next to the interactive viewer; fixed crash, black frames, overlays, `camera_mode`, `run.gpu` honoured
- Acquire takes local high-res DEMs and a configurable master grid
- DEM terracing fixed (dtype promotion on interpolation); overlay grid coupled to DEM, alpha-aware overlay export
- GPU adapter selection added

### 2026-07-25 – 2026-07-28
- `forge3d_pipes`: config-driven pipeline (validated schema, cached prep chain, reproject/scale_to_gsd ops, orbit animations, sweeps, video export, CLI)
- Raster overlays, saturation, and the acquire pipe added

### 2026-05-16
- Neural style transfer (Gatys + AdaIN) as optional `neural` extra, exposed via `RenderStep.style_transfer()`; animation and hillshade added

### 2026-05-01
- Package scaffolded; `EOData` with kind auto-detection and GeoTIFF loader (`io.py`)
