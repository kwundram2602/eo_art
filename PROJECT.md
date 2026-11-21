# eo_art

## Offen
- [ ] Commit the Flet app on `feat/flet-app`
- [ ] Run a real acquire and a forge3d render through the new Run tabs
- [ ] dk_sk `render.yaml`: point `input.path`/overlay at `out/runs/latest-acquire/` after the next acquire
- [ ] Real SR run with `sentinel.zones` on a 2×2 AOI: check the VRT and how forge3d handles many tile overlays
- [ ] Real zoned SR timeseries (2×2 AOI, ~4 monthly steps, `single_date: true`): profile across a former zone border, shifts in `steps.json`, render as one overlay
- [ ] Pin stac2cube in `sentinel_sr` to 5ffaffd (`uv add ... --rev`), blocked in session; `superres.py` does its own edge padding meanwhile
- [ ] LR-first for `run_sr_zones` and acquire with zones too (one SR mosaic instead of per-zone tiles), then drop `zones.mosaic`

## Journal
### 2026-11-22
- Zoned SR timeseries now LR-first: 10 m per zone → mosaic per step on the full grid → AROSICS → SR over the whole mosaic with overlapping patches and edge padding, so zone borders leave no SR seam (`run_sr_timeseries_zones`, `sentinel_sr/{superres,coreg}.py`)
- Co-registration via stac2cube's `coregister_cube` (consensus windows, cloud-aware placement, one cubic warp), `sentinel.coreg` block; shifts and dropped steps in `steps.json`; multi-date steps get one shift per acquisition date
- `timeseries.single_date` added as the strict alternative to `max_zone_offset_days`: a step only on a day every zone has a scene (`zone_steps._common_date`)
- AROSICS underestimates shifts several-fold on smooth texture (Gaussian σ≥2 noise: 2 px → 0.4 px); stac2cube's edge roughness is NaN below 3 scenes, so `iteration=auto` only from 3 steps on
### 2026-11-21
- Acquire/SR results per run: `out/runs/<time>-<kind>/` with `config.yaml`, cache stays shared in `out/_acquire/`, `out/runs/latest-acquire|latest-sr` symlinks for render configs; app passes its log name as run id, Results grouped per run (`acquire/runs.py`, `results.result_groups`)
- Zoned SR timeseries now shares one step per interval: candidates of all zones first (`--candidates-only`), dates chosen together within `timeseries.max_zone_offset_days`, zones without a scene in the window relaxed to the nearest one with the offset reported (`zone_steps.py`, `steps.json`); VRT and optional mosaic per step; `max_items` stays ignored for timeseries
- Copy project added: new project from another's top-level configs and AOIs, out/ stays behind (`projects.copy_project`, Projects page)
- SR/acquire for large AOIs: `sentinel.zones` cuts the AOI into a grid of 128 px chips, one sentinel_sr run per zone, `max_items` per zone, resumable zone by zone (`acquire/zones.py`, `run_sr_zones`, `run_sr_timeseries_zones`)
- Acquire with zones fetches one DEM over all zones and aligns nothing (DTM always native, `dtm.mode` ignored so the shared acquire/SR config stays valid); tiles reach forge3d as one overlay each, aligned at render time via `align_to_dem`, with an `rgb_stretch` shared across the tiles (`lower_value`/`upper_value`)
- Zone tiles + VRT in `out/<aoi>_sr_zones/`, mosaic as COG on request (`zones.mosaic`, `eo-art-sr --mosaic-only`, Export button in Results); AOI map shows the zone grid
- Overlays moved into the Overlay view (editor + own Save next to the preview, which rebuilds on save); Configs tabs now Form · Overlay · YAML · Resolved, Form keeps an overlay summary in second place and DEM prepare
- Overlays and prepare chains editable in the render form (add/remove/reorder overlays and ops, op params generated from the registry schemas, tuple fields for `bands`/`extent`) (`components/list_editors.py`, `components/fields.py`, `config_forms.clean_lists`/`check_lists`)
- Optical overlay rendered with red and blue swapped (no `prepare`, so B02,B03,B04 went in as R,G,B); starter render config and dk_sk now `rgb_stretch bands: [3, 2, 1]`
- Overlay preview added as 4th view in Forge3D › Configs: renderer's PNG plus overlay on a DEM hillshade, no GPU, same prep cache (`app/overlay_preview.py`, `pipeline.prepare_scene`)
- Orbit video ignored `render.camera` and flew at the `orbit.radius_start` default of 26000 m; unset `orbit.*_start` now come from `render.camera`, unset `phi_end` = start + 360 (`render/animation.py`)
- SR timeseries, acquire timeseries mode and Streamlit app committed on `feat/render-quality`
- Streamlit replaced by Flet: NavigationRail over Projects · AOI · Acquire · Forge3D, one tab per step and route (`src/eo_art/app/ui/`) — uncommitted, on `feat/flet-app`
- Acquire and SR kept as one stage sharing one acquire config; SR is its own section in the config form
- AOI drawing moved from folium to flet-map clicks (rectangle / polygon) — flet-webview has no Linux desktop support
- UI-free helpers split out for tests (`app/results.py`, `composite.build_gif`/`histogram_png`, `aoi.rectangle_feature`, `config_forms.save_edits`)
- Config form regrouped: every subgroup collapsible, `enabled` as switch in the group header, field/set/unsaved badges, pinned save bar with expand/collapse all (`config_forms.group_tree`, `components/config_editor.py`) — render form opens as 5 collapsed-ish sections instead of one 66-field block
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
