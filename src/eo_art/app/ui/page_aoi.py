"""Draw, upload and manage the project's AOIs on a map."""

from pathlib import Path

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from eo_art.app import aoi
from eo_art.app.configs import (
    ConfigKind,
    load_file,
    set_top_level_value,
)
from eo_art.app.projects import Project, validate_name
from eo_art.app.ui.common import configs_of_kind, rel, require_project

_DEFAULT_CENTER = (47.5, 11.0)
_SATELLITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def _sr_edge_m(project: Project) -> float:
    """SR chip edge in metres, from the first acquire config (or defaults)."""
    from eo_art.forge3d_pipes.acquire.schema import Sentinel

    edge, res = Sentinel.edge_size, Sentinel.resolution
    for path in configs_of_kind(project, ConfigKind.acquire):
        try:
            sentinel = load_file(path).get("sentinel") or {}
            edge = int(sentinel.get("edge_size", edge))
            res = int(sentinel.get("resolution", res))
        except Exception:
            continue
        break
    return float(edge * res)


def _build_map(aois: list[Path], edge_m: float) -> folium.Map:
    fmap = folium.Map(location=_DEFAULT_CENTER, zoom_start=5, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(_SATELLITE, attr="Esri World Imagery", name="Satellite").add_to(
        fmap
    )

    all_bounds: list[tuple[float, float, float, float]] = []
    for path in aois:
        try:
            geojson = aoi.load_geojson(path)
            minx, miny, maxx, maxy = aoi.bounds(path)
            lat, lon = aoi.center(path)
        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")
            continue
        group = folium.FeatureGroup(name=path.stem)
        folium.GeoJson(geojson, tooltip=path.stem).add_to(group)
        south, west, north, east = aoi.chip_bounds(lat, lon, edge_m)
        folium.Rectangle(
            [(south, west), (north, east)],
            color="#d9480f",
            dash_array="6",
            fill=False,
            tooltip=f"{path.stem}: Sentinel SR chip ({edge_m / 1000:.2f} km)",
        ).add_to(group)
        folium.CircleMarker((lat, lon), radius=3, color="#d9480f").add_to(group)
        group.add_to(fmap)
        all_bounds.append((minx, miny, maxx, maxy))

    if all_bounds:
        fmap.fit_bounds(
            [
                (min(b[1] for b in all_bounds), min(b[0] for b in all_bounds)),
                (max(b[3] for b in all_bounds), max(b[2] for b in all_bounds)),
            ]
        )

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": False},
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap


def _target(project: Project, name: str) -> Path:
    return project.aoi_dir / f"{validate_name(name)}.gpkg"


def _save_drawn(project: Project, drawings: list[dict]) -> None:
    st.subheader("Save drawing")
    st.caption(f"{len(drawings)} shape(s) drawn on the map.")
    with st.form("save_drawn", clear_on_submit=True):
        name = st.text_input("AOI name", placeholder="e.g. valley")
        overwrite = st.checkbox("Overwrite if it exists")
        if st.form_submit_button("Save AOI", type="primary", disabled=not drawings):
            try:
                target = _target(project, name)
                if target.exists() and not overwrite:
                    raise FileExistsError(f"{target.name} exists")
                aoi.save_features(drawings, target)
            except (ValueError, FileExistsError) as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved {rel(project, target)}")
                st.rerun()


def _upload(project: Project) -> None:
    st.subheader("Upload")
    with st.form("upload_aoi", clear_on_submit=True):
        upload = st.file_uploader(
            "gpkg / GeoJSON / zipped shapefile",
            type=[s.lstrip(".") for s in aoi.UPLOAD_SUFFIXES],
        )
        name = st.text_input("AOI name (default: file name)")
        if st.form_submit_button("Import") and upload is not None:
            try:
                target = _target(project, name or Path(upload.name).stem)
                aoi.import_upload(upload.getvalue(), upload.name, target)
            except Exception as exc:
                st.error(f"Import failed: {exc}")
            else:
                st.success(f"Imported {rel(project, target)}")
                st.rerun()


def _aoi_list(project: Project, aois: list[Path]) -> None:
    st.subheader("Project AOIs")
    if not aois:
        st.caption("None yet.")
        return
    acquire_cfgs = configs_of_kind(project, ConfigKind.acquire)
    for path in aois:
        with st.expander(path.stem):
            try:
                lat, lon = aoi.center(path)
                st.caption(f"Center {lat:.5f}, {lon:.5f}")
            except Exception as exc:
                st.error(str(exc))
            if acquire_cfgs:
                cfg = st.selectbox(
                    "Acquire config",
                    acquire_cfgs,
                    format_func=lambda p: p.name,
                    key=f"cfg_{path.stem}",
                )
                if st.button("Set as aoi_path", key=f"use_{path.stem}"):
                    set_top_level_value(cfg, "aoi_path", rel(project, path))
                    st.success(f"{cfg.name}: aoi_path = {rel(project, path)}")
            if st.button("Delete", key=f"del_{path.stem}"):
                path.unlink()
                st.rerun()


def render() -> None:
    project = require_project()
    st.title(f"AOI · {project.name}")
    st.caption(
        "Draw a polygon or rectangle, then save it. The dashed square is the "
        "chip Sentinel SR actually processes: it only uses the AOI's **center** "
        "(edge_size × resolution from the acquire config)."
    )

    aois = project.aoi_files()
    map_col, side_col = st.columns([3, 1])
    with map_col:
        state = st_folium(
            _build_map(aois, _sr_edge_m(project)),
            key=f"aoi_map_{project.name}",
            height=620,
            use_container_width=True,
            returned_objects=["all_drawings"],
        )
    drawings = (state or {}).get("all_drawings") or []
    with side_col:
        _save_drawn(project, drawings)
        _upload(project)
        _aoi_list(project, aois)
