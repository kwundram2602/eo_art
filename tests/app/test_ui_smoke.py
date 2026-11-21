"""Headless smoke tests: every page renders without raising."""

import pytest
import rasterio
from omegaconf import OmegaConf
from streamlit.testing.v1 import AppTest

from eo_art.app.launcher import APP_SCRIPT, BASE_DIR_ENV, PROJECTS_DIRNAME
from eo_art.app.projects import create_project, list_projects
from eo_art.app.ui.common import PROJECT_PARAM

PAGES = ["aoi", "configs", "run", "results", "composite"]


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv(BASE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / ".mpl"))
    return tmp_path / PROJECTS_DIRNAME


def _app() -> AppTest:
    return AppTest.from_file(str(APP_SCRIPT), default_timeout=60)


def test_projects_page_creates_a_project(base):
    at = _app().run()
    assert not at.exception

    at.text_input[0].input("alps")
    at.button[0].click().run()

    assert not at.exception
    assert [p.name for p in list_projects(base)] == ["alps"]
    assert at.session_state[PROJECT_PARAM] == "alps"


def _page_script(page: str) -> None:
    import importlib

    import streamlit as st

    st.session_state["project"] = "alps"
    importlib.import_module(f"eo_art.app.ui.page_{page}").render()


@pytest.mark.parametrize("page", PAGES)
def test_project_pages_render(base, page):
    create_project(base, "alps")

    at = AppTest.from_function(_page_script, args=(page,), default_timeout=60).run()

    assert not at.exception, at.exception


def _configs_page(filename: str) -> None:
    import streamlit as st

    from eo_art.app.ui import page_configs

    st.session_state["project"] = "alps"
    st.session_state["config_file"] = filename
    page_configs.render()


@pytest.mark.parametrize("filename", ["acquire.yaml", "render.yaml"])
def test_untouched_form_save_changes_nothing(base, filename):
    project = create_project(base, "alps")
    before = (project.configs_dir / filename).read_text()

    at = AppTest.from_function(_configs_page, args=(filename,), default_timeout=60)
    at.run()
    next(b for b in at.button if b.label == "Save").click().run()

    assert not at.exception, at.exception
    assert [i.value for i in at.info] == ["Nothing changed."]
    assert (project.configs_dir / filename).read_text() == before


def test_form_save_writes_only_the_edited_field(base):
    project = create_project(base, "alps")
    path = project.configs_dir / "render.yaml"

    at = AppTest.from_function(_configs_page, args=("render.yaml",), default_timeout=60)
    at.run()
    width = next(n for n in at.number_input if n.label == "width")
    width.set_value(640)
    next(b for b in at.button if b.label == "Save").click().run()

    assert not at.exception, at.exception
    saved = OmegaConf.to_container(OmegaConf.load(path))
    assert saved["render"]["width"] == 640
    assert set(saved["render"]) == {"width", "height", "snapshot_name"}


def test_projects_go_into_projects_subfolder_of_start_folder(base):
    at = _app().run()
    at.text_input[0].input("alps")
    at.button[0].click().run()

    assert not at.exception
    assert (base.parent / PROJECTS_DIRNAME / "alps" / "eoart-project.yaml").is_file()
    assert not (base.parent / "alps").exists()


def _composite_page(mode: str) -> None:
    import streamlit as st

    from eo_art.app.ui import page_composite

    st.session_state["project"] = "alps"
    st.session_state["composite_mode"] = mode
    page_composite.render()


@pytest.mark.parametrize("mode", ["preset", "custom", "single"])
def test_composite_page_renders_project_sr_tif(base, make_sr_tif, mode):
    project = create_project(base, "alps")
    make_sr_tif(project.out_dir / "alps_sr.tif")

    at = AppTest.from_function(_composite_page, args=(mode,), default_timeout=60)
    at.run()

    assert not at.exception, at.exception
    assert at.selectbox(key="composite_tif").value.name == "alps_sr.tif"
    assert len(at.get("image")) == 1
    assert [b.proto.label for b in at.get("download_button")] == ["Download PNG"]


def test_composite_page_offers_indices_geotiff(base, make_sr_tif):
    project = create_project(base, "alps")
    make_sr_tif(project.out_dir / "alps_sr.tif")

    at = AppTest.from_function(_composite_page, args=("preset",), default_timeout=60)
    at.run()
    at.multiselect(key="composite_indices").set_value(["NDVI"]).run()
    next(b for b in at.button if b.label == "Save indices to out/").click().run()

    assert not at.exception, at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Download PNG", "Download indices GeoTIFF"]
    assert (project.out_dir / "alps_sr_indices.tif").is_file()


def test_enabled_toggle_reveals_timeseries_fields(base):
    project = create_project(base, "alps")
    path = project.configs_dir / "acquire.yaml"
    before = path.read_text()

    at = AppTest.from_function(
        _configs_page, args=("acquire.yaml",), default_timeout=60
    )
    at.run()
    labels = {c.label for c in at.checkbox}
    assert "enabled" in labels
    assert "every" not in {n.label for n in at.number_input}

    toggle = next(
        c for c in at.checkbox if c.label == "enabled" and "timeseries" in c.key
    )
    toggle.check().run()
    assert "every" in {n.label for n in at.number_input}
    next(n for n in at.number_input if n.label == "every").set_value(2)
    next(b for b in at.button if b.label == "Save").click().run()

    assert not at.exception, at.exception
    saved = OmegaConf.to_container(OmegaConf.load(path))
    assert saved["sentinel"]["timeseries"]["enabled"] is True
    assert saved["sentinel"]["timeseries"]["every"] == 2
    assert path.read_text() != before


def test_composite_lists_timeseries_steps(base, make_sr_tif):
    project = create_project(base, "alps")
    make_sr_tif(project.out_dir / "alps_sr.tif")
    folder = project.out_dir / "alps_sr_timeseries"
    make_sr_tif(folder / "alps_sr_2023-02-01.tif")
    make_sr_tif(folder / "alps_sr_2023-01-01.tif")

    at = AppTest.from_function(_composite_page, args=("preset",), default_timeout=60)
    at.run()

    assert not at.exception, at.exception
    assert at.selectbox(key="composite_tif").options == [
        "out/alps_sr.tif",
        "out/alps_sr_timeseries/alps_sr_2023-01-01.tif",
        "out/alps_sr_timeseries/alps_sr_2023-02-01.tif",
    ]


def _run_page() -> None:
    import streamlit as st

    from eo_art.app.ui import page_run

    st.session_state["project"] = "alps"
    page_run.render()


def test_run_page_warns_that_acquire_ignores_timeseries(base):
    project = create_project(base, "alps")
    path = project.configs_dir / "acquire.yaml"
    path.write_text(path.read_text().replace("    enabled: false", "    enabled: true"))

    at = AppTest.from_function(_run_page, default_timeout=60).run()

    assert not at.exception, at.exception
    assert any("acquire ignores it" in i.value for i in at.info)


def _gif_project(base, make_sr_tif):
    project = create_project(base, "alps")
    folder = project.out_dir / "alps_sr_timeseries"
    for i, date in enumerate(("2023-01-01", "2023-02-01", "2023-03-01")):
        path = make_sr_tif(folder / f"alps_sr_{date}.tif")
        # Distinct brightness per date (identical frames get merged in a GIF).
        with rasterio.open(path, "r+") as dst:
            dst.write(dst.read() * (1 + i))
    # Saved next to the frames, but not a frame itself.
    make_sr_tif(folder / "alps_sr_2023-01-01_indices.tif")
    return project, folder


def _composite_frame_page(frame: str) -> None:
    import streamlit as st

    from eo_art.app.ui import page_composite

    st.session_state["project"] = "alps"
    st.session_state["composite_mode"] = "preset"
    st.session_state["composite_tif"] = frame
    page_composite.render()


def test_timeseries_gif_build_and_save(base, make_sr_tif):
    project, folder = _gif_project(base, make_sr_tif)

    at = AppTest.from_function(
        _composite_frame_page,
        args=(folder / "alps_sr_2023-02-01.tif",),
        default_timeout=60,
    ).run()
    frames = at.multiselect(key=f"composite_gif_frames_{folder}")
    assert [p.name for p in frames.value] == [
        "alps_sr_2023-01-01.tif",
        "alps_sr_2023-02-01.tif",
        "alps_sr_2023-03-01.tif",
    ]

    frames.set_value(frames.value[:2]).run()
    at.button(key="composite_gif_build").click().run()
    next(b for b in at.button if b.label == "Save GIF to out/").click().run()

    assert not at.exception, at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert "Download GIF" in labels
    gif = project.out_dir / "alps_sr_timeseries_natural-true-color.gif"
    from PIL import Image

    with Image.open(gif) as image:
        assert image.n_frames == 2
        assert image.size == (16, 8)


def test_changed_settings_hide_a_stale_gif(base, make_sr_tif):
    _, folder = _gif_project(base, make_sr_tif)

    at = AppTest.from_function(
        _composite_frame_page,
        args=(folder / "alps_sr_2023-01-01.tif",),
        default_timeout=60,
    ).run()
    at.button(key="composite_gif_build").click().run()
    at.slider(key="composite_gif_seconds").set_value(1.5).run()

    assert not at.exception, at.exception
    assert "Download GIF" not in [b.proto.label for b in at.get("download_button")]
    assert any("Settings changed" in c.value for c in at.caption)


def test_no_gif_section_outside_a_timeseries_folder(base, make_sr_tif):
    project = create_project(base, "alps")
    make_sr_tif(project.out_dir / "alps_sr.tif")

    at = AppTest.from_function(_composite_page, args=("preset",), default_timeout=60)
    at.run()

    assert not at.exception, at.exception
    assert not [b for b in at.button if b.key == "composite_gif_build"]
