from eo_art.app.configs import (
    ConfigKind,
    detect_kind,
    overlay_names,
    set_top_level_value,
    validate_acquire,
    validate_render,
    validate_sr,
)
from eo_art.app.templates import ACQUIRE_TEMPLATE, RENDER_TEMPLATE


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_detect_kind(tmp_path):
    assert detect_kind(_write(tmp_path, "a.yaml", ACQUIRE_TEMPLATE)) is (
        ConfigKind.acquire
    )
    assert detect_kind(_write(tmp_path, "r.yaml", RENDER_TEMPLATE)) is (
        ConfigKind.render
    )
    assert detect_kind(_write(tmp_path, "e.yaml", "")) is ConfigKind.render


def test_templates_validate(tmp_path):
    render = _write(tmp_path, "r.yaml", RENDER_TEMPLATE.format(name="x"))
    acquire = _write(tmp_path, "a.yaml", ACQUIRE_TEMPLATE)

    assert validate_render([render]).ok
    assert validate_acquire(acquire).ok
    assert validate_sr(acquire).ok
    assert overlay_names([render]) == ["optical"]


def test_render_layers_merge_left_to_right(tmp_path):
    base = _write(tmp_path, "r.yaml", RENDER_TEMPLATE.format(name="x"))
    look = _write(tmp_path, "look.yaml", "render: {width: 640}\n")

    result = validate_render([base, look])

    assert result.ok
    assert "width: 640" in result.resolved


def test_range_error_is_reported_with_resolved_config(tmp_path):
    base = _write(tmp_path, "r.yaml", RENDER_TEMPLATE.format(name="x"))
    bad = _write(tmp_path, "bad.yaml", "render: {camera: {fov: 500}}\n")

    result = validate_render([base, bad])

    assert not result.ok
    assert "camera.fov" in result.error
    assert result.resolved is not None


def test_type_error_is_reported_without_raising(tmp_path):
    bad = _write(tmp_path, "bad.yaml", "render: {width: wide}\n")

    result = validate_render([bad])

    assert not result.ok
    assert result.resolved is None


def test_acquire_missing_ee_project_is_reported(tmp_path):
    path = _write(
        tmp_path,
        "a.yaml",
        ACQUIRE_TEMPLATE.replace("ee_project: your-ee-project\n", ""),
    )

    result = validate_acquire(path)

    assert not result.ok
    assert "ee_project" in result.error
    # SR only needs the sentinel slice, so it still validates.
    assert validate_sr(path).ok


def test_set_top_level_value_keeps_comments(tmp_path):
    path = _write(
        tmp_path,
        "a.yaml",
        "# header\naoi_path: aoi.gpkg   # override me\nsentinel:\n  aoi_path: x\n",
    )

    set_top_level_value(path, "aoi_path", "aoi/new one.gpkg")

    assert path.read_text() == (
        '# header\naoi_path: "aoi/new one.gpkg"   # override me\n'
        "sentinel:\n  aoi_path: x\n"
    )


def test_set_top_level_value_appends_missing_key(tmp_path):
    path = _write(tmp_path, "a.yaml", "out_dir: out")

    set_top_level_value(path, "aoi_path", "aoi/a.gpkg")

    assert path.read_text() == 'out_dir: out\naoi_path: "aoi/a.gpkg"\n'
