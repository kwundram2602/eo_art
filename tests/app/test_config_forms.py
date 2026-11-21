from omegaconf import OmegaConf

from eo_art.app.config_forms import (
    apply_changes,
    changed,
    current_values,
    is_active,
    iter_fields,
    merged_with_defaults,
)
from eo_art.forge3d_pipes.acquire.schema import AcquireConfig
from eo_art.forge3d_pipes.config.schema import PipelineConfig


def _by_path(cls):
    return {spec.path: spec for spec in iter_fields(cls)}


def test_render_schema_walk_covers_nested_scalars_and_enums():
    specs = _by_path(PipelineConfig)

    assert specs["render.camera.phi"].kind == "float"
    assert specs["render.width"].kind == "int"
    assert specs["render.pbr.height_ao.enabled"].kind == "bool"
    tonemap = specs["render.pbr.tonemap.operator"]
    assert tonemap.kind == "enum"
    assert tonemap.choices == ("aces", "reinhard", "linear")
    assert tonemap.default == "aces"
    assert specs["input.path"].required
    assert specs["animation.orbit.theta_end"].optional
    for complex_path in ("prepare", "overlays", "sweep"):
        assert specs[complex_path].kind == "complex"


def test_acquire_schema_walk():
    specs = _by_path(AcquireConfig)

    assert specs["aoi_path"].required
    assert specs["ee_project"].optional and specs["ee_project"].kind == "str"
    assert specs["dem.source"].choices == ("copernicus", "fabdem", "srtm", "local")
    assert specs["sentinel.bands"].kind == "complex"
    assert specs["sentinel.max_items"].kind == "int"


def test_form_edit_writes_only_changed_keys():
    layer = OmegaConf.create({"render": {"sun": {"azimuth": 285.0}}})
    specs = iter_fields(PipelineConfig)
    initial = current_values(merged_with_defaults(PipelineConfig, layer), specs)
    assert initial["render.sun.azimuth"] == 285.0
    assert initial["render.pbr.tonemap.operator"] == "aces"

    edited = dict(initial)
    edited["render.pbr.tonemap.operator"] = "reinhard"
    edited["render.width"] = 640

    saved = apply_changes(layer, changed(initial, edited))

    assert OmegaConf.to_container(saved) == {
        "render": {
            "sun": {"azimuth": 285.0},
            "pbr": {"tonemap": {"operator": "reinhard"}},
            "width": 640,
        }
    }
    # The edited layer still merges cleanly against the schema.
    merged = merged_with_defaults(PipelineConfig, saved)
    assert merged.render.pbr.tonemap.operator.name == "reinhard"


def test_apply_changes_does_not_mutate_input():
    layer = OmegaConf.create({"render": {"width": 10}})
    apply_changes(layer, {"render.width": 20})
    assert layer.render.width == 10


def test_enabled_field_gates_its_siblings_and_nested_groups():
    specs = _by_path(AcquireConfig)
    render = _by_path(PipelineConfig)

    assert specs["sentinel.timeseries.enabled"].gate is None
    assert specs["sentinel.timeseries.interval"].gate == "sentinel.timeseries.enabled"
    assert specs["sentinel.timeseries.every"].gate == "sentinel.timeseries.enabled"
    assert specs["sentinel.start_date"].gate is None
    # Nested groups inherit the outer gate; their own `enabled` is gated by it.
    assert render["render.pbr.height_ao.enabled"].gate == "render.pbr.enabled"
    assert render["render.pbr.height_ao.steps"].gate == "render.pbr.height_ao.enabled"


def test_is_active_follows_the_gate_chain():
    specs = iter_fields(PipelineConfig)
    steps = next(s for s in specs if s.path == "render.pbr.height_ao.steps")
    on = {"render.pbr.enabled": True, "render.pbr.height_ao.enabled": True}

    assert is_active(steps, on, specs)
    assert not is_active(steps, {**on, "render.pbr.height_ao.enabled": False}, specs)
    assert not is_active(steps, {**on, "render.pbr.enabled": False}, specs)
