import pytest
from omegaconf import OmegaConf

from eo_art.app.config_forms import (
    apply_changes,
    changed,
    current_values,
    group_tree,
    is_active,
    iter_fields,
    merged_with_defaults,
    set_in_file,
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


def _form_state(path, cls):
    from eo_art.app.configs import load_file

    specs = [s for s in iter_fields(cls) if s.kind != "complex"]
    return current_values(merged_with_defaults(cls, load_file(path)), specs)


def test_save_edits_without_changes_leaves_file_alone(tmp_path):
    from eo_art.app.config_forms import save_edits
    from eo_art.app.templates import ACQUIRE_TEMPLATE

    path = tmp_path / "acquire.yaml"
    path.write_text(ACQUIRE_TEMPLATE)
    initial = _form_state(path, AcquireConfig)

    assert save_edits(path, AcquireConfig, initial, dict(initial)) == {}
    assert path.read_text() == ACQUIRE_TEMPLATE


def test_save_edits_writes_only_the_edited_field(tmp_path):
    from eo_art.app.config_forms import save_edits
    from eo_art.app.templates import RENDER_TEMPLATE

    path = tmp_path / "render.yaml"
    path.write_text(RENDER_TEMPLATE.format(name="alps"))
    initial = _form_state(path, PipelineConfig)

    changes = save_edits(
        path, PipelineConfig, initial, {**initial, "render.width": 640}
    )

    assert changes == {"render.width": 640}
    saved = OmegaConf.to_container(OmegaConf.load(path))
    assert saved["render"]["width"] == 640
    assert set(saved["render"]) == {"width", "height", "snapshot_name"}


def test_save_edits_rejects_schema_errors_without_writing(tmp_path):
    import pytest

    from eo_art.app.config_forms import save_edits
    from eo_art.app.templates import RENDER_TEMPLATE

    path = tmp_path / "render.yaml"
    path.write_text(RENDER_TEMPLATE.format(name="alps"))
    before = path.read_text()
    initial = _form_state(path, PipelineConfig)

    with pytest.raises(Exception):
        save_edits(path, PipelineConfig, initial, {**initial, "render.width": "wide"})
    assert path.read_text() == before


def test_render_help_covers_every_form_field_and_nothing_else():
    from eo_art.app.field_help import RENDER_HELP

    form_paths = {s.path for s in iter_fields(PipelineConfig) if s.kind != "complex"}

    assert set(RENDER_HELP) == form_paths


def test_acquire_help_covers_every_form_field_and_nothing_else():
    from eo_art.app.field_help import ACQUIRE_HELP
    from eo_art.forge3d_pipes.acquire.schema import AcquireConfig

    form_paths = {s.path for s in iter_fields(AcquireConfig) if s.kind != "complex"}

    assert set(ACQUIRE_HELP) == form_paths


def _node(nodes, path):
    for node in nodes:
        if node.path == path:
            return node
        if path.startswith(node.path + ".") or node.path == "":
            found = _node(node.children, path)
            if found is not None:
                return found
    return None


def test_group_tree_nests_groups_and_lifts_enabled_into_the_header():
    specs = iter_fields(PipelineConfig)
    tree = group_tree(specs)

    assert [n.path for n in tree] == ["run", "input", "render", "animation", "export"]
    pbr = _node(tree, "render.pbr")
    assert pbr is not None
    assert pbr.gate is not None and pbr.gate.path == "render.pbr.enabled"
    assert "render.pbr.enabled" not in [s.path for s in pbr.fields]
    assert [c.path.rpartition(".")[2] for c in pbr.children] == [
        "height_ao",
        "sun_visibility",
        "materials",
        "tonemap",
        "lens_effects",
        "sky",
    ]
    export = _node(tree, "export")
    assert export is not None and not export.fields and export.gate is None
    assert [c.path for c in export.children] == ["export.video"]


def test_group_tree_covers_every_scalar_field_once():
    for cls in (PipelineConfig, AcquireConfig):
        specs = iter_fields(cls)
        covered = [s.path for node in group_tree(specs) for s in node.all_fields()]
        scalars = [s.path for s in specs if s.kind != "complex"]
        assert sorted(covered) == sorted(scalars)


def test_group_tree_keeps_top_level_fields_in_a_general_node():
    tree = group_tree(iter_fields(AcquireConfig))

    assert tree[0].path == ""
    assert tree[0].fields and not tree[0].children
    timeseries = _node(tree, "sentinel.timeseries")
    assert timeseries is not None
    assert timeseries.gate is not None


def test_set_in_file_lists_only_keys_the_file_has():
    specs = iter_fields(PipelineConfig)
    layer = OmegaConf.create({"render": {"sun": {"azimuth": 300.0}}, "run": {}})

    assert set_in_file(layer, specs) == {"render.sun.azimuth"}


# --- list editors: prepare chains and overlays ---


def test_fixed_length_tuples_become_tuple_fields():
    from eo_art.forge3d_pipes.prep.ops import RgbStretchCfg

    bands = {s.path: s for s in iter_fields(RgbStretchCfg)}["bands"]

    assert bands.kind == "tuple"
    assert (bands.item, bands.size) == ("int", 3)
    assert bands.default == [3, 2, 1]


def test_variable_length_tuples_stay_complex():
    specs = _by_path(AcquireConfig)

    assert specs["sentinel.bands"].kind == "complex"


def test_prep_ops_lists_user_ops_but_not_internal_ones():
    from eo_art.app.config_forms import prep_ops

    ops = prep_ops()

    assert {"reproject", "rgb_stretch", "saturate"} <= set(ops)
    assert "validity_mask" not in ops


def test_clean_chain_parses_typed_numbers_and_drops_unset_params():
    from eo_art.app.config_forms import clean_chain

    chain = [
        {"op": "rgb_stretch", "bands": ["3", "2", "1"], "lower_percentile": "1.5"},
        {"op": "saturate", "factor": None},
    ]

    assert clean_chain(chain) == [
        {"op": "rgb_stretch", "bands": [3, 2, 1], "lower_percentile": 1.5},
        {"op": "saturate"},
    ]


def test_clean_chain_rejects_text_in_a_number_field():
    from eo_art.app.config_forms import clean_chain

    with pytest.raises(ValueError, match=r"prepare\[0\]\.factor"):
        clean_chain([{"op": "saturate", "factor": "lots"}])


def test_check_lists_rejects_an_overlay_without_a_path():
    from eo_art.app.config_forms import check_lists

    with pytest.raises(ValueError, match=r"overlays\[0\].*path"):
        check_lists({"overlays": [{"name": "optical"}], "prepare": []})


def test_check_lists_rejects_an_op_missing_a_required_param():
    from eo_art.app.config_forms import check_lists

    with pytest.raises(Exception, match=r"prepare\[0\]"):
        check_lists({"overlays": [], "prepare": [{"op": "reproject"}]})


def test_save_edits_replaces_a_changed_list_and_keeps_the_rest(tmp_path):
    from eo_art.app.config_forms import list_values, save_edits

    path = tmp_path / "render.yaml"
    path.write_text(
        "input: {path: out/dtm.tif}\n"
        "overlays:\n"
        "  - {name: optical, path: out/o.tif, prepare: [{op: saturate}]}\n"
    )
    initial = list_values(OmegaConf.load(path))
    edited = {
        **initial,
        "overlays": [
            {
                "name": "optical",
                "path": "out/o.tif",
                "prepare": [{"op": "rgb_stretch", "bands": [3, 2, 1]}],
            }
        ],
    }

    assert save_edits(path, PipelineConfig, initial, edited) == {
        "overlays": edited["overlays"]
    }
    saved = OmegaConf.to_container(OmegaConf.load(path))
    assert saved == {"input": {"path": "out/dtm.tif"}, "overlays": edited["overlays"]}


def test_list_help_covers_every_op_param_and_overlay_field():
    from eo_art.app.config_forms import op_fields, overlay_fields, prep_ops
    from eo_art.app.field_help import OP_HELP, OVERLAY_HELP

    assert set(OP_HELP) == set(prep_ops())
    for op in prep_ops():
        assert set(OP_HELP[op][1]) == {s.path for s in op_fields(op)}, op
    assert set(OVERLAY_HELP) == {s.path for s in overlay_fields()}


def test_clean_lists_normalises_what_a_file_may_contain():
    from eo_art.app.config_forms import clean_lists

    raw = {
        "overlays": [{"name": "o", "path": "a.tif", "opacity": None, "prepare": []}],
        "prepare": [],
    }

    assert clean_lists(raw) == {
        "overlays": [{"name": "o", "path": "a.tif"}],
        "prepare": [],
    }


def test_clean_lists_only_returns_the_lists_it_was_given():
    from eo_art.app.config_forms import clean_lists

    assert clean_lists({"prepare": [{"op": "saturate", "factor": "1.3"}]}) == {
        "prepare": [{"op": "saturate", "factor": 1.3}]
    }


def test_list_values_can_pick_one_list():
    from eo_art.app.config_forms import list_values

    cfg = OmegaConf.create({"overlays": [{"name": "o", "path": "a.tif"}]})

    assert list_values(cfg, ["overlays"]) == {
        "overlays": [{"name": "o", "path": "a.tif"}]
    }
