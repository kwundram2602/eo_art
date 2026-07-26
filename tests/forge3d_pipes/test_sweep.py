import pytest

from eo_art.forge3d_pipes.config.schema import Sweep, SweepMode
from eo_art.forge3d_pipes.sweep import Variant, expand


def test_no_sweep_yields_single_default_variant():
    assert expand(None) == [Variant(name="default", overrides=())]


def test_empty_params_yields_single_default_variant():
    assert expand(Sweep()) == [Variant(name="default", overrides=())]


def test_product_yields_cartesian_grid():
    variants = expand(
        Sweep(
            mode=SweepMode.product,
            params={"render.pbr.exposure": [1.0, 2.0], "render.camera.phi": [10, 20]},
        )
    )
    assert len(variants) == 4
    assert [v.overrides for v in variants] == [
        ("render.pbr.exposure=1.0", "render.camera.phi=10"),
        ("render.pbr.exposure=1.0", "render.camera.phi=20"),
        ("render.pbr.exposure=2.0", "render.camera.phi=10"),
        ("render.pbr.exposure=2.0", "render.camera.phi=20"),
    ]


def test_zip_walks_lists_in_lockstep():
    variants = expand(
        Sweep(
            mode=SweepMode.zip,
            params={"render.pbr.exposure": [1.0, 2.0], "render.camera.phi": [10, 20]},
        )
    )
    assert len(variants) == 2
    assert variants[0].overrides == ("render.pbr.exposure=1.0", "render.camera.phi=10")
    assert variants[1].overrides == ("render.pbr.exposure=2.0", "render.camera.phi=20")


def test_zip_rejects_length_mismatch():
    with pytest.raises(ValueError, match="zip sweep requires equal-length"):
        expand(
            Sweep(
                mode=SweepMode.zip,
                params={"a.b": [1, 2, 3], "c.d": [1, 2]},
            )
        )


def test_variant_names_describe_their_params():
    variants = expand(
        Sweep(params={"render.pbr.exposure": [1.35], "render.camera.phi": [280]})
    )
    assert variants[0].name == "exposure=1.35__phi=280"


def test_variant_names_are_filesystem_safe():
    variants = expand(Sweep(params={"input.path": ["/data/a b.tif"]}))
    assert variants[0].name == "path=_data_a_b.tif"


def test_variant_names_are_unique():
    variants = expand(Sweep(params={"a.x": [1, 2], "b.x": [3, 4]}))
    assert len({v.name for v in variants}) == len(variants)


def test_non_list_value_is_rejected():
    with pytest.raises(ValueError, match="must be a list"):
        expand(Sweep(params={"render.camera.phi": 10}))


def test_slug_aliasing_collisions_get_unique_names():
    """`a b` and `a_b` slug identically; variants must not share a directory."""
    variants = expand(Sweep(params={"input.path": ["a b", "a_b"]}))
    assert len(variants) == 2
    assert len({v.name for v in variants}) == 2
    # Overrides stay faithful to the original values.
    assert variants[0].overrides == ("input.path=a b",)
    assert variants[1].overrides == ("input.path=a_b",)


def test_duplicate_values_in_one_list_get_unique_names():
    variants = expand(Sweep(params={"render.camera.phi": [10, 10]}))
    assert len({v.name for v in variants}) == 2
