from pathlib import Path

import pytest

from eo_art.forge3d_pipes.config.schema import (
    HeightAO,
    LensEffects,
    Materials,
    MipFilter,
    PipelineConfig,
    ShadowTechnique,
    Sky,
    SunVisibility,
    SunVisMode,
    Tonemap,
    TonemapOperator,
)
from eo_art.forge3d_pipes.render.params import build_terrain_params_kwargs
from eo_art.forge3d_pipes.render.results import ResolvedOverlay


def _cfg(**overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.input.path = "dem.tif"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _kwargs(cfg=None, **kw):
    cfg = cfg if cfg is not None else _cfg()
    return build_terrain_params_kwargs(
        cfg, terrain_span=1000.0, domain=(0.0, 100.0), **kw
    )


# ---------------------------------------------------------------------------
# Mandatory / directly-mapped fields
# ---------------------------------------------------------------------------


def test_mandatory_fields_are_all_present():
    kwargs = _kwargs()
    for key in (
        "size_px",
        "render_scale",
        "terrain_span",
        "msaa_samples",
        "z_scale",
        "exposure",
        "domain",
    ):
        assert key in kwargs


def test_size_px_from_render_width_height():
    cfg = _cfg()
    cfg.render.width = 640
    cfg.render.height = 480
    assert _kwargs(cfg)["size_px"] == (640, 480)


def test_terrain_span_and_domain_are_passed_through_as_given():
    kwargs = build_terrain_params_kwargs(
        _cfg(), terrain_span=12345.0, domain=(10.0, 20.0)
    )
    assert kwargs["terrain_span"] == 12345.0
    assert kwargs["domain"] == (10.0, 20.0)


def test_camera_fields_mapped_from_config_camera():
    cfg = _cfg()
    cfg.render.camera.phi = 111.0
    cfg.render.camera.theta = 22.0
    cfg.render.camera.radius = 5000.0
    cfg.render.camera.fov = 45.0
    kwargs = _kwargs(cfg)
    assert kwargs["cam_phi_deg"] == 111.0
    assert kwargs["cam_theta_deg"] == 22.0
    assert kwargs["cam_radius"] == 5000.0
    assert kwargs["fov_y_deg"] == 45.0


def test_camera_mode_is_screen():
    # See params.py's _CAMERA_MODE comment: "screen" reproduces the
    # interactive viewer's camera math, which is what phi/theta/radius/fov
    # already target in this pipeline.
    assert _kwargs()["camera_mode"] == "screen"


def test_sun_fields_mapped():
    cfg = _cfg()
    cfg.render.sun.azimuth = 200.0
    cfg.render.sun.elevation = 33.0
    cfg.render.sun.intensity = 2.5
    kwargs = _kwargs(cfg)
    assert kwargs["light_azimuth_deg"] == 200.0
    assert kwargs["light_elevation_deg"] == 33.0
    assert kwargs["sun_intensity"] == 2.5


def test_z_scale_from_terrain_zscale():
    cfg = _cfg()
    cfg.render.terrain.zscale = 7.5
    assert _kwargs(cfg)["z_scale"] == 7.5


def test_exposure_and_msaa_from_pbr():
    cfg = _cfg()
    cfg.render.pbr.exposure = 2.0
    cfg.render.pbr.msaa = 4
    kwargs = _kwargs(cfg)
    assert kwargs["exposure"] == 2.0
    assert kwargs["msaa_samples"] == 4


def test_ibl_intensity_from_pbr_ibl_intensity():
    cfg = _cfg()
    cfg.render.pbr.ibl_intensity = 3.3
    assert _kwargs(cfg)["ibl_intensity"] == 3.3


def test_ibl_enabled_is_false_pending_hdri_task():
    # No config field drives this yet -- a later task wires real HDRI maps.
    assert _kwargs()["ibl_enabled"] is False


def test_aa_samples_from_quality():
    cfg = _cfg()
    cfg.render.quality.aa_samples = 64
    assert _kwargs(cfg)["aa_samples"] == 64


# ---------------------------------------------------------------------------
# Shadows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("technique", "expected"),
    [
        (ShadowTechnique.pcss, "PCSS"),
        (ShadowTechnique.pcf, "PCF"),
        (ShadowTechnique.none, "NONE"),
    ],
)
def test_shadow_technique_uppercased(technique, expected):
    cfg = _cfg()
    cfg.render.pbr.shadow_technique = technique
    assert _kwargs(cfg)["shadows"]["technique"] == expected


def test_shadow_disabled_when_technique_is_none():
    cfg = _cfg()
    cfg.render.pbr.shadow_technique = ShadowTechnique.none
    assert _kwargs(cfg)["shadows"]["enabled"] is False


def test_shadow_enabled_when_technique_is_not_none():
    cfg = _cfg()
    cfg.render.pbr.shadow_technique = ShadowTechnique.pcf
    assert _kwargs(cfg)["shadows"]["enabled"] is True


def test_shadow_resolution_from_shadow_map_res():
    cfg = _cfg()
    cfg.render.pbr.shadow_map_res = 2048
    assert _kwargs(cfg)["shadows"]["resolution"] == 2048


# ---------------------------------------------------------------------------
# Triplanar / normal strength
# ---------------------------------------------------------------------------


def test_triplanar_normal_strength_from_pbr():
    cfg = _cfg()
    cfg.render.pbr.normal_strength = 1.8
    assert _kwargs(cfg)["triplanar"]["normal_strength"] == 1.8


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampling_anisotropy_from_quality():
    cfg = _cfg()
    cfg.render.quality.anisotropy = 4
    assert _kwargs(cfg)["sampling"]["anisotropy"] == 4


@pytest.mark.parametrize(
    ("mip_filter", "expected"),
    [(MipFilter.linear, "Linear"), (MipFilter.nearest, "Nearest")],
)
def test_sampling_mip_filter_capitalized(mip_filter, expected):
    cfg = _cfg()
    cfg.render.quality.mip_filter = mip_filter
    assert _kwargs(cfg)["sampling"]["mip_filter"] == expected


# ---------------------------------------------------------------------------
# POM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False])
def test_pom_enabled_from_quality(enabled):
    cfg = _cfg()
    cfg.render.quality.pom_enabled = enabled
    assert _kwargs(cfg)["pom"]["enabled"] is enabled


# ---------------------------------------------------------------------------
# Tonemap
# ---------------------------------------------------------------------------


def test_tonemap_fields_mapped():
    cfg = _cfg()
    cfg.render.pbr.tonemap = Tonemap(
        operator=TonemapOperator.reinhard,
        white_point=5.0,
        white_balance_enabled=True,
        temperature=5500.0,
        tint=0.1,
    )
    tonemap = _kwargs(cfg)["tonemap"]
    assert tonemap["operator"] == "reinhard"
    assert tonemap["white_point"] == 5.0
    assert tonemap["white_balance_enabled"] is True
    assert tonemap["temperature"] == 5500.0
    assert tonemap["tint"] == 0.1


def test_tonemap_linear_passes_through_unmapped():
    # TonemapOperator.linear has no forge3d equivalent; params.py must not
    # silently remap it to something else.
    cfg = _cfg()
    cfg.render.pbr.tonemap = Tonemap(operator=TonemapOperator.linear)
    assert _kwargs(cfg)["tonemap"]["operator"] == "linear"


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def test_materials_fields_mapped():
    cfg = _cfg()
    cfg.render.pbr.materials = Materials(
        snow_enabled=False,
        snow_altitude_min=1000.0,
        snow_altitude_blend=100.0,
        snow_slope_max=30.0,
        rock_enabled=False,
        rock_slope_min=20.0,
        wetness_enabled=True,
        wetness_strength=0.7,
    )
    materials = _kwargs(cfg)["materials"]
    assert materials == {
        "snow_enabled": False,
        "snow_altitude_min": 1000.0,
        "snow_altitude_blend": 100.0,
        "snow_slope_max": 30.0,
        "rock_enabled": False,
        "rock_slope_min": 20.0,
        "wetness_enabled": True,
        "wetness_strength": 0.7,
    }


# ---------------------------------------------------------------------------
# Height AO / sun visibility / lens effects / sky
# ---------------------------------------------------------------------------


def test_height_ao_fields_mapped():
    cfg = _cfg()
    cfg.render.pbr.height_ao = HeightAO(
        enabled=True,
        directions=8,
        steps=32,
        max_distance=300.0,
        strength=1.5,
        resolution_scale=0.75,
    )
    height_ao = _kwargs(cfg)["height_ao"]
    assert height_ao == {
        "enabled": True,
        "resolution_scale": 0.75,
        "directions": 8,
        "steps": 32,
        "max_distance": 300.0,
        "strength": 1.5,
    }


def test_sun_visibility_fields_mapped():
    cfg = _cfg()
    cfg.render.pbr.sun_visibility = SunVisibility(
        enabled=True,
        mode=SunVisMode.hard,
        samples=8,
        steps=48,
        max_distance=500.0,
        softness=0.5,
        bias=0.02,
        resolution_scale=0.25,
    )
    sun_vis = _kwargs(cfg)["sun_visibility"]
    assert sun_vis == {
        "enabled": True,
        "mode": "hard",
        "resolution_scale": 0.25,
        "samples": 8,
        "steps": 48,
        "max_distance": 500.0,
        "softness": 0.5,
        "bias": 0.02,
    }


def test_lens_effects_fields_mapped():
    cfg = _cfg()
    cfg.render.pbr.lens_effects = LensEffects(
        enabled=True,
        distortion=0.2,
        chromatic_aberration=0.1,
        vignette_strength=0.5,
        vignette_radius=0.6,
        vignette_softness=0.4,
    )
    lens = _kwargs(cfg)["lens_effects"]
    assert lens == {
        "enabled": True,
        "distortion": 0.2,
        "chromatic_aberration": 0.1,
        "vignette_strength": 0.5,
        "vignette_radius": 0.6,
        "vignette_softness": 0.4,
    }


def test_sky_fields_mapped():
    cfg = _cfg()
    cfg.render.pbr.sky = Sky(
        enabled=False,
        turbidity=4.0,
        ground_albedo=0.5,
        sun_intensity=1.5,
        aerial_perspective=False,
        sky_exposure=1.2,
    )
    sky = _kwargs(cfg)["sky"]
    assert sky == {
        "enabled": False,
        "turbidity": 4.0,
        "ground_albedo": 0.5,
        "sun_intensity": 1.5,
        "aerial_perspective": False,
        "sky_exposure": 1.2,
    }


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def test_overlay_disabled_and_empty_when_no_overlays():
    overlay = _kwargs(overlays=[])["overlay"]
    assert overlay["enabled"] is False
    assert overlay["layers"] == []


def test_overlay_resolution_scale_from_quality():
    cfg = _cfg()
    cfg.render.quality.overlay_resolution_scale = 0.5
    assert _kwargs(cfg)["overlay"]["resolution_scale"] == 0.5


def test_overlay_enabled_and_layers_built_from_resolved_overlays():
    overlays = [
        ResolvedOverlay(
            name="ndvi",
            path=Path("ndvi.png"),
            extent=(0.1, 0.2, 0.8, 0.9),
            opacity=0.6,
            z_order=2,
        ),
        ResolvedOverlay(
            name="mask",
            path=Path("mask.png"),
            extent=(0.0, 0.0, 1.0, 1.0),
        ),
    ]
    overlay = _kwargs(overlays=overlays)["overlay"]
    assert overlay["enabled"] is True
    assert overlay["layers"] == [
        {
            "name": "ndvi",
            "source": "ndvi.png",
            "extent": (0.1, 0.2, 0.8, 0.9),
            "opacity": 0.6,
            "blend_mode": "normal",
            "visible": True,
            "z_order": 2,
        },
        {
            "name": "mask",
            "source": "mask.png",
            "extent": (0.0, 0.0, 1.0, 1.0),
            "opacity": 1.0,
            "blend_mode": "normal",
            "visible": True,
            "z_order": 0,
        },
    ]


# ---------------------------------------------------------------------------
# Real forge3d construction (pure Python, no GPU/Session touched -- same
# precedent as test_animation.py's use of forge3d.animation.CameraAnimation)
# ---------------------------------------------------------------------------


def test_kwargs_construct_a_real_terrain_render_params():
    from forge3d.terrain_params import make_terrain_params_config

    from eo_art.forge3d_pipes.render.backends.offscreen import _settings_kwargs

    params = make_terrain_params_config(**_settings_kwargs(_kwargs()))
    assert params.size_px == (1200, 720)
    assert params.terrain_span == 1000.0
    assert params.camera_mode == "screen"


@pytest.mark.parametrize(
    "override",
    [
        {"size_px": (8193, 720)},
        {"render_scale": 0.24},
        {"render_scale": 4.01},
        # forge3d itself only bounds aa_samples to [1, 4096] -- the discrete
        # {1, 16, 64, 256} set is *our* schema's stricter choice (see
        # test_schema.py's aa_samples=32 boundary test), so 32 alone
        # wouldn't be rejected here.
        {"aa_samples": 4097},
    ],
)
def test_out_of_range_boundary_values_rejected_by_forge3d(override):
    from forge3d.terrain_params import make_terrain_params_config

    from eo_art.forge3d_pipes.render.backends.offscreen import _settings_kwargs

    kwargs = _kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError):
        make_terrain_params_config(**_settings_kwargs(kwargs))


def test_anisotropy_out_of_range_rejected_by_forge3d():
    from forge3d.terrain_params import make_terrain_params_config

    from eo_art.forge3d_pipes.render.backends.offscreen import _settings_kwargs

    kwargs = _kwargs()
    kwargs["sampling"] = dict(kwargs["sampling"], anisotropy=17)
    with pytest.raises(ValueError):
        make_terrain_params_config(**_settings_kwargs(kwargs))
