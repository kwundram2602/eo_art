"""Pure config-to-kwargs translation for forge3d's offscreen terrain renderer.

Forge3d-free, I/O-free -- the same rule ``payloads.py`` states for the viewer
IPC translation, and for the same reason: it is what makes this module's
mapping logic testable without a GPU. ``render/backends/offscreen.py`` (which
does import forge3d, lazily) is the only place these dicts get turned into
real ``forge3d.terrain_params`` dataclass instances.

Nested "settings" values (``shadows``, ``triplanar``, ``sampling``, ``pom``,
``tonemap``, ``materials``, ``height_ao``, ``sun_visibility``,
``lens_effects``, ``sky``, ``overlay``) are plain dicts shaped like the
corresponding ``forge3d.terrain_params`` dataclass's constructor -- field
names must match exactly, since the offscreen backend just does
``SomeSettings(**plain_dict)``.

``terrain_span`` and ``domain`` are properties of the DEM raster, not of
``PipelineConfig``, so they arrive as explicit arguments rather than this
module reaching for a raster itself.
"""

from typing import TYPE_CHECKING, Any

from eo_art.forge3d_pipes.config.schema import (
    HeightAO,
    LensEffects,
    Materials,
    Pbr,
    PipelineConfig,
    Quality,
    Sky,
    SunVisibility,
    Tonemap,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eo_art.forge3d_pipes.render.results import ResolvedOverlay

# Empirically determined on a real GPU (source-level reasoning about the
# FORGE3D_DEBUG_CAMERA trace pointed at "screen" first, and was wrong): with
# this pipeline's cam_radius/cam_phi_deg/cam_theta_deg wired straight through,
# "screen" produced a degenerate, washed-out frame that did not change at all
# when cam_radius was varied by 3x -- i.e. it is not consuming these fields as
# an orbit camera the way this pipeline needs. "mesh" matches its own
# documented defect (oblique views come out rolled/garbled -- reproduced:
# a jagged, incoherent close-up). "mesh:zup" is the one that actually works:
# rendering the same DEM at phi in {0, 90, 180} with theta=60 rotates a
# marker feature smoothly around the terrain with a level, unrolled horizon
# at every step, and theta=0 vs theta=85 give a correct top-down vs
# near-horizon framing -- exactly this pipeline's Camera.theta/phi
# convention.
_CAMERA_MODE = "mesh:zup"

# A later task wires real HDRI environment maps into the offscreen backend's
# env_maps slot (forge3d.IBL has no "disabled" constructor -- see
# offscreen.py's _placeholder_ibl). Until then, params built here always
# report IBL as disabled; ibl_intensity is still threaded through so the
# value is already correct once a real map lands.
_IBL_ENABLED = False


def _shadow_settings(pbr: Pbr) -> dict[str, Any]:
    # Mirrors make_terrain_params_config's own ShadowSettings default
    # (terrain_params.py, the `if shadows is None` branch) for every field
    # our Pbr block carries no knob for; only technique and resolution are
    # config-driven.
    return {
        "enabled": pbr.shadow_technique.value != "none",
        "technique": pbr.shadow_technique.value.upper(),
        "resolution": pbr.shadow_map_res,
        "cascades": 3,
        "max_distance": 4000.0,
        "softness": 1.5,
        "pcss_light_radius": 0.0,
        "intensity": 0.8,
        "slope_scale_bias": 0.001,
        "depth_bias": 0.0005,
        "normal_bias": 0.0002,
        "min_variance": 1e-4,
        "light_bleed_reduction": 0.5,
        "evsm_exponent": 40.0,
        "fade_start": 1.0,
    }


def _triplanar_settings(pbr: Pbr) -> dict[str, Any]:
    # scale/blend_sharpness have no config knob; mirrors
    # make_terrain_params_config's own TriplanarSettings default (scale=6.0,
    # blend_sharpness=4.0). normal_strength is the real destination for
    # Pbr.normal_strength -- MaterialSet.terrain_default's own
    # normal_strength kwarg is a *different* knob (PBR texture normals, not
    # terrain-mesh normals; see forge3d/terrain_demo.py's comment at the
    # MaterialSet.terrain_default call) and is left at forge3d's default.
    return {
        "scale": 6.0,
        "blend_sharpness": 4.0,
        "normal_strength": pbr.normal_strength,
    }


def _sampling_settings(quality: Quality) -> dict[str, Any]:
    # mag_filter/min_filter/address_* have no config knob; mirrors
    # make_terrain_params_config's own SamplingSettings default.
    return {
        "mag_filter": "Linear",
        "min_filter": "Linear",
        "mip_filter": quality.mip_filter.value.capitalize(),
        "anisotropy": quality.anisotropy,
        "address_u": "Repeat",
        "address_v": "Repeat",
        "address_w": "Repeat",
    }


def _pom_settings(quality: Quality) -> dict[str, Any]:
    # PomSettings has no field defaults of its own, so every field must be
    # supplied; only `enabled` is config-exposed (YAGNI -- see
    # Quality.pom_enabled's docstring), the rest mirror
    # make_terrain_params_config's own PomSettings default.
    return {
        "enabled": quality.pom_enabled,
        "mode": "Occlusion",
        "scale": 0.04,
        "min_steps": 12,
        "max_steps": 40,
        "refine_steps": 4,
        "shadow": True,
        "occlusion": True,
    }


def _tonemap_settings(tonemap: Tonemap) -> dict[str, Any]:
    # lut_enabled/lut_path/lut_strength have no config knob, left at
    # forge3d's defaults. TonemapOperator.linear has NO offscreen
    # equivalent: forge3d.terrain_params.TonemapSettings only accepts
    # reinhard/reinhard_extended/aces/uncharted2/exposure, no pass-through
    # "linear". Passed through as-is rather than silently remapped to the
    # nearest operator, so choosing it raises a clear ValueError at
    # kwargs-to-dataclass conversion time instead of quietly changing the
    # rendered image.
    return {
        "operator": tonemap.operator.value,
        "white_point": tonemap.white_point,
        "white_balance_enabled": tonemap.white_balance_enabled,
        "temperature": tonemap.temperature,
        "tint": tonemap.tint,
    }


def _material_settings(materials: Materials) -> dict[str, Any]:
    # Every other MaterialLayerSettings field (slope blends, colors,
    # roughness, procedural variation, ...) has no config knob and is left
    # at forge3d's defaults.
    return {
        "snow_enabled": materials.snow_enabled,
        "snow_altitude_min": materials.snow_altitude_min,
        "snow_altitude_blend": materials.snow_altitude_blend,
        "snow_slope_max": materials.snow_slope_max,
        "rock_enabled": materials.rock_enabled,
        "rock_slope_min": materials.rock_slope_min,
        "wetness_enabled": materials.wetness_enabled,
        "wetness_strength": materials.wetness_strength,
    }


def _height_ao_settings(height_ao: HeightAO) -> dict[str, Any]:
    # `blur` has no config knob, left at forge3d's default (False).
    return {
        "enabled": height_ao.enabled,
        "resolution_scale": height_ao.resolution_scale,
        "directions": height_ao.directions,
        "steps": height_ao.steps,
        "max_distance": height_ao.max_distance,
        "strength": height_ao.strength,
    }


def _sun_visibility_settings(sun_visibility: SunVisibility) -> dict[str, Any]:
    return {
        "enabled": sun_visibility.enabled,
        "mode": sun_visibility.mode.value,
        "resolution_scale": sun_visibility.resolution_scale,
        "samples": sun_visibility.samples,
        "steps": sun_visibility.steps,
        "max_distance": sun_visibility.max_distance,
        "softness": sun_visibility.softness,
        "bias": sun_visibility.bias,
    }


def _lens_effects_settings(lens_effects: LensEffects) -> dict[str, Any]:
    return {
        "enabled": lens_effects.enabled,
        "distortion": lens_effects.distortion,
        "chromatic_aberration": lens_effects.chromatic_aberration,
        "vignette_strength": lens_effects.vignette_strength,
        "vignette_radius": lens_effects.vignette_radius,
        "vignette_softness": lens_effects.vignette_softness,
    }


def _sky_settings(sky: Sky) -> dict[str, Any]:
    # model/sun_size/aerial_density have no config knob, left at forge3d's
    # defaults.
    return {
        "enabled": sky.enabled,
        "turbidity": sky.turbidity,
        "ground_albedo": sky.ground_albedo,
        "sun_intensity": sky.sun_intensity,
        "aerial_perspective": sky.aerial_perspective,
        "sky_exposure": sky.sky_exposure,
    }


def _overlay_settings(
    quality: Quality, overlays: "Sequence[ResolvedOverlay]"
) -> dict[str, Any]:
    # NOTE: as of forge3d 1.34.0, native code never reads this `overlay`
    # field for TerrainRenderer.render_terrain_pbr_pom (confirmed by reading
    # the Rust source: only the differently-shaped `overlays` kwarg --
    # forge3d.OverlayLayer / colormap gradients, an unrelated feature -- is
    # extracted; see src/terrain/render_params/private_impl.rs). This is
    # still the structurally correct field for our RGBA-texture-drape
    # overlays (matching OverlayLayerConfig's name/source/extent/opacity/
    # z_order shape), so it is wired here for forward compatibility, but
    # overlays configured for the offscreen backend will not currently
    # appear in the rendered image.
    return {
        "enabled": bool(overlays),
        "global_opacity": 1.0,
        "resolution_scale": quality.overlay_resolution_scale,
        "layers": [
            {
                "name": overlay.name,
                "source": str(overlay.path),
                "extent": tuple(overlay.extent),
                "opacity": 1.0 if overlay.opacity is None else overlay.opacity,
                "blend_mode": "normal",
                "visible": True,
                "z_order": 0 if overlay.z_order is None else overlay.z_order,
            }
            for overlay in overlays
        ],
    }


def build_terrain_params_kwargs(
    cfg: PipelineConfig,
    *,
    terrain_span: float,
    domain: tuple[float, float],
    overlays: "Sequence[ResolvedOverlay]" = (),
) -> dict[str, Any]:
    """Translate ``cfg`` into kwargs for ``make_terrain_params_config``.

    ``render.sun.ambient`` and ``render.pbr.enabled`` have no offscreen
    equivalent (no scalar ambient-light input, no single PBR-block toggle on
    ``make_terrain_params_config``) and are intentionally not mapped to
    anything -- inventing a mapping would silently change the image.
    """
    render = cfg.render
    camera = render.camera
    sun = render.sun
    pbr = render.pbr
    quality = render.quality

    return {
        "size_px": (render.width, render.height),
        "render_scale": quality.render_scale,
        "terrain_span": terrain_span,
        "msaa_samples": pbr.msaa,
        "z_scale": render.terrain.zscale,
        "exposure": pbr.exposure,
        "domain": domain,
        "cam_phi_deg": camera.phi,
        "cam_theta_deg": camera.theta,
        "cam_radius": camera.radius,
        "fov_y_deg": camera.fov,
        "camera_mode": _CAMERA_MODE,
        "light_azimuth_deg": sun.azimuth,
        "light_elevation_deg": sun.elevation,
        "sun_intensity": sun.intensity,
        "ibl_enabled": _IBL_ENABLED,
        "ibl_intensity": pbr.ibl_intensity,
        "aa_samples": quality.aa_samples,
        "shadows": _shadow_settings(pbr),
        "triplanar": _triplanar_settings(pbr),
        "sampling": _sampling_settings(quality),
        "pom": _pom_settings(quality),
        "tonemap": _tonemap_settings(pbr.tonemap),
        "materials": _material_settings(pbr.materials),
        "height_ao": _height_ao_settings(pbr.height_ao),
        "sun_visibility": _sun_visibility_settings(pbr.sun_visibility),
        "lens_effects": _lens_effects_settings(pbr.lens_effects),
        "sky": _sky_settings(pbr.sky),
        "overlay": _overlay_settings(quality, overlays),
    }
