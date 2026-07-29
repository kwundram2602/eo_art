"""Drives forge3d's offscreen terrain renderer: build params, render, save."""

import math
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eo_art.forge3d_pipes.config.schema import GpuPreference, PipelineConfig
from eo_art.forge3d_pipes.render.params import build_terrain_params_kwargs
from eo_art.forge3d_pipes.render.results import RenderResult, ResolvedOverlay

# 18%-gray, the standard photographic neutral reference -- picked only so the
# placeholder file's content is a defensible, well-known constant rather than
# an arbitrary number.
_PLACEHOLDER_IBL_GRAY = 0.18


def _rgbe_encode(value: float) -> bytes:
    """Encode a single gray value as one Radiance RGBE texel (4 bytes)."""
    if value < 1e-32:
        return bytes((0, 0, 0, 0))
    mantissa, exponent = math.frexp(value)
    scale = mantissa * 256.0 / value
    channel = min(255, max(0, int(value * scale)))
    return bytes((channel, channel, channel, exponent + 128))


def _write_placeholder_hdr(path: str, width: int = 8, height: int = 4) -> None:
    """Write a tiny flat-gray Radiance .hdr file (uncompressed scanlines).

    Format verified against forge3d's own loader
    (``src/formats/hdr.rs::parse_header`` / ``read_scanline``): magic line,
    a ``FORMAT=`` line, a blank line, a ``-Y height +X width`` resolution
    line, then ``height`` scanlines of raw 4-byte RGBE texels (the "old
    style" path -- taken whenever a scanline's first 4 bytes don't match the
    new-style RLE marker, which a flat gray texel never does).
    """
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n"
    pixel = _rgbe_encode(_PLACEHOLDER_IBL_GRAY)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(pixel * (width * height))


def _placeholder_ibl() -> Any:
    """Construct a neutral, effectively no-op IBL environment map.

    ``forge3d.IBL`` has no "disabled"/empty constructor -- ``IBL.from_hdr``
    always reads a real Radiance .hdr or .exr file from disk (see
    ``src/lighting/ibl_wrapper.rs``), and
    ``TerrainRenderer.render_terrain_pbr_pom``'s ``env_maps`` parameter is
    not optional. A later task wires a real HDRI into this slot (see
    ``render/params.py``'s ``_IBL_ENABLED`` note); until then, a tiny flat
    placeholder is generated on demand and loaded with ``intensity=0.0`` on
    top of the render params' own ``ibl_enabled=False``, so it contributes
    nothing regardless of which of the two the native renderer honors.
    """
    import forge3d

    fd, path = tempfile.mkstemp(suffix=".hdr", prefix="eo_art_placeholder_ibl_")
    try:
        os.close(fd)
        _write_placeholder_hdr(path)
        return forge3d.IBL.from_hdr(path, intensity=0.0)
    finally:
        os.remove(path)


def _settings_kwargs(plain: dict[str, Any]) -> dict[str, Any]:
    """Turn ``render/params.py``'s plain-dict kwargs into real forge3d dataclasses."""
    from forge3d.terrain_params import (
        HeightAoSettings,
        LensEffectsSettings,
        MaterialLayerSettings,
        OverlayLayerConfig,
        OverlaySettings,
        PomSettings,
        SamplingSettings,
        ShadowSettings,
        SkySettings,
        SunVisibilitySettings,
        TonemapSettings,
        TriplanarSettings,
    )

    kwargs = dict(plain)
    kwargs["shadows"] = ShadowSettings(**plain["shadows"])
    kwargs["triplanar"] = TriplanarSettings(**plain["triplanar"])
    kwargs["sampling"] = SamplingSettings(**plain["sampling"])
    kwargs["pom"] = PomSettings(**plain["pom"])
    kwargs["tonemap"] = TonemapSettings(**plain["tonemap"])
    kwargs["materials"] = MaterialLayerSettings(**plain["materials"])
    kwargs["height_ao"] = HeightAoSettings(**plain["height_ao"])
    kwargs["sun_visibility"] = SunVisibilitySettings(**plain["sun_visibility"])
    kwargs["lens_effects"] = LensEffectsSettings(**plain["lens_effects"])
    kwargs["sky"] = SkySettings(**plain["sky"])

    overlay = plain["overlay"]
    kwargs["overlay"] = OverlaySettings(
        enabled=overlay["enabled"],
        global_opacity=overlay["global_opacity"],
        resolution_scale=overlay["resolution_scale"],
        layers=[OverlayLayerConfig(**layer) for layer in overlay["layers"]],
    )
    return kwargs


def _open_session(cfg: PipelineConfig) -> Any:
    """Open forge3d's offscreen session, honouring ``run.gpu``.

    ``run.gpu: off`` means "touch nothing" -- ``gpu.select_gpu`` itself
    changes no environment variables for that preference (see its own
    early return). Forcing a backend here unconditionally would defeat
    that intent from a different angle, so ``off`` is the one case where
    Vulkan is *not* forced: wgpu is left to auto-select across backends,
    exactly as it did before Vulkan was ever forced for this renderer.

    Every other preference forces ``backend="vulkan"``: it is the only
    backend ``select_gpu``'s ICD-environment steering can reach (see
    ``gpu.py``'s module docstring and ``_render_frame`` below for how this
    was discovered) -- without it, ``Session(window=False)`` auto-selects
    across *all* wgpu backends and was observed to silently prefer the
    integrated GPU via the GL backend on a machine with both an Intel iGPU
    and an NVIDIA discrete GPU, which both ignored the ICD steering and
    rendered a blank black frame instead of raising.
    """
    import forge3d

    if cfg.run.gpu is GpuPreference.off:
        return forge3d.Session(window=False)

    try:
        return forge3d.Session(window=False, backend="vulkan")
    except RuntimeError as exc:
        raise RuntimeError(
            "render: the offscreen backend forces the Vulkan wgpu backend "
            "so that run.gpu's adapter steering (gpu.select_gpu) actually "
            f"reaches the renderer, but no Vulkan driver was found ({exc}). "
            "Either install a Vulkan driver, or set run.gpu: off to skip "
            "GPU steering entirely and let wgpu auto-select a backend."
        ) from exc


def _render_frame(
    cfg: PipelineConfig, terrain_path: Path, overlays: Sequence[ResolvedOverlay]
) -> Any:
    """Indirection point so tests can substitute a fake renderer without a GPU.

    Import kept lazy: see ``backends/viewer.py``'s ``_open_viewer`` docstring
    -- forge3d must not initialise its native Vulkan instance before
    ``gpu.select_gpu`` sets the ICD environment variables.
    """
    import forge3d
    from forge3d import io as forge3d_io
    from forge3d.terrain_params import make_terrain_params_config

    dem = forge3d_io.load_dem(str(terrain_path))
    heightmap = dem.data
    dx, dy = dem.resolution
    height_px, width_px = heightmap.shape[:2]
    terrain_span = max(dx * width_px, dy * height_px)
    domain = forge3d_io.robust_dem_domain(
        heightmap,
        q_lo=0.0,
        q_hi=1.0,
        fallback=forge3d_io.infer_dem_domain(dem, fallback=(0.0, 1.0)),
    )

    plain_kwargs = build_terrain_params_kwargs(
        cfg, terrain_span=terrain_span, domain=domain, overlays=overlays
    )
    config = make_terrain_params_config(**_settings_kwargs(plain_kwargs))
    # make_terrain_params_config returns the pure-Python config dataclass;
    # render_terrain_pbr_pom wants the *native* TerrainRenderParams built
    # from it -- forge3d's own canonical wiring does the same wrap
    # (terrain_demo.py: `return f3d.TerrainRenderParams(config)`).
    params = forge3d.TerrainRenderParams(config)

    session = _open_session(cfg)
    renderer = forge3d.TerrainRenderer(session)
    # triplanar_scale/blend_sharpness/normal_strength here are the
    # MaterialSet's own PBR-texture knobs, distinct from the terrain-mesh
    # TriplanarSettings built in _settings_kwargs; forge3d's own canonical
    # wiring (terrain_demo.py) keeps normal_strength=1.0 fixed here too.
    materials = forge3d.MaterialSet.terrain_default(
        triplanar_scale=6.0, normal_strength=1.0, blend_sharpness=4.0
    )
    env_maps = _placeholder_ibl()

    return renderer.render_terrain_pbr_pom(
        material_set=materials,
        env_maps=env_maps,
        params=params,
        heightmap=heightmap,
        target=None,
    )


def render(
    cfg: PipelineConfig,
    terrain_path: Path,
    out_dir: Path,
    overlays: Sequence[ResolvedOverlay] = (),
) -> RenderResult:
    """Render a still into ``out_dir`` via forge3d's offscreen renderer.

    Animation is not supported here; ``PipelineConfig.__post_init__``
    rejects ``backend=offscreen`` combined with ``animation.kind != none``
    at config-load time, so this never has to build a frame sequence.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _render_frame(cfg, terrain_path, overlays)
    snapshot = out_dir / cfg.render.snapshot_name
    frame.save(str(snapshot))
    return RenderResult(snapshot=snapshot)
