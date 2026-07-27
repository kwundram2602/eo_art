"""Orchestrates prep -> render -> export for every sweep variant."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config.loader import load_raw, to_pipeline
from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.export import encode_video, write_resolved_config
from eo_art.forge3d_pipes.prep.extent import compute_normalized_extent
from eo_art.forge3d_pipes.prep.overlay_image import export_overlay_png_cached
from eo_art.forge3d_pipes.prep.registry import run_prep_chain
from eo_art.forge3d_pipes.render.runner import RenderResult, ResolvedOverlay, render
from eo_art.forge3d_pipes.sweep import Variant, expand

PREP_CACHE_DIRNAME = "_prep"
RESOLVED_CONFIG_NAME = "resolved.yaml"


@dataclass(frozen=True)
class VariantResult:
    name: str
    out_dir: Path
    ok: bool
    error: str | None = None
    snapshot: Path | None = None
    frames_dir: Path | None = None
    video: Path | None = None


def _plan(
    raw: DictConfig, variants: list[Variant]
) -> list[tuple[Variant, DictConfig, PipelineConfig]]:
    """Validate every variant up front, so nothing runs on a broken plan."""
    plan = []
    for variant in variants:
        # OmegaConf.merge's stub types its return as `ListConfig | DictConfig`;
        # merging dotlist overrides onto a DictConfig always yields a
        # DictConfig in practice.
        merged = cast(
            DictConfig,
            OmegaConf.merge(raw, OmegaConf.from_dotlist(list(variant.overrides)))
            if variant.overrides
            else raw,
        )
        plan.append((variant, merged, to_pipeline(merged)))
    return plan


def _run_variant(
    cfg: PipelineConfig,
    merged: DictConfig,
    out_dir: Path,
    cache_dir: Path,
    use_cache: bool,
) -> RenderResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(merged, out_dir / RESOLVED_CONFIG_NAME)

    prepared = run_prep_chain(
        Path(cfg.input.path), cfg.prepare, cache_dir, use_cache=use_cache
    )

    resolved_overlays = []
    for overlay in cfg.overlays:
        overlay_prepared = run_prep_chain(
            Path(overlay.path), overlay.prepare, cache_dir, use_cache=use_cache
        )
        extent = (
            tuple(overlay.extent)
            if overlay.extent is not None
            else compute_normalized_extent(prepared, overlay_prepared)
        )
        # forge3d's live-viewer load_overlay reads the file through Rust's
        # `image` crate, which does not support TIFF; every prep op writes
        # GeoTIFF, so the final prepped raster must be re-exported as a PNG.
        overlay_png = export_overlay_png_cached(
            overlay_prepared, cache_dir, use_cache=use_cache
        )
        resolved_overlays.append(
            ResolvedOverlay(
                name=overlay.name,
                path=overlay_png,
                extent=extent,
                opacity=overlay.opacity,
                z_order=overlay.z_order,
            )
        )

    return render(cfg, prepared, out_dir, overlays=resolved_overlays)


def run(
    configs: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
    use_cache: bool | None = None,
    fail_fast: bool | None = None,
) -> list[VariantResult]:
    """Load, validate, and execute the pipeline for every sweep variant.

    ``use_cache`` and ``fail_fast`` default to ``None``, meaning "use the
    value from ``run.cache`` / ``run.fail_fast`` in the config"; an explicit
    boolean overrides the config.
    """
    raw = load_raw(configs, overrides, out)
    root = to_pipeline(raw)

    source = Path(root.input.path)
    if not source.exists():
        raise FileNotFoundError(f"input raster not found: {source}")

    plan = _plan(raw, expand(root.sweep))

    run_root = Path(root.run.out_dir) / root.run.name
    cache_dir = run_root / PREP_CACHE_DIRNAME
    abort_on_error = root.run.fail_fast if fail_fast is None else fail_fast
    cache_enabled = root.run.cache if use_cache is None else use_cache

    results: list[VariantResult] = []
    for variant, merged, cfg in plan:
        out_dir = run_root / variant.name
        try:
            rendered = _run_variant(cfg, merged, out_dir, cache_dir, cache_enabled)
            video = None
            if cfg.export.video.enabled and rendered.frames_dir is not None:
                video = encode_video(
                    rendered.frames_dir,
                    out_dir / f"video.{cfg.export.video.format.value}",
                    cfg.export.video,
                )
            results.append(
                VariantResult(
                    name=variant.name,
                    out_dir=out_dir,
                    ok=True,
                    snapshot=rendered.snapshot,
                    frames_dir=rendered.frames_dir,
                    video=video,
                )
            )
        except Exception as exc:
            if abort_on_error:
                raise
            results.append(
                VariantResult(
                    name=variant.name, out_dir=out_dir, ok=False, error=str(exc)
                )
            )
    return results
