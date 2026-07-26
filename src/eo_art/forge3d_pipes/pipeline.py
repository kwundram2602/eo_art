"""Orchestrates prep -> render -> export for every sweep variant."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config.loader import load_raw, to_pipeline
from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.export import encode_video, write_resolved_config
from eo_art.forge3d_pipes.prep.registry import run_prep_chain
from eo_art.forge3d_pipes.render.runner import RenderResult, render
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
    return render(cfg, prepared, out_dir)


def run(
    configs: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
    use_cache: bool = True,
    fail_fast: bool | None = None,
) -> list[VariantResult]:
    """Load, validate, and execute the pipeline for every sweep variant."""
    raw = load_raw(configs, overrides, out)
    root = to_pipeline(raw)

    source = Path(root.input.path)
    if not source.exists():
        raise FileNotFoundError(f"input raster not found: {source}")

    plan = _plan(raw, expand(root.sweep))

    run_root = Path(root.run.out_dir) / root.run.name
    cache_dir = run_root / PREP_CACHE_DIRNAME
    abort_on_error = root.run.fail_fast if fail_fast is None else fail_fast

    results: list[VariantResult] = []
    for variant, merged, cfg in plan:
        out_dir = run_root / variant.name
        try:
            rendered = _run_variant(cfg, merged, out_dir, cache_dir, use_cache)
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
