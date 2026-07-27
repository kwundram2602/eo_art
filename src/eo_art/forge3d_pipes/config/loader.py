"""Config loading: merge defaults, files, and CLI overrides, then validate."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.prep.registry import validate_chain


def load_raw(
    paths: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
) -> DictConfig:
    """Merge schema defaults, config files, and dotlist overrides.

    Merge order, later wins: defaults -> files (in order) -> dotlist.
    ``out`` is applied in the dotlist layer, so it beats the files.
    """
    cfg = OmegaConf.structured(PipelineConfig)
    for path in paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(path)))
    dotlist = list(overrides)
    if out is not None:
        dotlist.append(f"run.out_dir={out}")
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    # OmegaConf's stubs type merge/load as `Any | ListConfig | DictConfig`; a
    # schema-rooted merge is always a mapping, so this is a `DictConfig` in
    # practice.
    return cast(DictConfig, cfg)


def to_pipeline(cfg: DictConfig) -> PipelineConfig:
    """Convert to dataclasses (running range checks) and validate prep ops."""
    # OmegaConf.to_object's stub returns a broad container union; structured
    # config rooted at PipelineConfig always yields a PipelineConfig instance.
    obj = cast(PipelineConfig, OmegaConf.to_object(cfg))
    validate_chain(obj.prepare)
    for index, overlay in enumerate(obj.overlays):
        try:
            validate_chain(overlay.prepare)
        except Exception as exc:
            raise type(exc)(f"overlays[{index}].prepare: {exc}") from exc
    return obj


def load_config(
    paths: Sequence[str | Path],
    overrides: Sequence[str] = (),
    out: str | Path | None = None,
) -> PipelineConfig:
    return to_pipeline(load_raw(paths, overrides, out))
