"""``eo-art-sr``: Sentinel-2 super-resolution for an AOI, nothing else.

Reads the ``aoi_path``/``out_dir``/``sentinel_sr_dir``/``sentinel`` part of an
acquire config and ignores the rest, so the same YAML serves both commands.
"""

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.acquire import (
    SrConfig,
    run_sr,
    run_sr_timeseries,
    run_sr_timeseries_zones,
    run_sr_zones,
)
from eo_art.forge3d_pipes.acquire.pipeline import (
    export_zones_mosaic,
    timeseries_dirname,
)
from eo_art.forge3d_pipes.acquire.runs import (
    latest_run,
    link_latest,
    new_run_dir,
    runs_dir,
    write_run_config,
)
from eo_art.forge3d_pipes.acquire.zone_steps import describe_step

SR_KEYS = ("aoi_path", "out_dir", "sentinel_sr_dir", "sentinel")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eo-art-sr",
        description="run only the Sentinel-2 super-resolution step for an AOI",
    )
    parser.add_argument("config", help="acquire config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="dotlist override, e.g. aoi_path=my_aoi.gpkg (repeatable)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="directory to write/cache SR outputs under "
        "(default: out_dir from the config)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="rerun even if cached outputs exist"
    )
    parser.add_argument(
        "--mosaic-only",
        action="store_true",
        help="only write <aoi>_sr_mosaic.tif from the VRT of an earlier "
        "sentinel.zones run (--run, default: the latest one), no SR",
    )
    parser.add_argument(
        "--run",
        default=None,
        metavar="DIR",
        help="--mosaic-only: the run folder holding the zones VRT",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="name of the run folder under <out>/runs/ (default: <YYYYmmdd-HHMMSS>-sr)",
    )
    return parser


def load_sr_config(config: str, overrides: Sequence[str] = ()) -> SrConfig:
    # Only the SR-relevant keys -- the acquire config's dem/dtm/ee_project
    # blocks would be rejected by SrConfig's structured schema.
    loaded = cast(DictConfig, OmegaConf.load(config))
    subset = OmegaConf.masked_copy(loaded, [k for k in SR_KEYS if k in loaded])
    raw = OmegaConf.merge(
        OmegaConf.structured(SrConfig), subset, OmegaConf.from_dotlist(list(overrides))
    )
    return cast(SrConfig, OmegaConf.to_object(raw))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_sr_config(args.config, args.overrides)
    out_dir = args.out or cfg.out_dir
    use_cache = not args.no_cache
    if args.mosaic_only:
        run = (
            Path(args.run)
            if args.run
            else latest_run(out_dir, "sr") or latest_run(out_dir, "acquire")
        )
        if run is None:
            raise SystemExit(f"no finished run under {runs_dir(out_dir)}; pass --run")
        print(f"sr mosaic -> {export_zones_mosaic(cfg.aoi_path, run)}")
        return 0

    run_dir = new_run_dir(out_dir, "sr", args.run_id)
    write_run_config(run_dir, cfg)
    print(f"run -> {run_dir}")
    _seed_fill_gaps(cfg, out_dir, run_dir)
    _run(cfg, out_dir, use_cache, run_dir)
    link_latest(out_dir, "sr", run_dir)
    return 0


def _seed_fill_gaps(cfg: SrConfig, out_dir: str, run_dir: Path) -> None:
    """fill_gaps continues the latest SR run's timeseries: copy it into the
    new run, which then only adds the open intervals."""
    if not (cfg.sentinel.timeseries.enabled and cfg.sentinel.timeseries.fill_gaps):
        return
    previous = latest_run(out_dir, "sr")
    name = timeseries_dirname(cfg.aoi_path)
    if previous is not None and (previous / name).is_dir():
        shutil.copytree(previous / name, run_dir / name)
        print(f"fill_gaps: continuing {previous / name}")


def _run(cfg: SrConfig, out_dir: str, use_cache: bool, run_dir: Path) -> None:
    zones = cfg.sentinel.zones.enabled
    if zones and cfg.sentinel.timeseries.enabled:
        series = run_sr_timeseries_zones(
            cfg, out_dir, use_cache=use_cache, run_dir=run_dir
        )
        offset = cfg.sentinel.timeseries.max_zone_offset_days
        for step in series.steps:
            print(describe_step(step.step, offset))
        for start in series.skipped_bins:
            print(f"no scene in any zone for the interval starting {start}")
        print(
            f"sr timeseries zones -> {series.out_dir} "
            f"({len(series.steps)} step(s), steps.json lists dates and offsets)"
        )
        return
    if zones:
        zones_result = run_sr_zones(cfg, out_dir, use_cache=use_cache, run_dir=run_dir)
        if zones_result.empty_zones:
            print(f"no scene for zone(s) {', '.join(zones_result.empty_zones)}")
        print(
            f"sr zones -> {zones_result.out_dir} "
            f"({len(zones_result.tiles)} tile(s), vrt {zones_result.vrt.name})"
        )
        if zones_result.mosaic:
            print(f"sr mosaic -> {zones_result.mosaic}")
        return
    if cfg.sentinel.timeseries.enabled:
        result = run_sr_timeseries(cfg, out_dir, use_cache=use_cache, run_dir=run_dir)
        for start in result.skipped_bins:
            print(f"no scene for the interval starting {start}")
        print(
            f"sr timeseries -> {result.out_dir} "
            f"({len(result.added)} new, {len(result.paths)} total)"
        )
        return
    path: Path = run_sr(cfg, out_dir, use_cache=use_cache, run_dir=run_dir)
    print(f"sr -> {path}")
