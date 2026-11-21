"""Thin argparse wrapper over ``pipeline.run`` and the acquire stage."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from eo_art.forge3d_pipes.pipeline import run


def _sweep_to_override(spec: str) -> str:
    """Turn ``path=a,b,c`` into an OmegaConf dotlist assignment."""
    path, _, values = spec.partition("=")
    if not values:
        raise argparse.ArgumentTypeError(
            f"--sweep expects 'dotted.path=v1,v2', got {spec!r}"
        )
    items = ", ".join(value.strip() for value in values.split(","))
    return f"sweep.params={{{path}: [{items}]}}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eo-art-f3d")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="run a render pipeline")
    run_cmd.add_argument(
        "configs", nargs="+", help="config files, merged left to right"
    )
    run_cmd.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="dotlist override (repeatable)",
    )
    run_cmd.add_argument("--out", default=None, help="shorthand for run.out_dir")
    run_cmd.add_argument(
        "--sweep",
        default=None,
        type=_sweep_to_override,
        metavar="PATH=V1,V2",
        help=(
            "add a swept parameter; combines with any sweep already defined "
            "in the config files rather than replacing it"
        ),
    )
    run_cmd.add_argument(
        "--no-cache", action="store_true", help="recompute the prep chain"
    )
    run_cmd.add_argument(
        "--fail-fast", action="store_true", help="abort on the first failing variant"
    )
    run_cmd.add_argument(
        "--acquire",
        default=None,
        metavar="CONFIG",
        help=(
            "acquire config; run the acquire stage first and feed its DTM/"
            "optical outputs into this render as input.path/overlay path "
            "before rendering"
        ),
    )
    run_cmd.add_argument(
        "--acquire-overlay",
        default=None,
        metavar="NAME",
        help="which overlay's path to set from the acquire stage "
        "(default: the first overlay in the render config)",
    )
    run_cmd.add_argument(
        "--acquire-no-cache",
        action="store_true",
        help="rerun the acquire stage instead of reusing a cached result",
    )
    run_cmd.add_argument(
        "--acquire-run-id",
        default=None,
        help="name of the acquire run folder under <out>/runs/ "
        "(default: <YYYYmmdd-HHMMSS>-acquire)",
    )

    acquire_cmd = sub.add_parser(
        "acquire", help="fetch and prepare terrain/optical inputs for a render"
    )
    acquire_cmd.add_argument("config", help="acquire config file")
    acquire_cmd.add_argument(
        "--out",
        default=None,
        help="directory to write/cache acquire outputs under "
        "(default: out_dir from the acquire config)",
    )
    acquire_cmd.add_argument(
        "--no-cache", action="store_true", help="rerun even if cached outputs exist"
    )
    acquire_cmd.add_argument(
        "--run-id",
        default=None,
        help="name of the run folder under <out>/runs/ "
        "(default: <YYYYmmdd-HHMMSS>-acquire)",
    )

    return parser


def _run_acquire_stage(
    acquire_config: str,
    out_dir: str | None,
    use_cache: bool,
    run_id: str | None = None,
):
    """Acquire into a new run folder ``<out>/runs/<run_id>/`` (cache shared
    under ``<out>/_acquire``); ``runs/latest-acquire`` follows on success."""
    from typing import cast

    from omegaconf import OmegaConf

    from eo_art.forge3d_pipes.acquire import run_acquire
    from eo_art.forge3d_pipes.acquire.schema import AcquireConfig

    raw = OmegaConf.merge(
        OmegaConf.structured(AcquireConfig), OmegaConf.load(acquire_config)
    )
    cfg = cast(AcquireConfig, OmegaConf.to_object(raw))
    if cfg.sentinel.timeseries.enabled:
        print(
            "note: acquire ignores sentinel.timeseries (it needs one reference "
            "scene); run eo-art-sr for the timeseries"
        )
    from eo_art.forge3d_pipes.acquire.runs import (
        link_latest,
        new_run_dir,
        write_run_config,
    )

    out = out_dir or cfg.out_dir
    run_dir = new_run_dir(out, "acquire", run_id)
    write_run_config(run_dir, cfg)
    print(f"run     -> {run_dir}")
    result = run_acquire(cfg, out, use_cache=use_cache, run_dir=run_dir)
    link_latest(out, "acquire", run_dir)
    return result


def _apply_acquire_result(configs, acquire_overlay, result):
    """Merge an AcquireResult's paths into the render config and return the
    path to a temp file holding the merged result.

    OmegaConf's dotlist overrides always build DictConfig nodes for numeric
    path segments (``overlays.0.path=...`` parses as ``{"overlays": {"0":
    {"path": ...}}}``), which then fails to merge against ``overlays``' real
    ListConfig -- so the overlay path has to be set by direct item
    assignment on the loaded config instead of via a dotlist string.
    """
    import tempfile

    from omegaconf import OmegaConf

    from eo_art.forge3d_pipes.config.loader import load_raw

    raw = load_raw(configs)
    raw.input.path = str(result.dtm_path)

    if not raw.overlays:
        raise SystemExit(
            "--acquire needs at least one entry under `overlays:` in the "
            "render config to attach the acquired optical raster to"
        )
    overlay_index = 0
    if acquire_overlay:
        names = [overlay.name for overlay in raw.overlays]
        if acquire_overlay not in names:
            raise SystemExit(
                f"--acquire-overlay {acquire_overlay!r} not found in "
                f"`overlays:` (have: {names})"
            )
        overlay_index = names.index(acquire_overlay)
    if result.zones is not None:
        _expand_zone_overlays(raw, overlay_index, result.zones.tiles)
    else:
        raw.overlays[overlay_index].path = str(result.optical_path)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="eo-art-acquired-"
    )
    OmegaConf.save(raw, tmp.name)
    return tmp.name


def _expand_zone_overlays(raw, overlay_index, tiles):
    """Replace overlay ``overlay_index`` by one copy per SR zone tile.

    Every copy keeps the entry's settings and is snapped onto the DTM at
    render time (``align_to_dem``). Its ``rgb_stretch`` ops get fixed bounds
    pooled over all tiles, so the tiles share one stretch instead of each
    being levelled on its own (which shows as brightness seams).
    """
    import copy

    from omegaconf import OmegaConf

    from eo_art.forge3d_pipes.prep.ops import RgbStretchCfg, stretch_bounds_over

    node = raw.overlays[overlay_index]
    name = node.name
    prepare = OmegaConf.to_container(node.prepare, resolve=True)
    assert isinstance(prepare, list)
    stretch_ops = [
        i
        for i, op in enumerate(prepare)
        if isinstance(op, dict) and op.get("op") == "rgb_stretch"
    ]
    if not stretch_ops:
        print(
            f"warning: overlay {name!r} has no rgb_stretch op, so each zone "
            "tile is stretched on its own and tiles may differ in brightness"
        )
    shared_bounds = {}
    for i in stretch_ops:
        params = {k: v for k, v in prepare[i].items() if k != "op"}
        if "lower_value" in params or "upper_value" in params:
            continue
        stretch = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(RgbStretchCfg), params)
        )
        assert isinstance(stretch, RgbStretchCfg)
        shared_bounds[i] = stretch_bounds_over(tiles, stretch)

    copies = []
    for tile in tiles:
        zone_id = Path(tile).stem.rsplit("_sr_", 1)[-1]
        entry = copy.deepcopy(node)
        entry.name = f"{name}_{zone_id}"
        entry.path = str(tile)
        entry.align_to_dem = True
        for i, (lo, hi) in shared_bounds.items():
            entry.prepare[i].lower_value = lo
            entry.prepare[i].upper_value = hi
        copies.append(entry)
    overlays = list(raw.overlays)
    raw.overlays = [
        *overlays[:overlay_index],
        *copies,
        *overlays[overlay_index + 1 :],
    ]
    print(f"--acquire: {len(copies)} zone tile overlay(s) from {name!r}")


def _run_command(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.sweep:
        overrides.append(args.sweep)

    configs = list(args.configs)
    if args.acquire:
        result = _run_acquire_stage(
            args.acquire,
            args.out,
            use_cache=not args.acquire_no_cache,
            run_id=args.acquire_run_id,
        )
        configs = [_apply_acquire_result(configs, args.acquire_overlay, result)]

    results = run(
        configs,
        overrides=overrides,
        out=args.out,
        use_cache=False if args.no_cache else None,
        fail_fast=args.fail_fast or None,
    )

    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    for result in failed:
        print(f"FAILED {result.name}: {result.error}")
    for result in succeeded:
        print(f"ok     {result.name} -> {result.out_dir}")
    print(f"{len(succeeded)} succeeded, {len(failed)} failed")
    return 1 if failed else 0


def _acquire_command(args: argparse.Namespace) -> int:
    result = _run_acquire_stage(
        args.config, args.out, use_cache=not args.no_cache, run_id=args.run_id
    )
    print(f"dtm     -> {result.dtm_path}")
    if result.zones is not None:
        zones = result.zones
        print(f"zones   -> {zones.out_dir} ({len(zones.tiles)} tile(s))")
        print(f"vrt     -> {zones.vrt}")
        if zones.mosaic:
            print(f"mosaic  -> {zones.mosaic}")
        if zones.empty_zones:
            print(f"no scene for zone(s) {', '.join(zones.empty_zones)}")
    else:
        print(f"optical -> {result.optical_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "acquire":
        return _acquire_command(args)
    return _run_command(args)
