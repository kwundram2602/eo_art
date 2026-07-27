"""Thin argparse wrapper over ``pipeline.run`` and the acquire stage."""

import argparse
from collections.abc import Sequence

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
    return parser


def _run_acquire_stage(acquire_config: str, out_dir: str | None, use_cache: bool):
    from typing import cast

    from omegaconf import OmegaConf

    from eo_art.forge3d_pipes.acquire import run_acquire
    from eo_art.forge3d_pipes.acquire.schema import AcquireConfig

    raw = OmegaConf.merge(
        OmegaConf.structured(AcquireConfig), OmegaConf.load(acquire_config)
    )
    cfg = cast(AcquireConfig, OmegaConf.to_object(raw))
    return run_acquire(cfg, out_dir or cfg.out_dir, use_cache=use_cache)


def _run_command(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.sweep:
        overrides.append(args.sweep)

    if args.acquire:
        result = _run_acquire_stage(
            args.acquire, args.out, use_cache=not args.acquire_no_cache
        )
        overrides.append(f"input.path={result.dtm_path}")
        overlay_index = 0
        if args.acquire_overlay:
            from eo_art.forge3d_pipes.config.loader import load_raw

            raw = load_raw(args.configs)
            names = [overlay.name for overlay in raw.overlays]
            overlay_index = names.index(args.acquire_overlay)
        overrides.append(f"overlays.{overlay_index}.path={result.optical_path}")

    results = run(
        args.configs,
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
    result = _run_acquire_stage(args.config, args.out, use_cache=not args.no_cache)
    print(f"dtm     -> {result.dtm_path}")
    print(f"optical -> {result.optical_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "acquire":
        return _acquire_command(args)
    return _run_command(args)
