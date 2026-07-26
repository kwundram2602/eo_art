"""Thin argparse wrapper over ``pipeline.run``."""

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    overrides = list(args.overrides)
    if args.sweep:
        overrides.append(args.sweep)

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
