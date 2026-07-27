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
    raw.overlays[overlay_index].path = str(result.optical_path)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="eo-art-acquired-"
    )
    OmegaConf.save(raw, tmp.name)
    return tmp.name


def _run_command(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.sweep:
        overrides.append(args.sweep)

    configs = list(args.configs)
    if args.acquire:
        result = _run_acquire_stage(
            args.acquire, args.out, use_cache=not args.acquire_no_cache
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
    result = _run_acquire_stage(args.config, args.out, use_cache=not args.no_cache)
    print(f"dtm     -> {result.dtm_path}")
    print(f"optical -> {result.optical_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "acquire":
        return _acquire_command(args)
    return _run_command(args)
