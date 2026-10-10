"""``eo-art-sr``: Sentinel-2 super-resolution for an AOI, nothing else.

Reads the ``aoi_path``/``out_dir``/``sentinel_sr_dir``/``sentinel`` part of an
acquire config and ignores the rest, so the same YAML serves both commands.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from eo_art.forge3d_pipes.acquire import SrConfig, run_sr

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
    path: Path = run_sr(cfg, args.out or cfg.out_dir, use_cache=not args.no_cache)
    print(f"sr -> {path}")
    return 0
