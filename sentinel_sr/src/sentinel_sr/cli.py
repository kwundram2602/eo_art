"""Argparse wrapper over ``rgbn.export_superres_tifs``, writing a manifest."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sentinel_sr.rgbn import BANDS_10, export_superres_tifs

MANIFEST_NAME = "manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eo-art-sentinel-sr")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--collection", default="sentinel-2-l2a")
    parser.add_argument(
        "--bands", default=",".join(BANDS_10), help="comma-separated band names"
    )
    parser.add_argument("--edge-size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument(
        "--stac", default="https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-cloud-cover", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    written = export_superres_tifs(
        lat=args.lat,
        lon=args.lon,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.out,
        model_path=args.model_path,
        collection=args.collection,
        bands=tuple(args.bands.split(",")),
        edge_size=args.edge_size,
        resolution=args.resolution,
        stac=args.stac,
        max_items=args.max_items,
        max_cloud_cover=args.max_cloud_cover,
    )

    manifest_path = Path(args.out) / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "scenes": [
                    {"path": str(scene.path), "cloud_cover": scene.cloud_cover}
                    for scene in written
                ]
            },
            indent=2,
        )
    )
    print(f"Wrote {manifest_path}")
    return 0
