"""Argparse wrapper over sentinel_sr, writing a manifest.

Modes: per-chip SR (default), ``--lr-only`` (fetch without SR),
``--candidates-only``, and the two file-based steps of the whole-AOI path,
``--coreg-jobs`` (AROSICS over LR rasters) and ``--superres-jobs`` (SR of
whole rasters with overlapping patches).
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sentinel_sr import coreg, superres
from sentinel_sr.clouds import DEFAULT_CLOUD_CLASSES
from sentinel_sr.rgbn import BANDS_10, export_superres_tifs, list_candidates
from sentinel_sr.timeseries import FREQUENCIES

MANIFEST_NAME = "manifest.json"
CANDIDATES_NAME = "candidates.json"
# Required for the scene modes, not for the --*-jobs modes.
_SCENE_ARGS = ("lat", "lon", "start_date", "end_date", "out")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eo-art-sentinel-sr")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--model-path")
    parser.add_argument("--out", help="output directory")
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
    parser.add_argument(
        "--interval",
        choices=sorted(FREQUENCIES),
        default=None,
        help="timeseries: keep only the least cloudy scene per interval",
    )
    parser.add_argument(
        "--every", type=int, default=1, help="timeseries: one scene every N intervals"
    )
    parser.add_argument(
        "--max-aoi-cloud",
        type=float,
        default=None,
        help="drop scenes with more than this percent of AOI pixels in the SCL "
        "cloud classes (default: no SCL check)",
    )
    parser.add_argument(
        "--scl-classes",
        default=",".join(map(str, DEFAULT_CLOUD_CLASSES)),
        help="comma-separated SCL classes counted as cloud",
    )
    parser.add_argument(
        "--skip-dates",
        default="",
        help="timeseries: comma-separated YYYY-MM-DD dates whose intervals "
        "are already done and get no scene",
    )
    parser.add_argument(
        "--dates",
        default="",
        help="comma-separated YYYY-MM-DD acquisition dates to super-resolve, "
        "chosen upstream (no interval selection, no SCL filter)",
    )
    parser.add_argument(
        "--candidates-only",
        action="store_true",
        help=f"write {CANDIDATES_NAME} with every scene that passes the cloud "
        "checks (and its interval), without running SR",
    )
    parser.add_argument(
        "--lr-only",
        action="store_true",
        help="write the selected scenes at native resolution (plus SCL) without SR",
    )
    parser.add_argument(
        "--coreg-jobs",
        help='JSON {"jobs": [{"src", "dst", "date", "scl"}]}: co-register the '
        "LR rasters as one series (AROSICS), report next to it as coreg.json",
    )
    parser.add_argument(
        "--anchor",
        default="auto",
        help="coreg: reference scene, auto | first | composite | YYYY-MM-DD",
    )
    parser.add_argument("--grid-size", type=int, default=7, help="coreg windows")
    parser.add_argument(
        "--superres-jobs",
        help='JSON {"jobs": [{"src", "dst"}]}: super-resolve whole rasters '
        "with overlapping patches (needs --model-path)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="SR patches per model call"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redo scenes whose GeoTIFF already exists instead of keeping them",
    )
    return parser


def _split(value: str) -> list[str]:
    return [v for v in value.split(",") if v]


def _write_candidates(args: argparse.Namespace) -> int:
    result = list_candidates(
        lat=args.lat,
        lon=args.lon,
        start_date=args.start_date,
        end_date=args.end_date,
        collection=args.collection,
        bands=tuple(args.bands.split(",")),
        edge_size=args.edge_size,
        resolution=args.resolution,
        stac=args.stac,
        max_cloud_cover=args.max_cloud_cover,
        interval=args.interval,
        every=args.every,
        max_aoi_cloud=args.max_aoi_cloud,
        scl_classes=tuple(int(c) for c in args.scl_classes.split(",")),
    )
    path = Path(args.out) / CANDIDATES_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "id": c.id,
                        "datetime": c.datetime,
                        "cloud_cover": c.cloud_cover,
                        "aoi_cloud": c.aoi_cloud,
                        "bin": c.bin,
                    }
                    for c in result.scenes
                ],
                "bins": result.bins,
            },
            indent=2,
        )
    )
    print(f"Wrote {path} ({len(result.scenes)} candidate(s))")
    return 0


def _run_coreg(args: argparse.Namespace) -> int:
    jobs_path = Path(args.coreg_jobs)
    results = coreg.coregister_jobs(
        coreg.read_jobs(jobs_path),
        anchor=args.anchor,
        grid_size=args.grid_size,
        scl_classes=tuple(int(c) for c in args.scl_classes.split(",")),
    )
    report = coreg.write_report(results, jobs_path.parent / "coreg.json")
    print(f"Wrote {report}")
    return 0


def _run_superres(args: argparse.Namespace) -> int:
    jobs = json.loads(Path(args.superres_jobs).read_text())["jobs"]
    superres.superres_jobs(
        jobs, args.model_path, batch_size=args.batch_size, overwrite=args.overwrite
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.coreg_jobs:
        return _run_coreg(args)
    if args.superres_jobs:
        if not args.model_path:
            parser.error("--superres-jobs needs --model-path")
        return _run_superres(args)
    missing = [
        f"--{a.replace('_', '-')}" for a in _SCENE_ARGS if getattr(args, a) is None
    ]
    if not (args.candidates_only or args.lr_only) and args.model_path is None:
        missing.append("--model-path")
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.candidates_only:
        return _write_candidates(args)

    result = export_superres_tifs(
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
        interval=args.interval,
        every=args.every,
        overwrite=args.overwrite,
        max_aoi_cloud=args.max_aoi_cloud,
        scl_classes=tuple(int(c) for c in args.scl_classes.split(",")),
        skip_dates=_split(args.skip_dates),
        dates=_split(args.dates),
        lr_only=args.lr_only,
    )

    manifest_path = Path(args.out) / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "path": str(scene.path),
                        "cloud_cover": scene.cloud_cover,
                        "datetime": scene.datetime,
                        "aoi_cloud": scene.aoi_cloud,
                        **(
                            {"scl_path": str(scene.scl_path)}
                            if scene.scl_path is not None
                            else {}
                        ),
                    }
                    for scene in result.scenes
                ],
                "skipped_bins": result.skipped_bins,
            },
            indent=2,
        )
    )
    print(f"Wrote {manifest_path}")
    return 0
