"""Command line entry point. Reads imagery and writes local files only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.evaluate import score_files
from osm_parking_spaces.pipeline import detect_geojson, detect_way


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="osm-parking-spaces",
        description=(
            "Detect painted parking stalls inside one parking polygon and write "
            "GeoJSON, a create-only OsmChange file, and a preview PNG. "
            "Nothing is uploaded to OpenStreetMap."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("detect", help="Detect stalls for one parking polygon")
    source = detect.add_mutually_exclusive_group(required=True)
    source.add_argument("--way", type=int, help="OSM way id of an amenity=parking polygon")
    source.add_argument("--geojson", type=Path, help="GeoJSON file with one polygon")
    detect.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the three outputs",
    )
    detect.add_argument(
        "--gsd-ft",
        type=float,
        default=1.0,
        help="Ground sample distance in US survey feet (default: 1, the NJOGIS native grid)",
    )
    evaluate = sub.add_parser(
        "evaluate",
        help="Score a detection GeoJSON against hand-labelled ground truth",
    )
    evaluate.add_argument("--detected", type=Path, required=True)
    evaluate.add_argument("--truth", type=Path, required=True)
    evaluate.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Minimum intersection-over-union for a match (default 0.5)",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = score_files(args.detected, args.truth, iou_min=args.iou)
            for key, value in result.as_dict().items():
                print(f"{key}: {value}")
            return 0
        if args.way is not None:
            result = detect_way(args.way, args.out_dir, gsd_ft=args.gsd_ft)
            label = f"way {args.way}"
        else:
            result = detect_geojson(args.geojson, args.out_dir, gsd_ft=args.gsd_ft)
            label = str(args.geojson)
    except ParkingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{label}: {result.stall_count} stalls")
    print(result.geojson_path)
    print(result.osc_path)
    print(result.preview_path)
    return 0
