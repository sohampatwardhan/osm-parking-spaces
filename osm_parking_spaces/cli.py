"""Command line entry point. Reads imagery and writes local files only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.evaluate import score_files
from osm_parking_spaces.pipeline import detect_geojson, detect_way


def _evaluate_way(args) -> int:
    import json

    from osm_parking_spaces.evaluate import load_polygons, render_eval
    from osm_parking_spaces.imagery import polygon_to_pixels

    result = detect_way(args.way, args.out_dir, detector=args.detector, weights=args.weights)
    truth_px = [polygon_to_pixels(p, result.grid) for p in load_polygons(args.truth)]
    detected_px = [stall.polygon for stall in result.stalls]
    render_eval(result.image, detected_px, truth_px, args.out_dir / "eval.png")
    score = score_files(result.geojson_path, args.truth)
    (args.out_dir / "score.json").write_text(json.dumps(score.as_dict(), indent=2) + "\n")
    print(f"way {args.way} ({args.detector}): {json.dumps(score.as_dict())}")
    print(args.out_dir / "eval.png")
    return 0


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
    detect.add_argument(
        "--detector",
        choices=("classical", "ml", "hybrid"),
        default="classical",
        help="classical line detector, the learned stall mask (v3), or both fused (v4)",
    )
    detect.add_argument("--weights", type=Path, help="Checkpoint for --detector ml")
    dataset = sub.add_parser("dataset", help="Download Natural2020 tiles and OSM stall masks")
    dataset.add_argument("--out", type=Path, required=True)
    dataset.add_argument("--max-lots", type=int, default=40)
    dataset.add_argument("--seed", type=int, default=0, help="Shuffle seed for lot selection")
    dataset.add_argument(
        "--cache-dir",
        type=Path,
        help="Where raw Overpass answers are kept (default OUT/overpass-cache)",
    )
    dataset.add_argument(
        "--bbox",
        type=str,
        default="38.92,-75.58,41.36,-73.89",
        help="south,west,north,east in degrees. Default covers New Jersey.",
    )
    train_cmd = sub.add_parser("train", help="Train the stall network")
    train_cmd.add_argument("--data", type=Path, required=True)
    train_cmd.add_argument("--out", type=Path, required=True)
    train_cmd.add_argument("--epochs", type=int, default=20)
    train_cmd.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    train_cmd.add_argument(
        "--arch",
        choices=("unet", "swin-unet"),
        default="swin-unet",
        help="swin-unet: pretrained Satlas aerial encoder (v4). unet: from scratch (v3).",
    )
    train_cmd.add_argument(
        "--chip", type=int, help="Chip size in px (default 256, or 128 for unet)"
    )
    train_cmd.add_argument("--batch-size", type=int, help="Default 8, or 4 for unet")
    train_cmd.add_argument(
        "--encoder",
        type=Path,
        default=Path("runs/pretrained"),
        help="Directory for the Satlas encoder weights (downloaded if missing)",
    )
    fetch_cmd = sub.add_parser(
        "fetch-encoder",
        help="Download the Satlas aerial Swin-B encoder (ODC-BY) and check its hash",
    )
    fetch_cmd.add_argument("--dir", type=Path, default=Path("runs/pretrained"))
    tune_cmd = sub.add_parser(
        "tune", help="Choose the pixel threshold and fusion settings on the validation lots"
    )
    tune_cmd.add_argument("--data", type=Path, required=True)
    tune_cmd.add_argument("--weights", type=Path, required=True)
    tune_cmd.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    tune_cmd.add_argument("--max-tiles", type=int)
    ev_way = sub.add_parser(
        "evaluate-way",
        help="Detect one way, write the three outputs plus eval.png, and score it",
    )
    ev_way.add_argument("--way", type=int, required=True)
    ev_way.add_argument("--truth", type=Path, required=True)
    ev_way.add_argument("--out-dir", type=Path, required=True)
    ev_way.add_argument("--detector", choices=("classical", "ml", "hybrid"), default="hybrid")
    ev_way.add_argument("--weights", type=Path)
    coreml = sub.add_parser("export-coreml", help="Export a checkpoint to Core ML")
    coreml.add_argument("--weights", type=Path, required=True)
    coreml.add_argument("--out", type=Path, required=True)
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
        if args.command == "dataset":
            from osm_parking_spaces.ml.dataset import build_dataset

            south, west, north, east = (float(part) for part in args.bbox.split(","))
            manifest = build_dataset(
                args.out,
                bbox=(south, west, north, east),
                max_lots=args.max_lots,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )
            print(
                f"lots: {manifest['lots']} stalls: {manifest['stalls']} "
                f"train: {manifest['train_lots']} val: {manifest['val_lots']} "
                f"angled: {manifest['angled_lots']} skipped: {manifest['skipped']}"
            )
            print(args.out / "manifest.json")
            return 0
        if args.command == "train":
            from osm_parking_spaces.ml.train import train

            swin = args.arch == "swin-unet"
            encoder = None
            if swin:
                from osm_parking_spaces.ml.encoder import fetch_encoder

                encoder = fetch_encoder(args.encoder)
            record = train(
                args.data,
                args.out,
                epochs=args.epochs,
                device_name=args.device,
                arch=args.arch,
                encoder_weights=encoder,
                chip=args.chip or (256 if swin else 128),
                batch_size=args.batch_size or (8 if swin else 4),
            )
            print(f"epoch: {record['epoch']} train_loss: {record['train_loss']}")
            print(args.out / "checkpoint.pt")
            print(args.out / "best.pt")
            print(args.out / "metrics.jsonl")
            return 0
        if args.command == "fetch-encoder":
            from osm_parking_spaces.ml.encoder import fetch_encoder

            print(fetch_encoder(args.dir))
            return 0
        if args.command == "tune":
            from osm_parking_spaces.ml.tune import tune

            result = tune(args.data, args.weights, args.device, args.max_tiles)
            print(f"threshold: {result['threshold']}")
            return 0
        if args.command == "evaluate-way":
            return _evaluate_way(args)
        if args.command == "export-coreml":
            from osm_parking_spaces.ml.infer import export_coreml

            export_coreml(args.weights, args.out)
            print(args.out)
            return 0
        if args.way is not None:
            result = detect_way(
                args.way,
                args.out_dir,
                gsd_ft=args.gsd_ft,
                detector=args.detector,
                weights=args.weights,
            )
            label = f"way {args.way}"
        else:
            result = detect_geojson(
                args.geojson,
                args.out_dir,
                gsd_ft=args.gsd_ft,
                detector=args.detector,
                weights=args.weights,
            )
            label = str(args.geojson)
    except ParkingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{label}: {result.stall_count} stalls")
    print(result.geojson_path)
    print(result.osc_path)
    print(result.preview_path)
    return 0
