"""Choose thresholds on the validation lots only. The test lots are never read here.

1. Pixel threshold: the value that maximises mean whole-tile Dice on the validation tiles.
2. Fusion parameters: coordinate descent on micro F1 of the fused stalls against the OSM stall
   polygons of each validation lot (IoU >= 0.5, one to one). OSM labels are incomplete, so the
   absolute numbers are pessimistic. Only the ranking of settings is used.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from osm_parking_spaces.detect import detect_stalls
from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.evaluate import score_polygons
from osm_parking_spaces.ml.dataset import load_meta, load_tile
from osm_parking_spaces.ml.device import pick_device
from osm_parking_spaces.ml.fusion import FusionConfig, fuse
from osm_parking_spaces.ml.infer import load_model, predict_probs
from osm_parking_spaces.ml.train import dice_score

THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.30, 0.76, 0.05))
GRID = {
    "keep": (0.30, 0.40, 0.50, 0.60),
    "reject": (0.10, 0.20, 0.30),
    "fill": (0.35, 0.45, 0.55, 0.65),
    "seed": (None, 0.50, 0.60, 0.70, 0.80),
    "trim": (False, True),
    "max_fill": (0, 3, 6),
}


def sweep_threshold(probs: list[np.ndarray], masks: list[np.ndarray]) -> dict[float, float]:
    """Mean Dice per threshold."""
    return {
        t: float(np.mean([dice_score(p >= t, m) for p, m in zip(probs, masks)])) for t in THRESHOLDS
    }


def micro_score(per_tile: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    tp = sum(t for t, _f, _n in per_tile)
    fp = sum(f for _t, f, _n in per_tile)
    fn = sum(n for _t, _f, n in per_tile)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def tune(
    data_dir: Path,
    weights: Path,
    device_name: str = "auto",
    max_tiles: int | None = None,
    log=print,
) -> dict:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    items = [i for i in manifest["tiles_detail"] if i.get("split") == "val"]
    if max_tiles:
        items = items[:max_tiles]
    if not items:
        raise ParkingError("The dataset has no validation lots to tune on.")
    device = pick_device(device_name)
    model, _old = load_model(weights, device)
    tiles = []
    for item in items:
        image, mask = load_tile(data_dir / item["file"])
        meta = load_meta(data_dir / item["file"])
        tiles.append((image, mask, meta))
    probs = [predict_probs(model, image, device, tta=True) for image, _m, _meta in tiles]
    dice = sweep_threshold(probs, [mask for _i, mask, _meta in tiles])
    best_t = max(dice, key=dice.get)
    log(f"pixel threshold sweep (val Dice): {dice}")
    log(f"chosen pixel threshold: {best_t}")
    result = {"threshold_dice": dice, "threshold": best_t, "val_tiles": len(items)}
    scoreable = [(i, t) for i, t in enumerate(tiles) if t[2] is not None]
    if scoreable:
        classical = [detect_stalls(tiles[i][0], tiles[i][2]["lot"]) for i, _t in scoreable]
        base = FusionConfig(pixel=best_t)

        def run(config: FusionConfig | None):
            rows = []
            for (i, (_img, _mask, meta)), stalls in zip(scoreable, classical):
                if config is None:
                    found = [s.polygon for s in stalls]
                else:
                    found = [s.polygon for s in fuse(stalls, probs[i], meta["lot"], 1.0, config)]
                score, _m = score_polygons(found, meta["stalls"])
                rows.append((score.true_positives, score.false_positives, score.false_negatives))
            return micro_score(rows)

        v2_only = run(None)
        log(f"val v2 only (P/R/F1 vs OSM): {v2_only}")
        best, best_f1 = base, run(base)[2]
        for sweep in range(2):
            for name, values in GRID.items():
                for value in values:
                    trial = replace(best, **{name: value})
                    f1 = run(trial)[2]
                    if f1 > best_f1 + 1e-4:
                        best, best_f1 = trial, f1
                        log(f"  pass {sweep} {name}={value}: F1 {f1:.4f}")
        final = run(best)
        log(f"val fused (P/R/F1 vs OSM): {final} with {best}")
        result.update(
            {
                "val_v2_prf": v2_only,
                "val_fused_prf": final,
                "fusion": best.__dict__,
                "val_lots_scored": len(scoreable),
            }
        )
        best.save(Path(weights).parent / "fusion.json")
    (Path(weights).parent / "tuning.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_threshold(Path(weights), best_t)
    return result


def _write_threshold(weights: Path, threshold: float) -> None:
    import torch

    saved = torch.load(weights, map_location="cpu", weights_only=False)
    saved["threshold"] = float(threshold)
    torch.save(saved, weights)
