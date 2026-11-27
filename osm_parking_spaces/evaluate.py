"""Match detected stalls to hand-labelled ground truth.

A detection matches a ground-truth stall when intersection-over-union is at
least 0.5. Matching is one-to-one. Mean centroid offset is reported only for
the matched pairs, in US survey feet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon, shape

from osm_parking_spaces.errors import ParkingError

IOU_MIN = 0.5
_GEOD = Geod(ellps="WGS84")
_FT = 0.3048006096012192


@dataclass(frozen=True)
class Score:
    truth: int
    detected: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_offset_ft: float

    def as_dict(self) -> dict:
        return {
            "truth": self.truth,
            "detected": self.detected,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_offset_ft": None if self.true_positives == 0 else round(self.mean_offset_ft, 2),
        }


def load_polygons(path: Path) -> list[Polygon]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ParkingError(f"Could not read {path}: {exc}") from exc
    polygons = []
    for feature in data.get("features", []):
        geometry = shape(feature["geometry"])
        if geometry.geom_type == "Polygon" and not geometry.is_empty:
            polygons.append(geometry)
    return polygons


def score_polygons(
    detected: list[Polygon],
    truth: list[Polygon],
    iou_min: float = IOU_MIN,
    pixel_gsd_ft: float | None = None,
) -> tuple[Score, list[tuple[int, int]]]:
    """Return the score and the list of (detection index, truth index) matches.

    Polygons are WGS84 by default. For polygons in image pixels pass `pixel_gsd_ft`, the feet
    per pixel, so the mean offset is the centroid distance in feet.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, det in enumerate(detected):
        for j, gt in enumerate(truth):
            inter = det.intersection(gt).area
            if inter <= 0:
                continue
            union = det.area + gt.area - inter
            iou = inter / union if union else 0.0
            if iou >= iou_min:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_det: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int]] = []
    offsets: list[float] = []
    for _iou, i, j in pairs:
        if i in used_det or j in used_truth:
            continue
        used_det.add(i)
        used_truth.add(j)
        matches.append((i, j))
        offsets.append(_offset_ft(detected[i], truth[j], pixel_gsd_ft))
    tp = len(matches)
    fp = len(detected) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_offset = float(sum(offsets) / len(offsets)) if offsets else 0.0
    result = Score(
        truth=len(truth),
        detected=len(detected),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_offset_ft=mean_offset,
    )
    return result, matches


def score_files(detected_path: Path, truth_path: Path, iou_min: float = IOU_MIN) -> Score:
    result, _matches = score_polygons(
        load_polygons(detected_path), load_polygons(truth_path), iou_min=iou_min
    )
    return result


def render_eval(
    image_bgr: np.ndarray,
    detected_px: list[Polygon],
    truth_px: list[Polygon],
    path: Path,
    iou_min: float = IOU_MIN,
    gsd_ft: float = 1.0,
) -> Score:
    """Draw true positives, false positives, and missed stalls on the imagery."""
    result, matches = score_polygons(detected_px, truth_px, iou_min=iou_min, pixel_gsd_ft=gsd_ft)
    matched_det = {i for i, _j in matches}
    matched_truth = {j for _i, j in matches}
    canvas = image_bgr.copy()
    overlay = image_bgr.copy()
    for index, polygon in enumerate(detected_px):
        color = (40, 170, 40) if index in matched_det else (40, 40, 220)
        _fill(overlay, polygon, color)
        _draw(canvas, polygon, color, 1)
    for index, polygon in enumerate(truth_px):
        if index not in matched_truth:
            _draw(canvas, polygon, (0, 140, 255), 2)
    painted = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    if not cv2.imwrite(str(path), painted):
        raise ParkingError(f"Could not write {path}")
    return result


def _offset_ft(detected: Polygon, truth: Polygon, pixel_gsd_ft: float | None = None) -> float:
    x1, y1 = detected.centroid.x, detected.centroid.y
    x2, y2 = truth.centroid.x, truth.centroid.y
    if pixel_gsd_ft is not None:
        return float(np.hypot(x1 - x2, y1 - y2)) * pixel_gsd_ft
    _az12, _az21, meters = _GEOD.inv(x1, y1, x2, y2)
    return abs(meters) / _FT


def _draw(image: np.ndarray, polygon: Polygon, color: tuple[int, int, int], thickness: int) -> None:
    points = np.round(polygon.exterior.coords).astype(np.int32)
    cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)


def _fill(image: np.ndarray, polygon: Polygon, color: tuple[int, int, int]) -> None:
    points = np.round(polygon.exterior.coords).astype(np.int32)
    cv2.fillPoly(image, [points], color)
