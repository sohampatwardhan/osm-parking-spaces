"""Turn a stall-interior mask into shared-corner stall polygons."""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

from osm_parking_spaces.detect import Stall

MIN_AREA_SQFT = 50.0
MAX_SINGLE_SQFT = 400.0
MIN_SHORT_FT = 6.0
MAX_SHORT_FT = 16.0
MIN_LONG_FT = 11.0
MAX_LONG_FT = 26.0
ROW_LONG_FT = 28.0
PITCH_FT = 9.5


def stalls_from_mask(mask: np.ndarray, lot: Polygon, gsd_ft: float = 1.0) -> list[Stall]:
    """Connected components, split fused rows, snap each row to one pitch."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    pieces: list[list[tuple[float, float]]] = []
    for index in range(1, count):
        area_ft = float(stats[index, cv2.CC_STAT_AREA]) * gsd_ft * gsd_ft
        if area_ft < MIN_AREA_SQFT:
            continue
        ys, xs = np.where(labels == index)
        if len(xs) < 12:
            continue
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(points)
        pieces.extend(_pieces_from_rect(rect, gsd_ft))
    kept = []
    for corners in pieces:
        polygon = Polygon(corners)
        if not polygon.is_valid or polygon.area <= 0:
            continue
        if polygon.intersection(lot).area < 0.55 * polygon.area:
            continue
        kept.append(corners)
    if not kept:
        return []
    rows = _group_rows(kept, gsd_ft)
    stalls: list[Stall] = []
    for row_index, row in enumerate(rows, start=1):
        for corners in _weld_row(row):
            stalls.append(_stall(corners, gsd_ft, row_index))
    return stalls


def _pieces_from_rect(rect, gsd_ft: float) -> list[list[tuple[float, float]]]:
    corners = cv2.boxPoints(rect)
    lengths = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    long_i = int(np.argmax(lengths))
    short_px = min(lengths)
    long_px = max(lengths)
    if short_px < 1 or long_px < 1:
        return []
    origin = corners[long_i]
    long_vec = corners[(long_i + 1) % 4] - origin
    short_vec = corners[(long_i - 1) % 4] - origin
    long_ft = long_px * gsd_ft
    short_ft = short_px * gsd_ft
    # A fused row is deep like one stall and long like several.
    if MIN_LONG_FT <= short_ft <= MAX_LONG_FT and long_ft >= ROW_LONG_FT:
        count = max(2, int(round(long_ft / PITCH_FT)))
        return _divide(origin, long_vec, short_vec, count)
    if MIN_SHORT_FT <= short_ft <= MAX_SHORT_FT and MIN_LONG_FT <= long_ft <= MAX_LONG_FT:
        return _divide(origin, long_vec, short_vec, 1)
    return []


def _divide(origin, long_vec, short_vec, count: int) -> list[list[tuple[float, float]]]:
    dividers = []
    for index in range(count + 1):
        start = origin + long_vec * (index / count)
        end = start + short_vec
        dividers.append(
            (
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
            )
        )
    quads = []
    for index in range(count):
        quads.append(
            [
                dividers[index][0],
                dividers[index + 1][0],
                dividers[index + 1][1],
                dividers[index][1],
            ]
        )
    return quads


def _group_rows(pieces: list[list[tuple[float, float]]], gsd_ft: float):
    annotated = []
    for corners in pieces:
        angle, center, across = _frame(corners)
        annotated.append((angle, center, across, corners))
    annotated.sort(key=lambda item: item[0])
    angle_groups: list[list] = []
    for item in annotated:
        if not angle_groups or _angle_gap(item[0], angle_groups[-1][0][0]) > 12:
            angle_groups.append([item])
        else:
            angle_groups[-1].append(item)
    rows = []
    across_gap = 14.0 / gsd_ft
    for group in angle_groups:
        group.sort(key=lambda item: item[2])
        current = [group[0]]
        for item in group[1:]:
            if abs(item[2] - current[-1][2]) > across_gap:
                rows.append([entry[3] for entry in current])
                current = [item]
            else:
                current.append(item)
        rows.append([entry[3] for entry in current])
    return rows


def _frame(corners: list[tuple[float, float]]):
    pts = [np.array(point, dtype=np.float64) for point in corners]
    edge = pts[1] - pts[0]
    other = pts[3] - pts[0]
    if np.linalg.norm(other) > np.linalg.norm(edge):
        edge = other
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)
    center = sum(pts) / 4.0
    theta = np.radians(angle)
    normal = np.array([-np.sin(theta), np.cos(theta)])
    across = float(center @ normal)
    return angle, center, across


def _angle_gap(left: float, right: float) -> float:
    gap = abs(left - right) % 180.0
    return min(gap, 180.0 - gap)


def _weld_row(row: list[list[tuple[float, float]]]) -> list[tuple]:
    """Average the facing corners of neighbours so both stalls reference one point."""
    items = [[(float(x), float(y)) for x, y in corners] for corners in row]
    items.sort(key=lambda corners: float(np.mean([point[0] for point in corners])))
    for index in range(len(items) - 1):
        left, right = items[index], items[index + 1]
        pairs = []
        for i, pa in enumerate(left):
            for j, pb in enumerate(right):
                dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
                pairs.append((dist, i, j))
        pairs.sort()
        used_left, used_right = set(), set()
        chosen = []
        for dist, i, j in pairs:
            if i in used_left or j in used_right:
                continue
            if dist > 6:
                break
            chosen.append((i, j))
            used_left.add(i)
            used_right.add(j)
            if len(chosen) == 2:
                break
        for i, j in chosen:
            shared = (
                (left[i][0] + right[j][0]) / 2.0,
                (left[i][1] + right[j][1]) / 2.0,
            )
            left[i] = shared
            right[j] = shared
    return [tuple(corners) for corners in items]


def _side_lengths(corners) -> tuple[float, float]:
    pts = [np.array(point, dtype=np.float64) for point in corners]
    lengths = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    short, long = sorted(lengths)[0], sorted(lengths)[-1]
    return short, long


def _stall(corners, gsd_ft: float, row_id: int) -> Stall:
    ring = tuple(corners)
    polygon = Polygon(ring)
    short, long = _side_lengths(ring)
    return Stall(
        polygon=polygon,
        width_ft=round(short * gsd_ft, 2),
        length_ft=round(long * gsd_ft, 2),
        corners_px=ring,
        support=1,
        row_id=row_id,
    )
