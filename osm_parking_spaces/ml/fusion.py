"""Fuse classical stalls with a learned probability map (v4).

Corners follow one convention everywhere in this module: ``c0 -> c1`` is the width edge along
the row, ``c0 -> c3`` is the depth edge (the longer pair, the divider direction). Neighbours in a
row share ``c1 == c0'`` and ``c2 == c3'``. All steps are pure NumPy, OpenCV and shapely, so they
are tested in CI with synthetic probability maps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from osm_parking_spaces.detect import Stall

PITCH_FT = 9.5
MIN_DEPTH_FT = 10.0
MAX_DEPTH_FT = 24.0
MIN_WIDTH_FT = 6.0
MAX_WIDTH_FT = 16.0
MIN_AREA_SQFT = 50.0


@dataclass(frozen=True)
class FusionConfig:
    """Thresholds. `seed=None` turns the model-seeded rows off."""

    keep: float = 0.45
    reject: float = 0.20
    fill: float = 0.50
    seed: float | None = 0.60
    pixel: float = 0.50
    max_fill: int = 6
    max_trim_ft: float = 6.0
    trim: bool = True

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load_for(cls, weights: Path, pixel: float = 0.5) -> FusionConfig:
        """`fusion.json` next to the weights if it exists, otherwise the defaults."""
        path = Path(weights).parent / "fusion.json"
        if path.exists():
            names = {field.name for field in fields(cls)}
            data = {k: v for k, v in json.loads(path.read_text()).items() if k in names}
            return cls(**data)
        return cls(pixel=pixel)


@dataclass
class _Item:
    corners: np.ndarray  # (4, 2) float64, canonical order
    row: int
    score: float = 0.0


def fuse(
    stalls: list[Stall],
    probs: np.ndarray,
    lot: Polygon,
    gsd_ft: float,
    config: FusionConfig | None = None,
) -> list[Stall]:
    """Confirm or reject, trim, fill, then optionally add model-seeded rows."""
    config = config or FusionConfig()
    items = [_Item(canonical(np.asarray(s.corners_px, float)), s.row_id) for s in stalls]
    items = _confirm(items, probs, config)
    if config.trim:
        items = _trim(items, probs, gsd_ft, config)
    items = _fill(items, probs, lot, gsd_ft, config)
    if config.seed is not None:
        seeds = _seeds(items, probs, lot, gsd_ft, config)
        if config.trim:
            seeds = _trim(seeds, probs, gsd_ft, config)
        items = _fill(items + seeds, probs, lot, gsd_ft, config)
    return [_to_stall(item, gsd_ft) for item in items if _mostly_inside(item, lot)]


def canonical(corners: np.ndarray) -> np.ndarray:
    """Order so that c0->c3 is the longer edge pair (depth). Winding is kept."""
    corners = np.asarray(corners, float)
    along = np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[3] - corners[2])
    across = np.linalg.norm(corners[2] - corners[1]) + np.linalg.norm(corners[0] - corners[3])
    if along > across:
        corners = np.roll(corners, -1, axis=0)
    return corners.copy()


def mean_prob(probs: np.ndarray, polygon: Polygon, shrink: float = 1.5) -> float:
    """Mean probability inside the polygon shrunk by `shrink` px."""
    inner = polygon.buffer(-shrink)
    if inner.is_empty or inner.geom_type != "Polygon":
        inner = polygon
    minx, miny, maxx, maxy = inner.bounds
    x0, y0 = max(int(np.floor(minx)), 0), max(int(np.floor(miny)), 0)
    x1, y1 = (
        min(int(np.ceil(maxx)) + 1, probs.shape[1]),
        min(int(np.ceil(maxy)) + 1, probs.shape[0]),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    ring = np.round(np.asarray(inner.exterior.coords) - [x0, y0]).astype(np.int32)
    cv2.fillPoly(local, [ring], 1)
    count = int(local.sum())
    if count == 0:
        return 0.0
    return float(probs[y0:y1, x0:x1][local > 0].sum() / count)


def _poly(corners: np.ndarray) -> Polygon:
    return Polygon([tuple(point) for point in corners])


def _vectors(corners: np.ndarray):
    width = ((corners[1] - corners[0]) + (corners[2] - corners[3])) / 2.0
    depth = ((corners[3] - corners[0]) + (corners[2] - corners[1])) / 2.0
    return width, depth


def _by_row(items: list[_Item]) -> dict[int, list[_Item]]:
    rows: dict[int, list[_Item]] = {}
    for item in items:
        rows.setdefault(item.row, []).append(item)
    return rows


def _confirm(items: list[_Item], probs: np.ndarray, config: FusionConfig) -> list[_Item]:
    for item in items:
        item.score = mean_prob(probs, _poly(item.corners))
    kept: list[_Item] = []
    for row in _by_row(items).values():
        median = float(np.median([item.score for item in row]))
        for item in row:
            if item.score >= config.keep:
                kept.append(item)
            elif item.score >= config.reject and median >= config.keep:
                kept.append(item)
    return kept


def _depth_run(probs: np.ndarray, corners: np.ndarray, gsd_ft: float, threshold: float):
    """Start and end (px along the depth axis) of the high-probability run, or None."""
    width, depth = _vectors(corners)
    length = float(np.linalg.norm(depth))
    breadth = float(np.linalg.norm(width))
    if length < 1 or breadth < 1:
        return None
    u = depth / length
    uw = width / breadth
    reach = max(2, int(round(4.0 / gsd_ft)))
    ts = np.arange(-reach, int(np.ceil(length)) + reach + 1, dtype=np.float64)
    ss = np.linspace(-0.3 * breadth, 0.3 * breadth, 5)
    top_mid = (corners[0] + corners[1]) / 2.0
    pts = top_mid[None, None, :] + ts[:, None, None] * u + ss[None, :, None] * uw
    xs = np.rint(pts[..., 0]).astype(int)
    ys = np.rint(pts[..., 1]).astype(int)
    inside = (xs >= 0) & (xs < probs.shape[1]) & (ys >= 0) & (ys < probs.shape[0])
    values = np.where(
        inside, probs[np.clip(ys, 0, probs.shape[0] - 1), np.clip(xs, 0, probs.shape[1] - 1)], 0.0
    )
    profile = values.mean(axis=1)
    on = profile >= threshold
    best = None
    best_overlap = 0.0
    index = 0
    while index < len(on):
        if not on[index]:
            index += 1
            continue
        end = index
        while end < len(on) and on[end]:
            end += 1
        start_t, end_t = ts[index], ts[end - 1] + 1.0
        overlap = max(0.0, min(end_t, length) - max(start_t, 0.0))
        if overlap > best_overlap:
            best_overlap, best = overlap, (start_t, end_t)
        index = end
    if best is None or best_overlap < 0.4 * length:
        return None
    return best[0], best[1], length, u


def _trim(
    items: list[_Item], probs: np.ndarray, gsd_ft: float, config: FusionConfig
) -> list[_Item]:
    """Move each row's near and far edges to where the probability says the stalls end.

    One translation per row, applied to every stall, so shared corners stay equal.
    """
    limit = config.max_trim_ft / gsd_ft
    for row in _by_row(items).values():
        runs = [_depth_run(probs, item.corners, gsd_ft, config.pixel) for item in row]
        runs = [run for run in runs if run is not None]
        if not runs:
            continue
        top = float(np.median([run[0] for run in runs]))
        bottom = float(np.median([run[1] - run[2] for run in runs]))
        top = float(np.clip(top, -limit, limit))
        bottom = float(np.clip(bottom, -limit, limit))
        if abs(top) < 1.5:
            top = 0.0
        if abs(bottom) < 1.5:
            bottom = 0.0
        if top == 0.0 and bottom == 0.0:
            continue
        axis = np.mean(
            [
                _vectors(item.corners)[1] / max(np.linalg.norm(_vectors(item.corners)[1]), 1e-9)
                for item in row
            ],
            axis=0,
        )
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        new_depth = (
            float(np.mean([np.linalg.norm(_vectors(i.corners)[1]) for i in row])) - top + bottom
        )
        if not MIN_DEPTH_FT <= new_depth * gsd_ft <= MAX_DEPTH_FT:
            continue
        for item in row:
            corners = item.corners.copy()
            corners[0] += axis * top
            corners[1] += axis * top
            corners[2] += axis * bottom
            corners[3] += axis * bottom
            item.corners = corners
    return items


def _overlaps(candidate: Polygon, others: list[Polygon]) -> bool:
    for other in others:
        inter = candidate.intersection(other).area
        if inter <= 0:
            continue
        union = candidate.area + other.area - inter
        if inter / union > 0.15 or inter > 0.25 * min(candidate.area, other.area):
            return True
    return False


def _mostly_inside(item: _Item, lot: Polygon) -> bool:
    polygon = _poly(item.corners)
    return (
        polygon.is_valid
        and polygon.area > 0
        and polygon.intersection(lot).area >= 0.5 * polygon.area
    )


def _step(corners: np.ndarray, pitch: np.ndarray, direction: int) -> np.ndarray:
    """The neighbour one pitch along the row, sharing the near corners exactly."""
    if direction > 0:
        return np.array([corners[1], corners[1] + pitch, corners[2] + pitch, corners[2]])
    return np.array([corners[0] - pitch, corners[0], corners[3], corners[3] - pitch])


def _fill(
    items: list[_Item], probs: np.ndarray, lot: Polygon, gsd_ft: float, config: FusionConfig
) -> list[_Item]:
    """Extend rows past their ends, and across interior gaps, at the row's own pitch."""
    polygons = [_poly(item.corners) for item in items]
    added: list[_Item] = []
    for row_id, row in _by_row(items).items():
        if len(row) < 2:
            continue
        pitch = np.mean([_vectors(item.corners)[0] for item in row], axis=0)
        pitch_len = float(np.linalg.norm(pitch))
        if pitch_len < 1:
            continue
        unit = pitch / pitch_len
        row.sort(key=lambda item: float(item.corners.mean(axis=0) @ unit))
        # Interior gaps first.
        for left, right in zip(row, row[1:]):
            gap = float((right.corners.mean(axis=0) - left.corners.mean(axis=0)) @ unit) / pitch_len
            steps = int(round(gap)) - 1
            previous = left.corners
            for index in range(max(0, min(steps, config.max_fill))):
                corners = _step(previous, pitch, +1)
                if (
                    index == steps - 1
                    and np.linalg.norm(corners[1] - right.corners[0]) <= 0.4 * pitch_len
                ):
                    corners[1], corners[2] = right.corners[0].copy(), right.corners[3].copy()
                if not _accept(corners, probs, lot, polygons, config, require_gap=True):
                    break
                polygons.append(_poly(corners))
                added.append(_Item(corners, row_id, mean_prob(probs, polygons[-1])))
                previous = corners
        for direction, end in ((+1, row[-1]), (-1, row[0])):
            previous = end.corners
            for _ in range(config.max_fill):
                corners = _step(previous, pitch, direction)
                if not _accept(corners, probs, lot, polygons, config):
                    break
                polygons.append(_poly(corners))
                added.append(_Item(corners, row_id, mean_prob(probs, polygons[-1])))
                previous = corners
    return items + added


def _accept(corners, probs, lot, polygons, config: FusionConfig, require_gap: bool = False) -> bool:
    polygon = _poly(corners)
    if not polygon.is_valid or polygon.area <= 0:
        return False
    if polygon.intersection(lot).area < 0.6 * polygon.area:
        return False
    if mean_prob(probs, polygon) < config.fill:
        return False
    # Neighbours share an edge, so only a real overlap counts. Shrink slightly before testing.
    core = polygon.buffer(-1.0)
    if core.is_empty or core.geom_type != "Polygon":
        core = polygon
    return not _overlaps(core, polygons)


def _fit_quad(hull: np.ndarray) -> np.ndarray:
    """A 4-point polygon for a component: a parallelogram when it fits better than a rectangle."""
    perimeter = cv2.arcLength(hull, True)
    hull_poly = Polygon(hull.reshape(-1, 2))
    rect = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
    best, best_iou = rect, _iou(Polygon(rect), hull_poly)
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10, 0.14):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True).reshape(-1, 2)
        if len(approx) == 4:
            quad = approx.astype(np.float64)
            poly = Polygon(quad)
            if poly.is_valid and poly.area > 0:
                iou = _iou(poly, hull_poly)
                if iou >= best_iou + 0.03:
                    best, best_iou = quad, iou
            break
    return best


def _iou(left: Polygon, right: Polygon) -> float:
    if not (left.is_valid and right.is_valid):
        return 0.0
    inter = left.intersection(right).area
    union = left.area + right.area - inter
    return inter / union if union > 0 else 0.0


def _seeds(
    items: list[_Item], probs: np.ndarray, lot: Polygon, gsd_ft: float, config: FusionConfig
) -> list[_Item]:
    """Rows from model pixels that no kept stall covers."""
    height, width = probs.shape
    covered = np.zeros((height, width), np.uint8)
    for item in items:
        cv2.fillPoly(covered, [np.round(item.corners).astype(np.int32)], 1)
    covered = cv2.dilate(covered, np.ones((3, 3), np.uint8))
    inside = np.zeros((height, width), np.uint8)
    cv2.fillPoly(inside, [np.round(np.asarray(lot.exterior.coords)).astype(np.int32)], 1)
    mask = ((probs >= config.pixel) & (covered == 0) & (inside > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    found: list[np.ndarray] = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] * gsd_ft * gsd_ft < MIN_AREA_SQFT:
            continue
        contours, _h = cv2.findContours(
            (labels == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        if len(hull) < 3:
            continue
        found.extend(_split_row(_orient_component(_fit_quad(hull), gsd_ft), probs, gsd_ft))
    existing = [_poly(item.corners) for item in items]
    accepted: list[_Item] = []
    for corners in found:
        polygon = _poly(corners)
        if not polygon.is_valid or polygon.area <= 0:
            continue
        width_ft = float(np.linalg.norm(_vectors(corners)[0])) * gsd_ft
        depth_ft = float(np.linalg.norm(_vectors(corners)[1])) * gsd_ft
        if not (
            MIN_WIDTH_FT <= width_ft <= MAX_WIDTH_FT
            and MIN_DEPTH_FT + 1 <= depth_ft <= MAX_DEPTH_FT + 2
        ):
            continue
        if polygon.intersection(lot).area < 0.55 * polygon.area:
            continue
        if mean_prob(probs, polygon) < config.seed:
            continue
        core = polygon.buffer(-1.0)
        if not core.is_empty and core.geom_type == "Polygon" and _overlaps(core, existing):
            continue
        existing.append(polygon)
        accepted.append(_Item(corners, -1, mean_prob(probs, polygon)))
    return _group_and_weld(accepted, gsd_ft)


def _orient_component(quad: np.ndarray, gsd_ft: float) -> np.ndarray:
    """A component deep like one stall and long like several is a fused row.

    For a fused row the long edge is the row (c0 -> c1). For a single stall the long edge is
    the depth, which is the usual convention.
    """
    a = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[3] - quad[2])) / 2.0
    b = (np.linalg.norm(quad[2] - quad[1]) + np.linalg.norm(quad[0] - quad[3])) / 2.0
    long_ft, short_ft = max(a, b) * gsd_ft, min(a, b) * gsd_ft
    if long_ft >= 2.6 * PITCH_FT and MIN_DEPTH_FT <= short_ft <= MAX_DEPTH_FT + 4:
        return quad.copy() if a >= b else np.roll(quad, -1, axis=0).copy()
    return canonical(quad)


def _dividers(quad: np.ndarray, probs: np.ndarray, gsd_ft: float) -> list[float]:
    """Fractions (0..1) along the row where the probability dips between neighbours."""
    top0, top1, bot1, bot0 = quad
    length = float(np.linalg.norm(top1 - top0))
    steps = max(int(round(length)), 2)
    values = []
    for index in range(steps + 1):
        t = index / steps
        column = []
        for depth in (0.3, 0.5, 0.7):
            point = (top0 + (top1 - top0) * t) * (1 - depth) + (bot0 + (bot1 - bot0) * t) * depth
            x, y = int(round(point[0])), int(round(point[1]))
            inside = 0 <= y < probs.shape[0] and 0 <= x < probs.shape[1]
            column.append(float(probs[y, x]) if inside else 0.0)
        values.append(min(column))
    profile = np.convolve(values, [0.25, 0.5, 0.25], mode="same")
    level = float(np.median(profile))
    min_gap = max(int(round(6.0 / gsd_ft)), 2)
    cuts: list[int] = []
    for index in range(2, steps - 1):
        if profile[index] < 0.85 * level and profile[index] <= profile[index - 1]:
            if profile[index] <= profile[index + 1]:
                if not cuts or index - cuts[-1] >= min_gap:
                    cuts.append(index)
    edges = [0] + cuts + [steps]
    widths = [(b - a) * gsd_ft for a, b in zip(edges, edges[1:])]
    if cuts and all(MIN_WIDTH_FT <= width <= MAX_WIDTH_FT for width in widths):
        return [edge / steps for edge in edges]
    return []


def _split_row(corners: np.ndarray, probs: np.ndarray, gsd_ft: float) -> list[np.ndarray]:
    """A component as wide as several stalls becomes that many, with shared dividers."""
    width, _depth = _vectors(corners)
    width_ft = float(np.linalg.norm(width)) * gsd_ft
    fractions = [0.0, 1.0]
    if width_ft >= 1.8 * PITCH_FT:
        found = _dividers(corners, probs, gsd_ft)
        if found:
            fractions = found
        else:
            count = max(1, int(round(width_ft / PITCH_FT)))
            fractions = [i / count for i in range(count + 1)]
    top0, top1, bot1, bot0 = corners
    pieces = []
    for a, b in zip(fractions, fractions[1:]):
        pieces.append(
            np.array(
                [
                    top0 + (top1 - top0) * a,
                    top0 + (top1 - top0) * b,
                    bot0 + (bot1 - bot0) * b,
                    bot0 + (bot1 - bot0) * a,
                ]
            )
        )
    return pieces


def _group_and_weld(items: list[_Item], gsd_ft: float) -> list[_Item]:
    """Link side-by-side stalls into rows and make facing corners one point."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(items):
        wa, da = _vectors(a.corners)
        for j in range(i + 1, len(items)):
            b = items[j]
            wb, db = _vectors(b.corners)
            if _angle_between(da, db) > 12.0 or _angle_between(wa, wb) > 12.0:
                continue
            delta = b.corners.mean(axis=0) - a.corners.mean(axis=0)
            wl, dl = float(np.linalg.norm(wa)), float(np.linalg.norm(da))
            along = abs(float(delta @ (wa / wl)))
            if abs(along - wl) > 0.4 * wl:
                continue
            # Depth offset measured along the divider direction.
            across = np.linalg.norm(delta - (delta @ (wa / wl)) * (wa / wl))
            if across > 0.45 * dl:
                continue
            parent[find(j)] = find(i)
    rows: dict[int, list[_Item]] = {}
    for i, item in enumerate(items):
        rows.setdefault(find(i), []).append(item)
    out: list[_Item] = []
    next_id = 100000
    for members in rows.values():
        pitch = np.mean([_vectors(m.corners)[0] for m in members], axis=0)
        unit = pitch / max(float(np.linalg.norm(pitch)), 1e-9)
        members.sort(key=lambda m: float(m.corners.mean(axis=0) @ unit))
        for left, right in zip(members, members[1:]):
            limit = 0.4 * float(np.linalg.norm(pitch))
            for a_idx, b_idx in ((1, 0), (2, 3)):
                if np.linalg.norm(left.corners[a_idx] - right.corners[b_idx]) <= limit:
                    shared = (left.corners[a_idx] + right.corners[b_idx]) / 2.0
                    left.corners[a_idx] = shared
                    right.corners[b_idx] = shared.copy()
        for member in members:
            member.row = next_id
        next_id += 1
        out.extend(members)
    return out


def _angle_between(left: np.ndarray, right: np.ndarray) -> float:
    cosine = abs(float(left @ right)) / max(
        float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-9
    )
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def _to_stall(item: _Item, gsd_ft: float) -> Stall:
    width, depth = _vectors(item.corners)
    ring = tuple((float(x), float(y)) for x, y in item.corners)
    return Stall(
        polygon=Polygon(ring),
        width_ft=round(float(np.linalg.norm(width)) * gsd_ft, 2),
        length_ft=round(float(np.linalg.norm(depth)) * gsd_ft, 2),
        corners_px=ring,
        support=1,
        row_id=int(item.row),
    )


def with_config(config: FusionConfig, **changes) -> FusionConfig:
    return replace(config, **changes)
