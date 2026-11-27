"""Find painted stall bays in a 1-foot ortho image.

Stall lines on the NJOGIS 2020 grid are about one pixel wide, so pairing raw
Hough segments mostly follows cars and broken paint. This module keeps pixels
that are brighter than the local asphalt, opens that mask with a short line
kernel, and keeps the orientations whose spacing matches a stall width
(about 8–12 ft) rather than a stall depth (about 18 ft).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon, box

# US commercial stall, in feet. See design.md.
MIN_WIDTH_FT = 7.0
MAX_WIDTH_FT = 13.0
MIN_DEPTH_FT = 12.0
MAX_DEPTH_FT = 22.0
MIN_AREA_SQFT = 80.0


@dataclass(frozen=True)
class Stall:
    polygon: Polygon  # pixel space, x right, y down
    width_ft: float
    length_ft: float


def detect_stalls(image_bgr: np.ndarray, lot: Polygon, gsd_ft: float = 1.0) -> list[Stall]:
    """Return stall polygons in the same pixel space as `image_bgr`."""
    if gsd_ft <= 0:
        raise ValueError("gsd_ft must be positive")
    height, width = image_bgr.shape[:2]
    lot = lot.intersection(box(0, 0, width, height))
    if lot.is_empty or lot.geom_type != "Polygon":
        return []
    scale = _Scale(gsd_ft)
    paint = _paint_mask(image_bgr, lot, scale)
    primary = _best_angle(paint, scale)
    found: list[Polygon] = []
    if primary is None:
        return []
    found.extend(_stalls_at_angle(paint, primary, scale))
    leftover = _residual(paint, primary, scale)
    secondary = _best_angle(leftover, scale)
    if secondary is not None and _angle_distance(secondary, primary) >= 15:
        found.extend(_stalls_at_angle(leftover, secondary, scale))
    return _cleanup(found, lot, scale)


@dataclass(frozen=True)
class _Scale:
    gsd_ft: float

    def px(self, feet: float) -> int:
        return max(1, int(round(feet / self.gsd_ft)))

    def to_ft(self, pixels: float) -> float:
        return pixels * self.gsd_ft


def _paint_mask(image_bgr: np.ndarray, lot: Polygon, scale: _Scale) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    high = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 2.5))
    bright = ((high > 8) & (gray > 80)).astype(np.uint8) * 255
    mask = np.zeros(gray.shape, np.uint8)
    _fill_polygon(mask, lot, 255)
    kernel = scale.px(2) * 2 + 1
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.bitwise_and(bright, cv2.erode(mask, erode))


def _fill_polygon(image: np.ndarray, polygon: Polygon, value: int) -> None:
    exterior = np.round(polygon.exterior.coords).astype(np.int32)
    cv2.fillPoly(image, [exterior], value)
    for hole in polygon.interiors:
        cv2.fillPoly(image, [np.round(hole.coords).astype(np.int32)], 0)


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    size = int(length) + 6
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), np.uint8)
    theta = math.radians(float(angle_deg))
    center = size // 2
    dx = math.cos(theta) * (length / 2)
    dy = math.sin(theta) * (length / 2)
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1,
        1,
    )
    return kernel


def _rotate_vertical(mask: np.ndarray, angle_deg: float):
    """Rotate so lines at `angle_deg` (image y-down atan2) stand vertical.

    OpenCV's positive angle is counter-clockwise in y-down pixel space. A line
    whose direction is `angle_deg` becomes vertical when the image is rotated
    by `angle_deg - 90`. (Rotating by `-angle_deg` only does that near 45°.)
    """
    height, width = mask.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg) - 90.0, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(height * sine + width * cosine) + 2
    new_h = int(height * cosine + width * sine) + 2
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    rotated = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, matrix


def _best_angle(paint: np.ndarray, scale: _Scale) -> float | None:
    kernel_len = scale.px(12)
    best_angle = None
    best_score = 0.0
    min_lag = scale.px(8)
    max_lag = scale.px(12)
    long_lo = scale.px(15)
    long_hi = scale.px(22)
    for angle in range(0, 180, 3):
        opened = cv2.morphologyEx(paint, cv2.MORPH_OPEN, _line_kernel(kernel_len, angle))
        rotated, _ = _rotate_vertical(opened, angle)
        columns = rotated.sum(axis=0).astype(np.float64)
        pixels = int((opened > 0).sum())
        if columns.max() < 5 or pixels < 40 or columns.size < long_hi + 2:
            continue
        centered = columns - columns.mean()
        autocorr = np.correlate(centered, centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        if autocorr.size <= long_hi:
            continue
        width_lag = min_lag + int(np.argmax(autocorr[min_lag : max_lag + 1]))
        length_lag = long_lo + int(np.argmax(autocorr[long_lo : long_hi + 1]))
        # The head-of-row lines repeat at stall depth. Skip that axis.
        if autocorr[length_lag] > autocorr[width_lag] * 1.05:
            continue
        smooth = np.convolve(columns, np.ones(3) / 3.0, mode="same")
        peaks = _local_peaks(smooth, 0.30 * float(smooth.max()), min_gap=scale.px(7))
        if len(peaks) < 5:
            continue
        half = max(width_lag // 2, 1)
        mids = [smooth[min(len(smooth) - 1, peak + half)] for peak in peaks]
        contrast = float(np.mean([smooth[peak] for peak in peaks]) / (np.mean(mids) + 1.0))
        # Contrast is only a gate. Sparse noise can have a huge ratio because the
        # gaps are empty; rank orientations by periodic energy and how much paint
        # supports them instead.
        if contrast < 1.4:
            continue
        score = (
            float(autocorr[width_lag])
            * math.log(len(peaks) + 1.0)
            * math.log(pixels + 1.0)
        )
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def _local_peaks(values: np.ndarray, threshold: float, min_gap: int) -> list[int]:
    peaks: list[int] = []
    for index in range(1, len(values) - 1):
        if values[index] < threshold:
            continue
        if values[index] >= values[index - 1] and values[index] > values[index + 1]:
            if not peaks or index - peaks[-1] >= min_gap:
                peaks.append(index)
    return peaks


def _angle_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 180.0
    return min(delta, 180.0 - delta)


def _residual(paint: np.ndarray, angle: float, scale: _Scale) -> np.ndarray:
    kernel_len = scale.px(12)
    accumulated = np.zeros_like(paint)
    for delta in (-6, -3, 0, 3, 6):
        opened = cv2.morphologyEx(paint, cv2.MORPH_OPEN, _line_kernel(kernel_len, angle + delta))
        accumulated = cv2.bitwise_or(accumulated, opened)
    radius = scale.px(2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.subtract(paint, cv2.dilate(accumulated, kernel))


def _stalls_at_angle(paint: np.ndarray, angle: float, scale: _Scale) -> list[Polygon]:
    opened = cv2.morphologyEx(paint, cv2.MORPH_OPEN, _line_kernel(scale.px(12), angle))
    rotated, matrix = _rotate_vertical(opened, angle)
    close_len = scale.px(5)
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_len))
    rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, vertical)
    row_energy = rotated.sum(axis=1).astype(np.float64)
    row_energy = np.convolve(row_energy, np.ones(scale.px(5)) / scale.px(5), mode="same")
    polygons: list[Polygon] = []
    for y0, y1 in _bands(row_energy, scale):
        columns = rotated[y0:y1].sum(axis=0).astype(np.float64)
        if columns.max() < 8:
            continue
        smooth = np.convolve(columns, np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
        centered = columns - columns.mean()
        autocorr = np.correlate(centered, centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        min_lag, max_lag = scale.px(8), scale.px(12)
        if autocorr.size <= max_lag or autocorr[min_lag : max_lag + 1].max() <= 0:
            continue
        period = min_lag + int(np.argmax(autocorr[min_lag : max_lag + 1]))
        for chain in _chains(smooth, period, scale):
            polygons.extend(_rectangles(chain, y0, y1, rotated, matrix, scale))
    return polygons


def _bands(row_energy: np.ndarray, scale: _Scale) -> list[tuple[int, int]]:
    positive = row_energy[row_energy > 0]
    if positive.size == 0:
        return []
    threshold = max(8.0, 0.18 * float(np.percentile(positive, 75)))
    active = row_energy >= threshold
    min_h = scale.px(MIN_DEPTH_FT)
    max_h = scale.px(MAX_DEPTH_FT + 4)
    max_gap = scale.px(3)
    spans: list[tuple[int, int]] = []
    index = 0
    limit = len(active)
    while index < limit:
        if not active[index]:
            index += 1
            continue
        end = index
        gap = 0
        while end < limit:
            if active[end]:
                gap = 0
                end += 1
            elif gap < max_gap:
                gap += 1
                end += 1
            else:
                end -= gap
                break
        if end - index >= min_h:
            spans.append((index, end))
        index = max(end, index + 1)
    split: list[tuple[int, int]] = []
    for start, end in spans:
        split.extend(_split_band(row_energy, start, end, min_h, max_h))
    return sorted(set(split))


def _split_band(row_energy, start, end, min_h, max_h) -> list[tuple[int, int]]:
    height = end - start
    if height <= max_h:
        return [(start, end)]
    guard = min_h // 2
    if height < 2 * guard + 4:
        return [(start, end)]
    segment = row_energy[start:end]
    interior = segment.copy()
    interior[:guard] = interior.max()
    interior[-guard:] = interior.max()
    valley = int(np.argmin(interior))
    if segment[valley] < 0.5 * float(np.median(segment)):
        left = _split_band(row_energy, start, start + valley, min_h, max_h)
        right = _split_band(row_energy, start + valley + 1, end, min_h, max_h)
        return [span for span in left + right if span[1] - span[0] >= min_h]
    window = np.convolve(segment, np.ones(max_h), mode="valid")
    offset = int(np.argmax(window))
    return [(start + offset, start + offset + max_h)]


def _chains(smooth: np.ndarray, period: int, scale: _Scale) -> list[list[int]]:
    if smooth.max() <= 0:
        return []
    candidates = _local_peaks(smooth, 0.22 * float(smooth.max()), min_gap=scale.px(6))
    if len(candidates) < 2:
        return []
    remaining = set(candidates)
    chains: list[list[int]] = []
    while remaining:
        chain = _best_chain(smooth, period, remaining, scale)
        if len(chain) < 2:
            break
        chains.append(chain)
        remaining = {item for item in remaining if all(abs(item - used) > 3 for used in chain)}
    return chains


def _best_chain(smooth, period, allowed: set[int], scale: _Scale) -> list[int]:
    best: list[int] = []
    best_score = -1.0
    tolerance = scale.px(2)
    for seed in sorted(allowed):
        forward = _walk(smooth, period, seed, allowed, tolerance, scale, direction=1)
        backward = _walk(smooth, period, seed, allowed, tolerance, scale, direction=-1)
        # `_walk` already returns positions in increasing x, seed included on both sides.
        chain = backward[:-1] + forward
        cleaned: list[int] = []
        for position in chain:
            if not cleaned or position - cleaned[-1] >= scale.px(7):
                cleaned.append(position)
        score = float(sum(smooth[min(len(smooth) - 1, int(pos))] for pos in cleaned)) * len(cleaned)
        if len(cleaned) >= 2 and score > best_score:
            best_score = score
            best = cleaned
    return best


def _walk(smooth, period, seed, allowed, tolerance, scale: _Scale, direction: int) -> list[int]:
    chain = [seed]
    current = seed
    min_step = scale.px(7)
    max_step = scale.px(14)
    while True:
        target = current + direction * period
        nxt = [
            item
            for item in allowed
            if min_step <= direction * (item - current) <= max_step
            and abs(item - target) <= tolerance + 1
        ]
        if nxt:
            nxt.sort(key=lambda item: (abs(item - target), -smooth[item]))
            current = nxt[0]
            chain.append(current)
            continue
        target2 = current + direction * 2 * period
        nxt2 = [
            item
            for item in allowed
            if abs(item - target2) <= tolerance + scale.px(2)
            and direction * (item - current) > max_step
        ]
        midpoint = current + direction * period
        if nxt2 and 0 <= midpoint < len(smooth) and smooth[midpoint] >= 0.12 * smooth.max():
            chain.append(int(midpoint))
            nxt2.sort(key=lambda item: abs(item - target2))
            current = nxt2[0]
            chain.append(current)
            continue
        break
    if direction < 0:
        chain.reverse()
    return chain


def _rectangles(peaks, y0, y1, rotated, matrix, scale: _Scale) -> list[Polygon]:
    polygons = []
    min_w, max_w = scale.px(MIN_WIDTH_FT), scale.px(MAX_WIDTH_FT)
    min_d, max_d = scale.px(MIN_DEPTH_FT), scale.px(MAX_DEPTH_FT)
    for left, right in zip(peaks, peaks[1:]):
        if not min_w <= (right - left) <= max_w:
            continue
        strip = rotated[y0:y1, max(0, left - 1) : right + 2]
        if strip.size == 0 or int(strip.sum()) < 8:
            continue
        rows_on = np.where(strip.sum(axis=1) > 0)[0]
        if rows_on.size:
            top = y0 + int(rows_on[0])
            bottom = y0 + int(rows_on[-1]) + 1
        else:
            top, bottom = y0, y1
        depth = bottom - top
        if depth < min_d:
            if min_d <= (y1 - y0) <= max_d:
                top, bottom = y0, y1
                depth = bottom - top
            else:
                continue
        if depth > max_d:
            center = (top + bottom) / 2
            top = int(round(center - max_d / 2))
            bottom = top + max_d
        corners = _invert(matrix, [(left, top), (right, top), (right, bottom), (left, bottom)])
        polygon = Polygon(corners)
        if polygon.is_valid and scale.to_ft(polygon.area) >= MIN_AREA_SQFT * 0.5:
            polygons.append(polygon)
    return polygons


def _invert(matrix: np.ndarray, points) -> np.ndarray:
    linear = matrix[:, :2]
    offset = matrix[:, 2]
    inverse = np.linalg.inv(linear)
    points = np.asarray(points, dtype=np.float64)
    return (points - offset) @ inverse.T


def _cleanup(polygons: list[Polygon], lot: Polygon, scale: _Scale) -> list[Stall]:
    stalls: list[Stall] = []
    for polygon in polygons:
        if not polygon.is_valid or polygon.area <= 0:
            continue
        clipped = polygon.intersection(lot)
        clipped = _largest_polygon(clipped)
        if clipped is None:
            continue
        if clipped.area < 0.55 * polygon.area:
            continue
        area_ft = clipped.area * scale.gsd_ft * scale.gsd_ft
        if area_ft < MIN_AREA_SQFT:
            continue
        width_ft, length_ft = _side_lengths(clipped, scale)
        if width_ft <= 0:
            continue
        aspect = length_ft / width_ft
        if aspect < 1.25 or aspect > 3.6:
            continue
        if not MIN_WIDTH_FT * 0.85 <= width_ft <= MAX_WIDTH_FT * 1.15:
            continue
        stalls.append(Stall(clipped, width_ft, length_ft))
    stalls.sort(key=lambda stall: -stall.polygon.area)
    kept: list[Stall] = []
    for stall in stalls:
        overlap = False
        for other in kept:
            shared = stall.polygon.intersection(other.polygon).area
            smaller = min(stall.polygon.area, other.polygon.area)
            if shared > 0.45 * smaller:
                overlap = True
                break
        if overlap:
            continue
        kept.append(stall)
    return kept


def _largest_polygon(geometry):
    if geometry.is_empty:
        return None
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        return max(geometry.geoms, key=lambda item: item.area)
    if geometry.geom_type == "GeometryCollection":
        parts = [_largest_polygon(part) for part in geometry.geoms]
        parts = [part for part in parts if part is not None]
        if not parts:
            return None
        return max(parts, key=lambda item: item.area)
    return None


def _side_lengths(polygon: Polygon, scale: _Scale) -> tuple[float, float]:
    rectangle = polygon.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0
    edges = []
    for start, end in zip(coords, coords[1:]):
        edges.append(math.hypot(end[0] - start[0], end[1] - start[1]))
    edges = [edge for edge in edges if edge > 0]
    if len(edges) < 2:
        return 0.0, 0.0
    short, long = sorted(edges)[0], sorted(edges)[-1]
    return round(scale.to_ft(short), 2), round(scale.to_ft(long), 2)
