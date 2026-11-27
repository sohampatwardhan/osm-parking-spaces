"""Find painted stall bays in a 1-foot ortho image.

Each row keeps its own divider angle. Stall width is snapped to that row's
pitch and stall depth is cut by the row's end lines, so a bay is a rectangle
or a parallelogram and does not run into the aisle. Missing dividers inside
the row are filled at the same pitch, including bays hidden by cars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon, box

# US commercial stall, in feet. See design.md.
MIN_WIDTH_FT = 7.5
MAX_WIDTH_FT = 12.5
MIN_DEPTH_FT = 12.0
MAX_DEPTH_FT = 22.0
MIN_AREA_SQFT = 80.0
MIN_DIVIDERS = 3


@dataclass(frozen=True)
class Stall:
    polygon: Polygon
    width_ft: float
    length_ft: float
    corners_px: tuple[tuple[float, float], ...] = ()
    support: int = 1
    row_id: int = 0


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
    candidates: list[Stall] = []
    for angle in range(0, 180, 3):
        candidates.extend(_rows_at_angle(paint, float(angle), scale, lot))
    return _nms(candidates)


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
    exterior = np.round(lot.exterior.coords).astype(np.int32)
    cv2.fillPoly(mask, [exterior], 255)
    for hole in lot.interiors:
        cv2.fillPoly(mask, [np.round(hole.coords).astype(np.int32)], 0)
    kernel = scale.px(2) * 2 + 1
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    return cv2.bitwise_and(bright, cv2.erode(mask, erode))


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    size = int(length) + 6
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), np.uint8)
    theta = math.radians(float(angle_deg))
    center = size // 2
    dx = math.cos(theta) * (length / 2.0)
    dy = math.sin(theta) * (length / 2.0)
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1,
        1,
    )
    return kernel


def _rotate_vertical(mask: np.ndarray, angle_deg: float):
    """Rotate so lines at `angle_deg` (image y-down) stand vertical.

    OpenCV's positive angle is counter-clockwise. A line at `angle_deg`
    becomes vertical under a rotation of `angle_deg - 90`.
    """
    height, width = mask.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg) - 90.0, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(height * sine + width * cosine) + 2
    new_h = int(height * cosine + width * sine) + 2
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, matrix


def _invert(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    linear = np.linalg.inv(matrix[:, :2])
    point = np.array([x, y], dtype=np.float64) - matrix[:, 2]
    col, row = point @ linear.T
    return float(col), float(row)


def _local_peaks(values: np.ndarray, threshold: float, min_gap: int) -> list[int]:
    peaks: list[int] = []
    for index in range(1, len(values) - 1):
        if values[index] < threshold:
            continue
        if values[index] >= values[index - 1] and values[index] > values[index + 1]:
            if not peaks or index - peaks[-1] >= min_gap:
                peaks.append(index)
    return peaks


def _bands(row_energy: np.ndarray, scale: _Scale) -> list[tuple[int, int]]:
    positive = row_energy[row_energy > 0]
    if positive.size == 0:
        return []
    threshold = max(8.0, 0.22 * float(np.percentile(positive, 70)))
    active = row_energy >= threshold
    min_h = scale.px(MIN_DEPTH_FT - 1)
    max_h = scale.px(MAX_DEPTH_FT + 2)
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
            spans.extend(_split_band(row_energy, index, end, min_h, max_h))
        index = max(end, index + 1)
    return spans


def _split_band(row_energy, start: int, end: int, min_h: int, max_h: int) -> list[tuple[int, int]]:
    if end - start <= max_h:
        return [(start, end)] if end - start >= min_h else []
    guard = max(min_h // 2, 4)
    segment = row_energy[start:end]
    if end - start < 2 * guard + 4:
        return []
    interior = segment.copy()
    interior[:guard] = interior.max()
    interior[-guard:] = interior.max()
    valley = int(np.argmin(interior))
    if segment[valley] <= 0.55 * float(np.median(segment)):
        left = _split_band(row_energy, start, start + valley, min_h, max_h)
        right = _split_band(row_energy, start + valley + 1, end, min_h, max_h)
        return left + right
    window = np.convolve(segment, np.ones(max_h), mode="valid")
    offset = int(np.argmax(window))
    return [(start + offset, start + offset + max_h)]


def _rows_at_angle(paint: np.ndarray, angle: float, scale: _Scale, lot: Polygon) -> list[Stall]:
    opened = cv2.morphologyEx(paint, cv2.MORPH_OPEN, _line_kernel(scale.px(10), angle))
    if int((opened > 0).sum()) < 30:
        return []
    rotated, matrix = _rotate_vertical(opened, angle)
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale.px(2)))
    rotated = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, close)
    energy = np.convolve(rotated.sum(axis=1).astype(np.float64), np.ones(5) / 5.0, mode="same")
    stalls: list[Stall] = []
    for y0, y1 in _bands(energy, scale):
        stalls.extend(_stalls_in_band(rotated, y0, y1, matrix, scale, lot))
    return stalls


def _stalls_in_band(rotated, y0: int, y1: int, matrix, scale: _Scale, lot: Polygon) -> list[Stall]:
    profile = rotated[y0:y1].sum(axis=0).astype(np.float64)
    if profile.max() < 8 or profile.size < scale.px(24):
        return []
    smooth = np.convolve(profile, np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0, mode="same")
    centered = profile - profile.mean()
    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    width_lo, width_hi = scale.px(MIN_WIDTH_FT), scale.px(MAX_WIDTH_FT)
    short_lo, short_hi = scale.px(2.5), scale.px(6.5)
    if autocorr.size <= width_hi + 2:
        return []
    width_lag = width_lo + int(np.argmax(autocorr[width_lo : width_hi + 1]))
    short_lag = short_lo + int(np.argmax(autocorr[short_lo : short_hi + 1]))
    # Hatches and crosswalks repeat every few feet. Their 2nd harmonic can
    # land on a stall width, so a stronger short period rejects the band.
    if autocorr[short_lag] > autocorr[width_lag] * 0.8:
        return []
    pitch = float(width_lag)
    peak_floor = max(8.0, 0.28 * float(smooth.max()))
    peaks = _local_peaks(smooth, peak_floor, min_gap=scale.px(6.5))
    if len(peaks) < MIN_DIVIDERS:
        return []
    diffs = np.diff(peaks)
    pitch_diffs = [float(diff) for diff in diffs if width_lo - 0.5 <= diff <= width_hi + 0.5]
    if len(pitch_diffs) >= 2:
        pitch = float(np.median(pitch_diffs))
    if not (width_lo - 0.5) <= pitch <= (width_hi + 0.5):
        return []
    stalls: list[Stall] = []
    for group in _peak_groups(peaks, max_gap=pitch * 3.4):
        if len(group) < MIN_DIVIDERS:
            continue
        lattice = _lattice(smooth, group, pitch)
        if len(lattice) < MIN_DIVIDERS:
            continue
        matched = sum(1 for item in lattice if min(abs(item - peak) for peak in group) <= 1.75)
        if matched < MIN_DIVIDERS:
            continue
        unsupported = 0
        for left, right in zip(lattice, lattice[1:]):
            near = any(abs(peak - left) <= 1.75 or abs(peak - right) <= 1.75 for peak in group)
            between = any(left < peak < right for peak in group)
            if not near and not between:
                unsupported += 1
        # Gap-fill inserts dividers between real ones. A long stretch with no
        # paint at all is not a hidden stall.
        if unsupported > max(2, int(0.5 * (len(lattice) - 1))):
            continue
        stalls.extend(_quads(rotated, y0, y1, lattice, matrix, scale, lot, matched))
    return stalls


def _peak_groups(peaks: list[int], max_gap: float) -> list[list[int]]:
    groups: list[list[int]] = [[peaks[0]]]
    for peak in peaks[1:]:
        if peak - groups[-1][-1] > max_gap:
            groups.append([peak])
        else:
            groups[-1].append(peak)
    return groups


def _lattice(smooth: np.ndarray, peaks: list[int], pitch: float) -> list[float]:
    best: list[float] = []
    best_score = -1e18
    first, last = float(peaks[0]), float(peaks[-1])
    for anchor in peaks:
        start = float(anchor)
        while start - pitch >= first - 0.45 * pitch:
            start -= pitch
        xs: list[float] = []
        position = start
        limit = last + 0.45 * pitch
        while position <= limit + 1e-6:
            xs.append(position)
            position += pitch
        if len(xs) < MIN_DIVIDERS:
            continue
        score = 0.0
        for item in xs:
            index = int(round(item))
            if 0 <= index < len(smooth):
                score += float(smooth[index])
            mid = int(round(item + pitch / 2.0))
            if 0 <= mid < len(smooth):
                score -= 0.45 * float(smooth[mid])
        if score > best_score:
            best_score = score
            best = xs
    return best


def _edge_fit(xs: list[float], ys: list[float], scale: _Scale) -> tuple[float, float] | None:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    median = float(np.median(y))
    keep = np.abs(y - median) <= scale.px(4)
    if int(keep.sum()) >= 3:
        x, y = x[keep], y[keep]
    if float(np.std(y)) <= 1.25:
        return 0.0, float(np.median(y))
    slope, intercept = [float(value) for value in np.polyfit(x, y, 1)]
    residual = float(np.std(y - (slope * x + intercept)))
    # A real chevron end-line is straight. A noisy fit is flattened so one
    # car cannot drag the whole row into the aisle.
    if abs(slope) > 1.6 or residual > 1.8:
        return 0.0, float(np.median(y))
    return slope, intercept


def _quads(
    rotated, y0, y1, lattice, matrix, scale: _Scale, lot: Polygon, support: int
) -> list[Stall]:
    tops_x: list[float] = []
    tops_y: list[float] = []
    bots_y: list[float] = []
    for position in lattice:
        column = int(round(position))
        if not 0 <= column < rotated.shape[1]:
            continue
        stripe = rotated[y0:y1, max(0, column - 1) : column + 2]
        if stripe.size == 0:
            continue
        on = np.where(stripe.max(axis=1) > 0)[0]
        if on.size < scale.px(8):
            continue
        tops_x.append(position)
        tops_y.append(float(y0 + int(on[0])))
        bots_y.append(float(y0 + int(on[-1])))
    top = _edge_fit(tops_x, tops_y, scale)
    bottom = _edge_fit(tops_x, bots_y, scale)
    if top is None or bottom is None:
        return []
    sample = lattice[len(lattice) // 2]
    depth_px = (bottom[0] * sample + bottom[1]) - (top[0] * sample + top[1])
    depth_ft = scale.to_ft(depth_px)
    if not MIN_DEPTH_FT <= depth_ft <= MAX_DEPTH_FT:
        return []
    pitch_ft = scale.to_ft(lattice[1] - lattice[0])
    if not MIN_WIDTH_FT - 0.4 <= pitch_ft <= MAX_WIDTH_FT + 0.4:
        return []
    corners_top: list[tuple[float, float]] = []
    corners_bot: list[tuple[float, float]] = []
    for position in lattice:
        north = top[0] * position + top[1]
        south = bottom[0] * position + bottom[1]
        if south - north < scale.px(MIN_DEPTH_FT - 1):
            return []
        corners_top.append(_invert(matrix, position, north))
        corners_bot.append(_invert(matrix, position, south))
    stalls: list[Stall] = []
    for index in range(len(lattice) - 1):
        corners = (
            corners_top[index],
            corners_top[index + 1],
            corners_bot[index + 1],
            corners_bot[index],
        )
        polygon = Polygon(corners)
        if not polygon.is_valid or polygon.area <= 0:
            continue
        inside = polygon.intersection(lot)
        inside_area = inside.area if not inside.is_empty else 0.0
        if inside_area < 0.6 * polygon.area:
            continue
        area_ft = polygon.area * scale.gsd_ft * scale.gsd_ft
        if area_ft < MIN_AREA_SQFT:
            continue
        # `support` is stored by biasing length only as a sort key via area in NMS.
        stalls.append(
            Stall(
                polygon=polygon,
                width_ft=round(pitch_ft, 2),
                length_ft=round(depth_ft, 2),
                corners_px=corners,
                support=support,
                row_id=id(lattice),
            )
        )
    return stalls


def _nms(stalls: list[Stall]) -> list[Stall]:
    grouped: dict[int, list[Stall]] = {}
    for stall in stalls:
        grouped.setdefault(stall.row_id, []).append(stall)
    rows = sorted(grouped.values(), key=lambda row: (row[0].support, len(row)), reverse=True)
    kept: list[Stall] = []
    for row in rows:
        for stall in row:
            if _overlaps_kept(stall, kept):
                continue
            kept.append(stall)
    return kept


def _overlaps_kept(stall: Stall, kept: list[Stall]) -> bool:
    for other in kept:
        inter = stall.polygon.intersection(other.polygon).area
        if inter <= 0:
            continue
        if _iou(stall.polygon, other.polygon) > 0.15:
            return True
        if inter > 0.25 * min(stall.polygon.area, other.polygon.area):
            return True
    return False


def _iou(left: Polygon, right: Polygon) -> float:
    inter = left.intersection(right).area
    if inter <= 0:
        return 0.0
    union = left.area + right.area - inter
    return inter / union if union else 0.0
