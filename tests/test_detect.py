import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box

from osm_parking_spaces.detect import detect_stalls

FIXTURES = Path(__file__).parent / "fixtures"


def _row(image, xs, y0, y1):
    for x in xs:
        cv2.line(image, (x, y0), (x, y1), (235, 235, 235), 1)


def test_two_painted_rows_become_stalls():
    image = np.full((140, 220, 3), 55, np.uint8)
    xs = [40, 49, 58, 67, 76, 85]
    _row(image, xs, 25, 43)
    _row(image, xs, 80, 98)
    lot = box(5, 5, 215, 135)
    stalls = detect_stalls(image, lot, gsd_ft=1.0)
    assert len(stalls) == 10
    widths = sorted(round(stall.width_ft) for stall in stalls)
    assert widths == [9] * 10
    assert all(16 <= stall.length_ft <= 22 for stall in stalls)


def test_cropped_natural2020_tile_finds_the_painted_row():
    """Offline crop of NJOGIS Natural2020 (way 1329737820) plus its pixel polygon.

    The count is whatever this detector returns on that frozen tile. It is a
    regression check, not a claim that every painted bay was found.
    """
    image = cv2.imread(str(FIXTURES / "stall_row.png"))
    assert image is not None
    ring = json.loads((FIXTURES / "stall_row_polygon.json").read_text())["polygon"]
    stalls = detect_stalls(image, Polygon(ring), gsd_ft=1.0)
    assert len(stalls) >= 8
    assert all(7.0 <= stall.width_ft <= 14.0 for stall in stalls)
    assert all(12.0 <= stall.length_ft <= 22.0 for stall in stalls)


def test_a_missing_divider_is_filled_at_the_row_pitch():
    image = np.full((80, 180, 3), 55, np.uint8)
    for x in (30, 39, 48, 66, 75, 84):
        cv2.line(image, (x, 20), (x, 38), (235, 235, 235), 1)
    stalls = detect_stalls(image, box(2, 2, 170, 75), gsd_ft=1.0)
    assert len(stalls) == 6


def test_hatched_lines_are_not_stalls():
    image = np.full((80, 140, 3), 55, np.uint8)
    for x in range(20, 110, 3):
        cv2.line(image, (x, 24), (x, 42), (230, 230, 230), 1)
    assert detect_stalls(image, box(2, 2, 130, 75), gsd_ft=1.0) == []


def test_neighbors_share_divider_corners():
    image = np.full((80, 180, 3), 55, np.uint8)
    for x in (30, 39, 48, 57, 66, 75):
        cv2.line(image, (x, 20), (x, 38), (235, 235, 235), 1)
    stalls = detect_stalls(image, box(2, 2, 170, 75), gsd_ft=1.0)
    row = sorted(stalls, key=lambda stall: stall.polygon.centroid.x)
    assert len(row) == 5
    for left, right in zip(row, row[1:]):
        shared_ids = {id(point) for point in left.corners_px}
        shared_ids &= {id(point) for point in right.corners_px}
        shared = shared_ids
        assert len(shared) == 2


def test_empty_asphalt_yields_no_stalls():
    image = np.full((80, 80, 3), 60, np.uint8)
    assert detect_stalls(image, box(2, 2, 78, 78), gsd_ft=1.0) == []
