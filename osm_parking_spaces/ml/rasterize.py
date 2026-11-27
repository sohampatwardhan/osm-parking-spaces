"""Turn stall polygons into a mask with a gap between neighbours."""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon


def rasterize_stall_mask(stalls_px: list[Polygon], height: int, width: int) -> np.ndarray:
    """Binary mask. A one-pixel outline is cleared so shared edges do not merge."""
    mask = np.zeros((height, width), np.uint8)
    rings = []
    for stall in stalls_px:
        if stall.is_empty:
            continue
        ring = np.round(stall.exterior.coords).astype(np.int32)
        if len(ring) < 3:
            continue
        cv2.fillPoly(mask, [ring], 1)
        rings.append(ring)
    for ring in rings:
        cv2.polylines(mask, [ring], True, 0, 1)
    return mask
