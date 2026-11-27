"""Training-time chips: arbitrary-angle rotation, flips and mild photometric jitter.

Pure NumPy and OpenCV, so it is tested in CI without PyTorch.
"""

from __future__ import annotations

import cv2
import numpy as np


def sample_chip(
    image: np.ndarray,
    mask: np.ndarray,
    chip: int,
    rng: np.random.Generator,
    rotate: bool = True,
    focus: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """One `chip` x `chip` crop, rotated by a random angle and randomly flipped.

    The centre is a random stall pixel with probability `focus`, so most chips contain stalls,
    otherwise a random point of the tile. The warp reflects at the tile edge instead of padding
    with black, so rotation does not paint a fake border. Returns float32 BGR in 0..1 as
    (3, chip, chip) and a float32 mask as (1, chip, chip).
    """
    height, width = mask.shape[:2]
    if focus > 0 and rng.random() < focus and mask.any():
        ys, xs = np.nonzero(mask)
        pick = int(rng.integers(0, len(xs)))
        cx, cy = float(xs[pick]), float(ys[pick])
    else:
        cx, cy = float(rng.uniform(0, width)), float(rng.uniform(0, height))
    angle = float(rng.uniform(0.0, 360.0)) if rotate else 0.0
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    matrix[0, 2] += chip / 2.0 - cx
    matrix[1, 2] += chip / 2.0 - cy
    warped = cv2.warpAffine(
        image, matrix, (chip, chip), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    label = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (chip, chip),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if rng.random() < 0.5:
        warped, label = warped[:, ::-1], label[:, ::-1]
    if rng.random() < 0.5:
        warped, label = warped[::-1], label[::-1]
    out = warped.astype(np.float32) / 255.0
    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.06, 0.06))
    out = np.clip((out - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    tensor = np.ascontiguousarray(np.transpose(out, (2, 0, 1)))
    return tensor, np.ascontiguousarray(label[None, ...].astype(np.float32))


def lot_weights(angled_fractions: list[float], threshold: float = 0.25, boost: float = 3.0):
    """Sampling weights that draw lots with angled bays `boost` times as often."""
    weights = np.array([boost if f >= threshold else 1.0 for f in angled_fractions], np.float64)
    return weights / weights.sum()
