"""Offline tests for the learned detector. Training imports torch when it is installed."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from osm_parking_spaces.ml.dataset import Candidate, select_lots, split_for
from osm_parking_spaces.ml.holdout import HOLDOUT_WAY_IDS, holdout_region, is_held_out
from osm_parking_spaces.ml.postprocess import stalls_from_mask
from osm_parking_spaces.ml.quality import passes_quality
from osm_parking_spaces.ml.rasterize import rasterize_stall_mask


def test_holdout_ids_and_buffer_are_excluded():
    lot = Polygon(
        [(-74.575, 40.383), (-74.570, 40.383), (-74.570, 40.386), (-74.575, 40.386)]
    )
    region = holdout_region([lot], meters=150)
    far = box(-75.30, 39.40, -75.20, 39.50)
    assert 1329737820 in HOLDOUT_WAY_IDS
    assert is_held_out(1329737820, far, region)
    assert is_held_out(1329737821, far, region)
    assert is_held_out(99, lot, region)
    assert not is_held_out(99, far, region)
    ox, oy = -75.28, 39.42
    far_stalls = _row_of_stalls(origin=(ox, oy))
    far_parking = box(ox - 0.00005, oy - 0.00005, ox + 0.0011, oy + 0.0004)
    kept = select_lots(
        [
            Candidate(1329737820, far_parking, far_stalls),
            Candidate(99, lot, _row_of_stalls()),
            Candidate(50, far_parking, far_stalls),
        ],
        region,
        min_stalls=8,
        min_coverage=0.05,
        max_coverage=0.95,
    )
    assert [item.way_id for item in kept] == [50]


def test_split_never_accepts_a_holdout_id():
    assert split_for(11) == "train"
    assert split_for(10) == "val"
    with pytest.raises(Exception):
        split_for(1329737820)


def test_quality_filter_rejects_sparse_and_solid_lots():
    parking = box(0, 0, 100, 80)
    good = [box(2 + i * 9, 2, 10 + i * 9, 20) for i in range(8)]
    assert passes_quality(parking, good)
    assert not passes_quality(parking, good[:3])
    assert not passes_quality(parking, [box(1, 1, 98, 78)])


def test_raster_mask_separates_adjacent_stalls():
    left = box(2, 2, 12, 22)
    right = box(12, 2, 22, 22)
    mask = rasterize_stall_mask([left, right], 30, 40)
    count, _labels = cv2.connectedComponents(mask)
    assert count - 1 == 2


def test_mask_postprocess_builds_a_row():
    mask = np.zeros((40, 90), np.uint8)
    for x0 in (4, 16, 28, 40):
        mask[6:26, x0 : x0 + 9] = 1
    stalls = stalls_from_mask(mask, box(0, 0, 90, 40), gsd_ft=1.0)
    assert len(stalls) == 4
    ordered = sorted(stalls, key=lambda stall: stall.polygon.centroid.x)
    for left, right in zip(ordered, ordered[1:]):
        shared = {id(point) for point in left.corners_px}
        shared &= {id(point) for point in right.corners_px}
        assert len(shared) == 2


def test_cpu_training_steps_resume(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from osm_parking_spaces.ml.train import train

    image = np.zeros((64, 64, 3), np.uint8)
    image[:, :] = (40, 40, 40)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:28, 8:18] = 1
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    np.savez_compressed(tile_dir / "10.npz", image=image, mask=mask)
    manifest = {
        "tiles_detail": [{"way_id": 10, "split": "train", "stalls": 1, "file": "tiles/10.npz"}]
    }
    (tmp_path / "manifest.json").write_text(__import__("json").dumps(manifest))
    out = tmp_path / "run"
    first = train(tmp_path, out, epochs=1, chip=32, batch_size=1, device_name="cpu", max_steps=2)
    assert first["epoch"] == 0
    assert (out / "checkpoint.pt").exists()
    assert (out / "metrics.jsonl").exists()
    second = train(tmp_path, out, epochs=1, chip=32, batch_size=1, device_name="cpu", max_steps=1)
    assert second["epoch"] == 1
    assert torch.cuda.is_available() or True


def _row_of_stalls(origin=(0.0, 0.0)):
    ox, oy = origin
    return [box(ox + i * 0.00012, oy, ox + i * 0.00012 + 0.00009, oy + 0.0002) for i in range(8)]
