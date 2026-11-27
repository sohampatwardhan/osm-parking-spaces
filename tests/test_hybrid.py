"""Offline tests for v4: fusion, augmentation, oversampling, dataset helpers, the offset fix.

Everything uses small synthetic arrays. Nothing here touches the network or trains a large model.
"""

import json

import cv2
import numpy as np
import pytest
from shapely.geometry import Polygon, box

import osm_parking_spaces.ml.dataset as ds
from osm_parking_spaces.detect import Stall
from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.evaluate import score_polygons
from osm_parking_spaces.ml.augment import lot_weights, sample_chip
from osm_parking_spaces.ml.fusion import (
    FusionConfig,
    canonical,
    fuse,
    mean_prob,
)

# ---------------------------------------------------------------- fusion fixtures


def _stall(x0, y0, width=9.0, depth=18.0, row=1) -> Stall:
    corners = ((x0, y0), (x0 + width, y0), (x0 + width, y0 + depth), (x0, y0 + depth))
    return Stall(Polygon(corners), width, depth, corners, 1, row)


def _row_stalls(count, x0=20.0, y0=20.0, row=1, depth=18.0):
    """Neighbours reuse the same corner tuples, as the classical detector does."""
    xs = [x0 + 9.0 * i for i in range(count + 1)]
    top = [(x, y0) for x in xs]
    bottom = [(x, y0 + depth) for x in xs]
    return [
        Stall(
            Polygon([top[i], top[i + 1], bottom[i + 1], bottom[i]]),
            9.0,
            depth,
            (top[i], top[i + 1], bottom[i + 1], bottom[i]),
            1,
            row,
        )
        for i in range(count)
    ]


def _probs(shape=(100, 200), rects=()):
    probs = np.full(shape, 0.02, np.float32)
    for x0, y0, x1, y1 in rects:
        probs[y0:y1, x0:x1] = 0.95
    return probs


LOT = box(0, 0, 200, 100)
NO_SEED = FusionConfig(seed=None)


def test_a_stall_the_model_rejects_is_dropped_and_a_good_one_kept():
    stalls = _row_stalls(3) + [_stall(120, 60, row=2)]
    probs = _probs(rects=[(20, 20, 47, 38)])
    kept = fuse(stalls, probs, LOT, 1.0, NO_SEED)
    centres = sorted(round(s.polygon.centroid.x) for s in kept)
    assert 124 not in centres
    assert len(kept) == 3


def test_a_borderline_stall_survives_only_with_a_confident_row():
    stalls = _row_stalls(4)
    probs = _probs(rects=[(20, 20, 56, 38)])
    probs[20:38, 56:65] = 0.3  # the fourth stall is borderline
    kept = fuse(stalls, probs, LOT, 1.0, NO_SEED)
    assert len(kept) == 4
    weak = fuse(_row_stalls(1), _probs(rects=[(20, 20, 29, 38)]) * 0 + 0.3, LOT, 1.0, NO_SEED)
    assert weak == []


def test_the_row_is_extended_at_its_own_pitch_with_shared_corners():
    stalls = _row_stalls(3)
    probs = _probs(rects=[(20, 20, 20 + 9 * 6, 38)])
    kept = fuse(stalls, probs, LOT, 1.0, NO_SEED)
    assert len(kept) == 6
    ordered = sorted(kept, key=lambda s: s.polygon.centroid.x)
    for left, right in zip(ordered, ordered[1:]):
        shared = set(left.corners_px) & set(right.corners_px)
        assert len(shared) == 2
    # 6 stalls in a row: 7 top and 7 bottom corners
    assert len({point for s in kept for point in s.corners_px}) == 14


def test_an_interior_gap_is_filled():
    left = _row_stalls(2)
    right = _row_stalls(2, x0=20 + 9 * 3)
    stalls = left + right
    probs = _probs(rects=[(20, 20, 20 + 9 * 5, 38)])
    kept = fuse(stalls, probs, LOT, 1.0, NO_SEED)
    assert len(kept) == 5


def test_fill_stops_where_the_model_sees_no_stall_or_the_lot_ends():
    stalls = _row_stalls(3)
    probs = _probs(rects=[(20, 20, 47, 38)])
    assert len(fuse(stalls, probs, LOT, 1.0, NO_SEED)) == 3
    wide = _probs(rects=[(20, 20, 120, 38)])
    small_lot = box(0, 0, 60, 100)
    kept = fuse(stalls, wide, small_lot, 1.0, NO_SEED)
    assert max(s.polygon.centroid.x for s in kept) < 60


def test_a_row_that_runs_into_the_aisle_is_pulled_back_as_one_unit():
    # Stalls drawn 26 px deep, but the model only sees the first 18.
    stalls = _row_stalls(4, depth=26.0)
    probs = _probs(rects=[(20, 20, 56, 38)])
    kept = fuse(stalls, probs, LOT, 1.0, NO_SEED)
    assert len(kept) == 4
    for stall in kept:
        assert 16.0 <= stall.length_ft <= 20.0
    ordered = sorted(kept, key=lambda s: s.polygon.centroid.x)
    for left, right in zip(ordered, ordered[1:]):
        assert len(set(left.corners_px) & set(right.corners_px)) == 2


def test_a_row_that_stops_short_is_extended_to_the_painted_depth():
    stalls = _row_stalls(4, depth=12.0)
    probs = _probs(rects=[(20, 20, 56, 38)])
    kept = fuse(stalls, probs, LOT, 1.0, FusionConfig(seed=None, max_fill=0))
    assert all(16.0 <= s.length_ft <= 20.0 for s in kept)


def test_trim_off_leaves_the_classical_boxes_alone():
    stalls = _row_stalls(4, depth=26.0)
    probs = _probs(rects=[(20, 20, 56, 38)])
    kept = fuse(stalls, probs, LOT, 1.0, FusionConfig(seed=None, trim=False, max_fill=0))
    assert all(s.length_ft == 26.0 for s in kept)


def test_model_seeded_rows_find_angled_bays_the_classical_pass_missed():
    probs = np.full((120, 220), 0.02, np.float32)
    mask = np.zeros(probs.shape, np.uint8)
    # Five parallelogram bays: 9 px along the row, sheared 10 px over an 18 px depth.
    for index in range(5):
        x = 30 + 9 * index
        ring = np.array([[x + 1, 30], [x + 8, 30], [x + 18, 48], [x + 11, 48]], np.int32)
        cv2.fillPoly(mask, [ring], 1)
    probs[mask > 0] = 0.95
    seeded = fuse([], probs, box(0, 0, 220, 120), 1.0, FusionConfig(seed=0.6))
    assert len(seeded) >= 3
    assert all(6.0 <= s.width_ft <= 16.0 for s in seeded)
    assert all(11.0 <= s.length_ft <= 26.0 for s in seeded)
    assert fuse([], probs, box(0, 0, 220, 120), 1.0, NO_SEED) == []


def test_seeds_do_not_overlap_classical_stalls():
    stalls = _row_stalls(3)
    probs = _probs(rects=[(20, 20, 47, 38)])
    seeded = fuse(stalls, probs, LOT, 1.0, FusionConfig(seed=0.6))
    assert len(seeded) == 3


def test_canonical_order_puts_the_depth_edge_second():
    corners = np.array([[0, 0], [18, 0], [18, 9], [0, 9]], float)
    out = canonical(corners)
    assert np.linalg.norm(out[3] - out[0]) > np.linalg.norm(out[1] - out[0])
    assert {tuple(p) for p in out} == {tuple(p) for p in corners}


def test_mean_prob_uses_the_polygon_interior():
    probs = _probs(rects=[(20, 20, 29, 38)])
    assert mean_prob(probs, box(20, 20, 29, 38)) > 0.9
    assert mean_prob(probs, box(100, 20, 109, 38)) < 0.1


def test_fusion_config_round_trips_and_defaults(tmp_path):
    weights = tmp_path / "best.pt"
    assert FusionConfig.load_for(weights, pixel=0.4).pixel == 0.4
    FusionConfig(keep=0.5, seed=None, max_fill=2).save(tmp_path / "fusion.json")
    loaded = FusionConfig.load_for(weights)
    assert (loaded.keep, loaded.seed, loaded.max_fill) == (0.5, None, 2)


# ---------------------------------------------------------------- augmentation


def test_chip_shape_range_and_a_stall_at_the_centre():
    rng = np.random.default_rng(0)
    image = np.full((200, 200, 3), 90, np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    mask[90:110, 90:110] = 1
    for _ in range(10):
        tensor, label = sample_chip(image, mask, 64, rng, focus=1.0)
        assert tensor.shape == (3, 64, 64) and label.shape == (1, 64, 64)
        assert tensor.dtype == np.float32 and 0.0 <= tensor.min() and tensor.max() <= 1.0
        assert label[0, 31:33, 31:33].any()
        assert set(np.unique(label)) <= {0.0, 1.0}


def test_rotation_turns_a_straight_row_to_many_angles():
    image = np.full((240, 240, 3), 60, np.uint8)
    mask = np.zeros((240, 240), np.uint8)
    mask[110:130, 40:200] = 1  # horizontal bar
    angles = set()
    for seed in range(40):
        _t, label = sample_chip(image, mask, 128, np.random.default_rng(seed), focus=1.0)
        ys, xs = np.nonzero(label[0])
        centred = np.stack([xs - xs.mean(), ys - ys.mean()])
        eig = np.linalg.eigh(np.cov(centred))[1][:, -1]
        angles.add(int(np.degrees(np.arctan2(eig[1], eig[0])) % 180 // 15))
    assert len(angles) >= 5


def test_rotation_can_be_disabled_and_is_seeded():
    image = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 40:60] = 1
    a = sample_chip(image, mask, 32, np.random.default_rng(5), focus=1.0)
    b = sample_chip(image, mask, 32, np.random.default_rng(5), focus=1.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    flat = sample_chip(image, mask, 32, np.random.default_rng(5), rotate=False, focus=1.0)
    assert flat[1].sum() > 0


def test_angled_lots_are_oversampled():
    weights = lot_weights([0.0, 0.0, 0.5, 0.9], threshold=0.25, boost=3.0)
    assert weights[2] == pytest.approx(3 * weights[0])
    assert weights.sum() == pytest.approx(1.0)
    counts = np.bincount(np.random.default_rng(0).choice(4, 6000, p=weights), minlength=4)
    assert counts[2] > 2 * counts[0]


# ---------------------------------------------------------------- dataset helpers


def test_angled_fraction_ignores_rotation_but_sees_shear():
    rect = Polygon([(0, 0), (9, 0), (9, 18), (0, 18)])
    turned = Polygon([(0, 0), (6.4, 6.4), (-6.4, 19.1), (-12.7, 12.7)])
    sheared = Polygon([(0, 0), (9, 0), (19, 18), (10, 18)])
    assert ds.angled_fraction([rect, turned]) == 0.0
    assert ds.angled_fraction([sheared, rect]) == 0.5
    assert ds.angled_fraction([]) == 0.0


def test_blank_tiles_from_outside_the_imagery_are_detected():
    assert ds.is_blank(np.full((50, 50, 3), 255, np.uint8))
    assert ds.is_blank(np.zeros((50, 50, 3), np.uint8))
    noisy = np.random.default_rng(0).integers(30, 200, (50, 50, 3), dtype=np.uint8)
    assert not ds.is_blank(noisy)


def test_stalls_are_paired_to_the_lot_that_contains_them():
    def way(way_id, tag, x0, y0, x1, y1):
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        return {
            "type": "way",
            "id": way_id,
            "tags": {"amenity": tag},
            "geometry": [{"lon": x, "lat": y} for x, y in ring],
        }

    elements = [
        way(1, "parking", 0, 0, 1, 1),
        way(2, "parking", 5, 5, 6, 6),
        way(3, "parking_space", 0.1, 0.1, 0.2, 0.2),
        way(4, "parking_space", 0.5, 0.5, 0.6, 0.6),
        way(5, "parking_space", 9, 9, 9.1, 9.1),
    ]
    candidates = ds._pair_stalls(elements)
    assert [c.way_id for c in candidates] == [1]
    assert len(candidates[0].stalls) == 2


def test_shuffled_selection_is_seeded_and_still_drops_the_holdout():
    from osm_parking_spaces.ml.holdout import holdout_region

    near = Polygon([(-74.575, 40.383), (-74.570, 40.383), (-74.570, 40.386), (-74.575, 40.386)])
    region = holdout_region([near], meters=150)
    lots = []
    for index in range(30):
        ox = -75.3 + 0.01 * index
        parking = box(ox, 39.4, ox + 0.0011, 39.4004)
        stalls = [
            box(ox + i * 0.00012, 39.4001, ox + i * 0.00012 + 0.00009, 39.4003) for i in range(8)
        ]
        lots.append(ds.Candidate(100 + index, parking, stalls))
    lots.append(ds.Candidate(1329737820, lots[0].parking, lots[0].stalls))
    quality = {"min_coverage": 0.05, "max_coverage": 0.95}
    a = ds.select_shuffled(lots, region, wanted=10, seed=3, batch=5, **quality)
    b = ds.select_shuffled(lots, region, wanted=10, seed=3, batch=5, **quality)
    c = ds.select_shuffled(lots, region, wanted=10, seed=4, batch=5, **quality)
    assert [x.way_id for x in a] == [x.way_id for x in b]
    assert [x.way_id for x in a] != [x.way_id for x in c]
    assert 1329737820 not in [x.way_id for x in a + c]
    assert len(a) >= 10


def test_overpass_is_split_cached_retried_and_merged(tmp_path, monkeypatch):
    calls = []

    def fake(bbox, url):
        calls.append(bbox)
        if len(calls) == 1:
            raise ParkingError("Overpass returned HTTP 502")
        return [{"type": "way", "id": 7, "tags": {}}, {"type": "way", "id": len(calls)}]

    monkeypatch.setattr(ds, "_query_overpass", fake)
    sleeps = []
    out = ds._query_chunked((0, 0, 2, 3), "u", tmp_path, grid=(2, 3), sleep=sleeps.append)
    assert len(calls) == 7  # six boxes, one retried
    assert sleeps and max(sleeps) >= 30
    assert len(out) == 6  # element ids 2..7 are unique, and id 7 repeats in every box
    again = ds._query_chunked((0, 0, 2, 3), "u", tmp_path, grid=(2, 3), sleep=lambda s: None)
    assert len(calls) == 7 and len(again) == 6  # second run came from the cache


def test_chunk_boxes_tile_the_bbox():
    boxes = list(ds._chunk_boxes((38.9, -75.6, 41.4, -73.9), (4, 3)))
    assert len(boxes) == 12
    assert min(b[0] for b in boxes) == 38.9 and max(b[2] for b in boxes) == pytest.approx(41.4)
    assert min(b[1] for b in boxes) == -75.6 and max(b[3] for b in boxes) == pytest.approx(-73.9)


def test_tile_meta_round_trips(tmp_path):
    meta = {"lot": [[0, 0], [9, 0], [9, 9], [0, 0]], "stalls": [[[1, 1], [2, 1], [2, 2], [1, 1]]]}
    meta["angled_fraction"] = 0.25
    path = tmp_path / "t.npz"
    image = np.zeros((4, 4, 3), np.uint8)
    np.savez_compressed(
        path, image=image, mask=np.zeros((4, 4), np.uint8), meta=np.array(json.dumps(meta))
    )
    loaded = ds.load_meta(path)
    assert loaded["angled_fraction"] == 0.25 and len(loaded["stalls"]) == 1
    old = tmp_path / "old.npz"
    np.savez_compressed(old, image=image, mask=np.zeros((4, 4), np.uint8))
    assert ds.load_meta(old) is None


# ---------------------------------------------------------------- the offset fix


def test_pixel_offsets_are_in_feet_for_small_pixel_coordinates():
    truth = [box(150, 60, 159, 78)]
    shifted = [box(153, 60, 162, 78)]
    result, _ = score_polygons(shifted, truth, pixel_gsd_ft=1.0)
    assert result.true_positives == 1 and result.mean_offset_ft == pytest.approx(3.0)
    half, _ = score_polygons(shifted, truth, pixel_gsd_ft=0.5)
    assert half.mean_offset_ft == pytest.approx(1.5)


def test_wgs84_offsets_stay_geodesic():
    dlat = 0.00002  # about 7.3 ft
    truth = [box(-74.5, 40.0, -74.4999, 40.0001)]
    shifted = [box(-74.5, 40.0 + dlat, -74.4999, 40.0001 + dlat)]
    result, _ = score_polygons(shifted, truth)
    assert result.true_positives == 1
    assert result.mean_offset_ft == pytest.approx(7.3, abs=0.2)


# ---------------------------------------------------------------- model plumbing


def test_swin_unet_runs_forward_on_cpu_with_any_size():
    torch = pytest.importorskip("torch")
    from osm_parking_spaces.ml.model import build_model

    model = build_model("swin-unet").eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 70, 90))
    assert out.shape == (1, 1, 70, 90)


def test_encoder_state_keeps_only_the_swin_features(tmp_path):
    torch = pytest.importorskip("torch")
    from osm_parking_spaces.ml.encoder import load_encoder_state

    path = tmp_path / "fake.pth"
    torch.save(
        {
            "backbone.backbone.features.0.0.weight": torch.zeros(2),
            "backbone.fpn.layer.weight": torch.zeros(1),
            "heads.0.weight": torch.zeros(1),
        },
        path,
    )
    assert list(load_encoder_state(path)) == ["0.0.weight"]
    torch.save({"heads.0.weight": torch.zeros(1)}, path)
    with pytest.raises(ParkingError):
        load_encoder_state(path)


def test_fetch_encoder_skips_a_download_when_the_hash_matches(tmp_path, monkeypatch):
    from osm_parking_spaces.ml import encoder

    target = tmp_path / encoder.ENCODER_NAME
    target.write_bytes(b"abc")
    digest = encoder.sha256_of(target)
    with pytest.raises(ParkingError):
        encoder.fetch_encoder(tmp_path, url="http://127.0.0.1:9/x", sha256="0" * 64)
    assert target.exists()
    target.write_bytes(b"abc")
    monkeypatch.setattr(encoder.requests, "get", lambda *a, **k: pytest.fail("downloaded"))
    assert encoder.fetch_encoder(tmp_path, sha256=digest) == target


def test_training_keeps_the_best_checkpoint_and_the_hybrid_runs(tmp_path):
    pytest.importorskip("torch")
    from osm_parking_spaces.ml.infer import detect_stalls_hybrid
    from osm_parking_spaces.ml.train import train

    image = np.full((96, 96, 3), 50, np.uint8)
    mask = np.zeros((96, 96), np.uint8)
    mask[20:38, 10:60] = 1
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    for name in ("10", "11"):
        np.savez_compressed(tiles / f"{name}.npz", image=image, mask=mask)
    detail = [
        {
            "way_id": 11,
            "split": "train",
            "stalls": 1,
            "file": "tiles/11.npz",
            "angled_fraction": 0.5,
        },
        {"way_id": 10, "split": "val", "stalls": 1, "file": "tiles/10.npz"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"tiles_detail": detail}))
    out = tmp_path / "run"
    record = train(
        tmp_path, out, epochs=1, chip=32, batch_size=2, device_name="cpu", max_steps=2, arch="unet"
    )
    assert (out / "best.pt").exists() and record["val_dice"] is not None
    with pytest.raises(ParkingError):
        train(tmp_path, out, epochs=1, chip=32, device_name="cpu", max_steps=1, arch="swin-unet")
    stalls = detect_stalls_hybrid(image, box(0, 0, 96, 96), out / "best.pt", device_name="cpu")
    assert isinstance(stalls, list)
