"""Build a Natural2020 chip set from OSM parking-space polygons.

Labels are ODbL. Imagery is NJOGIS Natural2020 only. The ShopRite evaluation
lots are removed before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.imagery import fetch_mosaic, plan_grid, polygon_to_pixels
from osm_parking_spaces.ml.holdout import (
    HOLDOUT_BUFFER_M,
    HOLDOUT_WAY_IDS,
    holdout_region,
    is_held_out,
)
from osm_parking_spaces.ml.quality import passes_quality
from osm_parking_spaces.ml.rasterize import rasterize_stall_mask
from osm_parking_spaces.osm_api import USER_AGENT, fetch_way

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# South, west, north, east. Matches the Natural2020 footprint over New Jersey.
NJ_BBOX = (38.92, -75.58, 41.36, -73.89)
# A lot whose angled-bay fraction reaches this is oversampled in training.
ANGLED_LOT = 0.25


@dataclass
class Candidate:
    way_id: int
    parking: Polygon
    stalls: list[Polygon]


def split_for(way_id: int) -> str:
    """Lot-level split. Every fifth way id is validation. Holdout ids never arrive here."""
    if way_id in HOLDOUT_WAY_IDS:
        raise ParkingError(f"way {way_id} is held out and cannot be split into train or val")
    return "val" if way_id % 5 == 0 else "train"


def select_lots(
    candidates: list[Candidate],
    holdout,
    min_stalls: int = 8,
    min_coverage: float = 0.12,
    max_coverage: float = 0.90,
) -> list[Candidate]:
    kept = []
    for candidate in candidates:
        if is_held_out(candidate.way_id, candidate.parking, holdout):
            continue
        if not passes_quality(
            candidate.parking,
            candidate.stalls,
            min_stalls=min_stalls,
            min_coverage=min_coverage,
            max_coverage=max_coverage,
        ):
            continue
        kept.append(candidate)
    return kept


def select_shuffled(
    candidates: list[Candidate],
    holdout,
    wanted: int,
    seed: int = 0,
    batch: int = 100,
    **quality,
) -> list[Candidate]:
    """Shuffle with a fixed seed, then filter in batches until `wanted` lots pass.

    Overpass returns ways in id order, so taking the first lots would bias the set to
    old ways in one area. A seeded shuffle spreads the lots over the whole bounding box.
    """
    pool = list(candidates)
    random.Random(seed).shuffle(pool)
    kept: list[Candidate] = []
    for start in range(0, len(pool), batch):
        kept.extend(select_lots(pool[start : start + batch], holdout, **quality))
        if len(kept) >= wanted:
            break
    return kept


def is_blank(image: np.ndarray, white_fraction: float = 0.3) -> bool:
    """True for a failed fetch or a window outside the imagery's coverage (flat white or black)."""
    if image.size == 0 or int(image.max()) == 0 or float(image.std()) < 8.0:
        return True
    return float((image.min(axis=2) >= 250).mean()) > white_fraction


def angled_fraction(stalls_px: list[Polygon], tolerance_deg: float = 15.0) -> float:
    """Share of stalls drawn as parallelograms rather than rectangles.

    A quadrilateral counts as an angled bay when one corner is more than `tolerance_deg`
    away from a right angle. Rotating a rectangle does not change this number, so it
    separates angled bays (herringbone, chevron) from plain rows at any compass bearing.
    """
    if not stalls_px:
        return 0.0
    angled = 0
    counted = 0
    for stall in stalls_px:
        ring = np.asarray(stall.exterior.coords)[:-1]
        if len(ring) != 4:
            continue
        counted += 1
        worst = 0.0
        for index in range(4):
            a = ring[index - 1] - ring[index]
            b = ring[(index + 1) % 4] - ring[index]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 0:
                continue
            angle = float(np.degrees(np.arccos(np.clip(float(a @ b) / denom, -1.0, 1.0))))
            worst = max(worst, abs(angle - 90.0))
        if worst > tolerance_deg:
            angled += 1
    return angled / counted if counted else 0.0


def build_dataset(
    out_dir: Path,
    bbox: tuple[float, float, float, float] = NJ_BBOX,
    max_lots: int = 40,
    overpass_url: str = OVERPASS_URL,
    cache_dir: Path | None = None,
    seed: int = 0,
    grid: tuple[int, int] = (4, 3),
    sleep=time.sleep,
) -> dict:
    """Download lots and write `manifest.json` plus one npz tile per lot.

    Each tile holds the image, the stall mask, and (as JSON) the lot polygon and the OSM stall
    polygons in tile pixels. `cache_dir` keeps raw Overpass answers so a retry does not repeat
    the large queries.
    """
    out_dir = Path(out_dir)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    holdout_polys = [fetch_way(way_id) for way_id in sorted(HOLDOUT_WAY_IDS)]
    region = holdout_region(holdout_polys, HOLDOUT_BUFFER_M)
    raw = _query_chunked(
        bbox, overpass_url, cache_dir or out_dir / "overpass-cache", grid=grid, sleep=sleep
    )
    candidates = _pair_stalls(raw)
    # Blank tiles are dropped after download, so ask for more lots than the cap.
    pool = select_shuffled(candidates, region, wanted=max(1, max_lots) * 3, seed=seed)
    tiles = []
    skipped = {"blank": 0, "too_large": 0, "few_pixels": 0, "fetch_failed": 0}
    for candidate in pool:
        if len(tiles) >= max_lots:
            break
        lot_grid = plan_grid(candidate.parking, gsd_ft=1.0, pad_ft=4.0)
        if lot_grid.width * lot_grid.height > 1_500_000:
            skipped["too_large"] += 1
            continue
        image = _fetch_with_retry(lot_grid, sleep)
        if image is None:
            skipped["fetch_failed"] += 1
            continue
        if is_blank(image):
            skipped["blank"] += 1
            continue
        stall_px = [polygon_to_pixels(stall, lot_grid) for stall in candidate.stalls]
        mask = rasterize_stall_mask(stall_px, lot_grid.height, lot_grid.width)
        if int(mask.sum()) < 30:
            skipped["few_pixels"] += 1
            continue
        lot_px = polygon_to_pixels(candidate.parking, lot_grid)
        angled = angled_fraction(stall_px)
        meta = {
            "lot": [list(point) for point in lot_px.exterior.coords],
            "stalls": [[list(point) for point in stall.exterior.coords] for stall in stall_px],
            "angled_fraction": round(angled, 3),
        }
        path = tile_dir / f"{candidate.way_id}.npz"
        np.savez_compressed(path, image=image, mask=mask, meta=np.array(json.dumps(meta)))
        tiles.append(
            {
                "way_id": candidate.way_id,
                "split": split_for(candidate.way_id),
                "stalls": len(candidate.stalls),
                "angled_fraction": round(angled, 3),
                "file": str(path.relative_to(out_dir)),
                "height": lot_grid.height,
                "width": lot_grid.width,
            }
        )
    splits = _ensure_val(tiles)
    manifest = {
        "imagery": "NJOGIS Natural2020",
        "labels": "OSM amenity=parking_space (ODbL)",
        "holdout_way_ids": sorted(HOLDOUT_WAY_IDS),
        "holdout_buffer_m": HOLDOUT_BUFFER_M,
        "bbox": list(bbox),
        "seed": seed,
        "lots": len(splits),
        "stalls": sum(item["stalls"] for item in splits),
        "tiles": len(splits),
        "train_lots": sum(1 for item in splits if item["split"] == "train"),
        "val_lots": sum(1 for item in splits if item["split"] == "val"),
        "angled_lots": sum(1 for item in splits if item["angled_fraction"] >= ANGLED_LOT),
        "skipped": skipped,
        "split_rule": "way_id % 5 == 0 is val, otherwise train; holdout ids are never written",
        "tiles_detail": splits,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _ensure_val(tiles: list[dict]) -> list[dict]:
    if not tiles or any(item["split"] == "val" for item in tiles):
        return tiles
    if len(tiles) == 1:
        return tiles
    tiles[-1]["split"] = "val"
    return tiles


def _fetch_with_retry(lot_grid, sleep, attempts: int = 5, pause: float = 0.3):
    for attempt in range(attempts):
        try:
            sleep(pause)
            return fetch_mosaic(lot_grid)
        except ParkingError:
            sleep(10.0 * (attempt + 1))
    return None


def _chunk_boxes(bbox, grid: tuple[int, int]):
    south, west, north, east = bbox
    rows, cols = grid
    for i in range(rows):
        for j in range(cols):
            yield (
                south + (north - south) * i / rows,
                west + (east - west) * j / cols,
                south + (north - south) * (i + 1) / rows,
                west + (east - west) * (j + 1) / cols,
            )


def _query_chunked(bbox, url: str, cache_dir: Path, grid=(4, 3), sleep=time.sleep) -> list[dict]:
    """One Overpass query per sub-box, cached on disk, merged by element id."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[str, int], dict] = {}
    for sub in _chunk_boxes(bbox, grid):
        key = hashlib.sha1(repr([round(v, 5) for v in sub]).encode()).hexdigest()[:12]
        cached = cache_dir / f"{key}.json"
        if cached.exists():
            elements = json.loads(cached.read_text())
        else:
            elements = _query_with_retry(sub, url, sleep)
            cached.write_text(json.dumps(elements))
            sleep(5.0)
        for element in elements:
            merged[(str(element.get("type")), int(element.get("id", 0)))] = element
    return list(merged.values())


def _query_with_retry(bbox, url: str, sleep, attempts: int = 8) -> list[dict]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return _query_overpass(bbox, url)
        except ParkingError as exc:
            last = exc
            sleep(min(30.0 * (attempt + 1), 180.0))
    raise ParkingError(f"Overpass kept failing for {bbox}: {last}")


def _query_overpass(bbox, url: str) -> list[dict]:
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:180];
    (
      way["amenity"="parking"]({south},{west},{north},{east});
      way["amenity"="parking_space"]({south},{west},{north},{east});
    );
    out geom;
    """
    try:
        response = requests.post(
            url,
            data={"data": query},
            headers={"User-Agent": USER_AGENT},
            timeout=240,
        )
    except requests.RequestException as exc:
        raise ParkingError(f"Overpass request failed: {exc}") from exc
    if response.status_code != 200:
        raise ParkingError(f"Overpass returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ParkingError("Overpass response was not JSON") from exc
    return list(payload.get("elements", []))


def _pair_stalls(elements: list[dict]) -> list[Candidate]:
    """Attach each stall to the parking polygon that contains its centroid (STRtree)."""
    parking = []
    stalls = []
    for element in elements:
        polygon = _way_polygon(element)
        if polygon is None:
            continue
        tags = element.get("tags") or {}
        way_id = int(element.get("id", 0))
        if tags.get("amenity") == "parking_space":
            stalls.append(polygon)
        elif tags.get("amenity") == "parking":
            parking.append((way_id, polygon))
    if not stalls:
        return []
    tree = STRtree([stall.centroid for stall in stalls])
    candidates = []
    for way_id, lot in parking:
        inside = [stalls[int(i)] for i in tree.query(lot, predicate="contains")]
        if inside:
            candidates.append(Candidate(way_id, lot, inside))
    return candidates


def _way_polygon(element: dict) -> Polygon | None:
    geometry = element.get("geometry") or []
    coords = [(float(point["lon"]), float(point["lat"])) for point in geometry]
    if len(coords) < 4:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return None
    return polygon


def load_tile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    image = data["image"]
    mask = data["mask"]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, mask


def load_meta(path: Path) -> dict | None:
    """Lot polygon, OSM stall polygons (pixels) and angled fraction, if the tile has them."""
    data = np.load(path)
    if "meta" not in data.files:
        return None
    meta = json.loads(str(data["meta"]))
    return {
        "lot": Polygon(meta["lot"]),
        "stalls": [Polygon(ring) for ring in meta["stalls"]],
        "angled_fraction": float(meta.get("angled_fraction", 0.0)),
    }
