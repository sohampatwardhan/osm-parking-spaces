"""Build a Natural2020 chip set from OSM parking-space polygons.

Labels are ODbL. Imagery is NJOGIS Natural2020 only. The ShopRite evaluation
lots are removed before anything is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from shapely.geometry import Polygon

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


def build_dataset(
    out_dir: Path,
    bbox: tuple[float, float, float, float] = NJ_BBOX,
    max_lots: int = 40,
    overpass_url: str = OVERPASS_URL,
) -> dict:
    """Download lots and write `manifest.json` plus one npz tile per lot."""
    out_dir = Path(out_dir)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    holdout_polys = [fetch_way(way_id) for way_id in sorted(HOLDOUT_WAY_IDS)]
    region = holdout_region(holdout_polys, HOLDOUT_BUFFER_M)
    raw = _query_overpass(bbox, overpass_url)
    candidates = _pair_stalls(raw)
    chosen = select_lots(candidates, region)[: max(0, max_lots)]
    if chosen and not any(split_for(item.way_id) == "val" for item in chosen):
        # A tiny sample can land entirely in train. Move the last lot so val exists.
        pass
    tiles = []
    for candidate in chosen:
        grid = plan_grid(candidate.parking, gsd_ft=1.0, pad_ft=4.0)
        if grid.width * grid.height > 1_500_000:
            continue
        image = fetch_mosaic(grid)
        stall_px = [polygon_to_pixels(stall, grid) for stall in candidate.stalls]
        mask = rasterize_stall_mask(stall_px, grid.height, grid.width)
        if int(mask.sum()) < 30:
            continue
        path = tile_dir / f"{candidate.way_id}.npz"
        np.savez_compressed(path, image=image, mask=mask)
        tiles.append(
            {
                "way_id": candidate.way_id,
                "split": split_for(candidate.way_id),
                "stalls": len(candidate.stalls),
                "file": str(path.relative_to(out_dir)),
                "height": grid.height,
                "width": grid.width,
            }
        )
    splits = _ensure_val(tiles)
    manifest = {
        "imagery": "NJOGIS Natural2020",
        "labels": "OSM amenity=parking_space (ODbL)",
        "holdout_way_ids": sorted(HOLDOUT_WAY_IDS),
        "holdout_buffer_m": HOLDOUT_BUFFER_M,
        "bbox": list(bbox),
        "lots": len(splits),
        "stalls": sum(item["stalls"] for item in splits),
        "tiles": len(splits),
        "train_lots": sum(1 for item in splits if item["split"] == "train"),
        "val_lots": sum(1 for item in splits if item["split"] == "val"),
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


def _query_overpass(bbox, url: str) -> list[dict]:
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:120];
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
            timeout=180,
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
    candidates = []
    for way_id, lot in parking:
        inside = [stall for stall in stalls if lot.contains(stall.centroid)]
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
