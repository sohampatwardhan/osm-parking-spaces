"""Write GeoJSON, a create-only OsmChange file, and a preview PNG."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from osm_parking_spaces.detect import Stall
from osm_parking_spaces.errors import ParkingError

SOURCE = "NJOGIS Natural Color Imagery 2020"


def write_outputs(
    directory: Path,
    image_bgr: np.ndarray,
    lot_px: Polygon,
    stalls_px: list[Stall],
    rings_wgs84: list[list[tuple[float, float]]],
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    geojson_path = directory / "stalls.geojson"
    osc_path = directory / "stalls.osc"
    preview_path = directory / "preview.png"
    geojson_path.write_text(json.dumps(_geojson(rings_wgs84, stalls_px), indent=2) + "\n")
    osc_path.write_text(to_osc(rings_wgs84))
    _write_preview(preview_path, image_bgr, lot_px, stalls_px)
    return geojson_path, osc_path, preview_path


def to_osc(stalls, source: str = SOURCE) -> str:
    """Create-only OsmChange. Ids are negative. Shared corners are one node.

    `stalls` may be shapely polygons or rings of (lon, lat) pairs. Equal
    coordinates become one node, so adjacent stalls reuse divider corners.
    """
    rings = [_orient_ring(_as_ring(item)) for item in stalls]
    root = ET.Element("osmChange", version="0.6", generator="osm-parking-spaces")
    create = ET.SubElement(root, "create")
    node_ids: dict[tuple[float, float], int] = {}
    node_id = -1
    for ring in rings:
        for point in ring:
            if point not in node_ids:
                node_ids[point] = node_id
                lon, lat = point
                ET.SubElement(create, "node", id=str(node_id), lon=f"{lon:.7f}", lat=f"{lat:.7f}")
                node_id -= 1
    way_id = -1
    for ring in rings:
        if len(ring) < 3:
            continue
        way = ET.SubElement(create, "way", id=str(way_id))
        way_id -= 1
        refs = [node_ids[point] for point in ring]
        for ref in refs:
            ET.SubElement(way, "nd", ref=str(ref))
        ET.SubElement(way, "nd", ref=str(refs[0]))
        ET.SubElement(way, "tag", k="amenity", v="parking_space")
        ET.SubElement(way, "tag", k="source", v=source)
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml + "\n"


def _as_ring(item) -> list[tuple[float, float]]:
    if isinstance(item, Polygon):
        coords = list(orient(item, sign=1.0).exterior.coords)[:-1]
        return [(round(float(x), 7), round(float(y), 7)) for x, y in coords]
    return [(round(float(x), 7), round(float(y), 7)) for x, y in item]


def _orient_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(ring) < 3:
        return ring
    polygon = orient(Polygon(ring), sign=1.0)
    ordered = []
    for x, y in list(polygon.exterior.coords)[:-1]:
        key = (round(float(x), 7), round(float(y), 7))
        ordered.append(min(ring, key=lambda point: abs(point[0] - key[0]) + abs(point[1] - key[1])))
    return ordered


def _geojson(rings: list[list[tuple[float, float]]], stalls: list[Stall]) -> dict:
    features = []
    for index, (ring, stall) in enumerate(zip(rings, stalls), start=1):
        ring = _orient_ring(_as_ring(ring))
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": index,
                    "amenity": "parking_space",
                    "source": SOURCE,
                    "width_ft": round(stall.width_ft, 1),
                    "length_ft": round(stall.length_ft, 1),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring + [ring[0]]],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _write_preview(path: Path, image_bgr: np.ndarray, lot: Polygon, stalls: list[Stall]) -> None:
    vis = image_bgr.copy()
    overlay = image_bgr.copy()
    for stall in stalls:
        points = _ring(stall.polygon)
        cv2.fillPoly(overlay, [points], (40, 180, 40))
        cv2.polylines(vis, [points], True, (0, 190, 0), 1, cv2.LINE_AA)
    painted = cv2.addWeighted(overlay, 0.40, vis, 0.60, 0)
    cv2.polylines(painted, [_ring(lot)], True, (0, 220, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), painted):
        raise ParkingError(f"Could not write preview {path}")


def _ring(polygon: Polygon) -> np.ndarray:
    return np.round(polygon.exterior.coords).astype(np.int32)
