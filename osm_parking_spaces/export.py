"""Write GeoJSON, a create-only OsmChange file, and a preview PNG."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.geometry.polygon import orient
from shapely.ops import transform

from osm_parking_spaces.detect import Stall
from osm_parking_spaces.errors import ParkingError

SOURCE = "NJOGIS Natural Color Imagery 2020"


def write_outputs(
    directory: Path,
    image_bgr: np.ndarray,
    lot_px: Polygon,
    stalls_px: list[Stall],
    stalls_wgs84: list[Polygon],
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    geojson_path = directory / "stalls.geojson"
    osc_path = directory / "stalls.osc"
    preview_path = directory / "preview.png"
    geojson_path.write_text(json.dumps(_geojson(stalls_wgs84, stalls_px), indent=2) + "\n")
    osc_path.write_text(to_osc(stalls_wgs84))
    _write_preview(preview_path, image_bgr, lot_px, stalls_px)
    return geojson_path, osc_path, preview_path


def to_osc(polygons: list[Polygon], source: str = SOURCE) -> str:
    """Create-only OsmChange. Ids are negative. There is no modify or delete."""
    root = ET.Element("osmChange", version="0.6", generator="osm-parking-spaces")
    create = ET.SubElement(root, "create")
    node_id = -1
    way_id = -1
    for polygon in polygons:
        ring = list(orient(polygon, sign=1.0).exterior.coords)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        body = ring[:-1]
        if len(body) < 3:
            continue
        refs: list[int] = []
        for lon, lat in body:
            ET.SubElement(
                create,
                "node",
                id=str(node_id),
                lon=f"{lon:.7f}",
                lat=f"{lat:.7f}",
            )
            refs.append(node_id)
            node_id -= 1
        way = ET.SubElement(create, "way", id=str(way_id))
        way_id -= 1
        for ref in refs:
            ET.SubElement(way, "nd", ref=str(ref))
        ET.SubElement(way, "nd", ref=str(refs[0]))
        ET.SubElement(way, "tag", k="amenity", v="parking_space")
        ET.SubElement(way, "tag", k="source", v=source)
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml + "\n"


def _geojson(polygons: list[Polygon], stalls: list[Stall]) -> dict:
    features = []
    for index, (polygon, stall) in enumerate(zip(polygons, stalls), start=1):
        rounded = transform(lambda x, y: (round(x, 7), round(y, 7)), orient(polygon, sign=1.0))
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
                "geometry": mapping(rounded),
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
