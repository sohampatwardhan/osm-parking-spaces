"""Load a single parking-area polygon from GeoJSON."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape

from osm_parking_spaces.errors import ParkingError


def load_polygon(path: Path):
    """Return one WGS84 polygon from a Polygon, Feature, or one-feature collection."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ParkingError(f"Could not read GeoJSON {path}: {exc}") from exc
    geometries = _geometries(data)
    polygons = []
    for geometry in geometries:
        if geometry is None:
            continue
        kind = geometry.get("type")
        if kind == "Polygon":
            polygons.append(shape(geometry))
        elif kind == "MultiPolygon":
            raise ParkingError("MultiPolygon input is not supported; pass one Polygon")
        elif kind is not None:
            raise ParkingError(f"Unsupported GeoJSON geometry type {kind}")
    if len(polygons) != 1:
        raise ParkingError(f"Expected exactly one polygon, found {len(polygons)}")
    polygon = polygons[0]
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        raise ParkingError("GeoJSON did not contain a valid polygon")
    return polygon


def _geometries(data: dict) -> list:
    kind = data.get("type")
    if kind == "FeatureCollection":
        return [feature.get("geometry") for feature in data.get("features", [])]
    if kind == "Feature":
        return [data.get("geometry")]
    if kind in {"Polygon", "MultiPolygon"}:
        return [data]
    raise ParkingError(f"Unsupported GeoJSON type {kind!r}")
