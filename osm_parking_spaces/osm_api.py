"""Read a closed way from the OSM API, or parse a saved API document."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import requests
from shapely.geometry import Polygon

from osm_parking_spaces.errors import ParkingError

OSM_API = "https://api.openstreetmap.org/api/0.6"
USER_AGENT = "osm-parking-spaces/0.1 (https://github.com/sohampatwardhan/osm-parking-spaces)"


def fetch_way(way_id: int, timeout: float = 60) -> Polygon:
    """Download one way and its nodes. Read-only; no credentials."""
    if way_id <= 0:
        raise ParkingError(f"OSM way id must be a positive integer, got {way_id}")
    url = f"{OSM_API}/way/{way_id}/full"
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    except requests.RequestException as exc:
        raise ParkingError(f"Could not reach the OSM API: {exc}") from exc
    if response.status_code == 404:
        raise ParkingError(f"OSM way {way_id} was not found")
    if response.status_code != 200:
        raise ParkingError(f"OSM API returned HTTP {response.status_code} for way {way_id}")
    return polygon_from_osm(response.content, way_id)


def polygon_from_osm(xml_bytes: bytes, way_id: int | None = None) -> Polygon:
    """Build a WGS84 polygon from an OSM API 0.6 document."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ParkingError(f"OSM response was not valid XML: {exc}") from exc
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall("node"):
        nodes[node.attrib["id"]] = (float(node.attrib["lon"]), float(node.attrib["lat"]))
    ways = root.findall("way")
    if not ways:
        raise ParkingError("OSM document has no way")
    way = ways[0]
    if way_id is not None:
        match = [item for item in ways if item.attrib.get("id") == str(way_id)]
        if not match:
            raise ParkingError(f"OSM document does not contain way {way_id}")
        way = match[0]
    refs = [nd.attrib["ref"] for nd in way.findall("nd")]
    if len(refs) < 4:
        raise ParkingError(f"Way {way.attrib.get('id', '')} is not a closed ring")
    if refs[0] != refs[-1]:
        raise ParkingError(f"Way {way.attrib.get('id', '')} is not closed")
    try:
        ring = [nodes[ref] for ref in refs]
    except KeyError as exc:
        raise ParkingError(f"Way references missing node {exc}") from exc
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        raise ParkingError(f"Way {way.attrib.get('id', '')} did not form a polygon")
    return polygon
