"""Tie polygon input, imagery, detection, and the three output files together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from osm_parking_spaces.detect import detect_stalls
from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.export import write_outputs
from osm_parking_spaces.geojson_io import load_polygon
from osm_parking_spaces.imagery import (
    WMS_LAYER,
    WMS_URL,
    fetch_mosaic,
    plan_grid,
    polygon_to_pixels,
    shared_rings,
)
from osm_parking_spaces.osm_api import fetch_way


@dataclass
class RunResult:
    stall_count: int
    geojson_path: Path
    osc_path: Path
    preview_path: Path


def detect_way(way_id: int, out_dir: Path, gsd_ft: float = 1.0) -> RunResult:
    return run(fetch_way(way_id), out_dir, gsd_ft=gsd_ft)


def detect_geojson(path: Path, out_dir: Path, gsd_ft: float = 1.0) -> RunResult:
    return run(load_polygon(path), out_dir, gsd_ft=gsd_ft)


def run(
    lot_wgs84,
    out_dir: Path,
    gsd_ft: float = 1.0,
    wms_url: str = WMS_URL,
    wms_layer: str = WMS_LAYER,
) -> RunResult:
    grid = plan_grid(lot_wgs84, gsd_ft=gsd_ft)
    if grid.width * grid.height > 8_000_000:
        raise ParkingError(
            f"Parking area would be {grid.width}x{grid.height} pixels; "
            "pass a smaller polygon or a coarser --gsd-ft"
        )
    image = fetch_mosaic(grid, url=wms_url, layer=wms_layer)
    lot_px = polygon_to_pixels(lot_wgs84, grid)
    stalls = detect_stalls(image, lot_px, gsd_ft=gsd_ft)
    rings = shared_rings(stalls, grid)
    geojson_path, osc_path, preview_path = write_outputs(out_dir, image, lot_px, stalls, rings)
    return RunResult(len(stalls), geojson_path, osc_path, preview_path)
