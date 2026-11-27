"""Fetch NJOGIS Natural2020 and keep a foot-grid georeference without GDAL."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import Polygon

from osm_parking_spaces.errors import ParkingError

WMS_URL = "https://img.nj.gov/imagerywms/Natural2020"
WMS_LAYER = "Natural2020"
IMAGE_CRS = "EPSG:6527"
USER_AGENT = "osm-parking-spaces/0.1 (https://github.com/sohampatwardhan/osm-parking-spaces)"
MAX_TILE_PX = 1500


@dataclass(frozen=True)
class Grid:
    """Axis-aligned image grid in EPSG:6527 US survey feet. Y pixel axis points south."""

    minx: float
    miny: float
    maxx: float
    maxy: float
    width: int
    height: int
    gsd_ft: float

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.minx) / self.gsd_ft, (self.maxy - y) / self.gsd_ft

    def pixel_to_world(self, col: float, row: float) -> tuple[float, float]:
        return self.minx + col * self.gsd_ft, self.maxy - row * self.gsd_ft


def plan_grid(polygon: Polygon, gsd_ft: float = 1.0, pad_ft: float = 8.0) -> Grid:
    """Cover a WGS84 polygon with a 6527 grid at `gsd_ft` feet per pixel."""
    if gsd_ft <= 0:
        raise ParkingError("--gsd-ft must be positive")
    to_state_plane = Transformer.from_crs("EPSG:4326", IMAGE_CRS, always_xy=True)
    xs, ys = [], []
    for lon, lat in polygon.exterior.coords:
        x, y = to_state_plane.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    minx = math.floor(min(xs) - pad_ft)
    miny = math.floor(min(ys) - pad_ft)
    maxx = math.ceil(max(xs) + pad_ft)
    maxy = math.ceil(max(ys) + pad_ft)
    width = max(1, int(round((maxx - minx) / gsd_ft)))
    height = max(1, int(round((maxy - miny) / gsd_ft)))
    # Snap the far edges so each pixel is exactly gsd_ft wide.
    maxx = minx + width * gsd_ft
    maxy = miny + height * gsd_ft
    return Grid(minx, miny, maxx, maxy, width, height, gsd_ft)


def polygon_to_pixels(polygon: Polygon, grid: Grid) -> Polygon:
    to_state_plane = Transformer.from_crs("EPSG:4326", IMAGE_CRS, always_xy=True)

    def ring(coords):
        return [grid.world_to_pixel(*to_state_plane.transform(lon, lat)) for lon, lat in coords]

    holes = [ring(interior.coords) for interior in polygon.interiors]
    return Polygon(ring(polygon.exterior.coords), holes)


def pixels_to_wgs84(polygon: Polygon, grid: Grid) -> Polygon:
    to_wgs84 = Transformer.from_crs(IMAGE_CRS, "EPSG:4326", always_xy=True)

    def ring(coords):
        return [to_wgs84.transform(*grid.pixel_to_world(col, row)) for col, row in coords]

    holes = [ring(interior.coords) for interior in polygon.interiors]
    return Polygon(ring(polygon.exterior.coords), holes)


def fetch_mosaic(
    grid: Grid,
    url: str = WMS_URL,
    layer: str = WMS_LAYER,
    timeout: float = 120,
) -> np.ndarray:
    """Download Natural2020 for `grid` as a BGR uint8 image. Tiles if needed."""
    mosaic = np.zeros((grid.height, grid.width, 3), np.uint8)
    for row0 in range(0, grid.height, MAX_TILE_PX):
        for col0 in range(0, grid.width, MAX_TILE_PX):
            tile_w = min(MAX_TILE_PX, grid.width - col0)
            tile_h = min(MAX_TILE_PX, grid.height - row0)
            west = grid.minx + col0 * grid.gsd_ft
            north = grid.maxy - row0 * grid.gsd_ft
            east = west + tile_w * grid.gsd_ft
            south = north - tile_h * grid.gsd_ft
            tile = _get_map(url, layer, west, south, east, north, tile_w, tile_h, timeout)
            mosaic[row0 : row0 + tile_h, col0 : col0 + tile_w] = tile
    return mosaic


def _get_map(url, layer, west, south, east, north, width, height, timeout) -> np.ndarray:
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "SRS": IMAGE_CRS,
        "BBOX": f"{west},{south},{east},{north}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
    }
    try:
        response = requests.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
    except requests.RequestException as exc:
        raise ParkingError(f"Could not reach the imagery service: {exc}") from exc
    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200 or "image" not in content_type:
        detail = response.content[:200]
        raise ParkingError(
            f"Imagery request failed (HTTP {response.status_code}, {content_type}): {detail!r}"
        )
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ParkingError("Imagery response was not a readable image")
    if image.shape[1] != width or image.shape[0] != height:
        raise ParkingError(
            f"Imagery size {image.shape[1]}x{image.shape[0]} did not match {width}x{height}"
        )
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
