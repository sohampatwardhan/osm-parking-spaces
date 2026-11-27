"""Keep the ShopRite evaluation lots out of the training set."""

from __future__ import annotations

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import unary_union

HOLDOUT_WAY_IDS = frozenset({1329737820, 1329737821})
HOLDOUT_BUFFER_M = 150.0


def buffer_meters(polygon: Polygon, meters: float) -> Polygon:
    """Buffer a WGS84 polygon by `meters` using a local metric CRS."""
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:26918", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:26918", "EPSG:4326", always_xy=True)

    def _apply(transformer, coords):
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    projected = shapely.transform(polygon, lambda coords: _apply(to_metric, coords))
    widened = projected.buffer(meters)
    return shapely.transform(widened, lambda coords: _apply(to_wgs84, coords))


def holdout_region(polygons: list[Polygon], meters: float = HOLDOUT_BUFFER_M) -> Polygon:
    """Union of the held-out lots, expanded by `meters`."""
    if not polygons:
        raise ValueError("holdout region needs at least one polygon")
    return unary_union([buffer_meters(polygon, meters) for polygon in polygons])


def is_held_out(way_id: int | None, polygon: Polygon, region: Polygon) -> bool:
    """True for the two evaluation ways and for any lot that touches their buffer."""
    if way_id in HOLDOUT_WAY_IDS:
        return True
    return polygon.intersects(region)
