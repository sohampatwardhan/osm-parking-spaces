"""Drop parking lots that are too sparse or drawn as one big polygon."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

MIN_STALLS = 8
MIN_COVERAGE = 0.12
MAX_COVERAGE = 0.90


def stall_coverage(parking: Polygon, stalls: list[Polygon]) -> float:
    if parking.is_empty or parking.area <= 0 or not stalls:
        return 0.0
    covered = unary_union(stalls).intersection(parking)
    return float(covered.area / parking.area)


def passes_quality(
    parking: Polygon,
    stalls: list[Polygon],
    min_stalls: int = MIN_STALLS,
    min_coverage: float = MIN_COVERAGE,
    max_coverage: float = MAX_COVERAGE,
) -> bool:
    if len(stalls) < min_stalls:
        return False
    coverage = stall_coverage(parking, stalls)
    return min_coverage <= coverage <= max_coverage
