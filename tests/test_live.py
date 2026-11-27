"""Opt-in checks against the public OSM API and the NJOGIS WMS.

Run with: pytest -m live
"""

import pytest

from osm_parking_spaces.imagery import Grid, fetch_mosaic, plan_grid
from osm_parking_spaces.osm_api import fetch_way

pytestmark = pytest.mark.live


def test_shoprite_way_is_a_closed_parking_polygon():
    polygon = fetch_way(1329737820)
    assert polygon.geom_type == "Polygon"
    assert polygon.centroid.y == pytest.approx(40.384, abs=0.01)
    assert polygon.centroid.x == pytest.approx(-74.573, abs=0.01)


def test_wms_returns_a_small_natural2020_tile():
    polygon = fetch_way(1329737820)
    grid = plan_grid(polygon, gsd_ft=1.0, pad_ft=4)
    # Request only the north-west 32 ft so the live check stays small.
    sample = Grid(grid.minx, grid.maxy - 32, grid.minx + 32, grid.maxy, 32, 32, 1.0)
    image = fetch_mosaic(sample)
    assert image.shape == (32, 32, 3)
    assert int(image.mean()) > 10
