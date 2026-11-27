import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from osm_parking_spaces.cli import main
from osm_parking_spaces.errors import ParkingError
from osm_parking_spaces.export import to_osc, write_outputs
from osm_parking_spaces.geojson_io import load_polygon
from osm_parking_spaces.imagery import pixels_to_wgs84, plan_grid, polygon_to_pixels
from osm_parking_spaces.osm_api import polygon_from_osm

FIXTURES = Path(__file__).parent / "fixtures"


def test_closed_way_becomes_a_polygon():
    polygon = polygon_from_osm(FIXTURES.joinpath("closed_way.xml").read_bytes(), way_id=10)
    assert polygon.geom_type == "Polygon"
    assert polygon.exterior.is_ring
    assert abs(polygon.centroid.y - 40.0005) < 1e-6


def test_open_way_is_rejected():
    with pytest.raises(ParkingError, match="not a closed"):
        polygon_from_osm(FIXTURES.joinpath("open_way.xml").read_bytes())


def test_geojson_feature_loads_one_polygon():
    polygon = load_polygon(FIXTURES / "lot.geojson")
    assert polygon.geom_type == "Polygon"
    assert len(list(polygon.exterior.coords)) == 5


def test_geojson_with_two_polygons_is_rejected(tmp_path: Path):
    collection = {
        "type": "FeatureCollection",
        "features": [
            json.loads((FIXTURES / "lot.geojson").read_text()),
            json.loads((FIXTURES / "lot.geojson").read_text()),
        ],
    }
    path = tmp_path / "two.geojson"
    path.write_text(json.dumps(collection))
    with pytest.raises(ParkingError, match="exactly one polygon"):
        load_polygon(path)


def test_state_plane_grid_round_trips_a_corner():
    polygon = load_polygon(FIXTURES / "lot.geojson")
    grid = plan_grid(polygon, gsd_ft=1.0, pad_ft=0)
    assert grid.gsd_ft == 1.0
    assert grid.width > 50
    assert grid.height > 50
    pixels = polygon_to_pixels(polygon, grid)
    restored = pixels_to_wgs84(pixels, grid)
    for original, returned in zip(polygon.exterior.coords, restored.exterior.coords):
        assert abs(original[0] - returned[0]) < 1e-7
        assert abs(original[1] - returned[1]) < 1e-7


def test_osc_is_create_only_with_negative_ids_and_required_tags():
    square = box(-74.001, 40.0, -74.0009, 40.0001)
    # Give the square a stall-like shape in lon/lat just to exercise the writer.
    stall = Polygon(
        [(-74.001, 40.0), (-74.00095, 40.0), (-74.00095, 40.00012), (-74.001, 40.00012)]
    )
    xml = to_osc([square, stall])
    root = ET.fromstring(xml)
    assert root.tag == "osmChange"
    assert root.find("modify") is None
    assert root.find("delete") is None
    create = root.find("create")
    ways = create.findall("way")
    assert len(ways) == 2
    for way in ways:
        assert int(way.attrib["id"]) < 0
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        assert tags == {
            "amenity": "parking_space",
            "source": "NJOGIS Natural Color Imagery 2020",
        }
        refs = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
        assert refs[0] == refs[-1]
        assert all(ref < 0 for ref in refs)
        assert len(set(refs[:-1])) == 4
    node_ids = [int(node.attrib["id"]) for node in create.findall("node")]
    assert node_ids == list(range(-1, -1 - len(node_ids), -1))
    assert "changeset" not in xml


def test_adjacent_rings_share_one_osc_node():
    left = [(-74.0, 40.0), (-73.9999, 40.0), (-73.9999, 40.0001), (-74.0, 40.0001)]
    right = [(-73.9999, 40.0), (-73.9998, 40.0), (-73.9998, 40.0001), (-73.9999, 40.0001)]
    root = ET.fromstring(to_osc([left, right]))
    create = root.find("create")
    assert len(create.findall("node")) == 6
    refs = [nd.attrib["ref"] for way in create.findall("way") for nd in way.findall("nd")]
    assert len(refs) == 10  # 4 corners + closing node, twice
    assert len(set(refs)) == 6


def test_evaluate_matches_one_to_one_at_iou():
    from shapely.geometry import box as square

    from osm_parking_spaces.evaluate import score_polygons

    truth = [square(0, 0, 2, 2), square(10, 10, 12, 12)]
    detected = [square(0.2, 0.2, 2.2, 2.2), square(5, 5, 6, 6)]
    result, matches = score_polygons(detected, truth, iou_min=0.5)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert len(matches) == 1


def test_empty_detection_still_writes_three_files(tmp_path: Path):
    image = np.zeros((24, 24, 3), np.uint8)
    write_outputs(tmp_path, image, box(1, 1, 22, 22), [], [])
    geojson = json.loads((tmp_path / "stalls.geojson").read_text())
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"] == []
    root = ET.fromstring((tmp_path / "stalls.osc").read_text())
    assert root.find("modify") is None
    assert root.find("delete") is None
    assert root.find("create").find("way") is None
    assert (tmp_path / "preview.png").stat().st_size > 0


def test_cli_rejects_a_missing_geojson_file(tmp_path: Path):
    code = main(
        ["detect", "--geojson", str(tmp_path / "missing.geojson"), "--out-dir", str(tmp_path)]
    )
    assert code == 2


def test_package_does_not_call_the_osm_write_api():
    root = Path(__file__).parents[1] / "osm_parking_spaces"
    blob = "\n".join(path.read_text() for path in root.glob("*.py"))
    assert "changeset" not in blob.lower()
    assert "requests.post" not in blob
    assert "requests.put" not in blob
    assert "requests.delete" not in blob
