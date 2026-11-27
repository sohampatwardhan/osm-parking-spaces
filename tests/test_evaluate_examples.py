"""Score the committed ShopRite examples against the hand-traced ground truth."""

import xml.etree.ElementTree as ET
from pathlib import Path

from osm_parking_spaces.evaluate import score_files

ROOT = Path(__file__).parents[1]

# Locked to the files in this commit so a silent regression fails CI.
EXPECTED = {
    1329737820: {"tp": 58, "fp": 38, "fn": 26},
    1329737821: {"tp": 117, "fp": 59, "fn": 78},
}


def test_shoprite_examples_match_the_recorded_scores():
    for way, expected in EXPECTED.items():
        result = score_files(
            ROOT / f"examples/way-{way}/stalls.geojson",
            ROOT / f"tests/ground_truth/way-{way}.geojson",
        )
        assert result.true_positives == expected["tp"]
        assert result.false_positives == expected["fp"]
        assert result.false_negatives == expected["fn"]
        assert 0 < result.precision < 1
        assert 0 < result.recall < 1
        precision = result.true_positives / (result.true_positives + result.false_positives)
        recall = result.true_positives / (result.true_positives + result.false_negatives)
        assert result.precision == precision
        assert result.recall == recall
        assert result.mean_offset_ft < 4


def test_example_osc_reuses_shared_divider_nodes():
    root = ET.parse(ROOT / "examples/way-1329737820/stalls.osc").getroot()
    create = root.find("create")
    assert root.find("modify") is None
    assert root.find("delete") is None
    nodes = create.findall("node")
    ways = create.findall("way")
    assert ways
    assert all(int(node.attrib["id"]) < 0 for node in nodes)
    assert all(int(way.attrib["id"]) < 0 for way in ways)
    # Four private corners per stall would be 4 nodes per way. Sharing is lower.
    assert len(nodes) < 3 * len(ways)
