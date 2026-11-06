"""Negative controls for audit denominators and screens; no dataset mutation."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "r30_readiness",
    Path(__file__).parents[1] / "scripts/migrations/audit_r30_training_readiness.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def cargo(allocations):
    return {
        "containerInformation": [
            {"equipmentIdentifier": "AAAA0000001"},
            {"equipmentIdentifier": "AAAA0000002"},
        ],
        "goodsItemDetails": [{"splitGoodsPlacement": allocations}],
    }


def test_partial_quantities_do_not_enter_complete_equal_denominator():
    target = cargo(
        [
            {"equipmentIdentifier": "AAAA0000001", "packageQuantity": 2},
            {"equipmentIdentifier": "AAAA0000002"},
        ]
    )
    assert not audit.features(target)["complete_multi_allocation"]
    assert not audit.features(target)["equal_multi_allocation"]


def test_complete_equal_requires_correct_distinct_id_membership():
    target = cargo(
        [
            {"equipmentIdentifier": "AAAA0000001", "packageQuantity": 2},
            {"equipmentIdentifier": "AAAA0000001", "packageQuantity": 2},
        ]
    )
    assert not audit.features(target)["complete_multi_allocation"]
    target["goodsItemDetails"][0]["splitGoodsPlacement"][1]["equipmentIdentifier"] = "AAAA0000002"
    assert audit.features(target)["equal_multi_allocation"]


def test_identifier_rendering_punctuation_is_not_a_missing_fact():
    row = {"joinedRawText": "AAAA 000000-1", "target": cargo([])}
    flags = audit.literal_screen(row)
    assert len(flags) == 1
    assert flags[0]["value"] == "AAAA0000002"


def test_handling_array_screen_is_included():
    row = {
        "joinedRawText": "KEEP DRY",
        "target": {"goodsItemDetails": [{"handlingInstructions": ["KEEP FROZEN"]}]},
    }
    assert audit.literal_screen(row)[0]["missing"] == ["FROZEN"]


def test_vertical_source_wording_does_not_distort_horizontal_density():
    row = {
        "pages": [
            {
                "index": 0,
                "width": 1000,
                "height": 1000,
                "regions": [
                    {"index": 0, "text": "FOUR", "box": [1, 1, 101, 21]},
                    {"index": 1, "text": "VERTICAL", "box": [900, 1, 910, 800]},
                ],
            }
        ]
    }
    g = audit.geometry(row)
    assert g["density_p99"] > 100
    assert g["horizontal_density_p99"] == 0.8


def test_quantized_centroid_collision_is_visible_to_screen():
    row = {
        "pages": [
            {
                "index": 0,
                "width": 1000,
                "height": 1000,
                "regions": [
                    {"index": 0, "text": "A", "box": [10, 10, 20, 20]},
                    {"index": 1, "text": "B", "box": [10.1, 10, 20.1, 20]},
                ],
            }
        ]
    }
    assert audit.geometry(row)["centroid_collisions"] == 1
