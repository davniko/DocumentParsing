"""The bounded refresh distinguishes public occurrences from sampled groups."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/synthesis/refresh_r29_equal_loads.py"
SPEC = importlib.util.spec_from_file_location("r29_refresh", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        (["PACKAGE_BAG"], 1),
        (["PACKAGE_BAG"] * 9, 2),
        (["PACKAGE_BAG", "PACKAGE_DRUM"] * 8, 2),
        (["PACKAGE_BAG", "PACKAGE_DRUM", "PACKAGE_CARTON"] * 6, 3),
    ],
)
def test_group_count_does_not_multiply_container_occurrences(categories, expected):
    row = {
        "target": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"typeCategory": category, "packageQuantity": 10} for category in categories
                    ]
                }
            ]
        }
    }
    assert MODULE.package_groups(row) == expected


def test_mixed_probability_has_an_explicit_eligible_scope():
    cfg = {"source_ids": ["a", "b"], "independent_preparation": {"mixed_source_ids": []}}
    MODULE.configure_equipment(cfg, 0.5)
    assert cfg["independent_preparation"] == {
        "mixed_source_ids": ["a", "b"],
        "mixed_probability": 0.5,
    }


def test_mixed_scope_cannot_be_silently_empty():
    with pytest.raises(ValueError, match="selected sources"):
        MODULE.configure_equipment({"source_ids": [], "independent_preparation": {}}, 1)
