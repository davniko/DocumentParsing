import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from complete_r30_missing_measures import complete, context


def fixture():
    return {
        "target": {
            "documentPatch": {
                "goodsItemDetails": [
                    {
                        "description": "CABLE",
                        "numberAndTypeOfPackages": [
                            {"packageQuantity": 607, "typeCategory": "PACKAGE_CARTON"}
                        ],
                    }
                ],
                "containerInformation": [{"equipmentIdentifier": "TEST1234567"}],
            }
        },
        "physical": {"measures": {}},
        "scenario": {
            "provenance": {
                "physicalRows": [{"capacity": {"payloadKg": "28300", "volumeM3": "33.2"}}]
            }
        },
        "joinedRawText": "retained text",
        "proof": {"retained": True},
    }


def test_completes_private_missing_totals_without_changing_retained_owners():
    original = fixture()
    before = copy.deepcopy(original)
    result = complete(original, {"unknown": False, "grossKg": 12800, "volumeM3": 30.4})
    assert original == before
    assert result["scenario"]["provenance"].pop("explicitPhysicalFacts") == {
        "grossWeight": {"unit": "kilogram", "value": 12800.0},
        "volume": {"unit": "cubic_metre", "value": 30.4},
    }
    assert result == original


@pytest.mark.parametrize(
    "key,value",
    [
        ("unknown", True),
        ("grossKg", 28300.01),
        ("volumeM3", 33.2001),
        ("grossKg", 0),
        ("volumeM3", -1),
        ("grossKg", "NaN"),
        ("volumeM3", "Infinity"),
    ],
)
def test_rejects_unknown_and_outside_capacity(key, value):
    values = {"unknown": False, "grossKg": 12800, "volumeM3": 30.4, key: value}
    with pytest.raises(ValueError):
        complete(fixture(), values)


@pytest.mark.parametrize("kind", ["public", "private", "explicit", "goods", "equipment"])
def test_does_not_overwrite_existing_facts_or_handle_unreviewed_topologies(kind):
    candidate = fixture()
    if kind == "public":
        candidate["target"]["documentPatch"]["goodsItemDetails"][0]["grossWeight"] = 1
    elif kind == "private":
        candidate["physical"]["measures"] = {"grossWeight": 1}
    elif kind == "explicit":
        candidate["scenario"]["provenance"]["explicitPhysicalFacts"] = {}
    elif kind == "goods":
        candidate["target"]["documentPatch"]["goodsItemDetails"].append({})
    else:
        candidate["scenario"]["provenance"]["physicalRows"].append({})
    with pytest.raises(ValueError):
        context(candidate)
