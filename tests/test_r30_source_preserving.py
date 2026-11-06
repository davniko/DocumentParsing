"""Identifier-only migrations cannot create cargo facts or hide new conflicts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from r30_source_preserving import accounting_signature, identifier_mapping, identifier_target


def contract():
    return {"variables": [{"key": "id", "value": "ABCU1234567"}]}


def test_identity_mapping_retains_reviewed_ocr_error():
    assert identifier_mapping(contract(), {"id": "FGHU7654321"}, {
        "id": {"original": "ABCU1234567", "native": "A8CU1234567", "retainedOcrMask": {"1": "8"}}
    }) == {"A8CU1234567": "F8HU7654321"}


@pytest.mark.parametrize("values", [{}, {"id": "ABCU1234567"}, {"id": "TOO LONG VALUE"}, {"id": "FGHU7654321", "extra": "A"}])
def test_bad_identifier_values_rejected(values):
    with pytest.raises(ValueError):
        identifier_mapping(contract(), values, {})


def target():
    return {
        "billOfLadingNumber": "BL123",
        "containerInformation": [{"equipmentIdentifier": "ABCU1234567", "sealNumbers": ["98765"]}],
        "goodsItemDetails": [{"descriptionOfGoods": "GOODS", "numberAndTypeOfPackages": [{"packageQuantity": 800, "packageType": "PACKAGE"}],
                              "splitGoodsPlacement": [{"equipmentIdentifier": "ABCU1234567", "packageQuantity": 20}]}],
    }


def test_exact_target_inversion_and_preserved_conflict():
    before = target()
    mapping = {"BL123": "BL456", "ABCU1234567": "FGHU7654321", "98765": "54321"}
    after = identifier_target(before, mapping)
    assert after != before
    assert identifier_target(after, {v: k for k, v in mapping.items()}) == before
    assert accounting_signature(after) == accounting_signature(before)
    assert after["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["packageQuantity"] == 800


def test_non_identifier_target_must_not_change():
    t = target()
    t["goodsItemDetails"][0]["descriptionOfGoods"] = "ABCU1234567"
    with pytest.raises(ValueError, match="non-identifier"):
        identifier_target(t, {"ABCU1234567": "FGHU7654321"})


def test_invalid_allocation_still_rejected():
    t = target()
    t["goodsItemDetails"][0]["splitGoodsPlacement"][0]["equipmentIdentifier"] = "MISSING"
    with pytest.raises(ValueError, match="allocation identity"):
        accounting_signature(t)


def test_duplicate_equipment_still_rejected():
    t = target()
    t["containerInformation"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        accounting_signature(t)


def test_duplicate_allocation_still_rejected():
    t = target()
    t["goodsItemDetails"][0]["splitGoodsPlacement"] *= 2
    with pytest.raises(ValueError, match="allocation identity"):
        accounting_signature(t)
