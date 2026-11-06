import copy
from pathlib import Path

import pytest

from document_ocr.synthesis.paddle_render import extra_value


def evaluate(target, receipt=None, index=0):
    return extra_value(
        {"allocation": {"index": index, "authority": "public_target"}},
        {"target": {"documentPatch": receipt or target}},
        {"target": target},
        "sample",
        None,
        project_root=Path("."),
    )


@pytest.fixture
def target():
    return {
        "containerInformation": [{"equipmentIdentifier": "A"}, {"equipmentIdentifier": "B"}],
        "goodsItemDetails": [
            {"splitGoodsPlacement": [{"equipmentIdentifier": "B", "packageQuantity": 7}]}
        ],
    }


def test_explicit_public_repetition_needs_no_private_plan(target):
    assert evaluate(target, index=1) == ("7", 7)


def test_missing_allocation_is_not_inferred(target):
    with pytest.raises(ValueError, match="explicit allocation"):
        evaluate(target)


def test_stale_quantity_is_rejected(target):
    other = copy.deepcopy(target)
    other["goodsItemDetails"][0]["splitGoodsPlacement"][0]["packageQuantity"] = 8
    with pytest.raises(ValueError, match="retained shipment receipt"):
        evaluate(target, other, index=1)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_public_count_is_positive_integer(target, value):
    target["goodsItemDetails"][0]["splitGoodsPlacement"][0]["packageQuantity"] = value
    with pytest.raises(ValueError, match="positive integer"):
        evaluate(target, index=1)
