import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from audit_r30_paddle_pilot import leaf_changes

from document_ocr.synthesis.paddle_validation import (
    check_private_residue,
    check_synthetic_relations,
)


def test_replay_target_enforces_reduced_training_schema_as_well_as_full_v7():
    from audit_r30_paddle_pilot import check_training_target
    from jsonschema import ValidationError

    valid = {"billOfLadingNumber": "SYN12345", "negotiability": "non_negotiable"}
    check_training_target(valid)
    # Full V7 accepts this annotation, but the frozen training projection does not.
    with pytest.raises(ValidationError, match="forwardingAndExportReferences"):
        check_training_target({**valid, "forwardingAndExportReferences": ["12345678"]})
    with pytest.raises(ValidationError):
        check_training_target({"goodsItemDetails": [{"grossWeight": {"value": "bad"}}]})


def test_private_owner_screen_checks_all_repetitions_including_split_text():
    privacy = {"proof": [{"owner": "tax", "spans": [{"expected": "1234567"}]}]}
    check_private_residue(privacy, {"joinedRawText": "TAX: 7654321"})
    with pytest.raises(ValueError, match="survives"):
        check_private_residue(privacy, {"joinedRawText": "TAX: 7654321\nOTHER TAX: 1234 567"})


def test_complete_ledger_conserves_multiple_rows_but_partial_remains_partial():
    target = {
        "containerInformation": [{"equipmentIdentifier": "A"}, {"equipmentIdentifier": "B"}],
        "goodsItemDetails": [
            {
                "numberAndTypeOfPackages": [{"packageQuantity": 12}, {"packageQuantity": 8}],
                "splitGoodsPlacement": [
                    {"equipmentIdentifier": "A", "packageQuantity": 7},
                    {"equipmentIdentifier": "B", "packageQuantity": 13},
                ],
            }
        ],
    }
    check_synthetic_relations(target)
    partial = copy.deepcopy(target)
    partial["goodsItemDetails"][0]["splitGoodsPlacement"].pop()
    check_synthetic_relations(partial)
    target["goodsItemDetails"][0]["splitGoodsPlacement"][1]["packageQuantity"] = 12
    with pytest.raises(ValueError, match="ledger"):
        check_synthetic_relations(target)


def test_leaf_changes_preserve_index_and_distinguish_absence_from_null():
    assert leaf_changes({"p": [{"x": None, "y": 1}]}, {"p": [{"x": 2, "z": None}]}) == [
        {"path": "/p/0/x", "action": "change"},
        {"path": "/p/0/y", "action": "remove"},
        {"path": "/p/0/z", "action": "add"},
    ]
