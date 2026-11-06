import copy
import importlib.util
from pathlib import Path
import sys

import pytest

from document_ocr.synthesis.curated import digest


scripts = Path(__file__).resolve().parents[1] / "scripts/migrations"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location(
        "support_clarification", scripts / "clarify_r30_disjoint_supports.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def fixture():
    children = [
        {"relation": "NONE", "wordingAppends": {"product": "6 PALLETS."}},
        {"relation": "ON_SUPPORT", "wordingAppends": {"product": "144 ROLLS ON 6 PALLETS."}},
    ]
    clause = "6 PALLETS. 144 ROLLS ON 6 PALLETS."
    value = "TEXTILE. " + clause
    ledger = {
        "relation": "DISJOINT_GROUPS",
        "facts": {"declaredGroupLedgers": children},
        "wordingAppends": {"product": clause},
        "publicQuantity": 150,
    }
    return {
        "physical": {"freshPrivatePacking": ledger},
        "scenario": {"provenance": {"freshPrivatePacking": copy.deepcopy(ledger)}},
        "renderValues": {"product": value},
        "joinedRawText": value,
        "proof": {
            "sourceSha256": digest(b"OLD"),
            "renderedSha256": digest(value.encode()),
            "edits": [
                {"key": "product", "byteStart": 0, "byteEnd": 3, "before": "OLD", "after": value}
            ],
        },
        "target": {
            "documentPatch": {
                "goodsItemDetails": [
                    {
                        "description": value,
                        "numberAndTypeOfPackages": [
                            {"packageQuantity": 6, "typeCategory": "PACKAGE_PALLET"},
                            {"packageQuantity": 144, "typeCategory": "PACKAGE_ROLL"},
                        ],
                        "grossWeight": {"value": 1000},
                    }
                ]
            }
        },
        "numericWordingProjection": {"ownedClauses": {"product": clause}},
    }


def test_clarification_preserves_accounting_and_original_receipt():
    candidate = fixture()
    before = copy.deepcopy(candidate)
    result, changes = module.clarify(candidate)
    assert candidate == before
    assert (
        result["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
        == before["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
    )
    assert result["target"]["documentPatch"]["goodsItemDetails"][0]["grossWeight"] == {
        "value": 1000
    }
    assert result["joinedRawText"].endswith(
        "SUPPORT UNITS ARE ADDITIONAL TO THE DECLARED PACKAGE GROUPS."
    )
    assert (
        changes["product"]["before"]
        == before["physical"]["freshPrivatePacking"]["wordingAppends"]["product"]
    )


def test_inexact_clauses_and_disagreeing_ledger_are_rejected():
    candidate = fixture()
    candidate["proof"]["edits"][0]["after"] = "different"
    with pytest.raises(ValueError, match="exact unique"):
        module.clarify(candidate)
    candidate = fixture()
    candidate["scenario"]["provenance"]["freshPrivatePacking"]["publicQuantity"] = 42
    with pytest.raises(ValueError, match="authorities disagree"):
        module.clarify(candidate)
