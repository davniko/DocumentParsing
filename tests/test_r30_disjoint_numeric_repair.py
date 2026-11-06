import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from amend_r30_numeric_surfaces import checked_target_changes
from repair_r30_disjoint_numeric_surfaces import repair

from document_ocr.synthesis.curated import digest


def fixture():
    terms = ["2 PALLETS X10KG NET EACH OF ", "3 CARTONS X5KG NET EACH OF "]
    before = "\n".join(terms)
    ledger = {
        "relation": "DISJOINT_GROUPS",
        "surfaces": {"held-numeric:0": [before]},
        "facts": {
            "declaredGroupLedgers": [
                {"publicQuantity": n, "surfaces": {"held-numeric:0": [term]}}
                for n, term in zip([2, 3], terms, strict=True)
            ]
        },
    }
    source = b"BEGIN OLD GOODS"
    raw = "BEGIN " + before + " GOODS"
    candidate = {
        "physical": {"freshPrivatePacking": ledger},
        "scenario": {"provenance": {"freshPrivatePacking": copy.deepcopy(ledger)}},
        "joinedRawText": raw,
        "joinedRawTextSha256": digest(raw.encode()),
        "target": {"quantity": 5, "unrelated": "unchanged"},
        "proof": {
            "sourceSha256": digest(source),
            "renderedSha256": digest(raw.encode()),
            "edits": [
                {
                    "key": "held-numeric:0",
                    "byteStart": 6,
                    "byteEnd": 9,
                    "before": "OLD",
                    "after": before,
                    "targetPaths": ["documentPatch.goodsItemDetails[0].netWeight.value"],
                }
            ],
        },
        "renderValues": {"held-numeric:0": before},
        "surfaceValues": {"held-numeric:0": [before]},
    }
    return candidate, {"regions": [{"role": "unit_net_prefix"}]}, ["2 PALLETS", "3 CARTONS"]


def test_disjoint_repair_preserves_facts_and_original_receipt():
    candidate, certificate, labels = fixture()
    before = copy.deepcopy(candidate)
    result, changes = repair(candidate, certificate, labels)
    assert candidate == before
    assert result["target"] == candidate["target"]
    assert result["joinedRawText"] == (
        "BEGIN 2 PALLETS X10KG NET EACH + 3 CARTONS X5KG NET EACH OF  GOODS"
    )
    assert changes["held-numeric:0"]["role"] == "unit_net_prefix"
    assert (
        result["physical"]["freshPrivatePacking"]["facts"]
        == candidate["physical"]["freshPrivatePacking"]["facts"]
    )


@pytest.mark.parametrize("kind", ["ledger", "count", "surface", "owner", "source"])
def test_disjoint_repair_rejects_changed_authorities(kind):
    candidate, certificate, labels = fixture()
    if kind == "ledger":
        candidate["scenario"]["provenance"]["freshPrivatePacking"]["relation"] = "CONTAINS"
    elif kind == "count":
        labels[0] = "3 PALLETS"
    elif kind == "surface":
        candidate["renderValues"]["held-numeric:0"] = "different"
    elif kind == "owner":
        candidate["proof"]["edits"][0]["targetPaths"].append("documentPatch.parties.shipper.name")
    else:
        candidate["proof"]["sourceSha256"] = "0" * 64
    with pytest.raises(ValueError):
        repair(candidate, certificate, labels)


def test_published_amendment_allows_only_exact_literal_description_delta():
    before = {"goodsItemDetails": [{"description": "2 BOXES OF 3 CARTONS OF GOODS"}]}
    after = {"goodsItemDetails": [{"description": "2 BOXES + 3 CARTONS OF GOODS"}]}
    before["negotiability"] = after["negotiability"] = "non_negotiable"
    expected = [
        [
            "/goodsItemDetails/0/description",
            before["goodsItemDetails"][0]["description"],
            after["goodsItemDetails"][0]["description"],
        ]
    ]
    checked_target_changes(before, after, expected)
    with pytest.raises(ValueError, match="explicit review"):
        checked_target_changes(before, after, [])
    with pytest.raises(ValueError, match="only change literal description"):
        checked_target_changes(
            {}, {"billOfLadingNumber": "OTHER"}, [["/billOfLadingNumber", "ABSENT", "OTHER"]]
        )
