import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from repair_r30_retained_description import repair

from document_ocr.synthesis.curated import digest


def fixture():
    before, after = "NEW GOODS\nname_component: PRIVATE", "NEW GOODS"
    raw = "PREFIX " + before + " SUFFIX"
    edit = {
        "key": "goods",
        "byteStart": 7,
        "byteEnd": 10,
        "before": "OLD",
        "after": before,
        "targetPaths": ["documentPatch.goodsItemDetails[0].description"],
    }
    candidate = {
        "joinedRawText": raw,
        "joinedRawTextSha256": digest(raw.encode()),
        "proof": {
            "sourceSha256": digest(b"PREFIX OLD SUFFIX"),
            "renderedSha256": digest(raw.encode()),
            "edits": [edit],
        },
        "target": {
            "documentPatch": {
                "goodsItemDetails": [
                    {"description": " ".join(before.upper().split()), "origin": {"country": "IRAN"}}
                ]
            }
        },
        "renderValues": {"goods": before},
        "surfaceValues": {"goods": before},
        "unrelated": "PRESERVED",
    }
    return candidate, {"key": "goods", "before": before, "after": after, "reason": "Reviewed tail"}


def test_reviewed_description_repair_preserves_source_and_unrelated_values():
    candidate, decision = fixture()
    original = copy.deepcopy(candidate)
    result = repair(candidate, decision)
    assert candidate == original
    assert result["joinedRawText"] == "PREFIX NEW GOODS SUFFIX"
    assert result["target"]["documentPatch"]["goodsItemDetails"][0] == {
        "description": "NEW GOODS",
        "origin": {"country": "IRAN"},
    }
    assert result["unrelated"] == "PRESERVED"


@pytest.mark.parametrize("failure", ["text", "owner", "target", "surface", "empty"])
def test_description_repair_rejects_unreviewed_authorities(failure):
    candidate, decision = fixture()
    if failure == "text":
        decision["before"] = "DIFFERENT"
    elif failure == "owner":
        candidate["proof"]["edits"][0]["targetPaths"].append("documentPatch.parties.shipper.name")
    elif failure == "target":
        candidate["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "OTHER"
    elif failure == "surface":
        candidate["surfaceValues"]["goods"] = "OTHER"
    else:
        decision["after"] = ""
    with pytest.raises(ValueError):
        repair(candidate, decision)
