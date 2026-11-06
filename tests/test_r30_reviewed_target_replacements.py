import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
import r30_reviewed_target_replacements as module


def fixture(monkeypatch, tmp_path):
    row = {
        "documentId": "sample",
        "sourceDocumentId": "source",
        "target": {"parties": {"shipper": {"name": "A A"}}},
    }
    replacement = copy.deepcopy(row)
    replacement["target"]["parties"]["shipper"]["name"] = "A"
    before = {**row, "pages": [{"text": "A A"}], "proof": ["unchanged"]}
    after = {**copy.deepcopy(replacement), "pages": before["pages"], "proof": before["proof"]}
    receipt = {
        "currentRecordSha256": "baseline",
        "candidate": {"sha256": "candidate"},
        "replacementRecord": {"sha256": "replacement"},
        "targetCorrectionReview": "review",
    }
    review = {
        "status": "MAIN_AGENT_REVIEWED_TARGET_ONLY_REPLACEMENT",
        "documentId": "sample",
        "sourceDocumentId": "source",
        "baselineSha256": "baseline",
        "candidate": receipt["candidate"],
        "replacementRecord": receipt["replacementRecord"],
        "reason": "Reviewed duplicated name; complete native text retained.",
        "beforeRendered": "before",
        "afterRendered": "after",
        "paths": ["parties.shipper.name"],
    }
    files = {"review": review, "before": before, "after": after}

    def pin(value):
        path = tmp_path / f"{value}.json"
        path.write_text(json.dumps(files[value]))
        return path

    monkeypatch.setattr(module, "verify_pin", pin)
    return row, replacement, receipt, files


def test_explicit_exact_native_correction_passes(monkeypatch, tmp_path):
    row, replacement, receipt, _ = fixture(monkeypatch, tmp_path)
    module.verify_target_replacement(row, replacement, receipt, {"parties.shipper.name"})


@pytest.mark.parametrize(
    "failure",
    [
        "missing_review",
        "candidate",
        "baseline",
        "identity",
        "native_identity",
        "geometry",
        "unreviewed",
        "replacement",
    ],
)
def test_unreviewed_or_stale_corrections_fail(monkeypatch, tmp_path, failure):
    row, replacement, receipt, files = fixture(monkeypatch, tmp_path)
    if failure == "missing_review":
        del receipt["targetCorrectionReview"]
    elif failure == "candidate":
        receipt["candidate"] = {"sha256": "changed"}
    elif failure == "baseline":
        receipt["currentRecordSha256"] = "changed"
    elif failure == "identity":
        files["review"]["sourceDocumentId"] = "other"
    elif failure == "native_identity":
        files["after"]["documentId"] = "other"
    elif failure == "geometry":
        files["after"]["pages"] = [{"text": "A A", "bbox": [0, 0, 1, 1]}]
    elif failure == "unreviewed":
        files["review"]["paths"] = []
    else:
        replacement["target"]["parties"]["shipper"]["name"] = "B"
    with pytest.raises(ValueError):
        module.verify_target_replacement(row, replacement, receipt, {"parties.shipper.name"})
