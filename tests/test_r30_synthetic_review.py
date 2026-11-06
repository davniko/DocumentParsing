import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
import publish_r30_sources
from review_r30_synthetic_batch import reviewed_records


@pytest.fixture(autouse=True)
def pin_root(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_r30_sources, "ROOT", tmp_path)


def fixture(tmp_path):
    path = tmp_path / "review.json"
    path.write_text("{}")
    entries = [
        {"id": str(i), "source": "doc_12345678", "rendered": {"sha256": str(i)}} for i in range(100)
    ]
    proposals = {
        e["id"]: [(path, {**e, "review": {"findings": []}, "invalidEvidence": []})] for e in entries
    }
    return {"selected": 100, "failures": [], "results": entries}, proposals


def test_exact_current_review_required(tmp_path):
    staged, proposals = fixture(tmp_path)
    proposals["0"][0][1]["rendered"] = {"sha256": "stale"}
    with pytest.raises(ValueError, match="exact-current"):
        reviewed_records(staged, proposals, {})


@pytest.mark.parametrize("field,value", [("invalidEvidence", [1]), ("review", {"findings": [1]})])
def test_every_finding_requires_main_agent_decision(tmp_path, field, value):
    staged, proposals = fixture(tmp_path)
    proposals["0"][0][1][field] = value
    with pytest.raises(ValueError, match="finding disposition"):
        reviewed_records(staged, proposals, {})
    assert len(reviewed_records(staged, proposals, {"0": "Reviewed exact evidence."})) == 100


def test_other_source_rejected(tmp_path):
    staged, proposals = fixture(tmp_path)
    proposals["0"][0][1]["source"] = "doc_another"
    with pytest.raises(ValueError, match="another source"):
        reviewed_records(staged, proposals, {})


def test_revised_decision_creates_immutable_independent_receipt(tmp_path, monkeypatch):
    import review_r30_synthetic_batch as module

    out = tmp_path / "out"
    (out / "batches").mkdir(parents=True)
    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "ART", tmp_path / "art")
    staged = {"results": []}
    (out / "batches/001-staging.json").write_text(json.dumps(staged))
    monkeypatch.setattr(module, "verify_pin", lambda p: None)
    # No rendering data is needed for this publication-address regression.
    monkeypatch.setattr(module, "reviewed_records", lambda s, p, d: [])
    decisions = tmp_path / "working-review.json"
    decisions.write_text(json.dumps({"decisions": {"x": "first reviewed explanation"}}))
    module.main(1, decisions)
    first = next((out / "batches").glob("001-final-review-*.json"))
    before = first.read_bytes()
    decisions.write_text(json.dumps({"decisions": {"x": "corrected reviewed explanation"}}))
    module.main(1, decisions)
    module.main(1, decisions)
    receipts = list((out / "batches").glob("001-final-review-*.json"))
    assert len(receipts) == 2 and first.read_bytes() == before
    for path in receipts:
        receipt = json.loads(path.read_text())
        assert publish_r30_sources.verify_pin(receipt["adjudications"]).exists()
