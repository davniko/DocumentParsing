import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
import r30_finding_dispositions as module


@pytest.fixture
def fixture(monkeypatch):
    pin = {"path": "review", "sha256": "sha"}
    entry = {"id": "record", "source": "source", "rendered": {"path": "render", "sha256": "render-sha"}}
    review = {**entry, "status": "PROPOSED_REQUIRES_MAIN_AGENT_ADJUDICATION",
              "review": {"findings": [{"path": "one"}, {"path": "two"}]},
              "invalidEvidence": [{"finding": 1}]}
    rows = [{"id": entry["id"], "rendered": entry["rendered"], "review": pin,
             "finding": i, "decision": f"Exact finding {i} reviewed"} for i in range(2)]
    document = {"status": "MAIN_AGENT_REVIEWED_IDENTIFIED_FINDINGS", "records": rows}
    monkeypatch.setattr(module, "verify_pin", lambda p: p["path"])
    monkeypatch.setattr(module, "read", lambda p: {"ledger": document, "review": review}[p])
    return entry, review, document, [{"pin": pin}]


def test_requires_every_finding_including_invalid_evidence(fixture):
    entry, _, document, items = fixture
    dispositions = module.load_dispositions("ledger")
    assert "finding 1" in module.complete_decision(entry, items, dispositions)
    document["records"].pop()
    assert module.complete_decision(entry, items, module.load_dispositions("ledger")) is None


@pytest.mark.parametrize("field,value", [("id", "wrong"), ("rendered", {}), ("finding", -1),
                                        ("finding", True), ("finding", 2), ("decision", "")])
def test_bad_individual_decision_is_rejected(fixture, field, value):
    _, _, document, _ = fixture
    document["records"][0][field] = value
    with pytest.raises(ValueError):
        module.load_dispositions("ledger")


def test_duplicate_disposition_is_rejected(fixture):
    _, _, document, _ = fixture
    document["records"].append(copy.deepcopy(document["records"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        module.load_dispositions("ledger")


def test_changed_review_or_render_cannot_reuse_a_decision(fixture):
    entry, review, _, items = fixture
    dispositions = module.load_dispositions("ledger")
    entry = {**entry, "rendered": {"path": "new", "sha256": "new"}}
    with pytest.raises(ValueError, match="exact-current"):
        module.complete_decision(entry, items, dispositions)


def test_clean_review_needs_no_disposition(fixture):
    entry, review, _, items = fixture
    review["review"]["findings"] = []
    review["invalidEvidence"] = []
    assert module.complete_decision(entry, items, {}) == "No findings; exact-current review verified."


def test_invalid_evidence_without_finding_is_rejected(fixture):
    entry, review, _, items = fixture
    review["invalidEvidence"] = [{"finding": 9}]
    with pytest.raises(ValueError, match="absent finding"):
        module.complete_decision(entry, items, {})
