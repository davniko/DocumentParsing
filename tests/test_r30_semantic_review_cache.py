import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from r30_semantic_review_cache import rebind_review, rebound_review_name


def fixtures():
    render = {"documentId": "one", "sourceDocumentId": "source", "pages": [
        {"regions": [{"text": "A", "box": [0, 1, 2, 3]}]}], "target": {"name": "A"},
        "proof": ["old"]}
    review = {"status": "PROPOSED_REQUIRES_MAIN_AGENT_ADJUDICATION", "id": "one",
              "source": "source", "requestSha256": "request", "rendered": {"sha256": "old"},
              "review": {"findings": [{"reason": "Must still be adjudicated"}]},
              "invalidEvidence": [1]}
    return render, review


def test_provenance_only_reuse_preserves_findings_and_previous_receipt():
    before, review = fixtures()
    after = {**before, "proof": ["new"]}
    old = copy.deepcopy(review)
    result = rebind_review(review, before, after, {"sha256": "new"}, "request", {"path": "previous"})
    assert review == old
    assert result["rendered"] == {"sha256": "new"}
    assert result["review"] == review["review"] and result["invalidEvidence"] == [1]
    assert result["unchangedSemanticInputReuse"] == {"previousReview": {"path": "previous"}}


@pytest.mark.parametrize("field", ["documentId", "sourceDocumentId", "target", "pages"])
def test_changed_input_rejected(field):
    before, review = fixtures()
    after = {**before, field: "changed"}
    with pytest.raises(ValueError, match="input changed"):
        rebind_review(review, before, after, {}, "request", {})


@pytest.mark.parametrize("field", ["status", "requestSha256", "id", "source"])
def test_changed_request_or_identity_rejected(field):
    render, review = fixtures()
    review[field] = "changed"
    with pytest.raises(ValueError):
        rebind_review(review, render, render, {}, "request", {})


def test_rebound_artifact_identity_preserves_distinct_provenance(tmp_path):
    from document_ocr.atomic import atomic_publish_json

    render, review = fixtures()
    first = rebind_review(review, render, render, {"sha256": "new"}, "request", {"path": "a"})
    second = rebind_review(review, render, render, {"sha256": "new"}, "request", {"path": "b"})
    assert first["review"] == second["review"]
    assert rebound_review_name(first) != rebound_review_name(second)
    for item in (first, second, first):
        atomic_publish_json(tmp_path / rebound_review_name(item), item)
    assert len(list(tmp_path.iterdir())) == 2
    assert rebound_review_name(first) == rebound_review_name(dict(reversed(list(first.items()))))
