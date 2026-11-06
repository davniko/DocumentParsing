import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from admit_r30_missing_wording import reviewed_proposal


def test_review_is_exact_and_does_not_mutate_proposal():
    spec = {
        "documentId": "x",
        "candidateSha256": "a",
        "sourceFingerprint": "b",
        "requested": {"name": "name"},
    }
    proposal = {**spec, "values": {"name": "OLD"}}
    decision = {
        "reason": "Exact readback",
        "corrections": {"name": {"before": "OLD", "after": "NEW"}},
    }
    assert reviewed_proposal(spec, proposal, decision)["values"] == {"name": "NEW"}
    assert proposal["values"] == {"name": "OLD"}
    with pytest.raises(ValueError, match="baseline"):
        reviewed_proposal(spec, {**proposal, "values": {"name": "OTHER"}}, decision)
    with pytest.raises(ValueError, match="context"):
        reviewed_proposal(spec, {**proposal, "candidateSha256": "changed"}, decision)
    with pytest.raises(ValueError, match="membership"):
        reviewed_proposal(spec, {**proposal, "values": {}}, decision)


def test_additional_cohort_keeps_earlier_immutable_review(tmp_path, monkeypatch):
    import admit_r30_missing_wording as module

    out, art = tmp_path / "out", tmp_path / "art"
    (out / "batches").mkdir(parents=True)
    (art / "wording-proposals").mkdir(parents=True)
    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "ART", art)
    monkeypatch.setattr(module, "pin", lambda p: {"path": str(p), "sha256": p.read_text()})
    historical = out / "batches/001-wording-review.json"
    historical.write_text('{"historical":true}')
    for identity in ("first", "later"):
        spec = {
            "documentId": identity,
            "candidateSha256": "a",
            "sourceFingerprint": "b",
            "requested": {"name": "short name"},
        }
        (out / "batches/001-staging.json").write_text(json.dumps({"newWording": [spec]}))
        (art / "wording-proposals" / (identity + ".json")).write_text(
            json.dumps({**spec, "values": {"name": identity}})
        )
        review = tmp_path / (identity + "-review.json")
        review.write_text(json.dumps({"decisions": {identity: {"reason": "Reviewed exact owner"}}}))
        module.main(1, review)
    assert historical.read_text() == '{"historical":true}'
    assert len(list((out / "batches").glob("001-wording-review-*.json"))) == 2
    assert len(list((art / "wording").glob("*.json"))) == 2
