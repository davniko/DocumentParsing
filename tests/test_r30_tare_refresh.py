import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import digest

scripts = Path(__file__).resolve().parents[1] / "scripts/migrations"
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location(
        "tare_refresh", scripts / "refresh_r30_retained_tare_total.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from propose_r30_retained_support import reviewed_feedback
    from publish_r30_sources import pin
finally:
    sys.path.pop(0)


def fixture():
    raw = "TARE 5660 KG"
    return {
        "surfaceValues": {"tare": ["5660"]},
        "physical": {"tares": [{"owners": [0, 1], "kg": "5660"}]},
        "joinedRawText": raw,
        "proof": {
            "edits": [],
            "sourceSha256": digest(raw.encode()),
            "renderedSha256": digest(raw.encode()),
        },
        "target": {"documentPatch": {"goodsItemDetails": [{"description": "UNCHANGED"}]}},
    }


def test_tare_refresh_is_source_only_and_exact():
    candidate = fixture()
    old = copy.deepcopy(candidate)
    slot = {"key": "tare", "byteStart": 5, "byteEnd": 9, "before": "5660", "targetPaths": []}
    result = module.replace_total(candidate, slot, "tare", "5660", "5900", {"review": "test"})
    assert candidate == old
    assert result["joinedRawText"] == "TARE 5900 KG"
    assert result["target"] == old["target"]
    assert result["physical"]["tares"][0]["kg"] == "5900"
    for value in ("5660", "NaN", "-1", "0"):
        with pytest.raises(ValueError):
            module.replace_total(candidate, slot, "tare", "5660", value, {})
    with pytest.raises(ValueError, match="source-only"):
        module.replace_total(
            candidate, {**slot, "targetPaths": ["/grossWeight"]}, "tare", "5660", "5900", {}
        )


def test_feedback_cannot_change_candidate_or_rewrite_previous_proposal(tmp_path, monkeypatch):
    import publish_r30_sources

    from document_ocr.atomic import atomic_publish_json

    monkeypatch.setattr(publish_r30_sources, "ROOT", tmp_path)
    candidate = {"path": "sample", "sha256": "a"}
    previous = tmp_path / "previous.json"
    atomic_publish_json(previous, {"candidate": candidate})
    feedback = {"previousProposal": pin(previous), "reason": "Recheck volume arithmetic."}
    assert reviewed_feedback(feedback, candidate) == feedback
    with pytest.raises(ValueError, match="exact same shipment"):
        reviewed_feedback(feedback, {**candidate, "sha256": "different"})
    with pytest.raises(ValueError):
        reviewed_feedback({**feedback, "reason": " "}, candidate)
