"""Certification requires exact native, geometry and main-review evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synthesis/certify_package_alias_rollout.py"
spec = importlib.util.spec_from_file_location("alias_certification", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, monkeypatch, *, findings=False):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    sid = "doc_12345678"
    write(tmp_path / "inspection/inventory.json", {"config": {"source_ids": [sid]}})
    candidate = write(
        tmp_path / "campaign/candidates/sample.json",
        {"sourceDocumentId": sid, "documentId": "sample", "target": {"good": 1}},
    )
    positioned = write(
        tmp_path / "positions.json",
        {"candidateSha256": sha256_file(candidate), "target": {"good": 1}},
    )
    write(
        tmp_path / "campaign/reviews" / f"{sid}.json",
        {
            "candidateHashes": {"sample": digest(json.loads(candidate.read_text()))},
            "review": {"findings": [{"field": "example"}] if findings else []},
        },
    )
    replay = write(
        tmp_path / "replay.json",
        {
            "rows": [
                {
                    "sourceId": sid,
                    "status": "PASS",
                    "candidate": str(candidate),
                    "positions": str(positioned),
                    "aliasChanges": 1,
                }
            ]
        },
    )
    controls = write(tmp_path / "controls.json", {"counts": {"PASS": 3}})
    decisions = write(tmp_path / "decisions.json", {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--work",
            str(tmp_path),
            "--replay",
            str(replay),
            "--controls",
            str(controls),
            "--decisions",
            str(decisions),
        ],
    )
    return replay, positioned, controls, decisions


def test_complete_evidence_certifies_but_does_not_promote(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    module.main()
    result = json.loads((tmp_path / "acceptance.json").read_text())
    assert result["sources"] == 1
    assert result["batches"][0]["status"] == "PASS"
    assert not (tmp_path.parent / "package-presentations-v1").exists()


@pytest.mark.parametrize("defect", ["missing_source", "held", "wrong_target", "missing_control"])
def test_incomplete_or_mismatched_evidence_never_certifies(tmp_path, monkeypatch, defect):
    replay, positioned, controls, _ = fixture(tmp_path, monkeypatch)
    path = (
        positioned
        if defect == "wrong_target"
        else controls
        if defect == "missing_control"
        else replay
    )
    data = json.loads(path.read_text())
    if defect == "missing_source":
        data["rows"] = []
    elif defect == "held":
        data["rows"][0]["status"] = "HELD"
    elif defect == "wrong_target":
        data["target"] = {"good": 2}
    else:
        data["counts"]["PASS"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.main()
    assert not (tmp_path / "acceptance.json").exists()


def test_native_findings_require_explicit_main_review(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch, findings=True)
    with pytest.raises(ValueError, match="main-agent finding decision missing"):
        module.main()


def test_review_requires_candidate_object_digest_not_an_unrelated_hash(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    path = tmp_path / "campaign/reviews/doc_12345678.json"
    review = json.loads(path.read_text())
    review["candidateHashes"]["sample"] = "0" * 64
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="semantic review is stale"):
        module.main()
