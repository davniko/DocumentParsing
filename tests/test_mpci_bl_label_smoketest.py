from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/mpci_bl_label_smoketest.py"
_SPEC = importlib.util.spec_from_file_location("mpci_bl_label_smoketest", _TOOL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to import the smoke-test runner")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
parse_semantic_review = _MODULE.parse_semantic_review
target_path_is_within_group = _MODULE.target_path_is_within_group


def _document_id(character: str) -> str:
    return f"doc_{character * 64}"


def _passing_review(document_ids: list[str]) -> dict[str, object]:
    results = [
        {
            "documentId": document_id,
            "result": "pass",
            "checks": {
                "rawOcrSupport": True,
                "mappingSemantics": True,
                "relationshipsAndOrder": True,
                "ambiguityHandling": True,
            },
        }
        for document_id in document_ids
    ]
    return {
        "status": "passed",
        "reviewedDocumentIds": document_ids,
        "results": results,
        "needsReview": [],
        "aggregate": {"reviewed": len(document_ids), "pass": len(document_ids), "needsReview": 0},
    }


def test_parse_semantic_review_requires_exact_complete_passing_decisions() -> None:
    document_ids = [_document_id("a"), _document_id("b")]

    review, needs_review = parse_semantic_review(_passing_review(document_ids), document_ids)

    assert review["status"] == "passed"
    assert review["reviewedDocumentIds"] == document_ids
    assert needs_review == set()


def test_parse_semantic_review_rejects_passing_result_with_failed_check() -> None:
    document_ids = [_document_id("a"), _document_id("b")]
    review = _passing_review(document_ids)
    results = review["results"]
    assert isinstance(results, list)
    first = results[0]
    assert isinstance(first, dict)
    checks = first["checks"]
    assert isinstance(checks, dict)
    checks["mappingSemantics"] = False

    with pytest.raises(RuntimeError, match="failed check"):
        parse_semantic_review(review, document_ids)


def test_target_path_group_match_includes_concrete_array_indexes() -> None:
    prefix = "documentPatch.containerInformation"

    assert target_path_is_within_group(
        "documentPatch.containerInformation[0].equipmentIdentification.equipmentIdentifier", prefix
    )
    assert target_path_is_within_group("documentPatch.containerInformation", prefix)
    assert not target_path_is_within_group("documentPatch.containerInformations", prefix)
