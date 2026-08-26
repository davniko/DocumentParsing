from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from document_ocr.label_schemas.semantic_review import (
    BillOfLadingIndependentReview,
)


def _review_payload() -> dict[str, object]:
    return {
        "reviewSchemaVersion": "1.0.0",
        "taskType": "bill_of_lading_kie_semantic_review",
        "documentId": f"doc_{'1' * 64}",
        "candidateAttempt": 1,
        "reviewNumber": 1,
        "workItemSha256": "2" * 64,
        "candidateSha256": "3" * 64,
        "reviewerModel": "gpt-5.6-luna",
        "reviewerReasoningEffort": "max",
        "result": "pass",
        "checks": {
            "rawOcrTruthBoundary": "pass",
            "evidenceIntegrity": "pass",
            "semanticCompleteness": "pass",
            "semanticCorrectness": "pass",
            "contaminationAndRedundancy": "pass",
            "documentUnitAndRelationships": "pass",
        },
        "findings": [],
        "summary": "Complete, correct, and OCR-grounded.",
    }


def test_passing_review_requires_all_checks_to_pass() -> None:
    payload = _review_payload()
    payload["checks"] = dict(payload["checks"], semanticCompleteness="fail")

    with pytest.raises(ValidationError, match="every semantic check to pass"):
        BillOfLadingIndependentReview.model_validate_json(json.dumps(payload), strict=True)


def test_failing_review_requires_ocr_grounded_blocking_finding() -> None:
    payload = _review_payload()
    payload["result"] = "fail"
    payload["checks"] = dict(payload["checks"], semanticCompleteness="fail")

    with pytest.raises(ValidationError, match="requires a blocking finding"):
        BillOfLadingIndependentReview.model_validate_json(json.dumps(payload), strict=True)

    payload["findings"] = [
        {
            "severity": "blocking",
            "category": "missing_field",
            "message": "Explicit delivery agent is absent from the candidate.",
            "targetPaths": ["documentPatch.parties.deliveryAgent.name"],
            "rawOcrEvidence": [
                {
                    "pageNumber": 1,
                    "rawValue": "DELIVERY AGENT",
                    "ocrExcerpt": "DELIVERY AGENT: EXAMPLE SHIPPING",
                }
            ],
            "imageUse": "not_used",
        }
    ]

    review = BillOfLadingIndependentReview.model_validate_json(json.dumps(payload), strict=True)

    assert review.result == "fail"
    assert review.findings[0].severity == "blocking"
