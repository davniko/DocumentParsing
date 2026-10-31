import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/synthesis/audit_r28_multirow_wording.py"
    spec = importlib.util.spec_from_file_location("wording_audit", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_review_cannot_omit_duplicate_or_substitute_records():
    m = module()
    items = [{"sample_id": "a", "description": "COTTON"}]
    for rows in (
        [],
        [{"sample_id": "b", "findings": []}],
        [
            {"sample_id": "a", "findings": []},
            {"sample_id": "a", "findings": []},
        ],
    ):
        with pytest.raises(ValueError, match="omitted, duplicated or substituted"):
            m.validate_reviews(m.Reviews(records=rows), items)
    m.validate_reviews(m.Reviews(records=[{"sample_id": "a", "findings": []}]), items)


def test_evidence_must_be_literal_not_a_summary_or_invented_text():
    m = module()
    items = [{"sample_id": "a", "description": "COTTON IN PAPER BAGS"}]
    finding = {"kind": "material", "evidence": "STEEL DRUMS", "reason": "Unsupported material"}
    answer = m.Reviews(records=[{"sample_id": "a", "findings": [finding]}])
    with pytest.raises(ValueError, match="not literal"):
        m.validate_reviews(answer, items)
    answer.records[0].findings[0].evidence = "IN PAPER BAGS"
    m.validate_reviews(answer, items)


def test_native_schema_requires_all_records_and_literal_evidence_before_retry():
    m = module()
    model = m.response_model(
        [
            {"sample_id": "a", "description": "COTTON"},
            {"sample_id": "b", "description": "PAPER BAGS"},
        ]
    )
    assert set(model.model_json_schema()["required"]) == {"s0", "s1"}
    with pytest.raises(ValidationError):
        model.model_validate({"s0": {"findings": []}})
    with pytest.raises(ValidationError, match="not literal"):
        model.model_validate(
            {
                "s0": {
                    "findings": [{"kind": "material", "evidence": "PAPER", "reason": "wrong quote"}]
                },
                "s1": {"findings": []},
            }
        )
    model.model_validate({"s0": {"findings": []}, "s1": {"findings": []}})
