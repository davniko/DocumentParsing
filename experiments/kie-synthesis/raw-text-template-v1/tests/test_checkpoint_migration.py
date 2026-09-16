from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from raw_text_template_experiment.checkpoint_migration import (
    LegacyExtractionStateCheckpointV1,
    _translate_legacy_checkpoint,
)


def _legacy_checkpoint_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "document_id": "doc_fixture",
        "source_sha256": "a" * 64,
        "source_label_sha256": "b" * 64,
        "drafts": [
            {
                "draft_id": "literal_fixture",
                "logical_key": "literal:fixture",
                "render_mode": "literal_static",
                "value_kind": "legal_text",
                "group_kind": "legal",
                "group_key": "legal:0",
                "target_paths": [],
                "derivation": None,
                "dependency_paths": [],
                "dependency_bindings": [],
                "char_start": 0,
                "char_end": 7,
                "source_text": "FIXTURE",
                "evidence_origin": "host_verified_agent_proposal",
                "render_policy": "exact_surface",
                "rationale": "Migration fixture.",
            }
        ],
        "carrier_assessment": {
            "canonical_name": "TEST CARRIER",
            "aliases": [],
            "evidence_occurrences": [
                {
                    "line_start": "L00001",
                    "line_end": "L00001",
                    "source_text": "TEST CARRIER",
                    "occurrence_index": 0,
                }
            ],
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Migration fixture.",
        },
        "semantic_only_target_facts": [],
        "applied_critic_revisions": [],
    }


def test_legacy_checkpoint_translation_is_explicit_and_current_schema() -> None:
    payload = _legacy_checkpoint_payload()
    legacy = LegacyExtractionStateCheckpointV1.model_validate_json(json.dumps(payload), strict=True)

    current = _translate_legacy_checkpoint(legacy)

    assert current.schema_version == 2
    assert current.document_id == "doc_fixture"
    assert current.coherence_constraints == ()
    assert current.drafts == legacy.drafts


def test_legacy_checkpoint_translation_rejects_unrecognized_fields() -> None:
    payload = _legacy_checkpoint_payload()
    payload["coherence_constraints"] = []

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LegacyExtractionStateCheckpointV1.model_validate_json(json.dumps(payload), strict=True)
