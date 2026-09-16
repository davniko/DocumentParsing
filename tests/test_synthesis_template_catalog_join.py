from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler.catalog_join import TemplateCatalogJoinConfig


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "bill_of_lading_compiled_template_catalog_join_v1",
        "run_name": "joined-templates",
        "output_dir": "artifacts/kie-synthesis",
        "sources": (
            {
                "path": "artifacts/run-a",
                "commit_sha256": "a" * 64,
                "transaction_sha256": "b" * 64,
            },
            {
                "path": "artifacts/run-b",
                "commit_sha256": "c" * 64,
                "transaction_sha256": "d" * 64,
            },
        ),
        "expected_source_outcomes": 230,
        "expected_certified_templates": 226,
        "expected_review_required": 4,
        "require_no_rejected_outcomes": True,
    }


def test_join_contract_accepts_exact_certified_and_review_partition() -> None:
    config = TemplateCatalogJoinConfig.model_validate_json(json.dumps(_config()))

    assert config.expected_source_outcomes == 230
    assert config.expected_certified_templates == 226


def test_join_contract_rejects_duplicate_sources() -> None:
    payload = _config()
    sources = cast(tuple[object, ...], payload["sources"])
    payload["sources"] = (sources[0], sources[0])

    with pytest.raises(ValidationError, match="sources must be unique"):
        TemplateCatalogJoinConfig.model_validate_json(json.dumps(payload))


def test_join_contract_rejects_unaccounted_outcomes() -> None:
    payload = _config()
    payload["expected_review_required"] = 3

    with pytest.raises(ValidationError, match="must cover every expected outcome"):
        TemplateCatalogJoinConfig.model_validate_json(json.dumps(payload))
