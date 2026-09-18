from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler.production_catalog import (
    ProductionTemplateCatalogConfig,
    load_production_catalog_config,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_CATALOG_CONFIG = (
    _PROJECT_ROOT / "configs/synthesis/production/mpci_bl_production_template_catalog1344_v4.yaml"
)
_FINAL_PRODUCTION_CATALOG_CONFIG = (
    _PROJECT_ROOT / "configs/synthesis/production/mpci_bl_production_template_catalog1510_v5.yaml"
)


def _pin(character: str, path: str) -> dict[str, str]:
    return {
        "path": path,
        "commit_sha256": character * 64,
        "transaction_sha256": chr(ord(character) + 1) * 64,
    }


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "bill_of_lading_production_template_catalog_v1",
        "run_name": "production-catalog",
        "output_dir": "artifacts/production",
        "shards": [
            {
                "catalog_run": _pin("a", "artifacts/catalog-a"),
                "resolution_run": _pin("c", "artifacts/resolution-a"),
                "outcomes_file": "outcomes.jsonl",
                "resolution_scope": "review_subset",
                "expected_documents": 10,
                "expected_certified": 9,
                "expected_rejected": 0,
                "expected_review_required": 1,
            },
            {
                "catalog_run": _pin("e", "artifacts/catalog-b"),
                "resolution_run": _pin("1", "artifacts/resolution-b"),
                "outcomes_file": "results.jsonl",
                "resolution_scope": "complete_source",
                "expected_documents": 20,
                "expected_certified": 17,
                "expected_rejected": 2,
                "expected_review_required": 1,
            },
        ],
        "expected_source_documents": 30,
        "expected_usable_templates": 26,
        "expected_excluded_documents": 4,
        "require_current_template_schema": 6,
        "require_zero_unresolved_reviews": True,
        "require_disjoint_shards": True,
    }


def test_production_catalog_contract_accepts_closed_partition() -> None:
    config = ProductionTemplateCatalogConfig.model_validate_json(json.dumps(_config()))

    assert config.expected_source_documents == 30
    assert config.expected_usable_templates == 26
    assert len(config.shards) == 2


def test_production_catalog_contract_rejects_unaccounted_documents() -> None:
    payload = _config()
    payload["expected_excluded_documents"] = 3

    with pytest.raises(ValidationError, match="must cover every source document"):
        ProductionTemplateCatalogConfig.model_validate_json(json.dumps(payload))


def test_production_catalog_contract_rejects_review_subset_with_rejections() -> None:
    payload = _config()
    first = payload["shards"][0]
    first["expected_certified"] = 8
    first["expected_rejected"] = 1

    with pytest.raises(ValidationError, match="cannot leave compiler rejections unresolved"):
        ProductionTemplateCatalogConfig.model_validate_json(json.dumps(payload))


def test_production_catalog_contract_rejects_duplicate_catalog_runs() -> None:
    payload = _config()
    payload["shards"][1]["catalog_run"] = payload["shards"][0]["catalog_run"]

    with pytest.raises(ValidationError, match="catalog shard runs must be unique"):
        ProductionTemplateCatalogConfig.model_validate_json(json.dumps(payload))


def test_production_catalog1344_config_pins_closed_adjudicated_population() -> None:
    config = load_production_catalog_config(_PRODUCTION_CATALOG_CONFIG)

    assert config.run_name == "mpci-bl-production-template-catalog1344-v4"
    assert len(config.shards) == 5
    assert config.expected_source_documents == 1460
    assert config.expected_usable_templates == 1344
    assert config.expected_excluded_documents == 116
    assert sum(row.expected_certified for row in config.shards) == 1344
    assert sum(row.expected_rejected for row in config.shards) == 74
    assert sum(row.expected_review_required for row in config.shards) == 42
    assert config.require_current_template_schema == 6
    assert config.require_zero_unresolved_reviews is True
    assert config.require_disjoint_shards is True


def test_production_catalog1510_config_pins_complete_eligible_population() -> None:
    config = load_production_catalog_config(_FINAL_PRODUCTION_CATALOG_CONFIG)

    assert config.run_name == "mpci-bl-production-template-catalog1510-v5"
    assert len(config.shards) == 6
    assert config.expected_source_documents == 1650
    assert config.expected_usable_templates == 1510
    assert config.expected_excluded_documents == 140
    assert sum(row.expected_certified for row in config.shards) == 1510
    assert sum(row.expected_rejected for row in config.shards) == 85
    assert sum(row.expected_review_required for row in config.shards) == 55
    assert config.require_current_template_schema == 6
    assert config.require_zero_unresolved_reviews is True
    assert config.require_disjoint_shards is True
