from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler.production_synthesis import (
    ProductionSynthesisPlanConfig,
    TemplateInventoryRow,
    _even_repetitions,
    _sample_id,
    load_production_synthesis_plan_config,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXACT_CONFIG = (
    _PROJECT_ROOT
    / "configs/synthesis/production/mpci_bl_production_synthesis10000_exact_validation_id_v1.yaml"
)
_PROXY_CONFIG = (
    _PROJECT_ROOT
    / "configs/synthesis/production/mpci_bl_production_synthesis10000_layout_proxy_holdout_v1.yaml"
)


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "bill_of_lading_compiled_template_synthesis_plan_v1",
        "run_name": "plan",
        "output_dir": "artifacts/plan",
        "template_run": {
            "path": "artifacts/templates",
            "commit_sha256": "a" * 64,
            "transaction_sha256": "b" * 64,
        },
        "validation_partition": {
            "path": "artifacts/training/report.json",
            "sha256": "c" * 64,
            "report_format": "t5gemma2_runtime_partition_dataset_report_v1",
            "split": "validation",
            "expected_documents": 2,
            "document_ids_sha256": "d" * 64,
        },
        "target_task": "bill_of_lading_relation_explicit_v5",
        "target_schema_version": "5.0.0-experimental",
        "target_generation": "controlled_latest_schema_from_source_v1",
        "selection": {
            "documents": 10,
            "seed": 17,
            "sample_namespace": "test",
            "exclusion_mode": "exact_document_id",
            "capability_counts": {
                "standard": 7,
                "dangerous_goods": 1,
                "temperature_controlled": 2,
            },
            "balancing": "even_reuse_within_capability_hmac_rank_v1",
            "require_disjoint_capability_pools": True,
            "require_every_eligible_template_selected": False,
        },
        "expected_inventory": {
            "validation_documents_in_catalog": 1,
            "validation_template_proxy_families": 1,
            "latest_schema_incompatible_templates": 0,
            "excluded_templates": 1,
            "eligible_templates": 3,
            "eligible_standard_templates": 1,
            "eligible_dangerous_goods_templates": 1,
            "eligible_temperature_controlled_templates": 1,
        },
    }


def _inventory(document_id: str) -> TemplateInventoryRow:
    return TemplateInventoryRow(
        source_document_id=document_id,
        template_proxy_id="proxy",
        carrier="Carrier",
        carrier_family="CARRIER",
        document_type="bill_of_lading",
        pages=1,
        lines=10,
        characters=100,
        bindings=5,
        occurrences=5,
        deterministic_bindings=5,
        agent_residual_bindings=0,
        container_count=1,
        cargo_group_count=1,
        cargo_package_count=1,
        allocation_group_count=1,
        dangerous_goods=False,
        temperature_controlled=False,
        cohort="standard",
        latest_schema_compatible=True,
        latest_schema_incompatibility=None,
    )


def test_plan_contract_requires_capability_counts_to_cover_documents() -> None:
    payload = _config()
    payload["selection"]["capability_counts"]["standard"] = 6

    with pytest.raises(ValidationError, match=r"must sum to selection\.documents"):
        ProductionSynthesisPlanConfig.model_validate_json(json.dumps(payload))


def test_plan_contract_accepts_arbitrary_reuse_count() -> None:
    payload = _config()
    payload["selection"]["documents"] = 1_000_001
    payload["selection"]["capability_counts"] = {
        "standard": 999_998,
        "dangerous_goods": 1,
        "temperature_controlled": 2,
    }
    config = ProductionSynthesisPlanConfig.model_validate_json(json.dumps(payload))

    assert config.selection.documents == 1_000_001
    assert config.target_schema_version == "5.0.0-experimental"


def test_even_reuse_assigns_every_requested_variant_without_a_one_template_cap() -> None:
    repetitions = _even_repetitions(
        templates=(_inventory("doc_a"), _inventory("doc_b"), _inventory("doc_c")),
        requested=10,
        namespace="test",
        seed=17,
        cohort="standard",
    )

    counts = sorted(count for _template, count in repetitions)
    assert counts == [3, 3, 4]
    assert sum(counts) == 10


def test_variant_identity_is_unique_and_repeatable() -> None:
    identities = tuple(
        _sample_id(
            namespace="test",
            seed=17,
            source_document_id="doc_a",
            variant_index=index,
        )
        for index in range(10_000)
    )

    assert len(set(identities)) == 10_000
    assert identities[123] == _sample_id(
        namespace="test",
        seed=17,
        source_document_id="doc_a",
        variant_index=123,
    )


def test_primary_and_transfer_configs_pin_v5_and_distinct_exclusion_modes() -> None:
    exact = load_production_synthesis_plan_config(_EXACT_CONFIG)
    proxy = load_production_synthesis_plan_config(_PROXY_CONFIG)

    assert exact.selection.documents == proxy.selection.documents == 10_000
    assert exact.target_task == proxy.target_task == "bill_of_lading_relation_explicit_v5"
    assert exact.target_schema_version == proxy.target_schema_version == "5.0.0-experimental"
    assert exact.selection.exclusion_mode == "exact_document_id"
    assert proxy.selection.exclusion_mode == "template_proxy_family"
    assert exact.expected_inventory.eligible_templates == 1_441
    assert proxy.expected_inventory.eligible_templates == 985
    assert exact.selection.capability_counts.dangerous_goods == 1_000
    assert exact.selection.capability_counts.temperature_controlled == 1_500
