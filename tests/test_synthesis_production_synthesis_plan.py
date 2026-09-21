from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler.production_synthesis import (
    ProductionSynthesisPlanConfig,
    TemplateInventoryRow,
    _even_repetitions,
    _sample_id,
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
        "target_generation": "complete_latest_schema_targets_v1",
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


def test_plan_cannot_request_the_incomplete_source_copy_generator() -> None:
    payload = _config()
    payload["target_generation"] = "controlled_latest_schema_from_source_v1"
    with pytest.raises(ValidationError, match="complete_latest_schema_targets_v1"):
        ProductionSynthesisPlanConfig.model_validate_json(json.dumps(payload))


def test_review_exclusion_happens_before_sampling_without_reducing_requested_count(
    monkeypatch, tmp_path
):
    from dataclasses import replace

    from document_ocr.synthesis.template_compiler import production_synthesis as planning

    inventory = (
        _inventory("validation"),
        _inventory("safe"),
        _inventory("review"),
        replace(_inventory("dg"), dangerous_goods=True, cohort="dangerous_goods"),
        replace(
            _inventory("thermal"), temperature_controlled=True, cohort="temperature_controlled"
        ),
    )
    monkeypatch.setattr(planning, "_validate_committed_run", lambda *args: tmp_path)
    monkeypatch.setattr(planning, "_validation_document_ids", lambda **kwargs: ("validation",))
    monkeypatch.setattr(planning, "_template_inventory", lambda *args: inventory)
    payload = _config()
    payload["source_review_exclusions"] = {
        "review": "Source container totals contradict printed rows"
    }
    payload["expected_inventory"]["excluded_templates"] = 2
    config = ProductionSynthesisPlanConfig.model_validate_json(json.dumps(payload))
    result = planning._analyze_plan(project_root=tmp_path, config=config)
    assert len(result["plan_rows"]) == 10
    assert {r["sourceDocumentId"] for r in result["plan_rows"]} == {"safe", "dg", "thermal"}
    assert {r.source_document_id for r in result["excluded"]} == {"review", "validation"}
    payload["source_review_exclusions"] = {"nonexistent": "Invalid review identity"}
    with pytest.raises(ValueError, match="outside the pinned catalog"):
        planning._analyze_plan(
            project_root=tmp_path,
            config=ProductionSynthesisPlanConfig.model_validate_json(json.dumps(payload)),
        )
