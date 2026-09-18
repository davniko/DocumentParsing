from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.raw_text_template import printed_topology_mismatches
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.template_compiler.descendant import _load_cases
from document_ocr.synthesis.template_compiler.descendant_models import DescendantConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_CATALOG = (
    _PROJECT_ROOT
    / "artifacts/kie-synthesis-production/template-base/catalogs/"
    "mpci-bl-production-template-catalog1510-v5"
)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _commit_run(
    *, root: Path, name: str, transaction: str, artifacts: dict[str, bytes]
) -> dict[str, str]:
    stage = StagedArtifactRun(
        output_parent=root / "artifacts",
        run_name=name,
        transaction_sha256=transaction,
    )
    for relative, payload in artifacts.items():
        stage.publish_bytes(relative, payload)
    stage.commit(expected_artifacts=artifacts, metadata={})
    return {
        "path": f"artifacts/{name}",
        "commit_sha256": sha256_file(stage.final_root / "_COMMIT.json"),
        "transaction_sha256": transaction,
    }


def _fixture_config(
    tmp_path: Path,
) -> DescendantConfig:
    catalog_row = json.loads(
        (_PRODUCTION_CATALOG / "catalog.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    source_id = catalog_row["documentId"]
    source_case = _PRODUCTION_CATALOG / "cases" / source_id
    template_pin = _commit_run(
        root=tmp_path,
        name="templates",
        transaction="1" * 64,
        artifacts={
            "catalog.jsonl": _jsonl([catalog_row]),
            f"cases/{source_id}/source.txt": (source_case / "source.txt").read_bytes(),
            f"cases/{source_id}/source-label.json": (
                source_case / "source-label.json"
            ).read_bytes(),
            f"cases/{source_id}/template.json": (source_case / "template.json").read_bytes(),
        },
    )
    samples = [
        {
            "sampleId": "sample_a",
            "sourceDocumentId": source_id,
            "variantIndex": 0,
            "targetTask": "bill_of_lading_relation_explicit_v5",
            "targetSchemaVersion": "5.0.0-experimental",
            "targetGeneration": "controlled_latest_schema_from_source_v1",
        },
        {
            "sampleId": "sample_b",
            "sourceDocumentId": source_id,
            "variantIndex": 1,
            "targetTask": "bill_of_lading_relation_explicit_v5",
            "targetSchemaVersion": "5.0.0-experimental",
            "targetGeneration": "controlled_latest_schema_from_source_v1",
        },
    ]
    plan_bytes = _jsonl(samples)
    plan_pin = _commit_run(
        root=tmp_path,
        name="plan",
        transaction="2" * 64,
        artifacts={"plan.jsonl": plan_bytes},
    )
    country_bytes = b'[{"alpha_2":"US","alpha_3":"USA","name":"United States"}]\n'
    (tmp_path / "data").mkdir()
    (tmp_path / "data/countries.json").write_bytes(country_bytes)
    prompt_bytes = b"Render only the requested residual bindings.\n"
    (tmp_path / "prompt.md").write_bytes(prompt_bytes)
    return DescendantConfig.model_validate(
        {
            "schema_version": 1,
            "task": "bill_of_lading_compiled_raw_text_pipeline_v1",
            "run_name": "variants",
            "output_dir": "artifacts/output",
            "environment_file": ".env",
            "inputs": {
                "template_run": template_pin,
                "sample_plan_run": plan_pin,
                "sample_plan": {
                    "path": "artifacts/plan/plan.jsonl",
                    "sha256": sha256_bytes(plan_bytes),
                    "records": 2,
                },
                "iso3166_snapshot": {
                    "path": "data/countries.json",
                    "sha256": sha256_bytes(country_bytes),
                },
            },
            "prompts": {
                "residual_renderer": {
                    "path": "prompt.md",
                    "sha256": sha256_bytes(prompt_bytes),
                }
            },
            "provider": {
                "kind": "openai_responses",
                "model": "test-model",
                "api_key_env": "OPENAI_API_KEY",
                "reasoning_effort": "high",
                "request_timeout_seconds": 10.0,
                "transport_max_retries": 0,
                "max_output_tokens": 100,
                "store_responses": False,
                "pricing": {
                    "currency": "USD",
                    "effective_date": "2026-09-17",
                    "source_url": "https://example.invalid",
                    "input_usd_per_million": Decimal(0),
                    "cached_input_usd_per_million": Decimal(0),
                    "cache_write_multiplier": Decimal(1),
                    "output_usd_per_million": Decimal(0),
                },
            },
            "workflow": {
                "documents": 2,
                "controlled_target_seed": 17,
                "target_schema_version": "5.0.0-experimental",
                "target_generation": "controlled_latest_schema_from_source_v1",
                "max_concurrent_requests": 2,
                "max_requests_per_document": 1,
                "provider_launch_authorized": False,
                "require_exact_topology": True,
                "require_source_carrier": True,
                "require_every_slot_bound_once": True,
                "require_exact_literal_regions": True,
                "require_page_markers_unchanged": True,
                "require_line_endings_preserved": True,
                "publish_training_records": True,
            },
        },
        strict=True,
    )


def test_planned_variants_reuse_one_template_with_unique_runtime_identities(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)

    cases = _load_cases(project_root=tmp_path, config=config)

    assert tuple(row.document_id for row in cases) == ("sample_a", "sample_b")
    assert len({row.source_document_id for row in cases}) == 1
    assert len({row.target_receipt.synthetic_document_id for row in cases}) == 2
    assert len({row.target_receipt.prepared_target_sha256 for row in cases}) == 2
    assert {row.target["schemaVersion"] for row in cases} == {"5.0.0-experimental"}
    assert {row.topology_reference_target["schemaVersion"] for row in cases} == {
        "5.0.0-experimental"
    }
    assert all(
        not printed_topology_mismatches(row.topology_reference_target, row.target)
        for row in cases
    )
    assert all(
        row.target_receipt.target_origin == "planned_v5_controlled_source_variant"
        for row in cases
    )


def test_planned_variant_rejects_external_target_inputs(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    payload = config.model_dump(mode="python")
    payload["inputs"]["synthetic_target_run"] = {
        "path": "artifacts/targets",
        "commit_sha256": "3" * 64,
        "transaction_sha256": "4" * 64,
    }
    payload["inputs"]["synthetic_targets"] = {
        "path": "artifacts/targets/targets.jsonl",
        "sha256": "5" * 64,
        "records": 2,
    }

    with pytest.raises(ValidationError, match="cannot also accept external synthetic targets"):
        DescendantConfig.model_validate(payload, strict=True)


def test_planned_variant_count_is_not_capped_by_template_inventory(tmp_path: Path) -> None:
    payload = _fixture_config(tmp_path).model_dump(mode="python")
    payload["workflow"]["documents"] = 250_000

    config = DescendantConfig.model_validate(payload, strict=True)

    assert config.workflow.documents == 250_000
