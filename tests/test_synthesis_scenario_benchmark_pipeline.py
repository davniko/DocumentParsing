from __future__ import annotations

import json
from pathlib import Path

import yaml

from document_ocr.synthesis.config import load_synthesis_party_structure_benchmark_config
from document_ocr.synthesis.country_registry import CountryEntry, CountryRegistry
from document_ocr.synthesis.scenario_benchmark_pipeline import (
    _country_clean_fit_ids,
    _publish_top_manifest,
)


def _registry() -> CountryRegistry:
    return CountryRegistry(
        entries=(
            CountryEntry.model_validate(
                {
                    "alpha2": "BE",
                    "alpha3": "BEL",
                    "numeric": "056",
                    "name": "Belgium",
                },
                strict=True,
            ),
            CountryEntry.model_validate(
                {
                    "alpha2": "EG",
                    "alpha3": "EGY",
                    "numeric": "818",
                    "name": "Egypt",
                },
                strict=True,
            ),
        ),
        observed_aliases={},
        iso_sha256="1" * 64,
        observed_aliases_sha256="2" * 64,
    )


def test_country_clean_scope_excludes_unresolved_document_without_aliasing() -> None:
    tables = {
        "document_locations": (
            {
                "document_id": "clean",
                "location_id": "clean-port",
                "country": "Belgium",
            },
            {
                "document_id": "unresolved",
                "location_id": "unknown-port",
                "country": "BELGIQUE CUSTOM SURFACE",
            },
        ),
        "parties": (
            {
                "document_id": "clean",
                "party_id": "clean-party",
                "country": "Egypt",
            },
            {
                "document_id": "missing",
                "party_id": "missing-party",
                "country": None,
            },
        ),
    }

    eligible, audit = _country_clean_fit_ids(
        tables=tables,
        fit_ids=("clean", "missing", "unresolved"),
        registry=_registry(),
    )

    assert eligible == ("clean", "missing")
    assert audit["aliasesApplied"] == 0
    assert audit["excludedDocuments"] == 1
    assert audit["unresolvedSurfaceCounts"] == {"BELGIQUE CUSTOM SURFACE": 1}
    assert audit["missingCells"] == 1


def test_top_manifest_preserves_partial_comparison_status_and_candidate_sets(
    tmp_path: Path,
) -> None:
    summary = {
        "benchmarkStatus": "complete_with_candidate_failures",
        "requestedCandidates": ["empirical", "gaussian_copula", "ctgan", "tvae"],
        "completeCandidates": ["empirical", "ctgan", "tvae"],
        "failedCandidates": ["gaussian_copula"],
        "productionSelectionPerformed": False,
    }
    _publish_top_manifest(
        tmp_path,
        summary=summary,
        benchmark_status="complete_with_candidate_failures",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == ("complete_with_candidate_failures_no_production_selection")
    assert manifest["summary"] == summary


def test_party_benchmark_config_requires_full_paired_gpu_contract(tmp_path: Path) -> None:
    value = {
        "schema_version": 1,
        "task": "bill_of_lading_relation_explicit_v3",
        "run": {"run_id": "party-gpu-test", "output_dir": "artifacts"},
        "inputs": {
            "preparation_root": "prepared",
            "preparation_manifest": {"path": "prepared/manifest.json", "sha256": "1" * 64},
            "template_groups": {
                "path": "templates.jsonl",
                "sha256": "2" * 64,
                "records": 10,
            },
            "partition_report": {"path": "partition.json", "sha256": "3" * 64},
            "iso3166_snapshot": {"path": "iso.json", "sha256": "4" * 64},
        },
        "selection": {
            "split": "train",
            "require_template_wholly_in_split": True,
            "unresolved_source_country_policy": "exclude_document_and_audit_v1",
        },
        "modeling": {
            "scope": "full_gpu",
            "view": "party_structure",
            "candidates": ["empirical", "gaussian_copula", "ctgan", "tvae"],
            "fold_count": 3,
            "fold_seed": 17,
            "seeds": [19],
            "neural_epochs": 300,
            "neural_batch_size": 200,
            "proposal_multiplier": 8,
            "proposal_batch_rows": 2048,
            "production_selection": False,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    config = load_synthesis_party_structure_benchmark_config(path)

    assert config.modeling.candidates == [
        "empirical",
        "gaussian_copula",
        "ctgan",
        "tvae",
    ]
    assert config.modeling.neural_batch_size == 200
