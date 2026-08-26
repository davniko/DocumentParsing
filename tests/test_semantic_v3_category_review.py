from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.semantic_v3.category_review import (
    CategoryReviewError,
    publish_category_assignments,
)
from document_ocr.semantic_v3.config import SemanticV3TransformConfig
from document_ocr.semantic_v3.transform import CategoryAssignments, _source_inventory_sha256


def _write_json(path: Path, value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _fixture(tmp_path: Path) -> tuple[SemanticV3TransformConfig, Path]:
    source_path = tmp_path / "source.jsonl"
    source_payload = b'{"fixture":true}\n'
    source_path.write_bytes(source_payload)
    config = SemanticV3TransformConfig.model_validate(
        {
            "schema_version": 1,
            "transform": "mpci_bl_semantic_v2_to_relation_explicit_v3",
            "source": {
                "expected_documents": 1,
                "files": [
                    {
                        "id": "fixture_train",
                        "split": "train",
                        "path": str(source_path),
                        "sha256": sha256_bytes(source_payload),
                        "records": 1,
                    }
                ],
            },
            "registries": {
                "package_categories": None,
                "container_categories": None,
                "category_assignments": None,
                "relation_assignments": None,
            },
            "allowed_relation_exclusions": [],
            "output": {
                "audit_root": str(tmp_path / "audit"),
                "dataset_root": str(tmp_path / "dataset"),
                "dataset_id": "category-review-fixture",
            },
        },
        strict=True,
    )
    package_registry = tmp_path / "package-registry.json"
    package_sha = _write_json(
        package_registry,
        {
            "schemaVersion": 1,
            "registryKind": "package",
            "sourceAuthority": "fixture",
            "sourceRevision": "1" * 40,
            "sourcePath": "package.ts",
            "sourceSha256": "2" * 64,
            "entries": [
                {
                    "categoryToken": "PACKAGE_CARTON",
                    "applicationCode": "CT",
                    "displayName": "Carton",
                }
            ],
        },
    )
    container_registry = tmp_path / "container-registry.json"
    container_sha = _write_json(
        container_registry,
        {
            "schemaVersion": 1,
            "registryKind": "container",
            "sourceAuthority": "blocked fixture",
            "sourceRevision": "1" * 40,
            "sourcePath": "container.ts",
            "sourceSha256": "3" * 64,
            "entries": [],
        },
    )
    package_inventory = tmp_path / "package-review.jsonl"
    package_payload = canonical_json_bytes(
        {
            "affectedDocuments": ["doc_" + "a" * 64],
            "assignmentStatus": "unassigned",
            "candidateCategoryToken": None,
            "occurrences": 2,
            "sourceTypeCode": None,
            "sourceTypeText": "CARTONS",
        }
    ) + b"\n"
    package_inventory.write_bytes(package_payload)
    container_inventory = tmp_path / "container-review.jsonl"
    container_payload = canonical_json_bytes(
        {
            "affectedDocuments": ["doc_" + "a" * 64],
            "assignmentStatus": "unassigned",
            "candidateCategoryToken": None,
            "occurrences": 1,
            "sourceTypeCode": "40HC",
            "sourceTypeDescription": None,
        }
    ) + b"\n"
    container_inventory.write_bytes(container_payload)
    review_path = tmp_path / "review.yaml"
    review_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source_inventory_sha256": _source_inventory_sha256(config),
                "reviewed_by": "fixture-reviewer",
                "reviewed_on": date(2026, 8, 22),
                "package_registry": {
                    "path": str(package_registry),
                    "sha256": package_sha,
                },
                "container_registry": {
                    "path": str(container_registry),
                    "sha256": container_sha,
                },
                "package_inventory": {
                    "path": str(package_inventory),
                    "sha256": sha256_bytes(package_payload),
                    "rows": 1,
                },
                "container_inventory": {
                    "path": str(container_inventory),
                    "sha256": sha256_bytes(container_payload),
                    "rows": 1,
                },
                "expected_package_occurrences": 2,
                "expected_container_occurrences": 1,
                "package_resolution_groups": [
                    {
                        "category_token": "PACKAGE_CARTON",
                        "source_type_texts": ["CARTONS"],
                    }
                ],
                "unresolved_package_type_texts": [],
                "container_policy": "preserve_printed_type_no_semantic_registry",
                "output_path": str(tmp_path / "category-assignments.json"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, review_path


def test_category_review_publishes_exact_package_mapping_and_code_only_container_fallback(
    tmp_path: Path,
) -> None:
    config, review_path = _fixture(tmp_path)

    output = publish_category_assignments(config, review_path)
    artifact = CategoryAssignments.model_validate_json(output.read_bytes(), strict=True)

    assert artifact.reviewConfigSha256 == sha256_bytes(review_path.read_bytes())
    assert artifact.packageAssignments[0].categoryToken == "PACKAGE_CARTON"
    assert artifact.containerAssignments[0].sourceTypeCode == "40HC"
    assert artifact.containerAssignments[0].categoryToken is None


def test_category_review_rejects_incomplete_package_decisions(tmp_path: Path) -> None:
    config, review_path = _fixture(tmp_path)
    value = yaml.safe_load(review_path.read_text())
    value["package_resolution_groups"] = [
        {"category_token": "PACKAGE_CARTON", "source_type_texts": ["BOXES"]}
    ]
    review_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    with pytest.raises(CategoryReviewError, match="do not exactly cover"):
        publish_category_assignments(config, review_path)
