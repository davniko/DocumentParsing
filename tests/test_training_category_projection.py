from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.semantic_v3.transform import CategoryAssignments, CategoryRegistry
from document_ocr.training.category_projection import (
    CategoryProjectionError,
    load_category_projection_config,
    project_package_categories,
)


def _publish(path: Path, value: dict[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[str, int]:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return sha256_bytes(payload), len(payload)


def _normal_target(package_type: str) -> dict[str, Any]:
    return {
        "documentPatch": {
            "containers": [
                {"containerNumber": "SUDU8630469", "typeDescription": "40HC"}
            ],
            "goodsItems": [
                {"packages": [{"quantity": 1, "type": package_type}]}
            ],
        },
        "schemaVersion": "2.0.0",
    }


def _relation_target(package_type: str) -> dict[str, Any]:
    return {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 1,
                    "typeDescription": package_type,
                }
            ],
            "containers": [
                {"containerNumber": "SUDU8630469", "typeDescription": "40HC"}
            ],
        },
        "schemaVersion": "3.0.0-experimental",
    }


def _fixture(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir()
    package_types = ("CARTONS", "PAILS", "MYSTERY")
    records: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for index, package_type in enumerate(package_types, start=1):
        document_id = f"doc_{index:064x}"
        raw_text = f"1 {package_type}"
        records.append(
            {
                "documentId": document_id,
                "joinedRawText": raw_text,
                "joinedRawTextSha256": sha256_bytes(raw_text.encode()),
                "normalTarget": _normal_target(package_type),
                "target": _relation_target(package_type),
            }
        )
        lineage.append({"documentId": document_id, "fixture": True})
    records_sha, records_bytes = _jsonl(source_root / "records.jsonl", records)
    lineage_sha, lineage_bytes = _jsonl(source_root / "lineage.jsonl", lineage)
    manifest_sha = _publish(
        source_root / "manifest.json",
        {
            "datasetId": "source-dataset",
            "files": [
                {
                    "bytes": records_bytes,
                    "kind": "training_records",
                    "path": "records.jsonl",
                    "rows": len(records),
                    "sha256": records_sha,
                },
                {
                    "bytes": lineage_bytes,
                    "kind": "lineage",
                    "path": "lineage.jsonl",
                    "rows": len(lineage),
                    "sha256": lineage_sha,
                },
            ],
        },
    )

    package_registry_path = tmp_path / "package-registry.json"
    package_registry = CategoryRegistry.model_validate(
        {
            "schemaVersion": 1,
            "registryKind": "package",
            "sourceAuthority": "fixture",
            "sourceRevision": "a" * 40,
            "sourcePath": "fixture/package",
            "sourceSha256": "b" * 64,
            "entries": (
                {
                    "applicationCode": "CT",
                    "categoryToken": "PACKAGE_CARTON",
                    "displayName": "Carton",
                },
                {
                    "applicationCode": "PL",
                    "categoryToken": "PACKAGE_PAIL",
                    "displayName": "Pail",
                },
            ),
        },
        strict=True,
    )
    package_registry_sha = _publish(
        package_registry_path, package_registry.model_dump(mode="json")
    )
    container_registry_path = tmp_path / "container-registry.json"
    container_registry = CategoryRegistry.model_validate(
        {
            "schemaVersion": 1,
            "registryKind": "container",
            "sourceAuthority": "fixture",
            "sourceRevision": "c" * 40,
            "sourcePath": "fixture/container",
            "sourceSha256": "d" * 64,
            "entries": (),
        },
        strict=True,
    )
    container_registry_sha = _publish(
        container_registry_path, container_registry.model_dump(mode="json")
    )
    prior_path = tmp_path / "prior.json"
    prior = CategoryAssignments.model_validate(
        {
            "schemaVersion": 2,
            "reviewConfigSha256": "e" * 64,
            "sourceInventorySha256": "f" * 64,
            "packageRegistrySha256": package_registry_sha,
            "containerRegistrySha256": container_registry_sha,
            "packageAssignments": (
                {
                    "sourceTypeText": "CARTONS",
                    "sourceTypeCode": None,
                    "occurrences": 1,
                    "categoryToken": "PACKAGE_CARTON",
                    "reviewBasis": "manual_semantic_review",
                    "reviewedBy": "fixture-reviewer",
                    "reviewedOn": date(2026, 8, 22),
                },
            ),
            "containerAssignments": (),
        },
        strict=True,
    )
    prior_sha = _publish(prior_path, prior.model_dump(mode="json"))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "projection": "mpci_bl_task_facing_package_categories_v1",
                "source": {
                    "dataset_id": "source-dataset",
                    "root": str(source_root),
                    "manifest_sha256": manifest_sha,
                    "expected_records": 3,
                },
                "registries": {
                    "package": {
                        "path": str(package_registry_path),
                        "sha256": package_registry_sha,
                    },
                    "container": {
                        "path": str(container_registry_path),
                        "sha256": container_registry_sha,
                    },
                    "prior_assignments": {
                        "path": str(prior_path),
                        "sha256": prior_sha,
                    },
                },
                "review": {
                    "reviewed_by": "fixture-extension-reviewer",
                    "reviewed_on": date(2026, 8, 26),
                    "expected_package_occurrences": 3,
                    "expected_package_variants": 3,
                    "expected_container_occurrences": 3,
                    "expected_container_variants": 1,
                    "extension_resolution_groups": [
                        {
                            "category_token": "PACKAGE_PAIL",
                            "source_type_descriptions": ["PAILS"],
                            "rationale": "The source explicitly names pails.",
                        }
                    ],
                    "unresolved_descriptions": [
                        {
                            "source_type_description": "MYSTERY",
                            "rationale": "No exact registry meaning is supported.",
                        }
                    ],
                },
                "output": {
                    "root": str(tmp_path / "output"),
                    "dataset_id": "projected-dataset",
                },
            },
            sort_keys=False,
        )
    )
    return config_path


def test_category_projection_reuses_extends_and_preserves_raw_fallback(
    tmp_path: Path,
) -> None:
    config_path = _fixture(tmp_path)
    config = load_category_projection_config(config_path)
    manifest_path = project_package_categories(config)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["records"] == 3
    assert manifest["categoryPolicy"]["containerCategoriesInferred"] is False
    records = [
        json.loads(line)
        for line in (manifest_path.parent / "records.jsonl").read_text().splitlines()
    ]
    package_types = [
        row["target"]["documentPatch"]["cargoPackages"][0] for row in records
    ]
    assert package_types[0]["typeCategory"] == "PACKAGE_CARTON"
    assert package_types[1]["typeCategory"] == "PACKAGE_PAIL"
    assert package_types[2]["typeDescription"] == "MYSTERY"
    assert all(
        row["target"]["documentPatch"]["containers"][0]["typeDescription"] == "40HC"
        for row in records
    )

    constraints = json.loads((manifest_path.parent / "task-constraints.json").read_text())
    assert constraints["packageCategoryTokens"] == ["PACKAGE_CARTON", "PACKAGE_PAIL"]
    assert constraints["containerCategoryTokens"] == []
    metadata = [
        json.loads(line)
        for line in (manifest_path.parent / "category-metadata.jsonl").read_text().splitlines()
    ]
    assert metadata[0]["sourceRelationTarget"]["documentPatch"]["cargoPackages"][0][
        "typeDescription"
    ] == "CARTONS"
    inventory = [
        json.loads(line)
        for line in (
            manifest_path.parent / "package-category-inventory.jsonl"
        ).read_text().splitlines()
    ]
    inventory_by_description = {row["sourceTypeDescription"]: row for row in inventory}
    assert inventory_by_description["CARTONS"]["decisionProvenance"] == "reused_prior_review"
    assert inventory_by_description["PAILS"]["decisionProvenance"] == (
        "current_exact_source_review"
    )
    assert inventory_by_description["MYSTERY"]["categoryToken"] is None


def test_category_projection_fails_closed_on_inventory_drift(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    value = yaml.safe_load(config_path.read_text())
    value["review"]["expected_package_occurrences"] = 4
    config_path.write_text(yaml.safe_dump(value, sort_keys=False))

    with pytest.raises(CategoryProjectionError, match="inventory differs"):
        project_package_categories(load_category_projection_config(config_path))


def test_category_projection_rejects_re_reviewing_prior_exact_key(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    value = yaml.safe_load(config_path.read_text())
    value["review"]["extension_resolution_groups"][0][
        "source_type_descriptions"
    ].append("CARTONS")
    config_path.write_text(yaml.safe_dump(value, sort_keys=False))

    with pytest.raises(CategoryProjectionError, match="redundantly re-reviews"):
        project_package_categories(load_category_projection_config(config_path))


def test_category_projection_reuses_a_pinned_prior_projection_inventory(
    tmp_path: Path,
) -> None:
    config_path = _fixture(tmp_path)
    value = yaml.safe_load(config_path.read_text())
    inventory_path = tmp_path / "prior-projection-inventory.jsonl"
    inventory_sha, _ = _jsonl(
        inventory_path,
        [
            {
                "categoryToken": "PACKAGE_PAIL",
                "rationale": "The exact prior source description names pails.",
                "reviewBasis": "manual_semantic_review",
                "sourceTypeDescription": "PAILS",
            }
        ],
    )
    value["registries"]["prior_projection_inventory"] = {
        "path": str(inventory_path),
        "sha256": inventory_sha,
    }
    value["review"]["extension_resolution_groups"] = []
    config_path.write_text(yaml.safe_dump(value, sort_keys=False))

    manifest_path = project_package_categories(
        load_category_projection_config(config_path)
    )
    inventory = [
        json.loads(line)
        for line in (
            manifest_path.parent / "package-category-inventory.jsonl"
        ).read_text().splitlines()
    ]
    by_description = {row["sourceTypeDescription"]: row for row in inventory}
    assert by_description["PAILS"]["categoryToken"] == "PACKAGE_PAIL"
    assert by_description["PAILS"]["decisionProvenance"] == "reused_prior_projection"
