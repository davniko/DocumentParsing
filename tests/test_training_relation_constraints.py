from __future__ import annotations

import json
from pathlib import Path

from document_ocr.hashing import canonical_json_bytes, sha256_file
from document_ocr.training.config import RelationConstraintsBuildConfig
from document_ocr.training.relation_constraints import build_relation_constraints
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


def test_build_relation_constraints_validates_targets_and_publishes_exact_union(
    tmp_path: Path,
) -> None:
    source = tmp_path / "records.jsonl"
    rows = [
        {
            "target": {
                "schemaVersion": "5.0.0-experimental",
                "documentPatch": {
                    "cargoGroups": [{"groupId": "g1", "description": "CARGO"}],
                    "cargoPackages": [
                        {
                            "packageId": "p1",
                            "groupId": "g1",
                            "quantity": 2,
                            "typeCategory": "PACKAGE_CARTON",
                        }
                    ],
                },
            }
        },
        {
            "target": {
                "schemaVersion": "5.0.0-experimental",
                "documentPatch": {
                    "containers": [
                        {
                            "containerNumber": "TTNU8111930",
                            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                            "typeCategory": "REFRIGERATED",
                        }
                    ]
                },
            }
        },
    ]
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    package_registry = tmp_path / "package.json"
    container_registry = tmp_path / "container.json"
    package_registry.write_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "registryKind": "package",
                "sourceAuthority": "test registry",
                "sourceRevision": "a" * 40,
                "sourcePath": "test/package-registry.json",
                "sourceSha256": "b" * 64,
                "entries": [
                    {
                        "categoryToken": "PACKAGE_CARTON",
                        "applicationCode": "CT",
                        "displayName": "Carton",
                    }
                ],
            }
        )
        + b"\n"
    )
    container_registry.write_bytes(b"{}\n")
    config = RelationConstraintsBuildConfig.model_validate(
        {
            "schema_version": 1,
            "task": "bill_of_lading_relation_explicit_v5",
            "sources": [
                {
                    "path": str(source),
                    "sha256": sha256_file(source),
                    "records": 2,
                }
            ],
            "target_field": "target",
            "package_registry": {
                "path": str(package_registry),
                "sha256": sha256_file(package_registry),
            },
            "container_registry": {
                "path": str(container_registry),
                "sha256": sha256_file(container_registry),
            },
            "output_dir": str(tmp_path / "output"),
        },
        strict=True,
    )

    manifest = build_relation_constraints(
        project_root=tmp_path,
        config=config,
        config_sha256="a" * 64,
    )

    payload = (tmp_path / "output" / "task-constraints.json").read_bytes()
    constraints = RelationExplicitTaskConstraints.model_validate_json(payload, strict=True)
    assert constraints.packageCategoryTokens == ("PACKAGE_CARTON",)
    assert constraints.containerCategoryTokens == ("REFRIGERATED",)
    assert constraints.basePromptSchemaSha256 == get_training_task(
        "bill_of_lading_relation_explicit_v5"
    ).base_prompt_schema_sha256()
    assert manifest["output"]["packageCategoryTokens"] == 1
    assert manifest["output"]["containerCategoryTokens"] == 1
    assert json.loads(payload)["task"] == "bill_of_lading_relation_explicit_v5"
