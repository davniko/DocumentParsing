from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingRelationExplicitLabel,
    CargoAllocationGroup,
    project_relation_to_normal,
    validate_dual_cargo_consistency,
)
from document_ocr.labeling_agents.models import AnnotationDraft
from document_ocr.semantic_v3.config import (
    SemanticV3TransformConfig,
    TrainingConfigPublication,
)
from document_ocr.semantic_v3.transform import (
    SemanticV3TransformError,
    _publish_training_config,
    _relation_audit,
    _source_inventory_sha256,
    _transform_target,
    _valid_reviewed_relation_choices,
    audit_semantic_v3_readiness,
    publish_semantic_v3_dataset,
)
from document_ocr.training.config import PromptConfig, TrainingConfig, load_training_config
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import (
    RelationExplicitTaskConstraints,
    get_training_task,
    load_training_task,
)


def _publish_json(path: Path, value: dict[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _source_row() -> dict[str, Any]:
    joined = "--- PAGE 1 ---\nB/L NO HBL-001\nMSKU1200040 40HC\n10 CARTONS"
    return {
        "documentId": "doc_" + "a" * 64,
        "joinedRawText": joined,
        "joinedRawTextSha256": sha256_bytes(joined.encode()),
        "target": {
            "schemaVersion": "2.0.0",
            "documentPatch": {
                "billOfLadingNumber": "HBL-001",
                "containers": [
                    {
                        "containerNumber": "MSKU1200040",
                        "typeDescription": "40HC",
                        "sealNumbers": ["SEAL-1"],
                    }
                ],
                "goodsItems": [
                    {
                        "description": "MACHINE PARTS",
                        "packages": [{"quantity": 10, "type": "CARTONS"}],
                        "dangerousGoods": [
                            {
                                "unNumber": "1203",
                                "hazardClass": "3",
                                "subsidiaryHazard": "8",
                                "flashPoint": {
                                    "temperature": {"value": -20.0, "unit": "celsius"},
                                    "packingGroup": "II",
                                },
                            }
                        ],
                        "containerAllocations": [
                            {
                                "containerNumber": "MSKU1200040",
                                "packageQuantity": 10,
                            }
                        ],
                    }
                ],
                "route": {"portOfLoading": {"name": "BUSAN", "country": "KOREA"}},
            },
        },
    }


def _base_config(tmp_path: Path, source_path: Path, source_sha: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "transform": "mpci_bl_semantic_v2_to_relation_explicit_v3",
        "source": {
            "expected_documents": 1,
            "files": [
                {
                    "id": "fixture_train",
                    "split": "train",
                    "path": str(source_path),
                    "sha256": source_sha,
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
            "dataset_id": "fixture-semantic-v3",
        },
    }


def _configured(
    tmp_path: Path,
    *,
    package_category: str | None = "CARTON",
    container_category: str | None = "FORTY_FOOT_HIGH_CUBE_DRY",
) -> tuple[SemanticV3TransformConfig, bytes]:
    source_path = tmp_path / "source.jsonl"
    source_payload = canonical_json_bytes(_source_row()) + b"\n"
    source_path.write_bytes(source_payload)
    config_value = _base_config(tmp_path, source_path, sha256_bytes(source_payload))
    audit_config = SemanticV3TransformConfig.model_validate(config_value, strict=True)

    package_registry_path = tmp_path / "package-registry.json"
    package_sha = _publish_json(
        package_registry_path,
        {
            "schemaVersion": 1,
            "registryKind": "package",
            "sourceAuthority": "fixture",
            "sourceRevision": "1" * 40,
            "sourcePath": "package-types.ts",
            "sourceSha256": "2" * 64,
            "entries": [
                {
                    "categoryToken": "CARTON",
                    "applicationCode": "CT",
                    "displayName": "Carton",
                }
            ],
        },
    )
    container_registry_path = tmp_path / "container-registry.json"
    container_sha = _publish_json(
        container_registry_path,
        {
            "schemaVersion": 1,
            "registryKind": "container",
            "sourceAuthority": "fixture",
            "sourceRevision": "1" * 40,
            "sourcePath": "container-size-types.ts",
            "sourceSha256": "3" * 64,
            "entries": [
                {
                    "categoryToken": "FORTY_FOOT_HIGH_CUBE_DRY",
                    "applicationCode": "45G1",
                    "displayName": "40 foot high cube dry",
                }
            ],
        },
    )
    assignments_path = tmp_path / "assignments.json"
    assignments_sha = _publish_json(
        assignments_path,
        {
            "schemaVersion": 2,
            "reviewConfigSha256": "f" * 64,
            "sourceInventorySha256": _source_inventory_sha256(audit_config),
            "packageRegistrySha256": package_sha,
            "containerRegistrySha256": container_sha,
            "packageAssignments": [
                {
                    "sourceTypeText": "CARTONS",
                    "sourceTypeCode": None,
                    "occurrences": 1,
                    "categoryToken": package_category,
                    "reviewBasis": (
                        "manual_semantic_review"
                        if package_category is not None
                        else "insufficient_source_specificity"
                    ),
                    "reviewedBy": "fixture-reviewer",
                    "reviewedOn": "2026-08-22",
                }
            ],
            "containerAssignments": [
                {
                    "sourceTypeDescription": "40HC",
                    "sourceTypeCode": None,
                    "occurrences": 1,
                    "categoryToken": container_category,
                    "reviewBasis": (
                        "manual_semantic_review"
                        if container_category is not None
                        else "insufficient_source_specificity"
                    ),
                    "reviewedBy": "fixture-reviewer",
                    "reviewedOn": "2026-08-22",
                }
            ],
        },
    )
    relation_assignments_path = tmp_path / "relation-assignments.json"
    relation_assignments_sha = _publish_json(
        relation_assignments_path,
        {
            "schemaVersion": 1,
            "sourceInventorySha256": _source_inventory_sha256(audit_config),
            "assignments": [],
        },
    )
    config_value["registries"] = {
        "package_categories": {
            "path": str(package_registry_path),
            "sha256": package_sha,
        },
        "container_categories": {
            "path": str(container_registry_path),
            "sha256": container_sha,
        },
        "category_assignments": {
            "path": str(assignments_path),
            "sha256": assignments_sha,
        },
        "relation_assignments": {
            "path": str(relation_assignments_path),
            "sha256": relation_assignments_sha,
        },
    }
    return SemanticV3TransformConfig.model_validate(config_value, strict=True), source_payload


def test_readiness_audit_is_non_destructive_and_reports_exact_inventory(
    tmp_path: Path,
) -> None:
    config, source_payload = _configured(tmp_path)
    source_path = Path(config.source.files[0].path)
    manifest_path = audit_semantic_v3_readiness(config)

    assert source_path.read_bytes() == source_payload
    summary = json.loads((manifest_path.parent / "summary.json").read_text())
    assert summary["source_documents"] == 1
    assert summary["package_category_source_variants"] == 1
    assert summary["container_category_source_variants"] == 1
    assert summary["cargo_relation_classifications"] == {"one_to_one_package_allocations": 1}
    assert summary["transform_ready"] is True
    categoricals = json.loads((manifest_path.parent / "categorical-ownership.json").read_text())
    assert categoricals["observedSmallCategoricals"][
        "dangerousGoods.subsidiaryHazard"
    ] == {"8": 1}


def test_transform_preserves_source_fields_and_emits_explicit_relations_and_categories(
    tmp_path: Path,
) -> None:
    config, source_payload = _configured(tmp_path)
    source_path = Path(config.source.files[0].path)
    manifest_path = publish_semantic_v3_dataset(config)

    assert source_path.read_bytes() == source_payload
    output = json.loads((manifest_path.parent / "train.jsonl").read_text())
    source = _source_row()
    provenance = output.pop("semanticV3Transform")
    target = output.pop("target")
    source_target = source.pop("target")
    assert output == source
    assert provenance["sourceTargetCanonicalSha256"] == sha256_bytes(
        canonical_json_bytes(source_target)
    )
    assert target["documentPatch"]["route"] == source_target["documentPatch"]["route"]
    assert target["documentPatch"]["containers"][0]["typeCategory"] == ("FORTY_FOOT_HIGH_CUBE_DRY")
    assert target["documentPatch"]["containers"][0]["typeDescription"] == "40HC"
    assert "typeCode" not in target["documentPatch"]["containers"][0]
    assert target["documentPatch"]["cargoPackages"] == [
        {"groupId": "g1", "packageId": "p1", "quantity": 10, "typeCategory": "CARTON"}
    ]
    assert target["documentPatch"]["cargoGroups"][0]["dangerousGoods"] == [
        {
            "unNumber": "1203",
            "hazardCategory": "FLAMMABLE_LIQUIDS",
            "subsidiaryHazardCategory": "CORROSIVE_SUBSTANCES",
            "flashPoint": {
                "temperature": {"value": -20.0, "unit": "celsius"},
                "packingGroupCategory": "MEDIUM_DANGER",
            },
        }
    ]
    assert target["documentPatch"]["cargoAllocationGroups"] == [
        {
            "allocations": [
                {
                    "containerNumber": "MSKU1200040",
                    "packageId": "p1",
                    "packageQuantity": 10,
                }
            ],
            "coverage": "one_to_one_package_allocations",
            "groupId": "g1",
            "packageIds": ["p1"],
        }
    ]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["category_projection"]["package_token_to_application_code"] == {"CARTON": "CT"}
    assert manifest["category_projection"]["container_token_to_application_code"] == {
        "FORTY_FOOT_HIGH_CUBE_DRY": "45G1"
    }
    assert manifest["category_projection"]["hazard_category_to_imdg_class"][
        "FLAMMABLE_LIQUIDS"
    ] == "3"
    assert manifest["category_projection"]["packing_group_category_to_code"] == {
        "HIGH_DANGER": "I",
        "LOW_DANGER": "III",
        "MEDIUM_DANGER": "II",
    }
    constraints_path = manifest_path.parent / manifest["training_task_constraints"]["path"]
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        constraints_path.read_bytes(), strict=True
    )
    task = get_training_task("bill_of_lading_relation_explicit_v3").bind_constraints(
        constraints
    )
    prompt_schema = json.loads(task.prompt_schema_json())
    assert prompt_schema["$defs"]["CargoPackageFact"]["properties"]["typeCategory"] == {
        "enum": ["CARTON"],
        "type": "string",
    }
    assert prompt_schema["$defs"]["RelationExplicitContainer"]["properties"][
        "typeCategory"
    ] == {
        "enum": ["FORTY_FOOT_HIGH_CUBE_DRY"],
        "type": "string",
    }

    invalid_target = json.loads(json.dumps(target))
    invalid_target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "UNKNOWN"
    with pytest.raises(ValueError, match="outside the frozen vocabulary"):
        task.canonicalize(invalid_target)

    project_root = Path(__file__).resolve().parents[1]
    training_value = load_training_config(
        project_root / "configs/training/t5gemma2_270m_lora.pilot106.yaml"
    ).model_dump(mode="python")
    training_value["task"] = "bill_of_lading_relation_explicit_v3"
    training_value["task_constraints"] = {
        "path": str(constraints_path),
        "sha256": sha256_bytes(constraints_path.read_bytes()),
    }
    training_config = TrainingConfig.model_validate(training_value, strict=True)
    loaded_task = load_training_task(project_root, training_config)
    assert loaded_task.constraints == constraints


def test_transform_preserves_printed_type_fallbacks_without_guessing_categories(
    tmp_path: Path,
) -> None:
    config, _ = _configured(
        tmp_path,
        package_category=None,
        container_category=None,
    )

    manifest_path = publish_semantic_v3_dataset(config)
    output = json.loads((manifest_path.parent / "train.jsonl").read_text())
    patch = output["target"]["documentPatch"]

    assert patch["containers"][0]["typeDescription"] == "40HC"
    assert "typeCategory" not in patch["containers"][0]
    assert patch["cargoPackages"] == [
        {
            "groupId": "g1",
            "packageId": "p1",
            "quantity": 10,
            "typeDescription": "CARTONS",
        }
    ]
    projection = json.loads(manifest_path.read_text())["category_projection"]
    assert projection["unresolved_package_source_keys"] == 1
    assert projection["unresolved_container_source_keys"] == 1


def test_transform_preserves_a_code_only_container_as_printed_type_text() -> None:
    source = _source_row()["target"]
    source["documentPatch"]["containers"][0] = {
        "containerNumber": "MSKU1200040",
        "typeCode": "40HC",
    }

    transformed = _transform_target(
        source,
        package_map={("CARTONS", None): None},
        container_map={(None, "40HC"): None},
        relation_choices={0: "one_to_one_package_allocations"},
    )

    assert transformed["documentPatch"]["containers"][0] == {
        "containerNumber": "MSKU1200040",
        "typeDescription": "40HC",
    }


def test_prompt_schema_omits_category_fields_with_no_frozen_vocabulary() -> None:
    base_task = get_training_task("bill_of_lading_relation_explicit_v3")
    constraints = RelationExplicitTaskConstraints.model_validate(
        {
            "schemaVersion": 1,
            "task": "bill_of_lading_relation_explicit_v3",
            "basePromptSchemaSha256": base_task.base_prompt_schema_sha256(),
            "targetSchemaSha256": sha256_bytes(
                canonical_json_bytes(
                    base_task.target_model.model_json_schema(mode="serialization")
                )
            ),
            "packageRegistrySha256": "1" * 64,
            "containerRegistrySha256": "2" * 64,
            "packageCategoryTokens": ("PACKAGE_CARTON",),
            "containerCategoryTokens": (),
        },
        strict=True,
    )
    schema = json.loads(base_task.bind_constraints(constraints).prompt_schema_json())

    assert schema["$defs"]["CargoPackageFact"]["properties"]["typeCategory"] == {
        "enum": ["PACKAGE_CARTON"],
        "type": "string",
    }
    assert "typeCategory" not in schema["$defs"]["RelationExplicitContainer"][
        "properties"
    ]


def test_training_config_publication_clones_baseline_and_binds_generated_dataset(
    tmp_path: Path,
) -> None:
    config, _ = _configured(tmp_path)
    project_root = tmp_path / "project"
    project_root.mkdir()
    repository_root = Path(__file__).resolve().parents[1]
    baseline_path = project_root / "baseline.yaml"
    prompt_path = project_root / "relation-v3.txt"
    baseline_payload = (
        repository_root
        / "configs/training/t5gemma2_270m_lora.mpci_bl_combined487.yaml"
    ).read_bytes()
    prompt_payload = (
        repository_root / "prompts/kie/bill_of_lading_relation_explicit_v3.txt"
    ).read_bytes()
    baseline_path.write_bytes(baseline_payload)
    prompt_path.write_bytes(prompt_payload)
    output_root = project_root / "dataset"
    output_root.mkdir()
    train_payload = b"".join(
        f'{{"documentId":"doc_train_{index}"}}\n'.encode() for index in range(423)
    )
    validation_payload = b"".join(
        f'{{"documentId":"doc_validation_{index}"}}\n'.encode()
        for index in range(60)
    )
    (output_root / "train.jsonl").write_bytes(train_payload)
    (output_root / "validation.jsonl").write_bytes(validation_payload)
    constraints_path = output_root / "task-constraints.json"
    constraints_payload = b"{}\n"
    constraints_path.write_bytes(constraints_payload)
    publication = TrainingConfigPublication.model_validate(
        {
            "source": {
                "project_root": str(project_root),
                "baseline_path": str(baseline_path),
                "baseline_sha256": sha256_bytes(baseline_payload),
                "prompt_path": str(prompt_path),
                "prompt_sha256": sha256_bytes(prompt_payload),
            },
            "capacity": {
                "max_source_length": 9216,
                "max_target_length": 4096,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 24,
                "per_device_eval_batch_size": 2,
            },
            "output_filename": "training.yaml",
            "run_id": "relation-v3-fixture",
            "adapter_name": "relation_v3_fixture",
            "evaluation_every_epochs": 5,
        },
        strict=True,
    )
    config = config.model_copy(update={"training_config": publication})
    split_files = [
        {
            "kind": "train_records",
            "path": "train.jsonl",
            "records": 423,
            "sha256": sha256_bytes(train_payload),
        },
        {
            "kind": "validation_records",
            "path": "validation.jsonl",
            "records": 60,
            "sha256": sha256_bytes(validation_payload),
        },
    ]

    output_path, _, token_audit = _publish_training_config(
        config,
        output_root=output_root,
        split_files=split_files,
        task_constraints_path=constraints_path,
        task_constraints_sha256=sha256_bytes(constraints_payload),
        train_records=423,
    )

    baseline = load_training_config(baseline_path)
    generated = load_training_config(output_path)
    assert token_audit is None
    assert generated.task == "bill_of_lading_relation_explicit_v3"
    assert generated.run.run_id == "relation-v3-fixture"
    assert generated.peft.adapter_name == "relation_v3_fixture"
    assert generated.peft.target_modules_regex == baseline.peft.target_modules_regex
    assert generated.optimization.learning_rate == baseline.optimization.learning_rate
    assert generated.optimization.adam_beta1 == baseline.optimization.adam_beta1
    assert generated.optimization.adam_beta2 == baseline.optimization.adam_beta2
    assert generated.optimization.per_device_train_batch_size == 1
    assert generated.optimization.gradient_accumulation_steps == 24
    assert generated.dataset.preprocessing.max_source_length == 9216
    assert generated.dataset.preprocessing.max_target_length == 4096
    assert generated.evaluation.per_device_batch_size == 2
    assert generated.evaluation.steps == 90
    assert generated.checkpoint.steps == 90
    assert generated.evaluation.on_start is True
    assert generated.dataset.splits.train[0].records == 423
    assert generated.dataset.splits.validation[0].records == 60


def test_transform_fails_closed_without_frozen_category_inputs(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_payload = canonical_json_bytes(_source_row()) + b"\n"
    source_path.write_bytes(source_payload)
    config = SemanticV3TransformConfig.model_validate(
        _base_config(tmp_path, source_path, sha256_bytes(source_payload)), strict=True
    )

    with pytest.raises(SemanticV3TransformError, match="requires frozen"):
        publish_semantic_v3_dataset(config)


def test_relation_schema_rejects_allocation_total_different_from_package_scope() -> None:
    invalid = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "containers": [{"containerNumber": "MSKU1200040"}],
            "cargoGroups": [{"groupId": "g1", "description": "MACHINE PARTS"}],
            "cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": 10}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageIds": ["p1"],
                    "allocations": [{"containerNumber": "MSKU1200040", "packageQuantity": 9}],
                }
            ],
        },
    }
    with pytest.raises(ValidationError, match="allocation total differs"):
        BillOfLadingRelationExplicitLabel.model_validate_json(
            canonical_json_bytes(invalid), strict=True
        )


def test_dual_cargo_draft_preserves_both_views_and_rejects_unfrozen_categories() -> None:
    normal_value = _source_row()["target"]
    normal = BillOfLadingLabel.model_validate_json(
        canonical_json_bytes(normal_value), strict=True
    )
    relation_value = _transform_target(
        normal_value,
        package_map={("CARTONS", None): None},
        container_map={("40HC", None): None},
        relation_choices={0: "one_to_one_package_allocations"},
    )
    relation = BillOfLadingRelationExplicitLabel.model_validate_json(
        canonical_json_bytes(relation_value), strict=True
    )
    validate_dual_cargo_consistency(normal, relation)
    draft_value = {
        "decision": "annotation",
        "documentType": "bill_of_lading",
        "normalLabel": normal_value,
        "relationExplicitLabel": relation_value,
        "evidence": [
            {
                "targetPath": "documentPatch.billOfLadingNumber",
                "evidenceKind": "verbatim",
                "rawOcrEvidence": [
                    {
                        "pageNumber": 1,
                        "rawValue": "HBL-001",
                        "ocrExcerpt": "B/L NO HBL-001",
                    }
                ],
                "imageUse": "not_used",
            }
        ],
        "relationEvidence": [
            {
                "groupId": "g1",
                "coverage": "one_to_one_package_allocations",
                "relationshipBasis": "quantity_reconciliation",
                "rawOcrEvidence": [
                    {
                        "pageNumber": 1,
                        "rawValue": "10 CARTONS",
                        "ocrExcerpt": "10 CARTONS",
                    }
                ],
                "pdfUse": "not_used",
                "note": "The printed allocation quantity matches the sole package level.",
            }
        ],
        "warnings": [],
        "decisionNotes": ["Both cargo views preserve the same source facts."],
    }
    assert AnnotationDraft.model_validate_json(
        canonical_json_bytes(draft_value), strict=True
    ).normalLabel == normal

    categorized = json.loads(json.dumps(draft_value))
    categorized["relationExplicitLabel"]["documentPatch"]["cargoPackages"][0].pop(
        "typeDescription"
    )
    categorized["relationExplicitLabel"]["documentPatch"]["cargoPackages"][0][
        "typeCategory"
    ] = "CARTON"
    with pytest.raises(ValidationError, match="category mapping is downstream"):
        AnnotationDraft.model_validate_json(
            canonical_json_bytes(categorized), strict=True
        )


def test_relation_audit_distinguishes_direct_container_rows_from_combined_levels() -> None:
    direct = _relation_audit(
        "doc_" + "a" * 64,
        0,
        {
            "packages": [{"quantity": 4}, {"quantity": 5}],
            "containerAllocations": [
                {"containerNumber": "MSKU1200040", "packageQuantity": 4},
                {"containerNumber": "MSKU1200041", "packageQuantity": 5},
            ],
        },
    )
    combined = _relation_audit(
        "doc_" + "b" * 64,
        0,
        {
            "packages": [{"quantity": 2383}, {"quantity": 633}],
            "containerAllocations": [{"containerNumber": "MSKU1200040", "packageQuantity": 3016}],
        },
    )

    assert direct.classification == "one_to_one_package_allocations"
    assert direct.matched_package_indexes == (0, 1)
    assert combined.classification == "all_package_levels_combined"
    assert _valid_reviewed_relation_choices(
        {
            "packages": [{"quantity": 4}, {"quantity": 5}],
            "containerAllocations": [
                {"containerNumber": "MSKU1200040", "packageQuantity": 4},
                {"containerNumber": "FCIU3651201", "packageQuantity": 5},
            ],
        }
    ) == {"one_to_one_package_allocations", "all_package_levels_combined"}


def test_one_to_one_relation_rejects_wrong_package_quantity_link() -> None:
    invalid = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "containers": [
                {"containerNumber": "MSKU1200040"},
                {"containerNumber": "FCIU3651201"},
            ],
            "cargoGroups": [{"groupId": "g1", "description": "COILS"}],
            "cargoPackages": [
                {"packageId": "p1", "groupId": "g1", "quantity": 4},
                {"packageId": "p2", "groupId": "g1", "quantity": 5},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1", "p2"],
                    "allocations": [
                        {
                            "containerNumber": "MSKU1200040",
                            "packageQuantity": 5,
                            "packageId": "p1",
                        },
                        {
                        "containerNumber": "FCIU3651201",
                            "packageQuantity": 4,
                            "packageId": "p2",
                        },
                    ],
                }
            ],
        },
    }
    with pytest.raises(ValidationError, match="quantity differs from its package fact"):
        BillOfLadingRelationExplicitLabel.model_validate_json(
            canonical_json_bytes(invalid), strict=True
        )


def test_one_container_can_carry_multiple_explicit_package_levels() -> None:
    value = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "containers": [{"containerNumber": "MSKU1200040"}],
            "cargoGroups": [{"groupId": "g1", "description": "FILTERS"}],
            "cargoPackages": [
                {"packageId": "p1", "groupId": "g1", "quantity": 24},
                {"packageId": "p2", "groupId": "g1", "quantity": 956},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1", "p2"],
                    "allocations": [
                        {
                            "containerNumber": "MSKU1200040",
                            "packageQuantity": 24,
                            "packageId": "p1",
                        },
                        {
                            "containerNumber": "MSKU1200040",
                            "packageQuantity": 956,
                            "packageId": "p2",
                        },
                    ],
                }
            ],
        },
    }

    parsed = BillOfLadingRelationExplicitLabel.model_validate_json(
        canonical_json_bytes(value), strict=True
    )

    allocations = parsed.documentPatch.cargoAllocationGroups
    assert allocations is not None
    assert tuple(row.containerNumber for row in allocations[0].allocations) == (
        "MSKU1200040",
        "MSKU1200040",
    )
    normal = project_relation_to_normal(parsed)
    normal_allocations = normal.documentPatch.goodsItems[0].containerAllocations
    assert normal_allocations is not None
    assert tuple(row.packageQuantity for row in normal_allocations) == (24, 956)


def test_relation_explicit_training_task_and_semantic_prompt_are_registered() -> None:
    project_root = Path(__file__).resolve().parents[1]
    task = get_training_task("bill_of_lading_relation_explicit_v3")
    prompt_config = PromptConfig.model_validate(
        {
            "path": "prompts/kie/bill_of_lading_relation_explicit_v3.txt",
            "placeholder": "{{document_text}}",
            "schema_placeholder": "{{output_schema}}",
        },
        strict=True,
    )
    prompt = load_prompt(project_root, prompt_config, task)
    rendered = prompt.render("--- PAGE 1 ---\nRAW OCR")

    assert '"const":"3.0.0-experimental"' in rendered
    assert "single_package_level" in rendered
    assert "FLAMMABLE_LIQUIDS" in rendered
    assert "Country/locality strings remain printed" in rendered
    assert "--- PAGE 1 ---\nRAW OCR" in rendered


def test_relation_explicit_allocation_package_ids_must_be_source_ordered() -> None:
    with pytest.raises(ValidationError, match="packageIds must be source ordered"):
        CargoAllocationGroup.model_validate(
            {
                "groupId": "g1",
                "coverage": "one_to_one_package_allocations",
                "packageIds": ("p2", "p1"),
                "allocations": (
                    {
                        "containerNumber": "MSKU1200040",
                        "packageQuantity": 20,
                        "packageId": "p2",
                    },
                    {
                        "containerNumber": "FCIU3651201",
                        "packageQuantity": 10,
                        "packageId": "p1",
                    },
                ),
            },
            strict=True,
        )
