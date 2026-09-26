from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v6 import project_relation_v5_target_to_v6
from document_ocr.training.metrics import structured_metrics
from document_ocr.training.mpci_aligned_projection import migrate_dataset
from document_ocr.training.tasks import canonical_json, get_training_task


def _source_target(*, duplicate: bool) -> dict[str, object]:
    packages = [
        {
            "packageId": "p1",
            "groupId": "g1",
            "quantity": 22 if duplicate else 47,
            "typeCategory": "PACKAGE_BAG",
        }
    ]
    if duplicate:
        packages.append(
            {"packageId": "p2", "groupId": "g1", "quantity": 2, "typeCategory": "PACKAGE_BAG"}
        )
    allocations = [{"containerNumber": "TTNU8111930", "packageQuantity": 22 if duplicate else 6}]
    if duplicate:
        allocations[0]["packageId"] = "p1"
        allocations.append(
            {"containerNumber": "TTNU8111930", "packageQuantity": 2, "packageId": "p2"}
        )
    return {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [{"containerNumber": "TTNU8111930"}],
            "cargoGroups": [{"groupId": "g1", "description": "BAGGED CARGO"}],
            "cargoPackages": packages,
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": (
                        "one_to_one_package_allocations"
                        if duplicate
                        else "unlinked_package_quantities"
                    ),
                    "packageIds": ["p1", "p2"] if duplicate else [],
                    "allocations": allocations,
                }
            ],
        },
    }


@pytest.mark.parametrize("duplicate", [False, True])
def test_projection_keeps_independent_printed_facts_and_order(duplicate: bool) -> None:
    source = _source_target(duplicate=duplicate)
    target = project_relation_v5_target_to_v6(source)
    assert target["schemaVersion"] == "6.0.0-experimental"
    patch = target["documentPatch"]
    assert "cargoGroups" not in patch
    assert "cargoAllocationGroups" not in patch
    assert patch["containerInformation"][0]["equipmentIdentifier"] == "TTNU8111930"
    goods = patch["goodsItemDetails"][0]
    assert "groupId" not in goods
    assert "coverage" not in json.dumps(target)
    assert list(goods)[-1] == "splitGoodsPlacement"
    assert [row["packageQuantity"] for row in goods["splitGoodsPlacement"]] == (
        [22, 2] if duplicate else [6]
    )
    assert [row["packageQuantity"] for row in goods["numberAndTypeOfPackages"]] == (
        [22, 2] if duplicate else [47]
    )
    task = get_training_task("bill_of_lading_mpci_aligned_v6")
    assert task.canonicalize(target) == target
    schema = json.loads(task.prompt_schema_json())
    assert list(schema["$defs"]["GoodsItemDetailsV6"]["properties"])[-1] == ("splitGoodsPlacement")
    assert canonical_json(target).find("numberAndTypeOfPackages") < canonical_json(target).find(
        "splitGoodsPlacement"
    )


def test_new_relation_metrics_measure_nested_placements() -> None:
    task = get_training_task("bill_of_lading_mpci_aligned_v6")
    reference = project_relation_v5_target_to_v6(_source_target(duplicate=True))
    predicted = json.loads(json.dumps(reference))
    predicted["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][1][
        "packageQuantity"
    ] = 3
    scores, assessments = structured_metrics(
        [canonical_json(predicted)], [canonical_json(reference)], task
    )
    assert scores["cargo_relation_f1"] < 1
    assert scores["category_value_f1"] == 1
    assert len(assessments[0].reference_cargo_relation_facts) == 3


def test_projection_preserves_membership_only_shared_container_and_package_fallback() -> None:
    source = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [{"containerNumber": "TTNU8111930"}],
            "cargoGroups": [
                {"groupId": "g1"},
                {"groupId": "g2", "description": "SECOND ITEM"},
            ],
            "cargoPackages": [
                {"packageId": "p1", "groupId": "g1", "typeDescription": "PACKAGES"}
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "container_membership_only",
                    "packageIds": [],
                    "allocations": [{"containerNumber": "TTNU8111930"}],
                },
                {
                    "groupId": "g2",
                    "coverage": "container_membership_only",
                    "packageIds": [],
                    "allocations": [{"containerNumber": "TTNU8111930"}],
                },
            ],
        },
    }
    target = project_relation_v5_target_to_v6(source)
    first, second = target["documentPatch"]["goodsItemDetails"]
    assert first["numberAndTypeOfPackages"] == [{"typeOfPackages": "PACKAGES"}]
    assert first["splitGoodsPlacement"] == [{"equipmentIdentifier": "TTNU8111930"}]
    assert second["splitGoodsPlacement"] == [{"equipmentIdentifier": "TTNU8111930"}]
    assert "numberAndTypeOfPackages" not in second

    container_only = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"containers": [{"containerNumber": "TTNU8111930"}]},
    }
    assert "goodsItemDetails" not in project_relation_v5_target_to_v6(container_only)[
        "documentPatch"
    ]


def test_published_derivative_pins_parent_and_refuses_overwrite(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    source = _source_target(duplicate=False)
    row = {
        "documentId": "doc_" + "a" * 64,
        "joinedRawText": "BAGGED CARGO 47 BAGS TTNU8111930 6 BAGS",
        "target": source,
    }
    row["joinedRawTextSha256"] = hashlib.sha256(row["joinedRawText"].encode()).hexdigest()
    for split in ("train", "validation"):
        row["documentId"] = "doc_" + ("a" if split == "train" else "b") * 64
        (parent / f"{split}.jsonl").write_bytes(canonical_json_bytes(row) + b"\n")
    (parent / "lineage.jsonl").write_bytes(b"{}\n")
    manifest = {
        "outputs": {
            split: {"sha256": sha256_file(parent / f"{split}.jsonl"), "records": 1}
            for split in ("train", "validation")
        }
    }
    (parent / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    manifest_hash = sha256_file(parent / "manifest.json")
    output = tmp_path / "derived"
    result = migrate_dataset(
        source_root=parent, output_root=output, expected_manifest_sha256=manifest_hash
    )
    assert result["outputs"]["train"]["records"] == 1
    assert result["retainedFacts"]["placements"] == 2
    assert sha256_file(parent / "train.jsonl") == manifest["outputs"]["train"]["sha256"]
    assert (output / "source-target-receipts.jsonl").is_file()
    with pytest.raises(ValueError, match="will not be overwritten"):
        migrate_dataset(
            source_root=parent, output_root=output, expected_manifest_sha256=manifest_hash
        )

    # A later source publication with train/validation ID overlap must fail
    # before publishing any derivative dataset.
    (parent / "validation.jsonl").write_bytes((parent / "train.jsonl").read_bytes())
    manifest["outputs"]["validation"]["sha256"] = sha256_file(parent / "validation.jsonl")
    (parent / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="duplicate documentId"):
        migrate_dataset(
            source_root=parent,
            output_root=rejected,
            expected_manifest_sha256=sha256_file(parent / "manifest.json"),
        )
    assert not rejected.exists()
