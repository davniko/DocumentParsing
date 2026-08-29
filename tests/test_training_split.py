from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.training.config import (
    DatasetPartitionConfig,
    RuntimeDatasetPartitionConfig,
)
from document_ocr.training.splitting import (
    PartitionCandidate,
    partition_dataset,
    resolve_validation_records,
    select_runtime_partition,
    target_leaf_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _payload(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for row in rows
    )


def _config(source: Path, payload: bytes, output_dir: Path) -> DatasetPartitionConfig:
    return DatasetPartitionConfig.model_validate(
        {
            "schema_version": 1,
            "source": {
                "path": str(source),
                "sha256": sha256_bytes(payload),
                "records": 10,
            },
            "fields": {
                "document_id": "documentId",
                "input_sha256": "joinedRawTextSha256",
                "target": "target",
            },
            "algorithm": "seeded_sha256_rank_v1",
            "seed": 42,
            "validation_records": 4,
            "coverage_policy": "retain_each_target_leaf_in_train",
            "output_dir": str(output_dir),
        },
        strict=True,
    )


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        return set().union(
            *(
                _leaf_paths(child, f"{prefix}.{key}" if prefix else key)
                for key, child in value.items()
            )
        )
    if isinstance(value, list):
        return set().union(*(_leaf_paths(child, f"{prefix}[]") for child in value))
    return {prefix}


def test_partition_is_seeded_idempotent_and_retains_train_leaf_coverage(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "documentId": f"doc_{index:02d}",
            "joinedRawTextSha256": f"{index + 1:064x}",
            "target": {
                "schemaVersion": "2.0.0",
                "documentPatch": {
                    "billOfLadingNumber": f"BL-{index}",
                    **({"masterBillOfLadingNumber": "ONLY-ONE"} if index == 0 else {}),
                },
            },
        }
        for index in range(10)
    ]
    payload = _payload(rows)
    source = tmp_path / "source.jsonl"
    source.write_bytes(payload)
    config = _config(source, payload, tmp_path / "split")

    first = partition_dataset(project_root=tmp_path, config=config)
    second = partition_dataset(project_root=tmp_path, config=config)

    assert first == second
    assert first["outputs"]["train"]["records"] == 6
    assert first["outputs"]["validation"]["records"] == 4
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "split" / "train.jsonl").read_text().splitlines()
    ]
    validation_rows = [
        json.loads(line)
        for line in (tmp_path / "split" / "validation.jsonl").read_text().splitlines()
    ]
    train_ids = {row["documentId"] for row in train_rows}
    validation_ids = {row["documentId"] for row in validation_rows}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == {row["documentId"] for row in rows}
    all_paths = set().union(*(_leaf_paths(row["target"]) for row in rows))
    train_paths = set().union(*(_leaf_paths(row["target"]) for row in train_rows))
    assert train_paths == all_paths
    assert (tmp_path / "split" / "manifest.json").is_file()


def test_partition_rejects_exact_input_duplicates(tmp_path: Path) -> None:
    rows = [
        {
            "documentId": f"doc_{index:02d}",
            "joinedRawTextSha256": f"{index + 1:064x}",
            "target": {
                "schemaVersion": "2.0.0",
                "documentPatch": {"billOfLadingNumber": f"BL-{index}"},
            },
        }
        for index in range(10)
    ]
    rows[1]["joinedRawTextSha256"] = rows[0]["joinedRawTextSha256"]
    payload = _payload(rows)
    source = tmp_path / "source.jsonl"
    source.write_bytes(payload)

    with pytest.raises(ValueError, match="deduplicate upstream"):
        partition_dataset(
            project_root=tmp_path,
            config=_config(source, payload, tmp_path / "split"),
        )


def test_target_coverage_treats_empty_collections_as_leaves_and_rejects_null() -> None:
    assert target_leaf_paths(
        {
            "documentPatch": {
                "cargoAllocationGroups": [
                    {"allocationId": "a1", "packageIds": []}
                ],
                "metadata": {},
            }
        }
    ) == {
        "documentPatch.cargoAllocationGroups[].allocationId",
        "documentPatch.cargoAllocationGroups[].packageIds",
        "documentPatch.metadata",
    }

    with pytest.raises(ValueError, match="contains null"):
        target_leaf_paths({"documentPatch": {"billOfLadingNumber": None}})


def test_fractional_validation_size_uses_half_up_rounding() -> None:
    config = RuntimeDatasetPartitionConfig.model_validate(
        {
            "algorithm": "seeded_sha256_rank_v1",
            "seed": 42,
            "validation_size": {
                "kind": "fraction",
                "value": 0.25,
                "rounding": "half_up",
            },
            "coverage_policy": "retain_each_target_leaf_in_train",
        },
        strict=True,
    )

    assert resolve_validation_records(config, 10) == 3


def test_runtime_partition_is_deterministic_and_preserves_source_order() -> None:
    candidates = [
        PartitionCandidate(
            document_id=f"doc_{index:02d}",
            input_sha256=f"{index + 1:064x}",
            target_leaf_paths=frozenset(
                {"documentPatch.billOfLadingNumber"}
                | ({"documentPatch.rare"} if index == 0 else set())
            ),
        )
        for index in range(10)
    ]
    config = RuntimeDatasetPartitionConfig.model_validate(
        {
            "algorithm": "seeded_sha256_rank_v1",
            "seed": 42,
            "validation_size": {"kind": "records", "value": 4},
            "coverage_policy": "retain_each_target_leaf_in_train",
        },
        strict=True,
    )

    first = select_runtime_partition(candidates, config)
    second = select_runtime_partition(candidates, config)

    assert first == second
    assert list(first.train_document_ids) == sorted(first.train_document_ids)
    assert list(first.validation_document_ids) == sorted(first.validation_document_ids)
    assert set(first.train_document_ids).isdisjoint(first.validation_document_ids)
    assert len(first.train_document_ids) == 6
    assert len(first.validation_document_ids) == 4
    assert first.to_dict()["outputs"]["validation"]["document_ids_sha256"] == (
        second.to_dict()["outputs"]["validation"]["document_ids_sha256"]
    )


def test_current_combined1109_runtime_partition_regression() -> None:
    source = (
        PROJECT_ROOT
        / "artifacts"
        / "kie-training"
        / "datasets"
        / "mpci-bl-combined1109-current-policy-v1"
        / "records.jsonl"
    )
    candidates = []
    for line in source.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        candidates.append(
            PartitionCandidate(
                document_id=record["documentId"],
                input_sha256=record["joinedRawTextSha256"],
                target_leaf_paths=frozenset(target_leaf_paths(record["target"])),
            )
        )
    config = RuntimeDatasetPartitionConfig.model_validate(
        {
            "algorithm": "seeded_sha256_rank_v1",
            "seed": 42,
            "validation_size": {"kind": "records", "value": 60},
            "coverage_policy": "retain_each_target_leaf_in_train",
        },
        strict=True,
    )

    selection = select_runtime_partition(candidates, config)

    assert len(selection.train_document_ids) == 1049
    assert len(selection.validation_document_ids) == 60
    assert len(selection.coverage_skipped_document_ids) == 1
    report = selection.to_dict()
    assert report["outputs"]["train"]["document_ids_sha256"] == (
        "b2a7a6980eed9b32028bca7c588c0ae9b7a0c04973caea69f5c06f12601b42f9"
    )
    assert report["outputs"]["validation"]["document_ids_sha256"] == (
        "2940aa27c968947506b25cc1e4f9c868b2097cd12592c14ce00e3f8829a3e74d"
    )
