from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.training.config import DatasetPartitionConfig
from document_ocr.training.splitting import partition_dataset


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
