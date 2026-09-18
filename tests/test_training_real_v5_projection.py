from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.training.config import RealV5ProjectionConfig
from document_ocr.training.real_v5_projection import project_real_v5_split


def _record(document_id: str, raw_text: str, target: dict[str, Any]) -> dict[str, Any]:
    return {
        "documentId": document_id,
        "joinedRawText": raw_text,
        "joinedRawTextSha256": sha256_bytes(raw_text.encode("utf-8")),
        "target": target,
    }


def _publish_inputs(
    tmp_path: Path, records: list[dict[str, Any]], validation_ids: list[str]
) -> RealV5ProjectionConfig:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in records))
    validation_sha256 = sha256_bytes(canonical_json_bytes(validation_ids))
    report = tmp_path / "report.json"
    report.write_bytes(
        canonical_json_bytes(
            {
                "inspection": {
                    "partition": {
                        "outputs": {
                            "validation": {
                                "document_ids": validation_ids,
                                "document_ids_sha256": validation_sha256,
                            }
                        }
                    },
                    "source_files": [
                        {"sha256": sha256_file(source), "records": len(records)}
                    ],
                }
            }
        )
    )
    return RealV5ProjectionConfig.model_validate(
        {
            "schema_version": 1,
            "projection": "relation_v3_real_split_to_v5_v1",
            "source": {
                "path": str(source),
                "sha256": sha256_file(source),
                "records": len(records),
            },
            "baseline_dataset_report": {
                "path": str(report),
                "sha256": sha256_file(report),
            },
            "validation": {
                "records": len(validation_ids),
                "document_ids_sha256": validation_sha256,
            },
            "corrections": [],
            "output_dir": str(tmp_path / "output"),
        },
        strict=True,
    )


def test_projection_preserves_exact_validation_membership_and_migrates_targets(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            "doc_train",
            "--- PAGE 1 ---\nB/L ABC1",
            {
                "schemaVersion": "3.0.0-experimental",
                "documentPatch": {"billOfLadingNumber": "ABC1"},
            },
        ),
        _record(
            "doc_validation",
            "--- PAGE 1 ---\nB/L ABC2",
            {
                "schemaVersion": "3.0.0-experimental",
                "documentPatch": {"billOfLadingNumber": "ABC2"},
            },
        ),
    ]
    config = _publish_inputs(tmp_path, records, ["doc_validation"])

    manifest = project_real_v5_split(
        project_root=tmp_path,
        config=config,
        config_sha256="a" * 64,
    )

    assert manifest["outputs"]["train"]["records"] == 1
    assert manifest["outputs"]["validation"]["records"] == 1
    validation = json.loads((tmp_path / "output" / "validation.jsonl").read_text())
    assert validation["documentId"] == "doc_validation"
    assert validation["target"]["schemaVersion"] == "5.0.0-experimental"


def test_projection_applies_only_exact_source_grounded_equipment_correction(
    tmp_path: Path,
) -> None:
    evidence = (
        "Cargo is stowed in a refrigerated container set at the shipper's requested "
        "carrying temperature of -18 degrees Celsius."
    )
    record = _record(
        "doc_temperature",
        f"--- PAGE 1 ---\n{evidence}",
        {
            "schemaVersion": "3.0.0-experimental",
            "documentPatch": {
                "containers": [
                    {
                        "containerNumber": "TTNU8111930",
                        "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
                    }
                ]
            },
        },
    )
    config = _publish_inputs(tmp_path, [record], ["doc_temperature"])
    value = config.model_dump(mode="python")
    value["corrections"] = [
        {
            "document_id": "doc_temperature",
            "input_sha256": record["joinedRawTextSha256"],
            "evidence_text": evidence,
            "container_index": 0,
            "type_description": "refrigerated container",
        }
    ]
    config = RealV5ProjectionConfig.model_validate(value, strict=True)

    project_real_v5_split(project_root=tmp_path, config=config, config_sha256="b" * 64)

    output = json.loads((tmp_path / "output" / "validation.jsonl").read_text())
    container = output["target"]["documentPatch"]["containers"][0]
    assert container["typeDescription"] == "refrigerated container"
    assert container["temperatureSetpoint"] == {"unit": "celsius", "value": -18.0}


def test_projection_rejects_non_literal_correction_evidence(tmp_path: Path) -> None:
    evidence = "Cargo is stowed in a refrigerated container."
    record = _record(
        "doc_temperature",
        f"--- PAGE 1 ---\n{evidence}",
        {
            "schemaVersion": "3.0.0-experimental",
            "documentPatch": {
                "containers": [
                    {
                        "containerNumber": "TTNU8111930",
                        "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
                    }
                ]
            },
        },
    )
    config = _publish_inputs(tmp_path, [record], ["doc_temperature"])
    value = config.model_dump(mode="python")
    value["corrections"] = [
        {
            "document_id": "doc_temperature",
            "input_sha256": record["joinedRawTextSha256"],
            "evidence_text": "refrigerated container set",
            "container_index": 0,
            "type_description": "refrigerated container",
        }
    ]
    config = RealV5ProjectionConfig.model_validate(value, strict=True)

    with pytest.raises(ValueError, match="evidence must occur exactly once"):
        project_real_v5_split(project_root=tmp_path, config=config, config_sha256="c" * 64)
