from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingExclusion
from document_ocr.labeling_agents.dataset_merge import (
    DatasetMergeConfig,
    DatasetMergeError,
    merge_label_datasets,
)
from document_ocr.labeling_agents.models import CompactAnnotationDraft, NeedsReviewRecord
from document_ocr.labeling_agents.orchestrator import build_compact_annotation
from document_ocr.labeling_agents.work_items import AgentWorkItem


def _work_item(document_id: str, number: str) -> AgentWorkItem:
    page_text = f"B/L NO: {number}"
    joined = f"--- PAGE 1 ---\n{page_text}"
    return AgentWorkItem.model_validate(
        {
            "source": {
                "documentId": document_id,
                "extractionRunId": "fixture-extraction",
                "sourceUri": f"file:///{document_id}.pdf",
                "localCanonicalPath": f"/{document_id}.pdf",
                "sourceSha256": "d" * 64,
                "documentPageCount": 1,
                "joinedRawTextSha256": sha256_bytes(joined.encode()),
                "pages": (
                    {
                        "pageIndex": 0,
                        "pageNumber": 1,
                        "pageId": f"page-{document_id[-1]}",
                        "extractionId": f"extract-{document_id[-1]}",
                        "rawOcrTextSha256": sha256_bytes(page_text.encode()),
                        "rawResponsePath": f"raw-responses/{document_id}.json",
                        "rawResponseSha256": "e" * 64,
                        "rasterPath": f"page-images/{document_id}.png",
                        "rasterSha256": "f" * 64,
                    },
                ),
            },
            "joinedRawText": joined,
        },
        strict=True,
    )


def _publish_source_dataset(
    root: Path,
    *,
    dataset_id: str,
    document_id: str,
    status: str,
) -> None:
    root.mkdir(parents=True)
    item = _work_item(document_id, f"HBL-{document_id[-1]}")
    selection_row = {
        "documentId": document_id,
        "documentPageCount": 1,
        "workItemSha256": sha256_bytes(
            canonical_json_bytes(item.model_dump(mode="json"))
        ),
    }
    selection_payload = canonical_json_bytes(selection_row) + b"\n"
    training_rows: list[dict[str, Any]] = []
    if status == "validated":
        draft = CompactAnnotationDraft.model_validate(
            {
                "decision": "annotation",
                "documentType": "bill_of_lading",
                "relationExplicitLabel": {
                    "schemaVersion": "3.0.0-experimental",
                    "documentPatch": {"billOfLadingNumber": f"HBL-{document_id[-1]}"},
                },
                "warnings": (),
                "decisionNotes": ("Fixture label.",),
            },
            strict=True,
        )
        annotation = build_compact_annotation(
            item, draft, pdf_grouping_used=False
        ).model_copy(update={"reviewStatus": "validated"})
        artifact_payload = canonical_json_bytes(annotation.model_dump(mode="json"))
        artifact_relative = f"validated/{document_id}.json"
        artifact_sha256 = sha256_bytes(artifact_payload)
        training_rows.append(
            {
                "documentId": document_id,
                "joinedRawText": item.joinedRawText,
                "joinedRawTextSha256": item.source.joinedRawTextSha256,
                "target": annotation.relationExplicitLabel.canonical_target(),
                "normalTarget": annotation.normalLabel.canonical_target(),
                "validatedAnnotationPath": artifact_relative,
                "validatedAnnotationSha256": artifact_sha256,
            }
        )
    elif status == "needs_review":
        record = NeedsReviewRecord.model_validate(
            {
                "schemaVersion": 1,
                "documentId": document_id,
                "workItemSha256": selection_row["workItemSha256"],
                "attempts": 2,
                "reviews": 2,
                "documentEscalations": 0,
                "reason": "candidate_attempts_exhausted",
                "findings": ("Fixture review hold.",),
            },
            strict=True,
        )
        artifact_payload = canonical_json_bytes(record.model_dump(mode="json"))
        artifact_relative = f"needs-review/{document_id}.json"
        artifact_sha256 = sha256_bytes(artifact_payload)
    elif status == "excluded":
        exclusion = BillOfLadingExclusion.model_validate(
            {
                "exclusionSchemaVersion": "2.0.0",
                "taskType": "bill_of_lading_kie",
                "source": item.source,
                "reason": "not_bill_of_lading_or_sea_waybill",
                "rawOcrEvidence": (
                    {
                        "pageNumber": 1,
                        "rawValue": "B/L NO",
                        "ocrExcerpt": item.joinedRawText,
                    },
                ),
                "reviewStatus": "rejected",
                "reviewNotes": ("Fixture exclusion.",),
            },
            strict=True,
        )
        artifact_payload = canonical_json_bytes(exclusion.model_dump(mode="json"))
        artifact_relative = f"exclusions/{document_id}.json"
        artifact_sha256 = sha256_bytes(artifact_payload)
    else:
        raise ValueError(f"unsupported fixture status: {status}")

    artifact_path = root / artifact_relative
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(artifact_payload)
    training_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in training_rows)
    lineage_payload = canonical_json_bytes(
        {
            "documentId": document_id,
            "consolidatedStatus": status,
            "consolidatedArtifactPath": artifact_relative,
            "consolidatedArtifactSha256": artifact_sha256,
            "selectedSourceRunId": f"{dataset_id}-source",
        }
    ) + b"\n"
    files = {
        "selection": ("selection.jsonl", selection_payload, 1),
        "training_records": ("training/records.jsonl", training_payload, len(training_rows)),
        "lineage": ("lineage.jsonl", lineage_payload, 1),
    }
    manifest_files: list[dict[str, Any]] = []
    for kind, (relative, payload, rows) in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest_files.append(
            {
                "kind": kind,
                "path": relative,
                "rows": rows,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    counts = {
        "validated": int(status == "validated"),
        "excluded": int(status == "excluded"),
        "needs_review": int(status == "needs_review"),
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "datasetId": dataset_id,
                "task": "bill_of_lading_relation_single_source_v4",
                "selectedDocuments": 1,
                "selectedPages": 1,
                "outcomes": counts,
                "trainingRecords": len(training_rows),
                "trainingReady": counts["needs_review"] == 0,
                "files": manifest_files,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    sources: list[tuple[str, Path]],
    *,
    dataset_id: str = "merged-v1",
) -> DatasetMergeConfig:
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    return DatasetMergeConfig.model_validate(
        {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "output_root": str(output),
            "source_datasets": [
                {"dataset_id": dataset_id, "root": str(root)}
                for dataset_id, root in sources
            ],
        },
        strict=True,
    )


def test_merge_preserves_disjoint_outcomes_lineage_and_readiness(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_id = "doc_" + "a" * 64
    second_id = "doc_" + "b" * 64
    _publish_source_dataset(
        first, dataset_id="first-v1", document_id=first_id, status="validated"
    )
    _publish_source_dataset(
        second, dataset_id="second-v1", document_id=second_id, status="needs_review"
    )

    manifest_path = merge_label_datasets(
        _config(tmp_path, [("first-v1", first), ("second-v1", second)])
    )
    rerun_path = merge_label_datasets(
        _config(tmp_path, [("first-v1", first), ("second-v1", second)])
    )

    assert rerun_path == manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == 2
    assert manifest["selectedDocuments"] == 2
    assert manifest["outcomes"] == {"validated": 1, "excluded": 0, "needs_review": 1}
    assert manifest["trainingRecords"] == 1
    assert manifest["trainingReady"] is False
    root = manifest_path.parent
    training = [
        json.loads(line)
        for line in (root / "training" / "records.jsonl").read_text().splitlines()
    ]
    lineage = [json.loads(line) for line in (root / "lineage.jsonl").read_text().splitlines()]
    assert [row["documentId"] for row in training] == [first_id]
    assert [row["sourceDatasetId"] for row in lineage] == ["first-v1", "second-v1"]
    assert (root / "validated" / f"{first_id}.json").is_file()
    assert (root / "needs-review" / f"{second_id}.json").is_file()


def test_merge_preserves_the_consolidation_exclusions_directory_contract(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_id = "doc_" + "a" * 64
    second_id = "doc_" + "b" * 64
    _publish_source_dataset(
        first, dataset_id="first-v1", document_id=first_id, status="validated"
    )
    _publish_source_dataset(
        second, dataset_id="second-v1", document_id=second_id, status="excluded"
    )

    manifest_path = merge_label_datasets(
        _config(tmp_path, [("first-v1", first), ("second-v1", second)])
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["outcomes"] == {"validated": 1, "excluded": 1, "needs_review": 0}
    assert manifest["trainingReady"] is True
    assert (manifest_path.parent / "exclusions" / f"{second_id}.json").is_file()


def test_merge_rejects_duplicate_document_ids_across_sources(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    document_id = "doc_" + "a" * 64
    _publish_source_dataset(
        first, dataset_id="first-v1", document_id=document_id, status="validated"
    )
    _publish_source_dataset(
        second, dataset_id="second-v1", document_id=document_id, status="validated"
    )

    with pytest.raises(DatasetMergeError, match="duplicate document IDs"):
        merge_label_datasets(
            _config(tmp_path, [("first-v1", first), ("second-v1", second)])
        )


def test_merged_dataset_can_be_a_source_without_losing_origin_lineage(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"
    first_id = "doc_" + "a" * 64
    second_id = "doc_" + "b" * 64
    third_id = "doc_" + "c" * 64
    _publish_source_dataset(
        first, dataset_id="first-v1", document_id=first_id, status="validated"
    )
    _publish_source_dataset(
        second, dataset_id="second-v1", document_id=second_id, status="needs_review"
    )
    _publish_source_dataset(
        third, dataset_id="third-v1", document_id=third_id, status="excluded"
    )
    first_merge = merge_label_datasets(
        _config(tmp_path, [("first-v1", first), ("second-v1", second)])
    )

    second_merge = merge_label_datasets(
        _config(
            tmp_path,
            [("merged-v1", first_merge.parent), ("third-v1", third)],
            dataset_id="merged-v2",
        )
    )

    lineage = [
        json.loads(line)
        for line in (second_merge.parent / "lineage.jsonl").read_text().splitlines()
    ]
    first_chain = lineage[0]["sourceDatasetLineage"]
    assert [row["datasetId"] for row in first_chain] == ["first-v1", "merged-v1"]
    assert lineage[2]["sourceDatasetId"] == "third-v1"
