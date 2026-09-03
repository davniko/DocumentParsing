from __future__ import annotations

import pytest

from document_ocr.synthesis.fit_partition import fit_document_ids, partition_document_ids


def _outputs() -> dict[str, object]:
    return {
        "train": {"records": 2, "document_ids": ["doc_a", "doc_b"]},
        "validation": {"records": 1, "document_ids": ["doc_c"]},
    }


def test_reads_dedicated_split_manifest() -> None:
    report = {
        "schema_version": 1,
        "publication_status": "complete",
        "algorithm": "seeded_sha256_rank_v1",
        "outputs": _outputs(),
    }
    assert fit_document_ids(report) == ("doc_a", "doc_b")


def test_reads_legacy_training_dataset_report() -> None:
    report = {
        "inspection": {
            "partition": {
                "algorithm": "seeded_sha256_rank_v1",
                "outputs": _outputs(),
            }
        }
    }
    assert partition_document_ids(report)["validation"] == ("doc_c",)


def test_rejects_cross_split_overlap() -> None:
    outputs = _outputs()
    outputs["validation"] = {"records": 1, "document_ids": ["doc_b"]}
    report = {
        "schema_version": 1,
        "publication_status": "complete",
        "algorithm": "seeded_sha256_rank_v1",
        "outputs": outputs,
    }
    with pytest.raises(ValueError, match="multiple partition"):
        fit_document_ids(report)
