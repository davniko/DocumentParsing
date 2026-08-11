from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pypdf import PdfWriter

from document_ocr.config import (
    ClassificationManifestConfig,
    PageCountQuarantineConfig,
    S3LocalSnapshotConfig,
    SnapshotDownloadConfig,
)
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.snapshot import (
    CandidateSelectionRecord,
    SnapshotIntegrityError,
    SnapshotSelectionError,
    materialize_snapshot,
    verify_snapshot,
)
from document_ocr.sources import discover_sources

NOW = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
BUCKET = "snapshot-fixture-bucket"
PREFIX = "raw/"
MANIFEST_KEY = "classified/manifest.jsonl"


def _pdf(page_count: int) -> bytes:
    stream = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    writer.write(stream)
    return stream.getvalue()


class _Paginator:
    def __init__(self, owner: _FakeS3) -> None:
        self.owner = owner

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.owner.list_calls.append(kwargs)
        return [{"Contents": list(self.owner.listing)}]


class _FakeS3:
    def __init__(
        self,
        *,
        manifest_payload: bytes,
        pdfs: dict[str, bytes],
    ) -> None:
        self.manifest_payload = manifest_payload
        self.pdfs = pdfs
        self.manifest_etag = '"manifest-etag"'
        self.pdf_etags = {key: f'"etag-{index}"' for index, key in enumerate(pdfs)}
        self.listing = [
            {
                "Key": key,
                "Size": len(payload),
                "ETag": self.pdf_etags[key],
                "LastModified": NOW,
            }
            for key, payload in reversed(tuple(pdfs.items()))
        ]
        self.list_calls: list[dict[str, Any]] = []
        self.head_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> _Paginator:
        assert name == "list_objects_v2"
        return _Paginator(self)

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.head_calls.append(kwargs)
        assert kwargs["Key"] == MANIFEST_KEY
        return {
            "ContentLength": len(self.manifest_payload),
            "ETag": self.manifest_etag,
            "LastModified": NOW,
        }

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        key = kwargs["Key"]
        if key == MANIFEST_KEY:
            assert kwargs["IfMatch"] == self.manifest_etag
            payload = self.manifest_payload
            etag = self.manifest_etag
        else:
            payload = self.pdfs[key]
            etag = self.pdf_etags[key]
            assert kwargs["IfMatch"] == etag
        return {
            "ContentLength": len(payload),
            "ETag": etag,
            "LastModified": NOW,
            "ChecksumCRC64NVME": "fixture-checksum",
            "ChecksumType": "FULL_OBJECT",
            "Body": io.BytesIO(payload),
        }


def _classification_payload(*, blc_pages: int = 1) -> bytes:
    rows = []
    for page_index in range(blc_pages):
        rows.append(
            {
                "doc_id": "aci_blc/blc-document",
                "original_pdf_path": "/workspace/raw/source/blc-document.pdf",
                "page_index": page_index,
                "is_first_page": page_index == 0,
                "final_classification_label": "blc",
            }
        )
    rows.append(
        {
            "doc_id": "standard_swb/swb-document",
            "original_pdf_path": "/workspace/raw/source/swb-document.pdf",
            "page_index": 0,
            "is_first_page": True,
            "final_classification_label": "swb",
        }
    )
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _selection_digest(pdfs: dict[str, bytes], *, blc_pages: int = 1) -> str:
    labels = {
        "blc-document.pdf": ("blc", "aci_blc/blc-document", blc_pages),
        "swb-document.pdf": ("swb", "standard_swb/swb-document", 1),
    }
    records = []
    for index, (key, payload) in enumerate(pdfs.items()):
        name = Path(key).name
        label, doc_id, pages = labels[name]
        records.append(
            CandidateSelectionRecord(
                source_bucket=BUCKET,
                source_key=key,
                source_size_bytes=len(payload),
                source_etag=f'"etag-{index}"',
                source_last_modified=NOW.isoformat(),
                document_type=label,
                classification_doc_id=doc_id,
                classification_manifest_key=MANIFEST_KEY,
                document_page_count=pages,
            )
        )
    records.sort(key=lambda item: item.source_key)
    return sha256_bytes(
        b"".join(canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in records)
    )


def _config(
    root: Path,
    manifest_payload: bytes,
    pdfs: dict[str, bytes],
    *,
    blc_pages: int = 1,
) -> S3LocalSnapshotConfig:
    return S3LocalSnapshotConfig(
        schema_version=1,
        bucket=BUCKET,
        raw_prefix=PREFIX,
        region="us-east-1",
        destination_root=str(root),
        document_types=["blc", "swb"],
        classification_label_mapping={"blc": "blc", "swb": "swb"},
        classification_manifests=[
            ClassificationManifestConfig(
                key=MANIFEST_KEY,
                expected_sha256=sha256_bytes(manifest_payload),
                local_name="classifier.jsonl",
            )
        ],
        expected_selection_sha256=_selection_digest(pdfs, blc_pages=blc_pages),
        page_count_quarantine=[],
        download=SnapshotDownloadConfig(
            workers=2,
            chunk_size_bytes=64 * 1024,
            page_inspection_processes=2,
            max_attempts=2,
            max_pdf_bytes=1_000_000,
        ),
    )


def _fixture(
    tmp_path: Path,
    *,
    manifest_blc_pages: int = 1,
    pdf_blc_pages: int = 1,
) -> tuple[S3LocalSnapshotConfig, _FakeS3]:
    payload = _classification_payload(blc_pages=manifest_blc_pages)
    pdfs = {
        "raw/source/blc-document.pdf": _pdf(pdf_blc_pages),
        "raw/source/swb-document.pdf": _pdf(1),
    }
    return (
        _config(tmp_path / "snapshot", payload, pdfs, blc_pages=manifest_blc_pages),
        _FakeS3(manifest_payload=payload, pdfs=pdfs),
    )


def test_materialize_and_verify_snapshot_preserve_joinable_provenance(
    tmp_path: Path,
) -> None:
    config, s3 = _fixture(tmp_path)

    result = materialize_snapshot(config, s3_client=s3)

    assert result.created is True
    assert result.commit.document_count == 2
    assert result.commit.actual_pages == 2
    assert set(result.commit.by_document_type) == {"blc", "swb"}
    manifest_rows = [
        json.loads(line)
        for line in (result.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["source_key"] for row in manifest_rows} == set(s3.pdfs)
    assert {row["classification_doc_id"] for row in manifest_rows} == {
        "aci_blc/blc-document",
        "standard_swb/swb-document",
    }
    assert all(row["source_sha256"] for row in manifest_rows)
    assert all(call["IfMatch"] for call in s3.get_calls)

    verified = verify_snapshot(config)
    assert verified.created is False
    assert verified.commit == result.commit

    # The extraction discovery contract sees the same relative key used by the
    # snapshot manifest; it is therefore a direct, lossless provenance join.
    from document_ocr.config import LocalSourceConfig

    discovered = discover_sources(
        LocalSourceConfig(
            type="local",
            root=str(result.root / "files" / "blc"),
            include_glob="**/*.pdf",
            dataset_version=result.commit.manifest_sha256,
            require_content_sha256=True,
        ),
        max_pdf_bytes=config.download.max_pdf_bytes,
    )
    assert [item.local_relative_key for item in discovered] == ["source/blc-document.pdf"]
    assert discovered[0].source_sha256 == next(
        row["source_sha256"] for row in manifest_rows if row["document_type"] == "blc"
    )


def test_materialize_maps_classifier_label_to_explicit_document_type(tmp_path: Path) -> None:
    row = {
        "doc_id": "unclassified/copied-bill",
        "original_pdf_path": "/workspace/raw/copied-bill.pdf",
        "page_index": 0,
        "is_first_page": True,
        "final_classification_label": "unclassified",
    }
    payload = canonical_json_bytes(row) + b"\n"
    pdfs = {"raw/copied-bill.pdf": _pdf(1)}
    candidate = CandidateSelectionRecord(
        source_bucket=BUCKET,
        source_key="raw/copied-bill.pdf",
        source_size_bytes=len(pdfs["raw/copied-bill.pdf"]),
        source_etag='"etag-0"',
        source_last_modified=NOW.isoformat(),
        document_type="blc",
        classification_doc_id="unclassified/copied-bill",
        classification_manifest_key=MANIFEST_KEY,
        document_page_count=1,
    )
    selection_sha256 = sha256_bytes(canonical_json_bytes(candidate.model_dump(mode="json")) + b"\n")
    config = S3LocalSnapshotConfig(
        schema_version=1,
        bucket=BUCKET,
        raw_prefix=PREFIX,
        region="us-east-1",
        destination_root=str(tmp_path / "mapped-snapshot"),
        document_types=["blc"],
        classification_label_mapping={"unclassified": "blc"},
        classification_manifests=[
            ClassificationManifestConfig(
                key=MANIFEST_KEY,
                expected_sha256=sha256_bytes(payload),
                local_name="classifier.jsonl",
            )
        ],
        expected_selection_sha256=selection_sha256,
        page_count_quarantine=[],
        download=SnapshotDownloadConfig(
            workers=1,
            chunk_size_bytes=64 * 1024,
            page_inspection_processes=1,
            max_attempts=1,
            max_pdf_bytes=1_000_000,
        ),
    )

    result = materialize_snapshot(
        config,
        s3_client=_FakeS3(manifest_payload=payload, pdfs=pdfs),
    )

    manifest_row = json.loads((result.root / "manifest.jsonl").read_text().strip())
    assert manifest_row["classification_doc_id"] == "unclassified/copied-bill"
    assert manifest_row["document_type"] == "blc"


def test_download_resume_reuses_only_rehashed_committed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.snapshot as snapshot_module

    config, s3 = _fixture(tmp_path)
    original_inspector = snapshot_module._inspect_page_counts

    def interrupt_after_download(**_: Any) -> tuple[Any, ...]:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(snapshot_module, "_inspect_page_counts", interrupt_after_download)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_snapshot(config, s3_client=s3)
    first_pdf_get_count = sum(call["Key"] in s3.pdfs for call in s3.get_calls)
    assert first_pdf_get_count == 2

    monkeypatch.setattr(snapshot_module, "_inspect_page_counts", original_inspector)
    result = materialize_snapshot(config, s3_client=s3)

    assert result.commit.document_count == 2
    assert sum(call["Key"] in s3.pdfs for call in s3.get_calls) == first_pdf_get_count


def test_verify_rejects_local_pdf_tampering(tmp_path: Path) -> None:
    config, s3 = _fixture(tmp_path)
    result = materialize_snapshot(config, s3_client=s3)
    target = result.root / "files" / "blc" / "source" / "blc-document.pdf"
    target.write_bytes(_pdf(1) + b"tampered")

    with pytest.raises(SnapshotIntegrityError, match="differs from snapshot manifest"):
        verify_snapshot(config)


def test_selection_digest_drift_stops_before_any_pdf_download(tmp_path: Path) -> None:
    config, s3 = _fixture(tmp_path)
    changed = config.model_copy(update={"expected_selection_sha256": "f" * 64})

    with pytest.raises(SnapshotSelectionError, match="selection SHA-256 changed"):
        materialize_snapshot(changed, s3_client=s3)

    assert all(call["Key"] == MANIFEST_KEY for call in s3.get_calls)
    assert not (Path(config.destination_root) / "selection.jsonl").exists()


def test_materialization_rejects_classifier_and_pdf_page_count_mismatch(
    tmp_path: Path,
) -> None:
    config, s3 = _fixture(tmp_path, manifest_blc_pages=2, pdf_blc_pages=1)

    with pytest.raises(SnapshotIntegrityError, match="undeclared mismatch"):
        materialize_snapshot(config, s3_client=s3)

    assert not (Path(config.destination_root) / "snapshot.json").exists()


def test_explicit_content_pinned_page_mismatch_is_quarantined(
    tmp_path: Path,
) -> None:
    config, s3 = _fixture(tmp_path, manifest_blc_pages=2, pdf_blc_pages=1)
    source_key = "raw/source/blc-document.pdf"
    quarantined = config.model_copy(
        update={
            "page_count_quarantine": [
                PageCountQuarantineConfig(
                    source_key=source_key,
                    source_sha256=sha256_bytes(s3.pdfs[source_key]),
                    classified_page_count=2,
                    actual_page_count=1,
                    reason="fixture malformed page tree",
                )
            ]
        }
    )

    result = materialize_snapshot(quarantined, s3_client=s3)

    assert result.commit.quarantined_document_count == 1
    assert result.commit.extraction_document_count == 1
    assert not (result.root / "extraction" / "blc").exists()
    assert (result.root / "files" / "blc" / "source" / "blc-document.pdf").is_file()


def test_extraction_link_publication_fsyncs_once_per_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.snapshot as snapshot_module

    root = tmp_path / "snapshot"
    source = root / "files" / "blc" / "source"
    source.mkdir(parents=True)
    (source / "a.pdf").write_bytes(b"first")
    (source / "b.pdf").write_bytes(b"second")
    records = tuple(
        SimpleNamespace(
            extraction_status="ready",
            snapshot_relative_path=f"files/blc/source/{name}",
            extraction_relative_path=f"extraction/blc/source/{name}",
        )
        for name in ("a.pdf", "b.pdf")
    )
    fsync_calls: list[int] = []
    monkeypatch.setattr(snapshot_module.os, "fsync", fsync_calls.append)

    snapshot_module._publish_extraction_links(root=root, records=records)

    assert len(fsync_calls) == 1
    assert (root / "extraction" / "blc" / "source" / "a.pdf").samefile(source / "a.pdf")
