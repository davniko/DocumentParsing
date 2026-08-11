from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

import document_ocr.corpus as corpus_module
from document_ocr.config import (
    CorpusSnapshotSourceConfig,
    LocalCorpusConfig,
)
from document_ocr.corpus import (
    CorpusSelectionError,
    materialize_corpus,
    plan_corpus,
    verify_corpus,
)
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.snapshot import (
    SnapshotCommit,
    SnapshotFileRecord,
    SnapshotResult,
    SnapshotSelectionAudit,
    SnapshotTypeSummary,
    StoredClassificationManifest,
)

NOW = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
UUID = "8b96c127-9e8a-49c8-b9ac-a85336ca98ad"


def _pdf() -> bytes:
    stream = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(stream)
    return stream.getvalue()


def _snapshot_record(
    root: Path,
    *,
    filename: str,
    payload: bytes,
    source_name: str,
) -> SnapshotFileRecord:
    relative_key = f"source/{filename}"
    snapshot_relative_path = f"files/blc/{relative_key}"
    path = root / snapshot_relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return SnapshotFileRecord(
        source_bucket="fixture-corpus-bucket",
        source_key=f"raw/{source_name}/{filename}",
        source_size_bytes=len(payload),
        source_etag=f'"etag-{source_name}-{filename}"',
        source_last_modified=NOW.isoformat(),
        document_type="blc",
        classification_doc_id=f"blc/{filename.removesuffix('.pdf')}",
        classification_manifest_key=f"classification/{source_name}.jsonl",
        document_page_count=1,
        schema_version=1,
        classification_manifest_sha256="c" * 64,
        local_relative_key=relative_key,
        snapshot_relative_path=snapshot_relative_path,
        source_sha256=sha256_bytes(payload),
        actual_page_count=1,
        extraction_status="ready",
        extraction_relative_path=f"extraction/blc/{relative_key}",
    )


def _snapshot_result(
    root: Path,
    records: list[SnapshotFileRecord],
    *,
    source_name: str,
) -> SnapshotResult:
    ordered = sorted(records, key=lambda item: item.source_key)
    manifest_payload = b"".join(
        canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in ordered
    )
    (root / "manifest.jsonl").write_bytes(manifest_payload)
    summary = SnapshotTypeSummary(
        documents=len(ordered),
        source_bytes=sum(item.source_size_bytes for item in ordered),
        classified_pages=len(ordered),
        actual_pages=len(ordered),
        extraction_documents=len(ordered),
        extraction_pages=len(ordered),
        quarantined_documents=0,
    )
    commit = SnapshotCommit(
        schema_version=1,
        bucket="fixture-corpus-bucket",
        raw_prefix=f"raw/{source_name}/",
        region="us-east-1",
        document_types=["blc"],
        selection_path="selection.jsonl",
        selection_sha256=sha256_bytes(source_name.encode()),
        manifest_path="manifest.jsonl",
        manifest_sha256=sha256_bytes(manifest_payload),
        classification_manifests=[
            StoredClassificationManifest(
                key=f"classification/{source_name}.jsonl",
                sha256="c" * 64,
                snapshot_relative_path=f"classification/{source_name}.jsonl",
                source_etag='"manifest-etag"',
                source_last_modified=NOW,
                source_size_bytes=100,
            )
        ],
        selection_audit=SnapshotSelectionAudit(
            raw_pdf_count=len(ordered),
            raw_pdf_bytes=sum(item.source_size_bytes for item in ordered),
            raw_pdfs_without_filtered_manifest_evidence=0,
            filtered_manifest_documents_without_raw_pdf=0,
        ),
        by_document_type={"blc": summary},
        document_count=len(ordered),
        source_bytes=summary.source_bytes,
        classified_pages=len(ordered),
        actual_pages=len(ordered),
        extraction_document_count=len(ordered),
        extraction_pages=len(ordered),
        quarantined_document_count=0,
    )
    (root / "snapshot.json").write_text(
        json.dumps(
            commit.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SnapshotResult(root=root, commit=commit, created=False)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    conflict: bool = False,
) -> LocalCorpusConfig:
    payload = _pdf()
    first_filename = f"2024-01-01_{UUID}.pdf"
    second_filename = f"2024-01-02_{UUID}.pdf"
    first_root = tmp_path / "snapshot-one"
    second_root = tmp_path / "snapshot-two"
    first_records = [
        _snapshot_record(
            first_root,
            filename=first_filename,
            payload=payload,
            source_name="one",
        ),
        _snapshot_record(
            first_root,
            filename=second_filename,
            payload=payload,
            source_name="one",
        ),
    ]
    duplicate_payload = _pdf() + (b"conflict" if conflict else b"")
    second_records = [
        _snapshot_record(
            second_root,
            filename=first_filename,
            payload=duplicate_payload,
            source_name="two",
        )
    ]
    results = {
        "one": _snapshot_result(first_root, first_records, source_name="one"),
        "two": _snapshot_result(second_root, second_records, source_name="two"),
    }
    config_paths = {name: (tmp_path / f"snapshot-{name}.yaml").absolute() for name in results}
    for name, path in config_paths.items():
        path.write_text(f"fixture: {name}\n", encoding="utf-8")

    def fake_load(path: Path) -> str:
        return path.stem.removeprefix("snapshot-")

    def fake_verify(source: Any) -> SnapshotResult:
        return results[str(source)]

    monkeypatch.setattr(corpus_module, "load_snapshot_config", fake_load)
    monkeypatch.setattr(corpus_module, "verify_snapshot", fake_verify)
    return LocalCorpusConfig(
        schema_version=1,
        document_type="blc",
        destination_root=str((tmp_path / "combined").absolute()),
        sources=[
            CorpusSnapshotSourceConfig(snapshot_config=str(config_paths["one"])),
            CorpusSnapshotSourceConfig(snapshot_config=str(config_paths["two"])),
        ],
        expected_manifest_sha256="0" * 64,
        verification_workers=2,
        max_pdf_bytes=1_000_000,
    )


def test_materialize_deduplicates_exact_filename_and_preserves_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpinned = _fixture(tmp_path, monkeypatch)
    digest, document_count = plan_corpus(unpinned)
    config = unpinned.model_copy(update={"expected_manifest_sha256": digest})

    result = materialize_corpus(config)

    assert result.created is True
    assert document_count == 2
    assert result.commit.document_count == 2
    assert result.commit.source_alias_count == 3
    assert result.commit.deduplicated_alias_count == 1
    rows = [json.loads(line) for line in (result.root / "manifest.jsonl").read_text().splitlines()]
    assert [row["document_date"] for row in rows] == ["2024-01-01", "2024-01-02"]
    assert rows[0]["document_uuid"] == rows[1]["document_uuid"] == UUID
    assert [len(row["source_aliases"]) for row in rows] == [2, 1]

    first_corpus_path = result.root / rows[0]["corpus_relative_path"]
    first_source_path = (
        tmp_path
        / "snapshot-one"
        / rows[0]["source_aliases"][0]["snapshot_record"]["snapshot_relative_path"]
    )
    assert (os.stat(first_corpus_path).st_dev, os.stat(first_corpus_path).st_ino) == (
        os.stat(first_source_path).st_dev,
        os.stat(first_source_path).st_ino,
    )

    verified = verify_corpus(config)
    assert verified.created is False
    assert verified.commit == result.commit


def test_plan_rejects_same_filename_with_different_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture(tmp_path, monkeypatch, conflict=True)

    with pytest.raises(CorpusSelectionError, match="same filename has conflicting content"):
        plan_corpus(config)


def test_materialize_resumes_hard_links_before_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpinned = _fixture(tmp_path, monkeypatch)
    digest, _ = plan_corpus(unpinned)
    config = unpinned.model_copy(update={"expected_manifest_sha256": digest})
    original_verify = corpus_module._verify_files

    def interrupt_after_links(*_: Any, **__: Any) -> None:
        raise RuntimeError("simulated interruption after link publication")

    monkeypatch.setattr(corpus_module, "_verify_files", interrupt_after_links)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_corpus(config)
    assert not (Path(config.destination_root) / "corpus.json").exists()
    assert len(tuple((Path(config.destination_root) / "files").rglob("*.pdf"))) == 2

    monkeypatch.setattr(corpus_module, "_verify_files", original_verify)
    resumed = materialize_corpus(config)

    assert resumed.created is True
    assert resumed.commit.document_count == 2
