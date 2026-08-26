from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import document_ocr.catalog_selection as selection_module
from document_ocr.catalog_selection import (
    materialize_catalog_selection,
    plan_catalog_selection,
    verify_catalog_selection,
)
from document_ocr.classification_catalog import (
    CatalogDocumentRecord,
    CatalogLocalAlias,
    FinalDocumentEvidence,
    ManifestDocumentEvidence,
)
from document_ocr.config import CatalogExtractionSelectionConfig
from document_ocr.hashing import sha256_bytes
from document_ocr.snapshot import SnapshotFileRecord

NOW = datetime(2026, 8, 22, 8, 0, tzinfo=UTC).isoformat()
CATALOG_SHA256 = "c" * 64


def _catalog_record(
    tmp_path: Path,
    *,
    filename: str,
    payload: bytes,
    final_label: str,
    dummy: bool = False,
    page_count: int = 1,
) -> CatalogDocumentRecord:
    snapshot_root = (tmp_path / "snapshot").absolute()
    source_path = snapshot_root / "files" / final_label / filename
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(payload)
    source_sha256 = sha256_bytes(payload)
    snapshot_record = SnapshotFileRecord(
        source_bucket="fixture-bucket",
        source_key=f"raw/{final_label}/{filename}",
        source_size_bytes=len(payload),
        source_etag=f'"etag-{filename}"',
        source_last_modified=NOW,
        document_type=final_label,
        classification_doc_id=f"{final_label}/{filename.removesuffix('.pdf')}",
        classification_manifest_key="classified/final.jsonl",
        document_page_count=page_count,
        schema_version=1,
        classification_manifest_sha256="a" * 64,
        local_relative_key=filename,
        snapshot_relative_path=f"files/{final_label}/{filename}",
        source_sha256=source_sha256,
        actual_page_count=page_count,
        extraction_status="ready",
        extraction_relative_path=f"extraction/{final_label}/{filename}",
    )
    alias = CatalogLocalAlias(
        snapshot_config_path=str((tmp_path / "snapshot.yaml").absolute()),
        snapshot_root=str(snapshot_root),
        snapshot_config_sha256="1" * 64,
        snapshot_commit_sha256="2" * 64,
        snapshot_manifest_sha256="3" * 64,
        snapshot_selection_sha256="4" * 64,
        snapshot_record=snapshot_record,
    )
    document_id = f"{final_label}/{filename.removesuffix('.pdf')}"
    initial = ManifestDocumentEvidence(
        manifest_key="classified/initial.jsonl",
        manifest_sha256="5" * 64,
        doc_id=document_id,
        original_pdf_path=f"/workspace/raw/{filename}",
        declared_source_key=f"raw/{final_label}/{filename}",
        classification_label=final_label,
        document_page_count=page_count,
        source_lines_by_page=list(range(1, page_count + 1)),
        classification_llm={},
        dummy={"is_dummy": dummy},
        triage={},
    )
    final = FinalDocumentEvidence(
        manifest_key="classified/final.jsonl",
        manifest_sha256="6" * 64,
        final_classification_label=final_label,
        document_page_count=page_count,
        source_lines_by_page=list(range(1, page_count + 1)),
    )
    return CatalogDocumentRecord(
        schema_version=1,
        catalog_document_id=hashlib.sha256(
            f"{final_label}:{filename}".encode()
        ).hexdigest(),
        lineage="fixture",
        status="retained_final",
        document_filename=filename,
        document_uuid=filename.removeprefix("2024-01-01_").removesuffix(".pdf"),
        initial=initial,
        final=final,
        dropped_report=None,
        relabeled_report=None,
        review_report=None,
        declared_source_present=False,
        s3_source_aliases=[],
        local_snapshot_aliases=[alias],
    )


def _config(tmp_path: Path) -> CatalogExtractionSelectionConfig:
    return CatalogExtractionSelectionConfig(
        schema_version=2,
        catalog_config=str((tmp_path / "catalog.yaml").absolute()),
        expected_catalog_sha256=CATALOG_SHA256,
        destination_root=str((tmp_path / "selection").absolute()),
        final_labels=["blc", "swb"],
        dummy_is_dummy=False,
        excluded_pilots=[],
        deduplicate_by_content_sha256=True,
        max_pages_per_document=2,
        expected_documents=2,
        expected_pages=2,
        expected_manifest_sha256="0" * 64,
        verification_workers=2,
        max_pdf_bytes=1_000_000,
    )


def _verified(tmp_path: Path, records: tuple[CatalogDocumentRecord, ...]) -> Any:
    return SimpleNamespace(
        config_path=(tmp_path / "catalog.yaml").absolute(),
        config_sha256="7" * 64,
        result=SimpleNamespace(commit=SimpleNamespace(catalog_sha256=CATALOG_SHA256)),
        records=records,
    )


def test_selection_deduplicates_content_and_preserves_label_conflicts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    duplicate = b"%PDF-1.7\nbyte-identical BLC/SWB fixture\n"
    records = (
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000001.pdf",
            payload=duplicate,
            final_label="blc",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000002.pdf",
            payload=duplicate,
            final_label="swb",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000003.pdf",
            payload=b"%PDF-1.7\nunique fixture\n",
            final_label="blc",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000004.pdf",
            payload=b"%PDF-1.7\ndummy fixture\n",
            final_label="blc",
            dummy=True,
        ),
    )
    monkeypatch.setattr(
        selection_module,
        "_verified_catalog",
        lambda *_args, **_kwargs: _verified(tmp_path, records),
    )

    planned = plan_catalog_selection(_config(tmp_path))

    assert planned.statistics.catalog_retained_target_documents == 4
    assert planned.statistics.non_dummy_target_documents == 3
    assert planned.statistics.locally_ready_target_documents == 3
    assert planned.statistics.duplicate_catalog_rows_collapsed == 1
    assert planned.statistics.selected_documents == 2
    assert planned.statistics.classification_conflict_documents == 1
    assert planned.statistics.by_classification_set == {"blc": 1, "blc+swb": 1}


def test_selection_audits_and_excludes_documents_above_page_limit(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    records = (
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000021.pdf",
            payload=b"%PDF-1.7\nfirst eligible fixture\n",
            final_label="blc",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000022.pdf",
            payload=b"%PDF-1.7\nsecond eligible fixture\n",
            final_label="swb",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000023.pdf",
            payload=b"%PDF-1.7\nover page limit fixture\n",
            final_label="blc",
            page_count=3,
        ),
    )
    monkeypatch.setattr(
        selection_module,
        "_verified_catalog",
        lambda *_args, **_kwargs: _verified(tmp_path, records),
    )

    planned = plan_catalog_selection(_config(tmp_path))

    assert planned.statistics.content_groups_before_exclusions == 3
    assert planned.statistics.excluded_page_limit_content_groups == 1
    assert planned.statistics.excluded_page_limit_pages == 3
    assert planned.statistics.excluded_by_page_count == {"3": 1}
    assert planned.statistics.selected_documents == 2
    assert planned.statistics.selected_pages == 2


def test_selection_materializes_hard_links_and_is_manifest_last(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    records = (
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000011.pdf",
            payload=b"%PDF-1.7\nfirst fixture\n",
            final_label="blc",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000012.pdf",
            payload=b"%PDF-1.7\nsecond fixture\n",
            final_label="swb",
        ),
    )
    monkeypatch.setattr(
        selection_module,
        "_verified_catalog",
        lambda *_args, **_kwargs: _verified(tmp_path, records),
    )
    unpinned = _config(tmp_path)
    plan = selection_module._build_plan(unpinned, progress=None)
    config = unpinned.model_copy(update={"expected_manifest_sha256": plan.manifest_sha256})
    monkeypatch.setattr(selection_module, "_build_plan", lambda *_args, **_kwargs: plan)

    created = materialize_catalog_selection(config)

    assert created.created is True
    assert (created.root / "selection.json").is_file()
    assert (created.root / "manifest.jsonl").is_file()
    for record in plan.records:
        source = selection_module._source_path(record)
        destination = created.root / record.selection_relative_path
        assert destination.stat().st_ino == source.stat().st_ino
    verified = verify_catalog_selection(config)
    assert verified.created is False
    assert verified.commit == created.commit
