from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import document_ocr.pilot as pilot_module
from document_ocr.classification_catalog import (
    CatalogDocumentRecord,
    CatalogLocalAlias,
    FinalDocumentEvidence,
    ManifestDocumentEvidence,
)
from document_ocr.config import (
    CatalogPilotConfig,
    ExcludedPilotConfig,
    PilotQualityConfig,
    PilotStratumConfig,
)
from document_ocr.hashing import sha256_bytes
from document_ocr.pilot import materialize_pilot, plan_pilot, verify_pilot
from document_ocr.snapshot import SnapshotFileRecord

NOW = datetime(2026, 8, 11, 8, 0, tzinfo=UTC).isoformat()
CATALOG_SHA256 = "c" * 64


def _catalog_record(
    tmp_path: Path,
    *,
    filename: str,
    payload: bytes,
    category: str = "scanned",
    dummy: bool = False,
    readability: str = "fully_readable",
) -> CatalogDocumentRecord:
    snapshot_root = (tmp_path / "snapshot").absolute()
    source_path = snapshot_root / "files" / "blc" / filename
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(payload)
    source_sha256 = sha256_bytes(payload)
    snapshot_record = SnapshotFileRecord(
        source_bucket="fixture-bucket",
        source_key=f"raw/{filename}",
        source_size_bytes=len(payload),
        source_etag=f'"etag-{filename}"',
        source_last_modified=NOW,
        document_type="blc",
        classification_doc_id=f"blc/{filename.removesuffix('.pdf')}",
        classification_manifest_key="classified/final.jsonl",
        document_page_count=1,
        schema_version=1,
        classification_manifest_sha256="a" * 64,
        local_relative_key=filename,
        snapshot_relative_path=f"files/blc/{filename}",
        source_sha256=source_sha256,
        actual_page_count=1,
        extraction_status="ready",
        extraction_relative_path=f"extraction/blc/{filename}",
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
    doc_id = f"blc/{filename.removesuffix('.pdf')}"
    initial = ManifestDocumentEvidence(
        manifest_key="classified/initial.jsonl",
        manifest_sha256="5" * 64,
        doc_id=doc_id,
        original_pdf_path=f"/workspace/raw/{filename}",
        declared_source_key=f"raw/{filename}",
        classification_label="blc",
        document_page_count=1,
        source_lines_by_page=[1],
        classification_llm={},
        dummy={"is_dummy": dummy},
        triage={
            "requires_augmentation": False,
            "details": {
                "augmentation_need": "use_as_is",
                "category": category,
                "readability": readability,
            },
        },
    )
    final = FinalDocumentEvidence(
        manifest_key="classified/final.jsonl",
        manifest_sha256="6" * 64,
        final_classification_label="blc",
        document_page_count=1,
        source_lines_by_page=[1],
    )
    return CatalogDocumentRecord(
        schema_version=1,
        catalog_document_id=hashlib.sha256(filename.encode()).hexdigest(),
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


def _config(tmp_path: Path) -> CatalogPilotConfig:
    return CatalogPilotConfig(
        schema_version=1,
        catalog_config=str((tmp_path / "catalog.yaml").absolute()),
        expected_catalog_sha256=CATALOG_SHA256,
        destination_root=str((tmp_path / "pilot").absolute()),
        final_label="blc",
        document_count=2,
        max_pages_per_document=2,
        selection_namespace="fixture-pilot-v1",
        quality=PilotQualityConfig(
            dummy_is_dummy=False,
            triage_requires_augmentation=False,
            readability="fully_readable",
            augmentation_need="use_as_is",
        ),
        deduplicate_by_content_sha256=True,
        strata=[
            PilotStratumConfig(
                lineage="fixture",
                triage_category="scanned",
                documents=2,
            )
        ],
        expected_manifest_sha256="0" * 64,
        verification_workers=2,
        max_pdf_bytes=1_000_000,
    )


def test_pilot_selection_enforces_quality_and_content_deduplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = b"%PDF-1.7\nduplicate fixture\n"
    records = (
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000001.pdf",
            payload=common,
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000002.pdf",
            payload=common,
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000003.pdf",
            payload=b"%PDF-1.7\nunique fixture\n",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000004.pdf",
            payload=b"%PDF-1.7\ndummy fixture\n",
            dummy=True,
        ),
    )
    verified: Any = SimpleNamespace(
        config_path=(tmp_path / "catalog.yaml").absolute(),
        config_sha256="7" * 64,
        result=SimpleNamespace(commit=SimpleNamespace(catalog_sha256=CATALOG_SHA256)),
        records=records,
    )
    monkeypatch.setattr(pilot_module, "_verified_catalog", lambda *_args, **_kwargs: verified)

    planned = plan_pilot(_config(tmp_path))

    assert planned.statistics.catalog_final_label_local_documents == 4
    assert planned.statistics.quality_eligible_documents == 3
    assert planned.statistics.bounded_eligible_documents == 3
    assert planned.statistics.bounded_unique_content_sha256 == 2
    assert planned.statistics.selected_documents == 2
    assert planned.statistics.selected_unique_content_sha256 == 2


def test_pilot_materialization_is_manifest_last_hard_linked_and_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000011.pdf",
            payload=b"%PDF-1.7\nfirst fixture\n",
        ),
        _catalog_record(
            tmp_path,
            filename="2024-01-01_00000000-0000-4000-8000-000000000012.pdf",
            payload=b"%PDF-1.7\nsecond fixture\n",
        ),
    )
    verified: Any = SimpleNamespace(
        config_path=(tmp_path / "catalog.yaml").absolute(),
        config_sha256="7" * 64,
        result=SimpleNamespace(commit=SimpleNamespace(catalog_sha256=CATALOG_SHA256)),
        records=records,
    )
    monkeypatch.setattr(pilot_module, "_verified_catalog", lambda *_args, **_kwargs: verified)
    unpinned = _config(tmp_path)
    plan = pilot_module._build_plan(unpinned, progress=None)
    config = unpinned.model_copy(update={"expected_manifest_sha256": plan.manifest_sha256})
    monkeypatch.setattr(pilot_module, "_build_plan", lambda *_args, **_kwargs: plan)

    created = materialize_pilot(config)

    assert created.created is True
    assert created.commit.statistics.selected_documents == 2
    assert (created.root / "pilot.json").is_file()
    assert (created.root / "manifest.jsonl").is_file()
    for record in plan.records:
        source = pilot_module._source_path(record)
        destination = created.root / record.pilot_relative_path
        assert destination.stat().st_ino == source.stat().st_ino
    verified_result = verify_pilot(config)
    assert verified_result.created is False
    assert verified_result.commit == created.commit


def test_followup_pilot_excludes_every_document_from_completed_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(
        _catalog_record(
            tmp_path,
            filename=(f"2024-01-01_00000000-0000-4000-8000-{index:012d}.pdf"),
            payload=f"%PDF-1.7\nfixture {index}\n".encode(),
        )
        for index in range(1, 5)
    )
    verified: Any = SimpleNamespace(
        config_path=(tmp_path / "catalog.yaml").absolute(),
        config_sha256="7" * 64,
        result=SimpleNamespace(commit=SimpleNamespace(catalog_sha256=CATALOG_SHA256)),
        records=records,
    )
    monkeypatch.setattr(pilot_module, "_verified_catalog", lambda *_args, **_kwargs: verified)

    first_unpinned = _config(tmp_path)
    first_plan = pilot_module._build_plan(first_unpinned, progress=None)
    first = first_unpinned.model_copy(
        update={"expected_manifest_sha256": first_plan.manifest_sha256}
    )
    first_result = materialize_pilot(first)

    followup_unpinned = CatalogPilotConfig(
        schema_version=2,
        catalog_config=first.catalog_config,
        expected_catalog_sha256=first.expected_catalog_sha256,
        destination_root=str((tmp_path / "followup").absolute()),
        final_label="blc",
        document_count=2,
        max_pages_per_document=2,
        selection_namespace="fixture-followup-v1",
        quality=first.quality,
        deduplicate_by_content_sha256=True,
        excluded_pilots=[
            ExcludedPilotConfig(
                root=str(first_result.root),
                expected_manifest_sha256=first_result.commit.manifest_sha256,
            )
        ],
        strata=first.strata,
        expected_manifest_sha256="0" * 64,
        verification_workers=2,
        max_pdf_bytes=1_000_000,
    )
    followup_plan = pilot_module._build_plan(followup_unpinned, progress=None)
    followup = followup_unpinned.model_copy(
        update={"expected_manifest_sha256": followup_plan.manifest_sha256}
    )
    followup_result = materialize_pilot(followup)
    verified_followup = verify_pilot(followup)

    first_names = {record.document_filename for record in first_plan.records}
    followup_names = {record.document_filename for record in followup_plan.records}
    first_hashes = {record.source_sha256 for record in first_plan.records}
    followup_hashes = {record.source_sha256 for record in followup_plan.records}
    assert first_names.isdisjoint(followup_names)
    assert first_hashes.isdisjoint(followup_hashes)
    assert followup_result.commit.schema_version == 2
    assert verified_followup.commit == followup_result.commit
