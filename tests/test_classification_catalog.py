from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest

import document_ocr.classification_catalog as catalog_module
from document_ocr.classification_catalog import (
    CatalogArtifactKind,
    CatalogJoinError,
    CatalogLocalAlias,
    CatalogS3Alias,
    CatalogSnapshotSummary,
    CatalogSourceArtifact,
    _assemble_plan,
    _FetchedArtifact,
    _VerifiedCatalogSnapshot,
    materialize_catalog,
    verify_catalog,
)
from document_ocr.config import (
    CatalogArtifactConfig,
    CatalogLineageConfig,
    CatalogRawSourceConfig,
    ClassificationCatalogConfig,
    CorpusSnapshotSourceConfig,
)
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.snapshot import SnapshotFileRecord

NOW = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
BUCKET = "fixture-catalog-bucket"
LINEAGE = "fixture"


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _manifest_row(
    doc_id: str,
    filename: str,
    *,
    label: str = "incoming",
    dummy: bool = False,
    final_label: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "classification_label": label,
        "classification_llm": {
            "confidence": 0.95,
            "predicted": "blc",
            "raw": {"classification": "blc", "confidence": 0.95},
        },
        "doc_id": doc_id,
        "dummy": {
            "confidence": 1.0,
            "detection_method": "fixture",
            "is_dummy": dummy,
            "reasoning": "fixture evidence",
        },
        "image_path": f"/workspace/images/{filename}/page_0001.png",
        "is_first_page": True,
        "original_pdf_path": f"/workspace/raw/primary/{filename}",
        "page_index": 0,
        "triage": {"details": {}, "requires_augmentation": False},
    }
    if final_label is not None:
        row["final_classification_label"] = final_label
    return row


def _artifact(
    kind: CatalogArtifactKind,
    payload: bytes,
    *,
    local_name: str,
) -> _FetchedArtifact:
    key = f"artifacts/{kind}/{local_name}"
    return _FetchedArtifact(
        metadata=CatalogSourceArtifact(
            lineage=LINEAGE,
            kind=kind,
            source_bucket=BUCKET,
            source_key=key,
            source_size_bytes=len(payload),
            source_etag=f'"etag-{kind}"',
            source_last_modified=NOW.isoformat(),
            source_version_id=None,
            sha256=sha256_bytes(payload),
            catalog_relative_path=f"sources/{LINEAGE}/{local_name}",
        ),
        payload=payload,
    )


def _raw_alias(
    filename: str,
    *,
    role: Literal["primary", "mirror"] = "primary",
    prefix: str | None = None,
) -> CatalogS3Alias:
    selected_prefix = prefix or f"raw/{role}"
    return CatalogS3Alias(
        source_bucket=BUCKET,
        lineage=LINEAGE,
        role=role,
        source_key=f"{selected_prefix}/{filename}",
        source_size_bytes=100,
        source_etag=f'"etag-{role}-{filename}"',
        source_last_modified=NOW.isoformat(),
        source_storage_class="STANDARD",
        source_version_id=None,
        checksum_algorithms=[],
        checksum_type=None,
    )


def _config(tmp_path: Path, artifacts: tuple[_FetchedArtifact, ...]) -> ClassificationCatalogConfig:
    by_kind = {item.metadata.kind: item for item in artifacts}

    def artifact(kind: CatalogArtifactKind) -> CatalogArtifactConfig:
        item = by_kind[kind]
        return CatalogArtifactConfig(
            key=item.metadata.source_key,
            expected_sha256=item.metadata.sha256,
            local_name=Path(item.metadata.catalog_relative_path).name,
        )

    return ClassificationCatalogConfig(
        schema_version=1,
        bucket=BUCKET,
        region="us-east-1",
        destination_root=str((tmp_path / "catalog").absolute()),
        lineages=[
            CatalogLineageConfig(
                name=LINEAGE,
                initial_manifest=artifact("initial_manifest"),
                final_manifest=artifact("final_manifest"),
                dropped_report=artifact("dropped_report"),
                relabeled_report=artifact("relabeled_report"),
                review_report=artifact("review_report"),
            )
        ],
        raw_sources=[
            CatalogRawSourceConfig(lineage=LINEAGE, role="primary", prefix="raw/primary/"),
            CatalogRawSourceConfig(lineage=LINEAGE, role="mirror", prefix="raw/mirror/"),
        ],
        local_snapshots=[
            CorpusSnapshotSourceConfig(
                snapshot_config=str((tmp_path / "snapshot.yaml").absolute())
            )
        ],
        expected_catalog_sha256="0" * 64,
        expected_raw_inventory_sha256="0" * 64,
        network_workers=2,
        max_attempts=3,
        download_chunk_size_bytes=65536,
        max_source_artifact_bytes=1_000_000,
    )


def _fixture(
    tmp_path: Path,
) -> tuple[
    ClassificationCatalogConfig,
    tuple[_FetchedArtifact, ...],
    tuple[CatalogS3Alias, ...],
    tuple[_VerifiedCatalogSnapshot, ...],
]:
    retained = "2024-01-01_8b96c127-9e8a-49c8-b9ac-a85336ca98ad.pdf"
    dropped = "2024-01-02_0c7796ac-7697-4d95-8122-764be3b4aa9e.pdf"
    pending = "HBL fixture.pdf"
    raw_only = "2024-01-03_f3db0d3b-5b75-47bd-bca1-a72367896f81.pdf"
    initial_rows = [
        _manifest_row("incoming/retained", retained),
        _manifest_row("incoming/dropped", dropped, dummy=True),
        _manifest_row("incoming/pending", pending),
    ]
    final_row = _manifest_row("incoming/retained", retained)
    final_row["final_classification_label"] = "blc"
    artifacts = (
        _artifact("initial_manifest", _jsonl(initial_rows), local_name="initial.jsonl"),
        _artifact("final_manifest", _jsonl([final_row]), local_name="final.jsonl"),
        _artifact(
            "dropped_report",
            b"doc_id,reason\nincoming/dropped,Dummy document\n",
            local_name="dropped.csv",
        ),
        _artifact(
            "relabeled_report",
            b"doc_id,from,to,confidence,rule_id\n"
            b"incoming/retained,incoming,blc,0.950,fixture-rule\n",
            local_name="relabeled.csv",
        ),
        _artifact(
            "review_report",
            b"doc_id,reason\nincoming/pending,Needs review\n",
            local_name="review.csv",
        ),
    )
    raw = tuple(
        sorted(
            (
                _raw_alias(retained),
                _raw_alias(dropped),
                _raw_alias(pending),
                _raw_alias(raw_only),
                _raw_alias(retained, role="mirror", prefix="raw/mirror"),
            ),
            key=lambda item: (item.lineage, item.role, item.source_key),
        )
    )
    retained_raw = next(
        item
        for item in raw
        if item.source_key.endswith(retained) and item.role == "primary"
    )
    snapshot_record = SnapshotFileRecord(
        source_bucket=BUCKET,
        source_key=retained_raw.source_key,
        source_size_bytes=retained_raw.source_size_bytes,
        source_etag=retained_raw.source_etag,
        source_last_modified=retained_raw.source_last_modified,
        document_type="blc",
        classification_doc_id="incoming/retained",
        classification_manifest_key="artifacts/final_manifest/final.jsonl",
        document_page_count=1,
        schema_version=1,
        classification_manifest_sha256="c" * 64,
        local_relative_key=retained,
        snapshot_relative_path=f"files/blc/{retained}",
        source_sha256="d" * 64,
        actual_page_count=1,
        extraction_status="ready",
        extraction_relative_path=f"extraction/blc/{retained}",
    )
    config_path = str((tmp_path / "snapshot.yaml").absolute())
    snapshot_root = str((tmp_path / "snapshot").absolute())
    summary = CatalogSnapshotSummary(
        snapshot_config_path=config_path,
        snapshot_root=snapshot_root,
        snapshot_config_sha256="1" * 64,
        snapshot_commit_sha256="2" * 64,
        snapshot_manifest_sha256="3" * 64,
        snapshot_selection_sha256="4" * 64,
        document_count=1,
        source_bytes=100,
    )
    local_alias = CatalogLocalAlias(
        snapshot_config_path=config_path,
        snapshot_root=snapshot_root,
        snapshot_config_sha256=summary.snapshot_config_sha256,
        snapshot_commit_sha256=summary.snapshot_commit_sha256,
        snapshot_manifest_sha256=summary.snapshot_manifest_sha256,
        snapshot_selection_sha256=summary.snapshot_selection_sha256,
        snapshot_record=snapshot_record,
    )
    snapshots = (_VerifiedCatalogSnapshot(summary=summary, aliases=(local_alias,)),)
    return _config(tmp_path, artifacts), artifacts, raw, snapshots


def test_catalog_preserves_all_statuses_reports_and_source_aliases(tmp_path: Path) -> None:
    config, artifacts, raw, snapshots = _fixture(tmp_path)

    plan = _assemble_plan(
        config=config,
        artifacts=artifacts,
        raw_inventory=raw,
        snapshots=snapshots,
        bucket_versioning_status=None,
    )

    assert plan.statistics.catalog_documents == 4
    assert plan.statistics.status_counts == {
        "excluded_reported": 1,
        "missing_final_without_drop_report": 1,
        "never_classified": 1,
        "retained_final": 1,
    }
    assert plan.statistics.dummy_documents == 1
    assert plan.statistics.review_documents == 1
    assert plan.statistics.s3_source_aliases == 5
    assert plan.statistics.local_snapshot_aliases == 1
    retained = next(item for item in plan.records if item.status == "retained_final")
    assert len(retained.s3_source_aliases) == 2
    assert len(retained.local_snapshot_aliases) == 1
    assert retained.final is not None
    assert retained.final.final_classification_label == "blc"
    pending = next(item for item in plan.records if item.review_report is not None)
    assert pending.status == "missing_final_without_drop_report"
    assert pending.review_report is not None
    assert pending.review_report.values["reason"] == "Needs review"


def test_catalog_rejects_relabel_report_that_differs_from_manifest(tmp_path: Path) -> None:
    config, artifacts, raw, snapshots = _fixture(tmp_path)
    bad_report = _artifact(
        "relabeled_report",
        b"doc_id,from,to,confidence,rule_id\n"
        b"incoming/retained,incoming,swb,0.950,fixture-rule\n",
        local_name="relabeled.csv",
    )
    changed = tuple(
        bad_report if item.metadata.kind == "relabeled_report" else item for item in artifacts
    )

    with pytest.raises(CatalogJoinError, match="relabeled report values differ"):
        _assemble_plan(
            config=config,
            artifacts=changed,
            raw_inventory=raw,
            snapshots=snapshots,
            bucket_versioning_status=None,
        )


def test_materialize_is_manifest_last_and_offline_verify_rebuilds_joins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpinned, artifacts, raw, snapshots = _fixture(tmp_path)
    plan = _assemble_plan(
        config=unpinned,
        artifacts=artifacts,
        raw_inventory=raw,
        snapshots=snapshots,
        bucket_versioning_status=None,
    )
    config = unpinned.model_copy(
        update={
            "expected_catalog_sha256": plan.catalog_sha256,
            "expected_raw_inventory_sha256": plan.raw_inventory_sha256,
        }
    )
    monkeypatch.setattr(catalog_module, "_remote_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        catalog_module,
        "_verified_local_snapshots",
        lambda *_args, **_kwargs: snapshots,
    )

    result = materialize_catalog(config, s3_client=object())

    assert result.created is True
    assert (result.root / "catalog.json").is_file()
    assert result.commit.document_count == 4
    verified = verify_catalog(config)
    assert verified.created is False
    assert verified.commit == result.commit
    assert verified.statistics == result.statistics
    monkeypatch.setattr(
        catalog_module,
        "_remote_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("committed materialization must remain offline")
        ),
    )
    resumed = materialize_catalog(config, s3_client=object())
    assert resumed.created is False
    assert resumed.commit == result.commit


def test_materialize_resumes_identical_artifacts_after_precommit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpinned, artifacts, raw, snapshots = _fixture(tmp_path)
    plan = _assemble_plan(
        config=unpinned,
        artifacts=artifacts,
        raw_inventory=raw,
        snapshots=snapshots,
        bucket_versioning_status=None,
    )
    config = unpinned.model_copy(
        update={
            "expected_catalog_sha256": plan.catalog_sha256,
            "expected_raw_inventory_sha256": plan.raw_inventory_sha256,
        }
    )
    monkeypatch.setattr(catalog_module, "_remote_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        catalog_module,
        "_verified_local_snapshots",
        lambda *_args, **_kwargs: snapshots,
    )
    publish = catalog_module._publish_artifact
    failed = False

    def interrupt(path: Path, payload: bytes) -> str:
        nonlocal failed
        if path.name == "catalog.jsonl" and not failed:
            failed = True
            raise OSError("simulated interruption")
        return publish(path, payload)

    monkeypatch.setattr(catalog_module, "_publish_artifact", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        materialize_catalog(config, s3_client=object())
    assert not (Path(config.destination_root) / "catalog.json").exists()

    monkeypatch.setattr(catalog_module, "_publish_artifact", publish)
    resumed = materialize_catalog(config, s3_client=object())
    assert resumed.created is True
    assert resumed.commit.catalog_sha256 == plan.catalog_sha256


def test_catalog_output_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    config, artifacts, raw, snapshots = _fixture(tmp_path)

    first = _assemble_plan(
        config=config,
        artifacts=artifacts,
        raw_inventory=raw,
        snapshots=snapshots,
        bucket_versioning_status=None,
    )
    second = _assemble_plan(
        config=config,
        artifacts=artifacts,
        raw_inventory=raw,
        snapshots=snapshots,
        bucket_versioning_status=None,
    )

    assert first.catalog_payload == second.catalog_payload
    assert first.raw_inventory_payload == second.raw_inventory_payload
    assert first.statistics_payload == second.statistics_payload
    assert json.loads(first.statistics_payload) == first.statistics.model_dump(mode="json")
