from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from document_ocr import snapshot_cli
from document_ocr.config import S3LocalSnapshotConfig, load_snapshot_config
from document_ocr.hashing import canonical_json_bytes
from document_ocr.snapshot import (
    SnapshotCommit,
    SnapshotResult,
    SnapshotSelectionAudit,
    SnapshotTypeSummary,
    StoredClassificationManifest,
)


def _config(tmp_path: Path) -> S3LocalSnapshotConfig:
    source = load_snapshot_config(
        Path(__file__).parents[1] / "configs" / "s3_snapshot.blc_swb.yaml"
    )
    return source.model_copy(update={"destination_root": str(tmp_path / "snapshot")})


def _result(config: S3LocalSnapshotConfig, *, created: bool) -> SnapshotResult:
    blc = SnapshotTypeSummary(
        documents=2,
        source_bytes=200,
        classified_pages=3,
        actual_pages=3,
        extraction_documents=2,
        extraction_pages=3,
        quarantined_documents=0,
    )
    swb = SnapshotTypeSummary(
        documents=1,
        source_bytes=100,
        classified_pages=1,
        actual_pages=1,
        extraction_documents=1,
        extraction_pages=1,
        quarantined_documents=0,
    )
    commit = SnapshotCommit(
        schema_version=1,
        bucket=config.bucket,
        raw_prefix=config.raw_prefix,
        region=config.region,
        document_types=["blc", "swb"],
        selection_path="selection.jsonl",
        selection_sha256=config.expected_selection_sha256,
        manifest_path="manifest.jsonl",
        manifest_sha256="a" * 64,
        classification_manifests=[
            StoredClassificationManifest(
                key=item.key,
                sha256=item.expected_sha256,
                snapshot_relative_path=f"classification/{item.local_name}",
                source_etag='"etag"',
                source_last_modified=datetime(2026, 8, 10, tzinfo=UTC),
                source_size_bytes=100,
            )
            for item in config.classification_manifests
        ],
        selection_audit=SnapshotSelectionAudit(
            raw_pdf_count=4,
            raw_pdf_bytes=400,
            raw_pdfs_without_filtered_manifest_evidence=1,
            filtered_manifest_documents_without_raw_pdf=0,
        ),
        by_document_type={"blc": blc, "swb": swb},
        document_count=3,
        source_bytes=300,
        classified_pages=4,
        actual_pages=4,
        extraction_document_count=3,
        extraction_pages=4,
        quarantined_document_count=0,
    )
    return SnapshotResult(root=Path(config.destination_root), commit=commit, created=created)


def _line(value: dict[str, Any]) -> str:
    return (canonical_json_bytes(value) + b"\n").decode("utf-8")


@pytest.mark.parametrize(
    ("command", "function_name", "created"),
    [("materialize", "materialize_snapshot", True), ("verify", "verify_snapshot", False)],
)
def test_snapshot_commands_emit_canonical_summary(
    command: str,
    function_name: str,
    created: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    result = _result(config, created=created)
    monkeypatch.setattr(snapshot_cli, "load_snapshot_config", lambda _: config)
    calls: list[S3LocalSnapshotConfig] = []

    def fake(actual: S3LocalSnapshotConfig, **_: Any) -> SnapshotResult:
        calls.append(actual)
        return result

    monkeypatch.setattr(snapshot_cli, function_name, fake)

    exit_code = snapshot_cli.main([command, "--config", "snapshot.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [config]
    assert captured.err == ""
    assert captured.out == _line(
        {
            "actual_pages": 4,
            "by_document_type": {
                "blc": result.commit.by_document_type["blc"].model_dump(mode="json"),
                "swb": result.commit.by_document_type["swb"].model_dump(mode="json"),
            },
            "command": command,
            "created": created,
            "document_count": 3,
            "extraction_document_count": 3,
            "extraction_pages": 4,
            "manifest_sha256": "a" * 64,
            "quarantined_document_count": 0,
            "selection_sha256": config.expected_selection_sha256,
            "selection_audit": result.commit.selection_audit.model_dump(mode="json"),
            "snapshot_root": config.destination_root,
            "source_bytes": 300,
            "status": "complete",
        }
    )


def test_snapshot_cli_reports_configuration_failure_without_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_: Path) -> S3LocalSnapshotConfig:
        raise ValueError("invalid secret detail")

    monkeypatch.setattr(snapshot_cli, "load_snapshot_config", fail)

    exit_code = snapshot_cli.main(["verify", "--config", "bad.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert captured.err == _line(
        {
            "command": "verify",
            "error_type": "ValueError",
            "message": "snapshot configuration could not be loaded: bad.yaml",
            "status": "error",
        }
    )
