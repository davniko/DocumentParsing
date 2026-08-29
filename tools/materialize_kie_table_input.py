#!/usr/bin/env python3
"""Materialize a fail-closed raw-text plus GLM-OCR-table training input view."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file

VIEW_KEY = "glmOcrTableRecognition"
MODEL_INPUT_FIELD = "modelInputText"
MODEL_INPUT_SHA_FIELD = "modelInputTextSha256"
RAW_OPEN = "<raw_ocr_text>"
RAW_CLOSE = "</raw_ocr_text>"
TABLE_OPEN = "<glm_ocr_table_view>"
TABLE_CLOSE = "</glm_ocr_table_view>"


def _strict_object(line: bytes, *, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    value = json.loads(line.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON row is not an object: {context}")
    return cast(dict[str, Any], value)


def _source_file(manifest: dict[str, Any], *, kind: str) -> dict[str, Any]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("source manifest files must be a list")
    matches = [row for row in raw_files if isinstance(row, dict) and row.get("kind") == kind]
    if len(matches) != 1:
        raise ValueError(f"source manifest must contain exactly one {kind!r} file")
    return cast(dict[str, Any], matches[0])


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{context} must be a non-empty NUL-free string")
    return value


def _validate_view(row: dict[str, Any], *, context: str) -> str:
    auxiliary = row.get("auxiliaryViews")
    if not isinstance(auxiliary, dict) or set(auxiliary) != {VIEW_KEY}:
        raise ValueError(f"{context}: auxiliaryViews must contain only {VIEW_KEY!r}")
    view = auxiliary[VIEW_KEY]
    if not isinstance(view, dict):
        raise ValueError(f"{context}: table view must be an object")
    joined = _require_nonempty_string(view.get("joinedTableText"), context="joinedTableText")
    if sha256_bytes(joined.encode("utf-8")) != view.get("joinedTableTextSha256"):
        raise ValueError(f"{context}: joinedTableText SHA-256 mismatch")
    page_count = view.get("documentPageCount")
    pages = view.get("pages")
    if type(page_count) is not int or page_count <= 0 or not isinstance(pages, list):
        raise ValueError(f"{context}: invalid table page count/list")
    if len(pages) != page_count:
        raise ValueError(f"{context}: table page list differs from documentPageCount")
    reconstructed: list[str] = []
    for expected_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"{context}: table page is not an object")
        raw = _require_nonempty_string(page.get("rawTableText"), context="rawTableText")
        if (
            page.get("pageIndex") != expected_index
            or page.get("pageNumber") != expected_index + 1
            or sha256_bytes(raw.encode("utf-8")) != page.get("rawTableTextSha256")
        ):
            raise ValueError(f"{context}: table page ordering or digest mismatch")
        reconstructed.append(f"--- PAGE {expected_index + 1} ---\n{raw}")
    if "\n\n".join(reconstructed) != joined:
        raise ValueError(f"{context}: joinedTableText differs from ordered page texts")
    return joined


def _render_input(raw_text: str, table_text: str) -> str:
    for sentinel in (RAW_OPEN, RAW_CLOSE, TABLE_OPEN, TABLE_CLOSE):
        if sentinel in raw_text or sentinel in table_text:
            raise ValueError(f"source OCR contains reserved input sentinel: {sentinel}")
    return f"{RAW_OPEN}\n{raw_text}\n{RAW_CLOSE}\n\n{TABLE_OPEN}\n{table_text}\n{TABLE_CLOSE}"


def _load_target_source(
    *, path: Path, expected_sha256: str, expected_records: int
) -> dict[str, dict[str, Any]]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"target-source records path is invalid: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError("target-source records SHA-256 mismatch")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line:
            raise ValueError(f"blank target-source line at {path}:{line_number}")
        row = _strict_object(line, context=f"{path}:{line_number}")
        document_id = _require_nonempty_string(row.get("documentId"), context="documentId")
        if document_id in rows:
            raise ValueError(f"duplicate target-source documentId: {document_id}")
        if not isinstance(row.get("target"), dict):
            raise ValueError(f"target-source target is not an object: {document_id}")
        _require_nonempty_string(row.get("joinedRawTextSha256"), context="joinedRawTextSha256")
        rows[document_id] = {**row, "_sourceRow": line_number}
    if len(rows) != expected_records:
        raise ValueError(
            f"target-source row count differs: expected {expected_records}, found {len(rows)}"
        )
    return rows


def _materialize_split(
    *,
    source_dir: Path,
    source_file: dict[str, Any],
    split: str,
    target_source: dict[str, dict[str, Any]] | None,
    used_target_ids: set[str],
) -> tuple[bytes, bytes, int]:
    relative = source_file.get("path")
    expected_sha = source_file.get("sha256")
    expected_rows = source_file.get("records")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"invalid source path for {split}")
    path = (source_dir / relative).resolve(strict=True)
    if not path.is_relative_to(source_dir) or not path.is_file() or path.is_symlink():
        raise ValueError(f"source file escapes dataset root for {split}: {path}")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"source SHA-256 mismatch for {split}")
    output_lines: list[bytes] = []
    lineage_lines: list[bytes] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line:
            raise ValueError(f"blank line at {path}:{line_number}")
        context = f"{path}:{line_number}"
        row = _strict_object(line, context=context)
        document_id = _require_nonempty_string(row.get("documentId"), context="documentId")
        if document_id in seen_ids:
            raise ValueError(f"duplicate documentId in {split}: {document_id}")
        seen_ids.add(document_id)
        raw_text = _require_nonempty_string(row.get("joinedRawText"), context="joinedRawText")
        if sha256_bytes(raw_text.encode("utf-8")) != row.get("joinedRawTextSha256"):
            raise ValueError(f"{context}: joinedRawText SHA-256 mismatch")
        if not isinstance(row.get("target"), dict):
            raise ValueError(f"{context}: target must be an object")
        if MODEL_INPUT_FIELD in row or MODEL_INPUT_SHA_FIELD in row:
            raise ValueError(f"{context}: source already contains materialized model input")
        table_text = _validate_view(row, context=context)
        model_input = _render_input(raw_text, table_text)
        target_lineage: dict[str, Any] = {}
        if target_source is not None:
            override = target_source.get(document_id)
            if override is None:
                raise ValueError(f"target source has no table-covered document: {document_id}")
            if override["joinedRawTextSha256"] != row["joinedRawTextSha256"]:
                raise ValueError(f"target source raw-text identity differs: {document_id}")
            used_target_ids.add(document_id)
            row = {**row, "target": override["target"]}
            if "normalTarget" in override:
                row["normalTarget"] = override["normalTarget"]
            target_lineage = {
                "targetSourceRow": override["_sourceRow"],
                "targetSourceTargetSha256": sha256_bytes(canonical_json_bytes(override["target"])),
            }
        projected = {
            **row,
            MODEL_INPUT_FIELD: model_input,
            MODEL_INPUT_SHA_FIELD: sha256_bytes(model_input.encode("utf-8")),
        }
        projected_bytes = canonical_json_bytes(projected) + b"\n"
        output_lines.append(projected_bytes)
        lineage_lines.append(
            canonical_json_bytes(
                {
                    "documentId": document_id,
                    "sourceSplit": split,
                    "sourcePath": str(path),
                    "sourceRow": line_number,
                    "sourceRowSha256": sha256_bytes(line + b"\n"),
                    "joinedRawTextSha256": row["joinedRawTextSha256"],
                    "joinedTableTextSha256": row["auxiliaryViews"][VIEW_KEY][
                        "joinedTableTextSha256"
                    ],
                    MODEL_INPUT_SHA_FIELD: projected[MODEL_INPUT_SHA_FIELD],
                    "projectedRowSha256": sha256_bytes(projected_bytes),
                    **target_lineage,
                }
            )
            + b"\n"
        )
    if len(output_lines) != expected_rows:
        raise ValueError(
            f"source row count differs for {split}: expected {expected_rows}, "
            f"found {len(output_lines)}"
        )
    return b"".join(output_lines), b"".join(lineage_lines), len(output_lines)


def materialize(
    *,
    source_dir: Path,
    output_dir: Path,
    dataset_id: str,
    target_records: Path | None = None,
    target_records_sha256: str | None = None,
    target_records_count: int | None = None,
) -> dict[str, Any]:
    source_dir = source_dir.resolve(strict=True)
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise ValueError(f"source dataset root is invalid: {source_dir}")
    manifest_path = source_dir / "manifest.json"
    manifest = _strict_object(manifest_path.read_bytes(), context=str(manifest_path))
    if output_dir.exists():
        raise ValueError(f"output dataset already exists: {output_dir}")

    target_arguments = (target_records, target_records_sha256, target_records_count)
    if any(value is not None for value in target_arguments) and not all(
        value is not None for value in target_arguments
    ):
        raise ValueError("target-source path, SHA-256, and record count must be provided together")
    if target_records is None:
        target_source = None
    else:
        if target_records_sha256 is None or target_records_count is None:
            raise AssertionError("complete target-source arguments were validated above")
        target_source = _load_target_source(
            path=target_records,
            expected_sha256=target_records_sha256,
            expected_records=target_records_count,
        )
    used_target_ids: set[str] = set()

    materialized_splits: list[tuple[str, str, bytes, bytes, int]] = []
    for split, kind in (("train", "train_records"), ("validation", "validation_records")):
        payload, lineage, rows = _materialize_split(
            source_dir=source_dir,
            source_file=_source_file(manifest, kind=kind),
            split=split,
            target_source=target_source,
            used_target_ids=used_target_ids,
        )
        materialized_splits.append((split, kind, payload, lineage, rows))

    output_dir.mkdir(parents=True)
    published: list[dict[str, Any]] = []
    lineage_parts: list[bytes] = []
    for split, kind, payload, lineage, rows in materialized_splits:
        relative = f"{split}.jsonl"
        atomic_publish_bytes(output_dir / relative, payload)
        published.append(
            {
                "kind": kind,
                "path": relative,
                "records": rows,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        lineage_parts.append(lineage)
    lineage_payload = b"".join(lineage_parts)
    atomic_publish_bytes(output_dir / "lineage.jsonl", lineage_payload)
    published.append(
        {
            "kind": "lineage",
            "path": "lineage.jsonl",
            "records": sum(item["records"] for item in published),
            "bytes": len(lineage_payload),
            "sha256": sha256_bytes(lineage_payload),
        }
    )
    result = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "created_at": datetime.now(UTC).isoformat(),
        "documents": sum(item["records"] for item in published if item["kind"] != "lineage"),
        "source": {
            "dataset_id": manifest.get("dataset_id"),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "target_source": (
            {
                "path": str(cast(Path, target_records).resolve(strict=True)),
                "sha256": target_records_sha256,
                "records": target_records_count,
                "matched_records": len(used_target_ids),
            }
            if target_source is not None
            else None
        ),
        "input_contract": {
            "raw_text_is_value_truth_boundary": True,
            "table_view_is_auxiliary_structure_evidence": True,
            "view_key": VIEW_KEY,
            "model_input_field": MODEL_INPUT_FIELD,
            "model_input_sha256_field": MODEL_INPUT_SHA_FIELD,
            "sentinels": [RAW_OPEN, RAW_CLOSE, TABLE_OPEN, TABLE_CLOSE],
        },
        "files": published,
    }
    atomic_publish_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--target-records", type=Path)
    parser.add_argument("--target-records-sha256")
    parser.add_argument("--target-records-count", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                dataset_id=args.dataset_id,
                target_records=args.target_records,
                target_records_sha256=args.target_records_sha256,
                target_records_count=args.target_records_count,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
