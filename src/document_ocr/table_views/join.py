"""Audited, non-destructive join of table views into cloned KIE datasets."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    sha256_file,
)
from document_ocr.table_views.config import JoinInputConfig, TableJoinConfig, TableViewConfig
from document_ocr.table_views.models import (
    JoinedTableDocumentView,
    JoinedTablePage,
    TablePageResult,
)
from document_ocr.table_views.pipeline import (
    TableViewError,
    load_frozen_prepared_pages,
    table_run_paths,
)

_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VIEW_KEY = "glmOcrTableRecognition"


def _require_join_config(config: TableViewConfig) -> TableJoinConfig:
    if config.join is None:
        raise TableViewError(
            "table-view join is not configured; add a join section before using join or verify-join"
        )
    return config.join


def _strict_json(payload: bytes, *, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite number {value!r}")

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, UnicodeError, ValueError) as error:
        raise TableViewError(f"invalid JSON object in {context}") from error
    if not isinstance(decoded, dict):
        raise TableViewError(f"JSON root is not an object in {context}")
    return cast(dict[str, Any], decoded)


def _canonical_regular_file(path: Path, *, context: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TableViewError(f"missing {context}: {path}") from error
    if resolved != absolute or not resolved.is_file() or resolved.is_symlink():
        raise TableViewError(f"{context} must be a canonical regular file: {path}")
    return resolved


def _read_input(config: JoinInputConfig) -> tuple[dict[str, Any], ...]:
    path = _canonical_regular_file(Path(config.path), context="join input")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != config.sha256:
        raise TableViewError(f"join input SHA-256 mismatch: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise TableViewError(f"blank line in join input: {path}:{line_number}")
        row = _strict_json(line, context=f"{path}:{line_number}")
        document_id = row.get("documentId")
        joined_raw_text = row.get("joinedRawText")
        joined_raw_text_sha256 = row.get("joinedRawTextSha256")
        if not isinstance(document_id, str) or _DOCUMENT_ID.fullmatch(document_id) is None:
            raise TableViewError(f"invalid documentId in join input: {path}:{line_number}")
        if not isinstance(joined_raw_text, str) or not joined_raw_text.strip():
            raise TableViewError(f"invalid joinedRawText in join input: {path}:{line_number}")
        if (
            not isinstance(joined_raw_text_sha256, str)
            or sha256_bytes(joined_raw_text.encode("utf-8")) != joined_raw_text_sha256
        ):
            raise TableViewError(
                f"joinedRawText SHA-256 mismatch in join input: {path}:{line_number}"
            )
        if not isinstance(row.get("target"), dict):
            raise TableViewError(f"target is not an object in join input: {path}:{line_number}")
        if "auxiliaryViews" in row:
            raise TableViewError(
                f"join input already contains auxiliaryViews: {path}:{line_number}"
            )
        rows.append(row)
    if len(rows) != config.records:
        raise TableViewError(
            f"join input row count differs: expected {config.records}, found {len(rows)}"
        )
    return tuple(rows)


def _joined_table_text(results: tuple[TablePageResult, ...]) -> str:
    return "\n\n".join(
        f"--- PAGE {result.pageNumber} ---\n{result.rawTableText}" for result in results
    )


def _published_results(config: TableViewConfig) -> tuple[TablePageResult, ...]:
    """Load the manifest-last table dataset without reopening mutable run state."""

    paths = table_run_paths(config)
    manifest_path = _canonical_regular_file(
        paths.dataset_root / "manifest.json", context="table dataset manifest"
    )
    manifest = _strict_json(read_regular_file_bytes(manifest_path), context=str(manifest_path))
    expected_manifest_fields = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "view_type": config.view_type,
        "config_sha256": canonical_json_sha256(config.model_dump(mode="json")),
        "documents": config.source.expected_documents,
        "pages": config.source.expected_pages,
    }
    for field_name, expected in expected_manifest_fields.items():
        if manifest.get(field_name) != expected:
            raise TableViewError(f"table dataset manifest {field_name} differs from configuration")

    frozen_pages = load_frozen_prepared_pages(config)
    if manifest.get("input_pages_sha256") != sha256_file(paths.input_pages):
        raise TableViewError("table dataset manifest input page digest differs")
    input_by_view_id = {page.tableViewId: page for page in frozen_pages}
    if len(input_by_view_id) != len(frozen_pages):
        raise TableViewError("frozen table inputs contain duplicate tableViewId values")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise TableViewError("table dataset manifest files is not a list")
    expected_kinds = {"table_pages", "attempts", "quality_summary", "quality_flags"}
    payloads: dict[str, bytes] = {}
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise TableViewError("table dataset manifest file entry is not an object")
        kind = raw_file.get("kind")
        relative_value = raw_file.get("path")
        rows = raw_file.get("rows")
        byte_count = raw_file.get("bytes")
        digest = raw_file.get("sha256")
        if not isinstance(kind, str) or kind not in expected_kinds or kind in payloads:
            raise TableViewError("table dataset manifest file kinds are invalid")
        if not isinstance(relative_value, str):
            raise TableViewError("table dataset manifest path is invalid")
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "dataset"
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in relative_value
        ):
            raise TableViewError("table dataset manifest path escapes the dataset")
        if type(rows) is not int or rows < 0:
            raise TableViewError("table dataset manifest row count is invalid")
        if type(byte_count) is not int or byte_count < 0:
            raise TableViewError("table dataset manifest byte count is invalid")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise TableViewError("table dataset manifest digest is invalid")
        artifact_path = _canonical_regular_file(
            paths.run_root / Path(relative), context=f"published table {kind}"
        )
        try:
            artifact_path.relative_to(paths.dataset_root)
        except ValueError as error:
            raise TableViewError("published table artifact escapes dataset root") from error
        payload = read_regular_file_bytes(artifact_path)
        if len(payload) != byte_count or sha256_bytes(payload) != digest:
            raise TableViewError(f"published table {kind} differs from its manifest")
        if kind == "quality_summary":
            if rows != 1:
                raise TableViewError("published table quality summary row count differs")
        else:
            lines = payload.splitlines()
            if len(lines) != rows or any(not line for line in lines):
                raise TableViewError(f"published table {kind} row count differs")
        payloads[kind] = payload
    if set(payloads) != expected_kinds:
        raise TableViewError("table dataset manifest file set is incomplete")

    quality_summary = _strict_json(
        payloads["quality_summary"], context="published table quality summary"
    )
    if quality_summary != manifest.get("quality_summary"):
        raise TableViewError("published quality summary differs from table manifest")
    for kind in ("attempts", "quality_flags"):
        for line_number, line in enumerate(payloads[kind].splitlines(), start=1):
            _strict_json(line, context=f"published table {kind}:{line_number}")

    results: list[TablePageResult] = []
    seen_page_ids: set[str] = set()
    seen_view_ids: set[str] = set()
    for line_number, line in enumerate(payloads["table_pages"].splitlines(), start=1):
        try:
            result = TablePageResult.model_validate_json(line, strict=True)
        except ValueError as error:
            raise TableViewError(f"invalid published table page at line {line_number}") from error
        page = input_by_view_id.get(result.tableViewId)
        if page is None:
            raise TableViewError("published table page has no frozen input")
        input_digest = canonical_json_sha256(page.model_dump(mode="json"))
        if (
            result.inputPageSha256 != input_digest
            or result.documentId != page.documentId
            or result.documentPageCount != page.documentPageCount
            or result.pageIndex != page.pageIndex
            or result.pageNumber != page.pageNumber
            or result.pageId != page.pageId
            or result.rasterPath != page.rasterPath
            or result.rasterSha256 != page.rasterSha256
            or result.prompt != config.vllm.prompt
            or result.promptSha256 != sha256_bytes(config.vllm.prompt.encode("utf-8"))
            or result.model != config.vllm.model
            or result.modelRevision != config.vllm.revision
            or result.servedModelName != config.vllm.served_model_name
            or result.vllmEngineVersion != config.vllm.engine_version
            or sha256_bytes(result.rawTableText.encode("utf-8")) != result.rawTableTextSha256
        ):
            raise TableViewError("published table page conflicts with its frozen input")
        if result.pageId in seen_page_ids or result.tableViewId in seen_view_ids:
            raise TableViewError("published table pages contain duplicate identities")
        seen_page_ids.add(result.pageId)
        seen_view_ids.add(result.tableViewId)
        results.append(result)
    if seen_view_ids != set(input_by_view_id):
        raise TableViewError("published table page set differs from frozen inputs")
    if sum(result.attemptCount for result in results) != manifest.get("attempts"):
        raise TableViewError("published table attempt total differs from manifest")
    return tuple(results)


def _document_views(config: TableViewConfig) -> dict[str, JoinedTableDocumentView]:
    grouped: dict[str, list[TablePageResult]] = defaultdict(list)
    for result in _published_results(config):
        grouped[result.documentId].append(result)

    views: dict[str, JoinedTableDocumentView] = {}
    for document_id, unordered in grouped.items():
        results = tuple(sorted(unordered, key=lambda row: row.pageIndex))
        joined_text = _joined_table_text(results)
        pages = tuple(
            JoinedTablePage.model_validate(
                {
                    "pageIndex": result.pageIndex,
                    "pageNumber": result.pageNumber,
                    "pageId": result.pageId,
                    "tableViewId": result.tableViewId,
                    "rasterSha256": result.rasterSha256,
                    "finishReason": result.finishReason,
                    "completionTokens": result.completionTokens,
                    "rawTableText": result.rawTableText,
                    "rawTableTextSha256": result.rawTableTextSha256,
                    "resultSha256": sha256_bytes(
                        canonical_json_bytes(result.model_dump(mode="json")) + b"\n"
                    ),
                },
                strict=True,
            )
            for result in results
        )
        views[document_id] = JoinedTableDocumentView.model_validate(
            {
                "schemaVersion": 1,
                "viewType": config.view_type,
                "runId": config.run.run_id,
                "documentPageCount": results[0].documentPageCount,
                "joinedTableText": joined_text,
                "joinedTableTextSha256": sha256_bytes(joined_text.encode("utf-8")),
                "pages": pages,
            },
            strict=True,
        )
    return views


def publish_joined_dataset(config: TableViewConfig) -> Path:
    """Clone configured splits, add only an auxiliary table view, and publish manifest last."""

    join_config = _require_join_config(config)
    paths = table_run_paths(config)
    table_manifest = paths.dataset_root / "manifest.json"
    if not table_manifest.is_file():
        raise TableViewError("the complete table-view dataset must be published before joining")
    table_manifest_sha256 = sha256_file(table_manifest)
    views = _document_views(config)

    source_rows: list[tuple[str, JoinInputConfig, int, dict[str, Any]]] = []
    seen_document_ids: set[str] = set()
    for input_config in join_config.inputs:
        for row_number, row in enumerate(_read_input(input_config), start=1):
            document_id = cast(str, row["documentId"])
            if document_id in seen_document_ids:
                raise TableViewError(f"duplicate documentId across join inputs: {document_id}")
            seen_document_ids.add(document_id)
            source_rows.append((input_config.split, input_config, row_number, row))
    if seen_document_ids != set(views):
        missing = sorted(seen_document_ids - set(views))
        extra = sorted(set(views) - seen_document_ids)
        raise TableViewError(
            f"table/source document sets differ; missing_table={missing!r}, extra_table={extra!r}"
        )

    output_root = Path(join_config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.resolve(strict=True) != Path(os.path.abspath(output_root)):
        raise TableViewError("joined output root must not traverse symbolic links")

    by_split: dict[str, list[tuple[JoinInputConfig, int, dict[str, Any]]]] = defaultdict(list)
    for split, input_config, row_number, row in source_rows:
        by_split[split].append((input_config, row_number, row))

    published_files: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        split_rows = by_split.get(split)
        if not split_rows:
            continue
        output_payload_parts: list[bytes] = []
        for input_config, row_number, row in split_rows:
            document_id = cast(str, row["documentId"])
            source_payload = canonical_json_bytes(row)
            cloned = dict(row)
            cloned["auxiliaryViews"] = {_VIEW_KEY: views[document_id].model_dump(mode="json")}
            output_payload_parts.append(canonical_json_bytes(cloned) + b"\n")
            lineage_rows.append(
                {
                    "documentId": document_id,
                    "split": split,
                    "sourcePath": input_config.path,
                    "sourceFileSha256": input_config.sha256,
                    "sourceRowNumber": row_number,
                    "sourceRowCanonicalSha256": sha256_bytes(source_payload),
                    "joinedRawTextSha256": row["joinedRawTextSha256"],
                    "targetCanonicalSha256": sha256_bytes(canonical_json_bytes(row["target"])),
                    "tableViewSha256": sha256_bytes(
                        canonical_json_bytes(views[document_id].model_dump(mode="json"))
                    ),
                }
            )
        payload = b"".join(output_payload_parts)
        output_path = output_root / f"{split}.jsonl"
        atomic_publish_bytes(output_path, payload)
        published_files.append(
            {
                "kind": f"{split}_records",
                "path": output_path.name,
                "records": len(split_rows),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )

    lineage_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in lineage_rows)
    lineage_path = output_root / "lineage.jsonl"
    atomic_publish_bytes(lineage_path, lineage_payload)
    published_files.append(
        {
            "kind": "lineage",
            "path": lineage_path.name,
            "records": len(lineage_rows),
            "bytes": len(lineage_payload),
            "sha256": sha256_bytes(lineage_payload),
        }
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": join_config.dataset_id,
        "view_key": _VIEW_KEY,
        "documents": len(source_rows),
        "table_pages": sum(view.documentPageCount for view in views.values()),
        "table_run_id": config.run.run_id,
        "table_dataset_manifest_path": str(table_manifest),
        "table_dataset_manifest_sha256": table_manifest_sha256,
        "source_inputs": [row.model_dump(mode="json") for row in join_config.inputs],
        "split_records": {split: len(rows) for split, rows in sorted(by_split.items())},
        "training_input_contract": {
            "unchanged_field": "joinedRawText",
            "auxiliary_view_is_not_injected_into_prompt": True,
        },
        "files": published_files,
    }
    manifest_path = output_root / "manifest.json"
    atomic_publish_json(manifest_path, manifest)
    _verify_joined_dataset(config, views=views)
    return manifest_path


def _verify_joined_dataset(
    config: TableViewConfig,
    *,
    views: dict[str, JoinedTableDocumentView],
) -> dict[str, int]:
    join_config = _require_join_config(config)
    output_root = Path(join_config.output_root)
    manifest_path = _canonical_regular_file(output_root / "manifest.json", context="join manifest")
    manifest = _strict_json(read_regular_file_bytes(manifest_path), context=str(manifest_path))
    if manifest.get("dataset_id") != join_config.dataset_id:
        raise TableViewError("joined manifest dataset_id differs from configuration")
    views = _document_views(config)

    expected_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for input_config in join_config.inputs:
        expected_by_split[input_config.split].extend(_read_input(input_config))
    seen: set[str] = set()
    for split, expected_rows in expected_by_split.items():
        output_path = _canonical_regular_file(
            output_root / f"{split}.jsonl", context=f"joined {split} split"
        )
        output_rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(read_regular_file_bytes(output_path).splitlines(), 1):
            output_rows.append(_strict_json(line, context=f"{output_path}:{line_number}"))
        if len(output_rows) != len(expected_rows):
            raise TableViewError(f"joined {split} row count differs from source")
        for source, cloned in zip(expected_rows, output_rows, strict=True):
            document_id = cast(str, source["documentId"])
            auxiliary = cloned.pop("auxiliaryViews", None)
            if cloned != source:
                raise TableViewError(f"joined clone changed a source field: {document_id}")
            expected_auxiliary = {_VIEW_KEY: views[document_id].model_dump(mode="json")}
            if auxiliary != expected_auxiliary:
                raise TableViewError(
                    f"joined table view differs from committed result: {document_id}"
                )
            if document_id in seen:
                raise TableViewError(f"duplicate documentId in joined outputs: {document_id}")
            seen.add(document_id)
    if seen != set(views):
        raise TableViewError("joined output document set differs from table-view inputs")
    return {
        "documents": len(seen),
        "pages": sum(view.documentPageCount for view in views.values()),
        "train_records": len(expected_by_split.get("train", [])),
        "validation_records": len(expected_by_split.get("validation", [])),
        "test_records": len(expected_by_split.get("test", [])),
    }


def verify_joined_dataset(config: TableViewConfig) -> dict[str, int]:
    """Independently prove source preservation and all table-view digests."""

    _require_join_config(config)
    return _verify_joined_dataset(config, views=_document_views(config))
