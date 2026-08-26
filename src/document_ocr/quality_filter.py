"""Audited, document-level quality projection for a local raw-OCR run.

The extraction ledger and its artifacts remain immutable.  This module validates
their provenance, applies explicit page-to-document exclusion rules, and writes a
manifest-last JSONL projection containing only complete, Latin-script documents
whose pages all ended normally.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import regex

from document_ocr.atomic import atomic_publish_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256, sha256_bytes
from document_ocr.models import (
    InferenceAttempt,
    PageExtractionFailure,
    PageExtractionRecord,
    PageProvenance,
    SourceObject,
)

ProgressCallback = Callable[[str, int, int], None]
DocumentStatus = Literal["pending", "failed", "incomplete", "complete"]
PageStatus = Literal["pending", "failed", "success"]

_SCHEMA_VERSION = 1
_DATASET_KIND = "glm-ocr-quality-filtered-page-extractions"
_NON_LATIN_LETTER = regex.compile(r"(?V1)[\p{L}&&\P{scx=Latin}]")
_LATIN_LETTER = regex.compile(r"(?V1)[\p{L}&&\p{scx=Latin}]")
_SCRIPT_PATTERNS = tuple(
    (name, regex.compile(rf"(?V1)\A\p{{scx={name}}}\Z"))
    for name in (
        "Arabic",
        "Han",
        "Cyrillic",
        "Hiragana",
        "Katakana",
        "Hebrew",
        "Greek",
        "Hangul",
        "Devanagari",
        "Thai",
        "Armenian",
        "Georgian",
    )
)
_HASH_CHUNK_SIZE = 1024 * 1024


class QualityFilterError(RuntimeError):
    """The source run or one of its immutable artifacts failed validation."""


@dataclass(frozen=True, slots=True)
class PageQualityEvidence:
    """The quality facts needed to assess one registered page."""

    extraction_id: str
    page_id: str
    page_index: int
    page_number: int
    status: PageStatus
    finish_reason: Literal["stop", "repetition"] | None
    raw_ocr_text: str | None
    failure: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class UnicodeQualityScan:
    """Exact Unicode-script evidence found in raw OCR text."""

    non_latin_occurrences_by_script: Mapping[str, int]
    non_latin_characters: tuple[Mapping[str, Any], ...]
    allowed_non_ascii_latin_occurrences: int
    allowed_non_ascii_latin_characters: tuple[Mapping[str, Any], ...]

    @property
    def contains_non_latin_letters(self) -> bool:
        return bool(self.non_latin_characters)


@dataclass(frozen=True, slots=True)
class DocumentQualityAssessment:
    """Document-level eligibility and all independently applicable reasons."""

    eligible: bool
    reasons: tuple[Mapping[str, Any], ...]
    unicode_scan: UnicodeQualityScan


@dataclass(frozen=True, slots=True)
class QualityFilterResult:
    """The committed quality projection and its deterministic summary."""

    output_dir: Path
    created: bool
    manifest_sha256: str
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ArtifactExpectation:
    path: Path
    allowed_root: Path
    size_bytes: int
    sha256: str
    kind: Literal["source_pdf", "page_image"]


@dataclass(frozen=True, slots=True)
class _ValidatedRun:
    run: Mapping[str, Any]
    documents: tuple[sqlite3.Row, ...]
    pages_by_document: Mapping[str, tuple[sqlite3.Row, ...]]
    attempts_by_extraction: Mapping[str, tuple[InferenceAttempt, ...]]
    source_by_document: Mapping[str, SourceObject]
    page_record_by_extraction: Mapping[str, PageExtractionRecord]
    page_failure_by_extraction: Mapping[str, PageExtractionFailure]
    source_expectations: tuple[_ArtifactExpectation, ...]
    raster_expectations: tuple[_ArtifactExpectation, ...]
    raw_response_bytes: int


def _strict_json_loads(payload: str | bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise QualityFilterError(f"JSON contains duplicate object key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(payload, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise QualityFilterError("artifact contains invalid JSON") from error


def _script_name(character: str) -> str:
    for script, pattern in _SCRIPT_PATTERNS:
        if pattern.fullmatch(character) is not None:
            return script
    name = unicodedata.name(character, "UNNAMED")
    return f"Other:{name.split(maxsplit=1)[0]}"


def _character_records(
    counts: Counter[str], *, include_script: bool
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    for character in sorted(counts, key=ord):
        record: dict[str, Any] = {
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "name": unicodedata.name(character, "UNNAMED"),
            "occurrences": counts[character],
        }
        if include_script:
            record["script"] = _script_name(character)
        records.append(record)
    return tuple(records)


def scan_unicode_quality(text: str) -> UnicodeQualityScan:
    """Classify OCR letters using Unicode Script_Extensions, without normalization.

    Extended Latin letters are allowed.  A letter is excluded only when its
    Script_Extensions set does not include Latin, which correctly retains values
    such as ``é`` and ``º`` while detecting Arabic, Han, Cyrillic, and
    other non-Latin scripts.
    """

    non_latin = Counter(match.group() for match in _NON_LATIN_LETTER.finditer(text))
    allowed_non_ascii_latin = Counter(
        character
        for character in text
        if ord(character) > 127 and _LATIN_LETTER.fullmatch(character) is not None
    )
    scripts: Counter[str] = Counter()
    for character, count in non_latin.items():
        scripts[_script_name(character)] += count
    return UnicodeQualityScan(
        non_latin_occurrences_by_script=dict(sorted(scripts.items())),
        non_latin_characters=_character_records(non_latin, include_script=True),
        allowed_non_ascii_latin_occurrences=sum(allowed_non_ascii_latin.values()),
        allowed_non_ascii_latin_characters=_character_records(
            allowed_non_ascii_latin, include_script=False
        ),
    )


def assess_document_quality(
    *,
    document_status: DocumentStatus,
    expected_page_count: int | None,
    pages: Sequence[PageQualityEvidence],
) -> DocumentQualityAssessment:
    """Apply the three whole-document exclusion rules to ordered page evidence."""

    page_indexes = [page.page_index for page in pages]
    if page_indexes != sorted(page_indexes) or len(page_indexes) != len(set(page_indexes)):
        raise QualityFilterError("document pages are not uniquely ordered by page_index")
    if expected_page_count is not None and expected_page_count <= 0:
        raise QualityFilterError("document page_count must be positive when present")

    reasons: list[Mapping[str, Any]] = []
    expected_indexes = set(range(expected_page_count or 0))
    observed_indexes = set(page_indexes)
    unsuccessful = [page for page in pages if page.status != "success"]
    if (
        document_status != "complete"
        or expected_page_count is None
        or observed_indexes != expected_indexes
        or unsuccessful
    ):
        reasons.append(
            {
                "code": "document_not_complete",
                "document_status": document_status,
                "expected_page_count": expected_page_count,
                "missing_page_indexes": sorted(expected_indexes - observed_indexes),
                "unsuccessful_pages": [
                    {
                        "extraction_id": page.extraction_id,
                        "page_id": page.page_id,
                        "page_index": page.page_index,
                        "page_number": page.page_number,
                        "status": page.status,
                        "failure": page.failure,
                    }
                    for page in unsuccessful
                ],
            }
        )

    repeated_pages = [page for page in pages if page.finish_reason == "repetition"]
    if repeated_pages:
        reasons.append(
            {
                "code": "ocr_repetition_detected",
                "pages": [
                    {
                        "extraction_id": page.extraction_id,
                        "page_id": page.page_id,
                        "page_index": page.page_index,
                        "page_number": page.page_number,
                    }
                    for page in repeated_pages
                ],
            }
        )

    joined_text = "\n".join(page.raw_ocr_text or "" for page in pages)
    unicode_scan = scan_unicode_quality(joined_text)
    if unicode_scan.contains_non_latin_letters:
        reasons.append(
            {
                "code": "non_latin_script_in_raw_ocr",
                "occurrences_by_script": unicode_scan.non_latin_occurrences_by_script,
                "characters": list(unicode_scan.non_latin_characters),
            }
        )
    return DocumentQualityAssessment(
        eligible=not reasons,
        reasons=tuple(reasons),
        unicode_scan=unicode_scan,
    )


def _canonical_regular_path(path: Path, *, allowed_root: Path | None = None) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
        details = path.stat(follow_symlinks=False)
    except OSError as error:
        raise QualityFilterError(f"required artifact is missing: {path}") from error
    if resolved != absolute or not stat.S_ISREG(details.st_mode):
        raise QualityFilterError(f"artifact is not a canonical regular file: {path}")
    if allowed_root is not None:
        root = allowed_root.resolve(strict=True)
        if root != Path(os.path.abspath(allowed_root)) or not resolved.is_relative_to(root):
            raise QualityFilterError(f"artifact escapes its allowed root: {path}")
    return resolved


def _hash_regular_file(path: Path, *, allowed_root: Path) -> tuple[int, str]:
    resolved = _canonical_regular_path(path, allowed_root=allowed_root)
    missing = [name for name in ("O_CLOEXEC", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing:
        raise QualityFilterError(
            "safe artifact hashing requires operating-system flags: " + ", ".join(missing)
        )
    try:
        descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise QualityFilterError(f"artifact cannot be opened safely: {path}") from error
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise QualityFilterError(f"artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
                size_bytes += len(chunk)
    finally:
        os.close(descriptor)
    return size_bytes, digest.hexdigest()


def _validate_expected_artifact(expectation: _ArtifactExpectation) -> int:
    size_bytes, digest = _hash_regular_file(expectation.path, allowed_root=expectation.allowed_root)
    if size_bytes != expectation.size_bytes or digest != expectation.sha256:
        raise QualityFilterError(
            f"{expectation.kind} no longer matches ledger provenance: {expectation.path}"
        )
    return size_bytes


def _validate_artifacts(
    expectations: Sequence[_ArtifactExpectation],
    *,
    workers: int,
    progress: ProgressCallback | None,
    phase: str,
) -> int:
    if workers <= 0:
        raise ValueError("artifact verification workers must be positive")
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(expectations) or 1)) as executor:
        futures = {
            executor.submit(_validate_expected_artifact, expectation): expectation
            for expectation in expectations
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                total_bytes += future.result()
                if progress is not None and (completed % 50 == 0 or completed == len(futures)):
                    progress(phase, completed, len(futures))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return total_bytes


def _source_from_page(record: PageExtractionRecord | PageExtractionFailure) -> SourceObject:
    return SourceObject.model_validate(
        {field: getattr(record, field) for field in SourceObject.model_fields}, strict=True
    )


def _validate_page_identity(
    record: PageExtractionRecord | PageExtractionFailure,
    *,
    row: sqlite3.Row,
    source: SourceObject,
    page_count: int,
    run: Mapping[str, Any],
) -> None:
    if (
        record.run_id != run["run_id"]
        or record.config_sha256 != run["config_sha256"]
        or record.pipeline_fingerprint != run["pipeline_fingerprint"]
        or record.document_id != row["document_id"]
        or record.extraction_id != row["extraction_id"]
        or record.page_id != row["page_id"]
        or record.page_index != row["page_index"]
        or record.document_page_count != page_count
        or _source_from_page(record) != source
    ):
        raise QualityFilterError(f"page provenance conflicts with ledger: {row['extraction_id']}")


def _validate_raw_response(run_dir: Path, record: PageExtractionRecord) -> int:
    relative = PurePosixPath(record.raw_response_path)
    path = run_dir.joinpath(*relative.parts)
    _canonical_regular_path(path, allowed_root=run_dir)
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != record.raw_response_sha256:
        raise QualityFilterError(f"raw response hash mismatch: {record.raw_response_path}")
    parsed = _strict_json_loads(payload)
    if not isinstance(parsed, dict):
        raise QualityFilterError("raw vLLM response root must be an object")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise QualityFilterError("raw vLLM response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    usage = parsed.get("usage")
    if not isinstance(message, dict) or not isinstance(usage, dict):
        raise QualityFilterError("raw vLLM response is missing message or usage")
    if (
        message.get("content") != record.raw_ocr_text
        or choice.get("finish_reason") != record.inference_finish_reason
        or parsed.get("model") != record.inference_served_model_name
        or usage.get("prompt_tokens") != record.inference_prompt_tokens
        or usage.get("completion_tokens") != record.inference_completion_tokens
    ):
        raise QualityFilterError(
            f"raw response content conflicts with page record: {record.extraction_id}"
        )
    if (
        record.inference_finish_reason == "repetition"
        and choice.get("stop_reason") != "repetition_detected"
    ):
        raise QualityFilterError(
            f"repetition result lacks server detection evidence: {record.extraction_id}"
        )
    return len(payload)


def _attempts_for_page(
    *,
    attempt_rows: Sequence[sqlite3.Row],
    page_row: sqlite3.Row,
    record: PageExtractionRecord | PageExtractionFailure | None,
) -> tuple[InferenceAttempt, ...]:
    attempts = tuple(
        InferenceAttempt.model_validate_json(row["attempt_json"], strict=True)
        for row in attempt_rows
    )
    numbers = [attempt.attempt_number for attempt in attempts]
    if numbers != list(range(1, len(attempts) + 1)):
        raise QualityFilterError(f"attempt history is noncontiguous: {page_row['extraction_id']}")
    for attempt in attempts:
        if (
            attempt.run_id != page_row["run_id"]
            or attempt.extraction_id != page_row["extraction_id"]
            or attempt.document_id != page_row["document_id"]
            or attempt.page_id != page_row["page_id"]
            or attempt.page_index != page_row["page_index"]
        ):
            raise QualityFilterError(
                f"attempt provenance conflicts with page: {page_row['extraction_id']}"
            )
        if record is not None and any(
            getattr(attempt, field) != getattr(record, field)
            for field in PageProvenance.model_fields
        ):
            raise QualityFilterError(
                f"attempt provenance conflicts with terminal record: {page_row['extraction_id']}"
            )
    if record is None:
        if attempts:
            raise QualityFilterError("pending page unexpectedly has attempt history")
        return attempts
    if len(attempts) != record.inference_attempt_count:
        raise QualityFilterError(
            f"attempt count conflicts with page record: {page_row['extraction_id']}"
        )
    if not attempts:
        if isinstance(record, PageExtractionFailure) and record.inference_attempt_count == 0:
            return attempts
        raise QualityFilterError("terminal page record has no attempt history")
    expected_final = (
        "success"
        if isinstance(record, PageExtractionRecord)
        or (isinstance(record, PageExtractionFailure) and record.failure_stage == "persist")
        else "terminal_error"
    )
    # A failed page is explicitly resumable. Each invocation validates its own
    # request sequence before the ledger appends it, so an earlier invocation
    # may end in terminal_error (inference failure) or success (persist failure)
    # before a later invocation establishes the current terminal page state.
    # The immutable history has no invocation marker; its provable cross-run
    # invariants are contiguous numbering, common provenance, total count, and
    # an outcome matching the current final record.
    if attempts[-1].attempt_outcome != expected_final:
        raise QualityFilterError(
            f"attempt outcomes conflict with terminal page state: {page_row['extraction_id']}"
        )
    return attempts


def _read_validated_run(
    run_dir: Path,
    *,
    run_id: str,
    progress: ProgressCallback | None,
) -> _ValidatedRun:
    absolute_run_dir = Path(os.path.abspath(run_dir))
    run_dir = run_dir.resolve(strict=True)
    if run_dir != absolute_run_dir or not run_dir.is_dir():
        raise QualityFilterError(f"run directory must be canonical: {run_dir}")
    ledger_path = _canonical_regular_path(run_dir / "state.sqlite3", allowed_root=run_dir)
    connection = sqlite3.connect(f"file:{ledger_path}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if [row[0] for row in integrity] != ["ok"]:
            raise QualityFilterError(f"SQLite integrity check failed: {integrity!r}")
        run_row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if run_row is None:
            raise QualityFilterError(f"run_id is absent from ledger: {run_id}")
        run = dict(run_row)
        documents = tuple(
            connection.execute(
                "SELECT * FROM documents WHERE run_id = ? ORDER BY document_id", (run_id,)
            ).fetchall()
        )
        pages = tuple(
            connection.execute(
                """
                SELECT * FROM pages WHERE run_id = ?
                ORDER BY document_id, page_index
                """,
                (run_id,),
            ).fetchall()
        )
        attempt_rows = tuple(
            connection.execute(
                """
                SELECT * FROM attempts WHERE run_id = ?
                ORDER BY extraction_id, attempt_number
                """,
                (run_id,),
            ).fetchall()
        )
    finally:
        connection.close()

    inventory_payload = read_regular_file_bytes(
        _canonical_regular_path(run_dir / "inventory.jsonl", allowed_root=run_dir)
    )
    if sha256_bytes(inventory_payload) != run["inventory_sha256"]:
        raise QualityFilterError("inventory.jsonl does not match the ledger digest")
    digest_sidecar = read_regular_file_bytes(
        _canonical_regular_path(run_dir / "inventory.jsonl.sha256", allowed_root=run_dir)
    ).decode("ascii")
    if digest_sidecar != f"{run['inventory_sha256']}  inventory.jsonl\n":
        raise QualityFilterError("inventory digest sidecar conflicts with the ledger")
    inventory_sources: dict[str, SourceObject] = {}
    for line_number, line in enumerate(inventory_payload.splitlines(), start=1):
        try:
            source = SourceObject.model_validate_json(line, strict=True)
        except Exception as error:
            raise QualityFilterError(f"invalid inventory row {line_number}") from error
        if source.document_id in inventory_sources:
            raise QualityFilterError(f"duplicate inventory document_id {source.document_id}")
        inventory_sources[source.document_id] = source
    if len(inventory_sources) != run["inventory_document_count"]:
        raise QualityFilterError("inventory row count conflicts with the ledger")
    provenance_payload = read_regular_file_bytes(
        _canonical_regular_path(run_dir / "run-provenance.json", allowed_root=run_dir)
    )
    if _strict_json_loads(provenance_payload) != _strict_json_loads(run["provenance_json"]):
        raise QualityFilterError("run-provenance.json conflicts with the ledger")
    if len(documents) != run["inventory_document_count"]:
        raise QualityFilterError("registered document count conflicts with inventory")

    pages_by_document_mutable: dict[str, list[sqlite3.Row]] = defaultdict(list)
    page_by_extraction: dict[str, sqlite3.Row] = {}
    for page in pages:
        pages_by_document_mutable[cast(str, page["document_id"])].append(page)
        page_by_extraction[cast(str, page["extraction_id"])] = page
    attempt_rows_by_extraction: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for attempt in attempt_rows:
        extraction_id = cast(str, attempt["extraction_id"])
        if extraction_id not in page_by_extraction:
            raise QualityFilterError(f"attempt references unknown extraction: {extraction_id}")
        attempt_rows_by_extraction[extraction_id].append(attempt)

    source_by_document: dict[str, SourceObject] = {}
    page_record_by_extraction: dict[str, PageExtractionRecord] = {}
    page_failure_by_extraction: dict[str, PageExtractionFailure] = {}
    attempts_by_extraction: dict[str, tuple[InferenceAttempt, ...]] = {}
    source_expectations: list[_ArtifactExpectation] = []
    raster_expectations: list[_ArtifactExpectation] = []
    raw_response_bytes = 0

    for completed, document in enumerate(documents, start=1):
        document_id = cast(str, document["document_id"])
        try:
            source = SourceObject.model_validate_json(document["source_json"], strict=True)
        except Exception as error:
            raise QualityFilterError(f"invalid source record for {document_id}") from error
        if source != inventory_sources.get(document_id):
            raise QualityFilterError(f"document source conflicts with inventory: {document_id}")
        if source.source_type != "local" or source.local_canonical_path is None:
            raise QualityFilterError("quality projection requires retained local source files")
        if document["source_sha256"] != source.source_sha256 or source.source_sha256 is None:
            raise QualityFilterError(f"document source hash is not proven: {document_id}")
        source_path = Path(source.local_canonical_path)
        source_path = _canonical_regular_path(source_path, allowed_root=source_path.parent)
        details = source_path.stat(follow_symlinks=False)
        if (
            source.local_device != details.st_dev
            or source.local_inode != details.st_ino
            or source.local_mtime_ns != details.st_mtime_ns
        ):
            raise QualityFilterError(f"local source identity drifted: {source_path}")
        source_expectations.append(
            _ArtifactExpectation(
                path=source_path,
                allowed_root=source_path.parent,
                size_bytes=source.source_size_bytes,
                sha256=source.source_sha256,
                kind="source_pdf",
            )
        )
        source_by_document[document_id] = source
        document_pages = pages_by_document_mutable.get(document_id, [])
        page_count_value = document["page_count"]
        page_count = cast(int | None, page_count_value)
        if (
            page_count is not None
            and [row["page_index"] for row in document_pages] != list(range(page_count))
            and document["status"] == "complete"
        ):
            raise QualityFilterError(f"complete document page inventory drifted: {document_id}")
        for page in document_pages:
            extraction_id = cast(str, page["extraction_id"])
            status = cast(PageStatus, page["status"])
            record: PageExtractionRecord | PageExtractionFailure | None
            if status == "success":
                if page["failure_json"] is not None or page["result_json"] is None:
                    raise QualityFilterError(f"invalid successful page row: {extraction_id}")
                result = PageExtractionRecord.model_validate_json(page["result_json"], strict=True)
                if page_count is None:
                    raise QualityFilterError(f"successful page has no page_count: {extraction_id}")
                _validate_page_identity(
                    result,
                    row=page,
                    source=source,
                    page_count=page_count,
                    run=run,
                )
                if (
                    page["ocr_text_sha256"] != result.raw_ocr_text_sha256
                    or page["raw_response_sha256"] != result.raw_response_sha256
                ):
                    raise QualityFilterError(f"page digest columns drifted: {extraction_id}")
                raw_response_bytes += _validate_raw_response(run_dir, result)
                if result.raster_path is not None:
                    relative = PurePosixPath(result.raster_path)
                    raster_expectations.append(
                        _ArtifactExpectation(
                            path=run_dir.joinpath(*relative.parts),
                            allowed_root=run_dir,
                            size_bytes=result.raster_size_bytes,
                            sha256=result.raster_sha256,
                            kind="page_image",
                        )
                    )
                page_record_by_extraction[extraction_id] = result
                record = result
            elif status == "failed":
                if page["result_json"] is not None or page["failure_json"] is None:
                    raise QualityFilterError(f"invalid failed page row: {extraction_id}")
                failure = PageExtractionFailure.model_validate_json(
                    page["failure_json"], strict=True
                )
                if page_count is None:
                    raise QualityFilterError(f"failed page has no page_count: {extraction_id}")
                _validate_page_identity(
                    failure,
                    row=page,
                    source=source,
                    page_count=page_count,
                    run=run,
                )
                page_failure_by_extraction[extraction_id] = failure
                record = failure
            elif status == "pending":
                if page["result_json"] is not None or page["failure_json"] is not None:
                    raise QualityFilterError(f"invalid pending page row: {extraction_id}")
                record = None
            else:
                raise QualityFilterError(f"unknown page status {status!r}")
            attempts_by_extraction[extraction_id] = _attempts_for_page(
                attempt_rows=attempt_rows_by_extraction.get(extraction_id, []),
                page_row=page,
                record=record,
            )
        if progress is not None and (completed % 50 == 0 or completed == len(documents)):
            progress("validate_ledger_and_raw_responses", completed, len(documents))

    return _ValidatedRun(
        run=run,
        documents=documents,
        pages_by_document={key: tuple(value) for key, value in pages_by_document_mutable.items()},
        attempts_by_extraction=attempts_by_extraction,
        source_by_document=source_by_document,
        page_record_by_extraction=page_record_by_extraction,
        page_failure_by_extraction=page_failure_by_extraction,
        source_expectations=tuple(source_expectations),
        raster_expectations=tuple(raster_expectations),
        raw_response_bytes=raw_response_bytes,
    )


def _page_quality_evidence(
    page: sqlite3.Row,
    *,
    validated: _ValidatedRun,
) -> PageQualityEvidence:
    extraction_id = cast(str, page["extraction_id"])
    record = validated.page_record_by_extraction.get(extraction_id)
    failure = validated.page_failure_by_extraction.get(extraction_id)
    return PageQualityEvidence(
        extraction_id=extraction_id,
        page_id=cast(str, page["page_id"]),
        page_index=cast(int, page["page_index"]),
        page_number=cast(int, page["page_index"]) + 1,
        status=cast(PageStatus, page["status"]),
        finish_reason=record.inference_finish_reason if record is not None else None,
        raw_ocr_text=record.raw_ocr_text if record is not None else None,
        failure=failure.model_dump(mode="json") if failure is not None else None,
    )


def _page_reference(record: PageExtractionRecord) -> Mapping[str, Any]:
    return {
        "extraction_id": record.extraction_id,
        "page_id": record.page_id,
        "page_index": record.page_index,
        "page_number": record.page_number,
        "raw_ocr_text_sha256": record.raw_ocr_text_sha256,
        "raw_response_path": record.raw_response_path,
        "raster_path": record.raster_path,
        "raster_sha256": record.raster_sha256,
    }


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> tuple[bytes, int]:
    materialized = tuple(rows)
    return b"".join(canonical_json_bytes(row) + b"\n" for row in materialized), len(materialized)


def _file_manifest(path: str, payload: bytes, rows: int) -> Mapping[str, Any]:
    return {
        "path": path,
        "rows": rows,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def filter_local_ocr_run(
    *,
    run_dir: Path,
    run_id: str,
    output_dir: Path,
    artifact_verification_workers: int = 8,
    progress: ProgressCallback | None = None,
) -> QualityFilterResult:
    """Validate and publish a non-destructive quality projection of one OCR run."""

    validated = _read_validated_run(run_dir, run_id=run_id, progress=progress)
    source_bytes = _validate_artifacts(
        validated.source_expectations,
        workers=artifact_verification_workers,
        progress=progress,
        phase="verify_source_pdfs",
    )
    raster_bytes = _validate_artifacts(
        validated.raster_expectations,
        workers=artifact_verification_workers,
        progress=progress,
        phase="verify_page_images",
    )

    eligibility_rows: list[Mapping[str, Any]] = []
    eligible_document_rows: list[Mapping[str, Any]] = []
    excluded_document_rows: list[Mapping[str, Any]] = []
    eligible_page_rows: list[Mapping[str, Any]] = []
    failed_page_rows: list[Mapping[str, Any]] = []
    reason_document_counts: Counter[str] = Counter()
    reason_combination_counts: Counter[str] = Counter()
    script_document_counts: Counter[str] = Counter()
    script_occurrences: Counter[str] = Counter()
    allowed_latin_documents = 0
    allowed_latin_occurrences = 0
    allowed_latin_characters: Counter[str] = Counter()

    for completed, document in enumerate(validated.documents, start=1):
        document_id = cast(str, document["document_id"])
        source = validated.source_by_document[document_id]
        pages = validated.pages_by_document.get(document_id, ())
        evidence = tuple(_page_quality_evidence(page, validated=validated) for page in pages)
        assessment = assess_document_quality(
            document_status=cast(DocumentStatus, document["status"]),
            expected_page_count=cast(int | None, document["page_count"]),
            pages=evidence,
        )
        codes = tuple(cast(str, reason["code"]) for reason in assessment.reasons)
        for code in codes:
            reason_document_counts[code] += 1
        if codes:
            reason_combination_counts["+".join(codes)] += 1
        for script, count in assessment.unicode_scan.non_latin_occurrences_by_script.items():
            script_document_counts[script] += 1
            script_occurrences[script] += count
        if assessment.unicode_scan.allowed_non_ascii_latin_occurrences:
            allowed_latin_documents += 1
            allowed_latin_occurrences += assessment.unicode_scan.allowed_non_ascii_latin_occurrences
            for record in assessment.unicode_scan.allowed_non_ascii_latin_characters:
                allowed_latin_characters[cast(str, record["character"])] += cast(
                    int, record["occurrences"]
                )

        eligibility = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": run_id,
            "document_id": document_id,
            "document_status": document["status"],
            "document_page_count": document["page_count"],
            "eligible": assessment.eligible,
            "exclusion_reasons": list(assessment.reasons),
            "source": source.model_dump(mode="json"),
        }
        eligibility_rows.append(eligibility)
        if assessment.eligible:
            page_records = tuple(
                validated.page_record_by_extraction[cast(str, page["extraction_id"])]
                for page in pages
            )
            if [record.page_index for record in page_records] != list(
                range(cast(int, document["page_count"]))
            ) or any(record.inference_finish_reason != "stop" for record in page_records):
                raise QualityFilterError(
                    f"eligible document violates page order or finish policy: {document_id}"
                )
            eligible_document_rows.append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "run_id": run_id,
                    "document_id": document_id,
                    "document_page_count": document["page_count"],
                    "source": source.model_dump(mode="json"),
                    "pages": [_page_reference(record) for record in page_records],
                }
            )
            eligible_page_rows.extend(record.model_dump(mode="json") for record in page_records)
        else:
            excluded_document_rows.append(eligibility)
        for page in pages:
            extraction_id = cast(str, page["extraction_id"])
            failure = validated.page_failure_by_extraction.get(extraction_id)
            if failure is not None:
                failed_page_rows.append(
                    {
                        "failure": failure.model_dump(mode="json"),
                        "attempts": [
                            attempt.model_dump(mode="json")
                            for attempt in validated.attempts_by_extraction[extraction_id]
                        ],
                    }
                )
        if progress is not None and (completed % 50 == 0 or completed == len(validated.documents)):
            progress("assess_documents", completed, len(validated.documents))

    total_documents = len(validated.documents)
    total_pages = sum(len(pages) for pages in validated.pages_by_document.values())
    successful_pages = len(validated.page_record_by_extraction)
    failed_pages = len(validated.page_failure_by_extraction)
    pending_pages = total_pages - successful_pages - failed_pages
    eligible_pages = len(eligible_page_rows)
    policy: Mapping[str, Any] = {
        "scope": "whole_document",
        "rules": [
            {
                "code": "document_not_complete",
                "action": "exclude the entire document when every expected page did not succeed",
            },
            {
                "code": "ocr_repetition_detected",
                "action": (
                    "exclude the entire document when any successful page has vLLM "
                    "finish_reason=repetition and stop_reason=repetition_detected"
                ),
            },
            {
                "code": "non_latin_script_in_raw_ocr",
                "action": (
                    "exclude the entire document when any Unicode Letter has no Latin "
                    "Script_Extensions membership"
                ),
            },
        ],
        "extended_latin_allowed": True,
        "unicode_normalization": "none",
        "page_order": "ascending zero-based page_index, required contiguous per eligible document",
    }
    summary: Mapping[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset_kind": _DATASET_KIND,
        "run_id": run_id,
        "source_run_status": validated.run["status"],
        "documents": {
            "total": total_documents,
            "eligible": len(eligible_document_rows),
            "excluded_unique": len(excluded_document_rows),
            "complete_in_ledger": sum(
                document["status"] == "complete" for document in validated.documents
            ),
            "incomplete_in_ledger": sum(
                document["status"] != "complete" for document in validated.documents
            ),
        },
        "pages": {
            "total": total_pages,
            "successful": successful_pages,
            "failed": failed_pages,
            "pending": pending_pages,
            "eligible": eligible_pages,
            "excluded_source_pages": total_pages - eligible_pages,
            "successful_finish_stop": sum(
                record.inference_finish_reason == "stop"
                for record in validated.page_record_by_extraction.values()
            ),
            "successful_finish_repetition": sum(
                record.inference_finish_reason == "repetition"
                for record in validated.page_record_by_extraction.values()
            ),
        },
        "exclusions": {
            "documents_by_reason": dict(sorted(reason_document_counts.items())),
            "documents_by_reason_combination": dict(sorted(reason_combination_counts.items())),
        },
        "unicode": {
            "documents_by_non_latin_script": dict(sorted(script_document_counts.items())),
            "non_latin_letter_occurrences_by_script": dict(sorted(script_occurrences.items())),
            "documents_with_allowed_non_ascii_latin_letters": allowed_latin_documents,
            "allowed_non_ascii_latin_letter_occurrences": allowed_latin_occurrences,
            "allowed_non_ascii_latin_characters": list(
                _character_records(allowed_latin_characters, include_script=False)
            ),
        },
        "artifact_validation": {
            "ledger_integrity_check": "ok",
            "inventory_documents": len(validated.source_by_document),
            "source_pdfs_hashed": len(validated.source_expectations),
            "source_pdf_bytes_hashed": source_bytes,
            "raw_responses_hashed_and_parsed": successful_pages,
            "raw_response_bytes_hashed": validated.raw_response_bytes,
            "page_images_hashed": len(validated.raster_expectations),
            "page_image_bytes_hashed": raster_bytes,
        },
        "quality_policy": policy,
    }
    if total_documents != validated.run["inventory_document_count"] or pending_pages < 0:
        raise QualityFilterError("source run totals changed during quality projection")
    if len(eligible_document_rows) + len(excluded_document_rows) != total_documents:
        raise QualityFilterError("document partition is not exhaustive")
    if sum(cast(int, row["document_page_count"]) for row in eligible_document_rows) != len(
        eligible_page_rows
    ):
        raise QualityFilterError("eligible page rows do not match eligible document page counts")

    payloads_with_rows: list[tuple[str, bytes, int]] = []
    for path, rows in (
        ("eligibility.jsonl", eligibility_rows),
        ("eligible-documents.jsonl", eligible_document_rows),
        ("excluded-documents.jsonl", excluded_document_rows),
        ("eligible-pages.jsonl", eligible_page_rows),
        ("failed-pages.jsonl", failed_page_rows),
    ):
        payload, row_count = _jsonl_payload(rows)
        payloads_with_rows.append((path, payload, row_count))
    summary_payload = canonical_json_bytes(summary) + b"\n"
    payloads_with_rows.append(("summary.json", summary_payload, 1))

    run_dir_resolved = run_dir.resolve(strict=True)
    source_files = {
        "state.sqlite3": {
            "bytes": (run_dir_resolved / "state.sqlite3").stat().st_size,
            "sha256": _hash_regular_file(
                run_dir_resolved / "state.sqlite3", allowed_root=run_dir_resolved
            )[1],
        },
        "inventory.jsonl": {
            "bytes": (run_dir_resolved / "inventory.jsonl").stat().st_size,
            "sha256": validated.run["inventory_sha256"],
        },
        "run-provenance.json": {
            "bytes": (run_dir_resolved / "run-provenance.json").stat().st_size,
            "sha256": _hash_regular_file(
                run_dir_resolved / "run-provenance.json", allowed_root=run_dir_resolved
            )[1],
        },
    }
    manifest: Mapping[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "dataset_kind": _DATASET_KIND,
        "run_id": run_id,
        "source": {
            "run_directory": str(run_dir_resolved),
            "config_sha256": validated.run["config_sha256"],
            "pipeline_fingerprint": validated.run["pipeline_fingerprint"],
            "inventory_sha256": validated.run["inventory_sha256"],
            "files": source_files,
        },
        "quality_policy_sha256": canonical_json_sha256(policy),
        "summary_sha256": sha256_bytes(summary_payload),
        "files": [
            _file_manifest(path, payload, rows) for path, payload, rows in payloads_with_rows
        ],
        "publication": "manifest_last_immutable",
    }
    manifest_payload = canonical_json_bytes(manifest) + b"\n"

    output_dir = Path(os.path.abspath(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve(strict=True) != output_dir or not output_dir.is_dir():
        raise QualityFilterError(f"output directory must be canonical: {output_dir}")
    created = False
    for path, payload, _ in payloads_with_rows:
        created = atomic_publish_bytes(output_dir / path, payload) or created
    created = atomic_publish_bytes(output_dir / "manifest.json", manifest_payload) or created
    return QualityFilterResult(
        output_dir=output_dir,
        created=created,
        manifest_sha256=sha256_bytes(manifest_payload),
        summary=summary,
    )
