"""Frozen work-item inventory, deterministic sampling, and OCR evidence checks."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import model_validator
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, stable_id
from document_ocr.label_schemas.bill_of_lading import (
    BillOfLadingAnnotation,
    BillOfLadingExclusion,
    BillOfLadingLabel,
)
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingDualCargoAnnotation
from document_ocr.label_schemas.common import (
    ExtractionSourceReference,
    LabelSchemaModel,
    RawOcrAnchor,
    RawOcrValueEvidence,
)
from document_ocr.labeling_agents.config import AgentLabelingConfig, PromptArtifactConfig
from document_ocr.labeling_agents.models import PdfConstructionMethod

_PAGE_HEADER = re.compile(r"(?:\A|\n\n)--- PAGE ([1-9][0-9]*) ---\n")
_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{64}$")


class WorkItemError(RuntimeError):
    """A work item or one of its provenance artifacts is invalid."""


class AgentWorkItem(LabelSchemaModel):
    source: ExtractionSourceReference
    joinedRawText: str

    @model_validator(mode="after")
    def raw_text_matches_source_digest(self) -> AgentWorkItem:
        if not self.joinedRawText.strip():
            raise ValueError("joinedRawText must contain non-whitespace text")
        if sha256_bytes(self.joinedRawText.encode("utf-8")) != self.source.joinedRawTextSha256:
            raise ValueError("joinedRawTextSha256 does not match joinedRawText")
        pages = page_texts(self.joinedRawText)
        if tuple(pages) != tuple(range(1, self.source.documentPageCount + 1)):
            raise ValueError("joinedRawText pages must be complete and ordered")
        for page in self.source.pages:
            if sha256_bytes(pages[page.pageNumber].encode("utf-8")) != page.rawOcrTextSha256:
                raise ValueError("joinedRawText page hash differs from source page reference")
        return self


@dataclass(frozen=True, slots=True)
class InventoriedWorkItem:
    root_id: str
    path: Path
    sha256: str
    item: AgentWorkItem

    def inventory_row(self) -> dict[str, Any]:
        return {
            "documentId": self.item.source.documentId,
            "documentPageCount": self.item.source.documentPageCount,
            "extractionRunId": self.item.source.extractionRunId,
            "joinedRawTextSha256": self.item.source.joinedRawTextSha256,
            "path": str(self.path),
            "rootId": self.root_id,
            "sourceSha256": self.item.source.sourceSha256,
            "workItemSha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PdfAssistancePayload:
    data: bytes
    source_pdf_sha256: str
    source_page_numbers: tuple[int, ...]
    attachment_pdf_sha256: str
    construction_method: PdfConstructionMethod
    media_type: str = "application/pdf"


@dataclass(frozen=True, slots=True)
class AgentRunPaths:
    output_root: Path
    run_root: Path
    inventory: Path
    selection: Path
    resolved_config: Path
    prompt_root: Path
    work_item_root: Path
    state_root: Path
    accepted_root: Path
    exclusions_root: Path
    needs_review_root: Path
    training_root: Path


def page_texts(joined_raw_text: str) -> dict[int, str]:
    matches = list(_PAGE_HEADER.finditer(joined_raw_text))
    if not matches or matches[0].start() != 0:
        raise ValueError("joinedRawText must start with an exact page delimiter")
    pages: dict[int, str] = {}
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        content_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(joined_raw_text)
        )
        content = joined_raw_text[match.end() : content_end]
        if content.endswith("\n\n"):
            content = content[:-2]
        if page_number in pages:
            raise ValueError("joinedRawText contains a duplicate page delimiter")
        pages[page_number] = content
    return pages


def validate_raw_ocr_evidence(
    pages: dict[int, str],
    records: Sequence[RawOcrValueEvidence],
) -> None:
    """Validate exact evidence by raw-value order while allowing overlapping excerpts."""

    previous_page = 0
    previous_raw_end = 0
    for record in records:
        source = pages.get(record.pageNumber)
        if source is None or record.pageNumber < previous_page:
            raise WorkItemError("OCR evidence is not verbatim/in source order")
        raw_inside_excerpt = record.ocrExcerpt.find(record.rawValue)
        if raw_inside_excerpt < 0:
            raise WorkItemError("OCR evidence is not verbatim/in source order")
        minimum_raw_offset = previous_raw_end if record.pageNumber == previous_page else 0
        excerpt_search = 0
        raw_offset = -1
        while True:
            excerpt_offset = source.find(record.ocrExcerpt, excerpt_search)
            if excerpt_offset < 0:
                break
            candidate = excerpt_offset + raw_inside_excerpt
            if candidate >= minimum_raw_offset:
                raw_offset = candidate
                break
            excerpt_search = excerpt_offset + 1
        if raw_offset < 0:
            raise WorkItemError("OCR evidence is not verbatim/in source order")
        previous_page = record.pageNumber
        previous_raw_end = raw_offset + len(record.rawValue)


def _context_excerpt(source: str, start: int, end: int) -> str:
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer(r"\n", source))
    start_line = max(index for index, value in enumerate(line_starts) if value <= start)
    end_line = max(index for index, value in enumerate(line_starts) if value < max(end, 1))
    excerpt_start = line_starts[max(0, start_line - 2)]
    next_line = end_line + 2
    excerpt_end = line_starts[next_line] - 1 if next_line < len(line_starts) else len(source)
    excerpt = source[excerpt_start:excerpt_end]
    return excerpt if excerpt.strip() else source[start:end]


def materialize_raw_ocr_evidence(
    pages: dict[int, str],
    anchors: Sequence[RawOcrAnchor],
) -> tuple[RawOcrValueEvidence, ...]:
    """Resolve model-selected values to exact, locally generated OCR excerpts.

    Repeated values are assigned to distinct source occurrences, then all
    anchors are sorted by their exact local offsets.  This permits a layout
    helper to describe a table in logical row order even when flattened OCR is
    column-major, without permitting one OCR occurrence to support two facts.
    Invented or abbreviated context remains impossible by construction.
    """

    positions_by_key: dict[tuple[int, str], tuple[int, ...]] = {}

    def positions(anchor: RawOcrAnchor) -> tuple[int, ...]:
        key = (anchor.pageNumber, anchor.rawValue)
        existing = positions_by_key.get(key)
        if existing is not None:
            return existing
        source = pages[anchor.pageNumber]
        offsets: list[int] = []
        start_at = 0
        while (raw_offset := source.find(anchor.rawValue, start_at)) >= 0:
            offsets.append(raw_offset)
            start_at = raw_offset + len(anchor.rawValue)
        result = tuple(offsets)
        positions_by_key[key] = result
        return result

    def value_at(anchor: RawOcrAnchor, raw_offset: int) -> RawOcrValueEvidence:
        source = pages[anchor.pageNumber]
        raw_end = raw_offset + len(anchor.rawValue)
        excerpt = _context_excerpt(source, raw_offset, raw_end)
        occurrences = positions(anchor)
        occurrence_index = occurrences.index(raw_offset)
        if len(occurrences) > 1:
            excerpt_offset = source.find(excerpt)
            if excerpt_offset < 0:
                raise WorkItemError("locally generated OCR excerpt is absent")
            excerpt_end = excerpt_offset + len(excerpt)
            if occurrence_index > 0:
                excerpt_offset = max(
                    excerpt_offset,
                    occurrences[occurrence_index - 1] + len(anchor.rawValue),
                )
            if occurrence_index + 1 < len(occurrences):
                excerpt_end = min(excerpt_end, occurrences[occurrence_index + 1])
            excerpt = source[excerpt_offset:excerpt_end]
        return RawOcrValueEvidence.model_validate(
            {
                "pageNumber": anchor.pageNumber,
                "rawValue": anchor.rawValue,
                "ocrExcerpt": excerpt,
            },
            strict=True,
        )

    occurrence_by_key: dict[tuple[int, str], int] = {}
    resolved: list[tuple[int, int, RawOcrAnchor]] = []
    for anchor in anchors:
        source = pages.get(anchor.pageNumber)
        if source is None:
            raise WorkItemError(f"OCR anchor cites absent page {anchor.pageNumber}")
        key = (anchor.pageNumber, anchor.rawValue)
        occurrence = occurrence_by_key.get(key, 0) + 1
        occurrence_by_key[key] = occurrence
        available = positions(anchor)
        if len(available) < occurrence:
            raise WorkItemError(
                f"OCR anchor occurrence {occurrence} is absent from page "
                f"{anchor.pageNumber}: {anchor.rawValue!r}"
            )
        raw_offset = available[occurrence - 1]
        resolved.append((anchor.pageNumber, raw_offset, anchor))
    ordered = sorted(
        resolved,
        key=lambda row: (row[0], row[1], row[2].rawValue),
    )
    evidence = tuple(value_at(anchor, offset) for _, offset, anchor in ordered)
    validate_raw_ocr_evidence(pages, evidence)
    return evidence


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
        raise WorkItemError(f"invalid work-item JSON: {context}") from error
    if not isinstance(decoded, dict):
        raise WorkItemError(f"work-item root must be an object: {context}")
    return cast(dict[str, Any], decoded)


def _canonical_directory(path: Path, *, context: str, create: bool = False) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise WorkItemError(f"missing {context}: {path}") from error
    if resolved != absolute or not resolved.is_dir() or resolved.is_symlink():
        raise WorkItemError(f"{context} must be a canonical directory: {path}")
    return resolved


def _canonical_file(path: Path, *, root: Path, context: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise WorkItemError(f"missing {context}: {path}") from error
    if (
        resolved != absolute
        or not resolved.is_relative_to(root)
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise WorkItemError(f"{context} must be a canonical contained regular file: {path}")
    return resolved


def _canonical_uncontained_file(path: Path, *, context: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise WorkItemError(f"missing {context}: {path}") from error
    if resolved != absolute or not resolved.is_file() or resolved.is_symlink():
        raise WorkItemError(f"{context} must be a canonical regular file: {path}")
    return resolved


def _selection_records(source_root: Any) -> tuple[Path, tuple[dict[str, Any], ...]]:
    records_path = _canonical_uncontained_file(
        Path(source_root.selection_records_path), context="selection records"
    )
    payload = read_regular_file_bytes(records_path)
    if sha256_bytes(payload) != source_root.selection_records_sha256:
        raise WorkItemError(f"selection records SHA-256 mismatch: {records_path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise WorkItemError(f"blank line in selection records: {records_path}:{line_number}")
        rows.append(_strict_json(line, context=f"{records_path}:{line_number}"))
    if len(rows) != source_root.documents:
        raise WorkItemError("selection record count differs from configured documents")
    return records_path, tuple(rows)


def _selected_document_ids(source_root: Any) -> tuple[str, ...]:
    records_path, rows = _selection_records(source_root)
    document_ids: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        document_id = row.get("documentId")
        if not isinstance(document_id, str):
            raise WorkItemError(f"selection record lacks documentId: {records_path}:{line_number}")
        document_ids.append(document_id)
    if len(document_ids) != source_root.documents or len(set(document_ids)) != len(document_ids):
        raise WorkItemError("selection record count or documentId uniqueness differs")
    return tuple(document_ids)


def reference_targets(config: AgentLabelingConfig) -> dict[str, dict[str, Any]]:
    """Load hash-pinned accepted targets for post-run comparison only.

    These values are never copied into prepared work items or supplied to a
    model call. They are loaded only when publishing the quality benchmark.
    """

    targets: dict[str, dict[str, Any]] = {}
    for source_root in config.source.work_item_roots:
        records_path, rows = _selection_records(source_root)
        for line_number, row in enumerate(rows, start=1):
            document_id = row.get("documentId")
            target = row.get("target")
            if config.source.reference_target_mode == "absent":
                if not isinstance(document_id, str):
                    raise WorkItemError(
                        f"selection record lacks documentId: {records_path}:{line_number}"
                    )
                if "target" in row:
                    raise WorkItemError(
                        "reference_target_mode=absent forbids target values in selection "
                        f"records: {records_path}:{line_number}"
                    )
                continue
            if not isinstance(document_id, str) or not isinstance(target, dict):
                raise WorkItemError(
                    f"selection record lacks documentId/target: {records_path}:{line_number}"
                )
            if document_id in targets:
                raise WorkItemError(
                    f"duplicate reference documentId across selection records: {document_id}"
                )
            try:
                label = BillOfLadingLabel.model_validate_json(
                    canonical_json_bytes(target), strict=True
                )
            except ValueError as error:
                raise WorkItemError(
                    f"reference target failed semantic-v2 validation: {records_path}:{line_number}"
                ) from error
            canonical = label.canonical_target()
            if canonical != target:
                raise WorkItemError(
                    f"reference target is not canonical: {records_path}:{line_number}"
                )
            targets[document_id] = canonical
    expected_targets = (
        config.source.expected_documents if config.source.reference_target_mode == "required" else 0
    )
    if len(targets) != expected_targets:
        raise WorkItemError("reference target count differs from reference_target_mode")
    return targets


def agent_run_paths(config: AgentLabelingConfig) -> AgentRunPaths:
    output_root = _canonical_directory(
        Path(config.run.output_root), context="labeling output root", create=True
    )
    run_root = _canonical_directory(
        output_root / "runs" / config.run.run_id, context="labeling run root", create=True
    )

    def directory(name: str) -> Path:
        return _canonical_directory(run_root / name, context=name, create=True)

    return AgentRunPaths(
        output_root=output_root,
        run_root=run_root,
        inventory=run_root / "inventory.jsonl",
        selection=run_root / "selection.jsonl",
        resolved_config=run_root / "resolved-config.json",
        prompt_root=directory("prompts"),
        work_item_root=directory("work-items"),
        state_root=directory("state"),
        accepted_root=directory("validated"),
        exclusions_root=directory("exclusions"),
        needs_review_root=directory("needs-review"),
        training_root=directory("training"),
    )


def inventory_work_items(config: AgentLabelingConfig) -> tuple[InventoriedWorkItem, ...]:
    rows: list[InventoriedWorkItem] = []
    document_ids: set[str] = set()
    total_pages = 0
    configured_runs = {row.run_id for row in config.source.extraction_runs}
    for source_root in config.source.work_item_roots:
        root = _canonical_directory(Path(source_root.root), context="work-item root")
        selected_ids = _selected_document_ids(source_root)
        paths = [root / f"{document_id}.json" for document_id in selected_ids]
        for unresolved in paths:
            path = _canonical_file(unresolved, root=root, context="work item")
            payload = read_regular_file_bytes(path)
            _strict_json(payload, context=str(path))
            try:
                item = AgentWorkItem.model_validate_json(payload, strict=True)
            except ValueError as error:
                raise WorkItemError(f"work item failed validation: {path}: {error}") from error
            document_id = item.source.documentId
            if path.stem != document_id:
                raise WorkItemError(f"work-item filename differs from documentId: {path}")
            if document_id in document_ids:
                raise WorkItemError(f"duplicate documentId across work-item roots: {document_id}")
            if item.source.extractionRunId not in configured_runs:
                raise WorkItemError(
                    f"work item references an unconfigured extraction run: {document_id}"
                )
            document_ids.add(document_id)
            total_pages += item.source.documentPageCount
            rows.append(
                InventoriedWorkItem(
                    root_id=source_root.id,
                    path=path,
                    sha256=sha256_bytes(payload),
                    item=item,
                )
            )
    rows.sort(key=lambda row: row.item.source.documentId)
    if len(rows) != config.source.expected_documents:
        raise WorkItemError("inventory document count differs from expected_documents")
    if total_pages != config.source.expected_pages:
        raise WorkItemError("inventory page count differs from expected_pages")
    return tuple(rows)


def inventory_payload(rows: tuple[InventoriedWorkItem, ...]) -> bytes:
    return b"".join(canonical_json_bytes(row.inventory_row()) + b"\n" for row in rows)


def select_work_items(
    config: AgentLabelingConfig, rows: tuple[InventoriedWorkItem, ...]
) -> tuple[InventoriedWorkItem, ...]:
    explicit_ids: tuple[str, ...] | None = None
    if config.selection.document_ids is not None:
        explicit_ids = tuple(config.selection.document_ids)
    elif config.selection.document_ids_file is not None:
        configured = config.selection.document_ids_file
        path = Path(configured.path)
        try:
            payload = read_regular_file_bytes(path)
        except (OSError, ValueError) as error:
            raise WorkItemError(f"selection document ID file is not readable: {path}") from error
        if sha256_bytes(payload) != configured.sha256:
            raise WorkItemError(f"selection document ID file SHA-256 differs: {path}")
        explicit: list[str] = []
        for line_number, raw in enumerate(payload.splitlines(keepends=True), start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise WorkItemError(
                    f"selection document ID row {line_number} is blank or unterminated"
                )
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise WorkItemError(
                    f"selection document ID row {line_number} is invalid JSON"
                ) from error
            if not isinstance(value, dict) or set(value) != {"documentId"}:
                raise WorkItemError(
                    f"selection document ID row {line_number} must contain only documentId"
                )
            document_id = value["documentId"]
            if not isinstance(document_id, str) or _DOCUMENT_ID.fullmatch(document_id) is None:
                raise WorkItemError(
                    f"selection document ID row {line_number} has an invalid documentId"
                )
            explicit.append(document_id)
        if len(explicit) != configured.records:
            raise WorkItemError("selection document ID file count differs from configuration")
        if len(explicit) != len(set(explicit)):
            raise WorkItemError("selection document ID file contains duplicate documentIds")
        explicit_ids = tuple(explicit)
    if explicit_ids is not None:
        by_id = {row.item.source.documentId: row for row in rows}
        missing = [value for value in explicit_ids if value not in by_id]
        if missing:
            raise WorkItemError(
                "explicit selection contains documentIds absent from the frozen inventory: "
                + ", ".join(missing)
            )
        return tuple(by_id[value] for value in explicit_ids)
    if config.selection.count > len(rows):
        raise WorkItemError("selection count exceeds the frozen work-item inventory")
    ranked = sorted(
        rows,
        key=lambda row: stable_id(
            "sample",
            config.selection.namespace,
            str(config.selection.seed),
            row.item.source.documentId,
            row.sha256,
        ),
    )
    return tuple(ranked[: config.selection.count])


def _load_prompt(project_root: Path, configured: PromptArtifactConfig) -> bytes:
    root = project_root.resolve(strict=True)
    relative = PurePosixPath(configured.path)
    path = _canonical_file(root / Path(relative), root=root, context="agent prompt")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != configured.sha256:
        raise WorkItemError(f"prompt SHA-256 mismatch: {path}")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise WorkItemError(f"prompt is not valid UTF-8: {path}") from error
    if not text.strip() or "\x00" in text:
        raise WorkItemError(f"prompt is empty or contains NUL: {path}")
    return payload


def prepare_agent_run(
    config: AgentLabelingConfig, *, project_root: Path
) -> tuple[InventoriedWorkItem, ...]:
    paths = agent_run_paths(config)
    rows = inventory_work_items(config)
    reference_targets(config)
    full_inventory = inventory_payload(rows)
    digest = sha256_bytes(full_inventory)
    if digest != config.source.expected_inventory_sha256:
        raise WorkItemError(
            "work-item inventory SHA-256 differs from expected_inventory_sha256: "
            f"expected {config.source.expected_inventory_sha256}, found {digest}"
        )
    selected = select_work_items(config, rows)
    selection_payload = b"".join(
        canonical_json_bytes(row.inventory_row()) + b"\n" for row in selected
    )
    atomic_publish_bytes(paths.inventory, full_inventory)
    atomic_publish_bytes(paths.selection, selection_payload)
    atomic_publish_json(paths.resolved_config, config.model_dump(mode="json"))

    prompt_rows = (
        ("extractor.md", config.prompts.extractor),
        ("reviewer.md", config.prompts.reviewer),
        ("document-layout.md", config.prompts.document_layout),
    )
    for output_name, prompt_config in prompt_rows:
        atomic_publish_bytes(
            paths.prompt_root / output_name, _load_prompt(project_root, prompt_config)
        )
    for row in selected:
        payload = read_regular_file_bytes(row.path)
        if sha256_bytes(payload) != row.sha256:
            raise WorkItemError(f"work item changed during run preparation: {row.path}")
        atomic_publish_bytes(paths.work_item_root / f"{row.item.source.documentId}.json", payload)
    return selected


def validate_annotation_evidence(
    work_item: AgentWorkItem,
    annotation: BillOfLadingAnnotation | BillOfLadingDualCargoAnnotation,
) -> None:
    if annotation.source != work_item.source:
        raise WorkItemError("annotation source differs from immutable work item")
    pages = page_texts(work_item.joinedRawText)
    evidence_groups = [row.rawOcrEvidence for row in annotation.evidence]
    if isinstance(annotation, BillOfLadingDualCargoAnnotation):
        evidence_groups.extend(row.rawOcrEvidence for row in annotation.relationEvidence)
    for raw_evidence in evidence_groups:
        validate_raw_ocr_evidence(pages, raw_evidence)


def validate_exclusion_evidence(work_item: AgentWorkItem, exclusion: BillOfLadingExclusion) -> None:
    if exclusion.source != work_item.source:
        raise WorkItemError("exclusion source differs from immutable work item")
    pages = page_texts(work_item.joinedRawText)
    for evidence in exclusion.rawOcrEvidence:
        page = pages.get(evidence.pageNumber)
        if (
            page is None
            or evidence.ocrExcerpt not in page
            or evidence.rawValue not in evidence.ocrExcerpt
        ):
            raise WorkItemError("exclusion evidence is not verbatim in raw OCR")


def pdf_bytes_for_pages(
    config: AgentLabelingConfig,
    work_item: AgentWorkItem,
    page_numbers: tuple[int, ...],
) -> PdfAssistancePayload:
    """Build a requested-page PDF from the hash-verified original document.

    The PDF remains auxiliary layout evidence. The caller must continue to
    require every emitted target value and layout anchor to occur in raw OCR.
    """

    if config.workflow.pdf_page_scope != "requested_pages":
        raise WorkItemError("unsupported PDF page scope")
    if tuple(sorted(set(page_numbers))) != page_numbers or not page_numbers:
        raise WorkItemError("PDF assistance pages must be unique and sorted")
    expected_pages = set(range(1, work_item.source.documentPageCount + 1))
    if not set(page_numbers).issubset(expected_pages):
        raise WorkItemError("PDF assistance request cites a page outside the work item")

    source_path = _canonical_uncontained_file(
        Path(work_item.source.localCanonicalPath), context="source PDF"
    )
    if source_path.suffix.lower() != ".pdf":
        raise WorkItemError("source document for PDF assistance must end in .pdf")
    source_payload = read_regular_file_bytes(source_path)
    if sha256_bytes(source_payload) != work_item.source.sourceSha256:
        raise WorkItemError("source PDF SHA-256 differs from the immutable work item")
    if not source_payload.startswith(b"%PDF-"):
        raise WorkItemError("source document lacks a PDF header")

    construction_method: PdfConstructionMethod = "pypdf_strict"
    try:
        try:
            reader = PdfReader(BytesIO(source_payload), strict=True)
        except PdfReadError:
            # Some source PDFs render successfully but contain malformed xref or
            # duplicate dictionary entries. Recovery is explicit in the immutable
            # call receipt; all identity and page-count checks still apply.
            reader = PdfReader(BytesIO(source_payload), strict=False)
            construction_method = "pypdf_recovery"
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise WorkItemError(
                    "source PDF requires a non-empty password for document assistance"
                )
            construction_method = (
                "pypdf_strict_empty_password"
                if construction_method == "pypdf_strict"
                else "pypdf_recovery_empty_password"
            )
        if len(reader.pages) != work_item.source.documentPageCount:
            raise WorkItemError("source PDF page count differs from the immutable work item")
        writer = PdfWriter()
        for page_number in page_numbers:
            writer.add_page(reader.pages[page_number - 1])
        output = BytesIO()
        writer.write(output)
        attachment = output.getvalue()
    except WorkItemError:
        raise
    except Exception as error:
        raise WorkItemError("failed to construct requested-page PDF assistance payload") from error

    if not attachment.startswith(b"%PDF-"):
        raise WorkItemError("derived document assistance payload lacks a PDF header")
    if len(attachment) > config.workflow.max_pdf_bytes:
        raise WorkItemError(
            "derived document assistance PDF exceeds the configured/API byte ceiling"
        )
    return PdfAssistancePayload(
        data=attachment,
        source_pdf_sha256=work_item.source.sourceSha256,
        source_page_numbers=page_numbers,
        attachment_pdf_sha256=sha256_bytes(attachment),
        construction_method=construction_method,
    )
