"""Explicit extraction -> validation -> independent review state machine."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pydantic import BaseModel

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading import (
    BillOfLadingExclusion,
)
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingDualCargoAnnotation
from document_ocr.labeling_agents.config import (
    AgentLabelingConfig,
    OllamaProviderConfig,
    OpenAIResponsesProviderConfig,
)
from document_ocr.labeling_agents.corrections import (
    merge_correction_values,
    normalize_correction_paths,
    unchanged_correction_paths,
)
from document_ocr.labeling_agents.deterministic_annotation import build_compact_annotation
from document_ocr.labeling_agents.models import (
    AgentCallReceipt,
    AnnotationDraft,
    CompactAnnotationDraft,
    CompactCorrectionEnvelope,
    CompactReviewDraft,
    DocumentAssistanceRequest,
    DocumentLayoutGuidance,
    DocumentRunOutcome,
    ExclusionDraft,
    IndependentReviewArtifact,
    NeedsReviewRecord,
    ReviewDocumentAssistanceRequest,
    ReviewDraft,
)
from document_ocr.labeling_agents.provider import (
    AgentCallFailure,
    AgentCallResult,
    AgentProviderError,
)
from document_ocr.labeling_agents.work_items import (
    AgentRunPaths,
    AgentWorkItem,
    InventoriedWorkItem,
    PdfAssistancePayload,
    WorkItemError,
    agent_run_paths,
    page_texts,
    pdf_bytes_for_pages,
    prepare_agent_run,
    reference_targets,
    validate_annotation_evidence,
    validate_exclusion_evidence,
    validate_raw_ocr_evidence,
)
from document_ocr.training.metrics import assess_prediction, structured_metrics
from document_ocr.training.tasks import canonical_json, get_training_task


class LabelingRunError(RuntimeError):
    """The agent labeling run violated its immutable lifecycle contract."""


class AgentGateway(Protocol):
    async def extract(
        self,
        work_item: AgentWorkItem,
        *,
        work_item_sha256: str,
        candidate_attempt: int,
        retry_feedback: str | None,
        layout_guidance: DocumentLayoutGuidance | None,
        correction_base: CompactAnnotationDraft | None = None,
        correction_paths: tuple[str, ...] = (),
    ) -> AgentCallResult[BaseModel]: ...

    async def review(
        self,
        work_item: AgentWorkItem,
        candidate: BaseModel,
        *,
        work_item_sha256: str,
        candidate_attempt: int,
        layout_guidance: DocumentLayoutGuidance | None,
    ) -> AgentCallResult[BaseModel]: ...

    async def document_layout(
        self,
        work_item: AgentWorkItem,
        *,
        work_item_sha256: str,
        candidate_attempt: int,
        request: BaseModel,
        pdf: PdfAssistancePayload,
    ) -> AgentCallResult[DocumentLayoutGuidance]: ...


@dataclass(frozen=True, slots=True)
class LabelingProgress:
    total_documents: int
    processed_documents: int
    validated_documents: int
    excluded_documents: int
    needs_review_documents: int


ProgressCallback = Callable[[LabelingProgress], None]


def _relative_to_run(path: Path, run_root: Path) -> str:
    try:
        return path.relative_to(run_root).as_posix()
    except ValueError as error:
        raise LabelingRunError("artifact path escapes the labeling run") from error


def _publish_model(path: Path, model: BaseModel) -> tuple[str, str]:
    payload = canonical_json_bytes(model.model_dump(mode="json")) + b"\n"
    digest = sha256_bytes(payload)
    target = path / f"{digest}.json"
    atomic_publish_bytes(target, payload)
    return str(target), digest


def _publish_receipt(
    paths: AgentRunPaths,
    receipt: AgentCallReceipt,
) -> tuple[str, str]:
    payload = canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"
    digest = sha256_bytes(payload)
    path = (
        paths.state_root
        / "call-receipts"
        / receipt.documentId
        / f"{receipt.stage}-{receipt.callId}-{digest[:16]}.json"
    )
    atomic_publish_bytes(path, payload)
    return _relative_to_run(path, paths.run_root), digest


def _validate_raw_evidence(work_item: AgentWorkItem, records: Sequence[Any]) -> None:
    pages = page_texts(work_item.joinedRawText)
    validate_raw_ocr_evidence(pages, records)


def _validate_document_request(
    work_item: AgentWorkItem,
    request: DocumentAssistanceRequest | ReviewDocumentAssistanceRequest,
) -> tuple[int, ...]:
    page_numbers = request.pageNumbers
    if not set(page_numbers).issubset({row.pageNumber for row in work_item.source.pages}):
        raise WorkItemError("document request cites a page outside the work item")
    _validate_raw_evidence(work_item, request.rawOcrEvidence)
    return page_numbers


def _validate_layout_guidance(work_item: AgentWorkItem, guidance: DocumentLayoutGuidance) -> None:
    for observation in guidance.observations:
        if observation.pageNumber not in {row.pageNumber for row in work_item.source.pages}:
            raise WorkItemError("layout guidance cites a page outside the work item")
        _validate_raw_evidence(work_item, observation.rawOcrAnchors)


def _validate_review_draft(work_item: AgentWorkItem, review: ReviewDraft) -> None:
    for finding in review.findings:
        _validate_raw_evidence(work_item, finding.rawOcrEvidence)


def _provider_metadata(config: AgentLabelingConfig, provider_id: str) -> tuple[str, str | None]:
    provider = next(row for row in config.providers if row.id == provider_id)
    if isinstance(provider, OpenAIResponsesProviderConfig):
        return provider.model, provider.reasoning_effort
    if isinstance(provider, OllamaProviderConfig):
        return provider.model, provider.reasoning_mode
    raise LabelingRunError(f"unsupported provider configuration: {provider_id}")


def _validate_receipt_contract(
    config: AgentLabelingConfig,
    row: InventoriedWorkItem,
    receipt: AgentCallReceipt,
    *,
    maximum_candidate_attempt: int,
) -> None:
    assigned_provider = {
        "extract": config.assignments.labeler,
        "review": config.assignments.reviewer,
        "document_layout": config.assignments.document_layout,
    }[receipt.stage]
    provider = next(value for value in config.providers if value.id == assigned_provider)
    expected_model, expected_reasoning = _provider_metadata(config, assigned_provider)
    if (
        receipt.documentId != row.item.source.documentId
        or receipt.workItemSha256 != row.sha256
        or receipt.candidateAttempt > maximum_candidate_attempt
        or receipt.providerId != assigned_provider
        or receipt.providerKind != provider.kind
        or receipt.model != expected_model
        or receipt.reasoningEffort != expected_reasoning
    ):
        raise LabelingRunError(
            f"call receipt conflicts with the frozen run contract: {receipt.callId}"
        )
    if receipt.stage == "document_layout":
        attachment = receipt.pdfAttachment
        if attachment is None or attachment.sourcePdfSha256 != row.item.source.sourceSha256:
            raise LabelingRunError("PDF attachment receipt conflicts with the source document")
        if not set(attachment.sourcePageNumbers).issubset(
            {page.pageNumber for page in row.item.source.pages}
        ):
            raise LabelingRunError("PDF attachment receipt cites an absent source page")


def _stored_call_receipts(
    config: AgentLabelingConfig,
    paths: AgentRunPaths,
    row: InventoriedWorkItem,
    outcome: DocumentRunOutcome,
) -> tuple[tuple[str, AgentCallReceipt, bool], ...]:
    """Load every persisted provider call and identify outcome-referenced calls.

    A process can stop after atomically publishing a receipt but before writing
    the document outcome.  Those orphaned receipts still represent provider
    usage and must remain visible in run accounting after a resume.
    """

    referenced = dict(
        zip(outcome.callReceiptPaths, outcome.callReceiptSha256s, strict=True)
    )
    receipt_root = paths.state_root / "call-receipts" / outcome.documentId
    if not receipt_root.is_dir() or receipt_root.is_symlink():
        raise LabelingRunError(
            f"call receipt directory is missing or unsafe: {outcome.documentId}"
        )
    loaded: list[tuple[str, AgentCallReceipt, bool]] = []
    referenced_seen: set[str] = set()
    call_ids: set[str] = set()
    for receipt_path in sorted(receipt_root.iterdir()):
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise LabelingRunError(
                f"call receipt directory contains an unsafe entry: {receipt_path}"
            )
        relative_value = _relative_to_run(receipt_path, paths.run_root)
        digest = sha256_file(receipt_path)
        expected_digest = referenced.get(relative_value)
        is_referenced = expected_digest is not None
        if is_referenced:
            if digest != expected_digest:
                raise LabelingRunError("call receipt changed before publication")
            referenced_seen.add(relative_value)
        try:
            receipt = AgentCallReceipt.model_validate_json(
                read_regular_file_bytes(receipt_path), strict=True
            )
        except ValueError as error:
            raise LabelingRunError(f"invalid persisted call receipt: {receipt_path}") from error
        if receipt.callId in call_ids:
            raise LabelingRunError(
                f"duplicate persisted call receipt identity: {receipt.callId}"
            )
        call_ids.add(receipt.callId)
        _validate_receipt_contract(
            config,
            row,
            receipt,
            maximum_candidate_attempt=(
                outcome.attempts if is_referenced else config.workflow.max_candidate_attempts
            ),
        )
        loaded.append((relative_value, receipt, is_referenced))
    missing = set(referenced) - referenced_seen
    if missing:
        raise LabelingRunError(
            "document outcome references call receipts absent from its receipt directory"
        )
    return tuple(loaded)


def _stored_uncommitted_call_receipts(
    config: AgentLabelingConfig,
    paths: AgentRunPaths,
    row: InventoriedWorkItem,
) -> tuple[tuple[str, AgentCallReceipt, bool], ...]:
    """Validate calls persisted before an interrupted document committed an outcome."""

    receipt_root = paths.state_root / "call-receipts" / row.item.source.documentId
    if not receipt_root.is_dir() or receipt_root.is_symlink():
        raise LabelingRunError(
            f"uncommitted call receipt directory is missing or unsafe: "
            f"{row.item.source.documentId}"
        )
    loaded: list[tuple[str, AgentCallReceipt, bool]] = []
    call_ids: set[str] = set()
    for receipt_path in sorted(receipt_root.iterdir()):
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise LabelingRunError(
                f"call receipt directory contains an unsafe entry: {receipt_path}"
            )
        try:
            receipt = AgentCallReceipt.model_validate_json(
                read_regular_file_bytes(receipt_path), strict=True
            )
        except ValueError as error:
            raise LabelingRunError(f"invalid persisted call receipt: {receipt_path}") from error
        if receipt.callId in call_ids:
            raise LabelingRunError(
                f"duplicate persisted call receipt identity: {receipt.callId}"
            )
        call_ids.add(receipt.callId)
        _validate_receipt_contract(
            config,
            row,
            receipt,
            maximum_candidate_attempt=config.workflow.max_candidate_attempts,
        )
        loaded.append((_relative_to_run(receipt_path, paths.run_root), receipt, False))
    if not loaded:
        raise LabelingRunError(
            f"uncommitted receipt directory is empty: {row.item.source.documentId}"
        )
    return tuple(loaded)


@dataclass(frozen=True, slots=True)
class _ReceiptAccounting:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost: Decimal
    referenced_cost: Decimal
    orphaned_cost: Decimal
    priced_calls: int
    referenced_calls: int
    orphaned_calls: int
    local_calls: int
    unpriced_openai_calls: int


def _account_receipts(
    stored_receipts: Sequence[tuple[str, AgentCallReceipt, bool]],
) -> _ReceiptAccounting:
    cost = referenced_cost = orphaned_cost = Decimal(0)
    input_tokens = output_tokens = cache_read_tokens = cache_write_tokens = 0
    priced_calls = referenced_calls = orphaned_calls = 0
    local_calls = unpriced_openai_calls = 0
    for _, receipt, is_referenced in stored_receipts:
        input_tokens += receipt.inputTokens
        output_tokens += receipt.outputTokens
        cache_read_tokens += receipt.cacheReadTokens
        cache_write_tokens += receipt.cacheWriteTokens
        referenced_calls += int(is_referenced)
        orphaned_calls += int(not is_referenced)
        if receipt.costUsd is not None:
            cost += receipt.costUsd
            priced_calls += 1
            if is_referenced:
                referenced_cost += receipt.costUsd
            else:
                orphaned_cost += receipt.costUsd
        elif receipt.providerKind == "openai_responses":
            unpriced_openai_calls += 1
        else:
            local_calls += 1
    return _ReceiptAccounting(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost=cost,
        referenced_cost=referenced_cost,
        orphaned_cost=orphaned_cost,
        priced_calls=priced_calls,
        referenced_calls=referenced_calls,
        orphaned_calls=orphaned_calls,
        local_calls=local_calls,
        unpriced_openai_calls=unpriced_openai_calls,
    )


def _usage_row(
    *,
    document_id: str,
    status: str,
    attempts: int,
    reviews: int,
    document_escalations: int,
    receipts: Sequence[tuple[str, AgentCallReceipt, bool]],
) -> tuple[dict[str, Any], _ReceiptAccounting]:
    accounting = _account_receipts(receipts)
    return (
        {
            "documentId": document_id,
            "status": status,
            "attempts": attempts,
            "reviews": reviews,
            "documentEscalations": document_escalations,
            "calls": len(receipts),
            "outcomeReferencedCalls": accounting.referenced_calls,
            "orphanedCalls": accounting.orphaned_calls,
            "inputTokens": accounting.input_tokens,
            "cacheReadTokens": accounting.cache_read_tokens,
            "cacheWriteTokens": accounting.cache_write_tokens,
            "outputTokens": accounting.output_tokens,
            "pricedCalls": accounting.priced_calls,
            "unpricedOpenAiCalls": accounting.unpriced_openai_calls,
            "localCalls": accounting.local_calls,
            "configuredListPriceEstimateComplete": accounting.unpriced_openai_calls == 0,
            "configuredListPriceEstimateUsd": (
                str(accounting.cost) if accounting.priced_calls else None
            ),
            "outcomeReferencedConfiguredListPriceEstimateUsd": (
                str(accounting.referenced_cost) if accounting.priced_calls else None
            ),
            "orphanedConfiguredListPriceEstimateUsd": (
                str(accounting.orphaned_cost) if accounting.priced_calls else None
            ),
            "costBasis": "configured_public_list_rates_not_billed_cost",
        },
        accounting,
    )


def _candidate_annotation(
    work_item: AgentWorkItem, draft: AnnotationDraft
) -> BillOfLadingDualCargoAnnotation:
    annotation = BillOfLadingDualCargoAnnotation.model_validate(
        {
            "annotationSchemaVersion": "3.0.0-experimental",
            "taskType": "bill_of_lading_dual_cargo_kie",
            "documentType": draft.documentType,
            "source": work_item.source,
            "normalLabel": draft.normalLabel,
            "relationExplicitLabel": draft.relationExplicitLabel,
            "evidence": draft.evidence,
            "relationEvidence": draft.relationEvidence,
            "warnings": draft.warnings,
            "reviewStatus": "candidate",
            "reviewNotes": draft.decisionNotes,
        },
        strict=True,
    )
    validate_annotation_evidence(work_item, annotation)
    return annotation


def _candidate_compact_annotation(
    work_item: AgentWorkItem,
    draft: CompactAnnotationDraft,
    *,
    pdf_grouping_used: bool,
) -> BillOfLadingDualCargoAnnotation:
    return build_compact_annotation(
        work_item,
        draft,
        pdf_grouping_used=pdf_grouping_used,
    )


_REVIEW_CHECK_BY_CATEGORY = {
    "truth_boundary": "rawOcrTruthBoundary",
    "evidence": "evidenceIntegrity",
    "missing_field": "semanticCompleteness",
    "incorrect_field": "semanticCorrectness",
    "contamination": "contaminationAndRedundancy",
    "redundancy": "contaminationAndRedundancy",
    "document_unit": "documentUnitAndRelationships",
    "relationship": "documentUnitAndRelationships",
    "ambiguity": "documentUnitAndRelationships",
    "other": "semanticCorrectness",
}

_CORRECT_OMISSION_WARNING = re.compile(
    r"(?ix)\b(?:correctly|appropriately)\s+omitted\b.*\bwarning\b|"
    r"\bwarning\b.*\b(?:correctly|appropriately)\s+omitted\b"
)


def _resolve_compact_review(draft: CompactReviewDraft) -> ReviewDraft:
    """Derive the verdict/check matrix instead of asking the model to duplicate it."""

    findings = tuple(
        row.model_copy(update={"severity": "blocking"})
        if (
            row.category != "other"
            and row.severity != "blocking"
            and not _CORRECT_OMISSION_WARNING.search(row.message)
        )
        else row
        for row in draft.findings
    )
    checks = {
        "rawOcrTruthBoundary": "pass",
        "evidenceIntegrity": "pass",
        "semanticCompleteness": "pass",
        "semanticCorrectness": "pass",
        "contaminationAndRedundancy": "pass",
        "documentUnitAndRelationships": "pass",
    }
    blocking = tuple(row for row in findings if row.severity == "blocking")
    for finding in blocking:
        checks[_REVIEW_CHECK_BY_CATEGORY[finding.category]] = "fail"
    return ReviewDraft.model_validate(
        {
            "decision": "review",
            "result": "fail" if blocking else "pass",
            "checks": checks,
            "findings": findings,
            "summary": draft.summary,
        },
        strict=True,
    )


def _review_retry_feedback(review: IndependentReviewArtifact) -> str:
    """Return only actionable blocking defects to the next fresh extractor."""

    findings = [
        {
            "category": finding.category,
            "message": finding.message,
            "targetPaths": list(finding.targetPaths),
            "rawOcrEvidence": [
                {
                    "pageNumber": evidence.pageNumber,
                    "rawValue": evidence.rawValue,
                }
                for evidence in finding.rawOcrEvidence
            ],
        }
        for finding in review.findings
        if finding.severity == "blocking"
    ]
    if not findings:
        raise LabelingRunError("failed independent review contains no blocking finding")
    correction_paths = _review_correction_paths(review)
    instruction = (
        "Correct every grounded defect below and return the exact-path correction envelope. "
        "Emit every allowedCorrectionPaths key once and no other key; the pipeline preserves "
        "all other candidate facts. Do not add OCR-external facts."
        if correction_paths
        else (
            "Correct every grounded defect below, re-audit the complete raw OCR, and return one "
            "complete replacement decision without adding OCR-external facts."
        )
    )
    return json.dumps(
        {
            "instruction": instruction,
            "allowedCorrectionPaths": list(correction_paths),
            "findings": findings,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _review_correction_paths(review: IndependentReviewArtifact) -> tuple[str, ...]:
    blocking = tuple(row for row in review.findings if row.severity == "blocking")
    if not blocking or any(not row.targetPaths for row in blocking):
        return ()
    return normalize_correction_paths(tuple(path for row in blocking for path in row.targetPaths))


def _candidate_exclusion(work_item: AgentWorkItem, draft: ExclusionDraft) -> BillOfLadingExclusion:
    exclusion = BillOfLadingExclusion.model_validate(
        {
            "exclusionSchemaVersion": "2.0.0",
            "taskType": "bill_of_lading_kie",
            "source": work_item.source,
            "reason": draft.reason,
            "rawOcrEvidence": draft.rawOcrEvidence,
            "reviewStatus": "rejected",
            "reviewNotes": draft.reviewNotes,
        },
        strict=True,
    )
    validate_exclusion_evidence(work_item, exclusion)
    return exclusion


def _review_identity_sha256(candidate: BaseModel) -> str:
    """Hash only the semantic payload visible to the independent reviewer."""

    value = candidate.model_dump(mode="json")
    if isinstance(candidate, BillOfLadingDualCargoAnnotation):
        value = {
            "documentType": value["documentType"],
            "relationExplicitLabel": value["relationExplicitLabel"],
            "warnings": value["warnings"],
        }
    return sha256_bytes(canonical_json_bytes(value))


def _outcome_path(paths: AgentRunPaths, document_id: str) -> Path:
    return paths.state_root / "outcomes" / f"{document_id}.json"


def _load_existing_outcome(paths: AgentRunPaths, document_id: str) -> DocumentRunOutcome | None:
    path = _outcome_path(paths, document_id)
    if not path.exists():
        return None
    try:
        outcome = DocumentRunOutcome.model_validate_json(read_regular_file_bytes(path), strict=True)
    except ValueError as error:
        raise LabelingRunError(f"invalid document outcome: {path}") from error
    if outcome.documentId != document_id:
        raise LabelingRunError("document outcome identity differs from its path")
    references = (
        (outcome.finalArtifactPath, outcome.finalArtifactSha256),
        *zip(outcome.callReceiptPaths, outcome.callReceiptSha256s, strict=True),
    )
    for relative_value, digest in references:
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise LabelingRunError("document outcome contains an unsafe artifact path")
        artifact = paths.run_root / Path(relative)
        if not artifact.is_file() or artifact.is_symlink() or sha256_file(artifact) != digest:
            raise LabelingRunError("document outcome references a missing or changed artifact")
    return outcome


def _publish_outcome(paths: AgentRunPaths, outcome: DocumentRunOutcome) -> None:
    atomic_publish_bytes(
        _outcome_path(paths, outcome.documentId),
        canonical_json_bytes(outcome.model_dump(mode="json")) + b"\n",
    )


async def _document_guidance(
    config: AgentLabelingConfig,
    paths: AgentRunPaths,
    gateway: AgentGateway,
    row: InventoriedWorkItem,
    *,
    request: DocumentAssistanceRequest | ReviewDocumentAssistanceRequest,
    candidate_attempt: int,
    document_escalation: int,
    receipt_paths: list[str],
    receipt_hashes: list[str],
) -> DocumentLayoutGuidance:
    page_numbers = _validate_document_request(row.item, request)
    pdf = pdf_bytes_for_pages(config, row.item, page_numbers)
    result = await gateway.document_layout(
        row.item,
        work_item_sha256=row.sha256,
        candidate_attempt=candidate_attempt,
        request=request,
        pdf=pdf,
    )
    receipt_path, receipt_hash = _publish_receipt(paths, result.receipt)
    receipt_paths.append(receipt_path)
    receipt_hashes.append(receipt_hash)
    _validate_layout_guidance(row.item, result.output)
    _publish_model(
        paths.state_root
        / "layout-guidance"
        / row.item.source.documentId
        / f"attempt-{candidate_attempt}-escalation-{document_escalation}",
        result.output,
    )
    return result.output


async def _needs_review(
    paths: AgentRunPaths,
    row: InventoriedWorkItem,
    *,
    attempts: int,
    reviews: int,
    document_escalations: int,
    reason: str,
    findings: Sequence[str],
    receipt_paths: list[str],
    receipt_hashes: list[str],
) -> DocumentRunOutcome:
    record = NeedsReviewRecord.model_validate(
        {
            "schemaVersion": 1,
            "documentId": row.item.source.documentId,
            "workItemSha256": row.sha256,
            "attempts": max(1, attempts),
            "reviews": reviews,
            "documentEscalations": document_escalations,
            "reason": reason,
            "findings": tuple(findings) or ("No validated candidate was produced.",),
        },
        strict=True,
    )
    final_path_value, final_hash = _publish_model(paths.needs_review_root, record)
    final_path = Path(final_path_value)
    outcome = DocumentRunOutcome.model_validate(
        {
            "schemaVersion": 1,
            "documentId": row.item.source.documentId,
            "status": "needs_review",
            "attempts": max(1, attempts),
            "reviews": reviews,
            "documentEscalations": document_escalations,
            "finalArtifactPath": _relative_to_run(final_path, paths.run_root),
            "finalArtifactSha256": final_hash,
            "callReceiptPaths": tuple(receipt_paths),
            "callReceiptSha256s": tuple(receipt_hashes),
        },
        strict=True,
    )
    _publish_outcome(paths, outcome)
    return outcome


async def process_document(
    config: AgentLabelingConfig,
    paths: AgentRunPaths,
    gateway: AgentGateway,
    row: InventoriedWorkItem,
) -> DocumentRunOutcome:
    existing = _load_existing_outcome(paths, row.item.source.documentId)
    if existing is not None:
        return existing

    receipt_paths: list[str] = []
    receipt_hashes: list[str] = []
    reviews = 0
    document_escalations = 0
    retry_feedback: str | None = None
    correction_base: CompactAnnotationDraft | None = None
    correction_paths: tuple[str, ...] = ()
    retained_layout_guidance: DocumentLayoutGuidance | None = None
    validation_findings: list[str] = []
    rejected_candidate_contexts: set[tuple[str, str | None]] = set()

    for attempt in range(1, config.workflow.max_candidate_attempts + 1):
        layout_guidance = retained_layout_guidance
        try:
            extracted = await gateway.extract(
                row.item,
                work_item_sha256=row.sha256,
                candidate_attempt=attempt,
                retry_feedback=retry_feedback,
                layout_guidance=layout_guidance,
                correction_base=correction_base,
                correction_paths=correction_paths,
            )
            receipt_path, receipt_hash = _publish_receipt(paths, extracted.receipt)
            receipt_paths.append(receipt_path)
            receipt_hashes.append(receipt_hash)
            decision = extracted.output
            _publish_model(
                paths.state_root
                / "model-decisions"
                / row.item.source.documentId
                / f"attempt-{attempt}",
                decision,
            )
            while isinstance(decision, DocumentAssistanceRequest):
                if document_escalations >= config.workflow.max_document_escalations_per_document:
                    return await _needs_review(
                        paths,
                        row,
                        attempts=attempt,
                        reviews=reviews,
                        document_escalations=document_escalations,
                        reason="document_escalation_unavailable",
                        findings=(decision.ambiguity,),
                        receipt_paths=receipt_paths,
                        receipt_hashes=receipt_hashes,
                    )
                layout_guidance = await _document_guidance(
                    config,
                    paths,
                    gateway,
                    row,
                    request=decision,
                    candidate_attempt=attempt,
                    document_escalation=document_escalations + 1,
                    receipt_paths=receipt_paths,
                    receipt_hashes=receipt_hashes,
                )
                retained_layout_guidance = layout_guidance
                document_escalations += 1
                extracted = await gateway.extract(
                    row.item,
                    work_item_sha256=row.sha256,
                    candidate_attempt=attempt,
                    retry_feedback=(
                        "Use the supplied layout guidance only for grouping. All target values "
                        "still require verbatim raw-OCR evidence."
                    ),
                    layout_guidance=layout_guidance,
                    correction_base=correction_base,
                    correction_paths=correction_paths,
                )
                receipt_path, receipt_hash = _publish_receipt(paths, extracted.receipt)
                receipt_paths.append(receipt_path)
                receipt_hashes.append(receipt_hash)
                decision = extracted.output
                _publish_model(
                    paths.state_root
                    / "model-decisions"
                    / row.item.source.documentId
                    / f"attempt-{attempt}-after-document-{document_escalations}",
                    decision,
                )

            if correction_base is not None:
                if not isinstance(decision, CompactCorrectionEnvelope):
                    raise LabelingRunError(
                        "exact-path correction returned a full extraction decision"
                    )
                merged_decision = merge_correction_values(
                    correction_base,
                    decision.corrections,
                    correction_paths,
                )
                unchanged_paths = unchanged_correction_paths(
                    correction_base,
                    merged_decision,
                    correction_paths,
                )
                _publish_model(
                    paths.state_root
                    / "target-scoped-corrections"
                    / row.item.source.documentId
                    / f"attempt-{attempt}",
                    merged_decision,
                )
                if unchanged_paths:
                    changed_paths = tuple(
                        path for path in correction_paths if path not in unchanged_paths
                    )
                    correction_base = merged_decision
                    correction_paths = unchanged_paths
                    applied = (
                        f" Applied and retained: {', '.join(changed_paths)}."
                        if changed_paths
                        else ""
                    )
                    retry_feedback = (
                        "The previous exact-path correction left these reviewer-authorized "
                        f"targets unchanged: {', '.join(unchanged_paths)}.{applied} Return "
                        "replacement values only for the remaining unchanged targets."
                    )
                    validation_findings.append(retry_feedback)
                    if attempt == config.workflow.max_candidate_attempts:
                        return await _needs_review(
                            paths,
                            row,
                            attempts=attempt,
                            reviews=reviews,
                            document_escalations=document_escalations,
                            reason="correction_non_convergence",
                            findings=(retry_feedback,),
                            receipt_paths=receipt_paths,
                            receipt_hashes=receipt_hashes,
                        )
                    continue
                decision = merged_decision
                correction_base = None
                correction_paths = ()
            elif isinstance(decision, CompactCorrectionEnvelope):
                raise LabelingRunError("corrector returned without an active correction")

            candidate: BaseModel
            if isinstance(decision, CompactAnnotationDraft):
                candidate = _candidate_compact_annotation(
                    row.item,
                    decision,
                    pdf_grouping_used=layout_guidance is not None,
                )
            elif isinstance(decision, AnnotationDraft):
                candidate = _candidate_annotation(row.item, decision)
            elif isinstance(decision, ExclusionDraft):
                candidate = _candidate_exclusion(row.item, decision)
            else:
                raise LabelingRunError("extractor returned an unsupported decision")
            _, candidate_hash = _publish_model(
                paths.state_root
                / "candidate-attempts"
                / row.item.source.documentId
                / f"attempt-{attempt}",
                candidate,
            )
            layout_context_hash = (
                sha256_bytes(canonical_json_bytes(layout_guidance.model_dump(mode="json")))
                if layout_guidance is not None
                else None
            )
            candidate_context = (_review_identity_sha256(candidate), layout_context_hash)
            if candidate_context in rejected_candidate_contexts:
                return await _needs_review(
                    paths,
                    row,
                    attempts=attempt,
                    reviews=reviews,
                    document_escalations=document_escalations,
                    reason="correction_non_convergence",
                    findings=(
                        "A correction reproduced an identical candidate previously rejected "
                        "under the same document-layout context.",
                    ),
                    receipt_paths=receipt_paths,
                    receipt_hashes=receipt_hashes,
                )
            if isinstance(candidate, BillOfLadingDualCargoAnnotation):
                ambiguity_warnings = tuple(
                    warning
                    for warning in candidate.warnings
                    if warning.code == "ambiguous_ocr_candidates"
                )
                if ambiguity_warnings:
                    return await _needs_review(
                        paths,
                        row,
                        attempts=attempt,
                        reviews=reviews,
                        document_escalations=document_escalations,
                        reason="training_truth_ambiguity",
                        findings=tuple(warning.message for warning in ambiguity_warnings),
                        receipt_paths=receipt_paths,
                        receipt_hashes=receipt_hashes,
                    )
            review_result = await gateway.review(
                row.item,
                candidate,
                work_item_sha256=row.sha256,
                candidate_attempt=attempt,
                layout_guidance=layout_guidance,
            )
            reviews += 1
            receipt_path, receipt_hash = _publish_receipt(paths, review_result.receipt)
            receipt_paths.append(receipt_path)
            receipt_hashes.append(receipt_hash)
            review_decision = review_result.output
            while isinstance(review_decision, ReviewDocumentAssistanceRequest):
                if document_escalations >= config.workflow.max_document_escalations_per_document:
                    return await _needs_review(
                        paths,
                        row,
                        attempts=attempt,
                        reviews=reviews,
                        document_escalations=document_escalations,
                        reason="document_escalation_unavailable",
                        findings=(review_decision.ambiguity,),
                        receipt_paths=receipt_paths,
                        receipt_hashes=receipt_hashes,
                    )
                layout_guidance = await _document_guidance(
                    config,
                    paths,
                    gateway,
                    row,
                    request=review_decision,
                    candidate_attempt=attempt,
                    document_escalation=document_escalations + 1,
                    receipt_paths=receipt_paths,
                    receipt_hashes=receipt_hashes,
                )
                retained_layout_guidance = layout_guidance
                document_escalations += 1
                review_result = await gateway.review(
                    row.item,
                    candidate,
                    work_item_sha256=row.sha256,
                    candidate_attempt=attempt,
                    layout_guidance=layout_guidance,
                )
                reviews += 1
                receipt_path, receipt_hash = _publish_receipt(paths, review_result.receipt)
                receipt_paths.append(receipt_path)
                receipt_hashes.append(receipt_hash)
                review_decision = review_result.output
            if isinstance(review_decision, CompactReviewDraft):
                review_decision = _resolve_compact_review(review_decision)
            if not isinstance(review_decision, ReviewDraft):
                raise LabelingRunError("reviewer returned an unresolved document request")
            _validate_review_draft(row.item, review_decision)
            reviewer_model, reviewer_reasoning = _provider_metadata(
                config, config.assignments.reviewer
            )
            review = IndependentReviewArtifact.model_validate(
                {
                    "schemaVersion": 1,
                    "documentId": row.item.source.documentId,
                    "candidateAttempt": attempt,
                    "reviewNumber": reviews,
                    "workItemSha256": row.sha256,
                    "candidateSha256": candidate_hash,
                    "providerId": config.assignments.reviewer,
                    "reviewerModel": reviewer_model,
                    "reviewerReasoningEffort": reviewer_reasoning,
                    "result": review_decision.result,
                    "checks": review_decision.checks,
                    "findings": review_decision.findings,
                    "summary": review_decision.summary,
                },
                strict=True,
            )
            _publish_model(
                paths.state_root
                / "independent-reviews"
                / row.item.source.documentId
                / f"attempt-{attempt}",
                review,
            )
            if review.result == "fail":
                rejected_candidate_contexts.add(candidate_context)
                retry_feedback = _review_retry_feedback(review)
                if isinstance(decision, CompactAnnotationDraft):
                    correction_paths = _review_correction_paths(review)
                    correction_base = decision if correction_paths else None
                else:
                    correction_paths = ()
                    correction_base = None
                validation_findings.extend(finding.message for finding in review.findings)
                continue

            if isinstance(candidate, BillOfLadingDualCargoAnnotation):
                final = candidate.model_copy(
                    update={
                        "reviewStatus": "validated",
                        "reviewNotes": (*candidate.reviewNotes, review.summary),
                    }
                )
                validate_annotation_evidence(row.item, final)
                final_path_value, final_hash = _publish_model(paths.accepted_root, final)
                status = "validated"
            else:
                validate_exclusion_evidence(row.item, candidate)
                final_path_value, final_hash = _publish_model(paths.exclusions_root, candidate)
                status = "excluded"
            final_path = Path(final_path_value)
            outcome = DocumentRunOutcome.model_validate(
                {
                    "schemaVersion": 1,
                    "documentId": row.item.source.documentId,
                    "status": status,
                    "attempts": attempt,
                    "reviews": reviews,
                    "documentEscalations": document_escalations,
                    "finalArtifactPath": _relative_to_run(final_path, paths.run_root),
                    "finalArtifactSha256": final_hash,
                    "callReceiptPaths": tuple(receipt_paths),
                    "callReceiptSha256s": tuple(receipt_hashes),
                },
                strict=True,
            )
            _publish_outcome(paths, outcome)
            return outcome
        except AgentCallFailure as error:
            receipt_path, receipt_hash = _publish_receipt(paths, error.receipt)
            receipt_paths.append(receipt_path)
            receipt_hashes.append(receipt_hash)
            validation_findings.append(str(error))
            if error.fatal_run or config.workflow.fail_fast:
                raise
            if error.terminal_document:
                return await _needs_review(
                    paths,
                    row,
                    attempts=attempt,
                    reviews=reviews,
                    document_escalations=document_escalations,
                    reason="provider_or_validation_failure",
                    findings=validation_findings,
                    receipt_paths=receipt_paths,
                    receipt_hashes=receipt_hashes,
                )
            retry_feedback = validation_findings[-1]
        except (AgentProviderError, WorkItemError, ValueError) as error:
            validation_findings.append(f"{type(error).__name__}: {error}")
            retry_feedback = validation_findings[-1]
            if config.workflow.fail_fast:
                raise

    return await _needs_review(
        paths,
        row,
        attempts=config.workflow.max_candidate_attempts,
        reviews=reviews,
        document_escalations=document_escalations,
        reason="candidate_attempts_exhausted",
        findings=validation_findings,
        receipt_paths=receipt_paths,
        receipt_hashes=receipt_hashes,
    )


def _load_outcomes(
    paths: AgentRunPaths, selected: Sequence[InventoriedWorkItem]
) -> tuple[DocumentRunOutcome, ...]:
    outcomes: list[DocumentRunOutcome] = []
    for row in selected:
        outcome = _load_existing_outcome(paths, row.item.source.documentId)
        if outcome is None:
            raise LabelingRunError(
                f"document has no committed outcome: {row.item.source.documentId}"
            )
        outcomes.append(outcome)
    return tuple(outcomes)


def _reference_quality_artifacts(
    config: AgentLabelingConfig,
    outcomes: Sequence[DocumentRunOutcome],
    generated_targets: dict[str, dict[str, Any]],
    references_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], bytes | None, int]:
    if config.source.reference_target_mode == "absent":
        if references_by_id:
            raise LabelingRunError("reference targets exist while reference_target_mode=absent")
        return (
            {
                "status": "not_configured",
                "reason": (
                    "the source is unlabeled; no reference targets were supplied to models or "
                    "used for post-run comparison"
                ),
            },
            None,
            0,
        )

    task = get_training_task("bill_of_lading_semantic_v2")
    generated_texts: list[str] = []
    reference_texts: list[str] = []
    validated_generated_texts: list[str] = []
    validated_reference_texts: list[str] = []
    quality_rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        reference = references_by_id.get(outcome.documentId)
        if reference is None:
            raise LabelingRunError(
                f"selected document lacks a frozen reference target: {outcome.documentId}"
            )
        generated = generated_targets.get(outcome.documentId)
        generated_text = canonical_json(generated) if generated is not None else ""
        reference_text = canonical_json(reference)
        generated_texts.append(generated_text)
        reference_texts.append(reference_text)
        if generated is not None:
            validated_generated_texts.append(generated_text)
            validated_reference_texts.append(reference_text)
        assessment = assess_prediction(generated_text, reference_text, task)
        true_positive = len(assessment.predicted_field_values & assessment.reference_field_values)
        predicted_total = len(assessment.predicted_field_values)
        reference_total = len(assessment.reference_field_values)
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / reference_total if reference_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        quality_rows.append(
            {
                "documentId": outcome.documentId,
                "status": outcome.status,
                "referenceTargetSha256": sha256_bytes(canonical_json_bytes(reference)),
                "generatedTargetSha256": (
                    sha256_bytes(canonical_json_bytes(generated)) if generated is not None else None
                ),
                "jsonValid": assessment.json_valid,
                "schemaValid": assessment.schema_valid,
                "canonicalExactMatch": assessment.canonical_exact_match,
                "predictedFieldValues": predicted_total,
                "referenceFieldValues": reference_total,
                "fieldValuePrecision": precision,
                "fieldValueRecall": recall,
                "fieldValueF1": f1,
            }
        )
    all_selected_metrics, _ = structured_metrics(generated_texts, reference_texts, task)
    validated_metrics = (
        structured_metrics(validated_generated_texts, validated_reference_texts, task)[0]
        if validated_generated_texts
        else None
    )
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in quality_rows)
    return (
        {
            "status": "available",
            "reference": (
                "hash-pinned previously accepted semantic-v2 labels; comparison-only and never "
                "supplied to extractor or reviewer calls"
            ),
            "all_selected": all_selected_metrics,
            "validated_only": validated_metrics,
        },
        payload,
        len(quality_rows),
    )


def publish_agent_run(
    config: AgentLabelingConfig,
    selected: Sequence[InventoriedWorkItem],
    *,
    frozen_selection: Sequence[InventoriedWorkItem] | None = None,
) -> Path:
    """Publish terminal outcomes and complete persisted-call accounting manifest-last.

    ``frozen_selection`` is supplied only for an interrupted-run snapshot.  It
    keeps the original selected population in scope while ``selected`` contains
    exactly the documents with atomically committed outcomes.  Provider calls
    persisted for an in-flight document remain costed and explicitly orphaned;
    untouched documents remain explicitly unprocessed.
    """

    paths = agent_run_paths(config)
    outcomes = _load_outcomes(paths, selected)
    selected_by_id = {row.item.source.documentId: row for row in selected}
    if len(selected_by_id) != len(selected):
        raise LabelingRunError("terminal publication selection contains duplicate documents")
    frozen = tuple(frozen_selection) if frozen_selection is not None else tuple(selected)
    frozen_by_id = {row.item.source.documentId: row for row in frozen}
    if len(frozen_by_id) != len(frozen):
        raise LabelingRunError("frozen publication selection contains duplicate documents")
    if not set(selected_by_id).issubset(frozen_by_id):
        raise LabelingRunError("terminal outcomes are not a subset of the frozen selection")
    references_by_id = reference_targets(config)
    training_rows: list[dict[str, Any]] = []
    generated_targets: dict[str, dict[str, Any]] = {}
    usage_rows: list[dict[str, Any]] = []
    total_input = total_output = total_cache_read = total_cache_write = 0
    total_cost = Decimal(0)
    referenced_cost = Decimal(0)
    orphaned_cost = Decimal(0)
    priced_receipts = 0
    total_receipts = 0
    referenced_receipts = 0
    orphaned_receipts = 0
    local_receipts = 0
    unpriced_openai_receipts = 0
    receipt_root = paths.state_root / "call-receipts"
    if not receipt_root.is_dir() or receipt_root.is_symlink():
        raise LabelingRunError("run call receipt directory is missing or unsafe")
    unexpected_receipt_documents = {
        child.name
        for child in receipt_root.iterdir()
        if child.name not in frozen_by_id
    }
    if unexpected_receipt_documents:
        raise LabelingRunError(
            "run contains call receipts for unselected document(s): "
            + ", ".join(sorted(unexpected_receipt_documents))
        )

    def account(row: dict[str, Any], receipt_accounting: _ReceiptAccounting) -> None:
        nonlocal total_input, total_output, total_cache_read, total_cache_write
        nonlocal total_cost, referenced_cost, orphaned_cost
        nonlocal priced_receipts, total_receipts, referenced_receipts, orphaned_receipts
        nonlocal local_receipts, unpriced_openai_receipts
        usage_rows.append(row)
        total_input += receipt_accounting.input_tokens
        total_output += receipt_accounting.output_tokens
        total_cache_read += receipt_accounting.cache_read_tokens
        total_cache_write += receipt_accounting.cache_write_tokens
        total_cost += receipt_accounting.cost
        referenced_cost += receipt_accounting.referenced_cost
        orphaned_cost += receipt_accounting.orphaned_cost
        priced_receipts += receipt_accounting.priced_calls
        total_receipts += receipt_accounting.referenced_calls + receipt_accounting.orphaned_calls
        referenced_receipts += receipt_accounting.referenced_calls
        orphaned_receipts += receipt_accounting.orphaned_calls
        local_receipts += receipt_accounting.local_calls
        unpriced_openai_receipts += receipt_accounting.unpriced_openai_calls

    for outcome in outcomes:
        row = selected_by_id[outcome.documentId]
        if outcome.status == "validated":
            annotation_path = paths.run_root / outcome.finalArtifactPath
            annotation = BillOfLadingDualCargoAnnotation.model_validate_json(
                read_regular_file_bytes(annotation_path), strict=True
            )
            validate_annotation_evidence(row.item, annotation)
            normal_target = annotation.normalLabel.canonical_target()
            relation_target = annotation.relationExplicitLabel.canonical_target()
            generated_targets[outcome.documentId] = normal_target
            training_rows.append(
                {
                    "documentId": outcome.documentId,
                    "joinedRawText": row.item.joinedRawText,
                    "joinedRawTextSha256": row.item.source.joinedRawTextSha256,
                    "target": relation_target,
                    "normalTarget": normal_target,
                    "validatedAnnotationPath": outcome.finalArtifactPath,
                    "validatedAnnotationSha256": outcome.finalArtifactSha256,
                }
            )
        stored_receipts = _stored_call_receipts(config, paths, row, outcome)
        usage_row, receipt_accounting = _usage_row(
            document_id=outcome.documentId,
            status=outcome.status,
            attempts=outcome.attempts,
            reviews=outcome.reviews,
            document_escalations=outcome.documentEscalations,
            receipts=stored_receipts,
        )
        account(usage_row, receipt_accounting)

    receipt_document_ids = {child.name for child in receipt_root.iterdir()}
    incomplete_receipt_ids = sorted(receipt_document_ids - set(selected_by_id))
    for document_id in incomplete_receipt_ids:
        row = frozen_by_id[document_id]
        stored_receipts = _stored_uncommitted_call_receipts(config, paths, row)
        receipt_values = tuple(receipt for _, receipt, _ in stored_receipts)
        usage_row, receipt_accounting = _usage_row(
            document_id=document_id,
            status="interrupted_without_outcome",
            attempts=max(receipt.candidateAttempt for receipt in receipt_values),
            reviews=sum(receipt.stage == "review" for receipt in receipt_values),
            document_escalations=sum(
                receipt.stage == "document_layout" for receipt in receipt_values
            ),
            receipts=stored_receipts,
        )
        account(usage_row, receipt_accounting)

    reference_quality, quality_payload, quality_rows = _reference_quality_artifacts(
        config, outcomes, generated_targets, references_by_id
    )

    training_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in training_rows)
    training_path = paths.training_root / "records.jsonl"
    atomic_publish_bytes(training_path, training_payload)
    usage_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in usage_rows)
    usage_path = paths.run_root / "usage-by-document.jsonl"
    atomic_publish_bytes(usage_path, usage_payload)
    quality_path = paths.run_root / "reference-quality-by-document.jsonl"
    if quality_payload is not None:
        atomic_publish_bytes(quality_path, quality_payload)
    counts = {
        status: sum(outcome.status == status for outcome in outcomes)
        for status in ("validated", "excluded", "needs_review")
    }
    manifest = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "task": config.task,
        "publication_scope": (
            "interrupted_snapshot" if len(frozen) != len(selected) else "complete_run"
        ),
        "selected_documents": len(selected),
        "selected_pages": sum(row.item.source.documentPageCount for row in selected),
        "frozen_selection_documents": len(frozen),
        "frozen_selection_pages": sum(
            row.item.source.documentPageCount for row in frozen
        ),
        "interrupted_documents_with_receipts": len(incomplete_receipt_ids),
        "unprocessed_documents": len(frozen) - len(selected) - len(incomplete_receipt_ids),
        "outcomes": counts,
        "training_records": len(training_rows),
        "training_ready": counts["needs_review"] == 0 and len(frozen) == len(selected),
        "provider_assignments": config.assignments.model_dump(mode="json"),
        "reference_quality": reference_quality,
        "usage": {
            "input_tokens": total_input,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "output_tokens": total_output,
            "call_receipts": total_receipts,
            "outcome_referenced_call_receipts": referenced_receipts,
            "orphaned_call_receipts": orphaned_receipts,
            "priced_call_receipts": priced_receipts,
            "unpriced_openai_call_receipts": unpriced_openai_receipts,
            "local_call_receipts": local_receipts,
            "configured_list_price_estimate_complete": unpriced_openai_receipts == 0,
            "configured_list_price_estimate_usd": str(total_cost),
            "outcome_referenced_configured_list_price_estimate_usd": str(
                referenced_cost
            ),
            "orphaned_configured_list_price_estimate_usd": str(orphaned_cost),
            "cost_basis": "configured_public_list_rates_not_billed_cost",
            "cost_scope": (
                "Estimate from the pinned public OpenAI list rates and provider-reported token "
                "buckets. It is not an invoice or account-balance measurement; credits, billing "
                "adjustments, and local compute are excluded. Every persisted receipt is counted, "
                "including calls orphaned from the final outcome by an interrupted/resumed run. "
                "Any response lacking token usage is counted separately."
            ),
        },
        "files": [
            {
                "kind": "training_records",
                "path": _relative_to_run(training_path, paths.run_root),
                "rows": len(training_rows),
                "bytes": len(training_payload),
                "sha256": sha256_bytes(training_payload),
            },
            {
                "kind": "usage_by_document",
                "path": _relative_to_run(usage_path, paths.run_root),
                "rows": len(usage_rows),
                "bytes": len(usage_payload),
                "sha256": sha256_bytes(usage_payload),
            },
        ]
        + (
            [
                {
                    "kind": "reference_quality_by_document",
                    "path": _relative_to_run(quality_path, paths.run_root),
                    "rows": quality_rows,
                    "bytes": len(quality_payload),
                    "sha256": sha256_bytes(quality_payload),
                }
            ]
            if quality_payload is not None
            else []
        ),
    }
    manifest_path = paths.run_root / "manifest.json"
    atomic_publish_json(manifest_path, manifest)
    return manifest_path


async def run_agent_labeling(
    config: AgentLabelingConfig,
    *,
    project_root: Path,
    gateway: AgentGateway,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    selected = prepare_agent_run(config, project_root=project_root)
    limit = asyncio.Semaphore(config.workflow.max_concurrent_documents)
    progress_lock = asyncio.Lock()
    outcomes: list[DocumentRunOutcome] = []

    async def bounded(row: InventoriedWorkItem) -> None:
        async with limit:
            outcome = await process_document(config, agent_run_paths(config), gateway, row)
        async with progress_lock:
            outcomes.append(outcome)
            if progress is not None:
                progress(
                    LabelingProgress(
                        total_documents=len(selected),
                        processed_documents=len(outcomes),
                        validated_documents=sum(row.status == "validated" for row in outcomes),
                        excluded_documents=sum(row.status == "excluded" for row in outcomes),
                        needs_review_documents=sum(
                            row.status == "needs_review" for row in outcomes
                        ),
                    )
                )

    async with asyncio.TaskGroup() as group:
        for row in selected:
            group.create_task(bounded(row))
    publish_agent_run(config, selected)
    return {
        "selected_documents": len(selected),
        "validated_documents": sum(row.status == "validated" for row in outcomes),
        "excluded_documents": sum(row.status == "excluded" for row in outcomes),
        "needs_review_documents": sum(row.status == "needs_review" for row in outcomes),
    }
