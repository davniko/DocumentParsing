from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.usage import RequestUsage, RunUsage
from pypdf import PdfReader, PdfWriter

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingDualCargoAnnotation
from document_ocr.label_schemas.common import LabelWarning, RawOcrAnchor, RawOcrValueEvidence
from document_ocr.label_schemas.semantic_review import SemanticReviewFinding
from document_ocr.labeling_agents.adjudication import (
    AdjudicationConfig,
    AdjudicationError,
    publish_adjudication_run,
)
from document_ocr.labeling_agents.cli import _exception_leaf_details
from document_ocr.labeling_agents.config import (
    AgentLabelingConfig,
    OpenAIPricingConfig,
    OpenAIResponsesProviderConfig,
    load_agent_labeling_config,
)
from document_ocr.labeling_agents.consolidation import (
    ConsolidationConfig,
    consolidate_labeling_runs,
)
from document_ocr.labeling_agents.corrections import (
    correction_envelope_json_schema,
    merge_correction_values,
    merge_target_scoped_correction,
    normalize_correction_paths,
    unchanged_correction_paths,
)
from document_ocr.labeling_agents.deterministic_annotation import (
    DeterministicAnnotationError,
    _occurrence_excerpt,
    build_compact_annotation,
)
from document_ocr.labeling_agents.models import (
    AgentCallReceipt,
    AnnotationDraft,
    CompactAnnotationDraft,
    CompactCorrectionEnvelope,
    CompactDocumentAssistanceRequest,
    CompactDocumentLayoutGuidance,
    CompactReviewDraft,
    CompactReviewWireDraft,
    DocumentAssistanceRequest,
    DocumentLayoutGuidance,
    IndependentReviewArtifact,
    ModelResponseReceipt,
    NeedsReviewRecord,
    ReviewDocumentAssistanceRequest,
    ReviewDraft,
)
from document_ocr.labeling_agents.orchestrator import (
    _publish_receipt,
    _resolve_compact_review,
    _review_retry_feedback,
    process_document,
    publish_agent_run,
)
from document_ocr.labeling_agents.provider import (
    AgentCallFailure,
    AgentCallResult,
    AgentProviderError,
    PydanticAgentGateway,
    _call_transcript,
    _compact_correction_wire,
    _ExtractionValidationContext,
    _hydrate_document_request,
    _hydrate_layout_guidance,
    _hydrate_review,
    _parse_compact_correction_wire,
    _parse_compact_extraction_wire,
    _parse_extraction_wire,
    _parse_ollama_version,
    _postprocess_failure,
    _price_request,
    _provider_schema,
    _ReviewValidationContext,
    _validate_compact_correction_wire,
    _validate_compact_review_wire,
)
from document_ocr.labeling_agents.review_policy import (
    ReviewPolicyError,
    validate_review_policy,
)
from document_ocr.labeling_agents.work_items import (
    AgentWorkItem,
    InventoriedWorkItem,
    WorkItemError,
    agent_run_paths,
    materialize_raw_ocr_evidence,
    page_texts,
    pdf_bytes_for_pages,
    select_work_items,
    validate_raw_ocr_evidence,
)


def test_exception_leaf_details_exposes_nested_task_failures() -> None:
    error = ExceptionGroup(
        "concurrent labeling failed",
        [
            ValueError("invalid candidate"),
            ExceptionGroup("nested", [RuntimeError("")]),
        ],
    )

    assert _exception_leaf_details(error) == (
        {"error_type": "ValueError", "message": "invalid candidate"},
        {"error_type": "RuntimeError", "message": "RuntimeError"},
    )


def _pricing() -> OpenAIPricingConfig:
    return OpenAIPricingConfig.model_validate(
        {
            "currency": "USD",
            "effective_date": date(2026, 8, 22),
            "source_url": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
            "input_usd_per_million": 0.20,
            "cached_input_usd_per_million": 0.02,
            "cache_write_multiplier": 1.25,
            "output_usd_per_million": 1.20,
            "long_context_input_threshold_tokens": 272000,
            "long_context_input_multiplier": 2.0,
            "long_context_output_multiplier": 1.5,
        },
        strict=True,
    )


def _config(
    tmp_path: Path,
    *,
    document_escalations: int = 0,
    reference_target_mode: str = "required",
) -> AgentLabelingConfig:
    work_items = tmp_path / "work-items"
    extraction = tmp_path / "extraction"
    output = tmp_path / "output"
    work_items.mkdir(exist_ok=True)
    extraction.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    selection = tmp_path / "selection.jsonl"
    selection_row: dict[str, Any] = {"documentId": "doc_" + "a" * 64}
    if reference_target_mode == "required":
        selection_row["target"] = {
            "schemaVersion": "2.0.0",
            "documentPatch": {"billOfLadingNumber": "HBL-001"},
        }
    selection_payload = canonical_json_bytes(selection_row) + b"\n"
    selection.write_bytes(selection_payload)
    return AgentLabelingConfig.model_validate(
        {
            "schema_version": 3,
            "task": "bill_of_lading_dual_cargo_v3",
            "environment_file": ".env",
            "run": {
                "run_id": "test-labeling-v1",
                "output_root": str(output),
                "resume": True,
            },
            "source": {
                "reference_target_mode": reference_target_mode,
                "expected_documents": 1,
                "expected_pages": 1,
                "expected_inventory_sha256": "b" * 64,
                "work_item_roots": [
                    {
                        "id": "fixture",
                        "root": str(work_items),
                        "include_glob": "*.json",
                        "selection_records_path": str(selection),
                        "selection_records_sha256": sha256_bytes(selection_payload),
                        "documents": 1,
                    }
                ],
                "extraction_runs": [{"run_id": "fixture-extraction", "root": str(extraction)}],
            },
            "selection": {"count": 1, "seed": 42, "namespace": "fixture"},
            "prompts": {
                name: {"path": f"prompts/{name}.md", "sha256": "c" * 64}
                for name in ("extractor", "reviewer", "document_layout")
            },
            "providers": [
                {
                    "id": "ollama_test",
                    "kind": "ollama_openai_chat",
                    "model": "fixture-model:latest",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_key": "ollama",
                    "native_json_schema": True,
                    "supports_pdf_documents": False,
                    "reasoning_mode": "provider_default",
                    "request_timeout_seconds": 30.0,
                    "transport_max_retries": 2,
                    "max_output_tokens": 4096,
                }
            ],
            "assignments": {
                "labeler": "ollama_test",
                "reviewer": "ollama_test",
                "document_layout": "ollama_test",
            },
            "workflow": {
                "max_concurrent_documents": 2,
                "max_concurrent_model_requests": 2,
                "max_candidate_attempts": 2,
                "max_document_escalations_per_document": document_escalations,
                "max_pdf_bytes": 50_000_000,
                "structured_output_retries": 1,
                "max_requests_per_agent_run": 2,
                "input_tokens_limit_per_agent_run": 100000,
                "output_tokens_limit_per_agent_run": 10000,
                "document_mode": "on_explicit_request",
                "pdf_page_scope": "requested_pages",
                "require_independent_review": True,
                "fail_fast": False,
            },
        },
        strict=True,
    )


def _work_item() -> InventoriedWorkItem:
    page_text = "B/L NO: HBL-001"
    joined = f"--- PAGE 1 ---\n{page_text}"
    item = AgentWorkItem.model_validate(
        {
            "source": {
                "documentId": "doc_" + "a" * 64,
                "extractionRunId": "fixture-extraction",
                "sourceUri": "file:///fixture.pdf",
                "localCanonicalPath": "/fixture.pdf",
                "sourceSha256": "d" * 64,
                "documentPageCount": 1,
                "joinedRawTextSha256": sha256_bytes(joined.encode()),
                "pages": (
                    {
                        "pageIndex": 0,
                        "pageNumber": 1,
                        "pageId": "page-1",
                        "extractionId": "extract-1",
                        "rawOcrTextSha256": sha256_bytes(page_text.encode()),
                        "rawResponsePath": "raw-responses/page-1.json",
                        "rawResponseSha256": "e" * 64,
                        "rasterPath": "page-images/page-1.png",
                        "rasterSha256": "f" * 64,
                    },
                ),
            },
            "joinedRawText": joined,
        },
        strict=True,
    )
    payload = canonical_json_bytes(item.model_dump(mode="json"))
    return InventoriedWorkItem("fixture", Path("/fixture.json"), sha256_bytes(payload), item)


def _work_item_with_text(page_text: str) -> AgentWorkItem:
    row = _work_item()
    joined = f"--- PAGE 1 ---\n{page_text}"
    page = row.item.source.pages[0].model_copy(
        update={"rawOcrTextSha256": sha256_bytes(page_text.encode())}
    )
    source = row.item.source.model_copy(
        update={
            "joinedRawTextSha256": sha256_bytes(joined.encode()),
            "pages": (page,),
        }
    )
    return row.item.model_copy(update={"source": source, "joinedRawText": joined})


def _work_item_with_pages(*page_texts: str) -> AgentWorkItem:
    if not page_texts:
        raise ValueError("at least one page is required")
    row = _work_item()
    joined = "\n\n".join(
        f"--- PAGE {page_number} ---\n{page_text}"
        for page_number, page_text in enumerate(page_texts, start=1)
    )
    page_template = row.item.source.pages[0]
    pages = tuple(
        page_template.model_copy(
            update={
                "pageIndex": page_number - 1,
                "pageNumber": page_number,
                "pageId": f"page-{page_number}",
                "extractionId": f"extract-{page_number}",
                "rawOcrTextSha256": sha256_bytes(page_text.encode()),
            }
        )
        for page_number, page_text in enumerate(page_texts, start=1)
    )
    source = row.item.source.model_copy(
        update={
            "documentPageCount": len(page_texts),
            "joinedRawTextSha256": sha256_bytes(joined.encode()),
            "pages": pages,
        }
    )
    return row.item.model_copy(update={"source": source, "joinedRawText": joined})


def _work_item_with_pdf(
    tmp_path: Path,
    *,
    empty_password_encrypted: bool = False,
    broken_xref: bool = False,
) -> InventoriedWorkItem:
    source_path = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    if empty_password_encrypted:
        writer.encrypt(user_password="", owner_password="fixture-owner-password")
    with source_path.open("wb") as stream:
        writer.write(stream)
    source_payload = source_path.read_bytes()
    if broken_xref:
        marker = b"startxref\n"
        offset_start = source_payload.rfind(marker) + len(marker)
        offset_end = source_payload.find(b"\n", offset_start)
        xref_offset = int(source_payload[offset_start:offset_end])
        source_payload = (
            source_payload[:offset_start]
            + str(xref_offset + 1).encode("ascii")
            + source_payload[offset_end:]
        )
        source_path.write_bytes(source_payload)
    base = _work_item()
    source = base.item.source.model_copy(
        update={
            "sourceUri": source_path.as_uri(),
            "localCanonicalPath": str(source_path),
            "sourceSha256": sha256_bytes(source_payload),
        }
    )
    item = base.item.model_copy(update={"source": source})
    payload = canonical_json_bytes(item.model_dump(mode="json"))
    return InventoriedWorkItem("fixture", tmp_path / "fixture.json", sha256_bytes(payload), item)


def _receipt(
    stage: str,
    attempt: int,
    *,
    work_item_sha256: str = "1" * 64,
) -> AgentCallReceipt:
    response = ModelResponseReceipt.model_validate(
        {
            "inputTokens": 100,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "outputTokens": 20,
            "reasoningTokens": 5,
            "costUsd": None,
        },
        strict=True,
    )
    now = datetime.now(UTC)
    return AgentCallReceipt.model_validate(
        {
            "receiptSchemaVersion": 2,
            "callId": f"call-{stage}-{attempt}",
            "documentId": "doc_" + "a" * 64,
            "stage": stage,
            "candidateAttempt": attempt,
            "providerId": "ollama_test",
            "providerKind": "ollama_openai_chat",
            "model": "fixture-model:latest",
            "reasoningEffort": "provider_default",
            "workerId": f"worker-{stage}-{attempt}",
            "startedAt": now,
            "completedAt": now,
            "durationMs": 1.0,
            "workItemSha256": work_item_sha256,
            "promptSha256": "2" * 64,
            "outputSha256": "3" * 64,
            "requests": 1,
            "responses": (response,),
            "inputTokens": 100,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "outputTokens": 20,
            "costUsd": None,
            "costStatus": "not_applicable_local",
        },
        strict=True,
    )


def _failed_receipt(
    stage: str,
    attempt: int,
    *,
    work_item_sha256: str = "1" * 64,
) -> AgentCallReceipt:
    value = _receipt(stage, attempt, work_item_sha256=work_item_sha256).model_dump(mode="python")
    value.update(
        {
            "status": "error",
            "outputSha256": None,
            "errorType": "UnexpectedModelBehavior",
            "errorMessage": "structured output retries exhausted",
        }
    )
    return AgentCallReceipt.model_validate(value, strict=True)


def _annotation() -> AnnotationDraft:
    return AnnotationDraft.model_validate(
        {
            "decision": "annotation",
            "documentType": "bill_of_lading",
            "normalLabel": {
                "schemaVersion": "2.0.0",
                "documentPatch": {"billOfLadingNumber": "HBL-001"},
            },
            "relationExplicitLabel": {
                "schemaVersion": "3.0.0-experimental",
                "documentPatch": {"billOfLadingNumber": "HBL-001"},
            },
            "evidence": (
                {
                    "targetPath": "documentPatch.billOfLadingNumber",
                    "evidenceKind": "verbatim",
                    "rawOcrEvidence": (
                        {
                            "pageNumber": 1,
                            "rawValue": "HBL-001",
                            "ocrExcerpt": "B/L NO: HBL-001",
                        },
                    ),
                    "imageUse": "not_used",
                },
            ),
            "warnings": (),
            "relationEvidence": (),
            "decisionNotes": ("The headed B/L number is explicit in raw OCR.",),
        },
        strict=True,
    )


def _compact_annotation(document_patch: dict[str, Any]) -> CompactAnnotationDraft:
    return CompactAnnotationDraft.model_validate(
        {
            "decision": "annotation",
            "documentType": "bill_of_lading",
            "relationExplicitLabel": {
                "schemaVersion": "3.0.0-experimental",
                "documentPatch": document_patch,
            },
            "warnings": (),
            "decisionNotes": ("Compact source-of-truth fixture.",),
        },
        strict=True,
    )


def _compact_correction(corrections: dict[str, Any]) -> CompactCorrectionEnvelope:
    return CompactCorrectionEnvelope.model_validate(
        {"corrections": corrections},
        strict=True,
    )


def _review(*, passed: bool) -> ReviewDraft:
    checks = {
        "rawOcrTruthBoundary": "pass",
        "evidenceIntegrity": "pass",
        "semanticCompleteness": "pass" if passed else "fail",
        "semanticCorrectness": "pass",
        "contaminationAndRedundancy": "pass",
        "documentUnitAndRelationships": "pass",
    }
    findings: list[dict[str, Any]] = []
    if not passed:
        findings.append(
            {
                "severity": "blocking",
                "category": "missing_field",
                "message": "Retry fixture for the complete headed value.",
                "targetPaths": ("documentPatch.billOfLadingNumber",),
                "rawOcrEvidence": (
                    {
                        "pageNumber": 1,
                        "rawValue": "HBL-001",
                        "ocrExcerpt": "B/L NO: HBL-001",
                    },
                ),
                "imageUse": "not_used",
            }
        )
    return ReviewDraft.model_validate(
        {
            "decision": "review",
            "result": "pass" if passed else "fail",
            "checks": checks,
            "findings": tuple(findings),
            "summary": "Candidate passed." if passed else "Candidate needs a fresh retry.",
        },
        strict=True,
    )


def test_native_json_boundary_accepts_json_arrays_and_iso_dates_without_coercing_python() -> None:
    value = _annotation().model_dump(mode="json")
    for label_name in ("normalLabel", "relationExplicitLabel"):
        patch = value[label_name]["documentPatch"]
        patch["issueDate"] = "2024-06-03"
        patch["forwardingAndExportReferences"] = ["REF-001"]

    with pytest.raises(ValueError, match=r"date_type|tuple_type"):
        AnnotationDraft.model_validate(value, strict=True)

    parsed = _parse_extraction_wire(value)

    assert isinstance(parsed, AnnotationDraft)
    assert parsed.normalLabel.documentPatch.issueDate == date(2024, 6, 3)
    assert parsed.normalLabel.documentPatch.forwardingAndExportReferences == ("REF-001",)


def test_compact_native_json_boundary_contains_one_label_view() -> None:
    value = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "issueDate": date(2024, 6, 3)}
    ).model_dump(mode="json")

    parsed = _parse_compact_extraction_wire(value)

    assert isinstance(parsed, CompactAnnotationDraft)
    assert parsed.relationExplicitLabel.documentPatch.issueDate == date(2024, 6, 3)
    assert "normalLabel" not in value
    assert "evidence" not in value
    assert "relationEvidence" not in value


def test_compact_wire_joins_physical_lines_and_deduplicates_repeated_facts() -> None:
    value = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "ASSY OPEN CELL",
                    "hsCodes": ("8524911000",),
                },
            )
        }
    ).model_dump(mode="json")
    cargo_group = value["relationExplicitLabel"]["documentPatch"]["cargoGroups"][0]
    cargo_group["description"] = "ASSY\n  OPEN CELL"
    cargo_group["hsCodes"] = ["8524911000", "8524911000"]

    parsed = _parse_compact_extraction_wire(value)

    assert isinstance(parsed, CompactAnnotationDraft)
    group = parsed.relationExplicitLabel.documentPatch.cargoGroups
    assert group is not None
    assert group[0].description == "ASSY OPEN CELL"
    assert group[0].hsCodes == ("8524911000",)


def test_compact_wire_collapses_null_only_nested_objects() -> None:
    value = _compact_annotation(
        {
            "parties": {
                "deliveryAgent": {
                    "name": "UNIFREIGHT GLOBAL LOGISTICS",
                    "contactDetails": None,
                }
            }
        }
    ).model_dump(mode="json")
    delivery_agent = value["relationExplicitLabel"]["documentPatch"]["parties"]["deliveryAgent"]
    delivery_agent["contactDetails"] = {
        "contactName": None,
        "phoneNumbers": None,
        "emailAddresses": None,
        "websiteUrls": None,
    }

    parsed = _parse_compact_extraction_wire(value)

    assert isinstance(parsed, CompactAnnotationDraft)
    parties = parsed.relationExplicitLabel.documentPatch.parties
    assert parties is not None
    assert parties.deliveryAgent is not None
    assert parties.deliveryAgent.contactDetails is None


def test_target_scoped_correction_preserves_every_untargeted_fact() -> None:
    base = _compact_annotation(
        {
            "billOfLadingNumber": "HBL-001",
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "MINIBAR",
                    "hsCodes": ("841850190000",),
                },
            ),
        }
    )
    revision = _compact_annotation(
        {
            "billOfLadingNumber": "WRONG-UNRELATED-CHANGE",
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "WRONG UNRELATED DESCRIPTION",
                    "hsCodes": ("84185019000000",),
                },
            ),
        }
    )

    merged = merge_target_scoped_correction(
        base,
        revision,
        ("documentPatch.cargoGroups[0].hsCodes",),
    )
    patch = merged.relationExplicitLabel.documentPatch
    groups = patch.cargoGroups

    assert patch.billOfLadingNumber == "HBL-001"
    assert groups is not None
    assert groups[0].description == "MINIBAR"
    assert groups[0].hsCodes == ("84185019000000",)


def test_unchanged_correction_paths_detects_no_op_but_not_a_real_revision() -> None:
    base = _compact_annotation({"billOfLadingNumber": "HBL-001"})
    unchanged = _compact_annotation({"billOfLadingNumber": "HBL-001"})
    changed = _compact_annotation({"billOfLadingNumber": "HBL-002"})

    assert unchanged_correction_paths(
        base, unchanged, ("documentPatch.billOfLadingNumber",)
    ) == ("documentPatch.billOfLadingNumber",)
    assert unchanged_correction_paths(
        base, changed, ("documentPatch.billOfLadingNumber",)
    ) == ()


def test_unchanged_correction_paths_treats_a_null_ancestor_as_an_absent_leaf() -> None:
    base = _compact_annotation({"parties": {"shipper": {"name": "ACME", "contactDetails": None}}})
    paths = ("documentPatch.parties.shipper.contactDetails.phoneNumbers",)
    changed = merge_correction_values(
        base,
        {paths[0]: ["0579-85299742"]},
        paths,
    )

    assert unchanged_correction_paths(base, changed, paths) == ()
    shipper = changed.relationExplicitLabel.documentPatch.parties.shipper
    assert shipper is not None and shipper.contactDetails is not None
    assert shipper.contactDetails.phoneNumbers == ("0579-85299742",)


def test_target_scoped_document_type_correction_preserves_document_patch() -> None:
    base = _compact_annotation({"billOfLadingNumber": "HBL-001"})

    assert normalize_correction_paths(("documentType",)) == ("documentType",)
    merged = merge_correction_values(
        base,
        {"documentType": "sea_waybill"},
        ("documentType",),
    )

    assert merged.documentType == "sea_waybill"
    assert merged.relationExplicitLabel.documentPatch.billOfLadingNumber == "HBL-001"


def test_exact_path_document_patch_correction_maps_into_relation_label() -> None:
    base = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "issueDate": None}
    )

    merged = merge_correction_values(
        base,
        {"documentPatch.issueDate": "2024-04-28"},
        ("documentPatch.issueDate",),
    )

    patch = merged.relationExplicitLabel.documentPatch
    assert patch.billOfLadingNumber == "HBL-001"
    assert patch.issueDate == date(2024, 4, 28)


def test_exact_path_correction_fails_closed_when_authorized_scope_was_omitted() -> None:
    base = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "issueDate": None}
    )
    with pytest.raises(
        ValueError,
        match=r"missing authorized path: documentPatch\.issueDate",
    ):
        merge_correction_values(
            base,
            {},
            ("documentPatch.issueDate",),
        )


def test_exact_path_correction_rejects_an_empty_authorization_surface() -> None:
    base = _compact_annotation({"billOfLadingNumber": "HBL-001"})

    with pytest.raises(ValueError, match="requires at least one path"):
        merge_correction_values(base, {}, ())


def test_exact_path_correction_preserves_an_explicit_null_deletion() -> None:
    base = _compact_annotation({"route": {"placeOfReceipt": {"name": "MARDAS"}}})

    merged = merge_correction_values(
        base,
        {"documentPatch.route": None},
        ("documentPatch.route",),
    )

    assert merged.relationExplicitLabel.documentPatch.route is None


def test_exact_path_correction_wire_normalizes_only_its_authored_values() -> None:
    parsed = _parse_compact_correction_wire(
        {
            "corrections": {
                "documentPatch.containers": [{"containerNumber": "UACU 507479-1"}],
                "documentPatch.cargoGroups": [
                    {"groupId": "g1", "description": "MINI\nBAR"}
                ],
            },
        }
    )

    assert isinstance(parsed, CompactCorrectionEnvelope)
    containers = parsed.corrections["documentPatch.containers"]
    cargo_groups = parsed.corrections["documentPatch.cargoGroups"]
    assert isinstance(containers, list) and isinstance(containers[0], dict)
    assert containers[0]["containerNumber"] == "UACU5074791"
    assert isinstance(cargo_groups, list) and isinstance(cargo_groups[0], dict)
    assert cargo_groups[0]["description"] == "MINI BAR"


def test_exact_path_correction_rejects_every_unexpected_scope() -> None:
    base = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "issueDate": None}
    )

    with pytest.raises(
        ValueError,
        match=r"unauthorized path: documentPatch\.billOfLadingNumber",
    ):
        merge_correction_values(
            base,
            {
                "documentPatch.issueDate": "2024-04-28",
                "documentPatch.billOfLadingNumber": "WRONG",
            },
            ("documentPatch.issueDate",),
        )


def test_exact_path_indexed_scalar_replaces_only_that_scalar() -> None:
    base = _compact_annotation(
        {
            "containers": (
                {"containerNumber": "ADMU5112728"},
                {"containerNumber": "CAAU8913108"},
                {"containerNumber": "CAAU9218501"},
            ),
            "cargoGroups": ({"groupId": "g1", "description": "UNCHANGED"},),
        }
    )

    merged = merge_correction_values(
        base,
        {"documentPatch.containers[2].containerNumber": "UACU5074791"},
        ("documentPatch.containers[2].containerNumber",),
    )

    patch = merged.relationExplicitLabel.documentPatch
    assert patch.containers is not None
    assert tuple(row.containerNumber for row in patch.containers) == (
        "ADMU5112728",
        "CAAU8913108",
        "UACU5074791",
    )
    assert patch.cargoGroups is not None
    assert patch.cargoGroups[0].description == "UNCHANGED"


def test_exact_path_lifted_collection_replaces_complete_collection() -> None:
    base = _compact_annotation(
        {
            "billOfLadingNumber": "HBL-001",
            "parties": {
                "shipper": {
                    "contactDetails": {"phoneNumbers": ("+20 111", "+20 222")}
                }
            },
        }
    )

    merged = merge_correction_values(
        base,
        {"documentPatch.parties.shipper.contactDetails.phoneNumbers": ["+20 111"]},
        ("documentPatch.parties.shipper.contactDetails.phoneNumbers[1]",),
    )

    patch = merged.relationExplicitLabel.documentPatch
    assert patch.billOfLadingNumber == "HBL-001"
    assert patch.parties is not None and patch.parties.shipper is not None
    contacts = patch.parties.shipper.contactDetails
    assert contacts is not None
    assert contacts.phoneNumbers == ("+20 111",)


def test_exact_path_contextual_validator_checks_the_merged_candidate() -> None:
    item = _work_item_with_text("B/L NO: HBL-001\nISSUE DATE: 28 APR 2024")
    base = _compact_annotation({"billOfLadingNumber": "HBL-001", "issueDate": None})
    context = RunContext(
        deps=_ExtractionValidationContext(
            work_item=item,
            pdf_grouping_used=False,
            correction_base=base,
            correction_paths=("documentPatch.issueDate",),
        ),
        model=TestModel(),
        usage=RunUsage(),
    )
    wire = {"corrections": {"documentPatch.issueDate": "2024-04-28"}}

    assert _validate_compact_correction_wire(context, wire) == wire


def test_reviewer_schema_accepts_top_level_document_type_target() -> None:
    finding = SemanticReviewFinding.model_validate(
        {
            "severity": "blocking",
            "category": "incorrect_field",
            "message": "Correct the document type.",
            "targetPaths": ("documentType",),
            "rawOcrEvidence": (
                {
                    "pageNumber": 1,
                    "rawValue": "BILL OF LADING",
                    "ocrExcerpt": "BILL OF LADING",
                },
            ),
            "imageUse": "not_used",
        },
        strict=True,
    )

    assert finding.targetPaths == ("documentType",)


def test_correction_path_validation_aggregates_invalid_and_sidecar_targets() -> None:
    with pytest.raises(
        ValueError,
        match=r"correction contract found 2 issue\(s\).+warnings sidecar.+doesNotExist",
    ):
        normalize_correction_paths(
            (
                "documentPatch.warnings[1]",
                "documentPatch.cargoGroups[0].doesNotExist",
            )
        )


def test_exact_list_item_correction_scope_lifts_to_parent_collection() -> None:
    assert normalize_correction_paths(
        (
            "documentPatch.parties.shipper.contactDetails.phoneNumbers[1]",
            "documentPatch.cargoPackages[1]",
        )
    ) == (
        "documentPatch.parties.shipper.contactDetails.phoneNumbers",
        "documentPatch.cargoPackages",
    )


def test_goods_origin_leaf_correction_lifts_to_the_content_bearing_parent() -> None:
    target = "documentPatch.cargoGroups[0].origin.identifier"

    assert normalize_correction_paths((target,)) == (
        "documentPatch.cargoGroups[0].origin",
    )
    schema = correction_envelope_json_schema((target,))
    corrections = schema["properties"]["corrections"]
    assert corrections["required"] == ["documentPatch.cargoGroups[0].origin"]
    assert target not in corrections["properties"]


def test_goods_origin_leaf_deletion_removes_the_optional_parent_atomically() -> None:
    base = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "ROPE FOLDING MACHINE",
                    "origin": {"identifier": "MLTRLS2600891"},
                },
            ),
        }
    )

    merged = merge_correction_values(
        base,
        {"documentPatch.cargoGroups[0].origin": None},
        ("documentPatch.cargoGroups[0].origin.identifier",),
    )

    groups = merged.relationExplicitLabel.documentPatch.cargoGroups
    assert groups is not None
    assert groups[0].description == "ROPE FOLDING MACHINE"
    assert groups[0].origin is None


def test_lifted_list_item_deletion_preserves_unrelated_facts() -> None:
    base = _compact_annotation(
        {
            "billOfLadingNumber": "HBL-001",
            "parties": {
                "shipper": {
                    "name": "SHIPPER NAME",
                    "contactDetails": {"phoneNumbers": ("+20 111", "+20 222")},
                }
            },
        }
    )
    revision = _compact_annotation(
        {
            "billOfLadingNumber": "UNRELATED-WRONG-CHANGE",
            "parties": {
                "shipper": {
                    "name": "UNRELATED-WRONG-NAME",
                    "contactDetails": {"phoneNumbers": ("+20 111",)},
                }
            },
        }
    )

    merged = merge_target_scoped_correction(
        base,
        revision,
        ("documentPatch.parties.shipper.contactDetails.phoneNumbers[1]",),
    )

    patch = merged.relationExplicitLabel.documentPatch
    assert patch.billOfLadingNumber == "HBL-001"
    assert patch.parties is not None
    assert patch.parties.shipper is not None
    assert patch.parties.shipper.name == "SHIPPER NAME"
    assert patch.parties.shipper.contactDetails is not None
    assert patch.parties.shipper.contactDetails.phoneNumbers == ("+20 111",)


def test_lifted_list_item_scope_can_reorder_the_parent_collection() -> None:
    base = _compact_annotation(
        {
            "parties": {
                "shipper": {
                    "contactDetails": {"phoneNumbers": ("+20 111", "+20 222")}
                }
            }
        }
    )
    revision = _compact_annotation(
        {
            "parties": {
                "shipper": {
                    "contactDetails": {"phoneNumbers": ("+20 222", "+20 111")}
                }
            }
        }
    )

    merged = merge_target_scoped_correction(
        base,
        revision,
        ("documentPatch.parties.shipper.contactDetails.phoneNumbers[0]",),
    )

    parties = merged.relationExplicitLabel.documentPatch.parties
    assert parties is not None and parties.shipper is not None
    assert parties.shipper.contactDetails is not None
    assert parties.shipper.contactDetails.phoneNumbers == ("+20 222", "+20 111")


def test_indexed_object_scalar_correction_does_not_widen_to_its_collection() -> None:
    base = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "MINIBAR",
                    "hsCodes": ("841850190000",),
                },
            )
        }
    )
    revision = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "MINI BAR",
                    "hsCodes": ("999999999999",),
                },
            )
        }
    )

    merged = merge_target_scoped_correction(
        base,
        revision,
        ("documentPatch.cargoGroups[0].description",),
    )

    groups = merged.relationExplicitLabel.documentPatch.cargoGroups
    assert groups is not None
    assert groups[0].description == "MINI BAR"
    assert groups[0].hsCodes == ("841850190000",)


def test_merge_aggregates_all_missing_revision_paths_without_mutating_base() -> None:
    base = _compact_annotation(
        {
            "cargoGroups": (
                {"groupId": "g1", "description": "FIRST"},
                {"groupId": "g2", "description": "SECOND", "hsCodes": ("123456",)},
            )
        }
    )
    revision = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "description": "FIRST"},)}
    )
    before = base.model_dump(mode="json")

    with pytest.raises(
        ValueError,
        match=r"correction contract found 2 issue\(s\).+description.+hsCodes",
    ):
        merge_target_scoped_correction(
            base,
            revision,
            (
                "documentPatch.cargoGroups[1].description",
                "documentPatch.cargoGroups[1].hsCodes",
            ),
        )

    assert base.model_dump(mode="json") == before


def test_correction_regenerates_warning_sidecar_from_untouched_base_warnings() -> None:
    corrected_warning = LabelWarning.model_validate(
        {
            "code": "ambiguous_ocr_candidates",
            "message": "The issue date was previously unresolved.",
            "pageNumbers": (1,),
            "targetPath": "documentPatch.issueDate",
        },
        strict=True,
    )
    retained_warning = LabelWarning.model_validate(
        {
            "code": "schema_cannot_represent",
            "message": "The standalone fax field is not represented.",
            "pageNumbers": (1,),
            "targetPath": None,
        },
        strict=True,
    )
    injected_warning = LabelWarning.model_validate(
        {
            "code": "other",
            "message": "A revision-side warning must not replace immutable sidecars.",
            "pageNumbers": (1,),
            "targetPath": "documentPatch.billOfLadingNumber",
        },
        strict=True,
    )
    base = _compact_annotation({"billOfLadingNumber": "HBL-001"}).model_copy(
        update={"warnings": (corrected_warning, retained_warning)}
    )
    revision = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "issueDate": date(2024, 4, 28)}
    ).model_copy(update={"warnings": (injected_warning,)})

    merged = merge_target_scoped_correction(
        base,
        revision,
        ("documentPatch.issueDate",),
    )

    assert merged.warnings == (retained_warning,)
    assert merged.decisionNotes == (
        "Applied reviewer-authorized target-scoped correction to: documentPatch.issueDate",
    )


def test_narrow_full_revision_cannot_remove_a_nullable_ancestor() -> None:
    base = _compact_annotation(
        {
            "route": {
                "placeOfReceipt": {"name": "MARDAS"},
                "portOfLoading": None,
            }
        }
    )
    revision = _compact_annotation(
        {
            "route": {
                "placeOfReceipt": None,
                "portOfLoading": {"name": "MARDAS"},
            }
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            r"null ancestor for narrow correction path: "
            r"documentPatch\.route\.placeOfReceipt\.name; target "
            r"documentPatch\.route\.placeOfReceipt explicitly"
        ),
    ):
        merge_target_scoped_correction(
            base,
            revision,
            (
                "documentPatch.route.placeOfReceipt.name",
                "documentPatch.route.portOfLoading.name",
            ),
        )

    merged = merge_target_scoped_correction(
        base,
        revision,
        (
            "documentPatch.route.placeOfReceipt",
            "documentPatch.route.portOfLoading",
        ),
    )
    route = merged.relationExplicitLabel.documentPatch.route
    assert route is not None and route.portOfLoading is not None
    assert route.placeOfReceipt is None
    assert route.portOfLoading.name == "MARDAS"


def test_allocation_correction_scope_is_the_complete_discriminated_group() -> None:
    shared = {
        "cargoGroups": ({"groupId": "g1", "description": "MINIBAR"},),
        "cargoPackages": (
            {
                "packageId": "p1",
                "groupId": "g1",
                "quantity": 3,
                "typeDescription": "PALLETS",
            },
        ),
    }
    base = _compact_annotation(
        {
            **shared,
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "container_membership_only",
                    "allocations": ({"containerNumber": "UACU5074791"},),
                },
            ),
        }
    )
    revision = _compact_annotation(
        {
            **shared,
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageId": "p1",
                    "allocations": ({"containerNumber": "UACU5074791", "packageQuantity": 3},),
                },
            ),
        }
    )

    assert normalize_correction_paths(
        (
            "documentPatch.cargoAllocationGroups[0].coverage",
            "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
        )
    ) == ("documentPatch.cargoAllocationGroups[0]",)
    merged = merge_target_scoped_correction(
        base,
        revision,
        ("documentPatch.cargoAllocationGroups[0].coverage",),
    )

    allocations = merged.relationExplicitLabel.documentPatch.cargoAllocationGroups
    assert allocations is not None
    allocation = allocations[0]
    assert allocation.coverage == "single_package_level"
    assert allocation.packageId == "p1"
    assert allocation.allocations[0].packageQuantity == 3


def test_document_wide_hs_evidence_can_ground_each_governed_cargo_group() -> None:
    item = _work_item_with_text(
        "DESCRIPTION OF GOODS\nPRODUCT A\nPRODUCT B\nHS.CODE : 3920 1025 0000"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "PRODUCT A",
                    "hsCodes": ("392010250000",),
                },
                {
                    "groupId": "g2",
                    "description": "PRODUCT B",
                    "hsCodes": ("392010250000",),
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    groups = annotation.relationExplicitLabel.documentPatch.cargoGroups
    assert groups is not None
    assert groups[0].hsCodes == groups[1].hsCodes == ("392010250000",)


def test_compact_wire_normalizes_printed_container_separators() -> None:
    value = _compact_annotation({"containers": ({"containerNumber": "ADMU5112728"},)}).model_dump(
        mode="json"
    )
    value["relationExplicitLabel"]["documentPatch"]["containers"][0]["containerNumber"] = (
        "ADMU/511272/8"
    )

    parsed = _parse_compact_extraction_wire(value)

    assert isinstance(parsed, CompactAnnotationDraft)
    containers = parsed.relationExplicitLabel.documentPatch.containers
    assert containers is not None
    assert containers[0].containerNumber == "ADMU5112728"


def test_provider_schema_explains_critical_extraction_and_review_fields() -> None:
    extraction_schema = _provider_schema(CompactAnnotationDraft)
    review_schema = _provider_schema(CompactReviewWireDraft)
    request_schema = _provider_schema(CompactDocumentAssistanceRequest)
    serialized_extraction = json.dumps(extraction_schema)
    serialized_review = json.dumps(review_schema)
    serialized_request = json.dumps(request_schema)

    def descriptions(schema: Any, field_name: str) -> list[str]:
        found: list[str] = []
        if isinstance(schema, dict):
            properties = schema.get("properties")
            if isinstance(properties, dict) and isinstance(
                field_schema := properties.get(field_name), dict
            ):
                description = field_schema.get("description")
                if isinstance(description, str):
                    found.append(description)
            for child in schema.values():
                found.extend(descriptions(child, field_name))
        elif isinstance(schema, list):
            for child in schema:
                found.extend(descriptions(child, field_name))
        return found

    for field_name in (
        "issueDate",
        "shippedOnBoardDate",
        "documentType",
        "shipper",
        "notifyParties",
        "forwardingAgent",
        "placeOfReceipt",
        "portOfDischarge",
        "paymentArrangement",
        "forwardingAndExportReferences",
        "cargoAllocationGroups",
        "hsCodes",
        "phoneNumbers",
        "sameAs",
        "typeDescription",
        "verifiedGrossMass",
        "description",
    ):
        assert descriptions(extraction_schema, field_name)
    assert "Departure, sailing, or ETD".lower() in serialized_extraction.lower()
    assert "day/month/year" in serialized_extraction.lower()
    assert "completed zero-original count" in serialized_extraction
    assert "standalone FAX" in serialized_extraction
    assert "blank adjacent headings" in serialized_extraction
    assert "ERN" in serialized_extraction
    assert "referenced party is present" in serialized_extraction
    assert "notify-specific" in serialized_extraction
    assert "blocking for any representable missing" in serialized_review
    assert "nearest preceding governing table/header page" in serialized_request


def test_v4_prompts_include_maritime_equivalents_and_exclude_form_business_caps() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prompt_root = project_root / "prompts" / "labeling_agents"
    extractor = (
        prompt_root / "mpci_bl_relation_single_source_v4_extractor.md"
    ).read_text(encoding="utf-8")
    reviewer = (
        prompt_root / "mpci_bl_relation_single_source_v4_reviewer.md"
    ).read_text(encoding="utf-8")
    layout_helper = (prompt_root / "mpci_bl_pdf_layout_helper.md").read_text(
        encoding="utf-8"
    )

    for prompt in (extractor, reviewer):
        lowered = prompt.lower()
        assert "maritime" in lowered
        assert "multimodal" in lowered
        assert "sea leg" in lowered
        assert "air" in lowered
        assert "road" in lowered
        assert "submission cap" in lowered
        assert "geographic locality" in lowered or "another locality" in lowered
        assert "month-first" in lowered
        assert "named-month" in lowered
        assert "arithmetic sum" in lowered or "never add or sum" in lowered
        assert "acconts" in lowered
        assert "volumes:24" in lowered
        assert "standalone `fax:`" in lowered
        assert "tel & fax" in lowered
        assert "zero-original" in lowered
        assert "5848932722024070020" in lowered
        assert "export references svc contract" in lowered
        assert "via medicinos linija uab" in lowered or "unscoped `via x`" in lowered
        assert "referenced party" in lowered
        assert "under tare" in lowered or "tare-column" in lowered
        assert "sameas" in lowered
        assert "override" in lowered
        assert "exact repeated container/seal/package block" in lowered
        assert "size 2x2-8" in lowered
        assert "ordinary container" in lowered
        assert "p/i no" in lowered
        assert "foreign exporter country" in lowered
        assert "20 boxes" in lowered and "100 boxes" in lowered
        assert "application for delivery must be made to" in lowered
        assert "b/l-number box" in lowered
        assert "unheaded continuation row" in lowered
        assert "governing" in lowered and "header page" in lowered

    lowered_extractor = extractor.lower()
    assert "exact-path correction envelope" in lowered_extractor
    assert "required literal property" in lowered_extractor
    assert "normalized `allowedcorrectionpaths`" in lowered_extractor
    assert "complete corrected collection" in lowered_extractor
    assert "do not emit" in lowered_extractor and "warnings" in lowered_extractor
    lowered_layout = layout_helper.lower()
    assert "governing table/header page" in lowered_layout
    assert "visual column geometry" in lowered_layout
    assert "continuation page" in lowered_layout


def test_v6_prompts_pin_audited_hard_case_semantics() -> None:
    prompt_root = Path(__file__).resolve().parents[1] / "prompts" / "labeling_agents"
    extractor = (
        prompt_root / "mpci_bl_relation_single_source_v6_extractor.md"
    ).read_text(encoding="utf-8")
    reviewer = (
        prompt_root / "mpci_bl_relation_single_source_v6_reviewer.md"
    ).read_text(encoding="utf-8")

    for prompt in (extractor, reviewer):
        lowered = " ".join(prompt.lower().split())
        assert "0.000 m3" in lowered
        assert "wrapped hs" in lowered
        assert "unlinked_package_quantities" in lowered
        assert "distinct" in lowered and "container" in lowered
        assert "20st" in lowered and "intermediate bulk containers" in lowered
        assert "forwarding agent references" in lowered
        assert "flashpoint" in lowered and "packing group" in lowered
        assert "kilo" in lowered and "mtq" in lowered

    assert "never replace a printed zero" in extractor.lower()
    assert "block any candidate that substitutes a nonzero" in reviewer.lower()


def test_v7_prompts_pin_observed_remediation_semantics() -> None:
    prompt_root = Path(__file__).resolve().parents[1] / "prompts" / "labeling_agents"
    extractor = (
        prompt_root / "mpci_bl_relation_single_source_v7_extractor.md"
    ).read_text(encoding="utf-8")
    reviewer = (
        prompt_root / "mpci_bl_relation_single_source_v7_reviewer.md"
    ).read_text(encoding="utf-8")

    for prompt in (extractor, reviewer):
        lowered = " ".join(prompt.lower().split())
        assert "caed" in lowered and "acid" in lowered
        assert "3 0 jan 2024" in lowered and "2024-01-30" in lowered
        assert "53,334 kg" in lowered and "15 m3" in lowered
        assert "0.249 cbm" in lowered
        assert "measurement: mtq" in lowered
        assert "non-negotiable copy" in lowered
        assert "flash pt" in lowered
        assert "two" in lowered and "contact" in lowered


def test_v8_prompts_pin_correction_boundary_semantics() -> None:
    prompt_root = Path(__file__).resolve().parents[1] / "prompts" / "labeling_agents"
    prompts = tuple(
        (prompt_root / name).read_text(encoding="utf-8")
        for name in (
            "mpci_bl_relation_single_source_v8_extractor.md",
            "mpci_bl_relation_single_source_v8_reviewer.md",
        )
    )

    for prompt in prompts:
        lowered = " ".join(prompt.lower().split())
        assert "53,334 kg" in lowered and "15,700 kgm" in lowered
        assert "generic `ref #`" in lowered and "forwarding agent" in lowered
        assert "valid imo checksum" in lowered
        assert "shippers load, stow and count" in lowered
        assert "responsibility" in lowered and "not an operational" in lowered


def test_v9_prompts_pin_face_field_and_title_term_semantics() -> None:
    prompt_root = Path(__file__).resolve().parents[1] / "prompts" / "labeling_agents"
    prompts = tuple(
        (prompt_root / name).read_text(encoding="utf-8")
        for name in (
            "mpci_bl_relation_single_source_v9_extractor.md",
            "mpci_bl_relation_single_source_v9_reviewer.md",
        )
    )

    for prompt in prompts:
        lowered = " ".join(prompt.lower().split())
        assert "delivered unto order or assigns" in lowered
        assert "freight payable at prepaid" in lowered
        assert "as agreed payable at destination" in lowered
        assert "completed face" in lowered


def test_v10_prompts_pin_explicit_cargo_origin_semantics() -> None:
    prompt_root = Path(__file__).resolve().parents[1] / "prompts" / "labeling_agents"
    prompts = tuple(
        (prompt_root / name).read_text(encoding="utf-8")
        for name in (
            "mpci_bl_relation_single_source_v10_extractor.md",
            "mpci_bl_relation_single_source_v10_reviewer.md",
        )
    )

    for prompt in prompts:
        lowered = " ".join(prompt.lower().split())
        assert "made in <country>" in lowered
        assert "cargogroups[0].origin.name" in lowered
        assert "exactly as printed" in lowered


def test_v11_reviewer_pins_dg_scope_and_express_release_semantics() -> None:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "labeling_agents"
        / "mpci_bl_relation_single_source_v11_reviewer.md"
    ).read_text(encoding="utf-8")
    lowered = " ".join(prompt.lower().split())

    assert "express release - no originals issued" in lowered
    assert "negotiability=non_negotiable" in lowered
    assert "no standalone proper-shipping-name or packing-group field" in lowered
    assert "never force them into `description`, `additionalinformation`" in lowered
    assert "class `2` or `2.1`" in lowered
    assert "category `gases`" in lowered


def test_compact_allocation_union_enforces_coverage_shape_before_projection() -> None:
    value = _compact_annotation(
        {
            "containers": ({"containerNumber": "UACU5074791"},),
            "cargoGroups": ({"groupId": "g1", "description": "SEEDS"},),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "container_membership_only",
                    "allocations": ({"containerNumber": "UACU5074791"},),
                },
            ),
        }
    )
    canonical = value.relationExplicitLabel.to_canonical()
    allocation = canonical.documentPatch.cargoAllocationGroups
    assert allocation is not None
    assert allocation[0].packageIds == ()
    assert allocation[0].allocations[0].packageQuantity is None

    malformed = value.model_dump(mode="json")
    malformed_group = malformed["relationExplicitLabel"]["documentPatch"]["cargoAllocationGroups"][
        0
    ]
    malformed_group["packageIds"] = ["p1"]

    with pytest.raises(ValueError, match="extra_forbidden"):
        _parse_compact_extraction_wire(malformed)


def test_compact_projection_removes_only_structurally_duplicated_values() -> None:
    draft = _compact_annotation(
        {
            "route": {"portOfDischarge": {"name": "ALEXANDRIA, EGYPT", "country": "EGYPT"}},
            "containers": ({"containerNumber": "UACU5074791"},),
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "SEEDS",
                    "marksAndNumbers": ("UACU5074791", "1289901"),
                },
            ),
        }
    )

    canonical = draft.relationExplicitLabel.to_canonical()

    route = canonical.documentPatch.route
    assert route is not None
    assert route.portOfDischarge is not None
    assert route.portOfDischarge.name == "ALEXANDRIA"
    assert route.portOfDischarge.country == "EGYPT"
    groups = canonical.documentPatch.cargoGroups
    assert groups is not None
    assert groups[0].marksAndNumbers == ("1289901",)


def test_compact_annotation_projects_and_grounds_evidence_locally() -> None:
    item = _work_item_with_text("B/L NO: HBL-001\nHS CODE: 9001.90\n10 CARTONS")
    draft = _compact_annotation(
        {
            "billOfLadingNumber": "HBL-001",
            "cargoGroups": ({"groupId": "g1", "hsCodes": ("900190",)},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 10,
                    "typeDescription": "CARTONS",
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    goods = annotation.normalLabel.documentPatch.goodsItems
    assert goods is not None
    assert goods[0].hsCodes == ("900190",)
    assert goods[0].packages is not None
    assert goods[0].packages[0].quantity == 10
    assert {row.targetPath for row in annotation.evidence} == {
        "documentPatch.billOfLadingNumber",
        "documentPatch.goodsItems[0].packages[0].quantity",
        "documentPatch.goodsItems[0].packages[0].type",
        "documentPatch.goodsItems[0].hsCodes[0]",
    }
    hs_evidence = next(row for row in annotation.evidence if ".hsCodes[" in row.targetPath)
    assert hs_evidence.evidenceKind == "contextual_code"
    assert hs_evidence.rawOcrEvidence[0].rawValue == "9001.90"


def test_compact_annotation_rejects_an_ungrounded_identifier_instead_of_repairing_it() -> None:
    item = _work_item_with_text("B/L NO: HBL-001\nHS CODE: 9001.90")
    draft = _compact_annotation(
        {
            "billOfLadingNumber": "HBL-001",
            "cargoGroups": ({"groupId": "g1", "hsCodes": ("900120",)},),
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="not exactly groundable"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


def test_local_evidence_handles_glued_numbers_units_and_intervening_modeled_lines() -> None:
    item = _work_item_with_text(
        "HYBRID SUDAN GRASS\nHS CODE: 1209.29.91.60\n"
        "HYBRID SORGHUM SUDAN GRASS II, TREATED\n"
        "1,000x20KG BAGS\nGROSS WEIGHT: 20,500KGS"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": ("HYBRID SUDAN GRASS; HYBRID SORGHUM SUDAN GRASS II, TREATED"),
                    "grossWeight": {"value": 20500.0, "unit": "kilogram"},
                    "hsCodes": ("1209299160",),
                },
            ),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 1000,
                    "typeDescription": "20KG BAGS",
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    evidence = {row.targetPath: row for row in annotation.evidence}
    assert len(evidence["documentPatch.goodsItems[0].description"].rawOcrEvidence) > 1
    assert (
        evidence["documentPatch.goodsItems[0].packages[0].quantity"].rawOcrEvidence[0].rawValue
        == "1,000"
    )
    assert (
        evidence["documentPatch.goodsItems[0].grossWeight.unit"].rawOcrEvidence[0].rawValue == "KGS"
    )


@pytest.mark.parametrize(
    ("source", "printed_unit"),
    (
        ("NET WEIGHT: 40MT", "MT"),
        ("NET WEIGHT: 40 MTS", "MTS"),
        ("NET WEIGHT: 40 M/T", "M/T"),
        ("NET WEIGHT: 40 METRIC TONS", "METRIC TONS"),
        ("NET WEIGHT: 40 TONNE", "TONNE"),
        ("NET WEIGHT: 40 TONNES", "TONNES"),
    ),
)
def test_compact_annotation_retains_and_grounds_printed_metric_tonnes(
    source: str, printed_unit: str
) -> None:
    item = _work_item_with_text(source)
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "netWeight": {"value": 40.0, "unit": "metric_tonne"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    relation_mass = annotation.relationExplicitLabel.documentPatch.cargoGroups[0].netWeight
    normal_mass = annotation.normalLabel.documentPatch.goodsItems[0].netWeight
    assert relation_mass is not None
    assert normal_mass is not None
    assert relation_mass.unit == "metric_tonne"
    assert normal_mass.unit == "metric_tonne"
    evidence = {row.targetPath: row for row in annotation.evidence}
    assert (
        evidence["documentPatch.goodsItems[0].netWeight.value"].rawOcrEvidence[0].rawValue
        == "40"
    )
    assert (
        evidence["documentPatch.goodsItems[0].netWeight.unit"].rawOcrEvidence[0].rawValue
        == printed_unit
    )


def test_kilogram_unit_evidence_does_not_match_pkgs_suffix() -> None:
    item = _work_item_with_text(
        "MARKS AND NUMBERS\nNO.OF PKGS\n16CARTONS\n"
        "GROSS WEIGHT\n46.6KGS\n0.472CBM"
    )
    annotation = build_compact_annotation(
        item,
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "description": "CARTONS",
                        "grossWeight": {"value": 46.6, "unit": "kilogram"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    evidence = next(
        row
        for row in annotation.evidence
        if row.targetPath == "documentPatch.goodsItems[0].grossWeight.unit"
    )
    assert evidence.rawOcrEvidence[0].rawValue == "KGS"
    assert "46.6KGS" in evidence.rawOcrEvidence[0].ocrExcerpt
    assert "NO.OF PKGS" not in evidence.rawOcrEvidence[0].ocrExcerpt


def test_compact_annotation_rejects_bare_tons_as_metric_tonnes() -> None:
    item = _work_item_with_text("NET WEIGHT: 40 TONS")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "netWeight": {"value": 40.0, "unit": "metric_tonne"},
                },
            )
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="not exactly groundable"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


def test_metric_tonne_measure_keeps_three_decimal_fraction() -> None:
    item = _work_item_with_text(
        "Total Net weight: 42.000 MT\nTotal Gross weight: 43.512 MT"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "netWeight": {"value": 42.0, "unit": "metric_tonne"},
                    "grossWeight": {"value": 43.512, "unit": "metric_tonne"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    goods = annotation.normalLabel.documentPatch.goodsItems[0]
    assert goods.netWeight is not None
    assert goods.grossWeight is not None
    assert goods.netWeight.value == 42.0
    assert goods.grossWeight.value == 43.512


def test_local_evidence_resolves_numeric_style_per_measure() -> None:
    item = _work_item_with_text(
        "TOTAL NET WEIGHT: 27.065,897 KG\nTOTAL VOLUME: 44.190 M3"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "netWeight": {"value": 27065.897, "unit": "kilogram"},
                    "volume": {"value": 44.19, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    evidence = {row.targetPath: row for row in annotation.evidence}
    assert (
        evidence["documentPatch.goodsItems[0].netWeight.value"].rawOcrEvidence[0].rawValue
        == "27.065,897"
    )
    assert (
        evidence["documentPatch.goodsItems[0].volume.value"].rawOcrEvidence[0].rawValue
        == "44.190"
    )


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("TOTAL NET WEIGHT: 27,950KGS", 27950.0),
        ("GROSS WEIGHT 27.947 KG 16 CAPS", 27947.0),
        ("TOTAL GROSS WEIGHT: 60,000KG", 60000.0),
    ],
)
def test_local_evidence_parses_grouped_mass_values(raw: str, value: float) -> None:
    field = "netWeight" if "NET" in raw else "grossWeight"
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    field: {"value": value, "unit": "kilogram"},
                },
            )
        }
    )

    annotation = build_compact_annotation(
        _work_item_with_text(raw), draft, pdf_grouping_used=False
    )

    mass = annotation.normalLabel.documentPatch.goodsItems[0].model_dump()[field]
    assert mass["value"] == value


def test_local_evidence_parses_dot_grouped_livestock_count_as_integer() -> None:
    item = _work_item_with_text("Description of Cargo\n4.713BULLS FOR FATTENING")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {"groupId": "g1", "description": "BULLS FOR FATTENING"},
            ),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 4713,
                    "typeDescription": "BULLS",
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    package = annotation.normalLabel.documentPatch.goodsItems[0].packages[0]
    assert package.quantity == 4713
    quantity_evidence = next(
        row
        for row in annotation.evidence
        if row.targetPath == "documentPatch.goodsItems[0].packages[0].quantity"
    )
    assert quantity_evidence.rawOcrEvidence[0].rawValue == "4.713"


@pytest.mark.parametrize(
    ("raw", "value", "raw_number", "raw_unit"),
    [
        ("VOLUME:26.347CBM", 26.347, "26.347", "CBM"),
        ("VOLUME:16.080M3", 16.08, "16.080", "M3"),
        ("VOLUME:39.010 cu. m.", 39.01, "39.010", "cu. m."),
    ],
)
def test_local_evidence_parses_observed_cubic_metre_units(
    raw: str, value: float, raw_number: str, raw_unit: str
) -> None:
    item = _work_item_with_text(raw)
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "volume": {"value": value, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    evidence = {row.targetPath: row for row in annotation.evidence}
    assert (
        evidence["documentPatch.goodsItems[0].volume.value"].rawOcrEvidence[0].rawValue
        == raw_number
    )
    assert (
        evidence["documentPatch.goodsItems[0].volume.unit"].rawOcrEvidence[0].rawValue
        == raw_unit
    )


def test_local_evidence_parses_split_heading_comma_decimal_volume() -> None:
    item = _work_item_with_text("Measure CBM\n13,230")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "volume": {"value": 13.23, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    volume = annotation.relationExplicitLabel.documentPatch.cargoGroups[0].volume
    assert volume is not None
    assert volume.value == 13.23


def test_local_evidence_parses_comma_decimal_volume_in_flattened_table_row() -> None:
    item = _work_item_with_text(
        "Marks & Nos.: No. package Kind of pack Description of goods Gross weight Kg M3\n"
        "Electro 1/PACKAGE s.l.w.a.c. 20,79 0,414"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "volume": {"value": 0.414, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    volume = annotation.relationExplicitLabel.documentPatch.cargoGroups[0].volume
    assert volume is not None
    assert volume.value == 0.414


@pytest.mark.parametrize(
    "printed_un_number",
    ("UN NO. 3082", "UN NUMBER: 3082", "UNDG NO:3082"),
)
def test_local_evidence_accepts_observed_un_number_headings(
    printed_un_number: str,
) -> None:
    item = _work_item_with_text(f"Goods description:\nIMCO 9 {printed_un_number}")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "dangerousGoods": ({"unNumber": "3082"},),
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    dangerous = annotation.normalLabel.documentPatch.goodsItems[0].dangerousGoods
    assert dangerous is not None
    assert dangerous[0].unNumber == "3082"


def test_local_evidence_requires_explicit_flash_point_context() -> None:
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "dangerousGoods": (
                        {
                            "unNumber": "1866",
                            "flashPoint": {
                                "temperature": {"value": 0.0, "unit": "celsius"}
                            },
                        },
                    ),
                },
            )
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="flash-point context"):
        build_compact_annotation(
            _work_item_with_text("UN 1866\nMEASUREMENT 0 C"),
            draft,
            pdf_grouping_used=False,
        )

    annotation = build_compact_annotation(
        _work_item_with_text("UN 1866\nFLASH POINT: 0 C"),
        draft,
        pdf_grouping_used=False,
    )
    dangerous = annotation.normalLabel.documentPatch.goodsItems[0].dangerousGoods
    assert dangerous is not None
    assert dangerous[0].flashPoint is not None
    assert dangerous[0].flashPoint.temperature.value == 0.0

    closed_cup = build_compact_annotation(
        _work_item_with_text("UN 1866 PACKING GROUP III (63.00 C-CC)"),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "dangerousGoods": (
                            {
                                "unNumber": "1866",
                                "flashPoint": {
                                    "temperature": {"value": 63.0, "unit": "celsius"},
                                    "packingGroupCategory": "LOW_DANGER",
                                },
                            },
                        ),
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )
    closed_cup_dangerous = (
        closed_cup.normalLabel.documentPatch.goodsItems[0].dangerousGoods
    )
    assert closed_cup_dangerous is not None
    assert closed_cup_dangerous[0].flashPoint is not None
    assert closed_cup_dangerous[0].flashPoint.temperature.value == 63.0


def test_local_evidence_accepts_glued_closed_cup_flash_point() -> None:
    item = _work_item_with_text(
        "HOUSTON TX\n10.750 CBM\n\n"
        "UN1987 ETHANOL SOLUTION, CLASS 3, PG III, (25C.C.C.)"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "dangerousGoods": (
                        {
                            "unNumber": "1987",
                            "flashPoint": {
                                "temperature": {"value": 25.0, "unit": "celsius"},
                                "packingGroupCategory": "LOW_DANGER",
                            },
                        },
                    ),
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    dangerous = annotation.normalLabel.documentPatch.goodsItems[0].dangerousGoods
    assert dangerous is not None
    assert dangerous[0].flashPoint is not None
    assert dangerous[0].flashPoint.temperature.value == 25.0


def test_local_evidence_parses_glued_pk_package_quantity() -> None:
    item = _work_item_with_text("CONTAINER CAAU8913108 95PK")
    draft = _compact_annotation(
        {
            "containers": ({"containerNumber": "CAAU8913108"},),
            "cargoGroups": ({"groupId": "g1"},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 95,
                    "typeDescription": "PK",
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    package = annotation.normalLabel.documentPatch.goodsItems[0].packages[0]
    assert package.quantity == 95
    assert package.type == "PK"


def test_local_evidence_uses_unambiguous_document_numeric_locale() -> None:
    item = _work_item_with_text(
        "90 BUNDLES\nGROSS WEIGHT 4.087,40 KGS\nTOTAL 406.253,00\nVOLUME 59,364 CBM"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 4087.4, "unit": "kilogram"},
                    "volume": {"value": 59.364, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    evidence = {row.targetPath: row for row in annotation.evidence}
    assert (
        evidence["documentPatch.goodsItems[0].grossWeight.value"].rawOcrEvidence[0].rawValue
        == "4.087,40"
    )
    assert (
        evidence["documentPatch.goodsItems[0].volume.value"].rawOcrEvidence[0].rawValue == "59,364"
    )


def test_local_evidence_handles_mixed_generated_container_number_format() -> None:
    item = _work_item_with_text(
        "CARGO GROSS 6.577,20 KG\nVGM\n"
        "TEMU2081080 ML-CL0261266 20 DRY 8'6 10 PALLET 7123.400 KGS"
    )
    draft = _compact_annotation(
        {
            "containers": (
                {
                    "containerNumber": "TEMU2081080",
                    "verifiedGrossMass": {"value": 7123.4, "unit": "kilogram"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    vgm = annotation.relationExplicitLabel.documentPatch.containers[0].verifiedGrossMass
    assert vgm is not None
    assert vgm.value == 7123.4


def test_container_row_mass_without_vgm_context_is_not_verified_gross_mass() -> None:
    item = _work_item_with_text(
        "MRSU5428343 ML-AE4174072 40 DRY 9'6 20 BINS 21756.000 KGS 40.000 CBM"
    )
    draft = _compact_annotation(
        {
            "containers": (
                {
                    "containerNumber": "MRSU5428343",
                    "verifiedGrossMass": {"value": 21756.0, "unit": "kilogram"},
                },
            )
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="requires explicit local VGM"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


def test_local_evidence_accepts_spacing_before_printed_date_comma() -> None:
    item = _work_item_with_text("DATE OF ISSUE: 10 APRIL , 2024")
    draft = _compact_annotation({"issueDate": date(2024, 4, 10)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.issueDate == date(2024, 4, 10)


def test_local_evidence_normalizes_named_month_ordinal_date() -> None:
    item = _work_item_with_text("PLACE AND DATE OF ISSUE\nJEDDAH\nMAR 4TH 2025")
    draft = _compact_annotation({"issueDate": date(2025, 3, 4)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    evidence = next(
        row for row in annotation.evidence if row.targetPath == "documentPatch.issueDate"
    )
    assert annotation.normalLabel.documentPatch.issueDate == date(2025, 3, 4)
    assert evidence.rawOcrEvidence[0].rawValue == "MAR 4TH 2025"


def test_local_evidence_normalizes_ocr_spaced_named_month_day() -> None:
    item = _work_item_with_text("PLACE AND DATE OF ISSUE\nDUBAI, 3 0 JAN 2024")
    draft = _compact_annotation({"issueDate": date(2024, 1, 30)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.issueDate == date(2024, 1, 30)


@pytest.mark.parametrize(
    ("raw", "expected", "raw_date"),
    [
        ("DATE OF ISSUE: JAN. 25,2023", date(2023, 1, 25), "JAN. 25,2023"),
        ("DATE OF ISSUE: JUL.15,2023", date(2023, 7, 15), "JUL.15,2023"),
        ("PLACE AND DATE OF ISSUE: GYAL 23/JUL/26", date(2026, 7, 23), "23/JUL/26"),
        ("DATE OF ISSUE: NOV/18/2023", date(2023, 11, 18), "NOV/18/2023"),
        ("DATE OF ISSUE: JAN.11.2026", date(2026, 1, 11), "JAN.11.2026"),
        (
            "PLACE AND DATE OF ISSUE: Norderstedt / 2025-DEC-24",
            date(2025, 12, 24),
            "2025-DEC-24",
        ),
    ],
)
def test_local_evidence_parses_observed_named_month_dates(
    raw: str, expected: date, raw_date: str
) -> None:
    item = _work_item_with_text(raw)
    draft = _compact_annotation({"issueDate": expected})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    evidence = next(
        row for row in annotation.evidence if row.targetPath == "documentPatch.issueDate"
    )
    assert annotation.normalLabel.documentPatch.issueDate == expected
    assert evidence.rawOcrEvidence[0].rawValue == raw_date


def test_ambiguous_numeric_date_defaults_to_day_first() -> None:
    item = _work_item_with_text("B/L NO: HBL-001\nSHIPPED ON BOARD\n01/04/2024")
    day_first = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "shippedOnBoardDate": date(2024, 4, 1)}
    )

    month_first = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "shippedOnBoardDate": date(2024, 1, 4)}
    )
    with pytest.raises(DeterministicAnnotationError, match="not exactly groundable"):
        build_compact_annotation(item, month_first, pdf_grouping_used=False)

    annotation = build_compact_annotation(item, day_first, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.shippedOnBoardDate == date(2024, 4, 1)


def test_ambiguous_numeric_date_remains_day_first_despite_document_month_first_date() -> None:
    item = _work_item_with_text(
        "B/L NO: HBL-001\nSHIPPED ON BOARD\n01/04/2024\nETA: 12/31/2024"
    )
    draft = _compact_annotation(
        {"billOfLadingNumber": "HBL-001", "shippedOnBoardDate": date(2024, 4, 1)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.shippedOnBoardDate == date(2024, 4, 1)


def test_adjudication_publishes_day_first_correction_with_pinned_lineage(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("B/L NO: HBL-001\nSHIPPED ON BOARD\n01/04/2024")
    annotation = build_compact_annotation(
        item,
        _compact_annotation(
            {
                "billOfLadingNumber": "HBL-001",
                "shippedOnBoardDate": date(2024, 4, 1),
            }
        ),
        pdf_grouping_used=False,
    )
    old_month_first = date(2024, 1, 4)
    annotation = annotation.model_copy(
        update={
            "normalLabel": annotation.normalLabel.model_copy(
                update={
                    "documentPatch": annotation.normalLabel.documentPatch.model_copy(
                        update={"shippedOnBoardDate": old_month_first}
                    )
                }
            ),
            "relationExplicitLabel": annotation.relationExplicitLabel.model_copy(
                update={
                    "documentPatch": annotation.relationExplicitLabel.documentPatch.model_copy(
                        update={"shippedOnBoardDate": old_month_first}
                    )
                }
            ),
        }
    )
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "candidate.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(annotation.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "day-first-adjudication-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "date_corrections": [
                        {
                            "target_path": "documentPatch.shippedOnBoardDate",
                            "raw_value": "01/04/2024",
                            "normalized_value": date(2024, 4, 1),
                        },
                    ],
                    "adjudication_notes": ["Applied the frozen day-first policy."],
                },
            ],
        },
        strict=True,
    )

    manifest_path = publish_adjudication_run(config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item_manifest = manifest["items"][0]
    artifact = BillOfLadingDualCargoAnnotation.model_validate_json(
        (manifest_path.parent / item_manifest["finalArtifactPath"]).read_bytes(),
        strict=True,
    )
    outcome_path = (
        manifest_path.parent / "state" / "outcomes" / f"{item.source.documentId}.json"
    )
    assert manifest["validatedDocuments"] == 1
    assert sha256_bytes(work_item_payload) == item_manifest["workItemSha256"]
    assert sha256_bytes(candidate_payload) == item_manifest["sourceCandidateSha256"]
    assert json.loads(outcome_path.read_text(encoding="utf-8"))["status"] == "validated"
    assert artifact.reviewStatus == "validated"
    assert artifact.normalLabel.documentPatch.shippedOnBoardDate == date(2024, 4, 1)
    assert artifact.relationExplicitLabel.documentPatch.shippedOnBoardDate == date(2024, 4, 1)


def test_adjudication_materializes_a_pinned_compact_draft(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("BILL OF LADING\nB/L NO: HBL-001")
    draft = _compact_annotation({"billOfLadingNumber": "HBL-001"})
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "compact-draft.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(draft.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "compact-draft-adjudication-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "candidate_kind": "compact_draft",
                    "pdf_grouping_used": False,
                    "adjudication_notes": ["Promoted the retained compact correction."],
                }
            ],
        },
        strict=True,
    )

    manifest_path = publish_adjudication_run(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = BillOfLadingDualCargoAnnotation.model_validate_json(
        (
            manifest_path.parent
            / manifest["items"][0]["finalArtifactPath"]
        ).read_bytes(),
        strict=True,
    )

    assert published.reviewStatus == "validated"
    assert published.normalLabel.documentPatch.billOfLadingNumber == "HBL-001"
    assert manifest["items"][0]["sourceCandidateKind"] == "compact_draft"
    assert manifest["items"][0]["sourcePdfGroupingUsed"] is False


def test_compact_draft_correction_is_applied_before_completeness_validation(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("GOODS\nHS CODE: 330290")
    draft = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "description": "GOODS"},)}
    )
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "compact-draft.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(draft.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "compact-draft-prebuild-correction-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "candidate_kind": "compact_draft",
                    "pdf_grouping_used": False,
                    "rebuild": {
                        "decision_notes": ["Added the omitted headed HS code."]
                    },
                    "target_corrections": [
                        {
                            "target_path": "/documentPatch/cargoGroups/0/hsCodes",
                            "expected_value": None,
                            "corrected_value": ["330290"],
                        }
                    ],
                    "adjudication_notes": ["Added the omitted headed HS code."],
                }
            ],
        },
        strict=True,
    )

    manifest_path = publish_adjudication_run(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = BillOfLadingDualCargoAnnotation.model_validate_json(
        (manifest_path.parent / manifest["items"][0]["finalArtifactPath"]).read_bytes(),
        strict=True,
    )

    assert published.normalLabel.documentPatch.goodsItems[0].hsCodes == ("330290",)


def test_compact_draft_adjudication_requires_explicit_pdf_grouping_provenance(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="explicit pdf_grouping_used"):
        AdjudicationConfig.model_validate(
            {
                "schema_version": 1,
                "run_id": "missing-pdf-provenance-v1",
                "output_root": str(tmp_path),
                "items": [
                    {
                        "document_id": "doc_" + "a" * 64,
                        "work_item_path": str(tmp_path / "work-item.json"),
                        "work_item_sha256": "b" * 64,
                        "candidate_path": str(tmp_path / "compact-draft.json"),
                        "candidate_sha256": "c" * 64,
                        "candidate_kind": "compact_draft",
                        "adjudication_notes": ["This configuration must fail."],
                    }
                ],
            },
            strict=True,
        )


def test_adjudication_removes_only_an_exact_pinned_warning(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("QUANTITY\n4.713 BULLS")
    warning = {
        "code": "ambiguous_ocr_candidates",
        "message": "The dot-grouped count was left unresolved.",
        "pageNumbers": (1,),
        "targetPath": "documentPatch.cargoPackages[0].quantity",
    }
    draft = _compact_annotation(
        {
            "cargoGroups": ({"groupId": "g1", "description": "BULLS"},),
            "cargoPackages": (
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 4713,
                    "typeDescription": "BULLS",
                },
            ),
        }
    ).model_copy(
        update={"warnings": (LabelWarning.model_validate(warning, strict=True),)}
    )
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "compact-draft.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(draft.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "warning-removal-adjudication-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "candidate_kind": "compact_draft",
                    "pdf_grouping_used": False,
                    "rebuild": {"decision_notes": ["Confirmed the printed integer count."]},
                    "warning_removals": [{**warning, "pageNumbers": [1]}],
                    "adjudication_notes": ["Removed the resolved ambiguity warning."],
                }
            ],
        },
        strict=True,
    )

    manifest_path = publish_adjudication_run(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = BillOfLadingDualCargoAnnotation.model_validate_json(
        (manifest_path.parent / manifest["items"][0]["finalArtifactPath"]).read_bytes(),
        strict=True,
    )

    assert published.warnings == ()
    assert manifest["items"][0]["warningRemovals"] == [
        {**warning, "pageNumbers": [1]}
    ]


def test_adjudication_adds_an_exact_pinned_warning_after_rebuild(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("MACHINERY\nWEIGHT\n2744.000 KGS")
    draft = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "description": "MACHINERY"},)}
    )
    added_warning = {
        "code": "schema_cannot_represent",
        "message": "Printed mass is governed only by a generic Weight heading.",
        "pageNumbers": [1],
        "targetPath": None,
    }
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "compact-draft.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(draft.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "warning-addition-adjudication-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "candidate_kind": "compact_draft",
                    "pdf_grouping_used": False,
                    "rebuild": {"decision_notes": ["Preserved generic mass warning."]},
                    "warning_additions": [added_warning],
                    "adjudication_notes": ["Added the audited representability warning."],
                }
            ],
        },
        strict=True,
    )

    manifest_path = publish_adjudication_run(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = BillOfLadingDualCargoAnnotation.model_validate_json(
        (manifest_path.parent / manifest["items"][0]["finalArtifactPath"]).read_bytes(),
        strict=True,
    )

    assert tuple(row.model_dump(mode="json") for row in published.warnings) == (
        added_warning,
    )
    assert manifest["items"][0]["warningAdditions"] == [added_warning]


def test_adjudication_rebuilds_evidence_and_applies_pinned_document_type(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text(
        "BILL OF LADING\nCONSIGNEE\nACME IMPORTS LIMITED\nNOTIFY PARTY\n"
        "NON-NEGOTIABLE COPY"
    )
    candidate = build_compact_annotation(
        item,
        CompactAnnotationDraft.model_validate(
            {
                "decision": "annotation",
                "documentType": "sea_waybill",
                "relationExplicitLabel": {
                    "schemaVersion": "3.0.0-experimental",
                    "documentPatch": {
                        "parties": {"consignee": {"name": "ACME IMPORTS LIMITED"}},
                        "negotiability": "non_negotiable",
                    },
                },
                "warnings": (),
                "decisionNotes": ("Initial candidate.",),
            },
            strict=True,
        ),
        pdf_grouping_used=False,
    )
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "candidate.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(candidate.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "candidate-rebuild-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "rebuild": {
                        "document_type": "bill_of_lading",
                        "decision_notes": [
                            "Rebuilt from the pinned semantic patch under the audited policy."
                        ],
                    },
                    "adjudication_notes": ["Confirmed the straight-bill document type."],
                }
            ],
        },
        strict=True,
    )

    manifest_path = publish_adjudication_run(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = manifest_path.parent / manifest["items"][0]["finalArtifactPath"]
    published = BillOfLadingDualCargoAnnotation.model_validate_json(
        artifact_path.read_bytes(), strict=True
    )

    assert published.documentType == "bill_of_lading"
    negotiability = next(
        row for row in published.evidence if row.targetPath == "documentPatch.negotiability"
    )
    assert "ACME IMPORTS LIMITED" in negotiability.rawOcrEvidence[0].rawValue


def test_adjudication_applies_guarded_target_correction_and_rebuilds_evidence(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("BILL OF LADING\nB/L NO: HBL-001\nFREIGHT COLLECT")
    candidate = build_compact_annotation(
        item,
        _compact_annotation(
            {
                "billOfLadingNumber": "HBL-001",
                "freight": {"paymentArrangement": "collect"},
            }
        ),
        pdf_grouping_used=False,
    )
    prepaid = candidate.normalLabel.documentPatch.freight.model_copy(
        update={"paymentArrangement": "prepaid"}
    )
    candidate = candidate.model_copy(
        update={
            "normalLabel": candidate.normalLabel.model_copy(
                update={
                    "documentPatch": candidate.normalLabel.documentPatch.model_copy(
                        update={"freight": prepaid}
                    )
                }
            ),
            "relationExplicitLabel": candidate.relationExplicitLabel.model_copy(
                update={
                    "documentPatch": candidate.relationExplicitLabel.documentPatch.model_copy(
                        update={"freight": prepaid}
                    )
                }
            ),
        }
    )
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "candidate.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(candidate.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "guarded-target-correction-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "rebuild": {
                        "decision_notes": [
                            "Rebuilt after applying the exact overseer correction."
                        ]
                    },
                    "target_corrections": [
                        {
                            "target_path": "/documentPatch/freight/paymentArrangement",
                            "expected_value": "prepaid",
                            "corrected_value": "collect",
                        }
                    ],
                    "adjudication_notes": ["Confirmed explicit collect wording."],
                }
            ],
        },
        strict=True,
    )

    manifest_path = publish_adjudication_run(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = manifest_path.parent / manifest["items"][0]["finalArtifactPath"]
    published = BillOfLadingDualCargoAnnotation.model_validate_json(
        artifact_path.read_bytes(), strict=True
    )

    assert published.normalLabel.documentPatch.freight.paymentArrangement == "collect"
    assert manifest["items"][0]["targetCorrections"] == [
        {
            "target_path": "/documentPatch/freight/paymentArrangement",
            "expected_value": "prepaid",
            "corrected_value": "collect",
        }
    ]
    freight_evidence = next(
        row
        for row in published.evidence
        if row.targetPath == "documentPatch.freight.paymentArrangement"
    )
    assert freight_evidence.rawOcrEvidence[0].rawValue == "FREIGHT COLLECT"


def test_adjudication_rejects_target_correction_when_source_guard_differs(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("BILL OF LADING\nB/L NO: HBL-001")
    candidate = build_compact_annotation(
        item,
        _compact_annotation({"billOfLadingNumber": "HBL-001"}),
        pdf_grouping_used=False,
    )
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "candidate.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(candidate.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "rejected-target-correction-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "rebuild": {"decision_notes": ["This correction must fail."]},
                    "target_corrections": [
                        {
                            "target_path": "/documentPatch/billOfLadingNumber",
                            "expected_value": "A-DIFFERENT-VALUE",
                            "corrected_value": "HBL-001",
                        }
                    ],
                    "adjudication_notes": ["This correction must fail."],
                }
            ],
        },
        strict=True,
    )

    with pytest.raises(AdjudicationError, match="expected value differs"):
        publish_adjudication_run(config)

    assert not (output_root / "runs" / config.run_id).exists()


def test_adjudication_rejects_a_non_day_first_correction_before_publication(
    tmp_path: Path,
) -> None:
    item = _work_item_with_text("SHIPPED ON BOARD\n01/04/2024")
    annotation = build_compact_annotation(
        item,
        _compact_annotation({"shippedOnBoardDate": date(2024, 4, 1)}),
        pdf_grouping_used=False,
    )
    work_item_path = tmp_path / "work-item.json"
    candidate_path = tmp_path / "candidate.json"
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    candidate_payload = canonical_json_bytes(annotation.model_dump(mode="json"))
    work_item_path.write_bytes(work_item_payload)
    candidate_path.write_bytes(candidate_payload)
    output_root = tmp_path / "output"
    output_root.mkdir()
    config = AdjudicationConfig.model_validate(
        {
            "schema_version": 1,
            "run_id": "rejected-month-first-adjudication-v1",
            "output_root": str(output_root),
            "items": [
                {
                    "document_id": item.source.documentId,
                    "work_item_path": str(work_item_path),
                    "work_item_sha256": sha256_bytes(work_item_payload),
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "date_corrections": [
                        {
                            "target_path": "documentPatch.shippedOnBoardDate",
                            "raw_value": "01/04/2024",
                            "normalized_value": date(2024, 1, 4),
                        },
                    ],
                    "adjudication_notes": ["This correction must fail."],
                },
            ],
        },
        strict=True,
    )

    with pytest.raises(AdjudicationError, match="differs from day-first policy"):
        publish_adjudication_run(config)

    assert not (output_root / "runs" / config.run_id).exists()


def test_day_first_adjudication_supports_two_digit_year() -> None:
    from document_ocr.labeling_agents.adjudication import _day_first_date

    assert _day_first_date("03.02.25") == date(2025, 2, 3)


def test_unambiguous_numeric_date_in_document_establishes_date_order() -> None:
    item = _work_item_with_text(
        "INVOICE DATE: 18.04.2025\nSHIPPED ON BOARD\n08.05.2025"
    )
    draft = _compact_annotation({"shippedOnBoardDate": date(2025, 5, 8)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.shippedOnBoardDate == date(2025, 5, 8)


def test_local_evidence_grounds_negotiability_from_original_surrender_terms() -> None:
    item = _work_item_with_text(
        "One of the MTDs must be surrendered, duly endorsed in exchange for the goods."
    )
    draft = _compact_annotation({"negotiability": "negotiable"})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.negotiability == "negotiable"


def test_non_negotiable_copy_status_does_not_override_positive_original_bill() -> None:
    raw = (
        "NON-NEGOTIABLE COPY\nBILL OF LADING\nNumber of Original B(s)/L\nTHREE(3)\n"
        "ONE of which being accomplished, the others to stand void."
    )

    with pytest.raises(DeterministicAnnotationError, match="copy-status stamp"):
        build_compact_annotation(
            _work_item_with_text(raw),
            _compact_annotation({"negotiability": "non_negotiable"}),
            pdf_grouping_used=False,
        )

    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"negotiability": "negotiable"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "negotiable"


def test_named_straight_consignee_supports_non_negotiable_despite_copy_stamp() -> None:
    raw = (
        "NON-NEGOTIABLE COPY\nBILL OF LADING\nCONSIGNEE\n"
        "ACME IMPORTS LIMITED\nNOTIFY PARTY\nSAME AS CONSIGNEE\n"
        "Number of Original B(s)/L\nTHREE(3)\n"
        "ONE of which being accomplished, the others to stand void."
    )

    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation(
            {
                "parties": {"consignee": {"name": "ACME IMPORTS LIMITED"}},
                "negotiability": "non_negotiable",
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"
    evidence = next(
        row for row in annotation.evidence if row.targetPath == "documentPatch.negotiability"
    )
    assert "ACME IMPORTS LIMITED" in evidence.rawOcrEvidence[0].rawValue
    assert "NON-NEGOTIABLE COPY" not in evidence.rawOcrEvidence[0].rawValue


def test_named_straight_consignee_supports_observed_insert_name_heading() -> None:
    raw = (
        "2. Consignee Insert Name Address and Phone/Fax\n"
        "ACME IMPORTS LIMITED\n"
        "3. Notify Party\nSAME AS CONSIGNEE"
    )

    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation(
            {
                "parties": {"consignee": {"name": "ACME IMPORTS LIMITED"}},
                "negotiability": "non_negotiable",
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


def test_order_consignee_under_insert_name_heading_is_not_straight() -> None:
    raw = (
        "2. Consignee Insert Name Address and Phone/Fax\n"
        "TO THE ORDER OF ACME BANK"
    )

    with pytest.raises(DeterministicAnnotationError):
        build_compact_annotation(
            _work_item_with_text(raw),
            _compact_annotation({"negotiability": "non_negotiable"}),
            pdf_grouping_used=False,
        )


@pytest.mark.parametrize(
    ("raw", "target"),
    [
        ("CONSIGNEE: TO THE ORDER OF ARAB AFRICAN BANK", "negotiable"),
        (
            "On presentation of this document (duly endorsed) to Carrier, delivery "
            "will be made.",
            "negotiable",
        ),
        (
            "On presentation of one original of this bill of Lading (duly endorsed) "
            "to the Carrier, delivery will be made.",
            "negotiable",
        ),
        (
            "One of the original Bills of Lading must be surrendered and endorsed "
            "in exchange for the goods.",
            "negotiable",
        ),
        (
            "One original Bill of Lading, duly endorsed, must be surrendered by the "
            "Merchant to the Carrier in exchange for the Goods.",
            "negotiable",
        ),
        (
            "One of This Bill of Lading Duly Endorsed Must be Surrendered to the Carrier.",
            "negotiable",
        ),
        (
            "The goods are there to be delivered unto order or assigns. THREE (3) originals.",
            "negotiable",
        ),
        (
            "Three original Bills of Lading have been signed, one of which being "
            "accomplished the other(s) to be void.",
            "negotiable",
        ),
        ("EXPRESS BILL OF LADING", "non_negotiable"),
    ],
)
def test_local_evidence_grounds_observed_negotiability_variants(
    raw: str, target: str
) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"negotiability": target}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == target


def test_local_evidence_grounds_obliged_surrender_word_order() -> None:
    raw = (
        "The Merchant is obliged to surrender one original bill of lading, "
        "duly endorsed, in exchange for the Goods."
    )

    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"negotiability": "negotiable"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "negotiable"


def test_local_evidence_grounds_source_ordered_description_across_pages() -> None:
    item = _work_item_with_pages(
        "DESCRIPTION OF GOODS\nASSY OPEN CELL\nINVOICE: 800016691",
        "DESC: TELEVISIONS-VD PARTS\nHS CODE: 8524911000",
    )
    draft = _compact_annotation(
        {
            "forwardingAndExportReferences": ("800016691",),
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "ASSY OPEN CELL; TELEVISIONS-VD PARTS",
                    "hsCodes": ("8524911000",),
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    description = next(
        row for row in annotation.evidence if row.targetPath.endswith(".description")
    )
    assert tuple(row.pageNumber for row in description.rawOcrEvidence) == (1, 2)
    assert "ASSY OPEN CELL" in description.rawOcrEvidence[0].rawValue
    assert "TELEVISIONS-VD PARTS" in description.rawOcrEvidence[-1].rawValue


def test_explicit_invoice_and_all_hs_block_values_are_required_once() -> None:
    item = _work_item_with_text(
        "INVOICE: 800016691\n\nHS CODE\n281129 392119 392310\n392410 392490\n\nDECLARATION"
    )
    incomplete = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "hsCodes": ("281129", "392119")},)}
    )

    with pytest.raises(DeterministicAnnotationError) as caught:
        build_compact_annotation(item, incomplete, pdf_grouping_used=False)

    message = str(caught.value)
    assert "800016691" in message
    assert "392310" in message
    assert "392410" in message
    assert "392490" in message

    complete = _compact_annotation(
        {
            "forwardingAndExportReferences": ("800016691",),
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "hsCodes": ("281129", "392119", "392310", "392410", "392490"),
                },
            ),
        }
    )
    annotation = build_compact_annotation(item, complete, pdf_grouping_used=False)
    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == ("800016691",)


def test_explicit_package_total_in_english_words_grounds_integer_quantity() -> None:
    item = _work_item_with_text(
        "BIRCH PLYWOOD\n"
        "Total Number Of Packages(in words): Four Hundred and Fifty One Only"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": ({"groupId": "g1", "description": "BIRCH PLYWOOD"},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 451,
                    "typeDescription": "PACKAGES",
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    package = annotation.relationExplicitLabel.documentPatch.cargoPackages[0]
    quantity_evidence = next(
        row for row in annotation.evidence if row.targetPath.endswith(".quantity")
    )
    assert package.quantity == 451
    assert quantity_evidence.rawOcrEvidence[0].rawValue == "Four Hundred and Fifty One"


def test_explicit_container_volume_rows_require_reconciled_package_allocations() -> None:
    item = _work_item_with_text(
        "CMAU1989869 SEAL:L9643682\n"
        "GW:14275.584KG NW:13267.584KG\n"
        "CBM:16.080M3 Volumes:24\n"
        "CMAU2547049 SEAL:L9643674\n"
        "GW:13059.600KG NW:12135.600KG\n"
        "CBM:14.628M3 Volumes:22\n"
        "2 CONTAINERS WITH 46 PALLETS"
    )
    shared = {
        "containers": (
            {"containerNumber": "CMAU1989869", "sealNumbers": ("L9643682",)},
            {"containerNumber": "CMAU2547049", "sealNumbers": ("L9643674",)},
        ),
        "cargoGroups": ({"groupId": "g1"},),
        "cargoPackages": (
            {
                "packageId": "p1",
                "groupId": "g1",
                "quantity": 46,
                "typeDescription": "PALLETS",
            },
        ),
    }

    with pytest.raises(DeterministicAnnotationError, match="Volumes rows exactly reconcile"):
        build_compact_annotation(
            item,
            _compact_annotation(shared),
            pdf_grouping_used=False,
        )

    complete = _compact_annotation(
        {
            **shared,
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageId": "p1",
                    "allocations": (
                        {"containerNumber": "CMAU1989869", "packageQuantity": 24},
                        {"containerNumber": "CMAU2547049", "packageQuantity": 22},
                    ),
                },
            ),
        }
    )
    annotation = build_compact_annotation(item, complete, pdf_grouping_used=False)

    allocations = annotation.relationExplicitLabel.documentPatch.cargoAllocationGroups
    assert allocations is not None
    assert tuple(row.packageQuantity for row in allocations[0].allocations) == (24, 22)


def test_repeated_allocation_quantities_are_grounded_by_container_occurrence() -> None:
    containers = (
        "CAAU8913108",
        "CAAU9218501",
        "MRSU7178691",
        "MRKU4947233",
        "TCNU2807240",
    )
    item = _work_item_with_text(
        "TOTAL 2560 PACKAGE\n"
        + "\n".join(
            f"{container} SEAL{index} 40 DRY 9'6 512 PACKAGE"
            for index, container in enumerate(containers, start=1)
        )
    )
    draft = _compact_annotation(
        {
            "containers": tuple(
                {"containerNumber": container} for container in containers
            ),
            "cargoGroups": ({"groupId": "g1"},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 2560,
                    "typeDescription": "PACKAGE",
                },
            ),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageId": "p1",
                    "allocations": tuple(
                        {"containerNumber": container, "packageQuantity": 512}
                        for container in containers
                    ),
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    allocation_evidence = tuple(
        row
        for row in annotation.evidence
        if row.targetPath.endswith(".packageQuantity")
    )
    assert len(allocation_evidence) == len(containers)
    for container, evidence in zip(containers, allocation_evidence, strict=True):
        assert evidence.rawOcrEvidence[0].rawValue == "512"
        assert container in evidence.rawOcrEvidence[0].ocrExcerpt
    relation_evidence = annotation.relationEvidence[0].rawOcrEvidence
    assert sum(row.rawValue == "512" for row in relation_evidence) == len(containers)


def test_relation_evidence_deduplicates_one_scalar_used_by_both_views() -> None:
    item = _work_item_with_text(
        "UACU5074791 1 INTERMEDIATE BULK CONTAINERS"
    )
    draft = _compact_annotation(
        {
            "containers": ({"containerNumber": "UACU5074791"},),
            "cargoGroups": ({"groupId": "g1"},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 1,
                    "typeDescription": "INTERMEDIATE BULK CONTAINERS",
                },
            ),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageId": "p1",
                    "allocations": (
                        {"containerNumber": "UACU5074791", "packageQuantity": 1},
                    ),
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    relation_evidence = annotation.relationEvidence[0].rawOcrEvidence
    assert sum(row.rawValue == "1" for row in relation_evidence) == 1


def test_explicit_valid_container_identifiers_are_complete_or_fail_closed() -> None:
    item = _work_item_with_text("CAAU8913108\nCAAU9218501")
    incomplete = _compact_annotation(
        {"containers": ({"containerNumber": "CAAU8913108"},)}
    )

    with pytest.raises(
        DeterministicAnnotationError,
        match=r"valid printed ISO 6346 container identifier.*CAAU9218501",
    ):
        build_compact_annotation(item, incomplete, pdf_grouping_used=False)

    complete = _compact_annotation(
        {
            "containers": (
                {"containerNumber": "CAAU8913108"},
                {"containerNumber": "CAAU9218501"},
            )
        }
    )
    annotation = build_compact_annotation(item, complete, pdf_grouping_used=False)

    assert tuple(
        row.containerNumber for row in annotation.normalLabel.documentPatch.containers
    ) == ("CAAU8913108", "CAAU9218501")


def test_invalid_container_check_digit_does_not_create_a_completeness_requirement() -> None:
    item = _work_item_with_text("CAAU8913108\nCAAU9218502")
    draft = _compact_annotation(
        {"containers": ({"containerNumber": "CAAU8913108"},)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert tuple(
        row.containerNumber for row in annotation.normalLabel.documentPatch.containers
    ) == ("CAAU8913108",)


def test_hsn_heading_is_valid_hs_context() -> None:
    item = _work_item_with_text("HSN: 25151210")
    draft = _compact_annotation({"cargoGroups": ({"groupId": "g1", "hsCodes": ("25151210",)},)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("25151210",)


def test_ncm_heading_is_valid_hs_context() -> None:
    item = _work_item_with_text("NCM 21061000")
    draft = _compact_annotation({"cargoGroups": ({"groupId": "g1", "hsCodes": ("21061000",)},)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("21061000",)


def test_compact_projection_removes_only_dangling_address_locality_separator() -> None:
    item = _work_item_with_text(
        "CONSIGNEE\nOCEAN FOODS\nNEW BORG EL ARAB CITY - 3RD INDUSTRIAL ZONE - ALEXANDRIA"
    )
    draft = _compact_annotation(
        {
            "parties": {
                "consignee": {
                    "name": "OCEAN FOODS",
                    "address": "NEW BORG EL ARAB CITY - 3RD INDUSTRIAL ZONE -",
                    "city": "ALEXANDRIA",
                }
            }
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.relationExplicitLabel.documentPatch.parties is not None
    consignee = annotation.relationExplicitLabel.documentPatch.parties.consignee
    assert consignee is not None
    assert consignee.address == "NEW BORG EL ARAB CITY - 3RD INDUSTRIAL ZONE"


def test_compact_annotation_rejects_regulatory_metadata_from_reference_target() -> None:
    item = _work_item_with_text("B/L NO: HBL-001\nACID NUMBER: 5939523832023090010")
    draft = _compact_annotation(
        {
            "billOfLadingNumber": "HBL-001",
            "forwardingAndExportReferences": ("5939523832023090010",),
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="excluded tax/regulatory"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


@pytest.mark.parametrize("placeholder", ["NO REF", "N/A", "NONE", "NIL"])
def test_compact_annotation_rejects_empty_export_reference_placeholder(
    placeholder: str,
) -> None:
    item = _work_item_with_text(f"EXPORT REFERENCES\n{placeholder}")
    draft = _compact_annotation({"forwardingAndExportReferences": (placeholder,)})

    with pytest.raises(DeterministicAnnotationError, match="empty placeholder"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


@pytest.mark.parametrize(
    ("heading", "target"),
    [
        ("EXPORT REFERENCES", "39A/2024"),
        ("AES", "X20240402251113"),
        ("CAED", "PC8656202304042500630"),
        ("ED NO", "332501006596"),
        ("SB NO", "8068650"),
    ],
)
def test_compact_annotation_accepts_explicit_export_reference_headings(
    heading: str, target: str
) -> None:
    item = _work_item_with_text(f"{heading}: {target}")
    draft = _compact_annotation({"forwardingAndExportReferences": (target,)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (target,)


def test_compact_annotation_accepts_ref_exp_export_reference_heading() -> None:
    item = _work_item_with_text("REF.EXP. 001/24")
    draft = _compact_annotation({"forwardingAndExportReferences": ("001/24",)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == ("001/24",)


def test_compact_annotation_rejects_reference_label_inside_target_value() -> None:
    item = _work_item_with_text("REF.EXP. 001/24")
    draft = _compact_annotation(
        {"forwardingAndExportReferences": ("REF.EXP. 001/24",)}
    )

    with pytest.raises(DeterministicAnnotationError, match="value only, not its label"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


def test_invoice_reference_accepts_a_separate_inv_number_heading_line() -> None:
    item = _work_item_with_text("INV NO.\nB-240411-SEEG-A")
    draft = _compact_annotation({"forwardingAndExportReferences": ("B-240411-SEEG-A",)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "B-240411-SEEG-A",
    )


def test_plural_proforma_invoice_heading_grounds_reference_on_next_line() -> None:
    item = _work_item_with_text(
        "ACCORDING TO PROFORMA INVOICES NO.\n.834349(1) DD. 10.JUL.2023"
    )
    draft = _compact_annotation({"forwardingAndExportReferences": (".834349(1)",)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        ".834349(1)",
    )


def test_bill_number_starting_pir_is_not_invented_as_pi_reference() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("BILL OF LADING NUMBER\nPIR0235694"),
        _compact_annotation({"billOfLadingNumber": "PIR0235694"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences is None


def test_invoice_reference_is_not_rejected_by_neighboring_acid_metadata() -> None:
    item = _work_item_with_text("ACID: 1002704682024081371\nINVOICE NO : GLH2024000003785-3786")
    draft = _compact_annotation({"forwardingAndExportReferences": ("GLH2024000003785-3786",)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "GLH2024000003785-3786",
    )


@pytest.mark.parametrize(
    ("raw", "reference"),
    [
        ("ITN X20240205979223", "X20240205979223"),
        ("P/I NO. 863089 (4)", "863089"),
        ("INVOICE NUMBER: 240003 - ORDER\nNUMBER: 1689", "240003"),
        ("INVOICE NO. & DATE : HG0240301 & MAR. 18,2024", "HG0240301"),
        ("PROFORMA: 500359", "500359"),
        ("DU-E 24BR001164384-4", "24BR001164384-4"),
        ("DUE: 23BR001666144-1", "23BR001666144-1"),
        ("DAE: 028-2023-40-01245411", "028-2023-40-01245411"),
        ("F/Agent Name & Ref.\n108288", "108288"),
        (
            "FORWARDING AGENT:\nTTS WORLDWIDE\n265 POST AVENUE\nREF #: 100246627",
            "100246627",
        ),
    ],
)
def test_observed_reference_labels_ground_value_only(raw: str, reference: str) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"forwardingAndExportReferences": (reference,)}),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.forwardingAndExportReferences == (
        reference,
    )


def test_domestic_routing_export_instructions_ground_value_only() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "DOMESTIC ROUTING/EXPORT INSTRUCTIONS (9)\nX20231227986364"
        ),
        _compact_annotation(
            {"forwardingAndExportReferences": ("X20231227986364",)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "X20231227986364",
    )


def test_domestic_routing_without_export_context_does_not_ground_reference() -> None:
    with pytest.raises(DeterministicAnnotationError, match="lacks a qualifying heading"):
        build_compact_annotation(
            _work_item_with_text("DOMESTIC ROUTING\nX20231227986364"),
            _compact_annotation(
                {"forwardingAndExportReferences": ("X20231227986364",)}
            ),
            pdf_grouping_used=False,
        )


def test_forwarding_agent_generic_ref_survives_visual_blank_line() -> None:
    raw = (
        "SHIPPER: ETG COMMODITIES INC.\nREF #: SC_ETG015392\n\n"
        "FORWARDING AGENT: RAY-MONT LOGISTICS\nEMAIL: DAN.MAO@RAY-MONT.COM\n\n"
        "REF #: 141053-1\nCONTACT: DAN MAO\n\nCONSIGNEE\nTO ORDER"
    )
    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"forwardingAndExportReferences": ("141053-1",)}),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.forwardingAndExportReferences == (
        "141053-1",
    )
    with pytest.raises(DeterministicAnnotationError, match="lacks a qualifying heading"):
        build_compact_annotation(
            _work_item_with_text(raw),
            _compact_annotation({"forwardingAndExportReferences": ("SC_ETG015392",)}),
            pdf_grouping_used=False,
        )


def test_spaced_hs_heading_is_valid_hs_context() -> None:
    item = _work_item_with_text("H. S. CODE 79011100")
    draft = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "hsCodes": ("79011100",)},)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("79011100",)


def test_customs_cde_ocr_heading_is_valid_hs_context() -> None:
    item = _work_item_with_text("CUSTOMS CDE 2401108590")
    draft = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "hsCodes": ("2401108590",)},)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("2401108590",)


def test_ocr_glued_package_plural_and_hs_heading_is_valid_context() -> None:
    item = _work_item_with_text("DRUMSHS:3302.90")
    draft = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "hsCodes": ("330290",)},)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("330290",)


def test_hyphenated_hs_heading_is_valid_hs_context() -> None:
    item = _work_item_with_text("H-S CODE :- 850211")
    draft = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "hsCodes": ("850211",)},)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("850211",)


def test_kilos_heading_grounds_kilogram_unit() -> None:
    item = _work_item_with_text("Gross Weight in kilos\n243480.000")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 243480.0, "unit": "kilogram"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    gross = annotation.normalLabel.documentPatch.goodsItems[0].grossWeight
    assert gross is not None
    assert gross.unit == "kilogram"


def test_iso_measurement_codes_ground_mass_and_volume_units() -> None:
    item = _work_item_with_text(
        "CARGO GROSS WEIGHT: 26000.00 KGM\nMEASUREMENT: 17 MTQ"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 26000.0, "unit": "kilogram"},
                    "volume": {"value": 17.0, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    cargo = annotation.normalLabel.documentPatch.goodsItems[0]
    assert cargo.grossWeight is not None and cargo.grossWeight.unit == "kilogram"
    assert cargo.volume is not None and cargo.volume.unit == "cubic_metre"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "NO. OF CONTAINERS OR PKGS.\nDESCRIPTION OF PACKAGES AND GOODS\n"
            "CARGO GROSS WEIGHT\nMEASUREMENT\n\nTotal:\n15 PACKAGE\n53,334 KG\n15 M3",
            53334.0,
        ),
        (
            "Container ID\nQuantity/Number & kind of packages/pieces/Other marks & numbers\n"
            "Gross cargo weight\nGross cargo volume\n\nNumber of packages\n25\n"
            "15,700 KGM\n17 MTQ",
            15700.0,
        ),
    ],
)
def test_flattened_mass_table_uses_unit_qualified_grouping_style(
    raw: str, expected: float
) -> None:
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "grossWeight": {"value": expected, "unit": "kilogram"},
                },
            )
        }
    )

    annotation = build_compact_annotation(
        _work_item_with_text(raw), draft, pdf_grouping_used=False
    )

    gross = annotation.normalLabel.documentPatch.goodsItems[0].grossWeight
    assert gross is not None and gross.value == expected


def test_export_number_and_pi_number_are_explicit_value_only_references() -> None:
    item = _work_item_with_text("EXPORT NUMBER.: 134-81-42369\nPI NO.: P8101")
    draft = _compact_annotation(
        {"forwardingAndExportReferences": ("134-81-42369", "P8101")}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "134-81-42369",
        "P8101",
    )


def test_packing_construction_after_flattened_acid_field_is_not_metadata() -> None:
    raw = (
        "ACID NO.: 1000713762025020684 TOTAL CARTONS: 495 "
        "TOBACCO ARE PACKED IN CARD BOARD CASES FINAL DESTINATION: CAIRO"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "additionalInformation": ("TOBACCO ARE PACKED IN CARD BOARD CASES",),
                },
            )
        }
    )

    annotation = build_compact_annotation(
        _work_item_with_text(raw), draft, pdf_grouping_used=False
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].additionalInformation == (
        "TOBACCO ARE PACKED IN CARD BOARD CASES",
    )


def test_punctuated_hs_code_completeness_preserves_all_digits() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("HS CODE: 3903.30-0000"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("3903300000",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "3903300000",
    )


def test_same_line_acid_does_not_contaminate_preceding_invoice_or_marks() -> None:
    item = _work_item_with_text(
        "VIN Number(s): WP0ZZZY1ZRSA31126 COMM. NO. 091989 "
        "ENGINE NO : EBG131298 EBF133928 AS PER INVOICE NUMBER 1338025136 "
        "HS CODE 8703 ACID No:7059976422023100139"
    )
    draft = _compact_annotation(
        {
            "forwardingAndExportReferences": ("1338025136",),
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "marksAndNumbers": (
                        "WP0ZZZY1ZRSA31126",
                        "091989",
                        "EBG131298",
                        "EBF133928",
                    ),
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    patch = annotation.relationExplicitLabel.documentPatch
    assert patch.forwardingAndExportReferences == ("1338025136",)
    assert patch.cargoGroups is not None
    assert patch.cargoGroups[0].marksAndNumbers == (
        "WP0ZZZY1ZRSA31126",
        "091989",
        "EBG131298",
        "EBF133928",
    )


def test_metadata_heading_still_rejects_its_bare_value_as_cargo_mark() -> None:
    item = _work_item_with_text("ACID No: 7059976422023100139")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "marksAndNumbers": ("7059976422023100139",),
                },
            )
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="excluded tax/regulatory"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


def test_destination_payable_yes_is_grounded_as_collect_without_payment_place() -> None:
    item = _work_item_with_text("Freight and Charges payable at destination:\nYes")
    draft = _compact_annotation(
        {"freight": {"paymentArrangement": "collect", "paymentPlace": None}}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    freight = annotation.relationExplicitLabel.documentPatch.freight
    assert freight is not None
    assert freight.paymentArrangement == "collect"
    assert freight.paymentPlace is None


def test_destination_payable_yes_cannot_be_silently_omitted() -> None:
    item = _work_item_with_text(
        "B/L NO: HBL-001\nFreight and Charges payable at destination:\nYes"
    )

    with pytest.raises(
        DeterministicAnnotationError,
        match=r"requires freight\.paymentArrangement=collect",
    ):
        build_compact_annotation(
            item,
            _compact_annotation({"billOfLadingNumber": "HBL-001"}),
            pdf_grouping_used=False,
        )


def test_destination_payable_without_yes_is_grounded_as_collect() -> None:
    item = _work_item_with_text("FREIGHT PAYABLE AT DESTINATION")
    draft = _compact_annotation(
        {"freight": {"paymentArrangement": "collect", "paymentPlace": None}}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    freight = annotation.relationExplicitLabel.documentPatch.freight
    assert freight is not None
    assert freight.paymentArrangement == "collect"
    assert freight.paymentPlace is None


def test_destination_payable_in_generic_contract_prose_is_not_a_completed_field() -> None:
    item = _work_item_with_text(
        "B/L NO: B/L-001\nFreight and charges\nAs agreed payable at destination\n"
        "prepayable freight on dispatch and freight payable at destination on arrival"
    )

    annotation = build_compact_annotation(
        item,
        _compact_annotation({"billOfLadingNumber": "B/L-001"}),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.freight is None


def test_explicit_face_field_prepaid_takes_precedence_over_generic_destination_text() -> None:
    item = _work_item_with_text(
        "Freight payable at PREPAID\n"
        "Freight and charges\nAs agreed payable at destination"
    )
    draft = _compact_annotation(
        {"freight": {"paymentArrangement": "prepaid", "paymentPlace": None}}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    freight = annotation.relationExplicitLabel.documentPatch.freight
    assert freight is not None
    assert freight.paymentArrangement == "prepaid"
    assert freight.paymentPlace is None


def test_final_freight_prepaid_status_resolves_conflicting_destination_phrase() -> None:
    item = _work_item_with_text(
        "FREIGHT TO BE PAID AT\nLIMA\n"
        "FREIGHT PAYABLE AT DESTINATION\nFREIGHT PREPAID"
    )
    annotation = build_compact_annotation(
        item,
        _compact_annotation(
            {
                "freight": {
                    "paymentArrangement": "prepaid",
                    "paymentPlace": {"name": "LIMA", "country": None},
                }
            }
        ),
        pdf_grouping_used=False,
    )

    freight = annotation.relationExplicitLabel.documentPatch.freight
    assert freight is not None
    assert freight.paymentArrangement == "prepaid"
    assert freight.paymentPlace is not None and freight.paymentPlace.name == "LIMA"


def test_explicit_face_field_prepaid_cannot_be_labeled_collect() -> None:
    item = _work_item_with_text(
        "Freight payable at PREPAID\n"
        "Freight and charges\nAs agreed payable at destination"
    )
    draft = _compact_annotation(
        {"freight": {"paymentArrangement": "collect", "paymentPlace": None}}
    )

    with pytest.raises(
        DeterministicAnnotationError,
        match=r"requires freight\.paymentArrangement=prepaid",
    ):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


def test_unselected_prepaid_collect_options_do_not_override_explicit_freight_collect() -> None:
    item = _work_item_with_text(
        "Freight Payable at:\nPREPAID COLLECT\n\nFREIGHT COLLECT ORIGINAL"
    )
    draft = _compact_annotation(
        {"freight": {"paymentArrangement": "collect", "paymentPlace": None}}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    freight = annotation.relationExplicitLabel.documentPatch.freight
    assert freight is not None
    assert freight.paymentArrangement == "collect"


def test_express_release_no_originals_requires_and_grounds_non_negotiable() -> None:
    item = _work_item_with_text(
        "B/L NO: HBL-001\nSHIPPED ON BOARD\n\n"
        "EXPRESS RELEASE - NO ORIGINALS ISSUED\nFREIGHT COLLECT"
    )
    base_patch = {"billOfLadingNumber": "HBL-001"}

    with pytest.raises(
        DeterministicAnnotationError,
        match="requires negotiability=non_negotiable",
    ):
        build_compact_annotation(item, _compact_annotation(base_patch), pdf_grouping_used=False)

    annotation = build_compact_annotation(
        item,
        _compact_annotation({**base_patch, "negotiability": "non_negotiable"}),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.negotiability == "non_negotiable"
    evidence = next(
        row for row in annotation.evidence if row.targetPath == "documentPatch.negotiability"
    )
    assert evidence.rawOcrEvidence[0].rawValue == (
        "EXPRESS RELEASE - NO ORIGINALS ISSUED"
    )


def test_explicit_total_packages_loaded_into_container_requires_allocation() -> None:
    item = _work_item_with_text(
        "B/L NO: HBL-001\n6 PALLET(S) STC: CLEANER\n"
        "TOTAL PACKAGES: 6, TOTAL GROSS WEIGHT: 1676.990 KGS\n"
        "LOADED INTO CONTAINER(S): TCNU2181330 WITH SEAL: 1306219"
    )
    base_patch = {
        "billOfLadingNumber": "HBL-001",
        "containers": (
            {"containerNumber": "TCNU2181330", "sealNumbers": ("1306219",)},
        ),
        "cargoGroups": ({"groupId": "g1", "description": "CLEANER"},),
        "cargoPackages": (
            {
                "packageId": "p1",
                "groupId": "g1",
                "quantity": 6,
                "typeDescription": "PALLET(S)",
            },
        ),
    }
    membership_only = _compact_annotation(
        {
            **base_patch,
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "container_membership_only",
                    "allocations": ({"containerNumber": "TCNU2181330"},),
                },
            ),
        }
    )

    with pytest.raises(
        DeterministicAnnotationError,
        match="requires a single_package_level allocation",
    ):
        build_compact_annotation(item, membership_only, pdf_grouping_used=False)

    linked = _compact_annotation(
        {
            **base_patch,
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageId": "p1",
                    "allocations": (
                        {
                            "containerNumber": "TCNU2181330",
                            "packageQuantity": 6,
                        },
                    ),
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, linked, pdf_grouping_used=False)
    group = annotation.relationExplicitLabel.documentPatch.cargoAllocationGroups
    assert group is not None
    assert group[0].coverage == "single_package_level"
    assert group[0].allocations[0].packageQuantity == 6


def test_partitioned_pallet_ranges_ground_outer_packages_and_allocations() -> None:
    item = _work_item_with_text(
        "1 Container Said to Contain 6 PALLET\n"
        "Pallet No. 1 - 4:\n"
        "132 Tinplate Containers PRODUCT A\n"
        "Pallet No. 5 - 6:\n"
        "8 Drums PRODUCT B\n"
        "UACU5074791 40 DRY 6 PALLET"
    )
    draft = _compact_annotation(
        {
            "containers": (
                {"containerNumber": "UACU5074791", "typeDescription": "40 DRY"},
            ),
            "cargoGroups": (
                {"groupId": "g1", "description": "PRODUCT A"},
                {"groupId": "g2", "description": "PRODUCT B"},
            ),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 4,
                    "typeDescription": "PALLET",
                },
                {
                    "packageId": "p2",
                    "groupId": "g1",
                    "quantity": 132,
                    "typeDescription": "Tinplate Containers",
                },
                {
                    "packageId": "p3",
                    "groupId": "g2",
                    "quantity": 2,
                    "typeDescription": "PALLET",
                },
                {
                    "packageId": "p4",
                    "groupId": "g2",
                    "quantity": 8,
                    "typeDescription": "Drums",
                },
            ),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageId": "p1",
                    "allocations": (
                        {"containerNumber": "UACU5074791", "packageQuantity": 4},
                    ),
                },
                {
                    "groupId": "g2",
                    "coverage": "single_package_level",
                    "packageId": "p3",
                    "allocations": (
                        {"containerNumber": "UACU5074791", "packageQuantity": 2},
                    ),
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)
    evidence_by_path = {row.targetPath: row for row in annotation.evidence}

    expected_ranges = {
        "documentPatch.goodsItems[0].packages[0].quantity": "Pallet No. 1 - 4:",
        "documentPatch.goodsItems[0].containerAllocations[0].packageQuantity": (
            "Pallet No. 1 - 4:"
        ),
        "documentPatch.goodsItems[1].packages[0].quantity": "Pallet No. 5 - 6:",
        "documentPatch.goodsItems[1].containerAllocations[0].packageQuantity": (
            "Pallet No. 5 - 6:"
        ),
    }
    for path, raw_range in expected_ranges.items():
        evidence = evidence_by_path[path]
        assert evidence.rawOcrEvidence[0].rawValue == raw_range
        assert evidence.normalizationRule is not None
        assert "inclusive outer-pallet" in evidence.normalizationRule


def test_partitioned_pallet_ranges_reject_one_aggregate_package_fact() -> None:
    item = _work_item_with_text(
        "1 Container Said to Contain 6 PALLET\n"
        "Pallet No. 1 - 4:\n"
        "132 Tinplate Containers PRODUCT A\n"
        "Pallet No. 5 - 6:\n"
        "8 Drums PRODUCT B\n"
        "UACU5074791 40 DRY 6 PALLET"
    )
    collapsed = _compact_annotation(
        {
            "containers": ({"containerNumber": "UACU5074791"},),
            "cargoGroups": ({"groupId": "g1", "description": "PRODUCT A PRODUCT B"},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 6,
                    "typeDescription": "PALLET",
                },
            ),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageId": "p1",
                    "allocations": (
                        {"containerNumber": "UACU5074791", "packageQuantity": 6},
                    ),
                },
            ),
        }
    )

    with pytest.raises(
        DeterministicAnnotationError,
        match="partitioned pallet-range quantity does not match",
    ):
        build_compact_annotation(item, collapsed, pdf_grouping_used=False)


def test_single_cargo_group_requires_explicit_made_in_origin() -> None:
    item = _work_item_with_text(
        "MARKS & NOS.\nJWELL MACHINERY MADE IN CHINA\n"
        "DESCRIPTION OF GOODS\nPET SHEET EXTRUSION LINE"
    )
    missing_origin = _compact_annotation(
        {
            "cargoGroups": (
                {"groupId": "g1", "description": "PET SHEET EXTRUSION LINE"},
            )
        }
    )

    with pytest.raises(
        DeterministicAnnotationError,
        match=r"requires cargoGroups\[0\]\.origin",
    ):
        build_compact_annotation(item, missing_origin, pdf_grouping_used=False)

    annotation = build_compact_annotation(
        item,
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "description": "PET SHEET EXTRUSION LINE",
                        "origin": {"name": "CHINA", "identifier": None},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    groups = annotation.relationExplicitLabel.documentPatch.cargoGroups
    assert groups is not None
    assert groups[0].origin is not None
    assert groups[0].origin.name == "CHINA"


@pytest.mark.parametrize(
    ("raw", "reference"),
    [
        ("* INV NO.AMIMED-230601-01", "AMIMED-230601-01"),
        ("INVOICE NO1/210723/1", "1/210723/1"),
        ("INV 15135", "15135"),
        ("INVOICE FAC-DFC2307-0430", "FAC-DFC2307-0430"),
    ],
)
def test_explicit_invoice_reference_parser_and_grounder_share_observed_prefixes(
    raw: str, reference: str
) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"forwardingAndExportReferences": (reference,)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        reference,
    )


def test_invoice_reference_allows_ampersand_in_printed_identifier() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("INVOICE NO. S&P-23224"),
        _compact_annotation(
            {"forwardingAndExportReferences": ("S&P-23224",)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "S&P-23224",
    )


def test_mass_style_does_not_match_dot_grouped_suffix_inside_mixed_number() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "FERRO MOLYBDENUM\nNET WEIGHT: 60,000KG\n"
            "GROSS WEIGHT: 60,960.000 KGS"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "description": "FERRO MOLYBDENUM",
                        "netWeight": {"value": 60000, "unit": "kilogram"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    mass = annotation.normalLabel.documentPatch.goodsItems[0].netWeight
    assert mass is not None and mass.value == 60000


def test_bare_invoice_heading_without_a_numeric_value_is_not_a_reference() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("B/L NO: HBL-001\nINVOICE DESCRIPTION\nSTEEL PARTS"),
        _compact_annotation({"billOfLadingNumber": "HBL-001"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences is None


def test_deterministic_annotation_rejects_standalone_fax_as_phone() -> None:
    item = _work_item_with_text(
        "DELIVERY AGENT\nACME LOGISTICS\nFAX: +20 2 1234567"
    )
    draft = _compact_annotation(
        {
            "parties": {
                "deliveryAgent": {
                    "name": "ACME LOGISTICS",
                    "contactDetails": {"phoneNumbers": ("+20 2 1234567",)},
                }
            }
        }
    )

    with pytest.raises(
        DeterministicAnnotationError, match="standalone FAX value cannot populate phoneNumbers"
    ):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


@pytest.mark.parametrize("heading", ("TEL", "TEL/FAX", "TEL:/FAX", "TEL & FAX"))
def test_deterministic_annotation_accepts_phone_or_combined_phone_fax(
    heading: str,
) -> None:
    number = "+20 2 1234567"
    item = _work_item_with_text(
        f"DELIVERY AGENT\nACME LOGISTICS\n{heading}: {number}\nFAX: {number}"
    )
    draft = _compact_annotation(
        {
            "parties": {
                "deliveryAgent": {
                    "name": "ACME LOGISTICS",
                    "contactDetails": {"phoneNumbers": (number,)},
                }
            }
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    evidence = next(
        row
        for row in annotation.evidence
        if row.targetPath.endswith("contactDetails.phoneNumbers[0]")
    )
    assert heading in evidence.rawOcrEvidence[0].ocrExcerpt
    assert not evidence.rawOcrEvidence[0].ocrExcerpt.lstrip().startswith("FAX:")


@pytest.mark.parametrize(
    ("page_text", "patch", "message"),
    [
        (
            "SHIPPER\nACME\nTELANGANA INDIA",
            {
                "parties": {
                    "shipper": {
                        "name": "ACME",
                        "contactDetails": {"phoneNumbers": ("TELANGANA INDIA",)},
                    }
                }
            },
            "phone number contains no printed digit",
        ),
        (
            "SIGNED\nTurkon America Inc. as agents for Turkon Container "
            "Transportation & Shipping Inc. The Carrier",
            {
                "parties": {
                    "carrier": {
                        "name": (
                            "Turkon America Inc. as agents for Turkon Container "
                            "Transportation & Shipping Inc. The Carrier"
                        )
                    }
                }
            },
            "signing-role prose",
        ),
        (
            "as agent for and on behalf of\n205439",
            {"parties": {"carrier": {"name": "205439"}}},
            "party name contains no alphabetic character",
        ),
        (
            "Vessel\nMELCHIOR SCHULTE-0US66E1TK",
            {"transport": {"vesselName": "MELCHIOR SCHULTE-0US66E1TK"}},
            "joined voyage suffix",
        ),
        (
            "Description of Goods\nWHEY POWDER PACKED WITH POLYETHYLENE INNER BAG",
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "description": "WHEY POWDER PACKED WITH POLYETHYLENE INNER BAG",
                    },
                )
            },
            "packing-construction text",
        ),
        (
            "1 G I DRUM OF 180 KGS NET WEIGHT Net Wt: 180.000 KGS",
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "netWeight": {"value": 180, "unit": "kilogram"},
                    },
                )
            },
            "per-package net mass",
        ),
    ],
)
def test_compact_annotation_rejects_unlearnable_semantic_conflations(
    page_text: str, patch: dict[str, Any], message: str
) -> None:
    item = _work_item_with_text(page_text)
    draft = _compact_annotation(patch)

    with pytest.raises(DeterministicAnnotationError, match=message):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


def test_compact_review_derives_pass_fail_and_checks_from_grounded_findings() -> None:
    passing = CompactReviewDraft.model_validate(
        {"decision": "review", "findings": (), "summary": "No blocking defect found."},
        strict=True,
    )
    failing = CompactReviewDraft.model_validate(
        {
            "decision": "review",
            "findings": (
                {
                    "severity": "blocking",
                    "category": "missing_field",
                    "message": "The explicit B/L number is missing.",
                    "targetPaths": ("documentPatch.billOfLadingNumber",),
                    "rawOcrEvidence": (
                        {
                            "pageNumber": 1,
                            "rawValue": "HBL-001",
                            "ocrExcerpt": "B/L NO: HBL-001",
                        },
                    ),
                    "imageUse": "not_used",
                },
            ),
            "summary": "One blocking omission found.",
        },
        strict=True,
    )

    assert _resolve_compact_review(passing).result == "pass"
    resolved = _resolve_compact_review(failing)
    assert resolved.result == "fail"
    assert resolved.checks.semanticCompleteness == "fail"
    assert resolved.checks.semanticCorrectness == "pass"


def test_evidence_validation_allows_overlapping_context_with_ordered_raw_values() -> None:
    pages = {1: "CONTAINER SLAC:\n1,000x20KG BAGS\n(20 HEAT TREATED PALLETS)"}
    records = (
        {
            "pageNumber": 1,
            "rawValue": "1,000x20KG BAGS",
            "ocrExcerpt": "CONTAINER SLAC:\n1,000x20KG BAGS\n(20 HEAT TREATED PALLETS)",
        },
        {
            "pageNumber": 1,
            "rawValue": "20 HEAT TREATED PALLETS",
            "ocrExcerpt": "1,000x20KG BAGS\n(20 HEAT TREATED PALLETS)",
        },
    )
    parsed = tuple(RawOcrValueEvidence.model_validate(record, strict=True) for record in records)

    validate_raw_ocr_evidence(pages, parsed)


def test_compact_anchors_are_hydrated_to_exact_context_without_model_excerpts() -> None:
    pages = {1: "MARKS AND NUMBERS\nUACU5074791\n1289901\n1x40HC"}
    anchors = (
        RawOcrAnchor.model_validate({"pageNumber": 1, "rawValue": "UACU5074791"}, strict=True),
        RawOcrAnchor.model_validate({"pageNumber": 1, "rawValue": "1289901"}, strict=True),
    )

    evidence = materialize_raw_ocr_evidence(pages, anchors)

    assert tuple(row.rawValue for row in evidence) == ("UACU5074791", "1289901")
    assert all(row.ocrExcerpt in pages[1] for row in evidence)
    assert all("..." not in row.ocrExcerpt for row in evidence)
    validate_raw_ocr_evidence(pages, evidence)

    reversed_evidence = materialize_raw_ocr_evidence(pages, tuple(reversed(anchors)))
    assert tuple(row.rawValue for row in reversed_evidence) == (
        "UACU5074791",
        "1289901",
    )


def test_compact_anchor_hydration_distinguishes_repeated_source_occurrences() -> None:
    pages = {1: ("CONTAINER A\n1,000 BAGS\nA1\nA2\nA3\nA4\nCONTAINER B\n1,000 BAGS\nB1\nB2")}
    anchor = RawOcrAnchor.model_validate({"pageNumber": 1, "rawValue": "1,000 BAGS"}, strict=True)

    evidence = materialize_raw_ocr_evidence(pages, (anchor, anchor))

    assert len(evidence) == 2
    assert evidence[0].ocrExcerpt != evidence[1].ocrExcerpt
    validate_raw_ocr_evidence(pages, evidence)
    review = CompactReviewWireDraft.model_validate(
        {
            "decision": "review",
            "findings": (
                {
                    "severity": "blocking",
                    "category": "relationship",
                    "message": "Both repeated package rows require allocation.",
                    "rawOcrEvidence": (
                        {"pageNumber": 1, "rawValue": "1,000 BAGS"},
                        {"pageNumber": 1, "rawValue": "1,000 BAGS"},
                    ),
                },
            ),
            "summary": "Repeated rows are intentionally distinct evidence.",
        },
        strict=True,
    )
    assert len(review.findings[0].rawOcrEvidence) == 2


def test_occurrence_excerpt_keeps_raw_value_in_duplicate_context_blocks() -> None:
    source = "DAMIETTA\nEGYPT\n\nDAMIETTA\nEGYPT"
    raw_value = "DAMIETTA"
    start = source.rindex(raw_value)

    excerpt = _occurrence_excerpt(source, start, start + len(raw_value))

    assert raw_value in excerpt
    assert excerpt in source


def test_compact_anchor_hydration_sorts_logical_rows_from_column_major_ocr() -> None:
    pages = {1: "C1\nC2\n1,000 BAGS\n1,000 BAGS\nGOODS A\nGOODS B"}
    anchors = tuple(
        RawOcrAnchor.model_validate({"pageNumber": 1, "rawValue": value}, strict=True)
        for value in ("C1", "1,000 BAGS", "GOODS A", "C2", "1,000 BAGS", "GOODS B")
    )

    evidence = materialize_raw_ocr_evidence(pages, anchors)

    assert tuple(row.rawValue for row in evidence) == (
        "C1",
        "C2",
        "1,000 BAGS",
        "1,000 BAGS",
        "GOODS A",
        "GOODS B",
    )
    validate_raw_ocr_evidence(pages, evidence)


def test_compact_anchor_hydration_rejects_reusing_one_ocr_occurrence() -> None:
    pages = {1: "No. of Packages\n1,000 BAGS"}
    anchor = RawOcrAnchor.model_validate({"pageNumber": 1, "rawValue": "1,000 BAGS"}, strict=True)

    with pytest.raises(WorkItemError, match="occurrence 2 is absent"):
        materialize_raw_ocr_evidence(pages, (anchor, anchor))


def test_compact_anchor_hydration_rejects_same_offset_prefixes_without_type_error() -> None:
    pages = {1: "PORT OF LOADING\nHAMBURG"}
    anchors = tuple(
        RawOcrAnchor.model_validate({"pageNumber": 1, "rawValue": value}, strict=True)
        for value in ("HAMBURG", "HAMB")
    )

    with pytest.raises(WorkItemError, match="not verbatim/in source order"):
        materialize_raw_ocr_evidence(pages, anchors)


def _review_finding(
    *,
    raw_value: str,
    excerpt: str,
    target_path: str,
    category: str = "missing_field",
    message: str = "The candidate omitted a value.",
    severity: str = "blocking",
) -> SemanticReviewFinding:
    return SemanticReviewFinding.model_validate(
        {
            "severity": severity,
            "category": category,
            "message": message,
            "targetPaths": (target_path,),
            "rawOcrEvidence": ({"pageNumber": 1, "rawValue": raw_value, "ocrExcerpt": excerpt},),
            "imageUse": "not_used",
        },
        strict=True,
    )


@pytest.mark.parametrize(
    "finding",
    [
        _review_finding(
            raw_value="2X40' CNTR(S)",
            excerpt="Description of Goods\n2X40' CNTR(S)",
            target_path="documentPatch.containers",
        ),
        _review_finding(
            raw_value="FREE IN FREE OUT",
            excerpt="Freight Details\nFREE IN FREE OUT",
            target_path="documentPatch.freight",
        ),
        _review_finding(
            raw_value="EXPORTER CONTRY: CANADA",
            excerpt="EXPORTER VAT ID: 123\nEXPORTER CONTRY: CANADA",
            target_path="documentPatch.cargoGroups[0].origin",
        ),
    ],
)
def test_review_policy_rejects_proven_out_of_schema_missing_claims(
    finding: SemanticReviewFinding,
) -> None:
    with pytest.raises(ReviewPolicyError):
        validate_review_policy((finding,))


def test_review_policy_accepts_grounded_container_and_goods_origin_claims() -> None:
    findings = (
        _review_finding(
            raw_value="UACU5074791",
            excerpt="CONTAINER NO\nUACU5074791",
            target_path="documentPatch.containers",
        ),
        _review_finding(
            raw_value="U.S.A.",
            excerpt="COUNTRY OF ORIGIN: U.S.A.",
            target_path="documentPatch.cargoGroups[0].origin",
        ),
    )

    validate_review_policy(findings)


def test_review_policy_rejects_exporter_id_as_reference_and_false_iso_claim() -> None:
    findings = (
        _review_finding(
            raw_value="EXP ID DE362889675",
            excerpt="HS CODE 87012190\nEXP ID DE362889675",
            target_path="documentPatch.forwardingAndExportReferences",
        ),
        _review_finding(
            raw_value="ADMU/511272/8",
            excerpt="Container No / Seal No\nADMU/511272/8",
            target_path="documentPatch.containers[0].containerNumber",
            category="incorrect_field",
            message="The container identifier has an invalid ISO check digit.",
        ),
    )

    with pytest.raises(ReviewPolicyError) as caught:
        validate_review_policy(findings)

    assert "identity or registration" in str(caught.value)
    assert "valid ISO 6346 check digit" in str(caught.value)


def test_review_policy_rejects_unrepresentable_correction_paths() -> None:
    findings = (
        _review_finding(
            raw_value="PACKING GROUP II",
            excerpt="UN 1993 CLASS 3 PACKING GROUP II",
            target_path="documentPatch.cargoGroups[0].dangerousGoods[0].packingGroup",
        ),
        _review_finding(
            raw_value="FLAMMABLE LIQUID N.O.S.",
            excerpt="PROPER SHIPPING NAME: FLAMMABLE LIQUID N.O.S.",
            target_path="documentPatch.cargoGroups[0].dangerousGoods[0].properShippingName",
        ),
        _review_finding(
            raw_value="USA - Gulf",
            excerpt="COUNTRY OF ORIGIN: USA - Gulf",
            target_path="documentPatch.cargoGroups[0].origin.country",
        ),
    )

    with pytest.raises(ReviewPolicyError) as caught:
        validate_review_policy(findings)

    message = str(caught.value)
    assert "packingGroup" in message
    assert "properShippingName" in message
    assert "origin.country" in message


def test_review_policy_preserves_single_package_level_union_shape() -> None:
    candidate = {
        "relationExplicitLabel": {
            "documentPatch": {
                "cargoAllocationGroups": [
                    {
                        "groupId": "g1",
                        "coverage": "single_package_level",
                        "packageId": "p1",
                        "allocations": [
                            {"containerNumber": "UACU5074791", "packageQuantity": 6}
                        ],
                    }
                ]
            }
        }
    }
    finding = _review_finding(
        raw_value="6 PALLETS",
        excerpt="UACU5074791 6 PALLETS",
        target_path="documentPatch.cargoAllocationGroups[0].allocations[0].packageId",
        category="relationship",
        message="The allocation row is missing packageId p1.",
    )

    with pytest.raises(ReviewPolicyError, match="single_package_level stores its packageId"):
        validate_review_policy((finding,), candidate=candidate)


def test_review_policy_rejects_broad_demand_for_row_package_id_in_single_level_union() -> None:
    candidate = {
        "relationExplicitLabel": {
            "documentPatch": {
                "cargoAllocationGroups": [
                    {
                        "groupId": "g1",
                        "coverage": "single_package_level",
                        "packageId": "p1",
                        "allocations": [
                            {"containerNumber": "MRKU6557430", "packageQuantity": 15}
                        ],
                    }
                ]
            }
        }
    }
    finding = _review_finding(
        raw_value="1 Container Said to Contain 15 PALLET",
        excerpt="Container MRKU6557430\n1 Container Said to Contain 15 PALLET",
        target_path="documentPatch.cargoAllocationGroups[0]",
        category="relationship",
        message=(
            "The pallet allocation does not explicitly reference package p1, so the "
            "allocation is not linked to the pallet package fact."
        ),
    )

    with pytest.raises(ReviewPolicyError, match="single_package_level stores its packageId"):
        validate_review_policy((finding,), candidate=candidate)


def test_review_policy_rejects_false_duplicate_reference_finding() -> None:
    candidate = {
        "relationExplicitLabel": {
            "documentPatch": {
                "forwardingAndExportReferences": [
                    "110657",
                    "1900502 DT 16.05.2025",
                ]
            }
        }
    }
    finding = _review_finding(
        raw_value="INV NO. 110657 DT 16.05.2025",
        excerpt="RMS NUMBER-21074503\nINV NO. 110657 DT 16.05.2025",
        target_path="documentPatch.forwardingAndExportReferences[0]",
        category="redundancy",
        message=(
            "The invoice reference 110657 is represented twice: once standalone and again "
            "within a second reference entry."
        ),
    )

    with pytest.raises(ReviewPolicyError, match="occurs only once"):
        validate_review_policy((finding,), candidate=candidate)


def test_review_policy_preserves_canonical_class_two_hazard_category() -> None:
    candidate = {
        "relationExplicitLabel": {
            "documentPatch": {
                "cargoGroups": [
                    {
                        "groupId": "g1",
                        "dangerousGoods": [{"hazardCategory": "GASES"}],
                    }
                ]
            }
        }
    }
    finding = _review_finding(
        raw_value="2.1",
        excerpt="IMO CLASS 2.1",
        target_path="documentPatch.cargoGroups[0].dangerousGoods[0].hazardCategory",
        category="incorrect_field",
        message="GASES should be replaced by the printed class 2.1.",
    )

    with pytest.raises(ReviewPolicyError, match=r"canonically maps.*GASES"):
        validate_review_policy((finding,), candidate=candidate)


@pytest.mark.parametrize(
    ("raw_value", "category", "target_path"),
    [
        (
            "CL 2.1",
            "incorrect_field",
            "documentPatch.cargoGroups[0].dangerousGoods[0].hazardCategory",
        ),
        (
            "UN 1950 CL 2.1 LQ",
            "missing_field",
            "documentPatch.cargoGroups[0].dangerousGoods[0]",
        ),
    ],
)
def test_review_policy_preserves_canonical_class_two_from_observed_ocr_forms(
    raw_value: str,
    category: str,
    target_path: str,
) -> None:
    candidate = {
        "relationExplicitLabel": {
            "documentPatch": {
                "cargoGroups": [
                    {
                        "groupId": "g1",
                        "dangerousGoods": [{"hazardCategory": "GASES"}],
                    }
                ]
            }
        }
    }
    finding = _review_finding(
        raw_value=raw_value,
        excerpt="UN 1950 CL 2.1 LQ",
        target_path=target_path,
        category=category,
        message="The printed class 2.1 is missing or not preserved.",
    )

    with pytest.raises(ReviewPolicyError, match=r"canonically maps.*GASES"):
        validate_review_policy((finding,), candidate=candidate)


def test_review_policy_does_not_force_out_of_scope_dg_text_into_cargo_fields() -> None:
    finding = SemanticReviewFinding.model_validate(
        {
            "severity": "blocking",
            "category": "missing_field",
            "message": (
                "Include the printed dangerous-goods name in description and packing group "
                "in additional information."
            ),
            "targetPaths": (
                "documentPatch.cargoGroups[0].description",
                "documentPatch.cargoGroups[0].additionalInformation",
            ),
            "rawOcrEvidence": (
                {
                    "pageNumber": 1,
                    "rawValue": "PSN: TETRACHLOROETHYLENE",
                    "ocrExcerpt": "PSN: TETRACHLOROETHYLENE\nUN 1897 CLASS 6.1 PG: III",
                },
                {
                    "pageNumber": 1,
                    "rawValue": "PG: III",
                    "ocrExcerpt": "PSN: TETRACHLOROETHYLENE\nUN 1897 CLASS 6.1 PG: III",
                },
            ),
            "imageUse": "not_used",
        },
        strict=True,
    )

    with pytest.raises(ReviewPolicyError) as caught:
        validate_review_policy((finding,))

    message = str(caught.value)
    assert "proper shipping name is outside" in message
    assert "packing group is outside" in message


def test_review_policy_rejects_digits_only_rewrite_of_ocr_phone() -> None:
    finding = _review_finding(
        raw_value="+20 -3-481 2 336 14F2",
        excerpt="TEL: +20 -3-481 2 336 14F2",
        target_path="documentPatch.parties.deliveryAgent.contactDetails.phoneNumbers[0]",
        category="incorrect_field",
        message=(
            "The phone contains the non-digit character F; phone values must contain "
            "printed digits only."
        ),
    )

    with pytest.raises(ReviewPolicyError, match="phone values may retain printed OCR"):
        validate_review_policy((finding,))


def test_review_policy_rejects_generic_payment_place_and_per_package_net_mass() -> None:
    findings = (
        _review_finding(
            raw_value="Ocean Freight payable at Origin",
            excerpt="Ocean Freight payable at Origin",
            target_path="documentPatch.freight.paymentPlace",
        ),
        _review_finding(
            raw_value="1 G I DRUM OF 180 KGS NET WEIGHT",
            excerpt="1 G I DRUM OF 180 KGS NET WEIGHT Net Wt: 180.000 KGS",
            target_path="documentPatch.cargoGroups[0].netWeight",
        ),
    )

    with pytest.raises(ReviewPolicyError) as caught:
        validate_review_policy(findings)

    assert "not a named freight payment location" in str(caught.value)
    assert "per-package net mass" in str(caught.value)


def test_review_policy_excludes_standalone_fax_but_accepts_joint_phone_fax() -> None:
    standalone_fax = _review_finding(
        raw_value="222-2222",
        excerpt="TEL: 111-1111\nFAX: 222-2222",
        target_path="documentPatch.parties.consignee.contactDetails.phoneNumbers",
    )
    joint_phone_fax = _review_finding(
        raw_value="333-3333",
        excerpt="PHONE-FAX: 333-3333",
        target_path="documentPatch.parties.consignee.contactDetails.phoneNumbers",
    )

    with pytest.raises(ReviewPolicyError, match="separately labeled fax"):
        validate_review_policy((standalone_fax,))
    validate_review_policy((joint_phone_fax,))


def test_review_policy_enforces_hs_digit_range_without_business_caps() -> None:
    out_of_range = _review_finding(
        raw_value="5848932722024070020",
        excerpt="HS CODE / NCM CODE:\n5848932722024070020",
        target_path="documentPatch.cargoGroups[0].hsCodes",
    )
    maximum_supported = _review_finding(
        raw_value="584893272202407002",
        excerpt="HS CODE / NCM CODE:\n584893272202407002",
        target_path="documentPatch.cargoGroups[0].hsCodes",
    )

    with pytest.raises(ReviewPolicyError, match="supports 6-18 printed digits"):
        validate_review_policy((out_of_range,))
    validate_review_policy((maximum_supported,))


def test_review_policy_rejects_prefix_of_a_longer_printed_hs_code() -> None:
    truncated = _review_finding(
        raw_value="140490100",
        excerpt="NCM: 1404.90.10\nHS CODE 140490100000",
        target_path="documentPatch.cargoGroups[0].hsCodes",
    )

    with pytest.raises(ReviewPolicyError, match="complete printed token"):
        validate_review_policy((truncated,))


def test_review_policy_preserves_partial_unlinked_container_quantity() -> None:
    candidate = _compact_annotation(
        {
            "cargoGroups": ({"groupId": "g1", "description": "MINING EQUIPMENT"},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 40,
                    "typeDescription": "Package(s)",
                },
            ),
            "containers": (
                {"containerNumber": "MSDU2001032"},
                {"containerNumber": "MSMU1744955"},
            ),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "unlinked_package_quantities",
                    "allocations": (
                        {"containerNumber": "MSDU2001032", "packageQuantity": 20},
                    ),
                },
            ),
        }
    ).model_dump(mode="json")
    impossible_link = _review_finding(
        raw_value="20 Package(s)",
        excerpt="MSDU2001032\n20 Package(s)\nTotal Items: 40",
        target_path="documentPatch.cargoAllocationGroups[0].packageIds",
        category="relationship",
        message="Link the explicit 20-package allocation to cargo package p1.",
    )

    with pytest.raises(ReviewPolicyError, match="unlinked_package_quantities"):
        validate_review_policy((impossible_link,), candidate=candidate)


def test_review_policy_preserves_partial_unlinked_quantity_for_group_target() -> None:
    candidate = _compact_annotation(
        {
            "cargoGroups": ({"groupId": "g1", "description": "MACHINERY"},),
            "cargoPackages": (
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 47,
                    "typeDescription": "Piece(s)",
                },
            ),
            "containers": ({"containerNumber": "BSIU9478390"},),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "unlinked_package_quantities",
                    "allocations": (
                        {"containerNumber": "BSIU9478390", "packageQuantity": 6},
                    ),
                },
            ),
        }
    ).model_dump(mode="json")
    impossible_link = _review_finding(
        raw_value="6 Piece(s)",
        excerpt="BSIU9478390\n47 Piece(s)\n6 Piece(s)",
        target_path="documentPatch.cargoAllocationGroups[0]",
        category="relationship",
        message="Link the explicit 6-piece row to the single 47-piece package level.",
    )

    with pytest.raises(ReviewPolicyError, match="unlinked_package_quantities"):
        validate_review_policy((impossible_link,), candidate=candidate)


def test_review_policy_preserves_distinct_row_linked_repeated_package_facts() -> None:
    candidate = _compact_annotation(
        {
            "cargoGroups": ({"groupId": "g1", "description": "POLYMERS"},),
            "cargoPackages": tuple(
                {
                    "packageId": f"p{index}",
                    "groupId": "g1",
                    "quantity": 20,
                    "typeDescription": "PALLETS",
                }
                for index in range(1, 4)
            ),
            "containers": tuple(
                {"containerNumber": value}
                for value in ("APZU4768830", "DFSU4298730", "TCLU4419919")
            ),
            "cargoAllocationGroups": (
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "allocations": tuple(
                        {
                            "containerNumber": container,
                            "packageId": f"p{index}",
                            "packageQuantity": 20,
                        }
                        for index, container in enumerate(
                            ("APZU4768830", "DFSU4298730", "TCLU4419919"), start=1
                        )
                    ),
                },
            ),
        }
    ).model_dump(mode="json")
    collapse = _review_finding(
        raw_value="= 20 PALLETS",
        excerpt=(
            "APZU4768830 = 20 PALLETS\n"
            "DFSU4298730 = 20 PALLETS\n"
            "TCLU4419919 = 20 PALLETS"
        ),
        target_path="documentPatch.cargoPackages[1]",
        category="redundancy",
        message=(
            "The repeated PALLETS entries represent one shipment-level package fact; "
            "remove the duplicate package entries."
        ),
    )

    with pytest.raises(ReviewPolicyError, match="distinct printed container rows"):
        validate_review_policy((collapse,), candidate=candidate)


def test_wrapped_hs_suffix_is_not_treated_as_a_second_truncated_code() -> None:
    item = _work_item_with_text(
        "SPARE PARTS\n"
        "HS CODE:\n"
        "392690979018,853890990000,73181639000\n"
        "0,853650800014,760429900000,72166190900\n"
        "0,\n\n"
        "HS CODE:\n"
        "392690979018,853890990000,731816390000,853650800014,760429900000,"
        "721661909000"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "description": "SPARE PARTS",
                    "hsCodes": (
                        "392690979018",
                        "853890990000",
                        "731816390000",
                        "853650800014",
                        "760429900000",
                        "721661909000",
                    ),
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "392690979018",
        "853890990000",
        "731816390000",
        "853650800014",
        "760429900000",
        "721661909000",
    )


def test_space_grouped_hs_codes_are_not_treated_as_truncated_suffix_codes() -> None:
    item = _work_item_with_text(
        "GOODS\n"
        "HS CODE: 7610 900000\n"
        "8302 410000\n"
        "3919 100000\n"
        "3919 901000\n"
        "8302 410010\n"
        "4008 290000"
    )
    expected = (
        "7610900000",
        "8302410000",
        "3919100000",
        "3919901000",
        "8302410010",
        "4008290000",
    )
    draft = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "hsCodes": expected},)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == expected


def test_hs_heading_combines_valid_six_digit_prefix_with_printed_suffix() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("HS CODE: 090111 49"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("09011149",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "09011149",
    )


def test_hs_heading_combines_odd_length_prefix_with_wrapped_suffix_line() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("HS CODE 2710199\n900\nFRECO ISO 68"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("2710199900",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "2710199900",
    )


def test_unambiguous_document_volume_style_governs_three_decimal_mass() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "TOTAL:\n1 PALLET\n230,080 KGS\n0,672 CBM\n"
            "IMO GROSS WEIGHT: 105,540 KGS\nIMO NET WEIGHT: 100,000 KGS"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "grossWeight": {"value": 105.54, "unit": "kilogram"},
                        "netWeight": {"value": 100.0, "unit": "kilogram"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    group = annotation.relationExplicitLabel.documentPatch.cargoGroups[0]
    assert group.grossWeight is not None and group.grossWeight.value == 105.54
    assert group.netWeight is not None and group.netWeight.value == 100.0


def test_headed_volume_style_governs_same_document_mass_separator() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "TOTAL PACKAGES: 3, TOTAL GROSS WEIGHT: 240,000 KGS, "
            "TOTAL CBM: 1,602"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "grossWeight": {"value": 240.0, "unit": "kilogram"},
                        "volume": {"value": 1.602, "unit": "cubic_metre"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    group = annotation.relationExplicitLabel.documentPatch.cargoGroups[0]
    assert group.grossWeight is not None and group.grossWeight.value == 240.0
    assert group.volume is not None and group.volume.value == 1.602


def test_multiline_metric_tonne_pair_grounds_gross_and_net_values() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "NUMBER OF PACKAGES\n789\n\nMT\n"
            "GROSS WEIGHT / NET WEIGHT\n982,981 982,979"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "grossWeight": {"value": 982.981, "unit": "metric_tonne"},
                        "netWeight": {"value": 982.979, "unit": "metric_tonne"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    group = annotation.relationExplicitLabel.documentPatch.cargoGroups[0]
    assert group.grossWeight is not None and group.grossWeight.value == 982.981
    assert group.netWeight is not None and group.netWeight.value == 982.979


def test_composite_address_prefers_later_role_continuation_over_route_token() -> None:
    target = (
        "ROOM 2303, BUILDING ONE, NO.18 FENJIANGNAN RD., "
        "CHANCHENG DISTRICT, GUANGDONG PROVINCE"
    )
    annotation = build_compact_annotation(
        _work_item_with_pages(
            "SHIPPER/EXPORTER\nADD: ROOM 2303, BUILDING ONE,\n"
            "NO.18 FENJIANGNAN RD.,\nCHANCHENG DISTRICT, FOSHAN, SH>\n\n"
            "PORT OF LOADING\nSHEKOU, GUANGDONG",
            "SH>\nGUANGDONG PROVINCE, CHINA",
        ),
        _compact_annotation(
            {
                "parties": {
                    "shipper": {
                        "name": None,
                        "address": target,
                        "city": None,
                        "country": None,
                        "sameAs": None,
                        "contactDetails": None,
                    }
                }
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.parties.shipper is not None
    assert annotation.normalLabel.documentPatch.parties.shipper.address == target


def test_asterisk_keyed_postal_footnote_extends_party_address() -> None:
    target = "NO.169, CHUANGQIANG ROAD, YONGNING STREET, ZENGCHENG DISTRICT, 511300"
    annotation = build_compact_annotation(
        _work_item_with_text(
            "SHIPPER\nNO.169, CHUANGQIANG ROAD,\n"
            "YONGNING STREET, ZENGCHENG\nDISTRICT, GUANGZHOU, CHINA*\n\n"
            "CARGO DETAILS\n" + ("PALLET GOODS 12345\n" * 40) +
            "*TEL:+86-020-3282-8888\nZIP CODE / POSTAL CODE : 511300"
        ),
        _compact_annotation(
            {"parties": {"shipper": {"address": target}}}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.parties.shipper.address == target


def test_composite_party_name_preserves_slash_connected_cross_page_continuation() -> None:
    target = "CMPC IGUACU EMBALAGENS LTDA C/O ARABCO INTERNATIONAL LOGISTICS"
    annotation = build_compact_annotation(
        _work_item_with_pages(
            "Shipper\nCMPC IGUACU EMBALAGENS LTDA\nOther face fields",
            "*SHIPPER:\nC/O ARABCO INTERNATIONAL LOGISTICS",
        ),
        _compact_annotation(
            {"parties": {"shipper": {"name": target}}}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.parties.shipper.name == target


@pytest.mark.parametrize("raw", ["BOX -836 CRT", "ROW 22PL"])
def test_attached_abbreviated_package_counts_ground_allocations(raw: str) -> None:
    quantity = 836 if "CRT" in raw else 22
    annotation = build_compact_annotation(
        _work_item_with_text(f"SEKU6020780\nGOODS\n{raw}"),
        _compact_annotation(
            {
                "cargoGroups": ({"groupId": "g1", "description": "GOODS"},),
                "containers": ({"containerNumber": "SEKU6020780"},),
                "cargoAllocationGroups": (
                    {
                        "groupId": "g1",
                        "coverage": "unlinked_package_quantities",
                        "allocations": (
                            {
                                "containerNumber": "SEKU6020780",
                                "packageQuantity": quantity,
                            },
                        ),
                    },
                ),
            }
        ),
        pdf_grouping_used=False,
    )

    assert (
        annotation.normalLabel.documentPatch.goodsItems[0]
        .containerAllocations[0]
        .packageQuantity
        == quantity
    )


def test_package_count_glued_after_mass_unit_is_grounded() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("4.366,21 KG954 PACKAGES\n616 PACKAGES 1.809,97 KG"),
        _compact_annotation(
            {
                "cargoGroups": ({"groupId": "g1"},),
                "cargoPackages": (
                    {
                        "packageId": "p1",
                        "groupId": "g1",
                        "quantity": 954,
                        "typeDescription": "PACKAGES",
                    },
                ),
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoPackages[0].quantity == 954


def test_pda_tariff_prefix_grounds_hs_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("PDA.1302.19"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("130219",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "130219",
    )


def test_hc_tariff_prefix_grounds_hs_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("H.C. #8716.80.1000"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("8716801000",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "8716801000",
    )


def test_hsc_tariff_prefix_grounds_hs_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("HSC 84295210"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("84295210",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "84295210",
    )


def test_spanish_tariff_fraction_heading_grounds_hs_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("FRACCIÓN ARANCELARIA 340213"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("340213",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "340213",
    )


def test_harmonised_code_heading_grounds_hs_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("HARMONISED CODE 08081080"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("08081080",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "08081080",
    )


def test_spaced_hs_suffix_groups_form_one_complete_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("HTS Code:68042218 00 00"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("680422180000",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes == (
        "680422180000",
    )


def test_cers_heading_grounds_export_reference() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("CERS# DC5535202309052875366"),
        _compact_annotation(
            {"forwardingAndExportReferences": ("DC5535202309052875366",)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.forwardingAndExportReferences == (
        "DC5535202309052875366",
    )


def test_numbered_references_nos_heading_grounds_multiple_references() -> None:
    references = (
        "OA164-00007080",
        "PHO23A3533/656110/R141",
        "PHO23A3534/656016/TR16",
    )
    annotation = build_compact_annotation(
        _work_item_with_text("(6) REFERENCES NOS:\n" + "\n".join(references)),
        _compact_annotation({"forwardingAndExportReferences": references}),
        pdf_grouping_used=False,
    )

    assert (
        annotation.relationExplicitLabel.documentPatch.forwardingAndExportReferences
        == references
    )


def test_bare_hc_container_type_does_not_create_an_hs_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "PIECES CONTAINER NO SIZE / TYPE SEAL NO TARE IMDG UN CARGO GROSS WEIGHT\n"
            "20 Pallets MKLU4004696 40 HC 413811 3700 2.1 1950 13.560,00"
        ),
        _compact_annotation(
            {
                "containers": ({"containerNumber": "MKLU4004696"},),
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "dangerousGoods": ({"unNumber": "1950"},),
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.relationExplicitLabel.documentPatch.cargoGroups[0].hsCodes is None


def test_iso_looking_identifier_in_explicit_seal_field_is_not_a_container() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "B/L NO HBL001\ncontainer no SEAWAY BILL OF LADING\n"
            "seal no SIMU2529232\n"
            "cargo description 1 x 20 FT ISO TANK CONTAINER"
        ),
        _compact_annotation({"billOfLadingNumber": "HBL001"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.containers is None


def test_explicit_express_release_is_non_negotiable() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("BILL OF LADING\nExpress Release"),
        _compact_annotation({"negotiability": "non_negotiable"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


def test_un_number_is_grounded_by_adjacent_imdg_table_heading() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "PIECES CONTAINER NO SIZE / TYPE SEAL NO TARE IMDG UN CARGO GROSS WEIGHT\n"
            "20 Pallets MKLU4004696 40 HC 413811 3700 2.1 1950 13.560,00"
        ),
        _compact_annotation(
            {
                "containers": ({"containerNumber": "MKLU4004696"},),
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "dangerousGoods": (
                            {"unNumber": "1950"},
                        ),
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert (
        annotation.relationExplicitLabel.documentPatch.cargoGroups[0]
        .dangerousGoods[0]
        .unNumber
        == "1950"
    )


def test_dotted_proforma_invoice_number_is_a_qualified_forwarding_reference() -> None:
    item = _work_item_with_text("GOODS\nP.I.NO.5004709 DD.20-02-2024")
    draft = _compact_annotation({"forwardingAndExportReferences": ("5004709",)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "5004709",
    )


def test_spaced_invoice_identifier_is_retained_as_one_reference() -> None:
    reference = "TP 11 0016487/22.05.25"
    annotation = build_compact_annotation(
        _work_item_with_text(f"INVOICE NO {reference}"),
        _compact_annotation({"forwardingAndExportReferences": (reference,)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        reference,
    )


def test_pi_fragment_inside_purchase_order_is_not_an_invoice_reference() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "*INVOICE NO.: HLC-I-24-039\n*P/O NO.: HLC-PI-23-333"
        ),
        _compact_annotation(
            {"forwardingAndExportReferences": ("HLC-I-24-039",)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "HLC-I-24-039",
    )


def test_reference_invoice_row_qualifies_each_printed_value() -> None:
    references = ("SAM/2183", "24/00233")
    annotation = build_compact_annotation(
        _work_item_with_text(
            "Reference/Invoices numbers: SAM/2183 , 24/00233"
        ),
        _compact_annotation({"forwardingAndExportReferences": references}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == references


def test_repeated_character_invoice_ocr_variant_does_not_force_duplicate_reference() -> None:
    item = _work_item_with_text(
        "INVOICE: 80001668-01\n"
        "INVOICE: 80001669-01\n\n"
        "INVOICE: 800016668-01\n"
        "INVOICE: 800016669-01"
    )
    draft = _compact_annotation(
        {
            "forwardingAndExportReferences": (
                "80001668-01",
                "80001669-01",
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "80001668-01",
        "80001669-01",
    )


def test_complete_repeated_invoice_copy_supersedes_one_character_ocr_loss() -> None:
    item = _work_item_with_text(
        "INVOICE: 80001668-01\n"
        "INVOICE: 80001669-01\n\n"
        "INVOICE: 800016668-01\n"
        "INVOICE: 800016669-01"
    )
    draft = _compact_annotation(
        {
            "forwardingAndExportReferences": (
                "800016668-01",
                "800016669-01",
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=True)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "800016668-01",
        "800016669-01",
    )


def test_digit_qualified_hs_heading_grounds_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("HS8:39199000"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("39199000",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("39199000",)


def test_glued_numeric_field_before_hs_heading_still_grounds_code() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "D NO. 1000562372023090019HS CODE: 3802.90TAX NUMBER: 100/056/237"
        ),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("380290",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("380290",)


def test_invoice_number_and_date_heading_excludes_adjacent_date_value() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "INVOICE NO&DATE:\nSHIN-FRESH-230410VCM & 10.APR.2023"
        ),
        _compact_annotation(
            {"forwardingAndExportReferences": ("SHIN-FRESH-230410VCM",)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "SHIN-FRESH-230410VCM",
    )


def test_invoice_value_starting_with_exp_is_not_mistaken_for_a_label() -> None:
    item = _work_item_with_text("INVOICE NO: EXP-PB-191-24-25")
    draft = _compact_annotation(
        {"forwardingAndExportReferences": ("EXP-PB-191-24-25",)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "EXP-PB-191-24-25",
    )


@pytest.mark.parametrize(
    ("raw_text", "value"),
    (
        ("Measure CBM 1,170", 1.17),
        ("Total: 9,297.674kgs. 19,255cu. m.", 19.255),
    ),
)
def test_volume_decimal_style_is_grounded_on_either_side_of_unit(
    raw_text: str, value: float
) -> None:
    item = _work_item_with_text(raw_text)
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "volume": {"value": value, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].volume.value == value


def test_multidot_grouped_integer_mass_is_grounded_under_gross_weight_heading() -> None:
    item = _work_item_with_text("Gross weight kg.\n1.650.780")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 1650780.0, "unit": "kilogram"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].grossWeight.value == 1650780.0


def test_line_wrapped_invoice_value_is_joined_under_its_explicit_heading() -> None:
    item = _work_item_with_text("INVOICE NUMBER: MELI-\nEGI20240730")
    draft = _compact_annotation(
        {"forwardingAndExportReferences": ("MELI-EGI20240730",)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "MELI-EGI20240730",
    )


def test_commercial_invoice_nr_prefix_keeps_only_the_reference_value() -> None:
    item = _work_item_with_text(
        "AS PER COMMERCIAL INVOICE NR.4806\nCOMMERCIAL INVOICE .4806"
    )
    draft = _compact_annotation({"forwardingAndExportReferences": ("4806",)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "4806",
    )


def test_invoice_hash_dash_prefix_qualifies_reference_value() -> None:
    item = _work_item_with_text("INV # - FDN-EG-1671-2023")
    draft = _compact_annotation(
        {"forwardingAndExportReferences": ("FDN-EG-1671-2023",)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "FDN-EG-1671-2023",
    )


def test_export_license_number_is_a_qualified_export_reference() -> None:
    item = _work_item_with_text("EXPORT LICENSE NO. RI3273824")
    draft = _compact_annotation(
        {"forwardingAndExportReferences": ("RI3273824",)}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "RI3273824",
    )


def test_invoice_num_and_parenthesized_negative_reefer_temperature_are_grounded() -> None:
    item = _work_item_with_text(
        "FBIU5385937\nINVOICE NUM: AAFT/081/23-24\nTEMP:(-)18 DEG CEL"
    )
    draft = _compact_annotation(
        {
            "forwardingAndExportReferences": ("AAFT/081/23-24",),
            "containers": (
                {
                    "containerNumber": "FBIU5385937",
                    "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
                },
            ),
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "AAFT/081/23-24",
    )
    assert annotation.normalLabel.documentPatch.containers[0].temperatureSetpoint.value == -18.0


def test_spaced_numeric_date_uses_explicit_printed_month_day_year_order() -> None:
    item = _work_item_with_text(
        "DATE AT\nBy\n04 26 2024\n\nMonth Day Year"
    )
    draft = _compact_annotation({"issueDate": date(2024, 4, 26)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.issueDate.isoformat() == "2024-04-26"


@pytest.mark.parametrize(
    ("raw_date", "expected"),
    [
        ("28 MARS 2024", date(2024, 3, 28)),
        ("18 FEV. 2024", date(2024, 2, 18)),
    ],
)
def test_french_named_month_dates_are_normalized(
    raw_date: str, expected: date
) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(f"Place and date of issue\n{raw_date}"),
        _compact_annotation({"issueDate": expected}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.issueDate == expected


def test_compact_yymmdd_issue_date_is_normalized_under_explicit_heading() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("DATE OF ISSUE : 230508"),
        _compact_annotation({"issueDate": date(2023, 5, 8)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.issueDate == date(2023, 5, 8)


def test_ppd_is_a_grounded_prepaid_freight_abbreviation() -> None:
    item = _work_item_with_text("Freight Charges\nPPD | COL")
    draft = _compact_annotation(
        {"freight": {"paymentArrangement": "prepaid", "paymentPlace": None}}
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=True)

    assert annotation.normalLabel.documentPatch.freight is not None
    assert annotation.normalLabel.documentPatch.freight.paymentArrangement == "prepaid"


def test_compact_seawaybill_printing_is_non_negotiable() -> None:
    item = _work_item_with_text("*** SEAWAYBILL ***")
    draft = _compact_annotation({"negotiability": "non_negotiable"})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


def test_multimodal_tcn_waybill_title_is_non_negotiable() -> None:
    item = _work_item_with_text(
        "MULTIMODAL TRANSPORT BILL OF LADING / TCN / WAYBILL"
    )
    draft = _compact_annotation({"negotiability": "non_negotiable"})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


def test_odd_length_hs_copy_missing_a_repeated_digit_does_not_force_bad_variant() -> None:
    item = _work_item_with_text(
        "HS CODE: 85049010\nIMPORTER HS CODE: 850490090\n\n"
        "HS CODE: 85049010\nIMPORTER HS CODE: 8504900090"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "hsCodes": ("85049010", "8504900090"),
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == (
        "85049010",
        "8504900090",
    )


def test_volume_uses_adjacent_flattened_table_unit_row_for_decimal_style() -> None:
    item = _work_item_with_text(
        "1024 PACKAGES 50850,000 114,000\nKGM MTQ"
    )
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "volume": {"value": 114.0, "unit": "cubic_metre"},
                },
            )
        }
    )

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.goodsItems[0].volume.value == 114.0


def test_two_digit_dotted_date_uses_day_first_ambiguity_policy() -> None:
    item = _work_item_with_text("LADEN ON BOARD 03.02.25")
    draft = _compact_annotation({"shippedOnBoardDate": date(2025, 2, 3)})

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=False)

    assert annotation.normalLabel.documentPatch.shippedOnBoardDate == date(2025, 2, 3)


def test_wrong_two_digit_dotted_date_reports_day_first_repair_policy() -> None:
    item = _work_item_with_text("LADEN ON BOARD 03.02.25")
    draft = _compact_annotation({"shippedOnBoardDate": date(2025, 3, 2)})

    with pytest.raises(DeterministicAnnotationError, match="day-first-on-ambiguity"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)


@pytest.mark.parametrize(
    ("raw_value", "excerpt"),
    [
        ("ERN NO: 123456", "EXPORT REFERENCES\nERN NO: 123456"),
        ("4567890123456789012", "ACID: 4567890123456789012"),
        ("Export references Svc Contract", "Export references Svc Contract"),
    ],
)
def test_review_policy_rejects_excluded_or_blank_reference_values(
    raw_value: str,
    excerpt: str,
) -> None:
    finding = _review_finding(
        raw_value=raw_value,
        excerpt=excerpt,
        target_path="documentPatch.forwardingAndExportReferences",
    )

    with pytest.raises(ReviewPolicyError):
        validate_review_policy((finding,))


def test_review_policy_accepts_itn_export_reference() -> None:
    finding = _review_finding(
        raw_value="X20240123456789",
        excerpt="ITN: X20240123456789",
        target_path="documentPatch.forwardingAndExportReferences",
    )

    validate_review_policy((finding,))


def test_review_policy_accepts_caed_but_rejects_acid_reference() -> None:
    caed = _review_finding(
        raw_value="PC8656202304042500630",
        excerpt="CAED: PC8656202304042500630",
        target_path="documentPatch.forwardingAndExportReferences",
    )
    acid = _review_finding(
        raw_value="1002704682024081371",
        excerpt="ACID: 1002704682024081371",
        target_path="documentPatch.forwardingAndExportReferences",
    )

    validate_review_policy((caed,))
    with pytest.raises(ReviewPolicyError, match="regulatory/customs"):
        validate_review_policy((acid,))


def test_review_policy_accepts_dg_local_closed_cup_flash_point() -> None:
    closed_cup = _review_finding(
        raw_value="(63.00 C-CC)",
        excerpt="UN 1866 PACKING GROUP III (63.00 C-CC)",
        target_path="documentPatch.cargoGroups[0].dangerousGoods[0].flashPoint",
    )
    explicit = _review_finding(
        raw_value="63.00 C",
        excerpt="FLASH PT: 63.00 C",
        target_path="documentPatch.cargoGroups[0].dangerousGoods[0].flashPoint",
    )
    unrelated_temperature = _review_finding(
        raw_value="46 C.C.",
        excerpt="REEFER SET POINT 46 C.C.",
        target_path="documentPatch.cargoGroups[0].dangerousGoods[0].flashPoint",
    )

    validate_review_policy((closed_cup,))
    validate_review_policy((explicit,))
    with pytest.raises(ReviewPolicyError, match="closed-cup notation"):
        validate_review_policy((unrelated_temperature,))

    glued_closed_cup = _review_finding(
        raw_value="25C.C.C.",
        excerpt="UN1987, ALCOHOLS, CLASS 3, PG III, (25C.C.C.)",
        target_path="documentPatch.cargoGroups[0].dangerousGoods[0].flashPoint",
    )
    validate_review_policy((glued_closed_cup,))


def test_review_policy_rejects_invalid_imo_demand_but_accepts_valid_imo() -> None:
    invalid = _review_finding(
        raw_value="9200426",
        excerpt="LLOYDS/MO NUMBER 9200426",
        target_path="documentPatch.transport.vesselImoNumber",
    )
    valid = _review_finding(
        raw_value="9319466",
        excerpt="IMO NUMBER 9319466",
        target_path="documentPatch.transport.vesselImoNumber",
    )

    with pytest.raises(ReviewPolicyError, match="valid IMO checksum"):
        validate_review_policy((invalid,))
    validate_review_policy((valid,))


def test_review_policy_rejects_false_invalid_imo_assertion() -> None:
    false_rejection = _review_finding(
        raw_value="9293442",
        excerpt="IMO NUMBER 9293442",
        target_path="documentPatch.transport.vesselImoNumber",
        category="incorrect_field",
        message="IMO 9293442 has an invalid checksum and must be removed.",
    )

    with pytest.raises(ReviewPolicyError, match="valid checksum"):
        validate_review_policy((false_rejection,))


def test_review_policy_rejects_load_stow_count_as_handling_instruction() -> None:
    boilerplate = _review_finding(
        raw_value="Shippers Load, Stow and Count",
        excerpt="Said to Contain :-\nShippers Load, Stow and Count\n25,288 KG",
        target_path="documentPatch.cargoGroups[0].handlingInstructions",
    )
    operational = _review_finding(
        raw_value="KEEP AWAY FROM HEAT",
        excerpt="HANDLING INSTRUCTIONS: KEEP AWAY FROM HEAT",
        target_path="documentPatch.cargoGroups[0].handlingInstructions",
    )

    with pytest.raises(ReviewPolicyError, match="responsibility boilerplate"):
        validate_review_policy((boilerplate,))
    validate_review_policy((operational,))


def test_review_policy_rejects_observed_false_positive_review_patterns() -> None:
    false_findings = (
        _review_finding(
            raw_value="MRSU5428343 ML-AE4174072 40 DRY 9'6 20 BINS 21756.000 KGS",
            excerpt="CONTAINER ROW\nMRSU5428343 ML-AE4174072 40 DRY 9'6 20 BINS 21756.000 KGS",
            target_path="documentPatch.containers[0].verifiedGrossMass",
        ),
        _review_finding(
            raw_value="Shipper Ref# PORSCHE MIDDLE EAST AND AFRICA FZE DUBAI U.A.E.",
            excerpt="Shipper Ref# PORSCHE MIDDLE EAST AND AFRICA FZE DUBAI U.A.E.",
            target_path="documentPatch.parties.shipper",
        ),
        _review_finding(
            raw_value="FOREIGN EXPORTER COUNTRY: TURKEY",
            excerpt="FOREIGN EXPORTER ID: 206 137 5043\nFOREIGN EXPORTER COUNTRY: TURKEY",
            target_path="documentPatch.parties.shipper.country",
        ),
        _review_finding(
            raw_value="SIZE 2X2-8",
            excerpt="VERTICAL PUMP WITH:\nMODEL CV3171 M, SIZE 2X2-8",
            target_path="documentPatch.cargoGroups[0].description",
            category="contamination",
            message="Remove the measure SIZE 2X2-8 from the cargo description.",
        ),
    )

    with pytest.raises(ReviewPolicyError) as caught:
        validate_review_policy(false_findings)

    message = str(caught.value)
    assert "ordinary container-row" in message
    assert "Shipper Ref" in message
    assert "does not prove the shipper's country" in message
    assert "product model/specification" in message


def test_review_policy_rejects_an_already_satisfied_absence_finding() -> None:
    finding = _review_finding(
        raw_value="NINGBO, CHINA",
        excerpt="Pre-carriage by\nNINGBO, CHINA",
        target_path="documentPatch.route.placeOfReceipt",
        category="truth_boundary",
        message="placeOfReceipt is unsupported and must remain absent.",
    )
    absent = _compact_annotation(
        {"route": {"placeOfReceipt": None, "portOfLoading": {"name": "NINGBO"}}}
    ).model_dump(mode="json")

    with pytest.raises(ReviewPolicyError, match="already null or absent"):
        validate_review_policy((finding,), candidate=absent)

    present = _compact_annotation(
        {"route": {"placeOfReceipt": {"name": "NINGBO, CHINA"}}}
    ).model_dump(mode="json")
    validate_review_policy((finding,), candidate=present)


def test_review_policy_rejects_container_allocation_without_a_container() -> None:
    finding = _review_finding(
        raw_value="3 PACKAGES",
        excerpt="Quantity\n3 PACKAGES",
        target_path="documentPatch.cargoAllocationGroups",
        category="relationship",
        message="Add an allocation linking the package quantity to its cargo group.",
    )

    with pytest.raises(ReviewPolicyError, match=r"cargoPackages\.groupId"):
        validate_review_policy((finding,), raw_ocr_text="Quantity\n3 PACKAGES")

    validate_review_policy(
        (finding,),
        raw_ocr_text="Container\nUACU5074791\nQuantity\n3 PACKAGES",
    )


def test_review_policy_rejects_a_measurement_removal_absent_from_description() -> None:
    finding = _review_finding(
        raw_value="11000.000 KGM",
        excerpt="USED MACHINES TEXTIL AND KGM\n11000.000 KGM",
        target_path="documentPatch.cargoGroups[0].description",
        category="contamination",
        message="Remove the mass unit KGM from the cargo description.",
    )
    clean = _compact_annotation(
        {"cargoGroups": ({"groupId": "g1", "description": "USED MACHINES TEXTIL AND ACCESSORIES"},)}
    ).model_dump(mode="json")

    with pytest.raises(ReviewPolicyError, match="absent from the reviewed candidate"):
        validate_review_policy((finding,), candidate=clean)


def test_review_policy_rejects_origin_as_a_prepaid_payment_arrangement() -> None:
    finding = _review_finding(
        raw_value="ORIGIN 1/3",
        excerpt="Freight payable at:\nORIGIN 1/3",
        target_path="documentPatch.freight",
        message="Represent ORIGIN as prepaid.",
    )

    with pytest.raises(ReviewPolicyError, match="not evidence for prepaid"):
        validate_review_policy((finding,))


def test_review_policy_accepts_explicit_vgm_and_explicit_shipper_exporter_identity() -> None:
    findings = (
        _review_finding(
            raw_value="21756.000 KGS",
            excerpt="VGM: 21756.000 KGS",
            target_path="documentPatch.containers[0].verifiedGrossMass",
        ),
        _review_finding(
            raw_value="FOREIGN EXPORTER COUNTRY: TURKEY",
            excerpt="SHIPPER IS THE EXPORTER\nFOREIGN EXPORTER COUNTRY: TURKEY",
            target_path="documentPatch.parties.shipper.country",
        ),
    )

    validate_review_policy(findings)


def test_review_policy_checks_address_redundancy_against_reviewed_candidate() -> None:
    candidate: dict[str, Any] = {
        "relationExplicitLabel": {
            "documentPatch": {
                "parties": {
                    "shipper": {
                        "address": "7106, DONG-II TECHNO TOWN 7TH, 823, GWANGYANG2-DONG, "
                        "DONGAN-GU, GYEONGI, 431-062",
                        "city": "ANYANG",
                        "country": "SOUTH KOREA",
                    }
                }
            }
        }
    }
    false_finding = _review_finding(
        raw_value="GWANGYANG2-DONG, DONGAN-GU, ANYANG, GYEONGI, SOUTH KOREA",
        excerpt="SHIPPER\nGWANGYANG2-DONG, DONGAN-GU, ANYANG, GYEONGI, SOUTH KOREA",
        target_path="documentPatch.parties.shipper.address",
        category="incorrect_field",
        message=(
            "The shipper address redundantly includes the separately modeled city ANYANG and "
            "country SOUTH KOREA."
        ),
    )

    with pytest.raises(ReviewPolicyError, match="claimed address redundancy is absent"):
        validate_review_policy((false_finding,), candidate=candidate)

    candidate["relationExplicitLabel"]["documentPatch"]["parties"]["shipper"]["address"] += (
        ", ANYANG, SOUTH KOREA"
    )
    validate_review_policy((false_finding,), candidate=candidate)


def test_review_policy_rejects_description_omission_already_present_in_candidate() -> None:
    candidate: dict[str, Any] = {
        "relationExplicitLabel": {
            "documentPatch": {
                "cargoGroups": [
                    {
                        "description": (
                            "POLYETHYLENE XLPE FOR M.V CABLES TREE RETARDANT TYPE "
                            "CLNA-TR8142EC"
                        )
                    }
                ]
            }
        }
    }
    false_finding = _review_finding(
        raw_value="(POLYETHYLENE)",
        excerpt="20 BOXES\n(POLYETHYLENE)",
        target_path="documentPatch.cargoGroups[0].description",
        category="incorrect_field",
        message="Cargo group description omits the explicit product line POLYETHYLENE.",
    )

    with pytest.raises(ReviewPolicyError, match="claimed description omission is absent"):
        validate_review_policy((false_finding,), candidate=candidate)

    candidate["relationExplicitLabel"]["documentPatch"]["cargoGroups"][0]["description"] = (
        "XLPE FOR M.V CABLES TREE RETARDANT TYPE CLNA-TR8142EC"
    )
    validate_review_policy((false_finding,), candidate=candidate)


@pytest.mark.parametrize(
    "target_path",
    [
        "documentPatch.cargoGroups[0].netWeight",
        "documentPatch.cargoGroups[0].grossWeight",
    ],
)
def test_review_policy_rejects_tare_column_as_cargo_mass(target_path: str) -> None:
    finding = _review_finding(
        raw_value="4700",
        excerpt="GROSS WEIGHT\nTARE WEIGHT\n4700",
        target_path=target_path,
    )

    with pytest.raises(ReviewPolicyError, match="column-scoped"):
        validate_review_policy((finding,))


def test_review_policy_accepts_scalar_under_its_gross_column() -> None:
    finding = _review_finding(
        raw_value="9550.000",
        excerpt="GROSS WEIGHT\n9550.000\nKGS",
        target_path="documentPatch.cargoGroups[0].grossWeight",
    )

    validate_review_policy((finding,))


def test_review_policy_rejects_unitless_measurement_as_volume() -> None:
    unitless = _review_finding(
        raw_value="50.000",
        excerpt="MEASUREMENT\n50.000",
        target_path="documentPatch.cargoGroups[0].volume",
    )
    cubic_metres = _review_finding(
        raw_value="50.000 CBM",
        excerpt="MEASUREMENT\n50.000 CBM",
        target_path="documentPatch.cargoGroups[0].volume",
    )

    with pytest.raises(ReviewPolicyError, match="explicit printed cubic volume unit"):
        validate_review_policy((unitless,))
    validate_review_policy((cubic_metres,))


def test_review_policy_rejects_unscoped_via_as_forwarding_agent() -> None:
    unscoped_via = _review_finding(
        raw_value="VIA MEDICINOS LINIJA UAB",
        excerpt="SHIPPER\nACME EXPORTS\nVIA MEDICINOS LINIJA UAB",
        target_path="documentPatch.parties.forwardingAgent",
    )
    role_scoped_via = _review_finding(
        raw_value="VIA MEDICINOS LINIJA UAB",
        excerpt="FORWARDING AGENT\nVIA MEDICINOS LINIJA UAB",
        target_path="documentPatch.parties.forwardingAgent",
    )

    with pytest.raises(ReviewPolicyError, match="explicit forwarding-agent role heading"):
        validate_review_policy((unscoped_via,))
    validate_review_policy((role_scoped_via,))


def test_review_policy_requires_explicit_delivery_agent_scope() -> None:
    generic_shipping_agent = _review_finding(
        raw_value="ACME SHIPPING LLC",
        excerpt="SHIPPING AGENT DETAILS\nACME SHIPPING LLC",
        target_path="documentPatch.parties.deliveryAgent",
    )
    destination_agent = _review_finding(
        raw_value="ACME SHIPPING LLC",
        excerpt="AGENT AT DESTINATION\nACME SHIPPING LLC",
        target_path="documentPatch.parties.deliveryAgent",
    )

    with pytest.raises(ReviewPolicyError, match="does not establish the delivery-agent role"):
        validate_review_policy((generic_shipping_agent,))
    validate_review_policy((destination_agent,))


def test_review_policy_accepts_same_as_only_with_grounded_referent() -> None:
    relation_only = SemanticReviewFinding.model_validate(
        {
            "severity": "blocking",
            "category": "relationship",
            "message": "The explicit notify-party relationship is missing.",
            "targetPaths": ("documentPatch.parties.notifyParties",),
            "rawOcrEvidence": (
                {
                    "pageNumber": 1,
                    "rawValue": "THE SAME AS CONSIGNEE",
                    "ocrExcerpt": "Notify\nTHE SAME AS CONSIGNEE",
                },
            ),
            "imageUse": "not_used",
        },
        strict=True,
    )
    grounded_relation = relation_only.model_copy(
        update={
            "rawOcrEvidence": (
                *relation_only.rawOcrEvidence,
                RawOcrValueEvidence.model_validate(
                    {
                        "pageNumber": 1,
                        "rawValue": "ACME IMPORTS",
                        "ocrExcerpt": "CONSIGNEE\nACME IMPORTS",
                    },
                    strict=True,
                ),
            ),
        }
    )

    with pytest.raises(ReviewPolicyError, match="cited OCR-grounded value"):
        validate_review_policy((relation_only,))
    validate_review_policy((grounded_relation,))


def test_review_policy_rejects_false_objection_to_zero_original_sea_waybill() -> None:
    finding = SemanticReviewFinding.model_validate(
        {
            "severity": "blocking",
            "category": "document_unit",
            "message": "The document type should not be sea_waybill without generic form wording.",
            "targetPaths": (),
            "rawOcrEvidence": (
                {
                    "pageNumber": 1,
                    "rawValue": "0 (ZERO)",
                    "ocrExcerpt": "NUMBER OF ORIGINAL BILLS OF LADING\n0 (ZERO)",
                },
            ),
            "imageUse": "not_used",
        },
        strict=True,
    )

    with pytest.raises(ReviewPolicyError, match="zero-original count supports sea_waybill"):
        validate_review_policy((finding,))


def test_review_policy_rejects_copy_stamp_override_of_underlying_bill_of_lading() -> None:
    finding = _review_finding(
        raw_value="NON-NEGOTIABLE COPY",
        excerpt="NON-NEGOTIABLE COPY",
        target_path="documentType",
        category="incorrect_field",
        message="Classify the document as a sea waybill and mark it non-negotiable.",
    )
    candidate: dict[str, Any] = {"documentType": "bill_of_lading"}
    raw_ocr = (
        "NON-NEGOTIABLE COPY\nBILL OF LADING\n"
        "Number of Original B(s)/L\nTHREE(3)\n"
        "ONE of which being accomplished, the others to stand void."
    )

    with pytest.raises(ReviewPolicyError, match="copy-status stamp"):
        validate_review_policy(
            (finding,),
            candidate=candidate,
            raw_ocr_text=raw_ocr,
        )

    validate_review_policy(
        (finding,),
        candidate=candidate,
        raw_ocr_text=f"{raw_ocr}\nEXPRESS RELEASE",
    )


def test_review_resolution_promotes_representable_advisory_omission_to_blocking() -> None:
    finding = _review_finding(
        raw_value="62589948",
        excerpt="DELIVERY AGENT TEL: 62589948",
        target_path="documentPatch.parties.deliveryAgent.contactDetails.phoneNumbers[0]",
        severity="advisory",
    )
    draft = CompactReviewDraft.model_validate(
        {
            "decision": "review",
            "findings": (finding,),
            "summary": "One explicit contact was omitted.",
        },
        strict=True,
    )

    resolved = _resolve_compact_review(draft)

    assert resolved.result == "fail"
    assert resolved.findings[0].severity == "blocking"
    assert resolved.checks.semanticCompleteness == "fail"


def test_review_resolution_keeps_correct_omission_warning_as_nonblocking() -> None:
    finding = _review_finding(
        raw_value="Weight in Kgs Total: 1 CONTAINER(S)",
        excerpt="Weight in Kgs Total: 1 CONTAINER(S)",
        target_path="documentPatch.cargoGroups[0].netWeight",
        severity="advisory",
        message=(
            "The total mass is correctly omitted, but a schema warning could identify the "
            "unsupported value."
        ),
    )
    draft = CompactReviewDraft.model_validate(
        {
            "decision": "review",
            "findings": (finding,),
            "summary": "The target facts are correct.",
        },
        strict=True,
    )

    resolved = _resolve_compact_review(draft)

    assert resolved.result == "pass"
    assert resolved.findings[0].severity == "advisory"


def test_review_retry_feedback_contains_only_actionable_grounded_findings() -> None:
    finding = _review_finding(
        raw_value="62589948",
        excerpt="DELIVERY AGENT TEL: 62589948",
        target_path="documentPatch.parties.deliveryAgent.contactDetails.phoneNumbers[0]",
    )
    review = IndependentReviewArtifact.model_validate(
        {
            "schemaVersion": 1,
            "documentId": "doc_" + "a" * 64,
            "candidateAttempt": 1,
            "reviewNumber": 1,
            "workItemSha256": "b" * 64,
            "candidateSha256": "c" * 64,
            "providerId": "luna_high",
            "reviewerModel": "gpt-5.6-luna",
            "reviewerReasoningEffort": "high",
            "result": "fail",
            "checks": {
                "rawOcrTruthBoundary": "pass",
                "evidenceIntegrity": "pass",
                "semanticCompleteness": "fail",
                "semanticCorrectness": "pass",
                "contaminationAndRedundancy": "pass",
                "documentUnitAndRelationships": "pass",
            },
            "findings": (finding,),
            "summary": "One explicit contact was omitted.",
        },
        strict=True,
    )

    payload = json.loads(_review_retry_feedback(review))

    assert "exact-path correction envelope" in payload["instruction"]
    assert "every allowedCorrectionPaths key once" in payload["instruction"]
    assert payload["findings"] == [
        {
            "category": "missing_field",
            "message": "The candidate omitted a value.",
            "targetPaths": ["documentPatch.parties.deliveryAgent.contactDetails.phoneNumbers[0]"],
            "rawOcrEvidence": [{"pageNumber": 1, "rawValue": "62589948"}],
        }
    ]
    assert "candidateSha256" not in json.dumps(payload)
    assert "ocrExcerpt" not in json.dumps(payload)


def test_contextual_output_validator_repairs_false_review_inside_agent_run() -> None:
    item = _work_item_with_text("Freight Details\nFREE IN FREE OUT")
    wire = CompactReviewWireDraft.model_validate(
        {
            "decision": "review",
            "findings": (
                {
                    "severity": "blocking",
                    "category": "missing_field",
                    "message": "The freight field is missing.",
                    "targetPaths": ("documentPatch.freight",),
                    "rawOcrEvidence": ({"pageNumber": 1, "rawValue": "FREE IN FREE OUT"},),
                    "imageUse": "not_used",
                },
            ),
            "summary": "One omission found.",
        },
        strict=True,
    ).model_dump(mode="json")
    context = RunContext(
        deps=_ReviewValidationContext(work_item=item, candidate={}),
        model=TestModel(),
        usage=RunUsage(),
    )

    with pytest.raises(ModelRetry, match="supported freight payment values"):
        _validate_compact_review_wire(context, wire)

    corrected = {"decision": "review", "findings": [], "summary": "No defect."}
    assert _validate_compact_review_wire(context, corrected) == corrected


def test_successful_provider_response_is_receipted_if_local_hydration_fails() -> None:
    result = AgentCallResult(_annotation(), _receipt("extract", 1))

    failure = _postprocess_failure(result, WorkItemError("anchor is absent"))

    assert failure.receipt.status == "error"
    assert failure.receipt.outputSha256 is None
    assert failure.receipt.errorType == "WorkItemError"
    assert failure.receipt.inputTokens == result.receipt.inputTokens
    assert failure.receipt.outputTokens == result.receipt.outputTokens
    assert failure.receipt.transcript == result.receipt.transcript


def test_compact_layout_and_review_hydrate_evidence_before_orchestration() -> None:
    item = _work_item_with_text("MARKS AND NUMBERS\nUACU5074791\n1289901")
    layout = CompactDocumentLayoutGuidance.model_validate(
        {
            "decision": "layout_guidance",
            "observations": (
                {
                    "pageNumber": 1,
                    "observationType": "row_association",
                    "rawOcrAnchors": (
                        {"pageNumber": 1, "rawValue": "UACU5074791"},
                        {"pageNumber": 1, "rawValue": "1289901"},
                    ),
                    "guidance": "The two printed values share one marks block.",
                },
            ),
            "decisionNotes": ("Only layout association was used.",),
        },
        strict=True,
    )
    review = CompactReviewWireDraft.model_validate(
        {
            "decision": "review",
            "findings": (
                {
                    "severity": "blocking",
                    "category": "missing_field",
                    "message": "The printed container value is missing.",
                    "targetPaths": ("documentPatch.containers[0].containerNumber",),
                    "rawOcrEvidence": ({"pageNumber": 1, "rawValue": "UACU5074791"},),
                    "imageUse": "not_used",
                },
            ),
            "summary": "One grounded omission was found.",
        },
        strict=True,
    )
    request = CompactDocumentAssistanceRequest.model_validate(
        {
            "decision": "document_required",
            "pageNumbers": (1,),
            "ambiguity": "The printed values need row association.",
            "affectedTargetPaths": ("documentPatch.containers[0].sealNumbers",),
            "rawOcrEvidence": ({"pageNumber": 1, "rawValue": "1289901"},),
        },
        strict=True,
    )

    hydrated_layout = _hydrate_layout_guidance(item, layout)
    hydrated_review = _hydrate_review(item, review)
    hydrated_request = _hydrate_document_request(item, request, review=False)

    assert (
        hydrated_layout.observations[0].rawOcrAnchors[0].ocrExcerpt
        in page_texts(item.joinedRawText)[1]
    )
    assert (
        hydrated_review.findings[0].rawOcrEvidence[0].ocrExcerpt
        in page_texts(item.joinedRawText)[1]
    )
    assert hydrated_request.rawOcrEvidence[0].ocrExcerpt in page_texts(item.joinedRawText)[1]


def test_call_transcript_retains_visible_responses_and_exact_repair_feedback() -> None:
    messages = [
        ModelResponse(parts=[TextPart(content='{"decision":"annotation"}')]),
        ModelRequest(parts=[RetryPromptPart(content="Fix the cargo relation.")]),
    ]

    transcript = _call_transcript(messages)

    assert [row.eventType for row in transcript.events] == [
        "model_response",
        "validation_feedback",
    ]
    response_payload = transcript.events[0].payload
    assert isinstance(response_payload, dict)
    response_parts = response_payload["parts"]
    assert isinstance(response_parts, list)
    first_part = response_parts[0]
    assert isinstance(first_part, dict)
    assert first_part["content"] == '{"decision":"annotation"}'
    assert "Fix the cargo relation." in str(transcript.events[1].payload)


class _RetryGateway:
    extract_attempts: list[int]
    review_attempts: list[int]

    def __init__(self) -> None:
        self.extract_attempts = []
        self.review_attempts = []

    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        return AgentCallResult(
            (
                _compact_annotation({"originalBillOfLadingNumber": "HBL-001"})
                if attempt == 1
                else _annotation()
            ),
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        output = _review(passed=attempt == 2)
        if attempt == 1:
            value = output.model_dump(mode="python")
            value["findings"][0]["targetPaths"] = ()
            output = ReviewDraft.model_validate(value, strict=True)
        return AgentCallResult(
            output,
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def document_layout(
        self, *args: Any, **kwargs: Any
    ) -> AgentCallResult[DocumentLayoutGuidance]:
        raise AssertionError("the fixture must not load or send a PDF")


class _DocumentRequestGateway(_RetryGateway):
    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        request = DocumentAssistanceRequest.model_validate(
            {
                "decision": "document_required",
                "pageNumbers": (1,),
                "ambiguity": "A row relationship cannot be resolved from flattened OCR.",
                "affectedTargetPaths": ("documentPatch.billOfLadingNumber",),
                "rawOcrEvidence": (
                    {
                        "pageNumber": 1,
                        "rawValue": "HBL-001",
                        "ocrExcerpt": "B/L NO: HBL-001",
                    },
                ),
            },
            strict=True,
        )
        return AgentCallResult(
            request,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _ReviewerLayoutRetentionGateway(_RetryGateway):
    correction_layout_guidance: DocumentLayoutGuidance | None

    def __init__(self) -> None:
        super().__init__()
        self.correction_layout_guidance = None

    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        if attempt == 2:
            self.correction_layout_guidance = kwargs["layout_guidance"]
        output: BaseModel = (
            _compact_correction({"documentPatch.billOfLadingNumber": "HBL-001"})
            if kwargs["correction_base"] is not None
            else _compact_annotation({"originalBillOfLadingNumber": "HBL-001"})
        )
        return AgentCallResult(
            output,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        layout_guidance = kwargs["layout_guidance"]
        if attempt == 1 and layout_guidance is None:
            output: BaseModel = ReviewDocumentAssistanceRequest.model_validate(
                {
                    "decision": "document_required",
                    "pageNumbers": (1,),
                    "ambiguity": "Resolve the flattened B/L-number heading association.",
                    "affectedTargetPaths": ("documentPatch.billOfLadingNumber",),
                    "rawOcrEvidence": (
                        {
                            "pageNumber": 1,
                            "rawValue": "HBL-001",
                            "ocrExcerpt": "B/L NO: HBL-001",
                        },
                    ),
                },
                strict=True,
            )
        else:
            output = _review(passed=attempt == 2)
        return AgentCallResult(
            output,
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def document_layout(
        self, work_item: AgentWorkItem, **kwargs: Any
    ) -> AgentCallResult[DocumentLayoutGuidance]:
        attempt = int(kwargs["candidate_attempt"])
        pdf = kwargs["pdf"]
        guidance = DocumentLayoutGuidance.model_validate(
            {
                "decision": "layout_guidance",
                "observations": (
                    {
                        "pageNumber": 1,
                        "observationType": "heading_scope",
                        "rawOcrAnchors": (
                            {
                                "pageNumber": 1,
                                "rawValue": "HBL-001",
                                "ocrExcerpt": "B/L NO: HBL-001",
                            },
                        ),
                        "guidance": "HBL-001 is positioned in the B/L-number box.",
                    },
                ),
                "decisionNotes": ("Resolved heading scope only.",),
            },
            strict=True,
        )
        now = datetime.now(UTC)
        response = ModelResponseReceipt.model_validate(
            {
                "inputTokens": 100,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "outputTokens": 20,
                "reasoningTokens": 5,
                "costUsd": None,
            },
            strict=True,
        )
        receipt = AgentCallReceipt.model_validate(
            {
                "receiptSchemaVersion": 2,
                "callId": f"call-document-layout-{attempt}",
                "documentId": work_item.source.documentId,
                "stage": "document_layout",
                "candidateAttempt": attempt,
                "providerId": "layout_fixture",
                "providerKind": "openai_responses",
                "model": "gpt-5.6-luna",
                "reasoningEffort": "max",
                "workerId": f"worker-document-layout-{attempt}",
                "startedAt": now,
                "completedAt": now,
                "durationMs": 1.0,
                "workItemSha256": kwargs["work_item_sha256"],
                "promptSha256": "2" * 64,
                "outputSha256": "3" * 64,
                "requests": 1,
                "responses": (response,),
                "inputTokens": 100,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "outputTokens": 20,
                "costUsd": None,
                "costStatus": "unavailable",
                "pdfAttachment": {
                    "sourcePdfSha256": pdf.source_pdf_sha256,
                    "sourcePageNumbers": pdf.source_page_numbers,
                    "attachmentPdfSha256": pdf.attachment_pdf_sha256,
                    "attachmentBytes": len(pdf.data),
                    "mediaType": pdf.media_type,
                    "constructionMethod": pdf.construction_method,
                },
            },
            strict=True,
        )
        return AgentCallResult(
            guidance,
            receipt,
        )


class _RepeatedReviewerDocumentRequestGateway(_ReviewerLayoutRetentionGateway):
    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        request = ReviewDocumentAssistanceRequest.model_validate(
            {
                "decision": "document_required",
                "pageNumbers": (1,),
                "ambiguity": "Resolve the flattened B/L-number heading association.",
                "affectedTargetPaths": ("documentPatch.billOfLadingNumber",),
                "rawOcrEvidence": (
                    {
                        "pageNumber": 1,
                        "rawValue": "HBL-001",
                        "ocrExcerpt": "B/L NO: HBL-001",
                    },
                ),
            },
            strict=True,
        )
        return AgentCallResult(
            request,
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _RepeatedExtractorDocumentRequestGateway(_ReviewerLayoutRetentionGateway):
    extract_calls: int

    def __init__(self) -> None:
        super().__init__()
        self.extract_calls = 0

    async def extract(
        self, work_item: AgentWorkItem, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        self.extract_calls += 1
        if self.extract_calls <= 2:
            output: BaseModel = DocumentAssistanceRequest.model_validate(
                {
                    "decision": "document_required",
                    "pageNumbers": (1,),
                    "ambiguity": "Resolve the flattened B/L-number heading association.",
                    "affectedTargetPaths": ("documentPatch.billOfLadingNumber",),
                    "rawOcrEvidence": (
                        {
                            "pageNumber": 1,
                            "rawValue": "HBL-001",
                            "ocrExcerpt": "B/L NO: HBL-001",
                        },
                    ),
                },
                strict=True,
            )
        else:
            output = _compact_annotation({"billOfLadingNumber": "HBL-001"})
        return AgentCallResult(
            output,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        return AgentCallResult(
            _review(passed=True),
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _ReceiptedFailureGateway(_RetryGateway):
    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        if attempt == 1:
            raise AgentCallFailure(
                "UnexpectedModelBehavior: structured output retries exhausted",
                receipt=_failed_receipt(
                    "extract", attempt, work_item_sha256=kwargs["work_item_sha256"]
                ),
                fatal_run=False,
                terminal_document=False,
            )
        return AgentCallResult(
            _annotation(),
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _TargetScopedRetryGateway(_RetryGateway):
    correction_paths_seen: tuple[str, ...] = ()

    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        if attempt == 1:
            output = _compact_annotation({"billOfLadingNumber": "HBL-001"})
        else:
            self.correction_paths_seen = tuple(kwargs["correction_paths"])
            output = _compact_correction(
                {"documentPatch.issueDate": "2024-04-28"}
            )
        return AgentCallResult(
            output,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        if attempt == 1:
            output = ReviewDraft.model_validate(
                {
                    "decision": "review",
                    "result": "fail",
                    "checks": {
                        "rawOcrTruthBoundary": "pass",
                        "evidenceIntegrity": "pass",
                        "semanticCompleteness": "fail",
                        "semanticCorrectness": "pass",
                        "contaminationAndRedundancy": "pass",
                        "documentUnitAndRelationships": "pass",
                    },
                    "findings": (
                        {
                            "severity": "blocking",
                            "category": "missing_field",
                            "message": "Add the explicit issue date.",
                            "targetPaths": ("documentPatch.issueDate",),
                            "rawOcrEvidence": (
                                {
                                    "pageNumber": 1,
                                    "rawValue": "28 APR 2024",
                                    "ocrExcerpt": "ISSUE DATE: 28 APR 2024",
                                },
                            ),
                            "imageUse": "not_used",
                        },
                    ),
                    "summary": "The explicit issue date is missing.",
                },
                strict=True,
            )
        else:
            assert isinstance(candidate, BillOfLadingDualCargoAnnotation)
            patch = candidate.relationExplicitLabel.documentPatch
            assert patch.billOfLadingNumber == "HBL-001"
            assert patch.issueDate == date(2024, 4, 28)
            output = _review(passed=True)
        return AgentCallResult(
            output,
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _NoOpCorrectionGateway(_RetryGateway):
    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        output: BaseModel = (
            _compact_correction({"documentPatch.billOfLadingNumber": "HBL-001"})
            if kwargs["correction_base"] is not None
            else _compact_annotation({"billOfLadingNumber": "HBL-001"})
        )
        return AgentCallResult(
            output,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        return AgentCallResult(
            _review(passed=False),
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _OscillatingCorrectionGateway(_RetryGateway):
    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        value = "HBL-001" if attempt in {1, 3} else "HBL-002"
        output: BaseModel = (
            _compact_correction({"documentPatch.billOfLadingNumber": value})
            if kwargs["correction_base"] is not None
            else _compact_annotation({"billOfLadingNumber": value})
        )
        return AgentCallResult(
            output,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        return AgentCallResult(
            _review(passed=False),
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _PartialCorrectionGateway(_RetryGateway):
    correction_paths_seen: list[tuple[str, ...]]

    def __init__(self) -> None:
        super().__init__()
        self.correction_paths_seen = []

    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        correction_paths = tuple(kwargs["correction_paths"])
        if not correction_paths:
            output: BaseModel = _compact_annotation(
                {"billOfLadingNumber": "HBL-001", "issueDate": None}
            )
        elif attempt == 2:
            self.correction_paths_seen.append(correction_paths)
            output = _compact_correction(
                {
                    "documentPatch.billOfLadingNumber": "HBL-001",
                    "documentPatch.issueDate": "2024-04-28",
                }
            )
        else:
            self.correction_paths_seen.append(correction_paths)
            output = _compact_correction(
                {"documentPatch.billOfLadingNumber": "HBL-002"}
            )
        return AgentCallResult(
            output,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.review_attempts.append(attempt)
        if attempt == 1:
            output = ReviewDraft.model_validate(
                {
                    "decision": "review",
                    "result": "fail",
                    "checks": {
                        "rawOcrTruthBoundary": "pass",
                        "evidenceIntegrity": "pass",
                        "semanticCompleteness": "fail",
                        "semanticCorrectness": "fail",
                        "contaminationAndRedundancy": "pass",
                        "documentUnitAndRelationships": "pass",
                    },
                    "findings": (
                        {
                            "severity": "blocking",
                            "category": "incorrect_field",
                            "message": "Use the primary headed bill number.",
                            "targetPaths": ("documentPatch.billOfLadingNumber",),
                            "rawOcrEvidence": (
                                {
                                    "pageNumber": 1,
                                    "rawValue": "HBL-002",
                                    "ocrExcerpt": "B/L NO: HBL-002",
                                },
                            ),
                            "imageUse": "not_used",
                        },
                        {
                            "severity": "blocking",
                            "category": "missing_field",
                            "message": "Add the explicit issue date.",
                            "targetPaths": ("documentPatch.issueDate",),
                            "rawOcrEvidence": (
                                {
                                    "pageNumber": 1,
                                    "rawValue": "28 APR 2024",
                                    "ocrExcerpt": "ISSUE DATE: 28 APR 2024",
                                },
                            ),
                            "imageUse": "not_used",
                        },
                    ),
                    "summary": "Two exact target corrections are required.",
                },
                strict=True,
            )
        else:
            assert isinstance(candidate, BillOfLadingDualCargoAnnotation)
            patch = candidate.relationExplicitLabel.documentPatch
            assert patch.billOfLadingNumber == "HBL-002"
            assert patch.issueDate == date(2024, 4, 28)
            output = _review(passed=True)
        return AgentCallResult(
            output,
            _receipt("review", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )


class _AmbiguousDateGateway(_RetryGateway):
    async def extract(self, work_item: AgentWorkItem, **kwargs: Any) -> AgentCallResult[BaseModel]:
        attempt = int(kwargs["candidate_attempt"])
        self.extract_attempts.append(attempt)
        warning = LabelWarning.model_validate(
            {
                "code": "ambiguous_ocr_candidates",
                "message": "The headed numeric date order is unresolved from raw OCR.",
                "pageNumbers": (1,),
                "targetPath": "documentPatch.shippedOnBoardDate",
            },
            strict=True,
        )
        output = _compact_annotation({"billOfLadingNumber": "HBL-001"}).model_copy(
            update={"warnings": (warning,)}
        )
        return AgentCallResult(
            output,
            _receipt("extract", attempt, work_item_sha256=kwargs["work_item_sha256"]),
        )

    async def review(
        self, work_item: AgentWorkItem, candidate: BaseModel, **kwargs: Any
    ) -> AgentCallResult[BaseModel]:
        raise AssertionError("training-truth ambiguity must be held before reviewer spend")


def test_luna_pricing_tracks_cache_buckets_and_long_context_multiplier() -> None:
    usage = RequestUsage(
        input_tokens=1_000_000,
        cache_read_tokens=100_000,
        cache_write_tokens=50_000,
        output_tokens=200_000,
    )
    assert _price_request(usage, _pricing()) == Decimal("0.729000000000")


def test_luna_pricing_long_context_boundary_is_strictly_above_threshold() -> None:
    pricing = _pricing()
    at_boundary = RequestUsage(input_tokens=272_000, output_tokens=10)
    above_boundary = RequestUsage(input_tokens=272_001, output_tokens=10)
    assert _price_request(at_boundary, pricing) == Decimal("0.054412000000")
    assert _price_request(above_boundary, pricing) == Decimal("0.108818400000")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0.5", (0, 5, 0)), ("0.5.7", (0, 5, 7)), ("1.2.3-rc1", (1, 2, 3))],
)
def test_ollama_version_parser_accepts_numeric_release_prefix(
    raw: str, expected: tuple[int, int, int]
) -> None:
    assert _parse_ollama_version(raw) == expected


@pytest.mark.parametrize("raw", ["", "v0.5.0", "current"])
def test_ollama_version_parser_rejects_malformed_values(raw: str) -> None:
    with pytest.raises(AgentProviderError, match="invalid version"):
        _parse_ollama_version(raw)


def test_gateway_constructs_native_schema_agents_for_ollama_without_network(
    tmp_path: Path,
) -> None:
    gateway = PydanticAgentGateway(
        _config(tmp_path),
        project_root=tmp_path,
        extractor_prompt="extract",
        reviewer_prompt="review",
        document_prompt="layout",
    )
    assert gateway.config.assignments.labeler == "ollama_test"
    for agent in (gateway._extractor, gateway._reviewer, gateway._document_agent):
        processor = agent._output_schema.text_processor
        assert processor is not None
        assert processor.object_def.strict is True


def test_schema_v4_selects_compact_extraction_and_review_contracts(tmp_path: Path) -> None:
    value = _config(tmp_path).model_dump(mode="python")
    value.update(
        {
            "schema_version": 4,
            "task": "bill_of_lading_relation_single_source_v4",
        }
    )
    config = AgentLabelingConfig.model_validate(value, strict=True)

    gateway = PydanticAgentGateway(
        config,
        project_root=tmp_path,
        extractor_prompt="extract",
        reviewer_prompt="review",
        document_prompt="layout",
    )

    assert gateway._extraction_parser is _parse_compact_extraction_wire
    extractor_schema = json.dumps(
        gateway._extractor._output_schema.text_processor.object_def.json_schema
    )
    corrector = gateway._corrector_for_paths(("documentPatch.issueDate",))
    corrector_schema = json.dumps(
        corrector._output_schema.text_processor.object_def.json_schema
    )
    assert "relationExplicitLabel" in extractor_schema
    assert "normalLabel" not in extractor_schema
    assert '"evidence"' not in extractor_schema
    assert "ocrExcerpt" not in extractor_schema
    assert "documentPatch.issueDate" in corrector_schema
    assert "relationExplicitLabel" not in corrector_schema
    assert "warnings" not in corrector_schema
    reviewer_schema = json.dumps(
        gateway._reviewer._output_schema.text_processor.object_def.json_schema
    )
    layout_schema = json.dumps(
        gateway._document_agent._output_schema.text_processor.object_def.json_schema
    )
    assert "ocrExcerpt" not in reviewer_schema
    assert "ocrExcerpt" not in layout_schema


def test_exact_path_corrector_schema_is_openai_strict_and_scope_confined(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path).model_dump(mode="python")
    value.update({"schema_version": 4, "task": "bill_of_lading_relation_single_source_v4"})
    gateway = PydanticAgentGateway(
        AgentLabelingConfig.model_validate(value, strict=True),
        project_root=tmp_path,
        extractor_prompt="extract",
        reviewer_prompt="review",
        document_prompt="layout",
    )
    paths = (
        "documentPatch.cargoGroups[0].description",
        "documentPatch.parties.shipper.contactDetails.phoneNumbers[1]",
        "documentType",
    )
    normalized = normalize_correction_paths(paths)
    corrector = gateway._corrector_for_paths(paths)
    assert corrector is gateway._corrector_for_paths(tuple(reversed(normalized)))
    processor = corrector._output_schema.text_processor
    assert processor is not None

    strict_schema = OpenAIJsonSchemaTransformer(
        processor.object_def.json_schema,
        strict=True,
    ).walk()
    assert strict_schema["required"] == ["corrections"]
    assert strict_schema["additionalProperties"] is False
    corrections = strict_schema["properties"]["corrections"]
    assert corrections["required"] == sorted(normalized)
    assert tuple(corrections["properties"]) == tuple(sorted(normalized))
    assert corrections["additionalProperties"] is False
    assert "documentPatch.parties.shipper.contactDetails.phoneNumbers" in normalized
    assert "documentPatch.parties.shipper.contactDetails.phoneNumbers[1]" not in normalized
    assert corrections["properties"]["documentType"]["enum"] == [
        "bill_of_lading",
        "sea_waybill",
    ]
    phone_schema = corrections["properties"][
        "documentPatch.parties.shipper.contactDetails.phoneNumbers"
    ]
    phone_array = next(branch for branch in phone_schema["anyOf"] if branch["type"] == "array")
    assert phone_array["items"] == {"type": "string"}
    description_schema = corrections["properties"][
        "documentPatch.cargoGroups[0].description"
    ]
    assert {branch["type"] for branch in description_schema["anyOf"]} == {"string", "null"}
    assert "product/goods wording" in description_schema["description"]


def test_exact_path_schema_distinguishes_explicit_null_from_omission() -> None:
    schema = correction_envelope_json_schema(("documentPatch.route",))
    wire = _compact_correction_wire(("documentPatch.route",))
    assert schema["properties"]["corrections"]["required"] == ["documentPatch.route"]

    strict_schema = OpenAIJsonSchemaTransformer(
        TypeAdapter(wire).json_schema(mode="serialization"),
        strict=True,
    ).walk()
    route_schema = strict_schema["properties"]["corrections"]["properties"][
        "documentPatch.route"
    ]
    assert {branch.get("type") for branch in route_schema["anyOf"]} >= {"object", "null"}


@pytest.mark.asyncio
async def test_exact_path_gateway_retains_and_returns_raw_correction_envelope(
    tmp_path: Path,
) -> None:
    value = _config(tmp_path).model_dump(mode="python")
    value.update({"schema_version": 4, "task": "bill_of_lading_relation_single_source_v4"})
    gateway = PydanticAgentGateway(
        AgentLabelingConfig.model_validate(value, strict=True),
        project_root=tmp_path,
        extractor_prompt="extract",
        reviewer_prompt="review",
        document_prompt="layout",
    )
    raw_envelope = {"corrections": {"documentPatch.issueDate": "2024-04-28"}}

    def respond(_: Any, __: Any) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(json.dumps(raw_envelope))],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
            model_name="offline-function-model",
        )

    provider_id = gateway.config.assignments.labeler
    runtime = gateway._runtimes[provider_id]
    gateway._runtimes[provider_id] = replace(
        runtime,
        model=FunctionModel(
            respond,
            profile=ModelProfile(
                supports_json_schema_output=True,
                default_structured_output_mode="native",
            ),
        ),
    )
    item = _work_item_with_text("B/L NO: HBL-001\nISSUE DATE: 28 APR 2024")
    base = _compact_annotation({"billOfLadingNumber": "HBL-001", "issueDate": None})

    result = await gateway.extract(
        item,
        work_item_sha256="a" * 64,
        candidate_attempt=2,
        retry_feedback="Correct the issue date.",
        layout_guidance=None,
        correction_base=base,
        correction_paths=("documentPatch.issueDate",),
    )

    assert isinstance(result.output, CompactCorrectionEnvelope)
    assert result.output.corrections == {"documentPatch.issueDate": "2024-04-28"}
    raw_model = CompactCorrectionEnvelope.model_validate(raw_envelope, strict=True)
    assert result.receipt.outputSha256 == sha256_bytes(
        canonical_json_bytes(raw_model.model_dump(mode="json"))
    )
    assert result.receipt.transcript is not None
    transcript = json.dumps(result.receipt.transcript.model_dump(mode="json"))
    assert "documentPatch.issueDate" in transcript
    assert "2024-04-28" in transcript


def test_agent_config_rejects_mismatched_schema_and_task(tmp_path: Path) -> None:
    value = _config(tmp_path).model_dump(mode="python")
    value["schema_version"] = 4

    with pytest.raises(ValueError, match="requires task"):
        AgentLabelingConfig.model_validate(value, strict=True)


def test_gateway_constructs_luna_max_responses_agents_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _config(tmp_path).model_dump(mode="python")
    value["providers"] = [
        {
            "id": "luna_max",
            "kind": "openai_responses",
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "max",
            "supports_pdf_documents": True,
            "request_timeout_seconds": 300.0,
            "transport_max_retries": 2,
            "max_output_tokens": 16384,
            "pricing": _pricing().model_dump(mode="python"),
        }
    ]
    value["assignments"] = {
        "labeler": "luna_max",
        "reviewer": "luna_max",
        "document_layout": "luna_max",
    }
    config = AgentLabelingConfig.model_validate(value, strict=True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-used-only-for-offline-construction")

    gateway = PydanticAgentGateway(
        config,
        project_root=tmp_path,
        extractor_prompt="extract",
        reviewer_prompt="review",
        document_prompt="layout",
    )

    provider = gateway.config.providers[0]
    assert isinstance(provider, OpenAIResponsesProviderConfig)
    assert provider.model == "gpt-5.6-luna"
    assert provider.reasoning_effort == "max"


@pytest.mark.parametrize(
    ("model", "reasoning_effort"),
    [("gpt-5.6-luna", "high"), ("gpt-5.6-terra", "medium")],
)
def test_openai_comparison_models_and_efforts_are_supported(
    tmp_path: Path, model: str, reasoning_effort: str
) -> None:
    value = _config(tmp_path).model_dump(mode="python")
    value["providers"] = [
        {
            "id": "comparison_provider",
            "kind": "openai_responses",
            "model": model,
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": reasoning_effort,
            "supports_pdf_documents": True,
            "request_timeout_seconds": 300.0,
            "transport_max_retries": 2,
            "max_output_tokens": 32768,
            "pricing": _pricing().model_dump(mode="python"),
        }
    ]
    value["assignments"] = {
        "labeler": "comparison_provider",
        "reviewer": "comparison_provider",
        "document_layout": "comparison_provider",
    }

    config = AgentLabelingConfig.model_validate(value, strict=True)

    provider = config.providers[0]
    assert isinstance(provider, OpenAIResponsesProviderConfig)
    assert provider.model == model
    assert provider.reasoning_effort == reasoning_effort


def test_selection_is_deterministic_and_independent_of_input_order(tmp_path: Path) -> None:
    config = _config(tmp_path)
    one = _work_item()
    alternate_item = one.item.model_copy(
        update={"source": one.item.source.model_copy(update={"documentId": "doc_" + "9" * 64})}
    )
    two = InventoriedWorkItem("fixture", Path("/alternate.json"), "8" * 64, alternate_item)
    selected_forward = select_work_items(config, (one, two))
    selected_reverse = select_work_items(config, (two, one))
    assert [row.item.source.documentId for row in selected_forward] == [
        row.item.source.documentId for row in selected_reverse
    ]


def test_explicit_selection_is_exact_ordered_and_count_bound(tmp_path: Path) -> None:
    config_value = _config(tmp_path).model_dump(mode="python")
    one = _work_item()
    alternate_id = "doc_" + "9" * 64
    alternate_item = one.item.model_copy(
        update={"source": one.item.source.model_copy(update={"documentId": alternate_id})}
    )
    two = InventoriedWorkItem("fixture", Path("/alternate.json"), "8" * 64, alternate_item)
    config_value["source"]["expected_documents"] = 2
    config_value["source"]["expected_pages"] = 2
    config_value["source"]["work_item_roots"][0]["documents"] = 2
    config_value["selection"].update(
        {"count": 2, "document_ids": [alternate_id, one.item.source.documentId]}
    )
    config = AgentLabelingConfig.model_validate(config_value, strict=True)

    selected = select_work_items(config, (one, two))

    assert [row.item.source.documentId for row in selected] == [
        alternate_id,
        one.item.source.documentId,
    ]

    config_value["selection"]["count"] = 1
    with pytest.raises(ValueError, match="length must equal"):
        AgentLabelingConfig.model_validate(config_value, strict=True)


def test_pinned_selection_file_is_exact_ordered_and_hash_bound(tmp_path: Path) -> None:
    config_value = _config(tmp_path).model_dump(mode="python")
    one = _work_item()
    alternate_id = "doc_" + "9" * 64
    alternate_item = one.item.model_copy(
        update={"source": one.item.source.model_copy(update={"documentId": alternate_id})}
    )
    two = InventoriedWorkItem("fixture", Path("/alternate.json"), "8" * 64, alternate_item)
    payload = b"".join(
        canonical_json_bytes({"documentId": value}) + b"\n"
        for value in (alternate_id, one.item.source.documentId)
    )
    selection_path = tmp_path / "selection.jsonl"
    selection_path.write_bytes(payload)
    config_value["source"]["expected_documents"] = 2
    config_value["source"]["expected_pages"] = 2
    config_value["source"]["work_item_roots"][0]["documents"] = 2
    config_value["selection"] = {
        "count": 2,
        "seed": 8127,
        "namespace": "pinned-file-fixture",
        "document_ids_file": {
            "path": str(selection_path),
            "sha256": sha256_bytes(payload),
            "records": 2,
        },
    }
    config = AgentLabelingConfig.model_validate(config_value, strict=True)

    selected = select_work_items(config, (one, two))

    assert [row.item.source.documentId for row in selected] == [
        alternate_id,
        one.item.source.documentId,
    ]

    selection_path.write_bytes(payload + b"\n")
    with pytest.raises(WorkItemError, match="SHA-256 differs"):
        select_work_items(config, (one, two))


def test_selection_rejects_inline_and_file_ids_together(tmp_path: Path) -> None:
    config_value = _config(tmp_path).model_dump(mode="python")
    config_value["selection"]["document_ids"] = [_work_item().item.source.documentId]
    config_value["selection"]["document_ids_file"] = {
        "path": str(tmp_path / "selection.jsonl"),
        "sha256": "a" * 64,
        "records": 1,
    }

    with pytest.raises(ValueError, match="document_ids or document_ids_file"):
        AgentLabelingConfig.model_validate(config_value, strict=True)


@pytest.mark.parametrize(
    "hold_code",
    (
        "ambiguous_numeric_date_order",
        "audited_label_defect",
        "evidence_revalidation_required",
    ),
)
def test_consolidation_uses_latest_source_and_holds_ambiguous_truth_out_of_training(
    tmp_path: Path, hold_code: str,
) -> None:
    item = _work_item_with_text("B/L NO: HBL-001\nB/L NO: HBL-002")
    document_id = item.source.documentId
    work_item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    selection_root = tmp_path / "selection-run"
    (selection_root / "work-items").mkdir(parents=True)
    (selection_root / "work-items" / f"{document_id}.json").write_bytes(work_item_payload)
    selection_payload = canonical_json_bytes(
        {
            "documentId": document_id,
            "documentPageCount": 1,
            "workItemSha256": sha256_bytes(work_item_payload),
        }
    ) + b"\n"
    (selection_root / "selection.jsonl").write_bytes(selection_payload)

    source_configs: list[dict[str, str]] = []
    for index, number in enumerate(("HBL-001", "HBL-002"), start=1):
        run_id = f"source-{index}"
        run_root = tmp_path / run_id
        (run_root / "state" / "outcomes").mkdir(parents=True)
        (run_root / "validated").mkdir()
        (run_root / "manifest.json").write_text(
            json.dumps({"run_id": run_id}), encoding="utf-8"
        )
        (run_root / "resolved-config.json").write_text(
            json.dumps({"run": {"run_id": run_id}}), encoding="utf-8"
        )
        annotation = build_compact_annotation(
            item,
            _compact_annotation({"billOfLadingNumber": number}),
            pdf_grouping_used=False,
        ).model_copy(update={"reviewStatus": "validated"})
        annotation_payload = canonical_json_bytes(annotation.model_dump(mode="json"))
        artifact_path = f"validated/{number}.json"
        (run_root / artifact_path).write_bytes(annotation_payload)
        outcome = {
            "schemaVersion": 1,
            "documentId": document_id,
            "status": "validated",
            "attempts": 1,
            "reviews": 1,
            "documentEscalations": 0,
            "finalArtifactPath": artifact_path,
            "finalArtifactSha256": sha256_bytes(annotation_payload),
            "callReceiptPaths": [],
            "callReceiptSha256s": [],
        }
        (run_root / "state" / "outcomes" / f"{document_id}.json").write_bytes(
            canonical_json_bytes(outcome)
        )
        source_configs.append({"run_id": run_id, "root": str(run_root)})

    output_root = tmp_path / "output"
    output_root.mkdir()
    config = ConsolidationConfig.model_validate(
        {
            "schema_version": 1,
            "dataset_id": "consolidated-test-v1",
            "output_root": str(output_root),
            "selection_run_root": str(selection_root),
            "selection_sha256": sha256_bytes(selection_payload),
            "expected_documents": 1,
            "source_runs": source_configs,
            "quality_holds": [
                {
                    "document_id": document_id,
                    "code": hold_code,
                    "message": "The printed numeric date order has no internal format evidence.",
                    "raw_values": ["HBL-002"],
                }
            ],
        },
        strict=True,
    )

    manifest_path = consolidate_labeling_runs(config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lineage = json.loads((manifest_path.parent / "lineage.jsonl").read_text(encoding="utf-8"))
    assert manifest["outcomes"] == {"excluded": 0, "needs_review": 1, "validated": 0}
    assert manifest["trainingRecords"] == 0
    assert manifest["trainingReady"] is False
    assert lineage["selectedSourceRunId"] == "source-2"
    assert len(lineage["consideredSourceOutcomes"]) == 2
    assert (manifest_path.parent / "needs-review" / f"{document_id}.json").is_file()


def test_remaining1779_test30_config_is_the_full_run_selection_prefix() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "configs" / "labeling_agents"
    full = load_agent_labeling_config(config_root / "mpci_bl_dual_cargo_v3_luna_remaining1779.yaml")
    pilot = load_agent_labeling_config(
        config_root / "mpci_bl_dual_cargo_v3_luna_remaining1779_test30.yaml"
    )

    assert pilot.run.run_id != full.run.run_id
    assert pilot.source == full.source
    assert pilot.prompts == full.prompts
    assert pilot.providers == full.providers
    assert pilot.assignments == full.assignments
    assert pilot.workflow == full.workflow
    assert pilot.selection.count == 30
    assert full.selection.count == 1779
    assert pilot.selection.seed == full.selection.seed
    assert pilot.selection.namespace == full.selection.namespace


def test_remaining1779_max128k_pilot_changes_only_output_budget_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "configs" / "labeling_agents"
    capped_pilot = load_agent_labeling_config(
        config_root / "mpci_bl_dual_cargo_v3_luna_remaining1779_test30.yaml"
    )
    raised_pilot = load_agent_labeling_config(
        config_root / "mpci_bl_dual_cargo_v3_luna_remaining1779_test30_max128k.yaml"
    )

    assert raised_pilot.run.run_id != capped_pilot.run.run_id
    assert raised_pilot.source == capped_pilot.source
    assert raised_pilot.selection == capped_pilot.selection
    assert raised_pilot.prompts == capped_pilot.prompts
    assert raised_pilot.assignments == capped_pilot.assignments
    capped_provider = capped_pilot.providers[0]
    raised_provider = raised_pilot.providers[0]
    assert isinstance(capped_provider, OpenAIResponsesProviderConfig)
    assert isinstance(raised_provider, OpenAIResponsesProviderConfig)
    assert raised_provider.model == capped_provider.model == "gpt-5.6-luna"
    assert raised_provider.pricing == capped_provider.pricing
    assert raised_provider.reasoning_effort == capped_provider.reasoning_effort == "max"
    assert capped_provider.max_output_tokens == 32768
    assert raised_provider.max_output_tokens == 128000
    assert raised_pilot.workflow.output_tokens_limit_per_agent_run == 256000
    assert raised_pilot.workflow.max_concurrent_documents == 8
    assert raised_pilot.workflow.max_concurrent_model_requests == 8


def test_pdf_assistance_is_hash_verified_page_scoped_and_deterministic(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    row = _work_item_with_pdf(tmp_path)

    first = pdf_bytes_for_pages(config, row.item, (1,))
    second = pdf_bytes_for_pages(config, row.item, (1,))

    assert first.source_pdf_sha256 == row.item.source.sourceSha256
    assert first.source_page_numbers == (1,)
    assert first.attachment_pdf_sha256 == sha256_bytes(first.data)
    assert first.attachment_pdf_sha256 == second.attachment_pdf_sha256
    assert first.construction_method == "pypdf_strict"
    assert len(PdfReader(BytesIO(first.data), strict=True).pages) == 1


def test_pdf_assistance_decrypts_empty_password_and_receipts_the_method(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    row = _work_item_with_pdf(tmp_path, empty_password_encrypted=True)

    payload = pdf_bytes_for_pages(config, row.item, (1,))

    assert payload.construction_method == "pypdf_strict_empty_password"
    derived_reader = PdfReader(BytesIO(payload.data), strict=True)
    assert derived_reader.is_encrypted is False
    assert len(derived_reader.pages) == 1


def test_pdf_assistance_recovers_malformed_xref_and_receipts_the_method(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    row = _work_item_with_pdf(tmp_path, broken_xref=True)

    payload = pdf_bytes_for_pages(config, row.item, (1,))

    assert payload.construction_method == "pypdf_recovery"
    assert len(PdfReader(BytesIO(payload.data), strict=True).pages) == 1


def test_pdf_assistance_rejects_source_identity_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    row = _work_item_with_pdf(tmp_path)
    Path(row.item.source.localCanonicalPath).write_bytes(b"%PDF-tampered")

    with pytest.raises(WorkItemError, match="SHA-256"):
        pdf_bytes_for_pages(config, row.item, (1,))


@pytest.mark.asyncio
async def test_failed_review_uses_a_fresh_attempt_and_preserves_attempt_history(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = agent_run_paths(config)
    gateway = _RetryGateway()
    outcome = await process_document(config, paths, gateway, _work_item())

    assert outcome.status == "validated"
    assert outcome.attempts == 2
    assert outcome.reviews == 2
    assert gateway.extract_attempts == [1, 2]
    assert gateway.review_attempts == [1, 2]
    attempts = paths.state_root / "candidate-attempts" / outcome.documentId
    assert sorted(path.name for path in attempts.iterdir()) == ["attempt-1", "attempt-2"]
    assert len(outcome.callReceiptPaths) == 4


@pytest.mark.asyncio
async def test_no_op_correction_is_held_without_paying_for_another_review(
    tmp_path: Path,
) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(mode="python")
    config_value.update(
        {"schema_version": 4, "task": "bill_of_lading_relation_single_source_v4"}
    )
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    paths = agent_run_paths(config)
    gateway = _NoOpCorrectionGateway()

    outcome = await process_document(config, paths, gateway, _work_item())

    assert outcome.status == "needs_review"
    assert outcome.attempts == 2
    assert outcome.reviews == 1
    assert gateway.extract_attempts == [1, 2]
    assert gateway.review_attempts == [1]
    record = NeedsReviewRecord.model_validate_json(
        (paths.run_root / outcome.finalArtifactPath).read_bytes(), strict=True
    )
    assert record.reason == "correction_non_convergence"
    assert "left these reviewer-authorized targets unchanged" in record.findings[0]


@pytest.mark.asyncio
async def test_partial_correction_is_retained_and_only_unchanged_paths_are_retried(
    tmp_path: Path,
) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(mode="python")
    config_value.update(
        {"schema_version": 4, "task": "bill_of_lading_relation_single_source_v4"}
    )
    config_value["workflow"]["max_candidate_attempts"] = 3
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    item = _work_item_with_text(
        "B/L NO: HBL-002\nISSUE DATE: 28 APR 2024\nSECOND PRINT: HBL-001"
    )
    payload = canonical_json_bytes(item.model_dump(mode="json"))
    row = InventoriedWorkItem("fixture", Path("/fixture.json"), sha256_bytes(payload), item)
    paths = agent_run_paths(config)
    gateway = _PartialCorrectionGateway()

    outcome = await process_document(config, paths, gateway, row)

    assert outcome.status == "validated"
    assert outcome.attempts == 3
    assert outcome.reviews == 2
    assert gateway.correction_paths_seen == [
        ("documentPatch.billOfLadingNumber", "documentPatch.issueDate"),
        ("documentPatch.billOfLadingNumber",),
    ]
    second = next(
        (paths.state_root / "target-scoped-corrections" / outcome.documentId / "attempt-2").glob(
            "*.json"
        )
    )
    second_value = CompactAnnotationDraft.model_validate_json(second.read_bytes(), strict=True)
    assert second_value.relationExplicitLabel.documentPatch.issueDate == date(2024, 4, 28)


@pytest.mark.asyncio
async def test_oscillating_correction_is_held_before_re_reviewing_rejected_candidate(
    tmp_path: Path,
) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(mode="python")
    config_value.update(
        {"schema_version": 4, "task": "bill_of_lading_relation_single_source_v4"}
    )
    config_value["workflow"]["max_candidate_attempts"] = 3
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    item = _work_item_with_text("B/L NO: HBL-001\nSECOND PRINT: HBL-002")
    payload = canonical_json_bytes(item.model_dump(mode="json"))
    row = InventoriedWorkItem("fixture", Path("/fixture.json"), sha256_bytes(payload), item)
    paths = agent_run_paths(config)
    gateway = _OscillatingCorrectionGateway()

    outcome = await process_document(config, paths, gateway, row)

    assert outcome.status == "needs_review"
    assert outcome.attempts == 3
    assert outcome.reviews == 2
    assert gateway.extract_attempts == [1, 2, 3]
    assert gateway.review_attempts == [1, 2]
    record = NeedsReviewRecord.model_validate_json(
        (paths.run_root / outcome.finalArtifactPath).read_bytes(), strict=True
    )
    assert record.reason == "correction_non_convergence"
    assert "identical candidate previously rejected" in record.findings[0]


@pytest.mark.asyncio
async def test_reviewer_layout_guidance_is_retained_for_the_correction_attempt(
    tmp_path: Path,
) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(
        mode="python"
    )
    config_value.update(
        {
            "schema_version": 4,
            "task": "bill_of_lading_relation_single_source_v4",
        }
    )
    config_value["providers"] = [
        {
            "id": "layout_fixture",
            "kind": "openai_responses",
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "max",
            "supports_pdf_documents": True,
            "request_timeout_seconds": 30.0,
            "transport_max_retries": 2,
            "max_output_tokens": 4096,
            "pricing": _pricing().model_dump(mode="python"),
        }
    ]
    config_value["assignments"] = {
        "labeler": "layout_fixture",
        "reviewer": "layout_fixture",
        "document_layout": "layout_fixture",
    }
    config_value["workflow"]["max_document_escalations_per_document"] = 1
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    row = _work_item_with_pdf(tmp_path)
    paths = agent_run_paths(config)
    gateway = _ReviewerLayoutRetentionGateway()

    outcome = await process_document(config, paths, gateway, row)

    assert outcome.status == "validated"
    assert outcome.documentEscalations == 1
    assert outcome.attempts == 2
    assert gateway.correction_layout_guidance is not None
    assert (
        gateway.correction_layout_guidance.observations[0].guidance
        == "HBL-001 is positioned in the B/L-number box."
    )


@pytest.mark.asyncio
async def test_repeated_reviewer_document_request_becomes_a_document_hold(
    tmp_path: Path,
) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(
        mode="python"
    )
    config_value.update(
        {
            "schema_version": 4,
            "task": "bill_of_lading_relation_single_source_v4",
        }
    )
    config_value["providers"] = [
        {
            "id": "layout_fixture",
            "kind": "openai_responses",
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "max",
            "supports_pdf_documents": True,
            "request_timeout_seconds": 30.0,
            "transport_max_retries": 2,
            "max_output_tokens": 4096,
            "pricing": _pricing().model_dump(mode="python"),
        }
    ]
    config_value["assignments"] = {
        "labeler": "layout_fixture",
        "reviewer": "layout_fixture",
        "document_layout": "layout_fixture",
    }
    config_value["workflow"]["max_document_escalations_per_document"] = 1
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    paths = agent_run_paths(config)
    gateway = _RepeatedReviewerDocumentRequestGateway()

    outcome = await process_document(config, paths, gateway, _work_item_with_pdf(tmp_path))

    assert outcome.status == "needs_review"
    assert outcome.attempts == 1
    assert outcome.reviews == 2
    assert outcome.documentEscalations == 1
    assert gateway.review_attempts == [1, 1]
    hold = NeedsReviewRecord.model_validate_json(
        (paths.run_root / outcome.finalArtifactPath).read_bytes(), strict=True
    )
    assert hold.reason == "document_escalation_unavailable"
    assert hold.findings == ("Resolve the flattened B/L-number heading association.",)


@pytest.mark.asyncio
async def test_repeated_extractor_document_requests_use_the_configured_escalation_budget(
    tmp_path: Path,
) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(
        mode="python"
    )
    config_value.update(
        {
            "schema_version": 4,
            "task": "bill_of_lading_relation_single_source_v4",
        }
    )
    config_value["providers"] = [
        {
            "id": "layout_fixture",
            "kind": "openai_responses",
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "max",
            "supports_pdf_documents": True,
            "request_timeout_seconds": 30.0,
            "transport_max_retries": 2,
            "max_output_tokens": 4096,
            "pricing": _pricing().model_dump(mode="python"),
        }
    ]
    config_value["assignments"] = {
        "labeler": "layout_fixture",
        "reviewer": "layout_fixture",
        "document_layout": "layout_fixture",
    }
    config_value["workflow"]["max_document_escalations_per_document"] = 2
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    paths = agent_run_paths(config)
    gateway = _RepeatedExtractorDocumentRequestGateway()

    outcome = await process_document(config, paths, gateway, _work_item_with_pdf(tmp_path))

    assert outcome.status == "validated"
    assert outcome.attempts == 1
    assert outcome.reviews == 1
    assert outcome.documentEscalations == 2
    assert gateway.extract_attempts == [1, 1, 1]
    assert gateway.review_attempts == [1]
    guidance_root = paths.state_root / "layout-guidance" / outcome.documentId
    assert sorted(path.name for path in guidance_root.iterdir()) == [
        "attempt-1-escalation-1",
        "attempt-1-escalation-2",
    ]


@pytest.mark.asyncio
async def test_reviewer_retry_merges_only_targeted_fields(tmp_path: Path) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(mode="python")
    config_value.update(
        {
            "schema_version": 4,
            "task": "bill_of_lading_relation_single_source_v4",
        }
    )
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    item = _work_item_with_text("B/L NO: HBL-001\nISSUE DATE: 28 APR 2024")
    item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    row = InventoriedWorkItem(
        "fixture",
        Path("/fixture.json"),
        sha256_bytes(item_payload),
        item,
    )
    paths = agent_run_paths(config)
    gateway = _TargetScopedRetryGateway()

    outcome = await process_document(config, paths, gateway, row)

    assert outcome.status == "validated"
    assert outcome.attempts == 2
    assert gateway.correction_paths_seen == ("documentPatch.issueDate",)
    assert (
        paths.state_root / "target-scoped-corrections" / outcome.documentId / "attempt-2"
    ).is_dir()


@pytest.mark.asyncio
async def test_stale_ambiguous_date_warning_retries_instead_of_direct_hold(
    tmp_path: Path,
) -> None:
    config_value = _config(tmp_path, reference_target_mode="absent").model_dump(mode="python")
    config_value.update(
        {
            "schema_version": 4,
            "task": "bill_of_lading_relation_single_source_v4",
        }
    )
    config = AgentLabelingConfig.model_validate(config_value, strict=True)
    item = _work_item_with_text("B/L NO: HBL-001\nSHIPPED ON BOARD\n01/04/2024")
    payload = canonical_json_bytes(item.model_dump(mode="json"))
    row = InventoriedWorkItem("fixture", Path("/fixture.json"), sha256_bytes(payload), item)
    paths = agent_run_paths(config)
    gateway = _AmbiguousDateGateway()

    outcome = await process_document(config, paths, gateway, row)

    assert outcome.status == "needs_review"
    assert outcome.attempts == 2
    assert outcome.reviews == 0
    assert gateway.extract_attempts == [1, 2]
    record = json.loads((paths.run_root / outcome.finalArtifactPath).read_text())
    assert record["reason"] == "candidate_attempts_exhausted"
    assert "day-first ambiguity policy" in " ".join(record["findings"])


@pytest.mark.asyncio
async def test_document_request_does_not_load_pdf_when_escalation_is_disabled(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, document_escalations=0)
    paths = agent_run_paths(config)
    outcome = await process_document(config, paths, _DocumentRequestGateway(), _work_item())

    assert outcome.status == "needs_review"
    assert outcome.documentEscalations == 0
    assert outcome.reviews == 0
    assert len(outcome.callReceiptPaths) == 1
    manifest = json.loads(publish_agent_run(config, (_work_item(),)).read_text())
    quality = json.loads((paths.run_root / "reference-quality-by-document.jsonl").read_text())
    assert quality["canonicalExactMatch"] is False
    assert quality["schemaValid"] is False
    assert quality["fieldValueRecall"] == 0.0
    assert manifest["reference_quality"]["all_selected"]["schema_valid"] == 0.0
    assert manifest["reference_quality"]["validated_only"] is None


@pytest.mark.asyncio
async def test_failed_structured_output_is_receipted_and_included_before_fresh_retry(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = agent_run_paths(config)
    gateway = _ReceiptedFailureGateway()

    outcome = await process_document(config, paths, gateway, _work_item())

    assert outcome.status == "validated"
    assert outcome.attempts == 2
    assert gateway.extract_attempts == [1, 2]
    assert len(outcome.callReceiptPaths) == 3
    first_receipt = AgentCallReceipt.model_validate_json(
        (paths.run_root / outcome.callReceiptPaths[0]).read_bytes(), strict=True
    )
    assert first_receipt.status == "error"
    assert first_receipt.errorType == "UnexpectedModelBehavior"
    assert first_receipt.inputTokens == 100
    assert first_receipt.outputTokens == 20


@pytest.mark.asyncio
async def test_publication_reports_complete_per_document_and_run_local_usage(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = agent_run_paths(config)
    row = _work_item()
    await process_document(config, paths, _RetryGateway(), row)

    manifest_path = publish_agent_run(config, (row,))
    manifest = json.loads(manifest_path.read_text())
    usage = json.loads((paths.run_root / "usage-by-document.jsonl").read_text())

    assert usage["calls"] == 4
    assert usage["outcomeReferencedCalls"] == 4
    assert usage["orphanedCalls"] == 0
    assert usage["localCalls"] == 4
    assert usage["unpricedOpenAiCalls"] == 0
    assert usage["configuredListPriceEstimateComplete"] is True
    assert usage["costBasis"] == "configured_public_list_rates_not_billed_cost"
    assert manifest["usage"]["call_receipts"] == 4
    assert manifest["usage"]["outcome_referenced_call_receipts"] == 4
    assert manifest["usage"]["orphaned_call_receipts"] == 0
    assert manifest["usage"]["local_call_receipts"] == 4
    assert manifest["usage"]["unpriced_openai_call_receipts"] == 0
    assert manifest["usage"]["configured_list_price_estimate_complete"] is True
    assert manifest["usage"]["cost_basis"] == "configured_public_list_rates_not_billed_cost"
    quality = json.loads((paths.run_root / "reference-quality-by-document.jsonl").read_text())
    assert quality["canonicalExactMatch"] is True
    assert quality["fieldValueF1"] == 1.0
    assert manifest["reference_quality"]["all_selected"]["canonical_exact_match"] == 1.0
    assert manifest["reference_quality"]["validated_only"]["field_value_f1"] == 1.0


@pytest.mark.asyncio
async def test_publication_accounts_for_receipt_orphaned_by_an_interrupted_attempt(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = agent_run_paths(config)
    row = _work_item()
    outcome = await process_document(config, paths, _RetryGateway(), row)
    orphan = _receipt("extract", 1, work_item_sha256=row.sha256).model_copy(
        update={
            "callId": "call-orphaned-extract-1",
            "workerId": "worker-orphaned-extract-1",
        }
    )
    orphan_path, _ = _publish_receipt(paths, orphan)
    assert orphan_path not in outcome.callReceiptPaths

    manifest = json.loads(publish_agent_run(config, (row,)).read_text())
    usage = json.loads((paths.run_root / "usage-by-document.jsonl").read_text())

    assert usage["calls"] == 5
    assert usage["outcomeReferencedCalls"] == 4
    assert usage["orphanedCalls"] == 1
    assert manifest["usage"]["call_receipts"] == 5
    assert manifest["usage"]["outcome_referenced_call_receipts"] == 4
    assert manifest["usage"]["orphaned_call_receipts"] == 1


@pytest.mark.asyncio
async def test_interrupted_publication_retains_in_flight_receipts_and_frozen_scope(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = agent_run_paths(config)
    terminal_row = _work_item()
    await process_document(config, paths, _RetryGateway(), terminal_row)

    document_id = "doc_" + "b" * 64
    source = terminal_row.item.source.model_copy(update={"documentId": document_id})
    item = terminal_row.item.model_copy(update={"source": source})
    item_payload = canonical_json_bytes(item.model_dump(mode="json"))
    interrupted_row = InventoriedWorkItem(
        "fixture",
        Path("/fixture-interrupted.json"),
        sha256_bytes(item_payload),
        item,
    )
    receipt = _receipt(
        "extract", 1, work_item_sha256=interrupted_row.sha256
    ).model_copy(
        update={
            "callId": "call-interrupted-extract-1",
            "documentId": document_id,
            "workerId": "worker-interrupted-extract-1",
        }
    )
    _publish_receipt(paths, receipt)

    manifest = json.loads(
        publish_agent_run(
            config,
            (terminal_row,),
            frozen_selection=(terminal_row, interrupted_row),
        ).read_text()
    )
    usage = [
        json.loads(line)
        for line in (paths.run_root / "usage-by-document.jsonl").read_text().splitlines()
    ]

    assert manifest["publication_scope"] == "interrupted_snapshot"
    assert manifest["selected_documents"] == 1
    assert manifest["frozen_selection_documents"] == 2
    assert manifest["interrupted_documents_with_receipts"] == 1
    assert manifest["unprocessed_documents"] == 0
    assert manifest["training_ready"] is False
    assert manifest["usage"]["call_receipts"] == 5
    assert manifest["usage"]["outcome_referenced_call_receipts"] == 4
    assert manifest["usage"]["orphaned_call_receipts"] == 1
    assert usage[1]["documentId"] == document_id
    assert usage[1]["status"] == "interrupted_without_outcome"
    assert usage[1]["outcomeReferencedCalls"] == 0
    assert usage[1]["orphanedCalls"] == 1


@pytest.mark.asyncio
async def test_unlabeled_source_publishes_training_records_without_fake_reference_metrics(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, reference_target_mode="absent")
    paths = agent_run_paths(config)
    row = _work_item()
    await process_document(config, paths, _RetryGateway(), row)

    manifest_path = publish_agent_run(config, (row,))
    manifest = json.loads(manifest_path.read_text())

    assert manifest["training_records"] == 1
    assert manifest["reference_quality"]["status"] == "not_configured"
    assert not (paths.run_root / "reference-quality-by-document.jsonl").exists()
    assert {entry["kind"] for entry in manifest["files"]} == {
        "training_records",
        "usage_by_document",
    }


@pytest.mark.parametrize(
    "heading",
    [
        (
            'Consignee (Negotiable only if consigned "to order" or "to order of")\n'
            'As principal, where "care of" variants are used.\nACME IMPORTS LIMITED'
        ),
        (
            'CONSIGNEE: This B/L is not negotiable unless marked "To Order" here.\n'
            "ACME IMPORTS LIMITED"
        ),
        "Consignee (if 'To Order' so indicate)\nACME IMPORTS LIMITED",
        "Consignee (if To Order, so indicate)\nACME IMPORTS LIMITED",
        "Consignee (if 'to order' is indicated)\nACME IMPORTS LIMITED",
        "CONSIGNEE (If “to order” so indicate)\nACME IMPORTS LIMITED",
        'Consignee( if"To order" so indicate ) / Alci\nACME IMPORTS LIMITED',
        (
            'Consignee\u2019s Name and Address (unless provided otherwise, a consignment '
            '"To Order" shall mean "To Order of Shipper")\nACME IMPORTS LIMITED'
        ),
        "CONSIGNED TO\nACME IMPORTS LIMITED",
        "CONSIGNEE: ACME IMPORTS LIMITED",
    ],
)
def test_observed_straight_consignee_layouts_ground_non_negotiable(
    heading: str,
) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(heading),
        _compact_annotation(
            {
                "parties": {"consignee": {"name": "ACME IMPORTS LIMITED"}},
                "negotiability": "non_negotiable",
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


def test_pdf_grouping_can_assign_an_exact_ocr_consignee_without_supplying_its_value() -> None:
    item = _work_item_with_text(
        "NON-NEGOTIABLE COPY\nBILL OF LADING\nACME IMPORTS LIMITED\n"
        "Notify Party\nSame as Consignee\nNumber of Original B(s)/L THREE(3)\n"
        "ONE of which being accomplished, the others to stand void."
    )
    draft = _compact_annotation(
        {
            "parties": {"consignee": {"name": "ACME IMPORTS LIMITED"}},
            "negotiability": "non_negotiable",
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="copy-status stamp"):
        build_compact_annotation(item, draft, pdf_grouping_used=False)

    annotation = build_compact_annotation(item, draft, pdf_grouping_used=True)
    evidence = next(
        row for row in annotation.evidence if row.targetPath == "documentPatch.negotiability"
    )
    assert evidence.imageUse == "grouping_only"
    assert evidence.rawOcrEvidence[0].rawValue == "ACME IMPORTS LIMITED"


@pytest.mark.parametrize(
    "raw",
    [
        "TELEX RELEASE",
        "No. of original 0/ORIGINAL",
        "No. of original B(s)/L / 0 / N O N E",
        "Number of Original BL's\nB/L No.\n00/zeroes\n3909024A",
    ],
)
def test_release_or_zero_original_field_grounds_non_negotiable(raw: str) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"negotiability": "non_negotiable"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


def test_spaced_zero_original_field_grounds_non_negotiable() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("No. of Original B(s) /L\nZERO(0)"),
        _compact_annotation({"negotiability": "non_negotiable"}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


def test_importer_name_field_grounds_straight_non_negotiable_bill() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "Exporter Name\nACME EXPORTS\n\nImporter Name\nACME IMPORTS LIMITED"
        ),
        _compact_annotation(
            {
                "parties": {"consignee": {"name": "ACME IMPORTS LIMITED"}},
                "negotiability": "non_negotiable",
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.negotiability == "non_negotiable"


@pytest.mark.parametrize(
    ("raw", "references"),
    [
        ("PFI MS20240523", ("MS20240523",)),
        ("Export Contract 298763641", ("298763641",)),
        (
            "ED NO.201-07444443-24, 201-07447683-24",
            ("201-07444443-24", "201-07447683-24"),
        ),
    ],
)
def test_observed_export_reference_headings_are_supported(
    raw: str, references: tuple[str, ...]
) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(raw),
        _compact_annotation({"forwardingAndExportReferences": references}),
        pdf_grouping_used=False,
    )

    assert (
        annotation.normalLabel.documentPatch.forwardingAndExportReferences
        == references
    )


def test_comma_separated_day_month_year_date_is_grounded() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("Shipped on Board Date:\n18,APR,2024"),
        _compact_annotation({"shippedOnBoardDate": date(2024, 4, 18)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.shippedOnBoardDate == date(2024, 4, 18)


def test_un_table_heading_governs_following_table_row() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "CONTAINER NO / UN NO / DG CLASS / FLASH POINT /\n"
            "PACKAGING GROUP\nTCNU5538437 / 1993 / 3 / 15.0 C / II"
        ),
        _compact_annotation(
            {
                "containers": ({"containerNumber": "TCNU5538437"},),
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "dangerousGoods": ({"unNumber": "1993"},),
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    dangerous = annotation.normalLabel.documentPatch.goodsItems[0].dangerousGoods
    assert dangerous is not None and dangerous[0].unNumber == "1993"


def test_customs_tarif_ocr_heading_is_valid_hs_context() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("CUSTOMS TARIF NO.:84385000"),
        _compact_annotation(
            {"cargoGroups": ({"groupId": "g1", "hsCodes": ("84385000",)},)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == ("84385000",)


def test_arithmetic_gross_tare_total_block_grounds_grouped_gross_mass() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "Gross Weight\nKGS\n16,939\n\nTara: 3,900\nTotal Weight: 20,839"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "grossWeight": {"value": 16939.0, "unit": "kilogram"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].grossWeight.value == 16939.0


@pytest.mark.parametrize(
    "containment",
    ["PALLET STC: 1 PAIL", "442 CTN (STC : 670 PCS)"],
)
def test_explicit_package_containment_is_valid_additional_information(
    containment: str,
) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(containment),
        _compact_annotation(
            {
                "cargoGroups": (
                    {"groupId": "g1", "additionalInformation": (containment,)},
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert (
        annotation.normalLabel.documentPatch.goodsItems[0].additionalInformation
        == (containment,)
    )


def test_allocation_quantity_does_not_bind_to_container_check_digit() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "Marks & Numbers\nMCLU 509007/7\nSEAL: 6265313\n\n"
            "7 PALLETS\nPUMPS SPARE PARTS"
        ),
        _compact_annotation(
            {
                "containers": ({"containerNumber": "MCLU5090077"},),
                "cargoGroups": ({"groupId": "g1", "description": "PUMPS SPARE PARTS"},),
                "cargoPackages": (
                    {
                        "packageId": "p1",
                        "groupId": "g1",
                        "quantity": 7,
                        "typeDescription": "PALLETS",
                    },
                ),
                "cargoAllocationGroups": (
                    {
                        "groupId": "g1",
                        "coverage": "single_package_level",
                        "packageId": "p1",
                        "allocations": (
                            {"containerNumber": "MCLU5090077", "packageQuantity": 7},
                        ),
                    },
                ),
            }
        ),
        pdf_grouping_used=False,
    )

    relation_evidence = annotation.relationEvidence[0].rawOcrEvidence
    quantity_rows = tuple(row for row in relation_evidence if row.rawValue == "7")
    assert len(quantity_rows) == 1
    assert "7 PALLETS" in quantity_rows[0].ocrExcerpt


def test_hyphenated_ordinal_named_month_issue_date_is_normalized() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("Place and date of issue\nNOVOROSSIYSK\n18-th June 2023"),
        _compact_annotation({"issueDate": date(2023, 6, 18)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.issueDate.isoformat() == "2023-06-18"


def test_comma_separated_named_month_date_is_grounded() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("PLACE AND DATE OF ISSUE\nISTANBUL, MAY,30TH 2025"),
        _compact_annotation({"issueDate": date(2025, 5, 30)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.issueDate == date(2025, 5, 30)


def test_glued_freight_field_is_not_part_of_invoice_reference() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "INVOICE NO:LWIX805501-25FREIGHT COLLECT (FOB SHENZHEN, CHINA)"
        ),
        _compact_annotation(
            {"forwardingAndExportReferences": ("LWIX805501-25",)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "LWIX805501-25",
    )


def test_explicit_marks_may_contain_customs_clearance_company_wording() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "Marks and numbers\nEL-BASHA FOR IMPORT, EXPORT AND CUSTOMS CLEARANCE."
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "marksAndNumbers": (
                            "EL-BASHA FOR IMPORT, EXPORT AND CUSTOMS CLEARANCE.",
                        ),
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].marksAndNumbers == (
        "EL-BASHA FOR IMPORT, EXPORT AND CUSTOMS CLEARANCE.",
    )


@pytest.mark.parametrize(
    ("raw", "reference"),
    [
        (
            "PRN (Proof of Report Number): DB9257202403013308159",
            "DB9257202403013308159",
        ),
        ("P.E. 25001EC01082946Z", "25001EC01082946Z"),
        ("DUS 11475832-9", "11475832-9"),
        (
            "S/B NO. 6600762,7634148,6854337 DT. 08-01-2024",
            "6600762",
        ),
    ],
)
def test_explicit_report_and_export_permit_references_are_grounded(
    raw: str, reference: str
) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(raw),
            _compact_annotation({"forwardingAndExportReferences": (reference,)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        reference,
    )


def test_hyphen_before_container_check_digit_is_normalized() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("CONTAINER\nTGBU224635-0\n20 DC"),
        _compact_annotation(
            {
                "containers": (
                    {
                        "containerNumber": "TGBU2246350",
                        "typeDescription": "20 DC",
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.containers[0].containerNumber == (
        "TGBU2246350"
    )


def test_multiline_gross_weight_column_grounds_its_following_kgs_scalar() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "GROSS WEIGHT\nCargo\n\nKGS\n500.000\n\nKGS\nCBM\n\nTARE\n120.000"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "grossWeight": {"unit": "kilogram", "value": 500.0},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].grossWeight.value == 500.0


def test_multiline_gross_weight_heading_wins_over_equal_tare_scalar() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "TARE\nKGS\n500.000\n\nGROSS WEIGHT\nCargo\nKGS\n500.000"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "grossWeight": {"unit": "kilogram", "value": 500.0},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    evidence = {row.targetPath: row for row in annotation.evidence}
    gross_evidence = evidence[
        "documentPatch.goodsItems[0].grossWeight.value"
    ].rawOcrEvidence[0]
    assert gross_evidence.ocrExcerpt == "Cargo\nKGS\n500.000"


def test_glued_package_abbreviations_and_overlapping_table_columns_ground_counts() -> None:
    item = _work_item_with_text(
        "2 x 40HC CONTAINER\n1960 Bag(s)\n"
        "CAAU5799774 UL-5709306 40HC 22448.5 30 980 BAG CY/CY\n"
        "MSDU8660968 UL-5709305 40HC 22448.5 30 980 BAG CY/CY\n"
        "16PLTS 130PIECES"
    )
    annotation = build_compact_annotation(
        item,
        _compact_annotation(
            {
                "containers": (
                    {"containerNumber": "CAAU5799774"},
                    {"containerNumber": "MSDU8660968"},
                ),
                "cargoGroups": ({"groupId": "g1"},),
                "cargoPackages": (
                    {
                        "packageId": "p1",
                        "groupId": "g1",
                        "quantity": 1960,
                        "typeDescription": "Bag(s)",
                    },
                    {
                        "packageId": "p2",
                        "groupId": "g1",
                        "quantity": 16,
                        "typeDescription": "PLTS",
                    },
                    {
                        "packageId": "p3",
                        "groupId": "g1",
                        "quantity": 130,
                        "typeDescription": "PIECES",
                    },
                ),
                "cargoAllocationGroups": (
                    {
                        "groupId": "g1",
                        "coverage": "single_package_level",
                        "packageId": "p1",
                        "allocations": (
                            {"containerNumber": "CAAU5799774", "packageQuantity": 980},
                            {"containerNumber": "MSDU8660968", "packageQuantity": 980},
                        ),
                    },
                ),
            }
        ),
        pdf_grouping_used=False,
    )

    allocation_evidence = tuple(
        row for row in annotation.evidence if row.targetPath.endswith(".packageQuantity")
    )
    assert tuple(row.rawOcrEvidence[0].rawValue for row in allocation_evidence) == (
        "980",
        "980",
    )


def test_parenthesized_units_disambiguate_comma_grouped_mass_and_decimal_volume() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "Gross weight (Kgs) Measurement (M3)\n"
            "TOTALS 1 PALLET FREIGHT COLLECT 58,000 (Kgs) 0,100 (M3)"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "grossWeight": {"value": 58000.0, "unit": "kilogram"},
                        "volume": {"value": 0.1, "unit": "cubic_metre"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    cargo = annotation.normalLabel.documentPatch.goodsItems[0]
    assert cargo.grossWeight is not None and cargo.grossWeight.value == 58000.0
    assert cargo.volume is not None and cargo.volume.value == 0.1


def test_pdf_grouping_can_disambiguate_three_decimal_mass_separator() -> None:
    work_item = _work_item_with_text("Gross Weight\n703.075 KG")
    draft = _compact_annotation(
        {
            "cargoGroups": (
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 703.075, "unit": "kilogram"},
                },
            )
        }
    )

    with pytest.raises(DeterministicAnnotationError, match="not exactly groundable"):
        build_compact_annotation(work_item, draft, pdf_grouping_used=False)

    annotation = build_compact_annotation(work_item, draft, pdf_grouping_used=True)

    evidence = next(
        row
        for row in annotation.evidence
        if row.targetPath == "documentPatch.goodsItems[0].grossWeight.value"
    )
    assert evidence.rawOcrEvidence[0].rawValue == "703.075"
    assert evidence.imageUse == "grouping_only"


def test_pdf_grouping_does_not_ground_unqualified_ambiguous_number_as_mass() -> None:
    with pytest.raises(DeterministicAnnotationError, match="not exactly groundable"):
        build_compact_annotation(
            _work_item_with_text("Reference 703.075"),
            _compact_annotation(
                {
                    "cargoGroups": (
                        {
                            "groupId": "g1",
                            "grossWeight": {"value": 703.075, "unit": "kilogram"},
                        },
                    )
                }
            ),
            pdf_grouping_used=True,
        )


def test_temperature_setpoint_accepts_glued_celsius_value() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "CAAU5799774 REEFER\nSTOWED AT TEMPERATURE OF PLUS 2C TILL PLUS 2C"
        ),
        _compact_annotation(
            {
                "containers": (
                    {
                        "containerNumber": "CAAU5799774",
                        "temperatureSetpoint": {"value": 2.0, "unit": "celsius"},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.containers[0].temperatureSetpoint is not None


def test_clean_duplicate_mark_occurrence_wins_over_metadata_governed_occurrence() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "EL-NAHDA\nMARKS AND NUMBERS ACID: 123456789 EL-NAHDA"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {"groupId": "g1", "marksAndNumbers": ("EL-NAHDA",)},
                )
            }
        ),
        pdf_grouping_used=False,
    )

    evidence = next(
        row for row in annotation.evidence if row.targetPath.endswith("marksAndNumbers[0]")
    )
    assert evidence.rawOcrEvidence[0].rawValue == "EL-NAHDA"
    assert evidence.rawOcrEvidence[0].ocrExcerpt.count("EL-NAHDA") == 1


def test_gtip_heading_is_explicit_hs_context() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("TEXTILE GARMENTS\nGTIP: 620140 610343 611020"),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "description": "TEXTILE GARMENTS",
                        "hsCodes": ("620140", "610343", "611020"),
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].hsCodes == (
        "620140",
        "610343",
        "611020",
    )


def test_post_terminal_correspondence_does_not_expand_bill_container_completeness() -> None:
    annotation = build_compact_annotation(
        _work_item_with_pages(
            "ESDU4122375 1000 BAGS\nEND OF BILL OF LADING HBL001",
            "TO WHOM IT MAY CONCERN\nBOOKING ESDU4342282 & ESDU4122375",
        ),
        _compact_annotation(
            {"containers": ({"containerNumber": "ESDU4122375"},)}
        ),
        pdf_grouping_used=False,
    )

    assert tuple(
        row.containerNumber for row in annotation.normalLabel.documentPatch.containers
    ) == ("ESDU4122375",)


def test_unit_qualified_row_volume_controls_style_over_following_row_mass() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "TRAILER ONE\nWeight kg. 10,000.000 KGS 142.506 CBM\n"
            "TRAILER TWO\n8,864.000 KGS 139.400 CBM"
        ),
        _compact_annotation(
            {
                "cargoGroups": (
                    {
                        "groupId": "g1",
                        "description": "TRAILER ONE",
                        "volume": {"unit": "cubic_metre", "value": 142.506},
                    },
                    {
                        "groupId": "g2",
                        "description": "TRAILER TWO",
                        "volume": {"unit": "cubic_metre", "value": 139.4},
                    },
                )
            }
        ),
        pdf_grouping_used=False,
    )

    assert tuple(
        row.volume.value for row in annotation.normalLabel.documentPatch.goodsItems
    ) == (142.506, 139.4)


def test_glued_pces_quantity_is_grounded_as_a_package_count() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("56PLTS=1000CTNS=20000PCES"),
        _compact_annotation(
            {
                "cargoGroups": ({"groupId": "g1"},),
                "cargoPackages": (
                    {
                        "packageId": "p1",
                        "groupId": "g1",
                        "quantity": 20000,
                        "typeDescription": "PCES",
                    },
                ),
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.goodsItems[0].packages[0].quantity == 20000


def test_single_package_total_reconciles_explicit_container_row_quantities() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("FYCU1086910 20 BAG\nONLU2207538 20 BAG"),
        _compact_annotation(
            {
                "containers": (
                    {"containerNumber": "FYCU1086910"},
                    {"containerNumber": "ONLU2207538"},
                ),
                "cargoGroups": ({"groupId": "g1"},),
                "cargoPackages": (
                    {
                        "packageId": "p1",
                        "groupId": "g1",
                        "quantity": 40,
                        "typeDescription": "BAG",
                    },
                ),
                "cargoAllocationGroups": (
                    {
                        "groupId": "g1",
                        "coverage": "single_package_level",
                        "packageId": "p1",
                        "allocations": (
                            {
                                "containerNumber": "FYCU1086910",
                                "packageQuantity": 20,
                            },
                            {
                                "containerNumber": "ONLU2207538",
                                "packageQuantity": 20,
                            },
                        ),
                    },
                ),
            }
        ),
        pdf_grouping_used=False,
    )

    quantity_evidence = next(
        row
        for row in annotation.evidence
        if row.targetPath == "documentPatch.goodsItems[0].packages[0].quantity"
    )
    assert len(quantity_evidence.rawOcrEvidence) == 2
    assert "summed the exact OCR-grounded" in (
        quantity_evidence.normalizationRule or ""
    )


def test_spaced_proforma_reference_is_governed_by_previous_nonblank_heading() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "AS PER PROFORMA INVOICE NO.:\n\nP - 480 / 2025 DATED: 03.03.2025"
        ),
        _compact_annotation(
            {"forwardingAndExportReferences": ("P - 480 / 2025",)}
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences == (
        "P - 480 / 2025",
    )


@pytest.mark.parametrize("raw", ["PI22-05-076R1", "SHIPMENT # PI0224"])
def test_unqualified_pi_text_is_not_an_explicit_invoice_reference(raw: str) -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(f"CARGO DESCRIPTION\nPRODUCT\n{raw}"),
        _compact_annotation({"cargoGroups": ({"groupId": "g1", "description": "PRODUCT"},)}),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences is None


def test_inv_prefix_inside_investment_address_is_not_an_invoice_reference() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text("CONSIGNEE\nACME\nINVESTMENT-10TH OF RAMADAN"),
        _compact_annotation(
            {
                "parties": {
                    "consignee": {
                        "name": "ACME",
                        "address": "INVESTMENT-10TH OF RAMADAN",
                    }
                }
            }
        ),
        pdf_grouping_used=False,
    )

    assert annotation.normalLabel.documentPatch.forwardingAndExportReferences is None


def test_footnote_linked_address_continuation_is_grounded_without_image_text() -> None:
    annotation = build_compact_annotation(
        _work_item_with_text(
            "SHIPPER\nACME EXPORTS\nZIYANG AIRPORT*\n"
            + ("UNRELATED CARGO TEXT " * 80)
            + "\n*ECONOMIC ZONE, SICHUAN PROVINCE"
        ),
        _compact_annotation(
            {
                "parties": {
                    "shipper": {
                        "name": "ACME EXPORTS",
                        "address": "ZIYANG AIRPORT, ECONOMIC ZONE, SICHUAN PROVINCE",
                    }
                }
            }
        ),
        pdf_grouping_used=False,
    )

    address = annotation.normalLabel.documentPatch.parties.shipper.address
    evidence = next(row for row in annotation.evidence if row.targetPath.endswith(".address"))
    assert address == "ZIYANG AIRPORT, ECONOMIC ZONE, SICHUAN PROVINCE"
    assert evidence.imageUse == "not_used"
    assert len(evidence.rawOcrEvidence) == 2
