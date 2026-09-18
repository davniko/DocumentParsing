"""Pinned manual resolution and EDA for compiled-template review queues.

The paid compiler run remains immutable.  This module publishes a small adjudication overlay that
pins the source commit, proves that every review-required outcome received exactly one decision,
and revalidates every retained template.  A manual exclusion is not a compiler rejection: both are
preserved separately so that yield and failure-mode analyses remain honest.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.template_integrity import (
    explicit_carrier_receipt_container_counts,
    source_template_integrity_issues,
)

from .descendant import _validate_committed_run
from .descendant_models import PinnedCommittedRun
from .host import source_carrier, template_summary
from .models import CertifiedSemanticTemplate, ExtractionCaseResult, NonEmptyText

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_CONTAINER_ID = re.compile(r"(?<![A-Z0-9])[A-Z]{4}\s?[0-9]{7}(?![A-Z0-9])")
_COLORS = ("#2563EB", "#DC2626", "#D97706", "#059669", "#7C3AED", "#0891B2")


class ManualReviewDecision(BaseModel):
    model_config = _STRICT

    document_id: NonEmptyText
    resolution: Literal["exclude"]
    issue_class: Literal[
        "incomplete_container_topology",
        "compound_package_dependency",
    ]
    expected_review_reasons: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    source_line_numbers: Annotated[tuple[int, ...], Field(min_length=1)]
    rationale: NonEmptyText
    expected_printed_container_counts: tuple[int, ...] = ()
    expected_labeled_container_count: int | None = None
    expected_observed_container_identity_count: int | None = None
    expected_coherence_constraint_id: NonEmptyText | None = None

    @model_validator(mode="after")
    def evidence_matches_issue_class(self) -> ManualReviewDecision:
        if tuple(sorted(set(self.source_line_numbers))) != self.source_line_numbers:
            raise ValueError("source evidence line numbers must be unique and sorted")
        if any(value <= 0 for value in self.source_line_numbers):
            raise ValueError("source evidence line numbers must be positive")
        is_container = self.issue_class == "incomplete_container_topology"
        container_fields_present = (
            bool(self.expected_printed_container_counts)
            and self.expected_labeled_container_count is not None
            and self.expected_observed_container_identity_count is not None
        )
        if is_container != container_fields_present:
            raise ValueError("container evidence fields must be present only for container issues")
        if is_container and self.expected_coherence_constraint_id is not None:
            raise ValueError("container issues cannot name a coherence constraint")
        if not is_container and self.expected_coherence_constraint_id is None:
            raise ValueError("compound dependency issues must name their coherence constraint")
        return self


class TemplateReviewResolutionConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["bill_of_lading_template_manual_review_resolution_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    source_run: PinnedCommittedRun
    certified_catalog_run: PinnedCommittedRun | None = None
    review_date: NonEmptyText
    review_method: Literal["full_queue_source_target_and_checkpoint_review"]
    expected_source_documents: Annotated[int, Field(gt=0)]
    expected_source_certified: Annotated[int, Field(ge=0)]
    expected_source_rejected: Annotated[int, Field(ge=0)]
    expected_source_review_required: Annotated[int, Field(gt=0)]
    expected_usable_templates: Annotated[int, Field(gt=0)]
    expected_excluded_documents: Annotated[int, Field(ge=0)]
    require_zero_unresolved_reviews: Literal[True]
    decisions: Annotated[tuple[ManualReviewDecision, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def counts_and_decisions_are_closed(self) -> TemplateReviewResolutionConfig:
        if (
            self.expected_source_certified
            + self.expected_source_rejected
            + self.expected_source_review_required
            != self.expected_source_documents
        ):
            raise ValueError("source status counts must cover every source document")
        if self.expected_usable_templates != self.expected_source_certified:
            raise ValueError("manual exclusion cannot change the certified template count")
        if self.expected_excluded_documents != (
            self.expected_source_rejected + self.expected_source_review_required
        ):
            raise ValueError("excluded count must cover rejected and manually excluded documents")
        ids = tuple(row.document_id for row in self.decisions)
        if len(ids) != self.expected_source_review_required or len(set(ids)) != len(ids):
            raise ValueError("manual decisions must uniquely cover the expected review queue")
        return self


def load_review_resolution_config(path: Path) -> TemplateReviewResolutionConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return TemplateReviewResolutionConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            yield row


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _csv_bytes(*, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _stage_cost(result: ExtractionCaseResult) -> Decimal:
    return sum(
        (row.usage.estimatedCostUsd for row in (*result.compiler_stages, *result.critic_stages)),
        Decimal(0),
    )


def _stage_requests(result: ExtractionCaseResult) -> int:
    return sum(row.usage.requests for row in (*result.compiler_stages, *result.critic_stages))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summary_statistics(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _result_category(result: ExtractionCaseResult) -> str:
    joined = "\n".join(result.rejection_reasons).lower()
    if "exceeded maximum output retries" in joined:
        return "structured output retries exhausted"
    if "local_compiler_repair_projection" in joined or "compiler made no progress" in joined:
        return "unsafe or repeated host-rejected repair"
    if "terminal critic confirmation revised the state" in joined:
        return "terminal critic revision"
    if "additional-call launch threshold reached" in joined:
        return "cost guard"
    return "other compiler rejection"


def _evidence_lines(raw: str, numbers: Sequence[int]) -> tuple[dict[str, Any], ...]:
    lines = raw.splitlines()
    if any(number > len(lines) for number in numbers):
        raise ValueError("manual evidence line exceeds the source document")
    return tuple(
        {
            "lineNumber": number,
            "text": lines[number - 1],
            "sha256": sha256_bytes(lines[number - 1].encode("utf-8")),
        }
        for number in numbers
    )


def _audit_manual_decision(
    *,
    decision: ManualReviewDecision,
    result: ExtractionCaseResult,
    raw: str,
    source_target: Mapping[str, Any],
    case_root: Path,
) -> dict[str, Any]:
    if result.status != "review_required":
        raise ValueError(f"manual decision targets a non-review outcome: {decision.document_id}")
    if result.rejection_reasons != decision.expected_review_reasons:
        raise ValueError(f"review reason differs from the pinned decision: {decision.document_id}")
    computed: dict[str, Any]
    if decision.issue_class == "incomplete_container_topology":
        issues = source_template_integrity_issues(raw, source_target)
        expected_issues = tuple(
            reason.removeprefix("source template integrity requires review: ")
            for reason in decision.expected_review_reasons
        )
        if issues != expected_issues:
            raise ValueError(f"source-integrity issue no longer reproduces: {decision.document_id}")
        printed = explicit_carrier_receipt_container_counts(raw)
        patch = source_target.get("documentPatch")
        containers = patch.get("containers") if isinstance(patch, Mapping) else None
        labeled_count = len(containers) if isinstance(containers, list) else 0
        identities = tuple(dict.fromkeys(_CONTAINER_ID.findall(raw.upper())))
        if printed != decision.expected_printed_container_counts:
            raise ValueError(f"printed container count differs: {decision.document_id}")
        if labeled_count != decision.expected_labeled_container_count:
            raise ValueError(f"labeled container count differs: {decision.document_id}")
        if len(identities) != decision.expected_observed_container_identity_count:
            raise ValueError(f"observed container identity count differs: {decision.document_id}")
        if not any(count != labeled_count for count in printed):
            raise ValueError(
                f"manual container exclusion has no count contradiction: {decision.document_id}"
            )
        computed = {
            "printedContainerCounts": printed,
            "labeledContainerCount": labeled_count,
            "observedDistinctContainerIdentities": identities,
            "observedDistinctContainerIdentityCount": len(identities),
        }
    else:
        checkpoint_path = case_root / "state-checkpoint.json"
        checkpoint = json.loads(read_regular_file_bytes(checkpoint_path))
        if not isinstance(checkpoint, dict):
            raise ValueError(f"review checkpoint is not an object: {decision.document_id}")
        constraints = checkpoint.get("coherence_constraints")
        if not isinstance(constraints, list):
            raise ValueError(f"review checkpoint has no constraints: {decision.document_id}")
        matching = tuple(
            row
            for row in constraints
            if isinstance(row, Mapping)
            and row.get("constraint_id") == decision.expected_coherence_constraint_id
            and row.get("kind") == "review_required"
        )
        if len(matching) != 1:
            raise ValueError(f"coherence constraint differs: {decision.document_id}")
        computed = {
            "coherenceConstraintId": decision.expected_coherence_constraint_id,
            "constraint": matching[0],
        }
    return {
        "schemaVersion": 1,
        "documentId": decision.document_id,
        "resolution": "excluded",
        "issueClass": decision.issue_class,
        "originalReviewReasons": decision.expected_review_reasons,
        "manualRationale": decision.rationale,
        "sourceEvidence": _evidence_lines(raw, decision.source_line_numbers),
        "computedEvidence": computed,
    }


def _bar_chart(
    *, title: str, labels: Sequence[str], values: Sequence[float], value_format: str = ".0f"
) -> bytes:
    if len(labels) != len(values) or not labels:
        raise ValueError("bar chart labels and values must be non-empty and aligned")
    width = 1400
    row_height = 58
    height = 130 + row_height * len(labels)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((36, 28), title, fill="#111827", font=font)
    maximum = max(values) or 1.0
    chart_start = 400
    chart_width = width - chart_start - 150
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 82 + index * row_height
        short = label if len(label) <= 52 else label[:49] + "..."
        draw.text((36, y + 14), short, fill="#374151", font=font)
        bar_width = int(chart_width * value / maximum)
        draw.rounded_rectangle(
            (chart_start, y + 7, chart_start + max(bar_width, 2), y + 36),
            radius=5,
            fill=_COLORS[index % len(_COLORS)],
        )
        draw.text(
            (chart_start + max(bar_width, 2) + 12, y + 14),
            format(value, value_format),
            fill="#111827",
            font=font,
        )
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _artifact_inputs(
    *, project_root: Path, config: TemplateReviewResolutionConfig
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, Any]]:
    source_root = _validate_committed_run(project_root, config.source_run)
    certified_root = (
        _validate_committed_run(project_root, config.certified_catalog_run)
        if config.certified_catalog_run is not None
        else source_root
    )
    catalog_rows = list(_iter_jsonl(certified_root / "catalog.jsonl"))
    summary = json.loads(read_regular_file_bytes(source_root / "summary.json"))
    if not isinstance(summary, dict):
        raise ValueError("source summary is not an object")
    return source_root, certified_root, catalog_rows, summary


def _analyze(*, project_root: Path, config: TemplateReviewResolutionConfig) -> dict[str, Any]:
    source_root, certified_root, catalog_rows, source_summary = _artifact_inputs(
        project_root=project_root, config=config
    )
    catalog_by_id = {cast(str, row.get("documentId")): row for row in catalog_rows}
    if len(catalog_by_id) != len(catalog_rows):
        raise ValueError("source catalog contains duplicate or invalid document IDs")
    decisions = {row.document_id: row for row in config.decisions}

    status_counts: Counter[str] = Counter()
    resolved_counts: Counter[str] = Counter()
    exclusion_categories: Counter[str] = Counter()
    carrier_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    cost_by_origin: dict[str, Decimal] = defaultdict(Decimal)
    requests_by_origin: Counter[str] = Counter()
    certified_order: list[str] = []
    reviewed_ids: set[str] = set()
    resolved_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    template_metrics: list[dict[str, Any]] = []

    for raw_result in _iter_jsonl(source_root / "results.jsonl"):
        result = ExtractionCaseResult.model_validate_json(
            canonical_json_bytes(raw_result), strict=True
        )
        document_id = result.document_id
        case_root = source_root / "cases" / document_id
        source_path = case_root / "source.txt"
        label_path = case_root / "source-label.json"
        result_path = case_root / "result.json"
        raw_bytes = read_regular_file_bytes(source_path)
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"source is not UTF-8: {document_id}") from error
        source_target = json.loads(read_regular_file_bytes(label_path))
        if not isinstance(source_target, dict):
            raise ValueError(f"source label is not an object: {document_id}")
        case_result = ExtractionCaseResult.model_validate_json(
            read_regular_file_bytes(result_path), strict=True
        )
        if case_result != result:
            raise ValueError(f"case result differs from results.jsonl: {document_id}")
        carrier = source_carrier(source_target)
        carrier_name = carrier or "UNRESOLVED"
        status_counts[result.status] += 1
        stage_cost = _stage_cost(result)
        stage_requests = _stage_requests(result)
        base = {
            "documentId": document_id,
            "sourceStatus": result.status,
            "sourceSha256": sha256_bytes(raw_bytes),
            "sourceLabelSha256": sha256_bytes(canonical_json_bytes(source_target)),
            "sourceResultSha256": sha256_file(result_path),
            "templateSha256": result.template_sha256,
            "carrier": carrier,
            "sourceRejectionReasons": result.rejection_reasons,
            "sourceEstimatedCostUsd": str(stage_cost),
            "sourceRequests": stage_requests,
        }

        if result.status == "certified":
            certified_order.append(document_id)
            resolved_counts["usable"] += 1
            origin = "compiled_certification"
            template_case_root = certified_root / "cases" / document_id
            template_path = template_case_root / "template.json"
            template = CertifiedSemanticTemplate.model_validate_json(
                read_regular_file_bytes(template_path), strict=True
            )
            if template.document_id != document_id:
                raise ValueError(f"template document identity differs: {document_id}")
            template_sha256 = sha256_file(template_path)
            if certified_root == source_root:
                if result.template_sha256 != template_sha256:
                    raise ValueError(f"template hash differs from source result: {document_id}")
            else:
                recertified_source = read_regular_file_bytes(template_case_root / "source.txt")
                recertified_label = json.loads(
                    read_regular_file_bytes(template_case_root / "source-label.json")
                )
                if recertified_source != raw_bytes:
                    raise ValueError(
                        f"recertified source differs from compiler source: {document_id}"
                    )
                if canonical_json_bytes(recertified_label) != canonical_json_bytes(source_target):
                    raise ValueError(
                        f"recertified label differs from compiler label: {document_id}"
                    )
                lineage = json.loads(read_regular_file_bytes(template_case_root / "lineage.json"))
                if not isinstance(lineage, Mapping):
                    raise ValueError(f"recertified lineage is not an object: {document_id}")
                expected_lineage = {
                    "documentId": document_id,
                    "status": "certified",
                    "sourceSha256": sha256_bytes(raw_bytes),
                    "sourceLabelSha256": sha256_bytes(canonical_json_bytes(source_target)),
                    "sourceResultSha256": sha256_file(result_path),
                    "sourceRunCommitSha256": config.source_run.commit_sha256,
                    "templateSha256": template_sha256,
                }
                mismatched = tuple(
                    key for key, value in expected_lineage.items() if lineage.get(key) != value
                )
                if mismatched:
                    raise ValueError(
                        f"recertified lineage differs for {document_id}: {', '.join(mismatched)}"
                    )
            base["templateSha256"] = template_sha256
            if template.source_sha256 != sha256_bytes(raw_bytes):
                raise ValueError(f"template source hash differs: {document_id}")
            if template.certification.carrier_matches_source_label is not True:
                raise ValueError(f"template carrier is not source-pinned: {document_id}")
            expected_catalog = catalog_by_id.get(document_id)
            if expected_catalog is None or canonical_json_bytes(template_summary(template)) != (
                canonical_json_bytes(expected_catalog)
            ):
                raise ValueError(f"catalog summary differs from template: {document_id}")
            binding_count = len(template.bindings)
            agent_count = sum(
                binding.realization.mode == "agent_required" for binding in template.bindings
            )
            deterministic_count = binding_count - agent_count
            metric = {
                **expected_catalog,
                "deterministicFraction": deterministic_count / binding_count,
                "sourceEstimatedCostUsd": str(stage_cost),
                "sourceRequests": stage_requests,
            }
            template_metrics.append(metric)
        elif result.status == "review_required":
            decision = decisions.get(document_id)
            if decision is None:
                raise ValueError(f"review-required outcome lacks a manual decision: {document_id}")
            reviewed_ids.add(document_id)
            resolved_counts["excluded"] += 1
            origin = "manual_review_exclusion"
            audited = _audit_manual_decision(
                decision=decision,
                result=result,
                raw=raw,
                source_target=source_target,
                case_root=case_root,
            )
            manual_rows.append(
                {
                    **audited,
                    "sourceSha256": sha256_bytes(raw_bytes),
                    "sourceLabelSha256": sha256_bytes(canonical_json_bytes(source_target)),
                    "sourceResultSha256": sha256_file(result_path),
                }
            )
            exclusion_categories[f"manual: {decision.issue_class}"] += 1
        else:
            resolved_counts["excluded"] += 1
            origin = "compiler_rejection"
            exclusion_categories[f"compiler: {_result_category(result)}"] += 1
        carrier_outcomes[carrier_name][origin] += 1
        cost_by_origin[origin] += stage_cost
        requests_by_origin[origin] += stage_requests
        resolved_rows.append(
            {
                **base,
                "resolvedStatus": "usable" if result.status == "certified" else "excluded",
                "resolutionOrigin": origin,
                "manualIssueClass": (
                    decisions[document_id].issue_class if document_id in decisions else None
                ),
            }
        )

    if len(resolved_rows) != config.expected_source_documents:
        raise ValueError("source result count differs from the configured contract")
    expected_statuses = Counter(
        {
            "certified": config.expected_source_certified,
            "rejected": config.expected_source_rejected,
            "review_required": config.expected_source_review_required,
        }
    )
    if status_counts != expected_statuses:
        raise ValueError(f"source status counts differ: {dict(status_counts)}")
    if reviewed_ids != set(decisions):
        raise ValueError("manual decisions do not exactly cover the source review queue")
    missing_catalog_ids = tuple(
        document_id for document_id in certified_order if document_id not in catalog_by_id
    )
    if missing_catalog_ids:
        raise ValueError(
            "certified catalog omits source-certified documents: " + ", ".join(missing_catalog_ids)
        )
    if certified_root == source_root and certified_order != list(catalog_by_id):
        raise ValueError("catalog order does not match certified result order")
    resolved_catalog_rows = [catalog_by_id[document_id] for document_id in certified_order]
    if resolved_counts != Counter(
        usable=config.expected_usable_templates,
        excluded=config.expected_excluded_documents,
    ):
        raise ValueError(f"resolved status counts differ: {dict(resolved_counts)}")
    if len(template_metrics) != config.expected_usable_templates:
        raise ValueError("usable template metrics do not cover the certified catalog")

    bindings = [float(row["bindings"]) for row in template_metrics]
    occurrences = [float(row["occurrences"]) for row in template_metrics]
    deterministic = [float(row["deterministicFraction"]) for row in template_metrics]
    residual = [float(row["agentResidualBindings"]) for row in template_metrics]
    pages = [float(row["pages"]) for row in template_metrics]
    top_carriers = Counter(cast(str, row["carrierFamily"]) for row in template_metrics)
    document_types = Counter(cast(str, row["documentType"]) for row in template_metrics)
    total_bindings = sum(int(row["bindings"]) for row in template_metrics)
    total_deterministic = sum(int(row["deterministicBindings"]) for row in template_metrics)
    total_agent = sum(int(row["agentAssistedBindings"]) for row in template_metrics)
    analysis = {
        "schemaVersion": 1,
        "sourceDocuments": len(resolved_rows),
        "usableTemplates": resolved_counts["usable"],
        "excludedDocuments": resolved_counts["excluded"],
        "unresolvedReviews": 0,
        "usableYield": resolved_counts["usable"] / len(resolved_rows),
        "sourceStatusCounts": dict(sorted(status_counts.items())),
        "resolutionOriginCounts": dict(
            sorted(Counter(row["resolutionOrigin"] for row in resolved_rows).items())
        ),
        "exclusionCategories": dict(sorted(exclusion_categories.items())),
        "templateIntegrity": {
            "strictSchemaValidated": len(template_metrics),
            "sourceHashesMatched": len(template_metrics),
            "templateHashesMatched": len(template_metrics),
            "catalogRowsMatched": len(template_metrics),
            "carrierPinsMatched": len(template_metrics),
            "failedChecks": 0,
        },
        "bindingTotals": {
            "bindings": total_bindings,
            "deterministic": total_deterministic,
            "agentAssisted": total_agent,
            "deterministicFraction": total_deterministic / total_bindings,
        },
        "templateDistributions": {
            "bindings": _summary_statistics(bindings),
            "occurrences": _summary_statistics(occurrences),
            "deterministicFraction": _summary_statistics(deterministic),
            "agentResidualBindings": _summary_statistics(residual),
            "pages": _summary_statistics(pages),
        },
        "documentTypes": dict(sorted(document_types.items())),
        "distinctCarrierFamilies": len(top_carriers),
        "topCarrierFamilies": dict(top_carriers.most_common(20)),
        "sourceLineageUsage": source_summary.get("usage"),
        "sourceMarginalUsage": source_summary.get("marginalUsage"),
        "costByResolutionOriginUsd": {
            key: str(value) for key, value in sorted(cost_by_origin.items())
        },
        "requestsByResolutionOrigin": dict(sorted(requests_by_origin.items())),
        "manualResolutionProviderRequests": 0,
        "manualResolutionIncrementalCostUsd": "0",
    }
    return {
        "source_root": source_root,
        "source_summary": source_summary,
        "catalog_rows": resolved_catalog_rows,
        "certified_root": certified_root,
        "resolved_rows": resolved_rows,
        "manual_rows": sorted(manual_rows, key=lambda row: cast(str, row["documentId"])),
        "template_metrics": template_metrics,
        "analysis": analysis,
        "exclusion_categories": exclusion_categories,
        "carrier_outcomes": carrier_outcomes,
    }


def preflight_review_resolution(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_review_resolution_config(config_path)
    analyzed = _analyze(project_root=project_root, config=config)
    analysis = cast(dict[str, Any], analyzed["analysis"])
    return {
        "schemaVersion": 1,
        "sourceDocuments": analysis["sourceDocuments"],
        "manualDecisions": len(config.decisions),
        "usableTemplates": analysis["usableTemplates"],
        "excludedDocuments": analysis["excludedDocuments"],
        "unresolvedReviews": analysis["unresolvedReviews"],
        "providerRequests": 0,
        "estimatedCostUsd": "0",
    }


def resolve_template_reviews(*, project_root: Path, config_path: Path) -> Path:
    started = time.perf_counter()
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("review-resolution configuration is outside the project root") from error
    config = load_review_resolution_config(config_path)
    output_parent = (project_root / config.output_dir).resolve()
    if output_parent != project_root and project_root not in output_parent.parents:
        raise ValueError("review-resolution output directory escapes the project root")
    analyzed = _analyze(project_root=project_root, config=config)
    source_root = cast(Path, analyzed["source_root"])
    certified_root = cast(Path, analyzed["certified_root"])
    catalog_rows = cast(list[dict[str, Any]], analyzed["catalog_rows"])
    resolved_rows = cast(list[dict[str, Any]], analyzed["resolved_rows"])
    manual_rows = cast(list[dict[str, Any]], analyzed["manual_rows"])
    template_metrics = cast(list[dict[str, Any]], analyzed["template_metrics"])
    analysis = cast(dict[str, Any], analyzed["analysis"])
    exclusion_categories = cast(Counter[str], analyzed["exclusion_categories"])

    implementation_path = Path(__file__).resolve(strict=True)
    transaction = {
        "config": config.model_dump(mode="json"),
        "configSha256": sha256_file(config_path),
        "implementationSha256": sha256_file(implementation_path),
        "sourceRun": {
            "path": config.source_run.path,
            "commitSha256": config.source_run.commit_sha256,
            "transactionSha256": config.source_run.transaction_sha256,
            "catalogSha256": sha256_file(source_root / "catalog.jsonl"),
            "resultsSha256": sha256_file(source_root / "results.jsonl"),
        },
        "certifiedCatalogRun": (
            {
                "path": config.certified_catalog_run.path,
                "commitSha256": config.certified_catalog_run.commit_sha256,
                "transactionSha256": config.certified_catalog_run.transaction_sha256,
                "catalogSha256": sha256_file(certified_root / "catalog.jsonl"),
            }
            if config.certified_catalog_run is not None
            else None
        ),
    }
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    expected: set[str] = set()

    def publish_json(relative: str, value: Any) -> None:
        staged.publish_json(relative, value)
        expected.add(relative)

    def publish_bytes(relative: str, value: bytes) -> None:
        staged.publish_bytes(relative, value)
        expected.add(relative)

    publish_json("config.json", config.model_dump(mode="json"))
    publish_json("lineage.json", transaction)
    publish_bytes("catalog.jsonl", _jsonl_bytes(catalog_rows))
    publish_bytes("resolved-outcomes.jsonl", _jsonl_bytes(resolved_rows))
    publish_bytes("manual-decisions.jsonl", _jsonl_bytes(manual_rows))
    publish_bytes(
        "usable-document-ids.txt",
        (
            "\n".join(
                row["documentId"] for row in resolved_rows if row["resolvedStatus"] == "usable"
            )
            + "\n"
        ).encode(),
    )
    publish_bytes(
        "excluded-document-ids.txt",
        (
            "\n".join(
                row["documentId"] for row in resolved_rows if row["resolvedStatus"] == "excluded"
            )
            + "\n"
        ).encode(),
    )
    publish_json("analysis/analysis.json", analysis)
    publish_bytes(
        "analysis/data/status-summary.csv",
        _csv_bytes(
            fieldnames=("resolutionOrigin", "documents", "costUsd", "requests"),
            rows=(
                {
                    "resolutionOrigin": origin,
                    "documents": count,
                    "costUsd": analysis["costByResolutionOriginUsd"].get(origin, "0"),
                    "requests": analysis["requestsByResolutionOrigin"].get(origin, 0),
                }
                for origin, count in analysis["resolutionOriginCounts"].items()
            ),
        ),
    )
    publish_bytes(
        "analysis/data/exclusion-summary.csv",
        _csv_bytes(
            fieldnames=("category", "documents"),
            rows=(
                {"category": category, "documents": count}
                for category, count in sorted(exclusion_categories.items())
            ),
        ),
    )
    metric_fields = (
        "documentId",
        "carrier",
        "carrierFamily",
        "documentType",
        "pages",
        "lines",
        "characters",
        "bindings",
        "occurrences",
        "deterministicBindings",
        "agentAssistedBindings",
        "agentResidualBindings",
        "deterministicFraction",
        "semanticOnlyTargetFacts",
        "coherenceBindings",
        "sourceEstimatedCostUsd",
        "sourceRequests",
    )
    publish_bytes(
        "analysis/data/template-metrics.csv",
        _csv_bytes(
            fieldnames=metric_fields,
            rows=({key: row[key] for key in metric_fields} for row in template_metrics),
        ),
    )
    publish_bytes(
        "analysis/data/manual-review.csv",
        _csv_bytes(
            fieldnames=(
                "documentId",
                "resolution",
                "issueClass",
                "manualRationale",
                "originalReviewReasons",
                "sourceEvidence",
                "computedEvidence",
                "sourceSha256",
                "sourceLabelSha256",
                "sourceResultSha256",
            ),
            rows=(
                {
                    "documentId": row["documentId"],
                    "resolution": row["resolution"],
                    "issueClass": row["issueClass"],
                    "manualRationale": row["manualRationale"],
                    "originalReviewReasons": json.dumps(row["originalReviewReasons"]),
                    "sourceEvidence": json.dumps(row["sourceEvidence"], ensure_ascii=False),
                    "computedEvidence": json.dumps(row["computedEvidence"], ensure_ascii=False),
                    "sourceSha256": row["sourceSha256"],
                    "sourceLabelSha256": row["sourceLabelSha256"],
                    "sourceResultSha256": row["sourceResultSha256"],
                }
                for row in manual_rows
            ),
        ),
    )

    origins = analysis["resolutionOriginCounts"]
    publish_bytes(
        "analysis/plots/outcome-resolution.png",
        _bar_chart(
            title=f"Outcome resolution ({analysis['sourceDocuments']} source documents)",
            labels=tuple(origins),
            values=tuple(float(value) for value in origins.values()),
        ),
    )
    publish_bytes(
        "analysis/plots/exclusion-causes.png",
        _bar_chart(
            title="Excluded document causes",
            labels=tuple(exclusion_categories),
            values=tuple(float(value) for value in exclusion_categories.values()),
        ),
    )
    binding_bands = (
        ("<40", 0, 40),
        ("40-59", 40, 60),
        ("60-79", 60, 80),
        ("80-99", 80, 100),
        ("100-149", 100, 150),
        ("150-249", 150, 250),
        ("250+", 250, math.inf),
    )
    binding_band_counts = tuple(
        sum(low <= int(row["bindings"]) < high for row in template_metrics)
        for _label, low, high in binding_bands
    )
    publish_bytes(
        "analysis/plots/template-binding-distribution.png",
        _bar_chart(
            title="Certified templates by binding count",
            labels=tuple(label for label, _low, _high in binding_bands),
            values=binding_band_counts,
        ),
    )
    coverage_bins: Counter[int] = Counter()
    for row in template_metrics:
        fraction = float(row["deterministicFraction"])
        coverage_bins[100 if math.isclose(fraction, 1.0) else int(fraction * 100) // 2 * 2] += 1
    coverage_keys = sorted(coverage_bins)
    publish_bytes(
        "analysis/plots/deterministic-coverage.png",
        _bar_chart(
            title="Certified templates by deterministic binding coverage",
            labels=tuple(
                "100%" if value == 100 else f"{value}-<{value + 2}%" for value in coverage_keys
            ),
            values=tuple(float(coverage_bins[value]) for value in coverage_keys),
        ),
    )
    carriers = list(Counter(row["carrierFamily"] for row in template_metrics).most_common(15))
    publish_bytes(
        "analysis/plots/top-carrier-families.png",
        _bar_chart(
            title="Top certified carrier families",
            labels=tuple(str(row[0]) for row in carriers),
            values=tuple(float(row[1]) for row in carriers),
        ),
    )

    elapsed = time.perf_counter() - started
    summary = {
        "schemaVersion": 1,
        "phase": "compiled_template_manual_review_resolution",
        "sourceDocuments": analysis["sourceDocuments"],
        "usableTemplates": analysis["usableTemplates"],
        "excludedDocuments": analysis["excludedDocuments"],
        "sourceRejectedDocuments": config.expected_source_rejected,
        "manualReviewDecisions": len(config.decisions),
        "manualExcludedDocuments": len(config.decisions),
        "unresolvedReviews": 0,
        "usableYield": analysis["usableYield"],
        "providerRequests": 0,
        "incrementalCostUsd": "0",
        "integrityChecksFailed": 0,
        "wallSeconds": elapsed,
    }
    publish_json("summary.json", summary)
    report = "\n".join(
        (
            "# Compiled-template manual resolution and EDA",
            "",
            f"- Source documents: **{analysis['sourceDocuments']}**.",
            f"- Usable certified templates: **{analysis['usableTemplates']}** "
            f"(**{analysis['usableYield']:.2%}**).",
            f"- Compiler-rejected documents retained as excluded: "
            f"**{config.expected_source_rejected}**.",
            f"- Review-required documents manually inspected and excluded: "
            f"**{len(config.decisions)}**.",
            "- Unresolved reviews: **0**.",
            "- Manual-resolution provider calls: **0**; incremental model cost: **$0**.",
            "",
            "All retained templates passed current-schema parsing, source and template hash "
            "checks, "
            "catalog equivalence, carrier-pin validation, and—when a recertified catalog was "
            "configured—exact source, label, result, and source-run lineage checks.",
            "",
            "The paid source run is unchanged. This artifact is a pinned adjudication overlay; "
            "`catalog.jsonl` contains only usable source templates, while every exclusion "
            "and its provenance remains in `resolved-outcomes.jsonl` and `manual-decisions.jsonl`.",
            "",
        )
    )
    publish_bytes("REPORT.md", report.encode("utf-8"))
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 1,
            "phase": "compiled_template_manual_review_resolution",
            "sourceDocuments": analysis["sourceDocuments"],
            "usableTemplates": analysis["usableTemplates"],
            "excludedDocuments": analysis["excludedDocuments"],
            "unresolvedReviews": 0,
        },
    )
    return staged.final_root
