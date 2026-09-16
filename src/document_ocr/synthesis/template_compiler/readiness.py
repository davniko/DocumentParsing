"""Zero-provider readiness gates for carrier-bound template compilation.

The paid compiler is deliberately not the first place where deterministic host or corpus-shape
defects should be discovered.  This module runs the real production preprocessing path over every
carrier-eligible source and replays previously paid checkpoints through the current host contract.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import read_regular_file_bytes

from .coherence import CoherenceReviewRequired
from .host import all_risk_candidates, certify_template
from .models import CriticAgentOutput, ExtractionCaseResult, ExtractionConfig
from .pipeline import (
    _config_inputs,
    _corpus_carrier_resolution_report,
    _preflight_summary,
    _restore_state_checkpoint,
    _validated_project_root,
    load_config,
)
from .selection import build_selection_manifest

_PAGE_MARKER = re.compile(r"^--- PAGE \d+ ---$")
_NUMERIC_RANGE = re.compile(
    r"(?i)(?<![A-Z0-9])\d{1,6}[ \t\r\n]*(?:[-\u2013\u2014~]|\bTO\b)"
    r"[ \t\r\n]*\d{1,6}(?![A-Z0-9])"
)
_DANGEROUS_GOODS_UN = re.compile(r"(?i)\bUN\s*(?:(?:NO\.?|NUMBER)\s*)?:?\s*(?:UN\s*)?\d{4}\b")
_DANGEROUS_GOODS_LABELED_UN = re.compile(r"(?i)\bUN\s*(?:NO\.?|NUMBER)\s*:?\s*(?:UN\s*)?\d{4}\b")
_DANGEROUS_GOODS_REPEATED_UN = re.compile(r"(?i)\bUN\s*(?:NO\.?|NUMBER)?\s*:?\s*UN\s*\d{4}\b")
_STANDALONE_DANGEROUS_GOODS_CLASS = re.compile(r"(?i)^\s*CLASS\s*:?\s*[1-9](?:\.[1-9])?\s*$")
_STANDALONE_MEASUREMENT_UNIT = re.compile(
    r"(?i)^\s*(?:KGS?|KG|LBS?|MT|TONS?|CBM|M3|CU\.?\s*M\.?)\s*$"
)
_PACKAGE_COLUMN_CAPTION = re.compile(
    r"(?i)(?:QUANTITY|NUMBER\s+OF\s+PACKAGES?|PACKAGES?|PKGS?|CARTONS?|BOXES?|PALLETS?)"
)
_NUMERIC_ONLY_LINE = re.compile(r"^\s*\d[\d,. ]*\s*$")


def _required_nonnegative_integer(feature: Mapping[str, Any], key: str) -> int:
    value = feature.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"document feature {key} must be a non-negative integer")
    return value


def structural_variant_tags(
    *, raw: str, feature: Mapping[str, Any], source_target: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return high-recall audit strata, never rendering or semantic decisions.

    Lexical tags only prove that a structural family is represented in a validation cohort.  They
    do not authorize a binding, rewrite text, or bypass compiler/critic review.
    """

    if not isinstance(raw, str) or not raw:
        raise ValueError("raw OCR text must be a non-empty string")
    pages = _required_nonnegative_integer(feature, "page_count")
    goods = _required_nonnegative_integer(feature, "goods_group_count")
    packages = _required_nonnegative_integer(feature, "package_fact_count")
    containers = _required_nonnegative_integer(feature, "container_count")
    allocations = _required_nonnegative_integer(feature, "allocation_row_count")
    dangerous_goods = _required_nonnegative_integer(feature, "dangerous_goods_count")
    temperatures = _required_nonnegative_integer(feature, "temperature_setting_count")
    contacts = _required_nonnegative_integer(feature, "party_with_contact_details_count")

    tags: set[str] = set()
    if pages > 1:
        tags.add("multi_page")
    if goods > 1:
        tags.add("multi_cargo")
    if packages > 1:
        tags.add("multi_package")
    if containers > 1:
        tags.add("multi_container")
    if allocations > 1:
        tags.add("multi_allocation")
    if dangerous_goods:
        tags.add("dangerous_goods")
    if temperatures:
        tags.add("temperature_control")
    if contacts:
        tags.add("party_contacts")
    if _NUMERIC_RANGE.search(raw):
        tags.add("numeric_range_surface")
    if any(
        candidate.kind == "relational_numeric_range"
        for candidate in all_risk_candidates(raw, (), source_target)
    ):
        tags.add("structured_quantity_range_surface")
    if _DANGEROUS_GOODS_UN.search(raw):
        tags.add("dangerous_goods_un_surface")
    if _DANGEROUS_GOODS_LABELED_UN.search(raw):
        tags.add("dangerous_goods_labeled_un_variant")
    if _DANGEROUS_GOODS_REPEATED_UN.search(raw):
        tags.add("dangerous_goods_repeated_un_prefix")

    lines = raw.splitlines()
    if any(_STANDALONE_DANGEROUS_GOODS_CLASS.fullmatch(line) for line in lines):
        tags.add("standalone_dangerous_goods_class")
    if any(_STANDALONE_MEASUREMENT_UNIT.fullmatch(line) for line in lines):
        tags.add("standalone_measurement_unit")
    for index, line in enumerate(lines):
        if _PACKAGE_COLUMN_CAPTION.search(line) is None:
            continue
        for following in lines[index + 1 :]:
            if not following.strip() or _PAGE_MARKER.fullmatch(following):
                break
            if _NUMERIC_ONLY_LINE.fullmatch(following):
                tags.add("columnar_package_quantity")
                break
        if "columnar_package_quantity" in tags:
            break
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")
    allocation_groups = patch.get("cargoAllocationGroups")
    if allocation_groups is not None:
        if not isinstance(allocation_groups, list) or any(
            not isinstance(row, Mapping) for row in allocation_groups
        ):
            raise ValueError("source target cargoAllocationGroups must be a list of objects")
        allocation_rows = 0
        for group in cast(Sequence[Mapping[str, Any]], allocation_groups):
            coverage = group.get("coverage")
            if isinstance(coverage, str) and coverage:
                tags.add("allocation_coverage:" + coverage)
            rows = group.get("allocations")
            if not isinstance(rows, list):
                raise ValueError("source target allocation group lacks an allocations list")
            allocation_rows += len(rows)
        if len(allocation_groups) > 1:
            tags.add("multiple_allocation_groups")
        if allocation_rows:
            magnitude = (
                "1"
                if allocation_rows == 1
                else "2-9"
                if allocation_rows < 10
                else "10-99"
                if allocation_rows < 100
                else "100+"
            )
            tags.add("allocation_row_magnitude:" + magnitude)
    return tuple(sorted(tags))


def _structural_coverage(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    eligible_document_ids: Sequence[str],
    selected_document_ids: Sequence[str],
) -> dict[str, Any]:
    sources = {str(row["documentId"]): row for row in source_rows}
    features = {str(row["document_id"]): row for row in feature_rows}
    eligible = tuple(eligible_document_ids)
    selected = tuple(selected_document_ids)
    if len(set(eligible)) != len(eligible) or len(set(selected)) != len(selected):
        raise ValueError("readiness coverage document IDs must be unique")
    if not set(selected) <= set(eligible):
        raise ValueError("selected readiness cohort is not a subset of eligible documents")
    if set(sources) != set(features):
        raise ValueError("source corpus and document features have different document IDs")

    tags_by_id = {
        document_id: structural_variant_tags(
            raw=cast(str, sources[document_id]["joinedRawText"]),
            feature=features[document_id],
            source_target=cast(Mapping[str, Any], sources[document_id]["target"]),
        )
        for document_id in eligible
    }
    corpus_counts = Counter(tag for tags in tags_by_id.values() for tag in tags)
    selected_counts = Counter(tag for document_id in selected for tag in tags_by_id[document_id])
    missing = tuple(sorted(set(corpus_counts) - set(selected_counts)))

    required_tags = set(corpus_counts)
    covered: set[str] = set()
    prefix_size: int | None = None
    for index, document_id in enumerate(selected, start=1):
        covered.update(tags_by_id[document_id])
        if covered == required_tags:
            prefix_size = index
            break

    def distinct_values(document_ids: Sequence[str], key: str) -> set[str]:
        return {str(features[document_id][key]) for document_id in document_ids}

    return {
        "corpusTagCounts": dict(sorted(corpus_counts.items())),
        "selectedTagCounts": dict(sorted(selected_counts.items())),
        "corpusStructuralTags": len(corpus_counts),
        "selectedStructuralTags": len(selected_counts),
        "missingStructuralTags": missing,
        "selectedCoversCorpusStructuralTags": not missing,
        "minimumSelectedPrefixForFullStructuralCoverage": prefix_size,
        "hardCanaryPrefixDocumentIds": selected[:prefix_size] if prefix_size is not None else (),
        "documentTypeCoverage": {
            "corpus": len(distinct_values(eligible, "document_type")),
            "selected": len(distinct_values(selected, "document_type")),
        },
        "carrierFamilyCoverage": {
            "corpus": len(distinct_values(eligible, "carrier_family")),
            "selected": len(distinct_values(selected, "carrier_family")),
        },
        "templateProxyCoverage": {
            "corpus": len(distinct_values(eligible, "template_proxy_id")),
            "selected": len(distinct_values(selected, "template_proxy_id")),
        },
    }


def _full_corpus_config(
    *, config: ExtractionConfig, eligible_document_ids: Sequence[str]
) -> ExtractionConfig:
    payload = config.model_dump(mode="json")
    payload["workflow"]["documents"] = len(eligible_document_ids)
    payload["workflow"]["provider_launch_authorized"] = False
    payload["pinned_document_ids"] = list(eligible_document_ids)
    payload["excluded_document_ids"] = []
    payload["resume_from"] = None
    return ExtractionConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _resolve_checkpoint_roots(
    *, project_root: Path, checkpoint_roots: Sequence[Path]
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for supplied in checkpoint_roots:
        candidate = supplied if supplied.is_absolute() else project_root / supplied
        if candidate.is_symlink():
            raise ValueError(f"checkpoint root must not be a symbolic link: {supplied}")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"checkpoint root must be a directory: {supplied}")
        try:
            root.relative_to(project_root)
        except ValueError as error:
            raise ValueError(f"checkpoint root escapes the project: {supplied}") from error
        case_root = root / "cases"
        checkpoints = tuple(sorted(case_root.glob("*/state-checkpoint.json")))
        if not checkpoints:
            raise ValueError(f"checkpoint root contains no case checkpoints: {supplied}")
        resolved.append(root)
    if len(set(resolved)) != len(resolved):
        raise ValueError("checkpoint roots must be unique")
    return tuple(resolved)


def _validate_expected_review_checkpoint(
    *, result_path: Path, document_id: str, error: CoherenceReviewRequired
) -> None:
    if not result_path.is_file():
        raise ValueError("review-required checkpoint has no case result")
    result = ExtractionCaseResult.model_validate_json(
        read_regular_file_bytes(result_path), strict=True
    )
    expected_reason = f"semantic coherence requires review: {error}"
    if result.document_id != document_id:
        raise ValueError("review-required checkpoint result has a different document ID")
    if result.status != "review_required" or expected_reason not in result.rejection_reasons:
        raise ValueError(
            "checkpoint raised semantic review that is not recorded by its case result"
        )
    if result.template_sha256 is not None:
        raise ValueError("review-required checkpoint unexpectedly published a template")


def _replay_checkpoints(
    *,
    project_root: Path,
    checkpoint_roots: Sequence[Path],
    source_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    roots = _resolve_checkpoint_roots(project_root=project_root, checkpoint_roots=checkpoint_roots)
    sources = {str(row["documentId"]): row for row in source_rows}
    features = {str(row["document_id"]): row for row in feature_rows}
    statuses: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    replayed_document_ids: set[str] = set()
    checkpoint_count = 0
    for root in roots:
        for checkpoint_path in sorted((root / "cases").glob("*/state-checkpoint.json")):
            checkpoint_count += 1
            relative = str(checkpoint_path.relative_to(project_root))
            try:
                raw_checkpoint = json.loads(read_regular_file_bytes(checkpoint_path))
                if raw_checkpoint.get("schema_version") != 2:
                    raise ValueError("checkpoint schema is not the current schema version 2")
                document_id = raw_checkpoint.get("document_id")
                if not isinstance(document_id, str) or document_id not in sources:
                    raise ValueError("checkpoint document is absent from the pinned source corpus")
                replayed_document_ids.add(document_id)
                source = sources[document_id]
                raw = cast(str, source["joinedRawText"])
                source_target = cast(Mapping[str, Any], source["target"])
                result_path = checkpoint_path.with_name("result.json")
                try:
                    drafts, assessment, semantic_only, coherence, revisions = (
                        _restore_state_checkpoint(
                            checkpoint_payload=read_regular_file_bytes(checkpoint_path),
                            document_id=document_id,
                            raw=raw,
                            source_target=source_target,
                        )
                    )
                except CoherenceReviewRequired as error:
                    _validate_expected_review_checkpoint(
                        result_path=result_path,
                        document_id=document_id,
                        error=error,
                    )
                    statuses["current_host_review_required"] += 1
                    continue
                status = "current_host_restored"
                if result_path.is_file():
                    result = ExtractionCaseResult.model_validate_json(
                        read_regular_file_bytes(result_path), strict=True
                    )
                    final_stage = result.critic_stages[-1] if result.critic_stages else None
                    if (
                        final_stage is not None
                        and final_stage.status == "success"
                        and isinstance(final_stage.output, Mapping)
                        and final_stage.output.get("verdict") == "pass"
                    ):
                        final_review = CriticAgentOutput.model_validate_json(
                            json.dumps(
                                final_stage.output,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        try:
                            certify_template(
                                raw=raw,
                                document_id=document_id,
                                feature=features[document_id],
                                source_target=source_target,
                                assessment=assessment,
                                drafts=drafts,
                                risks=all_risk_candidates(raw, (), source_target),
                                critic_outputs=(*revisions, final_review),
                                semantic_only_target_facts=semantic_only,
                                coherence_constraints=coherence,
                            )
                        except CoherenceReviewRequired as error:
                            _validate_expected_review_checkpoint(
                                result_path=result_path,
                                document_id=document_id,
                                error=error,
                            )
                            status = "current_host_review_required"
                        else:
                            status = "current_host_certified"
                statuses[status] += 1
            except Exception as error:  # Every incompatible checkpoint is reported, never skipped.
                failures.append(
                    {
                        "checkpoint": relative,
                        "errorType": type(error).__name__,
                        "diagnostic": str(error),
                    }
                )
    return {
        "configured": True,
        "roots": tuple(str(root.relative_to(project_root)) for root in roots),
        "checkpoints": checkpoint_count,
        "uniqueDocuments": len(replayed_document_ids),
        "statusCounts": dict(sorted(statuses.items())),
        "failures": tuple(failures),
        "passed": not failures,
    }


def preflight_corpus_readiness(
    *, project_root: Path, config_path: Path, checkpoint_roots: Sequence[Path] = ()
) -> dict[str, Any]:
    """Audit every future carrier-eligible source and optional paid checkpoints with zero calls."""

    project_root = _validated_project_root(project_root=project_root, config_path=config_path)
    config = load_config(config_path)
    (
        source_rows,
        feature_rows,
        anchor_rows,
        compiler_prompt,
        critic_prompt,
        compiler_repair_prompt,
        critic_audit_prompt,
        critic_plan_prompt,
    ) = _config_inputs(config, project_root)
    carrier_report = _corpus_carrier_resolution_report(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )
    eligible_document_ids = tuple(
        sorted(
            cast(str, row["documentId"])
            for row in carrier_report["rows"]
            if row["eligibleForCarrierBoundExtraction"]
        )
    )
    if not eligible_document_ids:
        raise ValueError("source corpus contains no carrier-eligible templates")
    audit_config = _full_corpus_config(config=config, eligible_document_ids=eligible_document_ids)
    corpus_manifest = build_selection_manifest(
        config=audit_config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )
    corpus_preflight = _preflight_summary(
        config=audit_config,
        manifest=corpus_manifest,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        compiler_prompt=compiler_prompt,
        critic_prompt=critic_prompt,
        compiler_repair_prompt=compiler_repair_prompt,
        critic_audit_prompt=critic_audit_prompt,
        critic_plan_prompt=critic_plan_prompt,
    )
    selected_manifest = build_selection_manifest(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )
    coverage = _structural_coverage(
        source_rows=source_rows,
        feature_rows=feature_rows,
        eligible_document_ids=eligible_document_ids,
        selected_document_ids=tuple(row.document_id for row in selected_manifest.rows),
    )
    issue_counts: Counter[str] = Counter()
    review_rows: list[dict[str, Any]] = []
    for row in cast(Sequence[Mapping[str, Any]], corpus_preflight["cases"]):
        issues = cast(Sequence[str], row["sourceTemplateIntegrityIssues"])
        if issues:
            review_rows.append({"documentId": row["documentId"], "issues": tuple(issues)})
            issue_counts.update(issue.split(":", 1)[0] for issue in issues)

    def largest(metric: str) -> tuple[dict[str, Any], ...]:
        rows = sorted(
            cast(Sequence[Mapping[str, Any]], corpus_preflight["cases"]),
            key=lambda row: (cast(int, row[metric]), cast(str, row["documentId"])),
            reverse=True,
        )[:10]
        return tuple({"documentId": row["documentId"], metric: row[metric]} for row in rows)

    checkpoint_replay: dict[str, Any]
    if checkpoint_roots:
        checkpoint_replay = _replay_checkpoints(
            project_root=project_root,
            checkpoint_roots=checkpoint_roots,
            source_rows=source_rows,
            feature_rows=feature_rows,
        )
    else:
        checkpoint_replay = {
            "configured": False,
            "roots": (),
            "checkpoints": 0,
            "uniqueDocuments": 0,
            "statusCounts": {},
            "failures": (),
            "passed": None,
        }
    offline_gate_passed = bool(coverage["selectedCoversCorpusStructuralTags"]) and (
        checkpoint_replay["passed"] is not False
    )
    return {
        "schemaVersion": 1,
        "phase": "template_compilation_readiness",
        "providerRequests": 0,
        "providerLaunchAuthorized": False,
        "sourceCorpusDocuments": len(source_rows),
        "carrierEligibleDocuments": len(eligible_document_ids),
        "requiresExternalCarrierEnrichment": carrier_report["requiresExternalCarrierEnrichment"],
        "carrierClassificationCounts": carrier_report["classificationCounts"],
        "deterministicallyPreflightedDocuments": corpus_preflight["documents"],
        "providerEligibleAfterIntegrityGate": corpus_preflight["providerEligibleDocuments"],
        "sourceIntegrityReviewDocuments": corpus_preflight["sourceTemplateIntegrityReviewRequired"],
        "sourceIntegrityIssueCounts": dict(sorted(issue_counts.items())),
        "sourceIntegrityReviews": tuple(review_rows),
        "carrierResolutionRequired": corpus_preflight["carrierResolutionRequired"],
        "sourceBytes": corpus_preflight["totalSourceBytes"],
        "acceptedAnchorBindings": corpus_preflight["totalAcceptedAnchorBindings"],
        "riskCandidates": corpus_preflight["totalRiskCandidates"],
        "maximumCompilerRequestBytes": corpus_preflight["maxCompilerRequestBytes"],
        "largestCompilerRequests": largest("compilerRequestBytes"),
        "largestRiskInventories": largest("riskCandidates"),
        "selectedTransferDocuments": len(selected_manifest.rows),
        "structuralCoverage": coverage,
        "checkpointReplay": checkpoint_replay,
        "offlineGatePassed": offline_gate_passed,
    }
