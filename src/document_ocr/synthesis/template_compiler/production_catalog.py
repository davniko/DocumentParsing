"""Assemble adjudicated, current-schema template shards into a production catalog."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun

from .descendant import _validate_committed_run
from .descendant_models import PinnedCommittedRun
from .host import template_summary
from .models import CertifiedSemanticTemplate, NonEmptyText

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ProductionCatalogShard(BaseModel):
    model_config = _STRICT

    catalog_run: PinnedCommittedRun
    resolution_run: PinnedCommittedRun
    outcomes_file: Literal["results.jsonl", "outcomes.jsonl"]
    resolution_scope: Literal["complete_source", "review_subset"]
    expected_documents: Annotated[int, Field(gt=0)]
    expected_certified: Annotated[int, Field(gt=0)]
    expected_rejected: Annotated[int, Field(ge=0)]
    expected_review_required: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def counts_cover_shard(self) -> ProductionCatalogShard:
        if self.expected_certified + self.expected_rejected + self.expected_review_required != (
            self.expected_documents
        ):
            raise ValueError("shard status counts must cover every document")
        if self.resolution_scope == "review_subset" and self.expected_rejected:
            raise ValueError("review-subset resolution cannot leave compiler rejections unresolved")
        return self


class ProductionTemplateCatalogConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["bill_of_lading_production_template_catalog_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    shards: Annotated[tuple[ProductionCatalogShard, ...], Field(min_length=2)]
    expected_source_documents: Annotated[int, Field(gt=0)]
    expected_usable_templates: Annotated[int, Field(gt=0)]
    expected_excluded_documents: Annotated[int, Field(ge=0)]
    require_current_template_schema: Literal[6]
    require_zero_unresolved_reviews: Literal[True]
    require_disjoint_shards: Literal[True]

    @model_validator(mode="after")
    def aggregate_counts_match_shards(self) -> ProductionTemplateCatalogConfig:
        catalog_paths = tuple(row.catalog_run.path for row in self.shards)
        if len(set(catalog_paths)) != len(catalog_paths):
            raise ValueError("catalog shard runs must be unique")
        if sum(row.expected_documents for row in self.shards) != self.expected_source_documents:
            raise ValueError("shard document counts differ from aggregate source count")
        if sum(row.expected_certified for row in self.shards) != self.expected_usable_templates:
            raise ValueError("shard certified counts differ from aggregate usable count")
        if self.expected_usable_templates + self.expected_excluded_documents != (
            self.expected_source_documents
        ):
            raise ValueError("usable and excluded counts must cover every source document")
        return self


def load_production_catalog_config(path: Path) -> ProductionTemplateCatalogConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return ProductionTemplateCatalogConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_regular_file_bytes(path).splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: JSONL is empty")
    return tuple(rows)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _outcome_identity(row: Mapping[str, Any], *, path: Path) -> tuple[str, str]:
    document_id = row.get("document_id", row.get("documentId"))
    status = row.get("status")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError(f"outcome has no document identity: {path}")
    if status not in {"certified", "rejected", "review_required"}:
        raise ValueError(f"outcome has invalid status for {document_id}: {status!r}")
    return document_id, cast(str, status)


def _validate_template_case(
    *, root: Path, document_id: str, catalog_row: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    case_root = root / "cases" / document_id
    source_path = case_root / "source.txt"
    label_path = case_root / "source-label.json"
    template_path = case_root / "template.json"
    catalog_path = case_root / "catalog-row.json"
    source = read_regular_file_bytes(source_path)
    label = json.loads(read_regular_file_bytes(label_path))
    if not isinstance(label, dict):
        raise ValueError(f"source label is not an object: {document_id}")
    template = CertifiedSemanticTemplate.model_validate_json(
        read_regular_file_bytes(template_path), strict=True
    )
    if template.document_id != document_id:
        raise ValueError(f"template document identity differs: {document_id}")
    if template.source_sha256 != sha256_bytes(source):
        raise ValueError(f"template source hash differs: {document_id}")
    if template.certification.carrier_matches_source_label is not True:
        raise ValueError(f"template carrier is not source-pinned: {document_id}")
    expected_catalog = template_summary(template)
    if canonical_json_bytes(expected_catalog) != canonical_json_bytes(catalog_row):
        raise ValueError(f"catalog row differs from current template: {document_id}")
    case_catalog = json.loads(read_regular_file_bytes(catalog_path))
    if not isinstance(case_catalog, dict) or canonical_json_bytes(case_catalog) != (
        canonical_json_bytes(catalog_row)
    ):
        raise ValueError(f"case catalog artifact differs from catalog: {document_id}")
    return label, {
        "sourceSha256": sha256_bytes(source),
        "sourceLabelSha256": sha256_bytes(canonical_json_bytes(label)),
        "templateSha256": sha256_file(template_path),
    }


def _load_shard(*, project_root: Path, shard: ProductionCatalogShard) -> dict[str, Any]:
    catalog_root = _validate_committed_run(project_root, shard.catalog_run)
    resolution_root = _validate_committed_run(project_root, shard.resolution_run)
    catalog_rows = _read_jsonl(catalog_root / "catalog.jsonl")
    outcome_path = catalog_root / shard.outcomes_file
    outcome_rows = _read_jsonl(outcome_path)
    outcome_pairs = tuple(_outcome_identity(row, path=outcome_path) for row in outcome_rows)
    outcome_ids = tuple(row[0] for row in outcome_pairs)
    if len(set(outcome_ids)) != len(outcome_ids):
        raise ValueError(f"catalog shard outcomes contain duplicate documents: {catalog_root.name}")
    status_counts = Counter(row[1] for row in outcome_pairs)
    expected_statuses = Counter(
        {
            "certified": shard.expected_certified,
            "rejected": shard.expected_rejected,
            "review_required": shard.expected_review_required,
        }
    )
    if len(outcome_rows) != shard.expected_documents or status_counts != expected_statuses:
        raise ValueError(
            f"catalog shard outcome contract differs for {catalog_root.name}: "
            f"rows={len(outcome_rows)}, statuses={dict(status_counts)}"
        )
    certified_ids = tuple(
        document_id for document_id, status in outcome_pairs if status == "certified"
    )
    catalog_ids = tuple(cast(str, row.get("documentId")) for row in catalog_rows)
    if catalog_ids != certified_ids or len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError(f"catalog order differs from certified outcomes: {catalog_root.name}")

    manual_rows = _read_jsonl(resolution_root / "manual-decisions.jsonl")
    manual_ids = tuple(cast(str, row.get("documentId")) for row in manual_rows)
    if len(set(manual_ids)) != len(manual_ids) or any(
        row.get("resolution") != "excluded" for row in manual_rows
    ):
        raise ValueError(f"manual decisions are not unique exclusions: {resolution_root.name}")
    review_ids = tuple(
        document_id for document_id, status in outcome_pairs if status == "review_required"
    )
    if set(manual_ids) != set(review_ids):
        raise ValueError(
            f"manual decisions do not exactly close the shard review queue: {catalog_root.name}"
        )

    resolved_rows = _read_jsonl(resolution_root / "resolved-outcomes.jsonl")
    resolved_by_id = {cast(str, row.get("documentId")): row for row in resolved_rows}
    if len(resolved_by_id) != len(resolved_rows):
        raise ValueError(f"resolution outcomes contain duplicate documents: {resolution_root.name}")
    for document_id in manual_ids:
        row = resolved_by_id.get(document_id)
        if row is None or row.get("resolvedStatus") != "excluded":
            raise ValueError(f"manual exclusion is absent from resolution: {document_id}")
    if shard.resolution_scope == "complete_source":
        if set(resolved_by_id) != set(outcome_ids):
            raise ValueError(f"complete resolution does not cover its shard: {catalog_root.name}")
        resolved_usable = tuple(
            document_id
            for document_id, row in resolved_by_id.items()
            if row.get("resolvedStatus") == "usable"
        )
        if set(resolved_usable) != set(certified_ids):
            raise ValueError(f"resolved usable set differs from catalog: {catalog_root.name}")
        if read_regular_file_bytes(resolution_root / "catalog.jsonl") != read_regular_file_bytes(
            catalog_root / "catalog.jsonl"
        ):
            raise ValueError(
                f"resolved catalog bytes differ from source catalog: {catalog_root.name}"
            )

    template_receipts: dict[str, dict[str, str]] = {}
    for row in catalog_rows:
        document_id = cast(str, row["documentId"])
        _label, receipt = _validate_template_case(
            root=catalog_root, document_id=document_id, catalog_row=row
        )
        template_receipts[document_id] = receipt

    normalized_outcomes = tuple(
        {
            "documentId": document_id,
            "status": "usable" if status == "certified" else "excluded",
            "sourceStatus": status,
            "resolutionOrigin": (
                "compiled_certification"
                if status == "certified"
                else "manual_review_exclusion"
                if status == "review_required"
                else "compiler_rejection"
            ),
            "catalogRun": shard.catalog_run.path,
            "catalogRunCommitSha256": shard.catalog_run.commit_sha256,
            "resolutionRun": shard.resolution_run.path,
            "resolutionRunCommitSha256": shard.resolution_run.commit_sha256,
            **template_receipts.get(document_id, {}),
        }
        for document_id, status in outcome_pairs
    )
    return {
        "catalog_root": catalog_root,
        "resolution_root": resolution_root,
        "catalog_rows": catalog_rows,
        "outcomes": normalized_outcomes,
        "manual_rows": manual_rows,
        "all_ids": frozenset(outcome_ids),
        "lineage": {
            "catalogRun": shard.catalog_run.model_dump(mode="json"),
            "resolutionRun": shard.resolution_run.model_dump(mode="json"),
            "outcomesFile": shard.outcomes_file,
            "catalogSha256": sha256_file(catalog_root / "catalog.jsonl"),
            "outcomesSha256": sha256_file(outcome_path),
            "resolvedOutcomesSha256": sha256_file(resolution_root / "resolved-outcomes.jsonl"),
            "manualDecisionsSha256": sha256_file(resolution_root / "manual-decisions.jsonl"),
        },
    }


def _analyze(*, project_root: Path, config: ProductionTemplateCatalogConfig) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    for shard_config in config.shards:
        shard = _load_shard(project_root=project_root, shard=shard_config)
        shard_ids = cast(frozenset[str], shard["all_ids"])
        overlap = sorted(all_ids & shard_ids)
        if overlap:
            raise ValueError("production catalog shards overlap: " + ", ".join(overlap))
        all_ids.update(shard_ids)
        shards.append(shard)

    catalog_rows = tuple(
        row for shard in shards for row in cast(tuple[dict[str, Any], ...], shard["catalog_rows"])
    )
    outcomes = tuple(
        row for shard in shards for row in cast(tuple[dict[str, Any], ...], shard["outcomes"])
    )
    manual_rows = tuple(
        row for shard in shards for row in cast(tuple[dict[str, Any], ...], shard["manual_rows"])
    )
    counts = Counter(cast(str, row["status"]) for row in outcomes)
    if len(outcomes) != config.expected_source_documents:
        raise ValueError("assembled source coverage differs from configuration")
    if len(catalog_rows) != config.expected_usable_templates or counts != Counter(
        usable=config.expected_usable_templates,
        excluded=config.expected_excluded_documents,
    ):
        raise ValueError("assembled usable/excluded partition differs from configuration")
    if len({cast(str, row["documentId"]) for row in catalog_rows}) != len(catalog_rows):
        raise ValueError("assembled catalog contains duplicate documents")
    return {
        "shards": shards,
        "catalog_rows": catalog_rows,
        "outcomes": outcomes,
        "manual_rows": manual_rows,
    }


def preflight_production_template_catalog(
    *, project_root: Path, config_path: Path
) -> dict[str, Any]:
    config = load_production_catalog_config(config_path.resolve(strict=True))
    analyzed = _analyze(project_root=project_root.resolve(strict=True), config=config)
    return {
        "schemaVersion": 1,
        "sourceDocuments": len(cast(tuple[Any, ...], analyzed["outcomes"])),
        "usableTemplates": len(cast(tuple[Any, ...], analyzed["catalog_rows"])),
        "excludedDocuments": config.expected_excluded_documents,
        "manualReviewDecisions": len(cast(tuple[Any, ...], analyzed["manual_rows"])),
        "unresolvedReviews": 0,
        "providerRequests": 0,
        "estimatedCostUsd": "0",
    }


def build_production_template_catalog(*, project_root: Path, config_path: Path) -> Path:
    started = time.perf_counter()
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("production-catalog configuration is outside the project root") from error
    config = load_production_catalog_config(config_path)
    output_parent = (project_root / config.output_dir).resolve()
    if output_parent != project_root and project_root not in output_parent.parents:
        raise ValueError("production-catalog output directory escapes the project root")
    analyzed = _analyze(project_root=project_root, config=config)
    shards = cast(list[dict[str, Any]], analyzed["shards"])
    catalog_rows = cast(tuple[dict[str, Any], ...], analyzed["catalog_rows"])
    outcomes = cast(tuple[dict[str, Any], ...], analyzed["outcomes"])
    manual_rows = cast(tuple[dict[str, Any], ...], analyzed["manual_rows"])

    implementation_path = Path(__file__).resolve(strict=True)
    lineage = {
        "config": config.model_dump(mode="json"),
        "configSha256": sha256_file(config_path),
        "implementationSha256": sha256_file(implementation_path),
        "shards": tuple(cast(dict[str, Any], row["lineage"]) for row in shards),
    }
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(lineage)),
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
    publish_json("lineage.json", lineage)
    publish_bytes("catalog.jsonl", _jsonl_bytes(catalog_rows))
    publish_bytes("outcomes.jsonl", _jsonl_bytes(outcomes))
    publish_bytes("manual-decisions.jsonl", _jsonl_bytes(manual_rows))
    for shard in shards:
        root = cast(Path, shard["catalog_root"])
        shard_outcomes = {
            cast(str, outcome["documentId"]): outcome
            for outcome in cast(tuple[dict[str, Any], ...], shard["outcomes"])
        }
        for row in cast(tuple[dict[str, Any], ...], shard["catalog_rows"]):
            document_id = cast(str, row["documentId"])
            source_root = root / "cases" / document_id
            for name in ("source.txt", "source-label.json", "template.json", "catalog-row.json"):
                relative = f"cases/{document_id}/{name}"
                publish_bytes(relative, read_regular_file_bytes(source_root / name))
            publish_json(
                f"cases/{document_id}/lineage.json",
                shard_outcomes[document_id],
            )

    elapsed = time.perf_counter() - started
    summary = {
        "schemaVersion": 1,
        "phase": "production_compiled_template_catalog",
        "sourceShards": len(shards),
        "sourceDocuments": len(outcomes),
        "usableTemplates": len(catalog_rows),
        "excludedDocuments": config.expected_excluded_documents,
        "manualReviewDecisions": len(manual_rows),
        "unresolvedReviews": 0,
        "duplicateDocuments": 0,
        "currentTemplateSchemaVersion": 6,
        "providerRequests": 0,
        "incrementalCostUsd": "0",
        "wallSeconds": elapsed,
    }
    publish_json("summary.json", summary)
    publish_bytes(
        "REPORT.md",
        (
            "# Production compiled-template catalog\n\n"
            f"- Source documents accounted for: **{len(outcomes)}**.\n"
            f"- Current-schema usable templates: **{len(catalog_rows)}**.\n"
            f"- Fail-closed exclusions: **{config.expected_excluded_documents}**.\n"
            f"- Manual review decisions: **{len(manual_rows)}**.\n"
            "- Unresolved reviews: **0**.\n"
            "- Duplicate documents: **0**.\n"
            "- Assembly provider calls and incremental model cost: **0 / $0**.\n\n"
            "Every included case was re-parsed as a schema-v6 certified template and passed "
            "source-hash, carrier-pin, catalog-equivalence, disjointness, and committed-lineage "
            "checks. Excluded documents remain represented in `outcomes.jsonl`; no missing source "
            "fact was invented to promote a template.\n"
        ).encode(),
    )
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 1,
            "phase": "production_compiled_template_catalog",
            "sourceDocuments": len(outcomes),
            "usableTemplates": len(catalog_rows),
            "excludedDocuments": config.expected_excluded_documents,
            "unresolvedReviews": 0,
        },
    )
    return staged.final_root
