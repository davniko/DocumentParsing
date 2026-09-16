"""Losslessly join current-schema compiled-template runs into one committed catalog."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
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
from .models import CertifiedSemanticTemplate, ExtractionCaseResult, NonEmptyText

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TemplateCatalogJoinConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["bill_of_lading_compiled_template_catalog_join_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    sources: Annotated[tuple[PinnedCommittedRun, ...], Field(min_length=2)]
    expected_source_outcomes: Annotated[int, Field(gt=0)]
    expected_certified_templates: Annotated[int, Field(gt=0)]
    expected_review_required: Annotated[int, Field(ge=0)]
    require_no_rejected_outcomes: Literal[True]

    @model_validator(mode="after")
    def source_runs_are_unique(self) -> TemplateCatalogJoinConfig:
        paths = tuple(row.path for row in self.sources)
        if len(set(paths)) != len(paths):
            raise ValueError("template catalog join sources must be unique")
        if self.expected_certified_templates + self.expected_review_required != (
            self.expected_source_outcomes
        ):
            raise ValueError("certified and review counts must cover every expected outcome")
        return self


def load_join_config(path: Path) -> TemplateCatalogJoinConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return TemplateCatalogJoinConfig.model_validate_json(
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


def _source_lineage(root: Path, pin: PinnedCommittedRun) -> dict[str, Any]:
    summary_path = root / "summary.json"
    summary = json.loads(read_regular_file_bytes(summary_path))
    if not isinstance(summary, dict):
        raise ValueError(f"source summary is not an object: {root.name}")
    usage = summary.get("usage")
    all_usage = usage.get("all") if isinstance(usage, Mapping) else None
    cost = all_usage.get("estimatedCostUsd") if isinstance(all_usage, Mapping) else None
    if cost is not None:
        Decimal(cast(str, cost))
    return {
        "run": root.name,
        "path": pin.path,
        "commitSha256": pin.commit_sha256,
        "transactionSha256": pin.transaction_sha256,
        "summarySha256": sha256_file(summary_path),
        "reportedEstimatedCostUsd": cost,
    }


def _validated_source(
    *, root: Path, pin: PinnedCommittedRun
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    result_rows = _read_jsonl(root / "results.jsonl")
    catalog_rows = _read_jsonl(root / "catalog.jsonl")
    results = tuple(
        ExtractionCaseResult.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in result_rows
    )
    result_ids = tuple(row.document_id for row in results)
    if len(set(result_ids)) != len(result_ids):
        raise ValueError(f"source results contain duplicate documents: {root.name}")
    rejected = tuple(row.document_id for row in results if row.status == "rejected")
    if rejected:
        raise ValueError(
            f"source run contains rejected outcomes: {root.name}: " + ", ".join(rejected)
        )
    certified_ids = tuple(row.document_id for row in results if row.status == "certified")
    catalog_ids = tuple(row.get("documentId") for row in catalog_rows)
    if catalog_ids != certified_ids:
        raise ValueError(f"source catalog differs from certified result order: {root.name}")

    compact_results: list[dict[str, Any]] = []
    for result in results:
        case_root = root / "cases" / result.document_id
        source_path = case_root / "source.txt"
        label_path = case_root / "source-label.json"
        result_path = case_root / "result.json"
        for path in (source_path, label_path, result_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source case artifact is missing: {path}")
        source = read_regular_file_bytes(source_path)
        label = json.loads(read_regular_file_bytes(label_path))
        if not isinstance(label, dict):
            raise ValueError(f"source label is not an object: {result.document_id}")
        compact_results.append(
            {
                "documentId": result.document_id,
                "status": result.status,
                "rejectionReasons": result.rejection_reasons,
                "sourceSha256": sha256_bytes(source),
                "sourceLabelSha256": sha256_bytes(canonical_json_bytes(label)),
                "templateSha256": result.template_sha256,
                "sourceRun": root.name,
                "sourceRunCommitSha256": pin.commit_sha256,
                "sourceResultSha256": sha256_file(result_path),
            }
        )

    certified_results = tuple(row for row in results if row.status == "certified")
    for catalog, result in zip(catalog_rows, certified_results, strict=True):
        template_path = root / "cases" / result.document_id / "template.json"
        template = CertifiedSemanticTemplate.model_validate_json(
            read_regular_file_bytes(template_path), strict=True
        )
        if result.template_sha256 != sha256_file(template_path):
            raise ValueError(f"template hash differs from result: {result.document_id}")
        if canonical_json_bytes(template_summary(template)) != canonical_json_bytes(catalog):
            raise ValueError(f"catalog row differs from current template: {result.document_id}")
    return tuple(catalog_rows), tuple(compact_results), _source_lineage(root, pin)


def join_template_catalogs(*, project_root: Path, config_path: Path) -> Path:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("join configuration is outside the project root") from error
    config = load_join_config(config_path)
    output_parent = (project_root / config.output_dir).resolve()
    if output_parent != project_root and project_root not in output_parent.parents:
        raise ValueError("join output directory escapes the project root")

    validated: list[
        tuple[Path, PinnedCommittedRun, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]
    ] = []
    lineages: list[dict[str, Any]] = []
    all_ids: set[str] = set()
    for pin in config.sources:
        root = _validate_committed_run(project_root, pin)
        catalog, outcomes, lineage = _validated_source(root=root, pin=pin)
        ids = {cast(str, row["documentId"]) for row in outcomes}
        overlap = sorted(all_ids & ids)
        if overlap:
            raise ValueError("source runs overlap: " + ", ".join(overlap))
        all_ids.update(ids)
        validated.append((root, pin, catalog, outcomes))
        lineages.append(lineage)

    outcomes = tuple(row for _, _, _, rows in validated for row in rows)
    catalogs = tuple(row for _, _, rows, _ in validated for row in rows)
    statuses = Counter(cast(str, row["status"]) for row in outcomes)
    if len(outcomes) != config.expected_source_outcomes:
        raise ValueError("joined source-outcome count differs from configuration")
    if len(catalogs) != config.expected_certified_templates:
        raise ValueError("joined certified-template count differs from configuration")
    if statuses.get("review_required", 0) != config.expected_review_required:
        raise ValueError("joined review-required count differs from configuration")

    implementation_path = Path(__file__).resolve(strict=True)
    transaction = {
        "config": config.model_dump(mode="json"),
        "configSha256": sha256_file(config_path),
        "implementationSha256": sha256_file(implementation_path),
        "sources": lineages,
    }
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    staged.publish_json("config.json", config.model_dump(mode="json"))
    staged.publish_json("lineage.json", transaction)
    staged.publish_bytes("catalog.jsonl", _jsonl_bytes(catalogs))
    staged.publish_bytes("outcomes.jsonl", _jsonl_bytes(outcomes))

    expected = {"config.json", "lineage.json", "catalog.jsonl", "outcomes.jsonl"}
    for root, _, catalog_rows, source_outcomes in validated:
        certified = {cast(str, row["documentId"]) for row in catalog_rows}
        for outcome in source_outcomes:
            document_id = cast(str, outcome["documentId"])
            source_case = root / "cases" / document_id
            names: tuple[str, ...]
            if document_id in certified:
                relative_root = f"cases/{document_id}"
                names = ("source.txt", "source-label.json", "template.json", "catalog-row.json")
            else:
                relative_root = f"review-cases/{document_id}"
                names = ("source.txt", "source-label.json")
            for name in names:
                relative = f"{relative_root}/{name}"
                staged.publish_bytes(relative, read_regular_file_bytes(source_case / name))
                expected.add(relative)
            lineage_relative = f"{relative_root}/lineage.json"
            staged.publish_json(lineage_relative, outcome)
            expected.add(lineage_relative)

    reported_cost = sum(
        (
            Decimal(cast(str, row["reportedEstimatedCostUsd"]))
            for row in lineages
            if row["reportedEstimatedCostUsd"] is not None
        ),
        Decimal(0),
    )
    summary = {
        "schemaVersion": 1,
        "phase": "compiled_template_catalog_join",
        "sourceRuns": len(validated),
        "sourceOutcomes": len(outcomes),
        "certifiedTemplates": len(catalogs),
        "reviewRequired": statuses.get("review_required", 0),
        "rejected": statuses.get("rejected", 0),
        "duplicateDocuments": 0,
        "currentTemplateSchemaVersion": 5,
        "reportedSourceLineageEstimatedCostUsd": str(reported_cost),
    }
    report = "\n".join(
        (
            "# Compiled template catalog join",
            "",
            f"- Source outcomes: **{len(outcomes)}**.",
            f"- Current-schema certified templates: **{len(catalogs)}**.",
            f"- Review-required source outcomes: **{statuses.get('review_required', 0)}**.",
            "- Rejected source outcomes: **0**.",
            "- Duplicate document IDs: **0**.",
            "",
            "Only certified templates are present in `catalog.jsonl` and `cases/`. "
            "Fail-closed review outcomes are preserved separately in `review-cases/`.",
            "",
        )
    )
    staged.publish_json("summary.json", summary)
    staged.publish_bytes("REPORT.md", report.encode("utf-8"))
    expected.update({"summary.json", "REPORT.md"})
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 1,
            "phase": "compiled_template_catalog_join",
            "sourceOutcomes": len(outcomes),
            "certifiedTemplates": len(catalogs),
            "reviewRequired": statuses.get("review_required", 0),
        },
    )
    return staged.final_root
