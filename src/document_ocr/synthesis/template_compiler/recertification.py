"""Deterministically upgrade an immutable compiled-template catalog to schema v6."""

from __future__ import annotations

import json
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
from .models import CertifiedSemanticTemplate, NonEmptyText, SemanticBinding
from .semantic_plan import build_auxiliary_semantic_plan

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TemplateRecertificationConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["bill_of_lading_template_semantic_recertification_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    source_catalog: PinnedCommittedRun
    expected_source_outcomes: Annotated[int, Field(gt=0)]
    expected_source_certified: Annotated[int, Field(gt=0)]
    expected_source_review_required: Annotated[int, Field(ge=0)]
    expected_recertified: Annotated[int, Field(gt=0)]
    expected_review_required: Annotated[int, Field(ge=0)]
    require_no_rejected_outcomes: Literal[True]

    @model_validator(mode="after")
    def counts_are_closed(self) -> TemplateRecertificationConfig:
        if self.expected_source_certified + self.expected_source_review_required != (
            self.expected_source_outcomes
        ):
            raise ValueError("source certified and review counts must cover every outcome")
        if self.expected_recertified + self.expected_review_required != (
            self.expected_source_outcomes
        ):
            raise ValueError("recertified and review counts must cover every outcome")
        return self


def load_recertification_config(path: Path) -> TemplateRecertificationConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return TemplateRecertificationConfig.model_validate_json(
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


def _load_source_catalog(
    *, project_root: Path, config: TemplateRecertificationConfig
) -> tuple[Path, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    root = _validate_committed_run(project_root, config.source_catalog)
    outcomes = _read_jsonl(root / "outcomes.jsonl")
    catalog = _read_jsonl(root / "catalog.jsonl")
    if len(outcomes) != config.expected_source_outcomes:
        raise ValueError("source outcome count differs from recertification configuration")
    statuses = Counter(cast(str, row.get("status")) for row in outcomes)
    if statuses != Counter(
        {
            "certified": config.expected_source_certified,
            "review_required": config.expected_source_review_required,
        }
    ):
        raise ValueError(f"source outcome statuses differ: {dict(statuses)}")
    catalog_ids = tuple(row.get("documentId") for row in catalog)
    certified_ids = tuple(
        row.get("documentId") for row in outcomes if row.get("status") == "certified"
    )
    if set(catalog_ids) != set(certified_ids) or len(catalog_ids) != len(certified_ids):
        raise ValueError("source catalog does not exactly cover certified outcomes")
    return root, outcomes, catalog


def _upgrade_template(*, case_root: Path) -> tuple[CertifiedSemanticTemplate, dict[str, int]]:
    raw_bytes = read_regular_file_bytes(case_root / "source.txt")
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"template source is not UTF-8: {case_root.name}") from error
    label = json.loads(read_regular_file_bytes(case_root / "source-label.json"))
    payload = json.loads(read_regular_file_bytes(case_root / "template.json"))
    if not isinstance(label, dict) or not isinstance(payload, dict):
        raise ValueError(f"template case has non-object JSON: {case_root.name}")
    if payload.get("schema_version") != 5 or payload.get("compiler") != (
        "carrier_bound_semantic_template_v5"
    ):
        raise ValueError(f"source template is not schema v5: {case_root.name}")
    if payload.get("source_sha256") != sha256_bytes(raw_bytes):
        raise ValueError(f"source hash differs from template: {case_root.name}")
    raw_bindings = payload.get("bindings")
    if not isinstance(raw_bindings, list):
        raise ValueError(f"source template bindings are absent: {case_root.name}")
    bindings = tuple(
        SemanticBinding.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in raw_bindings
    )
    plan = build_auxiliary_semantic_plan(raw=raw, bindings=bindings, source_target=label)
    upgraded = {
        **payload,
        "schema_version": 6,
        "compiler": "carrier_bound_semantic_template_v6",
        "auxiliary_semantic_plan": plan.model_dump(mode="json"),
        "certification": {
            **cast(dict[str, Any], payload["certification"]),
            "auxiliary_semantic_plan_valid": True,
        },
    }
    template = CertifiedSemanticTemplate.model_validate_json(
        canonical_json_bytes(upgraded), strict=True
    )
    counts: dict[str, int] = dict(Counter(str(row.disposition) for row in plan.dispositions))
    counts.update(
        {
            "entities": len(plan.entities),
            "composite_numbers": len(plan.composite_numbers),
            "document_sequences": len(plan.document_sequences),
        }
    )
    return template, counts


def _preflight(
    *, project_root: Path, config: TemplateRecertificationConfig
) -> tuple[
    Path,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, CertifiedSemanticTemplate],
    dict[str, dict[str, int]],
]:
    root, outcomes, catalog = _load_source_catalog(project_root=project_root, config=config)
    upgraded: dict[str, CertifiedSemanticTemplate] = {}
    counts: dict[str, dict[str, int]] = {}
    for row in catalog:
        document_id = row.get("documentId")
        if not isinstance(document_id, str):
            raise ValueError("source catalog contains an invalid document ID")
        template, document_counts = _upgrade_template(case_root=root / "cases" / document_id)
        if template.document_id != document_id:
            raise ValueError(f"upgraded template identity differs: {document_id}")
        upgraded[document_id] = template
        counts[document_id] = document_counts
    if len(upgraded) != config.expected_recertified:
        raise ValueError("recertified template count differs from configuration")
    if len(outcomes) - len(upgraded) != config.expected_review_required:
        raise ValueError("post-recertification review count differs from configuration")
    return root, outcomes, catalog, upgraded, counts


def preflight_template_recertification(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_recertification_config(config_path)
    _root, outcomes, _catalog, upgraded, counts = _preflight(
        project_root=project_root, config=config
    )
    totals = Counter(
        {
            key: sum(row.get(key, 0) for row in counts.values())
            for key in {key for row in counts.values() for key in row}
        }
    )
    return {
        "schemaVersion": 1,
        "sourceOutcomes": len(outcomes),
        "recertifiedTemplates": len(upgraded),
        "reviewRequired": len(outcomes) - len(upgraded),
        "templateSchemaVersion": 6,
        "auxiliarySemanticTotals": dict(sorted(totals.items())),
        "providerRequests": 0,
        "estimatedCostUsd": "0",
    }


def recertify_template_catalog(*, project_root: Path, config_path: Path) -> Path:
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("recertification configuration is outside the project root") from error
    config = load_recertification_config(config_path)
    output_parent = (project_root / config.output_dir).resolve()
    if output_parent != project_root and project_root not in output_parent.parents:
        raise ValueError("recertification output directory escapes the project root")
    source_root, source_outcomes, source_catalog, upgraded, counts = _preflight(
        project_root=project_root,
        config=config,
    )
    implementation_files = (
        Path(__file__).resolve(strict=True),
        Path(__file__).with_name("semantic_plan.py").resolve(strict=True),
        Path(__file__).with_name("models.py").resolve(strict=True),
    )
    transaction = {
        "config": config.model_dump(mode="json"),
        "configSha256": sha256_file(config_path),
        "implementationSha256": {path.name: sha256_file(path) for path in implementation_files},
        "sourceCatalog": {
            "path": config.source_catalog.path,
            "commitSha256": config.source_catalog.commit_sha256,
            "transactionSha256": config.source_catalog.transaction_sha256,
        },
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

    source_outcomes_by_id = {cast(str, row["documentId"]): row for row in source_outcomes}
    catalog_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    expected = {"config.json", "lineage.json"}
    for source_row in source_catalog:
        document_id = cast(str, source_row["documentId"])
        source_case = source_root / "cases" / document_id
        template = upgraded[document_id]
        template_bytes = canonical_json_bytes(template.model_dump(mode="json")) + b"\n"
        catalog_row = template_summary(template)
        catalog_rows.append(catalog_row)
        outcome = {
            **source_outcomes_by_id[document_id],
            "status": "certified",
            "rejectionReasons": [],
            "templateSha256": sha256_bytes(template_bytes),
            "semanticRecertification": {
                "sourceTemplateSchemaVersion": 5,
                "templateSchemaVersion": 6,
                "providerRequests": 0,
                "auxiliarySemanticCounts": counts[document_id],
            },
        }
        outcome_rows.append(outcome)
        relative_root = f"cases/{document_id}"
        for name in ("source.txt", "source-label.json"):
            relative = f"{relative_root}/{name}"
            staged.publish_bytes(relative, read_regular_file_bytes(source_case / name))
            expected.add(relative)
        for relative, payload in (
            (f"{relative_root}/template.json", template_bytes),
            (f"{relative_root}/catalog-row.json", canonical_json_bytes(catalog_row) + b"\n"),
            (f"{relative_root}/lineage.json", canonical_json_bytes(outcome) + b"\n"),
        ):
            staged.publish_bytes(relative, payload)
            expected.add(relative)

    for source_outcome in source_outcomes:
        if source_outcome.get("status") != "review_required":
            continue
        document_id = cast(str, source_outcome["documentId"])
        outcome_rows.append(dict(source_outcome))
        source_case = source_root / "review-cases" / document_id
        relative_root = f"review-cases/{document_id}"
        for name in ("source.txt", "source-label.json"):
            relative = f"{relative_root}/{name}"
            staged.publish_bytes(relative, read_regular_file_bytes(source_case / name))
            expected.add(relative)
        lineage_relative = f"{relative_root}/lineage.json"
        staged.publish_bytes(lineage_relative, canonical_json_bytes(source_outcome) + b"\n")
        expected.add(lineage_relative)

    staged.publish_bytes("catalog.jsonl", _jsonl_bytes(catalog_rows))
    staged.publish_bytes("outcomes.jsonl", _jsonl_bytes(outcome_rows))
    expected.update({"catalog.jsonl", "outcomes.jsonl"})
    totals = {
        key: sum(row.get(key, 0) for row in counts.values())
        for key in sorted({key for row in counts.values() for key in row})
    }
    summary = {
        "schemaVersion": 1,
        "phase": "compiled_template_semantic_recertification",
        "sourceOutcomes": len(source_outcomes),
        "recertifiedTemplates": len(upgraded),
        "reviewRequired": len(source_outcomes) - len(upgraded),
        "rejected": 0,
        "currentTemplateSchemaVersion": 6,
        "auxiliarySemanticTotals": totals,
        "providerRequests": 0,
        "estimatedCostUsd": "0",
    }
    report = "\n".join(
        (
            "# Compiled template semantic re-certification",
            "",
            f"- Source outcomes: **{len(source_outcomes)}**.",
            f"- Schema-v6 certified templates: **{len(upgraded)}**.",
            f"- Review-required outcomes: **{len(source_outcomes) - len(upgraded)}**.",
            "- Provider requests: **0**.",
            "- Incremental model cost: **$0**.",
            "",
            "Every mutable source-only binding has one explicit semantic disposition. "
            "Existing review cases remain excluded from the certified catalog.",
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
            "phase": "compiled_template_semantic_recertification",
            "sourceOutcomes": len(source_outcomes),
            "certifiedTemplates": len(upgraded),
            "reviewRequired": len(source_outcomes) - len(upgraded),
            "templateSchemaVersion": 6,
        },
    )
    return staged.final_root
