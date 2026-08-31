"""Pinned corpus runner for the PII-free party-structure SDV comparison."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.config import SynthesisPartyStructureBenchmarkConfig
from document_ocr.synthesis.country_registry import CountryRegistry, load_iso_country_registry
from document_ocr.synthesis.scenario_benchmark import (
    PartyStructureBenchmarkSettings,
    party_structure_candidate_specs,
    run_party_structure_benchmark,
)
from document_ocr.synthesis.scenario_views import build_scenario_views

_TABLE_NAMES = (
    "documents",
    "document_locations",
    "parties",
    "party_contacts",
    "containers",
    "cargo_groups",
    "cargo_hs_codes",
    "packages",
    "allocation_groups",
    "allocations",
    "dangerous_goods",
)
_COUNTRY_COLUMNS = {
    "document_locations": ("location_id", "country"),
    "parties": ("party_id", "country"),
}


class ScenarioBenchmarkPipelineError(RuntimeError):
    """A pinned scenario benchmark input or publication contract failed."""


def _resolve_file(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _resolve_directory(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink() or not path.resolve(strict=True).is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _read_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _read_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    expected_records: int,
    label: str,
) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} path or SHA-256 differs")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank rows are forbidden")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(value)
    if len(rows) != expected_records:
        raise ValueError(f"{label} expected {expected_records} records, found {len(rows)}")
    return tuple(rows)


def _load_prepared_tables(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], ...]]:
    files = manifest.get("files")
    row_counts = (manifest.get("domainProjection") or {}).get("tableRows")
    if not isinstance(files, Mapping) or not isinstance(row_counts, Mapping):
        raise ValueError("preparation manifest lacks files or domain table counts")
    tables: dict[str, tuple[dict[str, Any], ...]] = {}
    for name in _TABLE_NAMES:
        relative = f"tables/{name}.jsonl"
        receipt = files.get(relative)
        records = row_counts.get(name)
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(receipt.get("sha256"), str)
            or type(records) is not int
        ):
            raise ValueError(f"preparation manifest lacks a valid receipt for {relative}")
        path = (root / relative).resolve(strict=True)
        if path.parent != (root / "tables").resolve(strict=True):
            raise ValueError(f"prepared table escaped its pinned root: {relative}")
        tables[name] = _read_jsonl(
            path,
            expected_sha256=cast(str, receipt["sha256"]),
            expected_records=records,
            label=relative,
        )
    return tables


def _template_maps(
    rows: Sequence[Mapping[str, Any]], corpus_ids: frozenset[str]
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    by_document: dict[str, str] = {}
    members_by_template: dict[str, tuple[str, ...]] = {}
    for row in rows:
        template_id = row.get("template_id")
        members = row.get("member_document_ids")
        if not isinstance(template_id, str) or not isinstance(members, list):
            raise ValueError("template row has an invalid identity contract")
        frozen = tuple(members)
        if (
            not template_id
            or template_id in members_by_template
            or any(not isinstance(value, str) or not value for value in frozen)
            or len(frozen) != len(set(frozen))
        ):
            raise ValueError(f"template group is invalid: {template_id!r}")
        members_by_template[template_id] = frozen
        for document_id in frozen:
            if document_id in by_document:
                raise ValueError(f"document occurs in multiple templates: {document_id}")
            by_document[document_id] = template_id
    if set(by_document) != corpus_ids:
        raise ValueError("template groups do not cover the prepared corpus exactly")
    return by_document, members_by_template


def _partition_map(report: Mapping[str, Any], corpus_ids: frozenset[str]) -> dict[str, str]:
    try:
        outputs = report["inspection"]["partition"]["outputs"]
    except (KeyError, TypeError) as error:
        raise ValueError("partition report has an unexpected contract") from error
    if not isinstance(outputs, Mapping):
        raise ValueError("partition outputs must be an object")
    by_document: dict[str, str] = {}
    for split, raw in outputs.items():
        if not isinstance(split, str) or not isinstance(raw, Mapping):
            raise ValueError("partition split contract is invalid")
        values = raw.get("document_ids")
        if not isinstance(values, list) or raw.get("records") != len(values):
            raise ValueError(f"partition split count differs: {split}")
        for document_id in values:
            if not isinstance(document_id, str) or document_id in by_document:
                raise ValueError(f"partition document identity is invalid: {document_id!r}")
            by_document[document_id] = split
    if set(by_document) != corpus_ids:
        raise ValueError("partition report does not cover the prepared corpus exactly")
    return by_document


def _isolated_fit_ids(
    *,
    split: str,
    partition_by_document: Mapping[str, str],
    template_by_document: Mapping[str, str],
    members_by_template: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            document_id
            for document_id, partition in partition_by_document.items()
            if partition == split
            and all(
                partition_by_document[member] == split
                for member in members_by_template[template_by_document[document_id]]
            )
        )
    )


def _country_clean_fit_ids(
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_ids: Sequence[str],
    registry: CountryRegistry,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    fit = frozenset(fit_ids)
    issues_by_document: dict[str, list[dict[str, str]]] = defaultdict(list)
    surface_counts: Counter[str] = Counter()
    checked_present_cells = 0
    resolved_cells = 0
    missing_cells = 0
    for table, (identity_column, country_column) in _COUNTRY_COLUMNS.items():
        for row in tables[table]:
            document_id = row.get("document_id")
            if document_id not in fit:
                continue
            value = row.get(country_column)
            if value is None:
                missing_cells += 1
                continue
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(
                    f"{table}:{row.get(identity_column)}:{country_column} is malformed"
                )
            checked_present_cells += 1
            if registry.resolve(value) is None:
                surface_counts[value] += 1
                issues_by_document[cast(str, document_id)].append(
                    {
                        "table": table,
                        "rowId": cast(str, row[identity_column]),
                        "value": value,
                    }
                )
            else:
                resolved_cells += 1
    clean = tuple(document_id for document_id in fit_ids if document_id not in issues_by_document)
    audit = {
        "policy": "exclude_document_and_audit_v1",
        "inputDocuments": len(fit_ids),
        "eligibleDocuments": len(clean),
        "excludedDocuments": len(issues_by_document),
        "checkedPresentCells": checked_present_cells,
        "resolvedCells": resolved_cells,
        "unresolvedCells": sum(surface_counts.values()),
        "missingCells": missing_cells,
        "unresolvedSurfaceCounts": dict(sorted(surface_counts.items())),
        "excludedDocumentIssues": dict(sorted(issues_by_document.items())),
        "aliasesApplied": 0,
    }
    return clean, audit


def _prepare_output_root(path: Path) -> None:
    if path.is_symlink():
        raise ScenarioBenchmarkPipelineError("benchmark output cannot be a symbolic link")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise ScenarioBenchmarkPipelineError("benchmark output must be new or empty")
    else:
        path.mkdir(parents=True)


def _publish_top_manifest(
    root: Path,
    *,
    summary: Mapping[str, Any],
    benchmark_status: str,
) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ScenarioBenchmarkPipelineError(f"benchmark artifact is a symlink: {path}")
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_file(path),
                    "sizeBytes": path.stat().st_size,
                }
            )
    atomic_publish_json(
        root / "manifest.json",
        {
            "schemaVersion": 1,
            "status": f"{benchmark_status}_no_production_selection",
            "summary": dict(summary),
            "files": files,
        },
    )


def run_party_structure_benchmark_pipeline(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisPartyStructureBenchmarkConfig,
) -> dict[str, Any]:
    """Build the exact PII-free view and run the paired GPU comparison."""

    started = time.perf_counter()
    output_parent = Path(config.run.output_dir)
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    output_root = output_parent / config.run.run_id
    _prepare_output_root(output_root)
    try:
        preparation_root = _resolve_directory(
            project_root,
            config.inputs.preparation_root,
            label="synthesis preparation root",
        )
        preparation_manifest_path = _resolve_file(
            project_root,
            config.inputs.preparation_manifest.path,
            label="synthesis preparation manifest",
        )
        if preparation_manifest_path.parent != preparation_root:
            raise ValueError("preparation manifest is outside its pinned root")
        preparation_manifest = _read_json(
            preparation_manifest_path,
            expected_sha256=config.inputs.preparation_manifest.sha256,
            label="synthesis preparation manifest",
        )
        tables = _load_prepared_tables(preparation_root, preparation_manifest)
        document_ids = frozenset(cast(str, row["document_id"]) for row in tables["documents"])

        template_path = _resolve_file(
            project_root, config.inputs.template_groups.path, label="template groups"
        )
        template_rows = _read_jsonl(
            template_path,
            expected_sha256=config.inputs.template_groups.sha256,
            expected_records=config.inputs.template_groups.records,
            label="template groups",
        )
        template_by_document, members_by_template = _template_maps(template_rows, document_ids)
        partition_path = _resolve_file(
            project_root, config.inputs.partition_report.path, label="partition report"
        )
        partition_report = _read_json(
            partition_path,
            expected_sha256=config.inputs.partition_report.sha256,
            label="partition report",
        )
        partition_by_document = _partition_map(partition_report, document_ids)
        isolated_ids = _isolated_fit_ids(
            split=config.selection.split,
            partition_by_document=partition_by_document,
            template_by_document=template_by_document,
            members_by_template=members_by_template,
        )
        iso_path = _resolve_file(
            project_root, config.inputs.iso3166_snapshot.path, label="ISO-3166 snapshot"
        )
        countries = load_iso_country_registry(
            iso_path=iso_path,
            iso_sha256=config.inputs.iso3166_snapshot.sha256,
        )
        fit_ids, country_audit = _country_clean_fit_ids(
            tables=tables,
            fit_ids=isolated_ids,
            registry=countries,
        )
        views = build_scenario_views(
            tables=tables,
            fit_document_ids=fit_ids,
            partition_by_document=partition_by_document,
            template_by_document=template_by_document,
            country_resolver=countries,
            allowed_partition=config.selection.split,
        )
        candidates = party_structure_candidate_specs(
            enable_gpu=True,
            neural_epochs=config.modeling.neural_epochs,
            neural_batch_size=config.modeling.neural_batch_size,
        )
        settings = PartyStructureBenchmarkSettings(
            scope=config.modeling.scope,
            candidates=candidates,
            fold_count=config.modeling.fold_count,
            seeds=tuple(config.modeling.seeds),
            fold_seed=config.modeling.fold_seed,
            proposal_multiplier=config.modeling.proposal_multiplier,
            proposal_batch_rows=config.modeling.proposal_batch_rows,
        )
        input_contract = {
            "configSha256": sha256_file(config_path),
            "preparationManifestSha256": config.inputs.preparation_manifest.sha256,
            "templateGroupsSha256": config.inputs.template_groups.sha256,
            "partitionReportSha256": config.inputs.partition_report.sha256,
            "iso3166Sha256": config.inputs.iso3166_snapshot.sha256,
            "templateIsolatedDocuments": len(isolated_ids),
            "countryCleanDocuments": len(fit_ids),
            "partyStructureRows": len(views.party_structure.data),
            "partyStructureTemplates": len(set(views.party_structure.group_ids)),
            "rawPartyPiiModeled": False,
            "productionSelection": False,
        }
        atomic_publish_json(output_root / "input-contract.json", input_contract)
        atomic_publish_json(output_root / "country-resolution-audit.json", country_audit)
        atomic_publish_json(
            output_root / "scenario-view-audit.json",
            {
                "fitDocuments": views.audit.fit_document_count,
                "fitTemplates": views.audit.fit_template_count,
                "partyStructureRows": views.audit.party_structure_rows,
                "partyStructureInverseProjectionRows": (
                    views.audit.party_structure_inverse_projection_rows
                ),
                "routeFreightDocumentRows": views.audit.route_freight_document_rows,
                "countryCellsPresent": views.audit.country_cells_present,
                "countryCellsResolved": views.audit.country_cells_resolved,
                "countryCellsMissing": views.audit.country_cells_missing,
                "rawPiiColumnsExposed": views.audit.raw_pii_columns_exposed,
            },
        )
        result = run_party_structure_benchmark(
            view=views.party_structure,
            settings=settings,
            artifact_dir=output_root / "party-structure",
        )
        summary = {
            "runId": config.run.run_id,
            "benchmarkStatus": result.status,
            "rows": result.row_count,
            "templates": result.template_group_count,
            "requestedCandidates": list(result.requested_candidates),
            "completeCandidates": list(result.complete_candidates),
            "failedCandidates": list(result.failed_candidates),
            "completedCandidateRuns": len(result.runs),
            "failedCandidateRuns": len(result.candidate_failures),
            "productionSelectionPerformed": result.production_selection_performed,
            "elapsedSeconds": time.perf_counter() - started,
            "candidateSummaries": [row.to_dict() for row in result.summaries],
            "candidateFailures": [row.to_dict() for row in result.candidate_failures],
        }
        atomic_publish_json(output_root / "summary.json", summary)
        _publish_top_manifest(
            output_root,
            summary=summary,
            benchmark_status=result.status,
        )
        return summary
    except Exception as error:
        atomic_publish_json(
            output_root / "failure.json",
            {
                "status": "failed",
                "errorType": type(error).__name__,
                "message": str(error),
                "elapsedSeconds": time.perf_counter() - started,
            },
        )
        raise ScenarioBenchmarkPipelineError(
            f"party-structure benchmark pipeline failed: {type(error).__name__}: {error}"
        ) from error
