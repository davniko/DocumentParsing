"""End-to-end deterministic synthesis smoke planning and draft publication."""

from __future__ import annotations

import json
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import (
    AtomicConflictError,
    atomic_publish_bytes,
    atomic_publish_json,
    json_artifact_bytes,
    read_regular_file_bytes,
)
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.anchors import page_texts
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.config import SynthesisDeterministicSmokeConfig
from document_ocr.synthesis.domain import rows_for_document
from document_ocr.synthesis.drafts import build_draft, numeric_cargo_tuples
from document_ocr.synthesis.generation_models import RenderedDraftReceipt
from document_ocr.synthesis.policies import validate_policy_registry
from document_ocr.synthesis.preparation import _resolve_file
from document_ocr.synthesis.rendering import apply_patch_plan, build_patch_plan
from document_ocr.synthesis.selection import (
    SelectionCandidate,
    SelectionRequest,
    solve_selection,
)
from document_ocr.training.tasks import (
    RelationExplicitTaskConstraints,
    TrainingTask,
    get_training_task,
)


def _read_pinned_jsonl(
    project_root: Path,
    *,
    path_value: str,
    expected_sha256: str,
    expected_records: int,
    label: str,
) -> tuple[Path, list[dict[str, Any]]]:
    path = _resolve_file(project_root, path_value, label)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual}")
    rows = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(value)
    if len(rows) != expected_records:
        raise ValueError(
            f"{label} row count mismatch: expected {expected_records}, found {len(rows)}"
        )
    return path, rows


def _read_pinned_json(
    project_root: Path, *, path_value: str, expected_sha256: str, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_file(project_root, path_value, label)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual}")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return path, value


def _validate_preparation_directory(
    project_root: Path, *, path_value: str, manifest_sha256: str
) -> Path:
    unresolved = Path(path_value)
    root = unresolved if unresolved.is_absolute() else project_root / unresolved
    if root.is_symlink():
        raise ValueError("preparation directory must not be a symbolic link")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("preparation path is not a directory")
    manifest = resolved / "manifest.json"
    if sha256_file(manifest) != manifest_sha256:
        raise ValueError("preparation manifest SHA-256 mismatch")
    return resolved


def _task(config: SynthesisDeterministicSmokeConfig, project_root: Path) -> TrainingTask:
    path = _resolve_file(project_root, config.task_constraints.path, "task constraints")
    if sha256_file(path) != config.task_constraints.sha256:
        raise ValueError("task-constraints SHA-256 mismatch")
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        path.read_bytes(), strict=True
    )
    return get_training_task(config.task).bind_constraints(constraints)


def _validate_target(target: dict[str, Any], document_id: str, task: Any) -> None:
    if task.canonicalize(target) != target:
        raise ValueError("draft target is not canonical")
    tables = ADAPTER.project(document_id=document_id, source_row_index=0, target=target)
    reconstructed = ADAPTER.reconstruct(
        document_id=document_id,
        tables=rows_for_document(tables.rows, document_id),
    )
    if reconstructed != target:
        raise ValueError("draft target fails relational inverse projection")


def _page_bucket(page_count: int) -> str:
    if page_count == 1:
        return "one"
    if page_count == 2:
        return "two"
    return "three_or_more"


def _contexts(feature: Mapping[str, Any]) -> frozenset[str]:
    mapping = {
        "multi_container": "multi_container",
        "multi_cargo": "multi_goods",
        "multiple_package_levels": "multi_package_level",
        "container_allocations": "has_allocations",
        "refrigerated": "temperature_present",
        "dangerous_goods": "dangerous_goods_present",
        "delivery_agent": "delivery_agent_present",
        "forwarding_agent": "forwarding_agent_present",
    }
    return frozenset(name for name, field in mapping.items() if bool(feature.get(field)))


def _risk(feature: Mapping[str, Any]) -> int:
    similarity = feature.get("template_proxy_combined_similarity")
    if not isinstance(similarity, (int, float)) or isinstance(similarity, bool):
        raise ValueError("document feature has no numeric template similarity")
    if not 0 <= similarity <= 1:
        raise ValueError("template similarity is outside [0, 1]")
    return round((1 - float(similarity)) * 10_000)


def _report(
    *,
    run_id: str,
    selected: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    elapsed: float,
    peak_mib: float,
) -> str:
    family_counts = Counter(cast(str, row["assigned_family"]) for row in selected)
    family_support = Counter(
        family
        for row in candidate_rows
        for family in cast(Sequence[str], row["eligible_families"])
    )
    pending_counts: Counter[str] = Counter()
    for row in selected:
        pending_counts.update(cast(Mapping[str, int], row["pending_realization_kinds"]))
    eligible_candidates = sum(bool(row["eligible"]) for row in candidate_rows)
    total_changes = sum(cast(int, row["semantic_change_count"]) for row in selected)
    total_edits = sum(cast(int, row["text_edit_count"]) for row in selected)
    lines = [
        f"# Deterministic synthesis smoke: `{run_id}`",
        "",
        "This run exercises only deterministic, non-agentic components. Every output remains ",
        "`training_eligible=false` because registry-backed and linguistic realization is pending.",
        "",
        "## Result",
        "",
        f"- Selected and rendered drafts: **{len(selected)}**",
        f"- Candidate documents inspected: **{len(candidate_rows)}**",
        f"- Base-eligible candidates with at least one rendered family: **{eligible_candidates}**",
        f"- Deterministic semantic changes: **{total_changes}**",
        f"- Exact OCR text edits: **{total_edits}**",
        f"- Runtime: **{elapsed:.3f} s**",
        f"- Throughput: **{len(candidate_rows) / elapsed:.1f} source documents/s**",
        f"- Process peak RSS: **{peak_mib:.1f} MiB**",
        "- PydanticAI/API/model calls: **0**",
        "- Training records published: **0**",
        "",
        "## Selected mutation families",
        "",
        "| Family | Drafts |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(family_counts.items()))
    lines.extend(
        [
            "",
            "## Real-corpus renderer support",
            "",
            "| Family | Fully eligible source documents |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(family_support.items()))
    lines.extend(
        [
            "",
            "## Pending realization in the selected drafts",
            "",
            "| Pending kind | Concrete leaves |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{name}` | {count} |" for name, count in sorted(pending_counts.items()))
    lines.extend(
        [
            "",
            "## Selected drafts",
            "",
            "| # | Base document | Family | Changes | OCR edits | Pending leaves |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        "| {position} | `{document_id}` | `{assigned_family}` | {semantic_change_count} | "
        "{text_edit_count} | {pending_realization_count} |".format(**row)
        for row in selected
    )
    lines.extend(
        [
            "",
            "## Publication boundary",
            "",
            "The rendered OCR and deterministic targets are inspection drafts, not synthetic ",
            "training examples. The scenario plan lists every remaining registry or linguistic ",
            "realization. A later whole-document agent stage must finish and validate those tasks ",
            "before any record can cross the training boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def _publish(path: Path, payload: bytes, files: dict[str, dict[str, Any]], run_root: Path) -> None:
    atomic_publish_bytes(path, payload)
    relative = str(path.relative_to(run_root))
    files[relative] = {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _implementation_sha256() -> str:
    module_root = Path(__file__).resolve().parent
    filenames = (
        "config.py",
        "drafts.py",
        "generation.py",
        "generation_models.py",
        "generators.py",
        "policies.py",
        "rendering.py",
        "selection.py",
    )
    payload = [
        {"path": f"document_ocr/synthesis/{name}", "sha256": sha256_file(module_root / name)}
        for name in filenames
    ]
    return sha256_bytes(canonical_json_bytes(payload))


def _existing_run(
    *,
    run_root: Path,
    config_payload: bytes,
    implementation_sha256: str,
) -> dict[str, Any] | None:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(read_regular_file_bytes(manifest_path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AtomicConflictError("existing synthesis manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise AtomicConflictError("existing synthesis manifest root is not an object")
    if manifest.get("implementationSha256") != implementation_sha256:
        raise AtomicConflictError(
            "existing synthesis run was produced by a different implementation; use a new run ID"
        )
    if read_regular_file_bytes(run_root / "config.yaml") != config_payload:
        raise AtomicConflictError("existing synthesis run uses different configuration bytes")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise AtomicConflictError("existing synthesis manifest has no file inventory")
    for relative, receipt in files.items():
        if not isinstance(relative, str) or not isinstance(receipt, dict):
            raise AtomicConflictError("existing synthesis file receipt is malformed")
        payload = read_regular_file_bytes(run_root / relative)
        if len(payload) != receipt.get("bytes") or sha256_bytes(payload) != receipt.get("sha256"):
            raise AtomicConflictError(f"existing synthesis artifact fails its receipt: {relative}")
    return manifest


def run_deterministic_smoke(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisDeterministicSmokeConfig,
) -> dict[str, Any]:
    """Plan, generate, render, and validate non-publishable deterministic smoke drafts."""

    started = time.perf_counter()
    run_root = Path(config.run.output_dir)
    if not run_root.is_absolute():
        run_root = project_root / run_root
    run_root = run_root / config.run.run_id
    config_payload = config_path.read_bytes()
    implementation_sha256 = _implementation_sha256()
    existing = _existing_run(
        run_root=run_root,
        config_payload=config_payload,
        implementation_sha256=implementation_sha256,
    )
    if existing is not None:
        return existing
    preparation_root = _validate_preparation_directory(
        project_root,
        path_value=config.inputs.preparation.path,
        manifest_sha256=config.inputs.preparation.manifest_sha256,
    )
    source_path, source_rows = _read_pinned_jsonl(
        project_root,
        path_value=config.source.file.path,
        expected_sha256=config.source.file.sha256,
        expected_records=config.source.file.records,
        label="synthesis source",
    )
    _anchor_path, anchor_rows = _read_pinned_jsonl(
        project_root,
        path_value=config.inputs.anchors.path,
        expected_sha256=config.inputs.anchors.sha256,
        expected_records=config.inputs.anchors.records,
        label="OCR anchors",
    )
    _feature_path, feature_rows = _read_pinned_jsonl(
        project_root,
        path_value=config.inputs.document_features.path,
        expected_sha256=config.inputs.document_features.sha256,
        expected_records=config.inputs.document_features.records,
        label="document features",
    )
    _template_path, template_rows = _read_pinned_jsonl(
        project_root,
        path_value=config.inputs.template_groups.path,
        expected_sha256=config.inputs.template_groups.sha256,
        expected_records=config.inputs.template_groups.records,
        label="template groups",
    )
    partition_path, partition = _read_pinned_json(
        project_root,
        path_value=config.inputs.partition_report.path,
        expected_sha256=config.inputs.partition_report.sha256,
        label="partition report",
    )
    task = _task(config, project_root)
    policy_audit = validate_policy_registry(
        tuple(cast(Mapping[str, Any], row["target"]) for row in source_rows)
    )

    fields = config.source.fields
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        document_id = row.get(fields.document_id)
        raw_text = row.get(fields.input_text)
        expected_raw_hash = row.get(fields.input_sha256 or "")
        if not isinstance(document_id, str) or not isinstance(raw_text, str):
            raise ValueError("source row lacks the configured document or OCR text field")
        if (
            not isinstance(expected_raw_hash, str)
            or sha256_bytes(raw_text.encode()) != expected_raw_hash
        ):
            raise ValueError(f"source OCR SHA-256 mismatch: {document_id}")
        if document_id in source_by_id:
            raise ValueError(f"duplicate source document ID: {document_id}")
        source_by_id[document_id] = row
    features = {cast(str, row["document_id"]): row for row in feature_rows}
    if set(features) != set(source_by_id):
        raise ValueError("document-feature IDs do not exactly cover the source corpus")
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchor_rows:
        anchors[cast(str, anchor["document_id"])].append(anchor)
    if set(anchors) - set(source_by_id):
        raise ValueError("OCR anchors contain unknown document IDs")
    templates = {cast(str, row["template_id"]): row for row in template_rows}
    member_template: dict[str, str] = {}
    for template_id, row in templates.items():
        for document_id in cast(Sequence[str], row["member_document_ids"]):
            if document_id in member_template:
                raise ValueError(f"document belongs to multiple template proxies: {document_id}")
            member_template[document_id] = template_id
    if set(member_template) != set(source_by_id):
        raise ValueError("template groups do not exactly cover the source corpus")
    try:
        split_ids = frozenset(
            cast(
                Sequence[str],
                partition["inspection"]["partition"]["outputs"][config.selection.split][
                    "document_ids"
                ],
            )
        )
    except (KeyError, TypeError) as error:
        raise ValueError("partition report lacks the configured split document IDs") from error
    if not split_ids <= set(source_by_id):
        raise ValueError("partition split contains unknown document IDs")

    donors = numeric_cargo_tuples(source_rows)
    preflight: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    candidates: list[SelectionCandidate] = []
    families = tuple(sorted(config.selection.family_exact))
    for row_index, row in enumerate(source_rows):
        document_id = cast(str, row[fields.document_id])
        feature = features[document_id]
        template_id = member_template[document_id]
        template = templates[template_id]
        members = frozenset(cast(Sequence[str], template["member_document_ids"]))
        template_size = len(members)
        base_reasons = []
        if document_id not in split_ids:
            base_reasons.append("outside_configured_split")
        if template_size < config.selection.minimum_template_documents:
            base_reasons.append("template_below_minimum_documents")
        if config.selection.require_template_wholly_in_split and not members <= split_ids:
            base_reasons.append("template_not_wholly_in_split")
        eligible_families = []
        family_failures: dict[str, str] = {}
        if not base_reasons:
            for family in families:
                try:
                    target, plan = build_draft(
                        row=row,
                        template_id=template_id,
                        family=family,
                        seed=config.generation.seed,
                        variant_index=config.generation.variant_index,
                        date_minimum=config.generation.date_minimum,
                        date_maximum=config.generation.date_maximum,
                        donors=donors,
                    )
                    _validate_target(target, plan.synthetic_document_id, task)
                    patch_plan = build_patch_plan(
                        synthetic_document_id=plan.synthetic_document_id,
                        source_raw_text_sha256=plan.source_raw_text_sha256,
                        deterministic_target_sha256=plan.deterministic_target_sha256,
                        changes=plan.changes,
                        anchors=anchors.get(document_id, []),
                    )
                    if patch_plan.unrendered_change_paths:
                        raise ValueError(
                            "unrendered deterministic paths: "
                            + ", ".join(patch_plan.unrendered_change_paths)
                        )
                    rendered, changed_characters = apply_patch_plan(
                        cast(str, row[fields.input_text]), patch_plan
                    )
                    if rendered == row[fields.input_text] or changed_characters <= 0:
                        raise ValueError("renderer produced no source-text change")
                    for change in plan.changes:
                        if change.family in {
                            "container_identifier",
                            "seal_identifier",
                            "allocation_reference",
                        } and str(change.old_value) in rendered:
                            raise ValueError("old formal identifier remains in rendered OCR")
                    preflight[(document_id, family)] = {
                        "target": target,
                        "plan": plan,
                        "patch_plan": patch_plan,
                        "rendered": rendered,
                        "changed_characters": changed_characters,
                    }
                    eligible_families.append(family)
                except (KeyError, TypeError, ValueError) as error:
                    family_failures[family] = str(error)
        strata = {
            "document_type": cast(str, feature["document_type"]),
            "source_corpus": cast(str, feature["source_corpus"]),
            "page_bucket": _page_bucket(cast(int, feature["page_count"])),
        }
        contexts = _contexts(feature)
        risk = _risk(feature)
        carrier_family = cast(str, feature["carrier_family"])
        inventory_row = {
            "document_id": document_id,
            "source_row_index": row_index,
            "split": config.selection.split if document_id in split_ids else "outside",
            "template_id": template_id,
            "template_size": template_size,
            "template_wholly_in_split": members <= split_ids,
            "carrier_family": carrier_family,
            "strata": strata,
            "contexts": sorted(contexts),
            "eligible_families": sorted(eligible_families),
            "family_failures": dict(sorted(family_failures.items())),
            "base_exclusion_reasons": base_reasons,
            "risk_score": risk,
            "eligible": not base_reasons and bool(eligible_families),
        }
        candidate_rows.append(inventory_row)
        if inventory_row["eligible"]:
            candidates.append(
                SelectionCandidate(
                    document_id=document_id,
                    source_row_index=row_index,
                    template_id=template_id,
                    template_size=template_size,
                    carrier_family=carrier_family,
                    strata=strata,
                    contexts=contexts,
                    eligible_families=frozenset(eligible_families),
                    risk_score=risk,
                )
            )

    request = SelectionRequest(
        requested_documents=config.selection.requested_documents,
        family_exact={str(name): count for name, count in config.selection.family_exact.items()},
        strata_exact={
            str(name): {str(value): count for value, count in values.items()}
            for name, values in config.selection.strata_exact.items()
        },
        context_minimums={
            str(name): count for name, count in config.selection.context_minimums.items()
        },
        maximum_per_template=config.selection.maximum_per_template,
        maximum_per_carrier=config.selection.maximum_per_carrier,
        minimum_carriers=config.selection.minimum_carriers,
        seed=config.selection.seed,
    )
    result = solve_selection(candidates, request)
    candidate_by_id = {row.document_id: row for row in candidates}
    selected_rows = []
    draft_outputs = []
    for assignment in result.assignments:
        candidate = candidate_by_id[assignment.document_id]
        output = preflight[(assignment.document_id, assignment.family)]
        plan = output["plan"]
        patch_plan = output["patch_plan"]
        rendered = cast(str, output["rendered"])
        target = cast(dict[str, Any], output["target"])
        receipt = RenderedDraftReceipt.model_validate(
            {
                "schema_version": 1,
                "synthetic_document_id": plan.synthetic_document_id,
                "rendered_raw_text_sha256": sha256_bytes(rendered.encode()),
                "changed_source_characters": output["changed_characters"],
                "unchanged_source_characters": len(
                    cast(str, source_by_id[assignment.document_id][fields.input_text])
                )
                - output["changed_characters"],
                "edit_count": len(patch_plan.edits),
                "page_count": len(page_texts(rendered)),
                "exact_unchanged_regions": True,
                "target_schema_valid": True,
                "relational_inverse_valid": True,
                "all_deterministic_changes_rendered": True,
                "training_eligible": False,
            },
            strict=True,
        )
        selected_rows.append(
            {
                "position": assignment.position,
                "document_id": assignment.document_id,
                "synthetic_document_id": plan.synthetic_document_id,
                "assigned_family": assignment.family,
                "eligible_families": sorted(candidate.eligible_families),
                "template_id": candidate.template_id,
                "carrier_family": candidate.carrier_family,
                "strata": dict(candidate.strata),
                "contexts": sorted(candidate.contexts),
                "source_raw_text_sha256": plan.source_raw_text_sha256,
                "deterministic_target_sha256": plan.deterministic_target_sha256,
                "random_stream_namespace": "mpci-bl-synthesis-v1",
                "semantic_change_count": len(plan.changes),
                "changed_target_paths": [change.target_path for change in plan.changes],
                "text_edit_count": len(patch_plan.edits),
                "changed_source_characters": output["changed_characters"],
                "pending_realization_count": len(plan.pending_realizations),
                "pending_realization_kinds": dict(
                    sorted(Counter(row.kind for row in plan.pending_realizations).items())
                ),
                "training_eligible": False,
            }
        )
        draft_outputs.append((plan, patch_plan, receipt, target, rendered))

    elapsed = time.perf_counter() - started
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    files: dict[str, dict[str, Any]] = {}
    _publish(
        run_root / "config.yaml",
        config_payload,
        files,
        run_root,
    )
    _publish(run_root / "candidate-inventory.jsonl", _jsonl(candidate_rows), files, run_root)
    _publish(run_root / "selection-plan.jsonl", _jsonl(selected_rows), files, run_root)
    _publish(
        run_root / "feasibility.json",
        json_artifact_bytes(result.feasibility),
        files,
        run_root,
    )
    solver = {
        **result.solver_receipt,
        "planning_runtime_seconds": elapsed,
        "peak_rss_mib": peak_mib,
    }
    _publish(run_root / "solver-receipt.json", json_artifact_bytes(solver), files, run_root)
    _publish(
        run_root / "field-policy-audit.json",
        json_artifact_bytes(policy_audit),
        files,
        run_root,
    )
    for plan, patch_plan, receipt, target, rendered in draft_outputs:
        synthetic_id = plan.synthetic_document_id
        _publish(
            run_root / "drafts/scenario-plans" / f"{synthetic_id}.json",
            json_artifact_bytes(plan.model_dump(mode="json")),
            files,
            run_root,
        )
        _publish(
            run_root / "drafts/patch-plans" / f"{synthetic_id}.json",
            json_artifact_bytes(patch_plan.model_dump(mode="json")),
            files,
            run_root,
        )
        _publish(
            run_root / "drafts/receipts" / f"{synthetic_id}.json",
            json_artifact_bytes(receipt.model_dump(mode="json")),
            files,
            run_root,
        )
        _publish(
            run_root / "drafts/deterministic-targets" / f"{synthetic_id}.json",
            json_artifact_bytes(target),
            files,
            run_root,
        )
        _publish(
            run_root / "drafts/rendered-raw" / f"{synthetic_id}.txt",
            rendered.encode(),
            files,
            run_root,
        )
    report = _report(
        run_id=config.run.run_id,
        selected=selected_rows,
        candidate_rows=candidate_rows,
        elapsed=elapsed,
        peak_mib=peak_mib,
    )
    _publish(run_root / "REPORT.md", report.encode(), files, run_root)
    manifest = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "task": config.task,
        "implementationSha256": implementation_sha256,
        "status": "deterministic_smoke_complete_pending_realization",
        "trainingEligible": False,
        "modelOrApiCalls": 0,
        "source": {
            "path": str(source_path.relative_to(project_root)),
            "sha256": config.source.file.sha256,
            "records": len(source_rows),
        },
        "preparation": {
            "path": str(preparation_root.relative_to(project_root)),
            "manifestSha256": config.inputs.preparation.manifest_sha256,
        },
        "partitionReport": {
            "path": str(partition_path.relative_to(project_root)),
            "sha256": config.inputs.partition_report.sha256,
            "split": config.selection.split,
        },
        "selection": result.feasibility,
        "drafts": {
            "selected": len(selected_rows),
            "schemaValid": len(selected_rows),
            "relationalInverseValid": len(selected_rows),
            "fullyDeterministicallyRendered": len(selected_rows),
            "changeLedgerExact": len(selected_rows),
            "trainingRecordsPublished": 0,
            "semanticChanges": sum(
                cast(int, row["semantic_change_count"]) for row in selected_rows
            ),
            "exactTextEdits": sum(cast(int, row["text_edit_count"]) for row in selected_rows),
            "changedSourceCharacters": sum(
                cast(int, row["changed_source_characters"]) for row in selected_rows
            ),
            "pendingRealizationLeaves": sum(
                cast(int, row["pending_realization_count"]) for row in selected_rows
            ),
        },
        "benchmark": {
            "elapsedSeconds": elapsed,
            "sourceDocumentsPerSecond": len(candidate_rows) / elapsed,
            "selectedDraftsPerSecond": len(selected_rows) / elapsed,
            "peakRssMiB": peak_mib,
        },
        "files": dict(sorted(files.items())),
    }
    atomic_publish_json(run_root / "manifest.json", manifest)
    return manifest
