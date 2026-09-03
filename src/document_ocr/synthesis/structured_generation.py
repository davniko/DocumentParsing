"""Train-isolated profile-routed structured synthesis for reviewed B/L labels."""

from __future__ import annotations

import json
import math
import os
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.config import SynthesisStructuredBaselineConfig
from document_ocr.synthesis.domain import rows_for_document
from document_ocr.synthesis.fit_partition import fit_document_ids
from document_ocr.synthesis.generators import DeterministicStream, validate_container_number
from document_ocr.synthesis.modeling_views import build_modeling_views, modeling_views_sha256
from document_ocr.synthesis.package_registry import load_package_registry
from document_ocr.synthesis.preparation import _resolve_file
from document_ocr.synthesis.profile_sdv import ProfileNumericRow, ProfileRunResult
from document_ocr.synthesis.run_safety import (
    StagedArtifactRun,
    TrainOnlySourceScope,
    build_behavior_environment_fingerprint,
)
from document_ocr.synthesis.selection import SelectionCandidate, SelectionRequest, solve_selection
from document_ocr.synthesis.structured_models import (
    SourceScopeReceipt,
    StatisticalProposalReceipt,
    StructuredBaselinePlan,
)
from document_ocr.synthesis.structured_profiles import (
    GroupProfileContext,
    PreparedProfileSource,
    ProfileCohortBatchRejected,
    ProfileCohortKey,
    ProfileCohortRejection,
    ProfileCohortResult,
    ProfileContextBatchRejected,
    build_group_profile_contexts,
    generate_profile_proposals,
    prepare_profile_source,
    profile_cohort_key,
    profile_support_reasons,
)
from document_ocr.synthesis.structured_semantics import (
    apply_cargo_group_numeric_proposals,
    apply_date_proposal,
    apply_identifier_plan,
    build_identifier_request_inventory,
    finalize_non_linguistic_target,
    pending_realizations,
    reserve_structured_identifiers,
)
from document_ocr.synthesis.transport_capacity import (
    TransportCapacityReceipt,
    capacity_limits,
    document_capacity_receipt,
    numeric_fit_envelope_violations,
)
from document_ocr.training.tasks import (
    RelationExplicitTaskConstraints,
    TrainingTask,
    get_training_task,
)


class StructuredBaselineError(RuntimeError):
    """The structured baseline cannot satisfy its audited contract."""


def _read_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    label: str,
) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(f"{label} expected {expected_rows} rows, found {len(rows)}")
    return rows


def _read_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _pinned_path(project_root: Path, value: str, label: str) -> Path:
    return _resolve_file(project_root, value, label)


def _preparation_root(
    project_root: Path,
    config: SynthesisStructuredBaselineConfig,
) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(config.inputs.preparation.path)
    root = unresolved if unresolved.is_absolute() else project_root / unresolved
    if root.is_symlink() or not root.resolve(strict=True).is_dir():
        raise ValueError("preparation root must be a regular directory")
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != config.inputs.preparation.manifest_sha256:
        raise ValueError("preparation manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("preparation manifest has no file inventory")
    for relative, raw_receipt in cast(dict[str, Any], manifest["files"]).items():
        if not isinstance(raw_receipt, dict):
            raise ValueError("preparation manifest contains a malformed receipt")
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_file():
            raise ValueError(f"preparation artifact is not a regular file: {relative}")
        if path.stat().st_size != raw_receipt.get("bytes") or sha256_file(path) != raw_receipt.get(
            "sha256"
        ):
            raise ValueError(f"preparation artifact fails its receipt: {relative}")
    return root, manifest


def _load_tables(root: Path, manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    expected_counts = cast(Mapping[str, int], manifest["domainProjection"]["tableRows"])
    tables: dict[str, list[dict[str, Any]]] = {}
    for table in ADAPTER.table_order:
        path = root / "tables" / f"{table}.jsonl"
        rows: list[dict[str, Any]] = []
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{table}:{line_number}: invalid JSON") from error
                if not isinstance(row, dict):
                    raise ValueError(f"{table}:{line_number}: row must be an object")
                rows.append(row)
        if len(rows) != expected_counts[table]:
            raise ValueError(f"prepared table row count mismatch: {table}")
        tables[table] = rows
    return tables


def _task(config: SynthesisStructuredBaselineConfig, project_root: Path) -> TrainingTask:
    path = _pinned_path(project_root, config.task_constraints.path, "task constraints")
    if sha256_file(path) != config.task_constraints.sha256:
        raise ValueError("task constraints SHA-256 mismatch")
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        path.read_bytes(), strict=True
    )
    return get_training_task(config.task).bind_constraints(constraints)


def _template_maps(
    rows: Sequence[Mapping[str, Any]], corpus_ids: frozenset[str]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    templates: dict[str, Mapping[str, Any]] = {}
    member_template: dict[str, str] = {}
    for row in rows:
        template_id = row.get("template_id")
        members = row.get("member_document_ids")
        if not isinstance(template_id, str) or not isinstance(members, list):
            raise ValueError("template inventory row is malformed")
        if template_id in templates:
            raise ValueError(f"duplicate template ID: {template_id}")
        templates[template_id] = row
        for document_id in members:
            if not isinstance(document_id, str) or document_id in member_template:
                raise ValueError("template membership is invalid or duplicated")
            member_template[document_id] = template_id
    if set(member_template) != corpus_ids:
        raise ValueError("template membership does not exactly cover source corpus")
    return templates, member_template


def _split_ids(partition: Mapping[str, Any], split: str) -> frozenset[str]:
    return frozenset(fit_document_ids(partition, expected_split=split))


def _page_bucket(page_count: int) -> str:
    if page_count == 1:
        return "one"
    if page_count == 2:
        return "two"
    return "three_or_more"


def _container_bucket(count: int) -> str:
    if count == 0:
        return "none"
    if count == 1:
        return "one"
    if count <= 4:
        return "two_to_four"
    if count <= 10:
        return "five_to_ten"
    return "over_ten"


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
    return frozenset(name for name, source in mapping.items() if feature.get(source) is True)


def _risk(feature: Mapping[str, Any]) -> int:
    similarity = feature.get("template_proxy_combined_similarity")
    if not isinstance(similarity, (int, float)) or isinstance(similarity, bool):
        raise ValueError("document feature has no numeric template similarity")
    if not 0 <= float(similarity) <= 1:
        raise ValueError("template similarity is outside [0, 1]")
    return round((1 - float(similarity)) * 10_000)


def _date_pool_key(feature: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[Any, ...]:
    patch = cast(Mapping[str, Any], target["documentPatch"])
    return (
        feature["document_type"],
        patch.get("issueDate") is not None,
        patch.get("shippedOnBoardDate") is not None,
    )


def _date_proposal(
    *,
    document_id: str,
    source_target: Mapping[str, Any],
    feature: Mapping[str, Any],
    date_pools: Mapping[tuple[Any, ...], Sequence[str]],
    source_targets: Mapping[str, Mapping[str, Any]],
    stream: DeterministicStream,
    minimum: date,
    maximum: date,
    maximum_jitter: int,
) -> tuple[date | None, date | None, str | None, dict[str, Any] | None]:
    patch = cast(Mapping[str, Any], source_target["documentPatch"])
    source_issue = date.fromisoformat(patch["issueDate"]) if patch.get("issueDate") else None
    source_shipped = (
        date.fromisoformat(patch["shippedOnBoardDate"]) if patch.get("shippedOnBoardDate") else None
    )
    if source_issue is None and source_shipped is None:
        return None, None, None, None
    donors = sorted(
        value
        for value in date_pools[_date_pool_key(feature, source_target)]
        if value != document_id
    )
    if not donors:
        raise ValueError("date proposal pool is empty after excluding source document")
    donor_order = sorted(
        donors,
        key=lambda value: (stream.derive(f"donor:{value}").bytes(counter=0, length=32), value),
    )
    jitter_order = sorted(
        (*range(-maximum_jitter, 0), *range(1, maximum_jitter + 1)),
        key=lambda value: (
            stream.derive(f"jitter:{value}").bytes(counter=0, length=32),
            value,
        ),
    )
    for donor_id in donor_order:
        donor_patch = cast(Mapping[str, Any], source_targets[donor_id]["documentPatch"])
        issue = (
            date.fromisoformat(donor_patch["issueDate"]) if donor_patch.get("issueDate") else None
        )
        shipped = (
            date.fromisoformat(donor_patch["shippedOnBoardDate"])
            if donor_patch.get("shippedOnBoardDate")
            else None
        )
        for jitter in jitter_order:
            proposed_issue = issue + timedelta(days=jitter) if issue is not None else None
            proposed_shipped = shipped + timedelta(days=jitter) if shipped is not None else None
            present = [value for value in (proposed_issue, proposed_shipped) if value is not None]
            if any(value < minimum or value > maximum for value in present):
                continue
            if (proposed_issue, proposed_shipped) == (source_issue, source_shipped):
                continue
            return (
                proposed_issue,
                proposed_shipped,
                donor_id,
                {
                    "issueDate": donor_patch.get("issueDate"),
                    "shippedOnBoardDate": donor_patch.get("shippedOnBoardDate"),
                    "jitterDays": jitter,
                },
            )
    raise ValueError("bounded empirical date proposal produced no changed in-window pair")


def _source_scope_receipt(
    *,
    scope: TrainOnlySourceScope,
    template_by_document: Mapping[str, str],
    partition_sha256: str,
) -> SourceScopeReceipt:
    allowed = tuple(sorted(scope.isolated_fit_document_ids))
    templates = tuple(sorted({template_by_document[value] for value in allowed}))
    excluded = tuple(sorted(scope.corpus_document_ids - scope.isolated_fit_document_ids))
    return SourceScopeReceipt(
        split=scope.fit_split,
        allowed_document_count=len(allowed),
        allowed_document_ids_sha256=sha256_bytes(canonical_json_bytes(allowed)),
        allowed_template_ids_sha256=sha256_bytes(canonical_json_bytes(templates)),
        excluded_document_ids_sha256=sha256_bytes(canonical_json_bytes(excluded)),
        partition_report_sha256=partition_sha256,
    )


def _behavior_fingerprint(config: SynthesisStructuredBaselineConfig) -> Any:
    runtime_root = Path(__file__).resolve().parents[3]
    synthesis_root = runtime_root / "src/document_ocr/synthesis"
    behavior_files = {
        f"synthesis:{path.name}": path.relative_to(runtime_root)
        for path in sorted(synthesis_root.glob("*.py"))
    }
    for relative in (
        "src/document_ocr/atomic.py",
        "src/document_ocr/hashing.py",
        "src/document_ocr/label_schemas/bill_of_lading_v3.py",
        "src/document_ocr/training/tasks.py",
    ):
        behavior_files[relative] = Path(relative)
    actual_versions = {name: version(name) for name in config.runtime.expected_versions}
    if actual_versions != config.runtime.expected_versions:
        raise ValueError(
            "runtime package versions differ from config: "
            f"expected={config.runtime.expected_versions}, actual={actual_versions}"
        )
    actual_image = os.environ.get("SYNTHESIS_IMAGE_DIGEST")
    if actual_image != config.runtime.image_digest:
        raise ValueError(
            "running synthesis image identity differs from config: "
            f"expected={config.runtime.image_digest}, actual={actual_image!r}"
        )
    return build_behavior_environment_fingerprint(
        project_root=runtime_root,
        behavior_files=behavior_files,
        lock_files={
            "synthesis-uv-lock": Path("environments/synthesis/uv.lock"),
            "synthesis-dockerfile": Path("docker/synthesis/Dockerfile"),
        },
        image_reference=config.runtime.image_reference,
        image_digest=config.runtime.image_digest,
        environment_identity={
            name.upper() + "_VERSION": value for name, value in actual_versions.items()
        },
    )


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _stage_inventory(stage: StagedArtifactRun) -> tuple[str, ...]:
    return tuple(
        path.relative_to(stage.stage_root).as_posix()
        for path in sorted(stage.stage_root.rglob("*"))
        if path.is_file() and path.name not in {"_TRANSACTION.json", "_COMMIT.json"}
    )


def _validate_target(target: dict[str, Any], *, document_id: str, task: TrainingTask) -> None:
    if task.canonicalize(target) != target:
        raise ValueError("structured target is not task canonical")
    projected = ADAPTER.project(document_id=document_id, source_row_index=0, target=target)
    reconstructed = ADAPTER.reconstruct(
        document_id=document_id,
        tables=rows_for_document(projected.rows, document_id),
    )
    if reconstructed != target:
        raise ValueError("structured target fails exact relational inverse")


def _profile_row_payload(row: ProfileNumericRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "document_id": row.document_id,
        "template_id": row.template_id,
        "exact_identity": row.exact_identity,
        "semantic_family": row.semantic_family,
        "package_role": row.package_role,
        "profile": row.profile.key,
        "values": row.model_values(),
    }


def _selection_candidates(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    source_targets: Mapping[str, Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]],
    templates: Mapping[str, Mapping[str, Any]],
    member_template: Mapping[str, str],
    isolated_fit_ids: frozenset[str],
    contexts_by_document: Mapping[str, Sequence[GroupProfileContext]],
    prepared: Any,
    source_capacity: Mapping[str, TransportCapacityReceipt],
    config: SynthesisStructuredBaselineConfig,
) -> tuple[list[SelectionCandidate], list[dict[str, Any]], dict[str, list[str]]]:
    date_pools: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for document_id in sorted(isolated_fit_ids):
        patch = cast(Mapping[str, Any], source_targets[document_id]["documentPatch"])
        if patch.get("issueDate") is not None or patch.get("shippedOnBoardDate") is not None:
            date_pools[_date_pool_key(features[document_id], source_targets[document_id])].append(
                document_id
            )
    candidates: list[SelectionCandidate] = []
    inventory: list[dict[str, Any]] = []
    route_summaries: dict[str, list[str]] = {}
    for source_index, row in enumerate(source_rows):
        document_id = cast(str, row[config.source.fields.document_id])
        feature = features[document_id]
        template_id = member_template[document_id]
        template_members = cast(Sequence[str], templates[template_id]["member_document_ids"])
        reasons: list[str] = []
        if document_id not in isolated_fit_ids:
            reasons.append("outside_template_isolated_train_scope")
        if len(template_members) < config.selection.minimum_template_documents:
            reasons.append("template_below_minimum_documents")
        capacity_receipt = source_capacity[document_id]
        reasons.extend(
            f"source_transport_capacity_violation:{reason}"
            for reason in capacity_receipt.violations
        )

        patch = cast(Mapping[str, Any], source_targets[document_id]["documentPatch"])
        route = patch.get("route")
        parties = patch.get("parties")
        if not isinstance(route, Mapping):
            reasons.append("route_synthesis_missing_route")
        else:
            if route.get("transshipmentPort") is not None:
                reasons.append("route_synthesis_transshipment_requires_connectivity")
            if route.get("portOfLoading") is None or route.get("portOfDischarge") is None:
                reasons.append("route_synthesis_missing_physical_endpoint")
        if not isinstance(parties, Mapping) or not isinstance(parties.get("shipper"), Mapping):
            reasons.append("route_synthesis_missing_shipper")
        packages = cast(Sequence[Mapping[str, Any]], patch.get("cargoPackages") or [])
        cargo_groups = cast(Sequence[Mapping[str, Any]], patch.get("cargoGroups") or [])
        quantified_groups = {
            cast(str, package["groupId"])
            for package in packages
            if package.get("quantity") is not None
        }
        if not quantified_groups:
            reasons.append("no_quantified_cargo_group")
        context_by_group = {
            context.group_id: context for context in contexts_by_document.get(document_id, ())
        }
        routes: list[str] = []
        if document_id in isolated_fit_ids and capacity_receipt.valid:
            for group_id in sorted(quantified_groups, key=lambda value: int(value[1:])):
                context = context_by_group.get(group_id)
                if context is None:
                    reasons.append(f"missing_cargo_group_model_projection:{group_id}")
                    continue
                support = profile_support_reasons(
                    context,
                    prepared=prepared,
                    source_target=source_targets[document_id],
                    source_targets=source_targets,
                    config=config,
                )
                reasons.extend(f"{reason}:{group_id}" for reason in support)
                decision = prepared.decisions_by_request[context.request_id]
                attempts = ",".join(
                    f"{attempt.tier}={attempt.rows}/{attempt.templates}"
                    for attempt in decision.attempts
                )
                routes.append(
                    f"{group_id}:{context.request.package_role}:"
                    f"{decision.selected_tier or 'unrouted'}:{attempts}"
                )
            for group in cargo_groups:
                group_id = cast(str, group["groupId"])
                has_measure = any(
                    group.get(name) is not None for name in ("grossWeight", "netWeight", "volume")
                )
                if has_measure and group_id not in quantified_groups:
                    reasons.append(f"cargo_measure_has_no_quantified_driver:{group_id}")
            for allocation_group in cast(
                Sequence[Mapping[str, Any]], patch.get("cargoAllocationGroups") or []
            ):
                if (
                    allocation_group["coverage"] == "unlinked_package_quantities"
                    and allocation_group["groupId"] not in quantified_groups
                ):
                    reasons.append("unlinked_allocation_has_no_quantified_driver")
        if (
            patch.get("issueDate") is not None or patch.get("shippedOnBoardDate") is not None
        ) and len(date_pools[_date_pool_key(feature, source_targets[document_id])]) < 2:
            reasons.append("date_pattern_has_no_train_donor")
        if not reasons:
            try:
                build_identifier_request_inventory(
                    source_targets=source_targets,
                    selected_document_ids=(document_id,),
                )
            except ValueError as error:
                reasons.append("unsupported_identifier_surface:" + str(error))
        reasons = sorted(set(reasons))
        strata = {
            "document_type": cast(str, feature["document_type"]),
            "source_corpus": cast(str, feature["source_corpus"]),
            "page_bucket": _page_bucket(cast(int, feature["page_count"])),
            "container_bucket": _container_bucket(cast(int, feature["container_count"])),
        }
        inventory.append(
            {
                "document_id": document_id,
                "source_row_index": source_index,
                "template_id": template_id,
                "template_size": len(template_members),
                "carrier_family": feature["carrier_family"],
                "strata": strata,
                "contexts": sorted(_contexts(feature)),
                "cargo_profile_routes": routes,
                "exclusion_reasons": reasons,
                "eligible": not reasons,
            }
        )
        if reasons:
            continue
        route_summaries[document_id] = routes
        candidates.append(
            SelectionCandidate(
                document_id=document_id,
                source_row_index=source_index,
                template_id=template_id,
                template_size=len(template_members),
                carrier_family=cast(str, feature["carrier_family"]),
                strata=strata,
                contexts=_contexts(feature),
                eligible_families=frozenset({"structured"}),
                risk_score=_risk(feature),
            )
        )
    return candidates, inventory, route_summaries


def _profile_run_deterministic_payload(
    *,
    cohort_id: str,
    request_ids: Sequence[str],
    sample_seed: int,
    run_result: ProfileRunResult,
    quality_acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    run = deepcopy(run_result.to_dict())
    runtime = {
        "final_fit": run.pop("final_fit"),
        "final_sample": run.pop("final_sample"),
        "run_resources": run.pop("run_resources"),
        "benchmark_runs": [],
    }
    for row in cast(list[dict[str, Any]], run["benchmark_runs"]):
        runtime["benchmark_runs"].append(
            {
                "candidate": row["candidate"],
                "fold_index": row["fold_index"],
                "seed": row["seed"],
                "fit": row.pop("fit"),
                "sample": row.pop("sample"),
            }
        )
    return {
        "cohort_id": cohort_id,
        "request_ids": list(request_ids),
        "sample_seed": sample_seed,
        "quality_acceptance": dict(quality_acceptance),
        "run": run,
        "runtime": runtime,
    }


def _profile_deterministic_payload(result: ProfileCohortResult) -> dict[str, Any]:
    return _profile_run_deterministic_payload(
        cohort_id=result.cohort_id,
        request_ids=result.request_ids,
        sample_seed=result.sample_seed,
        run_result=result.run,
        quality_acceptance=result.quality_acceptance,
    )


def _rejected_profile_payload(result: ProfileCohortRejection) -> dict[str, Any]:
    return _profile_run_deterministic_payload(
        cohort_id=result.cohort_id,
        request_ids=result.request_ids,
        sample_seed=result.sample_seed,
        run_result=result.run,
        quality_acceptance=result.quality_acceptance,
    )


def _candidate_profile_cohort_keys(
    *,
    document_id: str,
    contexts_by_document: Mapping[str, Sequence[GroupProfileContext]],
    prepared: PreparedProfileSource,
) -> frozenset[ProfileCohortKey]:
    contexts = contexts_by_document.get(document_id)
    if not contexts:
        raise StructuredBaselineError(
            f"eligible structured candidate has no profile contexts: {document_id}"
        )
    return frozenset(profile_cohort_key(prepared, context) for context in contexts)


def _filter_rejected_profile_cohorts(
    *,
    candidates: Sequence[SelectionCandidate],
    cohort_keys_by_document: Mapping[str, frozenset[ProfileCohortKey]],
    rejected_cohort_keys: frozenset[ProfileCohortKey],
) -> tuple[list[SelectionCandidate], tuple[str, ...]]:
    """Remove candidates requiring a statistically rejected cohort, with no relaxation."""

    retained: list[SelectionCandidate] = []
    removed: list[str] = []
    for candidate in candidates:
        keys = cohort_keys_by_document.get(candidate.document_id)
        if not keys:
            raise StructuredBaselineError(
                "eligible structured candidate has no receipted profile cohort keys: "
                f"{candidate.document_id}"
            )
        if keys & rejected_cohort_keys:
            removed.append(candidate.document_id)
        else:
            retained.append(candidate)
    if not removed:
        raise StructuredBaselineError(
            "statistically rejected profile cohort did not remove any candidate"
        )
    return retained, tuple(sorted(removed))


def _publish_runtime_once(stage: StagedArtifactRun, relative: str, payload: Any) -> Any:
    path = stage.stage_root / relative
    if path.exists():
        return json.loads(path.read_bytes())
    stage.publish_json(relative, payload)
    return payload


def _numeric_summary(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "mean": sum(ordered) / len(ordered),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
    }


def _measure_values(
    proposal_rows: Sequence[Mapping[str, Any]], prefix: str
) -> tuple[list[int | float], list[int | float]]:
    source: list[int | float] = []
    generated: list[int | float] = []
    for row in proposal_rows:
        old = row[f"source_{prefix}_value"]
        new = row[f"generated_{prefix}_value"]
        if old is not None:
            source.append(cast(int | float, old))
            generated.append(cast(int | float, new))
    return source, generated


def _distribution_summary(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    changes: Sequence[Sequence[Mapping[str, Any]]],
    profile_results: Sequence[Mapping[str, Any]],
    capacity_receipts: Sequence[TransportCapacityReceipt],
) -> dict[str, Any]:
    source_driver = [cast(int, row["source_quantity"]) for row in proposal_rows]
    generated_driver = [cast(int, row["generated_quantity"]) for row in proposal_rows]
    source_all = [
        cast(int, value)
        for row in proposal_rows
        for value in cast(Mapping[str, Any], row["source_group_quantities"]).values()
    ]
    generated_all = [
        cast(int, value)
        for row in proposal_rows
        for value in cast(Mapping[str, Any], row["generated_group_quantities"]).values()
    ]
    containers = [
        container
        for target in targets
        for container in cast(Mapping[str, Any], target["documentPatch"]).get("containers", [])
    ]
    measure_summary: dict[str, Any] = {}
    for name in ("gross_weight", "net_weight", "volume"):
        source, generated = _measure_values(proposal_rows, name)
        measure_summary[name] = {
            "source": _numeric_summary(source),
            "generated": _numeric_summary(generated),
        }
    raw_total = sum(
        cast(int, row["run"]["final_proposal"]["raw_proposals"]) for row in profile_results
    )
    rejected_total = sum(
        cast(int, row["run"]["final_proposal"]["rejected_proposals"]) for row in profile_results
    )
    capacity_values = {
        "gross_payload_utilization": [
            float(row.gross_payload_utilization)
            for row in capacity_receipts
            if row.gross_payload_utilization is not None
        ],
        "net_payload_utilization": [
            float(row.net_payload_utilization)
            for row in capacity_receipts
            if row.net_payload_utilization is not None
        ],
        "volume_utilization": [
            float(row.volume_utilization)
            for row in capacity_receipts
            if row.volume_utilization is not None
        ],
    }
    return {
        "selected_documents": len(selected_rows),
        "distinct_templates": len({row["template_id"] for row in selected_rows}),
        "distinct_carriers": len({row["carrier_family"] for row in selected_rows}),
        "strata": {
            field: dict(sorted(Counter(row["strata"][field] for row in selected_rows).items()))
            for field in ("document_type", "source_corpus", "page_bucket", "container_bucket")
        },
        "contexts": dict(
            sorted(Counter(value for row in selected_rows for value in row["contexts"]).items())
        ),
        "cargo_group_proposals": len(proposal_rows),
        "profile_cohorts": len(profile_results),
        "profile_route_tiers": dict(
            sorted(Counter(cast(str, row["route_tier"]) for row in proposal_rows).items())
        ),
        "selected_models": dict(
            sorted(Counter(cast(str, row["selected_candidate"]) for row in proposal_rows).items())
        ),
        "source_driver_quantity": _numeric_summary(source_driver),
        "generated_driver_quantity": _numeric_summary(generated_driver),
        "source_all_package_quantity": _numeric_summary(source_all),
        "generated_all_package_quantity": _numeric_summary(generated_all),
        "cargo_measures": measure_summary,
        "changed_driver_quantity_fraction": sum(
            source != generated
            for source, generated in zip(source_driver, generated_driver, strict=True)
        )
        / len(proposal_rows),
        "proposal_raw_total": raw_total,
        "proposal_rejected_total": rejected_total,
        "proposal_acceptance_yield": (
            (raw_total - rejected_total) / raw_total if raw_total else 0.0
        ),
        "change_families": dict(
            sorted(
                Counter(cast(str, change["family"]) for rows in changes for change in rows).items()
            )
        ),
        "generated_container_count": len(containers),
        "valid_iso6346_container_count": sum(
            validate_container_number(cast(str, row["containerNumber"])) for row in containers
        ),
        "transport_capacity": {
            "policy": capacity_receipts[0].policy,
            "validated_documents": sum(row.valid for row in capacity_receipts),
            "containerized_documents": sum(row.container_count > 0 for row in capacity_receipts),
            "non_containerized_documents": sum(
                row.container_count == 0 for row in capacity_receipts
            ),
            "unconstrained_volume_documents": sum(
                row.container_count > 0 and row.volume_capacity_m3 is None
                for row in capacity_receipts
            ),
            "violation_count": sum(len(row.violations) for row in capacity_receipts),
            **{name: _numeric_summary(values) for name, values in capacity_values.items()},
        },
        "training_records_published": 0,
    }


def _validate_cross_artifact_joins(
    *,
    plans: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if len(plans) != len(targets):
        raise StructuredBaselineError("plan and proposed-target counts differ")
    proposals_by_base: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in proposal_rows:
        proposals_by_base[cast(str, row["base_document_id"])].append(row)
    seen_synthetic: set[str] = set()
    seen_base: set[str] = set()
    proposal_count = 0
    for plan, target_row in zip(plans, targets, strict=True):
        synthetic_id = cast(str, plan["synthetic_document_id"])
        base_id = cast(str, plan["base_document_id"])
        if synthetic_id in seen_synthetic or base_id in seen_base:
            raise StructuredBaselineError("structured cross-artifacts repeat an identity")
        seen_synthetic.add(synthetic_id)
        seen_base.add(base_id)
        if (
            target_row["syntheticDocumentId"] != synthetic_id
            or target_row["baseDocumentId"] != base_id
            or target_row["templateId"] != plan["template_id"]
            or target_row["trainingEligible"] is not False
            or plan["training_eligible"] is not False
        ):
            raise StructuredBaselineError("plan and proposed-target identity fields differ")
        target_sha = sha256_bytes(canonical_json_bytes(target_row["target"]))
        if target_row["targetSha256"] != target_sha or plan["proposed_target_sha256"] != target_sha:
            raise StructuredBaselineError("plan and proposed-target hashes differ")
        proposal_ids = {cast(str, row["proposal_id"]) for row in proposals_by_base.get(base_id, ())}
        receipt_ids = {
            cast(str, row["proposal_id"])
            for row in cast(Sequence[Mapping[str, Any]], plan["proposal_receipts"])
        }
        if not proposal_ids or proposal_ids != receipt_ids:
            raise StructuredBaselineError("proposal rows and plan receipts do not join exactly")
        changed_groups = {
            cast(str, row["coupling_group"]).removeprefix("cargo_numeric:")
            for row in cast(Sequence[Mapping[str, Any]], plan["changes"])
            if str(row["coupling_group"]).startswith("cargo_numeric:")
        }
        proposed_groups = {cast(str, row["group_id"]) for row in proposals_by_base[base_id]}
        if changed_groups != proposed_groups:
            raise StructuredBaselineError("changed and proposed cargo groups differ")
        proposal_count += len(proposal_ids)
    if set(proposals_by_base) != seen_base:
        raise StructuredBaselineError("proposal rows contain an unknown base document")
    return {
        "validated_documents": len(plans),
        "validated_group_proposals": proposal_count,
        "failures": 0,
    }


def _source_group_values(
    target: Mapping[str, Any], group_id: str
) -> tuple[dict[str, int], Mapping[str, Any]]:
    patch = cast(Mapping[str, Any], target["documentPatch"])
    quantities = {
        cast(str, package["packageId"]): cast(int, package["quantity"])
        for package in cast(Sequence[Mapping[str, Any]], patch.get("cargoPackages") or [])
        if package["groupId"] == group_id and package.get("quantity") is not None
    }
    group = next(
        row
        for row in cast(Sequence[Mapping[str, Any]], patch.get("cargoGroups") or [])
        if row["groupId"] == group_id
    )
    return quantities, group


def run_structured_baseline(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisStructuredBaselineConfig,
) -> dict[str, Any]:
    """Publish 50 non-training structured scenarios and their Pass-2 SDV evidence."""

    started = time.perf_counter()
    project_root = project_root.resolve(strict=True)
    config_bytes = config_path.read_bytes()
    fingerprint = _behavior_fingerprint(config)
    preparation_root, preparation_manifest = _preparation_root(project_root, config)
    tables = _load_tables(preparation_root, preparation_manifest)

    source_path = _pinned_path(project_root, config.source.file.path, "structured source")
    source_rows = _read_jsonl(
        source_path,
        expected_sha256=config.source.file.sha256,
        expected_rows=config.source.file.records,
        label="structured source",
    )
    feature_rows = _read_jsonl(
        _pinned_path(project_root, config.inputs.document_features.path, "features"),
        expected_sha256=config.inputs.document_features.sha256,
        expected_rows=config.inputs.document_features.records,
        label="document features",
    )
    template_rows = _read_jsonl(
        _pinned_path(project_root, config.inputs.template_groups.path, "templates"),
        expected_sha256=config.inputs.template_groups.sha256,
        expected_rows=config.inputs.template_groups.records,
        label="template groups",
    )
    category_rows = _read_jsonl(
        _pinned_path(project_root, config.inputs.category_metadata.path, "categories"),
        expected_sha256=config.inputs.category_metadata.sha256,
        expected_rows=config.inputs.category_metadata.records,
        label="package category metadata",
    )
    hierarchy_rows = _read_jsonl(
        _pinned_path(
            project_root,
            config.inputs.package_hierarchy_metadata.path,
            "package hierarchy",
        ),
        expected_sha256=config.inputs.package_hierarchy_metadata.sha256,
        expected_rows=config.inputs.package_hierarchy_metadata.records,
        label="package hierarchy metadata",
    )
    partition = _read_json(
        _pinned_path(project_root, config.inputs.partition_report.path, "partition"),
        expected_sha256=config.inputs.partition_report.sha256,
        label="partition report",
    )
    registry = load_package_registry(
        _pinned_path(project_root, config.inputs.package_registry.path, "package registry"),
        expected_sha256=config.inputs.package_registry.sha256,
        expected_entries=config.inputs.package_registry_entries,
    )

    fields = config.source.fields
    source_by_id: dict[str, dict[str, Any]] = {}
    source_targets: dict[str, Mapping[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for row in source_rows:
        document_id = row.get(fields.document_id)
        raw_text = row.get(fields.input_text)
        raw_hash = row.get(fields.input_sha256 or "")
        target = row.get(fields.target)
        if (
            not isinstance(document_id, str)
            or not isinstance(raw_text, str)
            or not isinstance(raw_hash, str)
            or sha256_bytes(raw_text.encode()) != raw_hash
            or not isinstance(target, dict)
        ):
            raise ValueError("structured source row violates configured field/hash contract")
        if document_id in source_by_id:
            raise ValueError(f"duplicate source document: {document_id}")
        source_by_id[document_id] = row
        source_targets[document_id] = target
        source_hashes[document_id] = sha256_bytes(canonical_json_bytes(row))
    corpus_ids = frozenset(source_by_id)
    features = {cast(str, row["document_id"]): row for row in feature_rows}
    if set(features) != corpus_ids:
        raise ValueError("document features do not exactly cover source corpus")
    templates, member_template = _template_maps(template_rows, corpus_ids)
    fit_ids = _split_ids(partition, config.selection.split)
    if not fit_ids <= corpus_ids:
        raise ValueError("partition split contains unknown source IDs")
    scope = TrainOnlySourceScope(
        corpus_document_ids=tuple(corpus_ids),
        fit_document_ids=tuple(fit_ids),
        template_by_document=member_template,
        source_sha256_by_document=source_hashes,
        fit_split=config.selection.split,
    )
    isolated_ids = scope.isolated_fit_document_ids
    if len(isolated_ids) != config.selection.expected_isolated_fit_documents:
        raise ValueError(
            "pinned partition/template isolation changed: "
            f"expected {config.selection.expected_isolated_fit_documents} documents, "
            f"found {len(isolated_ids)}"
        )
    transport_limits = capacity_limits(config.generation.transport_capacity)
    source_capacity = {
        document_id: document_capacity_receipt(target, transport_limits)
        for document_id, target in source_targets.items()
    }
    source_fit_capacity_violations = {
        document_id: numeric_fit_envelope_violations(target, transport_limits)
        for document_id, target in source_targets.items()
    }
    capacity_invalid_fit_ids = frozenset(
        document_id for document_id in isolated_ids if source_fit_capacity_violations[document_id]
    )
    capacity_fit_ids = isolated_ids - capacity_invalid_fit_ids
    isolated_source_order = [
        cast(str, row[fields.document_id])
        for row in source_rows
        if row[fields.document_id] in capacity_fit_ids
    ]
    views = build_modeling_views(
        tables=tables,
        train_document_ids=isolated_source_order,
        package_registry=registry,
        category_metadata_rows=category_rows,
        package_hierarchy_metadata_rows=hierarchy_rows,
    )
    contexts = build_group_profile_contexts(
        views.cargo_group_numeric,
        template_by_document=member_template,
        partition=config.selection.split,
    )
    prepared = prepare_profile_source(contexts, config=config)
    contexts_by_document: dict[str, list[GroupProfileContext]] = defaultdict(list)
    for context in contexts:
        contexts_by_document[context.document_id].append(context)

    clean_payloads: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for clean_row in prepared.audit.accepted_rows:
        clean_payloads[clean_row.document_id].append(_profile_row_payload(clean_row))
    fit_receipt = scope.create_statistical_fit_receipt(
        view_name=config.modeling.view,
        row_payload_by_document=cast(Mapping[str, Any], clean_payloads),
        purpose="quality-audited profile-routed cargo-group numeric SDV fit",
        field_paths=(
            "quantity",
            "gross_per_driver_package_kg",
            "net_per_driver_package_kg",
            "volume_per_driver_package_m3",
        ),
    )
    scope.validate_statistical_fit_receipt(
        fit_receipt,
        row_payload_by_document=cast(Mapping[str, Any], clean_payloads),
    )
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "configSha256": sha256_bytes(config_bytes),
                "fingerprintSha256": fingerprint.fingerprint_sha256,
                "sourceSha256": config.source.file.sha256,
                "preparationManifestSha256": config.inputs.preparation.manifest_sha256,
                "scopeSha256": scope.scope_sha256,
                "fitId": fit_receipt.fit_id,
            }
        )
    )
    output_parent = Path(config.run.output_dir)
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.run_id,
        transaction_sha256=transaction,
    )
    if stage.completed:
        stage.validate_committed_run()
        return cast(dict[str, Any], json.loads((stage.final_root / "manifest.json").read_bytes()))
    stage.recover_interrupted_temporary_files()
    stage.publish_bytes("config.yaml", config_bytes)
    stage.publish_json("behavior-environment-fingerprint.json", fingerprint.model_dump(mode="json"))
    source_scope_receipt = _source_scope_receipt(
        scope=scope,
        template_by_document=member_template,
        partition_sha256=config.inputs.partition_report.sha256,
    )
    stage.publish_json("source-scope.json", source_scope_receipt.model_dump(mode="json"))
    stage.publish_json("statistical-fit-provenance.json", fit_receipt.model_dump(mode="json"))
    modeling_audit_payload = {
        **views.audit.model_dump(mode="json"),
        "content_sha256": modeling_views_sha256(views),
        "fit_document_count": len(fit_receipt.inputs),
        "fit_row_count": prepared.audit.accepted_count,
        "transport_capacity_excluded_fit_document_count": len(capacity_invalid_fit_ids),
        "validation_document_count": 0,
    }
    stage.publish_json("modeling/modeling-view-audit.json", modeling_audit_payload)
    stage.publish_json("modeling/source-quality-audit.json", prepared.audit.to_dict())
    stage.publish_bytes(
        "modeling/source-transport-capacity-audit.jsonl",
        _jsonl(
            [
                {
                    "document_id": document_id,
                    "fit_split_member": document_id in isolated_ids,
                    "included_in_numeric_fit": document_id in capacity_fit_ids,
                    "numeric_fit_envelope_violations": list(
                        source_fit_capacity_violations[document_id]
                    ),
                    **source_capacity[document_id].to_dict(),
                }
                for document_id in sorted(source_capacity)
            ]
        ),
    )
    stage.publish_bytes(
        "modeling/profile-routing.jsonl",
        _jsonl(
            [prepared.decisions_by_request[context.request_id].to_dict() for context in contexts]
        ),
    )
    stage.publish_json(
        "modeling/package-surface-inventory.json",
        views.package_surface_inventory.model_dump(mode="json"),
    )
    stage.publish_bytes(
        "modeling/package-hierarchy.jsonl",
        _jsonl([row.model_dump(mode="json") for row in views.package_hierarchy]),
    )

    candidates, candidate_inventory, route_summaries = _selection_candidates(
        source_rows=source_rows,
        source_targets=source_targets,
        features=features,
        templates=templates,
        member_template=member_template,
        isolated_fit_ids=isolated_ids,
        contexts_by_document=contexts_by_document,
        prepared=prepared,
        source_capacity=source_capacity,
        config=config,
    )
    stage.publish_bytes("selection/candidate-inventory.jsonl", _jsonl(candidate_inventory))
    selection_request = SelectionRequest(
        requested_documents=config.selection.requested_documents,
        family_exact={"structured": config.selection.requested_documents},
        strata_exact=config.selection.strata_exact,
        context_minimums=config.selection.context_minimums,
        maximum_per_template=config.selection.maximum_per_template,
        maximum_per_carrier=config.selection.maximum_per_carrier,
        minimum_carriers=config.selection.minimum_carriers,
        seed=config.selection.seed,
    )
    candidate_by_id = {row.document_id: row for row in candidates}
    cohort_keys_by_document = {
        candidate.document_id: _candidate_profile_cohort_keys(
            document_id=candidate.document_id,
            contexts_by_document=contexts_by_document,
            prepared=prepared,
        )
        for candidate in candidates
    }
    active_candidates = list(candidates)
    rejected_cohort_keys: set[ProfileCohortKey] = set()
    rejected_context_document_ids: set[str] = set()
    selection_attempts: list[dict[str, Any]] = []
    while True:
        selection = solve_selection(active_candidates, selection_request)
        selected_ids = [row.document_id for row in selection.assignments]
        selected_contexts = tuple(
            context for document_id in selected_ids for context in contexts_by_document[document_id]
        )
        try:
            profile_cohorts = generate_profile_proposals(
                prepared=prepared,
                selected_contexts=selected_contexts,
                source_targets=source_targets,
                config=config,
            )
        except ProfileCohortBatchRejected as error:
            new_keys = {row.cohort_key for row in error.rejections}
            if not new_keys - rejected_cohort_keys:
                raise StructuredBaselineError(
                    "profile preflight repeated an already rejected cohort"
                ) from error
            rejection_rows: list[dict[str, Any]] = []
            for rejection in error.rejections:
                attempt_id = sha256_bytes(
                    canonical_json_bytes([rejection.cohort_id, list(rejection.request_ids)])
                )[:20]
                payload = _rejected_profile_payload(rejection)
                runtime = payload.pop("runtime")
                relative = f"modeling/rejected-cohorts/{rejection.cohort_id}-{attempt_id}"
                stage.publish_json(f"{relative}.json", payload)
                _publish_runtime_once(stage, f"{relative}.runtime.json", runtime)
                rejection_rows.append(
                    {
                        "cohort_id": rejection.cohort_id,
                        "attempt_id": attempt_id,
                        "profile": rejection.cohort_key[0],
                        "route_tier": rejection.cohort_key[1],
                        "selected_requests": len(rejection.request_ids),
                        "failures": list(rejection.quality_acceptance["failures"]),
                    }
                )
            rejected_cohort_keys.update(new_keys)
            active_candidates, removed_documents = _filter_rejected_profile_cohorts(
                candidates=candidates,
                cohort_keys_by_document=cohort_keys_by_document,
                rejected_cohort_keys=frozenset(rejected_cohort_keys),
            )
            active_candidates = [
                row
                for row in active_candidates
                if row.document_id not in rejected_context_document_ids
            ]
            removed_documents = tuple(
                sorted(set(candidate_by_id) - {row.document_id for row in active_candidates})
            )
            selection_attempts.append(
                {
                    "iteration": len(selection_attempts) + 1,
                    "status": "statistical_cohort_rejected",
                    "active_candidates_before": len(candidate_by_id)
                    if not selection_attempts
                    else selection_attempts[-1]["active_candidates_after"],
                    "active_candidates_after": len(active_candidates),
                    "selected_document_ids_sha256": sha256_bytes(
                        canonical_json_bytes(selected_ids)
                    ),
                    "rejected_cohorts": rejection_rows,
                    "cumulatively_removed_documents": list(removed_documents),
                }
            )
            if len(active_candidates) < config.selection.requested_documents:
                raise StructuredBaselineError(
                    "statistically accepted profile support cannot provide the requested "
                    "document count"
                ) from error
            continue
        except ProfileContextBatchRejected as error:
            rejected_documents = {row.document_id for row in error.rejections}
            new_documents = rejected_documents - rejected_context_document_ids
            if not new_documents:
                raise StructuredBaselineError(
                    "profile preflight repeated an already rejected package-aware context"
                ) from error
            rejection_payload = {
                "status": "package_aware_context_rejected",
                "selected_document_ids_sha256": sha256_bytes(canonical_json_bytes(selected_ids)),
                "rejections": [row.to_dict() for row in error.rejections],
            }
            attempt_id = sha256_bytes(canonical_json_bytes(rejection_payload))[:20]
            stage.publish_json(
                f"modeling/rejected-contexts/{attempt_id}.json",
                rejection_payload,
            )
            rejected_context_document_ids.update(new_documents)
            cohort_filtered, _removed = _filter_rejected_profile_cohorts(
                candidates=candidates,
                cohort_keys_by_document=cohort_keys_by_document,
                rejected_cohort_keys=frozenset(rejected_cohort_keys),
            )
            active_before = len(active_candidates)
            active_candidates = [
                row
                for row in cohort_filtered
                if row.document_id not in rejected_context_document_ids
            ]
            removed_documents = tuple(
                sorted(set(candidate_by_id) - {row.document_id for row in active_candidates})
            )
            selection_attempts.append(
                {
                    "iteration": len(selection_attempts) + 1,
                    "status": "package_aware_context_rejected",
                    "active_candidates_before": active_before,
                    "active_candidates_after": len(active_candidates),
                    "selected_document_ids_sha256": sha256_bytes(
                        canonical_json_bytes(selected_ids)
                    ),
                    "rejected_contexts": [row.to_dict() for row in error.rejections],
                    "cumulatively_removed_documents": list(removed_documents),
                }
            )
            if len(active_candidates) < config.selection.requested_documents:
                raise StructuredBaselineError(
                    "package-aware profile support cannot provide the requested document count"
                ) from error
            continue
        selection_attempts.append(
            {
                "iteration": len(selection_attempts) + 1,
                "status": "accepted",
                "active_candidates_before": len(active_candidates),
                "active_candidates_after": len(active_candidates),
                "selected_document_ids_sha256": sha256_bytes(canonical_json_bytes(selected_ids)),
                "rejected_cohorts": [],
                "cumulatively_removed_documents": sorted(
                    set(candidate_by_id) - {row.document_id for row in active_candidates}
                ),
            }
        )
        break

    selected_rows = [
        {
            "position": assignment.position,
            "document_id": assignment.document_id,
            "template_id": candidate_by_id[assignment.document_id].template_id,
            "carrier_family": candidate_by_id[assignment.document_id].carrier_family,
            "strata": dict(candidate_by_id[assignment.document_id].strata),
            "contexts": sorted(candidate_by_id[assignment.document_id].contexts),
            "cargo_profile_routes": route_summaries[assignment.document_id],
        }
        for assignment in selection.assignments
    ]
    stage.publish_bytes("selection/selected-sources.jsonl", _jsonl(selected_rows))
    stage.publish_json("selection/feasibility.json", selection.feasibility)
    stage.publish_json("selection/solver-receipt.json", selection.solver_receipt)
    stage.publish_json(
        "selection/statistical-profile-preflight.json",
        {
            "status": "accepted",
            "iterations": selection_attempts,
            "rejected_cohort_count": len(rejected_cohort_keys),
            "original_candidate_count": len(candidates),
            "final_candidate_count": len(active_candidates),
            "selected_documents": len(selected_ids),
            "policy": (
                "explicitly_reject_failed_statistical_cohorts_then_resolve_without_"
                "relaxing_selection_constraints_v1"
            ),
        },
    )
    profile_payloads: list[dict[str, Any]] = []
    profile_reporting_payloads: list[dict[str, Any]] = []
    for result in profile_cohorts:
        payload = _profile_deterministic_payload(result)
        runtime = payload.pop("runtime")
        profile_payloads.append(payload)
        stage.publish_json(f"modeling/cohorts/{result.cohort_id}.json", payload)
        persisted_runtime = _publish_runtime_once(
            stage,
            f"modeling/cohorts/{result.cohort_id}.runtime.json",
            runtime,
        )
        profile_reporting_payloads.append({**payload, "runtime": persisted_runtime})
    assignment_by_request = {
        assignment.request_id: (cohort, assignment)
        for cohort in profile_cohorts
        for assignment in cohort.assignments
    }
    if set(assignment_by_request) != {row.request_id for row in selected_contexts}:
        raise StructuredBaselineError("profile assignments do not cover selected cargo groups")

    identifier_inventory = build_identifier_request_inventory(
        source_targets=source_targets,
        selected_document_ids=selected_ids,
    )
    identifier_plan = reserve_structured_identifiers(
        all_source_targets=source_targets,
        inventory=identifier_inventory,
        seed=config.generation.seed,
    )
    if identifier_plan.containers is not None:
        stage.publish_json(
            "identifiers/container-reservations.json",
            identifier_plan.containers.model_dump(mode="json"),
        )
    if identifier_plan.generic is not None:
        stage.publish_json(
            "identifiers/generic-reservations.json",
            identifier_plan.generic.model_dump(mode="json"),
        )

    date_pools: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for document_id in sorted(isolated_ids):
        patch = cast(Mapping[str, Any], source_targets[document_id]["documentPatch"])
        if patch.get("issueDate") is not None or patch.get("shippedOnBoardDate") is not None:
            date_pools[_date_pool_key(features[document_id], source_targets[document_id])].append(
                document_id
            )
    all_identifier_allocations = identifier_plan.values()
    task = _task(config, project_root)
    plans: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    date_receipts: list[dict[str, Any]] = []
    change_rows: list[list[dict[str, Any]]] = []
    generated_capacity_receipts: list[TransportCapacityReceipt] = []
    donor_audit_inputs = []
    for position, document_id in enumerate(selected_ids):
        source_target = source_targets[document_id]
        target = deepcopy(dict(source_target))
        changes = list(
            apply_identifier_plan(
                document_id=document_id,
                target=target,
                allocations=all_identifier_allocations,
            )
        )
        stream = DeterministicStream(
            config.generation.seed,
            "mpci-bl-structured-baseline-v2",
            f"{document_id}:0",
        )
        issue, shipped, donor_id, donor_payload = _date_proposal(
            document_id=document_id,
            source_target=source_target,
            feature=features[document_id],
            date_pools=date_pools,
            source_targets=source_targets,
            stream=stream.derive("dates"),
            minimum=config.generation.date_minimum,
            maximum=config.generation.date_maximum,
            maximum_jitter=config.generation.maximum_date_jitter_days,
        )
        if donor_id is not None and donor_payload is not None:
            changes.extend(
                apply_date_proposal(
                    target=target,
                    issue_date=issue,
                    shipped_on_board_date=shipped,
                )
            )
            donor_values = {
                donor_id: {
                    "issueDate": donor_payload["issueDate"],
                    "shippedOnBoardDate": donor_payload["shippedOnBoardDate"],
                }
            }
            donor_receipt = scope.create_donor_receipt(
                source_document_id=document_id,
                donor_payload_by_document=donor_values,
                purpose="empirical date anchor plus bounded jitter",
                field_paths=("documentPatch.issueDate", "documentPatch.shippedOnBoardDate"),
            )
            donor_audit_inputs.append((donor_receipt, donor_values))
            date_receipts.append(
                {
                    "baseDocumentId": document_id,
                    "donorReceipt": donor_receipt.model_dump(mode="json"),
                    "jitterDays": donor_payload["jitterDays"],
                    "proposedIssueDate": issue.isoformat() if issue is not None else None,
                    "proposedShippedOnBoardDate": (
                        shipped.isoformat() if shipped is not None else None
                    ),
                }
            )

        group_proposals = []
        proposal_receipts = []
        for context in sorted(contexts_by_document[document_id], key=lambda row: row.group_id):
            cohort, assignment = assignment_by_request[context.request_id]
            deterministic_cohort = next(
                row for row in profile_payloads if row["cohort_id"] == cohort.cohort_id
            )
            model_receipt_sha = sha256_bytes(canonical_json_bytes(deterministic_cohort))
            selected_row_ids = prepared.decisions_by_request[context.request_id].selected_row_ids
            fit_rows = [
                _profile_row_payload(prepared.clean_rows_by_id[row_id])
                for row_id in selected_row_ids
            ]
            fit_documents = tuple(
                sorted(
                    {prepared.clean_rows_by_id[row_id].document_id for row_id in selected_row_ids}
                )
            )
            receipt_body = {
                "cohort_id": cohort.cohort_id,
                "request_id": context.request_id,
                "sampled_row_index": assignment.sampled_row_index,
                "source_numeric_equivalence_id": (assignment.source_numeric_equivalence_id),
                "contextual_support_tier": assignment.contextual_support_tier,
                "contextual_support_sha256": assignment.contextual_support_sha256,
                "contextual_distance": assignment.contextual_distance,
                "contextual_maximum_distance": (assignment.contextual_maximum_distance),
                "sampled_values": dict(assignment.sampled_values),
                "model_receipt_sha256": model_receipt_sha,
            }
            proposal_id = sha256_bytes(canonical_json_bytes(receipt_body))
            statistical_receipt = StatisticalProposalReceipt(
                proposal_id=proposal_id,
                view_name=config.modeling.view,
                method=cast(Any, cohort.run.selection.selected_candidate),
                model_receipt_sha256=model_receipt_sha,
                fit_document_ids_sha256=sha256_bytes(canonical_json_bytes(fit_documents)),
                fit_rows_sha256=sha256_bytes(canonical_json_bytes(fit_rows)),
                sample_seed=cohort.sample_seed,
                sampled_row_index=assignment.sampled_row_index,
                source_numeric_equivalence_id=(assignment.source_numeric_equivalence_id),
                contextual_support_tier=cast(Any, assignment.contextual_support_tier),
                contextual_support_rows=assignment.contextual_support_rows,
                contextual_support_templates=assignment.contextual_support_templates,
                contextual_support_sha256=assignment.contextual_support_sha256,
                contextual_distance=assignment.contextual_distance,
                contextual_maximum_distance=assignment.contextual_maximum_distance,
                raw_proposals_attempted=cohort.run.final_proposal.raw_proposals,
                rejected_before_acceptance=cohort.run.final_proposal.rejected_proposals,
            )
            proposal_receipts.append(statistical_receipt)
            group_proposals.append(assignment.proposal)
            source_quantities, source_group = _source_group_values(source_target, context.group_id)
            source_features = context.observation.features
            source_measures = {
                "gross_weight": source_group.get("grossWeight"),
                "net_weight": source_group.get("netWeight"),
                "volume": source_group.get("volume"),
            }
            generated_measures = {
                "gross_weight": assignment.proposal.gross_weight_value,
                "net_weight": assignment.proposal.net_weight_value,
                "volume": assignment.proposal.volume_value,
            }
            proposal_row: dict[str, Any] = {
                "proposal_id": proposal_id,
                "synthetic_position": position,
                "base_document_id": document_id,
                "request_id": context.request_id,
                "cohort_id": cohort.cohort_id,
                "group_id": context.group_id,
                "package_id": context.observation.driver_package_id,
                "package_count_in_group": len(assignment.proposal.quantity_by_package_id),
                "package_identity": context.request.exact_identity,
                "package_role": context.request.package_role,
                "route_tier": assignment.route_tier,
                "contextual_support_tier": assignment.contextual_support_tier,
                "contextual_support_rows": assignment.contextual_support_rows,
                "contextual_support_templates": (assignment.contextual_support_templates),
                "contextual_support_sha256": (assignment.contextual_support_sha256),
                "contextual_distance": assignment.contextual_distance,
                "contextual_maximum_distance": (assignment.contextual_maximum_distance),
                "selected_candidate": cohort.run.selection.selected_candidate,
                "source_quantity": source_features.driver_quantity,
                "generated_quantity": assignment.proposal.quantity_by_package_id[
                    assignment.proposal.driver_package_id
                ],
                "source_group_quantities": source_quantities,
                "generated_group_quantities": assignment.proposal.quantity_by_package_id,
                "sampled_features": dict(assignment.sampled_values),
                "raw_proposals": cohort.run.final_proposal.raw_proposals,
                "rejected_proposals": cohort.run.final_proposal.rejected_proposals,
                "acceptance_yield": cohort.run.final_proposal.acceptance_yield,
                "exact_train_row_copy": False,
            }
            for measure, source_measure in source_measures.items():
                proposal_row[f"source_{measure}_value"] = (
                    source_measure["value"] if isinstance(source_measure, Mapping) else None
                )
                proposal_row[f"generated_{measure}_value"] = generated_measures[measure]
                proposal_row[f"{measure}_unit"] = (
                    source_measure["unit"] if isinstance(source_measure, Mapping) else None
                )
            proposal_rows.append(proposal_row)
        changes.extend(
            apply_cargo_group_numeric_proposals(target=target, proposals=group_proposals)
        )
        generated_capacity = document_capacity_receipt(target, transport_limits)
        if not generated_capacity.valid:
            raise StructuredBaselineError(
                "generated target violates transport capacity: "
                f"{document_id}: {generated_capacity.violations}"
            )
        generated_capacity_receipts.append(generated_capacity)
        ordered_changes = tuple(sorted(changes, key=lambda row: row.target_path))
        target_sha = finalize_non_linguistic_target(
            source_target=source_target,
            target=target,
            changes=ordered_changes,
        )
        synthetic_id = (
            "syn_"
            + sha256_bytes(
                canonical_json_bytes(
                    [document_id, member_template[document_id], config.generation.seed, target_sha]
                )
            )[:40]
        )
        _validate_target(target, document_id=synthetic_id, task=task)
        pending = pending_realizations(
            target,
            changed_paths=tuple(row.target_path for row in ordered_changes),
        )
        plan = StructuredBaselinePlan(
            schema_version=1,
            status="structured_baseline_pending_realization",
            synthetic_document_id=synthetic_id,
            base_document_id=document_id,
            template_id=member_template[document_id],
            source_scope=source_scope_receipt,
            variant_index=0,
            seed=config.generation.seed,
            source_target_sha256=sha256_bytes(canonical_json_bytes(source_target)),
            proposed_target_sha256=target_sha,
            changes=ordered_changes,
            proposal_receipts=tuple(proposal_receipts),
            pending_realizations=pending,
            training_eligible=False,
        )
        plans.append(plan.model_dump(mode="json"))
        targets.append(
            {
                "syntheticDocumentId": synthetic_id,
                "baseDocumentId": document_id,
                "templateId": member_template[document_id],
                "target": target,
                "targetSha256": target_sha,
                "trainingEligible": False,
            }
        )
        change_rows.append([row.model_dump(mode="json") for row in ordered_changes])

    donor_audit = scope.audit_donor_receipts(donor_audit_inputs)
    if len(plans) != config.selection.requested_documents:
        raise StructuredBaselineError("structured plan count differs from requested count")
    if len({row["synthetic_document_id"] for row in plans}) != len(plans):
        raise StructuredBaselineError("synthetic document IDs are not unique")
    join_validation = _validate_cross_artifact_joins(
        plans=plans,
        targets=targets,
        proposal_rows=proposal_rows,
    )

    target_by_base = {
        cast(str, row["baseDocumentId"]): cast(Mapping[str, Any], row["target"]) for row in targets
    }
    synthetic_id_by_base = {
        cast(str, row["baseDocumentId"]): cast(str, row["syntheticDocumentId"]) for row in targets
    }
    hierarchy_output: list[dict[str, Any]] = []
    for record in views.package_hierarchy:
        if record.source_document_id not in target_by_base:
            continue
        generated_quantity: int | None = None
        status = "pending_metadata_only" if record.disposition == "metadata_only" else "realized"
        if record.disposition == "task_facing":
            if record.projected_package_id is None:
                raise StructuredBaselineError("task-facing hierarchy row has no package ID")
            packages = cast(
                Sequence[Mapping[str, Any]],
                target_by_base[record.source_document_id]["documentPatch"].get("cargoPackages")
                or [],
            )
            generated = next(
                row for row in packages if row["packageId"] == record.projected_package_id
            )
            generated_quantity = cast(int | None, generated.get("quantity"))
        hierarchy_output.append(
            {
                **record.model_dump(mode="json"),
                "synthetic_document_id": synthetic_id_by_base[record.source_document_id],
                "generated_quantity": generated_quantity,
                "generation_policy": (
                    "profile_routed_group_driver_then_reconcile_v1"
                    if record.disposition == "task_facing"
                    else "pending_later_metadata_only_realization"
                ),
                "realization_status": status,
            }
        )

    distribution = _distribution_summary(
        selected_rows=selected_rows,
        proposal_rows=proposal_rows,
        targets=[cast(Mapping[str, Any], row["target"]) for row in targets],
        changes=change_rows,
        profile_results=profile_payloads,
        capacity_receipts=generated_capacity_receipts,
    )
    distribution["selected_package_hierarchy_facts"] = len(hierarchy_output)
    distribution["pending_metadata_only_package_facts"] = sum(
        row["realization_status"] == "pending_metadata_only" for row in hierarchy_output
    )
    distribution["source_transport_capacity"] = {
        "audited_documents": len(source_capacity),
        "invalid_documents": sum(not row.valid for row in source_capacity.values()),
        "invalid_fit_documents": len(capacity_invalid_fit_ids),
        "numeric_fit_envelope_violation_counts": dict(
            sorted(
                Counter(
                    violation
                    for rows in source_fit_capacity_violations.values()
                    for violation in rows
                ).items()
            )
        ),
        "violation_counts": dict(
            sorted(
                Counter(
                    violation for row in source_capacity.values() for violation in row.violations
                ).items()
            )
        ),
    }
    if distribution["valid_iso6346_container_count"] != distribution["generated_container_count"]:
        raise StructuredBaselineError("not every generated container passes ISO 6346 validation")

    stage.publish_json("date-donor-leakage-audit.json", donor_audit.model_dump(mode="json"))
    stage.publish_bytes("date-donor-receipts.jsonl", _jsonl(date_receipts))
    stage.publish_bytes("generation/cargo-group-proposals.jsonl", _jsonl(proposal_rows))
    generated_capacity_rows = [
        {
            "synthetic_document_id": cast(str, target["syntheticDocumentId"]),
            "base_document_id": cast(str, target["baseDocumentId"]),
            **receipt.to_dict(),
        }
        for target, receipt in zip(targets, generated_capacity_receipts, strict=True)
    ]
    stage.publish_bytes(
        "generation/transport-capacity-receipts.jsonl",
        _jsonl(generated_capacity_rows),
    )
    stage.publish_bytes("generation/selected-package-hierarchy.jsonl", _jsonl(hierarchy_output))
    stage.publish_bytes("generation/structured-plans.jsonl", _jsonl(plans))
    stage.publish_bytes("generation/proposed-targets.jsonl", _jsonl(targets))
    stage.publish_json("generation/distribution-summary.json", distribution)
    validation_summary = {
        "requested_documents": config.selection.requested_documents,
        "strict_schema_valid": len(plans),
        "task_canonical": len(plans),
        "relational_inverse_valid": len(plans),
        "exact_change_ledger": len(plans),
        "allocation_arithmetic_valid": len(plans),
        "identifier_collision_free": len(plans),
        "transport_capacity_valid": sum(row.valid for row in generated_capacity_receipts),
        "transport_capacity_violations": sum(
            len(row.violations) for row in generated_capacity_receipts
        ),
        "source_transport_capacity_documents_excluded_from_fit": len(capacity_invalid_fit_ids),
        "statistical_exact_train_row_copies": 0,
        "profile_source_rows_quarantined": prepared.audit.quarantined_count,
        "cross_artifact_joins": join_validation,
        "pending_deterministic_or_statistical_paths": 0,
        "training_records_published": 0,
    }
    stage.publish_json("generation/validation-summary.json", validation_summary)

    from document_ocr.synthesis.structured_reporting import publish_structured_baseline_report

    publish_structured_baseline_report(
        stage=stage,
        run_id=config.run.run_id,
        selected_rows=selected_rows,
        proposal_rows=proposal_rows,
        targets=targets,
        change_rows=change_rows,
        distribution=distribution,
        profile_results=profile_reporting_payloads,
        modeling_audit=modeling_audit_payload,
        capacity_rows=generated_capacity_rows,
    )
    runtime = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "pid": os.getpid(),
    }
    _publish_runtime_once(stage, "runtime.json", runtime)
    manifest = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": "structured_baseline_complete_pending_linguistic_and_text_realization",
        "trainingEligible": False,
        "source": {
            "path": str(source_path.relative_to(project_root)),
            "sha256": config.source.file.sha256,
            "records": len(source_rows),
        },
        "scope": {
            "partitionTrainDocuments": len(fit_ids),
            "templateIsolatedFitDocuments": len(isolated_ids),
            "qualityAcceptedFitDocuments": len(fit_receipt.inputs),
            "qualityAcceptedFitRows": prepared.audit.accepted_count,
            "qualityQuarantinedFitRows": prepared.audit.quarantined_count,
            "scopeSha256": scope.scope_sha256,
        },
        "selection": selection.feasibility,
        "modeling": {
            "view": config.modeling.view,
            "rows": len(views.cargo_group_numeric),
            "profileCohorts": len(profile_cohorts),
            "selectedCandidates": distribution["selected_models"],
            "fitId": fit_receipt.fit_id,
        },
        "generation": distribution,
        "modelOrApiCalls": 0,
        "trainingRecordsPublished": 0,
        "transactionSha256": transaction,
    }
    stage.publish_json("manifest.json", manifest)
    commit = stage.commit(
        expected_artifacts=_stage_inventory(stage),
        metadata=cast(
            dict[str, Any],
            {
                "status": cast(str, manifest["status"]),
                "trainingEligible": False,
                "structuredPlans": len(plans),
                "profileCohorts": len(profile_cohorts),
            },
        ),
    )
    return {
        **manifest,
        "commitCreated": commit.created,
        "commitContentSha256": commit.receipt.content_sha256,
        "runtime": runtime,
    }
