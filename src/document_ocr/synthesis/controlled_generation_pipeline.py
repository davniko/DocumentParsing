"""Compose the controlled semantic B/L generators into one audited pilot.

This runner is intentionally not a training-record publisher.  It proves the
semantic generation path before party anonymization, cargo-language
realization, and raw-OCR patching.  Every generated semantic value is present
in ``generation/controlled-scenarios.jsonl`` and in a strict-schema draft
target where the current task schema can represent it.  Unsupported work is
named per document; source values are never presented as synthetic output.
"""

from __future__ import annotations

import io
import json
import math
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import ArtifactReadError, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.cargo_origin_scenarios import (
    CargoOriginScenarioError,
    build_cargo_origin_support,
    project_cargo_origin_target,
    sample_cargo_origin_projection,
)
from document_ocr.synthesis.config import SynthesisControlledPilotConfig
from document_ocr.synthesis.country_registry import load_iso_country_registry
from document_ocr.synthesis.equipment_registry import (
    EquipmentRegistryError,
    load_bic_equipment_registry,
)
from document_ocr.synthesis.equipment_scenarios import (
    EquipmentFitObservation,
    EquipmentTypeScenario,
    build_equipment_scenario_support,
    sample_equipment_type,
)
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import (
    compile_uk_global_tariff_registry,
    load_ukgt_source_pin,
)
from document_ocr.synthesis.hs_scenarios import (
    HsScenarioPolicy,
    build_hs_scenario_support,
    hs_fit_observations,
    sample_hs_scenario,
)
from document_ocr.synthesis.modeling_views import ModelingViews, build_modeling_views
from document_ocr.synthesis.package_registry import load_package_registry
from document_ocr.synthesis.package_scenarios import (
    PackageFitObservation,
    PackageTypeScenario,
    build_package_scenario_support,
    package_fit_observations,
    sample_package_type,
)
from document_ocr.synthesis.reefer_scenarios import (
    ReeferEquipmentIdentity,
    ReeferTemperatureObservation,
    TemperatureScenario,
    build_reefer_temperature_support,
    sample_generated_equipment_temperature,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun, TrainOnlySourceScope
from document_ocr.synthesis.transport_identity import (
    VoyageNumberPolicy,
    build_transport_identity_bundle,
    expand_transport_identity_structure,
    realize_voyage_numbers,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


class ControlledGenerationError(RuntimeError):
    """The controlled pilot cannot satisfy its pinned generation contract."""


def _resolve_file(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    candidate = unresolved if unresolved.is_absolute() else project_root / unresolved
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


def _resolve_directory(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    candidate = unresolved if unresolved.is_absolute() else project_root / unresolved
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")
    path = candidate.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} must be a plain directory")
    return path


def _read_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(read_regular_file_bytes(path))
    except (ArtifactReadError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _verify_sha256(path: Path, *, expected_sha256: str, label: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")


def _validate_committed_dependency(
    *, root: Path, commit_sha256: str, transaction_sha256: str, label: str
) -> None:
    commit_path = root / "_COMMIT.json"
    _verify_sha256(commit_path, expected_sha256=commit_sha256, label=f"{label} commit")
    run = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=transaction_sha256,
    )
    receipt = run.validate_committed_run()
    if receipt.transaction_sha256 != transaction_sha256:
        raise ValueError(f"{label} committed transaction differs from configuration")


def _bound_task(project_root: Path, config: SynthesisControlledPilotConfig) -> Any:
    path = _resolve_file(
        project_root,
        config.task_constraints.path,
        label="task constraints",
    )
    payload = read_regular_file_bytes(path)
    _verify_sha256(
        path,
        expected_sha256=config.task_constraints.sha256,
        label="task constraints",
    )
    constraints = RelationExplicitTaskConstraints.model_validate_json(payload, strict=True)
    if payload != canonical_json_bytes(constraints.model_dump(mode="json")) + b"\n":
        raise ValueError("task constraints must be canonical JSON plus one newline")
    return get_training_task(config.task).bind_constraints(constraints)


def _read_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    expected_records: int,
    label: str,
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    output: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, payload in enumerate(stream, start=1):
            if not payload.strip():
                raise ValueError(f"{label}:{line_number}: blank rows are forbidden")
            try:
                value = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            output.append(value)
    if len(output) != expected_records:
        raise ValueError(f"{label} expected {expected_records} rows, found {len(output)}")
    return output


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _template_map(rows: Sequence[Mapping[str, Any]], corpus_ids: frozenset[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        template_id = row.get("template_id")
        members = row.get("member_document_ids")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError(f"template row {row_number} has no template identity")
        if not isinstance(members, list) or any(
            not isinstance(value, str) or not value for value in members
        ):
            raise ValueError(f"template row {row_number} has invalid members")
        for document_id in members:
            if document_id in output:
                raise ValueError(f"document occurs in multiple templates: {document_id}")
            output[document_id] = template_id
    if set(output) != corpus_ids:
        raise ValueError("template groups do not cover the source corpus exactly")
    return output


def _partition_map(report: Mapping[str, Any]) -> tuple[dict[str, str], tuple[str, ...]]:
    try:
        outputs = report["inspection"]["partition"]["outputs"]
    except (KeyError, TypeError) as error:
        raise ValueError("partition report has no inspection partition outputs") from error
    if not isinstance(outputs, Mapping):
        raise ValueError("partition outputs must be an object")
    partition: dict[str, str] = {}
    train: tuple[str, ...] = ()
    for split, raw in sorted(outputs.items()):
        if not isinstance(split, str) or not isinstance(raw, Mapping):
            raise ValueError("partition split contract is invalid")
        ids = raw.get("document_ids")
        records = raw.get("records")
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
            raise ValueError(f"partition {split!r} document IDs are invalid")
        if records != len(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"partition {split!r} count or uniqueness differs")
        for document_id in ids:
            if document_id in partition:
                raise ValueError(f"document occurs in multiple partitions: {document_id}")
            partition[document_id] = split
        if split == "train":
            train = tuple(ids)
    if not train:
        raise ValueError("partition report has no non-empty train split")
    return partition, train


def _manifest_file(
    *, root: Path, manifest: Mapping[str, Any], relative: str, label: str
) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("preparation manifest has no file inventory")
    receipt = files.get(relative)
    if not isinstance(receipt, Mapping):
        raise ValueError(f"preparation manifest omits {relative}")
    expected_sha256 = receipt.get("sha256")
    if not isinstance(expected_sha256, str):
        raise ValueError(f"preparation manifest has no SHA-256 for {relative}")
    records = _preparation_table_records(manifest, relative)
    return _read_jsonl(
        root / relative,
        expected_sha256=expected_sha256,
        expected_records=records,
        label=label,
    )


def _preparation_table_records(manifest: Mapping[str, Any], relative: str) -> int:
    table = Path(relative).stem
    try:
        records = manifest["domainProjection"]["tableRows"][table]
    except (KeyError, TypeError) as error:
        raise ValueError(f"preparation manifest has no row count for {table}") from error
    if not isinstance(records, int) or records < 0:
        raise ValueError(f"preparation row count for {table} is invalid")
    return records


def _source_inventory(
    rows: Sequence[Mapping[str, Any]], config: SynthesisControlledPilotConfig
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, str]]:
    document_field = config.source.fields.document_id
    target_field = config.source.fields.target
    input_hash_field = config.source.fields.input_sha256
    if input_hash_field is None:
        raise ValueError("controlled source requires an input SHA-256 field")
    by_id: dict[str, dict[str, Any]] = {}
    targets: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        document_id = row.get(document_field)
        target = row.get(target_field)
        source_hash = row.get(input_hash_field)
        if (
            not isinstance(document_id, str)
            or not isinstance(target, dict)
            or not isinstance(source_hash, str)
        ):
            raise ValueError(f"source row {row_number} misses controlled fields")
        if document_id in by_id:
            raise ValueError(f"duplicate source document: {document_id}")
        by_id[document_id] = dict(row)
        targets[document_id] = target
        hashes[document_id] = source_hash
    return by_id, targets, hashes


def _preparation_tables(
    *, root: Path, manifest: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    return {
        table: _manifest_file(
            root=root,
            manifest=manifest,
            relative=f"tables/{table}.jsonl",
            label=f"preparation table {table}",
        )
        for table in ADAPTER.table_order
    }


def _allowed_package_categories(rows: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    output: set[str] = set()
    for row in rows:
        decisions = row.get("packageDecisions")
        if not isinstance(decisions, list):
            raise ValueError("category metadata row has no packageDecisions list")
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("category metadata decision must be an object")
            token = decision.get("categoryToken")
            if token is not None:
                if not isinstance(token, str):
                    raise ValueError("package category token must be a string")
                output.add(token)
    if not output:
        raise ValueError("category metadata has no resolved package vocabulary")
    return frozenset(output)


def _equipment_observations(views: ModelingViews) -> tuple[EquipmentFitObservation, ...]:
    output: list[EquipmentFitObservation] = []
    for row in views.container_equipment:
        features = row.features
        output.append(
            EquipmentFitObservation(
                source_document_id=row.projection.source_document_id,
                equipment_row_id=row.projection.view_row_key,
                printed_surface=features.type_description_surface,
                exact_registry_code=None,
                temperature_setpoint_present=(features.temperature_setpoint_value is not None),
            )
        )
    return tuple(output)


def _classify_exact_equipment(
    observations: Sequence[EquipmentFitObservation], registry: Any
) -> tuple[EquipmentFitObservation, ...]:
    output: list[EquipmentFitObservation] = []
    for row in observations:
        exact: str | None = None
        if row.printed_surface is not None:
            try:
                identity = registry.classify(row.printed_surface)
            except EquipmentRegistryError:
                pass
            else:
                exact = identity.size_type_code
        output.append(
            EquipmentFitObservation(
                source_document_id=row.source_document_id,
                equipment_row_id=row.equipment_row_id,
                printed_surface=row.printed_surface,
                exact_registry_code=exact,
                temperature_setpoint_present=row.temperature_setpoint_present,
            )
        )
    return tuple(output)


def _reefer_observations(views: ModelingViews) -> tuple[ReeferTemperatureObservation, ...]:
    output: list[ReeferTemperatureObservation] = []
    for row in views.container_equipment:
        features = row.features
        if features.temperature_setpoint_value is None:
            continue
        if features.type_description_surface is None:
            continue
        if features.temperature_setpoint_unit is None:
            raise ValueError("temperature value is present without its unit")
        output.append(
            ReeferTemperatureObservation(
                source_document_id=row.projection.source_document_id,
                equipment_row_id=row.projection.view_row_key,
                identity=ReeferEquipmentIdentity(
                    value=features.type_description_surface,
                    authority="preserved_reviewed_surface",
                    is_reefer=True,
                ),
                setpoint_value=float(features.temperature_setpoint_value),
                setpoint_unit=features.temperature_setpoint_unit,
            )
        )
    return tuple(output)


def _patch_package_target(target: dict[str, Any], scenarios: Sequence[PackageTypeScenario]) -> None:
    patch = cast(dict[str, Any], target["documentPatch"])
    packages = cast(list[dict[str, Any]], patch.get("cargoPackages") or [])
    by_identity = {(row.get("groupId"), row.get("packageId")): row for row in packages}
    for scenario in scenarios:
        if scenario.disposition != "task_facing":
            continue
        assert scenario.projected_package_id is not None
        try:
            row = by_identity[(scenario.group_id, scenario.projected_package_id)]
        except KeyError as error:
            raise ControlledGenerationError(
                "package scenario has no projected target row"
            ) from error
        if scenario.resolution == "sampled_category":
            assert scenario.category_token is not None
            row.pop("typeDescription", None)
            row["typeCategory"] = scenario.category_token
        elif scenario.type_description is not None:
            row.pop("typeCategory", None)
            row["typeDescription"] = scenario.type_description
        else:
            row.pop("typeCategory", None)
            row.pop("typeDescription", None)


def _patch_hs_target(target: dict[str, Any], scenarios: Sequence[Mapping[str, Any]]) -> None:
    patch = cast(dict[str, Any], target["documentPatch"])
    groups = cast(list[dict[str, Any]], patch.get("cargoGroups") or [])
    by_group = {cast(str, row["groupId"]): row for row in groups}
    output: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for scenario in scenarios:
        output[cast(str, scenario["groupId"])].append(
            (cast(int, scenario["valueOrder"]), cast(str, scenario["outputCode"]))
        )
    for group_id, rows in output.items():
        group = by_group[group_id]
        ordered = [value for _order, value in sorted(rows)]
        if len(ordered) != len(group.get("hsCodes") or []):
            raise ControlledGenerationError("HS scenario cardinality differs from source target")
        group["hsCodes"] = ordered


def _patch_voyage_target(target: dict[str, Any], realization: Mapping[str, Any]) -> None:
    patch = cast(dict[str, Any], target["documentPatch"])
    source_transport = patch.get("transport")
    if source_transport is None:
        if realization["voyageNumber"] is not None:
            raise ControlledGenerationError("voyage realization added an absent transport object")
        return
    transport = cast(dict[str, Any], source_transport)
    source_present = "voyageNumber" in transport
    generated = realization["voyageNumber"]
    if source_present != (generated is not None):
        raise ControlledGenerationError("transport presence changed for voyageNumber")
    if generated is not None:
        transport["voyageNumber"] = generated


def _patch_temperature_target(
    target: dict[str, Any], scenarios: Sequence[Mapping[str, Any]]
) -> None:
    patch = cast(dict[str, Any], target["documentPatch"])
    containers = cast(list[dict[str, Any]], patch.get("containers") or [])
    for scenario in scenarios:
        order = cast(int, scenario["containerOrder"])
        if not 0 <= order < len(containers):
            raise ControlledGenerationError("temperature scenario container order is invalid")
        if scenario["value"] is None:
            containers[order].pop("temperatureSetpoint", None)
        else:
            containers[order]["temperatureSetpoint"] = {
                "value": scenario["value"],
                "unit": "celsius",
            }


def _package_scenario_payload(row: PackageTypeScenario) -> dict[str, Any]:
    """Serialize generated package semantics without republishing source wording."""

    return {
        "groupId": row.group_id,
        "sourcePackageId": row.source_package_id,
        "projectedPackageId": row.projected_package_id,
        "sourcePackagePosition": row.source_package_position,
        "sourcePackageLevelCount": row.source_package_level_count,
        "metadataOnlyPackageCount": row.metadata_only_package_count,
        "role": row.role,
        "disposition": row.disposition,
        "resolution": row.resolution,
        "samplingComponent": row.sampling_component,
        "categoryToken": row.category_token,
        "applicationCode": row.application_code,
        "printedSurfaceStatus": row.printed_surface_status,
    }


def _equipment_scenario_payload(
    *, container_order: int, row: EquipmentTypeScenario
) -> dict[str, Any]:
    return {
        "containerOrder": container_order,
        "sourceResolution": row.source_resolution,
        "samplingComponent": row.sampling_component,
        "sizeTypeCode": row.size_type_code,
        "sizeCode": row.size_code,
        "typeCode": row.type_code,
        "typeFamily": row.type_family,
        "thermalCapability": row.thermal_capability,
        "supportsTemperatureSetpoint": row.supports_temperature_setpoint,
        "printedSurfaceStatus": row.printed_surface_status,
        "formProjection": row.form_projection,
        "trainingTypeCategoryStatus": row.training_type_category_status,
    }


def _temperature_scenario_payload(
    *, container_order: int, row: TemperatureScenario
) -> dict[str, Any]:
    return {
        "containerOrder": container_order,
        "equipmentIdentity": row.identity.value,
        "equipmentIdentityAuthority": row.identity.authority,
        "resolution": row.resolution,
        "value": row.value,
        "unit": row.unit,
    }


def _validate_target(document_id: str, target: Mapping[str, Any], *, task: Any) -> str:
    canonical = task.canonicalize(dict(target))
    if canonical != target:
        raise ControlledGenerationError(f"controlled target is not canonical: {document_id}")
    tables = ADAPTER.project(document_id=document_id, source_row_index=0, target=canonical)
    reconstructed = ADAPTER.reconstruct(document_id=document_id, tables=tables.rows)
    if reconstructed != canonical:
        raise ControlledGenerationError(
            f"controlled target fails relational inverse: {document_id}"
        )
    return sha256_bytes(canonical_json_bytes(canonical))


def _plot_bytes(distribution: Mapping[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-untyped]

    sns.set_theme(style="whitegrid", context="talk")
    labels = list(distribution["componentCoverage"])
    complete = [distribution["componentCoverage"][key]["generatedDocuments"] for key in labels]
    blocked = [distribution["componentCoverage"][key]["blockedDocuments"] for key in labels]
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    positions = list(range(len(labels)))
    axis.bar(positions, complete, label="generated", color="#2a9d8f")
    axis.bar(positions, blocked, bottom=complete, label="blocked", color="#e76f51")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("Documents")
    axis.set_title("Controlled 50-scenario semantic coverage")
    axis.legend(frameon=True)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180)
    plt.close(figure)
    return buffer.getvalue()


def _publish_runtime_once(
    stage: StagedArtifactRun, payload: Mapping[str, float]
) -> dict[str, float]:
    root = stage.final_root if stage.completed else stage.stage_root
    path = root / "runtime.json"
    if path.exists() or path.is_symlink():
        value = json.loads(read_regular_file_bytes(path))
        if not isinstance(value, dict) or set(value) != {"elapsedSeconds", "peakRssMiB"}:
            raise ControlledGenerationError("runtime receipt has an invalid contract")
        parsed = {key: float(raw) for key, raw in value.items()}
        if any(not math.isfinite(raw) or raw < 0 for raw in parsed.values()):
            raise ControlledGenerationError("runtime receipt contains an invalid value")
        return parsed
    runtime = {key: float(value) for key, value in payload.items()}
    stage.publish_json("runtime.json", runtime)
    return runtime


def run_controlled_pilot(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisControlledPilotConfig,
) -> dict[str, Any]:
    """Run the controlled semantic-component pilot without publishing training rows."""

    started = time.perf_counter()
    task = _bound_task(project_root, config)
    source_path = _resolve_file(project_root, config.source.file.path, label="source corpus")
    source_rows = _read_jsonl(
        source_path,
        expected_sha256=config.source.file.sha256,
        expected_records=config.source.file.records,
        label="source corpus",
    )
    source_by_id, source_targets, source_hashes = _source_inventory(source_rows, config)
    corpus_ids = frozenset(source_by_id)

    route_root = _resolve_directory(
        project_root, config.inputs.route_scenario_run.path, label="route scenario run"
    )
    _validate_committed_dependency(
        root=route_root,
        commit_sha256=config.inputs.route_scenario_run.commit_sha256,
        transaction_sha256=config.inputs.route_scenario_run.transaction_sha256,
        label="route scenario run",
    )
    route_manifest = _read_json(
        route_root / "manifest.json",
        expected_sha256=config.inputs.route_scenario_run.manifest_sha256,
        label="route scenario manifest",
    )
    route_scenarios = _read_jsonl(
        _resolve_file(project_root, config.inputs.route_scenarios.path, label="route scenarios"),
        expected_sha256=config.inputs.route_scenarios.sha256,
        expected_records=config.inputs.route_scenarios.records,
        label="route scenarios",
    )
    route_targets = _read_jsonl(
        _resolve_file(project_root, config.inputs.route_targets.path, label="route targets"),
        expected_sha256=config.inputs.route_targets.sha256,
        expected_records=config.inputs.route_targets.records,
        label="route targets",
    )
    route_by_base = {cast(str, row["baseDocumentId"]): row for row in route_scenarios}
    target_by_base = {cast(str, row["baseDocumentId"]): row for row in route_targets}
    if (
        len(route_by_base) != len(route_scenarios)
        or len(target_by_base) != len(route_targets)
        or set(route_by_base) != set(target_by_base)
    ):
        raise ValueError("route scenarios and targets do not have one matching row per base")
    selected_ids = tuple(cast(str, row["baseDocumentId"]) for row in route_scenarios)
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("route scenario base document IDs are duplicated")
    for document_id in selected_ids:
        if document_id not in source_targets:
            raise ValueError(f"route target source document is absent: {document_id}")
        route_target = target_by_base[document_id]
        if route_target.get("sourceTargetSha256") != sha256_bytes(
            canonical_json_bytes(dict(source_targets[document_id]))
        ):
            raise ValueError(f"route source target lineage differs: {document_id}")
        projected_target = route_target.get("target")
        if not isinstance(projected_target, dict) or route_target.get(
            "projectedTargetSha256"
        ) != sha256_bytes(canonical_json_bytes(projected_target)):
            raise ValueError(f"route projected target lineage differs: {document_id}")

    preparation_root = _resolve_directory(
        project_root, config.inputs.preparation.path, label="preparation run"
    )
    preparation_manifest = _read_json(
        preparation_root / "manifest.json",
        expected_sha256=config.inputs.preparation.manifest_sha256,
        label="preparation manifest",
    )
    tables = _preparation_tables(root=preparation_root, manifest=preparation_manifest)
    template_rows = _read_jsonl(
        _resolve_file(project_root, config.inputs.template_groups.path, label="template groups"),
        expected_sha256=config.inputs.template_groups.sha256,
        expected_records=config.inputs.template_groups.records,
        label="template groups",
    )
    template_by_document = _template_map(template_rows, corpus_ids)
    partition_report = _read_json(
        _resolve_file(project_root, config.inputs.partition_report.path, label="partition report"),
        expected_sha256=config.inputs.partition_report.sha256,
        label="partition report",
    )
    partition_by_document, train_ids = _partition_map(partition_report)
    if set(partition_by_document) != corpus_ids:
        raise ValueError("partition report does not cover the source corpus exactly")
    scope = TrainOnlySourceScope(
        corpus_document_ids=tuple(sorted(corpus_ids)),
        fit_document_ids=train_ids,
        template_by_document=template_by_document,
        source_sha256_by_document=source_hashes,
        fit_split="train",
    )
    fit_ids = tuple(sorted(scope.isolated_fit_document_ids))
    scope.assert_fit_rows(fit_ids, purpose="controlled semantic support")
    if not set(selected_ids) <= set(fit_ids):
        raise ValueError("controlled base documents lie outside the isolated fit scope")

    category_rows = _read_jsonl(
        _resolve_file(
            project_root,
            config.inputs.category_metadata.path,
            label="category metadata",
        ),
        expected_sha256=config.inputs.category_metadata.sha256,
        expected_records=config.inputs.category_metadata.records,
        label="category metadata",
    )
    hierarchy_rows = _read_jsonl(
        _resolve_file(
            project_root,
            config.inputs.package_hierarchy_metadata.path,
            label="package hierarchy metadata",
        ),
        expected_sha256=config.inputs.package_hierarchy_metadata.sha256,
        expected_records=config.inputs.package_hierarchy_metadata.records,
        label="package hierarchy metadata",
    )
    package_registry = load_package_registry(
        _resolve_file(project_root, config.inputs.package_registry.path, label="package registry"),
        expected_sha256=config.inputs.package_registry.sha256,
        expected_entries=config.inputs.package_registry_entries,
    )
    views = build_modeling_views(
        tables=tables,
        train_document_ids=fit_ids,
        package_registry=package_registry,
        category_metadata_rows=category_rows,
        package_hierarchy_metadata_rows=hierarchy_rows,
    )

    package_observations = package_fit_observations(
        fit_document_ids=fit_ids,
        hierarchy=views.package_hierarchy,
        packages=views.cargo_package_numeric,
    )
    package_support = build_package_scenario_support(
        observations=package_observations,
        fit_document_ids=fit_ids,
        allowed_category_tokens=_allowed_package_categories(category_rows),
        registry=package_registry,
        expected_registry_sha256=config.inputs.package_registry.sha256,
    )
    packages_by_document: dict[str, list[PackageFitObservation]] = defaultdict(list)
    for package_observation in package_observations:
        packages_by_document[package_observation.source_document_id].append(package_observation)

    equipment_manifest_path = _resolve_file(
        project_root,
        config.inputs.equipment_registry_manifest.path,
        label="equipment registry manifest",
    )
    _verify_sha256(
        equipment_manifest_path,
        expected_sha256=config.inputs.equipment_registry_manifest.sha256,
        label="equipment registry manifest",
    )
    equipment_registry = load_bic_equipment_registry(equipment_manifest_path)
    equipment_observations = _classify_exact_equipment(
        _equipment_observations(views), equipment_registry
    )
    equipment_support = build_equipment_scenario_support(
        observations=equipment_observations,
        fit_document_ids=fit_ids,
        registry=equipment_registry,
    )
    equipment_by_document: dict[str, list[EquipmentFitObservation]] = defaultdict(list)
    for equipment_observation in equipment_observations:
        equipment_by_document[equipment_observation.source_document_id].append(
            equipment_observation
        )

    reefer_observations = _reefer_observations(views)
    reefer_support = build_reefer_temperature_support(
        observations=reefer_observations,
        fit_document_ids=fit_ids,
    )
    reefer_by_identity = {
        (row.source_document_id, row.equipment_row_id): row for row in reefer_observations
    }

    hs_manifest_path = _resolve_file(
        project_root, config.inputs.hs_registry_manifest.path, label="HS source pin"
    )
    _verify_sha256(
        hs_manifest_path,
        expected_sha256=config.inputs.hs_registry_manifest.sha256,
        label="HS source pin",
    )
    hs_metadata_path = _resolve_file(
        project_root, config.inputs.hs_metadata.path, label="HS metadata"
    )
    hs_report_path = _resolve_file(
        project_root, config.inputs.hs_commodities_report.path, label="HS commodities report"
    )
    _verify_sha256(
        hs_metadata_path,
        expected_sha256=config.inputs.hs_metadata.sha256,
        label="HS metadata",
    )
    _verify_sha256(
        hs_report_path,
        expected_sha256=config.inputs.hs_commodities_report.sha256,
        label="HS commodities report",
    )
    hs_source_pin = load_ukgt_source_pin(hs_manifest_path)
    hs_registry = compile_uk_global_tariff_registry(
        metadata_path=hs_metadata_path,
        report_path=hs_report_path,
        source=hs_source_pin,
    )
    hs_fit = hs_fit_observations(
        fit_document_ids=fit_ids,
        cargo_hs_code_rows=tables["cargo_hs_codes"],
    )
    hs_support = build_hs_scenario_support(
        fit_document_ids=fit_ids,
        observations=hs_fit,
        registry=hs_registry,
        on_date=hs_source_pin.snapshot_date,
    )
    hs_policy = HsScenarioPolicy.model_validate(
        config.generation.hs.model_dump(mode="python"), strict=True
    )
    hs_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hs_table_row in tables["cargo_hs_codes"]:
        document_id = cast(str, hs_table_row["document_id"])
        if document_id in set(selected_ids):
            hs_by_document[document_id].append(hs_table_row)

    countries = load_iso_country_registry(
        iso_path=_resolve_file(
            project_root, config.inputs.iso3166_snapshot.path, label="ISO-3166 snapshot"
        ),
        iso_sha256=config.inputs.iso3166_snapshot.sha256,
    )
    cargo_origin_support = build_cargo_origin_support(
        source_targets=source_targets,
        fit_document_ids=fit_ids,
        country_registry=countries,
    )

    document_rows = tables["documents"]
    transport_bundle = build_transport_identity_bundle(
        source_documents=document_rows,
        fit_document_ids=fit_ids,
        template_by_document=template_by_document,
        partition_by_document=partition_by_document,
        allowed_partition="train",
    )
    transport_index = {row_id: index for index, row_id in enumerate(transport_bundle.view.row_ids)}
    compact_transport_by_document = {
        document_id: transport_bundle.view.data.iloc[transport_index[document_id]].to_dict()
        for document_id in selected_ids
    }
    compact_transport = [compact_transport_by_document[document_id] for document_id in selected_ids]
    expanded_transport = [expand_transport_identity_structure(row) for row in compact_transport]
    expanded_transport_by_document = dict(zip(selected_ids, expanded_transport, strict=True))
    voyage_policy = VoyageNumberPolicy(
        minimum_normalized_edit_distance=(
            config.generation.transport.minimum_normalized_edit_distance
        ),
        maximum_attempts=config.generation.transport.maximum_realization_attempts,
    )
    if config.generation.transport.voyage_number_method == "observed_character_class_shape_v1":
        voyage_realizations = realize_voyage_numbers(
            rows=expanded_transport,
            row_ids=selected_ids,
            stream=DeterministicStream(
                config.generation.seed,
                "controlled-voyage-v1",
                config.run.run_id,
            ),
            guard=transport_bundle.guard,
            policy=voyage_policy,
        )
    else:
        voyage_realizations = ()
    voyage_by_document = {row.row_id: row for row in voyage_realizations}
    generated_voyages = tuple(
        row.voyage_number for row in voyage_realizations if row.voyage_number is not None
    )
    source_voyage_collisions = sum(
        transport_bundle.guard.classify_voyage_number(value, policy=voyage_policy) != "none"
        for value in generated_voyages
    )
    duplicate_generated_voyages = len(generated_voyages) - len(set(generated_voyages))
    if source_voyage_collisions or duplicate_generated_voyages:
        raise ControlledGenerationError(
            "generated voyage numbers failed source or batch collision validation"
        )
    party_benchmark = _read_json(
        _resolve_file(
            project_root,
            config.inputs.party_identity_benchmark_summary.path,
            label="party benchmark summary",
        ),
        expected_sha256=config.inputs.party_identity_benchmark_summary.sha256,
        label="party benchmark summary",
    )
    if party_benchmark.get("decision", {}).get("sdvRole") != (
        "model_party_structure_not_identity_strings"
    ):
        raise ValueError("party benchmark decision differs from the controlled safety contract")

    dangerous_by_document = Counter(
        cast(str, dangerous_row["document_id"]) for dangerous_row in tables["dangerous_goods"]
    )
    handling_by_document = Counter(
        cast(str, handling_row["document_id"])
        for handling_row in tables["cargo_handling_instructions"]
    )
    parties_by_document = Counter(
        cast(str, party_row["document_id"]) for party_row in tables["parties"]
    )
    party_contacts_by_document = Counter(
        cast(str, contact_row["document_id"]) for contact_row in tables["party_contacts"]
    )
    cargo_groups_by_document = Counter(
        cast(str, cargo_row["document_id"]) for cargo_row in tables["cargo_groups"]
    )
    container_order_by_row: dict[str, int] = {}
    for container_row in tables["containers"]:
        container_order_by_row[cast(str, container_row["container_id"])] = cast(
            int, container_row["container_order"]
        )
    equipment_container_row_by_identity: dict[str, str] = {}
    for view_row in views.container_equipment:
        direct = tuple(
            projection.source
            for projection in view_row.projection.direct
            if projection.feature == "container_order"
        )
        if len(direct) != 1 or direct[0].table != "containers":
            raise ControlledGenerationError(
                "equipment projection has no exact source container reference"
            )
        equipment_container_row_by_identity[view_row.projection.view_row_key] = direct[0].row_key

    output_rows: list[dict[str, Any]] = []
    draft_target_rows: list[dict[str, Any]] = []
    validated_draft_targets = 0
    blocker_counts: Counter[str] = Counter()
    component_coverage: dict[str, Counter[str]] = defaultdict(Counter)
    package_categories: Counter[str] = Counter()
    package_sampling_components: Counter[str] = Counter()
    equipment_codes: Counter[str] = Counter()
    equipment_sampling_components: Counter[str] = Counter()
    hs_chapters: Counter[str] = Counter()
    hs_sampling_components: Counter[str] = Counter()
    hs_output_lengths: Counter[int] = Counter()
    hs_extension_statuses: Counter[str] = Counter()
    reserved_hs_outputs = {cast(str, row["value"]) for row in tables["cargo_hs_codes"]}
    reserved_global_hs6 = {value[:6] for value in reserved_hs_outputs}
    commercial_origins: Counter[str] = Counter()
    commercial_destinations: Counter[str] = Counter()
    freight_arrangements: Counter[str] = Counter()

    for document_id in selected_ids:
        route = route_by_base[document_id]
        route_target_row = target_by_base[document_id]
        synthetic_id = cast(str, route_target_row["syntheticDocumentId"])
        if synthetic_id != route.get("syntheticDocumentId"):
            raise ValueError("route scenario and target synthetic IDs differ")
        target = cast(dict[str, Any], deepcopy(route_target_row["target"]))
        blockers: list[str] = []
        shipment_payload = {
            "commercialOriginCountryCode": route["commercialOriginCountryCode"],
            "commercialDestinationCountryCode": route["commercialDestinationCountryCode"],
            "commercialOriginPriorEvidence": deepcopy(route["commercialOriginPriorEvidence"]),
            "tradeFlowEvidence": deepcopy(route["tradeFlowEvidence"]),
            "physicalEndpointCountryMethods": deepcopy(route["physicalEndpointCountryMethods"]),
            "loadingPort": deepcopy(route["loadingPort"]),
            "dischargePort": deepcopy(route["dischargePort"]),
            "routeLocations": deepcopy(route["routeLocations"]),
            "placeOfIssue": deepcopy(route["placeOfIssue"]),
            "partyLocalities": deepcopy(route["partyLocalities"]),
            "freight": deepcopy(route["freight"]),
            "transshipmentStatus": route["transshipmentStatus"],
        }
        commercial_origins[cast(str, route["commercialOriginCountryCode"])] += 1
        commercial_destinations[cast(str, route["commercialDestinationCountryCode"])] += 1
        freight_arrangement = route["freight"]["arrangement"]
        if freight_arrangement is not None and not isinstance(freight_arrangement, str):
            raise ControlledGenerationError("freight arrangement must be text or null")
        freight_arrangements[freight_arrangement or "missing"] += 1
        component_coverage["routeAndFreight"]["generatedDocuments"] += 1

        package_scenarios: list[PackageTypeScenario] = []
        for package_observation in packages_by_document.get(document_id, ()):
            if package_observation.identity in package_support.excluded_identities:
                blockers.append("package_role_category_conflict")
                continue
            package_scenario = sample_package_type(
                package_observation,
                support=package_support,
                registry_exploration_permyriad=(
                    config.generation.package_registry_exploration_permyriad
                ),
                stream=DeterministicStream(
                    config.generation.seed,
                    "controlled-package-v1",
                    f"{synthetic_id}:{package_observation.group_id}:"
                    f"{package_observation.source_package_id}",
                ),
            )
            package_scenarios.append(package_scenario)
            package_sampling_components[package_scenario.sampling_component] += 1
            if package_scenario.category_token is not None:
                package_categories[package_scenario.category_token] += 1
        _patch_package_target(target, package_scenarios)
        component_coverage["packages"]["generatedDocuments"] += int(
            any(row.resolution == "sampled_category" for row in package_scenarios)
        )
        component_coverage["packages"]["blockedDocuments"] += int(
            "package_role_category_conflict" in blockers
        )

        equipment_scenarios: list[dict[str, Any]] = []
        temperature_scenarios: list[dict[str, Any]] = []
        for equipment_observation in equipment_by_document.get(document_id, ()):
            try:
                container_row_id = equipment_container_row_by_identity[
                    equipment_observation.equipment_row_id
                ]
                order = container_order_by_row[container_row_id]
            except KeyError as error:
                raise ControlledGenerationError(
                    "equipment row cannot resolve its exact source container: "
                    f"{equipment_observation.equipment_row_id}"
                ) from error
            sampled_equipment: EquipmentTypeScenario | None = None
            if equipment_observation.printed_surface is not None:
                sampled_equipment = sample_equipment_type(
                    equipment_observation,
                    support=equipment_support,
                    registry=equipment_registry,
                    registry_exploration_permyriad=(
                        config.generation.equipment_registry_exploration_permyriad
                    ),
                    stream=DeterministicStream(
                        config.generation.seed,
                        "controlled-equipment-v1",
                        f"{synthetic_id}:{equipment_observation.equipment_row_id}",
                    ),
                )
                equipment_codes[sampled_equipment.size_type_code] += 1
                equipment_sampling_components[sampled_equipment.sampling_component] += 1
                equipment_scenarios.append(
                    _equipment_scenario_payload(
                        container_order=order,
                        row=sampled_equipment,
                    )
                )
            reefer_observation = reefer_by_identity.get(
                (document_id, equipment_observation.equipment_row_id)
            )
            if equipment_observation.temperature_setpoint_present:
                if reefer_observation is None or sampled_equipment is None:
                    blockers.append("reefer_identity_missing")
                else:
                    generated_identity = ReeferEquipmentIdentity(
                        value=sampled_equipment.size_type_code,
                        authority="authoritative_registry",
                        is_reefer=sampled_equipment.supports_temperature_setpoint,
                    )
                    sampled_temperature = sample_generated_equipment_temperature(
                        identity=generated_identity,
                        source_setpoint_present=True,
                        support=reefer_support,
                        stream=DeterministicStream(
                            config.generation.seed,
                            "controlled-reefer-v1",
                            f"{synthetic_id}:{equipment_observation.equipment_row_id}",
                        ),
                    )
                    temperature_scenarios.append(
                        _temperature_scenario_payload(
                            container_order=order,
                            row=sampled_temperature,
                        )
                    )
        _patch_temperature_target(target, temperature_scenarios)
        component_coverage["equipment"]["generatedDocuments"] += int(bool(equipment_scenarios))
        component_coverage["equipment"]["blockedDocuments"] += int(
            "reefer_identity_missing" in blockers
        )

        hs_scenarios: list[dict[str, Any]] = []
        for source_hs in sorted(
            hs_by_document.get(document_id, ()),
            key=lambda row: (cast(str, row["cargo_group_row_id"]), cast(int, row["value_order"])),
        ):
            group_id = cast(str, source_hs["cargo_group_row_id"]).rsplit(":", maxsplit=1)[-1]
            sampled_hs = sample_hs_scenario(
                support=hs_support,
                registry=hs_registry,
                policy=hs_policy,
                customs_jurisdiction=cast(str, route["commercialDestinationCountryCode"]),
                source_output_digits=len(cast(str, source_hs["value"])),
                stream=DeterministicStream(
                    config.generation.seed,
                    "controlled-hs-v1",
                    f"{synthetic_id}:{cast(str, source_hs['cargo_group_value_id'])}",
                ),
                excluded_output_codes=reserved_hs_outputs,
                excluded_global_hs6=reserved_global_hs6,
            )
            reserved_hs_outputs.add(sampled_hs.output_code)
            reserved_global_hs6.add(sampled_hs.global_identity.code)
            hs_chapters[sampled_hs.global_identity.chapter_code] += 1
            hs_sampling_components[sampled_hs.component] += 1
            hs_output_lengths[sampled_hs.output_digits] += 1
            hs_extension_statuses[sampled_hs.extension_status] += 1
            hs_scenarios.append(
                {
                    "groupId": group_id,
                    "valueOrder": source_hs["value_order"],
                    "sourceRowId": source_hs["cargo_group_value_id"],
                    "component": sampled_hs.component,
                    "customsJurisdiction": sampled_hs.customs_jurisdiction,
                    "outputScope": sampled_hs.output_scope,
                    "outputCode": sampled_hs.output_code,
                    "outputDigits": sampled_hs.output_digits,
                    "extensionStatus": sampled_hs.extension_status,
                    "globalHs6": sampled_hs.global_identity.code,
                    "chapterCode": sampled_hs.global_identity.chapter_code,
                    "printedSurfaceStatus": sampled_hs.printed_surface_status,
                    "dangerousGoodsStatus": sampled_hs.dangerous_goods_status,
                }
            )
        _patch_hs_target(target, hs_scenarios)
        component_coverage["hsCodes"]["generatedDocuments"] += int(bool(hs_scenarios))

        cargo_origin: dict[str, Any]
        try:
            origin_projection = sample_cargo_origin_projection(
                source_target=target,
                support=cargo_origin_support,
                country_registry=countries,
                commercial_origin_country_code=cast(str, route["commercialOriginCountryCode"]),
                stream=DeterministicStream(
                    config.generation.seed,
                    "controlled-cargo-origin-v1",
                    synthetic_id,
                ),
            )
        except CargoOriginScenarioError:
            blockers.append("cargo_origin_unresolved_source_form")
            cargo_origin = {
                "status": "blocked",
                "reasonCode": "unresolved_source_form",
            }
            component_coverage["cargoOrigins"]["blockedDocuments"] += 1
        else:
            target = project_cargo_origin_target(source_target=target, projection=origin_projection)
            cargo_origin = {"status": "generated", **origin_projection.to_dict()}
            component_coverage["cargoOrigins"]["generatedDocuments"] += int(
                bool(origin_projection.origins)
            )

        source_transport_structure = expanded_transport_by_document[document_id]
        vessel_source_present = bool(source_transport_structure["vessel_name_present"])
        imo_source_present = bool(source_transport_structure["source_imo_present"])
        if config.generation.transport.voyage_number_method == (
            "observed_character_class_shape_v1"
        ):
            voyage = voyage_by_document[document_id]
            voyage_number = voyage.voyage_number
            voyage_attempts = voyage.attempts
            _patch_voyage_target(target, {"voyageNumber": voyage_number})
            component_coverage["transport"]["generatedDocuments"] += int(voyage_number is not None)
        else:
            patch = cast(dict[str, Any], target["documentPatch"])
            source_transport = cast(dict[str, Any] | None, patch.get("transport"))
            voyage_number = (
                cast(str | None, source_transport.get("voyageNumber"))
                if source_transport is not None
                else None
            )
            voyage_attempts = 0
        transport_payload = {
            "vesselName": None,
            "vesselNameSourcePresent": vessel_source_present,
            "vesselNameStatus": (
                "deferred_by_explicit_scope" if vessel_source_present else "not_present"
            ),
            "voyageNumber": voyage_number,
            "voyageAttempts": voyage_attempts,
            "voyageMethod": config.generation.transport.voyage_number_method,
            "vesselImoNumber": None,
            "vesselImoSourcePresent": imo_source_present,
            "imoStatus": (
                "pending_authoritative_assigned_number_registry"
                if imo_source_present
                else "not_present"
            ),
            "imoPolicy": config.generation.transport.imo_policy,
        }
        if vessel_source_present:
            blockers.append("vessel_name_deferred_by_explicit_scope")
        if imo_source_present:
            blockers.append("vessel_imo_requires_authoritative_assigned_number_registry")
        component_coverage["transport"]["blockedDocuments"] += int(
            vessel_source_present or imo_source_present
        )

        dangerous_count = dangerous_by_document[document_id]
        dangerous_goods = {
            "sourceRecordCount": dangerous_count,
            "status": (
                "deferred_goods_first_coherent_semantic_realization"
                if dangerous_count
                else "not_present"
            ),
            "hsInferenceForbidden": True,
        }
        if dangerous_count:
            blockers.append("dangerous_goods_requires_goods_first_semantic_realization")
            component_coverage["dangerousGoods"]["blockedDocuments"] += 1

        handling_count = handling_by_document[document_id]
        handling_instructions = {
            "sourceRecordCount": handling_count,
            "status": (
                "pending_cargo_fact_conditioned_linguistic_realization"
                if handling_count
                else "not_present"
            ),
            "hardcodedInstructionRegistryForbidden": True,
        }
        if handling_count:
            blockers.append("handling_instructions_require_linguistic_realization")
            component_coverage["handlingInstructions"]["blockedDocuments"] += 1

        party_count = parties_by_document[document_id]
        contact_count = party_contacts_by_document[document_id]
        cargo_group_count = cargo_groups_by_document[document_id]
        if party_count:
            blockers.append("party_identity_and_contact_anonymization")
        if cargo_group_count:
            blockers.append("cargo_and_auxiliary_text_realization")
        if any(
            row.printed_surface_status == "pending_text_realization" for row in package_scenarios
        ):
            blockers.append("package_printed_surface_realization")
        if equipment_scenarios:
            blockers.append("equipment_printed_surface_realization")
        if hs_scenarios:
            blockers.append("hs_printed_surface_realization")
        blockers.append("raw_ocr_patch_planning_and_execution")
        if equipment_scenarios:
            blockers.append("container_type_task_schema_projection")
        unique_blockers = tuple(sorted(set(blockers)))
        blocker_counts.update(unique_blockers)
        target_sha256 = _validate_target(synthetic_id, target, task=task)
        validated_draft_targets += 1
        output_scenario = {
            "schemaVersion": 1,
            "syntheticDocumentId": synthetic_id,
            "baseDocumentId": document_id,
            "templateId": template_by_document[document_id],
            "commercialRoute": {
                "originCountryCode": route["commercialOriginCountryCode"],
                "destinationCountryCode": route["commercialDestinationCountryCode"],
                "loadingLocode": route["loadingPort"]["locode"],
                "dischargeLocode": route["dischargePort"]["locode"],
            },
            "shipmentScenario": shipment_payload,
            "transportIdentity": transport_payload,
            "packageTypes": [_package_scenario_payload(row) for row in package_scenarios],
            "equipmentTypes": equipment_scenarios,
            "temperatureSetpoints": temperature_scenarios,
            "hsCodes": hs_scenarios,
            "cargoOrigins": cargo_origin,
            "dangerousGoods": dangerous_goods,
            "handlingInstructions": handling_instructions,
            "remainingLinguisticWork": {
                "partyRecords": party_count,
                "partyContactRecords": contact_count,
                "cargoGroupRecords": cargo_group_count,
                "packagePrintedSurfaces": sum(
                    row.printed_surface_status == "pending_text_realization"
                    for row in package_scenarios
                ),
                "equipmentPrintedSurfaces": len(equipment_scenarios),
                "hsPrintedSurfaces": len(hs_scenarios),
            },
            "remainingBlockers": list(unique_blockers),
            "draftTargetSha256": target_sha256,
            "trainingEligible": False,
        }
        output_rows.append(output_scenario)
        draft_target_rows.append(
            {
                "baseDocumentId": document_id,
                "syntheticDocumentId": synthetic_id,
                "templateId": template_by_document[document_id],
                "sourceTargetSha256": route_target_row["sourceTargetSha256"],
                "routeProjectedTargetSha256": route_target_row["projectedTargetSha256"],
                "draftTargetSha256": target_sha256,
                "target": target,
                "trainingEligible": False,
            }
        )

    component_keys = (
        "routeAndFreight",
        "transport",
        "packages",
        "equipment",
        "hsCodes",
        "cargoOrigins",
        "dangerousGoods",
        "handlingInstructions",
    )
    distribution = {
        "documents": len(output_rows),
        "componentCoverage": {
            key: {
                "generatedDocuments": component_coverage[key]["generatedDocuments"],
                "blockedDocuments": component_coverage[key]["blockedDocuments"],
            }
            for key in component_keys
        },
        "remainingBlockerCounts": dict(sorted(blocker_counts.items())),
        "packageCategoryCounts": dict(sorted(package_categories.items())),
        "packageSamplingComponentCounts": dict(sorted(package_sampling_components.items())),
        "equipmentSizeTypeCodeCounts": dict(sorted(equipment_codes.items())),
        "equipmentSamplingComponentCounts": dict(sorted(equipment_sampling_components.items())),
        "hsChapterCounts": dict(sorted(hs_chapters.items())),
        "hsSamplingComponentCounts": dict(sorted(hs_sampling_components.items())),
        "hsOutputLengthCounts": {
            str(length): count for length, count in sorted(hs_output_lengths.items())
        },
        "hsExtensionStatusCounts": dict(sorted(hs_extension_statuses.items())),
        "commercialOriginCountryCounts": dict(sorted(commercial_origins.items())),
        "commercialDestinationCountryCounts": dict(sorted(commercial_destinations.items())),
        "freightArrangementCounts": dict(sorted(freight_arrangements.items())),
        "voyageShapeGenerationMethod": config.generation.transport.voyage_number_method,
    }
    validation = {
        "selectedDocuments": len(selected_ids),
        "controlledScenarios": len(output_rows),
        "strictSchemaValidDraftTargets": validated_draft_targets,
        "relationalInverseValidDraftTargets": validated_draft_targets,
        "actualGeneratedVesselNames": 0,
        "pendingVesselNames": sum(
            row["transportIdentity"]["vesselNameSourcePresent"] for row in output_rows
        ),
        "actualGeneratedVoyageNumbers": sum(
            row["transportIdentity"]["voyageNumber"] is not None
            for row in output_rows
            if row["transportIdentity"]["voyageMethod"] == "observed_character_class_shape_v1"
        ),
        "preservedUpstreamVoyageNumbers": sum(
            row["transportIdentity"]["voyageNumber"] is not None
            for row in output_rows
            if row["transportIdentity"]["voyageMethod"]
            == "preserve_for_upstream_structured_identifier_v1"
        ),
        "sourceTransportIdentityCollisions": source_voyage_collisions,
        "duplicateGeneratedVoyageNumbers": duplicate_generated_voyages,
        "generatedVoyageNumberAttemptsMean": (
            sum(row.attempts for row in voyage_realizations if row.voyage_number is not None)
            / len(generated_voyages)
            if generated_voyages
            else 0.0
        ),
        "generatedVoyageNumberAttemptsMaximum": max(
            (row.attempts for row in voyage_realizations), default=0
        ),
        "distinctTemplates": len({row["templateId"] for row in output_rows}),
        "trainingRecordsPublished": 0,
    }
    implementation = {
        path.name: sha256_file(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("transport_identity.py"),
            Path(__file__).with_name("package_scenarios.py"),
            Path(__file__).with_name("equipment_scenarios.py"),
            Path(__file__).with_name("reefer_scenarios.py"),
            Path(__file__).with_name("hs_registry.py"),
            Path(__file__).with_name("hs_scenarios.py"),
            Path(__file__).with_name("cargo_origin_scenarios.py"),
        )
    }
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "contract": "mpci-bl-controlled-semantic-component-pilot-v1",
                "configSha256": sha256_file(config_path),
                "config": config.model_dump(mode="json"),
                "sourceScopeSha256": scope.scope_sha256,
                "routeManifestSha256": config.inputs.route_scenario_run.manifest_sha256,
                "preparationManifestSha256": config.inputs.preparation.manifest_sha256,
                "partyBenchmarkSha256": config.inputs.party_identity_benchmark_summary.sha256,
                "implementation": implementation,
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
    stage.recover_interrupted_temporary_files()
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    stage.publish_json(
        "source-scope.json",
        {
            "fitSplit": "train",
            "partitionDocuments": len(train_ids),
            "templateIsolatedDocuments": len(fit_ids),
            "scopeSha256": scope.scope_sha256,
        },
    )
    stage.publish_json("modeling/package-support-audit.json", asdict(package_support.audit))
    stage.publish_json("modeling/equipment-support-audit.json", asdict(equipment_support.audit))
    stage.publish_json("modeling/reefer-support-audit.json", asdict(reefer_support.audit))
    stage.publish_json("modeling/hs-support-audit.json", asdict(hs_support.audit))
    stage.publish_json(
        "modeling/cargo-origin-support-audit.json", cargo_origin_support.audit.to_dict()
    )
    stage.publish_json(
        "modeling/transport-generation-policy.json",
        {
            "vesselNameMethod": config.generation.transport.vessel_name_method,
            "voyageNumberMethod": config.generation.transport.voyage_number_method,
            "minimumNormalizedEditDistance": (
                config.generation.transport.minimum_normalized_edit_distance
            ),
            "maximumRealizationAttempts": (
                config.generation.transport.maximum_realization_attempts
            ),
            "imoPolicy": config.generation.transport.imo_policy,
        },
    )
    stage.publish_json(
        "modeling/party-identity-benchmark-decision.json",
        party_benchmark["decision"],
    )
    stage.publish_bytes("generation/controlled-scenarios.jsonl", _jsonl(output_rows))
    stage.publish_bytes("generation/draft-targets.jsonl", _jsonl(draft_target_rows))
    stage.publish_json("generation/distribution-summary.json", distribution)
    stage.publish_json("generation/validation-summary.json", validation)
    stage.publish_bytes("plots/01_component_coverage.png", _plot_bytes(distribution))
    runtime = _publish_runtime_once(
        stage,
        {
            "elapsedSeconds": time.perf_counter() - started,
            "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )
    manifest = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "status": "controlled_semantic_component_pilot_complete",
        "trainingEligible": False,
        "sourceRecords": len(source_rows),
        "fitDocuments": len(fit_ids),
        "scenarioDocuments": len(output_rows),
        "actualSyntheticTransportIdentities": {
            "vesselNames": validation["actualGeneratedVesselNames"],
            "pendingVesselNames": validation["pendingVesselNames"],
            "voyageNumbers": validation["actualGeneratedVoyageNumbers"],
            "vesselNameMethod": config.generation.transport.vessel_name_method,
            "voyageNumberMethod": config.generation.transport.voyage_number_method,
        },
        "generationMethods": {
            "packages": config.generation.package_method,
            "equipment": config.generation.equipment_method,
            "hs": config.generation.hs.schema_version,
            "cargoOrigin": config.generation.cargo_origin_method,
            "dangerousGoods": config.generation.dangerous_goods_method,
            "vesselName": config.generation.transport.vessel_name_method,
            "voyageNumber": config.generation.transport.voyage_number_method,
        },
        "modelOrApiCalls": 0,
        "trainingRecordsPublished": 0,
        "distribution": distribution,
        "validation": validation,
        "implementationSha256": implementation,
        "transactionSha256": transaction,
    }
    stage.publish_json("manifest.json", manifest)
    expected = (
        "config.yaml",
        "generation/controlled-scenarios.jsonl",
        "generation/draft-targets.jsonl",
        "generation/distribution-summary.json",
        "generation/validation-summary.json",
        "manifest.json",
        "modeling/cargo-origin-support-audit.json",
        "modeling/equipment-support-audit.json",
        "modeling/hs-support-audit.json",
        "modeling/package-support-audit.json",
        "modeling/party-identity-benchmark-decision.json",
        "modeling/reefer-support-audit.json",
        "modeling/transport-generation-policy.json",
        "plots/01_component_coverage.png",
        "runtime.json",
        "source-scope.json",
    )
    commit = stage.commit(
        expected_artifacts=expected,
        metadata={
            "status": "controlled_semantic_component_pilot_complete",
            "trainingEligible": False,
            "scenarioDocuments": len(output_rows),
            "actualGeneratedVesselNames": validation["actualGeneratedVesselNames"],
            "actualGeneratedVoyageNumbers": validation["actualGeneratedVoyageNumbers"],
        },
    )
    return {
        **manifest,
        "commitCreated": commit.created,
        "commitContentSha256": commit.receipt.content_sha256,
        "runtime": runtime,
        "routeManifestStatus": route_manifest.get("status"),
        "preparationRunId": preparation_manifest.get("runId"),
    }
