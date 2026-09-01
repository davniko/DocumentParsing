"""Complete deterministic B/L semantics before linguistic OCR realization.

This stage composes four related decisions in their dependency order:

1. assign the configured thermal-cargo cohort at run level;
2. sample and reclassify a compatible cargo semantic identity;
3. derive compatible equipment and a setpoint from that cargo identity; and
4. sample sparse transport auxiliaries and formulation-level flashpoints.

The cohort assignment is stratified sampling: it controls prevalence without
ever assigning refrigeration from an unrelated equipment draw.  The sampled
goods identity remains the source of truth for all downstream thermal fields.

The output is a relation-v5 target plus complete provenance.  It is not a
training record: printed surfaces and raw-OCR edits belong to the later
linguistic realization stage.
"""

from __future__ import annotations

import json
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas import bill_of_lading_v5 as bill_of_lading_v5_module
from document_ocr.label_schemas.bill_of_lading_v5 import (
    CONTAINER_TYPE_CATEGORIES,
    migrate_relation_v4_target_to_v5,
)
from document_ocr.synthesis import container_semantics as container_semantics_module
from document_ocr.synthesis import flashpoint_scenarios as flashpoint_scenarios_module
from document_ocr.synthesis import task_adapter as task_adapter_module
from document_ocr.synthesis import thermal_goods as thermal_goods_module
from document_ocr.synthesis import transport_auxiliary as transport_auxiliary_module
from document_ocr.synthesis.anchors import leaf_items, normalized_role_path
from document_ocr.synthesis.config import SynthesisSemanticCompletionConfig
from document_ocr.synthesis.container_semantics import (
    EquipmentSemanticSupport,
    SourceEquipmentObservation,
    build_equipment_semantic_support,
    review_source_equipment_surface,
    sample_equipment_semantic,
)
from document_ocr.synthesis.country_registry import load_iso_country_registry
from document_ocr.synthesis.dangerous_goods_plan_pipeline import DangerousGoodsPlanRow
from document_ocr.synthesis.equipment_registry import load_bic_equipment_registry
from document_ocr.synthesis.flashpoint_scenarios import (
    FlashpointEligibility,
    classify_flashpoint_eligibility,
    sample_flashpoint,
)
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import (
    UkGlobalTariffRegistry,
    compile_uk_global_tariff_registry,
    load_ukgt_source_pin,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.scenario_state import ScenarioChange
from document_ocr.synthesis.task_adapter import (
    BILL_OF_LADING_TASK_ADAPTER,
    BILL_OF_LADING_V5_TASK_ADAPTER,
)
from document_ocr.synthesis.thermal_goods import (
    AmbientGoodsIdentity,
    ThermalGoodsIdentity,
    ThermalGoodsSupport,
    build_thermal_goods_support,
    render_synthetic_hs_code,
    sample_ambient_goods,
    sample_temperature_setpoint,
    sample_thermal_goods,
    sample_thermal_profile,
)
from document_ocr.synthesis.transport_auxiliary import (
    SyntheticImoNumber,
    maritime_flag_country_codes,
    sample_imo_number,
    sample_vessel_flag,
)
from document_ocr.synthesis.world_port_registry import load_pinned_world_port_records
from document_ocr.training import tasks as training_tasks_module
from document_ocr.training.config import resolve_config_path
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_STAGE_ID = "semantic-completion-v1"


class CargoSemanticRealization(BaseModel):
    model_config = _STRICT

    cargo_group_id: NonEmptyText
    identity_order: Annotated[int, Field(ge=0)]
    semantic_source: Literal["ambient_hs_registry", "thermal_hs_registry"]
    hs6: Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
    output_hs_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")] | None
    chapter_description: NonEmptyText
    heading_description: NonEmptyText
    description: NonEmptyText
    thermal_profile: Literal["FROZEN", "CHILLED"] | None
    associated_container_numbers: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def thermal_branch_is_complete(self) -> CargoSemanticRealization:
        thermal = self.semantic_source == "thermal_hs_registry"
        if thermal != (self.thermal_profile is not None):
            raise ValueError("thermal cargo source and profile must be paired")
        if thermal != bool(self.associated_container_numbers):
            raise ValueError("thermal cargo must name its allocated containers")
        return self


class EquipmentSemanticRealization(BaseModel):
    model_config = _STRICT

    container_number: NonEmptyText
    size_category: NonEmptyText
    type_category: NonEmptyText
    application_code: Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{4}$")]
    active_temperature: bool
    temperature_value_celsius: float | None
    sampling_method: NonEmptyText
    linked_thermal_cargo_groups: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def setpoint_and_operation_match(self) -> EquipmentSemanticRealization:
        if self.active_temperature != (self.temperature_value_celsius is not None):
            raise ValueError("active thermal equipment and setpoint must be paired")
        if self.active_temperature != bool(self.linked_thermal_cargo_groups):
            raise ValueError("active thermal equipment must name linked thermal cargo")
        return self


class TransportAuxiliaryRealization(BaseModel):
    model_config = _STRICT

    imo_number: str | None = None
    imo_generation_attempts: Annotated[int, Field(gt=0)] | None = None
    flag_country_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] | None = None
    flag_country_name: str | None = None

    @model_validator(mode="after")
    def paired_values(self) -> TransportAuxiliaryRealization:
        if (self.imo_number is None) != (self.imo_generation_attempts is None):
            raise ValueError("IMO value and generation attempts must be paired")
        if (self.flag_country_code is None) != (self.flag_country_name is None):
            raise ValueError("flag country code and name must be paired")
        return self


class FlashpointRealization(BaseModel):
    model_config = _STRICT

    cargo_group_id: NonEmptyText
    dangerous_goods_order: Annotated[int, Field(ge=0)]
    eligibility: FlashpointEligibility
    generated: bool
    value_celsius: float | None
    basis: str | None

    @model_validator(mode="after")
    def generated_value_is_paired(self) -> FlashpointRealization:
        if self.generated != (self.value_celsius is not None and self.basis is not None):
            raise ValueError("generated flashpoint value and basis must be paired")
        return self


class SemanticCompletionPlanRow(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    scenario_id: NonEmptyText
    base_document_id: NonEmptyText
    template_id: NonEmptyText
    upstream_target_sha256: Sha256
    target_task: Literal["bill_of_lading_relation_explicit_v5"]
    target: dict[str, JsonValue]
    target_sha256: Sha256
    changes: tuple[ScenarioChange, ...]
    cargo_realizations: tuple[CargoSemanticRealization, ...]
    equipment_realizations: tuple[EquipmentSemanticRealization, ...]
    transport_auxiliary: TransportAuxiliaryRealization
    flashpoint_realizations: tuple[FlashpointRealization, ...]
    upstream_dangerous_goods_realizations: tuple[dict[str, JsonValue], ...]
    remaining_blockers: tuple[NonEmptyText, ...]
    status: Literal["resolved_non_linguistic_v5_pending_text_realization"]
    training_eligible: Literal[False]

    @model_validator(mode="after")
    def target_and_collections_are_canonical(self) -> SemanticCompletionPlanRow:
        if sha256_bytes(canonical_json_bytes(self.target)) != self.target_sha256:
            raise ValueError("completion target SHA-256 differs from payload")
        paths = tuple(row.target_path for row in self.changes)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("completion changes must be unique and sorted")
        if self.remaining_blockers != tuple(sorted(set(self.remaining_blockers))):
            raise ValueError("completion blockers must be unique and sorted")
        expected_containers: dict[str, frozenset[str]] = {}
        for cargo in self.cargo_realizations:
            if cargo.thermal_profile is None:
                continue
            current = frozenset(cargo.associated_container_numbers)
            previous = expected_containers.setdefault(cargo.cargo_group_id, current)
            if previous != current:
                raise ValueError("one thermal cargo group has inconsistent container relations")
        observed_containers: dict[str, set[str]] = defaultdict(set)
        observed_setpoints: dict[str, set[float]] = defaultdict(set)
        for equipment in self.equipment_realizations:
            for group_id in equipment.linked_thermal_cargo_groups:
                observed_containers[group_id].add(equipment.container_number)
                assert equipment.temperature_value_celsius is not None
                observed_setpoints[group_id].add(equipment.temperature_value_celsius)
        if expected_containers != {
            group_id: frozenset(containers) for group_id, containers in observed_containers.items()
        }:
            raise ValueError("thermal cargo and equipment relations differ")
        if any(len(values) != 1 for values in observed_setpoints.values()):
            raise ValueError("one thermal cargo group must use one shared setpoint")
        return self


def _resolve_file(project_root: Path, value: str, expected_sha256: str, *, label: str) -> Path:
    path = resolve_config_path(project_root, value)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} differs from its configured pin: {path}")
    return path.resolve(strict=True)


def _validate_committed_run(project_root: Path, configured: Any) -> Path:
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or sha256_file(commit) != configured.commit_sha256:
        raise ValueError(f"semantic-completion dependency differs from its pin: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    ).validate_committed_run()
    return root.resolve(strict=True)


def _load_dg_plans(path: Path, *, expected_records: int) -> tuple[DangerousGoodsPlanRow, ...]:
    rows: list[DangerousGoodsPlanRow] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"DG plan row {line_number} is blank or unterminated")
            try:
                rows.append(DangerousGoodsPlanRow.model_validate_json(raw, strict=True))
            except ValueError as error:
                raise ValueError(f"DG plan row {line_number} is invalid") from error
    if len(rows) != expected_records:
        raise ValueError("DG plan record count differs from configuration")
    ids = tuple(row.base_document_id for row in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("DG plan contains duplicate source documents")
    return tuple(rows)


def _load_source_targets(
    *, project_root: Path, config: SynthesisSemanticCompletionConfig
) -> dict[str, dict[str, Any]]:
    source_path = _resolve_file(
        project_root,
        config.source.file.path,
        config.source.file.sha256,
        label="source corpus",
    )
    fields = config.source.fields
    output: dict[str, dict[str, Any]] = {}
    with source_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"source corpus row {line_number} is blank or unterminated")
            try:
                value = json.loads(raw)
                document_id = value[fields.document_id]
                target = value[fields.target]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"source corpus row {line_number} is malformed") from error
            if not isinstance(document_id, str) or not isinstance(target, dict):
                raise ValueError(f"source corpus row {line_number} has invalid fields")
            if document_id in output:
                raise ValueError(f"source corpus repeats document ID {document_id}")
            output[document_id] = BILL_OF_LADING_TASK_ADAPTER.validate_target(
                document_id=document_id,
                target=target,
            )
    if len(output) != config.source.file.records:
        raise ValueError("source corpus record count differs from configuration")
    return output


def _source_equipment_support(
    source_targets: Mapping[str, Mapping[str, Any]],
) -> tuple[EquipmentSemanticSupport, tuple[dict[str, Any], ...], set[str]]:
    observations: list[SourceEquipmentObservation] = []
    surface_counts: Counter[tuple[str | None, bool]] = Counter()
    source_imo: set[str] = set()
    for document_id, target in source_targets.items():
        patch = cast(Mapping[str, Any], target["documentPatch"])
        transport = patch.get("transport")
        if isinstance(transport, dict) and isinstance(transport.get("vesselImoNumber"), str):
            source_imo.add(transport["vesselImoNumber"])
        for order, container in enumerate(patch.get("containers") or []):
            printed = container.get("typeDescription")
            temperature_present = container.get("temperatureSetpoint") is not None
            observations.append(
                SourceEquipmentObservation(
                    document_id=document_id,
                    row_id=str(order),
                    printed_surface=printed,
                    temperature_present=temperature_present,
                )
            )
            surface_counts[(printed, temperature_present)] += 1
    support = build_equipment_semantic_support(observations)
    surface_rows = []
    for (printed, temperature_present), count in sorted(
        surface_counts.items(), key=lambda row: (row[0][0] or "", row[0][1])
    ):
        reviewed = review_source_equipment_surface(
            printed,
            temperature_present=temperature_present,
        )
        surface_rows.append(
            {
                "printedSurface": printed,
                "temperaturePresent": temperature_present,
                "occurrences": count,
                "resolution": reviewed.resolution,
                "sizeCategory": reviewed.size_category,
                "typeCategory": reviewed.type_category,
                "thermalOperation": reviewed.thermal_operation,
                "reviewRule": reviewed.review_rule,
            }
        )
    return support, tuple(surface_rows), source_imo


def _load_hs_registry(
    project_root: Path, config: SynthesisSemanticCompletionConfig
) -> UkGlobalTariffRegistry:
    manifest = _resolve_file(
        project_root,
        config.inputs.hs_registry_manifest.path,
        config.inputs.hs_registry_manifest.sha256,
        label="HS source manifest",
    )
    metadata = _resolve_file(
        project_root,
        config.inputs.hs_metadata.path,
        config.inputs.hs_metadata.sha256,
        label="HS metadata",
    )
    report = _resolve_file(
        project_root,
        config.inputs.hs_commodities_report.path,
        config.inputs.hs_commodities_report.sha256,
        label="HS commodities report",
    )
    return compile_uk_global_tariff_registry(
        metadata_path=metadata,
        report_path=report,
        source=load_ukgt_source_pin(manifest),
    )


def _validate_mpci_equipment_vocabulary(path: Path) -> dict[str, str]:
    try:
        value = json.loads(read_regular_file_bytes(path))
        exports = value["exports"]
        codes = exports["ContainerSizeCodes"]
        labels = exports["ContainerSizeTypeLabels"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("MPCI container registry has an unexpected contract") from error
    if (
        not isinstance(codes, list)
        or not all(isinstance(row, str) for row in codes)
        or not isinstance(labels, dict)
        or set(codes) != set(labels)
        or len(codes) != len(set(codes))
    ):
        raise ValueError("MPCI container type codes and labels are inconsistent")
    projected = {
        category: bill_of_lading_v5_module.semantic_container_code(
            "FORTY_FOOT_STANDARD_HEIGHT", category
        )[2:]
        for category in CONTAINER_TYPE_CATEGORIES
    }
    if set(projected.values()) != set(codes):
        raise ValueError("relation-v5 semantic container types differ from MPCI vocabulary")
    return {category: cast(str, labels[code]) for category, code in projected.items()}


def _group_allocations(patch: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for row in patch.get("cargoAllocationGroups") or []:
        group_id = cast(str, row["groupId"])
        values = tuple(
            dict.fromkeys(
                cast(str, allocation["containerNumber"]) for allocation in row["allocations"]
            )
        )
        if group_id in output or not values:
            raise ValueError("cargo allocation groups must be unique and non-empty")
        output[group_id] = values
    return output


def _eligible_thermal_groups(target: Mapping[str, Any]) -> tuple[str, ...]:
    patch = cast(Mapping[str, Any], target["documentPatch"])
    containers = {row["containerNumber"] for row in patch.get("containers") or []}
    allocations = _group_allocations(patch)
    dangerous_groups = {
        cast(str, group["groupId"])
        for group in patch.get("cargoGroups") or []
        if group.get("dangerousGoods")
    }
    dangerous_containers = {
        number for group_id in dangerous_groups for number in allocations.get(group_id, ())
    }
    return tuple(
        cast(str, group["groupId"])
        for group in patch.get("cargoGroups") or []
        if group["groupId"] not in dangerous_groups
        and allocations.get(group["groupId"])
        and set(allocations[group["groupId"]]) <= containers
        and not set(allocations[group["groupId"]]) & dangerous_containers
    )


def _ranked_quota(
    *,
    document_ids: Sequence[str],
    permyriad: int,
    seed: int,
    namespace: str,
    population_size: int | None = None,
) -> frozenset[str]:
    if not 0 <= permyriad <= 10_000:
        raise ValueError("ranked quota must be within [0, 10000]")
    population = len(document_ids) if population_size is None else population_size
    if population < len(document_ids):
        raise ValueError("quota population cannot be smaller than its eligible subset")
    count = (population * permyriad + 5_000) // 10_000
    if count > len(document_ids):
        raise ValueError("configured quota exceeds the eligible document population")
    ranked = sorted(
        document_ids,
        key=lambda document_id: (
            DeterministicStream(
                seed=seed,
                namespace=namespace,
                identity=document_id,
            )
            .derive("quota-rank")
            .bytes(counter=0, length=32)
        ),
    )
    return frozenset(ranked[:count])


def _sample_distinct_identity(
    *,
    support: ThermalGoodsSupport,
    profile: Literal["FROZEN", "CHILLED"] | None,
    stream: DeterministicStream,
    used_hs6: set[str],
) -> ThermalGoodsIdentity | AmbientGoodsIdentity:
    for attempt in range(512):
        attempt_stream = stream.derive(f"attempt-{attempt}")
        value = (
            sample_thermal_goods(support=support, profile=profile, stream=attempt_stream)
            if profile is not None
            else sample_ambient_goods(support=support, stream=attempt_stream)
        )
        if value.hs6 not in used_hs6:
            used_hs6.add(value.hs6)
            return value
    raise RuntimeError("HS semantic identity collision budget exhausted")


def _apply_cargo_semantics(
    *,
    target: dict[str, Any],
    thermal_group_id: str | None,
    support: ThermalGoodsSupport,
    stream: DeterministicStream,
    config: SynthesisSemanticCompletionConfig,
) -> tuple[
    tuple[CargoSemanticRealization, ...],
    dict[str, Literal["FROZEN", "CHILLED"]],
    dict[str, tuple[str, ...]],
]:
    patch = cast(dict[str, Any], target["documentPatch"])
    allocations = _group_allocations(patch)
    profile_by_group: dict[str, Literal["FROZEN", "CHILLED"]] = {}
    if thermal_group_id is not None:
        profile_by_group[thermal_group_id] = sample_thermal_profile(
            weights_permyriad=config.generation.thermal.profile_weights_permyriad,
            stream=stream.derive("selected-thermal-group"),
        )
    output: list[CargoSemanticRealization] = []
    used_hs6: set[str] = set()
    for group_order, group in enumerate(patch.get("cargoGroups") or []):
        if group.get("dangerousGoods"):
            continue
        group_id = cast(str, group["groupId"])
        profile = profile_by_group.get(group_id)
        existing = tuple(cast(Sequence[str], group.get("hsCodes") or ()))
        identity_count = len(existing) or 1
        generated_codes: list[str] = []
        for identity_order in range(identity_count):
            identity = _sample_distinct_identity(
                support=support,
                profile=profile,
                stream=stream.derive(f"group-{group_order}-identity-{identity_order}"),
                used_hs6=used_hs6,
            )
            output_code = None
            if existing:
                output_code = render_synthetic_hs_code(
                    hs6=identity.hs6,
                    output_digits=len(existing[identity_order]),
                    stream=stream.derive(f"group-{group_order}-hs-{identity_order}"),
                )
                generated_codes.append(output_code)
            output.append(
                CargoSemanticRealization(
                    cargo_group_id=group_id,
                    identity_order=identity_order,
                    semantic_source=(
                        "thermal_hs_registry" if profile is not None else "ambient_hs_registry"
                    ),
                    hs6=identity.hs6,
                    output_hs_code=output_code,
                    chapter_description=identity.chapter_description,
                    heading_description=identity.heading_description,
                    description=identity.description,
                    thermal_profile=profile,
                    associated_container_numbers=(allocations[group_id] if profile else ()),
                )
            )
        if existing:
            group["hsCodes"] = generated_codes
    return tuple(output), profile_by_group, allocations


def _apply_equipment_semantics(
    *,
    target: dict[str, Any],
    profile_by_group: Mapping[str, Literal["FROZEN", "CHILLED"]],
    allocations: Mapping[str, tuple[str, ...]],
    support: EquipmentSemanticSupport,
    equipment_registry: Any,
    stream: DeterministicStream,
    config: SynthesisSemanticCompletionConfig,
) -> tuple[EquipmentSemanticRealization, ...]:
    thermal_groups_by_container: dict[str, list[str]] = defaultdict(list)
    for group_id in profile_by_group:
        for container_number in allocations[group_id]:
            thermal_groups_by_container[container_number].append(group_id)
    temperature_by_group = {
        group_id: sample_temperature_setpoint(
            profile=profile,
            stream=stream.derive(f"group-{group_id}-temperature"),
            frozen_minimum_celsius=config.generation.thermal.frozen_minimum_celsius,
            frozen_maximum_celsius=config.generation.thermal.frozen_maximum_celsius,
            chilled_minimum_celsius=config.generation.thermal.chilled_minimum_celsius,
            chilled_maximum_celsius=config.generation.thermal.chilled_maximum_celsius,
            step_celsius=config.generation.thermal.step_celsius,
        )
        for group_id, profile in sorted(profile_by_group.items())
    }
    patch = cast(dict[str, Any], target["documentPatch"])
    output: list[EquipmentSemanticRealization] = []
    for order, container in enumerate(patch.get("containers") or []):
        number = cast(str, container["containerNumber"])
        groups = tuple(sorted(thermal_groups_by_container.get(number, ())))
        active = bool(groups)
        generated = sample_equipment_semantic(
            support=support,
            active_temperature=active,
            stream=stream.derive(f"container-{order}"),
            configured_joint_weights=(
                config.generation.equipment.configured_active_joint_weights
                if active
                else config.generation.equipment.configured_inactive_joint_weights
            ),
        )
        classified = equipment_registry.classify(generated.application_code)
        if classified.type.type_code != generated.application_code[2:]:
            raise RuntimeError("semantic equipment projection differs from BIC registry")
        container.pop("typeDescription", None)
        container["sizeCategory"] = generated.size_category
        container["typeCategory"] = generated.type_category
        temperature = None
        if active:
            profiles = {profile_by_group[group_id] for group_id in groups}
            if len(profiles) != 1:
                raise ValueError("one container cannot receive incompatible thermal profiles")
            group_temperatures = {temperature_by_group[group_id] for group_id in groups}
            if len(group_temperatures) != 1:
                raise ValueError("one container cannot receive conflicting group setpoints")
            temperature = next(iter(group_temperatures))
            container["temperatureSetpoint"] = {
                "value": temperature.value,
                "unit": temperature.unit,
            }
        else:
            container.pop("temperatureSetpoint", None)
        output.append(
            EquipmentSemanticRealization(
                container_number=number,
                size_category=generated.size_category,
                type_category=generated.type_category,
                application_code=generated.application_code,
                active_temperature=active,
                temperature_value_celsius=(temperature.value if temperature else None),
                sampling_method=generated.sampling_method,
                linked_thermal_cargo_groups=groups,
            )
        )
    return tuple(output)


def _transport_allocations(
    *,
    plans: Sequence[DangerousGoodsPlanRow],
    source_imo: set[str],
    config: SynthesisSemanticCompletionConfig,
) -> tuple[dict[str, SyntheticImoNumber], frozenset[str]]:
    eligible_rows: list[str] = []
    for row in plans:
        patch = row.target.get("documentPatch")
        if isinstance(patch, dict) and isinstance(patch.get("transport"), dict):
            eligible_rows.append(row.base_document_id)
    eligible = tuple(eligible_rows)
    imo_requested = _ranked_quota(
        document_ids=eligible,
        permyriad=config.generation.transport.imo_presence_permyriad,
        seed=config.generation.seed,
        namespace=f"{config.run.run_id}-imo-presence",
        population_size=len(plans),
    )
    flags_requested = _ranked_quota(
        document_ids=eligible,
        permyriad=config.generation.transport.flag_presence_permyriad,
        seed=config.generation.seed,
        namespace=f"{config.run.run_id}-flag-presence",
        population_size=len(plans),
    )
    used: set[str] = set()
    allocated: dict[str, SyntheticImoNumber] = {}
    for document_id in sorted(imo_requested):
        sampled = sample_imo_number(
            stream=DeterministicStream(
                seed=config.generation.seed,
                namespace=config.run.run_id,
                identity=document_id,
            ).derive("imo"),
            excluded_values=tuple(sorted(source_imo)),
            used_values=used,
            maximum_attempts=config.generation.transport.maximum_collision_attempts,
            leading_digit_weights=config.generation.transport.imo_leading_digit_weights,
        )
        allocated[document_id] = sampled
        used.add(sampled.value)
    return allocated, flags_requested


def _apply_transport_auxiliary(
    *,
    target: dict[str, Any],
    document_id: str,
    imo_by_document: Mapping[str, SyntheticImoNumber],
    flag_requested: frozenset[str],
    flag_codes: Sequence[str],
    countries: Any,
    stream: DeterministicStream,
) -> TransportAuxiliaryRealization:
    patch = cast(dict[str, Any], target["documentPatch"])
    transport = patch.get("transport")
    if not isinstance(transport, dict):
        return TransportAuxiliaryRealization()
    imo = imo_by_document.get(document_id)
    if imo is None:
        transport.pop("vesselImoNumber", None)
    else:
        transport["vesselImoNumber"] = imo.value
    flag = None
    if document_id in flag_requested:
        flag = sample_vessel_flag(
            country_codes=flag_codes,
            countries=countries,
            stream=stream.derive("flag"),
        )
        transport["vesselFlagCountry"] = flag.printed_country
    else:
        transport.pop("vesselFlagCountry", None)
    if not transport:
        patch.pop("transport")
    return TransportAuxiliaryRealization(
        imo_number=(imo.value if imo is not None else None),
        imo_generation_attempts=(imo.attempts if imo is not None else None),
        flag_country_code=(flag.country_code if flag else None),
        flag_country_name=(flag.printed_country if flag else None),
    )


def _apply_flashpoints(
    *,
    target: dict[str, Any],
    plan: DangerousGoodsPlanRow,
    stream: DeterministicStream,
    config: SynthesisSemanticCompletionConfig,
) -> tuple[FlashpointRealization, ...]:
    realization_by_key = {
        (row.cargo_group_id, row.dangerous_goods_order): row for row in plan.realizations
    }
    output: list[FlashpointRealization] = []
    groups = target["documentPatch"].get("cargoGroups") or []
    for group in groups:
        group_id = cast(str, group["groupId"])
        for order, dangerous in enumerate(group.get("dangerousGoods") or []):
            try:
                upstream = realization_by_key[(group_id, order)]
            except KeyError as error:
                raise ValueError("DG target row lacks proper-shipping-name provenance") from error
            subsidiaries = tuple(dangerous.get("subsidiaryHazardCategories") or ())
            eligibility = classify_flashpoint_eligibility(
                proper_shipping_name=upstream.proper_shipping_name,
                hazard_category=dangerous.get("hazardCategory"),
                subsidiary_hazard_categories=subsidiaries,
            )
            sampled = sample_flashpoint(
                proper_shipping_name=upstream.proper_shipping_name,
                hazard_category=dangerous.get("hazardCategory"),
                subsidiary_hazard_categories=subsidiaries,
                packing_group_category=dangerous.get("packingGroupCategory"),
                stream=stream.derive(f"flashpoint-{group_id}-{order}"),
                presence_permyriad=config.generation.flashpoint.presence_permyriad,
                class3_minimum_celsius=config.generation.flashpoint.class3_minimum_celsius,
                class3_maximum_celsius=config.generation.flashpoint.class3_maximum_celsius,
                non_class3_liquid_minimum_celsius=(
                    config.generation.flashpoint.non_class3_liquid_minimum_celsius
                ),
                non_class3_liquid_maximum_celsius=(
                    config.generation.flashpoint.non_class3_liquid_maximum_celsius
                ),
                desensitized_solid_minimum_celsius=(
                    config.generation.flashpoint.desensitized_solid_minimum_celsius
                ),
                desensitized_solid_maximum_celsius=(
                    config.generation.flashpoint.desensitized_solid_maximum_celsius
                ),
                step_celsius=config.generation.flashpoint.step_celsius,
            )
            dangerous.pop("flashPoint", None)
            if sampled is not None:
                dangerous["flashPoint"] = {
                    "temperature": {"value": sampled.value, "unit": sampled.unit}
                }
            if eligibility is not None:
                output.append(
                    FlashpointRealization(
                        cargo_group_id=group_id,
                        dangerous_goods_order=order,
                        eligibility=eligibility,
                        generated=sampled is not None,
                        value_celsius=(sampled.value if sampled else None),
                        basis=(sampled.basis if sampled else None),
                    )
                )
    return tuple(output)


def _change_ledger(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[ScenarioChange, ...]:
    source_leaves = dict(leaf_items(source))
    target_leaves = dict(leaf_items(target))
    output: list[ScenarioChange] = []
    for path in sorted(source_leaves.keys() | target_leaves.keys()):
        old_present = path in source_leaves
        new_present = path in target_leaves
        if old_present == new_present and source_leaves.get(path) == target_leaves.get(path):
            continue
        role = normalized_role_path(path)
        if role == "schemaVersion":
            kind, rendering, method, coupling = (
                "structural",
                "structural_only",
                "relation_v4_to_v5_schema_migration_v1",
                "schema",
            )
        elif role.endswith((".sizeCategory", ".typeCategory")) and ".containers[]" in role:
            kind, rendering, method, coupling = (
                "categorical",
                "exact_scalar",
                "goods_conditioned_source_type_marginal_size_conditional_v1",
                "container_equipment",
            )
        elif role.endswith(".typeDescription") and ".containers[]" in role:
            kind, rendering, method, coupling = (
                "structural",
                "structural_only",
                "printed_equipment_surface_moved_to_renderer_v1",
                "container_equipment",
            )
        elif ".temperatureSetpoint." in role:
            kind, rendering, method, coupling = (
                "numeric" if role.endswith(".value") else "categorical",
                "formatted_number" if role.endswith(".value") else "exact_scalar",
                "goods_profile_conditioned_temperature_v1",
                "thermal_cargo_equipment",
            )
        elif ".cargoGroups[].hsCodes[]" in role:
            kind, rendering, method, coupling = (
                "categorical",
                "exact_scalar",
                "goods_identity_conditioned_hs_v1",
                "cargo_semantics",
            )
        elif role.endswith(".vesselImoNumber"):
            kind, rendering, method, coupling = (
                "identifier",
                "exact_scalar",
                "random_six_digits_plus_imo_check_digit_v1",
                "transport",
            )
        elif role.endswith(".vesselFlagCountry"):
            kind, rendering, method, coupling = (
                "geography",
                "exact_scalar",
                "uniform_pinned_world_port_country_v1",
                "transport",
            )
        elif ".dangerousGoods[].flashPoint." in role:
            kind, rendering, method, coupling = (
                "numeric" if role.endswith(".value") else "categorical",
                "formatted_number" if role.endswith(".value") else "exact_scalar",
                "hazard_and_physical_form_conditioned_flashpoint_v1",
                "dangerous_goods",
            )
        else:
            raise ValueError(f"semantic completion changed an unowned target path: {path}")
        output.append(
            ScenarioChange.model_validate(
                {
                    "target_path": path,
                    "role_path": role,
                    "stage_id": _STAGE_ID,
                    "change_kind": kind,
                    "rendering_kind": rendering,
                    "old_present": old_present,
                    "old_value": source_leaves.get(path),
                    "new_present": new_present,
                    "new_value": target_leaves.get(path),
                    "method": method,
                    "coupling_group": coupling,
                },
                strict=True,
            )
        )
    return tuple(output)


def run_semantic_completion(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisSemanticCompletionConfig,
) -> dict[str, Any]:
    """Publish one strict, integrated non-linguistic relation-v5 plan."""

    started = time.perf_counter()
    _validate_committed_run(project_root, config.inputs.dangerous_goods_run)
    dg_path = _resolve_file(
        project_root,
        config.inputs.dangerous_goods_plans.path,
        config.inputs.dangerous_goods_plans.sha256,
        label="dangerous-goods plans",
    )
    plans = _load_dg_plans(
        dg_path,
        expected_records=config.inputs.dangerous_goods_plans.records,
    )
    source_targets = _load_source_targets(project_root=project_root, config=config)
    if not {row.base_document_id for row in plans} <= set(source_targets):
        raise ValueError("DG plans reference documents absent from the pinned source corpus")

    equipment_manifest = _resolve_file(
        project_root,
        config.inputs.equipment_registry_manifest.path,
        config.inputs.equipment_registry_manifest.sha256,
        label="BIC equipment registry manifest",
    )
    equipment_registry = load_bic_equipment_registry(equipment_manifest)
    mpci_registry_path = _resolve_file(
        project_root,
        config.inputs.mpci_container_registry.path,
        config.inputs.mpci_container_registry.sha256,
        label="MPCI container registry",
    )
    mpci_labels = _validate_mpci_equipment_vocabulary(mpci_registry_path)
    equipment_support, source_surface_rows, source_imo = _source_equipment_support(source_targets)

    hs_registry = _load_hs_registry(project_root, config)
    thermal_support = build_thermal_goods_support(
        registry=hs_registry,
        ambient_chapters=config.generation.thermal.ambient_hs_chapters,
    )
    iso_path = _resolve_file(
        project_root,
        config.inputs.iso3166_snapshot.path,
        config.inputs.iso3166_snapshot.sha256,
        label="ISO country registry",
    )
    countries = load_iso_country_registry(
        iso_path=iso_path,
        iso_sha256=config.inputs.iso3166_snapshot.sha256,
    )
    world_port_path = _resolve_file(
        project_root,
        config.inputs.world_ports.path,
        config.inputs.world_ports.sha256,
        label="world-port whitelist",
    )
    _resolve_file(
        project_root,
        config.inputs.world_port_registry_receipt.path,
        config.inputs.world_port_registry_receipt.sha256,
        label="world-port registry receipt",
    )
    world_ports = load_pinned_world_port_records(
        world_port_path,
        expected_sha256=config.inputs.world_ports.sha256,
        expected_records=config.inputs.world_ports.records,
    )
    flag_codes = maritime_flag_country_codes(world_ports)

    eligible_groups = {row.base_document_id: _eligible_thermal_groups(row.target) for row in plans}
    thermal_documents = _ranked_quota(
        document_ids=tuple(
            document_id for document_id, groups in eligible_groups.items() if groups
        ),
        permyriad=config.generation.thermal.document_prevalence_permyriad,
        seed=config.generation.seed,
        namespace=f"{config.run.run_id}-thermal",
        population_size=len(plans),
    )
    imo_by_document, flag_requested = _transport_allocations(
        plans=plans,
        source_imo=source_imo,
        config=config,
    )

    source_constraints_path = (
        _validate_committed_run(project_root, config.inputs.dangerous_goods_run)
        / "schema/task-constraints.json"
    )
    source_constraints = RelationExplicitTaskConstraints.model_validate_json(
        read_regular_file_bytes(source_constraints_path), strict=True
    )
    target_task = get_training_task(config.target_task)
    target_schema = target_task.target_model.model_json_schema(mode="serialization")
    target_constraints = RelationExplicitTaskConstraints(
        schemaVersion=1,
        task=config.target_task,
        basePromptSchemaSha256=target_task.base_prompt_schema_sha256(),
        targetSchemaSha256=sha256_bytes(canonical_json_bytes(target_schema)),
        packageRegistrySha256=source_constraints.packageRegistrySha256,
        containerRegistrySha256=config.inputs.mpci_container_registry.sha256,
        packageCategoryTokens=source_constraints.packageCategoryTokens,
        containerCategoryTokens=tuple(sorted(CONTAINER_TYPE_CATEGORIES)),
    )
    bound_target_task = target_task.bind_constraints(target_constraints)

    output_rows: list[SemanticCompletionPlanRow] = []
    equipment_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    thermal_profile_counts: Counter[str] = Counter()
    flashpoint_counts: Counter[str] = Counter()
    for plan in plans:
        document_id = plan.base_document_id
        stream = DeterministicStream(
            seed=config.generation.seed,
            namespace=config.run.run_id,
            identity=document_id,
        )
        target = migrate_relation_v4_target_to_v5(plan.target)
        thermal_group = None
        if document_id in thermal_documents:
            groups = eligible_groups[document_id]
            thermal_group = groups[stream.derive("thermal-group").randbelow(len(groups))]
        cargo_rows, profiles, allocations = _apply_cargo_semantics(
            target=target,
            thermal_group_id=thermal_group,
            support=thermal_support,
            stream=stream.derive("cargo"),
            config=config,
        )
        equipment_rows = _apply_equipment_semantics(
            target=target,
            profile_by_group=profiles,
            allocations=allocations,
            support=equipment_support,
            equipment_registry=equipment_registry,
            stream=stream.derive("equipment"),
            config=config,
        )
        transport = _apply_transport_auxiliary(
            target=target,
            document_id=document_id,
            imo_by_document=imo_by_document,
            flag_requested=flag_requested,
            flag_codes=flag_codes,
            countries=countries,
            stream=stream.derive("transport"),
        )
        flashpoints = _apply_flashpoints(
            target=target,
            plan=plan,
            stream=stream.derive("dangerous-goods"),
            config=config,
        )
        canonical = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
            document_id=plan.scenario_id,
            target=target,
        )
        if bound_target_task.canonicalize(canonical) != canonical:
            raise RuntimeError("bound relation-v5 task changed a canonical completion target")
        blockers = set(plan.remaining_blockers)
        blockers.discard("container_type_task_schema_projection")
        blockers.discard("vessel_imo_requires_authoritative_assigned_number_registry")
        if transport.imo_number is not None or transport.flag_country_name is not None:
            blockers.add("transport_auxiliary_printed_surface_realization")
        row = SemanticCompletionPlanRow(
            schema_version=1,
            scenario_id=plan.scenario_id,
            base_document_id=document_id,
            template_id=plan.template_id,
            upstream_target_sha256=plan.target_sha256,
            target_task=config.target_task,
            target=canonical,
            target_sha256=sha256_bytes(canonical_json_bytes(canonical)),
            changes=_change_ledger(plan.target, canonical),
            cargo_realizations=cargo_rows,
            equipment_realizations=equipment_rows,
            transport_auxiliary=transport,
            flashpoint_realizations=flashpoints,
            upstream_dangerous_goods_realizations=tuple(
                cast(dict[str, JsonValue], value.model_dump(mode="json"))
                for value in plan.realizations
            ),
            remaining_blockers=tuple(sorted(blockers)),
            status="resolved_non_linguistic_v5_pending_text_realization",
            training_eligible=False,
        )
        output_rows.append(row)
        for equipment_value in equipment_rows:
            equipment_counts[equipment_value.type_category] += 1
            size_counts[equipment_value.size_category] += 1
        thermal_profile_counts.update(profiles.values())
        for flashpoint_value in flashpoints:
            outcome = "generated" if flashpoint_value.generated else "omitted"
            flashpoint_counts[f"{flashpoint_value.eligibility}:{outcome}"] += 1

    plan_payload = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in output_rows
    )
    target_payload = b"".join(
        canonical_json_bytes(
            {
                "scenarioId": row.scenario_id,
                "baseDocumentId": row.base_document_id,
                "target": row.target,
                "targetSha256": row.target_sha256,
            }
        )
        + b"\n"
        for row in output_rows
    )
    source_surface_payload = b"".join(
        canonical_json_bytes(value) + b"\n" for value in source_surface_rows
    )
    source_joint_distribution = [
        {
            "sizeCategory": row.size_category,
            "typeCategory": row.type_category,
            "activeTemperature": row.active_temperature,
            "occurrences": row.occurrences,
            "documentCount": row.document_count,
        }
        for row in equipment_support.rows
    ]
    source_type_distribution = [
        {
            "typeCategory": row.type_category,
            "activeTemperature": row.active_temperature,
            "occurrences": row.occurrences,
            "documentCount": row.document_count,
        }
        for row in equipment_support.type_rows
    ]
    summary = {
        "schemaVersion": 1,
        "selectedDocuments": len(output_rows),
        "strictV5SchemaValidTargets": len(output_rows),
        "relationalInverseValidTargets": len(output_rows),
        "trainingRecordsPublished": 0,
        "sourceEquipmentAudit": {
            "inputRows": equipment_support.audit.input_rows,
            "typeResolvedRows": equipment_support.audit.type_resolved_rows,
            "resolvedRows": equipment_support.audit.resolved_rows,
            "unresolvedRows": equipment_support.audit.unresolved_rows,
            "temperatureRows": equipment_support.audit.temperature_rows,
            "typeResolvedTemperatureRows": (equipment_support.audit.type_resolved_temperature_rows),
            "resolvedTemperatureRows": equipment_support.audit.resolved_temperature_rows,
            "nonOperatingReeferRows": equipment_support.audit.non_operating_reefer_rows,
            "typeSupportRows": equipment_support.audit.type_support_rows,
            "jointSupportRows": equipment_support.audit.joint_support_rows,
            "unresolvedSurfaces": dict(equipment_support.unresolved_surfaces),
            "typeDistribution": source_type_distribution,
            "jointDistribution": source_joint_distribution,
        },
        "thermalGoodsSupport": {
            "frozenHs6": len(thermal_support.frozen),
            "chilledHs6": len(thermal_support.chilled),
            "ambientHs6": len(thermal_support.ambient),
        },
        "thermalEligibleDocuments": sum(bool(value) for value in eligible_groups.values()),
        "thermalDocuments": len(thermal_documents),
        "thermalProfileCounts": dict(sorted(thermal_profile_counts.items())),
        "generatedContainerTypeCounts": dict(sorted(equipment_counts.items())),
        "generatedContainerSizeCounts": dict(sorted(size_counts.items())),
        "generatedImoNumbers": len(imo_by_document),
        "generatedVesselFlags": len(flag_requested),
        "maritimeFlagCountrySupport": len(flag_codes),
        "flashpointDecisions": dict(sorted(flashpoint_counts.items())),
        "mpciContainerTypeLabels": mpci_labels,
    }
    target_schema_payload = json_artifact_bytes(target_schema)
    task_constraints_payload = (
        canonical_json_bytes(target_constraints.model_dump(mode="json")) + b"\n"
    )
    prompt_schema_payload = bound_target_task.prompt_schema_json().encode("utf-8") + b"\n"
    summary_payload = json_artifact_bytes(summary)
    resolved_equipment = equipment_support.audit.resolved_rows
    input_equipment = equipment_support.audit.input_rows
    resolved_temperature = equipment_support.audit.resolved_temperature_rows
    input_temperature = equipment_support.audit.temperature_rows
    report = f"""# Integrated non-linguistic relation-v5 semantic completion

- Relation-v5 targets: **{len(output_rows):,}**
- Source equipment rows resolved: **{resolved_equipment:,} / {input_equipment:,}**
- Source temperature rows resolved: **{resolved_temperature:,} / {input_temperature:,}**
- Thermal documents generated: **{len(thermal_documents):,}**
- IMO values / vessel flags generated: **{len(imo_by_document):,} / {len(flag_requested):,}**
- Strict schema and relational inverse passes: **{len(output_rows):,} / {len(output_rows):,}**

The stage samples cargo semantics before refrigeration and equipment.  Generated equipment
uses readable size/type labels in the model target and projects deterministically to an MPCI/BIC
four-character code retained in provenance.  Sparse IMO values are random checksum-valid
identifiers; vessel flags are sampled independently from countries represented in the pinned
world-port whitelist because the source audit found no supported route/party correlation.
Flashpoints are noisy formulation properties conditioned on hazard and explicit physical-form
semantics, never fixed lookups by UN number.

No training text is published.  Party/cargo wording, printed equipment/category surfaces, and
raw-OCR patching remain explicit blockers for the linguistic realization stage.
"""
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "configSha256": sha256_file(config_path),
                "sourceSha256": config.source.file.sha256,
                "dgPlansSha256": config.inputs.dangerous_goods_plans.sha256,
                "equipmentRegistrySha256": config.inputs.equipment_registry_manifest.sha256,
                "mpciContainerRegistrySha256": config.inputs.mpci_container_registry.sha256,
                "hsRegistrySha256": config.inputs.hs_registry_manifest.sha256,
                "worldPortsSha256": config.inputs.world_ports.sha256,
                "plansSha256": sha256_bytes(plan_payload),
                "targetsSha256": sha256_bytes(target_payload),
                "targetSchemaSha256": sha256_bytes(target_schema_payload),
                "taskConstraintsSha256": sha256_bytes(task_constraints_payload),
                "promptSchemaSha256": sha256_bytes(prompt_schema_payload),
                "implementationSha256": sha256_file(Path(__file__)),
                "containerImplementationSha256": sha256_file(
                    Path(container_semantics_module.__file__)
                ),
                "thermalImplementationSha256": sha256_file(Path(thermal_goods_module.__file__)),
                "transportImplementationSha256": sha256_file(
                    Path(transport_auxiliary_module.__file__)
                ),
                "flashpointImplementationSha256": sha256_file(
                    Path(flashpoint_scenarios_module.__file__)
                ),
                "schemaImplementationSha256": sha256_file(Path(bill_of_lading_v5_module.__file__)),
                "taskAdapterImplementationSha256": sha256_file(Path(task_adapter_module.__file__)),
                "trainingTaskImplementationSha256": sha256_file(
                    Path(training_tasks_module.__file__)
                ),
            }
        )
    )
    stage = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=transaction,
    )
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    stage.publish_bytes("generation/completion-plans.jsonl", plan_payload)
    stage.publish_bytes("generation/v5-targets.jsonl", target_payload)
    stage.publish_bytes("generation/summary.json", summary_payload)
    stage.publish_bytes("audit/source-equipment-surfaces.jsonl", source_surface_payload)
    stage.publish_bytes("schema/relation-v5.schema.json", target_schema_payload)
    stage.publish_bytes("schema/task-constraints.json", task_constraints_payload)
    stage.publish_bytes("schema/prompt-schema.json", prompt_schema_payload)
    stage.publish_bytes("REPORT.md", report.encode("utf-8"))
    committed = stage.commit(
        expected_artifacts=(
            "REPORT.md",
            "audit/source-equipment-surfaces.jsonl",
            "config.yaml",
            "generation/completion-plans.jsonl",
            "generation/summary.json",
            "generation/v5-targets.jsonl",
            "schema/prompt-schema.json",
            "schema/relation-v5.schema.json",
            "schema/task-constraints.json",
        ),
        metadata={
            "schema_version": 1,
            "documents": len(output_rows),
            "strict_v5_targets": len(output_rows),
            "thermal_documents": len(thermal_documents),
        },
    )
    elapsed = time.perf_counter() - started
    return {
        "output_dir": str(stage.final_root),
        "created": committed.created,
        "runtimeSeconds": round(elapsed, 6),
        "peakRssMiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
        **summary,
    }
