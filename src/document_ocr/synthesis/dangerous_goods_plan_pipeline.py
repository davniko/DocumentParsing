"""Apply coherent non-linguistic DG semantics to a composed synthesis plan."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas import bill_of_lading_v4 as bill_of_lading_v4_module
from document_ocr.label_schemas.bill_of_lading_v4 import migrate_relation_v3_target_to_v4
from document_ocr.synthesis import bill_of_lading_domain as bill_of_lading_domain_module
from document_ocr.synthesis import dangerous_goods_registry as dangerous_goods_registry_module
from document_ocr.synthesis import task_adapter as task_adapter_module
from document_ocr.synthesis.anchors import leaf_items, normalized_role_path
from document_ocr.synthesis.config import SynthesisDangerousGoodsPlanConfig
from document_ocr.synthesis.dangerous_goods_registry import (
    DangerousGoodsRegistryReceipt,
    LoadedDangerousGoodsRegistry,
    SampledDangerousGoods,
    load_dangerous_goods_registry,
)
from document_ocr.synthesis.flashpoint_scenarios import classify_flashpoint_eligibility
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.raw_text_template import printed_topology_mismatches
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.scenario_state import ScenarioChange, ScenarioState
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_V4_TASK_ADAPTER
from document_ocr.training import tasks as training_tasks_module
from document_ocr.training.config import resolve_config_path
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_DG_STAGE_ID = "dangerous-goods-regulatory-semantics-v1"
_DG_METHOD = "phmsa_atomic_tuple_with_optional_exact_identity_ecics_hs6_v2"


class DangerousGoodsRealization(BaseModel):
    model_config = _STRICT

    cargo_group_id: NonEmptyText
    dangerous_goods_order: Annotated[int, Field(ge=0)]
    branch: Literal["general_regulatory_tuple", "hs_linked_exact_chemical"]
    hmt_record_id: NonEmptyText
    hmt_source_row: Annotated[int, Field(ge=2)]
    un_number: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]
    proper_shipping_name: NonEmptyText
    exact_hazard_class: NonEmptyText
    exact_subsidiary_hazards: tuple[str, ...]
    semantic_hazard_category: NonEmptyText
    semantic_subsidiary_hazard_categories: tuple[str, ...]
    packing_group_code: str | None = None
    packing_group_category: str | None = None
    technical_name_required: bool
    nos_entry: bool
    symbols: tuple[str, ...]
    vessel_stowage_location: str | None = None
    generated_hs_codes: tuple[str, ...]
    ecics_cus_number: str | None = None
    ecics_cn_code: str | None = None
    ecics_chemical_name: str | None = None
    ecics_cas_numbers: tuple[str, ...]
    flashpoint_disposition: Literal["omitted_no_formulation_property_source"]

    @model_validator(mode="after")
    def exact_branch_fields_are_paired(self) -> DangerousGoodsRealization:
        exact = self.branch == "hs_linked_exact_chemical"
        if exact != bool(self.generated_hs_codes):
            raise ValueError("DG exact-chemical branch and generated HS codes differ")
        if exact != all(
            value is not None
            for value in (self.ecics_cus_number, self.ecics_cn_code, self.ecics_chemical_name)
        ):
            raise ValueError("DG exact-chemical branch and ECICS provenance differ")
        return self


class DangerousGoodsPlanRow(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    scenario_id: NonEmptyText
    base_document_id: NonEmptyText
    template_id: NonEmptyText
    upstream_state_sha256: Sha256
    target_task: Literal["bill_of_lading_relation_explicit_v4"]
    target: dict[str, JsonValue]
    target_sha256: Sha256
    changes: tuple[ScenarioChange, ...]
    realizations: tuple[DangerousGoodsRealization, ...]
    remaining_blockers: tuple[NonEmptyText, ...]
    status: Literal["resolved_non_linguistic_v4_pending_text_realization"]
    training_eligible: Literal[False]

    @model_validator(mode="after")
    def target_and_changes_are_canonical(self) -> DangerousGoodsPlanRow:
        if sha256_bytes(canonical_json_bytes(self.target)) != self.target_sha256:
            raise ValueError("DG plan target SHA-256 differs from payload")
        paths = tuple(row.target_path for row in self.changes)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("DG plan change paths must be unique and sorted")
        if self.remaining_blockers != tuple(sorted(set(self.remaining_blockers))):
            raise ValueError("DG plan blockers must be unique and sorted")
        return self


def _resolve_file(project_root: Path, path_value: str, expected_sha256: str) -> Path:
    path = resolve_config_path(project_root, path_value)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"DG plan input differs from its pin: {path}")
    return path.resolve(strict=True)


def _validate_committed_run(
    project_root: Path,
    configured: Any,
    *,
    require_manifest: bool,
) -> Path:
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or sha256_file(commit) != configured.commit_sha256:
        raise ValueError(f"DG plan committed dependency differs from its pin: {root}")
    if require_manifest and sha256_file(root / "manifest.json") != configured.manifest_sha256:
        raise ValueError(f"DG plan dependency manifest differs from its pin: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    ).validate_committed_run()
    return root.resolve(strict=True)


def _load_semantic_plans(
    path: Path,
    *,
    expected_records: int,
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise ValueError(f"semantic plan has a blank row at {line_number}")
            try:
                row = json.loads(raw)
                state = ScenarioState.model_validate_json(
                    canonical_json_bytes(row["scenario_state"]), strict=True
                )
            except Exception as error:
                raise ValueError(f"semantic plan row {line_number} is invalid") from error
            state.assert_integrity()
            if (
                row.get("state_sha256") != state.content_sha256()
                or row.get("target") != state.target
                or row.get("target_sha256") != state.target_sha256
            ):
                raise ValueError(f"semantic plan row {line_number} differs from its scenario state")
            output.append(row)
    if len(output) != expected_records:
        raise ValueError("semantic plan record count differs from configuration")
    ids = tuple(cast(str, row["scenario_id"]) for row in output)
    if len(ids) != len(set(ids)):
        raise ValueError("semantic plan scenario IDs are not unique")
    return tuple(output)


def _realization(
    *,
    group_id: str,
    order: int,
    sampled: SampledDangerousGoods,
) -> DangerousGoodsRealization:
    hmt = sampled.hmt_record
    ecics = sampled.ecics_link
    return DangerousGoodsRealization(
        cargo_group_id=group_id,
        dangerous_goods_order=order,
        branch=sampled.method,
        hmt_record_id=hmt.record_id,
        hmt_source_row=hmt.source_row,
        un_number=hmt.un_number,
        proper_shipping_name=hmt.proper_shipping_name,
        exact_hazard_class=hmt.exact_hazard_class,
        exact_subsidiary_hazards=hmt.exact_subsidiary_hazards,
        semantic_hazard_category=hmt.hazard_category,
        semantic_subsidiary_hazard_categories=hmt.subsidiary_hazard_categories,
        packing_group_code=hmt.packing_group_code,
        packing_group_category=hmt.packing_group_category,
        technical_name_required=hmt.technical_name_required,
        nos_entry=hmt.nos_entry,
        symbols=hmt.symbols,
        vessel_stowage_location=hmt.vessel_stowage_location,
        generated_hs_codes=sampled.hs_codes,
        ecics_cus_number=ecics.cus_number if ecics is not None else None,
        ecics_cn_code=ecics.cn_code if ecics is not None else None,
        ecics_chemical_name=ecics.name if ecics is not None else None,
        ecics_cas_numbers=ecics.cas_numbers if ecics is not None else (),
        flashpoint_disposition="omitted_no_formulation_property_source",
    )


def _sample_unique(
    registry: LoadedDangerousGoodsRegistry,
    *,
    stream: DeterministicStream,
    branch: Literal["general_regulatory_tuple", "hs_linked_exact_chemical"],
    category_weights: Mapping[Any, int],
    maximum_subsidiary_hazards: int,
    maximum_attempts: int,
    used_signatures: set[tuple[str, tuple[str, ...]]],
    accepts: Callable[[SampledDangerousGoods], bool] | None = None,
) -> SampledDangerousGoods:
    for attempt in range(maximum_attempts):
        sampled = registry.sample(
            stream=stream.derive(f"attempt-{attempt}"),
            method=branch,
            category_weights=cast(Any, category_weights),
            maximum_subsidiary_hazards=maximum_subsidiary_hazards,
        )
        signature = (sampled.hmt_record.record_id, sampled.hs_codes)
        if signature not in used_signatures and (accepts is None or accepts(sampled)):
            used_signatures.add(signature)
            return sampled
    raise RuntimeError("DG tuple collision budget exhausted")


def _source_dg_shape(value: Mapping[str, Any]) -> tuple[int, bool, bool]:
    """Return the printed optional shape that a sampled DG tuple must fit.

    The UN number and primary hazard are independently projected because a
    task label may omit either even though the regulatory tuple retains both
    internally.  Subsidiary hazards and packing group affect which facts must
    be printed and therefore constrain tuple selection.
    """

    subsidiaries = value.get("subsidiaryHazardCategories") or ()
    return (
        len(cast(list[Any], subsidiaries)),
        "packingGroupCategory" in value,
        "flashPoint" in value,
    )


def _sample_matches_source_dg_shape(
    sampled: SampledDangerousGoods,
    *,
    source: Mapping[str, Any],
) -> bool:
    subsidiary_count, packing_present, flashpoint_present = _source_dg_shape(source)
    candidate = sampled.target.model_dump(mode="json", exclude_none=True)
    if len(candidate.get("subsidiaryHazardCategories") or ()) != subsidiary_count:
        return False
    if ("packingGroupCategory" in candidate) != packing_present:
        return False
    if flashpoint_present:
        return (
            classify_flashpoint_eligibility(
                proper_shipping_name=sampled.hmt_record.proper_shipping_name,
                hazard_category=sampled.hmt_record.hazard_category,
                subsidiary_hazard_categories=(
                    sampled.hmt_record.subsidiary_hazard_categories
                ),
            )
            is not None
        )
    return True


def _project_sampled_dg_to_source_shape(
    sampled: SampledDangerousGoods,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a complete sampled tuple onto exactly the template's fields."""

    projected = sampled.target.model_dump(mode="json", exclude_none=True)
    for field in (
        "unNumber",
        "hazardCategory",
        "subsidiaryHazardCategories",
        "packingGroupCategory",
    ):
        if field not in source:
            projected.pop(field, None)
    if "flashPoint" in source:
        # Flashpoint values are sampled in the following semantic-completion
        # stage.  Retaining the source-shaped placeholder here prevents an
        # intermediate structural delete/re-add cycle.
        projected["flashPoint"] = source["flashPoint"]
    return projected


def _branch_category_weights(
    registry: LoadedDangerousGoodsRegistry,
    *,
    branch: Literal["general_regulatory_tuple", "hs_linked_exact_chemical"],
    configured: Mapping[Any, int],
    maximum_subsidiary_hazards: int,
) -> dict[Any, int]:
    if branch == "general_regulatory_tuple":
        supported = {
            row.hazard_category
            for row in registry.hmt_records
            if row.maritime_eligible
            and len(row.exact_subsidiary_hazards) <= maximum_subsidiary_hazards
        }
    else:
        hmt_by_id = {row.record_id: row for row in registry.hmt_records}
        supported = {
            hmt_by_id[cast(str, link.hmt_record_id)].hazard_category
            for link in registry.ecics_links
            if link.disposition == "eligible_unique_maritime_hmt"
            and len(hmt_by_id[cast(str, link.hmt_record_id)].exact_subsidiary_hazards)
            <= maximum_subsidiary_hazards
        }
    weights = {category: weight for category, weight in configured.items() if category in supported}
    if set(weights) != supported or not weights:
        missing = sorted(supported - set(weights))
        raise RuntimeError(f"configured DG weights do not cover branch support: {missing}")
    return weights


def _apply_dg(
    *,
    source_v3: Mapping[str, Any],
    registry: LoadedDangerousGoodsRegistry,
    stream: DeterministicStream,
    config: SynthesisDangerousGoodsPlanConfig,
) -> tuple[dict[str, Any], tuple[DangerousGoodsRealization, ...]]:
    target = migrate_relation_v3_target_to_v4(source_v3)
    groups = target.get("documentPatch", {}).get("cargoGroups", [])
    realizations: list[DangerousGoodsRealization] = []
    for group_order, group in enumerate(groups):
        source_dangerous = group.get("dangerousGoods") or []
        if not source_dangerous:
            continue
        branch: Literal["general_regulatory_tuple", "hs_linked_exact_chemical"] = (
            "hs_linked_exact_chemical" if group.get("hsCodes") else "general_regulatory_tuple"
        )
        used_signatures: set[tuple[str, tuple[str, ...]]] = set()
        branch_weights = _branch_category_weights(
            registry,
            branch=branch,
            configured=config.generation.category_weights_permyriad,
            maximum_subsidiary_hazards=config.generation.maximum_subsidiary_hazards,
        )
        generated_hs: list[str] = []
        generated_dangerous: list[dict[str, Any]] = []
        group_id = cast(str, group["groupId"])
        for dangerous_order, source_dangerous_row in enumerate(source_dangerous):
            preserve_shape = (
                config.generation.printed_topology_policy
                == "preserve_selected_template_v1"
            )
            source_shape = cast(Mapping[str, Any], source_dangerous_row)

            def accepts_shape(
                candidate: SampledDangerousGoods,
                source: Mapping[str, Any] = source_shape,
            ) -> bool:
                return _sample_matches_source_dg_shape(candidate, source=source)

            sampled = _sample_unique(
                registry,
                stream=stream.derive(f"group-{group_order}-dg-{dangerous_order}"),
                branch=branch,
                category_weights=branch_weights,
                maximum_subsidiary_hazards=config.generation.maximum_subsidiary_hazards,
                maximum_attempts=config.generation.maximum_collision_attempts,
                used_signatures=used_signatures,
                accepts=accepts_shape if preserve_shape else None,
            )
            generated_dangerous.append(
                _project_sampled_dg_to_source_shape(
                    sampled,
                    source=source_shape,
                )
                if preserve_shape
                else sampled.target.model_dump(mode="json", exclude_none=True)
            )
            generated_hs.extend(sampled.hs_codes)
            realizations.append(
                _realization(
                    group_id=group_id,
                    order=dangerous_order,
                    sampled=sampled,
                )
            )
        group["dangerousGoods"] = generated_dangerous
        if branch == "hs_linked_exact_chemical":
            if len(generated_hs) != len(set(generated_hs)):
                raise RuntimeError("DG exact-chemical branch produced duplicate HS codes")
            group["hsCodes"] = generated_hs
        elif group.get("hsCodes"):
            raise RuntimeError("DG general branch cannot retain independently sampled HS codes")
    canonical = BILL_OF_LADING_V4_TASK_ADAPTER.validate_target(
        document_id=stream.identity,
        target=target,
    )
    if config.generation.printed_topology_policy == "preserve_selected_template_v1":
        source_v4 = migrate_relation_v3_target_to_v4(source_v3)
        mismatches = printed_topology_mismatches(source_v4, canonical)
        if mismatches:
            detail = ", ".join(row.path for row in mismatches[:8])
            raise RuntimeError(f"DG generation changed printed template topology: {detail}")
    return canonical, tuple(realizations)


def _change_ledger(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[ScenarioChange, ...]:
    source_leaves = dict(leaf_items(source))
    target_leaves = dict(leaf_items(target))
    changes: list[ScenarioChange] = []
    for path in sorted(source_leaves.keys() | target_leaves.keys()):
        old_present = path in source_leaves
        new_present = path in target_leaves
        if old_present == new_present and source_leaves.get(path) == target_leaves.get(path):
            continue
        is_dg = ".dangerousGoods[" in path or ".hsCodes[" in path
        changes.append(
            ScenarioChange.model_validate(
                {
                    "target_path": path,
                    "role_path": normalized_role_path(path),
                    "stage_id": _DG_STAGE_ID,
                    "change_kind": "structural" if old_present != new_present else "categorical",
                    "rendering_kind": (
                        "structural_only" if old_present != new_present else "exact_scalar"
                    ),
                    "old_present": old_present,
                    "old_value": source_leaves.get(path),
                    "new_present": new_present,
                    "new_value": target_leaves.get(path),
                    "method": _DG_METHOD if is_dg else "relation_v3_to_v4_schema_migration_v1",
                    "coupling_group": "dangerous_goods_semantics" if is_dg else "schema",
                },
                strict=True,
            )
        )
    return tuple(changes)


def _sampling_validation(
    *,
    registry: LoadedDangerousGoodsRegistry,
    config: SynthesisDangerousGoodsPlanConfig,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    draws = config.generation.sampling_validation_draws_per_branch
    for branch in ("general_regulatory_tuple", "hs_linked_exact_chemical"):
        weights = _branch_category_weights(
            registry,
            branch=branch,
            configured=config.generation.category_weights_permyriad,
            maximum_subsidiary_hazards=config.generation.maximum_subsidiary_hazards,
        )
        total_weight = sum(weights.values())
        counts: Counter[str] = Counter()
        un_numbers: set[str] = set()
        hs_codes: set[str] = set()
        for index in range(draws):
            sampled = registry.sample(
                stream=DeterministicStream(
                    seed=config.generation.seed,
                    namespace=f"{config.generation.sampling_namespace}-distribution-validation",
                    identity=f"{branch}-{index}",
                ),
                method=branch,
                category_weights=cast(Any, weights),
                maximum_subsidiary_hazards=(config.generation.maximum_subsidiary_hazards),
            )
            counts[sampled.hmt_record.hazard_category] += 1
            un_numbers.add(sampled.hmt_record.un_number)
            hs_codes.update(sampled.hs_codes)
        deviations = {
            category: abs(counts[category] / draws - weight / total_weight)
            for category, weight in weights.items()
        }
        maximum_deviation_permyriad = round(max(deviations.values()) * 10_000)
        if maximum_deviation_permyriad > config.generation.maximum_category_deviation_permyriad:
            raise RuntimeError(
                f"DG {branch} sampling distribution exceeds its configured tolerance"
            )
        output[branch] = {
            "draws": draws,
            "categoryCounts": dict(sorted(counts.items())),
            "maximumCategoryDeviationPermyriad": maximum_deviation_permyriad,
            "distinctUnNumbers": len(un_numbers),
            "distinctHs6": len(hs_codes),
            "hsInvariant": (
                "always_absent" if branch == "general_regulatory_tuple" else "always_present"
            ),
        }
    return output


def run_dangerous_goods_plan(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisDangerousGoodsPlanConfig,
) -> dict[str, Any]:
    """Produce one integrated relation-v4 semantic target for every upstream plan."""

    _validate_committed_run(
        project_root,
        config.inputs.semantic_plan_run,
        require_manifest=True,
    )
    _validate_committed_run(project_root, config.inputs.registry.run, require_manifest=False)
    semantic_path = _resolve_file(
        project_root,
        config.inputs.semantic_plans.path,
        config.inputs.semantic_plans.sha256,
    )
    plans = _load_semantic_plans(
        semantic_path,
        expected_records=config.inputs.semantic_plans.records,
    )
    source_constraints_path = _resolve_file(
        project_root,
        config.inputs.source_task_constraints.path,
        config.inputs.source_task_constraints.sha256,
    )
    source_constraints = RelationExplicitTaskConstraints.model_validate_json(
        read_regular_file_bytes(source_constraints_path), strict=True
    )
    if source_constraints.task != config.source_task:
        raise ValueError("DG source constraints name a different source task")
    target_task = get_training_task(config.target_task)
    target_schema = target_task.target_model.model_json_schema(mode="serialization")
    target_constraints = RelationExplicitTaskConstraints(
        schemaVersion=1,
        task=config.target_task,
        basePromptSchemaSha256=target_task.base_prompt_schema_sha256(),
        targetSchemaSha256=sha256_bytes(canonical_json_bytes(target_schema)),
        packageRegistrySha256=source_constraints.packageRegistrySha256,
        containerRegistrySha256=source_constraints.containerRegistrySha256,
        packageCategoryTokens=source_constraints.packageCategoryTokens,
        containerCategoryTokens=source_constraints.containerCategoryTokens,
    )
    bound_target_task = target_task.bind_constraints(target_constraints)
    target_schema_payload = json_artifact_bytes(target_schema)
    task_constraints_payload = (
        canonical_json_bytes(target_constraints.model_dump(mode="json")) + b"\n"
    )
    prompt_schema_payload = bound_target_task.prompt_schema_json().encode("utf-8") + b"\n"
    receipt_path = _resolve_file(
        project_root,
        config.inputs.registry.receipt.path,
        config.inputs.registry.receipt.sha256,
    )
    receipt = DangerousGoodsRegistryReceipt.model_validate_json(
        read_regular_file_bytes(receipt_path), strict=True
    )
    hmt_path = _resolve_file(
        project_root,
        config.inputs.registry.hmt_records.path,
        config.inputs.registry.hmt_records.sha256,
    )
    ecics_path = _resolve_file(
        project_root,
        config.inputs.registry.ecics_links.path,
        config.inputs.registry.ecics_links.sha256,
    )
    registry = load_dangerous_goods_registry(
        hmt_path=hmt_path,
        hmt_sha256=config.inputs.registry.hmt_records.sha256,
        ecics_path=ecics_path,
        ecics_sha256=config.inputs.registry.ecics_links.sha256,
    )
    if (
        len(registry.hmt_records) != config.inputs.registry.hmt_records.records
        or len(registry.ecics_links) != config.inputs.registry.ecics_links.records
    ):
        raise ValueError("DG plan registry record counts differ from configuration")
    branch_support = {
        branch: tuple(
            sorted(
                _branch_category_weights(
                    registry,
                    branch=cast(Any, branch),
                    configured=config.generation.category_weights_permyriad,
                    maximum_subsidiary_hazards=(config.generation.maximum_subsidiary_hazards),
                )
            )
        )
        for branch in ("general_regulatory_tuple", "hs_linked_exact_chemical")
    }

    output_rows: list[DangerousGoodsPlanRow] = []
    branch_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    generated_un: set[str] = set()
    generated_hs: set[str] = set()
    for plan in plans:
        document_id = cast(str, plan["base_document_id"])
        stream = DeterministicStream(
            seed=config.generation.seed,
            namespace=config.generation.sampling_namespace,
            identity=document_id,
        )
        target, realizations = _apply_dg(
            source_v3=cast(dict[str, Any], plan["target"]),
            registry=registry,
            stream=stream,
            config=config,
        )
        blockers = set(cast(list[str], plan["remaining_blockers"]))
        blockers.discard("dangerous_goods_requires_goods_first_semantic_realization")
        if realizations:
            blockers.add("dangerous_goods_printed_text_realization")
        row = DangerousGoodsPlanRow(
            schema_version=1,
            scenario_id=cast(str, plan["scenario_id"]),
            base_document_id=document_id,
            template_id=cast(str, plan["template_id"]),
            upstream_state_sha256=cast(str, plan["state_sha256"]),
            target_task=config.target_task,
            target=target,
            target_sha256=sha256_bytes(canonical_json_bytes(target)),
            changes=_change_ledger(cast(dict[str, Any], plan["target"]), target),
            realizations=realizations,
            remaining_blockers=tuple(sorted(blockers)),
            status="resolved_non_linguistic_v4_pending_text_realization",
            training_eligible=False,
        )
        output_rows.append(row)
        for realization in realizations:
            branch_counts[realization.branch] += 1
            category_counts[realization.semantic_hazard_category] += 1
            generated_un.add(realization.un_number)
            generated_hs.update(realization.generated_hs_codes)

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
    provenance_payload = b"".join(
        canonical_json_bytes(
            {
                "scenarioId": row.scenario_id,
                "baseDocumentId": row.base_document_id,
                "realizations": [value.model_dump(mode="json") for value in row.realizations],
            }
        )
        + b"\n"
        for row in output_rows
        if row.realizations
    )
    summary = {
        "schemaVersion": 1,
        "selectedDocuments": len(output_rows),
        "dangerousGoodsDocuments": sum(bool(row.realizations) for row in output_rows),
        "dangerousGoodsRows": sum(len(row.realizations) for row in output_rows),
        "branchCounts": dict(sorted(branch_counts.items())),
        "hazardCategoryCounts": dict(sorted(category_counts.items())),
        "branchSupportedHazardCategories": branch_support,
        "distinctGeneratedUnNumbers": len(generated_un),
        "distinctGeneratedHs6": len(generated_hs),
        "flashpointsGenerated": 0,
        "strictV4SchemaValidTargets": len(output_rows),
        "relationalInverseValidTargets": len(output_rows),
        "trainingRecordsPublished": 0,
        "samplingValidation": _sampling_validation(registry=registry, config=config),
    }
    summary_payload = json_artifact_bytes(summary)
    dg_document_count = cast(int, summary["dangerousGoodsDocuments"])
    dg_row_count = cast(int, summary["dangerousGoodsRows"])
    report = f"""# Integrated non-linguistic dangerous-goods pilot

- Final relation-v4 targets: **{len(output_rows):,}**
- DG documents / rows: **{dg_document_count:,} / {dg_row_count:,}**
- General regulatory tuples: **{branch_counts["general_regulatory_tuple"]:,}**
- Exact ECICS chemical + HS6 tuples: **{branch_counts["hs_linked_exact_chemical"]:,}**
- Numeric flashpoints invented: **0**

Every source v3 target was migrated deterministically and validated through the relation-v4
schema and relational inverse. DG fields are sampled as one PHMSA tuple. A cargo group that
already carried HS uses the exact-CUS ECICS branch; otherwise HS remains absent. Exact
regulatory and chemical facts are retained in `generation/dg-provenance.jsonl` for the later
text renderer. These records remain non-publishable training examples until party/cargo text
anonymization and raw-OCR patching are complete.
"""
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "configSha256": sha256_file(config_path),
                "semanticPlansSha256": config.inputs.semantic_plans.sha256,
                "sourceTaskConstraintsSha256": (config.inputs.source_task_constraints.sha256),
                "registryReceiptSha256": config.inputs.registry.receipt.sha256,
                "hmtSha256": config.inputs.registry.hmt_records.sha256,
                "ecicsSha256": config.inputs.registry.ecics_links.sha256,
                "registryContentSha256": receipt.content_sha256,
                "planSha256": sha256_bytes(plan_payload),
                "targetSha256": sha256_bytes(target_payload),
                "targetSchemaSha256": sha256_bytes(target_schema_payload),
                "taskConstraintsSha256": sha256_bytes(task_constraints_payload),
                "promptSchemaSha256": sha256_bytes(prompt_schema_payload),
                "implementationSha256": sha256_file(Path(__file__)),
                "registryImplementationSha256": sha256_file(
                    Path(dangerous_goods_registry_module.__file__)
                ),
                "schemaImplementationSha256": sha256_file(Path(bill_of_lading_v4_module.__file__)),
                "domainImplementationSha256": sha256_file(
                    Path(bill_of_lading_domain_module.__file__)
                ),
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
    stage.publish_bytes("generation/dg-plans.jsonl", plan_payload)
    stage.publish_bytes("generation/v4-targets.jsonl", target_payload)
    stage.publish_bytes("generation/dg-provenance.jsonl", provenance_payload)
    stage.publish_bytes("generation/summary.json", summary_payload)
    stage.publish_bytes(
        "generation/sampling-validation.json",
        json_artifact_bytes(summary["samplingValidation"]),
    )
    stage.publish_bytes("schema/relation-v4.schema.json", target_schema_payload)
    stage.publish_bytes("schema/task-constraints.json", task_constraints_payload)
    stage.publish_bytes("schema/prompt-schema.json", prompt_schema_payload)
    stage.publish_bytes("REPORT.md", report.encode("utf-8"))
    committed = stage.commit(
        expected_artifacts=(
            "REPORT.md",
            "config.yaml",
            "generation/dg-plans.jsonl",
            "generation/dg-provenance.jsonl",
            "generation/summary.json",
            "generation/sampling-validation.json",
            "generation/v4-targets.jsonl",
            "schema/prompt-schema.json",
            "schema/relation-v4.schema.json",
            "schema/task-constraints.json",
        ),
        metadata={
            "schema_version": 1,
            "documents": len(output_rows),
            "dangerous_goods_rows": dg_row_count,
            "strict_v4_targets": len(output_rows),
        },
    )
    return {
        "output_dir": str(stage.final_root),
        "created": committed.created,
        **summary,
    }
