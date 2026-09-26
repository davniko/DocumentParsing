"""Build deterministic, validation-safe production synthesis sample plans."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.reviewed_package_projection import load_reviewed_package_contract

from .dangerous_goods_realization import validate_explicit_un_source_coverage
from .descendant import _read_jsonl, _validate_committed_run
from .descendant_models import PinnedCommittedRun
from .latest_target import LatestTargetConstructionError, latest_target_from_source
from .models import CertifiedSemanticTemplate, NonEmptyText, PinnedFile, Sha256
from .route_derivations import supports_transshipment

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_TARGET_TASK = "bill_of_lading_relation_explicit_v5"
_TARGET_SCHEMA_VERSION = "5.0.0-experimental"
_LATEST_TARGET_PATH = Path(__file__).with_name("latest_target.py").resolve(strict=True)

CapabilityCohort = Literal["standard", "dangerous_goods", "temperature_controlled"]
ExclusionMode = Literal["exact_document_id", "template_proxy_family"]


class ValidationPartitionPin(PinnedFile):
    model_config = _STRICT

    report_format: Literal["t5gemma2_runtime_partition_dataset_report_v1"]
    split: Literal["validation"]
    expected_documents: Annotated[int, Field(gt=0)]
    document_ids_sha256: Sha256


class CapabilityCounts(BaseModel):
    model_config = _STRICT

    standard: Annotated[int, Field(ge=0)]
    dangerous_goods: Annotated[int, Field(ge=0)]
    temperature_controlled: Annotated[int, Field(ge=0)]

    def total(self) -> int:
        return self.standard + self.dangerous_goods + self.temperature_controlled


class ExpectedSelectionInventory(BaseModel):
    model_config = _STRICT

    validation_documents_in_catalog: Annotated[int, Field(ge=0)]
    validation_template_proxy_families: Annotated[int, Field(ge=0)]
    latest_schema_incompatible_templates: Annotated[int, Field(ge=0)]
    excluded_templates: Annotated[int, Field(ge=0)]
    eligible_templates: Annotated[int, Field(gt=0)]
    eligible_standard_templates: Annotated[int, Field(ge=0)]
    eligible_dangerous_goods_templates: Annotated[int, Field(ge=0)]
    eligible_temperature_controlled_templates: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def capability_counts_cover_eligible_templates(self) -> ExpectedSelectionInventory:
        if (
            self.eligible_standard_templates
            + self.eligible_dangerous_goods_templates
            + self.eligible_temperature_controlled_templates
            != self.eligible_templates
        ):
            raise ValueError("expected capability pools must cover every eligible template")
        return self


class ProductionSynthesisSelection(BaseModel):
    model_config = _STRICT

    documents: Annotated[int, Field(gt=0)]
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    sample_namespace: NonEmptyText
    scenario_mode: Literal["fixed_source_geography", "sampled_route", "sampled_route_and_cargo"] = (
        "fixed_source_geography"
    )
    exclusion_mode: ExclusionMode
    capability_counts: CapabilityCounts
    transshipment_counts: CapabilityCounts | None = None
    balancing: Literal[
        "even_reuse_within_capability_hmac_rank_v1", "weighted_reuse_largest_remainder_v1"
    ]
    template_weights: PinnedFile | None = None
    require_disjoint_capability_pools: Literal[True]
    require_every_eligible_template_selected: bool

    @model_validator(mode="after")
    def counts_cover_requested_documents(self) -> ProductionSynthesisSelection:
        if self.capability_counts.total() != self.documents:
            raise ValueError("capability counts must sum to selection.documents")
        if self.transshipment_counts is not None:
            for cohort, count in self.transshipment_counts.model_dump().items():
                if count > getattr(self.capability_counts, cohort):
                    raise ValueError("transshipment count exceeds its capability cohort count")
        if bool(self.template_weights) != (self.balancing == "weighted_reuse_largest_remainder_v1"):
            raise ValueError("weighted balancing requires explicit positive template weights")
        return self


class ProductionSynthesisPlanConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["bill_of_lading_compiled_template_synthesis_plan_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    template_run: PinnedCommittedRun
    validation_partition: ValidationPartitionPin
    target_task: Literal["bill_of_lading_relation_explicit_v5"]
    target_schema_version: Literal["5.0.0-experimental"]
    target_generation: Literal["complete_latest_schema_targets_v1"]
    selection: ProductionSynthesisSelection
    expected_inventory: ExpectedSelectionInventory
    source_review_exclusions: dict[NonEmptyText, NonEmptyText] = Field(default_factory=dict)
    task_package_contract: PinnedFile | None = None


@dataclass(frozen=True, slots=True)
class TemplateInventoryRow:
    source_document_id: str
    template_proxy_id: str
    carrier: str
    carrier_family: str
    document_type: str
    pages: int
    lines: int
    characters: int
    bindings: int
    occurrences: int
    deterministic_bindings: int
    agent_residual_bindings: int
    container_count: int
    cargo_group_count: int
    cargo_package_count: int
    allocation_group_count: int
    dangerous_goods: bool
    temperature_controlled: bool
    transshipment: bool
    cohort: CapabilityCohort
    latest_schema_compatible: bool
    latest_schema_incompatibility: str | None

    def plan_payload(self, *, sample_id: str, variant_index: int) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "sampleId": sample_id,
            "sourceDocumentId": self.source_document_id,
            "variantIndex": variant_index,
            "capabilityCohort": self.cohort,
            "dangerousGoods": self.dangerous_goods,
            "temperatureControlled": self.temperature_controlled,
            "transshipment": self.transshipment,
            "templateProxyId": self.template_proxy_id,
            "carrier": self.carrier,
            "carrierFamily": self.carrier_family,
            "documentType": self.document_type,
            "pages": self.pages,
            "lines": self.lines,
            "characters": self.characters,
            "bindings": self.bindings,
            "occurrences": self.occurrences,
            "deterministicBindings": self.deterministic_bindings,
            "agentResidualBindings": self.agent_residual_bindings,
            "containerCount": self.container_count,
            "cargoGroupCount": self.cargo_group_count,
            "cargoPackageCount": self.cargo_package_count,
            "allocationGroupCount": self.allocation_group_count,
            "targetTask": _TARGET_TASK,
            "targetSchemaVersion": _TARGET_SCHEMA_VERSION,
            "targetGeneration": "complete_latest_schema_targets_v1",
        }


def load_production_synthesis_plan_config(path: Path) -> ProductionSynthesisPlanConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return ProductionSynthesisPlanConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


def _validation_document_ids(
    *, project_root: Path, configured: ValidationPartitionPin
) -> tuple[str, ...]:
    path = (project_root / configured.path).resolve(strict=True)
    if project_root not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError("validation partition report is not a regular project file")
    if sha256_file(path) != configured.sha256:
        raise ValueError("validation partition report hash differs")
    report = json.loads(read_regular_file_bytes(path))
    try:
        ids = report["inspection"]["partition"]["outputs"][configured.split]["document_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError("validation partition report has an unexpected structure") from error
    if (
        not isinstance(ids, list)
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("validation partition document IDs are invalid or duplicated")
    frozen = tuple(ids)
    if len(frozen) != configured.expected_documents:
        raise ValueError("validation partition document count differs")
    if sha256_bytes(canonical_json_bytes(frozen)) != configured.document_ids_sha256:
        raise ValueError("validation partition document identity hash differs")
    return frozen


def _target_capabilities(target: Mapping[str, Any]) -> tuple[bool, bool, tuple[int, int, int, int]]:
    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source label lacks documentPatch")
    containers = patch.get("containers") or []
    cargo_groups = patch.get("cargoGroups") or []
    cargo_packages = patch.get("cargoPackages") or []
    allocations = patch.get("cargoAllocationGroups") or []
    for label, rows in (
        ("containers", containers),
        ("cargoGroups", cargo_groups),
        ("cargoPackages", cargo_packages),
        ("cargoAllocationGroups", allocations),
    ):
        if not isinstance(rows, list):
            raise ValueError(f"source label {label} is not a list")
    dangerous_goods = any(
        isinstance(row, Mapping) and bool(row.get("dangerousGoods")) for row in cargo_groups
    )
    temperature = any(
        isinstance(row, Mapping) and row.get("temperatureSetpoint") is not None
        for row in containers
    )
    return (
        dangerous_goods,
        temperature,
        (
            len(containers),
            len(cargo_groups),
            len(cargo_packages),
            len(allocations),
        ),
    )


def _template_inventory(template_root: Path) -> tuple[TemplateInventoryRow, ...]:
    catalog_rows = _read_jsonl(template_root / "catalog.jsonl", records=None)
    inventory: list[TemplateInventoryRow] = []
    seen: set[str] = set()
    for row in catalog_rows:
        document_id = row.get("documentId")
        proxy_id = row.get("templateProxyId")
        if (
            row.get("certified") is not True
            or not isinstance(document_id, str)
            or not document_id
            or not isinstance(proxy_id, str)
            or not proxy_id
        ):
            raise ValueError("production catalog contains an invalid certified identity row")
        if document_id in seen:
            raise ValueError(f"production catalog repeats document {document_id}")
        seen.add(document_id)
        case_root = template_root / "cases" / document_id
        label = json.loads(read_regular_file_bytes(case_root / "source-label.json"))
        if not isinstance(label, dict):
            raise ValueError(f"source label is not an object: {document_id}")
        latest_schema_incompatibility: str | None
        try:
            latest_target_from_source(label)
        except LatestTargetConstructionError as error:
            latest_schema_compatible = False
            latest_schema_incompatibility = str(error)
        else:
            latest_schema_compatible = True
            latest_schema_incompatibility = None
        dangerous_goods, temperature, topology = _target_capabilities(label)
        labelled_transshipment = bool(
            label["documentPatch"].get("route", {}).get("transshipmentPort")
        )
        route_summary_present = "routeCapabilities" in row
        route_capabilities = row.get("routeCapabilities")
        if route_summary_present and (
            not isinstance(route_capabilities, dict)
            or set(route_capabilities) != {"transshipment"}
            or type(route_capabilities["transshipment"]) is not bool
        ):
            raise ValueError("catalog route-capability metadata is invalid")
        reported_transshipment: bool | None
        if route_summary_present:
            assert isinstance(route_capabilities, dict)
            reported_transshipment = route_capabilities["transshipment"]
        else:
            reported_transshipment = None
        value_kinds = row.get("valueKinds")
        if not isinstance(value_kinds, Mapping):
            raise ValueError(f"catalog value-kind summary is absent: {document_id}")
        summarized_dangerous_goods = bool(value_kinds.get("dangerous_goods", 0))
        summarized_temperature = bool(value_kinds.get("temperature", 0))
        if (
            dangerous_goods
            or temperature
            or labelled_transshipment
            or reported_transshipment
            or summarized_dangerous_goods
            or summarized_temperature
            or not route_summary_present
        ):
            template = CertifiedSemanticTemplate.model_validate_json(
                read_regular_file_bytes(case_root / "template.json"), strict=True
            )
            if template.document_id != document_id:
                raise ValueError(f"template identity differs from catalog: {document_id}")
            if summarized_dangerous_goods:
                validate_explicit_un_source_coverage(
                    read_regular_file_bytes(case_root / "source.txt").decode("utf-8"),
                    label,
                )
            target_paths = tuple(
                path for binding in template.bindings for path in binding.target_paths
            )
            transshipment = supports_transshipment(template.bindings)
            if (
                reported_transshipment is not None
                and reported_transshipment != transshipment
            ):
                raise ValueError(
                    f"catalog route capability and compiled bindings differ: {document_id}"
                )
            bound_dangerous_goods = any(".dangerousGoods" in path for path in target_paths)
            bound_temperature = any(".temperatureSetpoint" in path for path in target_paths)
            if labelled_transshipment and not any(
                path.startswith("documentPatch.route.transshipmentPort.") for path in target_paths
            ):
                raise ValueError(f"transshipment source has no compiled binding: {document_id}")
            if transshipment and not labelled_transshipment:
                from . import route_derivations

                route_derivations.validate_source(label, template.bindings)
                physical = route_derivations.physical_source(label, template.bindings)
                if not physical["documentPatch"].get("route", {}).get("transshipmentPort"):
                    raise ValueError(
                        f"transshipment metadata lacks source-only evidence: {document_id}"
                    )
        else:
            bound_dangerous_goods = False
            bound_temperature = False
            transshipment = False
        if labelled_transshipment and not transshipment:
            raise ValueError(f"transshipment source has no compiled capability: {document_id}")
        if (dangerous_goods, temperature) != (bound_dangerous_goods, bound_temperature):
            raise ValueError(
                f"source capability and compiled bindings differ for {document_id}: "
                f"source={(dangerous_goods, temperature)}, "
                f"bindings={(bound_dangerous_goods, bound_temperature)}"
            )
        if dangerous_goods and temperature:
            cohort: CapabilityCohort = "dangerous_goods"
        elif dangerous_goods:
            cohort = "dangerous_goods"
        elif temperature:
            cohort = "temperature_controlled"
        else:
            cohort = "standard"
        inventory.append(
            TemplateInventoryRow(
                source_document_id=document_id,
                template_proxy_id=proxy_id,
                carrier=cast(str, row["carrier"]),
                carrier_family=cast(str, row["carrierFamily"]),
                document_type=cast(str, row["documentType"]),
                pages=cast(int, row["pages"]),
                lines=cast(int, row["lines"]),
                characters=cast(int, row["characters"]),
                bindings=cast(int, row["bindings"]),
                occurrences=cast(int, row["occurrences"]),
                deterministic_bindings=cast(int, row["deterministicBindings"]),
                agent_residual_bindings=cast(int, row["agentResidualBindings"]),
                container_count=topology[0],
                cargo_group_count=topology[1],
                cargo_package_count=topology[2],
                allocation_group_count=topology[3],
                dangerous_goods=dangerous_goods,
                temperature_controlled=temperature,
                transshipment=transshipment,
                cohort=cohort,
                latest_schema_compatible=latest_schema_compatible,
                latest_schema_incompatibility=latest_schema_incompatibility,
            )
        )
    return tuple(inventory)


def _cohort_repetitions(
    *,
    templates: Sequence[TemplateInventoryRow],
    cohort: CapabilityCohort,
    selection: ProductionSynthesisSelection,
    weights: Mapping[str, int],
) -> tuple[tuple[TemplateInventoryRow, int], ...]:
    total = getattr(selection.capability_counts, cohort)
    if selection.transshipment_counts is None:
        pools = [(tuple(templates), total)]
    else:
        via_count = getattr(selection.transshipment_counts, cohort)
        pools = [
            (tuple(row for row in templates if row.transshipment == via), count)
            for via, count in ((True, via_count), (False, total - via_count))
        ]
    result: list[tuple[TemplateInventoryRow, int]] = []
    for pool, requested in pools:
        if selection.require_every_eligible_template_selected and requested < len(pool):
            raise ValueError("route-topology quota cannot cover every eligible template")
        if selection.balancing == "weighted_reuse_largest_remainder_v1":
            rows = _weighted_repetitions(
                templates=pool,
                requested=requested,
                weights=weights,
                namespace=selection.sample_namespace,
                seed=selection.seed,
                cohort=cohort,
                require_every=selection.require_every_eligible_template_selected,
            )
        else:
            rows = _even_repetitions(
                templates=pool,
                requested=requested,
                namespace=selection.sample_namespace,
                seed=selection.seed,
                cohort=cohort,
            )
        result.extend(rows)
    return tuple(result)


def _rank(*, namespace: str, seed: int, cohort: str, value: str) -> str:
    return sha256_bytes(canonical_json_bytes([namespace, seed, cohort, value]))


def _even_repetitions(
    *,
    templates: Sequence[TemplateInventoryRow],
    requested: int,
    namespace: str,
    seed: int,
    cohort: CapabilityCohort,
) -> tuple[tuple[TemplateInventoryRow, int], ...]:
    if requested and not templates:
        raise ValueError(f"no eligible templates can satisfy the {cohort} cohort")
    ordered = tuple(
        sorted(
            templates,
            key=lambda row: (
                _rank(
                    namespace=namespace,
                    seed=seed,
                    cohort=cohort,
                    value=row.source_document_id,
                ),
                row.source_document_id,
            ),
        )
    )
    if not ordered:
        return ()
    quotient, remainder = divmod(requested, len(ordered))
    return tuple(
        (row, quotient + int(position < remainder))
        for position, row in enumerate(ordered)
        if quotient + int(position < remainder)
    )


def _weighted_repetitions(
    *,
    templates: Sequence[TemplateInventoryRow],
    requested: int,
    weights: Mapping[str, int],
    namespace: str,
    seed: int,
    cohort: CapabilityCohort,
    require_every: bool,
) -> tuple[tuple[TemplateInventoryRow, int], ...]:
    """Exact-count Hamilton allocation, with deterministic ties and optional coverage.

    Categories remain joint template capabilities, so reweighting cannot pair a
    reefer cargo with an incompatible independently drawn tank/container type.
    """
    if requested and not templates:
        raise ValueError(f"no eligible templates can satisfy the {cohort} cohort")
    if not templates:
        return ()
    if any(weights.get(row.source_document_id, 0) <= 0 for row in templates):
        raise ValueError("every eligible template requires an explicit positive weight")
    base = int(require_every)
    remaining = requested - base * len(templates)
    if remaining < 0:
        raise ValueError("requested cohort cannot cover every eligible template")
    total = sum(weights[row.source_document_id] for row in templates)
    counts = {}
    remainders = {}
    for row in templates:
        count, remainder = divmod(remaining * weights[row.source_document_id], total)
        counts[row.source_document_id] = count + base
        remainders[row.source_document_id] = remainder
    ordered = sorted(
        templates,
        key=lambda row: (
            -remainders[row.source_document_id],
            _rank(namespace=namespace, seed=seed, cohort=cohort, value=row.source_document_id),
            row.source_document_id,
        ),
    )
    for row in ordered[: requested - sum(counts.values())]:
        counts[row.source_document_id] += 1
    return tuple(
        (row, counts[row.source_document_id]) for row in templates if counts[row.source_document_id]
    )


def _sample_id(*, namespace: str, seed: int, source_document_id: str, variant_index: int) -> str:
    digest = sha256_bytes(
        canonical_json_bytes(
            [namespace, seed, source_document_id, variant_index, _TARGET_SCHEMA_VERSION]
        )
    )
    # Upstream label-synthesis stages use the same strict document-identity shape as
    # source records. The final rendered record receives its separate ``syn_tpl_`` ID.
    return f"doc_{digest}"


def _bucket(value: int) -> str:
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 3:
        return "two_to_three"
    return "four_plus"


def _numeric_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _plan_summary(
    *,
    config: ProductionSynthesisPlanConfig,
    inventory: Sequence[TemplateInventoryRow],
    validation_ids: Sequence[str],
    validation_proxy_ids: frozenset[str],
    excluded: Sequence[TemplateInventoryRow],
    eligible: Sequence[TemplateInventoryRow],
    plan_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repeat_counts = Counter(cast(str, row["sourceDocumentId"]) for row in plan_rows)
    cohort_repeats: dict[str, dict[str, float | int]] = {}
    for cohort in ("standard", "dangerous_goods", "temperature_controlled"):
        ids = {
            cast(str, row["sourceDocumentId"])
            for row in plan_rows
            if row["capabilityCohort"] == cohort
        }
        cohort_repeats[cohort] = _numeric_summary([repeat_counts[value] for value in ids])
    selected_ids = frozenset(repeat_counts)
    target_counts = Counter(cast(str, row["capabilityCohort"]) for row in plan_rows)
    validation_set = frozenset(validation_ids)
    remaining_validation_ids = sorted(selected_ids & validation_set)
    retained_validation_proxy_rows = tuple(
        row for row in eligible if row.template_proxy_id in validation_proxy_ids
    )
    remaining_validation_proxies = sorted(
        {row.template_proxy_id for row in retained_validation_proxy_rows}
    )
    return {
        "schemaVersion": 1,
        "status": "ready",
        "documents": len(plan_rows),
        "targetTask": config.target_task,
        "targetSchemaVersion": config.target_schema_version,
        "targetGeneration": config.target_generation,
        "exclusionMode": config.selection.exclusion_mode,
        "validationDocuments": len(validation_ids),
        "validationDocumentsInCatalog": len(
            {row.source_document_id for row in inventory} & validation_set
        ),
        "validationTemplateProxyFamilies": len(validation_proxy_ids),
        "excludedTemplates": len(excluded),
        "latestSchemaIncompatibleTemplates": sum(
            not row.latest_schema_compatible for row in inventory
        ),
        "eligibleTemplates": len(eligible),
        "selectedTemplates": len(selected_ids),
        "unselectedEligibleTemplates": len(
            {row.source_document_id for row in eligible} - selected_ids
        ),
        "remainingExactValidationDocuments": len(remaining_validation_ids),
        "remainingValidationTemplateProxyFamilies": len(remaining_validation_proxies),
        "retainedValidationProxySiblingTemplates": len(retained_validation_proxy_rows),
        "capabilityDocuments": {
            "standard": target_counts["standard"],
            "dangerousGoods": target_counts["dangerous_goods"],
            "temperatureControlled": target_counts["temperature_controlled"],
        },
        "capabilityFractions": {
            "standard": target_counts["standard"] / len(plan_rows),
            "dangerousGoods": target_counts["dangerous_goods"] / len(plan_rows),
            "temperatureControlled": target_counts["temperature_controlled"] / len(plan_rows),
        },
        "transshipmentDocuments": sum(bool(row["transshipment"]) for row in plan_rows),
        "eligibleTransshipmentTemplates": sum(row.transshipment for row in eligible),
        "eligibleCapabilityTemplates": dict(
            sorted(Counter(row.cohort for row in eligible).items())
        ),
        "templateReuse": {
            "overall": _numeric_summary(list(repeat_counts.values())),
            "byCapability": cohort_repeats,
        },
        "distinctTemplateProxyFamilies": len(
            {cast(str, row["templateProxyId"]) for row in plan_rows}
        ),
        "distinctCarriers": len({cast(str, row["carrierFamily"]) for row in plan_rows}),
        "carrierFamilies": _distribution(plan_rows, "carrierFamily"),
        "documentTypes": _distribution(plan_rows, "documentType"),
        "pageCounts": dict(
            sorted(Counter(str(cast(int, row["pages"])) for row in plan_rows).items())
        ),
        "containerCountBuckets": dict(
            sorted(Counter(_bucket(cast(int, row["containerCount"])) for row in plan_rows).items())
        ),
        "cargoGroupCountBuckets": dict(
            sorted(Counter(_bucket(cast(int, row["cargoGroupCount"])) for row in plan_rows).items())
        ),
        "cargoPackageCountBuckets": dict(
            sorted(
                Counter(_bucket(cast(int, row["cargoPackageCount"])) for row in plan_rows).items()
            )
        ),
        "documentsWithMultipleContainers": sum(
            cast(int, row["containerCount"]) > 1 for row in plan_rows
        ),
        "documentsWithMultipleCargoGroups": sum(
            cast(int, row["cargoGroupCount"]) > 1 for row in plan_rows
        ),
        "documentsWithMultipleCargoPackages": sum(
            cast(int, row["cargoPackageCount"]) > 1 for row in plan_rows
        ),
        "documentsWithAllocations": sum(
            cast(int, row["allocationGroupCount"]) > 0 for row in plan_rows
        ),
        "documentsWithAgentResidualBindings": sum(
            cast(int, row["agentResidualBindings"]) > 0 for row in plan_rows
        ),
        "weightedBindings": sum(cast(int, row["bindings"]) for row in plan_rows),
        "weightedDeterministicBindings": sum(
            cast(int, row["deterministicBindings"]) for row in plan_rows
        ),
        "weightedOccurrences": sum(cast(int, row["occurrences"]) for row in plan_rows),
        "providerRequests": 0,
        "incrementalCostUsd": "0",
    }


def _analyze_plan(*, project_root: Path, config: ProductionSynthesisPlanConfig) -> dict[str, Any]:
    template_root = _validate_committed_run(project_root, config.template_run)
    validation_ids = _validation_document_ids(
        project_root=project_root, configured=config.validation_partition
    )
    inventory = _template_inventory(template_root)
    by_id = {row.source_document_id: row for row in inventory}
    if set(config.source_review_exclusions) - set(by_id):
        raise ValueError("source review exclusions contain documents outside the pinned catalog")
    package_review_reasons: dict[str, list[str]] = {}
    if config.task_package_contract is not None:
        pin = config.task_package_contract
        contract_path = (project_root / pin.path).resolve(strict=True)
        if project_root not in contract_path.parents:
            raise ValueError("task package contract escapes project")
        contract = load_reviewed_package_contract(
            contract_path,
            expected_sha256=pin.sha256,
            catalog_commit_sha256=config.template_run.commit_sha256,
        )
        for (source_id, group_id), decision in contract.decisions.items():
            if decision.status == "review":
                package_review_reasons.setdefault(source_id, []).append(
                    f"{group_id}: {decision.rationale}"
                )
        package_review_exclusions = {
            source_id: "; ".join(sorted(reasons))
            for source_id, reasons in package_review_reasons.items()
        }
        if set(package_review_exclusions) - set(by_id):
            raise ValueError(
                "package review exclusions contain documents outside the pinned catalog"
            )
    else:
        package_review_exclusions = {}
    validation_catalog_ids = frozenset(by_id) & frozenset(validation_ids)
    validation_proxy_ids = frozenset(
        by_id[document_id].template_proxy_id for document_id in validation_catalog_ids
    )
    if config.selection.require_disjoint_capability_pools and any(
        row.dangerous_goods and row.temperature_controlled for row in inventory
    ):
        raise ValueError("compiled catalog contains overlapping DG and temperature capabilities")
    if config.selection.exclusion_mode == "exact_document_id":
        validation_excluded = tuple(
            row for row in inventory if row.source_document_id in validation_catalog_ids
        )
    else:
        validation_excluded = tuple(
            row for row in inventory if row.template_proxy_id in validation_proxy_ids
        )
    latest_schema_incompatible = tuple(row for row in inventory if not row.latest_schema_compatible)
    excluded_ids = {
        row.source_document_id for row in (*validation_excluded, *latest_schema_incompatible)
    } | set(config.source_review_exclusions) | set(package_review_exclusions)
    excluded = tuple(row for row in inventory if row.source_document_id in excluded_ids)
    eligible = tuple(row for row in inventory if row.source_document_id not in excluded_ids)
    capability_pools: dict[CapabilityCohort, tuple[TemplateInventoryRow, ...]] = {
        cohort: tuple(row for row in eligible if row.cohort == cohort)
        for cohort in ("standard", "dangerous_goods", "temperature_controlled")
    }
    expected = config.expected_inventory
    observed = {
        "validation_documents_in_catalog": len(validation_catalog_ids),
        "validation_template_proxy_families": len(validation_proxy_ids),
        "latest_schema_incompatible_templates": len(latest_schema_incompatible),
        "excluded_templates": len(excluded),
        "eligible_templates": len(eligible),
        "eligible_standard_templates": len(capability_pools["standard"]),
        "eligible_dangerous_goods_templates": len(capability_pools["dangerous_goods"]),
        "eligible_temperature_controlled_templates": len(
            capability_pools["temperature_controlled"]
        ),
    }
    if observed != expected.model_dump(mode="python"):
        raise ValueError(f"production synthesis inventory differs: {observed!r}")
    weights: dict[str, int] = {}
    if config.selection.template_weights is not None:
        pin = config.selection.template_weights
        weights_path = (project_root / pin.path).resolve(strict=True)
        if project_root not in weights_path.parents or sha256_file(weights_path) != pin.sha256:
            raise ValueError("template weights file identity differs")
        weights = json.loads(read_regular_file_bytes(weights_path))
        if not isinstance(weights, dict) or any(
            type(v) is not int or v <= 0 for v in weights.values()
        ):
            raise ValueError("template weights must be positive integers")
    if weights and set(weights) != {row.source_document_id for row in eligible}:
        raise ValueError("template weights must cover the eligible catalog exactly")
    if config.selection.require_every_eligible_template_selected:
        requested_by_cohort = config.selection.capability_counts.model_dump(mode="python")
        for cohort, templates in capability_pools.items():
            if cast(int, requested_by_cohort[cohort]) < len(templates):
                raise ValueError(f"{cohort} document count cannot select every eligible template")

    plan_rows: list[dict[str, Any]] = []
    requested_by_cohort = config.selection.capability_counts.model_dump(mode="python")
    for cohort in ("standard", "dangerous_goods", "temperature_controlled"):
        repetitions = _cohort_repetitions(
            templates=capability_pools[cohort],
            cohort=cohort,
            selection=config.selection,
            weights=weights,
        )
        for template, count in repetitions:
            for variant_index in range(count):
                sample_id = _sample_id(
                    namespace=config.selection.sample_namespace,
                    seed=config.selection.seed,
                    source_document_id=template.source_document_id,
                    variant_index=variant_index,
                )
                plan_rows.append(
                    {
                        **template.plan_payload(sample_id=sample_id, variant_index=variant_index),
                        "scenarioMode": config.selection.scenario_mode,
                    }
                )
    plan_rows.sort(
        key=lambda row: (
            _rank(
                namespace=config.selection.sample_namespace,
                seed=config.selection.seed,
                cohort="sample-order",
                value=cast(str, row["sampleId"]),
            ),
            cast(str, row["sampleId"]),
        )
    )
    sample_ids = tuple(cast(str, row["sampleId"]) for row in plan_rows)
    if len(plan_rows) != config.selection.documents or len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("production synthesis plan count or identity uniqueness failed")
    summary = _plan_summary(
        config=config,
        inventory=inventory,
        validation_ids=validation_ids,
        validation_proxy_ids=validation_proxy_ids,
        excluded=excluded,
        eligible=eligible,
        plan_rows=plan_rows,
    )
    summary["packageReviewExcludedTemplates"] = len(package_review_exclusions)
    if summary["remainingExactValidationDocuments"]:
        raise RuntimeError("production synthesis plan contains an exact validation document")
    if (
        config.selection.exclusion_mode == "template_proxy_family"
        and summary["remainingValidationTemplateProxyFamilies"]
    ):
        raise RuntimeError("layout-proxy plan contains a validation proxy family")
    return {
        "template_root": template_root,
        "validation_ids": validation_ids,
        "validation_proxy_ids": validation_proxy_ids,
        "validation_excluded": validation_excluded,
        "latest_schema_incompatible": latest_schema_incompatible,
        "package_review_exclusions": package_review_exclusions,
        "inventory": inventory,
        "excluded": excluded,
        "eligible": eligible,
        "plan_rows": tuple(plan_rows),
        "summary": summary,
    }


def preflight_production_synthesis_plan(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    config = load_production_synthesis_plan_config(config_path.resolve(strict=True))
    analyzed = _analyze_plan(project_root=project_root, config=config)
    return cast(dict[str, Any], analyzed["summary"])


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _template_payload(row: TemplateInventoryRow) -> dict[str, Any]:
    return {
        "sourceDocumentId": row.source_document_id,
        "templateProxyId": row.template_proxy_id,
        "carrier": row.carrier,
        "carrierFamily": row.carrier_family,
        "documentType": row.document_type,
        "capabilityCohort": row.cohort,
        "transshipment": row.transshipment,
        "pages": row.pages,
        "containerCount": row.container_count,
        "cargoGroupCount": row.cargo_group_count,
        "cargoPackageCount": row.cargo_package_count,
        "allocationGroupCount": row.allocation_group_count,
        "agentResidualBindings": row.agent_residual_bindings,
        "latestSchemaCompatible": row.latest_schema_compatible,
        "latestSchemaIncompatibility": row.latest_schema_incompatibility,
    }


def build_production_synthesis_plan(*, project_root: Path, config_path: Path) -> Path:
    started = time.perf_counter()
    project_root = project_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise ValueError("production synthesis plan config is outside the project root") from error
    config = load_production_synthesis_plan_config(config_path)
    analyzed = _analyze_plan(project_root=project_root, config=config)
    output_parent = (project_root / config.output_dir).resolve()
    if output_parent != project_root and project_root not in output_parent.parents:
        raise ValueError("production synthesis plan output directory escapes the project root")
    lineage = {
        "schemaVersion": 1,
        "config": config.model_dump(mode="json"),
        "configSha256": sha256_file(config_path),
        "implementationSha256": sha256_file(Path(__file__).resolve(strict=True)),
        "latestTargetImplementationSha256": sha256_file(_LATEST_TARGET_PATH),
        "templateRunCommitSha256": config.template_run.commit_sha256,
        "templateRunTransactionSha256": config.template_run.transaction_sha256,
        "validationPartitionSha256": config.validation_partition.sha256,
        "validationDocumentIdsSha256": config.validation_partition.document_ids_sha256,
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

    plan_rows = cast(tuple[dict[str, Any], ...], analyzed["plan_rows"])
    eligible = cast(tuple[TemplateInventoryRow, ...], analyzed["eligible"])
    excluded = cast(tuple[TemplateInventoryRow, ...], analyzed["excluded"])
    validation_ids = cast(tuple[str, ...], analyzed["validation_ids"])
    validation_proxy_ids = cast(frozenset[str], analyzed["validation_proxy_ids"])
    validation_excluded = cast(tuple[TemplateInventoryRow, ...], analyzed["validation_excluded"])
    latest_schema_incompatible = cast(
        tuple[TemplateInventoryRow, ...], analyzed["latest_schema_incompatible"]
    )
    summary = dict(cast(dict[str, Any], analyzed["summary"]))
    summary["wallSeconds"] = time.perf_counter() - started
    publish_json("config.json", config.model_dump(mode="json"))
    publish_json("source-review-exclusions.json", config.source_review_exclusions)
    package_review_exclusions = cast(dict[str, str], analyzed["package_review_exclusions"])
    publish_json("package-review-exclusions.json", package_review_exclusions)
    publish_json("lineage.json", lineage)
    publish_bytes("plan.jsonl", _jsonl_bytes(plan_rows))
    publish_bytes(
        "eligible-templates.jsonl",
        _jsonl_bytes([_template_payload(row) for row in eligible]),
    )
    publish_bytes(
        "excluded-templates.jsonl",
        _jsonl_bytes([_template_payload(row) for row in excluded]),
    )
    publish_bytes(
        "latest-schema-incompatible-templates.jsonl",
        _jsonl_bytes([_template_payload(row) for row in latest_schema_incompatible]),
    )
    publish_json(
        "validation-exclusion.json",
        {
            "schemaVersion": 1,
            "mode": config.selection.exclusion_mode,
            "validationDocumentIds": validation_ids,
            "validationDocumentIdsSha256": config.validation_partition.document_ids_sha256,
            "validationTemplateProxyIds": sorted(validation_proxy_ids),
            "validationExcludedTemplateDocumentIds": sorted(
                row.source_document_id for row in validation_excluded
            ),
            "latestSchemaIncompatibleTemplateDocumentIds": sorted(
                row.source_document_id for row in latest_schema_incompatible
            ),
            "sourceReviewExcludedTemplateDocumentIds": sorted(config.source_review_exclusions),
            "packageReviewExcludedTemplateDocumentIds": sorted(package_review_exclusions),
            "excludedTemplateDocumentIds": sorted(row.source_document_id for row in excluded),
        },
    )
    publish_json("summary.json", summary)
    capability = cast(dict[str, int], summary["capabilityDocuments"])
    reuse = cast(dict[str, Any], summary["templateReuse"])["overall"]
    publish_bytes(
        "REPORT.md",
        (
            "# Production compiled-template synthesis plan\n\n"
            f"- Samples: **{summary['documents']:,}** using **{summary['selectedTemplates']:,}** "
            f"of **{summary['eligibleTemplates']:,}** eligible templates.\n"
            f"- Validation exclusion: **{summary['exclusionMode']}**; exact validation "
            "documents remaining: **0**.\n"
            f"- Dangerous goods: **{capability['dangerousGoods']:,}**; temperature-controlled: "
            f"**{capability['temperatureControlled']:,}**; standard: "
            f"**{capability['standard']:,}**.\n"
            f"- Template reuse: **{reuse['min']}-{reuse['max']}** samples per selected template "
            f"(mean **{reuse['mean']:.3f}**).\n"
            f"- Target contract: **{summary['targetTask']} / "
            f"{summary['targetSchemaVersion']}** only.\n"
            f"- Latest-schema incompatible source templates excluded: "
            f"**{summary['latestSchemaIncompatibleTemplates']:,}**.\n"
            "- Planner provider calls and model cost: **0 / $0**.\n"
        ).encode(),
    )
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 1,
            "phase": "production_compiled_template_synthesis_plan",
            "documents": cast(int, summary["documents"]),
            "eligibleTemplates": cast(int, summary["eligibleTemplates"]),
            "selectedTemplates": cast(int, summary["selectedTemplates"]),
            "exclusionMode": config.selection.exclusion_mode,
            "targetSchemaVersion": config.target_schema_version,
        },
    )
    return staged.final_root
