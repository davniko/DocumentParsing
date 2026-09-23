"""Train-only joint cargo sampling for compiled templates, before model calls.

Registries supply identities; the fit partition supplies packaging and equipment
compatibility. Candidate rejection is explicit and bounded. No rejection restores
source values, and none can consume a linguistic-generation request.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from fractions import Fraction
from math import gcd, lcm
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v5 import (
    TEMPERATURE_CAPABLE_CONTAINER_TYPES,
    ContainerSizeCategory,
    ContainerTypeCategory,
)
from document_ocr.synthesis.config import SemanticCompletionThermalConfig, TransportCapacityConfig
from document_ocr.synthesis.container_semantics import (
    EquipmentSemanticSupport,
    SourceEquipmentObservation,
    build_equipment_semantic_support,
    canonical_equipment_surface,
    sample_equipment_semantic,
)
from document_ocr.synthesis.container_semantics import (
    partial_equipment_constraint as _partial_equipment_constraint,
)
from document_ocr.synthesis.dangerous_goods_registry import (
    LoadedDangerousGoodsRegistry,
    load_dangerous_goods_registry,
)
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import UkGlobalTariffRegistry
from document_ocr.synthesis.package_goods_compatibility import (
    PackageGoodsFitSupport,
    apply_package_signature,
    build_package_goods_fit_support,
    sample_compatible_cargo,
)
from document_ocr.synthesis.semantic_completion_pipeline import _group_allocations
from document_ocr.synthesis.thermal_goods import (
    ThermalGoodsIdentity,
    ThermalGoodsSupport,
    build_thermal_goods_support,
    render_synthetic_hs_code,
    sample_ambient_goods,
    sample_temperature_setpoint,
    sample_thermal_goods,
    sample_thermal_profile,
)
from document_ocr.synthesis.transport_capacity import capacity_limits, document_capacity_receipt

from . import (
    anonymous_equipment,
    cargo_identifiers,
    cargo_measurements,
    equipment_row_constraints,
    equipment_tares,
    measurement_prose,
    mixed_inventory,
    numeric_auxiliary,
    observed_thermal_goods,
    package_equations,
    package_observations,
    package_prose,
    temperature_prose,
    transport_derivations,
    unit_package_loads,
)
from . import complete_targets as targets
from . import dangerous_goods_packaging as dg_packaging
from . import descendant as render
from .dangerous_goods_realization import (
    DangerousGoodsFact,
    compile_surfaces,
    render_facts,
    validate_un_references,
)
from .descendant_models import PinnedCommittedRun
from .models import CertifiedSemanticTemplate, NonEmptyText, PinnedJsonl
from .pipeline import resolve_input


class CargoSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    method: Literal["registry_goods_fit_joint_packages_equipment_v1"]
    fit_records: PinnedJsonl
    validation_records: PinnedJsonl
    dangerous_goods_run: PinnedCommittedRun
    hmt: PinnedJsonl
    ecics: PinnedJsonl
    thermal: SemanticCompletionThermalConfig
    transport_capacity: TransportCapacityConfig
    equipment_joint_weights: dict[NonEmptyText, Annotated[int, Field(gt=0)]]
    maximum_candidates: Annotated[int, Field(gt=0, le=1000)]
    measurement_lower_multiplier: Annotated[float, Field(gt=0, le=1)]
    measurement_upper_multiplier: Annotated[float, Field(ge=1)]

    @model_validator(mode="after")
    def equipment_weights_are_valid(self) -> CargoSamplingConfig:
        if not self.equipment_joint_weights:
            raise ValueError("explicit joint equipment weights are required")
        for key in self.equipment_joint_weights:
            size, category = key.split("|")
            TypeAdapter(ContainerSizeCategory).validate_python(size, strict=True)
            TypeAdapter(ContainerTypeCategory).validate_python(category, strict=True)
        return self


@dataclass(frozen=True)
class CargoSupport:
    config: CargoSamplingConfig
    hs: UkGlobalTariffRegistry
    goods: ThermalGoodsSupport
    packages: PackageGoodsFitSupport
    equipment: EquipmentSemanticSupport
    equipment_types_by_heading: Mapping[str, frozenset[str]]
    dg: LoadedDangerousGoodsRegistry
    validation_ids: frozenset[str]
    measurements: cargo_measurements.MeasurementSupport
    descriptions: Mapping[str, str]
    tares: equipment_tares.TareSupport


def _records(root: Path, pin: PinnedJsonl) -> tuple[dict[str, Any], ...]:
    path = resolve_input(root, pin.path)
    if sha256_file(path) != pin.sha256:
        raise ValueError(f"cargo sampling dependency hash differs: {pin.path}")
    return render._read_jsonl(path, records=pin.records)


def load_support(
    root: Path, config: CargoSamplingConfig, hs: UkGlobalTariffRegistry
) -> CargoSupport:
    fit = _records(root, config.fit_records)
    validation = _records(root, config.validation_records)
    fit_targets = {r["documentId"]: package_observations.fit_target(r["target"]) for r in fit}
    validation_ids = frozenset(r["documentId"] for r in validation)
    if len(fit_targets) != len(fit) or len(validation_ids) != len(validation):
        raise ValueError("cargo fit/validation document IDs must be unique")
    if fit_targets.keys() & validation_ids:
        raise ValueError("validation documents cannot enter cargo sampling fit")
    dg_root = render._validate_committed_run(root, config.dangerous_goods_run)
    for pin in (config.hmt, config.ecics):
        if dg_root not in resolve_input(root, pin.path).parents:
            raise ValueError("DG sampling file is outside its pinned registry run")
        _records(root, pin)
    dg = load_dangerous_goods_registry(
        hmt_path=resolve_input(root, config.hmt.path),
        hmt_sha256=config.hmt.sha256,
        ecics_path=resolve_input(root, config.ecics.path),
        ecics_sha256=config.ecics.sha256,
    )
    thermal = config.thermal
    packages = build_package_goods_fit_support(
        source_targets=fit_targets,
        fit_document_ids=tuple(fit_targets),
        allowed_category_tokens=sorted(
            {
                p["typeCategory"]
                for t in fit_targets.values()
                for p in t["documentPatch"].get("cargoPackages", [])
                if "typeCategory" in p
            }
        ),
        frozen_minimum_celsius=thermal.frozen_minimum_celsius,
        frozen_maximum_celsius=thermal.frozen_maximum_celsius,
        chilled_minimum_celsius=thermal.chilled_minimum_celsius,
        chilled_maximum_celsius=thermal.chilled_maximum_celsius,
    )
    observations = []
    by_heading: dict[str, set[str]] = defaultdict(set)
    for sid, target in fit_targets.items():
        patch = target["documentPatch"]
        containers = {c["containerNumber"]: c for c in patch.get("containers", [])}
        for number, c in containers.items():
            if {"sizeCategory", "typeCategory"} <= c.keys():
                observations.append(
                    SourceEquipmentObservation(
                        document_id=sid,
                        row_id=number,
                        printed_surface=canonical_equipment_surface(
                            c["sizeCategory"], c["typeCategory"]
                        ),
                        temperature_present="temperatureSetpoint" in c,
                    )
                )
        allocations = _group_allocations(patch)
        for group in patch.get("cargoGroups", []):
            if group.get("dangerousGoods"):
                continue
            numbers = allocations.get(group["groupId"], ())
            if len(patch.get("cargoGroups", [])) == 1:
                numbers = tuple(containers)
            for code in group.get("hsCodes", []):
                heading = re.sub(r"\D", "", code)[:4]
                for number in numbers:
                    if number in containers and "typeCategory" in containers[number]:
                        by_heading[heading].add(containers[number]["typeCategory"])
    return CargoSupport(
        config,
        hs,
        observed_thermal_goods.extend_support(
            build_thermal_goods_support(
                registry=hs,
                ambient_chapters=thermal.ambient_hs_chapters,
                ambient_headings=thermal.ambient_hs_headings,
            ),
            registry=hs,
            fit_targets=fit_targets,
            bounds={
                "FROZEN": (thermal.frozen_minimum_celsius, thermal.frozen_maximum_celsius),
                "CHILLED": (thermal.chilled_minimum_celsius, thermal.chilled_maximum_celsius),
            },
        ),
        packages,
        build_equipment_semantic_support(observations),
        {k: frozenset(v) for k, v in by_heading.items()},
        dg_packaging.supported_registry(dg),
        validation_ids,
        cargo_measurements.build_support(
            fit_targets,
            lower_multiplier=config.measurement_lower_multiplier,
            upper_multiplier=config.measurement_upper_multiplier,
        ),
        {code: ": ".join(hs.global_description_path(code)) for code in hs.global_codes},
        equipment_tares.build_support(fit),
    )


@dataclass(frozen=True)
class CargoScenario:
    target: dict[str, Any]
    identities: dict[str, list[dict[str, Any]]]
    dangerous_goods_facts: tuple[DangerousGoodsFact, ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class EquipmentConstraints:
    template: CertifiedSemanticTemplate
    configured_weights: Mapping[str, int]
    groups: tuple[tuple[int, ...], ...]
    weights: tuple[dict[str, int], ...]
    retained_tare_bindings: tuple[str, ...]
    latent_tare_options: Mapping[int, frozenset[str]]
    row_measurements: tuple[equipment_row_constraints.RowMeasurement, ...]
    row_loads: tuple[equipment_row_constraints.RowLoad, ...] = ()
    tare_constraints: equipment_tares.TareConstraints | None = None
    anonymous_inventory: anonymous_equipment.AnonymousInventory | None = None
    mixed_inventory: mixed_inventory.MixedInventory | None = None
    package_unit_loads: tuple[unit_package_loads.PackageUnitLoad, ...] = ()


def equipment_constraints(
    source: targets.SourceTemplate,
    config: CargoSamplingConfig,
    *,
    numeric_contracts: Mapping[str, numeric_auxiliary.NumericContract] | None = None,
    tare_support: equipment_tares.TareSupport | None = None,
) -> EquipmentConstraints:
    inventory = anonymous_equipment.compile_inventory(source, numeric_contracts or {})
    if inventory is None:
        mixed = mixed_inventory.compile_inventory(source)
        prepared = _equipment_constraints(
            source,
            config,
            numeric_contracts=numeric_contracts,
            tare_support=tare_support,
            ignored_receipt_keys=frozenset(mixed.binding_keys)
            if mixed is not None
            else frozenset(),
        )
        if (
            mixed is not None
            and mixed.witness(
                {
                    i: frozenset(domain)
                    for group, domain in zip(prepared.groups, prepared.weights, strict=True)
                    for i in group
                }
            )
            is None
        ):
            raise ValueError("printed aggregate equipment lacks a configured joint assignment")
        return replace(prepared, mixed_inventory=mixed)
    if set(inventory.equipment_pairs) - config.equipment_joint_weights.keys():
        raise ValueError("anonymous printed equipment is absent from configured joint support")
    prepared = _equipment_constraints(
        inventory.physical_source,
        config,
        numeric_contracts=numeric_contracts,
        tare_support=tare_support,
    )
    # Explicit complete printed types constrain their own physical rows. Count-
    # or length-only receipts retain the normal private-equipment choice; the
    # absent type/height must not become an inferred extraction label.
    weights = []
    for indices, domain in zip(prepared.groups, prepared.weights, strict=True):
        explicit = {inventory.row_pairs[i] for i in indices} - {None}
        if len(explicit) > 1:
            raise ValueError("shared anonymous equipment observations disagree")
        allowed = {
            pair: weight for pair, weight in domain.items() if not explicit or pair in explicit
        }
        if not allowed:
            raise ValueError("anonymous printed inventory has no compatible equipment domain")
        weights.append(allowed)
    return replace(
        prepared,
        template=source.template,
        configured_weights=config.equipment_joint_weights,
        weights=tuple(weights),
        anonymous_inventory=inventory,
    )


def _equipment_constraints(
    source: targets.SourceTemplate,
    config: CargoSamplingConfig,
    *,
    numeric_contracts: Mapping[str, numeric_auxiliary.NumericContract] | None = None,
    tare_support: equipment_tares.TareSupport | None = None,
    ignored_receipt_keys: frozenset[str] = frozenset(),
) -> EquipmentConstraints:
    """Intersect rendering support for all rows sharing one printed size/type."""
    containers = source.target["documentPatch"].get("containers", [])
    numeric_keys = (
        {b.logical_key for b in numeric_auxiliary.numeric_bindings(source.template)}
        if source.template.bindings
        else set()
    )
    contracts = {} if numeric_contracts is None else numeric_contracts
    if numeric_keys != contracts.keys():
        raise ValueError("equipment sampling requires the complete source numeric contracts")
    row_measurements = equipment_row_constraints.compile_rows(source, contracts)
    row_numeric_keys = {r.binding_key for r in row_measurements}
    all_tares = tuple(sorted(k for k, c in contracts.items() if c.role == "tare"))
    fixed_tares = tuple(k for k in all_tares if contracts[k].mode == "source_fixed")
    sampled_tares = tuple(k for k in all_tares if contracts[k].mode == "sampled_equipment_tare")
    if set(all_tares) != set(fixed_tares) | set(sampled_tares):
        raise ValueError("mutable equipment tare requires a physical equipment contract")
    if all_tares and not containers:
        raise ValueError("retained tare requires an observed equipment inventory")
    latent = latent_equipment_indices(source, reviewed_receipt_keys=ignored_receipt_keys)
    tare_options = {}
    tare_constraints = None
    if (fixed_tares and latent) or sampled_tares:
        if tare_support is None:
            raise ValueError("retained tare requires an observed equipment envelope or fit support")
        tare_constraints = equipment_tares.compile_constraints(
            source,
            contracts,
            tare_support,
            latent,
            configured_pairs=frozenset(config.equipment_joint_weights),
        )
        tare_options = tare_constraints.options
    package_loads = unit_package_loads.compile_loads(source, contracts)
    package_load_keys = {load.binding_key for load in package_loads}
    if latent:
        # Derived measurements already constrain the physical scenario through
        # their structured cargo owners. Unowned physical totals still cannot
        # be ignored merely because numeric rendering is possible.
        unresolved = [
            key
            for key, contract in contracts.items()
            if contract.role not in {"operational", "commercial", "metadata"}
            and not (contract.role == "tare" and tare_options)
            and key not in row_numeric_keys
            and key not in package_load_keys
            and contract.mode
            not in {
                "target_sum",
                "target_share",
                "target_average",
                "target_converted",
                "target_equation_count",
                "unit_product",
            }
        ]
        if unresolved:
            raise ValueError(
                "unprinted equipment requires source-only numeric facts in the "
                "physical scenario: " + ", ".join(sorted(unresolved))
            )
    observations = {i: _partial_equipment_constraint(containers[i]) for i in latent}
    parent = list(range(len(containers)))

    def owner(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    bindings = []
    receipts = []
    for binding in source.template.bindings:
        if binding.logical_key in ignored_receipt_keys:
            continue  # The complete multiset contract owns these jointly, not per row.
        if binding.derivation == "equipment_receipt":
            # Compiler receipts can own both an inventory and its projected
            # typeDescription paths. They are composite, not shared scalars.
            render._render_equipment_receipt_binding(
                binding, source_target=source.target, target=source.target
            )
            receipts.append(binding)
            continue
        indices = tuple(
            int(m[1])
            for path in binding.target_paths
            if (m := re.fullmatch(r"documentPatch\.containers\[(\d+)\]\.typeDescription", path))
        )
        if not indices:
            continue
        if len(indices) != len(binding.target_paths):
            raise ValueError("mixed equipment/value binding requires a composite contract")
        if len(indices) > 1 and binding.target_relationship != "shared_value_equality":
            raise ValueError("multi-equipment printed surface requires a composite contract")
        for index in indices[1:]:
            parent[owner(index)] = owner(indices[0])
        bindings.append((binding, indices))
    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(len(containers)):
        grouped[owner(i)].append(i)
    groups = tuple(tuple(v) for v in grouped.values())
    accepted = []
    for indices in groups:
        weights = {}
        for pair, weight in config.equipment_joint_weights.items():
            size, kind = pair.split("|")
            if fixed_tares and any(
                (size, kind) != (containers[i]["sizeCategory"], containers[i]["typeCategory"])
                for i in indices
                if i not in latent
            ):
                continue
            if any(pair not in tare_options[i] for i in indices if i in tare_options):
                continue
            if any(
                not _matches_partial_equipment(size, kind, observations[i])
                for i in indices
                if i in latent
            ):
                continue
            if (
                any("temperatureSetpoint" in containers[i] for i in indices)
                and kind not in TEMPERATURE_CAPABLE_CONTAINER_TYPES
            ):
                continue
            trial = deepcopy(source.target)
            for index in indices:
                trial["documentPatch"]["containers"][index].update(
                    sizeCategory=size, typeCategory=kind
                )
            try:
                for binding, owners in bindings:
                    if not set(indices) & set(owners):
                        continue
                    if set(owners) <= latent:
                        # This binding retains only its observed partial wording.
                        # The hidden dimensions must fit it, not replace it.
                        continue
                    if set(owners) & latent:
                        raise ValueError("shared equipment mixes partial and complete observations")
                    output = render._render_semantic_equipment_binding(binding, trial)
                    render._validate_binding_format(
                        source=source.source, template=source.template.byte_template, output=output
                    )
                for binding in receipts:
                    output = render._render_equipment_receipt_binding(
                        binding, source_target=source.target, target=trial
                    )
                    render._validate_binding_format(
                        source=source.source, template=source.template.byte_template, output=output
                    )
            except ValueError:
                continue  # This pair has no representation; it is not a sampled fallback.
            weights[pair] = weight
        if not weights:
            raise ValueError("equipment group has no configured representable size/type pair")
        accepted.append(weights)
    return EquipmentConstraints(
        source.template,
        dict(config.equipment_joint_weights),
        groups,
        tuple(accepted),
        fixed_tares,
        tare_options,
        row_measurements,
        tare_constraints=tare_constraints,
        package_unit_loads=package_loads,
    )


def _condition_row_capacity(
    source: targets.SourceTemplate,
    target: Mapping[str, Any],
    equipment: EquipmentConstraints,
    config: CargoSamplingConfig,
    numeric_contracts: Mapping[str, numeric_auxiliary.NumericContract] | None,
    sample_id: str,
    seed: int,
) -> EquipmentConstraints:
    if not equipment.row_measurements:
        return equipment
    if numeric_contracts is None:
        raise ValueError("printed container row capacity requires numeric contracts")
    # Tares are sampled after equipment selection. Cargo row conditioning must
    # not fill those absent choices with source values or fabricated placeholders.
    cargo_contracts = {k: c for k, c in numeric_contracts.items() if c.role != "tare"}
    prepared = numeric_auxiliary.prepare(
        tuple(
            b
            for b in numeric_auxiliary.numeric_bindings(source.template)
            if b.logical_key in cargo_contracts
        ),
        cargo_contracts,
        source_target=source.target,
        target=target,
        scale=targets.scenario_scale(sample_id, seed),
        source_template=source.template,
    )
    loads = equipment_row_constraints.prepare_loads(equipment.row_measurements, prepared)
    limits = capacity_limits(config.transport_capacity)
    weights = equipment_row_constraints.condition_weights(
        loads, equipment.groups, equipment.weights, limits
    )
    return replace(equipment, weights=weights, row_loads=loads)


def _matches_partial_equipment(
    size: str, kind: str, observation: tuple[str | None, str | None, str | None]
) -> bool:
    length, observed_kind, observed_size = observation
    prefix = {"20": "TWENTY_", "40": "FORTY_FOOT_", "45": "FORTY_FIVE_"}
    return (
        (observed_size is None or size == observed_size)
        and (length is None or size.startswith(prefix[length]))
        and (observed_kind is None or kind == observed_kind)
    )


def latent_equipment_indices(
    source: targets.SourceTemplate, *, reviewed_receipt_keys: frozenset[str] = frozenset()
) -> frozenset[int]:
    """Separate latent physical equipment from observable extraction fields.

    A document can identify a container without printing its dimensions or type.
    Sampling still needs a physical envelope, but that envelope must never become
    an inferred training label. Reviewed partial wording is retained verbatim
    and constrains the draw; unknown or unowned statements remain review cases.
    """
    missing = frozenset(
        index
        for index, container in enumerate(source.target["documentPatch"].get("containers", []))
        if not {"sizeCategory", "typeCategory"} <= container.keys()
    )
    if not missing:
        return missing
    if not reviewed_receipt_keys:
        mixed = mixed_inventory.compile_inventory(source)
        if mixed is not None:
            reviewed_receipt_keys = frozenset(mixed.binding_keys)
    containers = source.target["documentPatch"]["containers"]
    for index in missing:
        _partial_equipment_constraint(containers[index])
    for binding in source.template.bindings:
        if binding.logical_key in reviewed_receipt_keys:
            continue
        # A source-only equipment statement may cover several rows. Do not infer
        # its owner from proximity or use a latent draw to ignore it.
        if binding.derivation in transport_derivations.DERIVATIONS:
            transport_derivations.validate_source((binding,), source.target)
            continue
        if binding.derivation == "equipment_receipt":
            # Count-only and exactly retained partial descriptions have a
            # complete owner contract without asserting an unseen size/type.
            # The receipt renderer proves the count and every printed type.
            render._render_equipment_receipt_binding(
                binding, source_target=source.target, target=source.target
            )
            continue
        if binding.render_mode != "carrier_static" and (
            binding.value_kind == "equipment" and not binding.target_paths
        ):
            raise ValueError("source-only equipment requires an explicit latent owner")
    return missing


def require_contract(source: targets.SourceTemplate) -> None:
    """Reject unrepresented dependencies before starting any candidate draws."""
    from . import cargo_identity_derivations, shipment_totals

    patch = source.target["documentPatch"]
    groups = patch.get("cargoGroups", [])
    described = [g for g in groups if g.get("description")]
    identity_or_measurement = {"hsCodes", "dangerousGoods", "grossWeight", "netWeight", "volume"}
    if any(not g.get("description") and identity_or_measurement & g.keys() for g in groups) and any(
        g["groupId"] not in {p["groupId"] for p in patch.get("cargoPackages", [])}
        and not identity_or_measurement & g.keys()
        for g in described
    ):
        # A summary carrying the only tariff/packing facts plus separate bare
        # item descriptions cannot be sampled as independent commodities. The
        # source grouping must be reconciled explicitly, not guessed by order.
        raise ValueError("cargo summary and bare item groups require explicit identity ownership")
    if not source.target["documentPatch"].get("cargoGroups") and any(
        b.group_kind == "cargo" and b.render_mode != "carrier_static"
        for b in source.template.bindings
    ):
        raise ValueError("source-only cargo requires a physical owner before equipment sampling")
    if cargo_identity_derivations.unowned_alias_spans(
        source.source.decode(), source.template.bindings
    ):
        raise ValueError("printed HS commodity aliases lack mutable goods ownership")
    shipment_totals.require_owned(source.source, source.template)
    temperature_prose.require_contract(source.template, source.target)
    package_equations.require_segmented_package_owner(source.source, source.template)
    package_equations.require_pallet_mark_contract(source.template, source.target)
    if any(
        g.get("additionalInformation")
        for g in source.target["documentPatch"].get("cargoGroups", [])
    ):
        package_equations.require_numeric_level_coverage(
            source.source, source.template, source.target
        )
    latent_equipment_indices(source)
    package_observations.retained_category_constraints(source.template, source.target)
    if targets.fixed_dimension_paths(source):
        raise ValueError("fixed out-of-gauge dimensions require a joint geometry/goods contract")
    if any(g.get("dangerousGoods") for g in source.target["documentPatch"].get("cargoGroups", [])):
        compile_surfaces(source.template)
    for g in source.target["documentPatch"].get("cargoGroups", []):
        declarations = g.get("dangerousGoods", [])
        if any("flashPoint" in d for d in declarations):
            raise ValueError("DG flashpoint requires a formulation-property contract")
        if declarations and g.get("hsCodes") and len(g["hsCodes"]) != len(declarations):
            raise ValueError("DG/HS cardinality needs an explicit identity association contract")
        if (
            any(
                cargo_identifiers.vehicle_ids(text)
                for text in render._flatten_leaves(g).values()
                if isinstance(text, str)
            )
            or any(
                re.fullmatch(r"[A-HJ-NPR-Z0-9]{11}[0-9]{6}", text.strip(), re.I)
                for text in g.get("marksAndNumbers", [])
            )
            or any(
                p.get("typeCategory") == "PACKAGE_VEHICLE" and p["groupId"] == g["groupId"]
                for p in source.target["documentPatch"].get("cargoPackages", [])
            )
        ):
            raise ValueError("vehicle-specific cargo requires a joint vehicle/goods scenario")
    if not source.target["documentPatch"].get("cargoPackages"):
        # Label missingness is not evidence that printed packaging is absent.
        # Without a latent package owner, a new goods draw cannot establish its
        # compatibility with an immutable source-only count/type declaration.
        nouns = {
            surface for surfaces in render._PACKAGE_SURFACES.values() for surface in surfaces
        } | {
            "BALE",
            "BALES",
            "BUNDLE",
            "BUNDLES",
            "IBC",
            "IBCS",
            "INTERMEDIATE BULK CONTAINER",
            "INTERMEDIATE BULK CONTAINERS",
        }
        pattern = (
            r"(?<![A-Z0-9])(?:[0-9][0-9,]*|ONE|TWO|THREE)[ \t]*(?:"
            + "|".join(re.escape(noun) for noun in sorted(nouns, key=len, reverse=True))
            + r")\b"
        )
        text = source.source.decode("utf-8")
        declarations = (
            match
            for match in re.finditer(pattern, text, re.I)
            # Comparative legal conditions are not shipment-count declarations.
            # Do not bridge a numeric field and the next line's package heading.
            if not re.search(r"\b(?:more|less) than[ \t]+$", text[: match.start()], re.I)
        )
        if next(declarations, None) is not None:
            raise ValueError(
                "printed packaging without target packages requires a latent package contract"
            )


def _printed_hs(old: str, hs6: str, stream: DeterministicStream) -> str:
    digits = re.sub(r"\D", "", old)
    generated = render_synthetic_hs_code(hs6=hs6, output_digits=len(digits), stream=stream)
    return render._shape_alphanumeric_like_source(old, generated)


def _shared_hs_components(template: CertifiedSemanticTemplate) -> dict[tuple[int, int], str]:
    """Only explicit shared printed owners constrain independently indexed goods."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def owner(item: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(item, item)
        while parent[item] != item:
            item = parent[item]
        return item

    for binding in template.bindings:
        paths = [
            re.fullmatch(r"documentPatch\.cargoGroups\[(\d+)\]\.hsCodes\[(\d+)\]", p)
            for p in binding.target_paths
        ]
        if binding.target_relationship != "shared_value_equality" or len(paths) < 2:
            continue
        if not all(paths):
            continue
        positions = [(int(m[1]), int(m[2])) for m in paths if m is not None]
        for position in positions[1:]:
            left, right = sorted((owner(positions[0]), owner(position)))
            parent[right] = left
    return {position: f"shared-hs-{owner(position)[0]}-{owner(position)[1]}" for position in parent}


def _tariff_identity_indices(codes: list[str]) -> tuple[int, ...]:
    """National suffixes do not make distinct six-digit goods classifications.

    Preserve this equivalence within a cargo group while each printed code keeps
    its own width/separators. Full-code equality still needs an explicit shared
    binding; equal HS6 prefixes alone do not authorize sharing national suffixes.
    """
    if not codes:
        return (0,)
    identities: dict[str, int] = {}
    indices = []
    for code in codes:
        digits = re.sub(r"\D", "", code)
        if len(digits) < 6:
            raise ValueError("source tariff classification has fewer than six digits")
        indices.append(identities.setdefault(digits[:6], len(identities)))
    return tuple(indices)


GoodsDomain = frozenset[tuple[tuple[str, ...], str]]


def _has_identity_support(domain: GoodsDomain, count: int) -> bool:
    signatures: dict[tuple[str, ...], int] = Counter(signature for signature, _ in domain)
    return any(value >= count for value in signatures.values())


def _equipment_goods_domains(
    patch: Mapping[str, Any],
    support: CargoSupport,
    measurement_contract: Mapping[str, frozenset[tuple[str, tuple[str, ...]]] | None],
    observed_domains: Mapping[str, tuple[frozenset[str] | None, ...]],
    equipment_contract: EquipmentConstraints,
) -> dict[str, tuple[int, dict[str, GoodsDomain]]]:
    """Finite goods/package domains used to condition equipment before drawing.

    Every equipment draw is conditioned on commodity/package support, including
    singleton groups. Multiple equipment rows maintain the same intersection;
    heterogeneous equipment remains possible whenever that domain supports it.
    """
    groups = patch.get("cargoGroups", [])
    if not groups:
        return {}
    containers = {c["containerNumber"]: c for c in patch.get("containers", [])}
    allocations = _group_allocations(patch)
    if len(groups) == 1:
        allocations = {groups[0]["groupId"]: tuple(containers)}
    owner_by_number = {
        patch["containers"][i]["containerNumber"]: owner
        for owner, indices in enumerate(equipment_contract.groups)
        for i in indices
    }
    groups = [group for group in groups if allocations.get(group["groupId"], ())]
    if not groups:
        return {}
    chronology = patch.get("issueDate") or patch.get("shippedOnBoardDate")
    on_date = date.fromisoformat(chronology) if chronology else support.hs.receipt.snapshot_date
    identities = tuple(
        i
        for i in support.goods.ambient
        if support.hs.require_global(i.hs6, on_date=support.hs.receipt.snapshot_date).valid_from
        <= on_date
        <= support.hs.receipt.snapshot_date
    )
    result = {}
    for group in groups:
        gid = group["groupId"]
        if group.get("dangerousGoods") or any(
            "temperatureSetpoint" in containers[n] for n in allocations.get(gid, ())
        ):
            continue
        packages = [p for p in patch.get("cargoPackages", []) if p["groupId"] == gid]
        count = len(set(_tariff_identity_indices(group.get("hsCodes", []))))
        by_type: dict[str, set[tuple[tuple[str, ...], str]]] = defaultdict(set)
        rows = (
            tuple(
                row
                for row in support.packages.hs_signature_pool_rows
                if row.package_count == len(packages)
                and package_observations.validate_signature(row.categories, observed_domains[gid])
            )
            if packages
            else ()
        )
        allowed = measurement_contract[gid]
        for identity in identities:
            heading = identity.hs6[:4]
            signatures = (
                tuple(
                    row.categories
                    for row in rows
                    if heading in dict(row.heading_occurrences)
                    and (allowed is None or (heading, row.categories) in allowed)
                )
                if packages
                else ((),)
            )
            for kind in support.equipment_types_by_heading.get(heading, ()):
                by_type[kind].update((signature, identity.hs6) for signature in signatures)
        feasible = frozenset().union(*by_type.values())
        for owner in {owner_by_number[n] for n in allocations[gid]}:
            kinds = {pair.split("|")[1] for pair in equipment_contract.weights[owner]}
            feasible &= frozenset().union(*(by_type.get(kind, set()) for kind in kinds))
        if not _has_identity_support(feasible, count):
            raise ValueError("cargo topology has no joint goods/package/equipment domain: " + gid)
        result[gid] = (
            count,
            {kind: frozenset(domain) & feasible for kind, domain in by_type.items()},
        )
    return result


def _sample_candidate(
    source: targets.SourceTemplate,
    proposed: Mapping[str, Any],
    support: CargoSupport,
    stream: DeterministicStream,
    equipment_contract: EquipmentConstraints,
    measurement_contract: Mapping[str, frozenset[tuple[str, tuple[str, ...]]] | None],
    observed_domains: Mapping[str, tuple[frozenset[str] | None, ...]],
    equipment_goods_domains: Mapping[str, tuple[int, dict[str, GoodsDomain]]],
    measurement_quanta: Mapping[str, Mapping[str, Decimal]],
    vary_fit_scale: bool = False,
) -> CargoScenario:
    target = deepcopy(dict(proposed))
    patch = target["documentPatch"]
    cfg = support.config
    packages = defaultdict(list)
    for p in patch.get("cargoPackages", []):
        packages[p["groupId"]].append(p)
    containers = {c["containerNumber"]: c for c in patch.get("containers", [])}
    allocations = _group_allocations(patch)
    if len(patch.get("cargoGroups", [])) == 1:
        allocations = {patch["cargoGroups"][0]["groupId"]: tuple(containers)}
    active = {n for n, c in containers.items() if "temperatureSetpoint" in c}
    if active - {n for values in allocations.values() for n in values}:
        raise ValueError("temperature equipment has no unambiguous cargo-group owner")
    dg_numbers = {
        n
        for g in patch.get("cargoGroups", [])
        if g.get("dangerousGoods")
        for n in allocations.get(g["groupId"], ())
    }
    ambient_types = {
        t
        for identity in support.goods.ambient
        for t in support.equipment_types_by_heading.get(identity.hs6[:4], ())
    }
    sampled_equipment = {}
    represented_types = {}
    remaining_goods = {
        gid: frozenset().union(*by_type.values())
        for gid, (_count, by_type) in equipment_goods_domains.items()
    }
    tare_domains = (
        {
            i: frozenset(domain)
            for indices, domain in zip(
                equipment_contract.groups, equipment_contract.weights, strict=True
            )
            for i in indices
        }
        if equipment_contract.tare_constraints is not None
        or equipment_contract.mixed_inventory is not None
        else {}
    )
    # Draw the equipment envelope first, then draw fit-supported goods/packages
    # conditional on it. Sampling goods first would silently restore the observed
    # near-all-GP marginal despite explicit equipment-balancing weights.
    for indices, represented in zip(
        equipment_contract.groups, equipment_contract.weights, strict=True
    ):
        numbers = tuple(patch["containers"][i]["containerNumber"] for i in indices)
        allowed = (
            set(TEMPERATURE_CAPABLE_CONTAINER_TYPES)
            if set(numbers) & active
            else {"GENERAL_PURPOSE"}
            if set(numbers) & dg_numbers
            else {pair.split("|")[1] for pair in represented}
            if not patch.get("cargoGroups")
            else ambient_types
        )
        weights = {k: v for k, v in represented.items() if k.split("|")[1] in allowed}
        related = [
            gid
            for gid, owners in allocations.items()
            if set(numbers).intersection(owners) and gid in equipment_goods_domains
        ]
        weights = {
            pair: weight
            for pair, weight in weights.items()
            if all(
                _has_identity_support(
                    remaining_goods[gid]
                    & equipment_goods_domains[gid][1].get(pair.split("|")[1], frozenset()),
                    equipment_goods_domains[gid][0],
                )
                for gid in related
            )
        }
        if equipment_contract.tare_constraints is not None:
            weights = {
                pair: weight
                for pair, weight in weights.items()
                if equipment_contract.tare_constraints.witness(
                    {**tare_domains, **{i: frozenset({pair}) for i in indices}}
                )
                is not None
            }
        if equipment_contract.mixed_inventory is not None:
            weights = {
                pair: weight
                for pair, weight in weights.items()
                if equipment_contract.mixed_inventory.witness(
                    {**tare_domains, **{i: frozenset({pair}) for i in indices}}
                )
                is not None
            }
        if not weights:
            raise ValueError("template equipment envelope lacks joint goods support")
        equipment = sample_equipment_semantic(
            support=support.equipment,
            active_temperature=bool(set(numbers) & active),
            stream=stream.derive(numbers[0]),
            configured_joint_weights=weights,
        )
        if (
            equipment_contract.tare_constraints is not None
            or equipment_contract.mixed_inventory is not None
        ):
            for i in indices:
                tare_domains[i] = frozenset(
                    {equipment.size_category + "|" + equipment.type_category}
                )
        for gid in related:
            remaining_goods[gid] &= equipment_goods_domains[gid][1][equipment.type_category]
        for number in numbers:
            sampled_equipment[number] = equipment
            represented_types[number] = {equipment.type_category}
            containers[number].update(
                sizeCategory=equipment.size_category, typeCategory=equipment.type_category
            )
    # Capacity is checked after joint commodity/measurement generation below.
    # The initial numeric proposal is not an accepted physical scenario.
    row_capacity = equipment_row_constraints.validate(
        equipment_contract.row_loads,
        patch.get("containers", []),
        capacity_limits(cfg.transport_capacity),
    )
    private_tares, sampled_tares = (
        equipment_contract.tare_constraints.sample(tare_domains, stream.derive("equipment-tares"))
        if equipment_contract.tare_constraints is not None
        else ({}, {})
    )
    mixed_audit = (
        equipment_contract.mixed_inventory.audit(
            {i: next(iter(pairs)) for i, pairs in tare_domains.items()}
        )
        if equipment_contract.mixed_inventory is not None
        else None
    )
    profile = (
        sample_thermal_profile(
            weights_permyriad=cfg.thermal.profile_weights_permyriad, stream=stream
        )
        if active
        else None
    )
    facts = []
    identities = {}
    used_hs: set[str] = set()
    hs_components = _shared_hs_components(source.template)
    shared_hs: dict[str, dict[str, str]] = {}
    package_receipts = {}
    measurement_receipts = {}
    container_types: dict[str, set[str]] = {}
    for gi, group in enumerate(patch.get("cargoGroups", [])):
        gid = group["groupId"]
        group_stream = stream.derive(gid)
        numbers = allocations.get(gid, ())
        thermal_profile = profile if set(numbers) & active else None
        if thermal_profile and (set(numbers) - active):
            raise ValueError("one thermal cargo group spans active and unconditioned equipment")
        declarations = group.get("dangerousGoods", [])
        rows: list[dict[str, Any]] = []
        if declarations:
            if thermal_profile:
                raise ValueError("temperature-controlled DG needs a chemical-property contract")
            for di, old in enumerate(declarations):
                method: Literal["hs_linked_exact_chemical", "general_regulatory_tuple"] = (
                    "hs_linked_exact_chemical"
                    if group.get("hsCodes")
                    else "general_regulatory_tuple"
                )
                value = support.dg.sample(
                    stream=group_stream.derive(str(di)),
                    method=method,
                    maximum_subsidiary_hazards=len(old.get("subsidiaryHazardCategories", [])),
                )
                if value.hmt_record.technical_name_required or value.hmt_record.nos_entry:
                    raise ValueError("DG generic entry requires a technical-identity contract")
                candidate = value.target.model_dump(mode="json", exclude_none=True)
                if not old.keys() <= candidate.keys():
                    raise ValueError("sampled DG tuple lacks a printed property")
                replacement = {k: candidate[k] for k in old}
                if len(replacement.get("subsidiaryHazardCategories", [])) != len(
                    old.get("subsidiaryHazardCategories", [])
                ):
                    raise ValueError("DG subsidiary cardinality differs from printed topology")
                declarations[di] = replacement
                facts.append(
                    DangerousGoodsFact(
                        target_path=f"documentPatch.cargoGroups[{gi}].dangerousGoods[{di}]",
                        record=value.hmt_record,
                    )
                )
                code = None
                if group.get("hsCodes"):
                    code = _printed_hs(
                        group["hsCodes"][di], value.hs_codes[0], group_stream.derive(f"hs-{di}")
                    )
                    group["hsCodes"][di] = code
                rows.append(
                    dict(
                        code=code,
                        hs6=value.hs_codes[0] if value.hs_codes else None,
                        description=value.hmt_record.proper_shipping_name,
                        properShippingName=value.hmt_record.proper_shipping_name,
                        unNumber=value.hmt_record.un_number,
                        exactHazardClass=value.hmt_record.exact_hazard_class,
                        packingGroup=value.hmt_record.packing_group_code,
                        basis=value.method,
                        hmtRecordId=value.hmt_record.record_id,
                    )
                )
            if packages[gid]:
                signature, receipt = dg_packaging.sample_packages(
                    records=[
                        f.record
                        for f in facts
                        if f.target_path.startswith(f"documentPatch.cargoGroups[{gi}].")
                    ],
                    group=group,
                    packages=packages[gid],
                    stream=group_stream,
                )
                if not package_observations.validate_signature(signature, observed_domains[gid]):
                    raise ValueError(
                        "DG package signature contradicts a retained printed observation"
                    )
                apply_package_signature(target=target, group_id=gid, signature=signature)
                package_receipts[gid] = receipt
            # A packed regulatory scenario does not authorize assigning a bulk tank.
            eligible_types = {"GENERAL_PURPOSE"}
        else:
            existing_codes = group.get("hsCodes", [])
            identity_indices = _tariff_identity_indices(existing_codes)
            count = len(set(identity_indices))
            required: dict[int, str] = {}
            for hi in range(len(existing_codes)):
                key = hs_components.get((gi, hi))
                if key is None or key not in shared_hs:
                    continue
                shared_code = shared_hs[key]["hs6"]
                if required.setdefault(identity_indices[hi], shared_code) != shared_code:
                    raise ValueError("shared HS owners contradict source tariff equivalence")
            chronology = patch.get("issueDate") or patch.get("shippedOnBoardDate")
            on_date = (
                date.fromisoformat(chronology) if chronology else support.hs.receipt.snapshot_date
            )

            def eligible(
                identity: Any,
                *,
                on_date: date = on_date,
                is_thermal: bool = thermal_profile is not None,
                numbers: tuple[str, ...] = numbers,
            ) -> bool:
                registered = support.hs.require_global(
                    identity.hs6, on_date=support.hs.receipt.snapshot_date
                )
                if not registered.valid_from <= on_date <= support.hs.receipt.snapshot_date:
                    return False
                supported_types = support.equipment_types_by_heading.get(
                    identity.hs6[:4], frozenset()
                )
                return is_thermal or all(supported_types & represented_types[n] for n in numbers)

            goods_support = replace(
                support.goods,
                ambient=tuple(i for i in support.goods.ambient if eligible(i)),
                frozen=tuple(i for i in support.goods.frozen if eligible(i)),
                chilled=tuple(i for i in support.goods.chilled if eligible(i)),
            )
            if not (
                goods_support.thermal_candidates(thermal_profile)
                if thermal_profile
                else goods_support.ambient
            ):
                raise ValueError(
                    "no registered goods have date/equipment support for this template"
                )
            if packages[gid]:
                selection = sample_compatible_cargo(
                    support=support.packages,
                    goods_support=goods_support,
                    profile=thermal_profile,
                    package_count=len(packages[gid]),
                    identity_count=count,
                    stream=group_stream,
                    excluded_hs6=used_hs - set(required.values()),
                    allowed_heading_signatures=measurement_contract[gid],
                    allowed_package_categories=observed_domains[gid],
                    required_hs6_by_index=required,
                )
                selected = selection.identities
                apply_package_signature(
                    target=target, group_id=gid, signature=selection.package_signature
                )
                measurement_receipts[gid] = support.measurements.draw_measures(
                    selected[0].hs6,
                    group,
                    packages[gid],
                    quanta=measurement_quanta[gid],
                    container_count=len(numbers) if numbers else None,
                    stream=group_stream,
                    vary_fit_scale=vary_fit_scale,
                )
                if not all(
                    support.measurements.compatible(
                        identity.hs6,
                        group,
                        packages[gid],
                        container_count=len(numbers) if numbers else None,
                    )
                    for identity in selected
                ):
                    raise ValueError("cargo quantity/mass/volume lacks joint goods-package support")
                package_receipts[gid] = dict(
                    basis=selection.basis, supportKey=selection.support_key
                )
                used_hs.update(i.hs6 for i in selected)
            else:
                pool = (
                    goods_support.thermal_candidates(thermal_profile)
                    if thermal_profile
                    else goods_support.ambient
                )
                by_code = {i.hs6: i for i in pool}
                if set(required.values()) - by_code.keys():
                    raise ValueError("shared goods have no compatible equipment/date support")
                selected = tuple(
                    by_code[required[i]]
                    if i in required
                    else sample_thermal_goods(
                        support=goods_support,
                        profile=thermal_profile,
                        stream=group_stream.derive(str(i)),
                    )
                    if thermal_profile
                    else sample_ambient_goods(
                        support=goods_support, stream=group_stream.derive(str(i))
                    )
                    for i in range(count)
                )
                if len({i.hs6 for i in selected}) != count or (used_hs - set(required.values())) & {
                    i.hs6 for i in selected
                }:
                    raise ValueError("sampled cargo identities are not distinct")
                used_hs.update(i.hs6 for i in selected)
            type_sets = []
            for hi, identity_index in enumerate(identity_indices):
                identity = selected[identity_index]
                code = None
                if existing_codes:
                    shared_key = hs_components.get((gi, hi))
                    code = (
                        shared_hs[shared_key]["code"]
                        if shared_key in shared_hs
                        else _printed_hs(
                            existing_codes[hi], identity.hs6, group_stream.derive(f"hs-{hi}")
                        )
                    )
                    if shared_key is not None:
                        shared_hs[shared_key] = {"hs6": identity.hs6, "code": code}
                    existing_codes[hi] = code
                rows.append(
                    dict(
                        code=code,
                        hs6=identity.hs6,
                        description=identity.description,
                        heading=identity.heading_description,
                        requiredDescription=support.descriptions[identity.hs6],
                        thermalProfile=thermal_profile,
                        observedSetpointsCelsius=list(identity.observed_setpoints_celsius)
                        if isinstance(identity, ThermalGoodsIdentity)
                        else [],
                        thermalFitDocumentIds=list(identity.fit_document_ids)
                        if isinstance(identity, ThermalGoodsIdentity)
                        else [],
                        basis="fit_package_signature_and_registry_identity",
                        tariffSuffixPolicy="synthetic_shape_preserving_not_official_national_tariff",
                    )
                )
                if numbers and not thermal_profile:
                    available_types = support.equipment_types_by_heading.get(identity.hs6[:4])
                    if not available_types:
                        raise ValueError("sampled heading lacks equipment compatibility support")
                    type_sets.append(set(available_types))
            eligible_types = (
                set(TEMPERATURE_CAPABLE_CONTAINER_TYPES)
                if thermal_profile
                else set.intersection(*type_sets)
                if type_sets
                else set()
            )
        identities[gid] = rows
        for number in numbers:
            container_types[number] = (
                container_types[number] & eligible_types
                if number in container_types
                else set(eligible_types)
            )
    equipment_receipts: dict[str, dict[str, Any]] = {}
    temperature_bounds = temperature_prose.celsius_bounds(source.template, source.target)
    temperature_owners = temperature_prose.shared_setpoint_owners(source.template, source.target)
    shared_temperatures: dict[int, float] = {}
    for owners in sorted(set(temperature_owners.values())):
        if len(owners) < 2:
            continue
        if profile is None:
            raise ValueError("shared carrying temperature lacks a thermal goods profile")
        low, high = (
            (cfg.thermal.frozen_minimum_celsius, cfg.thermal.frozen_maximum_celsius)
            if profile == "FROZEN"
            else (cfg.thermal.chilled_minimum_celsius, cfg.thermal.chilled_maximum_celsius)
        )
        generic = tuple(
            low + i * cfg.thermal.step_celsius
            for i in range(round((high - low) / cfg.thermal.step_celsius) + 1)
        )
        domains = []
        for index in owners:
            number = tuple(containers)[index]
            observed = observed_thermal_goods.shared_setpoints(
                [
                    identity
                    for gid, values in identities.items()
                    if number in allocations.get(gid, ())
                    for identity in values
                ]
            )
            floor, ceiling = temperature_bounds.get(index, (low, high))
            domains.append(
                {v for v in observed if floor <= v <= ceiling}
                if observed is not None
                else {v for v in generic if floor <= v <= ceiling}
            )
        joint_temperatures = sorted(set.intersection(*domains))
        if not joint_temperatures:
            raise ValueError("cargo temperature instruction has no jointly compatible setpoint")
        shared_celsius = joint_temperatures[
            stream.derive("shared-carrying-temperature:" + str(owners)).randbelow(
                len(joint_temperatures)
            )
        ]
        shared_temperatures.update(dict.fromkeys(owners, shared_celsius))
    for container_index, (number, container) in enumerate(containers.items()):
        if not {"sizeCategory", "typeCategory"} <= container.keys():
            raise ValueError("container semantic shape is incomplete")
        equipment = sampled_equipment[number]
        if patch.get("cargoGroups") and equipment.type_category not in container_types.get(
            number, set()
        ):
            raise ValueError("sampled equipment and cargo identities lack joint fit support")
        container.update(sizeCategory=equipment.size_category, typeCategory=equipment.type_category)
        if number in active:
            assert profile is not None
            setting = sample_temperature_setpoint(
                profile=profile,
                stream=stream.derive(number),
                frozen_minimum_celsius=cfg.thermal.frozen_minimum_celsius,
                frozen_maximum_celsius=cfg.thermal.frozen_maximum_celsius,
                chilled_minimum_celsius=cfg.thermal.chilled_minimum_celsius,
                chilled_maximum_celsius=cfg.thermal.chilled_maximum_celsius,
                step_celsius=cfg.thermal.step_celsius,
            )
            observed = observed_thermal_goods.shared_setpoints(
                [
                    identity
                    for gid, values in identities.items()
                    if number in allocations.get(gid, ())
                    for identity in values
                ]
            )
            celsius = (
                observed[stream.derive(number).derive("fit-setpoint").randbelow(len(observed))]
                if observed is not None
                else setting.value
            )
            if container_index in temperature_bounds:
                low, high = temperature_bounds[container_index]
                profile_low, profile_high = (
                    (cfg.thermal.frozen_minimum_celsius, cfg.thermal.frozen_maximum_celsius)
                    if profile == "FROZEN"
                    else (cfg.thermal.chilled_minimum_celsius, cfg.thermal.chilled_maximum_celsius)
                )
                candidates = (
                    observed
                    if observed is not None
                    else tuple(
                        profile_low + i * cfg.thermal.step_celsius
                        for i in range(
                            round((profile_high - profile_low) / cfg.thermal.step_celsius) + 1
                        )
                    )
                )
                feasible = tuple(v for v in candidates if low <= v <= high)
                if not feasible:
                    raise ValueError(
                        "sampled goods have no setpoint inside the printed storage interval"
                    )
                celsius = feasible[
                    stream.derive(number).derive("interval-setpoint").randbelow(len(feasible))
                ]
            if container_index in shared_temperatures:
                celsius = shared_temperatures[container_index]
            surface = container["temperatureSetpoint"]
            unit = surface["unit"]
            surface["value"] = (
                celsius
                if unit == "celsius"
                else celsius * 1.8 + 32
                if unit == "fahrenheit"
                else celsius + 273.15
                if unit == "kelvin"
                else None
            )
            if surface["value"] is None:
                raise ValueError("unsupported temperature unit")
        equipment_receipts[number] = dict(
            size=equipment.size_category,
            type=equipment.type_category,
            basis=(
                "fit_heading_types_with_configured_size_type_weights"
                if patch.get("cargoGroups")
                else "configured_equipment_without_observed_cargo"
            ),
            labelVisibility=(
                "observed_size_type"
                if {"sizeCategory", "typeCategory"}
                <= source.target["documentPatch"]["containers"][container_index].keys()
                else "latent_physical_envelope_not_in_labels"
            ),
        )
        latent = (
            equipment_receipts[number]["labelVisibility"]
            == "latent_physical_envelope_not_in_labels"
        )
        if equipment_contract.retained_tare_bindings and (
            not latent or container_index in equipment_contract.latent_tare_options
        ):
            equipment_receipts[number]["retainedTareBindings"] = list(
                equipment_contract.retained_tare_bindings
            )
            equipment_receipts[number]["envelopeConstraintReason"] = (
                "source_observed_tare_and_partial_equipment_with_private_missing_dimensions"
                if equipment_contract.tare_constraints is not None
                and container_index
                in equipment_contract.tare_constraints.source_observed_partial_owners
                else "printed_fixed_tare_requires_exact_train_only_joint_equipment_support"
                if latent
                else "printed_fixed_tare_requires_source_equipment_envelope"
            )
            if latent:
                equipment_receipts[number]["tareSupportedSizeTypePairs"] = sorted(
                    equipment_contract.latent_tare_options[container_index]
                )
        if "typeDescription" in container:
            equipment_receipts[number]["retainedPrintedDescription"] = container["typeDescription"]
            equipment_receipts[number]["retentionReason"] = (
                "partial_observation_constrains_hidden_equipment_without_inventing_labels"
            )
    private_net = measurement_prose.private_net_weights(source.target, proposed)
    physical_target = target
    if private_net:
        physical_target = {
            **target,
            "documentPatch": {
                **patch,
                "cargoGroups": [
                    measurement_prose.physical_group(g, private_net.get(g["groupId"]))
                    for g in patch.get("cargoGroups", ())
                ],
            },
        }
        for group in physical_target["documentPatch"]["cargoGroups"]:
            gid = group["groupId"]
            if gid in private_net and not all(
                support.measurements.compatible(
                    identity["hs6"],
                    group,
                    packages[gid],
                    container_count=len(allocations[gid]) if allocations.get(gid) else None,
                )
                for identity in identities[gid]
            ):
                raise ValueError("private unit net mass lacks joint goods-package support")
    package_floors = unit_package_loads.lower_bounds(equipment_contract.package_unit_loads, target)
    unit_package_loads.validate_gross(package_floors, target)
    if package_floors:
        # A printed quantity x package mass proves a payload floor, not an
        # unprinted extraction netWeight. Use it only for physical validation.
        physical_target = deepcopy(physical_target)
        for group in physical_target["documentPatch"].get("cargoGroups", ()):
            package_mass_floor = package_floors.get(group["groupId"])
            if package_mass_floor is not None and "grossWeight" not in group:
                group["grossWeight"] = {"value": float(package_mass_floor), "unit": "kilogram"}
    capacity = document_capacity_receipt(physical_target, capacity_limits(cfg.transport_capacity))
    if not capacity.valid:
        raise ValueError("sampled equipment capacity is insufficient: " + str(capacity.violations))
    for gid, rows in packages.items():
        if not rows:
            continue  # This group has no printed package rows or sampled signature.
        package_receipts[gid]["physicalSignature"] = [p["typeCategory"] for p in rows]
        package_receipts[gid]["labelVisibility"] = [
            "observed_category" if "typeCategory" in p else "latent_not_in_labels"
            for p in source.target["documentPatch"].get("cargoPackages", [])
            if p["groupId"] == gid
        ]
    target = package_observations.observable_target(source.target, target)
    # Verify after package signatures have been sampled, not against the initial
    # numeric proposal. Reject incompatible draws locally before paid generation.
    for equation_path, equation_text in package_equations.generate(source.target, target).items():
        if render._resolve_path(target, equation_path) != equation_text:
            raise ValueError("sampled cargo changed a host-owned package equation")
    package_equations.mark_ranges(source.target, target)
    if facts:
        render_facts(
            source=source.source, template=source.template, target=target, facts=tuple(facts)
        )
    _validate_sampled_bindings(source, target)
    return CargoScenario(
        _observable_equipment_target(source, target),
        identities,
        tuple(facts),
        dict(
            packageSupport=package_receipts,
            measurementSupport=measurement_receipts,
            equipment=equipment_receipts,
            capacity=capacity.to_dict(),
            printedContainerRows=row_capacity,
            privateEquipmentTaresKg={str(i): str(v) for i, v in private_tares.items()},
            sampledEquipmentTares={k: str(v) for k, v in sampled_tares.items()},
            privateEquipmentTareUnitBindings=(
                list(equipment_contract.tare_constraints.private_unit_bindings)
                if equipment_contract.tare_constraints is not None
                else []
            ),
            sourceObservedPartialTareOwners=(
                list(equipment_contract.tare_constraints.source_observed_partial_owners)
                if equipment_contract.tare_constraints is not None
                else []
            ),
            mixedEquipmentInventory=mixed_audit,
            privatePackageMassFloorKg={k: str(v) for k, v in package_floors.items()},
            privateCargoNetKg={k: str(v) for k, v in private_net.items()},
            thermalProfile=profile,
            sharedTemperatureOwners=[list(v) for v in sorted(set(temperature_owners.values()))],
        ),
    )


def _validate_sampled_bindings(source: targets.SourceTemplate, target: Mapping[str, Any]) -> None:
    """Use final-renderer adapters for changed scalar facts before linguistic calls."""
    for binding in source.template.bindings:
        if not binding.target_paths or not any(
            render._binding_target_value(source.target, p)
            != render._binding_target_value(target, p)
            for p in binding.target_paths
        ):
            continue
        if any(".dangerousGoods[" in p for p in binding.target_paths):
            continue  # All these surfaces were checked against the registry tuple above.
        if binding.derivation == "equipment_receipt":
            output = render._render_equipment_receipt_binding(
                binding, source_target=source.target, target=target
            )
        elif render._has_semantic_equipment_values(binding, target):
            output = render._render_semantic_equipment_binding(binding, target)
        elif all(
            ".hsCodes[" in p
            or p.endswith(".typeCategory")
            or ".temperatureSetpoint" in p
            or any(
                p.endswith("." + field + ".value")
                for field in cargo_measurements.MEASUREMENT_FIELDS
            )
            for p in binding.target_paths
        ):
            measurement_output = render._render_target_measurement(
                binding,
                cast(
                    render.PreparedCase,
                    SimpleNamespace(
                        source=source.source,
                        source_target=source.target,
                        target=target,
                        template=source.template,
                    ),
                ),
            )
            output = measurement_output or (
                render._render_agent_target_binding(
                    binding, source_target=source.source_target, target=target
                )
                if binding.realization.requires_agent
                else render._render_target_binding(
                    binding, target, source=source.source, template=source.template
                )
            )
        else:
            continue
        render._validate_binding_format(
            source=source.source, template=source.template.byte_template, output=output
        )


def _observable_equipment_target(
    source: targets.SourceTemplate, target: dict[str, Any]
) -> dict[str, Any]:
    """Project a validated physical scenario back to the source's label visibility.

    The caller owns this target. Capacity and compatibility receipts were already
    computed using the complete physical scenario, which remains in the receipt.
    Only deliberately latent fields are removed; generated visible facts stay.
    """
    for index in latent_equipment_indices(source):
        container = target["documentPatch"]["containers"][index]
        observed = source.target["documentPatch"]["containers"][index]
        if container.get("typeDescription") != observed.get("typeDescription"):
            raise ValueError("partial equipment wording changed outside its observation contract")
        if not _matches_partial_equipment(
            container["sizeCategory"],
            container["typeCategory"],
            _partial_equipment_constraint(observed),
        ):
            raise ValueError("latent physical equipment contradicts the printed observation")
        del container["sizeCategory"]
        del container["typeCategory"]
    return target


def _measurement_quanta(
    source: targets.SourceTemplate,
    contracts: Mapping[str, numeric_auxiliary.NumericContract],
    equipment: EquipmentConstraints,
) -> dict[str, dict[str, Decimal]]:
    """Decide numeric freedom BEFORE drawing a scenario, never restore a value.

    Independent source-only physical observations and explicit row loads retain
    the shared source scaling until they have a complete joint dependency model.
    Typed equations lock their exact dependent measurement, not unrelated fields.
    """
    unowned_physics = any(
        c.role in {"cargo_mass", "cargo_volume", "per_unit_measurement", "density", "dimensions"}
        and c.mode in {"source_scaled", "source_fixed", "surface_fixed"}
        for c in contracts.values()
    )
    locked = {
        path for c in contracts.values() if c.mode == "unit_product" for path in c.target_paths
    }
    locked.update(measurement_prose.per_package_totals(source.target, source.target))
    # A packing-weight equation can live in a package category surface, not
    # only cargo prose. Both are already enforced by structured_proposal; keep
    # those derived totals out of the later joint-measurement draw as well.
    locked.update(package_prose.package_mass_values(source.template, source.target, source.target))
    partition_steps = numeric_auxiliary.equal_partition_steps(contracts, source.template)
    result = {}
    for index, group in enumerate(source.target.get("documentPatch", {}).get("cargoGroups", ())):
        eligible = (
            not unowned_physics
            and not equipment.row_measurements
            and not group.get("dangerousGoods")
        )
        result[group["groupId"]] = {
            field: targets._numeric_quantum(
                source, f"documentPatch.cargoGroups[{index}].{field}.value", group[field]["value"]
            )
            for field in sorted(cargo_measurements.MEASUREMENT_FIELDS & group.keys())
            if eligible and f"documentPatch.cargoGroups[{index}].{field}.value" not in locked
        }
        for field, quantum in result[group["groupId"]].items():
            path = f"documentPatch.cargoGroups[{index}].{field}.value"
            if path in partition_steps:
                steps = (Fraction(quantum), Fraction(partition_steps[path]))
                result[group["groupId"]][field] = Decimal(
                    lcm(*(s.numerator for s in steps))
                ) / Decimal(gcd(*(s.denominator for s in steps)))
    return result


def sample_scenario(
    source: targets.SourceTemplate,
    proposed: Mapping[str, Any],
    *,
    support: CargoSupport,
    sample_id: str,
    seed: int,
    prepared_equipment: EquipmentConstraints | None = None,
    numeric_contracts: Mapping[str, numeric_auxiliary.NumericContract] | None = None,
    validate_lexical: Callable[[CargoScenario], None] | None = None,
) -> CargoScenario:
    if source.document_id in support.validation_ids:
        raise ValueError("validation source cannot enter cargo synthesis")
    # Keep the ordinary sampler path unchanged. Anonymous-fleet preparation is
    # needed only when an actual equipment statement has no labelled inventory.
    if (prepared_equipment is not None and prepared_equipment.anonymous_inventory is None) or (
        prepared_equipment is None
        and (
            source.target["documentPatch"].get("containers")
            or not any(
                b.value_kind == "equipment"
                and not b.target_paths
                and b.render_mode != "carrier_static"
                for b in source.template.bindings
            )
        )
    ):
        return _sample_scenario(
            source,
            proposed,
            support=support,
            sample_id=sample_id,
            seed=seed,
            prepared_equipment=prepared_equipment,
            numeric_contracts=numeric_contracts,
            validate_lexical=validate_lexical,
        )
    equipment = prepared_equipment
    if equipment is None:
        equipment = equipment_constraints(
            source, support.config, numeric_contracts=numeric_contracts, tare_support=support.tares
        )
    if (
        equipment.template is not source.template
        or equipment.configured_weights != support.config.equipment_joint_weights
    ):
        raise ValueError("prepared equipment constraints differ from source/configuration")
    inventory = equipment.anonymous_inventory
    if inventory is None:
        return _sample_scenario(
            source,
            proposed,
            support=support,
            sample_id=sample_id,
            seed=seed,
            prepared_equipment=equipment,
            numeric_contracts=numeric_contracts,
            validate_lexical=validate_lexical,
        )
    if inventory.source is not source:
        raise ValueError("anonymous physical inventory differs from the pinned source")
    internal_equipment = replace(
        equipment,
        template=inventory.physical_source.template,
        anonymous_inventory=None,
    )

    def validate_observable(candidate: CargoScenario) -> None:
        # Validation sees the same public visibility as final publication. The
        # sampler's physical target stays intact for all compatibility checks.
        observable = deepcopy(candidate.target)
        audit = inventory.observable(observable)
        projected = replace(
            candidate, target=observable, receipt={**candidate.receipt, "anonymousEquipment": audit}
        )
        if validate_lexical is not None:
            validate_lexical(projected)

    candidate = _sample_scenario(
        inventory.physical_source,
        inventory.proposed(proposed, sample_id=sample_id, seed=seed),
        support=support,
        sample_id=sample_id,
        seed=seed,
        prepared_equipment=internal_equipment,
        numeric_contracts=numeric_contracts,
        validate_lexical=validate_observable,
    )
    candidate.receipt["anonymousEquipment"] = inventory.observable(candidate.target)
    candidate.receipt["sourceTargetSha256"] = sha256_bytes(canonical_json_bytes(source.target))
    return candidate


def _sample_scenario(
    source: targets.SourceTemplate,
    proposed: Mapping[str, Any],
    *,
    support: CargoSupport,
    sample_id: str,
    seed: int,
    prepared_equipment: EquipmentConstraints | None = None,
    numeric_contracts: Mapping[str, numeric_auxiliary.NumericContract] | None = None,
    validate_lexical: Callable[[CargoScenario], None] | None = None,
) -> CargoScenario:
    if source.document_id in support.validation_ids:
        raise ValueError("validation source cannot enter cargo synthesis")
    require_contract(source)
    if prepared_equipment is not None and (
        prepared_equipment.template is not source.template
        or prepared_equipment.configured_weights != support.config.equipment_joint_weights
    ):
        raise ValueError("prepared equipment constraints differ from source/configuration")
    equipment_contract = (
        prepared_equipment
        if prepared_equipment is not None
        else equipment_constraints(
            source, support.config, numeric_contracts=numeric_contracts, tare_support=support.tares
        )
    )
    equipment_contract = _condition_row_capacity(
        source,
        proposed,
        equipment_contract,
        support.config,
        numeric_contracts,
        sample_id,
        seed,
    )
    failures: Counter[str] = Counter()
    patch = proposed["documentPatch"]
    fixed_categories = package_equations.required_package_categories(source.target)
    private_net = measurement_prose.private_net_weights(source.target, proposed)
    if any(
        g.get("dangerousGoods") and g["groupId"] in private_net
        for g in patch.get("cargoGroups", ())
    ):
        raise ValueError("private DG packing mass requires formulation-specific fit support")
    for i, package in enumerate(patch.get("cargoPackages", ())):
        if package["groupId"] in private_net and "typeCategory" in package:
            previous = fixed_categories.setdefault(i, package["typeCategory"])
            if previous != package["typeCategory"]:
                raise ValueError("private unit mass contradicts package equation kind")
    observed_package_categories = package_observations.retained_category_constraints(
        source.template, source.target
    )
    if any(
        i in fixed_categories and fixed_categories[i] != category
        for i, category in observed_package_categories.items()
    ):
        raise ValueError("source packing equation contradicts its retained package observation")
    fixed_categories.update(observed_package_categories)
    package_domains: dict[str, list[frozenset[str] | None]] = defaultdict(list)
    for i, package in enumerate(patch.get("cargoPackages", [])):
        domain = package_observations.category_domains(
            [package], frozenset(support.packages.allowed_category_tokens)
        )[0]
        if i in fixed_categories:
            fixed = frozenset({fixed_categories[i]})
            domain = fixed if domain is None else domain & fixed
            if not domain:
                raise ValueError("package observation contradicts its printed packing equation")
        package_domains[package["groupId"]].append(domain)
    observed_domains = {k: tuple(v) for k, v in package_domains.items()}
    container_counts = cargo_measurements.allocated_container_counts(patch)
    measurement_contract = {}
    measurement_quanta = _measurement_quanta(source, numeric_contracts or {}, equipment_contract)
    for group in patch.get("cargoGroups", []):
        if group.get("dangerousGoods"):
            continue
        packages = [p for p in patch.get("cargoPackages", []) if p["groupId"] == group["groupId"]]
        allowed = support.measurements.allowed_heading_signatures(
            measurement_prose.physical_group(group, private_net.get(group["groupId"])),
            packages,
            container_count=container_counts.get(group["groupId"]),
            mutable_fields=frozenset(measurement_quanta[group["groupId"]]),
        )
        if allowed == frozenset():
            raise ValueError("cargo quantity/mass/volume has no train-only compatibility support")
        measurement_contract[group["groupId"]] = allowed
    equipment_goods_domains = _equipment_goods_domains(
        patch, support, measurement_contract, observed_domains, equipment_contract
    )
    for attempt in range(support.config.maximum_candidates):
        try:
            scenario = _sample_candidate(
                source,
                proposed,
                support,
                DeterministicStream(seed, "compiled-joint-cargo-v1", sample_id).derive(
                    str(attempt)
                ),
                equipment_contract,
                measurement_contract,
                observed_domains,
                equipment_goods_domains,
                measurement_quanta,
                # Preserve the inexpensive exact-fit first draw. A rejected
                # candidate is not accepted data: subsequent draws may explore
                # the already configured joint measurement support. Locked
                # equations and every final physical check remain in force.
                attempt > 0,
            )
            if numeric_contracts:
                prepared = numeric_auxiliary.prepare(
                    numeric_auxiliary.numeric_bindings(source.template),
                    numeric_contracts,
                    source_target=source.target,
                    target=scenario.target,
                    scale=targets.scenario_scale(sample_id, seed),
                    source_template=source.template,
                    equipment_tare_values={
                        k: Decimal(v) for k, v in scenario.receipt["sampledEquipmentTares"].items()
                    },
                )
                equipment_row_constraints.validate_prepared(
                    scenario.receipt["printedContainerRows"], prepared
                )
            if validate_lexical is not None:
                validate_lexical(scenario)
        except ValueError as error:
            failures[str(error)] += 1
        else:
            scenario.receipt.update(
                candidateAttempts=attempt + 1,
                rejectedCandidates=dict(failures),
                sourceTargetSha256=sha256_bytes(canonical_json_bytes(source.target)),
                retainedPackageCategories={
                    str(i): category for i, category in observed_package_categories.items()
                },
                retainedPackageBindings=[
                    b.logical_key
                    for b in source.template.bindings
                    if package_observations.unowned_package_observation(b)
                ],
            )
            return scenario
    raise ValueError("no representable joint cargo scenario: " + str(dict(failures)))


def validate_structured_facts(scenario: CargoScenario, target: Mapping[str, Any]) -> None:
    expected = render._flatten_leaves(scenario.target)
    actual = render._flatten_leaves(target)
    for path in actual.keys() | expected.keys():
        if re.fullmatch(
            r"documentPatch\.containers\[\d+\]\.(?:sizeCategory|typeCategory|typeDescription)",
            path,
        ) and (path in actual) != (path in expected):
            raise ValueError(f"equipment label visibility changed during generation: {path}")
        if re.fullmatch(
            r"documentPatch\.cargoPackages\[\d+\]\.(?:typeCategory|typeDescription)", path
        ) and (path in actual) != (path in expected):
            raise ValueError(f"package label visibility changed during generation: {path}")
    for path, value in expected.items():
        if (
            ".dangerousGoods[" in path
            or ".hsCodes[" in path
            or ".temperatureSetpoint" in path
            or path.endswith((".sizeCategory", ".typeCategory"))
            or re.fullmatch(r"documentPatch\.containers\[\d+\]\.typeDescription", path)
            or re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.typeDescription", path)
        ) and actual.get(path) != value:
            raise ValueError(f"sampled cargo fact changed during linguistic generation: {path}")


def validate_final(scenario: CargoScenario, target: Mapping[str, Any]) -> None:
    validate_structured_facts(scenario, target)
    observed_thermal_goods.validate_setpoints(target, scenario.identities)
    for group in target["documentPatch"].get("cargoGroups", []):
        declared_un = {d["unNumber"] for d in group.get("dangerousGoods", [])}
        for text in render._flatten_leaves(group).values():
            if not isinstance(text, str):
                continue
            validate_un_references(text, declared_un)
        description = group.get("description")
        if description is None:
            continue
        words = " ".join(re.findall(r"[a-z0-9]+", description.casefold()))
        for identity in scenario.identities[group["groupId"]]:
            required = (
                identity["properShippingName"]
                if group.get("dangerousGoods")
                else identity["requiredDescription"]
            )
            proper_name = " ".join(re.findall(r"[a-z0-9]+", required.casefold()))
            if not proper_name or proper_name not in words:
                raise ValueError(
                    "generated cargo description omits its authoritative goods description "
                    "or proper shipping name: " + required
                )
