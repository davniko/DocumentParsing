"""Fit-isolated, registry-backed container size/type scenarios.

The sampled identity is an exact BIC/ISO 6346 four-character size/type code.
Printed document wording is deliberately separate and remains pending for the
later text renderer.  Carrier aliases are never interpreted or emitted here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from document_ocr.synthesis.equipment_registry import (
    EquipmentIdentity,
    EquipmentRegistryError,
    EquipmentRegistryReceipt,
    ThermalCapability,
)
from document_ocr.synthesis.generators import DeterministicStream

EquipmentSourceResolution = Literal[
    "exact_registry_code",
    "unclassified_printed_surface",
    "untyped",
]
EquipmentSamplingComponent = Literal["fit_empirical", "registry_exploration"]


@dataclass(frozen=True, slots=True)
class EquipmentFitObservation:
    """One fit-scoped equipment row without alias-derived semantics."""

    source_document_id: str
    equipment_row_id: str
    printed_surface: str | None
    exact_registry_code: str | None
    temperature_setpoint_present: bool

    def __post_init__(self) -> None:
        if not self.source_document_id or not self.equipment_row_id:
            raise ValueError("equipment observations require non-empty source identities")
        if self.printed_surface is not None and (
            not self.printed_surface or self.printed_surface != self.printed_surface.strip()
        ):
            raise ValueError("equipment printed surfaces must be non-empty without outer space")


@dataclass(frozen=True, slots=True)
class EquipmentCodeSupport:
    identity: EquipmentIdentity
    occurrences: int
    document_count: int


@dataclass(frozen=True, slots=True)
class EquipmentScenarioAudit:
    fit_document_count: int
    input_observations: int
    exact_registry_observations: int
    unclassified_surface_observations: int
    absent_surface_observations: int
    temperature_observations: int
    observed_size_type_codes: int
    observed_size_codes: int
    observed_type_families: int


@dataclass(frozen=True, slots=True)
class EquipmentScenarioSupport:
    registry_manifest_sha256: str
    fit_document_ids: tuple[str, ...]
    observed_codes: tuple[EquipmentCodeSupport, ...]
    audit: EquipmentScenarioAudit
    _fit_document_id_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fit_set = frozenset(self.fit_document_ids)
        if not fit_set or len(fit_set) != len(self.fit_document_ids):
            raise ValueError("equipment support fit document IDs must be non-empty and unique")
        object.__setattr__(self, "_fit_document_id_set", fit_set)

    def includes_fit_document(self, document_id: str) -> bool:
        return document_id in self._fit_document_id_set


@dataclass(frozen=True, slots=True)
class EquipmentTypeScenario:
    source_document_id: str
    equipment_row_id: str
    source_resolution: EquipmentSourceResolution
    sampling_component: EquipmentSamplingComponent
    size_type_code: str
    size_code: str
    type_code: str
    type_family: str
    thermal_capability: ThermalCapability
    supports_temperature_setpoint: bool
    printed_surface: None
    printed_surface_status: Literal["pending_text_realization"]
    form_projection: Literal["exact_container_code"]
    training_type_category_status: Literal["not_in_current_empty_task_vocabulary"]


def build_equipment_scenario_support(
    *,
    observations: Sequence[EquipmentFitObservation],
    fit_document_ids: Sequence[str],
    registry: EquipmentRegistryReceipt,
) -> EquipmentScenarioSupport:
    """Compile exact fit support while leaving unclassified wording untouched."""

    fit_ids = tuple(fit_document_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise ValueError("fit document IDs must be non-empty and unique")
    fit_set = frozenset(fit_ids)
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    documents: dict[str, set[str]] = {}
    unclassified = absent = temperature = 0
    for row in observations:
        if row.source_document_id not in fit_set:
            raise ValueError("equipment observation lies outside the fit partition")
        key = (row.source_document_id, row.equipment_row_id)
        if key in seen:
            raise ValueError(f"duplicate equipment observation: {key!r}")
        seen.add(key)
        temperature += int(row.temperature_setpoint_present)
        if row.exact_registry_code is None:
            if row.printed_surface is None:
                absent += 1
            else:
                unclassified += 1
            continue
        identity = registry.classify(row.exact_registry_code)
        if row.temperature_setpoint_present and not identity.type.supports_temperature_setpoint:
            raise ValueError(
                "temperature-bearing equipment has a non-thermal exact registry identity"
            )
        counts[identity.size_type_code] += 1
        documents.setdefault(identity.size_type_code, set()).add(row.source_document_id)
    if not counts:
        raise ValueError("fit partition has no exact BIC/ISO equipment identities")
    observed = tuple(
        EquipmentCodeSupport(
            identity=registry.classify(code),
            occurrences=count,
            document_count=len(documents[code]),
        )
        for code, count in sorted(counts.items())
    )
    return EquipmentScenarioSupport(
        registry_manifest_sha256=registry.manifest_sha256,
        fit_document_ids=fit_ids,
        observed_codes=observed,
        audit=EquipmentScenarioAudit(
            fit_document_count=len(fit_ids),
            input_observations=len(observations),
            exact_registry_observations=sum(counts.values()),
            unclassified_surface_observations=unclassified,
            absent_surface_observations=absent,
            temperature_observations=temperature,
            observed_size_type_codes=len(counts),
            observed_size_codes=len({row.identity.size_code for row in observed}),
            observed_type_families=len({row.identity.type.family_code for row in observed}),
        ),
    )


def sample_equipment_type(
    observation: EquipmentFitObservation,
    *,
    support: EquipmentScenarioSupport,
    registry: EquipmentRegistryReceipt,
    registry_exploration_permyriad: int,
    stream: DeterministicStream,
) -> EquipmentTypeScenario:
    """Sample an exact code, preserving explicit thermal requirements.

    The empirical arm samples exact fit-observed codes.  The exploration arm
    composes a fit-observed size with an assigned BIC type from a fit-observed
    semantic family.  Thus it expands within proven structural support without
    interpreting any source alias.
    """

    if not support.includes_fit_document(observation.source_document_id):
        raise ValueError("equipment sampling source lies outside the fit partition")
    if not 0 <= registry_exploration_permyriad <= 10_000:
        raise ValueError("registry exploration weight must be within [0, 10000]")
    if registry.manifest_sha256 != support.registry_manifest_sha256:
        raise ValueError("equipment registry differs from the support receipt")
    exact_source: EquipmentIdentity | None = None
    if observation.exact_registry_code is not None:
        exact_source = registry.classify(observation.exact_registry_code)
    source_resolution: EquipmentSourceResolution
    if exact_source is not None:
        source_resolution = "exact_registry_code"
    elif observation.printed_surface is not None:
        source_resolution = "unclassified_printed_surface"
    else:
        source_resolution = "untyped"
    eligible_observed = tuple(
        row
        for row in support.observed_codes
        if _compatible(
            row.identity,
            source=exact_source,
            requires_setpoint=observation.temperature_setpoint_present,
        )
    )
    if not eligible_observed:
        raise ValueError("fit support has no equipment identity compatible with the source row")

    explore = (
        registry_exploration_permyriad > 0
        and stream.derive("component").randbelow(10_000) < registry_exploration_permyriad
    )
    if explore:
        identity = _sample_registry_identity(
            source=exact_source,
            requires_setpoint=observation.temperature_setpoint_present,
            observed=eligible_observed,
            registry=registry,
            stream=stream.derive("registry-exploration"),
        )
        component: EquipmentSamplingComponent = "registry_exploration"
    else:
        identity = _weighted_observed(
            eligible_observed,
            stream=stream.derive("fit-empirical"),
        ).identity
        component = "fit_empirical"
    return EquipmentTypeScenario(
        source_document_id=observation.source_document_id,
        equipment_row_id=observation.equipment_row_id,
        source_resolution=source_resolution,
        sampling_component=component,
        size_type_code=identity.size_type_code,
        size_code=identity.size_code,
        type_code=identity.type.type_code,
        type_family=identity.type.family_code,
        thermal_capability=identity.type.thermal_capability,
        supports_temperature_setpoint=identity.type.supports_temperature_setpoint,
        printed_surface=None,
        printed_surface_status="pending_text_realization",
        form_projection="exact_container_code",
        training_type_category_status="not_in_current_empty_task_vocabulary",
    )


def _compatible(
    identity: EquipmentIdentity,
    *,
    source: EquipmentIdentity | None,
    requires_setpoint: bool,
) -> bool:
    if requires_setpoint and not identity.type.supports_temperature_setpoint:
        return False
    return source is None or identity.type.family_code == source.type.family_code


def _weighted_observed(
    rows: Sequence[EquipmentCodeSupport], *, stream: DeterministicStream
) -> EquipmentCodeSupport:
    total = sum(row.occurrences for row in rows)
    draw = stream.randbelow(total)
    cumulative = 0
    for row in rows:
        cumulative += row.occurrences
        if draw < cumulative:
            return row
    raise RuntimeError("equipment empirical support is inconsistent")


def _sample_registry_identity(
    *,
    source: EquipmentIdentity | None,
    requires_setpoint: bool,
    observed: Sequence[EquipmentCodeSupport],
    registry: EquipmentRegistryReceipt,
    stream: DeterministicStream,
) -> EquipmentIdentity:
    size_rows = tuple(sorted({row.identity.size_code for row in observed}))
    family_rows = tuple(sorted({row.identity.type.family_code for row in observed}))
    if source is not None:
        size_rows = (source.size_code,)
        family_rows = (source.type.family_code,)
    type_rows = tuple(
        row
        for row in registry.type_codes
        if row.family_code in family_rows
        and (not requires_setpoint or row.supports_temperature_setpoint)
    )
    if not size_rows or not type_rows:
        raise ValueError("authoritative registry has no compatible exploration identity")
    size = size_rows[stream.randbelow(len(size_rows), counter=0)]
    type_row = type_rows[stream.randbelow(len(type_rows), counter=1)]
    try:
        return registry.classify(size + type_row.type_code)
    except EquipmentRegistryError as error:
        raise RuntimeError("registry composition produced a non-classifiable identity") from error
