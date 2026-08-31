"""Fit-isolated empirical temperature scenarios for reviewed reefer identities.

No equipment aliases are interpreted here.  A caller must supply an exact
preserved or authoritative equipment identity and its reviewed reefer status.
This keeps temperature generation independent of fragile free-text heuristics.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from document_ocr.synthesis.generators import DeterministicStream

EquipmentIdentityAuthority = Literal["preserved_reviewed_surface", "authoritative_registry"]
TemperatureResolution = Literal[
    "sampled_fit_empirical",
    "sampled_fit_empirical_reefer_pool",
    "preserved_missingness",
    "not_applicable_non_reefer",
]


@dataclass(frozen=True, slots=True)
class ReeferEquipmentIdentity:
    value: str
    authority: EquipmentIdentityAuthority
    is_reefer: bool

    def __post_init__(self) -> None:
        if not self.value or self.value != self.value.strip():
            raise ValueError("equipment identity must be non-empty without outer whitespace")

    @property
    def key(self) -> tuple[EquipmentIdentityAuthority, str]:
        return (self.authority, self.value)


@dataclass(frozen=True, slots=True)
class ReeferTemperatureObservation:
    source_document_id: str
    equipment_row_id: str
    identity: ReeferEquipmentIdentity
    setpoint_value: float | None
    setpoint_unit: str | None

    def __post_init__(self) -> None:
        if not self.source_document_id or not self.equipment_row_id:
            raise ValueError("reefer observations require non-empty source identities")
        if (self.setpoint_value is None) != (self.setpoint_unit is None):
            raise ValueError("temperature value and unit must be jointly present or absent")
        if self.setpoint_value is not None and not math.isfinite(self.setpoint_value):
            raise ValueError("temperature setpoint must be finite")


@dataclass(frozen=True, slots=True)
class TemperatureValueSupport:
    value: float
    occurrences: int
    document_count: int


@dataclass(frozen=True, slots=True)
class ReeferIdentitySupport:
    identity: ReeferEquipmentIdentity
    setpoint_present_count: int
    setpoint_absent_count: int
    values: tuple[TemperatureValueSupport, ...]


@dataclass(frozen=True, slots=True)
class ReeferTemperatureAudit:
    fit_document_count: int
    input_observations: int
    reefer_observations: int
    non_reefer_observations: int
    setpoint_present_observations: int
    reefer_without_setpoint_observations: int
    supported_reefer_identities: int


@dataclass(frozen=True, slots=True)
class ReeferTemperatureSupport:
    fit_document_ids: tuple[str, ...]
    identities: tuple[ReeferIdentitySupport, ...]
    pooled_values: tuple[TemperatureValueSupport, ...]
    audit: ReeferTemperatureAudit

    def identity_support(self, identity: ReeferEquipmentIdentity) -> ReeferIdentitySupport:
        for row in self.identities:
            if row.identity.key == identity.key:
                if row.identity.is_reefer != identity.is_reefer:
                    raise ValueError("equipment identity has conflicting reviewed reefer status")
                return row
        raise ValueError(
            "reefer identity has no exact fit-isolated empirical temperature support: "
            f"{identity.key!r}"
        )


@dataclass(frozen=True, slots=True)
class TemperatureScenario:
    identity: ReeferEquipmentIdentity
    resolution: TemperatureResolution
    value: float | None
    unit: Literal["celsius"] | None


def build_reefer_temperature_support(
    *,
    observations: Sequence[ReeferTemperatureObservation],
    fit_document_ids: Sequence[str],
) -> ReeferTemperatureSupport:
    """Build exact-identity Celsius support without aliases or cross-split donors."""

    fit_ids = tuple(fit_document_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise ValueError("fit document IDs must be non-empty and unique")
    fit_set = frozenset(fit_ids)
    seen: set[tuple[str, str]] = set()
    identity_status: dict[tuple[EquipmentIdentityAuthority, str], ReeferEquipmentIdentity] = {}
    present: Counter[tuple[EquipmentIdentityAuthority, str]] = Counter()
    absent: Counter[tuple[EquipmentIdentityAuthority, str]] = Counter()
    values: Counter[tuple[tuple[EquipmentIdentityAuthority, str], float]] = Counter()
    value_documents: dict[tuple[tuple[EquipmentIdentityAuthority, str], float], set[str]] = (
        defaultdict(set)
    )
    reefers = non_reefers = 0
    for row in observations:
        if row.source_document_id not in fit_set:
            raise ValueError("reefer observation lies outside the fit partition")
        row_key = (row.source_document_id, row.equipment_row_id)
        if row_key in seen:
            raise ValueError(f"duplicate reefer observation: {row_key!r}")
        seen.add(row_key)
        key = row.identity.key
        previous = identity_status.setdefault(key, row.identity)
        if previous.is_reefer != row.identity.is_reefer:
            raise ValueError("one equipment identity has conflicting reviewed reefer status")
        if row.identity.is_reefer:
            reefers += 1
        else:
            non_reefers += 1
        if row.setpoint_value is None:
            if row.identity.is_reefer:
                absent[key] += 1
            continue
        if not row.identity.is_reefer:
            raise ValueError("temperature setpoint is forbidden for reviewed non-reefer equipment")
        if row.setpoint_unit != "celsius":
            raise ValueError("reefer temperature support is Celsius-only for this corpus")
        present[key] += 1
        value_key = (key, row.setpoint_value)
        values[value_key] += 1
        value_documents[value_key].add(row.source_document_id)

    identity_rows: list[ReeferIdentitySupport] = []
    for key in sorted(identity_status):
        identity = identity_status[key]
        if not identity.is_reefer:
            continue
        supported_values = tuple(
            TemperatureValueSupport(
                value=value,
                occurrences=count,
                document_count=len(value_documents[(key, value)]),
            )
            for (candidate_key, value), count in sorted(values.items())
            if candidate_key == key
        )
        identity_rows.append(
            ReeferIdentitySupport(
                identity=identity,
                setpoint_present_count=present[key],
                setpoint_absent_count=absent[key],
                values=supported_values,
            )
        )
    if not any(row.values for row in identity_rows):
        raise ValueError("fit partition has no reviewed Celsius reefer setpoint support")
    pooled_counts: Counter[float] = Counter()
    pooled_documents: dict[float, set[str]] = defaultdict(set)
    for row in observations:
        if row.setpoint_value is None:
            continue
        pooled_counts[row.setpoint_value] += 1
        pooled_documents[row.setpoint_value].add(row.source_document_id)
    pooled_values = tuple(
        TemperatureValueSupport(
            value=value,
            occurrences=count,
            document_count=len(pooled_documents[value]),
        )
        for value, count in sorted(pooled_counts.items())
    )
    return ReeferTemperatureSupport(
        fit_document_ids=fit_ids,
        identities=tuple(identity_rows),
        pooled_values=pooled_values,
        audit=ReeferTemperatureAudit(
            fit_document_count=len(fit_ids),
            input_observations=len(observations),
            reefer_observations=reefers,
            non_reefer_observations=non_reefers,
            setpoint_present_observations=sum(present.values()),
            reefer_without_setpoint_observations=sum(absent.values()),
            supported_reefer_identities=sum(bool(row.values) for row in identity_rows),
        ),
    )


def sample_reefer_temperature(
    *,
    identity: ReeferEquipmentIdentity,
    source_setpoint_present: bool,
    support: ReeferTemperatureSupport,
    stream: DeterministicStream,
) -> TemperatureScenario:
    """Preserve setpoint missingness and sample only exact-identity fit values."""

    if not identity.is_reefer:
        if source_setpoint_present:
            raise ValueError("cannot request a temperature setpoint for non-reefer equipment")
        return TemperatureScenario(
            identity=identity,
            resolution="not_applicable_non_reefer",
            value=None,
            unit=None,
        )
    if not source_setpoint_present:
        return TemperatureScenario(
            identity=identity,
            resolution="preserved_missingness",
            value=None,
            unit=None,
        )
    identity_support = support.identity_support(identity)
    if not identity_support.values:
        raise ValueError("reefer identity has no observed setpoint values in the fit partition")
    total = sum(row.occurrences for row in identity_support.values)
    draw = stream.derive("temperature-setpoint").randbelow(total)
    selected: TemperatureValueSupport | None = None
    cumulative = 0
    for row in identity_support.values:
        cumulative += row.occurrences
        if draw < cumulative:
            selected = row
            break
    if selected is None:
        raise RuntimeError("temperature cumulative support is inconsistent")
    return TemperatureScenario(
        identity=identity,
        resolution="sampled_fit_empirical",
        value=selected.value,
        unit="celsius",
    )


def sample_generated_equipment_temperature(
    *,
    identity: ReeferEquipmentIdentity,
    source_setpoint_present: bool,
    support: ReeferTemperatureSupport,
    stream: DeterministicStream,
) -> TemperatureScenario:
    """Sample a reviewed setpoint for a newly generated equipment identity.

    Generated BIC identities cannot truthfully use the old source surface as an
    exact conditioning key.  This path instead draws from the complete
    fit-isolated pool of reviewed reefer setpoints, while the authoritative BIC
    identity supplies the semantic reefer/non-reefer gate.
    """

    if not identity.is_reefer:
        if source_setpoint_present:
            raise ValueError("generated non-reefer equipment cannot carry a setpoint")
        return TemperatureScenario(
            identity=identity,
            resolution="not_applicable_non_reefer",
            value=None,
            unit=None,
        )
    if not source_setpoint_present:
        return TemperatureScenario(
            identity=identity,
            resolution="preserved_missingness",
            value=None,
            unit=None,
        )
    if not support.pooled_values:
        raise ValueError("fit partition has no reviewed reefer setpoint pool")
    total = sum(row.occurrences for row in support.pooled_values)
    draw = stream.derive("temperature-setpoint-pool").randbelow(total)
    cumulative = 0
    for row in support.pooled_values:
        cumulative += row.occurrences
        if draw < cumulative:
            return TemperatureScenario(
                identity=identity,
                resolution="sampled_fit_empirical_reefer_pool",
                value=row.value,
                unit="celsius",
            )
    raise RuntimeError("pooled temperature cumulative support is inconsistent")
