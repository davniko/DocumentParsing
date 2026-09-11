"""Deterministic, line-owned invariants for synthetic raw-text certification.

The semantic auditor is deliberately advisory: it can quarantine a candidate, but its output is
not objective rewrite authority.  This module owns the complementary executable contract.  It
binds the exact source contract and inventory bytes, replays compiler postconditions against the
final candidate, and emits structured findings whose evidence and any repair are host-derived.

The implementation is intentionally source-grounded.  It does not infer arbitrary facts from
numbers or prose; every check below comes from an explicit compiler requirement, inventory owner,
or repeated field relationship already present in the immutable source document.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.container_semantics import review_source_equipment_surface
from document_ocr.synthesis.raw_text_certification_references import (
    CertificationReferenceIndex,
)
from document_ocr.synthesis.raw_text_inventory import (
    DeterministicInventoryEdit,
    InventoryCandidate,
    LocatedAuxiliaryValue,
    locate_auxiliary_values,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    AnchoredScalarReplacementRequirement,
    AppliedDeterministicPrefill,
    CargoFlavorRewriteRequirement,
    InlineSlotTopologyRequirement,
    JurisdictionalSurfaceRequirement,
    OperationalFlavorRequirement,
    SourceSemanticRoleHint,
    SourceStatusPreservationRequirement,
    SurfaceRenderingRequirement,
    TargetLiteralRequirement,
    TargetValueOccurrenceRequirement,
    _anchored_scalar_replacements_rendered,
    _cargo_flavor_rewrite_failures,
    _dense_aggregate_measurement_rows,
    _inline_slot_topology_mismatches,
    _jurisdictional_surfaces_rendered,
    _missing_target_literals,
    _operational_flavor_requirements_rendered,
    _required_surface_rendered,
    _source_status_surfaces_preserved,
    _structural_line_prefix_mismatches,
    _target_value_occurrence_count,
)
from document_ocr.synthesis.transport_capacity import (
    TransportCapacityLimits,
    equipment_capacity,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LineId = Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")

InvariantDimension = Literal[
    "target_fact_fidelity",
    "repeated_and_derived_relations",
    "source_private_and_auxiliary_replacement",
    "party_legal_and_negotiability",
    "route_jurisdiction_and_identifiers",
    "equipment_temperature_and_capacity",
    "cargo_packages_and_dangerous_goods",
    "template_format_and_model_artifacts",
]
InvariantFindingKind = Literal[
    "target_fact_mismatch",
    "repeated_or_derived_fact_mismatch",
    "source_only_private_or_auxiliary_fact",
    "party_or_legal_identity_mismatch",
    "route_or_jurisdiction_mismatch",
    "equipment_or_temperature_mismatch",
    "cargo_or_dangerous_goods_mismatch",
    "format_or_model_artifact",
]


class InvariantEvidence(BaseModel):
    """One exact physical candidate line supporting a deterministic failure."""

    model_config = _STRICT

    lineId: LineId
    currentLine: str


class DeterministicLineRepair(BaseModel):
    """A byte-exact replacement derived from an already-authoritative compiler requirement."""

    model_config = _STRICT

    method: Literal["replace_exact_fragment_v1"]
    lineId: LineId
    expectedCurrentLineSha256: Sha256
    oldFragment: NonEmptyText
    newFragment: NonEmptyText

    @model_validator(mode="after")
    def replacement_changes_bytes(self) -> DeterministicLineRepair:
        if self.oldFragment == self.newFragment:
            raise ValueError("deterministic line repair does not change its fragment")
        fragments = (self.oldFragment, self.newFragment)
        if any(character in fragment for fragment in fragments for character in "\r\n"):
            raise ValueError("deterministic line repair fragments must remain on one physical line")
        return self


class DeterministicCertificationFinding(BaseModel):
    """One host-proven certification defect and its narrowly bounded repair authority."""

    model_config = _STRICT

    invariantId: Annotated[str, StringConstraints(pattern=r"^D[0-9a-f]{16}$")]
    findingKind: InvariantFindingKind
    dimension: InvariantDimension
    evidence: Annotated[tuple[InvariantEvidence, ...], Field(min_length=1)]
    targetPaths: tuple[NonEmptyText, ...]
    problem: NonEmptyText
    repairs: tuple[DeterministicLineRepair, ...] = ()


class InvariantRequirementCount(BaseModel):
    model_config = _STRICT

    name: NonEmptyText
    count: Annotated[int, Field(ge=0)]


class CertificationInvariantEnvelope(BaseModel):
    """Immutable identities needed to reproduce the deterministic certification decision."""

    model_config = _STRICT

    schemaVersion: Literal[1]
    documentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    sourceTextSha256: Sha256
    candidateTextSha256: Sha256
    sourceContractSha256: Sha256
    inventorySha256: Sha256
    deterministicEditsSha256: Sha256
    sourceLabelSha256: Sha256
    targetLabelSha256: Sha256
    referenceReceiptSha256: Sha256
    capacityPolicySha256: Sha256
    implementationSha256: Sha256
    requirementCounts: tuple[InvariantRequirementCount, ...]


class InvariantCheck(BaseModel):
    model_config = _STRICT

    checkId: NonEmptyText
    findings: Annotated[int, Field(ge=0)]
    passed: bool

    @model_validator(mode="after")
    def verdict_matches_count(self) -> InvariantCheck:
        if self.passed != (self.findings == 0):
            raise ValueError("invariant check verdict and finding count disagree")
        return self


class CertificationInvariantAudit(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    envelopeSha256: Sha256
    passed: bool
    checks: Annotated[tuple[InvariantCheck, ...], Field(min_length=1)]
    findings: tuple[DeterministicCertificationFinding, ...]

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> CertificationInvariantAudit:
        if self.passed != (not self.findings):
            raise ValueError("invariant audit verdict and findings disagree")
        if sum(row.findings for row in self.checks) != len(self.findings):
            raise ValueError("invariant check counts do not cover every finding")
        return self


_CONTRACT_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("deterministicPrefills", AppliedDeterministicPrefill),
    ("surfaceRenderingRequirements", SurfaceRenderingRequirement),
    ("targetLiteralRequirements", TargetLiteralRequirement),
    ("targetValueOccurrenceRequirements", TargetValueOccurrenceRequirement),
    ("anchoredScalarReplacementRequirements", AnchoredScalarReplacementRequirement),
    ("sourceSemanticRoleHints", SourceSemanticRoleHint),
    ("inlineSlotTopologyRequirements", InlineSlotTopologyRequirement),
    ("sourceStatusPreservationRequirements", SourceStatusPreservationRequirement),
    ("operationalFlavorRequirements", OperationalFlavorRequirement),
    ("cargoFlavorRewriteRequirements", CargoFlavorRewriteRequirement),
    ("jurisdictionalSurfaceRequirements", JurisdictionalSurfaceRequirement),
)


def _parse_rows(
    contract: Mapping[str, Any], key: str, model: type[BaseModel]
) -> tuple[BaseModel, ...]:
    raw = contract.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"source contract lacks list {key}")
    return tuple(model.model_validate_json(canonical_json_bytes(row), strict=True) for row in raw)


def _line_number(line_id: str) -> int:
    return int(line_id[1:])


_MEASUREMENT_TOKEN = re.compile(r"(?<![0-9.,])[0-9]+(?:[.,][0-9]+)*(?![0-9.,])")
_DENSE_COMPONENT_TOTAL = re.compile(
    r"^[ \t]*={4,}[ \t]+(?P<weight>[0-9]+(?:[.,][0-9]+)*)[ \t]+"
    r"(?P<volume>[0-9]+(?:[.,][0-9]+)*)[ \t]*$"
)
_NUMERIC_ONLY_LINE = re.compile(r"^[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)*)[ \t]*$")
_SERIAL_RANGE = re.compile(
    r"^[ \t]*(?P<prefix>[A-Z]+)(?P<start>[0-9]+)[ \t]*-[ \t]*"
    r"(?P=prefix)(?P<end>[0-9]+)[ \t]*$",
    re.IGNORECASE,
)
_FREIGHT_CHARGES_HEADING = re.compile(r"^[ \t]*FREIGHT[ \t]*&[ \t]*CHARGES[ \t]*$", re.IGNORECASE)
_MONEY_ONLY_LINE = re.compile(
    r"^[ \t]*(?P<currency>[A-Z]{3})[ \t]+"
    r"(?P<amount>[0-9]+(?:[.,][0-9]+)*)[ \t]*$",
    re.IGNORECASE,
)
_MONEY_TOKEN = re.compile(
    r"(?<![A-Z])(?P<currency>[A-Z]{3})[ \t]+"
    r"(?P<amount>[0-9]+(?:[.,][0-9]+)*)(?![0-9.,])",
    re.IGNORECASE,
)
_TOTAL_FREIGHT_CHARGES = re.compile(r"\bTOTAL[ \t]+FREIGHT[ \t]*&[ \t]*CHARGES\b", re.IGNORECASE)
_CARRIER_REFERENCE_HEADING = re.compile(
    r"\bCARRIER(?:['\N{RIGHT SINGLE QUOTATION MARK}]?S)?[ \t]+REFERENCE\b",
    re.IGNORECASE,
)
_MIXED_IDENTIFIER = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9/-]*[A-Z])(?=[A-Z0-9/-]*[0-9])"
    r"[A-Z0-9]+(?:[-/][A-Z0-9]+)*(?![A-Z0-9])",
    re.IGNORECASE,
)
_CARRIER_NAME_TOKEN = re.compile(r"[A-Z0-9]+", re.IGNORECASE)
_GENERIC_CARRIER_NAME_TOKENS = frozenset(
    {
        "AG",
        "AKTIENGESELLSCHAFT",
        "AS",
        "BV",
        "CARRIER",
        "CARRIERS",
        "CO",
        "COMPANY",
        "CONTAINER",
        "CORP",
        "CORPORATION",
        "GROUP",
        "HOLDING",
        "HOLDINGS",
        "INC",
        "INTERNATIONAL",
        "LIMITED",
        "LINE",
        "LINES",
        "LLC",
        "LOGISTICS",
        "LTD",
        "MARINE",
        "MARITIME",
        "NV",
        "OCEAN",
        "OCEANIC",
        "PLC",
        "SA",
        "SAS",
        "SHIPPING",
        "SYSTEM",
        "TRADING",
        "TRANSPORT",
        "TRANSIT",
    }
)
_CARRIER_ROLE_CONTEXT = re.compile(
    r"(?:\b(?:SIGNED[ \t]+(?:ON[ \t]+BEHALF[ \t]+OF|FOR)[ \t]+(?:THE[ \t]+)?CARRIER|"
    r"AS[ \t]+AGENTS?[ \t]+(?:FOR|OF)[ \t]+(?:THE[ \t]+)?CARRIER|"
    r"FOR[ \t]+ABOVE[ \t]+NAMED[ \t]+CARRIER|"
    r"CARRIER[ \t]+AND[ \t]+SERVICE[ \t]+PROVIDER)\b|\bCARRIER[ \t]*:)",
    re.IGNORECASE,
)
_CARRIER_OFFICE_CONTEXT = re.compile(r"\bPLACE[ \t]+OF[ \t]+ISSUANCE\b.*\bOFFICE\b", re.IGNORECASE)
_PARTY_TARGET_PATH = re.compile(
    r"^(?P<party>documentPatch\.parties\.[A-Za-z][A-Za-z0-9]*(?:\[[0-9]+\])?)\."
    r"(?P<field>name|address|city|country)$"
)
_GENERIC_ADDRESS_TOKENS = frozenset(
    {
        "address",
        "area",
        "attached",
        "avenue",
        "block",
        "building",
        "city",
        "country",
        "estate",
        "floor",
        "house",
        "industrial",
        "market",
        "no",
        "office",
        "port",
        "road",
        "street",
        "suite",
        "the",
        "to",
        "zone",
    }
)
_OFFICE_SECTION_HEADING = re.compile(
    r"^[ \t]*(?:(?:DESTINATION|DELIVERY|ORIGIN|LOCAL)[ \t]+)?OFFICE[ \t]*$",
    re.IGNORECASE,
)
_FOREIGN_EXPORTER_CAPTION = re.compile(r"\bFOREIGN[ \t]+EXPORTER\b", re.IGNORECASE)
_FOREIGN_EXPORTER_COUNTRY = re.compile(
    r"^[ \t]*(?:COUNTRY[ \t]+OF[ \t]+FOREIGN[ \t]+EXPORTER|COUNTRY)"
    r"[ \t]*:[ \t]*(?P<value>.+?)[ \t]*$",
    re.IGNORECASE,
)
_FOREIGN_EXPORTER_CODE = re.compile(
    r"^[ \t]*CODE[ \t]*:[ \t]*(?P<value>[A-Z]{2})[ \t]*$", re.IGNORECASE
)
_FOREIGN_EXPORTER_NAME = re.compile(
    r"^[ \t]*FOREIGN[ \t]+EXPORTER[ \t]+NAME[ \t]*[-:][ \t]*(?P<value>.*?)[ \t]*$",
    re.IGNORECASE,
)
_EXPLICIT_TARE_VALUE = re.compile(
    r"\bTARE\b[^0-9\r\n]*(?P<value>[0-9]+(?:[.,][0-9]+)*)",
    re.IGNORECASE,
)
_EXPLICIT_VOLUME_VALUE = re.compile(
    r"^[ \t]*(?:VOL(?:UME)?|MEASUREMENT)[ \t]*:?[ \t]*"
    r"(?P<value>[0-9]+(?:[.,][0-9]+)*)(?:[ \t]*(?:CBM|M3|M\^3))?[ \t]*$",
    re.IGNORECASE,
)
_MEASUREMENT_HEADING = re.compile(r"^[ \t]*(?:MEASUREMENT|VOLUME)[ \t]*:?[ \t]*$", re.IGNORECASE)
_VOLUME_UNIT = re.compile(r"^[ \t]*(?:CBM(?:\(M3\))?|M3|M\^3|MTQ)[ \t]*$", re.IGNORECASE)
_CORROBORATING_MASS_TOTAL = re.compile(
    r"\bTOTAL[ \t]+(?:ADMT|MT|METRIC[ \t]+TON(?:NE)?S?)[ \t]*:[ \t]*"
    r"(?P<value>[0-9]+(?:[.,][0-9]+)*)",
    re.IGNORECASE,
)
_LEGAL_ENTITY_SURFACE = re.compile(
    r"\b(?:COMPANY|CORPORATION|INC|LIMITED|LLC|LTD|PLC)\b|\bS[ .]?A[ .]?(?:E[ .]?)?\b",
    re.IGNORECASE,
)

_NumericGrammar = Literal[
    "pair_before_seal",
    "trailing_pair",
    "component_total",
    "numeric_only",
    "aggregate_tuple",
]


@dataclass(frozen=True, slots=True)
class _NumericSlot:
    line_number: int
    grammar: _NumericGrammar
    value_index: int
    coefficient: int
    source_value: Decimal


@dataclass(frozen=True, slots=True)
class _AdditiveRelation:
    relation_id: str
    semantic_kind: Literal[
        "container_component_weight",
        "container_component_volume",
        "shipment_gross_weight",
        "shipment_tare_weight",
        "shipment_volume",
    ]
    components: tuple[_NumericSlot, ...]
    total: _NumericSlot


@dataclass(frozen=True, slots=True)
class _SerialCardinalityRelation:
    relation_id: str
    group_index: int
    source_quantity: int
    target_quantity: int
    target_range_surfaces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EquipmentAssertion:
    line_number: int
    count: int
    size_category: str
    type_category: str
    surface: str


@dataclass(frozen=True, slots=True)
class _OfficeReferenceRelation:
    relation_id: str
    identity_line_numbers: tuple[int, ...]
    locality_line_numbers: tuple[int, ...]
    country_line_numbers: tuple[int, ...]
    contact_line_numbers: tuple[int, ...]
    target_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MonetarySlot:
    line_number: int
    currency: str
    amount: Decimal
    half_unit: Decimal
    source_surface: str


@dataclass(frozen=True, slots=True)
class _FreightScheduleRelation:
    relation_id: str
    components: tuple[_MonetarySlot, ...]
    total: _MonetarySlot
    private_surfaces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PhysicalCapacityRelation:
    relation_id: str
    equipment_line_number: int
    measurement_line_number: int
    measurement_kind: Literal["payload", "volume"]
    source_value: Decimal
    source_half_unit: Decimal


@dataclass(frozen=True, slots=True)
class _PackageCargoContradiction:
    relation_id: str
    category_token: str
    package_display_name: str
    target_description: str
    package_line_ids: tuple[str, ...]
    cargo_line_ids: tuple[str, ...]
    target_paths: tuple[str, str]


@dataclass(frozen=True, slots=True)
class _CarrierReferencePrefixSlot:
    line_number: int
    source_token: str
    source_prefix: str


@dataclass(frozen=True, slots=True)
class _CarrierRoleTemplateSlot:
    line_number: int
    mutable_start: int
    mutable_end: int


@dataclass(frozen=True, slots=True)
class _CarrierRetirementRelation:
    source_name: str
    target_name: str
    distinctive_tokens: tuple[str, ...]
    source_identity_line_numbers: tuple[int, ...]
    reference_prefix_slots: tuple[_CarrierReferencePrefixSlot, ...]
    role_template_slots: tuple[_CarrierRoleTemplateSlot, ...]


@dataclass(frozen=True, slots=True)
class _SourceMeasurementRetirementRelation:
    relation_id: str
    measurement_kind: Literal["tare", "volume"]
    source_value_key: str
    source_surfaces: tuple[str, ...]
    line_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PartyAddressSlot:
    line_number: int
    source_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PartyAddressRetirementRelation:
    party_path: str
    anchor_line_numbers: tuple[int, ...]
    address_slots: tuple[_PartyAddressSlot, ...]


@dataclass(frozen=True, slots=True)
class _ForeignExporterRelation:
    section_line_numbers: tuple[int, ...]
    party_path: str
    target_name: str
    target_country: str
    target_country_codes: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ContainerPackageScopeFact:
    package_path: str
    quantity: int
    printed_name: str


@dataclass(frozen=True, slots=True)
class _ContainerPackageScopeRelation:
    line_number: int
    target_container_number: str
    forbidden_facts: tuple[_ContainerPackageScopeFact, ...]


def _label(
    contract: Mapping[str, Any], key: Literal["sourceLabel", "targetLabel"]
) -> Mapping[str, Any]:
    value = contract.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"source contract lacks its {key}")
    return value


def _label_containers(label: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("certification label lacks documentPatch")
    raw = patch.get("containers") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("certification label containers are not a sequence")
    if any(not isinstance(row, Mapping) for row in raw):
        raise ValueError("certification label contains a non-object container")
    return tuple(cast(Mapping[str, Any], row) for row in raw)


_PARTS_CARGO_DESCRIPTION = re.compile(r"\bPARTS?[ \t]+(?:FOR|OF)\b", re.IGNORECASE)


def _compile_package_cargo_contradictions(
    *,
    target_label: Mapping[str, Any],
    anchored_rows: Sequence[AnchoredScalarReplacementRequirement],
    cargo_rows: Sequence[CargoFlavorRewriteRequirement],
    references: CertificationReferenceIndex,
) -> tuple[_PackageCargoContradiction, ...]:
    """Compile only target contradictions proved by category, group, and rendered-line owners.

    ``PACKAGE_VEHICLE`` denotes a whole vehicle in the pinned application vocabulary.  A cargo
    group whose target description explicitly says ``PART(S) FOR/OF`` another article therefore
    cannot coherently use that category as its package unit.  Exact anchored package rows and the
    cargo-flavor owner are required so a finding has physical candidate evidence; an incomplete
    source contract fails closed instead of inventing line ownership.
    """

    patch = target_label.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("certification target label lacks documentPatch")
    raw_groups = patch.get("cargoGroups") or ()
    raw_packages = patch.get("cargoPackages") or ()
    if (
        not isinstance(raw_groups, Sequence)
        or isinstance(raw_groups, (str, bytes))
        or not isinstance(raw_packages, Sequence)
        or isinstance(raw_packages, (str, bytes))
    ):
        raise ValueError("certification target cargo groups/packages are not sequences")
    if any(not isinstance(row, Mapping) for row in (*raw_groups, *raw_packages)):
        raise ValueError("certification target cargo group/package is not an object")
    groups = tuple(cast(Mapping[str, Any], row) for row in raw_groups)
    packages = tuple(cast(Mapping[str, Any], row) for row in raw_packages)
    if not any(row.get("typeCategory") == "PACKAGE_VEHICLE" for row in packages):
        return ()
    group_indexes: dict[str, int] = {}
    for index, group in enumerate(groups):
        group_id = group.get("groupId")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("certification target cargo group lacks groupId")
        if group_id in group_indexes:
            raise ValueError(f"certification target repeats cargo groupId {group_id!r}")
        group_indexes[group_id] = index

    output: list[_PackageCargoContradiction] = []
    for package_index, package in enumerate(packages):
        category = package.get("typeCategory")
        if category != "PACKAGE_VEHICLE":
            continue
        display_name = references.package_display_name(category)
        group_id = package.get("groupId")
        if not isinstance(group_id, str) or group_id not in group_indexes:
            raise ValueError("vehicle package does not resolve to one target cargo group")
        group_index = group_indexes[group_id]
        description = groups[group_index].get("description")
        if not isinstance(description, str) or _PARTS_CARGO_DESCRIPTION.search(description) is None:
            continue
        package_path = f"documentPatch.cargoPackages[{package_index}].typeCategory"
        cargo_path = f"documentPatch.cargoGroups[{group_index}].description"
        package_pattern = re.compile(
            rf"(?<![A-Z]){re.escape(display_name)}S?(?![A-Z])", re.IGNORECASE
        )
        package_line_ids = tuple(
            dict.fromkeys(
                line_id
                for row in anchored_rows
                if package_path in row.targetPaths
                if package_pattern.search(row.targetSurface) is not None
                for line_id in row.sourceLineIds
            )
        )
        cargo_line_ids = tuple(
            dict.fromkeys(
                line_id
                for row in cargo_rows
                if row.targetPath == cargo_path
                for line_id in row.sourceLineIds
            )
        )
        if not package_line_ids or not cargo_line_ids:
            raise ValueError(
                "target vehicle/parts contradiction lacks exact package or cargo line ownership"
            )
        relation_payload = {
            "category": category,
            "displayName": display_name,
            "description": description,
            "packageLineIds": list(package_line_ids),
            "cargoLineIds": list(cargo_line_ids),
            "targetPaths": [package_path, cargo_path],
        }
        output.append(
            _PackageCargoContradiction(
                relation_id="package-cargo-"
                + sha256_bytes(canonical_json_bytes(relation_payload))[:16],
                category_token=category,
                package_display_name=display_name,
                target_description=description,
                package_line_ids=package_line_ids,
                cargo_line_ids=cargo_line_ids,
                target_paths=(package_path, cargo_path),
            )
        )
    return tuple(output)


def _package_printed_name(
    package: Mapping[str, Any], references: CertificationReferenceIndex
) -> str:
    category = package.get("typeCategory")
    description = package.get("typeDescription")
    if isinstance(category, str):
        return references.package_display_name(category)
    if isinstance(description, str) and description.strip():
        return description.strip()
    raise ValueError("target cargo package lacks a printable type")


def _compile_container_package_scope_relations(
    *,
    target_label: Mapping[str, Any],
    operational_rows: Sequence[OperationalFlavorRequirement],
    references: CertificationReferenceIndex,
) -> tuple[_ContainerPackageScopeRelation, ...]:
    """Bind allocation-owned package rows against shipment-wide package facts."""

    patch = _document_patch(target_label)
    raw_packages = patch.get("cargoPackages") or ()
    raw_allocation_groups = patch.get("cargoAllocationGroups") or ()
    if (
        not isinstance(raw_packages, Sequence)
        or isinstance(raw_packages, (str, bytes))
        or not isinstance(raw_allocation_groups, Sequence)
        or isinstance(raw_allocation_groups, (str, bytes))
        or any(not isinstance(row, Mapping) for row in (*raw_packages, *raw_allocation_groups))
    ):
        raise ValueError("certification target package/allocation graph is invalid")
    packages = tuple(cast(Mapping[str, Any], row) for row in raw_packages)
    package_by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, package in enumerate(packages):
        package_id = package.get("packageId")
        quantity = package.get("quantity")
        if (
            not isinstance(package_id, str)
            or not package_id
            or package_id in package_by_id
            or not isinstance(quantity, int)
            or isinstance(quantity, bool)
            or quantity <= 0
        ):
            raise ValueError("certification target cargo package identity/quantity is invalid")
        package_by_id[package_id] = (index, package)

    allocated: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for raw_group in raw_allocation_groups:
        group = cast(Mapping[str, Any], raw_group)
        package_ids = group.get("packageIds")
        allocations = group.get("allocations")
        if (
            group.get("coverage") != "single_package_level"
            or not isinstance(package_ids, Sequence)
            or isinstance(package_ids, (str, bytes))
            or len(package_ids) != 1
            or not isinstance(package_ids[0], str)
            or package_ids[0] not in package_by_id
            or not isinstance(allocations, Sequence)
            or isinstance(allocations, (str, bytes))
        ):
            continue
        package_id = package_ids[0]
        for raw_allocation in allocations:
            if not isinstance(raw_allocation, Mapping):
                raise ValueError("certification target allocation is not an object")
            container = raw_allocation.get("containerNumber")
            quantity = raw_allocation.get("packageQuantity")
            if (
                not isinstance(container, str)
                or not container
                or not isinstance(quantity, int)
                or isinstance(quantity, bool)
                or quantity <= 0
            ):
                raise ValueError("certification target allocation identity/quantity is invalid")
            allocated[container][package_id].add(quantity)

    output: dict[tuple[int, str], _ContainerPackageScopeRelation] = {}
    for row in operational_rows:
        if (
            row.kind != "package_quantity"
            or row.samplingMethod != "target_package_allocation_v1"
            or row.targetContainerNumber is None
        ):
            continue
        forbidden: list[_ContainerPackageScopeFact] = []
        container_allocations = allocated.get(row.targetContainerNumber, {})
        for package_id, (index, package) in package_by_id.items():
            quantity = cast(int, package["quantity"])
            if quantity in container_allocations.get(package_id, set()):
                continue
            forbidden.append(
                _ContainerPackageScopeFact(
                    package_path=f"documentPatch.cargoPackages[{index}]",
                    quantity=quantity,
                    printed_name=_package_printed_name(package, references),
                )
            )
        if forbidden:
            output[(_line_number(row.sourceLineId), row.targetContainerNumber)] = (
                _ContainerPackageScopeRelation(
                    line_number=_line_number(row.sourceLineId),
                    target_container_number=row.targetContainerNumber,
                    forbidden_facts=tuple(forbidden),
                )
            )
    return tuple(output[key] for key in sorted(output))


def _package_fact_in_line(line: str, fact: _ContainerPackageScopeFact) -> bool:
    quantity = f"{fact.quantity:,}"
    quantity_pattern = rf"(?:{re.escape(str(fact.quantity))}|{re.escape(quantity)})"
    name_pattern = r"[ \t]+".join(re.escape(part) for part in fact.printed_name.split())
    return (
        re.search(
            rf"(?<![0-9]){quantity_pattern}[ \t]+{name_pattern}(?:[ \t]*\((?:E)?S\)|S)?(?![A-Z])",
            line,
            re.IGNORECASE,
        )
        is not None
    )


def _decimal_surface(value: str) -> tuple[Decimal, Decimal] | None:
    """Parse one printed number and its half-unit rounding bound.

    When a single separator has a three-digit suffix, its decimal/grouping role is genuinely
    ambiguous without locale metadata. Treating it consistently as a decimal preserves every
    additive equality (and every scale mismatch) while avoiding the unsafe magnitude heuristic
    used for capacity parsing. Two different separators have an unambiguous final decimal mark.
    """

    raw = value.strip()
    if _MEASUREMENT_TOKEN.fullmatch(raw) is None:
        return None
    separators = {character for character in raw if character in ".,"}
    decimal_places = 0
    if len(separators) == 2:
        decimal_separator = "." if raw.rfind(".") > raw.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        integer, fraction = raw.rsplit(decimal_separator, 1)
        normalized = integer.replace(grouping_separator, "") + "." + fraction
        decimal_places = len(fraction)
    elif len(separators) == 1:
        separator = next(iter(separators))
        pieces = raw.split(separator)
        if len(pieces) == 2:
            normalized = pieces[0] + "." + pieces[1]
            decimal_places = len(pieces[1])
        elif all(len(piece) == 3 for piece in pieces[1:]):
            normalized = "".join(pieces)
        else:
            return None
    else:
        normalized = raw
    try:
        parsed = Decimal(normalized)
    except InvalidOperation:
        return None
    if not parsed.is_finite() or parsed < 0:
        return None
    half_unit = Decimal(1).scaleb(-decimal_places) / 2
    return parsed, half_unit


def _measurement_value_key(surface: str) -> str:
    digits = re.sub(r"[^0-9]", "", surface)
    return digits.lstrip("0") or "0"


def _line_measurement_surfaces(line: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _MEASUREMENT_TOKEN.finditer(line))


def _contains_populated_measurement_key(value: Any, keys: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = re.sub(r"[^a-z]", "", str(key).casefold())
            if normalized_key in keys and nested is not None:
                return True
            if _contains_populated_measurement_key(nested, keys):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_populated_measurement_key(nested, keys) for nested in value)
    return False


def _compile_source_measurement_retirement_relations(
    *, source_lines: Sequence[str], target_label: Mapping[str, Any]
) -> tuple[_SourceMeasurementRetirementRelation, ...]:
    """Own explicit source tare/volume values only when the complete target omits that fact.

    The rule intentionally does not reject a newly synthesized auxiliary measurement.  It rejects
    only the exact numeric identity of an explicit source value, including harmless punctuation
    variants, so target-absent physical data cannot leak through repeated template slots.
    """

    target_has_volume = _contains_populated_measurement_key(
        _document_patch(target_label), frozenset({"volume"})
    )
    target_has_tare = _contains_populated_measurement_key(
        _document_patch(target_label), frozenset({"tare", "tareweight", "tareweightkg"})
    )
    seeds: dict[tuple[Literal["tare", "volume"], str], list[tuple[int, str]]] = defaultdict(list)
    for number, line in enumerate(source_lines, start=1):
        if not target_has_tare:
            tare = _EXPLICIT_TARE_VALUE.search(line)
            if tare is not None:
                surface = tare.group("value")
                seeds[("tare", _measurement_value_key(surface))].append((number, surface))
        if target_has_volume:
            continue
        volume = _EXPLICIT_VOLUME_VALUE.fullmatch(line)
        if volume is not None:
            surface = volume.group("value")
            seeds[("volume", _measurement_value_key(surface))].append((number, surface))
            continue
        numeric = _NUMERIC_ONLY_LINE.fullmatch(line)
        if numeric is None:
            continue
        preceded_by_heading = number >= 2 and _MEASUREMENT_HEADING.fullmatch(
            source_lines[number - 2]
        )
        preceded_by_heading_and_unit = (
            number >= 3
            and _VOLUME_UNIT.fullmatch(source_lines[number - 2]) is not None
            and _MEASUREMENT_HEADING.fullmatch(source_lines[number - 3]) is not None
        )
        if preceded_by_heading or preceded_by_heading_and_unit:
            surface = numeric.group("value")
            seeds[("volume", _measurement_value_key(surface))].append((number, surface))

    output: list[_SourceMeasurementRetirementRelation] = []
    for (kind, value_key), rows in sorted(seeds.items()):
        line_numbers = {number for number, _surface in rows}
        surfaces = {surface for _number, surface in rows}
        if kind == "volume":
            for number, line in enumerate(source_lines, start=1):
                if "TOTAL" not in line.upper():
                    continue
                matches = tuple(
                    surface
                    for surface in _line_measurement_surfaces(line)
                    if _measurement_value_key(surface) == value_key
                )
                if matches:
                    line_numbers.add(number)
                    surfaces.update(matches)
        payload = {
            "kind": kind,
            "sourceValueKey": value_key,
            "lineNumbers": sorted(line_numbers),
        }
        output.append(
            _SourceMeasurementRetirementRelation(
                relation_id="source-measurement-"
                + sha256_bytes(canonical_json_bytes(payload))[:16],
                measurement_kind=kind,
                source_value_key=value_key,
                source_surfaces=tuple(sorted(surfaces, key=lambda value: (-len(value), value))),
                line_numbers=tuple(sorted(line_numbers)),
            )
        )
    return tuple(output)


def _source_measurement_residue_line_numbers(
    candidate_lines: Sequence[str], relation: _SourceMeasurementRetirementRelation
) -> tuple[int, ...]:
    return tuple(
        number
        for number in relation.line_numbers
        if number <= len(candidate_lines)
        and any(
            _measurement_value_key(surface) == relation.source_value_key
            for surface in _line_measurement_surfaces(candidate_lines[number - 1])
        )
    )


def _numeric_pair(
    line: str, grammar: Literal["pair_before_seal", "trailing_pair"]
) -> tuple[str, str] | None:
    searchable = line
    if grammar == "pair_before_seal":
        seal = re.search(r"\bSEAL[ \t]*:", line, re.IGNORECASE)
        if seal is None:
            return None
        searchable = line[: seal.start()]
    matches = tuple(_MEASUREMENT_TOKEN.finditer(searchable))
    if len(matches) < 2:
        return None
    return matches[-2].group(0), matches[-1].group(0)


def _slot_value(lines: Sequence[str], slot: _NumericSlot) -> tuple[Decimal, Decimal] | None:
    index = slot.line_number - 1
    if not 0 <= index < len(lines):
        return None
    line = lines[index]
    surfaces: tuple[str, ...]
    if slot.grammar in {"pair_before_seal", "trailing_pair"}:
        pair = _numeric_pair(line, slot.grammar)
        if pair is None:
            return None
        surfaces = pair
    elif slot.grammar == "component_total":
        match = _DENSE_COMPONENT_TOTAL.fullmatch(line)
        if match is None:
            return None
        surfaces = (match.group("weight"), match.group("volume"))
    elif slot.grammar == "numeric_only":
        match = _NUMERIC_ONLY_LINE.fullmatch(line)
        if match is None:
            return None
        surfaces = (match.group("value"),)
    else:
        rows = tuple(
            row
            for row in _dense_aggregate_measurement_rows("\n".join(lines))
            if tuple(value + 1 for value in row.value_line_indexes).count(slot.line_number)
        )
        if len(rows) != 1:
            return None
        row = rows[0]
        surfaces = row.value_surfaces
    if not 0 <= slot.value_index < len(surfaces):
        return None
    return _decimal_surface(surfaces[slot.value_index])


def _relation_holds(lines: Sequence[str], relation: _AdditiveRelation) -> bool:
    component_values = tuple(_slot_value(lines, slot) for slot in relation.components)
    total_value = _slot_value(lines, relation.total)
    if total_value is None or any(value is None for value in component_values):
        return False
    parsed_components = cast(tuple[tuple[Decimal, Decimal], ...], component_values)
    observed = sum(
        (
            value * slot.coefficient
            for slot, (value, _half_unit) in zip(
                relation.components, parsed_components, strict=True
            )
        ),
        start=Decimal(0),
    )
    tolerance = sum(
        (
            half_unit * slot.coefficient
            for slot, (_value, half_unit) in zip(
                relation.components, parsed_components, strict=True
            )
        ),
        start=total_value[1],
    )
    return abs(observed - total_value[0]) <= tolerance


def _container_number_line(lines: Sequence[str], number: str) -> int | None:
    key = re.sub(r"[^A-Z0-9]", "", number.upper())
    matches = tuple(
        index
        for index, line in enumerate(lines)
        if key and key in re.sub(r"[^A-Z0-9]", "", line.upper())
    )
    return matches[0] if len(matches) == 1 else None


def _container_component_slot(
    lines: Sequence[str], container_line: int
) -> tuple[_NumericSlot, _NumericSlot] | None:
    end = min(len(lines), container_line + 4)
    for index in range(container_line, end):
        if index > container_line and (
            not lines[index].strip() or _PAGE_MARKER.fullmatch(lines[index])
        ):
            break
        line = lines[index]
        seal = re.search(r"\bSEAL[ \t]*:", line, re.IGNORECASE)
        grammar: Literal["pair_before_seal", "trailing_pair"]
        if seal is not None and _numeric_pair(line, "pair_before_seal") is not None:
            grammar = "pair_before_seal"
        elif _numeric_pair(line, "trailing_pair") is not None:
            nearby = " ".join(lines[index : min(len(lines), index + 2)]).upper()
            if "KGM" not in nearby and " KG" not in nearby:
                continue
            grammar = "trailing_pair"
        else:
            continue
        pair = _numeric_pair(line, grammar)
        assert pair is not None
        parsed = tuple(_decimal_surface(value) for value in pair)
        if any(value is None for value in parsed):
            continue
        return (
            _NumericSlot(index + 1, grammar, 0, 1, cast(tuple[Decimal, Decimal], parsed[0])[0]),
            _NumericSlot(index + 1, grammar, 1, 1, cast(tuple[Decimal, Decimal], parsed[1])[0]),
        )
    return None


def _compile_container_component_relations(
    source: str, source_label: Mapping[str, Any]
) -> tuple[_AdditiveRelation, ...]:
    lines = source.splitlines()
    containers = _label_containers(source_label)
    if len(containers) < 2:
        return ()
    pairs: list[tuple[_NumericSlot, _NumericSlot]] = []
    for container in containers:
        number = container.get("containerNumber")
        if not isinstance(number, str):
            return ()
        line_index = _container_number_line(lines, number)
        if line_index is None:
            return ()
        pair = _container_component_slot(lines, line_index)
        if pair is None:
            return ()
        pairs.append(pair)
    if len({pair[0].line_number for pair in pairs}) != len(pairs):
        return ()
    totals = tuple(
        (index, match)
        for index, line in enumerate(lines)
        if (match := _DENSE_COMPONENT_TOTAL.fullmatch(line)) is not None
        and index > max(pair[0].line_number - 1 for pair in pairs)
    )
    if len(totals) != 1:
        return ()
    total_index, total_match = totals[0]
    output: list[_AdditiveRelation] = []
    for value_index, semantic_kind, group in (
        (0, "container_component_weight", "weight"),
        (1, "container_component_volume", "volume"),
    ):
        parsed_total = _decimal_surface(total_match.group(group))
        if parsed_total is None:
            return ()
        components = tuple(pair[value_index] for pair in pairs)
        total = _NumericSlot(
            total_index + 1,
            "component_total",
            value_index,
            1,
            parsed_total[0],
        )
        relation = _AdditiveRelation(
            relation_id=f"component-{semantic_kind}-L{total_index + 1:05d}",
            semantic_kind=cast(Any, semantic_kind),
            components=components,
            total=total,
        )
        if not _relation_holds(lines, relation):
            return ()
        output.append(relation)
    return tuple(output)


def _unique_subset(
    slots: Sequence[_NumericSlot], target: Decimal
) -> tuple[_NumericSlot, ...] | None:
    solutions: list[tuple[_NumericSlot, ...]] = []
    for size in range(1, len(slots) + 1):
        for selected in combinations(slots, size):
            if sum((slot.source_value for slot in selected), start=Decimal(0)) == target:
                solutions.append(selected)
                if len(solutions) > 1:
                    return None
    return solutions[0] if len(solutions) == 1 else None


def _compile_vertical_aggregate_relations(
    source: str, source_label: Mapping[str, Any]
) -> tuple[_AdditiveRelation, ...]:
    lines = source.splitlines()
    aggregates = _dense_aggregate_measurement_rows(source)
    container_count = len(_label_containers(source_label))
    if container_count < 2 or len(aggregates) != 1:
        return ()
    aggregate = aggregates[0]
    if aggregate.container_count != container_count:
        return ()
    first_total_index = min(aggregate.value_line_indexes)
    if not any(
        "GROSS" in line.upper() and "WEIGHT" in line.upper() for line in lines[:first_total_index]
    ):
        return ()
    unit_slots: dict[str, list[_NumericSlot]] = {"mass": [], "volume": []}
    # Flattened carrier tables may repeat the column headings on a later page after the actual
    # values. Unit/value adjacency is the stable ownership signal, so inspect the complete prefix
    # leading to the uniquely parsed shipment-total tuple rather than guessing which heading owns
    # the visually earlier columns.
    for index in range(1, first_total_index):
        match = _NUMERIC_ONLY_LINE.fullmatch(lines[index])
        if match is None:
            continue
        previous = lines[index - 1].strip().upper() if index else ""
        unit = (
            "mass"
            if previous in {"KG", "KGS", "KGM"}
            else ("volume" if previous in {"CBM", "MTQ", "M3", "MEASUREMENT"} else None)
        )
        if unit is None:
            continue
        parsed = _decimal_surface(match.group("value"))
        if parsed is None:
            continue
        unit_slots[unit].append(_NumericSlot(index + 1, "numeric_only", 0, 1, parsed[0]))
    if len(unit_slots["mass"]) > 12 or len(unit_slots["volume"]) > 12:
        return ()

    output: list[_AdditiveRelation] = []
    for value_index, semantic_kind, unit in (
        (0, "shipment_gross_weight", "mass"),
        (1, "shipment_tare_weight", "mass"),
        (2, "shipment_volume", "volume"),
    ):
        parsed_total = _decimal_surface(aggregate.value_surfaces[value_index])
        if parsed_total is None:
            continue
        components = _unique_subset(unit_slots[unit], parsed_total[0])
        if components is None:
            matching = tuple(
                slot
                for slot in unit_slots[unit]
                if sum((slot.source_value for _ in range(container_count)), start=Decimal(0))
                == parsed_total[0]
                and sum(1 for other in unit_slots[unit] if other.source_value == slot.source_value)
                == 1
            )
            if len(matching) != 1:
                continue
            components = (
                _NumericSlot(
                    matching[0].line_number,
                    matching[0].grammar,
                    matching[0].value_index,
                    container_count,
                    matching[0].source_value,
                ),
            )
        total = _NumericSlot(
            aggregate.value_line_indexes[value_index] + 1,
            "aggregate_tuple",
            value_index,
            1,
            parsed_total[0],
        )
        relation = _AdditiveRelation(
            relation_id=f"aggregate-{semantic_kind}-L{total.line_number:05d}",
            semantic_kind=cast(Any, semantic_kind),
            components=tuple(components),
            total=total,
        )
        if _relation_holds(lines, relation):
            output.append(relation)
    return tuple(output)


def _serial_cardinality(surface: str) -> int | None:
    match = _SERIAL_RANGE.fullmatch(surface)
    if match is None or len(match.group("start")) != len(match.group("end")):
        return None
    start = int(match.group("start"))
    end = int(match.group("end"))
    return end - start + 1 if end >= start else None


def _cargo_groups_and_packages(
    label: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, int]]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("certification label lacks documentPatch")
    raw_groups = patch.get("cargoGroups") or ()
    raw_packages = patch.get("cargoPackages") or ()
    if (
        not isinstance(raw_groups, Sequence)
        or isinstance(raw_groups, (str, bytes))
        or not isinstance(raw_packages, Sequence)
        or isinstance(raw_packages, (str, bytes))
    ):
        raise ValueError("certification cargo groups/packages are not sequences")
    groups = tuple(cast(Mapping[str, Any], row) for row in raw_groups if isinstance(row, Mapping))
    if len(groups) != len(raw_groups) or any(not isinstance(row, Mapping) for row in raw_packages):
        raise ValueError("certification cargo groups/packages contain non-object rows")
    quantities: dict[str, int] = {}
    for row in cast(Sequence[Mapping[str, Any]], raw_packages):
        group_id = row.get("groupId")
        quantity = row.get("quantity")
        if (
            isinstance(group_id, str)
            and isinstance(quantity, int)
            and not isinstance(quantity, bool)
        ):
            quantities[group_id] = quantities.get(group_id, 0) + quantity
    return groups, quantities


def _compile_serial_cardinality_relations(
    source_label: Mapping[str, Any], target_label: Mapping[str, Any]
) -> tuple[_SerialCardinalityRelation, ...]:
    source_groups, source_quantities = _cargo_groups_and_packages(source_label)
    target_groups, target_quantities = _cargo_groups_and_packages(target_label)
    target_by_id = {
        cast(str, row.get("groupId")): row
        for row in target_groups
        if isinstance(row.get("groupId"), str)
    }
    output: list[_SerialCardinalityRelation] = []
    for index, source_group in enumerate(source_groups):
        group_id = source_group.get("groupId")
        source_marks = source_group.get("marksAndNumbers") or ()
        if (
            not isinstance(group_id, str)
            or not isinstance(source_marks, Sequence)
            or isinstance(source_marks, (str, bytes))
            or any(not isinstance(value, str) for value in source_marks)
        ):
            continue
        source_ranges = tuple(
            cast(str, value) for value in source_marks if _serial_cardinality(cast(str, value))
        )
        source_cards = tuple(_serial_cardinality(value) for value in source_ranges)
        source_quantity = source_quantities.get(group_id)
        if (
            len(source_ranges) < 2
            or source_quantity is None
            or any(value is None for value in source_cards)
            or sum(cast(tuple[int, ...], source_cards)) != source_quantity
        ):
            continue
        target_group = target_by_id.get(group_id)
        target_quantity = target_quantities.get(group_id)
        if target_group is None or target_quantity is None:
            continue
        raw_target_marks = target_group.get("marksAndNumbers") or ()
        if (
            not isinstance(raw_target_marks, Sequence)
            or isinstance(raw_target_marks, (str, bytes))
            or any(not isinstance(value, str) for value in raw_target_marks)
        ):
            continue
        target_ranges = tuple(
            cast(str, value)
            for value in raw_target_marks
            if _serial_cardinality(cast(str, value)) is not None
        )
        if len(target_ranges) != len(source_ranges):
            continue
        output.append(
            _SerialCardinalityRelation(
                relation_id=f"cargo-serial-cardinality-{group_id}",
                group_index=index,
                source_quantity=source_quantity,
                target_quantity=target_quantity,
                target_range_surfaces=target_ranges,
            )
        )
    return tuple(output)


def _compile_additive_relations(
    source: str, source_label: Mapping[str, Any]
) -> tuple[_AdditiveRelation, ...]:
    return (
        *_compile_container_component_relations(source, source_label),
        *_compile_vertical_aggregate_relations(source, source_label),
    )


def _additive_relation_evidence_line_numbers(
    *,
    candidate_lines: Sequence[str],
    relation: _AdditiveRelation,
    target_label: Mapping[str, Any],
) -> tuple[int, ...]:
    evidence = {
        relation.total.line_number,
        *(slot.line_number for slot in relation.components),
    }
    if relation.semantic_kind.startswith("shipment_"):
        for container in _label_containers(target_label):
            number = container.get("containerNumber")
            if not isinstance(number, str):
                continue
            index = _container_number_line(candidate_lines, number)
            if index is not None:
                evidence.add(index + 1)

    if relation.semantic_kind != "container_component_weight":
        return tuple(sorted(evidence))
    components = tuple(_slot_value(candidate_lines, slot) for slot in relation.components)
    total = _slot_value(candidate_lines, relation.total)
    if total is None or any(value is None for value in components):
        return tuple(sorted(evidence))
    parsed_components = cast(tuple[tuple[Decimal, Decimal], ...], components)
    observed = sum(
        (
            value * slot.coefficient
            for slot, (value, _half_unit) in zip(
                relation.components, parsed_components, strict=True
            )
        ),
        start=Decimal(0),
    )
    tolerance = total[1] + sum(
        (
            half_unit * slot.coefficient
            for slot, (_value, half_unit) in zip(
                relation.components, parsed_components, strict=True
            )
        ),
        start=Decimal(0),
    )
    factor_mismatch = observed != 0 and (
        abs(total[0] - observed * 1000) <= tolerance or abs(observed - total[0] * 1000) <= tolerance
    )
    if not factor_mismatch:
        return tuple(sorted(evidence))
    next_number = relation.total.line_number + 1
    if next_number <= len(candidate_lines) and re.search(
        r"\bKGM\b", candidate_lines[next_number - 1], re.IGNORECASE
    ):
        evidence.add(next_number)
    for number in range(
        relation.total.line_number + 1,
        min(len(candidate_lines), relation.total.line_number + 20) + 1,
    ):
        match = _CORROBORATING_MASS_TOTAL.search(candidate_lines[number - 1])
        if match is None:
            continue
        parsed = _decimal_surface(match.group("value"))
        if parsed is not None and abs(parsed[0] - observed) <= parsed[1] + tolerance:
            evidence.add(number)
    return tuple(sorted(evidence))


_CONTAINER_PRINTED_EQUIPMENT_PATH = re.compile(
    r"^documentPatch\.containers\[(?P<index>[0-9]+)\]\.printedEquipmentSurface$"
)


def _aggregate_equipment_prefill_line_ids(
    rows: Sequence[AppliedDeterministicPrefill],
) -> tuple[str, ...]:
    output: list[str] = []
    for row in rows:
        indexes = {
            match.group("index")
            for path in row.targetPaths
            if (match := _CONTAINER_PRINTED_EQUIPMENT_PATH.fullmatch(path)) is not None
        }
        if len(indexes) >= 2:
            output.append(row.lineId)
    return tuple(dict.fromkeys(output))


def _money_slot(line: str, line_number: int, *, total: bool) -> _MonetarySlot | None:
    matches = tuple(_MONEY_TOKEN.finditer(line))
    if len(matches) != 1 or (not total and _MONEY_ONLY_LINE.fullmatch(line) is None):
        return None
    match = matches[0]
    parsed = _decimal_surface(match.group("amount"))
    if parsed is None:
        return None
    return _MonetarySlot(
        line_number=line_number,
        currency=match.group("currency").upper(),
        amount=parsed[0],
        half_unit=parsed[1],
        source_surface=match.group(0),
    )


def _compile_freight_schedule_relations(
    source_lines: Sequence[str],
) -> tuple[_FreightScheduleRelation, ...]:
    """Compile explicit, additive freight schedules whose monetary values are source-private."""

    output: list[_FreightScheduleRelation] = []
    headings = tuple(
        number
        for number, line in enumerate(source_lines, start=1)
        if _FREIGHT_CHARGES_HEADING.fullmatch(line) is not None
    )
    for heading_number in headings:
        upper = min(len(source_lines), heading_number + 80)
        total_numbers = tuple(
            number
            for number in range(heading_number + 1, upper + 1)
            if _TOTAL_FREIGHT_CHARGES.search(source_lines[number - 1]) is not None
            and _money_slot(source_lines[number - 1], number, total=True) is not None
        )
        if len(total_numbers) != 1:
            continue
        total_number = total_numbers[0]
        total = _money_slot(source_lines[total_number - 1], total_number, total=True)
        assert total is not None
        components = tuple(
            slot
            for number in range(heading_number + 1, total_number)
            if (slot := _money_slot(source_lines[number - 1], number, total=False)) is not None
        )
        if len(components) < 2 or any(slot.currency != total.currency for slot in components):
            continue
        component_sum = sum((slot.amount for slot in components), start=Decimal(0))
        tolerance = total.half_unit + sum((slot.half_unit for slot in components), start=Decimal(0))
        if abs(component_sum - total.amount) > tolerance:
            continue
        private_surfaces = tuple(
            sorted(
                {slot.source_surface for slot in (*components, total)},
                key=lambda value: (-len(value), value),
            )
        )
        output.append(
            _FreightScheduleRelation(
                relation_id=f"freight-schedule-L{heading_number:05d}-L{total_number:05d}",
                components=components,
                total=total,
                private_surfaces=private_surfaces,
            )
        )
    return tuple(output)


def _freight_schedule_holds(
    candidate_lines: Sequence[str], relation: _FreightScheduleRelation
) -> bool:
    if relation.total.line_number > len(candidate_lines):
        return False
    total = _money_slot(
        candidate_lines[relation.total.line_number - 1], relation.total.line_number, total=True
    )
    components = tuple(
        _money_slot(candidate_lines[slot.line_number - 1], slot.line_number, total=False)
        if slot.line_number <= len(candidate_lines)
        else None
        for slot in relation.components
    )
    if total is None or any(slot is None for slot in components):
        return False
    parsed = cast(tuple[_MonetarySlot, ...], components)
    if any(slot.currency != total.currency for slot in parsed):
        return False
    observed = sum((slot.amount for slot in parsed), start=Decimal(0))
    tolerance = total.half_unit + sum((slot.half_unit for slot in parsed), start=Decimal(0))
    return abs(observed - total.amount) <= tolerance


_COUNTED_EQUIPMENT_START = re.compile(
    r"(?<![A-Z0-9])(?P<count>[1-9][0-9,]*)[ \t]*X[ \t]*"
    r"(?P<size>20|40|45)(?![0-9])",
    re.IGNORECASE,
)
_FCL_COUNT = re.compile(r"(?<![A-Z0-9])(?P<count>[1-9][0-9,]*)[ \t]+FCL\b", re.IGNORECASE)
_PHYSICAL_MEASUREMENT = re.compile(
    r"^[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)*)[ \t]*"
    r"(?P<unit>KGS?|KGM|KILOGRAMS?|MTS?|METRIC[ \t]+TON(?:NE)?S?|"
    r"LBS?|POUNDS?|CBM|M3|M\^3|CUBIC[ \t]+MET(?:ER|RE)S?)[ \t.]*$",
    re.IGNORECASE,
)
_TOTAL_HEADING = re.compile(r"^[ \t]*TOTAL[ \t:]*$", re.IGNORECASE)

_PHYSICAL_MASS_FACTORS = {
    "KG": Decimal(1),
    "KGS": Decimal(1),
    "KGM": Decimal(1),
    "KILOGRAM": Decimal(1),
    "KILOGRAMS": Decimal(1),
    "MT": Decimal(1000),
    "MTS": Decimal(1000),
    "METRIC TON": Decimal(1000),
    "METRIC TONS": Decimal(1000),
    "METRIC TONNE": Decimal(1000),
    "METRIC TONNES": Decimal(1000),
    "LB": Decimal("0.45359237"),
    "LBS": Decimal("0.45359237"),
    "POUND": Decimal("0.45359237"),
    "POUNDS": Decimal("0.45359237"),
}
_PHYSICAL_VOLUME_FACTORS = {
    "CBM": Decimal(1),
    "M3": Decimal(1),
    "M^3": Decimal(1),
    "CUBIC METER": Decimal(1),
    "CUBIC METERS": Decimal(1),
    "CUBIC METRE": Decimal(1),
    "CUBIC METRES": Decimal(1),
}


def _equipment_pair(surface: str) -> tuple[str, str] | None:
    reviewed = review_source_equipment_surface(surface, temperature_present=False)
    if (
        reviewed.resolution != "reviewed_source_grammar"
        or reviewed.size_category is None
        or reviewed.type_category is None
    ):
        return None
    return reviewed.size_category, reviewed.type_category


@dataclass(slots=True)
class _EquipmentPairCache:
    """Document-scoped equipment parses; never retain raw OCR text beyond one evaluation."""

    by_surface: dict[str, tuple[str, str] | None]

    def resolve(self, surface: str) -> tuple[str, str] | None:
        if surface not in self.by_surface:
            self.by_surface[surface] = _equipment_pair(surface)
        return self.by_surface[surface]


def _resolve_equipment_pair(
    surface: str, equipment_cache: _EquipmentPairCache | None
) -> tuple[str, str] | None:
    return _equipment_pair(surface) if equipment_cache is None else equipment_cache.resolve(surface)


def _counted_equipment_assertions(
    lines: Sequence[str], *, equipment_cache: _EquipmentPairCache | None = None
) -> tuple[_EquipmentAssertion, ...]:
    output: list[_EquipmentAssertion] = []
    for line_number, line in enumerate(lines, start=1):
        starts = tuple(_COUNTED_EQUIPMENT_START.finditer(line))
        for ordinal, match in enumerate(starts):
            end = starts[ordinal + 1].start() if ordinal + 1 < len(starts) else len(line)
            segment = line[match.start() : end]
            pair = _resolve_equipment_pair(segment, equipment_cache)
            if pair is None:
                continue
            output.append(
                _EquipmentAssertion(
                    line_number=line_number,
                    count=int(match.group("count").replace(",", "")),
                    size_category=pair[0],
                    type_category=pair[1],
                    surface=segment.strip(),
                )
            )
    return tuple(output)


def _capacity_policy_payload(limits: TransportCapacityLimits) -> dict[str, str]:
    return {
        "policy": limits.policy,
        "publishedReferenceMarginFraction": str(limits.published_reference_margin_fraction),
        "twentyStandardPayloadKg": str(limits.twenty_standard_payload_kg),
        "twentyStandardVolumeM3": str(limits.twenty_standard_volume_m3),
        "fortyStandardPayloadKg": str(limits.forty_standard_payload_kg),
        "fortyStandardVolumeM3": str(limits.forty_standard_volume_m3),
        "fortyHighCubePayloadKg": str(limits.forty_high_cube_payload_kg),
        "fortyHighCubeVolumeM3": str(limits.forty_high_cube_volume_m3),
        "fortyFiveHighCubePayloadKg": str(limits.forty_five_high_cube_payload_kg),
        "fortyFiveHighCubeVolumeM3": str(limits.forty_five_high_cube_volume_m3),
        "outOfGaugePayloadKg": str(limits.out_of_gauge_payload_kg),
        "unclassifiedPayloadKg": str(limits.unclassified_payload_kg),
        "unclassifiedVolumeM3": str(limits.unclassified_volume_m3),
    }


def _physical_measurement(
    line: str,
) -> tuple[Literal["payload", "volume"], Decimal, Decimal] | None:
    match = _PHYSICAL_MEASUREMENT.fullmatch(line)
    if match is None:
        return None
    parsed = _decimal_surface(match.group("value"))
    if parsed is None:
        return None
    unit = " ".join(match.group("unit").upper().split())
    if unit in _PHYSICAL_MASS_FACTORS:
        factor = _PHYSICAL_MASS_FACTORS[unit]
        return "payload", parsed[0] * factor, parsed[1] * factor
    if unit in _PHYSICAL_VOLUME_FACTORS:
        factor = _PHYSICAL_VOLUME_FACTORS[unit]
        return "volume", parsed[0] * factor, parsed[1] * factor
    return None


def _assertion_capacity(
    assertion: _EquipmentAssertion,
    *,
    limits: TransportCapacityLimits,
    kind: Literal["payload", "volume"],
) -> Decimal | None:
    capacity = equipment_capacity(
        {
            "sizeCategory": assertion.size_category,
            "typeCategory": assertion.type_category,
        },
        limits,
    )
    per_unit = capacity.payload_kg if kind == "payload" else capacity.volume_m3
    return per_unit * assertion.count if per_unit is not None else None


def _compile_physical_capacity_relations(
    source_lines: Sequence[str],
    limits: TransportCapacityLimits,
    *,
    equipment_cache: _EquipmentPairCache | None = None,
) -> tuple[_PhysicalCapacityRelation, ...]:
    assertions = _counted_equipment_assertions(source_lines, equipment_cache=equipment_cache)
    output: list[_PhysicalCapacityRelation] = []
    for assertion in assertions:
        next_assertion = min(
            (row.line_number for row in assertions if row.line_number > assertion.line_number),
            default=len(source_lines) + 1,
        )
        upper = min(len(source_lines), assertion.line_number + 40, next_assertion - 1)
        measurements: list[tuple[int, Literal["payload", "volume"], Decimal, Decimal]] = []
        for number in range(assertion.line_number + 1, upper + 1):
            parsed = _physical_measurement(source_lines[number - 1])
            if parsed is None:
                continue
            previous = source_lines[max(0, number - 3) : number - 1]
            if not any(_TOTAL_HEADING.fullmatch(line) is not None for line in previous):
                continue
            measurements.append((number, *parsed))
        by_kind: dict[str, list[tuple[int, Decimal, Decimal]]] = defaultdict(list)
        for number, kind, value, half_unit in measurements:
            by_kind[kind].append((number, value, half_unit))
        for kind in ("payload", "volume"):
            rows = by_kind.get(kind, [])
            if len(rows) != 1:
                continue
            number, value, half_unit = rows[0]
            capacity = _assertion_capacity(assertion, limits=limits, kind=cast(Any, kind))
            if capacity is None or value - half_unit > capacity:
                continue
            output.append(
                _PhysicalCapacityRelation(
                    relation_id=(f"physical-{kind}-L{assertion.line_number:05d}-L{number:05d}"),
                    equipment_line_number=assertion.line_number,
                    measurement_line_number=number,
                    measurement_kind=cast(Any, kind),
                    source_value=value,
                    source_half_unit=half_unit,
                )
            )
    return tuple(output)


def _target_container_equipment(
    target_label: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...] | None:
    output: list[tuple[str, str, str]] = []
    for container in _label_containers(target_label):
        number = container.get("containerNumber")
        size = container.get("sizeCategory")
        category = container.get("typeCategory")
        if (
            not isinstance(number, str)
            or not isinstance(size, str)
            or not isinstance(category, str)
        ):
            return None
        output.append((number, size, category))
    return tuple(output)


def _source_equipment_line_numbers(
    source_lines: Sequence[str], *, equipment_cache: _EquipmentPairCache | None = None
) -> frozenset[int]:
    numbered = {
        row.line_number
        for row in _counted_equipment_assertions(source_lines, equipment_cache=equipment_cache)
    }
    for line_number, line in enumerate(source_lines, start=1):
        if (
            _FCL_COUNT.search(line) is not None
            or _resolve_equipment_pair(line, equipment_cache) is not None
        ):
            numbered.add(line_number)
    return frozenset(numbered)


def _local_equipment_pairs(
    *,
    candidate_lines: Sequence[str],
    container_line: int,
    target_number_keys: frozenset[str],
    counted_assertions: Sequence[_EquipmentAssertion],
    equipment_cache: _EquipmentPairCache | None = None,
) -> tuple[tuple[str, str, int], ...]:
    observed: list[tuple[str, str, int]] = []
    start = container_line
    for index in range(container_line - 1, max(-1, container_line - 3), -1):
        line = candidate_lines[index]
        if not line.strip() or _PAGE_MARKER.fullmatch(line):
            break
        if any(key in _identifier_key(line) for key in target_number_keys):
            break
        start = index
    counted_by_line: dict[int, list[_EquipmentAssertion]] = defaultdict(list)
    for assertion in counted_assertions:
        counted_by_line[assertion.line_number].append(assertion)
    aggregate_lines = {
        line_number
        for line_number, assertions in counted_by_line.items()
        if len(assertions) > 1 or assertions[0].count > 1
    }
    for index in range(start, min(len(candidate_lines), container_line + 7)):
        if index > container_line:
            line = candidate_lines[index]
            if not line.strip() or _PAGE_MARKER.fullmatch(line):
                break
            line_key = _identifier_key(line)
            if any(key in line_key for key in target_number_keys):
                break
        if index + 1 in aggregate_lines:
            continue
        pair = _resolve_equipment_pair(candidate_lines[index], equipment_cache)
        if pair is not None:
            observed.append((pair[0], pair[1], index + 1))
    return tuple(observed)


_POSITIVE_TO_ORDER = re.compile(
    r"^[ \t]*TO[ \t]+ORDER(?:[ \t]+OF(?:[ \t]+[^\r\n]+)?)?[ .\t]*$",
    re.IGNORECASE,
)
_CONSIGNED_TO_ORDER_HEADING = re.compile(
    r"^[ \t]*CONSIGNED[ \t]+TO[ \t]+ORDER[ \t]+OF[ \t]*:?\s*$",
    re.IGNORECASE,
)
_SEA_WAYBILL = re.compile(r"^[ \t]*SEA[ \t-]*WAYBILL[ \t]*$", re.IGNORECASE)
_NON_NEGOTIABLE = re.compile(r"^[ \t]*NON[ \t-]*NEGOTIABLE[ \t]*$", re.IGNORECASE)
_ORDER_LEGAL_CONTEXT = re.compile(
    r"\bNOT[ \t-]+NEGOTIABLE[ \t]+UNLESS[ \t]+CONSIGNED[ \t]+TO[ \t]+ORDER\b",
    re.IGNORECASE,
)
_ORIGINAL_COUNT_HEADING = re.compile(
    r"\b(?:NO\.?|NUMBER)[ \t]+OF[ \t]+ORIGINAL[ \t]+"
    r"(?:B/?L|BILLS?[ \t]+OF[ \t]+LADING)\b",
    re.IGNORECASE,
)
_SIGNED_ORIGINAL_CLAUSES = (
    re.compile(
        r"\[(?P<count>[0-9]+)\][ \t]+ORIGINAL[ \t]+BILLS?.{0,60}?HAVE[ \t]+BEEN[ \t]+SIGNED",
        re.IGNORECASE,
    ),
    re.compile(
        r"HAS[ \t]+SIGNED[ \t]+(?:[A-Z-]+[ \t]*)?\((?P<count>[0-9]+)\)"
        r"[ \t]+BILLS?[ \t]+OF[ \t]+LADING",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<count>[0-9]+)[ \t]+ORIGINAL[ \t]+BILLS?.{0,60}?HAVE[ \t]+BEEN[ \t]+SIGNED",
        re.IGNORECASE,
    ),
)
_CARDINALS = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
    "TEN": 10,
}


def _explicit_count(surface: str) -> int | None:
    parenthetical = re.search(r"\(([0-9]+)\)", surface)
    if parenthetical is not None:
        return int(parenthetical.group(1))
    digits = re.fullmatch(r"[ \t]*\[?([0-9]+)\]?[ \t]*", surface)
    if digits is not None:
        return int(digits.group(1))
    normalized = re.sub(r"[^A-Z]+", " ", surface.upper()).strip()
    return _CARDINALS.get(normalized)


def _original_count_fields(lines: Sequence[str]) -> tuple[tuple[int, int], ...]:
    output: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        if _ORIGINAL_COUNT_HEADING.search(line) is None:
            continue
        for value_index in range(index + 1, min(len(lines), index + 4)):
            value_line = lines[value_index]
            if not value_line.strip() or _PAGE_MARKER.fullmatch(value_line):
                continue
            count = _explicit_count(value_line)
            if count is not None:
                output.append((value_index + 1, count))
            break
    return tuple(output)


def _signed_original_counts(lines: Sequence[str]) -> tuple[tuple[int, int], ...]:
    output: list[tuple[int, int]] = []
    for line_number, line in enumerate(lines, start=1):
        matches = tuple(
            match
            for pattern in _SIGNED_ORIGINAL_CLAUSES
            if (match := pattern.search(line)) is not None
        )
        counts = {int(match.group("count")) for match in matches}
        if len(counts) == 1:
            output.append((line_number, next(iter(counts))))
    return tuple(output)


def _legal_relation_count(source: str, target_label: Mapping[str, Any]) -> int:
    lines = source.splitlines()
    patch = target_label.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("certification target label lacks documentPatch")
    negotiability = patch.get("negotiability")
    positive_order = sum(_POSITIVE_TO_ORDER.fullmatch(line) is not None for line in lines)
    consigned_headings = sum(
        _CONSIGNED_TO_ORDER_HEADING.fullmatch(line) is not None for line in lines
    )
    original_relation = int(
        bool(_original_count_fields(lines)) and bool(_signed_original_counts(lines))
    )
    if negotiability == "negotiable":
        return positive_order + original_relation
    if negotiability == "non_negotiable":
        explicit_seawaybill = any(_SEA_WAYBILL.fullmatch(line) is not None for line in lines)
        explicit_nonnegotiable = any(_NON_NEGOTIABLE.fullmatch(line) is not None for line in lines)
        return (
            consigned_headings if explicit_seawaybill and explicit_nonnegotiable else 0
        ) + original_relation
    return original_relation


def _line_evidence(
    candidate_lines: Sequence[str], line_ids: Sequence[str]
) -> tuple[InvariantEvidence, ...]:
    unique = tuple(sorted(set(line_ids), key=_line_number))
    if not unique:
        unique = ("L00001",)
    rows: list[InvariantEvidence] = []
    for line_id in unique:
        number = _line_number(line_id)
        if not 1 <= number <= len(candidate_lines):
            number = 1
            line_id = "L00001"
        rows.append(InvariantEvidence(lineId=line_id, currentLine=candidate_lines[number - 1]))
    return tuple(rows)


def _safe_repair(
    *,
    candidate_lines: Sequence[str],
    line_id: str,
    source_fragment: str,
    target_fragment: str,
) -> tuple[DeterministicLineRepair, ...]:
    number = _line_number(line_id)
    if not 1 <= number <= len(candidate_lines):
        return ()
    current_line = candidate_lines[number - 1]
    if (
        source_fragment == target_fragment
        or current_line.count(source_fragment) != 1
        or target_fragment in current_line
    ):
        return ()
    return (
        DeterministicLineRepair(
            method="replace_exact_fragment_v1",
            lineId=line_id,
            expectedCurrentLineSha256=sha256_bytes(current_line.encode("utf-8")),
            oldFragment=source_fragment,
            newFragment=target_fragment,
        ),
    )


def _finding(
    *,
    check_id: str,
    finding_kind: InvariantFindingKind,
    dimension: InvariantDimension,
    candidate_lines: Sequence[str],
    line_ids: Sequence[str],
    target_paths: Sequence[str],
    problem: str,
    repairs: Sequence[DeterministicLineRepair] = (),
) -> DeterministicCertificationFinding:
    payload = {
        "checkId": check_id,
        "findingKind": finding_kind,
        "dimension": dimension,
        "lineIds": list(line_ids),
        "targetPaths": list(target_paths),
        "problem": problem,
        "repairs": [row.model_dump(mode="json") for row in repairs],
    }
    return DeterministicCertificationFinding(
        invariantId="D" + sha256_bytes(canonical_json_bytes(payload))[:16],
        findingKind=finding_kind,
        dimension=dimension,
        evidence=_line_evidence(candidate_lines, line_ids),
        targetPaths=tuple(target_paths),
        problem=problem,
        repairs=tuple(repairs),
    )


def _validate_line_ids(
    *,
    source_lines: Sequence[str],
    rows: Sequence[InventoryCandidate | DeterministicInventoryEdit],
) -> None:
    for row in rows:
        for line_id in row.lineIds:
            number = _line_number(line_id)
            if not 1 <= number <= len(source_lines):
                raise ValueError(f"inventory line is outside the source: {line_id}")


def compile_certification_invariant_envelope(
    *,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    inventory: Sequence[InventoryCandidate],
    deterministic_edits: Sequence[DeterministicInventoryEdit],
    references: CertificationReferenceIndex,
    capacity_limits: TransportCapacityLimits,
    _equipment_cache: _EquipmentPairCache | None = None,
) -> CertificationInvariantEnvelope:
    """Validate and bind every deterministic input before an audit or provider call."""

    source_lines = source.splitlines()
    candidate_lines = candidate.splitlines()
    if not source_lines or not candidate_lines:
        raise ValueError("certification invariant input cannot be empty")
    _validate_line_ids(source_lines=source_lines, rows=inventory)
    _validate_line_ids(source_lines=source_lines, rows=deterministic_edits)
    inventory_ids = tuple(row.candidateId for row in inventory)
    if len(inventory_ids) != len(set(inventory_ids)):
        raise ValueError("certification inventory repeats a candidate ID")
    parsed: dict[str, tuple[BaseModel, ...]] = {
        key: _parse_rows(contract, key, model) for key, model in _CONTRACT_MODELS
    }
    source_label = _label(contract, "sourceLabel")
    target_label = _label(contract, "targetLabel")
    additive_relations = _compile_additive_relations(source, source_label)
    serial_relations = _compile_serial_cardinality_relations(source_label, target_label)
    source_equipment_lines = _source_equipment_line_numbers(
        source_lines, equipment_cache=_equipment_cache
    )
    legal_relations = _legal_relation_count(source, target_label)
    auxiliary_caption_slots = _edit_owned_auxiliary_rows(
        source=source, deterministic_edits=deterministic_edits
    )
    office_relations = _compile_office_reference_relations(
        source_lines=source_lines,
        inventory=inventory,
        references=references,
    )
    freight_relations = _compile_freight_schedule_relations(source_lines)
    physical_capacity_relations = _compile_physical_capacity_relations(
        source_lines,
        capacity_limits,
        equipment_cache=_equipment_cache,
    )
    carrier_retirement = _compile_carrier_retirement_relation(
        source_lines=source_lines,
        source_label=source_label,
        target_label=target_label,
    )
    party_address_relations = _compile_party_address_retirement_relations(
        source_lines=source_lines,
        inventory=inventory,
        source_label=source_label,
        target_label=target_label,
    )
    foreign_exporter_relations = _compile_foreign_exporter_relations(
        source_lines=source_lines,
        inventory=inventory,
        target_label=target_label,
        references=references,
    )
    container_package_scope_relations = _compile_container_package_scope_relations(
        target_label=target_label,
        operational_rows=cast(
            tuple[OperationalFlavorRequirement, ...],
            parsed["operationalFlavorRequirements"],
        ),
        references=references,
    )
    source_measurement_retirements = _compile_source_measurement_retirement_relations(
        source_lines=source_lines,
        target_label=target_label,
    )
    package_cargo_contradictions = _compile_package_cargo_contradictions(
        target_label=target_label,
        anchored_rows=cast(
            tuple[AnchoredScalarReplacementRequirement, ...],
            parsed["anchoredScalarReplacementRequirements"],
        ),
        cargo_rows=cast(
            tuple[CargoFlavorRewriteRequirement, ...],
            parsed["cargoFlavorRewriteRequirements"],
        ),
        references=references,
    )
    reference_receipt_sha256 = sha256_bytes(
        canonical_json_bytes(references.receipt.model_dump(mode="json"))
    )
    return CertificationInvariantEnvelope(
        schemaVersion=1,
        documentId=document_id,
        sourceTextSha256=sha256_bytes(source.encode("utf-8")),
        candidateTextSha256=sha256_bytes(candidate.encode("utf-8")),
        sourceContractSha256=sha256_bytes(canonical_json_bytes(contract)),
        inventorySha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in inventory])
        ),
        deterministicEditsSha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in deterministic_edits])
        ),
        sourceLabelSha256=sha256_bytes(canonical_json_bytes(source_label)),
        targetLabelSha256=sha256_bytes(canonical_json_bytes(target_label)),
        referenceReceiptSha256=reference_receipt_sha256,
        capacityPolicySha256=sha256_bytes(
            canonical_json_bytes(_capacity_policy_payload(capacity_limits))
        ),
        implementationSha256=sha256_file(_IMPLEMENTATION_PATH),
        requirementCounts=(
            *(
                InvariantRequirementCount(name=key, count=len(parsed[key]))
                for key, _model in _CONTRACT_MODELS
            ),
            InvariantRequirementCount(name="inventoryCandidates", count=len(inventory)),
            InvariantRequirementCount(
                name="deterministicInventoryEdits", count=len(deterministic_edits)
            ),
            InvariantRequirementCount(
                name="sourceProvenAdditiveRelations", count=len(additive_relations)
            ),
            InvariantRequirementCount(
                name="serialCardinalityRelations", count=len(serial_relations)
            ),
            InvariantRequirementCount(
                name="sourceEquipmentLines", count=len(source_equipment_lines)
            ),
            InvariantRequirementCount(name="sourceLegalRelations", count=legal_relations),
            InvariantRequirementCount(
                name="deterministicAuxiliaryCaptionSlots", count=len(auxiliary_caption_slots)
            ),
            InvariantRequirementCount(name="certificationReferenceIndexes", count=1),
            InvariantRequirementCount(
                name="sourceProvenOfficeRelations", count=len(office_relations)
            ),
            InvariantRequirementCount(
                name="sourceProvenOfficePhoneSlots",
                count=sum(len(row.contact_line_numbers) for row in office_relations),
            ),
            InvariantRequirementCount(
                name="sourceProvenOfficeIdentitySlots",
                count=sum(len(row.identity_line_numbers) for row in office_relations),
            ),
            InvariantRequirementCount(
                name="sourceProvenFreightSchedules", count=len(freight_relations)
            ),
            InvariantRequirementCount(
                name="sourcePrivateFreightMoneySurfaces",
                count=sum(len(row.private_surfaces) for row in freight_relations),
            ),
            InvariantRequirementCount(
                name="sourceProvenPhysicalCapacityRelations",
                count=len(physical_capacity_relations),
            ),
            InvariantRequirementCount(
                name="sourceCarrierIdentityRetirementSlots",
                count=(
                    len(carrier_retirement.source_identity_line_numbers)
                    if carrier_retirement is not None
                    else 0
                ),
            ),
            InvariantRequirementCount(
                name="sourceCarrierReferencePrefixSlots",
                count=(
                    len(carrier_retirement.reference_prefix_slots)
                    if carrier_retirement is not None
                    else 0
                ),
            ),
            InvariantRequirementCount(
                name="sourceCarrierRoleTemplateSlots",
                count=(
                    len(carrier_retirement.role_template_slots)
                    if carrier_retirement is not None
                    else 0
                ),
            ),
            InvariantRequirementCount(
                name="sourcePartyAddressRetirementRelations",
                count=len(party_address_relations),
            ),
            InvariantRequirementCount(
                name="sourcePartyAddressRetirementSlots",
                count=sum(len(row.address_slots) for row in party_address_relations),
            ),
            InvariantRequirementCount(
                name="sourceBoundForeignExporterRelations",
                count=len(foreign_exporter_relations),
            ),
            InvariantRequirementCount(
                name="containerPackageScopeRelations",
                count=len(container_package_scope_relations),
            ),
            InvariantRequirementCount(
                name="containerForbiddenShipmentPackageFacts",
                count=sum(len(row.forbidden_facts) for row in container_package_scope_relations),
            ),
            InvariantRequirementCount(
                name="sourceMeasurementRetirementRelations",
                count=len(source_measurement_retirements),
            ),
            InvariantRequirementCount(
                name="sourceMeasurementRetirementSlots",
                count=sum(len(row.line_numbers) for row in source_measurement_retirements),
            ),
            InvariantRequirementCount(
                name="targetPackageCargoContradictions",
                count=len(package_cargo_contradictions),
            ),
        ),
    )


def _normalized(value: str) -> str:
    return " ".join(value.upper().split())


def _identifier_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _target_semantics(inventory: InventoryCandidate) -> tuple[tuple[str, str], ...]:
    value = inventory.targetSemantics
    if not isinstance(value, list):
        return ()
    rows: list[tuple[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        path = raw.get("path")
        target = raw.get("target")
        if (
            isinstance(path, str)
            and isinstance(target, (str, int, float))
            and not isinstance(target, bool)
        ):
            rows.append((path, str(target)))
    return tuple(rows)


_HIGH_ENTROPY_PATH = re.compile(
    r"(?i)(?:containerNumber|sealNumber|billOfLadingNumber|booking|reference|"
    r"registration|tax|eori|acid|contactDetails\.(?:email|phone|fax))"
)


def _is_high_entropy_target(path: str, target: str) -> bool:
    key = _identifier_key(target)
    return (
        bool(_HIGH_ENTROPY_PATH.search(path))
        and len(key) >= 4
        and any(character.isdigit() for character in key)
    )


def _identifier_occurrence_lines(lines: Sequence[str], target: str) -> tuple[str, ...]:
    target_key = _identifier_key(target)
    if not target_key:
        return ()
    return tuple(
        f"L{number:05d}"
        for number, line in enumerate(lines, start=1)
        if target_key in _identifier_key(line)
    )


def _casefold_spans(line: str, surface: str) -> tuple[tuple[int, int], ...]:
    """Return literal, case-insensitive spans without changing Unicode offsets."""

    return tuple(match.span() for match in re.finditer(re.escape(surface), line, re.IGNORECASE))


def _target_authorized_auxiliary_spans(
    *,
    line_number: int,
    line: str,
    source_value: str,
    inventory: Sequence[InventoryCandidate],
    anchored_rows: Sequence[AnchoredScalarReplacementRequirement],
    deterministic_edits: Sequence[DeterministicInventoryEdit],
) -> tuple[tuple[int, int], ...]:
    """Locate source-value substrings that are part of an authoritative target identifier.

    A short source auxiliary can also be a legitimate substring of an independently
    authoritative target value.  The motivating example is the source SCAC ``MSCU`` inside a
    target-label container number such as ``MSCU9504539``.  Treating the whole candidate as an
    untyped string incorrectly calls that target-owned occurrence source leakage.  Authorization
    is deliberately narrow: an inventory semantic, anchored scalar, or deterministic edit must
    own this exact physical line and target value, and the observed occurrence must lie inside the
    rendered target span.  Inventory and anchored-label values must also have an identifier-shaped
    path.  A separate ``SCAC Code: MSCU`` occurrence on the same line remains unowned and fails.
    """

    line_id = f"L{line_number:05d}"
    source_key = source_value.casefold()
    spans: list[tuple[int, int]] = []
    for row in inventory:
        if line_id not in row.lineIds:
            continue
        for path, target in _target_semantics(row):
            if not _is_high_entropy_target(path, target) or source_key not in target.casefold():
                continue
            spans.extend(_casefold_spans(line, target))
    for anchored_row in anchored_rows:
        if (
            line_id not in anchored_row.sourceLineIds
            or source_key not in anchored_row.targetSurface.casefold()
            or not any(
                _is_high_entropy_target(path, anchored_row.targetSurface)
                for path in anchored_row.targetPaths
            )
        ):
            continue
        spans.extend(_casefold_spans(line, anchored_row.targetSurface))
    for edit in deterministic_edits:
        if (
            line_id not in edit.lineIds
            or source_key not in edit.sourceSurface.casefold()
            or source_key not in edit.targetSurface.casefold()
        ):
            continue
        spans.extend(_casefold_spans(line, edit.targetSurface))
    return tuple(sorted(set(spans)))


def _unretired_auxiliary_line_numbers(
    *,
    auxiliary: LocatedAuxiliaryValue,
    candidate_lines: Sequence[str],
    inventory: Sequence[InventoryCandidate],
    anchored_rows: Sequence[AnchoredScalarReplacementRequirement],
    deterministic_edits: Sequence[DeterministicInventoryEdit],
    ignored_line_numbers: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Return lines containing at least one unowned source-auxiliary occurrence."""

    surviving: list[int] = []
    for number, line in enumerate(candidate_lines, start=1):
        if number in ignored_line_numbers:
            continue
        occurrences = _casefold_spans(line, auxiliary.value)
        if not occurrences:
            continue
        allowed = _target_authorized_auxiliary_spans(
            line_number=number,
            line=line,
            source_value=auxiliary.value,
            inventory=inventory,
            anchored_rows=anchored_rows,
            deterministic_edits=deterministic_edits,
        )
        if any(
            not any(
                allowed_start <= start and end <= allowed_end
                for allowed_start, allowed_end in allowed
            )
            for start, end in occurrences
        ):
            surviving.append(number)
    return tuple(surviving)


def _deterministic_edit_owns_auxiliary_line(
    edit: DeterministicInventoryEdit,
    auxiliary: LocatedAuxiliaryValue,
    line_number: int,
) -> bool:
    if f"L{line_number:05d}" not in edit.lineIds:
        return False
    edit_key = edit.sourceSurface.casefold()
    value_key = auxiliary.value.casefold()
    return value_key == edit_key or value_key in edit_key or edit_key in value_key


def _jurisdiction_target_context(
    *,
    requirement: JurisdictionalSurfaceRequirement,
    candidate_lines: Sequence[str],
    inventory: Sequence[InventoryCandidate],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return rendered, compiler-owned route/party context for a customs mismatch.

    The jurisdiction requirement carries the resolved target country, but source-customs lines
    alone do not show the opposing target identity to a human reviewer.  Inventory target
    semantics provide exact physical ownership for the relevant export/import party and route
    fields.  Only a value actually rendered on its owned line is admitted as evidence.
    """

    prefixes: tuple[str, ...]
    if (
        requirement.rewriteBasis
        in {"target_party_country_changed", "target_party_registry_country_changed"}
        and requirement.targetPartyRole == "carrier"
    ):
        prefixes = ("documentPatch.parties.carrier.",)
    elif requirement.tradeDirection == "export":
        prefixes = (
            "documentPatch.parties.shipper.",
            "documentPatch.route.placeOfReceipt.",
            "documentPatch.route.portOfLoading.",
        )
    else:
        prefixes = (
            "documentPatch.parties.consignee.",
            "documentPatch.route.portOfDischarge.",
            "documentPatch.route.placeOfDelivery.",
        )
    line_ids: set[str] = set()
    target_paths: set[str] = set()
    for row in inventory:
        relevant = tuple(
            (path, target) for path, target in _target_semantics(row) if path.startswith(prefixes)
        )
        if not relevant:
            continue
        for line_id in row.lineIds:
            number = _line_number(line_id)
            if not 1 <= number <= len(candidate_lines):
                continue
            rendered = tuple(
                path
                for path, target in relevant
                if _normalized(target) in _normalized(candidate_lines[number - 1])
            )
            if rendered:
                line_ids.add(line_id)
                target_paths.update(rendered)
    return (
        tuple(sorted(line_ids, key=_line_number)),
        tuple(sorted(target_paths)),
    )


def _document_patch(label: Mapping[str, Any]) -> Mapping[str, Any]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("certification label lacks documentPatch")
    return patch


def _carrier_name(label: Mapping[str, Any]) -> str | None:
    parties = _document_patch(label).get("parties")
    if parties is None:
        return None
    if not isinstance(parties, Mapping):
        raise ValueError("certification label parties are not an object")
    carrier = parties.get("carrier")
    if carrier is None:
        return None
    if not isinstance(carrier, Mapping):
        raise ValueError("certification carrier party is not an object")
    name = carrier.get("name")
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip():
        raise ValueError("certification carrier name is not non-empty text")
    return name.strip()


def _distinctive_carrier_tokens(source_name: str, target_name: str) -> tuple[str, ...]:
    target_tokens = {
        token.upper() for token in _CARRIER_NAME_TOKEN.findall(target_name) if not token.isdigit()
    }
    return tuple(
        dict.fromkeys(
            token.upper()
            for token in _CARRIER_NAME_TOKEN.findall(source_name)
            if len(token) >= 4
            and not token.isdigit()
            and token.upper() not in _GENERIC_CARRIER_NAME_TOKENS
            and token.upper() not in target_tokens
        )
    )


def _token_on_line(line: str, token: str) -> bool:
    return (
        re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", line, re.IGNORECASE) is not None
    )


def _source_carrier_identity_line(
    *,
    source_lines: Sequence[str],
    line_number: int,
    distinctive_tokens: Sequence[str],
) -> bool:
    line = source_lines[line_number - 1]
    if _CARRIER_ROLE_CONTEXT.search(line) is not None:
        return True
    if _CARRIER_OFFICE_CONTEXT.search(line) is not None:
        return True
    for neighbor in range(max(1, line_number - 1), min(len(source_lines), line_number + 1) + 1):
        if _CARRIER_ROLE_CONTEXT.search(source_lines[neighbor - 1]) is not None:
            return True
    lexical = {
        token.upper()
        for token in _CARRIER_NAME_TOKEN.findall(line)
        if len(token) >= 4
        and token.upper() not in _GENERIC_CARRIER_NAME_TOKENS
        and not token.isdigit()
    }
    return bool(lexical) and lexical.issubset(set(distinctive_tokens))


def _compile_carrier_retirement_relation(
    *,
    source_lines: Sequence[str],
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> _CarrierRetirementRelation | None:
    """Bind source-carrier identity only when the complete target changes that party.

    Carrier-branded clauses are source-owned only on physical lines where a distinctive token
    from the source label already occurs.  Separately, a mixed identifier under an explicit
    ``Carrier's Reference`` caption is carrier-owned; retaining its alphabetic stem while merely
    changing the digits is not anonymization when the carrier itself changed.
    """

    source_name = _carrier_name(source_label)
    target_name = _carrier_name(target_label)
    if source_name is None or target_name is None:
        return None
    if _identifier_key(source_name) == _identifier_key(target_name):
        return None
    distinctive_tokens = _distinctive_carrier_tokens(source_name, target_name)
    identity_lines = tuple(
        number
        for number, line in enumerate(source_lines, start=1)
        if any(_token_on_line(line, token) for token in distinctive_tokens)
        and _source_carrier_identity_line(
            source_lines=source_lines,
            line_number=number,
            distinctive_tokens=distinctive_tokens,
        )
    )
    reference_slots: list[_CarrierReferencePrefixSlot] = []
    for number, line in enumerate(source_lines, start=1):
        if _CARRIER_REFERENCE_HEADING.search(line) is None:
            continue
        for match in _MIXED_IDENTIFIER.finditer(line):
            source_token = match.group(0)
            prefix_match = re.match(r"[A-Z]{4,}", source_token, re.IGNORECASE)
            if prefix_match is None:
                continue
            reference_slots.append(
                _CarrierReferencePrefixSlot(
                    line_number=number,
                    source_token=source_token,
                    source_prefix=prefix_match.group(0).upper(),
                )
            )
    role_template_slots: list[_CarrierRoleTemplateSlot] = []
    for number in identity_lines:
        line = source_lines[number - 1]
        context = _CARRIER_ROLE_CONTEXT.search(line)
        token_matches = tuple(
            match
            for token in distinctive_tokens
            for match in re.finditer(
                rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])",
                line,
                re.IGNORECASE,
            )
        )
        if context is None or not token_matches:
            continue
        first = min(match.start() for match in token_matches)
        last = max(match.end() for match in token_matches)
        if context.end() <= first:
            mutable_start = context.end()
            while mutable_start < len(line) and line[mutable_start].isspace():
                mutable_start += 1
            role_template_slots.append(
                _CarrierRoleTemplateSlot(
                    line_number=number,
                    mutable_start=mutable_start,
                    mutable_end=len(line),
                )
            )
        elif last <= context.start():
            mutable_end = context.start()
            while mutable_end > 0 and line[mutable_end - 1].isspace():
                mutable_end -= 1
            role_template_slots.append(
                _CarrierRoleTemplateSlot(
                    line_number=number,
                    mutable_start=0,
                    mutable_end=mutable_end,
                )
            )
    if not identity_lines and not reference_slots and not role_template_slots:
        return None
    return _CarrierRetirementRelation(
        source_name=source_name,
        target_name=target_name,
        distinctive_tokens=distinctive_tokens,
        source_identity_line_numbers=identity_lines,
        reference_prefix_slots=tuple(reference_slots),
        role_template_slots=tuple(role_template_slots),
    )


def _target_carrier_line_numbers(
    candidate_lines: Sequence[str], relation: _CarrierRetirementRelation
) -> tuple[int, ...]:
    target_key = _identifier_key(relation.target_name)
    return tuple(
        number
        for number, line in enumerate(candidate_lines, start=1)
        if target_key and target_key in _identifier_key(line)
    )


def _carrier_role_template_mismatch_line_numbers(
    *,
    source_lines: Sequence[str],
    candidate_lines: Sequence[str],
    relation: _CarrierRetirementRelation,
) -> tuple[int, ...]:
    return tuple(
        slot.line_number
        for slot in relation.role_template_slots
        if slot.line_number > len(candidate_lines)
        or not _skeleton_matches(
            source_lines[slot.line_number - 1],
            candidate_lines[slot.line_number - 1],
            ((slot.mutable_start, slot.mutable_end),),
        )
    )


def _carrier_identity_residue_line_numbers(
    candidate_lines: Sequence[str], relation: _CarrierRetirementRelation
) -> tuple[int, ...]:
    return tuple(
        number
        for number in relation.source_identity_line_numbers
        if number <= len(candidate_lines)
        and any(
            _token_on_line(candidate_lines[number - 1], token)
            for token in relation.distinctive_tokens
        )
    )


def _carrier_reference_prefix_residue_line_numbers(
    candidate_lines: Sequence[str], relation: _CarrierRetirementRelation
) -> tuple[int, ...]:
    residue: list[int] = []
    for slot in relation.reference_prefix_slots:
        if slot.line_number > len(candidate_lines):
            continue
        candidate_tokens = tuple(_MIXED_IDENTIFIER.findall(candidate_lines[slot.line_number - 1]))
        if any(
            token.upper().startswith(slot.source_prefix) and len(token) > len(slot.source_prefix)
            for token in candidate_tokens
        ):
            residue.append(slot.line_number)
    return tuple(dict.fromkeys(residue))


def _party_object(label: Mapping[str, Any], party_path: str) -> Mapping[str, Any] | None:
    prefix = "documentPatch.parties."
    if not party_path.startswith(prefix):
        raise ValueError(f"invalid certification party path: {party_path!r}")
    parties = _document_patch(label).get("parties")
    if not isinstance(parties, Mapping):
        return None
    selector = party_path[len(prefix) :]
    indexed = re.fullmatch(r"(?P<field>[A-Za-z][A-Za-z0-9]*)\[(?P<index>[0-9]+)\]", selector)
    if indexed is None:
        value = parties.get(selector)
    else:
        values = parties.get(indexed.group("field"))
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return None
        index = int(indexed.group("index"))
        value = values[index] if index < len(values) else None
    return value if isinstance(value, Mapping) else None


def _semantic_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in re.findall(r"[^\W_]+", value, re.UNICODE))


def _contains_token_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        tuple(haystack[start : start + len(needle)]) == tuple(needle)
        for start in range(len(haystack) - len(needle) + 1)
    )


def _longest_address_line_sequence(line: str, address: str) -> tuple[str, ...] | None:
    line_tokens = _semantic_tokens(line)
    address_tokens = _semantic_tokens(address)
    best: tuple[str, ...] = ()
    for address_start in range(len(address_tokens)):
        for line_start in range(len(line_tokens)):
            length = 0
            while (
                address_start + length < len(address_tokens)
                and line_start + length < len(line_tokens)
                and address_tokens[address_start + length] == line_tokens[line_start + length]
            ):
                length += 1
            candidate = address_tokens[address_start : address_start + length]
            if (sum(map(len, candidate)), len(candidate)) > (sum(map(len, best)), len(best)):
                best = candidate
    character_count = sum(map(len, best))
    numeric_identifier = (
        len(best) >= 2 and character_count >= 5 and all(token.isdigit() for token in best)
    )
    if character_count < 8 and not numeric_identifier:
        return None
    if not numeric_identifier and not any(
        len(token) >= 4 and token not in _GENERIC_ADDRESS_TOKENS for token in best
    ):
        return None
    if len(best) == 1 and len(best[0]) < 8:
        return None
    return best


def _paragraph_line_numbers(lines: Sequence[str], anchor: int) -> tuple[int, ...]:
    if not 1 <= anchor <= len(lines):
        raise ValueError(f"paragraph anchor is outside source text: {anchor}")
    lower = anchor
    while lower > 1:
        previous = lines[lower - 2]
        if not previous.strip() or _PAGE_MARKER.fullmatch(previous) is not None:
            break
        lower -= 1
    upper = anchor
    while upper < len(lines):
        following = lines[upper]
        if not following.strip() or _PAGE_MARKER.fullmatch(following) is not None:
            break
        upper += 1
    return tuple(range(lower, upper + 1))


def _compile_party_address_retirement_relations(
    *,
    source_lines: Sequence[str],
    inventory: Sequence[InventoryCandidate],
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[_PartyAddressRetirementRelation, ...]:
    """Bind address fragments to an inventory-proven changed party block.

    A source address is not searched globally.  At least one changed-source inventory owner must
    identify the exact party path and paragraph; only high-information contiguous fragments of
    that party's source-label address inside those physical paragraphs become retirement slots.
    Fragments already present in the target party are explicitly excluded.
    """

    anchors_by_party: dict[str, set[int]] = defaultdict(set)
    for row in inventory:
        if row.category != "changed_source_occurrence":
            continue
        for target_path in row.targetPaths:
            match = _PARTY_TARGET_PATH.fullmatch(target_path)
            if match is None:
                continue
            anchors_by_party[match.group("party")].update(
                _line_number(line_id) for line_id in row.lineIds
            )
    output: list[_PartyAddressRetirementRelation] = []
    for party_path, anchors in sorted(anchors_by_party.items()):
        source_party = _party_object(source_label, party_path)
        target_party = _party_object(target_label, party_path)
        if source_party is None or target_party is None:
            continue
        source_address = source_party.get("address")
        target_address = target_party.get("address")
        if not isinstance(source_address, str) or not source_address.strip():
            continue
        if isinstance(target_address, str) and _identifier_key(source_address) == _identifier_key(
            target_address
        ):
            continue
        target_tokens = tuple(
            _semantic_tokens(value)
            for key in ("name", "address", "city", "country")
            if isinstance((value := target_party.get(key)), str)
        )
        paragraph_numbers = {
            number for anchor in anchors for number in _paragraph_line_numbers(source_lines, anchor)
        }
        slots: list[_PartyAddressSlot] = []
        for number in sorted(paragraph_numbers):
            fragment = _longest_address_line_sequence(source_lines[number - 1], source_address)
            if fragment is None or any(
                _contains_token_sequence(surface, fragment) for surface in target_tokens
            ):
                continue
            slots.append(_PartyAddressSlot(line_number=number, source_tokens=fragment))
        if slots:
            output.append(
                _PartyAddressRetirementRelation(
                    party_path=party_path,
                    anchor_line_numbers=tuple(sorted(anchors)),
                    address_slots=tuple(slots),
                )
            )
    return tuple(output)


def _party_address_residue_line_numbers(
    candidate_lines: Sequence[str], relation: _PartyAddressRetirementRelation
) -> tuple[int, ...]:
    return tuple(
        slot.line_number
        for slot in relation.address_slots
        if slot.line_number <= len(candidate_lines)
        and _contains_token_sequence(
            _semantic_tokens(candidate_lines[slot.line_number - 1]), slot.source_tokens
        )
    )


def _compile_foreign_exporter_relations(
    *,
    source_lines: Sequence[str],
    inventory: Sequence[InventoryCandidate],
    target_label: Mapping[str, Any],
    references: CertificationReferenceIndex,
) -> tuple[_ForeignExporterRelation, ...]:
    """Resolve each explicit foreign-exporter paragraph to one target party.

    An inventory-owned party mapping wins.  When the compiler has no party owner for the block,
    the bill-of-lading shipper is the only schema role that asserts the exporter.  This rule does
    not infer addresses; it checks only explicit exporter name/country fields and ISO country
    identities printed inside the same source-bounded paragraph.
    """

    sections = tuple(
        dict.fromkeys(
            _paragraph_line_numbers(source_lines, number)
            for number, line in enumerate(source_lines, start=1)
            if _FOREIGN_EXPORTER_CAPTION.search(line) is not None
        )
    )
    output: list[_ForeignExporterRelation] = []
    for section in sections:
        section_set = set(section)
        mapped_parties: set[str] = set()
        for row in inventory:
            if not any(_line_number(line_id) in section_set for line_id in row.lineIds):
                continue
            for target_path in row.targetPaths:
                match = _PARTY_TARGET_PATH.fullmatch(target_path)
                if match is not None:
                    mapped_parties.add(match.group("party"))
        if len(mapped_parties) > 1:
            raise ValueError(
                "foreign-exporter paragraph maps to multiple target parties: "
                f"{sorted(mapped_parties)!r}"
            )
        party_path = (
            next(iter(mapped_parties)) if mapped_parties else "documentPatch.parties.shipper"
        )
        target_party = _party_object(target_label, party_path)
        if target_party is None:
            continue
        target_name = target_party.get("name")
        target_country = target_party.get("country")
        if (
            not isinstance(target_name, str)
            or not target_name.strip()
            or not isinstance(target_country, str)
            or not target_country.strip()
        ):
            continue
        target_country_codes = references.exact_country_codes(target_country)
        if len(target_country_codes) != 1:
            continue
        output.append(
            _ForeignExporterRelation(
                section_line_numbers=section,
                party_path=party_path,
                target_name=target_name,
                target_country=target_country,
                target_country_codes=target_country_codes,
            )
        )
    return tuple(output)


def _foreign_exporter_failure(
    *,
    candidate_lines: Sequence[str],
    relation: _ForeignExporterRelation,
    references: CertificationReferenceIndex,
) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    country_lines: list[int] = []
    name_lines: list[int] = []
    observed_country_codes: set[str] = set()
    name_values: list[str] = []
    section_set = set(relation.section_line_numbers)
    for number in relation.section_line_numbers:
        if number > len(candidate_lines):
            continue
        line = candidate_lines[number - 1]
        country_match = _FOREIGN_EXPORTER_COUNTRY.fullmatch(line)
        if country_match is not None:
            value = re.sub(
                r"[ \t]+COUNTRY[ \t]*$", "", country_match.group("value"), flags=re.IGNORECASE
            )
            country_lines.append(number)
            observed_country_codes.update(references.exact_country_codes(value))
            continue
        code_match = _FOREIGN_EXPORTER_CODE.fullmatch(line)
        if code_match is not None:
            country_lines.append(number)
            observed_country_codes.update(references.exact_country_codes(code_match.group("value")))
            continue
        name_match = _FOREIGN_EXPORTER_NAME.fullmatch(line)
        if name_match is None:
            continue
        value = name_match.group("value").strip()
        value_number = number
        if not value:
            value_number = number + 1
            if value_number not in section_set or value_number > len(candidate_lines):
                continue
            value = candidate_lines[value_number - 1].strip()
        if value:
            name_lines.append(value_number)
            name_values.append(value)

    target_name_key = _identifier_key(relation.target_name)
    country_failure = bool(country_lines) and (
        observed_country_codes != set(relation.target_country_codes)
    )
    name_failure = bool(name_values) and any(
        target_name_key not in _identifier_key(value) for value in name_values
    )
    explicit_lines = set(country_lines) | set(name_lines)
    unexpected_country_lines: list[int] = []
    for number in relation.section_line_numbers:
        if number in explicit_lines or number > len(candidate_lines):
            continue
        embedded = references.country_codes_in_text(candidate_lines[number - 1])
        if embedded and not embedded <= relation.target_country_codes:
            unexpected_country_lines.append(number)
    if not country_failure and not name_failure and not unexpected_country_lines:
        return None

    target_name_lines = tuple(
        number
        for number, line in enumerate(candidate_lines, start=1)
        if target_name_key and target_name_key in _identifier_key(line)
    )
    company_lines = tuple(
        number
        for number in relation.section_line_numbers
        if number <= len(candidate_lines)
        and _LEGAL_ENTITY_SURFACE.search(candidate_lines[number - 1]) is not None
        and ":" not in candidate_lines[number - 1]
        and len(_semantic_tokens(candidate_lines[number - 1])) >= 2
        and _FOREIGN_EXPORTER_CAPTION.search(candidate_lines[number - 1]) is None
    )
    evidence = tuple(
        sorted(
            {
                *(target_name_lines[:1]),
                *country_lines,
                *name_lines,
                *(company_lines[:1] if not name_lines else ()),
                *unexpected_country_lines,
            }
        )
    )
    problems: list[str] = []
    if country_failure:
        problems.append(
            f"explicit country codes {sorted(observed_country_codes)!r} do not equal target "
            f"{sorted(relation.target_country_codes)!r}"
        )
    if name_failure:
        problems.append(f"explicit name does not render target {relation.target_name!r}")
    if unexpected_country_lines:
        problems.append(
            "the same exporter paragraph contains a conflicting ISO country on lines "
            f"{tuple(unexpected_country_lines)!r}"
        )
    return evidence, tuple(problems)


def _cargo_description_surfaces(label: Mapping[str, Any]) -> tuple[str, ...]:
    raw_groups = _document_patch(label).get("cargoGroups") or ()
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        raise ValueError("certification cargo groups are not a sequence")
    surfaces: list[str] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            raise ValueError("certification cargo group is not an object")
        description = raw_group.get("description")
        if isinstance(description, str):
            surfaces.append(description)
        additional = raw_group.get("additionalInformation") or ()
        if not isinstance(additional, Sequence) or isinstance(additional, (str, bytes)):
            raise ValueError("certification cargo additional information is not a sequence")
        if any(not isinstance(value, str) for value in additional):
            raise ValueError("certification cargo additional information is not textual")
        surfaces.extend(cast(Sequence[str], additional))
    return tuple(surfaces)


def _cargo_owned_source_line(line: str, surfaces: Sequence[str]) -> bool:
    line_key = _identifier_key(line.strip(" \t'\"*"))
    line_words = tuple(re.findall(r"[A-Z]{2,}", line.upper()))
    if len(line_key) < 8 or len(line_words) < 2:
        return False
    return any(
        line_key in _identifier_key(surface) or _identifier_key(surface) in line_key
        for surface in surfaces
        if len(_identifier_key(surface)) >= 8
    )


def _vessel_cargo_collision_lines(
    *,
    source_lines: Sequence[str],
    candidate_lines: Sequence[str],
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[str, ...]:
    source_transport = _document_patch(source_label).get("transport")
    target_transport = _document_patch(target_label).get("transport")
    if not isinstance(source_transport, Mapping) or not isinstance(target_transport, Mapping):
        return ()
    source_vessel = source_transport.get("vesselName")
    target_vessel = target_transport.get("vesselName")
    if not isinstance(source_vessel, str) or not isinstance(target_vessel, str):
        return ()
    source_cargo = _cargo_description_surfaces(source_label)
    target_cargo = _cargo_description_surfaces(target_label)
    collisions = tuple(
        number
        for number, (source_line, candidate_line) in enumerate(
            zip(source_lines, candidate_lines, strict=True), start=1
        )
        if target_vessel.casefold() in candidate_line.casefold()
        and source_vessel.casefold() not in source_line.casefold()
        and _cargo_owned_source_line(source_line, source_cargo)
    )
    if not collisions:
        return ()
    evidence = set(collisions)
    for number in collisions:
        for neighbor in range(max(1, number - 1), min(len(candidate_lines), number + 1) + 1):
            if any(
                target_surface.casefold() in candidate_lines[neighbor - 1].casefold()
                for target_surface in target_cargo
            ):
                evidence.add(neighbor)
    return tuple(f"L{number:05d}" for number in sorted(evidence))


def _auxiliary_rows_by_line(
    rows: Sequence[LocatedAuxiliaryValue],
) -> dict[int, tuple[LocatedAuxiliaryValue, ...]]:
    grouped: dict[int, list[LocatedAuxiliaryValue]] = defaultdict(list)
    for row in rows:
        for number in row.line_numbers:
            grouped[number].append(row)
    return {number: tuple(values) for number, values in grouped.items()}


def _edit_owned_auxiliary_rows(
    *,
    source: str,
    deterministic_edits: Sequence[DeterministicInventoryEdit],
    source_auxiliary_rows: Sequence[LocatedAuxiliaryValue] | None = None,
) -> tuple[tuple[DeterministicInventoryEdit, int, str], ...]:
    """Bind deterministic edits to explicit source field captions.

    The value locator, rather than the edit's historical span, owns the caption boundary.  This
    matters when an older locator accidentally included part of a caption in ``sourceSurface``:
    certification must still see that changing ``FMC-OTI NO.`` into ``FMC-OTI EM.`` is a label
    mutation, not accept the bad span as authority.
    """

    source_by_line = _auxiliary_rows_by_line(
        locate_auxiliary_values(source) if source_auxiliary_rows is None else source_auxiliary_rows
    )
    output: list[tuple[DeterministicInventoryEdit, int, str]] = []
    for edit in deterministic_edits:
        for line_id in edit.lineIds:
            number = _line_number(line_id)
            line_rows = source_by_line.get(number, ())
            exact = tuple(
                row for row in line_rows if row.value.casefold() == edit.sourceSurface.casefold()
            )
            matches = exact or tuple(
                row
                for row in line_rows
                if row.value.casefold() in edit.sourceSurface.casefold()
                or edit.sourceSurface.casefold() in row.value.casefold()
            )
            if len(matches) == 1:
                output.append((edit, number, matches[0].category))
    return tuple(output)


def _label_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


def _is_phone_inventory_row(row: InventoryCandidate) -> bool:
    if row.category != "source_only_contact_identity" or not isinstance(
        row.targetSemantics, Mapping
    ):
        return False
    return row.targetSemantics.get("semanticField") in {"tel", "fax"}


def _compile_office_reference_relations(
    *,
    source_lines: Sequence[str],
    inventory: Sequence[InventoryCandidate],
    references: CertificationReferenceIndex,
) -> tuple[_OfficeReferenceRelation, ...]:
    """Compile only repeated source localities with an explicit, co-owned country field.

    This deliberately does not search arbitrary address blocks.  The inventory must already own
    at least two physical copies of one source locality, and a separate auxiliary inventory row in
    that exact span must name one ISO country consistent with the GeoNames source identity.
    """

    auxiliary = tuple(row for row in inventory if row.category == "changed_source_auxiliary_copy")
    output: list[_OfficeReferenceRelation] = []
    for locality_row in auxiliary:
        locality_numbers = tuple(sorted({_line_number(value) for value in locality_row.lineIds}))
        if len(locality_numbers) < 2:
            continue
        if any(
            locality_row.sourceSurface.casefold() not in source_lines[number - 1].casefold()
            for number in locality_numbers
        ):
            continue
        source_locality_ids = references.exact_locality_ids(locality_row.sourceSurface)
        if not source_locality_ids:
            continue
        source_locality_countries = references.locality_country_codes(source_locality_ids)
        lower, upper = min(locality_numbers), max(locality_numbers)
        country_candidates: list[tuple[InventoryCandidate, frozenset[str]]] = []
        for country_row in auxiliary:
            if country_row.candidateId == locality_row.candidateId:
                continue
            country_codes = references.exact_country_codes(country_row.sourceSurface)
            country_numbers = tuple(_line_number(value) for value in country_row.lineIds)
            if (
                len(country_codes) != 1
                or not country_codes <= source_locality_countries
                or not country_numbers
                or any(not lower <= number <= upper for number in country_numbers)
                or any(
                    country_row.sourceSurface.casefold() not in source_lines[number - 1].casefold()
                    for number in country_numbers
                )
            ):
                continue
            country_candidates.append((country_row, country_codes))
        if len(country_candidates) != 1:
            continue
        country_row, _source_country = country_candidates[0]
        country_numbers = tuple(sorted({_line_number(value) for value in country_row.lineIds}))
        contact_numbers = tuple(
            sorted(
                {
                    _line_number(line_id)
                    for row in inventory
                    if _is_phone_inventory_row(row)
                    for line_id in row.lineIds
                    if lower <= _line_number(line_id) <= upper
                }
            )
        )
        paragraph = _paragraph_line_numbers(source_lines, lower)
        identity_numbers: tuple[int, ...] = ()
        for heading_number in paragraph:
            if (
                heading_number >= lower
                or _OFFICE_SECTION_HEADING.fullmatch(source_lines[heading_number - 1]) is None
            ):
                continue
            candidate_number = heading_number + 1
            if candidate_number < lower and source_lines[candidate_number - 1].strip():
                identity_numbers = (candidate_number,)
                break
        output.append(
            _OfficeReferenceRelation(
                relation_id=f"office-{locality_row.candidateId}-{country_row.candidateId}",
                identity_line_numbers=identity_numbers,
                locality_line_numbers=locality_numbers,
                country_line_numbers=country_numbers,
                contact_line_numbers=contact_numbers,
                target_paths=tuple(sorted(set(locality_row.targetPaths))),
            )
        )
    return tuple(output)


def _mutable_source_spans(
    *,
    source_lines: Sequence[str],
    contract_rows: Mapping[str, tuple[BaseModel, ...]],
    inventory: Sequence[InventoryCandidate],
    deterministic_edits: Sequence[DeterministicInventoryEdit],
) -> dict[int, list[tuple[int, int]] | None]:
    """Return explicit grammar-owned intervals; ``None`` means the whole line is mutable.

    Inventory ownership alone cannot protect the surrounding bytes.  A city scalar on an address
    line, for example, does not prove that the source postal code must survive a country change.
    Only compiler requirements with their own exact rendering grammar participate in the line
    skeleton; broader inventory rows remain useful for target-location checks, not byte locks.
    """

    spans: dict[int, list[tuple[int, int]] | None] = {}

    def add(line_id: str, surface: str) -> None:
        number = _line_number(line_id)
        line = source_lines[number - 1]
        if spans.get(number) is None and number in spans:
            return
        matches = tuple(re.finditer(re.escape(surface), line))
        if not matches:
            spans[number] = None
            return
        spans.setdefault(number, [])
        assert spans[number] is not None
        cast(list[tuple[int, int]], spans[number]).extend(match.span() for match in matches)

    for surface_row in cast(
        tuple[SurfaceRenderingRequirement, ...], contract_rows["surfaceRenderingRequirements"]
    ):
        if surface_row.kind != "carrier_principal_identity":
            continue
        for line_id in surface_row.sourceLineIds:
            add(line_id, surface_row.sourceSurface)

    # A second mutation on the same physical row makes the remaining bytes non-static unless its
    # source surface is already fully contained by an explicit grammar-owned interval.
    owners: tuple[InventoryCandidate | DeterministicInventoryEdit, ...] = (
        *inventory,
        *deterministic_edits,
    )
    for owner in owners:
        for line_id in owner.lineIds:
            number = _line_number(line_id)
            if number not in spans or spans[number] is None:
                continue
            matches = tuple(re.finditer(re.escape(owner.sourceSurface), source_lines[number - 1]))
            explicit = cast(list[tuple[int, int]], spans[number])
            if not matches or any(
                not any(start <= match.start() and match.end() <= end for start, end in explicit)
                for match in matches
            ):
                spans[number] = None
    return spans


def _skeleton_matches(
    source_line: str, candidate_line: str, spans: Sequence[tuple[int, int]]
) -> bool:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    static: list[str] = []
    cursor = 0
    for start, end in merged:
        static.append(source_line[cursor:start])
        cursor = end
    static.append(source_line[cursor:])
    pattern = r"\A" + r".*?".join(re.escape(value) for value in static) + r"\Z"
    return re.fullmatch(pattern, candidate_line) is not None


def audit_certification_invariants(
    *,
    envelope: CertificationInvariantEnvelope,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    inventory: Sequence[InventoryCandidate],
    deterministic_edits: Sequence[DeterministicInventoryEdit],
    references: CertificationReferenceIndex,
    capacity_limits: TransportCapacityLimits,
    _equipment_cache: _EquipmentPairCache | None = None,
) -> CertificationInvariantAudit:
    """Replay all host-owned facts and return exact, deduplicated failures."""

    observed_envelope = compile_certification_invariant_envelope(
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        references=references,
        capacity_limits=capacity_limits,
        _equipment_cache=_equipment_cache,
    )
    if observed_envelope != envelope:
        raise ValueError("certification invariant envelope differs from exact inputs")
    source_lines = source.splitlines()
    candidate_lines = candidate.splitlines()
    contract_rows: dict[str, tuple[BaseModel, ...]] = {
        key: _parse_rows(contract, key, model) for key, model in _CONTRACT_MODELS
    }
    source_label = _label(contract, "sourceLabel")
    target_label = _label(contract, "targetLabel")
    by_check: dict[str, list[DeterministicCertificationFinding]] = defaultdict(list)

    def add(check_id: str, finding: DeterministicCertificationFinding) -> None:
        by_check[check_id].append(finding)

    carrier_retirement = _compile_carrier_retirement_relation(
        source_lines=source_lines,
        source_label=source_label,
        target_label=target_label,
    )
    target_carrier_lines: tuple[int, ...] = ()
    carrier_role_template_mismatches: tuple[int, ...] = ()
    if carrier_retirement is not None:
        target_carrier_lines = _target_carrier_line_numbers(candidate_lines, carrier_retirement)
        identity_residue = _carrier_identity_residue_line_numbers(
            candidate_lines, carrier_retirement
        )
        if identity_residue:
            add(
                "source_carrier_identity_retirement",
                _finding(
                    check_id="source_carrier_identity_retirement",
                    finding_kind="source_only_private_or_auxiliary_fact",
                    dimension="source_private_and_auxiliary_replacement",
                    candidate_lines=candidate_lines,
                    line_ids=tuple(
                        f"L{number:05d}" for number in (*target_carrier_lines, *identity_residue)
                    ),
                    target_paths=("documentPatch.parties.carrier.name",),
                    problem=(
                        f"Source-carrier identity {carrier_retirement.source_name!r} survives "
                        "on source-owned carrier-branded lines after the complete target changed "
                        f"the carrier to {carrier_retirement.target_name!r}."
                    ),
                ),
            )
        deterministic_reference_lines = frozenset(
            slot.line_number
            for slot in carrier_retirement.reference_prefix_slots
            if any(
                f"L{slot.line_number:05d}" in edit.lineIds
                and _identifier_key(edit.sourceSurface) == _identifier_key(slot.source_token)
                for edit in deterministic_edits
            )
        )
        reference_residue = tuple(
            number
            for number in _carrier_reference_prefix_residue_line_numbers(
                candidate_lines, carrier_retirement
            )
            if number not in deterministic_reference_lines
        )
        if reference_residue:
            add(
                "source_carrier_reference_retirement",
                _finding(
                    check_id="source_carrier_reference_retirement",
                    finding_kind="source_only_private_or_auxiliary_fact",
                    dimension="source_private_and_auxiliary_replacement",
                    candidate_lines=candidate_lines,
                    line_ids=tuple(
                        f"L{number:05d}" for number in (*target_carrier_lines, *reference_residue)
                    ),
                    target_paths=("documentPatch.parties.carrier.name",),
                    problem=(
                        "An explicit Carrier's Reference retains the alphabetic stem of its "
                        "source carrier-owned identifier after the complete target changed the "
                        f"carrier from {carrier_retirement.source_name!r} to "
                        f"{carrier_retirement.target_name!r}."
                    ),
                ),
            )
        carrier_role_template_mismatches = _carrier_role_template_mismatch_line_numbers(
            source_lines=source_lines,
            candidate_lines=candidate_lines,
            relation=carrier_retirement,
        )
        if carrier_role_template_mismatches:
            add(
                "source_carrier_role_template",
                _finding(
                    check_id="source_carrier_role_template",
                    finding_kind="format_or_model_artifact",
                    dimension="template_format_and_model_artifacts",
                    candidate_lines=candidate_lines,
                    line_ids=tuple(
                        f"L{number:05d}"
                        for number in (*target_carrier_lines, *carrier_role_template_mismatches)
                    ),
                    target_paths=("documentPatch.parties.carrier.name",),
                    problem=(
                        "A source carrier-role line changed static legal/template text outside "
                        "the carrier-identity slot."
                    ),
                ),
            )

    for measurement_relation in _compile_source_measurement_retirement_relations(
        source_lines=source_lines,
        target_label=target_label,
    ):
        residue_lines = _source_measurement_residue_line_numbers(
            candidate_lines, measurement_relation
        )
        if not residue_lines:
            continue
        add(
            "source_measurement_retirement",
            _finding(
                check_id="source_measurement_retirement",
                finding_kind="source_only_private_or_auxiliary_fact",
                dimension="source_private_and_auxiliary_replacement",
                candidate_lines=candidate_lines,
                line_ids=tuple(f"L{number:05d}" for number in residue_lines),
                target_paths=(),
                problem=(
                    f"Exact source {measurement_relation.measurement_kind} identity from "
                    f"{measurement_relation.relation_id!r} survives in source-owned physical "
                    "measurement slots although the complete synthetic target omits that fact."
                ),
            ),
        )

    for party_relation in _compile_party_address_retirement_relations(
        source_lines=source_lines,
        inventory=inventory,
        source_label=source_label,
        target_label=target_label,
    ):
        residue_lines = _party_address_residue_line_numbers(candidate_lines, party_relation)
        if not residue_lines:
            continue
        add(
            "source_party_address_retirement",
            _finding(
                check_id="source_party_address_retirement",
                finding_kind="source_only_private_or_auxiliary_fact",
                dimension="source_private_and_auxiliary_replacement",
                candidate_lines=candidate_lines,
                line_ids=tuple(
                    f"L{number:05d}"
                    for number in (*party_relation.anchor_line_numbers, *residue_lines)
                ),
                target_paths=(f"{party_relation.party_path}.address",),
                problem=(
                    f"Source address fragments survive inside the inventory-owned changed "
                    f"party block for {party_relation.party_path!r}."
                ),
            ),
        )

    for exporter_relation in _compile_foreign_exporter_relations(
        source_lines=source_lines,
        inventory=inventory,
        target_label=target_label,
        references=references,
    ):
        failure = _foreign_exporter_failure(
            candidate_lines=candidate_lines,
            relation=exporter_relation,
            references=references,
        )
        if failure is None:
            continue
        evidence_lines, problems = failure
        add(
            "foreign_exporter_identity",
            _finding(
                check_id="foreign_exporter_identity",
                finding_kind="route_or_jurisdiction_mismatch",
                dimension="route_jurisdiction_and_identifiers",
                candidate_lines=candidate_lines,
                line_ids=tuple(f"L{number:05d}" for number in evidence_lines),
                target_paths=(
                    f"{exporter_relation.party_path}.name",
                    f"{exporter_relation.party_path}.country",
                ),
                problem="Foreign-exporter identity is incoherent: " + "; ".join(problems) + ".",
            ),
        )

    for package_scope in _compile_container_package_scope_relations(
        target_label=target_label,
        operational_rows=cast(
            tuple[OperationalFlavorRequirement, ...],
            contract_rows["operationalFlavorRequirements"],
        ),
        references=references,
    ):
        if package_scope.line_number > len(candidate_lines):
            continue
        candidate_line = candidate_lines[package_scope.line_number - 1]
        leaked_facts = tuple(
            fact
            for fact in package_scope.forbidden_facts
            if _package_fact_in_line(candidate_line, fact)
        )
        if not leaked_facts:
            continue
        add(
            "container_package_scope",
            _finding(
                check_id="container_package_scope",
                finding_kind="cargo_or_dangerous_goods_mismatch",
                dimension="cargo_packages_and_dangerous_goods",
                candidate_lines=candidate_lines,
                line_ids=(f"L{package_scope.line_number:05d}",),
                target_paths=tuple(fact.package_path for fact in leaked_facts),
                problem=(
                    f"Container-local package row for {package_scope.target_container_number!r} "
                    "also claims shipment-wide package facts "
                    f"{tuple((fact.quantity, fact.printed_name) for fact in leaked_facts)!r}."
                ),
            ),
        )

    package_cargo_contradictions = _compile_package_cargo_contradictions(
        target_label=target_label,
        anchored_rows=cast(
            tuple[AnchoredScalarReplacementRequirement, ...],
            contract_rows["anchoredScalarReplacementRequirements"],
        ),
        cargo_rows=cast(
            tuple[CargoFlavorRewriteRequirement, ...],
            contract_rows["cargoFlavorRewriteRequirements"],
        ),
        references=references,
    )
    for package_relation in package_cargo_contradictions:
        add(
            "package_cargo_semantics",
            _finding(
                check_id="package_cargo_semantics",
                finding_kind="cargo_or_dangerous_goods_mismatch",
                dimension="cargo_packages_and_dangerous_goods",
                candidate_lines=candidate_lines,
                line_ids=(
                    *package_relation.package_line_ids,
                    *package_relation.cargo_line_ids,
                ),
                target_paths=package_relation.target_paths,
                problem=(
                    f"Pinned category {package_relation.category_token!r} denotes "
                    f"{package_relation.package_display_name!r}, but the same target cargo "
                    f"group explicitly describes {package_relation.target_description!r} as "
                    "parts. The "
                    "upstream target is internally contradictory and must be regenerated."
                ),
            ),
        )

    topology_lines = ("L00001",)
    if len(source_lines) != len(candidate_lines):
        add(
            "line_topology",
            _finding(
                check_id="line_topology",
                finding_kind="format_or_model_artifact",
                dimension="template_format_and_model_artifacts",
                candidate_lines=candidate_lines or [""],
                line_ids=topology_lines,
                target_paths=(),
                problem=(
                    f"OCR line count changed from {len(source_lines)} to {len(candidate_lines)}."
                ),
            ),
        )
    else:
        source_markers = tuple(
            (number, line)
            for number, line in enumerate(source_lines, start=1)
            if _PAGE_MARKER.fullmatch(line)
        )
        candidate_markers = tuple(
            (number, line)
            for number, line in enumerate(candidate_lines, start=1)
            if _PAGE_MARKER.fullmatch(line)
        )
        if source_markers != candidate_markers:
            add(
                "line_topology",
                _finding(
                    check_id="line_topology",
                    finding_kind="format_or_model_artifact",
                    dimension="template_format_and_model_artifacts",
                    candidate_lines=candidate_lines,
                    line_ids=topology_lines,
                    target_paths=(),
                    problem="Page markers or their physical line positions changed.",
                ),
            )
        for number, (source_line, current_line) in enumerate(
            zip(source_lines, candidate_lines, strict=True), start=1
        ):
            if bool(source_line.strip()) != bool(current_line.strip()):
                line_id = f"L{number:05d}"
                add(
                    "line_topology",
                    _finding(
                        check_id="line_topology",
                        finding_kind="format_or_model_artifact",
                        dimension="template_format_and_model_artifacts",
                        candidate_lines=candidate_lines,
                        line_ids=(line_id,),
                        target_paths=(),
                        problem=f"Occupied/blank topology changed on {line_id}.",
                    ),
                )

    if len(source_lines) == len(candidate_lines):
        for line_id in _inline_slot_topology_mismatches(source, candidate):
            add(
                "template_structure",
                _finding(
                    check_id="template_structure",
                    finding_kind="format_or_model_artifact",
                    dimension="template_format_and_model_artifacts",
                    candidate_lines=candidate_lines,
                    line_ids=(line_id,),
                    target_paths=(),
                    problem=f"Inline labeled-slot topology changed on {line_id}.",
                ),
            )
        for line_id in _structural_line_prefix_mismatches(source, candidate):
            add(
                "template_structure",
                _finding(
                    check_id="template_structure",
                    finding_kind="format_or_model_artifact",
                    dimension="template_format_and_model_artifacts",
                    candidate_lines=candidate_lines,
                    line_ids=(line_id,),
                    target_paths=(),
                    problem=f"Numbered field/clause prefix changed on {line_id}.",
                ),
            )

    status_rows = cast(
        tuple[SourceStatusPreservationRequirement, ...],
        contract_rows["sourceStatusPreservationRequirements"],
    )
    if not _source_status_surfaces_preserved(candidate, status_rows):
        for status_row in status_rows:
            if candidate_lines.count(status_row.sourceSurface) == status_row.sourceOccurrences:
                continue
            source_ids = tuple(
                f"L{number:05d}"
                for number, line in enumerate(source_lines, start=1)
                if line == status_row.sourceSurface
            )
            add(
                "source_status",
                _finding(
                    check_id="source_status",
                    finding_kind="format_or_model_artifact",
                    dimension="template_format_and_model_artifacts",
                    candidate_lines=candidate_lines,
                    line_ids=source_ids,
                    target_paths=(),
                    problem=(
                        f"Protected source status surface {status_row.sourceSurface!r} must occur "
                        f"{status_row.sourceOccurrences} time(s)."
                    ),
                ),
            )

    for surface_row in cast(
        tuple[SurfaceRenderingRequirement, ...], contract_rows["surfaceRenderingRequirements"]
    ):
        if _required_surface_rendered(candidate, surface_row):
            continue
        surface_repairs = tuple(
            repair
            for line_id in surface_row.sourceLineIds
            for repair in _safe_repair(
                candidate_lines=candidate_lines,
                line_id=line_id,
                source_fragment=surface_row.sourceSurface,
                target_fragment=surface_row.targetSurface,
            )
        )
        add(
            "required_surfaces",
            _finding(
                check_id="required_surfaces",
                finding_kind=(
                    "party_or_legal_identity_mismatch"
                    if surface_row.kind in {"carrier_header_identity", "carrier_principal_identity"}
                    else "equipment_or_temperature_mismatch"
                    if "equipment" in surface_row.kind
                    else "repeated_or_derived_fact_mismatch"
                ),
                dimension=(
                    "party_legal_and_negotiability"
                    if surface_row.kind in {"carrier_header_identity", "carrier_principal_identity"}
                    else "equipment_temperature_and_capacity"
                    if "equipment" in surface_row.kind
                    else "repeated_and_derived_relations"
                ),
                candidate_lines=candidate_lines,
                line_ids=surface_row.sourceLineIds,
                target_paths=(surface_row.targetPath,),
                problem=(
                    f"Required target surface {surface_row.targetSurface!r} is not rendered in "
                    "every "
                    "compiler-owned source slot or its retired source surface remains."
                ),
                repairs=surface_repairs,
            ),
        )

    jurisdiction_rows = cast(
        tuple[JurisdictionalSurfaceRequirement, ...],
        contract_rows["jurisdictionalSurfaceRequirements"],
    )
    failed_jurisdiction_caption_sources: dict[int, tuple[str, ...]] = defaultdict(tuple)
    for jurisdiction_row in jurisdiction_rows:
        if _jurisdictional_surfaces_rendered(
            candidate,
            (jurisdiction_row,),
            source_value=source,
        ):
            continue
        for line_id in jurisdiction_row.sourceLineIds:
            number = _line_number(line_id)
            failed_jurisdiction_caption_sources[number] = (
                *failed_jurisdiction_caption_sources[number],
                _label_key(jurisdiction_row.sourceSurface),
            )
        context_line_ids, context_target_paths = _jurisdiction_target_context(
            requirement=jurisdiction_row,
            candidate_lines=candidate_lines,
            inventory=inventory,
        )
        repairs = tuple(
            repair
            for line_id in jurisdiction_row.sourceLineIds
            for repair in _safe_repair(
                candidate_lines=candidate_lines,
                line_id=line_id,
                source_fragment=jurisdiction_row.sourceSurface,
                target_fragment=jurisdiction_row.targetSurface,
            )
        )
        add(
            "jurisdictional_surfaces",
            _finding(
                check_id="jurisdictional_surfaces",
                finding_kind="route_or_jurisdiction_mismatch",
                dimension="route_jurisdiction_and_identifiers",
                candidate_lines=candidate_lines,
                line_ids=tuple(
                    sorted(
                        {*context_line_ids, *jurisdiction_row.sourceLineIds},
                        key=_line_number,
                    )
                ),
                target_paths=context_target_paths,
                problem=(
                    f"Jurisdiction-bound {jurisdiction_row.programId!r} must retire in favor of "
                    f"{jurisdiction_row.targetSurface!r} for "
                    + (
                        f"target {jurisdiction_row.targetPartyRole} country "
                        f"{jurisdiction_row.targetPartyCountryCode}."
                        if jurisdiction_row.rewriteBasis
                        in {
                            "target_party_country_changed",
                            "target_party_registry_country_changed",
                        }
                        else f"target route country {jurisdiction_row.targetRouteCountryCode}."
                    )
                ),
                repairs=repairs,
            ),
        )

    literal_rows = cast(
        tuple[TargetLiteralRequirement, ...], contract_rows["targetLiteralRequirements"]
    )
    for literal_row in _missing_target_literals(candidate, literal_rows):
        owner_ids = tuple(
            sorted(
                {
                    line_id
                    for inventory_row in inventory
                    if literal_row.targetPath in inventory_row.targetPaths
                    for line_id in inventory_row.lineIds
                },
                key=_line_number,
            )
        )
        add(
            "target_literals",
            _finding(
                check_id="target_literals",
                finding_kind="target_fact_mismatch",
                dimension="target_fact_fidelity",
                candidate_lines=candidate_lines,
                line_ids=owner_ids,
                target_paths=(literal_row.targetPath,),
                problem=f"Required target literal {literal_row.targetValue!r} is absent.",
            ),
        )

    for occurrence_row in cast(
        tuple[TargetValueOccurrenceRequirement, ...],
        contract_rows["targetValueOccurrenceRequirements"],
    ):
        observed_count = _target_value_occurrence_count(candidate, occurrence_row)
        if observed_count == occurrence_row.requiredOccurrences:
            continue
        occurrence_ids = tuple(
            f"L{number:05d}"
            for number, line in enumerate(candidate_lines, start=1)
            if _normalized(occurrence_row.targetValue) in _normalized(line)
        )
        owner_ids = tuple(
            sorted(
                {
                    line_id
                    for inventory_row in inventory
                    if set(occurrence_row.targetPaths) & set(inventory_row.targetPaths)
                    for line_id in inventory_row.lineIds
                },
                key=_line_number,
            )
        )
        add(
            "target_occurrences",
            _finding(
                check_id="target_occurrences",
                finding_kind="repeated_or_derived_fact_mismatch",
                dimension="repeated_and_derived_relations",
                candidate_lines=candidate_lines,
                line_ids=occurrence_ids or owner_ids,
                target_paths=occurrence_row.targetPaths,
                problem=(
                    f"Target scalar {occurrence_row.targetValue!r} occurs {observed_count} "
                    f"time(s); the compiler requires exactly "
                    f"{occurrence_row.requiredOccurrences}."
                ),
            ),
        )

    for anchored_row in cast(
        tuple[AnchoredScalarReplacementRequirement, ...],
        contract_rows["anchoredScalarReplacementRequirements"],
    ):
        if _anchored_scalar_replacements_rendered(candidate, (anchored_row,)):
            continue
        repairs = tuple(
            repair
            for line_id in anchored_row.sourceLineIds
            for repair in _safe_repair(
                candidate_lines=candidate_lines,
                line_id=line_id,
                source_fragment=anchored_row.sourceSurface,
                target_fragment=anchored_row.targetSurface,
            )
        )
        add(
            "anchored_scalars",
            _finding(
                check_id="anchored_scalars",
                finding_kind="target_fact_mismatch",
                dimension="target_fact_fidelity",
                candidate_lines=candidate_lines,
                line_ids=anchored_row.sourceLineIds,
                target_paths=anchored_row.targetPaths,
                problem=(
                    f"Compiler-owned scalar {anchored_row.targetSurface!r} is missing from an "
                    "anchored "
                    "source line or its retired source value remains."
                ),
                repairs=repairs,
            ),
        )

    for role_row in cast(
        tuple[SourceSemanticRoleHint, ...], contract_rows["sourceSemanticRoleHints"]
    ):
        number = _line_number(role_row.sourceLineId)
        rendered = (
            1 <= number <= len(candidate_lines)
            and role_row.requiredOutputSurface in candidate_lines[number - 1]
        )
        if rendered:
            continue
        repairs = _safe_repair(
            candidate_lines=candidate_lines,
            line_id=role_row.sourceLineId,
            source_fragment=role_row.sourceSurface,
            target_fragment=role_row.requiredOutputSurface,
        )
        add(
            "semantic_role_hints",
            _finding(
                check_id="semantic_role_hints",
                finding_kind="equipment_or_temperature_mismatch",
                dimension="equipment_temperature_and_capacity",
                candidate_lines=candidate_lines,
                line_ids=(role_row.sourceLineId,),
                target_paths=(),
                problem=(
                    f"Anonymous equipment role must render {role_row.requiredOutputSurface!r} "
                    "on its "
                    "compiler-owned physical line."
                ),
                repairs=repairs,
            ),
        )

    for operational_row in cast(
        tuple[OperationalFlavorRequirement, ...],
        contract_rows["operationalFlavorRequirements"],
    ):
        if _operational_flavor_requirements_rendered(candidate, (operational_row,)):
            continue
        repairs = _safe_repair(
            candidate_lines=candidate_lines,
            line_id=operational_row.sourceLineId,
            source_fragment=operational_row.sourceValueSurface,
            target_fragment=operational_row.targetValueSurface,
        )
        add(
            "operational_flavor",
            _finding(
                check_id="operational_flavor",
                finding_kind="equipment_or_temperature_mismatch",
                dimension="equipment_temperature_and_capacity",
                candidate_lines=candidate_lines,
                line_ids=(operational_row.sourceLineId,),
                target_paths=(),
                problem=(
                    f"Operational {operational_row.kind} surface must render exact target value "
                    f"{operational_row.targetValueSurface!r} in its source-owned container row."
                ),
                repairs=repairs,
            ),
        )

    cargo_rows = cast(
        tuple[CargoFlavorRewriteRequirement, ...],
        contract_rows["cargoFlavorRewriteRequirements"],
    )
    for cargo_row in cargo_rows:
        failures = _cargo_flavor_rewrite_failures(candidate, (cargo_row,))
        if not failures:
            continue
        add(
            "cargo_flavor",
            _finding(
                check_id="cargo_flavor",
                finding_kind="cargo_or_dangerous_goods_mismatch",
                dimension="cargo_packages_and_dangerous_goods",
                candidate_lines=candidate_lines,
                line_ids=cargo_row.sourceLineIds,
                target_paths=(cargo_row.targetPath,),
                problem=(
                    f"Cargo block {cargo_row.requirementId!r} fails its label-grounded rewrite "
                    f"contract: {failures!r}."
                ),
            ),
        )

    source_aux = locate_auxiliary_values(source)
    owned_auxiliary_rows = _edit_owned_auxiliary_rows(
        source=source,
        deterministic_edits=deterministic_edits,
        source_auxiliary_rows=source_aux,
    )
    owned_category_by_edit_line = {
        (id(edit), number): category for edit, number, category in owned_auxiliary_rows
    }
    failed_deterministic_edit_lines: set[tuple[int, int]] = set()
    failed_deterministic_edits: list[
        tuple[
            DeterministicInventoryEdit,
            tuple[str, ...],
            tuple[DeterministicLineRepair, ...],
            frozenset[tuple[int, str]],
        ]
    ] = []
    for edit_index, edit in enumerate(deterministic_edits):
        failed_line_ids: list[str] = []
        edit_repairs: list[DeterministicLineRepair] = []
        for line_id in edit.lineIds:
            number = _line_number(line_id)
            rendered = (
                1 <= number <= len(candidate_lines)
                and edit.targetSurface in candidate_lines[number - 1]
                and edit.sourceSurface not in candidate_lines[number - 1]
            )
            if rendered:
                continue
            failed_line_ids.append(line_id)
            failed_deterministic_edit_lines.add((edit_index, number))
            edit_repairs.extend(
                _safe_repair(
                    candidate_lines=candidate_lines,
                    line_id=line_id,
                    source_fragment=edit.sourceSurface,
                    target_fragment=edit.targetSurface,
                )
            )
        if failed_line_ids:
            failed_deterministic_edits.append(
                (
                    edit,
                    tuple(failed_line_ids),
                    tuple(edit_repairs),
                    frozenset(
                        (number, category)
                        for line_id in failed_line_ids
                        for number in (_line_number(line_id),)
                        if (category := owned_category_by_edit_line.get((id(edit), number)))
                        is not None
                    ),
                )
            )

    parents = list(range(len(failed_deterministic_edits)))

    def find_failure_group(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge_failure_groups(left: int, right: int) -> None:
        left_root = find_failure_group(left)
        right_root = find_failure_group(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, deterministic_failure in enumerate(failed_deterministic_edits):
        left_owners = deterministic_failure[3]
        if not left_owners:
            continue
        for right_index in range(left_index + 1, len(failed_deterministic_edits)):
            right_owners = failed_deterministic_edits[right_index][3]
            if left_owners & right_owners:
                merge_failure_groups(left_index, right_index)

    failure_groups: dict[
        int,
        list[
            tuple[
                DeterministicInventoryEdit,
                tuple[str, ...],
                tuple[DeterministicLineRepair, ...],
                frozenset[tuple[int, str]],
            ]
        ],
    ] = defaultdict(list)
    for index, deterministic_failure in enumerate(failed_deterministic_edits):
        failure_groups[find_failure_group(index)].append(deterministic_failure)
    for grouped_failures in failure_groups.values():
        grouped_edits = tuple(row[0] for row in grouped_failures)
        line_ids = tuple(
            sorted(
                {
                    line_id
                    for _edit, rows, _repairs, _owners in grouped_failures
                    for line_id in rows
                },
                key=_line_number,
            )
        )
        repairs_by_identity = {
            canonical_json_bytes(repair.model_dump(mode="json")): repair
            for _edit, _rows, grouped_repairs, _owners in grouped_failures
            for repair in grouped_repairs
        }
        target_surfaces = tuple(edit.targetSurface for edit in grouped_edits)
        problem = (
            f"Host-generated auxiliary replacement {target_surfaces[0]!r} is missing from one "
            "or more exact source-owned lines or the source value remains."
            if len(target_surfaces) == 1
            else (
                f"Host-generated auxiliary replacements {target_surfaces!r} are missing from "
                "their shared exact source-owned field or source values remain."
            )
        )
        add(
            "deterministic_inventory_edits",
            _finding(
                check_id="deterministic_inventory_edits",
                finding_kind="source_only_private_or_auxiliary_fact",
                dimension="source_private_and_auxiliary_replacement",
                candidate_lines=candidate_lines,
                line_ids=line_ids,
                target_paths=(),
                problem=problem,
                repairs=tuple(repairs_by_identity.values()),
            ),
        )

    for auxiliary in source_aux:
        deterministic_owned_lines = frozenset(
            number
            for number in auxiliary.line_numbers
            if any(
                (edit_index, number) in failed_deterministic_edit_lines
                and _deterministic_edit_owns_auxiliary_line(edit, auxiliary, number)
                for edit_index, edit in enumerate(deterministic_edits)
            )
        )
        surviving_numbers = _unretired_auxiliary_line_numbers(
            auxiliary=auxiliary,
            candidate_lines=candidate_lines,
            inventory=inventory,
            anchored_rows=cast(
                tuple[AnchoredScalarReplacementRequirement, ...],
                contract_rows["anchoredScalarReplacementRequirements"],
            ),
            deterministic_edits=deterministic_edits,
            ignored_line_numbers=deterministic_owned_lines,
        )
        if not surviving_numbers:
            continue
        surviving = tuple(f"L{number:05d}" for number in surviving_numbers)
        carrier_context_lines: tuple[str, ...] = ()
        if carrier_retirement is not None and any(
            all(
                source_lines[index - 1].strip()
                for index in range(
                    min(auxiliary_number, carrier_number), max(auxiliary_number, carrier_number) + 1
                )
            )
            for auxiliary_number in auxiliary.line_numbers
            for carrier_number in carrier_retirement.source_identity_line_numbers
        ):
            carrier_context_lines = tuple(f"L{number:05d}" for number in target_carrier_lines)
        add(
            "source_auxiliary_retirement",
            _finding(
                check_id="source_auxiliary_retirement",
                finding_kind="source_only_private_or_auxiliary_fact",
                dimension="source_private_and_auxiliary_replacement",
                candidate_lines=candidate_lines,
                line_ids=tuple(dict.fromkeys((*carrier_context_lines, *surviving))),
                target_paths=(),
                problem=(
                    f"Source-only auxiliary {auxiliary.category!r} value "
                    f"{auxiliary.value!r} survives anonymization."
                ),
            ),
        )

    owner_lines_by_target: dict[tuple[str, str], set[str]] = defaultdict(set)
    for inventory_row in inventory:
        for path, target in _target_semantics(inventory_row):
            if _is_high_entropy_target(path, target):
                owner_lines_by_target[(path, target)].update(inventory_row.lineIds)
    for (path, target), owner_lines in sorted(owner_lines_by_target.items()):
        occurrences = _identifier_occurrence_lines(candidate_lines, target)
        unowned = tuple(line_id for line_id in occurrences if line_id not in owner_lines)
        if occurrences and not unowned:
            continue
        add(
            "identifier_line_ownership",
            _finding(
                check_id="identifier_line_ownership",
                finding_kind="target_fact_mismatch",
                dimension="target_fact_fidelity",
                candidate_lines=candidate_lines,
                line_ids=unowned or tuple(sorted(owner_lines, key=_line_number)),
                target_paths=(path,),
                problem=(
                    f"High-entropy target {target!r} must occur only on its inventory-owned "
                    f"lines {sorted(owner_lines, key=_line_number)!r}; observed {occurrences!r}."
                ),
            ),
        )

    vessel_cargo_lines = _vessel_cargo_collision_lines(
        source_lines=source_lines,
        candidate_lines=candidate_lines,
        source_label=source_label,
        target_label=target_label,
    )
    if vessel_cargo_lines:
        add(
            "cross_domain_target_collision",
            _finding(
                check_id="cross_domain_target_collision",
                finding_kind="cargo_or_dangerous_goods_mismatch",
                dimension="cargo_packages_and_dangerous_goods",
                candidate_lines=candidate_lines,
                line_ids=vessel_cargo_lines,
                target_paths=("documentPatch.transport.vesselName",),
                problem=(
                    "The target vessel identity occupies source-proven cargo-description slots; "
                    "route and cargo semantic roles collide."
                ),
            ),
        )

    if len(source_lines) == len(candidate_lines):
        mutable_spans = _mutable_source_spans(
            source_lines=source_lines,
            contract_rows=contract_rows,
            inventory=inventory,
            deterministic_edits=deterministic_edits,
        )
        for number, spans in sorted(mutable_spans.items()):
            if (
                number in carrier_role_template_mismatches
                or spans is None
                or _skeleton_matches(source_lines[number - 1], candidate_lines[number - 1], spans)
            ):
                continue
            line_id = f"L{number:05d}"
            add(
                "protected_line_skeleton",
                _finding(
                    check_id="protected_line_skeleton",
                    finding_kind="format_or_model_artifact",
                    dimension="template_format_and_model_artifacts",
                    candidate_lines=candidate_lines,
                    line_ids=(line_id,),
                    target_paths=(),
                    problem=(
                        f"Static punctuation/label segments outside compiler-owned spans changed "
                        f"on {line_id}."
                    ),
                ),
            )

    candidate_aux = locate_auxiliary_values(candidate)
    candidate_aux_by_line = _auxiliary_rows_by_line(candidate_aux)
    allowed_caption_changes: dict[int, tuple[tuple[str, str], ...]] = defaultdict(tuple)
    for row in jurisdiction_rows:
        source_numbers = {_line_number(line_id) for line_id in row.sourceLineIds}
        authorized_numbers = set(source_numbers)
        registered_source_key = _label_key(row.sourceSurface)
        for auxiliary in source_aux:
            auxiliary_key = _label_key(auxiliary.category)
            if not (
                auxiliary_key.startswith(registered_source_key)
                or registered_source_key.startswith(auxiliary_key)
            ):
                continue
            for value_number in auxiliary.line_numbers:
                if value_number in source_numbers or any(
                    value_number > source_number
                    and not any(
                        line.strip() for line in source_lines[source_number : value_number - 1]
                    )
                    for source_number in source_numbers
                ):
                    # A standalone or trailing field heading owns its next nonblank value line.
                    # Caption-integrity is evaluated where the value parser reports the field,
                    # so carry only this structurally proven authority to that physical line.
                    authorized_numbers.add(value_number)
        for number in authorized_numbers:
            allowed_caption_changes[number] = (
                *allowed_caption_changes[number],
                *(
                    (_label_key(row.sourceSurface), _label_key(target_surface))
                    for target_surface in (
                        row.targetSurface,
                        *row.alternativeTargetSurfaces,
                    )
                ),
            )
    owned_by_edit: dict[DeterministicInventoryEdit, list[tuple[int, str, str | None]]] = (
        defaultdict(list)
    )
    for edit, number, source_category in owned_auxiliary_rows:
        line_candidate_rows = candidate_aux_by_line.get(number, ())
        exact_candidate_rows = tuple(
            row
            for row in line_candidate_rows
            if row.value.casefold() == edit.targetSurface.casefold()
        )
        matching_candidate_rows = exact_candidate_rows or tuple(
            row
            for row in line_candidate_rows
            if row.value.casefold() in edit.targetSurface.casefold()
            or edit.targetSurface.casefold() in row.value.casefold()
        )
        if not matching_candidate_rows and len(line_candidate_rows) == 1:
            # An older candidate can contain a different generated value than the current
            # deterministic compiler.  The unique field parsed on this exact owned line still
            # proves whether its static caption changed; target-value enforcement remains a
            # separate finding.
            matching_candidate_rows = line_candidate_rows
        candidate_category = (
            matching_candidate_rows[0].category if len(matching_candidate_rows) == 1 else None
        )
        owned_by_edit[edit].append((number, source_category, candidate_category))
        if candidate_category is None or candidate_category == source_category:
            continue
        source_key = _label_key(source_category)
        candidate_key = _label_key(candidate_category) if candidate_category is not None else ""
        source_caption_is_leading = _label_key(source_lines[number - 1]).startswith(source_key)
        candidate_line_key = _label_key(candidate_lines[number - 1])
        explicitly_changed = any(
            (source_key.startswith(old) or old.startswith(source_key))
            and (
                candidate_key.startswith(new)
                or new.startswith(candidate_key)
                or candidate_line_key.startswith(new)
            )
            and (not source_caption_is_leading or candidate_line_key.startswith(new))
            for old, new in allowed_caption_changes.get(number, ())
        )
        if explicitly_changed:
            continue
        if any(
            source_key.startswith(registered_source) or registered_source.startswith(source_key)
            for registered_source in failed_jurisdiction_caption_sources.get(number, ())
        ):
            # The stricter explicit jurisdiction requirement already reports this same caption
            # failure. Keep one actionable finding while leaving unrelated fields on the same
            # physical line independently auditable.
            continue
        line_id = f"L{number:05d}"
        add(
            "auxiliary_caption_integrity",
            _finding(
                check_id="auxiliary_caption_integrity",
                finding_kind="format_or_model_artifact",
                dimension="template_format_and_model_artifacts",
                candidate_lines=candidate_lines,
                line_ids=(line_id,),
                target_paths=(),
                problem=(
                    f"Deterministically edited auxiliary value on {line_id} changed its "
                    f"source field caption from {source_category!r} to "
                    f"{candidate_category!r} without an explicit jurisdictional rewrite."
                ),
            ),
        )
    for edit, rows in owned_by_edit.items():
        if len(rows) < 2:
            continue
        if any(candidate_category is None for _number, _source, candidate_category in rows):
            # Exact deterministic-value enforcement already owns missing or unrecognizable
            # outputs.  A field-kind comparison has no evidence until every expected value can
            # actually be located, and emitting one here would duplicate that primary defect.
            continue
        source_categories = {source_category for _number, source_category, _current in rows}
        candidate_categories = {
            candidate_category
            for _number, _source_category, candidate_category in rows
            if candidate_category is not None
        }
        if len(source_categories) != 1 or (
            len(candidate_categories) == 1
            and all(candidate_category is not None for _n, _s, candidate_category in rows)
        ):
            continue
        line_ids = tuple(f"L{number:05d}" for number, _source, _candidate in rows)
        add(
            "auxiliary_caption_integrity",
            _finding(
                check_id="auxiliary_caption_integrity",
                finding_kind="route_or_jurisdiction_mismatch",
                dimension="route_jurisdiction_and_identifiers",
                candidate_lines=candidate_lines,
                line_ids=line_ids,
                target_paths=(),
                problem=(
                    f"One repeated deterministic identifier {edit.targetSurface!r} is assigned "
                    f"inconsistent field kinds across its source-owned copies: "
                    f"{tuple(candidate for _number, _source, candidate in rows)!r}."
                ),
            ),
        )
    candidate_values_by_category_line: dict[tuple[str, int], set[str]] = defaultdict(set)
    for candidate_aux_row in candidate_aux:
        for number in candidate_aux_row.line_numbers:
            candidate_values_by_category_line[(candidate_aux_row.category, number)].add(
                candidate_aux_row.value
            )
    for source_aux_row in source_aux:
        if len(source_aux_row.line_numbers) < 2:
            continue
        if all(
            any(
                _deterministic_edit_owns_auxiliary_line(edit, source_aux_row, number)
                for edit in deterministic_edits
            )
            for number in source_aux_row.line_numbers
        ):
            # The deterministic edit is a stronger one-to-one provenance contract for every
            # copy.  Its grouped finding covers both missing values and disagreement without
            # double-counting the same physical defect here.
            continue
        observed_values: list[str] = []
        complete = True
        for number in source_aux_row.line_numbers:
            values = candidate_values_by_category_line.get((source_aux_row.category, number), set())
            if len(values) != 1:
                complete = False
                break
            observed_values.append(next(iter(values)))
        if not complete or len(set(observed_values)) <= 1:
            continue
        line_ids = tuple(f"L{number:05d}" for number in source_aux_row.line_numbers)
        add(
            "repeated_auxiliary_consistency",
            _finding(
                check_id="repeated_auxiliary_consistency",
                finding_kind="repeated_or_derived_fact_mismatch",
                dimension="repeated_and_derived_relations",
                candidate_lines=candidate_lines,
                line_ids=line_ids,
                target_paths=(),
                problem=(
                    f"Repeated source {source_aux_row.category!r} field was one value but "
                    f"candidate copies disagree: {tuple(observed_values)!r}."
                ),
            ),
        )

    for office_relation in _compile_office_reference_relations(
        source_lines=source_lines,
        inventory=inventory,
        references=references,
    ):
        country_codes = frozenset(
            code
            for number in office_relation.country_line_numbers
            if number <= len(candidate_lines)
            for code in references.country_codes_in_text(candidate_lines[number - 1])
        )
        if len(country_codes) > 1:
            line_ids = tuple(
                f"L{number:05d}"
                for number in (
                    *office_relation.identity_line_numbers,
                    *office_relation.country_line_numbers,
                )
                if number <= len(candidate_lines)
            )
            add(
                "office_country_identity",
                _finding(
                    check_id="office_country_identity",
                    finding_kind="route_or_jurisdiction_mismatch",
                    dimension="route_jurisdiction_and_identifiers",
                    candidate_lines=candidate_lines,
                    line_ids=line_ids,
                    target_paths=office_relation.target_paths,
                    problem=(
                        f"Source-proven office relation {office_relation.relation_id!r} renders "
                        f"multiple ISO country identities: {sorted(country_codes)!r}."
                    ),
                ),
            )
        country_constraint = country_codes if len(country_codes) == 1 else None
        locality_sets = tuple(
            references.locality_ids_in_text(
                candidate_lines[number - 1], country_codes=country_constraint
            )
            for number in office_relation.locality_line_numbers
            if number <= len(candidate_lines)
        )
        if (
            locality_sets
            and all(locality_sets)
            and not set.intersection(*(set(values) for values in locality_sets))
        ):
            line_ids = tuple(
                f"L{number:05d}"
                for number in sorted(
                    {
                        *office_relation.identity_line_numbers,
                        *office_relation.locality_line_numbers,
                        *office_relation.country_line_numbers,
                    }
                )
                if number <= len(candidate_lines)
            )
            add(
                "office_locality_identity",
                _finding(
                    check_id="office_locality_identity",
                    finding_kind="route_or_jurisdiction_mismatch",
                    dimension="route_jurisdiction_and_identifiers",
                    candidate_lines=candidate_lines,
                    line_ids=line_ids,
                    target_paths=office_relation.target_paths,
                    problem=(
                        f"Repeated source-proven office relation "
                        f"{office_relation.relation_id!r} resolves to inconsistent GeoNames "
                        "locality identities across its owned physical lines."
                    ),
                ),
            )

        if len(country_codes) == 1:
            expected_country = next(iter(country_codes))
            mismatched_phone_lines: list[int] = []
            observed_regions: set[str] = set()
            for number in office_relation.contact_line_numbers:
                if number > len(candidate_lines):
                    continue
                phones = references.international_phones(candidate_lines[number - 1])
                if len(phones) != 1 or expected_country in phones[0].region_codes:
                    continue
                mismatched_phone_lines.append(number)
                observed_regions.update(phones[0].region_codes)
            if mismatched_phone_lines:
                line_ids = tuple(
                    f"L{number:05d}"
                    for number in sorted(
                        {
                            *office_relation.identity_line_numbers,
                            *office_relation.country_line_numbers,
                            *mismatched_phone_lines,
                        }
                    )
                )
                add(
                    "office_phone_country",
                    _finding(
                        check_id="office_phone_country",
                        finding_kind="route_or_jurisdiction_mismatch",
                        dimension="route_jurisdiction_and_identifiers",
                        candidate_lines=candidate_lines,
                        line_ids=line_ids,
                        target_paths=office_relation.target_paths,
                        problem=(
                            f"Explicit international telephone calling-code regions "
                            f"{sorted(observed_regions)!r} conflict with source-owned office "
                            f"country {expected_country!r}."
                        ),
                    ),
                )

    for freight_relation in _compile_freight_schedule_relations(source_lines):
        surviving_line_ids = tuple(
            f"L{number:05d}"
            for number, line in enumerate(candidate_lines, start=1)
            if any(
                _normalized(surface) in _normalized(line)
                for surface in freight_relation.private_surfaces
            )
        )
        if surviving_line_ids:
            add(
                "source_private_freight_schedule",
                _finding(
                    check_id="source_private_freight_schedule",
                    finding_kind="source_only_private_or_auxiliary_fact",
                    dimension="source_private_and_auxiliary_replacement",
                    candidate_lines=candidate_lines,
                    line_ids=surviving_line_ids,
                    target_paths=(),
                    problem=(
                        f"Source-proven freight schedule {freight_relation.relation_id!r} "
                        "retains one or more exact source monetary surfaces outside the complete "
                        "synthetic target."
                    ),
                ),
            )
        if not _freight_schedule_holds(candidate_lines, freight_relation):
            line_ids = tuple(
                f"L{number:05d}"
                for number in sorted(
                    {
                        freight_relation.total.line_number,
                        *(slot.line_number for slot in freight_relation.components),
                    }
                )
                if number <= len(candidate_lines)
            )
            add(
                "freight_schedule_arithmetic",
                _finding(
                    check_id="freight_schedule_arithmetic",
                    finding_kind="repeated_or_derived_fact_mismatch",
                    dimension="repeated_and_derived_relations",
                    candidate_lines=candidate_lines,
                    line_ids=line_ids,
                    target_paths=(),
                    problem=(
                        f"Source-proven additive freight schedule "
                        f"{freight_relation.relation_id!r} is no longer internally additive."
                    ),
                ),
            )

    for relation in _compile_additive_relations(source, source_label):
        if _relation_holds(candidate_lines, relation):
            continue
        line_ids = tuple(
            f"L{number:05d}"
            for number in _additive_relation_evidence_line_numbers(
                candidate_lines=candidate_lines,
                relation=relation,
                target_label=target_label,
            )
        )
        add(
            "source_proven_additive_relations",
            _finding(
                check_id="source_proven_additive_relations",
                finding_kind="repeated_or_derived_fact_mismatch",
                dimension="repeated_and_derived_relations",
                candidate_lines=candidate_lines,
                line_ids=line_ids,
                target_paths=(),
                problem=(
                    f"Source-proven additive relation {relation.relation_id!r} "
                    f"({relation.semantic_kind}) does not hold after rewriting."
                ),
            ),
        )

    for serial_relation in _compile_serial_cardinality_relations(source_label, target_label):
        cardinalities = tuple(
            _serial_cardinality(surface) for surface in serial_relation.target_range_surfaces
        )
        if (
            all(value is not None for value in cardinalities)
            and sum(cast(tuple[int, ...], cardinalities)) == serial_relation.target_quantity
        ):
            continue
        target_tokens = tuple(
            token
            for surface in serial_relation.target_range_surfaces
            for token in re.findall(r"[A-Z]+[0-9]+", surface.upper())
        )
        line_ids = tuple(
            f"L{number:05d}"
            for number, line in enumerate(candidate_lines, start=1)
            if any(token in _identifier_key(line) for token in target_tokens)
            or str(serial_relation.target_quantity) in re.sub(r"[^0-9]", "", line)
        )
        add(
            "serial_cardinality_relations",
            _finding(
                check_id="serial_cardinality_relations",
                finding_kind="repeated_or_derived_fact_mismatch",
                dimension="repeated_and_derived_relations",
                candidate_lines=candidate_lines,
                line_ids=line_ids,
                target_paths=(
                    f"documentPatch.cargoGroups[{serial_relation.group_index}].marksAndNumbers",
                    f"documentPatch.cargoPackages[{serial_relation.group_index}].quantity",
                ),
                problem=(
                    f"Source marks prove that inclusive serial-range cardinality equals package "
                    f"quantity ({serial_relation.source_quantity}); target ranges total "
                    f"{sum(value or 0 for value in cardinalities)} but target quantity is "
                    f"{serial_relation.target_quantity}."
                ),
            ),
        )

    target_equipment = _target_container_equipment(target_label)
    candidate_counted_assertions = _counted_equipment_assertions(
        candidate_lines, equipment_cache=_equipment_cache
    )
    aggregate_equipment_prefill_lines = _aggregate_equipment_prefill_line_ids(
        cast(
            tuple[AppliedDeterministicPrefill, ...],
            contract_rows["deterministicPrefills"],
        )
    )
    if target_equipment is not None:
        source_equipment_lines = _source_equipment_line_numbers(
            source_lines, equipment_cache=_equipment_cache
        )
        if not target_equipment:
            surviving_equipment_lines: list[int] = []
            for line_number in sorted(source_equipment_lines):
                current = candidate_lines[line_number - 1]
                if (
                    _resolve_equipment_pair(current, _equipment_cache) is None
                    and _FCL_COUNT.search(current) is None
                ):
                    continue
                surviving_equipment_lines.append(line_number)
            if surviving_equipment_lines:
                line_ids = tuple(f"L{line_number:05d}" for line_number in surviving_equipment_lines)
                add(
                    "equipment_topology",
                    _finding(
                        check_id="equipment_topology",
                        finding_kind="equipment_or_temperature_mismatch",
                        dimension="equipment_temperature_and_capacity",
                        candidate_lines=candidate_lines,
                        line_ids=line_ids,
                        target_paths=("documentPatch.containers",),
                        problem=(
                            f"Shipment-specific equipment assertions remain on {line_ids!r}, "
                            "but the complete synthetic target has no container inventory."
                        ),
                    ),
                )
        else:
            target_counts = Counter(
                (size, category) for _number, size, category in target_equipment
            )
            for assertion in candidate_counted_assertions:
                pair = (assertion.size_category, assertion.type_category)
                if (
                    assertion.line_number not in source_equipment_lines
                    or assertion.count <= target_counts[pair]
                ):
                    continue
                add(
                    "equipment_topology",
                    _finding(
                        check_id="equipment_topology",
                        finding_kind="equipment_or_temperature_mismatch",
                        dimension="equipment_temperature_and_capacity",
                        candidate_lines=candidate_lines,
                        line_ids=(f"L{assertion.line_number:05d}",),
                        target_paths=("documentPatch.containers",),
                        problem=(
                            f"Equipment assertion {assertion.surface!r} claims {assertion.count} "
                            f"unit(s), but target inventory contains only {target_counts[pair]} "
                            "of that exact size/type family."
                        ),
                    ),
                )

            if len(target_counts) > 1:
                number_keys = frozenset(
                    _identifier_key(number) for number, _size, _type in target_equipment
                )
                aggregate_lines = tuple(
                    dict.fromkeys(
                        (
                            *(
                                f"L{row.line_number:05d}"
                                for row in candidate_counted_assertions
                                if row.line_number in source_equipment_lines
                            ),
                            *aggregate_equipment_prefill_lines,
                        )
                    )
                )
                missing_assignments: list[
                    tuple[int, str, str, str, tuple[str, ...], tuple[tuple[str, str, int], ...]]
                ] = []
                for container_index, (target_number, size, category) in enumerate(target_equipment):
                    number_key = _identifier_key(target_number)
                    container_occurrence_indexes = tuple(
                        candidate_index
                        for candidate_index, line in enumerate(candidate_lines)
                        if number_key in _identifier_key(line)
                    )
                    local = tuple(
                        row
                        for occurrence in container_occurrence_indexes
                        for row in _local_equipment_pairs(
                            candidate_lines=candidate_lines,
                            container_line=occurrence,
                            target_number_keys=number_keys - {number_key},
                            counted_assertions=candidate_counted_assertions,
                            equipment_cache=_equipment_cache,
                        )
                    )
                    expected = tuple(row for row in local if (row[0], row[1]) == (size, category))
                    unexpected = tuple(row for row in local if (row[0], row[1]) != (size, category))
                    if len(expected) == 1 and not unexpected:
                        continue
                    evidence_ids = tuple(
                        sorted(
                            {
                                *(
                                    f"L{candidate_index + 1:05d}"
                                    for candidate_index in container_occurrence_indexes
                                ),
                                *(f"L{row[2]:05d}" for row in local),
                                *aggregate_lines,
                            },
                            key=_line_number,
                        )
                    )
                    if not expected:
                        missing_assignments.append(
                            (
                                container_index,
                                target_number,
                                size,
                                category,
                                evidence_ids,
                                tuple((row[0], row[1], row[2]) for row in local),
                            )
                        )
                        continue
                    add(
                        "equipment_assignment",
                        _finding(
                            check_id="equipment_assignment",
                            finding_kind="equipment_or_temperature_mismatch",
                            dimension="equipment_temperature_and_capacity",
                            candidate_lines=candidate_lines,
                            line_ids=evidence_ids,
                            target_paths=(
                                f"documentPatch.containers[{container_index}].sizeCategory",
                                f"documentPatch.containers[{container_index}].typeCategory",
                            ),
                            problem=(
                                f"Heterogeneous target container {target_number!r} requires "
                                "exactly "
                                "one "
                                f"local {size}/{category} realization; observed "
                                f"{tuple((row[0], row[1], row[2]) for row in local)!r}."
                            ),
                        ),
                    )
                if missing_assignments:
                    missing_line_ids = tuple(
                        sorted(
                            {
                                line_id
                                for _index, _number, _size, _category, evidence, _observed in (
                                    missing_assignments
                                )
                                for line_id in evidence
                            },
                            key=_line_number,
                        )
                    )
                    missing_paths = tuple(
                        path
                        for index, _number, _size, _category, _evidence, _observed in (
                            missing_assignments
                        )
                        for path in (
                            f"documentPatch.containers[{index}].sizeCategory",
                            f"documentPatch.containers[{index}].typeCategory",
                        )
                    )
                    descriptions = tuple(
                        (number, size, category, observed)
                        for _index, number, size, category, _evidence, observed in (
                            missing_assignments
                        )
                    )
                    add(
                        "equipment_assignment",
                        _finding(
                            check_id="equipment_assignment",
                            finding_kind="target_fact_mismatch",
                            dimension="target_fact_fidelity",
                            candidate_lines=candidate_lines,
                            line_ids=missing_line_ids,
                            target_paths=missing_paths,
                            problem=(
                                "Heterogeneous target equipment assignments are not locally "
                                f"recoverable for {descriptions!r}; aggregate fleet counts do "
                                "not identify the target size/type of each container."
                            ),
                        ),
                    )

    candidate_assertions_by_line: dict[int, list[_EquipmentAssertion]] = defaultdict(list)
    for candidate_assertion in candidate_counted_assertions:
        candidate_assertions_by_line[candidate_assertion.line_number].append(candidate_assertion)
    for capacity_relation in _compile_physical_capacity_relations(
        source_lines,
        capacity_limits,
        equipment_cache=_equipment_cache,
    ):
        assertions = candidate_assertions_by_line.get(capacity_relation.equipment_line_number, [])
        if len(assertions) != 1 or capacity_relation.measurement_line_number > len(candidate_lines):
            continue
        measurement = _physical_measurement(
            candidate_lines[capacity_relation.measurement_line_number - 1]
        )
        if measurement is None or measurement[0] != capacity_relation.measurement_kind:
            continue
        capacity = _assertion_capacity(
            assertions[0],
            limits=capacity_limits,
            kind=capacity_relation.measurement_kind,
        )
        if capacity is None or measurement[1] - measurement[2] <= capacity:
            continue
        add(
            "physical_capacity",
            _finding(
                check_id="physical_capacity",
                finding_kind="equipment_or_temperature_mismatch",
                dimension="equipment_temperature_and_capacity",
                candidate_lines=candidate_lines,
                line_ids=(
                    f"L{capacity_relation.equipment_line_number:05d}",
                    f"L{capacity_relation.measurement_line_number:05d}",
                ),
                target_paths=(),
                problem=(
                    f"Source-proven {capacity_relation.measurement_kind} relationship "
                    f"{capacity_relation.relation_id!r} renders {measurement[1]} above the "
                    f"receipt-bound equipment ceiling {capacity} under "
                    f"{capacity_limits.policy!r}."
                ),
            ),
        )

    target_patch = target_label.get("documentPatch")
    if not isinstance(target_patch, Mapping):
        raise ValueError("certification target label lacks documentPatch")
    negotiability = target_patch.get("negotiability")
    if negotiability == "negotiable":
        mismatched_order_slots: list[int] = []
        for line_number, source_line in enumerate(source_lines, start=1):
            if _POSITIVE_TO_ORDER.fullmatch(source_line) is None:
                continue
            if (
                line_number <= len(candidate_lines)
                and _POSITIVE_TO_ORDER.fullmatch(candidate_lines[line_number - 1]) is not None
            ):
                continue
            mismatched_order_slots.append(line_number)
        if mismatched_order_slots:
            legal_context_lines = {
                context_number
                for line_number in mismatched_order_slots
                for context_number in range(
                    max(1, line_number - 8), min(len(candidate_lines), line_number + 8) + 1
                )
                if _ORDER_LEGAL_CONTEXT.search(candidate_lines[context_number - 1]) is not None
            }
            line_ids = tuple(
                f"L{line_number:05d}"
                for line_number in sorted({*mismatched_order_slots, *legal_context_lines})
            )
            add(
                "negotiability",
                _finding(
                    check_id="negotiability",
                    finding_kind="party_or_legal_identity_mismatch",
                    dimension="party_legal_and_negotiability",
                    candidate_lines=candidate_lines,
                    line_ids=line_ids,
                    target_paths=("documentPatch.negotiability",),
                    problem=(
                        f"Negotiable source slots {tuple(mismatched_order_slots)!r} were positive "
                        "TO ORDER realizations; the candidate replaced them with straight named "
                        "consignee surfaces."
                    ),
                ),
            )
    elif (
        negotiability == "non_negotiable"
        and any(_SEA_WAYBILL.fullmatch(line) is not None for line in candidate_lines)
        and any(_NON_NEGOTIABLE.fullmatch(line) is not None for line in candidate_lines)
    ):
        for index, line in enumerate(candidate_lines):
            if _CONSIGNED_TO_ORDER_HEADING.fullmatch(line) is None:
                continue
            next_index = index + 1
            populated_index: int | None = (
                next_index
                if next_index < len(candidate_lines)
                and candidate_lines[next_index].strip()
                and _PAGE_MARKER.fullmatch(candidate_lines[next_index]) is None
                else None
            )
            if populated_index is None:
                continue
            add(
                "negotiability",
                _finding(
                    check_id="negotiability",
                    finding_kind="party_or_legal_identity_mismatch",
                    dimension="party_legal_and_negotiability",
                    candidate_lines=candidate_lines,
                    line_ids=tuple(
                        f"L{line_number:05d}"
                        for line_number in sorted(
                            {
                                index + 1,
                                populated_index + 1,
                                *(
                                    line_number
                                    for line_number, candidate_line in enumerate(
                                        candidate_lines, start=1
                                    )
                                    if _SEA_WAYBILL.fullmatch(candidate_line) is not None
                                    or _NON_NEGOTIABLE.fullmatch(candidate_line) is not None
                                ),
                            }
                        )
                    ),
                    target_paths=("documentPatch.negotiability",),
                    problem=(
                        "Non-negotiable sea-waybill populates a shipment-specific "
                        "'Consigned to order of' block."
                    ),
                ),
            )

    original_fields = _original_count_fields(candidate_lines)
    signed_originals = _signed_original_counts(candidate_lines)
    field_counts = {count for _line_number_value, count in original_fields}
    signed_counts = {count for _line_number_value, count in signed_originals}
    if (
        original_fields
        and signed_originals
        and (
            len(field_counts) != 1
            or len(signed_counts) != 1
            or next(iter(field_counts)) != next(iter(signed_counts))
        )
    ):
        original_line_ids = tuple(
            f"L{line_number:05d}" for line_number, _count in (*original_fields, *signed_originals)
        )
        add(
            "original_bill_count",
            _finding(
                check_id="original_bill_count",
                finding_kind="party_or_legal_identity_mismatch",
                dimension="party_legal_and_negotiability",
                candidate_lines=candidate_lines,
                line_ids=original_line_ids,
                target_paths=("documentPatch.negotiability",),
                problem=(
                    f"Shipment-specific original-bill field counts {sorted(field_counts)!r} "
                    f"conflict with signed-original clause counts {sorted(signed_counts)!r}."
                ),
            ),
        )

    ordered_check_ids = (
        "line_topology",
        "template_structure",
        "source_status",
        "required_surfaces",
        "jurisdictional_surfaces",
        "target_literals",
        "target_occurrences",
        "anchored_scalars",
        "semantic_role_hints",
        "operational_flavor",
        "cargo_flavor",
        "deterministic_inventory_edits",
        "source_auxiliary_retirement",
        "source_carrier_identity_retirement",
        "source_carrier_reference_retirement",
        "source_carrier_role_template",
        "source_measurement_retirement",
        "source_party_address_retirement",
        "foreign_exporter_identity",
        "container_package_scope",
        "identifier_line_ownership",
        "cross_domain_target_collision",
        "protected_line_skeleton",
        "auxiliary_caption_integrity",
        "repeated_auxiliary_consistency",
        "office_country_identity",
        "office_locality_identity",
        "office_phone_country",
        "source_private_freight_schedule",
        "freight_schedule_arithmetic",
        "source_proven_additive_relations",
        "serial_cardinality_relations",
        "equipment_topology",
        "equipment_assignment",
        "physical_capacity",
        "package_cargo_semantics",
        "negotiability",
        "original_bill_count",
    )
    findings: list[DeterministicCertificationFinding] = []
    checks: list[InvariantCheck] = []
    seen: set[str] = set()
    for check_id in ordered_check_ids:
        check_rows = by_check.get(check_id, [])
        unique_rows: list[DeterministicCertificationFinding] = []
        for finding_row in check_rows:
            if finding_row.invariantId in seen:
                continue
            seen.add(finding_row.invariantId)
            unique_rows.append(finding_row)
        findings.extend(unique_rows)
        checks.append(
            InvariantCheck(
                checkId=check_id,
                findings=len(unique_rows),
                passed=not unique_rows,
            )
        )
    return CertificationInvariantAudit(
        schemaVersion=1,
        envelopeSha256=sha256_bytes(canonical_json_bytes(envelope.model_dump(mode="json"))),
        passed=not findings,
        checks=tuple(checks),
        findings=tuple(findings),
    )


def compile_and_audit_certification_invariants(
    *,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    inventory: Sequence[InventoryCandidate],
    deterministic_edits: Sequence[DeterministicInventoryEdit],
    references: CertificationReferenceIndex,
    capacity_limits: TransportCapacityLimits,
) -> tuple[CertificationInvariantEnvelope, CertificationInvariantAudit]:
    """Compile and audit with one document-scoped parse cache.

    The envelope is still recomputed and compared inside the audit. The cache only memoizes pure
    equipment-surface parsing for the duration of this call, so raw OCR text is neither shared
    across documents nor retained globally.
    """

    equipment_cache = _EquipmentPairCache(by_surface={})
    envelope = compile_certification_invariant_envelope(
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        references=references,
        capacity_limits=capacity_limits,
        _equipment_cache=equipment_cache,
    )
    audit = audit_certification_invariants(
        envelope=envelope,
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        references=references,
        capacity_limits=capacity_limits,
        _equipment_cache=equipment_cache,
    )
    return envelope, audit
