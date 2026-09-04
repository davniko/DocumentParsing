"""Reviewed source-equipment semantics and configurable synthetic sampling.

The source corpus uses carrier spellings (``40HQ``, ``40RH``), prose
(``40' HIGH CUBE REEFER``), and exact codes.  Those surfaces are used only to
estimate a starting distribution.  Generated labels contain the separate,
readable size and type categories from relation-v5; printed surfaces are left
to the later linguistic renderer.

The classifier is deliberately narrow.  It recognizes only reviewed syntax
families present in the pinned B/L corpus and reports every unresolved surface;
it never silently treats an unknown container as general purpose.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter

from document_ocr.label_schemas.bill_of_lading_v5 import (
    TEMPERATURE_CAPABLE_CONTAINER_TYPES,
    ContainerSizeCategory,
    ContainerTypeCategory,
    semantic_container_code,
)
from document_ocr.synthesis.generators import DeterministicStream

EquipmentResolution = Literal["reviewed_source_grammar", "unresolved_source_surface"]
_SIZE_CATEGORY_ADAPTER: TypeAdapter[ContainerSizeCategory] = TypeAdapter(ContainerSizeCategory)
_TYPE_CATEGORY_ADAPTER: TypeAdapter[ContainerTypeCategory] = TypeAdapter(ContainerTypeCategory)
_PUNCTUATION_TRANSLATION: dict[int, str | int | None] = {
    0x2019: "'",
    0x0060: "'",
}


@dataclass(frozen=True, slots=True)
class SourceEquipmentObservation:
    document_id: str
    row_id: str
    printed_surface: str | None
    temperature_present: bool


@dataclass(frozen=True, slots=True)
class ReviewedEquipmentSurface:
    printed_surface: str | None
    normalized_surface: str | None
    resolution: EquipmentResolution
    size_category: ContainerSizeCategory | None
    type_category: ContainerTypeCategory | None
    thermal_operation: Literal["active", "non_operating", "not_indicated"]
    review_rule: str


@dataclass(frozen=True, slots=True)
class EquipmentJointSupport:
    size_category: ContainerSizeCategory
    type_category: ContainerTypeCategory
    active_temperature: bool
    occurrences: int
    document_count: int


@dataclass(frozen=True, slots=True)
class EquipmentTypeSupport:
    type_category: ContainerTypeCategory
    active_temperature: bool
    occurrences: int
    document_count: int


@dataclass(frozen=True, slots=True)
class EquipmentSemanticAudit:
    input_rows: int
    type_resolved_rows: int
    resolved_rows: int
    unresolved_rows: int
    temperature_rows: int
    type_resolved_temperature_rows: int
    resolved_temperature_rows: int
    non_operating_reefer_rows: int
    type_support_rows: int
    joint_support_rows: int


@dataclass(frozen=True, slots=True)
class EquipmentSemanticSupport:
    type_rows: tuple[EquipmentTypeSupport, ...]
    rows: tuple[EquipmentJointSupport, ...]
    unresolved_surfaces: tuple[tuple[str, int], ...]
    audit: EquipmentSemanticAudit


@dataclass(frozen=True, slots=True)
class GeneratedEquipmentSemantic:
    size_category: ContainerSizeCategory
    type_category: ContainerTypeCategory
    application_code: str
    active_temperature: bool
    sampling_method: Literal[
        "source_type_marginal_size_conditional",
        "configured_joint_override",
    ]


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).upper().translate(_PUNCTUATION_TRANSLATION)
    normalized = re.sub(r"(?<=\d)\s*[Xx]\s*(?=\d)", "X", normalized)
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def review_source_equipment_surface(
    printed_surface: str | None,
    *,
    temperature_present: bool,
) -> ReviewedEquipmentSurface:
    """Resolve one observed source surface using the reviewed corpus grammar."""

    if printed_surface is None:
        return ReviewedEquipmentSurface(
            printed_surface=None,
            normalized_surface=None,
            resolution="unresolved_source_surface",
            size_category=None,
            type_category=None,
            thermal_operation="not_indicated",
            review_rule="missing_surface",
        )
    normalized = _normalize(printed_surface)
    semantic = re.sub(r"^[1-9][0-9]*X(?=(?:20|40|45))", "", normalized)
    tokens = frozenset(semantic.split())

    length: Literal[20, 40, 45] | None = None
    if re.search(r"(?:^| )45", semantic):
        length = 45
    elif re.search(r"(?:^| )40", semantic) or semantic in {
        "D40H",
        "DC 4H CY FO",
        "HC40",
        "RH40",
        "HC 40",
    }:
        length = 40
    elif (
        re.search(r"(?:^| )20", semantic)
        or semantic
        in {
            "D20",
            "DC20",
            "DV20",
            "GP20",
        }
        or re.search(r"(?:^| )22(?: |G[0O]|$)", semantic)
    ):
        length = 20

    non_operating = "NOR" in tokens or "NOR" in semantic
    carrier_thermal_code = next(
        (
            value
            for value in ("HR", "RA", "RE", "RF", "RH", "RK", "RO", "RQ")
            if value in tokens
            or re.search(rf"(?:20|40){value}(?:\s|$)", semantic)
            or semantic == f"{value}40"
        ),
        None,
    )
    explicit_thermal = bool(
        re.search(r"\b(?:REEF|REEFER|REFRIGERATED|RF)\b", semantic)
        or carrier_thermal_code is not None
    )
    if explicit_thermal or temperature_present or non_operating:
        # In the observed carrier grammar, 40HR means a 40-foot high-cube
        # reefer.  It is not BIC type group HR (removable thermal equipment).
        type_category: ContainerTypeCategory = "REFRIGERATED"
        type_rule = "reviewed_reefer_surface_or_setpoint"
    elif "OPEN" in tokens or "OT" in tokens or re.search(r"(?:20|40)OT$", semantic):
        type_category = "OPEN_TOP"
        type_rule = "reviewed_open_top_surface"
    elif "FLAT" in tokens:
        type_category = "PLATFORM_COLLAPSIBLE" if "COLLAPSIBLE" in tokens else "PLATFORM"
        type_rule = "reviewed_platform_surface"
    elif "MAFI" in tokens:
        type_category = "PLATFORM_NAMED_CARGO"
        type_rule = "reviewed_mafi_platform_surface"
    elif "TANK" in tokens or re.search(r"(?:20|40)TANK$", semantic):
        type_category = "PRESSURIZED_TANK"
        type_rule = "reviewed_tank_surface"
    elif length is not None and (
        tokens
        & {
            "BO",
            "BOX",
            "BX",
            "CONT",
            "CONTAINER",
            "CONTAINERS",
            "DC",
            "DR",
            "DRY",
            "DV",
            "EC",
            "EQ",
            "FCL",
            "FT",
            "GE",
            "GO",
            "GP",
            "HC",
            "HCPW",
            "HICU",
            "HQ",
            "SD86",
            "SD96",
            "SH",
            "ST",
            "STANDARD",
            "VAN",
        }
        or {"HIGH", "CUBE"} <= tokens
        or "HIGHCUBE" in tokens
        or {"GENERAL", "PURPOSE"} <= tokens
        or re.search(
            r"(?:20|40|45)(?:BO|BX|DC|DR|DV|EC|EQ|G[01O]|GP|H|HC|HQ|SD86|SD96|ST)$",
            semantic,
        )
        or re.search(r"^(?:20|40|45)(?:DC|DR|DV|GP|HC|HQ|ST)(?:\s|$)", semantic)
        or semantic in {"D20", "D40H", "DC20", "DV20", "GP20", "HC40", "HC 40"}
    ):
        type_category = "GENERAL_PURPOSE"
        type_rule = "reviewed_general_purpose_surface"
    else:
        return ReviewedEquipmentSurface(
            printed_surface=printed_surface,
            normalized_surface=normalized,
            resolution="unresolved_source_surface",
            size_category=None,
            type_category=None,
            thermal_operation="not_indicated",
            review_rule="surface_has_no_reviewed_type_semantics",
        )

    if length == 45:
        size_category: ContainerSizeCategory = "FORTY_FIVE_FOOT_HIGH_CUBE"
        size_rule = "explicit_45_foot"
    elif length == 20:
        high_cube = bool(({"HIGH", "CUBE"} <= tokens) or ({"HI", "CUBE"} <= tokens)) or bool(
            tokens & {"HC", "HQ", "HICU", "HCPW", "SD96"}
        )
        size_category = "TWENTY_FOOT_HIGH_CUBE" if high_cube else "TWENTY_FOOT_STANDARD_HEIGHT"
        size_rule = "explicit_20_foot_with_height_marker"
    elif length == 40:
        if carrier_thermal_code in {"RA", "RK", "RO", "RQ"} and not (
            "HIGH" in tokens
            or "CUBE" in tokens
            or tokens & {"HC", "HQ", "HICU", "HCPW", "SD96"}
            or re.search(r"(?:^|\s)9\s+6(?:\s|$)", semantic)
        ):
            return ReviewedEquipmentSurface(
                printed_surface=printed_surface,
                normalized_surface=normalized,
                resolution="unresolved_source_surface",
                size_category=None,
                type_category=type_category,
                thermal_operation="active" if temperature_present else "not_indicated",
                review_rule="reviewed_reefer_type_with_unresolved_carrier_height_code",
            )
        explicit_nine_six = bool(re.search(r"(?:^|\s)9\s+6(?:\s|$)", semantic))
        carrier_high_reefer = carrier_thermal_code in {"HR", "RH"}
        high_cube = bool(
            ({"HIGH", "CUBE"} <= tokens)
            or ({"HI", "CUBE"} <= tokens)
            or tokens & {"HC", "HQ", "HICU", "HCPW", "SD96"}
            or "HIGHCUBE" in tokens
            or re.search(r"(?:40|45)(?:H|HC|HQ|SD96)(?:\s|$)", semantic)
            or semantic in {"D40H", "HC40", "HC 40"}
            or explicit_nine_six
            or carrier_high_reefer
            or non_operating
        )
        size_category = "FORTY_FOOT_HIGH_CUBE" if high_cube else "FORTY_FOOT_STANDARD_HEIGHT"
        size_rule = "explicit_40_foot_with_reviewed_height_or_thermal_marker"
    else:
        return ReviewedEquipmentSurface(
            printed_surface=printed_surface,
            normalized_surface=normalized,
            resolution="unresolved_source_surface",
            size_category=None,
            type_category=type_category,
            thermal_operation="not_indicated",
            review_rule="surface_has_no_reviewed_length_semantics",
        )

    return ReviewedEquipmentSurface(
        printed_surface=printed_surface,
        normalized_surface=normalized,
        resolution="reviewed_source_grammar",
        size_category=size_category,
        type_category=type_category,
        thermal_operation=(
            "active"
            if temperature_present
            else "non_operating"
            if non_operating
            else "not_indicated"
        ),
        review_rule=f"{size_rule}+{type_rule}",
    )


def build_equipment_semantic_support(
    observations: Sequence[SourceEquipmentObservation],
) -> EquipmentSemanticSupport:
    seen: set[tuple[str, str]] = set()
    counts: Counter[tuple[ContainerSizeCategory, ContainerTypeCategory, bool]] = Counter()
    documents: dict[tuple[ContainerSizeCategory, ContainerTypeCategory, bool], set[str]] = {}
    type_counts: Counter[tuple[ContainerTypeCategory, bool]] = Counter()
    type_documents: dict[tuple[ContainerTypeCategory, bool], set[str]] = {}
    unresolved: Counter[str] = Counter()
    temperature_rows = type_temperature_rows = resolved_temperature_rows = non_operating = 0
    for observation in observations:
        key = (observation.document_id, observation.row_id)
        if key in seen:
            raise ValueError(f"duplicate source equipment observation: {key!r}")
        seen.add(key)
        temperature_rows += int(observation.temperature_present)
        reviewed = review_source_equipment_surface(
            observation.printed_surface,
            temperature_present=observation.temperature_present,
        )
        if reviewed.type_category is not None:
            if observation.temperature_present and reviewed.type_category not in (
                TEMPERATURE_CAPABLE_CONTAINER_TYPES
            ):
                raise ValueError("resolved temperature row has non-thermal equipment")
            type_key = (reviewed.type_category, observation.temperature_present)
            type_counts[type_key] += 1
            type_documents.setdefault(type_key, set()).add(observation.document_id)
            type_temperature_rows += int(observation.temperature_present)
        if reviewed.resolution == "unresolved_source_surface":
            unresolved[observation.printed_surface or "<ABSENT>"] += 1
            continue
        assert reviewed.size_category is not None and reviewed.type_category is not None
        resolved_temperature_rows += int(observation.temperature_present)
        non_operating += int(reviewed.thermal_operation == "non_operating")
        support_key = (
            reviewed.size_category,
            reviewed.type_category,
            observation.temperature_present,
        )
        counts[support_key] += 1
        documents.setdefault(support_key, set()).add(observation.document_id)
    rows = tuple(
        EquipmentJointSupport(
            size_category=size,
            type_category=type_category,
            active_temperature=active,
            occurrences=count,
            document_count=len(documents[(size, type_category, active)]),
        )
        for (size, type_category, active), count in sorted(counts.items())
    )
    if not rows:
        raise ValueError("source equipment support is empty")
    type_rows = tuple(
        EquipmentTypeSupport(
            type_category=type_category,
            active_temperature=active,
            occurrences=count,
            document_count=len(type_documents[(type_category, active)]),
        )
        for (type_category, active), count in sorted(type_counts.items())
    )
    joint_type_keys = {(row.type_category, row.active_temperature) for row in rows}
    unsupported_type_keys = {
        (row.type_category, row.active_temperature)
        for row in type_rows
        if (row.type_category, row.active_temperature) not in joint_type_keys
    }
    if unsupported_type_keys:
        raise ValueError(
            "source type support has no size-conditional evidence: "
            f"{sorted(unsupported_type_keys)!r}"
        )
    return EquipmentSemanticSupport(
        type_rows=type_rows,
        rows=rows,
        unresolved_surfaces=tuple(sorted(unresolved.items())),
        audit=EquipmentSemanticAudit(
            input_rows=len(observations),
            type_resolved_rows=sum(type_counts.values()),
            resolved_rows=sum(counts.values()),
            unresolved_rows=sum(unresolved.values()),
            temperature_rows=temperature_rows,
            type_resolved_temperature_rows=type_temperature_rows,
            resolved_temperature_rows=resolved_temperature_rows,
            non_operating_reefer_rows=non_operating,
            type_support_rows=len(type_rows),
            joint_support_rows=len(rows),
        ),
    )


def sample_equipment_semantic(
    *,
    support: EquipmentSemanticSupport,
    active_temperature: bool,
    stream: DeterministicStream,
    configured_joint_weights: Mapping[str, int] | None = None,
) -> GeneratedEquipmentSemantic:
    """Sample one compatible semantic pair from source or explicit overrides."""

    empirical_types = tuple(
        row
        for row in support.type_rows
        if row.active_temperature == active_temperature
        and (not active_temperature or row.type_category in TEMPERATURE_CAPABLE_CONTAINER_TYPES)
    )
    if not empirical_types:
        raise ValueError("equipment support has no compatible semantic type")
    method: Literal[
        "source_type_marginal_size_conditional",
        "configured_joint_override",
    ]
    if configured_joint_weights:
        method = "configured_joint_override"
        configured: list[EquipmentJointSupport] = []
        weights: list[int] = []
        for key, weight in sorted(configured_joint_weights.items()):
            try:
                raw_size, raw_type = key.split("|", maxsplit=1)
                size = _SIZE_CATEGORY_ADAPTER.validate_python(raw_size, strict=True)
                type_category = _TYPE_CATEGORY_ADAPTER.validate_python(raw_type, strict=True)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid configured equipment pair: {key!r}") from error
            if active_temperature and type_category not in TEMPERATURE_CAPABLE_CONTAINER_TYPES:
                raise ValueError(
                    f"configured active equipment pair is not temperature-capable: {key!r}"
                )
            configured.append(
                EquipmentJointSupport(
                    size_category=size,
                    type_category=type_category,
                    active_temperature=active_temperature,
                    occurrences=weight,
                    document_count=0,
                )
            )
            weights.append(weight)
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("configured equipment weights must contain positive compatible mass")
        eligible = tuple(configured)
    else:
        method = "source_type_marginal_size_conditional"
        type_weights = [row.occurrences for row in empirical_types]
        type_draw = stream.derive("semantic-type").randbelow(sum(type_weights))
        type_cumulative = 0
        selected_type: ContainerTypeCategory | None = None
        for type_row, weight in zip(empirical_types, type_weights, strict=True):
            type_cumulative += weight
            if type_draw < type_cumulative:
                selected_type = type_row.type_category
                break
        if selected_type is None:
            raise RuntimeError("equipment type support cumulative weights are inconsistent")
        eligible = tuple(
            row
            for row in support.rows
            if row.active_temperature == active_temperature and row.type_category == selected_type
        )
        weights = [row.occurrences for row in eligible]
    draw = stream.derive("semantic-equipment").randbelow(sum(weights))
    cumulative = 0
    selected: EquipmentJointSupport | None = None
    for joint_row, weight in zip(eligible, weights, strict=True):
        cumulative += weight
        if draw < cumulative:
            selected = joint_row
            break
    if selected is None:
        raise RuntimeError("equipment support cumulative weights are inconsistent")
    return GeneratedEquipmentSemantic(
        size_category=selected.size_category,
        type_category=selected.type_category,
        application_code=semantic_container_code(
            selected.size_category,
            selected.type_category,
        ),
        active_temperature=active_temperature,
        sampling_method=method,
    )
