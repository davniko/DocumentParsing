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
from functools import lru_cache
from typing import Any, Literal

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
_SIZE_PRINTED_SURFACE: dict[ContainerSizeCategory, str] = {
    "TWENTY_FOOT_STANDARD_HEIGHT": "20' STANDARD HEIGHT",
    "TWENTY_FOOT_HIGH_CUBE": "20' HIGH CUBE",
    "FORTY_FOOT_STANDARD_HEIGHT": "40' STANDARD HEIGHT",
    "FORTY_FOOT_HIGH_CUBE": "40' HIGH CUBE",
    "FORTY_FIVE_FOOT_HIGH_CUBE": "45' HIGH CUBE",
}
_TYPE_PRINTED_SURFACE: dict[ContainerTypeCategory, str] = {
    "GENERAL_PURPOSE": "GENERAL PURPOSE",
    "VENTILATED_GENERAL_PURPOSE": "VENTILATED GENERAL PURPOSE",
    "DRY_BULK": "DRY BULK",
    "NAMED_CARGO": "NAMED CARGO",
    "REFRIGERATED": "REFRIGERATED",
    "REFRIGERATED_AND_HEATED": "REFRIGERATED AND HEATED",
    "SELF_POWERED_REFRIGERATED": "SELF POWERED REFRIGERATED",
    "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT": ("REFRIGERATED HEATED REMOVABLE EQUIPMENT"),
    "INSULATED": "INSULATED",
    "OPEN_TOP": "OPEN TOP",
    "PLATFORM": "PLATFORM",
    "PLATFORM_FIXED": "FIXED PLATFORM",
    "PLATFORM_COLLAPSIBLE": "COLLAPSIBLE PLATFORM",
    "PLATFORM_COMPLETE_SUPERSTRUCTURE": "PLATFORM COMPLETE SUPERSTRUCTURE",
    "PLATFORM_NAMED_CARGO": "NAMED CARGO PLATFORM",
    "PRESSURIZED_TANK": "PRESSURIZED TANK",
    "DRY_HOPPER_TANK": "DRY HOPPER TANK",
    "DRY_REAR_DISCHARGE_TANK": "DRY REAR DISCHARGE TANK",
    "AIR_SURFACE": "AIR SURFACE",
}

# ISO 6346 size digits encode length then height, not a length in feet.
# BIC: https://www.bic-code.org/size-type-code/ and /type-code-designation/.
_ISO_SIZES: dict[str, ContainerSizeCategory] = {
    "22": "TWENTY_FOOT_STANDARD_HEIGHT",
    "25": "TWENTY_FOOT_HIGH_CUBE",
    "42": "FORTY_FOOT_STANDARD_HEIGHT",
    "45": "FORTY_FOOT_HIGH_CUBE",
    "55": "FORTY_FIVE_FOOT_HIGH_CUBE",
}
_ISO_TYPES: dict[str, ContainerTypeCategory] = {
    **dict.fromkeys(("G0", "G1", "G2", "G3", "G9"), "GENERAL_PURPOSE"),
    **dict.fromkeys(("V0", "V2", "V4"), "VENTILATED_GENERAL_PURPOSE"),
    **dict.fromkeys(("B0", "B1", "B3", "B4", "B5", "B6", "B7", "B8", "B9"), "DRY_BULK"),
    **dict.fromkeys(("S0", "S1", "S2"), "NAMED_CARGO"),
    "R0": "REFRIGERATED",
    "R1": "REFRIGERATED_AND_HEATED",
    "R2": "SELF_POWERED_REFRIGERATED",
    "R3": "SELF_POWERED_REFRIGERATED",
    **dict.fromkeys(("H0", "H1", "H2"), "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT"),
    **dict.fromkeys(("H5", "H6"), "INSULATED"),
    **dict.fromkeys(("U0", "U1", "U2", "U3", "U4", "U6", "U9"), "OPEN_TOP"),
    "P0": "PLATFORM",
    **dict.fromkeys(("P1", "P2"), "PLATFORM_FIXED"),
    **dict.fromkeys(("P3", "P4"), "PLATFORM_COLLAPSIBLE"),
    "P5": "PLATFORM_COMPLETE_SUPERSTRUCTURE",
    **dict.fromkeys(("P6", "P7", "P8", "P9"), "PLATFORM_NAMED_CARGO"),
    **dict.fromkeys(("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"), "PRESSURIZED_TANK"),
    **dict.fromkeys(("N0", "N1"), "DRY_HOPPER_TANK"),
    **dict.fromkeys(("N3", "N4", "N5"), "DRY_REAR_DISCHARGE_TANK"),
    "A0": "AIR_SURFACE",
}
_COMPACT_TYPES: dict[str, ContainerTypeCategory] = {
    "GP": "GENERAL_PURPOSE",
    "DC": "GENERAL_PURPOSE",
    "DV": "GENERAL_PURPOSE",
    "VH": "VENTILATED_GENERAL_PURPOSE",
    "BU": "DRY_BULK",
    "SN": "NAMED_CARGO",
    "RE": "REFRIGERATED",
    "RF": "REFRIGERATED",
    "RT": "REFRIGERATED_AND_HEATED",
    "RS": "SELF_POWERED_REFRIGERATED",
    "HI": "INSULATED",
    "UT": "OPEN_TOP",
    "OT": "OPEN_TOP",
    "PL": "PLATFORM",
    "PF": "PLATFORM_FIXED",
    "PC": "PLATFORM_COLLAPSIBLE",
    "PS": "PLATFORM_COMPLETE_SUPERSTRUCTURE",
    "PT": "PLATFORM_NAMED_CARGO",
    "KL": "PRESSURIZED_TANK",
    "NH": "DRY_HOPPER_TANK",
    "NN": "DRY_REAR_DISCHARGE_TANK",
    "AS": "AIR_SURFACE",
}


def iso_equipment_surface(
    size: ContainerSizeCategory, kind: ContainerTypeCategory, source: str
) -> str | None:
    """Render a supported four-character ISO source form, retaining its detail when valid."""
    match = re.fullmatch(r"\s*[0-9A-Z]{2}(?P<gap>\s*)(?P<kind>[A-Z][0-9])\s*", source.upper())
    if match is None:
        return None
    sizes = [code for code, value in _ISO_SIZES.items() if value == size]
    kinds = [code for code, value in _ISO_TYPES.items() if value == kind]
    if not sizes or not kinds:
        raise ValueError("semantic equipment cannot be expressed by a supported ISO code")
    detail = match["kind"] if match["kind"] in kinds else kinds[0]
    return sizes[0] + match["gap"] + detail


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
    # OCR can remove this word boundary. Match the complete equipment noun,
    # never a prefix such as TANKCONTAINERIZATION or another carrier code.
    if "TANKCONTAINER" in normalized:
        normalized = re.sub(r"\bTANK(?=CONTAINERS?\b)", "TANK ", normalized)
    return " ".join(normalized.split())


def canonical_equipment_surface(
    size_category: ContainerSizeCategory,
    type_category: ContainerTypeCategory,
) -> str:
    """Return one unambiguous human-readable realization of an equipment semantic pair."""

    return f"{_SIZE_PRINTED_SURFACE[size_category]} {_TYPE_PRINTED_SURFACE[type_category]}"


_CANONICAL_EQUIPMENT_PATTERNS = tuple(
    (
        re.compile(rf"(?<![A-Z0-9]){re.escape(surface)}(?![A-Z0-9])"),
        size_category,
        type_category,
    )
    for surface, size_category, type_category in sorted(
        (
            (
                _normalize(canonical_equipment_surface(size_category, type_category)),
                size_category,
                type_category,
            )
            for size_category in _SIZE_PRINTED_SURFACE
            for type_category in _TYPE_PRINTED_SURFACE
        ),
        key=lambda row: len(row[0]),
        reverse=True,
    )
)


def singleton_equipment_surface(text: str) -> str:
    """Unwrap only an explicitly single-container count, not ISO digits or dimensions."""
    counted = re.fullmatch(
        r"(?:0*1\s*[xX\u00d7]\s*(?P<forward>(?:20|40|45).+)|"
        r"(?P<reverse>(?:20|40|45).+?)\s*[xX\u00d7]\s*0*1)",
        text.strip(),
    )
    return counted["forward"] or counted["reverse"] if counted else text


_DIMENSIONAL_SIZE = re.compile(
    r"(20|40|45)\s*['\u2019`]\s*[Xx\u00d7]\s*(8|9)\s*['\u2019`]\s*6\s*[\"\u201d]?"
)
_DIMENSIONAL_SIZES: dict[tuple[str, str], ContainerSizeCategory] = {
    ("20", "8"): "TWENTY_FOOT_STANDARD_HEIGHT",
    ("20", "9"): "TWENTY_FOOT_HIGH_CUBE",
    ("40", "8"): "FORTY_FOOT_STANDARD_HEIGHT",
    ("40", "9"): "FORTY_FOOT_HIGH_CUBE",
    ("45", "9"): "FORTY_FIVE_FOOT_HIGH_CUBE",
}


def dimensional_equipment_size(value: str) -> ContainerSizeCategory | None:
    """Read explicit length/height only, without inferring the equipment type."""
    match = _DIMENSIONAL_SIZE.fullmatch(value.strip())
    return _DIMENSIONAL_SIZES.get((match[1], match[2])) if match else None


def partial_equipment_constraint(
    container: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Return printed length, type and size constraints without inventing height.

    Shared by physical sampling and receipt reconciliation: an absent dimension
    is unknown, whereas an opaque carrier code still requires source review.
    """
    if {"sizeCategory", "typeCategory"} & container.keys():
        raise ValueError("partial semantic category pair is not a valid equipment label")
    description = container.get("typeDescription")
    if description is None:
        # A printed setpoint constrains the sampler's thermal capability, not
        # the extraction label's visibility or a guessed container size/type.
        return None, None, None
    text = description.strip()
    unwrapped = singleton_equipment_surface(text)
    if unwrapped != text:
        return partial_equipment_constraint({**container, "typeDescription": unwrapped})
    if re.fullmatch(r"(?:CONTAINER|CTNR|CTR|LCL\.?|CY/FO)", text, re.I):
        return None, None, None
    if re.fullmatch(r"GENERAL\s+PURPOSE(?:\s+(?:CONT\.?|CONTAINER))?", text, re.I):
        return None, "GENERAL_PURPOSE", None
    if re.fullmatch(r"REFRIGERATED\s+CONTAINER", text, re.I):
        return None, "REFRIGERATED", None
    flat = re.fullmatch(r"(20|40|45)\s*['\u2019`]?\s*FLATRACK\s+COLLAPSIBLE", text, re.I)
    if flat:
        return flat[1], "PLATFORM_COLLAPSIBLE", None
    high_cube = re.fullmatch(r"(20|40|45)\s*['\u2019`]?\s*HIGH\s*CUBE", text, re.I)
    if high_cube:
        return (
            high_cube[1],
            None,
            {
                "20": "TWENTY_FOOT_HIGH_CUBE",
                "40": "FORTY_FOOT_HIGH_CUBE",
                "45": "FORTY_FIVE_FOOT_HIGH_CUBE",
            }[high_cube[1]],
        )
    length_only = re.fullmatch(r"(?:\((20|40|45)\)|(20|40|45)\s*(?:['\u2019`]|FT)?)", text, re.I)
    if length_only:
        return length_only[1] or length_only[2], None, None
    dimensional_size = dimensional_equipment_size(text)
    if dimensional_size is not None:
        return None, None, dimensional_size
    reviewed = review_source_equipment_surface(
        description, temperature_present="temperatureSetpoint" in container
    )
    if reviewed.size_category is not None and reviewed.type_category is not None:
        return None, reviewed.type_category, reviewed.size_category
    thermal_partial = re.fullmatch(
        r"(?:(40)\s*['\u2019`]?\s*(?:RA|RK|RO|RQ)|(20|40|45)\s*['\u2019`]?\s*RFH)",
        text,
        re.I,
    )
    if (
        thermal_partial
        and reviewed.type_category == "REFRIGERATED"
        and reviewed.size_category is None
    ):
        return thermal_partial[1] or thermal_partial[2], reviewed.type_category, None
    raise ValueError("partial printed equipment requires a reviewed physical constraint")


# Immutable source grammar is repeatedly evaluated for the same carrier forms
# during scenario search. Bounded memoization avoids that work without holding
# document objects or mutating the returned frozen observations.
@lru_cache(maxsize=8192)
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
    # Equipment summaries commonly include a semantic heading before the count, for example
    # ``CONTAINER: 1 X 20FT GENERAL PURPOSE``. The count is structural, not part of the ISO/BIC
    # equipment meaning, and may occur either at the start or after that heading.
    semantic = re.sub(r"(?:^| )[1-9][0-9]*X(?=(?:20|40|45))", " ", normalized).strip()
    semantic = re.sub(r"\b(20|40|45)(?=DRY\b)", r"\1 ", semantic)
    tokens = frozenset(semantic.split())

    # Closed carrier/EDI spellings, not a substring or an OCR correction.
    # HMM publishes 4H dry as 12.032m / 9'6":
    # https://www.hmm21.com/e-service/information/containerInformation.do
    # Crane's EDI equipment list explicitly defines 40HO as high-cube open top:
    # https://developers.craneww.com/edi/documentation/index.html
    # CMA CGM's Directions for OOG in OT containers confirms OOG as a load
    # qualifier, not a different equipment type. Keep that printed qualifier.
    reviewed_pair = (
        ("FORTY_FOOT_HIGH_CUBE", "GENERAL_PURPOSE")
        if re.fullmatch(r"DC\s*4H(?:\s+CY\s+FO)?", semantic)
        else ("FORTY_FOOT_HIGH_CUBE", "OPEN_TOP")
        if re.fullmatch(r"40\s*HO", semantic)
        else (_ISO_SIZES[{"20": "22", "40": "42"}[oog[1]]], "OPEN_TOP")
        if (oog := re.fullmatch(r"(20|40)\s*OT\s+OOG", semantic))
        else None
    )
    if reviewed_pair is not None:
        return ReviewedEquipmentSurface(
            printed_surface=printed_surface,
            normalized_surface=normalized,
            resolution="reviewed_source_grammar",
            size_category=_SIZE_CATEGORY_ADAPTER.validate_python(reviewed_pair[0]),
            type_category=_TYPE_CATEGORY_ADAPTER.validate_python(reviewed_pair[1]),
            thermal_operation="not_indicated",
            review_rule="documented_complete_carrier_equipment_surface",
        )

    # A source carrier prints 40RF96 beside an independent 40'HC REEFER receipt.
    # The terminal 96 is the 9'6" height in the documented reefer size code,
    # not a two-digit container seal. Keep this observed grammar narrower than
    # an assumed family of unobserved equipment codes.
    if semantic == "40RF96":
        return ReviewedEquipmentSurface(
            printed_surface=printed_surface,
            normalized_surface=normalized,
            resolution="reviewed_source_grammar",
            size_category="FORTY_FOOT_HIGH_CUBE",
            type_category="REFRIGERATED",
            thermal_operation="active" if temperature_present else "not_indicated",
            review_rule="corroborated_40_foot_9_6_reefer_code",
        )

    iso = re.fullmatch(r"(?P<size>[0-9A-Z][0-9A-Z])\s*(?P<kind>[A-Z][0-9])", semantic)
    if iso:
        size, kind = _ISO_SIZES.get(iso["size"]), _ISO_TYPES.get(iso["kind"])
        supported = size is not None and kind is not None
        return ReviewedEquipmentSurface(
            printed_surface=printed_surface,
            normalized_surface=normalized,
            resolution="reviewed_source_grammar" if supported else "unresolved_source_surface",
            size_category=size if supported else None,
            type_category=kind if supported else None,
            thermal_operation=(
                "active"
                if kind in TEMPERATURE_CAPABLE_CONTAINER_TYPES and temperature_present
                else "not_indicated"
            ),
            review_rule="iso_6346_size_type" if supported else "unsupported_iso_6346_size_type",
        )

    # The synthesis renderer exposes this complete semantic phrase when a carrier-specific source
    # abbreviation cannot be projected safely. Recognize it before the corpus shorthand grammar
    # so every model-facing category has an exact, round-trippable printed representation.
    for pattern, canonical_size_category, canonical_type_category in _CANONICAL_EQUIPMENT_PATTERNS:
        if pattern.search(semantic):
            return ReviewedEquipmentSurface(
                printed_surface=printed_surface,
                normalized_surface=normalized,
                resolution="reviewed_source_grammar",
                size_category=canonical_size_category,
                type_category=canonical_type_category,
                thermal_operation=(
                    "active"
                    if canonical_type_category in TEMPERATURE_CAPABLE_CONTAINER_TYPES
                    and temperature_present
                    else "not_indicated"
                ),
                review_rule="canonical_semantic_equipment_surface",
            )

    compact = re.fullmatch(
        r"(?:(?P<length>20|40|45)\s*(?P<kind>[A-Z]{2})|"
        r"(?P<reverse_kind>[A-Z]{2})\s*(?P<reverse_length>20|40|45))",
        semantic,
    )
    if compact:
        compact_kind = _COMPACT_TYPES.get(compact["kind"] or compact["reverse_kind"])
        if compact_kind is not None:
            compact_size = _ISO_SIZES[
                {"20": "22", "40": "42", "45": "55"}[compact["length"] or compact["reverse_length"]]
            ]
            return ReviewedEquipmentSurface(
                printed_surface,
                normalized,
                "reviewed_source_grammar",
                compact_size,
                compact_kind,
                "active"
                if temperature_present and compact_kind in TEMPERATURE_CAPABLE_CONTAINER_TYPES
                else "not_indicated",
                "explicit_length_type_group",
            )

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
            for value in ("HR", "RA", "RE", "RF", "RH", "RK", "RO", "RQ", "RFH")
            if value in tokens
            or re.search(rf"(?:20|40|45){value}(?:\s|$)", semantic)
            or semantic == f"{value}40"
        ),
        None,
    )
    explicit_thermal = bool(
        re.search(r"\b(?:REEF|REEFER|REFRIGERATED|RF)\b", semantic)
        or carrier_thermal_code is not None
    )
    explicit_nonthermal = bool(
        tokens & {"DRY", "GP", "DC", "DV", "OT", "OPEN", "TANK", "PLATFORM", "FLAT"}
        or {"GENERAL", "PURPOSE"} <= tokens
    )
    if explicit_thermal or (temperature_present and not explicit_nonthermal) or non_operating:
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
            "STD",
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

    # Quoted dimensions normalize to ``40 X9 6``; unquoted dimensions can
    # normalize to ``40X9 6``. X is the dimension separator, not part of height.
    explicit_nine_six = bool(re.search(r"(?:^|[ X])9\s+6(?:\s|$)", semantic))
    # Scan Global's BLU Brasil equipment glossary (09/2024) lists RFH as
    # reefer equipment, without establishing height. This applies equally
    # to every explicit length, rather than inventing a default height.
    if carrier_thermal_code == "RFH" and not (
        {"HIGH", "CUBE"} <= tokens
        or tokens & {"HC", "HQ", "HICU", "HCPW", "SD96"}
        or explicit_nine_six
    ):
        return ReviewedEquipmentSurface(
            printed_surface,
            normalized,
            "unresolved_source_surface",
            None,
            type_category,
            "active" if temperature_present else "not_indicated",
            "reviewed_reefer_type_with_unresolved_carrier_height_code",
        )
    if length == 45:
        size_category: ContainerSizeCategory = "FORTY_FIVE_FOOT_HIGH_CUBE"
        size_rule = "explicit_45_foot"
    elif length == 20:
        high_cube = bool(({"HIGH", "CUBE"} <= tokens) or ({"HI", "CUBE"} <= tokens)) or bool(
            tokens & {"HC", "HQ", "HICU", "HCPW", "SD96"} or explicit_nine_six
        )
        size_category = "TWENTY_FOOT_HIGH_CUBE" if high_cube else "TWENTY_FOOT_STANDARD_HEIGHT"
        size_rule = "explicit_20_foot_with_height_marker"
    elif length == 40:
        if carrier_thermal_code in {"RA", "RK", "RO", "RQ"} and not (
            "HIGH" in tokens
            or "CUBE" in tokens
            or tokens & {"HC", "HQ", "HICU", "HCPW", "SD96"}
            or explicit_nine_six
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
        carrier_high_reefer = carrier_thermal_code in {"HR", "RH"}
        high_cube = bool(
            ({"HIGH", "CUBE"} <= tokens)
            or ({"HI", "CUBE"} <= tokens)
            or tokens & {"HC", "HQ", "HICU", "HCPW", "SD96"}
            or "HIGHCUBE" in tokens
            or re.search(r"(?:40|45)\s*(?:H|HC|HQ|SD96)(?:\s|$)", semantic)
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
