"""Pinned BIC/ISO 6346 size-type semantics for controlled equipment synthesis.

The MPCI application accepts a broad syntactic four-character code space but
does not publish a semantic label for each code.  This module therefore treats
the Bureau International des Containers tables as the semantic authority and
keeps three values deliberately separate:

* an exact ISO size/type code (semantic identity),
* the source document's printed surface, and
* the currently empty MPCI model-facing ``typeCategory`` vocabulary.

Only assigned BIC type rows and explicit group codes are eligible.  The parser
does not contain carrier aliases such as ``40HC`` or ``20DV``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
IsoSizeTypeCode = Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{4}$")]
IsoTypeCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][0-9A-Z]$")]
IsoSizeCode = Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{2}$")]
ThermalCapability = Literal["none", "refrigerated", "heated", "thermal_other"]

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_HEADINGS = (
    "Code",
    "Type designation",
    "Type group code",
    "Main characteristics",
    "Detailed type code a",
    "Detailed type code b",
)
_TYPE_CODE = re.compile(r"^[A-Z][0-9A-Z]$")
_SOURCE_FILE_NAMES = frozenset({"size-type-code.html", "type-code-designation.html"})


class EquipmentRegistryError(ValueError):
    """A BIC snapshot or requested equipment identity is invalid."""


class EquipmentSourceFile(BaseModel):
    model_config = _STRICT

    path: Annotated[str, StringConstraints(min_length=1)]
    url: Annotated[str, StringConstraints(pattern=r"^https://www\.bic-code\.org/")]
    sha256: Sha256
    bytes: Annotated[int, Field(gt=0)]


class EquipmentSourceManifest(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    authority: Literal["Bureau International des Containers (BIC)"]
    standardEdition: Literal["ISO 6346:2022"]
    retrievedAt: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")]
    purpose: Annotated[str, StringConstraints(min_length=1)]
    sources: tuple[EquipmentSourceFile, EquipmentSourceFile]

    @model_validator(mode="after")
    def sources_are_complete(self) -> EquipmentSourceManifest:
        names = tuple(row.path for row in self.sources)
        if frozenset(names) != _SOURCE_FILE_NAMES or len(names) != len(set(names)):
            raise ValueError("equipment manifest must pin each required BIC page exactly once")
        return self


class EquipmentTypeIdentity(BaseModel):
    """One assigned detailed or group-level BIC type identity."""

    model_config = _STRICT

    type_code: IsoTypeCode
    family_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]$")]
    family_designation: Annotated[str, StringConstraints(min_length=1)]
    type_designation: Annotated[str, StringConstraints(min_length=1)]
    characteristics: Annotated[str, StringConstraints(min_length=1)]
    code_kind: Literal["group", "detailed_numeric", "detailed_reduced_strength"]
    thermal_capability: ThermalCapability
    supports_temperature_setpoint: bool

    @model_validator(mode="after")
    def family_and_thermal_state_are_consistent(self) -> EquipmentTypeIdentity:
        if not self.type_code.startswith(self.family_code):
            raise ValueError("equipment type code differs from its family")
        if self.supports_temperature_setpoint and self.thermal_capability not in {
            "refrigerated",
            "heated",
        }:
            raise ValueError("only refrigerated or heated equipment can carry a setpoint")
        return self


class EquipmentIdentity(BaseModel):
    """A validated four-character code plus the BIC semantics it implies."""

    model_config = _STRICT

    size_type_code: IsoSizeTypeCode
    size_code: IsoSizeCode
    length_code: Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]$")]
    height_width_code: Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]$")]
    type: EquipmentTypeIdentity

    @model_validator(mode="after")
    def code_parts_balance(self) -> EquipmentIdentity:
        if self.size_type_code != self.size_code + self.type.type_code:
            raise ValueError("equipment size/type code differs from its components")
        if self.size_code != self.length_code + self.height_width_code:
            raise ValueError("equipment size code differs from its components")
        return self


class EquipmentRegistryReceipt(BaseModel):
    model_config = _STRICT

    manifest_path: Annotated[str, StringConstraints(min_length=1)]
    manifest_sha256: Sha256
    standard_edition: Literal["ISO 6346:2022"]
    length_codes: tuple[Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]$")], ...]
    height_width_codes: tuple[Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]$")], ...]
    type_codes: tuple[EquipmentTypeIdentity, ...]
    excluded_unassigned_rows: Annotated[int, Field(ge=0)]
    excluded_invalid_source_codes: tuple[Annotated[str, StringConstraints(min_length=1)], ...]

    @model_validator(mode="after")
    def registry_is_unique_and_sorted(self) -> EquipmentRegistryReceipt:
        if self.length_codes != tuple(sorted(set(self.length_codes))):
            raise ValueError("equipment length codes must be unique and sorted")
        if self.height_width_codes != tuple(sorted(set(self.height_width_codes))):
            raise ValueError("equipment height/width codes must be unique and sorted")
        type_codes = tuple(row.type_code for row in self.type_codes)
        if type_codes != tuple(sorted(set(type_codes))):
            raise ValueError("equipment type codes must be unique and sorted")
        return self

    def classify(self, value: str) -> EquipmentIdentity:
        """Resolve an exact ISO code; aliases and free text fail closed."""

        if re.fullmatch(r"[0-9A-Z]{4}", value) is None:
            raise EquipmentRegistryError(f"equipment value is not an exact ISO code: {value!r}")
        length_code, height_width_code = value[0], value[1]
        if length_code not in self.length_codes or height_width_code not in self.height_width_codes:
            raise EquipmentRegistryError(f"equipment code has an unknown size component: {value!r}")
        row = next((item for item in self.type_codes if item.type_code == value[2:]), None)
        if row is None:
            raise EquipmentRegistryError(f"equipment code has an unassigned type: {value!r}")
        return EquipmentIdentity(
            size_type_code=value,
            size_code=value[:2],
            length_code=length_code,
            height_width_code=height_width_code,
            type=row,
        )


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
        elif self._table_depth == 1 and tag == "tr":
            self._row = []
        elif self._table_depth == 1 and self._row is not None and tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth == 1 and tag in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif self._table_depth == 1 and tag == "tr" and self._row is not None:
            assert self._rows is not None
            if self._row:
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1:
                assert self._rows is not None
                self.tables.append(self._rows)
                self._rows = None
            self._table_depth -= 1


def _verified_source(root: Path, source: EquipmentSourceFile) -> bytes:
    path = root / source.path
    if path.is_symlink():
        raise EquipmentRegistryError(f"equipment source cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise EquipmentRegistryError(f"equipment source is not a file: {resolved}")
    payload = resolved.read_bytes()
    if len(payload) != source.bytes or hashlib.sha256(payload).hexdigest() != source.sha256:
        raise EquipmentRegistryError(f"equipment source receipt mismatch: {source.path}")
    return payload


def _tables(payload: bytes) -> list[list[list[str]]]:
    parser = _TableParser()
    parser.feed(payload.decode("utf-8"))
    return parser.tables


def _strip_heading(value: str, heading: str) -> str:
    return value[len(heading) :].strip() if value.startswith(heading) else value.strip()


def _thermal_capability(family: str, code: str) -> tuple[ThermalCapability, bool]:
    if family == "R":
        if code in {"RH", "R7", "R8", "RW", "RX"}:
            return "heated", True
        return "refrigerated", True
    if family == "H":
        if code in {"HR", "H0", "H1", "H2", "HA", "HB", "HD"}:
            return "refrigerated", True
        return "thermal_other", False
    return "none", False


def _parse_type_codes(
    payload: bytes,
) -> tuple[tuple[EquipmentTypeIdentity, ...], int, tuple[str, ...]]:
    tables = _tables(payload)
    if len(tables) != 1 or len(tables[0]) < 100:
        raise EquipmentRegistryError("BIC type page no longer contains the expected table")
    current_family = ""
    current_family_designation = ""
    current_type_designation = ""
    output: dict[str, EquipmentTypeIdentity] = {}
    unassigned = 0
    invalid: set[str] = set()
    for raw in tables[0][1:]:
        if len(raw) != 6:
            raise EquipmentRegistryError("BIC type table row does not contain six cells")
        cells = tuple(
            _strip_heading(value, heading) for value, heading in zip(raw, _HEADINGS, strict=True)
        )
        family, designation, group, characteristics, detailed_a, detailed_b = cells
        if family:
            if re.fullmatch(r"[A-Z]", family) is None or not designation:
                raise EquipmentRegistryError("BIC type family row is malformed")
            current_family = family
            current_family_designation = designation
            current_type_designation = designation
            if group:
                capability, supports_setpoint = _thermal_capability(family, group)
                output[group] = EquipmentTypeIdentity(
                    type_code=group,
                    family_code=family,
                    family_designation=designation,
                    type_designation=designation,
                    characteristics=designation,
                    code_kind="group",
                    thermal_capability=capability,
                    supports_temperature_setpoint=supports_setpoint,
                )
            continue
        if not current_family:
            raise EquipmentRegistryError("BIC detailed row occurs before a family")
        if designation:
            current_type_designation = designation
        if group and group not in output:
            capability, supports_setpoint = _thermal_capability(current_family, group)
            output[group] = EquipmentTypeIdentity(
                type_code=group,
                family_code=current_family,
                family_designation=current_family_designation,
                type_designation=current_type_designation or current_family_designation,
                characteristics=current_type_designation or current_family_designation,
                code_kind="group",
                thermal_capability=capability,
                supports_temperature_setpoint=supports_setpoint,
            )
        if characteristics.casefold() == "unassigned":
            unassigned += 1
            continue
        for index, code in enumerate((detailed_a, detailed_b)):
            if not code:
                continue
            if _TYPE_CODE.fullmatch(code) is None:
                invalid.add(code)
                continue
            capability, supports_setpoint = _thermal_capability(current_family, code)
            kind: Literal["detailed_numeric", "detailed_reduced_strength"] = (
                "detailed_numeric" if index == 0 else "detailed_reduced_strength"
            )
            row = EquipmentTypeIdentity(
                type_code=code,
                family_code=current_family,
                family_designation=current_family_designation,
                type_designation=current_type_designation or current_family_designation,
                characteristics=characteristics or current_type_designation,
                code_kind=kind,
                thermal_capability=capability,
                supports_temperature_setpoint=supports_setpoint,
            )
            previous = output.setdefault(code, row)
            if previous != row:
                raise EquipmentRegistryError(f"BIC type code has conflicting semantics: {code}")
    if len(output) < 60 or not {"GP", "G1", "RE", "R1", "UT", "P1"} <= output.keys():
        raise EquipmentRegistryError("compiled BIC type support is unexpectedly incomplete")
    return tuple(output[key] for key in sorted(output)), unassigned, tuple(sorted(invalid))


def _parse_size_codes(payload: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tables = _tables(payload)
    length_table = next(
        (table for table in tables if table and table[0][:1] == ["Container Length"]),
        None,
    )
    height_table = next((table for table in tables if table and table[0][:1] == [""]), None)
    if length_table is None or height_table is None:
        raise EquipmentRegistryError("BIC size page no longer contains the expected tables")
    length_codes: set[str] = set()
    for row in length_table[2:]:
        if not row:
            continue
        match = re.search(r"Character Code([0-9A-Z])$", row[-1])
        if match and "unassigned" not in " ".join(row).casefold():
            length_codes.add(match.group(1))
    height_codes: set[str] = set()
    for row in height_table[3:]:
        for cell in row[3:]:
            if cell and (cell[-1].isdigit() or cell[-1].isupper()):
                height_codes.add(cell[-1])
    if not {"2", "4", "5"} <= length_codes or not {"2", "5"} <= height_codes:
        raise EquipmentRegistryError("compiled BIC size support is unexpectedly incomplete")
    return tuple(sorted(length_codes)), tuple(sorted(height_codes))


def load_bic_equipment_registry(manifest_path: Path) -> EquipmentRegistryReceipt:
    """Verify a two-page BIC snapshot and compile assigned semantic codes."""

    if manifest_path.is_symlink():
        raise EquipmentRegistryError("equipment manifest cannot be a symlink")
    resolved = manifest_path.resolve(strict=True)
    payload = resolved.read_bytes()
    manifest = EquipmentSourceManifest.model_validate_json(payload, strict=True)
    by_name: Mapping[str, EquipmentSourceFile] = {row.path: row for row in manifest.sources}
    size_payload = _verified_source(resolved.parent, by_name["size-type-code.html"])
    type_payload = _verified_source(resolved.parent, by_name["type-code-designation.html"])
    length_codes, height_codes = _parse_size_codes(size_payload)
    type_codes, unassigned, invalid = _parse_type_codes(type_payload)
    return EquipmentRegistryReceipt(
        manifest_path=str(resolved),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        standard_edition=manifest.standardEdition,
        length_codes=length_codes,
        height_width_codes=height_codes,
        type_codes=type_codes,
        excluded_unassigned_rows=unassigned,
        excluded_invalid_source_codes=invalid,
    )


def canonical_equipment_receipt_json(registry: EquipmentRegistryReceipt) -> bytes:
    """Stable serialized registry evidence suitable for immutable run receipts."""

    return (
        json.dumps(
            registry.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def normalize_equipment_surface(value: str) -> str:
    """Comparison-only normalization; it is never an alias resolver."""

    if not isinstance(value, str) or not value.strip():
        raise EquipmentRegistryError("equipment surface must be non-empty")
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise EquipmentRegistryError("equipment surface contains a control character")
    return " ".join(normalized.split())
