"""Compile a pinned NGA World Port Index/UNLOCODE intersection.

The NGA CSV is authoritative for port membership and facility metadata.  ISO
identity comes only from an exact intersection with the separately pinned
canonical UN/LOCODE artifact: WPI country-name strings are deliberately never
resolved, aliased, or compared.  Every repeated stable WPI-number group is
excluded instead of selecting a preferred row.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import stat
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.route_registry import load_pinned_unlocode_locations
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Locode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}[A-Z0-9]{3}$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=512)]
DecimalText = Annotated[
    str,
    StringConstraints(pattern=r"^-?(?:0|[1-9][0-9]*)\.[0-9]+$", max_length=32),
]

NGA_WPI_SOURCE_URL: Final = (
    "https://msi.nga.mil/api/publications/download?"
    "key=16920959%2FSFH00000%2FUpdatedPub150.csv&type=download"
)
NGA_WPI_PAGE_URL: Final = "https://msi.nga.mil/Publications/WPI"
NGA_WPI_FIELD_REFERENCE_URL: Final = (
    "https://msi.nga.mil/api/publications/download?"
    "key=16920959%2FSFH00000%2FWPI_Explanation_of_Data_Fields.pdf&type=view"
)
NGA_WPI_AUTHORITY: Final = "National Geospatial-Intelligence Agency (NGA)"
NGA_WPI_DATASET: Final = "World Port Index (Pub. 150)"
NGA_WPI_ATTRIBUTION: Final = (
    "National Geospatial-Intelligence Agency (NGA), World Port Index (Pub. 150)"
)

CURRENT_WPI_SOURCE_BYTES: Final = 3_508_891
CURRENT_WPI_SOURCE_SHA256: Final = (
    "315644f1e77966291633145fd351c2762b37702d115926b9ed074b39e8e21667"
)
CURRENT_WPI_SOURCE_RECORDS: Final = 3_807
PINNED_UNLOCODE_RUN_NAME: Final = "unlocode-2025-1-all-functions-v1"
PINNED_UNLOCODE_RELEASE: Final = "2025-1"
PINNED_UNLOCODE_LOCATIONS_BYTES: Final = 25_475_870
PINNED_UNLOCODE_LOCATIONS_SHA256: Final = (
    "c5feda51b7a8e39bb97bdd107fc5d96eed4e973461c27adf053a134c7090c278"
)
PINNED_UNLOCODE_LOCATION_RECORDS: Final = 111_561

_PARSER_CONTRACT: Final = "nga_world_port_index_csv_v1"
_INTERSECTION_POLICY: Final = "exact_wpi_locode_layout_and_exact_unlocode_membership_v1"
_COUNTRY_IDENTITY_POLICY: Final = "first_two_characters_of_intersected_unlocode_v1"
_FACILITY_DEDUPLICATION_POLICY: Final = (
    "exclude_all_repeated_intersected_wpi_numbers_then_sort_by_locode_and_wpi_number_v1"
)
_WHITELIST_SCHEMA_VERSION = 1
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_READ_CHUNK_SIZE = 1024 * 1024
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_WPI_NUMBER = re.compile(r"^(0|[1-9][0-9]*)\.0$")
_SPACED_LOCODE = re.compile(r"^[A-Z]{2} [A-Z0-9]{3}$")
_COMPACT_LOCODE = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}$")

# The official CSV has 109 columns.  The exact ordered header is part of the
# parser contract even though this projection uses only seven fields.
WPI_CSV_COLUMNS: Final = (
    "OID_",
    "World Port Index Number",
    "Region Name",
    "Main Port Name",
    "Alternate Port Name",
    "UN/LOCODE",
    "Country Code",
    "World Water Body",
    "IHO S-130 Sea Area",
    "Sailing Direction or Publication",
    "Publication Link",
    "Standard Nautical Chart",
    "IHO S-57 Electronic Navigational Chart",
    "IHO S-101 Electronic Navigational Chart",
    "Digital Nautical Chart",
    "Tidal Range (m)",
    "Entrance Width (m)",
    "Channel Depth (m)",
    "Anchorage Depth (m)",
    "Cargo Pier Depth (m)",
    "Oil Terminal Depth (m)",
    "Liquified Natural Gas Terminal Depth (m)",
    "Maximum Vessel Length (m)",
    "Maximum Vessel Beam (m)",
    "Maximum Vessel Draft (m)",
    "Offshore Maximum Vessel Length (m)",
    "Offshore Maximum Vessel Beam (m)",
    "Offshore Maximum Vessel Draft (m)",
    "Harbor Size",
    "Harbor Type",
    "Harbor Use",
    "Shelter Afforded",
    "Entrance Restriction - Tide",
    "Entrance Restriction - Heavy Swell",
    "Entrance Restriction - Ice",
    "Entrance Restriction - Other",
    "Overhead Limits",
    "Underkeel Clearance Management System",
    "Good Holding Ground",
    "Turning Area",
    "Port Security",
    "Estimated Time of Arrival Message",
    "Quarantine - Pratique",
    "Quarantine - Sanitation",
    "Quarantine - Other",
    "Traffic Separation Scheme",
    "Vessel Traffic Service",
    "First Port of Entry",
    "US Representative",
    "Pilotage - Compulsory",
    "Pilotage - Available",
    "Pilotage - Local Assistance",
    "Pilotage - Advisable",
    "Tugs - Salvage",
    "Tugs - Assistance",
    "Communications - Telephone",
    "Communications - Telefax",
    "Communications - Radio",
    "Communications - Radiotelephone",
    "Communications - Airport",
    "Communications - Rail",
    "Search and Rescue",
    "NAVAREA",
    "Facilities - Wharves",
    "Facilities - Anchorage",
    "Facilities - Dangerous Cargo Anchorage",
    "Facilities - Med Mooring",
    "Facilities - Beach Mooring",
    "Facilities - Ice Mooring",
    "Facilities - Ro-Ro",
    "Facilities - Solid Bulk",
    "Facilities - Liquid Bulk",
    "Facilities - Container",
    "Facilities - Breakbulk",
    "Facilities - Oil Terminal",
    "Facilities - LNG Terminal",
    "Facilities - Other",
    "Medical Facilities",
    "Garbage Disposal",
    "Chemical Holding Tank Disposal",
    "Degaussing",
    "Dirty Ballast Disposal",
    "Cranes - Fixed",
    "Cranes - Mobile",
    "Cranes - Floating",
    "Cranes Container",
    "Lifts - 100+ Tons",
    "Lifts - 50-100 Tons",
    "Lifts - 25-49 Tons",
    "Lifts - 0-24 Tons",
    "Services - Longshoremen",
    "Services - Electricity",
    "Services -Steam",
    "Services - Navigation Equipment",
    "Services - Electrical Repair",
    "Services - Ice Breaking",
    "Services -Diving",
    "Supplies - Provisions",
    "Supplies - Potable Water",
    "Supplies - Fuel Oil",
    "Supplies - Diesel Oil",
    "Supplies - Aviation Fuel",
    "Supplies - Deck",
    "Supplies - Engine",
    "Repairs",
    "Dry Dock",
    "Railway",
    "Latitude",
    "Longitude",
)

_OID_INDEX = WPI_CSV_COLUMNS.index("OID_")
_WPI_NUMBER_INDEX = WPI_CSV_COLUMNS.index("World Port Index Number")
_MAIN_NAME_INDEX = WPI_CSV_COLUMNS.index("Main Port Name")
_ALTERNATE_NAME_INDEX = WPI_CSV_COLUMNS.index("Alternate Port Name")
_LOCODE_INDEX = WPI_CSV_COLUMNS.index("UN/LOCODE")
_LATITUDE_INDEX = WPI_CSV_COLUMNS.index("Latitude")
_LONGITUDE_INDEX = WPI_CSV_COLUMNS.index("Longitude")


class WorldPortRegistryError(RuntimeError):
    """A pinned WPI/UNLOCODE intersection cannot be compiled exactly."""


class WpiSourcePin(BaseModel):
    model_config = _STRICT

    source_url: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    bytes: Annotated[int, Field(gt=0, le=_MAX_SOURCE_BYTES)]
    sha256: Sha256
    records: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def source_is_official(self) -> WpiSourcePin:
        parsed = urlsplit(self.source_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "msi.nga.mil"
            or parsed.path != "/api/publications/download"
            or parse_qs(parsed.query, strict_parsing=True)
            != {
                "key": ["16920959/SFH00000/UpdatedPub150.csv"],
                "type": ["download"],
            }
            or parsed.fragment
        ):
            raise ValueError("source_url must be the official NGA UpdatedPub150.csv endpoint")
        return self


class UnlocodeArtifactPin(BaseModel):
    model_config = _STRICT

    run_name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    release: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[12]$")]
    relative_path: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    records: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def path_is_canonical(self) -> UnlocodeArtifactPin:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("UN/LOCODE artifact path must be a safe relative POSIX path")
        if self.relative_path != f"{self.run_name}/locations.jsonl":
            raise ValueError("UN/LOCODE artifact path differs from its run identity")
        return self


CURRENT_WPI_SOURCE: Final = WpiSourcePin(
    source_url=NGA_WPI_SOURCE_URL,
    bytes=CURRENT_WPI_SOURCE_BYTES,
    sha256=CURRENT_WPI_SOURCE_SHA256,
    records=CURRENT_WPI_SOURCE_RECORDS,
)
PINNED_UNLOCODE_ARTIFACT: Final = UnlocodeArtifactPin(
    run_name=PINNED_UNLOCODE_RUN_NAME,
    release=PINNED_UNLOCODE_RELEASE,
    relative_path=f"{PINNED_UNLOCODE_RUN_NAME}/locations.jsonl",
    bytes=PINNED_UNLOCODE_LOCATIONS_BYTES,
    sha256=PINNED_UNLOCODE_LOCATIONS_SHA256,
    records=PINNED_UNLOCODE_LOCATION_RECORDS,
)


def _safe_text(value: str, *, field: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field} must be non-empty and have canonical edges")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError(f"{field} must not contain control characters")
    return value


class WorldPortRecord(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = 1
    world_port_index_number: Annotated[int, Field(gt=0)]
    source_oid: Annotated[int, Field(ge=0)]
    locode: Locode
    country_code: CountryCode
    port_name: NonEmptyText
    alternate_port_name: NonEmptyText | None
    latitude: DecimalText
    longitude: DecimalText

    @model_validator(mode="after")
    def identity_and_metadata_are_exact(self) -> WorldPortRecord:
        if self.country_code != self.locode[:2]:
            raise ValueError("world-port country identity must be the UN/LOCODE prefix")
        _safe_text(self.port_name, field="WPI port name")
        if self.alternate_port_name is not None:
            _safe_text(self.alternate_port_name, field="WPI alternate port name")
        try:
            latitude = Decimal(self.latitude)
            longitude = Decimal(self.longitude)
        except InvalidOperation as error:
            raise ValueError("WPI coordinates are not decimal values") from error
        if not latitude.is_finite() or not Decimal("-90") <= latitude <= Decimal("90"):
            raise ValueError("WPI latitude is outside [-90, 90]")
        if not longitude.is_finite() or not Decimal("-180") <= longitude <= Decimal("180"):
            raise ValueError("WPI longitude is outside [-180, 180]")
        return self


class WorldPortIntersectionAudit(BaseModel):
    model_config = _STRICT

    source_rows: Annotated[int, Field(ge=0)]
    excluded_blank_locode_rows: Annotated[int, Field(ge=0)]
    excluded_malformed_locode_rows: Annotated[int, Field(ge=0)]
    excluded_locode_absent_from_pinned_unlocode_rows: Annotated[int, Field(ge=0)]
    intersected_source_rows: Annotated[int, Field(ge=0)]
    spaced_locode_rows: Annotated[int, Field(ge=0)]
    compact_locode_rows: Annotated[int, Field(ge=0)]
    accepted_port_records: Annotated[int, Field(ge=0)]
    excluded_duplicate_wpi_number_groups: Annotated[int, Field(ge=0)]
    excluded_duplicate_wpi_number_rows: Annotated[int, Field(ge=0)]
    multiple_facility_locodes: Annotated[int, Field(ge=0)]
    distinct_country_codes: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def accounting_balances(self) -> WorldPortIntersectionAudit:
        if self.source_rows != (
            self.excluded_blank_locode_rows
            + self.excluded_malformed_locode_rows
            + self.excluded_locode_absent_from_pinned_unlocode_rows
            + self.intersected_source_rows
        ):
            raise ValueError("WPI source-row audit does not balance")
        if self.intersected_source_rows != self.spaced_locode_rows + self.compact_locode_rows:
            raise ValueError("WPI accepted LOCODE-layout audit does not balance")
        if self.intersected_source_rows != (
            self.accepted_port_records + self.excluded_duplicate_wpi_number_rows
        ):
            raise ValueError("WPI repeated-facility exclusion audit does not balance")
        if (self.excluded_duplicate_wpi_number_groups == 0) != (
            self.excluded_duplicate_wpi_number_rows == 0
        ) or (
            self.excluded_duplicate_wpi_number_rows < 2 * self.excluded_duplicate_wpi_number_groups
        ):
            raise ValueError("WPI repeated-facility group accounting is inconsistent")
        if self.multiple_facility_locodes > self.accepted_port_records:
            raise ValueError("WPI multi-facility LOCODE count exceeds accepted records")
        if self.distinct_country_codes > self.accepted_port_records:
            raise ValueError("WPI country count exceeds whitelist locations")
        return self


class WorldPortRegistrySourceReceipt(BaseModel):
    model_config = _STRICT

    authority: Literal["National Geospatial-Intelligence Agency (NGA)"]
    dataset: Literal["World Port Index (Pub. 150)"]
    source_url: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    official_page_url: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    field_reference_url: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    source_bytes: Annotated[int, Field(gt=0)]
    source_sha256: Sha256
    source_records: Annotated[int, Field(gt=0)]
    parser_contract: Literal["nga_world_port_index_csv_v1"]
    attribution: Literal[
        "National Geospatial-Intelligence Agency (NGA), World Port Index (Pub. 150)"
    ]


class WorldPortRegistryPolicy(BaseModel):
    model_config = _STRICT

    intersection: Literal["exact_wpi_locode_layout_and_exact_unlocode_membership_v1"]
    country_identity: Literal["first_two_characters_of_intersected_unlocode_v1"]
    facility_deduplication: Literal[
        "exclude_all_repeated_intersected_wpi_numbers_then_sort_by_locode_and_wpi_number_v1"
    ]


class WorldPortRegistryArtifactReceipt(BaseModel):
    model_config = _STRICT

    role: Literal["canonical_port_whitelist_jsonl"]
    path: Literal["port-whitelist.jsonl"]
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    records: Annotated[int, Field(gt=0)]


class WorldPortRegistryReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    source: WorldPortRegistrySourceReceipt
    unlocode_artifact: UnlocodeArtifactPin
    policy: WorldPortRegistryPolicy
    audit: WorldPortIntersectionAudit
    artifacts: tuple[WorldPortRegistryArtifactReceipt, ...] = Field(min_length=1, max_length=1)
    content_sha256: Sha256

    @model_validator(mode="after")
    def receipt_is_self_hashed_and_balanced(self) -> WorldPortRegistryReceipt:
        artifact = self.artifacts[0]
        if artifact.records != self.audit.accepted_port_records:
            raise ValueError("WPI whitelist artifact count differs from intersection audit")
        body = self.model_dump(mode="json", exclude={"content_sha256"})
        if sha256_bytes(canonical_json_bytes(body)) != self.content_sha256:
            raise ValueError("WPI registry receipt content SHA-256 is invalid")
        return self


@dataclass(frozen=True, slots=True)
class CompiledWorldPortRegistry:
    root: Path
    receipt: WorldPortRegistryReceipt
    commit_receipt: StagedCommitReceipt
    created: bool


def _parse_integral_decimal(value: str, *, field: str, row_number: int) -> int:
    match = _WPI_NUMBER.fullmatch(value)
    if match is None:
        raise WorldPortRegistryError(f"WPI CSV row {row_number} has invalid {field}: {value!r}")
    return int(match.group(1))


def _parse_locode(value: str) -> tuple[str | None, str]:
    if not value.strip():
        return None, "blank"
    if _SPACED_LOCODE.fullmatch(value) is not None:
        return value[:2] + value[3:], "spaced"
    if _COMPACT_LOCODE.fullmatch(value) is not None:
        return value, "compact"
    return None, "malformed"


def _source_record(
    row: Sequence[str],
    *,
    row_number: int,
    locode: str,
    wpi_number: int,
) -> WorldPortRecord:
    raw_alternate = row[_ALTERNATE_NAME_INDEX]
    alternate: str | None = None if raw_alternate == " " else raw_alternate
    if alternate is not None and not alternate.strip():
        raise WorldPortRegistryError(
            f"WPI CSV row {row_number} has a non-canonical blank alternate port name"
        )
    try:
        return WorldPortRecord.model_validate(
            {
                "schema_version": 1,
                "world_port_index_number": wpi_number,
                "source_oid": _parse_integral_decimal(
                    row[_OID_INDEX], field="OID", row_number=row_number
                ),
                "locode": locode,
                "country_code": locode[:2],
                "port_name": row[_MAIN_NAME_INDEX],
                "alternate_port_name": alternate,
                "latitude": row[_LATITUDE_INDEX],
                "longitude": row[_LONGITUDE_INDEX],
            },
            strict=True,
        )
    except ValueError as error:
        raise WorldPortRegistryError(
            f"WPI CSV row {row_number} has invalid selected facility metadata: {error}"
        ) from error


def _read_pinned_source(
    path: Path,
    *,
    source: WpiSourcePin,
    accepted_locodes: frozenset[str],
) -> tuple[tuple[WorldPortRecord, ...], WorldPortIntersectionAudit]:
    if path.is_symlink():
        raise WorldPortRegistryError(f"WPI source must not be a symbolic link: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise WorldPortRegistryError(f"WPI source is not readable: {path}") from error
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise WorldPortRegistryError(f"WPI source is not a regular file: {path}")
        if source_stat.st_size != source.bytes:
            raise WorldPortRegistryError(
                f"WPI source size mismatch: expected {source.bytes}, found {source_stat.st_size}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as binary:
            for chunk in iter(lambda: binary.read(_READ_CHUNK_SIZE), b""):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != source.sha256:
                raise WorldPortRegistryError(
                    f"WPI source SHA-256 mismatch: expected {source.sha256}, found {actual_sha256}"
                )
            binary.seek(0)
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", errors="strict", newline="")
            try:
                reader = csv.reader(text, strict=True)
                try:
                    header = tuple(next(reader))
                except StopIteration as error:
                    raise WorldPortRegistryError("WPI source has no CSV header") from error
                if header != WPI_CSV_COLUMNS:
                    raise WorldPortRegistryError("WPI CSV header differs from the pinned contract")

                dispositions: Counter[str] = Counter()
                selected: dict[int, list[WorldPortRecord]] = defaultdict(list)
                seen_oids: set[int] = set()
                facility_locodes: dict[int, str] = {}
                for row_number, row in enumerate(reader, start=2):
                    dispositions["source"] += 1
                    if len(row) != len(WPI_CSV_COLUMNS):
                        raise WorldPortRegistryError(
                            f"WPI CSV row {row_number} has {len(row)} columns; "
                            f"expected {len(WPI_CSV_COLUMNS)}"
                        )
                    oid = _parse_integral_decimal(
                        row[_OID_INDEX], field="OID", row_number=row_number
                    )
                    if oid in seen_oids:
                        raise WorldPortRegistryError(
                            f"WPI CSV contains duplicate source OID {oid} at row {row_number}"
                        )
                    seen_oids.add(oid)
                    wpi_number = _parse_integral_decimal(
                        row[_WPI_NUMBER_INDEX], field="WPI number", row_number=row_number
                    )
                    if wpi_number <= 0:
                        raise WorldPortRegistryError(
                            f"WPI CSV row {row_number} has nonpositive WPI number"
                        )
                    locode, layout = _parse_locode(row[_LOCODE_INDEX])
                    if locode is None:
                        dispositions[layout] += 1
                        continue
                    if locode not in accepted_locodes:
                        dispositions["absent"] += 1
                        continue
                    dispositions["intersected"] += 1
                    dispositions[layout] += 1
                    previous_locode = facility_locodes.setdefault(wpi_number, locode)
                    if previous_locode != locode:
                        raise WorldPortRegistryError(
                            "intersected WPI number maps to multiple UN/LOCODE identities: "
                            f"{wpi_number} -> {previous_locode}, {locode}"
                        )
                    record = _source_record(
                        row,
                        row_number=row_number,
                        locode=locode,
                        wpi_number=wpi_number,
                    )
                    if record.source_oid != oid:
                        raise AssertionError("parsed WPI OID changed within one source row")
                    selected[wpi_number].append(record)
            except (UnicodeError, csv.Error) as error:
                raise WorldPortRegistryError("cannot parse pinned WPI CSV") from error
            finally:
                text.detach()
    finally:
        os.close(descriptor)

    if dispositions["source"] != source.records:
        raise WorldPortRegistryError(
            f"WPI source record count mismatch: expected {source.records}, "
            f"found {dispositions['source']}"
        )

    accepted: list[WorldPortRecord] = []
    duplicate_facility_groups = 0
    duplicate_facility_source_rows = 0
    for wpi_number in sorted(selected):
        records = selected[wpi_number]
        if len(records) > 1:
            duplicate_facility_groups += 1
            duplicate_facility_source_rows += len(records)
            continue
        accepted.append(records[0])
    accepted.sort(key=lambda row: (row.locode, row.world_port_index_number))
    locode_counts = Counter(row.locode for row in accepted)

    audit = WorldPortIntersectionAudit(
        source_rows=dispositions["source"],
        excluded_blank_locode_rows=dispositions["blank"],
        excluded_malformed_locode_rows=dispositions["malformed"],
        excluded_locode_absent_from_pinned_unlocode_rows=dispositions["absent"],
        intersected_source_rows=dispositions["intersected"],
        spaced_locode_rows=dispositions["spaced"],
        compact_locode_rows=dispositions["compact"],
        accepted_port_records=len(accepted),
        excluded_duplicate_wpi_number_groups=duplicate_facility_groups,
        excluded_duplicate_wpi_number_rows=duplicate_facility_source_rows,
        multiple_facility_locodes=sum(count > 1 for count in locode_counts.values()),
        distinct_country_codes=len({row.country_code for row in accepted}),
    )
    return tuple(accepted), audit


def _canonical_whitelist_jsonl(records: Sequence[WorldPortRecord]) -> bytes:
    keys = tuple((row.locode, row.world_port_index_number) for row in records)
    if keys != tuple(sorted(set(keys))):
        raise WorldPortRegistryError(
            "world-port records must be unique and sorted by LOCODE and WPI number"
        )
    wpi_numbers = tuple(row.world_port_index_number for row in records)
    if len(wpi_numbers) != len(set(wpi_numbers)):
        raise WorldPortRegistryError("world-port records contain duplicate WPI numbers")
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in records)


def load_pinned_world_port_records(
    path: Path,
    *,
    expected_sha256: str,
    expected_records: int,
) -> tuple[WorldPortRecord, ...]:
    """Load canonical world-port JSONL by exact byte pin and record count."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected world-port SHA-256 must be lowercase hexadecimal")
    if (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records < 0
    ):
        raise ValueError("expected world-port records must be a non-negative integer")
    if path.is_symlink():
        raise WorldPortRegistryError(f"world-port whitelist must not be a symbolic link: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise WorldPortRegistryError(f"world-port whitelist is not readable: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WorldPortRegistryError(f"world-port whitelist is not a regular file: {path}")
        digest = hashlib.sha256()
        records: list[WorldPortRecord] = []
        previous_key: tuple[str, int] | None = None
        seen_wpi_numbers: set[int] = set()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for line_number, encoded in enumerate(stream, start=1):
                digest.update(encoded)
                if encoded == b"\n" or not encoded.endswith(b"\n"):
                    raise WorldPortRegistryError(
                        f"world-port whitelist row {line_number} is blank or unterminated"
                    )
                try:
                    record = WorldPortRecord.model_validate_json(encoded, strict=True)
                except ValueError as error:
                    raise WorldPortRegistryError(
                        f"world-port whitelist row {line_number} is invalid: {error}"
                    ) from error
                if encoded != canonical_json_bytes(record.model_dump(mode="json")) + b"\n":
                    raise WorldPortRegistryError(
                        f"world-port whitelist row {line_number} is not canonical JSON"
                    )
                key = (record.locode, record.world_port_index_number)
                if previous_key is not None and key <= previous_key:
                    raise WorldPortRegistryError(
                        "world-port records are duplicated or not sorted by LOCODE and WPI number"
                    )
                if record.world_port_index_number in seen_wpi_numbers:
                    raise WorldPortRegistryError("world-port records contain a repeated WPI number")
                records.append(record)
                previous_key = key
                seen_wpi_numbers.add(record.world_port_index_number)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise WorldPortRegistryError(
                f"world-port whitelist SHA-256 mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
        if len(records) != expected_records:
            raise WorldPortRegistryError(
                f"world-port whitelist count mismatch: expected {expected_records}, "
                f"found {len(records)}"
            )
        return tuple(records)
    finally:
        os.close(descriptor)


def _read_unlocode_artifact(
    path: Path,
    *,
    pin: UnlocodeArtifactPin,
) -> frozenset[str]:
    if path.is_symlink():
        raise WorldPortRegistryError(f"UN/LOCODE artifact must not be a symbolic link: {path}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise WorldPortRegistryError(f"UN/LOCODE artifact is not readable: {path}") from error
    if size != pin.bytes:
        raise WorldPortRegistryError(
            f"UN/LOCODE artifact size mismatch: expected {pin.bytes}, found {size}"
        )
    try:
        locations = load_pinned_unlocode_locations(
            path,
            expected_sha256=pin.sha256,
            expected_records=pin.records,
        )
    except (OSError, ValueError) as error:
        raise WorldPortRegistryError("pinned UN/LOCODE artifact validation failed") from error
    locodes = frozenset(row.locode for row in locations)
    if len(locodes) != pin.records:
        raise WorldPortRegistryError("pinned UN/LOCODE artifact contains duplicate identities")
    return locodes


def compile_world_port_registry(
    *,
    source_csv: Path,
    source: WpiSourcePin,
    unlocode_locations_jsonl: Path,
    unlocode_artifact: UnlocodeArtifactPin,
    output_parent: Path,
    run_name: str = "nga-world-port-index-current-v1",
) -> CompiledWorldPortRegistry:
    """Compile and atomically publish one pinned WPI/UNLOCODE whitelist."""

    accepted_locodes = _read_unlocode_artifact(
        unlocode_locations_jsonl,
        pin=unlocode_artifact,
    )
    records, audit = _read_pinned_source(
        source_csv,
        source=source,
        accepted_locodes=accepted_locodes,
    )
    if not records:
        raise WorldPortRegistryError("WPI/UNLOCODE intersection is empty")
    whitelist_payload = _canonical_whitelist_jsonl(records)
    policy = WorldPortRegistryPolicy(
        intersection=_INTERSECTION_POLICY,
        country_identity=_COUNTRY_IDENTITY_POLICY,
        facility_deduplication=_FACILITY_DEDUPLICATION_POLICY,
    )
    transaction_body = {
        "schema_version": 1,
        "whitelist_schema_version": _WHITELIST_SCHEMA_VERSION,
        "source": source.model_dump(mode="json"),
        "unlocode_artifact": unlocode_artifact.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
        "whitelist_sha256": sha256_bytes(whitelist_payload),
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction_body))
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_name,
        transaction_sha256=transaction_sha256,
    )
    source_receipt = WorldPortRegistrySourceReceipt(
        authority=NGA_WPI_AUTHORITY,
        dataset=NGA_WPI_DATASET,
        source_url=source.source_url,
        official_page_url=NGA_WPI_PAGE_URL,
        field_reference_url=NGA_WPI_FIELD_REFERENCE_URL,
        source_bytes=source.bytes,
        source_sha256=source.sha256,
        source_records=source.records,
        parser_contract=_PARSER_CONTRACT,
        attribution=NGA_WPI_ATTRIBUTION,
    )
    artifact = WorldPortRegistryArtifactReceipt(
        role="canonical_port_whitelist_jsonl",
        path="port-whitelist.jsonl",
        bytes=len(whitelist_payload),
        sha256=sha256_bytes(whitelist_payload),
        records=len(records),
    )
    receipt_body = {
        "schema_version": 1,
        "source": source_receipt.model_dump(mode="json"),
        "unlocode_artifact": unlocode_artifact.model_dump(mode="json"),
        "policy": policy.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
        "artifacts": (artifact.model_dump(mode="json"),),
    }
    receipt = WorldPortRegistryReceipt(
        schema_version=1,
        source=source_receipt,
        unlocode_artifact=unlocode_artifact,
        policy=policy,
        audit=audit,
        artifacts=(artifact,),
        content_sha256=sha256_bytes(canonical_json_bytes(receipt_body)),
    )
    receipt_payload = canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"
    stage.publish_bytes("port-whitelist.jsonl", whitelist_payload)
    stage.publish_bytes("registry-receipt.json", receipt_payload)
    committed = stage.commit(
        expected_artifacts=("port-whitelist.jsonl", "registry-receipt.json"),
        metadata={
            "schema_version": 1,
            "source_sha256": source.sha256,
            "unlocode_sha256": unlocode_artifact.sha256,
            "whitelist_records": len(records),
            "port_records": audit.accepted_port_records,
            "registry_receipt_sha256": sha256_bytes(receipt_payload),
        },
    )
    return CompiledWorldPortRegistry(
        root=stage.final_root,
        receipt=receipt,
        commit_receipt=committed.receipt,
        created=committed.created,
    )
