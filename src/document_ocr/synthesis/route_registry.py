"""Compile a pinned UNECE UN/LOCODE release into immutable local route data.

The compiler deliberately accepts only the three CSV parts in the official
production ZIP layout.  It never extracts archive members to the filesystem.
The canonical JSONL is the provenance-bearing source of truth; SQLite is a
derived, indexed read model published in the same atomic artifact bundle.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import sqlite3
import stat
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Buffer, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol, cast
from urllib.parse import parse_qs, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.routes import RouteLocation
from document_ocr.synthesis.run_safety import (
    StagedArtifactRun,
    StagedCommitReceipt,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Release = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[12]$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
LocationCode = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9]{3}$")]
Locode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}[A-Z0-9]{3}$")]
StatusCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
FunctionCode = Annotated[str, StringConstraints(pattern=r"^[0-7B]$")]
HttpsUrl = Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]

UNLOCODE_AUTHORITY: Final[Literal["United Nations Economic Commission for Europe (UNECE)"]] = (
    "United Nations Economic Commission for Europe (UNECE)"
)
UNLOCODE_DATASET: Final[
    Literal["United Nations Code for Trade and Transport Locations (UN/LOCODE)"]
] = "United Nations Code for Trade and Transport Locations (UN/LOCODE)"
UNLOCODE_LICENSE_NAME: Final[
    Literal["Creative Commons Attribution 4.0 International (CC BY 4.0)"]
] = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
UNLOCODE_LICENSE_URL: Final[Literal["https://creativecommons.org/licenses/by/4.0/"]] = (
    "https://creativecommons.org/licenses/by/4.0/"
)
UNLOCODE_TERMS_URL: Final[Literal["https://unlocode.unece.org/terms/"]] = (
    "https://unlocode.unece.org/terms/"
)
UNLOCODE_ATTRIBUTION: Final[
    Literal[
        "United Nations Economic Commission for Europe (UNECE), United Nations Code for Trade "
        "and Transport Locations (UN/LOCODE)"
    ]
] = (
    "United Nations Economic Commission for Europe (UNECE), United Nations Code for Trade "
    "and Transport Locations (UN/LOCODE)"
)

DEFAULT_ACCEPTED_STATUSES = ("AA", "AC", "AF", "AI", "AM", "AS", "RL")
DEFAULT_ACCEPTED_FUNCTIONS = ("1", "2", "3", "4", "5", "6", "7", "B")

_CSV_MEMBERS = (
    "release/csv/UNLOCODE CodeListPart1.csv",
    "release/csv/UNLOCODE CodeListPart2.csv",
    "release/csv/UNLOCODE CodeListPart3.csv",
)
_CSV_COLUMNS = 12
_READ_CHUNK_SIZE = 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 256
_MAX_MEMBER_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_REQUIRED_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 100
_PARSER_CONTRACT: Final[Literal["unece_unlocode_production_csv_parts_v1"]] = (
    "unece_unlocode_production_csv_parts_v1"
)
_LOCATION_SCHEMA_VERSION = 1
_SQLITE_SCHEMA_VERSION = 1
_SQLITE_APPLICATION_ID = 0x554E4C43  # ASCII "UNLC"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROUTE_SUBDIVISION_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,7}$")
_FUNCTION_POSITIONS = (
    frozenset({"-", "0", "1"}),
    frozenset({"-", "2"}),
    frozenset({"-", "3"}),
    frozenset({"-", "4"}),
    frozenset({"-", "5"}),
    frozenset({"-", "6"}),
    frozenset({"-", "7"}),
    frozenset({"-", "B"}),
)
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class UnlocodeRegistryError(RuntimeError):
    """A pinned UN/LOCODE release cannot be compiled without ambiguity."""


class UnlocodeArchiveMemberPin(BaseModel):
    """Exact decompressed identity of one required production CSV member."""

    model_config = _STRICT

    path: NonEmptyText
    bytes: Annotated[int, Field(gt=0, le=_MAX_MEMBER_UNCOMPRESSED_BYTES)]
    sha256: Sha256


class UnlocodeSourcePin(BaseModel):
    """Immutable identity asserted for one official production ZIP."""

    model_config = _STRICT

    release: Release
    source_url: HttpsUrl
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    csv_members: tuple[UnlocodeArchiveMemberPin, ...] = Field(
        min_length=len(_CSV_MEMBERS), max_length=len(_CSV_MEMBERS)
    )

    @model_validator(mode="after")
    def source_is_official_and_members_are_exact(self) -> UnlocodeSourcePin:
        parsed = urlsplit(self.source_url)
        query = parse_qs(parsed.query, strict_parsing=True)
        expected_path = f"/un/unece/uncefact/vocab-locode/-/jobs/artifacts/{self.release}/download"
        if (
            parsed.scheme != "https"
            or parsed.netloc != "opensource.unicc.org"
            or parsed.path != expected_path
            or query != {"job": ["package-release"]}
            or parsed.fragment
        ):
            raise ValueError(
                "source_url must identify the official UNICC UN/LOCODE package-release "
                f"artifact for release {self.release}"
            )
        if tuple(row.path for row in self.csv_members) != _CSV_MEMBERS:
            raise ValueError("source member pins must name the exact production CSV parts")
        if sum(row.bytes for row in self.csv_members) > _MAX_TOTAL_REQUIRED_UNCOMPRESSED_BYTES:
            raise ValueError("source member pins exceed the bounded decompression contract")
        return self


class UnlocodeRegistryPolicy(BaseModel):
    """Explicit row-eligibility policy for a normalized registry."""

    model_config = _STRICT

    accepted_statuses: tuple[StatusCode, ...] = DEFAULT_ACCEPTED_STATUSES
    accepted_function_codes: tuple[FunctionCode, ...] = DEFAULT_ACCEPTED_FUNCTIONS

    @model_validator(mode="after")
    def values_are_nonempty_unique_and_sorted(self) -> UnlocodeRegistryPolicy:
        if not self.accepted_statuses or not self.accepted_function_codes:
            raise ValueError("accepted statuses and function codes must not be empty")
        if self.accepted_statuses != tuple(sorted(set(self.accepted_statuses))):
            raise ValueError("accepted statuses must be unique and sorted")
        if self.accepted_function_codes != tuple(sorted(set(self.accepted_function_codes))):
            raise ValueError("accepted function codes must be unique and sorted")
        return self


class UnlocodeLocation(BaseModel):
    """One unique accepted location, preserving the published route attributes."""

    model_config = _STRICT

    schema_version: Literal[1] = 1
    locode: Locode
    country_code: CountryCode
    location_code: LocationCode
    name: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    name_without_diacritics: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    subdivision_code: Annotated[str, StringConstraints(min_length=1, max_length=16)] | None
    function_codes: tuple[FunctionCode, ...] = Field(min_length=1)
    status: StatusCode
    coordinates: Annotated[str, StringConstraints(min_length=1, max_length=32)] | None

    @field_validator(
        "name",
        "name_without_diacritics",
        "subdivision_code",
        "coordinates",
    )
    @classmethod
    def text_is_exact_and_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip():
            raise ValueError("UN/LOCODE text must not contain outer whitespace")
        if any(unicodedata.category(character) == "Cc" for character in value):
            raise ValueError("UN/LOCODE text must not contain control characters")
        return value

    @model_validator(mode="after")
    def identity_and_functions_are_canonical(self) -> UnlocodeLocation:
        if self.locode != self.country_code + self.location_code:
            raise ValueError("UN/LOCODE identity differs from country and location components")
        if self.function_codes != tuple(sorted(set(self.function_codes))):
            raise ValueError("UN/LOCODE function codes must be unique and sorted")
        return self


class UnlocodeNormalizationAudit(BaseModel):
    """Mutually exclusive disposition for every source CSV row."""

    model_config = _STRICT

    source_rows: Annotated[int, Field(ge=0)]
    accepted_rows: Annotated[int, Field(ge=0)]
    excluded_deletion_rows: Annotated[int, Field(ge=0)]
    excluded_blank_locode_rows: Annotated[int, Field(ge=0)]
    excluded_malformed_locode_rows: Annotated[int, Field(ge=0)]
    excluded_status_rows: Annotated[int, Field(ge=0)]
    excluded_function_rows: Annotated[int, Field(ge=0)]
    excluded_duplicate_locodes: Annotated[int, Field(ge=0)]
    excluded_duplicate_locode_rows: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def row_accounting_balances(self) -> UnlocodeNormalizationAudit:
        classified = (
            self.accepted_rows
            + self.excluded_deletion_rows
            + self.excluded_blank_locode_rows
            + self.excluded_malformed_locode_rows
            + self.excluded_status_rows
            + self.excluded_function_rows
            + self.excluded_duplicate_locode_rows
        )
        if classified != self.source_rows:
            raise ValueError("UN/LOCODE normalization audit does not balance")
        if (self.excluded_duplicate_locodes == 0) != (self.excluded_duplicate_locode_rows == 0) or (
            self.excluded_duplicate_locode_rows < 2 * self.excluded_duplicate_locodes
        ):
            raise ValueError("UN/LOCODE duplicate-location accounting is inconsistent")
        return self


class UnlocodeArchiveMemberReceipt(BaseModel):
    model_config = _STRICT

    path: NonEmptyText
    compressed_bytes: Annotated[int, Field(ge=0)]
    uncompressed_bytes: Annotated[int, Field(ge=0)]
    crc32: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{8}$")]
    sha256: Sha256


class UnlocodeSourceReceipt(BaseModel):
    model_config = _STRICT

    authority: Literal["United Nations Economic Commission for Europe (UNECE)"]
    dataset: Literal["United Nations Code for Trade and Transport Locations (UN/LOCODE)"]
    release: Release
    source_url: HttpsUrl
    source_bytes: Annotated[int, Field(gt=0)]
    source_sha256: Sha256
    parser_contract: Literal["unece_unlocode_production_csv_parts_v1"]
    csv_members: tuple[UnlocodeArchiveMemberReceipt, ...] = Field(
        min_length=len(_CSV_MEMBERS), max_length=len(_CSV_MEMBERS)
    )
    license_name: Literal["Creative Commons Attribution 4.0 International (CC BY 4.0)"]
    license_url: Literal["https://creativecommons.org/licenses/by/4.0/"]
    terms_url: Literal["https://unlocode.unece.org/terms/"]
    attribution: Literal[
        "United Nations Economic Commission for Europe (UNECE), United Nations Code for Trade "
        "and Transport Locations (UN/LOCODE)"
    ]

    @model_validator(mode="after")
    def members_are_exact_and_ordered(self) -> UnlocodeSourceReceipt:
        paths = tuple(row.path for row in self.csv_members)
        if paths != _CSV_MEMBERS:
            raise ValueError("UN/LOCODE receipt does not name the exact production CSV parts")
        return self


class RegistryArtifactReceipt(BaseModel):
    model_config = _STRICT

    role: Literal["canonical_locations_jsonl", "queryable_sqlite"]
    path: Literal["locations.jsonl", "registry.sqlite3"]
    bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256
    records: Annotated[int, Field(ge=0)]


class UnlocodeRegistryReceipt(BaseModel):
    """Source, policy, audit, and output proof for one compiled registry."""

    model_config = _STRICT

    schema_version: Literal[1]
    source: UnlocodeSourceReceipt
    policy: UnlocodeRegistryPolicy
    audit: UnlocodeNormalizationAudit
    artifacts: tuple[RegistryArtifactReceipt, ...] = Field(min_length=2, max_length=2)
    content_sha256: Sha256

    @model_validator(mode="after")
    def receipt_is_canonical_and_self_hashed(self) -> UnlocodeRegistryReceipt:
        roles = tuple(row.role for row in self.artifacts)
        if roles != ("canonical_locations_jsonl", "queryable_sqlite"):
            raise ValueError("UN/LOCODE registry artifact roles are not canonical")
        paths = tuple(row.path for row in self.artifacts)
        if paths != ("locations.jsonl", "registry.sqlite3"):
            raise ValueError("UN/LOCODE registry artifact paths are not canonical")
        if any(row.records != self.audit.accepted_rows for row in self.artifacts):
            raise ValueError("UN/LOCODE artifact counts differ from the normalization audit")
        body = self.model_dump(mode="json", exclude={"content_sha256"})
        if sha256_bytes(canonical_json_bytes(body)) != self.content_sha256:
            raise ValueError("UN/LOCODE registry receipt content SHA-256 is invalid")
        return self


class RouteLocationProjectionAudit(BaseModel):
    """Explicit accounting for the narrower legacy route-location projection."""

    model_config = _STRICT

    source_records: Annotated[int, Field(ge=0)]
    projected_records: Annotated[int, Field(ge=0)]
    removed_unverified_function_code_records: Annotated[int, Field(ge=0)]
    normalized_placeholder_subdivision_records: Annotated[int, Field(ge=0)]
    omitted_nonconforming_subdivision_records: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def records_balance(self) -> RouteLocationProjectionAudit:
        if self.source_records != self.projected_records:
            raise ValueError("route-location projection record counts do not balance")
        for count in (
            self.removed_unverified_function_code_records,
            self.normalized_placeholder_subdivision_records,
            self.omitted_nonconforming_subdivision_records,
        ):
            if count > self.source_records:
                raise ValueError("route-location projection adjustment count is impossible")
        return self


class RouteLocationProjection(BaseModel):
    """Existing route-sampler records plus an audit of its narrower fields."""

    model_config = _STRICT

    locations: tuple[RouteLocation, ...] = Field(min_length=1)
    audit: RouteLocationProjectionAudit


@dataclass(frozen=True, slots=True)
class CompiledUnlocodeRegistry:
    root: Path
    receipt: UnlocodeRegistryReceipt
    commit_receipt: StagedCommitReceipt
    created: bool


class _Digest(Protocol):
    def update(self, payload: bytes | memoryview) -> None: ...


class _ReadInto(Protocol):
    def readinto(self, buffer: Buffer, /) -> int | None: ...


class _DigestingReader(io.RawIOBase):
    """Hash decompressed ZIP bytes as the CSV parser consumes them."""

    def __init__(self, stream: _ReadInto, digest: _Digest) -> None:
        super().__init__()
        self._stream = stream
        self._digest = digest
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        count = self._stream.readinto(buffer)
        if count is None:
            return 0
        if count:
            self._digest.update(memoryview(buffer)[:count])
            self.bytes_read += count
        return count


def _parse_function_codes(value: str, *, member: str, row_number: int) -> tuple[str, ...]:
    if len(value) != len(_FUNCTION_POSITIONS) or any(
        character not in allowed
        for character, allowed in zip(value, _FUNCTION_POSITIONS, strict=True)
    ):
        raise UnlocodeRegistryError(
            f"{member}:{row_number}: invalid 8-position Function value: {value!r}"
        )
    return tuple(sorted(character for character in value if character != "-"))


def _row_location(row: Sequence[str], *, member: str, row_number: int) -> UnlocodeLocation:
    country_code, location_code = row[1], row[2]
    try:
        return UnlocodeLocation.model_validate(
            {
                "schema_version": 1,
                "locode": country_code + location_code,
                "country_code": country_code,
                "location_code": location_code,
                "name": row[3],
                "name_without_diacritics": row[4],
                "subdivision_code": row[5] or None,
                "function_codes": _parse_function_codes(
                    row[6], member=member, row_number=row_number
                ),
                "status": row[7],
                "coordinates": row[10] or None,
            },
            strict=True,
        )
    except ValueError as error:
        raise UnlocodeRegistryError(
            f"{member}:{row_number}: invalid accepted UN/LOCODE row: {error}"
        ) from error


def _classify_rows(
    archive: zipfile.ZipFile,
    *,
    policy: UnlocodeRegistryPolicy,
    member_pins: Sequence[UnlocodeArchiveMemberPin],
) -> tuple[
    tuple[UnlocodeLocation, ...],
    UnlocodeNormalizationAudit,
    tuple[UnlocodeArchiveMemberReceipt, ...],
]:
    if len(archive.infolist()) > _MAX_ARCHIVE_MEMBERS:
        raise UnlocodeRegistryError(
            f"UN/LOCODE archive has too many members: {len(archive.infolist())}"
        )
    infos_by_name: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive.infolist():
        infos_by_name[info.filename].append(info)
    missing = [name for name in _CSV_MEMBERS if not infos_by_name[name]]
    repeated = [name for name in _CSV_MEMBERS if len(infos_by_name[name]) > 1]
    if missing or repeated:
        raise UnlocodeRegistryError(
            f"production CSV member inventory is invalid; missing={missing}, repeated={repeated}"
        )

    pins_by_name = {row.path: row for row in member_pins}
    member_receipts: list[UnlocodeArchiveMemberReceipt] = []
    dispositions: Counter[str] = Counter()
    candidates: dict[str, UnlocodeLocation] = {}
    duplicate_locodes: Counter[str] = Counter()
    accepted_statuses = frozenset(policy.accepted_statuses)
    accepted_functions = frozenset(policy.accepted_function_codes)

    for member in _CSV_MEMBERS:
        info = infos_by_name[member][0]
        unix_mode = info.external_attr >> 16 if info.create_system == 3 else 0
        if (
            info.is_dir()
            or info.flag_bits & 0x1
            or (unix_mode != 0 and not stat.S_ISREG(unix_mode))
        ):
            raise UnlocodeRegistryError(f"production CSV member is not a plain file: {member}")
        member_pin = pins_by_name[member]
        if info.file_size != member_pin.bytes:
            raise UnlocodeRegistryError(
                f"production CSV member size mismatch for {member}: "
                f"expected {member_pin.bytes}, found {info.file_size}"
            )
        if info.file_size > _MAX_MEMBER_UNCOMPRESSED_BYTES or (
            info.file_size / max(info.compress_size, 1) > _MAX_COMPRESSION_RATIO
        ):
            raise UnlocodeRegistryError(
                f"production CSV member exceeds decompression bounds: {member}"
            )
        member_digest = hashlib.sha256()
        digesting: _DigestingReader | None = None
        try:
            with (
                archive.open(info, mode="r") as encoded,
                _DigestingReader(cast(_ReadInto, encoded), member_digest) as digesting,
                io.BufferedReader(digesting, buffer_size=_READ_CHUNK_SIZE) as buffered,
                io.TextIOWrapper(
                    buffered, encoding="utf-8-sig", errors="strict", newline=""
                ) as text,
            ):
                reader = csv.reader(text, strict=True)
                for row_number, row in enumerate(reader, start=1):
                    dispositions["source"] += 1
                    if len(row) != _CSV_COLUMNS:
                        raise UnlocodeRegistryError(
                            f"{member}:{row_number}: expected {_CSV_COLUMNS} CSV fields, "
                            f"found {len(row)}"
                        )
                    change, country_code, location_code = row[0], row[1], row[2]
                    if change == "X":
                        dispositions["deletion"] += 1
                        continue
                    if not location_code:
                        dispositions["blank_locode"] += 1
                        continue
                    if (
                        re.fullmatch(r"[A-Z]{2}", country_code) is None
                        or re.fullmatch(r"[A-Z0-9]{3}", location_code) is None
                    ):
                        dispositions["malformed_locode"] += 1
                        continue
                    if row[7] not in accepted_statuses:
                        dispositions["status"] += 1
                        continue
                    functions = _parse_function_codes(row[6], member=member, row_number=row_number)
                    if not accepted_functions.intersection(functions):
                        dispositions["function"] += 1
                        continue
                    location = _row_location(row, member=member, row_number=row_number)
                    if location.locode in duplicate_locodes:
                        duplicate_locodes[location.locode] += 1
                    elif location.locode in candidates:
                        del candidates[location.locode]
                        duplicate_locodes[location.locode] = 2
                    else:
                        candidates[location.locode] = location
        except (UnicodeError, csv.Error, zipfile.BadZipFile) as error:
            raise UnlocodeRegistryError(f"cannot parse production CSV member: {member}") from error
        if digesting is None or digesting.bytes_read != member_pin.bytes:
            raise UnlocodeRegistryError(
                f"production CSV member decompressed size changed while reading: {member}"
            )
        actual_member_sha256 = member_digest.hexdigest()
        if actual_member_sha256 != member_pin.sha256:
            raise UnlocodeRegistryError(
                f"production CSV member SHA-256 mismatch for {member}: "
                f"expected {member_pin.sha256}, found {actual_member_sha256}"
            )
        member_receipts.append(
            UnlocodeArchiveMemberReceipt(
                path=member,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
                crc32=f"{info.CRC:08x}",
                sha256=actual_member_sha256,
            )
        )

    # The published directory contains controlled duplicate name listings.
    # Choosing one as the route label would be an undocumented semantic
    # preference, so every ambiguous row is excluded and explicitly counted.
    dispositions["duplicate_locode"] = sum(duplicate_locodes.values())
    dispositions["duplicate_locode_groups"] = len(duplicate_locodes)
    accepted = [candidates[locode] for locode in sorted(candidates)]
    dispositions["accepted"] = len(accepted)

    audit = UnlocodeNormalizationAudit(
        source_rows=dispositions["source"],
        accepted_rows=dispositions["accepted"],
        excluded_deletion_rows=dispositions["deletion"],
        excluded_blank_locode_rows=dispositions["blank_locode"],
        excluded_malformed_locode_rows=dispositions["malformed_locode"],
        excluded_status_rows=dispositions["status"],
        excluded_function_rows=dispositions["function"],
        excluded_duplicate_locodes=dispositions["duplicate_locode_groups"],
        excluded_duplicate_locode_rows=dispositions["duplicate_locode"],
    )
    return tuple(accepted), audit, tuple(member_receipts)


def _read_pinned_release(
    path: Path,
    *,
    source: UnlocodeSourcePin,
    policy: UnlocodeRegistryPolicy,
) -> tuple[
    tuple[UnlocodeLocation, ...],
    UnlocodeNormalizationAudit,
    tuple[UnlocodeArchiveMemberReceipt, ...],
]:
    if path.is_symlink():
        raise UnlocodeRegistryError(f"UN/LOCODE source must not be a symbolic link: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UnlocodeRegistryError(f"UN/LOCODE source is not readable: {path}") from error
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise UnlocodeRegistryError(f"UN/LOCODE source is not a regular file: {path}")
        if source_stat.st_size != source.bytes:
            raise UnlocodeRegistryError(
                f"UN/LOCODE source size mismatch: expected {source.bytes}, "
                f"found {source_stat.st_size}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(_READ_CHUNK_SIZE), b""):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != source.sha256:
                raise UnlocodeRegistryError(
                    f"UN/LOCODE source SHA-256 mismatch: expected {source.sha256}, "
                    f"found {actual_sha256}"
                )
            stream.seek(0)
            try:
                with zipfile.ZipFile(stream, mode="r") as archive:
                    return _classify_rows(
                        archive,
                        policy=policy,
                        member_pins=source.csv_members,
                    )
            except zipfile.BadZipFile as error:
                raise UnlocodeRegistryError("UN/LOCODE source is not a valid ZIP") from error
    finally:
        os.close(descriptor)


def _canonical_location_jsonl(locations: Sequence[UnlocodeLocation]) -> bytes:
    locodes = tuple(row.locode for row in locations)
    if locodes != tuple(sorted(set(locodes))):
        raise UnlocodeRegistryError("normalized UN/LOCODE rows must be unique and sorted")
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in locations)


def load_pinned_unlocode_locations(
    path: Path,
    *,
    expected_sha256: str,
    expected_records: int,
) -> tuple[UnlocodeLocation, ...]:
    """Strictly load a compiled canonical locations JSONL by byte pin and count."""

    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected UN/LOCODE locations SHA-256 must be lowercase hexadecimal")
    if (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records < 0
    ):
        raise ValueError("expected UN/LOCODE location records must be a non-negative integer")
    if path.is_symlink():
        raise UnlocodeRegistryError(
            f"compiled UN/LOCODE locations must not be a symbolic link: {path}"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UnlocodeRegistryError(
            f"compiled UN/LOCODE locations are not readable: {path}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnlocodeRegistryError(
                f"compiled UN/LOCODE locations are not a regular file: {path}"
            )
        digest = hashlib.sha256()
        locations: list[UnlocodeLocation] = []
        previous_locode: str | None = None
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for line_number, encoded in enumerate(stream, start=1):
                digest.update(encoded)
                if encoded == b"\n" or not encoded.endswith(b"\n"):
                    raise UnlocodeRegistryError(
                        f"compiled UN/LOCODE locations row {line_number} is blank or unterminated"
                    )
                try:
                    location = UnlocodeLocation.model_validate_json(encoded, strict=True)
                except ValueError as error:
                    raise UnlocodeRegistryError(
                        f"compiled UN/LOCODE locations row {line_number} is invalid: {error}"
                    ) from error
                canonical = canonical_json_bytes(location.model_dump(mode="json")) + b"\n"
                if encoded != canonical:
                    raise UnlocodeRegistryError(
                        f"compiled UN/LOCODE locations row {line_number} is not canonical JSON"
                    )
                if previous_locode is not None and location.locode <= previous_locode:
                    raise UnlocodeRegistryError(
                        "compiled UN/LOCODE locations are duplicated or not sorted by LOCODE"
                    )
                locations.append(location)
                previous_locode = location.locode
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise UnlocodeRegistryError(
                f"compiled UN/LOCODE locations SHA-256 mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
        if len(locations) != expected_records:
            raise UnlocodeRegistryError(
                f"compiled UN/LOCODE location count mismatch: expected {expected_records}, "
                f"found {len(locations)}"
            )
        return tuple(locations)
    finally:
        os.close(descriptor)


def project_route_locations(
    locations: Sequence[UnlocodeLocation],
) -> RouteLocationProjection:
    """Project registry rows to the existing route-sampler contract.

    The canonical registry remains lossless. This narrower projection removes
    the special unverified function marker ``0`` and records any published
    subdivision value that cannot fit the route sampler's ISO-like field.
    """

    locodes = tuple(row.locode for row in locations)
    if locodes != tuple(sorted(set(locodes))):
        raise UnlocodeRegistryError(
            "UN/LOCODE rows must be unique and sorted before route projection"
        )
    projected: list[RouteLocation] = []
    removed_unverified = 0
    normalized_placeholder = 0
    omitted_nonconforming = 0
    for location in locations:
        function_codes = tuple(code for code in location.function_codes if code != "0")
        if not function_codes:
            raise UnlocodeRegistryError(
                f"UN/LOCODE {location.locode} has no route-sampler function after removing 0"
            )
        if "0" in location.function_codes:
            removed_unverified += 1
        subdivision = location.subdivision_code
        if subdivision == "-":
            subdivision = None
            normalized_placeholder += 1
        elif subdivision is not None and _ROUTE_SUBDIVISION_PATTERN.fullmatch(subdivision) is None:
            subdivision = None
            omitted_nonconforming += 1
        projected.append(
            RouteLocation.model_validate(
                {
                    "locode": location.locode,
                    "country_code": location.country_code,
                    "name": location.name,
                    "subdivision_code": subdivision,
                    "function_codes": function_codes,
                    "status": location.status,
                },
                strict=True,
            )
        )
    audit = RouteLocationProjectionAudit(
        source_records=len(locations),
        projected_records=len(projected),
        removed_unverified_function_code_records=removed_unverified,
        normalized_placeholder_subdivision_records=normalized_placeholder,
        omitted_nonconforming_subdivision_records=omitted_nonconforming,
    )
    return RouteLocationProjection(locations=tuple(projected), audit=audit)


def _sqlite_bytes(
    locations: Sequence[UnlocodeLocation],
    *,
    source: UnlocodeSourcePin,
    policy: UnlocodeRegistryPolicy,
    temporary_parent: Path,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix=".unlocode-sqlite-", dir=temporary_parent) as raw:
        path = Path(raw) / "registry.sqlite3"
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA page_size = 4096")
            connection.execute(f"PRAGMA application_id = {_SQLITE_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SQLITE_SCHEMA_VERSION}")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.executescript(
                """
                CREATE TABLE registry_metadata (
                    key TEXT PRIMARY KEY NOT NULL,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE locations (
                    locode TEXT PRIMARY KEY NOT NULL CHECK (length(locode) = 5),
                    country_code TEXT NOT NULL CHECK (length(country_code) = 2),
                    location_code TEXT NOT NULL CHECK (length(location_code) = 3),
                    name TEXT NOT NULL CHECK (length(name) > 0),
                    name_without_diacritics TEXT NOT NULL
                        CHECK (length(name_without_diacritics) > 0),
                    subdivision_code TEXT,
                    function_codes_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (length(status) = 2),
                    coordinates TEXT,
                    CHECK (locode = country_code || location_code)
                ) WITHOUT ROWID;
                CREATE TABLE location_functions (
                    locode TEXT NOT NULL,
                    function_code TEXT NOT NULL CHECK (length(function_code) = 1),
                    PRIMARY KEY (locode, function_code),
                    FOREIGN KEY (locode) REFERENCES locations(locode)
                ) WITHOUT ROWID;
                CREATE INDEX locations_country_status_locode_idx
                    ON locations(country_code, status, locode);
                CREATE INDEX locations_country_name_locode_idx
                    ON locations(country_code, name_without_diacritics, locode);
                CREATE INDEX location_functions_code_locode_idx
                    ON location_functions(function_code, locode);
                """
            )
            metadata = {
                "accepted_function_codes": canonical_json_bytes(
                    policy.accepted_function_codes
                ).decode("utf-8"),
                "accepted_statuses": canonical_json_bytes(policy.accepted_statuses).decode("utf-8"),
                "location_records": str(len(locations)),
                "location_schema_version": str(_LOCATION_SCHEMA_VERSION),
                "parser_contract": _PARSER_CONTRACT,
                "source_release": source.release,
                "source_sha256": source.sha256,
            }
            connection.executemany(
                "INSERT INTO registry_metadata(key, value) VALUES (?, ?)",
                tuple(sorted(metadata.items())),
            )
            connection.executemany(
                """
                INSERT INTO locations(
                    locode, country_code, location_code, name,
                    name_without_diacritics, subdivision_code,
                    function_codes_json, status, coordinates
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        row.locode,
                        row.country_code,
                        row.location_code,
                        row.name,
                        row.name_without_diacritics,
                        row.subdivision_code,
                        canonical_json_bytes(row.function_codes).decode("utf-8"),
                        row.status,
                        row.coordinates,
                    )
                    for row in locations
                ),
            )
            connection.executemany(
                "INSERT INTO location_functions(locode, function_code) VALUES (?, ?)",
                (
                    (row.locode, function_code)
                    for row in locations
                    for function_code in row.function_codes
                ),
            )
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise UnlocodeRegistryError("generated UN/LOCODE SQLite failed integrity_check")
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise UnlocodeRegistryError(
                    f"generated UN/LOCODE SQLite has foreign-key errors: {foreign_key_errors}"
                )
        finally:
            connection.close()
        sidecars = tuple(path.parent.glob("registry.sqlite3-*"))
        if sidecars:
            raise UnlocodeRegistryError(
                f"generated UN/LOCODE SQLite left sidecar files: {sidecars}"
            )
        return path.read_bytes()


def _artifact_receipt(
    *,
    role: Literal["canonical_locations_jsonl", "queryable_sqlite"],
    path: Literal["locations.jsonl", "registry.sqlite3"],
    payload: bytes,
    records: int,
) -> RegistryArtifactReceipt:
    return RegistryArtifactReceipt(
        role=role,
        path=path,
        bytes=len(payload),
        sha256=sha256_bytes(payload),
        records=records,
    )


def compile_unlocode_registry(
    *,
    source_zip: Path,
    source: UnlocodeSourcePin,
    output_parent: Path,
    policy: UnlocodeRegistryPolicy | None = None,
    run_name: str | None = None,
) -> CompiledUnlocodeRegistry:
    """Compile and atomically publish one pinned official UN/LOCODE release."""

    effective_policy = policy or UnlocodeRegistryPolicy()
    locations, audit, csv_members = _read_pinned_release(
        source_zip, source=source, policy=effective_policy
    )
    if not locations:
        raise UnlocodeRegistryError("UN/LOCODE policy accepted no unique locations")
    jsonl_payload = _canonical_location_jsonl(locations)
    transaction_body = {
        "schema_version": 1,
        "location_schema_version": _LOCATION_SCHEMA_VERSION,
        "sqlite_schema_version": _SQLITE_SCHEMA_VERSION,
        "source": source.model_dump(mode="json"),
        "policy": effective_policy.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
        "canonical_locations_sha256": sha256_bytes(jsonl_payload),
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction_body))
    resolved_run_name = run_name or f"unlocode-{source.release}"
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=resolved_run_name,
        transaction_sha256=transaction_sha256,
    )
    sqlite_payload = _sqlite_bytes(
        locations,
        source=source,
        policy=effective_policy,
        temporary_parent=stage.output_parent,
    )
    source_receipt = UnlocodeSourceReceipt(
        authority=UNLOCODE_AUTHORITY,
        dataset=UNLOCODE_DATASET,
        release=source.release,
        source_url=source.source_url,
        source_bytes=source.bytes,
        source_sha256=source.sha256,
        parser_contract=_PARSER_CONTRACT,
        csv_members=csv_members,
        license_name=UNLOCODE_LICENSE_NAME,
        license_url=UNLOCODE_LICENSE_URL,
        terms_url=UNLOCODE_TERMS_URL,
        attribution=UNLOCODE_ATTRIBUTION,
    )
    artifacts = (
        _artifact_receipt(
            role="canonical_locations_jsonl",
            path="locations.jsonl",
            payload=jsonl_payload,
            records=len(locations),
        ),
        _artifact_receipt(
            role="queryable_sqlite",
            path="registry.sqlite3",
            payload=sqlite_payload,
            records=len(locations),
        ),
    )
    receipt_body = {
        "schema_version": 1,
        "source": source_receipt.model_dump(mode="json"),
        "policy": effective_policy.model_dump(mode="json"),
        "audit": audit.model_dump(mode="json"),
        "artifacts": tuple(row.model_dump(mode="json") for row in artifacts),
    }
    receipt = UnlocodeRegistryReceipt(
        schema_version=1,
        source=source_receipt,
        policy=effective_policy,
        audit=audit,
        artifacts=artifacts,
        content_sha256=sha256_bytes(canonical_json_bytes(receipt_body)),
    )
    receipt_payload = canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"
    stage.publish_bytes("locations.jsonl", jsonl_payload)
    stage.publish_bytes("registry.sqlite3", sqlite_payload)
    stage.publish_bytes("registry-receipt.json", receipt_payload)
    committed = stage.commit(
        expected_artifacts=(
            "locations.jsonl",
            "registry-receipt.json",
            "registry.sqlite3",
        ),
        metadata={
            "schema_version": 1,
            "release": source.release,
            "location_records": len(locations),
            "registry_receipt_sha256": sha256_bytes(receipt_payload),
        },
    )
    return CompiledUnlocodeRegistry(
        root=stage.final_root,
        receipt=receipt,
        commit_receipt=committed.receipt,
        created=committed.created,
    )
