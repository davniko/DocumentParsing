"""Compile a pinned GeoNames cities15000 archive into an ISO locality registry.

GeoNames is a provider dataset, not an identity authority. Every source row is
parsed against the documented 19-column contract, but only rows whose country
code exists in the separately pinned ISO-3166 snapshot are compiled. Provider
codes outside ISO are named and balanced in the receipt; they are never aliased
or mapped by hand.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import stat
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import ArtifactReadError, atomic_publish_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
FeatureCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9]{0,9}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

GEONAMES_SOURCE_URL = "https://download.geonames.org/export/dump/cities15000.zip"
GEONAMES_LICENSE = "Creative Commons Attribution 4.0"
GEONAMES_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
GEONAMES_ARCHIVE_PATH = "raw/cities15000.zip"
GEONAMES_MEMBER_PATH = "cities15000.txt"
PINNED_ISO3166_PATH = Path("/usr/share/iso-codes/json/iso_3166-1.json")

_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_RECEIPT_BYTES = 1024 * 1024
_MAX_ISO_BYTES = 1024 * 1024
_MAX_JSONL_BYTES = 32 * 1024 * 1024
_MAX_SQLITE_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024
_GEONAMES_COLUMNS = 19
_SQLITE_APPLICATION_ID = 0x474E4C43  # GNLC
_SUPPORTED_COMPRESSION = {
    zipfile.ZIP_STORED: "stored",
    zipfile.ZIP_DEFLATED: "deflate",
}


class GeoNamesLocalityError(RuntimeError):
    """A GeoNames acquisition or compilation contract failed."""


class GeoNamesZipMemberPin(BaseModel):
    model_config = _STRICT

    path: Literal["cities15000.txt"]
    compression: Literal["stored", "deflate"]
    compressed_bytes: Annotated[int, Field(gt=0, le=_MAX_MEMBER_BYTES)]
    uncompressed_bytes: Annotated[int, Field(gt=0, le=_MAX_MEMBER_BYTES)]
    crc32: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{8}$")]
    sha256: Sha256


class GeoNamesSourceReceipt(BaseModel):
    """Immutable acquisition identity for one official GeoNames ZIP."""

    model_config = _STRICT

    schema_version: Literal[1]
    provider: Literal["GeoNames"]
    dataset: Literal["cities15000"]
    source_url: Literal["https://download.geonames.org/export/dump/cities15000.zip"]
    license: Literal["Creative Commons Attribution 4.0"]
    license_url: Literal["https://creativecommons.org/licenses/by/4.0/"]
    archive_path: Literal["raw/cities15000.zip"]
    archive_bytes: Annotated[int, Field(gt=0, le=_MAX_ARCHIVE_BYTES)]
    archive_sha256: Sha256
    members: tuple[GeoNamesZipMemberPin, ...] = Field(min_length=1, max_length=1)
    content_sha256: Sha256

    @model_validator(mode="after")
    def content_hash_is_valid(self) -> GeoNamesSourceReceipt:
        if self.members[0].path != GEONAMES_MEMBER_PATH:
            raise ValueError("GeoNames source receipt has an unexpected member")
        body = self.model_dump(mode="json", exclude={"content_sha256"})
        if sha256_bytes(canonical_json_bytes(body)) != self.content_sha256:
            raise ValueError("GeoNames source receipt content SHA-256 is invalid")
        return self


class LocalityRecord(BaseModel):
    """One canonical GeoNames populated-place identity."""

    model_config = _STRICT

    geoname_id: Annotated[int, Field(gt=0)]
    canonical_name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    ascii_name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    country_code: CountryCode
    feature_code: FeatureCode
    population: Annotated[int, Field(ge=0)]
    latitude: Decimal
    longitude: Decimal
    timezone: Annotated[str, StringConstraints(min_length=1, max_length=40)]

    @model_validator(mode="after")
    def values_match_geonames_contract(self) -> LocalityRecord:
        for field in ("canonical_name", "ascii_name", "timezone"):
            value = getattr(self, field)
            if value != value.strip() or any(ord(character) < 32 for character in value):
                raise ValueError(f"GeoNames locality {field} contains unsafe whitespace/control")
        if not self.ascii_name.isascii():
            raise ValueError("GeoNames ascii_name contains non-ASCII characters")
        if not self.latitude.is_finite() or not Decimal("-90") <= self.latitude <= Decimal("90"):
            raise ValueError("GeoNames latitude is outside WGS84 bounds")
        if not self.longitude.is_finite() or not Decimal("-180") <= self.longitude <= Decimal(
            "180"
        ):
            raise ValueError("GeoNames longitude is outside WGS84 bounds")
        return self


class GeoNamesLocalityAudit(BaseModel):
    model_config = _STRICT

    source_records: Annotated[int, Field(gt=0)]
    accepted_records: Annotated[int, Field(gt=0)]
    excluded_non_iso_country_records: Annotated[int, Field(ge=0)]
    excluded_non_iso_country_counts: Mapping[CountryCode, Annotated[int, Field(gt=0)]]
    distinct_countries: Annotated[int, Field(gt=0)]
    country_code_counts: Mapping[CountryCode, Annotated[int, Field(gt=0)]]
    feature_code_counts: Mapping[FeatureCode, Annotated[int, Field(gt=0)]]
    zero_population_records: Annotated[int, Field(ge=0)]
    minimum_population: Annotated[int, Field(ge=0)]
    maximum_population: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def counts_balance(self) -> GeoNamesLocalityAudit:
        if self.source_records != (self.accepted_records + self.excluded_non_iso_country_records):
            raise ValueError("GeoNames source-row audit does not balance")
        if self.excluded_non_iso_country_records != sum(
            self.excluded_non_iso_country_counts.values()
        ):
            raise ValueError("GeoNames non-ISO exclusion audit does not balance")
        if self.accepted_records != sum(self.country_code_counts.values()):
            raise ValueError("GeoNames country counts do not balance")
        if self.accepted_records != sum(self.feature_code_counts.values()):
            raise ValueError("GeoNames feature-code counts do not balance")
        if self.distinct_countries != len(self.country_code_counts):
            raise ValueError("GeoNames distinct-country audit is invalid")
        if self.zero_population_records > self.accepted_records:
            raise ValueError("GeoNames zero-population count exceeds accepted rows")
        if self.minimum_population > self.maximum_population:
            raise ValueError("GeoNames population extrema are invalid")
        for label, values in (
            ("excluded non-ISO countries", self.excluded_non_iso_country_counts),
            ("countries", self.country_code_counts),
            ("feature codes", self.feature_code_counts),
        ):
            if tuple(values) != tuple(sorted(values)):
                raise ValueError(f"GeoNames {label} are not sorted")
        return self


class GeoNamesLocalityReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    provider: Literal["GeoNames"]
    dataset: Literal["cities15000"]
    source_url: Literal["https://download.geonames.org/export/dump/cities15000.zip"]
    license: Literal["Creative Commons Attribution 4.0"]
    license_url: Literal["https://creativecommons.org/licenses/by/4.0/"]
    source_receipt_sha256: Sha256
    archive_sha256: Sha256
    member_sha256: Sha256
    iso3166_path: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    iso3166_sha256: Sha256
    locality_audit: GeoNamesLocalityAudit
    jsonl_bytes: Annotated[int, Field(gt=0)]
    jsonl_sha256: Sha256
    sqlite_bytes: Annotated[int, Field(gt=0)]
    sqlite_sha256: Sha256
    content_sha256: Sha256

    @model_validator(mode="after")
    def content_hash_is_valid(self) -> GeoNamesLocalityReceipt:
        body = self.model_dump(mode="json", exclude={"content_sha256"})
        if sha256_bytes(canonical_json_bytes(body)) != self.content_sha256:
            raise ValueError("GeoNames locality receipt content SHA-256 is invalid")
        return self


class CompiledGeoNamesLocalities(BaseModel):
    """Immutable output paths and validated locality receipt."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    root: Path
    receipt: GeoNamesLocalityReceipt
    commit_receipt: StagedCommitReceipt
    created: bool


class GeoNamesLocalityRegistry:
    """Immutable in-memory locality lookup loaded from fully verified artifacts."""

    def __init__(self, *, rows: Sequence[LocalityRecord], receipt: GeoNamesLocalityReceipt) -> None:
        by_id: dict[int, LocalityRecord] = {}
        by_country: dict[str, list[LocalityRecord]] = {}
        for row in rows:
            if row.geoname_id in by_id:
                raise ValueError(f"duplicate GeoNames geoname id: {row.geoname_id}")
            by_id[row.geoname_id] = row
            by_country.setdefault(row.country_code, []).append(row)
        if not by_id:
            raise ValueError("GeoNames locality registry cannot be empty")
        self._by_id = MappingProxyType(by_id)
        self._by_country = MappingProxyType(
            {code: tuple(values) for code, values in sorted(by_country.items())}
        )
        self._receipt = receipt

    @property
    def audit(self) -> GeoNamesLocalityAudit:
        return self._receipt.locality_audit

    @property
    def receipt(self) -> GeoNamesLocalityReceipt:
        return self._receipt

    @property
    def country_codes(self) -> tuple[str, ...]:
        return tuple(self._by_country)

    def entry(self, geoname_id: int) -> LocalityRecord:
        try:
            return self._by_id[geoname_id]
        except KeyError as error:
            raise KeyError(f"unknown GeoNames geoname id: {geoname_id}") from error

    def rows_for_country(self, country_code: str) -> tuple[LocalityRecord, ...]:
        try:
            return self._by_country[country_code]
        except KeyError as error:
            raise KeyError(f"no GeoNames localities for ISO country: {country_code}") from error


@dataclass(frozen=True, slots=True)
class _ParsedArchive:
    rows: tuple[LocalityRecord, ...]
    source_records: int
    excluded_non_iso_country_counts: Mapping[str, int]
    member_sha256: str
    member_bytes: int


def _read_bounded_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        payload = read_regular_file_bytes(path)
    except ArtifactReadError as error:
        raise GeoNamesLocalityError(f"{label} is not a readable regular file: {path}") from error
    if not payload or len(payload) > maximum_bytes:
        raise GeoNamesLocalityError(
            f"{label} byte count is outside 1..{maximum_bytes}: {len(payload)}"
        )
    return payload


def _validate_zip_member(info: zipfile.ZipInfo, *, archive_entries: int) -> str:
    path = PurePosixPath(info.filename)
    if (
        archive_entries != 1
        or info.filename != GEONAMES_MEMBER_PATH
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or info.is_dir()
    ):
        raise GeoNamesLocalityError(
            "GeoNames ZIP must contain only the regular cities15000.txt member"
        )
    unix_mode = info.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise GeoNamesLocalityError("GeoNames ZIP member must not be a symbolic link")
    if info.flag_bits & 0x1:
        raise GeoNamesLocalityError("GeoNames ZIP member must not be encrypted")
    compression = _SUPPORTED_COMPRESSION.get(info.compress_type)
    if compression is None:
        raise GeoNamesLocalityError("GeoNames ZIP uses an unsupported compression method")
    invalid_size = (
        not 0 < info.file_size <= _MAX_MEMBER_BYTES
        or not 0 < info.compress_size <= _MAX_MEMBER_BYTES
    )
    if invalid_size:
        raise GeoNamesLocalityError("GeoNames ZIP member byte counts exceed safety bounds")
    return compression


def _zip_info(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, str]:
    infos = archive.infolist()
    if len({info.filename for info in infos}) != len(infos):
        raise GeoNamesLocalityError("GeoNames ZIP contains duplicate member names")
    if len(infos) != 1:
        raise GeoNamesLocalityError(
            "GeoNames ZIP must contain only the regular cities15000.txt member"
        )
    info = infos[0]
    return info, _validate_zip_member(info, archive_entries=len(infos))


def _open_zip(payload: bytes) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(payload), mode="r", allowZip64=False)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise GeoNamesLocalityError("GeoNames source is not a valid bounded ZIP") from error


def _hash_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    try:
        with archive.open(info, mode="r") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
                if total > _MAX_MEMBER_BYTES:
                    raise GeoNamesLocalityError("GeoNames ZIP member exceeds safety bounds")
                digest.update(chunk)
    except (OSError, zipfile.BadZipFile) as error:
        raise GeoNamesLocalityError("GeoNames ZIP member cannot be read safely") from error
    if total != info.file_size:
        raise GeoNamesLocalityError("GeoNames ZIP member size differs from central directory")
    return total, digest.hexdigest()


def build_geonames_source_receipt(
    *, archive_path: Path, output_path: Path
) -> GeoNamesSourceReceipt:
    """Inspect and immutably receipt an already-downloaded official archive."""

    payload = _read_bounded_regular(
        archive_path, maximum_bytes=_MAX_ARCHIVE_BYTES, label="GeoNames archive"
    )
    try:
        with _open_zip(payload) as archive:
            info, compression = _zip_info(archive)
            member_bytes, member_sha256 = _hash_member(archive, info)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise GeoNamesLocalityError("GeoNames source is not a valid bounded ZIP") from error
    member = {
        "path": GEONAMES_MEMBER_PATH,
        "compression": compression,
        "compressed_bytes": info.compress_size,
        "uncompressed_bytes": member_bytes,
        "crc32": f"{info.CRC:08x}",
        "sha256": member_sha256,
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "provider": "GeoNames",
        "dataset": "cities15000",
        "source_url": GEONAMES_SOURCE_URL,
        "license": GEONAMES_LICENSE,
        "license_url": GEONAMES_LICENSE_URL,
        "archive_path": GEONAMES_ARCHIVE_PATH,
        "archive_bytes": len(payload),
        "archive_sha256": sha256_bytes(payload),
        "members": (member,),
    }
    receipt = GeoNamesSourceReceipt.model_validate(
        {**body, "content_sha256": sha256_bytes(canonical_json_bytes(body))}, strict=True
    )
    atomic_publish_bytes(
        output_path,
        canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n",
    )
    return receipt


def _load_iso_country_codes(*, path: Path, expected_sha256: str) -> frozenset[str]:
    payload = _read_bounded_regular(path, maximum_bytes=_MAX_ISO_BYTES, label="ISO-3166 snapshot")
    if sha256_bytes(payload) != expected_sha256:
        raise GeoNamesLocalityError("ISO-3166 snapshot SHA-256 mismatch")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeoNamesLocalityError("ISO-3166 snapshot is not valid UTF-8 JSON") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"3166-1"}
        or not isinstance(value["3166-1"], list)
    ):
        raise GeoNamesLocalityError("ISO-3166 snapshot has an unexpected root contract")
    codes: set[str] = set()
    for index, raw in enumerate(value["3166-1"]):
        code = raw.get("alpha_2") if isinstance(raw, dict) else None
        if not isinstance(code, str) or re.fullmatch(r"[A-Z]{2}", code) is None:
            raise GeoNamesLocalityError(f"ISO-3166 record {index} has an invalid alpha-2 code")
        if code in codes:
            raise GeoNamesLocalityError(f"ISO-3166 snapshot has duplicate alpha-2 code: {code}")
        codes.add(code)
    if not codes:
        raise GeoNamesLocalityError("ISO-3166 snapshot contains no countries")
    return frozenset(codes)


def _parse_decimal(value: str, *, field: str, line_number: int) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise GeoNamesLocalityError(f"GeoNames line {line_number} has invalid {field}") from error
    if not result.is_finite():
        raise GeoNamesLocalityError(f"GeoNames line {line_number} has non-finite {field}")
    return result


def _parse_nonnegative_integer(value: str, *, field: str, line_number: int) -> int:
    if not value.isascii() or not value.isdigit():
        raise GeoNamesLocalityError(f"GeoNames line {line_number} has invalid {field}")
    result = int(value)
    if result < 0:
        raise GeoNamesLocalityError(f"GeoNames line {line_number} has negative {field}")
    return result


def _parse_source_row(raw: bytes, *, line_number: int) -> LocalityRecord:
    if len(raw) > _MAX_LINE_BYTES:
        raise GeoNamesLocalityError(f"GeoNames line {line_number} exceeds safety bounds")
    try:
        line = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GeoNamesLocalityError(f"GeoNames line {line_number} is not valid UTF-8") from error
    line = line.removesuffix("\n").removesuffix("\r")
    fields = line.split("\t")
    if len(fields) != _GEONAMES_COLUMNS:
        raise GeoNamesLocalityError(
            f"GeoNames line {line_number} has {len(fields)} columns, expected 19"
        )
    geoname_id = _parse_nonnegative_integer(fields[0], field="geoname id", line_number=line_number)
    if geoname_id == 0:
        raise GeoNamesLocalityError(f"GeoNames line {line_number} has non-positive geoname id")
    if fields[6] != "P":
        raise GeoNamesLocalityError(
            f"GeoNames line {line_number} has non-populated feature class: {fields[6]!r}"
        )
    population = _parse_nonnegative_integer(fields[14], field="population", line_number=line_number)
    try:
        return LocalityRecord.model_validate(
            {
                "geoname_id": geoname_id,
                "canonical_name": fields[1],
                "ascii_name": fields[2],
                "country_code": fields[8],
                "feature_code": fields[7],
                "population": population,
                "latitude": _parse_decimal(fields[4], field="latitude", line_number=line_number),
                "longitude": _parse_decimal(fields[5], field="longitude", line_number=line_number),
                "timezone": fields[17],
            },
            strict=True,
        )
    except ValueError as error:
        raise GeoNamesLocalityError(
            f"GeoNames line {line_number} violates the locality contract"
        ) from error


def _parse_archive(
    payload: bytes,
    *,
    pin: GeoNamesZipMemberPin,
    iso_country_codes: frozenset[str],
) -> _ParsedArchive:
    rows: list[LocalityRecord] = []
    excluded: Counter[str] = Counter()
    identifiers: set[int] = set()
    digest = hashlib.sha256()
    member_bytes = 0
    source_records = 0
    try:
        with _open_zip(payload) as archive:
            info, compression = _zip_info(archive)
            actual_metadata = (
                info.filename,
                compression,
                info.compress_size,
                info.file_size,
                f"{info.CRC:08x}",
            )
            expected_metadata = (
                pin.path,
                pin.compression,
                pin.compressed_bytes,
                pin.uncompressed_bytes,
                pin.crc32,
            )
            if actual_metadata != expected_metadata:
                raise GeoNamesLocalityError("GeoNames ZIP member metadata differs from its pin")
            with archive.open(info, mode="r") as stream:
                for source_records, raw in enumerate(stream, start=1):
                    member_bytes += len(raw)
                    if member_bytes > _MAX_MEMBER_BYTES:
                        raise GeoNamesLocalityError("GeoNames ZIP member exceeds safety bounds")
                    digest.update(raw)
                    row = _parse_source_row(raw, line_number=source_records)
                    if row.geoname_id in identifiers:
                        raise GeoNamesLocalityError(
                            f"duplicate GeoNames geoname id: {row.geoname_id}"
                        )
                    identifiers.add(row.geoname_id)
                    if row.country_code not in iso_country_codes:
                        excluded[row.country_code] += 1
                        continue
                    rows.append(row)
    except (OSError, zipfile.BadZipFile) as error:
        raise GeoNamesLocalityError("GeoNames ZIP member cannot be read safely") from error
    if not source_records or not rows:
        raise GeoNamesLocalityError("GeoNames source contains no eligible locality rows")
    if member_bytes != pin.uncompressed_bytes or digest.hexdigest() != pin.sha256:
        raise GeoNamesLocalityError("GeoNames ZIP member bytes differ from their pin")
    ordered = tuple(sorted(rows, key=lambda row: row.geoname_id))
    if tuple(row.geoname_id for row in ordered) != tuple(
        sorted({row.geoname_id for row in ordered})
    ):
        raise GeoNamesLocalityError("compiled GeoNames localities are duplicated or unsorted")
    return _ParsedArchive(
        rows=ordered,
        source_records=source_records,
        excluded_non_iso_country_counts=dict(sorted(excluded.items())),
        member_sha256=digest.hexdigest(),
        member_bytes=member_bytes,
    )


def _locality_jsonl(rows: Sequence[LocalityRecord]) -> bytes:
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows)


def _locality_sqlite(rows: Sequence[LocalityRecord], *, parent: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix=".geonames-sqlite-", dir=parent) as raw:
        path = Path(raw) / "localities.sqlite3"
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA page_size = 4096")
            connection.execute(f"PRAGMA application_id = {_SQLITE_APPLICATION_ID}")
            connection.execute("PRAGMA user_version = 1")
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.executescript(
                """
                CREATE TABLE localities (
                    geoname_id INTEGER PRIMARY KEY CHECK(geoname_id > 0),
                    canonical_name TEXT NOT NULL CHECK(length(canonical_name) > 0),
                    ascii_name TEXT NOT NULL CHECK(length(ascii_name) > 0),
                    country_code TEXT NOT NULL
                        CHECK(length(country_code) = 2 AND country_code GLOB '[A-Z][A-Z]'),
                    feature_code TEXT NOT NULL CHECK(length(feature_code) > 0),
                    population INTEGER NOT NULL CHECK(population >= 0),
                    latitude TEXT NOT NULL CHECK(length(latitude) > 0),
                    longitude TEXT NOT NULL CHECK(length(longitude) > 0),
                    timezone TEXT NOT NULL CHECK(length(timezone) > 0)
                );
                CREATE INDEX localities_country_population_id_idx
                    ON localities(country_code, population DESC, geoname_id);
                """
            )
            connection.executemany(
                "INSERT INTO localities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(
                    (
                        row.geoname_id,
                        row.canonical_name,
                        row.ascii_name,
                        row.country_code,
                        row.feature_code,
                        row.population,
                        str(row.latitude),
                        str(row.longitude),
                        row.timezone,
                    )
                    for row in rows
                ),
            )
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise GeoNamesLocalityError("GeoNames SQLite failed integrity_check")
            if connection.execute("SELECT COUNT(*) FROM localities").fetchone() != (len(rows),):
                raise GeoNamesLocalityError("GeoNames SQLite row count differs from JSONL")
        finally:
            connection.close()
        if tuple(path.parent.glob("localities.sqlite3-*")):
            raise GeoNamesLocalityError("GeoNames SQLite left journal sidecars")
        return path.read_bytes()


def _locality_audit(parsed: _ParsedArchive) -> GeoNamesLocalityAudit:
    countries = Counter(row.country_code for row in parsed.rows)
    feature_codes = Counter(row.feature_code for row in parsed.rows)
    populations = tuple(row.population for row in parsed.rows)
    return GeoNamesLocalityAudit.model_validate(
        {
            "source_records": parsed.source_records,
            "accepted_records": len(parsed.rows),
            "excluded_non_iso_country_records": sum(
                parsed.excluded_non_iso_country_counts.values()
            ),
            "excluded_non_iso_country_counts": parsed.excluded_non_iso_country_counts,
            "distinct_countries": len(countries),
            "country_code_counts": dict(sorted(countries.items())),
            "feature_code_counts": dict(sorted(feature_codes.items())),
            "zero_population_records": sum(value == 0 for value in populations),
            "minimum_population": min(populations),
            "maximum_population": max(populations),
        },
        strict=True,
    )


def compile_geonames_locality_registry(
    *,
    source_receipt_path: Path,
    expected_source_receipt_sha256: str,
    iso_path: Path,
    expected_iso_sha256: str,
    output_parent: Path,
    run_name: str,
) -> CompiledGeoNamesLocalities:
    """Compile stable, unique ISO-conditioned locality rows from cities15000."""

    source_receipt_payload = _read_bounded_regular(
        source_receipt_path,
        maximum_bytes=_MAX_SOURCE_RECEIPT_BYTES,
        label="GeoNames source receipt",
    )
    if sha256_bytes(source_receipt_payload) != expected_source_receipt_sha256:
        raise GeoNamesLocalityError("GeoNames source receipt SHA-256 mismatch")
    try:
        source_receipt = GeoNamesSourceReceipt.model_validate_json(
            source_receipt_payload, strict=True
        )
    except ValueError as error:
        raise GeoNamesLocalityError("GeoNames source receipt is invalid") from error
    archive_path = source_receipt_path.parent / source_receipt.archive_path
    archive_payload = _read_bounded_regular(
        archive_path, maximum_bytes=_MAX_ARCHIVE_BYTES, label="GeoNames archive"
    )
    if (
        len(archive_payload) != source_receipt.archive_bytes
        or sha256_bytes(archive_payload) != source_receipt.archive_sha256
    ):
        raise GeoNamesLocalityError("GeoNames archive bytes differ from source receipt")
    iso_codes = _load_iso_country_codes(path=iso_path, expected_sha256=expected_iso_sha256)
    parsed = _parse_archive(
        archive_payload,
        pin=source_receipt.members[0],
        iso_country_codes=iso_codes,
    )
    jsonl = _locality_jsonl(parsed.rows)
    audit = _locality_audit(parsed)
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_name,
        transaction_sha256=sha256_bytes(
            canonical_json_bytes(
                {
                    "contract": "geonames-cities15000-iso-locality-registry-v1",
                    "sourceReceiptSha256": expected_source_receipt_sha256,
                    "sourceReceipt": source_receipt.model_dump(mode="json"),
                    "iso3166Path": str(iso_path),
                    "iso3166Sha256": expected_iso_sha256,
                    "implementationSha256": sha256_file(Path(__file__)),
                }
            )
        ),
    )
    sqlite = _locality_sqlite(parsed.rows, parent=stage.output_parent)
    body: dict[str, Any] = {
        "schema_version": 1,
        "provider": "GeoNames",
        "dataset": "cities15000",
        "source_url": GEONAMES_SOURCE_URL,
        "license": GEONAMES_LICENSE,
        "license_url": GEONAMES_LICENSE_URL,
        "source_receipt_sha256": expected_source_receipt_sha256,
        "archive_sha256": source_receipt.archive_sha256,
        "member_sha256": parsed.member_sha256,
        "iso3166_path": str(iso_path),
        "iso3166_sha256": expected_iso_sha256,
        "locality_audit": audit.model_dump(mode="json"),
        "jsonl_bytes": len(jsonl),
        "jsonl_sha256": sha256_bytes(jsonl),
        "sqlite_bytes": len(sqlite),
        "sqlite_sha256": sha256_bytes(sqlite),
    }
    receipt = GeoNamesLocalityReceipt.model_validate(
        {**body, "content_sha256": sha256_bytes(canonical_json_bytes(body))}, strict=True
    )
    stage.publish_bytes("localities.jsonl", jsonl)
    stage.publish_bytes("localities.sqlite3", sqlite)
    stage.publish_bytes(
        "registry-receipt.json",
        canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n",
    )
    committed = stage.commit(
        expected_artifacts=(
            "localities.jsonl",
            "localities.sqlite3",
            "registry-receipt.json",
        ),
        metadata={
            "schema_version": 1,
            "locality_records": len(parsed.rows),
            "source_receipt_sha256": expected_source_receipt_sha256,
            "iso3166_sha256": expected_iso_sha256,
        },
    )
    return CompiledGeoNamesLocalities(
        root=stage.final_root,
        receipt=receipt,
        commit_receipt=committed.receipt,
        created=committed.created,
    )


def _validate_sqlite_equivalence(path: Path, rows: Sequence[LocalityRecord]) -> None:
    expected = tuple(
        (
            row.geoname_id,
            row.canonical_name,
            row.ascii_name,
            row.country_code,
            row.feature_code,
            row.population,
            str(row.latitude),
            str(row.longitude),
            row.timezone,
        )
        for row in rows
    )
    try:
        uri = path.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.execute("PRAGMA query_only = ON")
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise GeoNamesLocalityError("GeoNames SQLite failed integrity_check")
            if connection.execute("PRAGMA application_id").fetchone() != (
                _SQLITE_APPLICATION_ID,
            ) or connection.execute("PRAGMA user_version").fetchone() != (1,):
                raise GeoNamesLocalityError("GeoNames SQLite metadata contract is invalid")
            actual = tuple(
                connection.execute(
                    "SELECT geoname_id, canonical_name, ascii_name, country_code, "
                    "feature_code, population, latitude, longitude, timezone "
                    "FROM localities ORDER BY geoname_id"
                )
            )
    except sqlite3.Error as error:
        raise GeoNamesLocalityError("GeoNames SQLite cannot be validated") from error
    if actual != expected:
        raise GeoNamesLocalityError("GeoNames SQLite rows differ from canonical JSONL")


def load_geonames_locality_registry(
    *, root: Path, expected_receipt_sha256: str
) -> GeoNamesLocalityRegistry:
    """Load only a hash-pinned, internally equivalent compiled registry."""

    if root.is_symlink() or not root.is_dir():
        raise GeoNamesLocalityError("GeoNames registry root must be a plain directory")
    receipt_payload = _read_bounded_regular(
        root / "registry-receipt.json",
        maximum_bytes=_MAX_SOURCE_RECEIPT_BYTES,
        label="GeoNames locality receipt",
    )
    if sha256_bytes(receipt_payload) != expected_receipt_sha256:
        raise GeoNamesLocalityError("GeoNames locality receipt SHA-256 mismatch")
    try:
        receipt = GeoNamesLocalityReceipt.model_validate_json(receipt_payload, strict=True)
    except ValueError as error:
        raise GeoNamesLocalityError("GeoNames locality receipt is invalid") from error
    jsonl_payload = _read_bounded_regular(
        root / "localities.jsonl",
        maximum_bytes=_MAX_JSONL_BYTES,
        label="GeoNames locality JSONL",
    )
    if (
        len(jsonl_payload) != receipt.jsonl_bytes
        or sha256_bytes(jsonl_payload) != receipt.jsonl_sha256
    ):
        raise GeoNamesLocalityError("GeoNames locality JSONL differs from its receipt")
    if not jsonl_payload.endswith(b"\n"):
        raise GeoNamesLocalityError("GeoNames locality JSONL lacks a terminal newline")
    rows: list[LocalityRecord] = []
    for line_number, line in enumerate(jsonl_payload.splitlines(), start=1):
        if not line or len(line) > _MAX_LINE_BYTES:
            raise GeoNamesLocalityError(
                f"GeoNames locality JSONL line {line_number} violates size bounds"
            )
        try:
            row = LocalityRecord.model_validate_json(line, strict=True)
        except ValueError as error:
            raise GeoNamesLocalityError(
                f"GeoNames locality JSONL line {line_number} is invalid"
            ) from error
        if line != canonical_json_bytes(row.model_dump(mode="json")):
            raise GeoNamesLocalityError(
                f"GeoNames locality JSONL line {line_number} is not canonical"
            )
        rows.append(row)
    identifiers = tuple(row.geoname_id for row in rows)
    if identifiers != tuple(sorted(set(identifiers))):
        raise GeoNamesLocalityError("GeoNames locality JSONL is duplicated or unsorted")
    recomputed_audit = _locality_audit(
        _ParsedArchive(
            rows=tuple(rows),
            source_records=receipt.locality_audit.source_records,
            excluded_non_iso_country_counts=dict(
                receipt.locality_audit.excluded_non_iso_country_counts
            ),
            member_sha256=receipt.member_sha256,
            member_bytes=0,
        )
    )
    if recomputed_audit != receipt.locality_audit:
        raise GeoNamesLocalityError("GeoNames locality JSONL audit differs from its receipt")
    sqlite_path = root / "localities.sqlite3"
    sqlite_payload = _read_bounded_regular(
        sqlite_path,
        maximum_bytes=_MAX_SQLITE_BYTES,
        label="GeoNames locality SQLite",
    )
    if (
        len(sqlite_payload) != receipt.sqlite_bytes
        or sha256_bytes(sqlite_payload) != receipt.sqlite_sha256
    ):
        raise GeoNamesLocalityError("GeoNames locality SQLite differs from its receipt")
    _validate_sqlite_equivalence(sqlite_path, rows)
    return GeoNamesLocalityRegistry(rows=rows, receipt=receipt)
