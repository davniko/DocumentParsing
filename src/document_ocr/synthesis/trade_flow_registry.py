"""Compile pinned World Bank WITS SDMX responses into bilateral route support."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from document_ocr.atomic import atomic_publish_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.country_registry import CountryRegistry, normalize_country_alias
from document_ocr.synthesis.routes import TradeFlowRecord
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Alpha2 = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
Alpha3 = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
WitsCode = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9]{3}$")]
NumericCode = Annotated[str, StringConstraints(pattern=r"^[0-9]{3}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_URL_PREFIX = (
    "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/tradestats-trade/"
    "reporter/{reporter}/year/all/partner/all/product/Total/indicator/XPRT-TRD-VL"
)
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_COUNTRY_METADATA_BYTES = 1024 * 1024
_MAX_XML_ELEMENTS = 2_000_000
_MAX_METADATA_ELEMENTS = 2_000
_SQLITE_APPLICATION_ID = 0x57495453  # WITS
_SOURCE_NAME = re.compile(r"^[A-Z]{3}\.xml$")
_WITS_CODE = re.compile(r"^[A-Z0-9]{3}$")
_WITS_NAMESPACE = "http://wits.worldbank.org"
_COUNTRY_METADATA_URL = (
    "https://wits.worldbank.org/API/V1/wits/datasource/tradestats-trade/country/ALL"
)
_COUNTRY_METADATA_PATH = "metadata/countries.xml"
type WitsClassification = Literal["iso_country", "group", "non_country"]


class WitsTradeFlowError(RuntimeError):
    """A WITS acquisition or compilation contract failed."""


class WitsCountryMetadataPin(BaseModel):
    model_config = _STRICT

    path: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    source_url: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    bytes: Annotated[int, Field(gt=0, le=_MAX_COUNTRY_METADATA_BYTES)]
    sha256: Sha256

    @model_validator(mode="after")
    def identity_is_official(self) -> WitsCountryMetadataPin:
        if self.path != _COUNTRY_METADATA_PATH or self.source_url != _COUNTRY_METADATA_URL:
            raise ValueError("WITS country metadata pin differs from the official endpoint")
        return self


class WitsCountryMetadataAudit(BaseModel):
    model_config = _STRICT

    records: Annotated[int, Field(gt=0)]
    iso_country_records: Annotated[int, Field(gt=0)]
    group_records: Annotated[int, Field(ge=0)]
    non_country_records: Annotated[int, Field(ge=0)]
    normalized_name_match_records: Annotated[int, Field(ge=0)]
    normalized_name_difference_records: Annotated[int, Field(ge=0)]
    provider_code_match_records: Annotated[int, Field(ge=0)]
    provider_code_difference_records: Annotated[int, Field(ge=0)]
    classification_sha256: Sha256
    name_audit_sha256: Sha256

    @model_validator(mode="after")
    def classifications_balance(self) -> WitsCountryMetadataAudit:
        if self.iso_country_records + self.group_records + self.non_country_records != self.records:
            raise ValueError("WITS country metadata classifications do not balance")
        if (
            self.normalized_name_match_records + self.normalized_name_difference_records
            != self.iso_country_records
            or self.provider_code_match_records + self.provider_code_difference_records
            != self.iso_country_records
        ):
            raise ValueError("WITS country metadata ISO audit does not balance")
        return self


class WitsReporterPin(BaseModel):
    model_config = _STRICT

    wits_reporter_code: WitsCode
    wits_numeric_code: WitsCode
    wits_name: Annotated[str, StringConstraints(min_length=1, max_length=192)]
    classification: WitsClassification
    origin_country_code: Alpha2 | None
    origin_alpha3: Alpha3 | None
    iso_numeric_code: NumericCode | None
    normalized_name_matches: bool | None
    provider_code_matches_iso_alpha3: bool | None
    path: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    source_url: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    bytes: Annotated[int, Field(gt=0, le=_MAX_SOURCE_BYTES)]
    sha256: Sha256

    @model_validator(mode="after")
    def identity_is_safe_and_official(self) -> WitsReporterPin:
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("WITS source path must be a safe relative POSIX path")
        if self.path != f"raw/{self.wits_reporter_code}.xml":
            raise ValueError("WITS source path differs from its reporter identity")
        expected_url = _URL_PREFIX.format(reporter=self.wits_reporter_code)
        parsed = urlsplit(self.source_url)
        if self.source_url != expected_url or parsed.fragment or parsed.query:
            raise ValueError("WITS source URL does not match the official pinned endpoint")
        iso_values = (
            self.origin_country_code,
            self.origin_alpha3,
            self.iso_numeric_code,
            self.normalized_name_matches,
            self.provider_code_matches_iso_alpha3,
        )
        if self.classification == "iso_country":
            if any(value is None for value in iso_values):
                raise ValueError("ISO-resolved WITS reporter lacks its numeric identity audit")
            if self.wits_numeric_code != self.iso_numeric_code:
                raise ValueError("WITS reporter numeric code differs from its ISO numeric code")
        elif any(value is not None for value in iso_values):
            raise ValueError("non-country WITS reporter must not carry an ISO identity")
        return self


class WitsSourceManifest(BaseModel):
    model_config = _STRICT

    schema_version: Literal[2]
    provider: Literal["World Bank World Integrated Trade Solution (WITS)"]
    dataset: Literal["TradeStats - Trade"]
    requested_period: Literal["all"]
    frequency: Literal["A"]
    indicator: Literal["XPRT-TRD-VL"]
    product: Literal["Total"]
    observation_datasource: Literal["WITS-CMT"]
    iso3166_sha256: Sha256
    country_identity_policy: Literal["exact_wits_to_iso_numeric_code_v1"]
    country_metadata: WitsCountryMetadataPin
    country_metadata_audit: WitsCountryMetadataAudit
    reporters: tuple[WitsReporterPin, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_are_unique_and_sorted(self) -> WitsSourceManifest:
        reporters = tuple(row.wits_reporter_code for row in self.reporters)
        paths = tuple(row.path for row in self.reporters)
        if reporters != tuple(sorted(set(reporters))) or len(paths) != len(set(paths)):
            raise ValueError("WITS reporter identities must be unique and sorted")
        origins = tuple(
            row.origin_country_code for row in self.reporters if row.origin_country_code is not None
        )
        if len(origins) != len(set(origins)):
            raise ValueError("WITS resolved reporter origins must be unique")
        return self


class WitsReporterAudit(BaseModel):
    model_config = _STRICT

    origin_country_code: Alpha2
    wits_reporter_code: Alpha3
    selected_year: Annotated[int, Field(ge=1988, le=9999)]
    selected_year_observations: Annotated[int, Field(ge=0)]
    accepted_positive_country_observations: Annotated[int, Field(ge=0)]
    excluded_group_partner_observations: Annotated[int, Field(ge=0)]
    excluded_non_country_partner_observations: Annotated[int, Field(ge=0)]
    excluded_domestic_observations: Annotated[int, Field(ge=0)]
    excluded_nonpositive_observations: Annotated[int, Field(ge=0)]
    duplicate_partner_observations: Annotated[int, Field(ge=0)]
    max_trade_value_decimal_places: Annotated[int, Field(ge=0, le=11)]

    @model_validator(mode="after")
    def observations_balance(self) -> WitsReporterAudit:
        classified = (
            self.accepted_positive_country_observations
            + self.excluded_group_partner_observations
            + self.excluded_non_country_partner_observations
            + self.excluded_domestic_observations
            + self.excluded_nonpositive_observations
            + self.duplicate_partner_observations
        )
        if classified != self.selected_year_observations:
            raise ValueError("WITS selected-year observation audit does not balance")
        return self


class WitsExcludedReporterAudit(BaseModel):
    model_config = _STRICT

    wits_reporter_code: WitsCode
    wits_numeric_code: WitsCode
    wits_name: Annotated[str, StringConstraints(min_length=1, max_length=192)]
    classification: Literal["group", "non_country"]


class WitsTradeFlowReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[2]
    source_manifest_sha256: Sha256
    iso3166_sha256: Sha256
    wits_country_metadata_sha256: Sha256
    trade_value_representation: Literal["exact_decimal_max_11_places_v1"]
    source_reporter_files: Annotated[int, Field(gt=0)]
    compiled_reporter_files: Annotated[int, Field(gt=0)]
    excluded_reporter_files: Annotated[int, Field(ge=0)]
    trade_flow_records: Annotated[int, Field(gt=0)]
    distinct_origin_countries: Annotated[int, Field(gt=0)]
    distinct_destination_countries: Annotated[int, Field(gt=0)]
    reporter_audits: tuple[WitsReporterAudit, ...] = Field(min_length=1)
    excluded_reporters: tuple[WitsExcludedReporterAudit, ...]
    jsonl_bytes: Annotated[int, Field(gt=0)]
    jsonl_sha256: Sha256
    sqlite_bytes: Annotated[int, Field(gt=0)]
    sqlite_sha256: Sha256
    content_sha256: Sha256

    @model_validator(mode="after")
    def content_hash_is_valid(self) -> WitsTradeFlowReceipt:
        if self.compiled_reporter_files != len(self.reporter_audits):
            raise ValueError("WITS reporter audit count differs from source file count")
        if self.excluded_reporter_files != len(self.excluded_reporters):
            raise ValueError("WITS excluded-reporter count differs from its audit")
        if (
            self.compiled_reporter_files + self.excluded_reporter_files
            != self.source_reporter_files
        ):
            raise ValueError("WITS source reporter classification does not balance")
        origins = tuple(row.origin_country_code for row in self.reporter_audits)
        if origins != tuple(sorted(set(origins))):
            raise ValueError("WITS reporter audits are not sorted")
        if self.trade_flow_records != sum(
            row.accepted_positive_country_observations for row in self.reporter_audits
        ):
            raise ValueError("WITS trade-flow count differs from accepted reporter lanes")
        if self.distinct_origin_countries != self.compiled_reporter_files:
            raise ValueError("WITS compiled output does not cover every reporter origin")
        excluded_codes = tuple(row.wits_reporter_code for row in self.excluded_reporters)
        if excluded_codes != tuple(sorted(set(excluded_codes))):
            raise ValueError("WITS excluded reporter audits are not unique and sorted")
        body = self.model_dump(mode="json", exclude={"content_sha256"})
        if sha256_bytes(canonical_json_bytes(body)) != self.content_sha256:
            raise ValueError("WITS trade-flow receipt content SHA-256 is invalid")
        return self


class CompiledWitsTradeFlows(BaseModel):
    """Immutable output paths and validated registry receipt."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    root: Path
    receipt: WitsTradeFlowReceipt
    commit_receipt: StagedCommitReceipt
    created: bool


@dataclass(frozen=True, slots=True)
class _WitsCountryResolution:
    wits_code: str
    numeric_code: str
    name: str
    is_reporter: bool
    is_partner: bool
    classification: WitsClassification
    iso_alpha2: str | None
    iso_alpha3: str | None
    iso_numeric: str | None
    normalized_name_matches: bool | None
    provider_code_matches_iso_alpha3: bool | None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _decimal_places(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise WitsTradeFlowError("finite WITS trade value has a non-integral exponent")
    return max(0, -exponent)


def _forbid_xml_entities(payload: bytes, *, label: str) -> None:
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise WitsTradeFlowError(f"{label} XML declarations/entities are forbidden")


def _parse_wits_country_metadata(
    payload: bytes,
    *,
    countries: CountryRegistry,
) -> tuple[dict[str, _WitsCountryResolution], WitsCountryMetadataAudit]:
    _forbid_xml_entities(payload, label="WITS country metadata")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise WitsTradeFlowError("WITS country metadata is not valid XML") from error
    expected_root_tag = f"{{{_WITS_NAMESPACE}}}datasource"
    expected_root_attributes = {
        "datasourcecode": "tradestats-trade",
        "datasourcename": "WITS TradeStats - Trade",
        "language": "en",
    }
    if root.tag != expected_root_tag or any(
        root.attrib.get(key) != value for key, value in expected_root_attributes.items()
    ):
        raise WitsTradeFlowError("WITS country metadata root contract differs")
    if set(root.attrib) != {*expected_root_attributes, "total"}:
        raise WitsTradeFlowError("WITS country metadata root attributes differ")
    raw_total = root.attrib["total"]
    if not raw_total.isdigit() or int(raw_total) <= 0:
        raise WitsTradeFlowError("WITS country metadata total is invalid")
    if len(root) != 1 or root[0].tag != f"{{{_WITS_NAMESPACE}}}countries":
        raise WitsTradeFlowError("WITS country metadata must contain one countries element")

    by_numeric = {
        countries.entry(alpha2).numeric: countries.entry(alpha2)
        for alpha2 in countries.country_codes
    }
    resolutions: dict[str, _WitsCountryResolution] = {}
    numeric_codes: set[str] = set()
    classification_rows: list[dict[str, Any]] = []
    name_rows: list[dict[str, Any]] = []
    classifications: Counter[str] = Counter()
    name_matches = 0
    provider_matches = 0
    country_elements = tuple(root[0])
    if len(country_elements) != int(raw_total):
        raise WitsTradeFlowError("WITS country metadata total differs from its records")
    for elements, element in enumerate(country_elements, start=1):
        if elements > _MAX_METADATA_ELEMENTS:
            raise WitsTradeFlowError("WITS country metadata exceeds its element bound")
        if element.tag != f"{{{_WITS_NAMESPACE}}}country" or set(element.attrib) != {
            "countrycode",
            "isreporter",
            "ispartner",
            "isgroup",
            "grouptype",
        }:
            raise WitsTradeFlowError("WITS country metadata record contract differs")
        if tuple(_local_name(child.tag) for child in element) != (
            "iso3Code",
            "name",
            "notes",
        ) or any(
            child.tag != f"{{{_WITS_NAMESPACE}}}{_local_name(child.tag)}" for child in element
        ):
            raise WitsTradeFlowError("WITS country metadata child contract differs")
        numeric_code = element.attrib["countrycode"]
        wits_code = element[0].text or ""
        name = element[1].text or ""
        if (
            _WITS_CODE.fullmatch(numeric_code) is None
            or _WITS_CODE.fullmatch(wits_code) is None
            or name != name.strip()
            or not name
        ):
            raise WitsTradeFlowError("WITS country metadata identity fields are invalid")
        if wits_code in resolutions or numeric_code in numeric_codes:
            raise WitsTradeFlowError("WITS country metadata identities are duplicated")
        numeric_codes.add(numeric_code)
        if element.attrib["isreporter"] not in {"0", "1"} or element.attrib["ispartner"] not in {
            "0",
            "1",
        }:
            raise WitsTradeFlowError("WITS country metadata role flags are invalid")
        is_group = element.attrib["isgroup"]
        group_type = element.attrib["grouptype"]
        if (is_group, group_type) not in {
            ("Yes", "Region"),
            ("No", "N/A"),
            ("N/A", "N/A"),
        }:
            raise WitsTradeFlowError("WITS country metadata group flags are invalid")
        iso_entry = by_numeric.get(numeric_code) if is_group != "Yes" else None
        if is_group == "Yes":
            classification: WitsClassification = "group"
        elif iso_entry is not None:
            classification = "iso_country"
        else:
            classification = "non_country"
        classifications[classification] += 1

        normalized_name_matches: bool | None = None
        provider_code_matches: bool | None = None
        if iso_entry is not None:
            iso_names = tuple(
                value
                for value in (
                    iso_entry.name,
                    iso_entry.official_name,
                    iso_entry.common_name,
                )
                if value is not None
            )
            normalized_name_matches = normalize_country_alias(name) in {
                normalize_country_alias(value) for value in iso_names
            }
            provider_code_matches = wits_code == iso_entry.alpha3
            name_matches += normalized_name_matches
            provider_matches += provider_code_matches
            name_rows.append(
                {
                    "isoAlpha2": iso_entry.alpha2,
                    "isoAlpha3": iso_entry.alpha3,
                    "isoNames": iso_names,
                    "normalizedNameMatches": normalized_name_matches,
                    "providerCodeMatchesIsoAlpha3": provider_code_matches,
                    "witsCode": wits_code,
                    "witsName": name,
                    "witsNumericCode": numeric_code,
                }
            )
        resolution = _WitsCountryResolution(
            wits_code=wits_code,
            numeric_code=numeric_code,
            name=name,
            is_reporter=element.attrib["isreporter"] == "1",
            is_partner=element.attrib["ispartner"] == "1",
            classification=classification,
            iso_alpha2=None if iso_entry is None else iso_entry.alpha2,
            iso_alpha3=None if iso_entry is None else iso_entry.alpha3,
            iso_numeric=None if iso_entry is None else iso_entry.numeric,
            normalized_name_matches=normalized_name_matches,
            provider_code_matches_iso_alpha3=provider_code_matches,
        )
        resolutions[wits_code] = resolution
        classification_rows.append(
            {
                "classification": classification,
                "isPartner": resolution.is_partner,
                "isReporter": resolution.is_reporter,
                "isoAlpha2": resolution.iso_alpha2,
                "witsCode": wits_code,
                "witsNumericCode": numeric_code,
            }
        )
    ordered_classifications = sorted(classification_rows, key=lambda row: row["witsCode"])
    ordered_names = sorted(name_rows, key=lambda row: row["witsCode"])
    audit = WitsCountryMetadataAudit(
        records=len(resolutions),
        iso_country_records=classifications["iso_country"],
        group_records=classifications["group"],
        non_country_records=classifications["non_country"],
        normalized_name_match_records=name_matches,
        normalized_name_difference_records=classifications["iso_country"] - name_matches,
        provider_code_match_records=provider_matches,
        provider_code_difference_records=classifications["iso_country"] - provider_matches,
        classification_sha256=sha256_bytes(canonical_json_bytes(ordered_classifications)),
        name_audit_sha256=sha256_bytes(canonical_json_bytes(ordered_names)),
    )
    return resolutions, audit


def _regular_file_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    if path.is_symlink():
        raise WitsTradeFlowError(f"WITS source must not be a symbolic link: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise WitsTradeFlowError(f"WITS source is not readable: {path}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise WitsTradeFlowError(f"WITS input is not a regular file: {path}")
        if info.st_size <= 0 or info.st_size > maximum_bytes:
            raise WitsTradeFlowError(f"WITS input exceeds its size contract: {path}")
        if expected_bytes is not None and info.st_size != expected_bytes:
            raise WitsTradeFlowError(f"WITS source size mismatch: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                chunks.append(chunk)
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise WitsTradeFlowError(f"WITS source SHA-256 mismatch: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _parse_source_identity(payload: bytes) -> tuple[frozenset[str], frozenset[int]]:
    _forbid_xml_entities(payload, label="WITS source")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise WitsTradeFlowError("WITS source is not valid XML") from error
    if _local_name(root.tag) != "StructureSpecificData":
        raise WitsTradeFlowError("WITS XML root is not StructureSpecificData")
    reporters: set[str] = set()
    years: set[int] = set()
    for elements, element in enumerate(root.iter(), start=1):
        if elements > _MAX_XML_ELEMENTS:
            raise WitsTradeFlowError("WITS XML exceeds the bounded element count")
        if _local_name(element.tag) == "Series":
            reporter = element.attrib.get("REPORTER")
            if reporter is not None:
                reporters.add(reporter)
        elif _local_name(element.tag) == "Obs":
            raw_year = element.attrib.get("TIME_PERIOD")
            if raw_year is not None and raw_year.isdigit():
                years.add(int(raw_year))
    return frozenset(reporters), frozenset(years)


def build_wits_source_manifest(
    *,
    raw_dir: Path,
    country_metadata_path: Path,
    countries: CountryRegistry,
    output_path: Path,
) -> WitsSourceManifest:
    """Receipt an already-downloaded official WITS response set exactly once."""

    if raw_dir.is_symlink() or not raw_dir.is_dir():
        raise WitsTradeFlowError("WITS raw source root must be a regular directory")
    entries = tuple(sorted(raw_dir.iterdir(), key=lambda path: path.name))
    invalid = tuple(
        path.name
        for path in entries
        if path.is_symlink() or not path.is_file() or _SOURCE_NAME.fullmatch(path.name) is None
    )
    if invalid:
        raise WitsTradeFlowError(f"WITS raw source inventory contains invalid entries: {invalid}")
    if not entries:
        raise WitsTradeFlowError("WITS raw source inventory is empty")

    metadata_payload = _regular_file_bytes(
        country_metadata_path,
        maximum_bytes=_MAX_COUNTRY_METADATA_BYTES,
    )
    wits_countries, metadata_audit = _parse_wits_country_metadata(
        metadata_payload,
        countries=countries,
    )
    pins: list[WitsReporterPin] = []
    for path in entries:
        reporter = path.stem
        resolution = wits_countries.get(reporter)
        if resolution is None or not resolution.is_reporter:
            raise WitsTradeFlowError(
                f"WITS source reporter is absent or not reporter-eligible: {reporter}"
            )
        payload = _regular_file_bytes(path, maximum_bytes=_MAX_SOURCE_BYTES)
        reporters, years = _parse_source_identity(payload)
        if reporters != {reporter} or not years:
            raise WitsTradeFlowError(
                f"WITS source identity differs for {reporter}: reporters={sorted(reporters)}"
            )
        pins.append(
            WitsReporterPin(
                wits_reporter_code=reporter,
                wits_numeric_code=resolution.numeric_code,
                wits_name=resolution.name,
                classification=resolution.classification,
                origin_country_code=resolution.iso_alpha2,
                origin_alpha3=resolution.iso_alpha3,
                iso_numeric_code=resolution.iso_numeric,
                normalized_name_matches=resolution.normalized_name_matches,
                provider_code_matches_iso_alpha3=(resolution.provider_code_matches_iso_alpha3),
                path=f"raw/{reporter}.xml",
                source_url=_URL_PREFIX.format(reporter=reporter),
                bytes=len(payload),
                sha256=sha256_bytes(payload),
            )
        )
    manifest = WitsSourceManifest(
        schema_version=2,
        provider="World Bank World Integrated Trade Solution (WITS)",
        dataset="TradeStats - Trade",
        requested_period="all",
        frequency="A",
        indicator="XPRT-TRD-VL",
        product="Total",
        observation_datasource="WITS-CMT",
        iso3166_sha256=countries.audit.iso_sha256,
        country_identity_policy="exact_wits_to_iso_numeric_code_v1",
        country_metadata=WitsCountryMetadataPin(
            path=_COUNTRY_METADATA_PATH,
            source_url=_COUNTRY_METADATA_URL,
            bytes=len(metadata_payload),
            sha256=sha256_bytes(metadata_payload),
        ),
        country_metadata_audit=metadata_audit,
        reporters=tuple(sorted(pins, key=lambda row: row.wits_reporter_code)),
    )
    payload = canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    atomic_publish_bytes(output_path, payload)
    return manifest


def _parse_reporter(
    payload: bytes,
    *,
    pin: WitsReporterPin,
    wits_countries: Mapping[str, _WitsCountryResolution],
) -> tuple[tuple[TradeFlowRecord, ...], WitsReporterAudit]:
    if pin.classification != "iso_country" or pin.origin_country_code is None:
        raise WitsTradeFlowError("cannot parse bilateral lanes for a non-country reporter")
    _forbid_xml_entities(payload, label="WITS source")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise WitsTradeFlowError(f"invalid WITS XML for {pin.wits_reporter_code}") from error
    if _local_name(root.tag) != "StructureSpecificData":
        raise WitsTradeFlowError("WITS XML root is not StructureSpecificData")
    observations: list[tuple[int, str, Decimal]] = []
    elements = 0
    for series in root.iter():
        elements += 1
        if elements > _MAX_XML_ELEMENTS:
            raise WitsTradeFlowError("WITS XML exceeds the bounded element count")
        if _local_name(series.tag) != "Series":
            continue
        expected = {
            "FREQ": "A",
            "REPORTER": pin.wits_reporter_code,
            "PRODUCTCODE": "Total",
            "INDICATOR": "XPRT-TRD-VL",
        }
        if any(series.attrib.get(key) != value for key, value in expected.items()):
            raise WitsTradeFlowError(
                f"WITS series contract differs for {pin.wits_reporter_code}: {series.attrib}"
            )
        partner = series.attrib.get("PARTNER")
        if partner is None:
            raise WitsTradeFlowError("WITS series has no partner")
        for obs in series:
            if _local_name(obs.tag) != "Obs":
                raise WitsTradeFlowError("WITS Series contains an unexpected child")
            if obs.attrib.get("DATASOURCE") != "WITS-CMT":
                raise WitsTradeFlowError("WITS observation has an unexpected datasource")
            raw_year = obs.attrib.get("TIME_PERIOD")
            raw_value = obs.attrib.get("OBS_VALUE")
            if raw_year is None or not raw_year.isdigit() or raw_value is None:
                raise WitsTradeFlowError("WITS observation has invalid year/value fields")
            try:
                value = Decimal(raw_value)
            except InvalidOperation as error:
                raise WitsTradeFlowError("WITS observation value is not decimal") from error
            if not value.is_finite():
                raise WitsTradeFlowError("WITS observation value must be finite")
            observations.append((int(raw_year), partner, value))
    if not observations:
        raise WitsTradeFlowError(f"WITS reporter has no observations: {pin.wits_reporter_code}")
    latest_year = max(row[0] for row in observations)
    selected = [row for row in observations if row[0] == latest_year]
    dispositions: Counter[str] = Counter()
    values: dict[str, Decimal] = {}
    for _year, partner, value in selected:
        resolution = wits_countries.get(partner)
        if resolution is None or not resolution.is_partner:
            raise WitsTradeFlowError(f"WITS partner is absent or not partner-eligible: {partner}")
        if resolution.classification == "group":
            dispositions["group"] += 1
            continue
        if resolution.classification == "non_country":
            dispositions["non_country"] += 1
            continue
        destination = resolution.iso_alpha2
        if destination is None:
            raise WitsTradeFlowError("ISO-classified WITS partner has no ISO identity")
        if destination == pin.origin_country_code:
            dispositions["domestic"] += 1
            continue
        if value <= 0:
            dispositions["nonpositive"] += 1
            continue
        if destination in values:
            dispositions["duplicate"] += 1
            values[destination] += value
            continue
        dispositions["accepted"] += 1
        values[destination] = value
    rows = tuple(
        TradeFlowRecord.model_validate(
            {
                "origin_country_code": pin.origin_country_code,
                "destination_country_code": destination,
                "year": latest_year,
                "trade_value": value,
                "net_mass": None,
            },
            strict=True,
        )
        for destination, value in sorted(values.items())
    )
    audit = WitsReporterAudit(
        origin_country_code=pin.origin_country_code,
        wits_reporter_code=pin.wits_reporter_code,
        selected_year=latest_year,
        selected_year_observations=len(selected),
        accepted_positive_country_observations=dispositions["accepted"],
        excluded_group_partner_observations=dispositions["group"],
        excluded_non_country_partner_observations=dispositions["non_country"],
        excluded_domestic_observations=dispositions["domestic"],
        excluded_nonpositive_observations=dispositions["nonpositive"],
        duplicate_partner_observations=dispositions["duplicate"],
        max_trade_value_decimal_places=max(
            (_decimal_places(value) for value in values.values()),
            default=0,
        ),
    )
    if not rows:
        raise WitsTradeFlowError(
            f"WITS reporter has no accepted bilateral lanes: {pin.wits_reporter_code}"
        )
    return rows, audit


def _trade_jsonl(rows: Sequence[TradeFlowRecord]) -> bytes:
    identities = tuple((row.origin_country_code, row.destination_country_code) for row in rows)
    if identities != tuple(sorted(set(identities))):
        raise WitsTradeFlowError("compiled WITS trade flows are duplicated or unsorted")
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows)


def _trade_sqlite(rows: Sequence[TradeFlowRecord], *, parent: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix=".wits-sqlite-", dir=parent) as raw:
        path = Path(raw) / "trade-flows.sqlite3"
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA page_size = 4096")
            connection.execute(f"PRAGMA application_id = {_SQLITE_APPLICATION_ID}")
            connection.execute("PRAGMA user_version = 2")
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.executescript(
                """
                CREATE TABLE trade_flows (
                    origin_country_code TEXT NOT NULL CHECK(length(origin_country_code)=2),
                    destination_country_code TEXT NOT NULL
                        CHECK(length(destination_country_code)=2),
                    year INTEGER NOT NULL CHECK(year >= 1988),
                    trade_value TEXT NOT NULL CHECK(length(trade_value) > 0),
                    PRIMARY KEY(origin_country_code, destination_country_code)
                ) WITHOUT ROWID;
                CREATE INDEX trade_flows_destination_origin_idx
                    ON trade_flows(destination_country_code, origin_country_code);
                """
            )
            connection.executemany(
                "INSERT INTO trade_flows VALUES (?, ?, ?, ?)",
                tuple(
                    (
                        row.origin_country_code,
                        row.destination_country_code,
                        row.year,
                        str(row.trade_value),
                    )
                    for row in rows
                ),
            )
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise WitsTradeFlowError("WITS SQLite failed integrity_check")
        finally:
            connection.close()
        if tuple(path.parent.glob("trade-flows.sqlite3-*")):
            raise WitsTradeFlowError("WITS SQLite left journal sidecars")
        return path.read_bytes()


def compile_wits_trade_flows(
    *,
    source_manifest_path: Path,
    expected_manifest_sha256: str,
    countries: CountryRegistry,
    output_parent: Path,
    run_name: str,
) -> CompiledWitsTradeFlows:
    """Compile the latest available partner shares for every pinned reporter."""

    manifest_payload = _regular_file_bytes(
        source_manifest_path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    if sha256_bytes(manifest_payload) != expected_manifest_sha256:
        raise WitsTradeFlowError("WITS source manifest SHA-256 mismatch")
    manifest = WitsSourceManifest.model_validate_json(manifest_payload, strict=True)
    if manifest.iso3166_sha256 != countries.audit.iso_sha256:
        raise WitsTradeFlowError("WITS manifest references a different ISO-3166 snapshot")
    metadata_path = source_manifest_path.parent / manifest.country_metadata.path
    metadata_payload = _regular_file_bytes(
        metadata_path,
        maximum_bytes=_MAX_COUNTRY_METADATA_BYTES,
        expected_bytes=manifest.country_metadata.bytes,
        expected_sha256=manifest.country_metadata.sha256,
    )
    wits_countries, metadata_audit = _parse_wits_country_metadata(
        metadata_payload,
        countries=countries,
    )
    if metadata_audit != manifest.country_metadata_audit:
        raise WitsTradeFlowError("WITS country metadata audit differs from its manifest")
    rows: list[TradeFlowRecord] = []
    audits: list[WitsReporterAudit] = []
    excluded_reporters: list[WitsExcludedReporterAudit] = []
    for pin in manifest.reporters:
        resolution = wits_countries.get(pin.wits_reporter_code)
        if resolution is None or not resolution.is_reporter:
            raise WitsTradeFlowError(
                "WITS reporter is absent or not reporter-eligible in country metadata: "
                f"{pin.wits_reporter_code}"
            )
        expected_identity = (
            resolution.numeric_code,
            resolution.name,
            resolution.classification,
            resolution.iso_alpha2,
            resolution.iso_alpha3,
            resolution.iso_numeric,
            resolution.normalized_name_matches,
            resolution.provider_code_matches_iso_alpha3,
        )
        pinned_identity = (
            pin.wits_numeric_code,
            pin.wits_name,
            pin.classification,
            pin.origin_country_code,
            pin.origin_alpha3,
            pin.iso_numeric_code,
            pin.normalized_name_matches,
            pin.provider_code_matches_iso_alpha3,
        )
        if pinned_identity != expected_identity:
            raise WitsTradeFlowError(
                f"WITS reporter pin differs from numeric identity: {pin.wits_reporter_code}"
            )
        source_path = source_manifest_path.parent / pin.path
        payload = _regular_file_bytes(
            source_path,
            maximum_bytes=_MAX_SOURCE_BYTES,
            expected_bytes=pin.bytes,
            expected_sha256=pin.sha256,
        )
        reporters, years = _parse_source_identity(payload)
        if reporters != {pin.wits_reporter_code} or not years:
            raise WitsTradeFlowError(f"WITS source identity differs for {pin.wits_reporter_code}")
        if pin.classification != "iso_country":
            excluded_reporters.append(
                WitsExcludedReporterAudit(
                    wits_reporter_code=pin.wits_reporter_code,
                    wits_numeric_code=pin.wits_numeric_code,
                    wits_name=pin.wits_name,
                    classification=pin.classification,
                )
            )
            continue
        reporter_rows, audit = _parse_reporter(
            payload,
            pin=pin,
            wits_countries=wits_countries,
        )
        rows.extend(reporter_rows)
        audits.append(audit)
    ordered_audits = tuple(sorted(audits, key=lambda row: row.origin_country_code))
    ordered_excluded_reporters = tuple(
        sorted(excluded_reporters, key=lambda row: row.wits_reporter_code)
    )
    ordered = tuple(
        sorted(rows, key=lambda row: (row.origin_country_code, row.destination_country_code))
    )
    jsonl = _trade_jsonl(ordered)
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_name,
        transaction_sha256=sha256_bytes(
            canonical_json_bytes(
                {
                    "contract": "wits-latest-bilateral-trade-flow-registry-v2",
                    "sourceManifestSha256": expected_manifest_sha256,
                    "sourcePins": manifest.model_dump(mode="json"),
                    "isoRegistryAudit": countries.audit.model_dump(mode="json"),
                    "implementationSha256": sha256_file(Path(__file__)),
                }
            )
        ),
    )
    sqlite = _trade_sqlite(ordered, parent=stage.output_parent)
    body: dict[str, Any] = {
        "schema_version": 2,
        "source_manifest_sha256": expected_manifest_sha256,
        "iso3166_sha256": countries.audit.iso_sha256,
        "wits_country_metadata_sha256": manifest.country_metadata.sha256,
        "trade_value_representation": "exact_decimal_max_11_places_v1",
        "source_reporter_files": len(manifest.reporters),
        "compiled_reporter_files": len(ordered_audits),
        "excluded_reporter_files": len(ordered_excluded_reporters),
        "trade_flow_records": len(ordered),
        "distinct_origin_countries": len({row.origin_country_code for row in ordered}),
        "distinct_destination_countries": len({row.destination_country_code for row in ordered}),
        "reporter_audits": tuple(row.model_dump(mode="json") for row in ordered_audits),
        "excluded_reporters": tuple(
            row.model_dump(mode="json") for row in ordered_excluded_reporters
        ),
        "jsonl_bytes": len(jsonl),
        "jsonl_sha256": sha256_bytes(jsonl),
        "sqlite_bytes": len(sqlite),
        "sqlite_sha256": sha256_bytes(sqlite),
    }
    receipt = WitsTradeFlowReceipt.model_validate(
        {**body, "content_sha256": sha256_bytes(canonical_json_bytes(body))}, strict=True
    )
    receipt_bytes = canonical_json_bytes(receipt.model_dump(mode="json")) + b"\n"
    stage.publish_bytes("trade-flows.jsonl", jsonl)
    stage.publish_bytes("trade-flows.sqlite3", sqlite)
    stage.publish_bytes("registry-receipt.json", receipt_bytes)
    committed = stage.commit(
        expected_artifacts=(
            "registry-receipt.json",
            "trade-flows.jsonl",
            "trade-flows.sqlite3",
        ),
        metadata={
            "schema_version": 2,
            "trade_flow_records": len(ordered),
            "source_manifest_sha256": expected_manifest_sha256,
        },
    )
    return CompiledWitsTradeFlows(
        root=stage.final_root,
        receipt=receipt,
        commit_receipt=committed.receipt,
        created=committed.created,
    )
