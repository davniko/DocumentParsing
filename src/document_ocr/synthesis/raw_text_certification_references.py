"""Pinned geographic reference index for deterministic raw-text certification.

Certification uses this index only where the source document already proves that a field is a
country, locality, or international telephone number.  The registry is not an address parser and
does not guess semantic roles from arbitrary prose.  Every runtime lookup is bound to the exact
ISO, GeoNames, and libphonenumber identities recorded in :class:`CertificationReferenceReceipt`.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any

import phonenumbers
from phonenumbers import COUNTRY_CODE_TO_REGION_CODE, NumberParseException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.locality_registry import (
    GeoNamesLocalityRegistry,
    LocalityRecord,
    load_geonames_locality_registry,
)
from document_ocr.synthesis.package_registry import LoadedPackageRegistry, load_package_registry

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_MAX_ISO_BYTES = 2 * 1024 * 1024
_INTERNATIONAL_PHONE = re.compile(r"(?<![A-Z0-9])\+[0-9][0-9(). /-]{3,}[0-9]", re.IGNORECASE)


class CertificationReferenceReceipt(BaseModel):
    """Exact identity of every external fact available to certification."""

    model_config = _STRICT

    schemaVersion: int = Field(ge=2, le=2)
    iso3166Sha256: Sha256
    geonamesRegistryReceiptSha256: Sha256
    geonamesRegistryContentSha256: Sha256
    geonamesJsonlSha256: Sha256
    geonamesRecords: Annotated[int, Field(gt=0)]
    localityProjectionSha256: Sha256
    phonenumbersDistribution: str
    phonenumbersVersion: str
    callingCodeRegionMapSha256: Sha256
    packageRegistrySha256: Sha256
    packageRegistryEntries: Annotated[int, Field(gt=0)]
    packageRegistryPayloadSha256: Sha256
    contentSha256: Sha256

    @model_validator(mode="after")
    def content_hash_is_valid(self) -> CertificationReferenceReceipt:
        if self.phonenumbersDistribution != "phonenumberslite":
            raise ValueError("certification requires the pinned phonenumberslite distribution")
        if not self.phonenumbersVersion:
            raise ValueError("certification phonenumbers version cannot be empty")
        body = self.model_dump(mode="json", exclude={"contentSha256"})
        if sha256_bytes(canonical_json_bytes(body)) != self.contentSha256:
            raise ValueError("certification reference receipt content SHA-256 is invalid")
        return self


@dataclass(frozen=True, slots=True)
class InternationalPhoneReference:
    """One explicit international number and all ISO regions sharing its calling code."""

    surface: str
    calling_code: int
    region_codes: frozenset[str]


def _surface_key(value: str) -> str:
    normalized = "".join(
        character.casefold() if character.isalnum() else " " for character in value
    )
    return " ".join(normalized.split())


def _safe_locality_key(value: str) -> str | None:
    key = _surface_key(value)
    tokens = key.split()
    if not tokens or not any(character.isalpha() for character in key):
        return None
    # Short one-word place names are too collision-prone in arbitrary address and e-mail text.
    # Multi-word names retain their token boundaries, while a one-word name needs five letters.
    if len(tokens) == 1 and len(tokens[0]) < 5:
        return None
    if len(key.replace(" ", "")) < 5:
        return None
    return key


def _calling_code_projection() -> dict[str, list[str]]:
    return {
        str(code): sorted(region for region in regions if re.fullmatch(r"[A-Z]{2}|001", region))
        for code, regions in sorted(COUNTRY_CODE_TO_REGION_CODE.items())
    }


def _parse_iso_country_aliases(payload: bytes) -> dict[str, frozenset[str]]:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ISO-3166 snapshot is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"3166-1"} or not isinstance(raw["3166-1"], list):
        raise ValueError("ISO-3166 snapshot has an unexpected root contract")
    aliases: dict[str, set[str]] = defaultdict(set)
    seen_codes: set[str] = set()
    for index, raw_row in enumerate(raw["3166-1"]):
        if not isinstance(raw_row, dict):
            raise ValueError(f"ISO-3166 record {index} is not an object")
        code = raw_row.get("alpha_2")
        if not isinstance(code, str) or re.fullmatch(r"[A-Z]{2}", code) is None:
            raise ValueError(f"ISO-3166 record {index} has an invalid alpha-2 code")
        if code in seen_codes:
            raise ValueError(f"ISO-3166 snapshot repeats alpha-2 code {code}")
        seen_codes.add(code)
        names = tuple(
            value
            for key in ("name", "official_name")
            if isinstance((value := raw_row.get(key)), str) and value.strip()
        )
        if not names:
            raise ValueError(f"ISO-3166 record {index} lacks a country name")
        for name in names:
            key = _surface_key(name)
            if len(key.replace(" ", "")) >= 4:
                aliases[key].add(code)
    if len(seen_codes) < 200:
        raise ValueError("ISO-3166 snapshot contains implausibly few countries")
    return {key: frozenset(values) for key, values in aliases.items()}


def _locality_projection(rows: Sequence[LocalityRecord]) -> list[dict[str, Any]]:
    return [
        {
            "geonameId": row.geoname_id,
            "canonicalName": row.canonical_name,
            "asciiName": row.ascii_name,
            "countryCode": row.country_code,
        }
        for row in rows
    ]


class CertificationReferenceIndex:
    """Immutable, receipt-bound lookup for narrowly source-owned reference checks."""

    def __init__(
        self,
        *,
        receipt: CertificationReferenceReceipt,
        country_aliases: Mapping[str, frozenset[str]],
        locality_aliases: Mapping[str, frozenset[int]],
        locality_countries: Mapping[int, str],
        package_display_names: Mapping[str, str],
    ) -> None:
        self._receipt = receipt
        self._country_aliases = MappingProxyType(dict(country_aliases))
        self._locality_aliases = MappingProxyType(dict(locality_aliases))
        self._locality_countries = MappingProxyType(dict(locality_countries))
        self._package_display_names = MappingProxyType(dict(package_display_names))
        self._country_lengths = tuple(
            sorted({len(key.split()) for key in country_aliases}, reverse=True)
        )
        self._locality_lengths = tuple(
            sorted({len(key.split()) for key in locality_aliases}, reverse=True)
        )

    @property
    def receipt(self) -> CertificationReferenceReceipt:
        return self._receipt

    @staticmethod
    def _embedded_keys(
        value: str, *, lengths: Sequence[int], keys: Mapping[str, object]
    ) -> tuple[str, ...]:
        tokens = _surface_key(value).split()
        matches: set[str] = set()
        for length in lengths:
            if length > len(tokens):
                continue
            for start in range(len(tokens) - length + 1):
                candidate = " ".join(tokens[start : start + length])
                if candidate in keys:
                    matches.add(candidate)
        return tuple(sorted(matches, key=lambda key: (-len(key.split()), -len(key), key)))

    def exact_country_codes(self, value: str) -> frozenset[str]:
        return self._country_aliases.get(_surface_key(value), frozenset())

    def country_codes_in_text(self, value: str) -> frozenset[str]:
        return frozenset(
            code
            for key in self._embedded_keys(
                value, lengths=self._country_lengths, keys=self._country_aliases
            )
            for code in self._country_aliases[key]
        )

    def exact_locality_ids(self, value: str) -> frozenset[int]:
        key = _safe_locality_key(value)
        return self._locality_aliases.get(key, frozenset()) if key is not None else frozenset()

    def locality_country_codes(self, identifiers: Iterable[int]) -> frozenset[str]:
        try:
            return frozenset(self._locality_countries[identifier] for identifier in identifiers)
        except KeyError as error:
            raise ValueError(f"unknown certification locality identity: {error.args[0]}") from error

    def locality_ids_in_text(
        self, value: str, *, country_codes: frozenset[str] | None = None
    ) -> frozenset[int]:
        matches = self._embedded_keys(
            value, lengths=self._locality_lengths, keys=self._locality_aliases
        )
        identifiers = {
            identifier
            for key in matches
            for identifier in self._locality_aliases[key]
            if country_codes is None or self._locality_countries[identifier] in country_codes
        }
        return frozenset(identifiers)

    def international_phones(self, value: str) -> tuple[InternationalPhoneReference, ...]:
        output: list[InternationalPhoneReference] = []
        for match in _INTERNATIONAL_PHONE.finditer(value):
            surface = match.group(0).strip()
            try:
                parsed = phonenumbers.parse(surface, None)
            except NumberParseException:
                continue
            calling_code = parsed.country_code
            if calling_code is None:
                continue
            regions = COUNTRY_CODE_TO_REGION_CODE.get(calling_code)
            if not regions:
                continue
            output.append(
                InternationalPhoneReference(
                    surface=surface,
                    calling_code=calling_code,
                    region_codes=frozenset(region for region in regions if region != "001"),
                )
            )
        return tuple(output)

    def package_display_name(self, category_token: str) -> str:
        """Return the exact display surface owned by the pinned package registry."""

        try:
            return self._package_display_names[category_token]
        except KeyError as error:
            raise ValueError(
                f"package category is absent from certification references: {category_token!r}"
            ) from error


def compile_certification_reference_index(
    *,
    iso3166_payload: bytes,
    expected_iso3166_sha256: str,
    locality_records: Sequence[LocalityRecord],
    geonames_registry_receipt_sha256: str,
    geonames_registry_content_sha256: str,
    geonames_jsonl_sha256: str,
    expected_phonenumberslite_version: str,
    package_registry: LoadedPackageRegistry,
) -> CertificationReferenceIndex:
    """Compile a receipt-bound index from inputs already validated by their owners."""

    if sha256_bytes(iso3166_payload) != expected_iso3166_sha256:
        raise ValueError("ISO-3166 snapshot SHA-256 mismatch")
    country_aliases = _parse_iso_country_aliases(iso3166_payload)
    rows = tuple(locality_records)
    if not rows:
        raise ValueError("certification reference index requires GeoNames localities")
    identifiers = tuple(row.geoname_id for row in rows)
    if identifiers != tuple(sorted(set(identifiers))):
        raise ValueError("certification GeoNames locality records are duplicated or unsorted")

    locality_aliases: dict[str, set[int]] = defaultdict(set)
    locality_countries: dict[int, str] = {}
    for row in rows:
        locality_countries[row.geoname_id] = row.country_code
        for surface in (row.canonical_name, row.ascii_name):
            key = _safe_locality_key(surface)
            if key is not None:
                locality_aliases[key].add(row.geoname_id)
    if not locality_aliases:
        raise ValueError("certification reference index has no safely matchable localities")

    installed_version = importlib.metadata.version("phonenumberslite")
    if installed_version != expected_phonenumberslite_version:
        raise ValueError(
            "installed phonenumberslite version differs from certification pin: "
            f"{installed_version!r} != {expected_phonenumberslite_version!r}"
        )
    calling_codes = _calling_code_projection()
    package_entries = package_registry.payload.entries
    package_display_names = {row.categoryToken: row.displayName for row in package_entries}
    if len(package_display_names) != len(package_entries):
        raise ValueError("certification package registry repeats a category token")
    locality_projection_sha256 = sha256_bytes(canonical_json_bytes(_locality_projection(rows)))
    receipt_body = {
        "schemaVersion": 2,
        "iso3166Sha256": expected_iso3166_sha256,
        "geonamesRegistryReceiptSha256": geonames_registry_receipt_sha256,
        "geonamesRegistryContentSha256": geonames_registry_content_sha256,
        "geonamesJsonlSha256": geonames_jsonl_sha256,
        "geonamesRecords": len(rows),
        "localityProjectionSha256": locality_projection_sha256,
        "phonenumbersDistribution": "phonenumberslite",
        "phonenumbersVersion": installed_version,
        "callingCodeRegionMapSha256": sha256_bytes(canonical_json_bytes(calling_codes)),
        "packageRegistrySha256": package_registry.sha256,
        "packageRegistryEntries": len(package_entries),
        "packageRegistryPayloadSha256": sha256_bytes(
            canonical_json_bytes(package_registry.payload.model_dump(mode="json"))
        ),
    }
    receipt = CertificationReferenceReceipt.model_validate(
        {
            **receipt_body,
            "contentSha256": sha256_bytes(canonical_json_bytes(receipt_body)),
        },
        strict=True,
    )
    return CertificationReferenceIndex(
        receipt=receipt,
        country_aliases=country_aliases,
        locality_aliases={key: frozenset(values) for key, values in locality_aliases.items()},
        locality_countries=locality_countries,
        package_display_names=package_display_names,
    )


def _read_pinned_iso(path: Path, *, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"ISO-3166 snapshot is not a readable regular file: {path}")
    payload = path.read_bytes()
    if not payload or len(payload) > _MAX_ISO_BYTES:
        raise ValueError("ISO-3166 snapshot violates byte bounds")
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError("ISO-3166 snapshot SHA-256 mismatch")
    return payload


def load_certification_reference_index(
    *,
    iso3166_path: Path,
    expected_iso3166_sha256: str,
    geonames_registry_root: Path,
    expected_geonames_receipt_sha256: str,
    expected_phonenumberslite_version: str,
    package_registry_path: Path,
    expected_package_registry_sha256: str,
    expected_package_registry_entries: int,
) -> CertificationReferenceIndex:
    """Load and cross-bind the exact production reference inputs once per run."""

    iso_payload = _read_pinned_iso(iso3166_path, expected_sha256=expected_iso3166_sha256)
    registry: GeoNamesLocalityRegistry = load_geonames_locality_registry(
        root=geonames_registry_root,
        expected_receipt_sha256=expected_geonames_receipt_sha256,
    )
    receipt = registry.receipt
    if receipt.iso3166_sha256 != expected_iso3166_sha256:
        raise ValueError("GeoNames registry ISO-3166 identity differs from certification input")
    rows = tuple(
        row
        for country_code in registry.country_codes
        for row in registry.rows_for_country(country_code)
    )
    rows = tuple(sorted(rows, key=lambda row: row.geoname_id))
    package_registry = load_package_registry(
        package_registry_path,
        expected_sha256=expected_package_registry_sha256,
        expected_entries=expected_package_registry_entries,
    )
    return compile_certification_reference_index(
        iso3166_payload=iso_payload,
        expected_iso3166_sha256=expected_iso3166_sha256,
        locality_records=rows,
        geonames_registry_receipt_sha256=expected_geonames_receipt_sha256,
        geonames_registry_content_sha256=receipt.content_sha256,
        geonames_jsonl_sha256=receipt.jsonl_sha256,
        expected_phonenumberslite_version=expected_phonenumberslite_version,
        package_registry=package_registry,
    )
