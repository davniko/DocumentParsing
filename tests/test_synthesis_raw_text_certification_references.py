from __future__ import annotations

import json
from decimal import Decimal

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.locality_registry import LocalityRecord
from document_ocr.synthesis.package_registry import (
    LoadedPackageRegistry,
    PackageRegistryEntry,
    PackageRegistryPayload,
)
from document_ocr.synthesis.raw_text_certification_references import (
    CertificationReferenceIndex,
    compile_certification_reference_index,
)


def _iso_payload() -> bytes:
    reserved = {"EG", "SE", "FI", "AX"}
    rows = [
        {"alpha_2": "EG", "alpha_3": "EGY", "name": "Egypt", "numeric": "818"},
        {"alpha_2": "SE", "alpha_3": "SWE", "name": "Sweden", "numeric": "752"},
        {"alpha_2": "FI", "alpha_3": "FIN", "name": "Finland", "numeric": "246"},
        {"alpha_2": "AX", "alpha_3": "ALA", "name": "Åland Islands", "numeric": "248"},
    ]
    for first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for second in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            code = first + second
            if code in reserved:
                continue
            rows.append(
                {
                    "alpha_2": code,
                    "alpha_3": f"X{first}{second}",
                    "name": f"Test Country {code}",
                    "numeric": f"{len(rows):03d}",
                }
            )
            if len(rows) == 200:
                return json.dumps({"3166-1": rows}, ensure_ascii=False).encode("utf-8")
    raise AssertionError("test ISO fixture did not reach 200 countries")


def _localities() -> tuple[LocalityRecord, ...]:
    return (
        LocalityRecord(
            geoname_id=1,
            canonical_name="Damietta",
            ascii_name="Damietta",
            country_code="EG",
            feature_code="PPLA",
            population=200_000,
            latitude=Decimal("31.4165"),
            longitude=Decimal("31.8133"),
            timezone="Africa/Cairo",
        ),
        LocalityRecord(
            geoname_id=2,
            canonical_name="Kalmar",
            ascii_name="Kalmar",
            country_code="SE",
            feature_code="PPLA",
            population=41_000,
            latitude=Decimal("56.6616"),
            longitude=Decimal("16.3616"),
            timezone="Europe/Stockholm",
        ),
        LocalityRecord(
            geoname_id=3,
            canonical_name="Ystad",
            ascii_name="Ystad",
            country_code="SE",
            feature_code="PPLA3",
            population=20_000,
            latitude=Decimal("55.4297"),
            longitude=Decimal("13.8204"),
            timezone="Europe/Stockholm",
        ),
    )


def _packages() -> LoadedPackageRegistry:
    payload = PackageRegistryPayload(
        schemaVersion=1,
        registryKind="package",
        sourceAuthority="test",
        sourceRevision="1",
        sourcePath="test.json",
        sourceSha256="4" * 64,
        entries=(
            PackageRegistryEntry(
                categoryToken="PACKAGE_VEHICLE",
                applicationCode="VN",
                displayName="Vehicle",
            ),
        ),
    )
    return LoadedPackageRegistry(path="test.json", sha256="5" * 64, payload=payload)


def _index() -> CertificationReferenceIndex:
    payload = _iso_payload()
    return compile_certification_reference_index(
        iso3166_payload=payload,
        expected_iso3166_sha256=sha256_bytes(payload),
        locality_records=_localities(),
        geonames_registry_receipt_sha256="1" * 64,
        geonames_registry_content_sha256="2" * 64,
        geonames_jsonl_sha256="3" * 64,
        expected_phonenumberslite_version="9.0.38",
        package_registry=_packages(),
    )


def test_reference_index_resolves_only_pinned_country_and_locality_identities() -> None:
    index = _index()

    assert index.exact_country_codes("Sweden") == frozenset({"SE"})
    assert index.country_codes_in_text("Ystad; Sweden") == frozenset({"SE"})
    assert index.exact_locality_ids("Damietta") == frozenset({1})
    assert index.locality_ids_in_text("Email kalmar@nordlinkship.com") == frozenset({2})
    assert index.locality_ids_in_text("Kalmar Port", country_codes=frozenset({"EG"})) == frozenset()


def test_reference_index_reads_calling_code_regions_without_validity_guessing() -> None:
    index = _index()

    finnish = index.international_phones("Tel : +35819412288")
    swedish = index.international_phones("Fax: +46 40 123 456")

    assert [(row.calling_code, row.region_codes) for row in finnish] == [
        (358, frozenset({"FI", "AX"}))
    ]
    assert [(row.calling_code, row.region_codes) for row in swedish] == [(46, frozenset({"SE"}))]
    assert index.package_display_name("PACKAGE_VEHICLE") == "Vehicle"


def test_reference_receipt_rejects_a_different_dependency_version_or_iso_bytes() -> None:
    payload = _iso_payload()

    with pytest.raises(ValueError, match="installed phonenumberslite version differs"):
        compile_certification_reference_index(
            iso3166_payload=payload,
            expected_iso3166_sha256=sha256_bytes(payload),
            locality_records=_localities(),
            geonames_registry_receipt_sha256="1" * 64,
            geonames_registry_content_sha256="2" * 64,
            geonames_jsonl_sha256="3" * 64,
            expected_phonenumberslite_version="0.0.0",
            package_registry=_packages(),
        )
    with pytest.raises(ValueError, match="ISO-3166 snapshot SHA-256 mismatch"):
        compile_certification_reference_index(
            iso3166_payload=payload,
            expected_iso3166_sha256="0" * 64,
            locality_records=_localities(),
            geonames_registry_receipt_sha256="1" * 64,
            geonames_registry_content_sha256="2" * 64,
            geonames_jsonl_sha256="3" * 64,
            expected_phonenumberslite_version="9.0.38",
            package_registry=_packages(),
        )


def test_reference_index_rejects_unknown_package_category() -> None:
    with pytest.raises(ValueError, match="absent from certification references"):
        _index().package_display_name("PACKAGE_NOT_PINNED")
