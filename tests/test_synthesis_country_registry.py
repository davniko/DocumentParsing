from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from document_ocr.synthesis.country_registry import (
    CountryEntry,
    CountryRegistry,
    load_country_registry,
    load_iso_country_registry,
    normalize_country_alias,
)


def _json(path: Path, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _entry(code: str, alpha3: str, numeric: str, name: str) -> CountryEntry:
    return CountryEntry.model_validate(
        {"alpha2": code, "alpha3": alpha3, "numeric": numeric, "name": name},
        strict=True,
    )


def test_registry_resolves_reviewed_aliases_without_place_name_guessing() -> None:
    registry = CountryRegistry(
        entries=(
            _entry("EG", "EGY", "818", "Egypt"),
            _entry("TR", "TUR", "792", "Türkiye"),
        ),
        observed_aliases={"A.R. EGYPT": "EG", "TURKIYE": "TR"},
        iso_sha256="1" * 64,
        observed_aliases_sha256="2" * 64,
    )
    assert registry.resolve("  A.R. ÉGYPT  ") == "EG"
    assert registry.resolve("turkiye") == "TR"
    assert registry.resolve("Alexandria") is None
    assert registry.printable_name("TR") == "Türkiye"
    with pytest.raises(ValueError, match="unresolved country"):
        registry.require("Alexandria", field="route.portOfDischarge.country")
    assert normalize_country_alias("U.A.E.") == "UAE"


def test_registry_rejects_ambiguous_or_unknown_reviewed_aliases() -> None:
    entries = (
        _entry("CD", "COD", "180", "Congo, The Democratic Republic of the"),
        _entry("CG", "COG", "178", "Congo"),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        CountryRegistry(
            entries=entries,
            observed_aliases={"Congo": "CD"},
            iso_sha256="1" * 64,
            observed_aliases_sha256="2" * 64,
        )
    with pytest.raises(ValueError, match="absent ISO code"):
        CountryRegistry(
            entries=entries,
            observed_aliases={"Unknown": "ZZ"},
            iso_sha256="1" * 64,
            observed_aliases_sha256="2" * 64,
        )


def test_pinned_loader_checks_contract_hashes_and_unknown_fields(tmp_path: Path) -> None:
    iso_path = tmp_path / "iso.json"
    iso_sha = _json(
        iso_path,
        {
            "3166-1": [
                {
                    "alpha_2": "EG",
                    "alpha_3": "EGY",
                    "flag": "x",
                    "name": "Egypt",
                    "numeric": "818",
                    "official_name": "Arab Republic of Egypt",
                }
            ]
        },
    )
    aliases_path = tmp_path / "aliases.json"
    aliases_sha = _json(
        aliases_path,
        {
            "iso3166SnapshotPath": str(iso_path),
            "iso3166SnapshotSha256": iso_sha,
            "normalization": "NFKD ASCII, uppercase alphanumeric words",
            "observedMappings": {"EGYPT": "EG"},
        },
    )
    registry = load_country_registry(
        iso_path=iso_path,
        iso_sha256=iso_sha,
        observed_aliases_path=aliases_path,
        observed_aliases_sha256=aliases_sha,
    )
    assert registry.resolve("Arab Republic of Egypt") == "EG"
    assert registry.audit.iso_records == 1

    with pytest.raises(ValueError, match="reference a different ISO-3166 snapshot"):
        load_country_registry(
            iso_path=iso_path,
            iso_sha256="0" * 64,
            observed_aliases_path=aliases_path,
            observed_aliases_sha256=aliases_sha,
        )

    bad_path = tmp_path / "bad.json"
    bad_sha = _json(
        bad_path,
        {
            "3166-1": [
                {
                    "alpha_2": "EG",
                    "alpha_3": "EGY",
                    "numeric": "818",
                    "name": "Egypt",
                    "unexpected": True,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="unexpected fields"):
        load_iso_country_registry(
            iso_path=bad_path,
            iso_sha256=bad_sha,
        )
