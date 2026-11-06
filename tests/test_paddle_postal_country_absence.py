from pathlib import Path

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.paddle_render import project_printed_postal_country

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_country_absence_authority_does_not_infer_from_planning():
    manifest = "artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/production-manifest.json"
    action = {
        "target": "/parties/shipper/country",
        "addressTarget": "/parties/shipper/addressLine",
        "codePath": "/origin/country_code",
        "manifestPath": manifest,
        "sha256": sha256_file(ROOT / manifest),
        "omitIfUnprinted": True,
    }
    candidate = {"scenario": {"origin": {"country_code": "US"}}}
    target = {"parties": {"shipper": {"addressLine": "NEW YORK, NY", "country": "UNITED STATES"}}}
    project_printed_postal_country(target, candidate, action, ROOT)
    assert target == {"parties": {"shipper": {"addressLine": "NEW YORK, NY"}}}
    project_printed_postal_country(target, candidate, action, ROOT)
    assert "country" not in target["parties"]["shipper"]
    target["parties"]["shipper"]["addressLine"] += ", USA"
    project_printed_postal_country(target, candidate, action, ROOT)
    assert target["parties"]["shipper"]["country"] == "UNITED STATES"


def test_district_country_substring_is_not_an_independent_postal_country():
    manifest = "artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/production-manifest.json"
    action = {
        "target": "/parties/shipper/country",
        "addressTarget": "/parties/shipper/addressLine",
        "codePath": "/origin/country_code",
        "manifestPath": manifest,
        "sha256": sha256_file(ROOT / manifest),
        "omitIfUnprinted": True,
    }
    candidate = {"scenario": {"origin": {"country_code": "BN"}}}
    for district in ("BRUNEI-MUARA", "Brunei Muara", "BRUNEI–MUARA"):
        address = f"LOT 45, MENTIRI, {district} DISTRICT"
        target = {"parties": {"shipper": {"addressLine": address, "country": "BRUNEI"}}}
        project_printed_postal_country(target, candidate, action, ROOT)
        assert target == {"parties": {"shipper": {"addressLine": address}}}
        for country in ("BRUNEI", "BRUNEI DARUSSALAM", "BN"):
            target["parties"]["shipper"]["addressLine"] = address + ", " + country
            project_printed_postal_country(target, candidate, action, ROOT)
            assert target["parties"]["shipper"]["country"] == "BRUNEI"
