"""Source-pinned route restrictions survive sampling and registry joins."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.template_compiler.route_source_constraints import (
    RouteAdmissibilityCertificate,
    load_admin1_membership,
    requires_certificate,
    validate_scenario,
)


def _certificate() -> RouteAdmissibilityCertificate:
    return RouteAdmissibilityCertificate.model_validate(
        {
            "schema_version": 1,
            "source_document_id": "source",
            "source_sha256": "a" * 64,
            "source_label_sha256": "b" * 64,
            "template_sha256": "c" * 64,
            "commercial_origin_country_code": "BR",
            "loading_country_code": "BR",
            "shipper_country_code": "BR",
            "freight_arrangement": "prepaid",
            "source_loading_port_locode": "BRITJ",
            "loading_country_clause": {
                "byte_start": 10,
                "byte_end": 20,
                "source_text": "source law",
            },
        },
        strict=True,
    )


def test_country_specific_loading_policy_requires_certification():
    source = (
        b"Container detention tariffs and conditions applicable for the port of loading, "
        b"including free time, at:\n"
        b"https://www.maersk.com/local-information/latin-america/brazil/export"
    )
    assert requires_certificate(source)
    assert not requires_certificate(b"Generic terms and conditions for carriage.")


def test_sampled_route_and_freight_cannot_oppose_the_source_certificate():
    scenario = {
        "commercialOriginCountryCode": "BR",
        "loadingPort": {"countryCode": "BR"},
        "freight": {"arrangement": "prepaid"},
        "partyLocalities": [{"role": "shipper", "occurrence": 0, "conditioningCountryCode": "BR"}],
    }
    validate_scenario(_certificate(), scenario)
    for field, value in (
        ("commercialOriginCountryCode", "CN"),
        ("loadingPort", {"countryCode": "CN"}),
        ("freight", {"arrangement": "collect"}),
        (
            "partyLocalities",
            [{"role": "shipper", "occurrence": 0, "conditioningCountryCode": "CN"}],
        ),
    ):
        changed = {**scenario, field: value}
        with pytest.raises(ValueError, match="source-certified"):
            validate_scenario(_certificate(), changed)


def test_pinned_admin1_join_uses_city_identity_and_proved_subdivision_name(tmp_path: Path):
    cities = tmp_path / "cities.zip"
    with zipfile.ZipFile(cities, "w") as archive:
        columns = ["12345", "Example", "Example", "", "", "", "", "", "BR", "", "26"]
        archive.writestr("cities15000.txt", "\t".join(columns) + "\n")
    admin = tmp_path / "admin1.txt"
    admin.write_text("BR.26\tSanta Catarina\tSanta Catarina\t3450387\n")
    unlocode = tmp_path / "unlocode.zip"
    with zipfile.ZipFile(unlocode, "w") as archive:
        archive.writestr("release/csv/SubdivisionCodes.csv", "BR,SC,Santa Catarina,State\n")
    membership = load_admin1_membership(
        cities_archive=cities,
        cities_sha256=sha256_file(cities),
        admin1_codes=admin,
        admin1_sha256=sha256_file(admin),
        unlocode_archive=unlocode,
        unlocode_sha256=sha256_file(unlocode),
    )
    assert membership.geoname_admin1[12345] == "BR.26"
    assert membership.subdivision_admin1["BR", "SC"] == "BR.26"
    assert membership.region_for_locality(
        {
            "countryCode": "BR",
            "geonameId": 12345,
            "subdivisionCode": None,
            "source": "geonames_population_weighted",
        }
    ) == ("BR.26", "Santa Catarina")
    assert membership.region_for_locality(
        {
            "countryCode": "BR",
            "geonameId": None,
            "subdivisionCode": "SC",
            "source": "endpoint",
        }
    ) == ("BR.26", "Santa Catarina")
    with pytest.raises(ValueError, match="conflicting identities"):
        membership.region_for_locality(
            {
                "countryCode": "BR",
                "geonameId": 12345,
                "subdivisionCode": "SC",
                "source": "geonames_population_weighted",
            }
        )
    with pytest.raises(ValueError, match="hash differs"):
        load_admin1_membership(
            cities_archive=cities,
            cities_sha256="0" * 64,
            admin1_codes=admin,
            admin1_sha256=sha256_file(admin),
            unlocode_archive=unlocode,
            unlocode_sha256=sha256_file(unlocode),
        )
