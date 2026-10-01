import hashlib
import json
import zipfile
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import address_localities
from document_ocr.synthesis.template_compiler.address_localities import (
    AddressLocalities,
    normalized,
)
from document_ocr.synthesis.template_compiler.complete_pipeline import (
    _address_repair_requirements,
    _generation_schema,
    _validate_address_repairs,
)
from document_ocr.synthesis.template_compiler.request_batches import lexical_payload
from document_ocr.synthesis.template_compiler.route_projection import AddressLocalityConflict


def index():
    return AddressLocalities(
        {
            "CN": {"yishui": frozenset({1}), "shenzhen": frozenset({2})},
            "NL": {"spijkenisse": frozenset({3})},
        }
    )


def test_explicit_city_conflict_is_detected_without_matching_street_names():
    data = index()
    assert data.conflicts(
        "16 Qianhai Road, 518000 Shenzhen, China", country="CN", city="Yishui"
    ) == ("Shenzhen",)
    assert not data.conflicts("16 Shenzhen Road, Yishui, China", country="CN", city="Yishui")
    assert not data.conflicts(
        "41 Havenstraat, 3201 Spijkenisse, Netherlands", country="NL", city="Spijkenisse"
    )


def test_competing_city_in_hyphenated_district_or_after_expected_city_is_rejected():
    data = AddressLocalities(
        {
            "EG": {
                normalized("6th of October City"): frozenset({1}),
                normalized("Sheikh Zayed"): frozenset({2}),
            }
        }
    )
    assert data.conflicts(
        "Oasis Heights Street - Building No. 12B\nDistrict 7th - Sheikh Zayed",
        country="EG",
        city="6th of October City",
    ) == ("Sheikh Zayed",)
    assert data.conflicts(
        "Oasis Heights Street - Sheikh Zayed",
        country="EG",
        city="6th of October City",
    ) == ("Sheikh Zayed",)
    assert not data.conflicts(
        "Sheikh Zayed Street",
        country="EG",
        city="6th of October City",
    )


def test_source_pinned_city_alternates_share_identity() -> None:
    rows = [
        NS(
            canonical_name="6th of October City",
            ascii_name="6th of October City",
            geoname_id=1,
            country_code="EG",
            feature_code="PPL",
        ),
        NS(
            canonical_name="Ash-Shaykh Zayid",
            ascii_name="Ash-Shaykh Zayid",
            geoname_id=2,
            country_code="EG",
            feature_code="PPL",
        ),
    ]
    data = AddressLocalities.from_registry(
        NS(country_codes=["EG"], rows_for_country=lambda _: rows, entry=lambda i: rows[i - 1]),
        alternate_names=((2, "Sheikh Zayed City"), (2, "Sheikh Zayed")),
    )
    assert data.conflicts(
        "Road\nDistrict 7th - Sheikh Zayed",
        country="EG",
        city="6th of October City",
    ) == ("Sheikh Zayed",)
    assert normalized("6th of October City") != normalized("10th of October City")
    assert data.conflicts(
        "12 Oasis Street, 6th of October City, Sheikh Zayed, Egypt",
        country="EG",
        city="6th of October City",
    ) == ("Sheikh Zayed",)


def test_route_pinned_raw_city_aliases_are_loaded_and_hash_checked(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()
    archive = root / "cities15000.zip"
    parts = [""] * 19
    parts[0:4] = ["2", "Ash-Shaykh Zayid", "Ash-Shaykh Zayid", "Sheikh Zayed City"]
    parts[8] = "EG"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("cities15000.txt", "\t".join(parts) + "\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    route = root / "route"
    route.mkdir()
    (root / "compiled").mkdir()
    (route / "transaction.json").write_text(
        json.dumps(
            {
                "config": {
                    "locality_registry": {"path": "compiled", "manifest_sha256": "receipt"},
                    "geonames_raw_cities": {"path": archive.name, "sha256": digest},
                }
            }
        )
    )
    admin1 = root / "admin1.txt"
    admin1.write_text("")
    row = NS(
        geoname_id=2,
        canonical_name="Ash-Shaykh Zayid",
        ascii_name="Ash-Shaykh Zayid",
        country_code="EG",
        feature_code="PPL",
    )
    registry = NS(
        country_codes=("EG",),
        rows_for_country=lambda _: (row,),
        entry=lambda identifier: row if identifier == 2 else None,
        receipt=NS(archive_sha256=digest, locality_audit=NS(accepted_records=1)),
    )

    def load_registry(*, root, expected_receipt_sha256):
        assert root == tmp_path / "compiled"
        assert expected_receipt_sha256 == "receipt"
        return registry

    monkeypatch.setattr(address_localities, "load_geonames_locality_registry", load_registry)
    index = address_localities.load_address_localities(root, route, admin1)
    assert normalized("Sheikh Zayed") in index.names["EG"]

    (route / "transaction.json").write_text(
        json.dumps(
            {
                "config": {
                    "locality_registry": {"path": "compiled", "manifest_sha256": "receipt"},
                    "geonames_raw_cities": {"path": archive.name, "sha256": "0" * 64},
                }
            }
        )
    )
    with pytest.raises(ValueError, match="archive differs from its pin"):
        address_localities.load_address_localities(root, route, admin1)


def test_locality_ascii_aliases_share_identity():
    rows = [NS(canonical_name="Liège", ascii_name="Liege", geoname_id=1, feature_code="PPL")]
    data = AddressLocalities.from_registry(
        NS(country_codes=["BE"], rows_for_country=lambda _: rows)
    )
    assert not data.conflicts("8 Road, Liege, Belgium", country="BE", city="Liège")


def test_city_repair_schema_and_error_retain_both_authoritative_city_and_country():
    fields = [dict(key="address", paths=["documentPatch.parties.shipper.address"])]
    error = AddressLocalityConflict(
        "documentPatch.parties.shipper", "China", "Yishui", ("Shenzhen",)
    )
    requirements = _address_repair_requirements(fields, [], [], error)
    schema = _generation_schema(fields, requirements)
    # The native decoder must not be trapped in a free-text suffix grammar.
    assert "pattern" not in schema.model_json_schema()["properties"]["address"]
    with pytest.raises(ValueError):
        _validate_address_repairs(dict(address="16 Qianhai Road, Shenzhen, China"), requirements)
    schema.model_validate(dict(address="16 Qianhai Road, Yishui, China"))
    _validate_address_repairs(dict(address="16 Qianhai Road, Yishui, China"), requirements)
    with pytest.raises(ValueError, match="without removing"):
        _validate_address_repairs(
            dict(address="16 Qianhai Road, Shenzhen, Yishui, China"), requirements
        )
    assert (
        _address_repair_requirements(fields, [], requirements, ValueError("another field"))
        == requirements
    )


def test_local_address_validation_requires_city_country_and_allows_intervening_region():
    requirements = [
        dict(key="address", obligations=[dict(terminalCountry="China", terminalCity="Yishui")])
    ]
    for address in ["16 Road, Yishui, China", "16 Road, Yishui, Shandong, China"]:
        _validate_address_repairs(dict(address=address), requirements)
    for address in [
        "16 Road, Yishui, China extra",
        "16 Road, Other, China",
        "16 Road\nYishui, China",
    ]:
        with pytest.raises(ValueError, match="exact requested"):
            _validate_address_repairs(dict(address=address), requirements)


def test_fixed_address_prefix_and_country_repair_are_independent_constraints():
    fields = [dict(key="address")]
    requirements = [
        dict(
            key="address",
            obligations=[
                dict(partitionPattern=r"^NO\.\S[^\r\n]+$"),
                dict(terminalCountry="United States"),
            ],
        )
    ]
    schema = _generation_schema(fields, requirements)
    value = dict(address="NO.8 Harbor Road, Tampa, United States")
    schema.model_validate(value)
    _validate_address_repairs(value, requirements)
    with pytest.raises(ValueError):
        schema.model_validate(dict(address="8 Harbor Road, Tampa, United States"))
    with pytest.raises(ValueError):
        _validate_address_repairs(dict(address="NO.8 Harbor Road, Paris, France"), requirements)


def test_correct_country_before_postcode_is_not_rejected_as_wrong_geography():
    requirements = [
        dict(
            key="address",
            obligations=[
                dict(terminalCountry="Canada", terminalCity="Le Sud-Ouest"),
            ],
        )
    ]
    _validate_address_repairs(
        dict(address="19 Rue des Forges, Le Sud-Ouest, Canada H4E 6M2"), requirements
    )
    for address in ["19 Road, Le Sud-Ouest, France 12345", "19 Road, Montreal, Canada H4E 6M2"]:
        with pytest.raises(ValueError):
            _validate_address_repairs(dict(address=address), requirements)


def test_region_country_and_nested_localities_are_not_mislabeled_as_conflicts():
    data = AddressLocalities(
        {"JP": {"tokyo": frozenset({1}), "denenchofu": frozenset({2})}},
        {"JP": frozenset({"tokyo"})},
    )
    assert not data.conflicts(
        "14 Sakura Road, Den'enchofu, Tokyo 145-0071",
        country="JP",
        city="Den'enchofu",
        country_name="Japan",
    )
    assert not data.conflicts(
        "14 Sakura Road, Tokyo 145-0071", country="JP", city="Den'enchofu", country_name="Japan"
    )
    assert not data.conflicts(
        "Port Road, Tokyo", country="JP", city="Tokyo Pt", country_name="Japan"
    )


def test_address_request_carries_its_own_authoritative_geography():
    geo = dict(city="Yishui", country_code="CN", country_name="China")
    payload = dict(
        structuredScenario={"documentPatch": {"parties": {"shipper": {"address": "old"}}}},
        partyGeography={"documentPatch.parties.shipper": geo},
        requestedFields=[
            dict(
                key="a",
                paths=["documentPatch.parties.shipper.address"],
                source="old",
                constraints=[],
            )
        ],
    )
    result = lexical_payload(payload, {"a": "shipper_address"})
    assert result["requestedFields"][0]["expectedGeography"] == geo
    assert "expectedGeography" not in payload["requestedFields"][0]
