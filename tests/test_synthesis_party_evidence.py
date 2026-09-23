from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.generation_contract import (
    party_owned_surfaces,
    validate_party_evidence,
)


def check(party, surfaces, bound=()):
    validate_party_evidence(
        target={"documentPatch": {"parties": party}},
        binding_paths=set(bound),
        party_surfaces=surfaces,
    )


def test_notify_values_do_not_populate_an_empty_consignee_block():
    party = {"name": "Example Trading", "city": "Osaka", "country": "Japan"}
    with pytest.raises(ValueError, match=r"parties.consignee.name"):
        check(
            {"consignee": party, "notifyParties": [party]},
            {"documentPatch.parties.notifyParties[0]": ("Example Trading OSAKA JAPAN",)},
        )


def test_city_must_be_visible_but_can_be_embedded_in_its_owned_address():
    party = {"shipper": {"address": "40 Main Road, Osaka, Japan", "city": "Osaka"}}
    check(party, {"documentPatch.parties.shipper": ("40 MAIN ROAD,\nOSAKA, JAPAN",)})
    with pytest.raises(ValueError, match=r"shipper\.city"):
        check(
            party,
            {"documentPatch.parties.shipper": ("40 MAIN ROAD, JAPAN",)},
            ("documentPatch.parties.shipper.address",),
        )


def test_city_evidence_is_not_a_substring_or_a_street_geocoding_inference():
    with pytest.raises(ValueError, match=r"shipper\.city"):
        check(
            {"shipper": {"city": "Ham"}},
            {"documentPatch.parties.shipper": ("40 MAIN ROAD, HAMBURG",)},
        )


def test_country_under_another_party_is_not_evidence():
    with pytest.raises(ValueError, match=r"deliveryAgent\.country"):
        check(
            {"deliveryAgent": {"country": "Japan"}},
            {"documentPatch.parties.shipper": ("OSAKA JAPAN",)},
        )


def test_country_alias_is_supported_but_region_suffix_is_not_country_evidence():
    validate_party_evidence(
        target={"documentPatch": {"parties": {"consignee": {"country": "United States"}}}},
        binding_paths=set(),
        party_surfaces={"documentPatch.parties.consignee": ("12 Main Road, Albuquerque, USA",)},
        country_codes={"unitedstates": "US", "usa": "US"},
    )
    with pytest.raises(ValueError, match=r"consignee\.country"):
        check(
            {"consignee": {"country": "Mexico"}},
            {"documentPatch.parties.consignee": ("12 Main Road, New Mexico",)},
        )
    check(
        {"deliveryAgent": {"country": "Canada", "city": "Midland"}},
        {"documentPatch.parties.deliveryAgent": ("42 Midland Avenue, Canada, Midland",)},
    )


def test_owned_segments_are_joined_only_within_the_binding():
    bindings = (
        NS(
            logical_key="shipper-address",
            target_paths=("documentPatch.parties.shipper.address",),
            occurrences=(NS(slot_id="a"), NS(slot_id="b")),
        ),
        NS(
            logical_key="shared-name",
            target_paths=(
                "documentPatch.parties.consignee.name",
                "documentPatch.parties.notifyParties[0].name",
            ),
            occurrences=(NS(slot_id="c"),),
        ),
    )
    surfaces = party_owned_surfaces(
        bindings, {"a": "10 PORT ROAD, RIO DE", "b": "JANEIRO", "c": "SHARED TRADING"}, ()
    )
    check(
        {
            "shipper": {"city": "Rio de Janeiro"},
            "consignee": {"name": "Shared Trading"},
            "notifyParties": [{"name": "Shared Trading"}],
        },
        surfaces,
    )
    with pytest.raises(KeyError):
        party_owned_surfaces(bindings, {"a": "10 PORT ROAD"}, ())


def test_line_wrapping_does_not_split_a_country_identity():
    check(
        {"consignee": {"country": "United Arab Emirates"}},
        {
            "documentPatch.parties.consignee": (
                "18 AL NAHDA\nCRESCENT, BANI\nYAS CITY, UNITED\nARAB EMIRATES",
            )
        },
    )


def test_explicitly_linked_auxiliary_contact_is_party_owned():
    binding = NS(logical_key="aux-phone", target_paths=(), occurrences=(NS(slot_id="s"),))
    entity = NS(
        relationship="same_as_target_party",
        target_party_path="documentPatch.parties.consignee",
        members=(NS(logical_key="aux-phone"),),
    )
    surfaces = party_owned_surfaces((binding,), {"s": "+20 12345678"}, (entity,))
    check({"consignee": {"contactDetails": {"phoneNumbers": ["+2012345678"]}}}, surfaces)
    entity.relationship = "independent"
    with pytest.raises(ValueError, match="consignee"):
        check(
            {"consignee": {"contactDetails": {"phoneNumbers": ["+2012345678"]}}},
            party_owned_surfaces((binding,), {"s": "+20 12345678"}, (entity,)),
        )


def test_direct_typed_binding_and_carrier_keep_their_existing_proofs():
    check(
        {"shipper": {"country": "JAPAN"}, "carrier": {"name": "Canonical Carrier"}},
        {},
        ("documentPatch.parties.shipper.country",),
    )


def test_unbound_contact_cannot_be_a_substring_of_another_contact():
    for contact, surface in (
        ({"contactName": "Ann"}, "Joann"),
        ({"phoneNumbers": ["12345678"]}, "+20 12345678"),
        ({"emailAddresses": ["ann@example.com"]}, "joann@example.com"),
    ):
        with pytest.raises(ValueError, match="role-owned printed evidence"):
            check(
                {"consignee": {"contactDetails": contact}},
                {"documentPatch.parties.consignee": (surface,)},
            )


def test_order_reference_does_not_require_inventing_a_party():
    check({"consignee": {"name": "TO ORDER OF SHIPPER"}}, {})
