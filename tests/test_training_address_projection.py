from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.generation_contract import (
    validate_compiled_party_contract,
)
from document_ocr.training.address_projection import project_address, project_training_target

COUNTRIES = {
    "egypt": "EG",
    "unitedstates": "US",
    "saudiarabia": "SA",
    "unitedarabemirates": "AE",
    "ireland": "IE",
}


@pytest.mark.parametrize(
    ("address", "city", "country", "expected"),
    [
        (
            "41 Al Waha Industrial Road, Jeddah, Saudi Arabia",
            "Jeddah",
            "Saudi Arabia",
            "41 Al Waha Industrial Road",
        ),
        ("18 El Nasr Street, Cairo 11511, Egypt", "Cairo", "Egypt", "18 El Nasr Street, 11511"),
        ("9 Harbour Lane, Rubidoux, United States", "Rubidoux", "United States", "9 Harbour Lane"),
        (
            "7 East Road, Beni Suef Industrial City",
            "Beni Suef",
            "Egypt",
            "7 East Road, Beni Suef Industrial City",
        ),
        ("23 Rue de Djibouti", "Djibouti", "Djibouti", "23 Rue de Djibouti"),
        (
            "46 Portside Foundry Road Port Sudan",
            "Port Sudan",
            "Sudan",
            "46 Portside Foundry Road Port Sudan",
        ),
    ],
)
def test_project_address_preserves_street_and_site_context(address, city, country, expected):
    assert project_address(address, city, country, COUNTRIES)[0] == expected


def test_training_projection_keeps_the_render_target_frozen():
    render_target = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "Fixed Carrier", "address": "Carrier Lane, Cairo, Egypt"},
                "shipper": {
                    "address": "41 Al Waha Industrial Road, Jeddah, Saudi Arabia",
                    "city": "Jeddah",
                    "country": "Saudi Arabia",
                },
                "notifyParties": [
                    {
                        "address": "9 Harbour Lane, Rubidoux, United States",
                        "city": "Rubidoux",
                        "country": "United States",
                    }
                ],
            }
        }
    }
    frozen = deepcopy(render_target)
    projected, edits = project_training_target(render_target, country_codes=COUNTRIES)
    assert render_target == frozen
    assert (
        projected["documentPatch"]["parties"]["carrier"]
        == frozen["documentPatch"]["parties"]["carrier"]
    )
    assert (
        projected["documentPatch"]["parties"]["shipper"]["address"] == "41 Al Waha Industrial Road"
    )
    assert projected["documentPatch"]["parties"]["notifyParties"][0]["address"] == "9 Harbour Lane"
    assert {edit["path"] for edit in edits} == {
        "documentPatch.parties.shipper.address",
        "documentPatch.parties.notifyParties[0].address",
    }


def test_numeric_postal_address_interleaved_with_city_is_rejected():
    address = NS(
        logical_key="shipper_address",
        group_kind="party",
        value_kind="address",
        group_key="party:shipper:0",
        realization=NS(mode="segmented_surface"),
        occurrences=(
            NS(source_text="8755", byte_start=100, byte_end=104),
            NS(source_text="21492", byte_start=112, byte_end=117),
        ),
        target_paths=("documentPatch.parties.shipper.address",),
    )
    city = NS(
        logical_key="shipper_city",
        group_kind="party",
        value_kind="location",
        group_key="party:shipper:0",
        realization=NS(mode="single_surface"),
        occurrences=(NS(source_text="JEDDAH", byte_start=105, byte_end=111),),
        target_paths=("documentPatch.parties.shipper.city",),
    )
    with pytest.raises(ValueError, match="numeric-only address slots interleave"):
        validate_compiled_party_contract(
            raw=b" " * 200, source_target={}, bindings=(address, city), entities=()
        )
