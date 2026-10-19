"""Bounded controls for generated identity ownership in postal components."""

import asyncio
import json

import pytest
from test_curated_campaign import postal_campaign as postal_campaign

from document_ocr.synthesis.curated_campaign import (
    postal_assembly_context,
    postal_wording_errors,
    wording_request,
    wording_request_hash,
)
from document_ocr.synthesis.curated_wording import (
    POSTAL_CORRECTION_PROMPT,
    WORDING_PROMPT,
    WordingField,
    WordingRequest,
    validate_postal_identity,
    wording_output_type,
)


@pytest.mark.parametrize(
    "address",
    ["NORTH EXPORT BV\n14 CANAL ROAD\nROTTERDAM NETHERLANDS", "North Export Bv\n14 Canal Road"],
)
def test_exact_leading_party_identity_is_held(address):
    with pytest.raises(ValueError, match="repeats its party identity"):
        validate_postal_identity(
            address, name="NORTH EXPORT BV", locality="ROTTERDAM", country="NETHERLANDS"
        )


@pytest.mark.parametrize(
    "address",
    [
        "DELIVERY AGENT: OTHER LOGISTICS LLC\n14 CANAL ROAD\nROTTERDAM NETHERLANDS",
        "14 CANAL ROAD\nTAX ID: 12345678\nROTTERDAM NETHERLANDS",
    ],
)
def test_explicit_identity_contact_captions_are_not_postal_components(address):
    with pytest.raises(ValueError, match="identity/contact caption"):
        validate_postal_identity(
            address, name="NORTH EXPORT BV", locality="ROTTERDAM", country="NETHERLANDS"
        )


@pytest.mark.parametrize(
    ("name", "address"),
    [
        ("NORTH EXPORT BV", "C/O NORTH EXPORT BV\n14 CANAL ROAD\nROTTERDAM NETHERLANDS"),
        ("NORTH EXPORT BV", "NORTH EXPORT BV BUILDING\n14 CANAL ROAD\nROTTERDAM NETHERLANDS"),
        ("NORTH EXPORT BV", "14 CANAL ROAD\nNEAR NORTH EXPORT BV\nROTTERDAM NETHERLANDS"),
        ("ROTTERDAM", "ROTTERDAM\nNETHERLANDS"),
        ("NETHERLANDS", "NETHERLANDS\nROTTERDAM"),
        ("NORTH EXPORT BV", "14 CANAL ROAD\nROTTERDAM NETHERLANDS"),
    ],
)
def test_care_of_sites_and_geographic_names_are_not_classified_as_identity_repeats(name, address):
    validate_postal_identity(address, name=name, locality="ROTTERDAM", country="NETHERLANDS")


def test_party_field_description_and_prompts_explain_identity_ownership():
    request = WordingRequest(
        "sample", "", (WordingField("p", "shipper postal", "OLD ROAD", "Postal only."),)
    )
    schema = wording_output_type([request]).model_json_schema()
    description = schema["$defs"]["Shipment0"]["properties"]["shipper_postal_component_1"][
        "description"
    ]
    assert "name belongs only to its name field" in description
    assert "insert another company identity" in description
    assert "care-of recipients and named sites" in description
    assert "name field owns the party identity" in WORDING_PROMPT
    assert "Party names are fixed context" in POSTAL_CORRECTION_PROMPT


def test_name_context_and_guard_use_the_same_party_assembly(postal_campaign):
    f = postal_campaign
    values = {
        **f.receipt["values"],
        "shipper_street": "NORTH EXPORT BV\n14 CANAL ROAD, 3011AA",
    }
    errors = postal_wording_errors(f.blueprint, f.scenario, f.target, values)
    assert errors == {
        "documentPatch.parties.shipper": [
            "generated postal component repeats its party identity as a leading line"
        ]
    }
    context = postal_assembly_context(
        f.blueprint, f.scenario, f.target, {"shipper_street"}, fixed_values=values
    )
    assert (
        "PARTY NAME (identity only, never part of the postal component): NORTH EXPORT BV" in context
    )
    assert "SOUTH IMPORT BV" not in context


def test_scoped_postal_repair_preserves_names_and_other_fields(postal_campaign):
    f = postal_campaign
    f.receipt["values"]["shipper_street"] = "NORTH EXPORT BV\n14 CANAL ROAD, 3011AA"
    f.receipt["requestSha256"] = wording_request_hash(
        wording_request(f.blueprint, f.scenario, "sample")
    )
    f.path.write_text(json.dumps(f.receipt))
    original = f.receipt["values"].copy()

    async def call(stage, sid, output_type, system, prompt):
        assert stage == "postal-correction"
        assert "never part of the postal component): NORTH EXPORT BV" in prompt
        assert "repeats its party identity" in prompt
        assert "Structure example:\nNORTH EXPORT BV" not in prompt
        fields = output_type.model_fields["s0"].annotation.model_fields
        assert set(fields) == {"shipper_street", "shipper_locality"}
        return output_type.model_validate(
            {
                "s0": {
                    "shipper_street": "14 CANAL ROAD, 3011AA",
                    "shipper_locality": "ROTTERDAM NETHERLANDS",
                }
            }
        )

    f.campaign.calls.call = call
    assert asyncio.run(f.campaign.correct_postal("source"))["corrected"] == 1
    repaired = json.loads(f.path.read_text())
    assert repaired["values"] == {**original, "shipper_street": "14 CANAL ROAD, 3011AA"}
    assert repaired["values"]["shipper_name"] == original["shipper_name"]
    assert repaired["provenance"] == f.receipt["provenance"]
    assert not repaired["corrections"][-1]["postCheckErrors"]


def test_unchanged_identity_repeat_cannot_pass_scoped_repair(postal_campaign):
    f = postal_campaign
    f.receipt["values"]["shipper_street"] = "NORTH EXPORT BV\n14 CANAL ROAD, 3011AA"
    f.receipt["requestSha256"] = wording_request_hash(
        wording_request(f.blueprint, f.scenario, "sample")
    )
    f.path.write_text(json.dumps(f.receipt))

    async def call(stage, sid, output_type, system, prompt):
        return output_type.model_validate(
            {
                "s0": {
                    "shipper_street": f.receipt["values"]["shipper_street"],
                    "shipper_locality": f.receipt["values"]["shipper_locality"],
                }
            }
        )

    f.campaign.calls.call = call
    with pytest.raises(
        ValueError, match=r"postal correction still requires review.*party identity"
    ):
        asyncio.run(f.campaign.correct_postal("source"))
