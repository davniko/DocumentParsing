import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import Runner, SourceContract, digest
from document_ocr.synthesis.curated_campaign import (
    Campaign,
    assemble_lexical_target,
    combine_surfaces,
    host_product_wording,
    public_identity_updates,
    wording_request,
    wording_request_hash,
)
from document_ocr.synthesis.curated_scenarios import ScenarioLocation, ShipmentScenario
from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.curated_wording import (
    WordingBatch,
    WordingField,
    WordingRequest,
    review_output_type,
    unpack_review,
    unpack_wording,
    validate_postal_geography,
    validate_wording,
    wording_output_type,
    wording_prompt,
)
from document_ocr.synthesis.generators import DeterministicStream, validate_container_number


def test_review_schema_owns_complete_shipment_coverage_and_identity():
    from pydantic import ValidationError

    schema = review_output_type(2)
    payload = {
        "s0": {"findings": []},
        "s1": {
            "findings": [
                {
                    "field": "description",
                    "problem": "Wrong commodity",
                    "evidence": "INJECTION",
                    "correction": "Use the selected non-injection commodity",
                }
            ]
        },
    }
    review = unpack_review(schema.model_validate(payload), ["long-first-hash", "long-second-hash"])
    assert review.reviewed_ids == ["long-first-hash", "long-second-hash"]
    assert review.findings[0].sample_id == "long-second-hash"
    for invalid in ({"s0": payload["s0"]}, {**payload, "all three": payload["s0"]}):
        with pytest.raises(ValidationError):
            schema.model_validate(invalid)
    with pytest.raises(ValueError, match="coverage"):
        unpack_review(schema.model_validate(payload), ["duplicate", "duplicate"])


def test_wording_contract_checks_coverage_identity_and_layout_without_freezing_punctuation():
    request = WordingRequest(
        "s1",
        "Destination Rotterdam",
        (
            WordingField("name", "shipper name", "OLD LTD", "New company"),
            WordingField("postal", "shipper postal", "OLD STREET", "Postal address"),
        ),
    )
    payload = {
        "shipments": [
            {
                "sample_id": "s1",
                "values": [
                    {"key": "name", "text": "NORTH BV"},
                    {"key": "postal", "text": "8 CANAL ROAD\nROTTERDAM / NETHERLANDS"},
                ],
            }
        ]
    }
    assert "Structure example:\nOLD STREET" in wording_prompt([request])
    assert validate_wording(WordingBatch.model_validate(payload), [request])["s1"][
        "postal"
    ].endswith("NETHERLANDS")
    padded = deepcopy(payload)
    padded["shipments"][0]["values"][1]["text"] = " 8 CANAL ROAD\r\n\n ROTTERDAM / NETHERLANDS "
    assert validate_wording(WordingBatch.model_validate(padded), [request])["s1"]["postal"] == (
        "8 CANAL ROAD\nROTTERDAM / NETHERLANDS"
    )
    for field, bad, expected in [
        (0, "old ltd", "copied source identity"),
        (0, "NEW LTD\nATTN: JANE DOE", "contact caption"),
        (0, "NEW LTD\nATTENTION:JANE DOE", "contact caption"),
        (1, "8 CANAL ROAD\n, ROTTERDAM", "line-leading punctuation"),
        (1, "8 CANAL ROAD 000000", "placeholder postcode"),
        (1, " \n ", "empty generated region"),
    ]:
        changed = deepcopy(payload)
        changed["shipments"][0]["values"][field]["text"] = bad
        with pytest.raises(ValueError, match=expected):
            validate_wording(WordingBatch.model_validate(changed), [request])
    payload["shipments"][0]["values"].pop()
    with pytest.raises(ValueError, match="field coverage differs"):
        validate_wording(WordingBatch.model_validate(payload), [request])


def test_disjoint_semantic_renderers_must_not_silently_overwrite_each_other():
    assert combine_surfaces({"a": "X"}, {"a": "X", "b": ["Y"]}) == {"a": "X", "b": ["Y"]}
    with pytest.raises(ValueError, match="conflicting render ownership"):
        combine_surfaces({"a": "X"}, {"a": "Y"})


def test_generated_postal_contract_rejects_omissions_and_adjacent_duplicates():
    def check(address):
        validate_postal_geography(
            address, locality="SUCEAVA", country="ROMANIA", country_code="RO", country_labelled=True
        )

    check("14 STREET, SUCEAVA, ROMANIA")
    check("14 STREET, SUCEAVA, RO")
    for bad in (
        "14 STREET, ROMANIA",
        "14 STREET SUCEAVA ROMANIA (ROMANIA)",
        "SUCEAVA, 14 STREET, SUCEAVA, ROMANIA",
        "14 STREET SUCEAVA",
    ):
        with pytest.raises(ValueError):
            check(bad)
    validate_postal_geography(
        "14 STREET SAINT-PIERRE, SAINT PIERRE AND MIQUELON",
        locality="SAINT-PIERRE",
        country="SAINT PIERRE AND MIQUELON",
        country_code="PM",
        country_labelled=True,
    )
    validate_postal_geography(
        "14 STREET DJIBOUTI",
        locality="DJIBOUTI",
        country="DJIBOUTI",
        country_code="DJ",
        country_labelled=True,
    )
    with pytest.raises(ValueError, match="not requested"):
        validate_postal_geography(
            "14 STREET GIUSSANO, ITALY",
            locality="GIUSSANO",
            country="ITALY",
            country_code="IT",
            country_labelled=False,
        )
    validate_postal_geography(
        "12 DOCK ROAD SINGAPORE",
        locality="SINGAPORE",
        country="SINGAPORE",
        country_code="SG",
        country_labelled=False,
    )


def test_native_wording_schema_requires_exact_fields_instead_of_unconstrained_keys():
    request = WordingRequest(
        "source-independent-hash",
        "New shipment",
        (
            WordingField("n1", "shipper name", "OLD LTD", "New fictional company"),
            WordingField("a1", "shipper postal", "OLD ROAD", "Complete address"),
        ),
    )
    schema = wording_output_type([request])
    output = schema.model_validate({"s0": {"n1": "NORTH BV", "a1": "8 CANAL ROAD"}})
    assert unpack_wording(output, [request])[request.sample_id]["n1"] == "NORTH BV"
    with pytest.raises(ValueError):
        schema.model_validate({"s0": {"shipper_name": "NORTH BV", "a1": "8 CANAL ROAD"}})
    with pytest.raises(ValueError):
        schema.model_validate({"s0": {"n1": "NORTH BV"}})


def test_sampled_identifiers_dates_and_repeated_contacts_remain_coupled():
    geo = ScenarioLocation(
        name="ROTTERDAM",
        country_code="NL",
        country="NETHERLANDS",
        registry="unlocode_wpi",
        registry_id="NLRTM",
    )
    scenario = ShipmentScenario(
        source_id="s",
        variant=1,
        origin=geo,
        destination=geo.model_copy(update={"country_code": "BE", "country": "BELGIUM"}),
        party_localities={
            "documentPatch.parties.shipper": geo,
            "documentPatch.parties.consignee": geo,
        },
        goods_identities=(),
        replacements={},
        cargo={},
        provenance={},
    )
    target = {
        "documentPatch": {
            "issueDate": "2024-01-05",
            "shippedOnBoardDate": "2024-01-03",
            "parties": {
                role: {"contactDetails": {"phoneNumbers": ["+31123456789"]}}
                for role in ("shipper", "consignee")
            },
            "containerInformation": [
                {"equipmentIdentifier": "MSCU6639871", "sealNumbers": ["AB1234"]}
            ],
            "goodsItemDetails": [{"splitGoodsPlacement": [{"equipmentIdentifier": "MSCU6639871"}]}],
        }
    }
    changed = public_identity_updates(target, scenario, DeterministicStream(1, "test", "s"))[
        "documentPatch"
    ]
    equipment = changed["containerInformation"][0]["equipmentIdentifier"]
    assert equipment != "MSCU6639871" and validate_container_number(equipment)
    assert (
        changed["goodsItemDetails"][0]["splitGoodsPlacement"][0]["equipmentIdentifier"] == equipment
    )
    assert changed["parties"]["shipper"] == changed["parties"]["consignee"]
    from datetime import date

    assert (
        date.fromisoformat(changed["issueDate"]) - date.fromisoformat(changed["shippedOnBoardDate"])
    ).days == 2
    assert target["documentPatch"]["issueDate"] == "2024-01-05"


@pytest.mark.parametrize("invalid_native_output", [False, True])
def test_explicit_resume_preserves_and_replays_attempt_receipts(tmp_path, invalid_native_output):
    runner = Runner.__new__(Runner)
    runner.output = tmp_path
    runner.config = {"provider": {}}
    runner.retry_rate_limited = False
    runner.retry_invalid_output = False
    key = digest(["wording", "sample", WordingBatch.model_json_schema(), "sys", "text", {}])
    directory = tmp_path / "calls"
    directory.mkdir()
    original = directory / f"wording-{key}.json"
    failed = {
        "output": None,
        "error": "UnexpectedModelBehavior: invalid native JSON"
        if invalid_native_output
        else "ModelHTTPError: status_code: 429",
        "costStatus": "provider_reported" if invalid_native_output else "request_rejected",
        "costUsd": "0.001" if invalid_native_output else "0",
    }
    original.write_text(json.dumps(failed))
    (directory / f"wording-{key}-attempt2.json").write_text(
        json.dumps({"output": {"shipments": []}})
    )
    with pytest.raises(RuntimeError, match="cached failed call"):
        asyncio.run(runner.call("wording", "sample", WordingBatch, "sys", "text"))
    runner.retry_rate_limited = not invalid_native_output
    runner.retry_invalid_output = invalid_native_output
    result = asyncio.run(runner.call("wording", "sample", WordingBatch, "sys", "text"))
    assert result.shipments == []
    assert json.loads(original.read_text()) == failed
    failed["error"] = "ModelHTTPError: status_code: 402"
    original.write_text(json.dumps(failed))
    with pytest.raises(RuntimeError, match="cached failed call"):
        asyncio.run(runner.call("wording", "sample", WordingBatch, "sys", "text"))


@pytest.fixture
def postal_campaign(tmp_path):
    """A split postal region beside a clean party and unrelated lexical values."""
    source_id, sample_id = "source", "sample"
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "OLD EXPORT BV",
                    "addressLine": "2 OLD ROAD ROTTERDAM NETHERLANDS",
                    "country": "NETHERLANDS",
                    "contactDetails": {"phoneNumbers": ["+31 612345678"]},
                },
                "consignee": {
                    "name": "OLD IMPORT BV",
                    "addressLine": "7 OLD QUAY UTRECHT NETHERLANDS",
                    "country": "NETHERLANDS",
                },
            },
            "goodsItemDetails": [{"description": "OLD COFFEE"}],
        },
    }
    variables = [
        ("shipper_name", "name", "OLD EXPORT BV"),
        ("shipper_street", "postal", "2 OLD ROAD"),
        ("shipper_locality", "postal", "ROTTERDAM NETHERLANDS"),
        ("consignee_name", "name", "OLD IMPORT BV"),
        ("consignee_address", "postal", "7 OLD QUAY UTRECHT NETHERLANDS"),
        ("product", "product", "OLD COFFEE"),
    ]
    contract = SourceContract.model_validate(
        {
            "variables": [
                {
                    "key": key,
                    "kind": kind,
                    "value": value,
                    "meaning": kind,
                    "required_literals": [],
                    "occurrences": [{"text": value, "occurrence": 1, "presentation": "text"}],
                }
                for key, kind, value in variables
            ],
            "targets": [
                {"path": "documentPatch.parties.shipper.name", "expression": "{shipper_name}"},
                {
                    "path": "documentPatch.parties.shipper.addressLine",
                    "expression": "{shipper_street} {shipper_locality}",
                },
                {"path": "documentPatch.parties.consignee.name", "expression": "{consignee_name}"},
                {
                    "path": "documentPatch.parties.consignee.addressLine",
                    "expression": "{consignee_address}",
                },
                {
                    "path": "documentPatch.goodsItemDetails[0].description",
                    "expression": "{product}",
                },
            ],
            "fixed_context": "Test the existing party and product ownership contract.",
        }
    )
    blueprint = SamplingBlueprint(source_id, "SOURCE", target, contract, (), {}, {}, ())
    origin = ScenarioLocation(
        name="ROTTERDAM",
        country="NETHERLANDS",
        country_code="NL",
        registry="unlocode_wpi",
        registry_id="NLRTM",
    )
    destination = origin.model_copy(update={"name": "UTRECHT", "registry_id": "NLUTC"})
    scenario = ShipmentScenario(
        source_id=source_id,
        variant=1,
        origin=origin,
        destination=destination,
        party_localities={
            "documentPatch.parties.shipper": origin,
            "documentPatch.parties.consignee": destination,
        },
        goods_identities=(
            {
                "hs6": "090111",
                "phrase": "UNROASTED COFFEE",
                "classification": {
                    "authority": "fixture",
                    "chapter_code": "09",
                    "chapter_description": "Coffee, tea and spices",
                    "heading_code": "0901",
                    "heading_description": "Coffee",
                    "description_path": ["Coffee", "Not roasted", "Not decaffeinated"],
                    "same_heading_alternatives": [],
                    "immediate_national_children": [],
                },
            },
        ),
        replacements={},
        cargo={},
        provenance={},
    )
    values = {
        "shipper_name": "NORTH EXPORT BV",
        "shipper_street": "14 CANAL ROAD, 3011AA ROTTERDAM",
        "shipper_locality": "ROTTERDAM NETHERLANDS",
        "consignee_name": "SOUTH IMPORT BV",
        "consignee_address": "8 CANAL QUAY UTRECHT NETHERLANDS",
        "product": "ARABICA UNROASTED COFFEE",
    }
    receipt = {
        "sourceDocumentId": source_id,
        "documentId": sample_id,
        "requestSha256": wording_request_hash(wording_request(blueprint, scenario, sample_id)),
        "values": values,
        "provenance": {"originalPaidCall": "preserve-me"},
    }
    path = tmp_path / "wording" / f"{sample_id}.json"
    path.parent.mkdir()
    path.write_text(json.dumps(receipt))
    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.plans = lambda sid: [(blueprint, scenario, sample_id, deepcopy(target))]
    campaign.calls = SimpleNamespace()
    return SimpleNamespace(
        campaign=campaign,
        path=path,
        receipt=receipt,
        blueprint=blueprint,
        scenario=scenario,
        target=target,
    )


def test_whole_unit_products_are_host_owned_not_free_model_specifications(postal_campaign):
    fixture = postal_campaign
    assert host_product_wording(fixture.blueprint, fixture.scenario) == {}
    for profile in ("source_whole_units", "reviewed_whole_units"):
        scenario = fixture.scenario.model_copy(update={"provenance": {"physicalProfile": profile}})
        products = host_product_wording(fixture.blueprint, scenario)
        assert products == {"product": scenario.goods_identities[0].phrase.upper()}
        request = wording_request(fixture.blueprint, scenario, "sample")
        assert "product" not in {field.key for field in request.fields}
        assert "shipper_name" in {field.key for field in request.fields}
        lexical = {k: v for k, v in fixture.receipt["values"].items() if k != "product"}
        target = assemble_lexical_target(
            fixture.blueprint, fixture.target, {**lexical, **products}, scenario
        )
        assert target["documentPatch"]["goodsItemDetails"][0]["description"] == products["product"]
        with pytest.raises(ValueError, match="one certified commodity"):
            host_product_wording(
                fixture.blueprint,
                scenario.model_copy(update={"goods_identities": scenario.goods_identities * 2}),
            )


def test_postal_correction_changes_only_failed_owned_regions_and_preserves_cache_history(
    postal_campaign,
):
    fixture = postal_campaign
    calls = []
    correction = {
        "shipper_street": "14 CANAL ROAD, 3011AA",
        "shipper_locality": "ROTTERDAM NETHERLANDS",
    }

    async def call(stage, sid, output_type, system, prompt):
        calls.append((stage, sid))
        shipment_schema = output_type.model_fields["s0"].annotation
        assert set(shipment_schema.model_fields) == set(correction)
        assert "Previous complete address" in prompt
        assert "3011AA" in prompt
        return output_type.model_validate({"s0": correction})

    fixture.campaign.calls.call = call
    result = asyncio.run(fixture.campaign.correct_postal("source"))
    assert result == {"sourceDocumentId": "source", "corrected": 1}
    assert calls == [("postal-correction", "source")]
    repaired = json.loads(fixture.path.read_text())
    assert repaired["values"] == {**fixture.receipt["values"], **correction}
    assert repaired["provenance"] == fixture.receipt["provenance"]
    assert repaired["requestSha256"] == fixture.receipt["requestSha256"]
    history = repaired["corrections"][-1]
    assert history["before"] == {key: fixture.receipt["values"][key] for key in correction}
    assert history["after"] == correction
    assert history["issues"] and len(history["requestSha256"]) == 64
    assert history["postCheckErrors"] == []
    assembled = assemble_lexical_target(
        fixture.blueprint, fixture.target, repaired["values"], fixture.scenario
    )
    assert assembled["documentPatch"]["parties"]["shipper"]["addressLine"] == (
        "14 CANAL ROAD, 3011AA ROTTERDAM NETHERLANDS"
    )
    assert (
        assembled["documentPatch"]["parties"]["shipper"]["contactDetails"]
        == (fixture.target["documentPatch"]["parties"]["shipper"]["contactDetails"])
    )
    # A normal generation resume must keep both the correction and its receipt,
    # without another paid call or a silent return to the uncorrected wording.
    generation = asyncio.run(fixture.campaign.generate("source"))
    resumed = json.loads(fixture.path.read_text())
    assert generation["generated"] == 1
    assert calls == [("postal-correction", "source")]
    assert resumed["values"] == repaired["values"]
    assert resumed["corrections"] == repaired["corrections"]
    assert resumed["provenance"] == repaired["provenance"]


def test_clean_postal_roles_make_no_correction_request_or_receipt_edit(postal_campaign):
    fixture = postal_campaign
    receipt = deepcopy(fixture.receipt)
    receipt["values"]["shipper_street"] = "14 CANAL ROAD, 3011AA"
    fixture.path.write_text(json.dumps(receipt))
    before = fixture.path.read_bytes()

    async def call(*args):
        pytest.fail("clean postal geography must not incur a model request")

    fixture.campaign.calls.call = call
    assert asyncio.run(fixture.campaign.correct_postal("source")) == {
        "sourceDocumentId": "source",
        "corrected": 0,
    }
    assert fixture.path.read_bytes() == before


def test_bad_postal_proposal_is_diagnosed_not_claimed_corrected(postal_campaign):
    fixture = postal_campaign
    correction = {"shipper_street": "14 CANAL ROAD, 3011AA", "shipper_locality": "NETHERLANDS"}

    async def call(stage, sid, output_type, system, prompt):
        return output_type.model_validate({"s0": correction})

    fixture.campaign.calls.call = call
    with pytest.raises(ValueError, match=r"postal|locality"):
        asyncio.run(fixture.campaign.correct_postal("source"))
    receipt = json.loads(fixture.path.read_text())
    history = receipt["corrections"][-1]
    assert history["before"] == {key: fixture.receipt["values"][key] for key in correction}
    assert history["after"] == correction
    assert history["postCheckErrors"]
    assert all(
        receipt["values"][key] == value
        for key, value in fixture.receipt["values"].items()
        if key not in correction
    )
    assert not (fixture.campaign.output / "candidates").exists()
