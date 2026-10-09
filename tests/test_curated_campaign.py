import asyncio
import json
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import Runner, SourceContract, digest
from document_ocr.synthesis.curated_campaign import (
    Campaign,
    assemble_lexical_target,
    combine_surfaces,
    host_product_wording,
    public_identity_updates,
    validate_sample,
    wording_request,
    wording_request_hash,
)
from document_ocr.synthesis.curated_physical import PhysicalRenderPlan
from document_ocr.synthesis.curated_scenarios import ScenarioLocation, ShipmentScenario
from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.curated_wording import (
    WordingBatch,
    WordingField,
    WordingRequest,
    mass_in_kilograms,
    review_output_type,
    unpack_review,
    unpack_wording,
    validate_postal_geography,
    validate_product_masses,
    validate_wording,
    wording_output_type,
    wording_prompt,
)
from document_ocr.synthesis.generators import DeterministicStream, validate_container_number


@pytest.mark.parametrize(
    ("source_route", "candidate_route", "error"),
    [
        (None, None, None),
        (None, {"portOfLoading": {"name": "NEW PORT"}}, "field presence differs"),
        (
            {"portOfLoading": {"name": "OLD PORT"}},
            {"portOfLoading": {"name": "OLD PORT"}},
            "source-copy route",
        ),
        (
            {"portOfLoading": {"name": "OLD PORT"}},
            {"portOfLoading": {"name": "NEW PORT"}},
            None,
        ),
    ],
)
def test_sample_validation_applies_route_variability_only_to_source_route_fields(
    monkeypatch, source_route, candidate_route, error
):
    source = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": "non_negotiable",
            "goodsItemDetails": [{"description": "OLD GOODS"}],
        },
    }
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": "non_negotiable",
            "goodsItemDetails": [{"description": "NEW GOODS"}],
        },
    }
    if source_route is not None:
        source["documentPatch"]["route"] = source_route
    if candidate_route is not None:
        target["documentPatch"]["route"] = candidate_route
    replacements = {
        f"documentPatch.route.{field}.{key}": value
        for field, location in (candidate_route or {}).items()
        for key, value in location.items()
    }
    scenario = SimpleNamespace(party_localities={}, replacements=replacements, goods_identities=[])
    proof = {"changedTargetPaths": ["documentPatch.goodsItemDetails[0].description"]}
    candidate = {
        "target": target,
        "countryCodes": {},
        "renderValues": {},
        "surfaceValues": {},
        "seed": 1,
        "documentId": "sample",
        "joinedRawText": "NEW GOODS",
        "joinedRawTextSha256": digest(b"NEW GOODS"),
        "proof": proof,
    }
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.render_sampling_blueprint",
        lambda *args, **kwargs: ("NEW GOODS", target, proof),
    )

    def run():
        return validate_sample(
            SimpleNamespace(target=source, source="OLD GOODS"),
            scenario,
            candidate,
            SimpleNamespace(canonicalize=lambda x: x),
        )

    if error:
        with pytest.raises(ValueError, match=error):
            run()
    else:
        assert run()["sampledIdentity"]


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_generation_validation_repair_is_bounded_and_cannot_publish_bad_wording(
    postal_campaign, repair_succeeds
):
    fixture = postal_campaign
    fixture.path.unlink()
    calls = []

    async def call(stage, sid, output_type, system, prompt):
        calls.append(stage)
        values = dict(fixture.receipt["values"])
        if stage == "exact-wording" or not repair_succeeds:
            values["product"] = "TOTAL CARTONS: 120"
        else:
            assert "VALIDATION FAILURE" in prompt
            assert "shipment accounting" in prompt
        return output_type.model_validate({"s0": values})

    fixture.campaign.calls.call = call
    if repair_succeeds:
        result = asyncio.run(fixture.campaign.generate("source"))
        assert "shipment accounting" in result["wordingValidationRepair"]
        saved = json.loads(fixture.path.read_text())
        assert saved["values"] == fixture.receipt["values"]
        assert saved["wordingValidationRepair"]
    else:
        with pytest.raises(ValueError, match="shipment accounting"):
            asyncio.run(fixture.campaign.generate("source"))
        assert not fixture.path.exists()
    assert calls == ["exact-wording", "wording-validation-repair"]


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


def test_variable_quota_review_batches_preserve_every_candidate_and_local_ids(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.contained_packing_context",
        lambda blueprint, target: [{"contained_quantity": 240, "printed_contained_unit": "SACOS"}],
    )
    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.config = {"seed": 42, "variants_per_source": 2}
    campaign.variant_counts = {"source": 5}
    campaign.casing = SimpleNamespace(target="uppercase")
    campaign.replay_candidate = lambda candidate: None
    campaign.blueprint = lambda sid: SimpleNamespace(ownership_data={})
    (tmp_path / "candidates").mkdir()
    candidates = []
    for variant in range(1, 6):
        did = "syn_full_v7_" + digest(["source", 42, variant])[:24]
        candidate = {
            "documentId": did,
            "target": {},
            "joinedRawText": f"TEXT {variant}",
            "scenario": {"party_localities": {}, "goods_identities": [], "provenance": {}},
        }
        (tmp_path / "candidates" / f"{did}.json").write_text(json.dumps(candidate))
        candidates.append(candidate)
    requests = []

    async def call(stage, sid, model, system, prompt):
        requests.append(prompt)
        assert stage == "rendered-review" and "SHIPMENT s0\n" in prompt
        assert "SHIPMENT s2\n" not in prompt
        assert "contained_quantity: 240" in prompt and "printed_contained_unit: SACOS" in prompt
        return model.model_validate({key: {"findings": []} for key in model.model_fields})

    campaign.calls = SimpleNamespace(call=call)
    result = asyncio.run(campaign.review("source"))
    assert result["findings"] == 0 and len(requests) == 3
    receipt = json.loads((tmp_path / "reviews/source.json").read_text())
    assert receipt["review"]["reviewed_ids"] == [c["documentId"] for c in candidates]
    assert receipt["candidateHashes"] == {c["documentId"]: digest(c) for c in candidates}


def test_variable_quota_contacts_batch_by_shipment_and_replay_without_calls(tmp_path):
    from document_ocr.synthesis.curated_contacts import ContactField, ContactParty

    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.config = {"variants_per_source": 2}
    parties = [
        ContactParty(
            f"sample{i}",
            f"COMPANY {i}-{j}",
            "NETHERLANDS",
            (ContactField("c0", "email", "old@oldcompany.com", (f"party{j}.email",)),),
        )
        for i in range(5)
        for j in range(2)
    ]
    campaign.contact_requests = lambda sid: parties
    sizes = []

    async def call(stage, sid, model, system, prompt):
        sizes.append(len(model.model_fields))
        return model.model_validate(
            {key: {"c0": f"mail{len(sizes)}{key}@newcompany.com"} for key in model.model_fields}
        )

    campaign.calls = SimpleNamespace(call=call)
    assert asyncio.run(campaign.generate_contacts("source"))["companies"] == 10
    assert sizes == [4, 4, 2]
    receipt = json.loads((tmp_path / "contacts/source.json").read_text())
    for i in range(5):
        single = json.loads((tmp_path / f"contacts/samples/sample{i}.json").read_text())
        assert single["batchReceiptSha256"] == digest(receipt)
        assert single["values"] == receipt["values"][f"sample{i}"]
    asyncio.run(campaign.generate_contacts("source"))
    assert sizes == [4, 4, 2]


def test_independent_contact_domains_need_no_correction_or_regeneration(tmp_path):
    from document_ocr.synthesis.curated_contacts import ContactField, ContactParty

    campaign = Campaign.__new__(Campaign)
    campaign.output = tmp_path
    campaign.config = {"variants_per_source": 2}
    party = ContactParty(
        "sample",
        "NEW COMPANY",
        "NETHERLANDS",
        (
            ContactField("c0", "email", "info@oldcompany.com", ("email",)),
            ContactField("c1", "website", "www.oldcompany.com", ("website",)),
        ),
    )
    campaign.contact_requests = lambda sid: [party]
    calls = []

    async def call(stage, sid, model, system, prompt):
        calls.append(stage)
        return model.model_validate(
            {
                "p0": {
                    "c0": "newcompany.info@gmail.com",
                    "c1": "www.newcompany.com",
                }
            }
        )

    campaign.calls = SimpleNamespace(call=call)
    assert asyncio.run(campaign.generate_contacts("source"))["fields"] == 2
    asyncio.run(campaign.generate_contacts("source"))
    assert calls == ["company-contacts"]


@pytest.mark.parametrize(
    "text", ["GROSS WEIGHT 7500.0 KG", "NET WEIGHT: 6000 KGS", "PACKAGE_PACKAGE"]
)
def test_goods_wording_rejects_host_accounting_without_requiring_total_caption(text):
    request = WordingRequest(
        "sample", "Goods", (WordingField("g", "goods description", "OLD", "Product"),)
    )
    output = wording_output_type([request]).model_validate({"s0": {"g": text}})
    with pytest.raises(ValueError, match="inside generated description"):
        unpack_wording(output, [request])


def test_product_precision_is_not_a_postcode_placeholder():
    request = WordingRequest(
        "sample", "Goods", (WordingField("g", "goods description", "OLD", "Product"),)
    )
    text = "QUARTZ CRYSTAL, NOMINAL FREQUENCY 26.000000 MHZ"
    output = wording_output_type([request]).model_validate({"s0": {"g": text}})
    assert unpack_wording(output, [request])["sample"]["g"] == text


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
        (1, r"8 CANAL ROAD\nROTTERDAM", "escaped line break"),
        (1, " \n ", "empty generated region"),
    ]:
        changed = deepcopy(payload)
        changed["shipments"][0]["values"][field]["text"] = bad
        with pytest.raises(ValueError, match=expected):
            validate_wording(WordingBatch.model_validate(changed), [request])
    payload["shipments"][0]["values"].pop()
    with pytest.raises(ValueError, match="field coverage differs"):
        validate_wording(WordingBatch.model_validate(payload), [request])


@pytest.mark.parametrize(
    "text",
    [
        "NET 18.5 KG PER MASTER CARTON",
        "20 LITRES/DRUM",
        "30-40 PCS PER CARTON",
        "YARN ON 5KG SPOOLS",
        "WIRE ON10 KG COILS",
        "YARN CONES 1.5 KG",
        "BAGS 25KG",
    ],
)
def test_generated_product_cannot_invent_host_owned_package_fill(text):
    request = WordingRequest(
        "sample", "Sampled goods", (WordingField("g", "goods description", "OLD", "Product"),)
    )
    output = wording_output_type([request]).model_validate({"s0": {"g": text}})
    with pytest.raises(ValueError, match="host-owned package fill"):
        unpack_wording(output, [request])


@pytest.mark.parametrize(
    "text",
    [
        "FISH SIZE GRADE 500-800 G",
        "FABRIC 140 G/M2",
        "LIFT CAPACITY 500 KG",
        "CASTORS, RATED LOAD CAPACITY 150 KG PER CASTOR",
        "PRESS, 150 TON CLAMPING FORCE",
        "LOADER, 12 TONNE RATED CAPACITY",
        "RUBBER SHEETS, DENSITY 160 KG/M3",
        "PAPER, UNIT MASS 140 G/M2",
        "FIBC BAGS, 1000 KG SAFE WORKING LOAD",
        "EMPTY 25 KG CAPACITY BAGS",
        "INGOTS APPROXIMATELY 25 KG EACH",
        "CHILLED FISH, SIZE GRADE M 1.2-2.0 KG PER PIECE",
        "CHILLED CARCASS WEIGHT 1.2-1.6 KG PER BIRD",
        "ROASTING MACHINE, OUTPUT 180 KG/H",
        "COLD-ROLLED STEEL COILS, 1250 MM WIDTH",
        "FROZEN AT -35 C, CARRIAGE TEMPERATURE -20 C",
    ],
)
def test_product_size_and_capacity_are_not_package_fill(text):
    request = WordingRequest(
        "sample", "Sampled goods", (WordingField("g", "goods description", "OLD", "Product"),)
    )
    output = wording_output_type([request]).model_validate({"s0": {"g": text}})
    assert unpack_wording(output, [request])["sample"]["g"] == text


@pytest.mark.parametrize(
    "text",
    [
        "LOADER, OPERATING WEIGHT 16,500 KG",
        "LOADER, 12 TONNE OPERATING WEIGHT",
        "MACHINE SHIPPING MASS: 16,500 KILOGRAMS",
        "UNIT WEIGHT APPROX. 500 LBS",
        "ITEM MASS = 2500 GRAMS",
        "WEIGHT PER UNIT: 600 KG",
        "MASS OF EACH MACHINE 0.8 METRIC TONNES",
        "ASSEMBLY WEIGHS 450 KG",
        "ONE UNIT WEIGHING APPROXIMATELY 1,500 KG",
    ],
)
def test_actual_item_masses_cannot_exceed_supplied_whole_cargo_mass(text):
    request = WordingRequest(
        "sample",
        "Goods",
        (WordingField("g", "goods description", "OLD", "Product"),),
        gross_weight_kg=Decimal("1"),
    )
    output = wording_output_type([request]).model_validate({"s0": {"g": text}})
    with pytest.raises(ValueError, match="item-mass lower bound"):
        unpack_wording(output, [request])


def test_product_mass_bound_uses_distinct_list_entries_not_repeated_or_alternative_masses():
    text = "- MODEL A, OPERATING WEIGHT 16,500 KG\n- MODEL B, OPERATING WEIGHT 12,800 KG"
    with pytest.raises(ValueError, match=r"29300 kg exceeds.*20500 kg"):
        validate_product_masses([text], Decimal("20500"))
    validate_product_masses([text], Decimal("29300"))
    validate_product_masses(["MODEL A, OPERATING WEIGHT 16,200 KG"], Decimal("20500"))
    validate_product_masses([text], None)
    validate_product_masses(
        ["- MODEL A, OPERATING WEIGHT 12 TONNES, SHIPPING MASS 13 TONNES"], Decimal("14000")
    )
    repeated = "- MODEL A, OPERATING WEIGHT 12 TONNES"
    validate_product_masses([repeated, repeated.lower()], Decimal("14000"))
    validate_product_masses(
        ["1. MODEL A, OPERATING WEIGHT 12 TONNES\n2. MODEL A, OPERATING WEIGHT 12 TONNES"],
        Decimal("14000"),
    )
    with pytest.raises(ValueError, match="24000 kg exceeds"):
        validate_product_masses(
            ["1. MODEL A, OPERATING WEIGHT 12 TONNES\n2. MODEL B, OPERATING WEIGHT 12 TONNES"],
            Decimal("14000"),
        )
    # Unstructured alternatives are not assumed to be separate cargo items.
    validate_product_masses(
        ["MODEL A OPERATING WEIGHT 12 TONNES OR MODEL B SHIPPING MASS 13 TONNES"],
        Decimal("14000"),
    )
    validate_product_masses(
        ["MODEL A\n- OPERATING WEIGHT 12 TONNES\n- SHIPPING MASS 13 TONNES"],
        Decimal("14000"),
    )
    validate_product_masses(
        ["- MODEL A, OPERATING WEIGHT 12 TONNES\n- MODEL A, SHIPPING MASS 13 TONNES"],
        Decimal("14000"),
    )


def test_product_mass_bounds_preserve_ratings_and_convert_only_explicit_units():
    validate_product_masses(
        ["LIFT CAPACITY 5000 KG; DENSITY 160 KG/M3; PAPER UNIT MASS 140 G/M2"], Decimal("1")
    )
    validate_product_masses(["UNIT WEIGHT 2.2046226218 LBS"], Decimal("1"))
    with pytest.raises(ValueError, match="item-mass lower bound"):
        validate_product_masses(["UNIT WEIGHT 3 POUNDS"], Decimal("1"))
    validate_product_masses(["ITEM MASS 1000 GRAMS"], Decimal("1"))
    with pytest.raises(ValueError, match="ambiguous numeric notation"):
        validate_product_masses(["OPERATING WEIGHT 16,50 KG"], Decimal("100000"))
    assert mass_in_kilograms("2", "metric_tonne") == Decimal("2000")
    assert mass_in_kilograms("2", "pound") == Decimal("0.90718474")
    with pytest.raises(ValueError, match=r"unsupported.*unit"):
        mass_in_kilograms("2", "unknown")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mass_in_kilograms("NaN", "kilogram")


@pytest.mark.parametrize(
    "text",
    [
        "SAMPLED COMMODITY BRIEF: POLYMER RESIN",
        "POLYMER RESIN\nHOST SHIPMENT FACTS: 40HC, 200 BAGS",
        "STRUCTURE EXAMPLE:\nCOFFEE BEANS",
    ],
)
def test_goods_wording_rejects_generator_context_captions(text):
    request = WordingRequest(
        "sample", "Goods", (WordingField("g", "goods description", "OLD", "Product"),)
    )
    output = wording_output_type([request]).model_validate({"s0": {"g": text}})
    with pytest.raises(ValueError, match="generator context caption"):
        unpack_wording(output, [request])


@pytest.mark.parametrize(
    "caption", ["HS 854231", "HS:854231", "HS CODE-854231", "TARIFF CODE 854231", "HS-854231"]
)
def test_tariff_guard_distinguishes_captions_from_product_model_identifiers(caption):
    request = WordingRequest(
        "sample", "Electronic products", (WordingField("g", "goods description", "OLD", "Product"),)
    )
    schema = wording_output_type([request])
    assert unpack_wording(
        schema.model_validate({"s0": {"g": "HALL SENSOR MODEL HS-1881, DIP-8"}}), [request]
    )
    assert unpack_wording(
        schema.model_validate({"s0": {"g": "HALL SENSOR SERIES: HS-188124, DIP-8"}}), [request]
    )
    with pytest.raises(ValueError, match="tariff caption"):
        unpack_wording(schema.model_validate({"s0": {"g": "HALL SENSOR " + caption}}), [request])
    with pytest.raises(ValueError, match="tariff caption"):
        unpack_wording(
            schema.model_validate({"s0": {"g": "MODEL HS-1881, DIP-8; " + caption}}), [request]
        )


def test_disjoint_semantic_renderers_must_not_silently_overwrite_each_other():
    assert combine_surfaces({"a": "X"}, {"a": "X", "b": ["Y"]}) == {"a": "X", "b": ["Y"]}
    with pytest.raises(ValueError, match="conflicting render ownership"):
        combine_surfaces({"a": "X"}, {"a": "Y"})


def test_unrequested_country_is_detected_without_its_optional_parenthetical_qualifier():
    with pytest.raises(ValueError, match="country was not requested"):
        validate_postal_geography(
            "12 ROSS ROAD STANLEY FALKLAND ISLANDS",
            locality="STANLEY",
            country="FALKLAND ISLANDS (MALVINAS)",
            country_code="FK",
            country_labelled=False,
        )
    validate_postal_geography(
        "12 ROSS ROAD STANLEY",
        locality="STANLEY",
        country="FALKLAND ISLANDS (MALVINAS)",
        country_code="FK",
        country_labelled=False,
    )


def test_goods_fragments_share_full_brief_without_mapping_fragment_count_to_hs(postal_campaign):
    from dataclasses import replace

    fixture = postal_campaign
    contract = fixture.blueprint.contract.model_copy(deep=True)
    product = next(v for v in contract.variables if v.kind == "product")
    continuation = product.model_copy(
        update={
            "key": "continuation",
            "value": "COFFEE GRADE AA",
            "occurrences": [product.occurrences[0].model_copy(update={"text": "COFFEE GRADE AA"})],
        }
    )
    contract.variables.append(continuation)
    next(
        t for t in contract.targets if t.path.endswith(".description")
    ).expression = "{product} {continuation}"
    request = wording_request(replace(fixture.blueprint, contract=contract), fixture.scenario, "s")
    goods = [f for f in request.fields if f.role == "goods description"]
    assert len(goods) == 2 and len(fixture.scenario.goods_identities) == 1
    assert "{product} {continuation}" in request.context
    assert "one coherent commercial description" in request.context
    assert "never product text" in request.context
    assert "accounting goods group" not in request.context
    assert [f.example for f in goods] == ["OLD COFFEE", "COFFEE GRADE AA"]
    schema = wording_output_type([request])
    values = {f.key: "NEW WORDING" for f in request.fields}
    values.update(product="ARABICA COFFEE\nGRADE BB", continuation="ROASTING QUALITY")
    # Short schema keys own coverage; a missing continuation cannot pass native validation.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        schema.model_validate({"s0": {k: v for k, v in values.items() if k != "continuation"}})
    # One short unchanged continuation is legitimate; a wholly copied group is not.
    assert unpack_wording(
        schema.model_validate({"s0": {**values, "continuation": "COFFEE GRADE AA"}}), [request]
    )
    for changed, expected in [
        ({"product": "OLD COFFEE", "continuation": "COFFEE GRADE AA"}, "copied source"),
        ({"product": "NEW COFFEE TOTAL PACKAGES - 9"}, "shipment accounting"),
        ({"product": "NEW COFFEE (HS 090111)"}, "tariff caption"),
        ({"product": "NEW COFFEE, FREIGHT PREPAID"}, "non-product policy"),
        ({"product": "NEW COFFEE SOLD AS ONE ACCOUNTING GOODS GROUP"}, "non-product policy"),
        ({"product": "NEW COFFEE\nCOUNTRY OF ORIGIN: DENMARK"}, "non-product policy"),
    ]:
        bad = {**values, **changed}
        with pytest.raises(ValueError, match=expected):
            unpack_wording(schema.model_validate({"s0": bad}), [request])
    other = replace(request, sample_id="other")
    duplicate = wording_output_type([request, other]).model_validate({"s0": values, "s1": values})
    with pytest.raises(ValueError, match="duplicate generated description"):
        unpack_wording(duplicate, [request, other])


def test_represented_name_uses_its_complete_party_owner_geography(postal_campaign):
    from dataclasses import replace

    fixture = postal_campaign
    contract = fixture.blueprint.contract.model_copy(deep=True)
    next(
        t for t in contract.targets if t.path == "documentPatch.parties.shipper.name"
    ).expression = "{shipper_name} ON BEHALF OF {consignee_name}"
    request = wording_request(replace(fixture.blueprint, contract=contract), fixture.scenario, "s")
    represented = next(f for f in request.fields if f.key == "consignee_name")
    assert "New party in UTRECHT, NETHERLANDS" in represented.requirement
    assert "complete identity belongs to documentPatch.parties.consignee" in represented.requirement
    assert "ROTTERDAM" not in represented.requirement
    assert "shipper" in represented.role and "consignee" in represented.role
    principal = next(f for f in request.fields if f.key == "shipper_name")
    assert "New party in ROTTERDAM, NETHERLANDS" in principal.requirement


@pytest.mark.parametrize("ownership", ["no_direct_name", "two_direct_names", "shared_postal"])
def test_shared_regions_without_one_name_owner_keep_geography_conflict_guard(
    postal_campaign, ownership
):
    from dataclasses import replace

    fixture = postal_campaign
    contract = fixture.blueprint.contract.model_copy(deep=True)
    bindings = {t.path: t for t in contract.targets}
    if ownership == "no_direct_name":
        bindings[
            "documentPatch.parties.shipper.name"
        ].expression = "{shipper_name} ON BEHALF OF {consignee_name}"
        bindings["documentPatch.parties.consignee.name"].expression = "REPRESENTED {consignee_name}"
    elif ownership == "two_direct_names":
        bindings["documentPatch.parties.shipper.name"].expression = "{consignee_name}"
        contract.variables = [v for v in contract.variables if v.key != "shipper_name"]
    else:
        bindings[
            "documentPatch.parties.consignee.addressLine"
        ].expression = "{shipper_street} {consignee_address}"
    with pytest.raises(ValueError, match="conflicting sampled geography"):
        wording_request(replace(fixture.blueprint, contract=contract), fixture.scenario, "s")


def test_shared_complete_name_still_allows_multiple_owners_at_the_same_place(postal_campaign):
    from dataclasses import replace

    fixture = postal_campaign
    contract = fixture.blueprint.contract.model_copy(deep=True)
    next(
        t for t in contract.targets if t.path == "documentPatch.parties.shipper.name"
    ).expression = "{consignee_name}"
    contract.variables = [v for v in contract.variables if v.key != "shipper_name"]
    scenario = fixture.scenario.model_copy(deep=True)
    scenario.party_localities["documentPatch.parties.consignee"] = scenario.origin
    request = wording_request(replace(fixture.blueprint, contract=contract), scenario, "s")
    shared = next(f for f in request.fields if f.key == "consignee_name")
    assert "New party in ROTTERDAM, NETHERLANDS" in shared.requirement


@pytest.mark.parametrize("public_gross", [None, {"value": "13143.49", "unit": "kilogram"}])
def test_wording_context_and_bounds_use_printed_private_or_rounded_mass(
    postal_campaign, public_gross
):
    fixture = postal_campaign
    target = deepcopy(fixture.target)
    goods = target["documentPatch"]["goodsItemDetails"][0]
    if public_gross is not None:
        goods["grossWeight"] = {"value": "13143", "unit": "kilogram"}
    scenario = fixture.scenario.model_copy(
        deep=True, update={"cargo": {"goodsItemDetails": [{"description": "OLD COFFEE"}]}}
    )
    if public_gross is not None:
        scenario.cargo["goodsItemDetails"][0]["grossWeight"] = public_gross
    plan = PhysicalRenderPlan(
        target=target,
        variable_values={},
        surface_values={},
        receipt={
            "measures": {
                "grossWeight": {
                    "total": "13143",
                    "unit": "kilogram",
                    "method": "sampled_public_measure" if public_gross else "source_only_scaled",
                }
            }
        },
    )
    plain = wording_request(fixture.blueprint, scenario, "sample")
    request = wording_request(fixture.blueprint, scenario, "sample", physical_plan=plan)
    assert request.gross_weight_kg == Decimal("13143")
    assert "printed_measure_totals:" in request.context
    assert "13143.49" not in request.context
    assert wording_request_hash(request) != wording_request_hash(plain)
    values = {**fixture.receipt["values"], "product": "LOADER, OPERATING WEIGHT 13,143.1 KG"}
    output = wording_output_type([request]).model_validate({"s0": values})
    with pytest.raises(ValueError, match="item-mass lower bound"):
        unpack_wording(output, [request])
    values["product"] = "LOADER, OPERATING WEIGHT 13,143 KG"
    assert unpack_wording(wording_output_type([request]).model_validate({"s0": values}), [request])


def test_wording_receives_private_contained_packing(postal_campaign, monkeypatch):
    facts = [{"contained_quantity": 240, "printed_contained_unit": "SACOS"}]
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.contained_packing_context",
        lambda blueprint, target: facts,
    )
    fixture = postal_campaign
    request = wording_request(fixture.blueprint, fixture.scenario, "sample")
    assert "contained_packing:" in request.context
    assert "contained_quantity: 240" in request.context
    assert "printed_contained_unit: SACOS" in request.context


def test_campaign_wording_request_prepares_actual_physical_context(postal_campaign, monkeypatch):
    fixture = postal_campaign
    support = object()
    fixture.campaign.physical_support = support
    calls = []

    def physical(blueprint, scenario, target, *, support):
        calls.append((blueprint, scenario, target, support))
        return PhysicalRenderPlan(
            target=target,
            variable_values={},
            surface_values={},
            receipt={
                "measures": {
                    "grossWeight": {
                        "total": "13.143",
                        "unit": "metric_tonne",
                        "method": "source_only_scaled",
                    }
                }
            },
        )

    monkeypatch.setattr("document_ocr.synthesis.curated_campaign.prepare_physical_render", physical)
    request = Campaign.wording_request(
        fixture.campaign, fixture.blueprint, fixture.scenario, "sample", fixture.target
    )
    assert calls == [(fixture.blueprint, fixture.scenario, fixture.target, support)]
    assert request.gross_weight_kg == Decimal("13143")


def test_unasserted_source_zero_has_no_generation_measure_or_mass_bound(postal_campaign):
    fixture = postal_campaign
    plan = PhysicalRenderPlan(
        target=fixture.target,
        variable_values={},
        surface_values={},
        receipt={
            "measures": {
                "grossWeight": {"total": "0", "method": "source_zero_unasserted"},
                "volume": {"total": "0", "method": "source_zero_unasserted"},
            }
        },
    )
    request = wording_request(fixture.blueprint, fixture.scenario, "sample", physical_plan=plan)
    assert request.gross_weight_kg is None
    assert "printed_measure_totals: {}" in request.context
    values = {**fixture.receipt["values"], "product": "MODEL A, UNIT WEIGHT 400 KG"}
    assert unpack_wording(wording_output_type([request]).model_validate({"s0": values}), [request])


def test_generation_repairs_private_printed_gross_contradiction_before_saving(postal_campaign):
    fixture = postal_campaign
    fixture.path.unlink()
    plan = PhysicalRenderPlan(
        target=fixture.target,
        variable_values={},
        surface_values={},
        receipt={
            "measures": {
                "grossWeight": {
                    "total": "13143",
                    "unit": "kilogram",
                    "method": "source_only_scaled",
                }
            }
        },
    )
    fixture.campaign.wording_request = lambda bp, scenario, sid, target: wording_request(
        bp, scenario, sid, physical_plan=plan
    )
    calls = []

    async def call(stage, sid, output_type, system, prompt):
        calls.append(stage)
        values = dict(fixture.receipt["values"])
        assert "printed_measure_totals:" in prompt
        assert "13143" in prompt
        values["product"] = (
            "LOADER, OPERATING WEIGHT 21,000 KG"
            if stage == "exact-wording"
            else "LOADER WITH ENCLOSED CAB"
        )
        return output_type.model_validate({"s0": values})

    fixture.campaign.calls.call = call
    result = asyncio.run(fixture.campaign.generate("source"))
    assert calls == ["exact-wording", "wording-validation-repair"]
    assert "item-mass lower bound" in result["wordingValidationRepair"]
    assert json.loads(fixture.path.read_text())["values"]["product"] == "LOADER WITH ENCLOSED CAB"


@pytest.mark.parametrize("failure", ["physical", "unowned_target"])
def test_generation_preflight_failure_makes_no_paid_request(postal_campaign, monkeypatch, failure):
    fixture = postal_campaign
    fixture.path.unlink()
    calls = []

    async def call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("paid requests must not run after failed preflight")

    def invalid_physical_contract(*args, **kwargs):
        if failure == "physical":
            raise ValueError("unowned printed shipment mass")
        target = deepcopy(fixture.target)
        target["documentPatch"]["parties"]["shipper"]["country"] = "FRANCE"
        return SimpleNamespace(target=target)

    fixture.campaign.calls.call = call
    fixture.campaign.physical_support = object()
    fixture.campaign.preflight = Campaign.preflight.__get__(fixture.campaign, Campaign)
    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.prepare_physical_render",
        invalid_physical_contract,
    )
    message = (
        "unowned printed shipment mass"
        if failure == "physical"
        else "sampled target fields lack source render ownership"
    )
    with pytest.raises(ValueError, match=message):
        asyncio.run(fixture.campaign.generate("source"))
    assert calls == []
    assert not fixture.path.exists()


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
    validate_postal_geography(
        "14 STREET BELIZE CITY, BELIZE",
        locality="BELIZE CITY",
        country="BELIZE",
        country_code="BZ",
        country_labelled=True,
    )
    with pytest.raises(ValueError, match="country must occur once"):
        validate_postal_geography(
            "14 STREET BELIZE CITY",
            locality="BELIZE CITY",
            country="BELIZE",
            country_code="BZ",
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
    campaign.config = {"variants_per_source": 2}
    campaign.output = tmp_path
    campaign.plans = lambda sid: [(blueprint, scenario, sample_id, deepcopy(target))]
    campaign.preflight = lambda plans: None  # These fixtures isolate paid-wording behavior.
    campaign.wording_request = lambda bp, scenario, sid, target: wording_request(bp, scenario, sid)
    campaign.calls = SimpleNamespace()
    return SimpleNamespace(
        campaign=campaign,
        path=path,
        receipt=receipt,
        blueprint=blueprint,
        scenario=scenario,
        target=target,
    )


def test_regulated_and_whole_unit_products_have_host_owned_identities(postal_campaign):
    fixture = postal_campaign
    assert host_product_wording(fixture.blueprint, fixture.scenario) == {}
    for provenance in (
        {"physicalProfile": "source_whole_units"},
        {"physicalProfile": "reviewed_whole_units"},
        {"capabilities": {"family": "dg_chemical"}},
    ):
        scenario = fixture.scenario.model_copy(update={"provenance": provenance})
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


def test_target_casing_is_explicit_and_does_not_change_identifiers(postal_campaign):
    from document_ocr.synthesis.curated_publication import review_contract_hash
    from document_ocr.synthesis.curated_wording import rendered_review_prompt

    fixture = postal_campaign
    values = {k: v.title() for k, v in fixture.receipt["values"].items()}
    target = deepcopy(fixture.target)
    target["documentPatch"]["billOfLadingNumber"] = "Bl-ab123"
    for mode in ("uppercase", "preserve"):
        out = assemble_lexical_target(
            fixture.blueprint, target, values, fixture.scenario, target_casing=mode
        )
        name = out["documentPatch"]["parties"]["shipper"]["name"]
        assert name == (
            values["shipper_name"].upper() if mode == "uppercase" else values["shipper_name"]
        )
        assert out["documentPatch"]["billOfLadingNumber"] == "Bl-ab123"
    assert "preserve generated casing" in rendered_review_prompt("preserve")
    assert review_contract_hash(1, "preserve") != review_contract_hash(1, "uppercase")


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
        assert "Choose new street/site wording" in prompt
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
