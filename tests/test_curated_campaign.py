import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import Runner, SourceContract, digest
from document_ocr.synthesis.curated_campaign import (
    Campaign,
    _fixed_postal_country_components,
    _handling_replay_target,
    assemble_lexical_target,
    combine_surfaces,
    host_product_wording,
    postal_assembly_context,
    postal_wording_errors,
    public_identity_updates,
    validate_sample,
    wording_request,
    wording_request_hash,
)
from document_ocr.synthesis.curated_physical import PhysicalRenderPlan
from document_ocr.synthesis.curated_scenarios import ScenarioLocation, ShipmentScenario
from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.curated_wording import (
    WORDING_PROMPT,
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
    scenario = SimpleNamespace(
        party_localities={}, replacements=replacements, goods_identities=[], provenance={}
    )
    proof = {
        "changedTargetPaths": ["documentPatch.goodsItemDetails[0].description"],
        "coveredTargetPaths": ["documentPatch.goodsItemDetails[0].description"],
        "edits": [],
    }
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


def _handling_replay_fixture():
    from document_ocr.synthesis.curated_physical import project_nested_equipment_handling
    from document_ocr.synthesis.curated_templates import RenderRegion

    source = "KEEP AT 3C IN 2X20DC\nOTHER HANDLING: KEEP COOL\n"
    original = {
        "documentPatch": {
            "goodsItemDetails": [{"handlingInstructions": ["KEEP AT 3C IN 2X20DC", "KEEP COOL"]}]
        }
    }
    start = source.index("2X20DC")
    region = RenderRegion(
        "equipment",
        0,
        start,
        start + 6,
        "2X20DC",
        ("documentPatch.containerInformation[0].sizeCategory",),
        "equipment",
    )
    blueprint = SimpleNamespace(source=source, target=original, regions=(region,))
    edits = [
        dict(key="temperature", occurrence=0, byteStart=8, byteEnd=9, before="3", after="-18"),
        dict(
            key="equipment",
            occurrence=0,
            byteStart=start,
            byteEnd=start + 6,
            before="2X20DC",
            after="2 x 40RQ",
        ),
    ]
    rendered = "KEEP AT -18C IN 2 x 40RQ\nOTHER HANDLING: KEEP COOL\n"
    planned = deepcopy(original)
    planned["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][0] = (
        "KEEP AT -18C IN 2X20DC"
    )
    target, receipt = project_nested_equipment_handling(blueprint, rendered, planned, edits)
    return blueprint, {
        "target": target,
        "joinedRawText": rendered,
        "joinedRawTextSha256": digest(rendered.encode()),
        "proof": {"edits": edits, "handlingProjection": receipt},
    }


def test_handling_replay_restores_only_exact_verified_parent_and_is_nonmutating():
    blueprint, candidate = _handling_replay_fixture()
    before = deepcopy(candidate)
    target = _handling_replay_target(blueprint, candidate, casing="uppercase")
    assert target == blueprint.target
    assert candidate == before
    assert target is not candidate["target"]


@pytest.mark.parametrize(
    "defect",
    [
        "before",
        "value",
        "source_hash",
        "render_hash",
        "candidate_hash",
        "spans",
        "unrelated_parent",
        "target",
        "missing_fields",
        "casing",
    ],
)
def test_handling_replay_rejects_forged_receipt_or_target_without_masking_other_fields(defect):
    blueprint, candidate = _handling_replay_fixture()
    receipt = candidate["proof"]["handlingProjection"]
    field = receipt["fields"][0]
    if defect in {"before", "value"}:
        field[defect] = "FORGED"
    elif defect in {"source_hash", "render_hash"}:
        receipt["sourceSha256" if defect == "source_hash" else "renderedSha256"] = "0" * 64
    elif defect == "candidate_hash":
        candidate["joinedRawTextSha256"] = "0" * 64
    elif defect == "spans":
        field["spans"][0]["sourceByteStart"] += 1
    elif defect == "unrelated_parent":
        field["path"] = "documentPatch.goodsItemDetails[0].handlingInstructions[1]"
        field["before"] = "KEEP COOL"
        field["value"] = "FORGED"
        candidate["target"]["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][1] = (
            "FORGED"
        )
    elif defect == "target":
        candidate["target"]["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][0] = (
            "FORGED"
        )
    elif defect == "missing_fields":
        receipt["fields"] = []
    else:
        receipt["casing"] = "preserve"
    before = deepcopy(candidate)
    with pytest.raises(ValueError):
        _handling_replay_target(blueprint, candidate, casing="uppercase")
    assert candidate == before


def test_handling_replay_missing_receipt_cannot_silently_reset_a_label():
    blueprint, candidate = _handling_replay_fixture()
    candidate["proof"].pop("handlingProjection")
    result = _handling_replay_target(blueprint, candidate, casing="uppercase")
    assert result is candidate["target"]
    assert result != blueprint.target


def test_handling_replay_never_masks_unrelated_handling_changes():
    blueprint, candidate = _handling_replay_fixture()
    candidate["target"]["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"][1] = (
        "FORGED UNRELATED HANDLING"
    )
    before = deepcopy(candidate)
    result = _handling_replay_target(blueprint, candidate, casing="uppercase")
    assert result["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "KEEP AT 3C IN 2X20DC",
        "FORGED UNRELATED HANDLING",
    ]
    assert candidate == before


def test_handling_replay_enforces_current_configured_casing():
    blueprint, candidate = _handling_replay_fixture()
    with pytest.raises(ValueError, match="projection receipt"):
        _handling_replay_target(blueprint, candidate, casing="preserve")


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
    campaign.blueprint = lambda sid: SimpleNamespace(
        ownership_data={}, contract=SimpleNamespace(variables=[])
    )
    (tmp_path / "candidates").mkdir()
    candidates = []
    for variant in range(1, 6):
        did = "syn_full_v7_" + digest(["source", 42, variant])[:24]
        candidate = {
            "documentId": did,
            "sourceDocumentId": "source",
            "variant": variant,
            "target": {},
            "joinedRawText": f"TEXT {variant}",
            "renderValues": {},
            "scenario": {
                "party_localities": {},
                "goods_identities": [
                    {
                        "hs6": "090111",
                        "phrase": "GREEN COFFEE",
                        "classification": {
                            "authority": "fixture",
                            "chapter_code": "09",
                            "chapter_description": "Coffee, tea and spices",
                            "heading_code": "0901",
                            "heading_description": "Coffee",
                            "description_path": ["Coffee", "Not roasted"],
                            "same_heading_alternatives": [],
                            "immediate_national_children": [],
                        },
                    }
                ],
                "provenance": {},
            },
        }
        (tmp_path / "candidates" / f"{did}.json").write_text(json.dumps(candidate))
        candidates.append(candidate)
    snapshot = deepcopy(candidates)
    prepared = campaign.review_requests("source", candidates)
    assert candidates == snapshot and not (tmp_path / "reviews").exists()
    assert [r.sample_ids for r in prepared] == [
        tuple(c["documentId"] for c in candidates[:2]),
        tuple(c["documentId"] for c in candidates[2:4]),
        (candidates[4]["documentId"],),
    ]
    assert [list(r.output_type.model_fields) for r in prepared] == [
        ["s0", "s1"],
        ["s0", "s1"],
        ["s0"],
    ]
    with pytest.raises(ValueError, match="candidate count"):
        campaign.review_requests("source", candidates[:-1])
    with pytest.raises(ValueError, match="candidate count"):
        campaign.review_requests("foreign-source", candidates)
    for invalid in (candidates[::-1], [candidates[0]] * len(candidates)):
        with pytest.raises(ValueError, match="variant order"):
            campaign.review_requests("source", invalid)
    for key, value in (("sourceDocumentId", "foreign"), ("documentId", "wrong-id"), ("variant", 0)):
        invalid = deepcopy(candidates)
        invalid[0][key] = value
        with pytest.raises(ValueError, match="variant order"):
            campaign.review_requests("source", invalid)
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
    assert requests == [request.prompt for request in prepared]
    richer = deepcopy(candidates)
    richer[0]["scenario"]["goods_identities"][0]["phrase_role"] = "classification_context"
    richer[0]["scenario"]["provenance"]["registryLoadProfile"] = {"form": "DRY GRANULES"}
    richer[1]["scenario"]["provenance"]["thermalCommodityContext"] = {
        "identity": "FROZEN PEAS",
        "temperatureCelsius": -18,
    }
    context = campaign.review_requests("source", richer)[0].prompt
    assert "CLASSIFICATION REFERENCES" in context and "Physical form: DRY GRANULES" in context
    assert "SAMPLED COLD-CHAIN IDENTITY, PRODUCT FORM AND SETTINGS" in context
    assert "temperatureCelsius: -18" in context
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
        (0, "NORTH \u0421OMPANY", "Latin script"),
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
        "POLYETHYLENE GRANULES, EACH BAG 25 KG NET",
        "POLYETHYLENE GRANULES, 25 KG NET EACH BAG",
        "POLYETHYLENE GRANULES, BAGS OF 25 KG NET EACH",
    ],
)
def test_per_package_fill_is_not_itself_a_shipment_total(text):
    request = WordingRequest(
        "sample", "Sampled goods", (WordingField("g", "goods description", "OLD", "Product"),)
    )
    output = wording_output_type([request]).model_validate({"s0": {"g": text}})
    assert unpack_wording(output, [request])["sample"]["g"] == text


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


@pytest.mark.parametrize("role", ["goods description", "shipper name", "shipper postal"])
@pytest.mark.parametrize("text", ["TBD", "tbd", "  TbD\n\n"])
def test_wording_rejects_complete_drafting_placeholder(role, text):
    request = WordingRequest("sample", "Shipment", (WordingField("v", role, "OLD", "Final text"),))
    output = wording_output_type([request]).model_validate({"s0": {"v": text}})
    with pytest.raises(ValueError, match="unfinished generated region"):
        unpack_wording(output, [request])


@pytest.mark.parametrize(
    ("role", "text"),
    [
        ("goods description", "CONTROL MODULE MODEL TBD-42"),
        ("goods description", "CONTROL MODULE\nMODEL: TBD"),
        ("shipper name", "TBD INDUSTRIAL SYSTEMS LTD"),
        ("shipper postal", "12 TBD INDUSTRIAL ROAD"),
    ],
)
def test_wording_placeholder_guard_preserves_embedded_codes_and_names(role, text):
    request = WordingRequest("sample", "Shipment", (WordingField("v", role, "OLD", "Final text"),))
    output = wording_output_type([request]).model_validate({"s0": {"v": text}})
    assert unpack_wording(output, [request])["sample"]["v"] == text


@pytest.mark.parametrize("key_line", ["product_1", "PRODUCT_1", "PrOdUcT_1", "  product_1  "])
def test_goods_wording_rejects_standalone_actual_request_keys(key_line):
    request = WordingRequest(
        "sample",
        "Goods",
        (
            WordingField("product_0", "goods description", "OLD", "Product"),
            WordingField("product_1", "goods description", "OLD SPEC", "Specification"),
        ),
    )
    output = wording_output_type([request]).model_validate(
        {"s0": {"product_0": "COPPER WIRE\n" + key_line, "product_1": "DIAMETER 2 MM"}}
    )
    with pytest.raises(ValueError, match="requested field key"):
        unpack_wording(output, [request])


@pytest.mark.parametrize(
    "text",
    [
        "COPPER WIRE, DIAMETER 2 MM",
        "SHELF BRACKET, RATED CAPACITY 45 KG PER PAIR",
        "MODEL product_1, 24 VOLT",
        "CONTROL MODULE\nMODEL: PRODUCT_1",
        "PRODUCT_10",  # Whole-line equality, not a key-prefix match.
        "lexical_00",  # A code outside this request's actual field keys.
    ],
)
def test_goods_wording_key_guard_preserves_commercial_specs_and_embedded_codes(text):
    request = WordingRequest(
        "sample",
        "Goods",
        (
            WordingField("product_0", "goods description", "OLD", "Product"),
            WordingField("product_1", "goods description", "OLD SPEC", "Specification"),
        ),
    )
    output = wording_output_type([request]).model_validate(
        {"s0": {"product_0": text, "product_1": "GRADE A"}}
    )
    assert unpack_wording(output, [request])["sample"]["product_0"] == text


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
    assert unpack_wording(duplicate, [request, other]) == {"s": values, "other": values}


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


@pytest.mark.parametrize("ownership", ["no_direct_name", "shared_postal"])
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


def test_shared_company_identity_can_have_distinct_postal_offices(postal_campaign):
    from dataclasses import replace

    fixture = postal_campaign
    contract = fixture.blueprint.contract.model_copy(deep=True)
    next(
        t for t in contract.targets if t.path == "documentPatch.parties.shipper.name"
    ).expression = "{consignee_name}"
    contract.variables = [v for v in contract.variables if v.key != "shipper_name"]
    request = wording_request(replace(fixture.blueprint, contract=contract), fixture.scenario, "s")
    shared = next(f for f in request.fields if f.key == "consignee_name")
    assert "One new fictional identity" in shared.requirement
    assert "ROTTERDAM, NETHERLANDS" in shared.requirement
    assert "UTRECHT, NETHERLANDS" in shared.requirement


def test_cached_wording_new_policy_failure_uses_recorded_bounded_repair(postal_campaign):
    fixture = postal_campaign
    cached = json.loads(fixture.path.read_text())
    cached["values"]["product"] = "\u0421OTTON YARN"
    fixture.path.write_text(json.dumps(cached))
    calls = []

    async def call(stage, sid, output_type, system, prompt):
        calls.append(stage)
        values = dict(fixture.receipt["values"])
        values["product"] = "\u0421OTTON YARN" if stage == "exact-wording" else "COTTON YARN"
        return output_type.model_validate({"s0": values})

    fixture.campaign.calls.call = call
    result = asyncio.run(fixture.campaign.generate("source"))
    assert calls == ["exact-wording", "wording-validation-repair"]
    assert "Latin script" in result["wordingValidationRepair"]
    assert json.loads(fixture.path.read_text())["values"]["product"] == "COTTON YARN"


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
    assert "TRANSPORT COMPATIBILITY" in request.context
    assert "printed_measure_totals:" not in request.context
    assert "13143" not in request.context
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
    assert "containedPackageWordings:" in request.context
    assert "SACOS" in request.context
    assert "240" not in request.context
    assert "contained_quantity" not in request.context


def test_wording_preserves_source_presentation_without_universal_assortments(postal_campaign):
    request = wording_request(
        postal_campaign.blueprint, postal_campaign.scenario, "source-presentation"
    )
    assert (
        "concise commodity identity, a specification passage, or a product list" in request.context
    )
    assert "Represent every sampled identity at that level of detail" in request.context
    assert "plausible commercial assortment" not in request.context
    assert "concise identity, specification passage, or product list" in WORDING_PROMPT
    assert "without line or\ncharacter quotas" in WORDING_PROMPT
    assert request.description_groups
    assert any(field.example for field in request.fields if field.role == "goods description")


def test_campaign_wording_request_prepares_actual_physical_context(postal_campaign, monkeypatch):
    fixture = postal_campaign
    support = object()
    fixture.campaign.physical_support = support
    calls = []

    def physical(blueprint, scenario, target, *, support, equipment_inventory):
        assert equipment_inventory is None
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
    assert "printed_measure_totals" not in request.context
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
        assert "printed_measure_totals:" not in prompt
        assert ("13143" in prompt) == (stage == "wording-validation-repair")
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


def test_source_snapshot_checks_content_even_when_file_metadata_is_preserved(tmp_path):
    import os

    path = tmp_path / "ownership.yaml"
    path.write_bytes(b"certified")
    stamp = path.stat()
    campaign = Campaign.__new__(Campaign)
    campaign.source_ids = ["source"]
    campaign._blueprints = {
        "source": SimpleNamespace(
            ownership_data={"identifier_privacy": {}},
            target={"documentPatch": {}},
            regions=[],
            historical_bindings={},
        )
    }
    campaign._blueprint_file_pins = {path: digest(path.read_bytes())}
    campaign.equipment_inventories = {}
    campaign.validate_source_readiness()
    path.write_bytes(b"incorrect")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="source files changed after snapshot certification"):
        campaign.validate_source_readiness()


@pytest.mark.parametrize("preserve_metadata", [False, True])
def test_render_stage_rejects_changed_cached_ownership_before_any_render(
    tmp_path, preserve_metadata
):
    import os

    path = tmp_path / "ownership.yaml"
    path.write_bytes(b"certified")
    stamp = path.stat()
    campaign = Campaign.__new__(Campaign)
    campaign.source_ids = ["source"]
    campaign.variant_counts = {"source": 2}
    campaign._blueprints = {
        "source": SimpleNamespace(
            ownership_data={"identifier_privacy": {}},
            target={"documentPatch": {}},
            regions=[],
            historical_bindings={},
        )
    }
    campaign._blueprint_file_pins = {path: digest(path.read_bytes())}
    campaign.equipment_inventories = {}
    campaign.output = tmp_path / "output"
    rendered = []
    campaign.render = lambda *args: rendered.append(args)
    path.write_bytes(b"incorrect")
    if preserve_metadata:
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))

    with pytest.raises(ValueError, match="source files changed after snapshot certification"):
        campaign.render_all()

    assert rendered == []
    assert not campaign.output.exists()


def test_render_stage_certifies_snapshot_once_not_once_per_document(tmp_path):
    path = tmp_path / "ownership.yaml"
    path.write_bytes(b"certified")
    campaign = Campaign.__new__(Campaign)
    campaign.source_ids = ["source"]
    campaign.variant_counts = {"source": 3}
    campaign._blueprints = {
        "source": SimpleNamespace(
            ownership_data={"identifier_privacy": {}},
            target={"documentPatch": {}},
            regions=[],
            historical_bindings={},
        )
    }
    campaign._blueprint_file_pins = {path: digest(path.read_bytes())}
    campaign.equipment_inventories = {}
    campaign.output = tmp_path / "output"
    observed = []

    def certify():
        observed.append("snapshot")
        Campaign.validate_source_readiness(campaign)

    def render(sid, variant):
        observed.append((sid, variant))
        return {"documentId": f"sample-{variant}"}

    campaign.validate_source_readiness = certify
    campaign.render = render
    report = campaign.render_all()

    assert observed == ["snapshot", ("source", 1), ("source", 2), ("source", 3)]
    assert report["accepted"] == ["sample-1", "sample-2", "sample-3"]
    assert report["errors"] == []
    assert json.loads((campaign.output / "render-summary.json").read_text()) == report


def test_classification_context_is_not_a_second_product_form(postal_campaign):
    fixture = postal_campaign
    identity = fixture.scenario.goods_identities[0].model_copy(
        update={"phrase": "STEEL, NOT IN COILS", "phrase_role": "classification_context"}
    )
    scenario = fixture.scenario.model_copy(
        update={
            "goods_identities": (identity,),
            "provenance": {
                "registryLoadProfile": {
                    "form": "cold-rolled steel coils",
                    "package_category": "PACKAGE_COIL",
                }
            },
        }
    )
    request = wording_request(fixture.blueprint, scenario, "context-proof")
    assert request.context.count("STEEL, NOT IN COILS") == 1
    assert "contextual commodity scope, not shipment requirements" in request.context
    assert "Physical form: cold-rolled steel coils" in request.context
    assert "Sampled commodity HS6 references: 090111" in request.context
    assert "Include every sampled identity and its attributes" not in request.fields[-1].requirement


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
    blueprint = SamplingBlueprint(
        source_id,
        "SOURCE",
        target,
        contract,
        (),
        {},
        {},
        (),
        ownership_data={"identifier_privacy": {}},
    )
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
    campaign.equipment_inventories = {}
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
        original_fields = {
            field.key: field
            for field in fixture.campaign.wording_request(
                fixture.blueprint, fixture.scenario, "sample", fixture.target
            ).fields
        }
        for key, field in shipment_schema.model_fields.items():
            assert original_fields[key].requirement in field.description
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


@pytest.mark.parametrize(
    "value,expression,countries,fixed",
    [
        ("NETHERLANDS", "{country}", ["NETHERLANDS"], True),
        ("NETHERLANDS", "{country}", ["NETHERLANDS", "NETHERLANDS"], True),
        ("NETHERLANDS", "{country}", ["NETHERLANDS", "BELGIUM"], False),
        ("14 CANAL ROAD NETHERLANDS", "{country}", ["NETHERLANDS"], False),
        ("GERMANY", "{country}", ["NETHERLANDS"], False),
        ("Netherlands", "{country}", ["NETHERLANDS"], False),
        ("NL", "{country}", ["NETHERLANDS"], False),
        ("NETHERLANDS", "{country} NL", ["NETHERLANDS"], False),
        ("NETHERLANDS", "{country}", [], False),
    ],
)
def test_postal_country_is_fixed_only_by_exact_owners_and_all_canonical_values(
    value, expression, countries, fixed
):
    roles = [f"documentPatch.parties.notifyParties[{i}]" for i in range(len(countries))]
    blueprint = SimpleNamespace(
        contract=SimpleNamespace(
            targets=[
                SimpleNamespace(path=role + ".country", expression=expression) for role in roles
            ]
        )
    )
    scenario = SimpleNamespace(
        party_localities={
            role: SimpleNamespace(country=country)
            for role, country in zip(roles, countries, strict=True)
        }
    )
    assert _fixed_postal_country_components(
        blueprint, scenario, {"country": value}, {"country"}
    ) == ({"country"} if fixed else set())


def test_postal_country_fix_never_infers_country_from_a_route_or_unowned_literal():
    blueprint = SimpleNamespace(
        contract=SimpleNamespace(
            targets=[
                SimpleNamespace(path="documentPatch.route.portOfLoading.country", expression="{x}"),
                SimpleNamespace(path="documentPatch.parties.shipper.addressLine", expression="{x}"),
            ]
        )
    )
    assert _fixed_postal_country_components(
        blueprint, SimpleNamespace(party_localities={}), {"x": "NETHERLANDS"}, {"x"}
    ) == set()


def _country_component_postal_fixture(fixture, country_value):
    role = "documentPatch.parties.shipper"
    payload = fixture.blueprint.contract.model_dump(mode="json")
    payload["variables"].append(
        {
            "key": "shipper_country",
            "kind": "postal",
            "value": "NETHERLANDS",
            "meaning": "Exact, independently source-owned country component.",
            "required_literals": [],
            "occurrences": [{"text": "NETHERLANDS", "occurrence": 1, "presentation": "text"}],
        }
    )
    for binding in payload["targets"]:
        if binding["path"] == role + ".addressLine":
            binding["expression"] = "{shipper_street} {shipper_locality} {shipper_country}"
    payload["targets"].append({"path": role + ".country", "expression": "{shipper_country}"})
    blueprint = replace(
        fixture.blueprint,
        contract=SourceContract.model_validate(payload),
        regions=(
            SimpleNamespace(
                kind="postal", curated_key="shipper_country", target_paths=(role + ".country",)
            ),
        ),
    )
    receipt = deepcopy(fixture.receipt)
    receipt["values"].update(shipper_locality="ROTTERDAM", shipper_country=country_value)
    receipt["requestSha256"] = wording_request_hash(
        wording_request(blueprint, fixture.scenario, "sample")
    )
    fixture.path.write_text(json.dumps(receipt))
    fixture.campaign.plans = lambda sid: [
        (blueprint, fixture.scenario, "sample", deepcopy(fixture.target))
    ]
    return blueprint, receipt


@pytest.mark.parametrize("country_value", ["NETHERLANDS", "GERMANY", "14 CANAL ROAD NETHERLANDS"])
def test_postal_correction_freezes_good_country_and_still_repairs_wrong_country(
    postal_campaign, country_value
):
    fixture = postal_campaign
    blueprint, receipt = _country_component_postal_fixture(fixture, country_value)
    correction = {"shipper_street": "14 CANAL ROAD, 3011AA", "shipper_locality": "ROTTERDAM"}
    if country_value != "NETHERLANDS":
        correction["shipper_country"] = "NETHERLANDS"

    async def call(stage, sid, output_type, system, prompt):
        shipment = output_type.model_fields["s0"].annotation
        assert set(shipment.model_fields) == set(correction)
        assert "Fresh synthetic postal component" in prompt
        if country_value == "NETHERLANDS":
            assert "shipper_country" not in shipment.model_fields
            assert "country is already supplied by a host-owned literal" in prompt
        else:
            assert "Certified country owner: NETHERLANDS" in (
                shipment.model_fields["shipper_country"].description
            )
        return output_type.model_validate({"s0": correction})

    fixture.campaign.calls.call = call
    assert asyncio.run(fixture.campaign.correct_postal("source"))["corrected"] == 1
    repaired = json.loads(fixture.path.read_text())
    assert repaired["values"] == {**receipt["values"], **correction}
    assert repaired["requestSha256"] == receipt["requestSha256"]
    assert set(repaired["corrections"][-1]["after"]) == set(correction)
    assert (
        postal_wording_errors(blueprint, fixture.scenario, fixture.target, repaired["values"])
        == {}
    )


def test_failed_address_with_only_fixed_country_surfaces_unrepairable_ownership(postal_campaign):
    fixture = postal_campaign
    blueprint, receipt = _country_component_postal_fixture(fixture, "NETHERLANDS")
    contract = blueprint.contract.model_copy(
        update={
            "variables": [
                variable
                for variable in blueprint.contract.variables
                if variable.key not in {"shipper_street", "shipper_locality"}
            ],
            "targets": [
                binding.model_copy(update={"expression": "{shipper_country}"})
                if binding.path == "documentPatch.parties.shipper.addressLine"
                else binding
                for binding in blueprint.contract.targets
            ]
        }
    )
    blueprint = replace(blueprint, contract=contract)
    for key in ("shipper_street", "shipper_locality"):
        del receipt["values"][key]
    receipt["requestSha256"] = wording_request_hash(
        wording_request(blueprint, fixture.scenario, "sample")
    )
    fixture.path.write_text(json.dumps(receipt))
    before = fixture.path.read_bytes()
    fixture.campaign.plans = lambda sid: [
        (blueprint, fixture.scenario, "sample", deepcopy(fixture.target))
    ]

    async def call(*args):
        pytest.fail("country-only source ownership cannot authorize invented address fields")

    fixture.campaign.calls.call = call
    with pytest.raises(ValueError, match="no owned mutable regions"):
        asyncio.run(fixture.campaign.correct_postal("source"))
    assert fixture.path.read_bytes() == before


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


def test_postal_correction_omitted_country_is_not_overridden_by_global_prompt(postal_campaign):
    fixture = postal_campaign
    del fixture.target["documentPatch"]["parties"]["shipper"]["country"]
    fixture.receipt["requestSha256"] = wording_request_hash(
        wording_request(fixture.blueprint, fixture.scenario, "sample")
    )
    fixture.path.write_text(json.dumps(fixture.receipt))

    async def call(stage, sid, output_type, system, prompt):
        assert "Some parties omit country" in system
        assert "COUNTRY: omitted" in prompt
        assert "Country NETHERLANDS appears once" not in prompt
        return output_type.model_validate(
            {
                "s0": {
                    "shipper_street": "14 CANAL ROAD, 3011AA",
                    "shipper_locality": "ROTTERDAM",
                }
            }
        )

    fixture.campaign.calls.call = call
    assert asyncio.run(fixture.campaign.correct_postal("source"))["corrected"] == 1


@pytest.mark.parametrize(
    "country,code,bad,good",
    [
        ("CURAÇAO", "CW", "CURACAO", "CURAÇAO"),
        ("SENEGAL", "SN", "SÉNÉGAL", "SENEGAL"),
        ("MACAO", "MO", "MACAO (MO)", "MACAO"),
    ],
)
def test_postal_country_spelling_and_name_code_duplicates_are_correction_candidates(
    postal_campaign,
    country,
    code,
    bad,
    good,
):
    fixture = postal_campaign
    role = "documentPatch.parties.shipper"
    fixture.target["documentPatch"]["parties"]["shipper"]["country"] = country
    scenario = fixture.scenario.model_copy(
        update={
            "party_localities": {
                **fixture.scenario.party_localities,
                role: fixture.scenario.origin.model_copy(
                    update={"country": country, "country_code": code}
                ),
            }
        }
    )
    blueprint = replace(
        fixture.blueprint,
        regions=(
            SimpleNamespace(
                kind="postal",
                curated_key="shipper_locality",
                target_paths=(role + ".country",),
            ),
        ),
    )
    values = {
        **fixture.receipt["values"],
        "shipper_street": "14 CANAL ROAD",
        "shipper_locality": "ROTTERDAM " + bad,
    }
    assert role in postal_wording_errors(blueprint, scenario, fixture.target, values)
    values["shipper_locality"] = "ROTTERDAM " + good
    assert postal_wording_errors(blueprint, scenario, fixture.target, values) == {}
    context = postal_assembly_context(blueprint, scenario, fixture.target, {"shipper_locality"})
    assert "COUNTRY OWNER REGIONS: {shipper_locality}" in context
    assert "spelling and accents" in context
    assert "without an added ISO code" in context


def test_country_in_wrong_postal_region_is_detected_even_when_full_address_passes(postal_campaign):
    fixture = postal_campaign
    role = "documentPatch.parties.shipper"
    blueprint = replace(
        fixture.blueprint,
        regions=(
            SimpleNamespace(
                kind="postal",
                curated_key="shipper_street",
                target_paths=(role + ".country",),
            ),
        ),
    )
    values = {**fixture.receipt["values"], "shipper_street": "14 CANAL ROAD"}
    errors = postal_wording_errors(blueprint, fixture.scenario, fixture.target, values)
    assert len(errors[role]) == 1
    assert "Region shipper_street: sampled country absent" in errors[role][0]
    values.update(shipper_street="14 CANAL ROAD, NETHERLANDS", shipper_locality="ROTTERDAM")
    assert postal_wording_errors(blueprint, fixture.scenario, fixture.target, values) == {}


def test_postal_assembly_context_preserves_literal_country_and_country_code(postal_campaign):
    fixture = postal_campaign
    path = "documentPatch.parties.shipper.addressLine"
    for literal in ("NETHERLANDS", "NL"):
        contract = fixture.blueprint.contract.model_copy(
            update={
                "targets": [
                    binding.model_copy(
                        update={"expression": "{shipper_street} {shipper_locality} " + literal}
                    )
                    if binding.path == path
                    else binding
                    for binding in fixture.blueprint.contract.targets
                ]
            }
        )
        blueprint = replace(fixture.blueprint, contract=contract)
        context = postal_assembly_context(
            blueprint, fixture.scenario, fixture.target, {"shipper_street"}
        )
        assert "{shipper_street} {shipper_locality} " + literal in context
        assert "country is already supplied by a host-owned literal" in context
        assert "keep generated regions free" in context


@pytest.mark.parametrize(
    "caption",
    [
        "POSTAL ASSEMBLY",
        "country owner regions",
        "locality owner regions",
        "INVALID OLD VALUE",
        "previous complete address",
        "POSTAL CORRECTION",
        'POSTAL_COMPONENT_2": "',
        "CONSIGNEE_POSTAL_COMPONENT_3:",
        "shipper_postal_component_11': '",
        '{"POSTAL_COMPONENT_4": "ROTTERDAM"}',
        "documentPatch.parties.shipper.address_line_1:",
        'documentPatch.parties.consignee.addressLine": "',
        "documentPatch.parties.deliveryAgent.address_line:",
        "documentPatch.parties.notifyParties[0].addressLine:",
        "documentPatch.parties.notifyParties[12].address_line_3:",
        "documentPatch.parties.shipper:",
        "documentPatch.parties.consignee:",
        'documentPatch.parties.deliveryAgent": "',
        "documentPatch.parties.notifyParties[0]:",
        "deliveryagent_postal_component_2",
        "shipper_postal_component_11",
        "NOTIFY_PARTY_POSTAL_COMPONENT_3",
        "REGION LEXICAL_00",
        "lexical_12_3",
        "REGION delivery_postal (documentPatch.parties.deliveryAgent postal)",
        "REGION notify_postal (documentPatch.parties.notifyParties[0] postal)",
        "region consignee_postal (documentPatch.parties.consignee postal)",
    ],
)
def test_postal_generator_captions_cannot_pass_correct_geography(caption):
    with pytest.raises(ValueError, match="instruction caption"):
        validate_postal_geography(
            "14 CANAL ROAD ROTTERDAM NETHERLANDS; " + caption,
            locality="ROTTERDAM",
            country="NETHERLANDS",
            country_code="NL",
            country_labelled=True,
        )
    validate_postal_geography(
        "14 COUNTRY CLUB ROAD, OLD QUARTER, ROTTERDAM NETHERLANDS",
        locality="ROTTERDAM",
        country="NETHERLANDS",
        country_code="NL",
        country_labelled=True,
    )


@pytest.mark.parametrize(
    "street",
    [
        "14 COUNTRY CLUB ROAD, UNIT 2/3",
        'BUILDING "B", P.O. BOX 42; ZONE: WEST',
        "POSTAL COMPONENT 2 BUSINESS PARK",
        "MAIL SORTING CENTRE POSTAL_COMPONENT_2 ROAD",
        "12 DOCUMENT PATCH ROAD, PARTIES BUSINESS CENTRE",
        "12 LEXICAL ROAD, REGION 3",
        "12 REGION POSTAL PARK, UNIT 3",
    ],
)
def test_postal_serialization_guard_preserves_actual_street_punctuation(street):
    validate_postal_geography(
        street + " ROTTERDAM NETHERLANDS",
        locality="ROTTERDAM",
        country="NETHERLANDS",
        country_code="NL",
        country_labelled=True,
    )


def test_initial_postal_fragments_use_one_joint_geography_contract(postal_campaign):
    fixture = postal_campaign
    request = wording_request(fixture.blueprint, fixture.scenario, "sample")
    fields = {field.key: field for field in request.fields}
    for key in ("shipper_street", "shipper_locality"):
        assert "not once per component" in fields[key].requirement
        assert "Use the supplied locality spelling once" not in fields[key].requirement
    assert "A component is not a separate complete address" in request.context
    assert (
        "POSTAL ASSEMBLY documentPatch.parties.shipper: "
        "{shipper_street} {shipper_locality}"
    ) in request.context
    assert "Locality placement is joint" in request.context
    # The old absence of an explicit country owner was not proof of a literal.
    assert "country is already supplied by a host-owned literal" not in request.context


def test_country_only_component_does_not_invent_a_locality_owner(postal_campaign):
    fixture = postal_campaign
    role = "documentPatch.parties.shipper"
    blueprint = replace(
        fixture.blueprint,
        regions=(
            SimpleNamespace(
                kind="postal",
                curated_key="shipper_locality",
                target_paths=(role + ".country",),
            ),
        ),
    )
    context = postal_assembly_context(
        blueprint, fixture.scenario, fixture.target, {"shipper_street", "shipper_locality"}
    )
    assert "COUNTRY OWNER REGIONS: {shipper_locality}" in context
    assert "Locality placement is joint" in context
    assert "LOCALITY OWNER" not in context
    assert "not required in every fragment" in context


def test_partial_postal_correction_exposes_unchanged_component_literal(postal_campaign):
    fixture = postal_campaign
    context = postal_assembly_context(
        fixture.blueprint,
        fixture.scenario,
        fixture.target,
        {"shipper_street"},
        fixed_values=fixture.receipt["values"],
    )
    assert "{shipper_street} ROTTERDAM NETHERLANDS" in context
    assert "{shipper_locality}" not in context
    assert "country is already supplied by a host-owned literal" in context


def test_single_mutable_tail_has_proven_locality_owner_and_native_alias(postal_campaign):
    fixture = postal_campaign
    role = "documentPatch.parties.shipper"
    blueprint = replace(
        fixture.blueprint,
        regions=(
            SimpleNamespace(
                kind="postal",
                curated_key="shipper_locality",
                target_paths=(role + ".country",),
            ),
        ),
    )
    fixed = {**fixture.receipt["values"], "shipper_street": "14 CANAL ROAD"}
    context = postal_assembly_context(
        blueprint,
        fixture.scenario,
        fixture.target,
        {"shipper_locality"},
        fixed_values=fixed,
    )
    assert "LOCALITY OWNER REGIONS: {shipper_locality}" in context
    request = WordingRequest(
        "sample",
        context,
        (WordingField("shipper_locality", role + " postal", "OLD CITY", "Postal tail."),),
    )
    from document_ocr.synthesis.curated_wording import native_wording_prompt

    native = native_wording_prompt([request])
    assert "LOCALITY OWNER REGIONS: {shipper_postal_component_1}" in native
    assert "COUNTRY OWNER REGIONS: {shipper_postal_component_1}" in native
    assert "shipper_locality" not in native


@pytest.mark.parametrize(
    "text",
    [
        "ZARQA this line is a name field only: PETRA HOLLOW FIBER FILTRATION CO. LTD.",
        "THIS VALUE IS THE COMPANY NAME FIELD ONLY: NEW EXPORT BV",
        "THIS TEXT IS AN IDENTITY FIELD ONLY: NEW EXPORT BV",
    ],
)
def test_company_name_rejects_explicit_generation_scope_caption(text):
    request = WordingRequest(
        "sample",
        "One fictional legal company identity.",
        (WordingField("name", "shipper name", "OLD EXPORT BV", "New identity only."),),
    )
    output = wording_output_type([request]).model_validate({"s0": {"name": text}})
    with pytest.raises(ValueError, match="caption outside its field"):
        unpack_wording(output, [request])


@pytest.mark.parametrize("text", ["NAME FIELD TECHNOLOGIES LTD.", "IDENTITY SYSTEMS BV"])
def test_company_name_scope_guard_keeps_ordinary_identity_vocabulary(text):
    request = WordingRequest(
        "sample",
        "One fictional legal company identity.",
        (WordingField("name", "shipper name", "OLD EXPORT BV", "New identity only."),),
    )
    output = wording_output_type([request]).model_validate({"s0": {"name": text}})
    assert unpack_wording(output, [request])["sample"]["name"] == text


def test_preflight_lexical_coverage_rejects_static_addresses_and_descriptions():
    from types import SimpleNamespace

    from document_ocr.synthesis.curated import Occurrence, TargetBinding, Variable
    from document_ocr.synthesis.curated_campaign import validate_lexical_owners

    path = "documentPatch.goodsItemDetails[0].description"
    contract = SimpleNamespace(
        variables=[], targets=[TargetBinding(path=path, expression="SOURCE GOODS")]
    )
    blueprint = SimpleNamespace(
        target={"documentPatch": {"goodsItemDetails": [{"description": "SOURCE GOODS"}]}},
        contract=contract,
        ownership_data={},
    )
    with pytest.raises(ValueError, match="mutable source wording owners"):
        validate_lexical_owners(blueprint)
    contract.variables = [
        Variable(
            key="product",
            kind="product",
            value="SOURCE GOODS",
            meaning="Main description",
            required_literals=[],
            occurrences=[Occurrence(text="SOURCE GOODS", occurrence=1, presentation="text")],
        )
    ]
    contract.targets = [TargetBinding(path=path, expression="{product}")]
    validate_lexical_owners(blueprint)
    blueprint.target["documentPatch"]["parties"] = {"shipper": {"addressLine": "SOURCE STREET"}}
    with pytest.raises(ValueError, match=r"shipper\.addressLine"):
        validate_lexical_owners(blueprint)


@pytest.mark.parametrize("relationship", ["on_behalf", "via", "shipping_mark"])
def test_source_only_related_company_is_generated_without_a_public_party(
    postal_campaign, relationship
):
    from document_ocr.synthesis.curated import Occurrence, Variable
    from document_ocr.synthesis.curated_campaign import related_actor_contexts

    fixture = postal_campaign
    variable = Variable(
        key="represented_company",
        kind="name",
        value="OLD ACCOUNT BV",
        meaning="Distinct represented company, not the shipper legal identity",
        required_literals=[],
        occurrences=[Occurrence(text="OLD ACCOUNT BV", occurrence=1, presentation="text")],
    )
    contract = fixture.blueprint.contract.model_copy(
        update={"variables": [*fixture.blueprint.contract.variables, variable]}
    )
    declaration = {"context_party": "documentPatch.parties.shipper", "relationship": relationship}
    blueprint = replace(
        fixture.blueprint,
        source="SOURCE\nOLD ACCOUNT BV",
        contract=contract,
        ownership_data={
            "related_actors": {variable.key: declaration},
            "identity_kinds": {variable.key: "organization"},
        },
    )
    assert related_actor_contexts(blueprint) == {variable.key: declaration}
    request = wording_request(blueprint, fixture.scenario, "related-actor-probe")
    field = next(f for f in request.fields if f.key == variable.key)
    assert "Separate fictional actor" in field.requirement
    assert "not an additional extraction-party" in field.requirement
    assert "must not be merged" in field.requirement
    assert field.role.endswith(f"related {relationship} name")
    assert field.identity_kind == "organization"
    assert "represented_company" not in str(blueprint.target)
    from document_ocr.synthesis.curated_campaign import identity_review_context

    context = identity_review_context(blueprint, {variable.key: "NEW ACCOUNT BV"})
    assert "SOURCE-CERTIFIED RELATED ACTOR OWNERSHIP" in context
    assert "source-only context; no extraction-party field" in context
    assert "NEW ACCOUNT BV" in context
    with pytest.raises(ValueError, match="generated component"):
        identity_review_context(
            replace(blueprint, ownership_data={"related_actors": {variable.key: declaration}}), {}
        )


def test_related_actor_repeated_identity_requires_every_source_occurrence(postal_campaign):
    from document_ocr.synthesis.curated import Occurrence, Variable
    from document_ocr.synthesis.curated_campaign import related_actor_contexts

    variable = Variable(
        key="account_company",
        kind="name",
        value="RELATED ACCOUNT LTD",
        meaning="One separately printed account company",
        required_literals=[],
        occurrences=[Occurrence(text="RELATED ACCOUNT LTD", occurrence=1, presentation="text")],
    )
    blueprint = replace(
        postal_campaign.blueprint,
        source="A/C RELATED ACCOUNT LTD\nEXPORTER: RELATED\nACCOUNT LTD",
        contract=postal_campaign.blueprint.contract.model_copy(
            update={"variables": [*postal_campaign.blueprint.contract.variables, variable]}
        ),
        ownership_data={
            "related_actors": {
                "account_company": {
                    "context_party": "documentPatch.parties.shipper",
                    "relationship": "account_of",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="unowned repeated source identity"):
        related_actor_contexts(blueprint)
    variable.occurrences.append(
        Occurrence(
            text="RELATED\nACCOUNT LTD",
            occurrence=1,
            presentation="text",
        )
    )
    assert related_actor_contexts(blueprint)["account_company"]["relationship"] == "account_of"


@pytest.mark.parametrize(
    "bad",
    [
        {
            "unknown": {
                "context_party": "documentPatch.parties.shipper",
                "relationship": "on_behalf",
            }
        },
        {
            "shipper_name": {
                "context_party": "documentPatch.parties.shipper",
                "relationship": "on_behalf",
            }
        },
        {
            "shipper_street": {
                "context_party": "documentPatch.parties.unknown",
                "relationship": "care_of",
            }
        },
        {
            "shipper_street": {
                "context_party": "documentPatch.parties.shipper",
                "relationship": "unexplained",
            }
        },
    ],
)
def test_related_actor_contract_rejects_unknown_or_merged_actor(postal_campaign, bad):
    from document_ocr.synthesis.curated_campaign import related_actor_contexts

    with pytest.raises(ValueError, match="related actor"):
        related_actor_contexts(
            replace(postal_campaign.blueprint, ownership_data={"related_actors": bad})
        )


@pytest.mark.parametrize("relationship", ["invoice_party", "exporter", "manufacturer"])
def test_independent_commercial_company_is_source_only(postal_campaign, relationship):
    from document_ocr.synthesis.curated import Occurrence, Variable
    from document_ocr.synthesis.curated_campaign import related_actor_contexts

    variable = Variable(
        key="secondary_company",
        kind="name",
        value="OLD EXPORT LTD",
        meaning="Independent commercial company",
        required_literals=[],
        occurrences=[Occurrence(text="OLD EXPORT LTD", occurrence=1, presentation="text")],
    )
    declaration = {
        "context_party": "documentPatch.parties.shipper",
        "relationship": relationship,
    }
    bp = replace(
        postal_campaign.blueprint,
        source="OLD EXPORT LTD",
        contract=postal_campaign.blueprint.contract.model_copy(
            update={"variables": [*postal_campaign.blueprint.contract.variables, variable]}
        ),
        ownership_data={"related_actors": {variable.key: declaration}},
    )
    assert related_actor_contexts(bp)[variable.key] == declaration
    request = wording_request(bp, postal_campaign.scenario, "secondary")
    field = next(f for f in request.fields if f.key == variable.key)
    assert relationship in field.role
    assert "not an additional extraction-party" in field.requirement
    assert variable.value not in str(bp.target)


def test_secondary_exporter_location_stays_independent_of_primary(postal_campaign):
    from document_ocr.synthesis.curated import Occurrence, Variable
    from document_ocr.synthesis.curated_campaign import related_actor_contexts

    variable = Variable(
        key="exporter_address",
        kind="postal",
        value="PORT LOUIS MAURITIUS",
        meaning="Independent exporter address",
        required_literals=[],
        occurrences=[Occurrence(text="PORT LOUIS MAURITIUS", occurrence=1, presentation="text")],
    )
    location = {
        "name": "PORT LOUIS",
        "country_code": "MU",
        "country": "MAURITIUS",
        "registry": "geonames",
        "registry_id": "934154",
    }
    declaration = {
        "context_party": "documentPatch.parties.shipper",
        "relationship": "exporter",
        "location": location,
    }
    bp = replace(
        postal_campaign.blueprint,
        source="PORT LOUIS MAURITIUS",
        contract=postal_campaign.blueprint.contract.model_copy(
            update={"variables": [*postal_campaign.blueprint.contract.variables, variable]}
        ),
        ownership_data={"related_actors": {variable.key: declaration}},
    )
    field = next(
        f
        for f in wording_request(bp, postal_campaign.scenario, "secondary").fields
        if f.key == variable.key
    )
    assert "PORT LOUIS, MAURITIUS (MU)" in field.requirement
    with pytest.raises(ValueError, match="printed source country"):
        related_actor_contexts(replace(bp, source="UNRELATED COUNTRY"))
    for bad in ({**location, "registry_id": ""}, {**location, "unsupported": "value"}):
        with pytest.raises(ValueError):
            related_actor_contexts(
                replace(
                    bp,
                    ownership_data={
                        "related_actors": {variable.key: {**declaration, "location": bad}}
                    },
                )
            )


def test_source_only_name_alias_requires_explicit_generated_identity(postal_campaign):
    from document_ocr.synthesis.curated_ownership import ownership_surfaces

    bp = replace(
        postal_campaign.blueprint,
        ownership_data={"surfaces": {"freight_alias": {"lexical_key": "shipper_name"}}},
    )
    result = ownership_surfaces(
        bp,
        bp.target,
        postal_campaign.scenario,
        lexical_values={"shipper_name": "NEW REPRESENTED COMPANY BV"},
    )
    assert result["freight_alias"] == "NEW REPRESENTED COMPANY BV"
    for invalid in (None, {}, {"shipper_name": ""}, {"shipper_name": None}):
        with pytest.raises(ValueError, match="explicit generated lexical owner"):
            ownership_surfaces(bp, bp.target, postal_campaign.scenario, lexical_values=invalid)
    for key in ("unknown", "shipper_street"):
        with pytest.raises(ValueError, match="explicit generated lexical owner"):
            ownership_surfaces(
                replace(bp, ownership_data={"surfaces": {"alias": {"lexical_key": key}}}),
                bp.target,
                postal_campaign.scenario,
                lexical_values={key: "UNSUPPORTED OWNER"},
            )


def test_regulatory_product_alias_uses_product_identity_not_complete_description(postal_campaign):
    from document_ocr.synthesis.curated_ownership import ownership_surfaces

    bp = replace(
        postal_campaign.blueprint,
        ownership_data={"surfaces": {"proper_shipping_name": {"lexical_key": "product"}}},
    )
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"][0]["description"] = (
        "NEW CHEMICAL, 1016 KG, TOTE BIN"
    )
    result = ownership_surfaces(
        bp, target, postal_campaign.scenario, lexical_values={"product": "NEW CHEMICAL"}
    )
    assert result["proper_shipping_name"] == "NEW CHEMICAL"
    for invalid in (None, {}, {"product": ""}, {"product": None}):
        with pytest.raises(ValueError, match="explicit generated lexical owner"):
            ownership_surfaces(bp, target, postal_campaign.scenario, lexical_values=invalid)
    malformed = replace(
        bp,
        ownership_data={
            "surfaces": {"proper_shipping_name": {"lexical_key": "product", "unreviewed": True}}
        },
    )
    with pytest.raises(ValueError, match="explicit generated lexical owner"):
        ownership_surfaces(
            malformed, target, postal_campaign.scenario, lexical_values={"product": "NEW CHEMICAL"}
        )
