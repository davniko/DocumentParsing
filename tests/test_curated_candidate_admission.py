"""Real campaign acceptance boundary on the six-row integer-volume failure shape."""

from copy import deepcopy
from decimal import Decimal

import pytest
from test_curated_scenarios import catalog, source_row

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_campaign import Campaign
from document_ocr.synthesis.curated_phones import PhoneRenderingPolicy
from document_ocr.synthesis.curated_physical import PhysicalSupport, prepare_physical_render
from document_ocr.synthesis.curated_scenarios import SourceCapabilities
from document_ocr.synthesis.curated_templates import compile_sampling_blueprint
from document_ocr.synthesis.generators import iso6346_check_digit


def six_row_campaign():
    source = source_row("six-row-source", quantity=168)
    source["joinedRawText"] = "\n".join(["28100KGS/14CBM"] * 6)
    p = source["target"]["documentPatch"]
    p["negotiability"] = "non_negotiable"
    g = p["goodsItemDetails"][0]
    g["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_BAG"
    g.pop("description")
    g.pop("hsCodes")
    g.pop("netWeight")
    g["grossWeight"]["value"] = 168600.0
    g["volume"]["value"] = 84.0
    p["containerInformation"] = [
        {
            "equipmentIdentifier": stem + iso6346_check_digit(stem),
            "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
            "typeCategory": "GENERAL_PURPOSE",
        }
        for i in range(6)
        if (stem := f"MSCU12345{i}")
    ]
    g["splitGoodsPlacement"] = [
        {"equipmentIdentifier": c["equipmentIdentifier"], "packageQuantity": 28}
        for c in p["containerInformation"]
    ]
    variables = []
    for index in range(6):
        for field, kind, value in (("gross_weight", "mass", "28100"), ("volume", "volume", "14")):
            variables.append(
                dict(
                    key=f"{field}_{index}",
                    kind=kind,
                    value=value,
                    meaning=f"Printed cargo container:{index}:{field}_value",
                    required_literals=[],
                    occurrences=[dict(text=value, occurrence=index + 1, presentation="number")],
                )
            )
    contract = SourceContract(
        variables=variables,
        targets=[
            dict(
                path="documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity",
                expression="168",
            )
        ],
        fixed_context="Six declared rows each print integer kilos and integer cubic metres.",
    )
    blueprint = compile_sampling_blueprint(
        source,
        dict(
            bindings=[],
            document_id=source["documentId"],
            source_sha256=digest(source["joinedRawText"].encode()),
        ),
        contract,
    )
    donors = []
    for name, mass, volume in (("tiny", 148.0, 0.35), ("large", 1000.0, 10.0)):
        donor = source_row(name, quantity=1)
        dp = donor["target"]["documentPatch"]
        dp["containerInformation"] = deepcopy(p["containerInformation"][:1])
        dg = dp["goodsItemDetails"][0]
        dg.pop("netWeight")
        dg["grossWeight"]["value"] = mass
        dg["volume"]["value"] = volume
        dg["splitGoodsPlacement"] = [
            dict(
                equipmentIdentifier=dp["containerInformation"][0]["equipmentIdentifier"],
                packageQuantity=1,
            )
        ]
        donors.append(donor)
    rows = [source, *donors]
    campaign = Campaign.__new__(Campaign)
    campaign.rows = {r["documentId"]: r for r in rows}
    campaign.catalog = catalog(rows)
    campaign.capabilities = {source["documentId"]: SourceCapabilities(family="ambient")}
    campaign.physical_support = PhysicalSupport.fit(rows)
    campaign.equipment_inventories = {}
    campaign.equipment_compatibility = {}
    campaign.phone_policy = PhoneRenderingPolicy()
    campaign._source_phone_country_codes = {}
    campaign.blueprint = lambda sid: blueprint
    campaign.vessels = ()
    return campaign, source, blueprint


def test_campaign_retries_six_positive_integer_volume_rows_before_acceptance():
    campaign, source, blueprint = six_row_campaign()
    original = deepcopy(source)
    sid = source["documentId"]
    # Fix a reproducible seed whose first unconstrained host draw is the tiny
    # measured donor:6 pallets,888kg,2.1m³. No provider or randomness is involved.
    seed = next(
        seed
        for seed in range(1, 30)
        if campaign.catalog.sample(
            source, seed=seed, variant=1, capabilities=campaign.capabilities[sid]
        ).provenance["donorDocumentId"]
        == "tiny"
    )
    first = campaign.catalog.sample(
        source, seed=seed, variant=1, capabilities=campaign.capabilities[sid]
    )
    assert first.cargo["goodsItemDetails"][0]["volume"]["value"] == 2.1
    campaign.config = {"seed": seed}
    plan = campaign.plan(sid, 1)
    assert plan[2] == "syn_full_v7_" + digest([sid, seed, 1])[:24]
    assert plan == campaign.plan(sid, 1)
    assert plan[1].provenance["donorDocumentId"] == "large"
    assert plan[1].provenance["candidateAttempts"] > 1
    assert (
        plan[1].provenance["rejectedCandidates"][
            "positive physical row allocation cannot exactly represent the total"
        ]
        >= 1
    )
    physical = prepare_physical_render(
        blueprint, plan[1], plan[3], support=campaign.physical_support
    )
    for measure, values in physical.receipt["rows"].items():
        amounts = list(map(Decimal, values))
        assert len(amounts) == 6 and all(v > 0 for v in amounts)
        assert sum(amounts) == Decimal(physical.receipt["measures"][measure]["total"])
    assert source == original


def test_campaign_does_not_retry_invalid_source_measure_ownership(monkeypatch):
    campaign, source, blueprint = six_row_campaign()
    campaign.config = {"seed": 42}
    # This is a source-contract defect, not a shipment draw incompatibility.
    blueprint.contract.variables[0].meaning = "undeclared numeric owner"
    calls = []
    original = campaign.catalog._cargo

    def observe(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign.catalog, "_cargo", observe)
    with pytest.raises(ValueError, match="no declared measurement role"):
        campaign.plan(source["documentId"], 1)
    assert calls == [1]
