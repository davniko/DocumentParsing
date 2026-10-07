"""Scenario behavior: coherent draws, exact allocation and honest source visibility."""

from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.country_registry import CountryEntry, CountryRegistry
from document_ocr.synthesis.curated_scenarios import (
    ScenarioCatalog,
    ScenarioLocation,
    ScenarioSamplingConfig,
    SourceCapabilities,
    _hazard_identity_index,
)


def source_row(identifier="source", *, code="520511", quantity=100):
    return {
        "documentId": identifier,
        "joinedRawText": "Original source text is never mutated by scenario sampling.",
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "route": {
                    "portOfLoading": {"name": "SOURCE PORT", "country": "CHINA"},
                    "portOfDischarge": {"name": "SOKHNA", "country": "EGYPT"},
                },
                "parties": {
                    "shipper": {
                        "name": "SENDER",
                        "addressLine": "A ROAD CHINA",
                        "country": "CHINA",
                    },
                    "consignee": {
                        "name": "RECIPIENT",
                        "addressLine": "B ROAD EGYPT",
                        "country": "EGYPT",
                    },
                },
                "goodsItemDetails": [
                    {
                        "description": "COTTON YARN",
                        "hsCodes": [code],
                        "grossWeight": {"value": 5000.0, "unit": "kilogram"},
                        "netWeight": {"value": 4500.0, "unit": "kilogram"},
                        "volume": {"value": 10.0, "unit": "cubic_metre"},
                        "numberAndTypeOfPackages": [
                            {"packageQuantity": quantity, "typeCategory": "PACKAGE_BALE"}
                        ],
                        "splitGoodsPlacement": [
                            {
                                "equipmentIdentifier": "MSCU1234567",
                                "packageQuantity": quantity * 2 // 5,
                            },
                            {
                                "equipmentIdentifier": "MSCU7654321",
                                "packageQuantity": quantity * 3 // 5,
                            },
                        ],
                    }
                ],
                "containerInformation": [
                    {
                        "equipmentIdentifier": number,
                        "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                        "typeCategory": "GENERAL_PURPOSE",
                    }
                    for number in ("MSCU1234567", "MSCU7654321")
                ],
            },
        },
    }


def config(**overrides):
    pin = {"path": "fixture", "sha256": "0" * 64}
    data = {
        field: pin
        for field in (
            "countries",
            "locations",
            "world_ports",
            "localities",
            "hs_manifest",
            "hs_metadata",
            "hs_report",
            "commodity_phrases",
            "hmt",
            "ecics",
        )
    }
    data["transport_capacity"] = {
        "policy": "source_type_aware_maersk_upper_bounds_v1",
        "published_reference_margin_fraction": 0.0,
        "twenty_standard_payload_kg": 28300.0,
        "twenty_standard_volume_m3": 33.2,
        "forty_standard_payload_kg": 28870.0,
        "forty_standard_volume_m3": 67.7,
        "forty_high_cube_payload_kg": 28690.0,
        "forty_high_cube_volume_m3": 76.4,
        "forty_five_high_cube_payload_kg": 27650.0,
        "forty_five_high_cube_volume_m3": 86.0,
        "out_of_gauge_payload_kg": 47300.0,
        "unclassified_payload_kg": 47300.0,
        "unclassified_volume_m3": 86.0,
    }
    data.update(overrides)
    return ScenarioSamplingConfig.model_validate(data)


def catalog(rows, *, validation=frozenset(), phrases=None, **config_overrides):
    countries = CountryRegistry(
        entries=[
            CountryEntry(alpha2=code, alpha3=alpha3, numeric=numeric, name=name)
            for code, alpha3, numeric, name in (
                ("CN", "CHN", "156", "China"),
                ("EG", "EGY", "818", "Egypt"),
                ("DE", "DEU", "276", "Germany"),
            )
        ],
        observed_aliases={},
        iso_sha256="0" * 64,
        observed_aliases_sha256="0" * 64,
    )
    ports, localities = {}, {}
    for code in countries.country_codes:
        country = countries.entry(code).name.upper()
        ports[code] = (
            ScenarioLocation(
                name=f"PORT {code}",
                country_code=code,
                country=country,
                registry="unlocode_wpi",
                registry_id=code + "AAA",
            ),
        )
        localities[code] = (
            ScenarioLocation(
                name=f"CITY {code}",
                country_code=code,
                country=country,
                registry="geonames",
                registry_id=code,
            ),
        )
    phrases = phrases or {
        "520511": "Cotton yarn",
        "520512": "Finer cotton yarn",
        "520513": "Other cotton yarn",
    }
    hs = SimpleNamespace(
        receipt=SimpleNamespace(snapshot_date=date(2026, 8, 31)),
        require_global=lambda code, on_date: SimpleNamespace(
            code=code,
            chapter_code=code[:2],
            heading_code=code[:4],
            description=phrases[code],
            description_authority="test hierarchy",
            heading_description="Cotton yarn",
            chapter_description="Cotton",
        ),
        global_description_path=lambda code: ("Cotton yarn", phrases[code]),
        uk_candidates=lambda code, on_date: (),
    )
    return ScenarioCatalog(
        config=config(**config_overrides),
        countries=countries,
        hs=hs,
        phrases=phrases,
        ports=ports,
        localities=localities,
        dg=SimpleNamespace(hmt_records=(), ecics_links=()),
        train_rows=rows,
        validation_ids=validation,
    )


def test_sampling_changes_identity_geography_and_exact_joint_quantities_without_mutation():
    source = source_row()
    before = deepcopy(source)
    support = catalog([source])
    cap = SourceCapabilities(family="ambient", destination_country="EG")
    results = [support.sample(source, seed=42, variant=i, capabilities=cap) for i in range(1, 13)]
    assert source == before
    assert len({r.origin.country_code for r in results}) == 2
    assert all(r.destination.country_code == "EG" for r in results)
    assert all(r.goods_identities[0].hs6 != "520511" for r in results)
    assert (
        len(
            {
                r.cargo["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["packageQuantity"]
                for r in results
            }
        )
        > 1
    )
    for result in results:
        goods = result.cargo["goodsItemDetails"][0]
        quantity = goods["numberAndTypeOfPackages"][0]["packageQuantity"]
        assert sum(p["packageQuantity"] for p in goods["splitGoodsPlacement"]) == quantity
        assert all(p["packageQuantity"] > 0 for p in goods["splitGoodsPlacement"])
        assert abs(goods["splitGoodsPlacement"][0]["packageQuantity"] - quantity * 0.4) < 2
        assert goods["grossWeight"]["value"] == quantity * 50
        assert goods["netWeight"]["value"] == quantity * 45
        assert set(goods) == set(before["target"]["documentPatch"]["goodsItemDetails"][0])
        assert result.provenance["candidateAttempts"] == 1
        assert sum(r["shareNumerator"] for r in result.provenance["physicalRows"]) == quantity
    assert results[0] == support.sample(source, seed=42, variant=1, capabilities=cap)


def test_contact_domain_retries_only_unsupported_draws_and_records_exclusions():
    source = source_row()
    support = catalog([source])
    cap = SourceCapabilities(family="ambient")
    original = support.sample(source, seed=8, variant=1, capabilities=cap)
    support.contact_excluded_countries[original.destination.country_code] = (
        "no test mobile metadata"
    )
    repaired = support.sample(source, seed=8, variant=1, capabilities=cap)
    assert repaired.origin == original.origin
    assert repaired.destination.country_code != original.destination.country_code
    audit = repaired.provenance["geographyEligibility"]
    assert audit["rejectedDraws"][0]["country"] == original.destination.country_code
    assert (
        audit["excludedCountries"][original.destination.country_code] == "no test mobile metadata"
    )
    with pytest.raises(ValueError, match="pinned geography lacks certified contact"):
        support.sample(
            source,
            seed=8,
            variant=1,
            capabilities=cap.model_copy(
                update={"destination_country": original.destination.country_code}
            ),
        )


def test_freight_location_uses_explicit_arrangement_or_adjudicated_side():
    source = source_row()
    source["target"]["documentPatch"]["goodsItemDetails"][0]["origin"] = {"identifier": "CN"}
    source["target"]["documentPatch"]["freight"] = {
        "paymentArrangement": "collect",
        "paymentPlace": {"name": "OLD", "country": "OLD"},
    }
    support = catalog([source])
    cap = SourceCapabilities(family="ambient")
    result = support.sample(source, seed=7, variant=1, capabilities=cap)
    assert result.cargo["goodsItemDetails"][0]["origin"] == {
        "identifier": result.origin.country_code
    }
    assert (
        result.replacements["documentPatch.freight.paymentPlace.country"]
        == result.destination.country
    )
    source["target"]["documentPatch"]["freight"].pop("paymentArrangement")
    support = catalog([source])
    with pytest.raises(ValueError, match="payment place requires an explicit"):
        support.sample(source, seed=7, variant=1, capabilities=cap)
    result = support.sample(
        source,
        seed=7,
        variant=1,
        capabilities=cap.model_copy(
            update={"location_sides": {"documentPatch.freight.paymentPlace": "origin"}}
        ),
    )
    assert (
        result.replacements["documentPatch.freight.paymentPlace.country"] == result.origin.country
    )


def test_ventilation_slots_require_joint_observed_cold_chain_bundle():
    source = source_row(code="070320")
    source["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "FRESH GARLIC"
    source["target"]["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = [
        "VENTILATION 10 CBM PER HOUR"
    ]
    for c in source["target"]["documentPatch"]["containerInformation"]:
        c.update(
            typeCategory="REFRIGERATED", temperatureSetpoint={"value": -3.0, "unit": "celsius"}
        )
    donor = deepcopy(source)
    donor["documentId"] = "donor"
    donor["target"]["documentPatch"]["goodsItemDetails"][0].pop("handlingInstructions")
    for c in donor["target"]["documentPatch"]["containerInformation"]:
        c["temperatureSetpoint"]["value"] = 5.0
    support = catalog([source, donor], phrases={"070320": "Fresh garlic"})
    results = [
        support.sample(source, seed=6, variant=i, capabilities=SourceCapabilities(family="chilled"))
        for i in range(1, 6)
    ]
    assert all(r.provenance["donorDocumentId"] == "source" for r in results)
    assert all(r.provenance["ventilationCbmPerHour"] == "10" for r in results)
    for result in results:
        evidence = result.provenance["thermalCommodityContext"]
        assert evidence["observed_description"] == "FRESH GARLIC"
        assert evidence["observed_hs6"] == ["070320"]
        assert evidence["temperature_celsius"] == -3.0
        assert evidence["ventilation_cbm_per_hour"] == "10"
        assert evidence["donor_document_id"] == "source"
        assert len(evidence["donor_target_sha256"]) == 64
        assert evidence["preservation_scope"] == (
            "commodity identity, physical form and processing state"
        )


def test_enumerated_vehicle_profile_does_not_turn_spare_part_packages_into_unit_mass():
    source = source_row(code="870194", quantity=3)
    goods = source["target"]["documentPatch"]["goodsItemDetails"][0]
    goods.update(
        description="USED AGRICULTURAL TRACTORS", grossWeight={"value": 16500.0, "unit": "kilogram"}
    )
    goods["splitGoodsPlacement"][1]["packageQuantity"] = 2
    donor = deepcopy(source)
    donor["documentId"] = "parts"
    parts = donor["target"]["documentPatch"]["goodsItemDetails"][0]
    parts.update(
        description="USED SPARE PARTS FOR TRUCKS",
        hsCodes=["870121"],
        grossWeight={"value": 21400.0, "unit": "kilogram"},
    )
    parts["numberAndTypeOfPackages"][0]["packageQuantity"] = 19
    support = catalog(
        [source, donor],
        phrases={"870194": "Agricultural tractors 75 to 130 kW", "870121": "Road tractors"},
    )
    cap = SourceCapabilities(
        family="vehicle",
        allowed_hs_headings=("8701",),
        allowed_hs_codes=("870194",),
        fixed_package_quantity=3,
        physical_profile="source_whole_units",
    )
    result = support.sample(source, seed=13, variant=1, capabilities=cap)
    assert result.provenance["donorDocumentId"] == "source"
    assert result.cargo["goodsItemDetails"][0]["grossWeight"]["value"] == 16500
    assert result.goods_identities[0].hs6 == "870194"


def test_validation_cannot_fit_or_be_used_as_source_and_drift_is_rejected():
    source = source_row()
    with pytest.raises(ValueError, match="train-only"):
        catalog([source], validation=frozenset({"source"}))
    support = catalog([source])
    changed = deepcopy(source)
    changed["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = "CHANGED"
    with pytest.raises(ValueError, match="exact current train"):
        support.sample(
            changed, seed=1, variant=1, capabilities=SourceCapabilities(family="ambient")
        )


def test_multiple_hs_identities_are_unique_and_missing_fields_stay_missing():
    source = source_row()
    goods = source["target"]["documentPatch"]["goodsItemDetails"][0]
    del goods["hsCodes"]
    donor = source_row("donor")
    support = catalog([source, donor])
    result = support.sample(
        source,
        seed=9,
        variant=1,
        capabilities=SourceCapabilities(family="ambient", identity_count=2),
    )
    assert len({i.hs6 for i in result.goods_identities}) == 2
    assert "hsCodes" not in result.cargo["goodsItemDetails"][0]
    assert "documentPatch.goodsItemDetails[0].hsCodes" not in result.replacements


def test_bad_allocations_or_impossible_capacity_fail_closed_with_reasons():
    source = source_row()
    source["target"]["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][0][
        "packageQuantity"
    ] = 41
    support = catalog([source], maximum_candidates=2)
    with pytest.raises(ValueError, match="not a complete exact sum"):
        support.sample(source, seed=1, variant=1, capabilities=SourceCapabilities(family="ambient"))
    source = source_row()
    source["target"]["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] = 1000000
    support = catalog([source], maximum_candidates=2)
    with pytest.raises(ValueError, match="payload capacity"):
        support.sample(source, seed=1, variant=1, capabilities=SourceCapabilities(family="ambient"))


def test_membership_only_does_not_invent_allocation_counts_and_repeated_parties_share_geography():
    source = source_row()
    patch = source["target"]["documentPatch"]
    for placement in patch["goodsItemDetails"][0]["splitGoodsPlacement"]:
        del placement["packageQuantity"]
    patch["parties"]["notifyParties"] = [deepcopy(patch["parties"]["consignee"])]
    result = catalog([source]).sample(
        source, seed=3, variant=1, capabilities=SourceCapabilities(family="ambient")
    )
    assert all(
        "packageQuantity" not in p
        for p in result.cargo["goodsItemDetails"][0]["splitGoodsPlacement"]
    )
    assert (
        result.party_localities["documentPatch.parties.notifyParties[0]"]
        == result.party_localities["documentPatch.parties.consignee"]
    )


def test_explicit_capability_restrictions_are_not_silently_widened():
    source = source_row()
    support = catalog([source])
    with pytest.raises(ValueError, match="no complete train-only"):
        support.sample(
            source,
            seed=2,
            variant=1,
            capabilities=SourceCapabilities(
                family="ambient", allowed_package_categories=("PACKAGE_DRUM",)
            ),
        )
    with pytest.raises(ValueError, match="explicit source-owned"):
        SourceCapabilities(family="vehicle")
    with pytest.raises(ValueError, match="identical endpoint countries"):
        support.sample(
            source,
            seed=2,
            variant=1,
            capabilities=SourceCapabilities(
                family="ambient", origin_country="EG", destination_country="EG"
            ),
        )


def test_frozen_registry_identity_keeps_observed_temperature_and_reefer_pair():
    source = source_row(code="020622")
    patch = source["target"]["documentPatch"]
    patch["goodsItemDetails"][0]["description"] = "FROZEN BOVINE LIVERS"
    patch["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_CARTON"
    for container in patch["containerInformation"]:
        container["typeCategory"] = "REFRIGERATED"
        container["temperatureSetpoint"] = {"value": -22.0, "unit": "celsius"}
    support = catalog(
        [source],
        phrases={
            "020622": "Frozen bovine livers",
            "020621": "Frozen bovine tongues",
            "020610": "Fresh or chilled bovine offal",
        },
    )
    for variant in range(1, 5):
        result = support.sample(
            source, seed=3, variant=variant, capabilities=SourceCapabilities(family="frozen")
        )
        assert result.goods_identities[0].hs6 == "020621"
        evidence = result.provenance["thermalCommodityContext"]
        assert evidence["observed_description"] == "FROZEN BOVINE LIVERS"
        assert evidence["temperature_celsius"] == -22.0
        assert evidence["ventilation_cbm_per_hour"] is None
        assert all(
            c["typeCategory"] == "REFRIGERATED" and c["temperatureSetpoint"]["value"] == -22
            for c in result.cargo["containerInformation"]
        )


def test_quantity_allocation_does_not_inherit_coprime_source_count_lattice():
    source = source_row(quantity=1111)
    goods = source["target"]["documentPatch"]["goodsItemDetails"][0]
    goods["splitGoodsPlacement"][0]["packageQuantity"] = 444
    goods["splitGoodsPlacement"][1]["packageQuantity"] = 667
    result = catalog([source]).sample(
        source,
        seed=42,
        variant=1,
        capabilities=SourceCapabilities(family="ambient", fixed_package_quantity=100),
    )
    allocated = result.cargo["goodsItemDetails"][0]["splitGoodsPlacement"]
    assert sum(value["packageQuantity"] for value in allocated) == 100
    assert all(value["packageQuantity"] > 0 for value in allocated)


def test_reviewed_hs_observation_override_is_pinned_and_does_not_relabel_source():
    from document_ocr.hashing import canonical_json_bytes, sha256_bytes

    source = source_row(code="520500")
    original = deepcopy(source)
    override = {
        "source_target_sha256": sha256_bytes(canonical_json_bytes(source["target"])),
        "from_hs6": ["520500"],
        "to_hs6": ["520511"],
        "source_text_literals": ["Original source text"],
        "rationale": "Reviewed synthesis identity, extraction still preserves source code.",
    }
    support = catalog([source], observed_hs_overrides={"source": override})
    result = support.sample(
        source,
        seed=1,
        variant=1,
        capabilities=SourceCapabilities(family="ambient", explore_within_heading=False),
    )
    assert result.goods_identities[0].hs6 == "520511"
    assert result.provenance["observedHSOverride"]["to_hs6"] == ["520511"]
    assert source == original == support.train["source"]
    assert support.observations[0].hs6 == ("520511",)
    for key, value in (
        ("source_target_sha256", "0" * 64),
        ("from_hs6", ["520599"]),
        ("source_text_literals", ["not actually printed"]),
        ("to_hs6", ["999999"]),
    ):
        with pytest.raises(ValueError, match="disagrees with pinned source"):
            catalog([source], observed_hs_overrides={"source": {**override, key: value}})


def test_exact_hazard_identity_gate_includes_conditional_names_not_whole_hs_headings():
    record = SimpleNamespace(
        un_number="2214",
        maritime_eligible=True,
        proper_shipping_name="Phthalic anhydride with more than .05 percent maleic anhydride",
    )
    link = SimpleNamespace(
        un_number="2214",
        name="phthalic anhydride",
        iupac_description=None,
        synonyms=(),
    )
    index = _hazard_identity_index(SimpleNamespace(hmt_records=(record,), ecics_links=(link,)))
    assert index["phthalic anhydride"]["unNumbers"] == ["2214"]
    assert "other carboxylic anhydrides" not in index
    source = source_row(code="291735")
    support = catalog([source], phrases={"291735": "PHTHALIC ANHYDRIDE"}, maximum_candidates=2)
    support.hazard_identities = index
    with pytest.raises(ValueError, match="grade-conditional chemical"):
        support.sample(
            source,
            seed=1,
            variant=1,
            capabilities=SourceCapabilities(family="ambient", explore_within_heading=False),
        )


def test_classification_context_carries_scope_siblings_and_immediate_national_examples():
    source = source_row()
    support = catalog([source])

    def node(code, description):
        return SimpleNamespace(code=code, suffix="80", description=description)

    support.hs.uk_candidates = lambda code, on_date: (
        SimpleNamespace(
            description_path=(
                node("5205000000", "Cotton yarn"),
                node(code + "0000", "Selected HS6"),
                node(code + "1000", "Concrete in-scope option"),
                node(code + "1010", "National deeper limit, not immediate example"),
            )
        ),
        SimpleNamespace(
            description_path=(
                node("5205000000", "Cotton yarn"),
                node(code + "0000", "Selected HS6"),
                node(code + "1000", "Concrete in-scope option"),
                node(code + "1090", "Other national leaf"),
            )
        ),
    )
    context = support._classification("520511")
    assert context.description_path == ("Cotton yarn", "Cotton yarn")
    assert {a.hs6 for a in context.same_heading_alternatives} == {"520512", "520513"}
    assert all(len(a.description_path) == 1 for a in context.same_heading_alternatives)
    assert [(e.code, e.description) for e in context.immediate_national_children] == [
        ("5205111000", "Concrete in-scope option")
    ]
    assert support._classification("520511") is context


def test_reviewed_whole_vehicle_donor_transfers_joint_profile_and_rejects_drift():
    source = source_row(quantity=1)
    patch = source["target"]["documentPatch"]
    patch["containerInformation"] = patch["containerInformation"][:1]
    goods = patch["goodsItemDetails"][0]
    goods.update(
        description="NEW UNCLASSIFIED PASSENGER VEHICLE",
        grossWeight={"value": 2708.0, "unit": "kilogram"},
        volume={"value": 18.923, "unit": "cubic_metre"},
        dangerousGoods=[
            {
                "unNumber": "3166",
                "hazardCategory": "MISCELLANEOUS_DANGEROUS_SUBSTANCES_AND_ARTICLES",
            }
        ],
        splitGoodsPlacement=[{"equipmentIdentifier": "MSCU1234567", "packageQuantity": 1}],
    )
    goods.pop("hsCodes")
    goods.pop("netWeight")
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_UNPACKED_OR_UNPACKAGED"
    donor = deepcopy(source)
    donor["documentId"] = "whole_car"
    donor["joinedRawText"] = "1 UNIT PETROL PASSENGER VEHICLE HS870323 1996KG 15.582CBM"
    donor_goods = donor["target"]["documentPatch"]["goodsItemDetails"][0]
    donor_goods.update(
        description="NEW PETROL PASSENGER VEHICLE",
        hsCodes=["870323"],
        grossWeight={"value": 1996.0, "unit": "kilogram"},
        volume={"value": 15.582, "unit": "cubic_metre"},
    )
    donor_goods.pop("dangerousGoods")
    donor_goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_PACKAGE"
    donor["target"]["documentPatch"]["containerInformation"][0]["sizeCategory"] = (
        "FORTY_FOOT_STANDARD_HEIGHT"
    )
    support = catalog(
        [source, donor],
        phrases={"870323": "Passenger motor cars, spark-ignition, engine 1,500-3,000 cc"},
    )
    support.dg.hmt_records = (
        SimpleNamespace(
            un_number="3166",
            maritime_eligible=True,
            technical_name_required=False,
            proper_shipping_name="Vehicle, flammable liquid powered",
            record_id="un3166",
            hazard_category="MISCELLANEOUS_DANGEROUS_SUBSTANCES_AND_ARTICLES",
            packing_group_category=None,
            packing_group_code=None,
            exact_hazard_class="9",
            exact_subsidiary_hazards=(),
            subsidiary_hazard_categories=(),
        ),
    )
    pin = {
        "target_sha256": sha256_bytes(canonical_json_bytes(donor["target"])),
        "whole_unit_count": 1,
        "source_text_literals": [donor["joinedRawText"]],
        "rationale": "Reviewed complete passenger vehicle, not a package of spare parts.",
    }
    policy = {
        "family": "dg_vehicle",
        "allowed_hs_headings": ["8703"],
        "allowed_hs_codes": ["870323"],
        "fixed_package_quantity": 1,
        "explore_within_heading": False,
        "physical_profile": "reviewed_whole_units",
        "reviewed_whole_unit_donors": {"whole_car": pin},
    }
    original = deepcopy(source)
    result = support.sample(source, seed=42, variant=1, capabilities=SourceCapabilities(**policy))
    actual = result.cargo["goodsItemDetails"][0]
    assert actual["grossWeight"]["value"] == 1996
    assert actual["volume"]["value"] == 15.582
    assert actual["numberAndTypeOfPackages"][0]["typeCategory"] == "PACKAGE_UNPACKED_OR_UNPACKAGED"
    assert actual["dangerousGoods"][0]["unNumber"] == "3166"
    assert "hsCodes" not in actual
    assert result.goods_identities[0].hs6 == "870323"
    assert result.cargo["containerInformation"][0]["sizeCategory"] == "FORTY_FOOT_STANDARD_HEIGHT"
    assert result.provenance["donorDocumentId"] == "whole_car"
    assert source == original
    for key, value in (
        ("target_sha256", "0" * 64),
        ("whole_unit_count", 2),
        ("source_text_literals", ["not printed"]),
    ):
        changed = {**policy, "reviewed_whole_unit_donors": {"whole_car": {**pin, key: value}}}
        with pytest.raises(ValueError, match="pinned training evidence"):
            support.sample(source, seed=42, variant=1, capabilities=SourceCapabilities(**changed))
    with pytest.raises(ValueError, match="exact-HS"):
        SourceCapabilities(**{**policy, "explore_within_heading": True})
    support.phrases["870323"] = "Hybrid electric passenger car with petrol engine"
    with pytest.raises(ValueError, match="no physically compatible scenario"):
        support.sample(source, seed=42, variant=1, capabilities=SourceCapabilities(**policy))
    support.train.pop("whole_car")
    with pytest.raises(ValueError, match="pinned training evidence"):
        support.sample(source, seed=42, variant=1, capabilities=SourceCapabilities(**policy))
