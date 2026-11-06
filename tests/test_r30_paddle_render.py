import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from render_r30_paddle_pilot import load_current

from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion
from document_ocr.synthesis.curated_campaign import shifted_public_date
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.paddle_render import (
    apply_reviewed_country_coreferences,
    coalesced_edits,
    extra_targets,
    extra_value,
    literal_owner_value,
    order_containers_by_printed_occurrence,
    present_source_edits,
    project_printed_postal_country,
    render_one,
    rendered_mass_sum,
    rendered_owner_projection,
    sampled_equipment,
    spans_for,
)

ROOT = Path(__file__).resolve().parents[1]


def test_postal_country_requires_owned_literal_evidence():
    from document_ocr.hashing import sha256_file

    manifest = "artifacts/paddle-migration/r30-independent-audit-20261103/country-input-registry/production-manifest.json"
    action = {"target": "/parties/shipper/country", "addressTarget": "/parties/shipper/addressLine",
              "codePath": "/origin/country_code", "manifestPath": manifest,
              "sha256": sha256_file(ROOT / manifest)}
    for address, code, expected in (
        ("SAINT CROIX, USVI 00802", "VI", "UNITED STATES VIRGIN ISLANDS"),
        ("MALAYER 65819-36543", "IR", None),
        ("COSCOMATEPEC DE BRAVO, VERACRUZ 94600", "MX", None),
        ("NEW YORK, NY", "US", None),
    ):
        target = {"parties": {"shipper": {"addressLine": address}}}
        candidate = {"scenario": {"origin": {"country_code": code}}}
        project_printed_postal_country(target, candidate, action, ROOT)
        assert target["parties"]["shipper"].get("country") == expected
    target = {"parties": {"shipper": {"addressLine": "SAINT CROIX, USVI 00802", "country": "UNITED STATES"}}}
    with pytest.raises(ValueError, match="conflicts"):
        project_printed_postal_country(target, {"scenario": {"origin": {"country_code": "VI"}}}, action, ROOT)


def test_inventory_order_preserves_objects_and_allocation_identity():
    a = {"equipmentIdentifier": "ABCU1234567", "sealNumbers": ["s1"]}
    b = {"equipmentIdentifier": "XYZU7654321", "sealNumbers": ["s2"]}
    target = {"containerInformation": [a, b], "goodsItemDetails": [
        {"splitGoodsPlacement": [{"equipmentIdentifier": "ABCU1234567", "packageQuantity": 3}]}]}
    pages = (PaddlePage(0, 100, 100, (
        PaddleRegion(0, "XYZU 7654321 / SEAL s2", (1, 1, 90, 10)),
        PaddleRegion(1, "ABCU-1234567 / SEAL s1", (1, 20, 90, 30)),
    )),)
    order_containers_by_printed_occurrence(target, pages)
    assert target["containerInformation"] == [b, a]
    assert target["containerInformation"][0] is b
    assert target["goodsItemDetails"][0]["splitGoodsPlacement"][0]["equipmentIdentifier"] == "ABCU1234567"
    order_containers_by_printed_occurrence(target, pages)
    assert target["containerInformation"] == [b, a]
    target["containerInformation"] = [a, a]
    with pytest.raises(ValueError, match="unique, printed"):
        order_containers_by_printed_occurrence(target, pages)
    target["containerInformation"] = [{"equipmentIdentifier": "ABCU0000000"}]
    with pytest.raises(ValueError, match="unique, printed"):
        order_containers_by_printed_occurrence(target, pages)


def test_country_coreference_requires_retained_same_place_and_country_evidence():
    target = {"route": {"portOfLoading": {"name": "Mundra", "country": "INDIA"}},
              "placeOfIssue": {"name": " MUNDRA "}}
    actions = [{"from": "/route/portOfLoading", "to": "/placeOfIssue"}]
    apply_reviewed_country_coreferences(target, actions)
    assert target["placeOfIssue"]["country"] == "INDIA"
    target["placeOfIssue"] = {"name": "MUMBAI"}
    apply_reviewed_country_coreferences(target, actions)
    assert target["placeOfIssue"] == {"name": "MUMBAI"}
    target["route"]["portOfLoading"] = {"name": "SINGAPORE"}
    target["placeOfIssue"] = {"name": "SINGAPORE"}
    apply_reviewed_country_coreferences(target, actions)
    assert "country" not in target["placeOfIssue"]
    target["route"]["portOfLoading"]["country"] = "SINGAPORE"
    target["placeOfIssue"]["country"] = "MALAYSIA"
    with pytest.raises(ValueError, match="conflicts"):
        apply_reviewed_country_coreferences(target, actions)
    del target["placeOfIssue"]
    with pytest.raises(KeyError):
        apply_reviewed_country_coreferences(target, actions)
    with pytest.raises(ValueError, match="distinct"):
        apply_reviewed_country_coreferences(target, [{"from": "/route", "to": "/route"}])


def test_private_receipt_measure_is_explicit_not_a_missing_fact_fallback():
    recipe = {"physical": {
        "field": "grossWeight", "authority": "receipt_measure",
        "unit": "kilogram", "grouped": False, "minPlaces": 0,
        "prefix": "", "suffix": "",
    }}
    candidate = {"physical": {"measures": {"grossWeight": {
        "total": "10200", "beforeRounding": "10199.7", "unit": "kilogram",
    }}}}
    args = (candidate, {}, "retained", None)
    assert extra_value(recipe, *args, project_root=ROOT) == (
        "10200", {"value": 10200.0, "unit": "kilogram"},
    )
    del recipe["physical"]["authority"]
    with pytest.raises(KeyError):
        extra_value(recipe, *args, project_root=ROOT)
    recipe["physical"]["authority"] = "unknown"
    with pytest.raises(ValueError, match="authority"):
        extra_value(recipe, *args, project_root=ROOT)
    recipe["physical"]["authority"] = "receipt_measure"
    recipe["physical"]["sourceEquipmentIdentifier"] = "ABCU1234567"
    with pytest.raises(ValueError, match="shipment-total"):
        extra_value(recipe, *args, project_root=ROOT)
    del recipe["physical"]["sourceEquipmentIdentifier"]
    for value in ("NaN", "Infinity", "0", "-1"):
        candidate["physical"]["measures"]["grossWeight"]["total"] = value
        with pytest.raises(ValueError, match="finite and positive"):
            extra_value(recipe, *args, project_root=ROOT)
    candidate["physical"]["measures"]["grossWeight"]["total"] = "2"
    candidate["physical"]["measures"]["grossWeight"]["unit"] = "pound"
    with pytest.raises(ValueError, match="incompatible"):
        extra_value(recipe, *args, project_root=ROOT)


def test_explicit_native_display_precision_does_not_change_private_measure():
    recipe = {"physical": {
        "field": "grossWeight", "authority": "receipt_measure",
        "unit": "kilogram", "grouped": False, "minPlaces": 3,
        "displayPlaces": 3, "prefix": "", "suffix": "",
    }}
    exact = "25037.77777777777777777777778"
    candidate = {"physical": {"measures": {"grossWeight": {
        "total": exact, "unit": "kilogram",
    }}}}
    assert extra_value(recipe, candidate, {}, "retained", None, project_root=ROOT) == (
        "25037.778", {"value": 25037.778, "unit": "kilogram"},
    )
    assert candidate["physical"]["measures"]["grossWeight"]["total"] == exact
    candidate["physical"]["measures"]["grossWeight"]["total"] = "0.0001"
    with pytest.raises(ValueError, match="erases"):
        extra_value(recipe, candidate, {}, "retained", None, project_root=ROOT)


def test_native_display_precision_cannot_change_retained_public_measure():
    recipe = {"physical": {
        "field": "volume", "goods": 0, "authority": "public_target",
        "unit": "cubic_metre", "grouped": False, "minPlaces": 3,
        "displayPlaces": 3, "prefix": "", "suffix": "",
    }}
    current = {"target": {"goodsItemDetails": [{"volume": {
        "value": 1.2345, "unit": "cubic_metre",
    }}]}}
    with pytest.raises(ValueError, match="contradicts"):
        extra_value(recipe, {}, current, "retained", None, project_root=ROOT)


def test_native_dg_facts_require_matching_sampled_identity_and_explicit_properties():
    candidate = {
        "scenario": {
            "provenance": {
                "dgPrintedFacts": {"unNumber": "1395", "class": "4.3", "packingGroup": "III"}
            }
        }
    }
    current = {"target": {"goodsItemDetails": [{"dangerousGoods": [{"unNumber": "1395"}]}]}}
    assert extra_value(
        {"dgFact": {"field": "class"}}, candidate, current, "a", None, project_root=ROOT
    ) == (
        "4.3",
        "FLAMMABLE_SOLIDS",
    )
    assert extra_value(
        {"dgFact": {"field": "packingGroup"}}, candidate, current, "a", None, project_root=ROOT
    ) == (
        "III",
        "LOW_DANGER",
    )
    candidate["scenario"]["provenance"]["dgPrintedFacts"]["packingGroup"] = None
    with pytest.raises(ValueError, match="absent"):
        extra_value(
            {"dgFact": {"field": "packingGroup"}}, candidate, current, "a", None, project_root=ROOT
        )
    optional = {
        "dgFact": {"field": "packingGroup", "omitAbsent": True},
        "target": "/goodsItemDetails/0/dangerousGoods/0/packingGroupCategory",
    }
    assert extra_value(optional, candidate, current, "a", None, project_root=ROOT) == ("", None)
    assert extra_targets(optional, None) == {}
    public = current["target"]["goodsItemDetails"][0]["dangerousGoods"][0]
    public["packingGroupCategory"] = "LOW_DANGER"
    with pytest.raises(ValueError, match="conflicts"):
        extra_value(optional, candidate, current, "a", None, project_root=ROOT)
    del public["packingGroupCategory"]
    candidate["scenario"]["provenance"]["dgPrintedFacts"]["unNumber"] = "3082"
    with pytest.raises(ValueError, match="matching"):
        extra_value(
            {"dgFact": {"field": "class"}}, candidate, current, "a", None, project_root=ROOT
        )


def test_new_native_container_is_deterministic_checksummed_and_not_copied():
    from document_ocr.synthesis.generators import validate_container_number

    recipe = {"newContainer": {"source": "ZCSU8877824"}}
    current = {"target": {}}
    first = extra_value(recipe, {}, current, "sample-a", None, project_root=ROOT)
    assert first == extra_value(recipe, {}, current, "sample-a", None, project_root=ROOT)
    assert first != extra_value(recipe, {}, current, "sample-b", None, project_root=ROOT)
    assert first[0] == first[1] and first[0] != "ZCSU8877824"
    assert validate_container_number(first[0])
    with pytest.raises(ValueError, match="source identity"):
        extra_value(
            {"newContainer": {"source": "INVALID"}}, {}, current, "a", None, project_root=ROOT
        )


def test_native_package_summary_projects_whole_shipment_count_when_reviewed():
    current = {
        "target": {
            "goodsItemDetails": [
                {"numberAndTypeOfPackages": [{"packageQuantity": 12}, {"packageQuantity": 8}]}
            ]
        }
    }
    recipe = {
        "packageSummary": {"goods": 0, "part": "quantity"},
        "target": "/goodsItemDetails/0/splitGoodsPlacement/0/packageQuantity",
    }
    rendered, semantic = extra_value(recipe, {}, current, "a", None, project_root=ROOT)
    assert rendered == "20" and semantic == 20
    assert extra_targets(recipe, semantic) == {recipe["target"]: 20}


def test_new_country_target_uses_pinned_preferred_name_not_iso_display():
    from document_ocr.synthesis.country_presentations import (
        DEFAULT_MANIFEST_SHA256,
        DEFAULT_REGISTRY_MANIFEST,
    )

    recipe = {
        "countryName": {
            "manifestPath": str(DEFAULT_REGISTRY_MANIFEST),
            "sha256": DEFAULT_MANIFEST_SHA256,
            "codePath": "/origin/country_code",
        }
    }
    candidate = {
        "scenario": {
            "origin": {"country_code": "CD", "country": "CONGO, THE DEMOCRATIC REPUBLIC OF THE"}
        }
    }
    assert extra_value(recipe, candidate, {}, "id", None, project_root=ROOT) == (
        "DEMOCRATIC REPUBLIC OF CONGO",
        "DEMOCRATIC REPUBLIC OF CONGO",
    )
    candidate["scenario"]["origin"]["country_code"] = "XX"
    with pytest.raises(ValueError, match="unknown ISO"):
        extra_value(recipe, candidate, {}, "id", None, project_root=ROOT)
    recipe["countryName"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        extra_value(recipe, candidate, {}, "id", None, project_root=ROOT)


def test_native_omission_is_explicit_and_cannot_create_a_target():
    recipe = {
        "omission": {
            "reason": "one sampled product replaces a two-code list",
            "authority": "explicit user approval",
        }
    }
    assert extra_value(recipe, {}, {}, "id", None, project_root=ROOT) == ("", None)
    for decision in (
        None,
        {},
        {"reason": "missing authority"},
        {"reason": "", "authority": "review"},
        {"reason": "x", "authority": 1},
    ):
        with pytest.raises(ValueError, match="omission"):
            extra_value({"omission": decision}, {}, {}, "id", None, project_root=ROOT)
    for field in ("target", "targets"):
        with pytest.raises(ValueError, match="omission"):
            extra_value({**recipe, field: "/x"}, {}, {}, "id", None, project_root=ROOT)


def test_nested_piece_counts_use_one_coherent_private_ledger():
    from copy import deepcopy

    class Vocabulary:
        def choose(self, category, styles, weights, seed):
            assert category == "PACKAGE_CARTON"
            return "CTNS", {}

    fact = {
        "owner": "inner-counts",
        "nativeFingerprint": "a" * 64,
        "levels": 3,
        "contentsKind": "goods_pieces",
        "relatedQuantity": 28,
        "relatedCategory": "PACKAGE_CARTON",
        "contents": {
            "packageSingular": "PIECE",
            "packagePlural": "PIECES",
            "unitsPerRelated": 11,
            "quantity": 308,
        },
    }
    group = {
        "facts": {"externalPackingPresentation": fact},
        "relation": "SUPPORTS",
        "publicQuantity": 4,
        "countedCategory": "PACKAGE_PALLET",
        "relatedCategory": "PACKAGE_CARTON",
        "relatedQuantity": 28,
    }
    candidate = {"scenario": {"provenance": {"freshPrivatePacking": group}}}
    current = {
        "target": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 4, "typeCategory": "PACKAGE_PALLET"}
                    ]
                }
            ]
        }
    }
    recipe = {
        "owner": "inner-counts",
        "nestedPacking": {"nativeFingerprint": "a" * 64, "style": "short"},
    }
    assert extra_value(recipe, candidate, current, "id", None, Vocabulary(), project_root=ROOT) == (
        "QTY:28 CTNS=308PCS",
        None,
    )
    baseline = deepcopy(candidate)
    from document_ocr.synthesis.package_presentations import PackageVocabulary

    specific = PackageVocabulary({
        "version": 1, "excludedUnscopedTokens": [],
        "entries": {"PACKAGE_CARTON": {
            "full": ["FIBREBOARD BOXES"], "short": [], "optional": [],
            "family": "PACKAGE_CARTON",
        }},
    })
    with pytest.raises(ValueError, match="no eligible package presentation"):
        extra_value(recipe, candidate, current, "id", None, specific, project_root=ROOT)
    expanded = deepcopy(recipe)
    expanded["nestedPacking"]["additionalStyles"] = ["full"]
    assert extra_value(expanded, candidate, current, "id", None, specific, project_root=ROOT)[0] == (
        "QTY:28 FIBREBOARD BOXES=308PCS"
    )
    for styles in (["short"], ["unreviewed"]):
        expanded["nestedPacking"]["additionalStyles"] = styles
        with pytest.raises(ValueError, match="distinct reviewed"):
            extra_value(expanded, candidate, current, "id", None, specific, project_root=ROOT)
    for key, value in (("quantity", 309), ("unitsPerRelated", 0), ("packageSingular", "BOTTLE")):
        bad = deepcopy(baseline)
        bad["scenario"]["provenance"]["freshPrivatePacking"]["facts"][
            "externalPackingPresentation"
        ]["contents"][key] = value
        with pytest.raises(ValueError, match="complete sampled"):
            extra_value(recipe, bad, current, "id", None, Vocabulary(), project_root=ROOT)


def test_native_lot_mass_partitions_conserve_the_sampled_count_without_source_ratios():
    from decimal import Decimal

    candidate = {
        "scenario": {
            "cargo": {"containerInformation": [{"equipmentIdentifier": "X"}]},
            "provenance": {
                "independentDraw": {
                    "allocations": [{"row": 0, "packageQuantity": 30, "netKg": "5250"}]
                },
                "freshPrivatePacking": {"unitNetKg": "175"},
            },
        }
    }

    def parts(identity):
        return [
            Decimal(
                extra_value(
                    {
                        "allocationPartition": {
                            "parts": 3,
                            "index": i,
                            "group": "lot4",
                            "sourceEquipmentIdentifier": "X",
                        }
                    },
                    candidate,
                    {},
                    identity,
                    None,
                    project_root=ROOT,
                )[0]
            )
            for i in range(3)
        ]

    assert parts("a") == parts("a")
    assert len({tuple(parts(str(i))) for i in range(20)}) > 1
    for i in range(20):
        values = parts(str(i))
        assert sum(values) == 5250 and all(v > 0 and v % 175 == 0 for v in values)
    candidate["scenario"]["provenance"]["independentDraw"]["allocations"][0]["netKg"] = "5000"
    with pytest.raises(ValueError, match="complete counted-unit mass"):
        parts("a")


def test_native_support_declaration_retains_public_count_and_private_support_count():
    class Vocabulary:
        def choose(self, category, styles, weights, seed):
            return {"PACKAGE_DRUM": "DRUMS", "PACKAGE_PALLET": "PLTS"}[category], {}

    fact = {
        "owner": "support",
        "nativeFingerprint": "a" * 64,
        "levels": 2,
        "contentsKind": "support_packages",
        "relatedQuantity": 3,
        "relatedCategory": "PACKAGE_PALLET",
    }
    group = {
        "facts": {"externalPackingPresentation": fact, "supportEnvelope": {"quantity": 3}},
        "relation": "ON_SUPPORT",
        "publicQuantity": 12,
        "countedCategory": "PACKAGE_DRUM",
        "relatedQuantity": 3,
        "relatedCategory": "PACKAGE_PALLET",
    }
    candidate = {"scenario": {"provenance": {"freshPrivatePacking": group}}}
    current = {
        "target": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 12, "typeCategory": "PACKAGE_DRUM"}
                    ]
                }
            ]
        }
    }
    recipe = {
        "owner": "support",
        "nestedPacking": {"nativeFingerprint": "a" * 64, "style": "short"},
    }
    assert extra_value(recipe, candidate, current, "id", None, Vocabulary(), project_root=ROOT) == (
        "12 DRUMS ON 3 PLTS",
        None,
    )
    group["facts"]["supportEnvelope"]["quantity"] = 4
    with pytest.raises(ValueError, match="support authority"):
        extra_value(recipe, candidate, current, "id", None, Vocabulary(), project_root=ROOT)


def test_native_numeric_presentation_preserves_separator_and_exact_value():
    candidate = {
        "proof": {
            "edits": [
                {"key": "net", "byteStart": 10, "after": "2010.44"},
                {"key": "net", "byteStart": 30, "after": "999.99"},
            ]
        }
    }
    recipe = {
        "owner": "net",
        "numericPresentation": {
            "key": "net",
            "byteStart": 10,
            "surface": "14686,300",
            "baseline": "14686.3",
        },
    }
    assert extra_value(recipe, candidate, {}, "id", None, project_root=ROOT) == ("2010,440", None)
    for value in ("NaN", "1,000", "12 KG", "-1", "123.45678"):
        candidate["proof"]["edits"][0]["after"] = value
        with pytest.raises(ValueError):
            extra_value(recipe, candidate, {}, "id", None, project_root=ROOT)


def test_new_tare_remainder_preserves_sampled_mixed_equipment_ledger():
    recipe = {
        "owner": "second-tare",
        "equipmentTareRemainder": {"totalSurface": "total", "partSurfaces": ["first"]},
    }
    candidate = {"surfaceValues": {"total": ["7400"], "first": ["3800"]}}
    assert extra_value(recipe, candidate, {}, "id", None, project_root=ROOT) == ("3600", None)
    candidate["surfaceValues"] = {"total": ["4460.50"], "first": ["2230.25"]}
    assert extra_value(recipe, candidate, {}, "id", None, project_root=ROOT) == ("2230.25", None)
    for value in ("NaN", "-1", "0", "4,460", "4460 KG", "2000"):
        candidate["surfaceValues"]["total"] = [value]
        with pytest.raises(ValueError):
            extra_value(recipe, candidate, {}, "id", None, project_root=ROOT)
    candidate["surfaceValues"]["total"] = ["7400", "7500"]
    with pytest.raises(ValueError, match="shared sampled"):
        extra_value(recipe, candidate, {}, "id", None, project_root=ROOT)
    for parts in ([], ["first", "first"], ["total"], [""]):
        recipe["equipmentTareRemainder"]["partSurfaces"] = parts
        with pytest.raises(ValueError):
            extra_value(recipe, candidate, {}, "id", None, project_root=ROOT)
    recipe["equipmentTareRemainder"]["partSurfaces"] = ["missing"]
    with pytest.raises(KeyError):
        extra_value(
            recipe, {"surfaceValues": {"total": ["7400"]}}, {}, "id", None, project_root=ROOT
        )


def test_new_allocation_projects_only_a_complete_conserving_independent_load():
    import copy

    public = [{"equipmentIdentifier": "A"}, {"equipmentIdentifier": "B"}]
    current = {
        "target": {
            "containerInformation": public,
            "goodsItemDetails": [
                {"numberAndTypeOfPackages": [{"packageQuantity": 3}, {"packageQuantity": 4}]}
            ],
        }
    }
    candidate = {
        "target": {"documentPatch": {"containerInformation": copy.deepcopy(public)}},
        "scenario": {
            "cargo": {
                "containerInformation": [
                    {"equipmentIdentifier": "OLD1"},
                    {"equipmentIdentifier": "OLD2"},
                ]
            },
            "provenance": {
                "independentDraw": {
                    "allocations": [
                        {"row": 0, "packageQuantity": 2},
                        {"row": 1, "packageQuantity": 5},
                    ]
                }
            },
        },
    }
    recipe = {"allocation": {"index": 1}}
    assert extra_value(recipe, candidate, current, "id", None, project_root=ROOT) == ("5", 5)
    assert "splitGoodsPlacement" not in current["target"]["goodsItemDetails"][0]
    for mutate in (
        lambda c: c["scenario"]["provenance"]["independentDraw"]["allocations"].pop(),
        lambda c: c["scenario"]["provenance"]["independentDraw"]["allocations"].reverse(),
        lambda c: c["scenario"]["provenance"]["independentDraw"]["allocations"][0].update(row=True),
        lambda c: c["scenario"]["provenance"]["independentDraw"]["allocations"][0].update(
            packageQuantity=True
        ),
        lambda c: c["scenario"]["provenance"]["independentDraw"]["allocations"][0].update(
            packageQuantity=3
        ),
        lambda c: c["target"]["documentPatch"]["containerInformation"].reverse(),
    ):
        bad = copy.deepcopy(candidate)
        mutate(bad)
        with pytest.raises(ValueError):
            extra_value(recipe, bad, current, "id", None, project_root=ROOT)
    for index in (-1, True, 2):
        with pytest.raises(ValueError):
            extra_value(
                {"allocation": {"index": index}}, candidate, current, "id", None, project_root=ROOT
            )


def test_rendered_mass_sum_uses_complete_distinct_numeric_owners():
    receipts = [{"owner": "first", "after": "6441.71"}, {"owner": "second", "after": "3988.91"}]
    action = {"owners": ["first", "second"], "unit": "kilogram"}
    assert rendered_mass_sum(receipts, action) == {"value": 10430.62, "unit": "kilogram"}
    for owners in (["first"], ["first", "first"], ["first", "missing"]):
        with pytest.raises(ValueError):
            rendered_mass_sum(receipts, {**action, "owners": owners})
    for unit in ("unknown", "cubic_metre"):
        with pytest.raises(ValueError):
            rendered_mass_sum(receipts, {**action, "unit": unit})
    for value in ("1,000", "NaN", "Infinity", "0", "-1", "12 KG", "1_000"):
        with pytest.raises(ValueError):
            rendered_mass_sum([receipts[0], {"owner": "second", "after": value}], action)


def test_mass_sum_decimal_comma_requires_explicit_reviewed_convention():
    receipts = [{"owner": "a", "after": "21011,13"}, {"owner": "b", "after": "14814,15"}]
    action = {"owners": ["a", "b"], "unit": "kilogram", "decimalSeparator": ","}
    assert rendered_mass_sum(receipts, action) == {"value": 35825.28, "unit": "kilogram"}
    for separator in (".", "", ";"):
        with pytest.raises(ValueError):
            rendered_mass_sum(receipts, {**action, "decimalSeparator": separator})
    for invalid in ("21.011,13", "21,011,13", "21011.13"):
        with pytest.raises(ValueError):
            rendered_mass_sum([{**receipts[0], "after": invalid}, receipts[1]], action)


def test_compound_owner_projects_only_reviewed_semantic_components():
    pair = {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
    mapping = {key: "/containerInformation/0/" + key for key in pair}
    assert extra_targets({"targets": mapping}, pair) == {
        path: pair[key] for key, path in mapping.items()
    }
    assert extra_targets({}, pair) == {}
    assert extra_targets({"target": "/mass"}, {"value": 10, "unit": "kilogram"}) == {
        "/mass": {"value": 10, "unit": "kilogram"}
    }
    for invalid in (
        {"targets": {}},
        {"targets": {"absent": "/x"}},
        {"targets": {"sizeCategory": "/x", "typeCategory": "/x"}},
        {"targets": {"sizeCategory": "relative"}},
        {"target": "/x", "targets": mapping},
    ):
        with pytest.raises(ValueError):
            extra_targets(invalid, pair)
    with pytest.raises(ValueError, match="semantic"):
        extra_targets({"targets": mapping}, None)


def test_new_equipment_uses_an_explicit_independent_authority_without_label_guessing():
    import copy

    public = [{"equipmentIdentifier": "ABCU1234567"}]
    planned = [{"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}]
    candidate = {
        "target": {"documentPatch": {"containerInformation": copy.deepcopy(public)}},
        "scenario": {
            "cargo": {"containerInformation": [{"equipmentIdentifier": "OLDU1234567"}]},
            "provenance": {"independentDraw": {"equipment": planned}},
        },
    }
    current = {"target": {"containerInformation": public}}
    spec = {"index": 0, "authority": "independent_plan"}
    assert sampled_equipment(spec, candidate, current) == {**public[0], **planned[0]}
    assert sampled_equipment({"index": 0}, candidate, current) == public[0]
    for invalid in ({**spec, "index": -1}, {**spec, "index": True}, {**spec, "authority": "guess"}):
        with pytest.raises(ValueError):
            sampled_equipment(invalid, candidate, current)
    public[0]["sizeCategory"] = "TWENTY_FOOT_STANDARD_HEIGHT"
    with pytest.raises(ValueError, match="conflicts"):
        sampled_equipment(spec, candidate, current)
    del public[0]["sizeCategory"]
    public[0]["equipmentIdentifier"] = "OTHER"
    with pytest.raises(ValueError, match="ordering"):
        sampled_equipment(spec, candidate, current)
    public[0]["equipmentIdentifier"] = "ABCU1234567"
    planned.append(dict(planned[0]))
    with pytest.raises(ValueError, match="count"):
        sampled_equipment(spec, candidate, current)


def test_new_equipment_summary_counts_mixed_inventory_without_copying_source_size():
    class Registry:
        def select(self, size, kind, stream):
            assert kind == "GENERAL_PURPOSE"
            return {"FORTY_FOOT_HIGH_CUBE": "40HC", "TWENTY_FOOT_STANDARD_HEIGHT": "20GP"}[size]

    current = {
        "target": {
            "containerInformation": [
                {
                    "equipmentIdentifier": "A",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "equipmentIdentifier": "B",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "equipmentIdentifier": "C",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }
    extra = {"owner": "summary", "equipmentSummary": {"source": "20GP"}}
    assert extra_value(extra, {}, current, "shipment", Registry(), project_root=ROOT) == (
        "2 X 40HC + 1 X 20GP",
        None,
    )
    current["target"]["containerInformation"][2]["equipmentIdentifier"] = "A"
    with pytest.raises(ValueError, match="distinct"):
        extra_value(extra, {}, current, "shipment", Registry(), project_root=ROOT)


def test_existing_numeric_owner_can_supply_new_target_only_when_render_matches():
    from dataclasses import replace

    class Engine:
        def render(self, pages, blocks, **kwargs):
            assert len(blocks) == 1
            region = replace(pages[0].regions[0], text=blocks[0].text)
            return (replace(pages[0], regions=(region,)),), {}

    page = PaddlePage(0, 100, 100, (PaddleRegion(0, "10.000", (1, 1, 80, 10)),))
    record = {"documentId": "source", "pages": [asdict(page)]}
    span = {"page": 0, "region": 0, "start": 0, "end": 6, "expected": "10.000"}
    transfer = {
        "slots": [
            {"owner": "volume@0:6", "status": "exact_measured_owner_candidate", "spans": [span]}
        ]
    }
    candidate = {
        "joinedRawText": "old",
        "proof": {
            "edits": [
                {
                    "key": "volume",
                    "byteStart": 0,
                    "byteEnd": 6,
                    "before": "10.000",
                    "after": "12.500",
                }
            ]
        },
        "scenario": {
            "provenance": {
                "explicitPhysicalFacts": {"volume": {"unit": "cubic_metre", "value": "12.5"}}
            }
        },
    }
    current = {
        "joinedRawText": "old",
        "target": {
            "negotiability": "non_negotiable",
            "goodsItemDetails": [{"description": "CARGO"}],
        },
    }
    overlay = {
        "overrides": [],
        "targetCopies": [],
        "targetSets": {},
        "extras": [
            {
                "owner": "native-volume",
                "regions": [[0, 0]],
                "existingKey": "volume",
                "physical": {
                    "field": "volume",
                    "unit": "cubic_metre",
                    "minPlaces": 3,
                    "grouped": False,
                    "prefix": "",
                    "suffix": "",
                },
                "target": "/goodsItemDetails/0/volume",
            }
        ],
    }
    args = (record, transfer, candidate, current, overlay, Engine(), "x", None)
    result = render_one(*args, wording_receipt=None, project_root=ROOT)
    assert result["target"]["goodsItemDetails"][0]["volume"] == {
        "value": 12.5,
        "unit": "cubic_metre",
    }
    assert len(result["proof"]) == 1
    overlay["literalProjections"] = [
        {
            "target": "/goodsItemDetails/0/handlingInstructions",
            "separator": "",
            "asList": True,
            "pieces": [{"fromKey": "volume"}],
        }
    ]
    result = render_one(*args, wording_receipt=None, project_root=ROOT)
    assert result["target"]["goodsItemDetails"][0]["handlingInstructions"] == ["12.500"]
    candidate["proof"]["edits"][0]["after"] = "12.499"
    with pytest.raises(ValueError, match="disagrees"):
        render_one(*args, wording_receipt=None, project_root=ROOT)


def test_existing_adjacent_owner_separator_changes_only_its_presentation():
    from dataclasses import replace

    class Engine:
        def render(self, pages, blocks, **kwargs):
            assert len(blocks) == 1
            regions = tuple(
                PaddleRegion(i, text, (1, 1 + i * 12, 90, 10 + i * 12))
                for i, text in enumerate(blocks[0].text.splitlines())
            )
            return (replace(pages[0], regions=regions),), {}

    page = PaddlePage(0, 100, 100, (PaddleRegion(0, "FIRM3, ROAD", (1, 1, 90, 10)),))
    record = {"documentId": "source", "pages": [asdict(page)]}
    edit = {
        "key": "address",
        "byteStart": 4,
        "byteEnd": 11,
        "before": "3, ROAD",
        "after": "9, AVENUE",
    }
    candidate = {"joinedRawText": "old", "proof": {"edits": [edit]}}
    current = {
        "joinedRawText": "old",
        "target": {
            "negotiability": "non_negotiable",
            "goodsItemDetails": [{"description": "CARGO"}],
            "parties": {"shipper": {"name": "FIRM", "addressLine": "9, AVENUE"}},
        },
    }
    override = {"key": "address", "fragments": [[0, 0, "3, ROAD"]], "separatorBefore": "\n"}
    overlay = {"overrides": [override], "extras": [], "targetCopies": [], "targetSets": {}}
    args = (record, {"slots": []}, candidate, current, overlay, Engine(), "x", None)
    result = render_one(*args, wording_receipt=None, project_root=ROOT)
    assert "FIRM\n9, AVENUE" in result["joinedRawText"]
    assert result["proof"][0]["after"] == "\n9, AVENUE"
    assert result["target"] == current["target"]
    for invalid in ("NAME", None, 1):
        override["separatorBefore"] = invalid
        with pytest.raises(ValueError, match="separator"):
            render_one(*args, wording_receipt=None, project_root=ROOT)


def test_new_owner_can_preserve_separate_frames_around_unrelated_text():
    from dataclasses import replace
    from types import SimpleNamespace

    class Engine:
        metrics = SimpleNamespace(length=len)

        def render(self, pages, blocks, **kwargs):
            assert [b.regions for b in blocks] == [(0,), (2,)]
            regions = list(pages[0].regions)
            for block in blocks:
                regions[block.regions[0]] = replace(regions[block.regions[0]], text=block.text)
            return (replace(pages[0], regions=tuple(regions)),), {}

    page = PaddlePage(
        0,
        100,
        100,
        (
            PaddleRegion(0, "OLD COMPANY", (1, 1, 90, 10)),
            PaddleRegion(1, "UNRELATED CONTACT", (1, 15, 90, 25)),
            PaddleRegion(2, "AGENCY", (1, 30, 90, 40)),
        ),
    )
    record = {"documentId": "source", "pages": [asdict(page)]}
    candidate = {
        "joinedRawText": "old",
        "proof": {"edits": [{"key": "name", "after": "NEW MARITIME AGENCY"}]},
    }
    current = {
        "joinedRawText": "old",
        "target": {
            "negotiability": "non_negotiable",
            "goodsItemDetails": [{"description": "CARGO"}],
        },
    }
    # This value is already sampled, but its old occurrence is not transferred
    # in this fixture; only the newly visible occurrence owns these native spans.
    candidate["proof"]["edits"][0].update(byteStart=0, byteEnd=3, before="OLD")
    overlay = {
        "overrides": [],
        "targetCopies": [],
        "targetSets": {},
        "absentEdits": [
            {
                "key": "name",
                "byteStart": 0,
                "before": "OLD",
                "reason": "old occurrence not printed in this native view",
            }
        ],
        "extras": [
            {
                "owner": "new-name",
                "fromKey": "name",
                "regions": [[0, 0], [0, 2]],
                "separateRegionFrames": True,
            }
        ],
    }
    result = render_one(
        record,
        {"slots": []},
        candidate,
        current,
        overlay,
        Engine(),
        "x",
        None,
        wording_receipt=None,
        project_root=ROOT,
    )
    assert result["pages"][0]["regions"][1]["text"] == "UNRELATED CONTACT"
    assert " ".join(r["after"] for r in result["proof"]) == "NEW MARITIME AGENCY"
    assert [r["owner"] for r in result["proof"]] == ["new-name:frame:0", "new-name:frame:1"]
    overlay["extras"][0]["regions"] = [[0, 0]]
    with pytest.raises(ValueError, match="distinct ordered"):
        render_one(
            record,
            {"slots": []},
            candidate,
            current,
            overlay,
            Engine(),
            "x",
            None,
            wording_receipt=None,
            project_root=ROOT,
        )


def test_optional_new_occurrence_checks_only_its_explicit_existing_byte_owner():
    from dataclasses import replace

    class Engine:
        def render(self, pages, blocks, **kwargs):
            regions = tuple(replace(pages[0].regions[b.regions[0]], text=b.text) for b in blocks)
            return (replace(pages[0], regions=regions),), {}

    page = PaddlePage(
        0,
        100,
        100,
        (PaddleRegion(0, "BAGS", (1, 1, 40, 10)), PaddleRegion(1, "BAGS", (1, 20, 40, 30))),
    )
    record = {"documentId": "source", "pages": [asdict(page)]}
    edits = [
        {
            "key": "type",
            "byteStart": start,
            "byteEnd": start + 4,
            "before": "BAGS",
            "after": "DRUMS",
        }
        for start in (0, 20)
    ]
    candidate = {"joinedRawText": "old", "proof": {"edits": edits}}
    current = {
        "joinedRawText": "old",
        "target": {
            "negotiability": "non_negotiable",
            "goodsItemDetails": [{"description": "CARGO"}],
        },
    }
    overrides = [
        {"key": "type", "byteStart": start, "regions": [[0, i]]} for i, start in enumerate((0, 20))
    ]
    extra = {
        "owner": "header",
        "regions": [[0, 0]],
        "existingKey": "type",
        "existingByteStart": 0,
        "literalText": "DRUMS",
    }
    overlay = {"overrides": overrides, "extras": [extra], "targetCopies": [], "targetSets": {}}
    args = (record, {"slots": []}, candidate, current, overlay, Engine(), "x", None)
    assert len(render_one(*args, wording_receipt=None, project_root=ROOT)["proof"]) == 2
    candidate["proof"]["edits"] = edits[1:]
    assert len(render_one(*args, wording_receipt=None, project_root=ROOT)["proof"]) == 2
    extra["existingByteStart"] = 20
    with pytest.raises(ValueError, match="does not cover"):
        render_one(*args, wording_receipt=None, project_root=ROOT)


def test_absent_occurrence_is_explicit_and_does_not_remove_other_roles():
    edits = [
        {"key": "city", "byteStart": n, "before": "ALEXANDRIA", "after": "DUBAI"} for n in (10, 20)
    ]
    decision = {
        "key": "city",
        "byteStart": 20,
        "before": "ALEXANDRIA",
        "reason": "Payment value is absent from native Paddle; POD remains printed.",
    }
    assert present_source_edits(edits, []) == edits
    assert present_source_edits(edits, [decision]) == edits[:1]
    for invalid in (
        [decision, decision],
        [{**decision, "byteStart": 30}],
        [{**decision, "before": "OTHER"}],
        [{**decision, "reason": ""}],
    ):
        with pytest.raises(ValueError, match="absent source occurrence"):
            present_source_edits(edits, invalid)


def test_new_vessel_samples_valid_support_without_retaining_source():
    extra = {
        "owner": "vessel",
        "vessel": {
            "source": "CMA CGM MANTA RAY",
            "support": {
                "authority": {"method": "reviewed train-only names"},
                "names": ["CMA CGM MANTA RAY", "MAERSK HAMBURG"],
            },
        },
    }
    result = extra_value(extra, {}, {}, "ship", None, project_root=ROOT)
    assert result == ("MAERSK HAMBURG", "MAERSK HAMBURG")
    assert extra_value(extra, {}, {}, "ship", None, project_root=ROOT) == result
    extra["vessel"]["support"]["names"] = ["CMA CGM MANTA RAY"]
    with pytest.raises(ValueError, match="support"):
        extra_value(extra, {}, {}, "ship", None, project_root=ROOT)


def test_new_hs_uses_sampled_identity_and_native_format():
    candidate = {
        "scenario": {"goods_identities": [{"hs6": "610910"}]},
        "proof": {"edits": [{"key": "hs", "after": "61091012345"}]},
    }
    extra = {"owner": "hs-native", "hsCode": {"source": "4819.1000.0000", "fromKey": "hs"}}
    surface, target = extra_value(extra, candidate, {}, "shipment", None, project_root=ROOT)
    assert surface == target and surface.startswith("6109.10")
    assert len(surface.replace(".", "")) == 12
    assert extra_value(extra, candidate, {}, "shipment", None, project_root=ROOT) == (
        surface,
        target,
    )
    extra["hsCode"] = {"source": "961320", "goodsIdentity": 0}
    assert extra_value(extra, candidate, {}, "shipment", None, project_root=ROOT) == (
        "610910",
        "610910",
    )
    extra["hsCode"] = {"source": "961320", "fromKey": "absent"}
    with pytest.raises(ValueError, match="one sampled identity"):
        extra_value(extra, candidate, {}, "shipment", None, project_root=ROOT)


def test_fresh_record_requires_exact_receipt_identity_and_hash(tmp_path, monkeypatch):
    import json

    import render_r30_paddle_pilot as rendering

    from document_ocr.hashing import sha256_file

    monkeypatch.setattr(rendering, "ROOT", tmp_path)
    receipt = tmp_path / "shipment.json"
    receipt.write_text('{"shipment": 1}')
    digest = sha256_file(receipt)
    record = {"documentId": "fresh", "receiptSha256": digest, "target": {}}
    (tmp_path / "record.json").write_text(json.dumps(record))
    selection = {
        "id": "fresh",
        "path": "shipment.json",
        "freshRecord": "record.json",
        "receiptSha256": digest,
    }
    assert load_current(selection, {}) == record
    for bad in ({**selection, "id": "other"}, {**selection, "receiptSha256": "wrong"}):
        with pytest.raises(ValueError, match="differs"):
            load_current(bad, {})
    receipt.write_text('{"shipment": 2}')
    with pytest.raises(ValueError, match="differs"):
        load_current(selection, {})


def test_ordered_native_contact_projection_keeps_each_number_separate():
    receipts = [
        {"owner": "old:first", "after": "+20 22\n333"},
        {"owner": "old:last", "after": "01111111111"},
        {"owner": "new:middle", "after": "02222222222"},
    ]
    action = {"owners": ["old:first", "new:middle", "old:last"], "asList": True}
    assert rendered_owner_projection(receipts, action) == [
        "+20 22 333",
        "02222222222",
        "01111111111",
    ]
    assert (
        rendered_owner_projection(receipts, {**action, "asList": False, "separator": " / "})
        == "+20 22 333 / 02222222222 / 01111111111"
    )
    for invalid in ([], ["absent"]):
        with pytest.raises(ValueError, match="owner"):
            rendered_owner_projection(receipts, {"owners": invalid, "asList": True})
    with pytest.raises(ValueError, match="exact owner"):
        rendered_owner_projection(receipts + receipts, action)


def test_absent_occurrence_may_have_no_edit_only_with_unchanged_sampled_authority():
    decision = {"key": "equipment", "byteStart": 10, "before": "20DV", "reason": "GLM duplicate"}
    assert (
        present_source_edits([], [decision], surface_values={"equipment": ["20DV", "20DV"]}) == []
    )
    for values in ({}, {"equipment": []}, {"equipment": ["40HC"]}, {"equipment": ["20DV", "40HC"]}):
        with pytest.raises(ValueError, match="missing"):
            present_source_edits([], [decision], surface_values=values)
    with pytest.raises(ValueError, match="multiply disposed"):
        present_source_edits([], [decision, decision], surface_values={"equipment": "20DV"})
    with pytest.raises(ValueError, match="conflicts"):
        present_source_edits(
            [{"key": "equipment", "byteStart": 10, "before": "40HC", "after": "20DV"}],
            [decision],
            surface_values={"equipment": "20DV"},
        )


def test_new_date_uses_existing_shipment_offset_and_each_reviewed_format():
    identity, seed = "shipment", 42
    stream = DeterministicStream(seed, "full-current-v7", identity)
    expected = date(2024, 7, 1) + timedelta(days=30 + stream.derive("date-shift").randbelow(730))
    assert shifted_public_date("2024-07-01", stream) == expected.isoformat()
    for format_ in ("%d/%m/%Y", "%B {day_ordinal}, %Y"):
        extra = {
            "owner": "new-date",
            "sourceDate": {"source": "2024-07-01", "format": format_, "uppercase": True},
        }
        surface, value = extra_value(
            extra, {"seed": seed}, {"target": {}}, identity, None, project_root=ROOT
        )
        assert value == expected.isoformat()
        assert str(expected.year) in surface and "{day_ordinal}" not in surface
    with pytest.raises(ValueError):
        shifted_public_date("not-a-date", stream)


def test_new_date_follows_actual_preserved_date_not_cargo_refresh_seed():
    extra = {
        "owner": "onboard",
        "sourceDate": {
            "source": "2024-03-13",
            "format": "%b.%d, %Y",
            "uppercase": True,
            "reference": {"source": "2024-03-13", "target": "/issueDate"},
        },
    }
    current = {"target": {"issueDate": "2024-04-21"}}
    for seed in (1, 42, 20261031):
        assert extra_value(
            extra, {"seed": seed}, current, "refreshed-id", None, project_root=ROOT
        ) == (
            "APR.21, 2024",
            "2024-04-21",
        )
    extra["sourceDate"]["source"] = "2024-03-10"
    assert (
        extra_value(extra, {"seed": 1}, current, "different-id", None, project_root=ROOT)[1]
        == "2024-04-18"
    )
    del extra["sourceDate"]["reference"]
    with pytest.raises(ValueError, match="existing shipment date"):
        extra_value(extra, {"seed": 1}, current, "x", None, project_root=ROOT)


def test_new_linguistic_owner_preserves_render_lines_but_folds_target():
    surface = "PORT ADMINISTRATION BUILDING 3\nFLOOR 2, OFFICE 214"
    rule = {"owner": "new-address", "wording": "address"}
    assert extra_value(
        rule, {}, {}, "x", None, wording={"address": surface}, project_root=ROOT
    ) == (
        surface,
        "PORT ADMINISTRATION BUILDING 3 FLOOR 2, OFFICE 214",
    )


def test_new_repeated_measure_keeps_sampled_precision_and_explicit_unit():
    candidate = {
        "scenario": {
            "provenance": {
                "explicitPhysicalFacts": {"volume": {"value": "15.2345", "unit": "cubic_metre"}}
            }
        }
    }
    extra = {
        "owner": "new-volume",
        "physical": {
            "field": "volume",
            "unit": "cubic_metre",
            "minPlaces": 3,
            "grouped": False,
            "prefix": "",
            "suffix": " CBM",
        },
    }
    text, target = extra_value(extra, candidate, {}, "test", None, project_root=ROOT)
    assert text == "15.2345 CBM"
    assert target == {"value": 15.2345, "unit": "cubic_metre"}
    extra["physical"]["decimalSeparator"] = ","
    assert extra_value(extra, candidate, {}, "test", None, project_root=ROOT) == (
        "15,2345 CBM",
        {"value": 15.2345, "unit": "cubic_metre"},
    )
    candidate["scenario"]["provenance"]["explicitPhysicalFacts"]["volume"]["value"] = "1234.2345"
    extra["physical"]["grouped"] = True
    assert extra_value(extra, candidate, {}, "test", None, project_root=ROOT)[0] == "1.234,2345 CBM"
    extra["physical"]["decimalSeparator"] = ";"
    with pytest.raises(ValueError, match="decimal separator"):
        extra_value(extra, candidate, {}, "test", None, project_root=ROOT)
    del extra["physical"]["decimalSeparator"]
    extra["physical"]["unit"] = "kilogram"
    with pytest.raises(ValueError, match="incompatible"):
        extra_value(extra, candidate, {}, "test", None, project_root=ROOT)


def test_source_identifiers_share_identity_only_for_the_same_reference():
    def sample(value):
        return extra_value(
            {"owner": "ref", "sourceIdentifier": value}, {}, {}, "test", None, project_root=ROOT
        )[0]

    assert sample("1234567") == sample("1234567")
    assert sample("1234567") != sample("7654321")


def test_new_tare_uses_reviewed_equipment_support_without_a_public_target():
    pair = "FORTY_FIVE_FOOT_HIGH_CUBE|GENERAL_PURPOSE"
    current = {
        "target": {
            "containerInformation": [
                {"sizeCategory": "FORTY_FIVE_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
            ]
        }
    }
    rule = {
        "owner": "new-tare",
        "equipmentTare": {
            "index": 0,
            "minPlaces": 2,
            "support": {"pairs": {pair: [{"kg": "4700", "sourceIds": ["reviewed-train"]}]}},
        },
    }
    assert extra_value(rule, {}, current, "shipment", None, project_root=ROOT) == ("4700.00", None)
    rule["equipmentTare"]["grouped"] = True
    assert extra_value(rule, {}, current, "shipment", None, project_root=ROOT) == ("4,700.00", None)
    rule["equipmentTare"]["grouped"] = False
    rule["equipmentTare"]["presentation"] = {"surface": "3.840", "baseline": "3840"}
    assert extra_value(rule, {}, current, "shipment", None, project_root=ROOT) == ("4.700", None)
    rule["equipmentTare"]["presentation"] = {"surface": "3 840,00", "baseline": "3840"}
    assert extra_value(rule, {}, current, "shipment", None, project_root=ROOT) == ("4 700,00", None)
    del rule["equipmentTare"]["presentation"]
    rule["equipmentTare"]["support"]["pairs"][pair].append(
        {"kg": "4800", "sourceIds": ["second-reviewed-train"]}
    )
    result = extra_value(rule, {}, current, "shipment", None, project_root=ROOT)
    assert result == extra_value(rule, {}, current, "shipment", None, project_root=ROOT)
    assert result[0] in {"4700.00", "4800.00"}
    del rule["equipmentTare"]["support"]["pairs"][pair]
    with pytest.raises(ValueError, match="pair support"):
        extra_value(rule, {}, current, "shipment", None, project_root=ROOT)
    rule["equipmentTare"]["support"]["pairs"][pair] = [{"kg": "NaN", "sourceIds": ["x"]}]
    with pytest.raises(ValueError, match="finite"):
        extra_value(rule, {}, current, "shipment", None, project_root=ROOT)


def test_new_tare_rows_preserve_the_existing_aggregate_without_equal_division():
    pair = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
    current = {
        "target": {
            "containerInformation": [
                {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
                for _ in range(2)
            ]
        }
    }
    candidate = {"proof": {"edits": [{"key": "sum", "after": "4450"}]}}
    spec = {
        "index": 0,
        "minPlaces": 0,
        "aggregate": {"indices": [0, 1], "fromKey": "sum"},
        "support": {
            "pairs": {
                pair: [
                    {"kg": "2200", "sourceIds": ["reviewed-train"]},
                    {"kg": "2250", "sourceIds": ["reviewed-train"]},
                ]
            }
        },
    }

    def sample(index):
        return extra_value(
            {"owner": f"tare-{index}", "equipmentTare": {**spec, "index": index}},
            candidate,
            current,
            "shipment",
            None,
            project_root=ROOT,
        )

    assert sorted([sample(0), sample(1)]) == [("2200", None), ("2250", None)]
    assert sample(0) == sample(0)
    candidate["proof"]["edits"][0]["after"] = "4449"
    with pytest.raises(ValueError, match="fixed tare aggregate"):
        sample(0)
    candidate["proof"]["edits"][0]["after"] = "NaN"
    with pytest.raises(ValueError, match="positive finite"):
        sample(0)
    for indices in ([1], [1, 0], [0, 0], [False, 1]):
        spec["aggregate"]["indices"] = indices
        with pytest.raises(ValueError, match="ordered distinct"):
            sample(0)
    spec["aggregate"]["indices"] = [0, 2]
    with pytest.raises(ValueError, match="invalid inventory"):
        sample(0)


def test_new_tare_sum_matches_independently_sampled_rows():
    pairs = [
        "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE",
        "FORTY_FIVE_FOOT_HIGH_CUBE|GENERAL_PURPOSE",
    ]
    current = {
        "target": {
            "containerInformation": [
                dict(zip(("sizeCategory", "typeCategory"), pair.split("|"), strict=True))
                for pair in pairs
            ]
        }
    }
    spec = {
        "minPlaces": 0,
        "support": {
            "pairs": {
                pairs[0]: [{"kg": str(v), "sourceIds": ["reviewed-train"]} for v in (2200, 2250)],
                pairs[1]: [{"kg": str(v), "sourceIds": ["reviewed-train"]} for v in (4700, 4800)],
            }
        },
    }

    def sample(spec, identity):
        return extra_value(
            {"owner": "tare", "equipmentTare": spec}, {}, current, identity, None, project_root=ROOT
        )

    for identity in ("first", "second", "third"):
        values = [int(sample({**spec, "index": i}, identity)[0]) for i in range(2)]
        assert sample({**spec, "indices": [0, 1]}, identity) == (str(sum(values)), None)
    for indices in ([], [1, 0], [0, 0], [False, 1]):
        with pytest.raises(ValueError, match="ordered distinct"):
            sample({**spec, "indices": indices}, "bad")
    with pytest.raises(ValueError, match="cannot also"):
        sample({**spec, "indices": [0, 1], "index": 0}, "bad")
    with pytest.raises(ValueError, match="cannot also"):
        sample({**spec, "indices": [0, 1], "aggregate": {}}, "bad")


def test_split_email_uses_one_complete_sampled_identity():
    candidate = {"proof": {"edits": [{"key": "email", "after": "cargo.office@example.test"}]}}
    rule = {"owner": "email-continuation", "fromKey": "email"}
    assert extra_value(
        {**rule, "emailPart": "local"}, candidate, {}, "s", None, project_root=ROOT
    ) == (
        "cargo.office",
        None,
    )
    assert extra_value(
        {**rule, "emailPart": "domain"}, candidate, {}, "s", None, project_root=ROOT
    ) == (
        "example.test",
        None,
    )
    with pytest.raises(ValueError, match="unknown email"):
        extra_value({**rule, "emailPart": "other"}, candidate, {}, "s", None, project_root=ROOT)
    with pytest.raises(ValueError, match="cannot also"):
        extra_value(
            {**rule, "emailPart": "local", "line": 0}, candidate, {}, "s", None, project_root=ROOT
        )
    for invalid in ("not-mail", "@example.test", "name@", "name@two@hosts"):
        candidate["proof"]["edits"][0]["after"] = invalid
        with pytest.raises(ValueError, match="complete sampled mailbox"):
            extra_value({**rule, "emailPart": "local"}, candidate, {}, "s", None, project_root=ROOT)


def test_new_split_mailbox_keeps_complete_semantic_target():
    wording = {"mail": "cargo.office@example.test"}
    for component, rendered in (("local", "cargo.office"), ("domain", "example.test")):
        rule = {"wording": "mail", "emailPart": component}
        assert extra_value(rule, {}, {}, "s", None, wording=wording, project_root=ROOT) == (
            rendered,
            "cargo.office@example.test",
        )
    for invalid in ("no-mail", "@host", "name@", "a@b@c", "a @host"):
        with pytest.raises(ValueError, match="complete sampled mailbox"):
            extra_value(rule, {}, {}, "s", None, wording={"mail": invalid}, project_root=ROOT)
    with pytest.raises(ValueError, match="unknown wording email component"):
        extra_value(
            {"wording": "mail", "emailPart": "unknown"},
            {},
            {},
            "s",
            None,
            wording=wording,
            project_root=ROOT,
        )


def test_new_reference_combines_existing_owner_with_generated_private_suffix():
    candidate = {"proof": {"edits": [{"key": "tax", "after": "638194725"}]}}
    rule = {
        "owner": "master-customs-reference",
        "parts": [{"fromKey": "tax"}, {"sourceIdentifier": "2024040077"}],
    }
    value, semantic = extra_value(rule, candidate, {}, "shipment", None, project_root=ROOT)
    assert semantic is None
    assert value.startswith("638194725") and len(value) == 19 and value.isdigit()
    assert value[9:] != "2024040077"
    assert extra_value(rule, candidate, {}, "shipment", None, project_root=ROOT)[0] == value
    with pytest.raises(ValueError, match="one exact shared value"):
        extra_value(rule, {"proof": {"edits": []}}, {}, "shipment", None, project_root=ROOT)
    with pytest.raises(ValueError, match="at least two"):
        extra_value(
            {**rule, "parts": rule["parts"][:1]}, candidate, {}, "shipment", None, project_root=ROOT
        )
    with pytest.raises(ValueError, match="cannot be nested"):
        extra_value(
            {**rule, "parts": [rule, rule]}, candidate, {}, "shipment", None, project_root=ROOT
        )


def test_dependent_booking_retains_the_entire_sampled_bill_suffix():
    candidate = {"proof": {"edits": [{"key": "bill", "after": "ABCD123456789012"}]}}
    rule = {"owner": "booking", "fromKey": "bill", "segments": [[4, None]], "join": ""}
    assert extra_value(rule, candidate, {}, "shipment", None, project_root=ROOT) == (
        "123456789012",
        None,
    )
    candidate["proof"]["edits"][0]["after"] = "ABCD987654321"
    assert extra_value(rule, candidate, {}, "shipment", None, project_root=ROOT) == (
        "987654321",
        None,
    )


def test_converted_alias_rounds_printout_not_authoritative_value():
    candidate = {
        "scenario": {
            "provenance": {
                "explicitPhysicalFacts": {"grossWeight": {"value": "580", "unit": "kilogram"}}
            }
        }
    }
    extra = {
        "owner": "pounds",
        "physical": {
            "field": "grossWeight",
            "unit": "kilogram",
            "renderUnit": "pound",
            "minPlaces": 3,
            "grouped": False,
            "prefix": "",
            "suffix": " LB",
        },
    }
    assert extra_value(extra, candidate, {}, "x", None, project_root=ROOT) == (
        "1278.681 LB",
        {"value": 1278.681, "unit": "pound"},
    )
    assert (
        candidate["scenario"]["provenance"]["explicitPhysicalFacts"]["grossWeight"]["value"]
        == "580"
    )
    extra["physical"]["renderUnit"] = "cubic_foot"
    with pytest.raises(ValueError, match="dimension"):
        extra_value(extra, candidate, {}, "x", None, project_root=ROOT)


def test_decimal_comma_tonnes_project_the_converted_printed_measure():
    candidate = {
        "scenario": {
            "provenance": {
                "explicitPhysicalFacts": {"grossWeight": {"value": "3052", "unit": "kilogram"}}
            }
        }
    }
    recipe = {
        "owner": "mass",
        "physical": {
            "field": "grossWeight",
            "unit": "kilogram",
            "renderUnit": "metric_tonne",
            "minPlaces": 3,
            "grouped": False,
            "decimalSeparator": ",",
            "prefix": "",
            "suffix": "",
        },
    }
    assert extra_value(recipe, candidate, {}, "x", None, project_root=ROOT) == (
        "3,052",
        {"value": 3.052, "unit": "metric_tonne"},
    )
    recipe["physical"]["decimalSeparator"] = "?"
    with pytest.raises(ValueError, match="separator"):
        extra_value(recipe, candidate, {}, "x", None, project_root=ROOT)


def test_invalid_native_imo_does_not_require_an_invalid_synthetic_identity():
    from document_ocr.synthesis.transport_auxiliary import imo_check_digit, sample_imo_number

    for source in ("9299625", "9125047"):
        recipe = {
            "owner": "imo",
            "imo": {"source": source, "maximumAttempts": 128, "leadingDigitWeights": {"9": 1}},
        }
        result = extra_value(recipe, {}, {}, "x", None, project_root=ROOT)
        assert result == extra_value(recipe, {}, {}, "x", None, project_root=ROOT)
        value, target = result
        assert value == target and value != source
        assert imo_check_digit(value[:6]) == int(value[-1])
    # The underlying sampler's strict exclusion contract is unchanged.
    with pytest.raises(ValueError, match="checksum-valid"):
        sample_imo_number(
            stream=DeterministicStream(7, "test", "imo"),
            excluded_values=["9299625"],
            used_values=set(),
            maximum_attempts=128,
            leading_digit_weights={"9": 1},
        )
    for source in (
        "IMO 9125047",
        "0125047",
        "912504",
        "\uff19\uff11\uff12\uff15\uff10\uff14\uff17",
    ):
        recipe["imo"]["source"] = source
        with pytest.raises(ValueError, match="ASCII"):
            extra_value(recipe, {}, {}, "x", None, project_root=ROOT)


def test_final_phone_range_projection_uses_printed_compact_owners():
    receipts = [
        {"owner": "phone@1:5", "after": "+971501233449"},
        {"owner": "phone@20:25", "after": "+971501233449"},
        {"owner": "extension@6:9", "after": "501239483"},
    ]
    assert literal_owner_value(receipts, ["phone", "extension"], "-") == "+971501233449-501239483"
    # A source-equal value is not in the old change-only edit log. Its explicitly
    # witnessed native owner must still contribute to a literal target.
    receipts.append({"owner": "ventilation@native", "after": "20 CBM/HR"})
    assert literal_owner_value(receipts, ["ventilation"], " ") == "20 CBM/HR"
    with pytest.raises(ValueError, match="ambiguous"):
        literal_owner_value(receipts, ["absent"], " ")
    receipts[1]["after"] = "+971500000000"
    with pytest.raises(ValueError, match="ambiguous"):
        literal_owner_value(receipts, ["phone"], " ")


def test_private_repeated_measure_resolves_source_equipment_not_row_order():
    candidate = {
        "scenario": {
            "cargo": {
                "containerInformation": [{"equipmentIdentifier": "B"}, {"equipmentIdentifier": "A"}]
            },
            "provenance": {
                "independentDraw": {
                    "allocations": [{"row": 1, "netKg": "12.5"}, {"row": 0, "netKg": "20"}]
                }
            },
        }
    }
    extra = {
        "owner": "row-net",
        "physical": {
            "sourceEquipmentIdentifier": "A",
            "field": "netKg",
            "unit": "kilogram",
            "minPlaces": 3,
            "grouped": False,
            "prefix": "",
            "suffix": " KGS",
        },
    }
    assert extra_value(extra, candidate, {}, "x", None, project_root=ROOT)[0] == "12.500 KGS"
    extra["physical"]["sourceEquipmentIdentifier"] = "ABSENT"
    with pytest.raises(ValueError, match="unique source"):
        extra_value(extra, candidate, {}, "x", None, project_root=ROOT)


def test_new_native_temperature_uses_sampled_setpoint_not_source_literal():
    extra = {
        "owner": "temperature",
        "targetNumber": {
            "path": "/containerInformation/0/temperatureSetpoint",
            "unit": "celsius",
            "minPlaces": 1,
        },
    }
    current = {
        "target": {
            "containerInformation": [{"temperatureSetpoint": {"value": 0.5, "unit": "celsius"}}]
        }
    }
    assert extra_value(extra, {}, current, "x", None, project_root=ROOT) == ("0.5", None)


def test_fragment_proof_cannot_match_ambiguous_or_missing_occurrence():
    pages = (PaddlePage(0, 100, 100, (PaddleRegion(0, "ABC ABC", (1, 1, 50, 10)),)),)
    for text in ("ABC", "ABSENT"):
        with pytest.raises(ValueError, match="not unique"):
            spans_for({"fragments": [[0, 0, text]]}, pages)


def test_split_person_name_is_rendered_once_only_with_exact_reviewed_members():
    edits = [
        {"key": "name", "byteStart": 10, "before": "Amr", "after": "Omar Yousuf"},
        {"key": "name", "byteStart": 14, "before": "Ali", "after": "Omar Yousuf"},
        {"key": "name", "byteStart": 90, "before": "Amr Ali", "after": "Omar Yousuf"},
    ]
    rule = {"members": [{k: e[k] for k in ("key", "byteStart", "before")} for e in edits[:2]]}
    assert coalesced_edits(edits, [rule]) == [edits[0], edits[2]]
    assert coalesced_edits(edits, []) == edits
    with pytest.raises(ValueError, match="multiply owned"):
        coalesced_edits(edits, [rule, rule])
    edits[1] = {**edits[1], "after": "Different Person"}
    with pytest.raises(ValueError, match="distinct sampled"):
        coalesced_edits(edits, [rule])
    with pytest.raises(ValueError, match="missing"):
        coalesced_edits(edits[1:], [rule])


def test_new_wording_cannot_be_loaded_implicitly_from_a_previous_campaign():
    page = PaddlePage(0, 100, 100, (PaddleRegion(0, "PRIVATE NAME", (1, 1, 80, 10)),))
    record = {"pages": [asdict(page)]}
    candidate = current = {"joinedRawText": "PRIVATE NAME"}
    args = (record, {}, candidate, current, {"extras": [{"wording": "name"}]}, None, "x", None)
    with pytest.raises(ValueError, match="explicit reviewed receipt"):
        render_one(*args, wording_receipt=None, project_root=ROOT)
    with pytest.raises(ValueError, match="provenance"):
        render_one(
            *args,
            wording_receipt={"status": "REVIEWED", "sourceFingerprint": "wrong-campaign"},
            project_root=ROOT,
        )
    args = (record, {}, candidate, current, {"extras": []}, None, "x", None)
    with pytest.raises(ValueError, match="without new linguistic owners"):
        render_one(*args, wording_receipt={}, project_root=ROOT)


def test_override_wording_requires_the_same_explicit_receipt_as_new_owners():
    from document_ocr.synthesis.paddle_render import requires_wording

    overlay = {"extras": [], "overrides": [{"valueRecipe": {"wording": "contact"}}]}
    assert requires_wording(overlay)
    assert not requires_wording({"extras": [], "overrides": [{"key": "contact"}]})
    assert requires_wording({"extras": [{"parts": [{"wording": "contact"}, {"literalText": "X"}]}]})
    assert requires_wording(
        {
            "extras": [],
            "overrides": [
                {"valueRecipe": {"parts": [{"wording": "contact"}, {"literalText": "X"}]}}
            ],
        }
    )
    assert requires_wording(
        {"extras": [], "literalProjections": [{"pieces": [{"wording": "contact"}]}]}
    )
    assert not requires_wording({"extras": [{"parts": [{"literalText": "wording"}]}]})
    page = PaddlePage(0, 100, 100, (PaddleRegion(0, "PRIVATE NAME", (1, 1, 80, 10)),))
    record = {"pages": [asdict(page)]}
    candidate = current = {"joinedRawText": "PRIVATE NAME"}
    with pytest.raises(ValueError, match="explicit reviewed receipt"):
        render_one(
            record,
            {},
            candidate,
            current,
            overlay,
            None,
            "x",
            None,
            wording_receipt=None,
            project_root=ROOT,
        )


def test_adjacent_linguistic_owners_have_an_explicit_presentation_separator():
    recipe = {"wording": "address", "separatorBefore": "\n", "targetCase": "upper"}
    assert extra_value(
        recipe, {}, {}, "x", None, wording={"address": "Building 7"}, project_root=ROOT
    ) == (
        "\nBuilding 7",
        "BUILDING 7",
    )
    for invalid in (None, "ADDRESS:"):
        with pytest.raises(ValueError, match="separator"):
            extra_value(
                {**recipe, "separatorBefore": invalid},
                {},
                {},
                "x",
                None,
                wording={"address": "Building 7"},
                project_root=ROOT,
            )


def test_mixed_packing_repetitions_use_whole_shipment_not_first_group():
    def group(n):
        return {
            "relation": "MIXED",
            "publicQuantity": 4 * n,
            "relatedQuantity": 51 * n,
            "relatedCategory": "PACKAGE_BOX_FIBREBOARD",
            "facts": {
                "mixedAccounting": {
                    "supportQuantity": n,
                    "packedQuantity": 48 * n,
                    "looseQuantity": 3 * n,
                    "totalRelatedQuantity": 51 * n,
                }
            },
        }

    ledger = {
        "relation": "DISJOINT_GROUPS",
        "publicQuantity": 120,
        "facts": {"declaredGroupLedgers": [group(10), group(16), group(4)]},
    }
    candidate = {"scenario": {"provenance": {"freshPrivatePacking": ledger}}}
    for field, expected in (
        ("supportQuantity", "30"),
        ("looseQuantity", "90"),
        ("totalRelatedQuantity", "1530"),
    ):
        extra = {"owner": "marks", "packingSummary": {"field": field}}
        assert extra_value(extra, candidate, {}, "x", None, project_root=ROOT) == (expected, None)
    ledger["facts"]["declaredGroupLedgers"][0]["relatedQuantity"] = 731
    with pytest.raises(ValueError, match="conservation"):
        extra_value(extra, candidate, {}, "x", None, project_root=ROOT)
    ledger["relation"] = "NESTED"
    with pytest.raises(ValueError, match="mixed ledgers"):
        extra_value(extra, candidate, {}, "x", None, project_root=ROOT)


def test_static_reviewed_caption_is_not_an_invented_target():
    assert extra_value(
        {"owner": "caption", "literalText": "EXPORTER TAX ID:"},
        {},
        {},
        "x",
        None,
        project_root=ROOT,
    ) == ("EXPORTER TAX ID:", None)
    with pytest.raises(ValueError, match="nonempty"):
        extra_value({"owner": "caption", "literalText": ""}, {}, {}, "x", None, project_root=ROOT)
