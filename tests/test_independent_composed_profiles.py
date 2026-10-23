"""Complete-group physical authority, including mixed accounting and supports."""

from copy import deepcopy
from decimal import Decimal

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.independent_composed_profiles import (
    ComposedRequest,
    composed_output_type,
    composed_prompt,
    sample_composed_profile,
    scope_composition,
    topology_output_type,
    validate_composed,
)


def request(relation="ON_SUPPORT"):
    return ComposedRequest(
        goods=[{"hs6": "100630", "description": "Milled rice"}],
        regime="ambient",
        selectedCategory="PACKAGE_BAG" if relation == "ON_SUPPORT" else "PACKAGE_PACKAGE",
        countedUnitContext=None,
        equipment=[],
        payloadKg=28000.0,
        volumeBudgetM3=76.0,
        volumeBasis="explicit_test_capacity",
        privatePacking={
            "privatePackingRelation": relation,
            "relatedPackageType": "PACKAGE_CARTON"
            if relation != "ON_SUPPORT"
            else "PACKAGE_PALLET",
            "supportPackageType": "PACKAGE_PALLET",
        },
        topology={
            "ON_SUPPORT": {"countedPerSupport": 8},
            "SUPPORTS": {"innerPerSupport": 24},
            "CONTAINS": {"innerPerOuter": 12},
            "MIXED": {"supportsPerGroup": 1, "packedPerSupport": 24, "loosePerGroup": 2},
        }[relation],
    )


def answer(relation="ON_SUPPORT"):
    return {
        "unknown": False,
        "basis": "Complete planning group.",
        "examples": [
            {
                "grossKg": g,
                "packagingTareFraction": f,
                "occupiedVolumeM3": v,
                "supportShareOfPacking": 0.0 if relation == "CONTAINS" else share,
            }
            for g, f, v, share in [
                (300.0, 0.05, 1.0, 0.4),
                (450.0, 0.06, 1.3, 0.5),
                (600.0, 0.07, 1.6, 0.6),
            ]
        ],
    }


@pytest.mark.parametrize("relation", ["ON_SUPPORT", "SUPPORTS", "CONTAINS", "MIXED"])
def test_complete_group_arithmetic_varies_without_fixed_support_tare(relation):
    req, value = request(relation), answer(relation)
    profile = validate_composed(req, value)
    results = set()
    for seed in range(200):
        net, gross, cube, authority = sample_composed_profile(
            req,
            value,
            [(Decimal(28000), Decimal(76))],
            DeterministicStream(seed, "test", "group"),
            mass_quantum_kg=Decimal("0.01"),
            volume_quantum_m3=Decimal("0.0001"),
        )
        c = profile.countedPerGroup
        maximum = min(int(Decimal(28000) // (gross * c)), int(Decimal(76) // (cube * c)))
        groups = 1 + seed % maximum
        count = c * groups
        draw = {
            "provenance": {"numericEnvelope": authority},
            "internalCategory": req.selectedCategory,
            "physical": {"unitNetKg": str(net), "unitGrossKg": str(gross)},
        }
        facts = scope_composition(draw, req.privatePacking, count)
        assert facts["relatedQuantity"] == groups * profile.relatedPerGroup
        assert 0 < net * count <= gross * count <= 28000 and cube * count <= 76
        if relation == "MIXED":
            mixed = facts["mixed"]
            assert mixed["supportQuantity"] + mixed["looseQuantity"] == count
            assert mixed["packedQuantity"] + mixed["looseQuantity"] == mixed["totalRelatedQuantity"]
        if profile.supportPerGroup:
            support = facts["supportEnvelope"]
            assert 0 < Decimal(support["totalTareKg"]) <= (gross - net) * count
            results.add(support["totalTareKg"])
        else:
            assert facts["supportEnvelope"] is None
    if profile.supportPerGroup:
        assert len(results) > 150


@pytest.mark.parametrize("relation", ["ON_SUPPORT", "SUPPORTS", "CONTAINS", "MIXED"])
def test_invalid_topology_and_oversized_example_fail(relation):
    req = request(relation)
    bad_topology = req.topology.model_dump()
    for key in bad_topology:
        bad_topology[key] = 0
    with pytest.raises(ValueError):
        topology_output_type(relation).model_validate(
            {"topology": bad_topology, "basis": "Invalid zero count"}
        )
    bad = answer(relation)
    bad["examples"][0]["grossKg"] = 28001.0
    with pytest.raises(ValueError):
        validate_composed(req, bad)
    schema = composed_output_type(req).model_json_schema()
    assert (
        schema["$defs"]["CapacityBoundCompleteGroupExample"]["properties"]["grossKg"]["maximum"]
        == 28000
    )


def test_partial_group_cannot_be_allocated():
    req, value = request(), answer()
    net, gross, _, authority = sample_composed_profile(
        req,
        value,
        [(Decimal(28000), Decimal(76))],
        DeterministicStream(1, "test", "test"),
        mass_quantum_kg=Decimal("0.01"),
        volume_quantum_m3=Decimal("0.0001"),
    )
    draw = {
        "provenance": {"numericEnvelope": authority},
        "internalCategory": "PACKAGE_BAG",
        "physical": {"unitNetKg": str(net), "unitGrossKg": str(gross)},
    }
    before = deepcopy(draw)
    with pytest.raises(ValueError, match="whole frozen composition groups"):
        scope_composition(draw, req.privatePacking, 7)
    assert draw == before


def test_numeric_wire_cannot_choose_topology_or_confuse_capacity_with_unit_size():
    req = request("SUPPORTS")
    prompt = composed_prompt(req)
    assert "innerPerSupport" in prompt
    assert all(key not in prompt for key in ("payloadKg", "volumeBudgetM3", "equipment", "28000"))
    bad = answer("SUPPORTS")
    bad["topology"] = {"innerPerSupport": 5}
    with pytest.raises(ValueError):
        validate_composed(req, bad)
    assert validate_composed(req, answer("SUPPORTS")).relatedPerGroup == 24


def test_unknown_topology_cannot_be_used_or_silently_replaced():
    model = topology_output_type("ON_SUPPORT")
    assert model.model_validate({"topology": None, "basis": "Not plausible"}).unknown
    with pytest.raises(ValueError):
        model.model_validate({"basis": "Missing"})
    with pytest.raises(ValueError):
        model.model_validate(
            {"unknown": False, "topology": None, "basis": "A redundant flag is forbidden"}
        )


def test_coarse_source_precision_preserves_small_support_in_whole_groups():
    req, value = request("SUPPORTS"), answer("SUPPORTS")
    for example in value["examples"]:
        example["supportShareOfPacking"] = 0.1
    net, gross, cube, authority = sample_composed_profile(
        req,
        value,
        [(Decimal(28000), Decimal(76))],
        DeterministicStream(1, "precision", "group"),
        mass_quantum_kg=Decimal("10"),
        volume_quantum_m3=Decimal("0.0001"),
    )
    assert authority["quantityMultiple"] >= 10
    assert 0 < Decimal(authority["composition"]["supportTareKgPerGroup"]) <= gross - net
    assert gross * authority["quantityMultiple"] <= 28000
    assert cube * authority["quantityMultiple"] <= 76


def test_real_sampler_loads_only_whole_groups_without_posthoc_mass():
    from test_independent_sampling import ROOT, registry_config

    from document_ocr.synthesis.independent_sampling import IndependentSampler

    sampler = IndependentSampler.from_config(ROOT, registry_config())
    frozen = sampler.prepare_request(
        source_document_id="composed", variant=1, regime="ambient", identity_count=1, seed=1
    )
    req = request("MIXED").model_copy(update={"goods": frozen["goods"]})
    frozen.update(
        selectedCategory="PACKAGE_PACKAGE",
        countedUnitContext=None,
        nativeComposedProfileEvidence={
            "request": req.model_dump(mode="json"),
            "output": answer("MIXED"),
        },
    )
    for seed in range(100):
        draw = sampler.draw(
            request=frozen,
            seed=seed,
            draw_key="group-test",
            container_count=3,
            selected_category="PACKAGE_PACKAGE",
            row_capacities=[{"payloadKg": "28000", "volumeM3": "76"}] * 3,
            unit_profile_precision={"massKg": "0.01", "volumeM3": "0.0001"},
        )
        assert all(row["packageQuantity"] % 3 == 0 for row in draw["allocations"])
        assert (
            sum(row["packageQuantity"] for row in draw["allocations"])
            == draw["package"]["packageQuantity"]
        )
        facts = scope_composition(draw, req.privatePacking, draw["package"]["packageQuantity"])
        assert facts["mixed"]["declaredQuantity"] == draw["package"]["packageQuantity"]
        assert "preparedDiscreteArticleSupport" not in draw["provenance"]
