"""Exact component cardinality, capacity, and three-set/four-article arithmetic."""

import json
from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.independent_article_profiles import (
    article_output_type,
    article_prompt,
    component_roles,
    normalize_article_profile,
)
from document_ocr.synthesis.independent_unit_profiles import (
    UnitProfileRequest,
    sample_unit_profile,
    unit_profile_prompt,
)


def request():
    return UnitProfileRequest(
        goods=[{"hs6": "870380", "description": "Battery-powered vehicle"}],
        regime="dg",
        selectedCategory="PACKAGE_SET",
        countedUnitContext=None,
        equipment=[{"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}],
        payloadKg=28000.0,
        volumeBudgetM3=76.4,
        volumeBasis="hardware",
        physicalState="article",
        dangerousGoods={"recordId": "hmt", "unNumber": "3171"},
        articleGroups=[["passenger car", "motorbike"], ["passenger car"], ["passenger car"]],
    )


def answer():
    return {
        "unknown": False,
        "basis": "Each example includes exactly three cars and one motorbike.",
        "examples": [
            {
                key: {"grossKg": mass, "occupiedVolumeM3": cube}
                for key, mass, cube in (
                    ("passenger_car_1", 1700.0, 13.0),
                    ("motorbike_1", 150.0, 2.0),
                    ("passenger_car_2", 1800.0, 14.0),
                    ("passenger_car_3", 1900.0, 15.0),
                )
            }
            for _ in range(3)
        ],
    }


def test_four_explicit_articles_sum_before_three_declared_sets():
    req = request()
    value = answer()
    model = article_output_type(req)
    assert set(component_roles(req)) == {
        "passenger_car_1",
        "motorbike_1",
        "passenger_car_2",
        "passenger_car_3",
    }
    profile = normalize_article_profile(req, model.model_validate(value).model_dump(mode="json"))
    assert [e.grossKg for e in profile.examples] == [5550.0] * 3
    assert [e.occupiedVolumeM3 for e in profile.examples] == [44.0] * 3
    net, gross, cube, _ = sample_unit_profile(
        req,
        profile,
        [(Decimal(28000), Decimal("76.4"))],
        DeterministicStream(1, "article-test", "sample"),
        mass_quantum_kg=Decimal("0.001"),
        volume_quantum_m3=Decimal("0.001"),
    )
    assert net == gross and gross * 3 == 5550
    assert abs(cube * 3 - 44) < Decimal("0.003")


@pytest.mark.parametrize("key", ["motorbike_1", "passenger_car_3"])
def test_missing_or_extra_article_is_not_a_valid_manifest(key):
    value = answer()
    del value["examples"][0][key]
    with pytest.raises(ValidationError):
        normalize_article_profile(request(), value)
    value = answer()
    value["examples"][0]["extra_vehicle"] = {"grossKg": 1.0, "occupiedVolumeM3": 1.0}
    with pytest.raises(ValidationError):
        normalize_article_profile(request(), value)


@pytest.mark.parametrize("field,value", [("grossKg", 10000.0), ("occupiedVolumeM3", 25.0)])
def test_individually_bounded_components_cannot_overload_the_complete_manifest(field, value):
    output = answer()
    for component in output["examples"][0].values():
        component[field] = value
    with pytest.raises(ValueError, match="summed article"):
        normalize_article_profile(request(), output)


def test_unknown_does_not_acquire_source_or_family_defaults():
    unknown = {"unknown": True, "examples": [], "basis": "The supplied articles cannot fit."}
    assert normalize_article_profile(request(), unknown).unknown
    bad = deepcopy(unknown)
    bad["examples"] = answer()["examples"]
    with pytest.raises(ValidationError):
        normalize_article_profile(request(), bad)


def test_component_prompt_replays_after_sorted_persistence_and_has_no_aggregate_call_escape():
    req = request()
    persisted = UnitProfileRequest.model_validate(
        json.loads(json.dumps(req.model_dump(mode="json"), sort_keys=True))
    )
    assert article_prompt(req) == article_prompt(persisted)
    with pytest.raises(ValueError, match="per-component"):
        unit_profile_prompt(req)
