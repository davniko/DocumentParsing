"""Thermal ranges must preserve goods, source presence, units and compartment state."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.independent_thermal import (
    ThermalRequest,
    fresh_produce_eligible,
    make_request,
    output_type,
    product_context,
    sample_context,
    validate_profile,
)


def frozen_request(regime="chilled", respiration="non_respiring", ventilation="RATE"):
    row = {
        "sourceDocumentId": "thermal-test",
        "variant": 1,
        "regime": regime,
        "goods": [{"hs6": "020130", "description": "Boneless beef"}],
        "freshProduceCodes": [],
        "equipmentSlots": [
            {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "REFRIGERATED"}
        ],
    }
    request = ThermalRequest(
        goods=row["goods"],
        regime=regime,
        freshProduceCodes=[],
        equipment=row["equipmentSlots"],
        ventilation=ventilation,
    )
    row["nativeThermalEvidence"] = {
        "request": request.model_dump(mode="json"),
        "output": {
            "unknown": False,
            "basis": "Explicit test fixture, not a shipping recommendation.",
            "supportedRange": {
                "respiration": respiration,
                "temperatureMinC": -24 if regime == "frozen" else -1,
                "temperatureMaxC": -18 if regime == "frozen" else 2,
                "freshAirMinM3H": None if ventilation == "ABSENT" else 0,
                "freshAirMaxM3H": None if ventilation == "ABSENT" else 0,
            },
        },
    }
    return row


@pytest.mark.parametrize(
    "code,path,expected",
    [
        ("080810", ["Apples, fresh"], True),
        ("080450", ["Guavas and mangoes, fresh or dried"], True),
        ("080620", ["Dried", "Grapes, fresh or dried"], False),
        ("080111", ["Desiccated", "Coconuts, fresh or dried"], False),
        ("071010", ["Potatoes", "Vegetables, frozen"], False),
        ("070310", ["Onions and shallots", "Onions, fresh or chilled"], True),
        ("080000", ["Other than fresh fruit", "Fresh fruit"], False),
        ("030211", ["Trout, fresh or chilled"], False),
    ],
)
def test_explicit_produce_eligibility(code, path, expected):
    assert fresh_produce_eligible(code, path) is expected


@pytest.mark.parametrize(
    "instructions,mode",
    [
        ([], "ABSENT"),
        (["VENT OPENING 15"], "ABSENT"),
        (["VENTILATION CLOSED"], "CLOSED"),
        (["VENTILATION: 20 CBM/H"], "RATE"),
        (["VENTILATION: 0 CBM/H"], "RATE"),
    ],
)
def test_source_modes_do_not_invent_airflow_units(instructions, mode):
    row = frozen_request()
    native = make_request(row, {"goodsItemDetails": [{"handlingInstructions": instructions}]})
    assert native.ventilation == mode


@pytest.mark.parametrize(
    "regime,respiration,mode,field,value",
    [
        ("chilled", "non_respiring", "RATE", "freshAirMaxM3H", 20),
        ("frozen", "non_respiring", "RATE", "temperatureMaxC", -15),
        ("chilled", "respiring_produce", "ABSENT", "freshAirMaxM3H", 10),
        ("chilled", "respiring_produce", "CLOSED", "freshAirMaxM3H", 10),
        ("chilled", "respiring_produce", "RATE", "freshAirMaxM3H", 251),
        ("chilled", "non_respiring", "RATE", "temperatureMinC", float("nan")),
        ("chilled", "non_respiring", "RATE", "temperatureMaxC", -2),
        ("chilled", "non_respiring", "RATE", "temperatureMaxC", 1.25),
    ],
)
def test_schema_and_semantic_guards(regime, respiration, mode, field, value):
    row = frozen_request(regime, respiration, mode)
    row["nativeThermalEvidence"]["output"]["supportedRange"][field] = value
    with pytest.raises(ValidationError):
        validate_profile(row)


def test_variation_repeatability_and_no_fallback():
    row = frozen_request()
    values = {sample_context(row, seed)["temperatureCelsius"] for seed in range(100)}
    assert values == {"-1", "-0.5", "0", "0.5", "1", "1.5", "2"}
    assert sample_context(row, 42) == sample_context(deepcopy(row), 42)
    assert all(sample_context(row, i)["ventilationCbmPerHour"] == 0 for i in range(20))
    row.pop("nativeThermalEvidence")
    with pytest.raises(ValueError, match="explicitly prepared"):
        sample_context(row, 42)


def test_produce_variation_and_absence():
    row = frozen_request(respiration="respiring_produce")
    profile = row["nativeThermalEvidence"]["output"]["supportedRange"]
    profile.update(freshAirMinM3H=15, freshAirMaxM3H=30)
    assert len({sample_context(row, i)["ventilationCbmPerHour"] for i in range(100)}) == 16
    absent = frozen_request(ventilation="ABSENT")
    assert sample_context(absent, 1)["ventilationCbmPerHour"] is None


def test_unknown_and_tampered_context_are_not_sampling_authority():
    row = frozen_request()
    row["nativeThermalEvidence"]["output"].update(unknown=True, supportedRange=None)
    with pytest.raises(ValueError, match="UNKNOWN"):
        sample_context(row, 0)
    row = frozen_request()
    row["goods"][0]["description"] = "Unrelated commodity"
    with pytest.raises(ValueError, match="differs from sampled"):
        sample_context(row, 0)


def test_native_schema_contains_source_bounds():
    request = ThermalRequest.model_validate(
        frozen_request("frozen")["nativeThermalEvidence"]["request"]
    )
    schema = output_type(request).model_json_schema()["$defs"]["SourceBoundThermalRange"][
        "properties"
    ]
    assert schema["temperatureMaxC"]["maximum"] == -18
    assert schema["freshAirMaxM3H"]["const"] == 0


def test_respiration_is_native_not_inferred_from_hs_chapter():
    row = frozen_request(respiration="respiring_produce")
    goods = [{"hs6": "121291", "description": "Sugar beet, fresh or dried"}]
    row["goods"] = goods
    row["nativeThermalEvidence"]["request"]["goods"] = goods
    row["nativeThermalEvidence"]["output"]["supportedRange"].update(
        freshAirMinM3H=10, freshAirMaxM3H=20
    )
    assert sample_context(row, 2)["ventilationCbmPerHour"] >= 10
    context = product_context(row)
    assert context["respiration"] == "respiring_produce"
    assert "NON-LIVE" in context["instruction"]
