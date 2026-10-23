"""Capacity-by-construction native profiles and the real independent draw path."""

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.independent_unit_profiles import (
    UnitProfileRequest,
    sample_unit_profile,
    source_unit_profile_precision,
    unit_profile_output_type,
    validate_profile,
)


def test_precision_is_owned_cargo_typography_not_equipment_tare():
    prefix = "documentPatch.goodsItemDetails[0]"
    mass = SimpleNamespace(
        key="mass",
        kind="mass",
        value="12000.12",
        occurrences=[SimpleNamespace(text="12,000.120"), SimpleNamespace(text="12000.12")],
    )
    tare = SimpleNamespace(
        key="tare", kind="mass", value="2000", occurrences=[SimpleNamespace(text="2000")]
    )
    volume = SimpleNamespace(
        key="cube", kind="volume", value="20", occurrences=[SimpleNamespace(text="20")]
    )
    bindings = [
        SimpleNamespace(path=prefix + ".grossWeight.value", expression="{mass}"),
        SimpleNamespace(path=prefix + ".volume.value", expression="{cube}"),
        SimpleNamespace(
            path="documentPatch.containerInformation[0].tare.value", expression="{tare}"
        ),
    ]
    blueprint = SimpleNamespace(
        target={
            "documentPatch": {
                "goodsItemDetails": [
                    {
                        "grossWeight": {"value": 12000.12, "unit": "kilogram"},
                        "volume": {"value": 20, "unit": "cubic_metre"},
                    }
                ]
            }
        },
        contract=SimpleNamespace(variables=[mass, volume, tare], targets=bindings),
        regions=[],
    )
    assert source_unit_profile_precision(blueprint) == {"massKg": "0.01", "volumeM3": "1"}
    blueprint.target["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["unit"] = "metric_tonne"
    assert source_unit_profile_precision(blueprint)["massKg"] == "1E+1"


def test_no_printed_measure_uses_declared_private_arithmetic_precision_only():
    blueprint = SimpleNamespace(
        target={}, contract=SimpleNamespace(variables=[], targets=[]), regions=[]
    )
    assert source_unit_profile_precision(blueprint) == {"massKg": "0.01", "volumeM3": "0.0001"}
    assert (
        source_unit_profile_precision(
            blueprint,
            [
                {
                    "role": "private_measurement",
                    "source_text": "4550",
                    "baseline": "4550",
                    "fact": "grossKg",
                }
            ],
        )["massKg"]
        == "1"
    )


def request(**updates):
    values = {
        "goods": [{"hs6": "842951", "description": "Complete wheel loader"}],
        "regime": "ambient",
        "selectedCategory": "PACKAGE_UNIT",
        "countedUnitContext": None,
        "equipment": [],
        "payloadKg": 28000.0,
        "volumeBudgetM3": 76.4,
        "volumeBasis": "explicit_test_capacity",
    }
    return UnitProfileRequest.model_validate({**values, **updates})


def answer():
    return {
        "unknown": False,
        "basis": "Three plausible counted units, not shipment totals.",
        "examples": [
            {"grossKg": 12000.0, "packagingTareFraction": 0.01, "occupiedVolumeM3": 36.0},
            {"grossKg": 16000.0, "packagingTareFraction": 0.02, "occupiedVolumeM3": 50.0},
            {"grossKg": 22000.0, "packagingTareFraction": 0.03, "occupiedVolumeM3": 65.0},
        ],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("grossKg", 28000.1),
        ("grossKg", 0.0),
        ("occupiedVolumeM3", 76.5),
        ("occupiedVolumeM3", 0.0),
        ("packagingTareFraction", -0.01),
        ("packagingTareFraction", 1.0),
        ("grossKg", float("nan")),
        ("occupiedVolumeM3", float("inf")),
    ],
)
def test_invalid_example_rejected_without_defaults(field, value):
    data = answer()
    data["examples"][2][field] = value
    with pytest.raises(ValidationError):
        validate_profile(request(), data)


def test_capacity_limits_reach_json_schema_not_just_prompt():
    schema = unit_profile_output_type(request()).model_json_schema()
    fields = schema["$defs"]["CapacityBoundUnitExample"]["properties"]
    assert fields["grossKg"]["maximum"] == 28000
    assert fields["occupiedVolumeM3"]["maximum"] == 76.4
    assert fields["packagingTareFraction"]["exclusiveMaximum"] == 1


def test_semantic_zero_is_enforced_by_schema():
    req = request(countedUnitContext="tank_loading_lot", selectedCategory="PACKAGE_PACKAGE")
    with pytest.raises(ValidationError):
        validate_profile(req, answer())
    data = answer()
    for example in data["examples"]:
        example["packagingTareFraction"] = 0.0
    assert not validate_profile(req, data).unknown
    fields = unit_profile_output_type(req).model_json_schema()["$defs"]
    assert fields["CapacityBoundUnitExample"]["properties"]["packagingTareFraction"]["maximum"] == 0


def test_liquid_loading_lot_in_a_dry_box_includes_receptacle_or_liner_mass():
    req = request(
        selectedCategory="PACKAGE_BULK_LIQUID",
        countedUnitContext="bulk_loading_lot",
        equipment=[{"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}],
    )
    data = answer()
    assert not validate_profile(req, data).unknown
    data["examples"][0]["packagingTareFraction"] = 0.0
    with pytest.raises(ValidationError):
        validate_profile(req, data)
    fields = unit_profile_output_type(req).model_json_schema()["$defs"]
    assert (
        fields["CapacityBoundUnitExample"]["properties"]["packagingTareFraction"][
            "exclusiveMinimum"
        ]
        == 0
    )


def test_source_rounding_does_not_silently_erase_positive_packing_mass():
    data = answer()
    for example in data["examples"]:
        example.update(grossKg=21.0, packagingTareFraction=0.001, occupiedVolumeM3=0.1)
    net, gross, _, receipt = sample_unit_profile(
        request(),
        validate_profile(request(), data),
        [(Decimal(28000), Decimal("76.4"))],
        DeterministicStream(7, "test", "packing-precision"),
        mass_quantum_kg=Decimal("1"),
        volume_quantum_m3=Decimal("0.001"),
    )
    assert gross > net and receipt["quantityMultiple"] == 100


def test_unknown_exact_examples_and_order():
    for modify in (
        lambda d: d.update(unknown=True),
        lambda d: d.update(examples=d["examples"][:2]),
    ):
        data = answer()
        modify(data)
        with pytest.raises(ValidationError):
            validate_profile(request(), data)
    unknown = validate_profile(request(), {"unknown": True, "examples": [], "basis": "No fit"})
    with pytest.raises(ValueError, match="known numeric profile"):
        sample_unit_profile(
            request(),
            unknown,
            [(Decimal(28000), Decimal("76.4"))],
            DeterministicStream(1, "test", "test"),
        )


def test_presentation_order_does_not_reject_coherent_examples():
    data = answer()
    data["examples"].reverse()
    profile = validate_profile(request(), data)
    assert profile.examples[0].grossKg == 22000
    _, _, _, receipt = sample_unit_profile(
        request(),
        profile,
        [(Decimal(28000), Decimal("76.4"))],
        DeterministicStream(1, "test", "order"),
    )
    assert receipt["exampleOrder"] == [2, 1, 0]


def test_interpolation_varies_all_facts_and_is_deterministic():
    req, profile = request(), validate_profile(request(), answer())
    capacities = [(Decimal(28000), Decimal("76.4"))]
    results = []
    for seed in range(500):
        args = (req, profile, capacities, DeterministicStream(seed, "profile-test", "test"))
        net, gross, cube, receipt = sample_unit_profile(*args)
        assert sample_unit_profile(*args) == (net, gross, cube, receipt)
        assert 0 < net <= gross <= 28000 and 0 < cube <= Decimal("76.4")
        assert gross - net == Decimal(receipt["unitPackagingTareKg"])
        results.append((net, gross - net, cube))
    assert all(len({row[i] for row in results}) > 400 for i in range(3))


def test_different_capacity_cannot_consume_profile():
    with pytest.raises(ValueError, match="bounds exceed selected hardware"):
        sample_unit_profile(
            request(),
            validate_profile(request(), answer()),
            [(Decimal(16000), Decimal(24))],
            DeterministicStream(1, "test", "test"),
        )


def test_real_sampler_uses_profile_not_fixed_unit_scale():
    from test_independent_sampling import ROOT, registry_config

    from document_ocr.synthesis.independent_sampling import IndependentSampler

    sampler = IndependentSampler.from_config(ROOT, registry_config())
    frozen = sampler.prepare_request(
        source_document_id="unit-test", variant=1, regime="ambient", identity_count=1, seed=7
    )
    frozen.update(selectedCategory="PACKAGE_UNIT", countedUnitContext=None)
    req = request(goods=frozen["goods"])
    frozen["nativeUnitProfileEvidence"] = {
        "request": req.model_dump(mode="json"),
        "output": answer(),
    }
    before = deepcopy(frozen)
    draws = [
        sampler.draw(
            request=frozen,
            seed=seed,
            draw_key="test",
            container_count=2,
            selected_category="PACKAGE_UNIT",
            quantity_multiple=2,
            row_capacities=[{"payloadKg": "28000", "volumeM3": "76.4"}] * 2,
            unit_profile_precision={"massKg": "1", "volumeM3": "0.001"},
        )
        for seed in range(20)
    ]
    assert frozen == before
    for draw in draws:
        assert Decimal(draw["physical"]["unitNetKg"]) > 10000
        assert Decimal(draw["physical"]["unitVolumeM3"]) > 24
        assert draw["package"]["packageQuantity"] % 2 == 0
        assert (
            sum(r["packageQuantity"] for r in draw["allocations"])
            == draw["package"]["packageQuantity"]
        )
        assert all(Decimal(r["grossKg"]) <= 28000 for r in draw["allocations"])
    with pytest.raises(ValueError, match="explicit current planning capacities"):
        sampler.draw(
            request=frozen,
            seed=1,
            draw_key="test",
            container_count=1,
            selected_category="PACKAGE_UNIT",
        )


def test_integer_cbm_source_preserves_small_units_via_count_groups():
    req = request(selectedCategory="PACKAGE_BALE")
    data = answer()
    for example, mass, cube in zip(
        data["examples"], (180.0, 250.0, 320.0), (0.45, 0.6, 0.75), strict=True
    ):
        example.update(grossKg=mass, occupiedVolumeM3=cube)
    profile = validate_profile(req, data)
    for seed in range(100):
        net, gross, cube, receipt = sample_unit_profile(
            req,
            profile,
            [(Decimal(28000), Decimal("76.4"))],
            DeterministicStream(seed, "precision", "bale"),
            mass_quantum_kg=Decimal(1),
            volume_quantum_m3=Decimal(1),
        )
        step = receipt["quantityMultiple"]
        assert step == 10 and 0 < cube < 1
        assert all(
            value * step == (value * step).to_integral_value() for value in (net, gross, cube)
        )


def test_infeasible_precision_group_fails_instead_of_zero_cube_or_old_scale():
    req = request()
    data = answer()
    for example in data["examples"]:
        example.update(grossKg=27000.0, occupiedVolumeM3=0.0001)
    with pytest.raises(ValueError, match="count group that cannot fit"):
        sample_unit_profile(
            req,
            validate_profile(req, data),
            [(Decimal(28000), Decimal("76.4"))],
            DeterministicStream(1, "precision", "impossible"),
            mass_quantum_kg=Decimal(1),
            volume_quantum_m3=Decimal(1),
        )
