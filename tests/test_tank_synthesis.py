"""Installed tanks retain hardware, lot semantics and source field visibility."""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from test_synthesis_transport_capacity import _limits as limits

from document_ocr.synthesis.container_semantics import review_source_equipment_surface
from document_ocr.synthesis.independent_envelopes import package_envelope
from document_ocr.synthesis.independent_packages import (
    PackageChoiceRequest,
    equipment_choice_brief,
    package_choice_output_type,
)
from document_ocr.synthesis.independent_sampling import IndependentSampler, sample_equipment
from document_ocr.synthesis.tank_equipment import TANK_SIZES, TANK_TYPES
from document_ocr.synthesis.transport_capacity import classify_equipment, equipment_capacity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sampler():
    config = json.loads(
        (
            ROOT
            / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
            / "independent-flat-full-registry-20261021/working-config.json"
        ).read_text()
    )
    return IndependentSampler.from_config(ROOT, config["independent"]["registry"])


@pytest.mark.parametrize("kind", sorted(TANK_TYPES))
@pytest.mark.parametrize("size", TANK_SIZES)
@pytest.mark.parametrize("regime", ["ambient", "dg"])
def test_tanks_are_selected_and_capacity_is_never_dry_box(kind, size, regime):
    row = {"sizeCategory": size, "typeCategory": kind}
    assert sample_equipment(
        regime=regime,
        equipment_mode="uniform",
        seed=1,
        draw_key="tank",
        container_count=1,
        equipment_domain=[row],
    ) == [row]
    assert classify_equipment(row) in {"twenty_tank", "forty_tank"}
    capacity = equipment_capacity(row, limits())
    assert capacity.volume_m3 == (20 if size.startswith("TWENTY") else 40)
    brief = equipment_choice_brief([row])
    assert "tank_loading_lot" in brief and "not through dry-box doors" in brief
    with pytest.raises(ValueError, match="incompatible"):
        sample_equipment(
            regime="frozen",
            equipment_mode="uniform",
            seed=1,
            draw_key="tank",
            container_count=1,
            equipment_domain=[row],
        )


@pytest.mark.parametrize("code", ["22K0", "42K6", "22N0", "42N3"])
def test_detailed_tank_codes_never_receive_dry_box_capacity(code):
    assert classify_equipment({"typeCode": code}) in {"twenty_tank", "forty_tank"}


@pytest.mark.parametrize(
    "surface,kind",
    [
        ("20TANK", "PRESSURIZED_TANK"),
        ("20'isotank", "PRESSURIZED_TANK"),
        ("20 FT ISO TANKCONTAINER", "PRESSURIZED_TANK"),
        ("20'TK", "PRESSURIZED_TANK"),
        ("40'ISOTANK", "PRESSURIZED_TANK"),
        ("40'TK", "PRESSURIZED_TANK"),
        ("40' ISO TANK CONTAINER", "PRESSURIZED_TANK"),
        ("22KL", "PRESSURIZED_TANK"),
        ("42KL", "PRESSURIZED_TANK"),
        ("20NH", "DRY_HOPPER_TANK"),
        ("22NH", "DRY_HOPPER_TANK"),
        ("20 DRY HOPPER TANK", "DRY_HOPPER_TANK"),
        ("20 DRY REAR DISCHARGE TANK", "DRY_REAR_DISCHARGE_TANK"),
        ("42NN", "DRY_REAR_DISCHARGE_TANK"),
    ],
)
def test_aliases_have_unambiguous_size_type_not_pressure_values(surface, kind):
    value = review_source_equipment_surface(surface, temperature_present=False)
    assert value.type_category == kind and value.size_category in TANK_SIZES
    assert value.thermal_operation == "not_indicated"


def test_unsupported_tank_height_is_explicit_not_dry_fallback():
    row = {"sizeCategory": "FORTY_FIVE_FOOT_HIGH_CUBE", "typeCategory": "PRESSURIZED_TANK"}
    assert classify_equipment(row) == "unsupported_tank"
    with pytest.raises(ValueError, match="tank size/type"):
        equipment_capacity(row, limits())


@pytest.mark.parametrize("surface", ["20 TK PARTS", "ISOTANKER", "TANKCONTAINERIZATION"])
def test_observed_tank_boundaries_do_not_classify_cargo_or_longer_words(surface):
    result = review_source_equipment_surface(surface, temperature_present=False)
    assert result.type_category is None


def test_optional_lot_context_does_not_remove_ordinary_packages_or_loose_mixed_units():
    from document_ocr.synthesis.independent_relationships import RELATED_PACKAGE_TYPES

    assert "PACKAGE_PACKAGE" in RELATED_PACKAGE_TYPES
    assert "PACKAGE_LOT" in RELATED_PACKAGE_TYPES
    assert package_envelope("PACKAGE_PACKAGE").net_kg == (10, 30)
    assert package_envelope("PACKAGE_LOT").net_kg == (12, 40)
    request = PackageChoiceRequest(
        goods=[{"hs6": "150710", "description": "Soybean oil"}],
        regime="ambient",
        equipment="Installed tank",
    )
    for context in [None, "tank_loading_lot"]:
        output = package_choice_output_type(
            request.model_copy(update={"installedTank": context is not None})
        )
        result = output.model_validate(
            {
                "alternatives": [
                    {"packageType": "PACKAGE_PACKAGE", "declaredUnitContext": context}
                ],
                "goodsForm": "Oil",
                "candidateCompatible": None,
            }
        )
        assert result.alternatives[0].declaredUnitContext == context


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({}, "35ffb0b21ab6c71780fb4819570f7e3eef19413c787dfda878c9f949463068fc"),
        (
            {"requiredCategory": "PACKAGE_BALE"},
            "5064cf9bc62016d3f571d60826422eb02fc9d012aecac732c8592ec03fac5378",
        ),
        (
            {
                "allowedPackageTypes": ["PACKAGE_BALE", "PACKAGE_PALLET"],
                "requiredCategory": "PACKAGE_BALE",
            },
            "bfd107112ad0fc8a35e51995f70d92ed5ef0066fccb23e7be0b4da7faffe2e2c",
        ),
    ],
)
def test_ordinary_native_contract_stays_byte_identical_to_pre_tank_baseline(extra, expected):
    import hashlib

    from document_ocr.synthesis.independent_packages import (
        PACKAGE_CHOICE_SYSTEM,
        package_choice_prompt,
    )

    request = PackageChoiceRequest(
        goods=[{"hs6": "520100", "description": "Cotton"}],
        regime="ambient",
        equipment="Maritime dry container",
        **extra,
    )
    contract = {
        "request": request.model_dump(mode="json"),
        "schema": package_choice_output_type(request).model_json_schema(),
        "prompt": package_choice_prompt(request, hs_codes={"520100"}),
        "system": PACKAGE_CHOICE_SYSTEM,
    }
    assert hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() == expected


def test_installed_lot_draw_varies_mass_and_volume_but_never_renames_public_package(sampler):
    values = []
    for variant in range(1, 31):
        request = sampler.prepare_request(
            source_document_id="tank-presentation",
            variant=variant,
            regime="ambient",
            identity_count=1,
            seed=20261022,
        )
        request.update(selectedCategory="PACKAGE_PACKAGE", countedUnitContext="tank_loading_lot")
        row = {"sizeCategory": TANK_SIZES[variant % 2], "typeCategory": "PRESSURIZED_TANK"}
        draw = sampler.draw(
            request=request,
            seed=20261022,
            draw_key=str(variant),
            container_count=1,
            selected_category="PACKAGE_PACKAGE",
            counted_unit_context="tank_loading_lot",
            equipment_domain=[row],
            row_capacities=[{"payloadKg": "22000", "volumeM3": "20"}],
        )
        assert draw["internalCategory"] == "PACKAGE_PACKAGE"
        assert draw["package"]["packageQuantity"] == 1
        assert draw["allocations"][0]["packageQuantity"] == 1
        assert Decimal(draw["physical"]["unitGrossKg"]) <= 16000
        assert Decimal(draw["physical"]["unitVolumeM3"]) <= 20
        values.append((draw["physical"]["unitNetKg"], draw["physical"]["unitVolumeM3"]))
    assert len({v[0] for v in values}) > 20
    assert len({v[1] for v in values}) > 20
    request["countedUnitContext"] = None
    with pytest.raises(ValueError, match="explicit tank_loading_lot"):
        sampler.draw(
            request=request,
            seed=1,
            draw_key="wrong-context",
            container_count=1,
            selected_category="PACKAGE_PACKAGE",
            equipment_domain=[row],
        )


def test_mixed_tank_lengths_and_uniform_controls_share_one_hardware_family():
    domain = [{"sizeCategory": size, "typeCategory": "PRESSURIZED_TANK"} for size in TANK_SIZES]
    for mode in ["mixed", "uniform"]:
        equipment = sample_equipment(
            regime="ambient",
            equipment_mode=mode,
            seed=7,
            draw_key="two-tanks",
            container_count=2,
            equipment_domain=domain,
        )
        assert len({r["sizeCategory"] for r in equipment}) == (2 if mode == "mixed" else 1)
