"""Explicit thermal units/styles, with unchanged public-field presence."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
from test_curated_equipment_aliases import blueprint as equipment_blueprint
from test_curated_physical import PREFIX, fixture
from test_curated_scenarios import catalog, source_row

from document_ocr.synthesis.curated_equipment_aliases import (
    apply_equipment_aliases,
    preserve_non_operating_reefer,
)
from document_ocr.synthesis.curated_physical import (
    _complete_equipment_surface,
    _temperature_text,
    prepare_physical_render,
)
from document_ocr.synthesis.curated_scenarios import SourceCapabilities, _ventilation_rate
from document_ocr.synthesis.curated_thermal import (
    CFM_TO_CBM_PER_HOUR,
    UnrepresentableVentilationRate,
    render_ventilation,
    ventilation_meaning,
)
from document_ocr.synthesis.thermal_goods import classify_thermal_descriptions


@pytest.mark.parametrize(
    "source,rate,rendered",
    [
        ("VENTILLATION REQUIRED: 10 CBM PER HOUR", "0", "VENTILLATION REQUIRED: 0 CBM PER HOUR"),
        ("VENT.:20.0 CBM/H", "0", "VENT.:0 CBM/H"),
        ("VENTILATION: 15 M³/HR", "10", "VENTILATION: 10 M³/HR"),
        ("VENTILATION CLOSED", "0", "VENTILATION CLOSED"),
        ("FRESH AIR VENTS SHUT", "0", "FRESH AIR VENTS SHUT"),
        # Exact operational fragment printed by real e38c509e, outside this
        # candidate set; humidity and carrying temperature are separate facts.
        (
            "Ventilación: Cerrada - Humedad: 60% - Temperatura: 14°C",
            "0",
            "Ventilación: Cerrada - Humedad: 60% - Temperatura: 14°C",
        ),
    ],
)
def test_explicit_ventilation_styles_keep_units_and_unrelated_text(source, rate, rendered):
    assert ventilation_meaning(source) is not None
    assert render_ventilation(source, Decimal(rate)) == rendered


@pytest.mark.parametrize("quantity", ["0", "10", "20.0", "25.5"])
def test_cfm_is_explicit_flow_not_a_guessed_source_unit(quantity):
    text = f"VENTILATION: {quantity} CFM; HUMIDITY: 85%"
    expected = Decimal(quantity) * CFM_TO_CBM_PER_HOUR
    assert _ventilation_rate({"handlingInstructions": [text]}) == str(expected)
    assert Decimal(ventilation_meaning(text).rate_cbm_per_hour) == expected
    same = render_ventilation(text, expected)
    assert ventilation_meaning(same).rate_cbm_per_hour == expected
    assert same.endswith(" CFM; HUMIDITY: 85%")
    zero = format(
        Decimal(0).quantize(Decimal(1).scaleb(Decimal(quantity).as_tuple().exponent)), "f"
    )
    assert render_ventilation(text, Decimal(0)) == f"VENTILATION: {zero} CFM; HUMIDITY: 85%"


def test_cfm_sampling_preserves_source_grid_instead_of_expanding_recurring_decimals():
    source = "VENTILATION 20.0 CFM"
    assert render_ventilation(source, Decimal("33.9802159104")) == source
    assert render_ventilation(source, Decimal("16.9901079552")) == "VENTILATION 10.0 CFM"
    with pytest.raises(UnrepresentableVentilationRate, match="CFM precision"):
        render_ventilation(source, Decimal(20))


@pytest.mark.parametrize(
    "text", ["VENT 15", "VENTILATION 15%", "HUMIDITY 85%", "CARGO VOLUME 10 CBM/H"]
)
def test_unqualified_vent_openings_humidity_and_cargo_volume_are_not_airflow(text):
    assert ventilation_meaning(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "VENTILATION -1 CFM",
        "VENTILATION 10 CBM/H, 11 CBM/H",
        "VENTILATION CLOSED, 10 CBM/H",
        "VENTILATION NOT CLOSED",
        "VENTILATION OPEN/CLOSED",
    ],
)
def test_conflicting_or_negative_source_operations_do_not_fall_back(text):
    with pytest.raises(ValueError):
        ventilation_meaning(text)


def test_closed_presentation_cannot_silently_invent_positive_flow_or_units():
    with pytest.raises(ValueError, match="closed"):
        render_ventilation("VENTILATION CLOSED", Decimal(10))
    with pytest.raises(ValueError, match="unit"):
        render_ventilation("VENT 15", Decimal(10))
    with pytest.raises(ValueError, match="closed"):
        render_ventilation("VENTILATION CLOSED 0 CBM/H", Decimal(10))
    assert render_ventilation("VENT 10 CBM/H; DOOR CLOSED", Decimal(0)) == (
        "VENT 0 CBM/H; DOOR CLOSED"
    )
    assert (
        _ventilation_rate({"handlingInstructions": ["VENT 20.0 CBM/H", "VENTILATION 20 CBM/H"]})
        == "20.0"
    )


@pytest.mark.parametrize(
    "source,old,new,unit,expected",
    [
        ("PLUS1DEG'C", "1", "0", "celsius", "0DEG'C"),
        ("MINUS2DEG' C", "-2", "1", "celsius", "1DEG' C"),
        ("TEMP -4°F", "-4", "-0.4", "fahrenheit", "TEMP -0.4°F"),
        ("TEMP -18C", "-0.4", "32", "fahrenheit", "TEMP 0C"),
        ("TEMP -4°F", "-20", "0", "celsius", "TEMP 32°F"),
        ("TEMP:(-)18 DEG CEL", "-18", "-22.5", "celsius", "TEMP:(-)22.5 DEG CEL"),
        ("TEMP:-2'C", "-2", "1", "celsius", "TEMP:1'C"),
        ('TEMP -4"F', "-4", "-0.4", "fahrenheit", 'TEMP -0.4"F'),
    ],
)
def test_carrying_display_uses_each_explicit_temperature_unit(source, old, new, unit, expected):
    assert _temperature_text(source, Decimal(old), Decimal(new), unit=unit) == expected


def test_mixed_container_units_and_repeated_operational_surfaces_share_one_temperature():
    _, blueprint, scenario, support = fixture()
    old = deepcopy(blueprint.target)
    old["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = ["0°C / 32°F"]
    for i, container in enumerate(old["documentPatch"]["containerInformation"]):
        container["temperatureSetpoint"] = {
            "value": 0.0 if i == 0 else 32.0,
            "unit": "celsius" if i == 0 else "fahrenheit",
        }
    bindings = {
        "fahrenheit_value": {
            "target_paths": ["documentPatch.containers[1].temperatureSetpoint.value"],
            "value_kind": "temperature",
            "occurrences": [{"source_text": "32"}],
        },
        "carrying_instruction": {
            "target_paths": [PREFIX + ".handlingInstructions[0]"],
            "value_kind": "text",
            "occurrences": [{"source_text": "0°C / 32°F"}],
        },
    }
    blueprint = replace(
        blueprint,
        target=old,
        historical_bindings=bindings,
        regions=(*blueprint.regions, *(SimpleNamespace(key=k, curated_key=None) for k in bindings)),
    )
    target = deepcopy(old)
    target["documentPatch"]["containerInformation"][0]["temperatureSetpoint"]["value"] = -18.0
    target["documentPatch"]["containerInformation"][1]["temperatureSetpoint"]["value"] = -0.4
    result = prepare_physical_render(blueprint, scenario, target, support=support)
    assert result.target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "-18°C / -0.4°F"
    ]
    assert result.surface_values["fahrenheit_value"] == ["-0.4"]
    assert result.surface_values["carrying_instruction"] == ["-18°C / -0.4°F"]


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "SHIPPED UNDER REFRIGERATION AT -18 DEG CEL",
            "SHIPPED UNDER REFRIGERATION AT -19.5 DEG CEL",
        ),
        ("KEEP AT -18'C", "KEEP AT -19.5'C"),
        ("KEEP AT MINUS18DEG'C", "KEEP AT -19.5DEG'C"),
    ],
)
def test_direct_carrying_handling_label_and_surface_use_the_same_temperature_parser(
    source, expected
):
    _, blueprint, scenario, support = fixture()
    old = deepcopy(blueprint.target)
    old["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = [source]
    for container in old["documentPatch"]["containerInformation"]:
        container["temperatureSetpoint"] = {"value": -18, "unit": "celsius"}
    bindings = {
        **blueprint.historical_bindings,
        "carrying_temperature": {
            "target_paths": [
                "documentPatch.containerInformation[0].temperatureSetpoint",
                PREFIX + ".handlingInstructions[0]",
            ],
            "value_kind": "other_text",
            "occurrences": [{"source_text": "Temperature: -18.0 C"}, {"source_text": source}],
        },
    }
    blueprint = replace(
        blueprint,
        target=old,
        historical_bindings=bindings,
        regions=(*blueprint.regions, SimpleNamespace(key="carrying_temperature", curated_key=None)),
    )
    target = deepcopy(old)
    for container in target["documentPatch"]["containerInformation"]:
        container["temperatureSetpoint"]["value"] = -19.5
    result = prepare_physical_render(blueprint, scenario, target, support=support)
    assert result.target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        expected
    ]
    assert result.surface_values["carrying_temperature"] == ["Temperature: -19.5 C", expected]


def test_humidity_airflow_and_detached_recorded_temperature_are_not_carrying_values():
    _, blueprint, scenario, support = fixture()
    old = deepcopy(blueprint.target)
    old["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = [
        "HUMIDITY: 85%",
        "VENTILATION: 20 CBM/H",
    ]
    for container in old["documentPatch"]["containerInformation"]:
        container["temperatureSetpoint"] = {"value": -18, "unit": "celsius"}
    bindings = {
        **blueprint.historical_bindings,
        "recorded_gate_in": {
            "target_paths": [],
            "value_kind": "other_text",
            "occurrences": [{"source_text": "GATE IN TEMPERATURE RECORDED AT -6 DEG CEL"}],
        },
    }
    blueprint = replace(
        blueprint,
        target=old,
        historical_bindings=bindings,
        regions=(*blueprint.regions, SimpleNamespace(key="recorded_gate_in", curated_key=None)),
    )
    target = deepcopy(old)
    for container in target["documentPatch"]["containerInformation"]:
        container["temperatureSetpoint"]["value"] = -19.5
    scenario.provenance["ventilationCbmPerHour"] = "0"
    result = prepare_physical_render(blueprint, scenario, target, support=support)
    assert result.target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "HUMIDITY: 85%",
        "VENTILATION: 0 CBM/H",
    ]
    assert "recorded_gate_in" not in result.surface_values
    assert bindings["recorded_gate_in"]["occurrences"][0]["source_text"] == (
        "GATE IN TEMPERATURE RECORDED AT -6 DEG CEL"
    )


def test_fresh_produce_closed_ventilation_keeps_its_exact_observed_profile():
    row = source_row(code="070320")
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods["description"] = "FRESH GARLIC"
    goods["handlingInstructions"] = ["VENTILATION CLOSED"]
    for c in row["target"]["documentPatch"]["containerInformation"]:
        c.update(typeCategory="REFRIGERATED", temperatureSetpoint={"value": -3, "unit": "celsius"})
    support = catalog([row], phrases={"070320": "Fresh garlic"})
    result = support.sample(
        row,
        seed=9,
        variant=1,
        capabilities=SourceCapabilities(family="chilled", physical_profile="source_bundle"),
    )
    assert result.provenance["thermalCommodityContext"]["basis"] == "observed_produce_extension"
    assert result.provenance["ventilationCbmPerHour"] == "0"
    assert result.cargo["containerInformation"][0]["temperatureSetpoint"]["value"] == -3


def test_source_nor_survives_canonical_and_registry_alias_rendering_without_a_new_label():
    container = {"sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT", "typeCategory": "REFRIGERATED"}
    assert preserve_non_operating_reefer("40NOR", "40RF", container) == "40RF NOR"
    assert preserve_non_operating_reefer("40NOR", "40NOR", container) == "40NOR"
    assert _complete_equipment_surface(container, "40NOR").endswith("NOR")
    assert preserve_non_operating_reefer("40RH", "40RF", container) == "40RF"
    assert "temperatureSetpoint" not in container
    assert set(container) == {"sizeCategory", "typeCategory"}
    with pytest.raises(ValueError, match="non-operating"):
        preserve_non_operating_reefer(
            "40NOR", "40RF", {**container, "temperatureSetpoint": {"value": -18, "unit": "celsius"}}
        )


def private_produce_source():
    row = source_row(code="080940")
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods.pop("hsCodes")
    goods["description"] = "FRESH PLUMS"
    goods["handlingInstructions"] = ["VENT.:20.0 CBM/H"]
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_CARTON"
    for c in row["target"]["documentPatch"]["containerInformation"]:
        c.update(typeCategory="REFRIGERATED", temperatureSetpoint={"value": 5.5, "unit": "celsius"})
    return row


def test_private_source_classification_drives_produce_without_inventing_printed_hs():
    row = private_produce_source()
    original = deepcopy(row)
    support = catalog([row], phrases={"080940": "Fresh plums", "020130": "Fresh or chilled beef"})
    cap = SourceCapabilities(
        family="chilled", physical_profile="source_bundle", allowed_hs_codes=("080940",)
    )
    result = support.sample(row, seed=17, variant=1, capabilities=cap)
    assert row == original
    assert "hsCodes" not in result.cargo["goodsItemDetails"][0]
    assert support.observations[0].hs6 == ()
    context = result.provenance["thermalCommodityContext"]
    assert context["sampledHS6"] == ["080940"]
    assert context["temperatureCelsius"] == 5.5
    assert context["ventilationCbmPerHour"] == "20.0"
    assert context["privateClassificationContext"]["publicHSCodesAdded"] is False


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"allowed_hs_codes": ("999999",)}, "unregistered"),
        ({"allowed_hs_codes": ("520511",)}, "food/produce"),
        ({"allowed_hs_codes": (), "physical_profile": "source_bundle"}, "no complete"),
        ({"allowed_hs_codes": ("080940",), "identity_count": 2}, "one exact"),
    ],
)
def test_private_classification_needs_explicit_registered_single_source_food(changes, error):
    row = private_produce_source()
    support = catalog([row], phrases={"080940": "Fresh plums", "520511": "Cotton yarn"})
    cap = SourceCapabilities(
        **{"family": "chilled", "physical_profile": "source_bundle", **changes}
    )
    with pytest.raises(ValueError, match=error):
        support.sample(row, seed=17, variant=1, capabilities=cap)


def test_no_hs_frozen_source_uses_registry_profile_without_adding_a_code():
    row = private_produce_source()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods.update(description="FROZEN BOVINE LIVERS", handlingInstructions=["VENTILATION CLOSED"])
    for c in row["target"]["documentPatch"]["containerInformation"]:
        c["temperatureSetpoint"]["value"] = -22
    support = catalog([row], phrases={"020622": "Frozen bovine livers"})
    result = support.sample(
        row,
        seed=22,
        variant=1,
        capabilities=SourceCapabilities(
            family="frozen", physical_profile="source_bundle", allowed_hs_codes=("020622",)
        ),
    )
    assert "hsCodes" not in result.cargo["goodsItemDetails"][0]
    assert result.provenance["thermalCommodityContext"]["basis"] == "hs_registry_food_profile"
    assert -24 <= result.provenance["thermalCommodityContext"]["temperatureCelsius"] <= -18


def test_private_frozen_context_uses_the_registered_intermediate_branch_only():
    row = private_produce_source()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods.update(description="FROZEN BOVINE LIVERS", handlingInstructions=[])
    for container in row["target"]["documentPatch"]["containerInformation"]:
        container["temperatureSetpoint"]["value"] = -22
    support = catalog([row], phrases={"020622": "Frozen bovine livers"})
    # Exact HS2022 path: the broad heading is ambiguous, its intermediate
    # ancestor resolves frozen; the leaf itself contains no thermal adjective.
    support.thermal_codes["frozen"] = []
    support._hs_description_paths["020622"] = (
        "Edible offal of bovine animals, swine, sheep, goats, horses, asses, "
        "mules or hinnies, fresh, chilled or frozen",
        "Of bovine animals, frozen",
        "Livers",
    )
    result = support.sample(
        row,
        seed=22,
        variant=1,
        capabilities=SourceCapabilities(
            family="frozen", physical_profile="source_bundle", allowed_hs_codes=("020622",)
        ),
    )
    assert support.thermal_codes["frozen"] == []
    assert "hsCodes" not in result.cargo["goodsItemDetails"][0]
    context = result.provenance["thermalCommodityContext"]
    assert context["basis"] == "hs_registry_food_profile"
    assert context["sampledHS6"] == ["020622"]
    assert -24 <= context["temperatureCelsius"] <= -18


@pytest.mark.parametrize(
    "path,expected",
    [
        (("Livers", "Of bovine animals, frozen", "Fresh, chilled or frozen"), "FROZEN"),
        (("Not frozen livers", "Of bovine animals, frozen"), None),
        (("Fresh or frozen livers", "Frozen offal"), None),
        (("Fresh, chilled or frozen", "Frozen offal"), None),
        (("Livers", "Edible offal, fresh, chilled or frozen"), None),
        (("Fresh or chilled beef", "Frozen offal"), "CHILLED"),
    ],
)
def test_private_hierarchy_uses_existing_nearest_and_exclusion_rules(path, expected):
    assert classify_thermal_descriptions(path) == expected


def test_exact_source_bundle_pallet_food_does_not_require_unprinted_ventilation():
    row = private_produce_source()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods["handlingInstructions"] = []
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_PALLET"
    support = catalog([row], phrases={"080940": "Fresh plums"})
    result = support.sample(
        row,
        seed=17,
        variant=1,
        capabilities=SourceCapabilities(
            family="chilled", physical_profile="source_bundle", allowed_hs_codes=("080940",)
        ),
    )
    assert result.cargo["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["typeCategory"] == (
        "PACKAGE_PALLET"
    )
    context = result.provenance["thermalCommodityContext"]
    assert context["ventilationCbmPerHour"] is None
    assert result.cargo["goodsItemDetails"][0]["handlingInstructions"] == []


def test_unreviewed_pallet_donor_without_ventilation_is_still_excluded():
    row = private_produce_source()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods["hsCodes"] = ["080940"]
    goods["handlingInstructions"] = []
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_PALLET"
    support = catalog([row], phrases={"080940": "Fresh plums"})
    with pytest.raises(ValueError, match="no complete train-only cargo donors"):
        support.sample(
            row,
            seed=17,
            variant=1,
            capabilities=SourceCapabilities(family="chilled"),
        )


def test_nor_registry_alias_preserves_owned_counts_and_movement():
    bp = equipment_blueprint(("(2X40NOR FCL/FCL)",))
    for c in bp.target["documentPatch"]["containerInformation"][:2]:
        c.update(sizeCategory="FORTY_FOOT_STANDARD_HEIGHT", typeCategory="REFRIGERATED")
    target = deepcopy(bp.target)
    rendered, receipt = apply_equipment_aliases(
        bp, target, {"equipment": ["(2X40NOR FCL/FCL)"]}, "40RF"
    )
    assert rendered["equipment"] == ["(2X40RF NOR FCL/FCL)"]
    assert receipt[0]["referencedContainerIndices"] == [0, 1]
    assert target == bp.target


@pytest.mark.parametrize("source", ["40NOR", "40'NOR", "40' / NOR", "40'\nNOR"])
def test_nor_ordinary_equipment_receipt_accepts_reviewed_source_spellings(source):
    from document_ocr.synthesis.template_compiler.equipment_receipts import _semantic

    shape = _semantic(source)
    assert (shape.size, shape.kind) == ("FORTY_FOOT_STANDARD_HEIGHT", "REFRIGERATED")
    c = {"sizeCategory": shape.size, "typeCategory": shape.kind}
    assert _complete_equipment_surface(c, source) == source


def test_middle_fcl_loading_frame_does_not_mask_complete_reefer_equipment():
    from document_ocr.synthesis.template_compiler.equipment_receipts import _semantic

    shape = _semantic("40'HC FCL REEFER")
    assert (shape.size, shape.kind) == ("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")
    with pytest.raises(ValueError, match="unowned non-equipment"):
        _semantic("40'HC FCL PRIVATE COMPANY REEFER")


def test_nor_alias_cannot_hide_a_positive_carrying_temperature():
    bp = equipment_blueprint(("40RF",), indices=(0,))
    c = bp.target["documentPatch"]["containerInformation"][0]
    c.update(sizeCategory="FORTY_FOOT_STANDARD_HEIGHT", typeCategory="REFRIGERATED")
    target = deepcopy(bp.target)
    target["documentPatch"]["containerInformation"][0]["temperatureSetpoint"] = {
        "value": -18,
        "unit": "celsius",
    }
    with pytest.raises(ValueError, match="contradicts"):
        apply_equipment_aliases(bp, target, {"equipment": ["40RF"]}, "40NOR")
