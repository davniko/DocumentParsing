"""Conservation, provenance and dependent text on the executable render path."""

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_physical import (
    PhysicalSupport,
    _distribute,
    _temperature_text,
    prepare_physical_render,
)
from document_ocr.synthesis.curated_templates import (
    compile_sampling_blueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.generators import iso6346_check_digit
from document_ocr.synthesis.template_compiler.equipment_tares import TareSupport

PREFIX = "documentPatch.goodsItemDetails[0]"


def fixture(*, public_mass=True):
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [{"packageQuantity": 10}],
                    **({"grossWeight": {"value": 30.0, "unit": "kilogram"}} if public_mass else {}),
                }
            ],
            "containerInformation": [
                {
                    "equipmentIdentifier": "MSCU12345"
                    + str(i)
                    + iso6346_check_digit("MSCU12345" + str(i)),
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
                for i in range(2)
            ],
        },
    }
    row = {
        "documentId": "source",
        "joinedRawText": "TOTAL 30\nROW A 10.00\nROW B 20.00\n",
        "target": target,
    }
    variables = [
        {
            "key": key,
            "kind": "mass",
            "value": str(value),
            "meaning": meaning,
            "required_literals": [],
            "occurrences": [{"text": text, "occurrence": 1, "presentation": "number"}],
        }
        for key, value, meaning, text in (
            ("total", 30, "Printed cargo mass: cargo:gross_weight_total", "30"),
            ("first", 10, "Printed cargo mass: cargo:row_gross_weight:0", "10.00"),
            ("second", 20, "Printed cargo mass: container:1:gross_weight_value", "20.00"),
        )
    ]
    contract = SourceContract(
        variables=variables,
        targets=[{"path": PREFIX + ".grossWeight.value", "expression": "{total}"}]
        if public_mass
        else [{"path": PREFIX + ".numberAndTypeOfPackages[0].packageQuantity", "expression": "10"}],
        fixed_context="Two explicitly owned cargo rows, unrelated source text stays immutable.",
    )
    blueprint = compile_sampling_blueprint(
        row,
        {
            "bindings": [],
            "document_id": row["documentId"],
            "source_sha256": digest(row["joinedRawText"].encode()),
        },
        contract,
    )
    donor = {
        "numberAndTypeOfPackages": [{"packageQuantity": 10}],
        "grossWeight": {"value": 37, "unit": "kilogram"},
    }
    support = PhysicalSupport({"donor": donor}, TareSupport({}, {}))
    scenario = SimpleNamespace(
        source_id="source",
        variant=1,
        provenance={
            "donorDocumentId": "donor",
            "seed": 19,
            "physicalRows": [
                {
                    "shareNumerator": 1,
                    "shareDenominator": 3,
                    "capacity": {"payloadKg": "100", "volumeM3": "10"},
                },
                {
                    "shareNumerator": 2,
                    "shareDenominator": 3,
                    "capacity": {"payloadKg": "100", "volumeM3": "10"},
                },
            ],
        },
    )
    return row, blueprint, scenario, support


def test_physical_rows_reconcile_printed_precision_and_replay_without_mutation():
    row, blueprint, scenario, support = fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] = 11.7
    snapshot = deepcopy(target)
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    assert target == snapshot
    assert plan.target["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] == 12
    assert plan.variable_values == {"total": "12", "first": "4", "second": "8"}
    output, _, receipt = render_sampling_blueprint(blueprint, plan.target, plan.variable_values)
    assert output == "TOTAL 12\nROW A 4.00\nROW B 8.00\n"
    assert receipt["unchangedBytesPreserved"]
    assert plan.receipt["rows"]["grossWeight"] == ["4", "8"]


def test_private_measures_are_donor_owned_and_do_not_invent_public_labels():
    row, blueprint, scenario, support = fixture(public_mass=False)
    plan = prepare_physical_render(blueprint, scenario, row["target"], support=support)
    assert "grossWeight" not in plan.target["documentPatch"]["goodsItemDetails"][0]
    assert plan.variable_values["total"] == "37"
    assert sum(Decimal(plan.variable_values[k]) for k in ("first", "second")) == 37
    assert plan.receipt["measures"]["grossWeight"]["method"] == "train_donor_private_measure"


def test_unknown_role_and_unowned_donor_fail_instead_of_guessing_from_equal_numbers():
    row, blueprint, scenario, support = fixture(public_mass=False)
    from dataclasses import replace

    variables = list(blueprint.contract.variables)
    variables[1] = variables[1].model_copy(update={"meaning": "Some number near a seal"})
    invalid = replace(
        blueprint, contract=blueprint.contract.model_copy(update={"variables": variables})
    )
    with pytest.raises(ValueError, match="no declared measurement role"):
        prepare_physical_render(invalid, scenario, row["target"], support=support)
    with pytest.raises(ValueError, match="absent from train-only"):
        prepare_physical_render(
            blueprint, scenario, row["target"], support=PhysicalSupport({}, support.tares)
        )


def test_capacity_is_checked_after_precision_reconciliation():
    row, blueprint, scenario, support = fixture()
    scenario.provenance["physicalRows"][1]["capacity"]["payloadKg"] = "19"
    with pytest.raises(ValueError, match="exceeds sampled equipment payload"):
        prepare_physical_render(blueprint, scenario, row["target"], support=support)


def test_temperature_signs_repetition_and_compact_wording_are_coherent():
    assert _temperature_text("PLUS 2C TILL PLUS 2C", Decimal(2), Decimal(-3)) == "-3C TILL -3C"
    assert _temperature_text("TEMP.2CE", Decimal(2), Decimal("5.5")) == "TEMP.5.5CE"
    assert _temperature_text("SHIPPED AT -22C", Decimal(-22), Decimal(-18)) == "SHIPPED AT -18C"
    with pytest.raises(ValueError, match="disagrees"):
        _temperature_text("PLUS 5C", Decimal(2), Decimal(-3))


def test_operational_targets_and_all_regulatory_class_surfaces_change_together():
    from dataclasses import replace

    row, blueprint, scenario, support = fixture()
    target = deepcopy(row["target"])
    goods = target["documentPatch"]["goodsItemDetails"][0]
    goods["handlingInstructions"] = ["VENTILATION: 20 CBM/HR", "PLUS 2C TILL PLUS 2C"]
    goods["dangerousGoods"] = [{"unNumber": "3342", "hazardCategory": "SOURCE_CATEGORY"}]
    for container in target["documentPatch"]["containerInformation"]:
        container["temperatureSetpoint"] = {"value": 2.0, "unit": "celsius"}
    bindings = {
        "ventilation_rate": {
            "target_paths": [],
            "value_kind": "decimal_measurement",
            "occurrences": [{"source_text": "20 CBM/HR"}],
        },
        "class": {
            "target_paths": [PREFIX + ".dangerousGoods[0].hazardCategory"],
            "value_kind": "category",
            "occurrences": [{"source_text": "4.2"}] * 5,
        },
        "un": {
            "target_paths": [PREFIX + ".dangerousGoods[0].unNumber"],
            "value_kind": "identifier",
            "occurrences": [{"source_text": "NO.3342"}],
        },
    }
    regions = (*blueprint.regions, *(SimpleNamespace(key=k, curated_key=None) for k in bindings))
    blueprint = replace(
        blueprint, target=deepcopy(target), historical_bindings=bindings, regions=regions
    )
    for container in target["documentPatch"]["containerInformation"]:
        container["temperatureSetpoint"]["value"] = -3.0
    scenario.provenance.update(
        ventilationCbmPerHour="10", dgPrintedFacts={"class": "8", "unNumber": "1824"}
    )
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    assert plan.target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        "VENTILATION: 10 CBM/HR",
        "-3C TILL -3C",
    ]
    assert plan.surface_values["ventilation_rate"] == ["10 CBM/HR"]
    assert plan.surface_values["class"] == ["8"] * 5
    assert plan.surface_values["un"] == ["NO.1824"]


def test_row_rounding_is_exact_for_large_and_fractional_totals():
    for total in (Decimal("23.457"), Decimal("99000.001"), Decimal("0.003")):
        result = _distribute(total, [Decimal(1)] * 3, Decimal("0.001"))
        assert sum(result) == total
        assert all(v > 0 and v == v.quantize(Decimal("0.001")) for v in result)
    with pytest.raises(ValueError, match="cannot exactly represent"):
        _distribute(Decimal("0.001"), [Decimal(1)] * 3, Decimal("0.001"))


def test_changed_equipment_draws_matching_observed_tare_and_keeps_operational_suffix():
    from dataclasses import replace

    row, blueprint, scenario, support = fixture()
    bindings = {
        "agent:container:0:tare_weight": {
            "target_paths": [],
            "group_key": "container:0",
            "value_kind": "decimal_measurement",
            "occurrences": [{"source_text": "2,220"}],
        },
        "agent:aggregate_container_receipt": {
            "target_paths": ["documentPatch.containers"],
            "value_kind": "equipment",
            "occurrences": [{"source_text": "2X20GP FCL CNTR(S)"}],
        },
    }
    blueprint = replace(
        blueprint,
        historical_bindings=bindings,
        regions=(*blueprint.regions, *(SimpleNamespace(key=k, curated_key=None) for k in bindings)),
    )
    target = deepcopy(row["target"])
    for container in target["documentPatch"]["containerInformation"]:
        container["sizeCategory"] = "FORTY_FOOT_HIGH_CUBE"
    support = PhysicalSupport(
        support.goods,
        TareSupport(
            {},
            {
                (Decimal(2220), "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"): ("old",),
                (Decimal(3700), "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"): ("train-witness",),
            },
        ),
    )
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    assert plan.surface_values["agent:container:0:tare_weight"] == ["3,700"]
    assert plan.surface_values["agent:aggregate_container_receipt"] == ["2X40HC FCL CNTR(S)"]
    assert plan.receipt["tares"][0]["observationSources"] == ["train-witness"]
