"""Conservation, provenance and dependent text on the executable render path."""

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_physical import (
    PhysicalSupport,
    _couple_mass_rows,
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
            "negotiability": None,
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


@pytest.mark.parametrize("printed_mass", ["grossWeight", "netWeight"])
def test_total_only_mass_tracks_rounded_counterpart_for_capacity_checks(printed_mass):
    from dataclasses import replace

    row, blueprint, scenario, support = fixture()
    other = "netWeight" if printed_mass == "grossWeight" else "grossWeight"
    target = deepcopy(row["target"])
    goods = target["documentPatch"]["goodsItemDetails"][0]
    goods[printed_mass] = {"value": 31, "unit": "kilogram"}
    goods[other] = {"value": 31, "unit": "kilogram"}
    variables = [v.model_copy(deep=True) for v in blueprint.contract.variables]
    for variable in variables:
        variable.meaning = (
            variable.meaning.replace("gross_weight", "net_weight")
            if printed_mass == "netWeight"
            else variable.meaning
        )
    variables.append(
        variables[0].model_copy(
            update={
                "key": "other_total",
                "meaning": "Printed cargo mass: cargo:"
                + ("net_weight_total" if other == "netWeight" else "gross_weight_total"),
            }
        )
    )
    contract = blueprint.contract.model_copy(update={"variables": variables, "targets": []})
    blueprint = replace(blueprint, contract=contract)
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    assert plan.variable_values["total"] == plan.variable_values["other_total"] == "31"
    assert plan.receipt["rows"][printed_mass] == ["10", "21"]
    assert other not in plan.receipt["rows"]  # Do not invent printed row facts.
    # Genuine capacity violations must still fail, including inferred gross rows.
    scenario.provenance["physicalRows"][1]["capacity"]["payloadKg"] = "20"
    with pytest.raises(ValueError, match="exceeds sampled equipment payload"):
        prepare_physical_render(blueprint, scenario, target, support=support)


def test_historical_allocation_measurements_follow_equipment_identity_not_row_order():
    from dataclasses import replace

    from document_ocr.synthesis.curated_physical import _owners

    row, blueprint, scenario, support = fixture()
    identifier = row["target"]["documentPatch"]["containerInformation"][1]["equipmentIdentifier"]
    foreign_key = "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber"
    bindings = {
        "identity": {
            "value_kind": "identifier",
            "occurrences": [{"source_text": identifier}],
            "realization": {
                "target_values": [{"target_path": foreign_key, "source_value": identifier}]
            },
        },
        "agent:allocation:0:0:gross_weight": {
            "value_kind": "decimal_measurement",
            "target_paths": [],
            "occurrences": [{"source_text": "20.00"}],
        },
    }
    blueprint = replace(
        blueprint,
        historical_bindings=bindings,
        regions=(
            *blueprint.regions,
            SimpleNamespace(key="agent:allocation:0:0:gross_weight", curated_key=None),
        ),
    )
    assert _owners(blueprint)[-1].row == 1
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] = 12
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    assert plan.surface_values["agent:allocation:0:0:gross_weight"] == ["8.00"]
    bindings["identity"]["realization"]["target_values"][0]["source_value"] = "UNKNOWN"
    with pytest.raises(ValueError, match="one proven equipment identity"):
        _owners(blueprint)


def test_private_aggregate_uses_nested_curated_dependencies_and_checks_source_equation():
    from dataclasses import replace

    row, blueprint, scenario, support = fixture(public_mass=False)
    key = "agent:cargo:total_measurement"
    binding = {
        "value_kind": "decimal_measurement",
        "target_paths": [],
        "derivation": "sum_gross_weight",
        "dependency_bindings": ["old:first", "old:second"],
        "occurrences": [{"source_text": "30.000"}],
    }
    blueprint = replace(
        blueprint,
        historical_bindings={key: binding},
        nested_bindings={"old:first": ("first",), "old:second": ("second",)},
        regions=(*blueprint.regions, SimpleNamespace(key=key, curated_key=None)),
    )
    plan = prepare_physical_render(blueprint, scenario, row["target"], support=support)
    assert plan.surface_values[key] == ["37.000"]
    binding["dependency_bindings"] = ["old:first", "old:first"]
    with pytest.raises(ValueError, match="distinct complete measure ownership"):
        prepare_physical_render(blueprint, scenario, row["target"], support=support)
    binding["dependency_bindings"] = ["missing"]
    with pytest.raises(ValueError, match="missing or cyclic"):
        prepare_physical_render(blueprint, scenario, row["target"], support=support)
    binding["dependency_bindings"] = ["old:first", "old:second"]
    binding["occurrences"] = [{"source_text": "31.000"}]
    with pytest.raises(ValueError):
        prepare_physical_render(blueprint, scenario, row["target"], support=support)


def test_private_measure_does_not_transfer_source_product_density_to_another_donor():
    row, blueprint, scenario, support = fixture(public_mass=False)
    donor = {"numberAndTypeOfPackages": [{"packageQuantity": 10}]}
    support = PhysicalSupport({"donor": donor, "source": donor}, support.tares)
    with pytest.raises(
        ValueError, match="different-product donor lacks required private grossWeight"
    ):
        prepare_physical_render(blueprint, scenario, row["target"], support=support)
    scenario.provenance["donorDocumentId"] = "source"
    plan = prepare_physical_render(blueprint, scenario, row["target"], support=support)
    assert plan.variable_values["total"] == "30"
    assert (
        plan.receipt["measures"]["grossWeight"]["method"]
        == "source_owned_private_per_package_measure"
    )
    assert "grossWeight" not in plan.target["documentPatch"]["goodsItemDetails"][0]


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
    variables = list(blueprint.contract.variables)
    variables[1] = variables[1].model_copy(update={"meaning": "Printed cargo gross_weight"})
    invalid = replace(
        blueprint, contract=blueprint.contract.model_copy(update={"variables": variables})
    )
    with pytest.raises(ValueError, match="conflicting source totals"):
        prepare_physical_render(invalid, scenario, row["target"], support=support)
    variables[1] = blueprint.contract.variables[1].model_copy(update={"value": "11"})
    invalid = replace(
        blueprint, contract=blueprint.contract.model_copy(update={"variables": variables})
    )
    with pytest.raises(ValueError, match="rows contradict shipment total"):
        prepare_physical_render(invalid, scenario, row["target"], support=support)


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
        "temperature_value": {
            "target_paths": [
                "documentPatch.containers[0].temperatureSetpoint.value",
                PREFIX + ".handlingInstructions[1]",
            ],
            "value_kind": "temperature",
            "occurrences": [{"source_text": "2"}],
        },
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
    assert plan.surface_values["temperature_value"] == ["-3.0"]
    assert plan.surface_values["class"] == ["8"] * 5
    assert plan.surface_values["un"] == ["NO.1824"]


def test_row_rounding_is_exact_for_large_and_fractional_totals():
    for total in (Decimal("23.457"), Decimal("99000.001"), Decimal("0.003")):
        result = _distribute(total, [Decimal(1)] * 3, Decimal("0.001"))
        assert sum(result) == total
        assert all(v > 0 and v == v.quantize(Decimal("0.001")) for v in result)
    with pytest.raises(ValueError, match="cannot exactly represent"):
        _distribute(Decimal("0.001"), [Decimal(1)] * 3, Decimal("0.001"))


@pytest.mark.parametrize("gross_quantum,net_quantum", [(".001", ".01"), (".01", ".001")])
@pytest.mark.parametrize("slack", ["0", ".01", "100"])
def test_net_gross_row_lattices_are_jointly_feasible_and_conserve_both_totals(
    gross_quantum, net_quantum, slack
):
    totals = {
        "grossWeight": Decimal("130989.50") + Decimal(slack),
        "netWeight": Decimal("130989.50"),
    }
    lattices = {"grossWeight": Decimal(gross_quantum), "netWeight": Decimal(net_quantum)}
    weights = [Decimal(1)] * 8
    rows = {k: _distribute(v, weights, lattices[k]) for k, v in totals.items()}
    receipts = {k: {"unit": "kilogram"} for k in totals}
    _couple_mass_rows(rows, totals, lattices, receipts, weights)
    for field, values in rows.items():
        assert sum(values) == totals[field]
        assert all(v > 0 and v % lattices[field] == 0 for v in values)
    assert all(n <= g for n, g in zip(rows["netWeight"], rows["grossWeight"], strict=True))
    snapshot = deepcopy(rows)
    _couple_mass_rows(rows, totals, lattices, receipts, weights)
    assert rows == snapshot


def test_coupled_mass_rows_compare_units_and_reject_incompatible_lattices():
    weights = [Decimal(1)] * 3
    totals = {"grossWeight": Decimal("1.00"), "netWeight": Decimal("1000.000")}
    lattices = {"grossWeight": Decimal(".01"), "netWeight": Decimal(".001")}
    receipts = {"grossWeight": {"unit": "metric_tonne"}, "netWeight": {"unit": "kilogram"}}
    rows = {k: _distribute(v, weights, lattices[k]) for k, v in totals.items()}
    _couple_mass_rows(rows, totals, lattices, receipts, weights)
    assert rows["netWeight"] == [Decimal(340), Decimal(330), Decimal(330)]
    assert sum(rows["grossWeight"]) == 1 and sum(rows["netWeight"]) == 1000
    receipts["grossWeight"]["unit"] = "pound"
    with pytest.raises(ValueError, match="not nested"):
        _couple_mass_rows(rows, totals, lattices, receipts, weights)


@pytest.mark.parametrize("kind", ["GENERAL_PURPOSE", "REFRIGERATED", "OPEN_TOP"])
def test_curated_equipment_surface_prints_complete_sampled_type(kind):
    from document_ocr.synthesis.container_semantics import review_source_equipment_surface
    from document_ocr.synthesis.curated_physical import _complete_equipment_surface
    from document_ocr.synthesis.template_compiler.equipment_receipts import project_receipt

    before = {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
    after = dict(before, typeCategory=kind)
    printed = _complete_equipment_surface(after, "40HC")
    observed = review_source_equipment_surface(printed, temperature_present=False)
    assert (observed.size_category, observed.type_category) == (after["sizeCategory"], kind)
    receipt = project_receipt(
        "1X40HC CONTAINER(S)",
        [before],
        [after],
        format_equipment=_complete_equipment_surface,
        number_words=lambda count: "ONE",
    )
    assert printed in receipt
    if kind == "GENERAL_PURPOSE":
        assert receipt == "1X40HC CONTAINER(S)"


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


def test_tare_dependencies_sum_exactly_and_do_not_guess_aggregate_ownership():
    from dataclasses import replace

    from document_ocr.synthesis.curated_physical import _tare_surfaces

    row, blueprint, scenario, support = fixture()

    def item(text, **extra):
        return dict(
            target_paths=[],
            value_kind="decimal_measurement",
            occurrences=[{"source_text": text}],
            **extra,
        )

    bindings = {
        "agent:tare:total": item(
            "5,000",
            derivation="sum_tare_weight",
            dependency_bindings=["agent:container:0:tare", "agent:container:1:tare"],
        ),
        "agent:container:0:tare": item("2000"),
        "agent:container:1:tare": item("3000"),
    }
    blueprint = replace(
        blueprint,
        historical_bindings=bindings,
        regions=tuple(SimpleNamespace(key=k, curated_key=None) for k in bindings),
    )
    declarations = {
        "tare_baselines": {
            "agent:tare:total": {
                "kg": "5000",
                "reason": "Exact source row sum",
                "sourceSha256": digest(blueprint.source.encode()),
            }
        }
    }
    blueprint = replace(blueprint, ownership_data=declarations)
    surfaces = {}
    _tare_surfaces(blueprint, row["target"], support, scenario, surfaces)
    assert surfaces["agent:tare:total"] == ["5,000"]
    bindings["agent:loaded_weight"] = item(
        "5030.00",
        derivation="sum_decimal_values",
        dependency_bindings=["agent:tare:total"],
        dependency_paths=[PREFIX + ".grossWeight.value"],
    )
    loaded = replace(
        blueprint,
        regions=(*blueprint.regions, SimpleNamespace(key="agent:loaded_weight", curated_key=None)),
    )
    new_target = deepcopy(row["target"])
    new_target["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] = 42
    _tare_surfaces(loaded, new_target, support, scenario, surfaces)
    assert surfaces["agent:loaded_weight"] == ["5042.00"]
    invalid = deepcopy(bindings)
    invalid["agent:tare:total"].pop("dependency_bindings")
    with pytest.raises(ValueError, match="lacks exact equipment ownership"):
        _tare_surfaces(
            replace(blueprint, historical_bindings=invalid), row["target"], support, scenario, {}
        )
    invalid = deepcopy(bindings)
    invalid["agent:container:1:tare"]["occurrences"][0]["source_text"] = "3001"
    with pytest.raises(ValueError, match="exact source row sum"):
        _tare_surfaces(
            replace(blueprint, historical_bindings=invalid), row["target"], support, scenario, {}
        )


def test_single_container_tare_and_hash_bound_ambiguous_typography():
    from dataclasses import replace

    from document_ocr.synthesis.curated_physical import _tare_surfaces

    row, blueprint, scenario, support = fixture()
    target = deepcopy(row["target"])
    target["documentPatch"]["containerInformation"] = target["documentPatch"][
        "containerInformation"
    ][:1]
    bindings = {
        "agent:cargo:tare": dict(
            target_paths=[],
            value_kind="decimal_measurement",
            occurrences=[{"source_text": "3.840"}],
        )
    }
    blueprint = replace(
        blueprint,
        target=target,
        historical_bindings=bindings,
        regions=(SimpleNamespace(key="agent:cargo:tare", curated_key=None),),
    )
    with pytest.raises(ValueError, match="ambiguous source values"):
        _tare_surfaces(blueprint, target, support, scenario, {})
    declaration = {
        "kg": "3840",
        "reason": "Reviewed dot-grouped source convention",
        "sourceSha256": digest(blueprint.source.encode()),
    }
    blueprint = replace(
        blueprint, ownership_data={"tare_baselines": {"agent:cargo:tare": declaration}}
    )
    surfaces = {}
    _, receipt = _tare_surfaces(blueprint, target, support, scenario, surfaces)
    assert surfaces == {"agent:cargo:tare": ["3.840"]}
    assert receipt[0]["owner"] == 0 and receipt[0]["kg"] == "3840"
    declaration["sourceSha256"] = "0" * 64
    with pytest.raises(ValueError, match="current source hash"):
        _tare_surfaces(blueprint, target, support, scenario, {})


def test_explicit_aggregate_only_tare_uses_observed_new_equipment_not_an_invented_split():
    from dataclasses import replace

    from document_ocr.synthesis.curated_physical import _tare_surfaces

    row, blueprint, scenario, support = fixture()
    key = "agent:tare:summary"
    blueprint = replace(
        blueprint,
        historical_bindings={
            key: {
                "target_paths": [],
                "value_kind": "decimal_measurement",
                "occurrences": [{"source_text": "5000"}],
            }
        },
        regions=(SimpleNamespace(key=key, curated_key=None),),
        ownership_data={"tare_owners": {key: [0, 1]}},
    )
    target = deepcopy(row["target"])
    for container in target["documentPatch"]["containerInformation"]:
        container["sizeCategory"] = "FORTY_FOOT_HIGH_CUBE"
    support = PhysicalSupport(
        support.goods,
        TareSupport(
            {},
            {
                (Decimal(3700), "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"): ("train-source",),
            },
        ),
    )
    surfaces = {}
    _, receipts = _tare_surfaces(blueprint, target, support, scenario, surfaces)
    assert surfaces == {key: ["7400"]}
    assert receipts[0]["owners"] == [0, 1]
    assert receipts[0]["observationSources"] == ["train-source"]
    assert len(receipts) == 1  # No original per-container tare was inferred.


def test_one_missing_tare_is_exact_residual_not_a_fabricated_printed_row():
    from dataclasses import replace

    from document_ocr.synthesis.curated_physical import _tare_surfaces

    row, blueprint, scenario, support = fixture()
    bindings = {
        key: dict(
            target_paths=[], value_kind="decimal_measurement", occurrences=[{"source_text": text}]
        )
        for key, text in (("agent:container:0:tare", "2230"), ("agent:tare:total", "4460"))
    }
    blueprint = replace(
        blueprint,
        historical_bindings=bindings,
        regions=tuple(SimpleNamespace(key=k, curated_key=None) for k in bindings),
        ownership_data={"tare_owners": {"agent:tare:total": [0, 1]}},
    )
    surfaces = {}
    _, receipts = _tare_surfaces(blueprint, row["target"], support, scenario, surfaces)
    assert surfaces == {k: [v["occurrences"][0]["source_text"]] for k, v in bindings.items()}
    assert next(r for r in receipts if r.get("owner") == 1)["derivedSourceResidualKg"] == "2230"
    assert set(surfaces) == set(bindings)  # Derived private fact is not a new OCR claim.
    target = deepcopy(row["target"])
    target["documentPatch"]["containerInformation"][1]["sizeCategory"] = "FORTY_FOOT_HIGH_CUBE"
    support = PhysicalSupport(
        support.goods,
        TareSupport({}, {(Decimal(3700), "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"): ("donor",)}),
    )
    _tare_surfaces(blueprint, target, support, scenario, surfaces)
    assert surfaces["agent:tare:total"] == ["5930"]
    invalid = deepcopy(bindings)
    invalid["agent:tare:total"]["occurrences"][0]["source_text"] = "2230"
    with pytest.raises(ValueError, match="non-positive source row residual"):
        _tare_surfaces(
            replace(blueprint, historical_bindings=invalid), target, support, scenario, {}
        )


def test_partial_equipment_keeps_source_wording_only_when_categories_unchanged():
    from dataclasses import replace

    from document_ocr.synthesis.curated_physical import _equipment

    row, blueprint, scenario, support = fixture()
    target = deepcopy(row["target"])
    container = target["documentPatch"]["containerInformation"][0]
    container.pop("sizeCategory")
    container["typeCategory"] = "REEFER"
    key = "anchor:documentPatch.containers[0].typeDescription"
    blueprint = replace(
        blueprint,
        target=deepcopy(target),
        historical_bindings={
            key: dict(
                target_paths=["documentPatch.containers[0].typeDescription"],
                value_kind="equipment",
                occurrences=[{"source_text": "REFRIGERATED CONTAINER"}],
            )
        },
        regions=(SimpleNamespace(key=key, curated_key=None),),
    )
    surfaces = {}
    _equipment(blueprint, target, support, scenario, surfaces, {})
    assert surfaces[key] == ["REFRIGERATED CONTAINER"]
    container["typeCategory"] = "GENERAL_PURPOSE"
    with pytest.raises(ValueError, match="partial equipment wording"):
        _equipment(blueprint, target, support, scenario, {}, {})


@pytest.mark.parametrize("aggregate", [True, False])
def test_reviewed_equipment_ownership_covers_current_categories_on_exact_source_regions(aggregate):
    from document_ocr.synthesis.curated_ownership import build_owned_blueprint

    row, old_blueprint, scenario, support = fixture()
    source_text = "2X20GP" if aggregate else "20GP"
    row["joinedRawText"] += source_text + "\n"
    key = "agent:equipment_receipt_summary" if aggregate else "agent:container_equipment_type"
    indices = (0, 1) if aggregate else (0,)
    paths = [
        f"documentPatch.containerInformation[{i}].{field}"
        for i in indices
        for field in ("sizeCategory", "typeCategory")
    ]
    start = row["joinedRawText"].encode().index(source_text.encode())
    historical = {
        "document_id": row["documentId"],
        "source_sha256": digest(row["joinedRawText"].encode()),
        "bindings": [
            {
                "logical_key": key,
                "value_kind": "equipment",
                "target_paths": [],
                "occurrences": [
                    {
                        "source_text": source_text,
                        "byte_start": start,
                        "byte_end": start + len(source_text),
                    }
                ],
            }
        ],
    }
    declarations = {"bindings": {key: {"paths": paths, "kind": "equipment"}}}
    blueprint = build_owned_blueprint(row, historical, old_blueprint.contract, declarations)
    target = deepcopy(row["target"])
    for i in indices:
        target["documentPatch"]["containerInformation"][i]["sizeCategory"] = "FORTY_FOOT_HIGH_CUBE"
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    output, rendered_target, _ = render_sampling_blueprint(
        blueprint, plan.target, plan.variable_values, surface_values=plan.surface_values
    )
    assert output.endswith(("2X40HC" if aggregate else "40HC") + "\n")
    assert rendered_target == plan.target


def test_unit_owners_and_non_numeric_temperature_references_are_not_numbers():
    from dataclasses import replace

    from document_ocr.synthesis.curated_physical import _operational, _owners

    row, blueprint, scenario, _support = fixture()
    bindings = {
        "anchor:volume.unit": {
            "target_paths": [PREFIX + ".volume.unit"],
            "value_kind": "decimal_measurement",
            "occurrences": [{"source_text": "CBM"}],
        },
        "agent:container_gross_weight_unit": {
            "target_paths": [],
            "value_kind": "decimal_measurement",
            "occurrences": [{"source_text": "KG"}],
        },
    }
    blueprint = replace(
        blueprint,
        historical_bindings=bindings,
        regions=tuple(SimpleNamespace(key=k, curated_key=None) for k in bindings),
    )
    assert [o.key for o in _owners(blueprint)] == ["total", "first", "second"]
    source = deepcopy(row["target"])
    reference = "GOODS SHIPPED AT CARRIAGE TEMPERATURE AS PER ABOVE"
    source["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = [
        reference,
        "SET AT 5C",
    ]
    for c in source["documentPatch"]["containerInformation"]:
        c["temperatureSetpoint"] = {"value": 5, "unit": "celsius"}
    target = deepcopy(source)
    for c in target["documentPatch"]["containerInformation"]:
        c["temperatureSetpoint"]["value"] = 2
    blueprint = replace(blueprint, target=source, historical_bindings={}, regions=())
    _operational(blueprint, scenario, target, {}, {})
    assert target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == [
        reference,
        "SET AT 2C",
    ]
    source["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = [
        "TEMPERATURE UNKNOWN",
        "SET AT 5C",
    ]
    target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = [
        "TEMPERATURE UNKNOWN",
        "SET AT 5C",
    ]
    with pytest.raises(ValueError, match="explicit signed temperature"):
        _operational(replace(blueprint, target=source), scenario, target, {}, {})


def test_physical_stage_preserves_planned_route_and_origin_handling_changes():
    from dataclasses import replace

    row, blueprint, scenario, support = fixture()
    original = ["FREE IN EGYPT", "GOODS OF ORIGIN CHINA"]
    planned = ["FREE IN TURKEY", "GOODS OF ORIGIN INDIA"]
    row["target"]["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = original
    blueprint = replace(blueprint, target=deepcopy(row["target"]))
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] = planned
    plan = prepare_physical_render(blueprint, scenario, target, support=support)
    assert plan.target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == planned
    assert (
        blueprint.target["documentPatch"]["goodsItemDetails"][0]["handlingInstructions"] == original
    )
