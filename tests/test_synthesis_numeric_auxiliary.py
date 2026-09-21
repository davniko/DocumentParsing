from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.numeric_auxiliary import (
    NumericContract,
    prepare,
    render_prepared,
    validate_contract,
)


def binding(surface="16937.050 KGS"):
    return NS(
        logical_key="container_weight",
        target_paths=(),
        value_kind="decimal_measurement",
        occurrences=(NS(slot_id="slot_1", source_text=surface),),
    )


def test_package_counts_are_numeric_even_when_compiler_calls_them_stable_vocabulary():
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import numeric_bindings

    b = binding("5 BOXES")
    b.value_kind = "package"
    b.target_paths = ()
    b.derivation = None
    b.occurrences[0].render_policy = "categorical_surface"
    template = NS(
        bindings=[b],
        coherence_constraints=[],
        auxiliary_semantic_plan=NS(
            dispositions=[NS(logical_key=b.logical_key, disposition="stable_vocabulary")]
        ),
    )
    assert numeric_bindings(template) == (b,)
    with pytest.raises(ValueError, match="cargo quantity contract"):
        validate_contract(
            b, contract(mode="source_fixed", role="operational", source_value="5"), {}
        )
    template.coherence_constraints = [
        NS(kind="aggregate_inclusive_range_cardinality", member_logical_keys=(b.logical_key,))
    ]
    assert numeric_bindings(template) == ()


def contract(
    mode="source_scaled",
    role="cargo_mass",
    source_value="16937.05",
    target_paths=None,
    multiplier="1",
    divisor=1,
):
    return NumericContract(
        mode=mode,
        role=role,
        source_value=source_value,
        target_paths=target_paths or [],
        multiplier=multiplier,
        divisor=divisor,
        reason="Printed net shipment mass in the container row",
    )


def test_source_only_mass_scales_with_shipment_not_random_digits():
    b = binding()
    c = contract()
    row = prepare((b,), {b.logical_key: c}, source_target={}, target={}, scale=Decimal("0.8"))[
        b.logical_key
    ]
    assert row.value == "13549.640"
    assert render_prepared((b,), {b.logical_key: row}, source_target={}, target={}) == {
        b.logical_key: {"slot_1": "13549.640 KGS"}
    }


def test_source_only_total_is_sum_of_rounded_rows_and_cannot_scale_independently():
    members = [binding("28 PALLETS"), binding("28 PALLETS"), binding("56 PALLETS")]
    for i, member in enumerate(members):
        member.logical_key = str(i)
        member.value_kind = "package"
    contracts = {
        b.logical_key: contract(mode="source_scaled", role="cargo_quantity", source_value=str(v))
        for b, v in zip(members, (28, 28, 56), strict=True)
    }
    with pytest.raises(ValueError, match="explicit binding_sum"):
        prepare(members, contracts, source_target={}, target={}, scale=Decimal("0.7"))
    contracts["2"] = contracts["2"].model_copy(
        update={"mode": "binding_sum", "dependency_bindings": ["0", "1"]}
    )
    prepared = prepare(members, contracts, source_target={}, target={}, scale=Decimal("0.7"))
    assert [prepared[str(i)].value for i in range(3)] == ["20", "20", "40"]
    assert (
        render_prepared(members, prepared, source_target={}, target={})["2"]["slot_1"]
        == "40 PALLETS"
    )
    contracts["2"] = contracts["2"].model_copy(update={"source_value": "55"})
    members[2].occurrences[0].source_text = "55 PALLETS"
    with pytest.raises(ValueError, match="source arithmetic"):
        prepare(members, contracts, source_target={}, target={}, scale=Decimal("0.7"))


def test_source_only_sum_rejects_cycles_missing_dependencies_and_mixed_roles():
    members = [binding("1"), binding("1")]
    for i, b in enumerate(members):
        b.logical_key = str(i)
        b.value_kind = "package"
    contracts = {
        str(i): contract(role="cargo_quantity", source_value="1").model_copy(
            update={"mode": "binding_sum", "dependency_bindings": [str(1 - i)]}
        )
        for i in range(2)
    }
    with pytest.raises(ValueError, match="cycle"):
        prepare(members, contracts, source_target={}, target={}, scale=Decimal(1))
    contracts["0"] = contracts["0"].model_copy(update={"dependency_bindings": ["missing"]})
    with pytest.raises(ValueError, match="unknown binding"):
        prepare(members, contracts, source_target={}, target={}, scale=Decimal(1))
    contracts["0"] = contract(role="cargo_mass", source_value="1")
    members[0].value_kind = "decimal_measurement"
    with pytest.raises(ValueError, match="incompatible quantity roles"):
        prepare(members, contracts, source_target={}, target={}, scale=Decimal(1))


def test_tare_is_explicitly_fixed_and_never_scaled_as_cargo():
    b = binding("3840 KGS")
    c = contract(mode="source_fixed", role="tare", source_value="3840")
    row = prepare((b,), {b.logical_key: c}, source_target={}, target={}, scale=Decimal("0.8"))[
        b.logical_key
    ]
    assert row.value == "3840"
    with pytest.raises(ValueError, match="shipment totals"):
        validate_contract(b, contract(mode="source_fixed", source_value="3840"), {})


def test_target_sum_source_proof_and_live_target_arithmetic():
    b = binding("120.000 KGS")
    c = contract(
        mode="target_sum",
        source_value="120",
        target_paths=["documentPatch.gross", "documentPatch.tare"],
    )
    source = {"documentPatch": {"gross": 100, "tare": 20}}
    target = {"documentPatch": {"gross": 70, "tare": 20}}
    row = prepare(
        (b,), {b.logical_key: c}, source_target=source, target=target, scale=Decimal("0.7")
    )[b.logical_key]
    assert row.value == "90"
    assert render_prepared((b,), {b.logical_key: row}, source_target=source, target=target) == {
        b.logical_key: {"slot_1": "90.000 KGS"}
    }
    target["documentPatch"]["gross"] = 80
    with pytest.raises(ValueError, match="differs from its dependency"):
        render_prepared((b,), {b.logical_key: row}, source_target=source, target=target)


def test_false_source_dependency_and_numeric_identifier_dependency_are_rejected():
    b = binding("120 KGS")
    c = contract(mode="target_sum", source_value="120", target_paths=["documentPatch.value"])
    with pytest.raises(ValueError, match="source dependency is false"):
        validate_contract(b, c, {"documentPatch": {"value": 100}})
    with pytest.raises(ValueError, match="typed numeric leaves"):
        validate_contract(b, c, {"documentPatch": {"value": "120"}})


def test_printed_constituents_conserve_target_and_reproduce_source():
    members = [binding("810 CTNS"), binding("750 CTNS"), binding("0 CTNS")]
    for index, member in enumerate(members):
        member.logical_key = f"quantity_{index}"
        member.value_kind = "integer"
    contracts = {
        b.logical_key: contract(
            mode="target_share",
            role="cargo_quantity",
            source_value=str(n),
            target_paths=["documentPatch.quantity"],
        )
        for b, n in zip(members, (810, 750, 0), strict=True)
    }
    source = {"documentPatch": {"quantity": 1560}}
    original = prepare(members, contracts, source_target=source, target=source, scale=Decimal(1))
    assert [int(r.value) for r in original.values()] == [810, 750, 0]
    for total in (2, 3, 1000, 1561):
        target = {"documentPatch": {"quantity": total}}
        rows = prepare(
            members, contracts, source_target=source, target=target, scale=Decimal("0.7")
        )
        assert sum(Decimal(r.value) for r in rows.values()) == total
        assert rows["quantity_2"].value == "0"
        render_prepared(members, rows, source_target=source, target=target)
    with pytest.raises(ValueError, match="nonzero constituents"):
        prepare(
            members,
            contracts,
            source_target=source,
            target={"documentPatch": {"quantity": 1}},
            scale=Decimal("0.7"),
        )
    with pytest.raises(ValueError, match="exactly account"):
        prepare(
            members[:1],
            {members[0].logical_key: contracts[members[0].logical_key]},
            source_target=source,
            target=source,
            scale=Decimal(1),
        )


@pytest.mark.parametrize("separator", [" ", "\u00a0", "'"])
def test_numeric_grouping_reproduces_source_and_preserves_precision(separator):
    b = binding(f"16{separator}700,230 KGS")
    c = contract(source_value="16700.230")
    rows = prepare((b,), {b.logical_key: c}, source_target={}, target={}, scale=Decimal("0.8"))
    assert render_prepared((b,), rows, source_target={}, target={}) == {
        b.logical_key: {"slot_1": f"13{separator}360,184 KGS"}
    }


def test_per_unit_weight_constrains_generated_total_before_rendering():
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import derived_target_values

    b = binding("20 KGS")
    c = contract(
        mode="unit_product",
        role="per_unit_measurement",
        source_value="20",
        target_paths=["documentPatch.quantity", "documentPatch.netWeight"],
        multiplier="0.001",
    )
    source = {"documentPatch": {"quantity": 11525, "netWeight": 230.5}}
    target = {"documentPatch": {"quantity": 8586, "netWeight": 171.723}}
    validate_contract(b, c, source)
    values = derived_target_values({b.logical_key: c}, target)
    assert values == {"documentPatch.netWeight": Decimal("171.720")}
    with pytest.raises(ValueError, match="unit-product"):
        prepare(
            (b,), {b.logical_key: c}, source_target=source, target=target, scale=Decimal("0.745")
        )
    target["documentPatch"]["netWeight"] = float(values["documentPatch.netWeight"])
    rows = prepare(
        (b,), {b.logical_key: c}, source_target=source, target=target, scale=Decimal("0.745")
    )
    assert rows[b.logical_key].value == "20"


def test_leading_decimal_grammar_and_unit_labels_are_not_quantities():
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import measurement_unit_binding

    b = binding(".24")
    c = contract(role="cargo_volume", source_value="0.24")
    rows = prepare((b,), {b.logical_key: c}, source_target={}, target={}, scale=Decimal("0.5"))
    assert (
        render_prepared((b,), rows, source_target={}, target={})[b.logical_key]["slot_1"] == ".12"
    )
    for unit in ("M3", "CBM", "KGS"):
        b = binding(unit)
        b.target_paths = ()
        assert measurement_unit_binding(b)
    b = binding("1.2 M3")
    b.target_paths = ()
    assert not measurement_unit_binding(b)


def test_repeated_equal_constituents_constrain_total_divisibility():
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import generated_measurements

    b = binding("27510")
    b.occurrences = tuple(NS(slot_id=f"s_{i}", source_text="27510") for i in range(3))
    path = "documentPatch.cargoGroups[0].netWeight.value"
    c = contract(mode="target_average", source_value="27510", target_paths=[path], divisor=3)
    source = {"documentPatch": {"cargoGroups": [{"netWeight": {"value": 82530}}]}}
    target = {"documentPatch": {"cargoGroups": [{"netWeight": {"value": 61517.25}}]}}
    validate_contract(b, c, source)
    generated = generated_measurements((b,), {b.logical_key: c}, target)
    assert generated[path] == Decimal(61515)
    target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = float(generated[path])
    rows = prepare(
        (b,), {b.logical_key: c}, source_target=source, target=target, scale=Decimal("0.745")
    )
    assert rows[b.logical_key].value == "20505.0"
    assert set(
        render_prepared((b,), rows, source_target=source, target=target)[b.logical_key].values()
    ) == {"20505"}


@pytest.mark.parametrize("surface", ["0,00", "0.000"])
def test_zero_decimal_is_not_misread_as_thousands_grouping(surface):
    b = binding(surface)
    c = contract(mode="source_fixed", role="commercial", source_value="0")
    rows = prepare((b,), {b.logical_key: c}, source_target={}, target={}, scale=Decimal(1))
    assert (
        render_prepared((b,), rows, source_target={}, target={})[b.logical_key]["slot_1"] == surface
    )


def test_explicit_fixed_dimensions_survive_without_inventing_a_scalar():
    b = binding("1095 X 1910 X 280 CM")
    c = contract(mode="surface_fixed", role="dimensions", source_value=b.occurrences[0].source_text)
    rows = prepare((b,), {b.logical_key: c}, source_target={}, target={}, scale=Decimal("0.8"))
    assert (
        render_prepared((b,), rows, source_target={}, target={})[b.logical_key]["slot_1"]
        == b.occurrences[0].source_text
    )
    with pytest.raises(ValueError, match="mutable cargo"):
        validate_contract(b, c.model_copy(update={"role": "cargo_mass"}), {})


def test_invalid_numeric_text_is_an_explicit_validation_error():
    with pytest.raises(ValueError, match="not a decimal"):
        validate_contract(binding(), contract(source_value="not a number"), {})


def test_pound_conversion_proves_printed_rounding_intervals_and_new_values():
    path = "documentPatch.cargoGroups[0].grossWeight.value"
    source = {
        "documentPatch": {"cargoGroups": [{"grossWeight": {"value": 978.5, "unit": "kilogram"}}]}
    }
    target = {
        "documentPatch": {"cargoGroups": [{"grossWeight": {"value": 800.0, "unit": "kilogram"}}]}
    }
    b = binding("2157.223 Lbs")
    measured = binding("978.500 Kgs")
    measured.target_paths = (path,)
    template = NS(bindings=(measured,))
    c = contract(
        mode="target_converted", source_value="2157.223", target_paths=[path], multiplier="kg_to_lb"
    )
    validate_contract(b, c, source, source_template=template)
    rows = prepare(
        (b,),
        {b.logical_key: c},
        source_target=source,
        target=target,
        scale=Decimal("0.8"),
        source_template=template,
    )
    assert rows[b.logical_key].value == "1763.698"
    assert (
        render_prepared((b,), rows, source_target=source, target=target, source_template=template)[
            b.logical_key
        ]["slot_1"]
        == "1763.698 Lbs"
    )
    b = binding("2157.300 Lbs")
    with pytest.raises(ValueError, match="intervals do not overlap"):
        validate_contract(
            b, c.model_copy(update={"source_value": "2157.300"}), source, source_template=template
        )


def test_cubic_foot_conversion_uses_exact_foot_definition():
    path = "documentPatch.cargoGroups[0].volume.value"
    source = {
        "documentPatch": {"cargoGroups": [{"volume": {"value": 6.366, "unit": "cubic_metre"}}]}
    }
    measured = binding("6.366 Cbm")
    measured.target_paths = (path,)
    template = NS(bindings=(measured,))
    b = binding("224.814 Cbf")
    c = contract(
        mode="target_converted",
        role="cargo_volume",
        source_value="224.814",
        target_paths=[path],
        multiplier="cbm_to_cbf",
    )
    validate_contract(b, c, source, source_template=template)
    target = {"documentPatch": {"cargoGroups": [{"volume": {"value": 5.0, "unit": "cubic_metre"}}]}}
    row = prepare(
        (b,),
        {b.logical_key: c},
        source_target=source,
        target=target,
        scale=Decimal("0.8"),
        source_template=template,
    )
    assert row[b.logical_key].value == "176.573"


@pytest.mark.parametrize(
    "surface,value,new,expected",
    [("01 KGS", 1, 2, "02 KGS"), ("+06", 6, 8, "+08"), ("-01", -1, -2, "-02")],
)
def test_numeric_padding_and_explicit_sign_roundtrip(surface, value, new, expected):
    from document_ocr.synthesis.rendering import render_number_surface

    assert render_number_surface(surface, value, value) == surface
    assert render_number_surface(surface, value, new) == expected


@pytest.mark.parametrize("adapter", ["numeric", "agent"])
def test_unit_product_solves_package_quantity_multiple_from_all_total_surfaces(adapter):
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import quantity_multiples

    measure = "documentPatch.cargoGroups[0].netWeight.value"
    count = "documentPatch.cargoPackages[0].quantity"
    b = binding("24,498")
    b.target_paths = (measure,)
    b.realization = NS(adapter=adapter)
    source = {"documentPatch": {"cargoGroups": [{"netWeight": {"value": 24498}}]}}
    c = contract(
        mode="unit_product",
        role="per_unit_measurement",
        source_value="13.61",
        target_paths=[count, measure],
    )
    assert quantity_multiples({"per_carton": c}, source, NS(bindings=(b,))) == {count: 100}
