from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import unit_package_loads as loads
from document_ocr.synthesis.template_compiler.numeric_auxiliary import NumericContract


def fixture(text="400 X 25 KG BAGS", *, quantity=400, mass="25", gross=10200):
    raw = text.encode()
    start = raw.index(mass.encode(), raw.index(b"X") + 1)
    package = dict(packageId="p1", groupId="g1", quantity=quantity, typeCategory="PACKAGE_BAG")
    group = dict(groupId="g1")
    if gross is not None:
        group["grossWeight"] = dict(value=gross, unit="kilogram")
    quantity_slot = NS(byte_start=0, byte_end=len(str(quantity)), source_text=str(quantity))
    mass_slot = NS(byte_start=start, byte_end=start + len(mass), source_text=mass)
    source = NS(
        source=raw,
        target={"documentPatch": {"cargoGroups": [group], "cargoPackages": [package]}},
        template=NS(
            bindings=[
                NS(
                    logical_key="quantity",
                    target_paths=("documentPatch.cargoPackages[0].quantity",),
                    dependency_paths=(),
                    occurrences=[quantity_slot],
                ),
                NS(
                    logical_key="unit",
                    target_paths=(),
                    dependency_paths=("documentPatch.cargoPackages[0]",),
                    occurrences=[mass_slot],
                ),
            ]
        ),
    )
    contract = NumericContract(
        mode="source_fixed",
        role="per_unit_measurement",
        source_value=mass,
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Explicit printed package-mass row",
    )
    return source, {"unit": contract}


def test_exact_package_unit_mass_is_lower_bound_not_unseen_net_label():
    source, contracts = fixture()
    original = deepcopy(source.target)
    compiled = loads.compile_loads(source, contracts)
    assert compiled == (loads.PackageUnitLoad("unit", 0, "g1", Decimal(25)),)
    assert loads.lower_bounds(compiled, source.target) == {"g1": Decimal(10000)}
    assert source.target == original
    assert "netWeight" not in source.target["documentPatch"]["cargoGroups"][0]


def test_generated_quantities_change_private_floor_without_rewriting_labels():
    source, contracts = fixture(gross=None)
    compiled = loads.compile_loads(source, contracts)
    target = deepcopy(source.target)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 800
    assert loads.lower_bounds(compiled, target) == {"g1": Decimal(20000)}
    assert "netWeight" not in target["documentPatch"]["cargoGroups"][0]


def test_source_and_generated_gross_must_cover_printed_package_mass():
    source, contracts = fixture(gross=9999)
    with pytest.raises(ValueError, match="gross weight is below"):
        loads.compile_loads(source, contracts)
    source, contracts = fixture()
    compiled = loads.compile_loads(source, contracts)
    target = deepcopy(source.target)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 800
    with pytest.raises(ValueError, match="gross weight is below"):
        loads.validate_gross(loads.lower_bounds(compiled, target), target)


@pytest.mark.parametrize(
    "unit,mass_factor",
    [
        ("KG", "1"),
        ("KGS", "1"),
        ("KGM", "1"),
        ("KILOGRAMS", "1"),
        ("LB", "0.45359237"),
        ("LBS", "0.45359237"),
        ("POUNDS", "0.45359237"),
        ("MT", "1000"),
        ("TONNES", "1000"),
    ],
)
def test_printed_mass_units_are_converted_exactly(unit, mass_factor):
    source, contracts = fixture(f"400 X 25 {unit} BAGS", gross=None)
    assert loads.compile_loads(source, contracts)[0].mass_kg == Decimal(25) * Decimal(mass_factor)


@pytest.mark.parametrize("text", ["400 X 25 CBM BAGS", "400 X 25 BAGS", "400 X 25 KG DRUMS"])
def test_dimension_and_package_noun_must_be_source_proved(text):
    source, contracts = fixture(text)
    with pytest.raises(ValueError):
        loads.compile_loads(source, contracts)


def test_duplicate_equivalent_observation_does_not_double_count():
    source, contracts = fixture()
    source.template.bindings[1].occurrences *= 2
    assert loads.lower_bounds(loads.compile_loads(source, contracts), source.target) == {
        "g1": Decimal(10000)
    }


@pytest.mark.parametrize(
    "change", ["owner", "count_binding", "source_bytes", "mode", "count_value"]
)
def test_owned_scope_and_exact_source_spans_are_required(change):
    source, contracts = fixture()
    if change == "owner":
        source.template.bindings[1].dependency_paths = ("documentPatch.cargoPackages[2]",)
    if change == "count_binding":
        source.template.bindings[0].target_paths = ()
    if change == "source_bytes":
        source.template.bindings[1].occurrences[0].source_text = "26"
    if change == "mode":
        contracts["unit"] = contracts["unit"].model_copy(update={"mode": "source_scaled"})
    if change == "count_value":
        source.target["documentPatch"]["cargoPackages"][0]["quantity"] = 399
    with pytest.raises(ValueError):
        loads.compile_loads(source, contracts)


@pytest.mark.parametrize("count", [True, 0, -1, 1.5, None])
def test_invalid_generated_package_quantities_fail(count):
    source, contracts = fixture()
    compiled = loads.compile_loads(source, contracts)
    source.target["documentPatch"]["cargoPackages"][0]["quantity"] = count
    with pytest.raises(ValueError):
        loads.lower_bounds(compiled, source.target)


def test_independent_package_rows_sum_once_per_group():
    first = loads.PackageUnitLoad("a", 0, "g1", Decimal(25))
    second = loads.PackageUnitLoad("b", 1, "g1", Decimal(25))
    third = loads.PackageUnitLoad("c", 2, "g1", Decimal(25))
    target = {
        "documentPatch": {
            "cargoPackages": [dict(groupId="g1", quantity=n) for n in (400, 400, 220)],
            "cargoGroups": [dict(groupId="g1")],
        }
    }
    assert loads.lower_bounds((first, second, third), target) == {"g1": Decimal(25500)}


def test_duplicate_package_scope_is_rejected_at_runtime_boundary():
    source, contracts = fixture()
    compiled = loads.compile_loads(source, contracts)
    with pytest.raises(ValueError, match="duplicates"):
        loads.lower_bounds(compiled + compiled, source.target)
