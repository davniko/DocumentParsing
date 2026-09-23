import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.config import TransportCapacityConfig
from document_ocr.synthesis.template_compiler import equipment_row_constraints as rows
from document_ocr.synthesis.template_compiler.numeric_auxiliary import (
    NumericContract,
    PreparedNumeric,
)
from document_ocr.synthesis.transport_capacity import capacity_limits


def fixture(text="Équipment ABCU1234567/40HQ/SEAL123/60CBM", role="cargo_volume", surface="60"):
    raw = text.encode()

    def occurrence(value):
        start = raw.index(value.encode())
        return NS(byte_start=start, byte_end=start + len(value.encode()), source_text=value)

    source = NS(
        source=raw,
        target={"documentPatch": {"containers": [{}, {}], "cargoGroups": [{}]}},
        template=NS(
            bindings=[
                NS(
                    logical_key="id",
                    target_paths=("documentPatch.containers[0].containerNumber",),
                    occurrences=[occurrence("ABCU1234567")],
                ),
                NS(logical_key="value", target_paths=(), occurrences=[occurrence(surface)]),
            ]
        ),
    )
    contract = NumericContract(
        mode="source_scaled",
        role=role,
        source_value=surface,
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Printed container row",
    )
    return source, {"value": contract}


def test_exact_byte_spans_establish_row_owner_and_units_with_unicode():
    source, contracts = fixture()
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "volume", Decimal(1)),
    )


def test_numeric_slot_on_other_line_is_not_claimed_as_container_cargo():
    source, contracts = fixture("ABCU1234567\nTOTAL CARGO 60CBM")
    assert rows.compile_rows(source, contracts) == ()


def test_reviewed_private_row_owner_spans_ocr_lines_without_changing_labels():
    source, contracts = fixture("ABCU1234567\nROW VOLUME 60CBM")
    b = source.template.bindings[1]
    b.dependency_paths = ("documentPatch.containers[0]",)
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "volume", Decimal(1)),
    )
    assert source.target["documentPatch"]["containers"] == [{}, {}]


def test_repeated_equal_private_rows_cover_every_declared_container():
    source, contracts = fixture("ABCU1234567\nROW VOLUME 60CBM\nROW VOLUME 60CBM")
    b = source.template.bindings[1]
    first = b.occurrences[0]
    start = source.source.rindex(b"60")
    b.occurrences.append(NS(byte_start=start, byte_end=start + 2, source_text="60"))
    b.dependency_paths = ("documentPatch.containers[0]", "documentPatch.containers[1]")
    assert rows.compile_rows(source, contracts) == tuple(
        rows.RowMeasurement("value", (i,), "volume", Decimal(1)) for i in range(2)
    )
    b.occurrences = [first]
    with pytest.raises(ValueError, match="complete printed container scope"):
        rows.compile_rows(source, contracts)


@pytest.mark.parametrize(
    "dependencies",
    [
        ("documentPatch.containers[1]",),
        ("documentPatch.containers[2]",),
        ("documentPatch.containers[0]", "documentPatch.cargoGroups[0]"),
        ("documentPatch.containers[0]", "documentPatch.containers[0]"),
    ],
)
def test_private_row_contract_rejects_conflicting_or_incomplete_owner(dependencies):
    source, contracts = fixture()
    source.template.bindings[1].dependency_paths = dependencies
    with pytest.raises(ValueError, match="physical row dependency"):
        rows.compile_rows(source, contracts)


def test_private_row_owner_does_not_supply_missing_units():
    source, contracts = fixture("ABCU1234567\nROW VOLUME 60")
    source.template.bindings[1].dependency_paths = ("documentPatch.containers[0]",)
    with pytest.raises(ValueError, match="lacks a printed measurement unit"):
        rows.compile_rows(source, contracts)


@pytest.mark.parametrize("text", ["ABCU1234567 TOTAL 60CBM", "ABCU1234567\nTOTAL 60CBM"])
def test_declared_row_ownership_cannot_reinterpret_a_shipment_total(text):
    source, contracts = fixture(text)
    source.template.bindings[1].dependency_paths = ("documentPatch.containers[0]",)
    with pytest.raises(ValueError, match="aggregate equipment scope"):
        rows.compile_rows(source, contracts)


def test_singleton_explicit_cargo_total_has_exact_physical_owner_without_new_labels():
    source, contracts = fixture("ABCU1234567\nTOTAL CARGO 60CBM")
    source.target = {"documentPatch": {"containers": [{}], "cargoGroups": [{}]}}
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "volume", Decimal(1)),
    )
    assert source.target == {"documentPatch": {"containers": [{}], "cargoGroups": [{}]}}


def test_singleton_does_not_infer_unprinted_units_or_unrepresented_group_ownership():
    source, contracts = fixture("ABCU1234567\nTOTAL CARGO 60")
    source.target = {"documentPatch": {"containers": [{}], "cargoGroups": [{}]}}
    assert rows.compile_rows(source, contracts) == ()
    source, contracts = fixture("ABCU1234567\nTOTAL CARGO 60CBM")
    source.target = {"documentPatch": {"containers": [{}], "cargoGroups": [{}, {}]}}
    assert rows.compile_rows(source, contracts) == ()


def test_container_and_allocation_aliases_keep_the_physical_row_owner():
    source, contracts = fixture()
    source.template.bindings[0].target_paths += (
        "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
    )
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "volume", Decimal(1)),
    )


def test_seal_suffix_not_a_measurement_without_a_numeric_contract():
    source, contracts = fixture("ABCU1234567/KG-SEAL123/60CBM")
    assert len(rows.compile_rows(source, contracts)) == 1


def test_unit_cannot_be_guessed_from_number_or_copied_from_another_line():
    source, contracts = fixture("ABCU1234567/60\nCBM")
    assert rows.compile_rows(source, contracts) == ()


@pytest.mark.parametrize("mode", ["target_sum", "target_share", "target_average"])
def test_matching_digits_cannot_hide_wrong_tonne_conversion(mode):
    source, contracts = fixture("ABCU1234567\nNET WEIGHT: 25,925 MT", "cargo_mass", "25,925")
    source.target["documentPatch"]["cargoGroups"] = [
        {"grossWeight": {"value": 25925, "unit": "kilogram"}}
    ]
    contracts["value"] = contracts["value"].model_copy(
        update={
            "mode": mode,
            "source_value": "25925",
            "target_paths": ["documentPatch.cargoGroups[0].grossWeight.value"],
        }
    )
    # Check even without a uniquely owned container row: physical unit proof
    # does not depend on whether the inventory itself is complete.
    with pytest.raises(ValueError, match="conversion contradicts the printed unit"):
        rows.compile_rows(source, contracts)
    contracts["value"] = contracts["value"].model_copy(
        update={"source_value": "25.925", "multiplier": "0.001"}
    )
    assert rows.compile_rows(source, contracts) == ()
    source.target["documentPatch"]["containers"] = [{}]
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "mass", Decimal(1000)),
    )


def test_shared_row_with_two_containers_requires_explicit_scope():
    source, contracts = fixture()
    source.template.bindings[0].target_paths += ("documentPatch.containers[1].containerNumber",)
    with pytest.raises(ValueError, match="ambiguous equipment ownership"):
        rows.compile_rows(source, contracts)


def test_measurement_role_must_match_printed_unit():
    source, contracts = fixture(role="cargo_mass")
    with pytest.raises(ValueError, match="unit contradicts"):
        rows.compile_rows(source, contracts)


def test_collapsed_ocr_rows_are_owned_by_their_certified_identifier_boundaries():
    source, contracts = fixture("ABCU1234567/60CBM DEFU9876543/40CBM")
    start = source.source.index(b"DEFU")
    source.template.bindings.append(
        NS(
            logical_key="second_id",
            target_paths=("documentPatch.containers[1].containerNumber",),
            occurrences=[NS(byte_start=start, byte_end=start + 11, source_text="DEFU9876543")],
        )
    )
    number = source.source.index(b"40CBM")
    source.template.bindings.append(
        NS(
            logical_key="second_value",
            target_paths=(),
            occurrences=[NS(byte_start=number, byte_end=number + 2, source_text="40")],
        )
    )
    contracts["second_value"] = contracts["value"].model_copy(update={"source_value": "40"})
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "volume", Decimal(1)),
        rows.RowMeasurement("second_value", (1,), "volume", Decimal(1)),
    )


def test_same_line_total_is_not_assigned_to_the_last_container():
    source, contracts = fixture("ABCU1234567  TOTAL 60CBM")
    with pytest.raises(ValueError, match="aggregate equipment scope"):
        rows.compile_rows(source, contracts)


def test_container_capacity_uses_generated_value_not_source_and_preserves_labels():
    source, contracts = fixture()
    observations = rows.compile_rows(source, contracts)
    loads = rows.prepare_loads(
        observations,
        {"value": PreparedNumeric(contract=contracts["value"], value="53", scenario_scale="0.88")},
    )
    config = json.loads(
        Path("configs/synthesis/production/mpci_bl_diversified_cargo_sampling_v1.json").read_bytes()
    )
    limits = capacity_limits(TransportCapacityConfig.model_validate(config["transport_capacity"]))
    small = dict(sizeCategory="TWENTY_FOOT_STANDARD_HEIGHT", typeCategory="GENERAL_PURPOSE")
    large = dict(sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE")
    assert not rows.fits(loads[0], [small], limits)
    assert rows.fits(loads[0], [large], limits)
    with pytest.raises(ValueError, match="exceeds equipment capacity"):
        rows.validate(loads, [small], limits)
    assert rows.validate(loads, [large], limits)[0]["value"] == "53"
    proof = rows.validate(loads, [large], limits)
    prepared = PreparedNumeric(contract=contracts["value"], value="53", scenario_scale="0.88")
    rows.validate_prepared(proof, {"value": prepared})
    with pytest.raises(ValueError, match="changed after equipment sampling"):
        rows.validate_prepared(proof, {"value": prepared.model_copy(update={"value": "80"})})
    assert set(large) == {"sizeCategory", "typeCategory"}


def test_pound_mass_is_converted_exactly():
    source, contracts = fixture("ABCU1234567/10000LBS", "cargo_mass", "10000")
    loads = rows.prepare_loads(
        rows.compile_rows(source, contracts),
        {"value": PreparedNumeric(contract=contracts["value"], value="10000", scenario_scale="1")},
    )
    assert loads[0].value == Decimal("4535.92370000")


@pytest.mark.parametrize("surface,factor", [("254KG", "1"), ("16MT", "1000")])
def test_unit_inside_certified_numeric_slot_is_preserved_as_physical_evidence(surface, factor):
    source, contracts = fixture("ABCU1234567/" + surface, "cargo_mass", surface)
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "mass", Decimal(factor)),
    )


@pytest.mark.parametrize(
    "role,unit,dimension",
    [("cargo_mass", "kilogram", "mass"), ("cargo_volume", "cubic_metre", "volume")],
)
def test_private_units_require_explicit_contract_and_remain_outside_source_and_labels(
    role, unit, dimension
):
    source, contracts = fixture("ABCU1234567\nMEASUREMENT: 60", role)
    source.target["documentPatch"]["containers"] = [{}]
    original = json.dumps(source.target)
    assert rows.compile_rows(source, contracts) == ()
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "synthetic_unit": unit}
    )
    observed = rows.compile_rows(source, contracts)
    assert len(observed) == 1
    assert observed[0].dimension == dimension
    assert observed[0].unit_evidence == "synthetic_private"
    assert observed[0].unit_factor == 1
    assert json.dumps(source.target) == original
    assert source.source == b"ABCU1234567\nMEASUREMENT: 60"
    config = json.loads(
        Path("configs/synthesis/production/mpci_bl_diversified_cargo_sampling_v1.json").read_bytes()
    )
    limits = capacity_limits(TransportCapacityConfig.model_validate(config["transport_capacity"]))
    prepared = {
        "value": PreparedNumeric(contract=contracts["value"], value="55", scenario_scale="0.92")
    }
    proof = rows.validate(
        rows.prepare_loads(observed, prepared),
        [dict(sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE")],
        limits,
    )
    assert proof[0]["unitEvidence"] == "synthetic_private"
    assert proof[0]["unit"] == unit
    assert proof[0]["value"] == "55"
    rows.validate_prepared(proof, prepared)


@pytest.mark.parametrize(
    "text",
    [
        "ABCU1234567/60CBM",
        "ABCU1234567\nREAL CBM: 60",
        "ABCU1234567/60 SQM",
        "ABCU1234567/60 GALLONS",
    ],
)
def test_private_unit_cannot_replace_printed_unit_or_unrecognized_suffix(text):
    source, contracts = fixture(text)
    source.target["documentPatch"]["containers"] = [{}]
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "synthetic_unit": "cubic_metre"}
    )
    with pytest.raises(ValueError, match="private unit"):
        rows.compile_rows(source, contracts)


def test_private_unit_does_not_provide_missing_container_ownership():
    source, contracts = fixture("ABCU1234567\nMEASUREMENT: 60")
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "synthetic_unit": "cubic_metre"}
    )
    with pytest.raises(ValueError, match=r"private unit.*owner"):
        rows.compile_rows(source, contracts)


@pytest.mark.parametrize(
    "role,mode,unit",
    [
        ("cargo_mass", "source_scaled", "cubic_metre"),
        ("cargo_volume", "source_fixed", "cubic_metre"),
        ("commercial", "source_scaled", "kilogram"),
    ],
)
def test_private_unit_contract_rejects_wrong_dimension_or_retained_values(role, mode, unit):
    _, contracts = fixture()
    with pytest.raises(ValueError, match="private unit"):
        NumericContract.model_validate(
            {**contracts["value"].model_dump(), "role": role, "mode": mode, "synthetic_unit": unit}
        )


def test_unit_before_value_is_printed_evidence_not_an_inferred_private_choice():
    source, contracts = fixture("ABCU1234567\nREAL CBM: 60")
    source.target["documentPatch"]["containers"] = [{}]
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0,), "volume", Decimal(1)),
    )


def test_aggregate_measurement_uses_sum_capacity_not_equal_per_container_load():
    source, contracts = fixture("ABCU1234567\nTOTAL MEASUREMENT: 60")
    source.template.bindings[1].dependency_paths = ("documentPatch.containers",)
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "synthetic_unit": "cubic_metre"}
    )
    observed = rows.compile_rows(source, contracts)
    assert observed == (
        rows.RowMeasurement("value", (0, 1), "volume", Decimal(1), "synthetic_private"),
    )
    config = json.loads(
        Path("configs/synthesis/production/mpci_bl_diversified_cargo_sampling_v1.json").read_bytes()
    )
    limits = capacity_limits(TransportCapacityConfig.model_validate(config["transport_capacity"]))
    loads = rows.prepare_loads(
        observed,
        {"value": PreparedNumeric(contract=contracts["value"], value="80", scenario_scale="1.33")},
    )
    small = dict(sizeCategory="TWENTY_FOOT_STANDARD_HEIGHT", typeCategory="GENERAL_PURPOSE")
    large = dict(sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE")
    assert rows.fits(loads[0], [small, large], limits)
    assert not rows.fits(loads[0], [small, small], limits)
    proof = rows.validate(loads, [small, large], limits)
    assert proof[0]["containerIndices"] == [0, 1]
    assert proof[0]["value"] == "80"
    # One small + one large can contain this total even though an invented
    # equal 40m3 allocation would exceed the small container's capacity.
    small_pair = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
    large_pair = "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"
    domain = {small_pair: 3, large_pair: 1}
    assert rows.condition_weights(loads, ((0,), (1,)), (domain, domain), limits) == (domain, domain)
    assert rows.condition_weights(loads, ((0, 1),), (domain,), limits) == ({large_pair: 1},)
    with pytest.raises(ValueError, match="exceeds equipment capacity"):
        rows.validate(loads, [small, small], limits)
    with pytest.raises(ValueError, match="capacity-compatible"):
        rows.condition_weights(loads, ((0, 1),), ({small_pair: 1},), limits)
    assert source.target["documentPatch"]["containers"] == [{}, {}]


def test_aggregate_and_per_row_measurements_both_constrain_capacity():
    config = json.loads(
        Path("configs/synthesis/production/mpci_bl_diversified_cargo_sampling_v1.json").read_bytes()
    )
    limits = capacity_limits(TransportCapacityConfig.model_validate(config["transport_capacity"]))
    loads = [
        rows.RowLoad(rows.RowMeasurement("total", (0, 1), "volume", Decimal(1)), Decimal(80)),
        rows.RowLoad(rows.RowMeasurement("row", (0,), "volume", Decimal(1)), Decimal(50)),
    ]
    small = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
    large = "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"
    domain = {small: 3, large: 1}
    assert rows.condition_weights(loads, ((0,), (1,)), (domain, domain), limits) == (
        {large: 1},
        domain,
    )


def test_aggregate_cannot_mix_inventory_and_row_owner_contracts():
    source, contracts = fixture()
    source.template.bindings[1].dependency_paths = (
        "documentPatch.containers",
        "documentPatch.containers[0]",
    )
    with pytest.raises(ValueError, match="cannot mix"):
        rows.compile_rows(source, contracts)


@pytest.mark.parametrize("mode", ["deterministic_auxiliary", "agent_residual"])
@pytest.mark.parametrize(
    "dependencies",
    [
        ("documentPatch.containers[0]",),
        ("documentPatch.containers[0]", "documentPatch.containers[1]"),
        ("documentPatch.containers",),
    ],
)
def test_compiler_agent_can_express_reviewed_physical_scope(mode, dependencies):
    from document_ocr.synthesis.template_compiler.models import (
        AgentBindingProposal,
        AgentOccurrence,
    )

    proposal = AgentBindingProposal(
        logical_key="measurement",
        render_mode=mode,
        value_kind="decimal_measurement",
        group_kind="cargo",
        group_key="cargo:g1",
        dependency_paths=dependencies,
        occurrences=(AgentOccurrence(line_start="L00001", line_end="L00001", source_text="60"),),
        rationale="Reviewed printed shipment total or complete named row scope.",
    )
    assert proposal.target_paths == ()
    assert proposal.dependency_paths == dependencies


@pytest.mark.parametrize(
    "updates",
    [
        dict(value_kind="identifier"),
        dict(group_kind="party"),
        dict(dependency_paths=("documentPatch.containers", "documentPatch.containers[0]")),
        dict(dependency_paths=("documentPatch.containers[0]", "documentPatch.containers[0]")),
        dict(dependency_paths=("documentPatch.transport",)),
        dict(dependency_bindings=("random",)),
        dict(target_paths=("documentPatch.cargoGroups[0].grossWeight.value",)),
    ],
)
def test_physical_scope_schema_exception_is_not_a_general_derivation_bypass(updates):
    from document_ocr.synthesis.template_compiler.models import (
        AgentBindingProposal,
        AgentOccurrence,
    )

    values = dict(
        logical_key="measurement",
        render_mode="deterministic_auxiliary",
        value_kind="decimal_measurement",
        group_kind="cargo",
        group_key="cargo:g1",
        dependency_paths=("documentPatch.containers",),
        occurrences=(AgentOccurrence(line_start="L00001", line_end="L00001", source_text="60"),),
        rationale="Review scope",
    )
    with pytest.raises(ValueError):
        AgentBindingProposal(**{**values, **updates})


def test_explicit_printed_unit_quote_supports_ocr_separated_table_header():
    source, contracts = fixture(
        "ABCU1234567\n16. Gross Weight (KGS)\n60\n17. Measurement (CBM)", "cargo_mass"
    )
    source.template.bindings[1].dependency_paths = ("documentPatch.containers",)
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "printed_unit_quote": "16. Gross Weight (KGS)"}
    )
    assert rows.compile_rows(source, contracts) == (
        rows.RowMeasurement("value", (0, 1), "mass", Decimal(1), "printed"),
    )
    assert contracts["value"].synthetic_unit is None


@pytest.mark.parametrize(
    "quote", ["Gross Weight (MT)", "KGS", "17. Measurement (CBM)", "Gross Weight: 60 KGS / 0.06 MT"]
)
def test_invented_isolated_wrong_dimension_or_conflicting_unit_quotes_fail(quote):
    source, contracts = fixture(
        "ABCU1234567\n16. Gross Weight (KGS)\n60\n17. Measurement (CBM)\n"
        "Gross Weight: 60 KGS / 0.06 MT",
        "cargo_mass",
    )
    source.target["documentPatch"]["containers"] = [{}]
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "printed_unit_quote": quote}
    )
    with pytest.raises(ValueError, match="printed unit quote"):
        rows.compile_rows(source, contracts)


def test_local_unit_must_agree_with_reviewed_header_quote():
    source, contracts = fixture("ABCU1234567 60MT\nGross Weight (KGS)", "cargo_mass")
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "printed_unit_quote": "Gross Weight (KGS)"}
    )
    with pytest.raises(ValueError, match="contradicts the local"):
        rows.compile_rows(source, contracts)


def test_printed_unit_quote_and_private_unit_are_mutually_exclusive():
    _, contracts = fixture()
    with pytest.raises(ValueError, match="cannot be a private unit choice"):
        NumericContract.model_validate(
            {
                **contracts["value"].model_dump(),
                "printed_unit_quote": "Measurement (CBM)",
                "synthetic_unit": "cubic_metre",
            }
        )


def test_unit_quote_cannot_override_unrecognized_local_unit():
    source, contracts = fixture("ABCU1234567 60 SQM\nMeasurement (CBM)")
    contracts["value"] = NumericContract.model_validate(
        {**contracts["value"].model_dump(), "printed_unit_quote": "Measurement (CBM)"}
    )
    with pytest.raises(ValueError, match="cannot override unresolved"):
        rows.compile_rows(source, contracts)
