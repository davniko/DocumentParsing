from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.descendant import _render_one_derivation


def setup_case(
    *,
    printed="1502.00",
    printed_unit="kgs",
    gross_unit="kilogram",
    gross=13502.0,
    net_unit="kilogram",
    net=12000.0,
):
    paths = (
        "documentPatch.cargoGroups[0].grossWeight.value",
        "documentPatch.cargoGroups[0].netWeight.value",
    )
    raw = (
        f"Gross: {gross} {gross_unit}\nNet: {net} {net_unit}\nTare: {printed} {printed_unit}"
    ).encode()
    start = raw.index(b"Tare: ") + len(b"Tare: ")
    slot = NS(slot_id="tare", source_text=printed, byte_start=start, byte_end=start + len(printed))
    binding = NS(
        logical_key="cargo_tare",
        derivation="gross_minus_net_weight",
        target_paths=(),
        dependency_paths=paths,
        dependency_bindings=(),
        value_kind="decimal_measurement",
        render_mode="deterministic_derived",
        group_kind="cargo",
        occurrences=(slot,),
    )
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "grossWeight": {"value": gross, "unit": gross_unit},
                    "netWeight": {"value": net, "unit": net_unit},
                }
            ]
        }
    }
    members = tuple(
        NS(target_paths=(path,), occurrences=(NS(source_text=str(value)),))
        for path, value in zip(paths, (gross, net), strict=True)
    )
    case = NS(
        source=raw, source_target=target, target=deepcopy(target), template=NS(bindings=members)
    )
    return binding, case


def render(binding, case):
    return _render_one_derivation(
        binding=binding, case=case, outputs={}, bindings={}, country_codes={}
    )


def test_cargo_tare_is_gross_minus_net_not_their_sum_and_does_not_enrich_labels():
    binding, case = setup_case()
    assert render(binding, case).replacements == {"tare": "1502.00"}
    group = case.target["documentPatch"]["cargoGroups"][0]
    group["grossWeight"]["value"] = 10000.0
    group["netWeight"]["value"] = 9000.0
    before = deepcopy(case.target)
    assert render(binding, case).replacements == {"tare": "1000.00"}
    assert case.target == before


@pytest.mark.parametrize(
    "gross_unit,gross,net_unit,net,printed,unit,expected",
    [
        ("metric_tonne", 13.502, "kilogram", 12000.0, "1502.00", "kgs", "1502.00"),
        ("kilogram", 13502.0, "metric_tonne", 12.0, "1.502", "MT", "1.502"),
        ("pound", 100.0, "pound", 90.0, "4.5359237", "KG", "4.5359237"),
        ("kilogram", 10.0, "kilogram", 10.0, "0.00", "kg", "0.00"),
    ],
)
def test_mass_units_are_converted_explicitly(
    gross_unit, gross, net_unit, net, printed, unit, expected
):
    binding, case = setup_case(
        gross_unit=gross_unit,
        gross=gross,
        net_unit=net_unit,
        net=net,
        printed=printed,
        printed_unit=unit,
    )
    assert render(binding, case).replacements == {"tare": expected}


@pytest.mark.parametrize(
    "paths",
    [
        (),
        ("documentPatch.cargoGroups",),
        (
            "documentPatch.cargoGroups[0].netWeight.value",
            "documentPatch.cargoGroups[0].grossWeight.value",
        ),
        (
            "documentPatch.cargoGroups[0].grossWeight.value",
            "documentPatch.cargoGroups[1].netWeight.value",
        ),
        (
            "documentPatch.cargoGroups[0].grossWeight.value",
            "documentPatch.cargoGroups[0].volume.value",
        ),
    ],
)
def test_incomplete_or_cross_cargo_dependencies_are_rejected(paths):
    binding, case = setup_case()
    binding.dependency_paths = paths
    with pytest.raises(ValueError, match="gross-minus-net"):
        render(binding, case)


@pytest.mark.parametrize(
    "updates",
    [
        dict(target_paths=("documentPatch.cargoGroups[0].grossWeight.value",)),
        dict(dependency_bindings=("another_measurement",)),
        dict(render_mode="deterministic_auxiliary"),
        dict(group_kind="equipment"),
        dict(value_kind="integer"),
    ],
)
def test_ambiguous_contract_metadata_fails(updates):
    binding, case = setup_case()
    for key, value in updates.items():
        setattr(binding, key, value)
    with pytest.raises(ValueError, match="gross-minus-net"):
        render(binding, case)


def test_source_arithmetic_cannot_be_waived_by_a_declared_derivation():
    binding, case = setup_case(printed="1503.00")
    with pytest.raises(ValueError, match="source does not prove"):
        render(binding, case)


@pytest.mark.parametrize("old", [False, True])
def test_negative_packaging_mass_is_not_rendered(old):
    binding, case = setup_case()
    target = case.source_target if old else case.target
    target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = 15000.0
    with pytest.raises(ValueError, match="gross-minus-net"):
        render(binding, case)


@pytest.mark.parametrize("unit", ["", "CBM", "KOS"])
def test_unprinted_or_non_mass_output_units_require_review(unit):
    binding, case = setup_case(printed_unit=unit)
    with pytest.raises(ValueError, match="gross-minus-net"):
        render(binding, case)


def test_target_cannot_silently_change_input_units():
    binding, case = setup_case()
    case.target["documentPatch"]["cargoGroups"][0]["netWeight"]["unit"] = "pound"
    with pytest.raises(ValueError, match="unit contract"):
        render(binding, case)


@pytest.mark.parametrize("printed,valid", [("1502.00", True), ("1503.00", False)])
def test_compiler_materialization_proves_difference_before_publication(printed, valid):
    from document_ocr.synthesis.template_compiler.host import (
        SpanDraft,
        validate_binding_realizations,
    )

    binding, case = setup_case(printed=printed)
    slot = binding.occurrences[0]
    draft = SpanDraft(
        draft_id="difference",
        logical_key=binding.logical_key,
        render_mode=binding.render_mode,
        value_kind=binding.value_kind,
        group_kind=binding.group_kind,
        group_key="cargo:g1",
        target_paths=(),
        derivation=binding.derivation,
        dependency_paths=binding.dependency_paths,
        dependency_bindings=(),
        char_start=slot.byte_start,
        char_end=slot.byte_end,
        source_text=printed,
        evidence_origin="derived_operational_fact",
        render_policy="derived_surface",
        rationale="Source explicitly states packaging tare beside gross and net mass.",
    )
    if valid:
        validate_binding_realizations(
            raw=case.source.decode(), drafts=(draft,), source_target=case.source_target
        )
    else:
        with pytest.raises(ValueError, match="source does not prove"):
            validate_binding_realizations(
                raw=case.source.decode(), drafts=(draft,), source_target=case.source_target
            )
