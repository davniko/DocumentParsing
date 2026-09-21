from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import measurement_prose
from document_ocr.synthesis.template_compiler.complete_pipeline import _generation_schema
from document_ocr.synthesis.template_compiler.complete_targets import (
    assemble_lexical_value,
    deterministic_auxiliary_identifier,
    fixed_dimension_paths,
    immutable_lexical_frame,
    lexical_repair_requirements,
)


def constraint(interval, minimum=8):
    return {
        "mutableTokenIntervals": interval,
        "fixedLiteralTokens": [["c", "o", "fixed"]],
        "minimumWords": minimum,
    }


def test_host_keeps_fixed_party_context_and_changes_only_mutable_identity():
    original = "OLD PANEL INDUSTRIES COMPANY LIMITED C/O: CONTEXT TRADING LLC"
    frame = immutable_lexical_frame(original, [constraint([(0, 5)])])
    assert frame["suffix"] == " C/O: CONTEXT TRADING LLC"
    assert frame["mutableSource"] == "OLD PANEL INDUSTRIES COMPANY LIMITED"
    result = assemble_lexical_value({"hostAssembly": frame}, "NEW FOREST PRODUCTS LIMITED")
    assert result == "NEW FOREST PRODUCTS LIMITED C/O: CONTEXT TRADING LLC"
    assert assemble_lexical_value({"hostAssembly": frame}, result) == result


def test_prefix_suffix_and_unicode_offsets_are_original_bytes():
    frame = immutable_lexical_frame("REF: İstanbul ROAD NO. OLD", [constraint([(1, 4)], 6)])
    assert (
        assemble_lexical_value({"hostAssembly": frame}, "Yeni Liman") == "REF: Yeni Liman NO. OLD"
    )
    assert immutable_lexical_frame("REF: İç Kap\u0131 NO. OLD", [constraint([(1, 3)], 5)]) is None


def test_repeated_occurrences_are_one_mutable_interval_not_multiple_facts():
    original = "OLD COMPANY ON BEHALDF OF OTHER COMPANY"
    frame = immutable_lexical_frame(original, [constraint([(0, 2)] * 3, 6)])
    assert frame is not None
    assert frame["mutableSource"] == "OLD COMPANY"
    assert frame["suffix"] == " ON BEHALDF OF OTHER COMPANY"
    assert assemble_lexical_value({"hostAssembly": frame}, "NEW BUSINESS") == (
        "NEW BUSINESS ON BEHALDF OF OTHER COMPANY"
    )


def test_host_assembled_field_repair_does_not_require_full_target_grammar():
    path = "documentPatch.parties.consignee.address"
    original = "25 ORIGINAL STREET - DOKKI"
    constraints = [constraint([(0, 3)], 2)]
    frame = immutable_lexical_frame(original, constraints)
    binding = NS(
        target_paths=[path],
        logical_key="address",
        value_kind="address",
        derivation=None,
        realization=NS(
            mode="token_projected_surface",
            target_values=[NS(source_value=original)],
            slots=[NS(required_target_prefix_tokens=(), required_target_suffix_tokens=("dokki",))],
        ),
    )
    target = {"documentPatch": {"parties": {"consignee": {"address": original}}}}
    source = NS(target=target, template=NS(bindings=[binding], coherence_constraints=[]))
    request = dict(
        key="address", paths=[path], source=original, constraints=constraints, hostAssembly=frame
    )
    requirements = lexical_repair_requirements(source, target, [request])
    assert all(
        "partitionPattern" not in obligation and "partitionContract" not in obligation
        for row in requirements
        for obligation in row["obligations"]
    )
    schema = _generation_schema([request], requirements)
    value = schema.model_validate({"address": "19 NEW AVENUE"}).address
    assert assemble_lexical_value(request, value) == "19 NEW AVENUE - DOKKI"
    assert assemble_lexical_value(request, "19 NEW AVENUE - DOKKI") == "19 NEW AVENUE - DOKKI"


def test_locality_inside_mutable_name_is_not_a_duplicate_frame():
    frame = immutable_lexical_frame(
        "OLD COMPANY SHIPPING (EGYPT) - ALEXANDRIA", [constraint([(0, 4)])]
    )
    assert frame["mutableSource"] == "OLD COMPANY SHIPPING (EGYPT)"
    assert assemble_lexical_value({"hostAssembly": frame}, "Alexandria Coastline Delivery") == (
        "Alexandria Coastline Delivery - ALEXANDRIA"
    )


def test_idempotent_prefix_framing_retains_new_identity():
    frame = immutable_lexical_frame("NANJING OLD COMPANY,", [constraint([(1, 3)])])
    assert assemble_lexical_value({"hostAssembly": frame}, "Nanjing Verdant Axis") == (
        "NANJING Verdant Axis,"
    )
    with pytest.raises(ValueError, match="no mutable identity"):
        assemble_lexical_value({"hostAssembly": frame}, "NANJING ,")


def test_arithmetic_only_mass_text_is_host_owned_but_product_text_is_not():
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "grossWeight": {"value": 25000, "unit": "kilogram"},
                    "additionalInformation": ["WITH 25.000 KG."],
                }
            ]
        }
    }
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    assert measurement_prose.formal_paths(source) == {path}
    source["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [
        "SISAL FIBRE WITH 25.000 KG."
    ]
    assert not measurement_prose.formal_paths(source)


@pytest.mark.parametrize(
    "constraints",
    [
        [],
        [constraint(None)],
        [constraint([(0, 2), (3, 4)])],
        [constraint([(0, 2)]), constraint([(1, 2)])],
    ],
)
def test_unproven_or_multiple_frames_remain_explicit_linguistic_contracts(constraints):
    assert immutable_lexical_frame("OLD COMPANY C/O OTHER", constraints) is None


def test_bad_coordinates_fail_instead_of_guessing():
    with pytest.raises(ValueError, match="exceeds"):
        immutable_lexical_frame("OLD COMPANY", [constraint([(0, 9)])])


@pytest.mark.parametrize("field", ["tax_identifier", "registration_identifier"])
def test_registration_is_generated_by_existing_opaque_solver(field):
    binding = NS(
        target_paths=(),
        value_kind="identifier",
        occurrences=(NS(render_policy="opaque_identifier", source_text="1001365672023070163"),),
    )
    assert deterministic_auxiliary_identifier(binding, field)
    assert not deterministic_auxiliary_identifier(binding, "postal_code")
    binding.occurrences[0].source_text = "UNKNOWN"
    assert not deterministic_auxiliary_identifier(binding, field)


@pytest.mark.parametrize(
    "surface,eligible",
    [
        ("O/H 51 CMS", True),
        ("OVERWIDTH 12.5 INCHES", True),
        ("60 BAGS", False),
        ("O/H 51 CMS FOR ACME", False),
        ("REF: 12345", False),
    ],
)
def test_fixed_geometry_is_an_explicit_narrow_scenario_condition(surface, eligible):
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    binding = NS(
        logical_key="dimension",
        target_paths=[path],
        dependency_paths=(),
        dependency_bindings=(),
        source_relationships=(),
        derivation=None,
    )
    source = NS(
        template=NS(bindings=[binding], coherence_constraints=[]),
        target={"documentPatch": {"cargoGroups": [{"additionalInformation": [surface]}]}},
    )
    assert bool(fixed_dimension_paths(source)) == eligible
    binding.dependency_paths = ("documentPatch.some_quantity",)
    assert not fixed_dimension_paths(source)
