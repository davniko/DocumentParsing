from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import descendant as render
from document_ocr.synthesis.template_compiler.host import SpanDraft, validate_binding_realizations


def drafts(raw="42 PALLETS\n1-42"):
    count = SpanDraft(
        draft_id="count",
        logical_key="pallet_count",
        render_mode="deterministic_auxiliary",
        value_kind="integer",
        group_kind="cargo",
        group_key="cargo:g1",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=0,
        char_end=2,
        source_text=raw[:2],
        evidence_origin="host_verified_agent_proposal",
        render_policy="numeric_surface",
        rationale="Printed outer pallet quantity.",
    )
    start = raw.index("\n") + 1
    interval = replace(
        count,
        draft_id="range",
        logical_key="pallet_range",
        render_mode="deterministic_derived",
        derivation="inclusive_range_cardinality",
        dependency_bindings=("pallet_count",),
        char_start=start,
        char_end=len(raw),
        source_text=raw[start:],
        render_policy="derived_surface",
        rationale="Printed numbering of those same pallets.",
    )
    return count, interval


def test_outer_range_has_source_proven_owner_without_inner_quantity_label():
    target = {"documentPatch": {"cargoPackages": [{"quantity": 336}]}}
    validate_binding_realizations(raw="42 PALLETS\n1-42", drafts=drafts(), source_target=target)


@pytest.mark.parametrize(
    "defect", ["disagreement", "wrong_kind", "wrong_group", "missing", "two_owners"]
)
def test_auxiliary_range_rejects_invalid_source_owner(defect):
    raw = "41 PALLETS\n1-42" if defect == "disagreement" else "42 PALLETS\n1-42"
    count, interval = drafts(raw)
    if defect == "wrong_kind":
        count = replace(count, value_kind="decimal_measurement")
    if defect == "wrong_group":
        count = replace(count, group_kind="metadata")
    if defect == "missing":
        interval = replace(interval, dependency_bindings=("missing",))
    if defect == "two_owners":
        interval = replace(interval, dependency_paths=("documentPatch.cargoPackages[0].quantity",))
    with pytest.raises(ValueError):
        validate_binding_realizations(
            raw=raw,
            drafts=(count, interval),
            source_target={"documentPatch": {"cargoPackages": [{"quantity": 42}]}},
        )


@pytest.mark.parametrize(
    "surface,quantity,expected",
    [("1-42", 9, "1-9"), ("P001-P042", 7, "P001-P007"), ("42-1", 3, "42-40")],
)
def test_range_renders_prepared_outer_count_not_inner_cartons(surface, quantity, expected):
    binding = NS(
        dependency_paths=(),
        dependency_bindings=("outer",),
        occurrences=(NS(slot_id="slot_0001", source_text=surface),),
    )
    output = render._render_inclusive_range_cardinality(
        binding,
        target={"documentPatch": {"cargoPackages": [{"quantity": 336}]}},
        outputs={"outer": render.BindingOutput(replacements={}, canonical_value=str(quantity))},
    )
    assert output.replacements == {"slot_0001": expected}
    assert output.canonical_value == quantity


@pytest.mark.parametrize("value", ["0", "-2", "2.5", True, None])
def test_range_does_not_coerce_invalid_generated_count(value):
    binding = NS(dependency_paths=(), dependency_bindings=("outer",), occurrences=())
    with pytest.raises(ValueError):
        render._render_inclusive_range_cardinality(
            binding,
            target={},
            outputs={"outer": render.BindingOutput(replacements={}, canonical_value=value)},
        )


def test_target_backed_range_keeps_its_quantity_owner():
    binding = NS(
        dependency_paths=("documentPatch.quantity",),
        dependency_bindings=(),
        occurrences=(NS(slot_id="slot_0001", source_text="1-42"),),
    )
    assert render._render_inclusive_range_cardinality(
        binding,
        target={"documentPatch": {"quantity": 5}},
        outputs={},
    ).replacements == {"slot_0001": "1-5"}
