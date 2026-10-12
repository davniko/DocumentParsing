"""Authoritative physical count ownership and alias composition regression tests."""

import json
from copy import deepcopy
from dataclasses import replace
from random import Random
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import combine_surfaces
from document_ocr.synthesis.curated_ownership import ownership_surfaces
from document_ocr.synthesis.curated_packaging import (
    compile_package_count_context,
    project_package_aliases_with_context,
)
from document_ocr.synthesis.curated_templates import RenderRegion

PREFIX = "documentPatch.goodsItemDetails[0]"
TOTAL = PREFIX + ".numberAndTypeOfPackages[0].packageQuantity"
TYPE = PREFIX + ".numberAndTypeOfPackages[0].typeCategory"
ROW0 = PREFIX + ".splitGoodsPlacement[0].packageQuantity"
ROW1 = PREFIX + ".splitGoodsPlacement[1].packageQuantity"


def _aliases_from_plan(bp, plan, *, mode, rng):
    return project_package_aliases_with_context(
        bp, plan, compile_package_count_context(bp, plan), mode=mode, rng=rng
    )


def fixture():
    source = "1 PACKAGES\n2 PACKAGES\n"
    regions = (
        RenderRegion("q0", 0, 0, 1, "1", (ROW0, TOTAL), "count", "q0"),
        RenderRegion("unit", 0, 2, 10, "PACKAGES", (TYPE,), "package"),
        RenderRegion("q1", 0, 11, 12, "2", (ROW1,), "count", "q1"),
        RenderRegion("unit", 1, 13, 21, "PACKAGES", (TYPE,), "package"),
    )
    contract = SimpleNamespace(
        variables=[
            SimpleNamespace(key="q0", kind="count"),
            SimpleNamespace(key="q1", kind="count"),
        ],
        targets=[
            SimpleNamespace(path=ROW0, expression="{q0}"),
            SimpleNamespace(path=ROW1, expression="{q1}"),
        ],
    )
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 3, "typeCategory": "PACKAGE_DRUM_PLASTIC"}
                    ],
                    "splitGoodsPlacement": [{"packageQuantity": 1}, {"packageQuantity": 2}],
                }
            ]
        },
    }
    blueprint = SimpleNamespace(
        source=source,
        regions=regions,
        contract=contract,
        target=deepcopy(target),
        historical_bindings={
            "unit": {"occurrences": [{"source_text": "PACKAGES"}, {"source_text": "PACKAGES"}]}
        },
        ownership_data={},
    )
    plan = SimpleNamespace(
        target=target,
        variable_values={"q0": "1", "q1": "2"},
        surface_values={"equipment": ["unchanged equipment"], "quantity_echo": ["3"]},
    )
    return blueprint, plan


def test_reproduced_merged_alias_conflict_is_repaired_by_expression_context():
    bp, plan = fixture()
    assert set(bp.regions[0].target_paths) == {ROW0, TOTAL}
    facts = compile_package_count_context(bp, plan)
    assert [(fact.owner, fact.quantity, fact.target_paths) for fact in facts] == [
        ("q0", 1, (ROW0,)),
        ("q1", 2, (ROW1,)),
    ]
    surfaces = _aliases_from_plan(bp, plan, mode="source", rng=Random(2))
    assert surfaces["unit"] == ["PLASTIC DRUM", "PLASTIC DRUMS"]
    assert {key: value for key, value in surfaces.items() if key != "unit"} == plan.surface_values
    assert plan.target == bp.target
    assert plan.surface_values == {"equipment": ["unchanged equipment"], "quantity_echo": ["3"]}


def test_registry_replay_and_caller_immutability():
    bp, plan = fixture()
    original = deepcopy(plan.__dict__)
    first = _aliases_from_plan(bp, plan, mode="registry", rng=Random(9))
    second = _aliases_from_plan(bp, plan, mode="registry", rng=Random(9))
    assert first == second
    assert plan.__dict__ == original


def test_package_count_receipt_json_roundtrip_preserves_exact_metadata():
    bp, plan = fixture()
    facts = compile_package_count_context(bp, plan)
    before = deepcopy(facts)
    receipts = [fact.receipt() for fact in facts]
    assert all(isinstance(receipt["target_paths"], list) for receipt in receipts)
    assert json.loads(json.dumps(receipts, ensure_ascii=False)) == receipts
    assert facts == before
    assert all(isinstance(fact.target_paths, tuple) for fact in facts)


@pytest.mark.parametrize("bad", [None, True, "0", "-1", "1.2", "NaN", "Infinity", "unknown"])
def test_invalid_physical_count_is_not_trusted(bad):
    bp, plan = fixture()
    plan.variable_values["q0"] = bad
    with pytest.raises(ValueError, match="positive integer"):
        _aliases_from_plan(bp, plan, mode="source", rng=Random(1))


def test_wrong_canonical_planner_value_is_not_trusted():
    bp, plan = fixture()
    plan.variable_values["q0"] = "3"
    with pytest.raises(ValueError, match="differs from its authoritative target expression"):
        _aliases_from_plan(bp, plan, mode="source", rng=Random(1))


def test_missing_physical_value_is_not_derived_from_ambiguous_alias_targets():
    bp, plan = fixture()
    del plan.variable_values["q0"]
    with pytest.raises(ValueError, match="omitted its authoritative count variable"):
        _aliases_from_plan(bp, plan, mode="source", rng=Random(1))


def test_historical_conflict_without_expression_is_held():
    bp, plan = fixture()
    bp.regions = (replace(bp.regions[0], curated_key=None, kind="integer"), *bp.regions[1:])
    with pytest.raises(ValueError, match="conflicting canonical owners"):
        _aliases_from_plan(bp, plan, mode="source", rng=Random(1))


def test_explicit_historical_count_expression_resolves_its_exact_role():
    bp, plan = fixture()
    bp.regions = (replace(bp.regions[0], curated_key=None, kind="integer"), *bp.regions[1:])
    bp.ownership_data = {"surfaces": {"q0": {"expression": "{" + ROW0 + "}"}}}
    context = compile_package_count_context(bp, plan)
    assert context[0].basis == "declared_count_expression"
    assert context[0].quantity == 1
    assert _aliases_from_plan(bp, plan, mode="source", rng=Random(1))["unit"][0] == "PLASTIC DRUM"


def test_unambiguous_historical_count_owner_is_supported():
    bp, plan = fixture()
    bp.regions = (
        replace(bp.regions[0], curated_key=None, kind="integer", target_paths=(ROW0,)),
        *bp.regions[1:],
    )
    context = compile_package_count_context(bp, plan)
    assert context[0].basis == "unambiguous_historical_count_owner"
    assert context[0].quantity == 1


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "span",
        "target",
        "quantity",
        "role",
        "duplicate",
        "missing",
        "owner",
        "quote",
        "basis",
    ],
)
def test_falsified_or_stale_context_cannot_authorize_aliases(change):
    bp, plan = fixture()
    facts = list(compile_package_count_context(bp, plan))
    first = facts[0]
    if change == "source":
        facts[0] = replace(first, source_sha256="0" * 64)
    elif change == "span":
        facts[0] = replace(first, byte_start=1)
    elif change == "target":
        facts[0] = replace(first, target_sha256="0" * 64)
    elif change == "quantity":
        facts[0] = replace(first, quantity=2)
    elif change == "role":
        facts[0] = replace(first, quantity=3, target_paths=(TOTAL,))
    elif change == "duplicate":
        facts.append(first)
    elif change == "missing":
        facts.pop(0)
    elif change == "owner":
        facts[0] = replace(first, owner="unowned")
    elif change == "quote":
        facts[0] = replace(first, source_text_sha256=digest(b"2"))
    elif change == "basis":
        facts[0] = replace(first, basis="guess_from_merged_aliases")
    with pytest.raises(ValueError):
        project_package_aliases_with_context(bp, plan, facts, mode="source", rng=Random(1))


def test_corrupt_source_and_noun_spans_are_not_trusted():
    bp, plan = fixture()
    bp.source = "9" + bp.source[1:]
    with pytest.raises(ValueError, match="pinned source span"):
        _aliases_from_plan(bp, plan, mode="source", rng=Random(1))
    bp, plan = fixture()
    bp.regions = (bp.regions[0], replace(bp.regions[1], source="BAG"), *bp.regions[2:])
    with pytest.raises(ValueError, match="pinned source span"):
        _aliases_from_plan(bp, plan, mode="source", rng=Random(1))


def test_newline_does_not_establish_a_count_noun_relation():
    bp, plan = fixture()
    # Separate numeric and category blocks lack an exact horizontal relation;
    # declared count3 determines plural, not the unrelated previous rowcount1.
    bp.source = "1\nPACKAGES\n2 PACKAGES\n"
    assert _aliases_from_plan(bp, plan, mode="source", rng=Random(1))["unit"][0] == "PLASTIC DRUMS"


def test_declared_package_repair_recipe_remains_the_sole_noun_owner():
    bp, plan = fixture()
    bp.ownership_data = {"surfaces": {"unit": {"package_path": TYPE}}}
    # Registry optional-(S) selection must not independently override the
    # explicitly certified malformed-header-to-package-noun repair.
    projection = _aliases_from_plan(bp, plan, mode="registry", rng=Random(0))
    assert "unit" not in projection
    owned = ownership_surfaces(bp, plan.target, SimpleNamespace(party_localities={}))
    assert owned["unit"] == ["PLASTIC DRUMS", "PLASTIC DRUMS"]
    combined = combine_surfaces(projection, owned)
    assert combined["unit"] == owned["unit"]
    assert projection == plan.surface_values
    assert plan.target == bp.target


def test_existing_recipe_surface_is_preserved_not_removed_or_replaced():
    bp, plan = fixture()
    bp.ownership_data = {"surfaces": {"unit": {"package_path": TYPE}}}
    plan.surface_values["unit"] = ["PLASTIC DRUMS", "PLASTIC DRUMS"]
    assert _aliases_from_plan(bp, plan, mode="registry", rng=Random(0)) == plan.surface_values


def test_declared_surface_conflict_is_not_blanket_silenced():
    bp, plan = fixture()
    bp.ownership_data = {"surfaces": {"unit": {"package_path": TYPE}}}
    plan.surface_values["unit"] = ["WRONG", "WRONG"]
    projection = _aliases_from_plan(bp, plan, mode="registry", rng=Random(0))
    owned = ownership_surfaces(bp, plan.target, SimpleNamespace(party_localities={}))
    with pytest.raises(ValueError, match="conflicting render ownership"):
        combine_surfaces(projection, owned)


@pytest.mark.parametrize(
    "field,value",
    [
        ("occurrence", False),
        ("byte_start", False),
        ("byte_end", True),
    ],
)
def test_context_boolean_index_or_span_is_not_an_integer(field, value):
    bp, plan = fixture()
    facts = list(compile_package_count_context(bp, plan))
    facts[0] = replace(facts[0], **{field: value})
    with pytest.raises(ValueError, match="typed integer occurrence/span"):
        project_package_aliases_with_context(bp, plan, facts, mode="source", rng=Random(0))


def test_composite_count_and_type_owner_stays_under_its_physical_recipe():
    bp, plan = fixture()
    count_and_type = RenderRegion(
        "composite",
        0,
        0,
        10,
        "1 PACKAGES",
        (ROW0, TYPE),
        "owned_text",
    )
    bp.regions = (count_and_type, *bp.regions[2:])
    bp.historical_bindings["composite"] = {"occurrences": [{"source_text": "1 PACKAGES"}]}
    plan.surface_values["composite"] = ["1 PACKAGES"]
    facts = compile_package_count_context(bp, plan)
    assert [fact.owner for fact in facts] == ["q1"]
    surfaces = project_package_aliases_with_context(bp, plan, facts, mode="source", rng=Random(0))
    assert surfaces["composite"] == ["1 PACKAGES"]
    assert plan.target == bp.target
