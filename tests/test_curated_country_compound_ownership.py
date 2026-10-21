from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_curated_countries import blueprint, policy

from document_ocr.synthesis.country_presentations import Registry
from document_ocr.synthesis.curated_countries import prepare_country_rendering
from document_ocr.synthesis.curated_templates import RenderRegion
from document_ocr.synthesis.generators import DeterministicStream

ROLE = "documentPatch.placeOfIssue"
EXPRESSION = "{documentPatch.placeOfIssue.name}, {documentPatch.placeOfIssue.country}"


@pytest.fixture(scope="module")
def registry():
    return Registry.load(Path(__file__).resolve().parents[1])


@pytest.fixture
def compound(registry):
    bp = replace(
        blueprint(),
        source="ETTLINGEN",
        target={"documentPatch": {"placeOfIssue": {"name": "ETTLINGEN", "country": "GERMANY"}}},
        regions=(
            RenderRegion(
                "issue", 0, 0, 9, "ETTLINGEN", (ROLE + ".name", ROLE + ".country"), "location"
            ),
        ),
        ownership_data={"surfaces": {"issue": {"expression": EXPRESSION}}},
    )
    target = {"documentPatch": {"placeOfIssue": {"name": "DUBREKA", "country": "GUINEA"}}}
    kwargs = {
        "registry": registry,
        "policy": policy(alias=1.0, family="iso2"),
        "context_country_codes": {"GN"},
        "stream": DeterministicStream("test", "countries", "compound"),
        "protected_localities": {ROLE: "DUBREKA"},
    }
    return bp, target, kwargs, prepare_country_rendering(bp, target, **kwargs)


def test_explicit_compound_surface_is_not_a_whole_scalar_target(compound):
    bp, target, _, plan = compound
    plan.validate_for(bp, target)
    projection = plan.apply_regions([(bp.regions[0], "DUBREKA, GUINEA")], variable_values={})
    assert projection.regions[0][1] == "DUBREKA, GN"
    assert projection.target_values == {ROLE + ".country": "GUINEA"}
    assert dict(plan.protected_literal_values) == {ROLE + ".name": "DUBREKA"}


@pytest.mark.parametrize(
    "text",
    ["OTHER, GUINEA", "DUBREKA, IRAQ", "DUBREKA", "DUBREKA, GUINEA, GUINEA", "GUINEA, DUBREKA"],
)
def test_compound_wrong_city_country_count_or_order_rejects(compound, text):
    bp, _, _, plan = compound
    with pytest.raises(ValueError, match="certified compound expression"):
        plan.apply_regions([(bp.regions[0], text)], variable_values={})


@pytest.mark.parametrize("field", ["name", "country"])
def test_compound_scalar_target_inputs_remain_pinned(compound, field):
    bp, target, _, plan = compound
    target = deepcopy(target)
    target["documentPatch"]["placeOfIssue"][field] = "OTHER"
    with pytest.raises(ValueError, match="pinned source, ownership or inputs"):
        plan.validate_for(bp, target)


def test_compound_ownership_expression_remains_pinned(compound):
    bp, target, _, plan = compound
    bp = deepcopy(bp)
    bp.ownership_data["surfaces"]["issue"]["expression"] = EXPRESSION + "."
    with pytest.raises(ValueError, match="pinned source, ownership or inputs"):
        plan.validate_for(bp, target)


def test_compound_must_own_every_expression_input(compound):
    bp, target, kwargs, _ = compound
    bp.ownership_data["surfaces"]["issue"]["expression"] += "{unowned.path}"
    with pytest.raises(ValueError, match="owned scalar paths"):
        prepare_country_rendering(bp, target, **kwargs)


def test_compound_requires_single_expression_recipe(compound):
    bp, target, kwargs, _ = compound
    bp.ownership_data["surfaces"]["issue"]["extra"] = "ambiguous"
    with pytest.raises(ValueError, match="one explicit expression recipe"):
        prepare_country_rendering(bp, target, **kwargs)


@pytest.mark.parametrize("expression", [None, 7, []])
def test_compound_expression_must_be_text(compound, expression):
    bp, target, kwargs, _ = compound
    bp.ownership_data["surfaces"]["issue"]["expression"] = expression
    with pytest.raises(ValueError, match="one explicit expression recipe"):
        prepare_country_rendering(bp, target, **kwargs)
