"""Declared serial ranges are one count-coupled, anonymized source-only owner."""

from copy import deepcopy
from dataclasses import replace

import pytest

from document_ocr.synthesis.curated import Occurrence, SourceContract, TargetBinding, Variable
from document_ocr.synthesis.curated_auxiliary import (
    _package_range_shape,
    _render_package_range,
    _render_recipe,
    _validate_recipe,
    augment_auxiliary_blueprint,
    auxiliary_surfaces,
    validate_identifier_privacy,
)
from document_ocr.synthesis.curated_templates import (
    RenderRegion,
    SamplingBlueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.generators import DeterministicStream

PATH = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"
SOURCE = "C/NO. Y6P062520 - Y6P062977"
SPEC = {"quantity_path": PATH, "source_quantity": 458}


def stream(seed=1):
    return DeterministicStream(seed, "package-range-test", "source")


@pytest.mark.parametrize("quantity", [1, 18, 458, 2745, 999999])
def test_range_changes_prefix_once_and_couples_end_to_count(quantity):
    for seed in range(25):
        result = _render_package_range(SPEC, SOURCE, quantity, stream(seed))
        part, prefix, width = _package_range_shape(result, quantity)
        assert prefix != "Y6P" and width == 6
        assert part.end - part.start + 1 == quantity
        assert result.startswith("C/NO. ")
        assert len(result) == len(SOURCE)
        assert _render_package_range(SPEC, SOURCE, quantity, stream(seed)) == result
        assert "Y6P062520" not in result and "Y6P062977" not in result


def test_plain_serials_exclude_the_entire_original_domain_without_retrying():
    spec = {"quantity_path": PATH, "source_quantity": 10}
    for seed in range(100):
        result = _render_package_range(spec, "C/NO. 20 - 29", 15, stream(seed))
        part, prefix, width = _package_range_shape(result, 15)
        assert not prefix and width == 2
        assert part.end < 20 or part.start > 29
    with pytest.raises(ValueError, match="no disjoint"):
        _render_package_range(spec, "C/NO. 20 - 29", 99, stream())


@pytest.mark.parametrize("quantity", [0, -1, True, "458", 1.2, 1000000, None])
def test_invalid_or_overwidth_current_quantity_fails_closed(quantity):
    with pytest.raises(ValueError, match="quantity"):
        _render_package_range(SPEC, SOURCE, quantity, stream())


@pytest.mark.parametrize(
    "source",
    [
        "C/NO. Y6P062520 - Z6P062977",
        "C/NO. Y6P062977 - Y6P062520",
        "C/NO. Y6P62520 - Y6P062977",
        "PART123 - PART456 MODEL",
        "C/NO. 062520 - 062977; C/NO. 063000 - 063457",
        "C/NO. 62,520 - 62,977",
    ],
)
def test_nonformal_conflicting_or_unsupported_source_shapes_fail(source):
    with pytest.raises(ValueError, match="package range"):
        _render_package_range(SPEC, source, 20, stream())


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"quantity_path": PATH},
        {"quantity_path": PATH, "source_quantity": True},
        {"quantity_path": PATH, "source_quantity": 0},
        {
            "quantity_path": "documentPatch.goodsItemDetails[0].grossWeight.value",
            "source_quantity": 458,
        },
        {"quantity_path": PATH, "source_quantity": 458, "extra": 1},
    ],
)
def test_recipe_contract_rejects_missing_or_wrong_scope(spec):
    with pytest.raises(ValueError, match="package range"):
        _validate_recipe({"package_range": spec})


def fixture():
    source = "458 CARTONS\n" + SOURCE
    target = {
        "documentPatch": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [{"packageQuantity": 458}],
                }
            ]
        }
    }
    regions = (
        RenderRegion("quantity", 0, 0, 3, "458", (PATH,), "count", "quantity", "number", "458"),
        RenderRegion("marks", 0, 12, len(source), SOURCE, (), "identifier"),
    )
    bindings = {
        r.key: {
            "target_paths": list(r.target_paths),
            "occurrences": [{"source_text": r.source, "byte_start": r.start, "byte_end": r.end}],
        }
        for r in regions
    }
    bp = SamplingBlueprint(
        "doc_range",
        source,
        target,
        SourceContract(
            variables=[
                Variable(
                    key="quantity",
                    kind="count",
                    value="458",
                    meaning="Declared counted packages",
                    required_literals=[],
                    occurrences=[Occurrence(text="458", occurrence=1, presentation="number")],
                )
            ],
            targets=[TargetBinding(path=PATH, expression="{quantity}")],
            fixed_context="Counted source serial interval",
        ),
        regions,
        bindings,
        {},
        (),
        {"identifier_privacy": {"marks": "private"}},
    )
    declaration = {
        "version": 1,
        "sources": {
            "doc_range": {"bindings": {"marks": {"package_range": deepcopy(SPEC)}}},
        },
    }
    return bp, declaration


def test_real_augmentation_privacy_render_and_replay_preserve_unrelated_text():
    bp, declaration = fixture()
    bp = augment_auxiliary_blueprint(bp, declaration)
    validate_identifier_privacy(bp)
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ] = 22
    surfaces, receipts = auxiliary_surfaces(bp, None, target, stream())
    rendered, actual, proof = render_sampling_blueprint(bp, target, {}, surface_values=surfaces)
    assert rendered.startswith("22 CARTONS\nC/NO. ")
    assert actual == target and len(receipts) == 1
    assert _package_range_shape(rendered.splitlines()[1], 22)
    assert (
        proof and render_sampling_blueprint(bp, target, {}, surface_values=surfaces)[0] == rendered
    )


def test_source_quantity_and_public_owner_must_agree_before_sampling():
    bp, declaration = fixture()
    declaration["sources"]["doc_range"]["bindings"]["marks"]["package_range"]["source_quantity"] = (
        457
    )
    with pytest.raises(ValueError, match="source count"):
        augment_auxiliary_blueprint(bp, declaration)


def test_package_range_recipe_cannot_mask_a_public_leaf_or_duplicate_owner():
    bp, declaration = fixture()
    bp = augment_auxiliary_blueprint(bp, declaration)
    bp = replace(
        bp,
        regions=(
            *bp.regions[:-1],
            RenderRegion("marks", 0, 12, len(bp.source), SOURCE, (PATH,), "identifier"),
        ),
    )
    with pytest.raises(ValueError, match="public target"):
        auxiliary_surfaces(bp, None, bp.target, stream())
    bp, declaration = fixture()
    bp = augment_auxiliary_blueprint(bp, declaration)
    with pytest.raises(ValueError, match="two semantic stages"):
        auxiliary_surfaces(bp, None, bp.target, stream(), existing={"marks": SOURCE})


def test_recipe_uses_the_actual_sampled_count_and_never_writes_a_target():
    bp, _ = fixture()
    before = deepcopy(bp.target)
    result = _render_recipe({"package_range": SPEC}, None, bp.target, stream(), SOURCE, set())
    assert _package_range_shape(result, 458)
    assert bp.target == before
    with pytest.raises(ValueError, match="quantity"):
        _render_recipe(
            {"package_range": SPEC}, None, {"documentPatch": {}}, stream(), SOURCE, set()
        )
