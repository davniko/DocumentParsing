from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.range_generation import (
    _apportion,
    condition_lexical_ranges,
    formal_range_text,
    plan_ranges,
    render_composite_range,
)


def _template(texts, contracts, *, target_backed=True):
    bindings = [
        NS(
            logical_key=f"b{i}",
            target_paths=(f"documentPatch.marks[{i}]",) if target_backed else (),
            occurrences=[NS(slot_id=f"s{i}", source_text=text)],
        )
        for i, text in enumerate(texts)
    ]
    constraints = [
        NS(
            kind=kind,
            member_logical_keys=tuple(f"b{i}" for i in members),
            dependency_paths=tuple(f"documentPatch.counts[{i}]" for i in dependencies),
        )
        for kind, members, dependencies in contracts
    ]
    return NS(bindings=bindings, coherence_constraints=constraints)


def test_composite_range_uses_exact_static_caption_and_typed_cardinality():
    source = {
        "documentPatch": {
            "cargoGroups": [{"marksAndNumbers": ["PALLET NO:1-8"]}],
            "cargoPackages": [{"quantity": 8, "typeCategory": "PACKAGE_PALLET"}],
            "cargoAllocationGroups": [{"allocations": [{"packageQuantity": 8}]}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["PALLET NO:1-5"]
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 5
    target["documentPatch"]["cargoAllocationGroups"][0]["allocations"][0]["packageQuantity"] = 5
    binding = NS(
        target_paths=(
            "documentPatch.cargoGroups[0].marksAndNumbers[0]",
            "documentPatch.cargoPackages[0].quantity",
            "documentPatch.cargoPackages[0].typeCategory",
            "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
        ),
        occurrences=[NS(slot_id="range", source_text="1-8", byte_start=10, byte_end=13)],
    )
    raw = b"PALLET NO:1-8"
    assert render_composite_range(binding, source, target, raw) == {"range": "1-5"}
    assert render_composite_range(binding, source, target, b"OTHER  NO:1-8") is None
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 6
    with pytest.raises(ValueError, match="cardinality"):
        render_composite_range(binding, source, target, raw)


@pytest.mark.parametrize("quantity", [2, 631, 916, 1225])
def test_identity_range_marks_keep_arithmetic_out_of_linguistic_generation(quantity):
    from document_ocr.synthesis.template_compiler.coherence import inclusive_range_surfaces
    from document_ocr.synthesis.template_compiler.complete_targets import assemble_lexical_value

    text = "VARDHMAN C/NO. Y6P062062 - Y6P062519 Y6P055274 - Y6P055731"
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    source = {
        "documentPatch": {
            "cargoGroups": [{"marksAndNumbers": [text]}],
            "cargoPackages": [{"quantity": 916}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = quantity
    template = NS(
        bindings=[NS(logical_key="mark", target_paths=[path], occurrences=[NS(source_text=text)])],
        coherence_constraints=[
            NS(
                kind="aggregate_inclusive_range_cardinality",
                member_logical_keys=["mark"],
                dependency_paths=["documentPatch.cargoPackages[0].quantity"],
            )
        ],
    )
    field = dict(key="f", paths=[path], source=text, constraints=[dict(minimumWords=1)])
    conditioned = condition_lexical_ranges(template, source, target, [field])[0]
    assert conditioned["hostAssembly"]["mutableSource"] == "VARDHMAN"
    assembled = assemble_lexical_value(conditioned, "NEWBRAND")
    assert assembled.startswith("NEWBRAND C/NO. ")
    assert sum(r.cardinality for r in inclusive_range_surfaces(assembled)) == quantity
    assert field.get("hostAssembly") is None
    assert "CARTON" not in assembled and "PACKAGE" not in assembled


@pytest.mark.parametrize(
    "text,expected",
    [
        ("TO THE ORDER OF NEW BANK", True),
        ("TO ORDER", True),
        ("NON-NEGOTIABLE", False),
        ("NOT NEGOTIABLE", False),
        ("NON NEGOTIABLE BILL", False),
        ("UNRELATED WORDS", False),
    ],
)
def test_order_wording_does_not_confuse_positive_and_negative_negotiability(text, expected):
    from document_ocr.synthesis.template_compiler.descendant import _string_semantics_match

    assert _string_semantics_match("negotiable", text) is expected


def test_nested_range_arithmetic_preserves_fixed_child_and_whole_total():
    source = dict(marks=["1-52", "1-111", "1-1"], counts=[52, 164])
    target = dict(marks=source["marks"], counts=[40, 130])
    template = _template(
        source["marks"],
        [
            ("inclusive_range_cardinality", [0], [0]),
            ("aggregate_inclusive_range_cardinality", [0, 1, 2], [1]),
        ],
    )
    before = deepcopy(target)
    result = plan_ranges(template, {"documentPatch": source}, {"documentPatch": target})
    assert result.target_values == {
        "documentPatch.marks[0]": "1-40",
        "documentPatch.marks[1]": "1-089",
        "documentPatch.marks[2]": "1-1",
    }
    assert target == before  # Planning never changes the accepted target.


def test_source_only_repeated_ranges_are_one_fact_not_multiple_cardinalities():
    source = dict(marks=["1-13", "1-2"], counts=[15])
    template = _template(
        source["marks"],
        [
            ("aggregate_inclusive_range_cardinality", [0, 1], [0]),
        ],
        target_backed=False,
    )
    template.bindings[0].occurrences.append(NS(slot_id="repeated", source_text="1-13"))
    result = plan_ranges(template, {"documentPatch": source}, {"documentPatch": dict(counts=[10])})
    assert result.auxiliary_surfaces == {
        "b0": {"s0": "1-08", "repeated": "1-08"},
        "b1": {"s1": "1-2"},
    }


def test_crossing_range_ownership_requires_review_instead_of_arithmetic_guess():
    source = dict(marks=["1-2", "1-2", "1-2"], counts=[4, 4])
    template = _template(
        source["marks"],
        [
            ("aggregate_inclusive_range_cardinality", [0, 1], [0]),
            ("aggregate_inclusive_range_cardinality", [1, 2], [1]),
        ],
    )
    with pytest.raises(ValueError, match="crossing"):
        plan_ranges(template, {"documentPatch": source}, {"documentPatch": source})


def test_formal_captions_do_not_include_party_dependent_brand_words():
    assert formal_range_text("PALLET NO:1-11")
    assert formal_range_text("C/NO.Q4N053335 - Q4N053806")
    assert not formal_range_text("VARDHMAN C/NO. Y6P062062 - Y6P062519")


def test_range_support_and_source_arithmetic_are_mandatory():
    with pytest.raises(ValueError, match="positive source support"):
        _apportion(1, [10, 2])
    source = dict(marks=["1-10"], counts=[11])
    template = _template(source["marks"], [("inclusive_range_cardinality", [0], [0])])
    with pytest.raises(ValueError, match="source arithmetic"):
        plan_ranges(template, {"documentPatch": source}, {"documentPatch": source})


def test_apportion_exhaustively_preserves_positive_support_and_capacity():
    for weights in ([1], [1, 1], [13, 2], [7, 3, 15], [1, 2, 1, 5]):
        for total in range(len(weights), sum(weights) + 1):
            values = _apportion(total, weights)
            assert sum(values) == total
            assert all(1 <= value <= old for value, old in zip(values, weights, strict=True))
