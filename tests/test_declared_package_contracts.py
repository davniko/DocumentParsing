from dataclasses import replace

import pytest
from test_declared_package_rendering import fixture

from document_ocr.synthesis.curated import TargetBinding
from document_ocr.synthesis.declared_package_contracts import (
    PREFIX,
    compile_declarations,
    quantity_scope,
)


def test_direct_source_phrase_compiles_without_changing_source():
    bp, *_ = fixture()
    projection, owners = compile_declarations(bp)
    assert projection == "shipment"
    assert owners == [
        {
            "owners": [["quantity", 0], ["noun", 0]],
            "start": 12,
            "end": 20,
            "source": "10 BOXES",
            "scope": "shipment",
        }
    ]


def test_target_path_on_an_allocated_count_is_not_a_shipment_total():
    bp, *_ = fixture()
    contract = bp.contract.model_copy(
        update={
            "targets": [
                TargetBinding(path=PREFIX + ".packageQuantity", expression="{quantity}+{other}"),
                TargetBinding(
                    path="documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].packageQuantity",
                    expression="{quantity}",
                ),
            ]
        }
    )
    bp = replace(bp, contract=contract)
    assert quantity_scope(bp, bp.regions[1]) == 0


def test_typed_package_auxiliary_owner_has_explicit_scope():
    bp, *_ = fixture()
    number = replace(bp.regions[1], curated_key=None, kind="integer")
    bp = replace(
        bp, historical_bindings={"quantity": {"group_kind": "package", "value_kind": "integer"}}
    )
    assert quantity_scope(bp, number) == "shipment"
    bp = replace(bp, historical_bindings={})
    assert quantity_scope(bp, number) is None


@pytest.mark.parametrize("gap,accepted", [(" x ", True), ("\n\n", False), (" WEIGHT ", False)])
def test_punctuation_is_not_permission_to_swallow_a_field(gap, accepted):
    bp, *_ = fixture()
    source = "10" + gap + "BOXES"
    number = replace(bp.regions[1], start=0, end=2)
    noun = replace(bp.regions[2], start=2 + len(gap), end=len(source))
    bp = replace(bp, source=source, regions=(number, noun))
    if accepted:
        assert compile_declarations(bp)[1][0]["source"] == source
    else:
        with pytest.raises(ValueError, match="no complete"):
            compile_declarations(bp)


def test_parenthesized_word_numeric_repetition_is_one_complete_owner():
    bp, *_ = fixture()
    source = "TEN(10) BOXES"
    word = replace(bp.regions[1], start=0, end=3, source="TEN")
    number = replace(bp.regions[1], start=4, end=6, occurrence=1)
    noun = replace(bp.regions[2], start=8, end=13)
    bp = replace(bp, source=source, regions=(word, number, noun))
    assert compile_declarations(bp)[1][0]["owners"] == [
        ["quantity", 0],
        ["quantity", 1],
        ["noun", 0],
    ]


def test_repeated_next_line_phrase_does_not_steal_its_count():
    bp, *_ = fixture()
    source = "10 BOXES\n(TEN BOXES ONLY)"
    first = replace(bp.regions[1], start=0, end=2, source="10")
    second = replace(first, start=10, end=13, source="TEN", occurrence=1)
    nouns = (
        replace(bp.regions[2], start=3, end=8),
        replace(bp.regions[2], start=14, end=19, occurrence=1),
    )
    bp = replace(bp, source=source, regions=(first, nouns[0], second, nouns[1]))
    _, rows = compile_declarations(bp)
    assert [row["source"] for row in rows] == ["10 BOXES", "TEN BOXES"]
    assert len({tuple(owner) for row in rows for owner in row["owners"]}) == 4


def test_previous_line_serial_endpoint_is_not_part_of_package_declaration():
    bp, *_ = fixture()
    source = "10\n10 BOXES"
    first = replace(bp.regions[1], start=0, end=2)
    second = replace(first, start=3, end=5, occurrence=1)
    noun = replace(bp.regions[2], start=6, end=11)
    bp = replace(bp, source=source, regions=(first, second, noun))
    assert compile_declarations(bp)[1][0]["source"] == "10 BOXES"
