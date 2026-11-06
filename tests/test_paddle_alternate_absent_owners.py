import pytest

from document_ocr.synthesis.paddle_render import present_source_edits


def test_exact_alternate_dispositions_do_not_drop_unknown_or_partial_owners():
    composite = {"key": "rows", "byteStart": 10, "before": "10 PALLETS"}
    quantity = {"key": "quantity", "byteStart": 10, "before": "10"}
    family = {"key": "family", "byteStart": 13, "before": "PALLETS"}
    rule = {
        "alternatives": [[composite], [quantity, family]],
        "reason": "Native quantity/family cells are owned independently.",
    }
    unrelated = {"key": "weight", "byteStart": 30, "before": "22"}
    assert present_source_edits([composite, unrelated], [rule]) == [unrelated]
    assert present_source_edits([quantity, family, unrelated], [rule]) == [unrelated]
    assert present_source_edits(
        [quantity, unrelated], [rule], surface_values={"family": "PALLETS"}
    ) == [unrelated]
    for edits in (
        [quantity],
        [family],
        [],
        [composite, quantity],
        [composite, quantity, family],
        [composite, composite],
    ):
        with pytest.raises(ValueError):
            present_source_edits(edits, [rule])
    with pytest.raises(ValueError, match="exact reviewed"):
        present_source_edits([composite], [{**rule, "optional": True}])
