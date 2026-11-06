import pytest

from document_ocr.synthesis.paddle_render import existing_repetition_edits


OLD = {"key": "clause-fragment", "byteStart": 12, "byteEnd": 20}
NEW = {"key": "complete-clause", "byteStart": 0, "byteEnd": 21}


@pytest.mark.parametrize("edits", [[], [{**OLD, "after": "neutral"}], [{**NEW, "after": "neutral"}]])
def test_exact_alternate_owner(edits):
    assert existing_repetition_edits({"existingOwners": [OLD, NEW]}, edits) == edits


def test_does_not_match_other_occurrence_or_wider_interval():
    edits = [{**OLD, "byteEnd": 21}, {**NEW, "byteStart": 1}]
    assert existing_repetition_edits({"existingOwners": [OLD, NEW]}, edits) == []


def test_simultaneous_alternates_fail():
    with pytest.raises(ValueError, match="mutually exclusive"):
        existing_repetition_edits({"existingOwners": [OLD, NEW]}, [OLD, NEW])


@pytest.mark.parametrize("selectors", [[OLD], [OLD, OLD], [{"key": "a"}, NEW], [{**OLD, "byteStart": -1}, NEW]])
def test_malformed_alternates_fail(selectors):
    with pytest.raises(ValueError):
        existing_repetition_edits({"existingOwners": selectors}, [])


def test_no_mixed_selector_modes():
    with pytest.raises(ValueError, match="conflict"):
        existing_repetition_edits({"existingOwners": [OLD, NEW], "existingKey": "a"}, [])


def test_single_key_contract_unchanged():
    assert existing_repetition_edits({"existingKey": OLD["key"]}, [OLD, NEW]) == [OLD]
    assert existing_repetition_edits({"existingKey": OLD["key"], "existingByteStart": 1}, [OLD]) == []
