import pytest

from document_ocr.synthesis.paddle_render import present_source_edits


def rule(**changes):
    return dict(key="freight", byteStart=20, before="COLLECT",
                reason="Reviewed native OCR omits this specific payment cell.",
                optionalOccurrence=True, **changes)


def edit(**changes):
    return dict(key="freight", byteStart=20, before="COLLECT", after="PREPAID",
                targetPaths=["documentPatch.freight.paymentArrangement"], **changes)


def test_explicit_native_absence_handles_edited_and_unedited_receipt_shapes():
    other = dict(key="freight", byteStart=90, before="COLLECT", after="PREPAID")
    assert present_source_edits([other], [rule()]) == [other]
    assert present_source_edits([edit(), other], [rule()]) == [other]


def test_optional_absence_does_not_discard_conflicting_source_proof():
    with pytest.raises(ValueError, match="conflicts"):
        present_source_edits([{**edit(), "before": "COLLCT"}], [rule()])


def test_duplicate_owners_and_dispositions_still_fail():
    with pytest.raises(ValueError, match="missing or multiply"):
        present_source_edits([edit(), edit()], [rule()])
    with pytest.raises(ValueError, match="multiply disposed"):
        present_source_edits([], [rule(), rule()])


@pytest.mark.parametrize("changes", [
    {"optionalOccurrence": False}, {"optionalOccurrence": 1},
    {"retainedConstant": True}, {"unknown": True},
])
def test_invalid_optional_contracts_fail(changes):
    with pytest.raises(ValueError):
        present_source_edits([], [{**rule(), **changes}])


def test_existing_strict_absence_is_not_weakened():
    strict = rule()
    del strict["optionalOccurrence"]
    with pytest.raises(ValueError, match="missing or multiply"):
        present_source_edits([], [strict])
