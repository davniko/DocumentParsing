import pytest

from document_ocr.synthesis.paddle_render import present_source_edits


def rule():
    return {"key": "old-constant", "byteStart": 10, "before": "XIAMEN",
            "reason": "Reviewed phantom third occurrence; actual two places remain owned.",
            "retainedConstant": True}


def edit():
    return {"key": "old-constant", "byteStart": 10, "before": "XIAMEN",
            "after": "XIAMEN", "targetPaths": []}


def test_reviewed_retained_constant_accepts_absent_or_unchanged_receipt():
    other = {**edit(), "key": "actual-loading", "byteStart": 30, "after": "DUBAI"}
    assert present_source_edits([other], [rule()]) == [other]
    assert present_source_edits([edit(), other], [rule()]) == [other]


@pytest.mark.parametrize("change", [
    {"after": "DUBAI"}, {"before": "DUBAI"},
    {"targetPaths": ["documentPatch.route.portOfLoading.name"]},
])
def test_retained_constant_never_drops_sampled_or_targeted_owner(change):
    with pytest.raises(ValueError, match="changed or targeted"):
        present_source_edits([{**edit(), **change}], [rule()])


def test_retained_constant_rejects_unedited_changed_surface():
    with pytest.raises(ValueError, match="different sampled surface"):
        present_source_edits([], [rule()], surface_values={"old-constant": "DUBAI"})


def test_retained_constant_preserves_strict_default_and_duplicate_detection():
    strict = {k: v for k, v in rule().items() if k != "retainedConstant"}
    with pytest.raises(ValueError, match="missing or multiply"):
        present_source_edits([], [strict])
    with pytest.raises(ValueError, match="multiply disposed"):
        present_source_edits([edit()], [rule(), rule()])
    with pytest.raises(ValueError, match="explicitly enabled"):
        present_source_edits([], [{**rule(), "retainedConstant": False}])
