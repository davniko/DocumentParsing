import pytest

from document_ocr.synthesis.paddle_render import sampled_source_slice

SPEC = {"byteStart": 10, "byteEnd": 22, "source": "SOKHNA,EGYPT"}


def edit(start, before, after):
    return {
        "byteStart": start,
        "byteEnd": start + len(before.encode()),
        "before": before,
        "after": after,
    }


def test_compound_and_split_preserve_sampled_alias_and_separator():
    assert sampled_source_slice(SPEC, [edit(10, "SOKHNA,EGYPT", "DUBAI,ARE")]) == "DUBAI,ARE"
    assert (
        sampled_source_slice(SPEC, [edit(17, "EGYPT", "A.E."), edit(10, "SOKHNA", "DUBAI")])
        == "DUBAI,A.E."
    )


@pytest.mark.parametrize(
    "edits, message",
    [
        ([], "no sampled"),
        ([edit(10, "SOKHNA", "DUBAI")], "unsampled lexical"),
        ([edit(17, "EGYPT", "ARE")], "unsampled lexical"),
        ([edit(9, " SOKHNA,EGYPT", "DUBAI,ARE")], "crossing"),
        ([edit(10, "SOKHNA,EGYPT ", "DUBAI,ARE")], "crossing"),
        ([edit(10, "SOKHNA,EGYPT", "DUBAI,ARE"), edit(17, "EGYPT", "AE")], "overlapping"),
        ([edit(10, "SOKHNA,GHANA", "DUBAI,ARE")], "differs"),
    ],
)
def test_incomplete_or_conflicting_ownership_fails(edits, message):
    with pytest.raises(ValueError, match=message):
        sampled_source_slice(SPEC, edits)


def test_bad_byte_extent_fails():
    with pytest.raises(ValueError, match="byte interval"):
        sampled_source_slice({**SPEC, "byteEnd": 21}, [])
