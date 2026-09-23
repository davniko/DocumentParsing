from dataclasses import replace

import pytest

from document_ocr.synthesis.template_compiler import cargo_span_coalescing as spans
from document_ocr.synthesis.template_compiler import host


def fixture(raw="COFFEE\nBEANS\nGRADE 3", parts=("COFFEE", "BEANS", "GRADE 3")):
    target = {"documentPatch": {"cargoGroups": [{"description": "COFFEE BEANS GRADE 3"}]}}
    drafts = tuple(
        host.SpanDraft(
            draft_id=f"part-{i}",
            logical_key="cargo-description",
            render_mode="target_binding",
            value_kind="cargo_text",
            group_kind="cargo",
            group_key="cargo:g1",
            target_paths=("documentPatch.cargoGroups[0].description",),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=raw.index(text),
            char_end=raw.index(text) + len(text),
            source_text=text,
            evidence_origin="accepted_label_evidence",
            render_policy="natural_text",
            rationale="test",
        )
        for i, text in enumerate(parts)
    )
    return dict(raw=raw, drafts=drafts, source_target=target)


def test_adjacent_complete_cargo_value_is_one_slot_with_original_bytes_and_line_breaks():
    args = fixture()
    before = args["drafts"]
    result = spans.normalize(**args)
    assert len(result) == 1
    assert result[0].source_text == args["raw"]
    assert result[0].char_start == 0 and result[0].char_end == len(args["raw"])
    assert result[0].target_paths == before[0].target_paths
    assert result[0].logical_key == before[0].logical_key
    host.validate_draft_source_alignment(raw=args["raw"], drafts=result)
    host.validate_binding_realizations(**{**args, "drafts": result})
    assert spans.normalize(**{**args, "drafts": result}) == result


@pytest.mark.parametrize("gap", [" 25KG ", "\nHS CODE 090111\n", " / ", "\n--- PAGE 2 ---\n"])
def test_coalescing_cannot_absorb_unowned_facts_captions_or_page_boundaries(gap):
    args = fixture(raw="COFFEE" + gap + "BEANS\nGRADE 3")
    assert spans.normalize(**args) == args["drafts"]


def test_partial_scalar_and_distinct_owners_are_not_joined():
    args = fixture()
    args["source_target"]["documentPatch"]["cargoGroups"][0]["description"] += " ROBUSTA"
    assert spans.normalize(**args) == args["drafts"]
    args = fixture()
    args["drafts"] = (
        args["drafts"][0],
        replace(args["drafts"][1], logical_key="different"),
        args["drafts"][2],
    )
    assert spans.normalize(**args) == args["drafts"]


def test_other_owner_inside_whitespace_gap_is_not_absorbed():
    args = fixture()
    marker = replace(
        args["drafts"][0],
        draft_id="marker",
        logical_key="marker",
        char_start=6,
        char_end=7,
        source_text="\n",
        render_mode="literal_static",
        target_paths=(),
    )
    args["drafts"] = (args["drafts"][0], marker, *args["drafts"][1:])
    assert spans.normalize(**args) == args["drafts"]


def test_invalid_coordinates_fail_before_normalization():
    args = fixture()
    args["drafts"] = (replace(args["drafts"][0], source_text="CARGO"), *args["drafts"][1:])
    with pytest.raises(host.DraftSourceAlignmentError):
        spans.normalize(**args)
