from __future__ import annotations

import pytest

from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    normalize_original_bill_count_ownership,
    validate_draft_source_alignment,
)
from document_ocr.synthesis.template_compiler.original_bill_counts import parse_count


def _draft(raw: str, text: str, *, key: str, offset: int = 0) -> SpanDraft:
    start = raw.index(text, offset)
    return SpanDraft(
        draft_id=f"source_{start}",
        logical_key=key,
        render_mode="agent_residual",
        value_kind="legal_text",
        group_kind="legal",
        group_key="legal:negotiability",
        target_paths=("documentPatch.negotiability",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        evidence_origin="agent_proposal",
        render_policy="natural_text",
        rationale="Source-printed legal status/count selected by compiler.",
    )


@pytest.mark.parametrize("surface", ["THREE(2)", "THREE (4)", "3", "THREE(30)"])
def test_original_bill_count_requires_consistent_printed_word_and_digit(surface: str) -> None:
    with pytest.raises(ValueError, match="consistent word-and-digit"):
        parse_count(surface)


def test_captioned_original_count_is_not_negotiability_even_when_compiler_grouped_both() -> None:
    raw = "NON-NEGOTIABLE COPY\nNumber of Original B(s)/L\nTHREE(3)\n"
    drafts = (
        _draft(raw, "NON-NEGOTIABLE COPY", key="agent:negotiability"),
        _draft(raw, "THREE(3)", key="agent:negotiability"),
    )

    result = normalize_original_bill_count_ownership(
        raw=raw,
        drafts=drafts,
        source_target={"documentPatch": {"negotiability": "non_negotiable"}},
    )

    validate_draft_source_alignment(raw=raw, drafts=result)
    count = next(row for row in result if row.source_text == "THREE(3)")
    status = next(row for row in result if row.source_text == "NON-NEGOTIABLE COPY")
    assert count.logical_key == "aux:document:original_bill_count"
    assert count.value_kind == "original_bill_count"
    assert count.target_paths == ()
    assert count.render_mode == "deterministic_auxiliary"
    assert status.target_paths == ("documentPatch.negotiability",)
    assert status.logical_key == "agent:document_negotiability_status"
    assert status.group_key == "legal:negotiability"


def test_captioned_original_count_requires_separate_printed_status_owner() -> None:
    raw = "NON-NEGOTIABLE COPY\nNumber of Original B(s)/L\nTHREE(3)\n"

    result = normalize_original_bill_count_ownership(
        raw=raw,
        drafts=(_draft(raw, "THREE(3)", key="agent:count_as_negotiability"),),
        source_target={"documentPatch": {"negotiability": "non_negotiable"}},
    )

    validate_draft_source_alignment(raw=raw, drafts=result)
    assert {row.source_text for row in result} == {"NON-NEGOTIABLE COPY", "THREE(3)"}
    status = next(row for row in result if row.source_text == "NON-NEGOTIABLE COPY")
    assert status.target_paths == ("documentPatch.negotiability",)


def test_unheaded_number_does_not_trigger_original_bill_count_rewrite() -> None:
    raw = "NON-NEGOTIABLE COPY\nTHREE(3)\n"
    draft = _draft(raw, "THREE(3)", key="agent:count_as_negotiability")

    assert normalize_original_bill_count_ownership(
        raw=raw,
        drafts=(draft,),
        source_target={"documentPatch": {"negotiability": "non_negotiable"}},
    ) == (draft,)
