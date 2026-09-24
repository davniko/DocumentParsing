"""Document date facts must be supported by their OCR role, not a nearby date."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from document_ocr.date_evidence import date_immediately_under_declared_value
from document_ocr.labeling_agents.deterministic_annotation import (
    DeterministicAnnotationError,
    _best_match,
)
from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    validate_compiled_extraction_dates,
    validate_compiler_extraction_dates,
)

PATH = "documentPatch.shippedOnBoardDate"


def _draft(raw: str, value: str, path: str = PATH) -> SpanDraft:
    start = raw.index(value)
    return SpanDraft(
        draft_id="date",
        logical_key="date",
        render_mode="target_binding",
        value_kind="date",
        group_kind="document",
        group_key="document:date",
        target_paths=(path,),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(value),
        source_text=value,
        evidence_origin="accepted_label_evidence",
        render_policy="date_surface",
        rationale="Printed date evidence.",
    )


def test_date_role_guard_rejects_declared_value_but_keeps_explicit_date() -> None:
    raw = "Date of Issue of B/L\n2024-05-14\nDeclared Value\n2024-05-14\n"
    first = raw.index("2024-05-14")
    second = raw.rindex("2024-05-14")
    assert not date_immediately_under_declared_value(raw, first)
    assert date_immediately_under_declared_value(raw, second)

    issue = "documentPatch.issueDate"
    target = {"documentPatch": {"issueDate": "2024-05-14"}}
    validate_compiler_extraction_dates(
        raw=raw,
        source_target=target,
        drafts=(_draft(raw, "2024-05-14", issue),),
        semantic_only_target_facts=(),
    )
    wrong = SimpleNamespace(
        target_paths=(issue,),
        occurrences=(SimpleNamespace(byte_start=second),),
    )
    with pytest.raises(ValueError, match="under Declared Value"):
        validate_compiled_extraction_dates(
            raw=raw,
            source_target=target,
            template=SimpleNamespace(bindings=(wrong,), semantic_only_target_facts=()),
        )


def test_unprinted_date_cannot_become_a_semantic_only_extraction_fact() -> None:
    raw = "Date of Issue of B/L\n2024-05-14\nDeclared Value\n2024-05-12\n"
    target = {"documentPatch": {"shippedOnBoardDate": "2024-05-12"}}
    semantic = (SimpleNamespace(target_path=PATH),)
    with pytest.raises(ValueError, match="semantic-only"):
        validate_compiler_extraction_dates(
            raw=raw,
            source_target=target,
            drafts=(),
            semantic_only_target_facts=semantic,
        )
    with pytest.raises(ValueError, match="semantic-only"):
        validate_compiled_extraction_dates(
            raw=raw,
            source_target=target,
            template=SimpleNamespace(bindings=(), semantic_only_target_facts=semantic),
        )


def test_source_label_grounding_rejects_wrong_role_without_losing_valid_duplicate() -> None:
    wrong = "Declared Value (see clause 7.3)\n2024-05-14\n"
    with pytest.raises(DeterministicAnnotationError, match="Declared Value caption"):
        _best_match({1: wrong}, PATH, "2024-05-14")

    raw = "Shipped on Board Date\n2024-05-14\n" + wrong
    matched = _best_match({1: raw}, PATH, "2024-05-14")
    assert matched.start == raw.index("2024-05-14")


@pytest.mark.parametrize(
    ("printed", "target"),
    (
        ("17TH day of ___ MARCH 2024", "2024-03-17"),
        ("17- th June, 2023", "2023-06-17"),
    ),
)
def test_printed_ordinal_issue_dates_are_grounded(printed: str, target: str) -> None:
    raw = f"Date of Issue\n{printed}\n"
    match = _best_match({1: raw}, "documentPatch.issueDate", target)
    assert raw[match.start : match.end] == printed


def test_valid_ordinal_date_survives_a_wrong_role_duplicate() -> None:
    raw = (
        "Declared Value\n2024-03-17\n"
        "Date of Issue\n17TH day of ___ MARCH 2024\n"
    )
    match = _best_match({1: raw}, "documentPatch.issueDate", "2024-03-17")
    assert raw[match.start : match.end] == "17TH day of ___ MARCH 2024"
