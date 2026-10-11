"""Date-order policy and real synthesis renderer regression checks."""

from datetime import date

import pytest

from document_ocr.date_policy import render_numeric_date, resolve_numeric_date


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("05/07/2024", "2024-07-05"),
        ("05.07.24", "2024-07-05"),
        ("2024/05/07", "2024-05-07"),
        ("04/21/2024", "2024-04-21"),
        ("21/04/2024", "2024-04-21"),
        ("05/05/2024", "2024-05-05"),
    ],
)
def test_numeric_default(surface, expected):
    assert resolve_numeric_date(surface).isoformat() == expected


def test_explicit_and_same_convention_evidence_take_precedence():
    assert resolve_numeric_date("05/07/2024", explicit_order="MDY") == date(2024, 5, 7)
    assert resolve_numeric_date("05/07/2024", same_convention_witnesses=["04/21/2024"]) == date(
        2024, 5, 7
    )
    # A differently structured ISO witness does not establish MDY ordering.
    assert resolve_numeric_date("05/07/2024", same_convention_witnesses=["2024/04/21"]) == date(
        2024, 7, 5
    )


@pytest.mark.parametrize("surface", ["31/02/2024", "13/13/2024", "2024/02/30"])
def test_impossible_date_is_not_a_fallback(surface):
    with pytest.raises(ValueError, match="impossible"):
        resolve_numeric_date(surface)


def test_conflicting_convention_is_an_error():
    with pytest.raises(ValueError, match="conflicting"):
        resolve_numeric_date("05/07/2024", same_convention_witnesses=["04/21/2024", "21/04/2024"])
    with pytest.raises(ValueError, match="contradicts"):
        resolve_numeric_date("21/04/2024", explicit_order="MDY")


def test_synthetic_month_first_history_does_not_create_ambiguous_descendants():
    assert render_numeric_date("06/28/2024", date(2024, 6, 28), date(2025, 5, 8)) == "08/05/2025"
    assert render_numeric_date("05/05/2024", date(2024, 5, 5), date(2025, 11, 13)) == "13/11/2025"
    assert render_numeric_date("2024/05/05", date(2024, 5, 5), date(2025, 11, 13)) == "2025/11/13"


def test_actual_synthesis_date_renderer_obeys_policy():
    from document_ocr.synthesis.template_compiler.descendant import (
        _render_certified_date_surface,
        _render_date_surface,
    )

    for renderer in (_render_date_surface, _render_certified_date_surface):
        assert renderer("05/05/2025", "2025-05-05", "2026-08-19") == "19/08/2026"
        assert renderer("06/28/2024", "2024-06-28", "2025-05-08") == "08/05/2025"
        assert renderer("05/05/2025", "2025-05-05", "2025-05-05") == "05/05/2025"
        assert renderer("28 JUN 2024", "2024-06-28", "2025-05-08") == "08 MAY 2025"


def test_direct_flow_flags_omitted_owned_date_not_unrelated_invoice():
    from document_ocr.labeling_agents.direct_grounding import source_fidelity_findings

    source = "INVOICE DATE 05/07/2024\nSHIPPED ON BOARD: 05/07/2024"
    findings = source_fidelity_findings({"documentPatch": {}}, source, "metadata_freight")
    assert [finding.field for finding in findings] == ["shippedOnBoardDate"]
    assert (
        source_fidelity_findings(
            {"documentPatch": {"shippedOnBoardDate": "2024-07-05"}}, source, "metadata_freight"
        )
        == []
    )


def test_direct_flow_leaves_detached_date_ownership_for_review():
    from document_ocr.labeling_agents.date_grounding import date_fidelity_findings

    assert date_fidelity_findings({}, "Date of Issue\nSHANGHAI\n05/07/2024") == []
    assert date_fidelity_findings({}, "INVOICE DATE 05/07/2024\nCHARTER PARTY 05/07/2024") == []
    assert (
        date_fidelity_findings(
            {"issueDate": "2024-07-08"},
            "LETTER OF CREDIT NO.: 00001ILC24000354\nDATE OF ISSUE: 15.04.2024",
        )
        == []
    )


def test_explicit_printed_marker_overrides_default_in_actual_grounding():
    from document_ocr.labeling_agents.date_grounding import date_fidelity_findings
    from document_ocr.labeling_agents.deterministic_annotation import _date_matches

    source = "Date of Issue of B/L: 05/07/2024\nMonth/Day/Year B/L No."
    assert _date_matches({1: source}, "documentPatch.issueDate", "2024-05-07")
    assert not _date_matches({1: source}, "documentPatch.issueDate", "2024-07-05")
    assert date_fidelity_findings({"issueDate": "2024-05-07"}, source) == []
