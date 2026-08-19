from __future__ import annotations

from document_ocr.quality_filter import (
    PageQualityEvidence,
    assess_document_quality,
    scan_unicode_quality,
)


def _page(
    page_index: int,
    *,
    status: str = "success",
    finish_reason: str | None = "stop",
    text: str | None = "",
) -> PageQualityEvidence:
    return PageQualityEvidence(
        extraction_id=f"extract-{page_index}",
        page_id=f"page-{page_index}",
        page_index=page_index,
        page_number=page_index + 1,
        status=status,  # type: ignore[arg-type]
        finish_reason=finish_reason,  # type: ignore[arg-type]
        raw_ocr_text=text,
        failure={"error_type": "FixtureError"} if status == "failed" else None,
    )


def test_unicode_filter_allows_extended_latin_but_rejects_other_scripts() -> None:
    allowed = scan_unicode_quality("Café İstanbul São Tomé 12º")
    rejected = scan_unicode_quality("Arabic: العربية Han: 中文 Cyrillic: Б")

    assert allowed.contains_non_latin_letters is False
    assert allowed.allowed_non_ascii_latin_occurrences == 5
    assert {row["character"] for row in allowed.allowed_non_ascii_latin_characters} == {
        "ã",
        "é",
        "İ",
        "º",
    }
    assert rejected.contains_non_latin_letters is True
    assert set(rejected.non_latin_occurrences_by_script) == {"Arabic", "Cyrillic", "Han"}


def test_document_assessment_requires_complete_ordered_normal_pages() -> None:
    result = assess_document_quality(
        document_status="complete",
        expected_page_count=2,
        pages=(_page(0, text="BILL OF LADING Café"), _page(1, text="SECOND PAGE")),
    )

    assert result.eligible is True
    assert result.reasons == ()


def test_document_assessment_reports_all_overlapping_exclusion_reasons() -> None:
    result = assess_document_quality(
        document_status="incomplete",
        expected_page_count=2,
        pages=(
            _page(0, finish_reason="repetition", text="PORT PORT PORT العربية"),
            _page(1, status="failed", finish_reason=None, text=None),
        ),
    )

    assert result.eligible is False
    assert [reason["code"] for reason in result.reasons] == [
        "document_not_complete",
        "ocr_repetition_detected",
        "non_latin_script_in_raw_ocr",
    ]


def test_missing_registered_page_excludes_whole_document() -> None:
    result = assess_document_quality(
        document_status="incomplete",
        expected_page_count=2,
        pages=(_page(0, text="ONLY PAGE"),),
    )

    assert result.eligible is False
    assert result.reasons[0]["missing_page_indexes"] == [1]
