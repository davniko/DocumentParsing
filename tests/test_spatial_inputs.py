"""Text preservation, ambiguity rejection and page-local spatial alignment."""

from dataclasses import replace

import pytest

from document_ocr.spatial_inputs.alignment import (
    AlignmentPolicy,
    Region,
    align_page,
    centroid,
    enclosing_box,
    enrich,
    parse_lines,
    verify_preservation,
)


@pytest.fixture
def policy():
    return AlignmentPolicy(
        grid_size=1000,
        horizontal_gap_heights=8,
        vertical_gap_heights=3,
        vertical_overlap_tolerance=0.5,
        context_lines=2,
        context_radius_heights=6,
        context_margin_heights=2,
        left_alignment_weight=0.25,
        max_group_regions=32,
        max_search_states=10000,
    )


def regions(*specs):
    return [
        Region(i, text, (x, y, x + width, y + 10), 0.99)
        for i, (text, x, y, width) in enumerate(specs)
    ]


def align(text, boxes, policy):
    lines = parse_lines("--- PAGE 1 ---\n" + text, 1)
    return align_page(lines, boxes, 1000, 1000, policy)


def test_exact_and_punctuation_matches_preserve_original_text(policy):
    text = "--- PAGE 1 ---\r\n\r\nCompany LTD.  \r\n123, Main Street\r\nUNKNOWN"
    boxes = regions(("COMPANY LTD.", 10, 20, 100), ("123 Main Street", 10, 40, 150))
    lines = parse_lines(text, 1)
    matches = align_page(lines, boxes, 1000, 1000, policy)
    assert [m.method for m in matches] == [
        "exact_unique",
        "punctuation_unique",
        "no_complete_text_match",
    ]
    coords = {
        line.number: centroid(enclosing_box(boxes, m), 1000, 1000, 1000) if m.regions else None
        for line, m in zip(lines, matches, strict=True)
    }
    enriched = enrich(text, coords)
    assert "Company LTD.   || 60,25\r\n" in enriched
    assert enriched.endswith("UNKNOWN ||")
    verify_preservation(text, enriched, set(coords))
    with pytest.raises(ValueError, match="text edit"):
        verify_preservation(text, enriched.replace("123,", "124,"), set(coords))


def test_spatial_group_ignores_interleaved_other_column(policy):
    boxes = regions(
        ("STREET ONE", 10, 20, 100), ("UNRELATED", 600, 25, 100), ("CITY TWO", 10, 40, 100)
    )
    match = align("STREET ONE CITY TWO", boxes, policy)[0]
    assert match.method == "spatial_group"
    assert match.regions == (0, 2)
    assert enclosing_box(boxes, match) == (10, 20, 110, 50)


def test_identical_groups_and_far_apart_fragments_are_unresolved(policy):
    boxes = regions(
        ("ONE", 10, 20, 100), ("TWO", 10, 40, 100), ("ONE", 600, 20, 100), ("TWO", 600, 40, 100)
    )
    assert align("ONE TWO", boxes, policy)[0].method == "ambiguous_group"
    far = regions(("ONE", 10, 20, 100), ("TWO", 600, 700, 100))
    assert not align("ONE TWO", far, policy)[0].regions


def test_repeated_locations_use_same_block_anchors_not_first_occurrence(policy):
    boxes = regions(
        ("RECEIPT", 10, 20, 100),
        ("LOADING", 500, 20, 100),
        ("CITY", 500, 40, 100),
        ("CITY", 10, 40, 100),
    )
    matches = align("RECEIPT\nCITY\n\nLOADING\nCITY", boxes, policy)
    assert matches[1].regions == (3,)
    assert matches[3].regions == (2,)
    assert matches[1].method == matches[3].method == "repeated_context"
    disconnected = align("RECEIPT\n\nCITY", boxes, policy)
    assert not disconnected[1].regions


def test_one_detected_occurrence_cannot_supply_two_ocr_occurrences(policy):
    boxes = regions(("CITY", 10, 20, 100))
    assert all(not m.regions for m in align("CITY\nCITY", boxes, policy))


def test_containing_box_and_near_identifiers_are_not_exact_positions(policy):
    boxes = regions(("SEAL ABC1234568", 10, 20, 150), ("DATE 2023-01-01", 10, 40, 150))
    matches = align("SEAL ABC1234567\n2023-01-01", boxes, policy)
    assert matches[0].method == "no_complete_text_match"
    assert matches[1].method == "only_partial_region"
    assert all(not m.regions for m in matches)


def test_region_reuse_rejects_single_and_group_claims(policy):
    boxes = regions(("ONE", 10, 20, 100), ("TWO", 10, 40, 100))
    matches = align("ONE\nONE TWO", boxes, policy)
    assert [m.method for m in matches] == ["conflicting_region_claims"] * 2


def test_search_exhaustion_is_not_silent_acceptance(policy):
    policy = policy.model_copy(update={"max_search_states": 1})
    boxes = regions(("ONE", 10, 20, 100), ("TWO", 10, 40, 100))
    assert align("ONE TWO", boxes, policy)[0].method == "group_search_limit"


def test_local_coordinates_are_resolution_invariant_and_page_local(policy):
    a = (10.0, 20.0, 110.0, 30.0)
    assert centroid(a, 1000, 1000, 1000) == centroid(tuple(v * 2 for v in a), 2000, 2000, 1000)
    lines = parse_lines("--- PAGE 1 ---\nCITY\n--- PAGE 2 ---\nCITY", 2)
    boxes = regions(("CITY", 10, 20, 100))
    with pytest.raises(ValueError, match="page-local"):
        align_page(lines, boxes, 1000, 1000, policy)
    for line in lines:
        assert align_page([line], boxes, 1000, 1000, policy)[0].regions == (0,)


@pytest.mark.parametrize(
    "text,pages",
    [
        ("CITY", 1),
        ("--- PAGE 2 ---\nCITY", 2),
        ("--- PAGE 1 ---\nCITY", 2),
        ("--- PAGE 1 ---\nCITY || 1,2", 1),
    ],
)
def test_invalid_page_or_existing_enrichment_fails(text, pages):
    with pytest.raises(ValueError):
        parse_lines(text, pages)


def test_invalid_geometry_is_rejected(policy):
    boxes = regions(("CITY", 10, 20, 100))
    for bad in [replace(boxes[0], box=(-1, 0, 3, 4)), replace(boxes[0], score=float("nan"))]:
        with pytest.raises(ValueError):
            align("CITY", [bad], policy)
