from document_ocr.synthesis.paddle_reflow import foreign_in_new_envelope


def test_joining_native_cells_preserves_existing_adjacent_row_edge_overlap():
    # Measured table: preceding row ends at552; type box starts at551,
    # count starts at552. Joining the two cannot swallow the preceding row.
    own = [(606, 552, 646, 576), (652, 551, 728, 576)]
    assert not foreign_in_new_envelope((606, 551, 728, 576), own, (604, 524, 729, 552))


def test_right_aligned_count_keeps_existing_vertical_overlap_depth():
    assert not foreign_in_new_envelope(
        (565, 579, 645, 603), [(606, 579, 645, 603)], (604, 602, 647, 632)
    )


def test_new_edge_overlap_without_original_contact_still_fails():
    assert foreign_in_new_envelope(
        (565, 579, 645, 603), [(606, 579, 645, 603)], (566, 602, 600, 632)
    )


def test_increased_edge_depth_or_interior_foreign_text_still_fails():
    assert foreign_in_new_envelope(
        (565, 579, 645, 610), [(606, 579, 645, 603)], (604, 602, 647, 632)
    )
    assert foreign_in_new_envelope(
        (10, 10, 100, 100), [(10, 10, 100, 20), (10, 90, 100, 100)], (20, 19, 80, 91)
    )
    assert foreign_in_new_envelope((10, 10, 100, 100), [(10, 10, 30, 100)], (25, 30, 80, 50))
