from __future__ import annotations

import pytest

from raw_text_template_experiment.analysis import _full_corpus_document_count


def test_full_corpus_projection_count_comes_from_committed_run_config() -> None:
    assert _full_corpus_document_count({"inputs": {"source_corpus": {"records": 2174}}}) == 2174


@pytest.mark.parametrize("records", (None, True, 0, -1, "2174"))
def test_full_corpus_projection_count_fails_closed(records: object) -> None:
    with pytest.raises(
        ValueError,
        match="committed run config lacks a positive source-corpus record count",
    ):
        _full_corpus_document_count({"inputs": {"source_corpus": {"records": records}}})
