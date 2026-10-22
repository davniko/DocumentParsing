"""Batch sizing is explicit: never drop or pad the final registry remainder."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "scripts/synthesis/prepare_independent_batch.py"
_SPEC = importlib.util.spec_from_file_location("reviewed_batch", _PATH)
batch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(batch)


@pytest.mark.parametrize("count", [1, 14, 20])
def test_explicit_batch_count_preserves_order(count):
    ids = [f"doc_{index:08x}suffix" for index in range(20)]
    review = {"sources": [{"prefix": sid[4:12]} for sid in reversed(ids[:count])]}
    assert batch.reviewed_source_ids({"source_ids": ids}, review, count) == ids[:count][::-1]


@pytest.mark.parametrize("count", [0, 21])
def test_invalid_batch_count_rejected(count):
    with pytest.raises(ValueError, match="between one and twenty"):
        batch.reviewed_source_ids({"source_ids": []}, {"sources": []}, count)


def test_remainder_does_not_silently_pass_as_full_batch():
    ids = [f"doc_{index:08x}suffix" for index in range(14)]
    review = {"sources": [{"prefix": sid[4:12]} for sid in ids]}
    with pytest.raises(ValueError, match="exactly 20 unique"):
        batch.reviewed_source_ids({"source_ids": ids}, review, 20)


@pytest.mark.parametrize("ids", [["doc_00000000a", "doc_00000000b"], []])
def test_ambiguous_or_unknown_prefix_rejected(ids):
    with pytest.raises(ValueError, match="exactly one admitted"):
        batch.reviewed_source_ids(
            {"source_ids": ids}, {"sources": [{"prefix": "00000000"}]}, 1
        )


def test_duplicate_source_rejected():
    with pytest.raises(ValueError, match="exactly 2 unique"):
        batch.reviewed_source_ids(
            {"source_ids": ["doc_00000000suffix"]},
            {"sources": [{"prefix": "00000000"}, {"prefix": "00000000"}]},
            2,
        )


def test_reviewed_complete_product_span_retains_projection_and_byte_offsets():
    declaration = {"key": "product", "source_text": "25 KG/BAG", "append": "count_unit_net"}
    assert batch.reviewed_span("é\n25 KG/BAG", declaration) == {
        **declaration, "byte_start": 3, "byte_end": 12,
    }


def test_repeated_complete_phrase_requires_explicit_occurrence():
    with pytest.raises(ValueError, match="explicit occurrence"):
        batch.reviewed_span("ONE ONE", {"source_text": "ONE"})
    assert batch.reviewed_span("ONE ONE", {"source_text": "ONE", "occurrence": 2}) == {
        "source_text": "ONE", "byte_start": 4, "byte_end": 7,
    }
