"""Source-only customs references preserve their reviewed numeric sequence."""

from copy import deepcopy

import pytest

from document_ocr.synthesis.curated_auxiliary import (
    _source_format_identifier,
    _validate_recipe,
)
from document_ocr.synthesis.generators import DeterministicStream


def recipe(offset=0, *, identity="5464747", length=7):
    return {
        "identifier": "private customs/shipping mark reference",
        "format": "source",
        "canonical_identity": identity,
        "sequence_length": length,
        "sequence_offset": offset,
    }


def render(source, declaration, *, seed=20261020):
    _validate_recipe(declaration)
    return _source_format_identifier(
        declaration,
        source,
        DeterministicStream(seed, "sequence-test", "source"),
    )


@pytest.mark.parametrize("seed", range(100))
def test_full_sequence_replays_is_consecutive_and_excludes_all_old_members(seed):
    old = [str(n) for n in range(5464747, 5464754)]
    new = [render(value, recipe(i), seed=seed) for i, value in enumerate(old)]
    assert not set(old) & set(new)
    assert all(len(value) == 7 and value.isdecimal() for value in new)
    assert [int(value) for value in new] == list(range(int(new[0]), int(new[0]) + 7))
    assert new == [render(value, recipe(i), seed=seed) for i, value in enumerate(old)]


@pytest.mark.parametrize(
    ("source", "identity", "length"),
    [("1000000", "1000000", 7), ("9999993", "9999993", 7), ("10", "10", 1), ("90", "90", 10)],
)
def test_sequence_bounds_keep_the_complete_result_inside_source_width(source, identity, length):
    for seed in range(20):
        new = render(source, recipe(identity=identity, length=length), seed=seed)
        assert len(new) == len(source)
        assert int(new) + length - 1 < 10 ** len(source)
        assert not set(range(int(identity), int(identity) + length)) & set(
            range(int(new), int(new) + length)
        )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("sequence_length", None),
        ("sequence_length", 0),
        ("sequence_length", True),
        ("sequence_length", 1.5),
        ("sequence_offset", None),
        ("sequence_offset", -1),
        ("sequence_offset", 7),
        ("sequence_offset", True),
        ("canonical_identity", "0546474"),
        ("canonical_identity", "ABC1234"),
        ("format", None),
    ],
)
def test_invalid_sequence_options_fail_closed(option, value):
    declaration = recipe()
    declaration[option] = value
    with pytest.raises(ValueError, match="sequence"):
        _validate_recipe(declaration)


@pytest.mark.parametrize("missing", ["sequence_offset", "sequence_length", "canonical_identity"])
def test_partial_sequence_declaration_is_not_silently_ignored(missing):
    declaration = recipe()
    del declaration[missing]
    with pytest.raises(ValueError, match="sequence"):
        _validate_recipe(declaration)


def test_sequence_does_not_cross_source_width_and_has_no_fallback_when_exhausted():
    with pytest.raises(ValueError, match="sequence"):
        _validate_recipe(recipe(identity="9999997", length=7))
    with pytest.raises(ValueError, match="no disjoint"):
        render("1", recipe(identity="1", length=9))


def test_source_must_match_declared_offset_not_just_share_a_width():
    with pytest.raises(ValueError, match="declared offset"):
        render("5464749", recipe(1))


def test_punctuation_aliases_and_embedded_reference_preserve_literal_format():
    declaration = recipe(2)
    expected = render("5464749", declaration)
    assert render("546-4749", declaration) == expected[:3] + "-" + expected[3:]
    embedded = {**declaration, "source_value": "5464749"}
    assert render("CUSTOMS 5464749 ONLY", embedded) == "CUSTOMS " + expected + " ONLY"
    assert declaration == recipe(2)


def test_sequence_options_are_not_accepted_on_text_recipes():
    with pytest.raises(ValueError, match="unsupported auxiliary recipe options"):
        _validate_recipe({"text": "5464747", "sequence_length": 7, "sequence_offset": 0})


def test_ordinary_identifier_output_and_recipe_remain_unchanged():
    declaration = {"identifier": "private reference", "format": "source"}
    saved = deepcopy(declaration)
    value = _source_format_identifier(
        declaration,
        "AB-1234567",
        DeterministicStream(20261020, "marks-benchmark", "source"),
    )
    assert value == "WB-6215547"
    assert declaration == saved


@pytest.mark.parametrize("rank", [0, 4464741, 8999980])
def test_rank_mapping_avoids_excluded_interval_at_domain_edges(rank):
    class RankStream:
        def derive(self, _key):
            return self

        def randbelow(self, upper):
            assert 0 <= rank < upper
            return rank

    values = [
        _source_format_identifier(recipe(i), str(5464747 + i), RankStream()) for i in range(7)
    ]
    assert len(set(values)) == 7
    assert not set(values) & {str(v) for v in range(5464747, 5464754)}
    assert len(values[0]) == len(values[-1]) == 7
    assert int(values[-1]) == int(values[0]) + 6
