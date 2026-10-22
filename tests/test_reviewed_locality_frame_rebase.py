"""Mechanical frame rebasing must preserve exact occurrence and role ownership."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).parents[1] / "scripts/synthesis/rebase_reviewed_locality_frames.py"
_SPEC = importlib.util.spec_from_file_location("reviewed_locality_frames", _PATH)
frames = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(frames)

ROLE = "documentPatch.route.portOfLoading"


def blueprint(text="PORT, COUNTRY\nPORT, COUNTRY", gap=", "):
    raw = text.encode()
    regions = []
    for key, word, path in [
        ("name", b"PORT", ROLE + ".name"),
        ("country", b"COUNTRY", ROLE + ".country"),
    ]:
        start = 0
        occurrence = 0
        while (start := raw.find(word, start)) >= 0:
            regions.append(
                SimpleNamespace(
                    key=key,
                    occurrence=occurrence,
                    start=start,
                    end=start + len(word),
                    target_paths=(path,),
                    curated_key=None,
                )
            )
            start += len(word)
            occurrence += 1
    return SimpleNamespace(source=text, regions=regions)


def test_complete_repeated_frames_keep_both_occurrence_indices():
    result = frames.frame_proposals(blueprint())
    assert len(result) == 1
    assert result[0]["occurrences"] == [
        {"text": "PORT, COUNTRY", "occurrence": 1},
        {"text": "PORT, COUNTRY", "occurrence": 2},
    ]
    assert result[0]["expression"] == "{" + ROLE + ".name}, {" + ROLE + ".country}"


def test_partial_key_coverage_cannot_retire_the_unpaired_occurrence():
    with pytest.raises(ValueError, match="complete consistent"):
        frames.frame_proposals(blueprint("PORT, COUNTRY\nPORT\nCOUNTRY"))


def test_separately_captioned_country_is_not_an_inline_frame():
    assert not frames.frame_proposals(blueprint("PORT\nCountry: COUNTRY"))


def test_extra_state_literal_requires_an_exact_approved_gap():
    source = blueprint("PORT, SP, COUNTRY")
    assert not frames.frame_proposals(source)
    assert not frames.frame_proposals(source, {ROLE + ".country": {", ZZ, ": ", "}})
    result = frames.frame_proposals(source, {ROLE + ".country": {", SP, ": ", "}})
    assert result[0]["occurrences"][0]["text"] == "PORT, SP, COUNTRY"
    assert result[0]["expression"] == "{" + ROLE + ".name}, {" + ROLE + ".country}"


def test_explicit_continuation_keeps_its_distinct_line_break_style():
    source = blueprint("PORT, COUNTRY\nPORT,\nCOUNTRY")
    result = frames.frame_proposals(source, {ROLE + ".country": {",\n": ",\n"}})
    assert len(result) == 2
    assert "\n" not in result[0]["expression"]
    assert ",\n" in result[1]["expression"]


def test_utf8_prefix_does_not_shift_the_frame_quote():
    result = frames.frame_proposals(blueprint("ÉTIQUETTE\nPORT, COUNTRY"))
    assert result[0]["occurrences"] == [{"text": "PORT, COUNTRY", "occurrence": 1}]


def test_party_addresses_are_not_rewritten_as_locality_fields():
    source = blueprint("PORT, COUNTRY")
    source.regions[0].target_paths = ("documentPatch.parties.shipper.addressLine",)
    source.regions[1].target_paths = ("documentPatch.parties.shipper.country",)
    assert not frames.frame_proposals(source)


@pytest.mark.parametrize("gap", ["(", " (", ", ("])
def test_parenthesized_country_owns_the_matching_closing_delimiter(gap):
    text = "PORT" + gap + "COUNTRY)"
    result = frames.frame_proposals(blueprint(text))
    assert result[0]["occurrences"] == [{"text": text, "occurrence": 1}]
    assert result[0]["expression"] == "{" + ROLE + ".name}" + gap + "{" + ROLE + ".country})"


def test_missing_closing_delimiter_is_not_invented_and_uses_its_own_style():
    result = frames.frame_proposals(blueprint("PORT(COUNTRY)\nPORT(COUNTRY"))
    assert len(result) == 2
    assert result[0]["occurrences"] == [{"text": "PORT(COUNTRY)", "occurrence": 1}]
    assert result[1]["occurrences"] == [{"text": "PORT(COUNTRY", "occurrence": 2}]
    assert result[0]["expression"].endswith("})")
    assert result[1]["expression"].endswith("}")


def test_frame_rebase_replaces_added_owner_without_retiring_nonhistorical_key():
    proposal = frames.frame_proposals(blueprint("PORT, COUNTRY"))[0]
    before = {
        "add": [{"key": "name", "kind": "location"}, {"key": "unrelated"}],
        "bindings": {"country": {"paths": [ROLE + ".country"]}},
        "surfaces": {"name": {"expression": "old"}, "unrelated": {"expression": "keep"}},
    }
    result = frames.rebase_owner_frames(before, [proposal])
    assert [row["key"] for row in result["add"]] == ["unrelated", proposal["key"]]
    assert set(result["retire"]) == {"country"}
    assert result["bindings"] == {}
    assert "name" not in result["surfaces"]
    assert result["surfaces"]["unrelated"] == {"expression": "keep"}
    assert before["add"][0]["key"] == "name"


def test_frame_styles_retire_historical_owners_only_once():
    proposals = frames.frame_proposals(blueprint("PORT(COUNTRY)\nPORT(COUNTRY"))
    result = frames.rebase_owner_frames({}, proposals)
    assert set(result["retire"]) == {"name", "country"}
    assert len(result["add"]) == 2
