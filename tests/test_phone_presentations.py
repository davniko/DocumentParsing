import re

import phonenumbers
import pytest

from document_ocr.synthesis.phone_presentations import (
    PhonePresentationError,
    format_phone_presentation,
)


def present(
    source, *, surfaces=None, identity="+12025550123", country="US", old_country="EG", **kwargs
):
    return format_phone_presentation(
        source_value=source,
        source_surfaces=[source] if surfaces is None else surfaces,
        replacement_identity=identity,
        target_country_code=country,
        source_country_code=old_country,
        **kwargs,
    )


@pytest.mark.parametrize(
    "source,old_country,family",
    [
        ("+20 3 481 4958", "EG", "plus"),
        ("002 03 3991050", "EG", "double_zero"),
        ("(202)2180-0208", "EG", "international_bare"),
        ("732-640-2367", "US", "national_bare"),
        ("051/50.31.96", "IT", "national"),
        ("{+}20 2 24801010", "EG", "plus"),
        ("+2/ 01228589353", "EG", "plus"),
        ("+2(03)4853366", "EG", "plus"),
        ("020~6~23597220", "EG", "national"),
    ],
)
def test_source_family_preserves_all_identity_digits(source, old_country, family):
    result = present(source, old_country=old_country)
    assert result.identity == "+12025550123"
    assert result.families == (family,)
    digits = re.sub(r"\D", "", result.target_value)
    expected = (
        "0012025550123"
        if family == "double_zero"
        else "2025550123"
        if family in {"national", "national_bare"}
        else "12025550123"
    )
    assert digits == expected
    assert len(result.surfaces[0].splitlines()) == len(source.splitlines())


def test_different_country_length_does_not_truncate_or_invent_subscriber_digits():
    result = present("+1 (202) 555-0123", identity="+8613123456789", country="CN", old_country="US")
    assert re.sub(r"\D", "", result.target_value) == "8613123456789"
    assert result.target_value != "+8 (613) 123-4567"


def test_dialing_prefix_and_calling_code_groups_remain_semantic():
    result = present("00.90.212.982.49.58", identity="+8613123456789", country="CN")
    assert result.target_value.startswith("00.86.")
    split_code = present("+2/034800777", identity="+8613123456789", country="CN")
    assert split_code.target_value.startswith("+86/")


def test_source_target_is_authority_for_envelope_and_occurrences_keep_line_breaks():
    result = present(
        "00.90.212.982.49.58",
        surfaces=["00.90.2\n12.982.49.58", "00 90 212 982 49 58"],
        old_country="TR",
    )
    assert "." in result.target_value
    assert "\n" not in result.target_value
    assert result.surfaces[0].count("\n") == 1
    assert "." not in result.surfaces[1]
    assert all(re.sub(r"\D", "", value) == "0012025550123" for value in result.surfaces)


def test_external_literal_prefix_is_present_once_in_target_and_physical_projection():
    result = present(
        "+20 3 481 4950 / EXT 112",
        surfaces=["20 3 481 4950 / EXT\n112"],
        literal_prefixes=["+"],
    )
    assert result.target_value.startswith("+1")
    assert not result.surfaces[0].startswith("+")
    assert ("+" + result.surfaces[0]).count("+") == 1
    assert result.surfaces[0].count("\n") == 1
    with pytest.raises(PhonePresentationError, match="literal dialing prefix"):
        present(
            "+20 3 481 4958", surfaces=["20 3 481 4958"], literal_prefixes=["+"], family="national"
        )


@pytest.mark.parametrize(
    "source",
    [
        "+2(03)4853366/77/88",
        "+20 3 4814950 / EXT 112",
        "+2 0244810131/2",
        "+20 03 4857190/1",
        "732-640-2367 x 2367",
        "(07480) 233524, 25, 26, 27",
        "(852)2763 5048 EXT:112",
        "+202-38243141 EXT:117",
        "+20 222 685 981/82/83",
        "002 03 3991050 EXT: 1060/ 1261/ 1260/ 1052",
        "(202)2180-0208 TO 9",
        "+2 0554 334 210/15",
        "02 03 4838630/1/2",
        "+82-31-634-9772,3",
        "(20)2 38290770 EXT:170",
        "+202 22596601/2",
        "+1514-223-3044 X 251",
        "4853388 EXT:404",
        "+202 35394032/4/4",
        "+202 37764200 (EXT.:2637)",
        "+202 37764200 (EXT. : 2637)",
        "002 03 3991050 EXTENSION.: 1060/1261",
        "(852)2763 5048 extn.:112",
    ],
)
def test_finite_source_compounds_are_supported_and_deterministic(source):
    first = present(source)
    assert first == present(source)
    assert first.identity == "+12025550123"
    assert first.surfaces == (first.target_value,)


def test_repeated_abbreviations_remain_equal_and_are_not_independent_phones():
    result = present("+202 35394032/4/4")
    assert len(result.surfaces) == 1
    assert result.target_value.split("/")[-1] == result.target_value.split("/")[-2]


def test_dotted_extension_caption_preserves_punctuation_and_all_subscriber_digits():
    result = present("+202 37764200 (EXT.:2637)")
    core, extension = result.target_value.split(" (EXT.:")
    assert re.sub(r"\D", "", core) == "12025550123"
    assert extension.endswith(")")
    assert len(extension[:-1]) == 4 and extension[:-1].isdigit()
    assert extension[:-1] != "2637"
    assert result.surfaces == (result.target_value,)


@pytest.mark.parametrize("caption", ["EXT...:2637", "EXT.:ABC", "EXT.:"])
def test_invalid_extension_captions_still_fail_without_fallback(caption):
    with pytest.raises(PhonePresentationError):
        present(f"+202 37764200 ({caption})")


def test_ascending_range_widens_minimally_without_changing_primary_identity():
    result = present("(202)2180-0208 TO 9", identity="+12025550129")
    assert result.identity == "+12025550129"
    assert result.target_value.endswith(" TO 30")
    with pytest.raises(PhonePresentationError, match="source suffix range must be ascending"):
        present("(202)2180-0208 TO 7")


@pytest.mark.parametrize("family", ["plus", "double_zero", "international_bare", "national"])
def test_configurable_family_keeps_the_same_identity(family):
    result = present("+20 3 481 4958", family=family)
    assert result.families == (family,)
    raw = re.sub(r"\D", "", result.target_value)
    canonical = (
        "+" + raw[2:]
        if family == "double_zero"
        else "+" + raw
        if family != "national"
        else result.target_value
    )
    parsed = phonenumbers.parse(canonical, "US" if family == "national" else None)
    assert (
        phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164) == result.identity
    )


@pytest.mark.parametrize(
    "family", ["plus", "double_zero", "international_bare", "national", "national_bare"]
)
def test_family_change_uses_compatible_prefix_envelopes_and_preserves_newlines(family):
    result = present("(+202) 33351271", surfaces=["(+202)\n33351271"], family=family)
    assert "()" not in result.target_value
    assert result.surfaces[0].count("\n") == 1
    assert re.sub(r"\D", "", result.target_value).endswith("2025550123")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"identity": "+999123456"}, "international phone"),
        ({"identity": "+8613123456789"}, "target country"),
        ({"family": "anything"}, "unrecognized phone presentation family"),
        ({"literal_prefixes": ["PHONE:"]}, "literal prefix"),
        ({"surfaces": []}, "at least one"),
    ],
)
def test_bad_inputs_surface_without_fallback(kwargs, match):
    with pytest.raises(PhonePresentationError, match=match):
        present("202-555-0123", **kwargs)


def test_captions_and_untyped_full_number_lists_are_not_guessed():
    with pytest.raises(PhonePresentationError, match="unsupported characters"):
        present("TEL 202-555-0123")
    with pytest.raises(PhonePresentationError):
        present("+20 34814958 / +20 34814959")
    with pytest.raises(PhonePresentationError, match="separate owners"):
        present("+20 34814958 / 34814959")


@pytest.mark.parametrize("source", ["+21009962173", "0021 01641877", "904-374-1250", "01004240345"])
def test_old_phone_validity_and_missing_country_do_not_block_explicit_presentation_shapes(source):
    result = present(source, old_country=None)
    assert result.identity == "+12025550123"
    assert len(re.sub(r"\D", "", result.target_value)) >= 10
