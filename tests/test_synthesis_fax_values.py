import phonenumbers
import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler.contact_values import _fax_prefix, fax


@pytest.mark.parametrize("region", ["AE", "KR", "EG", "US", "BR", "CL"])
def test_fax_is_seeded_country_valid_fixed_line_and_varied(region):
    values = set()
    for index in range(100):
        stream = DeterministicStream(42, "fax-test", str(index))
        value = fax(stream, country_code=region)
        number = phonenumbers.parse(value, None)
        assert value == fax(stream, country_code=region)
        assert phonenumbers.is_valid_number_for_region(number, region)
        assert phonenumbers.number_type(number) in (
            phonenumbers.PhoneNumberType.FIXED_LINE,
            phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE,
        )
        assert value == phonenumbers.format_number(
            number, phonenumbers.PhoneNumberFormat.INTERNATIONAL
        )
        values.add(value)
    assert len(values) > 90


@pytest.mark.parametrize("region", ["AE", "KR", "EG", "US", "BR", "CL"])
def test_fax_exclusion_uses_number_identity_not_formatting(region):
    stream = DeterministicStream(42, "fax-test", "excluded")
    original = fax(stream, country_code=region)
    number = phonenumbers.parse(original, None)
    canonical = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
    national = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.NATIONAL)
    international_exclusion = fax(stream, country_code=region, excluded=(canonical,))
    national_exclusion = fax(stream, country_code=region, excluded=(national,))
    assert international_exclusion != original
    assert international_exclusion == national_exclusion
    assert international_exclusion == fax(stream, country_code=region, excluded=(original,))


def test_fax_rejects_missing_metadata_and_malformed_exclusion():
    stream = DeterministicStream(42, "fax-test", "rejected")
    with pytest.raises(ValueError, match="no fixed-line numbering metadata"):
        fax(stream, country_code="ZZ")
    with pytest.raises(ValueError, match="excluded fax is not a parseable"):
        fax(stream, country_code="AE", excluded=("not a number",))


def test_fax_never_returns_a_mobile_metadata_example(monkeypatch):
    mobile = phonenumbers.example_number_for_type("EG", phonenumbers.PhoneNumberType.MOBILE)
    monkeypatch.setattr(phonenumbers, "example_number_for_type", lambda *_: mobile)
    _fax_prefix.cache_clear()
    try:
        with pytest.raises(ValueError, match="fixed-line numbering metadata is invalid"):
            fax(DeterministicStream(42, "fax-test", "mobile"), country_code="EG")
    finally:
        _fax_prefix.cache_clear()


def test_exhausted_suffix_space_raises_instead_of_reusing_excluded_fax():
    base = _fax_prefix("AE")
    blocked = tuple(base[:-4] + f"{suffix:04d}" for suffix in range(10000))
    with pytest.raises(ValueError, match="no valid synthetic fax subscriber"):
        fax(DeterministicStream(42, "fax-test", "exhausted"), country_code="AE", excluded=blocked)
