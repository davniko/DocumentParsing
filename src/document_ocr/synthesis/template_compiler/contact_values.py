"""Typed fictional mailboxes, independent of malformed OCR punctuation."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import phonenumbers

from document_ocr.synthesis.generators import DeterministicStream

_LOCAL = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+")
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


def validate_phone(value: str, *, country_code: str | None = None) -> None:
    """Validate dialing syntax and country assignment, not subscriber ownership."""
    try:
        number = phonenumbers.parse(" ".join(value.split()), None)
    except phonenumbers.NumberParseException as error:
        raise ValueError("generated phone is not an international number") from error
    if not (
        phonenumbers.is_valid_number_for_region(number, country_code)
        if country_code is not None
        else phonenumbers.is_valid_number(number)
    ):
        raise ValueError("generated phone is invalid for its entity country")


def source_phone_country(value: str) -> str:
    """Resolve only explicit international dialing, never guess a local region."""
    compact = re.sub(r"[\s()./-]", "", value)
    if compact.startswith("00"):
        compact = "+" + compact[2:]
    if not re.fullmatch(r"\+[1-9][0-9]+", compact):
        raise ValueError("source phone lacks an explicit international country code")
    try:
        number = phonenumbers.parse(compact, None)
    except phonenumbers.NumberParseException as error:
        raise ValueError("source phone has an unrecognized country code") from error
    region = phonenumbers.region_code_for_number(number)
    if region is None or region == "001":
        raise ValueError("source phone country is unresolved")
    return region


@lru_cache(maxsize=256)
def _phone_prefix(country_code: str) -> str:
    # Mobile prefixes avoid attaching a different city's fixed-line area code to
    # the generated address. The pinned libphonenumber metadata owns the grammar.
    example = phonenumbers.example_number_for_type(
        country_code, phonenumbers.PhoneNumberType.MOBILE
    )
    if example is None:
        raise ValueError(f"no mobile numbering metadata for {country_code}")
    return phonenumbers.format_number(example, phonenumbers.PhoneNumberFormat.E164)


def phone(stream: DeterministicStream, *, country_code: str) -> str:
    """Generate a country-valid synthetic subscriber, with no claim it is unassigned."""
    base = _phone_prefix(country_code)
    # Retain the assigned country/network prefix and search its 10,000-subscriber
    # suffix space deterministically; never return an unchecked/default number.
    start = stream.randbelow(10000)
    for offset in range(10000):
        candidate = base[:-4] + f"{(start + offset) % 10000:04d}"
        number = phonenumbers.parse(candidate, None)
        if phonenumbers.is_valid_number_for_region(number, country_code):
            return phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    raise ValueError(f"no valid synthetic subscriber in the certified prefix for {country_code}")


def validate_mailbox(value: str) -> None:
    """Accept a single ASCII dot-atom mailbox, not display names or OCR noise."""
    if len(value) > 254 or value.count("@") != 1:
        raise ValueError("email must contain one complete mailbox")
    local, domain = value.split("@")
    if (
        not _LOCAL.fullmatch(local)
        or len(local) > 64
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or len(domain.split(".")) < 2
        or any(not _LABEL.fullmatch(label) for label in domain.split("."))
    ):
        raise ValueError("email has invalid mailbox syntax")


def mailbox(stream: DeterministicStream, *, organization: str | None = None) -> str:
    identity = stream.bytes(counter=0, length=6).hex()
    if organization is None:
        domain = "company-" + identity
    else:
        ascii_name = unicodedata.normalize("NFKD", organization).encode("ascii", "ignore").decode()
        stem = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
        # Identity remains unique even when transliteration has no Latin letters.
        domain = ((stem[:49].rstrip("-") + "-") if stem else "company-") + identity
    value = f"contact@{domain}.example"
    validate_mailbox(value)
    return value


def render_mailbox(source: str, value: str) -> str:
    """Keep case and line envelopes; wrapping never discards mailbox characters."""
    from .descendant import _allocate_words, _case_like, _line_edges

    validate_mailbox(value)
    lines = re.split(r"\r\n|\r|\n", source)
    endings = re.findall(r"\r\n|\r|\n", source)
    weights = tuple(len(_line_edges(line)[1]) for line in lines)
    if len(value) < sum(w > 0 for w in weights):
        raise ValueError("email cannot populate the source's occupied lines")
    parts = _allocate_words(tuple(_case_like(source, value)), weights)
    result = []
    for line, chars in zip(lines, parts, strict=True):
        leading, _, trailing = _line_edges(line)
        result.append(leading + _case_like(line, "".join(chars)) + trailing)
    rendered = result[0] + "".join(e + s for e, s in zip(endings, result[1:], strict=True))
    if "".join(rendered.split()).casefold() != value.casefold():
        raise ValueError("email wrapping changed its semantic value")
    return rendered
