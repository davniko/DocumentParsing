"""Source-conditioned telephone presentation, separate from telephone identity.

The caller owns identity sampling and region boundaries. This module changes only
presentation: it never truncates a subscriber number to fit an old digit mask.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import phonenumbers

PhoneFamily = Literal[
    "source", "plus", "double_zero", "international_bare", "national", "national_bare"
]
_FAMILIES = frozenset(
    ("source", "plus", "double_zero", "international_bare", "national", "national_bare")
)
_DIGITS = re.compile(r"[0-9]+")
_EXTENSION = re.compile(
    r"(?i)(?:\s*[/,]?\s*)(?:EXTENSION|EXTN|EXT|EX|X)\s*(?:\.\s*:|[:.])?\s*(?=[0-9])"
)
_ALTERNATES = re.compile(r"(?P<suffix>(?:\s*[/,]\s*[0-9]{1,4})+)(?P<tail>\s*\)?)$")
_RANGE = re.compile(r"(?i)(?P<separator>\s+TO\s+)(?P<end>[0-9]{1,4})(?P<tail>\s*\)?)$")
_ANNOTATION = re.compile(r"(?i)(?P<annotation>\s*\(?\s*[0-9]+\s*LINES?\s*\)?)$")


class PhonePresentationError(ValueError):
    """An identity, source grammar, or literal boundary cannot be represented."""


@dataclass(frozen=True, slots=True)
class PhonePresentation:
    """One valid identity and its reviewed-target and occurrence presentations."""

    identity: str
    target_value: str
    surfaces: tuple[str, ...]
    families: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Source:
    core: str
    suffix: str
    suffix_kind: str
    family: str
    prefix_digits: int


def _digits(value: str) -> str:
    return "".join(_DIGITS.findall(value))


def _country_prefix(digits: str) -> str | None:
    for width in (1, 2, 3):
        if int(digits[:width] or "0") in phonenumbers.COUNTRY_CODE_TO_REGION_CODE:
            return digits[:width]
    return None


def _source(value: str, source_country_code: str | None) -> _Source:
    if not value or not _DIGITS.search(value):
        raise PhonePresentationError("source phone is empty or has no digits")
    core, suffix, suffix_kind = value, "", ""
    extension = _EXTENSION.search(value)
    annotation = _ANNOTATION.search(value)
    ranges = _RANGE.search(value)
    alternatives = _ALTERNATES.search(value)
    if extension:
        core, suffix, suffix_kind = (
            value[: extension.start()],
            value[extension.start() :],
            "extension",
        )
        if re.sub(r"[0-9\s/,:.()-]", "", value[extension.end() :]):
            raise PhonePresentationError("unsupported extension expression")
    elif annotation:
        core, suffix, suffix_kind = (
            value[: annotation.start()],
            value[annotation.start() :],
            "annotation",
        )
    elif ranges:
        core, suffix, suffix_kind = value[: ranges.start()], value[ranges.start() :], "range"
    elif alternatives and len(_digits(value[: alternatives.start()])) >= 7:
        core, suffix, suffix_kind = (
            value[: alternatives.start()],
            value[alternatives.start() :],
            "alternates",
        )
    if not suffix_kind and "/" in core:
        # A slash within a short area/prefix fragment is grouping. A slash
        # between two complete numeric expressions needs separate typed owners.
        left, right = core.split("/", 1)
        if len(_digits(left)) >= 7 and len(_digits(right)) >= 5:
            raise PhonePresentationError(
                "multiple complete phone expressions require separate owners"
            )

    # Captions belong to the region owner, not an implicit phone parser. A clean
    # source field may contain a dialing envelope, whitespace, and punctuation.
    if re.search(r"[^0-9\s+(){}./~\-\u2010-\u2015]", core):
        raise PhonePresentationError(f"unsupported characters in source phone: {core!r}")
    digits = _digits(core)
    if not digits:
        raise PhonePresentationError("source phone has no primary subscriber number")
    before = core[: _DIGITS.search(core).start()]  # type: ignore[union-attr]
    if "+" in before:
        if before.count("+") != 1 or "+" in core[len(before) :]:
            raise PhonePresentationError("source has multiple international prefixes")
        calling = _country_prefix(digits)
        # An old OCR number is a presentation example, not a new valid identity.
        # A malformed calling code still establishes an international '+' shape;
        # its complete numeric mask is replaced, never repaired by geoguessing.
        family, prefix = "plus", len(calling) if calling else 0
    elif digits.startswith("00"):
        calling = _country_prefix(digits[2:])
        family, prefix = "double_zero", 2 + (len(calling) if calling else 0)
    else:
        if (
            source_country_code is not None
            and source_country_code not in phonenumbers.SUPPORTED_REGIONS
        ):
            raise PhonePresentationError("unrecognized source country")
        calling = (
            str(phonenumbers.country_code_for_region(source_country_code))
            if source_country_code
            else None
        )
        try:
            international = phonenumbers.parse("+" + digits, None)
            is_international = (
                calling is not None
                and digits.startswith(calling)
                and phonenumbers.is_valid_number_for_region(international, source_country_code)
            )
        except phonenumbers.NumberParseException:
            is_international = False
        if is_international:
            assert calling is not None
            family, prefix = "international_bare", len(calling)
        else:
            # Without an explicit calling prefix or source-country proof, bare
            # digits mean a prefixless national presentation, not guessed old
            # geography. Leading zero is the explicit national-dialing shape.
            family, prefix = ("national" if digits.startswith("0") else "national_bare"), 0
    if len(digits) <= prefix:
        raise PhonePresentationError("source contains a calling code but no subscriber")
    return _Source(core, suffix, suffix_kind, family, prefix)


def _boundaries(total: int, widths: list[int]) -> list[int]:
    """Proportional groups with nonempty runs where the new number permits it."""
    if not widths:
        return []
    if total < len(widths):
        raise PhonePresentationError("new number is too short for the source's digit groups")
    # Reserve one digit per run; Hamilton allocation handles the remainder
    # without biasing early groups or dropping any digit.
    remaining = total - len(widths)
    weight = sum(max(width - 1, 0) for width in widths)
    weights = [max(width - 1, 0) for width in widths] if weight else [1] * len(widths)
    weight = sum(weights)
    sizes = [1 + remaining * width // weight for width in weights]
    extras = total - sum(sizes)
    order = sorted(range(len(widths)), key=lambda i: (-(remaining * weights[i] % weight), i))
    for index in order[:extras]:
        sizes[index] += 1
    boundaries, offset = [], 0
    for size in sizes:
        offset += size
        boundaries.append(offset)
    return boundaries


def _reshape_core(source: _Source, number: phonenumbers.PhoneNumber, family: str) -> str:
    matches = list(_DIGITS.finditer(source.core))
    country = str(number.country_code)
    subscriber = phonenumbers.national_significant_number(number)
    if family == "plus":
        prefix, payload = country, subscriber
    elif family == "double_zero":
        prefix, payload = "00" + country, subscriber
    elif family == "international_bare":
        prefix, payload = country, subscriber
    elif family == "national":
        prefix = ""
        payload = _digits(
            phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.NATIONAL)
        )
    else:
        prefix, payload = "", subscriber

    # Split a run at the semantic calling-code boundary before applying widths.
    # Otherwise changing a one-digit calling code to three digits steals digits
    # from an area/subscriber group.
    pieces: list[tuple[int, int, bool]] = []
    consumed = 0
    for match in matches:
        width = len(match[0])
        cuts = {0, width}
        for boundary in (source.prefix_digits, 2 if family == "double_zero" else 0):
            if 0 < boundary - consumed < width:
                cuts.add(boundary - consumed)
        ordered = sorted(cuts)
        for start, end in pairwise(ordered):
            pieces.append(
                (match.start() + start, match.start() + end, consumed + end <= source.prefix_digits)
            )
        consumed += width
    prefix_pieces = [item for item in pieces if item[2]]
    payload_pieces = [item for item in pieces if not item[2]]
    # A national source has no country-code slot. Prepend the new calling code
    # to its first digit run; all original line and grouping separators remain.
    if not prefix_pieces:
        payload = prefix + payload
        prefix = ""
    payload_widths = [end - start for start, end, _ in payload_pieces]
    replacements: dict[tuple[int, int], str] = {}
    prefix_offset = 0
    country_written = False
    for start, end, _ in prefix_pieces:
        width = end - start
        if family == "double_zero" and prefix_offset < 2:
            replacements[(start, end)] = "00"[prefix_offset : prefix_offset + width]
        else:
            replacements[(start, end)] = country if not country_written else ""
            country_written = True
        prefix_offset += width
    if family == "double_zero" and prefix_pieces and not country_written:
        # An irregular old code may establish only its literal IDD prefix.
        # Keep that 00 intact and put the new calling code before the subscriber.
        payload = country + payload
    payload_ends = _boundaries(len(payload), payload_widths)
    offset = 0
    for (start, end, _), stop in zip(payload_pieces, payload_ends, strict=True):
        replacements[(start, end)] = payload[offset:stop]
        offset = stop
    result, offset = [], 0
    for start, end, _ in pieces:
        result.extend((source.core[offset:start], replacements[(start, end)]))
        offset = end
    result.append(source.core[offset:])
    rendered = "".join(result)
    if family != "plus":
        rendered = rendered.replace("+", "")
    elif "+" not in rendered:
        first = _DIGITS.search(rendered)
        assert first is not None
        rendered = rendered[: first.start()] + "+" + rendered[first.start() :]
    expected = (
        "00" + country + subscriber
        if family == "double_zero"
        else country + subscriber
        if family in {"plus", "international_bare"}
        else payload
    )
    if _digits(rendered) != expected:
        raise PhonePresentationError("phone presentation changed the canonical digit sequence")
    return rendered


def _render_core(source: _Source, number: phonenumbers.PhoneNumber, family: str) -> str:
    if family == source.family:
        return _reshape_core(source, number, family)
    # A requested family change is not an old literal-mask substitution: simply
    # deleting a parenthesized calling-code slot would leave empty parentheses.
    # Use the metadata presentation for the new family and transfer only the
    # physical line breaks, not incompatible prefix punctuation.
    if family in {"plus", "double_zero", "international_bare"}:
        rendered = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        rendered = {"plus": "+", "double_zero": "00", "international_bare": ""}[family] + rendered[
            1:
        ]
    elif family == "national":
        rendered = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.NATIONAL)
    else:
        rendered = phonenumbers.national_significant_number(number)
    source_total = len(_digits(source.core))
    positions = [match.start() for match in re.finditer(r"[0-9]", rendered)]
    breaks: list[int] = []
    for match in re.finditer("\n", source.core):
        before = len(_digits(source.core[: match.start()]))
        index = (before * len(positions) + source_total // 2) // source_total
        breaks.append(positions[index] if index < len(positions) else len(rendered))
    for offset in reversed(breaks):
        rendered = rendered[:offset] + "\n" + rendered[offset:]
    return rendered


def _derived_digits(identity: str, token: str, width: int) -> str:
    modulus = 10**width
    value = (
        int.from_bytes(hashlib.sha256(f"{identity}\0{token}".encode()).digest(), "big") % modulus
    )
    return str(value).zfill(width)


def _suffix(source: _Source, identity: str) -> str:
    if source.suffix_kind in {"", "annotation"}:
        return source.suffix
    if source.suffix_kind == "range":
        match = _DIGITS.search(source.suffix)
        assert match is not None
        width = len(match[0])
        delta = int(match[0]) - int(_digits(source.core)[-width:])
        if delta <= 0:
            raise PhonePresentationError("source suffix range must be ascending")
        endpoint = "+" + str(int(identity[1:]) + delta)
        parsed = phonenumbers.parse(endpoint, None)
        if (
            not phonenumbers.is_valid_number(parsed)
            or parsed.country_code != phonenumbers.parse(identity, None).country_code
        ):
            raise PhonePresentationError(
                "source-shaped range endpoint is not a valid same-country number"
            )
        common = 0
        for left, right in zip(identity, endpoint, strict=False):
            if left != right:
                break
            common += 1
        # Carrying across the original abbreviation widens only its endpoint:
        # ``...39 TO 40``, never ``...39 TO 0`` or a changed primary identity.
        endpoint_width = max(width, len(endpoint) - common)
        return (
            source.suffix[: match.start()]
            + endpoint[-endpoint_width:]
            + source.suffix[match.end() :]
        )
    # Equal abbreviations/extensions remain equal. They are explicitly typed
    # auxiliary digits, never promoted to independent phone entries.
    return _DIGITS.sub(
        lambda match: _derived_digits(identity, source.suffix_kind + ":" + match[0], len(match[0])),
        source.suffix,
    )


def format_phone_presentation(
    *,
    source_value: str,
    source_surfaces: Sequence[str],
    replacement_identity: str,
    target_country_code: str,
    source_country_code: str | None = None,
    family: PhoneFamily = "source",
    literal_prefixes: Sequence[str] | None = None,
) -> PhonePresentation:
    """Render a valid identity in target and per-occurrence source styles.

    ``literal_prefixes`` contains only unchanged, adjacent dialing syntax owned
    outside each region (usually ``""``; occasionally ``"+"``). Returned
    surfaces exclude that syntax. Captions must be removed by the typed owner.
    Physical newline counts are retained exactly. Extensions and abbreviated
    alternate suffixes are deterministic auxiliaries of the primary identity.
    """
    if family not in _FAMILIES:
        raise PhonePresentationError(f"unrecognized phone presentation family: {family}")
    if target_country_code not in phonenumbers.SUPPORTED_REGIONS:
        raise PhonePresentationError("unrecognized target country")
    try:
        number = phonenumbers.parse(replacement_identity, None)
    except phonenumbers.NumberParseException as error:
        raise PhonePresentationError(
            "replacement identity must be an international phone"
        ) from error
    if number.extension or not phonenumbers.is_valid_number_for_region(number, target_country_code):
        raise PhonePresentationError("replacement identity is not valid for the target country")
    identity = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
    prefixes = (
        tuple(literal_prefixes) if literal_prefixes is not None else ("",) * len(source_surfaces)
    )
    if len(prefixes) != len(source_surfaces):
        raise PhonePresentationError("one literal prefix is required per source occurrence")
    if not source_surfaces:
        raise PhonePresentationError("at least one owned phone occurrence is required")
    target_source = _source(source_value, source_country_code)
    target_family = target_source.family if family == "source" else family
    target = _render_core(target_source, number, target_family) + _suffix(target_source, identity)
    surfaces, families = [], []
    for surface, literal in zip(source_surfaces, prefixes, strict=True):
        if literal not in {"", "+", "00"}:
            raise PhonePresentationError("literal prefix must be an explicit adjacent '+' or '00'")
        parsed = _source(literal + surface, source_country_code)
        selected = parsed.family if family == "source" else family
        rendered = _render_core(parsed, number, selected) + _suffix(parsed, identity)
        if literal:
            if not rendered.startswith(literal):
                raise PhonePresentationError(
                    "selected family conflicts with the unchanged literal dialing prefix"
                )
            rendered = rendered[len(literal) :]
        if rendered.count("\n") != surface.count("\n"):
            raise PhonePresentationError("phone presentation changed physical line count")
        surfaces.append(rendered)
        families.append(selected)
    return PhonePresentation(identity, " ".join(target.split()), tuple(surfaces), tuple(families))
