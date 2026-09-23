"""Typed fictional mailboxes, independent of malformed OCR punctuation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import phonenumbers

from document_ocr.synthesis.generators import DeterministicStream

if TYPE_CHECKING:
    from .host import SpanDraft

_LOCAL = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+")
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


def phone_inside_literal_prefix(*, source: bytes, byte_start: int, rendered: str) -> str:
    """Do not duplicate an international '+' owned by the literal source frame.

    Only the adjacent source byte proves this boundary. A phone whose slot owns
    its plus, or whose original number was local, still renders its full value.
    The canonical target is never changed.
    """
    if byte_start > 0 and source[byte_start - 1 : byte_start] == b"+" and rendered.startswith("+"):
        return rendered[1:]
    return rendered


def normalize_attention_repeats(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own a repeated name in one explicit ``Name <email> ATTN: Name`` record.

    Matching names in different parties are not a relationship. The intervening
    mailbox must already belong to the same contact role, and only unowned or
    same-role contact-name fragments may be joined. No label is added or changed.
    """
    from .host import _resolve_target_path, _surface_match, validate_draft_source_alignment

    validate_draft_source_alignment(raw=raw, drafts=drafts)
    references = {key for d in drafts for key in d.dependency_bindings}
    attention = re.compile(r"\s*<(?P<email>[^<>\s]+)>\s*ATT(?:N|ENTION)\s*:\s*", re.I)
    removed: set[str] = set()
    additions: list[SpanDraft] = []
    for draft in drafts:
        if not (
            draft.render_mode == "target_binding"
            and draft.value_kind == "contact_name"
            and len(draft.target_paths) == 1
            and draft.target_paths[0].endswith(".contactDetails.contactName")
            and not draft.derivation
            and not draft.dependency_paths
            and not draft.dependency_bindings
        ):
            continue
        value = _resolve_target_path(source_target, draft.target_paths[0])
        if not isinstance(value, str) or _surface_match(
            source=draft.source_text,
            target=value,
            adapter="natural_text",
            value_kind="contact_name",
        ) != ("", ""):
            continue
        gap = attention.match(raw, draft.char_end)
        if gap is None:
            continue
        email_path = draft.target_paths[0].removesuffix("contactName") + "emailAddresses["
        if not any(
            d.char_start == gap.start("email")
            and d.char_end == gap.end("email")
            and d.group_key == draft.group_key
            and any(p.startswith(email_path) for p in d.target_paths)
            for d in drafts
        ):
            continue
        name = re.compile(r"\s+".join(map(re.escape, value.split())) + r"(?!\w)", re.I).match(
            raw, gap.end()
        )
        if name is None:
            continue
        overlaps = [d for d in drafts if d.char_start < name.end() and d.char_end > name.start()]
        if (
            len(overlaps) == 1
            and overlaps[0].char_start == name.start()
            and overlaps[0].char_end == name.end()
            and overlaps[0].logical_key == draft.logical_key
        ):
            continue
        if any(
            d.char_start < name.start()
            or d.char_end > name.end()
            or d.group_key != draft.group_key
            or d.group_kind != draft.group_kind
            or d.value_kind != "contact_name"
            or d.render_mode not in {"target_binding", "deterministic_auxiliary"}
            or (d.target_paths and d.target_paths != draft.target_paths)
            or d.dependency_paths
            or d.dependency_bindings
            or d.derivation
            or (d.logical_key != draft.logical_key and d.logical_key in references)
            for d in overlaps
        ):
            continue
        # Do not partly absorb a repeated auxiliary with other occurrences.
        absorbed_keys = {d.logical_key for d in overlaps if d.logical_key != draft.logical_key}
        if any(d.logical_key in absorbed_keys and d not in overlaps for d in drafts):
            continue
        if any(d.char_start < name.end() and d.char_end > name.start() for d in additions):
            continue
        removed.update(d.draft_id for d in overlaps)
        additions.append(
            replace(
                draft,
                draft_id=draft.draft_id + ":attention-repeat",
                char_start=name.start(),
                char_end=name.end(),
                source_text=name[0],
                rationale=draft.rationale
                + " Host proved a complete repeated attention name in the same "
                "role-owned mailbox record.",
            )
        )
    return (
        tuple(
            sorted(
                (*(d for d in drafts if d.draft_id not in removed), *additions),
                key=lambda d: (d.char_start, d.char_end, d.draft_id),
            )
        )
        if additions
        else tuple(drafts)
    )


def require_contact_name_coverage(source: bytes, template: Any, target: Mapping[str, Any]) -> None:
    """Require every repeated full contact name to have a semantic owner.

    OCR line breaks may split one name among multiple slots. Whitespace is not
    part of ownership, but every printed name character must be covered by a
    contact target or its explicitly declared dependency. An independently
    generated attention name does not establish that dependency.
    """
    from .descendant import _flatten_leaves

    names = {
        value
        for path, value in _flatten_leaves(target).items()
        if path.endswith(".contactDetails.contactName") and isinstance(value, str)
    }
    if not names:
        return
    covered = bytearray(len(source))
    for binding in template.bindings:
        if any(
            path.endswith(".contactDetails.contactName")
            for path in (*binding.target_paths, *binding.dependency_paths)
        ):
            for slot in binding.occurrences:
                covered[slot.byte_start : slot.byte_end] = b"\1" * (slot.byte_end - slot.byte_start)
    # A name inside a company/mailbox, or a complete independently typed
    # contact, has its own owner. Matching source strings alone do not prove
    # those separate fields must remain equal after synthesis.
    independent = [
        (slot.byte_start, slot.byte_end)
        for binding in template.bindings
        for slot in binding.occurrences
    ]
    text = source.decode("utf-8")
    for name in sorted(names):
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, name.split())) + r"(?!\w)"
        matches = list(re.finditer(pattern, text, re.I))
        if len(matches) < 2:
            continue
        for match in matches:
            offset = len(text[: match.start()].encode("utf-8"))
            end = offset + len(match.group().encode("utf-8"))
            if any(start <= offset and end <= stop for start, stop in independent):
                continue
            for char in match.group():
                width = len(char.encode("utf-8"))
                if not char.isspace() and not all(covered[offset : offset + width]):
                    raise ValueError(
                        "repeated contact name lacks complete target/dependency ownership "
                        f"at source byte {offset}: {name}"
                    )
                offset += width


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


def validate_party_phones(
    source: Sequence[str] | None, target: Sequence[str], *, country_code: str
) -> None:
    """Validate full numbers and source-proven shared-prefix local continuations.

    A printed list can contain one international number followed by local
    subscriber numbers. The source list must prove that representation; an
    invalid new full number must never be reinterpreted as a continuation.
    """
    if source is not None and len(source) != len(target):
        raise ValueError("party phone topology changed")
    for index, value in enumerate(target):
        compact = re.sub(r"[\s()./-]", "", value)
        explicit = compact.startswith(("+", "00"))
        try:
            number = phonenumbers.parse(value, country_code)
        except phonenumbers.NumberParseException:
            number = None
        if number is not None and phonenumbers.is_valid_number_for_region(number, country_code):
            continue
        if explicit or not compact.isdigit():
            raise ValueError("generated party phone contradicts sampled country")
        if source is None:
            raise ValueError("local phone continuation requires its source evidence")
        source_compact = re.sub(r"[\s()./-]", "", source[index])
        if not source_compact.isdigit() or source_compact.startswith("00"):
            raise ValueError("invalid full phone cannot become a local continuation")
        completions = set()
        for anchor_index, source_anchor in enumerate(source):
            try:
                anchor_region = source_phone_country(source_anchor)
                old_anchor = phonenumbers.parse(source_anchor, anchor_region)
                new_anchor = phonenumbers.parse(target[anchor_index], country_code)
            except (ValueError, phonenumbers.NumberParseException):
                continue
            if not phonenumbers.is_valid_number_for_region(new_anchor, country_code):
                continue
            old_national, new_national = (
                str(old_anchor.national_number),
                str(new_anchor.national_number),
            )
            if len(source_compact) >= len(old_national) or len(compact) >= len(new_national):
                continue
            old_completion = phonenumbers.parse(
                f"+{old_anchor.country_code}"
                + old_national[: -len(source_compact)]
                + source_compact,
                None,
            )
            if not phonenumbers.is_valid_number_for_region(old_completion, anchor_region):
                continue
            candidate = f"+{new_anchor.country_code}" + new_national[: -len(compact)] + compact
            if phonenumbers.is_valid_number_for_region(
                phonenumbers.parse(candidate, None), country_code
            ):
                completions.add(candidate)
        if len(completions) != 1:
            raise ValueError("local party phone lacks one source-proven country-valid completion")


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
