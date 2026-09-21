"""Pin a country when part of its registered name is immutable source context."""

import re
from collections.abc import Mapping
from functools import lru_cache

from .models import AuxiliaryEntity, CertifiedSemanticTemplate, SemanticBinding


def _compact(value: str) -> str:
    return "".join(c.casefold() for c in value if c.isalnum())


def _terminal_country(value: str, countries: Mapping[str, str]) -> str | None:
    # Resolve a complete address component, not a suffix of a locality such as
    # NEW JERSEY. A numeric postal/address prefix may share the country component.
    component = re.split(r"[,;\n]", value.rstrip(" .;,\t\r\n"))[-1].strip()
    component = re.sub(r"^\d[\d -]*\s+", "", component)
    normalized = _compact(component)
    return countries.get(normalized)


def address_country_context(
    entity: AuxiliaryEntity,
    template: CertifiedSemanticTemplate,
    source: bytes,
    countries: Mapping[str, str],
) -> frozenset[str]:
    """Find complete, unowned country components immediately after address facets.

    A country printed outside all slots is a constraint, not a generated facet.
    Never cross another owned slot or a line boundary to infer that constraint.
    Requiring a comma/semicolon-separated component avoids interpreting words
    inside an address or a neighbouring column as a country suffix.
    """
    keys = {
        member.logical_key
        for member in entity.members
        if member.field in {"address", "city", "region", "postal_code"}
    }
    slots = [slot for binding in template.bindings for slot in binding.occurrences]
    codes = set()
    for binding in template.bindings:
        if binding.logical_key not in keys:
            continue
        for slot in binding.occurrences:
            newline = source.find(b"\n", slot.byte_end)
            right = min(
                len(source) if newline < 0 else newline,
                min(
                    (s.byte_start for s in slots if s.byte_start >= slot.byte_end),
                    default=len(source),
                ),
            )
            tail = source[slot.byte_end : right].decode().strip()
            if tail.startswith((",", ";")):
                # The whole remaining component must be a registered alias.
                code = countries.get(_compact(tail.lstrip(",; ").rstrip(" .;,")))
                if code:
                    codes.add(code)
    return frozenset(codes)


def source_entity_conflicts(
    template: CertifiedSemanticTemplate, countries: Mapping[str, str]
) -> list[str]:
    """Detect source facets that the single-geography entity contract cannot express.

    Different registered/physical countries can be legitimate. They still require
    explicit separate ownership instead of one randomly generated geography.
    """
    by_key = {b.logical_key: b for b in template.bindings}
    issues = []
    for entity in template.auxiliary_semantic_plan.entities:
        declared = {
            code
            for m in entity.members
            if m.field == "country"
            for s in by_key[m.logical_key].occurrences
            if (code := countries.get(_compact(s.source_text))) is not None
        }
        addresses = {
            code
            for m in entity.members
            if m.field == "address"
            for s in by_key[m.logical_key].occurrences
            if (code := _terminal_country(s.source_text, countries)) is not None
        }
        if declared and addresses and declared != addresses:
            issues.append(
                f"{entity.entity_id}: country facets {sorted(declared)} and "
                f"address countries {sorted(addresses)} require reviewed ownership"
            )
    return issues


@lru_cache(maxsize=4096)
def _country_frame(
    before: str, value: str, after: str, countries: tuple[tuple[str, str], ...]
) -> str | None:
    text = _compact(before + value + after)
    left, right = len(_compact(before)), len(_compact(before + value))
    codes = set()
    for name, code in countries:
        if len(name) <= right - left:
            continue
        offset = text.find(name)
        while offset >= 0:
            if offset <= left and offset + len(name) >= right:
                codes.add(code)
            offset = text.find(name, offset + 1)
    if len(codes) > 1:
        raise ValueError("immutable country-name frame has conflicting registered meanings")
    return next(iter(codes), None)


def pinned_country(
    binding: SemanticBinding,
    template: CertifiedSemanticTemplate,
    source: bytes,
    countries: Mapping[str, str],
) -> str | None:
    codes = set()
    slots = [slot for b in template.bindings for slot in b.occurrences]
    for slot in binding.occurrences:
        left = max(
            source.rfind(b"\n", 0, slot.byte_start) + 1,
            max((s.byte_end for s in slots if s.byte_end <= slot.byte_start), default=0),
        )
        newline = source.find(b"\n", slot.byte_end)
        right = min(
            len(source) if newline < 0 else newline,
            min(
                (s.byte_start for s in slots if s.byte_start >= slot.byte_end), default=len(source)
            ),
        )
        code = _country_frame(
            source[left : slot.byte_start].decode(),
            slot.source_text,
            source[slot.byte_end : right].decode(),
            tuple(countries.items()),
        )
        if code:
            codes.add(code)
    if len(codes) > 1:
        raise ValueError("repeated country slots have conflicting immutable frames")
    return next(iter(codes), None)
