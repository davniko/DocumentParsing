"""Fail-closed preflight for source-proved party facts printed as cargo marks.

The source mark's role is established by its own compiled cargo-mark binding
and a source party fact, not by country equality or a whole-document search.
Bindings declare reviewed ownership with ``dependency_paths`` (party facts)
or ``dependency_bindings`` (same-party auxiliary identifiers).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import CertifiedSemanticTemplate, SemanticBinding

_MARK_PATH = re.compile(r"^documentPatch\.cargoGroups\[(\d+)\]\.marksAndNumbers\[(\d+)\]$")
_IDENTIFIER_FIELDS = frozenset({"registration_identifier", "tax_identifier"})


@dataclass(frozen=True, slots=True)
class _Party:
    path: str
    source: Mapping[str, Any]
    target: Mapping[str, Any]


def _words(value: str) -> tuple[str, ...]:
    folded = unicodedata.normalize("NFKD", value.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return tuple(re.findall(r"\w+", folded))


def _has_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    return bool(needle) and any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _has_identifier(words: tuple[str, ...], identifier: str) -> bool:
    return any(
        "".join(words[start:end]) == identifier
        for start in range(len(words))
        for end in range(start + 1, len(words) + 1)
    )


def _party_rows(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    patch = document.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    if not isinstance(parties, Mapping):
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for role, party in parties.items():
        if role == "carrier":
            continue
        if role == "notifyParties":
            if isinstance(party, Sequence) and not isinstance(party, (str, bytes)):
                for index, item in enumerate(party):
                    if isinstance(item, Mapping):
                        rows[f"documentPatch.parties.notifyParties[{index}]"] = item
        elif isinstance(party, Mapping):
            rows[f"documentPatch.parties.{role}"] = party
    return rows


def _resolved_party(party: Mapping[str, Any], document: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = party.get("sameAs")
    if reference is None:
        return party
    if reference not in {"shipper", "consignee"}:
        raise ValueError("cargo-mark party has an unsupported sameAs reference")
    patch = document.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    resolved = parties.get(reference) if isinstance(parties, Mapping) else None
    if not isinstance(resolved, Mapping) or resolved.get("sameAs") is not None:
        raise ValueError("cargo-mark party sameAs cannot be resolved")
    return resolved


def _parties(source_target: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[_Party, ...]:
    source = _party_rows(source_target)
    generated = _party_rows(target)
    return tuple(
        _Party(
            path,
            _resolved_party(source_party, source_target),
            _resolved_party(generated[path], target) if path in generated else {},
        )
        for path, source_party in source.items()
    )


def _source_description_words(
    source_target: Mapping[str, Any], group_index: int
) -> tuple[str, ...]:
    groups = source_target.get("documentPatch", {}).get("cargoGroups", [])
    if not isinstance(groups, Sequence) or group_index >= len(groups):
        return ()
    group = groups[group_index]
    if not isinstance(group, Mapping):
        return ()
    parts = [group.get("description")]
    additional = group.get("additionalInformation")
    if isinstance(additional, str):
        parts.append(additional)
    elif isinstance(additional, Sequence):
        parts.extend(additional)
    return _words(" ".join(part for part in parts if isinstance(part, str)))


def _mark_value(document: Mapping[str, Any], group_index: int, mark_index: int) -> str:
    try:
        value = document["documentPatch"]["cargoGroups"][group_index]["marksAndNumbers"][mark_index]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("cargo mark path is absent from source or synthetic target") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cargo mark path is not a nonempty string")
    return value


def _mark_bindings(template: CertifiedSemanticTemplate) -> dict[tuple[int, int], SemanticBinding]:
    result: dict[tuple[int, int], SemanticBinding] = {}
    for binding in template.bindings:
        for path in binding.target_paths:
            match = _MARK_PATH.fullmatch(path)
            if match is None:
                continue
            key = (int(match.group(1)), int(match.group(2)))
            if key in result:
                raise ValueError(f"cargo mark has multiple compiled owners: {path}")
            result[key] = binding
    return result


def _printed_mark(binding: SemanticBinding, value: str) -> bool:
    expected = _words(value)
    return any(_words(slot.source_text) == expected for slot in binding.occurrences)


def _owns_party_fact(
    binding: SemanticBinding,
    path: str,
    bindings_by_key: Mapping[str, SemanticBinding],
    visited: frozenset[str] = frozenset(),
) -> bool:
    if path in binding.target_paths or path in binding.dependency_paths:
        return True
    if binding.logical_key in visited:
        raise ValueError(f"cyclic cargo-mark dependency: {binding.logical_key}")
    ancestors = visited | {binding.logical_key}
    return any(
        _owns_party_fact(bindings_by_key[key], path, bindings_by_key, ancestors)
        for key in binding.dependency_bindings
        if key in bindings_by_key
    )


def _name_matches(
    mark: tuple[str, ...], party: _Party, description: tuple[str, ...], binding: SemanticBinding
) -> bool:
    source_name = party.source.get("name")
    if not isinstance(source_name, str):
        return False
    name = _words(source_name)
    if sum(map(len, mark)) < 6:
        return False
    unambiguous_prefix = name[: len(mark)] == mark and (len(mark) >= 2 or mark == name)
    full_name_inside_mark = len(name) >= 2 and _has_phrase(mark, name)
    if not unambiguous_prefix and not full_name_inside_mark:
        return False
    # A duplicated product brand is not source proof of party ownership.
    # An explicit reviewed dependency can still establish that relationship.
    return not _has_phrase(description, mark) or party.path + ".name" in (
        *binding.target_paths,
        *binding.dependency_paths,
    )


def _require_party_mark(
    *,
    binding: SemanticBinding,
    source_mark: str,
    target_mark: str,
    party: _Party,
    fields: tuple[str, ...],
    expected: str,
    bindings_by_key: Mapping[str, SemanticBinding],
) -> None:
    if all(party.source.get(field) == party.target.get(field) for field in fields):
        return
    required = {party.path + "." + field for field in fields}
    missing = sorted(
        path for path in required if not _owns_party_fact(binding, path, bindings_by_key)
    )
    if missing:
        raise ValueError(
            "source-proved party mark lacks explicit segment ownership: "
            f"{binding.logical_key}; source={source_mark!r}; missing={missing}"
        )
    target_words = _words(target_mark)
    source_words = _words(source_mark)
    source_name = party.source.get("name")
    source_name_words = _words(source_name) if isinstance(source_name, str) else ()
    composite_name_mark = len(source_words) > len(source_name_words) and _has_phrase(
        source_words, source_name_words
    )
    target_matches = (
        _has_phrase(target_words, _words(expected))
        and not _has_phrase(target_words, source_name_words)
        if composite_name_mark and fields == ("name",)
        else target_words == _words(expected)
    )
    if not target_matches:
        raise ValueError(
            "party-mark target differs from its declared generated party fact: "
            f"{binding.logical_key}; expected={expected!r}; got={target_mark!r}"
        )


def _selected_owners(
    *,
    binding: SemanticBinding,
    candidates: tuple[_Party, ...],
    fields: tuple[str, ...],
    bindings_by_key: Mapping[str, SemanticBinding],
) -> tuple[_Party, ...]:
    declared = tuple(
        party
        for party in candidates
        if all(
            _owns_party_fact(binding, party.path + "." + field, bindings_by_key) for field in fields
        )
    )
    if declared:
        return declared
    changed = next(
        (
            party
            for party in candidates
            if any(party.source.get(field) != party.target.get(field) for field in fields)
        ),
        None,
    )
    if changed is not None:
        required = sorted(changed.path + "." + field for field in fields)
        raise ValueError(
            "source-proved party mark lacks explicit segment ownership: "
            f"{binding.logical_key}; required ownership of {required}"
        )
    return ()


def _same_party_identifiers(
    template: CertifiedSemanticTemplate,
    bindings_by_key: Mapping[str, SemanticBinding],
) -> tuple[tuple[str, str, str], ...]:
    identifiers: list[tuple[str, str, str]] = []
    for entity in template.auxiliary_semantic_plan.entities:
        if entity.relationship != "same_as_target_party" or entity.target_party_path is None:
            continue
        for member in entity.members:
            if member.field not in _IDENTIFIER_FIELDS:
                continue
            identifier_binding = bindings_by_key.get(member.logical_key)
            if identifier_binding is None:
                raise ValueError("same-party registration member lacks a source binding")
            source_values = {
                "".join(_words(slot.source_text)) for slot in identifier_binding.occurrences
            }
            if len(source_values) != 1:
                raise ValueError("same-party registration source surfaces disagree")
            identifier = next(iter(source_values))
            if len(identifier) >= 6:
                identifiers.append((entity.target_party_path, member.logical_key, identifier))
    return tuple(identifiers)


def validate_party_mark_preflight(
    *,
    template: CertifiedSemanticTemplate,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    auxiliary_values: Mapping[str, str],
) -> None:
    """Reject changed parties when source-proved cargo marks lack coherent ownership.

    A one-word brand, a country match by itself, and an independent exporter
    entity never establish a party-mark dependency.
    """

    marks = _mark_bindings(template)
    source_groups = source_target.get("documentPatch", {}).get("cargoGroups", [])
    source_mark_words = {
        _words(mark)
        for group in source_groups
        if isinstance(group, Mapping)
        for mark in group.get("marksAndNumbers") or ()
        if isinstance(mark, str)
    }
    source_only = tuple(
        binding
        for binding in template.bindings
        if binding.group_kind == "cargo"
        and not binding.target_paths
        and (
            re.search(r"(?i)(?:^|[_:])marks?(?:\d|[_:]|$)", binding.logical_key)
            or any(_words(slot.source_text) in source_mark_words for slot in binding.occurrences)
        )
    )
    if not marks and not source_only:
        return
    parties = _parties(source_target, target)
    auxiliary_by_key = {binding.logical_key: binding for binding in template.bindings}
    entity_identifiers = _same_party_identifiers(template, auxiliary_by_key)
    # Some printed cargo marks are repeated through a source-only binding. Its
    # derived owner must ultimately reach the same party fact as the target
    # mark; a literal copy is unsafe when that party changes.
    descriptions = (
        tuple(
            word
            for index in range(len(source_groups))
            for word in _source_description_words(source_target, index)
        )
        if source_only
        else ()
    )
    for binding in source_only:
        for slot in binding.occurrences:
            mark = _words(slot.source_text)
            candidates = tuple(
                party for party in parties if _name_matches(mark, party, descriptions, binding)
            )
            _selected_owners(
                binding=binding,
                candidates=candidates,
                fields=("name",),
                bindings_by_key=auxiliary_by_key,
            )
            for party_path, member_key, source_identifier in entity_identifiers:
                registration_party = next(
                    (party for party in parties if party.path == party_path), None
                )
                if registration_party is None or not _has_identifier(mark, source_identifier):
                    continue
                new_identifier = auxiliary_values.get(member_key)
                if registration_party.source.get("name") == registration_party.target.get(
                    "name"
                ) and (
                    new_identifier is None or "".join(_words(new_identifier)) == source_identifier
                ):
                    continue
                if (
                    binding.logical_key != member_key
                    and member_key not in binding.dependency_bindings
                ) or not new_identifier:
                    raise ValueError(
                        "source-only party registration mark lacks explicit binding ownership: "
                        f"{binding.logical_key}; required={member_key}"
                    )
    if not marks:
        return

    for (group_index, mark_index), binding in marks.items():
        source_mark = _mark_value(source_target, group_index, mark_index)
        mark = _words(source_mark)
        description = _source_description_words(source_target, group_index)
        name_candidates = tuple(
            party for party in parties if _name_matches(mark, party, description, binding)
        )
        # A full party name or multiword prefix supplies source identity proof
        # unless it also occurs as product prose. One-word prefixes remain
        # ambiguous brand marks without an explicit reviewed owner.
        if any(
            party.source.get("name") != party.target.get("name") for party in name_candidates
        ) and not _printed_mark(binding, source_mark):
            raise ValueError(
                f"party-shaped cargo mark lacks physical source proof: {binding.logical_key}"
            )
        for party in _selected_owners(
            binding=binding,
            candidates=name_candidates,
            fields=("name",),
            bindings_by_key=auxiliary_by_key,
        ):
            target_mark = _mark_value(target, group_index, mark_index)
            expected = party.target.get("name")
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(f"generated party name is missing for {party.path}")
            _require_party_mark(
                binding=binding,
                source_mark=source_mark,
                target_mark=target_mark,
                party=party,
                fields=("name",),
                expected=expected,
                bindings_by_key=auxiliary_by_key,
            )

        # Adjacent city/country or country-only marks inherit the proven party
        # owner of the preceding name mark; country coincidence alone never does.
        if mark_index > 0:
            preceding = marks.get((group_index, mark_index - 1))
            if preceding is not None:
                preceding_mark = _words(_mark_value(source_target, group_index, mark_index - 1))
                preceding_owners = tuple(
                    party
                    for party in parties
                    if _name_matches(preceding_mark, party, description, preceding)
                )
                declared_name_owners = {
                    party.path
                    for party in preceding_owners
                    if _owns_party_fact(preceding, party.path + ".name", auxiliary_by_key)
                }
                candidates = tuple(
                    party
                    for party in preceding_owners
                    if not declared_name_owners or party.path in declared_name_owners
                )
                locality = tuple(
                    party
                    for party in candidates
                    if isinstance(party.source.get("city"), str)
                    and isinstance(party.source.get("country"), str)
                    and mark == _words(party.source["city"] + " " + party.source["country"])
                )
                country = tuple(
                    party
                    for party in candidates
                    if isinstance(party.source.get("country"), str)
                    and mark == _words(party.source["country"])
                )
                address = tuple(
                    party
                    for party in candidates
                    if isinstance(party.source.get("address"), str)
                    and mark == _words(party.source["address"])
                )
                if locality or country or address:
                    source_name_mark = _mark_value(source_target, group_index, mark_index - 1)
                    changed_adjacent_fact = (
                        any(
                            party.source.get(field) != party.target.get(field)
                            for party in locality
                            for field in ("city", "country")
                        )
                        or any(
                            party.source.get("country") != party.target.get("country")
                            for party in country
                        )
                        or any(
                            party.source.get("address") != party.target.get("address")
                            for party in address
                        )
                    )
                    if changed_adjacent_fact and not _printed_mark(preceding, source_name_mark):
                        raise ValueError(
                            "party-name mark lacks physical source proof for locality: "
                            + preceding.logical_key
                        )
                    if not _printed_mark(binding, source_mark):
                        raise ValueError(
                            "party locality mark lacks physical source proof: "
                            + binding.logical_key
                        )
                    target_mark = _mark_value(target, group_index, mark_index)
                    fields: tuple[str, ...]
                    if locality:
                        fields = ("city", "country")
                        owners = _selected_owners(
                            binding=binding,
                            candidates=locality,
                            fields=fields,
                            bindings_by_key=auxiliary_by_key,
                        )
                    elif country:
                        fields = ("country",)
                        owners = _selected_owners(
                            binding=binding,
                            candidates=country,
                            fields=fields,
                            bindings_by_key=auxiliary_by_key,
                        )
                    else:
                        fields = ("address",)
                        owners = _selected_owners(
                            binding=binding,
                            candidates=address,
                            fields=fields,
                            bindings_by_key=auxiliary_by_key,
                        )
                    for party in owners:
                        expected_parts: list[str] = []
                        for field in fields:
                            part = party.target.get(field)
                            if not isinstance(part, str) or not part:
                                raise ValueError(
                                    f"generated party mark component is missing for {party.path}"
                                )
                            expected_parts.append(part)
                        expected = " ".join(expected_parts)
                        _require_party_mark(
                            binding=binding,
                            source_mark=source_mark,
                            target_mark=target_mark,
                            party=party,
                            fields=fields,
                            expected=expected,
                            bindings_by_key=auxiliary_by_key,
                        )

        for party_path, member_key, source_identifier in entity_identifiers:
            registration_party = next(
                (candidate for candidate in parties if candidate.path == party_path), None
            )
            if registration_party is None or not _has_identifier(mark, source_identifier):
                continue
            new_identifier = auxiliary_values.get(member_key)
            if registration_party.source.get("name") == registration_party.target.get("name") and (
                new_identifier is None or "".join(_words(new_identifier)) == source_identifier
            ):
                continue
            if not _printed_mark(binding, source_mark):
                raise ValueError(
                    "party registration mark lacks physical source proof: " + binding.logical_key
                )
            if member_key not in binding.dependency_bindings or not new_identifier:
                raise ValueError(
                    "source-proved party registration mark lacks explicit binding ownership: "
                    f"{binding.logical_key}; required={member_key}"
                )
            new_mark = _mark_value(target, group_index, mark_index)
            if not _has_identifier(_words(new_mark), "".join(_words(new_identifier))):
                raise ValueError(
                    "party registration mark differs from generated entity value: "
                    + binding.logical_key
                )
