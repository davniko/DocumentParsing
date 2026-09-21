"""Source-proven identity equality and country prefixes for auxiliary party IDs."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .models import CertifiedSemanticTemplate


def _compact(value: str) -> str:
    return "".join(c for c in value.casefold() if c.isalnum())


def _party(target: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    from .synthetic_values import _resolve_mapping_path

    result = _resolve_mapping_path(target, path)
    if result is None:
        raise ValueError(f"auxiliary identifier party is absent: {path}")
    return result


def contracts(
    template: CertifiedSemanticTemplate,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    country_codes: Mapping[str, str],
) -> tuple[tuple[tuple[str, ...], ...], dict[str, str]]:
    """Equal source IDs on equal source AND generated parties are one fact.

    We never equate different original identifiers merely because party names
    match. Country prefixes require a tax-typed binding, a matching source ISO2
    prefix and a digit-only suffix, not an arbitrary alphabetic identifier.
    """
    by_key = {b.logical_key: b for b in template.bindings}
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    prefixes = {}
    for entity in template.auxiliary_semantic_plan.entities:
        if entity.target_party_path is None:
            continue
        old, new = (
            _party(source, entity.target_party_path),
            _party(target, entity.target_party_path),
        )
        identity = tuple(
            tuple(_compact(str(p.get(k, ""))) for k in ("name", "address", "city", "country"))
            for p in (old, new)
        )
        for member in entity.members:
            if member.field not in {"tax_identifier", "registration_identifier"}:
                continue
            binding = by_key[member.logical_key]
            if (
                binding.target_paths
                or binding.value_kind != "identifier"
                or any(s.render_policy != "opaque_identifier" for s in binding.occurrences)
            ):
                continue
            values = {_compact(s.source_text) for s in binding.occurrences}
            if len(values) != 1:
                continue
            value = values.pop()
            if len(value) >= 6 and all(parts[0] and parts[1] for parts in identity):
                groups[(*identity, member.field, value)].append(binding.logical_key)
            source_code = country_codes.get(_compact(str(old.get("country", ""))))
            target_code = country_codes.get(_compact(str(new.get("country", ""))))
            if (
                member.field == "tax_identifier"
                and source_code
                and target_code
                and len(value) > 2
                and value[:2] == source_code.casefold()
                and value[2:].isdigit()
            ):
                prefixes[binding.logical_key] = target_code.upper()
    return tuple(tuple(sorted(keys)) for keys in groups.values() if len(keys) > 1), prefixes


def validate(
    groups: tuple[tuple[str, ...], ...], prefixes: Mapping[str, str], outputs: Mapping[str, Any]
) -> None:
    for keys in groups:
        if len({_compact(str(outputs[key].canonical_value)) for key in keys}) != 1:
            raise ValueError("equivalent party identifiers disagree: " + ", ".join(keys))
    for key, prefix in prefixes.items():
        if not _compact(str(outputs[key].canonical_value)).startswith(prefix.casefold()):
            raise ValueError(f"party tax identifier lost its source-proven country prefix: {key}")
