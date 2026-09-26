"""Project renderable party addresses onto the task's component boundaries.

The renderer needs the full postal surface. The extraction target instead uses
the real-label convention: a separately labelled city/country is not repeated
as a complete address component. Street and site names are never geocoded.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_COMPONENT = re.compile(r"[^,;\n]+")
_SITE = re.compile(
    r"\b(?:park|terminal|garden|gardens|oasis|estate|free zone|financial cent(?:re|er)|"
    r"industrial city|science city|business district)\b",
    re.IGNORECASE,
)
_STREET_TERMINAL = re.compile(
    r"\b(?:avenida|rue de|port|calle|via|jalan)\s*$",
    re.IGNORECASE,
)


def _compact(value: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKD", value)
        if character.isalnum()
    )


def _fold_with_positions(value: str) -> tuple[str, list[int]]:
    letters: list[str] = []
    positions: list[int] = []
    for offset, character in enumerate(value):
        for folded in unicodedata.normalize("NFKD", character).casefold():
            if folded.isalnum():
                letters.append(folded)
                positions.append(offset)
    return "".join(letters), positions


def _postal_like(value: str) -> bool:
    stripped = value.strip(" .,-;:/")
    if not stripped or len(stripped) > 18 or not any(c.isdigit() for c in stripped):
        return False
    if not re.fullmatch(r"[\w .\-/]+", stripped):
        return False
    return all(len(word) <= 2 for word in re.findall(r"[^\W\d_]+", stripped))


def _remove_components(value: str, matches: list[re.Match[str]], first: int, last: int) -> str:
    if first == 0:
        raise ValueError("an address has no non-locality leading component")
    return (value[: matches[first - 1].end()] + value[matches[last].end() :]).strip(" ,;\n\t")


def _strip_country(
    value: str, country: str | None, codes: Mapping[str, str]
) -> tuple[str, str | None]:
    if not country:
        return value, None
    wanted = codes.get(_compact(country))

    def same_country(surface: str) -> bool:
        key = _compact(surface)
        return key == _compact(country) or bool(
            wanted and len(key) >= 4 and codes.get(key) == wanted
        )

    matches = list(_COMPONENT.finditer(value))
    for width in range(min(3, len(matches) - 1), 0, -1):
        first = len(matches) - width
        if same_country(" ".join(match.group() for match in matches[first:])):
            return _remove_components(value, matches, first, len(matches) - 1), "country_component"
    if not matches:
        return value, None
    last = matches[-1]
    folded, positions = _fold_with_positions(last.group())
    for split in range(1, len(folded)):
        prefix = last.group()[: positions[split - 1] + 1]
        suffix = last.group()[positions[split] :]
        if same_country(prefix) and _postal_like(suffix):
            replacement = suffix.strip(" ,-;")
            return value[: last.start()] + " " + replacement + value[last.end() :], "country_postal"
        if _postal_like(prefix) and same_country(suffix):
            replacement = prefix.strip(" ,-;")
            return value[: last.start()] + " " + replacement + value[last.end() :], "postal_country"
    if len(matches) == 1 and len(_compact(country)) >= 4:
        match = re.search(r"(?<!\w)" + re.escape(country) + r"\s*$", value, re.IGNORECASE)
        if match and match.start() > 0 and not _STREET_TERMINAL.search(value[: match.start()]):
            return value[: match.start()].rstrip(" ,;\n"), "unseparated_country_suffix"
    return value, None


def _strip_city(value: str, city: str | None) -> tuple[str, str | None]:
    if not city or _SITE.search(city):
        return value, None
    expected = _compact(city)
    matches = list(_COMPONENT.finditer(value))
    for width in range(min(3, len(matches) - 1), 0, -1):
        for first in range(len(matches) - width, 0, -1):
            last = first + width - 1
            if _compact(" ".join(match.group() for match in matches[first : last + 1])) == expected:
                return _remove_components(value, matches, first, last), "city_component"
    for match in reversed(matches[1:]):
        folded, positions = _fold_with_positions(match.group())
        if folded.startswith(expected) and len(folded) > len(expected):
            tail = match.group()[positions[len(expected)] :]
            if _postal_like(tail):
                return (
                    value[: match.start()] + " " + tail.strip(" ,-;") + value[match.end() :],
                    "city_postal",
                )
        if folded.endswith(expected) and len(folded) > len(expected):
            head = match.group()[: positions[-len(expected)]]
            if _postal_like(head):
                return (
                    value[: match.start()] + " " + head.strip(" ,-;") + value[match.end() :],
                    "postal_city",
                )
    return value, None


def project_address(
    address: str, city: str | None, country: str | None, country_codes: Mapping[str, str]
) -> tuple[str, tuple[str, ...]]:
    value, country_rule = _strip_country(address, country, country_codes)
    value, city_rule = _strip_city(value, city)
    if not value or not any(character.isalpha() for character in value):
        raise ValueError("address projection would remove all semantic address information")
    return value, tuple(rule for rule in (country_rule, city_rule) if rule)


def project_training_target(
    render_target: Mapping[str, Any], *, country_codes: Mapping[str, str]
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """Return a projected copy; the frozen render target is never mutated."""
    target = deepcopy(dict(render_target))
    parties = target["documentPatch"]["parties"]
    edits: list[dict[str, str]] = []
    for role, value in parties.items():
        if role == "carrier":
            continue
        members = value if isinstance(value, list) else [value]
        for index, party in enumerate(members):
            if not isinstance(party, dict) or not isinstance(party.get("address"), str):
                continue
            address = party["address"]
            city = party.get("city") if isinstance(party.get("city"), str) else None
            country = party.get("country") if isinstance(party.get("country"), str) else None
            projected, rules = project_address(address, city, country, country_codes)
            if not rules:
                continue
            party["address"] = projected
            path = f"documentPatch.parties.{role}"
            if isinstance(value, list):
                path += f"[{index}]"
            edits.append({"path": path + ".address", "before": address, "after": projected})
    return target, tuple(edits)
