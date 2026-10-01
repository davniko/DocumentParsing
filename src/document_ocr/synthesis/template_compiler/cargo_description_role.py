"""Keep source-proven product-only cargo slots free of shipment facts.

The compiler can bind a product-only source description while the linguistic
generator expands it into a package/weight/equipment summary. That summary
belongs in other typed fields. This guard only applies when every physical
source description binding proves the same product-only surface; mixed source
roles are left to their explicit template contracts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_DESCRIPTION_PATH = re.compile(r"documentPatch\.cargoGroups\[(\d+)\]\.description\Z")
_COUNT_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?)"
)
_PACKAGE_COUNT = re.compile(
    r"(?:^|[,;]|\b(?:in|containing|packed\s+in|total\s+of)\b)\s*"
    rf"(?:\d[\d,]*|{_COUNT_WORD})\s+"
    r"(?:(?:intermediate\s+bulk|woven|jute|steel|fibre|fiber|wooden|palletized)\s+)?"
    r"(?:containers?|cartons?|boxes?|pallets?|packages?|packs?|bags?|drums?|"
    r"rolls?|bales?|crates?|bundles?|coils?|spools?|reels?|cases?|units?|"
    r"pieces?|tanks?|cans?|bottles?|sets?|lots?)\b",
    re.IGNORECASE,
)
_UNCOUNTED_OUTER_PACKAGING = re.compile(
    r"\bin\s+(?:cartons?|boxes?|pallets?|packages?|bags?|drums?|crates?)\b",
    re.IGNORECASE,
)
_LABELED_MASS = re.compile(
    r"(?<!unit\s)\b(?:gross|net|cargo)\s+(?:(?:cargo\s+)?(?:weight|mass)\s*)?"
    r"[:=]?\s*\d[\d,.]*\s*(?:kgs?|kilograms?|mt|metric\s+tonn?es?|tonn?es?|lbs?|pounds?)\b"
    r"(?!\s*(?:per\b|each\b|cartons?\b|boxes?\b|bags?\b|drums?\b|packages?\b|"
    r"pieces?\b|units?\b|or\s+(?:less|more)\b|/))",
    re.IGNORECASE,
)
_BARE_MASS_CLAUSE = re.compile(
    r"(?:^|[,;])\s*\d[\d,.]*\s+"
    r"(?:kgs?|kilograms?|mt|metric\s+tonn?es?|tonn?es?|lbs?|pounds?)\b"
    r"\s*(?=[,;.]|$)",
    re.IGNORECASE,
)
_VOLUME = re.compile(
    r"\b(?:volume|measurement)\s*[:=]?\s*\d[\d,.]*\s*"
    r"(?:cbm|m3|m³|cubic\s+met(?:er|re)s?|cft|ft3|ft³|cubic\s+feet)\b"
    r"(?!\s*(?:per\b|each\b|/))"
    r"|(?:^|[,;])\s*\d[\d,.]*\s*"
    r"(?:cbm|m3|m³|cubic\s+met(?:er|re)s?|cft|ft3|ft³|cubic\s+feet)\b"
    r"\s*(?=[,;.]|$)",
    re.IGNORECASE,
)
_TARIFF_DG = re.compile(
    r"\b(?:HS\s*(?:CODES?)?\s*[:=]?\s*\d{4,}|UN\s*(?:NO\.?|NUMBER)?\s*[:=]?\s*\d{4})\b",
    re.IGNORECASE,
)
_EQUIPMENT = re.compile(
    r"\b(?:stowed|loaded|packed|secured|shipped|carried)\s+(?:in|on|as)\s+"
    r"(?:one|two|a|an|\d+\s*x\s*)?\s*(?:\d+-foot|forty-foot|twenty-foot)?\s*"
    r"(?:high-cube|standard-height|refrigerated|reefer)?\s*containers?\b"
    r"|\b(?:stowed|loaded|packed|secured|shipped|carried)\s+at\s+"
    r"-?\d+(?:\.\d+)?\s*(?:°\s*C|C|Celsius)\b"
    r"|\b(?:under|with)\s+seal\s+[A-Z0-9-]+\b",
    re.IGNORECASE,
)
_TEMPERATURE = re.compile(
    r"\b(?:keep|kept|maintained|stored|stowed|refrigerated|carried)\s+"
    r"(?:frozen\s+)?at\s+(?:minus\s+)?-?\d+(?:\.\d+)?\s*"
    r"(?:°\s*C|C|Celsius|degrees?\s+Celsius)\b"
    r"|\b(?:temperature|setpoint)\s*[:=]?\s*(?:minus\s+)?-?\d+(?:\.\d+)?\s*"
    r"(?:°\s*C|C|Celsius|degrees?\s+Celsius)\b",
    re.IGNORECASE,
)
_PATTERNS = (
    ("package_count", _PACKAGE_COUNT),
    ("outer_packaging", _UNCOUNTED_OUTER_PACKAGING),
    ("shipment_mass", _LABELED_MASS),
    ("shipment_mass", _BARE_MASS_CLAUSE),
    ("shipment_volume", _VOLUME),
    ("tariff_or_dangerous_goods_code", _TARIFF_DG),
    ("equipment_placement", _EQUIPMENT),
    ("temperature_setpoint", _TEMPERATURE),
)


def _words(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def structured_declarations(value: str) -> frozenset[str]:
    """Detect explicit shipment declarations, not product sizes or capacities."""
    return frozenset(name for name, pattern in _PATTERNS if pattern.search(value))


def product_only_source_groups(source_target: Mapping[str, Any], template: Any) -> frozenset[int]:
    """Certify description role only from complete, exact source bindings.

    A description with no physical owner, split fragments, or a printed
    structural declaration is not promoted to this narrower contract.
    """
    groups = source_target.get("documentPatch", {}).get("cargoGroups", ())
    if not isinstance(groups, (list, tuple)):
        raise ValueError("source cargoGroups is not a sequence")
    bindings: dict[int, list[str]] = {index: [] for index in range(len(groups))}
    uncertified: set[int] = set()
    for binding in template.bindings:
        indices = {
            int(match[1])
            for path in binding.target_paths
            if (match := _DESCRIPTION_PATH.fullmatch(path))
        }
        for index in indices:
            if index >= len(groups):
                raise ValueError("description binding group index exceeds source target")
            if binding.render_mode != "target_binding" or len(binding.target_paths) != 1:
                uncertified.add(index)
                continue
            path = f"documentPatch.cargoGroups[{index}].description"
            matched = [
                occurrence.source_text
                for occurrence in binding.occurrences
                if path in occurrence.target_paths
            ]
            if not matched or any(not isinstance(value, str) for value in matched):
                uncertified.add(index)
                continue
            bindings[index].extend(matched)
    certified: set[int] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError("source cargo group is not an object")
        description = group.get("description")
        if index in uncertified or not isinstance(description, str) or not description.strip():
            continue
        surfaces = bindings[index]
        if not surfaces or structured_declarations(description):
            continue
        if any(
            _words(surface) != _words(description) or structured_declarations(surface)
            for surface in surfaces
        ):
            continue
        certified.add(index)
    return frozenset(certified)


def requested_product_only_paths(
    source_target: Mapping[str, Any],
    template: Any,
    requested_fields: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Expose only requested product-only slots to the linguistic generator."""
    certified = {
        f"documentPatch.cargoGroups[{index}].description"
        for index in product_only_source_groups(source_target, template)
    }
    requested: set[str] = set()
    for field in requested_fields:
        paths = field.get("paths")
        if not isinstance(paths, (list, tuple)) or any(not isinstance(path, str) for path in paths):
            raise ValueError("lexical request paths are missing or malformed")
        requested.update(paths)
    return tuple(sorted(certified & requested))


def validate(*, source_target: Mapping[str, Any], template: Any, target: Mapping[str, Any]) -> None:
    """Reject generated structural padding in source-proven product-only slots."""
    source_groups = source_target.get("documentPatch", {}).get("cargoGroups", ())
    generated_groups = target.get("documentPatch", {}).get("cargoGroups", ())
    if not isinstance(generated_groups, (list, tuple)) or len(generated_groups) != len(
        source_groups
    ):
        raise ValueError("generated cargo group cardinality differs from source")
    for index in product_only_source_groups(source_target, template):
        group = generated_groups[index]
        if not isinstance(group, Mapping):
            raise ValueError("generated cargo group is not an object")
        description = group.get("description")
        if not isinstance(description, str):
            raise ValueError("product-only source description lost its generated text")
        declarations = structured_declarations(description)
        if declarations:
            raise ValueError(
                "generated cargo group "
                f"{index} pads its product-only source description with "
                + ", ".join(sorted(declarations))
            )
