"""Source-proven multi-level package arithmetic, outside the training schema."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from .descendant import _PACKAGE_SURFACES
from .models import SemanticBinding

_CATEGORIES = {s: k for k, values in _PACKAGE_SURFACES.items() for s in values}
_NOUN = "|".join(sorted(_CATEGORIES, key=lambda s: (-len(s), s)))


def _term(index: int) -> str:
    return rf"(?P<n{index}>[0-9]+)\s*(?P<u{index}>{_NOUN})"


_PAIR = re.compile(rf"\s*{_term(0)}\s*=\s*{_term(1)}\s*", re.I)
_MIXED = re.compile(
    rf"\s*{_term(0)}\s*=\s*{_term(1)}\s*/\s*{_term(2)}"
    rf"\s*\+\s*{_term(3)}\s*=\s*{_term(4)}\s*",
    re.I,
)
_SCALAR = re.compile(rf"\s*{_term(0)}\s*", re.I)


def binding_dependency(binding: SemanticBinding, source: Mapping[str, Any]) -> str | None:
    """Tie a repeated numeric surface to its unique same-unit equation term."""
    if binding.target_paths:
        return None
    terms = [_SCALAR.fullmatch(slot.source_text) for slot in binding.occurrences]
    if not terms or any(m is None for m in terms):
        return None
    identities = {(_CATEGORIES[m["u0"].upper()], int(m["n0"])) for m in terms if m is not None}
    if len(identities) != 1:
        return None
    identity = next(iter(identities))
    paths = []
    for path, text in generate(source, source).items():
        match = _MIXED.fullmatch(text) or _PAIR.fullmatch(text)
        assert match is not None
        matches = [
            i
            for i in range(5 if "n4" in match.groupdict() else 2)
            if (_CATEGORIES[match[f"u{i}"].upper()], int(match[f"n{i}"])) == identity
        ]
        if len(matches) == 1:
            paths.append(path)
    if len(paths) > 1:
        raise ValueError("numeric surface has ambiguous package-equation ownership")
    return paths[0] if paths else None


def binding_value(
    binding: SemanticBinding, source: Mapping[str, Any], target: Mapping[str, Any], path: str
) -> int:
    from .descendant import _resolve_path

    if binding_dependency(binding, source) != path:
        raise ValueError("numeric equation dependency lacks exact source proof")
    old = _SCALAR.fullmatch(binding.occurrences[0].source_text)
    assert old is not None
    original = _resolve_path(source, path)
    before = _MIXED.fullmatch(original) or _PAIR.fullmatch(original)
    rendered = generate(source, target)[path]
    if _resolve_path(target, path) != rendered:
        raise ValueError("numeric equation dependency disagrees with frozen target prose")
    after = _MIXED.fullmatch(rendered) or _PAIR.fullmatch(rendered)
    assert before is not None and after is not None
    index = next(
        i
        for i in range(5 if "n4" in before.groupdict() else 2)
        if (_CATEGORIES[before[f"u{i}"].upper()], int(before[f"n{i}"]))
        == (_CATEGORIES[old["u0"].upper()], int(old["n0"]))
    )
    return int(after[f"n{index}"])


def _replace(text: str, match: re.Match[str], numbers: Mapping[int, int]) -> str:
    for index in sorted(numbers, key=lambda i: match.start(f"n{i}"), reverse=True):
        start, end = match.span(f"n{index}")
        text = text[:start] + str(numbers[index]) + text[end:]
    return text


def _quantity(
    source: Mapping[str, Any], target: Mapping[str, Any], group: str, category: str, total: int
) -> int | None:
    candidates = [
        i
        for i, p in enumerate(source["documentPatch"].get("cargoPackages", ()))
        if p["groupId"] == group
        and p.get("typeCategory") == category
        and p.get("quantity") == total
    ]
    if len(candidates) > 1:
        raise ValueError("package equation lacks a unique structured total")
    if not candidates:
        # Some first-group summaries explicitly total all packages in the document.
        candidates = [
            i
            for i, p in enumerate(source["documentPatch"].get("cargoPackages", ()))
            if p.get("typeCategory") == category and "quantity" in p
        ]
        if (
            len(candidates) < 2
            or sum(source["documentPatch"]["cargoPackages"][i]["quantity"] for i in candidates)
            != total
        ):
            return None
    value = sum(target["documentPatch"]["cargoPackages"][i]["quantity"] for i in candidates)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("package equation requires a positive integer total")
    return value


def _apportion(total: int, weights: list[int]) -> list[int]:
    if total < len(weights) or any(w < 1 for w in weights):
        raise ValueError("package equation cannot allocate positive component counts")
    exact = [Fraction(total * w, sum(weights)) for w in weights]
    values = [int(v) for v in exact]
    for i in sorted(range(len(values)), key=lambda i: (-(exact[i] - values[i]), i))[
        : total - sum(values)
    ]:
        values[i] += 1
    if any(v < 1 for v in values):
        raise ValueError("package equation allocation would erase a printed component")
    return values


def generate(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, str]:
    """Retain outer topology; solve inner counts from an exact source identity.

    The supported grammars are complete unit equations, not incidental numbers
    in prose. Independent equations sharing a total require both their outer
    and inner sums to prove the same partition before proportional allocation.
    """
    output = {}
    for group_index, group in enumerate(source.get("documentPatch", {}).get("cargoGroups", ())):
        pairs = defaultdict(list)
        for i, text in enumerate(group.get("additionalInformation", ())):
            path = f"documentPatch.cargoGroups[{group_index}].additionalInformation[{i}]"
            mixed = _MIXED.fullmatch(text)
            if mixed:
                nums = [int(mixed[f"n{j}"]) for j in range(5)]
                units = tuple(_CATEGORIES[mixed[f"u{j}"].upper()] for j in range(5))
                if (
                    units[0] != "PACKAGE_PACKAGE"
                    or len(set(units[2:])) != 1
                    or units[1] == units[2]
                    or nums[0] != nums[1] + nums[3]
                    or nums[4] != nums[2] + nums[3]
                ):
                    raise ValueError("mixed package equation fails source arithmetic")
                total = _quantity(source, target, group["groupId"], units[4], nums[4])
                if total is None:
                    raise ValueError("mixed package equation lacks a structured inner total")
                if total <= nums[3]:
                    raise ValueError("mixed package equation would erase its packed component")
                output[path] = _replace(text, mixed, {2: total - nums[3], 4: total})
                continue
            pair = _PAIR.fullmatch(text)
            if pair:
                units = tuple(_CATEGORIES[pair[f"u{j}"].upper()] for j in range(2))
                if units[0] == units[1]:
                    raise ValueError("package-level equation requires distinct units")
                pairs[units].append((path, text, pair))
        for units, rows in pairs.items():
            totals = [
                i
                for i, (_, _, m) in enumerate(rows)
                if len(rows) == 1
                or (
                    len(rows) > 2
                    and all(
                        int(m[f"n{j}"])
                        == sum(
                            int(other[f"n{j}"]) for k, (_, _, other) in enumerate(rows) if k != i
                        )
                        for j in range(2)
                    )
                )
            ]
            if len(totals) != 1:
                raise ValueError("package equations lack a unique complete two-level sum")
            total_index = totals[0]
            path, text, match = rows[total_index]
            components = [row for i, row in enumerate(rows) if i != total_index]
            changes: dict[str, dict[int, int]] = defaultdict(dict)
            for axis, category in enumerate(units):
                total = _quantity(
                    source, target, group["groupId"], category, int(match[f"n{axis}"])
                )
                if total is None:
                    continue  # This level is explicitly fixed scenario topology.
                changes[path][axis] = total
                if components:
                    allocated = _apportion(total, [int(m[f"n{axis}"]) for _, _, m in components])
                    for (component_path, _, _), value in zip(components, allocated, strict=True):
                        changes[component_path][axis] = value
            if not changes:
                raise ValueError("package equation lacks a structured total on either level")
            for path, text, match in rows:
                output[path] = _replace(text, match, changes[path])
    return output
