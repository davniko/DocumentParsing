"""Source-proven multi-level package arithmetic, outside the training schema."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise
from typing import Any

from document_ocr.synthesis.rendering import render_number_surface

from .descendant import _PACKAGE_SURFACES
from .models import SemanticBinding

_CATEGORIES = {s: k for k, values in _PACKAGE_SURFACES.items() for s in values}
# Printed singular/plural notation is a surface variant, not another package level.
_CATEGORIES.update({s + "(S)": k for s, k in list(_CATEGORIES.items()) if s + "S" in _CATEGORIES})
_NOUN = "|".join(re.escape(s) for s in sorted(_CATEGORIES, key=lambda s: (-len(s), s)))


def _term(index: int) -> str:
    return rf"(?P<n{index}>[0-9]+)\s*(?P<u{index}>{_NOUN})"


_PAIR = re.compile(
    rf"\s*{_term(0)}\s*(?:=|STC\b|SAID\s+TO\s+CONTAIN\b|CONTAINING\b)\s*{_term(1)}\s*",
    re.I,
)
_MIXED = re.compile(
    rf"\s*{_term(0)}\s*=\s*{_term(1)}\s*/\s*{_term(2)}"
    rf"\s*\+\s*{_term(3)}\s*=\s*{_term(4)}\s*",
    re.I,
)
_SCALAR = re.compile(rf"\s*{_term(0)}\s*", re.I)
_LEVEL_UNITS = {
    **_CATEGORIES,
    **dict.fromkeys(("PIECE", "PIECES", "PCE", "PCES", "PCS"), "PACKAGE_PIECE"),
    **dict.fromkeys(("IBC", "IBCS", "IBC TANKS"), "PACKAGE_INTERMEDIATE_BULK_CONTAINER"),
}
_LEVEL_NOUN = "|".join(re.escape(s) for s in sorted(_LEVEL_UNITS, key=lambda s: (-len(s), s)))
_LEVEL_NUMBER = r"(?:[1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]*|ONE)"
_LEVEL_TEXT = rf"{_LEVEL_NUMBER}\s*(?:{_LEVEL_NOUN})"
_LEVEL = re.compile(rf"(?P<number>{_LEVEL_NUMBER})\s*(?P<unit>{_LEVEL_NOUN})", re.I)
_LEVEL_FORMS = {
    "chain": re.compile(rf"\s*{_LEVEL_TEXT}(?:\s*(?:=|/|ON\b|OF\b)\s*{_LEVEL_TEXT})+\s*", re.I),
    "parenthesis": re.compile(rf"\s*{_LEVEL_TEXT}\s*\(\s*{_LEVEL_TEXT}\s*\)\s*", re.I),
    "sum": re.compile(
        rf"\s*{_LEVEL_TEXT}\s*\(\s*{_LEVEL_TEXT}(?:\s*\+\s*{_LEVEL_TEXT})+\s*\)\s*",
        re.I,
    ),
    "alias": re.compile(
        rf"\s*{_LEVEL_TEXT}\s*:\s*{_LEVEL_TEXT}\s+SLAC\s*:\s*{_LEVEL_TEXT}\s*", re.I
    ),
}
_PALLET_MARK = re.compile(
    r"\s*(?:P/NO\.?|PALLET\s+NO\.?)\s*:\s*P?1\s*-\s*P?(?P<last>[1-9][0-9]*)\s*",
    re.I,
)
_PALLET_LOT = re.compile(
    r"\s*P/NO\.?\s*:\s*[A-Z0-9]+/"
    r"(?P<count>[1-9][0-9]*)(?:-|~)(?P=count)/(?P=count)\s*",
    re.I,
)
_COMPACT_LEVELS = re.compile(
    r"(?P<outer>[1-9][0-9]*)\s*(?P<outer_unit>PLTS|PALLETS)\s*=\s*"
    r"(?P<cartons>[1-9][0-9]*)\s*(?P<carton_unit>CTNS|CARTONS)"
    r"(?:\s*=\s*(?P<pieces>[1-9][0-9]*)\s*(?P<piece_unit>PCES|PIECES))?",
    re.I,
)


def require_segmented_package_owner(source: bytes, template: Any) -> None:
    """A packing word between cargo segments needs its own dependency owner.

    Segmentation explicitly omits these words from the cargo label. Leaving an
    unowned packing noun literal while changing package categories would imply
    a packing level the source contract has not represented.
    """
    owned = {
        position
        for binding in template.bindings
        if any(
            path.endswith(".typeCategory") and ".cargoPackages[" in path
            for path in (*binding.target_paths, *binding.dependency_paths)
        )
        for slot in binding.occurrences
        for position in range(slot.byte_start, slot.byte_end)
    }
    for binding in template.bindings:
        if binding.realization.mode != "segmented_surface" or not any(
            ".cargoGroups[" in path and path.endswith(".description")
            for path in binding.target_paths
        ):
            continue
        slots = sorted(binding.occurrences, key=lambda slot: slot.byte_start)
        for before, after in pairwise(slots):
            gap = source[before.byte_end : after.byte_start]
            text = gap.decode("utf-8")
            if text.strip().upper() not in _CATEGORIES:
                continue
            start = before.byte_end + len(text[: len(text) - len(text.lstrip())].encode("utf-8"))
            if any(
                position not in owned
                for position in range(start, start + len(text.strip().encode("utf-8")))
            ):
                raise ValueError(
                    "packing noun between cargo-description segments lacks a typed "
                    f"package owner: {binding.logical_key}: {gap!r}"
                )


def _overpack_values(
    source: Mapping[str, Any], target: Mapping[str, Any], group: str, text: str
) -> tuple[str, dict[int, set[int]]] | None:
    form = next((k for k, pattern in _LEVEL_FORMS.items() if pattern.fullmatch(text)), None)
    if form is None:
        return None
    terms = list(_LEVEL.finditer(text))
    units = [_LEVEL_UNITS[m["unit"].upper()] for m in terms]
    old = [1 if m["number"].upper() == "ONE" else int(m["number"].replace(",", "")) for m in terms]
    if len(set(units)) != len(units):
        raise ValueError("overpack declaration does not distinguish packing levels")
    changes = {
        j: value
        for j in range(len(terms))
        if (value := _quantity(source, target, group, units[j], old[j])) is not None
    }
    if not changes:
        raise ValueError("overpack declaration lacks a structured source quantity")
    values = [changes.get(j, value) for j, value in enumerate(old)]
    if form == "sum":
        if units[0] != "PACKAGE_PACKAGE" or old[0] != sum(old[1:]):
            raise ValueError("overpack component sum fails source arithmetic")
        total = sum(values[1:])
        if 0 in changes and changes[0] != total:
            raise ValueError("sampled overpack total disagrees with component counts")
        values[0] = total
    elif form == "alias":
        if units[:2] != ["PACKAGE_PALLET", "PACKAGE_PIECE"] or old[0] != old[1]:
            raise ValueError("overpack count alias lacks an exact source identity")
        bound = {changes[j] for j in (0, 1) if j in changes}
        if len(bound) > 1:
            raise ValueError("sampled pallet and generic piece alias counts disagree")
        values[0] = values[1] = next(iter(bound)) if bound else old[0]
    elif form == "parenthesis" and len(terms) == 2 and old[0] == old[1]:
        # A sole structured packing level plus an equal enclosing level in the
        # same complete declaration is a one-to-one count, not an independent
        # immutable pallet total. Do not infer this from equality when another
        # package fact could share the declaration's cargo group.
        packages = [
            p for p in source["documentPatch"].get("cargoPackages", ()) if p["groupId"] == group
        ]
        if len(packages) == 1 and len(changes) == 1:
            values[0] = values[1] = next(iter(changes.values()))
    # A declared inner packing level cannot contain fewer units than its outer
    # level. Preserve the proven source nesting, not an inferred fixed ratio.
    if form in {"chain", "parenthesis"}:
        for a, b in zip(range(len(terms) - 1), range(1, len(terms)), strict=True):
            if old[a] < old[b] and values[a] > values[b]:
                raise ValueError("sampled packing levels invert their source containment")
            if old[a] > old[b] and values[a] < values[b]:
                raise ValueError("sampled packing levels invert their source containment")
    rendered = text
    for m, before, after in reversed(list(zip(terms, old, values, strict=True))):
        from .descendant import _case_like, _number_to_words

        rendered = (
            rendered[: m.start("number")]
            + (
                _case_like(m["number"], _number_to_words(after))
                if m["number"].isalpha()
                else render_number_surface(m["number"], before, after)
            )
            + rendered[m.end("number") :]
        )
    pallets: dict[int, set[int]] = defaultdict(set)
    for unit, before, after in zip(units, old, values, strict=True):
        if unit == "PACKAGE_PALLET":
            pallets[before].add(after)
    return rendered, pallets


def one_to_one_target_surfaces(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, str]:
    """Project proved one-to-one overpack counts into extraction-facing prose.

    Only a complete two-level parenthetical declaration with equal source
    counts, one structured package row, and exactly one changed structured
    level qualifies. Other packing hierarchies remain under their existing
    explicit contracts.
    """
    result = {}
    for group_index, group in enumerate(source.get("documentPatch", {}).get("cargoGroups", ())):
        for index, text in enumerate(group.get("additionalInformation", ())):
            if _LEVEL_FORMS["parenthesis"].fullmatch(text) is None:
                continue
            terms = list(_LEVEL.finditer(text))
            if len(terms) != 2 or terms[0]["number"] != terms[1]["number"]:
                continue
            parsed = _overpack_values(source, target, group["groupId"], text)
            if parsed is None:
                continue
            rendered, _ = parsed
            same_group = [
                package
                for package in source["documentPatch"].get("cargoPackages", ())
                if package["groupId"] == group["groupId"]
            ]
            revised_terms = list(_LEVEL.finditer(rendered))
            if (
                len(same_group) == 1
                and len(revised_terms) == 2
                and revised_terms[0]["number"] == revised_terms[1]["number"]
                and rendered != text
            ):
                path = f"documentPatch.cargoGroups[{group_index}].additionalInformation[{index}]"
                result[path] = rendered
    return result


def numeric_composite_target_surfaces(
    template: Any,
    source_text: bytes,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    numeric_values: Mapping[str, Any],
) -> dict[str, str]:
    """Express compact pallet/carton/piece labels using proved numeric owners.

    A source-only pallet level may be sampled outside the extraction schema.
    When that level also appears inside a labelled cargo phrase, its frozen
    numeric receipt must supply the phrase before any linguistic generation.
    Neither a coincidentally equal number nor a row's position is ownership.
    """
    from .numeric_auxiliary import numeric_bindings

    source_patch = source.get("documentPatch", {})
    target_patch = target.get("documentPatch", {})
    if len(source_patch.get("cargoGroups", ())) != 1:
        return {}
    pallet_values = []
    for binding in numeric_bindings(template):
        prepared = numeric_values.get(binding.logical_key)
        if prepared is None or prepared.contract.role != "cargo_quantity":
            continue
        if not any(
            re.fullmatch(r"\s*[0-9]+\s*(?:PALLETS?|PLTS?)\s*", slot.source_text, re.I)
            or (
                re.fullmatch(r"[0-9]+", slot.source_text)
                and re.match(rb"[ \t]+(?:PALLETS?|PLTS?)\b", source_text[slot.byte_end :], re.I)
            )
            for slot in binding.occurrences
        ):
            continue
        old = int(prepared.contract.source_value)
        new = Decimal(prepared.value)
        if new != new.to_integral_value() or new <= 0:
            raise ValueError("prepared pallet quantity is not a positive integer")
        pallet_values.append((old, int(new)))

    result = {}
    for group_index, group in enumerate(source_patch.get("cargoGroups", ())):
        group_id = group["groupId"]
        for index, text in enumerate(group.get("additionalInformation", ())):
            match = _COMPACT_LEVELS.fullmatch(text)
            if match is None:
                continue
            old_outer = int(match["outer"])
            exact = {new for old, new in pallet_values if old == old_outer}
            if exact:
                if len(exact) != 1:
                    raise ValueError("compact pallet label has conflicting exact numeric owners")
                new_outer = exact.pop()
            elif len(pallet_values) >= 2 and sum(old for old, _ in pallet_values) == old_outer:
                new_outer = sum(new for _, new in pallet_values)
            else:
                continue  # No certified mutable source-only pallet owner.
            replacements = {"outer": new_outer}
            for key, category in (("cartons", "PACKAGE_CARTON"), ("pieces", "PACKAGE_PIECE")):
                if match[key] is None:
                    continue
                old_count = int(match[key])
                candidates = [
                    i
                    for i, package in enumerate(source_patch.get("cargoPackages", ()))
                    if package["groupId"] == group_id
                    and package.get("typeCategory") == category
                    and package.get("quantity") == old_count
                ]
                if len(candidates) != 1:
                    raise ValueError("compact package level lacks one structured source owner")
                current = target_patch["cargoPackages"][candidates[0]]
                if (
                    current.get("typeCategory") != category
                    or type(current.get("quantity")) is not int
                ):
                    raise ValueError("sampled compact package level changed its category")
                replacements[key] = current["quantity"]
            rendered = text
            for key in sorted(replacements, key=lambda name: match.start(name), reverse=True):
                start, end = match.span(key)
                rendered = rendered[:start] + str(replacements[key]) + rendered[end:]
            result[f"documentPatch.cargoGroups[{group_index}].additionalInformation[{index}]"] = (
                rendered
            )
    return result


def pending_numeric_composite_surfaces(
    target: Mapping[str, Any], surfaces: Mapping[str, str]
) -> dict[str, str]:
    """Do not seek an agent-field owner for a phrase already assembled by host prose."""
    from .descendant import _resolve_path

    return {
        path: rendered
        for path, rendered in surfaces.items()
        if _resolve_path(target, path) != rendered
    }


def _pallet_partition(counts: list[int], pallets: Mapping[int, set[int]]) -> list[int]:
    total = pallets.get(sum(counts), set())
    if len(total) == 1:
        return _apportion(next(iter(total)), counts)
    if all(len(pallets.get(count, set())) == 1 for count in counts):
        return [next(iter(pallets[count])) for count in counts]
    raise ValueError("pallet interval lacks a unique declared overpack total")


def overpack_surfaces(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, str]:
    """Bind complete packing-level declarations and their numbered pallet marks.

    These are runtime lexical surfaces, not equality equations. A structured
    source quantity must prove one level; other levels remain fixed. A pallet
    interval additionally requires an exact same-group declared pallet count.
    """
    output = {}
    equations = generate(source, target)
    for gi, group in enumerate(source.get("documentPatch", {}).get("cargoGroups", ())):
        pallets: dict[int, set[int]] = defaultdict(set)
        for i, text in enumerate(group.get("additionalInformation", ())):
            path = f"documentPatch.cargoGroups[{gi}].additionalInformation[{i}]"
            if path in equations:
                # Reuse the existing joint solver: a component is not itself the
                # structured total, and must not be reinterpreted in isolation.
                rendered = equations[path]
                for old, new in zip(_LEVEL.finditer(text), _LEVEL.finditer(rendered), strict=True):
                    if _LEVEL_UNITS[old["unit"].upper()] == "PACKAGE_PALLET":
                        pallets[int(old["number"])].add(int(new["number"]))
                output[path] = rendered
                continue
            parsed = _overpack_values(source, target, group["groupId"], text)
            if parsed is None:
                continue
            rendered, levels = parsed
            output[f"documentPatch.cargoGroups[{gi}].additionalInformation[{i}]"] = rendered
            for before, after in levels.items():
                pallets[before].update(after)
        intervals = [
            (i, text, m)
            for i, text in enumerate(group.get("marksAndNumbers", ()))
            if (m := _PALLET_MARK.fullmatch(text))
        ]
        if intervals and pallets:
            counts = [int(m["last"]) for _, _, m in intervals]
            for (i, text, m), value in zip(
                intervals,
                _owned_pallet_partition(source, target, group["groupId"], counts, pallets),
                strict=True,
            ):
                output[f"documentPatch.cargoGroups[{gi}].marksAndNumbers[{i}]"] = (
                    text[: m.start("last")] + str(value) + text[m.end("last") :]
                )
        lots = [
            (i, text, m)
            for i, text in enumerate(group.get("marksAndNumbers", ()))
            if (m := _PALLET_LOT.fullmatch(text))
        ]
        if lots and pallets:
            counts = [int(m["count"]) for _, _, m in lots]
            for (i, text, m), value in zip(
                lots,
                _owned_pallet_partition(source, target, group["groupId"], counts, pallets),
                strict=True,
            ):
                # The three suffix positions are the same declared lot count.
                start = m.start("count")
                output[f"documentPatch.cargoGroups[{gi}].marksAndNumbers[{i}]"] = text[
                    :start
                ] + re.sub(r"[0-9]+", str(value), text[start:])
    return output


def _owned_pallet_partition(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    group: str,
    counts: list[int],
    pallets: Mapping[int, set[int]],
) -> list[int]:
    """Use a group's complete printed pallet inventory before a global total.

    A first-group information field can carry the document-wide packing
    equation without making that group's numbered marks cover other groups.
    """
    indices = [
        i
        for i, row in enumerate(source["documentPatch"].get("cargoPackages", ()))
        if row["groupId"] == group and row.get("typeCategory") == "PACKAGE_PALLET"
    ]
    if indices and all("quantity" in source["documentPatch"]["cargoPackages"][i] for i in indices):
        old = sum(source["documentPatch"]["cargoPackages"][i]["quantity"] for i in indices)
        if old == sum(counts):
            rows = [target["documentPatch"]["cargoPackages"][i] for i in indices]
            if any(row.get("typeCategory") != "PACKAGE_PALLET" for row in rows):
                raise ValueError("numbered pallet marks require their owned pallet category")
            return _apportion(sum(row["quantity"] for row in rows), counts)
    return _pallet_partition(counts, pallets)


def require_pallet_mark_contract(template: Any, target: Mapping[str, Any]) -> None:
    """Numbered pallet marks are counts, not freely generated shipment codes."""
    paths = {
        f"documentPatch.cargoGroups[{gi}].marksAndNumbers[{i}]"
        for gi, group in enumerate(target.get("documentPatch", {}).get("cargoGroups", ()))
        for i, text in enumerate(group.get("marksAndNumbers", ()))
        if _PALLET_MARK.fullmatch(text) or _PALLET_LOT.fullmatch(text)
    }
    if not paths:
        return
    declared = overpack_surfaces(target, target)
    range_keys = {
        key
        for constraint in template.coherence_constraints
        if constraint.kind
        in {"inclusive_range_cardinality", "aggregate_inclusive_range_cardinality"}
        for key in constraint.member_logical_keys
    }
    owned = {
        path
        for binding in template.bindings
        if binding.logical_key in range_keys
        for path in binding.target_paths
    }
    missing = paths - declared.keys() - owned
    if missing:
        raise ValueError(
            "numbered pallet marks lack a typed count/range dependency contract: "
            + ", ".join(sorted(missing))
        )


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
    for path, text in overpack_surfaces(source, source).items():
        if ".additionalInformation[" not in path:
            continue
        matches = [
            i
            for i, term in enumerate(_LEVEL.finditer(text))
            if (
                _LEVEL_UNITS[term["unit"].upper()],
                1 if term["number"].upper() == "ONE" else int(term["number"].replace(",", "")),
            )
            == identity
        ]
        if len(matches) == 1:
            paths.append(path)
    if len(paths) > 1:
        raise ValueError("numeric surface has ambiguous package-equation ownership")
    return paths[0] if paths else None


def require_numeric_level_coverage(source: bytes, template: Any, target: Mapping[str, Any]) -> None:
    """Do not independently scale unbound pallet components of a printed overpack.

    Whole count+noun bindings use binding_dependency. Bare per-container counts
    require a component-to-level contract; adjacency proves their unit but does
    not prove which marked lot owns them. Such sources need explicit review.
    """
    from .descendant import _resolve_path

    for binding in template.bindings:
        categories = [
            p
            for p in binding.target_paths
            if re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.typeCategory", p)
        ]
        if not categories or not any(".additionalInformation[" in p for p in binding.target_paths):
            continue
        for slot in binding.occurrences:
            term = _LEVEL.match(slot.source_text.strip())
            if term is None:
                continue
            category = _LEVEL_UNITS[term["unit"].upper()]
            count = 1 if term["number"].upper() == "ONE" else int(term["number"].replace(",", ""))
            for path in categories:
                if _resolve_path(target, path) != category:
                    continue
                package = _resolve_path(target, path.removesuffix(".typeCategory"))
                if count == package.get("quantity"):
                    continue
                quantity_paths = [
                    p
                    for p in (*binding.target_paths, *binding.dependency_paths)
                    if p.endswith(".packageQuantity")
                ]
                if not any(_resolve_path(target, p) == count for p in quantity_paths):
                    raise ValueError(
                        "printed package subtotal lacks a typed allocation-quantity contract: "
                        + binding.logical_key
                    )
    declarations = overpack_surfaces(target, target)
    if not any(
        _LEVEL_UNITS[term["unit"].upper()] == "PACKAGE_PALLET"
        for path, text in declarations.items()
        if ".additionalInformation[" in path
        for term in _LEVEL.finditer(text)
    ):
        return
    from .numeric_auxiliary import numeric_bindings

    for binding in numeric_bindings(template):
        whole = [_SCALAR.fullmatch(slot.source_text) for slot in binding.occurrences]
        if (
            whole
            and all(
                m is not None and _CATEGORIES[m["u0"].upper()] == "PACKAGE_PALLET" for m in whole
            )
            and binding_dependency(binding, target) is None
        ):
            raise ValueError(
                "per-container pallet component lacks a unique overpack dependency contract "
                "before synthesis: " + binding.logical_key
            )
        for slot in binding.occurrences:
            if re.fullmatch(r"[1-9][0-9]*", slot.source_text) and re.match(
                rb"[ \t]+(?:PALLETS?|PLTS?)\b", source[slot.byte_end :], re.I
            ):
                raise ValueError(
                    "bare per-container pallet count needs an overpack component dependency "
                    "contract before synthesis: " + binding.logical_key
                )


def binding_value(
    binding: SemanticBinding, source: Mapping[str, Any], target: Mapping[str, Any], path: str
) -> int:
    from .descendant import _resolve_path

    if binding_dependency(binding, source) != path:
        raise ValueError("numeric equation dependency lacks exact source proof")
    old = _SCALAR.fullmatch(binding.occurrences[0].source_text)
    assert old is not None
    original = _resolve_path(source, path)
    before = list(_LEVEL.finditer(original))
    rendered = overpack_surfaces(source, target)[path]
    if _resolve_path(target, path) != rendered:
        raise ValueError("numeric equation dependency disagrees with frozen target prose")
    after = list(_LEVEL.finditer(rendered))
    assert len(before) == len(after)
    index = next(
        i
        for i, term in enumerate(before)
        if (
            _LEVEL_UNITS[term["unit"].upper()],
            1 if term["number"].upper() == "ONE" else int(term["number"].replace(",", "")),
        )
        == (_CATEGORIES[old["u0"].upper()], int(old["n0"]))
    )
    return (
        1
        if after[index]["number"].upper() == "ONE"
        else int(after[index]["number"].replace(",", ""))
    )


def _replace(text: str, match: re.Match[str], numbers: Mapping[int, int]) -> str:
    for index in sorted(numbers, key=lambda i: match.start(f"n{i}"), reverse=True):
        start, end = match.span(f"n{index}")
        text = text[:start] + str(numbers[index]) + text[end:]
    return text


def _quantity_indices(
    source: Mapping[str, Any], group: str, category: str, total: int
) -> list[int]:
    candidates = [
        i
        for i, p in enumerate(source["documentPatch"].get("cargoPackages", ()))
        if p["groupId"] == group
        and p.get("typeCategory") == category
        and p.get("quantity") == total
    ]
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
            return []
    return candidates


def _quantity(
    source: Mapping[str, Any], target: Mapping[str, Any], group: str, category: str, total: int
) -> int | None:
    candidates = _quantity_indices(source, group, category, total)
    if not candidates:
        return None
    if any(
        target["documentPatch"]["cargoPackages"][i].get("typeCategory") != category
        for i in candidates
    ):
        raise ValueError("package equation fixes its printed package-level categories")
    quantities = [target["documentPatch"]["cargoPackages"][i]["quantity"] for i in candidates]
    if len(candidates) > 1 and all(
        source["documentPatch"]["cargoPackages"][i]["quantity"] == total for i in candidates
    ):
        # Repeated equal rows are candidate owners, not additive components.
        # The statement remains true for every possible owner only if the new
        # values agree. No row ownership is inferred from coincidental equality.
        if len(set(quantities)) != 1:
            raise ValueError("candidate package owners disagree after generation")
        value = quantities[0]
    else:
        value = sum(quantities)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("package equation requires a positive integer total")
    return value


def required_package_categories(source: Mapping[str, Any]) -> dict[int, str]:
    """Expose the same proved packing constraints BEFORE categorical sampling.

    Only complete equations accepted by the existing solvers are interpreted.
    Their quantities still vary; their explicit distinct level nouns do not.
    """
    solved = {**generate(source, source), **overpack_surfaces(source, source)}
    result: dict[int, str] = {}
    groups = source.get("documentPatch", {}).get("cargoGroups", ())
    for path, text in solved.items():
        index = re.fullmatch(r"documentPatch\.cargoGroups\[(\d+)\]\.[A-Za-z]+\[\d+\]", path)
        if index is None:
            raise ValueError("packing equation has an unsupported owner path")
        group = groups[int(index[1])]["groupId"]
        pair, mixed = _PAIR.fullmatch(text), _MIXED.fullmatch(text)
        match = mixed if mixed is not None else pair
        if match is not None:
            terms = [
                (int(match[f"n{i}"]), _CATEGORIES[match[f"u{i}"].upper()])
                for i in range(5 if mixed is not None else 2)
            ]
        else:
            terms = [
                (
                    1 if m["number"].upper() == "ONE" else int(m["number"].replace(",", "")),
                    _LEVEL_UNITS[m["unit"].upper()],
                )
                for m in _LEVEL.finditer(text)
            ]
        for total, category in terms:
            for package_index in _quantity_indices(source, group, category, total):
                previous = result.setdefault(package_index, category)
                if previous != category:
                    raise ValueError("packing equations impose conflicting package categories")
    return result


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


def mark_ranges(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, str]:
    """Apportion carton-number ranges only when they prove a complete group total.

    Each source range starts at one within its own marked lot. Keep those lot
    identities and their source order; do not relabel every lot with the total.
    Other marking languages remain under their explicit lexical contracts.
    """
    result = {}
    grammar = re.compile(r"\s*C/NO\.?:\s*1\s*-\s*(?P<last>[1-9][0-9]*)\s*", re.I)
    for group_index, group in enumerate(source.get("documentPatch", {}).get("cargoGroups", ())):
        rows = [
            (i, text, match)
            for i, text in enumerate(group.get("marksAndNumbers", ()))
            if (match := grammar.fullmatch(text))
        ]
        if not rows:
            continue
        counts = [int(match["last"]) for _, _, match in rows]
        total = _quantity(source, target, group["groupId"], "PACKAGE_CARTON", sum(counts))
        if total is None:
            continue
        values = _apportion(total, counts)
        rendered = [
            text[: match.start("last")] + str(value) + text[match.end("last") :]
            for (_, text, match), value in zip(rows, values, strict=True)
        ]
        if len(set(rendered)) != len(rendered):
            raise ValueError("sampled carton count collapses distinct marked lot ranges")
        result.update(
            {
                f"documentPatch.cargoGroups[{group_index}].marksAndNumbers[{i}]": text
                for (i, _, _), text in zip(rows, rendered, strict=True)
            }
        )
    return result


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
