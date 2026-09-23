"""Source-proven package quantities in target-backed prose.

Generate arithmetic before linguistic facts, and verify it again before accepting
the target. A quantity is associated by both its source value and package noun in
the same cargo group, never by replacing arbitrary digits across a document.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any

from . import nested_package_prose, package_equations
from .models import AggregateRangeConstraint, CertifiedSemanticTemplate, NumericValuesConstraint

_TEXT = re.compile(
    r"documentPatch\.cargoGroups\[(\d+)\]\."
    r"(?:description|additionalInformation\[\d+\]|marksAndNumbers\[\d+\]|"
    r"handlingInstructions\[\d+\])$"
)
_COUNT = re.compile(
    r"(?<![A-Za-z0-9.])(?P<number>[0-9]+(?:,[0-9]{3})*)(?![0-9.,])"
    r"\s*(?P<words>[A-Za-z]+(?:[ \t]+[A-Za-z]+)*)"
)
_INTEGER = re.compile(r"(?<![A-Za-z0-9.])([0-9]+(?:,[0-9]{3})*)(?![A-Za-z0-9.,])")
_PER_PACKAGE_MASS = re.compile(
    r"(?<![A-Za-z0-9.,])(?P<number>[0-9]+(?:[.,][0-9]+)*)\s*"
    r"(?P<unit>KGS?|KGM|KILOGRAMS?|MT|TONNES?|LBS?|POUNDS?)\s+EACH\b",
    re.I,
)


def package_mass_values(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Decimal]:
    """Prove a complete group mass from its explicitly bound per-package mass.

    Package category slots can contain additional numeric facts ("BOXES OF 25KG
    EACH"). Their unchanged category does not make that arithmetic independent.
    Require the printed unit product to equal the complete source group mass;
    arbitrary digits and unrelated groups never provide a dependency contract.
    """
    from . import descendant as r

    updates: dict[str, Decimal] = {}
    factors = {
        "kilogram": Decimal(1),
        "metric_tonne": Decimal(1000),
        "pound": Decimal("0.45359237"),
    }
    for binding in template.bindings:
        package_indices = {
            int(match[1])
            for path in binding.target_paths
            if (
                match := re.fullmatch(
                    r"documentPatch\.cargoPackages\[(\d+)\]\.(?:typeCategory|typeDescription)", path
                )
            )
        }
        if not package_indices:
            continue
        for slot in binding.occurrences:
            for match in _PER_PACKAGE_MASS.finditer(slot.source_text):
                if len(package_indices) != 1:
                    raise ValueError("per-package mass has ambiguous package ownership")
                index = next(iter(package_indices))
                old_package = source["documentPatch"]["cargoPackages"][index]
                new_package = target["documentPatch"]["cargoPackages"][index]
                if "quantity" not in old_package or "quantity" not in new_package:
                    raise ValueError("per-package mass requires a declared package quantity")
                unit = match["unit"].upper()
                factor = (
                    Decimal(1)
                    if unit.startswith("K")
                    else Decimal("0.45359237")
                    if unit.startswith(("L", "P"))
                    else Decimal(1000)
                )
                candidates: dict[str, set[Decimal]] = {}
                for mass in r._numeric_interpretations(match["number"]):
                    if mass <= 0:
                        continue
                    for group_index, group in enumerate(
                        source["documentPatch"].get("cargoGroups", [])
                    ):
                        if group["groupId"] != old_package["groupId"]:
                            continue
                        for field in ("netWeight", "grossWeight"):
                            if field not in group:
                                continue
                            measurement = group[field]
                            target_measurement = target["documentPatch"]["cargoGroups"][
                                group_index
                            ][field]
                            if target_measurement["unit"] != measurement["unit"]:
                                raise ValueError("per-package mass target unit changed")
                            if (
                                mass * factor * old_package["quantity"]
                                == Decimal(str(measurement["value"])) * factors[measurement["unit"]]
                            ):
                                path = f"documentPatch.cargoGroups[{group_index}].{field}.value"
                                candidates.setdefault(path, set()).add(
                                    mass
                                    * factor
                                    * new_package["quantity"]
                                    / factors[measurement["unit"]]
                                )
                if not candidates or any(len(values) != 1 for values in candidates.values()):
                    raise ValueError("per-package mass lacks an exact source group-total proof")
                for path, values in candidates.items():
                    value = next(iter(values))
                    if updates.setdefault(path, value) != value:
                        raise ValueError("per-package mass contracts disagree")
    return updates


def validate_package_masses(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    from . import descendant as r

    for path, expected in package_mass_values(template, source, target).items():
        if Decimal(str(r._resolve_path(target, path))) != expected:
            raise ValueError(f"per-package mass contradicts generated total: {path}")


@dataclass(frozen=True)
class CountMention:
    start: int
    end: int
    noun_end: int
    quantity_paths: tuple[str, ...]
    aggregate: bool = False


def mentions(source: Mapping[str, Any], path: str, text: str) -> tuple[CountMention, ...]:
    from . import descendant as r

    match = _TEXT.fullmatch(path)
    if match is None:
        return ()
    patch = source["documentPatch"]
    group = patch["cargoGroups"][int(match[1])]["groupId"]
    result = []
    for printed in _COUNT.finditer(text):
        number = int(printed["number"].replace(",", ""))
        words = list(re.finditer(r"[A-Za-z]+", printed["words"]))
        candidates: list[tuple[str, int]] = []
        for index, package in enumerate(patch.get("cargoPackages", [])):
            if package["groupId"] != group or package.get("quantity") != number:
                continue
            meaning = package.get("typeCategory") or package.get("typeDescription")
            if not meaning:
                continue
            for word in words:
                prefix = printed["words"][: word.end()]
                if r._string_semantics_match(meaning, prefix):
                    candidates.append(
                        (f"documentPatch.cargoPackages[{index}].quantity", word.end())
                    )
                    break
        if candidates:
            result.append(
                CountMention(
                    printed.start("number"),
                    printed.end("number"),
                    printed.start("words") + max(end for _, end in candidates),
                    tuple(p for p, _ in candidates),
                )
            )
            continue
        # A whole-document summary may be filed under the first cargo group.
        # Prove its scope from the same noun and exact complete source sum.
        aggregate_candidates: list[tuple[str, int]] = []
        for index, package in enumerate(patch.get("cargoPackages", [])):
            meaning = package.get("typeCategory") or package.get("typeDescription")
            if not meaning or "quantity" not in package:
                continue
            for word in words:
                if r._string_semantics_match(meaning, printed["words"][: word.end()]):
                    aggregate_candidates.append(
                        (f"documentPatch.cargoPackages[{index}].quantity", word.end())
                    )
                    break
        if (
            len(aggregate_candidates) > 1
            and sum(r._resolve_path(source, p) for p, _ in aggregate_candidates) == number
        ):
            result.append(
                CountMention(
                    printed.start("number"),
                    printed.end("number"),
                    printed.start("words") + max(end for _, end in aggregate_candidates),
                    tuple(p for p, _ in aggregate_candidates),
                    True,
                )
            )
    return tuple(result)


def generate(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any], target: Mapping[str, Any]
) -> tuple[dict[str, str], frozenset[str]]:
    from . import descendant as r

    updates: dict[str, str] = {}
    formal: set[str] = set()
    for path, text in r._flatten_leaves(source).items():
        if not isinstance(text, str) or not _TEXT.fullmatch(path):
            continue
        counts = mentions(source, path, text)
        replacements: dict[tuple[int, int], str] = {}
        covered: set[int] = set()
        for count in counts:
            values = (
                {sum(r._resolve_path(target, p) for p in count.quantity_paths)}
                if count.aggregate
                else {r._resolve_path(target, p) for p in count.quantity_paths}
            )
            if len(values) != 1:
                raise ValueError(f"package prose has ambiguous quantity ownership: {path}")
            value = values.pop()
            old = text[count.start : count.end]
            replacements[(count.start, count.end)] = (
                format(value, ",") if "," in old else str(value)
            )
            covered.update(range(count.start, count.noun_end))
        # Compiler-confirmed numeric relationships also cover quantity columns
        # without an adjacent noun. Require a unique token in this exact field.
        for constraint in template.coherence_constraints:
            if not isinstance(constraint, NumericValuesConstraint):
                continue
            bindings = [
                b for b in template.bindings if b.logical_key in constraint.member_logical_keys
            ]
            if not any(path in b.target_paths for b in bindings):
                continue
            for dependency in constraint.dependency_paths:
                old = r._resolve_path(source, dependency)
                new = r._resolve_path(target, dependency)
                found = [m for m in _INTEGER.finditer(text) if int(m[1].replace(",", "")) == old]
                if len(found) != 1:
                    # This member can express another dependency in a grouped
                    # constraint; the existing complete coherence check decides.
                    continue
                span = found[0].span(1)
                replacement = format(new, ",") if "," in found[0][1] else str(new)
                if span in replacements and replacements[span] != replacement:
                    raise ValueError(f"numeric prose contracts disagree: {path}")
                replacements[span] = replacement
        if not replacements:
            continue
        updated = text
        for (start, end), replacement in sorted(replacements.items(), reverse=True):
            updated = updated[:start] + replacement + updated[end:]
        updates[path] = updated
        remaining = "".join(char for i, char in enumerate(text) if i not in covered)
        if counts and not re.search(r"[A-Za-z0-9]", remaining):
            formal.add(path)
    equations = package_equations.generate(source, target)
    updates.update(equations)
    formal.update(equations)
    nested = nested_package_prose.generate(template, source, target)
    updates.update(nested)
    formal.update(nested)
    return updates, frozenset(formal)


def _range_annotation_proven(
    template: CertifiedSemanticTemplate | None,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    path: str,
    category: str,
    number: int,
) -> bool:
    """A parenthetical range count is a subtotal, not a contradictory total."""
    from . import descendant as r
    from .coherence import inclusive_range_surfaces

    if template is None:
        return False
    before, after = r._resolve_path(source, path), r._resolve_path(target, path)
    old_ranges, new_ranges = inclusive_range_surfaces(before), inclusive_range_surfaces(after)
    if len(old_ranges) < 2 or len(new_ranges) != len(old_ranges):
        return False
    for constraint in template.coherence_constraints:
        if not isinstance(constraint, AggregateRangeConstraint):
            continue
        owned = {
            p
            for b in template.bindings
            if b.logical_key in constraint.member_logical_keys
            for p in b.target_paths
        }
        if path not in owned or len(constraint.dependency_paths) != 1:
            continue
        dependency = constraint.dependency_paths[0]
        if not re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", dependency):
            continue
        package = r._resolve_path(target, dependency.removesuffix(".quantity"))
        if package.get("typeCategory") != category:
            continue
        if sum(x.cardinality for x in old_ranges) != r._resolve_path(source, dependency) or sum(
            x.cardinality for x in new_ranges
        ) != r._resolve_path(target, dependency):
            continue
        claims = [
            m
            for m in _package_claim_pattern().finditer(after)
            if (category, number) in _printed_package_claims(m.group())
        ]
        if claims and all(
            any(
                item.cardinality == number
                and item.char_end <= claim.start()
                and re.fullmatch(r"\s*\(\s*", after[item.char_end : claim.start()])
                and re.match(r"\s*\)", after[claim.end() :])
                for item in new_ranges
            )
            for claim in claims
        ):
            return True
    return False


def validate(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    template: CertifiedSemanticTemplate | None = None,
) -> None:
    from . import descendant as r

    if template is not None:
        nested_package_prose.validate(template, source, target)
    equations = package_equations.generate(source, target)
    for path, text in r._flatten_leaves(source).items():
        if not isinstance(text, str) or (path_match := _TEXT.fullmatch(path)) is None:
            continue
        for count in mentions(source, path, text):
            rendered = r._resolve_path(target, path)
            if count.aggregate:
                quantity = sum(r._resolve_path(target, p) for p in count.quantity_paths)
                package = r._resolve_path(target, count.quantity_paths[0].removesuffix(".quantity"))
                aggregate_target = {
                    "documentPatch": {"cargoPackages": [{**package, "quantity": quantity}]}
                }
                if not r._composite_package_quantity_matches(
                    None, aggregate_target, "documentPatch.cargoPackages[0].quantity", rendered
                ):
                    raise ValueError(
                        "generated cargo prose contradicts complete package total "
                        f"{quantity}: {path}"
                    )
                continue
            for quantity_path in count.quantity_paths:
                if not r._composite_package_quantity_matches(None, target, quantity_path, rendered):
                    raise ValueError(f"generated cargo prose contradicts {quantity_path}: {path}")
        rendered = r._resolve_path(target, path)
        if path in equations:
            if rendered != equations[path]:
                raise ValueError(
                    f"generated package equation violates its source-proven arithmetic: {path}"
                )
            continue
        original_claims = _printed_package_claims(text)
        # Existing unlabelled package levels are not newly invented facts. Their
        # numeric contracts own validation; this check catches new packaging
        # assertions introduced by linguistic generation, not missing labels.
        original_categories = {category for category, _ in original_claims}
        new_claims = {
            claim
            for claim in _printed_package_claims(rendered)
            if claim[0] not in original_categories
        }
        if not new_claims:
            continue
        group_index = int(path_match[1])
        patch = target["documentPatch"]
        group_id = patch["cargoGroups"][group_index]["groupId"]
        for category, number in new_claims:
            quantities = [
                package["quantity"]
                for package in patch.get("cargoPackages", ())
                if package["groupId"] == group_id
                and "quantity" in package
                and (
                    package.get("typeCategory") == category
                    or category == "PACKAGE_PACKAGE"
                    or package.get("typeCategory") == "PACKAGE_PACKAGE"
                    # A carton is a box subtype. Its explicit count supports a
                    # broader BOX label, not a different number or a pallet.
                    or (
                        category == "PACKAGE_CARTON"
                        and package.get("typeCategory") == "PACKAGE_BOX"
                    )
                    or r._string_semantics_match(category, package.get("typeDescription", ""))
                )
            ]
            aggregate_matches = any(
                count.aggregate
                and sum(r._resolve_path(target, p) for p in count.quantity_paths) == number
                and r._resolve_path(target, count.quantity_paths[0].removesuffix(".quantity")).get(
                    "typeCategory"
                )
                == category
                for count in mentions(source, path, text)
            )
            if (
                not aggregate_matches
                and number not in quantities
                and (not quantities or number != sum(quantities))
                and not _range_annotation_proven(template, source, target, path, category, number)
            ):
                raise ValueError(
                    "new package claim lacks a matching structured fact: "
                    f"{path}: {number} {category}"
                )


@lru_cache(maxsize=1)
def _package_claim_pattern() -> re.Pattern[str]:
    from . import descendant as r

    words = "|".join(sorted(r._NUMBER_WORDS, key=lambda s: (-len(s), s)))
    nouns = "|".join(
        sorted(
            {s for variants in r._PACKAGE_SURFACES.values() for s in variants},
            key=lambda s: (-len(s), s),
        )
    )
    return re.compile(
        rf"(?<![\w.,])(?P<number>\d+(?:,\d{{3}})*|(?:{words})(?:[ -]+(?:{words}|and))*)"
        rf"\s*(?P<noun>{nouns})(?!\w)",
        re.I,
    )


def _printed_package_claims(text: str) -> set[tuple[str, int]]:
    from . import descendant as r

    noun_categories = {
        surface: category
        for category, surfaces in r._PACKAGE_SURFACES.items()
        for surface in surfaces
    }
    result = set()
    for match in _package_claim_pattern().finditer(text):
        # An explicitly labelled order/part/reference identity is not a quantity,
        # even when the next column's package heading has lost its OCR newline.
        if re.search(
            r"\b(?:ORD(?:ER)?|P[./]?O|PART|C/P|REF(?:ERENCE)?)\s*\.?\s*"
            r"(?:NO\.?|NUMBER|#)\s*:?\s*$",
            text[: match.start()],
            re.I,
        ):
            continue
        number = match["number"]
        value = (
            int(number.replace(",", ""))
            if number[0].isdigit()
            else r._number_word_phrase(number)[0]
        )
        result.add((noun_categories[match["noun"].upper()], value))
    return result
