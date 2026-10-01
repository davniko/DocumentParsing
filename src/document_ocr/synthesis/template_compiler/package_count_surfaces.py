"""Own printed package counts before varying structured package quantities.

The compiler may prove a unique source quantity from a printed package caption.
Different packing levels and repeated equal rows need explicit review instead
of an inferred equality. The same screen runs at synthesis preflight so an
unowned source count cannot silently survive in a generated document.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from document_ocr.hashing import sha256_bytes

_AFTER = re.compile(
    rb"^\s*(?:[(/-]\s*)?(?P<noun>PACKAGES?|PACKGS?|PKGS?|CARTONS?|CTNS?|"
    rb"PALLETS?|BAGS?|DRUMS?|PCS|PIECES|SKIDS?|BOXES|CRATES?|BALES?|ROLLS?|"
    rb"BUNDLES?)\b",
    re.I,
)
_BEFORE = re.compile(
    rb"\b(?P<noun>PACKAGES?|PACKGS?|PKGS?|CARTONS?|CTNS?|PALLETS?|BAGS?|"
    rb"DRUMS?|PCS|PIECES|SKIDS?|BOXES|CRATES?|BALES?|ROLLS?|BUNDLES?|QTY)"
    rb"\s*[:#=-]?\s*$",
    re.I,
)
_HEADING = re.compile(
    rb"\b(?:NO\.?\s*(?:OF\s*)?|NUMBER\s+OF\s+|TOTAL\s*(?:NO\.?\s*OF\s*)?)"
    rb"(?:PACKAGES?|PACKGS?|PKGS?|PACKS?|CARTONS?|CTNS?)\b",
    re.I,
)
_GENERIC = frozenset({"PACK", "PACKS", "PACKG", "PACKGS", "PKG", "PKGS", "PACKAGE", "PACKAGES"})
_PACKAGE_QUANTITY_PATH = re.compile(r"documentPatch\.cargoPackages\[(\d+)\]\.quantity")


@dataclass(frozen=True)
class PrintedCount:
    byte_start: int
    byte_end: int
    quantity: int
    noun: str
    kind: str


def occurrences(source: bytes, target: Mapping[str, Any]) -> tuple[PrintedCount, ...]:
    packages = target.get("documentPatch", {}).get("cargoPackages", ())
    quantities = {
        p["quantity"] for p in packages if type(p.get("quantity")) is int and p["quantity"] > 0
    }
    lines = source.splitlines(keepends=True)
    result: list[PrintedCount] = []
    offset = 0
    for index, line in enumerate(lines):
        for quantity in quantities:
            number = re.compile(
                rb"(?<![\d.,])(?:"
                + f"{quantity:,}".encode()
                + rb"|"
                + str(quantity).encode()
                + rb")(?![\d.,])"
            )
            for match in number.finditer(line):
                after = _AFTER.match(line[match.end() : match.end() + 35])
                before = _BEFORE.search(line[max(0, match.start() - 45) : match.start()])
                headed = (
                    not line[: match.start()].strip()
                    and not line[match.end() :].strip()
                    and any(_HEADING.search(row) for row in lines[max(0, index - 2) : index])
                )
                if after is None and before is None and not headed:
                    continue
                noun = (
                    after["noun"].decode().upper()
                    if after is not None
                    else before["noun"].decode().upper()
                    if before is not None
                    else "PACKAGE"
                )
                result.append(
                    PrintedCount(
                        byte_start=offset + match.start(),
                        byte_end=offset + match.end(),
                        quantity=quantity,
                        noun=noun,
                        kind="headed" if headed else "direct",
                    )
                )
        offset += len(line)
    return tuple(sorted(result, key=lambda row: row.byte_start))


def _owner(ranges: Sequence[tuple[int, int]], byte_start: int, byte_end: int) -> bool:
    return any(start < byte_end and end > byte_start for start, end in ranges)


def normalize(
    *, raw: str, drafts: Sequence[Any], source_target: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Bind only unique, same-level or generic package captions automatically."""
    from .host import SpanDraft, merge_drafts
    from .package_equations import _CATEGORIES

    packages = source_target.get("documentPatch", {}).get("cargoPackages", ())
    if len(packages) != 1 or type(packages[0].get("quantity")) is not int:
        return tuple(drafts)
    quantity = packages[0]["quantity"]
    category = packages[0].get("typeCategory")
    source = raw.encode("utf-8")
    ranges = tuple(
        (len(raw[: draft.char_start].encode("utf-8")), len(raw[: draft.char_end].encode("utf-8")))
        for draft in drafts
    )
    additions = []
    for occurrence in occurrences(source, source_target):
        if occurrence.quantity != quantity or _owner(
            ranges, occurrence.byte_start, occurrence.byte_end
        ):
            continue
        noun = occurrence.noun
        if (
            occurrence.kind != "headed"
            and noun not in _GENERIC
            and (_CATEGORIES.get(noun) != category)
        ):
            continue
        start = len(source[: occurrence.byte_start].decode("utf-8"))
        end = len(source[: occurrence.byte_end].decode("utf-8"))
        path = "documentPatch.cargoPackages[0].quantity"
        additions.append(
            SpanDraft(
                draft_id="host_package_count_"
                + sha256_bytes(f"{occurrence.byte_start}:{occurrence.byte_end}".encode())[:16],
                logical_key="host:package_count:" + str(occurrence.byte_start),
                render_mode="deterministic_derived",
                value_kind="integer",
                group_kind="cargo",
                group_key="cargo:package_count",
                target_paths=(),
                derivation="sum_package_quantity",
                dependency_paths=(path,),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="derived_operational_fact",
                render_policy="derived_surface",
                rationale=(
                    "Unique printed package count equals the sole structured package quantity."
                ),
            )
        )
    return merge_drafts(drafts, additions) if additions else tuple(drafts)


def require_owned(source: bytes, template: Any, target: Mapping[str, Any]) -> None:
    """Fail closed if a source quantity survives outside all compiled slots."""
    for occurrence in occurrences(source, target):
        owners = [
            binding.logical_key
            for binding in template.bindings
            if any(
                slot.byte_start < occurrence.byte_end and slot.byte_end > occurrence.byte_start
                for slot in binding.occurrences
            )
        ]
        if not owners:
            raise ValueError(
                "printed package count lacks a compiled owner: "
                f"{occurrence.quantity} {occurrence.noun} at byte {occurrence.byte_start}"
            )


def _word_category_dependencies(
    surface: str,
    paths: Sequence[str],
    source_target: Mapping[str, Any],
    *,
    source: bytes | None = None,
) -> tuple[str, ...]:
    from .package_equations import _CATEGORIES, _NOUN

    noun_pattern = re.compile(r"(?<![A-Za-z])(?:" + _NOUN + r")(?![A-Za-z])", re.I)
    packages = source_target.get("documentPatch", {}).get("cargoPackages", ())
    quantity_indices = {
        int(match[1])
        for path in paths
        if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
    }
    if not quantity_indices or any(character.isdigit() for character in surface):
        return ()
    if noun_pattern.search(surface) is None:
        return ()
    expected_count = sum(packages[index]["quantity"] for index in quantity_indices)
    try:
        printed_count, _, count_end = source_number_word_phrase(
            surface, expected_count, source=source, source_target=source_target
        )
    except ValueError as error:
        if "has no complete number-word phrase" in str(error):
            return ()
        raise
    nouns = {
        _CATEGORIES[match.group().upper()] for match in noun_pattern.finditer(surface[count_end:])
    }
    nouns.discard("PACKAGE_PACKAGE")
    if not nouns:
        return ()
    if printed_count != expected_count:
        raise ValueError("number-word package count differs from its typed source packages")
    categories = {packages[index].get("typeCategory") for index in quantity_indices}
    if len(nouns) != 1 or categories != nouns:
        raise ValueError(
            "number-word package noun lacks one matching typed package owner: " + surface
        )
    return tuple(
        f"documentPatch.cargoPackages[{index}].typeCategory" for index in sorted(quantity_indices)
    )


def _one_lexical_edit(observed: str, expected: str) -> bool:
    """Accept one substitution, insertion, or deletion, but not a different number word."""
    if observed == expected or abs(len(observed) - len(expected)) > 1:
        return False
    if len(observed) == len(expected):
        return sum(a != b for a, b in zip(observed, expected, strict=True)) == 1
    shorter, longer = sorted((observed, expected), key=len)
    return any(longer[:index] + longer[index + 1 :] == shorter for index in range(len(longer)))


def source_number_word_phrase(
    surface: str,
    expected: int,
    *,
    source: bytes | None,
    source_target: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Read a word count; a single typo needs a corroborating printed numeric count.

    This is deliberately narrower than general OCR correction. The source must
    print the same count with the same package category elsewhere, and there
    must be one unambiguous near-canonical word span. A different valid number
    word is never treated as a typo.
    """
    from .descendant import _number_to_words, _number_word_phrase, _number_word_value
    from .package_equations import _CATEGORIES

    try:
        parsed = _number_word_phrase(surface)
    except ValueError:
        parsed = None
    if parsed is not None and parsed[0] == expected:
        return parsed

    packages = source_target.get("documentPatch", {}).get("cargoPackages", ())
    if source is None or len(packages) != 1 or packages[0].get("quantity") != expected:
        raise ValueError("number-word package count lacks a corroborating source count")
    category = packages[0].get("typeCategory")
    corroborating = [
        occurrence
        for occurrence in occurrences(source, source_target)
        if occurrence.quantity == expected and _CATEGORIES.get(occurrence.noun) == category
    ]
    if not corroborating:
        raise ValueError("number-word package count lacks a corroborating source count")

    canonical = _number_to_words(expected).split()
    tokens = tuple(re.finditer(r"[A-Za-z]+", surface))
    candidates: list[tuple[int, int, int]] = []
    for start in range(len(tokens) - len(canonical) + 1):
        window = tokens[start : start + len(canonical)]
        observed = [token.group().casefold() for token in window]
        changed = [
            (actual, wanted)
            for actual, wanted in zip(observed, canonical, strict=True)
            if actual != wanted
        ]
        if len(changed) != 1:
            continue
        actual, wanted = changed[0]
        if _number_word_value(actual) is not None or not _one_lexical_edit(actual, wanted):
            continue
        candidates.append((expected, window[0].start(), window[-1].end()))
    if len(candidates) != 1:
        raise ValueError("number-word package typo has no unique source-proved span")
    return candidates[0]


def normalize_number_word_package_categories(
    *, drafts: Sequence[Any], source_target: Mapping[str, Any], raw: str | None = None
) -> tuple[Any, ...]:
    """Make a count's printed package noun depend on its typed package fact.

    A number-to-words binding may own an entire sentence such as ``SAY THREE
    HUNDRED CARTONS``. Without the type dependency it updates the count but
    silently leaves ``CARTONS`` when sampling drums. Only an unambiguous,
    source-matching typed noun can be promoted; generic ``PACKAGE(S)`` remains
    valid for any package type.
    """
    normalized = []
    for draft in drafts:
        paths = tuple(draft.dependency_paths)
        if (
            draft.derivation != "number_to_words"
            or draft.dependency_bindings
            or any(path.endswith(".typeCategory") for path in paths)
        ):
            normalized.append(draft)
            continue
        category_paths = _word_category_dependencies(
            draft.source_text,
            paths,
            source_target,
            source=None if raw is None else raw.encode("utf-8"),
        )
        if not category_paths:
            normalized.append(draft)
            continue
        normalized.append(replace(draft, dependency_paths=paths + category_paths))
    return tuple(normalized)


def require_number_word_package_categories(
    template: Any, target: Mapping[str, Any], source: bytes | None = None
) -> None:
    """Reject old or incomplete templates before they can vary package categories."""
    for binding in template.bindings:
        if binding.derivation != "number_to_words" or binding.dependency_bindings:
            continue
        for slot in binding.occurrences:
            needed = _word_category_dependencies(
                slot.source_text, binding.dependency_paths, target, source=source
            )
            if not set(needed) <= set(binding.dependency_paths):
                raise ValueError(
                    "number-word package noun lacks its typed category dependency: "
                    + binding.logical_key
                )


def _sum_package_indices(paths: Sequence[str]) -> tuple[int, ...]:
    indices = tuple(
        int(match[1])
        for path in paths
        if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
    )
    return indices if indices and len(indices) == len(paths) else ()


def _sum_source_category(indices: Sequence[int], source_target: Mapping[str, Any]) -> str | None:
    packages = source_target.get("documentPatch", {}).get("cargoPackages", ())
    if any(index >= len(packages) for index in indices):
        raise ValueError("aggregate package dependency exceeds source package rows")
    categories = {packages[index].get("typeCategory") for index in indices}
    return next(iter(categories)) if len(categories) == 1 else None


def _adjacent_sum_noun(
    source: bytes, byte_start: int, byte_end: int
) -> tuple[str, int, int, bool] | None:
    """Find one package noun on the count's own line, with exact byte ownership."""
    from .package_equations import _NOUN

    line_start = source.rfind(b"\n", 0, byte_start) + 1
    line_end = source.find(b"\n", byte_end)
    if line_end < 0:
        line_end = len(source)
    line_bytes = source[line_start:line_end]
    line = line_bytes.decode("utf-8")
    noun_pattern = re.compile(r"(?<![A-Za-z])(?:" + _NOUN + r")(?![A-Za-z])", re.I)
    matches = tuple(noun_pattern.finditer(line))
    if len(matches) != 1:
        return None
    match = matches[0]
    start = line_start + len(line[: match.start()].encode("utf-8"))
    end = line_start + len(line[: match.end()].encode("utf-8"))
    if end <= byte_start:
        gap = source[end:byte_start]
    elif start >= byte_end:
        gap = source[byte_end:start]
    else:
        gap = b""
    if not re.fullmatch(rb"[ \t:#=()\-/]*(?:[xX][ \t]*)?", gap):
        return None
    return match.group(), start, end, byte_start <= start and end <= byte_end


def normalize_sum_package_categories(
    *, raw: str, drafts: Sequence[Any], source_target: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Make a source-proven typed aggregate noun part of the numeric slot.

    Generic PACKAGE(S) and different-level nouns are left alone. An exact
    adjacent source-only noun can be absorbed only when every occurrence of
    its old logical binding has one source-proven sum owner and nothing else
    depends on that binding. Other competing owners fail closed.
    """
    from .host import merge_drafts
    from .package_equations import _CATEGORIES

    source = raw.encode("utf-8")
    normalized = []
    category_paths_by_key: dict[str, tuple[str, ...]] = {}
    transferred: dict[int, str] = {}
    transferred_keys: set[str] = set()
    for draft in drafts:
        indices = _sum_package_indices(draft.dependency_paths)
        if (
            draft.derivation not in {"sum_package_quantity", "number_to_words"}
            or draft.dependency_bindings
            or not indices
        ):
            normalized.append(draft)
            continue
        source_category = _sum_source_category(indices, source_target)
        if source_category is None:
            normalized.append(draft)
            continue
        start = len(raw[: draft.char_start].encode("utf-8"))
        end = len(raw[: draft.char_end].encode("utf-8"))
        noun = _adjacent_sum_noun(source, start, end)
        if noun is None or _CATEGORIES[noun[0].upper()] != source_category:
            normalized.append(draft)
            continue
        _, noun_start, noun_end, _inside = noun
        byte_start, byte_end = min(start, noun_start), max(end, noun_end)
        char_start = len(source[:byte_start].decode("utf-8"))
        char_end = len(source[:byte_end].decode("utf-8"))
        noun_char_start = len(source[:noun_start].decode("utf-8"))
        noun_char_end = len(source[:noun_end].decode("utf-8"))
        overlapping = tuple(
            other
            for other in drafts
            if other is not draft
            and other.char_start < char_end
            and other.char_end > char_start
        )
        if overlapping:
            owned_category_paths = {
                f"documentPatch.cargoPackages[{index}].typeCategory" for index in indices
            }
            if len(overlapping) != 1:
                raise ValueError("aggregate package noun overlaps a non-category owner")
            owner = overlapping[0]
            if owner.char_start == noun_char_start and owner.char_end == noun_char_end:
                if owned_category_paths.intersection(owner.target_paths):
                    # An existing target category slot already owns the noun.
                    normalized.append(draft)
                    continue
                if (
                    owner.render_mode in {"deterministic_auxiliary", "literal_static"}
                    and not owner.target_paths
                    and not owner.dependency_paths
                    and not owner.dependency_bindings
                    and owner.source_text == noun[0]
                ):
                    if draft.derivation == "number_to_words":
                        from .descendant import _number_word_phrase

                        packages = source_target["documentPatch"]["cargoPackages"]
                        expected = sum(packages[index]["quantity"] for index in indices)
                        parsed, _, _ = _number_word_phrase(draft.source_text)
                        if parsed != expected:
                            raise ValueError(
                                "number-word aggregate differs from printed package rows"
                            )
                    if id(owner) in transferred:
                        raise ValueError("aggregate package noun has two count owners")
                    transferred[id(owner)] = draft.logical_key
                    transferred_keys.add(owner.logical_key)
                else:
                    raise ValueError("aggregate package noun overlaps a non-category owner")
            else:
                raise ValueError("aggregate package noun overlaps a non-category owner")
        category_paths = tuple(
            f"documentPatch.cargoPackages[{index}].typeCategory"
            for index in sorted(set(indices))
        )
        prior = category_paths_by_key.setdefault(draft.logical_key, category_paths)
        if prior != category_paths:
            raise ValueError("aggregate package binding has inconsistent typed owners")
        normalized.append(
            replace(
                draft,
                char_start=char_start,
                char_end=char_end,
                source_text=raw[char_start:char_end],
                dependency_paths=tuple(draft.dependency_paths) + category_paths,
            )
        )
    if transferred:
        for key in transferred_keys:
            members = [draft for draft in drafts if draft.logical_key == key]
            if not members or any(id(member) not in transferred for member in members):
                raise ValueError("aggregate package noun binding is only partly transferred")
            if any(key in draft.dependency_bindings for draft in drafts):
                raise ValueError("aggregate package noun binding has dependent consumers")
        normalized = [draft for draft in normalized if id(draft) not in transferred]
    if category_paths_by_key:
        normalized = [
            replace(
                draft,
                dependency_paths=draft.dependency_paths
                + tuple(
                    path
                    for path in category_paths_by_key.get(draft.logical_key, ())
                    if path not in draft.dependency_paths
                ),
            )
            for draft in normalized
        ]
    return merge_drafts(normalized)


def sum_package_categories_changed(
    *, paths: Sequence[str], source_target: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    indices = tuple(
        int(match[1])
        for path in paths
        if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
    )
    if not indices:
        return False
    source_packages = source_target.get("documentPatch", {}).get("cargoPackages", ())
    target_packages = target.get("documentPatch", {}).get("cargoPackages", ())
    if any(index >= len(source_packages) or index >= len(target_packages) for index in indices):
        raise ValueError("aggregate package topology changed")
    return any(
        source_packages[index].get("typeCategory") != target_packages[index].get("typeCategory")
        for index in indices
    )


def render_sum_package_nouns(
    *,
    binding: Any,
    template: Any | None,
    source: bytes,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    replacements: Mapping[str, str],
    target_quantity: int,
) -> dict[str, str]:
    """Update same-level typed nouns, or reject an unowned changed noun."""
    from .package_equations import _CATEGORIES, _NOUN

    indices = tuple(
        int(match[1])
        for path in binding.dependency_paths
        if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
    )
    if not indices or binding.dependency_bindings:
        return dict(replacements)
    old_category = _sum_source_category(indices, source_target)
    if old_category is None:
        return dict(replacements)
    target_packages = target.get("documentPatch", {}).get("cargoPackages", ())
    if any(index >= len(target_packages) for index in indices):
        raise ValueError("aggregate package topology changed")
    target_categories = {target_packages[index].get("typeCategory") for index in indices}
    if target_categories == {old_category}:
        return dict(replacements)
    new_category = (
        next(iter(target_categories)) if len(target_categories) == 1 else "PACKAGE_PACKAGE"
    )
    if not isinstance(new_category, str) or not new_category.startswith("PACKAGE_"):
        raise ValueError("aggregate package target category is missing")
    noun_pattern = re.compile(r"(?<![A-Za-z])(?:" + _NOUN + r")(?![A-Za-z])", re.I)
    updated = dict(replacements)
    for slot in binding.occurrences:
        noun = _adjacent_sum_noun(source, slot.byte_start, slot.byte_end)
        if noun is None or _CATEGORIES[noun[0].upper()] != old_category:
            continue  # No source proof that this printed noun is the structured package level.
        old_noun, noun_start, noun_end, inside = noun
        if not inside:
            category_paths = {
                f"documentPatch.cargoPackages[{index}].typeCategory" for index in indices
            }
            separately_owned = (
                template is not None
                and len(target_categories) == 1
                and any(
                    category_paths.intersection(other.target_paths)
                    and any(
                        other_slot.byte_start <= noun_start
                        and other_slot.byte_end >= noun_end
                        for other_slot in other.occurrences
                    )
                    for other in template.bindings
                    if other.logical_key != binding.logical_key
                )
            )
            if separately_owned:
                continue
            raise ValueError("aggregate package noun is outside its derived slot")
        source_matches = tuple(noun_pattern.finditer(slot.source_text))
        rendered_matches = tuple(noun_pattern.finditer(updated[slot.slot_id]))
        if (
            len(source_matches) != 1
            or len(rendered_matches) != 1
            or source_matches[0].group().upper() != old_noun.upper()
        ):
            raise ValueError("aggregate package noun has no unique rendered owner")
        observed = rendered_matches[0]
        category_noun = new_category.removeprefix("PACKAGE_").replace("_", " ")
        if new_category == "PACKAGE_DRUM_FIBRE":
            category_noun = "FIBRE DRUM"
        elif new_category == "PACKAGE_INTERMEDIATE_BULK_CONTAINER":
            category_noun = "IBC"
        if old_noun.upper().endswith(("(S)", "(ES)")):
            words = category_noun.split()
            last = words[-1]
            ending = (
                "IES"
                if last.endswith("Y") and len(last) > 1 and last[-2] not in "AEIOU"
                else "ES"
                if last.endswith(("S", "X", "Z", "CH", "SH"))
                else "S"
            )
            words[-1] = last[:-1] + f"({ending})" if ending == "IES" else last + f"({ending})"
            category_noun = " ".join(words)
        elif target_quantity != 1:
            words = category_noun.split()
            last = words[-1]
            ending = (
                "IES"
                if last.endswith("Y") and len(last) > 1
                else "ES"
                if last.endswith(("S", "X", "Z", "CH", "SH"))
                else "S"
            )
            words[-1] = last[:-1] + ending if ending == "IES" else last + ending
            category_noun = " ".join(words)
        replacement = (
            category_noun.lower()
            if old_noun.islower()
            else category_noun.title()
            if old_noun.istitle()
            else category_noun
        )
        rendered = updated[slot.slot_id]
        updated[slot.slot_id] = (
            rendered[: observed.start()] + replacement + rendered[observed.end() :]
        )
    return updated
