"""Project owned equipment summaries from their complete container inventory.

Source counts prove whether a receipt covers the whole binding or a semantic
subset. Carrier prose is parsed once as evidence, never reused as target facts.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any, NamedTuple

from document_ocr.synthesis.container_semantics import (
    dimensional_equipment_size,
    partial_equipment_constraint,
    review_source_equipment_surface,
    singleton_equipment_surface,
)

_NOUN = r"(?:CONTAINER|CNTR|CTNR|CONTR|CONT\.?)(?:S|\(S\))?"
_TAIL = rf"(?:\s*(?:FCL|\(\s*FCL\s*\)|{_NOUN}|ONLY\.?|S\.?T\.?C\.?))*\s*"
_FORWARD = re.compile(
    r"(?P<count>0*\d+)(?P<join>\s*(?:[X*'\u2019]|CONTAINER|CONT\.?)\s*|\s+)"
    r"(?P<equipment>.+)",
    re.I,
)
_REVERSE = re.compile(r"(?P<equipment>.+?)(?P<join>\s*[X*]\s*)(?P<count>0*\d+)", re.I)
_COMPACT_REVERSE = re.compile(
    r"(?P<equipment>(?:20|40|45)(?:HC|HQ|GP|HR|RF|RH|RA|RE|ST|DV|DC|OT|FR|TK))"
    r"(?P<count>[1-9][0-9]*)",
    re.I,
)
_LENGTH_ONLY = re.compile(r"(20|40|45)\s*(?:['\u2019`]|FT\.?|FEET|FOOT)?(?:\s+FULL)?", re.I)
_EQUIPMENT_WORDS = frozenset(
    [
        "GP",
        "X",  # Explicit dimension separator, e.g. 40'X9'6".
        "HC",
        "HQ",
        "HR",
        "RF",
        "RFH",
        "RE",
        "RH",
        "RA",
        "RK",
        "RO",
        "RQ",
        "DR",
        "DV",
        "DC",
        "OT",
        "FR",
        "PL",
        "PF",
        "TK",
        "VT",
        "ST",
        "STD",
        "H",
        "D",
        "BO",
        "BOX",
        "BX",
        "FT",
        "FEET",
        "FOOT",
        "HIGH",
        "CUBE",
        "HI",
        "FULL",
        # These are already reviewed carrier tokens in container_semantics.
        "EC",
        "EQ",
        "GE",
        "GO",
        "HCPW",
        "HICU",
        "SH",
        "STANDARD",
        "HEIGHT",
        "DRY",
        "VAN",
        "GENERAL",
        "PURPOSE",
        "REEF",
        "REEFER",
        "REFRIGERATED",
        "HEATED",
        "INSULATED",
        "ISO",
        "SELF",
        "POWERED",
        "REMOVABLE",
        "EQUIPMENT",
        "VENTILATED",
        "BULK",
        "NAMED",
        "CARGO",
        "OPEN",
        "TOP",
        "PLATFORM",
        "FIXED",
        "COLLAPSIBLE",
        "COMPLETE",
        "SUPERSTRUCTURE",
        "PRESSURIZED",
        "TANK",
        "HOPPER",
        "REAR",
        "DISCHARGE",
        "AIR",
        "SURFACE",
    ]
)


class _EquipmentShape(NamedTuple):
    size: str | None
    kind: str | None
    length: str | None = None
    high_cube: bool = False


# Source spellings recur across descendants. Bound retention so an arbitrarily
# large synthesis run cannot grow this pure, immutable parse cache indefinitely.
@lru_cache(maxsize=8192)
def _semantic(text: str) -> _EquipmentShape:
    words = re.sub(_NOUN, " ", text, flags=re.I)
    bare = re.sub(r"^\s*-\s*", "", words).strip()
    length = _LENGTH_ONLY.fullmatch(bare)
    if length:
        # FT/FULL describes length/loading, not dry vs refrigerated equipment.
        return _EquipmentShape(None, None, length[1])
    # HC in a counted equipment receipt states height, not that the unit is dry.
    # A separate REEFER phrase can therefore coexist with ``40'HC``.
    high_cube = re.fullmatch(r"(20|40|45)\s*['\u2019`]?\s*HC", bare, re.I)
    if high_cube:
        high_cube_size = {
            "20": "TWENTY_FOOT_HIGH_CUBE",
            "40": "FORTY_FOOT_HIGH_CUBE",
            "45": "FORTY_FIVE_FOOT_HIGH_CUBE",
        }[high_cube[1]]
        return _EquipmentShape(high_cube_size, None)
    size = dimensional_equipment_size(bare)
    if size is not None:
        return _EquipmentShape(size, None)
    if re.fullmatch(r"(?:HI|HIGH)[ -]?CUBE", bare, re.I):
        return _EquipmentShape(None, None, high_cube=True)
    generic_tank = re.fullmatch(
        r"(20|40|45)\s*(?:['\u2019`]\s*)?(?:FT\s*)?(?:ISO\s*)?TANK",
        bare,
        re.I,
    )
    if generic_tank:
        # Internal fit proxy only; task-facing generic TANK remains wording.
        return _EquipmentShape(None, "PRESSURIZED_TANK", generic_tank[1])
    iso_code = re.fullmatch(r"\s*[0-9A-Z]{2}\s*[A-Z][0-9]\s*", words, re.I)
    if iso_code is None and set(re.findall(r"[A-Z]+", words.upper())) - _EQUIPMENT_WORDS:
        raise ValueError(f"equipment receipt includes unowned non-equipment wording: {text!r}")
    value = review_source_equipment_surface(text, temperature_present=False)
    # A prose receipt can state length and refrigeration without stating height.
    # Unlike the distinct compact HR/RF codes or an explicit HIGH CUBE claim,
    # "40 FT REEF" must not contradict a row that additionally prints 9'6.
    partial_reefer = re.fullmatch(
        r"(20|40|45)\s*(?:['\u2019`]|FT\.?|FOOT|FEET)?\s+"
        r"(?:REEF|REEFER|REFRIGERATED)(?:\s+CONTAINERS?)?",
        text.strip(),
        re.I,
    )
    if partial_reefer:
        return _EquipmentShape(None, "REFRIGERATED", partial_reefer[1])
    partial_rfh = re.fullmatch(r"(20|40|45)\s*['\u2019`]?\s*RFH", bare, re.I)
    if partial_rfh:
        return _EquipmentShape(None, "REFRIGERATED", partial_rfh[1])
    if value.size_category is None and value.type_category is None and length is None:
        raise ValueError(f"equipment receipt has unreviewed type wording: {text!r}")
    return _EquipmentShape(value.size_category, value.type_category)


def _matches(row: Mapping[str, Any], semantic: _EquipmentShape) -> bool:
    size, kind, length, high_cube = semantic
    observed_size, observed_kind = row.get("sizeCategory"), row.get("typeCategory")
    if observed_size is None or observed_kind is None:
        description = row.get("typeDescription")
        observed = review_source_equipment_surface(
            singleton_equipment_surface(description) if isinstance(description, str) else None,
            temperature_present=False,
        )
        observed_size, observed_kind = observed.size_category, observed.type_category
    if high_cube:
        return isinstance(observed_size, str) and observed_size.endswith("HIGH_CUBE")
    if length is not None:
        prefix = {"20": "TWENTY_", "40": "FORTY_FOOT_", "45": "FORTY_FIVE_"}[length]
        printed_length = _LENGTH_ONLY.fullmatch(str(row.get("typeDescription", "")).strip())
        if observed_size is None and printed_length is not None:
            return printed_length[1] == length and kind is None
        return (
            isinstance(observed_size, str)
            and observed_size.startswith(prefix)
            and (kind is None or kind == observed_kind)
        )
    return (size is None or size == observed_size) and (kind is None or kind == observed_kind)


def _row_shape(row: Mapping[str, Any]) -> _EquipmentShape:
    if {"sizeCategory", "typeCategory"} <= row.keys():
        return _EquipmentShape(row["sizeCategory"], row["typeCategory"])
    length, kind, size = partial_equipment_constraint(row)
    return _EquipmentShape(size, kind, length)


def _compatible(row: Mapping[str, Any], receipt: _EquipmentShape) -> bool:
    """Reconcile two explicit observations; unknown is not contradictory.

    This is used only for a homogeneous receipt owning the complete inventory.
    Subset assignment still needs positive row identity, not mere compatibility.
    """
    observed = _row_shape(row)
    for field in ("size", "kind"):
        left, right = getattr(observed, field), getattr(receipt, field)
        if left is not None and right is not None and left != right:
            return False
    prefixes = {"20": "TWENTY_", "40": "FORTY_FOOT_", "45": "FORTY_FIVE_"}
    for left, right in ((observed, receipt), (receipt, observed)):
        if left.length is not None:
            if right.length is not None and left.length != right.length:
                return False
            if right.size is not None and not right.size.startswith(prefixes[left.length]):
                return False
        if left.high_cube and right.size is not None and not right.size.endswith("HIGH_CUBE"):
            return False
    return True


_CONTAINER_OWNER = re.compile(
    r"documentPatch\.containers\[([0-9]+)\]"
    r"(?:\.(?:containerNumber|typeDescription|sizeCategory|typeCategory))?"
)
_ALLOCATION_OWNER = re.compile(r"documentPatch\.cargoAllocationGroups\[([0-9]+)\]\.allocations")


@lru_cache(maxsize=8192)
def _inventory_owners(paths: tuple[str, ...]) -> tuple[bool, tuple[int, ...], tuple[int, ...]]:
    """Compile the declared scope, never silently discard a malformed dependency."""
    whole = False
    containers: set[int] = set()
    allocations: set[int] = set()
    for path in paths:
        if path == "documentPatch.containers":
            whole = True
        elif match := _CONTAINER_OWNER.fullmatch(path):
            containers.add(int(match[1]))
        elif match := _ALLOCATION_OWNER.fullmatch(path):
            allocations.add(int(match[1]))
        else:
            raise ValueError("equipment receipt has an unsupported inventory dependency: " + path)
    if not whole and not containers and not allocations:
        raise ValueError("equipment receipt has no complete in-range container dependency")
    return whole, tuple(sorted(containers)), tuple(sorted(allocations))


def owned_inventory(
    target: Mapping[str, Any], paths: Sequence[str]
) -> tuple[Mapping[str, Any], ...]:
    """Resolve explicit rows or allocation references, never infer from row order.

    A cargo allocation is a valid inventory owner because it explicitly names its
    containers. Count distinct referenced identities even when several selected
    allocation groups share a container. Re-resolve the generated identities in
    the generated target; source row offsets are not an ownership contract.
    """
    rows = target.get("documentPatch", {}).get("containers")
    if not isinstance(rows, (tuple, list)) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("equipment receipt container dependencies are invalid")
    if tuple(paths) == ("documentPatch.containers",):
        selected = tuple(rows)
    else:
        whole, indices, allocation_indices = _inventory_owners(tuple(paths))
        if indices and indices[-1] >= len(rows):
            raise ValueError("equipment receipt has no complete in-range container dependency")
        if allocation_indices:
            groups = target["documentPatch"].get("cargoAllocationGroups")
            if not isinstance(groups, (tuple, list)) or allocation_indices[-1] >= len(groups):
                raise ValueError("equipment receipt has no complete in-range allocation dependency")
            by_number: dict[str, int] = {}
            for index, row in enumerate(rows):
                number = row.get("containerNumber")
                if not isinstance(number, str) or not number.strip() or number in by_number:
                    raise ValueError(
                        "equipment receipt allocation needs unique container identities"
                    )
                by_number[number] = index
            owned = set(indices)
            for index in allocation_indices:
                group = groups[index]
                allocations = group.get("allocations") if isinstance(group, Mapping) else None
                if not isinstance(allocations, (tuple, list)) or not allocations:
                    raise ValueError(
                        "equipment receipt allocation dependency is empty or malformed"
                    )
                for allocation in allocations:
                    number = (
                        allocation.get("containerNumber")
                        if isinstance(allocation, Mapping)
                        else None
                    )
                    if not isinstance(number, str) or number not in by_number:
                        raise ValueError(
                            "equipment receipt allocation has an unresolved container identity"
                        )
                    owned.add(by_number[number])
            indices = tuple(sorted(owned))
        selected = tuple(rows) if whole else tuple(rows[index] for index in indices)
    if not selected:
        raise ValueError("equipment receipt has no owned containers")
    return selected


def validate_source_receipt(
    surface: str, rows: Sequence[Mapping[str, Any]], *, number_words: Callable[[int], str]
) -> None:
    """Prove the printed receipt before compiler certification or mutation.

    Unclassified but exactly observed partial descriptions remain valid. This
    does not add a missing size/type label or assume a complete equipment class.
    """
    match = re.fullmatch(r"\s*(0*\d+)(.*)", surface, re.DOTALL)
    if match and int(match[1]) == len(rows):
        body = match[2]
        if re.fullmatch(r"(?i)\s*(?:X|" + _NOUN + r")?\s*", body):
            return
        descriptions = {row.get("typeDescription") for row in rows}
        description = next(iter(descriptions)) if len(descriptions) == 1 else None
        if isinstance(description, str):
            stripped = re.sub(_NOUN, "", re.sub(r"^\s*[xX]\s*", "", body), flags=re.I)

            def normalize(value: str) -> str:
                return re.sub(r"\W", "", value).casefold()

            if normalize(stripped) == normalize(description):
                # Explicit classes, when present, must still agree with that text.
                if all(not row.get("sizeCategory") and not row.get("typeCategory") for row in rows):
                    return
                semantic = _semantic(description)
                if all(
                    (not row.get("sizeCategory") and not row.get("typeCategory"))
                    or _matches(row, semantic)
                    for row in rows
                ):
                    return
    project_receipt(
        surface, rows, rows, format_equipment=lambda row, source: source, number_words=number_words
    )


def project_receipt(
    surface: str,
    source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    format_equipment: Callable[[Mapping[str, Any], str], str],
    number_words: Callable[[int], str],
) -> str:
    """Return an inventory-derived receipt, rejecting unproved subset ownership."""
    text = surface.strip()
    if re.fullmatch(r"(?:HI|HIGH)[ -]?CUBE", text, re.I):
        # This is a height-only predicate, not permission to print an inventory
        # count, length or equipment class into every repeated table cell.
        shape = _semantic(text)
        if not source_rows or not all(_matches(row, shape) for row in source_rows):
            raise ValueError("height-only receipt contradicts its source equipment")
        if len(source_rows) != len(target_rows):
            raise ValueError("height-only receipt cannot change inventory cardinality")
        sizes = [
            row["sizeCategory"] if "sizeCategory" in row else _row_shape(row).size
            for row in target_rows
        ]
        if any(not isinstance(size, str) for size in sizes):
            raise ValueError("height-only receipt requires explicit sampled equipment heights")
        heights = {str(size).endswith("HIGH_CUBE") for size in sizes}
        if len(heights) != 1:
            raise ValueError("one height-only receipt cannot represent mixed sampled heights")
        return surface if heights.pop() else "STANDARD HEIGHT"
    alias = re.fullmatch(
        r"(?P<words>(?:SAY\s+)?[A-Z -]+)\s*\((?P<body>.+?)\)(?P<tail>" + _TAIL + r")", text, re.I
    )
    if alias:
        text = alias["body"]
    tail = re.search(_TAIL + r"$", text, re.I)
    assert tail is not None
    suffix = tail.group()
    text = text[: tail.start()]
    count_prefix = re.fullmatch(r"(0*\d+)(\s*[xX])", text)
    if count_prefix:
        text, suffix = count_prefix[1], count_prefix[2] + suffix
    source_count_words = number_words(len(source_rows))
    if text.isdigit() or text.casefold() == source_count_words.casefold():
        count = int(text) if text.isdigit() else len(source_rows)
        if count != len(source_rows):
            raise ValueError("count-only equipment receipt disagrees with owned inventory")
        if (
            alias
            and re.sub(r"(?i)^(?:SAY\s+)?(?:PART\s+OF\s+)?", "", alias["words"]).strip().casefold()
            != source_count_words.casefold()
        ):
            raise ValueError("equipment receipt word/digit counts disagree")
        value = (
            str(len(target_rows)).zfill(len(text))
            if text.isdigit()
            else number_words(len(target_rows)).upper()
        )
        if alias:
            prefix = re.sub(re.escape(source_count_words) + r"\s*$", "", alias["words"], flags=re.I)
            value = (
                prefix + number_words(len(target_rows)).upper() + " (" + value + ")" + alias["tail"]
            )
        return value + suffix
    word_prefix = re.match(re.escape(source_count_words) + r"\s+", text, re.I)
    if word_prefix:
        text = text[word_prefix.end() :]
        for length in (20, 40, 45):
            text = re.sub(
                r"\b" + re.escape(number_words(length)) + r"\b", str(length), text, flags=re.I
            )
        text = str(len(source_rows)) + "X" + text
    parsed: list[tuple[int, str, str, str]] = []
    description_only = False
    compact_reverse = 0
    for term in re.split(r"\s*\+\s*", text):
        # A quoted length, e.g. 40' HIGH CUBE, is not a count of forty units.
        starts_with_length = re.match(r"^(?:20|40|45)(?:['\u2019`\s]|[A-Z])", term, re.I)
        bare_description = starts_with_length and not re.search(r"[X*]", term, re.I)
        forward = _FORWARD.fullmatch(term) if not bare_description else None
        reverse = _REVERSE.fullmatch(term) if forward is None else None
        compact = _COMPACT_REVERSE.fullmatch(term) if forward is None and reverse is None else None
        if forward:
            count, equipment, join, digits = (
                int(forward["count"]),
                forward["equipment"],
                forward["join"],
                forward["count"],
            )
        elif reverse:
            count, equipment, join, digits = (
                int(reverse["count"]),
                reverse["equipment"],
                reverse["join"],
                reverse["count"],
            )
        elif compact:
            count, equipment, join, digits = (
                int(compact["count"]),
                compact["equipment"],
                "",
                compact["count"],
            )
            compact_reverse += 1
        else:
            # A description-only receipt denotes the complete owned inventory.
            count, equipment, join, digits = len(source_rows), term, "X", str(len(source_rows))
            description_only = True
        _semantic(equipment)
        parsed.append((count, equipment, join, digits))
    if compact_reverse and compact_reverse != len(parsed):
        raise ValueError("compact reversed receipt cannot mix incompatible count grammars")
    count = sum(term[0] for term in parsed)
    if alias:
        words = re.sub(r"(?i)^(?:SAY\s+)?(?:PART\s+OF\s+)?", "", alias["words"]).strip()
        if words.casefold() != number_words(count).casefold():
            raise ValueError("equipment receipt word/digit counts disagree")
    if (
        count == len(source_rows)
        and len(parsed) == 1
        and any(not _matches(row, _semantic(parsed[0][1])) for row in source_rows)
    ):
        # The receipt itself supplies evidence absent from individual rows. Its
        # complete homogeneous scope makes assignment unambiguous. Retain that
        # evidence and constrain every private candidate before drawing goods;
        # never manufacture corresponding size/type extraction labels.
        semantic = _semantic(parsed[0][1])
        if not all(_compatible(row, semantic) for row in source_rows):
            raise ValueError("equipment receipt contradicts source count/type inventory")
        if len(target_rows) != len(source_rows) or not all(
            _compatible(row, semantic) for row in target_rows
        ):
            raise ValueError("sampled equipment contradicts retained whole-inventory receipt")
        return surface
    if count == len(source_rows) and len(parsed) > 1:
        # A matching total is not proof of matching types. Assign every printed
        # term to distinct source rows, including overlapping partial descriptions.
        # Homogeneous full inventories were already proved row-by-row above;
        # only mixed inventories need the bipartite assignment.
        candidates = [
            tuple(i for i, row in enumerate(source_rows) if _matches(row, _semantic(equipment)))
            for n, equipment, _, _ in parsed
            for _ in range(n)
        ]
        assigned: dict[int, int] = {}

        def assign(term: int, seen: set[int]) -> bool:
            for row in candidates[term]:
                if row in seen:
                    continue
                seen.add(row)
                if row not in assigned or assign(assigned[row], seen):
                    assigned[row] = term
                    return True
            return False

        if not all(
            assign(term, set()) for term in sorted(range(count), key=lambda i: len(candidates[i]))
        ):
            raise ValueError("equipment receipt contradicts source count/type inventory")
    if len(source_rows) != len(target_rows):
        # Cardinality changes cannot preserve indexed subset membership.
        if count != len(source_rows):
            raise ValueError("equipment subset cannot change container cardinality")
        selected = list(target_rows)
    elif count == len(source_rows):
        selected = list(target_rows)
    elif len(parsed) == 1:
        semantic = _semantic(parsed[0][1])
        selected = [
            new
            for old, new in zip(source_rows, target_rows, strict=True)
            if _matches(old, semantic)
        ]
        if len(selected) != count:
            raise ValueError("equipment receipt subset lacks an exact source count/type identity")
    else:
        raise ValueError("partial multi-term equipment receipt requires explicit subset ownership")
    if len(parsed) == 1 and len(source_rows) == len(target_rows):
        printed_shape = _semantic(parsed[0][1])
        if printed_shape.length is not None and printed_shape.kind is None:
            # A length-only receipt never asserts type or height. Changing a
            # private type within that length cannot require printing it.
            shapes = [_row_shape(row) for row in selected]
            length_prefixes = {"20": "TWENTY_", "40": "FORTY_FOOT_", "45": "FORTY_FIVE_"}
            if all(
                shape.length == printed_shape.length
                or (
                    shape.size is not None
                    and shape.size.startswith(length_prefixes[printed_shape.length])
                )
                for shape in shapes
            ):
                return surface
    if (
        len(source_rows) == len(target_rows)
        and selected
        and any(not {"sizeCategory", "typeCategory"} <= row.keys() for row in selected)
    ):
        owned_pairs = (
            tuple(zip(source_rows, target_rows, strict=True))
            if count == len(source_rows)
            else tuple(
                (old, new)
                for old, new in zip(source_rows, target_rows, strict=True)
                if _matches(old, _semantic(parsed[0][1]))
            )
        )
        fields = ("sizeCategory", "typeCategory", "typeDescription")
        if all(
            tuple(old.get(k) for k in fields) == tuple(new.get(k) for k in fields)
            for old, new in owned_pairs
        ):
            # The selected partial observation was proved above and is unchanged.
            # Its new container ID does not authorize inventing unprinted types.
            return surface
    if not selected or any(
        not isinstance(row.get(k), str)
        for row in selected
        for k in ("sizeCategory", "typeCategory")
    ):
        raise ValueError("equipment receipt requires complete target size/type pairs")
    inventory = Counter((row["sizeCategory"], row["typeCategory"]) for row in selected)
    _, exemplar, join, digits = parsed[0]
    if description_only and len(parsed) != 1:
        raise ValueError("mixed description-only and counted equipment terms are ambiguous")
    if description_only and len(inventory) == 1:
        size, kind = next(iter(inventory))
        return format_equipment({"sizeCategory": size, "typeCategory": kind}, exemplar) + suffix
    if compact_reverse:
        return (
            " + ".join(
                format_equipment({"sizeCategory": size, "typeCategory": kind}, exemplar)
                + str(n).zfill(len(digits))
                for (size, kind), n in inventory.items()
            )
            + suffix
        )
    # Keep a single reverse-count receipt in its original orientation. Rewriting
    # 40HQ*4 as 4X40HC changes a valid compiled identifier-shape contract.
    reverse_order = len(parsed) == 1 and reverse is not None
    # An apostrophe used as a count separator is normalized to an explicit X.
    if not re.fullmatch(r"\s*[X*]\s*", join, re.I):
        join = "X"
    terms = [
        (
            format_equipment({"sizeCategory": size, "typeCategory": kind}, exemplar)
            + join
            + str(n).zfill(len(digits))
            if reverse_order
            else str(n).zfill(len(digits))
            + join
            + format_equipment({"sizeCategory": size, "typeCategory": kind}, exemplar)
        )
        for (size, kind), n in inventory.items()
    ]
    result = " + ".join(terms) + suffix
    if description_only:
        result = "MIXED (" + " + ".join(terms) + ")" + suffix
    if alias:
        prefix = re.sub(re.escape(number_words(count)) + r"\s*$", "", alias["words"], flags=re.I)
        result = prefix + number_words(len(selected)).upper() + "(" + result + ")" + alias["tail"]
    return result
