"""Exact equipment ownership for immutable container inventories.

Composite receipts constrain their declared row multiset. Scalar cells constrain
one row or an explicitly shared equality group. A receipt alone never establishes
which identifier owns each member of a mixed inventory.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v5 import (
    CONTAINER_SIZE_CATEGORIES,
    CONTAINER_TYPE_CATEGORIES,
)
from document_ocr.synthesis.container_semantics import review_source_equipment_surface
from document_ocr.synthesis.curated_equipment_aliases import _frame
from document_ocr.synthesis.curated_templates import RenderRegion, current_path
from document_ocr.synthesis.template_compiler.descendant import _number_to_words
from document_ocr.synthesis.template_compiler.equipment_receipts import (
    _COMPACT_REVERSE,
    _FORWARD,
    _REVERSE,
    _TAIL,
    project_receipt,
    validate_source_receipt,
)

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_templates import SamplingBlueprint

_ROW = re.compile(r"documentPatch\.containerInformation\[(\d+)\](?:\.(\w+))?")
_ROW_PREFIX = re.compile(r"documentPatch\.containerInformation\[(\d+)\](?:\.|$)")
_PLACEMENT_PREFIX = re.compile(
    r"documentPatch\.goodsItemDetails\[(\d+)\]\.splitGoodsPlacement\[(\d+)\](?:\.|$)"
)
_FIELDS = frozenset(
    {None, "typeDescription", "sizeCategory", "typeCategory", "equipmentIdentifier"}
)
_TYPE_FIELDS = frozenset({"typeDescription", "sizeCategory", "typeCategory"})
_NOUN = r"(?:CONTAINER|CNTR|CTNR|CTR|CONT\.?)(?:S|\(S\))?"
_HEADER_GAP = re.compile(rf"[ \t]*(?:[X*]|(?:{_NOUN}[ \t]+)?OF)[ \t]*", re.I)


@dataclass(frozen=True, slots=True)
class EquipmentOwner:
    """One active exact source span and its immutable container-row scope."""

    key: str
    occurrence: int
    indices: tuple[int, ...]
    kind: Literal["scalar", "receipt", "count", "header"]
    source: str
    type_source: str | None = None

    @property
    def count_only(self) -> bool:
        return self.kind == "count"

    @property
    def receipt(self) -> bool:
        return self.kind in {"receipt", "header"}


@dataclass(frozen=True, slots=True)
class EquipmentIdentityOwner:
    """An independently printed identifier occurrence, not an inferred row order."""

    key: str
    occurrence: int
    index: int


@dataclass(frozen=True, slots=True)
class EquipmentScopeRepair:
    """An exact full-count proof for a previously unscoped receipt dependency."""

    key: str
    indices: tuple[int, ...]
    basis: str


@dataclass(frozen=True, slots=True)
class UniformAggregateSupport:
    """A complete uniform declaration supports rows, not arbitrary type-to-ID links."""

    owners: tuple[tuple[str, int], ...]
    indices: tuple[int, ...]
    unmapped_indices: tuple[int, ...]
    source_pair: tuple[str, str]


@dataclass(frozen=True, slots=True)
class EquipmentInventoryContract:
    """Frozen row cardinality, scalar equalities and source-supported ownership."""

    source_id: str
    source_sha256: str
    source_target_sha256: str
    row_count: int
    groups: tuple[tuple[int, ...], ...]
    spans: tuple[EquipmentOwner, ...]
    identity_spans: tuple[EquipmentIdentityOwner, ...]
    scope_repairs: tuple[EquipmentScopeRepair, ...] = ()
    uniform_aggregate: UniformAggregateSupport | None = None

    def validate_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Reject changed cardinality, missing categories and unequal shared cells."""
        if len(rows) != self.row_count:
            raise ValueError("equipment inventory changed source row cardinality")
        pairs = [_pair(row) for row in rows]
        if self.uniform_aggregate is not None and len(set(pairs)) != 1:
            raise ValueError("uniform-only aggregate ownership cannot label mixed container rows")
        for group in self.groups:
            if len({pairs[index] for index in group}) != 1:
                raise ValueError("equipment inventory contradicts a shared scalar owner")


def _pair(row: Mapping[str, Any]) -> tuple[str, str]:
    size, kind = row.get("sizeCategory"), row.get("typeCategory")
    if size not in CONTAINER_SIZE_CATEGORIES or kind not in CONTAINER_TYPE_CATEGORIES:
        raise ValueError("equipment inventory requires complete canonical size/type rows")
    return size, kind


def _scope(paths: Sequence[str], count: int) -> tuple[int, ...]:
    indices: set[int] = set()
    whole = False
    for original in paths:
        path = current_path(original)
        if path == "documentPatch.containerInformation":
            whole = True
            continue
        match = _ROW.fullmatch(path)
        if match is None or match[2] not in _FIELDS:
            raise ValueError("equipment inventory has an unsupported row dependency")
        index = int(match[1])
        if index >= count:
            raise ValueError("equipment inventory row dependency is out of range")
        indices.add(index)
    return tuple(range(count)) if whole else tuple(sorted(indices))


def _is_typed(paths: Sequence[str]) -> bool:
    return any(
        (match := _ROW.fullmatch(current_path(path))) and match[2] in _TYPE_FIELDS for path in paths
    )


def _count_only(surface: str, count: int) -> bool:
    """Recognize a declared count span, including words and caption abbreviations."""
    words = re.escape(_number_to_words(count)).replace(r"\ ", r"[ -]+")
    match = re.fullmatch(
        rf"\s*(?:SAY\s+)?(?:(?P<digits>\d+)|{words})"
        rf"(?:\s*\((?P<repeat>\d+)\))?\s*(?:{_NOUN}|[X*])?"
        rf"\s*(?:ONLY|TOTAL|IN TOTAL)?\s*\.?\s*",
        surface,
        re.I,
    )
    if match is None:
        return False
    if any(int(value) != count for value in (match["digits"], match["repeat"]) if value):
        raise ValueError("container count contradicts its declared inventory")
    return True


def _printed_count(surface: str) -> int | None:
    """Use the established receipt lexer to distinguish explicit counts from cells."""
    text = surface.strip()
    framed = re.fullmatch(
        r"(?:SAY\s+)?[A-Z -]+\s*\((?P<body>.+?)\)(?P<tail>" + _TAIL + r")",
        text,
        re.I,
    )
    if framed:
        text = framed["body"]
    tail = re.search(_TAIL + r"$", text, re.I)
    assert tail is not None
    text = text[: tail.start()]
    terms = re.split(r"\s*\+\s*", text)
    total = 0
    for term in terms:
        starts_with_length = re.match(r"^(?:20|40|45)(?:['\u2019`\s]|[A-Z])", term, re.I)
        bare = starts_with_length and not re.search(r"[X*]", term, re.I)
        match = _FORWARD.fullmatch(term) if not bare else None
        match = match or _REVERSE.fullmatch(term) or _COMPACT_REVERSE.fullmatch(term)
        if match is None:
            return None
        total += int(match["count"])
    return total


def _validate_scalar(surface: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Use the reviewed scalar vocabulary, including terminal movement framing."""
    frame = _frame(surface)
    observed = review_source_equipment_surface(frame.core, temperature_present=False)
    if (
        not frame.partial
        and observed.size_category is not None
        and observed.type_category is not None
    ):
        expected = observed.size_category, observed.type_category
        if any(_pair(row) != expected for row in rows):
            raise ValueError("printed equipment scalar contradicts its source rows")
        return
    validate_source_receipt(surface, rows, number_words=_number_to_words)


def _source_regions(blueprint: SamplingBlueprint) -> dict[tuple[str, int], RenderRegion]:
    raw = blueprint.source.encode()
    result = {}
    ordered = sorted(blueprint.regions, key=lambda region: region.start)
    if any(left.end > right.start for left, right in pairwise(ordered)):
        raise ValueError("equipment source ownership has overlapping active spans")
    for region in blueprint.regions:
        owner = region.key, region.occurrence
        if owner in result or raw[region.start : region.end].decode() != region.source:
            raise ValueError("equipment ownership does not reproduce exact source bytes")
        result[owner] = region
    return result


def compile_equipment_inventory(
    blueprint: SamplingBlueprint, *, allow_uniform_aggregate: bool = False
) -> EquipmentInventoryContract:
    """Compile direct row owners, never equate a composite receipt's members.

    An empty receipt scope is recoverable only when its explicit complete count
    equals the retained inventory and the established lexer validates that full
    inventory. The resulting repair is surfaced for source-contract maintenance.

    Explicit opt-in accepts a complete uniform whole-inventory declaration when
    some IDs lack direct type cells. It records that missing mapping and constrains
    all sampled rows to one pair; it never licenses arbitrary mixed assignments.
    """
    rows = blueprint.target["documentPatch"].get("containerInformation", [])
    if not rows:
        raise ValueError("equipment inventory has no retained container rows")
    for row in rows:
        _pair(row)
    regions = _source_regions(blueprint)
    parents = list(range(len(rows)))
    spans = []
    repairs = []
    direct: set[int] = set()

    def owner(index: int) -> int:
        while parents[index] != index:
            index = parents[index]
        return index

    for key, binding in blueprint.historical_bindings.items():
        paths = binding.get("target_paths", ())
        derivation = binding.get("derivation")
        if derivation not in {
            "equipment_receipt",
            "container_count",
            "inventory_header",
        } and not _is_typed(paths):
            continue
        scope = _scope(paths, len(rows))
        occurrences = binding.get("occurrences", ())
        if not occurrences:
            raise ValueError("equipment inventory owner has no printed occurrences")
        selected_regions = []
        for occurrence, slot in enumerate(occurrences):
            region = regions.get((key, occurrence))
            if region is None or region.curated_key is not None:
                raise ValueError("equipment inventory binding lacks active exact source ownership")
            if (region.start, region.end, region.source) != (
                slot["byte_start"],
                slot["byte_end"],
                slot["source_text"],
            ):
                raise ValueError("equipment binding occurrence differs from its source region")
            selected_regions.append(region)
        if not scope:
            if not all(
                _count_only(region.source, len(rows))
                or (derivation != "container_count" and _printed_count(region.source) == len(rows))
                for region in selected_regions
            ):
                raise ValueError("unscoped equipment owner lacks an explicit whole-inventory count")
            scope = tuple(range(len(rows)))
            repairs.append(EquipmentScopeRepair(key, scope, "explicit_whole_inventory_count"))
        scalar = derivation not in {"equipment_receipt", "container_count", "inventory_header"}
        if scalar and len(scope) > 1:
            if all(_printed_count(region.source) == len(scope) for region in selected_regions):
                scalar = False
            elif binding.get("target_relationship") == "shared_value_equality":
                for index in scope[1:]:
                    parents[owner(index)] = owner(scope[0])
            else:
                raise ValueError("multi-row equipment scalar requires explicit shared equality")
        for occurrence, region in enumerate(selected_regions):
            selected = scope
            type_source = None
            if derivation == "inventory_header":
                spec = next(
                    (
                        item
                        for item in blueprint.ownership_data.get("inventory_headers", ())
                        if item["key"] == key
                    ),
                    None,
                )
                if (
                    spec is None
                    or tuple(spec["indices"]) != scope
                    or spec["sourceCount"] != len(scope)
                ):
                    raise ValueError("whole-inventory header lacks its exact rebase certificate")
                type_source = spec.get("typeSource")
                type_start, type_end = spec.get("typeStart"), spec.get("typeEnd")
                if (
                    not isinstance(type_source, str)
                    or not type_source.strip()
                    or not isinstance(type_start, int)
                    or not isinstance(type_end, int)
                    or not region.start <= type_start < type_end <= region.end
                    or blueprint.source.encode()[type_start:type_end].decode() != type_source
                    or _printed_count(type_source) is not None
                ):
                    raise ValueError("equipment header lacks its exact type-only source span")
                _validate_scalar(type_source, [rows[index] for index in scope])
                kind = "header"
                if len(scope) == 1:
                    direct.update(scope)
            elif derivation in {"container_count", "equipment_receipt"} and _count_only(
                region.source, len(scope)
            ):
                kind = "count"
            else:
                kind = "scalar" if scalar else "receipt"
                count = _printed_count(region.source)
                if count is not None and count != len(scope):
                    if scalar:
                        raise ValueError("scalar equipment cell has a conflicting count")
                    # The existing receipt projector proves a semantic subset.
                    matching = []
                    for index in scope:
                        try:
                            validate_source_receipt(
                                region.source, [rows[index]] * count, number_words=_number_to_words
                            )
                        except ValueError:
                            continue
                        matching.append(index)
                    if len(matching) != count:
                        raise ValueError("equipment receipt lacks an exact source subset")
                    selected = tuple(matching)
                selected_rows = [rows[index] for index in selected]
                if scalar:
                    _validate_scalar(region.source, selected_rows)
                else:
                    validate_source_receipt(
                        region.source, selected_rows, number_words=_number_to_words
                    )
                if scalar or len(selected) == 1:
                    direct.update(selected)
            spans.append(
                EquipmentOwner(key, occurrence, selected, kind, region.source, type_source)
            )
    uniform_aggregate = None
    all_indices = tuple(range(len(rows)))
    if direct != set(all_indices):
        whole_uniform = (
            tuple(
                (span.key, span.occurrence)
                for span in spans
                if span.kind in {"receipt", "header"}
                and span.indices == all_indices
                and _printed_count(span.source) == len(rows)
            )
            if allow_uniform_aggregate and len({_pair(row) for row in rows}) == 1
            else ()
        )
        if not whole_uniform:
            raise ValueError("each equipment identifier needs direct printed type ownership")
        uniform_aggregate = UniformAggregateSupport(
            whole_uniform, all_indices, tuple(sorted(set(all_indices) - direct)), _pair(rows[0])
        )
        for index in all_indices[1:]:
            parents[owner(index)] = owner(0)
    identities = []
    for (key, occurrence), region in regions.items():
        paths = region.target_paths
        if region.curated_key is not None:
            token = "{" + region.curated_key + "}"
            paths = tuple(
                target.path for target in blueprint.contract.targets if token in target.expression
            )
        indices = {
            int(match[1])
            for path in paths
            if (match := _ROW.fullmatch(current_path(path))) and match[2] == "equipmentIdentifier"
        }
        if len(indices) > 1:
            raise ValueError("one identifier occurrence cannot identify several rows")
        if indices:
            index = indices.pop()
            if index >= len(rows):
                raise ValueError("identifier owner index exceeds retained inventory")
            printed = re.sub(r"[\s/-]", "", region.source).upper()
            if printed != rows[index]["equipmentIdentifier"]:
                raise ValueError("printed identifier differs from its assigned source row")
            identities.append(EquipmentIdentityOwner(key, occurrence, index))
    identifiers = [row["equipmentIdentifier"] for row in rows]
    if len(set(identifiers)) != len(rows) or {item.index for item in identities} != set(
        range(len(rows))
    ):
        raise ValueError("equipment inventory requires unique independently printed identifiers")
    # Source category equality cannot expose a stale row permutation. Compare
    # independently owned co-printed identifiers before admitting mutable types.
    # Cross-line association still requires a reviewed layout contract.
    raw = blueprint.source.encode()
    newlines = [match.start() for match in re.finditer(b"\n", raw)]
    identities_by_line: dict[int, set[int]] = {}
    for identity in identities:
        region = regions[identity.key, identity.occurrence]
        first = bisect_right(newlines, region.start)
        last = bisect_right(newlines, region.end - 1)
        if first == last:
            identities_by_line.setdefault(first, set()).add(identity.index)
    for span in spans:
        if span.kind != "scalar" and not (span.receipt and len(span.indices) == 1):
            continue
        region = regions[span.key, span.occurrence]
        first = bisect_right(newlines, region.start)
        last = bisect_right(newlines, region.end - 1)
        co_printed = identities_by_line.get(first, set()) if first == last else set()
        if len(co_printed) == 1 and not co_printed <= set(span.indices):
            raise ValueError(
                f"equipment type owner {span.key} contradicts its co-printed identifier row"
            )
    identifier_indices = {number: index for index, number in enumerate(identifiers)}
    goods = blueprint.target["documentPatch"].get("goodsItemDetails", [])
    for region in regions.values():
        first = bisect_right(newlines, region.start)
        last = bisect_right(newlines, region.end - 1)
        co_printed = identities_by_line.get(first, set()) if first == last else set()
        if len(co_printed) != 1:
            continue
        paths = region.target_paths
        if region.curated_key is not None:
            token = "{" + region.curated_key + "}"
            paths = tuple(
                target.path for target in blueprint.contract.targets if token in target.expression
            )
        path_rows = set()
        identity_scopes = []
        for original in paths:
            path = current_path(original)
            if match := _ROW_PREFIX.match(path):
                path_rows.add(int(match[1]))
                if path.endswith(".equipmentIdentifier"):
                    identity_scopes.append({int(match[1])})
            elif match := _PLACEMENT_PREFIX.match(path):
                good, placement = int(match[1]), int(match[2])
                reference = goods[good]["splitGoodsPlacement"][placement]["equipmentIdentifier"]
                if reference not in identifier_indices:
                    raise ValueError("placement owner references an unknown equipment identifier")
                path_rows.add(identifier_indices[reference])
                if path.endswith(".equipmentIdentifier"):
                    identity_scopes.append({identifier_indices[reference]})
        scopes = identity_scopes
        if path_rows:
            scopes.append(path_rows)
        measurement = blueprint.ownership_data.get("measure_owners", {}).get(region.key)
        if measurement and "row" in measurement:
            scopes.append({measurement["row"]})
        tare = blueprint.ownership_data.get("tare_owners", {}).get(region.key)
        if tare:
            scopes.append(set(tare))
        if any(not co_printed <= scope for scope in scopes):
            raise ValueError(
                f"row-dependent owner {region.key} contradicts its co-printed identifier row"
            )
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(owner(index), []).append(index)
    contract = EquipmentInventoryContract(
        blueprint.document_id,
        sha256_bytes(blueprint.source.encode()),
        sha256_bytes(canonical_json_bytes(blueprint.target)),
        len(rows),
        tuple(tuple(group) for group in groups.values()),
        tuple(spans),
        tuple(identities),
        tuple(repairs),
        uniform_aggregate,
    )
    contract.validate_rows(rows)
    return contract


def rebase_equipment_inventory_headers(blueprint: SamplingBlueprint) -> SamplingBlueprint:
    """Combine exact adjacent count/type cells into their certified inventory scope.

    Source text and labels remain unchanged. No keyword search assigns arbitrary
    OCR: both cells already have active equipment/count bindings and the gap is
    closed equipment-header syntax. A whole-inventory count overrides a narrower
    repeated type-cell scope only after every source row agrees with that type.
    """
    raw = blueprint.source.encode()
    rows = blueprint.target["documentPatch"].get("containerInformation", [])
    _source_regions(blueprint)
    removals: set[tuple[str, int]] = set()
    added = []
    specs = []
    for region in blueprint.regions:
        binding = blueprint.historical_bindings.get(region.key)
        if region.curated_key is not None or binding is None:
            continue
        if (
            not _is_typed(binding.get("target_paths", ()))
            and binding.get("derivation") != "equipment_receipt"
        ):
            continue
        # A count itself cannot be the right-hand equipment cell.
        right_scope = _scope(binding.get("target_paths", ()), len(rows))
        if binding.get("derivation") == "equipment_receipt" and _count_only(
            region.source, len(right_scope) if right_scope else len(rows)
        ):
            continue
        count_candidates = [
            item
            for item in blueprint.regions
            if item.curated_key is None
            and blueprint.historical_bindings.get(item.key, {}).get("derivation")
            in {"container_count", "equipment_receipt"}
        ]
        previous = [item for item in count_candidates if item.end <= region.start]
        following = [item for item in count_candidates if item.start >= region.end]
        forward = True
        count = max(previous, key=lambda item: item.end) if previous else None
        if count is None or not (
            _HEADER_GAP.fullmatch(raw[count.end : region.start].decode())
            or re.fullmatch(r"[ \t]+", raw[count.end : region.start].decode())
        ):
            count = min(following, key=lambda item: item.start) if following else None
            forward = False
        if count is None:
            continue
        count_binding = blueprint.historical_bindings.get(count.key)
        if (
            count.curated_key is not None
            or count_binding is None
            or count_binding.get("derivation") not in {"container_count", "equipment_receipt"}
        ):
            continue
        scope = _scope(count_binding.get("target_paths", ()), len(rows))
        if not scope:
            if not _count_only(count.source, len(rows)):
                continue
            scope = tuple(range(len(rows)))
        if not _count_only(count.source, len(scope)):
            continue
        gap = (
            raw[count.end : region.start].decode()
            if forward
            else raw[region.end : count.start].decode()
        )
        # An already noun-bearing count may directly precede its type.
        noun_count = re.search(_NOUN + r"\s*$", count.source, re.I) is not None
        trailing_noun = re.match(r"[ \t]+" + _NOUN, raw[region.end :].decode(), re.I)
        gap_valid = _HEADER_GAP.fullmatch(gap) is not None
        if not forward:
            gap_valid = re.fullmatch(r"[ \t]*[X*][ \t]*", gap) is not None
        elif not gap_valid:
            gap_valid = bool((noun_count or trailing_noun) and re.fullmatch(r"[ \t]+", gap))
        if not gap_valid:
            continue
        type_scope = _scope(binding.get("target_paths", ()), len(rows))
        if type_scope and not set(type_scope) <= set(scope):
            raise ValueError("adjacent equipment header crosses its count scope")
        if binding.get("target_relationship") == "shared_value_equality":
            independently_owned = {
                index
                for other_key, other in blueprint.historical_bindings.items()
                if other_key != region.key and _is_typed(other.get("target_paths", ()))
                if len(indices := _scope(other.get("target_paths", ()), len(rows))) == 1
                for index in indices
            }
            if not set(scope) <= independently_owned:
                # A global shared type is a legitimate uniform-only constraint,
                # not permission to manufacture per-ID equipment observations.
                continue
        _validate_scalar(region.source, [rows[index] for index in scope])
        start, end = (count.start, region.end) if forward else (region.start, count.end)
        key = f"inventory_header:{start}"
        if any(item["key"] == key for item in specs):
            raise ValueError("equipment header count is shared by several type cells")
        source = raw[start:end].decode()
        interior = [
            item
            for item in blueprint.regions
            if start <= item.start
            and item.end <= end
            and (item.key, item.occurrence)
            not in {(count.key, count.occurrence), (region.key, region.occurrence)}
        ]
        for item in interior:
            if (
                item.curated_key is not None
                or item.target_paths
                or re.fullmatch(rf"[ \t]*(?:[X*]|OF|{_NOUN})[ \t]*", item.source, re.I) is None
            ):
                raise ValueError("equipment header crosses a non-structural source owner")
        removals.update((item.key, item.occurrence) for item in interior)
        removals.update({(count.key, count.occurrence), (region.key, region.occurrence)})
        added.append(
            RenderRegion(
                key,
                0,
                start,
                end,
                source,
                tuple(
                    f"documentPatch.containerInformation[{index}].{field}"
                    for index in scope
                    for field in ("sizeCategory", "typeCategory")
                ),
                "equipment",
            )
        )
        specs.append(
            {
                "key": key,
                "source": source,
                "start": start,
                "end": end,
                "indices": list(scope),
                "sourceCount": len(scope),
                "countOwner": count.key,
                "typeOwner": region.key,
                "typeSource": region.source,
                "typeStart": region.start,
                "typeEnd": region.end,
            }
        )
    if not specs:
        return blueprint
    bindings = deepcopy(dict(blueprint.historical_bindings))
    remapping = {}
    for key, binding in list(bindings.items()):
        old_occurrence_count = len(binding["occurrences"])
        kept = [
            (i, slot) for i, slot in enumerate(binding["occurrences"]) if (key, i) not in removals
        ]
        remapping[key] = {old: new for new, (old, _) in enumerate(kept)}
        if len(kept) == old_occurrence_count:
            continue
        if not kept:
            del bindings[key]
            continue
        binding["occurrences"] = [slot for _, slot in kept]
        if "slots" in binding.get("realization", {}):
            slots = binding["realization"]["slots"]
            if any(old >= len(slots) for old, _ in kept):
                raise ValueError("changed equipment owner has incomplete realization slots")
            binding["realization"]["slots"] = [slots[old] for old, _ in kept]
    for spec in specs:
        bindings[spec["key"]] = {
            "logical_key": spec["key"],
            "value_kind": "equipment",
            "target_paths": [f"documentPatch.containers[{index}]" for index in spec["indices"]],
            "derivation": "inventory_header",
            "occurrences": [
                {
                    "byte_start": spec["start"],
                    "byte_end": spec["end"],
                    "source_text": spec["source"],
                }
            ],
        }
    regions = tuple(
        sorted(
            [
                replace(
                    region,
                    occurrence=remapping.get(region.key, {}).get(
                        region.occurrence, region.occurrence
                    ),
                )
                for region in blueprint.regions
                if (region.key, region.occurrence) not in removals
            ]
            + added,
            key=lambda item: item.start,
        )
    )
    if any(left.end > right.start for left, right in pairwise(regions)):
        raise ValueError("equipment header rebase overlaps another active source owner")
    ownership = deepcopy(dict(blueprint.ownership_data))
    ownership["inventory_headers"] = list(ownership.get("inventory_headers", ())) + specs
    return replace(
        blueprint, regions=regions, historical_bindings=bindings, ownership_data=ownership
    )


def project_equipment_inventory(
    contract: EquipmentInventoryContract,
    source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    format_equipment: Callable[[Mapping[str, Any], str], str],
    number_words: Callable[[int], str],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Project each source owner from its own row scope, preserving count-only text."""
    contract.validate_rows(source_rows)
    contract.validate_rows(target_rows)
    surfaces: dict[str, list[str]] = {}
    receipts = []
    for span in contract.spans:
        old = [source_rows[index] for index in span.indices]
        new = [target_rows[index] for index in span.indices]
        if span.kind == "count":
            after = span.source
        elif span.kind == "header":
            if span.type_source is None:
                raise ValueError("equipment header has no certified type-only source")
            counts = Counter(_pair(row) for row in new)
            terms = []
            for (size, kind), amount in sorted(counts.items()):
                alias = format_equipment(
                    {"sizeCategory": size, "typeCategory": kind}, span.type_source
                )
                terms.append(f"{amount} x {alias}")
            noun = " CONTAINERS" if re.search(_NOUN, span.source, re.I) else ""
            after = " + ".join(terms) + noun
        elif span.kind == "scalar" and not _frame(span.source).partial:
            frame = _frame(span.source)
            after = frame.prefix + format_equipment(new[0], frame.core) + frame.suffix
        else:
            after = project_receipt(
                span.source,
                old,
                new,
                format_equipment=format_equipment,
                number_words=number_words,
                preserve_length_only=False,
            )
        values = surfaces.setdefault(span.key, [])
        if span.occurrence != len(values):
            raise ValueError("equipment inventory lost occurrence-aligned owner order")
        values.append(after)
        receipts.append(
            {
                "owner": span.key,
                "occurrence": span.occurrence,
                "kind": span.kind,
                "indices": list(span.indices),
                "before": span.source,
                "after": after,
            }
        )
    return surfaces, receipts
