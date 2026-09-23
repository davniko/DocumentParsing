"""Whole-shipment equipment multisets without invented source type-to-ID links.

An aggregate can state two types for several identified containers while never
assigning either type to an individual number. The source claim is a multiset;
only a later synthetic physical choice assigns its members to generated IDs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.generators import DeterministicStream

from . import complete_targets as targets
from .anonymous_equipment import _observation
from .transport_derivations import DERIVATIONS as TRANSPORT_DERIVATIONS


@dataclass(frozen=True)
class MixedInventory:
    source_sha256: str
    binding_keys: tuple[str, ...]
    equipment_counts: tuple[tuple[str, int], ...]
    container_count: int
    source_numbers: tuple[str, ...]
    binding_evidence: tuple[tuple[str, tuple[tuple[str, int, int], ...]], ...]

    def witness(self, domains: Mapping[int, frozenset[str]]) -> dict[int, str] | None:
        """Find an exact capacitated matching under every supplied row constraint."""
        return self._solve(domains)

    def sample(
        self, domains: Mapping[int, frozenset[str]], stream: DeterministicStream
    ) -> dict[int, str]:
        result = self._solve(domains, stream)
        if result is None:
            raise ValueError("aggregate equipment inventory has no joint row-domain assignment")
        return result

    def _solve(
        self,
        domains: Mapping[int, frozenset[str]],
        stream: DeterministicStream | None = None,
    ) -> dict[int, str] | None:
        if set(domains) != set(range(self.container_count)):
            raise ValueError("aggregate equipment domains must cover every container exactly")
        pairs = tuple(pair for pair, _ in self.equipment_counts)
        options = []
        for row in range(self.container_count):
            allowed = tuple(i for i, pair in enumerate(pairs) if pair in domains[row])
            if not allowed:
                return None
            if stream is not None:
                offset = stream.derive(f"aggregate-equipment:{row}").randbelow(len(allowed))
                allowed = allowed[offset:] + allowed[:offset]
            options.append(allowed)
        order = tuple(sorted(range(self.container_count), key=lambda row: len(options[row])))

        @cache
        def solve(index: int, remaining: tuple[int, ...]) -> tuple[tuple[int, int], ...] | None:
            if index == len(order):
                return () if not any(remaining) else None
            # Capacity pruning prevents repeated exploration of impossible
            # suffixes when many IDs share the same small equipment domain.
            if any(
                amount > sum(pair in options[row] for row in order[index:])
                for pair, amount in enumerate(remaining)
            ):
                return None
            row = order[index]
            for pair in options[row]:
                if remaining[pair] == 0:
                    continue
                after = tuple(n - (i == pair) for i, n in enumerate(remaining))
                suffix = solve(index + 1, after)
                if suffix is not None:
                    return ((row, pair), *suffix)
            return None

        match = solve(0, tuple(count for _, count in self.equipment_counts))
        return None if match is None else {row: pairs[pair] for row, pair in match}

    def audit(self, assignment: Mapping[int, str]) -> dict[str, Any]:
        if set(assignment) != set(range(self.container_count)) or Counter(
            assignment.values()
        ) != dict(self.equipment_counts):
            raise ValueError("synthetic equipment assignment contradicts its printed multiset")
        return dict(
            sourceEquipmentBindings=self.binding_keys,
            observedAggregate=dict(self.equipment_counts),
            privateSyntheticAssignment=dict(assignment),
            originalTypeToIdentifierMappingClaimed=False,
            labelVisibilityUnchanged=True,
        )


def compile_inventory(source: targets.SourceTemplate) -> MixedInventory | None:
    """Require source-pinned whole-inventory owners before joining aggregate terms."""
    if not declared(source.template.bindings):
        return None
    return compile_bindings(source.source, source.target, source.template.bindings)


def declared(bindings: Sequence[Any]) -> bool:
    """Recognize a declared aggregate, not ordinary count-only/partial receipts.

    Whole-inventory ownership is the declaration. Individual fields or one
    homogeneous receipt remain on the existing receipt-contract path.
    """
    roots = [
        binding
        for binding in bindings
        if binding.value_kind == "equipment"
        and not binding.target_paths
        and binding.dependency_paths == ("documentPatch.containers",)
        and binding.derivation != "same_as_binding"
    ]
    if len(roots) < 2:
        return False
    # A separately printed bare count/size does not assert two additive terms.
    # Its complete grouped-scope contract is handled by normal receipt logic.
    terms = []
    for binding in roots:
        count, shape = _observation(binding.occurrences[0].source_text)
        if count is None or shape is None:
            return False
        terms.append((count, shape))
    return len(terms) >= 2


def compile_bindings(
    source: bytes, target: Mapping[str, Any], bindings: Sequence[Any]
) -> MixedInventory | None:
    """Shared compiler/renderer proof, independent of the full template wrapper."""
    rows = target["documentPatch"].get("containers", ())
    if not rows:
        return None
    if not declared(bindings):
        return None
    receipts = [
        binding
        for binding in bindings
        if binding.value_kind == "equipment"
        and not binding.target_paths
        and binding.render_mode != "carrier_static"
        and binding.derivation not in TRANSPORT_DERIVATIONS
    ]
    if not receipts:
        return None
    if any({"typeDescription", "sizeCategory", "typeCategory"} & row.keys() for row in rows):
        return None  # Existing visible/partial type contracts use their normal path.
    numbers = [row.get("containerNumber") for row in rows]
    if any(not isinstance(number, str) or not number for number in numbers) or len(
        set(numbers)
    ) != len(rows):
        raise ValueError("aggregate equipment requires distinct observed container identities")
    by_key = {binding.logical_key: binding for binding in receipts}
    counts: dict[str, int] = {}
    parsed: dict[str, tuple[str, int]] = {}
    for binding in receipts:
        observations = set()
        for slot in binding.occurrences:
            if source[slot.byte_start : slot.byte_end].decode() != slot.source_text:
                raise ValueError("aggregate equipment source evidence is stale")
            count, shape = _observation(slot.source_text)
            if (
                count is None
                or count <= 0
                or shape is None
                or shape.size is None
                or shape.kind is None
            ):
                raise ValueError("aggregate equipment requires complete positive count/type terms")
            observations.add((shape.size + "|" + shape.kind, count))
        if len(observations) != 1:
            raise ValueError("repeated aggregate equipment terms disagree")
        parsed[binding.logical_key] = observations.pop()
    for binding in receipts:
        pair, count = parsed[binding.logical_key]
        if binding.derivation == "same_as_binding":
            if (
                len(binding.dependency_bindings) != 1
                or binding.dependency_bindings[0] not in by_key
            ):
                raise ValueError("repeated aggregate equipment lacks its exact owner")
            owner = by_key[binding.dependency_bindings[0]]
            if owner.derivation == "same_as_binding" or parsed[owner.logical_key] != (pair, count):
                raise ValueError(
                    "repeated aggregate equipment ownership is cyclic or contradictory"
                )
            continue
        if binding.dependency_paths != ("documentPatch.containers",) or binding.dependency_bindings:
            raise ValueError("aggregate equipment requires reviewed whole-inventory ownership")
        if pair in counts:
            raise ValueError(
                "duplicate aggregate type needs explicit repeated-versus-additive ownership"
            )
        counts[pair] = count
    if sum(counts.values()) != len(rows):
        raise ValueError("aggregate equipment count differs from identified inventory")
    if len(counts) < 2:
        return None  # Homogeneous whole-inventory receipts have an existing path.
    return MixedInventory(
        sha256_bytes(source),
        tuple(by_key),
        tuple(sorted(counts.items())),
        len(rows),
        tuple(numbers),
        tuple(
            (
                binding.logical_key,
                tuple(
                    (slot.source_text, slot.byte_start, slot.byte_end)
                    for slot in binding.occurrences
                ),
            )
            for binding in receipts
        ),
    )
