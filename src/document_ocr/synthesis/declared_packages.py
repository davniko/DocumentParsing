"""Disjoint public package groups and their private physical composition.

This ledger is separate from inner/support packing. Container placements expose
only scalar counts; the private matrix never authorizes extra extraction fields.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_scenarios import CandidateIncompatibility


def _positive(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be a positive finite Decimal")


@dataclass(frozen=True)
class CountedPackage:
    """Physical facts for one counted unit, including any private inner packing."""

    category: str
    net_kg: Decimal
    gross_kg: Decimal
    volume_m3: Decimal
    multiple: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category.startswith("PACKAGE_"):
            raise ValueError("a counted package requires an extraction category")
        for value, name in (
            (self.net_kg, "net mass"),
            (self.gross_kg, "gross mass"),
            (self.volume_m3, "occupied volume"),
        ):
            _positive(value, name)
        if self.net_kg > self.gross_kg:
            raise ValueError("counted net mass exceeds gross mass")
        if type(self.multiple) is not int or self.multiple < 1:
            raise ValueError("counted quantity multiple must be a positive integer")

    def facts(self, count: int) -> dict[str, str]:
        if type(count) is not int or count < 0 or count % self.multiple:
            raise ValueError("count violates its group's nonnegative quantity lattice")
        return {
            "netKg": str(count * self.net_kg),
            "grossKg": str(count * self.gross_kg),
            "volumeM3": str(count * self.volume_m3),
        }


@dataclass(frozen=True)
class DeclaredPackages:
    """Ordered counted groups with exact private membership and no inferred labels.

    For a source without equipment, the matrix has one private shipment row. It is
    not a fabricated container and must never be projected as an allocation.
    """

    groups: tuple[CountedPackage, ...]
    matrix: tuple[tuple[int, ...], ...]
    capacities: tuple[tuple[Decimal, Decimal], ...]

    def __post_init__(self) -> None:
        if not self.groups or not self.matrix or len(self.matrix) != len(self.capacities):
            raise ValueError("group, load and capacity dimensions must be nonempty and aligned")
        for row, (payload, cube) in zip(self.matrix, self.capacities, strict=True):
            _positive(payload, "payload")
            _positive(cube, "cube capacity")
            if len(row) != len(self.groups) or not any(row):
                raise ValueError("each load must contain at least one declared group")
            for n, group in zip(row, self.groups, strict=True):
                group.facts(n)
            gross = sum(n * g.gross_kg for n, g in zip(row, self.groups, strict=True))
            volume = sum(n * g.volume_m3 for n, g in zip(row, self.groups, strict=True))
            if gross > payload or volume > cube:
                raise CandidateIncompatibility("declared package groups exceed carrying capacity")
        if any(sum(row[i] for row in self.matrix) < 1 for i in range(len(self.groups))):
            raise ValueError("every declared group requires positive membership")

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(sum(row[i] for row in self.matrix) for i in range(len(self.groups)))

    @property
    def total(self) -> int:
        return sum(self.counts)

    def public_rows(self) -> list[dict[str, Any]]:
        return [
            {"packageQuantity": count, "typeCategory": group.category}
            for count, group in zip(self.counts, self.groups, strict=True)
        ]

    def row_facts(self) -> list[dict[str, Any]]:
        return [
            {
                "row": index,
                "packageQuantity": sum(row),
                **{
                    name: str(
                        sum(n * getattr(g, attr) for n, g in zip(row, self.groups, strict=True))
                    )
                    for name, attr in (
                        ("netKg", "net_kg"),
                        ("grossKg", "gross_kg"),
                        ("volumeM3", "volume_m3"),
                    )
                },
            }
            for index, row in enumerate(self.matrix)
        ]

    def totals(self) -> dict[str, str]:
        rows = self.row_facts()
        return {
            name: str(sum(Decimal(row[name]) for row in rows))
            for name in ("netKg", "grossKg", "volumeM3")
        }

    def project_placements(
        self, original: Sequence[dict], identifiers: Sequence[str]
    ) -> list[dict]:
        """Keep the exact source placement mask, resolving by equipment identity."""
        if len(identifiers) != len(self.matrix) or len(set(identifiers)) != len(identifiers):
            raise ValueError("equipment identities must match private loads uniquely")
        lookup = {
            identifier: sum(row) for identifier, row in zip(identifiers, self.matrix, strict=True)
        }
        output = []
        for placement in original:
            identifier = placement["equipmentIdentifier"]
            if identifier not in lookup:
                raise ValueError("source placement points to unknown carrying equipment")
            item = dict(placement)
            if "packageQuantity" in item:
                item["packageQuantity"] = lookup[identifier]
            output.append(item)
        return output

    def receipt(self) -> dict[str, Any]:
        return {
            "groups": [
                {
                    "category": g.category,
                    "netKg": str(g.net_kg),
                    "grossKg": str(g.gross_kg),
                    "volumeM3": str(g.volume_m3),
                    "quantityMultiple": g.multiple,
                }
                for g in self.groups
            ],
            "matrix": [list(row) for row in self.matrix],
            "capacities": [[str(v) for v in row] for row in self.capacities],
            "rows": self.public_rows(),
            "loads": self.row_facts(),
            "totals": self.totals(),
        }

    @classmethod
    def from_receipt(cls, value: dict[str, Any]) -> DeclaredPackages:
        result = cls(
            tuple(
                CountedPackage(
                    g["category"],
                    Decimal(g["netKg"]),
                    Decimal(g["grossKg"]),
                    Decimal(g["volumeM3"]),
                    g["quantityMultiple"],
                )
                for g in value["groups"]
            ),
            tuple(tuple(row) for row in value["matrix"]),
            tuple(tuple(Decimal(v) for v in row) for row in value["capacities"]),
        )
        if result.receipt() != value:
            raise ValueError("declared package receipt differs from its exact physical facts")
        return result


def sample_groups(
    groups: Sequence[CountedPackage],
    capacities: Sequence[tuple[Decimal, Decimal]],
    *,
    seed: int | str,
    key: str,
    utilization: tuple[Decimal, Decimal],
    presence: Sequence[Sequence[bool]] | None = None,
    minimum_group_count: int = 1,
    minimum_load_counts: Sequence[int] | None = None,
    exclusive_groups: Sequence[int] = (),
) -> DeclaredPackages:
    """Sample fresh counts, not source proportions; preserve exact group units.

    Utilization is explicit sampling configuration, not a shipping rule. Membership
    can be supplied for a reviewed structural constraint, otherwise it is sampled.
    """
    if not groups or not capacities or not key:
        raise ValueError("positive group/load dimensions and draw identity required")
    minimum_load_counts = (
        [1] * len(capacities) if minimum_load_counts is None else minimum_load_counts
    )
    if (
        type(minimum_group_count) is not int
        or minimum_group_count < 1
        or len(minimum_load_counts) != len(capacities)
        or any(type(n) is not int or n < 1 for n in minimum_load_counts)
    ):
        raise ValueError("explicit counted-component minima must be positive and load-aligned")
    lo, hi = utilization
    _positive(lo, "minimum utilization")
    _positive(hi, "maximum utilization")
    if not lo <= hi <= 1:
        raise ValueError("utilization interval must be ordered within (0, 1]")
    rng = random.Random(digest([seed, key, "declared-packages-v1"]))
    exclusive = set(exclusive_groups)
    if len(exclusive) != len(exclusive_groups) or any(
        type(i) is not int or not 0 <= i < len(groups) for i in exclusive
    ):
        raise ValueError("exclusive group indexes must be distinct and in range")
    if exclusive and any(groups[i].multiple != 1 for i in exclusive):
        raise CandidateIncompatibility("an installed liquid system is one counted unit per load")
    if presence is None and exclusive:
        # An installed bladder occupies a carrying box, not just its calculated
        # liquid volume. Reserve separate hardware before filling ordinary loads.
        ordinary = [i for i in range(len(groups)) if i not in exclusive]
        if len(capacities) < len(exclusive) + bool(ordinary):
            # Immutable topology cannot be rescued by trying new unit masses.
            raise ValueError("disjoint installed systems require separate loads")
        order = list(range(len(capacities)))
        rng.shuffle(order)
        membership = [[False] * len(groups) for _ in capacities]
        for row, i in zip(order, sorted(exclusive), strict=False):
            membership[row][i] = True
        remaining = order[len(exclusive) :]
        for row in remaining:
            if ordinary:
                chosen = rng.sample(ordinary, rng.randint(1, len(ordinary)))
            else:
                chosen = [rng.choice(sorted(exclusive))]
            for i in chosen:
                membership[row][i] = True
        for i in ordinary:
            if not any(row[i] for row in membership):
                membership[rng.choice(remaining)][i] = True
    elif presence is None:
        membership = []
        for _ in capacities:
            chosen = set(rng.sample(range(len(groups)), rng.randint(1, len(groups))))
            membership.append([i in chosen for i in range(len(groups))])
        for i in range(len(groups)):
            if not any(row[i] for row in membership):
                membership[rng.randrange(len(capacities))][i] = True
    else:
        membership = [list(row) for row in presence]
    if (
        len(membership) != len(capacities)
        or any(
            len(row) != len(groups) or not any(row) or any(type(v) is not bool for v in row)
            for row in membership
        )
        or any(not any(row[i] for row in membership) for i in range(len(groups)))
    ):
        raise ValueError("membership must cover every group and load")
    for row in membership:
        if any(row[i] for i in exclusive) and sum(row) != 1:
            raise CandidateIncompatibility("installed liquid system cannot share a load")
    matrix = []
    for (payload, cube), members, minimum in zip(
        capacities, membership, minimum_load_counts, strict=True
    ):
        _positive(payload, "payload")
        _positive(cube, "cube capacity")
        counts = [
            ((minimum + g.multiple - 1) // g.multiple * g.multiple) if present else 0
            for g, present in zip(groups, members, strict=True)
        ]
        if any(counts[i] > 1 for i in exclusive):
            raise CandidateIncompatibility(
                "source component counts conflict with one installed unit"
            )
        gross = sum(n * g.gross_kg for n, g in zip(counts, groups, strict=True))
        volume = sum(n * g.volume_m3 for n, g in zip(counts, groups, strict=True))
        if gross > payload or volume > cube:
            raise CandidateIncompatibility("minimum selected groups cannot fit the load")
        matrix.append(counts)
    # Printed component rows require positive integer portions. Reserve their
    # minimum before discretionary filling; neither retrying tiny groups nor
    # dropping a component is a substitute for this source presentation fact.
    for i, group in enumerate(groups):
        required = (minimum_group_count + group.multiple - 1) // group.multiple * group.multiple
        remaining = max(0, required - sum(row[i] for row in matrix))
        if remaining and i in exclusive:
            raise CandidateIncompatibility("installed-system component minimum requires more loads")
        if not remaining:
            continue
        eligible = [r for r, row in enumerate(matrix) if row[i]]
        rng.shuffle(eligible)
        for r in eligible:
            payload, cube = capacities[r]
            gross = sum(n * g.gross_kg for n, g in zip(matrix[r], groups, strict=True))
            volume = sum(n * g.volume_m3 for n, g in zip(matrix[r], groups, strict=True))
            available = (
                min(
                    int((payload - gross) // group.gross_kg),
                    int((cube - volume) // group.volume_m3),
                )
                // group.multiple
                * group.multiple
            )
            added = min(remaining, available)
            matrix[r][i] += added
            remaining -= added
        if remaining:
            raise CandidateIncompatibility("printed component minimum cannot fit carrying capacity")
    for (payload, cube), members, counts in zip(capacities, membership, matrix, strict=True):
        gross = sum(n * g.gross_kg for n, g in zip(counts, groups, strict=True))
        volume = sum(n * g.volume_m3 for n, g in zip(counts, groups, strict=True))
        order = [i for i, present in enumerate(members) if present]
        rng.shuffle(order)
        for i in order:
            if i in exclusive:
                continue
            group = groups[i]
            maximum = min(
                int((payload - gross) // group.gross_kg), int((cube - volume) // group.volume_m3)
            )
            fraction = lo + (hi - lo) * Decimal(str(rng.random()))
            extra = int(maximum * fraction) // group.multiple * group.multiple
            counts[i] += extra
            gross += extra * group.gross_kg
            volume += extra * group.volume_m3
    return DeclaredPackages(tuple(groups), tuple(tuple(row) for row in matrix), tuple(capacities))
