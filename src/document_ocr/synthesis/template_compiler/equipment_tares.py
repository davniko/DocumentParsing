"""Printed tare constraints for latent equipment, never extraction-label enrichment.

Use exact row ownership from the operational-measurement parser or reviewed
physical-row contracts. Only unambiguous kilogram observations with a printed
semantic pair enter train-only support. Explicit sampled contracts draw new
tares from that support; retained source observations are not nearest-weight
classifications. Printed partial types may constrain private dimensions without
adding unseen facts to extraction labels or fabricating a fitted observation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import cache, lru_cache
from typing import Any

from document_ocr.synthesis import raw_text_rewrite_cycle_probe as operational
from document_ocr.synthesis.container_semantics import partial_equipment_constraint
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.rendering import _numeric_interpretations

from . import equipment_row_constraints, numeric_auxiliary
from .complete_targets import SourceTemplate


@dataclass(frozen=True)
class TareSupport:
    pairs_by_kg: Mapping[Decimal, frozenset[str]]
    source_ids_by_kg_pair: Mapping[tuple[Decimal, str], tuple[str, ...]]


@lru_cache(maxsize=4096)
def _sum_witness(
    choices: tuple[tuple[Decimal, ...], ...], total: Decimal | None
) -> tuple[Decimal, ...] | None:
    """Solve an exact finite sum; never approximate a printed aggregate."""
    if any(not domain for domain in choices):
        return None
    if total is None:
        return tuple(domain[0] for domain in choices)
    minimum = [Decimal(0)] * (len(choices) + 1)
    maximum = list(minimum)
    for index in range(len(choices) - 1, -1, -1):
        minimum[index] = minimum[index + 1] + min(choices[index])
        maximum[index] = maximum[index + 1] + max(choices[index])

    @cache
    def solve(index: int, remaining: Decimal) -> tuple[Decimal, ...] | None:
        if not minimum[index] <= remaining <= maximum[index]:
            return None
        if index == len(choices):
            return ()
        for value in choices[index]:
            suffix = solve(index + 1, remaining - value)
            if suffix is not None:
                return (value, *suffix)
        return None

    return solve(0, total)


@dataclass(frozen=True)
class TareConstraints:
    # Each domain contains only source-observed or train-only exact tare/pair
    # observations. A latent witness is not a claim about the original shipment.
    domains: tuple[tuple[int, tuple[tuple[str, tuple[Decimal, ...]], ...]], ...]
    total_kg: Decimal | None
    sampled_binding_owners: tuple[tuple[str, tuple[int, ...]], ...] = ()
    sampled_binding_steps: tuple[tuple[str, Decimal], ...] = ()
    private_unit_bindings: tuple[str, ...] = ()
    source_observed_partial_owners: tuple[int, ...] = ()

    @property
    def options(self) -> dict[int, frozenset[str]]:
        return {owner: frozenset(pair for pair, _ in values) for owner, values in self.domains}

    def witness(self, pairs: Mapping[int, frozenset[str]]) -> dict[int, Decimal] | None:
        values = tuple(
            tuple(sorted({v for pair, domain in choices if pair in pairs[owner] for v in domain}))
            for owner, choices in self.domains
        )
        result = _sum_witness(values, self.total_kg)
        return (
            None
            if result is None
            else {owner: value for (owner, _), value in zip(self.domains, result, strict=True)}
        )

    def sample(
        self, pairs: Mapping[int, frozenset[str]], stream: DeterministicStream
    ) -> tuple[dict[int, Decimal], dict[str, Decimal]]:
        """Draw exact train-supported tares, preserving every fixed sum constraint."""
        choices = []
        for owner, domain in self.domains:
            possible = tuple(
                sorted({v for pair, values in domain if pair in pairs[owner] for v in values})
            )
            if not possible:
                raise ValueError("sampled equipment has no exact supported tare")
            offset = stream.derive(f"equipment-tare:{owner}").randbelow(len(possible))
            choices.append(possible[offset:] + possible[:offset])
        witness = _sum_witness(tuple(choices), self.total_kg)
        if witness is None:
            raise ValueError("sampled equipment cannot satisfy the fixed tare aggregate")
        private = {owner: value for (owner, _), value in zip(self.domains, witness, strict=True)}
        values = {
            key: sum((private[owner] for owner in owners), Decimal(0))
            for key, owners in self.sampled_binding_owners
        }
        for key, step in self.sampled_binding_steps:
            if values[key].quantize(step) != values[key]:
                raise ValueError("sampled tare cannot preserve printed numeric precision")
        return private, values


def _occurrences(
    text: str, containers: Sequence[Mapping[str, Any]], descriptions: Sequence[str] = ()
) -> dict[tuple[int, str], frozenset[int]]:
    if not containers or not re.search(r"\bTARE\b", text, re.I):
        return {}
    # Blank OCR lines have no column meaning. Retain their original line numbers
    # for joining the parser evidence to certified byte spans below.
    cargo_lines = {
        line.strip()
        for d in descriptions
        for line in d.splitlines()
        if re.search(r"[A-Za-z]", line)
    }
    rows = [
        (i + 1, line)
        for i, line in enumerate(text.splitlines())
        if line.strip() and line.strip() not in cargo_lines
    ]
    compact = "\n".join(line for _, line in rows)
    # The aggregate caption cannot override an explicitly different tare unit.
    # The reused operational parser identifies column positions, not units.
    # Mixed-unit/repeated conflicting headers need a separately reviewed table
    # contract; do not put their numeric coincidences into kilogram fit support.
    tare_units = re.findall(
        r"\bTARE(?:\s+WEIGHT)?\s*[:(]?\s*"
        r"(KGS?|KILOGRAMS?|LBS?|POUNDS?|TONNES?|TONS?|MTS?)\b",
        compact,
        re.I,
    )
    kg_columns = bool(re.search(r"\bWeight\s+in\s+Kgs\s+Total\b", compact, re.I)) and all(
        unit.upper().startswith("K") for unit in tare_units
    )
    result: dict[tuple[int, str], set[int]] = defaultdict(set)
    per_container = operational._container_measurement_occurrences(compact, containers)
    for owner, values in per_container.items():
        for kind, line, surface, evidence, grammar in values:
            if kind != "tare_weight_kg":
                continue
            explicit_kg = bool(
                re.search(
                    r"\bTARE(?:\s+WEIGHT)?\s*[:=]?\s*"
                    + re.escape(surface)
                    + r"\s*(?:KGS?|KILOGRAMS?)\b",
                    evidence,
                    re.I,
                )
            )
            if not (explicit_kg or (grammar == "dense_container_table" and kg_columns)):
                continue
            captioned_total = bool(
                re.match(
                    r"\s*(?:GRAND\s+)?TOTAL\s+TARE(?:\s+WEIGHT)?\s*[:=]?\s*"
                    + re.escape(surface)
                    + r"\s*(?:KGS?|KILOGRAMS?)\b",
                    rows[line - 1][1],
                    re.I,
                )
            )
            result[(rows[line - 1][0], surface)].update(
                range(len(containers)) if captioned_total else (owner,)
            )
    if kg_columns:
        for aggregate in operational._dense_aggregate_measurement_rows(compact):
            if aggregate.container_count != len(containers):
                raise ValueError("printed tare total disagrees with source container inventory")
            line = rows[aggregate.value_line_indexes[1]][0]
            result[(line, aggregate.value_surfaces[1])].update(range(len(containers)))
    return {key: frozenset(value) for key, value in result.items()}


def build_support(rows: Sequence[Mapping[str, Any]]) -> TareSupport:
    evidence: dict[tuple[Decimal, str], set[str]] = defaultdict(set)
    for row in rows:
        containers = row["target"]["documentPatch"].get("containers", [])
        if not any({"sizeCategory", "typeCategory"} <= c.keys() for c in containers):
            continue
        occurrences = _occurrences(
            row["joinedRawText"],
            containers,
            [
                g["description"]
                for g in row["target"]["documentPatch"].get("cargoGroups", [])
                if "description" in g
            ],
        )
        per_owner: dict[int, set[Decimal]] = defaultdict(set)
        for (_, surface), owners in occurrences.items():
            if len(owners) != 1:
                continue  # An aggregate is not an individual container tare.
            values = {v[0] for v in _numeric_interpretations(surface) if v[0] > 0}
            if len(values) == 1:
                per_owner[next(iter(owners))].update(values)
        for owner, values in per_owner.items():
            container = containers[owner]
            if len(values) != 1 or not {"sizeCategory", "typeCategory"} <= container.keys():
                continue  # Ambiguous/contradictory observations cannot establish fit support.
            pair = container["sizeCategory"] + "|" + container["typeCategory"]
            evidence[(next(iter(values)), pair)].add(row["documentId"])
    pairs: dict[Decimal, set[str]] = defaultdict(set)
    for value, pair in evidence:
        pairs[value].add(pair)
    return TareSupport(
        {value: frozenset(domain) for value, domain in pairs.items()},
        {key: tuple(sorted(ids)) for key, ids in evidence.items()},
    )


def compile_constraints(
    source: SourceTemplate,
    contracts: Mapping[str, numeric_auxiliary.NumericContract],
    support: TareSupport,
    latent: frozenset[int],
    *,
    configured_pairs: frozenset[str] = frozenset(),
) -> TareConstraints:
    containers = source.target["documentPatch"].get("containers", [])
    occurrences = _occurrences(
        source.source.decode(),
        containers,
        [
            g["description"]
            for g in source.target["documentPatch"].get("cargoGroups", [])
            if "description" in g
        ],
    )
    bindings = {binding.logical_key: binding for binding in source.template.bindings}
    explicitly_owned = {
        key: contract
        for key, contract in contracts.items()
        if contract.role == "tare" and bool(getattr(bindings[key], "dependency_paths", ()))
    }
    reviewed_scopes: dict[str, set[frozenset[int]]] = defaultdict(set)
    if explicitly_owned:
        for row in equipment_row_constraints.compile_rows(
            source, explicitly_owned, include_tare=True
        ):
            if row.dimension != "mass" or row.unit_factor != 1:
                raise ValueError("reviewed tare ownership requires a kilogram unit contract")
            if row.unit_evidence == "synthetic_private" and (
                explicitly_owned[row.binding_key].synthetic_unit != "kilogram"
            ):
                raise ValueError("private tare unit lacks an explicit audited kilogram choice")
            reviewed_scopes[row.binding_key].add(frozenset(row.container_indices))
        if reviewed_scopes.keys() != explicitly_owned.keys():
            raise ValueError("reviewed tare ownership lacks complete unit/row evidence")
    values: dict[int, Decimal] = {}
    totals: list[tuple[frozenset[int], Decimal]] = []
    observed_values: dict[int, Decimal] = {}
    observed_totals: list[tuple[frozenset[int], Decimal]] = []
    sampled_bindings = []
    sampled_steps = []
    for key, contract in contracts.items():
        if contract.role != "tare":
            continue
        if contract.mode not in {"source_fixed", "sampled_equipment_tare"}:
            raise ValueError("latent tare support requires a fixed, reviewed numeric value")
        value = Decimal(contract.source_value)
        if value <= 0:
            raise ValueError("equipment tare must be positive")
        matched: set[frozenset[int]] = set()
        unmatched = 0
        for slot in bindings[key].occurrences:
            start = source.source[: slot.byte_start].count(b"\n") + 1
            end = source.source[: slot.byte_end].count(b"\n") + 1
            matching_scopes = {
                scope
                for (line, surface), scope in occurrences.items()
                if start <= line <= end
                and surface.strip() == slot.source_text.strip()
                and value in {v[0] for v in _numeric_interpretations(surface)}
            }
            if len(matching_scopes) > 1:
                raise ValueError("printed tare occurrence has ambiguous physical scope")
            if not matching_scopes:
                unmatched += 1
            matched.update(matching_scopes)
        if key in reviewed_scopes:
            if matched and not matched <= reviewed_scopes[key]:
                raise ValueError("reviewed tare owner contradicts the printed container row")
            matched = reviewed_scopes[key]
            unmatched = 0  # The explicit contract proved every slot independently above.
        # A certified repeated scalar in a one-container document has only one
        # possible equipment owner; require at least one explicit kg row proof.
        if not matched or (unmatched and len(containers) != 1):
            raise ValueError("printed tare lacks complete kilogram/container-row ownership: " + key)
        if contract.mode == "sampled_equipment_tare":
            if len(matched) != 1:
                raise ValueError(
                    "sampled tare needs one explicit row or aggregate scope per binding"
                )
            sampled_bindings.append((key, tuple(sorted(next(iter(matched))))))
            sampled_steps.append((key, numeric_auxiliary.quantum(bindings[key], value)))
        for owners in matched:
            if len(owners) == 1:
                owner = next(iter(owners))
                if owner in observed_values and observed_values[owner] != value:
                    raise ValueError("conflicting printed tares for one container")
                observed_values[owner] = value
                if contract.mode == "source_fixed":
                    values[owner] = value
            else:
                observed_totals.append((owners, value))
                if contract.mode == "source_fixed":
                    totals.append((owners, value))
    if any(owners != frozenset(range(len(containers))) for owners, _ in observed_totals):
        raise ValueError("partial aggregate tare needs an explicit subset constraint")
    if len({total for _, total in observed_totals}) > 1:
        raise ValueError("conflicting printed aggregate tares")
    if len(observed_values) == len(containers) and any(
        sum(observed_values.values(), Decimal(0)) != total for _, total in observed_totals
    ):
        raise ValueError("printed individual tares disagree with their aggregate")
    total_values = {total for _, total in totals}
    if len(total_values) > 1:
        raise ValueError("conflicting printed aggregate tares")
    total_kg = next(iter(total_values)) if total_values else None
    if (
        total_kg is not None
        and len(values) == len(containers)
        and sum(values.values(), Decimal(0)) != total_kg
    ):
        raise ValueError("printed individual tares disagree with their aggregate")
    owned = set(range(len(containers))) if total_kg is not None else set(values)
    fixed_owned = set(owned)
    owned.update(owner for _, owners in sampled_bindings for owner in owners)
    domains = []
    partial_source_owners = []
    for owner in sorted(owned):
        domain: dict[str, list[Decimal]] = defaultdict(list)
        observed = values.get(owner)
        known_pair = (
            None
            if owner in latent or owner not in fixed_owned
            else containers[owner]["sizeCategory"] + "|" + containers[owner]["typeCategory"]
        )
        source_pairs = (
            _source_partial_pairs(source, owner, configured_pairs)
            if observed is not None and known_pair is None and configured_pairs
            else frozenset()
        )
        if source_pairs and observed is not None:
            # This proves the observed tare with its printed length/type, not
            # an empirical tare/height relationship. Missing height remains a
            # private scenario choice and cannot enrich extraction labels.
            for pair in sorted(source_pairs):
                domain[pair].append(observed)
            partial_source_owners.append(owner)
        elif observed is not None and known_pair is not None:
            domain[known_pair].append(observed)
        else:
            for value, pairs in support.pairs_by_kg.items():
                if observed is not None and value != observed:
                    continue
                for pair in pairs:
                    if known_pair is None or pair == known_pair:
                        domain[pair].append(value)
        if not domain:
            raise ValueError("printed latent tare has no exact train-only equipment support")
        domains.append((owner, tuple((p, tuple(sorted(v))) for p, v in sorted(domain.items()))))
    result = TareConstraints(
        tuple(domains),
        total_kg,
        tuple(sampled_bindings),
        tuple(sampled_steps),
        tuple(sorted(k for k, c in contracts.items() if c.role == "tare" and c.synthetic_unit)),
        tuple(partial_source_owners),
    )
    if result.witness(result.options) is None:
        raise ValueError("printed aggregate tare has no exact joint train-only equipment support")
    return result


def _source_partial_pairs(
    source: SourceTemplate, owner: int, configured: frozenset[str]
) -> frozenset[str]:
    """Retain an observed tare beside an exactly owned printed length and type.

    This is not fitted support and does not identify an unprinted height. It is
    usable only with an explicitly supplied private-equipment domain; ordinary
    cargo/equipment capacity checks still apply to every generated scenario.
    """
    container = source.target["documentPatch"]["containers"][owner]
    length, kind, size = partial_equipment_constraint(container)
    if kind is None or (length is None and size is None):
        return frozenset()
    path = f"documentPatch.containers[{owner}].typeDescription"
    # A certified binding can project OCR-joined words (for example
    # TANKCONTAINER) onto the same target description. Only whitespace may
    # differ; a different letter or punctuation is not source proof.
    normalized = " ".join(container["typeDescription"].casefold().split())
    compact = "".join(normalized.split())
    proved = any(
        path in getattr(binding, "target_paths", ())
        and any(
            "".join(slot.source_text.casefold().split()) == compact
            and source.source[slot.byte_start : slot.byte_end].decode() == slot.source_text
            for slot in binding.occurrences
        )
        for binding in source.template.bindings
    )
    if not proved and len(source.target["documentPatch"]["containers"]) == 1:
        from .descendant import _number_to_words
        from .equipment_receipts import owned_inventory, validate_source_receipt

        for binding in source.template.bindings:
            if binding.derivation != "equipment_receipt":
                continue
            paths = (*binding.target_paths, *binding.dependency_paths)
            if paths != ("documentPatch.containers",):
                continue
            inventory = owned_inventory(source.target, paths)
            if len(inventory) != 1 or inventory[0] != container:
                continue
            if all(
                normalized in " ".join(slot.source_text.casefold().split())
                and source.source[slot.byte_start : slot.byte_end].decode() == slot.source_text
                for slot in binding.occurrences
            ):
                for slot in binding.occurrences:
                    validate_source_receipt(
                        slot.source_text, inventory, number_words=_number_to_words
                    )
                proved = True
                break
    if not proved:
        return frozenset()
    prefix = {"20": "TWENTY_", "40": "FORTY_FOOT_", "45": "FORTY_FIVE_"}
    return frozenset(
        pair for pair in configured
        if pair.split("|")[1] == kind
        and (length is None or pair.split("|")[0].startswith(prefix[length]))
        and (size is None or pair.split("|")[0] == size)
    )
