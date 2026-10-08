"""Couple public shipment facts to their owned numeric and operational printouts.

The sampler decides the shipment. This module only reconciles that shipment
with the source's proven numeric ownership and printed precision. Historical
numeric roles are read from binding metadata, never discovered by looking for
an equal number in arbitrary OCR. Private row measures and tares stay private.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any

from document_ocr.synthesis.container_semantics import (
    canonical_equipment_surface,
    review_source_equipment_surface,
)
from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_measurements import (
    format_measure as _format,
)
from document_ocr.synthesis.curated_measurements import (
    measured_number as _token,
)
from document_ocr.synthesis.curated_measurements import (
    printed_quantum as _quantum,
)
from document_ocr.synthesis.curated_scenarios import ShipmentScenario
from document_ocr.synthesis.curated_templates import SamplingBlueprint, current_path
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.rendering import _numeric_interpretations
from document_ocr.synthesis.template_compiler.descendant import (
    _number_to_words,
    _receipt_equipment_surface,
)
from document_ocr.synthesis.template_compiler.equipment_receipts import project_receipt
from document_ocr.synthesis.template_compiler.equipment_tares import TareSupport, build_support

_MEASURES = ("grossWeight", "netWeight", "volume")
_FACTORS = {
    "kilogram": Decimal(1),
    "metric_tonne": Decimal(1000),
    "pound": Decimal("0.45359237"),
    "cubic_metre": Decimal(1),
}


@dataclass(frozen=True)
class PhysicalSupport:
    """Train-only private empirical support, built once per campaign."""

    goods: Mapping[str, Mapping[str, Any]]
    tares: TareSupport

    @classmethod
    def fit(cls, train_rows: Sequence[Mapping[str, Any]]) -> PhysicalSupport:
        observations = []
        goods = {}
        for row in train_rows:
            patch = row["target"]["documentPatch"]
            groups = patch.get("goodsItemDetails", [])
            if len(groups) == 1:
                goods[row["documentId"]] = groups[0]
            # The established tare parser consumes a neutral observation shape;
            # this adapter does not convert or publish historical extraction labels.
            observations.append(
                {
                    "documentId": row["documentId"],
                    "joinedRawText": row["joinedRawText"],
                    "target": {
                        "documentPatch": {
                            "containers": [
                                {**c, "containerNumber": c["equipmentIdentifier"]}
                                for c in patch.get("containerInformation", [])
                            ],
                            "cargoGroups": groups,
                        }
                    },
                }
            )
        return cls(goods, build_support(observations))


@dataclass(frozen=True)
class PhysicalRenderPlan:
    """Apply target before lexical assembly; merge both value maps into rendering."""

    target: dict[str, Any]
    variable_values: dict[str, str]
    surface_values: dict[str, str | list[str]]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class _Owner:
    key: str
    measure: str
    row: int | None
    baseline: Decimal
    surfaces: tuple[str, ...]
    curated: bool


def _role(metadata: str, paths: Sequence[str]) -> tuple[str, int | None] | None:
    if paths and all(path.endswith(".unit") for path in paths):
        return None
    for path in paths:
        for measure in _MEASURES:
            if path.endswith(f".{measure}.value"):
                return measure, None
    text = metadata.casefold()
    if re.search(r"(?:weight|volume)[_:]unit\b", text):
        return None
    if "tare" in text or "ventilat" in text or "temperature" in text:
        return None
    if (
        "gross_weight" in text
        or "gross total" in text
        or "cargo_weight_total" in text
        or "cargo:weight_value" in text
    ):
        measure = "grossWeight"
    elif "net_weight" in text:
        measure = "netWeight"
    elif "volume" in text and not re.search(r"volume[_:]unit", text):
        measure = "volume"
    else:
        return None
    row = re.search(r"container[:_ ]?(\d+)|row_(?:gross_weight|net_weight|volume):(\d+)", text)
    return measure, int(next(value for value in row.groups() if value is not None)) if row else None


def _owners(blueprint: SamplingBlueprint) -> tuple[_Owner, ...]:
    old = flat(blueprint.target)
    result = []
    for variable in blueprint.contract.variables:
        if variable.kind not in {"mass", "volume"}:
            continue
        direct = [
            b.path for b in blueprint.contract.targets if b.expression == "{" + variable.key + "}"
        ]
        role = _role(variable.meaning, direct)
        if role is None:
            raise ValueError(f"physical variable has no declared measurement role: {variable.key}")
        result.append(
            _Owner(
                variable.key,
                *role,
                Decimal(variable.value),
                tuple(o.text for o in variable.occurrences),
                True,
            )
        )
    active = {r.key for r in blueprint.regions if r.curated_key is None}
    converted_aliases = {
        key
        for key, recipe in blueprint.ownership_data.get("surfaces", {}).items()
        if "measure_path" in recipe
    }
    aggregates = []
    sums = {
        "sum_gross_weight": "grossWeight",
        "sum_net_weight": "netWeight",
        "sum_volume": "volume",
    }
    for key, binding in blueprint.historical_bindings.items():
        if (
            key not in active
            or key in converted_aliases
            or binding["value_kind"] not in {"decimal_measurement", "integer"}
        ):
            continue
        paths = tuple(current_path(p) for p in binding.get("target_paths", ()))
        if binding.get("derivation") in sums:
            aggregates.append((key, binding, sums[binding["derivation"]]))
            continue
        role = _role(key + " " + binding.get("group_key", ""), paths)
        if role is None:
            continue
        allocation = re.search(r"allocation:(\d+):(\d+)", key + " " + binding.get("group_key", ""))
        if allocation:
            # Allocation order is not equipment order. Resolve the printed
            # foreign key before assigning a container's private measurements.
            path = (
                f"documentPatch.cargoAllocationGroups[{allocation[1]}]"
                f".allocations[{allocation[2]}].containerNumber"
            )
            identifiers = {
                v["source_value"]
                for b in blueprint.historical_bindings.values()
                for v in b.get("realization", {}).get("target_values", [])
                if v["target_path"] == path
                and any(v["source_value"] in o["source_text"] for o in b["occurrences"])
            }
            containers = blueprint.target["documentPatch"].get("containerInformation", [])
            indices = [
                i for i, c in enumerate(containers) if c.get("equipmentIdentifier") in identifiers
            ]
            if len(identifiers) != 1 or len(indices) != 1:
                raise ValueError(f"physical allocation lacks one proven equipment identity: {key}")
            role = role[0], indices[0]
        baseline_values = {
            Decimal(str(old[p])) for p in paths if p in old and isinstance(old[p], (float, int))
        }
        if len(baseline_values) > 1:
            raise ValueError(f"physical binding has conflicting target baselines: {key}")
        surfaces = tuple(o["source_text"] for o in binding["occurrences"])
        baseline = next(iter(baseline_values), None)
        parsed = [_token(s, baseline)[1] for s in surfaces]
        if len(set(parsed)) != 1:
            raise ValueError(f"repeated physical owner has conflicting baselines: {key}")
        result.append(_Owner(key, *role, parsed[0], surfaces, False))
    # Typed sums are not independent samples. Their exact source dependency
    # equations also disambiguate typography such as 132.000 versus 132000.
    pending = list(aggregates)
    while pending:
        advanced = False
        for item in list(pending):
            key, binding, measure = item
            by_key = {o.key: o for o in result}
            dependency_keys = [
                owned
                for dep in binding.get("dependency_bindings", [])
                for owned in blueprint.nested_bindings.get(dep, (dep,))
            ]
            if any(dep not in by_key for dep in dependency_keys):
                continue
            dependencies = [by_key[dep] for dep in dependency_keys]
            dependency_paths = [current_path(p) for p in binding.get("dependency_paths", [])]
            if dependencies and dependency_paths:
                raise ValueError(f"physical sum mixes duplicate ownership bases: {key}")
            if dependencies:
                rows = [o.row for o in dependencies]
                expected_rows = set(
                    range(len(blueprint.target["documentPatch"].get("containerInformation", [])))
                )
                if (
                    len(set(dependency_keys)) != len(dependency_keys)
                    or any(o.measure != measure for o in dependencies)
                    or len(set(rows)) != len(rows)
                    or (rows != [None] and set(rows) != expected_rows)
                ):
                    raise ValueError(
                        f"physical sum lacks distinct complete measure ownership: {key}"
                    )
                baseline = sum((o.baseline for o in dependencies), Decimal(0))
            elif dependency_paths and all(
                p.endswith(f".{measure}.value") and p in old for p in dependency_paths
            ):
                if len(set(dependency_paths)) != len(dependency_paths):
                    raise ValueError(f"physical sum repeats target dependencies: {key}")
                baseline = sum((Decimal(str(old[p])) for p in dependency_paths), Decimal(0))
            else:
                raise ValueError(f"physical sum has no proven measure dependencies: {key}")
            surfaces = tuple(o["source_text"] for o in binding["occurrences"])
            for surface in surfaces:
                _token(surface, baseline)
            result.append(_Owner(key, measure, None, baseline, surfaces, False))
            pending.remove(item)
            advanced = True
        if not advanced:
            raise ValueError("physical sum has missing or cyclic owned dependencies")
    for measure in _MEASURES:
        selected = [o for o in result if o.measure == measure]
        if len({o.baseline for o in selected if o.row is None}) > 1:
            raise ValueError(
                f"shipment measurement owners have conflicting source totals: {measure}"
            )
        source_rows = {}
        for owner in selected:
            if owner.row is not None:
                if owner.row in source_rows and source_rows[owner.row] != owner.baseline:
                    raise ValueError(f"repeated source measurement owners disagree: {measure}")
                source_rows[owner.row] = owner.baseline
        expected_rows = set(
            range(len(blueprint.target["documentPatch"].get("containerInformation", [])))
        )
        if source_rows and set(source_rows) == expected_rows:
            source_sum = sum(source_rows.values(), Decimal(0))
            if any(o.row is None and o.baseline != source_sum for o in selected):
                raise ValueError(
                    f"complete source measurement rows contradict shipment total: {measure}"
                )
    return tuple(result)


def _distribute(
    total: Decimal,
    weights: Sequence[Decimal],
    quantum: Decimal,
    *,
    allow_zero: bool = False,
) -> list[Decimal]:
    """Largest remainder on a common printed decimal lattice; exact total."""
    if not weights or any(w <= 0 for w in weights):
        raise ValueError("physical rows require positive, explicit sharing weights")
    units = total / quantum
    if units != units.to_integral_value():
        raise ValueError("physical total is not representable on the printed row lattice")
    exact = [units * w / sum(weights) for w in weights]
    values = [int(v.to_integral_value(rounding=ROUND_FLOOR)) for v in exact]
    remainder = int(units) - sum(values)
    order = sorted(range(len(values)), key=lambda i: (-(exact[i] - values[i]), i))
    for i in order[:remainder]:
        values[i] += 1
    result = [value * quantum for value in values]
    if sum(result) != total or any(v < 0 if allow_zero else v <= 0 for v in result):
        raise ValueError("positive physical row allocation cannot exactly represent the total")
    return result


def _couple_mass_rows(rows, totals, lattices, receipts, weights):
    """Reconcile differing printed precisions without making net exceed gross.

    Independent largest-remainder allocations can contradict one another when
    net and gross totals are close. Preserve already feasible allocations; for
    conflicting ones allocate on nested kilogram lattices with the coarser
    rows as the bounds. No shipment total or printed precision changes.
    """
    if not {"netWeight", "grossWeight"} <= rows.keys():
        return
    gross_factor = _FACTORS[receipts["grossWeight"]["unit"]]
    net_factor = _FACTORS[receipts["netWeight"]["unit"]]
    gross = [v * gross_factor for v in rows["grossWeight"]]
    net = [v * net_factor for v in rows["netWeight"]]
    if all(n <= g for n, g in zip(net, gross, strict=True)):
        return
    gross_quantum = lattices["grossWeight"] * gross_factor
    net_quantum = lattices["netWeight"] * net_factor
    ratio = max(gross_quantum, net_quantum) / min(gross_quantum, net_quantum)
    if ratio != ratio.to_integral_value():
        raise ValueError("net/gross printed row lattices are not nested in kilograms")
    if net_quantum >= gross_quantum:
        slack = totals["grossWeight"] * gross_factor - totals["netWeight"] * net_factor
        if slack < 0:
            raise ValueError("shipment net mass exceeds its gross mass")
        extra = _distribute(slack, weights, gross_quantum, allow_zero=True)
        rows["grossWeight"] = [(n + e) / gross_factor for n, e in zip(net, extra, strict=True)]
    else:
        bounded_net = _distribute(totals["netWeight"] * net_factor, gross, net_quantum)
        if any(n > g for n, g in zip(bounded_net, gross, strict=True)):
            raise ValueError("shipment net mass exceeds its gross mass")
        rows["netWeight"] = [n / net_factor for n in bounded_net]


def _measure_totals(blueprint, scenario, target, owners, support):
    goods = target["documentPatch"]["goodsItemDetails"][0]
    old_goods = blueprint.target["documentPatch"]["goodsItemDetails"][0]
    donor = support.goods.get(scenario.provenance["donorDocumentId"])
    if donor is None:
        raise ValueError("recorded physical donor is absent from train-only support")
    quantity = goods["numberAndTypeOfPackages"][0]["packageQuantity"]
    donor_quantity = sum(p["packageQuantity"] for p in donor["numberAndTypeOfPackages"])
    source_quantity = sum(p["packageQuantity"] for p in old_goods["numberAndTypeOfPackages"])
    totals, lattices, receipts = {}, {}, {}
    for measure in _MEASURES:
        relevant = [o for o in owners if o.measure == measure]
        if not relevant:
            continue
        quantum = max(_quantum(s, o.baseline) for o in relevant for s in o.surfaces)
        lattices[measure] = quantum
        if measure in goods:
            amount = Decimal(str(goods[measure]["value"]))
            method = "sampled_public_measure"
        elif all(o.baseline == 0 for o in relevant):
            # A printed zero/unknown volume is not a missing positive gold fact.
            totals[measure] = Decimal(0)
            receipts[measure] = {
                "method": "source_zero_unasserted",
                "total": "0",
                "quantum": str(quantum),
            }
            continue
        elif measure in donor:
            amount = (
                Decimal(str(donor[measure]["value"]))
                * _FACTORS[donor[measure]["unit"]]
                * quantity
                / donor_quantity
            )
            method = "train_donor_private_measure"
        else:
            if scenario.provenance["donorDocumentId"] != blueprint.document_id:
                raise ValueError(
                    f"different-product donor lacks required private {measure}; "
                    "declare the source's required_private_measures before sampling"
                )
            totals_seen = {o.baseline for o in relevant if o.row is None}
            if len(totals_seen) != 1:
                raise ValueError(f"source-only {measure} lacks a unique owned total or donor fact")
            amount = totals_seen.pop() * quantity / source_quantity
            method = "source_owned_private_per_package_measure"
        total = amount.quantize(quantum, rounding=ROUND_HALF_UP)
        if total <= 0:
            raise ValueError(f"physical {measure} rounds to zero")
        if measure in goods:
            goods[measure]["value"] = float(total)
        totals[measure] = total
        receipts[measure] = {
            "method": method,
            "beforeRounding": str(amount),
            "total": str(total),
            "quantum": str(quantum),
            "unit": goods[measure]["unit"]
            if measure in goods
            else "cubic_metre"
            if measure == "volume"
            else "kilogram",
        }
    if (
        "netWeight" in totals
        and "grossWeight" in totals
        and totals["netWeight"] * _FACTORS[receipts["netWeight"]["unit"]]
        > totals["grossWeight"] * _FACTORS[receipts["grossWeight"]["unit"]]
    ):
        raise ValueError("printed numeric rounding would make net exceed gross")
    return totals, lattices, receipts


def _temperature_text(text: str, old: Decimal, new: Decimal) -> str:
    pattern = re.compile(
        r"(?P<word>PLUS\s+|MINUS\s+)?(?P<number>[+-]?\d+(?:\.\d+)?)(?=\s*(?:°\s*)?(?:C(?:\b|E\b)|F\b|DEG(?:REE)?|CELSIUS|FAHRENHEIT))",
        re.I,
    )
    changed = False

    def replace(match):
        nonlocal changed
        value = Decimal(match["number"])
        if match["word"] and match["word"].strip().upper() == "MINUS":
            value = -abs(value)
        if value != old:
            raise ValueError(
                "owned carrying-temperature wording disagrees with its source setpoint"
            )
        changed = True
        return format(new, "f").rstrip("0").rstrip(".") if "." in format(new, "f") else str(new)

    result = pattern.sub(replace, text)
    if not changed:
        raise ValueError(f"temperature owner lacks an explicit signed temperature: {text!r}")
    return result


def _vent_text(text: str, rate: Decimal) -> str:
    pattern = re.compile(r"\d+(?:\.\d+)?(?=\s*(?:CBM|M3|M³)\s*(?:/\s*H(?:R)?|PER\s+HOUR))", re.I)
    if not pattern.search(text):
        raise ValueError("ventilation owner lacks an explicit cubic-metre/hour quantity")
    return pattern.sub(str(rate), text)


def _operational(blueprint, scenario, target, surfaces, receipt):
    before = flat(blueprint.target)
    old_containers = blueprint.target["documentPatch"].get("containerInformation", [])
    new_containers = target["documentPatch"].get("containerInformation", [])
    settings = [
        (
            Decimal(str(a["temperatureSetpoint"]["value"])),
            Decimal(str(b["temperatureSetpoint"]["value"])),
        )
        for a, b in zip(old_containers, new_containers, strict=True)
        if "temperatureSetpoint" in a
    ]
    if len(set(settings)) > 1:
        raise ValueError(
            "one-goods physical operational plan has conflicting carrying temperatures"
        )
    thermal = settings[0] if settings else None
    rate = scenario.provenance.get("ventilationCbmPerHour")
    old_handling = blueprint.target["documentPatch"]["goodsItemDetails"][0].get(
        "handlingInstructions", []
    )
    new_handling = target["documentPatch"]["goodsItemDetails"][0].get("handlingInstructions", [])
    if len(old_handling) != len(new_handling):
        raise ValueError("physical planner cannot invent or remove handling instruction fields")
    for i, text in enumerate(old_handling):
        # Route/origin dependencies are already applied by the shipment plan.
        # Thermal editing owns only its measured tokens, not the full field.
        new = new_handling[i]
        if re.search(r"\bVENT(?:IL|ILL)", text, re.I):
            if rate is None:
                raise ValueError("printed ventilation requires a sampled observed ventilation rate")
            new = _vent_text(new, Decimal(rate))
        reference = not re.search(r"\d", new) and re.search(
            r"\b(?:AS(?:\s+PER)?|SEE|SHOWN|STATED|SPECIFIED|INDICATED)\b"
            r"[^.;\n]{0,60}\b(?:ABOVE|BELOW|ELSEWHERE)\b",
            new,
            re.I,
        )
        if (
            thermal
            and not reference
            and re.search(r"(?:TEMP|SHIPPED\s+AT|DEGREES|\d\s*°?C\b)", new, re.I)
        ):
            new = _temperature_text(new, *thermal)
        new_handling[i] = new
    # Handle only historical executable regions, never overlapping whole lexical owners.
    active = {r.key for r in blueprint.regions if r.curated_key is None}
    for key, binding in blueprint.historical_bindings.items():
        if key not in active:
            continue
        paths = [current_path(p) for p in binding.get("target_paths", ())]
        texts = [o["source_text"] for o in binding["occurrences"]]
        output = None
        if thermal and any(".temperatureSetpoint" in p for p in paths):
            if all(p.endswith(".temperatureSetpoint.unit") for p in paths):
                continue
            if all(
                p.endswith(".temperatureSetpoint.value") or ".handlingInstructions[" in p
                for p in paths
            ):
                output = [
                    str(thermal[1])
                    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text)
                    else _temperature_text(text, *thermal)
                    for text in texts
                ]
            else:
                output = [_temperature_text(text, *thermal) for text in texts]
        elif rate is not None and (
            "ventilation_rate" in key
            or any(
                re.search(r"\bVENT(?:IL|ILL)", text, re.I) and re.search(r"\d", text)
                for text in texts
            )
        ):
            output = [_vent_text(text, Decimal(rate)) for text in texts]
        elif (
            rate is not None
            and any(".handlingInstructions[" in p for p in paths)
            and all(re.fullmatch(r"VENT(?:IL|ILL)ATION", text, re.I) for text in texts)
        ):
            # Caption and measured rate have separate owned regions in some
            # layouts. The rate region is independently rendered above.
            output = texts
        elif (
            thermal
            and any(".handlingInstructions[" in p for p in paths)
            and any(re.search(r"\d\s*(?:C\b|DEG)", text, re.I) for text in texts)
        ):
            output = [_temperature_text(text, *thermal) for text in texts]
        dg = scenario.provenance.get("dgPrintedFacts")
        if dg and any(p.endswith(".hazardCategory") for p in paths):
            output = []
            for text in texts:
                match = re.match(r"\s*\d(?:\.\d)?", text)
                if not match:
                    raise ValueError("DG hazard owner has no exact printed regulatory class")
                output.append(text[: match.start()] + dg["class"] + text[match.end() :])
        elif dg and any(p.endswith(".unNumber") for p in paths):
            old_codes = {str(before[p]) for p in paths if p.endswith(".unNumber")}
            if len(old_codes) != 1:
                raise ValueError("one DG UN-number owner has conflicting target values")
            code = old_codes.pop()
            output = []
            for text in texts:
                if len(re.findall(r"(?<!\d)" + re.escape(code) + r"(?!\d)", text)) != 1:
                    raise ValueError("DG UN-number owner lacks exactly its source identifier")
                output.append(
                    re.sub(r"(?<!\d)" + re.escape(code) + r"(?!\d)", dg["unNumber"], text)
                )
        if output is not None:
            surfaces[key] = output
    receipt["temperature"] = [str(v) for v in thermal] if thermal else None
    receipt["ventilationCbmPerHour"] = rate
    receipt["handlingChanges"] = [
        {"path": p, "before": before[p], "after": v}
        for p, v in flat(target).items()
        if ".handlingInstructions[" in p and before.get(p) != v
    ]


def _tare_surfaces(blueprint, target, support, scenario, surfaces):
    """Resolve typed tare rows and explicit sums before rendering equipment text."""
    old = blueprint.target["documentPatch"].get("containerInformation", [])
    new = target["documentPatch"].get("containerInformation", [])
    active = {r.key for r in blueprint.regions if r.curated_key is None}
    bindings = {
        key: b
        for key, b in blueprint.historical_bindings.items()
        if key in active
        and "tare" in key
        and b["value_kind"] in {"integer", "decimal_measurement"}
        and not re.search(r"tare(?:_weight)?[_:]unit\b", key)
    }
    declared_owners = blueprint.ownership_data.get("tare_owners", {})
    declared_baselines = blueprint.ownership_data.get("tare_baselines", {})
    if (declared_owners.keys() | declared_baselines.keys()) - bindings.keys():
        raise ValueError("tare declarations refer to an inactive or non-numeric tare owner")
    owners = {}

    def owner_indices(key, trail=()):
        if key in trail or key not in bindings:
            raise ValueError("tare sum has cyclic or missing measurement dependencies")
        if key in owners:
            return owners[key]
        binding = bindings[key]
        explicit = re.search(r"container[:_ ]?(\d+)", key + " " + binding.get("group_key", ""))
        deps = binding.get("dependency_bindings", [])
        if key in declared_owners:
            indices = tuple(declared_owners[key])
            if explicit and indices != (int(explicit[1]),):
                raise ValueError("declared tare ownership conflicts with its container metadata")
        elif deps and binding.get("derivation") in {"sum_decimal_values", "sum_tare_weight"}:
            indices = tuple(i for dep in deps for i in owner_indices(dep, (*trail, key)))
        elif explicit:
            indices = (int(explicit[1]),)
        elif len(old) == 1:
            indices = (0,)
        else:
            raise ValueError(f"tare binding lacks exact equipment ownership: {key}")
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(old) for i in indices)
        ):
            raise ValueError("tare owner indices must be unique members of source equipment")
        owners[key] = indices
        return indices

    def pair(container):
        parts = [container.get(k) for k in ("sizeCategory", "typeCategory")]
        return "|".join(parts) if all(parts) else None

    def unchanged_pair(index):
        return all(
            old[index].get(field) == new[index].get(field)
            for field in ("sizeCategory", "typeCategory")
        )

    def baseline(key):
        binding = bindings[key]
        texts = [o["source_text"] for o in binding["occurrences"]]
        if key in declared_baselines:
            declaration = declared_baselines[key]
            if (
                set(declaration) != {"kg", "reason", "sourceSha256"}
                or not declaration["reason"].strip()
                or declaration["sourceSha256"] != digest(blueprint.source.encode())
            ):
                raise ValueError("tare baseline requires a current source hash and review reason")
            value = Decimal(declaration["kg"])
            if not value.is_finite() or value <= 0:
                raise ValueError("tare baseline must be a positive finite kilogram value")
            for text in texts:
                _token(text, value)
            return value
        domains = []
        for text in texts:
            matches = list(re.finditer(r"[+-]?\d(?:[\d., ]*\d)?", text))
            if len(matches) != 1:
                raise ValueError(f"tare binding is not one scalar: {key}")
            values = set()
            for value, *_ in _numeric_interpretations(matches[0][0]):
                try:
                    _token(text, value)
                    if value > 0:
                        values.add(value)
                except ValueError:
                    continue
            domains.append(values)
        values = set.intersection(*domains)
        indices = owners[key]
        if len(values) > 1 and len(indices) == 1:
            old_pair = pair(old[indices[0]])
            values &= {v for v, p in support.tares.source_ids_by_kg_pair if p == old_pair}
        if len(values) != 1:
            raise ValueError(f"tare binding has ambiguous source values: {key}")
        return values.pop()

    for key in bindings:
        owner_indices(key)
    baselines = {key: baseline(key) for key in bindings}
    rendered_tares = {}
    by_index = {}
    receipts = []
    for key, indices in owners.items():
        if len(indices) != 1:
            continue
        index = indices[0]
        if index in by_index:
            if by_index[index][0] != baselines[key]:
                raise ValueError("repeated tare owners disagree about the same container")
            value, provenance = by_index[index][1:]
        elif unchanged_pair(index):
            value, provenance = baselines[key], [blueprint.document_id]
        else:
            value, provenance = _sample_tare(pair(new[index]), index, support, scenario)
        by_index[index] = (baselines[key], value, provenance)
        surfaces[key] = [
            _format(o["source_text"], baselines[key], value) for o in bindings[key]["occurrences"]
        ]
        rendered_tares[key] = value
        receipts.append(
            {
                "owner": index,
                "pair": pair(new[index]),
                "kg": str(value),
                "observationSources": provenance,
            }
        )
    for key, indices in owners.items():
        if len(indices) == 1:
            continue
        covered = set(indices) & by_index.keys()
        if covered and covered != set(indices):
            missing = set(indices) - covered
            if len(missing) != 1:
                raise ValueError("aggregate tare has multiple unproven individual source values")
            index = missing.pop()
            residual = baselines[key] - sum(by_index[i][0] for i in covered)
            if residual <= 0:
                raise ValueError("aggregate tare leaves a non-positive source row residual")
            if unchanged_pair(index):
                value, provenance = residual, [blueprint.document_id]
            else:
                value, provenance = _sample_tare(pair(new[index]), index, support, scenario)
            by_index[index] = (residual, value, provenance)
            receipts.append(
                {
                    "owner": index,
                    "pair": pair(new[index]),
                    "kg": str(value),
                    "observationSources": provenance,
                    "derivedSourceResidualKg": str(residual),
                    "aggregateOwner": key,
                }
            )
        if covered:
            if sum(by_index[i][0] for i in indices) != baselines[key]:
                raise ValueError("printed aggregate tare differs from its exact source row sum")
            value = sum(by_index[i][1] for i in indices)
            provenance = sorted({s for i in indices for s in by_index[i][2]})
        elif all(unchanged_pair(i) for i in indices):
            value, provenance = baselines[key], [blueprint.document_id]
        else:
            samples = [_sample_tare(pair(new[i]), i, support, scenario) for i in indices]
            value = sum(v for v, _ in samples)
            provenance = sorted({s for _, ids in samples for s in ids})
        surfaces[key] = [
            _format(o["source_text"], baselines[key], value) for o in bindings[key]["occurrences"]
        ]
        rendered_tares[key] = value
        receipts.append(
            {
                "owners": list(indices),
                "kg": str(value),
                "observationSources": provenance,
                "ownershipBasis": "explicit_aggregate_or_declared_sum",
            }
        )
    old_leaves, new_leaves = flat(blueprint.target), flat(target)
    for key, binding in blueprint.historical_bindings.items():
        deps = binding.get("dependency_bindings", [])
        if (
            key not in active
            or binding.get("derivation") != "sum_decimal_values"
            or not (set(deps) & bindings.keys())
        ):
            continue
        if key in bindings:
            continue  # A pure tare sum was handled with its exact equipment owners.
        paths = [current_path(p) for p in binding.get("dependency_paths", [])]
        if (
            not paths
            or len(set(deps)) != len(deps)
            or len(set(paths)) != len(paths)
            or any(dep not in rendered_tares for dep in deps)
            or any(
                not p.endswith(".grossWeight.value") or p not in old_leaves or p not in new_leaves
                for p in paths
            )
            or any(
                old_leaves[p.removesuffix("value") + "unit"] != "kilogram"
                or new_leaves[p.removesuffix("value") + "unit"] != "kilogram"
                for p in paths
            )
        ):
            raise ValueError(
                f"loaded-weight sum lacks distinct kilogram cargo/tare dependencies: {key}"
            )
        original = sum((baselines[d] for d in deps), Decimal(0)) + sum(
            (Decimal(str(old_leaves[p])) for p in paths), Decimal(0)
        )
        replacement = sum((rendered_tares[d] for d in deps), Decimal(0)) + sum(
            (Decimal(str(new_leaves[p])) for p in paths), Decimal(0)
        )
        surfaces[key] = [
            _format(o["source_text"], original, replacement) for o in binding["occurrences"]
        ]
    return bindings.keys(), receipts


def _sample_tare(pair, index, support, scenario):
    candidates = sorted(v for v, p in support.tares.source_ids_by_kg_pair if p == pair)
    if not pair or not candidates:
        raise ValueError(f"sampled equipment has no train-observed tare support: {pair}")
    stream = DeterministicStream(
        scenario.provenance["seed"],
        "current-v7-physical-tare",
        f"{scenario.source_id}:{scenario.variant}:{index}",
    )
    value = candidates[stream.randbelow(len(candidates))]
    return value, list(support.tares.source_ids_by_kg_pair[value, pair])


def _complete_equipment_surface(value, source):
    """Render the whole public pair, not merely a compatible height predicate.

    Historical receipts can legitimately say HC for a reefer identified elsewhere.
    A resampled curated unit must itself print its complete new classification;
    retaining only HC would hide a newly sampled refrigerated/open-top type.
    """
    surface = _receipt_equipment_surface(value, source)
    observed = review_source_equipment_surface(surface, temperature_present=False)
    if (observed.size_category, observed.type_category) == (
        value["sizeCategory"],
        value["typeCategory"],
    ):
        return surface
    return canonical_equipment_surface(value["sizeCategory"], value["typeCategory"])


def _equipment(blueprint, target, support, scenario, surfaces, receipt):
    old = blueprint.target["documentPatch"].get("containerInformation", [])
    new = target["documentPatch"].get("containerInformation", [])
    active = {r.key for r in blueprint.regions if r.curated_key is None}
    tare_keys, tares = _tare_surfaces(blueprint, target, support, scenario, surfaces)
    for key, binding in blueprint.historical_bindings.items():
        if key not in active or key in tare_keys:
            continue
        texts = [o["source_text"] for o in binding["occurrences"]]
        metadata = key + " " + binding.get("group_key", "")
        paths = [current_path(p) for p in binding.get("target_paths", ())]
        is_type = any(
            ".containerInformation" in p
            and p.endswith((".typeDescription", ".sizeCategory", ".typeCategory"))
            for p in paths
        )
        is_receipt = (
            "equipment_receipt" in key
            or ("equipment:container" in key and "receipt" in key)
            or "aggregate_container_receipt" in key
            or "container_summary_equipment" in key
            or binding.get("derivation") == "equipment_receipt"
        )
        if not (is_type or is_receipt):
            continue
        indices = {
            int(m[1]) for p in paths for m in re.finditer(r"containerInformation\[(\d+)\]", p)
        }
        if not indices:
            owner = re.search(r"container[:_ ]?(\d+)", metadata)
            indices = {int(owner[1])} if owner else set(range(len(new)))
        selected_old = [old[i] for i in sorted(indices)]
        selected_new = [new[i] for i in sorted(indices)]
        if any(not c.get("sizeCategory") or not c.get("typeCategory") for c in selected_new):
            if any(
                before.get(field) != after.get(field)
                for before, after in zip(selected_old, selected_new, strict=True)
                for field in ("sizeCategory", "typeCategory")
            ):
                raise ValueError("partial equipment wording cannot express a changed category")
            # Missing dimensions are not licensed by a historical text binding.
            # The scenario retained the same partial facts: keep their wording.
            surfaces[key] = texts
            continue
        output = []
        for text in texts:
            if is_receipt and re.fullmatch(r"\d+", text):
                output.append(str(len(indices)))
                continue
            if is_receipt:
                output.append(
                    project_receipt(
                        text,
                        selected_old,
                        selected_new,
                        format_equipment=_complete_equipment_surface,
                        number_words=_number_to_words,
                    )
                )
            else:
                if len({(c["sizeCategory"], c["typeCategory"]) for c in selected_new}) != 1:
                    raise ValueError("one equipment type owner refers to unequal sampled types")
                suffix = re.search(r"\s+(?:FCL|LCL|STC|SAID)\b.*$", text)
                core = text[: suffix.start()] if suffix else text
                output.append(
                    _complete_equipment_surface(selected_new[0], core)
                    + (suffix[0] if suffix else "")
                )
        surfaces[key] = output
    receipt["tares"] = tares


def prepare_physical_render(
    blueprint: SamplingBlueprint,
    scenario: ShipmentScenario,
    sampled_target: dict[str, Any],
    *,
    support: PhysicalSupport,
) -> PhysicalRenderPlan:
    """Create one coherent physical rendering; fail on unowned or ambiguous facts.

    Does not alter inputs, write files, call an LLM, add public fields, or sample
    cargo identities. Region ownership and source shape must already be pinned.
    """
    if scenario.source_id != blueprint.document_id:
        raise ValueError("physical plan/source identity mismatch")
    target = deepcopy(sampled_target)
    owners = _owners(blueprint)
    totals, lattices, measures_receipt = _measure_totals(
        blueprint, scenario, target, owners, support
    )
    physical_rows = scenario.provenance.get("physicalRows", [])
    weights = [
        Decimal(row["shareNumerator"]) / Decimal(row["shareDenominator"]) for row in physical_rows
    ]
    rows = {}
    for measure, total in totals.items():
        if any(o.row is not None for o in owners if o.measure == measure):
            rows[measure] = (
                [Decimal(0)] * len(weights)
                if total == 0
                else _distribute(total, weights, lattices[measure])
            )
    _couple_mass_rows(rows, totals, lattices, measures_receipt, weights)
    values, surfaces = {}, {}
    for owner in owners:
        if owner.row is not None and owner.row >= len(weights):
            raise ValueError(f"physical row index exceeds sampled equipment: {owner.key}")
        value = totals[owner.measure] if owner.row is None else rows[owner.measure][owner.row]
        if owner.curated:
            values[owner.key] = str(value)
        else:
            surfaces[owner.key] = [_format(text, owner.baseline, value) for text in owner.surfaces]
    leaves = flat(target)
    for variable in blueprint.contract.variables:
        if variable.kind != "count":
            continue
        direct = [
            b.path for b in blueprint.contract.targets if b.expression == "{" + variable.key + "}"
        ]
        counts = {Decimal(str(leaves[p])) for p in direct if p in leaves}
        if not counts and "unpackaged_status" in variable.meaning:
            counts = {
                Decimal(
                    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
                        "packageQuantity"
                    ]
                )
            }
        if len(counts) != 1:
            raise ValueError(f"count variable lacks one authoritative target: {variable.key}")
        values[variable.key] = str(counts.pop())
    # Source-only written-out package totals must track the newly sampled count.
    quantity = target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ]
    active = {r.key for r in blueprint.regions if r.curated_key is None}
    for key, binding in blueprint.historical_bindings.items():
        if key not in active or binding["value_kind"] != "integer":
            continue
        if re.search(r"(?:total_(?:carton_count|packages).*words|package:.*number_words)", key):
            surfaces[key] = [
                _number_to_words(quantity)
                + (" Only" if re.search(r"\bOnly$", o["source_text"], re.I) else "")
                for o in binding["occurrences"]
            ]
        elif re.search(r"reviewed:package_count|total_package_quantity|total_items", key):
            surfaces[key] = [
                _format(o["source_text"], _token(o["source_text"])[1], Decimal(quantity))
                for o in binding["occurrences"]
            ]
    receipt = {
        "measures": measures_receipt,
        "rows": {k: [str(v) for v in vv] for k, vv in rows.items()},
        "ownershipBasis": "current_target_bindings_and_declared_historical_numeric_roles",
    }
    _equipment(blueprint, target, support, scenario, surfaces, receipt)
    _operational(blueprint, scenario, target, surfaces, receipt)
    if set(flat(target)) != set(flat(sampled_target)):
        raise ValueError("physical plan changed public field presence")
    for measure, row_values in rows.items():
        if sum(row_values) != totals[measure]:
            raise ValueError("physical row conservation failed")
    for index, row in enumerate(physical_rows):
        capacity = row["capacity"]
        actual = {}
        for field in _MEASURES:
            if field not in totals or totals[field] == 0:
                continue
            amount = rows[field][index] if field in rows else totals[field] * weights[index]
            actual[field] = amount * _FACTORS[measures_receipt[field]["unit"]]
        # A total-only mass has no printed per-container allocation. Couple its
        # feasibility estimate to the rounded rows of the other mass, rather
        # than comparing rounded gross rows with an unrounded net split (or
        # vice versa). This changes neither printed values nor target labels.
        for field, counterpart in (("netWeight", "grossWeight"), ("grossWeight", "netWeight")):
            if (
                field in actual
                and field not in rows
                and counterpart in rows
                and counterpart in actual
            ):
                ratio = (
                    totals[field]
                    * _FACTORS[measures_receipt[field]["unit"]]
                    / (totals[counterpart] * _FACTORS[measures_receipt[counterpart]["unit"]])
                )
                actual[field] = actual[counterpart] * ratio
        if actual.get("grossWeight", 0) > Decimal(capacity["payloadKg"]):
            raise ValueError("rendered row exceeds sampled equipment payload after rounding")
        if capacity["volumeM3"] is not None and actual.get("volume", 0) > Decimal(
            capacity["volumeM3"]
        ):
            raise ValueError("rendered row exceeds sampled enclosed volume after rounding")
        if (
            "netWeight" in actual
            and "grossWeight" in actual
            and actual["netWeight"] > actual["grossWeight"]
        ):
            raise ValueError("rendered row net mass exceeds its gross mass")
    return PhysicalRenderPlan(target, values, surfaces, receipt)
