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

from document_ocr.synthesis.curated import _number_style, flat
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
    for path in paths:
        for measure in _MEASURES:
            if path.endswith(f".{measure}.value"):
                return measure, None
    text = metadata.casefold()
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


def _token(surface: str, baseline: Decimal | None = None) -> tuple[str, Decimal]:
    # A typed numeric slot can have an attached unit (11500.000KGS).
    matches = list(re.finditer(r"[+-]?\d+(?:[.,]\d+)*", surface))
    if len(matches) != 1:
        raise ValueError(f"numeric owner is not a single measured scalar: {surface!r}")
    token = matches[0][0]
    values = {value for value, *_ in _numeric_interpretations(token)}
    if baseline is not None:
        if baseline not in values:
            raise ValueError(f"numeric owner does not print its baseline {baseline}: {surface!r}")
        return token, baseline
    if len(values) != 1:
        raise ValueError(f"numeric owner lacks an unambiguous baseline: {surface!r}")
    return token, values.pop()


def _quantum(surface: str, baseline: Decimal) -> Decimal:
    token, _ = _token(surface, baseline)
    # The number of printed digits bounds possible decimal precision exactly.
    for places in range(sum(c.isdigit() for c in token), -1, -1):
        quantum = Decimal(1).scaleb(-places)
        try:
            _number_style(token, str(baseline), str(baseline + quantum))
            return quantum
        except ValueError:
            pass
    raise ValueError(f"numeric owner has no exact presentation: {surface!r}")


def _format(surface: str, baseline: Decimal, value: Decimal) -> str:
    token, _ = _token(surface, baseline)
    return surface.replace(token, _number_style(token, str(baseline), str(value)), 1)


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
    for key, binding in blueprint.historical_bindings.items():
        if key not in active or binding["value_kind"] not in {"decimal_measurement", "integer"}:
            continue
        paths = tuple(current_path(p) for p in binding.get("target_paths", ()))
        role = _role(key + " " + binding.get("group_key", ""), paths)
        if role is None:
            continue
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
    return tuple(result)


def _distribute(total: Decimal, weights: Sequence[Decimal], quantum: Decimal) -> list[Decimal]:
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
    if sum(result) != total or any(v <= 0 for v in result):
        raise ValueError("positive physical row allocation cannot exactly represent the total")
    return result


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
        new = text
        if re.search(r"\bVENT(?:IL|ILL)", text, re.I):
            if rate is None:
                raise ValueError("printed ventilation requires a sampled observed ventilation rate")
            new = _vent_text(new, Decimal(rate))
        if thermal and re.search(r"(?:TEMP|SHIPPED\s+AT|DEGREES|\d\s*°?C\b)", new, re.I):
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


def _equipment(blueprint, target, support, scenario, surfaces, receipt):
    old = blueprint.target["documentPatch"].get("containerInformation", [])
    new = target["documentPatch"].get("containerInformation", [])
    active = {r.key for r in blueprint.regions if r.curated_key is None}
    tares = []
    for key, binding in blueprint.historical_bindings.items():
        if key not in active:
            continue
        texts = [o["source_text"] for o in binding["occurrences"]]
        metadata = key + " " + binding.get("group_key", "")
        if "tare" in key and binding["value_kind"] in {"integer", "decimal_measurement"}:
            owner = re.search(r"container[:_ ]?(\d+)", metadata)
            if owner is None:
                raise ValueError(f"tare binding lacks exact equipment ownership: {key}")
            index = int(owner[1])
            pair = new[index]["sizeCategory"] + "|" + new[index]["typeCategory"]
            old_pair = old[index]["sizeCategory"] + "|" + old[index]["typeCategory"]
            domains = [
                {
                    v
                    for v, *_ in _numeric_interpretations(
                        re.search(r"[+-]?\d+(?:[.,]\d+)*", text)[0]
                    )
                }
                for text in texts
            ]
            values = set.intersection(*domains)
            if len(values) > 1:
                values &= {v for v, p in support.tares.source_ids_by_kg_pair if p == old_pair}
            if len(values) != 1:
                raise ValueError(f"tare binding has ambiguous source values: {key}")
            baseline = values.pop()
            if pair == old_pair:
                value, provenance = baseline, [blueprint.document_id]
            else:
                candidates = sorted(v for v, p in support.tares.source_ids_by_kg_pair if p == pair)
                if not candidates:
                    raise ValueError(
                        f"sampled equipment has no train-observed tare support: {pair}"
                    )
                stream = DeterministicStream(
                    scenario.provenance["seed"],
                    "current-v7-physical-tare",
                    f"{scenario.source_id}:{scenario.variant}:{index}",
                )
                value = candidates[stream.randbelow(len(candidates))]
                provenance = list(support.tares.source_ids_by_kg_pair[value, pair])
            surfaces[key] = [_format(text, baseline, value) for text in texts]
            tares.append(
                {"owner": index, "pair": pair, "kg": str(value), "observationSources": provenance}
            )
            continue
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
                        format_equipment=_receipt_equipment_surface,
                        number_words=_number_to_words,
                    )
                )
            else:
                if len({(c["sizeCategory"], c["typeCategory"]) for c in selected_new}) != 1:
                    raise ValueError("one equipment type owner refers to unequal sampled types")
                suffix = re.search(r"\s+(?:FCL|LCL|STC|SAID)\b.*$", text)
                core = text[: suffix.start()] if suffix else text
                output.append(
                    _receipt_equipment_surface(selected_new[0], core)
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
