"""Missing-only numeric completion of retained shipments for newly visible OCR.

Existing totals, row facts, IDs and public targets are immutable. The caller
must review the proposal and attach provenance before admitting its receipt.
"""

import copy
from decimal import ROUND_HALF_UP, Decimal

from document_ocr.synthesis.paddle_render import _UNITS
from document_ocr.synthesis.retained_measure_shares import has_exact_proportional_total

FIELDS = {
    "grossWeight": ("grossKg", "kilogram", "payloadKg"),
    "netWeight": ("netKg", "kilogram", "payloadKg"),
    "volume": ("volumeM3", "cubic_metre", "volumeM3"),
}

# The proposal prompt requests at most three decimal places. Private retained
# totals can be more precise; the renderer, not an LLM, owns their exact sums.
PROPOSAL_QUANTUM = Decimal("0.001")


def reconcile_numeric_precision(candidate, request, values):
    """Propose an explicit, auditable rounding remainder; never relax validation.

    Only newly proposed measures may change, by at most the combined half-unit
    rounding uncertainty. The last missing row receives the exact remainder.
    Callers must retain the adjustment record and review plausibility separately.
    """
    if request != numeric_context(
        candidate, request["requestedMeasures"], request["completeCounts"]
    ):
        raise ValueError("numeric reconciliation context changed")
    if set(values) != {"unknown", "basis", *request["requested"]} or values["unknown"]:
        raise ValueError("numeric reconciliation requires a complete known proposal")
    result, changes = copy.deepcopy(values), []
    for field in request["requestedMeasures"]:
        if field not in request["knownTotals"]:
            continue
        names = [name for name, spec in request["requested"].items() if spec["field"] == field]
        if not names:
            continue
        proposed = []
        for name in names:
            if isinstance(values[name], bool):
                raise ValueError("boolean is not a numeric proposal")
            value = Decimal(str(values[name]))
            if not value.is_finite():
                raise ValueError("nonfinite numeric proposal")
            proposed.append(value)
        retained = sum(
            (
                Decimal(row["knownMeasures"][field])
                for row in request["rows"]
                if field in row["knownMeasures"]
            ),
            Decimal(0),
        )
        total = Decimal(request["knownTotals"][field])
        difference = total - retained - sum(proposed, Decimal(0))
        if not difference:
            continue
        if (
            any(value % PROPOSAL_QUANTUM for value in proposed)
            or abs(difference) > len(names) * PROPOSAL_QUANTUM / 2
        ):
            raise ValueError("numeric discrepancy exceeds proposal rounding uncertainty")
        name = names[-1]
        remainder = total - retained - sum(proposed[:-1], Decimal(0))
        result[name] = str(remainder)
        changes.append(
            {
                "field": name,
                "before": values[name],
                "after": str(remainder),
                "retainedTotal": str(total),
                "proposalQuantum": str(PROPOSAL_QUANTUM),
                "sumDifference": str(difference),
            }
        )
    # All prior bounds, public facts, count conservation and net/gross checks
    # remain mandatory; a rounded remainder cannot bypass any of them.
    complete_numeric(candidate, request, result)
    return result, changes


def unbounded_open_volume(field, equipment):
    """Open equipment has no enclosed-cube limit, not an unknown mass limit."""
    return field == "volume" and equipment.get("typeCategory") in {"OPEN_TOP", "FLAT_RACK"}


def amount(value, unit, desired):
    value = Decimal(str(value))
    if not value.is_finite() or value <= 0 or _UNITS[unit][0] != _UNITS[desired][0]:
        raise ValueError("invalid retained measure or dimension")
    return value * _UNITS[unit][1] / _UNITS[desired][1]


def numeric_context(candidate, wanted, counts=False):
    provenance = candidate["scenario"]["provenance"]
    if provenance["method"] != "registry_goods_train_physical_support_geography_v2":
        raise ValueError("numeric completion needs the retained physical-row contract")
    if any(k in provenance for k in ("declaredPackages", "registryLoadProfile")):
        raise ValueError("complete existing planning ledgers must be projected, not redrawn")
    if not set(wanted) <= FIELDS.keys():
        raise ValueError("unknown requested physical field")
    goods = candidate["target"]["documentPatch"]["goodsItemDetails"]
    source = candidate["scenario"]["cargo"]["containerInformation"]
    public = candidate["target"]["documentPatch"]["containerInformation"]
    rows = provenance["physicalRows"]
    if len(goods) != 1 or not rows or len(source) != len(rows) or len(public) != len(rows):
        raise ValueError("numeric completion inventory or goods cardinality differs")
    ids = [r["equipmentIdentifier"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != {r["equipmentIdentifier"] for r in source}:
        raise ValueError("numeric completion requires exact source-ID ownership")
    rows = [
        next(r for r in rows if r["equipmentIdentifier"] == s["equipmentIdentifier"])
        for s in source
    ]
    total_packages = sum(r["packageQuantity"] for r in goods[0]["numberAndTypeOfPackages"])
    totals, printed_totals = {}, {}
    for field, m in candidate["physical"]["measures"].items():
        if field in FIELDS:
            printed_totals[field] = str(amount(m["total"], m["unit"], FIELDS[field][1]))
            raw = m["total"]
            if "beforeRounding" in m:
                value, quantum = Decimal(m["beforeRounding"]), Decimal(m["quantum"])
                if (
                    not value.is_finite()
                    or value <= 0
                    or not quantum.is_finite()
                    or quantum <= 0
                    or quantum.normalize().as_tuple().digits != (1,)
                    or value.quantize(quantum, rounding=ROUND_HALF_UP) != Decimal(m["total"])
                ):
                    raise ValueError("retained measurement rounding certificate is invalid")
                raw = m["beforeRounding"]
            totals[field] = str(amount(raw, m["unit"], FIELDS[field][1]))
    for field in FIELDS:
        if field in goods[0]:
            m = goods[0][field]
            v = amount(m["value"], m["unit"], FIELDS[field][1])
            if field not in printed_totals or Decimal(printed_totals[field]) != v:
                raise ValueError("public measure lacks its matching retained planning total")
    requested, records = {}, []
    for i, row in enumerate(rows):
        known = {}
        for field, m in row["measures"].items():
            if field not in FIELDS:
                continue
            value = amount(m["value"], m["unit"], FIELDS[field][1])
            if field not in totals or amount(m["total"], m["unit"], FIELDS[field][1]) != Decimal(
                totals[field]
            ):
                raise ValueError("retained row measure disagrees with shipment authority")
            known[field] = str(value)
        for field in wanted:
            if field not in known:
                _, _, cap = FIELDS[field]
                bound = row["capacity"][cap]
                if bound is None and not unbounded_open_volume(field, public[i]):
                    raise ValueError("missing explicit equipment capacity for new measure")
                upper = Decimal(str(bound)) if bound is not None else None
                if field == "netWeight" and "grossWeight" in known:
                    upper = min(upper, Decimal(known["grossWeight"]))
                requested[f"row{i}_{field}"] = {
                    "row": i,
                    "field": field,
                    "upper": str(upper) if upper is not None else None,
                    "lower": known.get("netWeight", "0") if field == "grossWeight" else "0",
                }
        quantity = row["quantity"]
        if quantity is not None and (type(quantity) is not int or quantity <= 0):
            raise ValueError("invalid retained private count")
        if counts and quantity is None:
            requested[f"row{i}_quantity"] = {"row": i, "field": "quantity", "upper": total_packages}
        records.append(
            {
                "equipment": public[i],
                "capacity": row["capacity"],
                "quantity": quantity,
                "knownMeasures": known,
            }
        )
    if not requested:
        raise ValueError("no missing numeric facts require generation")
    return {
        "goods": goods[0],
        "rows": records,
        "knownTotals": totals,
        "packageTotal": total_packages,
        "requested": requested,
        "requestedMeasures": sorted(wanted),
        "completeCounts": counts,
    }


def complete_numeric(candidate, request, values):
    """Fill only requested absent facts, checking capacity and exact conservation."""
    if request != numeric_context(
        candidate, request["requestedMeasures"], request["completeCounts"]
    ):
        raise ValueError("numeric proposal context changed")
    if values["unknown"] or not values["basis"].strip():
        raise ValueError("numeric proposal is unknown or lacks a plausibility basis")
    if set(values) != {"unknown", "basis", *request["requested"]}:
        raise ValueError("numeric proposal field membership differs")
    result = copy.deepcopy(candidate)
    source = result["scenario"]["cargo"]["containerInformation"]
    rows = result["scenario"]["provenance"]["physicalRows"]
    rows_by_id = {r["equipmentIdentifier"]: r for r in rows}
    totals = result["physical"]["measures"]
    merged = [copy.deepcopy(r["knownMeasures"]) for r in request["rows"]]
    quantities = [r["quantity"] for r in request["rows"]]
    for name, spec in request["requested"].items():
        raw = values[name]
        if spec["field"] == "quantity":
            if type(raw) is not int or not 0 < raw <= request["packageTotal"]:
                raise ValueError("new count is not a positive bounded integer")
            quantities[spec["row"]] = raw
        else:
            if isinstance(raw, bool):
                raise ValueError("boolean is not a numeric measure")
            value = Decimal(str(raw))
            if (
                not value.is_finite()
                or value <= 0
                or value < Decimal(spec["lower"])
                or (spec["upper"] is not None and value > Decimal(spec["upper"]))
            ):
                raise ValueError("new measure exceeds its valid bounds")
            merged[spec["row"]][spec["field"]] = str(value)
    if request["completeCounts"]:
        if any(q is None for q in quantities) or sum(quantities) != request["packageTotal"]:
            raise ValueError("new row counts do not conserve declared packages")
        public_counts = {
            r["equipmentIdentifier"]: r["packageQuantity"]
            for r in request["goods"].get("splitGoodsPlacement", [])
            if "packageQuantity" in r
        }
        for row, quantity in zip(request["rows"], quantities, strict=True):
            identity = row["equipment"]["equipmentIdentifier"]
            if identity in public_counts and public_counts[identity] != quantity:
                raise ValueError("new private count conflicts with existing public allocation")
    for row, known in zip(request["rows"], merged, strict=True):
        if (
            "netWeight" in known
            and "grossWeight" in known
            and Decimal(known["netWeight"]) > Decimal(known["grossWeight"])
        ):
            raise ValueError("net mass exceeds gross mass")
        for field, value in known.items():
            cap = row["capacity"][FIELDS[field][2]]
            if cap is None and not unbounded_open_volume(field, row["equipment"]):
                raise ValueError("missing explicit equipment capacity for completed measure")
            if cap is not None and Decimal(value) > Decimal(str(cap)):
                raise ValueError("completed row exceeds equipment capacity")
    for field in request["requestedMeasures"]:
        total = sum((Decimal(r[field]) for r in merged), Decimal(0))
        if field in request["knownTotals"] and total != Decimal(request["knownTotals"][field]):
            # Only untouched, pre-existing proportional rows have a rational
            # share certificate. Newly proposed values must conserve exactly.
            untouched = all(field in r["measures"] for r in rows)
            unit = totals[field]["unit"]
            retained_total = amount(request["knownTotals"][field], FIELDS[field][1], unit)
            if not untouched or not has_exact_proportional_total(rows, field, retained_total):
                raise ValueError("new row facts do not conserve retained shipment total")
        unit = totals[field]["unit"] if field in totals else FIELDS[field][1]
        if field not in totals:
            totals[field] = {
                "total": str(total),
                "unit": unit,
                "method": "reviewed_paddle_missing_measure",
            }
        for i, source_row in enumerate(source):
            target = rows_by_id[source_row["equipmentIdentifier"]]["measures"]
            if field not in target:
                target[field] = {
                    "value": str(amount(merged[i][field], FIELDS[field][1], unit)),
                    "total": str(amount(total, FIELDS[field][1], unit)),
                    "unit": unit,
                }
    if request["completeCounts"]:
        for i, source_row in enumerate(source):
            row = rows_by_id[source_row["equipmentIdentifier"]]
            if row["quantity"] is None:
                row["quantity"] = quantities[i]
    # Nothing except missing private values may change here. Targets are only
    # projected later from the native occurrences by the reviewed template.
    assert result["target"] == candidate["target"]
    assert result["proof"] == candidate["proof"]
    assert result["joinedRawText"] == candidate["joinedRawText"]
    return result
