"""Render reviewed native Paddle templates over explicit sampled shipment facts.

All source owners are exact; geometry, wording provenance and target projection
fail closed. A candidate still requires semantic review before publication.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.spatial_inputs.paddle import RegionSpan, fingerprint, serialize
from document_ocr.spatial_inputs.paddle import pages_from_record as load_pages
from document_ocr.synthesis.curated_auxiliary import _source_format_identifier
from document_ocr.synthesis.curated_campaign import shifted_public_date
from document_ocr.synthesis.curated_equipment_aliases import (
    preserve_non_operating_reefer,
)
from document_ocr.synthesis.curated_hs import hs_digits, render_source_hs
from document_ocr.synthesis.curated_measurements import _UNITS, format_measure
from document_ocr.synthesis.curated_ownership import _declared_date_surface
from document_ocr.synthesis.dangerous_goods_registry import _PACKING_GROUP, _broad_hazard
from document_ocr.synthesis.generators import DeterministicStream, generate_container_number
from document_ocr.synthesis.paddle_bindings import (
    PaddleReplacement,
    compile_blocks,
    partition_region_frames,
    project_replacements,
    validate_existing_repetition,
)
from document_ocr.synthesis.phone_presentations import format_phone_presentation
from document_ocr.synthesis.retained_measure_shares import has_exact_proportional_total
from document_ocr.synthesis.template_compiler.contact_values import fax, phone
from document_ocr.synthesis.template_compiler.equipment_tares import TareConstraints
from document_ocr.synthesis.thermal_goods import render_synthetic_hs_code
from document_ocr.synthesis.transport_auxiliary import imo_check_digit, sample_imo_number


def strip_postal_captions(value, action):
    """Remove explicitly reviewed captions, never their postal values or input text."""
    captions = action.get("stripPostalCaptions", [])
    if not isinstance(captions, list) or (
        captions and not action["target"].endswith("/addressLine")
    ):
        raise ValueError("postal-caption projection is restricted to addressLine")
    for caption in captions:
        if not isinstance(caption, str) or not re.fullmatch(r"[A-Za-z][A-Za-z ]*[: ]", caption):
            raise ValueError("reviewed postal caption must end with a colon or explicit space")
        # Unpunctuated labels require a digit-bearing postcode token, including
        # alphanumeric codes such as BD1727 or P96. 'POSTAL CODE ROAD' is not one.
        suffix = r"\s*(?=[A-Z0-9-]*\d)" if caption.endswith(" ") else r"\s*"
        value = re.sub(r"(?<!\w)" + re.escape(caption) + suffix, "", value, flags=re.I)
    return " ".join(value.split()) if captions else value


def render_soft_line_breaks(value, rule):
    """Reflow a reviewed paragraph without carrying over GLM hard line breaks.

    Ownership must already establish one continuous field. This changes only
    line-boundary whitespace; native column widths still determine wrapping.
    """
    if "softLineBreaks" not in rule:
        return value
    if rule["softLineBreaks"] is not True:
        raise ValueError("soft line breaks require explicit reviewed enablement")
    return re.sub(r"[^\S\r\n]*(?:\r\n|\r|\n)[^\S\r\n]*", " ", value)


def spans_for(rule, pages):
    if "fragments" in rule:
        entries = []
        for p, i, text in rule["fragments"]:
            source = pages[p].regions[i].text
            if not text or source.count(text) != 1:
                raise ValueError("reviewed fragment is not unique in its native region")
            start = source.index(text)
            entries.append((p, i, start, start + len(text)))
    elif "regions" in rule:
        entries = [(p, i, 0, len(pages[p].regions[i].text)) for p, i in rule["regions"]]
    else:
        entries = rule["spans"]
    return tuple(
        RegionSpan(p, i, lo, hi, pages[p].regions[i].text[lo:hi]) for p, i, lo, hi in entries
    )


def order_containers_by_printed_occurrence(target, pages):
    """Order whole inventory objects by their first literal native occurrence.

    Only identifier presentation separators are ignored. No identities are
    repaired, and allocation references remain attached to their identities.
    """
    containers = target["containerInformation"]
    text = re.sub(r"[^A-Z0-9]", "", serialize(pages, geometry="none").upper())
    positions = {}
    for container in containers:
        identity = container["equipmentIdentifier"]
        token = re.sub(r"[^A-Z0-9]", "", identity.upper())
        if not token or token not in text or identity in positions:
            raise ValueError("inventory ordering needs unique, printed identifiers")
        positions[identity] = text.index(token)
    target["containerInformation"] = sorted(
        containers, key=lambda container: positions[container["equipmentIdentifier"]]
    )


def sampled_shipment_fact(spec, candidate):
    """Read a total using the receipt's explicit sampling contract and unit."""
    provenance = candidate["scenario"]["provenance"]
    method = provenance["method"]
    if "migrationMissingMeasures" in candidate:
        # Reviewed migration completions explicitly own these previously absent
        # totals. Their old sampler method still describes the retained cargo.
        completion = candidate["migrationMissingMeasures"]
        if not completion["proposal"] or not completion["review"]:
            raise ValueError("missing-measure completion lacks review provenance")
        if spec["field"] in candidate["physical"]["measures"]:
            raise ValueError("missing-measure completion overwrites a retained total")
        fact = provenance["explicitPhysicalFacts"][spec["field"]]
    elif "registryLoadProfile" in provenance:
        profile = provenance["registryLoadProfile"]
        packages = candidate["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
        if len(packages) != 1 or packages[0]["typeCategory"] != profile["package_category"]:
            raise ValueError("sampled profile differs from declared packages")
        fields = {"grossWeight": ("unit_gross_kg", "kilogram"),
                  "netWeight": ("unit_net_kg", "kilogram"),
                  "volume": ("unit_volume_m3", "cubic_metre")}
        key, unit = fields[spec["field"]]
        fact = provenance["explicitPhysicalFacts"][spec["field"]]
        if fact["unit"] != unit or Decimal(str(fact["value"])) != Decimal(str(profile[key])) * packages[0]["packageQuantity"]:
            raise ValueError("sampled total differs from its registry profile")
    elif "declaredPackages" in provenance:
        from document_ocr.synthesis.declared_packages import DeclaredPackages

        ledger = DeclaredPackages.from_receipt(provenance["declaredPackages"])
        key, unit = {"grossWeight": ("grossKg", "kilogram"), "netWeight": ("netKg", "kilogram"),
                     "volume": ("volumeM3", "cubic_metre")}[spec["field"]]
        fact = provenance["explicitPhysicalFacts"][spec["field"]]
        if fact["unit"] != unit or Decimal(str(fact["value"])) != Decimal(str(ledger.totals()[key])):
            raise ValueError("sampled total differs from its declared-group ledger")
    elif method == "independent_full_registry_flat_v1":
        fact = provenance["explicitPhysicalFacts"][spec["field"]]
    elif method == "registry_goods_train_physical_support_geography_v2":
        measure = candidate["physical"]["measures"][spec["field"]]
        fact = {"value": measure["total"], "unit": measure["unit"]}
    else:
        raise ValueError("unsupported sampled shipment-measure method")
    source_unit, target_unit = fact["unit"], spec["unit"]
    if _UNITS[source_unit][0] != _UNITS[target_unit][0]:
        raise ValueError("sampled shipment measure changes dimension")
    value = Decimal(str(fact["value"]))
    if not value.is_finite() or value <= 0:
        raise ValueError("sampled shipment measure must be finite and positive")
    if method != "registry_goods_train_physical_support_geography_v2" or "registryLoadProfile" in provenance:
        measures = candidate.get("physical", {}).get("measures", {})
        goods = candidate.get("target", {}).get("documentPatch", {}).get("goodsItemDetails", [])
        public_measure = goods[0].get(spec["field"]) if len(goods) == 1 else None
        if (spec["field"] in measures and public_measure is not None
                and public_measure["unit"] == target_unit):
            # Repeated totals in the same printed unit must agree with the
            # retained target's verified rounding contract. Newly exposed Paddle
            # measurements (including different units) retain planning precision;
            # an unused old formatting contract is not their display authority.
            measure = measures[spec["field"]]
            unit = measure["unit"]
            if _UNITS[unit][0] != _UNITS[source_unit][0]:
                raise ValueError("rounded sampled measure changes dimension")
            before = value * _UNITS[source_unit][1] / _UNITS[unit][1]
            quantum, total = Decimal(str(measure["quantum"])), Decimal(str(measure["total"]))
            if not quantum.is_finite() or quantum <= 0 or not total.is_finite() or total <= 0:
                raise ValueError("rounded sampled measure must be finite and positive")
            if before != Decimal(str(measure["beforeRounding"])) or total != before.quantize(
                quantum, rounding=ROUND_HALF_UP
            ):
                raise ValueError("rounded sampled measure disagrees with its planning contract")
            if total * _UNITS[unit][1] / _UNITS[target_unit][1] != Decimal(str(public_measure["value"])):
                raise ValueError("rounded sampled measure disagrees with its retained public target")
            value, source_unit = total, unit
    return {"value": value * _UNITS[source_unit][1] / _UNITS[target_unit][1], "unit": target_unit}


def sampled_allocation_fact(spec, candidate):
    """Read an allocation from its declared sampler contract, without resampling.

    Retained observed-support receipts and independent receipts encode their
    already sampled row measures differently. Dispatch on the recorded method;
    a missing field in either contract is an error, not a reason to try another.
    """
    provenance = candidate["scenario"]["provenance"]
    source = candidate["scenario"]["cargo"]["containerInformation"]
    indices = [
        i
        for i, c in enumerate(source)
        if c["equipmentIdentifier"] == spec["sourceEquipmentIdentifier"]
    ]
    if len(indices) != 1:
        raise ValueError("sampled allocation needs one source equipment owner")
    field, index = spec["field"], indices[0]
    fields = {
        "grossKg": ("grossWeight", "kilogram"),
        "netKg": ("netWeight", "kilogram"),
        "volumeM3": ("volume", "cubic_metre"),
    }
    measure, unit = fields[field]
    method = provenance["method"]
    if method == "registry_goods_train_physical_support_geography_v2" and "registryLoadProfile" in provenance:
        # Registry-form receipts own private measures per declared unit even
        # when an old physicalRows projection omitted an unprinted measure.
        # Project that explicit contract; do not propose replacement numbers.
        rows = provenance["physicalRows"]
        ids = [row["equipmentIdentifier"] for row in rows]
        if len(ids) != len(source) or len(set(ids)) != len(ids) or set(ids) != {c["equipmentIdentifier"] for c in source}:
            raise ValueError("registry allocation identities differ")
        quantities = [row["quantity"] for row in rows]
        if any(type(q) is not int or q <= 0 for q in quantities):
            raise ValueError("registry allocation needs explicit positive counts")
        packages = candidate["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
        if len(packages) != 1 or sum(quantities) != packages[0]["packageQuantity"]:
            raise ValueError("registry allocation counts do not conserve declared packages")
        profile_key = {"grossKg": "unit_gross_kg", "netKg": "unit_net_kg", "volumeM3": "unit_volume_m3"}[field]
        per_unit = Decimal(str(provenance["registryLoadProfile"][profile_key]))
        total = sampled_shipment_fact({"field": measure, "unit": unit}, candidate)["value"]
        if not per_unit.is_finite() or per_unit <= 0 or per_unit * sum(quantities) != total:
            raise ValueError("registry allocation profile does not conserve shipment measure")
        for row, quantity in zip(rows, quantities, strict=True):
            existing = row["measures"].get(measure)
            if existing is not None:
                if _UNITS[existing["unit"]][0] != _UNITS[unit][0]:
                    raise ValueError("registry allocation measure has incompatible units")
                scale = _UNITS[existing["unit"]][1] / _UNITS[unit][1]
                if Decimal(str(existing["value"])) * scale != per_unit * quantity or Decimal(str(existing["total"])) * scale != total:
                    raise ValueError("registry allocation disagrees with retained row measure")
        value = per_unit * quantities[ids.index(spec["sourceEquipmentIdentifier"])]
    elif method == "independent_full_registry_flat_v1":
        rows = provenance["independentDraw"]["allocations"]
        matches = [r for r in rows if r["row"] == index]
        if len(matches) != 1:
            raise ValueError("sampled allocation has no unique independent row")
        value = matches[0][field]
    elif method == "registry_goods_train_physical_support_geography_v2":
        rows = provenance["physicalRows"]
        if len(rows) != len(source):
            raise ValueError("retained allocation inventory differs")
        ids = [r["equipmentIdentifier"] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) != {c["equipmentIdentifier"] for c in source}:
            raise ValueError("retained allocation identities differ")
        total = candidate["physical"]["measures"][measure]
        if _UNITS[total["unit"]][0] != _UNITS[unit][0]:
            raise ValueError("retained allocation total has incompatible units")
        facts = [r["measures"][measure] for r in rows]
        values = [Decimal(str(f["value"])) for f in facts]
        row_total = Decimal(str(total["total"]))
        if facts and Decimal(str(facts[0]["total"])) != row_total:
            # Older receipts retain exact private row masses before rounding
            # the separately printed shipment total. Verify that explicit
            # rounding contract; do not force a private row to absorb its delta.
            if not {"beforeRounding", "quantum"} <= total.keys():
                raise ValueError("retained allocation totals or units disagree")
            unrounded = Decimal(str(total["beforeRounding"]))
            quantum = Decimal(str(total["quantum"]))
            if (not unrounded.is_finite() or not quantum.is_finite() or quantum <= 0
                    or unrounded.quantize(quantum, rounding=ROUND_HALF_UP) != row_total):
                raise ValueError("retained allocation has an invalid total-rounding contract")
            row_total = unrounded
        if any(
            f["unit"] != total["unit"] or Decimal(str(f["total"])) != row_total
            for f in facts
        ):
            raise ValueError("retained allocation totals or units disagree")
        if any(not v.is_finite() or v <= 0 for v in values):
            raise ValueError("retained allocation must be finite and positive")
        if sum(values) != row_total and not has_exact_proportional_total(rows, measure, row_total):
            raise ValueError("retained allocation measures do not conserve their total")
        value = values[ids.index(spec["sourceEquipmentIdentifier"])] * _UNITS[total["unit"]][1] / _UNITS[unit][1]
    else:
        raise ValueError("unsupported sampled allocation method")
    return {"value": value, "unit": unit}


def sampled_equipment(spec, candidate, current):
    """Read the explicitly selected equipment authority, never a source-size guess."""
    index = spec["index"]
    public = current["target"]["containerInformation"]
    if type(index) is not int or not 0 <= index < len(public):
        raise ValueError("new equipment owner has an invalid inventory index")
    authority = spec.get("authority", "public_target")
    if authority == "public_target":
        return public[index]
    if authority not in ("independent_plan", "sampled_inventory"):
        raise ValueError("unknown equipment authority")
    if authority == "sampled_inventory" and "migrationMissingEquipment" in candidate:
        from document_ocr.synthesis.paddle_equipment_completion import completed_equipment_inventory

        return completed_equipment_inventory(candidate, current)[index]
    scenario = candidate["scenario"]
    if authority == "sampled_inventory":
        method = scenario["provenance"]["method"]
        if method == "registry_goods_train_physical_support_geography_v2":
            retained = candidate["target"]["documentPatch"]["containerInformation"]
            if [r["equipmentIdentifier"] for r in retained] != [r["equipmentIdentifier"] for r in public]:
                raise ValueError("retained equipment inventory ordering differs")
            for key in ("sizeCategory", "typeCategory"):
                value = retained[index].get(key)
                if not isinstance(value, str) or not value or public[index].get(key) != value:
                    raise ValueError("retained equipment needs its explicit matching classification")
            return public[index]
        if method != "independent_full_registry_flat_v1":
            raise ValueError("unsupported sampled equipment method")
    planned = scenario["provenance"]["independentDraw"]["equipment"]
    source = scenario["cargo"]["containerInformation"]
    receipt = candidate["target"]["documentPatch"]["containerInformation"]
    if not len(planned) == len(source) == len(public) == len(receipt):
        raise ValueError("independent equipment inventory count differs")
    if [c["equipmentIdentifier"] for c in receipt] != [c["equipmentIdentifier"] for c in public]:
        raise ValueError("independent equipment inventory ordering differs")
    selected = planned[index]
    for key in ("sizeCategory", "typeCategory"):
        if not isinstance(selected.get(key), str) or not selected[key]:
            raise ValueError("independent equipment lacks a complete semantic pair")
        if key in public[index] and public[index][key] != selected[key]:
            raise ValueError("independent equipment conflicts with the public target")
    return {**public[index], **selected}


def sampled_equipment_groups(candidate, current):
    """Read a printed mix without inventing a type-to-container assignment."""
    public = current["target"]["containerInformation"]
    if not public or len({c["equipmentIdentifier"] for c in public}) != len(public):
        raise ValueError("sampled equipment summary needs a distinct inventory")
    provenance = candidate["scenario"]["provenance"]
    if provenance["method"] == "independent_full_registry_flat_v1":
        counts = {}
        for i in range(len(public)):
            row = sampled_equipment({"index": i, "authority": "independent_plan"}, candidate, current)
            pair = (row["sizeCategory"], row["typeCategory"])
            counts[pair] = counts.get(pair, 0) + 1
        return [{"sizeCategory": s, "typeCategory": t, "count": n} for (s, t), n in counts.items()]
    if provenance["method"] != "registry_goods_train_physical_support_geography_v2":
        raise ValueError("unsupported sampled equipment-summary method")
    certificate = provenance["equipmentMix"]["certificate"]
    if certificate["source_sha256"] != candidate["proof"]["sourceSha256"]:
        raise ValueError("equipment mix belongs to another source")
    groups = certificate["members"]
    pairs = [(g["sizeCategory"], g["typeCategory"]) for g in groups]
    if (not groups or len(set(pairs)) != len(pairs)
            or any(type(g["count"]) is not int or g["count"] <= 0 for g in groups)
            or sum(g["count"] for g in groups) != len(public)):
        raise ValueError("equipment mix does not conserve the distinct inventory")
    # The certificate is summary-only. Existing per-ID labels must not conflict,
    # but a valid summary does not create any new per-ID labels.
    remaining = {pair: g["count"] for pair, g in zip(pairs, groups, strict=True)}
    for row in public:
        if "sizeCategory" in row and "typeCategory" in row:
            pair = (row["sizeCategory"], row["typeCategory"])
            if remaining.get(pair, 0) <= 0:
                raise ValueError("equipment mix conflicts with printed per-ID classifications")
            remaining[pair] -= 1
        elif "sizeCategory" in row or "typeCategory" in row:
            raise ValueError("partial per-ID classification requires explicit summary adjudication")
    return groups


def sampled_allocation_quantity(index, candidate, current):
    """Project an explicitly sampled count, never derive a missing real count."""
    public = current["target"]["containerInformation"]
    receipt = candidate["target"]["documentPatch"]["containerInformation"]
    source = candidate["scenario"]["cargo"]["containerInformation"]
    provenance = candidate["scenario"]["provenance"]
    if type(index) is not int or not 0 <= index < len(public):
        raise ValueError("sampled allocation has an invalid inventory index")
    if len(source) != len(public) or [c["equipmentIdentifier"] for c in public] != [c["equipmentIdentifier"] for c in receipt]:
        raise ValueError("sampled allocation inventory differs")
    method = provenance["method"]
    if method == "independent_full_registry_flat_v1":
        loads = provenance["independentDraw"]["allocations"]
        if [r["row"] for r in loads] != list(range(len(source))) or any(type(r["row"]) is not int for r in loads):
            raise ValueError("sampled allocation needs complete ordered load ownership")
        quantities = [r["packageQuantity"] for r in loads]
    elif method == "registry_goods_train_physical_support_geography_v2":
        rows = provenance["physicalRows"]
        ids = [r["equipmentIdentifier"] for r in rows]
        source_ids = [c["equipmentIdentifier"] for c in source]
        if len(ids) != len(source_ids) or len(set(ids)) != len(ids) or set(ids) != set(source_ids):
            raise ValueError("retained package-count inventory differs")
        by_id = {r["equipmentIdentifier"]: r["quantity"] for r in rows}
        quantities = [by_id[i] for i in source_ids]
    else:
        raise ValueError("unsupported sampled package-count method")
    if any(type(q) is not int or q <= 0 for q in quantities):
        raise ValueError("sampled allocation needs explicit positive integer counts")
    goods = current["target"]["goodsItemDetails"]
    if len(goods) != 1 or sum(quantities) != sum(r["packageQuantity"] for r in goods[0]["numberAndTypeOfPackages"]):
        raise ValueError("sampled allocation does not conserve declared packages")
    existing = {r["equipmentIdentifier"]: r["packageQuantity"] for r in goods[0].get("splitGoodsPlacement", []) if "packageQuantity" in r}
    for c, quantity in zip(public, quantities, strict=True):
        if c["equipmentIdentifier"] in existing and existing[c["equipmentIdentifier"]] != quantity:
            raise ValueError("sampled count conflicts with retained printed allocation")
    return quantities[index]


def sampled_unit_mass_groups(candidate, current):
    """Read counted-unit masses from the shipment's explicit packing authority.

    A single declared group uses its complete private ledger; disjoint public
    rows use their ordered group ledger. Neither branch estimates missing mass.
    """
    provenance = candidate["scenario"]["provenance"]
    public = current["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
    if "declaredPackages" in provenance:
        ledger = provenance["declaredPackages"]
        if ledger["rows"] != public:
            raise ValueError("unit-mass groups differ from retained public packages")
        groups = ledger["groups"]
        total = Decimal(ledger["totals"]["netKg"])
    elif "freshPrivatePacking" in provenance:
        ledger = provenance["freshPrivatePacking"]
        if public != [
            {"packageQuantity": ledger["publicQuantity"], "typeCategory": ledger["countedCategory"]}
        ]:
            raise ValueError("single-unit ledger differs from retained public packages")
        groups = [{"category": ledger["countedCategory"], "netKg": ledger["unitNetKg"]}]
        total = Decimal(ledger["netKg"])
    elif (
        provenance.get("method") == "registry_goods_train_physical_support_geography_v2"
        and provenance.get("physicalProfile") == "registry_form"
    ):
        # Retained registry-form receipts carry the original sampled unit fact.
        # Read that fact, not a new estimate or a quotient of rounded labels.
        profile = provenance["registryLoadProfile"]
        if len(public) != 1 or public[0]["typeCategory"] != profile["package_category"]:
            raise ValueError("retained unit profile differs from public packages")
        fact = provenance["explicitPhysicalFacts"]["netWeight"]
        if fact["unit"] != "kilogram":
            raise ValueError("retained unit profile requires an explicit kilogram authority")
        groups = [{"category": profile["package_category"], "netKg": profile["unit_net_kg"]}]
        total = Decimal(fact["value"])
    else:
        raise ValueError("no supported explicit counted-unit mass authority")
    if not public or len(groups) != len(public):
        raise ValueError("unit-mass groups require one explicit mass per public row")
    result, mass = [], Decimal(0)
    for group, row in zip(groups, public, strict=True):
        value = Decimal(group["netKg"])
        quantity = row["packageQuantity"]
        if (
            group["category"] != row["typeCategory"]
            or type(quantity) is not int
            or quantity <= 0
            or not value.is_finite()
            or value <= 0
        ):
            raise ValueError("unit-mass group lacks consistent category/count/positive mass")
        mass += value * quantity
        result.append((group["category"], value))
    if not total.is_finite() or mass != total:
        raise ValueError("unit-mass groups do not conserve the sampled net mass")
    return result


def sampled_source_slice(spec, edits):
    """Repeat an owned source passage across compound or split receipt edits.

    Only literal separators may remain unsampled. Every lexical byte must have
    an exact, non-overlapping owner; crossing edits or stale source text fail.
    """
    start, end, original = spec["byteStart"], spec["byteEnd"], spec["source"].encode()
    if type(start) is not int or type(end) is not int or start < 0 or end - start != len(original):
        raise ValueError("source repetition requires its exact byte interval")
    selected = sorted(
        (e for e in edits if e["byteStart"] < end and e["byteEnd"] > start),
        key=lambda e: e["byteStart"],
    )
    if not selected:
        raise ValueError("source repetition has no sampled owners")
    cursor, parts = start, []
    for edit in selected:
        lo, hi = edit["byteStart"], edit["byteEnd"]
        if not cursor <= lo < hi <= end:
            raise ValueError("source repetition has crossing or overlapping owners")
        if original[lo - start : hi - start] != edit["before"].encode():
            raise ValueError("source repetition owner differs from reviewed source")
        separator = original[cursor - start : lo - start].decode()
        if any(c.isalnum() for c in separator):
            raise ValueError("source repetition retains unsampled lexical text")
        parts.extend((separator, edit["after"]))
        cursor = hi
    suffix = original[cursor - start :].decode()
    if any(c.isalnum() for c in suffix):
        raise ValueError("source repetition retains unsampled lexical text")
    return "".join((*parts, suffix))


def project_sampled_lines(value, line, *, stride=None):
    """Project a repeated presentation row from every independently sampled group."""
    lines = value.splitlines()
    if type(line) is not int or line < 0:
        raise ValueError("sampled line index must be a nonnegative integer")
    if stride is None:
        if line >= len(lines):
            raise ValueError("sampled line projection is outside its owner")
        return lines[line]
    if type(stride) is not int or stride <= line or not lines or len(lines) % stride:
        raise ValueError("sampled group lines do not match the reviewed presentation stride")
    return "\n".join(lines[line::stride])


def extra_value(
    extra,
    candidate,
    current,
    identity,
    equipment_registry,
    package_vocabulary=None,
    wording=None,
    *,
    project_root,
):
    """Evaluate one reviewed new owner, without inventing missing shipment facts."""
    if "sourceSlice" in extra:
        value = sampled_source_slice(extra["sourceSlice"], candidate["proof"]["edits"])
        if "line" in extra:
            value = project_sampled_lines(value, extra["line"], stride=extra.get("lineStride"))
        return value, None
    if "omission" in extra:
        decision = extra["omission"]
        if (
            not isinstance(decision, dict)
            or set(decision) != {"reason", "authority"}
            or any(not isinstance(v, str) or not v.strip() for v in decision.values())
            or "target" in extra
            or "targets" in extra
        ):
            raise ValueError("native omission requires a reviewed reason and authority, no target")
        return "", None
    if "numericPresentation" in extra:
        spec = extra["numericPresentation"]
        value, _ = extra_value(
            {"owner": extra["owner"], "fromKey": spec["key"], "fromByteStart": spec["byteStart"]},
            candidate,
            current,
            identity,
            equipment_registry,
            package_vocabulary,
            wording,
            project_root=project_root,
        )
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
            raise ValueError("numeric presentation requires an unformatted sampled scalar")
        return format_measure(spec["surface"], Decimal(spec["baseline"]), Decimal(value)), None
    if "equipmentTareRemainder" in extra:
        # Paddle can expose the second equipment tare after the old transcript
        # exposed only the first and their aggregate. Reuse that complete sampled
        # ledger; copying the first tare would be wrong for mixed equipment.
        spec = extra["equipmentTareRemainder"]
        keys = [spec["totalSurface"], *spec["partSurfaces"]]
        if len(keys) < 2 or any(not isinstance(k, str) or not k for k in keys):
            raise ValueError("tare remainder requires named total and component surfaces")
        if len(keys) != len(set(keys)):
            raise ValueError("tare remainder requires distinct total and component owners")
        values = []
        for key in keys:
            value, _ = extra_value(
                {"owner": extra["owner"], "surfaceKey": key},
                candidate,
                current,
                identity,
                equipment_registry,
                package_vocabulary,
                wording,
                project_root=project_root,
            )
            if not isinstance(value, str) or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None:
                raise ValueError("tare remainder requires unformatted kilogram scalars")
            value = Decimal(value)
            if not value.is_finite() or value <= 0:
                raise ValueError("tare remainder requires positive finite kilogram scalars")
            values.append(value)
        remainder = values[0] - sum(values[1:], Decimal(0))
        if remainder <= 0:
            raise ValueError("tare remainder must be positive and conserve the aggregate")
        return format(remainder, "f"), None
    if "allocation" in extra:
        index = extra["allocation"]["index"]
        authority = extra["allocation"].get("authority", "independent_plan")
        if authority == "sampled_allocation":
            value = sampled_allocation_quantity(index, candidate, current)
            return str(value), value
        if authority == "public_target":
            # A repeated printed count can use an already extracted allocation.
            # This does not manufacture the absent allocations that require the
            # independent private ledger below.
            values = []
            for target in (current["target"], candidate["target"]["documentPatch"]):
                inventory = target["containerInformation"]
                if type(index) is not int or not 0 <= index < len(inventory):
                    raise ValueError("public allocation has an invalid inventory index")
                equipment_id = inventory[index]["equipmentIdentifier"]
                goods = target["goodsItemDetails"]
                if len(goods) != 1:
                    raise ValueError("public allocation repetition requires one goods item")
                matches = [
                    row
                    for row in goods[0]["splitGoodsPlacement"]
                    if row["equipmentIdentifier"] == equipment_id
                ]
                if len(matches) != 1:
                    raise ValueError("public allocation repetition lacks one explicit allocation")
                quantity = matches[0]["packageQuantity"]
                if type(quantity) is not int or quantity <= 0:
                    raise ValueError("public allocation repetition needs a positive integer")
                values.append((equipment_id, quantity))
            if values[0] != values[1]:
                raise ValueError("public allocation differs from the retained shipment receipt")
            return str(values[0][1]), values[0][1]
        if authority != "independent_plan":
            raise ValueError("unknown allocation authority")
        public = current["target"]["containerInformation"]
        receipt = candidate["target"]["documentPatch"]["containerInformation"]
        scenario = candidate["scenario"]
        source = scenario["cargo"]["containerInformation"]
        loads = scenario["provenance"]["independentDraw"]["allocations"]
        if type(index) is not int or not 0 <= index < len(public):
            raise ValueError("new allocation has an invalid inventory index")
        if not len(public) == len(receipt) == len(source) == len(loads):
            raise ValueError("new allocation inventory count differs")
        if [c["equipmentIdentifier"] for c in public] != [
            c["equipmentIdentifier"] for c in receipt
        ]:
            raise ValueError("new allocation inventory ordering differs")
        if any(type(load["row"]) is not int for load in loads) or [
            load["row"] for load in loads
        ] != list(range(len(public))):
            raise ValueError("new allocation requires complete ordered load ownership")
        quantities = [load["packageQuantity"] for load in loads]
        if any(type(q) is not int or q <= 0 for q in quantities):
            raise ValueError("new allocation requires positive integer counts")
        goods = current["target"]["goodsItemDetails"]
        if len(goods) != 1 or sum(quantities) != sum(
            row["packageQuantity"] for row in goods[0]["numberAndTypeOfPackages"]
        ):
            raise ValueError("new allocation does not conserve declared packages")
        return str(quantities[index]), quantities[index]
    if "literalText" in extra:
        value = extra["literalText"]
        if not isinstance(value, str) or not value:
            raise ValueError("reviewed caption must be nonempty text")
        return value, None
    if "packingSummary" in extra:
        spec = extra["packingSummary"]
        ledger = candidate["scenario"]["provenance"]["freshPrivatePacking"]
        groups = (
            ledger["facts"]["declaredGroupLedgers"]
            if ledger["relation"] == "DISJOINT_GROUPS"
            else [ledger]
        )
        if not groups or any(g["relation"] != "MIXED" for g in groups):
            raise ValueError("mixed-packing repetition requires explicit mixed ledgers")
        facts = [g["facts"]["mixedAccounting"] for g in groups]
        for group, fact in zip(groups, facts, strict=True):
            fields = ("supportQuantity", "packedQuantity", "looseQuantity", "totalRelatedQuantity")
            if any(type(fact[f]) is not int or fact[f] <= 0 for f in fields):
                raise ValueError("mixed-packing counts must be positive integers")
            if (
                fact["supportQuantity"] + fact["looseQuantity"] != group["publicQuantity"]
                or fact["packedQuantity"] + fact["looseQuantity"] != fact["totalRelatedQuantity"]
                or group["relatedQuantity"] != fact["totalRelatedQuantity"]
            ):
                raise ValueError("mixed-packing repetition violates ledger conservation")
        if sum(g["publicQuantity"] for g in groups) != ledger["publicQuantity"]:
            raise ValueError("mixed-packing group totals differ from shipment ledger")
        field = spec["field"]
        if field in {"supportQuantity", "looseQuantity", "totalRelatedQuantity"}:
            return str(sum(f[field] for f in facts)), None
        if field != "relatedType" or package_vocabulary is None:
            raise ValueError("unknown mixed-packing summary or missing alias vocabulary")
        categories = {g["relatedCategory"] for g in groups}
        if len(categories) != 1:
            raise ValueError("one shortened packing type cannot represent different categories")
        noun, _ = package_vocabulary.choose(
            categories.pop(),
            (spec["style"],),
            {spec["style"]: 1.0},
            identity + ":" + extra["owner"],
        )
        return noun, None
    if "nestedPacking" in extra:
        spec = extra["nestedPacking"]
        styles = (spec["style"], *spec.get("additionalStyles", []))
        if len(set(styles)) != len(styles) or any(s not in {"full", "short", "optional"} for s in styles):
            raise ValueError("nested packing requires distinct reviewed presentation styles")
        style_weights = dict.fromkeys(styles, 1.0)
        ledger = candidate["scenario"]["provenance"]["freshPrivatePacking"]
        groups = (
            ledger["facts"]["declaredGroupLedgers"]
            if ledger["relation"] == "DISJOINT_GROUPS"
            else [ledger]
        )
        public = current["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
        provenance = candidate["scenario"]["provenance"]
        if (
            len(public) > 1
            and ledger["relation"] != "DISJOINT_GROUPS"
            and provenance.get("declaredPackageContract", {}).get("planning") == "shared_units"
        ):
            # Shared-unit public rows partition one physical shipment. Its
            # support/inner-count statement still describes that whole shipment;
            # it does not establish a per-group division of supporting pallets.
            partition = provenance["declaredPackages"]
            if (
                partition["rows"] != public
                or any(row["typeCategory"] != ledger["countedCategory"] for row in public)
                or any(type(row["packageQuantity"]) is not int or row["packageQuantity"] <= 0 for row in public)
                or sum(row["packageQuantity"] for row in public) != ledger["publicQuantity"]
            ):
                raise ValueError("shared-unit nested presentation differs from declared partition")
            public = [{"packageQuantity": ledger["publicQuantity"], "typeCategory": ledger["countedCategory"]}]
        if len(groups) != len(public) or package_vocabulary is None:
            raise ValueError("nested presentation requires exact public group and alias authority")
        lines = []
        depths = {group["facts"]["externalPackingPresentation"]["levels"] for group in groups}
        if len(depths) != 1:
            raise ValueError("one native declaration cannot mix packing presentation depths")
        for i, (group, row) in enumerate(zip(groups, public, strict=True)):
            fact = group["facts"]["externalPackingPresentation"]
            if (
                fact["owner"] != extra["owner"]
                or fact["nativeFingerprint"] != spec["nativeFingerprint"]
                or group["publicQuantity"] != row["packageQuantity"]
                or group["countedCategory"] != row["typeCategory"]
                or fact["relatedQuantity"] != group["relatedQuantity"]
                or fact["relatedCategory"] != group["relatedCategory"]
                or type(fact["relatedQuantity"]) is not int
                or fact["relatedQuantity"] <= 0
            ):
                raise ValueError(
                    "nested packing presentation differs from complete sampled authority"
                )
            noun, _ = package_vocabulary.choose(
                group["relatedCategory"],
                styles,
                style_weights,
                identity + ":" + extra["owner"] + ":" + str(i),
            )
            if fact["levels"] == 2:
                support = group["facts"].get("supportEnvelope")
                if (
                    fact["contentsKind"] != "support_packages"
                    or group["relation"] != "ON_SUPPORT"
                    or support is None
                    or support["quantity"] != fact["relatedQuantity"]
                ):
                    raise ValueError("native support declaration lacks complete support authority")
                counted, _ = package_vocabulary.choose(
                    row["typeCategory"],
                    styles,
                    style_weights,
                    identity + ":" + extra["owner"] + ":counted:" + str(i),
                )
                lines.append(
                    f"{row['packageQuantity']} {counted} ON {fact['relatedQuantity']} {noun}"
                )
            else:
                contents = fact["contents"]
                if (
                    fact["levels"] != 3
                    or fact["contentsKind"] != "goods_pieces"
                    or group["relation"] not in {"CONTAINS", "SUPPORTS"}
                    or contents["packageSingular"] != "PIECE"
                    or contents["packagePlural"] != "PIECES"
                    or any(
                        type(contents[k]) is not int or contents[k] <= 0
                        for k in ("unitsPerRelated", "quantity")
                    )
                    or contents["quantity"] != fact["relatedQuantity"] * contents["unitsPerRelated"]
                ):
                    raise ValueError(
                        "nested packing presentation differs from complete sampled authority"
                    )
                lines.append(f"{fact['relatedQuantity']} {noun}={contents['quantity']}PCS")
        return ("QTY:" if depths == {3} else "") + "; ".join(lines), None
    if "parts" in extra:
        parts = extra["parts"]
        if len(parts) < 2:
            raise ValueError("composed reference requires at least two owned parts")
        values = []
        for index, part in enumerate(parts):
            if "parts" in part:
                raise ValueError("composed reference parts cannot be nested")
            value, _ = extra_value(
                {**part, "owner": f"{extra['owner']}:part:{index}"},
                candidate,
                current,
                identity,
                equipment_registry,
                package_vocabulary,
                wording,
                project_root=project_root,
            )
            if not isinstance(value, str) or not value:
                raise ValueError("composed reference part must be nonempty text")
            values.append(value)
        return "".join(values), None
    if "fromKey" in extra:
        values = {
            e["after"]
            for e in candidate["proof"]["edits"]
            if e["key"] == extra["fromKey"]
            and ("fromByteStart" not in extra or e["byteStart"] == extra["fromByteStart"])
        }
        if len(values) != 1:
            raise ValueError(f"extra needs one exact shared value: {extra['owner']}")
        value = values.pop()
        if "emailPart" in extra:
            if "line" in extra or "segments" in extra:
                raise ValueError("email component cannot also select lines or character segments")
            local, separator, domain = value.partition("@")
            if not local or not separator or not domain or "@" in domain:
                raise ValueError("email component requires one complete sampled mailbox")
            if extra["emailPart"] not in {"local", "domain"}:
                raise ValueError("unknown email component")
            value = local if extra["emailPart"] == "local" else domain
        if "line" in extra:
            value = project_sampled_lines(value, extra["line"], stride=extra.get("lineStride"))
        if "segments" in extra:
            pieces = []
            for start, end in extra["segments"]:
                end = len(value) if end is None else end
                if not 0 <= start < end <= len(value):
                    raise ValueError("dependent reference segment is outside its parent")
                pieces.append(value[start:end])
            value = extra["join"].join(pieces)
        return value, None
    if "surfaceKey" in extra:
        value = candidate["surfaceValues"][extra["surfaceKey"]]
        if "surfaceIndex" in extra:
            # A scalar receipt has exactly one occurrence; indexing it must not
            # index its characters or invent additional repeated occurrences.
            values = [value] if isinstance(value, str) else value
            index = extra["surfaceIndex"]
            if (
                not isinstance(values, list)
                or type(index) is not int
                or not 0 <= index < len(values)
                or any(not isinstance(v, str) or not v for v in values)
            ):
                raise ValueError("indexed surface requires a valid owned occurrence")
            return values[index], None
        values = {value} if isinstance(value, str) else set(value)
        if len(values) != 1:
            raise ValueError("a repeated surface does not have one shared sampled value")
        return values.pop(), None
    if "wording" in extra:
        if wording is None or extra["wording"] not in wording:
            raise ValueError("new linguistic owner lacks reviewed wording")
        value = wording[extra["wording"]]
        semantic = " ".join(value.split())
        if "emailPart" in extra:
            local, separator, domain = value.partition("@")
            if (
                not local
                or not separator
                or not domain
                or "@" in domain
                or any(c.isspace() for c in value)
            ):
                raise ValueError("wording email component requires one complete sampled mailbox")
            if extra["emailPart"] not in {"local", "domain"}:
                raise ValueError("unknown wording email component")
            value = local if extra["emailPart"] == "local" else domain
        # Adjacent native owners can have no intervening whitespace (for
        # example a parenthesized company immediately followed by its address).
        # The reviewed presentation owns the separator, not the linguistic model.
        separator = extra.get("separatorBefore", "")
        if not isinstance(separator, str) or (separator and not separator.isspace()):
            raise ValueError("wording separator must contain only whitespace")
        return (
            separator + value,
            semantic.upper() if extra.get("targetCase") == "upper" else semantic,
        )
    stream = DeterministicStream(seed=20261102, namespace="r30-paddle-pilot", identity=identity)
    if "newContainer" in extra:
        source = extra["newContainer"]["source"]
        if re.fullmatch(r"[A-Z]{3}[UJZ][0-9]{7}", source) is None:
            raise ValueError("new container requires its exact source identity")
        excluded = {source} | {
            row["equipmentIdentifier"] for row in current["target"].get("containerInformation", [])
        }
        value = generate_container_number(
            owner_and_category=source[:4],
            stream=stream.derive("native-container:" + source),
            excluded=excluded,
        )
        return value, value
    if "dgFact" in extra:
        spec = extra["dgFact"]
        facts = candidate["scenario"]["provenance"]["dgPrintedFacts"]
        public = current["target"]["goodsItemDetails"][0]["dangerousGoods"]
        if len(public) != 1 or facts["unNumber"] != public[0]["unNumber"]:
            raise ValueError("new DG fact requires one matching sampled identity")
        field = spec["field"]
        if field == "class":
            value = facts[field]
            if re.fullmatch(r"[1-9](?:\.[1-6])?[A-Z]?", value) is None:
                raise ValueError("new DG class requires an explicit valid class")
            return value, _broad_hazard(value)
        if field == "packingGroup":
            value = facts[field]
            if value is None and spec.get("omitAbsent") is True:
                if "packingGroupCategory" in public[0]:
                    raise ValueError("absent sampled packing group conflicts with public target")
                return "", None
            if value not in _PACKING_GROUP:
                raise ValueError("new DG packing group is absent or unsupported")
            return value, _PACKING_GROUP[value]
        raise ValueError("unsupported new DG fact")
    if "hsCode" in extra:
        spec = extra["hsCode"]
        if "fromKey" in spec:
            values = {
                e["after"] for e in candidate["proof"]["edits"] if e["key"] == spec["fromKey"]
            }
            if len(values) != 1:
                raise ValueError("new HS presentation requires one sampled identity")
            hs6 = hs_digits(values.pop())[:6]
        else:
            hs6 = candidate["scenario"]["goods_identities"][spec["goodsIdentity"]]["hs6"]
        digits = render_synthetic_hs_code(
            hs6=hs6,
            output_digits=len(hs_digits(spec["source"])),
            stream=stream.derive("new-hs:" + extra["owner"]),
        )
        value = render_source_hs(spec["source"], digits)
        return value, value
    if "vessel" in extra:
        from document_ocr.synthesis.curated_campaign import vessel_name_fit_exclusion_reason

        spec = extra["vessel"]
        names = spec["support"]["names"]
        if names != sorted(set(names)) or any(vessel_name_fit_exclusion_reason(n) for n in names):
            raise ValueError("new vessel requires reviewed distinct valid name support")
        eligible = [n for n in names if n != spec["source"]]
        if not eligible or not spec["support"]["authority"]:
            raise ValueError("new vessel lacks train-only support provenance")
        value = eligible[stream.derive("new-vessel-name").randbelow(len(eligible))]
        return value, value.upper()
    if "receiptEquipmentTare" in extra:
        spec = extra["receiptEquipmentTare"]
        index = spec["index"]
        if type(index) is not int or index < 0:
            raise ValueError("receipt equipment tare needs a nonnegative equipment index")
        rows = [row for row in candidate["physical"]["tares"] if row["owner"] == index]
        if len(rows) != 1 or not rows[0]["observationSources"]:
            raise ValueError("receipt equipment tare requires one provenance-backed owner")
        container = sampled_equipment(
            {"index": index, "authority": "public_target"}, candidate, current
        )
        pair = container["sizeCategory"] + "|" + container["typeCategory"]
        if rows[0]["pair"] != pair:
            raise ValueError("receipt equipment tare belongs to different equipment")
        value = Decimal(rows[0]["kg"])
        if not value.is_finite() or value <= 0:
            raise ValueError("receipt equipment tare must be positive and finite")
        places = max(spec["minPlaces"], -value.normalize().as_tuple().exponent)
        return format(value, f",.{places}f" if spec.get("grouped") else f".{places}f"), None
    if "equipmentTare" in extra:
        spec = extra["equipmentTare"]
        is_sum = "indices" in spec
        if is_sum and ("index" in spec or "aggregate" in spec):
            raise ValueError("computed tare sum cannot also select a row or fixed aggregate")
        index = None if is_sum else spec["index"]
        aggregate = spec.get("aggregate")
        owners = (
            spec["indices"]
            if is_sum
            else aggregate["indices"]
            if aggregate is not None
            else [index]
        )
        if (
            not owners
            or any(type(i) is not int for i in owners)
            or owners != sorted(set(owners))
            or (not is_sum and index not in owners)
        ):
            raise ValueError("tare group requires ordered distinct equipment owners")
        # The overlay carries reviewed observations/specifications and their provenance.
        # No cargo-donor value or unsupported equipment fallback is permitted.
        domains, pairs = [], {}
        for owner in owners:
            container = sampled_equipment({**spec, "index": owner}, candidate, current)
            pair = container["sizeCategory"] + "|" + container["typeCategory"]
            observations = spec["support"]["pairs"].get(pair, [])
            values = tuple(sorted({Decimal(row["kg"]) for row in observations}))
            if not values or any(not v.is_finite() or v <= 0 for v in values):
                raise ValueError("new equipment tare lacks positive finite pair support")
            if any(not row["sourceIds"] for row in observations):
                raise ValueError("new equipment tare lacks observation provenance")
            if (
                spec.get("authority") == "sampled_inventory"
                and "migrationMissingEquipment" in candidate
                and candidate["physical"]["tares"]
            ):
                # Completion already selected a coherent witness for all retained
                # tare rows/aggregates. Re-drawing here could contradict that sum.
                value = Decimal(candidate["migrationMissingEquipment"]["tareWitnessKg"][str(owner)])
                if value not in values:
                    raise ValueError("completed tare lacks native presentation support")
                values = (value,)
            domains.append((owner, ((pair, values),)))
            pairs[owner] = frozenset((pair,))
        total = None
        if aggregate is not None:
            selectors = set(aggregate) & {"fromKey", "surfaceKey"}
            if len(selectors) != 1 or set(aggregate) != selectors | {"indices"}:
                raise ValueError("tare aggregate requires one explicit sampled-value authority")
            selector = next(iter(selectors))
            total = Decimal(
                extra_value(
                    {"owner": extra["owner"], selector: aggregate[selector]},
                    candidate,
                    current,
                    identity,
                    equipment_registry,
                    package_vocabulary,
                    wording,
                    project_root=project_root,
                )[0]
            )
            if not total.is_finite() or total <= 0:
                raise ValueError("tare aggregate must be a positive finite kilogram scalar")
        contract = TareConstraints(tuple(domains), total)
        sampled, _ = contract.sample(pairs, stream.derive("new-equipment-tare"))
        value = sum(sampled.values(), Decimal(0)) if is_sum else sampled[index]
        if "presentation" in spec:
            presentation = spec["presentation"]
            return format_measure(
                presentation["surface"], Decimal(presentation["baseline"]), value
            ), None
        places = max(spec["minPlaces"], -value.normalize().as_tuple().exponent)
        return format(value, f",.{places}f" if spec.get("grouped") else f".{places}f"), None
    if "phone" in extra:
        spec = extra["phone"]
        country = get(candidate["scenario"], spec["countryPath"])
        generator = {"phone": phone, "fax": fax}[spec["kind"]]
        number = generator(stream.derive(spec["identity"]), country_code=country)
        presentation = format_phone_presentation(
            source_value=spec["source"],
            source_surfaces=[spec["source"]],
            replacement_identity=number,
            target_country_code=country,
            source_country_code=spec["sourceCountry"],
            family="source",
        )
        return presentation.surfaces[0], presentation.target_value
    if "imo" in extra:
        spec = extra["imo"]
        source = spec["source"]
        if re.fullmatch(r"[1-9][0-9]{6}", source) is None:
            raise ValueError("reviewed source IMO must have seven ASCII digits")
        # Native OCR can retain a bad checksum. Such a value cannot collide
        # with this generator's valid identities; do not loosen its contract.
        excluded = [source] if imo_check_digit(source[:6]) == int(source[-1]) else []
        value = sample_imo_number(
            stream=stream.derive("vessel-imo"),
            excluded_values=excluded,
            used_values=set(),
            maximum_attempts=spec["maximumAttempts"],
            leading_digit_weights=spec["leadingDigitWeights"],
        ).value
        return value, value
    if "scenarioText" in extra:
        value = get(candidate["scenario"], extra["scenarioText"])
        if not isinstance(value, str) or not value:
            raise ValueError("scenario text must be an explicit nonempty value")
        return value, value.upper() if extra.get("targetCase") == "upper" else value
    if "countryName" in extra:
        from document_ocr.synthesis.country_presentations import Registry

        spec = extra["countryName"]
        registry = Registry.load(
            project_root, manifest_path=spec["manifestPath"], expected_sha256=spec["sha256"]
        )
        # Planning geography retains ISO display names. Extraction uses the
        # reviewed human-readable vocabulary, not that planning display string.
        value = registry.preferred_name(get(candidate["scenario"], spec["codePath"]))
        return value, value
    if "sourceDate" in extra:
        spec = extra["sourceDate"]
        if "reference" in spec:
            reference = spec["reference"]
            offset = date.fromisoformat(
                get(current["target"], reference["target"])
            ) - date.fromisoformat(reference["source"])
            value = (date.fromisoformat(spec["source"]) + offset).isoformat()
        else:
            if any(k in current["target"] for k in ("issueDate", "shippedOnBoardDate")):
                raise ValueError("new date must share an existing shipment date reference")
            # With no previous public dates, all newly owned dates share one
            # explicit shipment stream. Never redraw dates of an existing cargo refresh.
            value = shifted_public_date(
                spec["source"],
                DeterministicStream(candidate["seed"], "full-current-v7", identity),
            )
        surface = _declared_date_surface(date.fromisoformat(value), spec["format"])
        return surface.upper() if spec.get("uppercase") else surface, value
    if "targetNumber" in extra:
        spec = extra["targetNumber"]
        fact = get(current["target"], spec["path"])
        if fact["unit"] != spec["unit"]:
            raise ValueError("native numeric repetition has incompatible target unit")
        value = Decimal(str(fact["value"]))
        places = max(spec["minPlaces"], -value.normalize().as_tuple().exponent)
        return format(value, f".{places}f"), None
    if "sourceIdentifier" in extra:
        recipe = extra.get("identifierSequence", {})
        if recipe and (
            set(recipe) != {"canonical_identity", "sequence_length", "sequence_offset"}
            or type(recipe["sequence_length"]) is not int
            or type(recipe["sequence_offset"]) is not int
            or not 0 <= recipe["sequence_offset"] < recipe["sequence_length"]
        ):
            raise ValueError("identifier sequence requires an exact base, length and offset")
        value = (
            extra.get("literalPrefix", "")
            + _source_format_identifier(recipe, extra["sourceIdentifier"], stream)
            + extra.get("literalSuffix", "")
        )
        return value, value
    if "allocationPartition" in extra:
        spec = extra["allocationPartition"]
        parts, index = spec["parts"], spec["index"]
        source_rows = candidate["scenario"]["cargo"]["containerInformation"]
        source_indices = [
            i
            for i, row in enumerate(source_rows)
            if row["equipmentIdentifier"] == spec["sourceEquipmentIdentifier"]
        ]
        if len(source_indices) != 1 or type(parts) is not int or parts < 2:
            raise ValueError("native lot partition needs one container and at least two parts")
        if type(index) is not int or not 0 <= index < parts:
            raise ValueError("native lot partition index is outside its declared parts")
        loads = candidate["scenario"]["provenance"]["independentDraw"]["allocations"]
        matches = [row for row in loads if row["row"] == source_indices[0]]
        if len(matches) != 1:
            raise ValueError("native lot partition lacks one sampled load")
        load = matches[0]
        quantity = load["packageQuantity"]
        unit = Decimal(candidate["scenario"]["provenance"]["freshPrivatePacking"]["unitNetKg"])
        if (
            type(quantity) is not int
            or quantity < parts
            or not unit.is_finite()
            or unit <= 0
            or unit * quantity != Decimal(load["netKg"])
        ):
            raise ValueError("native lot partition lacks positive complete counted-unit mass")
        rng = stream.derive("native-lots:" + spec["group"])
        counts, remaining = [], quantity
        for left in range(parts, 1, -1):
            count = 1 + rng.randbelow(remaining - left + 1)
            counts.append(count)
            remaining -= count
        counts.append(remaining)
        value = unit * counts[index]
        return format(value.normalize(), "f"), None
    if "physical" in extra:
        spec = extra["physical"]
        authority = spec.get("authority", "explicit_physical_facts")
        if authority not in (
            "explicit_physical_facts",
            "receipt_measure",
            "public_target",
            "sampled_allocation",
            "sampled_shipment",
        ):
            raise ValueError("unknown physical fact authority")
        if authority == "sampled_allocation":
            fact = sampled_allocation_fact(spec, candidate)
        elif authority == "sampled_shipment":
            if "sourceEquipmentIdentifier" in spec:
                raise ValueError("allocation cannot consume a sampled shipment total")
            fact = sampled_shipment_fact(spec, candidate)
        elif "sourceEquipmentIdentifier" in spec:
            if authority != "explicit_physical_facts":
                raise ValueError("allocation facts cannot use a shipment-total authority")
            containers = candidate["scenario"]["cargo"]["containerInformation"]
            indices = [
                i
                for i, c in enumerate(containers)
                if c["equipmentIdentifier"] == spec["sourceEquipmentIdentifier"]
            ]
            if len(indices) != 1:
                raise ValueError("numeric allocation has no unique source equipment owner")
            loads = candidate["scenario"]["provenance"]["independentDraw"]["allocations"]
            matches = [load for load in loads if load["row"] == indices[0]]
            if len(matches) != 1:
                raise ValueError("numeric allocation has no unique sampled load")
            units = {"netKg": "kilogram", "grossKg": "kilogram", "volumeM3": "cubic_metre"}
            fact = {"value": matches[0][spec["field"]], "unit": units[spec["field"]]}
        elif authority == "public_target":
            fact = current["target"]["goodsItemDetails"][spec["goods"]][spec["field"]]
        elif authority == "receipt_measure":
            # Retained shipments can own a printed mass privately in their
            # generation receipt even when the old public target omitted it.
            # This is an explicit authority selection, never a missing-fact
            # fallback. Use the already rounded total, not a planning estimate.
            measure = candidate["physical"]["measures"][spec["field"]]
            fact = {"value": measure["total"], "unit": measure["unit"]}
        else:
            fact = candidate["scenario"]["provenance"]["explicitPhysicalFacts"][spec["field"]]
        if fact["unit"] != spec["unit"]:
            raise ValueError("new numeric owner has incompatible sampled unit")
        # JSON numbers express decimal text, not a float's binary expansion.
        value = Decimal(str(fact["value"]))
        if not value.is_finite() or value <= 0:
            raise ValueError("physical measure must be finite and positive")
        if "displayPlaces" in spec:
            places = spec["displayPlaces"]
            if type(places) is not int or places < 0 or "renderUnit" in spec:
                raise ValueError("display precision requires nonnegative places and the fact unit")
            value = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
            if value <= 0:
                raise ValueError("display precision erases a positive physical measure")
            if authority == "public_target":
                public = fact
            elif authority == "sampled_shipment":
                public = current["target"]["goodsItemDetails"][0].get(spec["field"])
            else:
                public = None
            if public is not None and public["unit"] == fact["unit"] and Decimal(str(public["value"])) != value:
                raise ValueError("display precision contradicts the retained public shipment measure")
        if "renderUnit" in spec:
            render_unit = spec["renderUnit"]
            source_unit = fact["unit"]
            if _UNITS[source_unit][0] != _UNITS[render_unit][0]:
                raise ValueError("converted owner changes measurement dimension")
            # Independently printed converted aliases retain their source precision.
            # A target projected from this occurrence must use its printed value
            # and unit, without mutating the authoritative planning fact.
            converted = value * _UNITS[source_unit][1] / _UNITS[render_unit][1]
            value = converted.quantize(
                Decimal(1).scaleb(-spec["minPlaces"]), rounding=ROUND_HALF_UP
            )
            rendered = format(value, "f")
        else:
            # Retain source minimum precision and any additional sampled precision.
            # Rounding a repeated mass here would contradict its other declarations.
            places = spec.get("displayPlaces", max(spec["minPlaces"], -value.normalize().as_tuple().exponent))
            rendered = format(value, f",.{places}f" if spec["grouped"] else f".{places}f")
        if "decimalSeparator" in spec:
            if spec["decimalSeparator"] not in (".", ","):
                raise ValueError("numeric owner requires a reviewed decimal separator")
            if spec["decimalSeparator"] == ",":
                rendered = rendered.translate(str.maketrans(".,", ",."))
        return spec["prefix"] + rendered + spec["suffix"], {
            "value": float(value),
            "unit": spec.get("renderUnit", fact["unit"]),
        }
    if "equipment" in extra:
        spec = extra["equipment"]
        container = sampled_equipment(spec, candidate, current)
        alias = equipment_registry.select(
            container["sizeCategory"],
            container["typeCategory"],
            stream.derive("new-equipment:" + str(spec["index"])),
        )
        return preserve_non_operating_reefer(spec["source"], alias, container), {
            key: container[key] for key in ("sizeCategory", "typeCategory")
        }
    if "equipmentSummary" in extra:
        spec = extra["equipmentSummary"]
        containers = current["target"]["containerInformation"]
        if not containers or len({c["equipmentIdentifier"] for c in containers}) != len(containers):
            raise ValueError("equipment summary needs a nonempty distinct inventory")
        groups = {}
        if spec.get("authority") == "recorded_inventory_summary":
            members = sampled_equipment_groups(candidate, current)
        else:
            members = [{**sampled_equipment({**spec, "index": i}, candidate, current), "count": 1}
                       for i in range(len(containers))]
        for container in members:
            pair = (container["sizeCategory"], container["typeCategory"])
            alias = equipment_registry.select(*pair, stream.derive("summary:" + "|".join(pair)))
            alias = preserve_non_operating_reefer(spec["source"], alias, container)
            groups[alias] = groups.get(alias, 0) + container["count"]
        return " + ".join(f"{count} X {alias}" for alias, count in groups.items()), None
    if "packageRows" in extra:
        spec = extra["packageRows"]
        rows = current["target"]["goodsItemDetails"][spec["goods"]]["numberAndTypeOfPackages"]
        if package_vocabulary is None:
            raise ValueError("new package declaration requires its pinned presentation vocabulary")
        parts = []
        for row in rows:
            noun, _ = package_vocabulary.choose(
                row["typeCategory"],
                (spec["style"],),
                {spec["style"]: 1.0},
                identity + ":" + extra["owner"] + ":" + row["typeCategory"],
            )
            parts.append(f"{row['packageQuantity']} {noun}")
        return " + ".join(parts), None
    if "packageSummary" in extra:
        spec = extra["packageSummary"]
        packages = current["target"]["goodsItemDetails"][spec["goods"]]["numberAndTypeOfPackages"]
        if spec["part"] == "quantity":
            quantity = sum(row["packageQuantity"] for row in packages)
            return str(quantity), quantity
        if spec["part"] != "type" or package_vocabulary is None:
            raise ValueError("invalid package-summary projection")
        categories = {row["typeCategory"] for row in packages}
        category = categories.pop() if len(categories) == 1 else "PACKAGE_PACKAGE"
        if spec.get("familyOnly"):
            category = package_vocabulary.entries[category]["family"]
        if "styles" in spec:
            styles = spec["styles"]
            if (
                "style" in spec
                or not isinstance(styles, list)
                or not styles
                or any(s not in ("full", "short", "optional") for s in styles)
                or len(styles) != len(set(styles))
            ):
                raise ValueError("package summary needs distinct explicitly eligible styles")
            # Slot-specific ordered eligibility, not an invented abbreviation or
            # a missing-data fallback. Compact forms remain preferred when valid.
            forms = tuple(v for style in styles for v in package_vocabulary.forms(category, style))
        else:
            forms = package_vocabulary.forms(category, spec["style"])
        if not forms:
            raise ValueError("summary category has no reviewed alias for its selected style")
        return forms[0], None
    if "unitMassGroups" in extra:
        if package_vocabulary is None:
            raise ValueError("unit-mass group needs the package vocabulary")
        parts = []
        for category, mass in sampled_unit_mass_groups(candidate, current):
            noun = package_vocabulary.forms(category, "full")[0]
            value = format(mass.normalize(), "f")
            parts.append(f"NET WT. {value} KGS PER {noun}")
        return "; ".join(parts), None
    raise ValueError(f"new owner lacks a reviewed generation recipe: {extra['owner']}")


def get(target, path):
    for part in path.strip("/").split("/"):
        target = target[int(part)] if isinstance(target, list) else target[part]
    return target


def remove_optional_target(target, path):
    """Enforce a reviewed absence even when the optional target is already absent.

    Exclusion rules apply to all descendants, including those that never sampled
    the excluded field. Wrong container types and malformed selectors still fail.
    """
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("target exclusion requires an absolute nonempty path")
    parts = path[1:].split("/")
    if any(not part for part in parts):
        raise ValueError("target exclusion requires an absolute nonempty path")
    owner = target
    for index, part in enumerate(parts):
        if owner is None:
            return
        if isinstance(owner, dict):
            key = part
            if key not in owner:
                return
        elif isinstance(owner, list):
            if not part.isascii() or not part.isdecimal():
                raise ValueError("target exclusion requires a nonnegative list index")
            key = int(part)
            if key >= len(owner):
                return
        else:
            raise ValueError("target exclusion traverses a non-container value")
        if index == len(parts) - 1:
            del owner[key]
        else:
            owner = owner[key]


def extra_targets(extra, semantic):
    """Project only explicitly requested facts established by a rendered owner.

    A single equipment alias establishes size and type, but must not replace the
    containing inventory row or its existing identifier/seals.
    """
    if "target" in extra and "targets" in extra:
        raise ValueError("choose scalar or component target projection, not both")
    if "stripPostalCaptions" in extra:
        if "target" not in extra or not isinstance(semantic, str):
            raise ValueError("postal captions require a scalar text target")
        semantic = strip_postal_captions(semantic, extra)
    if "target" in extra:
        if semantic is None:
            if extra.get("dgFact") == {"field": "packingGroup", "omitAbsent": True}:
                return {}
            raise ValueError("new target requires an explicit semantic projection")
        return {extra["target"]: semantic}
    if "targets" not in extra:
        return {}
    targets = extra["targets"]
    if not isinstance(semantic, dict) or not isinstance(targets, dict) or not targets:
        raise ValueError("component targets require an explicit semantic object and mapping")
    if set(targets) - semantic.keys() or len(set(targets.values())) != len(targets):
        raise ValueError("component targets name absent facts or duplicate destinations")
    if any(not isinstance(path, str) or not path.startswith("/") for path in targets.values()):
        raise ValueError("component target destination must be an absolute JSON pointer")
    return {path: semantic[key] for key, path in targets.items()}


def rendered_owner_projection(receipts, action):
    """Project explicitly ordered occurrences, without merging separate contacts."""
    parts = []
    for owner in action["owners"]:
        matches = [r["after"] for r in receipts if r["owner"] == owner]
        if len(matches) != 1:
            raise ValueError("rendered target projection needs one exact owner")
        literal = " ".join(matches[0].split())
        parts.append(literal.upper() if action.get("targetCase") == "upper" else literal)
    if not parts:
        raise ValueError("rendered target projection needs at least one owner")
    return parts if action.get("asList") else " ".join(action["separator"].join(parts).split())


def source_passage_projection(pages, replacements, action):
    """Copy reviewed passages; deduplicate only complete repeated passages."""
    deduplicate = action.get("deduplicatePassages", False)
    if not isinstance(deduplicate, bool):
        raise ValueError("passage deduplication must be an explicit boolean")
    if "groups" in action:
        if any(key in action for key in ("regions", "spans", "fragments")):
            raise ValueError("choose one passage or ordered passage groups")
        groups = action["groups"]
        if not isinstance(groups, list) or not groups:
            raise ValueError("source passage groups must be a nonempty ordered list")
    else:
        groups = [action]
    values = []
    for group in groups:
        value = project_replacements(pages, spans_for(group, pages), replacements)
        if action.get("targetCase") == "upper":
            value = value.upper()
        # Equal complete occurrences are repetitions. Unequal quantities or
        # qualifiers remain distinct, even when their product words match.
        if not deduplicate or value not in values:
            values.append(value)
    return strip_postal_captions(" ".join(values), action)


def rendered_mass_sum(receipts, action):
    """Sum reviewed complete mass portions; never infer missing allocations."""
    owners = action["owners"]
    unit = action["unit"]
    if len(owners) < 2 or len(owners) != len(set(owners)):
        raise ValueError("mass sum requires distinct complete portion owners")
    if unit not in _UNITS or _UNITS[unit][0] != _UNITS["kilogram"][0]:
        raise ValueError("mass sum requires an explicit mass unit")
    separator = action.get("decimalSeparator", ".")
    if separator not in (".", ","):
        raise ValueError("mass sum requires an explicit decimal convention")
    values = []
    for owner in owners:
        matches = [r["after"] for r in receipts if r["owner"] == owner]
        if len(matches) != 1:
            raise ValueError("mass sum requires one exact receipt for each portion")
        if re.fullmatch(r"[0-9]+(?:" + re.escape(separator) + r"[0-9]+)?", matches[0]) is None:
            raise ValueError("mass sum portion must be an unformatted scalar")
        value = Decimal(matches[0].replace(separator, "."))
        if not value.is_finite() or value <= 0:
            raise ValueError("mass sum portions must be positive and finite")
        values.append(value)
    return {"value": float(sum(values)), "unit": unit}


def project_printed_postal_country(target, candidate, action, project_root):
    """Add a country only when its pinned alias occurs in this party's postal text.

    The sampled identity disambiguates the search vocabulary, not the evidence:
    a city, telephone prefix or planning country alone never produces a target.
    """
    from document_ocr.synthesis.country_presentations import Registry

    registry = Registry.load(
        project_root, manifest_path=action["manifestPath"], expected_sha256=action["sha256"]
    )
    text = get(target, action["addressTarget"])
    code = get(candidate["scenario"], action["codePath"])
    if not isinstance(text, str) or not text.strip():
        raise ValueError("postal country projection needs a nonempty owned address")
    name = registry.preferred_name(code)
    if not registry.find_spans(text, code, postal=True):
        if action.get("omitIfUnprinted") is True:
            # Explicit source-role policy: this address is the only permitted
            # country witness. A missing witness means an absent optional field,
            # never a country inferred from planning geography/exporter context.
            parent_path, _, field = action["target"].rpartition("/")
            parent = get(target, parent_path)
            if field in parent:
                del parent[field]
        return
    try:
        previous = get(target, action["target"])
    except KeyError:
        pass
    else:
        if previous != name and registry.resolve(previous) != code:
            raise ValueError("printed postal country conflicts with existing target")
    put(target, action["target"], name)


def put(target, path, value):
    parts = path.strip("/").split("/")
    for n, part in enumerate(parts[:-1]):
        if isinstance(target, list):
            while len(target) <= int(part):
                target.append({})
            target = target[int(part)]
        else:
            target = target.setdefault(part, [] if parts[n + 1].isdigit() else {})
    if isinstance(target, list):
        while len(target) <= int(parts[-1]):
            target.append({})
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


def literal_owner_value(
    receipts, keys, separator, *, deduplicate_postal_components=False, postal_code_key=None
):
    """Use final printed owners, not pre-formatting model surface values."""
    parts = []
    for key in keys:
        values = {r["after"] for r in receipts if r["owner"].startswith(key + "@")}
        if len(values) != 1:
            raise ValueError("literal target projection has ambiguous owner values")
        parts.append(values.pop())
    if postal_code_key is not None:
        if keys.count(postal_code_key) != 1:
            raise ValueError("postal-code deduplication requires one explicit code owner")
        index = keys.index(postal_code_key)
        code = " ".join(parts[index].split())
        if not code or not any(c.isdigit() for c in code):
            raise ValueError("postal-code owner must be a nonempty digit-bearing code")
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(s) for s in code.split()) + r"(?!\w)"
        if any(
            re.search(pattern, part, re.IGNORECASE)
            for i, part in enumerate(parts)
            if i != index
        ):
            parts.pop(index)
    if deduplicate_postal_components:
        # Postal continuations can repeat complete comma-delimited components.
        # Never remove a substring (York is not New York), reorder components,
        # or apply this operation to goods descriptions or other free text.
        if not parts:
            raise ValueError("postal projection requires at least one owner")
        seen = {" ".join(c.upper().split()) for c in parts[0].split(",")}
        unique = [parts[0]]
        for part in parts[1:]:
            components = part.split(",")
            kept = []
            for component in components:
                identity = " ".join(component.upper().split())
                if identity and identity not in seen:
                    kept.append(component.strip())
                    seen.add(identity)
            if kept:
                unique.append(", ".join(kept))
        parts = unique
    return " ".join(separator.join(parts).split())


def preserve_boundary_markers(value, source, markers):
    """Retain reviewed continuation punctuation owned by the source, not wording."""
    if set(markers) != {"prefix", "suffix"}:
        raise ValueError("boundary markers require explicit prefix and suffix")
    prefix, suffix = markers["prefix"], markers["suffix"]
    if not all(isinstance(v, str) for v in (prefix, suffix)) or not (prefix or suffix):
        raise ValueError("empty or nontext boundary markers")
    if any(c.isalnum() or c.isspace() for c in prefix + suffix):
        raise ValueError("boundary markers may only preserve punctuation")
    if not source.startswith(prefix) or (suffix and not source.endswith(suffix)):
        raise ValueError("boundary markers are not present in source owner")
    if prefix and not value.startswith(prefix):
        value = prefix + value
    if suffix and not value.endswith(suffix):
        value += suffix
    return value


def project_qualified_owner_countries(target, candidate, receipts, actions):
    """Use an explicitly owned qualified place, never planning geography alone."""

    def normalize(value):
        return " ".join(value.upper().split())

    for action in actions:
        if set(action) != {"key", "prefix", "separator", "namePath", "countryPath", "targets"}:
            raise ValueError("qualified-place country projection has unsupported fields")
        name = get(candidate["scenario"], action["namePath"])
        country = get(candidate["scenario"], action["countryPath"])
        if not all(isinstance(v, str) and v.strip() for v in (name, country)):
            raise ValueError("qualified place requires explicit name and country")
        literal = literal_owner_value(receipts, [action["key"]], " ")
        expected = action["prefix"] + name + action["separator"] + country
        if normalize(literal) != normalize(expected):
            raise ValueError("qualified place differs from its printed owner")
        if not action["targets"] or len(set(action["targets"])) != len(action["targets"]):
            raise ValueError("qualified place needs distinct reviewed target roles")
        for path in action["targets"]:
            role = get(target, path)
            if normalize(role["name"]) != normalize(name):
                continue
            if role.get("country") not in (None, country.upper()):
                raise ValueError("qualified place conflicts with target country")
            role["country"] = country.upper()


def project_owned_country_names(target, receipts, actions, project_root):
    """Normalize a reviewed country-only occurrence, not planning geography.

    An overlay establishes the occurrence's role (for example, an exporter
    declaration linked to the shipper by its repeated ID). The rendered value
    supplies the identity; stale sampled geography cannot override that text.
    """
    from document_ocr.synthesis.country_presentations import Registry

    for action in actions:
        if set(action) != {"target", "key", "manifestPath", "sha256"}:
            raise ValueError("owned country projection has unsupported fields")
        registry = Registry.load(
            project_root, manifest_path=action["manifestPath"], expected_sha256=action["sha256"]
        )
        literal = literal_owner_value(receipts, [action["key"]], " ")
        country = registry.preferred_name(registry.resolve(literal))
        try:
            previous = get(target, action["target"])
        except KeyError:
            previous = None
        if previous not in (None, country):
            raise ValueError("owned country conflicts with existing target")
        put(target, action["target"], country)


def target_projection_enabled(target, action):
    """Respect an independently reviewed ownership prerequisite for a projection.

    For example, an undifferentiated A/O-B contact block is not primary-party
    evidence; its already-reviewed postal target can gate that same block's
    new phone projection. This does not infer or remove existing target facts.
    """
    if "requiresTarget" not in action:
        return True
    prerequisite = action["requiresTarget"]
    if not isinstance(prerequisite, str) or not prerequisite.startswith("/"):
        raise ValueError("projection prerequisite requires an explicit target pointer")
    try:
        value = get(target, prerequisite)
    except KeyError:
        value = None
    if value is not None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("projection prerequisite requires nonempty reviewed text")
        return True
    try:
        existing = get(target, action["target"])
    except KeyError:
        existing = None
    if existing is not None:
        raise ValueError("ineligible projection conflicts with an existing target")
    return False


def coalesced_edits(edits, rules):
    """Combine reviewed split-source edits only when they emit one identical value.

    Old GLM bindings can independently expand each part of a wrapped person name
    into that person's full name. Paddle ownership covers the complete occurrence
    once; explicit member proofs prevent dropping a distinct repetition or value.
    """
    suppressed, reviewed = set(), set()
    for rule in rules:
        comparison = rule.get("comparison", "exact")
        if comparison not in ("exact", "casefold"):
            raise ValueError("unknown coalescence comparison")
        members = rule["members"]
        if len(members) < 2:
            raise ValueError("coalescence needs at least two explicit source members")
        found = []
        for member in members:
            matches = [
                (i, e)
                for i, e in enumerate(edits)
                if e["key"] == member["key"]
                and e["byteStart"] == member["byteStart"]
                and e["before"] == member["before"]
            ]
            if len(matches) != 1 or matches[0][0] in reviewed:
                raise ValueError("coalesced source member is missing or multiply owned")
            reviewed.add(matches[0][0])
            found.append(matches[0])
        if [i for i, _ in found] != sorted(i for i, _ in found):
            raise ValueError("coalescence members must retain source order")
        values = {e["after"] if comparison == "exact" else e["after"].casefold()
                  for _, e in found}
        if len(values) != 1:
            raise ValueError("coalescence cannot discard distinct sampled values")
        suppressed.update(i for i, _ in found[1:])
    return [e for i, e in enumerate(edits) if i not in suppressed]


def present_source_edits(edits, decisions, *, surface_values=None):
    """Exclude only individually reviewed GLM occurrences absent from Paddle.

    A value can still occur in another role: global text lookup cannot establish
    occurrence absence. Exact old-owner proof and an explicit rationale are
    required; the caller preserves these dispositions in the replay receipt.
    """
    resolved = []
    for decision in decisions:
        if "alternatives" not in decision:
            resolved.append(decision)
            continue
        if set(decision) != {"alternatives", "reason"} or not decision["reason"].strip():
            raise ValueError("alternate absent owners require an exact reviewed disposition")
        alternatives = decision["alternatives"]
        if len(alternatives) < 2 or any(not branch for branch in alternatives):
            raise ValueError("alternate absent owners require nonempty distinct branches")
        matching = []
        for branch in alternatives:
            if any(set(member) != {"key", "byteStart", "before"} for member in branch):
                raise ValueError("alternate absent member is not an exact source occurrence")
            present = []
            for member in branch:
                matches = [e for e in edits if all(e[k] == v for k, v in member.items())]
                value = (surface_values or {}).get(member["key"])
                values = [value] if isinstance(value, str) else value
                unchanged = bool(values) and all(v == member["before"] for v in values)
                present.append(len(matches) == 1 or (not matches and unchanged))
            if all(present):
                matching.append(branch)
        if len(matching) != 1:
            raise ValueError("alternate absent owners need exactly one complete matching branch")
        selected = matching[0]
        # A mixed composite/split receipt is not a permissible alternate topology.
        selected_owners = {(m["key"], m["byteStart"]) for m in selected}
        other_owners = {
            (m["key"], m["byteStart"]) for branch in alternatives for m in branch
        } - selected_owners
        if any((e["key"], e["byteStart"]) in other_owners for e in edits):
            raise ValueError("alternate absent owners mix incompatible receipt topologies")
        resolved.extend({**m, "reason": decision["reason"]} for m in selected)
    excluded, reviewed = set(), set()
    for decision in resolved:
        retained_constant = decision.get("retainedConstant", False)
        optional_occurrence = decision.get("optionalOccurrence", False)
        expected_fields = {"key", "byteStart", "before", "reason"}
        if "optionalOccurrence" in decision:
            if optional_occurrence is not True or retained_constant:
                raise ValueError("optional absent occurrence requires explicit, exclusive enablement")
            expected_fields.add("optionalOccurrence")
        if "retainedConstant" in decision:
            if retained_constant is not True:
                raise ValueError("retained constant disposition must be explicitly enabled")
            expected_fields.add("retainedConstant")
        if (
            set(decision) != expected_fields
            or not decision["reason"].strip()
        ):
            raise ValueError("absent source occurrence requires an exact reviewed disposition")
        occurrence = (decision["key"], decision["byteStart"])
        if occurrence in reviewed:
            raise ValueError("absent source occurrence is multiply disposed")
        reviewed.add(occurrence)
        matches = [
            i
            for i, edit in enumerate(edits)
            if all(edit[key] == decision[key] for key in ("key", "byteStart", "before"))
        ]
        if optional_occurrence and not matches:
            # Some certified receipt shapes record an edit here; others leave
            # the source text unchanged and have no owner. The native absence
            # decision still names one exact source occurrence, not a wildcard.
            if any(edit["byteStart"] == decision["byteStart"]
                   and edit["key"] == decision["key"] for edit in edits):
                raise ValueError("optional absent occurrence conflicts with its source proof")
            continue
        if retained_constant:
            # Some historical receipts record unchanged, non-target GLM
            # constants that newer receipts never own. Their explicitly
            # reviewed absence from Paddle must not discard a sampled value.
            at_occurrence = [e for e in edits if e["key"] == decision["key"]
                             and e["byteStart"] == decision["byteStart"]]
            if any(e["before"] != decision["before"] or e["after"] != e["before"]
                   or e.get("targetPaths") for e in at_occurrence):
                raise ValueError("retained constant disposition cannot discard a changed or targeted owner")
            if not at_occurrence:
                value = (surface_values or {}).get(decision["key"])
                values = [value] if isinstance(value, str) else value
                if values and any(v != decision["before"] for v in values):
                    raise ValueError("retained constant has a different sampled surface")
                continue
        if not matches and surface_values is not None:
            # The old renderer records changes, not all sampled owners. A
            # sampled value equal to the source legitimately has no edit.
            value = surface_values.get(decision["key"])
            values = [value] if isinstance(value, str) else value
            if values and all(v == decision["before"] for v in values):
                if any(
                    e["key"] == decision["key"] and e["byteStart"] == decision["byteStart"]
                    for e in edits
                ):
                    raise ValueError("unchanged absent occurrence conflicts with an edit")
                continue
        if len(matches) != 1 or matches[0] in excluded:
            raise ValueError("absent source occurrence is missing or multiply disposed")
        excluded.add(matches[0])
    return [edit for i, edit in enumerate(edits) if i not in excluded]


def apply_reviewed_country_coreferences(target, actions):
    """Share country evidence only between reviewed, retained same-place roles.

    This never geocodes a locality or copies a whole route role. Both name
    occurrences must already be present; a qualified occurrence supplies the
    country only when the sampled names still agree. A countryless homonym
    occurrence (for example a lone SINGAPORE) supplies no country evidence.
    """
    for action in actions:
        if set(action) != {"from", "to"} or action["from"] == action["to"]:
            raise ValueError("country co-reference requires two distinct reviewed roles")
        source, destination = get(target, action["from"]), get(target, action["to"])
        source_name, destination_name = source["name"], destination["name"]
        if not isinstance(source_name, str) or not isinstance(destination_name, str):
            raise ValueError("country co-reference needs printed locality names")
        if " ".join(source_name.upper().split()) != " ".join(destination_name.upper().split()):
            continue
        if "country" not in source or source["country"] is None:
            continue
        country = source["country"]
        if not isinstance(country, str) or not country.strip():
            raise ValueError("country co-reference needs nonempty country evidence")
        if "country" in destination and destination["country"] not in (None, country):
            raise ValueError("country co-reference conflicts with destination evidence")
        destination["country"] = country


def requires_wording(overlay):
    """Discover linguistic dependencies in every supported recipe position."""

    def uses_wording(recipe):
        return "wording" in recipe or any("wording" in part for part in recipe.get("parts", []))

    recipes = (
        overlay["extras"]
        + [item.get("valueRecipe", {}) for item in overlay.get("overrides", [])]
        + [piece for item in overlay.get("literalProjections", []) for piece in item["pieces"]]
    )
    return any(uses_wording(recipe) for recipe in recipes)


def existing_repetition_edits(extra, edits):
    """Resolve an explicit retained owner, including reviewed alternate receipts.

    Alternate selectors describe mutually exclusive historical owner boundaries,
    not a precedence order. More than one match is an ownership conflict.
    """
    if "existingOwners" in extra:
        if "existingKey" in extra or "existingByteStart" in extra:
            raise ValueError("alternate repetition selectors conflict with existingKey")
        selectors = extra["existingOwners"]
        if not isinstance(selectors, list) or len(selectors) < 2:
            raise ValueError("alternate repetition requires at least two explicit selectors")
        for selector in selectors:
            if set(selector) != {"key", "byteStart", "byteEnd"} or not (
                isinstance(selector["key"], str)
                and selector["key"]
                and type(selector["byteStart"]) is int
                and type(selector["byteEnd"]) is int
                and 0 <= selector["byteStart"] < selector["byteEnd"]
            ):
                raise ValueError("alternate repetition selector needs an exact owner interval")
        if len({(s["key"], s["byteStart"], s["byteEnd"]) for s in selectors}) != len(selectors):
            raise ValueError("alternate repetition selectors are duplicated")
        matches = [e for e in edits if any(all(e[k] == v for k, v in s.items()) for s in selectors)]
        if len(matches) > 1:
            raise ValueError("alternate repetition owners are not mutually exclusive")
        return matches
    if "existingKey" not in extra:
        return []
    return [
        e for e in edits
        if e["key"] == extra["existingKey"]
        and ("existingByteStart" not in extra or e["byteStart"] == extra["existingByteStart"])
    ]


def consumed_native_extra(extra, logical_owners, pages):
    """A reviewed compound mode replaces its scalar presentation components.

    This is explicit ownership, not text matching or a rendering fallback. Target
    projections cannot be suppressed: the aggregate/group target is owned by the
    sampled shipment, and this rule is limited to redundant presentation extras.
    """
    selectors = extra.get("consumedBy", [])
    if not selectors:
        return False
    if any(k in extra for k in ("target", "targets")):
        raise ValueError("compound consumption cannot discard a target projection")
    if (
        not isinstance(selectors, list)
        or not all(isinstance(s, str) and s for s in selectors)
        or len(set(selectors)) != len(selectors)
    ):
        raise ValueError("compound consumption needs distinct exact owner selectors")
    owners = [logical_owners[s] for s in selectors if s in logical_owners]
    if not owners:
        return False
    if len(owners) != 1:
        raise ValueError("compound consumption modes are not mutually exclusive")
    for span in spans_for(extra, pages):
        if not any(
            (span.page, span.region) == (part.page, part.region)
            and part.start <= span.start < span.end <= part.end
            and part.expected[span.start - part.start:span.end - part.start] == span.expected
            for part in owners[0].spans
        ):
            raise ValueError("compound owner does not contain its native scalar extra")
    return True


def render_one(
    record,
    transfer,
    candidate,
    current,
    overlay,
    engine,
    identity,
    equipment_registry,
    package_vocabulary=None,
    *,
    wording_receipt,
    project_root,
):
    if candidate["joinedRawText"] != current["joinedRawText"]:
        raise ValueError("candidate is not the current R29 content")
    pages = load_pages(record)
    nested_owners = [extra for extra in overlay["extras"] if "nestedPacking" in extra]
    if nested_owners:
        from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates

        pin = overlay["numericCertificate"]
        authority = IndependentNumericCertificates.from_path(
            project_root / pin["path"], expected_sha256=pin["sha256"]
        ).certificate(record["documentId"])["externalPackingPresentation"]
        if len(nested_owners) != 1 or nested_owners[0]["owner"] != authority["owner"]:
            raise ValueError("native packing requires its exact certified presentation owner")
        if authority["nativeFingerprint"] != fingerprint(pages):
            raise ValueError("native packing certificate belongs to a different source")
    for extra in overlay["extras"]:
        if "nestedPacking" in extra and extra["nestedPacking"]["nativeFingerprint"] != fingerprint(
            pages
        ):
            raise ValueError("nested packing presentation belongs to a different native source")
    wording = None
    if requires_wording(overlay):
        if wording_receipt is None:
            raise ValueError("new linguistic owners require an explicit reviewed receipt")
        receipt = wording_receipt
        if receipt["status"] != "REVIEWED" or receipt["sourceFingerprint"] != fingerprint(pages):
            raise ValueError("new wording lacks reviewed native-source provenance")
        selected = [entry for entry in transfer["descendants"] if entry["id"] == identity]
        if len(selected) != 1 or receipt["candidateSha256"] != sha256_file(
            project_root / selected[0]["path"]
        ):
            raise ValueError("reviewed wording belongs to a different shipment receipt")
        wording = receipt["values"]
    elif wording_receipt is not None:
        raise ValueError("wording receipt supplied without new linguistic owners")
    lookup = {s["owner"]: s for s in transfer["slots"]}
    replacements, receipts, new_targets = [], [], {}
    logical_owners = {}
    present = present_source_edits(
        candidate["proof"]["edits"],
        overlay.get("absentEdits", []),
        surface_values=candidate.get("surfaceValues"),
    )
    for edit in coalesced_edits(present, overlay.get("coalescedEdits", [])):
        owner = f"{edit['key']}@{edit['byteStart']}:{edit['byteEnd']}"
        override = [
            r
            for r in overlay["overrides"]
            if r["key"] == edit["key"]
            and ("byteStart" not in r or r["byteStart"] == edit["byteStart"])
            and ("byteEnd" not in r or r["byteEnd"] == edit["byteEnd"])
        ]
        if len(override) > 1:
            raise ValueError("ambiguous override")
        if override and "afterParts" in override[0]:
            parts = override[0]["afterParts"]
            if " ".join(" ".join(p["after"] for p in parts).split()) != " ".join(
                edit["after"].split()
            ):
                raise ValueError("split presentation does not conserve the sampled text")
            owned_spans = []
            for i, part in enumerate(parts):
                spans = spans_for(part, pages)
                owned_spans.extend(spans)
                subowner = f"{owner}:part:{i}"
                replacements.append(PaddleReplacement(subowner, spans, part["after"]))
                receipts.append(
                    {"owner": subowner, "spans": [asdict(s) for s in spans], "after": part["after"]}
                )
            logical_owners[owner] = PaddleReplacement(owner, tuple(owned_spans), edit["after"])
            continue
        if override:
            spans = spans_for(override[0], pages)
        else:
            binding = lookup[owner]
            if binding["status"] != "exact_measured_owner_candidate":
                raise ValueError(f"unreviewed binding: {owner}")
            spans = tuple(RegionSpan(**s) for s in binding["spans"])
        value = edit["after"]
        if override and "valueRecipe" in override[0]:
            value, _ = extra_value(
                {"owner": owner, **override[0]["valueRecipe"]},
                candidate,
                current,
                identity,
                equipment_registry,
                package_vocabulary,
                wording,
                project_root=project_root,
            )
        if override and "afterLine" in override[0]:
            value = project_sampled_lines(
                value, override[0]["afterLine"], stride=override[0].get("lineStride")
            )
        if override:
            value = render_soft_line_breaks(value, override[0])
        if override and "boundaryMarkers" in override[0]:
            value = preserve_boundary_markers(
                value, "".join(span.expected for span in spans), override[0]["boundaryMarkers"]
            )
        if override and "separatorBefore" in override[0]:
            separator = override[0]["separatorBefore"]
            if not isinstance(separator, str) or (separator and not separator.isspace()):
                raise ValueError("owner separator must contain only whitespace")
            value = separator + value
        replacement = PaddleReplacement(owner, spans, value)
        logical_owners[owner] = replacement
        pieces = (
            partition_region_frames(
                replacement,
                engine.metrics.length,
                allow_empty_continuations=override[0].get("allowEmptyContinuations", False),
            )
            if override and override[0].get("separateRegionFrames")
            else (replacement,)
        )
        replacements.extend(pieces)
        receipts.extend(
            {"owner": piece.owner, "spans": [asdict(s) for s in piece.spans], "after": piece.text}
            for piece in pieces
        )
    for extra in overlay["extras"]:
        if consumed_native_extra(extra, logical_owners, pages):
            continue
        if "existingKey" in extra or "existingOwners" in extra:
            matches = existing_repetition_edits(extra, candidate["proof"]["edits"])
            if matches:
                owner_ids = {f"{e['key']}@{e['byteStart']}:{e['byteEnd']}" for e in matches}
                owners = tuple(logical_owners[key] for key in owner_ids)
                expected = spans_for(extra, pages)
                validate_existing_repetition(owners, expected)
                if "target" not in extra and "targets" not in extra:
                    continue
                value, semantic = extra_value(
                    extra,
                    candidate,
                    current,
                    identity,
                    equipment_registry,
                    package_vocabulary,
                    wording,
                    project_root=project_root,
                )
                validate_existing_repetition(owners, expected, semantic_text=value)
                for path, fact in extra_targets(extra, semantic).items():
                    if path in new_targets and new_targets[path] != fact:
                        raise ValueError("repeated new targets disagree")
                    new_targets[path] = fact
                continue
        value, semantic = extra_value(
            extra,
            candidate,
            current,
            identity,
            equipment_registry,
            package_vocabulary,
            wording,
            project_root=project_root,
        )
        for path, fact in extra_targets(extra, semantic).items():
            if path in new_targets and new_targets[path] != fact:
                raise ValueError("repeated new targets disagree")
            new_targets[path] = fact
        spans = spans_for(extra, pages)
        replacement = PaddleReplacement(extra["owner"], spans, value)
        if extra.get("separateRegionFrames"):
            pieces = partition_region_frames(
                replacement,
                engine.metrics.length,
                allow_empty_continuations=extra.get("allowEmptyContinuations", False),
            )
            replacements.extend(pieces)
            receipts.extend(
                {
                    "owner": piece.owner,
                    "spans": [asdict(s) for s in piece.spans],
                    "after": piece.text,
                }
                for piece in pieces
            )
        else:
            replacements.append(replacement)
            receipts.append(
                {"owner": extra["owner"], "spans": [asdict(s) for s in spans], "after": value}
            )
    blocks = compile_blocks(
        pages,
        tuple(replacements),
        physical_groups=tuple((p, tuple(ix)) for p, ix in overlay.get("physicalGroups", [])),
    )
    rendered, geometry = engine.render(
        pages,
        blocks,
        identity=identity,
        watermark_regions=tuple(tuple(ref) for ref in overlay.get("watermarkRegions", [])),
        right_aligned_numeric_regions=tuple(
            tuple(ref) for ref in overlay.get("rightAlignedNumericRegions", [])
        ),
        row_groups=tuple((page, tuple(regions)) for page, regions in overlay.get("rowGroups", [])),
    )
    # Current R29 labels own subsequent policy repairs even when their OCR text
    # still equals an older synthesis receipt. Never restore stale target values.
    target = copy.deepcopy(current["target"])
    for action in overlay["targetCopies"]:
        put(target, action["to"], copy.deepcopy(get(target, action["from"])))
    for path, value in overlay["targetSets"].items():
        put(target, path, copy.deepcopy(value))
    for action in overlay.get("targetFromOwners", []):
        if not target_projection_enabled(target, action):
            continue
        postal = action.get("deduplicatePostalComponents", False)
        if type(postal) is not bool or (postal and not action["target"].endswith("/addressLine")):
            raise ValueError("component deduplication is restricted to postal addresses")
        postal_code = action.get("deduplicatePostalCodeKey")
        if postal_code is not None and not action["target"].endswith("/addressLine"):
            raise ValueError("postal-code deduplication is restricted to postal addresses")
        literal = literal_owner_value(
            receipts,
            action["keys"],
            action["separator"],
            deduplicate_postal_components=postal,
            postal_code_key=postal_code,
        )
        literal = strip_postal_captions(literal, action)
        if action.get("targetCase") == "upper":
            literal = literal.upper()
        put(target, action["target"], [literal] if action.get("asList") else literal)
    for action in overlay.get("targetFromRenderedOwners", []):
        if not target_projection_enabled(target, action):
            continue
        put(target, action["target"], rendered_owner_projection(receipts, action))
    for action in overlay.get("targetFromSourceSpans", []):
        literal = source_passage_projection(pages, tuple(replacements), action)
        put(target, action["target"], [literal] if action.get("asList") else literal)
    for action in overlay.get("targetMassSums", []):
        put(target, action["target"], rendered_mass_sum(receipts, action))
    for action in overlay.get("literalProjections", []):
        parts = []
        for piece in action["pieces"]:
            value, _ = extra_value(
                piece,
                candidate,
                current,
                identity,
                equipment_registry,
                package_vocabulary,
                wording,
                project_root=project_root,
            )
            parts.append(piece.get("before", "") + value + piece.get("after", ""))
        literal = " ".join(action["separator"].join(parts).split())
        value = literal.upper() if action.get("targetCase") == "upper" else literal
        put(
            target,
            action["target"],
            [value] if action.get("asList") else value,
        )
    for path, value in new_targets.items():
        try:
            previous = get(target, path)
        except (KeyError, IndexError):
            pass
        else:
            if previous != value:
                raise ValueError("new target conflicts with an existing field")
        put(target, path, value)
    for action in overlay.get("targetPrintedPostalCountries", []):
        project_printed_postal_country(target, candidate, action, project_root)
    for path in overlay.get("targetRemovals", []):
        remove_optional_target(target, path)
    apply_reviewed_country_coreferences(target, overlay.get("targetCountryCoreferences", []))
    project_qualified_owner_countries(
        target, candidate, receipts, overlay.get("targetQualifiedPlaceCountries", [])
    )
    project_owned_country_names(
        target, receipts, overlay.get("targetCountryFromOwners", []), project_root
    )
    if overlay.get("orderContainersByFirstOccurrence"):
        order_containers_by_printed_occurrence(target, rendered)
    for action in overlay.get("targetPostalCaptions", []):
        value = get(target, action["target"])
        if not isinstance(value, str):
            raise ValueError("postal-caption projection requires an existing text address")
        put(target, action["target"], strip_postal_captions(value, action))
    target = BillOfLadingExtractionV7DirectTarget.model_validate_json(
        json.dumps(target, ensure_ascii=False)
    ).canonical_target()
    return {
        "status": "FULL_RENDER_CANDIDATE_REQUIRES_AUDIT",
        "documentId": identity,
        "sourceDocumentId": record["documentId"],
        "sourceFingerprint": fingerprint(pages),
        "fingerprint": fingerprint(rendered),
        "pages": [asdict(p) for p in rendered],
        "target": target,
        "joinedRawText": serialize(rendered, geometry="none"),
        "positionedText": serialize(rendered),
        "bboxText": serialize(rendered, geometry="bbox"),
        "proof": receipts,
        "geometry": geometry,
        **({"absentSourceOwners": overlay["absentEdits"]} if overlay.get("absentEdits") else {}),
    }
