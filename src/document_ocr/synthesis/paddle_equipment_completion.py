"""Validate explicitly sampled missing hardware without relabeling retained text.

Some old GLM receipts did not sample classifications because GLM omitted the
equipment printout. A reviewed completion can add private equipment facts for
new Paddle owners, while retaining the old shipment and its public absence.
"""

from decimal import Decimal
from types import SimpleNamespace

from document_ocr.hashing import canonical_json_sha256
from document_ocr.synthesis.transport_capacity import capacity_limits, equipment_capacity

_FACTOR = {
    "grossWeight": {"kilogram": Decimal(1), "metric_tonne": Decimal(1000), "pound": Decimal("0.45359237")},
    "netWeight": {"kilogram": Decimal(1), "metric_tonne": Decimal(1000), "pound": Decimal("0.45359237")},
    "volume": {"cubic_metre": Decimal(1)},
}


def completed_equipment_inventory(candidate, current):
    completion = candidate["migrationMissingEquipment"]
    if completion["format"] != "reviewed-missing-equipment-v1":
        raise ValueError("unsupported missing-equipment completion")
    for name in ("previousCandidate", "review"):
        pin = completion[name]
        if not pin.get("path") or len(pin.get("sha256", "")) != 64:
            raise ValueError("equipment completion lacks pinned review provenance")
    base = {k: v for k, v in candidate.items() if k != "migrationMissingEquipment"}
    if canonical_json_sha256(base) != completion["baseContentSha256"]:
        raise ValueError("equipment completion changed the retained shipment")
    provenance = candidate["scenario"]["provenance"]
    if provenance["method"] != "registry_goods_train_physical_support_geography_v2" or provenance["capabilities"]["family"] != "ambient":
        raise ValueError("equipment completion needs the reviewed retained ambient contract")
    retained = candidate["target"]["documentPatch"]["containerInformation"]
    public = current["target"]["containerInformation"]
    source = candidate["scenario"]["cargo"]["containerInformation"]
    selected = completion["equipment"]
    if not retained or not len(retained) == len(public) == len(source) == len(selected):
        raise ValueError("completed equipment inventory count differs")
    ids = [c["equipmentIdentifier"] for c in retained]
    if len(set(ids)) != len(ids) or any([r["equipmentIdentifier"] for r in rows] != ids for rows in (public, selected)):
        raise ValueError("completed equipment inventory identity/order differs")
    if any("sizeCategory" in c or "typeCategory" in c for c in retained + public):
        raise ValueError("completion cannot overwrite an existing equipment classification")
    source_ids = [c["equipmentIdentifier"] for c in source]
    physical = provenance["physicalRows"]
    if len({p["equipmentIdentifier"] for p in physical}) != len(source) or {p["equipmentIdentifier"] for p in physical} != set(source_ids):
        raise ValueError("completed equipment needs exact retained physical owners")
    limits = capacity_limits(SimpleNamespace(**completion["capacityLimits"]))
    by_id = {p["equipmentIdentifier"]: p for p in physical}
    result = []
    for i, chosen in enumerate(selected):
        if chosen["typeCategory"] != "GENERAL_PURPOSE":
            raise ValueError("missing-equipment completion is not a special-hardware substitution")
        cap = equipment_capacity(chosen, limits)
        if cap.family not in {"twenty_standard", "forty_standard", "forty_high_cube", "forty_five_high_cube"}:
            raise ValueError("completed equipment has no admitted dry capacity")
        measures = by_id[source_ids[i]]["measures"]
        if not measures:
            raise ValueError("completed equipment requires retained load evidence")
        for field, measure in measures.items():
            value = Decimal(str(measure["value"])) * _FACTOR[field][measure["unit"]]
            bound = cap.volume_m3 if field == "volume" else cap.payload_kg
            if not value.is_finite() or value <= 0 or value > bound:
                raise ValueError("completed equipment cannot contain the retained load")
        result.append({**public[i], **{k: chosen[k] for k in ("sizeCategory", "typeCategory")}})
    witnesses = completion["tareWitnessKg"]
    tares = candidate["physical"]["tares"]
    if (tares and set(witnesses) != {str(i) for i in range(len(selected))}) or (not tares and witnesses):
        raise ValueError("equipment tare witness membership differs")
    for i, value in witnesses.items():
        kg = Decimal(value)
        if not kg.is_finite() or kg <= 0:
            raise ValueError("equipment tare witness must be finite and positive")
        pair = selected[int(i)]["sizeCategory"] + "|" + selected[int(i)]["typeCategory"]
        if not any(e["kg"] == value and e["pair"] == pair and e["sourceIds"] for e in completion["tareSupport"]):
            raise ValueError("equipment tare witness lacks reviewed train support")
    for observation in tares:
        owners = [observation["owner"]] if "owner" in observation else observation["owners"]
        if sum((Decimal(witnesses[str(i)]) for i in owners), Decimal(0)) != Decimal(observation["kg"]):
            raise ValueError("equipment completion changes a retained tare")
    return result
