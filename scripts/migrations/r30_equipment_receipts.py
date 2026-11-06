"""Recover saved equipment-only refreshes without resampling cargo.

The refresh proof owns text changes and the complete declared-package ledger
owns capacity changes. Both must agree before a private receipt can be reused.
This is transport preparation, not native synthesis acceptance.
"""

import copy
from decimal import Decimal

from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.declared_packages import DeclaredPackages
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits
from document_ocr.synthesis.transport_capacity import equipment_capacity


def recover_equipment_facts(candidate, stage, limits):
    before, after, evidence = stage["before"], stage["after"], stage["evidence"]
    if candidate["documentId"] != evidence["nativeCandidateId"]:
        raise ValueError("equipment refresh uses another native candidate")
    if not before["sourceDocumentId"] == after["sourceDocumentId"] == candidate["sourceDocumentId"]:
        raise ValueError("equipment refresh changes source membership")
    if before["documentId"] != after["documentId"]:
        raise ValueError("equipment refresh changes record identity")
    proof = evidence["proof"]
    raw = before["joinedRawText"].encode()
    if digest(raw) != proof["sourceSha256"] or digest(after["joinedRawText"].encode()) != proof["renderedSha256"]:
        raise ValueError("equipment refresh text fingerprint differs")
    if apply_edits(raw, proof["edits"]).decode() != after["joinedRawText"]:
        raise ValueError("equipment refresh does not replay exactly")
    old, new = flat(before["target"]), flat(after["target"])
    changes = [p for p in old.keys() | new.keys() if old.get(p) != new.get(p)]
    if not changes or any(not(p.startswith("containerInformation[") and p.endswith(".sizeCategory")) for p in changes):
        raise ValueError("equipment-only refresh changes other shipment facts")
    original = candidate["target"]["documentPatch"]
    if original["containerInformation"] != before["target"]["containerInformation"]:
        raise ValueError("equipment refresh before-inventory differs")
    def numeric_goods(target):
        return [{k: v for k, v in g.items() if k != "description"} for g in target["goodsItemDetails"]]
    if numeric_goods(original) != numeric_goods(before["target"]):
        raise ValueError("equipment refresh changes cargo accounting")
    if candidate["physical"]["tares"]:
        raise ValueError("equipment refresh requires explicit tare replanning")
    provenance = candidate["scenario"]["provenance"]
    previous, ledger = provenance["declaredPackages"], evidence["ledger"]
    for key in previous.keys() | ledger.keys():
        if key != "capacities" and previous.get(key) != ledger.get(key):
            raise ValueError("equipment refresh changes declared cargo ledger")
    if provenance["independentDraw"]["declaredPackages"] != previous:
        raise ValueError("native declared-package authorities disagree")
    DeclaredPackages.from_receipt(previous)
    DeclaredPackages.from_receipt(ledger)
    equipment = after["target"]["containerInformation"]
    capacities = []
    for item, expected in zip(equipment, ledger["capacities"], strict=True):
        capacity = equipment_capacity(item, limits)
        if (capacity.payload_kg, capacity.volume_m3) != tuple(map(Decimal, expected)):
            raise ValueError("saved capacity differs from selected equipment")
        capacities.append({"payloadKg": expected[0], "volumeM3": expected[1]})
    result = copy.deepcopy(candidate)
    p = result["scenario"]["provenance"]
    p["declaredPackages"] = copy.deepcopy(ledger)
    p["independentDraw"]["declaredPackages"] = copy.deepcopy(ledger)
    p["independentDraw"]["provenance"]["rowCapacities"] = copy.deepcopy(capacities)
    for scope in (result["scenario"]["cargo"]["containerInformation"], p["independentDraw"]["equipment"]):
        for row, sampled in zip(scope, equipment, strict=True):
            if row["typeCategory"] != sampled["typeCategory"]:
                raise ValueError("equipment refresh changes equipment type")
            row["sizeCategory"] = sampled["sizeCategory"]
    for row, cap in zip(p["physicalRows"], capacities, strict=True):
        row["capacity"] = cap
    owners = result["physical"]["equipmentInventoryOwners"]
    for edit in proof["edits"]:
        matches = [o for o in owners if (o["owner"], o["occurrence"]) == (edit["key"], edit["occurrence"])]
        if len(matches) != 1 or matches[0]["indices"] != edit["indices"] or matches[0]["after"] != edit["before"]:
            raise ValueError("equipment refresh lacks exact sampled inventory owner")
        matches[0]["after"] = edit["after"]
    result["migrationEquipmentRefresh"] = {
        "status": "PRIVATE_FACTS_RECOVERED_REQUIRES_EXACT_CURRENT_COMPOSITION_AND_NATIVE_REVIEW",
        "stageSha256": digest(stage), "beforeSizes": evidence["beforeSizes"],
        "afterSizes": evidence["afterSizes"], "cargoLedgerUnchangedExceptCapacities": True,
    }
    return result
