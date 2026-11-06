import copy
import importlib.util
from decimal import Decimal as D
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_scenarios import TransportCapacityConfig
from document_ocr.synthesis.declared_packages import CountedPackage, DeclaredPackages
from document_ocr.synthesis.transport_capacity import capacity_limits, equipment_capacity

spec = importlib.util.spec_from_file_location("r30_equipment_receipts", Path(__file__).resolve().parents[1] / "scripts/migrations/r30_equipment_receipts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    limits = capacity_limits(TransportCapacityConfig(
        policy="source_type_aware_maersk_upper_bounds_v1", published_reference_margin_fraction=0,
        twenty_standard_payload_kg=28300, twenty_standard_volume_m3=33.2,
        forty_standard_payload_kg=28870, forty_standard_volume_m3=67.7,
        forty_high_cube_payload_kg=28690, forty_high_cube_volume_m3=76.4,
        forty_five_high_cube_payload_kg=27650, forty_five_high_cube_volume_m3=86,
        out_of_gauge_payload_kg=47300, unclassified_payload_kg=47300, unclassified_volume_m3=86,
    ))
    equipment = {"equipmentIdentifier": "ID", "sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
    changed = {**equipment, "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT"}
    group = CountedPackage("PACKAGE_BOX", D(10), D(11), D(".1"))
    def ledger(row):
        c = equipment_capacity(row, limits)
        return DeclaredPackages((group,), ((10,),), ((c.payload_kg, c.volume_m3),)).receipt()
    old, new = ledger(equipment), ledger(changed)
    before = {"documentId": "current", "sourceDocumentId": "source", "joinedRawText": "40GP", "target": {"containerInformation": [equipment], "goodsItemDetails": [{"description": "BOLTS", "numberAndTypeOfPackages": old["rows"]}]}}
    after = copy.deepcopy(before); after["joinedRawText"] = "20GP"; after["target"]["containerInformation"] = [changed]
    edit = {"key": "equipment", "occurrence": 0, "indices": [0], "byteStart": 0, "byteEnd": 4, "before": "40GP", "after": "20GP"}
    stage = {"before": before, "after": after, "evidence": {"nativeCandidateId": "native", "ledger": new, "beforeSizes": [equipment["sizeCategory"]], "afterSizes": [changed["sizeCategory"]], "proof": {"edits": [edit], "sourceSha256": digest(b"40GP"), "renderedSha256": digest(b"20GP")}}}
    candidate = {"documentId": "native", "sourceDocumentId": "source", "target": {"documentPatch": copy.deepcopy(before["target"])}, "physical": {"tares": {}, "equipmentInventoryOwners": [{"owner": "equipment", "occurrence": 0, "indices": [0], "after": "40GP"}]}, "scenario": {"cargo": {"containerInformation": [copy.deepcopy(equipment)]}, "provenance": {"declaredPackages": old, "physicalRows": [{"quantity": 10, "capacity": {}}], "independentDraw": {"declaredPackages": copy.deepcopy(old), "equipment": [copy.deepcopy(equipment)], "provenance": {"rowCapacities": []}}}}}
    return candidate, stage, limits


def test_saved_equipment_facts_preserve_loads_and_inputs():
    candidate, stage, limits = fixture(); frozen = copy.deepcopy((candidate, stage))
    result = module.recover_equipment_facts(candidate, stage, limits)
    assert (candidate, stage) == frozen
    assert result["scenario"]["provenance"]["declaredPackages"]["loads"] == candidate["scenario"]["provenance"]["declaredPackages"]["loads"]
    assert result["scenario"]["cargo"]["containerInformation"][0]["sizeCategory"] == "TWENTY_FOOT_STANDARD_HEIGHT"
    assert result["physical"]["equipmentInventoryOwners"][0]["after"] == "20GP"
    assert result["target"] == candidate["target"]  # Exact-current composition remains separate.


@pytest.mark.parametrize("defect", ["text", "cargo", "tare", "capacity", "owner", "native", "ledger"])
def test_saved_equipment_facts_fail_closed(defect):
    candidate, stage, limits = fixture()
    if defect == "text": stage["after"]["joinedRawText"] = "WRONG"
    elif defect == "cargo": stage["after"]["target"]["goodsItemDetails"][0]["description"] = "OTHER"
    elif defect == "tare": candidate["physical"]["tares"] = {"ID": 2000}
    elif defect == "capacity": stage["evidence"]["ledger"]["capacities"][0][1] = "40"
    elif defect == "owner": candidate["physical"]["equipmentInventoryOwners"][0]["after"] = "40HC"
    elif defect == "native": stage["evidence"]["nativeCandidateId"] = "other"
    elif defect == "ledger": stage["evidence"]["ledger"]["matrix"][0][0] = 11
    with pytest.raises(ValueError): module.recover_equipment_facts(candidate, stage, limits)
