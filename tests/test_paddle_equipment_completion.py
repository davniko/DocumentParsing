import copy
import json
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_sha256
from document_ocr.synthesis.paddle_equipment_completion import completed_equipment_inventory
from document_ocr.synthesis.paddle_render import sampled_equipment


@pytest.fixture
def shipment():
    public = [{"equipmentIdentifier": "NEW_A"}, {"equipmentIdentifier": "NEW_B"}]
    c = {
        "target": {"documentPatch": {"containerInformation": copy.deepcopy(public)}},
        "physical": {"tares": [{"owners": [0, 1], "kg": "7400"}]},
        "scenario": {
            "cargo": {"containerInformation": [{"equipmentIdentifier": "OLD_A"}, {"equipmentIdentifier": "OLD_B"}]},
            "provenance": {
                "method": "registry_goods_train_physical_support_geography_v2",
                "capabilities": {"family": "ambient"},
                "physicalRows": [{"equipmentIdentifier": ident, "measures": {
                    "grossWeight": {"value": "10", "unit": "metric_tonne"},
                    "volume": {"value": "30", "unit": "cubic_metre"},
                }} for ident in ("OLD_A", "OLD_B")],
            },
        },
    }
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/synthesis/mpci_bl_curated_v7_paddle.json").read_text())
    selected = [{**r, "sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"} for r in public]
    c["migrationMissingEquipment"] = {
        "format": "reviewed-missing-equipment-v1",
        "baseContentSha256": canonical_json_sha256(c),
        "previousCandidate": {"path": "previous.json", "sha256": "a" * 64},
        "review": {"path": "review.json", "sha256": "b" * 64},
        "capacityLimits": cfg["sampling"]["transport_capacity"],
        "equipment": selected,
        "tareWitnessKg": {"0": "3700", "1": "3700"},
        "tareSupport": [{"kg": "3700", "pair": "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE", "sourceIds": ["TRAIN"]}],
    }
    return c, {"target": {"containerInformation": public}}


def repin(c):
    c["migrationMissingEquipment"]["baseContentSha256"] = canonical_json_sha256({k: v for k, v in c.items() if k != "migrationMissingEquipment"})


def test_completion_adds_only_explicit_private_pair_and_preserves_inputs(shipment):
    c, current = shipment
    before = copy.deepcopy(shipment)
    result = completed_equipment_inventory(c, current)
    assert result == c["migrationMissingEquipment"]["equipment"]
    assert sampled_equipment({"authority": "sampled_inventory", "index": 1}, c, current) == result[1]
    assert sampled_equipment({"index": 1}, c, current) == {"equipmentIdentifier": "NEW_B"}
    assert shipment == before


@pytest.mark.parametrize("change", [
    lambda c, r: c["scenario"]["provenance"].update(method="other"),
    lambda c, r: c["physical"]["tares"][0].update(kg="8000"),
    lambda c, r: c["target"]["documentPatch"]["containerInformation"][0].update(sealNumbers=["changed"]),
])
def test_completion_rejects_any_retained_mutation(shipment, change):
    c, current = shipment
    change(c, current)
    with pytest.raises(ValueError, match="retained shipment"):
        completed_equipment_inventory(c, current)


@pytest.mark.parametrize("change", [
    lambda c, r: c["migrationMissingEquipment"]["equipment"].pop(),
    lambda c, r: c["migrationMissingEquipment"]["equipment"].reverse(),
    lambda c, r: r["target"]["containerInformation"][0].update(equipmentIdentifier="OTHER"),
    lambda c, r: r["target"]["containerInformation"][0].update(sizeCategory="FORTY_FOOT_HIGH_CUBE"),
    lambda c, r: c["migrationMissingEquipment"]["equipment"][0].update(typeCategory="REFRIGERATED"),
    lambda c, r: c["migrationMissingEquipment"]["equipment"][0].update(sizeCategory="UNKNOWN"),
    lambda c, r: c["migrationMissingEquipment"].update(tareWitnessKg={}),
    lambda c, r: c["migrationMissingEquipment"]["tareWitnessKg"].update({"0": "NaN"}),
    lambda c, r: c["migrationMissingEquipment"]["tareWitnessKg"].update({"0": "3800"}),
    lambda c, r: c["migrationMissingEquipment"]["tareSupport"].clear(),
    lambda c, r: c["migrationMissingEquipment"].update(review={}),
])
def test_completion_rejects_invalid_new_authority(shipment, change):
    c, current = shipment
    change(c, current)
    with pytest.raises(ValueError):
        completed_equipment_inventory(c, current)


@pytest.mark.parametrize("field,value,unit", [
    ("grossWeight", "100", "metric_tonne"),
    ("netWeight", "30000", "kilogram"),
    ("volume", "100", "cubic_metre"),
    ("grossWeight", "NaN", "kilogram"),
    ("volume", "0", "cubic_metre"),
])
def test_completed_capacity_checked_against_real_retained_load(shipment, field, value, unit):
    c, current = shipment
    c["scenario"]["provenance"]["physicalRows"][0]["measures"][field] = {"value": value, "unit": unit}
    repin(c)
    with pytest.raises(ValueError, match="retained load"):
        completed_equipment_inventory(c, current)


def test_individual_and_aggregate_tare_must_both_match(shipment):
    c, current = shipment
    c["physical"]["tares"].append({"owner": 0, "kg": "3750"})
    repin(c)
    with pytest.raises(ValueError, match="retained tare"):
        completed_equipment_inventory(c, current)


def test_no_tare_completion_does_not_invent_tare(shipment):
    c, current = shipment
    c["physical"]["tares"] = []
    c["migrationMissingEquipment"]["tareWitnessKg"] = {}
    c["migrationMissingEquipment"]["tareSupport"] = []
    repin(c)
    assert completed_equipment_inventory(c, current)


def test_native_tare_reuses_completion_witness_instead_of_redrawing(shipment):
    from document_ocr.synthesis.paddle_render import extra_value
    c, current = shipment
    pair = 'FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE'
    spec = {'owner': 'new-tare', 'equipmentTare': {'authority': 'sampled_inventory',
        'index': 0, 'minPlaces': 0, 'support': {'pairs': {pair: [
            {'kg': '3700', 'sourceIds': ['TRAIN']}, {'kg': '4200', 'sourceIds': ['TRAIN2']}]}}}}
    for identity in ['a', 'b', 'c']:
        assert extra_value(spec, c, current, identity, None, None, None, project_root=Path('.'))[0] == '3700'
    spec['equipmentTare']['support']['pairs'][pair] = [{'kg': '4200', 'sourceIds': ['TRAIN2']}]
    with pytest.raises(ValueError, match='presentation support'):
        extra_value(spec, c, current, 'a', None, None, None, project_root=Path('.'))


def test_new_native_tare_samples_support_when_retained_receipt_has_no_tare(shipment):
    from document_ocr.synthesis.paddle_render import extra_value
    c, current = shipment
    c['physical']['tares'] = []
    c['migrationMissingEquipment']['tareWitnessKg'] = {}
    c['migrationMissingEquipment']['tareSupport'] = []
    repin(c)
    before = copy.deepcopy(c)
    pair = 'FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE'
    spec = {'owner': 'new-tare', 'equipmentTare': {'authority': 'sampled_inventory',
        'index': 0, 'minPlaces': 0, 'support': {'pairs': {pair: [
            {'kg': '3700', 'sourceIds': ['TRAIN']}, {'kg': '4200', 'sourceIds': ['TRAIN2']}]}}}}
    values = {extra_value(spec, c, current, str(i), None, None, None, project_root=Path('.'))[0]
              for i in range(20)}
    assert values == {'3700', '4200'}
    assert c == before
    spec['equipmentTare']['support']['pairs'][pair] = []
    with pytest.raises(ValueError, match='pair support'):
        extra_value(spec, c, current, 'a', None, None, None, project_root=Path('.'))
