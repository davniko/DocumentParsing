import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from r30_missing_numeric import complete_numeric, numeric_context
from propose_r30_missing_numeric import select_pending_ids


def test_numeric_selection_excludes_published_batches_before_loading_staging():
    manifest = {"syntheticBatches": ["published"], "syntheticMigration": {"plan": "plan"}}
    data = {"published": {"batch": 1}, "plan": {"batches": [["done"], ["held", "pass"]]},
            2: {"failures": [{"id": "held"}]}}
    assert select_pending_ids(manifest, data.__getitem__) == {"held"}
    assert select_pending_ids(manifest, data.__getitem__, ["held"]) == {"held"}
    for ids in (["done"], ["pass"], ["held", "held"]):
        with pytest.raises(ValueError, match="distinct currently held"):
            select_pending_ids(manifest, data.__getitem__, ids)


def candidate():
    return {
        "joinedRawText": "retained",
        "proof": {"retained": True},
        "target": {"documentPatch": {
            "goodsItemDetails": [{"description": "PAPER", "numberAndTypeOfPackages": [
                {"packageQuantity": 10, "typeCategory": "PACKAGE_BALE"}],
                "splitGoodsPlacement": [{"equipmentIdentifier": "NEW1", "packageQuantity": 4}]}],
            "containerInformation": [{"equipmentIdentifier": "NEW1"}, {"equipmentIdentifier": "NEW2"}],
        }},
        "physical": {"measures": {}},
        "scenario": {"cargo": {"containerInformation": [
            {"equipmentIdentifier": "OLD1"}, {"equipmentIdentifier": "OLD2"}]},
            "provenance": {"method": "registry_goods_train_physical_support_geography_v2",
                "physicalRows": [{"equipmentIdentifier": f"OLD{i}", "quantity": None,
                    "capacity": {"payloadKg": "28000", "volumeM3": "33.2"}, "measures": {}}
                    for i in (1, 2)]}},
    }


def values():
    return {"unknown": False, "basis": "4 and 6 bales at 100 kg and 0.6 m3 per bale",
            "row0_grossWeight": 400, "row1_grossWeight": 600,
            "row0_volume": 2.4, "row1_volume": 3.6, "row0_quantity": 4, "row1_quantity": 6}


def test_only_missing_facts_change_and_inputs_are_not_mutated():
    before = candidate()
    original = copy.deepcopy(before)
    request = numeric_context(before, ["grossWeight", "volume"], True)
    result = complete_numeric(before, request, values())
    assert before == original
    assert result["target"] == before["target"]
    assert result["proof"] == before["proof"]
    assert result["joinedRawText"] == before["joinedRawText"]
    assert result["physical"]["measures"]["grossWeight"]["total"] == "1000"
    assert result["physical"]["measures"]["volume"]["total"] == "6.0"
    assert [r["quantity"] for r in result["scenario"]["provenance"]["physicalRows"]] == [4, 6]


def test_open_equipment_volume_has_no_invented_enclosed_capacity():
    c = candidate()
    for item in c['target']['documentPatch']['containerInformation']:
        item['typeCategory'] = 'OPEN_TOP'
    for row in c['scenario']['provenance']['physicalRows']:
        row['capacity']['volumeM3'] = None
    request = numeric_context(c, ['grossWeight', 'volume'], True)
    assert request['requested']['row0_volume']['upper'] is None
    result = complete_numeric(c, request, values())
    assert result['physical']['measures']['volume']['total'] == '6.0'
    with pytest.raises(ValueError, match='valid bounds'):
        complete_numeric(c, request, {**values(), 'row0_grossWeight': 28001})


@pytest.mark.parametrize('kind', ['GENERAL_PURPOSE', 'REFRIGERATED', None])
def test_unknown_enclosed_capacity_still_fails(kind):
    c = candidate()
    c['target']['documentPatch']['containerInformation'][0]['typeCategory'] = kind
    c['scenario']['provenance']['physicalRows'][0]['capacity']['volumeM3'] = None
    with pytest.raises(ValueError, match='capacity'):
        numeric_context(c, ['volume'])


@pytest.mark.parametrize("key,value", [("unknown", True), ("basis", ""),
    ("row0_grossWeight", True), ("row0_grossWeight", "NaN"),
    ("row0_grossWeight", 28001), ("row0_volume", 33.21),
    ("row0_quantity", 4.0), ("row0_quantity", 0), ("row1_quantity", 7),
    ("unexpected", 1)])
def test_rejects_invalid_or_unconserved_proposals(key, value):
    c = candidate()
    with pytest.raises(ValueError):
        complete_numeric(c, numeric_context(c, ["grossWeight", "volume"], True), {**values(), key: value})


def test_rejects_count_split_that_changes_public_allocation():
    c = candidate()
    with pytest.raises(ValueError, match="public allocation"):
        complete_numeric(c, numeric_context(c, ["grossWeight", "volume"], True),
                         {**values(), "row0_quantity": 5, "row1_quantity": 5})


def test_existing_units_totals_and_row_values_remain_exact():
    c = candidate()
    c["physical"]["measures"]["grossWeight"] = {"unit": "metric_tonne", "total": "1.000"}
    c["scenario"]["provenance"]["physicalRows"][0]["measures"]["grossWeight"] = {
        "unit": "metric_tonne", "total": "1.000", "value": "0.4000"}
    request = numeric_context(c, ["grossWeight"], False)
    assert set(request["requested"]) == {"row1_grossWeight"}
    result = complete_numeric(c, request, {"unknown": False, "basis": "retained remainder", "row1_grossWeight": 600})
    assert result["physical"] == c["physical"]
    assert result["scenario"]["provenance"]["physicalRows"][0] == c["scenario"]["provenance"]["physicalRows"][0]
    assert result["scenario"]["provenance"]["physicalRows"][1]["measures"]["grossWeight"]["value"] == "0.6"
    with pytest.raises(ValueError, match="conserve"):
        complete_numeric(c, request, {"unknown": False, "basis": "wrong", "row1_grossWeight": 601})


def test_new_net_cannot_exceed_known_gross():
    c = candidate()
    c["physical"]["measures"]["grossWeight"] = {"unit": "kilogram", "total": "1000"}
    for row, val in zip(c["scenario"]["provenance"]["physicalRows"], (400, 600)):
        row["measures"]["grossWeight"] = {"unit": "kilogram", "total": "1000", "value": val}
    request = numeric_context(c, ["netWeight"])
    assert request["requested"]["row0_netWeight"]["upper"] == "400"
    with pytest.raises(ValueError, match="valid bounds"):
        complete_numeric(c, request, {"unknown": False, "basis": "invalid", "row0_netWeight": 401, "row1_netWeight": 590})


@pytest.mark.parametrize("key", ["registryLoadProfile", "declaredPackages"])
def test_never_replaces_complete_independent_ledgers(key):
    c = candidate()
    c["scenario"]["provenance"][key] = {}
    with pytest.raises(ValueError, match="projected, not redrawn"):
        numeric_context(c, ["grossWeight"])


def test_reordered_physical_rows_keep_source_id_ownership():
    c = candidate()
    c["scenario"]["provenance"]["physicalRows"].reverse()
    result = complete_numeric(c, numeric_context(c, ["grossWeight", "volume"], True), values())
    assert [r["quantity"] for r in result["scenario"]["provenance"]["physicalRows"]] == [6, 4]


def rounded_candidate():
    c = candidate()
    c['physical']['measures']['grossWeight'] = {
        'unit': 'kilogram', 'total': '1000', 'beforeRounding': '1000.3', 'quantum': '1'}
    c['target']['documentPatch']['goodsItemDetails'][0]['grossWeight'] = {
        'unit': 'kilogram', 'value': 1000}
    c['scenario']['provenance']['physicalRows'][0]['measures']['grossWeight'] = {
        'unit': 'kilogram', 'total': '1000.3', 'value': '400.1'}
    return c


def test_private_precision_is_distinct_from_certified_printed_rounding():
    c = rounded_candidate()
    original = copy.deepcopy(c)
    request = numeric_context(c, ['grossWeight'])
    assert request['knownTotals']['grossWeight'] == '1000.3'
    result = complete_numeric(c, request, {
        'unknown': False, 'basis': 'Exact remaining private mass.', 'row1_grossWeight': '600.2'})
    assert c == original
    assert result['target'] == c['target']
    assert result['physical'] == c['physical']
    assert result['scenario']['provenance']['physicalRows'][1]['measures']['grossWeight'] == {
        'unit': 'kilogram', 'total': '1000.3', 'value': '600.2'}


@pytest.mark.parametrize('change', [
    {'beforeRounding': '1001.3'}, {'quantum': '2'}, {'quantum': '-1'},
    {'beforeRounding': 'NaN'}, {'quantum': 'Infinity'},
])
def test_invalid_rounding_evidence_cannot_hide_changed_totals(change):
    c = rounded_candidate()
    c['physical']['measures']['grossWeight'].update(change)
    with pytest.raises(ValueError, match='rounding certificate'):
        numeric_context(c, ['grossWeight'])


def test_rounded_public_value_must_still_match_its_printed_authority():
    c = rounded_candidate()
    c['target']['documentPatch']['goodsItemDetails'][0]['grossWeight']['value'] = 999
    with pytest.raises(ValueError, match='public measure'):
        numeric_context(c, ['grossWeight'])
