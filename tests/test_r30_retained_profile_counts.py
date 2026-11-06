import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/migrations"))
from r30_retained_profile_counts import complete_profile_counts


def candidate():
    return {"scenario": {"cargo": {"containerInformation": [{"equipmentIdentifier": "A"}, {"equipmentIdentifier": "B"}]},
        "provenance": {"method": "registry_goods_train_physical_support_geography_v2",
            "registryLoadProfile": {"package_category": "PACKAGE_SKID", "unit_gross_kg": "600", "unit_net_kg": "550", "unit_volume_m3": "1.25"},
            "physicalRows": [{"equipmentIdentifier": x, "quantity": None, "measures": {
                "grossWeight": {"value": "0.6", "total": "1.2", "unit": "metric_tonne"},
                "netWeight": {"value": "550", "total": "1100", "unit": "kilogram"}}} for x in ("A", "B")]}},
        "target": {"documentPatch": {"goodsItemDetails": [{"numberAndTypeOfPackages": [{"packageQuantity": 2, "typeCategory": "PACKAGE_SKID"}]}]}},
        "joinedRawText": "unchanged", "proof": {"edits": []}}


def test_complete_counts_preserves_retained_facts():
    c = candidate(); before = copy.deepcopy(c); result = complete_profile_counts(c)
    assert c == before
    assert [r["quantity"] for r in result["scenario"]["provenance"]["physicalRows"]] == [1, 1]
    for row in result["scenario"]["provenance"]["physicalRows"]: row["quantity"] = None
    assert result == before


@pytest.mark.parametrize("change", [
    lambda c: c["scenario"]["provenance"]["physicalRows"][0]["measures"]["grossWeight"].update(value="0.9"),
    lambda c: c["scenario"]["provenance"]["physicalRows"][0]["measures"]["netWeight"].update(value="1100"),
    lambda c: c["scenario"]["provenance"]["physicalRows"][0].update(quantity=2),
    lambda c: c["scenario"]["provenance"]["physicalRows"][0].update(equipmentIdentifier="B"),
    lambda c: c["scenario"]["provenance"]["physicalRows"][0].update(measures={}),
    lambda c: c["scenario"]["provenance"]["registryLoadProfile"].update(unit_gross_kg="0"),
    lambda c: c["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0].update(typeCategory="PACKAGE_BALE"),
])
def test_fail_closed(change):
    c = candidate(); change(c)
    with pytest.raises(ValueError): complete_profile_counts(c)
