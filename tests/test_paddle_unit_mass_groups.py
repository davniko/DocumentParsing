import copy
from decimal import Decimal

import pytest

from document_ocr.synthesis.paddle_render import sampled_unit_mass_groups


def fixture(multiple=False):
    rows = [{"packageQuantity": 7546, "typeCategory": "PACKAGE_CARTON"}]
    if multiple:
        rows.append({"packageQuantity": 3, "typeCategory": "PACKAGE_DRUM"})
        provenance = {
            "declaredPackages": {
                "rows": copy.deepcopy(rows),
                "groups": [
                    {"category": "PACKAGE_CARTON", "netKg": "28"},
                    {"category": "PACKAGE_DRUM", "netKg": "100"},
                ],
                "totals": {"netKg": "211588"},
            }
        }
    else:
        provenance = {
            "freshPrivatePacking": {
                "countedCategory": "PACKAGE_CARTON",
                "publicQuantity": 7546,
                "unitNetKg": "28",
                "netKg": "211288",
            }
        }
    return {"scenario": {"provenance": provenance}}, {
        "target": {"goodsItemDetails": [{"numberAndTypeOfPackages": rows}]}
    }


@pytest.mark.parametrize("multiple", [False, True])
def test_uses_complete_explicit_unit_authority_without_mutation(multiple):
    args = fixture(multiple)
    before = copy.deepcopy(args)
    expected = [("PACKAGE_CARTON", Decimal("28"))]
    if multiple:
        expected.append(("PACKAGE_DRUM", Decimal("100")))
    assert sampled_unit_mass_groups(*args) == expected
    assert args == before


@pytest.mark.parametrize("multiple", [False, True])
def test_rejects_changed_public_packages(multiple):
    candidate, current = fixture(multiple)
    current["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["packageQuantity"] += 1
    with pytest.raises(ValueError, match="retained public"):
        sampled_unit_mass_groups(candidate, current)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", "29"])
def test_rejects_nonpositive_nonfinite_or_inconsistent_mass(value):
    candidate, current = fixture()
    candidate["scenario"]["provenance"]["freshPrivatePacking"]["unitNetKg"] = value
    with pytest.raises(ValueError):
        sampled_unit_mass_groups(candidate, current)


def test_missing_multirow_fact_cannot_fall_through_to_single_unit():
    candidate, current = fixture(True)
    del candidate["scenario"]["provenance"]["declaredPackages"]["groups"][1]["netKg"]
    with pytest.raises(KeyError):
        sampled_unit_mass_groups(candidate, current)


def test_missing_mass_is_not_inferred_from_a_public_total():
    candidate, current = fixture()
    del candidate["scenario"]["provenance"]["freshPrivatePacking"]["unitNetKg"]
    with pytest.raises(KeyError):
        sampled_unit_mass_groups(candidate, current)


def retained_fixture():
    candidate, current = fixture()
    candidate["scenario"]["provenance"] = {
        "method": "registry_goods_train_physical_support_geography_v2",
        "physicalProfile": "registry_form",
        "registryLoadProfile": {"package_category": "PACKAGE_CARTON", "unit_net_kg": "28"},
        "explicitPhysicalFacts": {"netWeight": {"unit": "kilogram", "value": "211288"}},
    }
    return candidate, current


def test_retained_profile_uses_exact_unrounded_sampled_fact():
    args = retained_fixture()
    before = copy.deepcopy(args)
    assert sampled_unit_mass_groups(*args) == [("PACKAGE_CARTON", Decimal("28"))]
    assert args == before


@pytest.mark.parametrize("change", ["count", "category", "unit", "total", "method", "missing"])
def test_retained_profile_rejects_inconsistent_or_absent_authority(change):
    candidate, current = retained_fixture()
    provenance = candidate["scenario"]["provenance"]
    if change == "count":
        current["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["packageQuantity"] += 1
    elif change == "category":
        provenance["registryLoadProfile"]["package_category"] = "PACKAGE_BAG"
    elif change == "unit":
        provenance["explicitPhysicalFacts"]["netWeight"]["unit"] = "metric_tonne"
    elif change == "total":
        provenance["explicitPhysicalFacts"]["netWeight"]["value"] = "211289"
    elif change == "method":
        provenance["method"] = "unsupported"
    else:
        del provenance["registryLoadProfile"]["unit_net_kg"]
    with pytest.raises((ValueError, KeyError)):
        sampled_unit_mass_groups(candidate, current)
