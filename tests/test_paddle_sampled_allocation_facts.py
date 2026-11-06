import copy

import pytest

from document_ocr.synthesis.paddle_render import sampled_allocation_fact, sampled_shipment_fact, sampled_allocation_quantity, sampled_equipment_groups, sampled_equipment

SPEC = {"field": "grossKg", "sourceEquipmentIdentifier": "A"}


def test_private_allocation_preserves_explicit_pre_rounding_total():
    c = retained()
    c['physical']['measures']['grossWeight'].update(total='30', beforeRounding='29.9', quantum='1')
    rows = c['scenario']['provenance']['physicalRows']
    for row, value in zip(rows, ['19.9', '10'], strict=True):
        row['measures']['grossWeight'].update(total='29.9', value=value)
    assert sampled_allocation_fact(SPEC, c)['value'] == 10
    assert sampled_allocation_fact({**SPEC, 'sourceEquipmentIdentifier': 'B'}, c)['value'] == __import__('decimal').Decimal('19.9')
    for field, value in [('beforeRounding', '29.8'), ('quantum', '0.1'), ('quantum', '-1'), ('quantum', 'NaN')]:
        bad = copy.deepcopy(c)
        bad['physical']['measures']['grossWeight'][field] = value
        with pytest.raises(ValueError):
            sampled_allocation_fact(SPEC, bad)
    bad = copy.deepcopy(c)
    del bad['physical']['measures']['grossWeight']['quantum']
    with pytest.raises(ValueError):
        sampled_allocation_fact(SPEC, bad)


def test_retained_equipment_requires_matching_sampled_public_classification():
    candidate = retained()
    inventory = [{"equipmentIdentifier": "NEW_A", "sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}]
    candidate["target"] = {"documentPatch": {"containerInformation": copy.deepcopy(inventory)}}
    current = {"target": {"containerInformation": inventory}}
    spec = {"index": 0, "authority": "sampled_inventory"}
    assert sampled_equipment(spec, candidate, current) == inventory[0]
    for field in ("equipmentIdentifier", "sizeCategory", "typeCategory"):
        altered = copy.deepcopy(candidate)
        altered["target"]["documentPatch"]["containerInformation"][0][field] = "OTHER"
        with pytest.raises(ValueError):
            sampled_equipment(spec, altered, current)
    del candidate["target"]["documentPatch"]["containerInformation"][0]["typeCategory"]
    with pytest.raises(ValueError, match="explicit matching"):
        sampled_equipment(spec, candidate, current)


def test_sampled_inventory_keeps_independent_private_evidence_and_rejects_unknown_method():
    candidate = retained()
    inventory = [{"equipmentIdentifier": "NEW_A"}]
    candidate["scenario"]["cargo"]["containerInformation"] = [{"equipmentIdentifier": "OLD_A"}]
    candidate["target"] = {"documentPatch": {"containerInformation": copy.deepcopy(inventory)}}
    current = {"target": {"containerInformation": inventory}}
    p = candidate["scenario"]["provenance"]
    p["method"] = "independent_full_registry_flat_v1"
    p["independentDraw"] = {"equipment": [{"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}]}
    spec = {"index": 0, "authority": "sampled_inventory"}
    assert sampled_equipment(spec, candidate, current)["sizeCategory"] == "TWENTY_FOOT_STANDARD_HEIGHT"
    assert "sizeCategory" not in inventory[0]
    p["method"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        sampled_equipment(spec, candidate, current)


def retained():
    return {
        "physical": {"measures": {"grossWeight": {"unit": "kilogram", "total": "30"}}},
        "scenario": {
            "cargo": {
                "containerInformation": [{"equipmentIdentifier": "A"}, {"equipmentIdentifier": "B"}]
            },
            "provenance": {
                "method": "registry_goods_train_physical_support_geography_v2",
                "physicalRows": [
                    {
                        "equipmentIdentifier": identity,
                        "measures": {
                            "grossWeight": {"value": value, "unit": "kilogram", "total": "30"}
                        },
                    }
                    for identity, value in [("B", "20"), ("A", "10")]
                ],
            },
        },
    }


def test_retained_identity_not_list_position_and_no_mutation():
    candidate = retained()
    before = copy.deepcopy(candidate)
    assert sampled_allocation_fact(SPEC, candidate) == {"value": 10, "unit": "kilogram"}
    assert candidate == before


@pytest.mark.parametrize(
    "case", ["duplicate", "missing", "unit", "total", "sum", "negative", "nan", "method"]
)
def test_invalid_retained_evidence_never_falls_back(case):
    candidate = retained()
    provenance = candidate["scenario"]["provenance"]
    rows = provenance["physicalRows"]
    fact = rows[0]["measures"]["grossWeight"]
    if case == "duplicate":
        rows[0]["equipmentIdentifier"] = "A"
    elif case == "missing":
        rows.pop()
    elif case == "unit":
        fact["unit"] = "tonne"
    elif case == "total":
        fact["total"] = "99"
    elif case == "sum":
        fact["value"] = "21"
    elif case == "negative":
        fact["value"] = "-20"
    elif case == "nan":
        fact["value"] = "NaN"
    else:
        provenance["method"] = "unknown"
    with pytest.raises(ValueError):
        sampled_allocation_fact(SPEC, candidate)


def test_independent_contract_and_missing_row_are_explicit():
    candidate = retained()
    provenance = candidate["scenario"]["provenance"]
    provenance["method"] = "independent_full_registry_flat_v1"
    provenance["independentDraw"] = {"allocations": [{"row": 0, "grossKg": "12"}]}
    assert sampled_allocation_fact(SPEC, candidate) == {"value": "12", "unit": "kilogram"}
    provenance["independentDraw"]["allocations"] = []
    with pytest.raises(ValueError, match="unique independent"):
        sampled_allocation_fact(SPEC, candidate)
    del provenance["independentDraw"]
    with pytest.raises(KeyError):
        sampled_allocation_fact(SPEC, candidate)


def test_retained_metric_tonnes_convert_without_resampling_or_rounding():
    candidate = retained()
    candidate["physical"]["measures"]["grossWeight"]["unit"] = "metric_tonne"
    for row in candidate["scenario"]["provenance"]["physicalRows"]:
        row["measures"]["grossWeight"]["unit"] = "metric_tonne"
    assert sampled_allocation_fact(SPEC, candidate) == {"value": 10000, "unit": "kilogram"}
    assert sampled_shipment_fact({"field": "grossWeight", "unit": "kilogram"}, candidate) == {
        "value": 30000, "unit": "kilogram",
    }


def test_total_contract_never_substitutes_other_receipt_fields():
    candidate = retained()
    spec = {"field": "grossWeight", "unit": "kilogram"}
    assert sampled_shipment_fact(spec, candidate) == {"value": 30, "unit": "kilogram"}
    provenance = candidate["scenario"]["provenance"]
    provenance["method"] = "independent_full_registry_flat_v1"
    with pytest.raises(KeyError):
        sampled_shipment_fact(spec, candidate)
    provenance["explicitPhysicalFacts"] = {"grossWeight": {"value": 41, "unit": "kilogram"}}
    candidate['target'] = {'documentPatch': {'goodsItemDetails': [
        {'grossWeight': {'value': 41, 'unit': 'kilogram'}}]}}
    # An existing printed measurement must carry matching rounding evidence.
    with pytest.raises(KeyError):
        sampled_shipment_fact(spec, candidate)
    candidate['physical']['measures']['grossWeight'] = {
        'beforeRounding': '41', 'quantum': '0.1', 'total': '41.0', 'unit': 'kilogram'
    }
    assert sampled_shipment_fact(spec, candidate) == {"value": 41, "unit": "kilogram"}
    provenance["method"] = "unrecognized"
    with pytest.raises(ValueError, match="unsupported"):
        sampled_shipment_fact(spec, candidate)
    provenance["method"] = "registry_goods_train_physical_support_geography_v2"
    with pytest.raises(ValueError, match="dimension"):
        sampled_shipment_fact({**spec, "unit": "cubic_metre"}, candidate)
    for bad in ["NaN", "Infinity", "0", "-1"]:
        candidate["physical"]["measures"]["grossWeight"]["total"] = bad
        with pytest.raises(ValueError, match="finite and positive"):
            sampled_shipment_fact(spec, candidate)


def test_retained_counts_preserve_identity_conservation_and_public_presence():
    candidate = retained()
    public = [{"equipmentIdentifier": "NEW_A"}, {"equipmentIdentifier": "NEW_B"}]
    candidate["target"] = {"documentPatch": {"containerInformation": public}}
    current = {"target": {"containerInformation": copy.deepcopy(public), "goodsItemDetails": [{
        "numberAndTypeOfPackages": [{"packageQuantity": 3}],
        "splitGoodsPlacement": [{"equipmentIdentifier": "NEW_A"}, {"equipmentIdentifier": "NEW_B", "packageQuantity": 2}],
    }]}}
    rows = candidate["scenario"]["provenance"]["physicalRows"]
    for row in rows:
        row["quantity"] = 1 if row["equipmentIdentifier"] == "A" else 2
    assert sampled_allocation_quantity(0, candidate, current) == 1
    assert sampled_allocation_quantity(1, candidate, current) == 2
    assert "packageQuantity" not in current["target"]["goodsItemDetails"][0]["splitGoodsPlacement"][0]
    rows[0]["quantity"] = None
    with pytest.raises(ValueError, match="explicit positive"):
        sampled_allocation_quantity(0, candidate, current)
    rows[0]["quantity"] = 3
    with pytest.raises(ValueError, match="conserve"):
        sampled_allocation_quantity(0, candidate, current)
    rows[0]["quantity"], rows[1]["quantity"] = 1, 2
    with pytest.raises(ValueError, match="conflicts"):
        sampled_allocation_quantity(0, candidate, current)


def test_reviewed_missing_total_contract_is_explicit_and_cannot_overwrite():
    candidate = retained()
    spec = {"field": "volume", "unit": "cubic_metre"}
    candidate["scenario"]["provenance"]["explicitPhysicalFacts"] = {
        "volume": {"value": 12.5, "unit": "cubic_metre"}
    }
    with pytest.raises(KeyError):
        sampled_shipment_fact(spec, candidate)
    candidate["migrationMissingMeasures"] = {"proposal": {"sha256": "a"}, "review": {"sha256": "b"}}
    assert sampled_shipment_fact(spec, candidate)["value"] == 12.5
    candidate["physical"]["measures"]["volume"] = {"total": 10, "unit": "cubic_metre"}
    with pytest.raises(ValueError, match="overwrites"):
        sampled_shipment_fact(spec, candidate)
    candidate["migrationMissingMeasures"]["review"] = None
    with pytest.raises(ValueError, match="provenance"):
        sampled_shipment_fact(spec, candidate)


def test_registry_profile_totals_are_derived_and_checked_not_fallbacks():
    candidate = retained()
    p = candidate["scenario"]["provenance"]
    p["registryLoadProfile"] = {"package_category": "PACKAGE_SACK", "unit_gross_kg": "26"}
    p["explicitPhysicalFacts"] = {"grossWeight": {"value": "260", "unit": "kilogram"}}
    candidate["target"] = {"documentPatch": {"goodsItemDetails": [{"numberAndTypeOfPackages": [
        {"typeCategory": "PACKAGE_SACK", "packageQuantity": 10}
    ]}]}}
    candidate['physical']['measures']['grossWeight'] = {
        'beforeRounding': '260', 'quantum': '0.1', 'total': '260.0', 'unit': 'kilogram'
    }
    spec = {"field": "grossWeight", "unit": "kilogram"}
    assert sampled_shipment_fact(spec, candidate)["value"] == 260
    p["explicitPhysicalFacts"]["grossWeight"]["value"] = "261"
    with pytest.raises(ValueError, match="registry profile"):
        sampled_shipment_fact(spec, candidate)


def test_registry_private_allocation_projects_existing_profile_not_new_numbers():
    candidate = retained()
    p = candidate['scenario']['provenance']
    p['registryLoadProfile'] = {'package_category': 'PACKAGE_CASE_WOODEN', 'unit_volume_m3': '1.5'}
    p['explicitPhysicalFacts'] = {'volume': {'value': '60', 'unit': 'cubic_metre'}}
    candidate['physical']['measures']['volume'] = {'total': '60', 'unit': 'cubic_metre'}
    candidate['target'] = {'documentPatch': {'goodsItemDetails': [{'numberAndTypeOfPackages': [
        {'packageQuantity': 40, 'typeCategory': 'PACKAGE_CASE_WOODEN'}]}]}}
    p['physicalRows'][0]['quantity'] = 10
    p['physicalRows'][1]['quantity'] = 30
    before = copy.deepcopy(candidate)
    spec = {'field': 'volumeM3', 'sourceEquipmentIdentifier': 'A'}
    assert sampled_allocation_fact(spec, candidate) == {'value': 45, 'unit': 'cubic_metre'}
    assert sampled_allocation_fact({**spec, 'sourceEquipmentIdentifier': 'B'}, candidate)['value'] == 15
    assert candidate == before
    for kind in ['identity', 'count', 'zero', 'unit', 'total', 'existing', 'category', 'missing']:
        bad = copy.deepcopy(candidate)
        bp = bad['scenario']['provenance']
        if kind == 'identity': bp['physicalRows'][0]['equipmentIdentifier'] = 'A'
        elif kind == 'count': bp['physicalRows'][0]['quantity'] = 11
        elif kind == 'zero': bp['physicalRows'][0]['quantity'] = 0
        elif kind == 'unit': bp['explicitPhysicalFacts']['volume']['unit'] = 'kilogram'
        elif kind == 'total': bp['explicitPhysicalFacts']['volume']['value'] = '59'
        elif kind == 'existing': bp['physicalRows'][0]['measures']['volume'] = {'value': '16', 'total': '60', 'unit': 'cubic_metre'}
        elif kind == 'category': bp['registryLoadProfile']['package_category'] = 'PACKAGE_BAG'
        else: del bp['registryLoadProfile']['unit_volume_m3']
        with pytest.raises((ValueError, KeyError)):
            sampled_allocation_fact(spec, bad)


def test_sampled_total_reuses_only_verified_printed_rounding():
    from decimal import Decimal
    candidate = retained()
    p = candidate['scenario']['provenance']
    p['method'] = 'independent_full_registry_flat_v1'
    p['explicitPhysicalFacts'] = {'volume': {'value': '14.7500000000000118', 'unit': 'cubic_metre'}}
    measure = {'beforeRounding': '14.7500000000000118', 'quantum': '0.0001',
               'total': '14.7500', 'unit': 'cubic_metre'}
    candidate['physical']['measures']['volume'] = measure
    candidate['target'] = {'documentPatch': {'goodsItemDetails': [
        {'volume': {'value': 14.75, 'unit': 'cubic_metre'}}]}}
    spec = {'field': 'volume', 'unit': 'cubic_metre'}
    assert sampled_shipment_fact(spec, candidate) == {'value': Decimal('14.7500'), 'unit': 'cubic_metre'}
    for key, value in [('beforeRounding', '14.8'), ('total', '14.7501'), ('quantum', '0')]:
        changed = copy.deepcopy(candidate)
        changed['physical']['measures']['volume'][key] = value
        with pytest.raises(ValueError):
            sampled_shipment_fact(spec, changed)
    changed = copy.deepcopy(candidate)
    changed['target']['documentPatch']['goodsItemDetails'][0]['volume']['value'] = 14.8
    with pytest.raises(ValueError, match='retained public target'):
        sampled_shipment_fact(spec, changed)
    changed = copy.deepcopy(candidate)
    del changed['target']['documentPatch']['goodsItemDetails'][0]['volume']
    assert sampled_shipment_fact(spec, changed)['value'] == Decimal('14.7500000000000118')
    del candidate['physical']['measures']['volume']
    assert sampled_shipment_fact(spec, candidate)['value'] == Decimal('14.7500000000000118')


def test_newly_exposed_different_unit_preserves_planning_precision():
    from decimal import Decimal
    candidate = retained()
    candidate['scenario']['provenance'].update(
        method='independent_full_registry_flat_v1',
        explicitPhysicalFacts={'grossWeight': {'value': '33247.412', 'unit': 'kilogram'}})
    candidate['physical']['measures']['grossWeight'] = {
        'beforeRounding': '33.247412', 'quantum': '0.001',
        'total': '33.247', 'unit': 'metric_tonne'}
    candidate['target'] = {'documentPatch': {'goodsItemDetails': [
        {'grossWeight': {'value': 33.247, 'unit': 'metric_tonne'}}]}}
    assert sampled_shipment_fact({'field': 'grossWeight', 'unit': 'kilogram'}, candidate)['value'] == Decimal('33247.412')
    assert sampled_shipment_fact({'field': 'grossWeight', 'unit': 'metric_tonne'}, candidate)['value'] == Decimal('33.247')


def equipment_summary():
    candidate = retained()
    candidate["proof"] = {"sourceSha256": "source"}
    groups = [
        {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE", "count": 1},
        {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE", "count": 3},
    ]
    candidate["scenario"]["provenance"]["equipmentMix"] = {
        "certificate": {"source_sha256": "source", "members": groups}
    }
    current = {"target": {"containerInformation": [{"equipmentIdentifier": str(i)} for i in range(4)]}}
    return candidate, current, groups


def test_summary_only_evidence_does_not_create_per_container_classifications():
    candidate, current, groups = equipment_summary()
    before = copy.deepcopy((candidate, current))
    assert sampled_equipment_groups(candidate, current) == groups
    assert (candidate, current) == before


@pytest.mark.parametrize("case", ["duplicate_id", "wrong_source", "wrong_total", "bool_count", "negative_count", "duplicate_pair", "partial_type", "type_conflict", "unknown_method"])
def test_summary_rejects_conflicting_or_unproven_evidence(case):
    candidate, current, groups = equipment_summary()
    public = current["target"]["containerInformation"]
    provenance = candidate["scenario"]["provenance"]
    if case == "duplicate_id":
        public[0]["equipmentIdentifier"] = public[1]["equipmentIdentifier"]
    elif case == "wrong_source":
        candidate["proof"]["sourceSha256"] = "different"
    elif case == "wrong_total":
        groups[0]["count"] = 2
    elif case == "bool_count":
        groups[0]["count"] = True
    elif case == "negative_count":
        groups[0]["count"] = -1
    elif case == "duplicate_pair":
        groups[1]["sizeCategory"] = groups[0]["sizeCategory"]
    elif case == "partial_type":
        public[0]["sizeCategory"] = groups[0]["sizeCategory"]
    elif case == "type_conflict":
        for row in public[:2]:
            row.update({k: v for k, v in groups[0].items() if k != "count"})
    else:
        provenance["method"] = "unknown"
    with pytest.raises(ValueError):
        sampled_equipment_groups(candidate, current)
