import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/migrations'))
from r30_retained_row_projection import reconcile_profile_rows


def fixture():
    return {'scenario': {'cargo': {'containerInformation': [{'equipmentIdentifier': x} for x in ['A','B']],
        'goodsItemDetails': [{'netWeight': {'value': 1.2, 'unit': 'metric_tonne'}}]},
        'provenance': {'method': 'registry_goods_train_physical_support_geography_v2',
            'registryLoadProfile': {'package_category': 'PACKAGE_CRATE', 'unit_net_kg': '615'},
            'explicitPhysicalFacts': {'netWeight': {'value': '1230', 'unit': 'kilogram'}},
            'physicalRows': [{'equipmentIdentifier': x, 'quantity': 1, 'shareNumerator': 1, 'shareDenominator': 2,
                'measures': {'netWeight': {'value': '0.6', 'total': '1.2', 'unit': 'metric_tonne'}}} for x in ['A','B']]}},
        'physical': {'measures': {'netWeight': {'beforeRounding': '1.23', 'quantum': '0.001', 'total': '1.230', 'unit': 'metric_tonne'}}},
        'target': {'documentPatch': {'goodsItemDetails': [{'numberAndTypeOfPackages': [{'packageQuantity': 2,'typeCategory': 'PACKAGE_CRATE'}],
            'netWeight': {'value': 1.23, 'unit': 'metric_tonne'}}]}}, 'proof': {'edits': []}, 'joinedRawText': 'retained'}


def test_only_private_projection_changes():
    old = fixture(); before = copy.deepcopy(old); new = reconcile_profile_rows(old)
    assert old == before
    for row in new['scenario']['provenance']['physicalRows']:
        assert row['measures']['netWeight']['value'] == '0.615'
        assert row['measures']['netWeight']['total'] == '1.230'
    new['scenario']['provenance']['physicalRows'] = old['scenario']['provenance']['physicalRows']
    assert new == old


@pytest.mark.parametrize('mutation', [
    lambda c: c['scenario']['provenance']['physicalRows'][0]['measures']['netWeight'].update(value='0.61'),
    lambda c: c['scenario']['provenance']['physicalRows'][0].update(equipmentIdentifier='B'),
    lambda c: c['scenario']['provenance']['physicalRows'][0].update(quantity=2),
    lambda c: c['scenario']['provenance']['physicalRows'][0].update(shareNumerator=2),
    lambda c: c['scenario']['provenance']['physicalRows'][0]['measures']['netWeight'].update(total='1.23'),
    lambda c: c['physical']['measures']['netWeight'].update(total='1.2'),
    lambda c: c['scenario']['provenance']['explicitPhysicalFacts']['netWeight'].update(value='1231'),
    lambda c: c['scenario']['provenance']['registryLoadProfile'].update(unit_net_kg='nan'),
    lambda c: c['target']['documentPatch']['goodsItemDetails'][0]['netWeight'].update(value=1.2),
])
def test_conflicting_authority_is_rejected(mutation):
    c = fixture(); mutation(c)
    with pytest.raises(ValueError): reconcile_profile_rows(c)
