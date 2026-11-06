import copy
from decimal import Decimal

import pytest

from document_ocr.synthesis.retained_measure_shares import has_exact_proportional_total


def rows():
    return [{'shareNumerator': 1, 'shareDenominator': 3, 'measures': {'grossWeight': {
        'total': '1', 'value': str(Decimal(1) / 3)}}} for _ in range(3)]


def test_repeating_decimal_certificate_without_tolerance():
    r = rows(); before = copy.deepcopy(r)
    assert sum(Decimal(x['measures']['grossWeight']['value']) for x in r) != 1
    assert has_exact_proportional_total(r, 'grossWeight', Decimal(1))
    assert r == before


@pytest.mark.parametrize('mutation', [
    lambda r: r[0].update(shareNumerator=2),
    lambda r: r[0].update(shareDenominator=0),
    lambda r: r[0].update(shareNumerator=True),
    lambda r: r[0].pop('shareNumerator'),
    lambda r: r[0]['measures']['grossWeight'].update(value='0.3333333333333333333333333334'),
    lambda r: r[0]['measures']['grossWeight'].update(total='1.0001'),
    lambda r: r.pop(),
])
def test_near_equal_or_invalid_proof_is_not_accepted(mutation):
    r = rows(); mutation(r)
    assert not has_exact_proportional_total(r, 'grossWeight', Decimal(1))
