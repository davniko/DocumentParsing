"""Audit classification must handle source captions and short address fragments."""
import pytest

from scripts.analysis.audit_r31_training_readiness import describe_boundary


def boundary(line, name='Example Trading Llc', address='12 New Road', kind='whole_name_shared'):
    return {'name': name, 'address': address, 'occurrences': [
        {'kind': kind, 'boundary': {'text': line}, 'nameStart': {'text': name}}]}


@pytest.mark.parametrize('line,delimiter', [
    ('Example Trading Llc 12 New Road', 'whitespace only'),
    ('Example Trading Llc, 12 New Road', 'punctuation and whitespace'),
    ('Example Trading Llc.12 New Road', 'punctuation without whitespace'),
    ('Example Trading Llc12 New Road', 'no separator'),
    ('Agent: Example Trading Llc 12 New Road', 'whitespace only'),
    ('Example Trading Llc 12 New Road TEL:12345', 'whitespace only'),
])
def test_owned_boundary_delimiters(line, delimiter):
    r = describe_boundary(boundary(line))
    assert r['classification'] == 'matched'
    assert r['delimiter'] == delimiter
    assert r['ending'] == 'legal/business suffix'


@pytest.mark.parametrize('line,address,lead', [
    ('Llc P.O.', 'P.O. Box 91824', 'PO box'),
    ('Llc C/O', 'C/O Another Company 12 New Road', 'care of'),
    ('Llc Warehouse', 'Warehouse 12 New Road', 'word-led'),
])
def test_short_owned_address_beginning(line, address, lead):
    r = describe_boundary(boundary(line, address=address, kind='continuation_shared'))
    assert r['classification'] == 'matched'
    assert r['addressStart'] == lead


def test_incorrect_anchor_not_classified():
    assert describe_boundary(boundary('Example Trading Llc 44 Old Road'))['classification'] == 'needs_junction_review'


def test_unmatched_role_has_no_boundary_type():
    assert describe_boundary({'name': 'Example', 'address': 'Road', 'occurrences': []}) is None
