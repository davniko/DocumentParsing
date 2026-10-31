from copy import deepcopy
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.declared_package_private import combine
from document_ocr.synthesis.declared_packages import CountedPackage, DeclaredPackages


@pytest.fixture
def case():
    groups = (
        CountedPackage("PACKAGE_BOX", D(2), D(3), D(".1")),
        CountedPackage("PACKAGE_DRUM_STEEL", D(10), D(12), D(".3")),
    )
    ledger = DeclaredPackages(groups, ((5, 0), (2, 8)), ((D(1000), D(100)),) * 2)
    original = {"sourceDocumentId": "s", "scopeLedgers": {}, "facts": {"old": 1}}
    children = []
    for count in ledger.counts:
        children.append(
            {
                "surfaces": {"held-numeric:0": [f"{count} outer contains {count * 3} inner"]},
                "facts": {},
                "wordingFacts": [],
                "wordingAppends": {},
                "presentationDecisions": [],
            }
        )
    return SimpleNamespace(ownership_data={}), ledger, original, children


def test_complete_nested_groups_stay_disjoint_and_do_not_add_inner_counts(case):
    bp, ledger, original, children = case
    before = deepcopy((original, children))
    result = combine(bp, {"regions": [{"role": "packing_clause"}]}, ledger, children, original)
    assert result["surfaces"]["held-numeric:0"] == [
        "7 outer contains 21 inner\n8 outer contains 24 inner"
    ]
    assert result["publicQuantity"] == 15
    assert result["netKg"] == "94"
    assert result["grossKg"] == "117"
    assert result["unitNetKg"] is None
    assert (original, children) == before


@pytest.mark.parametrize("fact,expected", [("netKg", "94.00"), ("grossKg", "117.00")])
def test_bare_total_cell_sums_group_measures_not_average_units(case, fact, expected):
    bp, ledger, original, children = case
    region = {
        "role": "private_measurement",
        "fact": fact,
        "baseline": "100",
        "source_text": "100.00",
    }
    result = combine(bp, {"regions": [region]}, ledger, children, original)
    assert result["surfaces"]["held-numeric:0"] == [expected]


def test_absent_group_in_container_is_not_a_phantom_private_clause(case):
    bp, ledger, original, children = case
    children[1]["surfaces"] = {}
    region = {
        "role": "container_net_mass",
        "scope": {"kind": "container", "index": 0},
        "baselineKg": "10",
        "source_text": "NET 10.00 KG",
    }
    result = combine(bp, {"regions": [region]}, ledger, children, original)
    assert result["surfaces"]["held-numeric:0"] == ["NET 10.00 KG"]


def test_component_measures_use_the_same_partition_as_marks_counts(case):
    bp, ledger, original, children = case
    children[0]["facts"] = {"marks": {"quantities": [2, 5]}}
    children[1]["facts"] = {"marks": {"quantities": [5, 3]}}
    region = {
        "role": "component_measurement",
        "fact": "grossKg",
        "factGroup": "marks",
        "componentIndex": 0,
        "baseline": "20",
        "source_text": "20.00",
    }
    result = combine(bp, {"regions": [region]}, ledger, children, original)
    assert result["surfaces"]["held-numeric:0"] == ["66.00"]


def test_bare_per_unit_measure_must_not_be_summed(case):
    bp, ledger, original, children = case
    region = {
        "role": "private_measurement",
        "fact": "unitNetKg",
        "baseline": "2",
        "source_text": "2",
    }
    with pytest.raises(ValueError, match="per-unit"):
        combine(bp, {"regions": [region]}, ledger, children, original)


def test_missing_owned_scope_fails_closed(case):
    bp, ledger, original, children = case
    for child in children:
        child["surfaces"] = {}
    with pytest.raises(ValueError, match="no positive"):
        combine(bp, {"regions": [{"role": "packing_clause"}]}, ledger, children, original)
