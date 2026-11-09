from copy import deepcopy
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.declared_package_private import (
    combine,
    disjoint_support_wording,
    join_packing_clauses,
    join_unit_net_prefixes,
)
from document_ocr.synthesis.declared_packages import CountedPackage, DeclaredPackages


def test_packing_clause_shares_wrapper_across_disjoint_groups():
    region = {"prefix": "(", "suffix": " ON DECK AT SHIPPER'S RISK"}
    terms = ["(4 DRUMS ON DECK AT SHIPPER'S RISK", "(8 BOXES ON DECK AT SHIPPER'S RISK"]
    assert join_packing_clauses(terms, region) == "(4 DRUMS; 8 BOXES ON DECK AT SHIPPER'S RISK"
    assert join_packing_clauses(terms[:1], region) == terms[0]
    assert join_packing_clauses(["4 DRUMS", "8 BOXES"], {}) == "4 DRUMS\n8 BOXES"
    with pytest.raises(ValueError, match="wrapper"):
        join_packing_clauses(["unowned composition"], region)
    with pytest.raises(ValueError, match="nonempty"):
        join_packing_clauses(["()"], {"prefix": "(", "suffix": ")"})


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


def test_disjoint_unit_prefixes_do_not_create_accidental_nested_reading(case):
    bp, ledger, original, children = case
    children[0]["surfaces"] = {"held-numeric:0": ["7 BOXES X 2 KG NET EACH OF "]}
    children[1]["surfaces"] = {"held-numeric:0": ["8 DRUMS X 10 KG NET EACH OF "]}
    result = combine(bp, {"regions": [{"role": "unit_net_prefix"}]}, ledger, children, original)
    assert result["surfaces"]["held-numeric:0"] == [
        "7 BOXES X 2 KG NET EACH + 8 DRUMS X 10 KG NET EACH OF "
    ]
    assert join_unit_net_prefixes(["7 BOXES X 2 KG NET EACH OF "]) == "7 BOXES X 2 KG NET EACH OF "
    with pytest.raises(ValueError, match="connector"):
        join_unit_net_prefixes(["unowned text", "8 DRUMS OF "])


@pytest.mark.parametrize("role", ["support_mass_breakdown", "component_mass_pair"])
def test_mass_breakdowns_explicitly_name_the_disjoint_counted_group(case, role):
    bp, ledger, original, children = case
    children[0]["surfaces"] = {"held-numeric:0": ["TOTAL SUPPORTS: 3 PALLETS"]}
    children[1]["surfaces"] = {"held-numeric:0": ["TOTAL SHIPPING GROSS WEIGHT: 96 KG"]}
    result = combine(
        bp,
        {"regions": [{"role": role}]},
        ledger,
        children,
        original,
        noun_renderer=lambda category, count: category,
    )
    assert result["surfaces"]["held-numeric:0"] == [
        "7 PACKAGE_BOX: TOTAL SUPPORTS: 3 PALLETS\n"
        "8 PACKAGE_DRUM_STEEL: TOTAL SHIPPING GROSS WEIGHT: 96 KG"
    ]


def test_private_supports_are_not_confused_with_another_declared_group(case):
    bp, ledger, original, children = case
    children[0]["relation"] = "NONE"
    children[1]["relation"] = "ON_SUPPORT"
    children[1]["surfaces"] = {"held-numeric:0": ["8 DRUMS ON 7 SUPPORT BOXES"]}
    children[1]["wordingAppends"] = {"product": "8 DRUMS ON 7 SUPPORT BOXES; 10 KG NET PER DRUM."}
    result = combine(bp, {"regions": [{"role": "packing_clause"}]}, ledger, children, original)
    suffix = " SUPPORT UNITS ARE ADDITIONAL TO THE DECLARED PACKAGE GROUPS."
    assert result["surfaces"]["held-numeric:0"][0].endswith("8 DRUMS ON 7 SUPPORT BOXES" + suffix)
    assert result["wordingAppends"]["product"].endswith("10 KG NET PER DRUM." + suffix)
    assert result["publicQuantity"] == 15
    assert result["grossKg"] == "117"
    assert children[1]["surfaces"]["held-numeric:0"] == ["8 DRUMS ON 7 SUPPORT BOXES"]
    for relation, count in (("NONE", 2), ("SUPPORTS", 2), ("CONTAINS", 2), ("ON_SUPPORT", 1)):
        assert disjoint_support_wording("unchanged", {"relation": relation}, count) == "unchanged"


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


@pytest.mark.parametrize("relations", [("NONE", "SUPPORTS"), ("NONE", "NONE")])
def test_negative_support_statement_is_scoped_to_its_disjoint_group(case, relations):
    bp, ledger, original, children = case
    for child, relation in zip(children, relations, strict=True):
        child["relation"] = relation
        child["surfaces"] = {
            "held-numeric:0": [
                "NO SUPPORT UNITS DECLARED" if relation == "NONE" else "24 CARTONS ON 8 DRUMS"
            ]
        }
    result = combine(
        bp,
        {"regions": [{"role": "support_statement"}]},
        ledger,
        children,
        original,
        noun_renderer=lambda category, count: {
            "PACKAGE_BOX": "BOXES",
            "PACKAGE_DRUM_STEEL": "STEEL DRUMS",
        }[category],
    )
    text = result["surfaces"]["held-numeric:0"][0]
    assert text.startswith("7 BOXES: NO SUPPORT UNITS DECLARED\n")
    assert text.endswith(
        "24 CARTONS ON 8 DRUMS"
        if relations[1] == "SUPPORTS"
        else "8 STEEL DRUMS: NO SUPPORT UNITS DECLARED"
    )
    assert result["publicQuantity"] == 15


def test_scoped_negative_support_never_uses_an_unconfigured_noun(case):
    bp, ledger, original, children = case
    children[0]["relation"] = "NONE"
    with pytest.raises(ValueError, match="noun renderer"):
        combine(bp, {"regions": [{"role": "support_statement"}]}, ledger, children, original)


def test_per_unit_nested_clauses_identify_their_disjoint_outer_group(case):
    bp, ledger, original, children = case
    children[0]["surfaces"] = {"held-numeric:0": ["PACKING: 4 DRUM / BOX; 2 KG NET / BOX"]}
    children[1]["surfaces"] = {"held-numeric:0": ["PACKING: 1 BAG / DRUM; 10 KG NET / DRUM"]}
    result = combine(
        bp,
        {"regions": [{"role": "unit_inner_packing"}]},
        ledger,
        children,
        original,
        noun_renderer=lambda category, count: {
            "PACKAGE_BOX": "BOXES",
            "PACKAGE_DRUM_STEEL": "STEEL DRUMS",
        }[category],
    )
    assert result["surfaces"]["held-numeric:0"] == [
        "7 BOXES: PACKING: 4 DRUM / BOX; 2 KG NET / BOX\n"
        "8 STEEL DRUMS: PACKING: 1 BAG / DRUM; 10 KG NET / DRUM"
    ]
    assert result["publicQuantity"] == 15


def test_per_unit_wording_append_names_its_counted_group(case):
    bp, ledger, original, children = case
    for value, child in zip(("2", "10"), children, strict=True):
        child["wordingAppends"] = {"product": f"NET CONTENT PER UNIT: {value} KG."}
    before = deepcopy(children)
    certificate = {"regions": [], "wordingOwners": [{"key": "product", "append": "dual_unit_net"}]}
    result = combine(
        bp,
        certificate,
        ledger,
        children,
        original,
        noun_renderer=lambda category, count: (
            "BOXES" if category == "PACKAGE_BOX" else "STEEL DRUMS"
        ),
    )
    assert result["wordingAppends"]["product"] == (
        "7 BOXES: NET CONTENT PER UNIT: 2 KG. 8 STEEL DRUMS: NET CONTENT PER UNIT: 10 KG."
    )
    assert children == before
    assert result["netKg"] == "94"
    with pytest.raises(ValueError, match="scoped per-unit append"):
        combine(bp, certificate, ledger, children, original)


def test_complete_counted_append_does_not_get_a_redundant_prefix(case):
    bp, ledger, original, children = case
    children[0]["wordingAppends"] = {"product": "7 BOXES; 2 KG NET PER BOX."}
    children[1]["wordingAppends"] = {"product": "8 STEEL DRUMS; 10 KG NET PER DRUM."}
    certificate = {"regions": [], "wordingOwners": [{"key": "product", "append": "count_unit_net"}]}
    result = combine(bp, certificate, ledger, children, original)
    assert result["wordingAppends"]["product"] == (
        "7 BOXES; 2 KG NET PER BOX. 8 STEEL DRUMS; 10 KG NET PER DRUM."
    )


def test_dual_unit_region_identifies_distinct_groups_even_with_the_same_noun(case):
    bp, ledger, original, children = case
    for value, child in zip(("2", "10"), children, strict=True):
        child["surfaces"] = {"held-numeric:0": [f"NET CONTENT PER PACKAGE: {value} KG."]}
    certificate = {"regions": [{"role": "dual_unit_net_statement"}]}
    result = combine(
        bp,
        certificate,
        ledger,
        children,
        original,
        noun_renderer=lambda category, count: "PACKAGES",
    )
    assert result["surfaces"]["held-numeric:0"] == [
        "7 PACKAGES: NET CONTENT PER PACKAGE: 2 KG.\n8 PACKAGES: NET CONTENT PER PACKAGE: 10 KG."
    ]
    assert result["netKg"] == "94" and result["publicQuantity"] == 15
    with pytest.raises(ValueError, match="noun renderer"):
        combine(bp, certificate, ledger, children, original)


def test_single_dual_unit_region_is_unchanged(case):
    bp, ledger, original, children = case
    children[1]["surfaces"] = {}
    children[0]["surfaces"] = {"held-numeric:0": ["NET CONTENT PER BOX: 2 KG."]}
    result = combine(
        bp, {"regions": [{"role": "dual_unit_net_statement"}]}, ledger, children, original
    )
    assert result["surfaces"]["held-numeric:0"] == ["NET CONTENT PER BOX: 2 KG."]
