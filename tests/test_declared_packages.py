"""Production package ledger invariants, including previously probed edge cases."""

from copy import deepcopy
from decimal import Decimal as D

import pytest

from document_ocr.synthesis.curated_scenarios import CandidateIncompatibility
from document_ocr.synthesis.declared_packages import (
    CountedPackage,
    DeclaredPackages,
    equalize_groups,
    sample_groups,
)


@pytest.mark.parametrize("seed", range(100))
def test_equal_load_campaign_conserves_multiple_groups_and_partial_visibility(seed):
    original = sample_groups(
        [group(), group("PACKAGE_BAG", multiple=2)],
        [(D(18000), D(28)), (D(20000), D(40)), (D(18000), D(28))],
        seed=seed,
        key="equal-campaign",
        utilization=(D(".2"), D(".4")),
    )
    equal = equalize_groups(original, minimum_group_count=5, minimum_load_counts=[1, 3, 1])
    assert len({sum(row) for row in equal.matrix}) == 1
    assert equal.groups == original.groups
    assert equal.capacities == original.capacities
    assert all(row[1] % 2 == 0 and min(row) >= 3 for row in equal.matrix)
    assert DeclaredPackages.from_receipt(equal.receipt()) == equal
    placements = equal.project_placements(
        [{"equipmentIdentifier": "A"}, {"equipmentIdentifier": "B", "packageQuantity": 3}],
        ["A", "B", "C"],
    )
    assert placements[0] == {"equipmentIdentifier": "A"}
    assert placements[1]["packageQuantity"] == sum(equal.matrix[1])


def test_equal_load_does_not_overfill_smaller_mixed_equipment():
    ledger = DeclaredPackages((group(),), ((1,), (100,)), ((D(150), D(1)), (D(20000), D(40))))
    with pytest.raises(CandidateIncompatibility):
        equalize_groups(ledger)


def test_equal_installed_systems_do_not_gain_shared_contents():
    ledger = DeclaredPackages(
        (group("PACKAGE_FLEXITANK"), group()),
        ((1, 0), (0, 2)),
        ((D(20000), D(40)),) * 2,
    )
    with pytest.raises(CandidateIncompatibility, match="installed-system"):
        equalize_groups(ledger, exclusive_groups=[0])


@pytest.mark.parametrize("minimum", [[], [0, 1], [True, 1], [1]])
def test_equal_load_minima_must_be_explicit_positive_and_aligned(minimum):
    ledger = DeclaredPackages((group(),), ((2,), (3,)), ((D(20000), D(40)),) * 2)
    with pytest.raises(ValueError, match="minima"):
        equalize_groups(ledger, minimum_load_counts=minimum)


@pytest.mark.parametrize("exclusive", [[-1], [1], [0, 0], [True]])
def test_equal_load_exclusive_indexes_cannot_be_guessed(exclusive):
    ledger = DeclaredPackages((group(),), ((1,), (1,)), ((D(20000), D(40)),) * 2)
    with pytest.raises(ValueError, match="indexes"):
        equalize_groups(ledger, exclusive_groups=exclusive)


def group(category="PACKAGE_DRUM", multiple=1):
    return CountedPackage(category, D(120), D(132), D(".28"), multiple)


@pytest.mark.parametrize("seed", range(200))
def test_fresh_groups_conserve_and_replay(seed):
    args = dict(
        groups=[group(), CountedPackage("PACKAGE_BOX", D(20), D(23), D(".08"), 2)],
        capacities=[(D(18000), D(28))] * 3,
        seed=seed,
        key="source:variant",
        utilization=(D(".35"), D(".75")),
    )
    value = sample_groups(**args)
    assert value == sample_groups(**args)
    assert sum(r["packageQuantity"] for r in value.public_rows()) == value.total
    assert value == DeclaredPackages.from_receipt(value.receipt())
    for row in value.matrix:
        assert row[1] % 2 == 0
    corrupted = deepcopy(value.receipt())
    corrupted["loads"][0]["grossKg"] = "1"
    with pytest.raises(ValueError, match="differs"):
        DeclaredPackages.from_receipt(corrupted)


def test_only_fifth_allocation_is_public():
    value = DeclaredPackages((group(), group()), ((3, 7),) * 5, ((D(18000), D(28)),) * 5)
    ids = list("ABCDE")
    original = [{"equipmentIdentifier": i} for i in ids]
    original[-1]["packageQuantity"] = 5
    assert value.project_placements(original, ids) == [
        {"equipmentIdentifier": i, **({"packageQuantity": 10} if i == "E" else {})} for i in ids
    ]
    assert value.public_rows() == [
        {"packageQuantity": 15, "typeCategory": "PACKAGE_DRUM"},
        {"packageQuantity": 35, "typeCategory": "PACKAGE_DRUM"},
    ]


def test_absent_equipment_has_no_public_allocation():
    value = DeclaredPackages((group(), group()), ((3, 7),), ((D(18000), D(28)),))
    assert value.total == 10
    with pytest.raises(ValueError, match="equipment identities"):
        value.project_placements([], [])


@pytest.mark.parametrize("matrix", [((0, 0),), ((1, -1),), ((1, True),), ((1, 0),), ((1,),)])
def test_invalid_membership_fails(matrix):
    with pytest.raises(ValueError):
        DeclaredPackages((group(), group()), matrix, ((D(18000), D(28)),))


def test_no_capacity_fallback():
    with pytest.raises(CandidateIncompatibility):
        sample_groups(
            [group(), group()], [(D(100), D(1))], seed=1, key="x", utilization=(D(".35"), D(".75"))
        )


@pytest.mark.parametrize("seed", range(50))
def test_printed_components_reserve_counts_before_optional_filling(seed):
    value = sample_groups(
        [group(), group(multiple=2)],
        [(D(3000), D(20))] * 2,
        seed=seed,
        key="components",
        utilization=(D(".01"), D(".01")),
        minimum_group_count=9,
        minimum_load_counts=[3, 1],
    )
    assert all(n >= 9 for n in value.counts)
    assert all(n == 0 or n >= 3 for n in value.matrix[0])
    assert all(row[1] % 2 == 0 for row in value.matrix)


def test_impossible_component_minimum_is_not_silently_reduced():
    with pytest.raises(CandidateIncompatibility, match="component minimum"):
        sample_groups(
            [group()],
            [(D(300), D(20))],
            seed=1,
            key="components",
            utilization=(D(".01"), D(".01")),
            minimum_group_count=3,
        )


@pytest.mark.parametrize("seed", range(50))
def test_installed_liquid_systems_have_exclusive_loads(seed):
    value = sample_groups(
        [CountedPackage("PACKAGE_FLEXITANK", D(1000), D(1010), D(2)), group(), group()],
        [(D(30000), D(33))] * 4,
        seed=seed,
        key="installed",
        utilization=(D(".35"), D(".75")),
        exclusive_groups=[0],
    )
    assert all(row[0] == 0 or row == (1, 0, 0) for row in value.matrix)
    assert all(n > 0 for n in value.counts)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacities": [(D(30000), D(33))]},
        {"presence": [[True, True], [False, True]]},
        {"minimum_group_count": 4},
        {"minimum_load_counts": [2, 2]},
    ],
)
def test_installed_system_constraints_never_silently_relax(kwargs):
    args = dict(
        groups=[group("PACKAGE_FLEXITANK"), group()],
        capacities=[(D(30000), D(33))] * 2,
        seed=1,
        key="installed",
        utilization=(D(".35"), D(".75")),
        exclusive_groups=[0],
    )
    error = ValueError if "capacities" in kwargs else CandidateIncompatibility
    with pytest.raises(error):
        sample_groups(**(args | kwargs))


@pytest.mark.parametrize("net,gross,cube", [("NaN", "10", "1"), ("11", "10", "1"), ("1", "2", "0")])
def test_invalid_physical_facts_fail(net, gross, cube):
    with pytest.raises(ValueError):
        CountedPackage("PACKAGE_BOX", D(net), D(gross), D(cube))
