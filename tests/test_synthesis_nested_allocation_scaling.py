from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from test_synthesis_nested_package_prose import fixture

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler import complete_targets as targets
from document_ocr.synthesis.template_compiler import nested_package_prose as nested


def test_unlinked_shared_rows_sample_their_own_subtotal_without_inventing_a_third_row():
    template, original = fixture()
    source = NS(source=b"", target=original, template=template)
    seen = set()
    for seed in range(200):
        target = deepcopy(original)
        targets._scale_numbers(
            source, target, DeterministicStream(seed, "nested", "sample"), {}, {}
        )
        patch = target["documentPatch"]
        package = patch["cargoPackages"][0]["quantity"]
        rows = patch["cargoAllocationGroups"][0]["allocations"]
        assert len(rows) == 2
        first, second = (row["packageQuantity"] for row in rows)
        assert 0 < first == second < 17
        assert 0 < package < 51
        assert sum(row["packageQuantity"] for row in rows) < package
        seen.add((package, first))
        generated = nested.generate(template, original, target)
        assert generated == {
            "documentPatch.cargoGroups[0].additionalInformation[0]": (
                f"{first} PALLETS WITH 68 PIECES EACH"
            )
        }
    assert len(seen) > 10
    assert original["documentPatch"]["cargoPackages"][0]["quantity"] == 51


def test_shared_lower_bound_is_applied_to_each_equal_owner_before_the_draw():
    template, original = fixture()
    source = NS(source=b"", target=original, template=template)
    minimums = {"documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity": 14}
    for seed in range(30):
        target = deepcopy(original)
        targets._scale_numbers(
            source, target, DeterministicStream(seed, "nested", "bounds"), minimums, {}
        )
        rows = target["documentPatch"]["cargoAllocationGroups"][0]["allocations"]
        assert rows[0]["packageQuantity"] == rows[1]["packageQuantity"] >= 14


@pytest.mark.parametrize("total", range(8, 40))
def test_equality_subset_keeps_integer_closure_without_constraining_other_rows(total):
    values = targets._equal_allocation_quantities(
        [17, 17, 9, 0, 6], [2, 3, 1, 0, 1], total, (0, 1), independent_total=False
    )
    assert sum(values) == total
    assert values[0] == values[1] >= 3
    assert values[2] >= 1 and values[4] >= 1
    assert values[3] == 0


def test_linked_total_cannot_be_silently_rounded_or_source_restored():
    with pytest.raises(ValueError, match="indivisible"):
        targets._equal_allocation_quantities(
            [17, 17], [1, 1], 23, (0, 1), independent_total=False
        )
    assert targets._equal_allocation_quantities(
        [17, 17], [1, 1], 23, (0, 1), independent_total=True
    ) == [11, 11]


@pytest.mark.parametrize(
    "weights,bounds,total,indices,reason",
    [
        ([17, 16], [1, 1], 20, (0, 1), "invalid shared"),
        ([17, 17], [1, 1], 20, (0, 0), "invalid shared"),
        ([17, 17], [1, 1], 20, (0, 2), "invalid shared"),
        ([17, 17, 0], [1, 1, 1], 20, (0, 1), "observed row support"),
        ([17, 17], [15, 14], 29, (0, 1), "lower bounds"),
        ([17, 17], [18, 18], 36, (0, 1), "source capacity"),
    ],
)
def test_invalid_or_infeasible_constraints_fail_explicitly(weights, bounds, total, indices, reason):
    with pytest.raises(ValueError, match=reason):
        targets._equal_allocation_quantities(
            weights, bounds, total, indices, independent_total=True
        )


def test_linked_one_to_one_reconciliation_cannot_ignore_shared_count_contract():
    template, original = fixture()
    patch = original["documentPatch"]
    patch["cargoPackages"] = [
        dict(groupId="g1", packageId="p1", quantity=17),
        dict(groupId="g1", packageId="p2", quantity=17),
    ]
    group = patch["cargoAllocationGroups"][0]
    group.update(coverage="one_to_one_package_allocations", packageIds=["p1", "p2"])
    for row, package_id in zip(group["allocations"], group["packageIds"], strict=True):
        row["packageId"] = package_id
    source = NS(source=b"", target=original, template=template)
    # Different printed lower bounds can make independently drawn package counts
    # incompatible. They must remain an explicit unsatisfied contract, not be copied.
    with pytest.raises(ValueError, match="linked package quantities contradict"):
        targets._scale_numbers(
            source, deepcopy(original), DeterministicStream(2, "nested", "linked"),
            {"documentPatch.cargoPackages[0].quantity": 17}, {},
        )
