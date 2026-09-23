from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.descendant import _render_equipment_receipt_binding
from document_ocr.synthesis.template_compiler.equipment_receipts import owned_inventory
from document_ocr.synthesis.template_compiler.host import SpanDraft, validate_binding_realizations


def source():
    return {
        "documentPatch": {
            "containers": [
                {"containerNumber": "ABCD1234567", "typeDescription": "40HC"},
                {"containerNumber": "EFGH1234567", "typeDescription": "20GP"},
            ],
            "cargoAllocationGroups": [
                {"allocations": [{"containerNumber": "EFGH1234567", "packageQuantity": 11}]},
                {"allocations": [{"containerNumber": "ABCD1234567", "packageQuantity": 20}]},
            ],
        }
    }


def test_receipt_uses_declared_allocation_membership_not_group_index_or_row_order():
    target = source()
    rows = owned_inventory(target, ("documentPatch.cargoAllocationGroups[0].allocations",))
    assert rows == (target["documentPatch"]["containers"][1],)
    assert owned_inventory(target, ("documentPatch.cargoAllocationGroups[1].allocations",)) == (
        target["documentPatch"]["containers"][0],
    )


def test_allocation_owned_receipt_tracks_new_identifiers_and_equipment_without_new_labels():
    old = source()
    new = deepcopy(old)
    row = new["documentPatch"]["containers"][1]
    row.pop("typeDescription")
    row.update(
        containerNumber="IJKL7654321",
        sizeCategory="FORTY_FIVE_FOOT_HIGH_CUBE",
        typeCategory="GENERAL_PURPOSE",
    )
    new["documentPatch"]["cargoAllocationGroups"][0]["allocations"][0]["containerNumber"] = row[
        "containerNumber"
    ]
    binding = SimpleNamespace(
        target_paths=(),
        dependency_paths=("documentPatch.cargoAllocationGroups[0].allocations",),
        occurrences=(SimpleNamespace(slot_id="s", source_text="1 X 20GP"),),
    )
    before = deepcopy(new)
    output = _render_equipment_receipt_binding(binding, source_target=old, target=new)
    assert output.replacements == {"s": "1 X 45HC"}
    assert new == before


def test_compiler_materialization_proves_allocation_receipt_on_actual_contract_path():
    text = "1 Container"
    draft = SpanDraft(
        draft_id="receipt",
        logical_key="equipment:allocation:0",
        render_mode="deterministic_derived",
        value_kind="equipment",
        group_kind="equipment",
        group_key="cargo:g1",
        target_paths=(),
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.cargoAllocationGroups[0].allocations",),
        dependency_bindings=(),
        char_start=0,
        char_end=len(text),
        source_text=text,
        evidence_origin="derived_operational_fact",
        render_policy="derived_surface",
        rationale="Count the distinct containers explicitly referenced by this allocation.",
    )
    validate_binding_realizations(raw=text, drafts=(draft,), source_target=source())


def test_shared_container_in_two_explicit_allocations_is_counted_once():
    target = source()
    target["documentPatch"]["cargoAllocationGroups"].append(
        {"allocations": [{"containerNumber": "EFGH1234567", "packageQuantity": 5}]}
    )
    assert (
        len(
            owned_inventory(
                target,
                (
                    "documentPatch.cargoAllocationGroups[0].allocations",
                    "documentPatch.cargoAllocationGroups[2].allocations",
                ),
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    "paths",
    [
        (),
        ("documentPatch.cargoPackages",),
        ("documentPatch.containers[0].madeUpField",),
        ("documentPatch.containers[0]trailing",),
        ("documentPatch.containers", "documentPatch.cargoPackages[0].quantity"),
        ("documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",),
        ("documentPatch.containers[50]",),
        ("documentPatch.cargoAllocationGroups[50].allocations",),
    ],
)
def test_unrelated_unknown_and_out_of_range_dependencies_are_never_ignored(paths):
    with pytest.raises(ValueError, match="equipment receipt"):
        owned_inventory(source(), paths)


@pytest.mark.parametrize("bad_number", [None, "", "MISSING0001"])
def test_missing_allocation_container_identity_fails_before_rendering(bad_number):
    target = source()
    target["documentPatch"]["cargoAllocationGroups"][0]["allocations"][0]["containerNumber"] = (
        bad_number
    )
    with pytest.raises(ValueError, match="equipment receipt"):
        owned_inventory(target, ("documentPatch.cargoAllocationGroups[0].allocations",))


def test_duplicate_container_identity_cannot_prove_allocation_ownership():
    target = source()
    target["documentPatch"]["containers"][0]["containerNumber"] = "EFGH1234567"
    with pytest.raises(ValueError, match="equipment receipt"):
        owned_inventory(target, ("documentPatch.cargoAllocationGroups[0].allocations",))


@pytest.mark.parametrize("allocations", [[], None, {}, [{"packageQuantity": 3}], ["invalid"]])
def test_empty_or_malformed_allocation_is_not_a_count_observation(allocations):
    target = source()
    target["documentPatch"]["cargoAllocationGroups"][0]["allocations"] = allocations
    with pytest.raises(ValueError, match="equipment receipt"):
        owned_inventory(target, ("documentPatch.cargoAllocationGroups[0].allocations",))
