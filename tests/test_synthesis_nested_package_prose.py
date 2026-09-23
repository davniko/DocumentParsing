from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import nested_package_prose as nested
from document_ocr.synthesis.template_compiler import package_prose


def fixture():
    text = "17 PALLETS WITH 68 PIECES EACH"
    source = {
        "documentPatch": {
            "cargoGroups": [dict(groupId="g1", additionalInformation=[text])],
            "cargoPackages": [
                dict(groupId="g1", packageId="p1", quantity=51, typeCategory="PACKAGE_PACKAGE")
            ],
            "containers": [
                dict(containerNumber="ABCU1234567"),
                dict(containerNumber="DEEU7654321"),
            ],
            "cargoAllocationGroups": [
                dict(
                    groupId="g1",
                    coverage="unlinked_package_quantities",
                    packageIds=[],
                    allocations=[
                        dict(containerNumber="ABCU1234567", packageQuantity=17),
                        dict(containerNumber="DEEU7654321", packageQuantity=17),
                    ],
                )
            ],
        }
    }
    template = NS(
        coherence_constraints=(),
        bindings=[
            NS(
                logical_key="c0",
                target_paths=("documentPatch.containers[0].containerNumber",),
                dependency_paths=(),
                occurrences=[NS(byte_start=0, byte_end=11, source_text="ABCU1234567")],
            ),
            NS(
                logical_key="c1",
                target_paths=("documentPatch.containers[1].containerNumber",),
                dependency_paths=(),
                occurrences=[NS(byte_start=60, byte_end=71, source_text="DEEU7654321")],
            ),
            NS(
                logical_key="nested",
                target_paths=("documentPatch.cargoGroups[0].additionalInformation[0]",),
                dependency_paths=tuple(
                    f"documentPatch.cargoAllocationGroups[0].allocations[{i}].packageQuantity"
                    for i in range(2)
                ),
                occurrences=[
                    NS(byte_start=12, byte_end=40, source_text=text),
                    NS(byte_start=72, byte_end=100, source_text=text),
                ],
            ),
        ],
    )
    return template, source


def test_inner_count_is_not_promoted_to_shipment_quantity():
    template, source = fixture()
    target = deepcopy(source)
    for allocation in target["documentPatch"]["cargoAllocationGroups"][0]["allocations"]:
        allocation["packageQuantity"] = 12
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    assert nested.generate(template, source, target) == {path: "12 PALLETS WITH 68 PIECES EACH"}
    assert target["documentPatch"]["cargoPackages"][0]["quantity"] == 51
    assert nested.allocation_equalities(template, source) == {0: (0, 1)}


def test_formal_package_generator_uses_owned_allocation_counts():
    template, source = fixture()
    target = deepcopy(source)
    for allocation in target["documentPatch"]["cargoAllocationGroups"][0]["allocations"]:
        allocation["packageQuantity"] = 11
    updates, formal = package_prose.generate(template, source, target)
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    assert updates[path] == "11 PALLETS WITH 68 PIECES EACH"
    assert path in formal
    with pytest.raises(ValueError, match="disagrees"):
        nested.validate(template, source, target)
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"][0] = updates[path]
    nested.validate(template, source, target)


def test_shared_prose_never_silently_uses_first_of_conflicting_counts():
    template, source = fixture()
    target = deepcopy(source)
    target["documentPatch"]["cargoAllocationGroups"][0]["allocations"][0]["packageQuantity"] = 12
    with pytest.raises(ValueError, match="equal generated"):
        nested.generate(template, source, target)


@pytest.mark.parametrize(
    "defect", ["count", "owner", "missing_occurrence", "changed_text", "mixed_group"]
)
def test_source_owner_and_exact_occurrence_proof_is_required(defect):
    template, source = fixture()
    if defect == "count":
        source["documentPatch"]["cargoAllocationGroups"][0]["allocations"][0]["packageQuantity"] = (
            18
        )
    if defect == "owner":
        source["documentPatch"]["cargoAllocationGroups"][0]["allocations"][0]["containerNumber"] = (
            "DEEU7654321"
        )
    if defect == "missing_occurrence":
        template.bindings[2].occurrences.pop()
    if defect == "changed_text":
        template.bindings[2].occurrences[0].source_text = "17 PALLETS WITH 69 PIECES EACH"
    if defect == "mixed_group":
        source["documentPatch"]["cargoAllocationGroups"][0]["groupId"] = "g2"
    with pytest.raises(ValueError):
        nested.compile_contracts(template, source)


def test_unowned_prose_does_not_gain_ownership_from_coincidental_number_match():
    template, source = fixture()
    template.bindings[2].dependency_paths = ()
    assert nested.compile_contracts(template, source) == ()


def test_other_explicit_cargo_text_contracts_are_not_reinterpreted():
    template, source = fixture()
    source["documentPatch"]["cargoGroups"][0]["additionalInformation"][0] = "BOOKING 17 REFERENCE"
    assert nested.compile_contracts(template, source) == ()
