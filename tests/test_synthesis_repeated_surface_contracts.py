from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.contact_values import require_contact_name_coverage
from document_ocr.synthesis.template_compiler.package_equations import (
    require_pallet_mark_contract,
    require_segmented_package_owner,
)


@pytest.mark.parametrize("gap", [" BAG ", "\u00a0BAG\u00a0", "\nCARTON\n"])
def test_packing_noun_between_segments_needs_package_owner(gap):
    raw = ("CODE" + gap + "GOODS").encode()
    end = len(raw) - 5
    binding = NS(
        logical_key="description",
        target_paths=["documentPatch.cargoGroups[0].description"],
        dependency_paths=(),
        realization=NS(mode="segmented_surface"),
        occurrences=[NS(byte_start=0, byte_end=4), NS(byte_start=end, byte_end=len(raw))],
    )
    template = NS(bindings=[binding])
    with pytest.raises(ValueError, match="lacks a typed package owner"):
        require_segmented_package_owner(raw, template)
    owner = NS(
        target_paths=["documentPatch.cargoPackages[0].typeCategory"],
        dependency_paths=(),
        realization=NS(mode="single_surface"),
        occurrences=[NS(byte_start=4, byte_end=end)],
    )
    template.bindings.append(owner)
    require_segmented_package_owner(raw, template)
    owner.target_paths = []
    owner.dependency_paths = ["documentPatch.cargoPackages[0].typeCategory"]
    require_segmented_package_owner(raw, template)


def test_other_cargo_segment_separators_are_not_package_claims():
    raw = b"CODE / GOODS"
    b = NS(
        logical_key="description",
        target_paths=["documentPatch.cargoGroups[0].description"],
        dependency_paths=(),
        realization=NS(mode="segmented_surface"),
        occurrences=[NS(byte_start=0, byte_end=4), NS(byte_start=7, byte_end=len(raw))],
    )
    require_segmented_package_owner(raw, NS(bindings=[b]))


def test_contact_repeat_requires_all_name_characters_but_not_linebreaks():
    name = "Émile Ali Abbas"
    text = f"Consignee: {name}\nATTN: Émile\nAli Abbas"
    raw = text.encode()
    path = "documentPatch.parties.consignee.contactDetails.contactName"
    target = {
        "documentPatch": {"parties": {"consignee": {"contactDetails": {"contactName": name}}}}
    }

    def slot(value, start=0):
        begin = raw.index(value.encode(), start)
        return NS(byte_start=begin, byte_end=begin + len(value.encode()))

    first = slot(name)
    binding = NS(target_paths=(path,), dependency_paths=(), occurrences=[first])
    template = NS(bindings=[binding])
    with pytest.raises(ValueError, match="complete target/dependency ownership"):
        require_contact_name_coverage(raw, template, target)
    binding.occurrences += [slot("Émile", first.byte_end), slot("Ali Abbas", first.byte_end)]
    require_contact_name_coverage(raw, template, target)
    # An independently generated first name cannot complete the target surname.
    fragment = binding.occurrences.pop(1)
    auxiliary = NS(target_paths=(), dependency_paths=(), occurrences=[fragment])
    template.bindings.append(auxiliary)
    with pytest.raises(ValueError, match="complete target/dependency ownership"):
        require_contact_name_coverage(raw, template, target)
    auxiliary.dependency_paths = (path,)
    require_contact_name_coverage(raw, template, target)


@pytest.mark.parametrize("mark", ["P/NO.: P1-P10", "PALLET NO:1-10", "P/NO.: 1 - 10"])
def test_pallet_range_needs_equation_or_compiled_count_owner(mark):
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    target = {"documentPatch": {"cargoGroups": [{"groupId": "g1", "marksAndNumbers": [mark]}]}}
    template = NS(bindings=[NS(logical_key="mark", target_paths=[path])], coherence_constraints=[])
    with pytest.raises(ValueError, match="typed count/range dependency"):
        require_pallet_mark_contract(template, target)
    template.coherence_constraints = [
        NS(kind="aggregate_inclusive_range_cardinality", member_logical_keys=["mark"])
    ]
    require_pallet_mark_contract(template, target)
    template.coherence_constraints = []
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = ["10PLTS/367CTNS"]
    target["documentPatch"]["cargoPackages"] = [
        {"groupId": "g1", "quantity": 367, "typeCategory": "PACKAGE_CARTON"}
    ]
    require_pallet_mark_contract(template, target)


def test_ordinary_shipment_marks_do_not_require_pallet_count_contract():
    target = {"documentPatch": {"cargoGroups": [{"marksAndNumbers": ["REF: NEW-123"]}]}}
    require_pallet_mark_contract(NS(bindings=(), coherence_constraints=()), target)


@pytest.mark.parametrize(
    "other", ["KAIA.THOMPSON@EXAMPLE.COM", "KAIA.THOMPSON", "KAIA.THOMPSON TRADING LTD"]
)
def test_complete_independent_fields_are_not_unbound_name_repetitions(other):
    name = "KAIA.THOMPSON"
    raw = f"CONTACT: {name}\nOTHER: {other}".encode()
    path = "documentPatch.parties.shipper.contactDetails.contactName"
    target = {"documentPatch": {"parties": {"shipper": {"contactDetails": {"contactName": name}}}}}
    first = raw.index(name.encode())
    second = raw.index(other.encode(), first + len(name))
    template = NS(
        bindings=[
            NS(
                target_paths=[path],
                dependency_paths=(),
                occurrences=[NS(byte_start=first, byte_end=first + len(name))],
            ),
            NS(
                target_paths=(),
                dependency_paths=(),
                occurrences=[NS(byte_start=second, byte_end=second + len(other))],
            ),
        ]
    )
    require_contact_name_coverage(raw, template, target)
