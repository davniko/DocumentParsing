from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import package_count_surfaces as package_counts
from document_ocr.synthesis.template_compiler import shipment_totals as totals


def target(*quantities):
    return {"documentPatch": {"cargoPackages": [{"quantity": q} for q in quantities]}}


def test_unique_total_is_owned_without_rewriting_ocr():
    raw = "é\nTotal Items: 495\nTotal Gross Weight: 105930.000 Kgs."
    drafts = totals.normalize(raw=raw, drafts=(), source_target=target(495))
    assert len(drafts) == 1
    row = drafts[0]
    assert raw[row.char_start : row.char_end] == "495"
    assert row.dependency_paths == ("documentPatch.cargoPackages[0].quantity",)
    assert not row.target_paths
    assert row.derivation == "sum_package_quantity"
    assert totals.normalize(raw=raw, drafts=drafts, source_target=target(495)) == drafts


@pytest.mark.parametrize("document", [target(400), target(99, 99, 99, 99, 99), target()])
def test_ambiguous_or_contradictory_total_is_not_guessed(document):
    assert not totals.normalize(raw="Total Items: 495", drafts=(), source_target=document)


def test_decimal_and_nonshipment_caption_are_not_package_totals():
    assert not totals.occurrences("Total Items: 495.5\nInvoice Items: 495\nSUBTOTAL ITEMS: 495")
    assert len(totals.occurrences("Total No. of Packages: 1,234")) == 1


def test_unowned_and_wrong_owner_are_rejected_before_generation():
    raw = "é\nTotal Items: 495"
    start = raw.encode().index(b"495")
    with pytest.raises(ValueError, match="lacks one complete"):
        totals.require_owned(raw.encode(), NS(bindings=()))
    binding = NS(
        occurrences=(NS(byte_start=start, byte_end=start + 3),),
        target_paths=("documentPatch.cargoPackages[0].quantity",),
        derivation=None,
        value_kind="integer",
        realization=NS(mode="single_surface"),
    )
    totals.require_owned(raw.encode(), NS(bindings=(binding,)))
    binding.target_paths = ("documentPatch.billOfLadingNumber",)
    with pytest.raises(ValueError, match="unrelated"):
        totals.require_owned(raw.encode(), NS(bindings=(binding,)))


def test_compiler_risk_inventory_exposes_ambiguous_total():
    from document_ocr.synthesis.template_compiler.host import all_risk_candidates

    raw = "Total Items: 495"
    risks = all_risk_candidates(raw, (), target(300, 195))
    assert any(r.kind == "shipment_package_total" and r.source_text == "495" for r in risks)


def test_existing_item_row_count_and_allocation_alias_are_not_reinterpreted():
    raw = b"Total Items: 1"
    binding = NS(
        occurrences=(NS(byte_start=13, byte_end=14),),
        target_paths=("documentPatch.cargoPackages",),
        derivation="package_count",
        value_kind="integer",
        realization=NS(mode="deterministic_derivation"),
    )
    totals.require_owned(raw, NS(bindings=(binding,)))
    binding.derivation = "same_as_binding"
    binding.target_paths = ()
    binding.dependency_paths = (
        "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
    )
    totals.require_owned(raw, NS(bindings=(binding,)))
    binding.dependency_paths = ("documentPatch.containers",)
    with pytest.raises(ValueError, match="unrelated"):
        totals.require_owned(raw, NS(bindings=(binding,)))


def test_small_and_headed_package_counts_receive_distinct_derived_slots():
    raw = "No of Packgs\n3\n3 PACKAGE (S) AS PER ATTACHED SHEET.\n"
    source = {
        "documentPatch": {"cargoPackages": [{"quantity": 3, "typeCategory": "PACKAGE_PACKAGE"}]}
    }
    drafts = package_counts.normalize(raw=raw, drafts=(), source_target=source)
    assert len(drafts) == 2
    assert all(d.derivation == "sum_package_quantity" for d in drafts)
    bindings = tuple(
        NS(
            occurrences=(NS(byte_start=d.char_start, byte_end=d.char_end),),
            logical_key=d.logical_key,
        )
        for d in drafts
    )
    package_counts.require_owned(raw.encode(), NS(bindings=bindings), source)


def test_different_packing_level_and_repeated_equal_rows_need_review():
    raw = "26 SKIDS\n"
    source = {
        "documentPatch": {"cargoPackages": [{"quantity": 26, "typeCategory": "PACKAGE_CASE"}]}
    }
    assert not package_counts.normalize(raw=raw, drafts=(), source_target=source)
    with pytest.raises(ValueError, match="lacks a compiled owner"):
        package_counts.require_owned(raw.encode(), NS(bindings=()), source)
    repeated = {
        "documentPatch": {
            "cargoPackages": [
                {"quantity": 20, "typeCategory": "PACKAGE_PALLET"},
                {"quantity": 20, "typeCategory": "PACKAGE_PALLET"},
            ]
        }
    }
    assert not package_counts.normalize(raw="20 PALLETS", drafts=(), source_target=repeated)
