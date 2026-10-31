"""Nested group prompts retain shipment facts, never certificate bookkeeping."""

import copy
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from document_ocr.synthesis.curated_wording import WordingField, WordingRequest
from document_ocr.synthesis.curated_wording_receipts import (
    wording_request_data,
    wording_request_hash,
)
from document_ocr.synthesis.independent_presentation import semantic_private_packing

spec = importlib.util.spec_from_file_location(
    "multirow_context_rebase",
    Path(__file__).parents[1] / "scripts/synthesis/multirow_context_rebase.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
projected_request, rebase = module.projected_request, module.rebase


def ledger():
    return {
        "version": 1,
        "sourceDocumentId": "source",
        "certificateSha256": "a" * 64,
        "certificateFileSha256": "b" * 64,
        "countedCategory": "PACKAGE_PALLET",
        "publicQuantity": 2,
        "relation": "CONTAINS",
        "relatedCategory": "PACKAGE_BAG",
        "relatedQuantity": 20,
        "unitNetKg": "200",
        "unitGrossKg": "210",
        "netKg": "400",
        "grossKg": "420",
        "facts": {"bagsPerPallet": 10},
        "surfaces": {"packing": "20 BAGS ON 2 PALLETS"},
        "wordingFacts": [],
        "wordingAppends": [],
        "presentationDecisions": {},
        "publicAllocationVisibilityUnchanged": True,
        "invariants": {"conserved": True},
        "scopeLedgers": {},
    }


def nested():
    result = ledger()
    result["facts"]["declaredGroupLedgers"] = [ledger(), ledger()]
    return result


def test_nested_group_projection_preserves_semantics_but_not_audit_hashes():
    original = nested()
    before = copy.deepcopy(original)
    projected = semantic_private_packing(original)
    children = projected["facts"]["declaredGroupLedgers"]
    assert children == [semantic_private_packing(ledger())] * 2
    assert original == before
    changed = copy.deepcopy(original)
    changed["facts"]["declaredGroupLedgers"][0]["certificateFileSha256"] = "c" * 64
    assert semantic_private_packing(changed) == projected
    changed["facts"]["declaredGroupLedgers"][0]["netKg"] = "399"
    assert semantic_private_packing(changed) != projected


@pytest.mark.parametrize("malformed", [None, {}, [], ["not a ledger"]])
def test_nested_group_projection_rejects_malformed_ledger_collection(malformed):
    value = nested()
    value["facts"]["declaredGroupLedgers"] = malformed
    with pytest.raises(ValueError):
        semantic_private_packing(value)


def test_nested_projection_cannot_silently_drop_an_unclassified_fact():
    value = nested()
    value["facts"]["declaredGroupLedgers"][0]["newCargoFact"] = "keep me"
    with pytest.raises(ValueError, match="unclassified"):
        semantic_private_packing(value)


def test_audit_rebase_preserves_generated_values_and_rejects_real_context_changes():
    # The historical prompt had a projected parent but raw child ledgers.
    parent = semantic_private_packing(ledger())
    parent["facts"]["declaredGroupLedgers"] = [ledger(), ledger()]
    facts = {"countedCategory": "PACKAGE_PALLET", "privatePacking": parent}
    original = WordingRequest(
        "sample",
        "Context\nAUTHORITATIVE COUNTED/PACKING FACTS\n"
        + yaml.safe_dump(facts, sort_keys=False)
        + "The host renders these facts. Preserve cargo.",
        (WordingField("goods", "product", "SOURCE", "Generate matching goods"),),
    )
    receipt = {
        "documentId": "sample",
        "requestSha256": wording_request_hash(original),
        "values": {"goods": "COTTON"},
    }
    archive = {
        "request": wording_request_data(original),
        "requestSha256": receipt["requestSha256"],
        "receipt": {"path": "original.json", "sha256": "d" * 64},
    }
    expected = projected_request(archive["request"])
    current = replace(original, context=expected["context"])
    result = rebase(
        receipt, archive, current, archive_pin={"path": "archive.json", "sha256": "e" * 64}
    )
    assert result["values"] == receipt["values"]
    assert result["validationContractRebase"]["newProviderCall"] is False
    assert receipt["requestSha256"] == wording_request_hash(original)
    with pytest.raises(ValueError, match="more than nested audit metadata"):
        rebase(
            receipt,
            archive,
            replace(current, context=current.context + " Change cargo."),
            archive_pin={},
        )
