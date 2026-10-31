"""One-off audit of metadata-only wording validation-contract migration.

This is not a production receipt fallback. Original generation receipts remain
untouched; a revised validation fingerprint never represents a new model call.
"""

import copy

import yaml

from document_ocr.synthesis.curated_wording_receipts import (
    request_from_data,
    wording_request_data,
    wording_request_hash,
)
from document_ocr.synthesis.independent_presentation import semantic_private_packing


def projected_request(data):
    projected = copy.deepcopy(data)
    marker = "\nAUTHORITATIVE COUNTED/PACKING FACTS\n"
    end = "The host renders these facts."
    context = data["context"]
    if context.count(marker) != 1:
        raise ValueError("expected exactly one explicit packing-facts block")
    prefix, remainder = context.split(marker)
    block, suffix = remainder.split(end, 1)
    facts = yaml.load(block, Loader=yaml.CSafeLoader)
    if "declaredGroupLedgers" not in facts["privatePacking"]["facts"]:
        raise ValueError("no nested audit metadata to rebase")
    facts["privatePacking"] = semantic_private_packing(facts["privatePacking"])
    projected["context"] = prefix + marker + yaml.safe_dump(facts, sort_keys=False) + end + suffix
    return projected


def rebase(receipt, archive, request, *, archive_pin):
    original = request_from_data(archive["request"])
    if (
        wording_request_hash(original) != archive["requestSha256"]
        or receipt["requestSha256"] != archive["requestSha256"]
        or receipt["documentId"] != request.sample_id
        or "scopedExtension" in receipt
    ):
        raise ValueError("original wording-validation lineage is not intact")
    expected = projected_request(archive["request"])
    if expected != wording_request_data(request):
        raise ValueError("wording rebase changes more than nested audit metadata")
    result = copy.deepcopy(receipt)
    result["requestSha256"] = wording_request_hash(request)
    result["validationContractRebase"] = {
        "reason": "Remove known audit-only nested-ledger provenance from wording context",
        "originalReceipt": archive["receipt"],
        "originalRequestSha256": archive["requestSha256"],
        "verifiedOriginalRequestArchive": archive_pin,
        "generatedValuesChanged": False,
        "newProviderCall": False,
    }
    return result
