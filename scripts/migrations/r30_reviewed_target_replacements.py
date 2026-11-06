"""Verify explicit target-only corrections outside a retained cargo refresh.

These are human-reviewed migration receipts, not permission for a generator to
rewrite unrelated shipment fields. Both native renders and the exact candidate
and replacement records are pinned, so a later modification requires re-review.
"""

import json

from publish_r30_sources import verify_pin

from document_ocr.synthesis.curated import flat


def verify_target_replacement(row, replacement, receipt, protected_paths):
    review_pin = receipt.get("targetCorrectionReview")
    if review_pin is None:
        raise ValueError("cargo replacement changes protected target without reviewed correction")
    review = json.loads(verify_pin(review_pin).read_text())
    if (
        review["status"] != "MAIN_AGENT_REVIEWED_TARGET_ONLY_REPLACEMENT"
        or review["documentId"] != row["documentId"]
        or review["sourceDocumentId"] != row["sourceDocumentId"]
        or review["baselineSha256"] != receipt["currentRecordSha256"]
        or review["candidate"] != receipt["candidate"]
        or review["replacementRecord"] != receipt["replacementRecord"]
        or not isinstance(review["reason"], str)
        or not review["reason"].strip()
    ):
        raise ValueError("target correction review identity or input pin changed")
    before = json.loads(verify_pin(review["beforeRendered"]).read_text())
    after = json.loads(verify_pin(review["afterRendered"]).read_text())
    for rendered in (before, after):
        if any(rendered[key] != row[key] for key in ("documentId", "sourceDocumentId")):
            raise ValueError("target correction native evidence has a different identity")
    if {k: v for k, v in before.items() if k != "target"} != {
        k: v for k, v in after.items() if k != "target"
    }:
        raise ValueError("target-only correction changed native input or geometry")
    old, new = flat(before["target"]), flat(after["target"])
    missing = object()
    changes = {k for k in old.keys() | new.keys() if old.get(k, missing) != new.get(k, missing)}
    if changes != set(review["paths"]) or not protected_paths <= changes:
        raise ValueError("target correction contains unreviewed changes")
    original, corrected = flat(row["target"]), flat(replacement["target"])
    if any(
        original.get(k, missing) != old.get(k, missing)
        or corrected.get(k, missing) != new.get(k, missing)
        for k in protected_paths
    ):
        raise ValueError("target correction evidence differs from replacement values")
