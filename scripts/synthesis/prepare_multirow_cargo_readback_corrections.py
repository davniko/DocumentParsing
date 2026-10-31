"""Record scoped wording corrections from the main-agent cargo readback."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

CASES = [
    (
        28,
        "7ade3b47",
        2,
        ["product_0", "product_1"],
        "The public groups are disjoint non-compressed bales and wooden pallets. Remove the "
        "whole-shipment ROLLED ON WOODEN PALLET qualifier. Retain fabric construction, blend, "
        "grammage, width and printed design without inventing a shared support relationship.",
    ),
    *[
        (
            30,
            "b2617901",
            variant,
            ["goods_main", "goods_specification"],
            "Origin and equipment are already host-rendered in the retained description "
            "continuation. Remove ORIGIN, country, container and outer packing captions from "
            "the generated wording. Retain the sampled commodity and useful intrinsic specs.",
        )
        for variant in (1, 2, 3)
    ],
    *[
        (
            32,
            "630cd159",
            variant,
            ["product"],
            "Generate the sampled DG commercial identity and useful product specifications. "
            "Do not repeat the example's PLASTIC DRUM / POLYETHYLENE DRUM lines: they are not "
            "an authority for the sampled outer or inner package material. Current declarations "
            "and per-counted-unit numeric clauses are rendered by the host. Do not copy them "
            "into the product wording or invent additional packing relationships.",
        )
        for variant in (1, 2, 3)
    ],
    (
        32,
        "61a51be1",
        1,
        ["product_0"],
        "The sampled outer category is generic DRUM, with no sampled material. Remove the "
        "unrequested STEEL material. Preserve the coal-tar benzol identity and industrial grade.",
    ),
    (
        32,
        "61a51be1",
        3,
        ["product_0"],
        "The shipment includes disjoint bulk granules, jumbo bags and multi-wall paper bags. "
        "Remove the incomplete whole-shipment packing enumeration. Preserve the DAP identity "
        "and intrinsic 18-46-0 granular fertilizer specification.",
    ),
    (
        33,
        "ba3dad0d",
        3,
        ["v008"],
        "The accepted bag-group profile describes solid powder and the drum profiles support "
        "other presentations of the sampled oxygen-heterocyclic compounds. Do not assert a "
        "universal LIQUID GRADE. Retain the sampled chemical family and technical purity "
        "without inventing a common phase or a more specific chemical identity.",
    ),
]


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    for number in sorted({r[0] for r in CASES}):
        batch = root / f"batch-{number:02}"
        candidates = {}
        for path in (batch / "campaign/candidates").glob("*.json"):
            row = json.loads(path.read_text())
            candidates[row["sourceDocumentId"][4:12], row["variant"]] = row
        audit = batch / "cargo-readback-findings.json"
        selected = [r for r in CASES if r[0] == number]
        atomic_publish_json(audit, {"status": "MAIN_AGENT_REVIEWED", "findings": selected})
        changes = []
        for _, prefix, variant, keys, reason in selected:
            row = candidates[prefix, variant]
            sample = row["documentId"]
            changes.append(
                {
                    "sourceDocumentId": row["sourceDocumentId"],
                    "variant": variant,
                    "sampleId": sample,
                    "keys": keys,
                    "reason": reason,
                    "wordingReceiptSha256": sha256_file(
                        batch / "campaign/wording" / (sample + ".json")
                    ),
                    "auditEvidence": {"path": str(audit), "sha256": sha256_file(audit)},
                }
            )
        atomic_publish_json(
            batch / "cargo-readback-plan.json",
            {"status": "approved_for_regeneration", "corrections": changes},
        )


if __name__ == "__main__":
    main()
