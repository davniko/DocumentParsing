"""Pin individually reviewed wording defects for the existing scoped regenerator."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    batch = root / "batch-03"
    reviewed = [
        (
            "syn_full_v7_2dd381f35c2491740004a744",
            1,
            "v003",
            "The sampled shipment is one bulk fine-powder lot, with no bag-count or "
            "bag-mass authority. Remove the invented 'IN 25KG BAGS, UNPALLETIZED' "
            "packing declaration. Retain the iron-oxide earth-colour/red-ochre "
            "identity and 70% FE2O3 MIN specification. Do not replace it with another "
            "unsampled packing fact.",
        ),
        (
            "syn_full_v7_40dc8782e73bb49361d6bf4a",
            3,
            "v007",
            "The three disjoint package groups are plastic drums, generic drums "
            "and steel drums. COMPOSITE is not sampled or supported. Remove that "
            "invented material and preferably omit the redundant outer-package "
            "list from the product prose. Retain ACETIC ANHYDRIDE 99% MIN LIQUID "
            "and do not change any authoritative DG fact.",
        ),
        (
            "syn_full_v7_31f49b1bf6878af94800cf2c",
            3,
            "v003",
            "The public fibreboard boxes, wooden pallets and cartons are disjoint "
            "accounted groups, not a statement that all boxes/cartons sit on those "
            "counted pallets. The wording 'IN FIBREBOARD BOXES AND CARTONS ON "
            "WOODEN PALLETS' invents that shared hierarchy. Retain the sampled "
            "SMOKED REPTILE MEAT AND EDIBLE OFFAL, SALTED product wording; omit "
            "the unnecessary packing sentence rather than invent another relation.",
        ),
    ]
    audit = batch / "reviewed-goods-findings.json"
    atomic_publish_json(
        audit,
        {
            "status": "MAIN_AGENT_REVIEWED",
            "findings": [
                {"sampleId": sid, "variant": variant, "key": key, "finding": reason}
                for sid, variant, key, reason in reviewed
            ],
        },
    )
    corrections = []
    for sample, variant, key, reason in reviewed:
        candidate = json.loads((batch / "campaign/candidates" / f"{sample}.json").read_text())
        wording = batch / "campaign/wording" / f"{sample}.json"
        corrections.append(
            {
                "sourceDocumentId": candidate["sourceDocumentId"],
                "variant": variant,
                "sampleId": sample,
                "wordingReceiptSha256": sha256_file(wording),
                "keys": [key],
                "reason": reason,
                "auditEvidence": {"path": str(audit), "sha256": sha256_file(audit)},
            }
        )
    atomic_publish_json(
        batch / "goods-correction-plan.json",
        {
            "status": "approved_for_regeneration",
            "corrections": corrections,
        },
    )


if __name__ == "__main__":
    main()
