"""Pin the second explicitly reviewed multi-row wording correction cohort."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

REVIEWED = [
    (
        5,
        "07708e396d183d759476033c",
        "v005",
        "The disjoint generic-drum and steel-drum groups do not establish that every drum is "
        "steel. Retain the chloroform identity, liquid form and purity; omit the unsupported "
        "whole-shipment IN STEEL DRUMS qualifier.",
    ),
    (
        7,
        "ded35c376a700d4a30ec0ac9",
        "v009",
        "The public groups are bins, super bulk bags and IBCs. IN BULK BAGS incorrectly covers "
        "the whole shipment. Retain MICA WASTE AND SCRAP and omit this unnecessary package "
        "qualifier.",
    ),
    (
        7,
        "4add41c792b28d890765861c",
        "v007",
        "The sampled goods include both cotton-seed oilcake and the second sampled yarn "
        "identity. Include both sampled products in the short goods description; do not omit "
        "the second identity, invent another product, or copy HS codes into the prose.",
    ),
    (
        10,
        "fd4c6aedf0ef05c6cba7781e",
        "product_0",
        "The pallet and IBC package rows are disjoint groups, with no shared private nesting. "
        "SACK-PACKED ON PALLETIZED IBC SUPPORT invents a relationship. Retain the amino-naphthol "
        "identity, technical powder form, colour and assay; omit that packing phrase.",
    ),
    (
        10,
        "2fdca444f6e1502612fe252a",
        "product_0",
        "Crates, wooden cases and cartons are disjoint declared groups. CARTON-PACKED IN WOODEN "
        "CASES invents a shared hierarchy. Retain the alumina ceramic laboratory crucibles/boats "
        "and their intrinsic specifications; omit the unnecessary packing qualifier.",
    ),
    (
        10,
        "3e7505421da79fe9fef11e02",
        "v009",
        "The public groups are generic drums and bins; steel was not sampled. Retain synthetic "
        "industrial diamond dust/powder and grit details. Remove IN SEALED STEEL DRUMS rather "
        "than invent another outer-packing fact.",
    ),
    (
        20,
        "69b5119d57e03634af446b29",
        "product_0",
        "IQF and BLOCK FROZEN are asserted together for the same fish. Retain the sampled frozen "
        "fish identities and intrinsic details, using one consistent freezing presentation; "
        "the sampled BLOCK group supports block-frozen wording. Do not introduce a new packing "
        "hierarchy or counts.",
    ),
    (
        20,
        "0581b83434bf134f8fcba541",
        "product_0",
        "The pallet, block and wooden-pallet rows are disjoint. IN WOODEN PALLET CRATES invents "
        "an outer family and extends it to all groups. Retain the frozen tuna/skipjack fillets, "
        "skinless and glazed details; omit the unnecessary packing qualifier.",
    ),
    (
        24,
        "6f43d35d8409ac427b6b81dc",
        "v018",
        "The public groups include a bulk lot as well as bags. IN BAGS overstates whole-shipment "
        "packing. Retain SESAMUM SEEDS and natural white quality, without an outer-package "
        "qualifier; the host owns the separate package presentations.",
    ),
]


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    for number in sorted({row[0] for row in REVIEWED}):
        batch = root / f"batch-{number:02}"
        rows = [row for row in REVIEWED if row[0] == number]
        audit = batch / "reviewed-followup-goods-findings.json"
        atomic_publish_json(
            audit,
            {
                "status": "MAIN_AGENT_REVIEWED",
                "findings": [
                    {"sampleId": "syn_full_v7_" + suffix, "key": key, "finding": reason}
                    for _, suffix, key, reason in rows
                ],
            },
        )
        corrections = []
        for _, suffix, key, reason in rows:
            sample = "syn_full_v7_" + suffix
            candidate = json.loads((batch / "campaign/candidates" / f"{sample}.json").read_text())
            wording = batch / "campaign/wording" / f"{sample}.json"
            corrections.append(
                {
                    "sourceDocumentId": candidate["sourceDocumentId"],
                    "variant": candidate["variant"],
                    "sampleId": sample,
                    "wordingReceiptSha256": sha256_file(wording),
                    "keys": [key],
                    "reason": reason,
                    "auditEvidence": {"path": str(audit), "sha256": sha256_file(audit)},
                }
            )
        atomic_publish_json(
            batch / "followup-goods-correction-plan.json",
            {"status": "approved_for_regeneration", "corrections": corrections},
        )


if __name__ == "__main__":
    main()
