"""Pin reviewed wording defects found in the fresh recovery cohort."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

CASES = (
    (
        4,
        "8284c7d337d4c2b82d10d33a",
        ["v009"],
        "The shipment has disjoint reels, wooden pallets and fibreboard boxes. Remove the "
        "whole-shipment IN REELS qualifier; retain the uncoated wood-free graphic paper and "
        "grammage.",
    ),
    (
        10,
        "fd4c6aedf0ef05c6cba7781e",
        ["product_0"],
        "The supplied group profiles permit a powder presentation on pallets and a liquid "
        "derivative in IBCs. Do not apply CRYSTALLINE POWDER or a powder colour to the whole "
        "shipment. Describe the sampled amino-phenol ether/derivative identity and technical "
        "grade neutrally, without inventing a uniform phase or common packing hierarchy.",
    ),
    (
        26,
        "9857d4d2d2bb630b351e059a",
        ["v005"],
        "The IBC and octabin groups have liquid and granular unit-profile bases, respectively. "
        "Remove the whole-shipment AMBIENT LIQUID qualifier. Retain the acyclic polycarboxylic "
        "acid derivative identity and purity without claiming one phase for both groups.",
    ),
    (
        26,
        "c471cdf769132a69fc1ebd6f",
        ["lexical_00_00", "lexical_00_01"],
        "The second product continuation only repeats a category caption, BULK SOLID GRANULAR "
        "PARTICLES, producing IN BULK BULK. Generate coherent whole-nutmeg product wording "
        "across both owned product lines, retaining dried whole kernels and useful grade or "
        "intrinsic quality detail. Do not append the outer package-category label as product text.",
    ),
    (
        27,
        "753f12dbd5b361088f7015fa",
        ["v009"],
        "The goods are the sampled non-metal halides/halide oxides, whose accepted group "
        "profiles describe liquid or solid derivatives in drums and octabins. SILICON "
        "TETRAFLUORIDE, COMPRESSED invents incompatible pressurized-gas cargo. Use a concise "
        "commercial description of the sampled halide/halide-oxide goods consistent with the "
        "provided profiles, without inventing a compressed-gas identity or a common phase.",
    ),
    (
        28,
        "a39eb39672e113bd056b161b",
        ["v011"],
        "The disjoint declared groups include bins as well as jumbo bags and IBCs. Remove "
        "the incomplete whole-shipment IN JUMBO BAGS AND IBCS enumeration; retain magnesium "
        "waste and scrap and its uncoated condition.",
    ),
    (
        29,
        "a75531dfa66a4cc4bff54414",
        ["v003"],
        "The description says mixed mainly with wool but then gives 65 percent polyester "
        "and 35 percent wool. Generate a consistent wool-dominant blend, retaining the "
        "synthetic staple-fibre yarn identity, ring-spun count and useful fancy-twist detail. "
        "Do not change packing, totals or any other fields.",
    ),
)


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    for number in sorted({row[0] for row in CASES}):
        batch = root / f"batch-{number:02}"
        cases = [row for row in CASES if row[0] == number]
        audit = batch / "recovery-wording-findings.json"
        atomic_publish_json(
            audit,
            {
                "status": "MAIN_AGENT_REVIEWED",
                "findings": [
                    {"sampleId": "syn_full_v7_" + suffix, "keys": keys, "finding": reason}
                    for _, suffix, keys, reason in cases
                ],
            },
        )
        corrections = []
        for _, suffix, keys, reason in cases:
            sample = "syn_full_v7_" + suffix
            candidate = json.loads((batch / "campaign/candidates" / f"{sample}.json").read_text())
            corrections.append(
                {
                    "sourceDocumentId": candidate["sourceDocumentId"],
                    "variant": candidate["variant"],
                    "sampleId": sample,
                    "wordingReceiptSha256": sha256_file(
                        batch / "campaign/wording" / f"{sample}.json"
                    ),
                    "keys": keys,
                    "reason": reason,
                    "auditEvidence": {"path": str(audit), "sha256": sha256_file(audit)},
                }
            )
        atomic_publish_json(
            batch / "recovery-wording-plan.json",
            {
                "status": "approved_for_regeneration",
                "corrections": corrections,
            },
        )


if __name__ == "__main__":
    main()
