"""Explicitly reviewed wording-only corrections from the complete cargo readback."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

CASES = (
    (
        13,
        "209476746213e7336073381b",
        "v003",
        "The accepted unit profile is dry crystalline fructose in multi-wall paper bags. "
        "The sampled registry identity permits fructose OR syrup, but the chosen packing "
        "does not permit syrup. Describe the dry crystalline food-grade product, retaining "
        "a similarly concise commercial description, without adding a liquid form.",
    ),
    (
        16,
        "77011b92bfa6b78107d19368",
        "v023",
        "The sampled complete crane article must fit intact inside the supplied closed "
        "containers. The generated 8 M SPAN portal gantry is incompatible with that context. "
        "Describe a compact complete crane consistent with the accepted numeric profile, "
        "retaining useful intrinsic specifications but not inventing dismantling or oversized "
        "carriage.",
    ),
    (
        16,
        "865a9bc3e01d984e42c36ac3",
        "v023",
        "The generated sentence contradicts itself: artificial fibres and 100 percent "
        "polyester. Keep the sampled printed knitted artificial-fibre fabric identity; use "
        "a consistent regenerated-fibre specification, retaining the floral print, width "
        "and useful source-style detail. Do not change the declared packing.",
    ),
    (
        18,
        "ef2a6dfe9184b661fe4febdd",
        "v006",
        "The two disjoint groups are generic drums and steel drums. The unqualified IN "
        "STEEL DRUMS applies steel to both. Retain THIONYL CHLORIDE, LIQUID and omit the "
        "unnecessary whole-shipment material qualifier.",
    ),
    (
        18,
        "403f6ec90145c3cd43d728e6",
        "v007",
        "BALED was not sampled: the declared groups are paper bags and sacks, without a "
        "private bale relation. Keep FROZEN POPPY STRAW and omit the unsupported BALED qualifier.",
    ),
    (
        21,
        "99ea17850c550b399162c420",
        "v003",
        "The opening AMBIENT LIQUID incorrectly describes both the liquid piperidine and "
        "the subsequently specified crystalline salt. Keep the useful two product lines "
        "and their respective forms/packing, but make the shared introduction neutral as to phase.",
    ),
    (
        21,
        "06d06324070dbe1047ad688e",
        "v003",
        "CARGO HOLD LOOSE invents a ship-hold arrangement instead of the supplied carrying "
        "equipment. Retain the separate bulk rice, bagged rice and sacked rice groups and "
        "useful intrinsic specifications without adding a different transport arrangement.",
    ),
)


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    for batch in sorted({case[0] for case in CASES}):
        directory = root / f"batch-{batch:02}"
        cases = [case for case in CASES if case[0] == batch]
        audit = directory / "semantic-wording-findings.json"
        atomic_publish_json(
            audit,
            {
                "status": "MAIN_AGENT_REVIEWED",
                "findings": [
                    {"sample": "syn_full_v7_" + suffix, "key": key, "reason": reason}
                    for _, suffix, key, reason in cases
                ],
            },
        )
        corrections = []
        for _, suffix, key, reason in cases:
            sample = "syn_full_v7_" + suffix
            candidate = json.loads(
                (directory / "campaign/candidates" / f"{sample}.json").read_text()
            )
            corrections.append(
                {
                    "sourceDocumentId": candidate["sourceDocumentId"],
                    "variant": candidate["variant"],
                    "sampleId": sample,
                    "wordingReceiptSha256": sha256_file(
                        directory / "campaign/wording" / f"{sample}.json"
                    ),
                    "keys": [key],
                    "reason": reason,
                    "auditEvidence": {"path": str(audit), "sha256": sha256_file(audit)},
                }
            )
        atomic_publish_json(
            directory / "semantic-wording-plan.json",
            {"status": "approved_for_regeneration", "corrections": corrections},
        )


if __name__ == "__main__":
    main()
