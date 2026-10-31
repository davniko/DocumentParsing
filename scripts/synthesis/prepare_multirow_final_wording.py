"""Pin two additional main-reviewed description corrections without changing facts."""

import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

CASES = (
    (
        17,
        "54ac88ed40ca94e623516897",
        "v019",
        "The accepted physical profile describes solid acyclic polycarboxylic acid derivatives "
        "in sacks/FIBC on wooden pallets. LIQUID and DRUMS are unsupported contradictory "
        "additions. Keep the acid-derivative identity and industrial grade, without adding "
        "new physical or packaging claims.",
    ),
    (
        22,
        "e469af56bb63f685e9d59034",
        "v007",
        "The current description omits the sampled vessel/article identity and mentions only "
        "photographic paper base. Include both sampled product identities in the available "
        "description. Keep them consistent with the supplied closed-container context and "
        "accepted small article profile; do not invent dismantling, out-of-gauge carriage, "
        "an intact full-size floating dock, counts, HS digits or a new packing hierarchy.",
    ),
)


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    for batch, suffix, key, reason in CASES:
        directory = root / f"batch-{batch:02}"
        sample = "syn_full_v7_" + suffix
        candidate = json.loads((directory / "campaign/candidates" / f"{sample}.json").read_text())
        audit = directory / "final-wording-findings.json"
        atomic_publish_json(
            audit, {"status": "MAIN_AGENT_REVIEWED", "sample": sample, "finding": reason}
        )
        atomic_publish_json(
            directory / "final-wording-plan.json",
            {
                "status": "approved_for_regeneration",
                "corrections": [
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
                ],
            },
        )


if __name__ == "__main__":
    main()
