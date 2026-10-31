"""Apply two explicitly reviewed lexical corrections to unpublished inspection receipts."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    config = json.loads((root / "batch-36/prepared-config.json").read_text())
    campaign = create_campaign(Path.cwd(), config)
    if (campaign.output / "manifest.json").exists():
        raise ValueError("only unpublished inspection samples may be corrected")
    corrections = [
        (
            "syn_full_v7_edb089a8d0368e4031da6c5b",
            "postal_1",
            "MEVASSERET ZION INDUSTRIAL ZONE\nARIEL, ISRAEL",
            "HASHITA INDUSTRIAL ZONE\nARIEL, ISRAEL",
            "Remove a second real-city name from the fictional street/site component; "
            "retain the exact sampled locality and country.",
        ),
        (
            "syn_full_v7_ef699e3dcf8289ba81780c0c",
            "product_0",
            "CHILLED TURBOT (PLAICE-GRADE FRESH TURBOTS)",
            "CHILLED TURBOT (FRESH WHOLE FISH)",
            "Remove invented PLAICE-GRADE, which confuses two different fish identities; "
            "retain turbot and its chilled/fresh presentation.",
        ),
    ]
    staged = []
    for sample, key, before, after, reason in corrections:
        path = campaign.output / "wording" / (sample + ".json")
        data = path.read_bytes()
        receipt = json.loads(data)
        if receipt["values"][key] != before:
            raise ValueError("reviewed wording changed")
        candidate_path = campaign.output / "candidates" / (sample + ".json")
        candidate_bytes = candidate_path.read_bytes()
        candidate = json.loads(candidate_bytes)
        bp, scenario, actual, target = campaign.plan(
            candidate["sourceDocumentId"], candidate["variant"]
        )
        if actual != sample or candidate["wordingReceiptSha256"] != digest(receipt):
            raise ValueError("candidate is not the reviewed receipt")
        request = campaign.wording_request(bp, scenario, sample, target)
        changed = copy.deepcopy(receipt)
        changed["values"][key] = after
        campaign.validate_generated_wording([request], {sample: changed["values"]})
        archive = campaign.output / "wording-history" / (sample + "-" + digest(data) + ".json")
        changed.setdefault("corrections", []).append(
            {
                "kind": "main_agent_scoped_lexical_review",
                "key": key,
                "reason": reason,
                "before": before,
                "after": after,
                "beforeReceipt": {"path": str(archive), "sha256": digest(data)},
            }
        )
        staged.append((path, data, changed, archive, candidate_path, candidate_bytes))
    rows = []
    for path, data, changed, archive, candidate_path, candidate_bytes in staged:
        atomic_publish_bytes(archive, data)
        candidate_archive = (
            campaign.output
            / "candidate-history"
            / (path.stem + "-" + digest(candidate_bytes) + ".json")
        )
        atomic_publish_bytes(candidate_archive, candidate_bytes)
        atomic_write_json(path, changed)
        candidate_path.unlink()
        rows.append(
            {
                "sampleId": path.stem,
                "wordingArchive": str(archive),
                "candidateArchive": str(candidate_archive),
                "correction": changed["corrections"][-1],
            }
        )
    atomic_publish_json(
        root / "batch-36/final-lexical-corrections.json",
        {
            "status": "CORRECTED_REQUIRES_FRESH_RENDER_REVIEW_GEOMETRY",
            "corrections": rows,
            "samplingFactsChanged": False,
        },
    )


if __name__ == "__main__":
    main()
