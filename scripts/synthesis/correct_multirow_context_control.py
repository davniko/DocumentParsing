"""Remove one reviewed lexical artifact from an unpublished live prompt control."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign


def main():
    batch = Path(
        "artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031/batch-41"
    )
    config = json.loads((batch / "prepared-config.json").read_text())
    campaign = create_campaign(Path.cwd(), config)
    sample = "syn_full_v7_40dc8782e73bb49361d6bf4a"
    path = campaign.output / "wording" / f"{sample}.json"
    candidate_path = campaign.output / "candidates" / f"{sample}.json"
    receipt = json.loads(path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if (campaign.output / "manifest.json").exists():
        raise ValueError("published samples cannot be edited by this inspection correction")
    if digest(receipt) != candidate["wordingReceiptSha256"]:
        raise ValueError("reviewed candidate/receipt lineage changed")
    prefix = "ACETIC ANHYDRIQUE — "
    keys = [key for key, value in receipt["values"].items() if value.startswith(prefix)]
    if len(keys) != 1:
        raise ValueError("expected exactly one reviewed lexical artifact")
    key = keys[0]
    before = receipt["values"][key]
    after = before.removeprefix(prefix)
    if after != "ACETIC ANHYDRIDE, LIQUID 99% MIN":
        raise ValueError("reviewed commodity wording changed")
    blueprint, scenario, actual, target = campaign.plan(
        candidate["sourceDocumentId"], candidate["variant"]
    )
    if actual != sample:
        raise ValueError("control identity changed")
    request = campaign.wording_request(blueprint, scenario, sample, target)
    changed = copy.deepcopy(receipt)
    changed["values"][key] = after
    campaign.validate_generated_wording([request], {sample: changed["values"]})
    archive = campaign.output / "wording-history" / f"{sample}-{sha256_file(path)}.json"
    old_candidate = (
        campaign.output / "candidate-history" / f"{sample}-{sha256_file(candidate_path)}.json"
    )
    atomic_publish_bytes(archive, path.read_bytes())
    atomic_publish_bytes(old_candidate, candidate_path.read_bytes())
    correction = {
        "kind": "main_agent_scoped_lexical_review",
        "key": key,
        "before": before,
        "after": after,
        "reason": (
            "Remove an unnecessary malformed alternate-name prefix; preserve the sampled "
            "acetic anhydride identity and all shipment facts. The original source is "
            "nitrocellulose, so this was generated wording, not a stale source binding."
        ),
        "beforeReceipt": {"path": str(archive), "sha256": sha256_file(archive)},
    }
    changed.setdefault("corrections", []).append(correction)
    atomic_write_json(path, changed)
    rendered = campaign.render(candidate["sourceDocumentId"], candidate["variant"], persist=False)
    atomic_write_json(candidate_path, rendered)
    atomic_publish_json(
        batch / "context-lexical-correction.json",
        {
            "status": "CORRECTED_REQUIRES_FRESH_REVIEW_GEOMETRY",
            "correction": correction,
            "candidateArchive": {"path": str(old_candidate), "sha256": sha256_file(old_candidate)},
            "samplingFactsChanged": False,
        },
    )


if __name__ == "__main__":
    main()
