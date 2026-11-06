"""Publish only explicitly reviewed, source-pinned missing linguistic owners."""

import argparse
import copy

from migrate_r30_synthetic import ART, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest


def reviewed_proposal(spec, proposal, decision):
    if proposal["documentId"] != spec["documentId"]:
        raise ValueError("wording identity differs")
    for field in ("candidateSha256", "sourceFingerprint"):
        if proposal[field] != spec[field]:
            raise ValueError("wording context differs")
    if set(proposal["values"]) != set(spec["requested"]):
        raise ValueError("wording owner membership differs")
    if not decision["reason"].strip():
        raise ValueError("missing explicit review")
    result = copy.deepcopy(proposal)
    for key, change in decision.get("corrections", {}).items():
        if result["values"][key] != change["before"] or not change["after"].strip():
            raise ValueError("manual wording correction baseline differs")
        result["values"][key] = change["after"]
    result["status"] = "REVIEWED"
    return result


def main(batch, review_path):
    review = read(review_path)
    staging_path = OUT / "batches" / f"{batch:03d}-staging.json"
    staged = read(staging_path)
    specs = {s["documentId"]: s for s in staged["newWording"]}
    if set(specs) != set(review["decisions"]):
        raise ValueError("review does not cover exact missing-owner cohort")
    retries_path = OUT / "batches" / f"{batch:03d}-wording-retries.json"
    # A first-pass cohort legitimately has no regeneration receipts.
    retry_pins = read(retries_path)["proposals"] if retries_path.exists() else []
    retries = {read(verify_pin(p))["documentId"]: verify_pin(p) for p in retry_pins}
    prepared = []
    for identity, spec in specs.items():
        proposal_path = retries.get(identity, ART / "wording-proposals" / (identity + ".json"))
        result = reviewed_proposal(spec, read(proposal_path), review["decisions"][identity])
        result["review"] = {
            "proposal": pin(proposal_path),
            "staging": pin(staging_path),
            "decision": pin(review_path),
            "scope": "Main-agent review of requested values and locality context; full native "
            "rendering, geometry and target review remain required before dataset publication.",
        }
        prepared.append((ART / "wording" / (identity + ".json"), result))
    for path, result in prepared:
        atomic_publish_json(path, result)
    receipt = {"accepted": [pin(p) for p, _ in prepared], "review": pin(review_path)}
    # A later template correction can introduce another owner in the same batch.
    # Preserve every admission rather than colliding with the first cohort.
    atomic_publish_json(
        OUT / "batches" / f"{batch:03d}-wording-review-{digest(receipt)}.json", receipt
    )
    print({"batch": batch, "accepted": len(prepared)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int)
    parser.add_argument("review")
    args = parser.parse_args()
    main(args.batch, ROOT / args.review)
