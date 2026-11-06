"""Pin explicit main-agent adjudications to the exact current rendered cohort."""

import argparse

from migrate_r30_synthetic import ART, OUT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest


def reviewed_records(staged, proposals, decisions):
    if staged["failures"] or staged["selected"] != 100 or len(staged["results"]) != 100:
        raise ValueError("review requires a completely rendered batch")
    records = []
    for entry in staged["results"]:
        matching = [
            (p, r) for p, r in proposals.get(entry["id"], []) if r["rendered"] == entry["rendered"]
        ]
        if not matching:
            raise ValueError(f"no exact-current semantic review: {entry['id']}")
        if any(r["source"] != entry["source"] for _, r in matching):
            raise ValueError("semantic review belongs to another source")
        decision = decisions.get(entry["id"], decisions.get(entry["source"][4:12]))
        if any(r["review"]["findings"] or r["invalidEvidence"] for _, r in matching) and (
            not isinstance(decision, str) or not decision.strip()
        ):
            raise ValueError(f"missing main-agent finding disposition: {entry['id']}")
        records.append(
            {
                "id": entry["id"],
                "rendered": entry["rendered"],
                "reviews": [pin(p) for p, _ in matching],
                "decision": decision or "No semantic findings; full structural replay passed.",
            }
        )
    return records


def main(batch, adjudications):
    adjudications = adjudications.resolve()
    path = OUT / "batches" / f"{batch:03d}-staging.json"
    staged = read(path)
    proposals = {}
    for entry in staged["results"]:
        paths = list((ART / "semantic-reviews" / entry["id"]).glob("*.json"))
        old = ART / "semantic-reviews" / (entry["id"] + ".json")
        if old.exists():
            paths.append(old)
        proposals[entry["id"]] = [(p, read(p)) for p in sorted(paths)]
    decisions = read(adjudications)
    records = reviewed_records(staged, proposals, decisions["decisions"])
    for row in records:
        verify_pin(row["rendered"])
    staging_pin = pin(path)
    frozen_staging = OUT / "batches" / f"{batch:03d}-staging-{staging_pin['sha256']}.json"
    atomic_publish_json(frozen_staging, staged)
    # Both decisions and model evidence may be revised without changing a render.
    # Freeze the decisions and address the complete receipt, not only its inputs.
    frozen_decisions = OUT / "batches" / f"{batch:03d}-adjudications-{digest(decisions)}.json"
    atomic_publish_json(frozen_decisions, decisions)
    receipt = {
        "status": "MAIN_AGENT_REVIEWED_FOR_PUBLICATION",
        "staging": pin(frozen_staging),
        "adjudications": pin(frozen_decisions),
        "records": records,
        "scope": "Fixed 100 IDs; current rendered evidence only. Main-agent dispositions "
        "are mandatory for every finding and invalid evidence quote. Provider "
        "proposals never mutate records or authorize publication.",
    }
    destination = OUT / "batches" / f"{batch:03d}-final-review-{digest(receipt)}.json"
    atomic_publish_json(destination, receipt)
    print(destination)
    return destination


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int)
    parser.add_argument("adjudications", type=Path)
    args = parser.parse_args()
    main(args.batch, args.adjudications)
