"""Pin explicit main-review decisions to the exact current native findings."""

import argparse
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--notes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    config, notes = [json.loads(x.read_text()) for x in (a.config, a.notes)]
    base = json.loads((ROOT / notes["baseReview"]).read_text()) if "baseReview" in notes else {}
    decisions = {**base.get("rejectedFindings", {}), **notes.get("rejectedFindings", {})}
    excluded = {s: r for s, r in base.get("redrawSources", {}).items() if s in config["source_ids"]}
    for prefix, reason in notes.get("excludedSourcePrefixes", {}).items():
        matches = [s for s in config["source_ids"] if s[4:].startswith(prefix)]
        if len(matches) != 1 or not reason:
            raise ValueError("source exclusion must identify exactly one configured source")
        excluded[matches[0]] = reason
    output = ROOT / config["output"]
    pinned = {}
    for sid in config["source_ids"]:
        if sid in excluded:
            continue
        receipt = json.loads((output / "reviews" / (sid + ".json")).read_text())
        for i, finding in enumerate(receipt["review"]["findings"]):
            key = f"{sid[4:12]}:{i}"
            decision = dict(decisions[key])
            row = json.loads((output / "candidates" / (finding["sample_id"] + ".json")).read_text())
            evidence = list(decision.get("evidence", []))
            for needle in decision.pop("evidenceLinesContaining", []):
                matches = [
                    line
                    for line in row["joinedRawText"].splitlines()
                    if needle.casefold() in line.casefold()
                ]
                if not matches:
                    raise ValueError("reviewed evidence line absent: " + key + "/" + needle)
                evidence.extend(dict.fromkeys(matches))
            decision["evidence"] = evidence
            if not decision["reason"] or not evidence:
                raise ValueError("reviewed reason and evidence required")
            if any(q not in row["joinedRawText"] for q in decision["evidence"]):
                raise ValueError("main evidence is not literal current OCR: " + key)
            pinned[key] = {**decision, "finding": finding}
    atomic_publish_json(
        a.output,
        {
            "status": "APPROVED_FOR_COMPONENT_VALIDATION",
            "excludedSources": excluded,
            "rejectedFindings": pinned,
            "rejectedScopedFindings": notes.get("rejectedScopedFindings", {}),
            "mainReview": notes["mainReview"],
            "notes": {"path": str(a.notes), "sha256": sha256_file(a.notes)},
        },
    )
    print(a.output)


if __name__ == "__main__":
    main()
