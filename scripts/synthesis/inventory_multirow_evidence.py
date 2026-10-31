"""Inventory current native, candidate and geometry evidence without certifying it."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest


def inventory(root):
    config = json.loads((root / "recovery-base-config.json").read_text())
    samples, rejected = defaultdict(dict), []
    for batch in sorted(root.glob("batch-*")):
        current_config = batch / "production-replay/config.json"
        if not current_config.exists():
            current_config = batch / "reviewed-authorities/config.json"
        if not current_config.exists():
            current_config = batch / "prepared-config.json"
        if not current_config.exists():
            continue
        allowed = set(json.loads(current_config.read_text())["source_ids"])
        paths = sorted(batch.glob("**/geometry/summary.json"))
        paths += sorted(batch.glob("edge-overlap-geometry/summary.json"))
        latest = {}
        for path in sorted(paths, key=lambda p: p.stat().st_mtime_ns):
            for row in json.loads(path.read_text())["rows"]:
                if "variant" not in row:
                    rejected.append(
                        {
                            "report": str(path),
                            "sourceId": row["sourceId"],
                            "reason": "historical_report_without_variant_identity",
                        }
                    )
                    continue
                latest[row["sourceId"], row["variant"]] = row
        for row in latest.values():
            if row["status"] != "PASS" or row["sourceId"] not in allowed:
                continue
            candidate_path = Path(row["candidate"])
            if not candidate_path.exists():
                continue
            candidate = json.loads(candidate_path.read_text())
            positions = Path(row["positions"])
            geometry = json.loads(positions.read_text())
            sid, sample = candidate["sourceDocumentId"], candidate["documentId"]
            review_path = candidate_path.parent.parent / "reviews" / f"{sid}.json"
            if not review_path.exists():
                continue
            review = json.loads(review_path.read_text())
            if (
                geometry["candidateSha256"] != sha256_file(candidate_path)
                or geometry["target"] != candidate["target"]
                or review["candidateHashes"].get(sample) != digest(candidate)
            ):
                rejected.append(
                    {"batch": batch.name, "sampleId": sample, "reason": "stale_evidence"}
                )
                continue
            ledger = candidate["scenario"]["provenance"].get("declaredPackages")
            count = len(ledger["groups"]) if ledger else 1
            samples[sid][sample] = {
                "batch": batch.name,
                "sourceId": sid,
                "sampleId": sample,
                "groupCount": count,
                "candidate": str(candidate_path),
                "candidateSha256": sha256_file(candidate_path),
                "positions": str(positions),
                "review": str(review_path),
                "findings": [f for f in review["review"]["findings"] if f["sample_id"] == sample],
            }
    sources = {}
    for sid in config["source_ids"]:
        rows = list(samples[sid].values())
        counts = {r["groupCount"] for r in rows}
        required = (
            {1}
            if sid.startswith(("doc_bef02d48", "doc_458d2fa0", "doc_0dbc75a0", "doc_2763b891"))
            else {1, 2, 3}
        )
        sources[sid] = {
            "controls": rows,
            "observedGroupCounts": sorted(counts),
            "missingGroupCounts": sorted(required - counts),
        }
    return {
        "status": "STRUCTURAL_EVIDENCE_ONLY_REQUIRES_MAIN_ADJUDICATION",
        "registrySources": len(sources),
        "sourcesWithAllControls": sum(not s["missingGroupCounts"] for s in sources.values()),
        "sources": sources,
        "staleEvidenceExcluded": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root)
    atomic_publish_json(args.output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"sources", "staleEvidenceExcluded"}}
        )
    )


if __name__ == "__main__":
    main()
