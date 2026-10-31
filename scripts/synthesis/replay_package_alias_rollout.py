"""Replay the newest explicit draw per source; keep failed and earlier draws intact."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from check_package_alias_candidates import check

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--retry-held", type=Path)
    args = parser.parse_args()
    output = args.work / "replay" / args.run
    if output.exists():
        raise FileExistsError(output)
    inventory = json.loads((args.work / "inspection/inventory.json").read_text())
    latest = {}
    for path in (args.work / "batches").glob("**/campaign/candidates/*.json"):
        row = json.loads(path.read_text())
        sid = row["sourceDocumentId"]
        if sid not in latest or row["seed"] > latest[sid][0]:
            latest[sid] = row["seed"], path
    groups = defaultdict(list)
    retained = {}
    if args.retry_held:
        previous = json.loads(args.retry_held.read_text())
        for row in previous["rows"]:
            if row["status"] == "PASS":
                if Path(row["candidate"]) != latest[row["sourceId"]][1]:
                    raise ValueError("retained replay is not the latest candidate")
                retained[row["sourceId"]] = row
    for sid, (_, path) in latest.items():
        if sid in retained:
            continue
        groups[path.parent.parent].append(sid)
    rows = list(retained.values())
    for directory, ids in sorted(groups.items()):
        paths = sorted(directory.glob("preparation/*/prepared-config.json"))
        if not paths:
            raise ValueError("candidate lacks prepared authority: " + str(directory))
        config = paths[-1]
        relative = directory.relative_to(args.work / "batches").parent
        result = check(config, output / relative, source_ids=sorted(ids))
        projected = output / relative / "replay-config.json"
        if projected.exists():
            config = projected
        for row in result["rows"]:
            row["preparedConfig"] = str(config)
            row["preparedConfigSha256"] = sha256_file(config)
        rows.extend(result["rows"])
    missing = sorted(set(inventory["config"]["source_ids"]) - latest.keys())
    rows.extend(
        {"sourceId": sid, "status": "HELD", "error": "no fresh candidate"} for sid in missing
    )
    atomic_publish_json(
        output / "summary.json",
        {
            "rows": sorted(rows, key=lambda r: r["sourceId"]),
            "counts": dict(Counter(r["status"] for r in rows)),
            "semanticReviewRequired": True,
            "retainedReplay": (
                {"path": str(args.retry_held), "sha256": sha256_file(args.retry_held)}
                if args.retry_held
                else None
            ),
        },
    )


if __name__ == "__main__":
    main()
