"""Archive and replay inspection candidates after an explicitly reviewed code repair.

This does not admit samples. It rejects changed shipment facts and retains every
old candidate. Fresh review and geometry must follow the replay.
"""

import argparse
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_campaign import create_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    campaign = create_campaign(Path.cwd(), json.loads(args.config.read_text()))
    if (campaign.output / "manifest.json").exists():
        raise ValueError("published data cannot be replayed in place")
    rows = []
    for sid in campaign.source_ids:
        for variant in range(1, campaign.variant_counts[sid] + 1):
            try:
                current = campaign.render(sid, variant, persist=False)
                path = campaign.output / "candidates" / f"{current['documentId']}.json"
                if not path.exists():
                    atomic_publish_json(path, current)
                    rows.append({"sourceId": sid, "variant": variant, "status": "NEW_RENDER"})
                    continue
                old_bytes = path.read_bytes()
                old = json.loads(old_bytes)
                if old == current:
                    rows.append({"sourceId": sid, "variant": variant, "status": "IDENTICAL"})
                    continue
                before, after = flat(old["target"]), flat(current["target"])
                changes = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
                omitted = {
                    f["countryPath"]
                    for f in current.get("compositeLocalityProjection", {}).get("fields", [])
                }
                if changes - omitted - {"documentPatch.goodsItemDetails[0].description"}:
                    raise ValueError("replay changed shipment facts: " + str(sorted(changes)))
                if old["wordingReceiptSha256"] != current["wordingReceiptSha256"]:
                    raise ValueError(
                        "wording changes require the separate scoped regeneration audit"
                    )
                archive = (
                    campaign.output / "candidate-history" / f"{path.stem}-{digest(old_bytes)}.json"
                )
                atomic_publish_bytes(archive, old_bytes)
                atomic_write_json(path, current)
                rows.append(
                    {
                        "sourceId": sid,
                        "variant": variant,
                        "status": "REPLAYED",
                        "archive": str(archive),
                        "beforeSha256": digest(old_bytes),
                        "afterSha256": digest(current),
                        "changedTargetPaths": sorted(changes),
                        "textChanged": old["joinedRawText"] != current["joinedRawText"],
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "sourceId": sid,
                        "variant": variant,
                        "status": "HELD",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    report = {
        "status": "REPLAY_ONLY_REQUIRES_REVIEW_AND_GEOMETRY",
        "reason": args.reason,
        "config": str(args.config),
        "records": rows,
    }
    atomic_publish_json(args.report, report)
    print(
        json.dumps(
            {
                "counts": {
                    s: sum(r["status"] == s for r in rows)
                    for s in sorted({r["status"] for r in rows})
                }
            }
        )
    )


if __name__ == "__main__":
    main()
