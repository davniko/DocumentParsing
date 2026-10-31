"""Inventory missing native descendants and make explicit, bounded redraw batches."""

import argparse
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--through", type=int, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reject", action="append", default=[])
    parser.add_argument("--presentations", type=Path)
    parser.add_argument("--numeric-certificates", type=Path)
    args = parser.parse_args()
    inventory = json.loads((args.work / "inspection/inventory.json").read_text())
    base = json.loads((args.work / "batches/native-v2/batch-01/config.json").read_text())
    base["capabilities"] = inventory["config"]["capabilities"]
    if args.presentations:
        for name in ("vocabulary", "certificates"):
            path = args.presentations / f"{name}.json"
            base["aliases"]["presentations"][name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    if args.numeric_certificates:
        path = args.numeric_certificates
        base["independent"]["numeric_certificates"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    sources = sorted(inventory["config"]["source_ids"])[: 20 * args.through]
    present = set()
    for path in (args.work / "batches").glob("**/campaign/candidates/*.json"):
        row = json.loads(path.read_text())
        present.add(row["sourceDocumentId"])
    rejected = {}
    for prefix in args.reject:
        ids = [sid for sid in sources if sid.startswith("doc_" + prefix)]
        if len(ids) != 1:
            raise ValueError("rejection prefix must resolve exactly once: " + prefix)
        rejected[ids[0]] = "main-review wording hold; regenerate, preserve rejected evidence"
    selected = [sid for sid in sources if sid not in present or sid in rejected]
    directory = args.work / "batches" / args.run
    if directory.exists():
        raise FileExistsError(directory)
    for start in range(0, len(selected), 20):
        config = copy.deepcopy(base)
        ids = selected[start : start + 20]
        config["source_ids"] = ids
        config["capabilities"] = {sid: base["capabilities"][sid] for sid in ids}
        config["seed"] = args.seed
        config["output"] = str(directory / f"batch-{start // 20 + 1:02d}/campaign")
        atomic_publish_json(directory / f"batch-{start // 20 + 1:02d}/config.json", config)
    atomic_publish_json(
        directory / "inventory.json",
        {
            "sourceIds": selected,
            "rejectedExisting": rejected,
            "seed": args.seed,
            "reason": "Explicit fresh draws for missing descendants and reviewed wording failures",
            "baseInventorySha256": sha256_file(args.work / "inspection/inventory.json"),
            "priorProposalsPreserved": True,
        },
    )
    print(json.dumps({"sources": len(selected), "batches": (len(selected) + 19) // 20}))


if __name__ == "__main__":
    main()
