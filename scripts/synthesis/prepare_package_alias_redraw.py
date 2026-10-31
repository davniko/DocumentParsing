"""Make an explicit fresh-draw cohort; never silently retry or replace a sample."""

import argparse
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--prefix", action="append", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reason", required=True)
    args = p.parse_args()
    config = json.loads(args.base.read_text())
    selected = []
    for prefix in args.prefix:
        matches = [sid for sid in config["source_ids"] if sid.startswith("doc_" + prefix)]
        if len(matches) != 1:
            raise ValueError("source prefix must resolve exactly once: " + prefix)
        selected.extend(matches)
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate redraw source")
    config["source_ids"] = sorted(selected)
    config["capabilities"] = {sid: config["capabilities"][sid] for sid in selected}
    config["seed"] = args.seed
    config["output"] = str(args.output / "campaign")
    atomic_publish_json(args.output / "config.json", config)
    atomic_publish_json(
        args.output / "reason.json",
        {
            "reason": args.reason,
            "sourceIds": selected,
            "base": str(args.base),
            "baseSha256": sha256_file(args.base),
            "seed": args.seed,
            "originalProposalsPreserved": True,
        },
    )


if __name__ == "__main__":
    main()
