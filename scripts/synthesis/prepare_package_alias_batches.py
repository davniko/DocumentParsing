"""Prepare reproducible twenty-source native alias-certification campaigns.

Existing numeric/geographic authorities are composed only over disjoint source
identities. This does not alter the registry or admit an unreviewed candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prepared", default="prepared")
    args = parser.parse_args()
    root = Path.cwd()
    inventory = json.loads((args.work / "inspection/inventory.json").read_text())
    config = inventory["config"]
    multi = yaml.safe_load(
        Path("configs/synthesis/mpci_bl_curated_v7_multidg_validation.yaml").read_text()
    )
    directory = args.work / "batches" / args.run
    for name in ("numeric_certificates", "geography_certificates", "wording_certificates"):
        primary = json.loads((root / config["independent"][name]["path"]).read_text())
        extra = json.loads((root / multi["independent"][name]["path"]).read_text())
        if set(primary["sources"]) & set(extra["sources"]):
            raise ValueError("refusing overlapping source authorities")
        combined = copy.deepcopy(primary)
        combined["sources"].update(extra["sources"])
        if "selectedSources" in combined:
            combined["selectedSources"] = sorted(combined["sources"])
        path = directory / "authorities" / f"{name}.json"
        atomic_publish_json(path, combined)
        config["independent"][name] = {"path": str(path), "sha256": sha256_file(path)}
    config["aliases"]["presentations"] = {
        name: {
            "path": str(args.work / args.prepared / f"{name}.json"),
            "sha256": sha256_file(args.work / args.prepared / f"{name}.json"),
        }
        for name in ("vocabulary", "certificates")
    }
    config["aliases"]["presentations"].update(
        full_weight=0.5, short_weight=0.3, optional_weight=0.2
    )
    config["variants_per_source"] = 1
    config["budget_usd"] = 3.0
    config["seed"] = 2026103007
    # Each native request includes its source's reviewed nesting/presentation
    # constraints; package choice and numeric estimates remain donor-independent.
    config["independent_preparation"] = {
        "mixed_source_ids": [],
        "mixed_probability": 0,
        "package_targets": {},
        "unit_profile_source_ids": None,
    }
    all_ids = sorted(config["source_ids"])
    batches = []
    for start in range(0, len(all_ids), 20):
        batch = copy.deepcopy(config)
        batch["source_ids"] = all_ids[start : start + 20]
        batch["capabilities"] = {sid: batch["capabilities"][sid] for sid in batch["source_ids"]}
        batch["output"] = str(directory / f"batch-{start // 20 + 1:02d}" / "campaign")
        path = directory / f"batch-{start // 20 + 1:02d}" / "config.json"
        atomic_publish_json(path, batch)
        batches.append(
            {"batch": start // 20 + 1, "config": str(path), "sources": batch["source_ids"]}
        )
    atomic_publish_json(
        directory / "inventory.json",
        {
            "batches": batches,
            "status": "pending_native_synthesis_review_and_geometry",
            "sources": len(all_ids),
        },
    )
    print(
        json.dumps({"batches": len(batches), "sources": len(all_ids), "directory": str(directory)})
    )


if __name__ == "__main__":
    main()
