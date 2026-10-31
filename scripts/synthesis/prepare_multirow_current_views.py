"""Pin retained inspection draws after an explicit runtime eligibility correction.

Rejected sources remain listed for fresh sampling; this never certifies or
publishes a reduced registry. Native facts of retained draws are byte-identical.
"""

import argparse
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_sampling import IndependentSampler


def subset(config, selected, directory):
    value = copy.deepcopy(config)
    if not set(selected) <= set(value["source_ids"]):
        raise ValueError("inspection subset is outside prepared sources")
    value["source_ids"] = selected
    value["capabilities"] = {sid: value["capabilities"][sid] for sid in selected}
    if "template_sampling" in value:
        counts = value["template_sampling"]["source_counts"]
        value["template_sampling"]["source_counts"] = {sid: counts[sid] for sid in selected}
        value["template_sampling"]["samples"] = sum(counts[sid] for sid in selected)
    for name in ("requests", "private_plans"):
        old = Path(value["independent"][name]["path"])
        if sha256_file(old) != value["independent"][name]["sha256"]:
            raise ValueError("prepared authority changed")
        rows = json.loads(old.read_text())
        rows = {key: row for key, row in rows.items() if key.rsplit(":", 1)[0] in selected}
        path = directory / f"{name}.json"
        atomic_publish_json(path, rows)
        value["independent"][name] = {"path": str(path), "sha256": sha256_file(path)}
    atomic_publish_json(directory / "config.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ownership", type=Path)
    parser.add_argument("--auxiliary", type=Path)
    args = parser.parse_args()
    sampler = None
    reports = []
    for batch in sorted(args.root.glob("batch-*")):
        # Batch 01's initial inspection was explicitly superseded by v2;
        # subsequent batches have only one prepared campaign authority.
        path = batch / (
            "prepared-config-v2.json" if batch.name == "batch-01" else "prepared-config.json"
        )
        if not path.exists():
            continue
        config = json.loads(path.read_text())
        if sampler is None:
            sampler = IndependentSampler.from_config(Path.cwd(), config["independent"]["registry"])
        requests = json.loads(Path(config["independent"]["requests"]["path"]).read_text())
        invalid = []
        for key, request in requests.items():
            try:
                sampler.validate_request(request)
            except ValueError as error:
                invalid.append({"key": key, "error": str(error), "goods": request["goods"]})
        rejected = {row["key"].rsplit(":", 1)[0] for row in invalid}
        selected = [sid for sid in config["source_ids"] if sid not in rejected]
        directory = batch / args.name
        if args.ownership:
            config["ownership"] = str(args.ownership)
        if args.auxiliary:
            config["auxiliary"] = str(args.auxiliary)
        if selected:
            subset(config, selected, directory)
        report = {
            "batch": batch.name,
            "status": "INSPECTION_ONLY",
            "retainedSources": selected,
            "requiresRedraw": sorted(rejected),
            "invalidRequests": invalid,
            "originalConfig": str(path),
        }
        atomic_publish_json(directory / "scope.json", report)
        reports.append(report)
    atomic_publish_json(args.root / f"{args.name}-scope.json", reports)
    print(
        json.dumps(
            {
                "batches": len(reports),
                "redrawSources": sorted(
                    {sid for report in reports for sid in report["requiresRedraw"]}
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
