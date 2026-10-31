"""Archive exact accepted validation requests before removing nested audit metadata.

No model calls, receipt edits or production mutations. Every reconstructed request
must match the existing accepted receipt before it can be used in a scoped rebase.
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign, wording_request_hash
from document_ocr.synthesis.curated_wording_receipts import wording_request_data

ROOT = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
ARCHIVE = ROOT / "wording-context-before-v2"


def read(path):
    return json.loads(path.read_text())


def archive(item):
    name, controls = item
    existing = ARCHIVE / f"{name}.json"
    prior = ROOT / "wording-context-before" / f"{name}.json"
    if not existing.exists() and prior.exists() and not read(prior).get("failures"):
        atomic_publish_json(existing, read(prior))
    if existing.exists():
        saved = read(existing)
        for row in saved["rows"]:
            if sha256_file(Path(row["receipt"]["path"])) != row["receipt"]["sha256"]:
                raise ValueError("archived original wording receipt changed")
        return {
            "batch": name,
            "requests": len(saved["rows"]),
            "failures": saved.get("failures", []),
        }
    batch = ROOT / name
    choices = [
        batch / "production-replay/config.json",
        batch / "reviewed-authorities/config.json",
        batch / "prepared-config.json",
    ]
    config = read(next(path for path in choices if path.exists()))
    campaign = create_campaign(Path.cwd(), config)
    rows, failures = [], []
    for sid, control in controls:
        candidate = read(Path(control["candidate"]["path"]))
        sample = candidate["documentId"]
        path = Path(config["output"]) / "wording" / f"{sample}.json"
        receipt = read(path)
        if "scopedExtension" in receipt:
            raise ValueError("context migration must not rewrite scoped lexical lineage")
        # Reviewed corrections can advance the validation contract while keeping
        # the original generation scenario as history. Reconstruct the actual
        # accepted plan and demand its exact current receipt hash.
        blueprint, scenario, planned_sample, target = campaign.plan(sid, candidate["variant"])
        if planned_sample != sample:
            raise ValueError("selected candidate differs from frozen draw identity")
        request = campaign.wording_request(blueprint, scenario, sample, target)
        if wording_request_hash(request) != receipt["requestSha256"]:
            failures.append(
                {
                    "sourceId": sid,
                    "sampleId": sample,
                    "variant": candidate["variant"],
                    "reason": "old exact validation contract cannot be reconstructed",
                }
            )
            continue
        rows.append(
            {
                "sourceId": sid,
                "sampleId": sample,
                "variant": candidate["variant"],
                "receipt": {"path": str(path), "sha256": sha256_file(path)},
                "request": wording_request_data(request),
                "requestSha256": receipt["requestSha256"],
            }
        )
    result = {"batch": name, "status": "ORIGINAL_REQUEST_AUDIT", "rows": rows, "failures": failures}
    atomic_publish_json(ARCHIVE / f"{name}.json", result)
    print(json.dumps({"batch": name, "requests": len(rows), "held": len(failures)}), flush=True)
    return {"batch": name, "requests": len(rows), "failures": failures}


def main():
    selected = defaultdict(list)
    for sid, source in read(ROOT / "main-adjudication-02.json")["sources"].items():
        for control in source["controls"]:
            candidate = read(Path(control["candidate"]["path"]))
            if (
                "declaredGroupLedgers"
                in candidate["scenario"]["provenance"]["freshPrivatePacking"]["facts"]
            ):
                selected[control["batch"]].append((sid, control))
    with ThreadPoolExecutor(max_workers=3) as executor:
        summaries = list(executor.map(archive, sorted(selected.items())))
    atomic_publish_json(
        ARCHIVE / "summary.json",
        {
            "requests": sum(row["requests"] for row in summaries),
            "batches": summaries,
            "projectionImplementationBefore": {
                "path": "src/document_ocr/synthesis/independent_presentation.py",
                "sha256": sha256_file(
                    Path("src/document_ocr/synthesis/independent_presentation.py")
                ),
            },
            "providerCalls": 0,
            "originalReceiptsUnchanged": True,
        },
    )


if __name__ == "__main__":
    main()
