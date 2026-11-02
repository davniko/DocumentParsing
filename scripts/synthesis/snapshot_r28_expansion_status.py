"""Read-only campaign status capture; does not accept drafts or modify R28."""

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
RUNS = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"


def main():
    campaigns, native = [], []
    for directory in sorted(RUNS.glob("**/campaign")):
        candidates = [json.loads(p.read_text()) for p in (directory / "candidates").glob("*.json")]
        reviews = [json.loads(p.read_text()) for p in (directory / "reviews").glob("*.json")]
        ids = {r["documentId"] for r in candidates}
        hashes = {r["documentId"]: digest(r) for r in candidates}
        reviewed = {s for r in reviews for s, h in r["candidateHashes"].items() if hashes.get(s) == h}
        receipt = {
            "campaign": str(directory.relative_to(ROOT)),
            "drafts": len(ids),
            "sources": len({r["sourceDocumentId"] for r in candidates}),
            "reviewedDrafts": len(reviewed),
            "findings": sum(len(r["review"]["findings"]) for r in reviews),
            "published": (directory / "manifest.json").exists(),
            "preflight": "thermal-preflight" in str(directory),
            "control": "owner-controls" in str(directory),
            "families": Counter(
                r["scenario"]["provenance"]["independentDraw"]["provenance"]["regime"]
                for r in candidates
            ),
        }
        campaigns.append(receipt)
    for path in RUNS.glob("**/calls/*.json"):
        row = json.loads(path.read_text())
        if "stage" not in row:
            continue
        usage = row.get("usage") or {}
        native.append(
            {
                "path": str(path.relative_to(ROOT)),
                "stage": row["stage"],
                "success": row.get("output") is not None,
                "costStatus": row.get("costStatus"),
                "costUsd": row.get("costUsd"),
                "inputTokens": usage.get("inputTokens", 0),
                "outputTokens": usage.get("outputTokens", 0),
                "seconds": row.get("seconds"),
                "error": row.get("error"),
            }
        )
    before = {p.name: sha256_file(p) for p in (OUT / "before").iterdir() if p.is_file()}
    current = {p.name: sha256_file(p) for p in DATA.iterdir() if p.is_file()}
    changed = [name for name in before if current.get(name) != before[name]]
    costs = {}
    for success in (True, False):
        rows = [r for r in native if r["success"] == success]
        costs["successfulCalls" if success else "failedCalls"] = {
            "calls": len(rows),
            "inputTokens": sum(r["inputTokens"] for r in rows),
            "outputTokens": sum(r["outputTokens"] for r in rows),
            "recordedUsd": str(sum((Decimal(r["costUsd"] or "0") for r in rows), Decimal(0))),
            "missingCost": sum(r["costUsd"] is None for r in rows),
        }
    blocked = [
        r
        for r in native
        if r["error"] and "status_code: 402" in r["error"]
    ]
    certificates = []
    for path in sorted((OUT / "components").glob("*/certification.json")):
        receipt = json.loads(path.read_text())
        certificates.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                             "samples": receipt["samples"], "status": receipt["status"]})
    def ids_in(path):
        with path.open() as stream:
            return {json.loads(line)["documentId"] for line in stream}
    added = ids_in(DATA / "synthetic-train.jsonl") - ids_in(OUT / "before/synthetic-train.jsonl")
    report = {
        "status": "IN_PROGRESS_NOT_CERTIFIED",
        "requestedNewRecords": 2500,
        "newCampaignDrafts": sum(r["drafts"] for r in campaigns if not r["preflight"] and not r["control"]),
        "preflightDrafts": sum(r["drafts"] for r in campaigns if r["preflight"]),
        "controlDrafts": sum(r["drafts"] for r in campaigns if r["control"]),
        "stagedCertifiedRecords": sum(r["samples"] for r in certificates),
        "componentCertificates": certificates,
        "admittedToR28": len(added),
        "remainingAdmissions": 2500 - len(added),
        "campaigns": campaigns,
        "nativeUsageAllAttempts": costs,
        "historicalCreditRefusals": len(blocked),
        "creditsEvidence": blocked[:3],
        "beforeFiles": before,
        "currentFiles": current,
        "changedR28Files": changed,
        "datasetUnchanged": not changed,
        "notes": "Successful calls include held drafts, preflights and reviews, not only accepted "
        "samples. Historic account refusals do not establish a current credit block. Component "
        "certificates record staged publications; final current-code and dataset checks remain required.",
    }
    atomic_write_json(OUT / "status.json", report)
    atomic_write_json(OUT / "native-call-inventory.json", native)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"campaigns", "beforeFiles", "currentFiles", "creditsEvidence"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
