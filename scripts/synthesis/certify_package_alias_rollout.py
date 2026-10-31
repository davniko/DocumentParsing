"""Publish package presentation authorities only after complete, reviewed replay.

This publishes template/configuration evidence, never training records. Native
critic findings need explicit main-agent decisions; geometry alone is not approval.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_numeric import _digest

DATASET_CHANGES = (
    "seven delivery-country continuations and twelve stale non-target location literals"
)


def pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    work = args.work
    inventory = json.loads((work / "inspection/inventory.json").read_text())
    source_ids = sorted(inventory["config"]["source_ids"])
    replay = json.loads(args.replay.read_text())
    rows = replay["rows"]
    if sorted(r["sourceId"] for r in rows) != source_ids or any(
        r["status"] != "PASS" for r in rows
    ):
        raise ValueError("all registered sources need passing fresh native replay")
    decisions = json.loads(args.decisions.read_text())
    controls_path = args.controls
    controls = json.loads(controls_path.read_text())
    if controls["counts"] != {"PASS": 3 * len(source_ids)}:
        raise ValueError("all source/style controls must pass")
    records = []
    for row in rows:
        candidate_path = Path(row["candidate"])
        candidate = json.loads(candidate_path.read_text())
        sid, sample = candidate["sourceDocumentId"], candidate["documentId"]
        positions_path = Path(row["positions"])
        positions = json.loads(positions_path.read_text())
        if (
            positions["candidateSha256"] != sha256_file(candidate_path)
            or positions["target"] != candidate["target"]
        ):
            raise ValueError("position evidence belongs to another candidate")
        review_path = candidate_path.parent.parent / "reviews" / f"{sid}.json"
        review = json.loads(review_path.read_text())
        if review["candidateHashes"].get(sample) != digest(candidate):
            raise ValueError("semantic review is stale")
        findings = review["review"]["findings"]
        prefix = sid[4:12]
        if findings and not decisions.get(prefix):
            raise ValueError("main-agent finding decision missing: " + prefix)
        records.append(
            {
                "sourceId": sid,
                "status": "PASS",
                "sampleId": sample,
                "candidate": pin(candidate_path),
                "positions": pin(positions_path),
                "nativeReview": pin(review_path),
                "aliasChanges": row["aliasChanges"],
                "mainReviewScope": (
                    "cargo/alias owners, source-policy adjudication, "
                    "target consistency and geometry receipts"
                ),
                "findingDecisions": [
                    {
                        "finding": finding,
                        "decision": "rejected_with_evidence",
                        "reason": decisions[prefix],
                    }
                    for finding in findings
                ],
            }
        )
    batches = [
        {"batch": start // 20 + 1, "status": "PASS", "sources": source_ids[start : start + 20]}
        for start in range(0, len(source_ids), 20)
    ]
    acceptance = {
        "version": 1,
        "status": "PASS",
        "sources": len(records),
        "batches": batches,
        "records": records,
        "replay": pin(args.replay),
        "controls": pin(controls_path),
        "decisions": pin(args.decisions),
        "trainingDatasetMutation": DATASET_CHANGES,
        "claims": (
            "source-certified alias functionality, not a first-attempt "
            "generation accuracy or complete privacy audit"
        ),
    }
    destination = work / "acceptance.json"
    atomic_publish_json(destination, acceptance)
    if args.promote:
        registry = work.parent
        runtime = registry / "package-presentations-v1"
        vocabulary = json.loads((work / "prepared-v3/vocabulary.json").read_text())
        certificates = json.loads((work / "prepared-v3/certificates.json").read_text())
        certificates["status"] = "certified_native_synthesis_and_geometry"
        certificates["acceptance"] = {
            r["sourceId"]: {"status": "PASS", "sampleId": r["sampleId"]} for r in records
        }
        certificates["acceptanceEvidence"] = pin(destination)
        atomic_publish_json(runtime / "vocabulary.json", vocabulary)
        atomic_publish_json(runtime / "certificates.json", certificates)
        config = json.loads((work / "batches/native-v2/batch-01/config.json").read_text())
        config["source_ids"] = source_ids
        config["capabilities"] = inventory["config"]["capabilities"]
        config["output"] = "artifacts/kie-synthesis-production/r28-package-aliases"
        config["budget_usd"] = 10.0
        config["aliases"]["presentations"].update(
            {name: pin(runtime / f"{name}.json") for name in ("vocabulary", "certificates")}
        )
        for name in ("numeric_certificates", "geography_certificates", "wording_certificates"):
            source = (
                work / "prepared-v3/numeric_certificates-final.json"
                if name == "numeric_certificates"
                else Path(config["independent"][name]["path"])
            )
            payload = json.loads(source.read_text())
            atomic_publish_json(runtime / f"{name}.json", payload)
            config["independent"][name] = pin(runtime / f"{name}.json")
        config_path = Path("configs/synthesis/mpci_bl_curated_v7_package_aliases.json")
        atomic_publish_json(config_path, config)
        manifest_path = registry / "manifest.json"
        atomic_publish_bytes(
            work / "registry-manifest-before-promotion.json", manifest_path.read_bytes()
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["packagePresentations20261030"] = {
            "sources": len(records),
            "batches": len(batches),
            "acceptance": pin(destination),
            "config": pin(config_path),
            "vocabulary": pin(runtime / "vocabulary.json"),
            "certificates": pin(runtime / "certificates.json"),
            "trainingDatasetMutation": DATASET_CHANGES,
        }
        # Refresh only existing shared-file hash records; other admissions stay intact.
        shared = manifest["sharedFiles"]
        for name in ("ownership.yaml", "auxiliary.yaml"):
            if not isinstance(shared[name], str) or len(shared[name]) != 64:
                raise ValueError("unexpected shared-file pin format: " + name)
            shared[name] = sha256_file(registry / name)
        sid = "doc_ab905fd40d2afadc45840dc00b569e510bfd6a1017218d16c2ce125ea38288a7"
        case = registry / "cases" / sid
        for name in ("contract.json", "target.json"):
            manifest["cases"][sid]["files"][name] = sha256_file(case / name)
        manifest["cases"][sid]["targetSha256"] = _digest(
            json.loads((case / "target.json").read_text())
        )
        manifest["cases"][sid]["sourceDataset"] = config["dataset"]
        manifest["cases"][sid]["deliveryCountryContinuationRepair"] = pin(
            work / "delivery-country-repair/validation.json"
        )
        atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "sources": len(records),
                "batches": len(batches),
                "promoted": args.promote,
                "aliasChanges": sum(r["aliasChanges"] for r in records),
                "findingCounts": dict(Counter(len(r["findingDecisions"]) for r in records)),
            }
        )
    )


if __name__ == "__main__":
    main()
