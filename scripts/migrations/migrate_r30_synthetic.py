"""Batch native rendering of explicitly reconciled current synthetic receipts.

Unreconciled history and new linguistic owners remain visible holds. This stage
does not admit a batch: it produces full render/provenance and target-delta
packets for targeted semantic review and subsequent certified publication.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from audit_r30_independent import independent_view
from publish_r30_sources import pin, verify_pin
from r30_literal_target_corrections import apply_literal_target_correction, correction_index
from r30_source_preserving import R30Replay
from review_r30_paddle import differences

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.synthesis.curated import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r30-synthetic-migration-20261103"
ART = ROOT / "artifacts/paddle-migration/r30-synthetic-migration-20261103"
DATA = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"


def read(path):
    return json.loads(path.read_text())


def current_for_render(row, receipt):
    """Use a baseline-pinned cargo refresh or explicitly reviewed target repair."""
    if not receipt or "replacementRecord" not in receipt:
        return row
    from document_ocr.synthesis.curated import flat
    from document_ocr.synthesis.scoped_cargo_refresh import cargo_path

    if receipt["currentRecordSha256"] != digest(row):
        raise ValueError("cargo replacement baseline differs")
    replacement = read(verify_pin(receipt["replacementRecord"]))
    if any(replacement[k] != row[k] for k in ("documentId", "sourceDocumentId")):
        raise ValueError("cargo replacement changes identity")
    old, new = flat(row["target"]), flat(replacement["target"])
    changes = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    protected = {k for k in changes if not cargo_path("documentPatch." + k)}
    if protected:
        from r30_reviewed_target_replacements import verify_target_replacement

        verify_target_replacement(row, replacement, receipt, protected)
    return replacement


def reconcile_identity_wording(wording, candidate, candidate_pin):
    """Reuse reviewed wording only after proving an identity-only receipt change."""
    identity = candidate.get("migrationIdentity")
    if identity is None or wording["status"] != "REVIEWED":
        raise ValueError("wording reuse needs a reviewed identity-only receipt")
    prior_pin = identity["previousCandidate"]
    original = read(verify_pin(prior_pin))
    expected = {**original, "documentId": candidate["documentId"], "migrationIdentity": identity}
    if expected != candidate or wording["candidateSha256"] != prior_pin["sha256"]:
        raise ValueError("wording reuse would change shipment context")
    return {
        **wording,
        "documentId": candidate["documentId"],
        "candidateSha256": candidate_pin["sha256"],
        "identityReconciliation": {"previousCandidate": prior_pin},
    }


def stage(batch, *, size=100, catalog_pin=None):
    started = time.perf_counter()
    manifest = read(DATA / "manifest.json")
    literal_corrections = correction_index(manifest)
    verify_pin(manifest["independentAudit"])
    catalog_pin = catalog_pin or manifest["templateCatalog"]
    catalog = R30Replay(ROOT, catalog_pin, manifest.get("sourcePreservingReceipts", ()))
    inventory = read(OUT / "receipt-inventory.json")
    verify_pin(inventory["baseline"])
    # Reconciled receipts join this same queue once their evidence has been
    # reviewed; selection is fixed in the plan, never silently replaced on failure.
    # A schedule amendment may regroup only unpublished identities. Old plans
    # stay immutable because published certificates retain their exact pins.
    plan_path = (
        verify_pin(manifest["syntheticMigration"]["plan"])
        if "syntheticMigration" in manifest
        else OUT / "batch-plan.json"
    )
    if plan_path.exists():
        plan = read(plan_path)
        if plan["inventory"] != pin(OUT / "receipt-inventory.json"):
            raise ValueError("batch plan predates current receipt inventory")
    else:
        ordered = sorted(
            inventory["records"], key=lambda r: (r["status"] != "EXACT_CURRENT_RECEIPT", r["id"])
        )
        plan = {
            "inventory": pin(OUT / "receipt-inventory.json"),
            "batchSize": size,
            "batches": [
                [r["id"] for r in ordered[start : start + size]]
                for start in range(0, len(ordered), size)
            ],
        }
        atomic_publish_json(plan_path, plan)
    if not 1 <= batch <= len(plan["batches"]) or size != plan["batchSize"]:
        raise ValueError("batch is outside the frozen migration plan")
    selected = set(plan["batches"][batch - 1])
    records = {
        r["documentId"]: r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] in selected
    }
    if records.keys() != selected:
        raise ValueError("batch does not resolve to exact current membership")
    entries = {r["id"]: r for r in inventory["records"]}
    results, failures, proposals = [], [], []
    known_wording = {}
    for directory in (
        ROOT / "artifacts/paddle-migration/r30-pilot-20261102/new-wording",
        ROOT / "artifacts/paddle-migration/r30-full-rebase-20261102/new-wording",
        ART / "wording",
    ):
        for path in directory.glob("*.json"):
            known_wording[path.stem] = path
    for identity in plan["batches"][batch - 1]:
        row, entry = records[identity], entries[identity]
        try:
            if digest(row) != entry["recordSha256"]:
                raise ValueError("current record changed after lineage inventory")
            reconciled = ART / "reconciled" / (identity + ".json")
            receipt = None
            receipt_wording = None
            if identity in catalog.source_preserving:
                if reconciled.exists():
                    raise ValueError("source-preserving and independent receipts conflict")
                path = catalog.source_preserving[identity]
            elif reconciled.exists():
                receipt = read(reconciled)
                if receipt["status"] != "REVIEWED_CURRENT_SHIPMENT_RECEIPT" or receipt[
                    "currentRecordSha256"
                ] != digest(row):
                    raise ValueError("reconciled receipt is not current and reviewed")
                path = verify_pin(receipt["candidate"])
                if "wording" in receipt:
                    receipt_wording = verify_pin(receipt["wording"])
            else:
                if entry["status"] != "EXACT_CURRENT_RECEIPT":
                    raise ValueError("current receipt lineage requires explicit reconciliation")
                original = ROOT / entry["candidate"]
                verify_pin({"path": entry["candidate"], "sha256": entry["candidateSha256"]})
                candidate = read(original)
                if (
                    candidate["joinedRawText"] != row["joinedRawText"]
                    or candidate["sourceDocumentId"] != row["sourceDocumentId"]
                ):
                    raise ValueError("exact receipt does not match current data")
                if candidate["documentId"] == identity:
                    path = original
                else:
                    candidate["documentId"] = identity
                    candidate["migrationIdentity"] = {
                        "previousCandidate": pin(original),
                        "currentRecordSha256": digest(row),
                    }
                    path = ART / "identity-receipts" / (identity + ".json")
                    atomic_publish_json(path, candidate)
            spec = catalog.wording_specification(path)
            wording = None
            if spec is not None:
                wp = receipt_wording or known_wording.get(identity)
                if wp is None:
                    proposals.append(spec)
                    raise ValueError("new native linguistic owners require targeted wording")
                wording = read(wp)
                if wording["candidateSha256"] != pin(path)["sha256"]:
                    if "migrationIdentity" not in read(path):
                        proposals.append(spec)
                        raise ValueError(
                            "new native linguistic owners require wording "
                            "for the reconciled receipt"
                        )
                    previous_wording = pin(wp)
                    wording = reconcile_identity_wording(wording, read(path), pin(path))
                    wording["identityReconciliation"]["previousWording"] = previous_wording
                    wp = ART / "wording" / (identity + ".json")
                    atomic_publish_json(wp, wording)
            rendered = catalog.render(
                path, current=current_for_render(row, receipt), wording_receipt=wording
            )
            correction = literal_corrections.get(identity)
            rendered = apply_literal_target_correction(rendered, correction)
            catalog.validator.validate(rendered["target"])
            for kind, key in (
                ("raw", "joinedRawText"),
                ("centroid", "positionedText"),
                ("bbox", "bboxText"),
            ):
                if independent_view(rendered["pages"], kind) != rendered[key]:
                    raise ValueError("independent coordinate serialization differs")
            output = ART / "rendered" / identity / (digest(rendered) + ".json")
            atomic_publish_json(output, rendered)
            results.append(
                {
                    "id": identity,
                    "source": row["sourceDocumentId"],
                    "rendered": pin(output),
                    "candidate": pin(path),
                    "wording": pin(wp) if spec else None,
                    "literalTargetCorrection": correction,
                    "currentRecordSha256": digest(row),
                    "reconciliation": pin(reconciled) if receipt else None,
                    "targetChanges": list(differences(row["target"], rendered["target"])),
                    "status": "RENDERED_REQUIRES_SCOPED_READBACK",
                }
            )
        except (ValueError, KeyError, IndexError) as exc:
            failures.append(
                {
                    "id": identity,
                    "source": row["sourceDocumentId"],
                    "error": str(exc),
                    "errorType": type(exc).__name__,
                }
            )
    report = {
        "batch": batch,
        "plan": pin(plan_path),
        "catalog": catalog_pin,
        "selected": len(selected),
        "results": results,
        "failures": failures,
        "newWording": proposals,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "batches" / f"{batch:03d}-staging.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"results", "failures", "newWording"}}
            | {"passed": len(results), "held": len(failures), "newWording": len(proposals)},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int)
    parser.add_argument(
        "--catalog", help="Explicit candidate catalog for prepublication validation"
    )
    args = parser.parse_args()
    stage(args.batch, catalog_pin=pin(ROOT / args.catalog) if args.catalog else None)
