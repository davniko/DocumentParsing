"""Read back published labels and audit the next batch's actual request receipts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator
from publish_real_v7_batch2 import ROOT, validate_rows
from run_real_v7_batch import usage, verify

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.labeling_agents.direct_cargo import CargoSourceMap, map_grounding_errors


def published(dataset):
    paths = sorted((dataset / "batches").glob("*/manifest.json"))
    assert paths
    receipts = {}
    manifests = []
    for path in paths:
        manifest = json.loads(path.read_text())
        assert len(manifest["documents"]) == manifest["samples"]
        for receipt in manifest["documents"]:
            assert receipt["documentId"] not in receipts
            receipts[receipt["documentId"]] = receipt
        manifests.append(manifest)
    latest = manifests[-1]
    assert len(receipts) == latest.get("totalDatasetSamples", latest["samples"])
    for name, digest in latest["files"].items():
        assert sha256_file(dataset / name) == digest
    result = validate_rows(dataset, receipts)
    # Historical batch receipts retain their original snapshot meaning.
    destination = paths[-1].parent / "combined-readback-validation.json"
    atomic_publish_json(destination, result)
    return result


def candidates(out):
    manifest = json.loads((out / "selection.json").read_text())
    timing = json.loads((out / "timing.json").read_text())
    verify(manifest)
    counts = Counter()
    tokens = Counter()
    cost = 0
    errors = []
    validators = {}
    rows = []
    for row in manifest["documents"]:
        folder = out / "runs" / row["name"]
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["documentId"] == row["documentId"]
        state_path = folder / "refine/status.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            target = state["target"]
            if target is not None:
                BillOfLadingExtractionV7Label.model_validate_json(json.dumps(target))
                assert json.loads((folder / "refine/target.json").read_text()) == target
                counts["applicationValidTargets"] += 1
            findings = [
                dict(section=section, **finding)
                for section, review in state["reviews"].items()
                for finding in review["findings"]
            ]
        else:
            findings = []
            target = None
        rows.append(
            dict(
                **summary,
                ocr=str(Path(row["input"]) / "ocr.txt"),
                pdf=row["pdf"],
                target=str((folder / "refine/target.json").relative_to(ROOT)) if target else None,
                findings=findings,
            )
        )
        failed_map = False
        for phase in ("extract", "refine"):
            calls = folder / phase / "calls"
            if not calls.exists():
                continue
            by_context = {}
            by_name = {}
            for wire in sorted((calls.parent / "wire").glob("*.json")):
                body = json.loads(wire.read_text())
                fmt = body["text"]["format"]
                assert fmt["strict"] and body["reasoning"]["effort"] == "high"
                schema = fmt["schema"]
                digest = sha256_bytes(json.dumps(schema, sort_keys=True).encode())
                by_name.setdefault(fmt["name"], set()).add(digest)
                validators.setdefault(digest, Draft202012Validator(schema))
                pdfs = []
                ocr_hits = 0
                for message in body["input"]:
                    for part in message.get("content", []):
                        if part.get("type") == "input_text":
                            h = sha256_bytes(part["text"].encode())
                            by_context[h] = digest
                            ocr_hits += h == row["ocrSha256"]
                        if part.get("type") == "input_file":
                            pdfs.append(part["file_data"]["completePdfSha256"])
                assert ocr_hits == 1
                if fmt["name"] in ("CargoSourceMap", "CargoRelationResponse"):
                    assert pdfs == [row["pdfSha256"]]
                    counts["fullPdfCargoRequests"] += 1
                counts["strictOcrRequests"] += 1
            for file in sorted(calls.glob("*-result.json")):
                receipt = json.loads(file.read_text())
                failed_map |= receipt["stage"] == "cargo_mapper" and receipt["status"] == "failed"
                request = json.loads(
                    file.with_name(file.name.replace("-result", "-request")).read_text()
                )
                if request["context"]:
                    digest = by_context[sha256_bytes(request["context"].encode())]
                else:
                    assert request["stage"] in ("extractor", "cargo_mapper")
                    matches = by_name[request["schema"]["title"]]
                    assert len(matches) == 1
                    digest = next(iter(matches))
                for response in receipt.get("responses", []):
                    raw = json.loads("".join(response["text"]))
                    errors.extend(
                        dict(file=str(file), error=str(e))
                        for e in validators[digest].iter_errors(raw)
                    )
                    counts["nativeSchemaResponses"] += 1
                counts["calls"] += 1
                counts["unknownBillingCalls"] += receipt["billingStatus"] == "unknown"
                u, c = usage(receipt)
                tokens.update(u)
                cost += c
        if failed_map and summary["status"] == "reviewed_candidate":
            repaired = []
            for file in (folder / "refine").glob("cargo-map-review-*.json"):
                proposal = json.loads(file.read_text())
                if not proposal["errors"]:
                    source_map = CargoSourceMap.model_validate_json(
                        json.dumps(proposal["proposal"])
                    )
                    assert not map_grounding_errors(
                        source_map, (ROOT / row["input"] / "ocr.txt").read_text()
                    )
                    repaired.append(str(file.relative_to(ROOT)))
            assert repaired, "Passing candidate with failed map needs an explicit valid replacement"
            rows[-1]["recoveredCargoMapReceipts"] = repaired
            counts["failedCargoMapsExplicitlyRecovered"] += 1
    assert counts["strictOcrRequests"] == timing["wireRequests"]
    assert counts["calls"] == sum(r["calls"] for r in timing["results"])
    assert abs(cost - timing["estimatedUsd"]) < 1e-8
    result = dict(
        counts=counts,
        tokens=tokens,
        estimatedUsd=cost,
        schemaErrors=errors,
        statuses=dict(Counter(r["status"] for r in rows)),
        documents=rows,
    )
    atomic_publish_json(out / "validation.json", result)
    atomic_publish_json(
        out / "adjudication-queue.json", [r for r in rows if r["status"] != "reviewed_candidate"]
    )
    assert not errors, errors
    return {k: v for k, v in result.items() if k != "documents"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/curated/mpci-bl-real-v7-reviewed")
    )
    parser.add_argument("--batch", type=Path)
    args = parser.parse_args()
    print("PUBLISHED", json.dumps(published(ROOT / args.dataset), indent=2))
    if args.batch:
        print("CANDIDATES", json.dumps(candidates(ROOT / args.batch), indent=2))
