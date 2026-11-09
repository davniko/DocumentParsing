"""Amend only inherited R31 text-checksum metadata; preserve historical batches."""
from __future__ import annotations

import json
import os
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.paddle import pages_from_record, serialize, text_view_digests
from scripts.synthesis.extend_r31_boundaries import DATA, OUT, ROOT, read, pin


def main():
    started = time.perf_counter()
    manifest = read(DATA / "manifest.json")
    if "textDigestAmendment" in manifest:
        raise ValueError("metadata amendment already published; use the readback audit")
    if manifest["stageB"]["addedSynthetic"] != 750:
        raise ValueError("repair requires the completed 750-record extension")
    aggregate = DATA / "synthetic-train.jsonl"
    if pin(aggregate) != manifest["aggregateRecords"]["synthetic-train"]:
        raise ValueError("input aggregate changed before metadata repair")
    destination = OUT / "text-digest-amendment"
    atomic_write_json(destination / "manifest-before.json", manifest)
    changes = []
    counts = Counter()
    with tempfile.NamedTemporaryFile(dir=DATA, prefix=".text-digests-", delete=False) as output:
        temporary = Path(output.name)
        for index, line in enumerate(aggregate.open("rb")):
            row = json.loads(line)
            pages = pages_from_record(row)
            for mode, field in (("none", "joinedRawText"), ("centroid", "positionedText"), ("bbox", "bboxText")):
                if serialize(pages, geometry=mode) != row[field]:
                    raise ValueError(f"text differs from page evidence: {row['documentId']}")
            expected = text_view_digests(row)
            differing = {key: {"before": row[key], "after": value}
                         for key, value in expected.items() if row[key] != value}
            if differing:
                if index >= 9400:
                    raise ValueError("new extension unexpectedly has stale checksums")
                before = dict(row)
                row.update(expected)
                if {k: v for k, v in row.items() if k not in expected} != {
                    k: v for k, v in before.items() if k not in expected
                }:
                    raise ValueError("metadata repair changed content")
                changes.append({"id": row["documentId"], "fields": differing})
                counts.update(differing.keys())
                line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            output.write(line)
        output.flush()
        os.fsync(output.fileno())
    if index + 1 != 10150:
        temporary.unlink()
        raise ValueError("unexpected synthetic record count")
    atomic_write_json(destination / "changes.json", changes)
    after = {"path": str(aggregate.relative_to(ROOT)), "sha256": sha256_file(temporary)}
    receipt = {"status": "PASS", "before": manifest["aggregateRecords"]["synthetic-train"],
               "after": after, "previousManifest": pin(destination / "manifest-before.json"),
               "changes": pin(destination / "changes.json"), "changedRecords": len(changes),
               "changedFields": dict(counts), "allNonHashFieldsUnchanged": True,
               "historicalBatchCertificatesUnchanged": True, "new750ByteIdentical": True,
               "seconds": time.perf_counter() - started,
               "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    atomic_write_json(destination / "receipt.json", receipt)
    os.replace(temporary, aggregate)
    manifest["aggregateRecords"]["synthetic-train"] = after
    manifest["textDigestAmendment"] = pin(destination / "receipt.json")
    manifest["unchanged"] = ["Original training inputs, targets, coordinates and evidence are unchanged from Stage A; only inherited text-checksum metadata is amended. Real and validation records remain byte-identical. Historical batch files and certificates are immutable; the active aggregate additionally applies textDigestAmendment."]
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
