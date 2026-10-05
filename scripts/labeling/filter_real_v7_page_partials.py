"""Publish the two explicitly approved source-quality exclusions without label edits."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from filter_real_v7_starting_dataset import ROOT, digest, read, tree_hashes, validate_records, write_json

SOURCE = ROOT / "data/curated/mpci-bl-real-v7-reviewed-r10-reduced-474"
DEST = SOURCE.with_name("mpci-bl-real-v7-reviewed-r11-reduced-472")
EXCLUDED = {
    "doc_188cb27f598cc26d9c125c2c4e2be2d906fc5a01a35a6d0f7691a4d78858b9f7":
        "005/060: only sheet 1/3 supplied; source shipper omitted by OCR and quantities lack usable units.",
    "doc_e352e3f6bde04bee8169254fdc6ac0043778118a972ed6582832c134ae56ab13":
        "005/130: only page 2/3 supplied; 6 of 17 declared containers and 1,225 of 3,425 packages represented.",
}


def main():
    start = time.monotonic()
    if DEST.exists():
        raise FileExistsError(DEST)
    before = tree_hashes(SOURCE)
    source_manifest = read(SOURCE / "projection-manifest.json")
    assert before == source_manifest["files"] | {
        "projection-manifest.json": digest(SOURCE / "projection-manifest.json")
    }
    original = {}
    for split in ("train", "validation"):
        for line in (SOURCE / f"{split}.jsonl").read_bytes().splitlines(keepends=True):
            key = json.loads(line)["documentId"]
            assert key not in original
            original[key] = (split, line)
    validate_records(SOURCE, original)
    assert len(original) == 474 and set(EXCLUDED) <= original.keys()
    assert all(original[k][0] == "train" for k in EXCLUDED)
    expected = {k: v for k, v in original.items() if k not in EXCLUDED}
    provenance = [r for r in source_manifest["documents"] if r["documentId"] in expected]
    assert len(provenance) == len(expected) == 472
    stage = Path(tempfile.mkdtemp(prefix=".real-v7-page-filter-", dir=DEST.parent))
    for key in expected:
        shutil.copytree(SOURCE / "samples" / key, stage / "samples" / key)
    for split in ("train", "validation"):
        (stage / f"{split}.jsonl").write_bytes(b"".join(v for s, v in expected.values() if s == split))
    for name in ("schema.json", "prompt-schema.json", "training-prompt.txt"):
        shutil.copy2(SOURCE / name, stage / name)
    exclusions = [r | {"reason": EXCLUDED[r["documentId"]]} for r in source_manifest["documents"]
                  if r["documentId"] in EXCLUDED]
    write_json(stage / "exclusions.json", exclusions)
    (stage / "README.md").write_text(
        "# Real baseline R11: approved missing-page partial exclusions\n\n"
        "472 documents: 425 training / 47 validation. Removed only 005/060-188cb27f "
        "and 005/130-e352e3f6, as approved on 2026-10-05. R10 is the unchanged backup. "
        "Retained JSONL records, labels, OCR, schemas, prompt, order and splits are byte-identical. "
        "Other R10 review recommendations were not silently applied.\n"
    )
    validate_records(stage, expected)
    assert tree_hashes(SOURCE) == before
    summary = dict(documents=472, splits=dict(Counter(s for s, _ in expected.values())),
                   exclusions=2, unchangedRetainedRecords=472, ocrEdits=0, labelEdits=0,
                   apiCalls=0, apiCostUsd=0, elapsedSeconds=round(time.monotonic()-start, 3))
    write_json(stage / "validation.json", summary)
    write_json(stage / "projection-manifest.json", dict(
        dataset=str(DEST.relative_to(ROOT)), sourceDataset=str(SOURCE.relative_to(ROOT)),
        sourceFiles=before, documents=provenance, summary=summary,
        implementationSha256=digest(Path(__file__)), files=tree_hashes(stage)))
    stage.rename(DEST)
    validate_records(DEST, expected)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
