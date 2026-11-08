"""Publish reviewed R30 amendments, retaining the exact completed-run snapshots."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"
OUT = ROOT / "artifacts/synthesis-probes/r30-binding-repair-20261108"
DEST = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30/package-modes-20261108"
FROZEN = DATA / "audit-history/posttraining-20261108"


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def main():
    accepted = json.loads((OUT / "acceptance.json").read_text())
    if accepted["status"] != "REVIEWED_PASS":
        raise ValueError("No completed main-agent acceptance")
    if accepted["catalogSha256"] != sha256_file(OUT / "catalog-draft.json"):
        raise ValueError("Accepted catalog changed")
    manifest = json.loads((DATA / "manifest.json").read_text())
    if "posttrainingRepair" in manifest:
        raise ValueError("Amendment already published")
    review = json.loads((OUT / "party-amendments/review.json").read_text())
    corrections = {r["id"]: r for r in review["records"]}
    assert len(corrections) == 7
    FROZEN.mkdir(parents=True, exist_ok=True)
    frozen = {}
    for split, original in manifest["aggregateRecords"].items():
        source = ROOT / original["path"]
        target = FROZEN / source.name
        assert sha256_file(source) == original["sha256"]
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha256_file(target) == original["sha256"]
        frozen[split] = pin(target)
    atomic_publish_json(FROZEN / "manifest.json", manifest)
    atomic_publish_json(FROZEN / "frozen-splits.json", frozen)
    catalog = json.loads((OUT / "catalog-draft.json").read_text())
    for sid, entry in catalog["templates"].items():
        if entry["template"]["path"].startswith(str(OUT.relative_to(ROOT)) + "/templates/"):
            t = json.loads((ROOT / entry["template"]["path"]).read_text())
            path = DEST / "templates" / f"{sid}.json"
            atomic_publish_json(path, t)
            entry["template"] = pin(path)
    catalog["modeCoverageReview"] = pin(OUT / "acceptance.json")
    atomic_publish_json(DEST / "catalog.json", catalog)
    changed = []
    for split, original in manifest["aggregateRecords"].items():
        source = ROOT / original["path"]
        with tempfile.NamedTemporaryFile(
            "wb", dir=source.parent, prefix="." + source.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for line in source.open("rb"):
                old = json.loads(line)
                decision = corrections.get(old["documentId"])
                if decision:
                    assert decision["split"] == split
                    path = ROOT / decision["record"]["path"]
                    assert sha256_file(path) == decision["record"]["sha256"]
                    row = json.loads(path.read_text())
                    assert (
                        BillOfLadingExtractionV7DirectTarget.model_validate_json(
                            json.dumps(row["target"])
                        ).canonical_target()
                        == row["target"]
                    )
                    stream.write(
                        (
                            json.dumps(
                                row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                            )
                            + "\n"
                        ).encode()
                    )
                    changed.append(row["documentId"])
                else:
                    stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, source)
        manifest["aggregateRecords"][split] = pin(source)
    assert set(changed) == set(corrections)
    publication = {
        "status": "PUBLISHED",
        "changedRecords": changed,
        "frozenSplits": frozen,
        "aggregateRecords": manifest["aggregateRecords"],
        "catalog": pin(DEST / "catalog.json"),
        "review": pin(OUT / "party-amendments/review.json"),
        "acceptance": pin(OUT / "acceptance.json"),
    }
    atomic_publish_json(DEST / "publication.json", publication)
    manifest["templateCatalog"] = pin(DEST / "catalog.json")
    manifest["posttrainingRepair"] = pin(DEST / "publication.json")
    atomic_write_json(DATA / "manifest.json", manifest)
    print(json.dumps(publication, indent=2))


if __name__ == "__main__":
    main()
