"""Read back published R30 repairs against immutable, byte-exact original splits."""

import json
import resource
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
from document_ocr.spatial_inputs.paddle import pages_from_record
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.paddle_validation import check_synthetic_relations, check_views

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"
OUT = ROOT / "artifacts/synthesis-probes/r30-binding-repair-20261108"


def changes(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [
            item
            for key in sorted(before.keys() | after.keys())
            for item in changes(before.get(key), after.get(key), prefix + "/" + key)
        ]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [
            item
            for i, (old, new) in enumerate(zip(before, after, strict=True))
            for item in changes(old, new, prefix + "/" + str(i))
        ]
    return [] if before == after else [{"path": prefix, "before": before, "after": after}]


def main():
    start = time.perf_counter()
    manifest = json.loads((DATA / "manifest.json").read_text())
    frozen = json.loads(
        (DATA / "audit-history/posttraining-20261108/frozen-splits.json").read_text()
    )
    review = json.loads((OUT / "party-amendments/review.json").read_text())["records"]
    decisions = {r["id"]: r for r in review}
    catalog = PaddleTemplateCatalog(ROOT, manifest["templateCatalog"])
    for sid in catalog.entries:
        catalog._template(sid)
    counts, changed, ids = Counter(), [], set()
    preserved_conflicts = []
    for split, pin in manifest["aggregateRecords"].items():
        assert sha256_file(ROOT / pin["path"]) == pin["sha256"]
        assert sha256_file(ROOT / frozen[split]["path"]) == frozen[split]["sha256"]
        with (ROOT / pin["path"]).open("rb") as new, (
            ROOT / frozen[split]["path"]
        ).open("rb") as old:
            for before_line, after_line in zip(old, new, strict=True):
                before, row = json.loads(before_line), json.loads(after_line)
                identity = row["documentId"]
                assert identity == before["documentId"] and identity not in ids
                ids.add(identity)
                assert (
                    BillOfLadingExtractionV7DirectTarget.model_validate_json(
                        json.dumps(row["target"])
                    ).canonical_target()
                    == row["target"]
                )
                pages = pages_from_record(row)
                check_views(row, pages)
                counts[split] += 1
                counts["regions"] += sum(len(page.regions) for page in pages)
                if split == "synthetic-train":
                    try:
                        check_synthetic_relations(row["target"])
                    except ValueError as exc:
                        # Literal source-preserving OCR is not fresh balanced generation.
                        # Verify its exact prior adjudication; never exempt fresh shipments.
                        assert before_line == after_line
                        evidence = row["evidence"]["candidate"]
                        assert sha256_file(ROOT / evidence["path"]) == evidence["sha256"]
                        receipt = json.loads((ROOT / evidence["path"]).read_text())
                        assert receipt["status"] == "REVIEWED_SOURCE_PRESERVING_REBASE"
                        assert str(exc) == (
                            "complete allocation ledger differs from declared packages"
                        )
                        accounting = [
                            {
                                "allocationQuantities": [
                                    a["packageQuantity"] for a in g["splitGoodsPlacement"]
                                ],
                                "equipmentCount": len(row["target"]["containerInformation"]),
                                "packages": g["numberAndTypeOfPackages"],
                            }
                            for g in row["target"]["goodsItemDetails"]
                        ]
                        assert accounting == receipt["preservedSourceAccounting"]
                        preserved_conflicts.append(
                            {"id": identity, "receipt": evidence, "accounting": accounting}
                        )
                if identity not in decisions:
                    assert before_line == after_line
                    counts["byteExactUnchangedRows"] += 1
                    continue
                decision = decisions[identity]
                assert row == json.loads((ROOT / decision["record"]["path"]).read_text())
                delta = changes(before["target"], row["target"])
                assert delta and all(
                    d["path"].startswith("/parties/")
                    and d["path"].endswith(("/name", "/addressLine"))
                    for d in delta
                )
                input_changed = row["pages"] != before["pages"]
                if not identity.endswith("b10833c"):
                    assert not input_changed
                if split == "synthetic-train":
                    evidence = row["evidence"]
                    receipt = (
                        json.loads((ROOT / evidence["wording"]["path"]).read_text())
                        if evidence.get("wording")
                        else None
                    )
                    rendered = catalog.render(
                        ROOT / evidence["candidate"]["path"], wording_receipt=receipt
                    )
                    assert rendered["pages"] == row["pages"]
                    assert rendered["target"] == row["target"]
                changed.append(
                    {"id": identity, "split": split, "inputChanged": input_changed, "delta": delta}
                )
    assert {r["id"] for r in changed} == decisions.keys()
    assert [counts[s] for s in ["synthetic-train", "real-train", "validation"]] == [9400, 600, 60]
    report = {
        "status": "PASS",
        "catalog": manifest["templateCatalog"],
        "counts": counts,
        "changed": changed,
        "reviewedSourcePreservingAccounting": preserved_conflicts,
        "seconds": time.perf_counter() - start,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "publication-readback.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "changed"}))


if __name__ == "__main__":
    main()
