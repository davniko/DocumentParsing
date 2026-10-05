"""Process the remaining previously screened reserves to replace rejected OCR sources."""
from collections import Counter

from run_real_v7_batch import ROOT, prepare
from review_real_v7_batch import read
from document_ocr.atomic import atomic_publish_json


def main():
    screen = ROOT / "artifacts/kie-labeling/batch006-screen-20261005"
    inventory = screen / "novelty-inventory-v2.json"
    used = {d["documentId"] for d in read(ROOT / "artifacts/kie-labeling/direct-real-batch006-20261005/selection.json")["documents"]}
    rows = [d for d in read(inventory)["ranked"] if d["documentId"] not in used]
    counts = Counter(d["split"] for d in rows)
    assert counts == {"train": 27, "validation": 4}, counts
    assignments = screen / "assignments-reserve007.json"
    atomic_publish_json(assignments, {
        "authorization": "User requested replacements sufficient for 600 train and 60 validation, with approved fresh split where necessary. These are remaining fresh sources passing the same original screen; acceptance still requires manual source/target review.",
        "documents": [{"documentId": d["documentId"], "split": d["split"], "historicalSplit": d["split"], "baselineOverlap": d["baselineOverlap"]} for d in rows],
        "splits": dict(counts),
    })
    prepare(ROOT / "artifacts/kie-labeling/direct-real-batch007-20261005", train=27, validation=4,
            seed=2026100507, inventory=inventory, assignments=assignments)


if __name__ == "__main__":
    main()
