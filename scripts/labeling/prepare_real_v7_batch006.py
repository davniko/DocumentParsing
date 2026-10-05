"""Freeze 175 training and 13 validation candidates under the approved fresh-split choice."""
import json
from collections import Counter

from run_real_v7_batch import ROOT, prepare
from document_ocr.atomic import atomic_publish_json

OUT = ROOT / "artifacts/kie-labeling/batch006-screen-20261005"
BATCH = ROOT / "artifacts/kie-labeling/direct-real-batch006-20261005"


def main():
    inventory_path = OUT / "novelty-inventory-v2.json"
    inventory = json.loads(inventory_path.read_text())
    train = [d for d in inventory["ranked"] if d["split"] == "train"]
    validation = [d for d in inventory["ranked"] if d["split"] == "validation" and not d["baselineOverlap"]]
    assert len(validation) == 5
    # Spread fresh validation across short/long, dry/thermal and package layouts;
    # none repeats a baseline entity/product key. These are source IDs, not rules.
    fresh_validation = {"197b795c", "1c3797a5", "66f90c14", "7fbe9954",
                        "624696c0", "9f97c8ed", "a44bb232", "ce3f0739"}
    moved = [d for d in train if d["documentId"][4:12] in fresh_validation]
    assert len(moved) == 8 and all(not d["baselineOverlap"] for d in moved)
    selected_train = [d for d in train if d not in moved][:175]
    assert len(selected_train) == 175
    rows = [(d, "train") for d in selected_train] + [(d, "validation") for d in validation + moved]
    assignments = OUT / "assignments-v2.json"
    atomic_publish_json(assignments, dict(
        authorization="2026-10-05 user: Allow a new validation split to reach 60. All additions are unused in the new dataset; historical training membership remains recorded and does not claim unseen-by-older-model status.",
        documents=[dict(documentId=d["documentId"], split=split, historicalSplit=d["split"],
                        baselineOverlap=d["baselineOverlap"], selectionRank=d["rankWithinSplit"])
                   for d, split in rows],
        splits=dict(Counter(split for d, split in rows)),
        allThreeBaselineKeysNovel=sum(not d["baselineOverlap"] for d, s in rows),
        completeTradeTripleRepeated=sum(len(d["baselineOverlap"]) == 3 for d, s in rows),
        historicalTrainToValidation=8))
    prepare(BATCH, train=175, validation=13, seed=2026100506,
            inventory=inventory_path, assignments=assignments)


if __name__ == "__main__":
    main()
