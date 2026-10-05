"""Record finite human review instructions against completed, immutable candidates.

The input is an explicitly authored list of decisions, not model-generated repair
logic. Paths are JSON key/index arrays. Removing missing fields or replacing an
unchanged value is an error. Amending a review requires an explicit flag and
preserves the previous receipt alongside the appended exact-edit history.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from publish_real_v7_batch2 import apply_decisions
from review_real_v7_batch import ROOT, candidate, read

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label


def record(batch, instructions):
    destination = batch / "manual-review/decisions.json"
    decisions = read(destination) if destination.exists() else {}
    source_notes_path = batch / "manual-review/source-notes.json"
    source_notes = read(source_notes_path)
    selection = {row["index"]: row for row in read(batch / "selection.json")["documents"]}
    assert len({i["index"] for i in instructions}) == len(instructions)
    for instruction in instructions:
        assert instruction["notes"] and all(instruction["notes"])
        row = selection[instruction["index"]]
        previous = decisions.get(row["name"])
        if instruction.get("amend"):
            assert previous is not None, "Cannot amend a nonexistent review"
            assert instruction.get("actions"), "An amendment needs explicit changes"
        else:
            assert previous is None, "Reviews are immutable; amend with an explicit receipt"
        path, initial, _ = candidate(batch, row)
        if previous is not None:
            assert previous["inputTargetSha256"] == sha256_file(path)
            target = apply_decisions(initial, previous["changes"])
            changes = copy.deepcopy(previous["changes"])
        else:
            target = copy.deepcopy(initial)
            changes = []
        for action in instruction.get("actions", []):
            keys = action["path"]
            assert keys and action["reason"]
            node = target["documentPatch"]
            for key in keys[:-1]:
                node = node[key]
            key = keys[-1]
            change = dict(path=keys, reason=action["reason"])
            if action.get("remove"):
                assert "value" not in action
                assert key in node if isinstance(node, dict) else 0 <= key < len(node)
                change.update(before=node[key], remove=True)
                del node[key]
            else:
                value = action["value"]
                if isinstance(node, dict) and key not in node:
                    change["add"] = value
                else:
                    assert node[key] != value, "No-op corrections are not changes"
                    change.update(before=node[key], after=value)
                node[key] = copy.deepcopy(value)
            changes.append(change)
        canonical = BillOfLadingExtractionV7Label.model_validate_json(
            json.dumps(target)
        ).canonical_target()
        if instruction.get("canonicalize"):
            # Failed model drafts retain null optionals. Record their canonical
            # serialization explicitly, rather than altering immutable drafts.
            before = target["documentPatch"]
            after = canonical["documentPatch"]
            for key in before.keys() | after.keys():
                if key in before and key not in after:
                    changes.append(
                        dict(
                            path=[key],
                            before=before[key],
                            remove=True,
                            reason="Canonical serialization: omit absent optional field.",
                        )
                    )
                elif before.get(key) != after.get(key):
                    changes.append(
                        dict(
                            path=[key],
                            before=before[key],
                            after=after[key],
                            reason="Canonical serialization: omit nested null optionals.",
                        )
                    )
            target = canonical
        assert canonical == target
        assert apply_decisions(initial, changes) == canonical
        decisions[row["name"]] = dict(
            reviewed=True,
            inputTargetSha256=sha256_file(path),
            notes=[*(previous["notes"] if previous else []), *instruction["notes"]],
            changes=changes,
        )
        if previous is not None:
            decisions[row["name"]]["previousReview"] = previous
        if instruction.get("sourceReviewed"):
            source_notes.setdefault(f"{instruction['index']:03}", " ".join(instruction["notes"]))
    # Amendments retain their prior receipt; immutable model candidates stay intact.
    atomic_write_json(destination, decisions)
    atomic_write_json(source_notes_path, source_notes)
    print(f"Recorded {len(instructions)} explicit reviews; {len(decisions)} total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    args = parser.parse_args()
    record(ROOT / args.batch, read(ROOT / args.instructions))
