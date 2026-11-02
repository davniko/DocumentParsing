"""Add two exact, reviewed notify-footnote leaves; preserve every input byte."""

import copy
import json
import time
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101/notify-repair"


def main():
    started = time.perf_counter()
    plan = json.loads((OUT / "legacy-reviewed.json").read_text())
    constraints = RelationExplicitTaskConstraints.model_validate_json((DATA / "task-constraints.json").read_bytes())
    task = get_training_task(constraints.task).bind_constraints(constraints)
    staged, seen, evidence = [], set(), []
    for path in sorted(DATA.glob("*.jsonl")):
        before = path.read_bytes()
        lines = before.splitlines(keepends=True)
        changed = []
        for index, line in enumerate(lines):
            original = json.loads(line)
            did = original["documentId"]
            if did not in plan:
                continue
            if path.name == "validation.jsonl":
                raise ValueError("validation is outside this repair")
            email, phone = plan[did]
            raw = original["joinedRawText"]
            if raw.count("**") != 3 or f"**EMAIL ADDRESS.: {email}\nPHONE NUMBER.: {phone}\n" not in raw:
                raise ValueError("exact shared-footnote evidence changed")
            row = copy.deepcopy(original)
            notify = row["target"]["parties"]["notifyParties"][0]
            if set(notify["contactDetails"]) != {"contactName"}:
                raise ValueError("existing notify contact is not the reviewed absence")
            expected = copy.deepcopy(row["target"])
            expected["parties"]["notifyParties"][0]["contactDetails"].update(emailAddresses=[email], phoneNumbers=[phone])
            canonical = task.canonicalize(expected)
            if canonical != expected:
                raise ValueError("target insertion needs explicit normalization review")
            row["target"] = canonical
            assert {k:v for k,v in row.items() if k != "target"} == {k:v for k,v in original.items() if k != "target"}
            lines[index] = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            seen.add(did)
            changed.append(did)
            evidence.append({"file": path.name, "documentId": did, "beforeTarget": original["target"], "afterTarget": row["target"], "rawSha256": digest(raw.encode()), "positionedSha256": digest(row["positionedText"].encode())})
        if changed:
            staged.append((path, before, b"".join(lines), changed))
    if seen != set(plan):
        raise ValueError("reviewed target identities incomplete")
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, before, after, changed in staged:
        if manifest["files"][path.name] != digest(before):
            raise ValueError("dataset manifest hash changed before repair")
    atomic_publish_bytes(OUT / "before/manifest.json", manifest_path.read_bytes())
    for path, before, after, changed in staged:
        atomic_publish_bytes(OUT / "before" / path.name, before)
    receipt = {"status": "VALIDATED_TARGET_ONLY_REPAIR", "uniqueDocuments": len(seen), "copies": len(evidence),
        "planSha256": sha256_file(OUT / "legacy-reviewed.json"), "records": evidence,
        "files": [{"file": p.name, "beforeSha256": digest(b), "afterSha256": digest(a), "documents": ids} for p,b,a,ids in staged],
        "inputAndCoordinatesUnchanged": True, "consigneeTargetsUnchanged": True, "seconds": time.perf_counter()-started}
    atomic_publish_json(OUT / "legacy-publication.json", receipt)
    for path, before, after, changed in staged:
        if path.read_bytes() != before:
            raise ValueError("concurrent data change")
        atomic_write_bytes(path, after)
        manifest["files"][path.name] = digest(after)
    manifest["sharedNotifyFootnoteRepair20261101"] = {"path": str((OUT/"legacy-publication.json").relative_to(ROOT)), "sha256": sha256_file(OUT/"legacy-publication.json")}
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({k:v for k,v in receipt.items() if k not in {"records","files"}}), flush=True)


if __name__ == "__main__":
    main()
