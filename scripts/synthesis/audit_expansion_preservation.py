"""Read back every pre-existing record against the three exact repair inventories."""

import json
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"


def read(path):
    return json.loads(path.read_text())


def main():
    legacy = {r["documentId"]: r["expectedAfterSha256"] for r in read(OUT / "legacy-repair/plan.json")["repairs"]}
    drum = {k:v["afterRecordSha256"] for k,v in read(OUT / "legacy-drum-wording/review-and-publication.json")["records"].items()}
    notify = {r["documentId"]: r for r in read(OUT / "notify-repair/legacy-publication.json")["records"]}
    expected = {**legacy, **drum}
    counts, unique = Counter(), set()
    for path in sorted((OUT / "before").glob("*.jsonl")):
        before = {r["documentId"]: r for r in map(json.loads, path.open())}
        after = {r["documentId"]: r for r in map(json.loads, (DATA / path.name).open())}
        if before.keys() != after.keys():
            raise ValueError("existing membership changed before append")
        for did, old in before.items():
            new = after[did]
            if old == new:
                continue
            if did not in expected and did not in notify:
                raise ValueError("uninventoried existing-record change: " + did)
            restored = dict(new)
            if did in notify:
                receipt = notify[did]
                if new["target"] != receipt["afterTarget"] or digest(new["joinedRawText"].encode()) != receipt["rawSha256"] or digest(new["positionedText"].encode()) != receipt["positionedSha256"]:
                    raise ValueError("notify correction differs from reviewed target/text")
                restored["target"] = receipt["beforeTarget"]
            if digest(restored) != expected.get(did, digest(old)):
                raise ValueError("record differs from final approved repair: " + did)
            if len(old["joinedRawText"].splitlines()) != len(new["joinedRawText"].splitlines()):
                raise ValueError("old record line count changed")
            if [s.rsplit(" ||",1)[-1] for s in old["positionedText"].splitlines()] != [s.rsplit(" ||",1)[-1] for s in new["positionedText"].splitlines()]:
                raise ValueError("old record coordinate suffixes changed")
            unique.add(did)
            counts[path.name] += 1
    if unique != set(expected) | set(notify):
        raise ValueError("not every inventoried repair is present")
    if sha256_file(DATA / "validation.jsonl") != sha256_file(OUT / "before/validation.jsonl"):
        raise ValueError("validation bytes changed")
    atomic_publish_json(OUT / "final/preservation.json", {"status": "PASS", "uniqueRepaired": len(unique), "fileOccurrences": dict(counts),
        "exactInventory": True, "existingMembershipUnchanged": True, "existingCoordinateSuffixesUnchanged": True,
        "validationBytesUnchanged": True, "unreviewedExistingRecordChanges": 0})
    print(len(unique), dict(counts), flush=True)


if __name__ == "__main__":
    main()
