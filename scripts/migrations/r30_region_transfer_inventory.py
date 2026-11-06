"""Build exact span transfers and explicit review queues for the pinned 20 sources.

This produces rebinding proposals, not admitted templates or anonymized data.
Repeated source occurrences are resolved through their old measured owners;
ambiguous global matches are never silently chosen.
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict

from prepare_r30_paddle_pilot import ART, BASE, OUT, ROOT, read, rows

from document_ocr.atomic import atomic_write_json


def locate(before, pages, hints):
    pattern = re.compile(r"\s+".join(re.escape(s) for s in before.split()), re.IGNORECASE)
    if not before.strip():
        return {"status": "empty_source_review", "matches": []}
    native = {(p["index"], r["index"]): r["text"] for p in pages for r in p["regions"]}

    def matches(refs, *, measured=False):
        refs = list(dict.fromkeys(refs))
        if not refs or any(ref not in native for ref in refs):
            return []
        joined, starts = "", []
        for ref in refs:
            starts.append(len(joined))
            joined += native[ref] + "\n"
        found = []
        for m in pattern.finditer(joined):
            # Joined cells such as 14X20'GP contain separate explicitly owned
            # subfields. An existing measured owner permits that candidate;
            # global searching must still reject incidental identifier substrings.
            if (
                not measured
                and m.start()
                and joined[m.start() - 1].isalnum()
                and before[0].isalnum()
            ):
                continue
            if (
                not measured
                and m.end() < len(joined)
                and joined[m.end()].isalnum()
                and before[-1].isalnum()
            ):
                continue
            spans = []
            for ref, start in zip(refs, starts, strict=True):
                lo, hi = max(m.start(), start), min(m.end(), start + len(native[ref]))
                if hi > lo:
                    spans.append(
                        {
                            "page": ref[0],
                            "region": ref[1],
                            "start": lo - start,
                            "end": hi - start,
                            "expected": native[ref][lo - start : hi - start],
                        }
                    )
            if spans and len({s["page"] for s in spans}) == 1:
                found.append(spans)
        return found

    found = matches(hints, measured=True)
    if len(found) == 1:
        return {"status": "exact_measured_owner_candidate", "spans": found[0]}
    found = []
    for page in pages:
        found += matches([(page["index"], r["index"]) for r in page["regions"]])
    if len(found) == 1:
        return {"status": "unique_literal_candidate_requires_role_review", "spans": found[0]}
    return {"status": "unmatched_or_ambiguous_review", "matches": found}


def main():
    plan = read(OUT / "plan.json")
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json")
    registry = ROOT / cfg["source_contracts"]
    selected = {s["id"]: s for s in plan["sources"] if s["split"] == "real-train"}
    current = {r["documentId"]: r for r in rows(BASE / "synthetic-train.jsonl")}
    provenance = read(
        ROOT / "docs/analysis/paddle-migration-scope-20261102/synthetic-provenance.json"
    )
    candidates = defaultdict(list)
    for item in provenance:
        row = current[item["id"]]
        if row["sourceDocumentId"] not in selected or not item["raw_match"]:
            continue
        candidate = read(item["path"])
        if "proof" not in candidate or candidate["joinedRawText"] != row["joinedRawText"]:
            continue
        candidates[row["sourceDocumentId"]].append((item, candidate))
    totals, inventory = Counter(), []
    for did, selection in selected.items():
        record = read(ROOT / selection["path"])
        source = (registry / did / "source.txt").read_text()
        alignment = read(registry / did / "alignment.json")
        starts, offset = [], 0
        for line in source.splitlines(keepends=True):
            starts.append(offset)
            offset += len(line.encode())
        anchors = {a["line_number"]: a for a in alignment["lines"]}
        template = read(registry / did / "template.json")
        compatible = [
            (i, c)
            for i, c in candidates[did]
            if c["proof"]["sourceSha256"] == hashlib.sha256(source.encode()).hexdigest()
        ]
        # The historical static bindings are only one input to compilation.
        # Actual render proofs also contain curated prose, auxiliary privacy,
        # package expressions and numeric ownership added by the current path.
        # Inventory the union; otherwise a static-only count understates work.
        owners = {}
        for binding in template["bindings"]:
            for occurrence in binding["occurrences"]:
                key = binding["logical_key"]
                lo, hi = occurrence["byte_start"], occurrence["byte_end"]
                owners[key, lo, hi] = {
                    "key": key,
                    "byteStart": lo,
                    "byteEnd": hi,
                    "before": occurrence["source_text"],
                    "targetPaths": binding["target_paths"],
                    "kind": "static_binding",
                }
        for _, candidate in compatible:
            for edit in candidate["proof"]["edits"]:
                key = edit["key"], edit["byteStart"], edit["byteEnd"]
                owners[key] = {
                    k: edit[k] for k in ("key", "byteStart", "byteEnd", "before", "targetPaths")
                } | {"kind": "full_render_proof"}
        mappings = []
        for (key, start, end), owner in sorted(owners.items(), key=lambda item: item[0][1:]):
            if source.encode()[start:end].decode() != owner["before"]:
                raise ValueError(f"source proof mismatch: {did} {key}")
            lo, hi = bisect_right(starts, start), bisect_right(starts, end - 1)
            hints = [
                (anchors[n]["page_index"], i)
                for n in range(lo, hi + 1)
                for i in anchors.get(n, {}).get("region_indices", [])
            ]
            mapped = locate(owner["before"], record["pages"], hints)
            mappings.append({"owner": f"{key}@{start}:{end}", **owner, "hints": hints, **mapped})
            totals[mapped["status"]] += 1
        compatible.sort(
            key=lambda p: (
                -len(
                    p[1]["target"]
                    .get("documentPatch", p[1]["target"])["goodsItemDetails"][0]
                    .get("numberAndTypeOfPackages", [])
                ),
                p[0]["id"],
            )
        )
        choices = [{"id": i["id"], "path": i["path"]} for i, _ in compatible[:3]]
        privacy = []
        for n, edit in enumerate(record["privacyPlan"]["rawEdits"]):
            m = locate(edit["before"], record["pages"], [])
            privacy.append({"index": n, **edit, **m})
        output = {
            "status": "REBINDING_PROPOSALS_NOT_ACCEPTED",
            "source": did,
            "slots": mappings,
            "privacyEdits": privacy,
            "descendants": choices,
            "remainingDescendants": max(0, 3 - len(choices)),
        }
        atomic_write_json(ART / "transfers" / f"{did}.json", output)
        inventory.append(
            {
                "id": did,
                "counts": dict(Counter(m["status"] for m in mappings)),
                "selectedDescendants": len(choices),
                "privacyEdits": len(privacy),
            }
        )
    atomic_write_json(
        OUT / "transfer-inventory.json", {"totals": dict(totals), "sources": inventory}
    )
    print(
        json.dumps(
            {
                "totals": dict(totals),
                "descendants": sum(s["selectedDescendants"] for s in inventory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
