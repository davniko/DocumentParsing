"""Independent R31 readback, paired catalog controls and Stage-B coverage census."""

from __future__ import annotations

import json
import resource
import statistics
import time
import unicodedata
from collections import Counter

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.labeling_agents.source_casing import selected_fields
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from scripts.migrations.migrate_r31_casing import (
    BASE,
    CATALOG,
    DEST,
    OUT,
    ROOT,
    pin,
    read,
    validate_case_only,
)


def readback():
    manifest = read(DEST / "manifest.json")
    counts = {}
    for split, expected in manifest["aggregateRecords"].items():
        path = ROOT / expected["path"]
        if sha256_file(path) != expected["sha256"]:
            raise ValueError("R31 aggregate checksum differs")
        stats = Counter()
        for old, new in zip(
            map(json.loads, (BASE / f"{split}.jsonl").open()),
            map(json.loads, path.open()),
            strict=True,
        ):
            validate_case_only(old["target"], new["target"])
            if {k: v for k, v in old.items() if k != "target"} != {
                k: v for k, v in new.items() if k not in {"target", "casingMigration"}
            }:
                raise ValueError("R31 changed non-target data")
            stats["documents"] += 1
            stats["regions"] += sum(len(p["regions"]) for p in new["pages"])
            stats["changedDocuments"] += old["target"] != new["target"]
            fields = dict(selected_fields(old["target"]))
            for field, value in selected_fields(new["target"]):
                stats["fields"] += 1
                stats["changedFields"] += fields[field] != value
        if sha256_file(BASE / f"{split}.jsonl") != manifest["audit"]["sourceHashes"][split]:
            raise ValueError("R30 input snapshot changed")
        counts[split] = dict(stats)
    return counts


def paired_controls():
    old_pin = read(BASE / "manifest.json")["templateCatalog"]
    new_pin = pin(CATALOG / "catalog.json")
    old, new = PaddleTemplateCatalog(ROOT, old_pin), PaddleTemplateCatalog(ROOT, new_pin)
    reviews = read(OUT / "reviewed-regions.json")
    changed = set(reviews["templateRegions"]) | set(reviews["templateProjections"])
    selected = {}
    for row in map(json.loads, (BASE / "synthetic-train.jsonl").open()):
        parent = row["sourceDocumentId"]
        if parent in old.entries and parent not in selected:
            selected[parent] = row["evidence"]
    controls = [(p, e) for p, e in selected.items() if p[4:12] in changed]
    controls += [(p, e) for p, e in sorted(selected.items()) if p[4:12] not in changed][:9]
    results = []
    times = [0.0, 0.0]
    warm_times = [[], []]
    for parent, evidence in controls:
        path = ROOT / evidence["candidate"]["path"]
        wording = read(evidence["wording"]["path"]) if evidence.get("wording") else None
        pair = []
        for i, cat in enumerate((old, new)):
            started = time.perf_counter()
            try:
                pair.append(cat.render(path, wording_receipt=wording))
            except (ValueError, KeyError) as exc:
                pair.append({"error": f"{type(exc).__name__}: {exc}"})
            times[i] += time.perf_counter() - started
        if "error" in pair[0] or "error" in pair[1]:
            results.append(
                {
                    "parent": parent,
                    "baseline": pair[0].get("error"),
                    "r31": pair[1].get("error"),
                    "status": "unchanged-baseline-hold" if pair[0] == pair[1] else "REGRESSION",
                }
            )
            continue
        validate_case_only(pair[0]["target"], pair[1]["target"])
        if {k: v for k, v in pair[0].items() if k != "target"} != {
            k: v for k, v in pair[1].items() if k not in {"target", "sourceCasing"}
        }:
            raise ValueError("paired rendering changed input or proof")
        again = new.render(path, wording_receipt=wording)
        if again != pair[1]:
            raise ValueError("R31 native rendering is not deterministic")
        for repetition in range(4):
            # Alternate order after both catalogs have loaded the template.
            for i in (0, 1) if repetition % 2 == 0 else (1, 0):
                started = time.perf_counter()
                value = (old, new)[i].render(path, wording_receipt=wording)
                warm_times[i].append(time.perf_counter() - started)
                if value != pair[i]:
                    raise ValueError("warm paired rendering is not deterministic")
        results.append(
            {
                "parent": parent,
                "status": "case-only-identical-input",
                "fields": pair[1]["sourceCasing"]["fields"],
            }
        )
    return {
        "catalogs": [old_pin, new_pin],
        "results": results,
        "pairedSeconds": times,
        "warmMedianMilliseconds": [statistics.median(v) * 1000 for v in warm_times],
        "warmTotalSeconds": [sum(v) for v in warm_times],
        "counts": dict(Counter(r["status"] for r in results)),
    }


def boundary_plan():
    prior = ROOT / "docs/analysis/r30-training-tweaks-boundaries-20261108"
    screened, reviewed = read(prior / "screen.json"), read(prior / "reviewed.json")
    candidates = read(prior / "nonseparate-cases.json")
    admitted = set(read(CATALOG / "catalog.json")["templates"])
    rejected = reviewed["training_screen_rejection"]
    # Fresh role-level readback: this notify name ends on its own EXPORT.
    # line; the old screen matched its identical consignee block instead.
    additional_rejection = (
        "doc_49feedabd4f29115ea2c3a89d4f2da0afe4db074d6c88fd027ca3d55b76c2885",
        "notifyParties[0]",
    )

    def compact(value):
        return "".join(c for c in unicodedata.normalize("NFKC", value).upper() if c.isalnum())

    result = {}
    all_parents = set()
    for split in ("real-train", "synthetic-train", "validation"):
        if sha256_file(BASE / f"{split}.jsonl") != screened["summary"][split]["sha256"]:
            raise ValueError("boundary evidence is not for current R30 inputs")
        rows = [
            r
            for r in candidates
            if r["split"] == split
            and r["presentation"].startswith("shared")
            and (r["id"], r["role"]) != (rejected["id"], rejected["role"])
            and (r["id"], r["role"]) != additional_rejection
        ]
        tails = [r for r in rows if r["name_anchor"]["matched"] < len(compact(r["name"]))]
        parents, tail_parents = {r["parent"] for r in rows}, {r["parent"] for r in tails}
        if split != "validation":
            all_parents.update(parents & admitted)
        result[split] = {
            "pairedRoles": screened["summary"][split]["counts"]["paired_roles"],
            "sharedRoles": len(rows),
            "sharedDocuments": len({r["id"] for r in rows}),
            "tailRoles": len(tails),
            "tailDocuments": len({r["id"] for r in tails}),
            "tailParents": len(tail_parents),
            "admittedSharedParents": sorted(parents & admitted),
            "admittedTailParents": sorted(tail_parents & admitted),
            "tailRolesByRole": dict(Counter(r["role"].split("[")[0] for r in tails)),
        }
    result["distinctAdmittedTrainingSharedParents"] = len(all_parents)
    result["additionalReviewedScreenRejection"] = {
        "id": additional_rejection[0],
        "role": additional_rejection[1],
        "reason": (
            "Notify name ends on its own EXPORT. line 34; address begins at line 37. "
            "The shared occurrence at line 19 belongs to consignee. "
            "Zero-based final Paddle region indices."
        ),
    }
    total_roles = sum(result[s]["pairedRoles"] for s in ("real-train", "synthetic-train"))
    total_tails = sum(result[s]["tailRoles"] for s in ("real-train", "synthetic-train"))
    result["illustrativeNewSampleRates"] = [
        {
            "newDocuments": n,
            "tailRolesPerNewDocument": h,
            "expectedRate": (total_tails + n * h) / (total_roles + n * total_roles / 10000),
        }
        for n in (500, 750, 1000)
        for h in (1, 1.5, 2)
    ]
    result["qualification"] = (
        "Training is a conservative mechanical screen, not an exhaustive semantic census; "
        "role counts can repeat the same consignee/notify block. Projections assume current "
        "average paired roles per document and are not generation guarantees."
    )
    atomic_write_json(OUT / "boundary-plan.json", result)
    return result


def main():
    started = time.perf_counter()
    boundary_plan()
    result = {"readback": readback(), "paired": paired_controls()}
    result["seconds"] = time.perf_counter() - started
    result["peakRssMiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    atomic_write_json(OUT / "validation.json", result)
    print(
        json.dumps(
            {**result, "paired": {k: v for k, v in result["paired"].items() if k != "results"}},
            indent=2,
        )
    )
    if result["paired"]["counts"].get("REGRESSION"):
        raise ValueError("R31 paired rendering needs review")


if __name__ == "__main__":
    main()
