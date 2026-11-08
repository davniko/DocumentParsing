"""Compile and verify native compound-package owners; no sampling or publication."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.paddle import pages_from_record
from document_ocr.synthesis.paddle_render import spans_for

ROOT = Path(__file__).resolve().parents[2]
CATALOG = (
    ROOT
    / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30"
    / "ownership-585-20261105/catalog.json"
)
OUT = ROOT / "artifacts/synthesis-probes/r30-binding-repair-20261108"


def inspect():
    catalog = json.loads(CATALOG.read_text())
    config = json.loads((ROOT / catalog["samplingConfig"]["path"]).read_text())
    contracts = json.loads((ROOT / config["declared_packages"]["contracts"]["path"]).read_text())[
        "sources"
    ]
    results = []
    for sid, entry in catalog["templates"].items():
        template = json.loads((ROOT / entry["template"]["path"]).read_text())
        transfer = json.loads((ROOT / template["transfer"]["path"]).read_text())
        source = json.loads((ROOT / template["source"]["path"]).read_text())
        pages = pages_from_record(source)
        overlay = template["overlay"]
        for i, declaration in enumerate(contracts[sid]["declarations"]):
            start, end = declaration["start"], declaration["end"]
            key = f"declared-packages:{i}"
            owner = f"{key}@{start}:{end}"
            if any(s["owner"] == owner for s in transfer["slots"]) or any(
                r["key"] == key
                and r.get("byteStart", start) == start
                and r.get("byteEnd", end) == end
                for r in overlay["overrides"]
            ):
                continue
            row = {
                "source": sid,
                "index": i,
                "declaration": declaration,
                "singleOnly": config["declared_packages"]["source_group_counts"].get(sid) == [1],
                "components": [],
                "absent": overlay.get("absentEdits", []),
            }
            for component, occurrence in declaration["owners"]:
                candidates = []
                for s in transfer["slots"]:
                    k, interval = s["owner"].rsplit("@", 1)
                    a, b = map(int, interval.split(":"))
                    if k == component and start <= a < b <= end:
                        candidates.append((a, b, s))
                overrides = [
                    r
                    for r in overlay["overrides"]
                    if r["key"] == component
                    and start <= r.get("byteStart", start) < r.get("byteEnd", end) <= end
                ]
                # Report uncertain selectors instead of inferring occurrence ownership.
                resolved = []
                if len(candidates) == 1:
                    a, b, s = candidates[0]
                    exact = [
                        o
                        for o in overrides
                        if o.get("byteStart", a) == a and o.get("byteEnd", b) == b
                    ]
                    if len(exact) == 1:
                        resolved = [vars(span) for span in spans_for(exact[0], pages)]
                    elif not exact and s["status"] == "exact_measured_owner_candidate":
                        resolved = s["spans"]
                elif len(overrides) == 1:
                    resolved = [vars(span) for span in spans_for(overrides[0], pages)]
                row["components"].append(
                    {
                        "key": component,
                        "occurrence": occurrence,
                        "slots": [s for _, _, s in candidates],
                        "overrides": overrides,
                        "spans": resolved,
                    }
                )
            results.append(row)
    atomic_write_json(
        OUT / "inventory.json",
        {
            "catalog": {"path": str(CATALOG.relative_to(ROOT)), "sha256": sha256_file(CATALOG)},
            "rows": results,
        },
    )
    print(
        json.dumps(
            {
                "missing": len(results),
                "sources": len({r["source"] for r in results}),
                "singleOnly": sum(r["singleOnly"] for r in results),
                "unresolved": [
                    [
                        r["source"][4:12],
                        r["index"],
                        [c["key"] for c in r["components"] if not c["spans"]],
                    ]
                    for r in results
                    if any(not c["spans"] for c in r["components"])
                ],
            },
            indent=2,
        )
    )


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def compose():
    catalog = json.loads(CATALOG.read_text())
    cfg = json.loads((ROOT / catalog["samplingConfig"]["path"]).read_text())
    catalog["declaredPackages"] = {
        "contracts": cfg["declared_packages"]["contracts"],
        "singleRowOnly": sorted(
            s for s, v in cfg["declared_packages"]["source_group_counts"].items() if v == [1]
        ),
    }
    rows = json.loads((OUT / "inventory.json").read_text())["rows"]
    templates, audit, held = {}, [], []
    for row in rows:
        if row["singleOnly"]:
            continue
        sid = row["source"]
        if sid not in templates:
            templates[sid] = json.loads(
                (ROOT / catalog["templates"][sid]["template"]["path"]).read_text()
            )
        t = templates[sid]
        source = json.loads((ROOT / t["source"]["path"]).read_text())
        pages = pages_from_record(source)
        spans = []
        keys = list(dict.fromkeys(c["key"] for c in row["components"]))
        selected = []
        for key in keys:
            comp = next(c for c in row["components"] if c["key"] == key)
            if comp["overrides"]:
                for o in comp["overrides"]:
                    selected.append(o)
                    spans.extend(spans_for(o, pages))
            else:
                for slot in comp["slots"]:
                    if slot["status"] != "exact_measured_owner_candidate":
                        raise ValueError(f"Unreviewed component {sid}: {slot['owner']}")
                    from document_ocr.spatial_inputs.paddle import RegionSpan

                    spans.extend(RegionSpan(**s) for s in slot["spans"])
        expected_keys = {c["key"] for c in row["components"] if c["slots"] or c["overrides"]}
        if expected_keys != set(keys):
            # Reviewed CHEREAU numeric alias: the quantity anchor already owns
            # the same literal 1, but fresh vehicle sampling calls it vehicle_count.
            if sid.startswith("doc_4831fa04") and set(keys) - expected_keys == {"vehicle_count"}:
                rule = {
                    "key": "vehicle_count",
                    "byteStart": 462,
                    "byteEnd": 463,
                    "spans": [[0, 44, 0, 1]],
                }
                assert source["pages"][0]["regions"][44]["text"] == "1 UNP. USED CHEREAU TRAILER"
                t["overlay"]["overrides"].append(rule)
                spans = list(spans_for(rule, pages)) + spans
                for ex in t["overlay"]["extras"]:
                    if ex.get("existingKey") == "anchor:documentPatch.cargoPackages[0].quantity":
                        ex["existingOwners"] = [
                            {
                                "key": ex.pop("existingKey"),
                                "byteStart": ex.pop("existingByteStart"),
                                "byteEnd": 463,
                            },
                            {"key": "vehicle_count", "byteStart": 462, "byteEnd": 463},
                        ]
            else:
                held.append(
                    {
                        "source": sid,
                        "index": row["index"],
                        "reason": "missing component",
                        "keys": sorted(set(keys) - expected_keys),
                    }
                )
                continue
        d = row["declaration"]
        key = f"declared-packages:{row['index']}"
        if len(keys) == 1 and len(selected) == 1:
            # A private compound clause can intentionally project only its first
            # line. Preserve that reviewed recipe rather than joining all lines.
            rule = {**selected[0], "key": key, "byteStart": d["start"], "byteEnd": d["end"]}
            gaps = []
        else:
            grouped = {}
            for s in spans:
                grouped.setdefault((s.page, s.region), []).append(s)
            merged, gaps = [], []
            for (p, r), parts in grouped.items():
                parts = sorted(set(parts), key=lambda s: s.start)
                for a, b in pairwise(parts):
                    if a.end > b.start:
                        raise ValueError(f"Overlapping component owners {sid}")
                    gap = pages[p].regions[r].text[a.end : b.start]
                    if gap.strip():
                        gaps.append(gap)
                merged.append([p, r, min(s.start for s in parts), max(s.end for s in parts)])
            rule = {"key": key, "byteStart": d["start"], "byteEnd": d["end"], "spans": merged}
        t["overlay"]["overrides"].append(rule)
        complete = spans_for(rule, pages)
        for extra in t["overlay"]["extras"]:
            if "packageSummary" not in extra:
                continue
            parts = spans_for(extra, pages)
            if all(
                any(
                    (s.page, s.region) == (p.page, p.region) and p.start <= s.start < s.end <= p.end
                    for p in complete
                )
                for s in parts
            ):
                extra.setdefault("consumedBy", []).append(f"{key}@{d['start']}:{d['end']}")
        audit.append(
            {
                "source": sid,
                "index": row["index"],
                "before": d["source"],
                "native": [s.expected for s in spans_for(rule, pages)],
                "gaps": gaps,
                "rule": rule,
            }
        )
    # Follow-up review of all 456 overlays found ten dependent recipes whose
    # scalar proof key is consumed by the compound mode (one outside the 58).
    for sid, entry in catalog["templates"].items():
        if sid[4:12] not in {
            "157fadd2",
            "2d5864aa",
            "3929e40b",
            "4ae247ca",
            "7484c804",
            "76dcec12",
            "e16b3787",
        }:
            continue
        if sid not in templates:
            templates[sid] = json.loads((ROOT / entry["template"]["path"]).read_text())
        t = templates[sid]
        for ex in t["overlay"]["extras"]:
            if sid.startswith("doc_157fadd2") and ex.get("fromKey") == "held-numeric:1":
                ex.pop("fromKey")
                ex["sourceSlice"] = {
                    "byteStart": 846,
                    "byteEnd": 904,
                    "source": "5 PALLET\n188 CARTON(S) IN 5 PALLET(S)\n188 CTNS IN 5 PALLET",
                }
                ex["lineStride"] = 3
            elif ex.get("fromKey") in {"package_count", "packages"}:
                ex.pop("fromKey")
                ex.pop("fromByteStart", None)
                ex["packageSummary"] = {"goods": 0, "part": "quantity"}
            elif ex.get("fromKey") in {"package_type", "package_noun"}:
                ex["surfaceKey"] = ex.pop("fromKey")
                ex.pop("fromByteStart", None)
                ex["surfaceIndex"] = 0
        for rule in t["overlay"]["overrides"]:
            if sid.startswith("doc_157fadd2") and rule["key"] in {
                "held-numeric:1",
                "declared-packages:0",
            }:
                rule["lineStride"] = 3
            if (
                sid.startswith("doc_76dcec12")
                and "valueRecipe" in rule
                and rule["valueRecipe"].get("fromByteStart") == 2161
            ):
                rule["valueRecipe"] = {"packageSummary": {"goods": 0, "part": "quantity"}}
    for sid, t in templates.items():
        if sid.startswith("doc_157fadd2"):
            t["overlay"]["newWording"]["signer"] = (
                "Fictional six-letter personal surname in title case. A readable alphabetic "
                "name, not an identifier or random character string."
            )
        if sid.startswith("doc_9f4d3ead"):
            t["overlay"]["newWording"]["stamp_company"] = (
                "Fictional source-only shipping company stamp name, two short words with mixed "
                "casing. Use a new identity, not the original company. Do not include GROUP, "
                "which is outside this owned slot."
            )
        path = OUT / "templates" / f"{sid}.json"
        atomic_write_json(path, t)
        catalog["templates"][sid]["template"] = pin(path)
    atomic_write_json(OUT / "catalog-draft.json", catalog)
    atomic_write_json(OUT / "composition-review.json", {"composed": audit, "held": held})
    print(
        json.dumps(
            {
                "composed": len(audit),
                "templates": len(templates),
                "held": held,
                "gaps": [
                    [r["source"][4:12], r["index"], r["before"], r["native"], r["gaps"]]
                    for r in audit
                    if r["gaps"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "compose"], default="inspect", nargs="?")
    args = parser.parse_args()
    {"inspect": inspect, "compose": compose}[args.action]()
