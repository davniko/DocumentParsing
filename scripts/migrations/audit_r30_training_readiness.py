"""Read-only, local training-risk census; screens are not semantic adjudications.

Run from the repository root. Uses the pinned tokenizer without network access,
the trainer's target canonicalizer/serializer, and independent geometric probes.
No provider calls or dataset mutations occur.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import resource
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from document_ocr.spatial_inputs.paddle import pages_from_record, serialize
from document_ocr.training.tasks import canonical_json, get_training_task

OUT = Path("docs/analysis/r30-training-readiness-20261106")
SPLITS = ("real-train", "synthetic-train", "validation")
TOKENIZER = "google/t5gemma-2-270m-270m"
REVISION = "7c38f16641f455ef0685b18431faf1b17722d5a1"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def compact(value):
    return re.sub(r"[^\w]", "", value).upper()


def stats(values):
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": float(a.mean()),
        **{
            k: float(np.percentile(a, p))
            for k, p in (
                ("min", 0),
                ("p50", 50),
                ("p90", 90),
                ("p95", 95),
                ("p99", 99),
                ("max", 100),
            )
        },
    }


def leaves(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from leaves(v, path + "/" + k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from leaves(v, path + "/" + str(i))
    else:
        yield path, value


def features(target):
    cs = target.get("containerInformation", [])
    gs = target.get("goodsItemDetails", [])
    alloc = [a for g in gs for a in g.get("splitGoodsPlacement", [])]
    ids = [c["equipmentIdentifier"] for c in cs]
    complete = bool(
        len(cs) > 1
        and len(alloc) == len(cs)
        and {a["equipmentIdentifier"] for a in alloc} == set(ids)
        and all("packageQuantity" in a for a in alloc)
    )
    equal = complete and len({a["packageQuantity"] for a in alloc}) == 1
    types = {(c.get("sizeCategory"), c.get("typeCategory")) for c in cs}
    multirow = any(len(g.get("numberAndTypeOfPackages", [])) > 1 for g in gs)
    return {
        "containers": len(cs),
        "multi_container": len(cs) > 1,
        "long_inventory": len(cs) >= 10,
        "mixed_equipment": len(types) > 1,
        "multi_package_rows": multirow,
        "mixed_multirow": len(types) > 1 and multirow,
        "complete_multi_allocation": complete,
        "equal_multi_allocation": equal,
        "equal_shared_prefix_8plus": bool(
            equal and len(cs) >= 8 and len({i[:4] for i in ids}) == 1
        ),
        "dg": any(g.get("dangerousGoods") for g in gs),
        "multi_dg": any(len(g.get("dangerousGoods", [])) > 1 for g in gs),
        "thermal_setpoint": any("temperatureSetpoint" in c for c in cs),
        "refrigerated_equipment": any(
            "REEF" in c.get("typeCategory", "") or "REFRIG" in c.get("typeCategory", "") for c in cs
        ),
        "forwarding_agent": bool(target.get("parties", {}).get("forwardingAgent")),
        "delivery_agent": bool(target.get("parties", {}).get("deliveryAgent")),
        "transshipment": bool(target.get("route", {}).get("transshipmentPort")),
        "handling": any(g.get("handlingInstructions") for g in gs),
    }


def geometry(row):
    """Screen area overlaps, not claimed font/raster collisions or wrong ownership."""
    heights, densities, horizontal_densities = [], [], []
    reversals, major, collision, regions, overlap = 0, 0, 0, 0, 0
    overlap_examples = []
    for p in row["pages"]:
        rs = [r for r in p["regions"] if r["text"].strip()]
        regions += len(rs)
        if not rs:
            continue
        b = np.asarray([r["box"] for r in rs], dtype=float)
        wh = b[:, 2:] - b[:, :2]
        ctr = (b[:, :2] + b[:, 2:]) / 2 / [p["width"], p["height"]]
        heights.extend(wh[:, 1] / p["height"] * 1000)
        densities.extend((len(r["text"]) * h / w) for r, (w, h) in zip(rs, wh, strict=True))
        horizontal_densities.extend(
            len(r["text"]) * h / w for r, (w, h) in zip(rs, wh, strict=True) if w >= h
        )
        reversals += int(np.count_nonzero(np.diff(ctr[:, 1]) < 0))
        major += int(np.count_nonzero(np.diff(ctr[:, 1]) < -0.02))
        collision += len(rs) - len({tuple(x) for x in np.round(ctr * 1000).astype(int)})
        intersection = np.maximum(
            0,
            np.minimum(b[:, None, 2:], b[None, :, 2:]) - np.maximum(b[:, None, :2], b[None, :, :2]),
        ).prod(axis=2)
        ratio = intersection / np.minimum(wh.prod(axis=1)[:, None], wh.prod(axis=1)[None, :])
        ii, jj = np.where(np.triu(ratio > 0.20, k=1))
        overlap += len(ii)
        for i, j in zip(ii[:4], jj[:4], strict=True):
            overlap_examples.append(
                {
                    "page": p["index"],
                    "regions": [rs[i]["index"], rs[j]["index"]],
                    "fraction": float(ratio[i, j]),
                    "texts": [rs[i]["text"], rs[j]["text"]],
                }
            )
    return {
        "regions": regions,
        "pages": len(row["pages"]),
        "height_median": float(np.median(heights)) if heights else 0,
        "density_median": float(np.median(densities)) if densities else 0,
        "density_p99": float(np.percentile(densities, 99)) if densities else 0,
        "horizontal_density_p99": float(np.percentile(horizontal_densities, 99))
        if horizontal_densities
        else 0,
        "y_reversals": reversals,
        "large_y_reversals": major,
        "centroid_collisions": collision,
        "overlap_pairs_20pct": overlap,
        "overlap_examples": overlap_examples,
    }


def literal_screen(row):
    raw = row["joinedRawText"]
    flat = compact(raw)
    tokens = set(re.findall(r"\w+", raw.upper()))
    flags = []
    for path, value in leaves(row["target"]):
        if not isinstance(value, str) or not value:
            continue
        if (
            any(
                x in path
                for x in ("equipmentIdentifier", "sealNumbers", "phoneNumbers", "emailAddresses")
            )
            and compact(value) not in flat
        ):
            flags.append({"path": path, "value": value, "kind": "not_contiguous_normalized"})
        if path.endswith("/description") or "/handlingInstructions/" in path:
            words = re.findall(r"\w+", value.upper())
            missing = [w for w in words if w not in tokens]
            if missing:
                flags.append(
                    {
                        "path": path,
                        "kind": "words_absent_anywhere",
                        "missing": missing,
                        "fraction": len(missing) / len(words),
                        "value": value,
                    }
                )
    return flags


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=REVISION, local_files_only=True)
    task = get_training_task("bill_of_lading_extraction_v7_reduced_direct")
    prompt = Path("prompts/training/input_only.txt").read_text()
    # PromptTemplate strips the template at load time, not the substituted OCR.
    prompt = prompt.strip()
    summaries, identifiers, hashes, parents, artifacts = {}, {}, {}, {}, {}
    baseline = {}

    def process_split(version, split):
        root = Path(
            f"data/curated/mpci-bl-{version}-"
            + ("positions-v1" if version == "r29" else "paddle-centroids-v1")
        )
        key = f"{version}/{split}"
        counters, categories, field_presence, changes = (collections.Counter() for _ in range(4))
        inputs, targets, geometries, changes_detail = [], [], [], []
        rows_out, flags = [], []
        identifiers[key], hashes[key], parents[key] = set(), set(), set()
        batch = []

        def flush():
            if not batch:
                return
            si = tokenizer(
                [prompt.replace("{{document_text}}", r["positionedText"]) for r, _ in batch],
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            ti = tokenizer(
                text_target=[canonical_json(t) for _, t in batch],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            for (r, target), a, b in zip(batch, si, ti, strict=True):
                inputs.append(len(a))
                targets.append(len(b) + 1)
                f = features(target)
                for name, value in f.items():
                    counters[name] += int(value)
                for name in {re.sub(r"/\d+", "/*", p) for p, _ in leaves(target)}:
                    field_presence[name] += 1
                for cat in {
                    p.get("typeCategory", "[CATEGORY_ABSENT]")
                    for g in target.get("goodsItemDetails", [])
                    for p in g.get("numberAndTypeOfPackages", [])
                }:
                    categories[cat] += 1
                mini = {
                    "id": r["documentId"],
                    "parent": r.get("sourceDocumentId"),
                    "input_tokens": len(a),
                    "target_tokens": len(b) + 1,
                    **f,
                }
                if version == "r30":
                    geo = geometry(r)
                    geometries.append(geo)
                    mini["geometry"] = {k: v for k, v in geo.items() if k != "overlap_examples"}
                    screened = literal_screen(r)
                    if screened or geo["overlap_pairs_20pct"]:
                        flags.append(
                            {
                                "id": r["documentId"],
                                "parent": r.get("sourceDocumentId"),
                                "literal": screened,
                                "overlap": geo["overlap_examples"],
                            }
                        )
                    old = baseline[(split, r["documentId"])]
                    ol, nl = dict(leaves(old)), dict(leaves(target))
                    delta = [
                        {"path": p, "before": ol.get(p), "after": nl.get(p)}
                        for p in sorted(set(ol) | set(nl))
                        if ol.get(p) != nl.get(p)
                    ]
                    if delta:
                        changes_detail.append({"id": r["documentId"], "changes": delta})
                        for p in {re.sub(r"/\d+", "/*", d["path"]) for d in delta}:
                            changes[p] += 1
                else:
                    baseline[(split, r["documentId"])] = target
                rows_out.append(mini)
            batch.clear()

        path = root / f"{split}.jsonl"
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["documentId"] in identifiers[key]:
                    raise ValueError("duplicate ID")
                identifiers[key].add(row["documentId"])
                hashes[key].add(digest(row["positionedText"]))
                parents[key].add(row.get("sourceDocumentId", row["documentId"]))
                target = task.canonicalize(row["target"])
                if version == "r30":
                    pages = pages_from_record(row)
                    for mode, name in (
                        ("centroid", "positionedText"),
                        ("bbox", "bboxText"),
                        ("none", "joinedRawText"),
                    ):
                        if serialize(pages, geometry=mode) != row[name]:
                            raise ValueError(f"view mismatch {row['documentId']} {name}")
                batch.append((row, target))
                if len(batch) == 32:
                    flush()
        flush()
        artifact = OUT / f"{version}-{split}.json"
        artifact.write_text(
            json.dumps(
                {"records": rows_out, "screens": flags, "target_changes": changes_detail},
                ensure_ascii=False,
                indent=2,
            )
        )
        artifacts[key] = {"path": str(artifact), "sha256": digest(artifact.read_text())}
        summaries[key] = {
            "count": len(rows_out),
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "features": counters,
            "package_documents": categories,
            "field_presence": field_presence,
            "input_tokens": stats(inputs),
            "target_tokens": stats(targets),
            "over_source_19200": sum(v > 19200 for v in inputs),
            "over_target_5500": sum(v > 5500 for v in targets),
            "over_old_generation_3072": sum(v > 3072 for v in targets),
            "changed_targets": len(changes_detail),
            "changed_fields": changes,
            "parents": len(parents[key]),
            "geometry": {
                k: stats([g[k] for g in geometries])
                for k in geometries[0]
                if k != "overlap_examples"
            }
            if geometries
            else {},
        }
        print(
            key,
            "count",
            len(rows_out),
            "max_input",
            max(inputs),
            "max_target",
            max(targets),
            flush=True,
        )

    for version in ("r29", "r30"):
        for split in SPLITS:
            process_split(version, split)
    leakage = {}
    for split in ("real-train", "synthetic-train"):
        a, b = f"r30/{split}", "r30/validation"
        leakage[split] = {
            "document_id_overlap": sorted(identifiers[a] & identifiers[b]),
            "exact_input_overlap": len(hashes[a] & hashes[b]),
            "source_identity_overlap": sorted(parents[a] & parents[b]),
        }
    report = {
        "status": "CENSUS_AND_SCREENS_NOT_SEMANTIC_CERTIFICATION",
        "summaries": summaries,
        "artifacts": artifacts,
        "split_overlap": leakage,
        "tokenizer": {"name": TOKENIZER, "revision": REVISION, "target_eos": True},
        "seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "provider_calls": 0,
    }
    (OUT / "census.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps({k: report[k] for k in ("seconds", "peak_rss_mib", "split_overlap")}), flush=True
    )


if __name__ == "__main__":
    main()
