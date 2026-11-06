"""Read-only factual coverage/accounting screens for the R30 readiness review."""

import collections
import json
import re
from pathlib import Path

from audit_r30_training_readiness import OUT, SPLITS, compact, features, leaves, stats


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    summaries, all_flags, id_sets, enum_sets = {}, {}, {}, {}
    for split in SPLITS:
        fields = collections.defaultdict(collections.Counter)
        enums = collections.defaultdict(set)
        parents = collections.Counter()
        long_parents = collections.Counter()
        equal_parents = collections.Counter()
        description_lengths, accounting, order, column_words = [], [], [], []
        public_ids = collections.defaultdict(set)
        prompt_flags, matched_container_ids = [], 0
        for r in records(Path(f"data/curated/mpci-bl-r30-paddle-centroids-v1/{split}.jsonl")):
            t, identity = r["target"], r["documentId"]
            parent = r["sourceDocumentId"]
            parents[parent] += 1
            f = features(t)
            if f["long_inventory"]:
                long_parents[parent] += 1
            if f["equal_multi_allocation"]:
                equal_parents[parent] += 1
            for path, value in leaves(t):
                key = re.sub(r"/\d+", "/*", path)
                if isinstance(value, str) and (
                    path.endswith(
                        ("Category", "/country", "/unit", "/paymentArrangement", "/negotiability")
                    )
                    or "/temperatureSetpoint/" in path
                ):
                    fields[key][value] += 1
                    if not path.endswith("/country"):
                        enums[key].add(value)
                if isinstance(value, (int, float)) and "/temperatureSetpoint/" in path:
                    fields[key][str(value)] += 1
                if path.endswith("equipmentIdentifier") or path == "/billOfLadingNumber":
                    public_ids["container" if path.endswith("equipmentIdentifier") else "bill"].add(
                        value
                    )
                if (
                    isinstance(value, str)
                    and path.endswith(("/name", "/description"))
                    and re.search(
                        r"\b(?:as an AI|return only|do not include|generate a|"
                        r"sampled goods|packageMaterial)\b",
                        value,
                        re.I,
                    )
                ):
                    prompt_flags.append({"id": identity, "path": path, "value": value})
            raw = compact(r["joinedRawText"])
            cs = t.get("containerInformation", [])
            positions = [raw.find(compact(c["equipmentIdentifier"])) for c in cs]
            matched_container_ids += sum(x >= 0 for x in positions)
            if len(cs) > 1 and all(x >= 0 for x in positions) and positions != sorted(positions):
                order.append(
                    {"id": identity, "parent": parent, "first_occurrence_positions": positions}
                )
            for g in t.get("goodsItemDetails", []):
                description_lengths.append(len(g.get("description", "").split()))
                pp, aa = g.get("numberAndTypeOfPackages", []), g.get("splitGoodsPlacement", [])
                if (
                    pp
                    and aa
                    and len(aa) == len(cs)
                    and all("packageQuantity" in x for x in pp + aa)
                ):
                    total, allocated = (
                        sum(x["packageQuantity"] for x in pp),
                        sum(x["packageQuantity"] for x in aa),
                    )
                    if total != allocated:
                        accounting.append(
                            {
                                "id": identity,
                                "parent": parent,
                                "total": total,
                                "allocated": allocated,
                            }
                        )
            for p in r["pages"]:
                # Exclude vertical text; report skinny, long chains, not isolated abbreviations.
                runs = collections.defaultdict(list)
                for a in p["regions"]:
                    x, y, xx, yy = a["box"]
                    if (
                        (xx - x) / p["width"] < 0.06
                        and yy - y < xx - x
                        and re.search(r"[A-Za-z]{3}", a["text"])
                    ):
                        runs[round(x / p["width"] * 100)].append(a)
                for run in runs.values():
                    if len(run) >= 10:
                        column_words.append(
                            {
                                "id": identity,
                                "parent": parent,
                                "page": p["index"],
                                "regions": [a["index"] for a in run],
                                "texts": [a["text"] for a in run],
                            }
                        )
        id_sets[split], enum_sets[split] = public_ids, enums
        summaries[split] = {
            "fields": fields,
            "parent_count": len(parents),
            "top_parents": parents.most_common(10),
            "long_inventory_parents": long_parents,
            "equal_allocation_parents": equal_parents,
            "description_words": stats(description_lengths),
            "accounting_screen_count": len(accounting),
            "order_screen_count": len(order),
            "narrow_chain_screen_count": len(column_words),
            "prompt_leak_screens": prompt_flags,
            "container_id_literal_witnesses": matched_container_ids,
        }
        all_flags[split] = {"accounting": accounting, "order": order, "narrow_chains": column_words}
    absent = {}
    for field, values in enum_sets["validation"].items():
        gap = values - enum_sets["real-train"][field] - enum_sets["synthetic-train"][field]
        if gap:
            absent[field] = sorted(gap)
    overlaps = {
        split: {
            kind: sorted(values & id_sets["validation"][kind])
            for kind, values in id_sets[split].items()
        }
        for split in ("real-train", "synthetic-train")
    }
    (OUT / "supervision.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "screens": all_flags,
                "validation_enum_gaps": absent,
                "exact_validation_identifier_overlap": overlaps,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                k: {
                    n: v[n]
                    for n in (
                        "accounting_screen_count",
                        "order_screen_count",
                        "narrow_chain_screen_count",
                    )
                }
                for k, v in summaries.items()
            }
        )
    )


if __name__ == "__main__":
    main()
