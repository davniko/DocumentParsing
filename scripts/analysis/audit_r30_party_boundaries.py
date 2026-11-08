"""Read-only, streaming party-supervision census and reviewed error taxonomy.

Text support is a screening statistic, not an ownership adjudication. In
particular, column interleaving, postal deduplication and linked continuations
can make a correct target non-contiguous in the serialized OCR.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r30-party-boundary-audit-20261108"
DATA = ROOT / "data/curated/mpci-bl-r30-paddle-centroids-v1"
OLD = ROOT / "data/curated/mpci-bl-r29-positions-v1"


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def party_fields(target):
    for role, value in target.get("parties", {}).items():
        for i, party in enumerate(value if isinstance(value, list) else [value]):
            if not isinstance(party, dict):
                continue
            prefix = f"{role}[{i}]" if isinstance(value, list) else role
            for field in ("name", "addressLine"):
                if party.get(field):
                    yield f"{prefix}.{field}", party[field]


def compact(value):
    return "".join(c for c in value.upper() if c.isalnum())


def words(value):
    return re.findall(r"[^\W_]+", value.upper())


def label_flags(path, value):
    result = []
    if any(c in value for c in "\n\r\t"):
        result.append("line_break_or_tab")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        result.append("control_character")
    if "||" in value or "--- PAGE" in value:
        result.append("serialization_marker")
    if re.search(r"\b(?:RETURN ONLY|AS AN AI|GENERATE A|DO NOT INCLUDE)\b", value, re.I):
        result.append("prompt_leak_screen")
    if path.endswith("addressLine") and re.search(
        r"\b(?:TEL(?:EPHONE)?|FAX|EMAIL|E-MAIL|TAX\s+(?:ID|NO)|"
        r"VAT\s*(?:NO|ID)|ENTERPRISE CODE|REGISTRATION NO)\s*[:.#]",
        value,
        re.I,
    ):
        result.append("contact_registration_caption_screen")
    return result


def support(value, raw, normalized, token_set):
    if " ".join(value.upper().split()) in " ".join(raw.upper().split()):
        return "literal_after_whitespace"
    if compact(value) in normalized:
        return "contiguous_after_punctuation_and_spacing"
    if set(words(value)) <= token_set:
        return "tokens_present_noncontiguous"
    # Character-wrapped native text can break a lexical token across regions.
    missing = [w for w in words(value) if w not in token_set and compact(w) not in normalized]
    return "missing_lexeme_screen" if missing else "fragmented_or_interleaved"


def taxonomy():
    source = ROOT / "docs/analysis/r30-training-results-20261108/errors.csv"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "79ec4d31af306e762b596151d3a87dcc78b14195b096af075f0bfab3fe8c6189"
    ), "re-review taxonomy after source predictions change"
    selected = [
        r
        for r in csv.DictReader(source.open())
        if r["run"] == "R30_1044"
        and r["field"].startswith("parties.")
        and r["field"].endswith((".name", ".addressLine"))
    ]
    # Manual classifications indexed against the immutable, hash-checked CSV.
    # Saved output retains complete source/field identities for review.
    groups = {
        "name_address_boundary": [
            1,
            2,
            16,
            17,
            19,
            20,
            32,
            33,
            34,
            35,
            43,
            44,
            50,
            51,
            53,
            54,
            66,
            67,
            68,
            69,
            70,
            71,
            80,
            81,
        ],
        "wrong_role_or_alternative_owner": [21, 22, 24, 26, 38, 41, 57, 58, 59, 60],
        "contact_caption_or_graphic_contamination": [12, 13, 52, 55, 82],
        "omitted_party": [3, 4, 39, 40],
        "missing_continuation_or_postal_component": [25, 36, 45, 47, 61, 62],
        "spelling_or_value_rewrite": [5, 7, 9, 11, 14, 15, 23, 48, 72, 75, 76, 77, 78, 79],
        "duplicated_postal_component": [49, 63, 64],
        "punctuation_or_marker_only": [8, 10, 37, 42, 56],
        "whitespace_only": [0, 6, 18, 46, 65, 73, 74],
        "unparseable_document": [27, 28, 29, 30, 31, 83, 84, 85, 86],
    }
    assert len(selected) == 87
    indexed = {i: group for group, indices in groups.items() for i in indices}
    assert sorted(indexed) == list(range(87))
    assert sum(map(len, groups.values())) == 87
    for i, r in enumerate(selected):
        r["primary_class"] = indexed[i]
        if r["primary_class"] == "whitespace_only":
            assert re.sub(r"\s", "", json.loads(r["gold"])) == re.sub(
                r"\s", "", json.loads(r["predicted"])
            )
    return selected


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, flags, unsupported = {}, [], []
    for split in ("real-train", "synthetic-train", "validation"):
        before = {
            r["documentId"]: dict(party_fields(r["target"])) for r in rows(OLD / f"{split}.jsonl")
        }
        stats, support_counts, changes = Counter(), defaultdict(Counter), Counter()
        examples = defaultdict(list)
        parents = defaultdict(set)
        hash_state = hashlib.sha256()
        with (DATA / f"{split}.jsonl").open("rb") as stream:
            for line in stream:
                hash_state.update(line)
                r = json.loads(line)
                stats["documents"] += 1
                raw = r["joinedRawText"]
                normalized, token_set = compact(raw), set(words(raw))
                current = dict(party_fields(r["target"]))
                previous = before[r["documentId"]]
                for key in previous.keys() | current.keys():
                    if previous.get(key) != current.get(key):
                        kind = (
                            "added"
                            if key not in previous
                            else "removed"
                            if key not in current
                            else "changed"
                        )
                        changes[kind] += 1
                        if kind == "changed" and compact(previous[key]) == compact(current[key]):
                            changes["changed_spacing_or_punctuation_only"] += 1
                for path, value in current.items():
                    field = path.split(".")[-1]
                    stats[field] += 1
                    found = label_flags(path, value)
                    for flag in found:
                        stats[flag] += 1
                        flags.append(
                            {
                                "split": split,
                                "id": r["documentId"],
                                "parent": r["sourceDocumentId"],
                                "path": path,
                                "value": value,
                                "flag": flag,
                            }
                        )
                    tier = support(value, raw, normalized, token_set)
                    support_counts[field][tier] += 1
                    if tier != "literal_after_whitespace" and len(examples[tier]) < 8:
                        examples[tier].append(
                            {
                                "id": r["documentId"],
                                "parent": r["sourceDocumentId"],
                                "path": path,
                                "value": value,
                            }
                        )
                    if tier == "missing_lexeme_screen":
                        missing = [
                            w
                            for w in words(value)
                            if w not in token_set and compact(w) not in normalized
                        ]
                        unsupported.append(
                            {
                                "split": split,
                                "id": r["documentId"],
                                "parent": r["sourceDocumentId"],
                                "path": path,
                                "value": value,
                                "missing": missing,
                            }
                        )
                        parents[tier].add(r["sourceDocumentId"])
                    # Screen whole repeated party name in its postal target.
                    if field == "addressLine":
                        name = current.get(path.removesuffix("addressLine") + "name", "")
                        if len(compact(name)) >= 10 and compact(name) in compact(value):
                            flags.append(
                                {
                                    "split": split,
                                    "id": r["documentId"],
                                    "parent": r["sourceDocumentId"],
                                    "path": path,
                                    "value": value,
                                    "name": name,
                                    "flag": "whole_name_inside_address_screen",
                                }
                            )
                            stats["whole_name_inside_address_screen"] += 1
                for g in r["target"].get("goodsItemDetails", []):
                    if isinstance(g.get("description"), str):
                        stats["descriptions"] += 1
                        stats["description_has_newline"] += "\n" in g["description"]
        summaries[split] = {
            "counts": stats,
            "support": support_counts,
            "r29_to_r30_party_changes": changes,
            "sha256": hash_state.hexdigest(),
            "screen_examples": examples,
            "missing_lexeme_parents": len(parents["missing_lexeme_screen"]),
        }
    errors = taxonomy()
    counts = Counter(r["primary_class"] for r in errors)
    for name, value in (
        ("census.json", summaries),
        ("flags.json", flags),
        ("missing-lexeme-screens.json", unsupported),
        ("reviewed-error-taxonomy.json", errors),
    ):
        (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    timing = {
        "seconds": time.perf_counter() - started,
        "maxRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "providerCalls": 0,
        "dataChanged": False,
        "errorCounts": counts,
        "errorDocuments": len({r["document_id"] for r in errors}),
    }
    (OUT / "run.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run": timing,
                "counts": {k: v["counts"] for k, v in summaries.items()},
                "support": {k: v["support"] for k, v in summaries.items()},
                "flags": len(flags),
                "missingLexemeScreens": len(unsupported),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
