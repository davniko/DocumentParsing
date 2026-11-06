"""Lossless, bidirectional review queues for Paddle rebasing; never an acceptance gate.

Whitespace-only matching is a presentation hint, not semantic ownership. Every
native region remains in the packet, including exact matches. Changed digits,
punctuation, casing, ambiguous repetitions and extra occurrences stay visible.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from rebase_r30_sources import ART, OUT, ROOT, batch_records, read, rows, source_target

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.alignment import parse_lines


@dataclass(frozen=True)
class LiteralIndex:
    text: str
    locations: tuple[tuple[int, int], ...]

    @classmethod
    def build(cls, lines):
        chars, locations = [], []
        for index, text in lines:
            for offset, char in enumerate(text):
                if not char.isspace():
                    chars.append(char)
                    locations.append((index, offset))
        return cls("".join(chars), tuple(locations))

    def find(self, text):
        needle = "".join(text.split())
        if not needle:
            return []
        matches, cursor = [], 0
        while (start := self.text.find(needle, cursor)) >= 0:
            end = start + len(needle)
            refs = self.locations[start:end]
            # Never disguise a substring of a longer identifier as the same value.
            left_inside = (
                start > 0
                and self.locations[start - 1][0] == refs[0][0]
                and self.locations[start - 1][1] + 1 == refs[0][1]
                and self.text[start - 1].isalnum()
                and needle[0].isalnum()
            )
            right_inside = (
                end < len(self.text)
                and self.locations[end][0] == refs[-1][0]
                and self.locations[end][1] == refs[-1][1] + 1
                and self.text[end].isalnum()
                and needle[-1].isalnum()
            )
            # Numeric fragments separated by punctuation still form one value:
            # 200 must not be promoted as the unchanged value in 200,000.
            for edge, direction, char in ((start, -1, needle[0]), (end - 1, 1, needle[-1])):
                adjacent, beyond = edge + direction, edge + 2 * direction
                numeric_tail = (
                    char.isdigit()
                    and 0 <= beyond < len(self.text)
                    and self.text[adjacent] in ",."
                    and self.text[beyond].isdigit()
                    and self.locations[edge][0] == self.locations[beyond][0]
                    and self.locations[beyond][1] - self.locations[edge][1] == 2 * direction
                )
                if direction < 0:
                    left_inside |= numeric_tail
                else:
                    right_inside |= numeric_tail
            if not left_inside and not right_inside:
                spans = []
                for line, offset in refs:
                    if spans and spans[-1][0] == line:
                        spans[-1][2] = offset + 1
                    else:
                        spans.append([line, offset, offset + 1])
                matches.append(spans)
            cursor = start + 1
        return matches


def compare_page(left, right):
    """Return all left occurrences with exact right-side evidence or an explicit queue."""
    index = LiteralIndex.build(right)
    result = []
    claims = defaultdict(list)
    for line, text in left:
        matches = index.find(text)
        status = (
            "literal_unique" if len(matches) == 1 else "ambiguous" if matches else "changed_or_new"
        )
        row = {"index": line, "text": text, "status": status, "matches": matches}
        result.append(row)
        if len(matches) == 1:
            for other, start, end in matches[0]:
                for offset in range(start, end):
                    claims[other, offset].append(len(result) - 1)
    # Extra identical Paddle occurrences must not disappear behind one GLM witness.
    conflicts = {i for owners in claims.values() if len(owners) > 1 for i in owners}
    for i in conflicts:
        result[i]["status"] = "competing_occurrence"
    return result


def binding_proposals(source, owners, active_owner_sets=None):
    """Mechanical native-span proposals only; roles and new owners remain review work.

    Keep measured matches. For other owners accept only one whitespace-equivalent
    occurrence across the document. Reject every competing character claim in one
    pass; never assign repeated strings by guessing reading order or proximity.
    """
    indexes = {
        p["index"]: LiteralIndex.build([(r["index"], r["text"]) for r in p["regions"]])
        for p in source["pages"]
    }
    texts = {(p["index"], r["index"]): r["text"] for p in source["pages"] for r in p["regions"]}
    regions = {(p["index"], r["index"]): r for p in source["pages"] for r in p["regions"]}
    proposals, claims = [], defaultdict(list)
    for owner in owners:
        if owner["status"] == "exact_measured_owner_candidate":
            choices = [owner["spans"]]
            basis = "existing_measured"
        else:
            choices = [
                [
                    {
                        "page": page,
                        "region": r,
                        "start": a,
                        "end": b,
                        "expected": texts[page, r][a:b],
                    }
                    for r, a, b in match
                ]
                for page, index in indexes.items()
                for match in index.find(owner["before"])
            ]
            basis = "whitespace_only"
        row = {
            "owner": owner["owner"],
            "key": owner["key"],
            "byteStart": owner["byteStart"],
            "before": owner["before"],
            "targetPaths": owner.get("targetPaths", []),
            "basis": basis,
            "choices": choices,
            # Full region boundaries expose retained locality qualifiers, caption
            # collisions and adjacent numbers. Boxes remain evidence for review,
            # never an automatic nearest-neighbour role assignment.
            "choiceContext": [
                [
                    {"page": page, **regions[page, region]}
                    for page, region in dict.fromkeys((s["page"], s["region"]) for s in choice)
                ]
                for choice in choices
            ],
            "status": "unique_span_role_review" if len(choices) == 1 else "manual_span_review",
        }
        proposals.append(row)
        if len(choices) == 1:
            for span in choices[0]:
                if (
                    texts[span["page"], span["region"]][span["start"] : span["end"]]
                    != span["expected"]
                ):
                    raise ValueError("measured owner evidence differs from native input")
                for offset in range(span["start"], span["end"]):
                    claims[span["page"], span["region"], offset].append(len(proposals) - 1)
    # A scalar count owner and its multi-row replacement can legitimately own
    # the same source text in different shipments. Only co-active owners clash.
    active = active_owner_sets or [{owner["owner"] for owner in owners}]
    competing = set()
    for claimants in {tuple(sorted(set(v))) for v in claims.values() if len(v) > 1}:
        for shipment in active:
            together = [i for i in claimants if proposals[i]["owner"] in shipment]
            if len(together) > 1:
                competing.update(together)
    for index in competing:
        proposals[index]["status"] = "competing_owner_review"
    return proposals


def packet(source, original):
    glm = defaultdict(list)
    for line in parse_lines(original, len(source["pages"])):
        glm[line.page].append((line.number, line.text))
    pages = []
    for page in source["pages"]:
        native = [(r["index"], r["text"]) for r in page["regions"]]
        native_review = compare_page(native, glm[page["index"]])
        for review, region in zip(native_review, page["regions"], strict=True):
            review["box"] = region["box"]
        pages.append(
            {
                "page": page["index"],
                "native": native_review,
                "glm": compare_page(glm[page["index"]], native),
            }
        )
    transfer_path = ART / "transfers" / (source["documentId"] + ".json")
    owners, unused, active = [], 0, []
    if source["activeTemplate"]:
        transfer = read(transfer_path)
        proofs = [read(ROOT / c["path"])["proof"]["edits"] for c in transfer["descendants"]]
        used = {(e["key"], e["byteStart"], e["byteEnd"]) for proof in proofs for e in proof}
        active = [
            {f"{e['key']}@{e['byteStart']}:{e['byteEnd']}" for e in proof} for proof in proofs
        ]
        for slot in transfer["slots"]:
            if (slot["key"], slot["byteStart"], slot["byteEnd"]) in used:
                owners.append(slot)
            else:
                unused += 1
    privacy = read(ART / "privacy-transfers" / (source["documentId"] + ".json"))
    return {
        "status": "REVIEW_PACKET_NOT_DECISIONS_OR_ACCEPTANCE",
        "source": source["documentId"],
        "nativeFingerprint": source["nativeFingerprint"],
        "pages": pages,
        "currentTarget": source_target(source),
        "privacy": privacy["privacyEdits"],
        "usedTemplateOwners": owners,
        "bindingProposals": binding_proposals(source, owners, active),
        "unusedHistoricalSlotCount": unused,
        "limits": [
            "Literal equivalence does not establish field ownership or label correctness.",
            "No case, punctuation, numeric or Unicode normalization is used.",
            "Joined hits can cross columns: inspect their explicit line spans and geometry.",
            "Matching text, semantic targets and new repeated bindings still need review.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--glm", required=True, help="Original non-anonymized GLM JSONL")
    args = parser.parse_args()
    started = time.perf_counter()
    sources = batch_records(args.batch)
    selected = {s["documentId"] for s in sources}
    originals = {r["documentId"]: r for r in rows(ROOT / args.glm) if r["documentId"] in selected}
    if originals.keys() != selected:
        raise ValueError("original GLM membership does not cover this exact batch")
    totals, owner_counts, records = Counter(), Counter(), []
    for source in sources:
        result = packet(source, originals[source["documentId"]]["joinedRawText"])
        counts = Counter(r["status"] for p in result["pages"] for r in p["native"])
        totals.update(counts)
        owner_counts.update(o["status"] for o in result["usedTemplateOwners"])
        path = OUT / "batches" / args.batch / "review-packets" / (source["documentId"] + ".json")
        atomic_write_json(path, result)
        records.append({"id": source["documentId"], "native": dict(counts)})
    summary = {
        "status": "SCREENING_ONLY_NOT_PUBLICATION",
        "glm": {"path": args.glm, "sha256": sha256_file(ROOT / args.glm)},
        "records": records,
        "native": dict(totals),
        "usedOwners": dict(owner_counts),
        "seconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    atomic_write_json(OUT / "batches" / args.batch / "review-packet-summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
