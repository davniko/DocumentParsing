"""Reconcile historical owners to current text with exact full-text replay.

No unmatched context is guessed or silently discarded. Changed source contracts,
scoped cargo shipments and unowned edits require their separate lineage proof.
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import re
from decimal import Decimal, InvalidOperation
import time
from collections import Counter
from pathlib import Path

from r30_owner_frames import unique_owner_spans

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_templates import current_path
from document_ocr.synthesis.template_compiler.descendant import _number_word_value, _date_candidates
from document_ocr.synthesis.scoped_cargo_refresh import (
    apply_edits,
    current_interval,
    equal_interval,
    reconstruct_source,
    rendered_interval,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r30-synthetic-migration-20261103"
ART = ROOT / "artifacts/paddle-migration/r30-synthetic-migration-20261103"


def read(path):
    return json.loads(path.read_text())


def retained_semantic_literal(field, old_value, new_value, rendered):
    """Constrain a retained surface only when its unchanged fact proves it.

    Receipt precision and date presentation are preserved. This is not a
    formatter: a changed fact or a different rendered spelling still needs
    its separate composition authority.
    """
    if old_value != new_value or old_value is None or isinstance(old_value, bool):
        return None
    if isinstance(old_value, (int, float)):
        try:
            parsed = Decimal(rendered.replace(",", ""))
            if parsed.is_finite() and parsed == Decimal(str(old_value)):
                return " ".join(rendered.upper().split())
        except InvalidOperation:
            pass
        if field.endswith("packageQuantity") and type(old_value) is int:
            # Parse the complete owner, not every possible subphrase of a
            # long cargo description carrying historical shared target paths.
            if _number_word_value(rendered.replace("-", " ")) == old_value:
                return " ".join(rendered.upper().split())
    if field.endswith("Date") and isinstance(old_value, str):
        if {d.isoformat() for d in _date_candidates(rendered)} == {old_value}:
            return " ".join(rendered.upper().split())
    return None


def compose(candidate, current, *, method="character_diff", owner_patterns=None, retained_equipment_ids=None):
    """Preserve complete historical owner boundaries; require exact reconstruction."""
    if candidate["sourceDocumentId"] != current["sourceDocumentId"]:
        raise ValueError("different source membership needs explicit lineage")
    old_facts = flat(candidate["target"]["documentPatch"])
    current_facts = flat(current["target"])
    if retained_equipment_ids is not None:
        old_ids = [r["equipmentIdentifier"] for r in candidate["target"]["documentPatch"]["containerInformation"]]
        new_ids = [r["equipmentIdentifier"] for r in current["target"]["containerInformation"]]
        if (len(old_ids) != len(new_ids) or len(set(old_ids)) != len(old_ids)
                or len(set(new_ids)) != len(new_ids)
                or retained_equipment_ids != dict(zip(old_ids, new_ids, strict=True))):
            raise ValueError("retained equipment rename lacks an exact ordered bijection")
    for field in old_facts.keys() & current_facts.keys():
        if (
            field.startswith(("goodsItemDetails[", "containerInformation["))
            and field.endswith((".value", ".packageQuantity", ".equipmentIdentifier"))
            and old_facts[field] != current_facts[field]
            and not (retained_equipment_ids is not None
                     and field.endswith(".equipmentIdentifier")
                     and retained_equipment_ids.get(old_facts[field]) == current_facts[field])
        ):
            raise ValueError("cargo facts changed; scoped lineage is required")
    source = reconstruct_source(candidate)
    before, after = candidate["joinedRawText"].encode(), current["joinedRawText"].encode()
    result = copy.deepcopy(candidate)
    old_spans = [
        rendered_interval(candidate["proof"], e["byteStart"], e["byteEnd"])
        for e in result["proof"]["edits"]
    ]
    if method == "literal_frames":
        literals = []
        for edit in result["proof"]["edits"]:
            authorities = set()
            for pointer in edit.get("targetPaths", []):
                field = current_path(pointer).removeprefix("documentPatch.")
                if field.endswith((".country", "Category")) or field not in current_facts:
                    continue
                old_value, new_value = old_facts.get(field), current_facts[field]
                retained = retained_semantic_literal(field, old_value, new_value, edit["after"])
                if retained is not None:
                    authorities.add(retained)
                if not isinstance(old_value, (str, int)) or not isinstance(
                    new_value, type(old_value)
                ):
                    continue
                if " ".join(str(old_value).upper().split()) == " ".join(
                    edit["after"].upper().split()
                ):
                    authorities.add(" ".join(str(new_value).upper().split()))
            if len(authorities) > 1:
                raise ValueError("shared literal owner has conflicting current target authorities")
            literals.append(next(iter(authorities)) if authorities else None)
        patterns = [owner_patterns.get((e["key"], e["byteStart"], e["byteEnd"]))
                    for e in result["proof"]["edits"]] if owner_patterns is not None else None
        new_spans = unique_owner_spans(before, after, old_spans, expected_literals=literals,
                                      expected_patterns=patterns)
    elif method == "character_diff":
        new_spans = []
        for a, b in old_spans:
            if a == b:
                # A receipt can intentionally delete an unsupported source
                # statement. It has no rendered value to relocate. Preserve
                # that deletion only when both adjacent bytes remain one
                # unchanged run; never absorb a new insertion at this point.
                if not 0 < a < len(before):
                    raise ValueError("empty owner lacks two retained boundary witnesses")
                lo, hi = equal_interval(before, after, a - 1, a + 1)
                if hi - lo != 2:
                    raise ValueError("empty owner boundary witnesses changed")
                new_spans.append((lo + 1, lo + 1))
            else:
                new_spans.append(current_interval(before, after, a, b))
    else:
        raise ValueError("unknown explicit composition method")
    previous_end, owner_values = 0, {}
    for edit, (a, b), (lo, hi) in zip(result["proof"]["edits"], old_spans, new_spans, strict=True):
        if before[a:b].decode() != edit["after"]:
            raise ValueError("historical owner does not match its output")
        if lo < previous_end:
            raise ValueError("current complete owners overlap")
        previous_end = hi
        edit["after"] = after[lo:hi].decode()
        owner_values.setdefault(edit["key"], set()).add(edit["after"])
    if apply_edits(source, result["proof"]["edits"]) != after:
        raise ValueError("current text has changes outside the historical complete owners")
    result["documentId"] = current["documentId"]
    result["joinedRawText"] = current["joinedRawText"]
    result["joinedRawTextSha256"] = result["proof"]["renderedSha256"] = digest(after)
    result["target"] = {
        "schemaVersion": candidate["target"]["schemaVersion"],
        "documentPatch": copy.deepcopy(current["target"]),
    }
    for key, values in owner_values.items():
        if len(values) == 1:
            value = next(iter(values))
            for field in ("renderValues", "surfaceValues"):
                if key in result.get(field, {}):
                    result[field][key] = value
    result["migrationComposition"] = {
        "status": "EXACT_CURRENT_TEXT_REPLAY_REQUIRES_NATIVE_RENDER_REVIEW",
        "currentRecordSha256": digest(current),
        "historicalCandidateId": candidate["documentId"],
        "unchangedScenario": True,
        "allCurrentBytesReconstructed": True,
        "alignmentMethod": method,
    }
    if reconstruct_source(result) != source or result["scenario"] != candidate["scenario"]:
        raise ValueError("composition changed source authority or sampled facts")
    return result


def compose_with_compiled_identifiers(candidate, current, compiled):
    """Recover reviewed non-target identifier owners absent from an old receipt.

    Older receipts predate the private-reference cleanup. Their unedited source
    identifiers may now have anonymized values. Only complete disjoint owners
    from the matching compiled source can extend the historical edit coverage;
    the existing exact replay and unchanged-cargo checks still decide acceptance.
    """
    source = reconstruct_source(candidate)
    if compiled["document_id"] != candidate["sourceDocumentId"] or compiled[
        "source_sha256"
    ] != digest(source):
        raise ValueError("compiled identifier authority belongs to a different source")
    augmented = copy.deepcopy(candidate)
    edits = augmented["proof"]["edits"]
    added = []
    for binding in compiled["bindings"]:
        if binding["value_kind"] != "identifier" or binding["target_paths"]:
            continue
        for occurrence in binding["occurrences"]:
            if occurrence["target_paths"]:
                continue
            start, end = occurrence["byte_start"], occurrence["byte_end"]
            text = occurrence["source_text"]
            if not 0 <= start < end <= len(source) or source[start:end].decode() != text:
                raise ValueError("compiled identifier span differs from source authority")
            if any(e["byteStart"] < end and e["byteEnd"] > start for e in edits):
                continue
            edit = {
                "key": binding["logical_key"],
                "byteStart": start,
                "byteEnd": end,
                "before": text,
                "after": text,
                "targetPaths": [],
            }
            edits.append(edit)
            added.append(edit)
    if not added:
        raise ValueError("compiled source adds no eligible historical identifier owners")
    edits.sort(key=lambda e: e["byteStart"])
    result = compose(augmented, current)
    added_keys = {(e["key"], e["byteStart"], e["byteEnd"]) for e in added}
    changed_additions = [
        e
        for e in result["proof"]["edits"]
        if (e["key"], e["byteStart"], e["byteEnd"]) in added_keys and e["after"] != e["before"]
    ]
    # Source-equal additions are not changes and must not introduce new native
    # owners that duplicate an already reviewed Paddle-only owner.
    result["proof"]["edits"] = [
        e
        for e in result["proof"]["edits"]
        if (e["key"], e["byteStart"], e["byteEnd"]) not in added_keys or e["after"] != e["before"]
    ]
    if not changed_additions:
        raise ValueError("compiled identifier owners do not explain a retained change")
    if (
        reconstruct_source(result) != source
        or apply_edits(source, result["proof"]["edits"]) != current["joinedRawText"].encode()
    ):
        raise ValueError("compiled identifier composition lost exact retained-text replay")
    result["migrationComposition"]["addedCompiledIdentifierOwners"] = changed_additions
    return result


def main(method="character_diff", limit=None):
    started = time.perf_counter()
    inventory = read(OUT / "receipt-inventory.json")
    current = {
        r["documentId"]: r for r in map(json.loads, (ROOT / inventory["baseline"]["path"]).open())
    }
    counts, results = Counter(), []
    held = None
    if method == "literal_frames":
        held = {
            r["id"]
            for r in read(OUT / "composition-proposals.json")["records"]
            if r["status"] == "HELD"
        }
    for entry in inventory["records"]:
        if not (
            entry["status"] == "RECEIPT_COMPOSITION_REQUIRED"
            and entry["lineage"] == "historical"
            and entry["matchingSourcePin"]
        ):
            continue
        identity = entry["id"]
        if held is not None and identity not in held:
            continue
        if limit is not None and len(results) >= limit:
            break
        try:
            path = ROOT / entry["candidate"]
            if (
                sha256_file(path) != entry["candidateSha256"]
                or digest(current[identity]) != entry["recordSha256"]
            ):
                raise ValueError("composition input changed since inventory")
            result = compose(read(path), current[identity], method=method)
            result["migrationComposition"]["historicalCandidate"] = {
                "path": entry["candidate"],
                "sha256": entry["candidateSha256"],
            }
            directory = (
                "composition-proposals"
                if method == "character_diff"
                else "composition-frame-proposals"
            )
            output = ART / directory / (identity + ".json")
            atomic_publish_json(output, result)
            row = {
                "id": identity,
                "source": entry["source"],
                "status": "EXACT_REPLAY_PROPOSAL",
                "path": str(output.relative_to(ROOT)),
                "sha256": sha256_file(output),
            }
        except (ValueError, KeyError, IndexError) as exc:
            row = {"id": identity, "source": entry["source"], "status": "HELD", "error": str(exc)}
        counts[row.get("error", row["status"])] += 1
        results.append(row)
        if len(results) % 500 == 0:
            print(json.dumps({"processed": len(results), "counts": counts}), flush=True)
    report = {
        "counts": dict(counts),
        "records": results,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    name = "composition-proposals" if method == "character_diff" else "composition-frame-proposals"
    if limit is not None:
        name += f"-probe{limit}"
    atomic_write_json(OUT / (name + ".json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("character_diff", "literal_frames"), default="character_diff"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args.method, args.limit)
