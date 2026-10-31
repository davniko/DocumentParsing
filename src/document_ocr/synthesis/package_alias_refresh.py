"""Bounded alias migration of repaired records, never historical full replay.

Only scalar, currently traceable owners are eligible. Compound-ledger replay and
expanded page reflow belong to fresh synthesis, not this text-only migration.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from difflib import SequenceMatcher

from document_ocr.synthesis.curated import assign, digest, flat
from document_ocr.synthesis.curated_positions import PAGE_MARKER, transfer_existing_positions


def _offsets(lines):
    result, total = [], 0
    for line in lines:
        result.append(total)
        total += len(line)
    return result


def _content(value):
    return "".join(c for c in value.decode().casefold() if c.isalnum())


def locate(old: bytes, new: bytes, start: int, end: int) -> tuple[int, int]:
    """Transfer equal text only; changed surrounding facts cannot establish ownership."""
    before, after = old.splitlines(keepends=True), new.splitlines(keepends=True)
    a, b = _offsets(before), _offsets(after)
    line = bisect_right(a, start) - 1
    if start == end or bisect_right(a, end - 1) - 1 != line:
        raise ValueError("multiline_or_empty_owner")
    lo, hi = start - a[line], end - a[line]
    for tag, i, j, k, last in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if not i <= line < j:
            continue
        destination = k + line - i
        mapped = None
        if tag == "equal":
            mapped = b[destination] + lo
        elif (
            tag == "replace"
            and j - i == last - k
            and _content(before[line]) == _content(after[destination])
        ):
            # Container-ID punctuation repairs can shift a noun without changing
            # its shipment facts. Regenerated goods or changed quantities cannot.
            for block in SequenceMatcher(
                None, before[line], after[destination], autojunk=False
            ).get_matching_blocks():
                if block.a <= lo and hi <= block.a + block.size:
                    mapped = b[destination] + block.b + lo - block.a
                    break
        if mapped is None:
            raise ValueError("changed_or_ambiguous_current_owner")

        # A coincidentally equal line on a different page is not the same owner.
        def page(lines):
            return sum(bool(PAGE_MARKER.fullmatch(s.decode().strip())) for s in lines)

        if page(before[:line]) != page(after[:destination]):
            raise ValueError("owner_page_changed")
        return mapped, mapped + end - start
    raise ValueError("unmapped_owner")


def owners(candidate: dict, current: dict, certificate: dict) -> list[dict]:
    """Join historical render receipts to certified scalar owners and current text."""
    for row in (candidate, current):
        if digest(row["joinedRawText"].encode()) != row["joinedRawTextSha256"]:
            raise ValueError("raw_hash_mismatch")
    if (candidate["documentId"], candidate["sourceDocumentId"]) != (
        current["documentId"],
        current["sourceDocumentId"],
    ):
        raise ValueError("record_identity_mismatch")
    raw, now = candidate["joinedRawText"].encode(), current["joinedRawText"].encode()
    if digest(raw) != candidate["proof"]["renderedSha256"]:
        raise ValueError("historical_render_hash_mismatch")
    actual, shift, previous = {}, 0, 0
    for edit in candidate["proof"]["edits"]:
        if edit["byteStart"] < previous or edit["byteEnd"] < edit["byteStart"]:
            raise ValueError("historical_edit_order")
        previous = edit["byteEnd"]
        start = edit["byteStart"] + shift
        end = start + len(edit["after"].encode())
        shift += len(edit["after"].encode()) - (edit["byteEnd"] - edit["byteStart"])
        if raw[start:end] != edit["after"].encode():
            raise ValueError("historical_render_proof_mismatch")
        key = (edit["key"], edit["occurrence"])
        if key in actual:
            raise ValueError("duplicate_owner")
        actual[key] = (start, end)
    found = []
    for slot in certificate["slots"]:
        if slot["mode"] != "scalar":
            raise ValueError("non_scalar_owner_requires_ledger_replay")
        if len(slot["categoryPaths"]) != 1:
            raise ValueError("ambiguous_category_owner")
        key = (slot["key"], slot["occurrence"])
        if key not in actual:
            raise ValueError("historical_owner_not_in_receipt")
        start, end = actual[key]
        a, b = locate(raw, now, start, end)
        if now[a:b] != raw[start:end]:
            raise ValueError("current_owner_mismatch")
        found.append({**slot, "currentStart": a, "currentEnd": b, "currentText": now[a:b].decode()})
    if not found:
        raise ValueError("no_explicit_scalar_owners")
    previous = 0
    for slot in sorted(found, key=lambda s: s["currentStart"]):
        if slot["currentStart"] < previous:
            raise ValueError("overlapping_current_owners")
        previous = slot["currentEnd"]
    return found


def _normalized(text):
    """Casing/whitespace comparison with exact character back-pointers."""
    chars, positions = [], []
    for i, char in enumerate(text):
        if char.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                positions.append(i)
        else:
            folded = char.casefold()
            chars.extend(folded)
            positions.extend([i] * len(folded))
    if chars and chars[-1] == " ":
        chars.pop()
        positions.pop()
    return "".join(chars), positions


def project_literal_targets(target, raw, edits):
    """Project aliases only inside the literal OCR occurrence of a target string.

    A repeated target must project identically at every occurrence. No synonym,
    summary, global noun replacement or non-goods-field update is permitted.
    """
    normalized_raw, raw_positions = _normalized(raw)
    character_edits = [
        (
            len(raw.encode()[: e["byteStart"]].decode()),
            len(raw.encode()[: e["byteEnd"]].decode()),
            e["after"],
        )
        for e in edits
    ]
    changes = []

    def visit(value, path):
        if isinstance(value, dict):
            return {k: visit(v, f"{path}.{k}".strip(".")) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if not isinstance(value, str) or not value or path.endswith(("Category", ".unit")):
            return value
        normalized_value, value_positions = _normalized(value)
        if (
            normalized_value not in normalized_raw
            and (
                re.fullmatch(r"goodsItemDetails\[\d+\]\.description", path)
                or "handlingInstructions" in path
            )
            and any(
                re.search(r"(?<!\w)" + re.escape(raw[a:b].casefold()) + r"(?!\w)", normalized_value)
                for a, b, _ in character_edits
            )
        ):
            raise ValueError("noncontiguous_literal_target_requires_owned_projection")
        results = set()
        for match in re.finditer(
            r"(?<!\w)" + re.escape(normalized_value) + r"(?!\w)", normalized_raw
        ):
            lo, hi = match.span()
            start, end = raw_positions[lo], raw_positions[hi - 1] + 1
            local = []
            for a, b, replacement in character_edits:
                if b <= start or a >= end:
                    continue
                if not (start <= a < b <= end):
                    raise ValueError("target_partially_overlaps_alias")
                x = bisect_right(raw_positions, a) - 1
                y = bisect_right(raw_positions, b - 1) - 1
                local.append((value_positions[x - lo], value_positions[y - lo] + 1, replacement))
            result = value
            for a, b, replacement in sorted(local, reverse=True):
                result = result[:a] + replacement + result[b:]
            results.add(result)
        if len(results) > 1:
            raise ValueError("repeated_literal_target_projects_differently")
        result = next(iter(results)) if results else value
        if result != value:
            if not (
                re.fullmatch(r"goodsItemDetails\[\d+\]\.description", path)
                or "handlingInstructions" in path
            ):
                raise ValueError("alias_intersects_nonliteral_target")
            changes.append({"path": path, "before": value, "after": result})
        return result

    return visit(target, ""), changes


def refresh(row, slots, presenter, *, seed):
    """Choose footprint-bounded production aliases; preserve all other current facts."""
    vocabulary, metrics = presenter.vocabulary, presenter.metrics
    presenter.require_certified([row["sourceDocumentId"]])
    groups = defaultdict(list)
    for slot in slots:
        groups[slot["categoryPaths"][0]].append(slot)
    target_leaves = flat({"documentPatch": row["target"]})
    evidence, edits = [], []
    for path, group in groups.items():
        category = target_leaves[path]
        family = vocabulary.entries[category]["family"]
        supported = [s for s in group if vocabulary.supports(s["currentText"], category)]
        if not supported:
            raise ValueError("no_current_specificity_witness")
        witness = min(supported, key=lambda s: (s["role"] != "declaration", s["currentStart"]))
        for slot in group:
            before = slot["currentText"]
            a, b = slot["currentStart"], slot["currentEnd"]
            if row["joinedRawText"].encode()[a:b] != before.encode():
                raise ValueError("stale_current_owner")
            presented = (
                family
                if slot is not witness and slot["role"] in {"allocation", "repetition"}
                else category
            )
            if slot.get("protectedSourceDetail"):
                after, choice = before, {"style": "protected"}
            else:
                # Entire noun ownership is required. Qualifiers such as HDPE or
                # HMHDPE must not disappear merely because PLASTIC is supported.
                allowed_before = {
                    form
                    for cat in {category, family}
                    for style in ("full", "short", "optional")
                    for form in vocabulary.forms(cat, style)
                }
                if before.strip().upper() not in allowed_before:
                    raise ValueError("unregistered_or_finer_current_wording")
                styles = (
                    ("full", "optional")
                    if slot is witness and family != category
                    else ("full", "short", "optional")
                )
                after, choice = vocabulary.choose(
                    presented,
                    styles,
                    presenter.policy.weights(),
                    f"refresh:{seed}:{row['documentId']}:{slot['key']}:{slot['occurrence']}",
                    maximum_advance=metrics.length(before),
                    metrics=metrics,
                )
            if metrics.length(after) > metrics.length(before) + 1e-7 or "\n" in after:
                raise ValueError("alias_exceeds_current_footprint")
            entry = {
                "key": slot["key"],
                "occurrence": slot["occurrence"],
                "byteStart": a,
                "byteEnd": b,
                "before": before,
                "after": after,
                "categoryPath": path,
                "category": category,
                "presentedCategory": presented,
                "role": slot["role"],
                "witness": slot is witness,
                "beforeAdvance": metrics.length(before),
                "afterAdvance": metrics.length(after),
                **choice,
            }
            evidence.append(entry)
            if before != after:
                edits.append(entry)
    edits.sort(key=lambda e: e["byteStart"])
    raw = row["joinedRawText"].encode()
    rendered, previous = b"", 0
    for edit in edits:
        if edit["byteStart"] < previous:
            raise ValueError("overlapping_alias_edits")
        rendered += raw[previous : edit["byteStart"]] + edit["after"].encode()
        previous = edit["byteEnd"]
    rendered += raw[previous:]
    proof = {"sourceSha256": digest(raw), "renderedSha256": digest(rendered), "edits": edits}
    positioned, geometry = transfer_existing_positions(
        row["joinedRawText"], row["positionedText"], rendered.decode(), proof
    )

    # A fitting scalar alias must preserve every existing anchor, including
    # explicit unknown anchors. Do not fabricate a more precise glyph centroid.
    def suffixes(text):
        return [line.rsplit(" ||", 1)[-1] for line in text.splitlines()]

    if suffixes(positioned) != suffixes(row["positionedText"]):
        raise ValueError("current_coordinate_suffix_changed")
    updated = deepcopy(row)
    updated["target"], target_changes = project_literal_targets(
        row["target"], row["joinedRawText"], edits
    )
    updated.update(
        joinedRawText=rendered.decode(),
        joinedRawTextSha256=digest(rendered),
        positionedText=positioned,
        positionedTextSha256=digest(positioned.encode()),
    )
    return updated, {
        "documentId": row["documentId"],
        "sourceDocumentId": row["sourceDocumentId"],
        "seed": seed,
        "beforeRecordSha256": digest(row),
        "afterRecordSha256": digest(updated),
        "proof": proof,
        "decisions": evidence,
        "targetChanges": target_changes,
        "geometryMethod": geometry["method"],
        "allCurrentAnchorsPreserved": True,
        "vocabularySha256": presenter.policy.vocabulary.sha256,
        "certificatesSha256": presenter.policy.certificates.sha256,
    }


def apply_reviewed_cleanup(row, repair):
    """Apply an explicitly reviewed, hash-pinned data correction after alias replay.

    This is not a heuristic repair or a production generation rule. The external
    review owns the exact text and target operations, including their rationale.
    """
    if row["documentId"] != repair["documentId"] or digest(row) != repair["beforeRecordSha256"]:
        raise ValueError("reviewed_cleanup_precondition")
    before = row["joinedRawText"]
    edits = []
    for operation in repair["rawEdits"]:
        matches = list(re.finditer(re.escape(operation["before"]), before))
        occurrence = operation.get("occurrence")
        if occurrence is None and len(matches) != 1:
            raise ValueError("reviewed_cleanup_not_unique")
        if occurrence is not None and (
            type(occurrence) is not int or not 1 <= occurrence <= len(matches)
        ):
            raise ValueError("reviewed_cleanup_occurrence_out_of_range")
        match = matches[0 if occurrence is None else occurrence - 1]
        a = len(before[: match.start()].encode())
        edits.append(
            {
                **operation,
                "key": "reviewed-wording-cleanup",
                "occurrence": occurrence if occurrence is not None else 1,
                "byteStart": a,
                "byteEnd": a + len(operation["before"].encode()),
            }
        )
    edits.sort(key=lambda e: e["byteStart"])
    raw, previous, pieces = before.encode(), 0, []
    for edit in edits:
        if edit["byteStart"] < previous:
            raise ValueError("reviewed_cleanup_overlaps")
        pieces.extend([raw[previous : edit["byteStart"]], edit["after"].encode()])
        previous = edit["byteEnd"]
    rendered = b"".join([*pieces, raw[previous:]]).decode()
    proof = dict(sourceSha256=digest(raw), renderedSha256=digest(rendered.encode()), edits=edits)
    positioned, receipt = transfer_existing_positions(
        before, row["positionedText"], rendered, proof
    )
    updated = deepcopy(row)
    for operation in repair["targetEdits"]:
        path = operation["path"]
        value = flat(updated["target"])[path]
        if value.count(operation["before"]) != 1:
            raise ValueError("reviewed_cleanup_target_precondition")
        assign(updated["target"], path, value.replace(operation["before"], operation["after"], 1))
    updated.update(
        joinedRawText=rendered,
        joinedRawTextSha256=digest(rendered.encode()),
        positionedText=positioned,
        positionedTextSha256=digest(positioned.encode()),
    )
    return updated, dict(
        beforeRecordSha256=digest(row),
        afterRecordSha256=digest(updated),
        proof=proof,
        targetEdits=repair["targetEdits"],
        geometry=receipt,
    )
