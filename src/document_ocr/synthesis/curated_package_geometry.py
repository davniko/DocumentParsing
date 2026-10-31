"""Exact package-edit joins for shared synthetic page layout.

These are source-calibrated synthetic boxes, not OCR measurements. Replacements
stay in their measured columns; vertical growth is solved by shared page reflow.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


def _pattern(text):
    return r"\s*".join(re.escape(c) for c in text if not c.isspace())


def _fold_map(text, *, omit_quotes=False):
    indices = [
        i
        for i, char in enumerate(text)
        if not char.isspace() and not (omit_quotes and char in "'\u2019\u2018`\"″“”")
    ]
    return "".join(text[i].upper() for i in indices), indices


def _alnum_map(text):
    indices = [i for i, char in enumerate(text) if char.isalnum()]
    return "".join(text[i].upper() for i in indices), indices


def _punctuation_span(line, region, old, offset):
    """Join a complete certified region before resolving an edited subspan.

    Only punctuation/whitespace may differ. All letters, digits and their order
    must agree, and the complete region must have exactly one line occurrence.
    This is geometry joining, never normalization of the rendered target.
    """
    text, positions = _alnum_map(line["original_text"])
    folded, mapping = _alnum_map(region["text"])
    joined = list(re.finditer(re.escape(folded), text)) if folded else []
    if len(joined) != 1:
        return None
    start, end = joined[0].span()
    lo = sum(i < offset for i in positions)
    hi = sum(i < offset + len(old) for i in positions)
    if not start <= lo < hi <= end or folded[lo - start : hi - start] != _alnum_map(old)[0]:
        return None
    a, b = mapping[lo - start], mapping[hi - start - 1] + 1
    old_positions = _alnum_map(old)[1]
    leading, trailing = old[: old_positions[0]], old[old_positions[-1] + 1 :]
    if leading and region["text"][max(0, a - len(leading)) : a] == leading:
        a -= len(leading)
    if trailing and region["text"][b : b + len(trailing)] == trailing:
        b += len(trailing)
    return region, a, b


def fragment_matches(line, old, offset):
    """Use exact line offsets when the full Paddle region joins to the GLM line.

    A repeated digit is not assigned by its first occurrence. If the full region
    cannot be joined, retain all exact fragment candidates for the caller's
    complete-identical-ownership check.
    """
    if (
        line.get("method") == "punctuation_unique"
        and len(line["regions"]) == 1
        and offset == 0
        and old == line["original_text"]
    ):
        region = line["regions"][0]

        def canonical(value):
            return "".join(c.upper() for c in value if c.isalnum())

        if canonical(old) == canonical(region["text"]):
            return [(region, 0, len(region["text"]))]
    omit_quotes = line.get("method") == "punctuation_unique"
    text, positions = _fold_map(line["original_text"], omit_quotes=omit_quotes)
    old_start = sum(i < offset for i in positions)
    old_end = sum(i < offset + len(old) for i in positions)
    result = []
    for region in line["regions"]:
        folded, mapping = _fold_map(region["text"], omit_quotes=omit_quotes)
        joined = list(re.finditer(re.escape(folded), text)) if folded else []
        if len(joined) == 1:
            start, end = joined[0].span()
            if start <= old_start < old_end <= end:
                a, b = old_start - start, old_end - start
                if folded[a:b] == _fold_map(old, omit_quotes=omit_quotes)[0]:
                    result.append((region, mapping[a], mapping[b - 1] + 1))
                    continue
            # A Paddle region may omit final punctuation present in GLM, so
            # its exact substring ends before the complete owned edit. Join
            # all letters/digits before projecting that edit; never guess a
            # repeated fragment elsewhere in the region.
            mapped = _punctuation_span(line, region, old, offset)
            if mapped is not None:
                result.append(mapped)
        else:
            mapped = _punctuation_span(line, region, old, offset)
            if mapped is not None:
                result.append(mapped)
            else:
                # A uniquely joined complete region that does not contain this
                # source offset cannot own an equal-looking number elsewhere
                # inside it (e.g. package count 20 versus equipment 20GP).
                alnum_region = _alnum_map(region["text"])[0]
                if (
                    alnum_region
                    and len(
                        list(
                            re.finditer(
                                re.escape(alnum_region), _alnum_map(line["original_text"])[0]
                            )
                        )
                    )
                    == 1
                ):
                    continue
                result.extend(
                    (region, m.start(), m.end())
                    for m in re.finditer(_pattern(old), region["text"], re.I)
                )
    return result


def merged_package_lines(source, proof, alignment):
    """Identify certified GLM lines composed of stacked regions in one column.

    A product edit can span several Paddle rows even when its GLM line count is
    unchanged. Such a line needs the existing complete-block reflow, not a guess
    assigning the whole product to one of its physical rows.
    """
    presentation = (proof or {}).get("packagePresentations")
    if presentation is None:
        return set()
    requested = {
        (s["key"], s["occurrence"]) for s in presentation["slots"] if s["before"] != s["after"]
    }
    requested.update(
        (s["key"], s["occurrence"])
        for s in presentation["nonScalarOwners"]
        if s["mode"] == "compound"
    )
    affected = set()
    for edit in proof["edits"]:
        if (edit["key"], edit["occurrence"]) in requested:
            first = source.encode()[: edit["byteStart"]].count(b"\n") + 1
            affected.update(range(first, first + len(edit["before"].split("\n"))))
    return {
        line["line_number"]
        for line in alignment["lines"]
        if line["line_number"] in affected
        and line.get("method") == "spatial_group"
        and len(line["regions"]) > 1
        and max(r["bbox"][0] for r in line["regions"]) < min(r["bbox"][2] for r in line["regions"])
    }


def declared_package_lines(source, proof):
    """Complete declared-list edits may cross adjacent count/type OCR boxes.

    Reflow their already joined logical line as a complete owned block, just as
    for a multiline edit. This also gives an expanded count-only slot its actual
    free column width instead of squeezing letters into one digit's old box.
    """
    if not (proof or {}).get("declaredPackageLists"):
        return set()
    return {
        source.encode()[: edit["byteStart"]].count(b"\n") + 1
        for edit in proof["edits"]
        if edit["key"].startswith("declared-packages:") and "\n" not in edit["before"]
    }


def package_regions(source, proof, alignment, *, excluded_regions=frozenset()):
    """Join complete exact edit groups; repeated fragments need full ownership."""
    presentation = proof.get("packagePresentations")
    if presentation is None:
        return [], []
    requested = {
        (s["key"], s["occurrence"]) for s in presentation["slots"] if s["before"] != s["after"]
    }
    requested.update(
        (s["key"], s["occurrence"])
        for s in presentation["nonScalarOwners"]
        if s["mode"] == "compound"
    )
    edits = proof["edits"]
    line_records = {r["line_number"]: r for r in alignment["lines"]}
    affected = set()
    fragments = []
    for edit in edits:
        prefix = source.encode()[: edit["byteStart"]].decode()
        first = prefix.count("\n") + 1
        column = len(prefix.rsplit("\n", 1)[-1])
        before, after = edit["before"].split("\n"), edit["after"].split("\n")
        if (edit["key"], edit["occurrence"]) in requested:
            affected.update(range(first, first + len(before)))
        if len(before) != len(after):
            continue  # Connected multiline ownership belongs to the vertical path.
        fragments.extend(
            (first + offset, old, new, edit["key"], column if offset == 0 else 0)
            for offset, (old, new) in enumerate(zip(before, after, strict=True))
            if old != new and old.strip()
        )
    jobs, repeated, reasons = defaultdict(list), defaultdict(list), []
    for number, old, new, key, offset in fragments:
        if number not in affected or number not in line_records:
            continue
        line = line_records[number]
        if any((line["page_index"], r["index"]) in excluded_regions for r in line["regions"]):
            continue
        if line["xy"] is None:
            reasons.append(
                {
                    "sourceLines": [number],
                    "reason": "unknown_source_anchor_retained",
                    "blocking": False,
                }
            )
            continue
        matches = fragment_matches(line, old, offset)
        if len(matches) > 1:
            repeated[number, old].append((line, new, matches, key))
            continue
        if not matches:
            reasons.append(
                {
                    "sourceLines": [number],
                    "reason": "unresolved_package_region_join",
                    "key": key,
                    "blocking": True,
                }
            )
            continue
        region, start, end = matches[0]
        jobs[line["page_index"], region["index"]].append((region, start, end, new, number))
    for (number, _old), group in repeated.items():
        line, new, matches, key = group[0]
        if len(group) != len(matches) or len({g[1] for g in group}) != 1:
            reasons.append(
                {
                    "sourceLines": [number],
                    "reason": "ambiguous_repeated_package_region",
                    "key": key,
                    "blocking": True,
                }
            )
            continue
        for region, start, end in matches:
            jobs[line["page_index"], region["index"]].append((region, start, end, new, number))
    result = []
    for (page, index), changes in jobs.items():
        region = changes[0][0]
        parts, cursor = [], 0
        for _, start, end, new, _number in sorted(changes, key=lambda row: row[1]):
            if start < cursor:
                raise ValueError("overlapping edits in package glyph region")
            parts.extend((region["text"][cursor:start], new))
            cursor = end
        parts.append(region["text"][cursor:])
        result.append(
            {
                "page": page,
                "index": index,
                "before": region["text"],
                "after": "".join(parts),
                "bbox": region["bbox"],
                "sourceLines": sorted({r[-1] for r in changes}),
            }
        )
    return result, reasons


def propose_package_regions(context, page, existing, calibration, metrics, wrap):
    """Produce ordinary layout nodes, so horizontal and vertical edits compose."""
    if not context.package_proof:
        return [], []
    excluded = {(page, index) for node in existing for index in node["regionIndices"]}
    matched, reasons = package_regions(
        context.source, context.package_proof, context.alignment, excluded_regions=excluded
    )
    wanted = [r for r in matched if r["page"] == page]
    page_reasons = [
        r
        for r in reasons
        if any(
            line["page_index"] == page and line["line_number"] in r["sourceLines"]
            for line in context.alignment["lines"]
        )
    ]
    if any(r["blocking"] for r in page_reasons):
        return [], page_reasons
    w, h = context.dimensions[str(page)]
    dimensions = np.array([w, h, w, h])
    result = []
    for region in wanted:
        # Preserve the shared page-region normalisation exactly: these tuples
        # are ownership keys, not approximate geometric nearest neighbours.
        box = np.array(region["bbox"], dtype=float) * 1000 / dimensions
        before_width = metrics.length(region["before"])
        if before_width <= 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("package glyph region lacks valid source calibration")
        unit = (box[2] - box[0]) / before_width
        widths = wrap(region["after"], box[2] - box[0], unit, metrics)
        height = box[3] - box[1]
        pitch = height * sum(calibration.pitch_ratio_range) / 2
        result.append(
            {
                "id": f"package:{region['index']}",
                "source": box.tolist(),
                "regionIndices": [region["index"]],
                "sourceLines": region["sourceLines"],
                "height": height + (len(widths) - 1) * pitch,
                "width": max(widths),
                "lineHeight": height,
                "pitch": pitch,
                "virtualLines": len(widths),
                "outputCount": 1,
                "horizontalRegion": region["index"],
                "glyphBefore": region["before"],
                "glyphAfter": region["after"],
                "coordinateMeaning": "source_calibrated_synthetic_glyphs",
            }
        )
    return result, page_reasons
