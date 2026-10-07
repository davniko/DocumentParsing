"""Source-conditioned stochastic reflow through verified synthetic edit ownership.

Synthetic text, labels and source pages are never rewritten. Proposal rejection
is explicit, preserving source-transfer anchors rather than inventing geometry.
The public engine is shared by campaign positioning and offline validation.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from document_ocr.spatial_inputs.alignment import enrich, verify_preservation
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_reflow_geometry import (
    layout_page,
    overlap,
    union,
    validate_page,
)
from document_ocr.synthesis.curated_reflow_policy import (
    FontMetrics,
    ReflowCalibration,
    ReflowPolicy,
    load_calibration,
)

METHOD = "source_conditioned_elastic_reflow_v1"


class GlyphWidthError(ValueError):
    """A replacement cannot fit even one glyph in its measured column."""


class DSU:
    def __init__(self, ids):
        self.p = {x: x for x in ids}

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def join(self, ids):
        ids = list(ids)
        for x in ids[1:]:
            self.p[self.find(x)] = self.find(ids[0])


@dataclass
class ReflowContext:
    sample_id: str
    rendered: str
    receipt: dict
    alignment: dict
    boxes: dict
    dimensions: dict
    groups: list


def groups_for(source: str, proof: dict, receipt: dict, alignment: dict) -> list[dict]:
    """Connected byte-edit owners, including shared physical source lines."""
    records = {r["line_number"]: r for r in alignment["lines"]}
    dsu = DSU(records)
    starts, pos = [], 0
    for s in source.splitlines(keepends=True):
        starts.append(pos)
        pos += len(s.encode())
    starts.append(pos)
    for e in proof["edits"]:
        lo, hi = bisect_right(starts, e["byteStart"]), bisect_right(starts, e["byteEnd"] - 1)
        owned = [n for n in range(lo, hi + 1) if n in records]
        if not owned or len({records[n]["page_index"] for n in owned}) != 1:
            raise ValueError("reflow edit crosses a page or has no source owner")
        dsu.join(owned)
    region_owners = defaultdict(list)
    for n, r in records.items():
        for ix in r["region_indices"]:
            region_owners[r["page_index"], ix].append(n)
    for ids in region_owners.values():
        dsu.join(ids)
    for line in receipt["lines"]:
        dsu.join(line["source_lines"])
    result = defaultdict(lambda: {"source": [], "output": []})
    for n, r in records.items():
        result[dsu.find(n)]["source"].append(r)
    for line in receipt["lines"]:
        result[dsu.find(line["source_lines"][0])]["output"].append(line)
    groups = list(result.values())
    if any(len({r["page_index"] for r in g["source"]}) != 1 for g in groups):
        raise ValueError("connected reflow ownership crosses page boundaries")
    return groups


def page_regions(alignment, page, dimensions):
    norm = np.array([*dimensions, *dimensions])
    return {
        r["index"]: {**r, "box": np.array(r["bbox"], dtype=float) * 1000 / norm}
        for line in alignment["lines"]
        if line["page_index"] == page
        for r in line["regions"]
    }


def wrapped_widths(text: str, width: float, unit: float, metrics: FontMetrics) -> list[float]:
    """Virtual font-based wrapping; original OCR string is never rewritten."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = (current + " " + word).strip()
        if metrics.length(trial) * unit <= width:
            current = trial
            continue
        if current:
            lines.append(metrics.length(current) * unit)
            current = ""
        # Long URLs/codes may be physically wrapped without losing characters.
        for char in word:
            if current and metrics.length(current + char) * unit > width:
                lines.append(metrics.length(current) * unit)
                current = ""
            current += char
    if current:
        lines.append(metrics.length(current) * unit)
    if not lines or max(lines) > width + 1e-7:
        raise GlyphWidthError("a glyph cannot fit the available column")
    return lines


def propose_groups(c, page, calibration, policy, metrics):
    """Build source-anchored replacement blocks; never infer an absent owner."""
    regions = page_regions(c.alignment, page, c.dimensions[str(page)])
    boxes = c.boxes[page]
    replacements, reasons = [], []
    for group in c.groups:
        src, output = group["source"], group["output"]
        if src[0]["page_index"] != page or len(src) == len(output) or not output:
            continue
        key = ":".join(str(r["line_number"]) for r in src)
        reason = {
            "sourceLines": [r["line_number"] for r in src],
            "outputLines": [r["line_number"] for r in output],
        }
        if any(r["xy"] is None for r in src):
            reasons.append({**reason, "reason": "incomplete_source_anchors"})
            continue
        own = {ix for r in src for ix in r["region_indices"]}
        ob = [regions[i]["box"] for i in own]
        bbox = union(ob)
        own_set = {tuple(b) for b in ob}
        foreign = [b for b in boxes if tuple(b) not in own_set]
        if any(overlap(bbox, b) for b in foreign):
            reasons.append({**reason, "reason": "foreign_text_inside_owner_envelope"})
            continue
        if any(src[i + 1]["xy"][1] <= src[i]["xy"][1] for i in range(len(src) - 1)):
            reasons.append({**reason, "reason": "owner_not_vertical_sequence"})
            continue
        height = float(np.median([b[3] - b[1] for b in ob]))
        rng = random.Random(f"reflow:{policy.seed}:{c.sample_id}:{page}:{key}")
        pitch = height * rng.uniform(*calibration.pitch_ratio_range)
        # A half-line guard models a clear boundary before the next column.
        right = min(
            [
                b[0] - height * policy.column_guard_height_fraction
                for b in foreign
                if b[0] >= bbox[2] and min(b[3], bbox[3]) > max(b[1], bbox[1])
            ]
            + [1000 - height * policy.column_guard_height_fraction]
        )
        right = max(float(bbox[2]), right)
        width = right - bbox[0]
        units = [
            (regions[ix]["box"][2] - regions[ix]["box"][0]) / metrics.length(regions[ix]["text"])
            for ix in own
            if metrics.length(regions[ix]["text"]) > 0
        ]
        if not units or height <= 0 or not np.isfinite(units).all():
            raise ValueError("invalid source font/height measurements")
        unit = float(np.median(units))
        cursor, line_boxes, wraps = 0.0, {}, 0
        for line in output:
            try:
                widths = wrapped_widths(line["text"], width, unit, metrics)
            except GlyphWidthError:
                reasons.append({**reason, "reason": "glyph_exceeds_column"})
                break
            wraps += len(widths)
            line_boxes[line["line_number"]] = [
                float(bbox[0]),
                cursor,
                float(bbox[0] + max(widths)),
                cursor + (len(widths) - 1) * pitch + height,
            ]
            cursor += len(widths) * pitch
        if len(line_boxes) != len(output):
            continue
        replacements.append(
            {
                "id": key,
                "source": bbox.tolist(),
                "regionIndices": sorted(own),
                "lineBoxes": line_boxes,
                "height": max(height, cursor - pitch + height),
                "width": max(b[2] for b in line_boxes.values()) - bbox[0],
                "lineHeight": height,
                "pitch": pitch,
                "virtualLines": wraps,
                "outputCount": len(output),
                "sourceLines": reason["sourceLines"],
            }
        )
    return replacements, reasons


def run_layout(
    c: ReflowContext, calibration: ReflowCalibration, policy: ReflowPolicy, metrics: FontMetrics
):
    regions_by_page = {p: page_regions(c.alignment, p, c.dimensions[str(p)]) for p in c.boxes}
    coords = {r["line_number"]: r["xy"] for r in c.receipt["lines"]}
    results = []
    for page, boxes in sorted(c.boxes.items()):
        reps, reasons = propose_groups(c, page, calibration, policy, metrics)
        for r in reps:
            r["ownedBoxKeys"] = {
                tuple(regions_by_page[page][ix]["box"]) for ix in r["regionIndices"]
            }
        if not reps:
            results.append({"page": page, "status": "no_eligible_reflow", "heldGroups": reasons})
            continue
        median_height = float(np.median(boxes[:, 3] - boxes[:, 1]))
        gap_ratio = calibration.clearance_height_ratio
        nodes, bands, _shifted, scale = layout_page(
            boxes,
            reps,
            gap_ratio=gap_ratio,
            row_lock_fraction=policy.row_lock_height_fraction,
            bottom_guard_fraction=policy.bottom_guard_height_fraction,
        )
        minimum_gap = median_height * gap_ratio
        density_floor = calibration.minimum_page_median_height
        failure = validate_page(nodes, scale, minimum_gap=minimum_gap)
        if median_height * scale < min(median_height, density_floor) - 1e-7:
            failure.append(("density_floor", median_height * scale, density_floor))
        record = {
            "page": page,
            "status": "rejected" if failure else "accepted",
            "scaleY": scale,
            "medianGlyphHeight": median_height * scale,
            "minimumGap": minimum_gap,
            "failures": failure,
            "heldGroups": reasons,
            "blocks": len(reps),
            "nodes": [{k: v for k, v in n.items() if k != "ownedBoxKeys"} for n in nodes],
            "rowBands": len(bands),
        }
        if failure:
            results.append(record)
            continue
        proposal = dict(coords)
        # Move existing points by exact source-region identity, not their rounded
        # y value: interpolation around a row boundary can move a next-row caption
        # into an expanded block even when the underlying rectangles are sound.
        moved = {tuple(n["source"]): n["box"] for n in nodes if n["id"].startswith("source:")}
        source_records = {r["line_number"]: r for r in c.alignment["lines"]}
        generated_numbers = {number for n in nodes for number in n.get("lineBoxes", {})}
        for line in c.receipt["lines"]:
            if (
                line["page_index"] == page
                and line["line_number"] not in generated_numbers
                and coords[line["line_number"]] is not None
            ):
                x, y = coords[line["line_number"]]
                own = [
                    regions_by_page[page][ix]["box"]
                    for number in line["source_lines"]
                    for ix in source_records[number]["region_indices"]
                ]
                before = union(own)
                after = union([moved[tuple(b)] for b in own])
                shift = ((after[1] + after[3]) - (before[1] + before[3])) / 2
                proposal[line["line_number"]] = [round(x), round((y + shift) * scale)]
        for n in nodes:
            if "lineBoxes" not in n:
                continue
            for number, b in n["lineBoxes"].items():
                proposal[number] = [
                    round((b[0] + b[2]) / 2),
                    round((n["box"][1] + (b[1] + b[3]) / 2) * scale),
                ]
        # Emitted anchors must also clear the new blocks. Rectangle validation
        # alone would not catch a stale/incorrectly shifted legacy line centre.
        page_lines = [line for line in c.receipt["lines"] if line["page_index"] == page]
        for n in nodes:
            if "lineBoxes" not in n:
                continue
            left, top, right, bottom = np.array(n["box"]) * [1, scale, 1, scale]
            for line in page_lines:
                number = line["line_number"]
                xy = proposal[number]
                if (
                    xy is not None
                    and number not in n["lineBoxes"]
                    and left + 0.5 < xy[0] < right - 0.5
                    and top + 0.5 < xy[1] < bottom - 0.5
                ):
                    failure.append(("foreign_anchor_inside_generated_block", number, n["id"]))
        # Logical GLM lines can aggregate several Paddle regions. Clear physical
        # rectangles alone do not guarantee distinct integer aggregate centroids.
        seen_points = defaultdict(list)
        row_lookup = {line["line_number"]: line for line in page_lines}
        for line in page_lines:
            number = line["line_number"]
            if proposal[number] is not None:
                seen_points[tuple(proposal[number])].append(number)
        for point, numbers in seen_points.items():
            if len(numbers) < 2:
                continue
            before = [row_lookup[n]["xy"] for n in numbers]
            if any(x is None for x in before) or len({tuple(x) for x in before}) > 1:
                failure.append(("new_logical_centroid_collision", numbers, list(point)))
        legacy_columns = defaultdict(list)
        for line in page_lines:
            number = line["line_number"]
            xy = coords[number]
            if number not in generated_numbers and xy is not None:
                legacy_columns[xy[0]].append((xy[1], proposal[number][1], number))
        for column in legacy_columns.values():
            ordered = sorted(column)
            for first, second in pairwise(ordered):
                if first[0] < second[0] and first[1] >= second[1]:
                    failure.append(("legacy_column_order", first[2], second[2]))
        if failure:
            record["status"] = "rejected"
        else:
            coords = proposal
        results.append(record)
    if any(r["xy"] is not None and coords[r["line_number"]] is None for r in c.receipt["lines"]):
        raise ValueError("reflow lost a previously known coordinate")
    if any(
        p is not None and any(type(v) is not int or not 0 <= v <= 1000 for v in p)
        for p in coords.values()
    ):
        raise ValueError("reflow emitted invalid normalized coordinates")
    text = enrich(c.rendered, coords)
    verify_preservation(c.rendered, text, set(coords))
    return coords, results, text


class ReflowEngine:
    """Load shared calibration/font once; reflow one verified document at a time."""

    def __init__(self, root: Path, policy: ReflowPolicy):
        self.policy = policy
        self.calibration = load_calibration(root, policy)
        self.metrics = FontMetrics(root / policy.font_path, policy.font_sha256)

    def apply(
        self,
        *,
        source: str,
        rendered: str,
        proof: dict,
        alignment: dict,
        receipt: dict,
        boxes: dict,
        dimensions: dict,
        sample_id: str,
    ) -> tuple[str, dict, dict]:
        """Return enriched text, replayable receipt and geometry for augmentation."""
        if receipt["sourceSha256"] != digest(source.encode()) or receipt[
            "renderedSha256"
        ] != digest(rendered.encode()):
            raise ValueError("reflow source/rendered text differs from transfer receipt")
        if receipt["alignmentSha256"] != digest(alignment) or receipt["editProofSha256"] != digest(
            proof
        ):
            raise ValueError("reflow provenance differs from verified transfer")
        if "anchor_xy" in next(iter(receipt["lines"]), {}):
            raise ValueError("reflow must precede whole-page augmentation")
        context = ReflowContext(
            sample_id,
            rendered,
            receipt,
            alignment,
            boxes,
            dimensions,
            groups_for(source, proof, receipt, alignment),
        )
        coordinates, pages, text = run_layout(context, self.calibration, self.policy, self.metrics)
        by_page = {p["page"]: p for p in pages}
        generated = {
            number
            for p in pages
            if p["status"] == "accepted"
            for n in p["nodes"]
            for number in n.get("lineBoxes", {})
        }
        lines = []
        for line in receipt["lines"]:
            number = line["line_number"]
            method = line["method"]
            if number in generated:
                method = "reflowed_generated_line"
            elif by_page[line["page_index"]]["status"] == "accepted" and line["xy"] is not None:
                method = "reflowed_source_anchor"
            lines.append(dict(line, xy=coordinates[number], transfer_xy=line["xy"], method=method))
        effective_geometry = {}
        for p in pages:
            page = p["page"]
            effective_geometry[page] = (
                np.asarray([n["box"] for n in p["nodes"]], dtype=float).reshape(-1, 4)
                * [1, p["scaleY"], 1, p["scaleY"]]
                if p["status"] == "accepted"
                else boxes[page]
            )
        # JSON object keys are strings. Normalize before hashing, not only during
        # serialization: otherwise key ordering differs after reading the receipt.
        for page in pages:
            for node in page.get("nodes", []):
                if "lineBoxes" in node:
                    node["lineBoxes"] = {str(k): v for k, v in node["lineBoxes"].items()}
        return (
            text,
            {
                **receipt,
                "method": METHOD,
                "transferPositionedSha256": receipt["positionedSha256"],
                "positionedSha256": digest(text.encode()),
                "reflowPolicy": self.policy.model_dump(mode="json"),
                "reflowPages": pages,
                "baselineCounts": receipt["counts"],
                "counts": dict(Counter(line["method"] for line in lines)),
                "lines": lines,
            },
            effective_geometry,
        )
