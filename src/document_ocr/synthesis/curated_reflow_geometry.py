"""Joint source-row/column reflow and independent rectangle validation.

Coordinates use a normalized 0..1000 page. Existing source overlaps are retained;
new overlap, broken row ties and lost left/right relationships reject a proposal.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def union(boxes):
    a = np.asarray(boxes)
    return np.array([*a[:, :2].min(0), *a[:, 2:].max(0)], dtype=float)


def overlap(a, b):
    # Arithmetic touchpoints can differ by ~2e-13 after longest-path sums.
    # 1e-9 of the normalized grid is numerical tolerance, not spatial clearance.
    return min(a[2], b[2]) - max(a[0], b[0]) > 1e-9 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-9


def x_overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0])


def layout_page(
    boxes, replacements, *, gap_ratio: float, row_lock_fraction: float, bottom_guard_fraction: float
):
    """Longest-path solution of row/column clearance constraints, not jitter.

    Shared source rows move together. Other source rows keep their vertical order;
    nodes in overlapping columns keep a source-fitted minimum clearance. No input
    text order or target field roles establish spatial order.
    """
    own_boxes = set()
    for r in replacements:
        own_boxes.update(r["ownedBoxKeys"])
    nodes = []
    for i, b in enumerate(boxes):
        if tuple(b) not in own_boxes:
            nodes.append(
                {
                    "id": f"source:{i}",
                    "source": b.tolist(),
                    "width": float(b[2] - b[0]),
                    "height": float(b[3] - b[1]),
                    "anchor": float((b[1] + b[3]) / 2),
                }
            )
    for r in replacements:
        nodes.append({**r, "anchor": r["source"][1] + r["lineHeight"] / 2})
    median_height = float(np.median(boxes[:, 3] - boxes[:, 1]))
    # Row-lock tolerance is exposed in the report and tested through ablation.
    tolerance = median_height * row_lock_fraction
    ordered = sorted(range(len(nodes)), key=lambda i: nodes[i]["anchor"])
    bands = []
    for ix in ordered:
        if not bands or nodes[ix]["anchor"] - bands[-1]["anchor"] > tolerance:
            bands.append({"anchor": nodes[ix]["anchor"], "nodes": []})
        bands[-1]["nodes"].append(ix)
        nodes[ix]["band"] = len(bands) - 1
    edges = defaultdict(list)
    for b in nodes:
        bb = b["source"]
        for a in nodes:
            if a["band"] >= b["band"]:
                continue
            ab = a["source"]
            xa = [ab[0], ab[1], ab[0] + a["width"], ab[3]]
            xb = [bb[0], bb[1], bb[0] + b["width"], bb[3]]
            if x_overlap(xa, xb) and ab[3] <= bb[1]:
                gap = bb[1] - ab[3]
                required_gap = min(gap, median_height * gap_ratio)
                extra = a["height"] - (ab[3] - ab[1]) + required_gap - gap
                edges[b["band"]].append(
                    (a["band"], bands[b["band"]]["anchor"] - bands[a["band"]]["anchor"] + extra)
                )
    shifted = []
    for g, band in enumerate(bands):
        y = band["anchor"]
        for predecessor, clearance in edges[g]:
            y = max(y, shifted[predecessor] + clearance)
        shifted.append(y)
    for node in nodes:
        shift = shifted[node["band"]] - bands[node["band"]]["anchor"]
        b = node["source"]
        node["box"] = [b[0], b[1] + shift, b[0] + node["width"], b[1] + shift + node["height"]]
    # Elastic placement can consume existing whitespace, including excess bottom
    # margin. Retain at least one source-typical glyph height (or the original
    # smaller margin) rather than extending the page merely to preserve blank air.
    bottom_margin = min(
        max(0, 1000 - float(boxes[:, 3].max())), median_height * bottom_guard_fraction
    )
    virtual_height = max(1000, max(n["box"][3] for n in nodes) + bottom_margin)
    scale = 1000 / virtual_height
    return nodes, bands, shifted, scale


def validate_page(nodes, scale, *, minimum_gap: float):
    """Independent checks on emitted rectangles, not solver edges."""
    failures = []
    if not math.isfinite(scale) or not 0 < scale <= 1:
        return [("invalid_scale", scale)]
    for i, a in enumerate(nodes):
        b = np.array(a["box"]) * [1, scale, 1, scale]
        if not np.isfinite(b).all() or (b < 0).any() or (b > 1000 + 1e-7).any():
            failures.append(("bounds", a["id"]))
        if b[2] <= b[0] or b[3] <= b[1]:
            failures.append(("invalid_rectangle", a["id"]))
        if abs(a["box"][0] - a["source"][0]) > 1e-7:
            failures.append(("changed_column_start", a["id"]))
        for other in nodes[i + 1 :]:
            if overlap(a["box"], other["box"]) and not overlap(a["source"], other["source"]):
                failures.append(("introduced_overlap", a["id"], other["id"]))
            if a["band"] == other["band"]:
                old = a["source"][1] - other["source"][1]
                new = a["box"][1] - other["box"][1]
                if abs(old - new) > 1e-6:
                    failures.append(("row_alignment", a["id"], other["id"]))
            for upper, lower in [(a, other), (other, a)]:
                if (
                    x_overlap(upper["box"], lower["box"])
                    and upper["source"][3] <= lower["source"][1]
                ):
                    old_gap = lower["source"][1] - upper["source"][3]
                    required_gap = min(old_gap, minimum_gap)
                    if lower["box"][1] - upper["box"][3] < required_gap - 1e-7:
                        failures.append(("column_order_clearance", upper["id"], lower["id"]))
            if min(a["source"][3], other["source"][3]) > max(a["source"][1], other["source"][1]):
                if a["source"][2] <= other["source"][0] and a["box"][2] > other["box"][0] + 1e-7:
                    failures.append(("lost_left_right_relation", a["id"], other["id"]))
                if other["source"][2] <= a["source"][0] and other["box"][2] > a["box"][0] + 1e-7:
                    failures.append(("lost_left_right_relation", other["id"], a["id"]))
    return failures
