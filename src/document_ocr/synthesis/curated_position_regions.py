"""Place changed line spans using measured source space, or explicitly abstain.

Text is never shortened to fit. The output is a layout signal, not font metrics.
Half of each measured vertical gap belongs to either adjoining source region;
independently expanded regions therefore cannot claim the same empty space.
"""

from __future__ import annotations

import math

import numpy as np


def place_region(
    records: list[dict], count: int, *, boxes=None, dimensions=None
) -> tuple[list[tuple[int, int] | None], dict]:
    """One page-local owned region; no labels or generated word meanings enter layout."""
    original = [r["xy"] for r in records]
    receipt = {"sourceLines": len(records), "outputLines": count}

    def unknown(reason):
        return [None] * count, {**receipt, "method": "unplaced_expansion", "reason": reason}

    if not count:
        return [], {**receipt, "method": "removed_region"}
    if len(original) == count:
        return [tuple(p) if p is not None else None for p in original], {
            **receipt,
            "method": "source_line_anchors",
        }
    if any(p is None for p in original):
        return unknown("missing_source_anchor")
    points = np.asarray(original, dtype=float)
    if count == 1:
        # Centre of the original region; contraction needs no additional space.
        return [tuple(map(int, np.rint((points.min(0) + points.max(0)) / 2)))], {
            **receipt,
            "method": "contracted_region_centre",
        }
    if boxes is None or dimensions is None:
        return unknown("missing_measured_geometry")
    if len(points) > 1 and np.any(np.diff(points[:, 1]) <= 0):
        return unknown("source_not_a_vertical_text_sequence")
    divisors = np.asarray([*dimensions, *dimensions], dtype=float)
    if any(not r.get("regions") or r.get("bbox") is None for r in records):
        return unknown("missing_measured_region")
    # Use the same operation order as load_page_geometry: exact region identity
    # must not be lost to a different floating-point normalization expression.
    own = (
        np.asarray([r["bbox"] for line in records for r in line["regions"]], dtype=float)
        * 1000
        / divisors
    )
    line_boxes = np.asarray([r["bbox"] for r in records], dtype=float) * 1000 / divisors
    height = float(np.median(own[:, 3] - own[:, 1]))
    if height <= 0:
        return unknown("nonpositive_measured_height")
    left, top = line_boxes[:, :2].min(0)
    right, bottom = line_boxes[:, 2:].max(0)
    own_set = {tuple(b) for b in own}
    other = np.asarray([b for b in boxes if tuple(b) not in own_set]).reshape(-1, 4)
    adjacent = other[(other[:, 0] < right) & (other[:, 2] > left)]
    if np.any((adjacent[:, 1] < bottom) & (adjacent[:, 3] > top)):
        return unknown("unowned_text_intersects_source_region")
    above, below = adjacent[adjacent[:, 3] <= top], adjacent[adjacent[:, 1] >= bottom]
    low = (float(above[:, 3].max()) + top) / 2 if len(above) else 0.0
    high = (float(below[:, 1].min()) + bottom) / 2 if len(below) else 1000.0
    receipt.update(envelope=[float(left), float(low), float(right), float(high)], lineHeight=height)
    # Half-height clearances keep line rectangles inside their assigned envelope.
    ymin, ymax = math.ceil(low + height / 2), math.floor(high - height / 2)
    minimum_pitch = math.ceil(height)
    if ymax - ymin < (count - 1) * minimum_pitch:
        return unknown("insufficient_local_space")
    positions = np.linspace(0, len(points) - 1, count)
    xs = np.interp(positions, np.arange(len(points)), points[:, 0])
    ys = np.interp(positions, np.arange(len(points)), points[:, 1])
    mode = "source_span_interpolation"
    if np.any(np.diff(np.rint(ys)) < minimum_pitch):
        observed = float(np.median(np.diff(points[:, 1]))) if len(points) > 1 else height
        pitch = min(max(minimum_pitch, math.ceil(observed)), (ymax - ymin) // (count - 1))
        start = min(max(round(points[0, 1]), ymin), ymax - (count - 1) * pitch)
        ys = start + np.arange(count) * pitch
        mode = "local_space_reflow"
    output = np.rint(np.column_stack((xs, ys))).astype(int)
    if (
        np.any(output < 0)
        or np.any(output > 1000)
        or np.any(output[:, 1] < ymin)
        or np.any(output[:, 1] > ymax)
        or np.any(np.diff(output[:, 1]) < minimum_pitch)
    ):
        return unknown("placement_does_not_fit_measured_envelope")
    return [tuple(map(int, p)) for p in output], {**receipt, "method": mode}
