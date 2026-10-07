"""Seeded page augmentation with checked geometry on the 0..1000 input grid.

One positive uniform scale and translation moves a complete page. Quantization
may change topology, so proposals must preserve axis order/alignment and every
nearest-neighbour tie. Integer translation is the explicitly recorded exact-
geometry mode when the bounded scale search cannot satisfy that contract.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

GRID = 1000
METHOD = "source_anchors_coherent_page_v1"


class PositionPolicy(BaseModel):
    """Limits in normalized page units; no text, labels or field roles are used."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    output_subdirectory: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    seed: int
    scale_min: float = Field(gt=0, le=1)
    scale_max: float = Field(ge=1)
    max_translation: float = Field(ge=0, le=GRID)
    scale_attempts: int = Field(ge=0, le=1024)


def _geometry(points, boxes) -> tuple[np.ndarray, np.ndarray]:
    points, boxes = np.asarray(points, dtype=float), np.asarray(boxes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("geometry requires N x 2 points and M x 4 boxes")
    if not all(np.isfinite(a).all() and np.all((a >= 0) & (a <= GRID)) for a in (points, boxes)):
        raise ValueError("source geometry is nonfinite or outside the page")
    if np.any(points != np.rint(points)) or np.any(boxes[:, :2] > boxes[:, 2:]):
        raise ValueError("source points must be integers and boxes ordered")
    if len(points) and not len(boxes):
        raise ValueError("known source anchors require measured page geometry")
    return points, boxes


def _nearest_equal(before: np.ndarray, after: np.ndarray) -> bool:
    """Compare complete neighbour sets, including duplicates, in bounded chunks.

    At most max(65,536, N) pairs per chunk; no N x N cache or N x N x 2 tensor.
    Exact squared integer distances avoid tolerance
    guesses when deciding a tie. A singleton has no neighbour.
    """
    n = len(before)
    if n < 2:
        return True
    chunk = max(1, 65536 // n)
    before, after = before.astype(np.int64), after.astype(np.int64)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        masks = []
        for points in (before, after):
            d = (points[start:end, None, 0] - points[None, :, 0]) ** 2
            d += (points[start:end, None, 1] - points[None, :, 1]) ** 2
            # Larger than any squared distance on this grid; self is excluded.
            d[np.arange(end - start), np.arange(start, end)] = 2 * GRID**2 + 1
            masks.append(d == d.min(axis=1, keepdims=True))
        if not np.array_equal(*masks):
            return False
    return True


def _violation(points, output, boxes, scale, dx, dy) -> str | None:
    expected = points * scale + (GRID / 2) * (1 - scale) + [dx, dy]
    if np.any(np.abs(expected - output) > 0.50000001):
        return "incoherent_page_transform"
    transformed_boxes = (boxes - GRID / 2) * scale + GRID / 2 + [dx, dy, dx, dy]
    if any(np.any((a < 0) | (a > GRID)) for a in (output, transformed_boxes)):
        return "outside_page"
    for axis in (0, 1):
        order = np.argsort(points[:, axis], kind="stable")
        if not np.array_equal(
            np.sign(np.diff(points[order, axis])), np.sign(np.diff(output[order, axis]))
        ):
            return "changed_axis_order_or_alignment"
    if not _nearest_equal(points, output):
        return "changed_nearest_neighbour_or_tie"
    return None


def validate_transform(points, output, boxes, *, scale: float, dx: float, dy: float) -> None:
    """Validate a replayed or externally supplied page transform, not just its seed."""
    points, boxes = _geometry(points, boxes)
    output = np.asarray(output, dtype=float)
    if (
        output.shape != points.shape
        or not np.isfinite(output).all()
        or np.any(output != np.rint(output))
        or not all(math.isfinite(v) for v in (scale, dx, dy))
        or scale <= 0
    ):
        raise ValueError("invalid transform or output points")
    if failure := _violation(points, output, boxes, scale, dx, dy):
        raise ValueError(failure)


def augment_page(
    points, boxes, *, policy: PositionPolicy, identity: str
) -> tuple[np.ndarray, dict]:
    """Return integer coordinates and a replayable receipt without mutating inputs."""
    points, boxes = _geometry(points, boxes)
    if not len(points):
        return points.astype(int), {
            "mode": "no_known_anchors",
            "scale": 1.0,
            "dx": 0,
            "dy": 0,
            "changed_points": 0,
            "rejections": {},
        }
    rng = random.Random(f"{METHOD}:{policy.seed}:{identity}")
    # Include rounded anchors as well as *all* source OCR regions, not just the
    # subset successfully matched to GLM text. This bounds the entire layout.
    corners = np.concatenate((boxes[:, :2], boxes[:, 2:], points))
    low, high = corners.min(axis=0), corners.max(axis=0)
    rejected = Counter()
    for _ in range(policy.scale_attempts):
        scale = rng.uniform(policy.scale_min, policy.scale_max)
        lower = np.maximum(-policy.max_translation, -(low - GRID / 2) * scale - GRID / 2)
        upper = np.minimum(policy.max_translation, GRID / 2 - (high - GRID / 2) * scale)
        if np.any(lower > upper):
            rejected["no_page_clearance"] += 1
            continue
        dx, dy = (rng.uniform(a, b) for a, b in zip(lower, upper, strict=True))
        output = np.rint((points - GRID / 2) * scale + GRID / 2 + [dx, dy]).astype(int)
        if failure := _violation(points, output, boxes, scale, dx, dy):
            rejected[failure] += 1
            continue
        mode = "scaled_and_translated"
        break
    else:
        lower = np.ceil(np.maximum(-policy.max_translation, -low)).astype(int)
        upper = np.floor(np.minimum(policy.max_translation, GRID - high)).astype(int)
        dx, dy = (rng.randint(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
        scale, mode = 1.0, "integer_translation_only"
        output = np.add(points, [dx, dy]).astype(int)
        validate_transform(points, output, boxes, scale=scale, dx=dx, dy=dy)
    return output, {
        "mode": mode,
        "scale": scale,
        "dx": dx,
        "dy": dy,
        "changed_points": int(np.any(output != points, axis=1).sum()),
        "rejections": dict(rejected),
    }
