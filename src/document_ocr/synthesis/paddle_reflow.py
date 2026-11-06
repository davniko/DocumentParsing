"""Render reviewed Paddle region owners into physical text fragments and boxes.

The cargo/party sampler remains independent of this presentation layer. A block
must own every replaced region explicitly; a label never supplies geometry.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np

from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion, validate
from document_ocr.synthesis.curated_reflow_geometry import (
    layout_page,
    overlap,
    union,
    validate_page,
)
from document_ocr.synthesis.curated_reflow_policy import FontMetrics, ReflowPolicy, load_calibration


@dataclass(frozen=True)
class PaddleBlock:
    owner: str
    page: int
    regions: tuple[int, ...]
    expected_texts: tuple[str, ...]
    text: str
    action: Literal["replace", "omit"] = "replace"


def physical_lines(text: str, width: float, unit: float, metrics: FontMetrics) -> list[str]:
    """Word-wrap physical lines; retain all non-whitespace characters in order.

    A long textual identifier can wrap between glyphs. A standalone numeric
    scalar cannot: separate digit rows are ambiguous counts or measurements.
    """
    if not text.strip() or width <= 0 or unit <= 0:
        raise ValueError("nonempty text and a measured positive column are required")
    lines = []
    for logical in text.splitlines():
        current = ""
        for token in re.findall(r"\s*\S+", logical):
            trial = current + (token if current else token.lstrip())
            if metrics.length(trial) * unit <= width:
                current = trial
                continue
            if re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", logical.strip()):
                raise ValueError(
                    "numeric scalar cannot wrap; review its measured alignment or envelope"
                )
            if current:
                lines.append(current)
                current = ""
            for char in token.lstrip():
                if metrics.length(char) * unit > width:
                    raise ValueError("a glyph cannot fit the measured column")
                if current and metrics.length(current + char) * unit > width:
                    lines.append(current)
                    current = ""
                current += char
        if current:
            lines.append(current)
    if "".join("".join(lines).split()) != "".join(text.split()):
        raise ValueError("physical wrapping lost text")
    return lines


def rectangle_order(regions: list[PaddleRegion]) -> list[PaddleRegion]:
    """PaddleX 3.7.2 SortQuadBoxes order for axis-aligned generated rectangles.

    Its near-row threshold is 10 rendered-page pixels, not 10 normalized units.
    Original native pages are not resorted by this helper.
    """
    result = sorted(regions, key=lambda r: (r.box[1], r.box[0]))
    for i in range(len(result) - 1):
        for j in range(i, -1, -1):
            a, b = result[j], result[j + 1]
            if abs(b.box[1] - a.box[1]) < 10 and b.box[0] < a.box[0]:
                result[j], result[j + 1] = b, a
            else:
                break
    return [PaddleRegion(i, r.text, r.box) for i, r in enumerate(result)]


def foreign_in_new_envelope(envelope, owned, foreign) -> bool:
    """Reject newly enclosed foreign text, not overlaps already in Paddle.

    OCR boxes can overlap a caption by a few pixels. Requiring zero source
    overlap incorrectly rejects legitimate owners; accepting any overlap would
    hide a foreign region inside the gaps of a multi-line owner. Compare exact
    covered rectangles using a small x sweep instead.
    """
    if not overlap(envelope, foreign):
        return False
    left, top = max(envelope[0], foreign[0]), max(envelope[1], foreign[1])
    right, bottom = min(envelope[2], foreign[2]), min(envelope[3], foreign[3])
    # Adjacent OCR rows can already intersect at a top/bottom edge. Joining
    # staggered cells or widening a count may extend that same edge sideways,
    # without enclosing another row or increasing its vertical overlap. Keep
    # that measured boundary contact; interior gaps still need exact coverage.
    if foreign[1] < envelope[1] or foreign[3] > envelope[3]:
        previous_depth = max(
            (min(b[3], foreign[3]) - max(b[1], foreign[1])
             for b in owned if overlap(b, foreign)),
            default=0,
        )
        if previous_depth > 0 and bottom - top <= previous_depth + 1e-9:
            return False
    clipped = [
        (max(left, b[0]), max(top, b[1]), min(right, b[2]), min(bottom, b[3]))
        for b in owned
        if overlap(b, foreign)
    ]
    xs = sorted({left, right, *(x for b in clipped for x in (b[0], b[2]))})
    for x0, x1 in pairwise(xs):
        if x1 - x0 <= 1e-9:
            continue
        intervals = sorted((b[1], b[3]) for b in clipped if b[0] <= x0 and b[2] >= x1)
        covered = top
        for y0, y1 in intervals:
            if y0 > covered + 1e-9:
                return True
            covered = max(covered, y1)
        if covered < bottom - 1e-9:
            return True
    return False


class PaddleReflow:
    def __init__(self, root: Path, policy: ReflowPolicy):
        self.policy = policy
        self.calibration = load_calibration(root, policy)
        self.metrics = FontMetrics(root / policy.font_path, policy.font_sha256)

    def render(
        self,
        pages: tuple[PaddlePage, ...],
        blocks: tuple[PaddleBlock, ...],
        *,
        identity: str,
        watermark_regions: tuple[tuple[int, int], ...] = (),
        right_aligned_numeric_regions: tuple[tuple[int, int], ...] = (),
        row_groups: tuple[tuple[int, tuple[int, ...]], ...] = (),
        reading_order: Literal["rectangles", "source"] = "rectangles",
    ) -> tuple[tuple[PaddlePage, ...], dict]:
        validate(pages)
        if reading_order not in {"rectangles", "source"}:
            raise ValueError("unknown Paddle reading order")
        watermarks = set(watermark_regions)
        if len(watermarks) != len(watermark_regions) or any(
            not 0 <= p < len(pages) or not 0 <= i < len(pages[p].regions) for p, i in watermarks
        ):
            raise ValueError("invalid reviewed watermark regions")
        if len({b.owner for b in blocks}) != len(blocks):
            raise ValueError("duplicate block owner")
        occupied = set()
        for block in blocks:
            if not 0 <= block.page < len(pages) or not block.regions:
                raise ValueError("invalid block page or empty ownership")
            if len(block.regions) != len(block.expected_texts):
                raise ValueError("incomplete block source proof")
            for ix, expected in zip(block.regions, block.expected_texts, strict=True):
                if (block.page, ix) in occupied:
                    raise ValueError("overlapping block ownership")
                occupied.add((block.page, ix))
                if (block.page, ix) in watermarks:
                    raise ValueError("a background watermark cannot own sampled text")
                if not 0 <= ix < len(pages[block.page].regions):
                    raise ValueError("absent block region")
                if pages[block.page].regions[ix].text != expected:
                    raise ValueError("source region differs from block proof")
            if block.action not in {"replace", "omit"}:
                raise ValueError("unknown region action")
            if block.action == "omit" and block.text != "":
                raise ValueError("an explicit omission cannot contain replacement text")
            if block.action == "replace" and not block.text.strip():
                raise ValueError("omission needs an explicit separate reviewed disposition")
        for page_ix, members in row_groups:
            if (
                not 0 <= page_ix < len(pages)
                or len(members) < 2
                or len(set(members)) != len(members)
                or any(not 0 <= i < len(pages[page_ix].regions) for i in members)
            ):
                raise ValueError("invalid reviewed row regions")
            row_boxes = [pages[page_ix].regions[i].box for i in members]
            if max(b[1] for b in row_boxes) >= min(b[3] for b in row_boxes):
                raise ValueError("reviewed row regions do not share a source row")
            if any((page_ix, i) in watermarks for i in members):
                raise ValueError("a background watermark cannot establish a row")
            for block in blocks:
                if (
                    block.page == page_ix
                    and block.action == "replace"
                    and any(i in block.regions[1:] for i in members)
                ):
                    raise ValueError("reviewed row requires the first region of a physical block")
        right_aligned = set(right_aligned_numeric_regions)
        numeric_owners = (
            {
                (b.page, b.regions[0])
                for b in blocks
                if len(b.regions) == 1
                and b.action == "replace"
                and re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", b.expected_texts[0])
                and re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", b.text)
            }
            if right_aligned
            else set()
        )
        if len(right_aligned) != len(right_aligned_numeric_regions) or any(
            not 0 <= p < len(pages)
            or not 0 <= i < len(pages[p].regions)
            or re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", pages[p].regions[i].text) is None
            for p, i in right_aligned
        ):
            raise ValueError("right alignment requires reviewed single-region numeric owners")
        # A declared-package slot can hold a scalar in one shipment and an
        # owned count/type list in another. Alignment applies only to scalar
        # blocks; complete lists retain their independently validated envelope.
        scalar_alignment = right_aligned & numeric_owners
        result, receipts = [], []
        for page in pages:
            selected = [b for b in blocks if b.page == page.index]
            if not selected:
                result.append(page)
                continue
            scale_xy = np.array([1000 / page.width, 1000 / page.height] * 2)
            boxes = np.array([r.box for r in page.regions]) * scale_xy
            if len({tuple(b) for b in boxes}) != len(boxes):
                raise ValueError(
                    "duplicate source rectangles require explicit ownership adjudication"
                )
            replacements, texts = [], {}
            omitted = {i for b in selected if b.action == "omit" for i in b.regions}
            background = {i for p, i in watermarks if p == page.index}
            # Reviewed watermarks are page furniture, not foreground columns or
            # rows. Keep their original text/geometry, but do not let a diagonal
            # COPY stamp constrain wrapping or push the shipment down the page.
            source_lookup = [
                i for i in range(len(page.regions)) if i not in omitted and i not in background
            ]
            if not source_lookup:
                retained = tuple(
                    PaddleRegion(n, page.regions[i].text, page.regions[i].box)
                    for n, i in enumerate(sorted(background))
                )
                result.append(PaddlePage(page.index, page.width, page.height, retained))
                receipts.append({"page": page.index, "omittedRegions": sorted(omitted)})
                continue
            for block in selected:
                if block.action == "omit":
                    continue
                own = [boxes[i] for i in block.regions]
                bbox = union(own)
                foreign = [
                    b for i, b in enumerate(boxes)
                    if i not in block.regions and i not in background
                ]
                if len(own) > 1 and any(
                    foreign_in_new_envelope(bbox, own, b)
                    for i, b in enumerate(boxes)
                    if i not in block.regions and (page.index, i) not in watermarks
                ):
                    raise ValueError(f"foreign text within owner envelope: {block.owner}")
                advances = [
                    (boxes[i][2] - boxes[i][0]) / self.metrics.length(page.regions[i].text)
                    for i in block.regions
                    if self.metrics.length(page.regions[i].text) > 0
                ]
                if not advances:
                    raise ValueError("no source glyph measurement for replacement")
                unit = float(np.median(advances))
                height = float(np.median([b[3] - b[1] for b in own]))
                if (page.index, block.regions[0]) in scalar_alignment:
                    # Right-aligned numeric cells grow left, never wrap a count
                    # into separate digits. The source font and right edge stay
                    # fixed; neighbours bound the available horizontal space.
                    left = max(
                        [height * self.policy.column_guard_height_fraction]
                        + [
                            b[2] + height * self.policy.column_guard_height_fraction
                            for b in foreign
                            if b[2] <= bbox[0] and min(b[3], bbox[3]) > max(b[1], bbox[1])
                        ]
                    )
                    width = self.metrics.length(block.text) * unit
                    if bbox[2] - width < left:
                        raise ValueError("right-aligned number cannot fit its measured row")
                    bbox[0] = bbox[2] - width
                    if any(
                        foreign_in_new_envelope(bbox, own, b)
                        for i, b in enumerate(boxes)
                        if i not in block.regions and (page.index, i) not in watermarks
                    ):
                        raise ValueError("right-aligned number overlaps foreign text")
                right = min(
                    [1000 - height * self.policy.column_guard_height_fraction]
                    + [
                        b[0] - height * self.policy.column_guard_height_fraction
                        for b in foreign
                        if b[0] >= bbox[2] and min(b[3], bbox[3]) > max(b[1], bbox[1])
                    ]
                )
                right = max(float(bbox[2]), right)
                fragments = (
                    [block.text]
                    if (page.index, block.regions[0]) in scalar_alignment
                    else physical_lines(block.text, right - bbox[0], unit, self.metrics)
                )
                pitch = height * random.Random(
                    f"paddle:{self.policy.seed}:{identity}:{page.index}:{block.owner}"
                ).uniform(*self.calibration.pitch_ratio_range)
                widths = [self.metrics.length(s) * unit for s in fragments]
                node_id = f"block:{block.owner}"
                texts[node_id] = (fragments, widths, pitch, height, block)
                replacements.append(
                    {
                        "id": node_id,
                        "source": bbox.tolist(),
                        "lineHeight": height,
                        "height": height + (len(fragments) - 1) * pitch,
                        "width": max(widths),
                        "ownedBoxKeys": [tuple(b) for b in own],
                    }
                )
            nodes, _, _, yscale = layout_page(
                boxes[source_lookup],
                replacements,
                gap_ratio=self.calibration.clearance_height_ratio,
                row_lock_fraction=self.policy.row_lock_height_fraction,
                bottom_guard_fraction=self.policy.bottom_guard_height_fraction,
                row_groups=tuple(
                    tuple(tuple(boxes[i]) for i in members if i not in omitted)
                    for page_ix, members in row_groups
                    if page_ix == page.index and sum(i not in omitted for i in members) >= 2
                ),
            )
            failures = validate_page(nodes, yscale, minimum_gap=0)
            source_median_height = float(np.median(boxes[:, 3] - boxes[:, 1]))
            median_height = source_median_height * yscale
            # The GLM-derived calibration is not a rejection threshold for
            # native Paddle text. Already smaller sources may retain their
            # measured size, but reflow must not shrink them further.
            minimum_height = min(source_median_height, self.calibration.minimum_page_median_height)
            if failures or median_height < minimum_height:
                raise ValueError(f"unsafe Paddle page geometry: {failures}; height={median_height}")
            emitted, source_order, blocks_receipt = [], [], []

            def pixel_box(b, sy=yscale, factor=scale_xy):
                return tuple(float(v) for v in np.asarray(b) * [1, sy, 1, sy] / factor)

            for node in nodes:
                x1, y1, _, _ = node["box"]
                if node["id"].startswith("source:"):
                    ix = source_lookup[int(node["id"].split(":")[1])]
                    emitted.append(PaddleRegion(0, page.regions[ix].text, pixel_box(node["box"])))
                    source_order.append((ix, 0))
                    continue
                fragments, widths, pitch, height, block = texts[node["id"]]
                physical = []
                for i, (text, width) in enumerate(zip(fragments, widths, strict=True)):
                    box = pixel_box([x1, y1 + i * pitch, x1 + width, y1 + i * pitch + height])
                    emitted.append(PaddleRegion(0, text, box))
                    source_order.append((min(block.regions), i))
                    physical.append({"text": text, "box": box})
                blocks_receipt.append(
                    {
                        "owner": block.owner,
                        "sourceRegions": block.regions,
                        "text": block.text,
                        "physicalFragments": physical,
                    }
                )
            for ix in sorted(background):
                region = page.regions[ix]
                emitted.append(PaddleRegion(0, region.text, region.box))
                source_order.append((ix, 0))
            if reading_order == "source":
                # Anonymization replaces an explicitly owned span in the native
                # sequence. Rectangle sorting cannot recover the original quad
                # ordering and must not reorder unrelated source lines. A
                # rewrapped block occupies its first owned region's position.
                ordered = [
                    r
                    for _, r in sorted(zip(source_order, emitted, strict=True), key=lambda v: v[0])
                ]
                ordered = [PaddleRegion(i, r.text, r.box) for i, r in enumerate(ordered)]
            else:
                ordered = rectangle_order(emitted)
            result.append(PaddlePage(page.index, page.width, page.height, tuple(ordered)))
            receipts.append(
                {
                    "page": page.index,
                    "verticalScale": yscale,
                    "minimumHeightCheck": median_height,
                    "blocks": blocks_receipt,
                    "omittedRegions": sorted(omitted),
                    "geometryFailures": failures,
                }
            )
        output = tuple(result)
        validate(output)
        receipt = {
            "method": "source_measured_physical_fragments_v1",
            "pages": receipts,
            "reviewedWatermarkRegions": sorted(watermarks),
        }
        if row_groups:
            receipt["reviewedRowGroups"] = row_groups
        if right_aligned:
            receipt["reviewedRightAlignedNumericRegions"] = sorted(right_aligned)
        if reading_order == "source":
            receipt["readingOrder"] = "native_source_anchors_with_in_place_physical_fragments"
        return output, receipt
