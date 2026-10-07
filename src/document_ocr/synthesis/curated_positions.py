"""Transfer measured source layout through exact synthetic text edits.

Coordinates describe inherited layout anchors, not measured synthetic glyphs.
No text matching, target labels, PDF reflow or invented missing boxes are used.
"""

from __future__ import annotations

import json
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import numpy as np
import pyarrow.parquet as pq

from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.alignment import (
    PAGE_MARKER,
    centroid,
    enrich,
    parse_lines,
    verify_preservation,
)
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_layout import GRID, METHOD, PositionPolicy, augment_page
from document_ocr.synthesis.curated_position_regions import place_region
from document_ocr.synthesis.curated_publication import _write_new_or_identical


@dataclass(frozen=True)
class Anchor:
    lines: tuple[int, ...]
    xy: tuple[int, int] | None
    edit: str | None = None
    relocated: bool = False


def _line_coordinates(contributions: list[Anchor]) -> tuple[int, int] | None:
    """Move inline captions/counts with their reflowed source line.

    A prefix and an expanded value share one measured line, not independent
    locations. Unchanged fragments from other source lines remain independent.
    Unknown reflow coordinates remain unknown, even when a prefix is measured.
    """
    relocated_lines = {
        number for anchor in contributions if anchor.relocated for number in anchor.lines
    }
    relevant = [
        anchor
        for anchor in contributions
        if anchor.relocated or not set(anchor.lines).issubset(relocated_lines)
    ]
    points = [anchor.xy for anchor in relevant]
    if any(point is None for point in points):
        return None
    return tuple(
        round((min(point[axis] for point in points) + max(point[axis] for point in points)) / 2)
        for axis in (0, 1)
    )


def transfer_positions(
    source: str,
    rendered: str,
    proof: dict,
    alignment: dict,
    *,
    page_boxes: dict | None = None,
    page_dimensions: dict | None = None,
) -> tuple[str, dict]:
    """Map byte-owned edits onto page-local anchors, including changed line spans.

    Unchanged segments retain their original line coordinates. Replacement lines
    follow the source span or measured adjacent free space. An expansion that
    cannot fit is explicitly unpositioned; it never stacks lines at one point.
    """
    raw = source.encode()
    if proof["sourceSha256"] != digest(raw) or alignment["original_input_sha256"] != digest(raw):
        raise ValueError("source/proof/alignment hash mismatch")
    if proof["renderedSha256"] != digest(rendered.encode()):
        raise ValueError("rendered proof hash mismatch")
    physical = source.splitlines(keepends=True)
    page_count = sum(bool(PAGE_MARKER.fullmatch(s.strip())) for s in physical)
    source_lines = {line.number: line for line in parse_lines(source, page_count)}
    anchors = {line["line_number"]: line for line in alignment["lines"]}
    if len(anchors) != len(alignment["lines"]) or set(anchors) != set(source_lines):
        raise ValueError("alignment does not cover each source content line exactly once")
    for number, line in source_lines.items():
        anchor = anchors[number]
        if anchor["original_text"] != line.text or anchor["page_index"] != line.page:
            raise ValueError("alignment text/page differs from source")
        xy = anchor["xy"]
        if xy is not None and (
            len(xy) != 2 or any(type(v) is not int or not 0 <= v <= 1000 for v in xy)
        ):
            raise ValueError("source coordinate outside normalized 0..1000 grid")
    measured = enrich(source, {n: anchors[n]["xy"] for n in source_lines})
    if digest(measured.encode()) != alignment["positioned_input_sha256"]:
        raise ValueError("source coordinates differ from the measured positioned input")
    starts, offset = [], 0
    for line in physical:
        starts.append(offset)
        offset += len(line.encode())
    starts.append(offset)
    if (page_boxes is None) != (page_dimensions is None):
        raise ValueError("page boxes and dimensions must be supplied together")
    pieces, placements = [], []
    reflow_owners = defaultdict(set)

    def unchanged(start: int, end: int) -> None:
        while start < end:
            index = bisect_right(starts, start) - 1
            stop = min(end, starts[index + 1])
            record = anchors.get(index + 1)
            xy = tuple(record["xy"]) if record and record["xy"] is not None else None
            pieces.append((raw[start:stop], Anchor((index + 1,), xy)))
            start = stop

    cursor = 0
    for edit in proof["edits"]:
        start, end = edit["byteStart"], edit["byteEnd"]
        if not cursor <= start < end <= len(raw) or raw[start:end].decode() != edit["before"]:
            raise ValueError("overlapping, unordered or stale source edit")
        unchanged(cursor, start)
        first, last = bisect_right(starts, start), bisect_right(starts, end - 1)
        physical_lines = list(range(first, last + 1))
        # A single owned product/address region can contain paragraph separators.
        # Blank lines have no measured text anchor; they neither supply a point
        # nor break ownership. Page markers remain hard structural boundaries.
        owned = [n for n in physical_lines if n in source_lines]
        if (
            not owned
            or any(PAGE_MARKER.fullmatch(physical[n - 1].strip()) for n in physical_lines)
            or len({source_lines[n].page for n in owned}) != 1
        ):
            raise ValueError("edit crosses a structural line or page boundary")
        replacement = edit["after"].splitlines(keepends=True)
        page = source_lines[owned[0]].page
        placed, placement = place_region(
            [anchors[n] for n in owned],
            len(replacement),
            boxes=page_boxes[page] if page_boxes is not None else None,
            dimensions=page_dimensions[str(page)] if page_dimensions is not None else None,
        )
        placements.append(
            {"key": edit["key"], "occurrence": edit["occurrence"], "page_index": page, **placement}
        )
        edit_id = f"{edit['key']}:{edit['occurrence']}"
        relocated = len(replacement) != len(owned)
        if relocated and replacement:
            for number in owned:
                reflow_owners[number].add(edit_id)
        for i, line in enumerate(replacement):
            if PAGE_MARKER.fullmatch(line.strip()):
                raise ValueError("replacement introduces a page marker")
            position = (len(owned) - 1) * (
                i / (len(replacement) - 1) if len(replacement) > 1 else 0.5
            )
            lo = int(position)
            hi = min(lo + 1, len(owned) - 1)
            fraction = position - lo
            selected = (owned[lo],) if not fraction else (owned[lo], owned[hi])
            xy = placed[i]
            pieces.append((line.encode(), Anchor(selected, xy, edit_id, relocated)))
        cursor = end
    unchanged(cursor, len(raw))
    # Separately expanded owners on one measured source line cannot each claim
    # its free space. Preserve their text, but do not invent a joint placement.
    conflicts = {key for owners in reflow_owners.values() if len(owners) > 1 for key in owners}
    if conflicts:
        pieces = [
            (piece, replace(anchor, xy=None) if anchor.edit in conflicts else anchor)
            for piece, anchor in pieces
        ]
        for placement in placements:
            if f"{placement['key']}:{placement['occurrence']}" in conflicts:
                placement.update(method="unplaced_expansion", reason="overlapping_reflow_owners")
    if b"".join(piece for piece, _ in pieces) != rendered.encode():
        raise ValueError("position transfer differs from exact rendered edit replay")

    # Split the composed stream, not the source's line numbers. A longer address
    # cannot shift the coordinates of a following telephone, heading or table row.
    mapped, number = {}, 1
    for piece, anchor in pieces:
        for part in piece.splitlines(keepends=True):
            if part.strip():
                mapped.setdefault(number, []).append(anchor)
            if part.endswith(b"\n"):
                number += 1
    lines, coordinates = [], {}
    for line in parse_lines(rendered, page_count):
        contributions = mapped[line.number]
        origins = sorted({n for anchor in contributions for n in anchor.lines})
        if any(n not in source_lines or source_lines[n].page != line.page for n in origins):
            raise ValueError("output line inherited a different page/structural anchor")
        xy = _line_coordinates(contributions)
        coordinates[line.number] = xy
        edits = sorted({a.edit for a in contributions if a.edit is not None})
        lines.append(
            {
                "line_number": line.number,
                "page_index": line.page,
                "text": line.text,
                "source_lines": origins,
                "xy": xy,
                "edits": edits,
                "method": "unpositioned"
                if xy is None
                else ("edited_region_anchor" if edits else "unchanged_source_line"),
            }
        )
    positioned = enrich(rendered, coordinates)
    verify_preservation(rendered, positioned, set(coordinates))
    return positioned, {
        "method": "source_edit_regions_v2",
        "measuredSyntheticGeometry": False,
        "sourceSha256": digest(raw),
        "renderedSha256": digest(rendered.encode()),
        "alignmentSha256": digest(alignment),
        "editProofSha256": digest(proof),
        "positionedSha256": digest(positioned.encode()),
        "counts": dict(Counter(line["method"] for line in lines)),
        "lines": lines,
        "placements": placements,
    }


def _geometry_rows(path, pdfs: list[str], columns=None):
    """Stream selected PDFs without importing Arrow's dataset/Pandas stack.

    Row-group statistics prune disk reads, then exact identity filtering applies
    to every returned row. Files without statistics are read rather than skipped.
    Batch size bounds Python row materialization to 1,024 records.
    """
    wanted = set(pdfs)
    with pq.ParquetFile(path) as parquet:
        column = parquet.schema.names.index("document_id")
        groups = []
        for i in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(i).column(column).statistics
            if stats is not None and stats.has_min_max:
                index = bisect_left(pdfs, stats.min)
                if index == len(pdfs) or pdfs[index] > stats.max:
                    continue
            groups.append(i)
        # Small selected row groups: serial decoding avoids per-column worker
        # buffers (about 49 MiB saved in the 24-source pilot, <10 ms difference).
        for batch in parquet.iter_batches(
            batch_size=1024, row_groups=groups, columns=columns, use_threads=False
        ):
            yield from (r for r in batch.to_pylist() if r["document_id"] in wanted)


def load_page_geometry(project_root, dataset, alignments: dict) -> tuple[dict, dict]:
    """Load hash-pinned source geometry once, checking its link to each alignment.

    The real-input manifest already records the Paddle run and parquet hashes;
    do not introduce a second configurable geometry source that could diverge.
    """
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["config"]["alignment"]["grid_size"] != GRID:
        raise ValueError("source geometry uses a different coordinate grid")
    if sha256_file(dataset / "train.jsonl") != manifest["output_sha256"]["train.jsonl"]:
        raise ValueError("source training dataset differs from its spatial manifest")
    paddle = project_root / manifest["config"]["paddle_run"]
    hashes = {}
    for name in ("pages.parquet", "lines.parquet"):
        expected = {
            value
            for path, value in manifest["source_sha256"].items()
            if path.endswith("/" + manifest["config"]["paddle_run"] + "/" + name)
        }
        actual = sha256_file(paddle / name)
        if expected != {actual}:
            raise ValueError(f"source geometry hash mismatch: {name}")
        hashes[name] = actual
    pdfs = sorted({a["pdf_id"] for a in alignments.values()})
    pages, regions = {}, defaultdict(dict)
    for page in _geometry_rows(paddle / "pages.parquet", pdfs):
        key = (page["document_id"], page["page_index"])
        if (
            key in pages
            or page["coordinate_frame"] != "rendered_page"
            or page["source_sha256"] != key[0].removeprefix("pdf_")
            or not all(np.isfinite(page[k]) and page[k] > 0 for k in ("width", "height"))
        ):
            raise ValueError("duplicate, invalid or mismatched source page")
        pages[key] = page
    columns = [
        "document_id",
        "page_index",
        "line_index",
        "text",
        "recognition_score",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    for r in _geometry_rows(paddle / "lines.parquet", pdfs, columns):
        key = (r["document_id"], r["page_index"])
        if key not in pages or r["line_index"] in regions[key]:
            raise ValueError("missing page or duplicate source region")
        regions[key][r["line_index"]] = {
            "index": r["line_index"],
            "bbox": [r[k] for k in ("x1", "y1", "x2", "y2")],
            "text": r["text"],
            "recognition_score": r["recognition_score"],
        }
    normalized = {}
    for key, p in pages.items():
        if len(regions[key]) != p["recognized_lines"]:
            raise ValueError("source region count mismatch")
        boxes = np.array([r["bbox"] for r in regions[key].values()], dtype=float).reshape(-1, 4)
        normalized[key] = boxes * GRID / [p["width"], p["height"], p["width"], p["height"]]
    result = {}
    for sid, alignment in alignments.items():
        pdf = alignment["pdf_id"]
        if alignment["document_id"] != sid or pdf != "pdf_" + alignment["source_sha256"]:
            raise ValueError("alignment/PDF identity mismatch")
        result[sid] = {p: b for (d, p), b in normalized.items() if d == pdf}
        if not result[sid] or set(result[sid]) != set(range(len(result[sid]))):
            raise ValueError("missing or noncontiguous source pages")
        for line in alignment["lines"]:
            key = (pdf, line["page_index"])
            if key not in pages:
                raise ValueError("alignment belongs to a nonexistent page")
            matched = [regions[key][i] for i in line["region_indices"]]
            if matched != line["regions"]:
                raise ValueError("alignment regions differ from measured page")
            if matched:
                b = np.array([r["bbox"] for r in matched])
                box = [*b[:, :2].min(axis=0), *b[:, 2:].max(axis=0)]
                p = pages[key]
                if (
                    box != line["bbox"]
                    or list(centroid(box, p["width"], p["height"], GRID)) != line["xy"]
                ):
                    raise ValueError("alignment coordinate differs from measured regions")
            elif line["bbox"] is not None or line["xy"] is not None:
                raise ValueError("unmeasured source line has invented coordinates")
    return result, {
        "sourceDatasetManifestSha256": digest(manifest_bytes),
        "paddleRun": manifest["config"]["paddle_run"],
        "geometrySha256": hashes,
        "pageDimensions": {
            sid: {
                str(p): [pages[(alignment["pdf_id"], p)][k] for k in ("width", "height")]
                for p in result[sid]
            }
            for sid, alignment in alignments.items()
        },
    }


def augment_positions(
    rendered: str, receipt: dict, geometry: dict, policy: PositionPolicy, sample_id: str
) -> tuple[str, dict]:
    """Augment inherited anchors; unknown lines, exact text and page membership persist."""
    parsed = parse_lines(rendered, len(geometry))
    if [(p.number, p.page, p.text) for p in parsed] != [
        (r["line_number"], r["page_index"], r["text"]) for r in receipt["lines"]
    ]:
        raise ValueError("inherited receipt does not describe the rendered document")
    lines = [dict(line, anchor_xy=line["xy"]) for line in receipt["lines"]]
    pages = []
    for page, boxes in sorted(geometry.items()):
        known = [r for r in lines if r["page_index"] == page and r["xy"] is not None]
        points = np.array([r["xy"] for r in known], dtype=float).reshape(-1, 2)
        new, transform = augment_page(points, boxes, policy=policy, identity=f"{sample_id}:{page}")
        for line, xy in zip(known, new.tolist(), strict=True):
            line["xy"] = xy
        pages.append({"page_index": page, "known_points": len(known), **transform})
    coordinates = {r["line_number"]: r["xy"] for r in lines}
    text = enrich(rendered, coordinates)
    verify_preservation(rendered, text, set(coordinates))
    return text, {
        **receipt,
        "method": METHOD,
        "anchorPositionedSha256": receipt["positionedSha256"],
        "positionedSha256": digest(text.encode()),
        "policy": policy.model_dump(),
        "pages": pages,
        "lines": lines,
    }


def position_campaign(campaign) -> dict:
    """Produce a separate positioned input variant of a published plain-text pilot."""
    start = time.perf_counter()
    root = campaign.output
    manifest = json.loads((root / "manifest.json").read_text())
    payload = (root / "dataset.jsonl").read_bytes()
    if digest(payload) != manifest["files"]["dataset.jsonl"]:
        raise ValueError("plain publication hash mismatch")
    policy = PositionPolicy.model_validate(campaign.config["positions"])
    published = [json.loads(line) for line in payload.splitlines()]
    if len({r["documentId"] for r in published}) != len(published):
        raise ValueError("duplicate published sample identity")
    dataset = campaign.root / campaign.config["dataset"]
    alignments = {
        sid: json.loads((dataset / "alignments" / f"{sid}.json").read_bytes())
        for sid in {r["sourceDocumentId"] for r in published}
    }
    geometry, provenance = load_page_geometry(campaign.root, dataset, alignments)
    output = root / policy.output_subdirectory
    output.mkdir(exist_ok=True)
    rows, receipts, counts, modes = [], {}, Counter(), Counter()
    changed_points = 0
    gallery = [
        "# Synthetic augmented source-anchored positions\n\n"
        "Coherent per-page scale/translation; approximate inherited layout, "
        "not measured synthetic glyphs. "
        "Plain OCR and extraction targets are unchanged.\n"
    ]
    for row in published:
        sid, sample_id = row["sourceDocumentId"], row["documentId"]
        candidate = json.loads((root / "candidates" / f"{sample_id}.json").read_text())
        campaign.replay_candidate(candidate)
        if any(row[k] != candidate[k] for k in row):
            raise ValueError("published row differs from replayed candidate")
        alignment = alignments[sid]
        if alignment["document_id"] != sid:
            raise ValueError("source alignment belongs to another document")
        source_row = campaign.rows[sid]
        if digest(source_row["positionedText"].encode()) != alignment["positioned_input_sha256"]:
            raise ValueError("alignment differs from current source positioned text")
        text, receipt = transfer_positions(
            campaign.rows[sid]["joinedRawText"],
            row["joinedRawText"],
            candidate["proof"],
            alignment,
            page_boxes=geometry[sid],
            page_dimensions=provenance["pageDimensions"][sid],
        )
        text, receipt = augment_positions(
            row["joinedRawText"], receipt, geometry[sid], policy, sample_id
        )
        modes.update(p["mode"] for p in receipt["pages"])
        changed_points += sum(p["changed_points"] for p in receipt["pages"])
        row.update(positionedText=text, positionedTextSha256=digest(text.encode()))
        rows.append(row)
        receipts[sample_id] = digest(receipt)
        counts.update(receipt["counts"])
        _write_new_or_identical(
            output / f"{sample_id}.json",
            (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode(),
        )
        gallery.append(f"\n## {sample_id}\n\n```text\n{text.rstrip()}\n```\n")
    data = (
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    ).encode()
    md = "\n".join(gallery).encode()
    result = {
        "method": METHOD,
        "policy": policy.model_dump(),
        "provenance": provenance,
        "pageModes": dict(modes),
        "changedPoints": changed_points,
        "records": len(rows),
        "counts": dict(counts),
        "sourceManifestSha256": digest((root / "manifest.json").read_bytes()),
        "datasetSha256": digest(data),
        "gallerySha256": digest(md),
        "receipts": receipts,
    }
    _write_new_or_identical(output / "dataset.jsonl", data)
    _write_new_or_identical(output / "samples.md", md)
    _write_new_or_identical(
        output / "manifest.json", (json.dumps(result, indent=2) + "\n").encode()
    )
    return {**result, "seconds": time.perf_counter() - start}
