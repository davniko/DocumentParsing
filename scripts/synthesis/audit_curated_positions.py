"""Audit a published positional campaign and plot a deterministic review subset.

Uses saved production receipts, not experimental layout code. All records are
checked; plot selection includes high gains, low coverage, rejections and a
hash-selected spread. Optional line categories are diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.alignment import enrich, verify_preservation
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_layout import PositionPolicy, validate_transform
from document_ocr.synthesis.curated_positions import load_page_geometry
from document_ocr.synthesis.curated_reflow_geometry import validate_page


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def transformed(boxes, transform):
    return (
        (np.asarray(boxes).reshape(-1, 4) - 500) * transform["scale"]
        + 500
        + [
            transform["dx"],
            transform["dy"],
            transform["dx"],
            transform["dy"],
        ]
    )


def validate_receipt(row, receipt, source_boxes):
    """Check serialized boxes and final coordinates independently of seed replay."""
    lines = receipt["lines"]
    coordinates = {r["line_number"]: r["xy"] for r in lines}
    if len(coordinates) != len(lines):
        raise ValueError("duplicate emitted line")
    verify_preservation(row["joinedRawText"], row["positionedText"], set(coordinates))
    if enrich(row["joinedRawText"], coordinates) != row["positionedText"]:
        raise ValueError("receipt differs from positioned text")
    if digest(row["positionedText"].encode()) != receipt["positionedSha256"]:
        raise ValueError("positioned text hash differs")
    layouts = {p["page"]: p for p in receipt["reflowPages"]}
    affine = {p["page_index"]: p for p in receipt["pages"]}
    if set(layouts) != set(source_boxes) or set(affine) != set(source_boxes):
        raise ValueError("incomplete page receipts")
    for page, layout in layouts.items():
        page_lines = [r for r in lines if r["page_index"] == page]
        if layout["status"] == "accepted":
            failures = validate_page(
                layout["nodes"], layout["scaleY"], minimum_gap=layout["minimumGap"]
            )
            if failures or layout["failures"]:
                raise ValueError(f"accepted page has geometry failures: {failures}")
            boxes = np.array([n["box"] for n in layout["nodes"]]) * [
                1,
                layout["scaleY"],
                1,
                layout["scaleY"],
            ]
            indexed = {r["line_number"]: r for r in page_lines}
            for node in layout["nodes"]:
                previous_y = None
                for number, box in sorted(
                    node.get("lineBoxes", {}).items(), key=lambda kv: int(kv[0])
                ):
                    expected = [
                        round((box[0] + box[2]) / 2),
                        round((node["box"][1] + (box[1] + box[3]) / 2) * layout["scaleY"]),
                    ]
                    actual = indexed[int(number)]["anchor_xy"]
                    if expected != actual or (previous_y is not None and actual[1] <= previous_y):
                        raise ValueError("generated logical-centre placement/order differs")
                    previous_y = actual[1]
        else:
            if layout["status"] == "rejected" and not layout["failures"]:
                raise ValueError("rejection lacks explanation")
            boxes = source_boxes[page]
            if any(r["anchor_xy"] != r["transfer_xy"] for r in page_lines):
                raise ValueError("unaccepted page was partially edited")
        known = [r for r in page_lines if r["xy"] is not None]
        validate_transform(
            np.array([r["anchor_xy"] for r in known]).reshape(-1, 2),
            np.array([r["xy"] for r in known]).reshape(-1, 2),
            boxes,
            **{k: affine[page][k] for k in ("scale", "dx", "dy")},
        )
        if any(r["transfer_xy"] is not None and r["xy"] is None for r in page_lines):
            raise ValueError("previously known anchor lost")


def category(categories, sample, line):
    value = categories.get((sample, line), "other")
    return (
        "goods"
        if "goods_description" in value
        else "address"
        if "party_address" in value
        else "other"
    )


def draw_page(path, *, sample, sid, page, alignment, boxes, before, after, categories, font_path):
    """Four source-frame/augmented panels; rectangles are coarse layout proxies."""
    width, height, margin, top = 430, 680, 20, 70
    image = Image.new("RGB", (4 * (width + margin) + margin, height + 130), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 14)
    small = ImageFont.truetype(str(font_path), 10)
    source_lines = [r for r in alignment["lines"] if r["page_index"] == page]
    old_lines = [r for r in before["lines"] if r["page_index"] == page]
    new_lines = [r for r in after["lines"] if r["page_index"] == page]
    layout = next(p for p in after["reflowPages"] if p["page"] == page)
    old_transform = next(p for p in before["pages"] if p["page_index"] == page)
    new_transform = next(p for p in after["pages"] if p["page_index"] == page)
    if layout["status"] == "accepted":
        new_boxes = np.array([n["box"] for n in layout["nodes"]]) * [
            1,
            layout["scaleY"],
            1,
            layout["scaleY"],
        ]
    else:
        new_boxes = boxes
    panels = [
        ("Measured source", boxes, source_lines, "xy"),
        ("Previous synthetic input", transformed(boxes, old_transform), old_lines, "xy"),
        ("Joint reflow, before augmentation", new_boxes, new_lines, "anchor_xy"),
        ("Published reflow + augmentation", transformed(new_boxes, new_transform), new_lines, "xy"),
    ]
    for i, (title, regions, lines, key) in enumerate(panels):
        left = margin + i * (width + margin)
        draw.text((left, 12), title, font=font, fill="black")
        draw.text(
            (left, 34),
            f"Positioned {sum(r[key] is not None for r in lines)}/{len(lines)}",
            font=font,
            fill="#444444",
        )
        draw.rectangle((left, top, left + width, top + height), outline="#333333")
        for b in regions:
            draw.rectangle(
                (
                    left + b[0] * width / 1000,
                    top + b[1] * height / 1000,
                    left + b[2] * width / 1000,
                    top + b[3] * height / 1000,
                ),
                outline="#cccccc",
            )
        for r in lines:
            if r[key] is None:
                continue
            kind = category(categories, sample, r["line_number"]) if i else "other"
            color = {"goods": "#c32b25", "address": "#16833c", "other": "#275aaa"}[kind]
            x, y = left + r[key][0] * width / 1000, top + r[key][1] * height / 1000
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            if kind in {"goods", "address"}:
                draw.text((x + 3, y - 5), str(r["line_number"]), font=small, fill=color)
    draw.text(
        (margin, height + top + 12),
        f"{sid} | page {page + 1} | {layout['status']} | red=goods, green=address, blue=other",
        font=font,
        fill="black",
    )
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-subdirectory", required=True)
    parser.add_argument("--line-categories", type=Path)
    parser.add_argument("--plot-documents", type=int, default=20)
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.project_root.resolve()
    cfg = yaml.safe_load((root / args.config).read_text())
    policy = PositionPolicy.model_validate(cfg["positions"])
    if policy.reflow is None or args.plot_documents <= 0:
        raise ValueError("audit requires reflow configuration and a positive plot count")
    run = root / cfg["output"]
    current = run / policy.output_subdirectory
    baseline = run / args.baseline_subdirectory
    output = current / "audit"
    output.mkdir(exist_ok=True)
    rows = [json.loads(s) for s in (current / "dataset.jsonl").read_text().splitlines()]
    plain = {
        r["documentId"]: r
        for r in map(json.loads, (run / "dataset.jsonl").read_text().splitlines())
    }
    previous = {
        r["documentId"]: r
        for r in map(json.loads, (baseline / "dataset.jsonl").read_text().splitlines())
    }
    if (
        len(rows) != len(plain)
        or {r["documentId"] for r in rows} != plain.keys()
        or plain.keys() != previous.keys()
    ):
        raise ValueError("publications have different sample identities")
    manifest = read(current / "manifest.json")
    if sha256_file(current / "dataset.jsonl") != manifest["datasetSha256"]:
        raise ValueError("published dataset hash differs")
    if sha256_file(run / "dataset.jsonl") != read(run / "manifest.json")["files"]["dataset.jsonl"]:
        raise ValueError("plain dataset hash differs")
    dataset = root / cfg["dataset"]
    alignments = {
        sid: read(dataset / "alignments" / f"{sid}.json")
        for sid in sorted({r["sourceDocumentId"] for r in rows})
    }
    geometry, _ = load_page_geometry(root, dataset, alignments)
    categories = {}
    if args.line_categories:
        with (root / args.line_categories).open() as stream:
            categories = {
                (r["sample"], int(r["line"])): r["category"] for r in csv.DictReader(stream)
            }
    counts, per_field, statuses, failures = Counter(), defaultdict(Counter), Counter(), Counter()
    documents, receipts, before = [], {}, {}
    for row in rows:
        sample, sid = row["documentId"], row["sourceDocumentId"]
        if {
            k: v for k, v in row.items() if k not in {"positionedText", "positionedTextSha256"}
        } != plain[sample]:
            raise ValueError("text, labels or other sample metadata changed")
        receipt = read(current / f"{sample}.json")
        if digest(receipt) != manifest["receipts"][sample]:
            raise ValueError("receipt hash differs")
        if row["positionedTextSha256"] != receipt["positionedSha256"]:
            raise ValueError("row positioned hash differs")
        validate_receipt(row, receipt, geometry[sid])
        old = read(baseline / f"{sample}.json")
        if [(r["line_number"], r["text"]) for r in old["lines"]] != [
            (r["line_number"], r["text"]) for r in receipt["lines"]
        ]:
            raise ValueError("baseline line inventory differs")
        receipts[sample], before[sample] = receipt, old
        gain = Counter()
        for a, b in zip(old["lines"], receipt["lines"], strict=True):
            kind = category(categories, sample, b["line_number"])
            if categories and (sample, b["line_number"]) not in categories:
                raise ValueError("diagnostic line categories are incomplete")
            for counter in (counts, per_field[kind]):
                counter["lines"] += 1
                counter["before"] += a["xy"] is not None
                counter["after"] += b["xy"] is not None
                counter["lost"] += a["xy"] is not None and b["xy"] is None
            gain[kind] += (b["xy"] is not None) - (a["xy"] is not None)
        statuses.update(p["status"] for p in receipt["reflowPages"])
        failures.update(f[0] for p in receipt["reflowPages"] for f in p.get("failures", []))
        documents.append(
            dict(
                sample=sample,
                source=sid,
                goods_gain=gain["goods"],
                address_gain=gain["address"],
                total_gain=sum(gain.values()),
                unknown=sum(r["xy"] is None for r in receipt["lines"]),
                rejected=sum(p["status"] == "rejected" for p in receipt["reflowPages"]),
            )
        )
    if counts["lost"]:
        raise ValueError("lost coverage relative to published baseline")
    selected = []

    def choose(candidates, limit):
        added = 0
        for row in candidates:
            if (
                row["sample"] not in {r["sample"] for r in selected}
                and len(selected) < args.plot_documents
            ):
                selected.append(row)
                added += 1
                if added == limit:
                    break

    choose(
        sorted(documents, key=lambda r: -r["rejected"]),
        sum(r["rejected"] > 0 for r in documents) or 1,
    )
    choose(sorted(documents, key=lambda r: -r["goods_gain"]), 5)
    choose(sorted(documents, key=lambda r: -r["address_gain"]), 3)
    choose(sorted(documents, key=lambda r: -r["unknown"]), 5)
    choose(sorted(documents, key=lambda r: digest(r["sample"])), args.plot_documents)
    gallery = output / "gallery"
    gallery.mkdir(exist_ok=True)
    md = [
        "# Production reflow: source / previous / reflow / final coordinates\n",
        "All records were audited. This gallery deliberately includes large gains, remaining "
        "gaps, rejected proposals and a hash-selected spread. Rectangles are approximate layout "
        "envelopes, not measured synthetic PDF glyphs. Red points: goods; green: addresses; "
        "blue: other text. Point labels refer to physical OCR line numbers.\n",
    ]
    images = []
    row_lookup = {r["documentId"]: r for r in rows}
    for item in selected:
        sample, sid = item["sample"], item["source"]
        md.append(
            f"\n## {sample}\n\nSource: `{sid}`. Added coordinates: {item['total_gain']}; "
            f"goods: {item['goods_gain']}; addresses: {item['address_gain']}. "
            f"Still unknown: {item['unknown']}.\n"
        )
        for page in sorted(geometry[sid]):
            path = gallery / f"{sample}-p{page + 1}.png"
            draw_page(
                path,
                sample=sample,
                sid=sid,
                page=page,
                alignment=alignments[sid],
                boxes=geometry[sid][page],
                before=before[sample],
                after=receipts[sample],
                categories=categories,
                font_path=root / policy.reflow.font_path,
            )
            images.append(str(path.relative_to(output)))
            md.append(f"\n![Page {page + 1}]({path.relative_to(output)})\n")
        md.append(
            "\n<details><summary>Rendered input with final positions</summary>\n\n```text\n"
            + row_lookup[sample]["positionedText"].rstrip()
            + "\n```\n\n</details>\n"
        )
    (output / "GALLERY.md").write_text("\n".join(md))
    report = dict(
        records=len(rows),
        source_families=len(alignments),
        counts=dict(counts),
        by_field={k: dict(v) for k, v in per_field.items()},
        pages=dict(statuses),
        rejections=dict(failures),
        unchanged_plain_records=len(rows),
        source_plain_sha256=sha256_file(run / "dataset.jsonl"),
        baseline_sha256=sha256_file(baseline / "dataset.jsonl"),
        positioned_sha256=sha256_file(current / "dataset.jsonl"),
        plotted_documents=len(selected),
        plotted_pages=len(images),
        plots=images,
        seconds=time.perf_counter() - started,
        api_cost_usd=0,
    )
    save(output / "validation.json", report)
    save(output / "documents.json", documents)
    print(json.dumps({k: v for k, v in report.items() if k != "plots"}, indent=2))


if __name__ == "__main__":
    main()
