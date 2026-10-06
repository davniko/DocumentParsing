"""Publish an immutable training-input variant with unchanged source OCR and gold labels."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.spatial_inputs.alignment import (
    AlignmentPolicy,
    Region,
    align_page,
    centroid,
    enclosing_box,
    enrich,
    parse_lines,
    verify_preservation,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


class Split(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["train", "validation", "test"]
    records: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SpatialDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    source_dataset: Path
    output_dataset: Path
    paddle_run: Path
    selection_manifest: Path
    selection_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    splits: tuple[Split, ...]
    alignment: AlignmentPolicy


def _read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(config: SpatialDatasetConfig, root: Path) -> dict[str, Any]:
    """Validate sources first; publish by directory rename only after exact replay passes."""
    start = time.perf_counter()
    source = (root / config.source_dataset).resolve()
    dest = (root / config.output_dataset).resolve()
    paddle = (root / config.paddle_run).resolve()
    selection_path = (root / config.selection_manifest).resolve()
    if dest.exists():
        raise FileExistsError(f"refusing to replace an existing dataset: {dest}")
    if source in dest.parents or dest in source.parents:
        raise ValueError("source and destination datasets must be separate")
    if not config.splits or len({s.name for s in config.splits}) != len(config.splits):
        raise ValueError("splits must be nonempty and unique")
    if sha256_file(selection_path) != config.selection_sha256:
        raise ValueError("source-selection manifest hash mismatch")
    manifest = _read(paddle / "manifest.json")
    artifacts = {a["path"]: a["sha256"] for a in manifest["artifacts"]}
    source_hashes = {str(selection_path): config.selection_sha256}
    for name in ("lines.parquet", "pages.parquet"):
        digest = sha256_file(paddle / name)
        if digest != artifacts[name]:
            raise ValueError(f"Paddle manifest hash mismatch: {name}")
        source_hashes[str(paddle / name)] = digest
    doc_map = {}
    for d in _read(selection_path)["documents"]:
        for doc_id in d["document_ids"]:
            if doc_id in doc_map:
                raise ValueError(f"duplicate source mapping: {doc_id}")
            doc_map[doc_id] = d
    rows = []
    seen = set()
    for split_spec in config.splits:
        path = source / f"{split_spec.name}.jsonl"
        if sha256_file(path) != split_spec.sha256:
            raise ValueError(f"source dataset hash mismatch: {split_spec.name}")
        source_hashes[str(path)] = split_spec.sha256
        current = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(current) != split_spec.records:
            raise ValueError(f"source record count mismatch: {split_spec.name}")
        for row in current:
            did = row["documentId"]
            if did in seen or did not in doc_map or "/" in did or "\\" in did:
                raise ValueError(f"missing/duplicate/unsafe source identity: {did}")
            seen.add(did)
            if row["joinedRawTextSha256"] != _digest(row["joinedRawText"]):
                raise ValueError(f"source input hash mismatch: {did}")
            if {"positionedText", "positionedTextSha256"} & row.keys():
                raise ValueError("source already has positioned inputs")
            if (source / "samples" / did / "ocr.txt").read_bytes() != row["joinedRawText"].encode():
                raise ValueError(f"source OCR/JSONL mismatch: {did}")
            if _read(source / "samples" / did / "labels.json") != row["target"]:
                raise ValueError(f"source labels/JSONL mismatch: {did}")
            rows.append((split_spec.name, row))
    contract = RelationExplicitTaskConstraints.model_validate_json(
        (source / "task-constraints.json").read_bytes()
    )
    task = get_training_task(contract.task).bind_constraints(contract)
    for _, row in rows:
        if task.canonicalize(row["target"]) != row["target"]:
            raise ValueError(
                "source labels require a semantic change; refusing input-only migration"
            )
    pdf_ids = {"pdf_" + doc_map[did]["source_sha256"] for did in seen}
    page_rows = pq.read_table(paddle / "pages.parquet").to_pylist()
    if len(page_rows) != manifest["counts"]["pages"]:
        raise ValueError("Paddle page count mismatch")
    pages = {
        (p["document_id"], p["page_index"]): p for p in page_rows if p["document_id"] in pdf_ids
    }
    expected_pages = sum(doc_map[did]["page_count"] for did in seen)
    if len(pages) != expected_pages:
        raise ValueError("missing or duplicate Paddle pages for selected PDFs")
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
    boxes = defaultdict(list)
    table = pq.read_table(
        paddle / "lines.parquet", columns=columns, filters=[("document_id", "in", sorted(pdf_ids))]
    )
    for b in table.to_pylist():
        boxes[b["document_id"], b["page_index"]].append(
            Region(
                b["line_index"],
                b["text"],
                tuple(b[k] for k in ("x1", "y1", "x2", "y2")),
                b["recognition_score"],
            )
        )
    for key, page in pages.items():
        boxes[key].sort(key=lambda r: r.index)
        if page["source_sha256"] != key[0].removeprefix("pdf_"):
            raise ValueError("Paddle PDF content identity mismatch")
        if page["coordinate_frame"] != "rendered_page":
            raise ValueError("this input format requires coordinates in the rendered-page frame")
        if len(boxes[key]) != page["recognized_lines"]:
            raise ValueError("Paddle recognized-line count mismatch")
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    counts: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    documents = []
    augmented_rows = defaultdict(list)
    for split, row in rows:
        did = row["documentId"]
        pdf_id = "pdf_" + doc_map[did]["source_sha256"]
        original = row["joinedRawText"]
        source_lines = parse_lines(original, doc_map[did]["page_count"])
        grouped = defaultdict(list)
        for line in source_lines:
            grouped[line.page].append(line)
        coordinates = {}
        receipts = []
        doc_counts: Counter[str] = Counter()
        for page_index, lines in grouped.items():
            key = (pdf_id, page_index)
            meta = pages[key]
            regions = boxes[key]
            matches = align_page(lines, regions, meta["width"], meta["height"], config.alignment)
            used: set[int] = set()
            for line, match in zip(lines, matches, strict=True):
                counts[match.method] += 1
                by_split[split][match.method] += 1
                doc_counts[match.method] += 1
                if used & set(match.regions):
                    raise ValueError("alignment assigned one region more than once")
                used.update(match.regions)
                box = enclosing_box(regions, match) if match.regions else None
                xy = (
                    centroid(box, meta["width"], meta["height"], config.alignment.grid_size)
                    if box
                    else None
                )
                coordinates[line.number] = xy
                receipts.append(
                    {
                        "line_number": line.number,
                        "page_index": page_index,
                        "block": line.block,
                        "original_text": line.text,
                        "method": match.method,
                        "region_indices": list(match.regions),
                        "bbox": box,
                        "xy": xy,
                        "anchor_lines": [lines[k].number for k in match.anchors],
                        "regions": [
                            {
                                "index": regions[i].index,
                                "text": regions[i].text,
                                "bbox": regions[i].box,
                                "recognition_score": regions[i].score,
                            }
                            for i in match.regions
                        ],
                    }
                )
        positioned = enrich(original, coordinates)
        verify_preservation(original, positioned, set(coordinates))
        result = {**row, "positionedText": positioned, "positionedTextSha256": _digest(positioned)}
        augmented_rows[split].append(result)
        for subdir, content in [("original", original), ("positioned", positioned)]:
            path = stage / "inputs" / subdir / f"{did}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode())
        labels_path = stage / "labels" / f"{did}.json"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "samples" / did / "labels.json", labels_path)
        atomic_publish_json(
            stage / "alignments" / f"{did}.json",
            {
                "document_id": did,
                "pdf_id": pdf_id,
                "source_sha256": doc_map[did]["source_sha256"],
                "original_input_sha256": row["joinedRawTextSha256"],
                "positioned_input_sha256": result["positionedTextSha256"],
                "lines": receipts,
            },
        )
        documents.append(
            {
                "document_id": did,
                "split": split,
                "pdf_id": pdf_id,
                "lines": len(receipts),
                "positioned": sum(v is not None for v in coordinates.values()),
                "methods": dict(doc_counts),
            }
        )
    output_hashes = {}
    for split, results in augmented_rows.items():
        p = stage / f"{split}.jsonl"
        p.write_bytes(
            b"".join(
                (json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                for r in results
            )
        )
        output_hashes[f"{split}.jsonl"] = sha256_file(p)
    for name in ("schema.json", "prompt-schema.json", "task-constraints.json"):
        shutil.copyfile(source / name, stage / name)
    # Read persisted JSONL, not just in-memory candidates; validate exact round-trip.
    published = {}
    for split_spec in config.splits:
        for encoded_line in (stage / f"{split_spec.name}.jsonl").read_bytes().splitlines():
            item = json.loads(encoded_line)
            did = item["documentId"]
            if _digest(item["positionedText"]) != item["positionedTextSha256"]:
                raise ValueError("persisted input hash mismatch")
            verify_preservation(
                item["joinedRawText"],
                item["positionedText"],
                {x.number for x in parse_lines(item["joinedRawText"], doc_map[did]["page_count"])},
            )
            if (stage / "inputs/positioned" / f"{did}.txt").read_bytes() != item[
                "positionedText"
            ].encode():
                raise ValueError("input file/JSONL mismatch")
            published[did] = {
                k: v for k, v in item.items() if k not in ("positionedText", "positionedTextSha256")
            }
    for _, row in rows:
        did = row["documentId"]
        if row != published[did]:
            raise ValueError("source fields or labels changed")
        if sha256_file(stage / "labels" / f"{did}.json") != sha256_file(
            source / "samples" / did / "labels.json"
        ):
            raise ValueError("label snapshot bytes changed")
    for source_path, digest in source_hashes.items():
        if sha256_file(Path(source_path)) != digest:
            raise ValueError("source changed while building")
    implementation = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    report = {
        "schema_version": 1,
        "source_dataset": str(config.source_dataset),
        "config": config.model_dump(mode="json"),
        "implementation_sha256": implementation,
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
        "split_records": dict(Counter(split for split, _ in rows)),
        "pages": expected_pages,
        "lines": sum(counts.values()),
        "positioned_lines": sum(d["positioned"] for d in documents),
        "methods": dict(counts),
        "by_split": dict(by_split),
        "documents": documents,
        "validation": {
            "unchanged_labels": len(rows),
            "unchanged_original_inputs": len(rows),
            "reversible_suffixes": len(rows),
            "region_reuse_conflicts": 0,
        },
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    atomic_publish_json(stage / "manifest.json", report)
    (stage / "README.md").write_text(
        "# Positional OCR input experiment\n\n"
        "Labels and original OCR are unchanged from the source dataset. "
        "inputs/original holds the original text; inputs/positioned contains the enriched text; "
        "labels contains byte-identical label snapshots. The JSONLs retain original fields and "
        "add positionedText and positionedTextSha256 for training.\n\n"
        f"Coordinates use a {config.alignment.grid_size}-unit page grid, top-left origin, "
        "and the centre of the enclosing matched text-region box. Unresolved nonblank lines "
        "end in ` ||` with no coordinates. Page markers and blank lines remain unchanged. "
        "Every assignment and abstention is recorded in alignments/. No fuzzy text replacement "
        "or label-derived position is used. This is an experimental input representation, "
        "not a claim that adding spatial tokens improves model accuracy.\n"
    )
    stage.rename(dest)
    return {k: v for k, v in report.items() if k not in ("documents", "source_sha256", "config")}
