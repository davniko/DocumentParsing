#!/usr/bin/env python3
"""Compile and audit byte-exact OCR templates for the pinned 50-document cohort."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v4 import migrate_relation_v3_target_to_v4
from document_ocr.label_schemas.bill_of_lading_v5 import migrate_relation_v4_target_to_v5
from document_ocr.synthesis.raw_text_template import (
    TemplateSlot,
    build_template_slot,
    compile_raw_text_template,
    printed_topology_mismatches,
    render_compiled_template,
    sentinel_bindings,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_PAGE_HEADER = re.compile(r"(?m)^--- PAGE ([1-9][0-9]*) ---$")


class _PinnedJsonl(BaseModel):
    model_config = _STRICT

    path: NonEmptyText
    sha256: Sha256
    records: Annotated[int, Field(gt=0)]


class _PinnedFile(BaseModel):
    model_config = _STRICT

    path: NonEmptyText
    sha256: Sha256


class _Inputs(BaseModel):
    model_config = _STRICT

    source_corpus: _PinnedJsonl
    synthetic_targets: _PinnedJsonl
    anchors: _PinnedFile


class _ProofPolicy(BaseModel):
    model_config = _STRICT

    documents: Literal[50]
    require_source_hash: Literal[True]
    require_source_round_trip: Literal[True]
    require_disjoint_utf8_spans: Literal[True]
    require_exact_literal_regions: Literal[True]
    require_page_markers_unchanged: Literal[True]
    require_line_endings_unchanged: Literal[True]
    run_all_slot_sentinel_mutation: Literal[True]
    publish_training_records: Literal[False]


class _Config(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["bill_of_lading_raw_text_template_proof"]
    inputs: _Inputs
    proof: _ProofPolicy
    output_dir: NonEmptyText

    @model_validator(mode="after")
    def target_count_matches_proof(self) -> _Config:
        if self.inputs.synthetic_targets.records != self.proof.documents:
            raise ValueError("synthetic target count must equal proof document count")
        return self


def _load_jsonl(path: Path, expected: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_regular_file_bytes(path).splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{path} contains {len(rows)} rows, expected {expected}")
    return rows


def _resolve(project_root: Path, configured: str) -> Path:
    path = (project_root / configured).resolve(strict=True)
    if project_root not in path.parents:
        raise ValueError(f"configured path escapes project root: {configured}")
    return path


def _verify(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def _page_body_spans(value: str) -> dict[int, tuple[int, int]]:
    matches = list(_PAGE_HEADER.finditer(value))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, len(matches) + 1)):
        raise ValueError("joined OCR page headers must be contiguous from page one")
    output: dict[int, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        start = match.end()
        while start < section_end and value[start] == "\n":
            start += 1
        end = section_end
        while end > start and value[end - 1] == "\n":
            end -= 1
        output[int(match.group(1))] = (start, end)
    return output


def _render_policy(rows: Sequence[Mapping[str, Any]]) -> str:
    families = {str(row["surface_family"]) for row in rows}
    paths = {str(row["relation_target_path"]) for row in rows}
    if len(rows) == 1 and (
        "iso6346_compact" in families
        or any(
            path.endswith(("billOfLadingNumber", "voyageNumber", "containerNumber"))
            or ".sealNumbers[" in path
            for path in paths
        )
    ):
        return "opaque_identifier"
    if any(
        family.startswith(("iso_ymd", "numeric_dmy", "month_name", "date_"))
        for family in families
    ):
        return "date_surface"
    if families and families <= {"integer", "decimal_measure"}:
        return "numeric_surface"
    if any(path.endswith(("typeCategory", "sizeCategory")) for path in paths):
        return "categorical_surface"
    return "natural_text"


def _merged_slots(raw: str, anchors: Sequence[Mapping[str, Any]]) -> tuple[TemplateSlot, ...]:
    pages = _page_body_spans(raw)
    located: list[tuple[int, int, Mapping[str, Any]]] = []
    for row in anchors:
        if not row.get("patchable"):
            continue
        page = int(row["page_number"])
        page_start, page_end = pages[page]
        local_start = row.get("page_start")
        local_end = row.get("page_end")
        if not isinstance(local_start, int) or not isinstance(local_end, int):
            raise ValueError("patchable anchor lacks integer character offsets")
        absolute_start = page_start + local_start
        absolute_end = page_start + local_end
        if absolute_end > page_end or raw[absolute_start:absolute_end] != row["raw_value"]:
            raise ValueError("accepted anchor no longer matches the pinned raw OCR")
        located.append((absolute_start, absolute_end, row))
    located.sort(key=lambda item: (item[0], item[1]))

    groups: list[list[tuple[int, int, Mapping[str, Any]]]] = []
    for item in located:
        if not groups or item[0] >= max(row[1] for row in groups[-1]):
            groups.append([item])
        else:
            groups[-1].append(item)

    slots: list[TemplateSlot] = []
    for ordinal, group in enumerate(groups, start=1):
        char_start = min(row[0] for row in group)
        char_end = max(row[1] for row in group)
        source_text = raw[char_start:char_end]
        byte_start = len(raw[:char_start].encode("utf-8"))
        byte_end = len(raw[:char_end].encode("utf-8"))
        anchor_rows = [row[2] for row in group]
        paths = tuple(sorted({str(row["relation_target_path"]) for row in anchor_rows}))
        roles = sorted({str(row["role_path"]) for row in anchor_rows})
        slots.append(
            build_template_slot(
                slot_id=f"slot_{ordinal:04d}",
                byte_start=byte_start,
                byte_end=byte_end,
                source_text=source_text,
                target_paths=paths,
                semantic_role=roles[0] if len(roles) == 1 else "compound:" + ",".join(roles),
                evidence_origin="accepted_label_evidence",
                render_policy=_render_policy(anchor_rows),
            )
        )
    return tuple(slots)


def _topology_mismatches(source: Mapping[str, Any], target: Mapping[str, Any]) -> list[str]:
    return [
        f"{row.path}={row.value_kind} {row.source_count}->{row.target_count}"
        for row in printed_topology_mismatches(source, target)
    ]


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _plot(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="notebook")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.histplot(frame, x="patchable_fact_coverage", bins=12, ax=axes[0], color="#2878B5")
    axes[0].set(title="Accepted-label facts with unambiguous OCR spans", xlabel="Coverage")
    sns.histplot(frame, x="mutable_source_fraction", bins=12, ax=axes[1], color="#D95319")
    axes[1].set(title="Source bytes inside accepted-label slots", xlabel="Mutable byte fraction")
    figure.tight_layout()
    figure.savefig(output / "01_source_template_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    categories: Counter[str] = Counter()
    for row in rows:
        for category in cast(str, row["topology_mismatch_categories"]).split("|"):
            if category:
                categories[category.split("=", 1)[0]] += 1
    category_frame = pd.DataFrame(
        [{"category": key, "documents": value} for key, value in categories.most_common()]
    )
    figure, axis = plt.subplots(figsize=(12, max(4.5, 0.42 * max(1, len(category_frame)))))
    if not category_frame.empty:
        sns.barplot(category_frame, x="documents", y="category", ax=axis, color="#7A5195")
    axis.set(title="Target/source printed-topology mismatches", xlabel="Documents", ylabel="")
    figure.tight_layout()
    figure.savefig(output / "02_topology_mismatches.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    ladder = pd.DataFrame(
        [
            {
                "gate": "Pinned source hash",
                "documents": sum(row["source_hash_valid"] for row in rows),
            },
            {
                "gate": "Byte-exact source round trip",
                "documents": sum(row["round_trip_valid"] for row in rows),
            },
            {
                "gate": "All-slot isolation mutation",
                "documents": sum(row["sentinel_isolation_valid"] for row in rows),
            },
            {
                "gate": "Printed topology compatible",
                "documents": sum(row["topology_compatible"] for row in rows),
            },
            {"gate": "Complete mutable inventory", "documents": 0},
            {"gate": "Certified synthetic render", "documents": 0},
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 5.5))
    sns.barplot(ladder, x="documents", y="gate", ax=axis, color="#2A9D8F")
    axis.set(xlim=(0, 50), title="Proof gates passed by the current 50 pairs", ylabel="")
    figure.tight_layout()
    figure.savefig(output / "03_proof_ladder.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _artifact_manifest(output: Path, *, config_path: Path, config_sha256: str) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schemaVersion": 1,
        "configPath": config_path.as_posix(),
        "configSha256": config_sha256,
        "files": files,
    }


def run(project_root: Path, config_path: Path) -> dict[str, Any]:
    raw_config = yaml.safe_load(read_regular_file_bytes(config_path))
    config = _Config.model_validate(raw_config, strict=True)
    source_path = _resolve(project_root, config.inputs.source_corpus.path)
    target_path = _resolve(project_root, config.inputs.synthetic_targets.path)
    anchor_path = _resolve(project_root, config.inputs.anchors.path)
    _verify(source_path, config.inputs.source_corpus.sha256)
    _verify(target_path, config.inputs.synthetic_targets.sha256)
    _verify(anchor_path, config.inputs.anchors.sha256)

    source_rows = _load_jsonl(source_path, config.inputs.source_corpus.records)
    target_rows = _load_jsonl(target_path, config.inputs.synthetic_targets.records)
    sources = {str(row["documentId"]): row for row in source_rows}
    target_ids = {str(row["baseDocumentId"]) for row in target_rows}
    anchors_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(anchor_path):
        document_id = str(row["document_id"])
        if document_id in target_ids:
            anchors_by_document[document_id].append(row)

    output = (project_root / config.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"proof output already exists: {output}")
    output.mkdir(parents=True)
    (output / "templates").mkdir()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for target_row in target_rows:
        document_id = str(target_row["baseDocumentId"])
        source_row = sources[document_id]
        raw = cast(str, source_row["joinedRawText"])
        source_bytes = raw.encode("utf-8")
        source_hash_valid = sha256_bytes(source_bytes) == source_row["joinedRawTextSha256"]
        if not source_hash_valid:
            raise ValueError(f"joined raw-text hash mismatch: {document_id}")
        anchors = anchors_by_document[document_id]
        slots = _merged_slots(raw, anchors)
        compile_started = time.perf_counter_ns()
        template = compile_raw_text_template(
            document_id=document_id, source=source_bytes, slots=slots
        )
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
        render_started = time.perf_counter_ns()
        round_trip, proof = render_compiled_template(
            source=source_bytes,
            template=template,
            bindings={slot.slot_id: slot.source_text for slot in slots},
        )
        round_trip_render_ms = (time.perf_counter_ns() - render_started) / 1_000_000
        sentinel_output, sentinel_proof = render_compiled_template(
            source=source_bytes,
            template=template,
            bindings=sentinel_bindings(template),
            validate_format=False,
        )
        if round_trip != source_bytes or sentinel_output == source_bytes:
            raise RuntimeError(f"template proof failed: {document_id}")
        source_v5 = migrate_relation_v4_target_to_v5(
            migrate_relation_v3_target_to_v4(
                cast(Mapping[str, Any], source_row["target"])
            )
        )
        mismatches = _topology_mismatches(
            source_v5,
            cast(Mapping[str, Any], target_row["target"]),
        )
        expected_facts = len(
            {
                str(row["relation_target_path"])
                for row in anchors
            }
        )
        patchable_facts = len(
            {
                str(row["relation_target_path"])
                for row in anchors
                if row.get("patchable")
            }
        )
        template_payload = template.model_dump(mode="json")
        atomic_publish_json(output / "templates" / f"{document_id}.json", template_payload)
        rows.append(
            {
                "document_id": document_id,
                "scenario_id": target_row["scenarioId"],
                "source_hash_valid": source_hash_valid,
                "round_trip_valid": proof.source_round_trip,
                "sentinel_isolation_valid": (
                    sentinel_proof.exact_literal_regions
                    and sentinel_proof.page_markers_unchanged
                    and sentinel_proof.line_endings_preserved
                ),
                "source_bytes": len(source_bytes),
                "slots": len(slots),
                "compile_ms": compile_ms,
                "round_trip_render_ms": round_trip_render_ms,
                "mutable_source_bytes": proof.mutable_source_bytes,
                "mutable_source_fraction": proof.mutable_source_bytes / len(source_bytes),
                "expected_label_facts": expected_facts,
                "patchable_label_facts": patchable_facts,
                "patchable_fact_coverage": (
                    patchable_facts / expected_facts if expected_facts else 0.0
                ),
                "ambiguous_anchor_rows": sum(
                    row.get("location_status") == "ambiguous_repeated" for row in anchors
                ),
                "topology_compatible": not mismatches,
                "topology_mismatch_count": len(mismatches),
                "topology_mismatch_categories": "|".join(mismatches),
                "remaining_blockers": "|".join(map(str, target_row["remainingBlockers"])),
                "certified_synthetic_render": False,
            }
        )

    fields = list(rows[0])
    atomic_publish_bytes(output / "document-proof.csv", _csv_bytes(rows, fields))
    elapsed = time.perf_counter() - started
    compile_latencies = [cast(float, row["compile_ms"]) for row in rows]
    render_latencies = [cast(float, row["round_trip_render_ms"]) for row in rows]
    summary = {
        "schemaVersion": 1,
        "documents": len(rows),
        "sourceHashValid": sum(row["source_hash_valid"] for row in rows),
        "sourceRoundTripValid": sum(row["round_trip_valid"] for row in rows),
        "sentinelIsolationValid": sum(row["sentinel_isolation_valid"] for row in rows),
        "printedTopologyCompatible": sum(row["topology_compatible"] for row in rows),
        "completeMutableInventory": 0,
        "certifiedSyntheticRenders": 0,
        "patchableFacts": sum(row["patchable_label_facts"] for row in rows),
        "expectedLabelFacts": sum(row["expected_label_facts"] for row in rows),
        "mutableSourceBytes": sum(row["mutable_source_bytes"] for row in rows),
        "sourceBytes": sum(row["source_bytes"] for row in rows),
        "auditRuntimeSeconds": round(elapsed, 6),
        "compileLatencyMsMedian": round(statistics.median(compile_latencies), 6),
        "compileLatencyMsP95": round(sorted(compile_latencies)[46], 6),
        "renderLatencyMsMedian": round(statistics.median(render_latencies), 6),
        "renderLatencyMsP95": round(sorted(render_latencies)[46], 6),
        "trainingRecordsPublished": False,
        "conclusion": (
            "The byte renderer is proven on all 50 sources. The current source/target pairs are "
            "not proven end-to-end because mutable source-only facts are not exhaustively "
            "inventoried and some targets alter printed topology."
        ),
    }
    atomic_publish_json(output / "summary.json", summary)
    _plot(rows, output)

    mismatch_docs = [row for row in rows if not row["topology_compatible"]]
    report_lines = [
        "# Raw OCR compiled-template proof — 50 documents",
        "",
        "## Verdict",
        "",
        "The deterministic byte renderer itself passes its proof boundary on **50/50** "
        "pinned OCR sources:",
        "",
        f"- source hash verified: **{summary['sourceHashValid']}/50**;",
        f"- byte-identical source round trip: **{summary['sourceRoundTripValid']}/50**;",
        "- mutation of every compiled slot leaves every literal byte, page marker, and line "
        f"ending unchanged: **{summary['sentinelIsolationValid']}/50**.",
        "",
        "That is not yet a 50/50 end-to-end synthetic-data pass. Only "
        f"**{summary['printedTopologyCompatible']}/50** current source/target pairs preserve "
        "the compared printed-field topology, and the source labels are not an exhaustive "
        "inventory of source-only identifiers, parties, totals, equipment, and operational "
        "facts. Consequently, **0/50** are certified synthetic renders and no training rows "
        "were published. Reporting otherwise would reproduce the false-positive failure mode "
        "found in the prior audit.",
        "",
        "## What this proves",
        "",
        "The compiler binds mutable slots to exact UTF-8 byte offsets and the source SHA-256. "
        "Rendering requires exactly one binding per slot, refuses overlaps and source drift, "
        "and copies all non-slot ranges directly from the pinned source bytes. The result "
        "therefore cannot alter punctuation, whitespace, line endings, page headers, "
        "boilerplate, or OCR artifacts outside an approved slot.",
        "",
        "Across the cohort, accepted label evidence covers "
        f"**{summary['expectedLabelFacts']:,}** source facts; "
        f"**{summary['patchableFacts']:,}** have an unambiguous host-resolved span. Compiled "
        f"slots contain **{summary['mutableSourceBytes']:,}** of "
        f"**{summary['sourceBytes']:,}** source bytes. Exact per-document measurements and "
        "immutable templates are in `document-proof.csv` and `templates/`.",
        "",
        "The measured median/P95 compiler latency was "
        f"**{summary['compileLatencyMsMedian']:.3f}/{summary['compileLatencyMsP95']:.3f} ms** "
        "per document; median/P95 source-round-trip rendering was "
        f"**{summary['renderLatencyMsMedian']:.3f}/{summary['renderLatencyMsP95']:.3f} ms**. "
        "These measurements cover the byte compiler/renderer, not semantic template discovery "
        "or model-assisted natural-text realization.",
        "",
        "## What deterministic substitution cannot prove from the present inputs",
        "",
        "1. A label is a lossy task view, not a complete inventory of every shipment-specific "
        "surface in OCR. Values such as tax/booking references, signing agents, tare/total "
        "relationships, and anonymous equipment can remain outside label evidence.",
        "2. A generated value is not always a literal substitution. Package/equipment "
        "categories require an observed printed realization; long party addresses and cargo "
        "descriptions may require line-aware realization.",
        "3. A target that adds/removes a temperature, DG, HS, or auxiliary-information slot "
        "changes the template topology. Such a pair must be re-paired/regenerated or compiled "
        "with an explicitly approved structural edit.",
        "",
        "## Required hybrid boundary",
        "",
        "Use deterministic rendering for all approved slots. Use a model only once at "
        "template-compilation time for unresolved source-only spans or natural-language block "
        "realization. The model must return span proposals or replacement bindings—not a "
        "complete document. The host must resolve proposals to exact source spans, validate "
        "topology/format/semantics, and reject incomplete inventories. A one-time independent "
        "human audit is required before a compiled template becomes reusable; descendants can "
        "then render without model calls.",
        "",
        "## Current topology mismatches",
        "",
        f"**{len(mismatch_docs)}** documents have at least one printed-field topology mismatch "
        "under the strict comparison. See `document-proof.csv` and "
        "`02_topology_mismatches.png` for paths and counts. These are data-pairing/generation "
        "issues, not renderer failures.",
        "",
        "## Proof artifacts",
        "",
        "- `document-proof.csv`: one row per document and every proof gate.",
        "- `templates/`: exact source-bound compiled templates.",
        "- `summary.json`: machine-readable aggregate.",
        "- `01_source_template_coverage.png`: evidence and mutable-byte distributions.",
        "- `02_topology_mismatches.png`: mismatch taxonomy.",
        "- `03_proof_ladder.png`: renderer proof versus full-pair certification.",
    ]
    report = "\n".join(report_lines) + "\n"
    atomic_publish_bytes(output / "REPORT.md", report.encode("utf-8"))
    manifest = _artifact_manifest(
        output,
        config_path=config_path.relative_to(project_root),
        config_sha256=sha256_file(config_path),
    )
    atomic_publish_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve(strict=True)
    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    result = run(project_root, config_path.resolve(strict=True))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
