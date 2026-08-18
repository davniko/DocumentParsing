#!/usr/bin/env python3
"""Prepare and finalize the bounded semantic-v2 four-document remediation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

from document_ocr.label_schemas.bill_of_lading import (
    BillOfLadingAnnotation,
    BillOfLadingExclusion,
    BillOfLadingLabel,
)
from document_ocr.label_schemas.mpci_bill_of_lading import MpciBillOfLadingAnnotation
from document_ocr.label_schemas.mpci_projection import project_bill_of_lading_to_mpci

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    ROOT / "artifacts/glm-ocr/pilots/blc150/runs/glm-ocr-blc-pilot150-faa3c717dbc4"
)
V1_RUN = ROOT / "artifacts/kie-labels/mpci-bl-pilot-v1-confined-r3"
DEST = ROOT / "artifacts/kie-labels/mpci-bl-semantic-v2-four-doc-r1"
RUN_ID = "mpci-bl-semantic-v2-four-doc-r1"
DOCUMENT_IDS = (
    "doc_4801b5c2bb9f42bbb9230de46eeb0546411474ec6fb220ecbf94fe7abc4b8458",
    "doc_9e250c2b03e31a25adeb92e09d3fddc43bc6847cf3193659c331f2a7c7af7840",
    "doc_564d4a7afe4727b965301ada963ace93adf693e279881afb6760a5e6b108be7a",
    "doc_6c21797c24bd50d3a25c84d1752ebedd7a8df801316b51ae4fe7e40328e85e8b",
)
_PAGE_MARKER = re.compile(r"^--- PAGE ([1-9][0-9]*) ---\n", re.MULTILINE)
_COUNTRY_CODES = {
    "BRA": "BR",
    "BRASIL": "BR",
    "EGYPT": "EG",
    "ESP": "ES",
    "KOREA": "KR",
    "NETHERLANDS": "NL",
    "TURKEY": "TR",
    "TURKIYE": "TR",
    "U.K": "GB",
    "UK": "GB",
    "UNITED KINGDOM": "GB",
}
_TARGET_CONTAMINATION = re.compile(
    r"(?ix)\b(?:"
    r"tax(?:\s*(?:id|no|number))?|vat|cif|v\.?d\.?|c\.?n\.?p\.?j|acid(?:\s*code)?|"
    r"shipper['\u2019]?s\s+load(?:\s*[,/&]\s*|\s+and\s+)count|"
    r"said\s+to\s+contain|s\.?t\.?c\.?|package\s+limitation\s+clause|"
    r"particulars\s+furnished\s+by\s+shipper|received\s+by\s+the\s+carrier"
    r")\b"
)
_CONTACT_LABEL = re.compile(r"(?i)\b(?:contact|e-?mail|tel(?:ephone)?|phone|ph|fax)\s*:")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values
        ).encode(),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _source_work_item(document_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(SOURCE_RUN / "state.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        document = connection.execute(
            "select * from documents where run_id=? and document_id=?",
            ("glm-ocr-blc-pilot150-faa3c717dbc4", document_id),
        ).fetchone()
        pages = connection.execute(
            "select * from pages where run_id=? and document_id=? order by page_index",
            ("glm-ocr-blc-pilot150-faa3c717dbc4", document_id),
        ).fetchall()
    finally:
        connection.close()
    if document is None or document["status"] != "complete":
        raise RuntimeError(f"source document is absent or incomplete: {document_id}")
    if [page["page_index"] for page in pages] != list(range(document["page_count"])):
        raise RuntimeError(f"source pages are incomplete or unordered: {document_id}")
    if any(page["status"] != "success" for page in pages):
        raise RuntimeError(f"source has a non-success page: {document_id}")

    joined: list[str] = []
    page_references: list[dict[str, Any]] = []
    for page in pages:
        result = json.loads(page["result_json"])
        raw_text = result["raw_ocr_text"]
        raw_path = SOURCE_RUN / result["raw_response_path"]
        raster_path = SOURCE_RUN / result["raster_path"]
        if _digest(raw_text.encode()) != page["ocr_text_sha256"]:
            raise RuntimeError(f"raw OCR digest mismatch: {page['page_id']}")
        if _digest(raw_path.read_bytes()) != page["raw_response_sha256"]:
            raise RuntimeError(f"raw response digest mismatch: {page['page_id']}")
        if _digest(raster_path.read_bytes()) != result["raster_sha256"]:
            raise RuntimeError(f"raster digest mismatch: {page['page_id']}")
        joined.append(f"--- PAGE {page['page_index'] + 1} ---\n{raw_text}")
        page_references.append(
            {
                "pageIndex": page["page_index"],
                "pageNumber": page["page_index"] + 1,
                "pageId": page["page_id"],
                "extractionId": page["extraction_id"],
                "rawOcrTextSha256": page["ocr_text_sha256"],
                "rawResponsePath": result["raw_response_path"],
                "rawResponseSha256": page["raw_response_sha256"],
                "rasterPath": result["raster_path"],
                "rasterSha256": result["raster_sha256"],
            }
        )
    joined_text = "\n\n".join(joined)
    source_json = json.loads(document["source_json"])
    source = {
        "documentId": document_id,
        "extractionRunId": "glm-ocr-blc-pilot150-faa3c717dbc4",
        "sourceUri": source_json["source_uri"],
        "localCanonicalPath": source_json["local_canonical_path"],
        "sourceSha256": document["source_sha256"],
        "documentPageCount": document["page_count"],
        "joinedRawTextSha256": _digest(joined_text.encode()),
        "pages": page_references,
    }
    work_item = {"source": source, "joinedRawText": joined_text}
    historical = _read_json(V1_RUN / "work-items" / f"{document_id}.json")
    if historical != work_item:
        raise RuntimeError(f"reconstructed work item differs from frozen r3: {document_id}")
    return work_item


def prepare() -> None:
    if DEST.exists():
        raise RuntimeError(f"destination already exists: {DEST}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{RUN_ID}.", dir=DEST.parent))
    try:
        for name in (
            "work-items",
            "candidate-attempts",
            "candidates",
            "validated",
            "exclusions",
            "training",
            "validation",
            "worker-logs",
        ):
            (temporary / name).mkdir()
        documents: list[dict[str, Any]] = []
        for document_id in DOCUMENT_IDS:
            work_item = _source_work_item(document_id)
            _atomic_json(temporary / "work-items" / f"{document_id}.json", work_item)
            documents.append(
                {
                    "documentId": document_id,
                    "pageCount": work_item["source"]["documentPageCount"],
                    "joinedRawTextSha256": work_item["source"]["joinedRawTextSha256"],
                    "workItemSha256": _digest(
                        (temporary / "work-items" / f"{document_id}.json").read_bytes()
                    ),
                }
            )
        _atomic_json(
            temporary / "run-metadata.json",
            {
                "runId": RUN_ID,
                "status": "prepared",
                "labelSchemaVersion": "2.0.0",
                "annotationSchemaVersion": "2.0.0",
                "sourceExtractionRunId": "glm-ocr-blc-pilot150-faa3c717dbc4",
                "historicalComparisonRun": str(V1_RUN.relative_to(ROOT)),
                "documents": documents,
                "workerContract": {
                    "model": "gpt-5.6-luna",
                    "reasoningEffort": "high",
                    "oneWorkerPerDocument": True,
                },
            },
        )
        os.replace(temporary, DEST)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"prepared {len(DOCUMENT_IDS)} work items at {DEST}")


def _page_texts(joined: str) -> dict[int, str]:
    markers = list(_PAGE_MARKER.finditer(joined))
    pages: dict[int, str] = {}
    for index, marker in enumerate(markers):
        page_number = int(marker.group(1))
        end = markers[index + 1].start() - 2 if index + 1 < len(markers) else len(joined)
        pages[page_number] = joined[marker.end() : end]
    if list(pages) != list(range(1, len(pages) + 1)):
        raise RuntimeError("joined OCR page markers are incomplete or unordered")
    return pages


def _verify_evidence_occurrences(
    *, evidence: list[dict[str, Any]], pages: dict[int, str], document_id: str
) -> None:
    for record in evidence:
        previous_page = 0
        previous_offset = -1
        for item in record.get("rawOcrEvidence", []):
            page_number = item["pageNumber"]
            source_page = pages.get(page_number)
            if source_page is None:
                raise RuntimeError(f"{document_id}: evidence cites absent page {page_number}")
            start = previous_offset if page_number == previous_page else 0
            offset = source_page.find(item["ocrExcerpt"], start)
            if offset < 0 or item["rawValue"] not in item["ocrExcerpt"]:
                raise RuntimeError(f"{document_id}: evidence is not verbatim/in source order")
            previous_page = page_number
            previous_offset = offset


def _country_resolver(value: str) -> str | None:
    normalized = re.sub(r"[^A-Z]+", " ", value.upper()).strip()
    exact = _COUNTRY_CODES.get(value.upper().strip())
    return exact if exact is not None else _COUNTRY_CODES.get(normalized)


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_leaf_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_leaf_count(child) for child in value)
    return 1


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(child, path))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def _compact_bytes(value: Any) -> int:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return len(serialized.encode())


def _benchmark_projection(labels: list[BillOfLadingLabel]) -> dict[str, Any]:
    """Measure the real semantic-to-MPCI path on the accepted pilot labels."""

    if not labels:
        raise RuntimeError("projection benchmark requires at least one accepted label")
    iterations_per_label = 2_000
    for _ in range(20):
        for label in labels:
            project_bill_of_lading_to_mpci(
                label, country_resolver=_country_resolver
            ).canonical_target()

    checksum = 0
    started_ns = time.perf_counter_ns()
    for _ in range(iterations_per_label):
        for label in labels:
            projected = project_bill_of_lading_to_mpci(
                label, country_resolver=_country_resolver
            ).canonical_target()
            checksum += _leaf_count(projected)
    elapsed_ns = time.perf_counter_ns() - started_ns
    projection_count = iterations_per_label * len(labels)

    tracemalloc.start()
    resident_outputs = [
        project_bill_of_lading_to_mpci(
            label, country_resolver=_country_resolver
        ).canonical_target()
        for label in labels
    ]
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not resident_outputs or checksum <= 0 or elapsed_ns <= 0:
        raise RuntimeError("projection benchmark did not exercise the projection path")

    return {
        "scope": "semantic-v2 to sparse MPCI projection plus canonical target materialization",
        "acceptedLabels": len(labels),
        "warmupIterationsPerLabel": 20,
        "measuredIterationsPerLabel": iterations_per_label,
        "projectionCount": projection_count,
        "elapsedMilliseconds": round(elapsed_ns / 1_000_000, 3),
        "meanMillisecondsPerProjection": round(
            elapsed_ns / projection_count / 1_000_000, 6
        ),
        "projectionsPerSecond": round(
            projection_count * 1_000_000_000 / elapsed_ns, 2
        ),
        "residentOutputBytes": current_bytes,
        "peakTracedBytes": peak_bytes,
        "checksum": checksum,
        "notes": (
            "Single-process CPU microbenchmark on the accepted four-document pilot labels; "
            "tracemalloc was measured separately from latency."
        ),
    }


def _semantic_contamination_findings(target: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    for path, value in _flatten(target).items():
        if not isinstance(value, str):
            continue
        if _TARGET_CONTAMINATION.search(value):
            findings.append(f"{path}: prohibited tax/boilerplate content {value!r}")
        if _CONTACT_LABEL.search(value):
            findings.append(f"{path}: field-label contamination {value!r}")

    parties = target.get("documentPatch", {}).get("parties", {})
    if isinstance(parties, dict):
        party_values = [
            value for key, value in parties.items() if key != "notifyParties" and value is not None
        ]
        notify = parties.get("notifyParties")
        if isinstance(notify, list):
            party_values.extend(notify)
        for party in party_values:
            if not isinstance(party, dict) or not isinstance(party.get("address"), str):
                continue
            address_words = set(re.findall(r"[A-Z0-9]+", party["address"].upper()))
            for component in ("city", "country"):
                value = party.get(component)
                if not isinstance(value, str):
                    continue
                component_words = set(re.findall(r"[A-Z0-9]+", value.upper()))
                if component_words and component_words <= address_words:
                    findings.append(
                        f"party address repeats separately emitted {component} {value!r}"
                    )
    return findings


def _validated_attempt(
    *, document_id: str, decision: dict[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    expected_keys = {"attemptPath", "decision", "trainingDisposition", "worker", "notes"}
    if set(decision) != expected_keys:
        raise RuntimeError(f"{document_id}: decision fields differ from contract")
    relative = Path(decision["attemptPath"])
    if relative.is_absolute() or relative.parent != Path("candidate-attempts"):
        raise RuntimeError(f"{document_id}: attemptPath escapes candidate-attempts")
    expected_prefix = f"{document_id}.attempt-"
    if not relative.name.startswith(expected_prefix) or relative.suffix != ".json":
        raise RuntimeError(f"{document_id}: attemptPath does not identify its document")
    attempt_path = DEST / relative
    raw_bytes = attempt_path.read_bytes()
    raw = json.loads(raw_bytes)
    work = _read_json(DEST / "work-items" / f"{document_id}.json")
    if raw.get("source") != work["source"]:
        raise RuntimeError(f"{document_id}: candidate source differs from immutable work item")
    pages = _page_texts(work["joinedRawText"])

    if decision["decision"] == "exclude":
        exclusion = BillOfLadingExclusion.model_validate_json(raw_bytes, strict=True)
        _verify_evidence_occurrences(
            evidence=[
                {"rawOcrEvidence": [item.model_dump(mode="json")]}
                for item in exclusion.rawOcrEvidence
            ],
            pages=pages,
            document_id=document_id,
        )
        if decision["trainingDisposition"] != "excluded":
            raise RuntimeError(f"{document_id}: excluded source has invalid training disposition")
        return "excluded", raw, None

    if decision["decision"] != "approve":
        raise RuntimeError(f"{document_id}: unsupported decision {decision['decision']!r}")
    if decision["trainingDisposition"] not in {"include", "duplicate_suppressed"}:
        raise RuntimeError(f"{document_id}: approved source has invalid training disposition")
    annotation = BillOfLadingAnnotation.model_validate_json(raw_bytes, strict=True)
    if annotation.reviewStatus != "candidate":
        raise RuntimeError(f"{document_id}: worker output is not a candidate")
    _verify_evidence_occurrences(
        evidence=[item.model_dump(mode="json") for item in annotation.evidence],
        pages=pages,
        document_id=document_id,
    )
    contamination = _semantic_contamination_findings(annotation.label.canonical_target())
    if contamination:
        raise RuntimeError(f"{document_id}: semantic contamination: {contamination!r}")
    projection = project_bill_of_lading_to_mpci(
        annotation.label, country_resolver=_country_resolver
    ).canonical_target()
    return "approved", raw, projection


def _comparison(document_id: str, v2_target: dict[str, Any]) -> dict[str, Any]:
    v1_path = V1_RUN / "validated" / f"{document_id}.json"
    v1 = MpciBillOfLadingAnnotation.model_validate_json(v1_path.read_bytes(), strict=True)
    v1_target = v1.label.canonical_target()
    return {
        "documentId": document_id,
        "v1Leaves": _leaf_count(v1_target),
        "v2Leaves": _leaf_count(v2_target),
        "v1CanonicalBytes": _compact_bytes(v1_target),
        "v2CanonicalBytes": _compact_bytes(v2_target),
    }


def _duplicate_comparison(
    first_id: str,
    first: dict[str, Any],
    second_id: str,
    second: dict[str, Any],
) -> dict[str, Any]:
    first_flat = _flatten(first["documentPatch"], "documentPatch")
    second_flat = _flatten(second["documentPatch"], "documentPatch")
    shared = sorted(set(first_flat) & set(second_flat))
    contradictions = [
        {"path": path, "first": first_flat[path], "second": second_flat[path]}
        for path in shared
        if first_flat[path] != second_flat[path]
    ]
    return {
        "documentIds": [first_id, second_id],
        "sharedLeafPaths": len(shared),
        "contradictions": contradictions,
        "firstOnlyPaths": sorted(set(first_flat) - set(second_flat)),
        "secondOnlyPaths": sorted(set(second_flat) - set(first_flat)),
    }


def _write_report(
    *,
    comparisons: list[dict[str, Any]],
    duplicate: dict[str, Any],
    approved: list[str],
    excluded: list[str],
    included: list[str],
    suppressed: list[str],
    attempt_reviews: list[dict[str, Any]],
    projection_benchmark: dict[str, Any],
) -> None:
    totals = {
        "v1Leaves": sum(item["v1Leaves"] for item in comparisons),
        "v2Leaves": sum(item["v2Leaves"] for item in comparisons),
        "v1CanonicalBytes": sum(item["v1CanonicalBytes"] for item in comparisons),
        "v2CanonicalBytes": sum(item["v2CanonicalBytes"] for item in comparisons),
    }
    totals["leafReduction"] = totals["v1Leaves"] - totals["v2Leaves"]
    totals["byteReduction"] = totals["v1CanonicalBytes"] - totals["v2CanonicalBytes"]
    report = {
        "status": "passed" if not duplicate["contradictions"] else "failed",
        "selected": len(DOCUMENT_IDS),
        "approved": len(approved),
        "excluded": len(excluded),
        "trainingIncluded": len(included),
        "duplicateSuppressed": len(suppressed),
        "approvedDocumentIds": sorted(approved),
        "excludedDocumentIds": sorted(excluded),
        "trainingDocumentIds": sorted(included),
        "duplicateSuppressedDocumentIds": sorted(suppressed),
        "workerAttempts": attempt_reviews,
        "workerAttemptCounts": {
            "total": len(attempt_reviews),
            "accepted": sum(item.get("result") == "accepted" for item in attempt_reviews),
            "rejected": sum(item.get("result") == "rejected" for item in attempt_reviews),
        },
        "v1V2Comparison": comparisons,
        "comparisonTotals": totals,
        "duplicateComparison": duplicate,
        "projection": {
            "status": "passed",
            "countryResolver": "bounded fixture map; production must inject versioned resolver",
            "benchmarkPath": "validation/projection-benchmark.json",
            "projectionsPerSecond": projection_benchmark["projectionsPerSecond"],
            "meanMillisecondsPerProjection": projection_benchmark[
                "meanMillisecondsPerProjection"
            ],
        },
        "semanticContaminationFindings": [],
        "qualityClaims": (
            "Leaf/byte reductions measure serialization only; they do not prove model accuracy."
        ),
    }
    if report["status"] != "passed":
        raise RuntimeError("duplicate semantic targets contradict on shared paths")
    _atomic_json(DEST / "validation" / "report.json", report)
    reduction = (
        100.0 * totals["byteReduction"] / totals["v1CanonicalBytes"]
        if totals["v1CanonicalBytes"]
        else 0.0
    )
    lines = [
        "# Semantic-v2 four-document validation",
        "",
        f"Status: **{report['status']}**.",
        "",
        f"- Selected: {len(DOCUMENT_IDS)}",
        f"- Approved annotations: {len(approved)}",
        f"- Fail-closed exclusions: {len(excluded)}",
        f"- Training records: {len(included)}",
        f"- Duplicate-suppressed: {len(suppressed)}",
        f"- Worker attempts: {len(attempt_reviews)}",
        (
            "- Rejected worker attempts: "
            f"{sum(item.get('result') == 'rejected' for item in attempt_reviews)}"
        ),
        (
            f"- Canonical JSON bytes across approved records: "
            f"{totals['v1CanonicalBytes']} v1 -> {totals['v2CanonicalBytes']} v2 "
            f"({reduction:.1f}% smaller)"
        ),
        (
            f"- Scalar leaves across approved records: {totals['v1Leaves']} v1 -> "
            f"{totals['v2Leaves']} v2"
        ),
        f"- Duplicate shared-path contradictions: {len(duplicate['contradictions'])}",
        (
            "- Semantic-to-MPCI projection microbenchmark: "
            f"{projection_benchmark['projectionsPerSecond']} projections/s "
            f"({projection_benchmark['meanMillisecondsPerProjection']} ms/projection)"
        ),
        "",
        "Serialization size is not an accuracy or learnability benchmark. See `report.json` for "
        "per-document and duplicate-path details.",
    ]
    _atomic_bytes(DEST / "validation" / "report.md", ("\n".join(lines) + "\n").encode())


def _manifest_entries() -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(DEST.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "manifest.json.sha256"}:
            continue
        files.append(
            {
                "path": path.relative_to(DEST).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _digest(path.read_bytes()),
            }
        )
    return files


def _manifest() -> None:
    files = _manifest_entries()
    _atomic_json(
        DEST / "manifest.json",
        {
            "runId": RUN_ID,
            "schemaVersion": "2.0.0",
            "publicationStatus": "complete",
            "files": files,
        },
    )
    _atomic_bytes(
        DEST / "manifest.json.sha256",
        (_digest((DEST / "manifest.json").read_bytes()) + "  manifest.json\n").encode(),
    )


def _recover_manifest_digest() -> None:
    manifest_path = DEST / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("runId") != RUN_ID
        or manifest.get("schemaVersion") != "2.0.0"
        or manifest.get("publicationStatus") != "complete"
        or manifest.get("files") != _manifest_entries()
    ):
        raise RuntimeError("incomplete publication manifest does not match current run files")
    _atomic_bytes(
        DEST / "manifest.json.sha256",
        (_digest(manifest_path.read_bytes()) + "  manifest.json\n").encode(),
    )


def finalize() -> None:
    if not DEST.is_dir():
        raise RuntimeError(f"run is not prepared: {DEST}")
    manifest_exists = (DEST / "manifest.json").exists()
    digest_exists = (DEST / "manifest.json.sha256").exists()
    if manifest_exists and digest_exists:
        raise RuntimeError("run is already published; create a new run ID")
    if digest_exists and not manifest_exists:
        raise RuntimeError("manifest digest exists without its manifest")
    if manifest_exists:
        _recover_manifest_digest()
        print(f"recovered final manifest digest at {DEST}")
        return
    decisions = _read_json(DEST / "validation" / "overseer-decisions.json")
    if decisions.get("decisionSchemaVersion") != "1.0.0":
        raise RuntimeError("overseer decision schema version mismatch")
    if set(decisions) != {
        "decisionSchemaVersion",
        "overseer",
        "documents",
        "attemptReviews",
    }:
        raise RuntimeError("overseer decision document fields differ from contract")
    decision_map = decisions.get("documents")
    if not isinstance(decision_map, dict) or set(decision_map) != set(DOCUMENT_IDS):
        raise RuntimeError("overseer decisions must cover exactly the four documents")
    for document_id, decision in decision_map.items():
        if not isinstance(decision, dict) or not isinstance(decision.get("attemptPath"), str):
            raise RuntimeError(f"{document_id}: decision must identify an attempt path")
    attempt_reviews = decisions.get("attemptReviews")
    if not isinstance(attempt_reviews, list) or not attempt_reviews:
        raise RuntimeError("overseer decisions require immutable attempt reviews")
    reviewed_paths: set[str] = set()
    for review in attempt_reviews:
        if not isinstance(review, dict) or set(review) != {
            "documentId",
            "attemptPath",
            "worker",
            "result",
            "findings",
        }:
            raise RuntimeError("attempt review fields differ from contract")
        if review["documentId"] not in DOCUMENT_IDS:
            raise RuntimeError("attempt review references an unexpected document")
        if review["result"] not in {"accepted", "rejected"}:
            raise RuntimeError("attempt review result is invalid")
        if not isinstance(review["findings"], list) or not review["findings"]:
            raise RuntimeError("attempt review requires at least one finding")
        attempt_path = review["attemptPath"]
        if attempt_path in reviewed_paths or not (DEST / attempt_path).is_file():
            raise RuntimeError("attempt review path is duplicate or absent")
        reviewed_paths.add(attempt_path)
        worker = review["worker"]
        if (
            not isinstance(worker, dict)
            or worker.get("model") != "gpt-5.6-luna"
            or worker.get("reasoningEffort") != "high"
            or not isinstance(worker.get("taskName"), str)
            or worker.get("usage") != "unavailable_from_collaboration_runtime"
        ):
            raise RuntimeError("attempt review worker contract is invalid")
    selected_paths = {str(decision["attemptPath"]) for decision in decision_map.values()}
    accepted_paths = {
        review["attemptPath"] for review in attempt_reviews if review["result"] == "accepted"
    }
    if selected_paths != accepted_paths:
        raise RuntimeError("selected attempts must exactly equal accepted attempt reviews")

    approved: list[str] = []
    excluded: list[str] = []
    included: list[str] = []
    suppressed: list[str] = []
    targets: dict[str, dict[str, Any]] = {}
    projections: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    accepted_labels: list[BillOfLadingLabel] = []

    for document_id in DOCUMENT_IDS:
        decision = decision_map[document_id]
        if not isinstance(decision, dict):
            raise RuntimeError(f"{document_id}: decision must be an object")
        status, raw, projection = _validated_attempt(
            document_id=document_id, decision=decision
        )
        _atomic_json(DEST / "candidates" / f"{document_id}.json", raw)
        if status == "excluded":
            _atomic_json(DEST / "exclusions" / f"{document_id}.json", raw)
            excluded.append(document_id)
            continue

        final = raw | {"reviewStatus": "validated"}
        final_json = json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        final_bytes = final_json.encode()
        BillOfLadingAnnotation.model_validate_json(final_bytes, strict=True)
        validated_path = DEST / "validated" / f"{document_id}.json"
        _atomic_bytes(validated_path, final_bytes)
        annotation = BillOfLadingAnnotation.model_validate_json(final_bytes, strict=True)
        accepted_labels.append(annotation.label)
        target = annotation.label.canonical_target()
        targets[document_id] = target
        if projection is None:
            raise RuntimeError(f"{document_id}: approved annotation lacks projection")
        projections[document_id] = projection
        approved.append(document_id)
        comparisons.append(_comparison(document_id, target))
        disposition = decision["trainingDisposition"]
        if disposition == "include":
            included.append(document_id)
            work = _read_json(DEST / "work-items" / f"{document_id}.json")
            records.append(
                {
                    "documentId": document_id,
                    "joinedRawText": work["joinedRawText"],
                    "joinedRawTextSha256": work["source"]["joinedRawTextSha256"],
                    "target": target,
                    "validatedAnnotationPath": f"validated/{document_id}.json",
                    "validatedAnnotationSha256": _digest(validated_path.read_bytes()),
                }
            )
        else:
            suppressed.append(document_id)

    first_id, second_id = DOCUMENT_IDS[:2]
    if first_id not in targets or second_id not in targets:
        raise RuntimeError("duplicate comparison pair must both be approved annotations")
    duplicate = _duplicate_comparison(
        first_id, targets[first_id], second_id, targets[second_id]
    )
    _atomic_json(DEST / "validation" / "duplicate-comparison.json", duplicate)
    sorted_records = sorted(records, key=lambda item: item["documentId"])
    _atomic_jsonl(DEST / "training" / "records.jsonl", sorted_records)
    _atomic_json(
        DEST / "validation" / "mpci-projections.json",
        {document_id: projections[document_id] for document_id in sorted(projections)},
    )
    projection_benchmark = _benchmark_projection(accepted_labels)
    _atomic_json(
        DEST / "validation" / "projection-benchmark.json", projection_benchmark
    )
    _write_report(
        comparisons=sorted(comparisons, key=lambda item: item["documentId"]),
        duplicate=duplicate,
        approved=approved,
        excluded=excluded,
        included=included,
        suppressed=suppressed,
        attempt_reviews=attempt_reviews,
        projection_benchmark=projection_benchmark,
    )
    metadata = _read_json(DEST / "run-metadata.json")
    metadata["status"] = "published"
    metadata["overseerDecisionPath"] = "validation/overseer-decisions.json"
    metadata["workerAttemptCount"] = len(attempt_reviews)
    _atomic_json(DEST / "run-metadata.json", metadata)
    _manifest()
    print(f"published {len(records)} training records and {len(excluded)} exclusions at {DEST}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "finalize"))
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare()
    else:
        finalize()


if __name__ == "__main__":
    main()
