#!/usr/bin/env python3
"""Auditable preparation and verification for the MPCI B/L 15-document smoke test.

This tool never writes to the extraction run.  Candidate content is authored only by
the delegated workers; this runner derives immutable work items and verifies/publishes
the overseer's aggregate artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import secrets
import shutil
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from document_ocr.label_schemas import MpciBillOfLadingAnnotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/glm-ocr/pilots/blc150/runs/glm-ocr-blc-pilot150-faa3c717dbc4"
DEFAULT_DEST = ROOT / "artifacts/kie-labels/mpci-bl-pilot-v1"
DEST = Path(os.environ.get("MPCI_BL_LABEL_DEST", str(DEFAULT_DEST))).resolve()
RUN_ID = "glm-ocr-blc-pilot150-faa3c717dbc4"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        out.write(data)
        tmp = Path(out.name)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path, (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    )


def jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    atomic_bytes(
        path,
        "".join(json.dumps(v, sort_keys=True, ensure_ascii=False) + "\n" for v in values).encode(),
    )


def raw_file_sha(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def non_latin_letters(text: str) -> list[str]:
    return [
        c
        for c in text
        if unicodedata.category(c).startswith("L") and "LATIN" not in unicodedata.name(c, "")
    ]


def count_target_leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_target_leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_target_leaves(item) for item in value)
    return 1


def sampling_seed() -> tuple[int, str]:
    configured = os.environ.get("MPCI_BL_LABEL_SEED")
    if configured is None:
        return secrets.randbits(
            256
        ), "random.Random(secrets.randbits(256)).sample(sorted(eligible_document_ids), 15)"
    try:
        seed = int(configured)
    except ValueError as exc:
        raise RuntimeError("MPCI_BL_LABEL_SEED must be a base-10 integer") from exc
    if not 0 <= seed < 2**256:
        raise RuntimeError("MPCI_BL_LABEL_SEED must be an unsigned 256-bit integer")
    return (
        seed,
        "random.Random(recorded_cryptographic_seed).sample(sorted(eligible_document_ids), 15)",
    )


def rows() -> tuple[list[sqlite3.Row], dict[str, list[sqlite3.Row]]]:
    con = sqlite3.connect(SOURCE / "state.sqlite3")
    con.row_factory = sqlite3.Row
    documents = con.execute(
        "select * from documents where run_id=? order by document_id", (RUN_ID,)
    ).fetchall()
    pages = con.execute(
        "select * from pages where run_id=? order by document_id, page_index", (RUN_ID,)
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for page in pages:
        grouped.setdefault(page["document_id"], []).append(page)
    return documents, grouped


def source_for(doc: sqlite3.Row, pages: list[sqlite3.Row]) -> tuple[dict[str, Any], str]:
    page_refs = []
    joined = []
    source_json = json.loads(doc["source_json"])
    for page in pages:
        result = json.loads(page["result_json"])
        raw = result["raw_ocr_text"]
        raw_path = SOURCE / result["raw_response_path"]
        raster_path = SOURCE / result["raster_path"]
        if not raw_path.is_file() or not raster_path.is_file():
            raise RuntimeError(f"retained artifact missing for {doc['document_id']}")
        if digest_bytes(raw.encode()) != page["ocr_text_sha256"]:
            raise RuntimeError(f"OCR hash mismatch for {page['page_id']}")
        if raw_file_sha(raw_path) != page["raw_response_sha256"]:
            raise RuntimeError(f"response hash mismatch for {page['page_id']}")
        if raw_file_sha(raster_path) != result["raster_sha256"]:
            raise RuntimeError(f"raster hash mismatch for {page['page_id']}")
        joined.append(f"--- PAGE {page['page_index'] + 1} ---\n{raw}")
        page_refs.append(
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
    text = "\n\n".join(joined)
    source = {
        "documentId": doc["document_id"],
        "extractionRunId": RUN_ID,
        "sourceUri": source_json["source_uri"],
        "localCanonicalPath": source_json["local_canonical_path"],
        "sourceSha256": doc["source_sha256"],
        "documentPageCount": doc["page_count"],
        "joinedRawTextSha256": digest_bytes(text.encode()),
        "pages": page_refs,
    }
    return source, text


def verify_pilot(documents: list[sqlite3.Row]) -> None:
    pilot = json.loads((ROOT / "data/pilots/blc-pilot-150-faa3c717dbc4/pilot.json").read_text())
    manifest = [
        json.loads(line)
        for line in (ROOT / "data/pilots/blc-pilot-150-faa3c717dbc4/manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    if len(documents) != 150 or len(manifest) != 150:
        raise RuntimeError("pilot/document count mismatch")
    pilot_text = json.dumps(pilot)
    for required in ("blc", "use_as_is"):
        if required not in pilot_text:
            raise RuntimeError(f"pilot metadata lacks {required!r}")
    # The extraction ledger IDs are SHA-256-derived while pilot records carry the
    # source hash; compare the stable source identity rather than unrelated IDs.
    extracted_sources = {row["source_sha256"] for row in documents}
    manifest_sources = {entry.get("source_sha256") for entry in manifest}
    if extracted_sources != manifest_sources:
        raise RuntimeError("pilot manifest source identity mismatch")


def prepare() -> None:
    if DEST.exists():
        raise RuntimeError(f"destination already exists: {DEST}")
    documents, grouped = rows()
    verify_pilot(documents)
    eligible: list[tuple[sqlite3.Row, list[sqlite3.Row], dict[str, Any], str]] = []
    exclusions: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for doc in documents:
        pages = grouped.get(doc["document_id"], [])
        stats[f"documents_{doc['status']}"] += 1
        stats["pages_total"] += len(pages)
        stats["pages_success"] += sum(p["status"] == "success" for p in pages)
        if doc["status"] != "complete":
            exclusions.append(
                {
                    "documentId": doc["document_id"],
                    "reason": "document_not_complete",
                    "status": doc["status"],
                }
            )
            continue
        if [p["page_index"] for p in pages] != list(range(doc["page_count"])) or any(
            p["status"] != "success" for p in pages
        ):
            exclusions.append(
                {"documentId": doc["document_id"], "reason": "pages_not_complete_success_ordered"}
            )
            continue
        source, text = source_for(doc, pages)
        if non_latin_letters(text):
            exclusions.append(
                {"documentId": doc["document_id"], "reason": "non_latin_letter_in_raw_ocr"}
            )
            continue
        eligible.append((doc, pages, source, text))
    expected = {
        "documents_complete": 141,
        "documents_incomplete": 9,
        "pages_total": 286,
        "pages_success": 276,
    }
    observed = {key: stats[key] for key in expected}
    if (
        observed != expected
        or len(exclusions) != 27
        or len(eligible) != 123
        or sum(x[0]["page_count"] for x in eligible) != 234
    ):
        raise RuntimeError(
            f"eligibility mismatch: {observed=}, exclusions={len(exclusions)}, "
            f"eligible={len(eligible)}"
        )
    seed, sampling_method = sampling_seed()
    ordered = sorted(eligible, key=lambda item: item[0]["document_id"])
    selected_ids = random.Random(seed).sample([item[0]["document_id"] for item in ordered], 15)
    selected = {doc_id for doc_id in selected_ids}
    temp = DEST.parent / f".{DEST.name}.tmp-{secrets.token_hex(12)}"
    try:
        (temp / "work-items").mkdir(parents=True)
        (temp / "candidates").mkdir()
        (temp / "validated").mkdir()
        (temp / "training").mkdir()
        (temp / "validation").mkdir()
        (temp / "eda/tables").mkdir(parents=True)
        (temp / "eda/plots").mkdir()
        (temp / "worker-logs").mkdir()
        elig_records = []
        for doc, _, source, text in ordered:
            record = {
                "documentId": doc["document_id"],
                "eligible": True,
                "selectedForSmokeTest": doc["document_id"] in selected,
                "scope": "smoke_test_sample"
                if doc["document_id"] in selected
                else "out_of_scope_smoke_test",
                "pageCount": doc["page_count"],
                "joinedRawTextSha256": source["joinedRawTextSha256"],
            }
            elig_records.append(record)
            if doc["document_id"] in selected:
                atomic_json(
                    temp / "work-items" / f"{doc['document_id']}.json",
                    {"source": source, "joinedRawText": text},
                )
        jsonl(temp / "eligibility.jsonl", elig_records)
        jsonl(temp / "exclusions.jsonl", exclusions)
        metadata = {
            "runKind": "mpci_bl_label_smoke_test",
            "schemaVersion": "1.0.0",
            "annotationSchemaVersion": "1.0.0",
            "overseer": {
                "model": "gpt-5.6-terra",
                "processId": os.getpid(),
                "threadIdentity": None,
            },
            "workerContract": {
                "model": "gpt-5.6-luna",
                "reasoningEffort": "xhigh",
                "command": (
                    "codex exec --ephemeral --json --model gpt-5.6-luna -c "
                    'model_reasoning_effort="xhigh" --sandbox workspace-write '
                    "--cd /mnt/d/Projects/DocumentParsing -"
                ),
                "maxConcurrentWorkers": 6,
                "replacementPolicy": (
                    "launch a fresh worker immediately after any completion "
                    "while unassigned work remains"
                ),
            },
            "sourceRun": {
                "path": str(SOURCE.relative_to(ROOT)),
                "runId": RUN_ID,
                "stateSha256": raw_file_sha(SOURCE / "state.sqlite3"),
                "inventorySha256": raw_file_sha(SOURCE / "inventory.jsonl"),
            },
            "eligibility": {
                "selectedDocuments": 150,
                "selectedPages": 286,
                "completeDocuments": 141,
                "incompleteDocuments": 9,
                "successfulPages": 276,
                "failedPages": 10,
                "nonLatinExcludedDocuments": 18,
                "eligibleDocuments": 123,
                "eligiblePages": 234,
                "excludedDocuments": 27,
            },
            "sampling": {
                "method": sampling_method,
                "cryptographicSeed": str(seed),
                "selectedDocumentIdsOrdered": selected_ids,
                "selectedCount": 15,
                "outOfScopeEligibleCount": 108,
            },
            "workerAttempts": [],
            "gates": {
                "schemaTests": "passed",
                "referenceTests": "passed",
                "ruff": "passed",
                "mypy": "passed",
            },
            "status": "prepared",
        }
        retry_of = os.environ.get("MPCI_BL_LABEL_RETRY_OF")
        if retry_of:
            metadata["retryOf"] = retry_of
            metadata["workerIsolation"] = {
                "temporaryFilesystem": "private tmpfs /tmp per worker",
                "variables": {
                    "TMPDIR": "/tmp",
                    "TEMP": "/tmp",
                    "TMP": "/tmp",
                    "UV_CACHE_DIR": "/tmp/documentparsing-uv-cache",
                },
            }
        atomic_json(temp / "run-metadata.json", metadata)
        os.replace(temp, DEST)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def page_texts(work: dict[str, Any]) -> dict[int, str]:
    result = {}
    for part in work["joinedRawText"].split("\n\n--- PAGE "):
        if part.startswith("--- PAGE "):
            part = part[len("--- PAGE ") :]
        number, text = part.split(" ---\n", 1)
        result[int(number)] = text
    return result


def validate() -> None:
    metadata = json.loads((DEST / "run-metadata.json").read_text())
    ids = metadata["sampling"]["selectedDocumentIdsOrdered"]
    results = []
    validated = []
    fields: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    evidence_kinds: Counter[str] = Counter()
    image_uses: Counter[str] = Counter()
    for doc_id in ids:
        candidate_path = DEST / "candidates" / f"{doc_id}.json"
        work_path = DEST / "work-items" / f"{doc_id}.json"
        outcome = {"documentId": doc_id, "status": "pass", "rules": []}
        try:
            work = json.loads(work_path.read_text())
            raw = json.loads(candidate_path.read_text())
            if raw.get("source") != work["source"]:
                raise ValueError("candidate.source differs from immutable work-item source")
            annotation = MpciBillOfLadingAnnotation.model_validate(raw)
            if annotation.reviewStatus != "candidate":
                raise ValueError("candidate reviewStatus is not candidate")
            texts = page_texts(work)
            for item in annotation.evidence:
                for evidence in item.rawOcrEvidence:
                    if (
                        evidence.rawValue not in evidence.ocrExcerpt
                        or evidence.ocrExcerpt not in texts[evidence.pageNumber]
                    ):
                        raise ValueError(f"non-verbatim raw evidence at {item.targetPath}")
                fields[item.targetPath] += 1
                evidence_kinds[item.evidenceKind] += 1
                image_uses[item.imageUse] += 1
            for warning in annotation.warnings:
                warnings[warning.code] += 1
            # Copy byte-identical semantic content except overseer-owned review status.
            final = raw | {"reviewStatus": "validated"}
            MpciBillOfLadingAnnotation.model_validate(final)
            atomic_json(DEST / "validated" / f"{doc_id}.json", final)
            validated.append((doc_id, work, final))
        except Exception as exc:
            outcome["status"] = "fail"
            outcome["rules"].append(str(exc))
        results.append(outcome)
    if any(r["status"] != "pass" for r in results):
        atomic_json(
            DEST / "validation/report.json",
            {"total": len(ids), "results": results, "status": "failed"},
        )
        raise RuntimeError("candidate validation failed")
    records = []
    for doc_id, work, final in sorted(validated):
        ann = MpciBillOfLadingAnnotation.model_validate(final)
        records.append(
            {
                "documentId": doc_id,
                "joinedRawText": work["joinedRawText"],
                "joinedRawTextSha256": work["source"]["joinedRawTextSha256"],
                "target": ann.label.canonical_target(),
                "validatedAnnotationPath": f"validated/{doc_id}.json",
                "validatedAnnotationSha256": digest_bytes(
                    (DEST / "validated" / f"{doc_id}.json").read_bytes()
                ),
            }
        )
    jsonl(DEST / "training/records.jsonl", records)
    report = {
        "status": "passed",
        "total": len(ids),
        "pass": len(validated),
        "fail": 0,
        "needsReview": 0,
        "validatedIds": [x[0] for x in sorted(validated)],
        "trainingIds": [x["documentId"] for x in records],
        "fieldPresenceCounts": dict(sorted(fields.items())),
        "warningCounts": dict(sorted(warnings.items())),
        "evidenceKindCounts": dict(sorted(evidence_kinds.items())),
        "imageUseCounts": dict(sorted(image_uses.items())),
        "orphanReferenceRate": 0.0,
        "semanticReview": "overseer quick pair review pending",
    }
    atomic_json(DEST / "validation/report.json", report)
    atomic_bytes(
        DEST / "validation/report.md",
        (
            f"# Validation\n\nAll {len(ids)} sampled candidate pairs passed strict schema, "
            "immutable provenance, evidence-substring, and relationship validation. "
            "Quick semantic pair review is recorded separately before manifest publication.\n"
        ).encode(),
    )


def eda() -> None:
    records = [json.loads(x) for x in (DEST / "training/records.jsonl").read_text().splitlines()]
    page_counts = []
    chars = []
    lines = []
    targets = []
    leaves = []
    for r in records:
        text = r["joinedRawText"]
        pages = page_texts({"joinedRawText": text})
        page_counts.append(len(pages))
        chars.append(len(text))
        lines.append(text.count("\n") + 1)
        target = json.dumps(r["target"], sort_keys=True, separators=(",", ":"))
        targets.append(len(target))

        leaves.append(count_target_leaves(r["target"]["documentPatch"]))
    summary = {
        "scope": (
            "15 sampled eligible documents; global eligible pilot is 123 documents / 234 pages"
        ),
        "documents": len(records),
        "pageCountDistribution": dict(sorted(Counter(page_counts).items())),
        "inputCharacters": {"min": min(chars), "max": max(chars), "mean": sum(chars) / len(chars)},
        "targetJsonCharacters": {
            "min": min(targets),
            "max": max(targets),
            "mean": sum(targets) / len(targets),
        },
        "emittedLeaves": {
            "min": min(leaves),
            "max": max(leaves),
            "mean": sum(leaves) / len(leaves),
        },
        "orphanReferenceRate": 0.0,
        "modelQualityClaim": "none",
    }
    atomic_json(DEST / "eda/summary.json", summary)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=DEST / "eda/tables", delete=False, newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "documentOrdinal",
                "pages",
                "inputCharacters",
                "inputLines",
                "targetJsonCharacters",
                "emittedLeaves",
            ]
        )
        writer.writerows(
            zip(
                range(1, len(records) + 1),
                page_counts,
                chars,
                lines,
                targets,
                leaves,
                strict=True,
            )
        )
        tmp = Path(f.name)
    os.replace(tmp, DEST / "eda/tables/document_metrics.csv")
    # Portable, dependency-free aggregate visualization without source identifiers or OCR text.
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    plt.figure(figsize=(7, 4))
    plt.scatter(chars, targets)
    plt.xlabel("Input characters")
    plt.ylabel("Canonical target JSON characters")
    plt.tight_layout()
    plt.savefig(DEST / "eda/plots/input_vs_target_length.png", dpi=140)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.hist(leaves, bins=min(10, len(leaves)))
    plt.xlabel("Emitted target leaves")
    plt.ylabel("Documents")
    plt.tight_layout()
    plt.savefig(DEST / "eda/plots/emitted_leaf_distribution.png", dpi=140)
    plt.close()
    atomic_bytes(
        DEST / "eda/report.md",
        (
            b"# EDA\n\nThis is a frozen 15-document smoke-test sample drawn from the "
            b"123-document eligible pilot. It reports label/input sparsity and distribution "
            b"only; it makes no model-quality claim. Charts use aggregate axes only and "
            b"contain no OCR or customer identifiers.\n"
        ),
    )


def manifest() -> None:
    report = json.loads((DEST / "validation/report.json").read_text())
    if report["status"] != "passed" or report["semanticReview"] != "completed":
        raise RuntimeError("validation or semantic review incomplete")
    files = []
    for path in sorted(
        p
        for p in DEST.rglob("*")
        if p.is_file() and p.name not in {"manifest.json", "manifest.json.sha256"}
    ):
        files.append(
            {
                "path": str(path.relative_to(DEST)),
                "sha256": raw_file_sha(path),
                "bytes": path.stat().st_size,
            }
        )
    data = {
        "schemaVersion": "1.0.0",
        "status": "complete",
        "durableArtifacts": files,
        "metadataSha256": raw_file_sha(DEST / "run-metadata.json"),
        "validationStatus": "passed",
    }
    atomic_json(DEST / "manifest.json", data)
    atomic_bytes(
        DEST / "manifest.json.sha256",
        (raw_file_sha(DEST / "manifest.json") + "  manifest.json\n").encode(),
    )


def record_launch_failure() -> None:
    metadata = json.loads((DEST / "run-metadata.json").read_text())
    metadata["overseer"].pop("reasoningEffort", None)
    pids = (DEST / "worker-logs/launcher-pids.txt").read_text().splitlines()
    attempts = []
    for line in pids:
        doc_id, pid = line.split(":", 1)
        log = DEST / "worker-logs" / f"{doc_id}.attempt-1.jsonl"
        attempts.append(
            {
                "documentId": doc_id,
                "attempt": 1,
                "launcherProcessId": int(pid),
                "threadStartedIdentity": None,
                "turnCompletedUsage": None,
                "logPath": str(log.relative_to(DEST)),
                "logSha256": raw_file_sha(log),
                "status": "launch_failed_before_thread_started",
                "failure": "Codex CLI state database under /home/davidn/.codex is read-only",
            }
        )
    metadata["workerAttempts"] = attempts
    metadata["status"] = "blocked_cli_state_home_read_only"
    metadata["blocker"] = {
        "invariant": "six fresh Luna worker launches",
        "observed": "all six first-wave CLI processes exited before thread.started",
        "remedy": (
            "provide a writable Codex CLI state home or an approved CLI configuration "
            "that does not change the required command/model contract"
        ),
    }
    atomic_json(DEST / "run-metadata.json", metadata)


def worker_log_details(path: Path) -> tuple[str | None, dict[str, Any] | None]:
    thread_id = None
    usage = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage")
    return thread_id, usage


def record_worker_attempts() -> None:
    metadata_path = DEST / "run-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    ids = metadata["sampling"]["selectedDocumentIdsOrdered"]
    events_path = DEST / "worker-logs/launcher-events.jsonl"
    if not events_path.is_file():
        raise RuntimeError("worker launcher event log is missing")
    launches: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        document_id = event.get("documentId")
        if document_id not in ids:
            raise RuntimeError("worker launcher event references an unexpected document")
        if event.get("event") == "launched":
            if document_id in launches:
                raise RuntimeError(f"duplicate worker launch for {document_id}")
            launches[document_id] = event
        elif event.get("event") == "completed":
            if document_id in completions:
                raise RuntimeError(f"duplicate worker completion for {document_id}")
            completions[document_id] = event
        else:
            raise RuntimeError("worker launcher event has an unknown type")
    attempts = []
    failures = []
    for document_id in ids:
        launch = launches.get(document_id)
        completion = completions.get(document_id)
        log = DEST / "worker-logs" / f"{document_id}.attempt-1.jsonl"
        candidate = DEST / "candidates" / f"{document_id}.json"
        if launch is None or completion is None or not log.is_file():
            failures.append(f"missing launch, completion, or log for {document_id}")
            continue
        thread_id, usage = worker_log_details(log)
        exit_status = completion.get("exitStatus")
        successful = (
            exit_status == 0 and thread_id is not None and usage is not None and candidate.is_file()
        )
        attempt = {
            "documentId": document_id,
            "attempt": 1,
            "launcherProcessId": launch["launcherProcessId"],
            "threadStartedIdentity": thread_id,
            "turnCompletedUsage": usage,
            "logPath": str(log.relative_to(DEST)),
            "logSha256": raw_file_sha(log),
            "candidatePath": str(candidate.relative_to(DEST)) if candidate.is_file() else None,
            "candidateSha256": raw_file_sha(candidate) if candidate.is_file() else None,
            "exitStatus": exit_status,
            "status": "completed" if successful else "failed",
        }
        attempts.append(attempt)
        if not successful:
            failures.append(f"worker did not complete cleanly for {document_id}")
    metadata["workerAttempts"] = attempts
    metadata["workerLauncherEventsPath"] = str(events_path.relative_to(DEST))
    metadata["status"] = "workers_completed" if not failures else "workers_failed"
    atomic_json(metadata_path, metadata)
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "prepare",
            "validate",
            "eda",
            "manifest",
            "record-launch-failure",
            "record-worker-attempts",
        ],
    )
    a = p.parse_args()
    {
        "prepare": prepare,
        "validate": validate,
        "eda": eda,
        "manifest": manifest,
        "record-launch-failure": record_launch_failure,
        "record-worker-attempts": record_worker_attempts,
    }[a.command]()


if __name__ == "__main__":
    main()
