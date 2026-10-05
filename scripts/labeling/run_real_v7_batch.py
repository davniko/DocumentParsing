"""Select fresh real sources and run the maintained direct labeling flow.

Candidates and review queues are experiment outputs, never automatically appended
to the manually adjudicated dataset. Selection and implementation are frozen.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextvars
import json
import os
import random
import resource
import shutil
import time
from collections import Counter
from pathlib import Path

import httpx
from dotenv import dotenv_values
from jsonschema import Draft202012Validator
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.labeling_agents.direct import DirectLabelingFlow, encoded
from document_ocr.labeling_agents.direct_models import DirectLabelingConfig

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / "artifacts/kie-training/analysis/real-data-baseline-audit-20261001/trained_real_v6.jsonl"
)
WORK = [
    ROOT / "artifacts/kie-labels" / p / "work-items"
    for p in (
        "mpci-bl-semantic-v2-pilot150-r1",
        "mpci-bl-semantic-v2-followup420-r3",
        "sources/mpci-bl-dual-cargo-v3-remaining1779-r1",
    )
]
WIRE_DIR = contextvars.ContextVar("wire_dir")


def usage(receipt):
    counts = Counter()
    for response in receipt.get("responses", []):
        u = response["usage"]
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            counts[key] += u.get(key, 0)
        counts["reasoning_tokens"] += u.get("details", {}).get("reasoning_tokens", 0)
    ordinary = counts["input_tokens"] - counts["cache_read_tokens"] - counts["cache_write_tokens"]
    assert ordinary >= 0
    # Same Luna estimate as the paired pilots; receipts, not this estimate, are
    # authoritative usage. No claim that estimated charges are settled invoices.
    cost = (
        ordinary * 0.10
        + counts["cache_read_tokens"] * 0.01
        + counts["cache_write_tokens"] * 0.125
        + counts["output_tokens"] * 0.50
    ) / 1e6
    return counts, cost


def verify(manifest):
    for path, digest in {**manifest["sources"], **manifest["implementation"]}.items():
        assert sha256_file(ROOT / path) == digest, path


def prepare(out, *, train, validation, seed, inventory=None, assignments=None):
    assert not out.exists(), "Batch directories are immutable; use a new batch name"
    accepted = ROOT / "data/curated/mpci-bl-real-v7-reviewed/batches/001/manifest.json"
    closed = json.loads(accepted.read_text())
    assert closed["samples"] == 50 and all(
        d["status"] == "manually_adjudicated" for d in closed["documents"]
    )
    exclusions, prior_hashes, previous = set(), set(), {}
    for path in sorted((ROOT / "artifacts/kie-labeling").glob("direct-*/selection.json")):
        manifest = json.loads(path.read_text())
        previous[str(path.relative_to(ROOT))] = sha256_file(path)
        for row in manifest.get("documents", []):
            exclusions.add(row["documentId"])
            prior_hashes.add(row["ocrSha256"])
    for row in closed["documents"]:
        exclusions.add(row["documentId"])
        prior_hashes.add(row["sourceOcrSha256"])
    records = list(map(json.loads, SOURCE.open()))
    assert len(records) == len({r["documentId"] for r in records})
    eligible = [
        r
        for r in records
        if r["documentId"] not in exclusions and r["joinedRawTextSha256"] not in prior_hashes
    ]
    if inventory is not None:
        screened = json.loads(inventory.read_text())
        assert screened["sourceSha256"] == sha256_file(SOURCE)
        allowed = {r["documentId"] for r in screened["documents"] if not r["exclusions"]}
        eligible = [r for r in eligible if r["documentId"] in allowed]
    split_assignments = {}
    if assignments is not None:
        assignment_data = json.loads(assignments.read_text())
        assert assignment_data["authorization"].strip()
        requested = assignment_data["documents"]
        assert len(requested) == len({d["documentId"] for d in requested}) == train + validation
        assert Counter(d["split"] for d in requested) == Counter(train=train, validation=validation)
        by_id = {r["documentId"]: r for r in eligible}
        selected = []
        for d in requested:
            r = by_id[d["documentId"]]
            assert d["historicalSplit"] == r["auditSplit"]
            selected.append(r)
            split_assignments[r["documentId"]] = d["split"]
    else:
        rng = random.Random(seed)
        selected = []
        for split, count in (("train", train), ("validation", validation)):
            pool = sorted(
                (r for r in eligible if r["auditSplit"] == split), key=lambda r: r["documentId"]
            )
            selected.extend(rng.sample(pool, count))
    assert len(selected) == len({r["joinedRawTextSha256"] for r in selected}) == train + validation
    sources = {
        str(SOURCE.relative_to(ROOT)): sha256_file(SOURCE),
        str(accepted.relative_to(ROOT)): sha256_file(accepted),
        **previous,
    }
    if inventory is not None:
        sources[str(inventory.relative_to(ROOT))] = sha256_file(inventory)
    if assignments is not None:
        sources[str(assignments.relative_to(ROOT))] = sha256_file(assignments)
    documents = []
    for index, r in enumerate(selected, 1):
        doc = r["documentId"]
        name = f"{index:03d}-{doc[4:12]}"
        matches = []
        for directory in WORK:
            work_path = directory / f"{doc}.json"
            if not work_path.exists():
                continue
            source = json.loads(work_path.read_text())["source"]
            pdf = ROOT / source["localCanonicalPath"].split("/DocumentParsing/", 1)[-1]
            if pdf.is_file() and sha256_file(pdf) == source["sourceSha256"]:
                matches.append((work_path, pdf, source["sourceSha256"]))
        assert matches and len({m[2] for m in matches}) == 1, (doc, matches)
        work_path, pdf, pdf_hash = sorted(matches)[0]
        folder = out / "inputs" / name
        text = r["joinedRawText"]
        assert sha256_bytes(text.encode()) == r["joinedRawTextSha256"]
        atomic_publish_bytes(folder / "ocr.txt", text.encode())
        # Old labels are deliberately not copied into the agent input folder.
        for path in (pdf, work_path, folder / "ocr.txt"):
            sources[str(path.relative_to(ROOT))] = sha256_file(path)
        documents.append(
            dict(
                index=index,
                name=name,
                documentId=doc,
                split=split_assignments.get(doc, r["auditSplit"]),
                historicalSplit=r["auditSplit"],
                input=str(folder.relative_to(ROOT)),
                ocrSha256=r["joinedRawTextSha256"],
                ocrChars=len(text),
                pdf=str(pdf.relative_to(ROOT)),
                pdfSha256=pdf_hash,
                pdfProvenance=str(work_path.relative_to(ROOT)),
            )
        )
    cfg = DirectLabelingConfig.model_validate_json(
        encoded(load_strict_yaml_mapping(ROOT / "configs/labeling_agents/mpci_bl_direct.yaml"))
    )
    assert (cfg.model, cfg.reasoning_effort, cfg.concurrency) == ("gpt-6-luna", "high", 16)
    atomic_publish_json(out / "config.json", cfg.model_dump(mode="json"))
    paths = [
        *sorted((ROOT / "src/document_ocr/labeling_agents").glob("direct*.py")),
        *sorted((ROOT / "src/document_ocr/label_schemas").glob("*.py")),
        *sorted((ROOT / "prompts/labeling_agents").glob("direct_*.md")),
        ROOT / cfg.package_registry,
        Path(__file__),
        out / "config.json",
    ]
    if inventory is not None:
        paths.append(ROOT / "scripts/labeling/screen_real_v7_topup.py")
    implementation = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        implementation[str(relative)] = sha256_file(path)
        dest = out / "implementation-snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    manifest = dict(
        seed=seed,
        selection=(
            f"Explicit approved allocation: {assignments.relative_to(ROOT)}"
            if assignments is not None else
            f"{train} train + {validation} validation, seeded random without replacement; "
            "excludes prior direct-pilot IDs and OCR hashes; original splits unchanged; "
            f"screening inventory: {inventory.relative_to(ROOT) if inventory else 'none'}"
        ),
        eligible=len(eligible),
        excludedPriorIds=sorted(exclusions),
        documents=documents,
        sources=sources,
        implementation=implementation,
        policy=(
            "Fresh OCR-only extraction, five scoped reviews, full-PDF candidate-blind "
            "cargo mapping and relation review, one scoped correction and final verification. "
            "No labels supplied from previous datasets. Candidates require separate "
            "adjudication before publication."
        ),
    )
    atomic_publish_json(out / "selection.json", manifest)
    verify(manifest)
    print(
        encoded(
            dict(
                prepared=len(documents),
                eligible=len(eligible),
                excluded=len(exclusions),
                splits=dict(Counter(r["split"] for r in documents)),
                output=str(out),
            )
        ),
        flush=True,
    )


async def run(out):
    manifest = json.loads((out / "selection.json").read_text())
    cfg = DirectLabelingConfig.model_validate_json((out / "config.json").read_bytes())
    verify(manifest)
    assert not (out / "runs").exists(), "No implicit rerun/overwrite"
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    sem = asyncio.Semaphore(cfg.concurrency)
    rows = {r["ocrSha256"]: r for r in manifest["documents"]}
    active = peak = wire_count = 0
    numbers = Counter()
    stop_admissions = asyncio.Event()
    started = time.monotonic()

    async def capture(request):
        nonlocal active, peak, wire_count
        if stop_admissions.is_set():
            raise RuntimeError(
                "Admissions paused after provider credit/authentication/schema failure"
            )
        body = json.loads(request.content)
        texts = [
            p["text"]
            for m in body["input"]
            for p in m.get("content", [])
            if p.get("type") == "input_text"
        ]
        matches = [rows[h] for t in texts if (h := sha256_bytes(t.encode())) in rows]
        assert len(matches) == 1, "Every request must contain the exact complete OCR text"
        fmt = body["text"]["format"]
        assert fmt["strict"] is True and fmt["type"] == "json_schema"
        assert body["reasoning"]["effort"] == "high"
        directory = WIRE_DIR.get()
        numbers[str(directory)] += 1
        pdf_files = []
        for message in body["input"]:
            for part in message.get("content", []):
                if part.get("type") == "input_file":
                    digest = sha256_bytes(base64.b64decode(part["file_data"].split(",", 1)[1]))
                    assert digest == matches[0]["pdfSha256"], "Complete original PDF required"
                    pdf_files.append(digest)
                    part["file_data"] = {"completePdfSha256": digest}
                elif part.get("type") == "input_image":
                    part["image_url"] = {
                        "redactedDataUrlSha256": sha256_bytes(part["image_url"].encode())
                    }
        # OpenAI wire schemas omit the root title; use the output format name.
        if fmt["name"] in ("CargoSourceMap", "CargoRelationResponse"):
            assert len(pdf_files) == 1, "Cargo mapping/review requires the complete PDF"
        atomic_publish_json(directory / "wire" / f"{numbers[str(directory)]:03d}.json", body)
        active += 1
        peak = max(active, peak)
        wire_count += 1

    async def received(response):
        nonlocal active
        active -= 1

    async with (
        httpx.AsyncClient(event_hooks={"request": [capture], "response": [received]}) as transport,
        AsyncOpenAI(
            api_key=key, max_retries=0, timeout=cfg.timeout_seconds, http_client=transport
        ) as sdk,
    ):
        model = OpenAIResponsesModel(cfg.model, provider=OpenAIProvider(openai_client=sdk))

        async def one(row):
            folder = out / "runs" / row["name"]
            flows = []
            extraction_status = "not_started"
            text = (ROOT / row["input"] / "ocr.txt").read_text()
            try:
                extract = DirectLabelingFlow(
                    model=model,
                    config=cfg,
                    project_root=ROOT,
                    output_dir=folder / "extract",
                    ocr=text,
                    request_semaphore=sem,
                )
                flows.append(extract)
                WIRE_DIR.set(extract.output_dir)
                try:
                    target = await extract.extract()
                    candidate = target.canonical_target()
                    extraction_status = "application_valid"
                except Exception:
                    # Invalid application draft may be reviewed, never accepted:
                    # require one intact strict-wire-schema-valid response first.
                    if (
                        len(extract.receipts) != 1
                        or len(extract.receipts[0].get("responses", [])) != 1
                    ):
                        raise
                    raw = json.loads("".join(extract.receipts[0]["responses"][0]["text"]))
                    wire = json.loads((extract.output_dir / "wire/001.json").read_text())
                    Draft202012Validator(wire["text"]["format"]["schema"]).validate(raw)
                    candidate = raw
                    extraction_status = "application_invalid_native_valid_draft"
                    atomic_publish_json(extract.output_dir / "invalid-draft.json", raw)
                atomic_publish_json(
                    folder / "extraction-summary.json", {"status": extraction_status}
                )
                refine = DirectLabelingFlow(
                    model=model,
                    config=cfg,
                    project_root=ROOT,
                    output_dir=folder / "refine",
                    ocr=text,
                    pdf_path=ROOT / row["pdf"],
                    request_semaphore=sem,
                )
                flows.append(refine)
                WIRE_DIR.set(refine.output_dir)
                result = await refine.refine(candidate)
                summary = dict(
                    name=row["name"],
                    documentId=row["documentId"],
                    split=row["split"],
                    extraction=extraction_status,
                    status=result["status"],
                    applicationValid=result["target"] is not None,
                )
            except Exception as error:
                summary = dict(
                    name=row["name"],
                    documentId=row["documentId"],
                    split=row["split"],
                    extraction=extraction_status,
                    status="failed",
                    errorType=type(error).__name__,
                    diagnostic=str(error),
                )
                diagnostics = json.dumps([r for f in flows for r in f.receipts])
                if any(
                    s in diagnostics
                    for s in (
                        "credit_balance_exhausted",
                        "insufficient_quota",
                        "invalid_api_key",
                        "invalid_json_schema",
                    )
                ):
                    stop_admissions.set()
                atomic_publish_json(folder / "failure.json", summary)
            receipts = [r for f in flows for r in f.receipts]
            summary.update(
                calls=len(receipts),
                estimatedUsd=sum(usage(r)[1] for r in receipts),
                unknownBillingCalls=sum(r.get("billingStatus") == "unknown" for r in receipts),
            )
            atomic_publish_json(folder / "summary.json", summary)
            print(encoded(summary), flush=True)
            return summary

        results = await asyncio.gather(*(one(row) for row in manifest["documents"]))
    verify(manifest)
    timing = dict(
        elapsedSeconds=time.monotonic() - started,
        peakConcurrentRequests=peak,
        wireRequests=wire_count,
        admissionsStopped=stop_admissions.is_set(),
        peakRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        statuses=dict(Counter(r["status"] for r in results)),
        estimatedUsd=sum(r["estimatedUsd"] for r in results),
        unknownBillingCalls=sum(r["unknownBillingCalls"] for r in results),
        results=results,
    )
    atomic_publish_json(out / "timing.json", timing)
    atomic_publish_json(
        out / "review-queue.json", [r for r in results if r["status"] != "reviewed_candidate"]
    )
    print(encoded({k: v for k, v in timing.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=90)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026100402)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--assignments", type=Path,
                        help="Explicit source IDs/splits with historical splits and user authorization.")
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    out.relative_to(ROOT)
    if args.action == "prepare":
        prepare(out, train=args.train, validation=args.validation, seed=args.seed,
                inventory=(ROOT / args.inventory).resolve() if args.inventory else None,
                assignments=(ROOT / args.assignments).resolve() if args.assignments else None)
    else:
        asyncio.run(run(out))
