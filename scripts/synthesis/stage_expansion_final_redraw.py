"""Preserve campaign quotas while redrawing explicitly incomplete source groups."""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from manage_r28_expansion import redraw
from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
AUTHORITY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v19/complete-source-authority-config.json"


def stage():
    jobs = []
    for previous in sorted(RUN.glob("batch-*/**/redraw-2/config.json")):
        config = json.loads(previous.read_text())
        candidates = Counter(json.loads(p.read_text())["sourceDocumentId"]
                             for p in (previous.parent / "campaign/candidates").glob("*.json"))
        component = previous.parent.relative_to(RUN).as_posix()
        reasons = {}
        for sid, count in config["template_sampling"]["source_counts"].items():
            if component == "batch-17/thermal/redraw-2" and sid.startswith("doc_639f54cd"):
                reasons[sid] = "Main review holds chilled coalfish paired with granular-grain bulk category; redraw the native goods/package proposal."
            elif candidates[sid] != count:
                reasons[sid] = "Previous source group lacks its complete rendered and reviewed variants; retain all attempt evidence and redraw its entire scheduled group."
        if not reasons:
            continue
        parent = previous.parent.parent / "config.json"
        # Original parent scope can contain more variants than the replacement
        # scope (e.g. an explicit staged withdrawal). Preserve replacement counts.
        base = json.loads(parent.read_text())
        for sid in reasons:
            if base["template_sampling"]["source_counts"][sid] != config["template_sampling"]["source_counts"][sid]:
                raise ValueError("redraw cardinality needs explicit variant selection: " + sid)
        review = OUT / "redraw-plans/round3" / component.replace("/", "-") / "review.json"
        atomic_publish_json(review, {"redrawSources": reasons,
                                   "scope": "Unfinished original campaign slots only; no replacements of admitted records."})
        redraw(parent, review, 3, AUTHORITY)
        path = parent.parent / "redraw-3/config.json"
        jobs.append({"config": str(path.relative_to(ROOT)), "configSha256": sha256_file(path),
                     "samples": sum(config["template_sampling"]["source_counts"][s] for s in reasons)})
    manifest = OUT / "redraw-plans/round3.json"
    atomic_publish_json(manifest, {"jobs": jobs, "samples": sum(j["samples"] for j in jobs)})
    print(json.dumps({"jobs": len(jobs), "samples": sum(j["samples"] for j in jobs)}), flush=True)


async def execute(manifest):
    jobs = json.loads(manifest.read_text())["jobs"]
    semaphore = asyncio.Semaphore(2)
    blocked = asyncio.Event()

    async def one(job):
        async with semaphore:
            if blocked.is_set():
                raise RuntimeError("provider billing blocked; remaining jobs not started")
            path = ROOT / job["config"]
            if sha256_file(path) != job["configSha256"]:
                raise ValueError("scheduled config changed")
            log = path.parent / "execution.log"
            with log.open("xb") as stream:
                process = await asyncio.create_subprocess_exec(
                    str(ROOT / ".venv/bin/python"), "scripts/synthesis/run_package_alias_batch.py",
                    "--config", str(path), "--retry-transient-refusals", "--concurrency", "2",
                    cwd=ROOT, stdout=stream, stderr=stream)
                code = await process.wait()
            if "status_code: 402" in log.read_text(errors="replace"):
                blocked.set()
            atomic_publish_json(path.parent / "execution-receipt.json", {
                **job, "exitCode": code, "status": "DRAFT_REQUIRES_MAIN_REVIEW",
                "log": str(log.relative_to(ROOT)),
            })
            print(job["config"], "exit", code, flush=True)
            if code:
                raise RuntimeError("generation execution failed; inspect preserved log")

    await asyncio.gather(*(one(job) for job in jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "execute"])
    parser.add_argument("--manifest", type=Path, default=OUT / "redraw-plans/round3.json")
    args = parser.parse_args()
    if args.command == "stage":
        stage()
    else:
        asyncio.run(execute(args.manifest))
