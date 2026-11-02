"""Run bounded draft-generation batches; never approve or publish their outputs."""

import argparse
import asyncio
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


async def run(phase, start, end, workers):
    manifest = json.loads((OUT / "selection.json").read_text())
    jobs = [
        j for j in manifest["cohortConfigs"] if j["phase"] == phase and start <= j["batch"] <= end
    ]
    semaphore = asyncio.Semaphore(workers)
    blocked = asyncio.Event()

    async def one(job):
        async with semaphore:
            if blocked.is_set():
                return {**job, "exitCode": 1, "status": "NOT_STARTED_PROVIDER_BLOCKED"}
            path = ROOT / job["config"]
            if sha256_file(path) != job["configSha256"]:
                raise ValueError("scheduled config changed before execution")
            if list(path.parent.glob("stage-runs/*/summary.json")):
                raise ValueError("batch already ran; resume or review explicitly")
            log = path.parent / "execution.log"
            with log.open("xb") as stream:
                proc = await asyncio.create_subprocess_exec(
                    str(ROOT / ".venv/bin/python"),
                    "scripts/synthesis/run_package_alias_batch.py",
                    "--config",
                    job["config"],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=stream,
                )
                try:
                    code = await proc.wait()
                except asyncio.CancelledError:
                    if proc.returncode is None:
                        proc.terminate()
                        await proc.wait()
                    raise
            # A per-document hold is not a reason to keep submitting batches
            # after an account-level refusal. Preserve the literal provider evidence.
            log_text = log.read_text(errors="replace")
            billing_block = "status_code: 402" in log_text
            if billing_block:
                blocked.set()
            receipt = {
                **job,
                "exitCode": code,
                "log": str(log.relative_to(ROOT)),
                "status": "PROVIDER_CREDITS_BLOCKED"
                if billing_block
                else "DRAFT_REQUIRES_MAIN_REVIEW"
                if code == 0
                else "EXECUTION_FAILED",
            }
            atomic_publish_json(path.parent / "execution-receipt.json", receipt)
            print(json.dumps(receipt), flush=True)
            return receipt

    results = await asyncio.gather(*(one(j) for j in jobs), return_exceptions=True)
    failures = [str(r) for r in results if isinstance(r, BaseException)]
    failures += [
        r["config"]
        for r in results
        if isinstance(r, dict) and (r["exitCode"] or "BLOCKED" in r["status"])
    ]
    if failures:
        raise RuntimeError(json.dumps(failures))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["ordinary", "thermal"], required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--workers", type=int, choices=[1, 2], default=2)
    a = p.parse_args()
    asyncio.run(run(a.phase, a.start, a.end, a.workers))
