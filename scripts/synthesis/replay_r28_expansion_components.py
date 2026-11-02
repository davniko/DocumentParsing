"""Replay explicitly selected unadmitted components, preserving their old receipts."""

import argparse
import asyncio
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    semaphore = asyncio.Semaphore(args.workers)

    async def replay(component):
        async with semaphore:
            directory = (RUN / component).resolve()
            if not directory.is_relative_to(RUN) or (directory / "campaign/manifest.json").exists():
                raise ValueError("replay requires an unadmitted campaign inside the expansion")
            configs = sorted((directory / "campaign/declared-package-preparation").glob("*/prepared-config.json"))
            if not configs:
                raise ValueError("component has no prepared draw: " + component)
            config = configs[-1]
            dest = OUT / "replays" / args.name / component
            pins = []
            for kind in ("candidates", "wording", "reviews", "contacts", "postal"):
                for path in sorted((directory / "campaign" / kind).glob("*.json")):
                    digest = sha256_file(path)
                    archived = dest / "before" / kind / path.name
                    if kind == "candidates":
                        archived.parent.mkdir(parents=True, exist_ok=True)
                        if archived.exists():
                            raise ValueError("candidate replay archive already exists")
                        path.rename(archived)
                    else:
                        atomic_publish_bytes(archived, path.read_bytes())
                    pins.append({"path": str(path.relative_to(ROOT)), "sha256": digest,
                                 "archive": str(archived.relative_to(ROOT))})
            atomic_publish_json(dest / "scope.json", {"config": str(config.relative_to(ROOT)),
                "configSha256": sha256_file(config), "before": pins,
                "scope": "Fresh current-contract wording where required, render, native review and geometry. No dataset admission; all obsolete evidence retained."})
            with (dest / "execution.log").open("xb") as stream:
                process = await asyncio.create_subprocess_exec(str(ROOT / ".venv/bin/python"),
                    "scripts/synthesis/run_package_alias_batch.py", "--config", str(config),
                    "--prepared", "--retry-transient-refusals", "--concurrency", "2",
                    cwd=ROOT, stdout=stream, stderr=stream)
                code = await process.wait()
            atomic_publish_json(dest / "result.json", {"exitCode": code, "status": "REQUIRES_MAIN_REVIEW"})
            print(component, "exit", code, flush=True)
            if code:
                raise RuntimeError("component replay failed: " + component)

    await asyncio.gather(*(replay(c) for c in args.components))


if __name__ == "__main__":
    asyncio.run(main())
