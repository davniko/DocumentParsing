"""Present sampled shipment receipts through a pinned native Paddle catalog.

This is a presentation stage after ordinary donor-free shipment generation.
Wording proposals and final documents require review; neither is auto-accepted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog, read_pin
from document_ocr.synthesis.paddle_wording import propose


async def run(root, catalog_pin, candidates, output, *, action, wording_directory=None):
    catalog = PaddleTemplateCatalog(root, catalog_pin)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("select a nonempty distinct set of shipment receipts")
    ids = [catalog.shipment(p)[3]["documentId"] for p in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate shipment identity")
    # Validate the whole selection before spending or publishing outputs.
    for identity in ids:
        if (output / (identity + ".json")).exists():
            raise ValueError("output exists; choose a new reviewed attempt directory")
    runner = None
    if action == "propose-wording":
        cfg = read_pin(root, catalog.manifest["samplingConfig"])
        cfg = {k: cfg[k] for k in CuratedSynthesisConfig.model_fields}
        cfg["output"] = str((output / "provider").relative_to(root))
        runner = Runner(root, cfg)
    failures, results = [], []

    async def one(path, identity):
        try:
            spec = catalog.wording_specification(path)
            if action == "propose-wording":
                if spec is None:
                    results.append({"documentId": identity, "status": "NO_ADDITIONAL_WORDING"})
                    return
                result = await propose(runner, spec)
            elif action == "render":
                wording = None
                if spec is not None:
                    if wording_directory is None:
                        raise ValueError("reviewed additional wording is required")
                    wording = json.loads((wording_directory / (identity + ".json")).read_text())
                result = catalog.render(path, wording_receipt=wording)
            else:
                raise ValueError("unknown native campaign action")
            destination = output / (identity + ".json")
            atomic_publish_json(destination, result)
            results.append(
                {
                    "documentId": identity,
                    "path": str(destination.relative_to(root)),
                    "sha256": sha256_file(destination),
                    "status": result["status"],
                }
            )
        except Exception as exc:
            failures.append(
                {"documentId": identity, "error": str(exc), "errorType": type(exc).__name__}
            )

    # Rendering is sequential. Provider calls retain the existing configured
    # concurrency without launching one live task per registry member.
    width = cfg["concurrency"] if runner else 1
    for start in range(0, len(candidates), width):
        await asyncio.gather(
            *(
                one(p, identity)
                for p, identity in zip(
                    candidates[start : start + width], ids[start : start + width], strict=True
                )
            )
        )
    receipt = {
        "catalog": catalog_pin,
        "action": action,
        "results": results,
        "failures": failures,
        "allAttemptCostUsd": str(runner.spent) if runner else "0",
    }
    atomic_publish_json(output / "campaign.json", receipt)
    if failures:
        raise ValueError(f"{len(failures)} native candidates held; see {output / 'campaign.json'}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("propose-wording", "render"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wording-directory", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    asyncio.run(
        run(
            root,
            {"path": args.catalog, "sha256": args.catalog_sha256},
            [(root / p).resolve() for p in args.candidates],
            (root / args.output).resolve(),
            action=args.action,
            wording_directory=(root / args.wording_directory).resolve()
            if args.wording_directory
            else None,
        )
    )


if __name__ == "__main__":
    main()
