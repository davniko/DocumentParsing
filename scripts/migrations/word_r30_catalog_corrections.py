"""Propose and explicitly admit missing linguistic owners for corrected controls/rows."""

import argparse
import asyncio

from admit_r30_missing_wording import reviewed_proposal
from migrate_r30_synthetic import ART, DATA, ROOT, read
from publish_r30_sources import pin, verify_pin
from rebase_r30_sources import ART as REBASE_ART
from rebase_r30_sources import PILOT_ART

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.paddle_wording import propose


async def prepare(catalog_path, prefix, destination):
    catalog = PaddleTemplateCatalog(ROOT, pin(catalog_path))
    (source,) = [sid for sid in catalog.entries if sid[4:12] == prefix]
    _, _, transfer, _ = catalog._template(source)
    control_root = (
        PILOT_ART
        if catalog.entries[source]["privacy"]["kind"] == "reviewed_replacement_spans"
        else REBASE_ART
    ) / "new-wording"
    jobs = {
        s["id"]: {
            "candidate": pin(ROOT / s["path"]),
            "outputs": [str((control_root / (s["id"] + ".json")).relative_to(ROOT))],
        }
        for s in transfer["descendants"]
    }
    manifest = read(DATA / "manifest.json")
    import json

    for cert_pin in manifest["syntheticBatches"]:
        cert = read(verify_pin(cert_pin))
        for row in map(json.loads, verify_pin(cert["records"]).open()):
            if row["sourceDocumentId"] != source:
                continue
            if row["evidence"]["wording"] is not None:
                raise ValueError(
                    "existing linguistic owners require a separately reviewed replacement"
                )
            job = jobs.setdefault(
                row["documentId"], {"candidate": row["evidence"]["candidate"], "outputs": []}
            )
            if job["candidate"] != row["evidence"]["candidate"]:
                raise ValueError("control and published shipment contexts differ")
            job["outputs"].append(
                str((ART / "wording" / (row["documentId"] + ".json")).relative_to(ROOT))
            )
    cfg = read(ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json")
    cfg = {k: cfg[k] for k in CuratedSynthesisConfig.model_fields}
    cfg.update(
        output=str((destination / "provider").relative_to(ROOT)), concurrency=4, budget_usd=1
    )
    runner = Runner(ROOT, cfg)

    async def one(identity, job):
        spec = catalog.wording_specification(verify_pin(job["candidate"]))
        if spec is None or spec["documentId"] != identity:
            raise ValueError("selected correction has no matching wording owner")
        path = destination / "proposals" / (identity + ".json")
        if path.exists():
            result = read(path)
            if result["candidateSha256"] != spec["candidateSha256"]:
                raise ValueError("existing proposal context differs")
        else:
            result = await propose(runner, spec)
            atomic_publish_json(path, result)
        job.update(specification=spec, proposal=pin(path))
        print(json.dumps({"id": identity, "values": result["values"]}), flush=True)

    await asyncio.gather(*(one(identity, job) for identity, job in jobs.items()))
    atomic_publish_json(
        destination / "proposals.json",
        {
            "status": "PROPOSED_REQUIRES_MAIN_AGENT_READBACK",
            "catalog": pin(catalog_path),
            "source": source,
            "jobs": jobs,
            "allAttemptCostUsd": str(runner.spent),
        },
    )


def admit(proposals_path, review_path, catalog_path=None):
    proposals, review = read(proposals_path), read(review_path)
    catalog_pin = pin(catalog_path) if catalog_path else proposals["catalog"]
    catalog = PaddleTemplateCatalog(ROOT, catalog_pin)
    if set(review["decisions"]) != set(proposals["jobs"]):
        raise ValueError("review does not cover the exact proposed cohort")
    outputs = {}
    for identity, job in proposals["jobs"].items():
        if catalog.wording_specification(verify_pin(job["candidate"])) != job["specification"]:
            raise ValueError("reviewed wording specification changed in the corrected catalog")
        result = reviewed_proposal(
            job["specification"], read(verify_pin(job["proposal"])), review["decisions"][identity]
        )
        result["review"] = {
            "proposal": job["proposal"],
            "decision": pin(review_path),
            "specifications": pin(proposals_path),
        }
        for relative in job["outputs"]:
            path = ROOT / relative
            atomic_publish_json(path, result)
        outputs[identity] = pin(ROOT / job["outputs"][-1])
    atomic_publish_json(
        proposals_path.parent / ("reviewed-wording-" + catalog_pin["sha256"] + ".json"),
        {
            "status": "MAIN_AGENT_REVIEWED_NEW_LINGUISTIC_OWNERS",
            "review": pin(review_path),
            "catalog": catalog_pin,
            "wording": outputs,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("propose")
    p.add_argument("catalog")
    p.add_argument("source")
    p.add_argument("destination")
    p = commands.add_parser("admit")
    p.add_argument("proposals")
    p.add_argument("review")
    p.add_argument("--catalog")
    args = parser.parse_args()
    if args.command == "propose":
        asyncio.run(prepare(ROOT / args.catalog, args.source, ROOT / args.destination))
    else:
        admit(
            ROOT / args.proposals, ROOT / args.review, ROOT / args.catalog if args.catalog else None
        )
