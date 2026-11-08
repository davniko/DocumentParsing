"""Bounded fresh-shipment probe; never publish training data or approve wording."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.independent_preparation import prepare_campaign
from document_ocr.synthesis.paddle_campaign import run as native_run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/synthesis-probes/r30-posttraining-20261108"
CATALOG = (
    ROOT
    / "artifacts/synthesis-templates/mpci-bl-v7-paddle-reviewed/r30"
    / "ownership-585-20261105/catalog.json"
)


async def main(action):
    started = time.perf_counter()
    pin = {"path": str(CATALOG.relative_to(ROOT)), "sha256": sha256_file(CATALOG)}
    if action == "generate":
        config = json.loads((ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json").read_text())
        prefixes = [
            "00388da5",
            "00a165a2",
            "0951955d",
            "1444835f",
            "0b7f3731",
            "1149521e",
            "0ede63c2",
            "bef02d48",
        ]
        selected = [next(s for s in config["source_ids"] if s[4:12] == p) for p in prefixes]
        config.update(
            source_ids=selected,
            variants_per_source=1,
            seed=2026110817,
            budget_usd=3,
            concurrency=4,
            output=str((OUT / "sampling").relative_to(ROOT)),
        )
        config["capabilities"] = {s: config["capabilities"][s] for s in selected}
        config["declared_packages"]["source_group_counts"] = {
            s: [2 if s[4:12] in {"00a165a2", "0951955d"} else 1] for s in selected
        }
        atomic_write_json(OUT / "requested-config.json", config)
        prepared = await prepare_campaign(ROOT, config, **config["independent_preparation"])
        atomic_write_json(OUT / "preparation.json", prepared)
        print(
            json.dumps(
                {
                    "stage": "preparation",
                    "ready": prepared["readySlots"],
                    "planned": prepared["plannedSlots"],
                    "counts": prepared["counts"],
                }
            ),
            flush=True,
        )
        if prepared["preparedConfig"] is None:
            raise ValueError("no ready fresh shipments; inspect preparation receipts")
        path = ROOT / prepared["preparedConfig"]["path"]
        campaign = create_campaign(ROOT, json.loads(path.read_text()))
        generation = await campaign.generate_all()
        print(
            json.dumps(
                {
                    "stage": "generation",
                    "errors": generation["errors"],
                    "cost": generation["costUsd"],
                }
            ),
            flush=True,
        )
        postal = await campaign.correct_postal_all()
        rendered = campaign.render_all()
        atomic_write_json(OUT / "legacy-render.json", rendered)
        print(
            json.dumps(
                {
                    "stage": "render",
                    "accepted": len(rendered["accepted"]),
                    "errors": rendered["errors"],
                }
            ),
            flush=True,
        )
        atomic_write_json(
            OUT / "generation.json",
            {"generation": generation, "postal": postal, "seconds": time.perf_counter() - started},
        )
    elif action in {"propose-wording", "render"}:
        prepared = json.loads((OUT / "preparation.json").read_text())
        config = json.loads((ROOT / prepared["preparedConfig"]["path"]).read_text())
        candidates = sorted((ROOT / config["output"] / "candidates").glob("*.json"))
        result = await native_run(
            ROOT,
            pin,
            candidates,
            OUT / ("wording" if action == "propose-wording" else "native"),
            action=action,
            wording_directory=OUT / "wording-reviewed",
        )
        print(
            json.dumps(
                {
                    "stage": action,
                    "count": len(result["results"]),
                    "failures": result["failures"],
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    else:
        raise ValueError(action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "propose-wording", "render"])
    asyncio.run(main(parser.parse_args().action))
