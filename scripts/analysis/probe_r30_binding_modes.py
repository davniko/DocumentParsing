"""Fresh, bounded single/multi-package mode controls for reviewed native owners."""

import argparse
import asyncio
import json
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.independent_preparation import prepare_campaign
from document_ocr.synthesis.paddle_campaign import run as native_run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/synthesis-probes/r30-binding-repair-20261108"


async def generate(mode, attempt=0, prefixes=None):
    directory = OUT / (f"fresh-{mode}" + (f"-retry-{attempt}" if attempt else ""))
    cfg = json.loads((ROOT / "configs/synthesis/mpci_bl_curated_v7_paddle.json").read_text())
    inv = json.loads((OUT / "inventory.json").read_text())["rows"]
    ids = sorted({r["source"] for r in inv if not r["singleOnly"]})
    if prefixes:
        ids = [s for s in cfg["source_ids"] if s[4:12] in prefixes]
    cfg.update(
        source_ids=ids,
        variants_per_source=1,
        seed=2026110860 + mode + attempt * 100,
        budget_usd=3,
        concurrency=6,
        output=str((directory / "sampling").relative_to(ROOT)),
    )
    cfg["capabilities"] = {s: cfg["capabilities"][s] for s in ids}
    cfg["declared_packages"]["source_group_counts"] = {s: [mode] for s in ids}
    atomic_write_json(directory / "requested-config.json", cfg)
    prepared = await prepare_campaign(ROOT, cfg, **cfg["independent_preparation"])
    atomic_write_json(directory / "preparation.json", prepared)
    print(
        json.dumps(
            {
                "ready": prepared["readySlots"],
                "planned": prepared["plannedSlots"],
                "counts": prepared["counts"],
            }
        ),
        flush=True,
    )
    if prepared["preparedConfig"] is None:
        raise ValueError("No prepared mode controls")
    cfg = json.loads((ROOT / prepared["preparedConfig"]["path"]).read_text())
    campaign = create_campaign(ROOT, cfg)
    generation = await campaign.generate_all()
    atomic_write_json(directory / "generation.json", generation)
    print(
        json.dumps(
            {"stage": "wording", "errors": generation["errors"], "cost": generation["costUsd"]}
        ),
        flush=True,
    )
    atomic_write_json(directory / "postal.json", await campaign.correct_postal_all())
    rendered = campaign.render_all()
    atomic_write_json(directory / "legacy-render.json", rendered)
    print(
        json.dumps(
            {"stage": "render", "accepted": len(rendered["accepted"]), "errors": rendered["errors"]}
        ),
        flush=True,
    )


async def native(mode, action, attempt=0):
    directory = OUT / (f"fresh-{mode}" + (f"-retry-{attempt}" if attempt else ""))
    catalog = OUT / "catalog-draft.json"
    candidates = sorted((directory / "sampling/candidates").glob("*.json"))
    result = await native_run(
        ROOT,
        {"path": str(catalog.relative_to(ROOT)), "sha256": sha256_file(catalog)},
        candidates,
        directory / ("native" if action == "render" else "native-wording"),
        action=action,
        wording_directory=directory / "native-wording-reviewed",
    )
    print(
        json.dumps({"results": len(result["results"]), "failures": result["failures"]}), flush=True
    )


def reviewed_two():
    """Explicit main-agent review of the enumerated first-round proposals."""
    suffixes = set(
        [
            "0ad100b0",
            "4aca3e52",
            "438f7861",
            "f38e97a4",
            "3a84f26f",
            "4672590b",
            "2b9557a0",
            "8ec0ee34",
            "56e671e0",
            "fc0d3fa1",
            "dbdf118b",
            "4998d564",
            "739d2cf1",
            "86fc72af",
            "858563e6",
            "20dfdf3b",
            "adc74f41",
            "83a2d460",
            "4c750c37",
            "7e0d1402",
            "c6e8416f",
            "8dcdd169",
            "b0388d36",
            "e381059d",
            "7661c193",
        ]
    )
    directory = OUT / "fresh-2"
    paths = list((directory / "native-wording").glob("syn*.json"))
    assert {p.stem[-8:] for p in paths} == suffixes
    for path in paths:
        r = json.loads(path.read_text())
        if path.stem.endswith("20dfdf3b"):
            assert r["values"]["signer"] == "Kp3wqn"
            r["values"]["signer"] = "Marcos"
        if path.stem.endswith("4aca3e52"):
            assert (
                r["values"]["deliveryAddress"]
                == "Portvej 8\nQasigiannguit\n3951 Qasigiannguit\nGreenland"
            )
            r["values"]["deliveryAddress"] = "Portvej 8\nUnit 4\n3951 Qasigiannguit\nGreenland"
        r["status"] = "REVIEWED"
        r["review"] = {
            "reviewer": "main-agent",
            "scope": (
                "Requested native-only linguistic owners against shipment context and source "
                "style; name and postal correction recorded, no target change inferred."
            ),
            "proposal": {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)},
        }
        atomic_write_json(directory / "native-wording-reviewed" / path.name, r)


def reviewed_one():
    suffixes = set(
        [
            "63cf7c4c",
            "59ec4fc1",
            "e0e44368",
            "9fdba9a8",
            "d5431db2",
            "5d8ba42d",
            "4e5316c9",
            "5f262f22",
            "a2368b08",
            "da66309e",
            "2428d11b",
            "37e42554",
            "d26a78fe",
            "fbd61ad4",
            "ecaee1fa",
            "38494ce2",
            "16b971ed",
            "83c49663",
            "78b06145",
            "bfb2d511",
            "0865344f",
            "953e0730",
            "75676bb3",
            "33e18644",
            "e141cc56",
            "103bbe23",
            "87097cd3",
        ]
    )
    directory = OUT / "fresh-1"
    paths = list((directory / "native-wording").glob("syn*.json"))
    assert {p.stem[-8:] for p in paths} == suffixes
    for path in paths:
        r = json.loads(path.read_text())
        if path.stem.endswith("78b06145"):
            assert r["values"]["signer"] == "Mp6khl"
            r["values"]["signer"] = "Ferraz"
        if path.stem.endswith("5d8ba42d"):
            assert r["values"]["stamp_company"] == "tHoR SvECOn"
            r["values"]["stamp_company"] = "bLuE MaRInE"
        r["status"] = "REVIEWED"
        r["review"] = {
            "reviewer": "main-agent",
            "scope": (
                "Explicit field-by-field native owner review; corrected unreadable signer "
                "and copied stamp identity. Both prompt recipes clarified in the template."
            ),
            "proposal": {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)},
        }
        atomic_write_json(directory / "native-wording-reviewed" / path.name, r)


def reviewed_retries():
    approved = {
        "fresh-1-retry-1": set(["1df5a3e7", "6687c4ed"]),
        "fresh-2-retry-1": set(["a4ee726b", "ffe2ec5c", "75ddd2c4", "da45d52b"]),
    }
    for name, suffixes in approved.items():
        d = OUT / name
        paths = list((d / "native-wording").glob("syn*.json"))
        assert {p.stem[-8:] for p in paths} == suffixes
        for p in paths:
            r = json.loads(p.read_text())
            r["status"] = "REVIEWED"
            r["review"] = {
                "reviewer": "main-agent",
                "scope": (
                    "Explicit readback of the six enumerated native-wording proposals; "
                    "appropriate owned names, contact, postal, marks or vessel field. "
                    "No modifications."
                ),
            }
            atomic_write_json(d / "native-wording-reviewed" / p.name, r)


def reviewed_final_wording():
    approved = {
        "fresh-1-retry-3": {
            "syn_full_v7_3b3647ac447f3fe27bcfa02c": {
                "issuer_name": "CORAL GATE MARINE LINES"
            }
        },
        "fresh-2-retry-3": {
            "syn_full_v7_36be4b943200230769f72fc4": {
                "signature_company": "Bluewave Forwardin",
                "stamp_company": "SeaLink Cargo",
            },
            "syn_full_v7_aabbeb59d5ead75a99adeb86": {
                "forwarder_address": "1205 OAKDALE INDUSTRIAL PARKWAY ROSWELL GA 30075",
                "forwarder_name": "ATLANTIC CROSS TRADE COMPANY INC",
            },
        },
    }
    for directory, records in approved.items():
        for identity, values in records.items():
            source = OUT / directory / "native-wording" / f"{identity}.json"
            receipt = json.loads(source.read_text())
            assert receipt["values"] == values
            receipt["status"] = "REVIEWED"
            receipt["review"] = {
                "reviewer": "main-agent",
                "scope": (
                    "Read all three native-only proposals against requested fields and "
                    "sampled parties/localities. Retained intentional signature clipping; "
                    "new stamp identity does not copy the original. No changes."
                ),
                "proposal": {"path": str(source.relative_to(ROOT)), "sha256": sha256_file(source)},
            }
            atomic_write_json(
                OUT / directory / "native-wording-reviewed" / source.name, receipt
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", type=int, choices=[1, 2, 3])
    p.add_argument(
        "--action",
        default="generate",
        choices=[
            "generate",
            "propose-wording",
            "render",
            "review-two",
            "review-one",
            "review-retries",
            "review-final-wording",
        ],
    )
    p.add_argument("--attempt", type=int, default=0)
    p.add_argument("--prefixes", nargs="*")
    a = p.parse_args()
    if a.action == "review-two":
        reviewed_two()
    elif a.action == "review-one":
        reviewed_one()
    elif a.action == "review-retries":
        reviewed_retries()
    elif a.action == "review-final-wording":
        reviewed_final_wording()
    else:
        asyncio.run(
            generate(a.mode, a.attempt, a.prefixes)
            if a.action == "generate"
            else native(a.mode, a.action, a.attempt)
        )
