"""Inventory draft batches and stage reviewed redraws and goods-correction plans."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"


def inventory(config_path: Path):
    config = json.loads(config_path.read_text())
    output = ROOT / config["output"]
    paths = sorted((output / "candidates").glob("*.json"))
    path = config_path.parent / "scope-candidates.json"
    if not path.exists():
        atomic_publish_json(path, [str(p.relative_to(ROOT)) for p in paths])
    result = {
        "config": str(config_path.relative_to(ROOT)),
        "candidates": len(paths),
        "findings": [],
        "records": [],
    }
    for p in sorted((output / "reviews").glob("*.json")):
        receipt = json.loads(p.read_text())
        for index, finding in enumerate(receipt["review"]["findings"]):
            result["findings"].append(
                {
                    "source": receipt["sourceDocumentId"],
                    "index": index,
                    "finding": finding,
                    "receipt": str(p.relative_to(ROOT)),
                }
            )
    for p in paths:
        row = json.loads(p.read_text())
        goods = row["target"]["documentPatch"]["goodsItemDetails"]
        result["records"].append(
            {
                "sampleId": row["documentId"],
                "source": row["sourceDocumentId"],
                "description": [g["description"] for g in goods],
                "packages": [g.get("numberAndTypeOfPackages", []) for g in goods],
                "thermal": row["scenario"]["provenance"].get("thermalCommodityContext"),
            }
        )
    return result


def redraw(config_path: Path, review_path: Path, attempt: int, authority_path: Path):
    config = json.loads(config_path.read_text())
    authority = json.loads(authority_path.read_text())
    if set(config["independent"]) != set(authority["independent"]):
        raise ValueError("redraw requires an unprepared campaign and compatible complete authority")
    review = json.loads(review_path.read_text())
    reasons = review["redrawSources"]
    if not reasons or not all(reasons.values()) or not reasons.keys() <= set(config["source_ids"]):
        raise ValueError("explicit reviewed reasons required for selected redraw sources")
    counts = {sid: config["template_sampling"]["source_counts"][sid] for sid in reasons}
    revised = copy.deepcopy(config)
    revised.update(
        source_ids=list(counts),
        capabilities={sid: config["capabilities"][sid] for sid in counts},
        seed=config["seed"] + attempt * 100000,
        template_sampling={"samples": sum(counts.values()), "source_counts": counts},
    )
    revised["declared_packages"]["source_group_counts"] = {
        sid: seq
        for sid, seq in config["declared_packages"]["source_group_counts"].items()
        if sid in counts
    }
    prep = revised["independent_preparation"]
    prep["mixed_source_ids"] = [s for s in prep["mixed_source_ids"] if s in counts]
    prep["thermal_goods_modes"] = {
        s: v for s, v in prep.get("thermal_goods_modes", {}).items() if s in counts
    }
    if "equipment_inventory" in revised["sampling"]:
        inv = revised["sampling"]["equipment_inventory"]
        inv["source_ids"] = [s for s in inv["source_ids"] if s in counts]
        if not inv["source_ids"]:
            revised["sampling"].pop("equipment_inventory")
    # A target-boundary repair changes several linked authorities. Carry the
    # complete reviewed source bundle, not merely its two owner overlays.
    for key in ("dataset", "source_contracts", "ownership", "auxiliary", "independent"):
        revised[key] = copy.deepcopy(authority[key])
    revised["declared_packages"]["contracts"] = copy.deepcopy(authority["declared_packages"]["contracts"])
    revised["aliases"]["presentations"]["certificates"] = copy.deepcopy(
        authority["aliases"]["presentations"]["certificates"]
    )
    path = config_path.parent / f"redraw-{attempt}" / "config.json"
    revised["output"] = str((path.parent / "campaign").relative_to(ROOT))
    atomic_publish_json(path, revised)
    atomic_publish_json(
        path.parent / "reviewed-reasons.json",
        {
            "review": str(review_path.relative_to(ROOT)),
            "reviewSha256": sha256_file(review_path),
            "parentConfig": str(config_path.relative_to(ROOT)),
            "parentSha256": sha256_file(config_path),
            "sourceAuthority": {"path": str(authority_path.relative_to(ROOT)), "sha256": sha256_file(authority_path)},
            "reasons": reasons,
            "status": "NEW_DRAW_REQUIRES_FULL_ACCEPTANCE",
        },
    )
    print(path.relative_to(ROOT))


def goods_plan(config_path, review_path):
    from document_ocr.synthesis.curated_campaign import create_campaign

    config = json.loads(config_path.read_text())
    campaign = create_campaign(ROOT, config)
    decisions = json.loads(review_path.read_text())["goodsCorrections"]
    corrections = []
    for sample, reason in decisions.items():
        candidate = json.loads((campaign.output / "candidates" / (sample + ".json")).read_text())
        sid, variant = candidate["sourceDocumentId"], candidate["variant"]
        bp, _, expected, _ = campaign.plan(sid, variant)
        assert expected == sample
        keys = [v.key for v in bp.contract.variables if v.kind == "product"]
        if not keys or not reason:
            raise ValueError("goods corrections require explicit product owners and reasons")
        corrections.append(
            {
                "sourceDocumentId": sid,
                "variant": variant,
                "sampleId": sample,
                "keys": keys,
                "wordingReceiptSha256": sha256_file(
                    campaign.output / "wording" / (sample + ".json")
                ),
                "reason": reason,
                "auditEvidence": {
                    "path": str(review_path.relative_to(ROOT)),
                    "sha256": sha256_file(review_path),
                },
            }
        )
    path = review_path.with_name(review_path.stem + "-goods-plan.json")
    atomic_publish_json(path, {"status": "approved_for_regeneration", "corrections": corrections})
    print(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "redraw", "goods-plan"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--authority", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        print(json.dumps(inventory(args.config.resolve()), ensure_ascii=False))
    elif args.command == "redraw":
        if args.review is None or args.authority is None or args.attempt < 1:
            parser.error("redraw requires review, complete source authority and positive attempt")
        redraw(args.config.resolve(), args.review.resolve(), args.attempt, args.authority.resolve())
    else:
        if args.review is None:
            parser.error("goods-plan requires reviewed decisions")
        goods_plan(args.config.resolve(), args.review.resolve())


if __name__ == "__main__":
    main()
