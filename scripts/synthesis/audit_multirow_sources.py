"""Inventory exact package/count owners before multi-row source admission."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_campaign import _complete_optional_package_suffix
from document_ocr.synthesis.independent_preparation import _load_source_context
from document_ocr.synthesis.independent_wording import prepare_blueprint


def audit(root: Path, config: dict) -> dict:
    context = _load_source_context(root, config)
    records = []
    for sid in context.source.source_ids:
        bp = context.numeric.apply_blueprint(context.source.blueprint(sid), sid)
        if sid in context.wording:
            bp = prepare_blueprint(bp, context.wording[sid])
        bp = _complete_optional_package_suffix(bp)
        patch = bp.target["documentPatch"]
        goods = patch["goodsItemDetails"][0]
        certificate = context.numeric.certificate(sid)
        regions = []
        for r in bp.regions:
            if any("goodsItemDetails" in path for path in r.target_paths) or r.key.startswith(
                "held-numeric"
            ):
                regions.append(
                    {
                        "key": r.key,
                        "occurrence": r.occurrence,
                        "start": r.start,
                        "end": r.end,
                        "source": r.source,
                        "kind": r.kind,
                        "paths": list(r.target_paths),
                    }
                )
        raw = bp.source.encode()
        type_path = re.compile(r"\.numberAndTypeOfPackages\[\d+\]\.typeCategory$")
        count_path = re.compile(
            r"\.(?:numberAndTypeOfPackages|splitGoodsPlacement)\[\d+\]\.packageQuantity$"
        )
        types = [r for r in bp.regions if any(type_path.search(p) for p in r.target_paths)]
        counts = [r for r in bp.regions if any(count_path.search(p) for p in r.target_paths)]
        witnesses = []
        for noun in types:
            for number in counts:
                if number.end <= noun.start:
                    start, end, gap = number.start, noun.end, raw[number.end : noun.start]
                elif noun.end <= number.start:
                    start, end, gap = noun.start, number.end, raw[noun.end : number.start]
                else:
                    continue
                if gap.strip() or gap.count(b"\n") > 1:
                    continue
                witnesses.append(
                    {
                        "number": [number.key, number.occurrence],
                        "noun": [noun.key, noun.occurrence],
                        "start": start,
                        "end": end,
                        "source": raw[start:end].decode(),
                        "quantityPaths": list(number.target_paths),
                    }
                )
        records.append(
            {
                "sourceId": sid,
                "sourceSha256": digest(bp.source.encode()),
                "packages": goods["numberAndTypeOfPackages"],
                "containers": len(patch.get("containerInformation", [])),
                "quantityMask": [
                    "packageQuantity" in p for p in goods.get("splitGoodsPlacement", [])
                ],
                "regime": context.source.capabilities[sid].family,
                "privateRoles": dict(Counter(r["role"] for r in certificate["regions"])),
                "privateRegions": certificate["regions"],
                "cargoPresentation": certificate.get("cargoPresentation"),
                "constraints": bp.ownership_data["freshPrivatePacking"]["constraints"],
                "regions": regions,
                "adjacentDeclarations": witnesses,
            }
        )
    return {
        "status": "inventory_not_admission",
        "sources": len(records),
        "regimes": dict(Counter(r["regime"] for r in records)),
        "privateRoleSources": dict(Counter(k for r in records for k in r["privateRoles"])),
        "withAdjacentDeclaration": sum(bool(r["adjacentDeclarations"]) for r in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path.cwd(), json.loads(args.config.read_text()))
    result["config"] = {"path": str(args.config), "sha256": sha256_file(args.config)}
    atomic_publish_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
