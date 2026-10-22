"""Compile an explicit reviewed source batch; never infer missing ownership.

Inspection exports the actual source compiler's complete text, target and owners.
Publication consumes authored, exact-quote amendments, validates their current
source pins and preserves every unamended catalog entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates, _digest
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign


def reviewed_source_ids(config: dict, review: dict, expected_sources: int) -> list[str]:
    """Resolve the explicitly sized batch, including the final registry remainder."""
    if not 1 <= expected_sources <= 20:
        raise ValueError("a reviewed batch must contain between one and twenty sources")
    ids = []
    for entry in review["sources"]:
        matches = [sid for sid in config["source_ids"] if sid[4:12] == entry["prefix"]]
        if len(matches) != 1:
            raise ValueError("source prefix must identify exactly one admitted source")
        ids.append(matches[0])
    if len(ids) != expected_sources or len(set(ids)) != expected_sources:
        raise ValueError(f"this reviewed batch must have exactly {expected_sources} unique sources")
    return ids


def reviewed_span(source: str, declaration: dict) -> dict:
    """Resolve an authored exact quote, requiring ownership of its occurrence."""
    quote = declaration["source_text"]
    raw = source.encode()
    offsets = []
    cursor = 0
    while (start := raw.find(quote.encode(), cursor)) >= 0:
        offsets.append(start)
        cursor = start + 1
    occurrence = declaration.get("occurrence")
    if occurrence is None and len(offsets) != 1:
        raise ValueError("repeated quote requires explicit occurrence: " + quote)
    index = 0 if occurrence is None else occurrence - 1
    if not 0 <= index < len(offsets):
        raise ValueError("reviewed occurrence missing: " + quote)
    region = {k: v for k, v in declaration.items() if k != "occurrence"}
    region.update(byte_start=offsets[index], byte_end=offsets[index] + len(quote.encode()))
    return region


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-sources", type=int, default=20)
    args = parser.parse_args()
    root = Path.cwd()
    config = json.loads(args.config.read_text())
    review = json.loads(args.review.read_text())
    original = IndependentNumericCertificates.from_path(
        root / config["independent"]["numeric_certificates"]["path"],
        expected_sha256=config["independent"]["numeric_certificates"]["sha256"],
    )
    ids = reviewed_source_ids(config, review, args.expected_sources)
    config.update(
        source_ids=ids,
        capabilities={sid: config["capabilities"][sid] for sid in ids},
        output=review["campaignOutput"],
        seed=review["seed"],
        variants_per_source=3,
        budget_usd=review["budgetUsd"],
    )
    config["independent_preparation"] = review["preparation"]
    source = _SourceOwnershipCampaign(root, config)
    catalog = original.certificates()
    for sid, entry in zip(ids, review["sources"], strict=True):
        blueprint = source.blueprint(sid)
        folder = args.output / "inspection" / sid
        atomic_publish_bytes(folder / "source.txt", blueprint.source.encode())
        atomic_publish_json(folder / "target.json", blueprint.target)
        atomic_publish_json(
            folder / "owners.json",
            {
                "bindings": blueprint.historical_bindings,
                "ownership": blueprint.ownership_data,
                "regions": [
                    {
                        "key": r.key,
                        "start": r.start,
                        "end": r.end,
                        "source": r.source,
                        "kind": r.kind,
                        "targetPaths": r.target_paths,
                        "curatedKey": r.curated_key,
                    }
                    for r in blueprint.regions
                ],
            },
        )
        if not args.publish:
            continue
        if not entry.get("completeRead") or not entry.get("decision"):
            raise ValueError("authored complete-source review required: " + sid)
        if "regions" not in entry:
            original.has_certificate(sid)
            continue
        regions = []
        for declaration in entry["regions"]:
            regions.append(reviewed_span(blueprint.source, declaration))
        patch = blueprint.target["documentPatch"]
        catalog[sid] = {
            "reviewStatus": "reviewed",
            "sourceSha256": _digest(blueprint.source.encode()),
            "sourceContractSha256": _digest(blueprint.contract.model_dump(mode="json")),
            "sourcePublicTargetSha256": _digest(blueprint.target),
            "sourceContainerCount": len(patch.get("containerInformation", [])),
            "sourceAllocationPresence": [
                "packageQuantity" in row
                for goods in patch.get("goodsItemDetails", [])
                for row in goods.get("splitGoodsPlacement", [])
            ],
            "regions": regions,
            "wordingOwners": [
                reviewed_span(blueprint.source, owner) for owner in entry.get("wordingOwners", [])
            ],
            "constraints": entry.get("constraints", []),
            "holdReasons": [],
            "reviewEvidence": {"path": str(args.review), "sha256": sha256_file(args.review)},
            "decision": entry["decision"],
        }
        if "cargoPresentation" in entry:
            catalog[sid]["cargoPresentation"] = entry["cargoPresentation"]
    if args.publish:
        authority = args.output / "numeric_certificates.json"
        atomic_publish_json(
            authority, {"version": 2, "selectedSources": sorted(catalog), "sources": catalog}
        )
        verifier = IndependentNumericCertificates.from_path(authority)
        for sid in ids:
            verifier.apply_blueprint(source.blueprint(sid))
        config["independent"]["numeric_certificates"] = {
            "path": str(authority),
            "sha256": sha256_file(authority),
        }
        target = args.output / "config.json"
        atomic_publish_json(target, config)
        print(
            json.dumps(
                {
                    "config": str(target),
                    "sources": len(ids),
                    "plannedSlots": len(ids) * config["variants_per_source"],
                    "numericCatalogSources": len(catalog),
                }
            )
        )
    else:
        print(json.dumps({"inspectedSources": len(ids), "path": str(args.output / "inspection")}))


if __name__ == "__main__":
    main()
