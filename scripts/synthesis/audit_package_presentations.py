"""Export package occurrence authority for the current registry, without mutation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import flat
from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    registry = root / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
    manifest = json.loads((registry / "manifest.json").read_text())
    config = json.loads(
        (registry / "independent-flat-full-registry-20261021/working-config.json").read_text()
    )
    config.update(
        dataset=manifest["dependencies"]["sourceDataset"]["path"],
        source_ids=sorted(manifest["cases"]),
        capabilities={s: c["capability"] for s, c in manifest["cases"].items()},
        output=str(args.output / "unused-campaign"),
    )
    config.pop("template_sampling", None)
    c = _SourceOwnershipCampaign(root, config)
    pins = [
        config["independent"]["numeric_certificates"],
        {"path": "docs/analysis/r28-multidg-integration-20261029/numeric_certificates.json"},
    ]
    catalogs = [
        IndependentNumericCertificates.from_path(root / p["path"], expected_sha256=p.get("sha256"))
        for p in pins
    ]
    by_source = {sid: catalog for catalog in catalogs for sid in catalog.certificates()}
    rows = []
    for sid in c.source_ids:
        original = c.blueprint(sid)
        numeric = by_source[sid]
        bp = numeric.apply_blueprint(original, sid)
        leaves = flat(bp.target)
        package_paths = {
            k: v
            for k, v in leaves.items()
            if ".numberAndTypeOfPackages[" in k and k.endswith("typeCategory")
        }
        regions = []
        raw = bp.source.encode()
        for r in bp.regions:
            if not set(r.target_paths) & package_paths.keys():
                continue
            first = raw.rfind(b"\n", 0, r.start) + 1
            last = raw.find(b"\n", r.end)
            if last < 0:
                last = len(raw)
            regions.append(
                {
                    **asdict(r),
                    "context": raw[first:last].decode(),
                    "recipe": bp.ownership_data.get("surfaces", {}).get(r.key),
                }
            )
        row = {
            "sourceId": sid,
            "categories": package_paths,
            "regions": regions,
            "numericRegions": numeric.certificate(sid)["regions"],
            "wordingOwners": numeric.certificate(sid)["wordingOwners"],
            "family": config["capabilities"][sid]["family"],
        }
        rows.append(row)
        atomic_publish_json(args.output / "sources" / f"{sid}.json", row)
    atomic_publish_json(
        args.output / "inventory.json",
        {
            "sources": rows,
            "registryManifestSha256": sha256_file(registry / "manifest.json"),
            "config": config,
        },
    )
    print(
        json.dumps(
            {
                "sources": len(rows),
                "packageOwners": sum(len(r["regions"]) for r in rows),
                "families": dict(Counter(r["family"] for r in rows)),
                "numericRoles": dict(Counter(n["role"] for r in rows for n in r["numericRegions"])),
                "withoutScalarNoun": [
                    r["sourceId"][4:12]
                    for r in rows
                    if not any(
                        set(n["target_paths"]) <= r["categories"].keys()
                        and n["curated_key"] is None
                        for n in r["regions"]
                    )
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
