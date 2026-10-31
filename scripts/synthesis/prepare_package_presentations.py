"""Compile explicit package-alias vocabulary and inspectable source certificates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_reflow_policy import FontMetrics
from document_ocr.synthesis.independent_campaign import _complete_optional_package_suffix
from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign
from document_ocr.synthesis.package_presentations import CATEGORY, OPTIONAL, PackageVocabulary

FAMILIES = {
    "PACKAGE_BAG_JUMBO": "PACKAGE_BAG",
    "PACKAGE_BAG_PAPER": "PACKAGE_BAG",
    "PACKAGE_BAG_PAPER_MULTI_WALL": "PACKAGE_BAG",
    "PACKAGE_BAG_PLASTIC": "PACKAGE_BAG",
    "PACKAGE_BAG_SUPER_BULK": "PACKAGE_BAG",
    "PACKAGE_BALE_NON_COMPRESSED": "PACKAGE_BALE",
    "PACKAGE_BOX_FIBREBOARD": "PACKAGE_BOX",
    "PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY": "PACKAGE_BOX",
    "PACKAGE_CASE_WOODEN": "PACKAGE_CASE",
    "PACKAGE_CRATE_WOODEN": "PACKAGE_CRATE",
    "PACKAGE_DRUM_FIBRE": "PACKAGE_DRUM",
    "PACKAGE_DRUM_PLASTIC": "PACKAGE_DRUM",
    "PACKAGE_DRUM_STEEL": "PACKAGE_DRUM",
    "PACKAGE_PALLET_WOODEN": "PACKAGE_PALLET",
}
# Reviewed source details finer than the current public category, not aliases
# for every member of that category. These remain intact for the same category.
PROTECTED = {
    "TOTE BIN",
    "CARTONS PALLETIZED",
    "DRUM, PLASTIC, REMOVABLE HEAD",
    "HDPE BAGS",
    "HMHDPE DRUMS",
    "HDPE\nBAGS",
    "Woven plastics bags, water-resistant-5H3",
    "PALLETS SLAC",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-certificates", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    staged = (
        root / "docs/analysis/r28-package-alias-followup-20261023/alias-registry-staged-safe.json"
    )
    research = json.loads(staged.read_text())
    entries = {}
    for item in research["currentEntries"]:
        category = item["category"]
        short = list(
            dict.fromkeys(
                s[k] for s in item["commercialCandidates"] for k in ("singular", "plural")
            )
        )
        forms = item["naturalAndOptionalForms"]
        if category == "PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY":
            forms = ["WOODEN BOX", "WOODEN BOXES", "WOODEN BOX(ES)"]
        entries[category] = {
            "full": [s for s in forms if not OPTIONAL.search(s) and s not in short],
            "optional": [s for s in forms if OPTIONAL.search(s)],
            "short": short,
            "family": FAMILIES.get(category, category),
        }
    vocabulary = {
        "version": 1,
        "entries": entries,
        "excludedUnscopedTokens": research["excludedUnscopedTokens"],
        "provenance": {"path": str(staged.relative_to(root)), "sha256": sha256_file(staged)},
    }
    PackageVocabulary(vocabulary)
    atomic_publish_json(args.output / "vocabulary.json", vocabulary)
    inventory = json.loads((args.inspection / "inventory.json").read_text())
    config = inventory["config"]
    campaign = _SourceOwnershipCampaign(root, config)
    pins = [
        config["independent"]["numeric_certificates"],
        {"path": "docs/analysis/r28-multidg-integration-20261029/numeric_certificates.json"},
    ]
    if args.numeric_certificates:
        pins = [
            {
                "path": str(args.numeric_certificates),
                "sha256": sha256_file(args.numeric_certificates),
            }
        ]
    numeric = {}
    for pin in pins:
        c = IndependentNumericCertificates.from_path(
            root / pin["path"], expected_sha256=pin.get("sha256")
        )
        numeric.update(dict.fromkeys(c.certificates(), c))
    font = config["positions"]["reflow"]
    metrics = FontMetrics(root / font["font_path"], font["font_sha256"])
    certificates = {}
    for row in inventory["sources"]:
        sid = row["sourceId"]
        bp = _complete_optional_package_suffix(
            numeric[sid].apply_blueprint(campaign.blueprint(sid), sid)
        )
        alignment = json.loads(
            (root / config["dataset"] / "alignments" / f"{sid}.json").read_text()
        )
        lines = {line["line_number"]: line for line in alignment["lines"]}
        slots = []
        for region in bp.regions:
            paths = [p for p in region.target_paths if CATEGORY.fullmatch(p)]
            if not paths:
                continue
            mode = (
                "compound"
                if region.key.startswith("held-numeric:")
                else ("implicit" if region.curated_key else "scalar")
            )
            if mode == "scalar" and region.target_paths != tuple(paths):
                raise ValueError("unreviewed scalar composite: " + sid)
            first = bp.source.encode()[: region.start].count(b"\n") + 1
            last = bp.source.encode()[: region.end - 1].count(b"\n") + 1
            neighbours = [
                r
                for r in bp.regions
                if r.end <= region.start and not bp.source.encode()[r.end : region.start].strip()
            ]
            adjacent = max(neighbours, key=lambda r: r.end) if neighbours else None
            paths_before = () if adjacent is None else adjacent.target_paths
            role = (
                "allocation"
                if any(".splitGoodsPlacement[" in p for p in paths_before)
                else (
                    "declaration"
                    if any(
                        ".numberAndTypeOfPackages[" in p and p.endswith("packageQuantity")
                        for p in paths_before
                    )
                    else "repetition"
                )
            )
            geometry = {"status": "unmeasured", "reason": "no_exact_unique_source_region"}
            if (
                mode == "scalar"
                and first == last
                and first in lines
                and lines[first]["xy"] is not None
            ):
                pattern = r"\s*".join(re.escape(c) for c in region.source if not c.isspace())
                matches = [
                    (r, m)
                    for r in lines[first]["regions"]
                    for m in re.finditer(pattern, r["text"], re.I)
                ]
                if len(matches) == 1:
                    r, match = matches[0]
                    geometry = {
                        "status": "measured",
                        "page": lines[first]["page_index"],
                        "region": r["index"],
                        "sourceLine": first,
                        "bbox": r["bbox"],
                        "regionText": r["text"],
                        "fragmentStart": match.start(),
                        "fragmentEnd": match.end(),
                        "maximumAdvance": metrics.length(region.source),
                        "alignment": "synthetic_left_anchor",
                    }
            slots.append(
                {
                    "key": region.key,
                    "occurrence": region.occurrence,
                    "start": region.start,
                    "end": region.end,
                    "source": region.source,
                    "categoryPaths": paths,
                    "mode": mode,
                    "role": role,
                    "protectedSourceDetail": region.source in PROTECTED,
                    "geometry": geometry,
                }
            )
        certificates[sid] = {
            "sourceSha256": digest(bp.source.encode()),
            "sourceTargetSha256": digest(bp.target),
            "alignmentSha256": digest(alignment),
            "slots": slots,
        }
    result = {
        "version": 1,
        "sources": certificates,
        "vocabularySha256": sha256_file(args.output / "vocabulary.json"),
        "registryManifestSha256": inventory["registryManifestSha256"],
        "status": "compiled_pending_batch_review_and_synthesis",
    }
    atomic_publish_json(args.output / "certificates.json", result)
    print(
        json.dumps(
            {
                "sources": len(certificates),
                "modes": dict(
                    Counter(s["mode"] for c in certificates.values() for s in c["slots"])
                ),
                "geometry": dict(
                    Counter(
                        s["geometry"]["status"]
                        for c in certificates.values()
                        for s in c["slots"]
                        if s["mode"] == "scalar"
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
