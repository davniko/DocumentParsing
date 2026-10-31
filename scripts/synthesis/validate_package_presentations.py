"""Batch-controlled source replay for package alias text and positioned geometry.

This is deterministic certification evidence, not a claim of fresh LLM synthesis.
The separate native campaign must pass before a batch is accepted.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import flat
from document_ocr.synthesis.curated_handling import validate_rendered_handling
from document_ocr.synthesis.curated_layout import PositionPolicy
from document_ocr.synthesis.curated_positions import load_page_geometry, transfer_positions
from document_ocr.synthesis.curated_reflow import ReflowEngine
from document_ocr.synthesis.curated_templates import render_sampling_blueprint
from document_ocr.synthesis.independent_campaign import _complete_optional_package_suffix
from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign
from document_ocr.synthesis.package_presentations import (
    PackagePresentationPolicy,
    PackagePresentations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--run", required=True, help="Immutable evidence namespace")
    parser.add_argument("--prepared", default="prepared")
    parser.add_argument("--numeric-certificates", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    inspection = json.loads((args.work / "inspection/inventory.json").read_text())
    config = inspection["config"]
    ids = sorted(config["source_ids"])[args.start : args.start + args.count]
    campaign = _SourceOwnershipCampaign(root, config)
    catalogs = [
        IndependentNumericCertificates.from_path(root / path)
        for path in (
            config["independent"]["numeric_certificates"]["path"],
            "docs/analysis/r28-multidg-integration-20261029/numeric_certificates.json",
        )
    ]
    if args.numeric_certificates:
        catalogs = [IndependentNumericCertificates.from_path(args.numeric_certificates)]
    numeric = {sid: catalog for catalog in catalogs for sid in catalog.certificates()}
    dataset = root / config["dataset"]
    alignments = {
        sid: json.loads((dataset / "alignments" / f"{sid}.json").read_text()) for sid in ids
    }
    geometry, provenance = load_page_geometry(root, dataset, alignments)
    engine = ReflowEngine(root, PositionPolicy.model_validate(config["positions"]).reflow)
    pins = {
        name: {
            "path": str(args.work / args.prepared / f"{name}.json"),
            "sha256": sha256_file(args.work / args.prepared / f"{name}.json"),
        }
        for name in ("vocabulary", "certificates")
    }
    presenters = {
        style: PackagePresentations(
            root,
            PackagePresentationPolicy.model_validate(
                {
                    **pins,
                    "full_weight": 1 if style == "full" else 0.000001,
                    "short_weight": 1 if style == "short" else 0,
                    "optional_weight": 1 if style == "optional" else 0,
                }
            ),
            engine.metrics,
        )
        for style in ("full", "short", "optional")
    }
    results = []
    for sid in ids:
        bp = _complete_optional_package_suffix(
            numeric[sid].apply_blueprint(campaign.blueprint(sid), sid)
        )
        for style, presenter in presenters.items():
            start = time.perf_counter()
            try:
                surfaces, alias = presenter.apply(bp, bp.target, {}, seed=20261030, variant=1)
                text, target, proof = render_sampling_blueprint(
                    bp, bp.target, {}, surface_values=surfaces
                )
                validate_rendered_handling(bp, text, target, proof["edits"])
                a, b = flat(bp.target), flat(target)
                changes = {
                    key: {"before": a.get(key), "after": b.get(key)}
                    for key in a.keys() | b.keys()
                    if a.get(key) != b.get(key)
                }
                if any(not key.endswith(".description") for key in changes):
                    raise ValueError(f"alias changed non-description target: {changes}")
                proof["packagePresentations"] = alias
                _, receipt = transfer_positions(
                    bp.source,
                    text,
                    proof,
                    alignments[sid],
                    page_boxes=geometry[sid],
                    page_dimensions=provenance["pageDimensions"][sid],
                )
                positioned, receipt, _ = engine.apply(
                    source=bp.source,
                    rendered=text,
                    proof=proof,
                    alignment=alignments[sid],
                    receipt=receipt,
                    boxes=geometry[sid],
                    dimensions=provenance["pageDimensions"][sid],
                    sample_id=f"alias-{sid}-{style}",
                )
                item = {
                    "sourceId": sid,
                    "style": style,
                    "status": "PASS",
                    "joinedRawText": text,
                    "target": target,
                    "positionedText": positioned,
                    "proof": proof,
                    "geometry": receipt,
                    "targetChanges": changes,
                }
                path = args.work / "controls" / args.run / f"{sid}-{style}.json"
                atomic_publish_json(path, item)
                row = {
                    "sourceId": sid,
                    "style": style,
                    "status": "PASS",
                    "path": str(path),
                    "edits": len(proof["edits"]),
                    "sha256": sha256_file(path),
                    "coordinateChanges": sum(
                        (None if r["xy"] is None else tuple(r["xy"]))
                        != (None if r["transfer_xy"] is None else tuple(r["transfer_xy"]))
                        for r in receipt["lines"]
                    ),
                }
            except (ValueError, KeyError, AssertionError) as error:
                row = {"sourceId": sid, "style": style, "status": "HELD", "error": str(error)}
            row["seconds"] = time.perf_counter() - start
            results.append(row)
            print(json.dumps(row), flush=True)
    atomic_publish_json(
        args.work / "controls" / args.run / f"batch-{args.start:03d}.json",
        {
            "results": results,
            "counts": dict(Counter(row["status"] for row in results)),
            "nativeSynthesis": "required_separately",
        },
    )


if __name__ == "__main__":
    main()
