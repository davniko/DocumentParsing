"""Measure incremental alias/reflow latency and allocations on source controls."""

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    inv = json.loads((args.work / "inspection/inventory.json").read_text())
    config = inv["config"]
    # A systematic spread across the sorted registry, including all 456 layouts
    # in the separate acceptance replay. This is a latency sample, not acceptance.
    ids = sorted(config["source_ids"])[::23]
    campaign = _SourceOwnershipCampaign(root, config)
    catalogs = [
        IndependentNumericCertificates.from_path(root / p)
        for p in (
            config["independent"]["numeric_certificates"]["path"],
            "docs/analysis/r28-multidg-integration-20261029/numeric_certificates.json",
        )
    ]
    numeric = {sid: cat for cat in catalogs for sid in cat.certificates()}
    blueprints = {
        sid: _complete_optional_package_suffix(
            numeric[sid].apply_blueprint(campaign.blueprint(sid), sid)
        )
        for sid in ids
    }
    dataset = root / config["dataset"]
    alignments = {
        sid: json.loads((dataset / "alignments" / f"{sid}.json").read_text()) for sid in ids
    }
    boxes, provenance = load_page_geometry(root, dataset, alignments)
    engine = ReflowEngine(root, PositionPolicy.model_validate(config["positions"]).reflow)
    pins = {
        name: {
            "path": str(args.work / "prepared-v2" / f"{name}.json"),
            "sha256": sha256_file(args.work / "prepared-v2" / f"{name}.json"),
        }
        for name in ("vocabulary", "certificates")
    }
    presenter = PackagePresentations(
        root, PackagePresentationPolicy.model_validate(pins), engine.metrics
    )

    def render(sid, aliases):
        bp = blueprints[sid]
        surfaces, receipt = (
            presenter.apply(bp, bp.target, {}, seed=20261030, variant=1) if aliases else ({}, None)
        )
        text, _, proof = render_sampling_blueprint(bp, bp.target, {}, surface_values=surfaces)
        if receipt is not None:
            proof["packagePresentations"] = receipt
        _, positioned = transfer_positions(
            bp.source,
            text,
            proof,
            alignments[sid],
            page_boxes=boxes[sid],
            page_dimensions=provenance["pageDimensions"][sid],
        )
        engine.apply(
            source=bp.source,
            rendered=text,
            proof=proof,
            alignment=alignments[sid],
            receipt=positioned,
            boxes=boxes[sid],
            dimensions=provenance["pageDimensions"][sid],
            sample_id=sid,
        )

    result = {
        "sources": ids,
        "rounds": 5,
        "scope": "warm text render plus position transfer and reflow; no network",
    }
    for mode in (False, True):
        for sid in ids:
            render(sid, mode)
        times = []
        for _ in range(5):
            for sid in ids:
                start = time.perf_counter()
                render(sid, mode)
                times.append(time.perf_counter() - start)
        tracemalloc.start()
        for sid in ids:
            render(sid, mode)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["aliases" if mode else "baseline"] = {
            "medianMs": statistics.median(times) * 1000,
            "meanMs": statistics.mean(times) * 1000,
            "documentsPerSecond": 1 / statistics.mean(times),
            "peakAllocatedBytes": peak,
        }
    atomic_publish_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
