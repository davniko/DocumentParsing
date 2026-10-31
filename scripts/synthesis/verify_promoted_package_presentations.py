"""Replay accepted draws with promoted authorities, without fresh paid calls."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_layout import PositionPolicy
from document_ocr.synthesis.curated_positions import load_page_geometry, transfer_positions
from document_ocr.synthesis.curated_reflow import ReflowEngine
from document_ocr.synthesis.package_presentations import (
    PackagePresentationPolicy,
    PackagePresentations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    production = json.loads(args.config.read_text())
    acceptance = json.loads(args.acceptance.read_text())
    assert sha256_file(Path(acceptance["replay"]["path"])) == acceptance["replay"]["sha256"]
    replay = json.loads(Path(acceptance["replay"]["path"]).read_text())
    ids = sorted(production["source_ids"])
    assert ids == sorted(r["sourceId"] for r in acceptance["records"])
    engine = ReflowEngine(root, PositionPolicy.model_validate(production["positions"]).reflow)
    presenter = PackagePresentations(
        root,
        PackagePresentationPolicy.model_validate(production["aliases"]["presentations"]),
        engine.metrics,
    )
    presenter.require_certified(ids)
    dataset = root / production["dataset"]
    alignments = {
        sid: json.loads((dataset / "alignments" / f"{sid}.json").read_text()) for sid in ids
    }
    geometry, provenance = load_page_geometry(root, dataset, alignments)
    groups = defaultdict(list)
    for row in replay["rows"]:
        groups[row["preparedConfig"]].append(row)
    results = []
    for path, rows in sorted(groups.items()):
        config = json.loads(Path(path).read_text())
        campaign = create_campaign(root, config)
        campaign.package_presentations = presenter
        for row in rows:
            candidate = json.loads(Path(row["candidate"]).read_text())
            sid = row["sourceId"]
            expected = copy.deepcopy(candidate)
            for receipt in (
                expected["packagePresentations"],
                expected["proof"]["packagePresentations"],
            ):
                receipt["certificatesSha256"] = presenter.policy.certificates.sha256
            actual = campaign.render(sid, candidate["variant"], persist=False)
            if actual != expected:
                changed = [
                    key
                    for key in actual.keys() | expected.keys()
                    if actual.get(key) != expected.get(key)
                ]
                raise ValueError(f"promotion changed accepted output for {sid}: {changed}")
            bp = campaign.blueprint(sid)
            _, receipt = transfer_positions(
                bp.source,
                actual["joinedRawText"],
                actual["proof"],
                alignments[sid],
                page_boxes=geometry[sid],
                page_dimensions=provenance["pageDimensions"][sid],
            )
            positioned, _, _ = engine.apply(
                source=bp.source,
                rendered=actual["joinedRawText"],
                proof=actual["proof"],
                alignment=alignments[sid],
                receipt=receipt,
                boxes=geometry[sid],
                dimensions=provenance["pageDimensions"][sid],
                sample_id=actual["documentId"],
            )
            previous = json.loads(Path(row["positions"]).read_text())
            if positioned != previous["positionedText"]:
                raise ValueError("promotion changed positioned text: " + sid)
            results.append({"sourceId": sid, "status": "PASS"})
        print(
            json.dumps({"campaign": path, "passed": len(rows), "total": len(results)}), flush=True
        )
    atomic_publish_json(
        args.output,
        {
            "status": "PASS",
            "sources": len(results),
            "results": results,
            "configSha256": sha256_file(args.config),
            "acceptanceSha256": sha256_file(args.acceptance),
            "onlyDifference": "promoted certificate file hash in two provenance receipts",
            "textTargetsCoordinatesUnchanged": True,
            "paidCalls": 0,
        },
    )


if __name__ == "__main__":
    main()
