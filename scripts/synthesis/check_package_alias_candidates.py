"""Replay native alias candidates through production text and positioning engines.

Review findings and fresh redraws remain separate; this does not publish data or
mark a source accepted merely because its deterministic checks pass.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_layout import PositionPolicy
from document_ocr.synthesis.curated_positions import load_page_geometry, transfer_positions
from document_ocr.synthesis.curated_reflow import ReflowEngine


def check(config_path: Path, output_path: Path, *, source_ids=None):
    root = Path.cwd()
    config = json.loads(config_path.read_text())
    if source_ids is not None and set(source_ids) != set(config["source_ids"]):
        if not set(source_ids) <= set(config["source_ids"]):
            raise ValueError("geometry selection outside prepared campaign")
        # Superseded draws must not prevent replay of unrelated retained draws.
        # Preserve every selected request byte-for-byte; pin the explicit subset.
        config["source_ids"] = source_ids
        config["capabilities"] = {sid: config["capabilities"][sid] for sid in source_ids}
        if "template_sampling" in config:
            counts = config["template_sampling"]["source_counts"]
            config["template_sampling"]["source_counts"] = {sid: counts[sid] for sid in source_ids}
            config["template_sampling"]["samples"] = sum(counts[sid] for sid in source_ids)
        for name in ("requests", "private_plans"):
            rows = json.loads((root / config["independent"][name]["path"]).read_text())
            rows = {k: v for k, v in rows.items() if k.rsplit(":", 1)[0] in source_ids}
            path = output_path / f"{name}.json"
            atomic_publish_json(path, rows)
            config["independent"][name] = {"path": str(path), "sha256": sha256_file(path)}
        atomic_publish_json(output_path / "replay-config.json", config)
    campaign = create_campaign(root, config)
    selected = campaign.source_ids if source_ids is None else source_ids
    if not set(selected) <= set(campaign.source_ids):
        raise ValueError("geometry selection outside prepared campaign")
    dataset = root / config["dataset"]
    alignments = {
        sid: json.loads((dataset / "alignments" / f"{sid}.json").read_text()) for sid in selected
    }
    geometry, provenance = load_page_geometry(root, dataset, alignments)
    engine = ReflowEngine(root, PositionPolicy.model_validate(config["positions"]).reflow)
    rows = []
    for sid, variant in (
        (source_id, index)
        for source_id in selected
        for index in range(1, campaign.variant_counts[source_id] + 1)
    ):
        start = time.perf_counter()
        try:
            bp, _, sample_id, _ = campaign.plan(sid, variant)
            path = campaign.output / "candidates" / f"{sample_id}.json"
            candidate = json.loads(path.read_text())
            campaign.replay_candidate(candidate)
            text, proof = candidate["joinedRawText"], candidate["proof"]
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
                sample_id=sample_id,
            )
            output = output_path / f"{sample_id}.json"
            atomic_publish_json(
                output,
                {
                    "candidate": str(path.relative_to(root)),
                    "candidateSha256": sha256_file(path),
                    "positionedText": positioned,
                    "target": candidate["target"],
                    "geometry": receipt,
                },
            )
            row = {
                "sourceId": sid,
                "variant": variant,
                "sampleId": sample_id,
                "status": "PASS",
                "candidate": str(path.relative_to(root)),
                "positions": str(output),
                "aliasChanges": sum(
                    s["before"] != s["after"] for s in candidate["packagePresentations"]["slots"]
                ),
            }
        except (ValueError, KeyError, FileNotFoundError, AssertionError) as error:
            row = {"sourceId": sid, "variant": variant, "status": "HELD", "error": str(error)}
        row["seconds"] = time.perf_counter() - start
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "rows": rows,
        "counts": dict(Counter(r["status"] for r in rows)),
        "semanticReviewRequired": True,
    }
    atomic_publish_json(output_path / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.config, args.output)


if __name__ == "__main__":
    main()
