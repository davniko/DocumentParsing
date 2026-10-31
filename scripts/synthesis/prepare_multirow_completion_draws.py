"""Prepare explicit fresh draws for rejected proposals; retain all earlier evidence."""

import json
from pathlib import Path

from prepare_multirow_batch import compile_batch

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.independent_preparation import _load_source_context


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    config = json.loads((root / "recovery-base-config.json").read_text())
    config["ownership"] = str(root / "authority-repairs/repeated-locality/ownership.yaml")
    config["independent_preparation"]["retry_invalid_output"] = True
    path = root / "completion-base-config.json"
    atomic_publish_json(path, config)
    contracts = {}
    for item in sorted(root.glob("batch-*/source-contracts.json")):
        contracts.update(json.loads(item.read_text())["sources"])
    context = _load_source_context(Path.cwd(), config)
    prefixes = [
        "08ec8a5f",
        "0dbc75a0",
        "1149521e",
        "2763b891",
        "7b12b905",
        "95102412",
        "9a55d93a",
        "c7060e22",
        "f8aba530",
        "fb241e5d",
        "275a9b4d",
        "3c5643cf",
        "5ce1f67e",
        "8a9d1985",
        "9496d433",
        "d5999e53",
        "16abe25c",
        "19acdcf6",
        "c5b964bc",
        "600a1e89",
    ]
    ids = {s[4:12]: s for s in config["source_ids"]}
    compile_batch(
        path,
        root / "batch-35",
        prefixes,
        2,
        group_counts=[1, 2, 3],
        reviewed={ids[p]: contracts[ids[p]] for p in prefixes},
        context=context,
        seed=2026110103,
        source_group_counts={ids["0dbc75a0"]: [1, 1, 1]},
    )
    atomic_publish_json(
        root / "completion-draw-plan.json",
        {
            "status": "AWAITING_SYNTHESIS_AND_MAIN_REVIEW",
            "sources": [ids[p] for p in prefixes],
            "previousAttemptsRetained": True,
            "fixedSingleGroup": {
                ids[
                    "0dbc75a0"
                ]: "One installed carrying tank with one declared contents lot. Do not fabricate "
                "separate packaging or compartments by partitioning that continuous contents lot."
            },
            "nativeInvalidOutputRetry": "One explicit provider repair; rejection still surfaces.",
        },
    )


if __name__ == "__main__":
    main()
