"""Account for every registry source and prepare bounded inspection batches.

Contract compilation is not synthesis acceptance. Unresolved source presentations
are inventoried explicitly and never quietly omitted from the registry total.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_multirow_batch import compile_batch

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.declared_package_contracts import compile_declarations
from document_ocr.synthesis.independent_campaign import _complete_optional_package_suffix
from document_ocr.synthesis.independent_preparation import _load_source_context
from document_ocr.synthesis.independent_wording import prepare_blueprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    context = _load_source_context(Path.cwd(), config)
    existing = {}
    for path in sorted(args.output.glob("batch-*/base-config.json")):
        for sid in json.loads(path.read_text())["source_ids"]:
            existing.setdefault(sid, []).append(str(path.parent))
    rows, eligible = [], []
    for sid in config["source_ids"]:
        row = {"sourceId": sid, "priorBatches": existing.get(sid, [])}
        bp = context.numeric.apply_blueprint(context.source.blueprint(sid), sid)
        if sid in context.wording:
            bp = prepare_blueprint(bp, context.wording[sid])
        bp = _complete_optional_package_suffix(bp)
        certificate = context.numeric.certificate(sid)
        try:
            projection, owners = compile_declarations(bp)
            if certificate.get("constraints"):
                raise ValueError("explicit constrained-group contract required")
            row.update(
                status="CONTRACT_COMPILED_NOT_CERTIFIED",
                projection=projection,
                declarationCount=len(owners),
            )
            if sid not in existing:
                eligible.append(sid)
        except ValueError as error:
            row.update(status="PRESENTATION_REVIEW_REQUIRED", reason=str(error))
        row["regime"] = context.source.capabilities[sid].family
        row["privateRoles"] = sorted({r["role"] for r in certificate["regions"]})
        rows.append(row)
    start = 1 + max(
        (
            int(p.name.removeprefix("batch-"))
            for p in args.output.glob("batch-*")
            if p.name[6:].isdigit()
        ),
        default=0,
    )
    for index, offset in enumerate(range(0, len(eligible), 20), start):
        selected = eligible[offset : offset + 20]
        out = args.output / f"batch-{index:02}"
        compile_batch(
            args.config,
            out,
            [sid[4:12] for sid in selected],
            2,
            group_counts=[1, 2, 3],
            context=context,
        )
        for row in rows:
            if row["sourceId"] in selected:
                row["plannedBatch"] = str(out)
        print(json.dumps({"batch": str(out), "sources": len(selected)}), flush=True)
    result = {"status": "PLANNING_ONLY", "sources": len(rows), "records": rows}
    atomic_publish_json(args.output / "full-rollout-plan.json", result)
    print(
        json.dumps(
            {
                "sources": len(rows),
                "newlyBatched": len(eligible),
                "presentationReview": sum(
                    r["status"] == "PRESENTATION_REVIEW_REQUIRED" for r in rows
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
