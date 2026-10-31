"""Schedule every unresolved source for explicit fresh controlled synthesis.

Previous attempts are retained. Rendering/geometry repairs reuse frozen native
draws separately; this redraw set covers unavailable proposals, lexical failure,
revised thermal eligibility and missing one/two/three-group controls.
"""

import json
from collections import defaultdict
from pathlib import Path

from prepare_multirow_batch import compile_batch

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.independent_preparation import _load_source_context


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    reasons = defaultdict(list)
    manual = {}
    seen = defaultdict(set)
    for number in range(1, 26):
        batch = root / f"batch-{number:02}"
        for stage in ("base-preparation", "group-preparation"):
            path = batch / f"{stage}.json"
            if not path.exists():
                raise ValueError("initial preparation is unfinished: " + str(path))
            for row in json.loads(path.read_text()).get("inventory", {}).values():
                if row["status"] not in {"ready", "blocked_by_source_group"}:
                    reasons[row["sourceDocumentId"]].append(
                        {
                            "batch": number,
                            "stage": stage,
                            "variant": row["variant"],
                            "status": row["status"],
                            "detail": row.get("error", row.get("reason")),
                        }
                    )
        config = json.loads((batch / "reviewed-authorities/config.json").read_text())
        scope = json.loads((batch / "reviewed-authorities/scope.json").read_text())
        for sid in scope["requiresRedraw"]:
            reasons[sid].append({"batch": number, "status": "revised_thermal_eligibility"})
        campaign = Path(config["output"])
        wordings = defaultdict(set)
        for path in (campaign / "wording").glob("*.json"):
            receipt = json.loads(path.read_text())
            wordings[receipt["sourceDocumentId"]].add(receipt["scenario"]["variant"])
        candidates = {
            json.loads(p.read_text())["sourceDocumentId"]: []
            for p in (campaign / "candidates").glob("*.json")
        }
        for p in (campaign / "candidates").glob("*.json"):
            value = json.loads(p.read_text())
            candidates[value["sourceDocumentId"]].append(value["variant"])
            source = value["sourceDocumentId"]
            policy = config["declared_packages"]
            counts = policy.get("source_group_counts", {}).get(source, policy["group_counts"])
            seen[source].add(counts[(value["variant"] - 1) % len(counts)])
        for sid in config["source_ids"]:
            # Goods corrections can archive a candidate; a valid wording receipt
            # still exists and will be replayed, so do not discard that good draw.
            counts = config.get("template_sampling", {}).get("source_counts", {})
            count = counts.get(sid, config["variants_per_source"])
            missing = sorted(set(range(1, count + 1)) - wordings[sid])
            if missing:
                reasons[sid].append(
                    {"batch": number, "status": "missing_native_wording", "variants": missing}
                )
        stages = sorted(batch.glob("stage-runs/*/render.json"))
        if stages:
            for error in json.loads(stages[-1].read_text())["errors"]:
                if "No such file" in error["error"]:
                    reasons[error["sourceDocumentId"]].append(
                        {
                            "batch": number,
                            "status": "missing_native_generation_receipt",
                            "variant": error["variant"],
                            "detail": error["error"],
                        }
                    )
        if number in (24, 25):
            manual.update(json.loads((batch / "source-contracts.json").read_text())["sources"])
    for sid, counts in seen.items():
        if counts == {2}:
            reasons[sid].append({"status": "first_batch_requires_single_and_three_group_controls"})
    # A single installed flexitank in one carrying box cannot become two or
    # three independently filled flexitanks. Exercise three fresh single-row
    # controls, retaining the source rather than inventing additional equipment.
    one_only = {sid: [1, 1, 1] for sid in reasons if sid.startswith("doc_458d2fa0")}
    config = json.loads(
        Path("configs/synthesis/mpci_bl_curated_v7_package_aliases.json").read_text()
    )
    authority = root / "authority-repairs/marks-and-localities"
    config.update(
        ownership=str(authority / "ownership.yaml"), auxiliary=str(authority / "auxiliary.yaml")
    )
    config_path = root / "recovery-base-config.json"
    atomic_publish_json(config_path, config)
    context = _load_source_context(Path.cwd(), config)
    sources = sorted(reasons)
    batches = []
    for index, start in enumerate(range(0, len(sources), 20), 26):
        ids = sources[start : start + 20]
        out = root / f"batch-{index:02}"
        compile_batch(
            config_path,
            out,
            [sid[4:12] for sid in ids],
            2,
            group_counts=[1, 2, 3],
            context=context,
            reviewed={s: manual[s] for s in ids if s in manual},
            seed=2026110101,
            source_group_counts={s: one_only[s] for s in ids if s in one_only},
        )
        batches.append({"batch": index, "sources": ids})
    atomic_publish_json(
        root / "recovery-plan.json",
        {
            "status": "AWAITING_FRESH_SYNTHESIS",
            "sources": dict(reasons),
            "batches": batches,
            "singleRowPhysicalControls": one_only,
            "initialSourcesRetained": True,
        },
    )
    print(json.dumps({"sources": len(sources), "batches": [b["batch"] for b in batches]}))


if __name__ == "__main__":
    main()
