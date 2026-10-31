"""Prepare fresh draws only for sources still lacking complete render controls."""

import json
from collections import defaultdict
from pathlib import Path

from prepare_multirow_batch import compile_batch

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_preparation import _load_source_context


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    seen, contracts, evidence = defaultdict(set), {}, defaultdict(list)
    for number in range(1, 32):
        batch = root / f"batch-{number:02}"
        base = batch / "reviewed-authorities"
        if not base.exists():
            base = batch
        summaries = sorted(base.glob("stage-runs/*/summary.json"))
        if not summaries:
            raise ValueError("unfinished batch: " + str(number))
        summary = summaries[-1]
        value = json.loads(summary.read_text())
        config_path = base / ("config.json" if base != batch else "prepared-config.json")
        config = json.loads(config_path.read_text())
        contracts.update(json.loads((batch / "source-contracts.json").read_text())["sources"])
        rows = value["stages"]["geometry"]["rows"]
        newer_geometry = batch / "edge-overlap-geometry/summary.json"
        if newer_geometry.exists():
            rows = json.loads(newer_geometry.read_text())["rows"]
        for row in rows:
            if row["status"] != "PASS":
                continue
            candidate = Path(row["candidate"])
            if not candidate.exists():
                continue  # Explicit wording correction has archived this candidate.
            geometry = json.loads(Path(row["positions"]).read_text())
            if geometry["candidateSha256"] != sha256_file(candidate):
                continue
            source = row["sourceId"]
            policy = config["declared_packages"]
            values = policy.get("source_group_counts", {}).get(source, policy["group_counts"])
            seen[source].add(values[(row["variant"] - 1) % len(values)])
            evidence[source].append({"batch": number, "variant": row["variant"]})
    config_path = root / "recovery-base-config.json"
    config = json.loads(config_path.read_text())
    one_only = {
        s: [1, 1, 1] for s in config["source_ids"] if s.startswith(("doc_458d2fa0", "doc_bef02d48"))
    }
    missing = {
        s: sorted(({1} if s in one_only else {1, 2, 3}) - seen[s]) for s in config["source_ids"]
    }
    # These have a separate diagnosed repair/review in progress, not a reason
    # to discard their frozen draws or consume unrelated new goods proposals.
    retained = {"1149521e", "3056615e", "e18486c6"}
    sources = [s for s, counts in missing.items() if counts and s[4:12] not in retained]
    # This accepted render's ambient bitumen/flexitank physical proposal is held
    # by the main reviewer; reject the draw rather than erase its phase wording.
    physical_hold = next(s for s in config["source_ids"] if s.startswith("doc_7046a05d"))
    if physical_hold not in sources:
        sources.append(physical_hold)
    context = _load_source_context(Path.cwd(), config)
    batches = []
    for number, start in enumerate(range(0, len(sources), 20), 32):
        selected = sorted(sources)[start : start + 20]
        compile_batch(
            config_path,
            root / f"batch-{number:02}",
            [s[4:12] for s in selected],
            2,
            group_counts=[1, 2, 3],
            context=context,
            reviewed={s: contracts[s] for s in selected},
            seed=2026110102,
            source_group_counts={s: one_only[s] for s in selected if s in one_only},
        )
        batches.append({"batch": number, "sources": selected})
    atomic_publish_json(
        root / "final-recovery-plan.json",
        {
            "status": "AWAITING_SYNTHESIS_NOT_ACCEPTANCE",
            "missingControls": missing,
            "retainedDrawRepairs": sorted(retained),
            "physicalDrawRejected": physical_hold,
            "batches": batches,
            "previousEvidenceRetained": True,
        },
    )
    print(json.dumps({"sources": len(sources), "batches": [b["batch"] for b in batches]}))


if __name__ == "__main__":
    main()
