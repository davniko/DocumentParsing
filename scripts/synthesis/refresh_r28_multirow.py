"""Prepare auditable, one-for-one R28 multirow regeneration campaigns.

This command only selects and stages. Native acceptance and dataset publication
remain separate: a generated candidate is never implicitly a training record.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import random
import resource
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"
OUT = ROOT / "docs/analysis/r28-multirow-refresh1500-20261031"
CAMPAIGNS = ROOT / "artifacts/kie-synthesis-production/r28-multirow-refresh1500-20261031"
SEED = 2026103115


def select_batches(groups: dict[str, list[dict]], *, seed: int) -> list[dict]:
    """Select 300 layouts, twenty per batch, with exactly 100 existing slots."""
    rng = random.Random(seed)
    eligible = sorted(sid for sid, rows in groups.items() if len(rows) >= 3)
    rng.shuffle(eligible)
    # Prefer broad coverage while avoiding a batch whose available slots cannot
    # reach 100. Five is a desired contribution, not a source eligibility rule.
    eligible.sort(key=lambda sid: min(len(groups[sid]), 5), reverse=True)
    selected = eligible[:300]
    if len(selected) != 300:
        raise ValueError("fewer than 300 eligible source layouts")
    rng.shuffle(selected)
    batches = []
    for offset in range(0, 300, 20):
        sources = selected[offset : offset + 20]
        counts = {sid: min(len(groups[sid]), 5) for sid in sources}
        while sum(counts.values()) < 100:
            expandable = [sid for sid in sources if counts[sid] < min(len(groups[sid]), 6)]
            if not expandable:
                raise ValueError("selected batch cannot supply 100 records")
            counts[min(expandable, key=lambda sid: (counts[sid], sources.index(sid)))] += 1
        if sum(counts.values()) != 100:
            raise ValueError("unexpected batch cardinality")
        rows = []
        for sid in sources:
            choices = list(groups[sid])
            rng.shuffle(choices)
            # Replace source-donor cargo first, while retaining random selection
            # within that stratum. Fresh campaigns sample independently.
            choices.sort(key=lambda row: row.get("donorDocumentId") is None)
            for variant, row in enumerate(choices[: counts[sid]], 1):
                rows.append(
                    {
                        "documentId": row["documentId"],
                        "sourceDocumentId": sid,
                        "variant": variant,
                        "packageRows": 3 if variant == 3 else 2,
                        "previousCandidate": row["candidatePath"],
                        "previousDonor": row.get("donorDocumentId"),
                    }
                )
        batches.append({"sources": sources, "sourceCounts": counts, "records": rows})
    return batches


def stage() -> dict:
    if (OUT / "selection.json").exists():
        raise ValueError("selection already exists; resume its pinned campaigns")
    config_path = ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json"
    config = json.loads(config_path.read_text())
    rows = {
        row["documentId"]: row for row in map(json.loads, (DATA / "synthetic-train.jsonl").open())
    }
    index_path = (
        ROOT / "docs/analysis/r28-quality-and-variability-20261019/donor/per-record-inventory.json"
    )
    index = json.loads(index_path.read_text())
    if {row["documentId"] for row in index} != rows.keys():
        raise ValueError("provenance index does not cover current dataset membership")
    groups = defaultdict(list)
    for item in index:
        sid = item["sourceDocumentId"]
        row = rows[item["documentId"]]
        if row["sourceDocumentId"] != sid:
            raise ValueError("provenance source differs from current record")
        if item["destinationCountry"] != "AE":
            continue
        if config["capabilities"].get(sid, {}).get("family") != "ambient":
            continue
        if sid in config["declared_packages"].get("source_group_counts", {}):
            continue
        if len({json.dumps(e, sort_keys=True) for e in item["equipment"]}) > 1:
            continue
        target = json.dumps(row["target"])
        if '"dangerousGoods"' in target or '"temperatureSetpoint"' in target:
            continue
        goods = row["target"]["goodsItemDetails"]
        if len(goods) != 1 or len(goods[0]["numberAndTypeOfPackages"]) != 1:
            raise ValueError("refresh assumes exactly one existing public package row")
        groups[sid].append(item)
    batches = select_batches(groups, seed=SEED)
    pins = {p.name: sha256_file(p) for p in DATA.iterdir() if p.is_file()}
    manifest = json.loads((DATA / "manifest.json").read_text())
    if any(pins[name] != expected for name, expected in manifest["files"].items()):
        raise ValueError("current dataset manifest is stale")
    before = OUT / "before"
    before.mkdir(parents=True, exist_ok=True)
    for name in pins:
        target = before / name
        if not target.exists():
            shutil.copy2(DATA / name, target)
        if sha256_file(target) != pins[name]:
            raise ValueError("before snapshot differs from current dataset")
    alias_ids = set(
        json.loads((ROOT / "docs/analysis/r28-alias-refresh3000-20261031/stage.json").read_text())[
            "selectedIds"
        ]
    )
    for number, batch in enumerate(batches, 1):
        candidate = copy.deepcopy(config)
        candidate["source_ids"] = batch["sources"]
        candidate["capabilities"] = {
            sid: {**candidate["capabilities"][sid], "destination_country": "AE"}
            for sid in batch["sources"]
        }
        candidate["seed"] = SEED + number
        candidate["variants_per_source"] = 6
        candidate["template_sampling"] = {"samples": 100, "source_counts": batch["sourceCounts"]}
        candidate["declared_packages"]["group_counts"] = [2, 2, 3, 2, 2, 2]
        candidate["declared_packages"]["source_group_counts"] = {}
        directory = CAMPAIGNS / f"batch-{number:02d}"
        candidate["output"] = str(directory.relative_to(ROOT) / "campaign")
        path = directory / "config.json"
        atomic_publish_json(path, candidate)
        batch["batch"] = number
        batch["config"] = str(path.relative_to(ROOT))
        batch["configSha256"] = sha256_file(path)
        for record in batch["records"]:
            record["candidateId"] = (
                "syn_full_v7_"
                + digest([record["sourceDocumentId"], candidate["seed"], record["variant"]])[:24]
            )
            record["previousRecordSha256"] = digest(rows[record["documentId"]])
            record["previousAliasRefresh"] = record["documentId"] in alias_ids
    selected = [row for batch in batches for row in batch["records"]]
    if len({r["documentId"] for r in selected}) != 1500:
        raise ValueError("selection has duplicate or missing identities")
    result = {
        "status": "STAGED_NOT_GENERATED_NOT_PUBLISHED",
        "seed": SEED,
        "beforeFiles": pins,
        "productionConfigSha256": sha256_file(config_path),
        "previousIndexSha256": sha256_file(index_path),
        "records": 1500,
        "parents": 300,
        "packageRowCounts": Counter(r["packageRows"] for r in selected),
        "previousAliasRefresh": sum(r["previousAliasRefresh"] for r in selected),
        "previousDonorBound": sum(r["previousDonor"] is not None for r in selected),
        "scope": (
            "UAE ambient, uniform equipment; existing thermal, DG and mixed equipment "
            "records preserved"
        ),
        "batches": batches,
    }
    atomic_publish_json(OUT / "selection.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}), flush=True)
    return result


def certify(batch_number: int, prepared_path: Path) -> dict:
    """Run production admission after explicit main-agent review of a batch."""
    from document_ocr.synthesis.curated_campaign import create_campaign
    from document_ocr.synthesis.curated_positions import position_campaign
    from document_ocr.synthesis.curated_publication import publish_campaign
    from document_ocr.synthesis.declared_package_rendering import public_rows
    from document_ocr.synthesis.declared_packages import DeclaredPackages
    from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

    started = time.perf_counter()
    selection = json.loads((OUT / "selection.json").read_text())
    batch = selection["batches"][batch_number - 1]
    if batch["batch"] != batch_number:
        raise ValueError("batch index mismatch")
    config = json.loads(prepared_path.read_text())
    counts = config["template_sampling"]["source_counts"]
    if not counts or any(batch["sourceCounts"].get(sid) != n for sid, n in counts.items()):
        raise ValueError("cannot certify partial source groups or substituted sources")
    campaign = create_campaign(ROOT, config)
    expected = {}
    for record in batch["records"]:
        if record["sourceDocumentId"] in counts:
            candidate_id = (
                "syn_full_v7_"
                + digest([record["sourceDocumentId"], config["seed"], record["variant"]])[:24]
            )
            expected[candidate_id] = {
                **record,
                "initialCandidateId": record["candidateId"],
                "candidateId": candidate_id,
            }
    if len(expected) != sum(counts.values()):
        raise ValueError("native component membership differs from selected slots")
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        (DATA / "task-constraints.json").read_bytes()
    )
    task = get_training_task(constraints.task).bind_constraints(constraints)
    with (campaign.output / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        admission = publish_campaign(campaign, publish=True)
        positions = position_campaign(campaign)
    path = campaign.output / config["positions"]["output_subdirectory"] / "dataset.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    publication_sha256 = sha256_file(path)
    if len(rows) != len(expected) or {r["documentId"] for r in rows} != expected.keys():
        raise ValueError("native publication membership differs from selection")
    categories = Counter()
    type_combinations = Counter()
    receipts = []
    for row in rows:
        owner = expected[row["documentId"]]
        if row["sourceDocumentId"] != owner["sourceDocumentId"]:
            raise ValueError("native publication changed source identity")
        if task.canonicalize(row["target"]) != row["target"]:
            raise ValueError("target is not canonical under R28 constraints")
        if (
            sha256_file(DATA / "task-constraints.json")
            != selection["beforeFiles"]["task-constraints.json"]
        ):
            raise ValueError("R28 target contract changed during generation")
        goods = row["target"]["goodsItemDetails"]
        _, scenario, sample_id, _ = campaign.plan(owner["sourceDocumentId"], owner["variant"])
        if sample_id != row["documentId"] or len(goods) != 1:
            raise ValueError("native plan identity differs from selected record")
        ledger = DeclaredPackages.from_receipt(scenario.provenance["declaredPackages"])
        projection = campaign.declared_package_contracts[owner["sourceDocumentId"]]["projection"]
        if ledger is None or len(ledger.groups) != owner["packageRows"]:
            raise ValueError("declared group count differs from controlled draw")
        expected_rows = public_rows(ledger, projection)
        if goods[0]["numberAndTypeOfPackages"] != expected_rows or len(expected_rows) < 2:
            raise ValueError("public package rows differ from source-faithful ledger projection")
        types = [p["typeCategory"] for p in goods[0]["numberAndTypeOfPackages"]]
        categories.update(types)
        type_combinations[" + ".join(types)] += 1
        receipts.append(
            {
                **owner,
                "publicPackageRows": len(expected_rows),
                "projection": projection,
                "nativeRecordSha256": digest(row),
                "nativePublication": str(path.relative_to(ROOT)),
                "nativePublicationSha256": publication_sha256,
            }
        )
    result = {
        "status": "PRODUCTION_ADMITTED_PENDING_R28_REPLACEMENT",
        "batch": batch_number,
        "preparedConfig": str(prepared_path.relative_to(ROOT)),
        "preparedConfigSha256": sha256_file(prepared_path),
        "records": receipts,
        "categories": categories,
        "typeCombinations": type_combinations,
        "rowCounts": Counter(r["packageRows"] for r in receipts),
        "publicRowCounts": Counter(r["publicPackageRows"] for r in receipts),
        "unresolvedReviewFindings": json.loads((campaign.output / "manifest.json").read_text())[
            "unresolvedReviewFindings"
        ],
        "rejectedReviewFindings": admission["rejectedReviewFindings"],
        "positions": {
            k: positions[k]
            for k in ("records", "knownLines", "counts", "reflowRejections", "heldGroups")
        },
        "seconds": time.perf_counter() - started,
        "maxRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    identity = digest(config)[:16]
    atomic_publish_json(
        OUT / "batches" / f"{batch_number:02d}-{identity}-component-admission.json", result
    )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return result


def redraw(batch_number: int, source_ids: list[str], reason: str) -> Path:
    """Record an explicit new draw for held sources, never overwrite an attempt."""
    selection = json.loads((OUT / "selection.json").read_text())
    batch = selection["batches"][batch_number - 1]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("redraw needs distinct, explicitly reviewed source IDs")
    if not set(source_ids) <= set(batch["sources"]) or not reason.strip():
        raise ValueError("redraw source/reason outside selected batch")
    config = json.loads((ROOT / batch["config"]).read_text())
    counts = {sid: batch["sourceCounts"][sid] for sid in source_ids}
    directory = CAMPAIGNS / f"batch-{batch_number:02d}" / "redraws" / str(time.time_ns())
    config["source_ids"] = source_ids
    config["capabilities"] = {sid: config["capabilities"][sid] for sid in source_ids}
    config["template_sampling"] = {"samples": sum(counts.values()), "source_counts": counts}
    config["seed"] = int(directory.name)
    config["output"] = str(directory.relative_to(ROOT) / "campaign")
    path = directory / "config.json"
    atomic_publish_json(path, config)
    atomic_publish_json(
        directory / "reason.json",
        {
            "batch": batch_number,
            "sources": source_ids,
            "reason": reason,
            "originalConfigSha256": batch["configSha256"],
            "configSha256": sha256_file(path),
            "previousAttemptsRetained": True,
            "status": "UNVALIDATED_NEW_DRAW_NOT_DATASET_REPLACEMENT",
        },
    )
    print(str(path.relative_to(ROOT)), flush=True)
    return path


def subset_config(prepared_path: Path, sources: list[str], output: Path) -> Path:
    """Pin a complete source-group subset without changing any accepted draw."""
    output = output.resolve()
    output.relative_to(ROOT)
    config = json.loads(prepared_path.read_text())
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("subset requires distinct source IDs")
    if not set(sources) <= set(config["source_ids"]):
        raise ValueError("subset contains an unprepared source")
    config["source_ids"] = sources
    config["capabilities"] = {sid: config["capabilities"][sid] for sid in sources}
    counts = config["template_sampling"]["source_counts"]
    config["template_sampling"] = {
        "samples": sum(counts[sid] for sid in sources),
        "source_counts": {sid: counts[sid] for sid in sources},
    }
    for name in ("requests", "private_plans"):
        pin = config["independent"][name]
        path = ROOT / pin["path"]
        if sha256_file(path) != pin["sha256"]:
            raise ValueError("subset input differs from frozen preparation")
        entries = json.loads(path.read_text())
        entries = {k: v for k, v in entries.items() if k.rsplit(":", 1)[0] in sources}
        path = output / (name + ".json")
        atomic_publish_json(path, entries)
        config["independent"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    path = output / "config.json"
    atomic_publish_json(path, config)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "certify", "redraw"])
    parser.add_argument("--batch", type=int, choices=range(1, 16))
    parser.add_argument("--prepared-config", type=Path)
    parser.add_argument("--source", action="append")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.command == "stage":
        stage()
    elif args.command == "redraw":
        if args.batch is None or not args.source or not args.reason:
            parser.error("redraw requires --batch, --source and --reason")
        redraw(args.batch, args.source, args.reason)
    elif args.batch is None or args.prepared_config is None:
        parser.error("certify requires --batch and --prepared-config")
    else:
        certify(args.batch, args.prepared_config.resolve())


if __name__ == "__main__":
    main()
