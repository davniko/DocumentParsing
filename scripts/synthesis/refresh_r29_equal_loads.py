"""Bounded equal-load refresh. Stage native cargo; preserve current non-cargo.

R28 is read-only. Publication requires individually reviewed, replayable batches.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/curated/mpci-bl-r28-positions-v1"
DATA = ROOT / "data/curated/mpci-bl-r29-positions-v1"
OUT = ROOT / "docs/analysis/r29-equal-load-refresh-20261102"
ART = ROOT / "artifacts/kie-synthesis-production/r29-equal-load-refresh-20261102"


def load(path):
    return json.loads((ROOT / path).read_text())


def package_groups(row):
    """Count cargo groups, not repeated per-container public package rows."""
    rows = row["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
    categories = {p["typeCategory"] for p in rows}
    return max(len(categories), 2 if len(rows) > 1 else 1)


def configure_equipment(config, mixed_probability):
    """The probability applies only to an explicit eligible source scope."""
    if not config["source_ids"]:
        raise ValueError("equipment sampling requires selected sources")
    config["independent_preparation"].update(
        mixed_probability=mixed_probability, mixed_source_ids=list(config["source_ids"])
    )


def plan():
    if DATA.exists() or (OUT / "plan.json").exists():
        raise ValueError("R29 or plan already exists; use the saved immutable plan")
    hashes = {p.name: sha256_file(p) for p in SOURCE.iterdir() if p.is_file()}
    manifest = load(SOURCE / "manifest.json")
    if any(hashes[k] != v for k, v in manifest["files"].items()):
        raise ValueError("R28 differs from its published manifest")
    inventory = load("docs/analysis/paddle-migration-scope-20261102/cargo-refresh-candidates.json")
    provenance = {
        r["id"]: r
        for r in load("docs/analysis/paddle-migration-scope-20261102/synthetic-provenance.json")
    }
    rows = {r["documentId"]: r for r in map(json.loads, (SOURCE / "synthetic-train.jsonl").open())}
    base = load("configs/synthesis/mpci_bl_curated_v7_multirow.json")
    pool = defaultdict(list)
    for r in inventory:
        if not (r["old"] and r["complete"] and not r["equal"]):
            continue
        p = provenance[r["id"]]
        path = Path(p["path"])
        if p["cohort"].startswith("staged-"):
            receipt = load(path)
            path = Path(
                receipt["evidence"].get("nativePath", receipt["evidence"].get("historicalPath", ""))
            )
            if not path.is_file():
                continue
        old = load(path)
        if "proof" not in old:
            continue
        current = rows[r["id"]]
        pairs = {
            (c.get("sizeCategory"), c.get("typeCategory"))
            for c in current["target"]["containerInformation"]
        }
        pool[r["parent"]].append({**r, "candidatePath": str(path), "mixed": len(pairs) > 1})
    for values in pool.values():
        values.sort(key=lambda r: (r["package_rows"] > 1, digest(["r29-equal", r["id"]])))
    parents = sorted(pool, key=lambda s: (-max(r["containers"] for r in pool[s]), digest(s)))
    selected = []
    # Round robin avoids concentrating the equal-load pattern in a few parents.
    while len(selected) < 450:
        changed = False
        for sid in parents:
            if pool[sid] and len(selected) < 450:
                selected.append(pool[sid].pop(0))
                changed = True
        if not changed:
            break
    if len(selected) < 400:
        raise ValueError("insufficient scoped historical candidates")
    shutil.copytree(SOURCE, DATA)
    if any(sha256_file(DATA / k) != v for k, v in hashes.items()):
        raise ValueError("R29 copy differs")
    atomic_write_json(OUT / "r28-protected-hashes.json", hashes)
    atomic_write_json(OUT / "pool.json", selected)
    paths = []
    for batch, start in enumerate(range(0, len(selected), 50), 1):
        items = selected[start : start + 50]
        counts = Counter(r["parent"] for r in items)
        cfg = copy.deepcopy(base)
        cfg.update(
            source_ids=list(counts),
            capabilities={s: base["capabilities"][s] for s in counts},
            seed=2026110200 + batch,
            concurrency=8,
            budget_usd=15,
            allocation_sampling="equal",
            output=str((ART / f"batch-{batch:02d}" / "campaign").relative_to(ROOT)),
        )
        cfg["template_sampling"] = {"samples": len(items), "source_counts": dict(counts)}
        groups = defaultdict(list)
        mapped = []
        variants = Counter()
        for r in items:
            sid = r["parent"]
            variants[sid] += 1
            # Preserve existing multi-row coverage; add it to one third of
            # previously single-row draws. No nested level becomes a public row.
            count = max(package_groups(rows[r["id"]]), 2 if len(mapped) % 3 == 0 else 1)
            groups[sid].append(count)
            mapped.append({**r, "variant": variants[sid], "groupCount": count})
        cfg["declared_packages"]["source_group_counts"] = dict(groups)
        configure_equipment(cfg, 0.5)
        path = ART / f"batch-{batch:02d}" / "config.json"
        atomic_write_json(path, cfg)
        atomic_write_json(path.parent / "selection.json", mapped)
        paths.append(str(path.relative_to(ROOT)))
    atomic_write_json(
        OUT / "plan.json",
        dict(
            target=400,
            proposed=len(selected),
            parents=len({r["parent"] for r in selected}),
            configs=paths,
            status="STAGED_NOT_PUBLISHED",
            sourceHashes=hashes,
        ),
    )
    print(
        json.dumps(
            dict(
                configs=paths,
                candidates=len(selected),
                parents=len({r["parent"] for r in selected}),
            )
        )
    )


def stage(batch):
    from stage_combination_candidates import position_preservation, validate_literals

    from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7DirectTarget
    from document_ocr.synthesis.curated_campaign import create_campaign
    from document_ocr.synthesis.curated_layout import PositionPolicy
    from document_ocr.synthesis.curated_positions import load_page_geometry, transfer_positions
    from document_ocr.synthesis.curated_reflow import ReflowEngine
    from document_ocr.synthesis.scoped_cargo_refresh import (
        cargo_auxiliary_keys,
        cargo_surface_keys,
        migrate,
    )

    started = time.perf_counter()
    directory = ART / f"batch-{batch:02d}"
    summary = load(max((directory / "stage-runs").glob("*/summary.json")))
    cfg = load(summary["preparedConfig"])
    campaign = create_campaign(ROOT, cfg)
    selection = {(r["parent"], r["variant"]): r for r in load(directory / "selection.json")}
    current = {r["documentId"]: r for r in map(json.loads, (DATA / "synthetic-train.jsonl").open())}
    auxiliary = load(cfg["auxiliary"])["sources"]
    scoped = load("docs/analysis/r28-combination-refresh300-20261101/scoped-owner-reviews.json")
    dataset = ROOT / cfg["dataset"]
    alignments = {s: load(dataset / "alignments" / f"{s}.json") for s in cfg["source_ids"]}
    boxes, geometry = load_page_geometry(ROOT, dataset, alignments)
    engine = ReflowEngine(ROOT, PositionPolicy.model_validate(cfg["positions"]).reflow)
    decisions = (
        load(OUT / "native-decisions.json") if (OUT / "native-decisions.json").exists() else {}
    )
    results = []
    for row in summary["stages"]["geometry"]["rows"]:
        if row["status"] != "PASS":
            results.append(row)
            continue
        sid, variant = row["sourceId"], row["variant"]
        item = selection[(sid, variant)]
        did = item["id"]
        try:
            native = load(row["candidate"])
            findings = [
                f
                for f in load(campaign.output / "reviews" / f"{sid}.json")["review"]["findings"]
                if f["sample_id"] == row["sampleId"]
            ]
            if findings:
                decision = decisions.get(row["sampleId"])
                if not decision or decision["candidateSha256"] != digest(native):
                    raise ValueError("native wording requires main review: " + json.dumps(findings))
            campaign.replay_candidate(native)
            bp, scenario, _, target = campaign.plan(sid, variant)
            physical = campaign.physical_render(bp, scenario, target)
            private = (
                set(physical.surface_values)
                | set(physical.variable_values)
                | {r.key for r in bp.regions if r.kind == "product"}
                | cargo_auxiliary_keys(auxiliary.get(sid, {}).get("bindings", {}))
            )
            private.update(scoped.get(sid, {}).get("keys", []))
            private.update(cargo_surface_keys(bp))
            before = current[did]
            after, evidence = migrate(
                before,
                load(item["candidatePath"]),
                native,
                private_keys=private,
                regions=bp.regions,
            )
            BillOfLadingExtractionV7DirectTarget.model_validate_json(json.dumps(after["target"]))
            validate_literals(after)
            allocations = after["target"]["goodsItemDetails"][0]["splitGoodsPlacement"]
            quantities = [r["packageQuantity"] for r in allocations]
            if len(set(quantities)) != 1 or len(allocations) != item["containers"]:
                raise ValueError("scoped output lost complete equal allocations")
            _, receipt = transfer_positions(
                bp.source,
                after["joinedRawText"],
                evidence["geometryProof"],
                alignments[sid],
                page_boxes=boxes[sid],
                page_dimensions=geometry["pageDimensions"][sid],
            )
            positioned, receipt, _ = engine.apply(
                source=bp.source,
                rendered=after["joinedRawText"],
                proof=evidence["geometryProof"],
                alignment=alignments[sid],
                receipt=receipt,
                boxes=boxes[sid],
                dimensions=geometry["pageDimensions"][sid],
                sample_id=did,
            )
            if any(p["status"] == "rejected" for p in receipt["reflowPages"]):
                raise ValueError("scoped coordinate layout rejected")
            after.update(
                positionedText=positioned, positionedTextSha256=digest(positioned.encode())
            )
            evidence.update(
                unchangedPositionedLinesChecked=position_preservation(before, after),
                nativePath=row["candidate"],
                nativeSha256=digest(native),
                historicalPath=item["candidatePath"],
                preparedConfig=summary["preparedConfig"],
                privateOwnerKeys=sorted(private),
                batch=batch,
            )
            atomic_write_json(
                OUT / "staged-cargo" / f"{did}.json",
                dict(
                    before=before,
                    after=after,
                    evidence=evidence,
                    geometry=receipt,
                    reviewStatus="REQUIRES_HYBRID_REVIEW",
                ),
            )
            results.append(dict(documentId=did, status="STAGED", sampleId=row["sampleId"]))
        except (ValueError, KeyError, AssertionError) as error:
            results.append(
                dict(documentId=did, sampleId=row["sampleId"], status="HELD", reason=str(error))
            )
    result = dict(
        batch=batch,
        seconds=time.perf_counter() - started,
        counts=dict(Counter(r["status"] for r in results)),
        rows=results,
    )
    atomic_write_json(directory / "scoped-staging.json", result)
    print(json.dumps(result), flush=True)


async def review():
    import asyncio

    from review_combination_refresh import SYSTEM, Review

    from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner

    base = load(ART / "batch-01/config.json")
    cfg = {k: base[k] for k in CuratedSynthesisConfig.model_fields}
    cfg.update(output=str((ART / "hybrid-review").relative_to(ROOT)), concurrency=8, budget_usd=10)
    runner = Runner(ROOT, cfg)

    async def one(path):
        row = load(path)
        did = row["after"]["documentId"]
        dest = OUT / "hybrid-reviews" / path.name
        pin = digest(row["after"])
        if dest.exists() and load(dest)["afterSha256"] == pin:
            return
        native = load(row["evidence"]["nativePath"])
        prompt = dict(
            beforeText=row["before"]["joinedRawText"],
            afterText=row["after"]["joinedRawText"],
            beforeCargo=row["before"]["target"]["goodsItemDetails"],
            afterTarget=row["after"]["target"],
            sampledGoods=native["scenario"]["goods_identities"],
            privatePacking=native["scenario"]["provenance"]["freshPrivatePacking"],
            changedText=[
                dict(before=e["before"], after=e["after"])
                for e in row["evidence"]["currentProof"]["edits"]
                if e["before"] != e["after"]
            ],
        )
        result = await runner.call(
            "r29-scoped-review", did, Review, SYSTEM, json.dumps(prompt, ensure_ascii=False)
        )
        atomic_write_json(
            dest, dict(documentId=did, afterSha256=pin, findings=result.model_dump()["findings"])
        )

    await asyncio.gather(*(one(p) for p in sorted((OUT / "staged-cargo").glob("*.json"))))


def prefixes():
    from document_ocr.synthesis.container_prefix_refresh import refresh_prefix
    from document_ocr.synthesis.curated_reflow_policy import FontMetrics
    from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

    cfg = load("configs/synthesis/mpci_bl_curated_v7_multirow.json")
    font = cfg["positions"]["reflow"]
    metrics = FontMetrics(Path(font["font_path"]), font["font_sha256"])
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        (DATA / "task-constraints.json").read_bytes()
    )
    task = get_training_task(constraints.task).bind_constraints(constraints)
    rows = {r["documentId"]: r for r in map(json.loads, (DATA / "synthetic-train.jsonl").open())}
    excluded = {
        c["equipmentIdentifier"]
        for name in ("synthetic-train.jsonl", "real-train.jsonl", "validation.jsonl")
        for r in map(json.loads, (DATA / name).open())
        for c in r["target"].get("containerInformation", [])
    }
    inventory = load("docs/analysis/paddle-migration-scope-20261102/cargo-refresh-candidates.json")
    results = []
    for item in inventory:
        if not (item["old"] and item["complete"] and item["equal"] and item["containers"] >= 8):
            continue
        before = rows[item["id"]]
        prefix = min(
            {c["equipmentIdentifier"][:4] for c in before["target"]["containerInformation"]},
            key=lambda x: (metrics.length(x), x),
        )
        after, receipt = refresh_prefix(before, prefix=prefix, seed=2026110229, excluded=excluded)
        if task.canonicalize(after["target"]) != after["target"]:
            raise ValueError("prefix refresh target is noncanonical")
        for span in receipt["spans"]:
            if metrics.length(span["after"]) > metrics.length(span["before"]):
                raise ValueError("prefix refresh exceeds original printed footprint")
        excluded.update(receipt["identifiers"].values())
        receipt["fontSha256"] = font["font_sha256"]
        receipt["allPrintedFootprintsNonexpanding"] = True
        atomic_write_json(
            OUT / "staged-prefix" / f"{item['id']}.json",
            dict(before=before, after=after, evidence=receipt),
        )
        results.append(
            dict(
                id=item["id"],
                parent=item["parent"],
                containers=item["containers"],
                prefix=prefix,
                occurrences=len(receipt["spans"]),
            )
        )
    atomic_write_json(
        OUT / "prefix-summary.json",
        dict(records=len(results), parents=len({r["parent"] for r in results}), rows=results),
    )
    print(json.dumps(dict(records=len(results), parents=len({r["parent"] for r in results}))))


def extend():
    """Draw replacement proposals from unused old records on passing layouts."""
    old = {r["id"] for r in load(OUT / "pool.json")}
    for path in ART.glob("batch-*/selection.json"):
        old.update(r["id"] for r in load(path))
    passed = {load(p)["after"]["sourceDocumentId"] for p in (OUT / "staged-cargo").glob("*.json")}
    provenance = {
        r["id"]: r
        for r in load("docs/analysis/paddle-migration-scope-20261102/synthetic-provenance.json")
    }
    inventory = load("docs/analysis/paddle-migration-scope-20261102/cargo-refresh-candidates.json")
    pool = defaultdict(list)
    for r in inventory:
        if not (
            r["old"]
            and r["complete"]
            and not r["equal"]
            and r["id"] not in old
            and r["parent"] in passed
        ):
            continue
        p = provenance[r["id"]]
        path = Path(p["path"])
        if p["cohort"].startswith("staged-"):
            evidence = load(path)["evidence"]
            path = Path(evidence.get("nativePath", evidence.get("historicalPath", "")))
        if not path.is_file():
            continue
        pool[r["parent"]].append({**r, "candidatePath": str(path), "exact": p["raw_match"]})
    for values in pool.values():
        values.sort(key=lambda x: (not x["exact"], digest(x["id"])))
    selected = []
    while len(selected) < 200 and any(pool.values()):
        for sid in sorted(pool, key=digest):
            if pool[sid] and len(selected) < 200:
                selected.append(pool[sid].pop(0))
    current = {r["documentId"]: r for r in map(json.loads, (DATA / "synthetic-train.jsonl").open())}
    base = load("configs/synthesis/mpci_bl_curated_v7_multirow.json")
    first = max(int(p.name.split("-")[-1]) for p in ART.glob("batch-*")) + 1
    for batch, start in enumerate(range(0, len(selected), 50), first):
        items = selected[start : start + 50]
        counts = Counter(r["parent"] for r in items)
        cfg = copy.deepcopy(base)
        cfg.update(
            source_ids=list(counts),
            capabilities={s: base["capabilities"][s] for s in counts},
            seed=2026110200 + batch,
            concurrency=8,
            budget_usd=15,
            allocation_sampling="equal",
            output=str((ART / f"batch-{batch:02d}" / "campaign").relative_to(ROOT)),
        )
        cfg["template_sampling"] = {"samples": len(items), "source_counts": dict(counts)}
        groups, variants, mapped = defaultdict(list), Counter(), []
        for r in items:
            sid = r["parent"]
            variants[sid] += 1
            count = package_groups(current[r["id"]])
            groups[sid].append(count)
            mapped.append({**r, "variant": variants[sid], "groupCount": count})
        cfg["declared_packages"]["source_group_counts"] = dict(groups)
        configure_equipment(cfg, 1.0)
        path = ART / f"batch-{batch:02d}" / "config.json"
        atomic_write_json(path, cfg)
        atomic_write_json(path.parent / "selection.json", mapped)
    atomic_write_json(
        OUT / f"extension-{first:02d}.json",
        dict(firstBatch=first, records=len(selected), parents=len(pool)),
    )
    print(json.dumps(dict(firstBatch=first, records=len(selected), parents=len(pool))))


def repair_previous():
    from stage_combination_candidates import position_preservation, validate_literals

    from document_ocr.synthesis.curated_positions import transfer_existing_positions
    from document_ocr.synthesis.scoped_cargo_refresh import (
        apply_edits,
        cargo_path,
        current_interval,
        rendered_interval,
    )

    directory = ROOT / "docs/analysis/r28-combination-refresh300-20261101"
    authorities = load(
        "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v21/ownership.json"
    )["sources"]
    current = {r["documentId"]: r for r in map(json.loads, (DATA / "synthetic-train.jsonl").open())}
    results = []
    for did in load(directory / "cargo-selection.json")["documents"]:
        original = load(directory / "staged-cargo" / f"{did}.json")
        native = load(original["evidence"]["nativePath"])
        sid = native["sourceDocumentId"]
        missing = {
            k
            for k, v in authorities[sid].get("surfaces", {}).items()
            if any(f in v and cargo_path(v[f]) for f in ("package_path", "measure_path"))
        }
        missing -= set(original["evidence"]["privateOwnerKeys"])
        if not any(e["key"] in missing for e in native["proof"]["edits"]):
            continue
        before = current[did]
        changes = []
        for edit in native["proof"]["edits"]:
            if edit["key"] not in missing:
                continue
            a, b = rendered_interval(
                original["evidence"]["geometryProof"], edit["byteStart"], edit["byteEnd"]
            )
            a, b = current_interval(
                original["after"]["joinedRawText"].encode(), before["joinedRawText"].encode(), a, b
            )
            text = before["joinedRawText"].encode()[a:b].decode()
            if text != edit["after"]:
                changes.append({**edit, "byteStart": a, "byteEnd": b, "before": text})
        if not changes:
            continue
        after = copy.deepcopy(before)
        after["joinedRawText"] = apply_edits(before["joinedRawText"].encode(), changes).decode()
        after["joinedRawTextSha256"] = digest(after["joinedRawText"].encode())
        proof = dict(
            sourceSha256=digest(before["joinedRawText"].encode()),
            renderedSha256=after["joinedRawTextSha256"],
            edits=changes,
        )
        evidence = dict(currentProof=proof)
        validate_literals(after)
        positioned, receipt = transfer_existing_positions(
            before["joinedRawText"],
            before["positionedText"],
            after["joinedRawText"],
            evidence["currentProof"],
        )
        after.update(positionedText=positioned, positionedTextSha256=digest(positioned.encode()))
        evidence.update(
            nativePath=original["evidence"]["nativePath"],
            missingOwners=sorted(missing),
            unchangedPositionedLinesChecked=position_preservation(before, after),
        )
        atomic_write_json(
            OUT / "staged-legacy" / f"{did}.json",
            dict(before=before, after=after, evidence=evidence, geometry=receipt),
        )
        results.append(dict(documentId=did, changes=changes))
    atomic_write_json(OUT / "legacy-summary.json", dict(records=len(results), rows=results))
    print(json.dumps(dict(records=len(results))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["plan", "stage", "review", "prefixes", "extend", "repair-previous"]
    )
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    if args.action == "plan":
        plan()
    elif args.action == "stage":
        stage(args.batch)
    elif args.action == "prefixes":
        prefixes()
    elif args.action == "extend":
        extend()
    elif args.action == "repair-previous":
        repair_previous()
    else:
        import asyncio

        asyncio.run(review())
