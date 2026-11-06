"""Replay and publish the bounded R29 equal-load refresh; R28 stays frozen."""

import argparse
import json
import resource
import time
from collections import Counter, defaultdict

from finalize_r28_expansion import census
from refresh_r28_package_aliases import promote_files
from refresh_r29_equal_loads import ART, DATA, OUT, ROOT, SOURCE, load
from stage_combination_candidates import position_preservation, validate_literals

from document_ocr.atomic import atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.scoped_cargo_refresh import (
    apply_edits,
    cargo_auxiliary_keys,
    cargo_surface_keys,
    migrate,
    reconstruct_source,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

SHARDS = ("synthetic-train.jsonl", "previous-uae-synthetic.jsonl", "old-uae-addition.jsonl")


def task():
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        (DATA / "task-constraints.json").read_bytes()
    )
    return get_training_task(constraints.task).bind_constraints(constraints)


def features(row):
    target = row["target"]
    containers = target.get("containerInformation", [])
    goods = target["goodsItemDetails"]
    assert len(goods) == 1
    placements = goods[0].get("splitGoodsPlacement", [])
    quantities = [p.get("packageQuantity") for p in placements]
    complete = (
        len(containers) > 1
        and len(placements) == len(containers)
        and {p["equipmentIdentifier"] for p in placements}
        == {c["equipmentIdentifier"] for c in containers}
        and all(q is not None for q in quantities)
    )
    return dict(
        containers=len(containers),
        complete=complete,
        equal=complete and len(set(quantities)) == 1,
        sharedPrefix=bool(containers)
        and len({c["equipmentIdentifier"][:4] for c in containers}) == 1,
        mixed=len({(c.get("sizeCategory"), c.get("typeCategory")) for c in containers}) > 1,
        packageRows=len(goods[0].get("numberAndTypeOfPackages", [])),
    )


def allocation_stats(rows):
    counts, parents = Counter(), defaultdict(set)
    for row in rows:
        f = features(row)
        for key, condition in (
            ("completeMulti", f["complete"]),
            ("equalMulti", f["equal"]),
            ("equalMixed", f["equal"] and f["mixed"]),
            ("equalMultiplePackageRows", f["equal"] and f["packageRows"] > 1),
            (
                "eightPlusEqualSharedPrefix",
                f["containers"] >= 8 and f["equal"] and f["sharedPrefix"],
            ),
            ("tenPlus", f["containers"] >= 10),
        ):
            if condition:
                counts[key] += 1
                parents[key].add(row.get("sourceDocumentId", row["documentId"]))
    return dict(
        counts=counts,
        parents={k: len(v) for k, v in parents.items()},
        equalFraction=counts["equalMulti"] / counts["completeMulti"],
    )


def audit():
    started = time.perf_counter()
    authority = task()
    scoped = load("docs/analysis/r28-combination-refresh300-20261101/scoped-owner-reviews.json")
    accepted, held, batches = {}, {}, []
    latest = {}
    for path in sorted(ART.glob("batch-*/selection.json")):
        batch = int(path.parent.name.split("-")[-1])
        latest.update({r["id"]: batch for r in load(path)})
    for summary_path in sorted(ART.glob("batch-*/scoped-staging.json")):
        stage = load(summary_path)
        paths = [
            OUT / "staged-cargo" / (r["documentId"] + ".json")
            for r in stage["rows"]
            if r["status"] == "STAGED"
        ]
        campaign = None
        passed = 0
        for path in paths:
            row = load(path)
            if (
                row["evidence"]["batch"] != stage["batch"]
                or latest[row["after"]["documentId"]] != stage["batch"]
            ):
                continue  # A separately reviewed proposal superseded this attempt.
            did = row["after"]["documentId"]
            try:
                evidence = row["evidence"]
                review = load(OUT / "hybrid-reviews" / path.name)
                assert review["afterSha256"] == digest(row["after"]), "stale text review"
                assert not review["findings"], "unadjudicated text finding"
                if campaign is None:
                    cfg = load(evidence["preparedConfig"])
                    campaign = create_campaign(ROOT, cfg)
                    auxiliary = load(cfg["auxiliary"])["sources"]
                native = load(evidence["nativePath"])
                assert digest(native) == evidence["nativeSha256"]
                campaign.replay_candidate(native)
                sid, variant = native["sourceDocumentId"], native["variant"]
                bp, scenario, _, target = campaign.plan(sid, variant)
                physical = campaign.physical_render(bp, scenario, target)
                keys = (
                    set(physical.surface_values)
                    | set(physical.variable_values)
                    | {r.key for r in bp.regions if r.kind == "product"}
                    | cargo_auxiliary_keys(auxiliary.get(sid, {}).get("bindings", {}))
                    | cargo_surface_keys(bp)
                    | set(scoped.get(sid, {}).get("keys", []))
                )
                assert keys == set(evidence["privateOwnerKeys"])
                expected, proof = migrate(
                    row["before"],
                    load(evidence["historicalPath"]),
                    native,
                    private_keys=keys,
                    regions=bp.regions,
                )
                assert all(expected[f] == row["after"][f] for f in ("joinedRawText", "target"))
                assert proof["currentProof"] == evidence["currentProof"]
                assert proof["geometryProof"] == evidence["geometryProof"]
                assert (
                    apply_edits(
                        reconstruct_source(native), proof["geometryProof"]["edits"]
                    ).decode()
                    == row["after"]["joinedRawText"]
                )
                assert (
                    digest(row["after"]["positionedText"].encode())
                    == row["geometry"]["positionedSha256"]
                )
                assert all(p["status"] != "rejected" for p in row["geometry"]["reflowPages"])
                position_preservation(row["before"], row["after"])
                validate_literals(row["after"])
                assert authority.canonicalize(row["after"]["target"]) == row["after"]["target"]
                old, new = features(row["before"]), features(row["after"])
                assert old["complete"] and not old["equal"] and new["equal"]
                assert new["packageRows"] >= old["packageRows"]
                goods = row["after"]["target"]["goodsItemDetails"][0]
                assert sum(p["packageQuantity"] for p in goods["splitGoodsPlacement"]) == sum(
                    p["packageQuantity"] for p in goods["numberAndTypeOfPackages"]
                )
                assert not scenario.provenance["independentDraw"]["provenance"][
                    "sourceShipmentDependence"
                ]
                accepted[did] = dict(
                    stagedSha256=sha256_file(path),
                    reviewSha256=digest(review),
                    batch=stage["batch"],
                    parent=sid,
                    features=new,
                )
                passed += 1
            except (ValueError, KeyError, AssertionError, FileNotFoundError) as error:
                held[did] = str(error)
        batches.append(dict(batch=stage["batch"], passed=passed, staged=len(paths)))
        print(json.dumps(batches[-1]), flush=True)
    result = dict(
        status="AUDITED_NOT_PUBLISHED",
        accepted=accepted,
        held=held,
        batches=batches,
        seconds=time.perf_counter() - started,
        peakRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_write_json(OUT / "replay-audit.json", result)
    print(json.dumps(dict(accepted=len(accepted), held=len(held))), flush=True)


def select():
    audit_result = load(OUT / "replay-audit.json")
    manual_holds = load(OUT / "manual-holds.json")
    pools = (defaultdict(list), defaultdict(list))
    before_mixed = {}
    for did, receipt in audit_result["accepted"].items():
        if did in manual_holds:
            assert receipt["stagedSha256"] == manual_holds[did]["stagedSha256"]
            continue
        before = load(OUT / "staged-cargo" / f"{did}.json")["before"]
        before_mixed[did] = features(before)["mixed"]
        lower_priority = before_mixed[did] and not receipt["features"]["mixed"]
        pools[int(lower_priority)][receipt["parent"]].append(did)
    for pool in pools:
        for ids in pool.values():
            ids.sort(
                key=lambda did: (
                    -audit_result["accepted"][did]["features"]["containers"],
                    digest(did),
                )
            )
    if sum(len(ids) for pool in pools for ids in pool.values()) < 400:
        raise ValueError("fewer than 400 fully accepted cargo proposals")
    chosen = []
    # Prefer preserving each existing mix. Any remaining conversions must be
    # compensated by new mixtures in this same, explicitly checked cohort.
    for pool in pools:
        while len(chosen) < 400 and any(pool.values()):
            for sid in sorted(pool, key=digest):
                if pool[sid] and len(chosen) < 400:
                    chosen.append(pool[sid].pop(0))
    mixed = sum(audit_result["accepted"][did]["features"]["mixed"] for did in chosen)
    assert 0 < mixed < len(chosen), "refresh must include mixed and uniform equipment"
    assert mixed >= sum(before_mixed[did] for did in chosen), "mixed coverage must not regress"
    atomic_write_json(
        OUT / "selection.json",
        dict(
            documents=chosen,
            auditSha256=sha256_file(OUT / "replay-audit.json"),
            manualHoldsSha256=sha256_file(OUT / "manual-holds.json"),
            parents=dict(Counter(audit_result["accepted"][d]["parent"] for d in chosen)),
        ),
    )
    print(
        json.dumps(
            dict(
                selected=len(chosen),
                mixed=mixed,
                parents=len({audit_result["accepted"][d]["parent"] for d in chosen}),
            )
        )
    )


def replacements():
    from refresh_r29_long_sources import identifier_contract, positioned_overlay, render_identifiers

    from document_ocr.synthesis.container_prefix_refresh import refresh_prefix

    selection, audit_result = load(OUT / "selection.json"), load(OUT / "replay-audit.json")
    assert selection["auditSha256"] == sha256_file(OUT / "replay-audit.json")
    assert selection["manualHoldsSha256"] == sha256_file(OUT / "manual-holds.json")
    assert not set(selection["documents"]) & load(OUT / "manual-holds.json").keys()
    rows = {}
    for did in selection["documents"]:
        path = OUT / "staged-cargo" / f"{did}.json"
        assert sha256_file(path) == audit_result["accepted"][did]["stagedSha256"]
        rows[did] = load(path)["after"]
    excluded = {
        c["equipmentIdentifier"]
        for name in ("synthetic-train.jsonl", "real-train.jsonl", "validation.jsonl")
        for r in map(json.loads, (SOURCE / name).open())
        for c in r["target"].get("containerInformation", [])
    }
    for item in load(OUT / "prefix-summary.json")["rows"]:
        did = item["id"]
        assert did not in rows
        record = load(OUT / "staged-prefix" / f"{did}.json")
        after, receipt = refresh_prefix(
            record["before"], prefix=item["prefix"], seed=2026110229, excluded=excluded
        )
        assert after == record["after"]
        assert receipt["spans"] == record["evidence"]["spans"]
        assert record["evidence"]["allPrintedFootprintsNonexpanding"]
        excluded.update(receipt["identifiers"].values())
        rows[did] = after
    for item in load(OUT / "legacy-summary.json")["rows"]:
        did = item["documentId"]
        assert did not in rows
        record = load(OUT / "staged-legacy" / f"{did}.json")
        assert record["before"]["target"] == record["after"]["target"]
        assert (
            apply_edits(
                record["before"]["joinedRawText"].encode(),
                record["evidence"]["currentProof"]["edits"],
            ).decode()
            == record["after"]["joinedRawText"]
        )
        rows[did] = record["after"]
    originals = {r["documentId"]: r for r in map(json.loads, (SOURCE / "real-train.jsonl").open())}
    for item in load(OUT / "long-source-summary.json")["rows"]:
        did = item["id"]
        assert did not in rows
        record = load(OUT / "staged-long" / f"{did}.json")
        assert digest(record) == item["stagedSha256"]
        source = originals[item["parent"]]
        assert source == record["source"]
        contract = identifier_contract(source)
        assert digest(contract.model_dump()) == record["evidence"]["contractSha256"]
        raw, target, _ = render_identifiers(source, contract, record["evidence"]["values"])
        assert raw == record["after"]["joinedRawText"] and target == record["after"]["target"]
        assert positioned_overlay(source, raw) == record["after"]["positionedText"]
        assert features(record["before"])["complete"] and not features(record["before"])["equal"]
        assert features(record["after"])["equal"]
        rows[did] = record["after"]
    return rows


def prepare():
    hashes = load(OUT / "r28-protected-hashes.json")
    assert all(sha256_file(DATA / n) == pin == sha256_file(SOURCE / n) for n, pin in hashes.items())
    selected, counts = replacements(), {}
    originals = {
        r["documentId"]: r
        for r in map(json.loads, (SOURCE / SHARDS[0]).open())
        if r["documentId"] in selected
    }
    publication = OUT / "publication"
    for name in SHARDS:
        parts, changed = [], 0
        for line in (SOURCE / name).open("rb"):
            row = json.loads(line)
            did = row["documentId"]
            if did in selected:
                assert row == originals[did], "diverged sidecar"
                line = (
                    json.dumps(selected[did], ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
                changed += 1
            parts.append(line)
        atomic_write_bytes(publication / name, b"".join(parts))
        counts[name] = changed
    header = """# R29 intermediary equal-load refresh

R29 contains 9,400 synthetic and 600 anonymized real training records, plus the
unchanged 60-document validation set. It is an intermediary copy of frozen R28;
no R29 training run is scheduled. Paddle conversion is separate and not applied.

400 older synthetic records received independently sampled equal-load cargo,
24 older equal-load inventories received a shared four-letter container prefix,
and two stale source-only net-tonnage repetitions were corrected. Six further
older records were replaced by explicitly source-preserving identifier variants
from two anonymized real long-inventory sources (11 and 40 containers). Those six
are not donor-free and retain their source cargo, parties, routes and dates.
For the other refreshes, current private parties, routes, dates and unrelated
text are retained. Cargo refresh preserves
container/seal IDs; the separate prefix cohort changes container IDs and links.
All overlapping UAE sidecars are synchronized; they must not be concatenated.

Report: `docs/analysis/r29-equal-load-refresh-20261102/REPORT.md`.
The inherited R28 history below is historical, not the current membership count.

## Inherited R28 history

"""
    atomic_write_bytes(
        publication / "README.md", (header + (SOURCE / "README.md").read_text()).encode()
    )
    manifest = load(SOURCE / "manifest.json")
    manifest.update(
        baseDataset=str(SOURCE.relative_to(ROOT)), baseManifestSha256=hashes["manifest.json"]
    )
    manifest["status"] = "r29_equal_load_refresh_audited_intermediary10000"
    manifest["equalLoadRefresh20261102"] = dict(
        cargo=400,
        prefixOnly=24,
        legacyTonnage=2,
        sourcePreservingLong=6,
        records=len(selected),
        changedCopies=counts,
        intermediaryNotTraining=True,
        report=str((OUT / "REPORT.md").relative_to(ROOT)),
        selectionSha256=sha256_file(OUT / "selection.json"),
    )
    for name in (*SHARDS, "README.md"):
        manifest["files"][name] = sha256_file(publication / name)
    atomic_write_json(publication / "manifest.json", manifest)
    atomic_write_json(
        OUT / "publication-preflight.json",
        dict(
            changedCopies=counts,
            files={
                n: sha256_file(publication / n) for n in (*SHARDS, "README.md", "manifest.json")
            },
        ),
    )
    print(json.dumps(counts))


def verify(prepared=False):
    started = time.perf_counter()
    location = OUT / "publication" if prepared else DATA
    selected, authority = replacements(), task()
    counts, changed = Counter(), Counter()
    before_rows, after_rows = [], []
    for name in SHARDS:
        with (SOURCE / name).open() as old, (location / name).open() as new:
            for a, b in zip(old, new, strict=True):
                before, after = json.loads(a), json.loads(b)
                did = before["documentId"]
                assert after["documentId"] == did
                counts[name] += 1
                if did in selected:
                    assert after == selected[did]
                    changed[name] += 1
                else:
                    assert a == b, "unselected row changed"
                assert authority.canonicalize(after["target"]) == after["target"], did
                for field in ("joinedRawText", "positionedText"):
                    assert digest(after[field].encode()) == after[field + "Sha256"]
                if name == SHARDS[0]:
                    before_rows.append(before)
                    after_rows.append(after)
    assert len(after_rows) == 9400
    protected = load(OUT / "r28-protected-hashes.json")
    assert all(sha256_file(SOURCE / n) == pin for n, pin in protected.items())
    assert all(
        sha256_file(DATA / n) == protected[n] for n in ("real-train.jsonl", "validation.jsonl")
    )
    for name, pin in load(location / "manifest.json")["files"].items():
        assert sha256_file((location if (location / name).is_file() else DATA) / name) == pin
    before_stats, after_stats = census(before_rows), census(after_rows)
    for key in ("dg", "thermal", "forwarding", "transshipment"):
        assert before_stats["flags"].get(key, 0) == after_stats["flags"].get(key, 0), key
    assert after_stats["flags"]["mixedEquipment"] >= before_stats["flags"]["mixedEquipment"]
    old, new = allocation_stats(before_rows), allocation_stats(after_rows)
    assert new["counts"]["completeMulti"] == old["counts"]["completeMulti"]
    assert new["counts"]["equalMulti"] - old["counts"]["equalMulti"] == 406
    assert new["equalFraction"] >= 0.3
    assert new["counts"]["eightPlusEqualSharedPrefix"] >= 24
    result = dict(
        status="PASS",
        selectionSha256=sha256_file(OUT / "selection.json"),
        publicationFiles=load(OUT / "publication-preflight.json")["files"],
        counts=counts,
        changedCopies=changed,
        before=before_stats,
        after=after_stats,
        allocationsBefore=old,
        allocationsAfter=new,
        seconds=time.perf_counter() - started,
        peakRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_write_json(
        OUT / ("prepared-validation.json" if prepared else "final-readback.json"), result
    )
    print(json.dumps(dict(status="PASS", allocationsAfter=new)))


def tokens():
    from transformers import AutoTokenizer

    from document_ocr.training.tasks import canonical_json

    tokenizer = AutoTokenizer.from_pretrained(
        "google/t5gemma-2-270m-270m",
        revision="7c38f16641f455ef0685b18431faf1b17722d5a1",
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    rows = list(replacements().values())
    lengths = dict(input=[], target=[])
    for start in range(0, len(rows), 50):
        batch = rows[start : start + 50]
        lengths["input"].extend(
            map(
                len,
                tokenizer([r["positionedText"] for r in batch], truncation=False, padding=False)[
                    "input_ids"
                ],
            )
        )
        lengths["target"].extend(
            len(ids) + 1
            for ids in tokenizer(
                text_target=[canonical_json(r["target"], pretty=False) for r in batch],
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
        )
    assert max(lengths["input"]) <= 19200 and max(lengths["target"]) <= 5500
    atomic_write_json(
        OUT / "token-validation.json",
        dict(
            status="PASS",
            records=len(rows),
            replacementsSha256=digest(rows),
            maxInput=max(lengths["input"]),
            maxTargetWithEOS=max(lengths["target"]),
            targetsAboveCurrent3072EvalCap=[
                dict(id=row["documentId"], tokens=count)
                for row, count in zip(rows, lengths["target"], strict=True)
                if count > 3072
            ],
        ),
    )


def publish():
    from xml.etree import ElementTree

    for name in ("prepared-validation.json", "token-validation.json"):
        assert load(OUT / name)["status"] == "PASS"
    suites = ElementTree.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites)
    expected = load(OUT / "publication-preflight.json")["files"]
    verified = load(OUT / "prepared-validation.json")
    assert verified["selectionSha256"] == sha256_file(OUT / "selection.json")
    assert verified["publicationFiles"] == expected
    token_check = load(OUT / "token-validation.json")
    assert token_check["replacementsSha256"] == digest(list(replacements().values()))
    promote_files(
        DATA, OUT / "publication", SOURCE, load(OUT / "r28-protected-hashes.json"), expected
    )
    atomic_write_json(OUT / "publication-state.json", dict(status="COMPLETE", files=expected))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["audit", "select", "prepare", "verify", "tokens", "publish"]
    )
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.prepared)
    else:
        globals()[args.action]()
