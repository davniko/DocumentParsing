"""Replay, audit and explicitly publish the bounded 100+200 R28 refresh."""

import argparse
import copy
import json
import re
import resource
import time
from collections import Counter, defaultdict
from decimal import Decimal
from xml.etree import ElementTree

from finalize_r28_expansion import census
from refresh_r28_package_aliases import promote_files
from stage_combination_candidates import position_preservation, validate_literals
from stage_combination_refresh import OUT, ROOT

from document_ocr.atomic import atomic_write_bytes, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.declared_packages import DeclaredPackages
from document_ocr.synthesis.scoped_cargo_refresh import (
    apply_edits,
    cargo_auxiliary_keys,
    cargo_surface_keys,
    migrate,
    reconstruct_source,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"
SHARDS = ("synthetic-train.jsonl", "previous-uae-synthetic.jsonl", "old-uae-addition.jsonl")
RARE = (
    "PACKAGE_RACK",
    "PACKAGE_CASE",
    "PACKAGE_REEL",
    "PACKAGE_BAG_SUPER_BULK",
    "PACKAGE_PARCEL",
    "PACKAGE_SLAB",
    "PACKAGE_BLOCK",
    "PACKAGE_FLEXIBAG",
    "PACKAGE_FLEXITANK",
    "PACKAGE_BULK_LIQUID",
)
OWNER = "Owner: Al Ghurair Centre LLC"
OWNER_SOURCE = "doc_35136b22922177501a794351a470cce57d8873a40338c10058abf1712f0459e8"


def read(path):
    return json.loads(path.read_text())


def task():
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        (DATA / "task-constraints.json").read_bytes()
    )
    return get_training_task(constraints.task).bind_constraints(constraints)


def categories(row):
    return {
        p.get("typeCategory")
        for g in row["target"]["goodsItemDetails"]
        for p in g["numberAndTypeOfPackages"]
    }


def mixed_multi(row):
    containers = row["target"].get("containerInformation", [])
    equipment = {
        (c.get("sizeCategory"), c.get("typeCategory"))
        for c in containers
        if c.get("sizeCategory") or c.get("typeCategory")
    }
    return len(equipment) > 1 and any(
        len(g.get("numberAndTypeOfPackages", [])) > 1
        for g in row["target"].get("goodsItemDetails", [])
    )


def audit():
    started = time.perf_counter()
    authority = task()
    scoped = read(OUT / "scoped-owner-reviews.json")
    grouped = defaultdict(list)
    for path in sorted((OUT / "staged-cargo").glob("*.json")):
        row = read(path)
        grouped[row["evidence"]["preparedConfig"]].append((path, row))
    accepted, held = {}, {}
    approvals_path = OUT / "reviewed-hybrid-decisions.json"
    approvals = read(approvals_path) if approvals_path.exists() else {}
    main_holds = read(OUT / "main-wording-holds.json")
    for config, items in grouped.items():
        cfg = read(ROOT / config)
        campaign = create_campaign(ROOT, cfg)
        auxiliary = read(ROOT / cfg["auxiliary"])["sources"]
        for path, row in items:
            did = row["after"]["documentId"]
            try:
                assert did not in main_holds, "main wording review: " + main_holds.get(did, "")
                evidence = row["evidence"]
                native = read(ROOT / evidence["nativePath"])
                assert digest(native) == evidence["nativeSha256"], "native candidate changed"
                campaign.replay_candidate(native)
                sid, variant = native["sourceDocumentId"], native["variant"]
                bp, scenario, _, target = campaign.plan(sid, variant)
                physical = campaign.physical_render(bp, scenario, target)
                keys = set(physical.surface_values) | set(physical.variable_values)
                keys |= {r.key for r in bp.regions if r.kind == "product"}
                keys |= cargo_auxiliary_keys(auxiliary.get(sid, {}).get("bindings", {}))
                keys |= cargo_surface_keys(bp)
                keys |= set(scoped.get(sid, {}).get("keys", []))
                expected, proof = migrate(
                    row["before"],
                    read(ROOT / evidence["historicalPath"]),
                    native,
                    private_keys=keys,
                    regions=bp.regions,
                )
                for field in ("joinedRawText", "target"):
                    assert expected[field] == row["after"][field], (
                        "complete ownership replay differs: " + field
                    )
                assert proof["currentProof"] == evidence["currentProof"], (
                    "current edit proof changed"
                )
                assert proof["geometryProof"] == evidence["geometryProof"], (
                    "geometry ownership proof changed"
                )
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
                assert mixed_multi(row["after"])
                assert not scenario.provenance["independentDraw"]["provenance"][
                    "sourceShipmentDependence"
                ]
                review = read(OUT / "hybrid-reviews" / path.name)
                assert review["afterSha256"] == digest(row["after"]), "hybrid review stale"
                if review["findings"]:
                    approval = approvals.get(did)
                    assert approval and approval["reviewSha256"] == digest(review), (
                        "hybrid findings require main adjudication"
                    )
                accepted[did] = dict(
                    stagedSha256=sha256_file(path),
                    reviewSha256=digest(review),
                    nativeSha256=digest(native),
                    physicalOwnerCount=len(physical.surface_values) + len(physical.variable_values),
                )
            except (AssertionError, ValueError, KeyError, FileNotFoundError) as error:
                held[did] = str(error)
        print(config, "cumulative pass", len(accepted), "held", len(held), flush=True)
    result = dict(
        status="AUDITED_NOT_PUBLISHED",
        accepted=accepted,
        held=held,
        seconds=time.perf_counter() - started,
        peakRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_write_json(OUT / "cargo-replay-audit.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("accepted", "held")}), flush=True
    )


def select():
    audit_result = read(OUT / "cargo-replay-audit.json")
    pool = {}
    same_type_repetitions = []
    for did, receipt in audit_result["accepted"].items():
        path = OUT / "staged-cargo" / f"{did}.json"
        assert sha256_file(path) == receipt["stagedSha256"]
        row = read(path)
        if len(categories(row["after"])) < 2:
            same_type_repetitions.append(did)
        else:
            pool[did] = row
    if len(pool) < 200:
        raise ValueError(f"only {len(pool)} audited cargo replacements")
    # Prefer scarce categories and diverse parents; this selects among accepted
    # plausible proposals, never forces a package onto an unsuitable commodity.
    baseline = census([json.loads(line) for line in (OUT / "before/synthetic-train.jsonl").open()])
    counts = Counter(baseline["packageDocumentCounts"])
    parents, selected = Counter(), []
    while len(selected) < 200:

        def score(did):
            row = pool[did]["after"]
            gain = sum(1 / (counts[c] + 1) for c in categories(row) if c in RARE)
            return gain / (1 + parents[row["sourceDocumentId"]] * 0.15), digest(["selection", did])

        did = max(pool, key=score)
        row = pool.pop(did)
        selected.append(did)
        parents[row["after"]["sourceDocumentId"]] += 1
        counts.update(categories(row["after"]))
    atomic_write_json(
        OUT / "cargo-selection.json",
        dict(
            documents=selected,
            parents=dict(parents),
            auditSha256=sha256_file(OUT / "cargo-replay-audit.json"),
            unselectedAccepted=sorted(pool),
            validSameTypeRepetitionsNotSelected=same_type_repetitions,
            scope=(
                "Each of these 200 cargo replacements has at least two distinct public package "
                "categories, not merely repeated rows of the same type."
            ),
        ),
    )
    print("Selected 200 cargo replacements across", len(parents), "parents", flush=True)


def omit_owner(row):
    """Delete only the approved line's text; keep all other line anchors exactly."""
    result = copy.deepcopy(row)
    raw, positioned = (
        row["joinedRawText"].splitlines(keepends=True),
        row["positionedText"].splitlines(keepends=True),
    )
    assert len(raw) == len(positioned)
    found = []
    for i, line in enumerate(raw):
        if OWNER not in line:
            continue
        assert line.strip() == OWNER and row["sourceDocumentId"] == OWNER_SOURCE
        assert OWNER not in json.dumps(row["target"])
        assert positioned[i].startswith(line.rstrip("\r\n"))
        found.append(i)
        raw[i] = "\n" if line.endswith("\n") else ""
        positioned[i] = " ||\n" if positioned[i].endswith("\n") else " ||"
    assert len(found) == 1
    result.update(joinedRawText="".join(raw), positionedText="".join(positioned))
    for field in ("joinedRawText", "positionedText"):
        result[field + "Sha256"] = digest(result[field].encode())
    return result, dict(
        documentId=row["documentId"],
        lines=found,
        beforeSha256=digest(row),
        afterSha256=digest(result),
    )


def replacements():
    selected = read(OUT / "cargo-selection.json")
    audit_result = read(OUT / "cargo-replay-audit.json")
    assert selected["auditSha256"] == sha256_file(OUT / "cargo-replay-audit.json")
    rows = {}
    equipment = sorted((OUT / "staged-equipment").glob("*.json"))
    assert len(equipment) == 100 and len(selected["documents"]) == 200
    for path in equipment + [OUT / "staged-cargo" / f"{did}.json" for did in selected["documents"]]:
        staged = read(path)
        did = staged["after"]["documentId"]
        assert did not in rows
        if path.parent.name == "staged-cargo":
            assert sha256_file(path) == audit_result["accepted"][did]["stagedSha256"]
        else:
            proof = staged["evidence"]["proof"]
            assert (
                apply_edits(staged["before"]["joinedRawText"].encode(), proof["edits"]).decode()
                == staged["after"]["joinedRawText"]
            )
            assert (
                staged["before"]["target"]["goodsItemDetails"]
                == staged["after"]["target"]["goodsItemDetails"]
            )
            DeclaredPackages.from_receipt(staged["evidence"]["ledger"])
        assert mixed_multi(staged["after"])
        rows[did] = staged["after"]
    cleanups = []
    for line in (OUT / "before/synthetic-train.jsonl").open():
        before = json.loads(line)
        if OWNER in before["joinedRawText"]:
            did = before["documentId"]
            after, receipt = omit_owner(rows.get(did, before))
            receipt["overlapsRefresh"] = did in rows
            rows[did] = after
            cleanups.append(receipt)
    assert len(cleanups) == 10
    return rows, cleanups


def prepare():
    started = time.perf_counter()
    hashes = read(OUT / "before/hashes.json")
    for name, expected in hashes.items():
        assert sha256_file(DATA / name) == expected, name + " changed"
    selected, cleanups = replacements()
    originals = {
        r["documentId"]: r
        for line in (OUT / "before/synthetic-train.jsonl").open()
        if (r := json.loads(line))["documentId"] in selected
    }
    publication = OUT / "publication"
    publication.mkdir(exist_ok=True)
    counts = {}
    for name in SHARDS:
        parts, changed = [], 0
        for line in (DATA / name).open("rb"):
            row = json.loads(line)
            did = row["documentId"]
            if did in selected:
                assert row == originals[did], (name, did, "diverged sidecar")
                line = (
                    json.dumps(selected[did], ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
                changed += 1
            parts.append(line)
        atomic_write_bytes(publication / name, b"".join(parts))
        counts[name] = changed
    atomic_write_json(
        OUT / "legacy-owner-cleanup.json",
        dict(
            userApproved=True,
            documents=cleanups,
            authority="artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v21/ownership.json",
            scope=(
                "Existing reviewed source-only omission. No extraction labels changed. "
                "No new source/template rule."
            ),
        ),
    )
    readme = (
        (DATA / "README.md").read_text()
        + """

## Mixed equipment and public package-row refresh — 2026-11-01

Exactly 300 existing records were replaced in place: 100 equipment-only changes
on existing multi-row records, and 200 donor-free cargo replacements on older
single-row records. All 300 now combine mixed equipment and multiple public
package rows. Current parties, routes, dates, private identifiers and unrelated
text were retained. Matching sidecar copies were synchronized. No membership
changes; real training and validation remain unchanged.

Separately, an existing reviewed source-only private owner omission was propagated
to 10 current synthetic records, without target changes. See the report for overlap.
Evidence, exact replay, scoped/native text reviews, geometry, before copies,
publication and full readback: `docs/analysis/r28-combination-refresh300-20261101/REPORT.md`.
This certifies the bounded refresh, not a new unrestricted full-corpus audit.
"""
    )
    atomic_write_bytes(publication / "README.md", readme.encode())
    manifest = read(DATA / "manifest.json")
    manifest["combinationRefresh20261101"] = dict(
        equipmentOnly=100,
        donorFreeCargo=200,
        legacyOwnerOmissions=10,
        uniqueRecords=len(selected),
        changedCopies=counts,
        report=str((OUT / "REPORT.md").relative_to(ROOT)),
        beforeCopies=str((OUT / "before").relative_to(ROOT)),
        selectionSha256=sha256_file(OUT / "cargo-selection.json"),
        realAndValidationUnchanged=True,
    )
    for name in (*SHARDS, "README.md"):
        manifest["files"][name] = sha256_file(publication / name)
    atomic_write_json(publication / "manifest.json", manifest)
    result = dict(
        status="PREPARED_NOT_PUBLISHED",
        changedCopies=counts,
        selectedRecords=len(selected),
        files={n: sha256_file(publication / n) for n in (*SHARDS, "README.md", "manifest.json")},
        seconds=time.perf_counter() - started,
        peakRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_write_json(OUT / "publication-preflight.json", result)
    print(json.dumps(result), flush=True)


def verify(prepared=False):
    started = time.perf_counter()
    authority = task()
    location = OUT / "publication" if prepared else DATA
    selected, _ = replacements()
    counts, changes = Counter(), Counter()
    before_train, after_train = [], []
    for name in SHARDS:
        with (OUT / "before" / name).open() as old, (location / name).open() as new:
            for a, b in zip(old, new, strict=True):
                before, after = json.loads(a), json.loads(b)
                did = before["documentId"]
                assert after["documentId"] == did
                counts[name] += 1
                if did in selected:
                    assert after == selected[did]
                    changes[name] += 1
                else:
                    assert a == b, (name, did, "unselected byte change")
                assert authority.canonicalize(after["target"]) == after["target"], did
                for field in ("joinedRawText", "positionedText"):
                    assert digest(after[field].encode()) == after[field + "Sha256"], (did, field)
                assert OWNER not in after["joinedRawText"]
                if name == SHARDS[0]:
                    before_train.append(before)
                    after_train.append(after)
    for name in ("real-train.jsonl", "validation.jsonl"):
        assert sha256_file(DATA / name) == read(OUT / "before/hashes.json")[name]
    manifest = read(location / "manifest.json")
    for name, expected in manifest["files"].items():
        assert (
            sha256_file((location if name in (*SHARDS, "README.md") else DATA) / name) == expected
        ), name
    frozen = ROOT / manifest["baseDataset"]
    for name, expected in read(frozen / "manifest.json")["files"].items():
        assert sha256_file(frozen / name) == expected, ("R27", name)
    before_stats, after_stats = census(before_train), census(after_train)
    real = [json.loads(line) for line in (DATA / "real-train.jsonl").open()]
    all_stats = census(after_train + real)
    for key in ("dg", "thermal", "forwarding", "transshipment"):
        assert before_stats["flags"].get(key, 0) == after_stats["flags"].get(key, 0), key
    assert len(after_train) == 9400 and len(real) == 600
    assert sum(map(mixed_multi, after_train)) - sum(map(mixed_multi, before_train)) == 300
    result = dict(
        status="PASS",
        prepared=prepared,
        counts=counts,
        changedCopies=changes,
        before=before_stats,
        after=after_stats,
        finalTraining=all_stats,
        syntheticMixedMultiBefore=sum(map(mixed_multi, before_train)),
        syntheticMixedMultiAfter=sum(map(mixed_multi, after_train)),
        seconds=time.perf_counter() - started,
        peakRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_write_json(
        OUT / ("prepared-validation.json" if prepared else "final-readback.json"), result
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("before", "after", "finalTraining")}
        ),
        flush=True,
    )


def tokens():
    from transformers import AutoTokenizer

    from document_ocr.training.tasks import canonical_json

    tokenizer = AutoTokenizer.from_pretrained(
        "google/t5gemma-2-270m-270m",
        revision="7c38f16641f455ef0685b18431faf1b17722d5a1",
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    rows, _ = replacements()
    lengths = {"input": [], "target": []}
    records = list(rows.values())
    started = time.perf_counter()
    for i in range(0, len(records), 50):
        batch = records[i : i + 50]
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
            records=len(records),
            maxInput=max(lengths["input"]),
            maxTargetWithEOS=max(lengths["target"]),
            inputLimit=19200,
            targetLimit=5500,
            seconds=time.perf_counter() - started,
        ),
    )
    print("Token limits PASS", max(lengths["input"]), max(lengths["target"]), flush=True)


def publish():
    for name in ("prepared-validation.json", "token-validation.json"):
        assert read(OUT / name)["status"] == "PASS"
    suites = ElementTree.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert suites and all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites)
    assert sum(int(s.attrib["tests"]) for s in suites) >= 440
    info = read(OUT / "publication-preflight.json")
    promote_files(
        DATA, OUT / "publication", OUT / "before", read(OUT / "before/hashes.json"), info["files"]
    )
    atomic_write_json(OUT / "publication-state.json", dict(status="COMPLETE", files=info["files"]))
    print("Publication COMPLETE", flush=True)


def summary():
    final = read(OUT / "final-readback.json")
    assert final["status"] == "PASS"
    selection = read(OUT / "cargo-selection.json")
    hs, before_hs, parents = set(), set(), Counter()
    packages, package_parents = Counter(), defaultdict(set)
    batches, examples = [], []
    for offset in range(0, 200, 50):
        items = []
        for did in selection["documents"][offset : offset + 50]:
            path = OUT / "staged-cargo" / f"{did}.json"
            row = read(path)
            native = read(ROOT / row["evidence"]["nativePath"])
            hs.update(g["hs6"] for g in native["scenario"]["goods_identities"] if g.get("hs6"))
            before_hs.update(
                re.sub(r"\D", "", c)[:6]
                for g in row["before"]["target"]["goodsItemDetails"]
                for c in g.get("hsCodes", [])
            )
            parents[row["after"]["sourceDocumentId"]] += 1
            for c in categories(row["after"]):
                packages[c] += 1
                package_parents[c].add(row["after"]["sourceDocumentId"])
            a = {
                x["equipmentIdentifier"]: "packageQuantity" in x
                for x in row["before"]["target"]["goodsItemDetails"][0].get(
                    "splitGoodsPlacement", []
                )
            }
            b = {
                x["equipmentIdentifier"]: "packageQuantity" in x
                for x in row["after"]["target"]["goodsItemDetails"][0].get(
                    "splitGoodsPlacement", []
                )
            }
            assert a == b, "allocation visibility changed"
            items.append(
                dict(
                    documentId=did, stagedSha256=sha256_file(path), afterSha256=digest(row["after"])
                )
            )
            if len(examples) < 6:
                examples.append(
                    dict(
                        documentId=did,
                        description=row["after"]["target"]["goodsItemDetails"][0]["description"],
                        packages=row["after"]["target"]["goodsItemDetails"][0][
                            "numberAndTypeOfPackages"
                        ],
                        equipment=row["after"]["target"]["containerInformation"],
                    )
                )
        result = dict(
            status="PASS",
            documents=items,
            scope=(
                "Exact native/physical/current-scope replay, native and hybrid text review, "
                "literal descriptions/HS, measured geometry, schema, frozen context, "
                "quantities and presence."
            ),
        )
        dest = OUT / "batches" / f"cargo-{offset // 50 + 1:02d}.json"
        atomic_write_json(dest, result)
        batches.append(dict(path=str(dest.relative_to(ROOT)), sha256=sha256_file(dest)))
    calls, stages, cost = Counter(), Counter(), Decimal(0)
    seen = set()
    art = ROOT / "artifacts/kie-synthesis-production/r28-combination-refresh300-20261101"
    for path in art.rglob("*.json"):
        if "calls" not in path.parts:
            continue
        receipt = read(path)
        if "costUsd" not in receipt:
            continue
        usage = receipt.get("usage", {})
        ids = tuple(usage.get("providerResponseIds", []))
        key = ids if ids else (str(path),)
        if key in seen:
            continue
        seen.add(key)
        cost += Decimal(receipt["costUsd"] or "0")
        stages[receipt["stage"]] += 1
        calls["receipts"] += 1
        calls["errors"] += bool(receipt.get("error"))
        for k in ("inputTokens", "outputTokens", "reasoningTokens", "requests"):
            calls[k] += usage.get(k, 0)
    result = dict(
        status="PUBLISHED_AND_READBACK_VERIFIED",
        cargoDocuments=200,
        cargoParents=len(parents),
        maxCargoPerParent=max(parents.values()),
        freshSampledHS6=len(hs),
        oldPrintedHS6Prefixes=len(before_hs),
        newCargoPackageDocuments=dict(packages),
        newCargoPackageParents={c: len(v) for c, v in package_parents.items()},
        batches=batches,
        examples=examples,
        billed=dict(
            usd=str(cost),
            usage=dict(calls),
            stages=dict(stages),
            scope=(
                "All unique campaign receipt IDs, including nonselected drafts, corrections "
                "and reviews. Excludes earlier-turn exploratory probes. No new calls for "
                "100 equipment-only changes or owner cleanup."
            ),
        ),
        regressionNotes=[
            (
                "Six ambient nonthermal reefer records now use general-purpose equipment; "
                "thermal targets and counts are unchanged."
            ),
            (
                "Container count and allocation quantity-presence are preserved in all "
                "200 cargo replacements."
            ),
            (
                "Mixed equipment counts distinct labeled size/type combinations; does not mean "
                "every document mixes dry and reefer equipment."
            ),
        ],
    )
    atomic_write_json(OUT / "completion-summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("examples", "batches")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "audit",
            "select",
            "prepare",
            "verify-prepared",
            "tokens",
            "publish",
            "verify",
            "summary",
        ),
    )
    action = parser.parse_args().action
    if action == "verify-prepared":
        verify(prepared=True)
    else:
        globals()[action]()
