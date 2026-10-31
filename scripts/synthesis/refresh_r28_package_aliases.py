"""Stage, batch-audit and publish a bounded alias-only refresh of working R28.

The immutable before copy and audited stage are required for publication. No
historical candidate is a publication record; current repaired R28 is authoritative.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_reflow_policy import FontMetrics
from document_ocr.synthesis.package_alias_refresh import apply_reviewed_cleanup, owners, refresh
from document_ocr.synthesis.package_presentations import (
    PackagePresentationPolicy,
    PackagePresentations,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"
OUT = ROOT / "docs/analysis/r28-alias-refresh3000-20261031"
SHARDS = ("synthetic-train.jsonl", "previous-uae-synthetic.jsonl", "old-uae-addition.jsonl")
SEED = 2026103103


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_rows(path):
    return {r["documentId"]: r for r in map(json.loads, path.open())}


def load():
    config = json.loads((ROOT / "configs/synthesis/mpci_bl_curated_v7_multirow.json").read_text())
    policy = PackagePresentationPolicy.model_validate(config["aliases"]["presentations"])
    font = config["positions"]["reflow"]
    metrics = FontMetrics(Path(font["font_path"]), font["font_sha256"])
    presenter = PackagePresentations(ROOT, policy, metrics)
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        (DATA / "task-constraints.json").read_bytes()
    )
    task = get_training_task(constraints.task).bind_constraints(constraints)
    return presenter, task


def stage():
    if (OUT / "stage.json").exists():
        raise ValueError("stage already exists; do not overwrite reviewed evidence")
    started = time.perf_counter()
    presenter, task = load()
    hashes = {p.name: sha256_file(p) for p in DATA.iterdir() if p.is_file()}
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert all(hashes[name] == pin for name, pin in manifest["files"].items())
    before = OUT / "before"
    before.mkdir(parents=True, exist_ok=True)
    for name in (*SHARDS, "manifest.json", "README.md"):
        if not (before / name).exists():
            shutil.copy2(DATA / name, before / name)
        assert sha256_file(before / name) == hashes[name]
    rows = read_rows(DATA / SHARDS[0])
    index = json.loads(
        (
            ROOT
            / "docs/analysis/r28-quality-and-variability-20261019/donor/per-record-inventory.json"
        ).read_text()
    )
    assert {r["documentId"] for r in index} == rows.keys()
    counts, failures, eligible = Counter(), [], []
    for n, item in enumerate(index, 1):
        row = rows[item["documentId"]]
        candidate = json.loads((ROOT / item["candidatePath"]).read_text())
        try:
            slots = owners(candidate, row, presenter.certificates[row["sourceDocumentId"]])
            updated, receipt = refresh(row, slots, presenter, seed=SEED)
            if task.canonicalize(updated["target"]) != updated["target"]:
                raise ValueError("noncanonical_refreshed_target")
            if not receipt["proof"]["edits"]:
                counts["unchanged_alias_draw"] += 1
                continue
        except ValueError as error:
            reason = str(error)
            counts[reason] += 1
            failures.append(
                dict(
                    documentId=row["documentId"],
                    sourceDocumentId=row["sourceDocumentId"],
                    reason=reason,
                )
            )
            continue
        counts["eligible_changed"] += 1
        receipt["candidatePath"] = item["candidatePath"]
        receipt["candidateSha256"] = sha256_file(ROOT / item["candidatePath"])
        receipt["currentOwners"] = slots
        eligible.append((updated, receipt))
        if n % 500 == 0:
            print(json.dumps(dict(scanned=n, counts=counts)), flush=True)
    save(OUT / "eligibility.json", dict(counts=counts, excluded=failures))
    if len(eligible) < 3000:
        raise ValueError(f"only {len(eligible)} changed, safely traceable records; no publication")
    # Round-robin parent/category cells prioritizes layout and category coverage
    # without altering cargo facts or filling unsupported old-template slots.
    cells = defaultdict(list)
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    for updated, receipt in eligible:
        category = updated["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
            "typeCategory"
        ]
        cells[(category, updated["sourceDocumentId"])].append((updated, receipt))
    keys = sorted(cells)
    rng.shuffle(keys)
    selected = []
    while len(selected) < 3000:
        for key in keys:
            if cells[key] and len(selected) < 3000:
                selected.append(cells[key].pop())
    staged = OUT / "staged.jsonl"
    with staged.open("w") as handle:
        for updated, _ in selected:
            handle.write(json.dumps(updated, ensure_ascii=False, sort_keys=True) + "\n")
    for batch in range(30):
        save(
            OUT / "batches" / f"{batch + 1:02d}-receipts.json",
            [r for _, r in selected[batch * 100 : (batch + 1) * 100]],
        )
    save(
        OUT / "stage.json",
        dict(
            status="STAGED_NOT_PUBLISHED",
            selected=3000,
            seed=SEED,
            beforeFiles=hashes,
            stagedSha256=sha256_file(staged),
            eligibility=counts,
            seconds=time.perf_counter() - started,
            maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            selection="round_robin_category_parent_cells",
            batches=30,
            selectedIds=[r["documentId"] for r, _ in selected],
            categoryCounts=Counter(
                r["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["typeCategory"]
                for r, _ in selected
            ),
            parents=len({r["sourceDocumentId"] for r, _ in selected}),
        ),
    )
    print(
        json.dumps(dict(counts=counts, selected=3000, seconds=time.perf_counter() - started)),
        flush=True,
    )


def audit_context():
    presenter, task = load()
    stage_info = json.loads((OUT / "stage.json").read_text())
    assert sha256_file(OUT / "staged.jsonl") == stage_info["stagedSha256"]
    return (
        presenter,
        task,
        stage_info,
        read_rows(OUT / "before" / SHARDS[0]),
        read_rows(OUT / "staged.jsonl"),
    )


def audit(batch, context=None):
    started = time.perf_counter()
    presenter, task, stage_info, before, staged = context or audit_context()
    receipts_path = OUT / "batches" / f"{batch:02d}-receipts.json"
    receipts = json.loads(receipts_path.read_text())
    assert len(receipts) == 100
    styles, roles, transitions, target_changes = Counter(), Counter(), Counter(), []
    review = []
    for receipt in receipts:
        did = receipt["documentId"]
        original, updated = before[did], staged[did]
        assert digest(original) == receipt["beforeRecordSha256"]
        assert digest(updated) == receipt["afterRecordSha256"]
        assert sha256_file(ROOT / receipt["candidatePath"]) == receipt["candidateSha256"]
        candidate = json.loads((ROOT / receipt["candidatePath"]).read_text())
        slots = owners(candidate, original, presenter.certificates[original["sourceDocumentId"]])
        assert slots == receipt["currentOwners"]
        replay, evidence = refresh(original, slots, presenter, seed=SEED)
        assert replay == updated
        assert all(evidence[k] == receipt[k] for k in evidence)
        assert task.canonicalize(updated["target"]) == updated["target"]
        # Identity controls use *current* private text, not historical candidates.
        identity, _ = refresh(
            original, [{**s, "protectedSourceDetail": True} for s in slots], presenter, seed=SEED
        )
        assert identity == original
        raw = original["joinedRawText"].encode()
        reverse = updated["joinedRawText"].encode()
        shift = 0
        changed_lines = []
        for edit in receipt["proof"]["edits"]:
            a, b = edit["byteStart"], edit["byteEnd"]
            current_start = a + shift
            current_end = current_start + len(edit["after"].encode())
            assert reverse[current_start:current_end] == edit["after"].encode()
            shift += len(edit["after"].encode()) - (b - a)
            line = raw[:a].count(b"\n")
            changed_lines.append(
                dict(
                    line=line + 1,
                    before=original["joinedRawText"].splitlines()[line],
                    after=updated["joinedRawText"].splitlines()[line],
                )
            )
            styles[edit["style"]] += 1
            roles[edit["role"]] += 1
            transitions[(edit["category"], edit["before"], edit["after"])] += 1
        # Independent reverse replay proves every byte outside owned nouns survived.
        for edit in reversed(receipt["proof"]["edits"]):
            prior = sum(
                len(e["after"].encode()) - (e["byteEnd"] - e["byteStart"])
                for e in receipt["proof"]["edits"]
                if e["byteStart"] < edit["byteStart"]
            )
            a = edit["byteStart"] + prior
            reverse = (
                reverse[:a] + edit["before"].encode() + reverse[a + len(edit["after"].encode()) :]
            )
        assert reverse == raw
        target_changes.extend(dict(documentId=did, **c) for c in receipt["targetChanges"])
        review.append(
            dict(
                documentId=did,
                sourceDocumentId=original["sourceDocumentId"],
                changedLines=changed_lines,
                targetChanges=receipt["targetChanges"],
            )
        )
    save(OUT / "batches" / f"{batch:02d}-review.json", review)
    result = dict(
        status="AUTOMATED_PASS_REQUIRES_SCOPED_REVIEW",
        documents=100,
        receiptSha256=sha256_file(receipts_path),
        stageSha256=stage_info["stagedSha256"],
        styles=styles,
        roles=roles,
        targetChanges=target_changes,
        transitions=[
            dict(category=k[0], before=k[1], after=k[2], occurrences=v)
            for k, v in sorted(transitions.items())
        ],
        seconds=time.perf_counter() - started,
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    save(OUT / "batches" / f"{batch:02d}-audit.json", result)
    print(
        json.dumps(
            dict(
                batch=batch,
                documents=100,
                styles=styles,
                targetChanges=len(target_changes),
                seconds=result["seconds"],
            )
        ),
        flush=True,
    )


def final_rows():
    rows = read_rows(OUT / "staged.jsonl")
    cleanups = json.loads((OUT / "reviewed-cleanups.json").read_text())
    evidence = []
    for repair in cleanups:
        did = repair["documentId"]
        rows[did], receipt = apply_reviewed_cleanup(rows[did], repair)
        evidence.append(dict(documentId=did, reason=repair["reason"], **receipt))
    return rows, evidence


def prepare_publication():
    """Validate all physical copies and construct complete shards before any write."""
    started = time.perf_counter()
    _, task = load()
    info = json.loads((OUT / "stage.json").read_text())
    assert sha256_file(OUT / "staged.jsonl") == info["stagedSha256"]
    for name, pin in info["beforeFiles"].items():
        assert sha256_file(DATA / name) == pin, name
    for batch in range(1, 31):
        audit_receipt = json.loads((OUT / "batches" / f"{batch:02d}-audit.json").read_text())
        assert audit_receipt["documents"] == 100
        assert audit_receipt["stageSha256"] == info["stagedSha256"]
        assert audit_receipt["receiptSha256"] == sha256_file(
            OUT / "batches" / f"{batch:02d}-receipts.json"
        )
    refreshed, cleanup_evidence = final_rows()
    original = read_rows(OUT / "before" / SHARDS[0])
    counts = {}
    publication = OUT / "publication"
    publication.mkdir(exist_ok=True)
    for name in SHARDS:
        changed = 0
        with (DATA / name).open() as source, (publication / name).open("w") as output:
            for line in source:
                row = json.loads(line)
                did = row["documentId"]
                if did in refreshed:
                    # Never replace a sidecar's different current facts wholesale.
                    assert row == original[did], (name, did, "diverged sidecar")
                    row = refreshed[did]
                    assert task.canonicalize(row["target"]) == row["target"], did
                    for field in ("joinedRawText", "positionedText"):
                        assert digest(row[field].encode()) == row[field + "Sha256"]
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    changed += 1
                else:
                    output.write(line)
        counts[name] = changed
    assert counts[SHARDS[0]] == 3000
    save(publication / "cleanup-validation.json", cleanup_evidence)
    readme = (
        (DATA / "README.md").read_text()
        + """

## Package-alias refresh — 2026-10-31

Exactly 3,000 existing synthetic training records received occurrence-owned package
aliases, validated in 30 batches of 100. Matching copies in both retained UAE
sidecars were updated. No records were added or removed; multi-row regeneration
remains separate. Existing current-frame line anchors are preserved for fitting
scalar aliases, without original-template reflow or new coordinate inference.
One separately reviewed generated packing-context echo was removed from one of
these records and its sidecar copies; its former line is blank, preserving all
other line anchors. Real training and validation are untouched.

Evidence: `docs/analysis/r28-alias-refresh3000-20261031/REPORT.md`, batch receipts,
scoped review, full before copies, publication/readback and explicit cleanup plan.
This is certification of the bounded changes, not a fresh full-corpus semantic audit.
"""
    )
    (publication / "README.md").write_text(readme)
    manifest = json.loads((DATA / "manifest.json").read_text())
    manifest["packageAliasRefresh20261031"] = dict(
        records=3000,
        batches=30,
        batchSize=100,
        sidecarCopies=counts,
        report=str(OUT.relative_to(ROOT) / "REPORT.md"),
        stageSha256=info["stagedSha256"],
        protectedRealAndValidationUnchanged=True,
        legacyWordingCorrections=len(cleanup_evidence),
        beforeCopies=str((OUT / "before").relative_to(ROOT)),
    )
    for name in (*SHARDS, "README.md"):
        manifest["files"][name] = sha256_file(publication / name)
    save(publication / "manifest.json", manifest)
    result = dict(
        status="PREPARED_NOT_PUBLISHED",
        changedCopies=counts,
        files={
            name: sha256_file(publication / name)
            for name in (*SHARDS, "README.md", "manifest.json")
        },
        cleanupPlanSha256=sha256_file(OUT / "reviewed-cleanups.json"),
        seconds=time.perf_counter() - started,
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    save(OUT / "publication-preflight.json", result)
    print(json.dumps(result), flush=True)


def promote_files(dataset, publication, before, old_hashes, new_hashes):
    """Replace already audited files; recover original bytes on a failed promotion."""
    for name in new_hashes:
        if sha256_file(dataset / name) != old_hashes[name]:
            raise ValueError(f"publication current file changed: {name}")
        if sha256_file(publication / name) != new_hashes[name]:
            raise ValueError(f"publication staged file changed: {name}")
    replaced = []
    try:
        for name in new_hashes:
            temporary = dataset / ("." + name + ".alias-refresh.tmp")
            shutil.copy2(publication / name, temporary)
            os.replace(temporary, dataset / name)
            replaced.append(name)
        for name, expected in new_hashes.items():
            assert sha256_file(dataset / name) == expected
    except BaseException:
        for name in replaced:
            shutil.copy2(before / name, dataset / name)
            assert sha256_file(dataset / name) == old_hashes[name]
        raise


def publish():
    from xml.etree import ElementTree

    info = json.loads((OUT / "stage.json").read_text())
    prepared = json.loads((OUT / "publication-preflight.json").read_text())
    review = json.loads((OUT / "scoped-review.json").read_text())
    assert review["status"] == "PASS" and review["batches"] == list(range(1, 31))
    assert review["stagedSha256"] == info["stagedSha256"]
    assert prepared["cleanupPlanSha256"] == sha256_file(OUT / "reviewed-cleanups.json")
    for name in ("prepared-validation.json", "token-validation.json"):
        assert json.loads((OUT / name).read_text())["status"] == "PASS"
    suites = ElementTree.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert suites and sum(int(s.attrib["tests"]) for s in suites) == 102
    assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites)
    for name, pin in info["beforeFiles"].items():
        assert sha256_file(DATA / name) == pin, name
    save(
        OUT / "publication-state.json",
        dict(status="APPLYING", old=info["beforeFiles"], new=prepared["files"]),
    )
    try:
        promote_files(
            DATA, OUT / "publication", OUT / "before", info["beforeFiles"], prepared["files"]
        )
    except BaseException:
        save(OUT / "publication-state.json", dict(status="ROLLED_BACK"))
        raise
    save(OUT / "publication-state.json", dict(status="COMPLETE", files=prepared["files"]))
    print(json.dumps(dict(status="COMPLETE", changedCopies=prepared["changedCopies"])), flush=True)


def verify(*, prepared=False):
    """Independent full-shard readback, preservation census and training-length check."""
    started = time.perf_counter()
    presenter, task = load()
    info = json.loads((OUT / "stage.json").read_text())
    selected, _ = final_rows()
    location = OUT / "publication" if prepared else DATA
    changed, counts = Counter(), Counter()
    for name in SHARDS:
        with (OUT / "before" / name).open() as old, (location / name).open() as new:
            for old_line, new_line in zip(old, new, strict=True):
                a, b = json.loads(old_line), json.loads(new_line)
                did = a["documentId"]
                assert b["documentId"] == did
                counts[name] += 1
                if did in selected:
                    assert b == selected[did], (name, did)
                    changed[name] += 1
                else:
                    assert old_line == new_line, (name, did, "unselected bytes changed")
                assert task.canonicalize(b["target"]) == b["target"], did
                for field in ("joinedRawText", "positionedText"):
                    assert digest(b[field].encode()) == b[field + "Sha256"]
    protected = {}
    for name, pin in info["beforeFiles"].items():
        if name not in (*SHARDS, "manifest.json", "README.md"):
            assert sha256_file(DATA / name) == pin, name
            protected[name] = pin
    for name in ("real-train.jsonl", "validation.jsonl"):
        for row in map(json.loads, (DATA / name).open()):
            assert task.canonicalize(row["target"]) == row["target"]
            for field in ("joinedRawText", "positionedText"):
                assert digest(row[field].encode()) == row[field + "Sha256"]
            counts[name] += 1
    manifest = json.loads((location / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = location / name if name in (*SHARDS, "README.md") else DATA / name
        assert sha256_file(path) == expected, name
    frozen = ROOT / manifest["baseDataset"]
    frozen_manifest = json.loads((frozen / "manifest.json").read_text())
    for name, expected in frozen_manifest["files"].items():
        assert sha256_file(frozen / name) == expected, ("frozen R27", name)
    receipts = [
        r
        for batch in range(1, 31)
        for r in json.loads((OUT / "batches" / f"{batch:02d}-receipts.json").read_text())
    ]
    edits = [e for r in receipts for e in r["proof"]["edits"]]
    assert len({r["documentId"] for r in receipts}) == 3000
    assert all(presenter.vocabulary.supports(e["after"], e["presentedCategory"]) for e in edits)
    import re

    assert all(re.findall(r"\d+", e["before"]) == re.findall(r"\d+", e["after"]) for e in edits)
    result = dict(
        status="PASS",
        prepared=prepared,
        counts=counts,
        changedCopies=changed,
        protectedFiles=protected,
        frozenR27FilesVerified=len(frozen_manifest["files"]),
        aliasEdits=len(edits),
        aliasesByStyle=Counter(e["style"] for e in edits),
        aliasesByRole=Counter(e["role"] for e in edits),
        shortenedSpecificOccurrences=sum(e["category"] != e["presentedCategory"] for e in edits),
        shortenedSpecificDocuments=sum(
            any(e["category"] != e["presentedCategory"] for e in r["proof"]["edits"])
            for r in receipts
        ),
        shortAliasDocuments=sum(
            any(e["style"] == "short" for e in r["proof"]["edits"]) for r in receipts
        ),
        literalTargetProjections=sum(len(r["targetChanges"]) for r in receipts),
        auditSeconds=sum(
            json.loads((OUT / "batches" / f"{i:02d}-audit.json").read_text())["seconds"]
            for i in range(1, 31)
        ),
        seconds=time.perf_counter() - started,
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    save(OUT / ("prepared-validation.json" if prepared else "final-readback.json"), result)
    print(json.dumps(result), flush=True)


def token_check():
    from transformers import AutoTokenizer

    from document_ocr.training.tasks import canonical_json

    tokenizer = AutoTokenizer.from_pretrained(
        "google/t5gemma-2-270m-270m",
        revision="7c38f16641f455ef0685b18431faf1b17722d5a1",
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    after, _ = final_rows()
    before = read_rows(OUT / "before" / SHARDS[0])
    keys = list(after)
    lengths = {"before": [], "after": [], "targets": []}
    started = time.perf_counter()
    for offset in range(0, len(keys), 100):
        batch = keys[offset : offset + 100]
        for name, rows in (("before", before), ("after", after)):
            encoded = tokenizer(
                [rows[k]["positionedText"] for k in batch],
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            lengths[name].extend(map(len, encoded["input_ids"]))
        encoded = tokenizer(
            text_target=[canonical_json(after[k]["target"], pretty=False) for k in batch],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths["targets"].extend(len(ids) + 1 for ids in encoded["input_ids"])
    assert max(lengths["after"]) <= 19200
    assert max(lengths["targets"]) <= 5500
    deltas = [b - a for a, b in zip(lengths["before"], lengths["after"], strict=True)]
    result = dict(
        status="PASS",
        records=len(keys),
        maxSourceBefore=max(lengths["before"]),
        maxSourceAfter=max(lengths["after"]),
        maxTargetWithEOS=max(lengths["targets"]),
        totalSourceTokenDelta=sum(deltas),
        maxPerRecordDelta=max(deltas),
        minPerRecordDelta=min(deltas),
        seconds=time.perf_counter() - started,
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        tokenizer="google/t5gemma-2-270m-270m@7c38f16641f455ef0685b18431faf1b17722d5a1",
        sourceLimit=19200,
        targetLimit=5500,
    )
    save(OUT / "token-validation.json", result)
    print(json.dumps(result), flush=True)


def benchmark():
    presenter, _ = load()
    before = read_rows(OUT / "before" / SHARDS[0])
    sample = [
        r
        for i in range(1, 31)
        for r in json.loads((OUT / "batches" / f"{i:02d}-receipts.json").read_text())[:10]
    ]
    timings = {"identity": [], "alias": []}
    for repeat in range(3):
        for mode in ("identity", "alias") if repeat % 2 == 0 else ("alias", "identity"):
            started = time.perf_counter()
            for receipt in sample:
                slots = receipt["currentOwners"]
                if mode == "identity":
                    slots = [{**s, "protectedSourceDetail": True} for s in slots]
                refresh(before[receipt["documentId"]], slots, presenter, seed=SEED)
            timings[mode].append(time.perf_counter() - started)
    from statistics import median

    result = dict(
        recordsPerPass=len(sample),
        passesPerMode=3,
        timingsSeconds=timings,
        medianMsPerRecord={k: median(v) * 1000 / len(sample) for k, v in timings.items()},
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        interpretation=(
            "Offline identity/no-change replay versus actual alias refresh; "
            "no production hot path changed."
        ),
    )
    save(OUT / "benchmark.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "stage",
            "audit",
            "audit-all",
            "prepare",
            "publish",
            "verify-prepared",
            "verify",
            "tokens",
            "benchmark",
        ],
    )
    parser.add_argument("--batch", type=int, choices=range(1, 31))
    args = parser.parse_args()
    if args.mode == "benchmark":
        benchmark()
    elif args.mode == "tokens":
        token_check()
    elif args.mode in ("verify", "verify-prepared"):
        verify(prepared=args.mode == "verify-prepared")
    elif args.mode == "stage":
        stage()
    elif args.mode == "prepare":
        prepare_publication()
    elif args.mode == "publish":
        publish()
    elif args.mode == "audit-all":
        context = audit_context()
        for batch in range(1, 31):
            audit(batch, context)
    else:
        if args.batch is None:
            parser.error("audit requires --batch")
        audit(args.batch)
