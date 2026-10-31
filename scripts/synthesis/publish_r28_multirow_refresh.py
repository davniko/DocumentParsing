"""Publish only the complete, admitted 1,500-record refresh and reviewed repairs.

Preparation and validation are read-only with respect to R28. Promotion checks
all original hashes again, replaces complete staged shards, and rolls back on
failure. The immutable before copy is retained after success.
"""

import argparse
import copy
import json
import resource
import time
from collections import Counter
from pathlib import Path

from refresh_r28_package_aliases import promote_files

from document_ocr.atomic import atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.package_alias_refresh import apply_reviewed_cleanup
from document_ocr.training.tasks import (
    RelationExplicitTaskConstraints,
    canonical_json,
    get_training_task,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/curated/mpci-bl-r28-positions-v1"
OUT = ROOT / "docs/analysis/r28-multirow-refresh1500-20261031"
SHARDS = ("synthetic-train.jsonl", "previous-uae-synthetic.jsonl", "old-uae-addition.jsonl")


def read(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    rows = {}
    for row in map(json.loads, path.open()):
        if row["documentId"] in rows:
            raise ValueError("duplicate record identity")
        rows[row["documentId"]] = row
    return rows


def replacement(original, native, receipt):
    """Retain R28 identity while replacing the complete accepted synthetic input."""
    if digest(original) != receipt["previousRecordSha256"]:
        raise ValueError("current selected record differs from selection")
    if digest(native) != receipt["nativeRecordSha256"]:
        raise ValueError("native record differs from production admission")
    if original["sourceDocumentId"] != native["sourceDocumentId"]:
        raise ValueError("replacement changes source identity")
    if native["documentId"] != receipt["candidateId"]:
        raise ValueError("replacement uses another native candidate")
    result = copy.deepcopy(original)
    for key in (
        "joinedRawText",
        "joinedRawTextSha256",
        "positionedText",
        "positionedTextSha256",
        "target",
    ):
        result[key] = copy.deepcopy(native[key])
    if (
        len(result["target"]["goodsItemDetails"]) != 1
        or len(result["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]) < 2
    ):
        raise ValueError("replacement is not one goods item with multiple public rows")
    return result


def prepare():
    start = time.perf_counter()
    selection = read(OUT / "selection.json")
    for name, pin in selection["beforeFiles"].items():
        if sha256_file(DATA / name) != pin or sha256_file(OUT / "before" / name) != pin:
            raise ValueError("current/before dataset changed: " + name)
    review = read(OUT / "final-scoped-review.json")
    if review["status"] != "PASS" or review["records"] != 1500:
        raise ValueError("complete final scoped review missing")
    components = read(OUT / "admission-components-consolidated.json")
    expected_configs = {p for batch in components.values() for p in batch["components"]}
    admissions = [read(p) for p in sorted((OUT / "batches").glob("*-component-admission.json"))]
    if {a["preparedConfig"] for a in admissions} != expected_configs or len(admissions) != len(
        expected_configs
    ):
        raise ValueError("missing or duplicate production component admission")
    selected = {r["documentId"]: r for b in selection["batches"] for r in b["records"]}
    original = read_rows(OUT / "before" / SHARDS[0])
    updated, receipts, batch_counts = {}, [], Counter()
    for admission in admissions:
        if admission["unresolvedReviewFindings"] or admission["positions"]["reflowRejections"]:
            raise ValueError("production admission retains holds")
        if admission["positions"]["records"] != len(admission["records"]):
            raise ValueError("native coordinate projection is incomplete")
        if sha256_file(ROOT / admission["preparedConfig"]) != admission["preparedConfigSha256"]:
            raise ValueError("admitted config changed")
        paths = {r["nativePublication"] for r in admission["records"]}
        if len(paths) != 1:
            raise ValueError("one component must have one native publication")
        path = ROOT / paths.pop()
        native_rows = read_rows(path)
        native_sha256 = sha256_file(path)
        for receipt in admission["records"]:
            did = receipt["documentId"]
            if did in updated or did not in selected:
                raise ValueError("admission duplicates or substitutes a selected slot")
            if native_sha256 != receipt["nativePublicationSha256"]:
                raise ValueError("native publication changed")
            updated[did] = replacement(original[did], native_rows[receipt["candidateId"]], receipt)
            receipts.append(receipt)
            batch_counts[admission["batch"]] += 1
    if set(updated) != selected.keys() or batch_counts != {i: 100 for i in range(1, 16)}:
        raise ValueError("refresh is not exactly fifteen complete batches of 100")
    all_original = dict(original)
    for row in map(json.loads, (OUT / "before" / SHARDS[1]).open()):
        prior = all_original.setdefault(row["documentId"], row)
        if prior != row:
            raise ValueError("sidecar has divergent pre-refresh facts")
    extra_receipts = []
    contacts = read(OUT / "contact-repairs.json")
    marks = read(OUT / "private-marks-repairs.json")
    repairs = list(contacts)
    for did, name in marks["replacements"].items():
        row = all_original[did]
        repairs.append(
            dict(
                documentId=did,
                beforeRecordSha256=digest(row),
                rawEdits=[dict(before=marks["before"], after=name)],
                targetEdits=[],
                reason=marks["reason"],
            )
        )
    for repair in repairs:
        did = repair["documentId"]
        if did in updated:
            raise ValueError("narrow repair overlaps full regeneration")
        row = all_original[did]
        revised, evidence = apply_reviewed_cleanup(row, repair)
        if [s.rsplit(" ||", 1)[-1] for s in row["positionedText"].splitlines()] != [
            s.rsplit(" ||", 1)[-1] for s in revised["positionedText"].splitlines()
        ]:
            raise ValueError("narrow repair changed coordinate anchors")
        updated[did] = revised
        extra_receipts.append(dict(documentId=did, **evidence))
    publication = OUT / "publication"
    publication.mkdir(exist_ok=True)
    changed = {}
    for name in SHARDS:
        counts = Counter()
        with (
            (OUT / "before" / name).open() as source,
            (publication / name).open("w") as destination,
        ):
            for line in source:
                row = json.loads(line)
                did = row["documentId"]
                if did in updated:
                    if row != all_original[did]:
                        raise ValueError("sidecar replacement would erase divergent facts")
                    destination.write(
                        json.dumps(updated[did], ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    counts["regenerated" if did in selected else "narrowRepair"] += 1
                else:
                    destination.write(line)
        changed[name] = dict(counts)
    atomic_publish_json(OUT / "replacement-receipts.json", receipts)
    atomic_publish_json(OUT / "additional-repair-receipts.json", extra_receipts)
    readme = (
        (DATA / "README.md").read_text() + "\n\n## Multi-package-row refresh — 2026-10-31\n\n"
        "Exactly 1,500 existing synthetic records, across 300 layouts in 15 batches of 100,\n"
        "were independently regenerated with two or three disjoint declared package groups.\n"
        "Source occurrence projections may expose additional public rows. Allocations remain\n"
        "scalar counts; nested/support packages are not additional public totals. Matching\n"
        "UAE sidecars are synchronized. Real600, validation60, membership, and unselected\n"
        "thermal/DG/mixed-equipment cohorts remain unchanged. Separate reviewed contact\n"
        "and Marks-only repairs are inventoried in the refresh report.\n\n"
        "Evidence: `docs/analysis/r28-multirow-refresh1500-20261031/REPORT.md`, exact native\n"
        "admission/replacement receipts, before copies, validation and readback. This is\n"
        "certification of these bounded changes, not a new full-corpus semantic audit.\n"
    )
    (publication / "README.md").write_text(readme)
    manifest = read(DATA / "manifest.json")
    manifest["multiPackageRowRefresh20261031"] = dict(
        records=1500,
        parents=300,
        batches=15,
        changedCopies=changed,
        additionalContactRepairs=len(contacts),
        privateMarksRepairs=len(marks["replacements"]),
        report=str((OUT / "REPORT.md").relative_to(ROOT)),
        before=str((OUT / "before").relative_to(ROOT)),
        replacementReceiptsSha256=sha256_file(OUT / "replacement-receipts.json"),
        realAndValidationUnchanged=True,
        publicPackageFactorization=False,
    )
    for name in (*SHARDS, "README.md"):
        manifest["files"][name] = sha256_file(publication / name)
    atomic_publish_json(publication / "manifest.json", manifest)
    result = dict(
        status="PREPARED_NOT_PUBLISHED",
        changedCopies=changed,
        files={
            name: sha256_file(publication / name)
            for name in (*SHARDS, "README.md", "manifest.json")
        },
        seconds=time.perf_counter() - start,
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_publish_json(OUT / "publication-preflight.json", result)
    print(json.dumps(result), flush=True)


def validate(*, published=False):
    start = time.perf_counter()
    selection, preflight = read(OUT / "selection.json"), read(OUT / "publication-preflight.json")
    folder = DATA if published else OUT / "publication"
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        (DATA / "task-constraints.json").read_bytes()
    )
    task = get_training_task(constraints.task).bind_constraints(constraints)
    changed_ids = {r["documentId"] for r in read(OUT / "replacement-receipts.json")}
    changed_ids.update(r["documentId"] for r in read(OUT / "additional-repair-receipts.json"))
    source_rows, seen, counts = {}, {}, {}
    for name in SHARDS:
        current = read_rows(folder / name)
        before = read_rows(OUT / "before" / name)
        if list(current) != list(before):
            raise ValueError("dataset membership changed")
        with (folder / name).open("rb") as new_file, (OUT / "before" / name).open("rb") as old_file:
            for new_line, old_line in zip(new_file, old_file, strict=True):
                did = json.loads(old_line)["documentId"]
                if did not in changed_ids and new_line != old_line:
                    raise ValueError("unselected serialized record bytes changed")
        for did, row in current.items():
            if did not in changed_ids and row != before[did]:
                raise ValueError("unselected record changed")
            if task.canonicalize(row["target"]) != row["target"]:
                raise ValueError("noncanonical target: " + did)
            for field in ("joinedRawText", "positionedText"):
                if digest(row[field].encode()) != row[field + "Sha256"]:
                    raise ValueError("text hash mismatch")
            raw, positioned = row["joinedRawText"].splitlines(), row["positionedText"].splitlines()
            if len(raw) != len(positioned) or any(
                p.rsplit(" ||", 1)[0] != r for r, p in zip(raw, positioned, strict=True)
            ):
                raise ValueError("coordinate input changes raw text")
            if did in seen and seen[did] != digest(row):
                raise ValueError("sidecar copies diverge")
            seen[did] = digest(row)
        counts[name] = len(current)
        if name == SHARDS[0]:
            source_rows = current
    if counts[SHARDS[0]] != 6900:
        raise ValueError("training synthetic cardinality changed")
    if (
        sum(
            len(r["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]) > 1
            for r in source_rows.values()
        )
        != 1500
    ):
        raise ValueError("unexpected multirow coverage")
    for name, expected in selection["beforeFiles"].items():
        if name not in preflight["files"] and sha256_file(DATA / name) != expected:
            raise ValueError("protected dataset file changed: " + name)
    for name, expected in preflight["files"].items():
        if sha256_file(folder / name) != expected:
            raise ValueError("staged/published file changed: " + name)
    result = dict(
        status="PASS",
        counts=counts,
        uniqueSyntheticDocuments=len(seen),
        changedUniqueDocuments=len(changed_ids),
        multirowDocuments=1500,
        realAndValidationUnchanged=True,
        publicRowCounts=Counter(
            len(r["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"])
            for r in source_rows.values()
        ),
        seconds=time.perf_counter() - start,
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_publish_json(
        OUT / ("final-readback.json" if published else "prepared-validation.json"), result
    )
    print(json.dumps(result), flush=True)


def tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "google/t5gemma-2-270m-270m",
        revision="7c38f16641f455ef0685b18431faf1b17722d5a1",
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    start = time.perf_counter()
    ids = {r["documentId"] for r in read(OUT / "replacement-receipts.json")}
    rows = [
        r
        for r in map(json.loads, (OUT / "publication" / SHARDS[0]).open())
        if r["documentId"] in ids
    ]
    before = read_rows(OUT / "before" / SHARDS[0])
    lengths = {"before": [], "after": [], "target": []}
    for offset in range(0, len(rows), 50):
        batch = rows[offset : offset + 50]
        for key, items in (("before", [before[r["documentId"]] for r in batch]), ("after", batch)):
            encoded = tokenizer(
                [r["positionedText"] for r in items],
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            lengths[key].extend(map(len, encoded["input_ids"]))
        encoded = tokenizer(
            text_target=[canonical_json(r["target"], pretty=False) for r in batch],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths["target"].extend(len(v) + 1 for v in encoded["input_ids"])
    if len(rows) != 1500 or max(lengths["after"]) > 19200 or max(lengths["target"]) > 5500:
        raise ValueError("training token budget exceeded")
    result = dict(
        status="PASS",
        records=1500,
        sourceLimit=19200,
        targetLimit=5500,
        maxSourceBefore=max(lengths["before"]),
        maxSourceAfter=max(lengths["after"]),
        maxTargetWithEOS=max(lengths["target"]),
        sourceTokensBefore=sum(lengths["before"]),
        sourceTokensAfter=sum(lengths["after"]),
        seconds=time.perf_counter() - start,
        maxRssMiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    atomic_publish_json(OUT / "token-validation.json", result)
    print(json.dumps(result), flush=True)


def promote_and_verify(dataset, publication, before, old_hashes, new_hashes, verify):
    """Include post-publication readback in the reversible transaction."""
    promote_files(dataset, publication, before, old_hashes, new_hashes)
    try:
        verify()
    except BaseException:
        promote_files(
            dataset,
            before,
            publication,
            new_hashes,
            {name: old_hashes[name] for name in new_hashes},
        )
        raise


def publish():
    from xml.etree import ElementTree

    selection, preflight = read(OUT / "selection.json"), read(OUT / "publication-preflight.json")
    for name in (
        "prepared-validation.json",
        "token-validation.json",
        "final-scoped-review.json",
        "completion-census.json",
    ):
        if read(OUT / name)["status"] != "PASS":
            raise ValueError("publication prerequisite failed")
    suites = ElementTree.parse(OUT / "final-tests.xml").getroot().findall("testsuite")
    if not suites or any(int(s.attrib["failures"]) or int(s.attrib["errors"]) for s in suites):
        raise ValueError("final tests missing/failed")
    for name, expected in selection["beforeFiles"].items():
        if sha256_file(DATA / name) != expected:
            raise ValueError("concurrent dataset modification: " + name)
    atomic_write_json(
        OUT / "publication-state.json", dict(status="APPLYING", files=preflight["files"])
    )
    try:
        promote_and_verify(
            DATA,
            OUT / "publication",
            OUT / "before",
            selection["beforeFiles"],
            preflight["files"],
            lambda: validate(published=True),
        )
    except BaseException:
        restored = all(
            sha256_file(DATA / name) == selection["beforeFiles"][name]
            for name in preflight["files"]
        )
        atomic_write_json(
            OUT / "publication-state.json",
            dict(status="ROLLED_BACK" if restored else "RECOVERY_REQUIRED"),
        )
        raise
    atomic_write_json(
        OUT / "publication-state.json", dict(status="COMPLETE", files=preflight["files"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "validate", "tokens", "publish"])
    args = parser.parse_args()
    {"prepare": prepare, "validate": validate, "tokens": tokens, "publish": publish}[args.command]()
