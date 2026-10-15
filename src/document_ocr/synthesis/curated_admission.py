"""Pure contracts shared by reviewed synthesis-template admission workflows."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import yaml

from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.synthesis.curated import digest

ASSETS = frozenset(
    (
        "source.txt",
        "source-positioned.txt",
        "target.json",
        "contract.json",
        "template.json",
        "alignment.json",
    )
)
SHARED = ("ownership.yaml", "auxiliary.yaml")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def records(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def yaml_bytes(value):
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode()


def pin_bytes(value):
    return sha256_file(value) if isinstance(value, Path) else sha256_bytes(value)


def verify_pins(pins):
    for path, expected in pins.items():
        require(sha256_file(path) == expected, f"Pinned prerequisite changed: {path}")


def unique_rows(rows, field):
    by_id = {row[field]: row for row in rows}
    require(len(by_id) == len(rows), f"Duplicate {field} inventory")
    return by_id


def merge_shared(original, staged, old_ids, new_ids):
    require(set(original["sources"]) == old_ids, "Existing shared inventory differs")
    require(set(staged["sources"]) == old_ids | new_ids, "Staged shared inventory differs")
    require(not old_ids & new_ids, "Admission cannot replace a registered source")
    require(
        {k: v for k, v in original.items() if k != "sources"}
        == {k: v for k, v in staged.items() if k != "sources"},
        "Shared format metadata changed",
    )
    require(
        all(original["sources"][sid] == staged["sources"][sid] for sid in old_ids),
        "Admission modified an old shared source",
    )
    return deepcopy(staged)


def validate_case(row, folder, *, expected=None):
    require(
        set(p.name for p in folder.iterdir() if p.is_file()) >= ASSETS,
        f"Incomplete case assets: {folder}",
    )
    files = {name: sha256_file(folder / name) for name in ASSETS}
    if expected is not None:
        require(files == expected, f"Existing case asset changed: {row['documentId']}")
    require(
        (folder / "source.txt").read_bytes() == row["joinedRawText"].encode(),
        "Case source differs from promoted authority",
    )
    require(
        (folder / "source-positioned.txt").read_bytes() == row["positionedText"].encode(),
        "Case positions differ from promoted authority",
    )
    require(
        read(folder / "target.json") == row["target"], "Case target differs from promoted authority"
    )
    contract = read(folder / "contract.json")
    require(
        contract["sourceSha256"] == files["source.txt"]
        and contract["targetSha256"] == digest(row["target"]),
        "Case source/target contract pin differs",
    )
    return files


def validate_exclusion(selected, validation, selection):
    pdfs = {}
    for item in selection["documents"]:
        for sid in item["document_ids"]:
            pdfs.setdefault(sid, set()).add(item["source_sha256"])
    ids = {row["documentId"] for row in validation}
    text = {digest(row["joinedRawText"].encode()) for row in validation}

    def bill(row):
        return re.sub(
            r"[^A-Z0-9]", "", row["target"]["documentPatch"].get("billOfLadingNumber", "").upper()
        )

    bills = {bill(row) for row in validation} - {""}
    require(all(sid in pdfs for sid in ids), "Validation PDF identity missing")
    validation_pdfs = set().union(*(pdfs[sid] for sid in ids))
    for row in selected:
        sid = row["documentId"]
        require(
            sid not in ids
            and digest(row["joinedRawText"].encode()) not in text
            and bill(row) not in bills,
            f"Validation source/input/B-L overlap: {sid}",
        )
        require(sid in pdfs and not pdfs[sid] & validation_pdfs, f"Validation PDF overlap: {sid}")
