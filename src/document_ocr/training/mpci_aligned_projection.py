"""Publish a verified MPCI-named v6 training view of a pinned v5 dataset.

The v5 parent is never modified.  Per-record target hashes retain an exact
lookup into that richer source for relations the MPCI-facing target omits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

from document_ocr.hashing import canonical_json_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v6 import project_relation_v5_target_to_v6
from document_ocr.training.tasks import get_training_task


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid non-standard JSON constant {value!r}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    result = json.loads(
        payload,
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(result, dict):
        raise ValueError("JSONL record must be an object")
    return result


def _require_hash(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pinned input must be a regular non-symlink file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"pinned SHA-256 mismatch for {path}: {actual} != {expected}")


def _audit_fact_preservation(old: dict[str, Any], new: dict[str, Any]) -> Counter[str]:
    old_patch = old["documentPatch"]
    new_patch = new["documentPatch"]
    for key, value in old_patch.items():
        if (
            key not in {"containers", "cargoGroups", "cargoPackages", "cargoAllocationGroups"}
            and new_patch.get(key) != value
        ):
            raise ValueError(f"non-cargo fact changed: {key}")
    for old_row, new_row in zip(
        old_patch.get("containers", []),
        new_patch.get("containerInformation", []),
        strict=True,
    ):
        expected = {"equipmentIdentifier": old_row["containerNumber"]}
        expected.update({key: value for key, value in old_row.items() if key != "containerNumber"})
        if new_row != expected:
            raise ValueError("container fact or source order changed")
    groups = old_patch.get("cargoGroups", [])
    new_goods = new_patch.get("goodsItemDetails", [])
    if len(groups) != len(new_goods):
        raise ValueError("goods item count changed")
    old_packages = old_patch.get("cargoPackages", [])
    old_allocations = old_patch.get("cargoAllocationGroups", [])
    package_groups: dict[str, list[dict[str, Any]]] = {}
    for row in old_packages:
        package_groups.setdefault(row["groupId"], []).append(row)
    allocation_groups = {row["groupId"]: row for row in old_allocations}
    for group, goods in zip(groups, new_goods, strict=True):
        group_id = group["groupId"]
        direct = {
            key: value
            for key, value in goods.items()
            if key not in {"numberAndTypeOfPackages", "splitGoodsPlacement"}
        }
        if direct != {key: value for key, value in group.items() if key != "groupId"}:
            raise ValueError(f"goods facts changed for {group_id}")
        source_packages = package_groups.get(group_id, [])
        target_packages = goods.get("numberAndTypeOfPackages", [])
        if len(source_packages) != len(target_packages):
            raise ValueError(f"package count changed for {group_id}")
        for old_row, new_row in zip(source_packages, target_packages, strict=True):
            if new_row.get("packageQuantity") != old_row.get("quantity"):
                raise ValueError(f"package quantity changed for {group_id}")
            if new_row.get("typeCategory") != old_row.get("typeCategory"):
                raise ValueError(f"package category changed for {group_id}")
            if new_row.get("typeOfPackages") != old_row.get("typeDescription"):
                raise ValueError(f"printed package fallback changed for {group_id}")
            if set(new_row) != ({"packageQuantity"} if "quantity" in old_row else set()) | (
                {"typeCategory"} if "typeCategory" in old_row else set()
            ) | ({"typeOfPackages"} if "typeDescription" in old_row else set()):
                raise ValueError(f"package fact keys changed for {group_id}")
        source_allocations = allocation_groups.get(group_id, {}).get("allocations", [])
        target_placements = goods.get("splitGoodsPlacement", [])
        if len(source_allocations) != len(target_placements):
            raise ValueError(f"placement count changed for {group_id}")
        for old_row, new_row in zip(source_allocations, target_placements, strict=True):
            if new_row.get("equipmentIdentifier") != old_row["containerNumber"]:
                raise ValueError(f"placement container changed for {group_id}")
            if new_row.get("packageQuantity") != old_row.get("packageQuantity"):
                raise ValueError(f"placement quantity changed for {group_id}")
            if set(new_row) != {"equipmentIdentifier"} | (
                {"packageQuantity"} if "packageQuantity" in old_row else set()
            ):
                raise ValueError(f"placement fact keys changed for {group_id}")
    return Counter(
        {
            "goods": len(groups),
            "packages": len(old_packages),
            "containers": len(old_patch.get("containers", [])),
            "placements": sum(len(row["allocations"]) for row in old_allocations),
            "allocationGroups": len(old_allocations),
            "oneToOnePackageLinks": sum(
                sum(placement.get("packageId") is not None for placement in row["allocations"])
                for row in old_allocations
            ),
            "coveredPackageIds": sum(len(row["packageIds"]) for row in old_allocations),
        }
    )


def _write_split(
    source: Path,
    output: Path,
    receipts: BinaryIO,
    split: str,
    seen_document_ids: set[str],
) -> tuple[int, Counter[str]]:
    task = get_training_task("bill_of_lading_mpci_aligned_v6")
    counts: Counter[str] = Counter()
    records = 0
    with source.open("rb") as incoming, output.open("wb") as outgoing:
        for line_number, line in enumerate(incoming, start=1):
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError(f"{source}:{line_number}: malformed JSONL line")
            row = _strict_json(line)
            document_id = row.get("documentId")
            if not isinstance(document_id, str) or document_id in seen_document_ids:
                raise ValueError(f"{source}:{line_number}: missing or duplicate documentId")
            seen_document_ids.add(document_id)
            records += 1
            text = row.get("joinedRawText")
            text_hash = row.get("joinedRawTextSha256")
            if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != text_hash:
                raise ValueError(f"{source}:{line_number}: joinedRawText SHA-256 mismatch")
            old = row.get("target")
            if not isinstance(old, dict):
                raise ValueError(f"{source}:{line_number}: target must be an object")
            new = project_relation_v5_target_to_v6(old)
            task.canonicalize(new)
            counts.update(_audit_fact_preservation(old, new))
            row["target"] = new
            outgoing.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
                + b"\n"
            )
            receipts.write(
                json.dumps(
                    {
                        "split": split,
                        "documentId": document_id,
                        "sourceTargetSha256": hashlib.sha256(canonical_json_bytes(old)).hexdigest(),
                        "targetSha256": hashlib.sha256(canonical_json_bytes(new)).hexdigest(),
                    },
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )
    return records, counts


def migrate_dataset(
    *,
    source_root: Path,
    output_root: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify parent bytes, project both splits, and publish an immutable derivative."""

    source_root = source_root.resolve(strict=True)
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"output already exists and will not be overwritten: {output_root}")
    manifest_path = source_root / "manifest.json"
    _require_hash(manifest_path, expected_manifest_sha256)
    parent = _strict_json(manifest_path.read_bytes())
    outputs = parent.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("parent manifest lacks pinned outputs")
    inputs: dict[str, Path] = {}
    for split in ("train", "validation"):
        receipt = outputs.get(split)
        if not isinstance(receipt, dict):
            raise ValueError(f"parent manifest lacks {split} receipt")
        path = source_root / f"{split}.jsonl"
        _require_hash(path, receipt["sha256"])
        inputs[split] = path
    lineage = source_root / "lineage.jsonl"
    if lineage.is_symlink() or not lineage.is_file():
        raise ValueError("parent lineage.jsonl must be a regular file")
    lineage_hash = sha256_file(lineage)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        counts: Counter[str] = Counter()
        projected: dict[str, Any] = {}
        seen_document_ids: set[str] = set()
        with (staging / "source-target-receipts.jsonl").open("wb") as receipts:
            for split in ("train", "validation"):
                records, split_counts = _write_split(
                    inputs[split],
                    staging / f"{split}.jsonl",
                    receipts,
                    split,
                    seen_document_ids,
                )
                if records != outputs[split]["records"]:
                    raise ValueError(f"{split} record count changed during migration")
                counts.update(split_counts)
                projected[split] = {
                    "path": f"{split}.jsonl",
                    "records": records,
                    "sha256": sha256_file(staging / f"{split}.jsonl"),
                }
        shutil.copyfile(lineage, staging / "lineage.jsonl")
        if sha256_file(staging / "lineage.jsonl") != lineage_hash:
            raise ValueError("lineage bytes changed during migration")
        manifest = {
            "schemaVersion": 1,
            "status": "complete",
            "task": "bill_of_lading_mpci_aligned_v6",
            "parent": {
                "path": str(source_root),
                "manifestSha256": expected_manifest_sha256,
                "outputs": outputs,
            },
            "outputs": projected,
            "lineageSha256": lineage_hash,
            "sourceTargetReceiptsSha256": sha256_file(staging / "source-target-receipts.jsonl"),
            "retainedFacts": dict(counts),
            "informationLoss": (
                "Model target drops v5 graph IDs, coverage, and package-level placement links. "
                "Every parent target is retained byte-for-byte and pinned by per-record SHA-256."
            ),
            "platformBoundary": (
                "Extraction target only; live MPCI converter submission validation was not "
                "available. Local sparse validator rejects some source-supported placements."
            ),
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        if output_root.exists() or output_root.is_symlink():
            raise ValueError(
                f"output appeared during migration and will not be overwritten: {output_root}"
            )
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    manifest = migrate_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        expected_manifest_sha256=args.source_manifest_sha256,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
