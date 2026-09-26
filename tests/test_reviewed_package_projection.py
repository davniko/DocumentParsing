from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.training.reviewed_package_projection import (
    load_reviewed_package_contract,
    project_reviewed_package_target,
)

SOURCE_ID = f"doc_{'1' * 64}"
CATALOG_COMMIT = "2" * 64


def _source() -> dict:
    return {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "WIDGETS"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 100,
                    "typeCategory": "PACKAGE_CARTON",
                },
                {
                    "groupId": "g1",
                    "packageId": "p2",
                    "quantity": 2,
                    "typeCategory": "PACKAGE_PALLET",
                },
            ],
        },
    }


def _contract(tmp_path: Path) -> tuple[Path, str]:
    payload = {
        "schemaVersion": 1,
        "catalogCommitSha256": CATALOG_COMMIT,
        "sourceGroupCount": 1,
        "decisions": [
            {
                "sourceDocumentId": SOURCE_ID,
                "sourceLabelCanonicalSha256": sha256_bytes(canonical_json_bytes(_source())),
                "groupId": "g1",
                "sourcePackageIds": ["p1", "p2"],
                "allowedTargetPackageIdSequences": [["p1", "p2"]],
                "status": "ready",
                "retainedPackageIds": ["p1"],
                "rationale": "Printed cartons are the inner level.",
            }
        ],
    }
    path = tmp_path / "contract.json"
    content = (json.dumps(payload) + "\n").encode()
    path.write_bytes(content)
    return path, sha256_bytes(content)


def test_reviewed_package_projection_preserves_retained_allocation(tmp_path: Path) -> None:
    path, digest = _contract(tmp_path)
    contract = load_reviewed_package_contract(
        path, expected_sha256=digest, catalog_commit_sha256=CATALOG_COMMIT
    )
    target = _source()
    target["documentPatch"]["cargoAllocationGroups"] = [
        {
            "groupId": "g1",
            "coverage": "single_package_level",
            "packageIds": ["p1"],
            "allocations": [{"containerNumber": "TCLU6905627", "packageQuantity": 100}],
        }
    ]
    before = canonical_json_bytes(target)

    projected, decisions = project_reviewed_package_target(
        source_document_id=SOURCE_ID,
        source_target=_source(),
        target=target,
        contract=contract,
    )

    assert canonical_json_bytes(target) == before
    assert projected["documentPatch"]["cargoPackages"] == [
        _source()["documentPatch"]["cargoPackages"][0]
    ]
    assert (
        projected["documentPatch"]["cargoAllocationGroups"]
        == target["documentPatch"]["cargoAllocationGroups"]
    )
    assert decisions[0]["metadataPackageIds"] == ["p2"]


def test_multi_level_package_without_review_fails_closed(tmp_path: Path) -> None:
    path, digest = _contract(tmp_path)
    contract = load_reviewed_package_contract(
        path, expected_sha256=digest, catalog_commit_sha256=CATALOG_COMMIT
    )
    with pytest.raises(ValueError, match="lacks review"):
        project_reviewed_package_target(
            source_document_id=f"doc_{'3' * 64}",
            source_target=_source(),
            target=_source(),
            contract=contract,
        )
    with pytest.raises(ValueError, match="unreviewed generated package topology"):
        shortened = _source()
        shortened["documentPatch"]["cargoPackages"].pop()
        project_reviewed_package_target(
            source_document_id=SOURCE_ID,
            source_target=_source(),
            target=shortened,
            contract=contract,
        )
    with pytest.raises(ValueError, match="reviewed package source changed"):
        altered = _source()
        altered["documentPatch"]["cargoPackages"][0]["quantity"] = 99
        project_reviewed_package_target(
            source_document_id=SOURCE_ID,
            source_target=altered,
            target=_source(),
            contract=contract,
        )


def test_contract_pin_must_match_catalog_and_file(tmp_path: Path) -> None:
    path, digest = _contract(tmp_path)
    with pytest.raises(ValueError, match="hash differs"):
        load_reviewed_package_contract(
            path, expected_sha256="f" * 64, catalog_commit_sha256=CATALOG_COMMIT
        )
    with pytest.raises(ValueError, match="not pinned"):
        load_reviewed_package_contract(path, expected_sha256=digest, catalog_commit_sha256="f" * 64)


def test_reviewed_source_is_held_before_projection(tmp_path: Path) -> None:
    path, _ = _contract(tmp_path)
    payload = json.loads(path.read_bytes())
    payload["decisions"][0]["status"] = "review"
    payload["decisions"][0]["retainedPackageIds"] = None
    content = canonical_json_bytes(payload)
    path.write_bytes(content)
    contract = load_reviewed_package_contract(
        path, expected_sha256=sha256_bytes(content), catalog_commit_sha256=CATALOG_COMMIT
    )

    with pytest.raises(ValueError, match="package source requires review"):
        project_reviewed_package_target(
            source_document_id=SOURCE_ID,
            source_target=_source(),
            target=_source(),
            contract=contract,
        )
