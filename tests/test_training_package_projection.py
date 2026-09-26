from __future__ import annotations

from pathlib import Path

import pytest

from document_ocr.training.package_projection import (
    PackageProjectionError,
    PackageRolePolicy,
    ReviewedPackageRoleOverride,
    _project_relation_target,
    classify_package_role,
    diagnose_package_group,
    load_package_projection_config,
)


def _policy() -> PackageRolePolicy:
    return PackageRolePolicy.model_validate(
        {
            "normalized_outer_types": ("PALLET", "PALLETS", "PALLET S", "PLT", "CTNR"),
            "normalized_generic_types": ("PACKAGE", "PACKAGES", "PACKAGE S"),
        },
        strict=True,
    )


def test_exact_role_policy_does_not_misclassify_an_ibc_description() -> None:
    policy = _policy()

    assert classify_package_role({"typeDescription": "Pallet(s)"}, policy) == ("outer_transport")
    assert classify_package_role({"typeDescription": "PACKAGE(S)"}, policy) == ("generic_aggregate")
    assert (
        classify_package_role({"typeDescription": "1000 KG IBC (STEEL PALLETS)"}, policy)
        == "direct_goods"
    )
    assert classify_package_role({"typeCategory": "PACKAGE_PALLET"}, policy) == "outer_transport"
    assert classify_package_role({"typeCategory": "PACKAGE_PACKAGE"}, policy) == "generic_aggregate"
    assert classify_package_role({"typeCategory": "PACKAGE_PIECE"}, policy) == "direct_goods"


def test_explicit_outer_and_generic_levels_project_to_one_direct_type() -> None:
    diagnosis = diagnose_package_group(
        "g1",
        (
            {"packageId": "p1", "groupId": "g1", "quantity": 21, "typeDescription": "PACKAGES"},
            {"packageId": "p2", "groupId": "g1", "quantity": 3, "typeDescription": "PALLETS"},
            {"packageId": "p3", "groupId": "g1", "quantity": 120, "typeDescription": "BAGS"},
        ),
        _policy(),
    )

    assert diagnosis.status == "projected"
    assert diagnosis.retained_package_ids == ("p3",)
    assert diagnosis.metadata_package_ids == ("p1", "p2")


def test_distinct_direct_types_remain_when_outer_role_is_known() -> None:
    diagnosis = diagnose_package_group(
        "g1",
        (
            {"packageId": "p1", "groupId": "g1", "quantity": 13, "typeDescription": "CTNS"},
            {"packageId": "p2", "groupId": "g1", "quantity": 2550, "typeDescription": "PCS"},
            {"packageId": "p3", "groupId": "g1", "quantity": 1, "typeDescription": "PALLET"},
        ),
        _policy(),
    )

    assert diagnosis.status == "projected"
    assert diagnosis.retained_package_ids == ("p1", "p2")
    assert diagnosis.metadata_package_ids == ("p3",)


def test_multiple_legitimate_inner_types_remain_task_facing() -> None:
    diagnosis = diagnose_package_group(
        "g1",
        (
            {
                "packageId": "p1",
                "groupId": "g1",
                "quantity": 4,
                "typeDescription": "PALLETS",
            },
            {
                "packageId": "p2",
                "groupId": "g1",
                "quantity": 132,
                "typeDescription": "TINPLATE CONTAINERS",
            },
            {
                "packageId": "p3",
                "groupId": "g1",
                "quantity": 8,
                "typeDescription": "DRUMS",
            },
        ),
        _policy(),
    )

    assert diagnosis.status == "projected"
    assert diagnosis.retained_package_ids == ("p2", "p3")
    assert diagnosis.metadata_package_ids == ("p1",)


def test_reviewed_quantity_only_direct_fact_remains_without_inventing_a_type() -> None:
    quantity_only = {"packageId": "p1", "groupId": "g1", "quantity": 1}
    override = ReviewedPackageRoleOverride.model_validate(
        {
            "document_id": f"doc_{'1' * 64}",
            "group_id": "g1",
            "package_id": "p1",
            "expected_package_sha256": (
                "7650e6cb7e8605989fe6ff7950f31b1b6c32f5182d06709fe4372abce972993b"
            ),
            "role": "direct_goods",
            "rationale": "The printed row links quantity one to the cargo.",
        },
        strict=True,
    )

    diagnosis = diagnose_package_group(
        "g1",
        (
            quantity_only,
            {
                "packageId": "p2",
                "groupId": "g1",
                "quantity": 27,
                "typeDescription": "PCS",
            },
        ),
        _policy(),
        {"p1": override},
    )

    assert diagnosis.status == "retained_non_hierarchical"
    assert diagnosis.retained_package_ids == ("p1", "p2")
    assert diagnosis.metadata_package_ids == ()
    assert diagnosis.packages[0].value == quantity_only
    assert diagnosis.packages[0].normalized_type is None
    assert diagnosis.packages[0].role_source == "reviewed_override"


def test_quantity_only_fact_without_review_remains_held() -> None:
    diagnosis = diagnose_package_group(
        "g1",
        (
            {"packageId": "p1", "groupId": "g1", "quantity": 1},
            {"packageId": "p2", "groupId": "g1", "quantity": 27, "typeDescription": "PCS"},
        ),
        _policy(),
    )

    assert diagnosis.status == "held_ambiguous"


def test_reviewed_role_override_fails_when_the_source_package_changes() -> None:
    override = ReviewedPackageRoleOverride.model_validate(
        {
            "document_id": f"doc_{'1' * 64}",
            "group_id": "g1",
            "package_id": "p1",
            "expected_package_sha256": "a" * 64,
            "role": "direct_goods",
            "rationale": "The printed row links quantity one to the cargo.",
        },
        strict=True,
    )

    with pytest.raises(PackageProjectionError, match="source hash differs"):
        diagnose_package_group(
            "g1",
            ({"packageId": "p1", "groupId": "g1", "quantity": 1},),
            _policy(),
            {"p1": override},
        )


def test_removed_outer_allocation_is_not_transferred_to_inner_quantity() -> None:
    target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "containers": [{"containerNumber": "TCLU6905627"}],
            "cargoGroups": [{"groupId": "g1", "description": "WIDGETS"}],
            "cargoPackages": [
                {"packageId": "p1", "groupId": "g1", "quantity": 2, "typeDescription": "PALLETS"},
                {"packageId": "p2", "groupId": "g1", "quantity": 100, "typeDescription": "CARTONS"},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageIds": ["p1"],
                    "allocations": [{"containerNumber": "TCLU6905627", "packageQuantity": 2}],
                }
            ],
        },
    }
    diagnosis = diagnose_package_group("g1", target["documentPatch"]["cargoPackages"], _policy())

    projected, audit = _project_relation_target(target, (diagnosis,))

    assert projected["documentPatch"]["cargoPackages"] == [
        {
            "packageId": "p1",
            "groupId": "g1",
            "quantity": 100,
            "typeDescription": "CARTONS",
        }
    ]
    assert projected["documentPatch"]["cargoAllocationGroups"] == [
        {
            "groupId": "g1",
            "coverage": "container_membership_only",
            "packageIds": [],
            "allocations": [{"containerNumber": "TCLU6905627"}],
        }
    ]
    assert audit[0]["decision"] == "downgraded_to_container_membership"


def test_retained_package_allocation_keeps_printed_quantity() -> None:
    target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "containers": [{"containerNumber": "TCLU6905627"}],
            "cargoGroups": [{"groupId": "g1", "description": "WIDGETS"}],
            "cargoPackages": [
                {"packageId": "p1", "groupId": "g1", "quantity": 100, "typeDescription": "CARTONS"},
                {"packageId": "p2", "groupId": "g1", "quantity": 2, "typeDescription": "PALLETS"},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageIds": ["p1"],
                    "allocations": [{"containerNumber": "TCLU6905627", "packageQuantity": 100}],
                }
            ],
        },
    }
    diagnosis = diagnose_package_group("g1", target["documentPatch"]["cargoPackages"], _policy())

    projected, audit = _project_relation_target(target, (diagnosis,))

    assert projected["documentPatch"]["cargoPackages"] == [
        {"packageId": "p1", "groupId": "g1", "quantity": 100, "typeDescription": "CARTONS"}
    ]
    assert projected["documentPatch"]["cargoAllocationGroups"] == [
        {
            "groupId": "g1",
            "coverage": "single_package_level",
            "packageIds": ["p1"],
            "allocations": [{"containerNumber": "TCLU6905627", "packageQuantity": 100}],
        }
    ]
    assert audit[0]["decision"] == "preserved"


def test_repository_projection_config_is_pinned_and_loadable() -> None:
    config = load_package_projection_config(
        Path("configs/transforms/mpci_bl_combined1157_task_facing_packages_v2.yaml")
    )

    assert config.source.expected_records == 1157
    assert config.output.dataset_id == "mpci-bl-combined1157-task-facing-packages-v2"
    assert len(config.reviewed_role_overrides) == 1
    assert "1000 KG IBC STEEL PALLETS" not in config.policy.normalized_outer_types
