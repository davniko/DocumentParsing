from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from document_ocr.synthesis.drafts import validate_allocation_arithmetic
from document_ocr.synthesis.generators import validate_container_number
from document_ocr.synthesis.structured_models import CargoGroupNumericProposal
from document_ocr.synthesis.structured_semantics import (
    apply_cargo_group_numeric_proposals,
    apply_date_proposal,
    apply_embedded_reference_date_shift,
    apply_identifier_plan,
    build_identifier_request_inventory,
    derive_scaled_group_quantities,
    finalize_non_linguistic_target,
    pending_realizations,
    reserve_structured_identifiers,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
)


def _rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]


def _eligible(row: dict[str, Any]) -> bool:
    patch = row["target"]["documentPatch"]
    return bool(
        patch.get("containers")
        and any(container.get("sealNumbers") for container in patch["containers"])
        and patch.get("cargoPackages")
        and all("quantity" in package for package in patch["cargoPackages"])
        and (patch.get("issueDate") or patch.get("shippedOnBoardDate"))
    )


def _numeric_proposals(target: dict[str, Any]) -> list[CargoGroupNumericProposal]:
    patch = target["documentPatch"]
    groups = {row["groupId"]: row for row in patch.get("cargoGroups", [])}
    package_groups: dict[str, list[dict[str, Any]]] = {}
    for package in patch.get("cargoPackages", []):
        package_groups.setdefault(package["groupId"], []).append(package)
    output: list[CargoGroupNumericProposal] = []
    for group_id, packages in package_groups.items():
        quantified = [row for row in packages if row.get("quantity") is not None]
        if not quantified:
            continue
        driver = quantified[0]
        quantities = derive_scaled_group_quantities(
            packages=packages,
            driver_package_id=driver["packageId"],
            generated_driver_quantity=driver["quantity"] * 2,
        )
        assert quantities is not None
        group = groups[group_id]
        output.append(
            CargoGroupNumericProposal(
                group_id=group_id,
                driver_package_id=driver["packageId"],
                quantity_by_package_id=quantities,
                gross_weight_value=(
                    group["grossWeight"]["value"] * 2
                    if group.get("grossWeight") is not None
                    else None
                ),
                net_weight_value=(
                    group["netWeight"]["value"] * 2 if group.get("netWeight") is not None else None
                ),
                volume_value=(
                    group["volume"]["value"] * 2 if group.get("volume") is not None else None
                ),
            )
        )
    return output


def test_complete_non_linguistic_semantics_change_every_supported_source_fact() -> None:
    rows = _rows()
    source_targets = {row["documentId"]: row["target"] for row in rows}
    source = next(row for row in rows if _eligible(row))
    document_id = source["documentId"]
    source_target = source["target"]
    target = deepcopy(source_target)

    inventory = build_identifier_request_inventory(
        source_targets=source_targets,
        selected_document_ids=(document_id,),
    )
    allocation_plan = reserve_structured_identifiers(
        all_source_targets=source_targets,
        inventory=inventory,
        seed=20260830,
    )
    changes = list(
        apply_identifier_plan(
            document_id=document_id,
            target=target,
            allocations=allocation_plan.values(),
        )
    )
    patch = target["documentPatch"]
    issue = date.fromisoformat(patch["issueDate"]) if patch.get("issueDate") else None
    shipped = (
        date.fromisoformat(patch["shippedOnBoardDate"]) if patch.get("shippedOnBoardDate") else None
    )
    changes.extend(
        apply_date_proposal(
            target=target,
            issue_date=issue + timedelta(days=17) if issue is not None else None,
            shipped_on_board_date=(shipped + timedelta(days=17) if shipped is not None else None),
        )
    )
    changes.extend(
        apply_cargo_group_numeric_proposals(
            target=target,
            proposals=_numeric_proposals(target),
        )
    )
    digest = finalize_non_linguistic_target(
        source_target=source_target,
        target=target,
        changes=changes,
    )
    pending = pending_realizations(
        target,
        changed_paths=tuple(change.target_path for change in changes),
    )

    all_real_containers = {
        container["containerNumber"]
        for row in rows
        for container in row["target"]["documentPatch"].get("containers", [])
    }
    generated = {container["containerNumber"] for container in patch.get("containers", [])}
    assert generated.isdisjoint(all_real_containers)
    assert all(validate_container_number(value) for value in generated)
    assert len(generated) == len(patch.get("containers", []))
    assert len(digest) == 64
    assert pending
    assert {row.kind for row in pending} <= {"registry", "linguistic"}


def test_embedded_reference_dates_are_valid_and_follow_document_date_shift() -> None:
    document_id = "doc_" + "1" * 64
    source = {
        "documentPatch": {
            "issueDate": "2025-05-28",
            "forwardingAndExportReferences": ["1900502 DT 16.05.2025"],
        }
    }
    target = deepcopy(source)
    path = f"{document_id}/documentPatch.forwardingAndExportReferences[0]"
    changes = list(
        apply_identifier_plan(
            document_id=document_id,
            target=target,
            allocations={path: "3910951 ZL 76.75.7979"},
        )
    )
    changes.extend(
        apply_date_proposal(
            target=target,
            issue_date=date(2023, 7, 7),
            shipped_on_board_date=None,
        )
    )

    receipts = apply_embedded_reference_date_shift(
        source_target=source,
        target=target,
        changes=changes,
        fallback_shift_days=17,
    )

    assert target["documentPatch"]["forwardingAndExportReferences"] == [
        "3910951 ZL 25.06.2023"
    ]
    assert receipts[0]["dateProjections"] == [
        {
            "sourceSurface": "16.05.2025",
            "targetSurface": "25.06.2023",
            "shiftDays": -691,
        }
    ]
    finalize_non_linguistic_target(
        source_target=source,
        target=target,
        changes=changes,
    )
    validate_allocation_arithmetic(target)


def test_non_calendar_reference_number_is_not_treated_as_an_embedded_date() -> None:
    document_id = "doc_" + "2" * 64
    source = {
        "documentPatch": {
            "forwardingAndExportReferences": ["REF 76.75.7979"],
        }
    }
    target = deepcopy(source)
    changes = list(
        apply_identifier_plan(
            document_id=document_id,
            target=target,
            allocations={
                f"{document_id}/documentPatch.forwardingAndExportReferences[0]": (
                    "ABC 12.34.5678"
                )
            },
        )
    )

    receipts = apply_embedded_reference_date_shift(
        source_target=source,
        target=target,
        changes=changes,
        fallback_shift_days=10,
    )

    assert receipts == ()
    assert target["documentPatch"]["forwardingAndExportReferences"] == ["ABC 12.34.5678"]
    finalize_non_linguistic_target(source_target=source, target=target, changes=changes)


def test_all_multi_package_quantities_and_allocations_are_reconciled_together() -> None:
    source = next(
        row
        for row in _rows()
        if len(row["target"]["documentPatch"].get("cargoPackages", [])) > 1
        and all(
            "quantity" in package for package in row["target"]["documentPatch"]["cargoPackages"]
        )
    )
    target = deepcopy(source["target"])
    packages = target["documentPatch"]["cargoPackages"]
    changes = apply_cargo_group_numeric_proposals(
        target=target,
        proposals=_numeric_proposals(target),
    )

    assert len([row for row in changes if row.family == "package_quantity"]) == len(packages)
    validate_allocation_arithmetic(target)


def test_absent_regenerated_or_resampled_facts_do_not_create_false_gaps() -> None:
    pending = pending_realizations(
        {"documentPatch": {"issueDate": None}},
        changed_paths=(),
    )

    assert all(row.target_path != "documentPatch.issueDate" for row in pending)


def test_package_proposals_fail_closed_on_partial_or_unchanged_values() -> None:
    source = next(row for row in _rows() if row["target"]["documentPatch"].get("cargoPackages"))
    target = deepcopy(source["target"])
    packages = [row for row in target["documentPatch"]["cargoPackages"] if "quantity" in row]
    if not packages:
        pytest.skip("fixture has no package quantity")
    with pytest.raises(ValueError, match="coverage mismatch"):
        apply_cargo_group_numeric_proposals(target=target, proposals=[])
    valid = _numeric_proposals(deepcopy(source["target"]))
    first = valid[0]
    invalid_quantities = dict(first.quantity_by_package_id)
    invalid_quantities[first.driver_package_id] //= 2
    invalid = first.model_copy(update={"quantity_by_package_id": invalid_quantities})
    with pytest.raises(ValueError, match="do not share one driver scale"):
        apply_cargo_group_numeric_proposals(
            target=deepcopy(source["target"]),
            proposals=[invalid, *valid[1:]],
        )


def test_unlinked_allocation_quantities_scale_without_erasing_membership() -> None:
    source = next(
        row
        for row in _rows()
        if _numeric_proposals(row["target"])
        and any(
            group["coverage"] == "unlinked_package_quantities"
            and any(
                package["groupId"] == group["groupId"] and package.get("quantity") is not None
                for package in row["target"]["documentPatch"].get("cargoPackages", [])
            )
            for group in row["target"]["documentPatch"].get("cargoAllocationGroups", [])
        )
    )
    target = deepcopy(source["target"])
    source_groups = {
        group["groupId"]: group
        for group in target["documentPatch"]["cargoAllocationGroups"]
        if group["coverage"] == "unlinked_package_quantities"
    }
    changes = apply_cargo_group_numeric_proposals(
        target=target,
        proposals=_numeric_proposals(target),
    )
    generated_groups = {
        group["groupId"]: group
        for group in target["documentPatch"]["cargoAllocationGroups"]
        if group["coverage"] == "unlinked_package_quantities"
    }

    assert source_groups
    for group_id, generated in generated_groups.items():
        source_values = [row["packageQuantity"] for row in source_groups[group_id]["allocations"]]
        generated_values = [row["packageQuantity"] for row in generated["allocations"]]
        assert all(value > 0 for value in generated_values)
        assert all(
            source_value != generated_value
            for source_value, generated_value in zip(source_values, generated_values, strict=True)
        )
    assert any(change.family == "allocation_quantity" for change in changes)
