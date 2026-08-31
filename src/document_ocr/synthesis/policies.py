"""Exhaustive task-facing field policies for cardinality-preserving B/L synthesis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingRelationExplicitLabel
from document_ocr.synthesis.anchors import leaf_items, normalized_role_path
from document_ocr.synthesis.generation_models import FieldPolicy


def _policy(
    role_path: str,
    value_policy: str,
    status: str,
    method: str,
    coupling_group: str,
    *,
    must_be_absent: bool = False,
) -> FieldPolicy:
    return FieldPolicy.model_validate(
        {
            "role_path": role_path,
            "value_policy": value_policy,
            "missingness_policy": "must_be_absent" if must_be_absent else "preserve_source",
            "implementation_status": status,
            "method": method,
            "coupling_group": coupling_group,
        },
        strict=True,
    )


def _registry() -> dict[str, FieldPolicy]:
    rows: dict[str, FieldPolicy] = {}

    def add(
        path: str,
        value_policy: str,
        status: str,
        method: str,
        coupling_group: str,
        *,
        must_be_absent: bool = False,
    ) -> None:
        if path in rows:
            raise RuntimeError(f"duplicate synthesis field policy: {path}")
        rows[path] = _policy(
            path,
            value_policy,
            status,
            method,
            coupling_group,
            must_be_absent=must_be_absent,
        )

    add(
        "schemaVersion",
        "preserve_nonidentifying",
        "structural",
        "preserve_relation_v3_schema_version",
        "schema",
    )
    for name in (
        "billOfLadingNumber",
        "originalBillOfLadingNumber",
        "masterBillOfLadingNumber",
    ):
        add(
            f"documentPatch.{name}",
            "regenerate",
            "implemented",
            "shape_preserving_identifier_v1",
            "document_identifiers",
        )
    for name in ("issueDate", "shippedOnBoardDate"):
        add(
            f"documentPatch.{name}",
            "regenerate",
            "implemented",
            "joint_bounded_date_shift_v1",
            "document_dates",
        )
    add(
        "documentPatch.negotiability",
        "preserve_nonidentifying",
        "structural",
        "preserve_document_release_semantics_v1",
        "document_release_semantics",
    )

    location_roles = (
        "placeOfIssue",
        "route.placeOfReceipt",
        "route.portOfLoading",
        "route.transshipmentPort",
        "route.portOfDischarge",
        "route.placeOfDelivery",
        "route.finalDestination",
    )
    for role in location_roles:
        for field in ("name", "country"):
            add(
                f"documentPatch.{role}.{field}",
                "derive",
                "pending_registry",
                "route_scenario_projection_v1",
                "route_scenario",
            )

    add(
        "documentPatch.transport.vesselName",
        "resample",
        "pending_registry",
        "vessel_registry_entity_v1",
        "vessel_entity",
    )
    for field in ("vesselImoNumber", "vesselFlagCountry"):
        add(
            f"documentPatch.transport.{field}",
            "derive",
            "pending_registry",
            "vessel_registry_entity_v1",
            "vessel_entity",
        )
    add(
        "documentPatch.transport.voyageNumber",
        "regenerate",
        "implemented",
        "shape_preserving_identifier_v1",
        "voyage_identifier",
    )
    add(
        "documentPatch.freight.paymentArrangement",
        "resample",
        "pending_registry",
        "source_supported_freight_arrangement_v1",
        "freight_terms",
    )
    for field in ("name", "country"):
        add(
            f"documentPatch.freight.paymentPlace.{field}",
            "derive",
            "pending_registry",
            "freight_scenario_projection_v1",
            "freight_terms",
        )

    supported_party_roles = (
        "shipper",
        "consignee",
        "notifyParties[]",
        "carrier",
        "forwardingAgent",
        "deliveryAgent",
    )
    all_party_roles = (*supported_party_roles, "consolidator")
    for role in all_party_roles:
        prefix = f"documentPatch.parties.{role}"
        if role == "notifyParties[]":
            add(
                f"{prefix}.sameAs",
                "derive",
                "structural",
                "preserve_party_relation_topology_v1",
                f"party:{role}",
            )
        else:
            add(
                f"{prefix}.sameAs",
                "forbidden",
                "disabled",
                "notify_only_same_as_contract",
                f"party:{role}",
                must_be_absent=True,
            )
        for field in (
            "name",
            "address",
            "city",
            "country",
            "contactDetails.contactName",
            "contactDetails.phoneNumbers[]",
            "contactDetails.emailAddresses[]",
            "contactDetails.websiteUrls[]",
        ):
            path = f"{prefix}.{field}"
            if role == "consolidator":
                add(
                    path,
                    "legitimately_absent",
                    "disabled",
                    "zero_support_in_pinned_corpus",
                    f"party:{role}",
                    must_be_absent=True,
                )
            elif field == "address":
                add(
                    path,
                    "pending_linguistic",
                    "pending_linguistic",
                    "route_conditioned_party_address_v1",
                    f"party:{role}",
                )
            elif field in {"name", "contactDetails.contactName"}:
                add(
                    path,
                    "resample",
                    "pending_registry",
                    "reserved_party_entity_seed_v1",
                    f"party:{role}",
                )
            elif field == "contactDetails.phoneNumbers[]":
                add(
                    path,
                    "regenerate",
                    "pending_registry",
                    "party_entity_contact_projection_v1",
                    f"party:{role}",
                )
            else:
                add(
                    path,
                    "derive",
                    "pending_registry",
                    "party_entity_route_projection_v1",
                    f"party:{role}",
                )

    add(
        "documentPatch.containers[].containerNumber",
        "regenerate",
        "implemented",
        "preserve_owner_prefix_iso6346_v1",
        "container_graph",
    )
    add(
        "documentPatch.containers[].typeDescription",
        "derive",
        "pending_registry",
        "equipment_registry_surface_projection_v1",
        "container_equipment",
    )
    add(
        "documentPatch.containers[].typeCategory",
        "legitimately_absent",
        "disabled",
        "empty_bound_container_vocabulary",
        "container_equipment",
        must_be_absent=True,
    )
    for field in ("value", "unit"):
        add(
            f"documentPatch.containers[].verifiedGrossMass.{field}",
            "legitimately_absent",
            "disabled",
            "zero_support_in_pinned_corpus",
            "container_vgm",
            must_be_absent=True,
        )
    add(
        "documentPatch.containers[].sealNumbers[]",
        "regenerate",
        "implemented",
        "shape_preserving_seal_v1",
        "container_graph",
    )
    add(
        "documentPatch.containers[].temperatureSetpoint.value",
        "resample",
        "pending_registry",
        "reefer_compatible_temperature_v1",
        "container_equipment",
    )
    add(
        "documentPatch.containers[].temperatureSetpoint.unit",
        "preserve_nonidentifying",
        "structural",
        "preserve_temperature_unit_v1",
        "container_equipment",
    )
    add(
        "documentPatch.forwardingAndExportReferences[]",
        "regenerate",
        "implemented",
        "shape_preserving_identifier_if_scalar_v1",
        "document_references",
    )

    add(
        "documentPatch.cargoGroups[].groupId",
        "derive",
        "structural",
        "preserve_contiguous_group_identity_v1",
        "cargo_graph",
    )
    for field in ("description", "additionalInformation[]", "marksAndNumbers[]"):
        add(
            f"documentPatch.cargoGroups[].{field}",
            "pending_linguistic",
            "pending_linguistic",
            "cargo_fact_conditioned_text_v1",
            "cargo_semantics",
        )
    for measure in ("grossWeight", "netWeight", "volume"):
        add(
            f"documentPatch.cargoGroups[].{measure}.value",
            "derive",
            "implemented",
            "profile_routed_per_driver_measure_then_exact_unit_inverse_v1",
            "cargo_measures",
        )
        add(
            f"documentPatch.cargoGroups[].{measure}.unit",
            "preserve_nonidentifying",
            "structural",
            "preserve_measure_unit_v1",
            "cargo_measures",
        )
    add(
        "documentPatch.cargoGroups[].hsCodes[]",
        "resample",
        "pending_registry",
        "hs_product_registry_v1",
        "cargo_semantics",
    )
    add(
        "documentPatch.cargoGroups[].handlingInstructions[]",
        "resample",
        "pending_registry",
        "controlled_handling_catalog_v1",
        "cargo_semantics",
    )
    add(
        "documentPatch.cargoGroups[].dangerousGoods[].unNumber",
        "resample",
        "pending_registry",
        "dangerous_goods_registry_tuple_v1",
        "dangerous_goods_tuple",
    )
    for field in (
        "hazardCategory",
        "subsidiaryHazardCategory",
        "flashPoint.packingGroupCategory",
        "flashPoint.temperature.value",
    ):
        add(
            f"documentPatch.cargoGroups[].dangerousGoods[].{field}",
            "derive",
            "pending_registry",
            "dangerous_goods_registry_tuple_v1",
            "dangerous_goods_tuple",
        )
    add(
        "documentPatch.cargoGroups[].dangerousGoods[].flashPoint.temperature.unit",
        "preserve_nonidentifying",
        "structural",
        "preserve_temperature_unit_v1",
        "dangerous_goods_tuple",
    )
    for field in ("name", "identifier"):
        add(
            f"documentPatch.cargoGroups[].origin.{field}",
            "derive",
            "pending_registry",
            "cargo_route_origin_projection_v1",
            "cargo_semantics",
        )

    for field, value_policy, status, method in (
        ("packageId", "derive", "structural", "preserve_contiguous_package_identity_v1"),
        ("groupId", "derive", "structural", "preserve_package_group_reference_v1"),
        ("quantity", "resample", "implemented", "empirical_same_category_quantity_v1"),
        ("typeCategory", "resample", "pending_registry", "package_registry_v1"),
        ("typeDescription", "derive", "pending_registry", "package_registry_surface_v1"),
    ):
        add(
            f"documentPatch.cargoPackages[].{field}",
            value_policy,
            status,
            method,
            "cargo_graph",
        )

    add(
        "documentPatch.cargoAllocationGroups[].groupId",
        "derive",
        "structural",
        "preserve_allocation_group_reference_v1",
        "allocation_graph",
    )
    add(
        "documentPatch.cargoAllocationGroups[].coverage",
        "preserve_nonidentifying",
        "structural",
        "preserve_relation_topology_v1",
        "allocation_graph",
    )
    for field, method in (
        ("packageIds[]", "derive_allocation_package_membership_v1"),
        ("allocations[].containerNumber", "derive_regenerated_container_reference_v1"),
        ("allocations[].packageQuantity", "derive_reconciled_allocation_quantity_v1"),
        ("allocations[].packageId", "derive_allocation_package_reference_v1"),
    ):
        add(
            f"documentPatch.cargoAllocationGroups[].{field}",
            "derive",
            "implemented" if "container" in field or "Quantity" in field else "structural",
            method,
            "allocation_graph",
        )
    return rows


FIELD_POLICIES = _registry()


def schema_leaf_paths(schema: Mapping[str, Any]) -> frozenset[str]:
    """Expand object/ref/array JSON Schema branches into normalized leaf paths."""

    definitions = schema.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValueError("task schema has no definitions map")

    def walk(node: Mapping[str, Any], path: str, stack: tuple[str, ...]) -> set[str]:
        reference = node.get("$ref")
        if isinstance(reference, str):
            prefix = "#/$defs/"
            if not reference.startswith(prefix):
                raise ValueError(f"unsupported task-schema reference: {reference}")
            name = reference.removeprefix(prefix)
            if name in stack:
                raise ValueError(f"recursive task-schema reference: {name}")
            resolved = definitions.get(name)
            if not isinstance(resolved, dict):
                raise ValueError(f"missing task-schema definition: {name}")
            return walk(resolved, path, (*stack, name))
        alternatives = node.get("anyOf")
        if isinstance(alternatives, list):
            output: set[str] = set()
            for child in alternatives:
                if not isinstance(child, dict) or child.get("type") == "null":
                    continue
                output.update(walk(child, path, stack))
            return output
        properties = node.get("properties")
        if isinstance(properties, dict):
            output = set()
            for name, child in properties.items():
                if not isinstance(child, dict):
                    raise ValueError(f"invalid schema property: {name}")
                output.update(walk(child, f"{path}.{name}" if path else name, stack))
            return output
        if node.get("type") == "array":
            items = node.get("items")
            if not isinstance(items, dict):
                raise ValueError(f"array schema has no item contract: {path}")
            return walk(items, path + "[]", stack)
        if not path:
            raise ValueError("task schema root resolved to a scalar")
        return {path}

    return frozenset(walk(schema, "", ()))


def validate_policy_registry(targets: tuple[Mapping[str, Any], ...]) -> dict[str, int]:
    """Prove schema and observed target leaves are covered exactly once."""

    schema_paths = schema_leaf_paths(
        BillOfLadingRelationExplicitLabel.model_json_schema(mode="serialization")
    )
    policy_paths = frozenset(FIELD_POLICIES)
    if schema_paths != policy_paths:
        missing = sorted(schema_paths - policy_paths)
        extra = sorted(policy_paths - schema_paths)
        raise ValueError(f"field-policy/schema mismatch: missing={missing}, extra={extra}")
    observed = {
        normalized_role_path(path) for target in targets for path, _value in leaf_items(target)
    }
    outside = sorted(observed - schema_paths)
    if outside:
        raise ValueError(f"observed target leaves are outside the task schema: {outside}")
    for path, policy in FIELD_POLICIES.items():
        if path in observed and policy.missingness_policy == "must_be_absent":
            raise ValueError(f"must-be-absent field is present in source corpus: {path}")
    counts: dict[str, int] = {}
    for policy in FIELD_POLICIES.values():
        counts[policy.value_policy] = counts.get(policy.value_policy, 0) + 1
    return {
        "schema_leaf_families": len(schema_paths),
        "observed_leaf_families": len(observed),
        "zero_support_leaf_families": len(schema_paths - observed),
        **{f"policy_{name}": count for name, count in sorted(counts.items())},
    }


def policy_for_target_path(path: str) -> FieldPolicy:
    role_path = normalized_role_path(path)
    try:
        return FIELD_POLICIES[role_path]
    except KeyError as error:
        raise ValueError(f"target path has no synthesis policy: {role_path}") from error
