"""Lossless relational projection for the relation-explicit B/L target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from document_ocr.synthesis.domain import RelationalTables

_PARTY_ROLES = (
    "shipper",
    "consignee",
    "notifyParties",
    "carrier",
    "forwardingAgent",
    "deliveryAgent",
    "consolidator",
)
_CONTACT_FIELDS = (
    ("phoneNumbers", "phone"),
    ("emailAddresses", "email"),
    ("websiteUrls", "website"),
)
_ROUTE_FIELDS = (
    "placeOfReceipt",
    "portOfLoading",
    "transshipmentPort",
    "portOfDischarge",
    "placeOfDelivery",
    "finalDestination",
)
_CARGO_VALUE_FIELDS = (
    ("additionalInformation", "additional_information"),
    ("marksAndNumbers", "marks_and_numbers"),
    ("hsCodes", "hs_code"),
    ("handlingInstructions", "handling_instruction"),
)


def _row_id(document_id: str, kind: str, *parts: object) -> str:
    suffix = ":".join(str(part) for part in parts)
    return f"{document_id}:{kind}:{suffix}"


def _measure_columns(prefix: str, value: object) -> dict[str, Any]:
    measure = value if isinstance(value, dict) else {}
    return {f"{prefix}_value": measure.get("value"), f"{prefix}_unit": measure.get("unit")}


def _measure_from_row(row: Mapping[str, Any], prefix: str) -> dict[str, Any] | None:
    value = row.get(f"{prefix}_value")
    unit = row.get(f"{prefix}_unit")
    if value is None and unit is None:
        return None
    return {"value": value, "unit": unit}


def _ordered(rows: Sequence[Mapping[str, Any]], field: str) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: cast(int, row[field]))


class BillOfLadingRelationDomainAdapter:
    """Task-specific, cardinality-preserving B/L relational contract."""

    task = "bill_of_lading_relation_explicit_v3"
    table_order = (
        "documents",
        "document_locations",
        "document_references",
        "parties",
        "party_contacts",
        "containers",
        "container_seals",
        "cargo_groups",
        "cargo_additional_information",
        "cargo_marks_numbers",
        "cargo_hs_codes",
        "cargo_handling_instructions",
        "packages",
        "allocation_groups",
        "allocation_group_packages",
        "allocations",
        "dangerous_goods",
    )

    def project(
        self, *, document_id: str, source_row_index: int, target: Mapping[str, Any]
    ) -> RelationalTables:
        patch = target.get("documentPatch")
        if target.get("schemaVersion") != "3.0.0-experimental" or not isinstance(patch, dict):
            raise ValueError("B/L domain projection requires a canonical relation-v3 target")
        tables = RelationalTables({name: [] for name in self.table_order})
        route_value = patch.get("route")
        transport_value = patch.get("transport")
        freight_value = patch.get("freight")
        route: Mapping[str, Any] = route_value if isinstance(route_value, dict) else {}
        transport: Mapping[str, Any] = transport_value if isinstance(transport_value, dict) else {}
        freight: Mapping[str, Any] = freight_value if isinstance(freight_value, dict) else {}
        document_row: dict[str, Any] = {
            "document_id": document_id,
            "source_row_index": source_row_index,
            "schema_version": target["schemaVersion"],
            "bill_of_lading_number": patch.get("billOfLadingNumber"),
            "original_bill_of_lading_number": patch.get("originalBillOfLadingNumber"),
            "master_bill_of_lading_number": patch.get("masterBillOfLadingNumber"),
            "issue_date": patch.get("issueDate"),
            "shipped_on_board_date": patch.get("shippedOnBoardDate"),
            "negotiability": patch.get("negotiability"),
            "transport_vessel_name": transport.get("vesselName"),
            "transport_vessel_imo_number": transport.get("vesselImoNumber"),
            "transport_voyage_number": transport.get("voyageNumber"),
            "transport_vessel_flag_country": transport.get("vesselFlagCountry"),
            "freight_payment_arrangement": freight.get("paymentArrangement"),
        }
        tables.add("documents", document_row)

        locations = (
            ("placeOfIssue", patch.get("placeOfIssue")),
            *((name, route.get(name)) for name in _ROUTE_FIELDS),
            ("paymentPlace", freight.get("paymentPlace")),
        )
        for location_order, (role, value) in enumerate(locations):
            if not isinstance(value, dict):
                continue
            tables.add(
                "document_locations",
                {
                    "location_id": _row_id(document_id, "location", role),
                    "document_id": document_id,
                    "location_order": location_order,
                    "location_role": role,
                    "name": value.get("name"),
                    "country": value.get("country"),
                },
            )

        for index, value in enumerate(patch.get("forwardingAndExportReferences") or []):
            tables.add(
                "document_references",
                {
                    "reference_id": _row_id(document_id, "reference", index),
                    "document_id": document_id,
                    "reference_order": index,
                    "value": value,
                },
            )

        parties_value = patch.get("parties")
        parties: Mapping[str, Any] = parties_value if isinstance(parties_value, dict) else {}
        for role_order, role in enumerate(_PARTY_ROLES):
            raw_values = parties.get(role)
            values = raw_values if role == "notifyParties" else [raw_values]
            if not isinstance(values, list):
                values = []
            for occurrence, party in enumerate(values):
                if not isinstance(party, dict):
                    continue
                party_id = _row_id(document_id, "party", role, occurrence)
                contact_value = party.get("contactDetails")
                contact: Mapping[str, Any] = (
                    contact_value if isinstance(contact_value, dict) else {}
                )
                tables.add(
                    "parties",
                    {
                        "party_id": party_id,
                        "document_id": document_id,
                        "role": role,
                        "role_order": role_order,
                        "occurrence_order": occurrence,
                        "same_as": party.get("sameAs"),
                        "name": party.get("name"),
                        "address": party.get("address"),
                        "city": party.get("city"),
                        "country": party.get("country"),
                        "contact_name": contact.get("contactName"),
                    },
                )
                for field_name, kind in _CONTACT_FIELDS:
                    for contact_order, value in enumerate(contact.get(field_name) or []):
                        tables.add(
                            "party_contacts",
                            {
                                "party_contact_id": _row_id(
                                    document_id,
                                    "party-contact",
                                    role,
                                    occurrence,
                                    kind,
                                    contact_order,
                                ),
                                "document_id": document_id,
                                "party_id": party_id,
                                "contact_type": kind,
                                "contact_order": contact_order,
                                "value": value,
                            },
                        )

        container_row_ids: dict[str, str] = {}
        for container_order, container in enumerate(patch.get("containers") or []):
            container_id = _row_id(document_id, "container", container_order)
            container_row_ids[container["containerNumber"]] = container_id
            tables.add(
                "containers",
                {
                    "container_id": container_id,
                    "document_id": document_id,
                    "container_order": container_order,
                    "container_number": container.get("containerNumber"),
                    "type_description": container.get("typeDescription"),
                    "type_category": container.get("typeCategory"),
                    **_measure_columns("verified_gross_mass", container.get("verifiedGrossMass")),
                    **_measure_columns(
                        "temperature_setpoint", container.get("temperatureSetpoint")
                    ),
                },
            )
            for seal_order, value in enumerate(container.get("sealNumbers") or []):
                tables.add(
                    "container_seals",
                    {
                        "container_seal_id": _row_id(
                            document_id, "container-seal", container_order, seal_order
                        ),
                        "document_id": document_id,
                        "container_id": container_id,
                        "seal_order": seal_order,
                        "value": value,
                    },
                )

        cargo_group_row_ids: dict[str, str] = {}
        for group_order, group in enumerate(patch.get("cargoGroups") or []):
            group_id = group["groupId"]
            cargo_group_row_id = _row_id(document_id, "cargo-group", group_id)
            cargo_group_row_ids[group_id] = cargo_group_row_id
            origin = group.get("origin") if isinstance(group.get("origin"), dict) else {}
            tables.add(
                "cargo_groups",
                {
                    "cargo_group_row_id": cargo_group_row_id,
                    "document_id": document_id,
                    "group_order": group_order,
                    "group_id": group_id,
                    "description": group.get("description"),
                    **_measure_columns("gross_weight", group.get("grossWeight")),
                    **_measure_columns("net_weight", group.get("netWeight")),
                    **_measure_columns("volume", group.get("volume")),
                    "origin_name": origin.get("name"),
                    "origin_identifier": origin.get("identifier"),
                },
            )
            for field_name, value_type in _CARGO_VALUE_FIELDS:
                for value_order, value in enumerate(group.get(field_name) or []):
                    table = {
                        "additional_information": "cargo_additional_information",
                        "marks_and_numbers": "cargo_marks_numbers",
                        "hs_code": "cargo_hs_codes",
                        "handling_instruction": "cargo_handling_instructions",
                    }[value_type]
                    tables.add(
                        table,
                        {
                            "cargo_group_value_id": _row_id(
                                document_id, "cargo-value", group_id, value_type, value_order
                            ),
                            "document_id": document_id,
                            "cargo_group_row_id": cargo_group_row_id,
                            "value_order": value_order,
                            "value": value,
                        },
                    )
            for dangerous_order, dangerous in enumerate(group.get("dangerousGoods") or []):
                flash = (
                    dangerous.get("flashPoint")
                    if isinstance(dangerous.get("flashPoint"), dict)
                    else {}
                )
                tables.add(
                    "dangerous_goods",
                    {
                        "dangerous_goods_id": _row_id(
                            document_id, "dangerous-goods", group_id, dangerous_order
                        ),
                        "document_id": document_id,
                        "cargo_group_row_id": cargo_group_row_id,
                        "dangerous_goods_order": dangerous_order,
                        "un_number": dangerous.get("unNumber"),
                        "hazard_category": dangerous.get("hazardCategory"),
                        "subsidiary_hazard_category": dangerous.get("subsidiaryHazardCategory"),
                        **_measure_columns("flash_point", flash.get("temperature")),
                        "packing_group_category": flash.get("packingGroupCategory"),
                    },
                )

        package_row_ids: dict[str, str] = {}
        for package_order, package in enumerate(patch.get("cargoPackages") or []):
            package_row_id = _row_id(document_id, "package", package["packageId"])
            package_row_ids[package["packageId"]] = package_row_id
            tables.add(
                "packages",
                {
                    "package_row_id": package_row_id,
                    "document_id": document_id,
                    "package_order": package_order,
                    "package_id": package["packageId"],
                    "group_id": package["groupId"],
                    "cargo_group_row_id": cargo_group_row_ids[package["groupId"]],
                    "quantity": package.get("quantity"),
                    "type_category": package.get("typeCategory"),
                    "type_description": package.get("typeDescription"),
                },
            )

        for allocation_group_order, allocation_group in enumerate(
            patch.get("cargoAllocationGroups") or []
        ):
            group_id = allocation_group["groupId"]
            allocation_group_id = _row_id(document_id, "allocation-group", group_id)
            tables.add(
                "allocation_groups",
                {
                    "allocation_group_id": allocation_group_id,
                    "document_id": document_id,
                    "allocation_group_order": allocation_group_order,
                    "group_id": group_id,
                    "cargo_group_row_id": cargo_group_row_ids[group_id],
                    "coverage": allocation_group["coverage"],
                },
            )
            for package_order, package_id in enumerate(allocation_group.get("packageIds") or []):
                tables.add(
                    "allocation_group_packages",
                    {
                        "allocation_group_package_id": _row_id(
                            document_id, "allocation-package", group_id, package_order
                        ),
                        "document_id": document_id,
                        "allocation_group_id": allocation_group_id,
                        "package_order": package_order,
                        "package_id": package_id,
                        "package_row_id": package_row_ids[package_id],
                    },
                )
            for allocation_order, allocation in enumerate(allocation_group["allocations"]):
                tables.add(
                    "allocations",
                    {
                        "allocation_id": _row_id(
                            document_id, "allocation", group_id, allocation_order
                        ),
                        "document_id": document_id,
                        "allocation_group_id": allocation_group_id,
                        "allocation_order": allocation_order,
                        "container_number": allocation["containerNumber"],
                        "container_id": container_row_ids[allocation["containerNumber"]],
                        "package_quantity": allocation.get("packageQuantity"),
                        "package_id": allocation.get("packageId"),
                        "package_row_id": (
                            package_row_ids[allocation["packageId"]]
                            if allocation.get("packageId") is not None
                            else None
                        ),
                    },
                )
        return tables

    def reconstruct(
        self, *, document_id: str, tables: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, Any]:
        documents = [
            row for row in tables.get("documents", ()) if row["document_id"] == document_id
        ]
        if len(documents) != 1:
            raise ValueError(f"expected one document row for {document_id}, found {len(documents)}")
        row = documents[0]
        patch: dict[str, Any] = {}
        scalar_fields = {
            "billOfLadingNumber": "bill_of_lading_number",
            "originalBillOfLadingNumber": "original_bill_of_lading_number",
            "masterBillOfLadingNumber": "master_bill_of_lading_number",
            "issueDate": "issue_date",
            "shippedOnBoardDate": "shipped_on_board_date",
            "negotiability": "negotiability",
        }
        for target_name, column in scalar_fields.items():
            if row.get(column) is not None:
                patch[target_name] = row[column]
        location_rows = _ordered(
            [r for r in tables.get("document_locations", ()) if r["document_id"] == document_id],
            "location_order",
        )
        locations = {
            cast(str, location["location_role"]): {
                key: location[key] for key in ("name", "country") if location.get(key) is not None
            }
            for location in location_rows
        }
        if "placeOfIssue" in locations:
            patch["placeOfIssue"] = locations["placeOfIssue"]
        route = {name: locations[name] for name in _ROUTE_FIELDS if name in locations}
        if route:
            patch["route"] = route
        transport = {
            key: row.get(column)
            for key, column in {
                "vesselName": "transport_vessel_name",
                "vesselImoNumber": "transport_vessel_imo_number",
                "voyageNumber": "transport_voyage_number",
                "vesselFlagCountry": "transport_vessel_flag_country",
            }.items()
            if row.get(column) is not None
        }
        if transport:
            patch["transport"] = transport
        freight = {}
        if row.get("freight_payment_arrangement") is not None:
            freight["paymentArrangement"] = row["freight_payment_arrangement"]
        if "paymentPlace" in locations:
            freight["paymentPlace"] = locations["paymentPlace"]
        if freight:
            patch["freight"] = freight

        references = _ordered(
            [r for r in tables.get("document_references", ()) if r["document_id"] == document_id],
            "reference_order",
        )
        if references:
            patch["forwardingAndExportReferences"] = [r["value"] for r in references]

        contact_rows = [
            r for r in tables.get("party_contacts", ()) if r["document_id"] == document_id
        ]
        party_rows = sorted(
            [r for r in tables.get("parties", ()) if r["document_id"] == document_id],
            key=lambda r: (cast(int, r["role_order"]), cast(int, r["occurrence_order"])),
        )
        parties: dict[str, Any] = {}
        for party_row in party_rows:
            party = {
                target_name: party_row.get(column)
                for target_name, column in {
                    "sameAs": "same_as",
                    "name": "name",
                    "address": "address",
                    "city": "city",
                    "country": "country",
                }.items()
                if party_row.get(column) is not None
            }
            contact: dict[str, Any] = {}
            if party_row.get("contact_name") is not None:
                contact["contactName"] = party_row["contact_name"]
            by_kind = {
                kind: _ordered(
                    [
                        r
                        for r in contact_rows
                        if r["party_id"] == party_row["party_id"] and r["contact_type"] == kind
                    ],
                    "contact_order",
                )
                for kind in ("phone", "email", "website")
            }
            for kind, target_name in (
                ("phone", "phoneNumbers"),
                ("email", "emailAddresses"),
                ("website", "websiteUrls"),
            ):
                if by_kind[kind]:
                    contact[target_name] = [r["value"] for r in by_kind[kind]]
            if contact:
                party["contactDetails"] = contact
            role = cast(str, party_row["role"])
            if role == "notifyParties":
                parties.setdefault(role, []).append(party)
            else:
                parties[role] = party
        if parties:
            patch["parties"] = parties

        seal_rows = [
            r for r in tables.get("container_seals", ()) if r["document_id"] == document_id
        ]
        containers = []
        for container_row in _ordered(
            [r for r in tables.get("containers", ()) if r["document_id"] == document_id],
            "container_order",
        ):
            container = {
                "containerNumber": container_row["container_number"],
                **{
                    name: container_row[column]
                    for name, column in (
                        ("typeDescription", "type_description"),
                        ("typeCategory", "type_category"),
                    )
                    if container_row.get(column) is not None
                },
            }
            vgm = _measure_from_row(container_row, "verified_gross_mass")
            temperature = _measure_from_row(container_row, "temperature_setpoint")
            if vgm is not None:
                container["verifiedGrossMass"] = vgm
            values = _ordered(
                [r for r in seal_rows if r["container_id"] == container_row["container_id"]],
                "seal_order",
            )
            if values:
                container["sealNumbers"] = [r["value"] for r in values]
            if temperature is not None:
                container["temperatureSetpoint"] = temperature
            containers.append(container)
        if containers:
            patch["containers"] = containers

        value_rows = {
            "additional_information": [
                r
                for r in tables.get("cargo_additional_information", ())
                if r["document_id"] == document_id
            ],
            "marks_and_numbers": [
                r for r in tables.get("cargo_marks_numbers", ()) if r["document_id"] == document_id
            ],
            "hs_code": [
                r for r in tables.get("cargo_hs_codes", ()) if r["document_id"] == document_id
            ],
            "handling_instruction": [
                r
                for r in tables.get("cargo_handling_instructions", ())
                if r["document_id"] == document_id
            ],
        }
        dangerous_rows = [
            r for r in tables.get("dangerous_goods", ()) if r["document_id"] == document_id
        ]
        groups = []
        for group_row in _ordered(
            [r for r in tables.get("cargo_groups", ()) if r["document_id"] == document_id],
            "group_order",
        ):
            group: dict[str, Any] = {"groupId": group_row["group_id"]}
            if group_row.get("description") is not None:
                group["description"] = group_row["description"]
            for target_name, prefix in (
                ("grossWeight", "gross_weight"),
                ("netWeight", "net_weight"),
                ("volume", "volume"),
            ):
                measure = _measure_from_row(group_row, prefix)
                if measure is not None:
                    group[target_name] = measure
            origin = {
                name: group_row.get(column)
                for name, column in (("name", "origin_name"), ("identifier", "origin_identifier"))
                if group_row.get(column) is not None
            }
            if origin:
                group["origin"] = origin
            for target_name, value_type in _CARGO_VALUE_FIELDS:
                values = _ordered(
                    [
                        r
                        for r in value_rows[value_type]
                        if r["cargo_group_row_id"] == group_row["cargo_group_row_id"]
                    ],
                    "value_order",
                )
                if values:
                    group[target_name] = [r["value"] for r in values]
            dangerous_values = []
            for dangerous_row in _ordered(
                [
                    r
                    for r in dangerous_rows
                    if r["cargo_group_row_id"] == group_row["cargo_group_row_id"]
                ],
                "dangerous_goods_order",
            ):
                dangerous = {
                    target_name: dangerous_row[column]
                    for target_name, column in (
                        ("unNumber", "un_number"),
                        ("hazardCategory", "hazard_category"),
                        ("subsidiaryHazardCategory", "subsidiary_hazard_category"),
                    )
                    if dangerous_row.get(column) is not None
                }
                flash_temperature = _measure_from_row(dangerous_row, "flash_point")
                packing = dangerous_row.get("packing_group_category")
                if flash_temperature is not None or packing is not None:
                    flash: dict[str, Any] = {}
                    if flash_temperature is not None:
                        flash["temperature"] = flash_temperature
                    if packing is not None:
                        flash["packingGroupCategory"] = packing
                    dangerous["flashPoint"] = flash
                dangerous_values.append(dangerous)
            if dangerous_values:
                group["dangerousGoods"] = dangerous_values
            groups.append(group)
        if groups:
            patch["cargoGroups"] = groups

        packages = []
        for package_row in _ordered(
            [r for r in tables.get("packages", ()) if r["document_id"] == document_id],
            "package_order",
        ):
            package = {
                "packageId": package_row["package_id"],
                "groupId": package_row["group_id"],
            }
            for target_name, column in (
                ("quantity", "quantity"),
                ("typeCategory", "type_category"),
                ("typeDescription", "type_description"),
            ):
                if package_row.get(column) is not None:
                    package[target_name] = package_row[column]
            packages.append(package)
        if packages:
            patch["cargoPackages"] = packages

        membership_rows = [
            r
            for r in tables.get("allocation_group_packages", ())
            if r["document_id"] == document_id
        ]
        allocation_rows = [
            r for r in tables.get("allocations", ()) if r["document_id"] == document_id
        ]
        allocation_groups = []
        for allocation_group_row in _ordered(
            [r for r in tables.get("allocation_groups", ()) if r["document_id"] == document_id],
            "allocation_group_order",
        ):
            memberships = _ordered(
                [
                    r
                    for r in membership_rows
                    if r["allocation_group_id"] == allocation_group_row["allocation_group_id"]
                ],
                "package_order",
            )
            allocations = []
            for allocation_row in _ordered(
                [
                    r
                    for r in allocation_rows
                    if r["allocation_group_id"] == allocation_group_row["allocation_group_id"]
                ],
                "allocation_order",
            ):
                allocation = {"containerNumber": allocation_row["container_number"]}
                if allocation_row.get("package_quantity") is not None:
                    allocation["packageQuantity"] = allocation_row["package_quantity"]
                if allocation_row.get("package_id") is not None:
                    allocation["packageId"] = allocation_row["package_id"]
                allocations.append(allocation)
            allocation_groups.append(
                {
                    "groupId": allocation_group_row["group_id"],
                    "coverage": allocation_group_row["coverage"],
                    "packageIds": [r["package_id"] for r in memberships],
                    "allocations": allocations,
                }
            )
        if allocation_groups:
            patch["cargoAllocationGroups"] = allocation_groups
        return {"schemaVersion": row["schema_version"], "documentPatch": patch}

    def sdv_metadata(self) -> dict[str, Any]:
        """Return a fixed Metadata V1 graph; no data-dependent type guessing."""

        primary_keys = {
            "documents": "document_id",
            "document_locations": "location_id",
            "document_references": "reference_id",
            "parties": "party_id",
            "party_contacts": "party_contact_id",
            "containers": "container_id",
            "container_seals": "container_seal_id",
            "cargo_groups": "cargo_group_row_id",
            "cargo_additional_information": "cargo_group_value_id",
            "cargo_marks_numbers": "cargo_group_value_id",
            "cargo_hs_codes": "cargo_group_value_id",
            "cargo_handling_instructions": "cargo_group_value_id",
            "packages": "package_row_id",
            "allocation_groups": "allocation_group_id",
            "allocation_group_packages": "allocation_group_package_id",
            "allocations": "allocation_id",
            "dangerous_goods": "dangerous_goods_id",
        }
        relationships = [
            _relationship("documents", "document_id", table, "document_id")
            for table in (
                "document_locations",
                "document_references",
                "parties",
                "containers",
                "cargo_groups",
            )
        ]
        relationships.extend(
            (
                _relationship("parties", "party_id", "party_contacts", "party_id"),
                _relationship("containers", "container_id", "container_seals", "container_id"),
                *(
                    _relationship("cargo_groups", "cargo_group_row_id", table, "cargo_group_row_id")
                    for table in (
                        "cargo_additional_information",
                        "cargo_marks_numbers",
                        "cargo_hs_codes",
                        "cargo_handling_instructions",
                        "packages",
                        "allocation_groups",
                        "dangerous_goods",
                    )
                ),
                _relationship(
                    "allocation_groups",
                    "allocation_group_id",
                    "allocation_group_packages",
                    "allocation_group_id",
                ),
                _relationship(
                    "allocation_groups",
                    "allocation_group_id",
                    "allocations",
                    "allocation_group_id",
                ),
            )
        )
        return {
            "METADATA_SPEC_VERSION": "V1",
            "tables": {
                table: {
                    "primary_key": primary_keys[table],
                    "columns": {column: _sdv_column(column) for column in _TABLE_COLUMNS[table]},
                }
                for table in self.table_order
            },
            "relationships": relationships,
        }


def _relationship(parent: str, primary_key: str, child: str, foreign_key: str) -> dict[str, str]:
    return {
        "parent_table_name": parent,
        "parent_primary_key": primary_key,
        "child_table_name": child,
        "child_foreign_key": foreign_key,
    }


_INTEGER_COLUMNS = {
    "source_row_index",
    "location_order",
    "reference_order",
    "role_order",
    "occurrence_order",
    "contact_order",
    "container_order",
    "seal_order",
    "group_order",
    "value_order",
    "dangerous_goods_order",
    "package_order",
    "allocation_group_order",
    "allocation_order",
    "quantity",
    "package_quantity",
}
_FLOAT_COLUMNS = {
    "verified_gross_mass_value",
    "temperature_setpoint_value",
    "gross_weight_value",
    "net_weight_value",
    "volume_value",
    "flash_point_value",
}
_DATE_COLUMNS = {"issue_date", "shipped_on_board_date"}


def _sdv_column(name: str) -> dict[str, Any]:
    if name.endswith("_id") or name == "document_id":
        return {"sdtype": "id"}
    if name in _INTEGER_COLUMNS:
        return {"sdtype": "numerical", "computer_representation": "Int64"}
    if name in _FLOAT_COLUMNS:
        return {"sdtype": "numerical", "computer_representation": "Float"}
    if name in _DATE_COLUMNS:
        return {"sdtype": "datetime", "datetime_format": "%Y-%m-%d"}
    return {"sdtype": "categorical"}


_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "documents": (
        "document_id",
        "source_row_index",
        "schema_version",
        "bill_of_lading_number",
        "original_bill_of_lading_number",
        "master_bill_of_lading_number",
        "issue_date",
        "shipped_on_board_date",
        "negotiability",
        "transport_vessel_name",
        "transport_vessel_imo_number",
        "transport_voyage_number",
        "transport_vessel_flag_country",
        "freight_payment_arrangement",
    ),
    "document_locations": (
        "location_id",
        "document_id",
        "location_order",
        "location_role",
        "name",
        "country",
    ),
    "document_references": (
        "reference_id",
        "document_id",
        "reference_order",
        "value",
    ),
    "parties": (
        "party_id",
        "document_id",
        "role",
        "role_order",
        "occurrence_order",
        "same_as",
        "name",
        "address",
        "city",
        "country",
        "contact_name",
    ),
    "party_contacts": (
        "party_contact_id",
        "document_id",
        "party_id",
        "contact_type",
        "contact_order",
        "value",
    ),
    "containers": (
        "container_id",
        "document_id",
        "container_order",
        "container_number",
        "type_description",
        "type_category",
        "verified_gross_mass_value",
        "verified_gross_mass_unit",
        "temperature_setpoint_value",
        "temperature_setpoint_unit",
    ),
    "container_seals": (
        "container_seal_id",
        "document_id",
        "container_id",
        "seal_order",
        "value",
    ),
    "cargo_groups": (
        "cargo_group_row_id",
        "document_id",
        "group_order",
        "group_id",
        "description",
        "gross_weight_value",
        "gross_weight_unit",
        "net_weight_value",
        "net_weight_unit",
        "volume_value",
        "volume_unit",
        "origin_name",
        "origin_identifier",
    ),
    "cargo_additional_information": (
        "cargo_group_value_id",
        "document_id",
        "cargo_group_row_id",
        "value_order",
        "value",
    ),
    "cargo_marks_numbers": (
        "cargo_group_value_id",
        "document_id",
        "cargo_group_row_id",
        "value_order",
        "value",
    ),
    "cargo_hs_codes": (
        "cargo_group_value_id",
        "document_id",
        "cargo_group_row_id",
        "value_order",
        "value",
    ),
    "cargo_handling_instructions": (
        "cargo_group_value_id",
        "document_id",
        "cargo_group_row_id",
        "value_order",
        "value",
    ),
    "packages": (
        "package_row_id",
        "document_id",
        "package_order",
        "package_id",
        "group_id",
        "cargo_group_row_id",
        "quantity",
        "type_category",
        "type_description",
    ),
    "allocation_groups": (
        "allocation_group_id",
        "document_id",
        "allocation_group_order",
        "group_id",
        "cargo_group_row_id",
        "coverage",
    ),
    "allocation_group_packages": (
        "allocation_group_package_id",
        "document_id",
        "allocation_group_id",
        "package_order",
        "package_id",
        "package_row_id",
    ),
    "allocations": (
        "allocation_id",
        "document_id",
        "allocation_group_id",
        "allocation_order",
        "container_number",
        "container_id",
        "package_quantity",
        "package_id",
        "package_row_id",
    ),
    "dangerous_goods": (
        "dangerous_goods_id",
        "document_id",
        "cargo_group_row_id",
        "dangerous_goods_order",
        "un_number",
        "hazard_category",
        "subsidiary_hazard_category",
        "flash_point_value",
        "flash_point_unit",
        "packing_group_category",
    ),
}


ADAPTER = BillOfLadingRelationDomainAdapter()
