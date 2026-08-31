from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from document_ocr.synthesis.scenario_views import (
    PARTY_STRUCTURE_COLUMNS,
    PARTY_STRUCTURE_PROJECTED_COLUMNS,
    ROUTE_FREIGHT_DOCUMENT_COLUMNS,
    CountryResolutionError,
    ScenarioViewError,
    build_scenario_views,
    project_party_structure_driver,
)


def _document(document_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "document_id": document_id,
        "bill_of_lading_number": None,
        "original_bill_of_lading_number": None,
        "master_bill_of_lading_number": None,
        "issue_date": None,
        "shipped_on_board_date": None,
        "negotiability": None,
        "transport_vessel_name": None,
        "transport_vessel_imo_number": None,
        "transport_voyage_number": None,
        "transport_vessel_flag_country": None,
        "freight_payment_arrangement": None,
    }
    row.update(overrides)
    return row


def _row(row_id: str, document_id: str, key: str, **values: object) -> dict[str, object]:
    return {key: row_id, "document_id": document_id, **values}


def _tables() -> dict[str, list[dict[str, object]]]:
    return {
        "documents": [
            _document(
                "d1",
                bill_of_lading_number="SECRET-BL-1",
                issue_date="2025-01-03",
                shipped_on_board_date="2025-01-01",
                negotiability="non_negotiable",
                transport_vessel_name="PRIVATE VESSEL",
                transport_voyage_number="V001",
                freight_payment_arrangement="prepaid",
            ),
            _document(
                "d2",
                master_bill_of_lading_number="SECRET-MASTER-2",
                negotiability="negotiable",
                transport_vessel_imo_number="1234567",
                transport_vessel_flag_country="GERMANY",
                freight_payment_arrangement="collect",
            ),
            _document("d3"),
        ],
        "document_locations": [
            _row(
                "d1:pol",
                "d1",
                "location_id",
                location_role="portOfLoading",
                name="SHANGHAI",
                country="CHINA",
            ),
            _row(
                "d1:pod",
                "d1",
                "location_id",
                location_role="portOfDischarge",
                name="ALEXANDRIA",
                country="EGYPT",
            ),
            _row(
                "d1:pay",
                "d1",
                "location_id",
                location_role="paymentPlace",
                name="SHANGHAI",
                country="CHINA",
            ),
            _row(
                "d2:pol",
                "d2",
                "location_id",
                location_role="portOfLoading",
                name="MERSIN",
                country="TURKEY",
            ),
            _row(
                "d2:pod",
                "d2",
                "location_id",
                location_role="portOfDischarge",
                name="HAMBURG",
                country="GERMANY",
            ),
            _row(
                "d2:final",
                "d2",
                "location_id",
                location_role="finalDestination",
                name="BERLIN",
                country="GERMANY",
            ),
            _row(
                "d2:pay",
                "d2",
                "location_id",
                location_role="paymentPlace",
                name="ZURICH",
                country="SWITZERLAND",
            ),
            _row(
                "d3:pol",
                "d3",
                "location_id",
                location_role="portOfLoading",
                name="MUMBAI",
                country="INDIA",
            ),
        ],
        "parties": [
            _row(
                "p1",
                "d1",
                "party_id",
                role="shipper",
                same_as=None,
                name="ALPHA SECRET EXPORTS",
                address="1 Export Road",
                city="SHANGHAI",
                country="CHINA",
                contact_name="PRIVATE ALICE",
            ),
            _row(
                "p2",
                "d1",
                "party_id",
                role="consignee",
                same_as=None,
                name="BETA SECRET IMPORTS",
                address="2 Import Road",
                city="CAIRO",
                country="EGYPT",
                contact_name=None,
            ),
            _row(
                "p3",
                "d1",
                "party_id",
                role="notifyParties",
                same_as="consignee",
                name=None,
                address=None,
                city=None,
                country=None,
                contact_name=None,
            ),
            _row(
                "p4",
                "d1",
                "party_id",
                role="carrier",
                same_as=None,
                name="GAMMA SECRET CARRIER",
                address=None,
                city="GENEVA",
                country="SWITZERLAND",
                contact_name=None,
            ),
            _row(
                "p5",
                "d2",
                "party_id",
                role="shipper",
                same_as=None,
                name="DELTA SECRET EXPORTS",
                address="3 Ankara Street",
                city="ANKARA",
                country="TURKEY",
                contact_name=None,
            ),
            _row(
                "p6",
                "d2",
                "party_id",
                role="deliveryAgent",
                same_as=None,
                name="EPSILON SECRET DELIVERY",
                address="4 Hamburg Street",
                city="HAMBURG",
                country="GERMANY",
                contact_name=None,
            ),
            _row(
                "p7",
                "d3",
                "party_id",
                role="shipper",
                same_as=None,
                name="VALIDATION SECRET",
                address="Validation Address",
                city="MUMBAI",
                country="INDIA",
                contact_name=None,
            ),
        ],
        "party_contacts": [
            _row("c1", "d1", "party_contact_id", party_id="p1", contact_type="phone", value="1"),
            _row("c2", "d1", "party_contact_id", party_id="p1", contact_type="phone", value="2"),
            _row(
                "c3",
                "d1",
                "party_contact_id",
                party_id="p1",
                contact_type="email",
                value="private@example.test",
            ),
            _row(
                "c4",
                "d1",
                "party_contact_id",
                party_id="p2",
                contact_type="website",
                value="private.example.test",
            ),
            _row("c5", "d2", "party_contact_id", party_id="p6", contact_type="phone", value="3"),
        ],
        "containers": [
            _row("ct1", "d1", "container_id"),
            _row("ct2", "d1", "container_id"),
            _row("ct3", "d2", "container_id"),
        ],
        "cargo_groups": [
            _row("cg1", "d1", "cargo_group_row_id"),
            _row("cg2", "d2", "cargo_group_row_id"),
        ],
        "cargo_hs_codes": [
            _row("hs1", "d1", "cargo_group_value_id"),
            _row("hs2", "d1", "cargo_group_value_id"),
            _row("hs3", "d2", "cargo_group_value_id"),
        ],
        "packages": [
            _row("pk1", "d1", "package_row_id"),
            _row("pk2", "d1", "package_row_id"),
            _row("pk3", "d2", "package_row_id"),
        ],
        "allocation_groups": [
            _row("ag1", "d1", "allocation_group_id"),
            _row("ag2", "d2", "allocation_group_id"),
        ],
        "allocations": [
            _row("a1", "d1", "allocation_id"),
            _row("a2", "d1", "allocation_id"),
            _row("a3", "d2", "allocation_id"),
        ],
        "dangerous_goods": [
            _row("dg1", "d1", "dangerous_goods_id"),
        ],
    }


def _resolver() -> dict[str, str]:
    return {
        "CHINA": "CN",
        "EGYPT": "EG",
        "TURKEY": "TR",
        "GERMANY": "DE",
        "SWITZERLAND": "CH",
    }


def _build(
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    templates: Mapping[str, str] | None = None,
    resolver: Mapping[str, str] | None = None,
):
    return build_scenario_views(
        tables=tables or _tables(),
        fit_document_ids=("d1", "d2"),
        partition_by_document={"d1": "train", "d2": "train", "d3": "validation"},
        template_by_document=templates or {"d1": "t1", "d2": "t2", "d3": "t3"},
        country_resolver=resolver or _resolver(),
    )


def test_views_expose_no_raw_party_pii_and_preserve_lineage_outside_data() -> None:
    result = _build()
    party = result.party_structure
    route = result.route_freight_document

    assert tuple(party.data.columns) == PARTY_STRUCTURE_COLUMNS
    assert tuple(route.data.columns) == ROUTE_FREIGHT_DOCUMENT_COLUMNS
    assert party.row_ids == ("p1", "p2", "p3", "p4", "p5", "p6")
    assert party.group_ids == ("t1", "t1", "t1", "t1", "t2", "t2")
    assert route.row_ids == ("d1", "d2")
    assert route.group_ids == ("t1", "t2")
    serialized = party.data.to_csv(index=False).upper()
    for private_value in (
        "ALPHA",
        "BETA",
        "GAMMA",
        "DELTA",
        "EPSILON",
        "EXPORT ROAD",
        "PRIVATE ALICE",
        "SHANGHAI",
        "CAIRO",
        "GENEVA",
        "CHINA",
        "EGYPT",
        "SWITZERLAND",
        "PRIVATE@EXAMPLE.TEST",
    ):
        assert private_value not in serialized
    assert result.audit.raw_pii_columns_exposed is False


def test_party_relations_city_modes_contacts_and_address_counts_are_exact() -> None:
    result = _build()
    rows = {
        row_id: result.party_structure.data.iloc[index].to_dict()
        for index, row_id in enumerate(result.party_structure.row_ids)
    }
    projected_rows = {row_id: project_party_structure_driver(row) for row_id, row in rows.items()}

    assert (rows["p1"]["relation_to_route"], rows["p1"]["city_mode"]) == (
        "origin",
        "endpoint_port",
    )
    assert projected_rows["p1"]["phone_count"] == 2
    assert projected_rows["p1"]["email_count"] == 1
    assert projected_rows["p1"]["website_count"] == 0
    assert rows["p1"]["address_character_count"] == len("1 Export Road")
    assert rows["p1"]["address_word_count"] == 3
    assert rows["p1"]["geography_presence"] == "city_and_country"
    assert (rows["p2"]["relation_to_route"], rows["p2"]["city_mode"]) == (
        "destination",
        "other_same_country",
    )
    assert projected_rows["p2"]["website_count"] == 1
    assert rows["p3"]["relation_to_route"] == "same_as"
    assert rows["p3"]["city_mode"] == "missing"
    assert rows["p3"]["geography_presence"] == "neither"
    assert rows["p4"]["relation_to_route"] == "third_country"
    assert rows["p4"]["city_mode"] == "third_country"
    assert rows["p5"]["relation_to_route"] == "origin"
    assert rows["p5"]["city_mode"] == "other_same_country"
    assert rows["p6"]["relation_to_route"] == "destination"
    assert rows["p6"]["city_mode"] == "endpoint_port"

    projected = project_party_structure_driver(rows["p1"])
    assert tuple(projected) == PARTY_STRUCTURE_PROJECTED_COLUMNS
    assert projected["address_present"] is True
    assert projected["city_present"] is True
    assert projected["country_present"] is True
    assert projected["phone_present"] is True
    assert projected["email_present"] is True
    assert projected["website_present"] is False


def test_endpoint_city_matching_requires_the_same_country() -> None:
    tables = _tables()
    carrier = next(row for row in tables["parties"] if row["party_id"] == "p4")
    carrier["city"] = "SHANGHAI"

    result = _build(tables=tables)
    row_index = result.party_structure.row_ids.index("p4")
    carrier_driver = result.party_structure.data.iloc[row_index].to_dict()

    assert carrier_driver["relation_to_route"] == "third_country"
    assert carrier_driver["city_mode"] == "third_country"


def test_route_freight_topology_and_structural_counts_are_exact() -> None:
    result = _build()
    rows = {
        row_id: result.route_freight_document.data.iloc[index].to_dict()
        for index, row_id in enumerate(result.route_freight_document.row_ids)
    }

    d1 = rows["d1"]
    assert (d1["origin_country_code"], d1["destination_country_code"]) == ("CN", "EG")
    assert d1["freight_payment_arrangement"] == "prepaid"
    assert d1["freight_payment_side"] == "origin"
    assert d1["party_count"] == 4
    assert d1["notify_party_count"] == 1
    assert d1["party_contact_count"] == 4
    assert d1["container_count"] == 2
    assert d1["cargo_group_count"] == 1
    assert d1["package_count"] == 2
    assert d1["allocation_group_count"] == 1
    assert d1["allocation_count"] == 2
    assert d1["dangerous_goods_count"] == 1
    assert d1["hs_code_count"] == 2
    assert d1["has_bill_of_lading_number"]
    assert d1["has_vessel_name"]
    assert d1["has_port_of_loading"]
    assert d1["has_port_of_discharge"]
    assert not d1["has_final_destination"]

    d2 = rows["d2"]
    assert (d2["origin_country_code"], d2["destination_country_code"]) == ("TR", "DE")
    assert d2["freight_payment_side"] == "third_country"
    assert d2["has_final_destination"]
    assert d2["has_master_bill_of_lading_number"]
    assert d2["has_vessel_imo_number"]
    assert d2["has_vessel_flag_country"]
    assert result.audit.fit_document_count == 2
    assert result.audit.fit_template_count == 2
    assert result.audit.party_structure_rows == 6
    assert result.audit.party_structure_inverse_projection_rows == 6
    assert result.audit.route_freight_document_rows == 2
    assert result.audit.country_cells_present == 12
    assert result.audit.country_cells_resolved == 12
    assert result.audit.country_cells_missing == 1


def test_fit_template_crossing_validation_partition_is_rejected() -> None:
    with pytest.raises(ScenarioViewError, match="fit template crosses"):
        _build(templates={"d1": "t1", "d2": "crossed", "d3": "crossed"})


def test_unresolved_countries_fail_with_complete_audit_locations() -> None:
    resolver = _resolver()
    del resolver["SWITZERLAND"]

    with pytest.raises(CountryResolutionError) as raised:
        _build(resolver=resolver)

    assert [
        (issue.document_id, issue.table, issue.row_id, issue.value) for issue in raised.value.issues
    ] == [
        ("d1", "parties", "p4", "SWITZERLAND"),
        ("d2", "document_locations", "d2:pay", "SWITZERLAND"),
    ]


def test_resolver_objects_and_callables_preserve_the_exhaustive_contract() -> None:
    mapping = _resolver()

    class Resolver:
        @staticmethod
        def resolve(value: str | None) -> str | None:
            return mapping.get(value) if value is not None else None

    object_result = _build(resolver=Resolver())
    callable_result = _build(
        resolver=lambda value: mapping.get(value) if value is not None else None
    )

    assert object_result.party_structure.data.equals(callable_result.party_structure.data)
    assert object_result.route_freight_document.data.equals(
        callable_result.route_freight_document.data
    )


def test_resolver_callable_rejects_non_iso_output_with_source_location() -> None:
    mapping = _resolver()
    mapping["CHINA"] = "CHN"

    with pytest.raises(
        ScenarioViewError,
        match=r"document_locations:d1:pol:country: 'CHN'",
    ):
        _build(resolver=mapping.get)
