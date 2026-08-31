from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.modeling_views import (
    ModelingViews,
    build_modeling_views,
    modeling_views_sha256,
)
from document_ocr.synthesis.package_registry import (
    LoadedPackageRegistry,
    load_package_registry,
)
from document_ocr.training.config import RuntimeDatasetPartitionConfig
from document_ocr.training.splitting import (
    PartitionCandidate,
    select_runtime_partition,
    target_leaf_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2"
)
HIERARCHY = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-packages-v2/package-metadata.jsonl"
)
PREPARATION = (
    PROJECT_ROOT / "artifacts/kie-synthesis/mpci-bl-combined1157-synthesis-preparation-v3/tables"
)
REGISTRY = (
    PROJECT_ROOT / "artifacts/mpci-ai-schema/categorical-registry-followup/"
    "package-category-registry.json"
)
REGISTRY_SHA256 = "2476deb46c3cabf2efe9a00bbb9ad1bf6b527e4adae0e4872d8a63807bbd374f"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def source_records() -> list[dict[str, Any]]:
    return _jsonl(DATASET / "records.jsonl")


@pytest.fixture(scope="module")
def tables() -> dict[str, list[dict[str, Any]]]:
    return {name: _jsonl(PREPARATION / f"{name}.jsonl") for name in ADAPTER.table_order}


@pytest.fixture(scope="module")
def category_metadata() -> list[dict[str, Any]]:
    return _jsonl(DATASET / "category-metadata.jsonl")


@pytest.fixture(scope="module")
def hierarchy_metadata() -> list[dict[str, Any]]:
    return _jsonl(HIERARCHY)


@pytest.fixture(scope="module")
def registry() -> LoadedPackageRegistry:
    return load_package_registry(
        REGISTRY,
        expected_sha256=REGISTRY_SHA256,
        expected_entries=405,
    )


@pytest.fixture(scope="module")
def partition(
    source_records: list[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    config = RuntimeDatasetPartitionConfig.model_validate(
        {
            "algorithm": "seeded_sha256_rank_v1",
            "seed": 424,
            "validation_size": {"kind": "records", "value": 100},
            "coverage_policy": "retain_each_target_leaf_in_train",
        },
        strict=True,
    )
    selection = select_runtime_partition(
        [
            PartitionCandidate(
                document_id=row["documentId"],
                input_sha256=row["joinedRawTextSha256"],
                target_leaf_paths=frozenset(target_leaf_paths(row["target"])),
            )
            for row in source_records
        ],
        config,
    )
    return selection.train_document_ids, selection.validation_document_ids


@pytest.fixture(scope="module")
def train_views(
    tables: dict[str, list[dict[str, Any]]],
    category_metadata: list[dict[str, Any]],
    hierarchy_metadata: list[dict[str, Any]],
    registry: LoadedPackageRegistry,
    partition: tuple[tuple[str, ...], tuple[str, ...]],
) -> ModelingViews:
    train_ids, _ = partition
    return build_modeling_views(
        tables=tables,
        train_document_ids=train_ids,
        package_registry=registry,
        category_metadata_rows=category_metadata,
        package_hierarchy_metadata_rows=hierarchy_metadata,
    )


@pytest.fixture(scope="module")
def full_views(
    tables: dict[str, list[dict[str, Any]]],
    category_metadata: list[dict[str, Any]],
    hierarchy_metadata: list[dict[str, Any]],
    registry: LoadedPackageRegistry,
    source_records: list[dict[str, Any]],
) -> ModelingViews:
    started = time.perf_counter()
    views = build_modeling_views(
        tables=tables,
        train_document_ids=tuple(row["documentId"] for row in source_records),
        package_registry=registry,
        category_metadata_rows=category_metadata,
        package_hierarchy_metadata_rows=hierarchy_metadata,
    )
    assert time.perf_counter() - started < 15.0
    return views


def test_package_registry_and_train_surface_inventory_are_pinned(
    registry: LoadedPackageRegistry,
    train_views: ModelingViews,
) -> None:
    assert registry.sha256 == REGISTRY_SHA256
    assert len(registry.payload.entries) == 405
    assert len(registry.category_tokens) == 405
    assert train_views.package_surface_inventory.model_dump(exclude={"buckets"}) == {
        "train_document_count": 1057,
        "decision_count": 1285,
        "typed_decision_count": 1234,
        "fallback_decision_count": 51,
    }
    assert len(train_views.package_surface_inventory.buckets) == 148
    assert {
        bucket.category_token
        for bucket in train_views.package_surface_inventory.buckets
        if bucket.category_token is not None
    } <= registry.category_tokens


def test_actual_partition_views_are_strictly_train_only(
    train_views: ModelingViews,
    partition: tuple[tuple[str, ...], tuple[str, ...]],
) -> None:
    train_ids, validation_ids = partition
    assert train_views.train_document_ids == train_ids
    assert set(train_ids).isdisjoint(validation_ids)
    assert train_views.audit.model_dump() == {
        "train_document_count": 1057,
        "validated_target_count": 1057,
        "document_scenario_rows": 1057,
        "cargo_package_numeric_rows": 1302,
        "cargo_group_numeric_rows": 1241,
        "container_equipment_rows": 1927,
        "package_hierarchy_rows": 1476,
        "metadata_only_package_rows": 174,
        "multi_level_document_count": 59,
        "coverage_counts": {
            "one_to_one_package_allocations": 312,
            "single_package_level": 422,
            "all_package_levels_combined": 6,
            "unlinked_package_quantities": 21,
            "container_membership_only": 171,
        },
        "direct_source_cell_count": 30600,
        "package_registry_entries": 405,
        "package_surface_decisions": 1285,
    }
    validation = set(validation_ids)
    assert not validation & {
        row.projection.source_document_id for row in train_views.document_scenario
    }
    assert not validation & {
        row.projection.source_document_id for row in train_views.cargo_package_numeric
    }
    assert not validation & {
        row.projection.source_document_id for row in train_views.cargo_group_numeric
    }
    assert not validation & {
        row.projection.source_document_id for row in train_views.container_equipment
    }
    assert not validation & {row.source_document_id for row in train_views.package_hierarchy}


def test_model_rows_exclude_identifiers_and_generic_free_text(train_views: ModelingViews) -> None:
    expected_columns = {
        "document_scenario": set(type(train_views.document_scenario[0].features).model_fields),
        "cargo_package_numeric": set(
            type(train_views.cargo_package_numeric[0].features).model_fields
        ),
        "cargo_group_numeric": set(type(train_views.cargo_group_numeric[0].features).model_fields),
        "container_equipment": set(type(train_views.container_equipment[0].features).model_fields),
    }
    forbidden = {
        "document_id",
        "container_number",
        "package_id",
        "group_id",
        "description",
        "name",
        "address",
        "marks_and_numbers",
        "reference",
        "contact",
    }
    for view_name, columns in expected_columns.items():
        rows = train_views.model_rows(view_name)  # type: ignore[arg-type]
        assert rows
        assert set(rows[0]) == columns
        assert not forbidden & columns
        assert all(set(row) == columns for row in rows)


def test_full_corpus_preserves_every_package_level_and_allocation_class(
    full_views: ModelingViews,
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    assert full_views.audit.model_dump() == {
        "train_document_count": 1157,
        "validated_target_count": 1157,
        "document_scenario_rows": 1157,
        "cargo_package_numeric_rows": 1423,
        "cargo_group_numeric_rows": 1360,
        "container_equipment_rows": 2115,
        "package_hierarchy_rows": 1612,
        "metadata_only_package_rows": 189,
        "multi_level_document_count": 66,
        "coverage_counts": {
            "one_to_one_package_allocations": 343,
            "single_package_level": 458,
            "all_package_levels_combined": 7,
            "unlinked_package_quantities": 26,
            "container_membership_only": 189,
        },
        "direct_source_cell_count": 33535,
        "package_registry_entries": 405,
        "package_surface_decisions": 1405,
    }
    packages_by_document_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for package in tables["packages"]:
        packages_by_document_group.setdefault(
            (package["document_id"], package["group_id"]), []
        ).append(package)
    multi_level_documents = {
        document_id
        for (document_id, _), rows in packages_by_document_group.items()
        if len(rows) > 1
    }
    assert len(multi_level_documents) == 66
    observed_multi_level_documents = {
        row.projection.source_document_id
        for row in full_views.cargo_package_numeric
        if row.features.task_package_level_count > 1
    }
    assert observed_multi_level_documents == multi_level_documents

    for (document_id, _), packages in packages_by_document_group.items():
        if len(packages) <= 1:
            continue
        row_keys = {package["package_row_id"] for package in packages}
        observations = [
            row
            for row in full_views.cargo_package_numeric
            if row.projection.source_document_id == document_id
            and row.projection.view_row_key.removeprefix("cargo-package:") in row_keys
        ]
        assert [row.features.task_package_position for row in observations] == list(
            range(len(packages))
        )
        assert all(row.features.task_package_level_count == len(packages) for row in observations)


def test_cargo_group_view_is_unique_and_uses_the_exact_driver_priority(
    full_views: ModelingViews,
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    observations = {
        (row.projection.source_document_id, row.cargo_group_id): row
        for row in full_views.cargo_group_numeric
    }
    source_groups = {(row["document_id"], row["group_id"]) for row in tables["cargo_groups"]}
    assert set(observations) == source_groups
    assert len(observations) == len(tables["cargo_groups"]) == 1360

    packages_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for package in sorted(tables["packages"], key=lambda row: row["package_order"]):
        packages_by_group[(package["document_id"], package["group_id"])].append(package)
    hierarchy = {
        (row.source_document_id, row.group_id, row.projected_package_id): row
        for row in full_views.package_hierarchy
        if row.projected_package_id is not None
    }
    role_priority = {
        "direct_goods": 0,
        "generic_aggregate": 1,
        "outer_transport": 2,
        "unknown": 3,
    }
    expected_without_driver = 0
    for group_key, observation in observations.items():
        packages = packages_by_group.get(group_key, [])
        quantified = [
            (position, package, hierarchy[(*group_key, package["package_id"])])
            for position, package in enumerate(packages)
            if package.get("quantity") is not None
        ]
        if not quantified:
            expected_without_driver += 1
            assert observation.driver_package_id is None
            assert observation.features.driver_quantity is None
            continue
        expected_position, expected_package, expected_hierarchy = min(
            quantified,
            key=lambda row: (
                role_priority[row[2].role],
                row[0],
                row[1]["package_id"],
            ),
        )
        assert observation.driver_package_id == expected_package["package_id"]
        assert observation.features.driver_task_package_position == expected_position
        assert observation.features.driver_source_package_position == (
            expected_hierarchy.source_package_position
        )
        assert observation.features.driver_package_role == expected_hierarchy.role
        assert observation.features.driver_quantity == expected_package["quantity"]
    assert expected_without_driver == sum(
        row.driver_package_id is None for row in full_views.cargo_group_numeric
    )


def test_cargo_group_mass_projection_is_decimal_exact_and_invertible(
    full_views: ModelingViews,
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    source_groups = {row["cargo_group_row_id"]: row for row in tables["cargo_groups"]}
    factors = {
        "kilogram": Decimal("1"),
        "metric_tonne": Decimal("1000"),
        "pound": Decimal("0.45359237"),
    }
    observed_units: set[str] = set()
    for observation in full_views.cargo_group_numeric:
        group_row_key = observation.projection.view_row_key.removeprefix("cargo-group:")
        source = source_groups[group_row_key]
        projections = {row.feature: row for row in observation.mass_projections}
        for prefix, feature in (
            ("gross_weight", "gross_weight_kg"),
            ("net_weight", "net_weight_kg"),
        ):
            projection = projections[feature]
            source_value = source[f"{prefix}_value"]
            source_unit = source[f"{prefix}_unit"]
            assert projection.source_value == source_value
            assert projection.source_unit == source_unit
            if source_value is None:
                assert projection.canonical_value_decimal is None
                assert getattr(observation.features, feature) is None
                continue
            observed_units.add(source_unit)
            source_decimal = Decimal(str(source_value))
            canonical = source_decimal * factors[source_unit]
            assert projection.source_value_decimal == format(source_decimal, "f")
            assert projection.source_decimal_places == max(0, -source_decimal.as_tuple().exponent)
            assert projection.multiplier_decimal == format(factors[source_unit], "f")
            assert projection.canonical_value_decimal == format(canonical, "f")
            assert Decimal(str(getattr(observation.features, feature))) == canonical
            assert Decimal(projection.canonical_value_decimal) / factors[source_unit] == (
                source_decimal
            )
    assert observed_units == {"kilogram", "metric_tonne", "pound"}

    inverted = {
        (row.source.table, row.source.row_key, row.source.column): row.value
        for row in full_views.inverse_direct_source_cells("cargo_group_numeric")
    }
    for row_key, source in source_groups.items():
        for column in (
            "gross_weight_value",
            "gross_weight_unit",
            "net_weight_value",
            "net_weight_unit",
        ):
            assert inverted[("cargo_groups", row_key, column)] == source[column]


def test_projection_metadata_is_exhaustive_and_direct_values_invert(
    full_views: ModelingViews,
    tables: dict[str, list[dict[str, Any]]],
) -> None:
    source_values = {
        (table, row[ADAPTER.sdv_metadata()["tables"][table]["primary_key"]], column): value
        for table, rows in tables.items()
        for row in rows
        for column, value in row.items()
    }
    for view_name in (
        "document_scenario",
        "cargo_package_numeric",
        "cargo_group_numeric",
        "container_equipment",
    ):
        observations = getattr(full_views, view_name)
        for observation in observations:
            fields = set(observation.features.model_dump())
            direct = {row.feature for row in observation.projection.direct}
            derived = {row.feature for row in observation.projection.derived}
            assert direct.isdisjoint(derived)
            assert direct | derived == fields
        for recovered in full_views.inverse_direct_source_cells(view_name):
            key = (
                recovered.source.table,
                recovered.source.row_key,
                recovered.source.column,
            )
            if recovered.source.table in tables:
                assert recovered.value == source_values[key]


def test_build_is_input_order_independent_for_a_real_subset(
    tables: dict[str, list[dict[str, Any]]],
    category_metadata: list[dict[str, Any]],
    hierarchy_metadata: list[dict[str, Any]],
    registry: LoadedPackageRegistry,
    full_views: ModelingViews,
) -> None:
    document_ids = tuple(
        row.projection.source_document_id
        for row in full_views.document_scenario
        if row.features.multi_level_cargo_group_count
    )[:10]
    first = build_modeling_views(
        tables=tables,
        train_document_ids=document_ids,
        package_registry=registry,
        category_metadata_rows=category_metadata,
        package_hierarchy_metadata_rows=hierarchy_metadata,
    )
    reversed_inputs = {name: list(reversed(rows)) for name, rows in tables.items()}
    second = build_modeling_views(
        tables=reversed_inputs,
        train_document_ids=document_ids,
        package_registry=registry,
        category_metadata_rows=list(reversed(category_metadata)),
        package_hierarchy_metadata_rows=list(reversed(hierarchy_metadata)),
    )
    assert modeling_views_sha256(first) == modeling_views_sha256(second)


def test_registry_and_hierarchy_corruption_fail_closed(
    tmp_path: Path,
    tables: dict[str, list[dict[str, Any]]],
    category_metadata: list[dict[str, Any]],
    hierarchy_metadata: list[dict[str, Any]],
    registry: LoadedPackageRegistry,
) -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_package_registry(
            REGISTRY,
            expected_sha256="0" * 64,
            expected_entries=405,
        )
    copied = tmp_path / "package-registry.json"
    copied.write_bytes(REGISTRY.read_bytes())
    with pytest.raises(ValueError, match="entry count mismatch"):
        load_package_registry(
            copied,
            expected_sha256=REGISTRY_SHA256,
            expected_entries=404,
        )
    duplicate_payload = json.loads(REGISTRY.read_text())
    duplicate_payload["entries"].append(deepcopy(duplicate_payload["entries"][0]))
    duplicate_bytes = json.dumps(duplicate_payload, separators=(",", ":")).encode()
    duplicate = tmp_path / "duplicate-package-registry.json"
    duplicate.write_bytes(duplicate_bytes)
    with pytest.raises(ValueError, match="categoryToken values must be unique"):
        load_package_registry(
            duplicate,
            expected_sha256=hashlib.sha256(duplicate_bytes).hexdigest(),
            expected_entries=406,
        )

    document_id = hierarchy_metadata[0]["documentId"]
    corrupted_hierarchy = deepcopy(hierarchy_metadata)
    corrupted_hierarchy[0]["groupDiagnoses"][0]["retainedPackageIds"] = []
    with pytest.raises(ValueError, match=r"does not partition|count differs"):
        build_modeling_views(
            tables=tables,
            train_document_ids=(document_id,),
            package_registry=registry,
            category_metadata_rows=category_metadata,
            package_hierarchy_metadata_rows=corrupted_hierarchy,
        )
