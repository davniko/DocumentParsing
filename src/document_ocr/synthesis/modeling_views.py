"""Lossless, train-scoped statistical views over the B/L relation graph.

The full 17-table projection remains the semantic authority.  These compact
views expose only bounded categorical, numeric, temporal, missingness and
cardinality features suitable for statistical proposal models.  High-cardinal
identifiers and free text live exclusively in projection metadata.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingRelationExplicitLabel
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.package_registry import (
    CategoryToken,
    LoadedPackageRegistry,
    PackageSurfaceInventory,
    build_package_surface_inventory,
    normalize_printed_surface,
)

NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
GroupId = Annotated[str, StringConstraints(pattern=r"^g[1-9][0-9]*$")]
PackageId = Annotated[str, StringConstraints(pattern=r"^p[1-9][0-9]*$")]
IsoDate = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Number = int | float
AllocationCoverage = Literal[
    "one_to_one_package_allocations",
    "single_package_level",
    "all_package_levels_combined",
    "unlinked_package_quantities",
    "container_membership_only",
]
ModeledAllocationCoverage = AllocationCoverage | Literal["none"]
PackageRole = Literal["direct_goods", "generic_aggregate", "outer_transport", "unknown"]
PackageDisposition = Literal["task_facing", "metadata_only"]
MassFeatureName = Literal["gross_weight_kg", "net_weight_kg"]
MassSourceUnit = Literal["kilogram", "metric_tonne", "pound"]

_MASS_TO_KILOGRAM = {
    "kilogram": Decimal("1"),
    "metric_tonne": Decimal("1000"),
    "pound": Decimal("0.45359237"),
}
_PACKAGE_ROLE_PRIORITY: Mapping[PackageRole, int] = {
    "direct_goods": 0,
    "generic_aggregate": 1,
    "outer_transport": 2,
    "unknown": 3,
}

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_COVERAGE_VALUES = (
    "one_to_one_package_allocations",
    "single_package_level",
    "all_package_levels_combined",
    "unlinked_package_quantities",
    "container_membership_only",
)
_ADAPTER_METADATA = ADAPTER.sdv_metadata()
_TABLE_PRIMARY_KEYS = {
    name: cast(str, _ADAPTER_METADATA["tables"][name]["primary_key"])
    for name in ADAPTER.table_order
}


class SourceCellRef(BaseModel):
    """A high-cardinality source coordinate kept outside model input."""

    model_config = _STRICT

    table: NonEmptyString
    row_key: NonEmptyString
    column: NonEmptyString


class DirectFeatureProjection(BaseModel):
    model_config = _STRICT

    feature: NonEmptyString
    source: SourceCellRef


class DerivedFeatureProjection(BaseModel):
    model_config = _STRICT

    feature: NonEmptyString
    method: NonEmptyString


class ViewRowProjection(BaseModel):
    """Exhaustive direct-or-derived classification for every model feature."""

    model_config = _STRICT

    view_row_key: NonEmptyString
    source_document_id: NonEmptyString
    direct: tuple[DirectFeatureProjection, ...]
    derived: tuple[DerivedFeatureProjection, ...]

    @model_validator(mode="after")
    def feature_names_are_unique(self) -> ViewRowProjection:
        direct = tuple(row.feature for row in self.direct)
        derived = tuple(row.feature for row in self.derived)
        if direct != tuple(sorted(set(direct))):
            raise ValueError("direct feature projections must be unique and sorted")
        if derived != tuple(sorted(set(derived))):
            raise ValueError("derived feature projections must be unique and sorted")
        if set(direct) & set(derived):
            raise ValueError("one feature cannot be both direct and derived")
        return self


class SourceCellValue(BaseModel):
    """One reconstructed direct source value."""

    model_config = _STRICT

    source: SourceCellRef
    value: JsonValue


class DocumentScenarioFeatures(BaseModel):
    """One bounded structural scenario row per source document."""

    model_config = _STRICT

    issue_date: IsoDate | None
    shipped_on_board_date: IsoDate | None
    issue_after_shipped_days: int | None
    negotiability: Literal["negotiable", "non_negotiable"] | None
    freight_payment_arrangement: Literal["prepaid", "collect"] | None
    container_count: NonNegativeInt
    seal_count: NonNegativeInt
    temperature_controlled_container_count: NonNegativeInt
    cargo_group_count: NonNegativeInt
    package_fact_count: NonNegativeInt
    multi_level_cargo_group_count: NonNegativeInt
    allocation_group_count: NonNegativeInt
    allocation_count: NonNegativeInt
    one_to_one_allocation_group_count: NonNegativeInt
    single_level_allocation_group_count: NonNegativeInt
    combined_levels_allocation_group_count: NonNegativeInt
    unlinked_quantity_allocation_group_count: NonNegativeInt
    membership_only_allocation_group_count: NonNegativeInt
    dangerous_goods_record_count: NonNegativeInt
    hs_code_count: NonNegativeInt
    party_count: NonNegativeInt
    notify_party_count: NonNegativeInt
    location_count: NonNegativeInt


class CargoPackageNumericFeatures(BaseModel):
    """One task-facing package fact with its cargo and relation context."""

    model_config = _STRICT

    cargo_group_order: NonNegativeInt
    task_package_position: NonNegativeInt
    task_package_level_count: PositiveInt
    source_package_position: NonNegativeInt
    source_package_level_count: PositiveInt
    metadata_only_package_count: NonNegativeInt
    package_role: PackageRole
    type_category: CategoryToken | None
    printed_surface: NonEmptyString | None
    quantity: NonNegativeInt | None
    gross_weight_value: Number | None
    gross_weight_unit: NonEmptyString | None
    net_weight_value: Number | None
    net_weight_unit: NonEmptyString | None
    volume_value: Number | None
    volume_unit: NonEmptyString | None
    allocation_coverage: ModeledAllocationCoverage
    package_in_allocation_scope: bool
    group_allocation_count: NonNegativeInt
    group_allocation_container_count: NonNegativeInt
    group_allocated_quantity: NonNegativeInt | None
    package_allocation_count: NonNegativeInt
    package_allocated_quantity: NonNegativeInt | None
    hs_code_count: NonNegativeInt
    dangerous_goods_record_count: NonNegativeInt

    @model_validator(mode="after")
    def measure_and_package_states_are_complete(self) -> CargoPackageNumericFeatures:
        for name in ("gross_weight", "net_weight", "volume"):
            value = getattr(self, f"{name}_value")
            unit = getattr(self, f"{name}_unit")
            if (value is None) != (unit is None):
                raise ValueError(f"{name} value and unit must be jointly present or absent")
        if self.type_category is not None and self.printed_surface is None:
            raise ValueError("a typed package requires its reviewed printed surface")
        if self.allocation_coverage == "none" and (
            self.group_allocation_count or self.package_in_allocation_scope
        ):
            raise ValueError("an unallocated package cannot carry allocation context")
        return self


class CargoGroupNumericFeatures(BaseModel):
    """One cargo-group payload with exactly one deterministic package driver."""

    model_config = _STRICT

    cargo_group_order: NonNegativeInt
    task_package_level_count: NonNegativeInt
    source_package_level_count: NonNegativeInt
    metadata_only_package_count: NonNegativeInt
    driver_task_package_position: NonNegativeInt | None
    driver_source_package_position: NonNegativeInt | None
    driver_package_role: PackageRole | None
    driver_type_category: CategoryToken | None
    driver_printed_surface: NonEmptyString | None
    driver_quantity: NonNegativeInt | None
    gross_weight_kg: Number | None
    net_weight_kg: Number | None
    volume_value: Number | None
    volume_unit: NonEmptyString | None
    allocation_coverage: ModeledAllocationCoverage
    group_allocation_count: NonNegativeInt
    group_allocation_container_count: NonNegativeInt
    group_allocated_quantity: NonNegativeInt | None
    hs_code_count: NonNegativeInt
    dangerous_goods_record_count: NonNegativeInt

    @model_validator(mode="after")
    def driver_and_volume_states_are_complete(self) -> CargoGroupNumericFeatures:
        driver_values = (
            self.driver_task_package_position,
            self.driver_source_package_position,
            self.driver_package_role,
            self.driver_quantity,
        )
        if any(value is None for value in driver_values) != all(
            value is None for value in driver_values
        ):
            raise ValueError("cargo-group driver core fields must be jointly present or absent")
        if self.driver_type_category is not None and self.driver_printed_surface is None:
            raise ValueError("a typed cargo-group driver requires its reviewed printed surface")
        if self.driver_package_role is None and (
            self.driver_type_category is not None or self.driver_printed_surface is not None
        ):
            raise ValueError("a cargo group without a driver cannot have driver package metadata")
        if (self.volume_value is None) != (self.volume_unit is None):
            raise ValueError("cargo-group volume value and unit must be jointly present or absent")
        if self.driver_task_package_position is not None and (
            self.driver_task_package_position >= self.task_package_level_count
        ):
            raise ValueError("cargo-group driver position exceeds its task package levels")
        return self


class CanonicalMassProjection(BaseModel):
    """Exact decimal provenance for a source mass converted to model-space kilograms."""

    model_config = _STRICT

    feature: MassFeatureName
    value_source: SourceCellRef
    unit_source: SourceCellRef
    source_value: Number | None
    source_unit: MassSourceUnit | None
    source_value_decimal: NonEmptyString | None
    source_decimal_places: NonNegativeInt | None
    canonical_unit: Literal["kilogram"] = "kilogram"
    multiplier_decimal: NonEmptyString | None
    canonical_value_decimal: NonEmptyString | None
    inverse_method: Literal[
        "divide_canonical_decimal_by_multiplier_then_restore_source_numeric_kind_v1"
    ]
    source_numeric_kind: Literal["integer", "decimal"] | None

    @model_validator(mode="after")
    def decimal_projection_is_exact(self) -> CanonicalMassProjection:
        optional_values = (
            self.source_value,
            self.source_unit,
            self.source_value_decimal,
            self.source_decimal_places,
            self.multiplier_decimal,
            self.canonical_value_decimal,
            self.source_numeric_kind,
        )
        if self.source_value is None:
            if any(value is not None for value in optional_values):
                raise ValueError("an absent source mass must have entirely absent decimal metadata")
            return self
        if any(value is None for value in optional_values):
            raise ValueError("a present source mass requires complete decimal projection metadata")
        assert self.source_unit is not None
        assert self.source_value_decimal is not None
        assert self.source_decimal_places is not None
        assert self.multiplier_decimal is not None
        assert self.canonical_value_decimal is not None
        assert self.source_numeric_kind is not None
        try:
            source_decimal = Decimal(self.source_value_decimal)
            multiplier = Decimal(self.multiplier_decimal)
            canonical = Decimal(self.canonical_value_decimal)
        except InvalidOperation as error:
            raise ValueError("mass projection contains invalid decimal text") from error
        if not source_decimal.is_finite() or not canonical.is_finite():
            raise ValueError("mass projection decimals must be finite")
        if multiplier != _MASS_TO_KILOGRAM[self.source_unit]:
            raise ValueError("mass projection multiplier differs from its source unit")
        if source_decimal != Decimal(str(self.source_value)):
            raise ValueError("mass projection decimal differs from its source value")
        if canonical != source_decimal * multiplier:
            raise ValueError("mass projection canonical decimal is not exact")
        if canonical / multiplier != source_decimal:
            raise ValueError("mass projection cannot invert exactly to its source decimal")
        expected_places = max(0, -cast(int, source_decimal.as_tuple().exponent))
        if self.source_decimal_places != expected_places:
            raise ValueError("mass projection source decimal places are inconsistent")
        expected_kind = "integer" if type(self.source_value) is int else "decimal"
        if self.source_numeric_kind != expected_kind:
            raise ValueError("mass projection source numeric kind is inconsistent")
        return self


class ContainerEquipmentFeatures(BaseModel):
    """One equipment row without its high-cardinality container identifier."""

    model_config = _STRICT

    container_order: NonNegativeInt
    document_container_count: PositiveInt
    type_category: NonEmptyString | None
    type_description_surface: NonEmptyString | None
    verified_gross_mass_value: Number | None
    verified_gross_mass_unit: NonEmptyString | None
    temperature_setpoint_value: Number | None
    temperature_setpoint_unit: NonEmptyString | None
    seal_count: NonNegativeInt
    allocation_count: NonNegativeInt
    allocated_package_quantity: NonNegativeInt | None
    allocated_cargo_group_count: NonNegativeInt
    one_to_one_allocation_count: NonNegativeInt
    single_level_allocation_count: NonNegativeInt
    combined_levels_allocation_count: NonNegativeInt
    unlinked_quantity_allocation_count: NonNegativeInt
    membership_only_allocation_count: NonNegativeInt

    @model_validator(mode="after")
    def measures_are_joint(self) -> ContainerEquipmentFeatures:
        for name in ("verified_gross_mass", "temperature_setpoint"):
            value = getattr(self, f"{name}_value")
            unit = getattr(self, f"{name}_unit")
            if (value is None) != (unit is None):
                raise ValueError(f"{name} value and unit must be jointly present or absent")
        return self


class DocumentScenarioObservation(BaseModel):
    model_config = _STRICT

    features: DocumentScenarioFeatures
    projection: ViewRowProjection


class CargoPackageNumericObservation(BaseModel):
    model_config = _STRICT

    features: CargoPackageNumericFeatures
    projection: ViewRowProjection


class CargoGroupNumericObservation(BaseModel):
    """One unique cargo group plus its non-modeled driver and inverse metadata."""

    model_config = _STRICT

    cargo_group_id: GroupId
    driver_package_id: PackageId | None
    features: CargoGroupNumericFeatures
    projection: ViewRowProjection
    mass_projections: tuple[CanonicalMassProjection, CanonicalMassProjection]

    @model_validator(mode="after")
    def driver_and_mass_metadata_match_features(self) -> CargoGroupNumericObservation:
        if (self.driver_package_id is None) != (self.features.driver_quantity is None):
            raise ValueError("cargo-group driver ID and driver features must be jointly present")
        projections = {row.feature: row for row in self.mass_projections}
        if set(projections) != {"gross_weight_kg", "net_weight_kg"}:
            raise ValueError("cargo-group mass projections must contain gross and net exactly once")
        for feature_name, projection in projections.items():
            modeled = getattr(self.features, feature_name)
            canonical = projection.canonical_value_decimal
            if (modeled is None) != (canonical is None):
                raise ValueError("cargo-group canonical mass and its projection disagree")
            if canonical is not None and Decimal(str(modeled)) != Decimal(canonical):
                raise ValueError("cargo-group canonical mass loses decimal projection value")
        return self


class ContainerEquipmentObservation(BaseModel):
    model_config = _STRICT

    features: ContainerEquipmentFeatures
    projection: ViewRowProjection


ModelingObservation = (
    DocumentScenarioObservation
    | CargoPackageNumericObservation
    | CargoGroupNumericObservation
    | ContainerEquipmentObservation
)


class PackageHierarchyRecord(BaseModel):
    """Lossless package-role ledger excluded from statistical model input."""

    model_config = _STRICT

    source_document_id: NonEmptyString
    group_id: GroupId
    source_package_id: PackageId
    source_package_position: NonNegativeInt
    role: PackageRole
    role_source: NonEmptyString
    disposition: PackageDisposition
    projected_package_id: PackageId | None
    quantity: NonNegativeInt | None
    type_description: NonEmptyString | None
    normalized_type: NonEmptyString | None

    @model_validator(mode="after")
    def disposition_and_projection_agree(self) -> PackageHierarchyRecord:
        if (self.disposition == "task_facing") != (self.projected_package_id is not None):
            raise ValueError("only task-facing package hierarchy rows have a projected ID")
        if (self.type_description is None) != (self.normalized_type is None):
            raise ValueError(
                "a printed hierarchy type and its normalized surface must be jointly present"
            )
        return self


class ModelingViewsAudit(BaseModel):
    model_config = _STRICT

    train_document_count: PositiveInt
    validated_target_count: PositiveInt
    document_scenario_rows: PositiveInt
    cargo_package_numeric_rows: NonNegativeInt
    cargo_group_numeric_rows: NonNegativeInt
    container_equipment_rows: NonNegativeInt
    package_hierarchy_rows: NonNegativeInt
    metadata_only_package_rows: NonNegativeInt
    multi_level_document_count: NonNegativeInt
    coverage_counts: dict[str, NonNegativeInt]
    direct_source_cell_count: NonNegativeInt
    package_registry_entries: PositiveInt
    package_surface_decisions: NonNegativeInt

    @model_validator(mode="after")
    def coverage_keys_are_exhaustive(self) -> ModelingViewsAudit:
        if set(self.coverage_counts) != set(_COVERAGE_VALUES):
            raise ValueError("modeling-view coverage audit must contain all coverage classes")
        return self


class ModelingViews(BaseModel):
    """Four compact views plus the exact metadata needed to audit them."""

    model_config = _STRICT

    schema_version: Literal[2]
    train_document_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    package_registry_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    package_surface_inventory: PackageSurfaceInventory
    package_hierarchy: tuple[PackageHierarchyRecord, ...]
    document_scenario: tuple[DocumentScenarioObservation, ...] = Field(min_length=1)
    cargo_package_numeric: tuple[CargoPackageNumericObservation, ...]
    cargo_group_numeric: tuple[CargoGroupNumericObservation, ...]
    container_equipment: tuple[ContainerEquipmentObservation, ...]
    audit: ModelingViewsAudit

    @model_validator(mode="after")
    def scope_and_counts_match(self) -> ModelingViews:
        if len(self.train_document_ids) != len(set(self.train_document_ids)):
            raise ValueError("modeling-view train document IDs must be unique")
        allowed = set(self.train_document_ids)
        document_ids = tuple(row.projection.source_document_id for row in self.document_scenario)
        if document_ids != self.train_document_ids:
            raise ValueError("document-scenario rows must exactly follow train source order")
        for rows in (
            self.cargo_package_numeric,
            self.cargo_group_numeric,
            self.container_equipment,
        ):
            if any(row.projection.source_document_id not in allowed for row in rows):
                raise ValueError("modeling view contains a document outside train scope")
        if any(row.source_document_id not in allowed for row in self.package_hierarchy):
            raise ValueError("package hierarchy contains a document outside train scope")
        if self.audit.document_scenario_rows != len(self.document_scenario):
            raise ValueError("document-scenario audit count differs from rows")
        if self.audit.cargo_package_numeric_rows != len(self.cargo_package_numeric):
            raise ValueError("cargo/package audit count differs from rows")
        if self.audit.cargo_group_numeric_rows != len(self.cargo_group_numeric):
            raise ValueError("cargo-group audit count differs from rows")
        if self.audit.container_equipment_rows != len(self.container_equipment):
            raise ValueError("container/equipment audit count differs from rows")
        if self.audit.package_hierarchy_rows != len(self.package_hierarchy):
            raise ValueError("package hierarchy audit count differs from rows")
        group_keys = tuple(row.projection.view_row_key for row in self.cargo_group_numeric)
        if len(group_keys) != len(set(group_keys)):
            raise ValueError("cargo-group view row keys must be unique")
        group_ids = tuple(
            (row.projection.source_document_id, row.cargo_group_id)
            for row in self.cargo_group_numeric
        )
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("cargo-group view must contain one row per document/group")
        return self

    def model_rows(
        self,
        view: Literal[
            "document_scenario",
            "cargo_package_numeric",
            "cargo_group_numeric",
            "container_equipment",
        ],
    ) -> tuple[dict[str, Any], ...]:
        """Return model-ready rows; source IDs and free-text lineage stay excluded."""

        observations = getattr(self, view)
        return tuple(row.features.model_dump(mode="python") for row in observations)

    def inverse_direct_source_cells(
        self,
        view: Literal[
            "document_scenario",
            "cargo_package_numeric",
            "cargo_group_numeric",
            "container_equipment",
        ],
    ) -> tuple[SourceCellValue, ...]:
        """Recover every directly projected source cell and reject conflicting repeats."""

        observations = getattr(self, view)
        recovered: dict[tuple[str, str, str], SourceCellValue] = {}
        for observation in observations:
            features = observation.features.model_dump(mode="json")
            for direct in observation.projection.direct:
                value = cast(JsonValue, features[direct.feature])
                key = (direct.source.table, direct.source.row_key, direct.source.column)
                current = SourceCellValue(source=direct.source, value=value)
                previous = recovered.setdefault(key, current)
                if previous.value != value:
                    raise ValueError(f"conflicting inverse values for source cell {key}")
            if isinstance(observation, CargoGroupNumericObservation):
                for mass in observation.mass_projections:
                    for source, value in (
                        (mass.value_source, mass.source_value),
                        (mass.unit_source, mass.source_unit),
                    ):
                        key = (source.table, source.row_key, source.column)
                        current = SourceCellValue(source=source, value=cast(JsonValue, value))
                        previous = recovered.setdefault(key, current)
                        if previous.value != value:
                            raise ValueError(f"conflicting inverse values for source cell {key}")
        return tuple(recovered[key] for key in sorted(recovered))


def _table_indexes(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[str, dict[str, Mapping[str, Any]]],
    dict[str, dict[str, tuple[Mapping[str, Any], ...]]],
]:
    if set(tables) != set(ADAPTER.table_order):
        missing = sorted(set(ADAPTER.table_order) - set(tables))
        extra = sorted(set(tables) - set(ADAPTER.table_order))
        raise ValueError(
            f"relation-table set differs from adapter: missing={missing}, extra={extra}"
        )
    indexes: dict[str, dict[str, Mapping[str, Any]]] = {}
    by_document: dict[str, dict[str, tuple[Mapping[str, Any], ...]]] = defaultdict(dict)
    for table in ADAPTER.table_order:
        primary_key = _TABLE_PRIMARY_KEYS[table]
        index: dict[str, Mapping[str, Any]] = {}
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in tables[table]:
            key = row.get(primary_key)
            document_id = row.get("document_id")
            if not isinstance(key, str) or not key or not isinstance(document_id, str):
                raise ValueError(f"{table} contains an invalid primary key or document_id")
            if key in index:
                raise ValueError(f"duplicate {table} primary key: {key}")
            index[key] = row
            grouped[document_id].append(row)
        indexes[table] = index
        for document_id, rows in grouped.items():
            by_document[document_id][table] = tuple(rows)
    return indexes, by_document


def _sidecars_by_document(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        document_id = row.get("documentId")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{label} contains an invalid documentId")
        if document_id in output:
            raise ValueError(f"duplicate {label} row: {document_id}")
        output[document_id] = row
    return output


def _source_ref(table: str, row_key: str, column: str) -> SourceCellRef:
    return SourceCellRef(table=table, row_key=row_key, column=column)


def _projection(
    *,
    view_row_key: str,
    document_id: str,
    feature_values: Mapping[str, Any],
    direct: Mapping[str, SourceCellRef],
    derived: Mapping[str, str],
    source_values: Mapping[tuple[str, str, str], Any],
) -> ViewRowProjection:
    feature_names = set(feature_values)
    classified = set(direct) | set(derived)
    if feature_names != classified:
        raise ValueError(
            f"feature projection is not exhaustive for {view_row_key}: "
            f"missing={sorted(feature_names - classified)}, "
            f"extra={sorted(classified - feature_names)}"
        )
    if set(direct) & set(derived):
        raise ValueError(f"feature projection overlaps for {view_row_key}")
    for feature, ref in direct.items():
        key = (ref.table, ref.row_key, ref.column)
        if key not in source_values:
            raise ValueError(f"direct feature references an absent source cell: {key}")
        if feature_values[feature] != source_values[key]:
            raise ValueError(f"direct feature differs from source cell: {view_row_key}:{feature}")
    return ViewRowProjection(
        view_row_key=view_row_key,
        source_document_id=document_id,
        direct=tuple(
            DirectFeatureProjection(feature=name, source=direct[name]) for name in sorted(direct)
        ),
        derived=tuple(
            DerivedFeatureProjection(feature=name, method=derived[name]) for name in sorted(derived)
        ),
    )


def _coverage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(cast(str, row["coverage"]) for row in rows)
    unknown = set(counts) - set(_COVERAGE_VALUES)
    if unknown:
        raise ValueError(f"unsupported allocation coverage classes: {sorted(unknown)}")
    return {value: counts[value] for value in _COVERAGE_VALUES}


def _sum_optional_integers(values: Sequence[Any]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in present):
        raise ValueError("allocation quantities must be non-negative integers")
    return sum(cast(int, value) for value in present)


@dataclass(frozen=True, slots=True)
class _CargoGroupRelationContext:
    allocation_group: Mapping[str, Any] | None
    allocations: tuple[Mapping[str, Any], ...]
    memberships: tuple[Mapping[str, Any], ...]
    scope_package_ids: frozenset[str]
    allocated_quantity: int | None
    coverage: ModeledAllocationCoverage


def _cargo_group_driver(
    *,
    task_packages: Sequence[Mapping[str, Any]],
    projected_hierarchy: Mapping[str, PackageHierarchyRecord],
) -> tuple[int, Mapping[str, Any], PackageHierarchyRecord] | None:
    """Select the same stable package driver used by group-level generation."""

    quantified: list[tuple[int, Mapping[str, Any], PackageHierarchyRecord]] = []
    for task_position, package in enumerate(task_packages):
        if package.get("quantity") is None:
            continue
        package_id = cast(str, package["package_id"])
        quantified.append((task_position, package, projected_hierarchy[package_id]))
    if not quantified:
        return None
    return min(
        quantified,
        key=lambda row: (
            _PACKAGE_ROLE_PRIORITY[row[2].role],
            row[0],
            cast(str, row[1]["package_id"]),
        ),
    )


def _canonical_mass_projection(
    *,
    feature: MassFeatureName,
    group_row_key: str,
    value_column: str,
    unit_column: str,
    value: Any,
    unit: Any,
) -> tuple[float | None, CanonicalMassProjection]:
    value_source = _source_ref("cargo_groups", group_row_key, value_column)
    unit_source = _source_ref("cargo_groups", group_row_key, unit_column)
    inverse_method: Literal[
        "divide_canonical_decimal_by_multiplier_then_restore_source_numeric_kind_v1"
    ] = "divide_canonical_decimal_by_multiplier_then_restore_source_numeric_kind_v1"
    if value is None or unit is None:
        if value is not None or unit is not None:
            raise ValueError("cargo-group mass value and unit must be jointly present or absent")
        return None, CanonicalMassProjection(
            feature=feature,
            value_source=value_source,
            unit_source=unit_source,
            source_value=None,
            source_unit=None,
            source_value_decimal=None,
            source_decimal_places=None,
            multiplier_decimal=None,
            canonical_value_decimal=None,
            inverse_method=inverse_method,
            source_numeric_kind=None,
        )
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError("cargo-group mass value must be an integer or float")
    if unit not in _MASS_TO_KILOGRAM:
        raise ValueError(f"unsupported cargo-group mass unit: {unit!r}")
    source_decimal = Decimal(str(value))
    if not source_decimal.is_finite():
        raise ValueError("cargo-group mass value must be finite")
    source_unit = cast(MassSourceUnit, unit)
    multiplier = _MASS_TO_KILOGRAM[source_unit]
    canonical = source_decimal * multiplier
    return float(canonical), CanonicalMassProjection(
        feature=feature,
        value_source=value_source,
        unit_source=unit_source,
        source_value=value,
        source_unit=source_unit,
        source_value_decimal=format(source_decimal, "f"),
        source_decimal_places=max(0, -cast(int, source_decimal.as_tuple().exponent)),
        multiplier_decimal=format(multiplier, "f"),
        canonical_value_decimal=format(canonical, "f"),
        inverse_method=inverse_method,
        source_numeric_kind="integer" if type(value) is int else "decimal",
    )


def build_modeling_views(
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    train_document_ids: Sequence[str],
    package_registry: LoadedPackageRegistry,
    category_metadata_rows: Sequence[Mapping[str, Any]],
    package_hierarchy_metadata_rows: Sequence[Mapping[str, Any]],
) -> ModelingViews:
    """Build and validate the three compact views for an explicit train scope."""

    if not train_document_ids or len(train_document_ids) != len(set(train_document_ids)):
        raise ValueError("train document IDs must be non-empty and unique")
    indexes, by_document = _table_indexes(tables)
    document_index = indexes["documents"]
    unknown = [
        document_id for document_id in train_document_ids if document_id not in document_index
    ]
    if unknown:
        raise ValueError(f"train scope references absent documents: {unknown[:5]}")
    category_by_document = _sidecars_by_document(
        category_metadata_rows, label="package category metadata"
    )
    hierarchy_by_document = _sidecars_by_document(
        package_hierarchy_metadata_rows, label="package hierarchy metadata"
    )
    for label, rows in (
        ("package category metadata", category_by_document),
        ("package hierarchy metadata", hierarchy_by_document),
    ):
        missing = [document_id for document_id in train_document_ids if document_id not in rows]
        if missing:
            raise ValueError(f"{label} is missing train documents: {missing[:5]}")

    surface_inventory = build_package_surface_inventory(
        category_metadata_rows,
        train_document_ids=train_document_ids,
        registry=package_registry,
    )
    source_values: dict[tuple[str, str, str], Any] = {}
    train_scope = set(train_document_ids)
    for table, index in indexes.items():
        for row_key, row in index.items():
            if row["document_id"] not in train_scope:
                continue
            for column, value in row.items():
                source_values[(table, row_key, column)] = value

    document_observations: list[DocumentScenarioObservation] = []
    cargo_observations: list[CargoPackageNumericObservation] = []
    cargo_group_observations: list[CargoGroupNumericObservation] = []
    container_observations: list[ContainerEquipmentObservation] = []
    hierarchy_records: list[PackageHierarchyRecord] = []
    coverage_counter: Counter[str] = Counter()
    multi_level_documents: set[str] = set()

    for document_id in train_document_ids:
        document_tables = {
            table: by_document.get(document_id, {}).get(table, ()) for table in ADAPTER.table_order
        }
        reconstructed = ADAPTER.reconstruct(document_id=document_id, tables=document_tables)
        BillOfLadingRelationExplicitLabel.model_validate_json(
            canonical_json_bytes(reconstructed), strict=True
        )
        document = document_index[document_id]
        cargo_groups = sorted(
            document_tables["cargo_groups"], key=lambda row: cast(int, row["group_order"])
        )
        packages = sorted(
            document_tables["packages"], key=lambda row: cast(int, row["package_order"])
        )
        containers = sorted(
            document_tables["containers"], key=lambda row: cast(int, row["container_order"])
        )
        allocation_groups = sorted(
            document_tables["allocation_groups"],
            key=lambda row: cast(int, row["allocation_group_order"]),
        )
        allocations = document_tables["allocations"]
        memberships = document_tables["allocation_group_packages"]
        coverage_counter.update(cast(str, row["coverage"]) for row in allocation_groups)

        # Validate package category decisions against task-facing package rows.
        raw_decisions = category_by_document[document_id].get("packageDecisions")
        if not isinstance(raw_decisions, list):
            raise ValueError(f"packageDecisions must be an array: {document_id}")
        decisions: dict[str, Mapping[str, Any]] = {}
        for decision in raw_decisions:
            if not isinstance(decision, Mapping):
                raise ValueError(f"invalid package category decision: {document_id}")
            package_id = decision.get("packageId")
            if not isinstance(package_id, str) or package_id in decisions:
                raise ValueError(f"invalid or duplicate package category decision: {document_id}")
            decisions[package_id] = decision

        packages_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for package in packages:
            packages_by_group[cast(str, package["group_id"])].append(package)

        # Losslessly map source hierarchy rows to renumbered task-facing package rows.
        raw_diagnoses = hierarchy_by_document[document_id].get("groupDiagnoses")
        if not isinstance(raw_diagnoses, list):
            raise ValueError(f"groupDiagnoses must be an array: {document_id}")
        diagnoses: dict[str, Mapping[str, Any]] = {}
        projected_hierarchy: dict[str, PackageHierarchyRecord] = {}
        for diagnosis in raw_diagnoses:
            if not isinstance(diagnosis, Mapping):
                raise ValueError(f"invalid package hierarchy diagnosis: {document_id}")
            group_id = diagnosis.get("groupId")
            if not isinstance(group_id, str) or group_id in diagnoses:
                raise ValueError(f"invalid or duplicate hierarchy group: {document_id}")
            diagnoses[group_id] = diagnosis
            raw_packages = diagnosis.get("packages")
            retained_ids = diagnosis.get("retainedPackageIds")
            metadata_ids = diagnosis.get("metadataPackageIds")
            if (
                not isinstance(raw_packages, list)
                or not isinstance(retained_ids, list)
                or not isinstance(metadata_ids, list)
            ):
                raise ValueError(f"invalid package hierarchy arrays: {document_id}:{group_id}")
            source_ids = [row.get("packageId") for row in raw_packages if isinstance(row, Mapping)]
            if (
                len(source_ids) != len(raw_packages)
                or any(not isinstance(value, str) for value in source_ids)
                or len(source_ids) != len(set(source_ids))
            ):
                raise ValueError(f"invalid package hierarchy identifiers: {document_id}:{group_id}")
            if set(retained_ids) & set(metadata_ids) or set(retained_ids) | set(
                metadata_ids
            ) != set(source_ids):
                raise ValueError(
                    "package hierarchy disposition does not partition rows: "
                    f"{document_id}:{group_id}"
                )
            task_packages = packages_by_group.get(group_id, [])
            retained_rows = [row for row in raw_packages if row["packageId"] in set(retained_ids)]
            if len(task_packages) != len(retained_rows):
                raise ValueError(
                    f"task-facing package count differs from hierarchy: {document_id}:{group_id}"
                )
            task_by_source_id = {
                cast(str, source_row["packageId"]): task_row
                for source_row, task_row in zip(retained_rows, task_packages, strict=True)
            }
            for source_position, raw_package in enumerate(raw_packages):
                if not isinstance(raw_package, Mapping) or not isinstance(
                    raw_package.get("value"), Mapping
                ):
                    raise ValueError(f"invalid package hierarchy row: {document_id}:{group_id}")
                source_package_id = cast(str, raw_package["packageId"])
                value = cast(Mapping[str, Any], raw_package["value"])
                role = raw_package.get("role")
                role_source = raw_package.get("roleSource")
                if role not in {"direct_goods", "generic_aggregate", "outer_transport", "unknown"}:
                    raise ValueError(f"unsupported package hierarchy role: {role!r}")
                if not isinstance(role_source, str) or not role_source:
                    raise ValueError("package hierarchy roleSource must be non-empty")
                task_row = task_by_source_id.get(source_package_id)
                disposition: PackageDisposition = (
                    "task_facing" if task_row is not None else "metadata_only"
                )
                projected_package_id = (
                    cast(str, task_row["package_id"]) if task_row is not None else None
                )
                quantity = value.get("quantity")
                type_description = value.get("typeDescription")
                normalized_type = raw_package.get("normalizedType")
                record = PackageHierarchyRecord(
                    source_document_id=document_id,
                    group_id=group_id,
                    source_package_id=source_package_id,
                    source_package_position=source_position,
                    role=cast(PackageRole, role),
                    role_source=role_source,
                    disposition=disposition,
                    projected_package_id=projected_package_id,
                    quantity=quantity,
                    type_description=type_description,
                    normalized_type=normalized_type,
                )
                hierarchy_records.append(record)
                hierarchy_key = f"{document_id}:{group_id}:{source_package_id}"
                for column, cell_value in {
                    "role": role,
                    "role_source": role_source,
                    "quantity": quantity,
                    "type_description": type_description,
                    "normalized_type": normalized_type,
                }.items():
                    source_values[("package_hierarchy_metadata", hierarchy_key, column)] = (
                        cell_value
                    )
                if task_row is not None:
                    if task_row.get("quantity") != quantity:
                        raise ValueError(
                            "projected package quantity differs from hierarchy: "
                            f"{document_id}:{group_id}"
                        )
                    projected_hierarchy[cast(str, task_row["package_id"])] = record

        if set(diagnoses) != {cast(str, row["group_id"]) for row in cargo_groups}:
            raise ValueError(f"package hierarchy groups differ from cargo groups: {document_id}")
        if set(projected_hierarchy) != {cast(str, row["package_id"]) for row in packages}:
            raise ValueError(
                f"package hierarchy projection differs from task packages: {document_id}"
            )
        if any(len(rows) > 1 for rows in packages_by_group.values()):
            multi_level_documents.add(document_id)

        # Register category sidecar cells after hierarchy mapping validates surfaces.
        for package in packages:
            package_id = cast(str, package["package_id"])
            decision = decisions.get(package_id)
            if decision is None:
                if (
                    package.get("type_category") is not None
                    or package.get("type_description") is not None
                ):
                    raise ValueError(
                        f"typed package is missing a category decision: {document_id}:{package_id}"
                    )
                continue
            category_token = decision.get("categoryToken")
            printed_surface = decision.get("sourceTypeDescription")
            if not isinstance(printed_surface, str):
                raise ValueError(
                    f"package category decision has no printed surface: {document_id}:{package_id}"
                )
            if category_token != package.get("type_category") and (
                category_token is not None or printed_surface != package.get("type_description")
            ):
                raise ValueError(
                    "package category decision differs from task target: "
                    f"{document_id}:{package_id}"
                )
            if decision.get("groupId") != package.get("group_id"):
                raise ValueError(
                    f"package category decision differs from task group: {document_id}:{package_id}"
                )
            source_record = projected_hierarchy[package_id]
            if printed_surface != source_record.type_description:
                raise ValueError(
                    f"package category surface differs from hierarchy: {document_id}:{package_id}"
                )
            decision_key = f"{document_id}:{package_id}"
            source_values[("package_category_metadata", decision_key, "printed_surface")] = (
                printed_surface
            )
            source_values[("package_category_metadata", decision_key, "category_token")] = (
                category_token
            )
        if set(decisions) != {
            cast(str, row["package_id"])
            for row in packages
            if row.get("type_category") is not None or row.get("type_description") is not None
        }:
            raise ValueError(
                f"package category decision IDs differ from typed packages: {document_id}"
            )

        # Document scenario view.
        issue = document.get("issue_date")
        shipped = document.get("shipped_on_board_date")
        day_delta = (
            (date.fromisoformat(issue) - date.fromisoformat(shipped)).days
            if isinstance(issue, str) and isinstance(shipped, str)
            else None
        )
        package_counts = Counter(cast(str, row["group_id"]) for row in packages)
        coverage = _coverage_counts(allocation_groups)
        notify_party_count = sum(
            row.get("role") == "notifyParties" for row in document_tables["parties"]
        )
        document_features = DocumentScenarioFeatures(
            issue_date=issue,
            shipped_on_board_date=shipped,
            issue_after_shipped_days=day_delta,
            negotiability=document.get("negotiability"),
            freight_payment_arrangement=document.get("freight_payment_arrangement"),
            container_count=len(containers),
            seal_count=len(document_tables["container_seals"]),
            temperature_controlled_container_count=sum(
                row.get("temperature_setpoint_value") is not None for row in containers
            ),
            cargo_group_count=len(cargo_groups),
            package_fact_count=len(packages),
            multi_level_cargo_group_count=sum(value > 1 for value in package_counts.values()),
            allocation_group_count=len(allocation_groups),
            allocation_count=len(allocations),
            one_to_one_allocation_group_count=coverage["one_to_one_package_allocations"],
            single_level_allocation_group_count=coverage["single_package_level"],
            combined_levels_allocation_group_count=coverage["all_package_levels_combined"],
            unlinked_quantity_allocation_group_count=coverage["unlinked_package_quantities"],
            membership_only_allocation_group_count=coverage["container_membership_only"],
            dangerous_goods_record_count=len(document_tables["dangerous_goods"]),
            hs_code_count=len(document_tables["cargo_hs_codes"]),
            party_count=len(document_tables["parties"]),
            notify_party_count=notify_party_count,
            location_count=len(document_tables["document_locations"]),
        )
        doc_values = document_features.model_dump(mode="python")
        doc_key = cast(str, document["document_id"])
        doc_direct = {
            name: _source_ref("documents", doc_key, name)
            for name in (
                "issue_date",
                "shipped_on_board_date",
                "negotiability",
                "freight_payment_arrangement",
            )
        }
        doc_derived = {
            name: "derive_relation_graph_cardinality_v1"
            for name in set(doc_values) - set(doc_direct)
        }
        doc_derived["issue_after_shipped_days"] = "derive_iso_date_delta_days_v1"
        document_observations.append(
            DocumentScenarioObservation(
                features=document_features,
                projection=_projection(
                    view_row_key=f"document:{document_id}",
                    document_id=document_id,
                    feature_values=doc_values,
                    direct=doc_direct,
                    derived=doc_derived,
                    source_values=source_values,
                ),
            )
        )

        cargo_group_by_id = {cast(str, row["group_id"]): row for row in cargo_groups}
        allocation_group_by_group = {cast(str, row["group_id"]): row for row in allocation_groups}
        hs_by_group = Counter(
            cast(str, row["cargo_group_row_id"]) for row in document_tables["cargo_hs_codes"]
        )
        dg_by_group = Counter(
            cast(str, row["cargo_group_row_id"]) for row in document_tables["dangerous_goods"]
        )

        relation_context_by_group: dict[str, _CargoGroupRelationContext] = {}
        for group in cargo_groups:
            group_id = cast(str, group["group_id"])
            allocation_group = allocation_group_by_group.get(group_id)
            group_allocations = tuple(
                row
                for row in allocations
                if allocation_group is not None
                and row["allocation_group_id"] == allocation_group["allocation_group_id"]
            )
            group_memberships = tuple(
                row
                for row in memberships
                if allocation_group is not None
                and row["allocation_group_id"] == allocation_group["allocation_group_id"]
            )
            relation_context_by_group[group_id] = _CargoGroupRelationContext(
                allocation_group=allocation_group,
                allocations=group_allocations,
                memberships=group_memberships,
                scope_package_ids=frozenset(
                    cast(str, row["package_id"]) for row in group_memberships
                ),
                allocated_quantity=_sum_optional_integers(
                    [row.get("package_quantity") for row in group_allocations]
                ),
                coverage=(
                    cast(AllocationCoverage, allocation_group["coverage"])
                    if allocation_group is not None
                    else "none"
                ),
            )

        # One row per cargo group. Package identity remains driver metadata and
        # group-level measures are canonicalized once rather than repeated for
        # every retained package level.
        for group in cargo_groups:
            group_id = cast(str, group["group_id"])
            group_row_key = cast(str, group["cargo_group_row_id"])
            task_packages = packages_by_group.get(group_id, [])
            diagnosis = diagnoses[group_id]
            source_package_count = len(cast(Sequence[Any], diagnosis["packages"]))
            metadata_package_count = len(cast(Sequence[Any], diagnosis["metadataPackageIds"]))
            context = relation_context_by_group[group_id]
            driver = _cargo_group_driver(
                task_packages=task_packages,
                projected_hierarchy=projected_hierarchy,
            )
            driver_package_id: str | None = None
            driver_position: int | None = None
            driver_package: Mapping[str, Any] | None = None
            driver_hierarchy: PackageHierarchyRecord | None = None
            driver_decision: Mapping[str, Any] | None = None
            if driver is not None:
                driver_position, driver_package, driver_hierarchy = driver
                driver_package_id = cast(str, driver_package["package_id"])
                driver_decision = decisions.get(driver_package_id)
            gross_kg, gross_projection = _canonical_mass_projection(
                feature="gross_weight_kg",
                group_row_key=group_row_key,
                value_column="gross_weight_value",
                unit_column="gross_weight_unit",
                value=group.get("gross_weight_value"),
                unit=group.get("gross_weight_unit"),
            )
            net_kg, net_projection = _canonical_mass_projection(
                feature="net_weight_kg",
                group_row_key=group_row_key,
                value_column="net_weight_value",
                unit_column="net_weight_unit",
                value=group.get("net_weight_value"),
                unit=group.get("net_weight_unit"),
            )
            group_features = CargoGroupNumericFeatures(
                cargo_group_order=group["group_order"],
                task_package_level_count=len(task_packages),
                source_package_level_count=source_package_count,
                metadata_only_package_count=metadata_package_count,
                driver_task_package_position=driver_position,
                driver_source_package_position=(
                    driver_hierarchy.source_package_position
                    if driver_hierarchy is not None
                    else None
                ),
                driver_package_role=(
                    driver_hierarchy.role if driver_hierarchy is not None else None
                ),
                driver_type_category=(
                    driver_package.get("type_category") if driver_package is not None else None
                ),
                driver_printed_surface=(
                    cast(str, driver_decision["sourceTypeDescription"])
                    if driver_decision is not None
                    else None
                ),
                driver_quantity=(
                    driver_package.get("quantity") if driver_package is not None else None
                ),
                gross_weight_kg=gross_kg,
                net_weight_kg=net_kg,
                volume_value=group.get("volume_value"),
                volume_unit=group.get("volume_unit"),
                allocation_coverage=context.coverage,
                group_allocation_count=len(context.allocations),
                group_allocation_container_count=len(
                    {row["container_id"] for row in context.allocations}
                ),
                group_allocated_quantity=context.allocated_quantity,
                hs_code_count=hs_by_group[group_row_key],
                dangerous_goods_record_count=dg_by_group[group_row_key],
            )
            group_values = group_features.model_dump(mode="python")
            group_direct: dict[str, SourceCellRef] = {
                "cargo_group_order": _source_ref("cargo_groups", group_row_key, "group_order"),
                "volume_value": _source_ref("cargo_groups", group_row_key, "volume_value"),
                "volume_unit": _source_ref("cargo_groups", group_row_key, "volume_unit"),
            }
            if driver_package is not None and driver_hierarchy is not None:
                driver_package_row_key = cast(str, driver_package["package_row_id"])
                group_direct.update(
                    {
                        "driver_type_category": _source_ref(
                            "packages", driver_package_row_key, "type_category"
                        ),
                        "driver_quantity": _source_ref(
                            "packages", driver_package_row_key, "quantity"
                        ),
                        "driver_package_role": _source_ref(
                            "package_hierarchy_metadata",
                            f"{document_id}:{group_id}:{driver_hierarchy.source_package_id}",
                            "role",
                        ),
                    }
                )
                if driver_decision is not None:
                    group_direct["driver_printed_surface"] = _source_ref(
                        "package_category_metadata",
                        f"{document_id}:{driver_package_id}",
                        "printed_surface",
                    )
            if context.allocation_group is not None:
                group_direct["allocation_coverage"] = _source_ref(
                    "allocation_groups",
                    cast(str, context.allocation_group["allocation_group_id"]),
                    "coverage",
                )
            group_derived = {
                name: "derive_cargo_group_driver_and_relation_context_v1"
                for name in set(group_values) - set(group_direct)
            }
            group_derived["gross_weight_kg"] = "canonicalize_mass_decimal_to_kilogram_v1"
            group_derived["net_weight_kg"] = "canonicalize_mass_decimal_to_kilogram_v1"
            if driver_package is not None and driver_decision is None:
                group_derived["driver_printed_surface"] = (
                    "derive_absent_quantity_only_package_surface_v1"
                )
            cargo_group_observations.append(
                CargoGroupNumericObservation(
                    cargo_group_id=group_id,
                    driver_package_id=driver_package_id,
                    features=group_features,
                    projection=_projection(
                        view_row_key=f"cargo-group:{group_row_key}",
                        document_id=document_id,
                        feature_values=group_values,
                        direct=group_direct,
                        derived=group_derived,
                        source_values=source_values,
                    ),
                    mass_projections=(gross_projection, net_projection),
                )
            )

        # Cargo/package numeric view, preserving every retained level and coverage class.
        for group_id, task_packages in packages_by_group.items():
            group = cargo_group_by_id[group_id]
            diagnosis = diagnoses[group_id]
            source_package_count = len(cast(Sequence[Any], diagnosis["packages"]))
            metadata_package_count = len(cast(Sequence[Any], diagnosis["metadataPackageIds"]))
            context = relation_context_by_group[group_id]
            allocation_group = context.allocation_group
            group_allocations = context.allocations
            scope_ids = context.scope_package_ids
            group_quantity = context.allocated_quantity
            allocation_coverage = context.coverage
            for task_position, package in enumerate(task_packages):
                package_id = cast(str, package["package_id"])
                hierarchy = projected_hierarchy[package_id]
                package_allocations = [
                    row for row in group_allocations if row.get("package_id") == package_id
                ]
                package_quantity = _sum_optional_integers(
                    [row.get("package_quantity") for row in package_allocations]
                )
                decision = decisions.get(package_id)
                printed_surface = (
                    cast(str, decision["sourceTypeDescription"]) if decision is not None else None
                )
                cargo_features = CargoPackageNumericFeatures(
                    cargo_group_order=group["group_order"],
                    task_package_position=task_position,
                    task_package_level_count=len(task_packages),
                    source_package_position=hierarchy.source_package_position,
                    source_package_level_count=source_package_count,
                    metadata_only_package_count=metadata_package_count,
                    package_role=hierarchy.role,
                    type_category=package.get("type_category"),
                    printed_surface=printed_surface,
                    quantity=package.get("quantity"),
                    gross_weight_value=group.get("gross_weight_value"),
                    gross_weight_unit=group.get("gross_weight_unit"),
                    net_weight_value=group.get("net_weight_value"),
                    net_weight_unit=group.get("net_weight_unit"),
                    volume_value=group.get("volume_value"),
                    volume_unit=group.get("volume_unit"),
                    allocation_coverage=allocation_coverage,
                    package_in_allocation_scope=package_id in scope_ids,
                    group_allocation_count=len(group_allocations),
                    group_allocation_container_count=len(
                        {row["container_id"] for row in group_allocations}
                    ),
                    group_allocated_quantity=group_quantity,
                    package_allocation_count=len(package_allocations),
                    package_allocated_quantity=package_quantity,
                    hs_code_count=hs_by_group[cast(str, group["cargo_group_row_id"])],
                    dangerous_goods_record_count=dg_by_group[
                        cast(str, group["cargo_group_row_id"])
                    ],
                )
                cargo_values = cargo_features.model_dump(mode="python")
                package_row_key = cast(str, package["package_row_id"])
                group_row_key = cast(str, group["cargo_group_row_id"])
                direct = {
                    "type_category": _source_ref("packages", package_row_key, "type_category"),
                    "quantity": _source_ref("packages", package_row_key, "quantity"),
                    **{
                        name: _source_ref("cargo_groups", group_row_key, name)
                        for name in (
                            "gross_weight_value",
                            "gross_weight_unit",
                            "net_weight_value",
                            "net_weight_unit",
                            "volume_value",
                            "volume_unit",
                        )
                    },
                    "package_role": _source_ref(
                        "package_hierarchy_metadata",
                        f"{document_id}:{group_id}:{hierarchy.source_package_id}",
                        "role",
                    ),
                }
                if decision is not None:
                    direct["printed_surface"] = _source_ref(
                        "package_category_metadata",
                        f"{document_id}:{package_id}",
                        "printed_surface",
                    )
                else:
                    # Absence is derived from a verified quantity-only task package.
                    pass
                if allocation_group is not None:
                    direct["allocation_coverage"] = _source_ref(
                        "allocation_groups",
                        cast(str, allocation_group["allocation_group_id"]),
                        "coverage",
                    )
                derived = {
                    name: "derive_cargo_package_relation_context_v1"
                    for name in set(cargo_values) - set(direct)
                }
                if decision is None:
                    derived["printed_surface"] = "derive_absent_quantity_only_package_surface_v1"
                cargo_observations.append(
                    CargoPackageNumericObservation(
                        features=cargo_features,
                        projection=_projection(
                            view_row_key=f"cargo-package:{package_row_key}",
                            document_id=document_id,
                            feature_values=cargo_values,
                            direct=direct,
                            derived=derived,
                            source_values=source_values,
                        ),
                    )
                )

        # Container/equipment view, with allocation links summarized deterministically.
        seals_by_container = Counter(
            cast(str, row["container_id"]) for row in document_tables["container_seals"]
        )
        allocation_group_index = {
            cast(str, row["allocation_group_id"]): row for row in allocation_groups
        }
        for container in containers:
            container_id = cast(str, container["container_id"])
            container_allocations = [
                row for row in allocations if row["container_id"] == container_id
            ]
            coverage_counts = Counter(
                cast(str, allocation_group_index[cast(str, row["allocation_group_id"])]["coverage"])
                for row in container_allocations
            )
            type_description = container.get("type_description")
            container_features = ContainerEquipmentFeatures(
                container_order=container["container_order"],
                document_container_count=len(containers),
                type_category=container.get("type_category"),
                type_description_surface=(
                    normalize_printed_surface(cast(str, type_description))
                    if type_description is not None
                    else None
                ),
                verified_gross_mass_value=container.get("verified_gross_mass_value"),
                verified_gross_mass_unit=container.get("verified_gross_mass_unit"),
                temperature_setpoint_value=container.get("temperature_setpoint_value"),
                temperature_setpoint_unit=container.get("temperature_setpoint_unit"),
                seal_count=seals_by_container[container_id],
                allocation_count=len(container_allocations),
                allocated_package_quantity=_sum_optional_integers(
                    [row.get("package_quantity") for row in container_allocations]
                ),
                allocated_cargo_group_count=len(
                    {
                        allocation_group_index[cast(str, row["allocation_group_id"])][
                            "cargo_group_row_id"
                        ]
                        for row in container_allocations
                    }
                ),
                one_to_one_allocation_count=coverage_counts["one_to_one_package_allocations"],
                single_level_allocation_count=coverage_counts["single_package_level"],
                combined_levels_allocation_count=coverage_counts["all_package_levels_combined"],
                unlinked_quantity_allocation_count=coverage_counts["unlinked_package_quantities"],
                membership_only_allocation_count=coverage_counts["container_membership_only"],
            )
            container_values = container_features.model_dump(mode="python")
            direct = {
                name: _source_ref("containers", container_id, name)
                for name in (
                    "container_order",
                    "type_category",
                    "verified_gross_mass_value",
                    "verified_gross_mass_unit",
                    "temperature_setpoint_value",
                    "temperature_setpoint_unit",
                )
            }
            derived = {
                name: "derive_container_relation_context_v1"
                for name in set(container_values) - set(direct)
            }
            derived["type_description_surface"] = "normalize_printed_equipment_surface_nfkc_v1"
            container_observations.append(
                ContainerEquipmentObservation(
                    features=container_features,
                    projection=_projection(
                        view_row_key=f"container:{container_id}",
                        document_id=document_id,
                        feature_values=container_values,
                        direct=direct,
                        derived=derived,
                        source_values=source_values,
                    ),
                )
            )

    final_coverage_counts = {value: coverage_counter[value] for value in _COVERAGE_VALUES}
    expected_cargo_group_rows = sum(
        len(by_document.get(document_id, {}).get("cargo_groups", ()))
        for document_id in train_document_ids
    )
    if len(cargo_group_observations) != expected_cargo_group_rows:
        raise ValueError("cargo-group view does not contain exactly one row per source group")
    all_observations: list[ModelingObservation] = []
    all_observations.extend(document_observations)
    all_observations.extend(cargo_observations)
    all_observations.extend(cargo_group_observations)
    all_observations.extend(container_observations)
    direct_cells = {
        (direct.source.table, direct.source.row_key, direct.source.column)
        for observation in all_observations
        for direct in observation.projection.direct
    }
    direct_cells.update(
        (source.table, source.row_key, source.column)
        for observation in cargo_group_observations
        for mass in observation.mass_projections
        for source in (mass.value_source, mass.unit_source)
    )
    direct_cell_count = len(direct_cells)
    return ModelingViews(
        schema_version=2,
        train_document_ids=tuple(train_document_ids),
        package_registry_sha256=package_registry.sha256,
        package_surface_inventory=surface_inventory,
        package_hierarchy=tuple(hierarchy_records),
        document_scenario=tuple(document_observations),
        cargo_package_numeric=tuple(cargo_observations),
        cargo_group_numeric=tuple(cargo_group_observations),
        container_equipment=tuple(container_observations),
        audit=ModelingViewsAudit(
            train_document_count=len(train_document_ids),
            validated_target_count=len(train_document_ids),
            document_scenario_rows=len(document_observations),
            cargo_package_numeric_rows=len(cargo_observations),
            cargo_group_numeric_rows=len(cargo_group_observations),
            container_equipment_rows=len(container_observations),
            package_hierarchy_rows=len(hierarchy_records),
            metadata_only_package_rows=sum(
                row.disposition == "metadata_only" for row in hierarchy_records
            ),
            multi_level_document_count=len(multi_level_documents),
            coverage_counts=final_coverage_counts,
            direct_source_cell_count=direct_cell_count,
            package_registry_entries=len(package_registry.payload.entries),
            package_surface_decisions=surface_inventory.decision_count,
        ),
    )


def modeling_views_sha256(views: ModelingViews) -> str:
    """Canonical content identity for receipts and benchmark caches."""

    return sha256_bytes(canonical_json_bytes(views.model_dump(mode="json")))
