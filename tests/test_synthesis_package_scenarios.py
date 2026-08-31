from __future__ import annotations

import itertools

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.modeling_views import (
    CargoPackageNumericFeatures,
    CargoPackageNumericObservation,
    PackageHierarchyRecord,
    ViewRowProjection,
)
from document_ocr.synthesis.package_registry import (
    LoadedPackageRegistry,
    PackageRegistryEntry,
    PackageRegistryPayload,
)
from document_ocr.synthesis.package_scenarios import (
    PackageFitObservation,
    build_package_scenario_support,
    package_fit_observations,
    sample_package_type,
)

REGISTRY_SHA256 = "a" * 64
MISMATCH_DOCUMENT_ID = "doc_c0f529a4e76ce0c3259b00c32a479fb6ec35b297ca5eef9a937bc222ed5dd265"
CORE_TOKENS = (
    "PACKAGE_PACKAGE",
    "PACKAGE_CARTON",
    "PACKAGE_PALLET",
    "PACKAGE_INTERMEDIATE_BULK_CONTAINER",
)
ALLOWED_TOKENS = CORE_TOKENS + tuple(f"PACKAGE_TEST_{index:03d}" for index in range(43))


def _registry() -> LoadedPackageRegistry:
    codes = (
        "".join(pair)
        for pair in itertools.product("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", repeat=2)
    )
    tokens = ALLOWED_TOKENS + tuple(
        f"PACKAGE_REGISTRY_{index:03d}" for index in range(405 - len(ALLOWED_TOKENS))
    )
    display_names = {
        "PACKAGE_PACKAGE": "Package",
        "PACKAGE_CARTON": "Carton",
        "PACKAGE_PALLET": "Pallet",
        "PACKAGE_INTERMEDIATE_BULK_CONTAINER": "Intermediate bulk container",
        "PACKAGE_TEST_000": "Carton, fibreboard",
    }
    entries = tuple(
        PackageRegistryEntry(
            categoryToken=token,
            applicationCode=next(codes),
            displayName=display_names.get(token, f"Registry family {index}"),
        )
        for index, token in enumerate(tokens)
    )
    return LoadedPackageRegistry(
        path="/tmp/package-category-registry.json",
        sha256=REGISTRY_SHA256,
        payload=PackageRegistryPayload(
            schemaVersion=1,
            registryKind="package",
            sourceAuthority="MPCI application registry",
            sourceRevision="test-revision",
            sourcePath="application/package-registry",
            sourceSha256="b" * 64,
            entries=entries,
        ),
    )


def _observation(
    document_id: str,
    group_id: str,
    package_id: str,
    *,
    role: str,
    category: str | None,
    surface: str | None,
    disposition: str = "task_facing",
    position: int = 0,
    level_count: int = 1,
    metadata_count: int = 0,
) -> PackageFitObservation:
    return PackageFitObservation(
        source_document_id=document_id,
        group_id=group_id,
        source_package_id=package_id,
        projected_package_id=package_id if disposition == "task_facing" else None,
        source_package_position=position,
        source_package_level_count=level_count,
        metadata_only_package_count=metadata_count,
        role=role,  # type: ignore[arg-type]
        disposition=disposition,  # type: ignore[arg-type]
        quantity=12,
        category_token=category,
        printed_surface=surface,
    )


def _support_fixture() -> tuple[
    tuple[PackageFitObservation, ...], tuple[str, ...], LoadedPackageRegistry
]:
    observations = (
        _observation(
            "doc_direct",
            "g1",
            "p1",
            role="direct_goods",
            category="PACKAGE_CARTON",
            surface="CARTONS",
        ),
        _observation(
            "doc_generic",
            "g1",
            "p1",
            role="generic_aggregate",
            category="PACKAGE_PACKAGE",
            surface="PACKAGES",
        ),
        _observation(
            "doc_outer",
            "g1",
            "p1",
            role="outer_transport",
            category="PACKAGE_PALLET",
            surface="PALLETS",
        ),
        _observation(
            "doc_fallback", "g1", "p1", role="direct_goods", category=None, surface="BALES"
        ),
        _observation("doc_untyped", "g1", "p1", role="direct_goods", category=None, surface=None),
        _observation(
            "doc_metadata",
            "g1",
            "p1",
            role="outer_transport",
            category=None,
            surface="PALLETS",
            disposition="metadata_only",
            metadata_count=1,
        ),
        _observation(
            MISMATCH_DOCUMENT_ID,
            "g1",
            "p1",
            role="direct_goods",
            category="PACKAGE_PACKAGE",
            surface="COLLIES",
            position=0,
            level_count=2,
        ),
        _observation(
            MISMATCH_DOCUMENT_ID,
            "g1",
            "p2",
            role="direct_goods",
            category="PACKAGE_INTERMEDIATE_BULK_CONTAINER",
            surface="IBC",
            position=1,
            level_count=2,
        ),
    )
    fit_ids = tuple(dict.fromkeys(row.source_document_id for row in observations))
    return observations, fit_ids, _registry()


def test_package_support_is_role_conditioned_registry_backed_and_audited() -> None:
    observations, fit_ids, registry = _support_fixture()
    support = build_package_scenario_support(
        observations=observations,
        fit_document_ids=fit_ids,
        allowed_category_tokens=ALLOWED_TOKENS,
        registry=registry,
        expected_registry_sha256=REGISTRY_SHA256,
    )

    assert support.registry_sha256 == REGISTRY_SHA256
    assert support.audit.allowed_category_count == 47
    assert support.audit.input_package_facts == 8
    assert support.audit.task_facing_facts == 7
    assert support.audit.metadata_only_facts == 1
    assert support.audit.resolved_facts == 5
    assert support.audit.fallback_facts == 1
    assert support.audit.untyped_facts == 1
    assert support.audit.excluded_resolved_facts == 1
    assert support.audit.registry_category_count == 405
    assert support.audit.observed_category_count == 4
    assert support.audit.registry_exploration_candidate_count >= 2
    assert set(support.allowed_category_tokens) == set(ALLOWED_TOKENS)
    assert {
        category.category_token for role in support.roles for category in role.categories
    } <= set(ALLOWED_TOKENS)

    assert len(support.exclusions) == 1
    exclusion = support.exclusions[0]
    assert exclusion.source_document_id == MISMATCH_DOCUMENT_ID
    assert exclusion.group_id == "g1"
    assert exclusion.source_package_id == "p1"
    assert exclusion.printed_surface == "COLLIES"
    assert exclusion.reason == "generic_category_conflicts_with_direct_multi_package_role"


def test_package_sampling_preserves_hierarchy_and_defers_resolved_surface() -> None:
    observations, fit_ids, registry = _support_fixture()
    support = build_package_scenario_support(
        observations=observations,
        fit_document_ids=fit_ids,
        allowed_category_tokens=ALLOWED_TOKENS,
        registry=registry,
        expected_registry_sha256=REGISTRY_SHA256,
    )
    source = observations[0]
    sampled = sample_package_type(
        source,
        support=support,
        registry_exploration_permyriad=0,
        stream=DeterministicStream(17, "package-scenario-test", source.source_document_id),
    )

    assert sampled.source_document_id == source.source_document_id
    assert sampled.group_id == source.group_id
    assert sampled.source_package_id == source.source_package_id
    assert sampled.projected_package_id == source.projected_package_id
    assert sampled.source_package_position == source.source_package_position
    assert sampled.source_package_level_count == source.source_package_level_count
    assert sampled.metadata_only_package_count == source.metadata_only_package_count
    assert sampled.role == source.role
    assert sampled.quantity == source.quantity
    assert sampled.resolution == "sampled_category"
    assert sampled.category_token in {"PACKAGE_CARTON", "PACKAGE_INTERMEDIATE_BULK_CONTAINER"}
    assert sampled.application_code == registry.entry(sampled.category_token).applicationCode
    assert sampled.type_description is None
    assert sampled.source_printed_surface == "CARTONS"
    assert sampled.printed_surface_status == "pending_text_realization"


def test_package_registry_exploration_stays_in_authoritative_display_family() -> None:
    observations, fit_ids, registry = _support_fixture()
    support = build_package_scenario_support(
        observations=observations,
        fit_document_ids=fit_ids,
        allowed_category_tokens=ALLOWED_TOKENS,
        registry=registry,
        expected_registry_sha256=REGISTRY_SHA256,
    )
    source = observations[0]

    sampled = sample_package_type(
        source,
        support=support,
        registry_exploration_permyriad=10_000,
        stream=DeterministicStream(29, "package-scenario-test", "registry-family"),
    )

    assert sampled.category_token == "PACKAGE_TEST_000"
    assert sampled.sampling_component == "registry_same_family_exploration"
    assert sampled.application_code == registry.entry("PACKAGE_TEST_000").applicationCode


def test_name_only_fallback_is_resampled_while_absence_and_metadata_are_preserved() -> None:
    observations, fit_ids, registry = _support_fixture()
    support = build_package_scenario_support(
        observations=observations,
        fit_document_ids=fit_ids,
        allowed_category_tokens=ALLOWED_TOKENS,
        registry=registry,
        expected_registry_sha256=REGISTRY_SHA256,
    )
    by_document = {row.source_document_id: row for row in observations}
    fallback = sample_package_type(
        by_document["doc_fallback"],
        support=support,
        registry_exploration_permyriad=0,
        stream=DeterministicStream(19, "package-scenario-test", "doc_fallback"),
    )
    assert fallback.resolution == "sampled_category"
    assert fallback.sampling_component == "fit_empirical"
    assert fallback.category_token in {
        "PACKAGE_CARTON",
        "PACKAGE_INTERMEDIATE_BULK_CONTAINER",
    }
    assert fallback.type_description is None
    assert fallback.source_printed_surface == "BALES"
    assert fallback.printed_surface_status == "pending_text_realization"

    expected = {
        "doc_untyped": ("preserved_untyped", None, "absent"),
        "doc_metadata": ("preserved_metadata", "PALLETS", "preserved_source_surface"),
    }
    for document_id, (resolution, surface, status) in expected.items():
        sampled = sample_package_type(
            by_document[document_id],
            support=support,
            registry_exploration_permyriad=0,
            stream=DeterministicStream(19, "package-scenario-test", document_id),
        )
        assert sampled.resolution == resolution
        assert sampled.category_token is None
        assert sampled.application_code is None
        assert sampled.type_description == surface
        assert sampled.source_printed_surface == surface
        assert sampled.printed_surface_status == status


def test_audited_role_category_mismatch_is_never_silently_sampled() -> None:
    observations, fit_ids, registry = _support_fixture()
    support = build_package_scenario_support(
        observations=observations,
        fit_document_ids=fit_ids,
        allowed_category_tokens=ALLOWED_TOKENS,
        registry=registry,
        expected_registry_sha256=REGISTRY_SHA256,
    )
    mismatch = next(
        row
        for row in observations
        if row.source_document_id == MISMATCH_DOCUMENT_ID and row.source_package_id == "p1"
    )
    with pytest.raises(ValueError, match="excluded by the audited role/category conflict"):
        sample_package_type(
            mismatch,
            support=support,
            registry_exploration_permyriad=0,
            stream=DeterministicStream(23, "package-scenario-test", "mismatch"),
        )


def test_package_support_accepts_any_nonempty_registry_subset_and_rejects_wrong_pin() -> None:
    observations, fit_ids, registry = _support_fixture()
    support = build_package_scenario_support(
        observations=tuple(
            row for row in observations if row.category_token in {None, "PACKAGE_CARTON"}
        ),
        fit_document_ids=fit_ids,
        allowed_category_tokens=("PACKAGE_CARTON",),
        registry=registry,
        expected_registry_sha256=REGISTRY_SHA256,
    )
    assert support.allowed_category_tokens == ("PACKAGE_CARTON",)
    with pytest.raises(ValueError, match="fit-scope receipt"):
        build_package_scenario_support(
            observations=observations,
            fit_document_ids=fit_ids,
            allowed_category_tokens=ALLOWED_TOKENS,
            registry=registry,
            expected_registry_sha256="c" * 64,
        )


def test_package_view_join_preserves_metadata_and_task_hierarchy() -> None:
    hierarchy = (
        PackageHierarchyRecord(
            source_document_id="doc_join",
            group_id="g1",
            source_package_id="p1",
            source_package_position=0,
            role="outer_transport",
            role_source="reviewed_outer_package",
            disposition="metadata_only",
            projected_package_id=None,
            quantity=4,
            type_description="PALLETS",
            normalized_type="PALLETS",
        ),
        PackageHierarchyRecord(
            source_document_id="doc_join",
            group_id="g1",
            source_package_id="p2",
            source_package_position=1,
            role="direct_goods",
            role_source="reviewed_direct_package",
            disposition="task_facing",
            projected_package_id="p1",
            quantity=80,
            type_description="CARTONS",
            normalized_type="CARTONS",
        ),
    )
    package = CargoPackageNumericObservation(
        features=CargoPackageNumericFeatures(
            cargo_group_order=0,
            task_package_position=0,
            task_package_level_count=1,
            source_package_position=1,
            source_package_level_count=2,
            metadata_only_package_count=1,
            package_role="direct_goods",
            type_category="PACKAGE_CARTON",
            printed_surface="CARTONS",
            quantity=80,
            gross_weight_value=None,
            gross_weight_unit=None,
            net_weight_value=None,
            net_weight_unit=None,
            volume_value=None,
            volume_unit=None,
            allocation_coverage="none",
            package_in_allocation_scope=False,
            group_allocation_count=0,
            group_allocation_container_count=0,
            group_allocated_quantity=None,
            package_allocation_count=0,
            package_allocated_quantity=None,
            hs_code_count=0,
            dangerous_goods_record_count=0,
        ),
        projection=ViewRowProjection(
            view_row_key="cargo-package:doc_join:package:p1",
            source_document_id="doc_join",
            direct=(),
            derived=(),
        ),
    )

    joined = package_fit_observations(
        fit_document_ids=("doc_join",), hierarchy=hierarchy, packages=(package,)
    )

    assert len(joined) == 2
    assert joined[0].disposition == "metadata_only"
    assert joined[0].role == "outer_transport"
    assert joined[0].printed_surface == "PALLETS"
    assert joined[1].disposition == "task_facing"
    assert joined[1].role == "direct_goods"
    assert joined[1].category_token == "PACKAGE_CARTON"
    assert joined[1].source_package_position == 1
    assert joined[1].source_package_level_count == 2
    assert joined[1].metadata_only_package_count == 1
