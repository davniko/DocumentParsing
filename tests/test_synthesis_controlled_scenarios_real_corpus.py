from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.modeling_views import ModelingViews, build_modeling_views
from document_ocr.synthesis.package_registry import LoadedPackageRegistry, load_package_registry
from document_ocr.synthesis.package_scenarios import (
    build_package_scenario_support,
    package_fit_observations,
    sample_package_type,
)
from document_ocr.synthesis.reefer_scenarios import (
    ReeferEquipmentIdentity,
    ReeferTemperatureObservation,
    build_reefer_temperature_support,
    sample_reefer_temperature,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    PROJECT_ROOT
    / "artifacts/kie-training/datasets/mpci-bl-combined1157-task-facing-package-categories-v2"
)
HIERARCHY = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/mpci-bl-combined1157-task-facing-packages-v2/"
    "package-metadata.jsonl"
)
PREPARATION = (
    PROJECT_ROOT / "artifacts/kie-synthesis/mpci-bl-combined1157-synthesis-preparation-v3/tables"
)
REGISTRY = (
    PROJECT_ROOT
    / "artifacts/mpci-ai-schema/categorical-registry-followup/package-category-registry.json"
)
REGISTRY_SHA256 = "2476deb46c3cabf2efe9a00bbb9ad1bf6b527e4adae0e4872d8a63807bbd374f"
MISMATCH_DOCUMENT_ID = "doc_c0f529a4e76ce0c3259b00c32a479fb6ec35b297ca5eef9a937bc222ed5dd265"
MISSING_REEFER_IDENTITY_DOCUMENT_ID = (
    "doc_243706979d0c12ffcdcd311dc31d417d9baf5c738232797cfef21ec20ade8bab"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def corpus() -> tuple[ModelingViews, tuple[str, ...], frozenset[str], LoadedPackageRegistry]:
    records = _jsonl(DATASET / "records.jsonl")
    category_metadata = _jsonl(DATASET / "category-metadata.jsonl")
    document_ids = tuple(row["documentId"] for row in records)
    registry = load_package_registry(
        REGISTRY,
        expected_sha256=REGISTRY_SHA256,
        expected_entries=405,
    )
    views = build_modeling_views(
        tables={name: _jsonl(PREPARATION / f"{name}.jsonl") for name in ADAPTER.table_order},
        train_document_ids=document_ids,
        package_registry=registry,
        category_metadata_rows=category_metadata,
        package_hierarchy_metadata_rows=_jsonl(HIERARCHY),
    )
    categories = frozenset(
        decision["categoryToken"]
        for row in category_metadata
        for decision in row["packageDecisions"]
        if decision["categoryToken"] is not None
    )
    return views, document_ids, categories, registry


def test_real_corpus_package_support_and_deterministic_pilot50(
    corpus: tuple[ModelingViews, tuple[str, ...], frozenset[str], LoadedPackageRegistry],
) -> None:
    views, document_ids, categories, registry = corpus
    observations = package_fit_observations(
        fit_document_ids=document_ids,
        hierarchy=views.package_hierarchy,
        packages=views.cargo_package_numeric,
    )
    support = build_package_scenario_support(
        observations=observations,
        fit_document_ids=document_ids,
        allowed_category_tokens=categories,
        registry=registry,
        expected_registry_sha256=REGISTRY_SHA256,
    )

    assert support.audit.fit_document_count == 1157
    assert support.audit.input_package_facts == 1612
    assert support.audit.task_facing_facts == 1423
    assert support.audit.metadata_only_facts == 189
    assert support.audit.resolved_facts == 1348
    assert support.audit.fallback_facts == 57
    assert support.audit.untyped_facts == 18
    assert support.audit.allowed_category_count == 47
    assert support.audit.observed_category_count == 47
    assert support.audit.excluded_resolved_facts == 1
    assert len(support.exclusions) == 1
    assert support.exclusions[0].source_document_id == MISMATCH_DOCUMENT_ID
    assert support.exclusions[0].printed_surface == "COLLIES"

    eligible = [
        row
        for row in observations
        if row.disposition == "task_facing"
        and row.category_token is not None
        and row.identity not in support.excluded_identities
    ][:50]
    sampled = tuple(
        sample_package_type(
            row,
            support=support,
            registry_exploration_permyriad=0,
            stream=DeterministicStream(
                20260831,
                "real-package50",
                f"{index}:{row.source_document_id}:{row.source_package_id}",
            ),
        )
        for index, row in enumerate(eligible)
    )
    assert Counter(row.role for row in eligible) == {
        "direct_goods": 27,
        "generic_aggregate": 14,
        "outer_transport": 9,
    }
    assert Counter(row.category_token for row in sampled) == {
        "PACKAGE_PACKAGE": 15,
        "PACKAGE_CARTON": 9,
        "PACKAGE_PALLET": 9,
        "PACKAGE_BAG": 3,
        "PACKAGE_CASE": 3,
        "PACKAGE_BOX": 2,
        "PACKAGE_DRUM": 2,
        "PACKAGE_ROLL": 2,
        "PACKAGE_OCTABIN": 1,
        "PACKAGE_PAIL": 1,
        "PACKAGE_PIECE": 1,
        "PACKAGE_UNIT": 1,
        "PACKAGE_VEHICLE": 1,
    }
    assert all(
        row.type_description is None and row.printed_surface_status == "pending_text_realization"
        for row in sampled
    )


def test_real_corpus_reefer_support_and_deterministic_pilot50(
    corpus: tuple[ModelingViews, tuple[str, ...], frozenset[str], LoadedPackageRegistry],
) -> None:
    views, document_ids, _, _ = corpus
    observations: list[ReeferTemperatureObservation] = []
    missing_identity_documents: list[str] = []
    for row in views.container_equipment:
        features = row.features
        if features.temperature_setpoint_value is None:
            continue
        if features.type_description_surface is None:
            missing_identity_documents.append(row.projection.source_document_id)
            continue
        assert features.temperature_setpoint_unit is not None
        observations.append(
            ReeferTemperatureObservation(
                source_document_id=row.projection.source_document_id,
                equipment_row_id=row.projection.view_row_key,
                identity=ReeferEquipmentIdentity(
                    value=features.type_description_surface,
                    authority="preserved_reviewed_surface",
                    is_reefer=True,
                ),
                setpoint_value=float(features.temperature_setpoint_value),
                setpoint_unit=features.temperature_setpoint_unit,
            )
        )

    assert missing_identity_documents == [MISSING_REEFER_IDENTITY_DOCUMENT_ID]
    support = build_reefer_temperature_support(
        observations=observations,
        fit_document_ids=document_ids,
    )
    assert support.audit.input_observations == 52
    assert support.audit.reefer_observations == 52
    assert support.audit.setpoint_present_observations == 52
    assert support.audit.supported_reefer_identities == 18
    sampled = tuple(
        sample_reefer_temperature(
            identity=row.identity,
            source_setpoint_present=True,
            support=support,
            stream=DeterministicStream(
                20260831,
                "real-reefer50",
                f"{index}:{row.source_document_id}:{row.equipment_row_id}",
            ),
        )
        for index, row in enumerate(observations[:50])
    )
    assert Counter(row.value for row in sampled) == {
        -24.0: 1,
        -22.0: 4,
        -21.0: 3,
        -20.0: 5,
        -19.0: 1,
        -18.0: 20,
        -3.0: 1,
        0.0: 2,
        1.0: 3,
        2.0: 1,
        3.0: 1,
        5.0: 4,
        5.5: 2,
        14.0: 1,
        19.0: 1,
    }
    assert all(row.unit == "celsius" for row in sampled)
    assert all(
        row.value
        in {
            value.value
            for identity in support.identities
            if identity.identity.key == row.identity.key
            for value in identity.values
        }
        for row in sampled
    )
