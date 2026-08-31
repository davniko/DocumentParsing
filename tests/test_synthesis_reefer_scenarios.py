from __future__ import annotations

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.reefer_scenarios import (
    ReeferEquipmentIdentity,
    ReeferTemperatureObservation,
    build_reefer_temperature_support,
    sample_generated_equipment_temperature,
    sample_reefer_temperature,
)


def _identity(value: str, *, is_reefer: bool = True) -> ReeferEquipmentIdentity:
    return ReeferEquipmentIdentity(
        value=value,
        authority="preserved_reviewed_surface",
        is_reefer=is_reefer,
    )


def _observations() -> tuple[ReeferTemperatureObservation, ...]:
    reefer = _identity("40 REEF 9'6")
    non_reefer = _identity("40 HIGH CUBE", is_reefer=False)
    return (
        ReeferTemperatureObservation("doc_1", "c1", reefer, -18.0, "celsius"),
        ReeferTemperatureObservation("doc_2", "c1", reefer, -18.0, "celsius"),
        ReeferTemperatureObservation("doc_3", "c1", reefer, -20.0, "celsius"),
        ReeferTemperatureObservation("doc_4", "c1", reefer, None, None),
        ReeferTemperatureObservation("doc_5", "c1", non_reefer, None, None),
    )


def test_reefer_support_is_fit_isolated_exact_identity_and_celsius_only() -> None:
    observations = _observations()
    fit_ids = tuple(row.source_document_id for row in observations)
    support = build_reefer_temperature_support(observations=observations, fit_document_ids=fit_ids)

    assert support.audit.fit_document_count == 5
    assert support.audit.input_observations == 5
    assert support.audit.reefer_observations == 4
    assert support.audit.non_reefer_observations == 1
    assert support.audit.setpoint_present_observations == 3
    assert support.audit.reefer_without_setpoint_observations == 1
    assert support.audit.supported_reefer_identities == 1
    assert len(support.identities) == 1
    identity_support = support.identities[0]
    assert identity_support.identity == _identity("40 REEF 9'6")
    assert identity_support.setpoint_present_count == 3
    assert identity_support.setpoint_absent_count == 1
    assert tuple(
        (row.value, row.occurrences, row.document_count) for row in identity_support.values
    ) == (
        (-20.0, 1, 1),
        (-18.0, 2, 2),
    )
    assert tuple(
        (row.value, row.occurrences, row.document_count) for row in support.pooled_values
    ) == (
        (-20.0, 1, 1),
        (-18.0, 2, 2),
    )


def test_reefer_sampling_is_deterministic_empirical_and_preserves_missingness() -> None:
    observations = _observations()
    fit_ids = tuple(row.source_document_id for row in observations)
    support = build_reefer_temperature_support(observations=observations, fit_document_ids=fit_ids)
    identity = _identity("40 REEF 9'6")
    stream = DeterministicStream(31, "reefer-scenario-test", "synthetic-1")

    first = sample_reefer_temperature(
        identity=identity, source_setpoint_present=True, support=support, stream=stream
    )
    second = sample_reefer_temperature(
        identity=identity, source_setpoint_present=True, support=support, stream=stream
    )
    assert first == second
    assert first.resolution == "sampled_fit_empirical"
    assert first.value in {-20.0, -18.0}
    assert first.unit == "celsius"

    missing = sample_reefer_temperature(
        identity=identity, source_setpoint_present=False, support=support, stream=stream
    )
    assert missing.resolution == "preserved_missingness"
    assert missing.value is None
    assert missing.unit is None


def test_non_reefer_never_receives_a_setpoint() -> None:
    observations = _observations()
    support = build_reefer_temperature_support(
        observations=observations,
        fit_document_ids=tuple(row.source_document_id for row in observations),
    )
    identity = _identity("40 HIGH CUBE", is_reefer=False)
    stream = DeterministicStream(37, "reefer-scenario-test", "non-reefer")

    absent = sample_reefer_temperature(
        identity=identity, source_setpoint_present=False, support=support, stream=stream
    )
    assert absent.resolution == "not_applicable_non_reefer"
    assert absent.value is None
    assert absent.unit is None
    with pytest.raises(ValueError, match="non-reefer"):
        sample_reefer_temperature(
            identity=identity, source_setpoint_present=True, support=support, stream=stream
        )


def test_reefer_support_rejects_fahrenheit_non_reefer_and_cross_split_rows() -> None:
    reefer = _identity("40RF")
    non_reefer = _identity("40GP", is_reefer=False)
    with pytest.raises(ValueError, match="Celsius-only"):
        build_reefer_temperature_support(
            observations=(
                ReeferTemperatureObservation("doc_1", "c1", reefer, -18.0, "fahrenheit"),
            ),
            fit_document_ids=("doc_1",),
        )
    with pytest.raises(ValueError, match="forbidden"):
        build_reefer_temperature_support(
            observations=(
                ReeferTemperatureObservation("doc_1", "c1", non_reefer, -18.0, "celsius"),
            ),
            fit_document_ids=("doc_1",),
        )
    with pytest.raises(ValueError, match="outside the fit partition"):
        build_reefer_temperature_support(
            observations=(ReeferTemperatureObservation("doc_2", "c1", reefer, -18.0, "celsius"),),
            fit_document_ids=("doc_1",),
        )


def test_reefer_sampling_never_aliases_an_unseen_identity() -> None:
    observations = _observations()
    support = build_reefer_temperature_support(
        observations=observations,
        fit_document_ids=tuple(row.source_document_id for row in observations),
    )
    with pytest.raises(ValueError, match="no exact fit-isolated"):
        sample_reefer_temperature(
            identity=ReeferEquipmentIdentity(
                value="40RF",
                authority="authoritative_registry",
                is_reefer=True,
            ),
            source_setpoint_present=True,
            support=support,
            stream=DeterministicStream(41, "reefer-scenario-test", "unseen-alias"),
        )


def test_generated_bic_identity_uses_reviewed_pool_not_old_surface_alias() -> None:
    observations = _observations()
    support = build_reefer_temperature_support(
        observations=observations,
        fit_document_ids=tuple(row.source_document_id for row in observations),
    )
    generated = ReeferEquipmentIdentity(
        value="45R1",
        authority="authoritative_registry",
        is_reefer=True,
    )
    stream = DeterministicStream(43, "reefer-scenario-test", "generated-bic")

    first = sample_generated_equipment_temperature(
        identity=generated,
        source_setpoint_present=True,
        support=support,
        stream=stream,
    )
    second = sample_generated_equipment_temperature(
        identity=generated,
        source_setpoint_present=True,
        support=support,
        stream=stream,
    )
    assert first == second
    assert first.identity == generated
    assert first.resolution == "sampled_fit_empirical_reefer_pool"
    assert first.value in {-20.0, -18.0}
