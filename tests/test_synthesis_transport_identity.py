from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.transport_identity import (
    FICTIONAL_VESSEL_RENDERER_ID,
    TRANSPORT_IDENTITY_COLUMNS,
    SourceTransportIdentityGuard,
    TransportIdentityBundle,
    TransportPrivacyPolicy,
    TransportRealizationError,
    VoyageNumberPolicy,
    build_transport_identity_bundle,
    expand_transport_identity_structure,
    normalized_edit_distance,
    realize_transport_identities,
    realize_voyage_numbers,
    transport_identity_driver_violations,
)


def _document(document_id: str, **values: object) -> dict[str, object]:
    return {
        "document_id": document_id,
        "transport_vessel_name": values.get("vessel"),
        "transport_voyage_number": values.get("voyage"),
        "transport_vessel_imo_number": values.get("imo"),
    }


def _bundle() -> TransportIdentityBundle:
    documents = [
        _document("train-a", vessel="PRIVATE OCEAN", voyage="VX-001", imo="1234567"),
        _document("train-b", vessel="SOURCE MARINER", voyage="AB/92"),
        _document("holdout", vessel="HOLDOUT SECRET", voyage="ZZ-999"),
    ]
    return build_transport_identity_bundle(
        source_documents=documents,
        fit_document_ids=("train-a", "train-b"),
        template_by_document={"train-a": "template-a", "train-b": "template-b"},
        partition_by_document={"train-a": "train", "train-b": "train"},
    )


def test_view_is_fit_only_template_grouped_and_contains_no_raw_identity() -> None:
    bundle = _bundle()
    view = bundle.view

    assert tuple(view.data.columns) == TRANSPORT_IDENTITY_COLUMNS
    assert view.row_ids == ("train-a", "train-b")
    assert view.group_ids == ("template-a", "template-b")
    assert bundle.source_document_count == 3
    assert bundle.fit_document_count == 2
    assert bundle.vessel_renderer_id == FICTIONAL_VESSEL_RENDERER_ID
    serialized = view.data.to_json().casefold()
    for secret in (
        "private ocean",
        "source mariner",
        "holdout secret",
        "vx-001",
        "ab/92",
        "zz-999",
        "1234567",
    ):
        assert secret not in serialized
    assert view.data.loc[0, "voyage_shape"] == "UUHDDD"
    assert view.data.loc[1, "voyage_shape"] == "UUSDD"
    assert view.data.loc[0, "source_imo_present"]
    structure_metadata = view.metadata["tables"][view.name]["columns"]["vessel_structure"]
    assert structure_metadata == {"sdtype": "categorical"}


def test_full_source_guard_rejects_exact_normalized_and_near_matches() -> None:
    bundle = _bundle()
    policy = TransportPrivacyPolicy(minimum_normalized_edit_distance=0.2, maximum_attempts=8)

    assert bundle.guard.classify_vessel_name("holdout-secret", policy=policy) == "exact"
    assert bundle.guard.classify_vessel_name("HOLDOUT SECREX", policy=policy) == "absolute_near"
    assert (
        bundle.guard.classify_vessel_name("XXHOLDOUTSECRETYX", policy=policy) == "substring_overlap"
    )
    assert bundle.guard.classify_vessel_name("UNRELATED FICTION", policy=policy) == "none"
    assert bundle.guard.classify_voyage_number("zz 999", policy=policy) == "exact"
    assert normalized_edit_distance("VX-001", "VX-002") == pytest.approx(1 / 5)
    normalized, absolute = bundle.guard.nearest_vessel_distance("UNRELATED FICTION")
    assert normalized > 0.2
    assert absolute > 0


def test_realization_is_deterministic_source_safe_and_keeps_imo_absent() -> None:
    bundle = _bundle()
    policy = TransportPrivacyPolicy(minimum_normalized_edit_distance=0.2, maximum_attempts=64)
    rows = [
        expand_transport_identity_structure(row)
        for row in bundle.view.data.to_dict(orient="records")
    ]
    stream = DeterministicStream(seed=41, namespace="transport-test", identity="batch")

    first = realize_transport_identities(
        rows=rows,
        row_ids=("synthetic-2", "synthetic-1"),
        stream=stream,
        guard=bundle.guard,
        policy=policy,
    )
    second = realize_transport_identities(
        rows=rows,
        row_ids=("synthetic-2", "synthetic-1"),
        stream=stream,
        guard=bundle.guard,
        policy=policy,
    )

    assert first == second
    assert tuple(item.row_id for item in first) == ("synthetic-1", "synthetic-2")
    assert len({item.vessel_name for item in first}) == 2
    assert len({item.voyage_number for item in first}) == 2
    for item in first:
        assert item.vessel_name is not None
        assert item.voyage_number is not None
        assert item.vessel_imo_number is None
        assert item.imo_policy == "absent_without_authoritative_registry_v1"
        assert item.vessel_renderer_id == FICTIONAL_VESSEL_RENDERER_ID
        assert item.vessel_render_strategy in {
            "generic_vocabulary",
            "generic_compound",
            "phonotactic_generated",
        }
        assert bundle.guard.classify_vessel_name(item.vessel_name, policy=policy) == "none"
        assert bundle.guard.classify_voyage_number(item.voyage_number, policy=policy) == "none"


def test_voyage_only_realization_redraws_each_class_without_vessel_generation() -> None:
    bundle = _bundle()
    rows = (
        {
            "voyage_number_present": True,
            "voyage_shape": "ULHDSDPWO",
            "voyage_letter_count": 2,
            "voyage_digit_count": 2,
            "voyage_separator_count": 5,
        },
        {
            "voyage_number_present": False,
            "voyage_shape": "missing",
            "voyage_letter_count": 0,
            "voyage_digit_count": 0,
            "voyage_separator_count": 0,
        },
    )
    policy = VoyageNumberPolicy(
        minimum_normalized_edit_distance=0.2,
        maximum_attempts=64,
    )

    first = realize_voyage_numbers(
        rows=rows,
        row_ids=("present", "missing"),
        stream=DeterministicStream(seed=83, namespace="voyage-only-test", identity="batch"),
        guard=bundle.guard,
        policy=policy,
    )
    second = realize_voyage_numbers(
        rows=rows,
        row_ids=("present", "missing"),
        stream=DeterministicStream(seed=83, namespace="voyage-only-test", identity="batch"),
        guard=bundle.guard,
        policy=policy,
    )

    assert first == second
    assert tuple(row.row_id for row in first) == ("missing", "present")
    assert first[0].voyage_number is None
    generated = first[1].voyage_number
    assert generated is not None
    assert len(generated) == len("Aa-0/0. _")
    assert generated[0].isupper() and generated[1].islower()
    assert generated[2] == "-"
    assert generated[3].isdigit()
    assert generated[4] == "/"
    assert generated[5].isdigit()
    assert generated[6:] == ". _"
    assert bundle.guard.classify_voyage_number(generated, policy=policy) == "none"


def test_literal_only_voyage_collision_fails_with_immutable_receipt() -> None:
    documents = [
        _document("a", vessel="SOURCE ONE", voyage="---"),
        _document("b", vessel="SOURCE TWO", voyage="A1"),
    ]
    guard = SourceTransportIdentityGuard.from_documents(documents)
    row = {
        "vessel_name_present": True,
        "vessel_word_count": 2,
        "vessel_character_count": 10,
        "vessel_case_style": "upper",
        "vessel_digit_count": 0,
        "voyage_number_present": True,
        "voyage_shape": "HHH",
        "voyage_letter_count": 0,
        "voyage_digit_count": 0,
        "voyage_separator_count": 3,
        "source_imo_present": False,
    }
    policy = TransportPrivacyPolicy(minimum_normalized_edit_distance=0.1, maximum_attempts=3)

    with pytest.raises(TransportRealizationError) as caught:
        realize_transport_identities(
            rows=(row,),
            row_ids=("synthetic",),
            stream=DeterministicStream(seed=1, namespace="transport-test", identity="failure"),
            guard=guard,
            policy=policy,
        )

    receipt = caught.value.receipt
    assert receipt.identity_kind == "voyage_number"
    assert receipt.attempts == 3
    assert "---" not in repr(receipt)
    with pytest.raises(FrozenInstanceError):
        receipt.attempts = 4  # type: ignore[misc]


def test_realization_honors_exact_vessel_character_structure() -> None:
    bundle = _bundle()
    row = {
        **expand_transport_identity_structure(bundle.view.data.to_dict(orient="records")[0]),
        "vessel_word_count": 3,
        "vessel_character_count": 18,
        "vessel_digit_count": 4,
        "vessel_case_style": "upper",
    }
    result = realize_transport_identities(
        rows=(row,),
        row_ids=("synthetic-exact",),
        stream=DeterministicStream(seed=19, namespace="transport-test", identity="exact"),
        guard=bundle.guard,
        policy=TransportPrivacyPolicy(minimum_normalized_edit_distance=0.2, maximum_attempts=64),
    )[0]

    assert result.vessel_name is not None
    assert len(result.vessel_name) == 18
    assert len(result.vessel_name.split()) == 3
    assert sum(character.isdigit() for character in result.vessel_name) == 4
    assert result.vessel_name.isupper()


def test_driver_validator_rejects_inconsistent_character_count() -> None:
    row = expand_transport_identity_structure(_bundle().view.data.to_dict(orient="records")[0])
    row["vessel_word_count"] = 4
    row["vessel_character_count"] = 4

    assert transport_identity_driver_violations(row) == (
        "vessel_character_count_too_small_for_words",
    )


@pytest.mark.parametrize(
    ("distance", "attempts"),
    [(-0.1, 1), (1.0, 1), (float("nan"), 1), (0.2, 0)],
)
def test_privacy_policy_rejects_invalid_bounds(distance: float, attempts: int) -> None:
    with pytest.raises(ValueError):
        TransportPrivacyPolicy(
            minimum_normalized_edit_distance=distance,
            maximum_attempts=attempts,
        )
