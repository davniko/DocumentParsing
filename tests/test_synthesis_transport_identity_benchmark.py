from __future__ import annotations

import pandas as pd
import pytest

from document_ocr.synthesis.sdv_harness import ProposalReceipt
from document_ocr.synthesis.transport_identity import (
    GENERIC_VESSEL_RENDERER,
    TransportIdentityBundle,
    TransportPrivacyPolicy,
    build_transport_identity_bundle,
)
from document_ocr.synthesis.transport_identity_benchmark import (
    TransportBenchmarkSettings,
    TransportIdentityBenchmarkError,
    _identity_connected_group_ids,
    _novelty,
    _ValidityAccumulator,
    fit_selected_vessel_renderer,
    generate_transport_identity_pilot,
    transport_identity_candidate_specs,
)


def _valid_row() -> dict[str, object]:
    return {
        "vessel_structure": "present:2:11:upper:0",
        "voyage_shape": "DDDU",
        "source_imo_present": False,
    }


def test_candidate_specs_are_complete_and_gpu_explicit() -> None:
    candidates = transport_identity_candidate_specs(neural_epochs=300, neural_batch_size=200)

    assert tuple(candidate.name for candidate in candidates) == (
        "empirical",
        "gaussian_copula",
        "ctgan",
        "tvae",
    )
    assert candidates[2].parameters["enable_gpu"] is True
    assert candidates[3].parameters["enable_gpu"] is True
    assert candidates[2].parameters["epochs"] == 300
    assert candidates[3].parameters["epochs"] == 300


def test_settings_reject_non_pac_batch_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        transport_identity_candidate_specs(neural_epochs=10, neural_batch_size=64)


def test_validity_receipt_preserves_violation_taxonomy() -> None:
    valid = _valid_row()
    invalid = {**valid, "vessel_structure": "present:not-an-int:11:upper:0"}
    frame = pd.DataFrame([valid, invalid])
    accumulator = _ValidityAccumulator()

    assert accumulator.accept(frame) == [True, False]
    receipt = accumulator.receipt(
        ProposalReceipt(
            requested_rows=1,
            raw_proposals=2,
            accepted_proposals_before_truncation=1,
            rejected_proposals=1,
            output_rows=1,
            acceptance_yield=0.5,
            bounded_maximum_raw_proposals=2,
        )
    )

    assert receipt.to_dict()["raw_valid_fraction"] == 0.5
    assert receipt.violation_counts == {"invalid_compact_structure": 1}


def test_novelty_is_exact_over_deidentified_structure() -> None:
    first = _valid_row()
    second = {**first, "vessel_structure": "present:2:12:upper:0"}
    train = pd.DataFrame([first])
    synthetic = pd.DataFrame([first, second])

    receipt = _novelty(synthetic, train)

    assert receipt.rows == 2
    assert receipt.novel_rows == 1
    assert receipt.unique_rows == 2


def test_settings_require_all_four_candidates_in_order() -> None:
    candidates = transport_identity_candidate_specs(neural_epochs=10, neural_batch_size=20)
    with pytest.raises(ValueError, match="requires candidates"):
        TransportBenchmarkSettings(
            candidates=candidates[:-1],
            fold_count=3,
            fold_seed=1,
            seeds=(2,),
            proposal_multiplier=8,
            proposal_batch_rows=32,
            quality_margin=0.01,
            stability_penalty=0.25,
            privacy_policy=TransportPrivacyPolicy(
                minimum_normalized_edit_distance=0.2,
                maximum_attempts=64,
            ),
        )


def test_controlled_pilot_generates_every_requested_source_safe_identity() -> None:
    pytest.importorskip("sdv")
    documents = tuple(
        {
            "document_id": f"doc-{index}",
            "transport_vessel_name": f"SOURCE VESSEL {index}",
            "transport_voyage_number": f"AB-{index:03d}",
            "transport_vessel_imo_number": None,
        }
        for index in range(8)
    )
    bundle: TransportIdentityBundle = build_transport_identity_bundle(
        source_documents=documents,
        fit_document_ids=tuple(f"doc-{index}" for index in range(8)),
        template_by_document={f"doc-{index}": f"template-{index}" for index in range(8)},
        partition_by_document={f"doc-{index}": "train" for index in range(8)},
    )
    candidate = transport_identity_candidate_specs(neural_epochs=1, neural_batch_size=10)[0]
    policy = TransportPrivacyPolicy(minimum_normalized_edit_distance=0.2, maximum_attempts=64)

    result = generate_transport_identity_pilot(
        bundle=bundle,
        candidate=candidate,
        requested_rows=5,
        seed=7,
        policy=policy,
        proposal_multiplier=4,
        proposal_batch_rows=32,
        request_id="test-pilot",
        vessel_renderer=GENERIC_VESSEL_RENDERER,
    )

    realization = result["realization"]
    assert realization["rows"] == 5
    assert len(realization["examples"]) == 5
    assert realization["exactStructureFraction"] == 1.0
    assert realization["sourceSafeFraction"] == 1.0
    assert realization["batchUniqueFraction"] == 1.0
    assert all(
        bundle.guard.classify_vessel_name(row["vesselName"], policy=policy) == "none"
        for row in realization["examples"]
    )


def test_identity_connected_groups_prevent_cross_template_name_leakage() -> None:
    documents = (
        {
            "document_id": "doc-a",
            "transport_vessel_name": "REPEATED VESSEL",
            "transport_voyage_number": "A1",
            "transport_vessel_imo_number": None,
        },
        {
            "document_id": "doc-b",
            "transport_vessel_name": "repeated-vessel",
            "transport_voyage_number": "B2",
            "transport_vessel_imo_number": None,
        },
        {
            "document_id": "doc-c",
            "transport_vessel_name": "INDEPENDENT NAME",
            "transport_voyage_number": "C3",
            "transport_vessel_imo_number": None,
        },
    )
    bundle = build_transport_identity_bundle(
        source_documents=documents,
        fit_document_ids=("doc-a", "doc-b", "doc-c"),
        template_by_document={
            "doc-a": "template-a",
            "doc-b": "template-b",
            "doc-c": "template-c",
        },
        partition_by_document={document["document_id"]: "train" for document in documents},
    )

    groups = _identity_connected_group_ids(bundle)

    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_selected_lexical_refit_validates_the_complete_selection_contract() -> None:
    documents = tuple(
        {
            "document_id": f"doc-{index}",
            "transport_vessel_name": f"VESSEL {chr(65 + index // 5)}{chr(65 + index % 5)}",
            "transport_voyage_number": None,
            "transport_vessel_imo_number": None,
        }
        for index in range(25)
    )
    bundle = build_transport_identity_bundle(
        source_documents=documents,
        fit_document_ids=tuple(document["document_id"] for document in documents),
        template_by_document={
            document["document_id"]: f"template-{index}" for index, document in enumerate(documents)
        },
        partition_by_document={document["document_id"]: "train" for document in documents},
    )
    selection = {
        "candidate": "source_character_3gram",
        "order": 3,
        "source": "source",
        "rendererId": "fit_isolated_interpolated_character_3gram_exact_structure_v2",
    }

    renderer = fit_selected_vessel_renderer(bundle=bundle, selection=selection, seed=7)

    assert renderer.order == 3
    assert renderer.renderer_id == selection["rendererId"]
    with pytest.raises(TransportIdentityBenchmarkError, match="internally inconsistent"):
        fit_selected_vessel_renderer(
            bundle=bundle,
            selection={**selection, "source": "faker"},
            seed=7,
        )
    with pytest.raises(TransportIdentityBenchmarkError, match="rendererId"):
        fit_selected_vessel_renderer(
            bundle=bundle,
            selection={**selection, "rendererId": "wrong-renderer"},
            seed=7,
        )
