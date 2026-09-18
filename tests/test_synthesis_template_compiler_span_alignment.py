from __future__ import annotations

from dataclasses import replace

import pytest

import document_ocr.synthesis.template_compiler.pipeline as pipeline
from document_ocr.synthesis.template_compiler.host import (
    DraftSourceAlignmentError,
    SpanDraft,
    validate_draft_source_alignment,
)
from document_ocr.synthesis.template_compiler.models import (
    CarrierAssessment,
    CriticAgentOutput,
)
from document_ocr.synthesis.template_compiler.pipeline import (
    _apply_validated_critic_review,
    _draft_inventory,
)


def _draft(raw: str) -> SpanDraft:
    source_text = "ONE THOUSAND, EIGHT HUNDRED"
    start = raw.index(source_text)
    return SpanDraft(
        draft_id="agent_binding_total_words",
        logical_key="agent:cargo:package_quantity:total_words",
        render_mode="deterministic_auxiliary",
        value_kind="integer",
        group_kind="cargo",
        group_key="cargo:totals",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(source_text),
        source_text=source_text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="numeric_surface",
        rationale="Regression fixture for exact source alignment.",
    )


def _review() -> CriticAgentOutput:
    return CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "unowned_private_or_auxiliary_fact",
                    "line_ids": ("L00002",),
                    "evidence": "Printed words are shipment-dependent.",
                    "explanation": "Retain the bounded auxiliary quantity wording.",
                },
            ),
            "additional_bindings": (),
            "rationale": "Apply one bounded local revision.",
        }
    )


def _assessment() -> CarrierAssessment:
    return CarrierAssessment.model_validate(
        {
            "canonical_name": "FIXTURE CARRIER",
            "aliases": (),
            "evidence_occurrences": (
                {
                    "line_start": "L00001",
                    "line_end": "L00001",
                    "source_text": "FIXTURE CARRIER",
                    "occurrence_index": 0,
                },
            ),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact fixture carrier evidence.",
        }
    )


def test_draft_source_alignment_requires_exact_pinned_slice() -> None:
    raw = "FIXTURE CARRIER\nONE THOUSAND, EIGHT HUNDRED\n"
    draft = _draft(raw)

    validate_draft_source_alignment(raw=raw, drafts=(draft,))

    corrupted = replace(draft, char_end=draft.char_end - 1)
    with pytest.raises(
        DraftSourceAlignmentError,
        match=r"agent:cargo:package_quantity:total_words.*does not match pinned OCR",
    ):
        validate_draft_source_alignment(raw=raw, drafts=(corrupted,))


def test_inventory_rejects_misaligned_state_before_occurrence_rendering() -> None:
    raw = "FIXTURE CARRIER\nONE THOUSAND, EIGHT HUNDRED\n"
    corrupted = replace(_draft(raw), source_text="ONE THOUSAND, EIGHT HUNDRED ")

    with pytest.raises(DraftSourceAlignmentError, match="does not match pinned OCR"):
        _draft_inventory(raw, (corrupted,), {"documentPatch": {}})


def test_critic_revision_rejects_normalizer_alignment_corruption_at_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = "FIXTURE CARRIER\nONE THOUSAND, EIGHT HUNDRED\n"
    draft = _draft(raw)

    monkeypatch.setattr(pipeline, "apply_critic_patch", lambda **_kwargs: (draft,))
    monkeypatch.setattr(
        pipeline,
        "normalize_structured_row_locality",
        lambda **kwargs: tuple(kwargs["drafts"]),
    )
    monkeypatch.setattr(
        pipeline,
        "normalize_deterministic_draft_semantics",
        lambda **kwargs: tuple(kwargs["drafts"]),
    )
    monkeypatch.setattr(
        pipeline,
        "normalize_source_boundaries",
        lambda **kwargs: (replace(next(iter(kwargs["drafts"])), char_start=draft.char_start + 1),),
    )

    with pytest.raises(DraftSourceAlignmentError, match="does not match pinned OCR"):
        _apply_validated_critic_review(
            review=_review(),
            raw=raw,
            drafts=(draft,),
            source_target={"documentPatch": {}},
            assessment=_assessment(),
            semantic_only_target_facts=(),
            coherence_constraints=(),
            review_candidates=(),
        )
