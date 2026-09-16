from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

import document_ocr.synthesis.template_compiler.agents as agents_module
from document_ocr.synthesis.template_compiler.agents import AgentRuntime
from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    normalize_package_quantity_row_locality,
)
from document_ocr.synthesis.template_compiler.models import (
    AgentStageArtifact,
    CriticAgentOutput,
)


def _span_draft(
    raw: str,
    source_text: str,
    logical_key: str,
    target_path: str,
    *,
    occurrence: int = 0,
) -> SpanDraft:
    char_start = -1
    cursor = 0
    for _index in range(occurrence + 1):
        char_start = raw.index(source_text, cursor)
        cursor = char_start + len(source_text)
    return SpanDraft(
        draft_id=f"fixture_{logical_key}_{occurrence}",
        logical_key=logical_key,
        render_mode="target_binding",
        value_kind="integer" if target_path.endswith(".quantity") else "package",
        group_kind="package",
        group_key="package:0",
        target_paths=(target_path,),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=char_start,
        char_end=char_start + len(source_text),
        source_text=source_text,
        evidence_origin="fixture",
        render_policy=(
            "numeric_surface" if target_path.endswith(".quantity") else "categorical_surface"
        ),
        rationale="Regression fixture.",
    )


def test_package_quantity_locality_recognizes_abbreviated_table_headings() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "CONTAINER NO\n"
        "MARKS & NUMBERS\n"
        "NO OF PKGS\n"
        "DESCRIPTION OF GOODS & PACKAGES\n"
        "TOTAL GR. WT. (KGS)\n"
        "VOL (CBM)\n"
        "\n"
        "16000\n"
        "SAID TO CONTAIN / WEIGH & MEASURE\n"
        "16000 BAGS\n"
        "UNRELATED CLAUSE\n"
        "16000\n"
    )
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    category_path = "documentPatch.cargoPackages[0].typeCategory"
    quantity_key = "anchor:" + quantity_path
    drafts = (
        _span_draft(raw, "16000", quantity_key, quantity_path, occurrence=0),
        _span_draft(raw, "16000", quantity_key, quantity_path, occurrence=1),
        _span_draft(raw, "16000", quantity_key, quantity_path, occurrence=2),
        _span_draft(raw, "BAGS", "anchor:" + category_path, category_path),
    )

    normalized = normalize_package_quantity_row_locality(raw=raw, drafts=drafts)

    quantity_starts = {
        row.char_start for row in normalized if row.logical_key == quantity_key
    }
    first = raw.index("16000")
    second = raw.index("16000", first + 1)
    assert quantity_starts == {first, second}


@pytest.mark.asyncio
async def test_hybrid_compact_critic_previews_revision_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixtureCompactOutput(BaseModel):
        value: str

    runtime = object.__new__(AgentRuntime)
    runtime._agent_contract_protocol = "hybrid_reference_partitioned_v10"
    runtime._partition_critic_above_request_bytes = 1_000_000
    runtime._providers = {}
    runtime._critic_substage_cache = {}
    compact_payload = {
        "reviewCandidates": (),
        "remainingRiskCandidates": (),
        "literalLineReviewIds": ("L00001",),
    }
    translated = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "unowned_shipment_fact",
                    "line_ids": ("L00001",),
                    "evidence": "VALUE",
                    "explanation": "The fixture requires host preview.",
                },
            ),
            "additional_bindings": (),
            "rationale": "Preview the compact transaction.",
        }
    )
    previewed: list[CriticAgentOutput] = []
    validator_was_present = False

    monkeypatch.setattr(agents_module, "compact_critic_payload", lambda _payload: compact_payload)
    monkeypatch.setattr(agents_module, "_critic_provider_payload", lambda _payload: {})
    monkeypatch.setattr(agents_module, "_reference_occurrence_rows", lambda _payload: ())
    monkeypatch.setattr(
        agents_module,
        "_compact_inventory_occurrence_lookup",
        lambda _payload: {},
    )
    monkeypatch.setattr(
        agents_module,
        "_scoped_compact_critic_output_type",
        lambda *_args, **_kwargs: FixtureCompactOutput,
    )
    monkeypatch.setattr(
        agents_module,
        "_restore_reference_compact_critic",
        lambda *_args, **_kwargs: translated,
    )

    async def fake_call(**kwargs: object) -> tuple[BaseModel, AgentStageArtifact]:
        nonlocal validator_was_present
        output = FixtureCompactOutput(value="fixture")
        validator = kwargs.get("output_validator")
        validator_was_present = validator is not None
        assert callable(validator)
        assert validator(output) == output
        timestamp = datetime.now(UTC)
        stage = AgentStageArtifact.model_validate(
            {
                "role": "critic",
                "pass_number": 1,
                "started_at": timestamp,
                "completed_at": timestamp,
                "duration_seconds": 0.0,
                "system_prompt_sha256": "a" * 64,
                "user_prompt_sha256": "b" * 64,
                "output_schema_sha256": "c" * 64,
                "status": "success",
                "output": output.model_dump(mode="json"),
                "error_type": None,
                "error_message": None,
                "messages": [],
                "usage": agents_module.empty_usage(),
            }
        )
        return output, stage

    monkeypatch.setattr(runtime, "_call", fake_call)
    output, _stage = await runtime.critic(
        pass_number=1,
        system_prompt="critic",
        payload={"allowedRemovalLogicalKeys": ("agent:value",)},
        retries=1,
        preview_review=previewed.append,
    )

    assert validator_was_present is True
    assert previewed == [translated]
    assert output == translated
