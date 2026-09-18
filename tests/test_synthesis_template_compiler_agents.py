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
    CompilerAgentOutput,
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


def _compiler_output_fixture() -> CompilerAgentOutput:
    return CompilerAgentOutput.model_validate(
        {
            "carrier": {
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
                "rationale": "Exact fixture evidence.",
            },
            "anchor_overrides": (),
            "bindings": (),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
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
async def test_hybrid_compiler_records_unprojectable_local_repair_without_repair_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._agent_contract_protocol = "hybrid_reference_partitioned_v10"
    prior = _compiler_output_fixture()

    def reject_projection(*_args: object, **_kwargs: object) -> object:
        raise ValueError("host rejection cannot be safely projected onto a local repair slice")

    monkeypatch.setattr(
        agents_module,
        "build_local_compiler_repair_payload",
        reject_projection,
    )
    system_prompt = "compiler system prompt"

    output, stage = await runtime.compiler(
        pass_number=2,
        system_prompt=system_prompt,
        payload={
            "documentId": "doc_fixture",
            "previousCandidateOutput": prior.model_dump(mode="json"),
            "requiredRevision": "Repair a defect outside the safe local slice.",
        },
        retries=1,
        repair_prompt=None,
    )

    effective_prompt = (
        f"{system_prompt}\n\n{agents_module._DISCRIMINATED_BINDING_ADAPTER}\n\n"
        f"{agents_module._COMPILER_REPAIR_ADAPTER}"
    )
    assert output is None
    assert stage.status == "host_rejected"
    assert stage.error_type == "ValueError"
    assert stage.error_message is not None
    assert "cannot be safely projected" in stage.error_message
    assert stage.system_prompt_sha256 == agents_module.sha256_bytes(
        effective_prompt.encode("utf-8")
    )
    assert stage.usage.requests == 0


@pytest.mark.asyncio
async def test_hybrid_compiler_records_invalid_local_repair_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._agent_contract_protocol = "hybrid_reference_partitioned_v10"
    prior = _compiler_output_fixture()
    repair_payload = {
        "anchorBindings": (),
        "candidateSlice": {
            "removableBindingKeys": (),
            "removableAnchorOverrideIds": (),
            "removableSemanticOnlyTargetPaths": (),
        }
    }
    monkeypatch.setattr(
        agents_module,
        "build_local_compiler_repair_payload",
        lambda *_args, **_kwargs: repair_payload,
    )

    def reject_schema(*_args: object, **_kwargs: object) -> object:
        raise ValueError("compiler repair binding scope is invalid")

    monkeypatch.setattr(
        agents_module,
        "_scoped_compiler_repair_output_type",
        reject_schema,
    )

    output, stage = await runtime.compiler(
        pass_number=2,
        system_prompt="compiler system prompt",
        payload={
            "documentId": "doc_fixture",
            "previousCandidateOutput": prior.model_dump(mode="json"),
            "requiredRevision": "Repair the invalid binding scope.",
        },
        retries=1,
        repair_prompt=None,
    )

    assert output is None
    assert stage.status == "host_rejected"
    assert stage.error_type == "ValueError"
    assert stage.error_message is not None
    assert "local_compiler_repair_schema" in stage.error_message
    assert "binding scope is invalid" in stage.error_message
    assert stage.usage.requests == 0


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
