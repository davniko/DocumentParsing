from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import document_ocr.synthesis.raw_text_pipeline as raw_text_pipeline
import document_ocr.synthesis.template_compiler.descendant as descendant
from document_ocr.hashing import sha256_bytes, sha256_file
from document_ocr.synthesis.config import load_synthesis_raw_text_pipeline_config
from document_ocr.synthesis.raw_text_pipeline import (
    _validate_pipeline_config_file,
    preflight_raw_text_pipeline,
    run_raw_text_pipeline,
)
from document_ocr.synthesis.template_compiler.agents import (
    _compiler_repair_error_intersects_scope,
    empty_usage,
)
from document_ocr.synthesis.template_compiler.coherence import (
    inclusive_range_cardinalities,
    validate_render_coherence,
)
from document_ocr.synthesis.template_compiler.descendant import (
    BindingOutput,
    RenderPlan,
    _build_initial_plan,
    _materialize_case,
)
from document_ocr.synthesis.template_compiler.descendant_models import (
    DescendantConfig,
    ResidualStageReceipt,
)
from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    all_risk_candidates,
    reconcile_draft_overlaps,
)
from document_ocr.synthesis.template_compiler.models import (
    AgentStageArtifact,
    CompilerAgentOutput,
    CriticAgentOutput,
    ExtractionConfig,
    InclusiveRangeConstraint,
)
from document_ocr.synthesis.template_compiler.pipeline import (
    _compilation_acceptance_gate_passed,
    _compiler_occurrence_candidates,
    _draft_inventory,
    _extract_case,
    _require_exact_resume_prefix,
    _resume_contract,
)
from document_ocr.synthesis.template_compiler.pipeline import (
    load_config as load_compilation_config,
)
from document_ocr.synthesis.template_compiler.staged_contract import (
    StagedFacetAuditOutput,
    normalize_staged_facet_audit,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COMPILATION_CONFIG = (
    _PROJECT_ROOT / "configs/synthesis/mpci_bl_template_compilation200_v1_luna.yaml"
)
_PIPELINE_CONFIG = (
    _PROJECT_ROOT / "configs/synthesis/mpci_bl_compiled_raw_text_pipeline30_validation_v1.yaml"
)
_QUANTITY = "documentPatch.cargoPackages[0].quantity"


def test_numeric_derivation_filter_excludes_non_numeric_package_fields() -> None:
    leaves = descendant._numeric_leaves(
        [{"quantity": 1, "typeCategory": "PACKAGE_CRATE"}],
        names=frozenset({"quantity", "packageQuantity"}),
    )

    assert leaves == (Decimal(1),)


@pytest.mark.parametrize(
    ("surface", "expected"),
    (("TWO", 2), ("THREE THOUSAND FOUR HUNDRED", 3400), ("TWO PALLETS", None)),
)
def test_complete_number_word_parser_is_strict(surface: str, expected: int | None) -> None:
    assert descendant._number_word_value(surface) == expected


def test_identifier_renderer_transfers_a_unique_certified_ocr_omission() -> None:
    binding = SimpleNamespace(
        value_kind="identifier",
        target_paths=("documentPatch.transport.voyageNumber",),
        occurrences=(
            SimpleNamespace(slot_id="slot_short", source_text="0INFRW1MA"),
            SimpleNamespace(slot_id="slot_full", source_text="01INFRW1MA"),
        ),
        realization=SimpleNamespace(
            target_values=(
                SimpleNamespace(
                    target_path="documentPatch.transport.voyageNumber",
                    source_value="01INFRW1MA",
                ),
            )
        ),
    )
    source_target = {
        "documentPatch": {"transport": {"voyageNumber": "01INFRW1MA"}}
    }
    target = {"documentPatch": {"transport": {"voyageNumber": "62RBJTY9TN"}}}

    output = descendant._render_agent_target_binding(
        binding,
        source_target=source_target,
        target=target,
    )

    assert output.canonical_value == "62RBJTY9TN"
    assert output.replacements == {
        "slot_short": "6RBJTY9TN",
        "slot_full": "62RBJTY9TN",
    }


def test_identifier_renderer_rejects_an_ambiguous_ocr_omission() -> None:
    with pytest.raises(ValueError, match="absent or ambiguous"):
        descendant._unique_subsequence_indices("AAB", "AB")


def test_unchanged_static_target_is_accepted_from_certified_source_surfaces() -> None:
    address = "Unit 3208, 32/F, The Octagon, 6 Sha Tsui Road, N.T."
    occurrences = (
        SimpleNamespace(slot_id="slot_one", source_text=address.removesuffix(" N.T.")),
        SimpleNamespace(slot_id="slot_two", source_text=address.removesuffix(", N.T.")),
    )
    binding = SimpleNamespace(
        logical_key="anchor:documentPatch.parties.carrier.address",
        value_kind="address",
        target_paths=("documentPatch.parties.carrier.address",),
        dependency_paths=(),
        occurrences=occurrences,
        derivation=None,
        realization=SimpleNamespace(
            requires_agent=False,
            mode="static",
            target_values=(
                SimpleNamespace(
                    target_path="documentPatch.parties.carrier.address",
                    source_value=address,
                ),
            ),
        ),
    )
    target = {"documentPatch": {"parties": {"carrier": {"address": address}}}}
    case = SimpleNamespace(
        source_target=target,
        target=target,
        template=SimpleNamespace(bindings=(binding,), coherence_constraints=()),
    )
    outputs = {
        binding.logical_key: BindingOutput(
            {slot.slot_id: slot.source_text for slot in occurrences},
            address,
        )
    }

    assert descendant._target_binding_semantics_valid(case=case, outputs=outputs) == (
        True,
        (),
    )


def test_container_package_count_can_follow_one_printed_package_quantity() -> None:
    source = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [
                {"quantity": 6},
                {"quantity": 1},
                {"quantity": 4},
            ],
        }
    }
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [
                {"quantity": 9},
                {"quantity": 1},
                {"quantity": 4},
            ],
        }
    }

    assert descendant._container_package_count_value(
        source_target=source,
        target=target,
        source_surface="6",
    ) == (6, 9)


def test_same_as_identifier_preserves_the_derived_literal_frame() -> None:
    dependency = SimpleNamespace(occurrences=(SimpleNamespace(source_text="84899112"),))
    binding = SimpleNamespace(
        value_kind="identifier",
        occurrences=(SimpleNamespace(slot_id="slot_exporter", source_text="TW-02-84899112"),),
    )

    output = descendant._render_same_as_binding(
        binding,
        dependency,
        BindingOutput({"slot_phone": "94807251"}, "94807251"),
    )

    assert output.replacements == {"slot_exporter": "TW-02-94807251"}


def test_country_code_vocabulary_covers_common_document_aliases() -> None:
    countries = descendant._country_code_map(
        _PROJECT_ROOT / "data/registries/countries/iso-codes-4.9.0-1/iso_3166-1.json"
    )

    assert countries["uae"] == "AE"
    assert countries["prchina"] == "CN"


def test_source_only_date_renderer_preserves_mixed_numeric_separators() -> None:
    binding = SimpleNamespace(
        logical_key="agent:detail_mfg_1a",
        occurrences=(
            SimpleNamespace(
                slot_id="slot_mixed_date",
                source_text="27.03/2024",
                render_policy="date_surface",
            ),
        ),
    )

    output = descendant._render_pattern_date_auxiliary(
        binding,
        descendant.DeterministicStream(20260912, "test", "mixed-date"),
    )
    rendered = output.replacements["slot_mixed_date"]

    assert len(rendered) == len("27.03/2024")
    assert rendered[2] == "."
    assert rendered[5] == "/"
    assert (rendered[:2] + rendered[3:5] + rendered[6:]).isdigit()


def test_production_resume_contract_is_json_round_trip_stable() -> None:
    contract = _resume_contract(_PROJECT_ROOT)

    assert json.loads(json.dumps(contract)) == contract


def test_compiler_repair_inventory_preserves_overlapping_exact_occurrence_index() -> None:
    raw = "--- PAGE 1 ---\nPG III, (25C.C.C.)\n"
    source_text = "C.C."
    first_start = raw.index(source_text)
    second_start = raw.index(source_text, first_start + 1)
    draft = SpanDraft(
        draft_id="agent_binding_flash_point_method",
        logical_key="agent:dangerous_goods:flash_point_method",
        render_mode="deterministic_auxiliary",
        value_kind="dangerous_goods",
        group_kind="dangerous_goods",
        group_key="dangerous_goods:0",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=second_start,
        char_end=second_start + len(source_text),
        source_text=source_text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="exact_surface",
        rationale="Fixture overlapping occurrence.",
    )

    inventory = _draft_inventory(raw, (draft,), {"documentPatch": {}})
    occurrence = inventory[0]["occurrences"][0]

    assert occurrence["occurrenceIndex"] == 1
    assert occurrence["exactMatchCount"] == 2
    assert [row["selected"] for row in occurrence["exactMatchCandidates"]] == [False, True]


def test_production_compiler_candidate_preserves_overlapping_occurrence_index() -> None:
    raw = "--- PAGE 1 ---\nPG III, (25C.C.C.)\n"
    surface = "C.C."
    risk = SimpleNamespace(source_text=surface)

    candidates = _compiler_occurrence_candidates(
        raw=raw,
        source_target={"documentPatch": {}},
        anchors=(),
        risks=(risk,),
    )

    selected = tuple(row for row in candidates if row["sourceText"] == surface)
    # The leading match ends inside the larger token and is deliberately not exposed. The safe
    # trailing match must nevertheless retain its true overlapping occurrence index.
    assert tuple(row["occurrenceIndex"] for row in selected) == (1,)


def test_production_compiler_preserves_uri_scheme_inside_agent_residual() -> None:
    raw = "--- PAGE 1 ---\nhttp://p.o.no:4512631286/]P.O.NO:4512631286\n"
    surface = "http://p.o.no:4512631286/]P.O.NO:4512631286"
    start = raw.index(surface)
    draft = SpanDraft(
        draft_id="agent_binding_booking_reference",
        logical_key="agent:booking_reference",
        render_mode="agent_residual",
        value_kind="operational_text",
        group_kind="document",
        group_key="document:references",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(surface),
        source_text=surface,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="URI-bearing operational reference fixture.",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(draft,),
        source_target={"documentPatch": {}},
    )

    assert len(normalized) == 1
    assert normalized[0].source_text == surface
    assert normalized[0].char_start == start


def _write_descendant_config(tmp_path: Path, config: DescendantConfig) -> Path:
    path = tmp_path / "compiled-pipeline.yaml"
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _coherence_binding() -> SimpleNamespace:
    return SimpleNamespace(
        binding_id="binding_0001",
        logical_key="agent:package_range",
        render_mode="agent_residual",
        value_kind="operational_text",
        group_kind="package",
        group_key="package:detail:range",
        target_paths=(),
        dependency_paths=(),
        dependency_bindings=(),
        derivation=None,
        occurrences=(
            SimpleNamespace(
                slot_id="slot_range",
                source_text="PACKAGE 1-7",
                render_policy="natural_text",
            ),
        ),
        realization=SimpleNamespace(
            mode="agent_required",
            requires_agent=True,
            target_values=(),
        ),
        source_relationships=(),
    )


def _range_constraint() -> InclusiveRangeConstraint:
    return InclusiveRangeConstraint(
        constraint_id="coherence_constraint_0123456789abcdef",
        candidate_fingerprint="a" * 64,
        kind="inclusive_range_cardinality",
        member_logical_keys=("agent:package_range",),
        dependency_paths=(_QUANTITY,),
        rationale="The printed inclusive package range expresses the package quantity.",
    )


def _target(quantity: int) -> dict[str, Any]:
    return {
        "documentPatch": {
            "cargoPackages": [
                {"packageId": "package-1", "groupId": "cargo-1", "quantity": quantity}
            ]
        }
    }


def _empty_stage() -> ResidualStageReceipt:
    return ResidualStageReceipt.model_validate(
        {
            "schema_version": 1,
            "document_id": "doc_test",
            "status": "not_required",
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0.0,
            "system_prompt_sha256": "b" * 64,
            "user_prompt_sha256": None,
            "output_schema_sha256": None,
            "output": None,
            "error_type": None,
            "error_message": None,
            "messages": [],
            "usage": {
                "requests": 0,
                "inputTokens": 0,
                "outputTokens": 0,
                "reasoningTokens": 0,
                "estimatedCostUsd": "0",
                "providerReportedCostUsd": None,
            },
        }
    )


def test_production_configs_expose_only_the_compiled_template_path() -> None:
    compilation = load_compilation_config(_COMPILATION_CONFIG)
    pipeline = load_synthesis_raw_text_pipeline_config(_PIPELINE_CONFIG)

    assert compilation.task == "bill_of_lading_raw_text_template_compilation_v1"
    assert compilation.workflow.documents == 200
    assert compilation.workflow.max_concurrent_documents == 8
    assert compilation.workflow.max_concurrent_requests == 8
    assert compilation.workflow.require_no_rejected_documents is True
    assert pipeline.task == "bill_of_lading_compiled_raw_text_pipeline_v1"
    assert pipeline.workflow.documents == 30
    assert pipeline.workflow.max_requests_per_document == 1
    assert pipeline.workflow.provider_launch_authorized is False
    assert pipeline.workflow.publish_training_records is True
    assert (
        pipeline.inputs.template_run.path
        == "artifacts/kie-synthesis/mpci-bl-template-compilation200-production-v4-luna-high"
    )
    assert (
        pipeline.inputs.template_run.commit_sha256
        == "5b35f5a35f94634c0a8d2ee0ea84e40092b30cf41df393e62d3133e5f320a5d6"
    )
    assert pipeline.inputs.residual_replay_run is None


def test_promoted_compiler_host_matches_the_validated_experiment_host() -> None:
    experiment_host = (
        _PROJECT_ROOT
        / "experiments/kie-synthesis/raw-text-template-v1/src/raw_text_template_experiment/host.py"
    )
    production_host = _PROJECT_ROOT / "src/document_ocr/synthesis/template_compiler/host.py"

    assert experiment_host.read_bytes() == production_host.read_bytes()


def test_production_compiler_detects_multiline_relational_package_range() -> None:
    raw = "--- PAGE 1 ---\nPACKAGE NO : 1\n- 18\n"
    target = {"documentPatch": {"cargoPackages": [{"quantity": 18}]}}

    relational = tuple(
        row
        for row in all_risk_candidates(raw, (), target)
        if row.kind == "relational_numeric_range"
    )

    assert tuple(row.source_text for row in relational) == ("1\n- 18",)
    assert inclusive_range_cardinalities(relational[0].source_text) == (18,)


def test_production_compiler_local_repair_does_not_retry_mixed_scope_failure() -> None:
    selected = "documentPatch.cargoGroups[4].hsCodes[1]"
    payload = {
        "candidateSlice": {
            "removableBindingKeys": ("hs_code_4_1",),
            "removableAnchorOverrideIds": (),
            "removableSemanticOnlyTargetPaths": (),
            "bindings": (
                {
                    "logical_key": "hs_code_4_1",
                    "target_paths": (selected,),
                    "dependency_paths": (),
                },
            ),
        }
    }

    assert not _compiler_repair_error_intersects_scope(
        ValueError(f"selected {selected} exposes latent documentPatch.cargoPackages[9].quantity"),
        payload,
    )


def test_production_facet_normalizes_omitted_host_required_candidate() -> None:
    payload = {
        "auditFacet": {"name": "surface_completeness"},
        "candidateRows": (
            {
                "candidateIndex": 0,
                "candidateId": "risk_0001",
                "kind": "private_identifier",
                "requiredRevision": True,
                "bindingIndexes": (),
                "lineIds": ("L00001",),
                "sourceTexts": ("987654",),
                "context": None,
                "details": {},
            },
        ),
        "literalLineRanges": ("L00001",),
        "bindings": (),
        "occurrences": (),
        "targetFacts": (),
        "annotatedSource": "L00001 | VAT 987654",
    }
    output = StagedFacetAuditOutput.model_validate(
        {
            "facet": "surface_completeness",
            "verdict": "pass",
            "findings": (),
            "candidate_dispositions": (
                {"candidate_index": 0, "conclusion": "valid_current_state"},
            ),
            "coverage": {
                "assigned_bindings_checked": True,
                "assigned_candidates_checked": True,
                "assigned_literal_lines_checked": True,
                "assigned_target_relationships_checked": True,
            },
            "rationale": "The candidate was accidentally omitted.",
        }
    )

    normalized = normalize_staged_facet_audit(output, payload)

    assert normalized.verdict == "revise"
    assert normalized.findings[0].candidate_indexes == (0,)
    assert normalized.candidate_dispositions[0].conclusion == "defect_requires_revision"


def test_compilation_config_rejects_the_removed_all_certified_contract() -> None:
    payload = yaml.safe_load(_COMPILATION_CONFIG.read_text(encoding="utf-8"))
    payload["workflow"]["require_all_documents_certified"] = payload["workflow"].pop(
        "require_no_rejected_documents"
    )

    with pytest.raises(ValidationError, match="require_no_rejected_documents"):
        ExtractionConfig.model_validate_json(json.dumps(payload, default=str))


def test_compilation_acceptance_allows_explicit_review_but_never_rejection() -> None:
    assert _compilation_acceptance_gate_passed(
        (SimpleNamespace(status="certified"), SimpleNamespace(status="review_required"))
    )
    assert not _compilation_acceptance_gate_passed(
        (SimpleNamespace(status="certified"), SimpleNamespace(status="rejected"))
    )


def test_compilation_resume_accepts_only_an_exact_nonempty_selection_prefix() -> None:
    first = (1, "doc_first", "a" * 64)
    second = (2, "doc_second", "b" * 64)
    current = (first, second)

    _require_exact_resume_prefix(prior_identity=(first,), current_identity=current)
    _require_exact_resume_prefix(prior_identity=current, current_identity=current)

    for invalid in ((), (second,), (first, (2, "doc_changed", "b" * 64))):
        with pytest.raises(ValueError, match="exact prefix"):
            _require_exact_resume_prefix(
                prior_identity=invalid,
                current_identity=current,
            )


def _successful_agent_stage(
    role: str,
    pass_number: int,
    output: CompilerAgentOutput | CriticAgentOutput,
) -> AgentStageArtifact:
    now = datetime.now(UTC)
    return AgentStageArtifact.model_validate(
        {
            "role": role,
            "pass_number": pass_number,
            "started_at": now,
            "completed_at": now,
            "duration_seconds": 0.0,
            "system_prompt_sha256": "0" * 64,
            "user_prompt_sha256": "1" * 64,
            "output_schema_sha256": "2" * 64,
            "status": "success",
            "output": output.model_dump(mode="json"),
            "error_type": None,
            "error_message": None,
            "messages": [],
            "usage": empty_usage(),
        }
    )


class _SuccessfulProductionRuntime:
    def __init__(
        self,
        compiler: CompilerAgentOutput,
        critic: CriticAgentOutput,
    ) -> None:
        self.compiler_output = compiler
        self.critic_output = critic
        self.compiler_calls = 0
        self.critic_calls = 0

    async def compiler(self, **kwargs: Any):
        self.compiler_calls += 1
        return self.compiler_output, _successful_agent_stage(
            "compiler", kwargs["pass_number"], self.compiler_output
        )

    async def critic(self, **kwargs: Any):
        self.critic_calls += 1
        return self.critic_output, _successful_agent_stage(
            "critic", kwargs["pass_number"], self.critic_output
        )


class _CapturingStagedRun:
    def __init__(self, root: Path) -> None:
        self.stage_root = root
        self.published: dict[str, Any] = {}

    def publish_bytes(self, relative_path: str, payload: bytes) -> None:
        self.published[relative_path] = payload

    def publish_json(self, relative_path: str, payload: Any) -> None:
        self.published[relative_path] = payload


@pytest.mark.asyncio
async def test_promoted_compiler_executes_a_successful_certification_transaction(
    tmp_path: Path,
) -> None:
    document_id = "doc_production_transaction"
    heading = "BILL OF LADING"
    carrier = "Acme Ocean Lines Ltd."
    body = f"{heading}\n{carrier}\n"
    raw = "--- PAGE 1 ---\n" + body
    source_target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"parties": {"carrier": {"name": carrier}}},
    }
    source = {
        "documentId": document_id,
        "joinedRawText": raw,
        "joinedRawTextSha256": sha256_bytes(raw.encode("utf-8")),
        "target": source_target,
    }
    anchor_rows = (
        {
            "anchor_id": "anchor_carrier",
            "document_id": document_id,
            "patchable": True,
            "page_number": 1,
            "page_start": len(heading) + 1,
            "page_end": len(heading) + 1 + len(carrier),
            "raw_value": carrier,
            "target_value": carrier,
            "relation_target_path": "documentPatch.parties.carrier.name",
            "role_path": "documentPatch.parties.carrier.name",
            "surface_family": "text",
        },
    )
    feature = {
        "document_type": "bill_of_lading",
        "template_proxy_id": "template_production_transaction",
        "page_count": 1,
        "ocr_lines": len(raw.splitlines()),
        "ocr_characters": len(raw),
        "container_count": 0,
        "seal_count": 0,
        "goods_group_count": 0,
        "package_fact_count": 0,
        "allocation_group_count": 0,
        "allocation_row_count": 0,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
        "additional_information_group_count": 0,
        "marks_group_count": 0,
        "target_leaf_paths": ["parties.carrier.name"],
        "carrier_family": "ACME",
    }
    compiler = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": carrier,
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00003",
                        "line_end": "L00003",
                        "source_text": carrier,
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact carrier evidence.",
            },
            "anchor_overrides": (),
            "bindings": (),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
    )
    critic = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "Every shipment-specific source surface is owned.",
        }
    )
    runtime = _SuccessfulProductionRuntime(compiler, critic)
    base = load_compilation_config(
        _PROJECT_ROOT / "configs/synthesis/mpci_bl_template_compilation15_hard_canary_v1_luna.yaml"
    )
    config = base.model_copy(
        update={
            "resume_from": None,
            "workflow": base.workflow.model_copy(
                update={
                    "documents": 1,
                    "additional_call_launch_threshold_usd_per_document": Decimal("1"),
                }
            ),
        }
    )
    staged = _CapturingStagedRun(tmp_path)

    result, template, masked = await _extract_case(
        ordinal=1,
        document_id=document_id,
        source=source,
        feature=feature,
        anchor_rows=anchor_rows,
        runtime=runtime,  # type: ignore[arg-type]
        compiler_prompt="compiler",
        critic_prompt="critic",
        config=config,
        staged=staged,  # type: ignore[arg-type]
        document_limiter=asyncio.Semaphore(1),
    )

    assert result.status == "certified", result.rejection_reasons
    assert template is not None
    assert masked != raw
    assert runtime.compiler_calls == 1
    assert runtime.critic_calls == 1
    assert result.template_sha256 == sha256_bytes(
        staged.published[f"cases/{document_id}/template.json"]
    )
    assert {
        f"cases/{document_id}/source.txt",
        f"cases/{document_id}/source-label.json",
        f"cases/{document_id}/masked-template.txt",
        f"cases/{document_id}/template.json",
        f"cases/{document_id}/catalog-row.json",
        f"cases/{document_id}/state-checkpoint.json",
        f"cases/{document_id}/result.json",
    }.issubset(staged.published)


def test_pipeline_config_object_must_match_its_real_file(tmp_path: Path) -> None:
    config = load_synthesis_raw_text_pipeline_config(_PIPELINE_CONFIG)
    path = _write_descendant_config(tmp_path, config)

    assert _validate_pipeline_config_file(config_path=path, config=config) == config
    changed = config.model_copy(update={"run_name": "different-run"})
    with pytest.raises(ValueError, match="differs from its source file"):
        _validate_pipeline_config_file(config_path=path, config=changed)


def test_pipeline_preflight_validates_root_and_delegates_exact_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_synthesis_raw_text_pipeline_config(_PIPELINE_CONFIG)
    observed: list[Path] = []
    monkeypatch.setattr(raw_text_pipeline, "project_root_from_config", lambda _path: _PROJECT_ROOT)
    monkeypatch.setattr(
        raw_text_pipeline,
        "preflight_descendants",
        lambda path: observed.append(path) or {"documents": 30, "plannedProviderRequests": 0},
    )

    result = preflight_raw_text_pipeline(
        project_root=_PROJECT_ROOT,
        config_path=_PIPELINE_CONFIG,
        config=config,
    )

    assert result == {"documents": 30, "plannedProviderRequests": 0}
    assert observed == [_PIPELINE_CONFIG]


def test_pipeline_run_returns_the_committed_descendant_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_synthesis_raw_text_pipeline_config(_PIPELINE_CONFIG)
    artifact_root = tmp_path / "compiled-run"
    artifact_root.mkdir()
    (artifact_root / "summary.json").write_text(
        json.dumps({"documents": 30, "passedDocuments": 30}), encoding="utf-8"
    )
    (artifact_root / "_COMMIT.json").write_text("{}\n", encoding="utf-8")

    async def fake_run(path: Path) -> Path:
        assert path == _PIPELINE_CONFIG
        return artifact_root

    monkeypatch.setattr(raw_text_pipeline, "project_root_from_config", lambda _path: _PROJECT_ROOT)
    monkeypatch.setattr(raw_text_pipeline, "run_descendants", fake_run)

    result = run_raw_text_pipeline(
        project_root=_PROJECT_ROOT,
        config_path=_PIPELINE_CONFIG,
        config=config,
    )

    assert result["documents"] == 30
    assert result["passedDocuments"] == 30
    assert result["artifactRoot"] == str(artifact_root)
    assert result["commitSha256"] == sha256_file(artifact_root / "_COMMIT.json")


@pytest.mark.parametrize(
    ("surface", "expected"),
    (
        ("PACKAGE 1-7", (7,)),
        ("PLT NO.1-7", (7,)),
        ("RANGE 1 TO 7", (7,)),
        ("SEQ 7~1", (7,)),
        ("SDW/1 TO SDW/80", (80,)),
        ("SKU1-7X", ()),
    ),
)
def test_production_range_grammar_is_format_general(
    surface: str, expected: tuple[int, ...]
) -> None:
    assert inclusive_range_cardinalities(surface) == expected


def test_source_only_range_is_checked_against_the_final_agent_surface() -> None:
    binding = _coherence_binding()
    constraint = _range_constraint()

    validate_render_coherence(
        bindings=(binding,),
        constraints=(constraint,),
        source_target=_target(7),
        target=_target(9),
        outputs={binding.logical_key: BindingOutput({"slot_range": "PACKAGE 1-9"}, "1-9")},
    )
    with pytest.raises(ValueError, match="cardinality"):
        validate_render_coherence(
            bindings=(binding,),
            constraints=(constraint,),
            source_target=_target(7),
            target=_target(9),
            outputs={binding.logical_key: BindingOutput({"slot_range": "PACKAGE 1-7"}, "1-7")},
        )


def test_coherence_gate_runs_before_descendant_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    case = SimpleNamespace(
        document_id="doc_test",
        source_target=_target(7),
        target=_target(9),
        template=SimpleNamespace(
            bindings=(_coherence_binding(),),
            coherence_constraints=(_range_constraint(),),
        ),
    )
    calls: list[str] = []

    def reject_coherence(**_kwargs: Any) -> None:
        calls.append("coherence")
        raise ValueError("stale package range")

    def forbidden_route(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("route")
        raise AssertionError("routing must not run after a failed coherence gate")

    monkeypatch.setattr(descendant, "validate_render_coherence", reject_coherence)
    monkeypatch.setattr(descendant, "_binding_route", forbidden_route)

    with pytest.raises(ValueError, match="stale package range"):
        _build_initial_plan(case, seed=20260912, country_codes={})
    assert calls == ["coherence"]


def test_changed_coherence_member_stays_on_the_joint_residual_route() -> None:
    binding = _coherence_binding()
    case = SimpleNamespace(
        document_id="doc_changed_range",
        source=b"PACKAGE 1-7\n",
        source_target=_target(7),
        target=_target(9),
        template=SimpleNamespace(
            bindings=(binding,),
            coherence_constraints=(_range_constraint(),),
            byte_template=SimpleNamespace(slots=binding.occurrences),
        ),
    )

    plan = _build_initial_plan(case, seed=20260912, country_codes={})

    assert plan.residual_bindings == (binding,)
    assert plan.routes[0].runtime_route == "agent"
    assert plan.deterministic_outputs == {}


def test_unchanged_coherence_member_preserves_the_certified_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _coherence_binding()
    case = SimpleNamespace(
        document_id="doc_unchanged_range",
        source=b"PACKAGE 1-7\n",
        source_target=_target(7),
        target=_target(7),
        template=SimpleNamespace(
            bindings=(binding,),
            coherence_constraints=(_range_constraint(),),
            byte_template=SimpleNamespace(slots=binding.occurrences),
        ),
    )
    monkeypatch.setattr(descendant, "_validate_binding_format", lambda **_kwargs: None)

    plan = _build_initial_plan(case, seed=20260912, country_codes={})

    assert plan.residual_bindings == ()
    assert plan.routes[0].runtime_route == "deterministic"
    assert plan.deterministic_outputs[binding.logical_key].replacements == {
        "slot_range": "PACKAGE 1-7"
    }


def test_typed_target_that_violates_a_slot_envelope_routes_to_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot = SimpleNamespace(slot_id="slot_ocr_variant")
    binding = SimpleNamespace(
        binding_id="binding_voyage",
        logical_key="anchor:documentPatch.transport.voyageNumber",
        realization=SimpleNamespace(requires_agent=True, mode="agent_required"),
        target_paths=("documentPatch.transport.voyageNumber",),
        derivation=None,
        occurrences=(slot,),
        source_relationships=(),
    )
    case = SimpleNamespace(
        document_id="doc_ocr_variant",
        source=b"0NV18N1MA\n",
        source_target={"documentPatch": {"transport": {"voyageNumber": "0NVI8N1MA"}}},
        target={"documentPatch": {"transport": {"voyageNumber": "7KCI3Z0YW"}}},
        template=SimpleNamespace(
            bindings=(binding,),
            coherence_constraints=(),
            byte_template=SimpleNamespace(slots=(slot,)),
        ),
    )
    monkeypatch.setattr(descendant, "validate_render_coherence", lambda **_kwargs: None)
    monkeypatch.setattr(descendant, "_binding_route", lambda *_args: ("agent", "compiled"))
    monkeypatch.setattr(descendant, "_unchanged_target_output", lambda *_args: None)
    monkeypatch.setattr(
        descendant,
        "_render_agent_target_binding",
        lambda *_args, **_kwargs: BindingOutput({"slot_ocr_variant": "7KCI3Z0YW"}, "7KCI3Z0YW"),
    )

    def reject_format(**_kwargs: Any) -> None:
        raise ValueError("slot slot_ocr_variant replacement changes identifier shape")

    monkeypatch.setattr(
        descendant,
        "_validate_binding_format",
        reject_format,
    )

    plan = _build_initial_plan(case, seed=20260912, country_codes={})

    assert plan.deterministic_outputs == {}
    assert plan.residual_bindings == (binding,)
    assert plan.routes[0].runtime_route == "agent"
    assert "changes identifier shape" in plan.routes[0].route_reason


def test_final_coherence_failure_is_a_host_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = _coherence_binding()
    slot = binding.occurrences[0]
    template = SimpleNamespace(
        bindings=(binding,),
        coherence_constraints=(_range_constraint(),),
        byte_template=SimpleNamespace(slots=(slot,)),
    )
    case = SimpleNamespace(
        document_id="doc_test",
        source=b"PACKAGE 1-7\n",
        source_target=_target(7),
        target=_target(9),
        template=template,
        target_receipt=SimpleNamespace(
            synthetic_document_id="syn_test",
            target_origin="controlled_source_variant",
            changed_target_leaf_count=1,
        ),
    )
    plan = RenderPlan(
        routes=(
            SimpleNamespace(
                runtime_route="deterministic",
                slot_ids=("slot_range",),
            ),
        ),
        deterministic_outputs={
            binding.logical_key: BindingOutput({"slot_range": "PACKAGE 1-7"}, "1-7")
        },
        residual_bindings=(),
    )

    def reject_coherence(**_kwargs: Any) -> None:
        raise ValueError("stale package range")

    monkeypatch.setattr(descendant, "validate_render_coherence", reject_coherence)

    executed = _materialize_case(
        case=case,
        plan=plan,
        raw_output={},
        stage=_empty_stage(),
        country_codes={},
    )

    assert executed.result.status == "host_rejected"
    assert executed.result.error_type == "ValueError"
    assert executed.result.error_message == "stale package range"
    assert executed.result.semantic_coherence_valid is False
