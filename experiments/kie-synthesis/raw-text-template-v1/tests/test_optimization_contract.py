from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from raw_text_template_experiment.models import AgentBindingProposal, CriticAgentOutput
from raw_text_template_experiment.optimization_contract import (
    DISCRIMINATED_BINDING_ADAPTER,
    DISCRIMINATED_CRITIC_ADAPTER,
    CriticPassOutput,
    DeterministicDerivedRendering,
    DiscriminatedCompilerAgentOutput,
    TargetBindingRendering,
    discriminate_binding,
    discriminate_critic,
    project_legacy_critic_candidate,
    restore_legacy_compiler,
)


def _legacy_binding(**updates: object) -> AgentBindingProposal:
    value: dict[str, object] = {
        "logical_key": "anchor:documentPatch.billOfLadingNumber",
        "render_mode": "target_binding",
        "value_kind": "identifier",
        "group_kind": "document",
        "group_key": "document",
        "target_paths": ("documentPatch.billOfLadingNumber",),
        "derivation": None,
        "dependency_paths": (),
        "dependency_bindings": (),
        "occurrences": (
            {
                "line_start": "L00001",
                "line_end": "L00001",
                "source_text": "ABC123",
                "occurrence_index": 0,
            },
        ),
        "rationale": "The printed document number is shipment-specific.",
    }
    value.update(updates)
    return AgentBindingProposal.model_validate(value)


def test_discriminated_contract_preserves_accepted_target_binding() -> None:
    translated = discriminate_binding(_legacy_binding())

    assert isinstance(translated.rendering, TargetBindingRendering)
    assert translated.rendering.target_paths == ("documentPatch.billOfLadingNumber",)
    assert (
        DISCRIMINATED_BINDING_ADAPTER.validate_python(translated.model_dump(mode="python"))
        == translated
    )


def test_discriminated_contract_requires_derived_dependencies() -> None:
    invalid = {
        "logical_key": "agent:derived:count",
        "value_kind": "integer",
        "group_kind": "package",
        "group_key": "package:total",
        "rendering": {
            "render_mode": "deterministic_derived",
            "target_paths": [],
            "derivation": "package_count",
            "dependencies": [],
        },
        "occurrences": [
            {
                "line_start": "L00001",
                "line_end": "L00001",
                "source_text": "ONE",
                "occurrence_index": 0,
            }
        ],
        "rationale": "Printed derived total.",
    }

    with pytest.raises(ValidationError, match="at least 1 item"):
        DISCRIMINATED_BINDING_ADAPTER.validate_json(json.dumps(invalid))


def test_discriminated_contract_excludes_dependencies_from_residual_mode() -> None:
    invalid = {
        "logical_key": "agent:residual",
        "value_kind": "identifier",
        "group_kind": "other",
        "group_key": "other",
        "rendering": {
            "render_mode": "agent_residual",
            "target_paths": [],
            "dependency_paths": ["documentPatch.billOfLadingNumber"],
        },
        "occurrences": [
            {
                "line_start": "L00001",
                "line_end": "L00001",
                "source_text": "ABC123-X",
                "occurrence_index": 0,
            }
        ],
        "rationale": "Printed residual identifier.",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DISCRIMINATED_BINDING_ADAPTER.validate_json(json.dumps(invalid))


def test_discriminated_contract_translates_derived_dependencies() -> None:
    legacy = _legacy_binding(
        logical_key="agent:derived:count",
        render_mode="deterministic_derived",
        target_paths=(),
        derivation="package_count",
        dependency_paths=("documentPatch.cargoPackages[0].quantity",),
    )

    translated = discriminate_binding(legacy)

    assert isinstance(translated.rendering, DeterministicDerivedRendering)
    assert translated.rendering.dependencies[0].kind == "target_path"
    assert translated.rendering.dependencies[0].value == "documentPatch.cargoPackages[0].quantity"


def test_discriminated_compiler_round_trip_preserves_host_contract() -> None:
    binding = _legacy_binding()
    output = DiscriminatedCompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier Ltd",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "Example Carrier Ltd",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact printed principal.",
            },
            "anchor_overrides": (),
            "bindings": (discriminate_binding(binding).model_dump(mode="python"),),
            "unresolved": (),
            "semantic_only_target_facts": (),
        }
    )

    restored = restore_legacy_compiler(output)

    assert restored.bindings == (binding,)


def test_discriminated_compiler_defers_duplicate_target_owners_to_host() -> None:
    first = discriminate_binding(_legacy_binding(logical_key="owner:first"))
    second = discriminate_binding(_legacy_binding(logical_key="owner:second"))
    payload = {
        "carrier": {
            "canonical_name": "Example Carrier Ltd",
            "aliases": (),
            "evidence_occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "Example Carrier Ltd",
                    "occurrence_index": 0,
                },
            ),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact printed principal.",
        },
        "anchor_overrides": (),
        "bindings": (
            first.model_dump(mode="python"),
            second.model_dump(mode="python"),
        ),
        "unresolved": (),
        "semantic_only_target_facts": (),
    }

    output = DiscriminatedCompilerAgentOutput.model_validate(payload)

    assert tuple(binding.logical_key for binding in output.bindings) == (
        "owner:first",
        "owner:second",
    )


def test_discriminated_compiler_defers_carrier_static_provenance_to_host() -> None:
    first = discriminate_binding(
        _legacy_binding(logical_key="carrier:name", render_mode="carrier_static")
    )
    second = discriminate_binding(
        _legacy_binding(logical_key="carrier:domain", render_mode="carrier_static")
    )
    payload = {
        "carrier": {
            "canonical_name": "Example Carrier Ltd",
            "aliases": (),
            "evidence_occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "Example Carrier Ltd",
                    "occurrence_index": 0,
                },
            ),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact printed principal.",
        },
        "anchor_overrides": (),
        "bindings": (
            first.model_dump(mode="python"),
            second.model_dump(mode="python"),
        ),
        "unresolved": (),
        "semantic_only_target_facts": (),
    }

    output = DiscriminatedCompilerAgentOutput.model_validate(payload)

    assert len(output.bindings) == 2


def test_discriminated_critic_pass_has_no_patch_surface() -> None:
    legacy = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "remove_inventory_binding_ids": (),
            "additional_bindings": (),
            "semantic_only_target_facts": (),
            "rationale": "No defect remains.",
        }
    )

    translated = discriminate_critic(legacy)

    assert isinstance(translated, CriticPassOutput)
    assert translated.model_dump(mode="json") == {
        "verdict": "pass",
        "rationale": "No defect remains.",
    }


def test_discriminated_critic_rejects_pass_with_patch_data() -> None:
    invalid = {
        "verdict": "pass",
        "findings": [],
        "remove_binding_logical_keys": [],
        "additional_bindings": [],
        "semantic_only_target_facts": [
            {
                "target_path": "documentPatch.negotiability",
                "rationale": "Retain the existing host classification.",
            }
        ],
        "rationale": "No defect remains.",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DISCRIMINATED_CRITIC_ADAPTER.validate_python(project_legacy_critic_candidate(invalid))


def test_discriminated_critic_revision_requires_finding() -> None:
    invalid = {
        "verdict": "revise",
        "findings": [],
        "remove_inventory_binding_ids": [],
        "additional_bindings": [],
        "semantic_only_target_facts": [],
        "rationale": "A revision was requested without evidence.",
    }

    with pytest.raises(ValidationError, match="at least 1 item"):
        DISCRIMINATED_CRITIC_ADAPTER.validate_json(json.dumps(invalid))
