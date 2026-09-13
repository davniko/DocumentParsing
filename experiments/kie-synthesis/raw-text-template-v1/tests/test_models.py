from __future__ import annotations

import pytest
from pydantic import ValidationError

from raw_text_template_experiment.models import AgentBindingProposal, CriticAgentOutput


def test_target_binding_requires_paths() -> None:
    with pytest.raises(ValidationError, match="target_binding requires target_paths"):
        AgentBindingProposal.model_validate(
            {
                "logical_key": "bad",
                "render_mode": "target_binding",
                "value_kind": "identifier",
                "group_kind": "document",
                "group_key": "document",
                "target_paths": (),
                "derivation": None,
                "dependency_paths": (),
                "occurrences": (
                    {
                        "line_start": "L00001",
                        "line_end": "L00001",
                        "source_text": "ABC123",
                        "occurrence_index": 0,
                    },
                ),
                "rationale": "Invalid fixture.",
            }
        )


def test_critic_cannot_pass_with_findings() -> None:
    with pytest.raises(ValidationError, match="pass requires zero findings"):
        CriticAgentOutput.model_validate(
            {
                "verdict": "pass",
                "findings": (
                    {
                        "finding_kind": "unowned_shipment_fact",
                        "line_ids": ("L00001",),
                        "evidence": "ABC123",
                        "explanation": "Fixture finding.",
                    },
                ),
                "additional_bindings": (),
                "rationale": "Contradictory fixture.",
            }
        )
