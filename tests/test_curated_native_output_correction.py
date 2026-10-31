"""Bounded native validation feedback, with both provider responses accounted."""

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import CuratedSynthesisConfig, Runner
from document_ocr.synthesis.independent_dg_context import hs_output_type

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "retries,expected_calls,passes,budget",
    [
        (0, 1, False, 1),
        (1, 2, True, 1),
        (1, 1, False, 0.007),
    ],
)
def test_actual_native_validator_feedback_and_accounting(
    tmp_path, retries, expected_calls, passes, budget
):
    config = yaml.safe_load(
        (ROOT / "configs/synthesis/mpci_bl_curated_v7_independent_validation.yaml").read_text()
    )
    config.update(output=str(tmp_path), budget_usd=budget)
    config["task_constraints_sha256"] = sha256_file(ROOT / config["task_constraints"])
    config = {
        key: value for key, value in config.items() if key in CuratedSynthesisConfig.model_fields
    }
    runner = Runner(ROOT, config)
    seen = []

    async def respond(messages, info):
        seen.append(messages)
        if len(seen) == 2:
            feedback = [
                p
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, RetryPromptPart)
            ]
            assert feedback and "285310, 285390" in str(feedback[-1].content)
        return ModelResponse(
            parts=[TextPart(json.dumps({"hs6": "285300" if len(seen) == 1 else "285390"}))],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
            provider_details={"cost": 0.00001},
        )

    runner.model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))
    model = hs_output_type({}, {"285310", "285390"})
    call = runner.call("hs", "test", model, "Select HS6", "Item", output_retries=retries)
    if passes:
        assert asyncio.run(call).hs6 == "285390"
    else:
        with pytest.raises(RuntimeError, match=r"maximum output retries|correction budget"):
            asyncio.run(call)
    assert len(seen) == expected_calls
    receipt = json.loads(next((tmp_path / "calls").glob("*.json")).read_text())
    assert receipt["usage"]["requests"] == expected_calls
    assert runner.spent == Decimal("0.00001") * expected_calls
    assert runner.reserved == 0
    assert receipt["outputRetries"] == retries


@pytest.mark.parametrize("value", [True, -1, 2])
def test_output_correction_is_explicitly_bounded(value):
    runner = Runner.__new__(Runner)
    with pytest.raises(ValueError, match="at most one"):
        asyncio.run(runner.call("hs", "test", None, "", "", output_retries=value))
