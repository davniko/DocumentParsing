from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

import document_ocr.synthesis.template_compiler.agents as compiler_agents
from document_ocr.synthesis.template_compiler.models import OpenRouterProviderConfig


class _StrictResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    quantity: int


def _provider(**changes: object) -> OpenRouterProviderConfig:
    payload = {
        "kind": "openrouter",
        "model": "z-ai/glm-5.3-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "reasoning_effort": "high",
        "request_timeout_seconds": 30.0,
        "transport_max_retries": 0,
        "max_output_tokens": 512,
        "require_parameters": True,
        "data_collection": "deny",
        "allow_fallbacks": False,
        "provider_only": ["gmicloud"],
        "max_prompt_price_usd_per_million": "0.09",
        "max_completion_price_usd_per_million": "0.30",
        "pricing": {
            "currency": "USD",
            "effective_date": "2026-09-28",
            "source_url": "https://openrouter.ai/z-ai/glm-5.3-flash",
            "input_usd_per_million": "0.09",
            "cached_input_usd_per_million": "0.018",
            "cache_write_multiplier": "1",
            "output_usd_per_million": "0.30",
        },
    }
    payload.update(changes)
    return OpenRouterProviderConfig.model_validate_json(json.dumps(payload))


def test_output_mode_is_explicit_and_native_profile_cannot_leak_into_prompted() -> None:
    native = _provider()
    assert native.output_mode == "native"
    assert isinstance(
        compiler_agents._output_spec(
            provider=native,
            output_type=_StrictResult,
            output_name="fixture",
            output_description="fixture schema",
        ),
        NativeOutput,
    )
    prompted = _provider(output_mode="prompted")
    spec = compiler_agents._output_spec(
        provider=prompted,
        output_type=_StrictResult,
        output_name="fixture",
        output_description="fixture schema",
    )
    assert isinstance(spec, PromptedOutput)
    assert spec.name == "fixture"
    assert spec.description == "fixture schema"
    with pytest.raises(ValidationError, match="prompted output cannot use a native"):
        _provider(
            output_mode="prompted",
            native_structured_output_profile="provider_verified",
            native_structured_output_source_url="https://example.test/endpoints",
        )


@pytest.mark.parametrize("role", ["compiler", "critic"])
@pytest.mark.asyncio
async def test_runtime_uses_prompted_output_for_either_role_without_changing_receipts(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    captured: dict[str, object] = {}

    class FakeAgent:
        def __class_getitem__(cls, _types: object) -> type[FakeAgent]:
            return cls

        def __init__(self, _model: object, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run(self, _prompt: str, **kwargs: object) -> SimpleNamespace:
            captured["run_kwargs"] = kwargs
            return SimpleNamespace(output=_StrictResult(quantity=7))

    monkeypatch.setattr(compiler_agents, "Agent", FakeAgent)
    runtime = object.__new__(compiler_agents.AgentRuntime)
    runtime._models = {role: object()}
    runtime._providers = {role: _provider(output_mode="prompted")}
    runtime._limiter = asyncio.Semaphore(1)
    output, stage = await runtime._call(
        role=role,
        pass_number=1,
        system_prompt="pinned system",
        payload={"documentId": "doc_fixture"},
        output_type=_StrictResult,
        output_name="fixture",
        retries=0,
    )

    assert isinstance(captured["output_type"], PromptedOutput)
    assert captured["retries"] == 0
    assert output == _StrictResult(quantity=7)
    assert stage.status == "success"
    assert stage.output == {"quantity": 7}
    assert stage.usage.estimatedCostUsd == Decimal(0)


def test_prompted_output_parser_rejects_wrong_json_types_before_host_acceptance() -> None:
    spec = compiler_agents._output_spec(
        provider=_provider(output_mode="prompted"),
        output_type=_StrictResult,
        output_name="strict_fixture",
        output_description="Return one integer quantity.",
    )
    good_model = FunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart('{"quantity":7}')])
    )
    good = Agent(good_model, output_type=spec, retries=0).run_sync("pinned request")
    assert good.output == _StrictResult(quantity=7)

    bad_model = FunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart('{"quantity":"7"}')])
    )
    with pytest.raises(Exception, match="Exceeded maximum output retries"):
        Agent(bad_model, output_type=spec, retries=0).run_sync("pinned request")
