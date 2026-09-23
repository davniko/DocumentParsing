import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace as NS

import httpx2 as httpx
import pytest
from pydantic import create_model

from document_ocr.synthesis.template_compiler import complete_pipeline as p
from document_ocr.synthesis.template_compiler.spending_guard import SpendingGuard


@pytest.mark.parametrize("verified", [False, True])
def test_renderer_honors_explicitly_verified_openrouter_output_profile(monkeypatch, verified):
    from pathlib import Path

    from document_ocr.synthesis.template_compiler import descendant as render
    from document_ocr.synthesis.template_compiler.models import OpenRouterProviderConfig

    provider = OpenRouterProviderConfig.model_validate_json(
        json.dumps(
            {
                "kind": "openrouter",
                "model": "z-ai/glm-5.3-flash",
                "api_key_env": "OPENROUTER_API_KEY",
                "reasoning_effort": "high",
                "request_timeout_seconds": 30,
                "transport_max_retries": 0,
                "max_output_tokens": 1024,
                "require_parameters": True,
                "data_collection": "deny",
                "allow_fallbacks": False,
                "provider_only": ["test-provider"],
                "native_structured_output_profile": "provider_verified"
                if verified
                else "provider_default",
                "native_structured_output_source_url": "https://example.test/profile"
                if verified
                else None,
                "max_prompt_price_usd_per_million": "1",
                "max_completion_price_usd_per_million": "1",
                "pricing": {
                    "currency": "USD",
                    "effective_date": "2026-09-21",
                    "source_url": "https://example.test/pricing",
                    "input_usd_per_million": "1",
                    "cached_input_usd_per_million": "0",
                    "cache_write_multiplier": "1",
                    "output_usd_per_million": "1",
                },
            }
        )
    )
    monkeypatch.setattr(render, "load_provider_key", lambda *args: "test-not-a-key")
    monkeypatch.setattr(render, "AsyncOpenAI", lambda **kwargs: NS())
    monkeypatch.setattr(render, "OpenRouterProvider", lambda **kwargs: NS())
    captured = {}

    def model(name, **kwargs):
        captured.update(kwargs)
        return NS()

    monkeypatch.setattr(render, "OpenRouterModel", model)
    render._provider_model(project_root=Path.cwd(), environment_file="unused", provider=provider)
    if verified:
        assert captured["profile"]["supports_json_schema_output"] is True
    else:
        assert captured["profile"] is None


@pytest.mark.parametrize(
    "kind,not_sent",
    [
        (httpx.ConnectError, True),
        (httpx.ConnectTimeout, True),
        (httpx.ReadError, False),
        (httpx.ReadTimeout, False),
        (httpx.WriteError, False),
        (httpx.WriteTimeout, False),
        (httpx.RemoteProtocolError, False),
        (RuntimeError, False),
    ],
)
def test_only_typed_connection_failures_release_the_reservation(kind, not_sent):
    error = RuntimeError("provider wrapper")
    error.__cause__ = kind("failure")
    assert p._request_was_not_sent(error) is not_sent
    assert p._request_was_not_sent(error.__cause__) is not_sent


def test_unrelated_context_and_nested_read_failures_are_not_unsent_proof():
    error = RuntimeError("wrapper")
    error.__context__ = httpx.ConnectError("previous unrelated failure")
    assert not p._request_was_not_sent(error)
    error.__cause__ = httpx.ReadTimeout("request already sent")
    error.__cause__.__cause__ = httpx.ConnectError("older failure")
    assert not p._request_was_not_sent(error)
    error.__cause__ = error
    assert not p._request_was_not_sent(error)


def test_actual_openai_pydantic_transport_preserves_connection_evidence():
    from openai import AsyncOpenAI
    from pydantic_ai import Agent, NativeOutput
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    async def handler(request):
        raise httpx.ConnectError("simulated TLS establishment failure", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            sdk = AsyncOpenAI(api_key="test-not-a-real-key", http_client=client, max_retries=0)
            model = OpenAIResponsesModel("gpt-5.6-luna", provider=OpenAIProvider(openai_client=sdk))
            agent = Agent(
                model, output_type=NativeOutput(create_model("Field", name=(str, ...))), retries=0
            )
            with pytest.raises(Exception) as captured:
                await agent.run("test")
            assert p._request_was_not_sent(captured.value)
            await sdk.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind,uncertain", [(httpx.ConnectError, 0), (httpx.ReadTimeout, 1)])
def test_failure_receipt_and_durable_settlement_follow_transmission_evidence(
    tmp_path, monkeypatch, kind, uncertain
):
    failure = RuntimeError("provider failure")
    failure.__cause__ = kind("transport failure")

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, *args, **kwargs):
            raise failure

    monkeypatch.setattr(p, "Agent", Agent)
    monkeypatch.setattr(p.render, "_provider_settings", lambda _: {})
    monkeypatch.setattr(p, "usage_receipt", lambda *args, **kwargs: p.render._empty_usage())
    provider = NS(
        kind="openai",
        model_dump=lambda **_: {},
        transport_max_retries=0,
        max_output_tokens=100,
        pricing=NS(
            input_usd_per_million=Decimal("0.2"),
            cached_input_usd_per_million=Decimal("0.02"),
            cache_write_multiplier=Decimal("1.25"),
            output_usd_per_million=Decimal("1.2"),
        ),
    )
    guard = SpendingGuard(tmp_path / "ledger.sqlite3", limit_usd=Decimal(1), pricing_sha256="test")

    async def run():
        with pytest.raises(p.ProviderCallError):
            await p._cached_fields(
                cache=tmp_path / "cache",
                model=None,
                provider=provider,
                system_prompt="test",
                payload={},
                output_type=create_model("Fields", name=(str, ...)),
                limiter=asyncio.Semaphore(1),
                authorized=True,
                spending=guard,
            )

    asyncio.run(run())
    snapshot = guard.snapshot()
    assert snapshot["requests"] == 1
    assert snapshot["uncertainRequests"] == uncertain
    assert (Decimal(snapshot["occupiedUsd"]) > 0) == bool(uncertain)
    receipts = list((tmp_path / "cache/failures").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_bytes())
    assert (receipt["transmissionEvidence"] == "not_proven_unsent") == bool(uncertain)
    assert receipt["errorCauses"][0]["type"] == kind.__name__
