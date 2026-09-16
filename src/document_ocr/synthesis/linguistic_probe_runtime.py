"""Shared provider accounting and transcript utilities for linguistic probes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import cast

from dotenv import dotenv_values
from pydantic import JsonValue
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.usage import RequestUsage

from document_ocr.synthesis.config import (
    LinguisticProbePricingConfig,
    LinguisticProbeProviderConfig,
)
from document_ocr.synthesis.usage_receipt import LinguisticUsageReceipt
from document_ocr.training.config import resolve_config_path

_USD_QUANTUM = Decimal("0.000000000001")


def load_openai_key(project_root: Path, environment_file: str) -> str:
    return load_provider_key(project_root, environment_file, "OPENAI_API_KEY")


def load_provider_key(project_root: Path, environment_file: str, key_name: str) -> str:
    path = resolve_config_path(project_root, environment_file)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"environment file is not a regular file: {path}")
    key = os.environ.get(key_name) or dotenv_values(path).get(key_name)
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"{key_name} is absent or empty")
    return key


def openai_responses_settings(
    provider: LinguisticProbeProviderConfig,
    *,
    max_output_tokens: int | None = None,
) -> OpenAIResponsesModelSettings:
    """Translate only explicitly configured sampling controls to PydanticAI.

    Operational limits and the selected reasoning mode are always explicit.
    Sampling controls are added one-by-one so an absent generation-settings
    section cannot accidentally override provider defaults.
    """

    output_limit = max_output_tokens or provider.max_output_tokens
    if output_limit > provider.max_output_tokens:
        raise ValueError("stage output limit exceeds provider max_output_tokens")
    values: OpenAIResponsesModelSettings = {
        "max_tokens": output_limit,
        "timeout": provider.request_timeout_seconds,
        "openai_reasoning_effort": provider.reasoning_effort,
        "openai_reasoning_mode": "standard",
        "openai_reasoning_context": "current_turn",
        "openai_store": provider.store_responses,
    }
    if provider.service_tier is not None:
        values["openai_service_tier"] = provider.service_tier
    generation = provider.generation_settings
    if generation is not None:
        if generation.temperature is not None:
            values["temperature"] = generation.temperature
        if generation.top_p is not None:
            values["top_p"] = generation.top_p
        if generation.text_verbosity is not None:
            values["openai_text_verbosity"] = generation.text_verbosity
    return values


def price_usage(usage: RequestUsage, pricing: LinguisticProbePricingConfig) -> Decimal:
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0:
        raise ValueError("provider reports cache tokens above total input tokens")
    cost = (
        Decimal(uncached) * Decimal(str(pricing.input_usd_per_million))
        + Decimal(usage.cache_read_tokens) * Decimal(str(pricing.cached_input_usd_per_million))
        + Decimal(usage.cache_write_tokens)
        * Decimal(str(pricing.input_usd_per_million))
        * Decimal(str(pricing.cache_write_multiplier))
        + Decimal(usage.output_tokens) * Decimal(str(pricing.output_usd_per_million))
    ) / Decimal(1_000_000)
    return cost.quantize(_USD_QUANTUM, rounding=ROUND_HALF_UP)


def usage_receipt(
    responses: Sequence[ModelResponse],
    pricing: LinguisticProbePricingConfig,
    *,
    require_provider_cost: bool = False,
) -> LinguisticUsageReceipt:
    usage = RequestUsage()
    for response in responses:
        usage.incr(response.usage)
    reasoning_tokens = usage.details.get("reasoning_tokens", 0)
    if not isinstance(reasoning_tokens, int) or reasoning_tokens < 0:
        raise ValueError("provider reports invalid reasoning token usage")
    response_ids = tuple(
        response.provider_response_id
        for response in responses
        if response.provider_response_id is not None
    )
    finish_reasons = tuple(response.finish_reason or "unknown" for response in responses)
    provider_costs: list[Decimal] = []
    downstream_providers: list[str] = []
    for response in responses:
        details = response.provider_details or {}
        raw_cost = details.get("cost")
        if raw_cost is not None:
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float, str, Decimal)):
                raise ValueError("provider reports invalid monetary cost")
            cost = Decimal(str(raw_cost))
            if not cost.is_finite() or cost < 0:
                raise ValueError("provider reports invalid monetary cost")
            provider_costs.append(cost)
        downstream = details.get("downstream_provider")
        if downstream is not None:
            if not isinstance(downstream, str) or not downstream.strip():
                raise ValueError("provider reports invalid downstream provider")
            downstream_providers.append(downstream)
    if provider_costs and len(provider_costs) != len(responses):
        raise ValueError("provider cost coverage is incomplete")
    if require_provider_cost and len(provider_costs) != len(responses):
        raise ValueError("provider cost receipt is required for every response")
    provider_cost = (
        sum(provider_costs, Decimal(0)).quantize(_USD_QUANTUM, rounding=ROUND_HALF_UP)
        if provider_costs
        else None
    )
    accounting_anomaly = reasoning_tokens > usage.output_tokens
    return LinguisticUsageReceipt(
        requests=len(responses),
        providerResponseIds=response_ids,
        finishReasons=finish_reasons,
        inputTokens=usage.input_tokens,
        cacheReadTokens=usage.cache_read_tokens,
        cacheWriteTokens=usage.cache_write_tokens,
        outputTokens=usage.output_tokens,
        reasoningTokens=reasoning_tokens,
        visibleOutputTokens=max(0, usage.output_tokens - reasoning_tokens),
        providerTokenAccountingAnomaly=accounting_anomaly,
        estimatedCostUsd=price_usage(usage, pricing),
        providerReportedCostUsd=provider_cost,
        downstreamProviders=tuple(downstream_providers),
    )


def model_messages(messages: Sequence[ModelRequest | ModelResponse]) -> JsonValue:
    return cast(JsonValue, ModelMessagesTypeAdapter.dump_python(list(messages), mode="json"))
