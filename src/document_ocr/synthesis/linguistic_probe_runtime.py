"""Shared provider accounting and transcript utilities for linguistic probes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated, cast

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.usage import RequestUsage

from document_ocr.synthesis.config import (
    LinguisticProbePricingConfig,
    LinguisticProbeProviderConfig,
)
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_USD_QUANTUM = Decimal("0.000000000001")


class LinguisticUsageReceipt(BaseModel):
    model_config = _STRICT

    requests: Annotated[int, Field(ge=0)]
    providerResponseIds: tuple[NonEmptyText, ...]
    finishReasons: tuple[NonEmptyText, ...]
    inputTokens: Annotated[int, Field(ge=0)]
    cacheReadTokens: Annotated[int, Field(ge=0)]
    cacheWriteTokens: Annotated[int, Field(ge=0)]
    outputTokens: Annotated[int, Field(ge=0)]
    reasoningTokens: Annotated[int, Field(ge=0)]
    visibleOutputTokens: Annotated[int, Field(ge=0)]
    estimatedCostUsd: Annotated[Decimal, Field(ge=0, decimal_places=12)]

    @model_validator(mode="after")
    def token_buckets_are_consistent(self) -> LinguisticUsageReceipt:
        if self.cacheReadTokens + self.cacheWriteTokens > self.inputTokens:
            raise ValueError("cache buckets exceed input token total")
        if self.reasoningTokens > self.outputTokens:
            raise ValueError("reasoning tokens exceed output token total")
        if self.visibleOutputTokens != self.outputTokens - self.reasoningTokens:
            raise ValueError("visible output tokens differ from output minus reasoning")
        if self.requests != len(self.finishReasons):
            raise ValueError("request count differs from response finish-reason count")
        return self


def load_openai_key(project_root: Path, environment_file: str) -> str:
    path = resolve_config_path(project_root, environment_file)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"environment file is not a regular file: {path}")
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(path).get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("OPENAI_API_KEY is absent or empty")
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
        + Decimal(usage.cache_read_tokens)
        * Decimal(str(pricing.cached_input_usd_per_million))
        + Decimal(usage.cache_write_tokens)
        * Decimal(str(pricing.input_usd_per_million))
        * Decimal(str(pricing.cache_write_multiplier))
        + Decimal(usage.output_tokens) * Decimal(str(pricing.output_usd_per_million))
    ) / Decimal(1_000_000)
    return cost.quantize(_USD_QUANTUM, rounding=ROUND_HALF_UP)


def usage_receipt(
    responses: Sequence[ModelResponse], pricing: LinguisticProbePricingConfig
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
    return LinguisticUsageReceipt(
        requests=len(responses),
        providerResponseIds=response_ids,
        finishReasons=finish_reasons,
        inputTokens=usage.input_tokens,
        cacheReadTokens=usage.cache_read_tokens,
        cacheWriteTokens=usage.cache_write_tokens,
        outputTokens=usage.output_tokens,
        reasoningTokens=reasoning_tokens,
        visibleOutputTokens=usage.output_tokens - reasoning_tokens,
        estimatedCostUsd=price_usage(usage, pricing),
    )


def model_messages(messages: Sequence[ModelRequest | ModelResponse]) -> JsonValue:
    return cast(JsonValue, ModelMessagesTypeAdapter.dump_python(list(messages), mode="json"))
