"""Dependency-neutral provider usage receipt shared by synthesis workflows."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


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
    providerTokenAccountingAnomaly: bool = False
    estimatedCostUsd: Annotated[Decimal, Field(ge=0, decimal_places=12)]
    providerReportedCostUsd: Annotated[Decimal, Field(ge=0, decimal_places=12)] | None = None
    downstreamProviders: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def token_buckets_are_consistent(self) -> LinguisticUsageReceipt:
        if self.cacheReadTokens + self.cacheWriteTokens > self.inputTokens:
            raise ValueError("cache buckets exceed input token total")
        if self.reasoningTokens > self.outputTokens and not self.providerTokenAccountingAnomaly:
            raise ValueError("reasoning tokens exceed output token total")
        if self.providerTokenAccountingAnomaly:
            if self.visibleOutputTokens > self.outputTokens:
                raise ValueError("anomalous visible output tokens exceed output total")
        elif self.visibleOutputTokens != self.outputTokens - self.reasoningTokens:
            raise ValueError("visible output tokens differ from output minus reasoning")
        if self.requests != len(self.finishReasons):
            raise ValueError("request count differs from response finish-reason count")
        return self
