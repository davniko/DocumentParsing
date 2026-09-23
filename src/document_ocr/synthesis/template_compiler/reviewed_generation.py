"""Explicit, source-pinned human corrections to rejected linguistic candidates.

Reviews do not bypass synthesis, geography, rendering or publication validators.
They are not provider responses and never masquerade as paid/cache receipts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.hashing import canonical_json_bytes, sha256_bytes


class ReviewedGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["reviewed_linguistic_correction_v1"]
    sample_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fields_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rationale: str = Field(min_length=1)
    output: dict[str, str]

    def validate_context(
        self,
        *,
        sample_id: str,
        source_document_id: str,
        source_template_sha256: str,
        scenario: Mapping[str, Any],
        fields: Sequence[Mapping[str, Any]],
    ) -> None:
        expected = (
            sample_id,
            source_document_id,
            source_template_sha256,
            sha256_bytes(canonical_json_bytes(scenario)),
            sha256_bytes(canonical_json_bytes(fields)),
        )
        actual = (
            self.sample_id,
            self.source_document_id,
            self.source_template_sha256,
            self.scenario_sha256,
            self.fields_sha256,
        )
        if actual != expected:
            raise ValueError("reviewed generation does not match its frozen scenario and fields")
        if set(self.output) != {field["key"] for field in fields}:
            raise ValueError(
                "reviewed generation must exactly cover the linguistic field inventory"
            )
        if any(
            not value.strip() or "\n" in value or "\r" in value for value in self.output.values()
        ):
            raise ValueError("reviewed generation values must be nonempty single-line text")

    def receipt(self, *, completed_at: str, usage: Mapping[str, Any]) -> dict[str, Any]:
        if (
            usage["requests"]
            or float(usage["estimatedCostUsd"])
            or usage["inputTokens"]
            or usage["outputTokens"]
        ):
            raise ValueError("manual review cannot claim provider usage")
        payload = self.model_dump(mode="json")
        return {
            "requestSha256": sha256_bytes(canonical_json_bytes(payload)),
            "outputSha256": sha256_bytes(canonical_json_bytes(self.output)),
            "output": self.output,
            "usage": dict(usage),
            "messages": [],
            "wallSeconds": 0,
            "completedAt": completed_at,
            "manualReview": payload,
            "cacheReused": False,
        }


class ReviewedResidual(BaseModel):
    """A human correction of presentation, with immutable source and target facts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["reviewed_residual_correction_v1"]
    sample_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rationale: str = Field(min_length=1)
    output: dict[str, str] = Field(min_length=1)

    def validate_context(
        self,
        *,
        sample_id: str,
        source_document_id: str,
        source: bytes,
        template: Mapping[str, Any],
        target: Mapping[str, Any],
        output: Mapping[str, str],
    ) -> None:
        expected = (
            sample_id,
            source_document_id,
            sha256_bytes(source),
            sha256_bytes(canonical_json_bytes(template)),
            sha256_bytes(canonical_json_bytes(target)),
        )
        actual = (
            self.sample_id,
            self.source_document_id,
            self.source_sha256,
            self.template_sha256,
            self.target_sha256,
        )
        if expected != actual or any(self.output.get(k) != v for k, v in output.items()):
            raise ValueError("reviewed residual differs from its frozen source, target or output")
        if any(not value.strip() for value in self.output.values()):
            raise ValueError("reviewed residual output cannot be empty")
