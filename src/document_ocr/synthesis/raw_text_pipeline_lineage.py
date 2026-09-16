"""Schemas for reading committed lineage from the retired rewrite pipeline.

These models describe immutable historical artifacts only.  They do not expose or retain the
retired execution path; keeping artifact readers separate prevents analysis tools from depending
on private implementation details of the active compiled-template pipeline.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_DOCUMENT_ID = Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class LineageRunReference(BaseModel):
    model_config = _STRICT

    path: str
    commitSha256: _SHA256
    transactionSha256: _SHA256


class InventoryHistoryRow(BaseModel):
    model_config = _STRICT

    stage: Literal["inventory"]
    round: int
    status: Literal[
        "training_ready",
        "needs_review",
        "call_failed",
        "compiler_blocked",
    ]
    run: LineageRunReference
    candidateSha256: _SHA256


class CertificationHistoryRow(BaseModel):
    model_config = _STRICT

    stage: Literal["certification"]
    attempt: Annotated[int, Field(ge=1)]
    status: Literal["certified", "needs_review", "call_failed"]
    semanticFindings: Annotated[int, Field(ge=0)]
    hostAuditPassed: bool
    run: LineageRunReference


class CorrectionHistoryRow(BaseModel):
    model_config = _STRICT

    stage: Literal["correction"]
    round: Annotated[int, Field(ge=1)]
    attempt: Annotated[int, Field(ge=1)] | None = None
    status: Literal[
        "unchanged_certified",
        "correction_candidate",
        "needs_review",
        "call_failed",
    ]
    run: LineageRunReference
    candidateSha256: _SHA256


PipelineHistoryRow = Annotated[
    InventoryHistoryRow | CertificationHistoryRow | CorrectionHistoryRow,
    Field(discriminator="stage"),
]


class PipelineResumeLineageRow(BaseModel):
    """One exact document row from a committed historical pipeline lineage."""

    model_config = _STRICT

    documentId: _DOCUMENT_ID
    status: Literal["certified", "blocked"]
    inventoryRound: Annotated[int, Field(ge=1)]
    inventoryRun: LineageRunReference
    sourceTextSha256: _SHA256
    currentCandidateSha256: _SHA256
    correctionRounds: Annotated[int, Field(ge=0)]
    finalCertificationRun: LineageRunReference | None
    blockingReason: str | None
    history: tuple[PipelineHistoryRow, ...]
