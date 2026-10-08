"""Deterministic template quotas for fixed consignee and notify instructions.

Sampling changes template frequency, never a template's party instructions.
Impossible requested margins fail before generation rather than changing OCR.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field


class TemplateSampling(BaseModel):
    """Exact document budget and optional document-level instruction fractions."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    samples: int = Field(gt=0)
    negotiable_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Fraction labeled negotiable; remaining sources may be non-negotiable or unknown."
        ),
    )
    notify_reference_fraction: float | None = Field(default=None, ge=0, le=1)


def instruction_stratum(target: dict) -> tuple[bool, bool]:
    patch = target["documentPatch"]
    if "negotiability" not in patch or patch["negotiability"] not in {
        "negotiable",
        "non_negotiable",
        None,
    }:
        raise ValueError("template sampling requires an explicit source negotiability label")
    return (
        patch["negotiability"] == "negotiable",
        any(p.get("sameAs") is not None for p in patch.get("parties", {}).get("notifyParties", [])),
    )


def source_variant_counts(config: dict, rows: dict[str, dict]) -> dict[str, int]:
    """Allocate exact integer quotas, then spread each stratum over its sources.

    Without a sampling policy every selected source retains variants_per_source.
    With one requested margin, other traits retain conditional source diversity.
    Requested margins are rounded half-up to document counts. The joint count closest
    to independence is chosen within the feasible interval; absent strata impose
    hard equalities. No rejection sampling, silent quota drift or duplicate IDs.
    """
    sources = config["source_ids"]
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= rows.keys():
        raise ValueError("template selection requires distinct available source IDs")
    if "template_sampling" not in config:
        count = config["variants_per_source"]
        if not isinstance(count, int) or count < 1:
            raise ValueError("variants_per_source must be positive")
        return dict.fromkeys(sources, count)
    policy = TemplateSampling.model_validate(config["template_sampling"])

    def allocate(groups):
        counts = dict.fromkeys(sources, 0)
        for ids, quota in groups:
            if not quota:
                continue
            if not ids:
                raise ValueError("requested template margin is unavailable in selected sources")
            ordered = sorted(
                ids,
                key=lambda sid: hashlib.sha256(
                    f"{config['seed']}:{sid}:template-quota".encode()
                ).digest(),
            )
            each, remainder = divmod(quota, len(ordered))
            counts.update({sid: each + (i < remainder) for i, sid in enumerate(ordered)})
        return {sid: count for sid, count in counts.items() if count}

    strata = defaultdict(list)
    for sid in sources:
        strata[instruction_stratum(rows[sid]["target"])].append(sid)
    n = policy.samples
    requested = (policy.negotiable_fraction, policy.notify_reference_fraction)
    specified = [i for i, fraction in enumerate(requested) if fraction is not None]
    if not specified:
        return allocate([(sources, n)])
    if len(specified) == 1:
        axis = specified[0]
        positive = math.floor(n * requested[axis] + 0.5)
        return allocate(
            [
                ([sid for key, ids in strata.items() if key[axis] == flag for sid in ids], count)
                for flag, count in ((True, positive), (False, n - positive))
            ]
        )
    neg, ref = (math.floor(n * fraction + 0.5) for fraction in requested)
    lower, upper = max(0, neg + ref - n), min(neg, ref)
    for key, required in (
        ((True, True), 0),
        ((True, False), neg),
        ((False, True), ref),
        ((False, False), neg + ref - n),
    ):
        if key not in strata:
            lower, upper = max(lower, required), min(upper, required)
    if lower > upper:
        available = {k: len(v) for k, v in strata.items()}
        raise ValueError(
            f"requested template margins are unavailable: {neg}/{n} negotiable, "
            f"{ref}/{n} notify-reference; available strata={available}"
        )
    joint = min(upper, max(lower, math.floor(neg * ref / n + 0.5)))
    quotas = {
        (True, True): joint,
        (True, False): neg - joint,
        (False, True): ref - joint,
        (False, False): n - neg - ref + joint,
    }
    return allocate([(strata[key], quota) for key, quota in quotas.items()])


def validate_instruction_inheritance(source: dict, target: dict) -> None:
    """Party wording/identity may vary; source instructions and references do not."""
    before, after = source["documentPatch"], target["documentPatch"]
    instruction_stratum(source)  # Missing keys are invalid; explicit unknown is valid.
    if "negotiability" not in after or after["negotiability"] != before["negotiability"]:
        raise ValueError("synthetic negotiability differs from source instruction")
    old = before.get("parties", {}).get("notifyParties", [])
    new = after.get("parties", {}).get("notifyParties", [])
    if any("sameAs" not in p for p in new) or [p.get("sameAs") for p in old] != [
        p.get("sameAs") for p in new
    ]:
        raise ValueError("synthetic notify references differ from source instructions")
