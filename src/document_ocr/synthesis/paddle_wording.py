"""Bounded proposals for additional native-only linguistic owners."""

from __future__ import annotations

import json

from pydantic import ConfigDict, Field, create_model

SYSTEM = """Generate the requested fictional source-only fields for a synthetic maritime document.
The shipment facts are authoritative. Do not rename sampled goods or change any route, country,
hazard or quantity. Only the requested fields are generated. Respect their stated ownership.
A company name belongs only in its name field, never repeated in its postal address. Names must
be actual plausible fictional names, not instructions or labels. Do not copy the original private
identity. Keep legal suffix and source casing style when requested. Return just the schema object.
"""


def wording_model(spec, patterns):
    if set(patterns) - spec.keys():
        raise ValueError("wording pattern names an unrequested owner")
    return create_model(
        "PaddleAdditionalWording",
        __config__=ConfigDict(extra="forbid"),
        **{
            key: (str, Field(min_length=1, description=description, pattern=patterns.get(key)))
            for key, description in spec.items()
        },
    )


async def propose(runner, specification):
    """Return a proposal; only explicit human review can make it renderable."""
    prompt = json.dumps(
        {"requested": specification["requested"], **specification["context"]}, ensure_ascii=False
    )
    model = wording_model(specification["requested"], specification["patterns"])
    result = await runner.call(
        "paddle-additional-wording",
        specification["documentId"],
        model,
        SYSTEM,
        prompt,
        output_retries=1,
    )
    return {
        "status": "PROPOSED_REQUIRES_REVIEW",
        "prompt": prompt,
        "values": result.model_dump(),
        **{k: specification[k] for k in ("documentId", "sourceFingerprint", "candidateSha256")},
    }
