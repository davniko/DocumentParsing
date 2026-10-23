"""Native private HS/physical context without changing authoritative DG identity.

An HS proposal is a synthetic plausibility aid, never an ECICS enrichment. The
source presentation, not this private context, controls public HS-field presence.
"""

from __future__ import annotations

import json
from collections.abc import Container, Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, create_model, field_validator

from document_ocr.synthesis.dangerous_goods_registry import DangerousGoodsHmtRecord
from document_ocr.synthesis.independent_native_evidence import verify_native_evidence
from document_ocr.synthesis.template_compiler.dangerous_goods_packaging import profile

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
PhysicalForm = Literal["liquid", "solid", "gas", "article"]
HS_STAGE = "independent-dg-hs-context-v1"
FORM_STAGE = "independent-dg-physical-form-v1"

HS_SYSTEM = """Return only one plausible recognized six-digit HS commodity code,
or null when uncertain, for the supplied maritime DG item. This is unverified
synthetic context, not legal customs classification. The authoritative DG name,
qualifiers, UN number and hazards must not change to fit a code. Do not rename the
product or return product wording. Respect supplied HS-heading exclusions.
"""
FORM_SYSTEM = """Identify the physical cargo form of this exact maritime DG item:
liquid, solid, gas, or article; return null if the identity does not settle it.
An intact battery, device, machine, vehicle, kit or other manufactured article is
an article, not the state of its internal chemical contents. A compressed,
liquefied or refrigerated gas remains gas for this planning classification.
Preserve all supplied qualifiers. Do not infer physical state from hazard-class
number alone. Do not rename the product, select packaging or add declarations.
"""


class HsRequest(BaseModel):
    model_config = _STRICT

    dangerousGoods: dict[str, Any]
    cargoPresentation: dict[str, Any] | None


class PhysicalRequest(BaseModel):
    model_config = _STRICT

    dangerousGoods: dict[str, Any]
    cargoPresentation: dict[str, Any] | None


class HsProposal(BaseModel):
    model_config = _STRICT

    hs6: Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")] | None


class PhysicalProposal(BaseModel):
    model_config = _STRICT

    form: PhysicalForm | None


@lru_cache(maxsize=32)
def _hs_output_type(codes: frozenset[str]) -> type[HsProposal]:
    @field_validator("hs6")
    @classmethod
    def pinned_leaf(cls, code):
        if code is not None and code not in codes:
            nearby = sorted(c for c in codes if c[:4] == code[:4])
            raise ValueError(
                "Use a recognized six-digit HS leaf, not an obsolete or padded heading. "
                + (
                    "Recognized leaves under that heading: " + ", ".join(nearby)
                    if nearby
                    else "That heading is not in the eligible vocabulary."
                )
            )
        return code

    return create_model(
        "PinnedSyntheticHsProposal",
        __base__=HsProposal,
        __validators__={"pinned_leaf": pinned_leaf},
    )


def hs_output_type(frozen: Mapping[str, Any], hs_codes: Container[str]) -> type[HsProposal]:
    """Local native-output validation, without sending 5,612 codes to the model.

    PydanticAI can request a bounded correction of a rejected leaf. A failed
    retry or explicit null still holds; no host substitution is possible.
    """
    presentation = frozen.get("cargoPresentation")
    codes = frozenset(hs_codes)
    if presentation is not None:
        codes = frozenset(
            c
            for c in codes
            if c[:4] in presentation["hsHeadings"] and c not in presentation["excludedHs6"]
        )
    if not codes:
        raise ValueError("synthetic HS vocabulary is empty for this presentation")
    return _hs_output_type(codes)


def named_eligible(record: DangerousGoodsHmtRecord) -> bool:
    """Candidate identity eligibility; not a certificate for every package/layout."""
    return record.maritime_eligible and not (record.nos_entry or record.technical_name_required)


def known_physical_form(record: DangerousGoodsHmtRecord) -> PhysicalForm | None:
    """Use explicit existing section context, not fabricated solid/liquid flags."""
    if not named_eligible(record):
        return None
    try:
        return profile(record).state
    except ValueError:
        pass
    # Class 2 also includes refrigerating machines (173.306), so the
    # hazard alone is not physical-form evidence. These explicit sections
    # cover nonliquefied, liquefied and refrigerated compressed gases.
    if record.exact_hazard_class.startswith("2.") and record.source_fields.get(
        "Non-bulk Packaging"
    ) in {"302", "304", "316"}:
        return "gas"
    if record.un_number in {"3166", "3171"}:
        return "article"
    return None


def _context(frozen: Mapping[str, Any]) -> dict[str, Any]:
    if frozen.get("regime") != "dg" or not isinstance(frozen.get("dangerousGoods"), Mapping):
        raise ValueError("private DG context requires an authoritative DG request")
    return {
        "dangerousGoods": deepcopy(dict(frozen["dangerousGoods"])),
        "cargoPresentation": deepcopy(frozen.get("cargoPresentation")),
    }


def hs_request(frozen: Mapping[str, Any]) -> HsRequest:
    return HsRequest.model_validate(_context(frozen))


def form_request(frozen: Mapping[str, Any]) -> PhysicalRequest:
    return PhysicalRequest.model_validate(_context(frozen))


def hs_prompt(request: HsRequest) -> str:
    """Keep the call tiny; registry membership is checked in code after receipt."""
    return json.dumps(request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def form_prompt(request: PhysicalRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def validate_hs(
    frozen: Mapping[str, Any], answer: Mapping[str, Any], hs_codes: Container[str]
) -> HsProposal:
    result = HsProposal.model_validate(answer)
    if result.hs6 is None:
        raise ValueError("UNKNOWN synthetic HS proposal has no random-code fallback")
    if result.hs6 not in hs_codes:
        raise ValueError("synthetic HS proposal is outside the pinned HS6 registry")
    presentation = hs_request(frozen).cargoPresentation
    if presentation is not None and (
        result.hs6[:4] not in presentation["hsHeadings"]
        or result.hs6 in presentation["excludedHs6"]
    ):
        raise ValueError("synthetic HS proposal violates the reviewed cargo presentation")
    return result


def validate_form(
    frozen: Mapping[str, Any], answer: Mapping[str, Any], known_states: Mapping[str, str]
) -> PhysicalProposal:
    result = PhysicalProposal.model_validate(answer)
    if result.form is None:
        raise ValueError("UNKNOWN DG physical form has no solid/liquid fallback")
    expected = known_states.get(frozen["dangerousGoods"]["recordId"])
    if expected is not None and result.form != expected:
        raise ValueError("received DG physical form contradicts known section context")
    return result


def physical_form(
    frozen: Mapping[str, Any], known_states: Mapping[str, str]
) -> PhysicalForm | None:
    if frozen.get("regime") != "dg":
        if "nativeDgPhysicalEvidence" in frozen:
            raise ValueError("non-DG request carries private DG physical evidence")
        return None
    expected = known_states.get(frozen["dangerousGoods"]["recordId"])
    evidence = frozen.get("nativeDgPhysicalEvidence")
    if evidence is not None:
        return validate_form(frozen, evidence["output"], known_states).form
    if expected is None:
        raise ValueError("DG physical form needs explicit native evidence")
    return PhysicalProposal(form=expected).form


def packing_context(frozen: Mapping[str, Any], known_states: Mapping[str, str]) -> str:
    form = physical_form(frozen, known_states)
    if form is None:
        raise ValueError("DG packing context requires an explicit physical form")
    text = "Frozen physical form (private synthesis context, not transport approval): " + form
    if form == "gas":
        text += (
            ". Gas cargo must remain inside suitable self-contained gas receptacles. "
            "PACKAGE_PACKAGE may count generic declared shipping packages containing those "
            "receptacles when no specific cylinder category is available. This is an explicit "
            "generic printed declaration, not an assertion that a cylinder is a drum, bag or "
            "ocean container. Outer crates/cases may also hold receptacles. Do not invent "
            "pressure, material, certification or approval. UNKNOWN remains available."
        )
    return text


def verify_dg_context_evidence(
    root: Path,
    frozen: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    known_states: Mapping[str, str],
    hs_codes: Container[str],
    hs_links: Mapping[str, Container[str]],
) -> None:
    """Reconstruct requests, then verify actual native receipts before use."""
    if frozen.get("regime") != "dg":
        if any(key in frozen for key in ("nativeDgHsEvidence", "nativeDgPhysicalEvidence")):
            raise ValueError("non-DG request carries private DG context evidence")
        return
    code = frozen["goods"][0]["hs6"]
    validate_hs(frozen, {"hs6": code}, hs_codes)
    record_id = frozen["dangerousGoods"]["recordId"]
    identity = f"{frozen['sourceDocumentId']}:{frozen['variant']}"
    is_joined = code in hs_links.get(record_id, ())
    hs_evidence = frozen.get("nativeDgHsEvidence")
    if is_joined:
        if hs_evidence is not None:
            raise ValueError("an exact HS join must not be relabeled as a native HS proposal")
    else:
        if not isinstance(hs_evidence, Mapping):
            raise ValueError("unjoined DG identity requires native synthetic HS evidence")
        request = hs_request(frozen)
        output = verify_native_evidence(
            root,
            hs_evidence,
            stage=HS_STAGE,
            identity=identity + ":hs",
            request=request.model_dump(mode="json"),
            model=hs_output_type(frozen, hs_codes),
            system=HS_SYSTEM,
            prompt=hs_prompt(request),
            provider=provider,
        )
        if validate_hs(frozen, output, hs_codes).hs6 != code:
            raise ValueError("frozen synthetic HS differs from the received native proposal")
    expected = known_states.get(record_id)
    form_evidence = frozen.get("nativeDgPhysicalEvidence")
    if expected is not None:
        if form_evidence is not None:
            raise ValueError("known DG physical form must not carry a redundant native proposal")
    else:
        if not isinstance(form_evidence, Mapping):
            raise ValueError("DG physical form requires native physical evidence")
        request = form_request(frozen)
        output = verify_native_evidence(
            root,
            form_evidence,
            stage=FORM_STAGE,
            identity=identity + ":form",
            request=request.model_dump(mode="json"),
            model=PhysicalProposal,
            system=FORM_SYSTEM,
            prompt=form_prompt(request),
            provider=provider,
        )
        validate_form(frozen, output, known_states)
    physical_form(frozen, known_states)
