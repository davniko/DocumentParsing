"""Bare-article component estimates; code owns manifest counts and summation."""

import json
from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from document_ocr.synthesis.independent_native_evidence import verify_native_evidence
from document_ocr.synthesis.independent_unit_profiles import (
    UnitProfile,
    UnitProfileRequest,
    validate_profile_context,
)

STAGE = "independent-article-component-profile-v1"
SYSTEM = """Estimate THREE plausible examples of the fixed bare-vehicle manifest.
Each named field is ONE physical article, not a SET, group or whole shipment.
Estimate that single car/motorbike's gross cargo mass in kg and its shipping
bounding volume in cubic metres (not engine displacement, trunk or cabin volume).
Keep ALL component roles and the supplied DG propulsion. Vary realistic models
and sizes between examples, not the number of articles. Code sums the fixed
components and groups; do not perform or print shipment accounting yourself.
No packaging/support mass is supplied. Exclude carrying-container hardware.
Each complete example must fit the supplied carrying payload and volume bounds.
Do not miniaturize vehicles to fit. UNKNOWN=true requires examples=[]; otherwise
return exactly three complete examples. Basis: one short sentence, <=300 chars.
No product prose, HS/UN/class resampling, new fields or shipping approval.
"""


def component_roles(request: UnitProfileRequest) -> dict[str, str]:
    if request.articleGroups is None:
        raise ValueError("component estimation requires explicit accounting groups")
    counts: Counter[str] = Counter()
    result = {}
    for group in request.articleGroups:
        for role in group:
            counts[role] += 1
            result[role.replace(" ", "_") + "_" + str(counts[role])] = role
    return result


class ArticleProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    unknown: bool
    examples: list[Any]
    basis: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def cardinality(self):
        if len(self.examples) != (0 if self.unknown else 3):
            raise ValueError("known component profiles need three complete examples")
        return self


def article_output_type(request: UnitProfileRequest) -> type[ArticleProfile]:
    part = create_model(
        "BoundedBareArticle",
        __config__=ArticleProfile.model_config,
        grossKg=(float, Field(gt=0, le=request.payloadKg, allow_inf_nan=False)),
        occupiedVolumeM3=(float, Field(gt=0, le=request.volumeBudgetM3, allow_inf_nan=False)),
    )
    manifest = create_model(
        "FixedArticleManifest",
        __config__=ArticleProfile.model_config,
        **{key: (part, ...) for key in component_roles(request)},
    )
    return create_model(
        "BareArticleProfiles",
        __base__=ArticleProfile,
        examples=(list[manifest], Field(max_length=3)),
    )


def article_prompt(request: UnitProfileRequest) -> str:
    return json.dumps(
        {
            "authoritativeIdentity": request.dangerousGoods,
            "components": component_roles(request),
            "accountingGroups": request.articleGroups,
            "equipment": request.equipment,
            "completeManifestPayloadKg": request.payloadKg,
            "completeManifestVolumeM3": request.volumeBudgetM3,
            "reviewFeedback": request.reviewFeedback,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def normalize_article_profile(
    request: UnitProfileRequest, output: Mapping[str, Any]
) -> UnitProfile:
    received = article_output_type(request).model_validate(output).model_dump(mode="json")
    examples = []
    for example in received["examples"]:
        gross = sum((Decimal(str(part["grossKg"])) for part in example.values()), Decimal(0))
        cube = sum(
            (Decimal(str(part["occupiedVolumeM3"])) for part in example.values()), Decimal(0)
        )
        if gross > Decimal(str(request.payloadKg)) or cube > Decimal(str(request.volumeBudgetM3)):
            raise ValueError("summed article components exceed complete carrying capacity")
        examples.append(
            {"grossKg": float(gross), "occupiedVolumeM3": float(cube), "packagingTareFraction": 0.0}
        )
    return UnitProfile.model_validate(
        {"unknown": received["unknown"], "examples": examples, "basis": received["basis"]}
    )


def verify_article_evidence(
    root: Path,
    frozen: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    physical_states: Mapping[str, str],
) -> None:
    evidence = frozen["nativeArticleProfileEvidence"]
    request = UnitProfileRequest.model_validate(evidence["request"])
    validate_profile_context(request, frozen, physical_states)
    output = verify_native_evidence(
        root,
        evidence,
        stage=STAGE,
        identity=f"{frozen['sourceDocumentId']}:{frozen['variant']}",
        request=request.model_dump(mode="json"),
        model=article_output_type(request),
        system=SYSTEM,
        prompt=article_prompt(request),
        provider=provider,
    )
    if normalize_article_profile(request, output).unknown:
        raise ValueError("UNKNOWN article profile has no source/donor fallback")
