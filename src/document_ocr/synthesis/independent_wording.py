"""Reviewed complete-phrase ownership and flat-category wording projection.

Certificates and review evidence are explicit pinned inputs. No source identity or
historical experiment artifact is implicitly selected by this module.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import (
    Occurrence,
    Variable,
    compile_contract,
    digest,
    locate,
)
from document_ocr.synthesis.curated_descriptions import (
    MutableRegion,
    normalized_description,
    validate_description_regions,
)
from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint
from document_ocr.synthesis.curated_wording import WordingRequest

META = "independentWordingCertificate"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Span(Strict):
    key: str
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    source: str = Field(min_length=1)
    sourceSha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(min_length=1)


class SurfaceOperation(Span):
    kind: Literal["surface"]
    replacement: str


class ProductOperation(Span):
    kind: Literal["complete_product"]
    targetPath: str = Field(pattern=r"^documentPatch\.goodsItemDetails\[\d+\]\.description$")
    meaning: str = Field(min_length=1)


Operation = Annotated[SurfaceOperation | ProductOperation, Field(discriminator="kind")]


class Certificate(Strict):
    sourceSha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewStatus: Literal["reviewed"]
    operations: tuple[Operation, ...]
    thermalReview: bool
    countedMaterialReview: bool


def load_certificates(path: Path, *, expected_sha256: str) -> dict[str, Certificate]:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("wording certificate file SHA-256 differs")
    data = json.loads(path.read_text())
    if data["version"] != 1:
        raise ValueError("unsupported wording certificate version")
    return {sid: Certificate.model_validate(value) for sid, value in data["sources"].items()}


def _span(bp: SamplingBlueprint, op: Span) -> list[RenderRegion]:
    if op.end - op.start != len(op.source.encode()):
        raise ValueError("certified span byte length differs")
    if bp.source.encode()[op.start : op.end] != op.source.encode():
        raise ValueError("source occurrence differs from the certified complete phrase")
    if digest(op.source.encode()) != op.sourceSha256:
        raise ValueError("complete phrase SHA-256 differs")
    overlap = [r for r in bp.regions if r.start < op.end and op.start < r.end]
    if any(r.start < op.start or op.end < r.end for r in overlap):
        raise ValueError("complete phrase partially intersects an existing owner")
    return overlap


def _remove_historical(
    bp: SamplingBlueprint, overlap: list[RenderRegion]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[RenderRegion]]:
    removed = {(r.key, r.occurrence) for r in overlap}
    bindings = copy.deepcopy(dict(bp.historical_bindings))
    ownership = copy.deepcopy(dict(bp.ownership_data))
    regions = [r for r in bp.regions if (r.key, r.occurrence) not in removed]
    for key in {r.key for r in overlap if r.curated_key is None}:
        if key not in bindings:
            raise ValueError("displaced historical owner lacks its complete binding")
        keep = [i for i in range(len(bindings[key]["occurrences"])) if (key, i) not in removed]
        if keep:
            remap = {old: new for new, old in enumerate(keep)}
            bindings[key]["occurrences"] = [bindings[key]["occurrences"][i] for i in keep]
            regions = [
                replace(r, occurrence=remap[r.occurrence]) if r.key == key else r for r in regions
            ]
        else:
            del bindings[key]
            for stage in ("surfaces", "delete", "constants", "identifier_privacy"):
                ownership.get(stage, {}).pop(key, None)
            if "omit" in ownership:
                ownership["omit"] = [item for item in ownership["omit"] if item["key"] != key]
    return bindings, ownership, regions


def _surface(bp: SamplingBlueprint, op: SurfaceOperation) -> SamplingBlueprint:
    overlap = _span(bp, op)
    if any(r.curated_key is not None or r.target_paths for r in overlap):
        raise ValueError("optional caption/suffix operation cannot displace a target owner")
    bindings, ownership, regions = _remove_historical(bp, overlap)
    if op.key in bindings or any(r.key == op.key for r in regions):
        raise ValueError("complete surface owner key already exists")
    bindings[op.key] = {
        "logical_key": op.key,
        "value_kind": "other_text",
        "target_paths": [],
        "occurrences": [{"byte_start": op.start, "byte_end": op.end, "source_text": op.source}],
    }
    regions.append(
        RenderRegion(
            op.key,
            0,
            op.start,
            op.end,
            op.source,
            (),
            "auxiliary",
            allow_empty=op.replacement == "",
        )
    )
    return replace(
        bp,
        regions=tuple(sorted(regions, key=lambda r: r.start)),
        historical_bindings=bindings,
        ownership_data=ownership,
    )


def _product(bp: SamplingBlueprint, op: ProductOperation) -> SamplingBlueprint:
    overlap = _span(bp, op)
    if bp.description_blocks is None:
        raise ValueError("complete product requires approved description blocks")
    spans = [s for f in bp.description_blocks.fields if f.path == op.targetPath for s in f.spans]
    if len(spans) != 1 or (spans[0].start, spans[0].end) != (op.start, op.end):
        raise ValueError("complete product must equal its unchanged approved description block")
    bindings, ownership, regions = _remove_historical(bp, overlap)
    variables, removed_keys, remaps = [], set(), {}
    for var in bp.contract.variables:
        kept = [
            i
            for i, occ in enumerate(var.occurrences)
            if not (op.start <= locate(bp.source, occ)[0] and locate(bp.source, occ)[1] <= op.end)
        ]
        if len(kept) == len(var.occurrences):
            variables.append(var)
            continue
        if var.kind == "product" and kept:
            raise ValueError("complete product cannot retire a repeated product only partially")
        if kept:
            remaps[var.key] = {old: new for new, old in enumerate(kept)}
            variables.append(
                var.model_copy(update={"occurrences": [var.occurrences[i] for i in kept]})
            )
        else:
            removed_keys.add(var.key)
    if any(v.key == op.key for v in variables):
        raise ValueError("new complete product key conflicts with another current variable")
    occurrence_index = bp.source.encode()[: op.start].count(op.source.encode()) + 1
    variables.append(
        Variable(
            key=op.key,
            kind="product",
            value=normalized_description(op.source),
            meaning=op.meaning,
            required_literals=[],
            occurrences=[
                Occurrence(text=op.source, occurrence=occurrence_index, presentation="text")
            ],
        )
    )
    targets = [
        t.model_copy(update={"expression": "{" + op.key + "}"}) if t.path == op.targetPath else t
        for t in bp.contract.targets
    ]
    for target in targets:
        if target.path != op.targetPath and any(
            "{" + key + "}" in target.expression for key in removed_keys
        ):
            raise ValueError("removed product still owns another public target")
    contract = bp.contract.model_copy(update={"variables": variables, "targets": targets})
    regions = [
        replace(
            r,
            occurrence=remaps[r.key][r.occurrence],
            target_paths=tuple(path for path in r.target_paths if path != op.targetPath),
        )
        if r.curated_key in remaps
        else r
        for r in regions
    ]
    regions.append(
        RenderRegion(
            op.key,
            0,
            op.start,
            op.end,
            op.source,
            (op.targetPath,),
            "product",
            curated_key=op.key,
            baseline=normalized_description(op.source),
        )
    )
    result = replace(
        bp,
        contract=contract,
        regions=tuple(sorted(regions, key=lambda r: r.start)),
        historical_bindings=bindings,
        ownership_data=ownership,
    )
    compile_contract(
        {"documentId": bp.document_id, "joinedRawText": bp.source, "target": bp.target},
        contract,
        description_blocks=bp.description_blocks,
    )
    return result


def prepare_blueprint(
    bp: SamplingBlueprint, certificate: Certificate | Mapping[str, Any]
) -> SamplingBlueprint:
    cert = (
        certificate
        if isinstance(certificate, Certificate)
        else Certificate.model_validate(certificate)
    )
    if digest(bp.source.encode()) != cert.sourceSha256:
        raise ValueError("wording certificate source SHA-256 differs")
    if META in bp.ownership_data:
        raise ValueError("wording certificate was already applied")
    result = bp
    for op in cert.operations:
        result = _surface(result, op) if isinstance(op, SurfaceOperation) else _product(result, op)
    validate_description_regions(
        result.description_blocks, cast(tuple[MutableRegion, ...], result.regions)
    )
    if result.target != bp.target or result.description_blocks != bp.description_blocks:
        raise ValueError(
            "wording ownership changed source public targets or description boundaries"
        )
    if set(result.inventory()["unownedTargetPaths"]) != set(bp.inventory()["unownedTargetPaths"]):
        raise ValueError("wording ownership changed public target coverage")
    ownership = copy.deepcopy(dict(result.ownership_data))
    ownership[META] = cert.model_dump(mode="json")
    return replace(result, ownership_data=ownership)


def phrase_surfaces(bp: SamplingBlueprint) -> dict[str, str]:
    """Exact complete optional surfaces; no sampled numeric facts are changed."""
    cert = Certificate.model_validate(bp.ownership_data[META])
    return {op.key: op.replacement for op in cert.operations if isinstance(op, SurfaceOperation)}


def prepare_request(
    request: WordingRequest, bp: SamplingBlueprint, thermal_state: str | None = None
) -> WordingRequest:
    cert = Certificate.model_validate(bp.ownership_data[META])
    complete = {op.key: op.meaning for op in cert.operations if isinstance(op, ProductOperation)}
    fields = tuple(
        replace(f, requirement=f.requirement + " Complete owned passage: " + complete[f.key])
        if f.key in complete
        else f
        for f in request.fields
    )
    if not set(complete) <= {f.key for f in fields if f.role == "goods description"}:
        raise ValueError("complete product is missing from the actual wording request")
    context = request.context
    if cert.thermalReview:
        if thermal_state not in {"chilled", "frozen"}:
            raise ValueError("thermal wording requires its explicit sampled cold-chain state")
        context += (
            "\nSAME-CARGO STATE AUTHORITY: "
            + thermal_state.upper()
            + ". Describe this same commodity consistently in that sampled state. "
            "Do not add a contradictory preservation/process state such as sun-dried "
            "for the same fresh/chilled product. Packaging media (for example dry ice), "
            "unrelated specifications and genuinely separate commodities are not "
            "contradictions merely because a word contains DRY. State uncertainty "
            "requires review; do not erase useful specifications blindly."
        )
    if cert.countedMaterialReview:
        context += (
            "\nCOUNTED-UNIT MATERIAL REVIEW: distinguish the declared counted package "
            "from separate inner receptacles and supports. Preserve the selected "
            "counted package category; do not infer its material from cargo or an "
            "inner/support package. An inner steel drum inside a counted wooden box "
            "does not change the box."
        )
    return replace(request, fields=fields, context=context)
