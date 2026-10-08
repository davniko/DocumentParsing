"""Direct V7 extraction, one scoped correction and non-editing final review.

No dataset mutation or gold publication happens here. OCR is always a text message;
only candidates/schemas/findings use JSON. Code owns receipts and stage transitions.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable
from copy import deepcopy
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from types import GenericAlias, UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_ai import Agent, BinaryContent, NativeOutput, RunContext, capture_run_messages
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.output import OutputContext
from pydantic_ai.usage import UsageLimits
from pydantic_core import to_jsonable_python

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.labeling_agents.direct_cargo import (
    CargoAccountingValues,
    CargoFactReview,
    CargoRelationResponse,
    CargoSourceMap,
    candidate_cargo_view,
    cargo_numeric_findings,
    map_grounding_errors,
)
from document_ocr.labeling_agents.direct_grounding import source_fidelity_findings
from document_ocr.labeling_agents.direct_models import (
    SECTION_FIELDS,
    SECTION_MODELS,
    SECTION_PRIORITIES,
    CorrectionDecision,
    CorrectionHold,
    DirectLabelingConfig,
    LayoutRequest,
    ReviewFinding,
    Section,
    SectionProjectionV7,
    SectionReview,
)
from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories
from document_ocr.labeling_agents.target_normalization import normalize_target_casing
from document_ocr.semantic_v3.transform import CategoryRegistry

OutputModel = TypeVar("OutputModel", bound=BaseModel)
Draft = BillOfLadingExtractionV7Label | dict[str, Any]


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def combine_reviews(*reviews: SectionReview) -> SectionReview:
    """Keep every distinct defect; no clean reviewer can override another's finding."""
    findings = {encoded(f.model_dump(mode="json")): f for r in reviews for f in r.findings}
    return SectionReview(
        status=(
            "unresolved"
            if any(f.issue == "ambiguous" for f in findings.values())
            else "corrections_needed"
            if findings
            else "pass"
        ),
        explanation="\n".join(r.explanation for r in reviews),
        findings=list(findings.values()),
    )


def normalize_optional_objects(annotation: Any, value: Any) -> Any:
    """Represent empty nullable fact objects as null, without deleting any facts.

    Apply before JSON validation, not in a Pydantic before-validator: recreating
    Python dicts there loses strict JSON-mode date/tuple validation. Unknown keys,
    required facts, empty arrays and empty list members remain validation errors.
    """
    args = get_args(annotation)
    nullable = type(None) in args
    candidates = (
        [arg for arg in args if arg is not type(None)]
        if get_origin(annotation) in (Union, UnionType)
        else [annotation]
    )
    if len(candidates) > 1 and isinstance(value, dict):
        # Correction responses have distinct section, hold and layout-request keys.
        # An ambiguous union is left untouched for normal Pydantic validation.
        candidates = [
            model
            for model in candidates
            if isinstance(model, type)
            and issubclass(model, BaseModel)
            and value.keys() <= model.model_fields.keys()
        ]
    if len(candidates) != 1:
        return value
    model = candidates[0]
    if get_origin(model) in (tuple, list) and isinstance(value, list):
        return [normalize_optional_objects(get_args(model)[0], item) for item in value]
    if (
        not isinstance(model, type)
        or not issubclass(model, BaseModel)
        or not isinstance(value, dict)
    ):
        return value
    fields = model.model_fields
    normalized = {
        key: normalize_optional_objects(fields[key].annotation, child) if key in fields else child
        for key, child in value.items()
    }
    if (
        nullable
        and not (normalized.keys() - fields.keys())
        and all(child is None for child in normalized.values())
        and all(type(None) in get_args(field.annotation) for field in fields.values())
        and all(not field.is_required() or name in normalized for name, field in fields.items())
    ):
        return None
    return normalized


def _absence_receipts(raw: Any, canonical: Any, path: str = "") -> list[dict[str, Any]]:
    """Record null-object canonicalization separately from the untouched provider text."""
    if isinstance(raw, dict) and canonical is None:
        return [{"path": path, "before": raw, "after": None}]
    if isinstance(raw, dict) and isinstance(canonical, dict):
        return [
            change
            for key, value in raw.items()
            if key in canonical
            for change in _absence_receipts(value, canonical[key], f"{path}.{key}".lstrip("."))
        ]
    if isinstance(raw, list) and isinstance(canonical, list):
        return [
            change
            for index, (old, new) in enumerate(zip(raw, canonical, strict=True))
            for change in _absence_receipts(old, new, f"{path}[{index}]")
        ]
    return []


def described_schema(output_model: type[BaseModel], registry: CategoryRegistry) -> dict[str, Any]:
    """Export the actual provider schema with the pinned package tokens and their meanings."""
    schema = output_model.model_json_schema()
    container = schema.get("$defs", {}).get("ContainerInformationV7")
    if container is not None:
        # Native JSON Schema does not infer Python model validators. Express the
        # category-or-printed-text contract with supported nested anyOf,
        # avoiding if/then and keeping the emitted label shape unchanged.
        canonical, printed = deepcopy(container), deepcopy(container)
        for field in ("sizeCategory", "typeCategory"):
            definition = canonical["properties"][field]
            # Known family and known dimensions are independent extraction facts.
            # Keep the nullable schema; a missing length must not force a guess.
            printed["properties"][field] = {
                "type": "null",
                "description": definition["description"],
            }
        canonical["properties"]["typeDescription"] = {
            "type": "null",
            "description": container["properties"]["typeDescription"]["description"],
        }
        schema["$defs"]["ContainerInformationV7"] = {
            "description": container["description"],
            "anyOf": [canonical, printed],
        }
    package = schema.get("$defs", {}).get("PackagesV7")
    if package is not None:
        field = package["properties"]["typeCategory"]
        field["anyOf"] = [
            {"type": "string", "enum": sorted(e.categoryToken for e in registry.entries)},
            {"type": "null"},
        ]
        # Current tokens are readable names (PACKAGE_DRUM_STEEL = Drum, steel).
        # Repeat a display meaning only if it adds information to the enum token.
        extra_meanings = [
            f"{e.categoryToken}={e.displayName}"
            for e in registry.entries
            if e.categoryToken
            != "PACKAGE_" + re.sub(r"[^A-Z0-9]+", "_", e.displayName.upper()).strip("_")
        ]
        field["description"] += " Vocabulary: PACKAGE_ tokens spell the package type."
        if extra_meanings:
            field["description"] += " Additional meanings: " + "; ".join(extra_meanings)
    return schema


class _RegistryBoundOutput(AbstractCapability[None]):
    """Bind the package vocabulary without expanding reusable Pydantic schema definitions."""

    def __init__(self, schema: dict[str, Any], output_model: type[BaseModel]) -> None:
        self.schema = schema
        self.output_model = output_model

    async def before_output_validate(
        self, ctx: RunContext[None], *, output_context: OutputContext, output: Any
    ) -> Any:
        if not isinstance(output, str):
            return output
        # Keep raw provider text in the receipt; validate normalized JSON in JSON mode.
        try:
            raw = json.loads(output)
        except json.JSONDecodeError:
            return output  # Pydantic reports the malformed response; never accept it.
        normalized = normalize_optional_objects(self.output_model, raw)
        return encoded(normalized) if normalized != raw else output

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        parameters = request_context.model_request_parameters
        if parameters.output_object is None:
            raise ValueError("direct labeling requires native structured output")
        return replace(
            request_context,
            model_request_parameters=replace(
                parameters,
                output_object=replace(parameters.output_object, json_schema=self.schema),
            ),
        )


class ResponseEnvelope(LabelSchemaModel):
    """Typed section response, specialized to the allowed responses for each stage."""

    response: BaseModel = Field(description="Section result or explicit request for assistance.")


ReviewResponse = create_model(
    "ReviewResponse",
    __base__=ResponseEnvelope,
    __doc__="One section review or a request for PDF layout context.",
    response=(
        SectionReview | LayoutRequest,
        Field(
            description=(
                "Return the review, or request layout pages only when OCR ownership is ambiguous."
            )
        ),
    ),
)

CargoFactResponse = create_model(
    "CargoFactResponse",
    __base__=ResponseEnvelope,
    __doc__="Cargo facts review or explicit request for layout.",
    response=(
        CargoFactReview | LayoutRequest,
        Field(
            description=(
                "Review only product facts; the separate cargo relationship review "
                "owns grouping, package counts and allocations."
            )
        ),
    ),
)


def correction_paths(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Object leaves and atomic list fields in the typed correction value contract."""
    paths = []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}".lstrip(".")
        nested = next(
            (
                a
                for a in (field.annotation, *get_args(field.annotation))
                if isinstance(a, type) and issubclass(a, BaseModel)
            ),
            None,
        )
        paths.extend(correction_paths(nested, path) if nested else [path])
    return paths


def correction_model(sections: tuple[Section, ...]) -> type[ResponseEnvelope]:
    """One typed correction for connected findings, including both ends of a relocation."""
    if not sections or len(set(sections)) != len(sections):
        raise ValueError("correction scope must be nonempty and unique")
    fields: dict[str, Any] = {
        name: (field.annotation, deepcopy(field))
        for section in sections
        for name, field in SECTION_MODELS[section].model_fields.items()
    }
    values = create_model(
        "CorrectedValues",
        __base__=LabelSchemaModel,
        __doc__="Complete values for precisely the assigned extraction sections.",
        **fields,
    )
    decision = create_model(
        "ScopedCorrectionDecision",
        __base__=CorrectionDecision,
        changedFields=(
            GenericAlias(list, Literal.__getitem__(tuple(correction_paths(values)))),
            deepcopy(CorrectionDecision.model_fields["changedFields"]),
        ),
    )
    audit_fields: dict[str, Any] = {
        "decisions": (
            GenericAlias(list, decision),
            Field(description="One decision per finding ID."),
        ),
        "values": (
            values,
            Field(description="Complete assigned section values after adjudication."),
        ),
        **(
            {
                "mapCorrection": (
                    CargoSourceMap | None,
                    Field(
                        default=None,
                        description=(
                            "Complete corrected cargo source map if its ownership or numeric "
                            "interpretation caused the defect; otherwise null. Map and target "
                            "must agree with OCR and PDF layout, not merely with each other."
                        ),
                    ),
                )
            }
            if "cargo" in sections
            else {}
        ),
    }
    audited = create_model(
        "AuditedCorrection",
        __base__=LabelSchemaModel,
        __doc__="Explicit adjudication of findings and the resulting extraction values.",
        **audit_fields,
    )
    return create_model(
        "SectionCorrection",
        __base__=ResponseEnvelope,
        __doc__="Corrected section, unresolved conflict, or layout request.",
        response=(
            audited | CorrectionHold | LayoutRequest,
            Field(
                description=(
                    "Audited decisions and corrected values, not edit operations. Preserve "
                    "other supported facts; request layout only when needed."
                )
            ),
        ),
    )


def connected_sections(edges: list[tuple[Section, ...]]) -> list[tuple[Section, ...]]:
    """Stable connected components; a relocation is never committed in separate halves."""
    groups: list[set[Section]] = []
    for edge in edges:
        merged = set(edge)
        untouched = []
        for group in groups:
            if group & merged:
                merged.update(group)
            else:
                untouched.append(group)
        groups = [*untouched, merged]
    return sorted(
        (tuple(s for s in SECTION_FIELDS if s in group) for group in groups),
        key=lambda group: tuple(SECTION_FIELDS).index(group[0]),
    )


def draft_value(target: Draft) -> dict[str, Any]:
    """Copy an untrusted V7 draft without pretending it passed fact validation."""
    if isinstance(target, BillOfLadingExtractionV7Label):
        return target.canonical_target()
    value: dict[str, Any] = json.loads(encoded(target))
    if (
        not isinstance(value, dict)
        or set(value) != {"schemaVersion", "documentPatch"}
        or value["schemaVersion"] != "7.0.0"
        or not isinstance(value["documentPatch"], dict)
        or not value["documentPatch"].keys() <= {k for v in SECTION_FIELDS.values() for k in v}
    ):
        raise ValueError("expected a V7 draft envelope with known section fields")
    return value


def section_values(target: Draft, section: Section) -> dict[str, Any]:
    patch = draft_value(target)["documentPatch"]
    return {key: patch[key] for key in SECTION_FIELDS[section] if key in patch}


def _review_changes(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    """Exact change context; lists stay whole because entity order/grouping may change."""
    if before == after:
        return []
    if (isinstance(before, dict) or before is None) and (isinstance(after, dict) or after is None):
        before, after = before or {}, after or {}
        return [
            change
            for key in sorted(before.keys() | after.keys())
            for change in _review_changes(
                before.get(key), after.get(key), f"{path}.{key}".lstrip(".")
            )
        ]
    return [{"field": path, "before": before, "after": after}]


def correction_change_error(
    before: dict[str, Any], after: dict[str, Any], decisions: list[dict[str, Any]]
) -> str | None:
    """Reject undeclared side effects; semantic justification remains a review task."""
    actual = {change["field"] for change in _review_changes(before, after)}
    declared = [field for decision in decisions for field in decision["changedFields"]]
    if any(d["disposition"] == "reject" and d["changedFields"] for d in decisions):
        return "rejected findings cannot authorize changes"
    if not actual <= set(declared):
        return encoded(
            {
                "undeclaredChanges": sorted(actual - set(declared)),
            }
        )
    return None


def merge_sections(
    target: Draft,
    replacements: dict[Section, BaseModel],
    *,
    validation_scope: tuple[Section, ...],
) -> dict[str, Any]:
    """Replace and validate one dependent group, preserving unrelated draft defects."""
    if not replacements.keys() <= set(validation_scope):
        raise ValueError("replacement lies outside its validation scope")
    value = draft_value(target)
    patch = value["documentPatch"]
    for section, replacement in replacements.items():
        checked = SECTION_MODELS[section].model_validate_json(replacement.model_dump_json())
        for key in SECTION_FIELDS[section]:
            patch.pop(key, None)
        patch.update(checked.model_dump(mode="json", exclude_none=True))
        # Nullable instruction decisions are explicit target values, unlike absent facts.
        if section == "metadata_freight":
            patch["negotiability"] = checked.negotiability
        if section == "parties":
            for party in patch.get("parties", {}).get("notifyParties", []):
                party.setdefault("sameAs", None)
    projection = {}
    for section in validation_scope:
        checked = SECTION_MODELS[section].model_validate_json(
            encoded(section_values(value, section))
        )
        projection.update(checked.model_dump(mode="json", exclude_none=True))
    if projection:
        for party in projection.get("parties", {}).get("notifyParties", []):
            party.setdefault("sameAs", None)
        SectionProjectionV7.model_validate_json(encoded(projection))
    return value


def _layout_hold(question: str, detail: str) -> SectionReview:
    return SectionReview.model_validate(
        {
            "status": "unresolved",
            "explanation": f"{question}: {detail}",
            "findings": [
                {"field": "section", "issue": "ambiguous", "explanation": f"{question}: {detail}"}
            ],
        }
    )


def render_pdf_pages(path: Path, pages: list[int]) -> list[bytes]:
    """Render only requested pages at 150 DPI for layout review, with bounded dimensions."""
    import pypdfium2 as pdfium

    images = []
    with pdfium.PdfDocument(path) as pdf:
        if any(page > len(pdf) for page in pages):
            raise ValueError(f"requested page exceeds PDF page count {len(pdf)}")
        for number in pages:
            page = pdf[number - 1]
            try:
                width, height = page.get_size()
                # Layout-only assistance: cap long side at 2500 px to bound RAM/provider input.
                scale = min(150 / 72, 2500 / max(width, height))
                bitmap = page.render(scale=scale)
                try:
                    with bitmap.to_pil() as image:
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        images.append(buffer.getvalue())
                finally:
                    bitmap.close()
            finally:
                page.close()
    return images


class DirectLabelingFlow:
    """One document per instance, with explicit extraction and refinement entry points."""

    def __init__(
        self,
        *,
        model: Model,
        config: DirectLabelingConfig,
        project_root: Path,
        output_dir: Path,
        ocr: str,
        pdf_path: Path | None = None,
        request_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        if not ocr.strip():
            raise ValueError("OCR must not be empty")
        registry_bytes = (project_root / config.package_registry).read_bytes()
        if sha256_bytes(registry_bytes) != config.package_registry_sha256:
            raise ValueError("package registry hash mismatch")
        self.registry = CategoryRegistry.model_validate_json(registry_bytes)
        if self.registry.registryKind != "package":
            raise ValueError("expected package registry")
        self.package_tokens = {entry.categoryToken for entry in self.registry.entries}
        self.config, self.model = config, model
        self.ocr, self.pdf_path = ocr, pdf_path
        self.output_dir = output_dir
        self.prompts = {
            stage: (project_root / f"prompts/labeling_agents/direct_{stage}.md").read_text()
            for stage in (
                "extractor",
                "reviewer",
                "cargo_mapper",
                "relations_reviewer",
                "corrector",
            )
        }
        self.semaphore = request_semaphore or asyncio.Semaphore(config.concurrency)
        self.receipts: list[dict[str, Any]] = []
        self.layout_sections: set[Section] = set()
        self._layout_pages: dict[Section, list[int]] = {}
        self._sequence = 0
        self._cargo_source: CargoSourceMap | None = None
        self._cargo_draft: CargoSourceMap | None = None
        self._cargo_map_errors: list[str] = []
        self._cargo_map_attempted = False
        # A new directory is mandatory: no overwriting, stale-result reuse or implicit resume.
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_publish_bytes(output_dir / "ocr.txt", ocr.encode())
        atomic_publish_json(output_dir / "config.json", config.model_dump(mode="json"))
        atomic_publish_json(
            output_dir / "schema.json", self.output_schema(BillOfLadingExtractionV7Label)
        )
        for stage, prompt in self.prompts.items():
            atomic_publish_bytes(output_dir / f"{stage}.md", prompt.encode())

    def output_schema(self, output_model: type[BaseModel]) -> dict[str, Any]:
        """Pydantic is authoritative; bind only the external package vocabulary at runtime."""
        return described_schema(output_model, self.registry)

    def _check_package_tokens(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "numberAndTypeOfPackages" and child:
                    for package in child:
                        token = package.get("typeCategory")
                        if token is not None and token not in self.package_tokens:
                            raise ValueError(f"unregistered package category: {token}")
                self._check_package_tokens(child)
        elif isinstance(value, list):
            for child in value:
                self._check_package_tokens(child)

    async def _call(
        self,
        stage: str,
        output_model: type[OutputModel],
        *,
        context: str = "",
        attachments: list[bytes] | None = None,
        complete_pdf: bool = False,
        audit: bool = False,
    ) -> OutputModel:
        schema = self.output_schema(output_model)
        # Native output receives this exact Pydantic-derived schema, including descriptions.
        effort = (
            self.config.audit_reasoning_effort
            if audit and self.config.audit_reasoning_effort is not None
            else self.config.reasoning_effort
        )
        settings = OpenAIResponsesModelSettings(
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout_seconds,
            openai_reasoning_effort=effort,
        )
        agent: Agent[None, OutputModel] = Agent(
            self.model,
            output_type=NativeOutput(output_model, strict=True),
            capabilities=[_RegistryBoundOutput(schema, output_model)],
            instructions=self.prompts[stage],
            retries=0,
            model_settings=settings,
        )

        @agent.output_validator
        def validate_output(_ctx: Any, answer: OutputModel) -> OutputModel:
            self._check_package_tokens(answer.model_dump(mode="json"))
            return answer

        # Raw OCR is a separate text message part, never a JSON-encoded string or line inventory.
        content: list[Any] = ["Complete OCR source follows:", self.ocr]
        if context:
            content.append(context)
        if attachments:
            content.append("Requested PDF pages: layout/context only; OCR remains value authority.")
            content.extend(BinaryContent(data=b, media_type="image/png") for b in attachments)
        pdf_digest = None
        if complete_pdf:
            if self.pdf_path is None:
                raise ValueError("complete PDF required for cargo association review")
            # Responses accepts a whole PDF (text plus page images). Never select/truncate pages.
            if self.pdf_path.stat().st_size >= 50_000_000:
                raise ValueError("PDF exceeds the provider's 50 MB request limit")
            pdf_bytes = self.pdf_path.read_bytes()
            pdf_digest = sha256_bytes(pdf_bytes)
            content.extend(
                [
                    "Complete source PDF: layout/ownership and page continuity only. The "
                    "separately "
                    "supplied OCR text is the sole authority for label values; PDF-only values "
                    "must stay absent.",
                    BinaryContent(data=pdf_bytes, media_type="application/pdf"),
                ]
            )
        async with self.semaphore:
            self._sequence += 1
            number = self._sequence
            request = {
                "stage": stage,
                "model": self.config.model,
                "outputMode": "native_json_schema",
                "strict": True,
                "reasoningEffort": effort,
                "ocrSha256": sha256_bytes(self.ocr.encode()),
                "promptSha256": sha256_bytes(self.prompts[stage].encode()),
                "schema": schema,
                "context": context,
                "imageSha256": [sha256_bytes(b) for b in attachments or []],
                "completePdfSha256": pdf_digest,
            }
            atomic_publish_json(self.output_dir / "calls" / f"{number:03d}-request.json", request)
            start = time.perf_counter()
            receipt: dict[str, Any] = {"call": number, "stage": stage}
            with capture_run_messages() as messages:
                try:
                    result = await agent.run(content, usage_limits=UsageLimits(request_limit=1))
                    receipt.update(
                        status="completed",
                        answer=result.output.model_dump(mode="json"),
                        usage=asdict(result.usage),
                    )
                    return result.output
                except BaseException as error:
                    receipt.update(
                        status="failed", errorType=type(error).__name__, diagnostic=str(error)
                    )
                    causes, seen = [], {id(error)}
                    cause = error.__cause__
                    while cause is not None and id(cause) not in seen:
                        seen.add(id(cause))
                        causes.append({"errorType": type(cause).__name__, "diagnostic": str(cause)})
                        cause = cause.__cause__
                    receipt["causes"] = causes
                    raise
                finally:
                    responses = [m for m in messages if isinstance(m, ModelResponse)]
                    receipt["responses"] = [
                        {
                            "provider": m.provider_name,
                            "model": m.model_name,
                            "responseId": m.provider_response_id,
                            "usage": asdict(m.usage),
                            "providerDetails": to_jsonable_python(m.provider_details),
                            "text": [
                                getattr(p, "content", None)
                                for p in m.parts
                                if p.part_kind == "text"
                            ],
                        }
                        for m in responses
                    ]
                    if receipt["status"] == "completed":
                        raw = json.loads("".join(receipt["responses"][-1]["text"]))
                        receipt["nullObjectNormalizations"] = _absence_receipts(
                            raw, receipt["answer"]
                        )
                    receipt["elapsedSeconds"] = time.perf_counter() - start
                    # No provider response means unknown billing, not a zero-cost assertion.
                    receipt["billingStatus"] = "usage_recorded" if responses else "unknown"
                    atomic_publish_json(
                        self.output_dir / "calls" / f"{number:03d}-result.json", receipt
                    )
                    self.receipts.append(receipt)

    async def extract(self) -> BillOfLadingExtractionV7Label:
        """One OCR-only call returning direct draft labels; no automatic review or training."""
        target = await self._call("extractor", BillOfLadingExtractionV7Label)
        normalized, equipment = reconcile_equipment_categories(
            target.canonical_target(), source_text=self.ocr
        )
        normalized, changes = normalize_target_casing(normalized)
        target = BillOfLadingExtractionV7Label.model_validate_json(encoded(normalized))
        atomic_publish_json(self.output_dir / "extraction-casing-edits.json", changes)
        atomic_publish_json(self.output_dir / "extraction-equipment-decisions.json", equipment)
        atomic_publish_bytes(
            self.output_dir / "target.json", (encoded(target.canonical_target()) + "\n").encode()
        )
        atomic_publish_json(
            self.output_dir / "status.json",
            {
                "status": "draft",
                "reviewed": False,
                "calls": len(self.receipts),
            },
        )
        return target

    def _section_context(
        self, target: Draft, section: Section, *, correction_scope: tuple[Section, ...] = ()
    ) -> str:
        # Preserve full OCR, but send only relevant candidate sections to reduce input overhead.
        related: dict[Section, tuple[Section, ...]] = {
            "parties": (),
            "metadata_freight": (),
            "cargo": ("equipment",),
            "equipment": ("cargo",),
            "route_transport": (),
        }
        candidate = section_values(target, section)
        context = (
            f"Assigned section: {section}. "
            + (
                "Joint correction scope: " + ", ".join(correction_scope) + ".\n"
                if correction_scope
                else "Only this section may be reviewed or replaced.\n"
            )
            + "Review priorities: "
            + SECTION_PRIORITIES[section]
            + "\n"
            + "Literal OCR checks (absence only; not proof of ownership/completeness):\n"
            + encoded(
                [
                    f.model_dump(mode="json")
                    for f in source_fidelity_findings(draft_value(target), self.ocr, section)
                ]
            )
            + "\n"
            + "Section field definitions:\n"
            + encoded(self.output_schema(SECTION_MODELS[section]))
            + "\nCandidate extraction (untrusted draft):\n"
            + encoded(candidate)
        )
        try:
            SECTION_MODELS[section].model_validate_json(
                encoded(normalize_optional_objects(SECTION_MODELS[section], candidate))
            )
        except ValidationError as error:
            context += "\nCandidate schema diagnostics (not correction evidence):\n" + encoded(
                error.errors(include_url=False, include_context=False, include_input=False)
            )
        if related[section]:
            context += "\nRead-only related extraction (for association checks):\n" + encoded(
                {
                    key: value
                    for other in related[section]
                    for key, value in section_values(target, other).items()
                }
            )
        return context

    async def _ensure_cargo_map(self) -> None:
        if self._cargo_map_attempted:
            return
        self._cargo_map_attempted = True
        if self.pdf_path is None:
            self._cargo_map_errors = [
                "Complete source PDF not supplied; cargo mapping cannot be certified."
            ]
            return
        try:
            source = await self._call("cargo_mapper", CargoSourceMap, complete_pdf=True)
        except UnexpectedModelBehavior as error:
            self._cargo_map_errors = [f"Cargo source mapping failed output validation: {error}"]
            atomic_publish_json(self.output_dir / "cargo-map-errors.json", self._cargo_map_errors)
            return
        self.layout_sections.add("cargo")
        self._cargo_draft = source
        self._cargo_map_errors = map_grounding_errors(source, self.ocr)
        atomic_publish_json(
            self.output_dir / "cargo-source-map.json", source.model_dump(mode="json")
        )
        atomic_publish_json(self.output_dir / "cargo-map-errors.json", self._cargo_map_errors)
        if not self._cargo_map_errors:
            self._cargo_source = source

    async def _review_cargo_relations(self, target: Draft, *, audit: bool) -> SectionReview:
        if self.pdf_path is None:
            return _layout_hold("cargo layout unavailable", "Complete source PDF is required")
        context = (
            "Independent source accounting (working interpretation, verify against source):\n"
            + encoded(self._cargo_draft.model_dump(mode="json") if self._cargo_draft else None)
            + "\nMap validation findings (repair the map when mistaken):\n"
            + encoded(self._cargo_map_errors)
            + "\nCargo field definitions:\n"
            + encoded(self.output_schema(CargoAccountingValues))
            + "\nCandidate cargo accounting:\n"
            + encoded(candidate_cargo_view(draft_value(target)))
        )
        answer = await self._call(
            "relations_reviewer",
            CargoRelationResponse,
            context=context,
            complete_pdf=True,
            audit=audit,
        )
        if answer.mapCorrection is not None:
            errors = map_grounding_errors(answer.mapCorrection, self.ocr)
            atomic_publish_json(
                self.output_dir / f"cargo-map-review-{self._sequence:03d}.json",
                {"proposal": answer.mapCorrection.model_dump(mode="json"), "errors": errors},
            )
            if errors:
                return combine_reviews(
                    answer.response, _layout_hold("cargo map correction invalid", "; ".join(errors))
                )
            self._cargo_source = self._cargo_draft = answer.mapCorrection
            self._cargo_map_errors = []
        review = SectionReview.model_validate_json(answer.response.model_dump_json())
        if self._cargo_map_errors or self._cargo_source is None:
            return combine_reviews(
                review,
                _layout_hold("cargo map validation failed", "; ".join(self._cargo_map_errors)),
            )
        return review

    async def _with_layout(
        self,
        stage: str,
        output_model: type[ResponseEnvelope],
        context: str,
        section: Section | tuple[Section, ...],
        *,
        audit: bool = False,
    ) -> Any:
        # The same section must not lose established layout context in correction
        # or re-review. Cache page numbers, not rasters, to bound retained memory.
        sections = (section,) if isinstance(section, str) else section
        if "cargo" in sections and stage == "corrector" and self.pdf_path is not None:
            context += (
                "\nIndependent cargo accounting (verify ownership; no PDF-only values):\n"
                + encoded(
                    self._cargo_draft.model_dump(mode="json")
                    if self._cargo_draft
                    else {"errors": self._cargo_map_errors}
                )
            )
            answer = (
                await self._call(
                    stage, output_model, context=context, complete_pdf=True, audit=audit
                )
            ).response
            if isinstance(answer, LayoutRequest):
                return _layout_hold(answer.question, "complete PDF was already supplied")
            return answer
        layout_scope = set(sections)
        if layout_scope.intersection({"cargo", "equipment"}):
            layout_scope.update({"cargo", "equipment"})
        pages = sorted({p for s in layout_scope for p in self._layout_pages.get(s, [])})
        if "route_transport" in sections and self.pdf_path is not None and not pages:
            # Misleading OCR headings can look certain; do not rely on a confidence trigger.
            pages = [1]
            self._layout_pages["route_transport"] = pages
            self.layout_sections.add("route_transport")
        images = None
        if pages:
            if self.pdf_path is None:
                raise ValueError("saved layout pages require their source PDF")
            images = render_pdf_pages(self.pdf_path, pages)
            context += "\nPreviously requested layout pages: " + encoded(pages)
        response = (
            await self._call(stage, output_model, context=context, attachments=images, audit=audit)
        ).response
        if not isinstance(response, LayoutRequest):
            return response
        if self.pdf_path is None:
            return _layout_hold(response.question, "no PDF supplied")
        try:
            # PDFium is synchronous and not used concurrently: small bounded renders avoid races.
            pages = sorted(set(pages) | set(response.pages))
            images = render_pdf_pages(self.pdf_path, pages)
        except (ValueError, OSError) as error:
            return _layout_hold(response.question, str(error))
        self.layout_sections.update(sections)
        for scope in sections:
            self._layout_pages[scope] = pages
        context += "\nLayout request: " + response.model_dump_json()
        final = (
            await self._call(stage, output_model, context=context, attachments=images, audit=audit)
        ).response
        if isinstance(final, LayoutRequest):
            return _layout_hold(final.question, "one layout assistance round exhausted")
        return final

    async def review(
        self,
        target: Draft,
        sections: tuple[Section, ...],
        *,
        previous: Draft | None = None,
    ) -> dict[Section, SectionReview]:
        async def one(section: Section) -> tuple[Section, SectionReview]:
            context = self._section_context(target, section)
            if previous is not None:
                context += (
                    "\nChanges in the last correction wave (audit both directions):\n"
                    + encoded(
                        _review_changes(
                            section_values(previous, section), section_values(target, section)
                        )
                    )
                )
            try:
                answer = await self._with_layout(
                    "reviewer",
                    CargoFactResponse if section == "cargo" else ReviewResponse,
                    context,
                    section,
                    audit=previous is not None,
                )
                answer = SectionReview.model_validate_json(answer.model_dump_json())
            except UnexpectedModelBehavior as error:
                answer = _layout_hold("review output validation failed", str(error))
            return section, answer

        # gather waits for siblings even on a failed request, preserving all finished receipts.
        tasks: list[Awaitable[Any]] = [one(s) for s in sections]
        if "cargo" in sections:
            tasks.append(self._ensure_cargo_map())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        reviews = {
            row[0]: row[1]
            for row in results
            if row is not None and not isinstance(row, BaseException)
        }
        if "cargo" in reviews:
            try:
                topology = await self._review_cargo_relations(target, audit=previous is not None)
                reviews["cargo"] = combine_reviews(reviews["cargo"], topology)
            except UnexpectedModelBehavior as error:
                reviews["cargo"] = combine_reviews(
                    reviews["cargo"],
                    _layout_hold("cargo comparison output validation failed", str(error)),
                )
            except Exception as error:
                results.append(error)
        for section, review in reviews.items():
            findings = source_fidelity_findings(draft_value(target), self.ocr, section)
            try:
                SECTION_MODELS[section].model_validate_json(
                    encoded(
                        normalize_optional_objects(
                            SECTION_MODELS[section], section_values(target, section)
                        )
                    )
                )
            except ValidationError as error:
                # A semantic pass cannot override a known application constraint.
                # Route the diagnostic to correction instead of discovering it only at commit.
                findings.append(
                    ReviewFinding(
                        field="section",
                        issue="normalization",
                        explanation=encoded(
                            error.errors(
                                include_url=False, include_context=False, include_input=False
                            )
                        ),
                        suggestedCorrection=(
                            "Resolve these section-schema constraints using the OCR and field "
                            "definitions; preserve supported facts. Report ambiguity if the "
                            "required correction cannot be established."
                        ),
                    )
                )
            if section == "cargo":
                findings += cargo_numeric_findings(
                    draft_value(target), self.ocr, self._cargo_source
                )
            if findings:
                reviews[section] = combine_reviews(
                    review,
                    SectionReview(
                        status="corrections_needed",
                        findings=findings,
                        explanation="Deterministic schema/fidelity failures require adjudication.",
                    ),
                )
        for section, review in reviews.items():
            atomic_publish_json(
                self.output_dir / "reviews" / f"{self._sequence:03d}-{section}.json",
                review.model_dump(mode="json"),
            )
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            raise ExceptionGroup("section review failed; completed siblings preserved", failures)
        if any(isinstance(r, BaseException) for r in results):
            raise asyncio.CancelledError
        return reviews

    async def _correction_wave(
        self,
        candidate: dict[str, Any],
        reviews: dict[Section, SectionReview],
    ) -> tuple[dict[str, Any], dict[Section, SectionReview], list[dict[str, Any]]]:
        """Commit independent supported corrections and recheck their dependency scopes."""
        wave = 1
        previous = deepcopy(candidate)
        replacements: dict[Section, BaseModel] = {}
        replacement_cargo_map: CargoSourceMap | None = None
        correction_holds: dict[Section, SectionReview] = {}
        correction_records: list[dict[str, Any]] = []
        edges: list[tuple[Section, ...]] = []
        for section, review in reviews.items():
            if not any(f.suggestedCorrection for f in review.findings):
                continue
            edges.append((section,))
            edges.extend(
                (section, finding.reassignTo)
                for finding in review.findings
                if finding.reassignTo is not None and finding.reassignTo != section
            )
        for scopes in connected_sections(edges):
            findings = [
                {"id": f"{s}:{i}", "section": s, **f.model_dump(mode="json")}
                for s in scopes
                for i, f in enumerate(reviews[s].findings)
            ]
            context = (
                "\n\n".join(
                    self._section_context(candidate, s, correction_scope=scopes) for s in scopes
                )
                + "\nFindings:\n"
                + encoded(findings)
            )
            try:
                answer = await self._with_layout(
                    "corrector", correction_model(scopes), context, scopes, audit=True
                )
            except UnexpectedModelBehavior as error:
                for section in scopes:
                    correction_holds[section] = _layout_hold(
                        "correction output validation failed", str(error)
                    )
                continue
            if isinstance(answer, CorrectionHold):
                for section in scopes:
                    correction_holds[section] = _layout_hold("correction conflict", answer.reason)
            elif isinstance(answer, SectionReview):
                for section in scopes:
                    correction_holds[section] = answer
            else:
                decisions = [d.model_dump(mode="json") for d in answer.decisions]
                actual_ids = [d["findingId"] for d in decisions]
                expected_ids = {f["id"] for f in findings}
                if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
                    for section in scopes:
                        correction_holds[section] = _layout_hold(
                            "incomplete adjudication", "decision IDs must cover every finding once"
                        )
                    continue
                values = answer.values.model_dump(mode="json", exclude_none=True)
                # Instruction decisions retain null; unrelated absent facts remain sparse.
                if "metadata_freight" in scopes:
                    values["negotiability"] = answer.values.negotiability
                for party in values.get("parties", {}).get("notifyParties", []):
                    party.setdefault("sameAs", None)
                before = {k: v for s in scopes for k, v in section_values(candidate, s).items()}
                change_error = correction_change_error(before, values, decisions)
                if all(f["field"] == "cargoSourceMap" for f in findings) and _review_changes(
                    before, values
                ):
                    change_error = "map-only findings do not authorize target changes"
                if change_error:
                    for section in scopes:
                        correction_holds[section] = _layout_hold(
                            "correction changed-field mismatch", change_error
                        )
                    continue
                correction_records.append(
                    {
                        "wave": wave,
                        "sections": list(scopes),
                        "findings": findings,
                        "decisions": decisions,
                    }
                )
                undecided = {
                    d["findingId"]: d["explanation"]
                    for d in decisions
                    if d["disposition"] == "unresolved"
                }
                for section in scopes:
                    pending = [
                        ReviewFinding(
                            field=f["field"],
                            issue="ambiguous",
                            explanation=undecided[f["id"]],
                            ocrExcerpt=f["ocrExcerpt"],
                        )
                        for f in findings
                        if f["id"] in undecided and f["section"] == section
                    ]
                    if pending:
                        correction_holds[section] = SectionReview(
                            status="unresolved",
                            explanation="Corrections retained; these facts remain undecidable.",
                            findings=pending,
                        )
                if "cargo" in scopes and answer.mapCorrection is not None:
                    errors = map_grounding_errors(answer.mapCorrection, self.ocr)
                    if errors:
                        for section in scopes:
                            correction_holds[section] = _layout_hold(
                                "cargo map correction invalid", "; ".join(errors)
                            )
                        continue
                    replacement_cargo_map = answer.mapCorrection
                for section in scopes:
                    replacements[section] = SECTION_MODELS[section].model_validate_json(
                        encoded({k: v for k, v in values.items() if k in SECTION_FIELDS[section]})
                    )

        atomic_publish_json(self.output_dir / f"wave-{wave}-decisions.json", correction_records)

        # Equipment/cargo and parties/metadata are coupled; commit each dependent group together.
        # A failed group does not discard independent, valid corrections in other groups.
        dependency_groups = connected_sections(
            [
                ("equipment", "cargo"),
                ("parties", "metadata_freight"),
                ("route_transport",),
                *[scopes for scopes in connected_sections(edges) if len(scopes) > 1],
            ]
        )
        recheck: set[Section] = set()
        for group in dependency_groups:
            changes = {s: replacements[s] for s in group if s in replacements}
            if not changes:
                continue
            try:
                # Validate the affected dependency group; an invalid untouched group
                # must not prevent independent corrections from being preserved.
                proposed = merge_sections(candidate, changes, validation_scope=group)
                for section in group:
                    self._check_package_tokens(section_values(proposed, section))
                    prior_findings = source_fidelity_findings(candidate, self.ocr, section)
                    next_findings = source_fidelity_findings(proposed, self.ocr, section)
                    if section == "cargo":
                        prior_findings += cargo_numeric_findings(
                            candidate, self.ocr, self._cargo_source
                        )
                        next_findings += cargo_numeric_findings(
                            proposed, self.ocr, replacement_cargo_map or self._cargo_source
                        )
                    known = {(f.field, f.explanation) for f in prior_findings}
                    introduced = [
                        f.explanation
                        for f in next_findings
                        if (f.field, f.explanation) not in known
                    ]
                    if introduced:
                        raise ValueError(
                            "Rejected newly unsupported correction: " + "; ".join(introduced)
                        )
            except ValueError as error:
                for section in changes:
                    correction_holds[section] = _layout_hold("cross-section validation", str(error))
            else:
                candidate = proposed
                if "cargo" in changes and replacement_cargo_map is not None:
                    self._cargo_source = self._cargo_draft = replacement_cargo_map
                    self._cargo_map_errors = []
                    atomic_publish_json(
                        self.output_dir / f"wave-{wave}-cargo-source-map.json",
                        replacement_cargo_map.model_dump(mode="json"),
                    )
                recheck.update(group)
        if recheck:
            reviews.update(
                await self.review(
                    candidate,
                    tuple(s for s in SECTION_FIELDS if s in recheck),
                    previous=previous,
                )
            )
        for section, hold in correction_holds.items():
            # A hold cannot hide defects detected in an independently corrected group.
            reviews[section] = (
                combine_reviews(reviews[section], hold) if section in recheck else hold
            )
        atomic_publish_json(
            self.output_dir / f"wave-{wave}-reviews.json",
            {s: r.model_dump(mode="json") for s, r in reviews.items()},
        )
        atomic_publish_json(self.output_dir / f"wave-{wave}-target.json", candidate)
        return candidate, reviews, correction_records

    async def refine(self, target: Draft) -> dict[str, Any]:
        """Review, correct once and leave final findings for manual adjudication."""
        candidate = draft_value(target)
        atomic_publish_bytes(
            self.output_dir / "initial-target.json", (encoded(candidate) + "\n").encode()
        )
        reviews = await self.review(candidate, tuple(SECTION_FIELDS))
        atomic_publish_json(
            self.output_dir / "initial-reviews.json",
            {s: r.model_dump(mode="json") for s, r in reviews.items()},
        )
        correction_records: list[dict[str, Any]] = []
        if any(f.suggestedCorrection for review in reviews.values() for f in review.findings):
            candidate, reviews, records = await self._correction_wave(candidate, reviews)
            correction_records.extend(records)
        atomic_publish_json(self.output_dir / "correction-decisions.json", correction_records)
        validation_error = None
        casing_changes: list[dict[str, Any]] = []
        equipment_decisions: tuple[dict[str, Any], ...] = ()
        try:
            candidate, equipment_decisions = reconcile_equipment_categories(
                candidate, source_text=self.ocr
            )
            candidate, casing_changes = normalize_target_casing(candidate)
            validated = BillOfLadingExtractionV7Label.model_validate_json(encoded(candidate))
            self._check_package_tokens(validated.canonical_target())
            candidate = validated.canonical_target()
        except ValueError as error:
            validation_error = str(error)
        result = {
            "status": "reviewed_candidate"
            if validation_error is None and all(r.status == "pass" for r in reviews.values())
            else "needs_adjudication",
            "goldApproved": False,
            "target": candidate if validation_error is None else None,
            "validationError": validation_error,
            "reviews": {s: r.model_dump(mode="json") for s, r in reviews.items()},
            "correctionDecisions": correction_records,
            "casingNormalizations": casing_changes,
            "equipmentNormalizations": equipment_decisions,
            "layoutAssistedSections": sorted(self.layout_sections),
            "calls": len(self.receipts),
        }
        atomic_publish_bytes(
            self.output_dir
            / ("target.json" if validation_error is None else "reviewed-draft.json"),
            (encoded(candidate) + "\n").encode(),
        )
        atomic_publish_json(self.output_dir / "status.json", result)
        return result
