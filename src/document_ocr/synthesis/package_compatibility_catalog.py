"""Compile reusable constrained decisions for unsupported DG/package contexts.

Most package decisions are resolved without an LLM from fit-only joint support.
This stage calls a provider only for dangerous-goods contexts for which that
support is absent.  The provider sees the pinned task vocabulary and returns
one or more physically plausible complete signatures under provider-native
strict JSON Schema.  The resulting catalog is immutable and reusable across
synthetic descendants; no per-document linguistic call is needed later.
"""

from __future__ import annotations

import asyncio
import json
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    create_model,
)
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import SynthesisPackageCompatibilityCatalogConfig
from document_ocr.synthesis.dangerous_goods_plan_pipeline import DangerousGoodsPlanRow
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_openai_key,
    model_messages,
    openai_responses_settings,
    usage_receipt,
)
from document_ocr.synthesis.package_goods_compatibility import (
    PackageCompatibilityCandidate,
    PackageCompatibilityContext,
    PackageCompatibilityResolution,
    PackageDangerousGoodsFact,
    build_package_goods_fit_support,
    load_fit_partition_document_ids,
    sample_dangerous_goods_package,
)
from document_ocr.synthesis.package_registry import load_package_registry
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.semantic_completion_pipeline import (
    _load_dg_plans,
    _load_source_targets,
    _resolve_file,
    _validate_committed_run,
)
from document_ocr.training.config import resolve_config_path
from document_ocr.training.tasks import RelationExplicitTaskConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)


class PackageCompatibilityCaseRecord(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    caseIndex: Annotated[int, Field(ge=0)]
    context: PackageCompatibilityContext
    outputSchemaSha256: Sha256
    providerNativeStrictJsonSchema: Literal[True]
    automaticRepairRequests: Literal[0]
    startedAt: datetime
    completedAt: datetime
    durationMs: Annotated[float, Field(ge=0)]
    status: Literal["success", "call_failed"]
    resolution: PackageCompatibilityResolution | None
    usage: LinguisticUsageReceipt
    errorType: str | None
    errorMessage: str | None


def _dynamic_output_model(
    *, allowed_tokens: tuple[str, ...], package_count: int
) -> type[BaseModel]:
    category_literal = Literal.__getitem__(allowed_tokens)
    categories_type = tuple[category_literal, ...]  # type: ignore[valid-type]
    candidate = create_model(
        f"PackageSignatureCandidate{package_count}",
        __config__=_STRICT,
        categories=(
            categories_type,
            Field(
                min_length=package_count,
                max_length=package_count,
                description=(
                    "One complete ordered package-category signature using only enum values."
                ),
            ),
        ),
        rationale=(
            Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)],
            Field(description="Brief physical-form and transport-compatibility justification."),
        ),
    )
    candidates_type = tuple[candidate, ...]  # type: ignore[valid-type]
    return create_model(
        f"PackageCompatibilityOutput{package_count}",
        __config__=_STRICT,
        candidates=(
            candidates_type,
            Field(
                min_length=1,
                max_length=8,
                description="Distinct physically plausible signatures, strongest first.",
            ),
        ),
    )


def _unsupported_contexts(
    *,
    plans: Sequence[DangerousGoodsPlanRow],
    support: Any,
    seed: int,
    namespace: str,
) -> tuple[PackageCompatibilityContext, ...]:
    contexts: dict[str, PackageCompatibilityContext] = {}
    for plan in plans:
        patch = cast(Mapping[str, Any], plan.target["documentPatch"])
        packages = cast(Sequence[Mapping[str, Any]], patch.get("cargoPackages") or ())
        realizations_by_group: dict[str, list[Any]] = {}
        for realization in plan.realizations:
            realizations_by_group.setdefault(realization.cargo_group_id, []).append(realization)
        for group_id, group_realizations in sorted(realizations_by_group.items()):
            package_count = sum(
                row.get("groupId") == group_id for row in packages
            )
            if package_count == 0:
                continue
            empirical = sample_dangerous_goods_package(
                support=support,
                hazard_categories=tuple(
                    value.semantic_hazard_category for value in group_realizations
                ),
                package_count=package_count,
                stream=DeterministicStream(
                    seed=seed,
                    namespace=namespace,
                    identity=f"{plan.base_document_id}:{group_id}",
                ),
            )
            if empirical is not None:
                continue
            if any(not value.generated_hs_codes for value in group_realizations):
                raise ValueError("DG realization lacks its generated HS identity")
            context = PackageCompatibilityContext(
                dangerousGoods=tuple(
                    PackageDangerousGoodsFact(
                        properShippingName=value.proper_shipping_name,
                        hs6=value.generated_hs_codes[0][:6],
                        hazardCategory=value.semantic_hazard_category,
                        exactHazardClass=value.exact_hazard_class,
                        subsidiaryHazardCategories=(
                            value.semantic_subsidiary_hazard_categories
                        ),
                        packingGroupCategory=value.packing_group_category,
                    )
                    for value in sorted(
                        group_realizations, key=lambda row: row.dangerous_goods_order
                    )
                ),
                packageCount=package_count,
            )
            contexts.setdefault(context.sha256, context)
    return tuple(contexts[key] for key in sorted(contexts))


async def _call_context(
    *,
    index: int,
    context: PackageCompatibilityContext,
    allowed_rows: Sequence[Mapping[str, str]],
    allowed_tokens: tuple[str, ...],
    allowed_tokens_sha256: str,
    system_prompt: str,
    model: OpenAIResponsesModel,
    settings: OpenAIResponsesModelSettings,
    config: SynthesisPackageCompatibilityCatalogConfig,
    semaphore: asyncio.Semaphore,
) -> tuple[PackageCompatibilityCaseRecord, JsonValue]:
    output_model = _dynamic_output_model(
        allowed_tokens=allowed_tokens, package_count=context.packageCount
    )
    output_schema_sha256 = sha256_bytes(
        canonical_json_bytes(output_model.model_json_schema(mode="validation"))
    )
    agent = Agent[object, BaseModel](
        model,
        output_type=NativeOutput(
            output_model,
            name=f"package_compatibility_{context.packageCount}",
            description=(
                "Return one or more physically plausible complete task-facing package signatures."
            ),
            strict=True,
        ),
        system_prompt=system_prompt,
        model_settings=settings,
        retries=config.workflow.structured_output_retries,
        name="bill-of-lading-package-compatibility",
    )
    request = {
        "context": context.model_dump(mode="json"),
        "allowedPackageCategories": list(allowed_rows),
    }
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    async with semaphore:
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                    usage_limits=UsageLimits(
                        request_limit=config.workflow.requests_per_context,
                        output_tokens_limit=config.provider.max_output_tokens,
                    ),
                )
                output_payload = result.output.model_dump(mode="python")
                raw_candidates = cast(Sequence[Mapping[str, Any]], output_payload["candidates"])
                candidates = tuple(
                    PackageCompatibilityCandidate(
                        categories=tuple(cast(Sequence[str], row["categories"])),
                        rationale=cast(str, row["rationale"]),
                    )
                    for row in raw_candidates
                )
                resolution = PackageCompatibilityResolution(
                    schemaVersion=1,
                    context=context,
                    contextSha256=context.sha256,
                    allowedCategoryTokensSha256=allowed_tokens_sha256,
                    candidates=candidates,
                    method="provider_native_constrained_package_compatibility_v1",
                )
                responses = tuple(
                    message
                    for message in result.new_messages()
                    if isinstance(message, ModelResponse)
                )
                if len(responses) != 1 or result.usage.requests != 1:
                    raise RuntimeError(
                        "package compatibility context produced other than one response"
                    )
                return (
                    PackageCompatibilityCaseRecord(
                        schemaVersion=1,
                        caseIndex=index,
                        context=context,
                        outputSchemaSha256=output_schema_sha256,
                        providerNativeStrictJsonSchema=True,
                        automaticRepairRequests=0,
                        startedAt=started_at,
                        completedAt=datetime.now(UTC),
                        durationMs=(time.perf_counter() - started) * 1000,
                        status="success",
                        resolution=resolution,
                        usage=usage_receipt(responses, config.provider.pricing),
                        errorType=None,
                        errorMessage=None,
                    ),
                    model_messages(captured),
                )
            except Exception as error:
                responses = tuple(
                    message for message in captured if isinstance(message, ModelResponse)
                )
                return (
                    PackageCompatibilityCaseRecord(
                        schemaVersion=1,
                        caseIndex=index,
                        context=context,
                        outputSchemaSha256=output_schema_sha256,
                        providerNativeStrictJsonSchema=True,
                        automaticRepairRequests=0,
                        startedAt=started_at,
                        completedAt=datetime.now(UTC),
                        durationMs=(time.perf_counter() - started) * 1000,
                        status="call_failed",
                        resolution=None,
                        usage=usage_receipt(responses, config.provider.pricing),
                        errorType=type(error).__name__,
                        errorMessage=str(error),
                    ),
                    model_messages(captured),
                )


async def _run_async(
    *, project_root: Path, config_path: Path, config: SynthesisPackageCompatibilityCatalogConfig
) -> dict[str, JsonValue]:
    _validate_committed_run(project_root, config.inputs.dangerous_goods_run)
    plans_path = _resolve_file(
        project_root,
        config.inputs.dangerous_goods_plans.path,
        config.inputs.dangerous_goods_plans.sha256,
        label="dangerous-goods plans",
    )
    plans = _load_dg_plans(plans_path, expected_records=config.inputs.dangerous_goods_plans.records)
    source_targets = _load_source_targets(project_root=project_root, config=config)
    fit_report = _resolve_file(
        project_root,
        config.inputs.fit_partition_report.path,
        config.inputs.fit_partition_report.sha256,
        label="fit partition report",
    )
    fit_ids = load_fit_partition_document_ids(fit_report)
    constraints_path = _resolve_file(
        project_root,
        config.inputs.source_task_constraints.path,
        config.inputs.source_task_constraints.sha256,
        label="source task constraints",
    )
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        read_regular_file_bytes(constraints_path), strict=True
    )
    package_path = _resolve_file(
        project_root,
        config.inputs.package_registry.path,
        config.inputs.package_registry.sha256,
        label="package registry",
    )
    package_registry = load_package_registry(
        package_path,
        expected_sha256=config.inputs.package_registry.sha256,
        expected_entries=config.inputs.package_registry_entries,
    )
    allowed_tokens = tuple(constraints.packageCategoryTokens)
    if not set(allowed_tokens) <= package_registry.category_tokens:
        raise ValueError("task package vocabulary is absent from the pinned package registry")
    support = build_package_goods_fit_support(
        source_targets=source_targets,
        fit_document_ids=fit_ids,
        allowed_category_tokens=allowed_tokens,
        frozen_minimum_celsius=config.fit_temperature.frozen_minimum_celsius,
        frozen_maximum_celsius=config.fit_temperature.frozen_maximum_celsius,
        chilled_minimum_celsius=config.fit_temperature.chilled_minimum_celsius,
        chilled_maximum_celsius=config.fit_temperature.chilled_maximum_celsius,
    )
    contexts = _unsupported_contexts(
        plans=plans,
        support=support,
        seed=1,
        namespace=config.run.run_id,
    )
    prompt_path = _resolve_file(
        project_root, config.prompt.path, config.prompt.sha256, label="compatibility prompt"
    )
    prompt_bytes = read_regular_file_bytes(prompt_path)
    system_prompt = prompt_bytes.decode("utf-8")
    allowed_rows = tuple(
        {
            "categoryToken": token,
            "displayName": package_registry.entry(token).displayName,
        }
        for token in allowed_tokens
    )
    allowed_tokens_sha256 = sha256_bytes(canonical_json_bytes(allowed_tokens))
    transaction = {
        "schemaVersion": 1,
        "configSha256": sha256_file(config_path),
        "plansSha256": config.inputs.dangerous_goods_plans.sha256,
        "sourceSha256": config.source.file.sha256,
        "fitPartitionSha256": config.inputs.fit_partition_report.sha256,
        "packageRegistrySha256": config.inputs.package_registry.sha256,
        "promptSha256": config.prompt.sha256,
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "contexts": [value.model_dump(mode="json") for value in contexts],
        "runtime": {
            "model": config.provider.model,
            "reasoningEffort": config.provider.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
    }
    stage = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    expected = [
        "REPORT.md",
        "config.yaml",
        "generation/resolutions.jsonl",
        "generation/summary.json",
        "prompt.md",
    ]
    for index in range(len(contexts)):
        expected.extend(
            (
                f"generation/cases/{index + 1:02d}.json",
                f"generation/requests/{index + 1:02d}.json",
                f"generation/transcripts/{index + 1:02d}.json",
            )
        )
    expected.sort()
    if stage.completed:
        stage.commit(
            expected_artifacts=expected,
            metadata={"contexts": len(contexts), "schema_version": 1},
        )
        return cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(stage.final_root / "generation/summary.json")),
        )
    stage.recover_interrupted_temporary_files()
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    stage.publish_bytes("prompt.md", prompt_bytes)
    for index, context in enumerate(contexts):
        stage.publish_json(
            f"generation/requests/{index + 1:02d}.json",
            {
                "systemPrompt": system_prompt,
                "context": context.model_dump(mode="json"),
                "allowedPackageCategories": allowed_rows,
            },
        )

    client = AsyncOpenAI(
        api_key=load_openai_key(project_root, config.environment_file),
        max_retries=config.provider.transport_max_retries,
        timeout=config.provider.request_timeout_seconds,
    )
    model = OpenAIResponsesModel(
        config.provider.model, provider=OpenAIProvider(openai_client=client)
    )
    settings = openai_responses_settings(config.provider)
    semaphore = asyncio.Semaphore(config.workflow.max_concurrent_requests)
    results = await asyncio.gather(
        *(
            _call_context(
                index=index,
                context=context,
                allowed_rows=allowed_rows,
                allowed_tokens=allowed_tokens,
                allowed_tokens_sha256=allowed_tokens_sha256,
                system_prompt=system_prompt,
                model=model,
                settings=settings,
                config=config,
                semaphore=semaphore,
            )
            for index, context in enumerate(contexts)
        )
    )
    records = tuple(value[0] for value in results)
    for record, transcript in results:
        stage.publish_json(
            f"generation/cases/{record.caseIndex + 1:02d}.json",
            record.model_dump(mode="json"),
        )
        stage.publish_json(
            f"generation/transcripts/{record.caseIndex + 1:02d}.json", transcript
        )
    failures = tuple(value for value in records if value.status != "success")
    if failures:
        raise RuntimeError(
            f"{len(failures)} package compatibility contexts failed; "
            "use a fresh run ID after diagnosis"
        )
    resolutions = tuple(cast(PackageCompatibilityResolution, value.resolution) for value in records)
    resolution_payload = b"".join(
        canonical_json_bytes(value.model_dump(mode="json")) + b"\n" for value in resolutions
    )
    summary = {
        "schemaVersion": 1,
        "selectedDocuments": len(plans),
        "fitDocuments": len(fit_ids),
        "dangerousGoodsRealizations": sum(len(value.realizations) for value in plans),
        "empiricallyUnsupportedUniqueContexts": len(contexts),
        "providerCalls": sum(value.usage.requests for value in records),
        "inputTokens": sum(value.usage.inputTokens for value in records),
        "outputTokens": sum(value.usage.outputTokens for value in records),
        "reasoningTokens": sum(value.usage.reasoningTokens for value in records),
        "estimatedCostUsd": str(
            sum((value.usage.estimatedCostUsd for value in records), start=0)
        ),
        "resolutionRecords": len(resolutions),
        "fitSupportAudit": asdict(support.audit),
    }
    report = (
        "# Package/goods compatibility catalog\n\n"
        f"- Selected DG plans: **{len(plans):,}**\n"
        f"- Fit-only documents: **{len(fit_ids):,}**\n"
        f"- Unsupported unique contexts sent to the constrained provider: **{len(contexts):,}**\n"
        f"- Provider calls: **{summary['providerCalls']:,}**\n"
        f"- Estimated cost: **${summary['estimatedCostUsd']}**\n\n"
        "Observed hazard/package relationships are resolved empirically. Only absent hazard and "
        "package-cardinality contexts reach this catalog, and every returned category is "
        "constrained "
        "to the pinned MPCI task vocabulary.\n"
    )
    stage.publish_bytes("generation/resolutions.jsonl", resolution_payload)
    stage.publish_bytes("generation/summary.json", json_artifact_bytes(summary))
    stage.publish_bytes("REPORT.md", report.encode())
    committed = stage.commit(
        expected_artifacts=expected,
        metadata={"contexts": len(contexts), "schema_version": 1},
    )
    return cast(
        dict[str, JsonValue],
        {"outputDir": str(stage.final_root), "created": committed.created, **summary},
    )


def run_package_compatibility_catalog(
    *, project_root: Path, config_path: Path, config: SynthesisPackageCompatibilityCatalogConfig
) -> dict[str, JsonValue]:
    started = time.perf_counter()
    result = asyncio.run(
        _run_async(project_root=project_root, config_path=config_path, config=config)
    )
    return {
        **result,
        "runtimeSeconds": round(time.perf_counter() - started, 6),
        "peakRssMiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
    }
