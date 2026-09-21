"""Concurrent complete generation, validated rendering, and immutable publication.

Provider responses are content-addressed and revalidated on reuse. Incomplete
cohorts retain diagnostic work artifacts but cannot publish training records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import resource
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import atomic_publish_bytes, atomic_write_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.hs_registry import (
    compile_uk_global_tariff_registry,
    load_ukgt_source_pin,
)
from document_ocr.synthesis.linguistic_probe_runtime import model_messages, usage_receipt
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.vessel_name_registry import load_vessel_name_registry

from . import cargo_identifiers, geographic_context
from . import complete_targets as targets
from . import descendant as render
from . import numeric_auxiliary as numeric
from .descendant_models import (
    DescendantConfig,
    PinnedCommittedRun,
    PreparedTargetReceipt,
    ResidualStageReceipt,
)
from .models import NonEmptyText, PinnedFile, PinnedJsonl, ProviderConfig
from .pipeline import project_root_from_config, resolve_input
from .request_batches import (
    BatchMemberError,
    RequestBatcher,
    run_template_waves,
    validate_batch_member,
)
from .spending_guard import SpendingGuard, SpendingLimitExceeded


class SpendingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    ledger_path: NonEmptyText
    maximum_estimated_cost_usd: Annotated[Decimal, Field(gt=0)]


class CompleteSynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1]
    task: Literal["bill_of_lading_complete_compiled_synthesis_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    response_cache_dir: NonEmptyText
    environment_file: NonEmptyText
    template_run: PinnedCommittedRun
    sample_plan_run: PinnedCommittedRun
    sample_plan: PinnedJsonl
    iso3166_snapshot: PinnedFile
    vessel_registry: PinnedJsonl
    hs_manifest: PinnedFile
    hs_metadata: PinnedFile
    hs_report: PinnedFile
    generation_prompt: PinnedFile
    numeric_prompt: PinnedFile
    numeric_contract_run: PinnedCommittedRun
    residual_prompt: PinnedFile
    provider: ProviderConfig
    seed: int
    documents: Annotated[int, Field(gt=0)]
    max_concurrent_requests: Annotated[int, Field(gt=0, le=16)]
    generation_attempts: Annotated[int, Field(gt=0, le=4)]
    provider_launch_authorized: bool
    spending: SpendingConfig
    request_batch_size: Annotated[int, Field(ge=2, le=16)]


def load_config(path: Path) -> CompleteSynthesisConfig:
    return CompleteSynthesisConfig.model_validate_json(
        json.dumps(yaml.safe_load(read_regular_file_bytes(path)), default=str)
    )


def _generation_schema(
    fields: Sequence[Mapping[str, Any]],
    requirements: list[dict[str, Any]],
) -> type[BaseModel]:
    """Put proved repair grammars in native output validation, not only prose."""
    patterns = {}
    for row in requirements:
        for obligation in row["obligations"]:
            if obligation.get("partitionPattern"):
                patterns[row["key"]] = obligation["partitionPattern"]
            if obligation.get("opaqueShapes"):
                source = obligation["opaqueShapes"][0]["source"]
                patterns[row["key"]] = (
                    "^"
                    + "".join(
                        "[0-9]"
                        if c.isdigit()
                        else "[A-Z]"
                        if c.isupper()
                        else "[a-z]"
                        if c.islower()
                        else re.escape(c)
                        for c in source
                    )
                    + "$"
                )
    definitions: dict[str, Any] = {
        field["key"]: (Annotated[str, Field(min_length=1, pattern=patterns.get(field["key"]))], ...)
        for field in fields
    }
    for field in fields:
        if frame := field.get("requiredBoundaryPunctuation"):
            definitions[field["key"]] = (
                Annotated[
                    str,
                    Field(
                        min_length=1,
                        pattern=(
                            "^"
                            + re.escape(frame["prefix"])
                            + r"[^\r\n]+"
                            + re.escape(frame["suffix"])
                            + "$"
                        ),
                    ),
                ],
                ...,
            )
    return create_model(
        "CompleteScenarioFields", __config__=ConfigDict(extra="forbid", strict=True), **definitions
    )


def _pinned(root: Path, pin: PinnedFile | PinnedJsonl) -> Path:
    path = resolve_input(root, pin.path)
    if sha256_file(path) != pin.sha256:
        raise ValueError(f"input hash mismatch: {pin.path}")
    return path


def _reusable_residual(
    checkpoint: Mapping[str, Any] | None,
    case: render.PreparedCase,
    plan: render.RenderPlan,
    prompt_sha256: str,
) -> tuple[dict[str, str], ResidualStageReceipt, dict[str, Any]] | None:
    """Reuse paid slots only for identical frozen facts, retaining the full receipt.

    Deterministic capability expansion can shrink the residual set. That does not
    invalidate the paid strings for remaining slots or authorize editing targets.
    The returned subset is revalidated by full materialization before acceptance.
    """
    if checkpoint is None or not checkpoint.get("residualResponse"):
        return None
    if (
        checkpoint["sampleId"] != case.document_id
        or checkpoint["sourceDocumentId"] != case.source_document_id
    ):
        raise ValueError("residual checkpoint identity differs")
    if checkpoint["targetSha256"] != sha256_bytes(canonical_json_bytes(checkpoint["target"])):
        raise ValueError("residual checkpoint target hash differs")
    if checkpoint["targetReceipt"] != case.target_receipt.model_dump(mode="json"):
        return None  # Different frozen facts require their own rendering request.
    if checkpoint["auxiliaryValues"] != case.auxiliary_values or checkpoint["numericAuxiliary"] != {
        k: v.model_dump(mode="json") for k, v in case.numeric_auxiliary.items()
    }:
        raise ValueError("residual checkpoint facts differ from their frozen receipt")
    stage = ResidualStageReceipt.model_validate_json(canonical_json_bytes(checkpoint["agentStage"]))
    if stage.document_id != case.document_id or stage.system_prompt_sha256 != prompt_sha256:
        raise ValueError("residual checkpoint stage identity or prompt differs")
    if stage.status != "success" or not isinstance(stage.output, dict):
        return None
    expected = {s.slot_id for b in plan.residual_bindings for s in b.occurrences}
    if not expected <= set(stage.output):
        return None
    receipt = checkpoint["residualResponse"]
    if receipt["output"] != stage.output or receipt["outputSha256"] != sha256_bytes(
        canonical_json_bytes(stage.output)
    ):
        raise ValueError("residual checkpoint provider receipt differs")
    return (
        {k: cast(str, stage.output[k]) for k in expected},
        stage,
        {**receipt, "cacheReused": True},
    )


def _prepared(
    source: targets.SourceTemplate,
    target: dict[str, Any],
    auxiliary: dict[str, str],
    *,
    sample_id: str,
    seed: int,
    numeric_values: dict[str, numeric.PreparedNumeric],
) -> render.PreparedCase:
    old, new = render._flatten_leaves(source.target), render._flatten_leaves(target)
    digest = sha256_bytes(canonical_json_bytes(target))
    receipt = PreparedTargetReceipt.model_validate(
        {
            "schema_version": 1,
            "document_id": sample_id,
            "source_document_id": source.document_id,
            "target_origin": "complete_synthetic_target",
            "source_schema_version": source.source_target["schemaVersion"],
            "target_schema_version": target["schemaVersion"],
            "fixed_carrier_name": source.template.carrier.canonical_name,
            "source_target_sha256": sha256_bytes(canonical_json_bytes(source.source_target)),
            "proposed_target_sha256": digest,
            "prepared_target_sha256": digest,
            "auxiliary_values_sha256": sha256_bytes(canonical_json_bytes(auxiliary)),
            "numeric_auxiliary_sha256": sha256_bytes(
                canonical_json_bytes(
                    {k: v.model_dump(mode="json") for k, v in numeric_values.items()}
                )
            ),
            "synthetic_document_id": "syn_tpl_"
            + sha256_bytes(
                canonical_json_bytes(
                    [sample_id, source.document_id, source.template.source_sha256, digest, seed]
                )
            )[:40],
            "topology_mismatch_count": 0,
            "target_leaf_count": len(new),
            "changed_target_leaf_count": sum(old[p] != value for p, value in new.items()),
            "carrier_leaf_count": sum(p.startswith("documentPatch.parties.carrier") for p in old),
            "carrier_changed_leaf_count": 0,
            "compatibility_adaptations": (),
            "training_eligible": False,
        }
    )
    return render.PreparedCase(
        sample_id,
        source.document_id,
        source.source,
        source.source_target,
        source.target,
        target,
        source.template,
        receipt,
        auxiliary,
        numeric_values,
    )


async def _cached_fields(
    *,
    cache: Path,
    model: Any,
    provider: ProviderConfig,
    system_prompt: str,
    payload: Mapping[str, Any],
    output_type: type[BaseModel],
    limiter: asyncio.Semaphore,
    authorized: bool,
    spending: SpendingGuard | None = None,
    batcher: RequestBatcher | None = None,
    request_group: str | None = None,
) -> dict[str, Any]:
    prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    contract = {
        "system": system_prompt,
        "prompt": prompt,
        "schema": output_type.model_json_schema(),
        "provider": provider.model_dump(mode="json"),
    }
    if batcher is not None:
        contract["batchMember"] = batcher.member_contract(payload, output_type)
    digest = sha256_bytes(canonical_json_bytes(contract))
    path = cache / digest[:2] / f"{digest}.json"
    if path.exists():
        receipt = await asyncio.to_thread(_read_receipt, path)
        if receipt["requestSha256"] != digest or receipt["outputSha256"] != sha256_bytes(
            canonical_json_bytes(receipt["output"])
        ):
            raise ValueError("cached provider response hash differs")
        output_type.model_validate_json(canonical_json_bytes(receipt["output"]), strict=True)
        if batcher is not None:
            validate_batch_member(
                provenance=receipt["batchProvenance"],
                output=receipt["output"],
                messages=receipt["messages"],
                usage=receipt["usage"],
            )
        return {**receipt, "cacheReused": True}
    if not authorized:
        raise ValueError("complete generation needs a provider response; launch is disabled")
    if batcher is not None:
        if request_group is None:
            raise ValueError("batched requests require an explicit semantic group")
        try:
            receipt = await batcher.request(
                group=request_group,
                payload=payload,
                output_type=output_type,
                system_prompt=system_prompt,
            )
        except BatchMemberError as error:
            raise ProviderCallError(error.receipt) from error
        receipt = {**receipt, "requestSha256": digest}
        await asyncio.to_thread(_publish_receipt, path, receipt)
        return receipt
    if spending is None:
        raise ValueError("paid synthesis requires a durable spending reservation")
    if provider.transport_max_retries:
        raise ValueError("budgeted synthesis forbids unreceipted transport retries")
    # UTF-8 bytes upper-bound text tokens. Reserving twice the complete serialized
    # request (including schema) conservatively covers message/schema framing.
    # Full configured output capacity is reserved, not a hoped-for average.
    input_bytes = len(canonical_json_bytes(contract))
    input_price = max(
        provider.pricing.input_usd_per_million,
        provider.pricing.cached_input_usd_per_million,
        provider.pricing.input_usd_per_million * provider.pricing.cache_write_multiplier,
    )
    maximum_cost = (
        Decimal(2 * input_bytes) * input_price
        + Decimal(provider.max_output_tokens) * provider.pricing.output_usd_per_million
    ) / 1_000_000
    agent = Agent(
        model,
        output_type=NativeOutput(output_type, strict=True),
        system_prompt=system_prompt,
        model_settings=render._provider_settings(provider),
        retries=0,
    )
    started = time.perf_counter()
    failure = None
    async with limiter:
        reservation = await spending.acquire(digest, maximum_cost)
        with capture_run_messages() as messages:
            try:
                result = await agent.run(
                    prompt,
                    usage_limits=UsageLimits(
                        request_limit=1, output_tokens_limit=provider.max_output_tokens
                    ),
                )
            except Exception as error:
                failure = error
        captured = list(messages)
    output = result.output.model_dump(mode="json") if failure is None else None
    responses = tuple(row for row in captured if isinstance(row, ModelResponse))
    receipt = {
        "requestSha256": digest,
        "outputSha256": sha256_bytes(canonical_json_bytes(output)),
        "output": output,
        "usage": usage_receipt(
            responses,
            cast(Any, provider.pricing),
            require_provider_cost=provider.kind == "openrouter",
        ).model_dump(mode="json"),
        "messages": model_messages(captured),
        "wallSeconds": time.perf_counter() - started,
        "completedAt": datetime.now(UTC).isoformat(),
    }
    if failure is not None:
        receipt.update(errorType=type(failure).__name__, error=str(failure), cacheReused=False)
        causes = []
        cause = failure.__cause__
        while cause is not None:
            causes.append(
                {"type": type(cause).__name__, "message": str(cause), "representation": repr(cause)}
            )
            cause = cause.__cause__ or cause.__context__
        receipt["errorCauses"] = causes
        failure_id = sha256_bytes(canonical_json_bytes(receipt))
        await asyncio.to_thread(
            _publish_receipt, cache / "failures" / f"{failure_id}.json", receipt
        )
        spending.settle(
            reservation, Decimal(receipt["usage"]["estimatedCostUsd"]) if responses else None
        )
        raise ProviderCallError(receipt) from failure
    await asyncio.to_thread(_publish_receipt, path, receipt)
    spending.settle(reservation, Decimal(receipt["usage"]["estimatedCostUsd"]))
    return {**receipt, "cacheReused": False}


class ProviderCallError(RuntimeError):
    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        super().__init__(receipt["error"])


def _read_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(read_regular_file_bytes(path))
    if not isinstance(receipt, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    return receipt


def _publish_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    atomic_publish_bytes(path, canonical_json_bytes(receipt))


def _provider_unavailable(error: Exception) -> bool:
    return isinstance(error, ProviderCallError) and any(
        code in str(error).lower()
        for code in (
            "connection error",
            "connection timed out",
            "insufficient_quota",
            "insufficient credits",
            "credit balance",
            "account_deactivated",
            "invalid_api_key",
        )
    )


async def _numeric_contracts(
    *,
    sources: Mapping[str, targets.SourceTemplate],
    cache: Path,
    work: Path,
    model: Any,
    config: CompleteSynthesisConfig,
    prompt: str,
    limiter: asyncio.Semaphore,
    existing_contracts: Mapping[str, Mapping[str, numeric.NumericContract]] | None = None,
) -> tuple[
    dict[str, dict[str, numeric.NumericContract]], list[dict[str, Any]], list[dict[str, str]]
]:
    contracts: dict[str, dict[str, numeric.NumericContract]] = {}
    receipts = []
    failures = []
    account_stopped = asyncio.Event()

    async def compile_one(source: targets.SourceTemplate) -> None:
        if account_stopped.is_set():
            failures.append(
                {
                    "sourceDocumentId": source.document_id,
                    "error": "provider account failure; source remains unprocessed",
                }
            )
            return
        bindings = numeric.numeric_bindings(source.template)
        existing = dict((existing_contracts or {}).get(source.document_id, {}))
        if set(existing) - {binding.logical_key for binding in bindings}:
            raise ValueError("existing numeric contracts contain unknown bindings")
        requested = tuple(b for b in bindings if b.logical_key not in existing)
        if not bindings:
            contracts[source.document_id] = {}
            return
        if not requested:
            numeric.prepare(
                bindings,
                existing,
                source_target=source.target,
                target=source.target,
                scale=Decimal(1),
                source_template=source.template,
            )
            contracts[source.document_id] = existing
            return
        paths = tuple(numeric.numeric_target_leaves(source.target))
        path_type: Any = Literal[paths] if paths else str
        contract_type = create_model(
            "SourceNumericContract",
            __base__=numeric.NumericContract,
            target_paths=(list[path_type], Field(..., max_length=None if paths else 0)),
        )
        fields: dict[str, Any] = {
            f"numeric_{i:04d}": (contract_type, ...) for i, _b in enumerate(requested)
        }
        schema = create_model(
            "NumericDependencyContracts",
            __config__=ConfigDict(extra="forbid", strict=True),
            **fields,
        )
        payload = numeric.numeric_payload(
            source=source.source, target=source.target, bindings=requested
        )
        if existing:
            payload["existingVerifiedContracts"] = {
                key: value.model_dump(mode="json") for key, value in existing.items()
            }
        attempts = []
        error_message = ""
        for _attempt in range(config.generation_attempts):
            try:
                response = await _cached_fields(
                    cache=cache,
                    model=model,
                    provider=config.provider,
                    system_prompt=prompt,
                    payload=payload,
                    output_type=schema,
                    limiter=limiter,
                    authorized=config.provider_launch_authorized,
                )
            except ProviderCallError as error:
                receipts.append(error.receipt)
                attempts.append(error.receipt)
                error_message = str(error)
                if _provider_unavailable(error):
                    account_stopped.set()
                break
            receipts.append(response)
            attempts.append(response)
            try:
                parsed = {
                    **existing,
                    **{
                        b.logical_key: numeric.NumericContract.model_validate_json(
                            canonical_json_bytes(response["output"][f"numeric_{i:04d}"]),
                            strict=True,
                        )
                        for i, b in enumerate(requested)
                    },
                }
                for binding in bindings:
                    numeric.validate_contract(
                        binding,
                        parsed[binding.logical_key],
                        source.target,
                        source_template=source.template,
                    )
                numeric.prepare(
                    bindings,
                    parsed,
                    source_target=source.target,
                    target=source.target,
                    scale=Decimal(1),
                    source_template=source.template,
                )
            except (ValueError, KeyError, TypeError) as error:
                error_message = str(error)
                payload = {
                    **payload,
                    "priorOutput": response["output"],
                    "validationError": error_message,
                }
                continue
            contracts[source.document_id] = parsed
            break
        if source.document_id not in contracts:
            failures.append({"sourceDocumentId": source.document_id, "error": error_message})
        await asyncio.to_thread(
            atomic_write_json,
            work / "numeric-contracts" / f"{source.document_id}.json",
            {
                "sourceDocumentId": source.document_id,
                "attempts": attempts,
                "contracts": {
                    k: v.model_dump(mode="json")
                    for k, v in contracts.get(source.document_id, {}).items()
                },
                "error": error_message if source.document_id not in contracts else None,
            },
        )

    pending: asyncio.Queue[targets.SourceTemplate] = asyncio.Queue()
    for source in sources.values():
        pending.put_nowait(source)

    async def worker() -> None:
        while not pending.empty():
            source = pending.get_nowait()
            await compile_one(source)
            pending.task_done()
            print(
                json.dumps(
                    {
                        "phase": "numeric_dependencies",
                        "completed": len(contracts) + len(failures),
                        "total": len(sources),
                        "reviewOrFailed": len(failures),
                    }
                ),
                flush=True,
            )

    await asyncio.gather(
        *(worker() for _ in range(min(config.max_concurrent_requests, len(sources))))
    )
    return contracts, receipts, failures


def _load_numeric_contracts(
    root: Path, pin: PinnedCommittedRun, sources: Mapping[str, targets.SourceTemplate]
) -> dict[str, dict[str, numeric.NumericContract]]:
    catalog = render._validate_committed_run(root, pin)
    result: dict[str, dict[str, numeric.NumericContract]] = {}
    seen = set()
    for line in read_regular_file_bytes(catalog / "contracts.jsonl").splitlines():
        row = json.loads(line)
        source_id = row["sourceDocumentId"]
        if source_id in seen:
            raise ValueError("duplicate source in numeric dependency catalog")
        seen.add(source_id)
        if source_id not in sources:
            continue
        source = sources[source_id]
        if (
            row["sourceSha256"] != sha256_bytes(source.source)
            or row["sourceTargetSha256"] != sha256_bytes(canonical_json_bytes(source.target))
            or row["effectiveTemplateSha256"]
            != sha256_bytes(canonical_json_bytes(source.template.model_dump(mode="json")))
        ):
            raise ValueError("numeric dependency catalog differs from pinned source/template")
        contracts = {
            key: numeric.NumericContract.model_validate_json(
                canonical_json_bytes(value), strict=True
            )
            for key, value in row["contracts"].items()
        }
        numeric.prepare(
            numeric.numeric_bindings(source.template),
            contracts,
            source_target=source.target,
            target=source.target,
            scale=Decimal(1),
            source_template=source.template,
        )
        result[source_id] = contracts
    if set(result) != set(sources):
        raise ValueError("numeric dependency catalog does not cover selected sources")
    return result


async def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = project_root_from_config(config_path)
    implementation = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted((root / "src/document_ocr").rglob("*.py"))
    }
    config = load_config(config_path)
    budget_path = root / config.spending.ledger_path
    if root not in budget_path.resolve().parents:
        raise ValueError("spending ledger escapes project")
    spending = SpendingGuard(
        budget_path,
        limit_usd=config.spending.maximum_estimated_cost_usd,
        pricing_sha256=sha256_bytes(
            canonical_json_bytes(config.provider.pricing.model_dump(mode="json"))
        ),
    )
    template_root = render._validate_committed_run(root, config.template_run)
    plan_root = render._validate_committed_run(root, config.sample_plan_run)
    plan_path = _pinned(root, config.sample_plan)
    if plan_root not in plan_path.parents:
        raise ValueError("sample plan is outside committed run")
    all_samples = render._read_jsonl(plan_path, records=config.sample_plan.records)
    if config.documents > len(all_samples):
        raise ValueError("requested documents exceed sample plan")
    if any(row["targetGeneration"] != "complete_latest_schema_targets_v1" for row in all_samples):
        raise ValueError("sample plan uses retired source-copy generation")
    if len({row["sampleId"] for row in all_samples}) != len(all_samples):
        raise ValueError("duplicate sample identities")
    # Probes interleave all capability cohorts. Production consumes the full pinned plan.
    if config.documents == len(all_samples):
        samples = list(all_samples)
    else:
        pools = [
            [row for row in all_samples if row["capabilityCohort"] == cohort]
            for cohort in ("standard", "dangerous_goods", "temperature_controlled")
        ]
        samples = []
        for index in range(max(map(len, pools))):
            for pool in pools:
                if index < len(pool) and len(samples) < config.documents:
                    samples.append(pool[index])
            if len(samples) == config.documents:
                break
    countries = render._country_code_map(_pinned(root, config.iso3166_snapshot))
    vessels = load_vessel_name_registry(
        _pinned(root, config.vessel_registry),
        expected_sha256=config.vessel_registry.sha256,
        expected_records=config.vessel_registry.records,
    )
    hs_registry = compile_uk_global_tariff_registry(
        source=load_ukgt_source_pin(_pinned(root, config.hs_manifest)),
        metadata_path=_pinned(root, config.hs_metadata),
        report_path=_pinned(root, config.hs_report),
    )
    generation_prompt = _pinned(root, config.generation_prompt).read_text()
    residual_prompt = _pinned(root, config.residual_prompt).read_text()
    _pinned(root, config.numeric_prompt)
    sources = {
        sid: targets.load_source(template_root, sid)
        for sid in sorted({row["sourceDocumentId"] for row in samples})
    }
    for source in sources.values():
        if conflicts := geographic_context.source_entity_conflicts(
            source.template, countries
        ):
            raise ValueError(
                "source auxiliary geography requires review before generation: "
                f"{source.document_id}: {conflicts}"
            )
    identifiers = targets.reserve_identifiers(samples, sources, seed=config.seed)
    requests = {sid: targets.lexical_contract(source) for sid, source in sources.items()}
    work = root / config.output_dir / ".work" / config.run_name
    cache = root / config.response_cache_dir
    for path in (work, cache):
        if root not in path.resolve().parents:
            raise ValueError("output path escapes project")
    work.mkdir(parents=True, exist_ok=True)
    model = render._provider_model(
        project_root=root, environment_file=config.environment_file, provider=config.provider
    )
    limiter = asyncio.Semaphore(config.max_concurrent_requests)
    numeric_contracts = _load_numeric_contracts(root, config.numeric_contract_run, sources)
    numeric_receipts: list[dict[str, Any]] = []

    async def submit_batch(**kwargs: Any) -> dict[str, Any]:
        return await _cached_fields(
            cache=cache / "batches",
            model=model,
            provider=config.provider,
            limiter=limiter,
            authorized=config.provider_launch_authorized,
            spending=spending,
            **kwargs,
        )

    batcher = RequestBatcher(size=config.request_batch_size, submit=submit_batch)
    outcomes: list[dict[str, Any]] = []
    provider_stopped = asyncio.Event()
    call_traces: dict[str, list[dict[str, Any]]] = {}

    async def execute(row: Mapping[str, Any]) -> dict[str, Any]:
        sample_id, source_id = row["sampleId"], row["sourceDocumentId"]
        checkpoint_path = work / f"{sample_id}.json"
        checkpoint = (
            await asyncio.to_thread(_read_receipt, checkpoint_path)
            if checkpoint_path.exists()
            else None
        )
        call_traces[sample_id] = []
        source = sources[source_id]
        proposed = targets.structured_proposal(
            source,
            sample_id=sample_id,
            seed=config.seed,
            allocations=identifiers,
            vessel_names=tuple(v.name for v in vessels.records),
            minimum_quantities=numeric.minimum_quantities(numeric_contracts[source_id]),
            quantity_multiples=numeric.quantity_multiples(
                numeric_contracts[source_id], source.target, source.template
            ),
        )
        for path, value in numeric.generated_measurements(
            numeric.numeric_bindings(source.template), numeric_contracts[source_id], proposed
        ).items():
            targets._set(proposed, path, float(value))
        goods = targets.propose_goods(
            proposed, registry=hs_registry, sample_id=sample_id, seed=config.seed
        )
        render._validate_target_compatibility(
            source=source.source,
            source_target=source.source_target,
            target=proposed,
            template=source.template,
        )
        fields = cargo_identifiers.conditioned_requests(
            requests[source_id], sample_id=sample_id, seed=config.seed
        )
        numeric_context = numeric.prepare(
            numeric.numeric_bindings(source.template),
            numeric_contracts[source_id],
            source_target=source.target,
            target=proposed,
            scale=targets.scenario_scale(sample_id, config.seed),
            source_template=source.template,
        )
        payload = {
            "sampleId": sample_id,
            "structuredScenario": proposed,
            "goodsIdentities": goods,
            "requestedFields": fields,
            "numericAuxiliary": {
                k: v.model_dump(mode="json", exclude_defaults=True)
                for k, v in numeric_context.items()
            },
            "coherenceConstraints": [
                c.model_dump(mode="json") for c in source.template.coherence_constraints
            ],
        }
        attempts = []
        residual_attempts: list[dict[str, Any]] = []
        for _attempt in range(config.generation_attempts):
            output_type = _generation_schema(fields, payload.get("repairRequirements", []))
            # Resume the last accepted linguistic candidate, not an earlier
            # rejected candidate whose repair request may have since improved.
            accepted = (
                checkpoint["attempts"][-1]
                if _attempt == 0
                and checkpoint is not None
                and checkpoint.get("target")
                and checkpoint.get("attempts")
                and set(checkpoint["attempts"][-1].get("output") or {})
                == set(output_type.model_fields)
                else None
            )
            if accepted is not None:
                assert checkpoint is not None  # Accepted responses only originate in checkpoints.
                if accepted["outputSha256"] != sha256_bytes(
                    canonical_json_bytes(accepted["output"])
                ):
                    raise ValueError("accepted generation checkpoint output hash differs")
                output_type.model_validate_json(
                    canonical_json_bytes(accepted["output"]), strict=True
                )
                validate_batch_member(
                    provenance=accepted["batchProvenance"],
                    output=accepted["output"],
                    messages=accepted["messages"],
                    usage=accepted["usage"],
                )
                attempts = [{**a, "cacheReused": True} for a in checkpoint["attempts"][:-1]]
                response = {**accepted, "cacheReused": True}
            else:
                response = await _cached_fields(
                    cache=cache,
                    model=model,
                    provider=config.provider,
                    system_prompt=generation_prompt,
                    payload=payload,
                    output_type=output_type,
                    limiter=limiter,
                    authorized=config.provider_launch_authorized,
                    batcher=batcher,
                    request_group=source_id + ":generation",
                )
            attempts.append(response)
            call_traces[sample_id].append(response)
            await asyncio.to_thread(
                atomic_write_json,
                work / f"{sample_id}.json",
                {
                    "status": "generating",
                    "sampleId": sample_id,
                    "sourceDocumentId": source_id,
                    "attempts": attempts,
                    "residualAttempts": residual_attempts,
                },
            )
            try:
                target, auxiliary, ledger = await asyncio.to_thread(
                    targets.complete_proposal,
                    source,
                    proposed,
                    fields,
                    response["output"],
                    sample_id=sample_id,
                )
                if (
                    accepted is not None
                    and checkpoint is not None
                    and target != checkpoint["target"]
                ):
                    raise ValueError(
                        "accepted generation checkpoint no longer reconstructs its frozen target"
                    )
                numeric_values = numeric.prepare(
                    numeric.numeric_bindings(source.template),
                    numeric_contracts[source_id],
                    source_target=source.target,
                    target=target,
                    scale=targets.scenario_scale(sample_id, config.seed),
                    source_template=source.template,
                )
                case = _prepared(
                    source,
                    target,
                    auxiliary,
                    sample_id=sample_id,
                    seed=config.seed,
                    numeric_values=numeric_values,
                )
                plan = await asyncio.to_thread(
                    render._build_initial_plan, case, seed=config.seed, country_codes=countries
                )
            except (ValueError, KeyError, TypeError) as error:
                payload = {
                    **payload,
                    "priorOutput": response["output"],
                    "validationError": str(error),
                    "repairAttempt": _attempt + 1,
                    "repairRequirements": targets.lexical_repair_requirements(
                        source, proposed, fields
                    ),
                }
                await asyncio.to_thread(
                    atomic_write_json,
                    work / f"{sample_id}.json",
                    {
                        "status": "generation_rejected",
                        "sampleId": sample_id,
                        "sourceDocumentId": source_id,
                        "error": str(error),
                        "attempts": attempts,
                        "residualAttempts": residual_attempts,
                    },
                )
                continue
            residual_payload = render._residual_payload(case=case, plan=plan)
            reusable = _reusable_residual(checkpoint, case, plan, config.residual_prompt.sha256)
            for _residual_attempt in range(config.generation_attempts):
                if reusable is not None and _residual_attempt == 0:
                    raw, stage, residual = reusable
                    residual_attempts.append(residual)
                    call_traces[sample_id].append(residual)
                elif plan.residual_bindings:
                    schema = render._residual_output_type(plan.residual_bindings)
                    residual = await _cached_fields(
                        cache=cache,
                        model=model,
                        provider=config.provider,
                        system_prompt=residual_prompt,
                        payload=residual_payload,
                        output_type=schema,
                        limiter=limiter,
                        authorized=config.provider_launch_authorized,
                        batcher=batcher,
                        request_group=source_id + ":residual",
                    )
                    residual_attempts.append(residual)
                    call_traces[sample_id].append(residual)
                    raw = residual["output"]
                    stage = ResidualStageReceipt.model_validate(
                        {
                            "schema_version": 1,
                            "document_id": sample_id,
                            "status": "success",
                            "started_at": residual["completedAt"],
                            "completed_at": residual["completedAt"],
                            "duration_seconds": residual["wallSeconds"],
                            "system_prompt_sha256": config.residual_prompt.sha256,
                            "user_prompt_sha256": sha256_bytes(
                                json.dumps(
                                    residual_payload,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ).encode()
                            ),
                            "output_schema_sha256": sha256_bytes(
                                canonical_json_bytes(schema.model_json_schema())
                            ),
                            "output": raw,
                            "error_type": None,
                            "error_message": None,
                            "messages": residual["messages"],
                            "usage": residual["usage"],
                            "batch_provenance": residual.get("batchProvenance"),
                        }
                    )
                else:
                    raw = {}
                    stage = render._not_required_stage(
                        document_id=sample_id, system_prompt_sha256=config.residual_prompt.sha256
                    )
                    residual = None
                execution = await asyncio.to_thread(
                    render._materialize_case,
                    case=case,
                    plan=plan,
                    raw_output=raw,
                    stage=stage,
                    country_codes=countries,
                )
                if execution.result.status == "passed" or not plan.residual_bindings:
                    break
                residual_payload = {
                    **residual_payload,
                    "priorOutput": raw,
                    "validationError": execution.result.error_message,
                    "repairAttempt": _residual_attempt + 1,
                    "repairRequirements": [
                        {
                            "logicalKey": binding.logical_key,
                            "exactTargetValues": [
                                {"path": path, "value": render._binding_target_value(target, path)}
                                for path in binding.target_paths
                            ],
                            "instructions": (
                                "Preserve the complete target wording, including all model/part "
                                "identities. Do not abbreviate or paraphrase it. Put it into its "
                                "source-owned slots; omit no target tokens. Opaque identifiers "
                                "must contain exactly the slot's alphanumericCount characters."
                            ),
                        }
                        for binding in plan.residual_bindings
                    ],
                }
            result = {
                "status": execution.result.status,
                "sampleId": sample_id,
                "sourceDocumentId": source_id,
                "target": target,
                "targetSha256": sha256_bytes(canonical_json_bytes(target)),
                "auxiliaryValues": auxiliary,
                "numericAuxiliary": {
                    k: v.model_dump(mode="json") for k, v in numeric_values.items()
                },
                "ledger": ledger,
                "targetReceipt": case.target_receipt.model_dump(mode="json"),
                "attempts": attempts,
                "residualResponse": residual,
                "residualAttempts": residual_attempts,
                "goodsIdentities": goods,
                "agentStage": stage.model_dump(mode="json"),
                "renderResult": execution.result.model_dump(mode="json"),
                "rendered": execution.rendered.decode() if execution.rendered else None,
                "routes": [route.model_dump(mode="json") for route in plan.routes],
            }
            await asyncio.to_thread(atomic_write_json, work / f"{sample_id}.json", result)
            # Residual exhaustion is not permission to regenerate already accepted
            # facts. Preserve the frozen target and report the exact failure.
            return result
        return {
            "status": "generation_rejected",
            "sampleId": sample_id,
            "sourceDocumentId": source_id,
            "error": payload["validationError"],
            "attempts": attempts,
            "residualAttempts": residual_attempts,
        }

    async def execute_one(row: dict[str, Any]) -> None:
        try:
            if provider_stopped.is_set():
                raise RuntimeError(
                    "provider account failure stopped new requests; sample remains unprocessed"
                )
            result = await execute(row)
        except Exception as error:
            if _provider_unavailable(error) or isinstance(error, SpendingLimitExceeded):
                provider_stopped.set()
            result = {
                "status": "error",
                "sampleId": row["sampleId"],
                "sourceDocumentId": row["sourceDocumentId"],
                "errorType": type(error).__name__,
                "error": str(error),
                "attempts": call_traces.get(row["sampleId"], [])
                + ([error.receipt] if isinstance(error, ProviderCallError) else []),
                "residualAttempts": [],
            }
            await asyncio.to_thread(
                atomic_write_json, work / f"{row['sampleId']}.error.json", result
            )
        outcomes.append(result)
        print(
            json.dumps(
                {
                    "completed": len(outcomes),
                    "total": len(samples),
                    "status": result["status"],
                    "sampleId": row["sampleId"],
                }
            ),
            flush=True,
        )

    # Admission is grouped; publication below still restores immutable plan order.
    # Stop only between waves so already admitted/billed work drains normally.
    await run_template_waves(
        samples,
        batch_size=config.request_batch_size,
        workers=config.max_concurrent_requests,
        execute=execute_one,
        should_stop=lambda: provider_stopped.is_set() or (work / "STOP_REQUESTED").exists(),
    )
    await batcher.drain()
    by_id = {row["sampleId"]: row for row in outcomes}
    ordered = [by_id[row["sampleId"]] for row in samples if row["sampleId"] in by_id]
    unprocessed = [row["sampleId"] for row in samples if row["sampleId"] not in by_id]
    failed = [row for row in ordered if row["status"] != "passed"]
    receipts = [attempt for row in ordered for attempt in row.get("attempts", [])]
    receipts.extend(attempt for row in ordered for attempt in row.get("residualAttempts", []))
    receipts.extend(numeric_receipts)
    summary = {
        "documents": len(samples),
        "passed": len(ordered) - len(failed),
        "failed": len(failed),
        "unprocessed": len(unprocessed),
        "wallSeconds": time.perf_counter() - started,
        "peakRssKiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "providerRequests": sum(r["usage"]["requests"] for r in receipts),
        "newProviderRequests": sum(
            r["usage"]["requests"] for r in receipts if not r["cacheReused"]
        ),
        "estimatedCostUsd": sum(float(r["usage"]["estimatedCostUsd"]) for r in receipts),
        "newEstimatedCostUsd": sum(
            float(r["usage"]["estimatedCostUsd"]) for r in receipts if not r["cacheReused"]
        ),
        "trainingPublished": False,
        "spending": spending.snapshot(),
    }
    atomic_write_json(work / "summary.json", summary)
    if failed or unprocessed:
        atomic_write_json(work / "unprocessed.json", unprocessed)
        return {**summary, "workRoot": str(work), "status": "incomplete"}
    if any(sha256_file(root / path) != digest for path, digest in implementation.items()):
        raise ValueError(
            "implementation changed during generation; "
            "revalidate cached responses before publication"
        )
    transaction = {
        "config": config.model_dump(mode="json"),
        "implementationFiles": implementation,
        "implementationSha256": sha256_file(Path(__file__)),
        "targetImplementationSha256": sha256_file(Path(targets.__file__)),
        "rendererImplementationSha256": sha256_file(Path(render.__file__)),
    }
    staged = StagedArtifactRun(
        output_parent=root / config.output_dir,
        run_name=config.run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    staged.recover_interrupted_temporary_files()
    artifacts = []

    def publish(path: str, value: Any, raw: bool = False) -> None:
        (staged.publish_bytes if raw else staged.publish_json)(path, value)
        artifacts.append(path)

    publish("transaction.json", transaction)
    publish("summary.json", summary)
    publish("numeric-contract-attempts.json", numeric_receipts)
    publish(
        "numeric-contracts.json",
        {
            source_id: {
                key: contract.model_dump(mode="json") for key, contract in contracts.items()
            }
            for source_id, contracts in numeric_contracts.items()
        },
    )
    publish("plan.jsonl", b"".join(canonical_json_bytes(row) + b"\n" for row in samples), True)
    target_rows = []
    for row in ordered:
        source = sources[row["sourceDocumentId"]]
        prefix = f"cases/{row['sampleId']}"
        publish(f"{prefix}/source.txt", source.source, True)
        publish(f"{prefix}/source-target.json", source.source_target)
        publish(f"{prefix}/target.json", row["target"])
        publish(f"{prefix}/target-receipt.json", row["targetReceipt"])
        publish(f"{prefix}/auxiliary-values.json", row["auxiliaryValues"])
        publish(f"{prefix}/numeric-auxiliary.json", row["numericAuxiliary"])
        publish(f"{prefix}/agent-stage.json", row["agentStage"])
        publish(f"{prefix}/generation-ledger.json", row["ledger"])
        publish(f"{prefix}/generation-attempts.json", row["attempts"])
        publish(f"{prefix}/residual-attempts.json", row["residualAttempts"])
        publish(f"{prefix}/goods-identities.json", row["goodsIdentities"])
        publish(f"{prefix}/render-result.json", row["renderResult"])
        publish(f"{prefix}/rendered.txt", row["rendered"].encode(), True)
        target_rows.append(
            {
                key: row[key]
                for key in (
                    "sampleId",
                    "sourceDocumentId",
                    "target",
                    "targetSha256",
                    "auxiliaryValues",
                    "numericAuxiliary",
                )
            }
        )
    publish(
        "targets.jsonl", b"".join(canonical_json_bytes(row) + b"\n" for row in target_rows), True
    )
    staged.commit(
        expected_artifacts=artifacts,
        metadata={"documents": len(ordered), "status": "complete_targets_and_rendering_validated"},
    )
    pin = {
        "path": str(staged.final_root.relative_to(root)),
        "commit_sha256": sha256_file(staged.final_root / "_COMMIT.json"),
        "transaction_sha256": staged.transaction_sha256,
    }
    render_config = {
        "schema_version": 1,
        "task": "bill_of_lading_compiled_raw_text_pipeline_v1",
        "run_name": config.run_name + "-published",
        "output_dir": "artifacts/kie-synthesis-production/synthesis-runs",
        "environment_file": config.environment_file,
        "inputs": {
            "template_run": config.template_run.model_dump(mode="json"),
            "sample_plan_run": pin,
            "sample_plan": {
                "path": pin["path"] + "/plan.jsonl",
                "sha256": sha256_file(staged.final_root / "plan.jsonl"),
                "records": len(ordered),
            },
            "synthetic_target_run": pin,
            "synthetic_targets": {
                "path": pin["path"] + "/targets.jsonl",
                "sha256": sha256_file(staged.final_root / "targets.jsonl"),
                "records": len(ordered),
            },
            "residual_replay_run": pin,
            "iso3166_snapshot": config.iso3166_snapshot.model_dump(mode="json"),
        },
        "prompts": {"residual_renderer": config.residual_prompt.model_dump(mode="json")},
        "provider": config.provider.model_dump(mode="json"),
        "workflow": {
            "documents": len(ordered),
            "controlled_target_seed": config.seed,
            "target_schema_version": "5.0.0-experimental",
            "target_generation": "complete_latest_schema_targets_v1",
            "max_concurrent_requests": config.max_concurrent_requests,
            "max_requests_per_document": 1,
            "provider_launch_authorized": False,
            "require_exact_topology": True,
            "require_source_carrier": True,
            "require_every_slot_bound_once": True,
            "require_exact_literal_regions": True,
            "require_page_markers_unchanged": True,
            "require_line_endings_preserved": True,
            "publish_training_records": True,
        },
    }
    DescendantConfig.model_validate_json(canonical_json_bytes(render_config), strict=True)
    renderer_config_path = work / "render-config.json"
    atomic_write_json(renderer_config_path, render_config)
    published = await render.run_descendants(renderer_config_path)
    return {
        **summary,
        "status": "published",
        "trainingPublished": True,
        "datasetRoot": str(published),
        "generationRoot": str(staged.final_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.config.resolve()))
    print(json.dumps(result, indent=2), flush=True)
    if not result.get("trainingPublished"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
