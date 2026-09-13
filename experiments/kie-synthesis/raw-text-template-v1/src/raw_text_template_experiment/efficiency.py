from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import yaml
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .analysis import _failure_category
from .models import AgentBindingProposal, CriticAgentOutput
from .optimization_contract import (
    DISCRIMINATED_BINDING_ADAPTER,
    DISCRIMINATED_CRITIC_ADAPTER,
    discriminate_binding,
    discriminate_critic,
    project_legacy_candidate,
    project_legacy_critic_candidate,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_LINE_ID = re.compile(r"^L(?P<number>[0-9]{5})$")
_BINDING_COLUMNS = (
    "bindingId",
    "sourceBindingIds",
    "logicalKey",
    "renderMode",
    "valueKind",
    "groupKind",
    "groupKey",
    "targetPathIds",
    "targetRelationship",
    "independentTargetFactComponentPathIds",
    "derivation",
    "dependencyPathIds",
    "dependencyBindings",
    "occurrenceIds",
)
_OCCURRENCE_COLUMNS = (
    "occurrenceId",
    "sourceBindingId",
    "lineStartNumber",
    "lineEndNumber",
    "sourceText",
    "occurrenceIndex",
    "exactMatchCount",
    "exactMatchCandidates",
)


class CommittedRunInput(BaseModel):
    model_config = _STRICT

    path: str
    commit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PricingScenario(BaseModel):
    model_config = _STRICT

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    input_usd_per_million: Decimal = Field(ge=0)
    cached_input_usd_per_million: Decimal = Field(ge=0)
    cache_write_multiplier: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(ge=0)


class NamedCommittedRun(BaseModel):
    model_config = _STRICT

    name: str = Field(min_length=1)
    run: CommittedRunInput


class EfficiencyAuditConfig(BaseModel):
    model_config = _STRICT

    schema_version: int
    run_name: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    baseline_transfer_run: CommittedRunInput
    final_transfer_run: CommittedRunInput
    final_development_run: CommittedRunInput
    carrier_audit_run: CommittedRunInput
    pricing_scenarios: tuple[PricingScenario, ...]
    model_probe_runs: tuple[NamedCommittedRun, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def named_inputs_are_unique(self) -> EfficiencyAuditConfig:
        names = tuple(probe.name for probe in self.model_probe_runs)
        if len(set(names)) != len(names):
            raise ValueError("model probe names must be unique")
        paths = tuple(probe.run.path for probe in self.model_probe_runs)
        if len(set(paths)) != len(paths):
            raise ValueError("model probe run paths must be unique")
        return self


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def _verify_run(spec: CommittedRunInput) -> tuple[Path, dict[str, Any]]:
    run_dir = Path(spec.path).resolve(strict=True)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError(f"input run is not a regular directory: {run_dir}")
    commit_path = run_dir / "_COMMIT.json"
    if sha256_file(commit_path) != spec.commit_sha256:
        raise ValueError(f"commit receipt hash differs: {run_dir}")
    commit = _load_object(commit_path)
    artifacts = commit.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"committed run has no artifact ledger: {run_dir}")
    root = run_dir.resolve()
    for row in artifacts:
        relative = row.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError(f"invalid artifact ledger entry: {run_dir}")
        path = (run_dir / relative).resolve(strict=True)
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe or missing committed artifact: {run_dir / relative}")
        if sha256_file(path) != row.get("sha256"):
            raise ValueError(f"committed artifact hash differs: {run_dir / relative}")
    return run_dir, commit


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _member_bytes(key: str, value: Any) -> int:
    return len(_json_bytes({key: value})) - 2


def _user_payload(stage: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    prompts: list[str] = []
    for message in stage["messages"]:
        if message.get("kind") != "request":
            continue
        for part in message.get("parts", ()):
            if part.get("part_kind") == "user-prompt":
                content = part.get("content")
                if not isinstance(content, str):
                    raise ValueError("user prompt is not text")
                prompts.append(content)
    if len(prompts) != 1:
        raise ValueError(f"stage contains {len(prompts)} user prompts instead of one")
    value = json.loads(prompts[0])
    if not isinstance(value, dict):
        raise ValueError("stage user prompt is not a JSON object")
    return value, len(prompts[0].encode("utf-8"))


def _line_number(value: Any) -> int:
    if not isinstance(value, str) or (match := _LINE_ID.fullmatch(value)) is None:
        raise ValueError(f"invalid line ID in critic inventory: {value!r}")
    return int(match.group("number"))


def _compact_critic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly encode the current critic payload with immutable short references.

    This is an offline contract probe. It deliberately retains every semantic value while moving
    verbose host provenance into indexed tables and replacing long masked-template markers.
    """

    required = {
        "allowedTargetPaths",
        "allowedRemovalLogicalKeys",
        "bindingInventory",
        "maskedTemplate",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"critic payload lacks required keys: {missing}")
    target_paths = payload["allowedTargetPaths"]
    bindings = payload["bindingInventory"]
    removals = payload["allowedRemovalLogicalKeys"]
    masked_template = payload["maskedTemplate"]
    if (
        not isinstance(target_paths, list)
        or not isinstance(bindings, list)
        or not isinstance(removals, list)
        or not isinstance(masked_template, str)
    ):
        raise ValueError("critic payload has invalid compactable field types")
    if len(set(target_paths)) != len(target_paths):
        raise ValueError("critic target paths are not unique")
    path_ids = {value: f"path_{index:04d}" for index, value in enumerate(target_paths)}
    logical_keys = [row.get("logicalKey") for row in bindings]
    if any(not isinstance(value, str) for value in logical_keys):
        raise ValueError("critic binding inventory has a non-text logical key")
    if len(set(logical_keys)) != len(logical_keys):
        raise ValueError("critic binding logical keys are not unique")
    binding_ids = {value: f"binding_{index:04d}" for index, value in enumerate(logical_keys)}
    if any(value not in binding_ids for value in removals):
        raise ValueError("critic removal key is absent from the binding inventory")

    source_binding_ids: list[str] = []
    source_binding_index: dict[str, int] = {}

    def source_index(value: Any) -> int:
        if not isinstance(value, str):
            raise ValueError("critic source binding ID is not text")
        if value not in source_binding_index:
            source_binding_index[value] = len(source_binding_ids)
            source_binding_ids.append(value)
        return source_binding_index[value]

    occurrence_rows: list[list[Any]] = []
    binding_rows: list[list[Any]] = []
    compact_template = masked_template
    expected_binding_keys = {
        "sourceBindingIds",
        "logicalKey",
        "renderMode",
        "valueKind",
        "groupKind",
        "groupKey",
        "targetPaths",
        "targetRelationship",
        "independentTargetFactComponents",
        "derivation",
        "dependencyPaths",
        "dependencyBindings",
        "occurrences",
    }
    expected_occurrence_keys = {
        "sourceBindingId",
        "lineStart",
        "lineEnd",
        "sourceText",
        "occurrenceIndex",
        "exactMatchCount",
        "exactMatchCandidates",
    }
    for binding_index, binding in enumerate(bindings):
        if not isinstance(binding, dict) or set(binding) != expected_binding_keys:
            raise ValueError("critic binding inventory shape differs from the pinned contract")
        logical_key = cast(str, binding["logicalKey"])
        binding_id = binding_ids[logical_key]
        render_mode = binding["renderMode"]
        marker = f"⟦{logical_key}:{render_mode}⟧"
        if marker not in compact_template:
            raise ValueError(f"binding marker is absent from masked template: {logical_key}")
        compact_template = compact_template.replace(marker, f"⟦{binding_id}⟧")
        occurrence_ids: list[str] = []
        for occurrence in binding["occurrences"]:
            if not isinstance(occurrence, dict) or set(occurrence) != expected_occurrence_keys:
                raise ValueError("critic occurrence shape differs from the pinned contract")
            occurrence_id = f"occurrence_{len(occurrence_rows):05d}"
            occurrence_ids.append(occurrence_id)
            occurrence_rows.append(
                [
                    occurrence_id,
                    source_index(occurrence["sourceBindingId"]),
                    _line_number(occurrence["lineStart"]),
                    _line_number(occurrence["lineEnd"]),
                    occurrence["sourceText"],
                    occurrence["occurrenceIndex"],
                    occurrence["exactMatchCount"],
                    occurrence["exactMatchCandidates"],
                ]
            )

        def path_id(value: Any) -> str:
            if value not in path_ids:
                raise ValueError(f"binding path is absent from allowed target paths: {value!r}")
            return path_ids[value]

        binding_rows.append(
            [
                binding_id,
                [source_index(value) for value in binding["sourceBindingIds"]],
                logical_key,
                render_mode,
                binding["valueKind"],
                binding["groupKind"],
                binding["groupKey"],
                [path_id(value) for value in binding["targetPaths"]],
                binding["targetRelationship"],
                [
                    [path_id(value) for value in component]
                    for component in binding["independentTargetFactComponents"]
                ],
                binding["derivation"],
                [path_id(value) for value in binding["dependencyPaths"]],
                binding["dependencyBindings"],
                occurrence_ids,
            ]
        )
        if binding_id != f"binding_{binding_index:04d}":
            raise AssertionError("non-contiguous compact binding IDs")

    output = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "allowedTargetPaths",
            "allowedRemovalLogicalKeys",
            "bindingInventory",
            "maskedTemplate",
        }
    }
    output.update(
        {
            "compactContract": {
                "schemaVersion": 1,
                "lineIdFormat": "L%05d",
                "bindingColumns": _BINDING_COLUMNS,
                "occurrenceColumns": _OCCURRENCE_COLUMNS,
                "sourceBindingIdEncoding": "zero-based index into sourceBindingIdTable",
            },
            "targetPathTable": [[path_ids[value], value] for value in target_paths],
            "sourceBindingIdTable": source_binding_ids,
            "occurrenceRows": occurrence_rows,
            "bindingRows": binding_rows,
            "removableBindingIds": [binding_ids[value] for value in removals],
            "maskedTemplate": compact_template,
        }
    )
    return output


def _expand_critic_payload(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Reverse `_compact_critic_payload`; used to prove information preservation."""

    contract = compact["compactContract"]
    if (
        contract.get("schemaVersion") != 1
        or tuple(contract.get("bindingColumns", ())) != _BINDING_COLUMNS
        or tuple(contract.get("occurrenceColumns", ())) != _OCCURRENCE_COLUMNS
    ):
        raise ValueError("unknown compact critic contract")
    target_table = compact["targetPathTable"]
    path_by_id = {row[0]: row[1] for row in target_table}
    if len(path_by_id) != len(target_table):
        raise ValueError("compact target path IDs are not unique")
    source_ids = compact["sourceBindingIdTable"]
    occurrence_by_id: dict[str, dict[str, Any]] = {}
    for row in compact["occurrenceRows"]:
        (
            occurrence_id,
            source_index,
            line_start,
            line_end,
            source_text,
            index,
            count,
            candidates,
        ) = row
        if occurrence_id in occurrence_by_id:
            raise ValueError("compact occurrence IDs are not unique")
        occurrence_by_id[occurrence_id] = {
            "sourceBindingId": source_ids[source_index],
            "lineStart": f"L{line_start:05d}",
            "lineEnd": f"L{line_end:05d}",
            "sourceText": source_text,
            "occurrenceIndex": index,
            "exactMatchCount": count,
            "exactMatchCandidates": candidates,
        }
    bindings: list[dict[str, Any]] = []
    binding_key_by_id: dict[str, str] = {}
    marker_by_id: dict[str, str] = {}
    for row in compact["bindingRows"]:
        (
            binding_id,
            source_indexes,
            logical_key,
            render_mode,
            value_kind,
            group_kind,
            group_key,
            target_ids,
            target_relationship,
            component_ids,
            derivation,
            dependency_ids,
            dependency_bindings,
            occurrence_ids,
        ) = row
        if binding_id in binding_key_by_id:
            raise ValueError("compact binding IDs are not unique")
        binding_key_by_id[binding_id] = logical_key
        marker_by_id[binding_id] = f"⟦{logical_key}:{render_mode}⟧"
        bindings.append(
            {
                "sourceBindingIds": [source_ids[index] for index in source_indexes],
                "logicalKey": logical_key,
                "renderMode": render_mode,
                "valueKind": value_kind,
                "groupKind": group_kind,
                "groupKey": group_key,
                "targetPaths": [path_by_id[value] for value in target_ids],
                "targetRelationship": target_relationship,
                "independentTargetFactComponents": [
                    [path_by_id[value] for value in component] for component in component_ids
                ],
                "derivation": derivation,
                "dependencyPaths": [path_by_id[value] for value in dependency_ids],
                "dependencyBindings": dependency_bindings,
                "occurrences": [occurrence_by_id[value] for value in occurrence_ids],
            }
        )
    masked_template = compact["maskedTemplate"]
    for binding_id, marker in marker_by_id.items():
        short = f"⟦{binding_id}⟧"
        if short not in masked_template:
            raise ValueError(f"compact binding marker is absent: {binding_id}")
        masked_template = masked_template.replace(short, marker)
    output = {
        key: value
        for key, value in compact.items()
        if key
        not in {
            "compactContract",
            "targetPathTable",
            "sourceBindingIdTable",
            "occurrenceRows",
            "bindingRows",
            "removableBindingIds",
            "maskedTemplate",
        }
    }
    output.update(
        {
            "allowedTargetPaths": [row[1] for row in target_table],
            "allowedRemovalLogicalKeys": [
                binding_key_by_id[value] for value in compact["removableBindingIds"]
            ],
            "bindingInventory": bindings,
            "maskedTemplate": masked_template,
        }
    )
    return output


def _stage_disposition(stage: Mapping[str, Any]) -> str:
    status = stage["status"]
    output = stage.get("output")
    verdict = output.get("verdict") if isinstance(output, Mapping) else None
    return f"{status}:{verdict}" if verdict is not None else cast(str, status)


def _new_usage() -> dict[str, Any]:
    return {
        "stages": 0,
        "requests": 0,
        "inputTokens": 0,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
        "outputTokens": 0,
        "reasoningTokens": 0,
        "visibleOutputTokens": 0,
        "estimatedCostUsd": Decimal(0),
    }


def _add_usage(total: dict[str, Any], usage: Mapping[str, Any]) -> None:
    total["stages"] += 1
    for key in (
        "requests",
        "inputTokens",
        "cacheReadTokens",
        "cacheWriteTokens",
        "outputTokens",
        "reasoningTokens",
        "visibleOutputTokens",
    ):
        total[key] += int(usage.get(key, 0))
    total["estimatedCostUsd"] += Decimal(str(usage["estimatedCostUsd"]))


def _usage_row(role: str, disposition: str, usage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "disposition": disposition,
        **{
            key: str(value) if isinstance(value, Decimal) else value for key, value in usage.items()
        },
    }


def _scenario_cost(usage: Mapping[str, Any], scenario: PricingScenario) -> Decimal:
    million = Decimal(1_000_000)
    input_tokens = Decimal(usage["inputTokens"])
    cache_read = Decimal(usage["cacheReadTokens"])
    cache_write = Decimal(usage["cacheWriteTokens"])
    ordinary = input_tokens - cache_read - cache_write
    if ordinary < 0:
        raise ValueError("cache-token accounting exceeds input tokens")
    return (
        ordinary * scenario.input_usd_per_million
        + cache_read * scenario.cached_input_usd_per_million
        + cache_write * scenario.input_usd_per_million * scenario.cache_write_multiplier
        + Decimal(usage["outputTokens"]) * scenario.output_usd_per_million
    ) / million


def _retry_errors(messages: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    for message in messages:
        if message.get("kind") != "request":
            continue
        for part in message.get("parts", ()):
            if part.get("part_kind") != "retry-prompt":
                continue
            content = part.get("content")
            if not isinstance(content, list):
                raise ValueError("retry prompt did not retain structured validation errors")
            yield from content


def _response_cost(response: Mapping[str, Any]) -> Decimal:
    usage = response.get("usage")
    cost = usage.get("cost") if isinstance(usage, Mapping) else None
    if cost is None:
        details = response.get("provider_details")
        cost = details.get("cost") if isinstance(details, Mapping) else None
    if cost is None:
        raise ValueError("billed response lacks provider-reported cost")
    return Decimal(str(cost))


def _document_metrics(
    result: Mapping[str, Any],
    catalog: Mapping[str, Any] | None,
) -> dict[str, Any]:
    role_usage: dict[str, dict[str, Any]] = {
        "compiler": _new_usage(),
        "critic": _new_usage(),
    }
    retry_responses = 0
    retry_response_cost = Decimal(0)
    retry_validation_errors: Counter[str] = Counter()
    stages = (*result["compiler_stages"], *result["critic_stages"])
    for stage in stages:
        role = cast(str, stage["role"])
        _add_usage(role_usage[role], stage["usage"])
        errors = list(_retry_errors(stage["messages"]))
        retry_validation_errors.update(str(error.get("msg")) for error in errors)
        responses = [message for message in stage["messages"] if message.get("kind") == "response"]
        if len(responses) > 1:
            if not errors:
                raise ValueError("provider retried without a retained validation error")
            retry_responses += len(responses) - 1
            retry_response_cost += sum(
                (_response_cost(response) for response in responses[1:]),
                Decimal(0),
            )
    total_usage = _new_usage()
    for usage in role_usage.values():
        total_usage["stages"] += usage["stages"]
        for key in (
            "requests",
            "inputTokens",
            "cacheReadTokens",
            "cacheWriteTokens",
            "outputTokens",
            "reasoningTokens",
            "visibleOutputTokens",
        ):
            total_usage[key] += usage[key]
        total_usage["estimatedCostUsd"] += usage["estimatedCostUsd"]
    last_error = next(
        (str(stage["error_message"]) for stage in reversed(stages) if stage.get("error_message")),
        "",
    )
    return {
        "document_id": result["document_id"],
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "rejection_reason": " | ".join(result.get("rejection_reasons", ())),
        "last_error": last_error,
        "stages": total_usage["stages"],
        "requests": total_usage["requests"],
        "input_tokens": total_usage["inputTokens"],
        "cache_read_tokens": total_usage["cacheReadTokens"],
        "cache_write_tokens": total_usage["cacheWriteTokens"],
        "output_tokens": total_usage["outputTokens"],
        "reasoning_tokens": total_usage["reasoningTokens"],
        "visible_output_tokens": total_usage["visibleOutputTokens"],
        "cost_usd": total_usage["estimatedCostUsd"],
        "compiler_stages": role_usage["compiler"]["stages"],
        "compiler_requests": role_usage["compiler"]["requests"],
        "compiler_cost_usd": role_usage["compiler"]["estimatedCostUsd"],
        "critic_stages": role_usage["critic"]["stages"],
        "critic_requests": role_usage["critic"]["requests"],
        "critic_cost_usd": role_usage["critic"]["estimatedCostUsd"],
        "retry_responses": retry_responses,
        "retry_response_cost_usd": retry_response_cost,
        "retry_validation_errors": dict(sorted(retry_validation_errors.items())),
        "bindings": catalog.get("bindings") if catalog is not None else None,
        "occurrences": catalog.get("occurrences") if catalog is not None else None,
        "agent_assisted_bindings": (
            catalog.get("agentAssistedBindings") if catalog is not None else None
        ),
        "deterministic_bindings": (
            catalog.get("deterministicBindings") if catalog is not None else None
        ),
    }


def _route_label(provider: Mapping[str, Any]) -> str:
    provider_only = provider.get("provider_only")
    if isinstance(provider_only, list) and provider_only:
        return ",".join(str(value) for value in provider_only)
    return f"direct:{provider['kind']}"


def _ratio(numerator: Decimal | int | float, denominator: Decimal | int | float) -> str:
    denominator_decimal = Decimal(str(denominator))
    if denominator_decimal == 0:
        raise ValueError("cannot calculate a ratio against zero")
    return str(Decimal(str(numerator)) / denominator_decimal)


def _model_probe_audit(
    verified_probes: Sequence[tuple[NamedCommittedRun, Path, Mapping[str, Any]]],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    baseline_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Mapping[str, Any]]]:
    rows: list[dict[str, Any]] = []
    commits: dict[str, Mapping[str, Any]] = {}
    invalid_critic_retries_excluded = 0
    accepted_probe_critic_outputs = 0
    accepted_probe_binding_outputs = 0
    for named, run_dir, commit in verified_probes:
        commits[named.name] = commit
        results = list(_stream_jsonl(run_dir / "results.jsonl"))
        if len(results) != 1:
            raise ValueError(f"model probe must contain exactly one result: {run_dir}")
        result = results[0]
        catalog_rows = list(_stream_jsonl(run_dir / "catalog.jsonl"))
        if len(catalog_rows) > 1:
            raise ValueError(f"model probe has multiple catalog rows: {run_dir}")
        catalog = catalog_rows[0] if catalog_rows else None
        if (result["status"] == "certified") != (catalog is not None):
            raise ValueError(f"model probe catalog/status mismatch: {run_dir}")
        metrics = _document_metrics(result, catalog)
        document_id = cast(str, result["document_id"])
        baseline = baseline_metrics.get(document_id)
        if baseline is None:
            raise ValueError(f"model probe document is absent from the baseline: {document_id}")
        probe_config = _load_object(run_dir / "config.json")
        compiler_provider = probe_config["compiler_provider"]
        critic_provider = probe_config["critic_provider"]
        for stage in (*result["compiler_stages"], *result["critic_stages"]):
            output = stage.get("output")
            if isinstance(output, Mapping):
                binding_rows = (
                    output.get("bindings", ())
                    if stage["role"] == "compiler"
                    else output.get("additional_bindings", ())
                )
                for binding in binding_rows:
                    legacy_binding = AgentBindingProposal.model_validate_json(_json_bytes(binding))
                    DISCRIMINATED_BINDING_ADAPTER.validate_python(
                        discriminate_binding(legacy_binding).model_dump(mode="python")
                    )
                    accepted_probe_binding_outputs += 1
                if stage["role"] == "critic":
                    legacy_critic = CriticAgentOutput.model_validate_json(_json_bytes(output))
                    DISCRIMINATED_CRITIC_ADAPTER.validate_python(
                        discriminate_critic(legacy_critic).model_dump(mode="python")
                    )
                    accepted_probe_critic_outputs += 1
            if stage["role"] != "critic":
                continue
            for error in _retry_errors(stage["messages"]):
                invalid_input = error.get("input")
                if not isinstance(invalid_input, dict):
                    raise ValueError("recorded critic retry input is not an object")
                try:
                    DISCRIMINATED_CRITIC_ADAPTER.validate_python(
                        project_legacy_critic_candidate(invalid_input)
                    )
                except ValidationError:
                    invalid_critic_retries_excluded += 1
                else:
                    raise ValueError(
                        "recorded invalid critic output remains valid under discriminated verdicts"
                    )
        certified = metrics["status"] == "certified"
        row: dict[str, Any] = {
            "probe": named.name,
            "run": run_dir.name,
            "commit_sha256": named.run.commit_sha256,
            "document_id": document_id,
            "status": metrics["status"],
            "compiler_model": compiler_provider["model"],
            "compiler_route": _route_label(compiler_provider),
            "critic_model": critic_provider["model"],
            "critic_route": _route_label(critic_provider),
            "structured_output_profile": compiler_provider.get(
                "native_structured_output_profile", "provider_default"
            ),
            "baseline_compiler_model": baseline_config["compiler_provider"]["model"],
            "baseline_compiler_route": _route_label(baseline_config["compiler_provider"]),
            "baseline_critic_model": baseline_config["critic_provider"]["model"],
            "baseline_critic_route": _route_label(baseline_config["critic_provider"]),
            **{
                key: str(value) if isinstance(value, Decimal) else value
                for key, value in metrics.items()
                if key != "document_id"
            },
            "baseline_status": baseline["status"],
            "baseline_cost_usd": str(baseline["cost_usd"]),
            "baseline_elapsed_seconds": baseline["elapsed_seconds"],
            "baseline_stages": baseline["stages"],
            "baseline_requests": baseline["requests"],
            "baseline_output_tokens": baseline["output_tokens"],
            "baseline_compiler_cost_usd": str(baseline["compiler_cost_usd"]),
            "baseline_critic_cost_usd": str(baseline["critic_cost_usd"]),
            "baseline_bindings": baseline["bindings"],
            "baseline_occurrences": baseline["occurrences"],
            "baseline_agent_assisted_bindings": baseline["agent_assisted_bindings"],
            "cost_fraction_of_baseline": (
                _ratio(metrics["cost_usd"], baseline["cost_usd"]) if certified else None
            ),
            "elapsed_fraction_of_baseline": (
                _ratio(metrics["elapsed_seconds"], baseline["elapsed_seconds"])
                if certified
                else None
            ),
            "output_token_fraction_of_baseline": (
                _ratio(metrics["output_tokens"], baseline["output_tokens"]) if certified else None
            ),
            "agent_assisted_binding_delta": (
                int(metrics["agent_assisted_bindings"]) - int(baseline["agent_assisted_bindings"])
                if certified
                else None
            ),
            "quality_equivalence_proven": False,
        }
        rows.append(row)
    summary = {
        "runs": len(rows),
        "certifiedRuns": sum(row["status"] == "certified" for row in rows),
        "zeroRequestCapabilityFailures": sum(
            row["requests"] == 0 and row["status"] != "certified" for row in rows
        ),
        "paidProbeCostUsd": str(
            sum(
                (Decimal(str(row["cost_usd"])) for row in rows),
                Decimal(0),
            )
        ),
        "acceptedProbeBindingOutputsConverted": accepted_probe_binding_outputs,
        "acceptedProbeCriticOutputsConverted": accepted_probe_critic_outputs,
        "criticRetryValidationErrorsExcludedByDiscriminatedSchema": (
            invalid_critic_retries_excluded
        ),
    }
    return rows, summary, commits


def _baseline_transfer_audit(
    run_dir: Path,
    comparison_document_ids: frozenset[str] = frozenset(),
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    selection = _load_object(run_dir / "selection-manifest.json")
    ordinal_by_id = {row["document_id"]: row["ordinal"] for row in selection["rows"]}
    usage_by_disposition: dict[tuple[str, str], dict[str, Any]] = defaultdict(_new_usage)
    usage_by_compiler_pass: dict[tuple[int, str], dict[str, Any]] = defaultdict(_new_usage)
    payload_components: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    compact_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retry_error_counts: Counter[str] = Counter()
    retry_responses = 0
    retry_response_cost = Decimal(0)
    retry_inputs_excluded_by_discriminated_schema = 0
    accepted_discriminated_bindings = 0
    accepted_discriminated_critic_outputs = 0
    critic_inventory_bindings = 0
    critic_inventory_occurrences = 0
    documents = 0
    total_usage = _new_usage()
    comparison_catalog = {
        cast(str, row["documentId"]): row
        for row in _stream_jsonl(run_dir / "catalog.jsonl")
        if row["documentId"] in comparison_document_ids
    }
    comparison_metrics: dict[str, dict[str, Any]] = {}
    for result in _stream_jsonl(run_dir / "results.jsonl"):
        documents += 1
        document_id = cast(str, result["document_id"])
        if document_id in comparison_document_ids:
            comparison_metrics[document_id] = _document_metrics(
                result,
                comparison_catalog.get(document_id),
            )
        if result["status"] != "certified":
            terminal_category = _failure_category(result["rejection_reasons"])
            stages = (*result["compiler_stages"], *result["critic_stages"])
            last_error = next(
                (
                    str(stage["error_message"])
                    for stage in reversed(stages)
                    if stage.get("error_message")
                ),
                "",
            )
            underlying_category = (
                _failure_category((last_error,))
                if terminal_category == "per_document_cost_guard" and last_error
                else terminal_category
            )
            failures.append(
                {
                    "ordinal": ordinal_by_id[document_id],
                    "document_id": document_id,
                    "baseline_terminal_category": terminal_category,
                    "baseline_underlying_category": underlying_category,
                    "baseline_terminal_reason": " | ".join(result["rejection_reasons"]),
                }
            )
        for stage in (*result["compiler_stages"], *result["critic_stages"]):
            role = cast(str, stage["role"])
            disposition = _stage_disposition(stage)
            usage = stage["usage"]
            _add_usage(total_usage, usage)
            _add_usage(usage_by_disposition[(role, disposition)], usage)
            if role == "compiler":
                _add_usage(
                    usage_by_compiler_pass[(int(stage["pass_number"]), disposition)],
                    usage,
                )
            payload, current_bytes = _user_payload(stage)
            pass_kind = "initial" if int(stage["pass_number"]) == 1 else "repair"
            for key, value in payload.items():
                payload_components[(role, pass_kind, key)].append(_member_bytes(key, value))
            payload_components[(role, pass_kind, "__total__")].append(current_bytes)
            if role == "critic":
                critic_inventory_bindings += len(payload["bindingInventory"])
                critic_inventory_occurrences += sum(
                    len(binding["occurrences"]) for binding in payload["bindingInventory"]
                )
                compact = _compact_critic_payload(payload)
                expanded = _expand_critic_payload(compact)
                if canonical_json_bytes(expanded) != canonical_json_bytes(payload):
                    raise ValueError("compact critic payload failed its lossless round trip")
                compact_bytes = len(_json_bytes(compact))
                compact_rows.append(
                    {
                        "document_id": document_id,
                        "critic_pass": stage["pass_number"],
                        "current_bytes": current_bytes,
                        "compact_bytes": compact_bytes,
                        "bytes_saved": current_bytes - compact_bytes,
                        "reduction_fraction": (current_bytes - compact_bytes) / current_bytes,
                    }
                )
            output = stage.get("output")
            if isinstance(output, Mapping):
                binding_rows = (
                    output.get("bindings", ())
                    if role == "compiler"
                    else output.get("additional_bindings", ())
                )
                for binding in binding_rows:
                    legacy = AgentBindingProposal.model_validate_json(_json_bytes(binding))
                    discriminated = discriminate_binding(legacy)
                    DISCRIMINATED_BINDING_ADAPTER.validate_python(
                        discriminated.model_dump(mode="python")
                    )
                    accepted_discriminated_bindings += 1
                if role == "critic":
                    legacy_critic = CriticAgentOutput.model_validate_json(_json_bytes(output))
                    discriminated_critic = discriminate_critic(legacy_critic)
                    DISCRIMINATED_CRITIC_ADAPTER.validate_python(
                        discriminated_critic.model_dump(mode="python")
                    )
                    accepted_discriminated_critic_outputs += 1
            if role == "compiler":
                errors = list(_retry_errors(stage["messages"]))
                for error in errors:
                    retry_error_counts[str(error.get("msg"))] += 1
                    invalid_input = error.get("input")
                    if not isinstance(invalid_input, dict):
                        raise ValueError("recorded retry input is not an object")
                    try:
                        DISCRIMINATED_BINDING_ADAPTER.validate_json(
                            _json_bytes(project_legacy_candidate(invalid_input))
                        )
                    except ValidationError:
                        retry_inputs_excluded_by_discriminated_schema += 1
                    else:
                        raise ValueError(
                            "recorded retry input remains valid under discriminated schema"
                        )
                responses = [
                    message for message in stage["messages"] if message.get("kind") == "response"
                ]
                if len(responses) > 1:
                    if not errors:
                        raise ValueError("compiler retried without retained validation errors")
                    retry_responses += len(responses) - 1
                    retry_response_cost += sum(
                        (_response_cost(response) for response in responses[1:]),
                        Decimal(0),
                    )

    if documents != 200:
        raise ValueError(f"baseline transfer run has {documents} documents instead of 200")
    stage_rows = [
        _usage_row(role, disposition, usage)
        for (role, disposition), usage in sorted(usage_by_disposition.items())
    ]
    compiler_pass_rows = [
        {
            "compiler_pass": pass_number,
            **_usage_row("compiler", disposition, usage),
        }
        for (pass_number, disposition), usage in sorted(usage_by_compiler_pass.items())
    ]
    component_rows: list[dict[str, Any]] = []
    for (role, pass_kind, component), values in sorted(payload_components.items()):
        component_rows.append(
            {
                "role": role,
                "pass_kind": pass_kind,
                "component": component,
                "stages": len(values),
                "total_bytes": sum(values),
                "mean_bytes": sum(values) / len(values),
                "max_bytes": max(values),
            }
        )
    old_schema_bytes = len(canonical_json_bytes(AgentBindingProposal.model_json_schema()))
    new_schema_bytes = len(canonical_json_bytes(DISCRIMINATED_BINDING_ADAPTER.json_schema()))
    old_critic_schema_bytes = len(canonical_json_bytes(CriticAgentOutput.model_json_schema()))
    new_critic_schema_bytes = len(canonical_json_bytes(DISCRIMINATED_CRITIC_ADAPTER.json_schema()))
    retry_errors = sum(retry_error_counts.values())
    expected_retry_errors = {
        "Value error, only deterministic_derived may declare derivation inputs",
        "Value error, deterministic_derived requires derivation and dependencies",
    }
    if set(retry_error_counts) != expected_retry_errors:
        raise ValueError(
            "recorded compiler retry errors differ from the audited conditional invariants"
        )
    if retry_inputs_excluded_by_discriminated_schema != retry_errors:
        raise ValueError("not every recorded invalid binding is excluded by the new schema")
    contract_probe = {
        "acceptedLegacyBindingsConverted": accepted_discriminated_bindings,
        "acceptedLegacyBindingsRejected": 0,
        "legacyBindingSchemaBytes": old_schema_bytes,
        "discriminatedBindingSchemaBytes": new_schema_bytes,
        "schemaByteDelta": new_schema_bytes - old_schema_bytes,
        "acceptedLegacyCriticOutputsConverted": accepted_discriminated_critic_outputs,
        "acceptedLegacyCriticOutputsRejected": 0,
        "legacyCriticSchemaBytes": old_critic_schema_bytes,
        "discriminatedCriticSchemaBytes": new_critic_schema_bytes,
        "criticSchemaByteDelta": new_critic_schema_bytes - old_critic_schema_bytes,
        "compilerRetryResponses": retry_responses,
        "compilerRetryValidationErrors": retry_errors,
        "compilerRetryValidationErrorCounts": dict(sorted(retry_error_counts.items())),
        "retryInputsExcludedByDiscriminatedSchema": (retry_inputs_excluded_by_discriminated_schema),
        "compilerRetryResponseProviderCostUsd": str(retry_response_cost),
        "criticPayloadsLosslesslyRoundTripped": len(compact_rows),
        "criticInventoryBindingRows": critic_inventory_bindings,
        "criticInventoryOccurrenceRows": critic_inventory_occurrences,
    }
    return (
        contract_probe,
        stage_rows,
        compiler_pass_rows,
        component_rows,
        [
            *failures,
            {
                "__compact_rows__": compact_rows,
                "__total_usage__": total_usage,
                "__comparison_metrics__": comparison_metrics,
            },
        ],
    )


def _final_results_audit(
    run_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    selection = _load_object(run_dir / "selection-manifest.json")
    selection_by_id = {row["document_id"]: row for row in selection["rows"]}
    status_by_id: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    for result in _stream_jsonl(run_dir / "results.jsonl"):
        document_id = cast(str, result["document_id"])
        status = cast(str, result["status"])
        status_by_id[document_id] = status
        if status == "certified":
            continue
        metrics = _document_metrics(result, None)
        compiler_statuses = Counter(stage["status"] for stage in result["compiler_stages"])
        critic_statuses = Counter(stage["status"] for stage in result["critic_stages"])
        critic_verdicts = Counter(
            stage["output"]["verdict"]
            for stage in result["critic_stages"]
            if isinstance(stage.get("output"), Mapping)
        )
        selected = selection_by_id[document_id]
        failures.append(
            {
                "ordinal": selected["ordinal"],
                "document_id": document_id,
                "pages": selected["page_count"],
                "lines": selected["ocr_lines"],
                "status": status,
                "rejection_reason": metrics["rejection_reason"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "stages": metrics["stages"],
                "requests": metrics["requests"],
                "cost_usd": str(metrics["cost_usd"]),
                "compiler_stages": metrics["compiler_stages"],
                "compiler_requests": metrics["compiler_requests"],
                "compiler_cost_usd": str(metrics["compiler_cost_usd"]),
                "compiler_successes": compiler_statuses["success"],
                "compiler_host_rejections": compiler_statuses["host_rejected"],
                "critic_stages": metrics["critic_stages"],
                "critic_requests": metrics["critic_requests"],
                "critic_cost_usd": str(metrics["critic_cost_usd"]),
                "critic_successes": critic_statuses["success"],
                "critic_host_rejections": critic_statuses["host_rejected"],
                "critic_pass_verdicts": critic_verdicts["pass"],
                "critic_revise_verdicts": critic_verdicts["revise"],
            }
        )
    return status_by_id, failures


def publish_efficiency_audit(config_path: Path) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = EfficiencyAuditConfig.model_validate_json(_json_bytes(raw))
    if config.schema_version != 1:
        raise ValueError(f"unsupported efficiency-audit schema: {config.schema_version}")
    baseline_dir, baseline_commit = _verify_run(config.baseline_transfer_run)
    final_transfer_dir, final_transfer_commit = _verify_run(config.final_transfer_run)
    development_dir, development_commit = _verify_run(config.final_development_run)
    carrier_dir, carrier_commit = _verify_run(config.carrier_audit_run)
    verified_probes: list[tuple[NamedCommittedRun, Path, Mapping[str, Any]]] = []
    probe_document_ids: set[str] = set()
    for named in config.model_probe_runs:
        run_dir, commit = _verify_run(named.run)
        selection = _load_object(run_dir / "selection-manifest.json")
        rows = selection.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError(f"model probe selection must contain exactly one row: {run_dir}")
        document_id = rows[0].get("document_id")
        if not isinstance(document_id, str):
            raise ValueError(f"model probe selection lacks a document ID: {run_dir}")
        probe_document_ids.add(document_id)
        verified_probes.append((named, run_dir, commit))

    (
        contract_probe,
        stage_rows,
        compiler_pass_rows,
        component_rows,
        failure_and_usage,
    ) = _baseline_transfer_audit(baseline_dir, frozenset(probe_document_ids))
    internals = failure_and_usage.pop()
    compact_rows = internals["__compact_rows__"]
    baseline_usage = internals["__total_usage__"]
    comparison_metrics = internals["__comparison_metrics__"]
    failures = failure_and_usage
    final_status, remaining_failures = _final_results_audit(final_transfer_dir)
    if len(remaining_failures) != 1:
        raise ValueError("final transfer run must retain exactly the one audited hard case")
    for failure in failures:
        failure["final_status"] = final_status[failure["document_id"]]
        failure["recovered"] = failure["final_status"] == "certified"
    failures.sort(key=lambda row: row["ordinal"])

    baseline_summary = _load_object(baseline_dir / "summary.json")
    transfer_summary = _load_object(final_transfer_dir / "summary.json")
    development_summary = _load_object(development_dir / "summary.json")
    carrier_summary = _load_object(carrier_dir / "summary.json")
    carrier_audit = _load_object(carrier_dir / "carrier-resolution-audit.json")
    selected_documents = int(transfer_summary["documents"]) + int(development_summary["documents"])
    certified_documents = int(transfer_summary["statusCounts"]["certified"]) + int(
        development_summary["statusCounts"]["certified"]
    )
    selected_cost = Decimal(transfer_summary["usage"]["all"]["estimatedCostUsd"]) + Decimal(
        development_summary["usage"]["all"]["estimatedCostUsd"]
    )
    selected_mean = selected_cost / Decimal(selected_documents)
    accepted_mean = selected_cost / Decimal(certified_documents)
    eligible_documents = int(carrier_audit["eligibleForCarrierBoundExtraction"])
    corpus_documents = int(carrier_audit["documents"])
    if eligible_documents != int(carrier_summary["carrierBoundEligibleDocuments"]):
        raise ValueError("carrier-audit eligible counts disagree")

    host_rejected_cost = sum(
        (
            Decimal(row["estimatedCostUsd"])
            for row in stage_rows
            if row["disposition"].startswith("host_rejected")
        ),
        Decimal(0),
    )
    baseline_cost = Decimal(baseline_summary["usage"]["all"]["estimatedCostUsd"])
    if Decimal(baseline_usage["estimatedCostUsd"]) != baseline_cost:
        raise ValueError("streamed stage cost does not match baseline summary")
    baseline_config = _load_object(baseline_dir / "config.json")
    model_probe_rows, model_probe_summary, model_probe_commits = _model_probe_audit(
        verified_probes,
        comparison_metrics,
        baseline_config,
    )
    compiler_pricing = baseline_config["compiler_provider"]["pricing"]
    critic_pricing = baseline_config["critic_provider"]["pricing"]
    if compiler_pricing != critic_pricing:
        raise ValueError("baseline compiler and critic pricing differ")
    no_cache_cost = (
        Decimal(baseline_usage["inputTokens"])
        * Decimal(str(compiler_pricing["input_usd_per_million"]))
        + Decimal(baseline_usage["outputTokens"])
        * Decimal(str(compiler_pricing["output_usd_per_million"]))
    ) / Decimal(1_000_000)
    compact_current_bytes = sum(row["current_bytes"] for row in compact_rows)
    compact_proposed_bytes = sum(row["compact_bytes"] for row in compact_rows)
    compact_reduction = Decimal(compact_current_bytes - compact_proposed_bytes) / Decimal(
        compact_current_bytes
    )
    component_totals: dict[tuple[str, str], int] = defaultdict(int)
    role_pass_totals: dict[tuple[str, str], int] = {}
    for row in component_rows:
        key = (cast(str, row["role"]), cast(str, row["component"]))
        if row["component"] == "__total__":
            role_pass_totals[(row["role"], row["pass_kind"])] = int(row["total_bytes"])
        else:
            component_totals[key] += int(row["total_bytes"])
    critic_payload_bytes = sum(
        total for (role, _), total in role_pass_totals.items() if role == "critic"
    )
    critic_component_fractions = {
        component: str(Decimal(total) / Decimal(critic_payload_bytes))
        for (role, component), total in sorted(component_totals.items())
        if role == "critic"
    }
    compiler_repair_bytes = role_pass_totals[("compiler", "repair")]
    compiler_repair_fractions = {
        component: str(
            Decimal(
                next(
                    int(row["total_bytes"])
                    for row in component_rows
                    if row["role"] == "compiler"
                    and row["pass_kind"] == "repair"
                    and row["component"] == component
                )
            )
            / Decimal(compiler_repair_bytes)
        )
        for component in ("previousCandidateOutput", "anchorBindings", "riskCandidates")
    }
    critic_four_surface_fraction = sum(
        Decimal(critic_component_fractions[key])
        for key in (
            "bindingInventory",
            "maskedTemplate",
            "allowedTargetPaths",
            "allowedRemovalLogicalKeys",
        )
    )
    counterfactual_rows = []
    for scenario in config.pricing_scenarios:
        cost = _scenario_cost(baseline_usage, scenario)
        counterfactual_rows.append(
            {
                "scenario": scenario.name,
                "model": scenario.model,
                "source_url": scenario.source_url,
                "same_recorded_token_cost_usd": str(cost),
                "relative_to_recorded_luna": str(cost / baseline_cost),
                "savings_vs_recorded_luna_usd": str(baseline_cost - cost),
                "quality_equivalence_proven": False,
            }
        )

    summary = {
        "schemaVersion": 1,
        "baselineTransferRun": baseline_dir.name,
        "finalTransferRun": final_transfer_dir.name,
        "finalDevelopmentRun": development_dir.name,
        "baselineTransferDocuments": int(baseline_summary["documents"]),
        "baselineTransferCertified": int(baseline_summary["statusCounts"]["certified"]),
        "baselineTransferFailures": len(failures),
        "baselineTransferCostUsd": str(baseline_cost),
        "baselineHostRejectedStageCostUsd": str(host_rejected_cost),
        "baselineHostRejectedStageCostFraction": str(host_rejected_cost / baseline_cost),
        "baselineNoCacheCounterfactualCostUsd": str(no_cache_cost),
        "baselineCacheDeltaVsNoCacheUsd": str(baseline_cost - no_cache_cost),
        "finalSelectedDocuments": selected_documents,
        "finalCertifiedDocuments": certified_documents,
        "finalSelectedCostUsd": str(selected_cost),
        "finalMeanPerSelectedDocumentUsd": str(selected_mean),
        "finalMeanPerCertifiedDocumentUsd": str(accepted_mean),
        "eligibleCarrierBoundDocuments": eligible_documents,
        "fullCorpusDocuments": corpus_documents,
        "eligibleProjectionAtSelectedRateUsd": str(selected_mean * Decimal(eligible_documents)),
        "eligibleProjectionAtCertifiedRateUsd": str(accepted_mean * Decimal(eligible_documents)),
        "fullCorpusProjectionAtSelectedRateUsd": str(selected_mean * Decimal(corpus_documents)),
        "fullCorpusProjectionAtCertifiedRateUsd": str(accepted_mean * Decimal(corpus_documents)),
        "baselineFailuresRecovered": sum(bool(row["recovered"]) for row in failures),
        "baselineFailuresRemaining": sum(not bool(row["recovered"]) for row in failures),
        "remainingFailureEconomics": remaining_failures,
        "criticPayloads": len(compact_rows),
        "criticCurrentPayloadBytes": compact_current_bytes,
        "criticLosslessCompactPayloadBytes": compact_proposed_bytes,
        "criticLosslessCompactPayloadReductionFraction": str(compact_reduction),
        "criticPayloadComponentFractions": critic_component_fractions,
        "compilerRepairPayloadComponentFractions": compiler_repair_fractions,
        "contractProbe": contract_probe,
        "modelProbes": model_probe_summary,
    }
    terminal_failure_counts = Counter(row["baseline_terminal_category"] for row in failures)
    underlying_failure_counts = Counter(row["baseline_underlying_category"] for row in failures)
    summary["baselineTerminalFailureCounts"] = dict(sorted(terminal_failure_counts.items()))
    summary["baselineUnderlyingFailureCounts"] = dict(sorted(underlying_failure_counts.items()))
    stage_columns = tuple(stage_rows[0])
    compiler_columns = tuple(compiler_pass_rows[0])
    component_columns = tuple(component_rows[0])
    compact_columns = tuple(compact_rows[0])
    failure_columns = tuple(failures[0])
    counterfactual_columns = tuple(counterfactual_rows[0])
    model_probe_columns = tuple(model_probe_rows[0])
    remaining_failure_columns = tuple(remaining_failures[0])
    model_probe_report_lines: list[str] = []
    for row in model_probe_rows:
        if row["status"] != "certified":
            model_probe_report_lines.append(
                f"- `{row['probe']}`: rejected locally after **{row['requests']} requests** "
                f"and **${row['cost_usd']}**; `{row['last_error']}`."
            )
            continue
        model_probe_report_lines.append(
            f"- `{row['probe']}`: certified at **${row['cost_usd']}** "
            f"(**{Decimal(row['cost_fraction_of_baseline']):.2%}** of the paired Luna cost), "
            f"**{Decimal(row['elapsed_fraction_of_baseline']):.2f}x** paired wall time, "
            f"**{Decimal(row['output_token_fraction_of_baseline']):.2f}x** output tokens, and "
            f"an agent-assisted-binding delta of **{int(row['agent_assisted_binding_delta']):+d}**."
        )
    report = "\n".join(
        [
            "# Provider-neutral template-extraction efficiency audit",
            "",
            "## Baseline completion",
            "",
            "- Development plus transfer: "
            f"**{certified_documents}/{selected_documents} certified**.",
            f"- Cumulative 230-document lineage cost: **${selected_cost}**.",
            "- Original transfer failures recovered: "
            f"**{summary['baselineFailuresRecovered']}/14**.",
            f"- Remaining hard case: **{summary['baselineFailuresRemaining']}**.",
            "- Its accumulated lineage is "
            f"**{remaining_failures[0]['stages']} stages / "
            f"{remaining_failures[0]['requests']} requests / "
            f"${remaining_failures[0]['cost_usd']} / "
            f"{Decimal(str(remaining_failures[0]['elapsed_seconds'])) / Decimal(3600):.2f} "
            "hours**, including "
            f"**{remaining_failures[0]['compiler_host_rejections']} compiler** and "
            f"**{remaining_failures[0]['critic_host_rejections']} critic** host rejections.",
            "",
            "## Why the original 14 stopped",
            "",
            "Terminal stop categories:",
            "",
            *[
                f"- `{name}`: **{count}**."
                for name, count in sorted(
                    terminal_failure_counts.items(), key=lambda row: (-row[1], row[0])
                )
            ],
            "",
            "Underlying failure modes (resolving the cost guard to its last host error):",
            "",
            *[
                f"- `{name}`: **{count}**."
                for name, count in sorted(
                    underlying_failure_counts.items(), key=lambda row: (-row[1], row[0])
                )
            ],
            "",
            "Only the two `provider_quota_exhausted` cases stopped on the credit balance. The "
            "other twelve were host or semantic contract failures. The final continuation "
            "recovered both quota cases and eleven of the twelve non-quota cases; the remaining "
            "document is a host/semantic hard case.",
            "",
            "## Measured cost drivers",
            "",
            f"- Host-rejected stages cost **${host_rejected_cost}** "
            f"(**{host_rejected_cost / baseline_cost:.2%}** of the original transfer bill).",
            "- Structured-output retry responses cost "
            f"**${contract_probe['compilerRetryResponseProviderCostUsd']}** "
            f"across **{contract_probe['compilerRetryResponses']}** extra responses.",
            f"- The lossless short-reference critic payload is **{compact_reduction:.2%}** smaller "
            f"in UTF-8 JSON bytes over all {len(compact_rows)} critic stages.",
            "- In critic payloads, the binding inventory alone is "
            f"**{Decimal(critic_component_fractions['bindingInventory']):.2%}**; the inventory, "
            "masked template, target-path list, and removal-key list together are "
            f"**{critic_four_surface_fraction:.2%}**.",
            "- In compiler repair payloads, the complete previous candidate is "
            f"**{Decimal(compiler_repair_fractions['previousCandidateOutput']):.2%}** and the "
            "unchanged anchor inventory is "
            f"**{Decimal(compiler_repair_fractions['anchorBindings']):.2%}**.",
            f"- Prompt caching changed the recorded cost by **${baseline_cost - no_cache_cost}** "
            "versus pricing every input token at the ordinary Luna input rate; it was not a "
            "saving.",
            "",
            "## Contract probes",
            "",
            f"- **{contract_probe['acceptedLegacyBindingsConverted']}** accepted legacy binding "
            "proposals converted to the discriminated schema with zero rejection.",
            f"- All **{contract_probe['compilerRetryValidationErrors']}** retained "
            "conditional-schema "
            "errors are structurally excluded by the proposed render-mode variants.",
            f"- **{contract_probe['acceptedLegacyCriticOutputsConverted']}** accepted critic "
            "outputs converted to verdict-discriminated pass/revise variants with zero rejection.",
            "- The live BaseTen critic's invalid pass-with-patch response is structurally "
            "excluded by the verdict-discriminated schema.",
            f"- All **{contract_probe['criticPayloadsLosslesslyRoundTripped']}** compact critic "
            "payloads "
            "round-tripped byte-semantically to their original JSON objects.",
            "- Schema trade-off: the binding schema grows by "
            f"**{contract_probe['schemaByteDelta']} bytes** and the critic schema by "
            f"**{contract_probe['criticSchemaByteDelta']} bytes**; those bytes buy structural "
            "exclusion of invalid state combinations.",
            "",
            "## Bounded live model/provider probes",
            "",
            *model_probe_report_lines,
            "",
            f"Total paid probe cost was **${model_probe_summary['paidProbeCostUsd']}**. "
            "Certification proves host-contract validity for these individual outputs, not "
            "semantic equivalence between models. The paired measurements show that cheaper "
            "token pricing can increase reasoning/output volume, latency, downstream residual "
            "work, or critic revisions.",
            "",
            "## Model-price counterfactual",
            "",
            *[
                f"- `{row['scenario']}`: **${row['same_recorded_token_cost_usd']}** for the exact "
                "recorded token/cache workload. This is a price-only counterfactual; it does not "
                "establish equal tokenization, retries, or quality."
                for row in counterfactual_rows
            ],
            "",
            "## Projection boundary",
            "",
            f"- Carrier-bound eligible line ({eligible_documents:,} documents), selected-rate: "
            f"**${summary['eligibleProjectionAtSelectedRateUsd']}**.",
            f"- Entire pinned corpus ({corpus_documents:,} documents), selected-rate: "
            f"**${summary['fullCorpusProjectionAtSelectedRateUsd']}**.",
            "",
            "These are descriptive baseline projections from a 229/230 certified cohort, not a "
            "full-corpus launch authorization. Token reductions from the compact contract require "
            "a paired live quality/cost canary before they can be converted into dollar savings.",
            "",
        ]
    )
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "configSha256": sha256_file(config_path),
                "baselineCommitSha256": config.baseline_transfer_run.commit_sha256,
                "finalTransferCommitSha256": config.final_transfer_run.commit_sha256,
                "developmentCommitSha256": config.final_development_run.commit_sha256,
                "carrierAuditCommitSha256": config.carrier_audit_run.commit_sha256,
                "modelProbeCommitSha256": {
                    probe.name: probe.run.commit_sha256 for probe in config.model_probe_runs
                },
                "efficiencySourceSha256": sha256_file(Path(__file__)),
                "optimizationContractSourceSha256": sha256_file(
                    Path(__file__).with_name("optimization_contract.py")
                ),
            }
        )
    )
    staged = StagedArtifactRun(
        output_parent=Path(config.output_dir).resolve(),
        run_name=config.run_name,
        transaction_sha256=transaction,
    )
    if staged.completed:
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    staged.publish_json("config.json", config.model_dump(mode="json"))
    staged.publish_json("summary.json", summary)
    staged.publish_json("contract-probe.json", contract_probe)
    staged.publish_json("input-baseline-commit.json", baseline_commit)
    staged.publish_json("input-final-transfer-commit.json", final_transfer_commit)
    staged.publish_json("input-development-commit.json", development_commit)
    staged.publish_json("input-carrier-audit-commit.json", carrier_commit)
    staged.publish_json("input-model-probe-commits.json", model_probe_commits)
    staged.publish_bytes("REPORT.md", report.encode("utf-8"))
    staged.publish_bytes("stage-economics.csv", _csv_bytes(stage_rows, stage_columns))
    staged.publish_bytes(
        "compiler-pass-economics.csv",
        _csv_bytes(compiler_pass_rows, compiler_columns),
    )
    staged.publish_bytes("payload-components.csv", _csv_bytes(component_rows, component_columns))
    staged.publish_bytes("critic-compact-contract.csv", _csv_bytes(compact_rows, compact_columns))
    staged.publish_bytes("baseline-failures.csv", _csv_bytes(failures, failure_columns))
    staged.publish_bytes(
        "model-price-counterfactuals.csv",
        _csv_bytes(counterfactual_rows, counterfactual_columns),
    )
    staged.publish_bytes(
        "model-probes.csv",
        _csv_bytes(model_probe_rows, model_probe_columns),
    )
    staged.publish_bytes(
        "remaining-hard-cases.csv",
        _csv_bytes(remaining_failures, remaining_failure_columns),
    )
    expected = (
        "REPORT.md",
        "baseline-failures.csv",
        "compiler-pass-economics.csv",
        "config.json",
        "contract-probe.json",
        "critic-compact-contract.csv",
        "input-baseline-commit.json",
        "input-carrier-audit-commit.json",
        "input-development-commit.json",
        "input-final-transfer-commit.json",
        "input-model-probe-commits.json",
        "model-price-counterfactuals.csv",
        "model-probes.csv",
        "payload-components.csv",
        "remaining-hard-cases.csv",
        "stage-economics.csv",
        "summary.json",
    )
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 1,
            "baselineDocuments": int(baseline_summary["documents"]),
            "finalCertifiedDocuments": certified_documents,
            "modelProbeRuns": len(model_probe_rows),
            "providerRequests": 0,
        },
    )
    return staged.final_root
