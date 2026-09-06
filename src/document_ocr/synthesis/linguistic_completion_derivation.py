"""Derive an immutable linguistic-completion run from validated cargo corrections.

This is an artifact migration, not a new generative stage.  It accepts one committed
linguistic-completion run plus committed one-case cargo probes, revalidates each probe against
the source slot contract, replaces only the corresponding cargo-language unit and target leaves,
and republishes the complete run under a new immutable transaction.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.cargo_language_probe import (
    CargoLanguageCaseRecord,
    CargoLanguageGenerationOutput,
    CargoLanguageGenerationSeed,
    _auxiliary_fact_kinds,
    _cargo_auxiliary_role,
    normalize_cargo_language_output,
    validate_cargo_language,
)
from document_ocr.synthesis.linguistic_completion_pipeline import (
    DocumentLinguisticPlan,
    LinguisticAttemptRecord,
    LinguisticCompletionDocumentRecord,
    LinguisticUnitArtifact,
)
from document_ocr.synthesis.raw_text_template import printed_topology_mismatches
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_V5_TASK_ADAPTER

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_RESERVED = frozenset({"_COMMIT.json", "_TRANSACTION.json"})
_DERIVED_FILES = frozenset(
    {
        "REPORT.md",
        "config.yaml",
        "derivation.json",
        "generation/results.jsonl",
        "generation/summary.json",
        "generation/targets.jsonl",
        "planning/document-plans.jsonl",
    }
)


def _load_committed_run(root: Path) -> tuple[Path, str, str]:
    resolved = root.resolve(strict=True)
    transaction = cast(
        dict[str, JsonValue], json.loads(read_regular_file_bytes(resolved / "_TRANSACTION.json"))
    )
    transaction_sha = cast(str, transaction["transactionSha256"])
    StagedArtifactRun(
        output_parent=resolved.parent,
        run_name=resolved.name,
        transaction_sha256=transaction_sha,
    ).validate_committed_run()
    return resolved, sha256_file(resolved / "_COMMIT.json"), transaction_sha


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"JSONL row {line_number} is blank or unterminated: {path}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return tuple(rows)


def _upgrade_plan_role_contract(row: Mapping[str, Any]) -> DocumentLinguisticPlan:
    """Rebuild the deterministic cargo-slot role fields added after the base run.

    The base artifact remains immutable and continues to describe its original schema.  The
    derived run, however, must publish plans that validate under the same current contract as its
    replacement cargo units.  Both fields are pure functions of the retained source-style string;
    no label value or model output is inferred during this migration.
    """

    upgraded = copy.deepcopy(dict(row))
    cargo_seed = upgraded.get("cargoSeed")
    groups = cargo_seed.get("cargoGroups") if isinstance(cargo_seed, dict) else None
    if not isinstance(groups, list):
        raise ValueError("base linguistic plan has invalid cargo groups")
    for group in groups:
        contract = group.get("fieldContract") if isinstance(group, dict) else None
        slots = contract.get("additionalInformationSlots") if isinstance(contract, dict) else None
        if not isinstance(slots, list):
            raise ValueError("base linguistic plan has invalid cargo auxiliary slots")
        for slot in slots:
            source = slot.get("sourceStyleReference") if isinstance(slot, dict) else None
            if not isinstance(source, str) or not source.strip():
                raise ValueError("base linguistic plan has invalid cargo auxiliary source")
            expected_role = _cargo_auxiliary_role(source)
            expected_kinds = tuple(sorted(_auxiliary_fact_kinds(source)))
            prior_role = slot.get("semanticRole")
            prior_kinds = slot.get("sourceFactKinds")
            if prior_role is not None and prior_role != expected_role:
                raise ValueError("base linguistic plan cargo role conflicts with current grammar")
            if prior_kinds is not None and tuple(prior_kinds) != expected_kinds:
                raise ValueError("base linguistic plan cargo facts conflict with current grammar")
            slot["semanticRole"] = expected_role
            slot["sourceFactKinds"] = list(expected_kinds)
    return DocumentLinguisticPlan.model_validate_json(canonical_json_bytes(upgraded), strict=True)


def _slot_values(group: Mapping[str, Any], field: str) -> tuple[str, ...]:
    raw = group.get(field)
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError(f"base cargo group has invalid {field}")
    return tuple(raw)


def _base_cargo_groups(
    payload: Mapping[str, JsonValue],
) -> tuple[dict[str, Any], ...]:
    raw_groups = payload.get("cargoGroups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("base cargo unit output has no cargo groups")
    groups: list[dict[str, Any]] = []
    expected_keys = {
        "groupId",
        "description",
        "additionalInformation",
        "marksAndNumbers",
        "handlingInstructions",
    }
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict) or set(raw_group) != expected_keys:
            raise ValueError("base cargo unit output has an invalid group object")
        group = cast(dict[str, Any], raw_group)
        if not isinstance(group["groupId"], str) or not group["groupId"]:
            raise ValueError("base cargo unit output has an invalid group identifier")
        if group["description"] is not None and not isinstance(group["description"], str):
            raise ValueError("base cargo unit output has an invalid description")
        for field in ("additionalInformation", "marksAndNumbers", "handlingInstructions"):
            values = group[field]
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"base cargo unit output has invalid {field}")
        groups.append(group)
    return tuple(groups)


def apply_cargo_language_correction(
    *,
    base_document_id: str,
    base_target: Mapping[str, JsonValue],
    base_output: Mapping[str, JsonValue],
    seed: CargoLanguageGenerationSeed,
    output: CargoLanguageGenerationOutput,
) -> dict[str, JsonValue]:
    """Apply one validated cargo-language output without changing printed topology."""

    if seed.sourceDocumentId != base_document_id:
        raise ValueError("cargo correction document differs from its base target")
    output = normalize_cargo_language_output(seed=seed, output=output)
    validation = validate_cargo_language(seed=seed, output=output)
    if not validation.passed:
        failed = sorted(name for name, passed in validation.checks.items() if not passed)
        raise ValueError(f"cargo correction fails current validation: {failed}")
    target = copy.deepcopy(dict(base_target))
    raw_patch = target.get("documentPatch")
    if not isinstance(raw_patch, dict):
        raise ValueError("base target has no document patch")
    patch = cast(dict[str, Any], raw_patch)
    raw_groups = patch.get("cargoGroups")
    if not isinstance(raw_groups, list) or any(not isinstance(row, dict) for row in raw_groups):
        raise ValueError("base target has invalid cargo groups")
    group_rows = cast(list[dict[str, Any]], raw_groups)
    groups = {cast(str, row["groupId"]): row for row in group_rows}
    base_groups = _base_cargo_groups(base_output)
    if tuple(groups) != tuple(row.groupId for row in seed.cargoGroups) or tuple(groups) != tuple(
        cast(str, row["groupId"]) for row in base_groups
    ):
        raise ValueError("cargo correction group topology differs from its base target")
    for previous, expected, generated in zip(
        base_groups,
        seed.cargoGroups,
        output.cargoGroups,
        strict=True,
    ):
        group = groups[expected.groupId]
        if group.get("description") != previous["description"]:
            raise ValueError("base cargo description differs from its selected unit output")
        for field, values in (
            ("additionalInformation", tuple(previous["additionalInformation"])),
            ("marksAndNumbers", tuple(previous["marksAndNumbers"])),
            ("handlingInstructions", tuple(previous["handlingInstructions"])),
        ):
            if _slot_values(group, field) != values:
                raise ValueError(f"base cargo {field} differs from its selected unit output")
        if generated.description is None:
            group.pop("description", None)
        else:
            group["description"] = generated.description
        for field, values in (
            ("additionalInformation", generated.additionalInformation),
            ("marksAndNumbers", generated.marksAndNumbers),
            ("handlingInstructions", generated.handlingInstructions),
        ):
            if values:
                group[field] = list(values)
            else:
                group.pop(field, None)
    canonical = cast(
        dict[str, JsonValue],
        BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
            document_id=base_document_id,
            target=target,
        ),
    )
    mismatches = printed_topology_mismatches(base_target, canonical)
    if mismatches:
        raise ValueError(
            "cargo correction changed printed topology: "
            + ", ".join(row.path for row in mismatches[:8])
        )
    return canonical


def _replacement_unit(
    record: CargoLanguageCaseRecord,
    *,
    output: CargoLanguageGenerationOutput,
) -> LinguisticUnitArtifact:
    """Publish the output revalidated by this derivation, not stale probe validation.

    Probe artifacts are immutable evidence of the validator version that produced them.  A later
    derivation may deliberately replay a provider-valid output through a corrected validator, but
    it must retain the original disposition and may select the output only after the current
    validator passes.  This keeps the old finding auditable without copying it into the derived
    unit as if it were still true.
    """

    if record.status == "call_failed" or record.output is None:
        raise ValueError("cargo correction has no provider output to revalidate")
    current_validation = validate_cargo_language(seed=record.seed, output=output)
    if not current_validation.passed:
        failed = sorted(name for name, passed in current_validation.checks.items() if not passed)
        raise ValueError(f"cargo correction fails current validation: {failed}")
    attempt = LinguisticAttemptRecord(
        attempt=1,
        schemaMode="topology_constrained",
        outputSchemaSha256=record.outputSchemaSha256,
        userPromptSha256=record.userPromptSha256,
        startedAt=record.startedAt,
        completedAt=record.completedAt,
        durationMs=record.durationMs,
        status="success",
        output=cast(dict[str, JsonValue], output.model_dump(mode="json")),
        checks=current_validation.checks,
        usage=record.usage,
        errorType=None,
        errorMessage=None,
    )
    output_sha = sha256_bytes(canonical_json_bytes(attempt.output))
    return LinguisticUnitArtifact(
        schemaVersion=1,
        stage="cargo_language",
        unitId="cargo",
        attempts=(attempt,),
        transcripts=(
            {
                "derivation": "current_validator_replay_of_standalone_cargo_language_probe_v2",
                "sourceRecordStatus": record.status,
                "sourceValidationChecks": (
                    record.validation.checks if record.validation is not None else None
                ),
                "providerResponseIds": list(record.usage.providerResponseIds),
            },
        ),
        selectedAttempt=1,
        selectedOutputSha256=output_sha,
        status="success",
    )


def _artifact_paths(root: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directory_names):
            raise ValueError("base artifact contains a symbolic-link directory")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("base artifact contains a non-regular file")
            relative = path.relative_to(root).as_posix()
            if relative not in _RESERVED:
                paths.append(relative)
    return tuple(sorted(paths))


def derive_linguistic_completion(
    *,
    base_run: Path,
    correction_runs: Sequence[Path],
    output_parent: Path,
    run_id: str,
) -> dict[str, JsonValue]:
    if not correction_runs:
        raise ValueError("at least one cargo correction run is required")
    base_root, base_commit_sha, base_transaction_sha = _load_committed_run(base_run)
    corrections: dict[str, tuple[CargoLanguageCaseRecord, Path, str, str]] = {}
    correction_prompt_sha: str | None = None
    correction_schema_sha: str | None = None
    correction_prompt_bytes: bytes | None = None
    correction_schema_bytes: bytes | None = None
    for correction_path in correction_runs:
        root, commit_sha, transaction_sha = _load_committed_run(correction_path)
        prompt_bytes = read_regular_file_bytes(root / "prompt.md")
        schema_bytes = read_regular_file_bytes(root / "schema/output.schema.json")
        prompt_sha = sha256_bytes(prompt_bytes)
        schema_sha = sha256_bytes(schema_bytes)
        if correction_prompt_sha is not None and correction_prompt_sha != prompt_sha:
            raise ValueError("cargo correction runs use different prompts")
        if correction_schema_sha is not None and correction_schema_sha != schema_sha:
            raise ValueError("cargo correction runs use different output schemas")
        correction_prompt_sha = prompt_sha
        correction_schema_sha = schema_sha
        correction_prompt_bytes = prompt_bytes
        correction_schema_bytes = schema_bytes
        rows = _load_jsonl(root / "generation/results.jsonl")
        if not rows:
            raise ValueError("cargo correction run contains no results")
        for row in rows:
            record = CargoLanguageCaseRecord.model_validate_json(
                canonical_json_bytes(row), strict=True
            )
            document_id = record.seed.sourceDocumentId
            if document_id in corrections:
                raise ValueError(f"duplicate cargo correction for {document_id}")
            corrections[document_id] = (record, root, commit_sha, transaction_sha)

    base_results = list(_load_jsonl(base_root / "generation/results.jsonl"))
    base_targets = list(_load_jsonl(base_root / "generation/targets.jsonl"))
    base_plans = tuple(
        _upgrade_plan_role_contract(row)
        for row in _load_jsonl(base_root / "planning/document-plans.jsonl")
    )
    result_by_id = {cast(str, row["baseDocumentId"]): row for row in base_results}
    target_by_id = {cast(str, row["baseDocumentId"]): row for row in base_targets}
    plan_by_id = {row.baseDocumentId: row for row in base_plans}
    if tuple(result_by_id) != tuple(target_by_id) or tuple(result_by_id) != tuple(plan_by_id):
        raise ValueError("base linguistic result/target order differs")
    derived_units: dict[str, bytes] = {}
    receipts: list[dict[str, JsonValue]] = []
    added_usage: dict[str, int] = {
        "requests": 0,
        "inputTokens": 0,
        "cacheReadTokens": 0,
        "outputTokens": 0,
        "reasoningTokens": 0,
        "visibleOutputTokens": 0,
    }
    added_cost = Decimal(0)
    for document_id, (record, root, commit_sha, transaction_sha) in corrections.items():
        try:
            result = result_by_id[document_id]
            target_row = target_by_id[document_id]
        except KeyError as error:
            raise ValueError(
                f"cargo correction document is absent from base: {document_id}"
            ) from error
        if result.get("scenarioId") != record.seed.scenarioId:
            raise ValueError("cargo correction scenario differs from base")
        if record.seed.model_dump(mode="json", exclude={"caseId"}) != plan_by_id[
            document_id
        ].cargoSeed.model_dump(mode="json", exclude={"caseId"}):
            raise ValueError("cargo correction seed differs from the base document plan")
        if result.get("target") != target_row.get("target"):
            raise ValueError("base linguistic result/target payload differs")
        base_target = target_row.get("target")
        base_target_sha = target_row.get("targetSha256")
        if not isinstance(base_target, dict):
            raise ValueError("base target row has no target object")
        if not isinstance(base_target_sha, str) or base_target_sha != sha256_bytes(
            canonical_json_bytes(base_target)
        ):
            raise ValueError("base target row fails its SHA-256")
        cargo_path = result.get("cargoUnitPath")
        if not isinstance(cargo_path, str):
            raise ValueError("base result has no cargo unit path")
        base_unit = LinguisticUnitArtifact.model_validate_json(
            read_regular_file_bytes(base_root / cargo_path), strict=True
        )
        if base_unit.selectedAttempt is None:
            raise ValueError("base cargo unit has no selected output")
        selected_attempt = next(
            row for row in base_unit.attempts if row.attempt == base_unit.selectedAttempt
        )
        if selected_attempt.output is None:
            raise ValueError("base cargo unit selected output is absent")
        if base_unit.selectedOutputSha256 != sha256_bytes(
            canonical_json_bytes(selected_attempt.output)
        ):
            raise ValueError("base cargo unit selected output fails its SHA-256")
        if record.output is None:
            raise ValueError(f"cargo correction has no output: {document_id}")
        normalized_output = normalize_cargo_language_output(
            seed=record.seed,
            output=record.output,
        )
        corrected = apply_cargo_language_correction(
            base_document_id=document_id,
            base_target=cast(Mapping[str, JsonValue], base_target),
            base_output=selected_attempt.output,
            seed=record.seed,
            output=normalized_output,
        )
        target_sha = sha256_bytes(canonical_json_bytes(corrected))
        result["target"] = corrected
        result["targetSha256"] = target_sha
        target_row["target"] = corrected
        target_row["targetSha256"] = target_sha
        derived_units[cargo_path] = json_artifact_bytes(
            _replacement_unit(record, output=normalized_output).model_dump(mode="json")
        )
        document_index = tuple(result_by_id).index(document_id)
        document_path = f"generation/documents/{document_index + 1:05d}.json"
        derived_units[document_path] = json_artifact_bytes(
            LinguisticCompletionDocumentRecord.model_validate_json(
                canonical_json_bytes(result), strict=True
            ).model_dump(mode="json")
        )
        usage = record.usage
        for key in (
            "requests",
            "inputTokens",
            "cacheReadTokens",
            "outputTokens",
            "reasoningTokens",
            "visibleOutputTokens",
        ):
            added_usage[key] += cast(int, getattr(usage, key))
        added_cost += usage.estimatedCostUsd
        receipts.append(
            {
                "documentId": document_id,
                "baseTargetSha256": base_target_sha,
                "correctedTargetSha256": target_sha,
                "correctionRun": root.name,
                "correctionCommitSha256": commit_sha,
                "correctionTransactionSha256": transaction_sha,
                "correctionResultSha256": sha256_file(root / "generation/results.jsonl"),
            }
        )

    results_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in base_results)
    targets_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in base_targets)
    plans_bytes = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in base_plans
    )
    transaction = {
        "schemaVersion": 1,
        "runId": run_id,
        "baseRun": base_root.name,
        "baseCommitSha256": base_commit_sha,
        "baseTransactionSha256": base_transaction_sha,
        "corrections": receipts,
        "cargoPromptSha256": correction_prompt_sha,
        "cargoOutputSchemaSha256": correction_schema_sha,
        "resultsSha256": sha256_bytes(results_bytes),
        "targetsSha256": sha256_bytes(targets_bytes),
        "plansSha256": sha256_bytes(plans_bytes),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
    }
    transaction_sha = sha256_bytes(canonical_json_bytes(transaction))
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=run_id,
        transaction_sha256=transaction_sha,
    )
    if staged.completed:
        receipt = staged.validate_committed_run()
        return {
            "runId": run_id,
            "corrections": len(corrections),
            "transactionSha256": transaction_sha,
            "commitSha256": sha256_file(staged.final_root / "_COMMIT.json"),
            "artifacts": len(receipt.artifacts),
        }
    staged.recover_interrupted_temporary_files()
    overrides = (
        _DERIVED_FILES
        | frozenset(derived_units)
        | frozenset({"prompts/cargo.md", "schema/cargo-output.schema.json"})
    )
    for relative in _artifact_paths(base_root):
        if relative not in overrides:
            staged.publish_bytes(relative, read_regular_file_bytes(base_root / relative))
    for relative, payload in derived_units.items():
        staged.publish_bytes(relative, payload)
    assert correction_prompt_bytes is not None
    assert correction_schema_bytes is not None
    staged.publish_bytes("prompts/cargo.md", correction_prompt_bytes)
    staged.publish_bytes("schema/cargo-output.schema.json", correction_schema_bytes)
    staged.publish_bytes("generation/results.jsonl", results_bytes)
    staged.publish_bytes("generation/targets.jsonl", targets_bytes)
    staged.publish_bytes("planning/document-plans.jsonl", plans_bytes)
    base_summary = cast(
        dict[str, JsonValue],
        json.loads(read_regular_file_bytes(base_root / "generation/summary.json")),
    )
    for key in (
        "requests",
        "inputTokens",
        "cacheReadTokens",
        "outputTokens",
        "reasoningTokens",
        "visibleOutputTokens",
    ):
        value = base_summary.get(key)
        if isinstance(value, int):
            base_summary[key] = value + added_usage[key]
    old_cost = Decimal(cast(str, base_summary.get("estimatedCostUsd", "0")))
    base_summary["estimatedCostUsd"] = str(old_cost + added_cost)
    base_summary["runId"] = run_id
    base_summary["derivedFromRun"] = base_root.name
    base_summary["cargoCorrectionDocuments"] = len(corrections)
    base_summary["transactionSha256"] = transaction_sha
    staged.publish_bytes("generation/summary.json", json_artifact_bytes(base_summary))
    staged.publish_bytes("derivation.json", json_artifact_bytes(transaction))
    staged.publish_bytes("config.yaml", canonical_json_bytes(transaction) + b"\n")
    report = (
        "# Derived linguistic-completion artifact\n\n"
        f"Base committed run: `{base_root.name}`.\n\n"
        f"Validated cargo corrections: {len(corrections)}.\n\n"
        "Only cargo-language leaves and their corresponding unit/document/result artifacts were "
        "replaced. Structured target fields and printed topology were revalidated unchanged.\n"
    )
    staged.publish_bytes("REPORT.md", report.encode("utf-8"))
    expected = _artifact_paths(staged.stage_root)
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "documents": len(base_results),
            "cargo_corrections": len(corrections),
            "schema_version": 1,
        },
    )
    return {
        "runId": run_id,
        "corrections": len(corrections),
        "transactionSha256": transaction_sha,
        "commitSha256": sha256_file(staged.final_root / "_COMMIT.json"),
        "resultsSha256": sha256_file(staged.final_root / "generation/results.jsonl"),
        "targetsSha256": sha256_file(staged.final_root / "generation/targets.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--cargo-correction-run", type=Path, action="append", required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    result = derive_linguistic_completion(
        base_run=arguments.base_run,
        correction_runs=arguments.cargo_correction_run,
        output_parent=arguments.output_parent,
        run_id=arguments.run_id,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
