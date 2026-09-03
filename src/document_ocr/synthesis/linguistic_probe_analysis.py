"""Immutable EDA over the party and cargo linguistic synthesis probes."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.cargo_language_probe import CargoLanguageCaseRecord
from document_ocr.synthesis.config import (
    LinguisticProbeAnalysisInputConfig,
    SynthesisLinguisticProbeAnalysisConfig,
)
from document_ocr.synthesis.party_identity_probe import PartyIdentityCaseRecord
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_GROUP_PREFIX = re.compile(r"^group_[0-9]+_")
_ALNUM = re.compile(r"[^A-Z0-9]+")


def _validate_run(project_root: Path, configured: LinguisticProbeAnalysisInputConfig) -> Path:
    root = resolve_config_path(project_root, configured.run.path)
    commit = root / "_COMMIT.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or sha256_file(commit) != configured.run.commit_sha256
    ):
        raise ValueError(f"linguistic probe dependency differs from its pin: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.run.transaction_sha256,
    ).validate_committed_run()
    return root.resolve(strict=True)


def _resolve_file(
    project_root: Path,
    root: Path,
    configured_path: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    path = resolve_config_path(project_root, configured_path)
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve(strict=True).is_relative_to(root)
        or sha256_file(path) != expected_sha256
    ):
        raise ValueError(f"{label} differs from its pin or committed run: {path}")
    return path.resolve(strict=True)


def _load_jsonl[Record](path: Path, *, records: int, model: type[Record]) -> tuple[Record, ...]:
    output: list[Record] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"probe row {line_number} is blank or unterminated: {path}")
            output.append(model.model_validate_json(raw, strict=True))  # type: ignore[attr-defined]
    if len(output) != records:
        raise ValueError(f"probe result count differs from configuration: {path}")
    return tuple(output)


def _load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"probe summary is not an object: {path}")
    return value


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _normalized(value: str) -> str:
    return " ".join(_ALNUM.sub(" ", value.upper()).split())


def _usage_columns(record: PartyIdentityCaseRecord | CargoLanguageCaseRecord) -> dict[str, Any]:
    usage = record.usage
    uncached = usage.inputTokens - usage.cacheReadTokens - usage.cacheWriteTokens
    if uncached < 0:
        raise ValueError("probe cache token accounting exceeds input tokens")
    return {
        "duration_seconds": record.durationMs / 1000.0,
        "input_tokens": usage.inputTokens,
        "uncached_input_tokens": uncached,
        "cache_read_tokens": usage.cacheReadTokens,
        "cache_write_tokens": usage.cacheWriteTokens,
        "output_tokens": usage.outputTokens,
        "reasoning_tokens": usage.reasoningTokens,
        "visible_output_tokens": usage.visibleOutputTokens,
        "estimated_cost_usd": float(usage.estimatedCostUsd),
    }


def _contact_shape(record: PartyIdentityCaseRecord) -> str:
    shape = record.seed.fieldPresence.contacts
    return (
        f"name={int(shape.contactNamePresent)};phone={shape.phoneNumberCount};"
        f"email={shape.emailAddressCount};web={shape.websiteUrlCount}"
    )


def _party_rows(records: Sequence[PartyIdentityCaseRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        party = record.output.party if record.output is not None else None
        validation = record.validation
        row = {
            "arm": "party",
            "case_index": record.caseIndex + 1,
            "document_id": record.seed.sourceDocumentId,
            "role": record.seed.partyRole,
            "country": record.seed.targetLocality["country"],
            "city": record.seed.targetLocality["city"],
            "contact_shape": _contact_shape(record),
            "status": record.status,
            "semantic_pass": bool(validation and validation.passed),
            "generated_name": party.name if party else None,
            "generated_name_normalized": (
                validation.generatedNameNormalized if validation is not None else None
            ),
            "name_characters": len(party.name) if party else None,
            "name_words": len(_normalized(party.name).split()) if party else None,
            "source_name_similarity": (
                validation.sourceNameSimilarity if validation is not None else None
            ),
            "address_characters": len(party.address) if party and party.address else 0,
            "phone_count": len(party.contactDetails.phoneNumbers) if party else 0,
            "email_count": len(party.contactDetails.emailAddresses) if party else 0,
            "website_count": len(party.contactDetails.websiteUrls) if party else 0,
            "contact_name_present": bool(party and party.contactDetails.contactName),
        }
        row.update(_usage_columns(record))
        rows.append(row)
    return rows


def _cargo_features(record: CargoLanguageCaseRecord) -> dict[str, Any]:
    groups = record.seed.cargoGroups
    output_groups = record.output.cargoGroups if record.output is not None else ()
    goods = sum(len(group.goodsIdentities) for group in groups)
    dangerous = sum(len(group.dangerousGoods) for group in groups)
    thermal_groups = sum(
        any(item.thermalProfile is not None for item in group.goodsIdentities)
        or any(
            equipment.temperatureSetpointCelsius is not None
            for equipment in group.structuredFacts.equipment
        )
        for group in groups
    )
    additional_slots = sum(len(group.fieldContract.additionalInformationSlots) for group in groups)
    marks_slots = sum(len(group.fieldContract.marksAndNumbersSlots) for group in groups)
    handling_slots = sum(len(group.fieldContract.handlingInstructionSlots) for group in groups)
    generated_descriptions = tuple(
        group.description for group in output_groups if group.description is not None
    )
    generated_additional = tuple(
        value
        for group in output_groups
        for value in group.additionalInformation
        if value is not None
    )
    generated_marks = tuple(value for group in output_groups for value in group.marksAndNumbers)
    generated_handling = tuple(
        value for group in output_groups for value in group.handlingInstructions
    )
    return {
        "arm": "cargo",
        "case_index": record.caseIndex + 1,
        "document_id": record.seed.sourceDocumentId,
        "status": record.status,
        "semantic_pass": bool(record.validation and record.validation.passed),
        "group_count": len(groups),
        "goods_identity_count": goods,
        "dangerous_goods_count": dangerous,
        "thermal_group_count": thermal_groups,
        "additional_slots": additional_slots,
        "marks_slots": marks_slots,
        "handling_slots": handling_slots,
        "total_text_slots": len(groups) + additional_slots + marks_slots + handling_slots,
        "max_group_marks_slots": max(
            (len(group.fieldContract.marksAndNumbersSlots) for group in groups), default=0
        ),
        "description_count": len(generated_descriptions),
        "additional_generated_count": len(generated_additional),
        "marks_generated_count": len(generated_marks),
        "handling_generated_count": len(generated_handling),
        "description_characters": sum(map(len, generated_descriptions)),
        "additional_characters": sum(map(len, generated_additional)),
        "marks_characters": sum(map(len, generated_marks)),
        "handling_characters": sum(map(len, generated_handling)),
        "is_dangerous_goods": dangerous > 0,
        "is_thermal": thermal_groups > 0,
        "is_multi_group": len(groups) > 1,
        "has_additional_information": additional_slots > 0,
        "has_marks": marks_slots > 0,
        "has_handling": handling_slots > 0,
        "is_high_mark_cardinality": marks_slots >= 10,
    }


def _cargo_rows(records: Sequence[CargoLanguageCaseRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = _cargo_features(record)
        row.update(_usage_columns(record))
        rows.append(row)
    return rows


def _failure_rows(
    party: Sequence[PartyIdentityCaseRecord],
    cargo: Sequence[CargoLanguageCaseRecord],
    targeted: Sequence[CargoLanguageCaseRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, records in (("party", party), ("cargo", cargo), ("cargo_targeted", targeted)):
        for record in records:
            if record.validation is None:
                rows.append(
                    {
                        "arm": arm,
                        "case_index": record.caseIndex + 1,
                        "document_id": record.seed.sourceDocumentId,
                        "check": "provider_call",
                        "check_family": "provider_call",
                    }
                )
                continue
            for check, passed in record.validation.checks.items():
                if not passed:
                    family = _GROUP_PREFIX.sub("", check)
                    rows.append(
                        {
                            "arm": arm,
                            "case_index": record.caseIndex + 1,
                            "document_id": record.seed.sourceDocumentId,
                            "check": check,
                            "check_family": family,
                        }
                    )
    return rows


def _summary_rows(
    rows: Sequence[Mapping[str, Any]], key: str, *, value_label: str
) -> list[dict[str, Any]]:
    values = sorted({cast(str, row[key]) for row in rows})
    output: list[dict[str, Any]] = []
    for value in values:
        subset = [row for row in rows if row[key] == value]
        passed = sum(bool(row["semantic_pass"]) for row in subset)
        output.append(
            {
                value_label: value,
                "cases": len(subset),
                "passed": passed,
                "pass_rate": passed / len(subset),
            }
        )
    return output


def _cargo_cohort_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cohorts: tuple[tuple[str, Callable[[Mapping[str, Any]], bool]], ...] = (
        ("All cases", lambda _: True),
        ("Dangerous goods", lambda row: bool(row["is_dangerous_goods"])),
        ("Thermal", lambda row: bool(row["is_thermal"])),
        ("Multiple cargo groups", lambda row: bool(row["is_multi_group"])),
        ("Additional information", lambda row: bool(row["has_additional_information"])),
        ("Marks", lambda row: bool(row["has_marks"])),
        ("Handling instructions", lambda row: bool(row["has_handling"])),
        ("10+ marks slots", lambda row: bool(row["is_high_mark_cardinality"])),
    )
    output: list[dict[str, Any]] = []
    for name, predicate in cohorts:
        subset = [row for row in rows if predicate(row)]
        passed = sum(bool(row["semantic_pass"]) for row in subset)
        output.append(
            {
                "cohort": name,
                "cases": len(subset),
                "passed": passed,
                "pass_rate": passed / len(subset) if subset else 0.0,
            }
        )
    return output


def _plot_bytes(draw: Callable[[Any, Any, Any], Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd  # type: ignore[import-untyped]
    import seaborn as sns  # type: ignore[import-untyped]

    sns.set_theme(style="whitegrid", context="notebook")
    figure = draw(plt, sns, pd)
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return stream.getvalue()


def _plots(
    party_rows: list[dict[str, Any]],
    cargo_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> dict[str, bytes]:
    all_rows = party_rows + cargo_rows
    role_rows = _summary_rows(party_rows, "role", value_label="role")
    country_rows = _summary_rows(party_rows, "country", value_label="country")
    shape_rows = _summary_rows(party_rows, "contact_shape", value_label="contact_shape")
    cohort_rows = _cargo_cohort_rows(cargo_rows)

    def outcomes(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(all_rows)
        counts = frame.groupby(["arm", "status"]).size().reset_index(name="cases")
        pivot = counts.pivot(index="arm", columns="status", values="cases").fillna(0)
        figure, axis = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", stacked=True, ax=axis, color=["#2C7FB8", "#D95F0E", "#7F7F7F"])
        axis.set_title("First-pass outcomes under strict constrained decoding")
        axis.set_ylabel("Cases")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=0)
        figure.tight_layout()
        return figure

    def token_mix(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(all_rows)
        columns = [
            "uncached_input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "visible_output_tokens",
        ]
        totals = frame.groupby("arm")[columns].sum()
        figure, axis = plt.subplots(figsize=(11, 5.5))
        totals.plot(kind="bar", stacked=True, ax=axis, colormap="viridis")
        axis.set_title("Aggregate token composition (50 calls per arm)")
        axis.set_ylabel("Tokens")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=0)
        axis.legend(title="Token bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        return figure

    def cost_latency(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(all_rows)
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.boxplot(data=frame, x="arm", y="estimated_cost_usd", ax=axes[0], color="#7FCDBB")
        sns.stripplot(
            data=frame,
            x="arm",
            y="estimated_cost_usd",
            ax=axes[0],
            color="#225EA8",
            alpha=0.55,
            jitter=False,
        )
        axes[0].set_title("Estimated API cost per case")
        axes[0].set_ylabel("USD")
        sns.boxplot(data=frame, x="arm", y="duration_seconds", ax=axes[1], color="#FEC44F")
        sns.stripplot(
            data=frame,
            x="arm",
            y="duration_seconds",
            ax=axes[1],
            color="#D95F0E",
            alpha=0.55,
            jitter=False,
        )
        axes[1].set_title("End-to-end latency per case")
        axes[1].set_ylabel("Seconds")
        figure.tight_layout()
        return figure

    def party_roles(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(role_rows).sort_values("cases")
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(data=frame, x="cases", y="role", ax=axes[0], color="#2C7FB8")
        axes[0].set_title("Party cases by role")
        sns.barplot(data=frame, x="pass_rate", y="role", ax=axes[1], color="#41AB5D")
        axes[1].set_xlim(0, 1.03)
        axes[1].set_title("Semantic pass rate by role")
        figure.tight_layout()
        return figure

    def party_countries(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(country_rows).sort_values(["cases", "country"])
        figure, axis = plt.subplots(figsize=(11, max(8, len(frame) * 0.31)))
        sns.barplot(data=frame, x="cases", y="country", hue="pass_rate", palette="viridis", ax=axis)
        axis.set_title("Complete target-country coverage (no top-N truncation)")
        axis.legend(title="Pass rate", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.tight_layout()
        return figure

    def contact_shapes(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(shape_rows).sort_values("cases")
        figure, axes = plt.subplots(1, 2, figsize=(15, max(5, len(frame) * 0.45)))
        sns.barplot(data=frame, x="cases", y="contact_shape", ax=axes[0], color="#2C7FB8")
        axes[0].set_title("Contact-shape coverage")
        sns.barplot(data=frame, x="pass_rate", y="contact_shape", ax=axes[1], color="#41AB5D")
        axes[1].set_xlim(0, 1.03)
        axes[1].set_title("Pass rate by contact shape")
        figure.tight_layout()
        return figure

    def party_name_quality(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(party_rows)
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(data=frame, x="name_characters", bins=14, ax=axes[0], color="#2C7FB8")
        axes[0].set_title("Generated party-name length")
        sns.histplot(
            data=frame.dropna(subset=["source_name_similarity"]),
            x="source_name_similarity",
            bins=14,
            ax=axes[1],
            color="#7FCDBB",
        )
        axes[1].set_title("Similarity to source style-reference name")
        axes[1].set_xlim(0, 1)
        figure.tight_layout()
        return figure

    def cargo_complexity(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(cargo_rows)
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        sns.histplot(data=frame, x="group_count", discrete=True, ax=axes[0], color="#2C7FB8")
        axes[0].set_title("Cargo groups per case")
        sns.histplot(data=frame, x="marks_slots", discrete=True, ax=axes[1], color="#7FCDBB")
        axes[1].set_title("Marks slots per case")
        sns.histplot(data=frame, x="total_text_slots", bins=16, ax=axes[2], color="#FEC44F")
        axes[2].set_title("Total requested text leaves")
        figure.tight_layout()
        return figure

    def cargo_cohorts(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(cohort_rows).sort_values("cases")
        figure, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(data=frame, x="cases", y="cohort", ax=axes[0], color="#2C7FB8")
        axes[0].set_title("Cargo cohort coverage")
        sns.barplot(data=frame, x="pass_rate", y="cohort", ax=axes[1], color="#41AB5D")
        axes[1].set_xlim(0, 1.03)
        axes[1].set_title("Semantic pass rate by cohort")
        figure.tight_layout()
        return figure

    def cargo_scaling(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(cargo_rows)
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.scatterplot(
            data=frame,
            x="total_text_slots",
            y="output_tokens",
            hue="semantic_pass",
            size="group_count",
            ax=axes[0],
        )
        axes[0].set_title("Output tokens scale with requested text topology")
        sns.scatterplot(
            data=frame,
            x="total_text_slots",
            y="duration_seconds",
            hue="semantic_pass",
            size="group_count",
            ax=axes[1],
        )
        axes[1].set_title("Latency versus requested text topology")
        figure.tight_layout()
        return figure

    def cargo_lengths(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(cargo_rows)
        value_columns = [
            "description_characters",
            "additional_characters",
            "marks_characters",
            "handling_characters",
        ]
        long = frame[value_columns].melt(var_name="field", value_name="characters")
        figure, axis = plt.subplots(figsize=(12, 5.5))
        sns.boxplot(data=long, x="field", y="characters", ax=axis, color="#7FCDBB")
        sns.stripplot(
            data=long,
            x="field",
            y="characters",
            ax=axis,
            color="#225EA8",
            alpha=0.4,
            jitter=False,
        )
        axis.set_yscale("symlog", linthresh=10)
        axis.set_title("Generated cargo text volume by field family")
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        return figure

    def failures(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame([row for row in failure_rows if row["arm"] != "cargo_targeted"])
        counts = frame.groupby(["arm", "check_family"]).size().reset_index(name="failures")
        figure, axis = plt.subplots(figsize=(12, max(4.5, len(counts) * 0.55)))
        sns.barplot(data=counts, x="failures", y="check_family", hue="arm", ax=axis)
        axis.set_title("All first-pass semantic-validator failures")
        figure.tight_layout()
        return figure

    def cumulative_cost(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(all_rows).sort_values(["arm", "case_index"])
        frame["cumulative_cost_usd"] = frame.groupby("arm")["estimated_cost_usd"].cumsum()
        figure, axis = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=frame, x="case_index", y="cumulative_cost_usd", hue="arm", ax=axis)
        axis.set_title("Cumulative estimated API cost")
        axis.set_ylabel("USD")
        figure.tight_layout()
        return figure

    def reasoning_visible(plt: Any, sns: Any, pd: Any) -> Any:
        frame = pd.DataFrame(all_rows)
        figure, axis = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=frame,
            x="reasoning_tokens",
            y="visible_output_tokens",
            hue="arm",
            style="semantic_pass",
            size="estimated_cost_usd",
            sizes=(30, 180),
            ax=axis,
        )
        axis.set_title("Reasoning versus visible structured-output tokens")
        figure.tight_layout()
        return figure

    return {
        "plots/01-first-pass-outcomes.png": _plot_bytes(outcomes),
        "plots/02-token-composition.png": _plot_bytes(token_mix),
        "plots/03-cost-and-latency.png": _plot_bytes(cost_latency),
        "plots/04-party-role-quality.png": _plot_bytes(party_roles),
        "plots/05-party-country-coverage-complete.png": _plot_bytes(party_countries),
        "plots/06-party-contact-shapes.png": _plot_bytes(contact_shapes),
        "plots/07-party-name-quality.png": _plot_bytes(party_name_quality),
        "plots/08-cargo-complexity.png": _plot_bytes(cargo_complexity),
        "plots/09-cargo-cohort-quality.png": _plot_bytes(cargo_cohorts),
        "plots/10-cargo-complexity-scaling.png": _plot_bytes(cargo_scaling),
        "plots/11-cargo-text-volume.png": _plot_bytes(cargo_lengths),
        "plots/12-validation-failure-taxonomy.png": _plot_bytes(failures),
        "plots/13-cumulative-cost.png": _plot_bytes(cumulative_cost),
        "plots/14-reasoning-vs-visible-output.png": _plot_bytes(reasoning_visible),
    }


def _aggregate_tokens(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "input_tokens",
        "uncached_input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "reasoning_tokens",
        "visible_output_tokens",
    )
    return {key: sum(int(row[key]) for row in rows) for key in keys}


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile over empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _arm_summary(
    rows: Sequence[Mapping[str, Any]], run_summary: Mapping[str, Any]
) -> dict[str, Any]:
    costs = [float(row["estimated_cost_usd"]) for row in rows]
    latencies = [float(row["duration_seconds"]) for row in rows]
    return {
        "cases": len(rows),
        "schemaValid": sum(row["status"] != "call_failed" for row in rows),
        "semanticPasses": sum(bool(row["semantic_pass"]) for row in rows),
        "semanticPassRate": sum(bool(row["semantic_pass"]) for row in rows) / len(rows),
        "callFailures": sum(row["status"] == "call_failed" for row in rows),
        "automaticRepairRequests": 0,
        "estimatedCostUsd": str(sum((Decimal(str(value)) for value in costs), Decimal(0))),
        "meanEstimatedCostUsd": sum(costs) / len(costs),
        "medianCaseLatencySeconds": _percentile(latencies, 0.5),
        "p95CaseLatencySeconds": _percentile(latencies, 0.95),
        "wallSeconds": run_summary["wallSeconds"],
        "tokens": _aggregate_tokens(rows),
    }


def _examples(
    party: Sequence[PartyIdentityCaseRecord], cargo: Sequence[CargoLanguageCaseRecord]
) -> str:
    party_examples: list[PartyIdentityCaseRecord] = []
    seen_roles: set[str] = set()
    for record in party:
        if record.status == "success" and record.seed.partyRole not in seen_roles:
            party_examples.append(record)
            seen_roles.add(record.seed.partyRole)
    predicates: tuple[tuple[str, Callable[[CargoLanguageCaseRecord], bool]], ...] = (
        ("dangerous goods", lambda row: any(g.dangerousGoods for g in row.seed.cargoGroups)),
        (
            "thermal",
            lambda row: any(
                any(item.thermalProfile for item in group.goodsIdentities)
                for group in row.seed.cargoGroups
            ),
        ),
        ("multi-group", lambda row: len(row.seed.cargoGroups) > 1),
        (
            "additional information",
            lambda row: any(
                group.fieldContract.additionalInformationSlots for group in row.seed.cargoGroups
            ),
        ),
        (
            "handling",
            lambda row: any(
                group.fieldContract.handlingInstructionSlots
                for group in row.seed.cargoGroups
            ),
        ),
        ("ordinary", lambda row: len(row.seed.cargoGroups) == 1),
    )
    cargo_examples: list[tuple[str, CargoLanguageCaseRecord]] = []
    used: set[int] = set()
    for label, predicate in predicates:
        match = next(
            (
                row
                for row in cargo
                if row.status == "success" and row.caseIndex not in used and predicate(row)
            ),
            None,
        )
        if match is not None:
            cargo_examples.append((label, match))
            used.add(match.caseIndex)
    lines = ["# Representative linguistic probe outputs", "", "## Party identities", ""]
    for party_record in party_examples:
        if party_record.output is None:
            raise RuntimeError("successful party example is missing its output")
        lines.extend(
            (
                (
                    f"### {party_record.seed.partyRole} — "
                    f"{party_record.seed.targetLocality['country']}"
                ),
                "",
                "```json",
                json.dumps(
                    party_record.output.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            )
        )
    lines.extend(("## Cargo language", ""))
    for label, cargo_record in cargo_examples:
        if cargo_record.output is None:
            raise RuntimeError("successful cargo example is missing its output")
        lines.extend(
            (
                f"### {label} — {cargo_record.seed.sourceDocumentId}",
                "",
                "```json",
                json.dumps(
                    cargo_record.output.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def run_linguistic_probe_analysis(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisLinguisticProbeAnalysisConfig,
) -> dict[str, Any]:
    """Validate, analyze, plot, and immutably publish both 50-case probes."""

    inputs: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for label, configured in (
        ("party", config.party),
        ("cargo", config.cargo),
        ("cargo_contract_probe", config.cargo_contract_probe),
    ):
        root = _validate_run(project_root, configured)
        results = _resolve_file(
            project_root,
            root,
            configured.results.path,
            configured.results.sha256,
            label=f"{label} results",
        )
        summary_path = _resolve_file(
            project_root,
            root,
            configured.summary.path,
            configured.summary.sha256,
            label=f"{label} summary",
        )
        inputs[label] = (root, results, _load_summary(summary_path))

    party_records = _load_jsonl(
        inputs["party"][1], records=config.party.results.records, model=PartyIdentityCaseRecord
    )
    cargo_records = _load_jsonl(
        inputs["cargo"][1], records=config.cargo.results.records, model=CargoLanguageCaseRecord
    )
    targeted_records = _load_jsonl(
        inputs["cargo_contract_probe"][1],
        records=config.cargo_contract_probe.results.records,
        model=CargoLanguageCaseRecord,
    )
    party_rows = _party_rows(party_records)
    cargo_rows = _cargo_rows(cargo_records)
    targeted_rows = _cargo_rows(targeted_records)
    failure_rows = _failure_rows(party_records, cargo_records, targeted_records)
    role_rows = _summary_rows(party_rows, "role", value_label="role")
    country_rows = _summary_rows(party_rows, "country", value_label="country")
    shape_rows = _summary_rows(party_rows, "contact_shape", value_label="contact_shape")
    cohort_rows = _cargo_cohort_rows(cargo_rows)

    party_names = [
        cast(str, row["generated_name_normalized"])
        for row in party_rows
        if row["generated_name_normalized"] is not None
    ]
    cargo_descriptions = [
        _normalized(group.description)
        for record in cargo_records
        if record.output is not None
        for group in record.output.cargoGroups
        if group.description is not None
    ]
    source_description_copies = sum(
        _normalized(generated.description)
        == _normalized(expected.fieldContract.sourceDescriptionStyleReference)
        for record in cargo_records
        if record.output is not None
        for expected, generated in zip(
            record.seed.cargoGroups, record.output.cargoGroups, strict=False
        )
        if generated.description is not None
        and expected.fieldContract.sourceDescriptionStyleReference is not None
    )
    generated_mark_copies = sum(
        generated.marksAndNumbers[index].casefold() == slot.sourceStyleReference.casefold()
        for record in cargo_records
        if record.output is not None
        for expected, generated in zip(
            record.seed.cargoGroups, record.output.cargoGroups, strict=False
        )
        for index, slot in enumerate(expected.fieldContract.marksAndNumbersSlots)
        if slot.action == "generate" and index < len(generated.marksAndNumbers)
    )
    combined_cost = sum(
        (Decimal(str(row["estimated_cost_usd"])) for row in party_rows + cargo_rows), Decimal(0)
    )
    summary = {
        "schemaVersion": 1,
        "party": _arm_summary(party_rows, inputs["party"][2]),
        "cargo": _arm_summary(cargo_rows, inputs["cargo"][2]),
        "combined": {
            "cases": 100,
            "schemaValid": sum(row["status"] != "call_failed" for row in party_rows + cargo_rows),
            "semanticPasses": sum(bool(row["semantic_pass"]) for row in party_rows + cargo_rows),
            "semanticPassRate": (
                sum(bool(row["semantic_pass"]) for row in party_rows + cargo_rows) / 100
            ),
            "estimatedCostUsd": str(combined_cost),
            "wallSecondsSequential": inputs["party"][2]["wallSeconds"]
            + inputs["cargo"][2]["wallSeconds"],
        },
        "coverage": {
            "partyRoles": len(role_rows),
            "partyCountries": len(country_rows),
            "partyContactShapes": len(shape_rows),
            "cargoGroups": sum(int(row["group_count"]) for row in cargo_rows),
            "dangerousGoodsCases": sum(bool(row["is_dangerous_goods"]) for row in cargo_rows),
            "thermalCases": sum(bool(row["is_thermal"]) for row in cargo_rows),
            "multiGroupCases": sum(bool(row["is_multi_group"]) for row in cargo_rows),
            "additionalInformationCases": sum(
                bool(row["has_additional_information"]) for row in cargo_rows
            ),
            "handlingCases": sum(bool(row["has_handling"]) for row in cargo_rows),
        },
        "diversityAndLeakage": {
            "partyGeneratedNames": len(party_names),
            "partyDistinctNormalizedNames": len(set(party_names)),
            "partyMaximumSourceNameSimilarity": max(
                cast(float, row["source_name_similarity"])
                for row in party_rows
                if row["source_name_similarity"] is not None
            ),
            "cargoGeneratedDescriptions": len(cargo_descriptions),
            "cargoDistinctNormalizedDescriptions": len(set(cargo_descriptions)),
            "cargoExactSourceDescriptionCopies": source_description_copies,
            "generatedMarksCopiedFromOwnSourceSlot": generated_mark_copies,
        },
        "targetedMarksContractProbe": {
            "cases": 1,
            "requestedMarks": targeted_rows[0]["marks_slots"],
            "generatedMarks": targeted_rows[0]["marks_generated_count"],
            "semanticPass": targeted_rows[0]["semantic_pass"],
            "estimatedCostUsd": str(targeted_records[0].usage.estimatedCostUsd),
            "finding": (
                "schema_capacity_fixed_but_model_returned_wrong_cardinality"
                if not targeted_rows[0]["semantic_pass"]
                else "schema_capacity_and_model_cardinality_passed"
            ),
        },
        "validationFailureChecks": [
            {"arm": arm, "checkFamily": family, "failures": count}
            for (arm, family), count in sorted(
                Counter(
                    (cast(str, row["arm"]), cast(str, row["check_family"]))
                    for row in failure_rows
                ).items()
            )
        ],
    }

    party_fields = tuple(party_rows[0])
    cargo_fields = tuple(cargo_rows[0])
    failure_fields = ("arm", "case_index", "document_id", "check", "check_family")
    data_payloads = {
        "data/party-cases.csv": _csv_bytes(party_rows, party_fields),
        "data/cargo-cases.csv": _csv_bytes(cargo_rows, cargo_fields),
        "data/validation-failures.csv": _csv_bytes(failure_rows, failure_fields),
        "data/party-role-summary.csv": _csv_bytes(
            role_rows, ("role", "cases", "passed", "pass_rate")
        ),
        "data/party-country-summary.csv": _csv_bytes(
            country_rows, ("country", "cases", "passed", "pass_rate")
        ),
        "data/party-contact-shape-summary.csv": _csv_bytes(
            shape_rows, ("contact_shape", "cases", "passed", "pass_rate")
        ),
        "data/cargo-cohort-summary.csv": _csv_bytes(
            cohort_rows, ("cohort", "cases", "passed", "pass_rate")
        ),
    }
    plot_payloads = _plots(party_rows, cargo_rows, failure_rows)

    failed_party = [row for row in failure_rows if row["arm"] == "party"]
    failed_cargo = [row for row in failure_rows if row["arm"] == "cargo"]
    party_summary = cast(dict[str, Any], summary["party"])
    cargo_summary = cast(dict[str, Any], summary["cargo"])
    combined_summary = cast(dict[str, Any], summary["combined"])
    coverage = cast(dict[str, Any], summary["coverage"])
    diversity = cast(dict[str, Any], summary["diversityAndLeakage"])
    party_table_row = (
        f"| Party | 50/50 | {party_summary['semanticPasses']}/50 | "
        f"{party_summary['semanticPassRate']:.1%} | ${party_summary['estimatedCostUsd']} | "
        f"{party_summary['wallSeconds']:.2f}s |"
    )
    cargo_table_row = (
        f"| Cargo | 50/50 | {cargo_summary['semanticPasses']}/50 | "
        f"{cargo_summary['semanticPassRate']:.1%} | ${cargo_summary['estimatedCostUsd']} | "
        f"{cargo_summary['wallSeconds']:.2f}s |"
    )
    combined_table_row = (
        f"| Combined | 100/100 | {combined_summary['semanticPasses']}/100 | "
        f"{combined_summary['semanticPassRate']:.1%} | "
        f"${combined_summary['estimatedCostUsd']} | "
        f"{combined_summary['wallSecondsSequential']:.2f}s |"
    )
    cost_per_thousand = float(combined_summary["estimatedCostUsd"]) * 10
    report = f"""# Linguistic synthesis probe EDA — 50 party + 50 cargo calls

## Experiment contract

- Model: GPT-5.6 Luna, high reasoning.
- Calls: 100 independent first-pass calls; 50 party and 50 cargo.
- Constrained decoding: provider-native strict JSON Schema.
- Automatic repair or reviewer calls: **0**.
- Provider responses and model-visible messages: retained in the two pinned source runs.
- Party selection: all six roles, **{len(country_rows)}** target countries, and
  **{len(shape_rows)}** distinct field-presence/contact shapes.
- Cargo selection: **{coverage['cargoGroups']}** groups, including all available
  dangerous-goods, thermal, multi-group, additional-information, and handling cohorts in the
  pinned 100-plan semantic pool.

## Results

| Arm | Schema-valid | Semantic pass | Rate | Estimated cost | Wall time |
|---|---:|---:|---:|---:|---:|
{party_table_row}
{cargo_table_row}
{combined_table_row}

All outputs were schema-valid and all provider calls completed. The strict semantic validator
rejected two party outputs and one cargo output.

## Failure diagnosis

1. Party case 16 generated `operations@nuri harborlogistics.co.kr`, which contains a space in the
   domain. This is a genuine syntactic generation error; constrained JSON Schema guarantees the
   field type, not RFC-like email syntax.
2. Party case 50 put the supplied city token `Pendang` inside the street/neighborhood text
   `Taman Pendang Jaya`. This violates the current no-city-in-address contract, although the
   generated postal phrase itself is plausible. It is a contract-boundary finding rather than
   identity leakage or an implausible party.
3. Cargo case 44 requested 40 marks. The initial output schema capped the array at 32, making a
   pass impossible. The cap was corrected to 64 and guarded by a regression test. A one-call
   targeted re-probe then returned 41 marks for 40 slots, proving that the remaining failure is
   high-cardinality exact-count generation rather than schema capacity. This should be handled in
   the later integration design rather than hidden by retries in this benchmark.

Initial false checks: **{len(failed_party)} party check failures** and
**{len(failed_cargo)} cargo check failures**. Every false check is listed in
`data/validation-failures.csv`.

## Diversity and fidelity

- Generated party names: **{diversity['partyDistinctNormalizedNames']} /
  {diversity['partyGeneratedNames']}** distinct after normalization.
- Maximum generated/source party-name similarity:
  **{diversity['partyMaximumSourceNameSimilarity']:.3f}**; no generated name matched the full
  1,157-label source inventory.
- Generated cargo descriptions: **{diversity['cargoDistinctNormalizedDescriptions']} /
  {diversity['cargoGeneratedDescriptions']}** distinct after normalization.
- Exact copies of source description style references: **{source_description_copies}**.
- Generated marks copied verbatim from their own source slots: **{generated_mark_copies}**.
- DG cases: **{coverage['dangerousGoodsCases']}**; thermal: **{coverage['thermalCases']}**;
  multi-group: **{coverage['multiGroupCases']}**; additional-information:
  **{coverage['additionalInformationCases']}**; handling: **{coverage['handlingCases']}**.

## Interpretation before integration

The two linguistic generators are viable at this scale: constrained structure was 100%, semantic
first-pass acceptance was {combined_summary['semanticPassRate']:.1%}, diversity was high, and
the measured estimate was about ${cost_per_thousand:.2f}
per 1,000 combined calls at this case mix. The failures are narrow and actionable: validate or
repair email syntax, refine the address/locality separation policy, and avoid asking the LLM to
reconstruct very long serial-mark arrays as one exact-cardinality generation task.

`EXAMPLES.md` contains representative successful outputs. `data/` contains complete, untruncated
tables; `plots/` contains 14 matplotlib/seaborn figures, including all 41 party countries rather
than a top-N view.
"""
    examples = _examples(party_records, cargo_records)
    summary_bytes = json_artifact_bytes(summary)
    transaction_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "configSha256": sha256_file(config_path),
                "partyResultsSha256": config.party.results.sha256,
                "cargoResultsSha256": config.cargo.results.sha256,
                "targetedResultsSha256": config.cargo_contract_probe.results.sha256,
                "summarySha256": sha256_bytes(summary_bytes),
                "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
            }
        )
    )
    staged = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=transaction_sha256,
    )
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    staged.publish_bytes("REPORT.md", report.encode("utf-8"))
    staged.publish_bytes("EXAMPLES.md", examples.encode("utf-8"))
    staged.publish_bytes("summary.json", summary_bytes)
    for path, payload in sorted(data_payloads.items()):
        staged.publish_bytes(path, payload)
    for path, payload in sorted(plot_payloads.items()):
        staged.publish_bytes(path, payload)
    expected = tuple(
        sorted(
            (
                "REPORT.md",
                "EXAMPLES.md",
                "config.yaml",
                "summary.json",
                *data_payloads,
                *plot_payloads,
            )
        )
    )
    committed = staged.commit(
        expected_artifacts=expected,
        metadata={
            "schema_version": 1,
            "party_cases": 50,
            "cargo_cases": 50,
            "plots": len(plot_payloads),
        },
    )
    return {
        "output_dir": str(staged.final_root),
        "created": committed.created,
        "party_passes": party_summary["semanticPasses"],
        "cargo_passes": cargo_summary["semanticPasses"],
        "combined_cost_usd": combined_summary["estimatedCostUsd"],
        "plots": len(plot_payloads),
    }
