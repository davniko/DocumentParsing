"""Complete inspection tables, plots, and narrative for structured synthesis."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, cast

from document_ocr.synthesis.run_safety import StagedArtifactRun


class StructuredReportingError(RuntimeError):
    """A complete structured-baseline report could not be produced."""


_PROPOSAL_COLUMNS = (
    "proposal_id",
    "synthetic_position",
    "base_document_id",
    "request_id",
    "cohort_id",
    "group_id",
    "package_id",
    "package_count_in_group",
    "package_identity",
    "package_role",
    "route_tier",
    "contextual_support_tier",
    "contextual_support_rows",
    "contextual_support_templates",
    "contextual_support_sha256",
    "contextual_distance",
    "contextual_maximum_distance",
    "selected_candidate",
    "source_quantity",
    "generated_quantity",
    "source_group_quantities",
    "generated_group_quantities",
    "source_gross_weight_value",
    "generated_gross_weight_value",
    "gross_weight_unit",
    "source_net_weight_value",
    "generated_net_weight_value",
    "net_weight_unit",
    "source_volume_value",
    "generated_volume_value",
    "volume_unit",
    "sampled_features",
    "raw_proposals",
    "rejected_proposals",
    "acceptance_yield",
    "exact_train_row_copy",
)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return stream.getvalue().encode("utf-8")


def _json_cell(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise StructuredReportingError("report input contains non-finite JSON") from error


def _mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuredReportingError(f"{location} must be an object")
    return cast(Mapping[str, Any], value)


def _list(value: Any, *, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StructuredReportingError(f"{location} must be an array")
    return value


def _text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise StructuredReportingError(f"{location} must be non-empty text")
    return value


def _integer(value: Any, *, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StructuredReportingError(
            f"{location} must be an integer greater than or equal to {minimum}"
        )
    return value


def _number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredReportingError(f"{location} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise StructuredReportingError(f"{location} must be finite")
    return converted


def _optional_positive_number(value: Any, *, location: str) -> int | float | None:
    if value is None:
        return None
    numeric = _number(value, location=location)
    if numeric <= 0:
        raise StructuredReportingError(f"{location} must be positive when present")
    return cast(int | float, value)


def _plot_runtime() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
        import pandas  # type: ignore[import-untyped]
        import seaborn  # type: ignore[import-untyped]
    except ImportError as error:
        raise StructuredReportingError(
            "structured reporting requires matplotlib, pandas, and seaborn"
        ) from error
    seaborn.set_theme(style="whitegrid", context="notebook")
    return pyplot, pandas, seaborn


def _publish_figure(
    *, stage: StagedArtifactRun, pyplot: Any, figure: Any, relative_path: str
) -> None:
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "document-ocr structured synthesis"},
    )
    pyplot.close(figure)
    stage.publish_bytes(relative_path, stream.getvalue())


def _selection_tables(
    selected_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    for index, source in enumerate(selected_rows):
        location = f"selected_rows[{index}]"
        position = _integer(source.get("position"), location=f"{location}.position")
        if position in seen_positions:
            raise StructuredReportingError("selected source positions are not unique")
        seen_positions.add(position)
        document_id = _text(source.get("document_id"), location=f"{location}.document_id")
        strata = _mapping(source.get("strata"), location=f"{location}.strata")
        expected_strata = (
            "document_type",
            "source_corpus",
            "page_bucket",
            "container_bucket",
        )
        strata_values = {
            name: _text(strata.get(name), location=f"{location}.strata.{name}")
            for name in expected_strata
        }
        contexts = tuple(
            _text(value, location=f"{location}.contexts")
            for value in _list(source.get("contexts"), location=f"{location}.contexts")
        )
        profile_routes = _list(
            source.get("cargo_profile_routes"),
            location=f"{location}.cargo_profile_routes",
        )
        sources.append(
            {
                "position": position,
                "document_id": document_id,
                "template_id": _text(source.get("template_id"), location=f"{location}.template_id"),
                "carrier_family": _text(
                    source.get("carrier_family"), location=f"{location}.carrier_family"
                ),
                **strata_values,
                "contexts": "|".join(contexts),
                "cargo_profile_routes": _json_cell(profile_routes),
                "complete_selection_receipt": _json_cell(source),
            }
        )
        strata_rows.extend(
            {
                "position": position,
                "document_id": document_id,
                "dimension": name,
                "value": value,
            }
            for name, value in strata_values.items()
        )
        context_rows.extend(
            {
                "position": position,
                "document_id": document_id,
                "context": context,
            }
            for context in contexts
        )
    return sources, strata_rows, context_rows


def _proposal_tables(
    proposal_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    proposals: list[dict[str, Any]] = []
    drivers: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    measures: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    proposal_ids: set[str] = set()
    request_ids: set[str] = set()
    for index, source in enumerate(proposal_rows):
        location = f"proposal_rows[{index}]"
        missing = set(_PROPOSAL_COLUMNS) - set(source)
        extra = set(source) - set(_PROPOSAL_COLUMNS)
        if missing or extra:
            raise StructuredReportingError(
                f"{location} differs from the reporting contract; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        proposal_id = _text(source["proposal_id"], location=f"{location}.proposal_id")
        request_id = _text(source["request_id"], location=f"{location}.request_id")
        if proposal_id in proposal_ids or request_id in request_ids:
            raise StructuredReportingError("proposal and request identifiers must be unique")
        proposal_ids.add(proposal_id)
        request_ids.add(request_id)
        source_quantities = _mapping(
            source["source_group_quantities"],
            location=f"{location}.source_group_quantities",
        )
        generated_quantities = _mapping(
            source["generated_group_quantities"],
            location=f"{location}.generated_group_quantities",
        )
        if set(source_quantities) != set(generated_quantities):
            raise StructuredReportingError(f"{location} changes cargo-group package topology")
        package_count = _integer(
            source["package_count_in_group"],
            location=f"{location}.package_count_in_group",
            minimum=1,
        )
        if len(source_quantities) != package_count:
            raise StructuredReportingError(
                f"{location}.package_count_in_group differs from its quantity maps"
            )
        driver_id = _text(source["package_id"], location=f"{location}.package_id")
        if driver_id not in source_quantities:
            raise StructuredReportingError(f"{location} driver is absent from its package map")
        source_driver = _integer(
            source["source_quantity"],
            location=f"{location}.source_quantity",
            minimum=1,
        )
        generated_driver = _integer(
            source["generated_quantity"],
            location=f"{location}.generated_quantity",
            minimum=1,
        )
        if source_quantities[driver_id] != source_driver:
            raise StructuredReportingError(f"{location} source driver quantity is inconsistent")
        if generated_quantities[driver_id] != generated_driver:
            raise StructuredReportingError(f"{location} generated driver quantity is inconsistent")
        if source["exact_train_row_copy"] is not False:
            raise StructuredReportingError(f"{location} is an exact fitted-train-row copy")
        acceptance_yield = _number(
            source["acceptance_yield"], location=f"{location}.acceptance_yield"
        )
        if not 0 < acceptance_yield <= 1:
            raise StructuredReportingError(f"{location}.acceptance_yield is outside (0, 1]")
        contextual_distance = _number(
            source["contextual_distance"],
            location=f"{location}.contextual_distance",
        )
        contextual_maximum = _number(
            source["contextual_maximum_distance"],
            location=f"{location}.contextual_maximum_distance",
        )
        if contextual_distance < 0 or contextual_maximum < 0:
            raise StructuredReportingError(f"{location} contextual distances must be non-negative")
        tolerance = 1e-12 * max(1.0, contextual_maximum)
        if contextual_distance > contextual_maximum + tolerance:
            raise StructuredReportingError(
                f"{location} lies outside package-aware contextual support"
            )
        scalar = {
            key: source[key]
            for key in _PROPOSAL_COLUMNS
            if key
            not in {
                "source_group_quantities",
                "generated_group_quantities",
                "sampled_features",
            }
        }
        proposals.append(
            {
                **scalar,
                "source_group_quantities": _json_cell(source_quantities),
                "generated_group_quantities": _json_cell(generated_quantities),
                "sampled_features": _json_cell(source["sampled_features"]),
            }
        )
        common = {
            "proposal_id": proposal_id,
            "synthetic_position": source["synthetic_position"],
            "base_document_id": source["base_document_id"],
            "request_id": request_id,
            "cohort_id": source["cohort_id"],
            "group_id": source["group_id"],
            "package_identity": source["package_identity"],
            "package_role": source["package_role"],
            "route_tier": source["route_tier"],
            "contextual_support_tier": source["contextual_support_tier"],
            "contextual_support_rows": source["contextual_support_rows"],
            "contextual_support_templates": source["contextual_support_templates"],
            "contextual_distance": contextual_distance,
            "contextual_maximum_distance": contextual_maximum,
            "selected_candidate": source["selected_candidate"],
        }
        drivers.append(
            {
                **common,
                "driver_package_id": driver_id,
                "source_quantity": source_driver,
                "generated_quantity": generated_driver,
                "changed": source_driver != generated_driver,
            }
        )
        for package_id in sorted(source_quantities):
            source_quantity = _integer(
                source_quantities[package_id],
                location=f"{location}.source_group_quantities.{package_id}",
                minimum=1,
            )
            generated_quantity = _integer(
                generated_quantities[package_id],
                location=f"{location}.generated_group_quantities.{package_id}",
                minimum=1,
            )
            packages.append(
                {
                    **{
                        key: value
                        for key, value in common.items()
                        if key not in {"package_identity", "package_role"}
                    },
                    "driver_package_identity": source["package_identity"],
                    "driver_package_role": source["package_role"],
                    "package_id": package_id,
                    "is_driver": package_id == driver_id,
                    "source_quantity": source_quantity,
                    "generated_quantity": generated_quantity,
                    "changed": source_quantity != generated_quantity,
                }
            )
        for measure, source_key, generated_key, unit_key in (
            (
                "gross_weight_total",
                "source_gross_weight_value",
                "generated_gross_weight_value",
                "gross_weight_unit",
            ),
            (
                "net_weight_total",
                "source_net_weight_value",
                "generated_net_weight_value",
                "net_weight_unit",
            ),
            (
                "volume_total",
                "source_volume_value",
                "generated_volume_value",
                "volume_unit",
            ),
        ):
            source_value = _optional_positive_number(
                source[source_key], location=f"{location}.{source_key}"
            )
            generated_value = _optional_positive_number(
                source[generated_key], location=f"{location}.{generated_key}"
            )
            unit = source[unit_key]
            if (source_value is None) != (generated_value is None) or (
                (source_value is None) != (unit is None)
            ):
                raise StructuredReportingError(
                    f"{location} changes {measure} presence or omits its unit"
                )
            if source_value is not None:
                measures.append(
                    {
                        **common,
                        "measure": measure,
                        "unit": _text(unit, location=f"{location}.{unit_key}"),
                        "source_value": source_value,
                        "generated_value": generated_value,
                        "changed": source_value != generated_value,
                    }
                )
        routes.append(
            {
                "proposal_id": proposal_id,
                "request_id": request_id,
                "cohort_id": source["cohort_id"],
                "base_document_id": source["base_document_id"],
                "group_id": source["group_id"],
                "route_tier": source["route_tier"],
                "contextual_support_tier": source["contextual_support_tier"],
                "contextual_support_rows": source["contextual_support_rows"],
                "contextual_support_templates": source["contextual_support_templates"],
                "contextual_distance": contextual_distance,
                "contextual_maximum_distance": contextual_maximum,
                "selected_candidate": source["selected_candidate"],
                "raw_proposals": source["raw_proposals"],
                "rejected_proposals": source["rejected_proposals"],
                "acceptance_yield": acceptance_yield,
            }
        )
    return proposals, drivers, packages, measures, routes


def _change_tables(
    change_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    dates: list[dict[str, Any]] = []
    for document_position, document_changes in enumerate(change_rows):
        for change_index, source in enumerate(document_changes):
            location = f"change_rows[{document_position}][{change_index}]"
            row = {
                "synthetic_position": document_position,
                "target_path": _text(source.get("target_path"), location=f"{location}.target_path"),
                "family": _text(source.get("family"), location=f"{location}.family"),
                "method": _text(source.get("method"), location=f"{location}.method"),
                "coupling_group": source.get("coupling_group"),
                "old_value": _json_cell(source.get("old_value")),
                "new_value": _json_cell(source.get("new_value")),
            }
            changes.append(row)
            if row["family"] != "document_date":
                continue
            old_value = source.get("old_value")
            new_value = source.get("new_value")
            if not isinstance(old_value, str) or not isinstance(new_value, str):
                raise StructuredReportingError(f"{location} date values are not ISO text")
            try:
                old_date = date.fromisoformat(old_value)
                new_date = date.fromisoformat(new_value)
            except ValueError as error:
                raise StructuredReportingError(
                    f"{location} date values are not ISO dates"
                ) from error
            dates.append(
                {
                    "synthetic_position": document_position,
                    "target_path": row["target_path"],
                    "method": row["method"],
                    "coupling_group": row["coupling_group"],
                    "source_date": old_value,
                    "generated_date": new_value,
                    "source_year": old_date.year,
                    "generated_year": new_date.year,
                    "source_month": old_date.month,
                    "generated_month": new_date.month,
                    "day_shift": (new_date - old_date).days,
                }
            )
    return changes, dates


def _profile_tables(
    profile_results: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cohorts: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    seen_cohorts: set[str] = set()
    all_requests: set[str] = set()
    for profile_index, wrapper in enumerate(profile_results):
        location = f"profile_results[{profile_index}]"
        required = {
            "cohort_id",
            "request_ids",
            "sample_seed",
            "quality_acceptance",
            "run",
            "runtime",
        }
        if set(wrapper) != required:
            raise StructuredReportingError(
                f"{location} differs from the profile wrapper contract; "
                f"missing={sorted(required - set(wrapper))}, "
                f"extra={sorted(set(wrapper) - required)}"
            )
        cohort_id = _text(wrapper["cohort_id"], location=f"{location}.cohort_id")
        if cohort_id in seen_cohorts:
            raise StructuredReportingError("profile cohort identifiers are not unique")
        seen_cohorts.add(cohort_id)
        request_ids = tuple(
            _text(value, location=f"{location}.request_ids")
            for value in _list(wrapper["request_ids"], location=f"{location}.request_ids")
        )
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise StructuredReportingError(f"{location}.request_ids must be non-empty and unique")
        if all_requests.intersection(request_ids):
            raise StructuredReportingError("profile cohorts overlap request identifiers")
        all_requests.update(request_ids)
        run = _mapping(wrapper["run"], location=f"{location}.run")
        contract = _mapping(run.get("contract"), location=f"{location}.run.contract")
        request = _mapping(contract.get("request"), location=f"{location}.run.contract.request")
        profile = _text(request.get("profile"), location=f"{location}.run.contract.request.profile")
        routing = _mapping(run.get("routing"), location=f"{location}.run.routing")
        decisions = _list(routing.get("decisions"), location=f"{location}.run.routing.decisions")
        if len(decisions) != 1:
            raise StructuredReportingError(f"{location} must contain one cohort route decision")
        decision = _mapping(decisions[0], location=f"{location}.run.routing.decisions[0]")
        route_tier = _text(
            decision.get("selected_tier"),
            location=f"{location}.run.routing.decisions[0].selected_tier",
        )
        selection = _mapping(run.get("selection"), location=f"{location}.run.selection")
        selected_candidate = _text(
            selection.get("selected_candidate"),
            location=f"{location}.run.selection.selected_candidate",
        )
        quality = _mapping(wrapper["quality_acceptance"], location=f"{location}.quality_acceptance")
        if quality.get("accepted") is not True:
            raise StructuredReportingError(f"{location} did not pass quality acceptance")
        runtime = _mapping(wrapper["runtime"], location=f"{location}.runtime")
        runtime_required = {
            "final_fit",
            "final_sample",
            "run_resources",
            "benchmark_runs",
        }
        if set(runtime) != runtime_required:
            raise StructuredReportingError(
                f"{location}.runtime differs from the runtime receipt contract; "
                f"missing={sorted(runtime_required - set(runtime))}, "
                f"extra={sorted(set(runtime) - runtime_required)}"
            )
        final_fit = _mapping(runtime.get("final_fit"), location=f"{location}.runtime.final_fit")
        final_sample = _mapping(
            runtime.get("final_sample"), location=f"{location}.runtime.final_sample"
        )
        final_proposal = _mapping(
            run.get("final_proposal"), location=f"{location}.run.final_proposal"
        )
        final_novelty = _mapping(run.get("final_novelty"), location=f"{location}.run.final_novelty")
        audit = _mapping(run.get("audit"), location=f"{location}.run.audit")
        resources = _mapping(
            runtime.get("run_resources"), location=f"{location}.runtime.run_resources"
        )
        cohorts.append(
            {
                "cohort_id": cohort_id,
                "profile": profile,
                "route_tier": route_tier,
                "request_count": len(request_ids),
                "request_ids": "|".join(request_ids),
                "sample_seed": _integer(wrapper["sample_seed"], location=f"{location}.sample_seed"),
                "selected_candidate": selected_candidate,
                "route_support_rows": _integer(
                    decision.get("selected_rows"),
                    location=f"{location}.run.routing.decisions[0].selected_rows",
                ),
                "accepted_source_rows": _integer(
                    audit.get("accepted_rows"),
                    location=f"{location}.run.audit.accepted_rows",
                ),
                "quarantined_source_rows": _integer(
                    audit.get("quarantined_rows"),
                    location=f"{location}.run.audit.quarantined_rows",
                ),
                "paired_runs": _integer(
                    quality.get("paired_runs"),
                    location=f"{location}.quality_acceptance.paired_runs",
                ),
                "mean_quality_difference_vs_empirical": _number(
                    quality.get("mean_quality_difference_vs_empirical"),
                    location=f"{location}.quality_acceptance.mean_quality_difference_vs_empirical",
                ),
                "paired_difference_lower_bound": _number(
                    quality.get("paired_difference_lower_bound"),
                    location=f"{location}.quality_acceptance.paired_difference_lower_bound",
                ),
                "final_novelty_fraction": _number(
                    quality.get("final_novelty_fraction"),
                    location=f"{location}.quality_acceptance.final_novelty_fraction",
                ),
                "final_acceptance_yield": _number(
                    quality.get("final_acceptance_yield"),
                    location=f"{location}.quality_acceptance.final_acceptance_yield",
                ),
                "final_fit_seconds": _number(
                    final_fit.get("elapsed_seconds"),
                    location=f"{location}.runtime.final_fit.elapsed_seconds",
                ),
                "final_sample_seconds": _number(
                    final_sample.get("elapsed_seconds"),
                    location=f"{location}.runtime.final_sample.elapsed_seconds",
                ),
                "final_raw_proposals": _integer(
                    final_proposal.get("raw_proposals"),
                    location=f"{location}.run.final_proposal.raw_proposals",
                ),
                "final_rejected_proposals": _integer(
                    final_proposal.get("rejected_proposals"),
                    location=f"{location}.run.final_proposal.rejected_proposals",
                ),
                "final_novel_rows": _integer(
                    final_novelty.get("novel_rows"),
                    location=f"{location}.run.final_novelty.novel_rows",
                ),
                "run_seconds": _number(
                    resources.get("elapsed_seconds"),
                    location=f"{location}.runtime.run_resources.elapsed_seconds",
                ),
                "quality_acceptance": _json_cell(quality),
            }
        )
        for eligibility_index, raw_eligibility in enumerate(
            _list(
                run.get("candidate_eligibility"),
                location=f"{location}.run.candidate_eligibility",
            )
        ):
            candidate = _mapping(
                raw_eligibility,
                location=f"{location}.run.candidate_eligibility[{eligibility_index}]",
            )
            eligibility_rows.append(
                {
                    "cohort_id": cohort_id,
                    "profile": profile,
                    "route_tier": route_tier,
                    "candidate": candidate.get("candidate"),
                    "rows": candidate.get("rows"),
                    "templates": candidate.get("templates"),
                    "minimum_rows": candidate.get("minimum_rows"),
                    "minimum_templates": candidate.get("minimum_templates"),
                    "eligible": candidate.get("eligible"),
                    "reason": candidate.get("reason"),
                    "selected": candidate.get("candidate") == selected_candidate,
                }
            )
        runtime_runs: dict[tuple[str, int, int], Mapping[str, Any]] = {}
        for runtime_index, raw_runtime in enumerate(
            _list(
                runtime.get("benchmark_runs"),
                location=f"{location}.runtime.benchmark_runs",
            )
        ):
            runtime_row = _mapping(
                raw_runtime,
                location=f"{location}.runtime.benchmark_runs[{runtime_index}]",
            )
            runtime_row_required = {
                "candidate",
                "fold_index",
                "seed",
                "fit",
                "sample",
            }
            if set(runtime_row) != runtime_row_required:
                raise StructuredReportingError(
                    f"{location}.runtime.benchmark_runs[{runtime_index}] differs "
                    "from the benchmark runtime receipt contract"
                )
            runtime_key = (
                _text(
                    runtime_row["candidate"],
                    location=f"{location}.runtime.benchmark_runs[{runtime_index}].candidate",
                ),
                _integer(
                    runtime_row["fold_index"],
                    location=f"{location}.runtime.benchmark_runs[{runtime_index}].fold_index",
                ),
                _integer(
                    runtime_row["seed"],
                    location=f"{location}.runtime.benchmark_runs[{runtime_index}].seed",
                ),
            )
            if runtime_key in runtime_runs:
                raise StructuredReportingError(f"{location}.runtime repeats a candidate/fold/seed")
            runtime_runs[runtime_key] = runtime_row

        candidate_runs: dict[tuple[str, int, int], Mapping[str, Any]] = {}
        for run_index, raw_benchmark in enumerate(
            _list(run.get("benchmark_runs"), location=f"{location}.run.benchmark_runs")
        ):
            benchmark = _mapping(
                raw_benchmark, location=f"{location}.run.benchmark_runs[{run_index}]"
            )
            candidate_name = _text(
                benchmark.get("candidate"),
                location=f"{location}.run.benchmark_runs[{run_index}].candidate",
            )
            fold = _integer(
                benchmark.get("fold_index"),
                location=f"{location}.run.benchmark_runs[{run_index}].fold_index",
            )
            seed = _integer(
                benchmark.get("seed"),
                location=f"{location}.run.benchmark_runs[{run_index}].seed",
            )
            key = (candidate_name, fold, seed)
            if key in candidate_runs:
                raise StructuredReportingError(f"{location} repeats a candidate/fold/seed")
            benchmark_runtime = runtime_runs.get(key)
            if benchmark_runtime is None:
                raise StructuredReportingError(f"{location} lacks benchmark runtime for {key!r}")
            benchmark = {
                **benchmark,
                "fit": benchmark_runtime["fit"],
                "sample": benchmark_runtime["sample"],
            }
            candidate_runs[key] = benchmark
            evaluation = _mapping(
                benchmark.get("evaluation"), location=f"{location}.benchmark.evaluation"
            )
            diagnostic = _mapping(
                evaluation.get("diagnostic"), location=f"{location}.benchmark.diagnostic"
            )
            quality_report = _mapping(
                evaluation.get("quality"), location=f"{location}.benchmark.quality"
            )
            fit = _mapping(benchmark.get("fit"), location=f"{location}.benchmark.fit")
            sample = _mapping(benchmark.get("sample"), location=f"{location}.benchmark.sample")
            proposal = _mapping(
                benchmark.get("proposal"), location=f"{location}.benchmark.proposal"
            )
            validity = _mapping(
                benchmark.get("validity"), location=f"{location}.benchmark.validity"
            )
            novelty = _mapping(benchmark.get("novelty"), location=f"{location}.benchmark.novelty")
            benchmark_rows.append(
                {
                    "cohort_id": cohort_id,
                    "profile": profile,
                    "route_tier": route_tier,
                    "candidate": candidate_name,
                    "selected_candidate": selected_candidate,
                    "is_selected": candidate_name == selected_candidate,
                    "is_empirical": candidate_name == "empirical",
                    "fold": fold,
                    "seed": seed,
                    "train_rows": benchmark.get("train_rows"),
                    "validation_rows": benchmark.get("validation_rows"),
                    "train_templates": benchmark.get("train_templates"),
                    "validation_templates": benchmark.get("validation_templates"),
                    "quality_score": _number(
                        quality_report.get("score"), location=f"{location}.benchmark.quality.score"
                    ),
                    "diagnostic_score": _number(
                        diagnostic.get("score"),
                        location=f"{location}.benchmark.diagnostic.score",
                    ),
                    "fit_seconds": _number(
                        fit.get("elapsed_seconds"),
                        location=f"{location}.benchmark.fit.elapsed_seconds",
                    ),
                    "sample_seconds": _number(
                        sample.get("elapsed_seconds"),
                        location=f"{location}.benchmark.sample.elapsed_seconds",
                    ),
                    "raw_proposals": proposal.get("raw_proposals"),
                    "rejected_proposals": proposal.get("rejected_proposals"),
                    "acceptance_yield": _number(
                        proposal.get("acceptance_yield"),
                        location=f"{location}.benchmark.proposal.acceptance_yield",
                    ),
                    "valid_fraction": _number(
                        validity.get("valid_fraction"),
                        location=f"{location}.benchmark.validity.valid_fraction",
                    ),
                    "novel_fraction": _number(
                        novelty.get("novel_fraction"),
                        location=f"{location}.benchmark.novelty.novel_fraction",
                    ),
                    "peak_fit_rss_increase_mib": _number(
                        fit.get("peak_rss_increase_bytes"),
                        location=f"{location}.benchmark.fit.peak_rss_increase_bytes",
                    )
                    / (1024 * 1024),
                }
            )
        if set(candidate_runs) != set(runtime_runs):
            raise StructuredReportingError(
                f"{location} deterministic benchmark rows and runtime receipts differ"
            )
        pair_keys = {
            (fold, seed)
            for candidate, fold, seed in candidate_runs
            if candidate in {"empirical", selected_candidate}
        }
        for fold, seed in sorted(pair_keys):
            empirical = candidate_runs.get(("empirical", fold, seed))
            selected = candidate_runs.get((selected_candidate, fold, seed))
            if empirical is None or selected is None:
                raise StructuredReportingError(f"{location} lacks a paired SDMetrics run")

            def metrics(
                raw: Mapping[str, Any], *, metric_location: str
            ) -> tuple[float, float, float, float]:
                evaluation = _mapping(raw["evaluation"], location=f"{metric_location}.evaluation")
                quality_report = _mapping(
                    evaluation["quality"], location=f"{metric_location}.quality"
                )
                fit = _mapping(raw["fit"], location=f"{metric_location}.fit")
                sample = _mapping(raw["sample"], location=f"{metric_location}.sample")
                proposal = _mapping(raw["proposal"], location=f"{metric_location}.proposal")
                return (
                    _number(quality_report["score"], location=f"{metric_location}.score"),
                    _number(fit["elapsed_seconds"], location=f"{metric_location}.fit_seconds"),
                    _number(
                        sample["elapsed_seconds"],
                        location=f"{metric_location}.sample_seconds",
                    ),
                    _number(
                        proposal["acceptance_yield"],
                        location=f"{metric_location}.yield",
                    ),
                )

            empirical_metrics = metrics(empirical, metric_location=f"{location}.paired.empirical")
            selected_metrics = metrics(selected, metric_location=f"{location}.paired.selected")
            paired_rows.append(
                {
                    "cohort_id": cohort_id,
                    "profile": profile,
                    "route_tier": route_tier,
                    "selected_candidate": selected_candidate,
                    "fold": fold,
                    "seed": seed,
                    "empirical_quality": empirical_metrics[0],
                    "selected_quality": selected_metrics[0],
                    "quality_difference": selected_metrics[0] - empirical_metrics[0],
                    "empirical_fit_seconds": empirical_metrics[1],
                    "selected_fit_seconds": selected_metrics[1],
                    "empirical_sample_seconds": empirical_metrics[2],
                    "selected_sample_seconds": selected_metrics[2],
                    "empirical_acceptance_yield": empirical_metrics[3],
                    "selected_acceptance_yield": selected_metrics[3],
                }
            )
    return cohorts, eligibility_rows, benchmark_rows, paired_rows


def _publish_csv(
    stage: StagedArtifactRun,
    relative_path: str,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    stage.publish_bytes(relative_path, _csv_bytes(rows, columns))


def _plot_reports(
    *,
    stage: StagedArtifactRun,
    drivers: Sequence[Mapping[str, Any]],
    packages: Sequence[Mapping[str, Any]],
    measures: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    cohorts: Sequence[Mapping[str, Any]],
    benchmark: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    strata: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    changes: Sequence[Mapping[str, Any]],
    dates: Sequence[Mapping[str, Any]],
    capacity: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    pyplot, pandas, seaborn = _plot_runtime()
    paths: list[str] = []

    def publish(figure: Any, name: str) -> None:
        relative = f"plots/{name}"
        _publish_figure(stage=stage, pyplot=pyplot, figure=figure, relative_path=relative)
        paths.append(relative)

    driver_frame = pandas.DataFrame(drivers)
    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5))
    for column, label, color in (
        ("source_quantity", "source", "#4472C4"),
        ("generated_quantity", "generated", "#ED7D31"),
    ):
        seaborn.histplot(
            driver_frame[column].map(math.log1p),
            bins=20,
            stat="density",
            color=color,
            alpha=0.45,
            label=label,
            ax=axes[0],
        )
    axes[0].set(xlabel="log(1 + driver quantity)", title="Cargo-group driver quantities")
    axes[0].legend()
    axes[1].scatter(
        driver_frame["source_quantity"],
        driver_frame["generated_quantity"],
        alpha=0.65,
        s=28,
    )
    maximum = max(
        driver_frame["source_quantity"].max(),
        driver_frame["generated_quantity"].max(),
    )
    axes[1].plot([1, maximum], [1, maximum], linestyle="--", color="black", linewidth=1)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="source quantity",
        ylabel="generated quantity",
        title="Source versus generated driver",
    )
    figure.tight_layout()
    publish(figure, "01_driver_quantities.png")

    package_frame = pandas.DataFrame(packages)
    package_long = package_frame.melt(
        id_vars=["is_driver"],
        value_vars=["source_quantity", "generated_quantity"],
        var_name="population",
        value_name="quantity",
    )
    package_long["log_quantity"] = package_long["quantity"].map(math.log1p)
    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5))
    seaborn.histplot(
        data=package_long,
        x="log_quantity",
        hue="population",
        bins=20,
        stat="density",
        common_norm=False,
        element="step",
        ax=axes[0],
    )
    axes[0].set(
        xlabel="log(1 + package quantity)",
        title="All task-facing package levels",
    )
    role_counts = (
        package_frame.groupby(["driver_package_role", "is_driver"]).size().reset_index(name="rows")
    )
    seaborn.barplot(
        data=role_counts,
        x="rows",
        y="driver_package_role",
        hue="is_driver",
        orient="h",
        ax=axes[1],
    )
    axes[1].set(
        title="Driver package role context",
        xlabel="task-facing package rows",
        ylabel="",
    )
    figure.tight_layout()
    publish(figure, "02_all_package_quantities.png")

    if measures:
        measure_frame = pandas.DataFrame(measures)
        measure_long = measure_frame.melt(
            id_vars=["measure", "unit"],
            value_vars=["source_value", "generated_value"],
            var_name="population",
            value_name="value",
        )
        measure_long["log_value"] = measure_long["value"].map(math.log1p)
        names = list(dict.fromkeys(measure_long["measure"]))
        figure, axes = pyplot.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
        for axis, name in zip(axes[0], names, strict=True):
            subset = measure_long[measure_long["measure"] == name]
            seaborn.histplot(
                data=subset,
                x="log_value",
                hue="population",
                bins=20,
                stat="density",
                common_norm=False,
                element="step",
                ax=axis,
            )
            units = ", ".join(sorted(set(subset["unit"])))
            axis.set(
                xlabel="log(1 + total)",
                title=f"{name.replace('_', ' ').title()} ({units})",
            )
        figure.tight_layout()
        publish(figure, "03_cargo_measure_totals.png")

    route_frame = pandas.DataFrame(routes)
    figure, axes = pyplot.subplots(1, 2, figsize=(13, 5))
    route_counts = (
        route_frame["route_tier"]
        .value_counts()
        .rename_axis("route_tier")
        .reset_index(name="requests")
    )
    seaborn.barplot(data=route_counts, x="requests", y="route_tier", orient="h", ax=axes[0])
    axes[0].set(title="Profile route tiers", xlabel="cargo-group requests", ylabel="")
    seaborn.histplot(data=route_frame, x="acceptance_yield", bins=20, ax=axes[1])
    axes[1].set(xlim=(0, 1), title="Assigned proposal acceptance yield")
    figure.tight_layout()
    publish(figure, "04_route_tiers_and_yield.png")

    cohort_frame = pandas.DataFrame(cohorts)
    figure, axes = pyplot.subplots(1, 2, figsize=(14, 6))
    candidate_counts = (
        cohort_frame["selected_candidate"]
        .value_counts()
        .rename_axis("candidate")
        .reset_index(name="cohorts")
    )
    seaborn.barplot(data=candidate_counts, x="cohorts", y="candidate", orient="h", ax=axes[0])
    axes[0].set(title="Selected SDV candidate per cohort", xlabel="cohorts", ylabel="")
    displayed = cohort_frame.nlargest(30, "request_count").sort_values("request_count")
    seaborn.barplot(
        data=displayed,
        x="request_count",
        y="cohort_id",
        hue="selected_candidate",
        dodge=False,
        ax=axes[1],
    )
    axes[1].set(
        title="Largest profile cohorts (top 30; full CSV published)",
        xlabel="assigned cargo-group requests",
        ylabel="",
    )
    figure.tight_layout()
    publish(figure, "05_profile_cohorts.png")

    benchmark_frame = pandas.DataFrame(benchmark)
    paired_frame = pandas.DataFrame(paired)
    figure, axes = pyplot.subplots(1, 3, figsize=(19, 6))
    seaborn.boxplot(data=benchmark_frame, x="candidate", y="quality_score", ax=axes[0])
    seaborn.stripplot(
        data=benchmark_frame,
        x="candidate",
        y="quality_score",
        color="black",
        alpha=0.4,
        size=2,
        ax=axes[0],
    )
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set(title="Grouped-fold SDMetrics quality", xlabel="", ylabel="quality")
    paired_long = paired_frame.melt(
        id_vars=["cohort_id", "fold", "seed"],
        value_vars=["empirical_quality", "selected_quality"],
        var_name="candidate_role",
        value_name="quality",
    )
    seaborn.boxplot(data=paired_long, x="candidate_role", y="quality", ax=axes[1])
    axes[1].set(title="Paired empirical versus selected quality", xlabel="", ylabel="quality")
    runtime = benchmark_frame.groupby("candidate", as_index=False)[
        ["fit_seconds", "sample_seconds"]
    ].mean()
    runtime_long = runtime.melt(id_vars="candidate", var_name="phase", value_name="mean_seconds")
    seaborn.barplot(
        data=runtime_long,
        x="candidate",
        y="mean_seconds",
        hue="phase",
        ax=axes[2],
    )
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set(title="Mean runtime per fold/seed", xlabel="", ylabel="seconds")
    figure.tight_layout()
    publish(figure, "06_sdv_quality_and_runtime.png")

    figure, axes = pyplot.subplots(1, 2, figsize=(14, 5))
    seaborn.histplot(data=paired_frame, x="quality_difference", bins=20, ax=axes[0])
    axes[0].axvline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set(title="Selected minus empirical paired quality")
    yield_long = paired_frame.melt(
        id_vars=["cohort_id", "fold", "seed"],
        value_vars=["empirical_acceptance_yield", "selected_acceptance_yield"],
        var_name="candidate_role",
        value_name="acceptance_yield",
    )
    seaborn.boxplot(data=yield_long, x="candidate_role", y="acceptance_yield", ax=axes[1])
    axes[1].set(title="Paired bounded-sampling yield", xlabel="", ylabel="yield")
    figure.tight_layout()
    publish(figure, "07_sdv_paired_quality_and_yield.png")

    strata_frame = pandas.DataFrame(strata)
    dimensions = list(dict.fromkeys(strata_frame["dimension"]))
    figure, axes = pyplot.subplots(2, 2, figsize=(14, 10))
    for axis, dimension in zip(axes.flat, dimensions, strict=True):
        counts = (
            strata_frame[strata_frame["dimension"] == dimension]["value"]
            .value_counts()
            .head(30)
            .rename_axis("value")
            .reset_index(name="documents")
        )
        seaborn.barplot(data=counts, x="documents", y="value", orient="h", ax=axis)
        axis.set(
            title=f"{dimension.replace('_', ' ').title()} (top 30)",
            xlabel="documents",
            ylabel="",
        )
    figure.tight_layout()
    publish(figure, "08_selection_strata.png")

    context_frame = pandas.DataFrame(contexts)
    figure, axis = pyplot.subplots(figsize=(11, 6))
    if context_frame.empty:
        axis.text(
            0.5,
            0.5,
            "No hard-context flags in this selection",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
    else:
        context_counts = (
            context_frame["context"]
            .value_counts()
            .head(30)
            .rename_axis("context")
            .reset_index(name="documents")
        )
        seaborn.barplot(data=context_counts, x="documents", y="context", orient="h", ax=axis)
        axis.set(
            title="Selected hard contexts (top 30; full CSV published)",
            xlabel="documents",
            ylabel="",
        )
    figure.tight_layout()
    publish(figure, "09_selection_contexts.png")

    change_frame = pandas.DataFrame(changes)
    figure, axis = pyplot.subplots(figsize=(11, 6))
    change_counts = (
        change_frame["family"]
        .value_counts()
        .head(30)
        .rename_axis("family")
        .reset_index(name="changed_leaves")
    )
    seaborn.barplot(data=change_counts, x="changed_leaves", y="family", orient="h", ax=axis)
    axis.set(
        title="Semantic change families (top 30; full CSV published)",
        xlabel="changed leaves",
        ylabel="",
    )
    figure.tight_layout()
    publish(figure, "10_semantic_changes.png")

    if dates:
        date_frame = pandas.DataFrame(dates)
        figure, axes = pyplot.subplots(1, 2, figsize=(13, 5))
        year_frame = date_frame[["source_year", "generated_year"]].melt(
            var_name="population", value_name="year"
        )
        seaborn.countplot(data=year_frame, x="year", hue="population", ax=axes[0])
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set(title="Source and generated date years")
        seaborn.histplot(data=date_frame, x="day_shift", bins=20, ax=axes[1])
        axes[1].set(title="Applied joint date shifts")
        figure.tight_layout()
        publish(figure, "11_date_changes.png")
    capacity_frame = pandas.DataFrame(capacity)
    utilization = (
        capacity_frame[
            [
                "gross_payload_utilization",
                "net_payload_utilization",
                "volume_utilization",
            ]
        ]
        .melt(var_name="dimension", value_name="utilization")
        .dropna()
    )
    figure, axes = pyplot.subplots(1, 2, figsize=(14, 5))
    if utilization.empty:
        axes[0].text(0.5, 0.5, "No constrained cargo measures", ha="center", va="center")
        axes[0].set_axis_off()
    else:
        seaborn.histplot(
            data=utilization,
            x="utilization",
            hue="dimension",
            bins=20,
            element="step",
            common_norm=False,
            ax=axes[0],
        )
        axes[0].axvline(1, color="red", linestyle="--", linewidth=1)
        axes[0].set(
            xlim=(0, max(1.05, float(utilization["utilization"].max()) * 1.05)),
            title="Generated document capacity utilization",
            xlabel="fraction of configured capacity",
        )
    family_counts: dict[str, int] = {}
    for values in capacity_frame["equipment_families"]:
        for family in values:
            family_counts[str(family)] = family_counts.get(str(family), 0) + 1
    family_frame = pandas.DataFrame(
        [
            {"equipment_family": family, "containers": count}
            for family, count in sorted(family_counts.items())
        ]
    )
    if family_frame.empty:
        axes[1].text(0.5, 0.5, "No containers selected", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        seaborn.barplot(
            data=family_frame,
            x="containers",
            y="equipment_family",
            orient="h",
            ax=axes[1],
        )
        axes[1].set(title="Capacity equipment classifications", ylabel="")
    figure.tight_layout()
    publish(figure, "12_transport_capacity.png")

    context_frame = route_frame.copy()
    context_frame["distance_fraction"] = context_frame.apply(
        lambda row: (
            0.0
            if float(row["contextual_maximum_distance"]) == 0
            else float(row["contextual_distance"]) / float(row["contextual_maximum_distance"])
        ),
        axis=1,
    )
    figure, axes = pyplot.subplots(1, 2, figsize=(14, 5))
    support_counts = (
        context_frame["contextual_support_tier"]
        .value_counts()
        .rename_axis("support_tier")
        .reset_index(name="requests")
    )
    seaborn.barplot(
        data=support_counts,
        x="requests",
        y="support_tier",
        orient="h",
        ax=axes[0],
    )
    axes[0].set(title="Package-aware plausibility support", xlabel="cargo groups", ylabel="")
    seaborn.histplot(data=context_frame, x="distance_fraction", bins=20, ax=axes[1])
    axes[1].axvline(1, color="red", linestyle="--", linewidth=1)
    axes[1].set(
        xlim=(0, 1.05),
        title="Distance within observed support neighborhood",
        xlabel="nearest-support distance / maximum accepted distance",
    )
    figure.tight_layout()
    publish(figure, "13_package_identity_plausibility.png")
    return tuple(paths)


def _report_markdown(
    *,
    run_id: str,
    distribution: Mapping[str, Any],
    modeling_audit: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    packages: Sequence[Mapping[str, Any]],
    measures: Sequence[Mapping[str, Any]],
    cohorts: Sequence[Mapping[str, Any]],
    changes: Sequence[Mapping[str, Any]],
    plot_paths: Sequence[str],
) -> str:
    route_counts = Counter(str(row["route_tier"]) for row in proposals)
    contextual_counts = Counter(str(row["contextual_support_tier"]) for row in proposals)
    candidate_counts = Counter(str(row["selected_candidate"]) for row in cohorts)
    measure_counts = Counter(str(row["measure"]) for row in measures)
    package_changes = sum(bool(row["changed"]) for row in packages)
    context_counts = cast(Mapping[str, Any], distribution.get("contexts", {}))
    context_lines = (
        "\n".join(f"- `{name}`: {count}" for name, count in sorted(context_counts.items()))
        or "- No hard-context flags were present in this selected baseline."
    )
    route_lines = "\n".join(
        f"- `{name}`: {count} cargo-group requests" for name, count in sorted(route_counts.items())
    )
    contextual_lines = "\n".join(
        f"- `{name}`: {count} cargo-group requests"
        for name, count in sorted(contextual_counts.items())
    )
    candidate_lines = "\n".join(
        f"- `{name}`: {count} cohorts" for name, count in sorted(candidate_counts.items())
    )
    measure_lines = (
        "\n".join(
            f"- `{name}`: {count} source/generated pairs"
            for name, count in sorted(measure_counts.items())
        )
        or "- No optional gross, net, or volume totals were present in the selected groups."
    )
    plot_lines = "\n".join(f"- [{path.rsplit('/', 1)[-1]}]({path})" for path in plot_paths)
    pending_metadata = distribution.get("pending_metadata_only_package_facts", "not reported")
    metadata_line = (
        f"- Metadata-only package hierarchy facts pending later realization: **{pending_metadata}**"
    )
    train_documents = modeling_audit.get("train_document_count", "not reported")
    cargo_groups = modeling_audit.get("cargo_group_numeric_rows", "not reported")
    cargo_packages = modeling_audit.get("cargo_package_numeric_rows", "not reported")
    hierarchy_rows = modeling_audit.get("package_hierarchy_rows", "not reported")
    metadata_rows = modeling_audit.get("metadata_only_package_rows", "not reported")
    capacity_excluded = modeling_audit.get(
        "transport_capacity_excluded_fit_document_count", "not reported"
    )
    capacity = cast(Mapping[str, Any], distribution["transport_capacity"])
    gross_utilization = cast(Mapping[str, Any], capacity["gross_payload_utilization"])
    volume_utilization = cast(Mapping[str, Any], capacity["volume_utilization"])
    capacity_valid = capacity["validated_documents"]
    selected_documents = distribution["selected_documents"]
    containerized = capacity["containerized_documents"]
    non_containerized = capacity["non_containerized_documents"]
    return f"""# Structured synthesis baseline: `{run_id}`

This is a **non-publishable structured baseline**. It tests structured, non-linguistic
generation and still contains source party/cargo language and source-format OCR text. It must not
enter model training until mandatory anonymization and text realization are complete.

## Exact scope

- Selected synthetic scenarios: **{distribution["selected_documents"]}**
- Profile-routed cargo groups: **{len(proposals)}**
- Profile cohorts benchmarked independently: **{len(cohorts)}**
- Task-facing package levels represented: **{len(packages)}**
- Task-facing package quantities changed: **{package_changes} / {len(packages)}**
- Exact semantic change-ledger rows: **{len(changes)}**
{metadata_line}
- Training records published: **0**
- Transport-capacity-valid scenarios: **{capacity_valid} / {selected_documents}**
- Containerized / non-containerized: **{containerized} / {non_containerized}**
- Maximum gross-payload utilization: **{gross_utilization.get("maximum", "not present")}**
- Maximum enclosed-volume utilization: **{volume_utilization.get("maximum", "not present")}**

Package categories, printed package identities, package roles, and package topology are
**preserved from the reviewed source label**. They are not generated or statistically modeled in
this baseline. Quantity changes are proposed once at cargo-group driver level; all task-facing
package quantities are then derived coherently, and allocation quantities are reconciled. Outer or
aggregate hierarchy facts retained only as metadata remain pending and are not claimed as realized.

## Profile-routed statistical generation

Each hard gross/net/volume missingness profile is routed only to a support-eligible train cohort.
That cohort is benchmarked independently using template-grouped folds. The empirical candidate is
a paired comparison baseline and is not production-selectable. The selected non-empirical
candidate must pass the configured paired SDMetrics quality, domain-validity, novelty, and bounded
proposal-yield gates. These results establish quality for the **reported numeric cohorts only**;
they do not establish coverage for package categories, language, parties, or unseen profiles.

Route tiers:

{route_lines}

Package-aware acceptance tiers (role-wide support is never allowed here):

{contextual_lines}

Selected candidates:

{candidate_lines}

Generated optional totals when present:

{measure_lines}

## Modeling and selection evidence

- Train documents in the modeling-view audit: **{train_documents}**
- Audited cargo-group numeric rows: **{cargo_groups}**
- Audited task-facing package numeric rows: **{cargo_packages}**
- Audited package hierarchy rows: **{hierarchy_rows}**
- Audited metadata-only package rows: **{metadata_rows}**
- Source documents excluded from numeric fit by transport-capacity audit: **{capacity_excluded}**

Full, untruncated CSV tables are published under `data/`. Top-N limits are used only where needed
to keep plots legible; every plot title that is truncated says so explicitly.

## Selected hard-context coverage

{context_lines}

## Published plots

{plot_lines}
"""


def publish_structured_baseline_report(
    *,
    stage: StagedArtifactRun,
    run_id: str,
    selected_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    change_rows: Sequence[Sequence[Mapping[str, Any]]],
    distribution: Mapping[str, Any],
    profile_results: Sequence[Mapping[str, Any]],
    modeling_audit: Mapping[str, Any],
    capacity_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Publish complete machine-readable evidence and deterministic diagnostics."""

    scenario_count = len(selected_rows)
    if not scenario_count or not proposal_rows or not profile_results:
        raise StructuredReportingError(
            "structured reporting requires selected scenarios, proposals, and profiles"
        )
    if scenario_count != len(targets) or scenario_count != len(change_rows):
        raise StructuredReportingError("report inputs do not describe the same scenarios")
    if scenario_count != len(capacity_rows) or any(
        row.get("valid") is not True for row in capacity_rows
    ):
        raise StructuredReportingError("report requires one valid capacity receipt per scenario")
    reported_count = distribution.get("selected_documents")
    if reported_count != scenario_count:
        raise StructuredReportingError(
            "distribution selected-document count differs from report inputs"
        )
    sources, strata, contexts = _selection_tables(selected_rows)
    proposals, drivers, packages, measures, routes = _proposal_tables(proposal_rows)
    changes, dates = _change_tables(change_rows)
    cohorts, eligibility, benchmark, paired = _profile_tables(profile_results)
    proposal_requests = {str(row["request_id"]) for row in proposals}
    cohort_requests = {
        request_id
        for row in cohorts
        for request_id in str(row["request_ids"]).split("|")
        if request_id
    }
    if proposal_requests != cohort_requests:
        raise StructuredReportingError(
            "profile cohort requests and published cargo proposals differ"
        )
    selected_candidate_by_cohort = {
        str(row["cohort_id"]): str(row["selected_candidate"]) for row in cohorts
    }
    for row in proposals:
        cohort_id = str(row["cohort_id"])
        if selected_candidate_by_cohort.get(cohort_id) != row["selected_candidate"]:
            raise StructuredReportingError(
                "proposal selected candidate differs from its cohort receipt"
            )

    _publish_csv(stage, "data/selected-sources.csv", sources, tuple(sources[0]))
    _publish_csv(
        stage,
        "data/selection-strata.csv",
        strata,
        ("position", "document_id", "dimension", "value"),
    )
    _publish_csv(
        stage,
        "data/selection-contexts.csv",
        contexts,
        ("position", "document_id", "context"),
    )
    _publish_csv(stage, "data/cargo-group-proposals.csv", proposals, tuple(proposals[0]))
    _publish_csv(stage, "data/driver-quantities.csv", drivers, tuple(drivers[0]))
    _publish_csv(stage, "data/package-quantities.csv", packages, tuple(packages[0]))
    _publish_csv(
        stage,
        "data/cargo-measure-totals.csv",
        measures,
        (
            "proposal_id",
            "synthetic_position",
            "base_document_id",
            "request_id",
            "cohort_id",
            "group_id",
            "package_identity",
            "package_role",
            "route_tier",
            "selected_candidate",
            "measure",
            "unit",
            "source_value",
            "generated_value",
            "changed",
        ),
    )
    _publish_csv(stage, "data/profile-routes.csv", routes, tuple(routes[0]))
    _publish_csv(stage, "data/profile-cohorts.csv", cohorts, tuple(cohorts[0]))
    _publish_csv(
        stage,
        "data/profile-candidate-eligibility.csv",
        eligibility,
        tuple(eligibility[0]),
    )
    _publish_csv(stage, "data/sdv-benchmark-scorecard.csv", benchmark, tuple(benchmark[0]))
    _publish_csv(stage, "data/sdv-paired-quality.csv", paired, tuple(paired[0]))
    _publish_csv(
        stage,
        "data/semantic-changes.csv",
        changes,
        (
            "synthetic_position",
            "target_path",
            "family",
            "method",
            "coupling_group",
            "old_value",
            "new_value",
        ),
    )
    _publish_csv(
        stage,
        "data/date-changes.csv",
        dates,
        (
            "synthetic_position",
            "target_path",
            "method",
            "coupling_group",
            "source_date",
            "generated_date",
            "source_year",
            "generated_year",
            "source_month",
            "generated_month",
            "day_shift",
        ),
    )
    _publish_csv(
        stage,
        "data/transport-capacity.csv",
        capacity_rows,
        tuple(capacity_rows[0]),
    )
    plot_paths = _plot_reports(
        stage=stage,
        drivers=drivers,
        packages=packages,
        measures=measures,
        routes=routes,
        cohorts=cohorts,
        benchmark=benchmark,
        paired=paired,
        strata=strata,
        contexts=contexts,
        changes=changes,
        dates=dates,
        capacity=capacity_rows,
    )
    report = _report_markdown(
        run_id=run_id,
        distribution=distribution,
        modeling_audit=modeling_audit,
        proposals=proposals,
        packages=packages,
        measures=measures,
        cohorts=cohorts,
        changes=changes,
        plot_paths=plot_paths,
    )
    stage.publish_bytes("REPORT.md", report.encode("utf-8"))
