#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the end-to-end EDA and quality audit for the additional 100 B/L scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]

RUNS = {
    "baseline": "artifacts/kie-synthesis/mpci-bl-combined2174-structured-baseline100-additional-v1",
    "route_v1": "artifacts/kie-synthesis/mpci-bl-combined2174-route-scenario100-additional-v1",
    "route_v2": "artifacts/kie-synthesis/mpci-bl-combined2174-route-scenario100-additional-maritime-v2",
    "controlled": "artifacts/kie-synthesis/mpci-bl-combined2174-controlled-semantic100-additional-v1",
    "semantic": "artifacts/kie-synthesis/mpci-bl-combined2174-semantic-plan100-additional-v1",
    "dangerous_goods": "artifacts/kie-synthesis/mpci-bl-combined2174-dangerous-goods-plan100-additional-v1",
    "package": "artifacts/kie-synthesis/mpci-bl-combined2174-package-compatibility100-additional-v1",
    "completion": "artifacts/kie-synthesis/mpci-bl-combined2174-semantic-completion100-additional-v1",
    "linguistic_v1": "artifacts/kie-synthesis/mpci-bl-combined2174-linguistic-completion100-additional-luna-high-v1",
    "linguistic_v2": "artifacts/kie-synthesis/mpci-bl-combined2174-linguistic-completion100-additional-luna-high-v2-resume",
    "compiler": "artifacts/kie-synthesis/mpci-bl-raw-text-inventory100-additional-glm53-v1",
    "cert10": "artifacts/kie-synthesis/mpci-bl-raw-text-certification10-additional100-glm53-v1",
    "cert50": "artifacts/kie-synthesis/mpci-bl-raw-text-certification50-additional100-glm53-v1",
    "prior_compiler": "artifacts/kie-synthesis/mpci-bl-raw-text-inventory50-glm53-clean-cost-v1",
    "prior_publication": "artifacts/kie-synthesis/mpci-bl-raw-text-certified-publication50-v1",
}

SOURCE_DATASET = (
    "artifacts/kie-training/datasets/"
    "mpci-bl-combined2174-task-facing-package-categories-v1/records.jsonl"
)
UNLOCODE = "artifacts/kie-synthesis/registries/unlocode-2025-1-all-functions-v1/locations.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save(path: Path) -> None:
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()


def _numeric_summary(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="raise")
    return {
        "count": int(values.count()),
        "minimum": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "maximum": float(values.max()),
    }


def _patch(row: Mapping[str, Any]) -> Mapping[str, Any]:
    target = row.get("target", row)
    if not isinstance(target, Mapping):
        raise ValueError("target is not an object")
    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("target has no documentPatch")
    return patch


def _as_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _party_presence(parties: Mapping[str, Any], role: str) -> int:
    value = parties.get(role)
    if isinstance(value, Mapping):
        return 1
    return int(bool(_as_rows(value)))


def _target_features(target_row: Mapping[str, Any]) -> dict[str, Any]:
    patch = _patch(target_row)
    parties = patch.get("parties") if isinstance(patch.get("parties"), Mapping) else {}
    cargo = _as_rows(patch.get("cargoGroups"))
    packages = _as_rows(patch.get("cargoPackages"))
    containers = _as_rows(patch.get("containers"))
    hs_codes = [str(code) for row in cargo for code in row.get("hsCodes", [])]
    dangerous_goods = [dg for row in cargo for dg in _as_rows(row.get("dangerousGoods"))]
    temperatures = [
        row.get("temperatureSetpoint")
        for row in containers
        if isinstance(row.get("temperatureSetpoint"), Mapping)
    ]
    return {
        "target_containers": len(containers),
        "target_cargo_groups": len(cargo),
        "target_packages": len(packages),
        "target_allocations": sum(
            len(_as_rows(row.get("allocations")))
            for row in _as_rows(patch.get("cargoAllocationGroups"))
        ),
        "target_hs_codes": len(hs_codes),
        "target_dangerous_goods": len(dangerous_goods),
        "target_temperature_setpoints": len(temperatures),
        "target_imo_present": int(
            isinstance(patch.get("transport"), Mapping)
            and bool(patch["transport"].get("vesselImoNumber"))
        ),
        "target_flag_present": int(
            isinstance(patch.get("transport"), Mapping)
            and bool(patch["transport"].get("vesselFlagCountry"))
        ),
        "target_notify_parties": len(_as_rows(parties.get("notifyParties"))),
        "target_shipper": _party_presence(parties, "shipper"),
        "target_consignee": _party_presence(parties, "consignee"),
        "target_carrier": _party_presence(parties, "carrier"),
        "target_delivery_agent": _party_presence(parties, "deliveryAgent"),
        "target_forwarding_agent": _party_presence(parties, "forwardingAgent"),
        "equipment_types": "|".join(
            sorted(str(row.get("typeCategory")) for row in containers if row.get("typeCategory"))
        ),
        "equipment_sizes": "|".join(
            sorted(str(row.get("sizeCategory")) for row in containers if row.get("sizeCategory"))
        ),
        "package_categories": "|".join(
            sorted(str(row.get("typeCategory")) for row in packages if row.get("typeCategory"))
        ),
        "hs_lengths": "|".join(sorted(str(len(value)) for value in hs_codes)),
    }


def _line_metrics(case: Path) -> dict[str, Any]:
    source = (case / "source.txt").read_text(encoding="utf-8")
    final = (case / "final.txt").read_text(encoding="utf-8")
    before = source.splitlines()
    after = final.splitlines()
    same_topology = len(before) == len(after)
    changed = sum(a != b for a, b in zip(before, after, strict=False))
    page_before = [line for line in before if line.startswith("--- PAGE ")]
    page_after = [line for line in after if line.startswith("--- PAGE ")]
    return {
        "source_lines": len(before),
        "final_lines": len(after),
        "line_topology_preserved": same_topology,
        "page_topology_preserved": page_before == page_after,
        "changed_lines": changed,
        "changed_line_fraction": changed / len(before) if before else 0.0,
        "source_characters": len(source),
        "final_characters": len(final),
    }


def _raw_failure_category(row: Mapping[str, Any]) -> str:
    status = str(row["status"])
    reason = str(row.get("reason") or "")
    error = str(row.get("compilerErrorMessage") or "")
    if status == "compiler_blocked":
        if "jurisdiction-bound" in error:
            return "preflight: unresolved customs jurisdiction"
        if "positive container allocation" in error:
            return "preflight: incomplete weight allocation"
        if "source HS surface" in error:
            return "preflight: source HS absent from OCR"
        return "preflight: other"
    if "Every configured provider route failed" in reason:
        return "provider routes exhausted"
    if "role-bound party values" in reason:
        return "atomic apply: party/carrier occurrence"
    if "omits changed target text" in reason:
        return "atomic apply: target literal omitted"
    if "full-document audit" in reason:
        return "deterministic full-document audit"
    if "host-locked literals" in reason:
        return "atomic apply: locked literal omitted"
    if "path-owned rendered surfaces" in reason:
        return "atomic apply: rendered surface omitted"
    if "raw auxiliary identity" in reason:
        return "atomic apply: carrier principal mismatch"
    if "unchanged source status" in reason:
        return "atomic apply: unchanged status altered"
    if "punctuation or whitespace" in reason:
        return "atomic apply: punctuation drift"
    return "other"


def _accepted_findings(stages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    accepted = [
        stage
        for stage in stages
        if stage.get("modelOutput") is not None and stage.get("hostError") is None
    ]
    if not accepted:
        return []
    output = accepted[-1]["modelOutput"]
    if not isinstance(output, Mapping):
        return []
    return _as_rows(output.get("findings"))


def _certification_data(
    roots: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        summary = _read_json(root / "summary.json")
        for case in sorted((root / "cases").iterdir()):
            result = _read_json(case / "result.json")
            document_id = str(result["documentId"])
            if document_id in seen:
                raise ValueError(f"certification document repeated: {document_id}")
            seen.add(document_id)
            stages = json.loads((case / "stages.json").read_text(encoding="utf-8"))
            if not isinstance(stages, list):
                raise ValueError(f"invalid stages: {case}")
            selected = _accepted_findings(stages)
            if len(selected) != int(result["semanticFindings"]):
                raise ValueError(f"final finding count mismatch: {document_id}")
            usage = result["usage"]
            cases.append(
                {
                    "document_id": document_id,
                    "certification_run": summary["runId"],
                    "certification_status": result["status"],
                    "certification_findings": result["semanticFindings"],
                    "certification_requests": usage["requests"],
                    "certification_input_tokens": usage["inputTokens"],
                    "certification_reasoning_tokens": usage["reasoningTokens"],
                    "certification_visible_tokens": usage["visibleOutputTokens"],
                    "certification_cost_usd": float(usage["providerReportedCostUsd"]),
                }
            )
            source_lines = (case / "source.txt").read_text(encoding="utf-8").splitlines()
            final_lines = (case / "final.txt").read_text(encoding="utf-8").splitlines()
            for finding_number, finding in enumerate(selected, start=1):
                evidence = _as_rows(finding.get("evidence"))
                evidence_rows = []
                exact = True
                unchanged = []
                for item in evidence:
                    line_id = str(item.get("lineId"))
                    number = int(line_id.removeprefix("L"))
                    fragment = str(item.get("currentFragment"))
                    if number < 1 or number > len(final_lines):
                        exact = False
                        continue
                    exact = exact and bool(fragment) and fragment in final_lines[number - 1]
                    unchanged.append(
                        number <= len(source_lines)
                        and source_lines[number - 1] == final_lines[number - 1]
                    )
                    evidence_rows.append(f"{line_id}: {fragment}")
                findings.append(
                    {
                        "document_id": document_id,
                        "finding_number": finding_number,
                        "finding_kind": finding.get("findingKind"),
                        "problem": finding.get("problem"),
                        "evidence": " || ".join(evidence_rows),
                        "evidence_exact": exact,
                        "any_evidence_line_unchanged": any(unchanged),
                        "all_evidence_lines_unchanged": bool(unchanged) and all(unchanged),
                    }
                )
            for stage in stages:
                usage = stage.get("usage") if isinstance(stage.get("usage"), Mapping) else {}
                attempts.append(
                    {
                        "document_id": document_id,
                        "run_id": summary["runId"],
                        "provider_route": stage.get("routeProvider"),
                        "error_type": stage.get("errorType"),
                        "host_error": stage.get("hostError"),
                        "billed_requests": int(usage.get("requests", 0)),
                        "input_tokens": int(usage.get("inputTokens", 0)),
                        "reasoning_tokens": int(usage.get("reasoningTokens", 0)),
                        "visible_tokens": int(usage.get("visibleOutputTokens", 0)),
                        "provider_cost_usd": float(usage.get("providerReportedCostUsd", 0) or 0),
                    }
                )
    return cases, findings, attempts


def _manual_certified_audit(cert_roots: Sequence[Path]) -> list[dict[str, Any]]:
    by_id = {
        case.name: case
        for root in cert_roots
        for case in (root / "cases").iterdir()
        if case.is_dir()
    }
    declared = {
        "doc_56e593aee2fe694e60f1049d8ede69ba55418e197f1a3a8d8d436c333ecc1815": (
            "NORWIND PROVINCE",
            "Fictional UK province inserted into an otherwise grounded delivery-agent address; "
            "the result is not realistic enough for the high-quality publication contract.",
        ),
        "doc_88ec16af95fce3eaea2950bc205a2820b5b289c64c12a4d32835b1a8d3340375": (
            "PFI# 963",
            "Source shipment reference survives unchanged; the same candidate also gives a "
            "Chinese delivery agent a Romanian +40 fax prefix.",
        ),
        "doc_f0f120de2654d5d6b09ff02003a3f53de38919d80d7707fcdb146d7b4d9dd948": (
            "Tare Weight: 4,690.000 kgs.",
            "Source operational tare values survive unchanged across the synthetic reefer blocks.",
        ),
    }
    rows: list[dict[str, Any]] = []
    for document_id, (needle, issue) in declared.items():
        case = by_id[document_id]
        source = (case / "source.txt").read_text(encoding="utf-8")
        final = (case / "final.txt").read_text(encoding="utf-8")
        if needle not in final:
            raise ValueError(f"manual-audit evidence missing: {document_id}: {needle}")
        rows.append(
            {
                "document_id": document_id,
                "model_certification_status": "certified",
                "manual_high_quality_status": "reject",
                "evidence": needle,
                "evidence_was_in_source": needle in source,
                "issue": issue,
            }
        )
    return rows


def _counter_rows(counter: Counter[str], *, name: str, count: str) -> list[dict[str, Any]]:
    return [{name: key, count: value} for key, value in counter.most_common()]


def _plot_bar(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str,
    y: str,
    title: str,
    path: Path,
    horizontal: bool = False,
) -> None:
    frame = pd.DataFrame(rows)
    plt.figure(figsize=(11, max(5.5, min(12, len(frame) * 0.38))))
    if horizontal:
        sns.barplot(data=frame, x=x, y=y, hue=y, legend=False)
    else:
        sns.barplot(data=frame, x=x, y=y, hue=x, legend=False)
        plt.xticks(rotation=30, ha="right")
    plt.title(title)
    _save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve(strict=True)
    output = (project / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"analysis output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "data"
    plots = output / "plots"
    data_dir.mkdir()
    plots.mkdir()
    roots = {name: (project / value).resolve(strict=True) for name, value in RUNS.items()}
    for name, root in roots.items():
        if name == "prior_publication":
            continue
        if not (root / "_COMMIT.json").is_file():
            raise ValueError(f"source run is not committed: {root}")

    selected = _read_jsonl(roots["baseline"] / "selection" / "selected-sources.jsonl")
    if len(selected) != 100 or len({row["document_id"] for row in selected}) != 100:
        raise ValueError("baseline selection is not exactly 100 unique documents")
    selected_by_id = {str(row["document_id"]): row for row in selected}
    source_records = {str(row["documentId"]): row for row in _read_jsonl(project / SOURCE_DATASET)}
    if not set(selected_by_id).issubset(source_records):
        raise ValueError("selected source documents are absent from the pinned task dataset")
    linguistic = _read_jsonl(roots["linguistic_v2"] / "generation" / "results.jsonl")
    linguistic_by_id = {str(row["baseDocumentId"]): row for row in linguistic}
    if set(linguistic_by_id) != set(selected_by_id):
        raise ValueError("linguistic output differs from selected source set")
    route_v1_rows = _read_jsonl(roots["route_v1"] / "generation" / "shipment-scenarios.jsonl")
    route_v2_rows = _read_jsonl(roots["route_v2"] / "generation" / "shipment-scenarios.jsonl")
    route_v1 = {str(row["baseDocumentId"]): row for row in route_v1_rows}
    route_v2 = {str(row["baseDocumentId"]): row for row in route_v2_rows}
    locodes = {str(row["locode"]): row for row in _read_jsonl(project / UNLOCODE)}

    route_defects: list[dict[str, Any]] = []
    route_changed: set[str] = set()
    for document_id in selected_by_id:
        old = route_v1[document_id]
        new = route_v2[document_id]
        for field in ("loadingPort", "dischargePort"):
            old_port = old[field]
            new_port = new[field]
            if old_port != new_port:
                route_changed.add(document_id)
            registry = locodes[str(old_port["locode"])]
            if "1" not in registry["function_codes"]:
                route_defects.append(
                    {
                        "document_id": document_id,
                        "endpoint": field,
                        "old_locode": old_port["locode"],
                        "old_name": old_port["name"],
                        "unlocode_functions": "|".join(registry["function_codes"]),
                        "corrected_locode": new_port["locode"],
                        "corrected_name": new_port["name"],
                    }
                )
    for row in route_v2_rows:
        for field in ("loadingPort", "dischargePort"):
            if "1" not in locodes[str(row[field]["locode"])]["function_codes"]:
                raise ValueError("corrected route run still contains a non-maritime endpoint")
    if len(route_defects) != 11 or len({row["document_id"] for row in route_defects}) != 10:
        raise ValueError("unexpected old route defect scope")

    raw_rows = _read_jsonl(roots["compiler"] / "generation" / "results.jsonl")
    raw_by_id = {str(row["documentId"]): row for row in raw_rows}
    if set(raw_by_id) != set(selected_by_id):
        raise ValueError("raw compiler result set differs from selection")
    raw_attempts: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    equipment_types: Counter[str] = Counter()
    equipment_sizes: Counter[str] = Counter()
    package_categories: Counter[str] = Counter()
    hs_lengths: Counter[str] = Counter()
    for document_id, source in selected_by_id.items():
        target_row = linguistic_by_id[document_id]
        target = _patch(target_row)
        for row in _as_rows(target.get("containers")):
            if row.get("typeCategory"):
                equipment_types[str(row["typeCategory"])] += 1
            if row.get("sizeCategory"):
                equipment_sizes[str(row["sizeCategory"])] += 1
        for row in _as_rows(target.get("cargoPackages")):
            if row.get("typeCategory"):
                package_categories[str(row["typeCategory"])] += 1
        for row in _as_rows(target.get("cargoGroups")):
            for code in row.get("hsCodes", []):
                hs_lengths[str(len(str(code)))] += 1
        raw = raw_by_id[document_id]
        usage = raw["usage"]
        case = roots["compiler"] / "cases" / document_id
        source_label_features = {
            key.replace("target_", "source_label_", 1): value
            for key, value in _target_features(source_records[document_id]).items()
            if key.startswith("target_")
        }
        metrics = {
            "document_id": document_id,
            "position": source["position"],
            "template_id": source["template_id"],
            "carrier_family": source["carrier_family"],
            "contexts": "|".join(source["contexts"]),
            **{f"source_{key}": value for key, value in source["strata"].items()},
            **source_label_features,
            **_target_features(target_row),
            "commercial_origin_country": route_v1[document_id]["commercialOriginCountryCode"],
            "commercial_destination_country": route_v1[document_id][
                "commercialDestinationCountryCode"
            ],
            "loading_port_source": route_v1[document_id]["loadingPort"]["source"],
            "discharge_port_source": route_v1[document_id]["dischargePort"]["source"],
            "old_route_non_maritime": int(
                document_id in {row["document_id"] for row in route_defects}
            ),
            "route_changed_by_maritime_fix": int(document_id in route_changed),
            "freight_arrangement": route_v1[document_id]["freight"]["arrangement"] or "missing",
            "raw_status": raw["status"],
            "raw_failure_category": (
                "none" if raw["status"] == "training_ready" else _raw_failure_category(raw)
            ),
            "compiler_work_items": raw["compilerWorkItems"],
            "inventory_candidates": raw["inventoryCandidates"],
            "model_lines": raw["modelLines"],
            "raw_requests": usage["requests"],
            "raw_input_tokens": usage["inputTokens"],
            "raw_reasoning_tokens": usage["reasoningTokens"],
            "raw_visible_tokens": usage["visibleOutputTokens"],
            "raw_cost_usd": float(usage.get("providerReportedCostUsd") or 0),
            **_line_metrics(case),
        }
        document_rows.append(metrics)
        for stage in json.loads((case / "stages.json").read_text(encoding="utf-8")):
            stage_usage = stage.get("usage") if isinstance(stage.get("usage"), Mapping) else {}
            raw_attempts.append(
                {
                    "document_id": document_id,
                    "provider_route": stage.get("routeProvider"),
                    "error_type": stage.get("errorType"),
                    "host_error": stage.get("hostError"),
                    "billed_requests": int(stage_usage.get("requests", 0)),
                    "input_tokens": int(stage_usage.get("inputTokens", 0)),
                    "reasoning_tokens": int(stage_usage.get("reasoningTokens", 0)),
                    "visible_tokens": int(stage_usage.get("visibleOutputTokens", 0)),
                    "provider_cost_usd": float(stage_usage.get("providerReportedCostUsd", 0) or 0),
                }
            )

    cert_roots = [roots["cert10"], roots["cert50"]]
    cert_cases, cert_findings, cert_attempts = _certification_data(cert_roots)
    cert_by_id = {row["document_id"]: row for row in cert_cases}
    if len(cert_by_id) != 60 or not set(cert_by_id).issubset(set(raw_by_id)):
        raise ValueError("certification does not cover 60 unique compiler cases")
    for row in document_rows:
        row.update(
            cert_by_id.get(
                row["document_id"],
                {
                    "certification_status": "not_audited",
                    "certification_findings": None,
                    "certification_cost_usd": 0.0,
                },
            )
        )
    manual_pass_audit = _manual_certified_audit(cert_roots)

    raw_summary = _read_json(roots["compiler"] / "summary.json")
    cert_summaries = [_read_json(root / "summary.json") for root in cert_roots]
    package_summary = _read_json(roots["package"] / "generation" / "summary.json")
    ling_v1 = _read_json(roots["linguistic_v1"] / "generation" / "summary.json")
    ling_v2 = _read_json(roots["linguistic_v2"] / "generation" / "summary.json")
    prior_summary = _read_json(roots["prior_compiler"] / "summary.json")
    cert_actual = sum(float(row["providerReportedCostUsd"]) for row in cert_summaries)
    cert_discount = sum(float(row["estimatedCostUsd"]) for row in cert_summaries)
    linguistic_lineage = float(ling_v1["estimatedCostUsd"]) + float(
        ling_v2["incrementalEstimatedCostUsd"]
    )
    package_cost = float(package_summary["estimatedCostUsd"])
    raw_actual = float(raw_summary["providerReportedCostUsd"])
    raw_discount = float(raw_summary["estimatedCostUsd"])
    first_pass_total = package_cost + linguistic_lineage + raw_actual + cert_actual
    first_pass_discount = package_cost + linguistic_lineage + raw_discount + cert_discount
    stage_costs = [
        {"stage": "package compatibility", "cost_usd": package_cost, "basis_documents": 100},
        {"stage": "linguistic completion", "cost_usd": linguistic_lineage, "basis_documents": 100},
        {"stage": "raw-text compiler", "cost_usd": raw_actual, "basis_documents": 100},
        {"stage": "independent audit", "cost_usd": cert_actual, "basis_documents": 60},
    ]
    for row in stage_costs:
        row["cost_per_1000_on_selected_batch"] = row["cost_usd"] * 10
    cost_projection = []
    for population in (1_000, 10_000, 50_000):
        cost_projection.extend(
            [
                {
                    "scenario": "raw compiler only - observed routes",
                    "documents": population,
                    "projected_cost_usd": raw_actual / 100 * population,
                },
                {
                    "scenario": "full first pass - observed provider mix",
                    "documents": population,
                    "projected_cost_usd": first_pass_total / 100 * population,
                },
                {
                    "scenario": "full first pass - discounted GLM rates",
                    "documents": population,
                    "projected_cost_usd": first_pass_discount / 100 * population,
                },
            ]
        )

    raw_failures = Counter(
        row["raw_failure_category"]
        for row in document_rows
        if row["raw_status"] != "training_ready"
    )
    finding_kinds = Counter(str(row["finding_kind"]) for row in cert_findings)
    status_counts = Counter(str(row["raw_status"]) for row in document_rows)
    cert_status = Counter(str(row["certification_status"]) for row in cert_cases)
    source_contexts = Counter(context for row in selected for context in row.get("contexts", []))
    origins = Counter(str(route_v1[row]["commercialOriginCountryCode"]) for row in route_v1)
    destinations = Counter(
        str(route_v1[row]["commercialDestinationCountryCode"]) for row in route_v1
    )
    freight = Counter(str(route_v1[row]["freight"]["arrangement"] or "missing") for row in route_v1)

    _write_csv(data_dir / "document-metrics.csv", document_rows)
    _write_csv(data_dir / "raw-provider-attempts.csv", raw_attempts)
    _write_csv(
        data_dir / "raw-failure-taxonomy.csv",
        _counter_rows(raw_failures, name="category", count="documents"),
    )
    _write_csv(data_dir / "certification-cases.csv", cert_cases)
    _write_csv(data_dir / "certification-findings.csv", cert_findings)
    _write_csv(data_dir / "certification-provider-attempts.csv", cert_attempts)
    _write_csv(data_dir / "manual-audit-of-model-certified-cases.csv", manual_pass_audit)
    _write_csv(data_dir / "route-v1-non-maritime-endpoints.csv", route_defects)
    _write_csv(data_dir / "stage-costs.csv", stage_costs)
    _write_csv(data_dir / "cost-projections.csv", cost_projection)

    sns.set_theme(style="whitegrid", context="notebook")
    funnel = [
        {"stage": "selected", "documents": 100},
        {"stage": "semantic complete", "documents": 100},
        {"stage": "linguistic complete", "documents": 100},
        {"stage": "host-ready", "documents": status_counts["training_ready"]},
        {
            "stage": "host-ready + route-clean",
            "documents": sum(
                row["raw_status"] == "training_ready" and not row["old_route_non_maritime"]
                for row in document_rows
            ),
        },
        {"stage": "model-certified", "documents": cert_status["certified"]},
        {"stage": "manual high-quality", "documents": 0},
    ]
    _plot_bar(
        funnel,
        x="stage",
        y="documents",
        title="Additional-100 end-to-end first-pass funnel",
        path=plots / "01_end_to_end_funnel.png",
    )
    _plot_bar(
        _counter_rows(status_counts, name="status", count="documents"),
        x="status",
        y="documents",
        title="Raw-text compiler outcomes",
        path=plots / "02_compiler_outcomes.png",
    )
    _plot_bar(
        _counter_rows(raw_failures, name="category", count="documents"),
        x="documents",
        y="category",
        horizontal=True,
        title="Compiler failure taxonomy",
        path=plots / "03_compiler_failure_taxonomy.png",
    )
    _plot_bar(
        _counter_rows(cert_status, name="status", count="documents"),
        x="status",
        y="documents",
        title="Independent semantic-audit outcomes (60 route-clean host passes)",
        path=plots / "04_certification_outcomes.png",
    )
    _plot_bar(
        _counter_rows(finding_kinds, name="kind", count="findings"),
        x="findings",
        y="kind",
        horizontal=True,
        title="Independent semantic findings",
        path=plots / "05_certification_finding_taxonomy.png",
    )

    docs = pd.DataFrame(document_rows)
    token_frame = docs.melt(
        id_vars=["document_id", "raw_status"],
        value_vars=["raw_input_tokens", "raw_reasoning_tokens", "raw_visible_tokens"],
        var_name="token_class",
        value_name="tokens",
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=token_frame, x="raw_status", y="tokens", hue="token_class", estimator="mean")
    plt.title("Mean compiler token consumption by outcome")
    plt.xlabel("")
    _save(plots / "06_compiler_tokens_by_outcome.png")

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=docs, x="raw_status", y="raw_cost_usd")
    sns.stripplot(data=docs, x="raw_status", y="raw_cost_usd", color="black", alpha=0.55)
    plt.title("Per-document raw compiler cost")
    plt.xlabel("")
    plt.ylabel("USD")
    _save(plots / "07_compiler_cost_distribution.png")

    provider_frame = pd.DataFrame(raw_attempts)
    provider_frame["outcome"] = provider_frame.apply(
        lambda row: (
            "transport/model failure"
            if pd.notna(row["error_type"]) and not row["billed_requests"]
            else "billed response rejected by host"
            if pd.notna(row["host_error"])
            else "billed response"
        ),
        axis=1,
    )
    plt.figure(figsize=(11, 6))
    sns.countplot(data=provider_frame, x="provider_route", hue="outcome")
    plt.title("Compiler provider routing and fallback outcomes")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    _save(plots / "08_provider_routes.png")

    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=docs,
        x="model_lines",
        y="raw_cost_usd",
        hue="raw_status",
        size="raw_requests",
        sizes=(35, 180),
    )
    plt.title("Compiler scope, retries, and cost")
    plt.xlabel("model-owned lines")
    plt.ylabel("USD")
    _save(plots / "09_compiler_scope_vs_cost.png")

    strata = []
    for field in (
        "source_document_type",
        "source_page_bucket",
        "source_source_corpus",
        "source_container_bucket",
    ):
        for value, count in Counter(docs[field]).most_common():
            strata.append(
                {"stratum": field.removeprefix("source_"), "value": value, "documents": count}
            )
    sf = pd.DataFrame(strata)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (key, part) in zip(axes.flat, sf.groupby("stratum", sort=False), strict=True):
        sns.barplot(data=part, x="value", y="documents", hue="value", legend=False, ax=axis)
        axis.set_title(key.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Selected source-template strata", y=1.01)
    fig.tight_layout()
    _save(plots / "10_source_strata.png")

    _plot_bar(
        _counter_rows(source_contexts, name="context", count="documents"),
        x="documents",
        y="context",
        horizontal=True,
        title="Template context coverage",
        path=plots / "11_template_contexts.png",
    )

    countries = pd.DataFrame(
        [
            {"country": key, "documents": value, "side": side}
            for side, counter in (("origin", origins), ("destination", destinations))
            for key, value in counter.most_common(15)
        ]
    )
    plt.figure(figsize=(13, 7))
    sns.barplot(data=countries, x="country", y="documents", hue="side")
    plt.title("Top-15 commercial origin and destination countries")
    _save(plots / "12_route_country_distribution.png")

    _plot_bar(
        _counter_rows(freight, name="arrangement", count="documents"),
        x="arrangement",
        y="documents",
        title="Freight arrangement distribution",
        path=plots / "13_freight_distribution.png",
    )

    _plot_bar(
        _counter_rows(equipment_types, name="type", count="containers"),
        x="type",
        y="containers",
        title="Synthetic equipment type categories",
        path=plots / "14_equipment_types.png",
    )
    _plot_bar(
        _counter_rows(equipment_sizes, name="size", count="containers"),
        x="size",
        y="containers",
        title="Synthetic equipment size categories",
        path=plots / "15_equipment_sizes.png",
    )
    _plot_bar(
        _counter_rows(package_categories, name="category", count="packages")[:20],
        x="packages",
        y="category",
        horizontal=True,
        title="Top synthetic package categories",
        path=plots / "16_package_categories.png",
    )
    _plot_bar(
        _counter_rows(hs_lengths, name="digits", count="codes"),
        x="digits",
        y="codes",
        title="Synthetic HS-code lengths",
        path=plots / "17_hs_code_lengths.png",
    )

    feature_rows = []
    for field, label in (
        ("target_dangerous_goods", "dangerous goods"),
        ("target_temperature_setpoints", "temperature setpoint"),
        ("target_imo_present", "IMO number"),
        ("target_flag_present", "vessel flag"),
        ("target_delivery_agent", "delivery agent"),
        ("target_forwarding_agent", "forwarding agent"),
        ("target_notify_parties", "notify party"),
    ):
        feature_rows.append({"feature": label, "documents": int((docs[field] > 0).sum())})
    _plot_bar(
        feature_rows,
        x="documents",
        y="feature",
        horizontal=True,
        title="Target feature coverage",
        path=plots / "18_feature_coverage.png",
    )

    cert_frame = pd.DataFrame(cert_cases)
    plt.figure(figsize=(11, 6))
    sns.histplot(data=cert_frame, x="certification_findings", discrete=True, binwidth=1)
    plt.title("Semantic findings per independently audited candidate")
    plt.xlabel("findings")
    _save(plots / "19_findings_per_document.png")

    merged = docs.merge(
        cert_frame[["document_id", "certification_findings"]],
        on="document_id",
        how="inner",
        suffixes=("", "_cert"),
    )
    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=merged,
        x="source_lines",
        y="certification_findings_cert",
        hue="target_containers",
        size="target_cargo_groups",
        sizes=(40, 180),
    )
    plt.title("Audit findings versus template length and target complexity")
    plt.xlabel("source OCR lines")
    plt.ylabel("semantic findings")
    _save(plots / "20_findings_vs_complexity.png")

    route_source_counts = Counter(
        row[field]["source"] for row in route_v1_rows for field in ("loadingPort", "dischargePort")
    )
    _plot_bar(
        _counter_rows(route_source_counts, name="source", count="endpoints"),
        x="source",
        y="endpoints",
        title="Route endpoint generation sources",
        path=plots / "21_route_endpoint_sources.png",
    )

    plt.figure(figsize=(11, 6))
    cp = pd.DataFrame(cost_projection)
    sns.barplot(data=cp, x="documents", y="projected_cost_usd", hue="scenario")
    plt.title("Observed first-pass cost projections (correction costs excluded)")
    plt.ylabel("USD")
    _save(plots / "22_cost_projections.png")

    quantities = pd.read_csv(roots["baseline"] / "data" / "driver-quantities.csv")
    quantity_frame = quantities.melt(
        value_vars=["source_quantity", "generated_quantity"],
        var_name="series",
        value_name="quantity",
    )
    plt.figure(figsize=(11, 6))
    sns.ecdfplot(data=quantity_frame, x="quantity", hue="series")
    plt.xscale("log")
    plt.title("Source versus generated driver-package quantities")
    plt.xlabel("quantity (log scale)")
    _save(plots / "23_source_vs_generated_quantities.png")

    measures = pd.read_csv(roots["baseline"] / "data" / "cargo-measure-totals.csv")
    grid = sns.relplot(
        data=measures,
        x="source_value",
        y="generated_value",
        col="measure",
        col_wrap=2,
        facet_kws={"sharex": False, "sharey": False},
        height=4.3,
    )
    for axis in grid.axes.flat:
        axis.set_xscale("log")
        axis.set_yscale("log")
    grid.figure.suptitle("Source versus generated cargo measures", y=1.02)
    grid.figure.savefig(plots / "24_source_vs_generated_measures.png", dpi=190, bbox_inches="tight")
    plt.close(grid.figure)

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=docs, x="raw_status", y="changed_line_fraction")
    sns.stripplot(
        data=docs,
        x="raw_status",
        y="changed_line_fraction",
        color="black",
        alpha=0.5,
    )
    plt.title("Changed-line fraction by compiler outcome")
    plt.xlabel("")
    _save(plots / "25_changed_line_fraction.png")

    outcome_by_page = (
        docs.groupby(["source_page_bucket", "raw_status"], observed=True)
        .size()
        .reset_index(name="documents")
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=outcome_by_page,
        x="source_page_bucket",
        y="documents",
        hue="raw_status",
    )
    plt.title("Compiler outcomes by source page bucket")
    plt.xlabel("page bucket")
    _save(plots / "26_compiler_outcomes_by_pages.png")

    evidence_origin = Counter(
        "at least one unchanged source line"
        if row["any_evidence_line_unchanged"]
        else "only rewritten lines"
        for row in cert_findings
    )
    _plot_bar(
        _counter_rows(evidence_origin, name="evidence_origin", count="findings"),
        x="evidence_origin",
        y="findings",
        title="Where independent-audit evidence occurs",
        path=plots / "27_finding_evidence_origin.png",
    )

    _plot_bar(
        stage_costs,
        x="stage",
        y="cost_usd",
        title="Observed API cost by first-pass stage",
        path=plots / "28_stage_cost_composition.png",
    )

    party_fields = [
        ("shipper", "source_label_shipper", "target_shipper"),
        ("consignee", "source_label_consignee", "target_consignee"),
        ("carrier", "source_label_carrier", "target_carrier"),
        ("delivery agent", "source_label_delivery_agent", "target_delivery_agent"),
        ("forwarding agent", "source_label_forwarding_agent", "target_forwarding_agent"),
        ("notify party", "source_label_notify_parties", "target_notify_parties"),
    ]
    party_rows = [
        {"role": role, "series": series, "documents": int((docs[field] > 0).sum())}
        for role, source_field, target_field in party_fields
        for series, field in (("source", source_field), ("synthetic target", target_field))
    ]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=pd.DataFrame(party_rows), x="role", y="documents", hue="series")
    plt.title("Party-role presence: source versus synthetic target")
    plt.xticks(rotation=25, ha="right")
    _save(plots / "29_party_role_coverage.png")

    structure_rows = []
    for label, source_field, target_field in (
        ("containers", "source_label_containers", "target_containers"),
        ("cargo groups", "source_label_cargo_groups", "target_cargo_groups"),
        ("package rows", "source_label_packages", "target_packages"),
        ("HS codes", "source_label_hs_codes", "target_hs_codes"),
    ):
        structure_rows.extend(
            {"structure": label, "series": "source", "count": value} for value in docs[source_field]
        )
        structure_rows.extend(
            {"structure": label, "series": "synthetic target", "count": value}
            for value in docs[target_field]
        )
    structure_frame = pd.DataFrame(structure_rows)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=structure_frame, x="structure", y="count", hue="series", showfliers=False)
    sns.stripplot(
        data=structure_frame,
        x="structure",
        y="count",
        hue="series",
        dodge=True,
        alpha=0.22,
        legend=False,
    )
    plt.title("Document structure: source versus synthetic target")
    _save(plots / "30_source_vs_target_structure.png")

    finding_source_unchanged = sum(
        bool(row["any_evidence_line_unchanged"]) for row in cert_findings
    )
    prior_ids = {
        case.name for case in (roots["prior_compiler"] / "cases").iterdir() if case.is_dir()
    }
    overlapping_ids = prior_ids & set(selected_by_id)
    overlap = len(overlapping_ids)
    identical_overlapping_targets = sum(
        (roots["prior_compiler"] / "cases" / document_id / "target-label.json").read_bytes()
        == (roots["compiler"] / "cases" / document_id / "target-label.json").read_bytes()
        for document_id in overlapping_ids
    )
    identical_overlapping_outputs = sum(
        (roots["prior_compiler"] / "cases" / document_id / "final.txt").read_bytes()
        == (roots["compiler"] / "cases" / document_id / "final.txt").read_bytes()
        for document_id in overlapping_ids
    )
    singleton_holds = [row for row in cert_cases if row["certification_findings"] == 1]
    docs_with_cohort = docs.assign(
        prior_fifty_cohort=docs["document_id"].map(
            lambda value: "overlap" if value in prior_ids else "fresh"
        )
    )
    cohort_outcomes = {
        str(cohort): {
            str(status): int(count)
            for status, count in part["raw_status"].value_counts().items()
        }
        for cohort, part in docs_with_cohort.groupby("prior_fifty_cohort")
    }
    page_outcomes = {
        str(bucket): {
            "documents": len(part),
            "trainingReady": int((part["raw_status"] == "training_ready").sum()),
            "trainingReadyFraction": float((part["raw_status"] == "training_ready").mean()),
        }
        for bucket, part in docs.groupby("source_page_bucket")
    }
    audited_docs = docs.loc[docs["certification_status"] != "not_audited"].copy()
    finding_correlations = {
        field: float(value)
        for field, value in audited_docs[
            [
                "source_lines",
                "model_lines",
                "compiler_work_items",
                "target_containers",
                "target_cargo_groups",
                "target_packages",
                "changed_line_fraction",
                "certification_findings",
            ]
        ]
        .corr(method="spearman")["certification_findings"]
        .drop("certification_findings")
        .items()
    }
    quantity_summaries = {
        "source": _numeric_summary(quantities["source_quantity"]),
        "generated": _numeric_summary(quantities["generated_quantity"]),
    }
    measure_summaries = {
        str(measure): {
            "source": _numeric_summary(part["source_value"]),
            "generated": _numeric_summary(part["generated_value"]),
        }
        for measure, part in measures.groupby("measure")
    }
    raw_reasoning_share = float(
        docs["raw_reasoning_tokens"].sum()
        / (docs["raw_reasoning_tokens"].sum() + docs["raw_visible_tokens"].sum())
    )
    normalized_full_first_pass_per_thousand = (
        package_cost * 10
        + linguistic_lineage * 10
        + raw_actual * 10
        + cert_actual / len(cert_cases) * 1000
    )
    summary = {
        "selectedDocuments": 100,
        "distinctTemplates": len({row["template_id"] for row in selected}),
        "distinctCarrierFamilies": len({row["carrier_family"] for row in selected}),
        "priorFiftySourceOverlap": overlap,
        "freshSourceTemplatesRelativeToPriorFifty": 100 - overlap,
        "identicalTargetLabelsAmongPriorFiftyOverlap": identical_overlapping_targets,
        "identicalRawOutputsAmongPriorFiftyOverlap": identical_overlapping_outputs,
        "routeV1NonMaritimeEndpoints": len(route_defects),
        "routeV1AffectedDocuments": len({row["document_id"] for row in route_defects}),
        "routeV2NonMaritimeEndpoints": 0,
        "routeDocumentsChangedByCorrectedPool": len(route_changed),
        "compiler": {
            "outcomes": dict(status_counts),
            "providerAttempts": raw_summary["providerAttempts"],
            "failedProviderAttempts": raw_summary["failedProviderAttempts"],
            "requests": raw_summary["requests"],
            "providerReportedCostUsd": raw_actual,
            "providerReportedCostPerThousandAttemptedUsd": raw_actual * 10,
            "discountRateCostPerThousandAttemptedUsd": raw_discount * 10,
            "throughputDocumentsPerHour": raw_summary["throughputDocumentsPerHour"],
            "reasoningShareOfOutputTokens": raw_reasoning_share,
            "outcomesByPriorFiftyOverlap": cohort_outcomes,
            "outcomesByPageBucket": page_outcomes,
        },
        "certification": {
            "auditedDocuments": len(cert_cases),
            "outcomes": dict(cert_status),
            "findings": len(cert_findings),
            "findingKinds": dict(finding_kinds),
            "findingsWithAtLeastOneUnchangedSourceLine": finding_source_unchanged,
            "singletonHeldDocuments": len(singleton_holds),
            "providerReportedCostUsd": cert_actual,
            "providerReportedCostPerThousandAuditedUsd": cert_actual / len(cert_cases) * 1000,
            "manualHighQualityPassesAmongModelCertified": 0,
            "findingCountSpearmanCorrelations": finding_correlations,
        },
        "cost": {
            "packageCompatibilityUsd": package_cost,
            "linguisticLineageBilledEstimateUsd": linguistic_lineage,
            "linguisticRetainedFinalUnitsEstimateUsd": float(ling_v2["estimatedCostUsd"]),
            "rawCompilerProviderReportedUsd": raw_actual,
            "certificationProviderReportedUsd": cert_actual,
            "fullFirstPassObservedUsdFor100Selected": first_pass_total,
            "fullFirstPassObservedPerThousandSelectedUsd": first_pass_total * 10,
            "fullFirstPassDiscountedGlmPerThousandSelectedUsd": first_pass_discount * 10,
            "normalizedFullFirstPassPerThousandIfEveryDocumentAuditedUsd": normalized_full_first_pass_per_thousand,
        },
        "priorRawCompiler": {
            "documents": prior_summary["liveDocuments"],
            "trainingReady": prior_summary["trainingReadyDocuments"],
            "providerReportedCostPerThousandAttemptedUsd": float(
                prior_summary["providerReportedCostPerThousandUsd"]
            ),
        },
        "target": {
            "containers": int(docs["target_containers"].sum()),
            "cargoGroups": int(docs["target_cargo_groups"].sum()),
            "packageRows": int(docs["target_packages"].sum()),
            "hsCodes": int(docs["target_hs_codes"].sum()),
            "dangerousGoodsRows": int(docs["target_dangerous_goods"].sum()),
            "temperatureSetpoints": int(docs["target_temperature_setpoints"].sum()),
            "imoNumbers": int(docs["target_imo_present"].sum()),
            "vesselFlags": int(docs["target_flag_present"].sum()),
        },
        "distributions": {
            "driverPackageQuantity": quantity_summaries,
            "cargoMeasures": measure_summaries,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    top_findings = finding_kinds.most_common()
    failure_table = "\n".join(f"| {key} | {value} |" for key, value in raw_failures.most_common())
    finding_table = "\n".join(f"| `{key}` | {value} |" for key, value in top_findings)
    report = f"""# Additional-100 synthetic B/L end-to-end audit

## Executive result

The 100-scenario structured and linguistic stages completed, but the resulting raw-OCR batch is
**diagnostic, not publishable**. The raw compiler admitted **{status_counts["training_ready"]}/100**,
of which **60** were both host-ready and unaffected by the old non-maritime route defect. Independent
read-only audit admitted **{cert_status["certified"]}/60** and held **{cert_status["needs_review"]}/60**
with **{len(cert_findings)}** concrete findings. Manual inspection of all three apparent passes found
at least one residual quality/anonymization defect in each, so **0 records are promoted**.

The low first-pass quality is not hidden by the deterministic compiler: every rejected or uncertain
case remains fail-closed, and no training records were published.

## Cost: what the numbers mean

- Raw compiler: **${raw_actual:.6f} for 100 attempts = ${raw_actual * 10:.2f}/1,000 attempted**.
- The previous clean 50 measured **${float(prior_summary["providerReportedCostPerThousandUsd"]):.2f}/1,000**;
  therefore this 100-run shows **no raw-compiler cost creep**.
- Independent audit: **${cert_actual:.6f} for 60 candidates = ${cert_actual / 60 * 1000:.2f}/1,000 audited**.
- Complete first-pass lineage (package compatibility, linguistic completion, raw compiler, and the
  60 audits actually run): **${first_pass_total:.6f} for 100 selected source templates =
  ${first_pass_total * 10:.2f}/1,000 selected**.
- Normalizing the audit rate to all selected documents gives an estimated **${normalized_full_first_pass_per_thousand:.2f}/1,000 selected**.
  This remains a first-pass operating estimate, not a certified-record cost.
- At the configured discounted GLM rates, that complete first-pass figure is
  **${first_pass_discount * 10:.2f}/1,000 selected**.
- The linguistic value uses actual lineage consumption: the incomplete v1 run plus the incremental
  v2 resume (**${linguistic_lineage:.6f}**). The v2 summary's lower **${float(ling_v2["estimatedCostUsd"]):.6f}**
  counts only retained final units and is not the amount consumed across both invocations.

These are first-pass costs, not cost per certified document. A production cost per certified record
cannot be estimated honestly while the manually accepted first-pass yield is zero and correction
cost has not been measured on this cohort. Projecting the observed first pass gives
**${first_pass_total / 100 * 10_000:.2f} for 10,000** or
**${first_pass_total / 100 * 50_000:.2f} for 50,000** selected templates, before corrections.

## Pipeline and distributions

- 100 unique templates from {len({row["carrier_family"] for row in selected})} carrier families;
  {overlap} source documents reuse templates from the earlier 50 and {100 - overlap} are fresh relative
  to it. This is template reuse, not duplicate augmentation: **{identical_overlapping_targets}/{overlap}**
  target labels and **{identical_overlapping_outputs}/{overlap}** rewritten outputs are byte-identical
  to their earlier counterparts.
- Source strata: 62 B/L and 38 SWB; 30 one-page, 48 two-page, and 22 three-or-more-page templates.
- Synthetic targets: {int(docs["target_containers"].sum())} containers,
  {int(docs["target_cargo_groups"].sum())} cargo groups,
  {int(docs["target_packages"].sum())} package rows, and {int(docs["target_hs_codes"].sum())} HS codes.
- Controlled rare features: {int(docs["target_dangerous_goods"].sum())} DG rows,
  {int(docs["target_temperature_setpoints"].sum())} temperature setpoints,
  {int(docs["target_imo_present"].sum())} IMO number, and
  {int(docs["target_flag_present"].sum())} vessel flags.
- Commercial coverage spans {len(origins)} origin countries and {len(destinations)} destination countries.
- Linguistic completion generated {ling_v2["generatedPartyNames"]} party names,
  {ling_v2["uniqueGeneratedPartyNames"]} unique; it completed 100/100 strict targets.

The complete distributions are in `data/document-metrics.csv`; plots 10-18 cover source strata,
contexts, route countries, freight, equipment, packages, HS lengths, and rare-feature support.
Plots 23-30 compare source versus generated quantities, measures, party topology, and document
structure, and show how rendering outcomes vary with page count and changed-line fraction.

### Statistical diagnostics

- Raw admission was stable across page buckets: **66.7%** for one-page, **68.8%** for two-page,
  and **63.6%** for three-or-more-page templates. The fresh-vs-prior comparison was similarly close:
  **33/54 (61.1%)** fresh templates and **34/46 (73.9%)** prior-overlap templates were host-ready.
- Finding count correlates much more strongly with source length (Spearman **0.584**) and model-owned
  line count (**0.431**) than with target containers (**0.238**), packages (**-0.102**), or cargo
  groups (**-0.196**). The dominant weakness is therefore incomplete coverage of repeated and auxiliary
  surfaces in long templates—not inability to construct complex structured targets.
- Raw compiler output comprised **143,815 reasoning tokens** and **93,748 visible tokens**;
  reasoning was **{raw_reasoning_share:.1%}** of generated tokens. Per-document compiler cost was
  **${float(docs["raw_cost_usd"].median()):.6f} median**, **${float(docs["raw_cost_usd"].quantile(0.95)):.6f} p95**, and
  **${float(docs["raw_cost_usd"].max()):.6f} maximum**.
- Driver-package quantities broadly retain scale (source/generated medians **96/101**, p95
  **2,393/2,345**) while reducing the extreme upper tail (maxima **6,665/6,588** and means
  **564/461**).
- Capacity-safe generation contracts the heaviest measure tails: median generated/source gross weight
  is **13,877/18,667**, net weight **13,065/20,851**, and volume **32.3/46.0**. This prevents impossible
  equipment loads but may underrepresent valid high-capacity/outlier documents; it is a distributional
  trade-off to revisit before large-scale augmentation.

## Route defect found and fixed

The v1 route artifact intersected WPI and UN/LOCODE by code but did not require the current UN/LOCODE
entry to retain maritime function `1`. It therefore emitted {len(route_defects)} non-maritime endpoints
across {len({row["document_id"] for row in route_defects})} documents. The corrected generator excludes
416 such stale/non-maritime WPI rows. Its same-100 replay produced **0/200 non-maritime loading/discharge
endpoints**. Because removal changes deterministic pool cardinality, {len(route_changed)}/100 scenario routes
changed, not merely the ten visibly invalid cases. Downstream v1 targets are consequently not reusable;
they must be regenerated from the corrected route artifact.

## Raw compiler failures

| Failure family | Documents |
|---|---:|
{failure_table}

The largest transport issue was provider saturation: 13 documents exhausted both configured routes.
All zero-response provider failures were cost-free; billed host-rejected attempts remain included in the
provider-reported total. The remaining failures are explicit contract rejections, not silent malformed text.

## Independent quality audit

| Finding kind | Findings |
|---|---:|
{finding_table}

The findings are substantive. Representative defects include old package/weight totals surviving on rider
pages, source tax and customs identities, source signing agents, wrong repeated container equipment,
jurisdiction-specific Egypt text on non-Egypt shipments, stale reefer temperatures, and cargo descriptions
or dangerous-goods names that disagree across pages. **{finding_source_unchanged}/{len(cert_findings)}**
findings cite at least one line that is byte-identical to the source, directly measuring incomplete source
cleanup. All {len(singleton_holds)} one-finding holds were inspected and each contains a decisive concrete
contradiction; the high hold rate is not explained by one noisy multi-finding outlier.

The three model-certified candidates also exposed false negatives during manual inspection:

1. `doc_56e593ae…`: invented `NORWIND PROVINCE` in a UK address is not plausible template flavor.
2. `doc_88ec16af…`: source `PFI# 963` survives, and a Chinese delivery agent receives a `+40` fax.
3. `doc_f0f120de…`: all three source container tare weights survive unchanged.

This establishes that the independent model audit remains necessary but is not sufficient. The current
compiler/auditor pair cannot yet certify production-grade synthetic OCR without an additional deterministic
coverage improvement and a measured correction pass.

## Strengths

- All 100 targets passed strict schema and relational inverse validation before raw rendering.
- Package/goods compatibility was resolved for all targets; route, equipment, DG, HS, temperature, and
  allocation structures are represented with meaningful diversity.
- Raw rendering remains fail-closed and preserves line/page topology in every committed case.
- The raw compiler's observed cost stayed near the earlier clean-50 benchmark.
- Every provider-visible prompt, output, receipt, rejection, and audit finding is retained for review.

## Weaknesses and next gate

1. Regenerate downstream targets from the corrected maritime route run; the v1 target lineage is invalid.
2. Expand compiler ownership from primary label values to **all repeated and dependent source surfaces**:
   rider totals, written-out dates, per-container measures, national-program labels, signature identities,
   and template-specific operational references.
3. Add deterministic freshness checks for source-only numbers/names and coherence checks for generated
   auxiliary contacts and jurisdictions. The auditor should become confirmation, not discovery.
4. Re-run a small stratified regression over the 57 held patterns, then repeat this 100-case gate. The exit
   criterion is a large-supermajority first-pass certification rate with zero manual false passes.
5. Only after that result should correction cost and cost per certified record be projected at 10k-50k scale.

## Artifact index

- `summary.json`: headline counts and cost semantics.
- `data/document-metrics.csv`: one row per selected source/template/target/compiler outcome.
- `data/raw-failure-taxonomy.csv`: compiler/preflight failure families.
- `data/certification-findings.csv`: all exact-evidence semantic findings.
- `data/manual-audit-of-model-certified-cases.csv`: the three false-pass inspections.
- `data/route-v1-non-maritime-endpoints.csv`: old endpoints and corrected replacements.
- `data/stage-costs.csv` and `data/cost-projections.csv`: observed cost basis and projections.
- `plots/`: 30 matplotlib/seaborn visualizations.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    manifest_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest_rows.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schemaVersion": 1,
        "analysis": "mpci-bl-additional100-end-to-end-v1",
        "sourceRuns": {
            key: {
                "path": RUNS[key],
                "commitSha256": _read_json(value / "_COMMIT.json").get("contentSha256"),
            }
            for key, value in roots.items()
            if (value / "_COMMIT.json").is_file()
        },
        "files": manifest_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
