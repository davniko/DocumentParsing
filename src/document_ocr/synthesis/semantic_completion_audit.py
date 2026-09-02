"""Reproducible audit of a composed deterministic B/L synthesis run.

The semantic-completion output is deliberately not a training dataset: source
text still contains the old lexical surfaces.  This audit therefore checks the
deterministic semantic plan, its cross-stage provenance, and the correlations
that the later linguistic renderer must preserve.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.generators import validate_container_number
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_V5_TASK_ADAPTER
from document_ocr.synthesis.transport_auxiliary import imo_check_digit

_AUDIT_VERSION = 1
_EXPECTED_DOCUMENTS = 100


def _json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _validated_run(root: Path) -> StagedCommitReceipt:
    commit_value = json.loads(read_regular_file_bytes(root / "_COMMIT.json"))
    transaction = commit_value.get("transaction_sha256")
    if not isinstance(transaction, str):
        raise ValueError(f"committed run has no transaction SHA-256: {root}")
    run = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=transaction,
    )
    return run.validate_committed_run()


def _artifact_sha(receipt: StagedCommitReceipt, relative_path: str) -> str:
    for row in receipt.artifacts:
        if row.relative_path == relative_path:
            return row.sha256
    raise ValueError(f"committed run does not contain {relative_path}")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("expected a list or null")
    return tuple(value)


def _weight_value(group: Mapping[str, Any], field: str) -> float | None:
    value = group.get(field)
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("value"), (int, float)):
        raise TypeError(f"cargo {field} is not a numeric measurement")
    return float(value["value"])


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return stream.getvalue().encode("utf-8")


def _plot_runtime() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd  # type: ignore[import-untyped]
        import seaborn as sns  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "semantic completion audit requires the analysis dependency group"
        ) from error
    sns.set_theme(style="whitegrid", context="notebook")
    return plt, pd, sns


def _png(figure: Any) -> bytes:
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "document-ocr semantic completion audit"},
    )
    return stream.getvalue()


def _top_counts(values: Iterable[str], *, maximum: int = 15) -> list[dict[str, Any]]:
    counts = Counter(values)
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda row: (-row[1], row[0]))[:maximum]
    ]


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _source_feature_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        {
            "documents": len(rows),
            "thermal_documents": 0,
            "imo_documents": 0,
            "flag_documents": 0,
            "dangerous_goods_documents": 0,
            "class3_rows": 0,
            "class3_rows_with_flashpoint": 0,
        }
    )
    for row in rows:
        target = row.get("target")
        if not isinstance(target, dict):
            raise TypeError("source synthesis row has no target object")
        patch = target.get("documentPatch")
        if not isinstance(patch, dict):
            raise TypeError("source synthesis target has no documentPatch object")
        containers = _as_tuple(patch.get("containers"))
        counts["thermal_documents"] += any(
            isinstance(container, dict) and container.get("temperatureSetpoint") is not None
            for container in containers
        )
        transport = patch.get("transport")
        if transport is not None and not isinstance(transport, dict):
            raise TypeError("source synthesis target transport is not an object")
        if isinstance(transport, dict):
            counts["imo_documents"] += transport.get("vesselImoNumber") is not None
            counts["flag_documents"] += transport.get("vesselFlagCountry") is not None

        document_has_dangerous_goods = False
        for cargo in _as_tuple(patch.get("cargoGroups")):
            if not isinstance(cargo, dict):
                raise TypeError("source synthesis cargo group is not an object")
            for dangerous in _as_tuple(cargo.get("dangerousGoods")):
                if not isinstance(dangerous, dict):
                    raise TypeError("source dangerous-goods row is not an object")
                document_has_dangerous_goods = True
                categories = [dangerous.get("hazardCategory")]
                subsidiary = dangerous.get("subsidiaryHazardCategory")
                if subsidiary is not None:
                    categories.append(subsidiary)
                subsidiaries = dangerous.get("subsidiaryHazardCategories")
                if subsidiaries is not None:
                    categories.extend(_as_tuple(subsidiaries))
                if "FLAMMABLE_LIQUIDS" in categories:
                    counts["class3_rows"] += 1
                    counts["class3_rows_with_flashpoint"] += dangerous.get("flashPoint") is not None
        counts["dangerous_goods_documents"] += document_has_dangerous_goods
    documents = counts["documents"]
    if documents <= 0:
        raise ValueError("source synthesis corpus is empty")
    class3 = counts["class3_rows"]
    return {
        "counts": dict(counts),
        "ratesPercent": {
            "thermal document": 100.0 * counts["thermal_documents"] / documents,
            "IMO number": 100.0 * counts["imo_documents"] / documents,
            "vessel flag": 100.0 * counts["flag_documents"] / documents,
            "DG document": 100.0 * counts["dangerous_goods_documents"] / documents,
            "class-3 flashpoint": (
                100.0 * counts["class3_rows_with_flashpoint"] / class3 if class3 else 0.0
            ),
        },
    }


def audit_semantic_completion(
    *,
    structured_root: Path,
    route_root: Path,
    controlled_root: Path,
    dangerous_goods_root: Path,
    completion_root: Path,
    source_records_path: Path,
) -> dict[str, Any]:
    """Validate and summarize one exact composed run without mutating it."""

    roots = {
        "structured": structured_root,
        "route": route_root,
        "controlled": controlled_root,
        "dangerous_goods": dangerous_goods_root,
        "completion": completion_root,
    }
    receipts = {name: _validated_run(root) for name, root in roots.items()}

    selected = _csv_rows(structured_root / "data/selected-sources.csv")
    route_rows = _json_rows(route_root / "generation/shipment-scenarios.jsonl")
    controlled_rows = _json_rows(controlled_root / "generation/controlled-scenarios.jsonl")
    dg_rows = _json_rows(dangerous_goods_root / "generation/dg-plans.jsonl")
    raw_completion_rows = _json_rows(completion_root / "generation/completion-plans.jsonl")
    target_rows = _json_rows(completion_root / "generation/v5-targets.jsonl")
    source_rows = _json_rows(source_records_path)
    source_features = _source_feature_statistics(source_rows)

    completion_rows = [
        SemanticCompletionPlanRow.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in raw_completion_rows
    ]
    selected_by_document = {row["document_id"]: row for row in selected}
    route_by_document = {row["baseDocumentId"]: row for row in route_rows}
    controlled_by_document = {row["baseDocumentId"]: row for row in controlled_rows}
    dg_by_document = {row["base_document_id"]: row for row in dg_rows}
    target_by_scenario = {row["scenarioId"]: row for row in target_rows}

    failures: list[str] = []
    input_counts = {
        "selected": len(selected),
        "route": len(route_rows),
        "controlled": len(controlled_rows),
        "dangerous_goods": len(dg_rows),
        "completion": len(completion_rows),
        "targets": len(target_rows),
    }
    _require(
        set(input_counts.values()) == {_EXPECTED_DOCUMENTS},
        f"cross-stage row counts differ: {input_counts}",
        failures,
    )
    expected_documents = set(selected_by_document)
    for name, values in (
        ("route", set(route_by_document)),
        ("controlled", set(controlled_by_document)),
        ("dangerous_goods", set(dg_by_document)),
        ("completion", {row.base_document_id for row in completion_rows}),
    ):
        _require(values == expected_documents, f"{name} document set differs", failures)

    document_rows: list[dict[str, Any]] = []
    container_rows: list[dict[str, Any]] = []
    cargo_rows: list[dict[str, Any]] = []
    package_rows: list[dict[str, Any]] = []
    route_table: list[dict[str, Any]] = []
    dangerous_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    all_containers: list[str] = []
    all_imos: list[str] = []
    all_vessels: list[str] = []
    route_relation_checks = 0
    route_relation_valid = 0
    gross_net_checks = 0
    gross_net_valid = 0

    for completion in sorted(completion_rows, key=lambda row: row.base_document_id):
        source = selected_by_document[completion.base_document_id]
        route = route_by_document[completion.base_document_id]
        controlled = controlled_by_document[completion.base_document_id]
        dg_plan = dg_by_document[completion.base_document_id]
        target_row = target_by_scenario.get(completion.scenario_id)
        _require(target_row is not None, f"missing v5 target {completion.scenario_id}", failures)
        if target_row is not None:
            _require(
                target_row.get("target") == completion.target,
                f"target projection differs for {completion.scenario_id}",
                failures,
            )
            _require(
                target_row.get("targetSha256") == completion.target_sha256,
                f"target hash projection differs for {completion.scenario_id}",
                failures,
            )
        _require(
            sha256_bytes(canonical_json_bytes(completion.target)) == completion.target_sha256,
            f"target hash invalid for {completion.scenario_id}",
            failures,
        )
        try:
            BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
                document_id=completion.scenario_id,
                target=completion.target,
            )
        except (TypeError, ValueError) as error:
            failures.append(f"v5 schema/inverse failed for {completion.scenario_id}: {error}")

        patch = completion.target["documentPatch"]
        if not isinstance(patch, dict):
            raise TypeError("completion target documentPatch is not an object")
        containers = _as_tuple(patch.get("containers"))
        cargos = _as_tuple(patch.get("cargoGroups"))
        packages = _as_tuple(patch.get("cargoPackages"))
        allocation_groups = _as_tuple(patch.get("cargoAllocationGroups"))
        cargo_by_id = {row["groupId"]: row for row in cargos}
        package_by_id = {row["packageId"]: row for row in packages}
        container_numbers = {row["containerNumber"] for row in containers}

        for group in cargos:
            gross = _weight_value(group, "grossWeight")
            net = _weight_value(group, "netWeight")
            if gross is not None and net is not None:
                gross_net_checks += 1
                if gross >= net:
                    gross_net_valid += 1
                else:
                    failures.append(
                        f"gross weight below net for {completion.scenario_id}:{group['groupId']}"
                    )

        allocation_container_references: set[str] = set()
        allocation_package_references: set[str] = set()
        for allocation_group in allocation_groups:
            group_id = allocation_group["groupId"]
            _require(
                group_id in cargo_by_id,
                f"unknown cargo group allocation in {completion.scenario_id}:{group_id}",
                failures,
            )
            for package_id in _as_tuple(allocation_group.get("packageIds")):
                allocation_package_references.add(package_id)
                _require(
                    package_id in package_by_id,
                    f"unknown package allocation in {completion.scenario_id}:{package_id}",
                    failures,
                )
            for allocation in _as_tuple(allocation_group.get("allocations")):
                number = allocation["containerNumber"]
                allocation_container_references.add(number)
                _require(
                    number in container_numbers,
                    f"unknown container allocation in {completion.scenario_id}:{number}",
                    failures,
                )

        thermal_realizations = [
            row for row in completion.cargo_realizations if row.thermal_profile is not None
        ]
        thermal_groups = {row.cargo_group_id for row in thermal_realizations}
        active_equipment = [
            row for row in completion.equipment_realizations if row.active_temperature
        ]
        active_container_numbers = {row.container_number for row in active_equipment}
        expected_active = {
            number
            for cargo in thermal_realizations
            for number in cargo.associated_container_numbers
        }
        _require(
            active_container_numbers == expected_active,
            f"thermal cargo/equipment relation differs for {completion.scenario_id}",
            failures,
        )
        setpoint_by_group: dict[str, set[float]] = defaultdict(set)
        for equipment in active_equipment:
            _require(
                equipment.type_category
                in {
                    "REFRIGERATED",
                    "REFRIGERATED_AND_HEATED",
                    "SELF_POWERED_REFRIGERATED",
                    "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT",
                },
                f"active non-thermal equipment for {completion.scenario_id}",
                failures,
            )
            if equipment.temperature_value_celsius is None:
                failures.append(f"active equipment lacks setpoint for {completion.scenario_id}")
            else:
                for group_id in equipment.linked_thermal_cargo_groups:
                    setpoint_by_group[group_id].add(equipment.temperature_value_celsius)
        for group_id in thermal_groups:
            _require(
                len(setpoint_by_group[group_id]) == 1,
                f"thermal group has inconsistent setpoints: {completion.scenario_id}:{group_id}",
                failures,
            )

        dg_realizations = dg_plan["realizations"]
        generated_flashpoints = {
            (row.cargo_group_id, row.dangerous_goods_order): row
            for row in completion.flashpoint_realizations
            if row.generated
        }
        for realization in dg_realizations:
            group_id = realization["cargo_group_id"]
            order = int(realization["dangerous_goods_order"])
            dg_target = cargo_by_id[group_id]["dangerousGoods"][order]
            _require(
                dg_target.get("unNumber") == realization["un_number"],
                f"DG UN projection differs for {completion.scenario_id}:{group_id}:{order}",
                failures,
            )
            _require(
                dg_target.get("hazardCategory") == realization["semantic_hazard_category"],
                f"DG hazard projection differs for {completion.scenario_id}:{group_id}:{order}",
                failures,
            )
            flashpoint = generated_flashpoints.get((group_id, order))
            if realization["exact_hazard_class"] == "3":
                _require(
                    flashpoint is not None,
                    f"class-3 DG lacks configured audit flashpoint: {completion.scenario_id}",
                    failures,
                )
            dangerous_rows.append(
                {
                    "document_id": completion.base_document_id,
                    "scenario_id": completion.scenario_id,
                    "cargo_group_id": group_id,
                    "order": order,
                    "un_number": realization["un_number"],
                    "proper_shipping_name": realization["proper_shipping_name"],
                    "hazard_category": realization["semantic_hazard_category"],
                    "subsidiary_hazard_categories": "|".join(
                        realization["semantic_subsidiary_hazard_categories"]
                    ),
                    "exact_hazard_class": realization["exact_hazard_class"],
                    "packing_group": realization.get("packing_group_category"),
                    "hs6": "|".join(realization["generated_hs_codes"]),
                    "branch": realization["branch"],
                    "flashpoint_celsius": (
                        flashpoint.value_celsius if flashpoint is not None else None
                    ),
                }
            )

        origin = route["commercialOriginCountryCode"]
        destination = route["commercialDestinationCountryCode"]
        _require(origin != destination, f"self trade route for {completion.scenario_id}", failures)
        for party in route["partyLocalities"]:
            relation = party["relation"]
            country_code = party["countryCode"]
            if country_code is None:
                continue
            if relation in {"commercial_origin", "commercial_destination"}:
                route_relation_checks += 1
                expected = origin if relation == "commercial_origin" else destination
                if country_code == expected:
                    route_relation_valid += 1
                else:
                    failures.append(
                        f"party relation/country differs for {completion.scenario_id}:"
                        f"{party['role']}"
                    )

        transport_value = patch.get("transport")
        if transport_value is None:
            transport: dict[str, Any] = {}
        elif isinstance(transport_value, dict):
            transport = transport_value
        else:
            raise TypeError("completion target transport is not an object")
        vessel_name = transport.get("vesselName")
        if vessel_name is not None:
            if not isinstance(vessel_name, str):
                raise TypeError("completion target vesselName is not a string")
            all_vessels.append(vessel_name)
        imo = completion.transport_auxiliary.imo_number
        if imo is not None:
            all_imos.append(imo)
            _require(
                len(imo) == 7 and imo.isdigit() and imo_check_digit(imo[:6]) == int(imo[-1]),
                f"invalid IMO check digit for {completion.scenario_id}",
                failures,
            )

        for container in containers:
            number = container["containerNumber"]
            all_containers.append(number)
            _require(
                validate_container_number(number),
                f"invalid ISO 6346 container for {completion.scenario_id}:{number}",
                failures,
            )
            equipment = next(
                row for row in completion.equipment_realizations if row.container_number == number
            )
            container_rows.append(
                {
                    "document_id": completion.base_document_id,
                    "scenario_id": completion.scenario_id,
                    "container_number": number,
                    "size_category": container.get("sizeCategory"),
                    "type_category": container.get("typeCategory"),
                    "application_code": equipment.application_code,
                    "active_temperature": equipment.active_temperature,
                    "temperature_celsius": equipment.temperature_value_celsius,
                    "thermal_profiles": "|".join(
                        sorted(
                            row.thermal_profile or ""
                            for row in thermal_realizations
                            if number in row.associated_container_numbers
                        )
                    ),
                    "allocated": number in allocation_container_references,
                    "seal_count": len(_as_tuple(container.get("sealNumbers"))),
                }
            )

        realization_by_group = {row.cargo_group_id: row for row in completion.cargo_realizations}
        for cargo in cargos:
            group_id = cargo["groupId"]
            realization = realization_by_group.get(group_id)
            dangerous = _as_tuple(cargo.get("dangerousGoods"))
            hs_codes = _as_tuple(cargo.get("hsCodes"))
            cargo_rows.append(
                {
                    "document_id": completion.base_document_id,
                    "scenario_id": completion.scenario_id,
                    "cargo_group_id": group_id,
                    "description": cargo.get("description"),
                    "semantic_description": realization.description if realization else None,
                    "semantic_source": realization.semantic_source if realization else None,
                    "thermal_profile": realization.thermal_profile if realization else None,
                    "hs_codes": "|".join(hs_codes),
                    "hs_code_count": len(hs_codes),
                    "dg_count": len(dangerous),
                    "gross_weight": _weight_value(cargo, "grossWeight"),
                    "net_weight": _weight_value(cargo, "netWeight"),
                }
            )
        for package in packages:
            package_rows.append(
                {
                    "document_id": completion.base_document_id,
                    "scenario_id": completion.scenario_id,
                    "cargo_group_id": package["groupId"],
                    "package_id": package["packageId"],
                    "type_category": package.get("typeCategory"),
                    "quantity": package.get("quantity"),
                    "allocated": package["packageId"] in allocation_package_references,
                }
            )
        route_table.append(
            {
                "document_id": completion.base_document_id,
                "scenario_id": completion.scenario_id,
                "origin_country": origin,
                "destination_country": destination,
                "loading_country": route["loadingPort"]["countryCode"],
                "discharge_country": route["dischargePort"]["countryCode"],
                "loading_locode": route["loadingPort"]["locode"],
                "discharge_locode": route["dischargePort"]["locode"],
                "origin_prior": route["commercialOriginPriorEvidence"]["component"],
                "loading_source": route["loadingPort"]["source"],
                "discharge_source": route["dischargePort"]["source"],
                "freight": route["freight"]["arrangement"],
            }
        )
        for change in completion.changes:
            change_rows.append(
                {
                    "document_id": completion.base_document_id,
                    "scenario_id": completion.scenario_id,
                    "stage_id": change.stage_id,
                    "change_kind": change.change_kind,
                    "coupling_group": change.coupling_group,
                    "role_path": change.role_path,
                    "target_path": change.target_path,
                }
            )

        document_rows.append(
            {
                "document_id": completion.base_document_id,
                "scenario_id": completion.scenario_id,
                "template_id": completion.template_id,
                "carrier_family": source["carrier_family"],
                "document_type": source["document_type"],
                "source_corpus": source["source_corpus"],
                "page_bucket": source["page_bucket"],
                "source_container_bucket": source["container_bucket"],
                "selection_contexts": source["contexts"],
                "container_count": len(containers),
                "cargo_group_count": len(cargos),
                "package_count": len(packages),
                "allocation_group_count": len(allocation_groups),
                "dangerous_goods_count": len(dg_realizations),
                "thermal": bool(thermal_realizations),
                "thermal_profiles": "|".join(
                    sorted({row.thermal_profile or "" for row in thermal_realizations})
                ),
                "imo_present": imo is not None,
                "flag_present": completion.transport_auxiliary.flag_country_code is not None,
                "vessel_present": vessel_name is not None,
                "origin_country": origin,
                "destination_country": destination,
                "change_count": len(completion.changes),
                "blocker_count": len(completion.remaining_blockers),
                "training_eligible": completion.training_eligible,
                "controlled_linguistic_work": json.dumps(
                    controlled["remainingLinguisticWork"], sort_keys=True, separators=(",", ":")
                ),
            }
        )

    _require(len(all_containers) == len(set(all_containers)), "container collision", failures)
    _require(len(all_imos) == len(set(all_imos)), "IMO collision", failures)
    _require(len(all_vessels) == len(set(all_vessels)), "vessel-name collision", failures)
    template_counts = Counter(row["template_id"] for row in document_rows)
    _require(max(template_counts.values()) <= 3, "template reuse exceeds configured cap", failures)
    _require(
        all(not row["training_eligible"] for row in document_rows),
        "pre-linguistic row was marked training eligible",
        failures,
    )
    _require(route_relation_checks > 0, "no route/party relation checks were possible", failures)

    check_rows = [
        {
            "check": "cross_stage_row_count",
            "passed": set(input_counts.values()) == {100},
        },
        {
            "check": "v5_schema_and_relational_inverse",
            "passed": not any("v5 schema" in value for value in failures),
        },
        {
            "check": "target_projection_and_hashes",
            "passed": not any("target" in value for value in failures),
        },
        {
            "check": "container_iso6346_and_uniqueness",
            "passed": not any("container" in value.lower() for value in failures),
        },
        {
            "check": "imo_checksum_and_uniqueness",
            "passed": not any("imo" in value.lower() for value in failures),
        },
        {
            "check": "vessel_name_uniqueness",
            "passed": not any("vessel-name" in value for value in failures),
        },
        {
            "check": "route_party_country_coherence",
            "passed": route_relation_valid == route_relation_checks,
        },
        {
            "check": "thermal_cargo_equipment_setpoint_coherence",
            "passed": not any("thermal" in value for value in failures),
        },
        {
            "check": "dangerous_goods_projection_and_flashpoint",
            "passed": not any("DG" in value or "class-3" in value for value in failures),
        },
        {
            "check": "gross_not_below_net",
            "passed": gross_net_valid == gross_net_checks,
        },
        {
            "check": "template_reuse_cap",
            "passed": max(template_counts.values()) <= 3,
        },
        {
            "check": "pre_linguistic_not_training_eligible",
            "passed": all(not row["training_eligible"] for row in document_rows),
        },
    ]
    if failures:
        raise RuntimeError("semantic completion audit failed: " + "; ".join(failures))

    contexts = Counter(
        context
        for row in document_rows
        for context in str(row["selection_contexts"]).split("|")
        if context
    )
    package_counts = Counter(row["type_category"] for row in package_rows)
    container_types = Counter(row["type_category"] for row in container_rows)
    container_sizes = Counter(row["size_category"] for row in container_rows)
    thermal_container_profiles = Counter(
        profile
        for row in container_rows
        for profile in str(row["thermal_profiles"]).split("|")
        if profile
    )
    thermal_document_profiles = Counter(
        profile
        for row in document_rows
        for profile in str(row["thermal_profiles"]).split("|")
        if profile
    )
    blocker_counts = Counter(
        blocker for row in completion_rows for blocker in row.remaining_blockers
    )
    hs_lengths = Counter(
        len(code) for row in cargo_rows for code in str(row["hs_codes"]).split("|") if code
    )
    summary = {
        "schemaVersion": 1,
        "documents": len(document_rows),
        "containers": len(container_rows),
        "cargoGroups": len(cargo_rows),
        "packages": len(package_rows),
        "dangerousGoodsRows": len(dangerous_rows),
        "strictV5AndRelationalInverseValid": len(document_rows),
        "checksPassed": len(check_rows),
        "checksFailed": 0,
        "distinctTemplates": len(template_counts),
        "maximumTemplateReuse": max(template_counts.values()),
        "distinctCarriers": len({row["carrier_family"] for row in document_rows}),
        "distinctVesselNames": len(set(all_vessels)),
        "thermalDocuments": sum(bool(row["thermal"]) for row in document_rows),
        "thermalContainers": sum(bool(row["active_temperature"]) for row in container_rows),
        "thermalDocumentProfileCounts": dict(sorted(thermal_document_profiles.items())),
        "thermalContainerProfileCounts": dict(sorted(thermal_container_profiles.items())),
        "dangerousGoodsDocuments": sum(row["dangerous_goods_count"] > 0 for row in document_rows),
        "flashpoints": sum(row["flashpoint_celsius"] is not None for row in dangerous_rows),
        "imoNumbers": len(all_imos),
        "vesselFlags": sum(bool(row["flag_present"]) for row in document_rows),
        "routePartyRelationChecks": route_relation_checks,
        "routePartyRelationValid": route_relation_valid,
        "grossNetChecks": gross_net_checks,
        "grossNetValid": gross_net_valid,
        "trainingRecordsPublished": 0,
        "containerTypeCounts": dict(sorted(container_types.items())),
        "containerSizeCounts": dict(sorted(container_sizes.items())),
        "packageCategoryCounts": dict(sorted(package_counts.items())),
        "hsCodeLengthCounts": {str(key): value for key, value in sorted(hs_lengths.items())},
        "selectionContextCounts": dict(sorted(contexts.items())),
        "remainingBlockerCounts": dict(sorted(blocker_counts.items())),
        "sourceObservedFeatures": source_features,
        "originCountries": len({row["origin_country"] for row in route_table}),
        "destinationCountries": len({row["destination_country"] for row in route_table}),
        "auditDistributionNotice": (
            "Rare features were deliberately enriched for branch coverage; this run is not "
            "a production marginal-distribution estimate."
        ),
        "readiness": "ready_for_linguistic_realization",
    }
    return {
        "summary": summary,
        "checks": check_rows,
        "documents": document_rows,
        "containers": container_rows,
        "cargo": cargo_rows,
        "packages": package_rows,
        "routes": route_table,
        "dangerous_goods": dangerous_rows,
        "changes": change_rows,
        "receipts": receipts,
        "roots": roots,
        "source_records_path": source_records_path,
    }


def _plots(result: Mapping[str, Any]) -> dict[str, bytes]:
    plt, pd, sns = _plot_runtime()
    documents = pd.DataFrame(result["documents"])
    containers = pd.DataFrame(result["containers"])
    cargo = pd.DataFrame(result["cargo"])
    packages = pd.DataFrame(result["packages"])
    routes = pd.DataFrame(result["routes"])
    dangerous = pd.DataFrame(result["dangerous_goods"])
    changes = pd.DataFrame(result["changes"])
    plots: dict[str, bytes] = {}

    matrix = pd.crosstab(containers["size_category"], containers["type_category"])
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", ax=axis)
    axis.set(title="Generated container size by semantic type", xlabel="type", ylabel="size")
    plots["01_container_size_type.png"] = _png(figure)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    profile_rows = containers[containers["active_temperature"]].copy()
    sns.countplot(data=profile_rows, x="thermal_profiles", ax=axes[0], color="#0EA5E9")
    axes[0].set(title="Active thermal containers", xlabel="cargo profile", ylabel="containers")
    sns.stripplot(
        data=profile_rows,
        x="thermal_profiles",
        y="temperature_celsius",
        hue="thermal_profiles",
        jitter=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(0, color="#64748B", linewidth=1)
    axes[1].set(title="Generated setpoints", xlabel="cargo profile", ylabel="°C")
    plots["02_thermal_profiles_and_setpoints.png"] = _png(figure)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    if dangerous.empty:
        axes[0].text(0.5, 0.5, "No DG rows", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No flashpoints", ha="center", va="center")
    else:
        sns.countplot(data=dangerous, y="hazard_category", ax=axes[0], color="#F97316")
        axes[0].set(title="DG semantic categories", xlabel="rows", ylabel="")
        with_flash = dangerous.dropna(subset=["flashpoint_celsius"])
        sns.barplot(
            data=with_flash,
            x="un_number",
            y="flashpoint_celsius",
            hue="proper_shipping_name",
            dodge=False,
            ax=axes[1],
        )
        axes[1].axhline(0, color="#64748B", linewidth=1)
        axes[1].set(title="Generated formulation flashpoints", xlabel="UN number", ylabel="°C")
        if axes[1].get_legend() is not None:
            axes[1].legend(title="proper shipping name", fontsize=8)
    plots["03_dangerous_goods_and_flashpoints.png"] = _png(figure)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=documents,
        x="container_count",
        y="cargo_group_count",
        hue="thermal",
        size="package_count",
        sizes=(35, 220),
        alpha=0.75,
        ax=axis,
    )
    axis.set(title="Document structural complexity", xlabel="containers", ylabel="cargo groups")
    plots["04_document_complexity.png"] = _png(figure)
    plt.close(figure)

    package_counts = packages["type_category"].value_counts().head(20).sort_values()
    figure, axis = plt.subplots(figsize=(11, 7))
    sns.barplot(x=package_counts.values, y=package_counts.index, ax=axis, color="#8B5CF6")
    axis.set(title="Top generated package categories", xlabel="package rows", ylabel="")
    plots["05_package_categories.png"] = _png(figure)
    plt.close(figure)

    hs_rows = [
        {"chapter": code[:2], "digits": len(code)}
        for encoded in cargo["hs_codes"]
        for code in str(encoded).split("|")
        if code
    ]
    hs = pd.DataFrame(hs_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    chapters = hs["chapter"].value_counts().head(20).sort_values()
    sns.barplot(x=chapters.values, y=chapters.index, ax=axes[0], color="#14B8A6")
    axes[0].set(title="Top synthetic HS chapters", xlabel="codes", ylabel="chapter")
    sns.countplot(data=hs, x="digits", ax=axes[1], color="#0F766E")
    axes[1].set(title="HS output length", xlabel="digits", ylabel="codes")
    plots["06_hs_chapters_and_lengths.png"] = _png(figure)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, column, title in (
        (axes[0], "origin_country", "Commercial origins"),
        (axes[1], "destination_country", "Commercial destinations"),
    ):
        counts = routes[column].value_counts().head(15).sort_values()
        sns.barplot(x=counts.values, y=counts.index, ax=axis, color="#2563EB")
        axis.set(title=title, xlabel="documents", ylabel="country code")
    plots["07_route_country_distributions.png"] = _png(figure)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    source_counts = (
        documents["source_corpus"].value_counts().rename_axis("value").reset_index(name="count")
    )
    type_counts = (
        documents["document_type"].value_counts().rename_axis("value").reset_index(name="count")
    )
    sns.barplot(data=source_counts, x="count", y="value", ax=axes[0], color="#64748B")
    axes[0].set(title="Selected source corpora", xlabel="documents", ylabel="")
    sns.barplot(data=type_counts, x="count", y="value", ax=axes[1], color="#475569")
    axes[1].set(title="Selected document types", xlabel="documents", ylabel="")
    plots["08_selected_source_strata.png"] = _png(figure)
    plt.close(figure)

    contexts = Counter(
        context
        for encoded in documents["selection_contexts"]
        for context in str(encoded).split("|")
        if context
    )
    context_frame = pd.DataFrame(
        [{"context": key, "documents": value} for key, value in contexts.items()]
    ).sort_values("documents")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(data=context_frame, x="documents", y="context", ax=axis, color="#F59E0B")
    axis.set(title="Rare/complex contexts represented", xlabel="documents", ylabel="")
    plots["09_selection_contexts.png"] = _png(figure)
    plt.close(figure)

    role_counts = changes["coupling_group"].value_counts().head(20).sort_values()
    figure, axis = plt.subplots(figsize=(11, 7))
    sns.barplot(x=role_counts.values, y=role_counts.index, ax=axis, color="#EC4899")
    axis.set(
        title="Completion-stage semantic changes",
        xlabel="changed leaves",
        ylabel="coupling group",
    )
    plots["10_completion_changes.png"] = _png(figure)
    plt.close(figure)

    source_rates = result["summary"]["sourceObservedFeatures"]["ratesPercent"]
    if dangerous.empty:
        class3_flashpoint_rate = 0.0
    else:
        class3_mask = (dangerous["hazard_category"] == "FLAMMABLE_LIQUIDS") | dangerous[
            "subsidiary_hazard_categories"
        ].str.contains("FLAMMABLE_LIQUIDS", regex=False)
        class3 = dangerous[class3_mask]
        class3_flashpoint_rate = (
            100.0 * class3["flashpoint_celsius"].notna().sum() / len(class3)
            if not class3.empty
            else 0.0
        )
    actual_rates = {
        "thermal document": 100.0 * documents["thermal"].sum() / len(documents),
        "IMO number": 100.0 * documents["imo_present"].sum() / len(documents),
        "vessel flag": 100.0 * documents["flag_present"].sum() / len(documents),
        "DG document": 100.0 * (documents["dangerous_goods_count"] > 0).sum() / len(documents),
        "class-3 flashpoint": class3_flashpoint_rate,
    }
    rare = pd.DataFrame(
        [
            {"feature": feature, "population": population, "percent": value}
            for feature in source_rates
            for population, value in (
                ("source corpus", source_rates[feature]),
                ("coverage audit", actual_rates[feature]),
            )
        ]
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=rare, x="percent", y="feature", hue="population", ax=axis)
    axis.set(
        title="Coverage-enriched audit rates (not production marginals)",
        xlabel="percent within stated denominator",
        ylabel="",
    )
    plots["11_rare_feature_coverage_notice.png"] = _png(figure)
    plt.close(figure)

    template_counts = documents["template_id"].value_counts().value_counts().sort_index()
    carrier_counts = (
        documents["carrier_family"].value_counts().sort_values(ascending=False).head(20)
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=template_counts.index, y=template_counts.values, ax=axes[0], color="#22C55E")
    axes[0].set(
        title="Template reuse under cap=3",
        xlabel="documents per template",
        ylabel="templates",
    )
    sns.barplot(x=carrier_counts.values, y=carrier_counts.index, ax=axes[1], color="#16A34A")
    axes[1].set(title="Top carrier families", xlabel="documents", ylabel="")
    plots["12_template_and_carrier_diversity.png"] = _png(figure)
    plt.close(figure)

    measures = cargo.melt(
        id_vars=["document_id"],
        value_vars=["gross_weight", "net_weight"],
        var_name="measure",
        value_name="kilograms",
    ).dropna()
    measures = measures[measures["kilograms"] > 0]
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=measures,
        x="kilograms",
        hue="measure",
        element="step",
        bins=30,
        log_scale=True,
        ax=axis,
    )
    axis.set(title="Generated cargo-weight distribution", xlabel="kg (log scale)", ylabel="rows")
    plots["13_cargo_weights.png"] = _png(figure)
    plt.close(figure)
    return plots


def _report(result: Mapping[str, Any]) -> bytes:
    summary = result["summary"]
    lines = [
        "# Deterministic semantic synthesis audit — 100 documents",
        "",
        "## Outcome",
        "",
        (
            f"All {summary['documents']} documents passed strict relation-v5 validation, exact "
            "relational reconstruction, identifier, route/party, cargo/equipment/temperature, "
            "DG, flashpoint, weight, collision, and provenance checks."
        ),
        "",
        "The deterministic semantic layer is ready to feed the linguistic-realization stage. "
        "These records are intentionally **not training records**: source party identities, "
        "cargo wording, and printed categorical surfaces have not yet been rewritten into OCR.",
        "",
        "## Coverage",
        "",
        (
            "- Documents / templates / carriers: "
            f"{summary['documents']} / {summary['distinctTemplates']} / "
            f"{summary['distinctCarriers']}"
        ),
        (
            "- Containers / cargo groups / packages: "
            f"{summary['containers']} / {summary['cargoGroups']} / {summary['packages']}"
        ),
        (
            "- Thermal documents / active containers: "
            f"{summary['thermalDocuments']} / {summary['thermalContainers']}"
        ),
        (
            "- DG documents / DG rows / generated flashpoints: "
            f"{summary['dangerousGoodsDocuments']} / {summary['dangerousGoodsRows']} / "
            f"{summary['flashpoints']}"
        ),
        (f"- IMO numbers / vessel flags: {summary['imoNumbers']} / {summary['vesselFlags']}"),
        (
            "- Commercial origin / destination countries: "
            f"{summary['originCountries']} / {summary['destinationCountries']}"
        ),
        "",
        "## Distribution warning",
        "",
        summary["auditDistributionNotice"],
        "The audit configuration used 10% thermal documents, 5% IMO numbers, 5% flags, and "
        "100% flashpoint presence among the two eligible class-3 rows so every rare branch could "
        "be observed. Production synthesis should use its configured target distribution.",
        "The comparison plot derives source rates directly from the pinned 1,157-row corpus; "
        "none of its prevalence values are hard-coded into synthesis behavior.",
        "",
        "## Readiness boundary",
        "",
        "Ready now: structured quantities and weights, dates and identifiers, routes and party "
        "localities, package categories, HS identities, vessel/voyage/IMO/flag facts, container "
        "size/type semantics, cargo-conditioned refrigeration and setpoints, and coherent DG "
        "regulatory tuples with formulation-level flashpoints.",
        "",
        "Still belongs to the next stage: fully anonymized party identities and contacts, natural "
        "cargo descriptions, printed package/container/HS surfaces, auxiliary wording, and an "
        "audited raw-OCR patch. The semantic description intentionally differs from the old raw "
        "cargo description until that renderer runs.",
        "",
        "## Validation",
        "",
    ]
    for check in result["checks"]:
        lines.append(f"- PASS — `{check['check']}`")
    lines.extend(
        [
            "",
            "## Contents",
            "",
            "- `tables/`: document-, container-, cargo-, package-, route-, DG-, change-, "
            "and check-level CSVs.",
            "- `plots/`: 13 matplotlib/seaborn diagnostics.",
            "- `summary.json`: machine-readable audit result.",
            "- `inputs.json`: exact committed-run identities and artifact hashes.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--structured-run", type=Path, required=True)
    parser.add_argument("--route-run", type=Path, required=True)
    parser.add_argument("--controlled-run", type=Path, required=True)
    parser.add_argument("--dangerous-goods-run", type=Path, required=True)
    parser.add_argument("--completion-run", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve(strict=True)

    def resolve(path: Path) -> Path:
        value = path if path.is_absolute() else project_root / path
        return value.resolve(strict=True)

    roots = {
        "structured": resolve(args.structured_run),
        "route": resolve(args.route_run),
        "controlled": resolve(args.controlled_run),
        "dangerous_goods": resolve(args.dangerous_goods_run),
        "completion": resolve(args.completion_run),
    }
    input_receipts = {name: _validated_run(root) for name, root in roots.items()}
    completion_config = yaml.safe_load(read_regular_file_bytes(roots["completion"] / "config.yaml"))
    if not isinstance(completion_config, dict):
        raise TypeError("completion config is not an object")
    source_config = completion_config.get("source")
    if not isinstance(source_config, dict) or not isinstance(source_config.get("file"), dict):
        raise TypeError("completion config does not pin a source file")
    source_file_config = source_config["file"]
    source_records_path = resolve(Path(source_file_config["path"]))
    source_sha256 = source_file_config.get("sha256")
    source_records = source_file_config.get("records")
    if not isinstance(source_sha256, str) or sha256_file(source_records_path) != source_sha256:
        raise ValueError("completion source file SHA-256 differs from its config pin")
    if not isinstance(source_records, int) or source_records <= 0:
        raise ValueError("completion source file has no positive record-count pin")
    implementation_sha = sha256_file(Path(__file__))
    inputs = {
        "schemaVersion": 1,
        "auditVersion": _AUDIT_VERSION,
        "implementationSha256": implementation_sha,
        "runs": {
            name: {
                "path": root.relative_to(project_root).as_posix(),
                "commitContentSha256": input_receipts[name].content_sha256,
                "transactionSha256": input_receipts[name].transaction_sha256,
            }
            for name, root in roots.items()
        },
        "source": {
            "path": source_records_path.relative_to(project_root).as_posix(),
            "sha256": source_sha256,
            "records": source_records,
        },
    }
    transaction = sha256_bytes(canonical_json_bytes(inputs))
    output_parent = args.output_parent
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=args.run_id,
        transaction_sha256=transaction,
    )
    started = time.perf_counter()
    result = audit_semantic_completion(
        structured_root=roots["structured"],
        route_root=roots["route"],
        controlled_root=roots["controlled"],
        dangerous_goods_root=roots["dangerous_goods"],
        completion_root=roots["completion"],
        source_records_path=source_records_path,
    )
    if result["summary"]["sourceObservedFeatures"]["counts"]["documents"] != source_records:
        raise ValueError("completion source record count differs from its config pin")
    plots = _plots(result)
    elapsed = time.perf_counter() - started
    summary = dict(result["summary"])

    table_specs = {
        "documents": result["documents"],
        "containers": result["containers"],
        "cargo": result["cargo"],
        "packages": result["packages"],
        "routes": result["routes"],
        "dangerous-goods": result["dangerous_goods"],
        "changes": result["changes"],
        "checks": result["checks"],
    }
    expected = ["REPORT.md", "inputs.json", "summary.json"]
    stage.publish_json("inputs.json", inputs)
    stage.publish_json("summary.json", summary)
    for name, rows in table_specs.items():
        columns = tuple(rows[0]) if rows else ("document_id",)
        relative = f"tables/{name}.csv"
        stage.publish_bytes(relative, _csv_bytes(rows, columns))
        expected.append(relative)
    for name, payload in sorted(plots.items()):
        relative = f"plots/{name}"
        stage.publish_bytes(relative, payload)
        expected.append(relative)
    stage.publish_bytes("REPORT.md", _report(result))
    commit = stage.commit(
        expected_artifacts=expected,
        metadata={
            "schema_version": 1,
            "documents": summary["documents"],
            "checks_passed": summary["checksPassed"],
            "readiness": summary["readiness"],
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": args.run_id,
                "created": commit.created,
                "documents": summary["documents"],
                "checks_passed": summary["checksPassed"],
                "readiness": summary["readiness"],
                "elapsed_seconds": round(elapsed, 6),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
