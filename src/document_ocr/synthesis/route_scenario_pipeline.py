"""Immutable pilot runner for route-first B/L scenario generation."""

from __future__ import annotations

import json
import math
import resource
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import ArtifactReadError, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, identity_sha256, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingRelationExplicitLabel
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER
from document_ocr.synthesis.config import SynthesisRouteScenarioPilotConfig
from document_ocr.synthesis.country_registry import load_iso_country_registry
from document_ocr.synthesis.fit_partition import fit_document_ids
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.locality_registry import (
    GeoNamesLocalityReceipt,
    load_geonames_locality_registry,
)
from document_ocr.synthesis.route_registry import (
    UnlocodeRegistryReceipt,
    load_pinned_unlocode_locations,
    project_route_locations,
)
from document_ocr.synthesis.routes import load_pinned_trade_flows
from document_ocr.synthesis.run_safety import StagedArtifactRun, TrainOnlySourceScope
from document_ocr.synthesis.shipment_scenarios import (
    ShipmentScenario,
    build_scenario_support,
    project_scenario_target,
    sample_shipment_scenario,
)
from document_ocr.synthesis.template_integrity import source_template_integrity_issues
from document_ocr.synthesis.trade_flow_registry import WitsTradeFlowReceipt
from document_ocr.synthesis.world_port_registry import (
    WorldPortRegistryReceipt,
    load_pinned_world_port_records,
)


class RouteScenarioPipelineError(RuntimeError):
    """The route pilot cannot meet its pinned, non-publishable contract."""


def _publish_runtime_once(
    stage: StagedArtifactRun,
    payload: Mapping[str, float],
) -> dict[str, float]:
    """Publish operational timing once and preserve it on exact resume.

    Runtime is observational rather than a deterministic generation result. A
    resumed transaction must revalidate the originally committed observation,
    not conflict merely because the second invocation took a different time.
    """

    root = stage.final_root if stage.completed else stage.stage_root
    path = root / "runtime.json"
    if path.exists() or path.is_symlink():
        try:
            value = json.loads(read_regular_file_bytes(path))
        except (ArtifactReadError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RouteScenarioPipelineError(
                "runtime receipt is not valid immutable JSON"
            ) from error
        if not isinstance(value, dict) or set(value) != {"elapsedSeconds", "peakRssMiB"}:
            raise RouteScenarioPipelineError("runtime receipt has an invalid field contract")
        parsed: dict[str, float] = {}
        for field in ("elapsedSeconds", "peakRssMiB"):
            observed = value[field]
            if type(observed) not in {int, float}:
                raise RouteScenarioPipelineError(f"runtime receipt {field} is not numeric")
            number = float(observed)
            if not math.isfinite(number) or number < 0:
                raise RouteScenarioPipelineError(f"runtime receipt {field} is invalid")
            parsed[field] = number
        return parsed
    runtime = {field: float(value) for field, value in payload.items()}
    stage.publish_json("runtime.json", runtime)
    return runtime


def _resolve_file(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _resolve_directory(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink() or not path.resolve(strict=True).is_dir():
        raise ValueError(f"{label} must be a plain directory")
    return path.resolve(strict=True)


def _read_jsonl(
    path: Path, *, expected_sha256: str, expected_records: int, label: str
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank rows are forbidden")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(value)
    if len(rows) != expected_records:
        raise ValueError(f"{label} expected {expected_records} rows, found {len(rows)}")
    return rows


def _read_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _validate_world_port_dependency(
    *,
    receipt: WorldPortRegistryReceipt,
    receipt_path: Path,
    whitelist_path: Path,
    expected_whitelist_sha256: str,
    expected_whitelist_records: int,
    unlocode_receipt: UnlocodeRegistryReceipt,
    unlocode_receipt_path: Path,
    unlocode_locations_path: Path,
    expected_unlocode_sha256: str,
    expected_unlocode_records: int,
) -> None:
    """Prove that WPI was compiled against this exact pinned UN/LOCODE artifact."""

    artifact = receipt.artifacts[0]
    unlocode_pin = receipt.unlocode_artifact
    if (
        artifact.sha256 != expected_whitelist_sha256
        or artifact.records != expected_whitelist_records
        or artifact.bytes != whitelist_path.stat().st_size
        or whitelist_path.name != artifact.path
        or whitelist_path.parent != receipt_path.parent
        or unlocode_pin.sha256 != expected_unlocode_sha256
        or unlocode_pin.records != expected_unlocode_records
        or unlocode_pin.bytes != unlocode_locations_path.stat().st_size
        or unlocode_pin.release != unlocode_receipt.source.release
        or unlocode_pin.run_name != unlocode_receipt_path.parent.name
        or unlocode_pin.relative_path
        != f"{unlocode_receipt_path.parent.name}/{unlocode_locations_path.name}"
    ):
        raise ValueError(
            "world-port pin/path/count or UN/LOCODE dependency differs from its receipt"
        )


def _validate_locality_dependency(
    *, receipt: GeoNamesLocalityReceipt, expected_iso3166_sha256: str
) -> None:
    """Prevent a locality registry compiled under another country identity snapshot."""

    if receipt.iso3166_sha256 != expected_iso3166_sha256:
        raise ValueError("GeoNames locality registry ISO-3166 pin differs from route pilot")


def _template_maps(
    rows: Sequence[Mapping[str, Any]], corpus_ids: frozenset[str]
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    by_document: dict[str, str] = {}
    members_by_template: dict[str, tuple[str, ...]] = {}
    for row_number, row in enumerate(rows, start=1):
        template_id = row.get("template_id")
        members = row.get("member_document_ids")
        if not isinstance(template_id, str) or not template_id or not isinstance(members, list):
            raise ValueError(f"template row {row_number} has an invalid identity contract")
        if template_id in members_by_template:
            raise ValueError(f"duplicate template ID: {template_id}")
        if any(not isinstance(value, str) or not value for value in members):
            raise ValueError(f"template {template_id} has an invalid member")
        frozen = tuple(members)
        if len(frozen) != len(set(frozen)):
            raise ValueError(f"template {template_id} has duplicate members")
        members_by_template[template_id] = frozen
        for document_id in frozen:
            if document_id in by_document:
                raise ValueError(f"document occurs in multiple templates: {document_id}")
            by_document[document_id] = template_id
    if set(by_document) != corpus_ids:
        raise ValueError("template groups do not cover the source corpus exactly")
    return by_document, members_by_template


def _partition_ids(report: Mapping[str, Any], split: str) -> tuple[str, ...]:
    return fit_document_ids(report, expected_split=split)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _eligible_base_document(target: Mapping[str, Any]) -> tuple[bool, str | None]:
    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return False, "missing_document_patch"
    route = patch.get("route")
    if not isinstance(route, Mapping):
        return False, "missing_route"
    if route.get("transshipmentPort") is not None:
        return False, "transshipment_requires_connectivity_provider"
    if route.get("portOfLoading") is None or route.get("portOfDischarge") is None:
        return False, "missing_physical_route_endpoint"
    parties = patch.get("parties")
    if not isinstance(parties, Mapping) or not isinstance(parties.get("shipper"), Mapping):
        return False, "missing_shipper"
    return True, None


def _validate_pinned_documents(
    *,
    pinned_document_ids: Sequence[str],
    candidate_ids: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    raw_texts: Mapping[str, str],
    template_by_document: Mapping[str, str],
    requested: int,
    maximum_per_template: int,
) -> tuple[str, ...]:
    """Validate an upstream selection without reordering or silently replacing rows."""

    selected = tuple(pinned_document_ids)
    if len(selected) != requested:
        raise RouteScenarioPipelineError(
            f"pinned route selection has {len(selected)} rows; expected {requested}"
        )
    if len(selected) != len(set(selected)):
        raise RouteScenarioPipelineError("pinned route selection contains duplicate documents")
    candidates = frozenset(candidate_ids)
    outside = tuple(document_id for document_id in selected if document_id not in candidates)
    if outside:
        raise RouteScenarioPipelineError(
            f"pinned route selection lies outside the isolated fit scope: {outside!r}"
        )
    templates = tuple(template_by_document[document_id] for document_id in selected)
    template_counts = Counter(templates)
    excess_templates = tuple(
        sorted(
            (template_id, count)
            for template_id, count in template_counts.items()
            if count > maximum_per_template
        )
    )
    if excess_templates:
        raise RouteScenarioPipelineError(
            "pinned route selection exceeds maximum_per_template="
            f"{maximum_per_template}: {excess_templates!r}"
        )
    failures: list[tuple[str, str]] = []
    for document_id in selected:
        accepted, route_reason = _eligible_base_document(targets[document_id])
        if not accepted and route_reason is not None:
            failures.append((document_id, route_reason))
        failures.extend(
            (document_id, reason)
            for reason in source_template_integrity_issues(
                raw_texts[document_id], targets[document_id]
            )
        )
    if failures:
        raise RouteScenarioPipelineError(
            f"pinned route selection contains unsupported documents: {failures!r}"
        )
    return selected


def _validate_projected_target(*, document_id: str, target: dict[str, Any]) -> None:
    canonical = BillOfLadingRelationExplicitLabel.model_validate_json(
        canonical_json_bytes(target), strict=True
    )
    if canonical.canonical_target() != target:
        raise RouteScenarioPipelineError(f"projected route target is not canonical: {document_id}")
    tables = ADAPTER.project(document_id=document_id, source_row_index=0, target=target)
    reconstructed = ADAPTER.reconstruct(document_id=document_id, tables=tables.rows)
    if reconstructed != target:
        raise RouteScenarioPipelineError(
            f"projected route target fails relational inverse: {document_id}"
        )


def _scenario_distribution(scenarios: Sequence[ShipmentScenario]) -> dict[str, Any]:
    def counts(values: Sequence[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    party_relations = Counter[str]()
    party_locality_modes = Counter[str]()
    port_sources = Counter[str]()
    for scenario in scenarios:
        port_sources[scenario.loading_port.source] += 1
        port_sources[scenario.discharge_port.source] += 1
        for party in scenario.party_localities:
            party_relations[f"{party.role}:{party.relation}"] += 1
            party_locality_modes[f"{party.role}:{party.locality_mode}"] += 1
    return {
        "documents": len(scenarios),
        "commercialOriginCountries": counts(
            [row.commercial_origin_country_code for row in scenarios]
        ),
        "commercialOriginPriorComponents": counts(
            [row.origin_prior_component for row in scenarios]
        ),
        "commercialDestinationCountries": counts(
            [row.commercial_destination_country_code for row in scenarios]
        ),
        "loadingCountries": counts([row.loading_port.country_code for row in scenarios]),
        "dischargeCountries": counts([row.discharge_port.country_code for row in scenarios]),
        "portSourceCounts": dict(sorted(port_sources.items())),
        "partyRelationCounts": dict(sorted(party_relations.items())),
        "partyLocalityModeCounts": dict(sorted(party_locality_modes.items())),
        "freightArrangementCounts": counts(
            [row.freight.arrangement or "missing" for row in scenarios]
        ),
        "vesselStatusCounts": counts([row.vessel_status for row in scenarios]),
    }


def run_route_scenario_pilot(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRouteScenarioPilotConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_path = _resolve_file(project_root, config.source.file.path, label="source corpus")
    source_rows = _read_jsonl(
        source_path,
        expected_sha256=config.source.file.sha256,
        expected_records=config.source.file.records,
        label="source corpus",
    )
    document_field = config.source.fields.document_id
    target_field = config.source.fields.target
    input_hash_field = config.source.fields.input_sha256
    if input_hash_field is None:
        raise ValueError("route scenario source requires input_sha256")
    source_by_id: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    raw_texts: dict[str, str] = {}
    targets: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(source_rows, start=1):
        document_id = row.get(document_field)
        target = row.get(target_field)
        input_sha256 = row.get(input_hash_field)
        raw_text = row.get(config.source.fields.input_text)
        if (
            not isinstance(document_id, str)
            or not isinstance(target, dict)
            or not isinstance(input_sha256, str)
            or not isinstance(raw_text, str)
        ):
            raise ValueError(f"source row {row_number} is missing required route fields")
        if document_id in source_by_id:
            raise ValueError(f"duplicate source document: {document_id}")
        source_by_id[document_id] = row
        source_hashes[document_id] = input_sha256
        raw_texts[document_id] = raw_text
        targets[document_id] = target
    corpus_ids = frozenset(source_by_id)

    template_path = _resolve_file(
        project_root, config.inputs.template_groups.path, label="template groups"
    )
    template_rows = _read_jsonl(
        template_path,
        expected_sha256=config.inputs.template_groups.sha256,
        expected_records=config.inputs.template_groups.records,
        label="template groups",
    )
    template_by_document, _members = _template_maps(template_rows, corpus_ids)
    partition_path = _resolve_file(
        project_root, config.inputs.partition_report.path, label="partition report"
    )
    partition = _read_json(
        partition_path,
        expected_sha256=config.inputs.partition_report.sha256,
        label="partition report",
    )
    fit_ids = _partition_ids(partition, config.selection.split)
    scope = TrainOnlySourceScope(
        corpus_document_ids=tuple(sorted(corpus_ids)),
        fit_document_ids=fit_ids,
        template_by_document=template_by_document,
        source_sha256_by_document=source_hashes,
        fit_split=config.selection.split,
    )
    isolated_ids = tuple(sorted(scope.isolated_fit_document_ids))
    scope.assert_fit_rows(isolated_ids, purpose="route scenario support")

    iso_path = _resolve_file(
        project_root, config.inputs.iso3166_snapshot.path, label="ISO-3166 snapshot"
    )
    countries = load_iso_country_registry(
        iso_path=iso_path,
        iso_sha256=config.inputs.iso3166_snapshot.sha256,
    )
    route_locations_path = _resolve_file(
        project_root, config.inputs.route_locations.path, label="route locations"
    )
    route_manifest_path = _resolve_file(
        project_root,
        config.inputs.route_registry_manifest.path,
        label="route registry manifest",
    )
    _read_json(
        route_manifest_path,
        expected_sha256=config.inputs.route_registry_manifest.sha256,
        label="route registry manifest",
    )
    route_manifest = UnlocodeRegistryReceipt.model_validate_json(
        route_manifest_path.read_bytes(), strict=True
    )
    canonical_artifact = next(
        (
            artifact
            for artifact in route_manifest.artifacts
            if artifact.role == "canonical_locations_jsonl"
        ),
        None,
    )
    if canonical_artifact is None or (
        canonical_artifact.sha256 != config.inputs.route_locations.sha256
        or canonical_artifact.records != config.inputs.route_locations.records
        or canonical_artifact.bytes != route_locations_path.stat().st_size
        or route_locations_path.name != canonical_artifact.path
        or route_locations_path.parent != route_manifest_path.parent
    ):
        raise ValueError("route location pin/path/count differs from the compiled registry receipt")
    unlocode_locations = load_pinned_unlocode_locations(
        route_locations_path,
        expected_sha256=config.inputs.route_locations.sha256,
        expected_records=config.inputs.route_locations.records,
    )
    route_projection = project_route_locations(unlocode_locations)
    route_locations = route_projection.locations

    world_ports_path = _resolve_file(
        project_root, config.inputs.world_ports.path, label="world-port whitelist"
    )
    world_port_manifest_path = _resolve_file(
        project_root,
        config.inputs.world_port_registry_manifest.path,
        label="world-port registry manifest",
    )
    _read_json(
        world_port_manifest_path,
        expected_sha256=config.inputs.world_port_registry_manifest.sha256,
        label="world-port registry manifest",
    )
    world_port_manifest = WorldPortRegistryReceipt.model_validate_json(
        world_port_manifest_path.read_bytes(), strict=True
    )
    _validate_world_port_dependency(
        receipt=world_port_manifest,
        receipt_path=world_port_manifest_path,
        whitelist_path=world_ports_path,
        expected_whitelist_sha256=config.inputs.world_ports.sha256,
        expected_whitelist_records=config.inputs.world_ports.records,
        unlocode_receipt=route_manifest,
        unlocode_receipt_path=route_manifest_path,
        unlocode_locations_path=route_locations_path,
        expected_unlocode_sha256=config.inputs.route_locations.sha256,
        expected_unlocode_records=config.inputs.route_locations.records,
    )
    world_ports = load_pinned_world_port_records(
        world_ports_path,
        expected_sha256=config.inputs.world_ports.sha256,
        expected_records=config.inputs.world_ports.records,
    )

    locality_registry_root = _resolve_directory(
        project_root,
        config.inputs.locality_registry.path,
        label="GeoNames locality registry",
    )
    locality_registry = load_geonames_locality_registry(
        root=locality_registry_root,
        expected_receipt_sha256=config.inputs.locality_registry.manifest_sha256,
    )
    locality_manifest = locality_registry.receipt
    _validate_locality_dependency(
        receipt=locality_manifest,
        expected_iso3166_sha256=config.inputs.iso3166_snapshot.sha256,
    )
    localities = tuple(
        locality
        for country_code in locality_registry.country_codes
        for locality in locality_registry.rows_for_country(country_code)
    )

    trade_flows_path = _resolve_file(
        project_root, config.inputs.trade_flows.path, label="bilateral trade flows"
    )
    trade_manifest_path = _resolve_file(
        project_root,
        config.inputs.trade_flow_registry_manifest.path,
        label="trade-flow registry manifest",
    )
    _read_json(
        trade_manifest_path,
        expected_sha256=config.inputs.trade_flow_registry_manifest.sha256,
        label="trade-flow registry manifest",
    )
    trade_manifest = WitsTradeFlowReceipt.model_validate_json(
        trade_manifest_path.read_bytes(), strict=True
    )
    if (
        trade_manifest.iso3166_sha256 != config.inputs.iso3166_snapshot.sha256
        or trade_manifest.jsonl_sha256 != config.inputs.trade_flows.sha256
        or trade_manifest.trade_flow_records != config.inputs.trade_flows.records
        or trade_manifest.jsonl_bytes != trade_flows_path.stat().st_size
        or trade_flows_path.name != "trade-flows.jsonl"
        or trade_flows_path.parent != trade_manifest_path.parent
    ):
        raise ValueError("trade-flow pin/path/count differs from the compiled registry receipt")
    trade_flows = load_pinned_trade_flows(
        trade_flows_path,
        expected_sha256=config.inputs.trade_flows.sha256,
        expected_records=config.inputs.trade_flows.records,
    )

    support = build_scenario_support(
        source_targets=targets,
        fit_document_ids=isolated_ids,
        country_registry=countries,
        route_locations=route_locations,
        world_ports=world_ports,
        localities=localities,
        trade_flows=trade_flows,
        origin_prior=config.generation.commercial_origin_prior,
    )
    upstream_selection_path = _resolve_file(
        project_root,
        config.inputs.upstream_selection.path,
        label="upstream structured selection",
    )
    upstream_selection_rows = _read_jsonl(
        upstream_selection_path,
        expected_sha256=config.inputs.upstream_selection.sha256,
        expected_records=config.inputs.upstream_selection.records,
        label="upstream structured selection",
    )
    upstream_document_ids: list[str] = []
    for row_number, row in enumerate(upstream_selection_rows, start=1):
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"upstream structured selection row {row_number} has no document_id")
        upstream_document_ids.append(document_id)
    selected_ids = _validate_pinned_documents(
        pinned_document_ids=upstream_document_ids,
        candidate_ids=isolated_ids,
        targets=targets,
        raw_texts=raw_texts,
        template_by_document=template_by_document,
        requested=config.selection.requested_documents,
        maximum_per_template=config.selection.maximum_per_template,
    )
    selection_exclusions: dict[str, int] = {}
    selection_method = "pinned_upstream_structured_selection_with_source_integrity_v2"
    scenarios: list[ShipmentScenario] = []
    projected_rows: list[dict[str, Any]] = []
    for document_id in selected_ids:
        scenario = sample_shipment_scenario(
            base_document_id=document_id,
            source_target=targets[document_id],
            support=support,
            country_registry=countries,
            stream=DeterministicStream(
                config.generation.seed,
                "mpci-bl-route-scenario-v2",
                document_id,
            ),
            registry_exploration_permyriad=config.generation.registry_exploration_permyriad,
        )
        projected = project_scenario_target(source_target=targets[document_id], scenario=scenario)
        synthetic_document_id = (
            "syn_"
            + identity_sha256(
                "mpci-bl-route-scenario-document-v1",
                "origin-prior-mixture-v1",
                config.run.run_id,
                document_id,
                config.generation.seed,
            )[:32]
        )
        _validate_projected_target(document_id=synthetic_document_id, target=projected)
        scenarios.append(scenario)
        projected_rows.append(
            {
                "syntheticDocumentId": synthetic_document_id,
                "baseDocumentId": document_id,
                "templateId": template_by_document[document_id],
                "sourceTargetSha256": sha256_bytes(canonical_json_bytes(targets[document_id])),
                "projectedTargetSha256": sha256_bytes(canonical_json_bytes(projected)),
                "target": projected,
                "trainingEligible": False,
            }
        )

    scenario_rows = [
        {
            "syntheticDocumentId": projected["syntheticDocumentId"],
            "templateId": projected["templateId"],
            **scenario.to_dict(),
            "trainingEligible": False,
        }
        for scenario, projected in zip(scenarios, projected_rows, strict=True)
    ]
    distribution = _scenario_distribution(scenarios)
    implementation_sha256 = {
        path.name: sha256_file(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("shipment_scenarios.py"),
            Path(__file__).with_name("country_registry.py"),
            Path(__file__).with_name("route_registry.py"),
            Path(__file__).with_name("world_port_registry.py"),
            Path(__file__).with_name("locality_registry.py"),
            Path(__file__).with_name("trade_flow_registry.py"),
            Path(__file__).with_name("routes.py"),
            Path(__file__).with_name("generators.py"),
            Path(__file__).with_name("template_integrity.py"),
        )
    }
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "contract": "mpci-bl-route-scenario-pilot-v4",
                "configSha256": sha256_file(config_path),
                "config": config.model_dump(mode="json"),
                "sourceSha256": config.source.file.sha256,
                "scopeSha256": scope.scope_sha256,
                "selection": {
                    "method": selection_method,
                    "documentIdsSha256": sha256_bytes(canonical_json_bytes(selected_ids)),
                    "upstreamSelectionSha256": config.inputs.upstream_selection.sha256,
                },
                "registries": {
                    "iso3166SnapshotSha256": config.inputs.iso3166_snapshot.sha256,
                    "unlocodeReceiptFileSha256": (config.inputs.route_registry_manifest.sha256),
                    "unlocodeReceipt": route_manifest.model_dump(mode="json"),
                    "worldPortReceiptFileSha256": (
                        config.inputs.world_port_registry_manifest.sha256
                    ),
                    "worldPortReceipt": world_port_manifest.model_dump(mode="json"),
                    "localityReceiptFileSha256": (config.inputs.locality_registry.manifest_sha256),
                    "localityReceipt": locality_manifest.model_dump(mode="json"),
                    "tradeFlowReceiptFileSha256": (
                        config.inputs.trade_flow_registry_manifest.sha256
                    ),
                    "tradeFlowReceipt": trade_manifest.model_dump(mode="json"),
                },
                "countryRegistryAudit": countries.audit.model_dump(mode="json"),
                "implementation": implementation_sha256,
            }
        )
    )
    output_parent = Path(config.run.output_dir)
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.run_id,
        transaction_sha256=transaction,
    )
    stage.recover_interrupted_temporary_files()
    stage.publish_json(
        "source-scope.json",
        {
            "fitSplit": config.selection.split,
            "partitionDocuments": len(fit_ids),
            "templateIsolatedDocuments": len(isolated_ids),
            "scopeSha256": scope.scope_sha256,
        },
    )
    stage.publish_json(
        "modeling/country-registry-audit.json", countries.audit.model_dump(mode="json")
    )
    stage.publish_json(
        "modeling/route-location-projection-audit.json",
        route_projection.audit.model_dump(mode="json"),
    )
    stage.publish_json(
        "modeling/unlocode-registry-metadata.json",
        {
            "receiptFileSha256": config.inputs.route_registry_manifest.sha256,
            "receipt": route_manifest.model_dump(mode="json"),
        },
    )
    stage.publish_json(
        "modeling/world-port-registry-audit.json",
        world_port_manifest.audit.model_dump(mode="json"),
    )
    stage.publish_json(
        "modeling/world-port-registry-metadata.json",
        {
            "receiptFileSha256": config.inputs.world_port_registry_manifest.sha256,
            "receipt": world_port_manifest.model_dump(mode="json"),
        },
    )
    stage.publish_json(
        "modeling/locality-registry-audit.json",
        locality_registry.audit.model_dump(mode="json"),
    )
    stage.publish_json(
        "modeling/locality-registry-metadata.json",
        {
            "receiptFileSha256": config.inputs.locality_registry.manifest_sha256,
            "receipt": locality_manifest.model_dump(mode="json"),
        },
    )
    stage.publish_json("modeling/scenario-support-audit.json", support.audit.to_dict())
    stage.publish_json("modeling/trade-flow-support-audit.json", support.trade_flow_audit.to_dict())
    stage.publish_json(
        "modeling/trade-flow-registry-metadata.json",
        {
            "receiptFileSha256": config.inputs.trade_flow_registry_manifest.sha256,
            "receipt": trade_manifest.model_dump(mode="json"),
        },
    )
    stage.publish_json("selection/exclusions.json", selection_exclusions)
    stage.publish_json(
        "selection/selection-receipt.json",
        {
            "method": selection_method,
            "upstreamSelectionSha256": config.inputs.upstream_selection.sha256,
            "documentIds": list(selected_ids),
            "documentIdsSha256": sha256_bytes(canonical_json_bytes(selected_ids)),
        },
    )
    stage.publish_bytes("generation/shipment-scenarios.jsonl", _jsonl(scenario_rows))
    stage.publish_bytes("generation/projected-targets.jsonl", _jsonl(projected_rows))
    stage.publish_json("generation/distribution-summary.json", distribution)
    validation = {
        "requestedDocuments": config.selection.requested_documents,
        "generatedScenarios": len(scenarios),
        "strictSchemaValid": len(projected_rows),
        "relationalInverseValid": len(projected_rows),
        "distinctTemplates": len({row["templateId"] for row in projected_rows}),
        "directRoutes": len(scenarios),
        "transshipmentRoutes": 0,
        "trainingRecordsPublished": 0,
    }
    stage.publish_json("generation/validation-summary.json", validation)
    runtime = _publish_runtime_once(
        stage,
        {
            "elapsedSeconds": time.perf_counter() - started,
            "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )
    manifest = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": "route_scenario_pilot_complete_pending_identity_vessel_text",
        "trainingEligible": False,
        "sourceRecords": len(source_rows),
        "fitDocuments": len(isolated_ids),
        "scenarioDocuments": len(scenarios),
        "routeRegistryRecords": len(route_locations),
        "worldPortRegistryRecords": len(world_ports),
        "localityRegistryRecords": len(localities),
        "tradeFlowRegistryRecords": len(trade_flows),
        "registryPins": {
            "iso3166SnapshotSha256": config.inputs.iso3166_snapshot.sha256,
            "unlocodeReceiptFileSha256": config.inputs.route_registry_manifest.sha256,
            "unlocodeSourceSha256": route_manifest.source.source_sha256,
            "unlocodeLocationsSha256": config.inputs.route_locations.sha256,
            "worldPortReceiptFileSha256": (config.inputs.world_port_registry_manifest.sha256),
            "worldPortSourceSha256": world_port_manifest.source.source_sha256,
            "worldPortWhitelistSha256": config.inputs.world_ports.sha256,
            "localityReceiptFileSha256": (config.inputs.locality_registry.manifest_sha256),
            "localitySourceReceiptSha256": locality_manifest.source_receipt_sha256,
            "localityArchiveSha256": locality_manifest.archive_sha256,
            "localityMemberSha256": locality_manifest.member_sha256,
            "localityJsonlSha256": locality_manifest.jsonl_sha256,
            "localitySqliteSha256": locality_manifest.sqlite_sha256,
            "tradeFlowReceiptFileSha256": (config.inputs.trade_flow_registry_manifest.sha256),
            "tradeFlowSourceManifestSha256": trade_manifest.source_manifest_sha256,
            "tradeFlowCountryMetadataSha256": (trade_manifest.wits_country_metadata_sha256),
            "tradeFlowsSha256": config.inputs.trade_flows.sha256,
        },
        "implementationSha256": implementation_sha256,
        "generationMethods": {
            "selection": selection_method,
            "physicalEndpoints": config.generation.physical_endpoint_relation_method,
            "partyLocalities": config.generation.party_locality_relation_method,
            "ports": config.generation.port_method,
            "populatedLocalities": config.generation.locality_method,
        },
        "modelOrApiCalls": 0,
        "transactionSha256": transaction,
        "distribution": distribution,
    }
    stage.publish_json("manifest.json", manifest)
    expected = (
        "generation/distribution-summary.json",
        "generation/projected-targets.jsonl",
        "generation/shipment-scenarios.jsonl",
        "generation/validation-summary.json",
        "manifest.json",
        "modeling/country-registry-audit.json",
        "modeling/locality-registry-audit.json",
        "modeling/locality-registry-metadata.json",
        "modeling/route-location-projection-audit.json",
        "modeling/scenario-support-audit.json",
        "modeling/trade-flow-support-audit.json",
        "modeling/trade-flow-registry-metadata.json",
        "modeling/unlocode-registry-metadata.json",
        "modeling/world-port-registry-audit.json",
        "modeling/world-port-registry-metadata.json",
        "runtime.json",
        "selection/exclusions.json",
        "selection/selection-receipt.json",
        "source-scope.json",
    )
    commit = stage.commit(
        expected_artifacts=expected,
        metadata={
            "status": cast(str, manifest["status"]),
            "trainingEligible": False,
            "scenarioDocuments": len(scenarios),
        },
    )
    return {
        **manifest,
        "commitCreated": commit.created,
        "commitContentSha256": commit.receipt.content_sha256,
        "runtime": runtime,
    }
