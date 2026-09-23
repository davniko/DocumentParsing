"""Build immutable, train-only-fit route scenarios without provider calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import PinnedDirectoryConfig, RouteScenarioOriginPriorConfig
from document_ocr.synthesis.country_registry import CountryRegistry, load_iso_country_registry
from document_ocr.synthesis.locality_registry import load_geonames_locality_registry
from document_ocr.synthesis.route_registry import (
    UnlocodeRegistryReceipt,
    load_pinned_unlocode_locations,
    project_route_locations,
)
from document_ocr.synthesis.route_scenario_pipeline import (
    _validate_locality_dependency,
    _validate_world_port_dependency,
)
from document_ocr.synthesis.routes import load_pinned_trade_flows
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.shipment_scenarios import ScenarioSupport, build_scenario_support
from document_ocr.synthesis.trade_flow_registry import WitsTradeFlowReceipt
from document_ocr.synthesis.transshipment_routes import TransshipmentObservation
from document_ocr.synthesis.world_port_registry import (
    WorldPortRegistryReceipt,
    load_pinned_world_port_records,
)

from . import complete_targets as targets
from . import descendant as render
from .descendant_models import PinnedCommittedRun
from .models import NonEmptyText, PinnedFile, PinnedJsonl
from .pipeline import project_root_from_config, resolve_input
from .route_projection import sample_projection


class RoutePlanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1]
    task: Literal["compiled_template_route_plan_v1"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    template_run: PinnedCommittedRun
    sample_plan_run: PinnedCommittedRun
    sample_plan: PinnedJsonl
    fit_records: PinnedJsonl
    validation_records: PinnedJsonl
    iso3166_snapshot: PinnedFile
    route_locations: PinnedJsonl
    route_registry_manifest: PinnedFile
    world_ports: PinnedJsonl
    world_port_registry_manifest: PinnedFile
    locality_registry: PinnedDirectoryConfig
    trade_flows: PinnedJsonl
    trade_flow_registry_manifest: PinnedFile
    commercial_origin_prior: RouteScenarioOriginPriorConfig
    registry_exploration_permyriad: Annotated[int, Field(ge=0, le=10000)]
    maximum_candidates_per_sample: Annotated[int, Field(gt=0)]
    seed: int
    transshipment_observations: PinnedJsonl | None = None


def _pin(root: Path, pin: PinnedFile | PinnedJsonl) -> Path:
    path = resolve_input(root, pin.path)
    if sha256_file(path) != pin.sha256:
        raise ValueError(f"route-plan dependency hash differs: {pin.path}")
    return path


def load_support(
    root: Path, config: RoutePlanConfig
) -> tuple[CountryRegistry, ScenarioSupport, frozenset[str]]:
    fit = render._read_jsonl(_pin(root, config.fit_records), records=config.fit_records.records)
    validation = render._read_jsonl(
        _pin(root, config.validation_records), records=config.validation_records.records
    )
    fit_ids = [r["documentId"] for r in fit]
    validation_ids = [r["documentId"] for r in validation]
    if len(set(fit_ids)) != len(fit_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("route fit/validation document identities are duplicated")
    if set(fit_ids) & set(validation_ids):
        raise ValueError("validation documents cannot enter route-prior fitting")
    countries = load_iso_country_registry(
        iso_path=_pin(root, config.iso3166_snapshot), iso_sha256=config.iso3166_snapshot.sha256
    )
    route_path = _pin(root, config.route_locations)
    route_receipt_path = _pin(root, config.route_registry_manifest)
    route_receipt = UnlocodeRegistryReceipt.model_validate_json(route_receipt_path.read_bytes())
    canonical = [a for a in route_receipt.artifacts if a.role == "canonical_locations_jsonl"]
    if len(canonical) != 1 or (
        canonical[0].sha256 != config.route_locations.sha256
        or canonical[0].records != config.route_locations.records
        or canonical[0].bytes != route_path.stat().st_size
        or route_path.name != canonical[0].path
        or route_path.parent != route_receipt_path.parent
    ):
        raise ValueError("UN/LOCODE content differs from its registry receipt")
    locations = project_route_locations(
        load_pinned_unlocode_locations(
            route_path,
            expected_sha256=config.route_locations.sha256,
            expected_records=config.route_locations.records,
        )
    ).locations
    world_path = _pin(root, config.world_ports)
    world_receipt_path = _pin(root, config.world_port_registry_manifest)
    world_receipt = WorldPortRegistryReceipt.model_validate_json(world_receipt_path.read_bytes())
    _validate_world_port_dependency(
        receipt=world_receipt,
        receipt_path=world_receipt_path,
        whitelist_path=world_path,
        expected_whitelist_sha256=config.world_ports.sha256,
        expected_whitelist_records=config.world_ports.records,
        unlocode_receipt=route_receipt,
        unlocode_receipt_path=route_receipt_path,
        unlocode_locations_path=route_path,
        expected_unlocode_sha256=config.route_locations.sha256,
        expected_unlocode_records=config.route_locations.records,
    )
    world = load_pinned_world_port_records(
        world_path,
        expected_sha256=config.world_ports.sha256,
        expected_records=config.world_ports.records,
    )
    locality_root = (root / config.locality_registry.path).resolve(strict=True)
    if (
        root not in locality_root.parents
        or locality_root.is_symlink()
        or not locality_root.is_dir()
    ):
        raise ValueError("locality registry must be a plain directory inside the project")
    localities = load_geonames_locality_registry(
        root=locality_root,
        expected_receipt_sha256=config.locality_registry.manifest_sha256,
    )
    _validate_locality_dependency(
        receipt=localities.receipt, expected_iso3166_sha256=config.iso3166_snapshot.sha256
    )
    trade_path = _pin(root, config.trade_flows)
    trade_receipt_path = _pin(root, config.trade_flow_registry_manifest)
    trade_receipt = WitsTradeFlowReceipt.model_validate_json(trade_receipt_path.read_bytes())
    if (
        trade_receipt.iso3166_sha256 != config.iso3166_snapshot.sha256
        or trade_receipt.jsonl_sha256 != config.trade_flows.sha256
        or trade_receipt.trade_flow_records != config.trade_flows.records
        or trade_receipt.jsonl_bytes != trade_path.stat().st_size
        or trade_path.parent != trade_receipt_path.parent
        or trade_path.name != "trade-flows.jsonl"
    ):
        raise ValueError("trade-flow registry dependencies differ")
    trade = load_pinned_trade_flows(
        trade_path,
        expected_sha256=config.trade_flows.sha256,
        expected_records=config.trade_flows.records,
    )
    support = build_scenario_support(
        source_targets={r["documentId"]: r["target"] for r in fit},
        fit_document_ids=sorted(fit_ids),
        country_registry=countries,
        route_locations=locations,
        world_ports=world,
        localities=tuple(
            row for code in localities.country_codes for row in localities.rows_for_country(code)
        ),
        trade_flows=trade,
        origin_prior=config.commercial_origin_prior,
        transshipment_observations=(
            tuple(
                TransshipmentObservation.model_validate(row, strict=True)
                for row in render._read_jsonl(
                    _pin(root, config.transshipment_observations),
                    records=config.transshipment_observations.records,
                )
            )
            if config.transshipment_observations is not None
            else ()
        ),
    )
    return countries, support, frozenset(validation_ids)


def run(config_path: Path) -> Path:
    root = project_root_from_config(config_path)
    config = RoutePlanConfig.model_validate_json(
        json.dumps(yaml.safe_load(read_regular_file_bytes(config_path)))
    )
    template_root = render._validate_committed_run(root, config.template_run)
    plan_root = render._validate_committed_run(root, config.sample_plan_run)
    plan_file = _pin(root, config.sample_plan)
    if plan_root not in plan_file.parents:
        raise ValueError("sample plan is outside its committed run")
    samples = render._read_jsonl(plan_file, records=config.sample_plan.records)
    if len({r["sampleId"] for r in samples}) != len(samples):
        raise ValueError("duplicate route-plan sample identity")
    countries, support, validation_ids = load_support(root, config)
    if {r["sourceDocumentId"] for r in samples} & validation_ids:
        raise ValueError("route synthesis includes a validation source template")
    codes = render._country_code_map(_pin(root, config.iso3166_snapshot))
    sources = {
        sid: targets.load_source(template_root, sid)
        for sid in sorted({r["sourceDocumentId"] for r in samples})
    }
    implementation = {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted((root / "src/document_ocr").rglob("*.py"))
    }
    transaction = {"config": config.model_dump(mode="json"), "implementationFiles": implementation}
    stage = StagedArtifactRun(
        output_parent=root / config.output_dir,
        run_name=config.run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    stage.recover_interrupted_temporary_files()
    projections = []
    failures = []
    for index, sample in enumerate(samples):
        try:
            projection = sample_projection(
                sources[sample["sourceDocumentId"]],
                sample_id=sample["sampleId"],
                seed=config.seed,
                support=support,
                countries=countries,
                country_codes=codes,
                registry_exploration_permyriad=config.registry_exploration_permyriad,
                maximum_candidates=config.maximum_candidates_per_sample,
            )
        except ValueError as error:
            failures.append(
                {
                    "sampleId": sample["sampleId"],
                    "sourceDocumentId": sample["sourceDocumentId"],
                    "error": str(error),
                }
            )
            continue
        projections.append(projection.model_dump(mode="json"))
        if index % 100 == 0:
            print(json.dumps({"routeSamples": index + 1, "total": len(samples)}), flush=True)
    if any(sha256_file(root / p) != h for p, h in implementation.items()):
        raise ValueError("implementation changed during route planning")
    if failures:
        stage.publish_json("rejected-samples.json", failures)
        raise ValueError(
            f"{len(failures)} route samples require review; nothing published. "
            f"See {stage.stage_root / 'rejected-samples.json'}"
        )
    stage.publish_json("transaction.json", transaction)
    stage.publish_json("fit-audit.json", support.audit.to_dict())
    stage.publish_json(
        "summary.json",
        {
            "samples": len(projections),
            "fitRecords": config.fit_records.records,
            "validationOverlap": 0,
            "providerRequests": 0,
            "originCountries": dict(
                Counter(r["scenario"]["commercialOriginCountryCode"] for r in projections)
            ),
            "destinationCountries": dict(
                Counter(r["scenario"]["commercialDestinationCountryCode"] for r in projections)
            ),
            "transshipmentDocuments": sum(
                r["scenario"]["transshipmentStatus"] != "not_present" for r in projections
            ),
            "transshipmentChains": dict(
                Counter(
                    r["scenario"]["transshipmentEvidenceDocumentId"]
                    for r in projections
                    if r["scenario"]["transshipmentEvidenceDocumentId"] is not None
                )
            ),
            "rejectedCandidates": sum(len(r["rejected_candidates"]) for r in projections),
        },
    )
    stage.publish_bytes(
        "projections.jsonl", b"".join(canonical_json_bytes(r) + b"\n" for r in projections)
    )
    stage.commit(
        expected_artifacts=[
            "transaction.json",
            "fit-audit.json",
            "summary.json",
            "projections.jsonl",
        ],
        metadata={"samples": len(projections), "providerRequests": 0},
    )
    return stage.final_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config.resolve()), flush=True)


if __name__ == "__main__":
    main()
