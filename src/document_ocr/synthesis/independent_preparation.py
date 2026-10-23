"""Goods-first native preparation for the donor-independent flat campaign.

All requested slots remain in the journal. Only complete source groups receive
a runnable configuration; a failed sibling is never replaced or renumbered.
The ordinary Campaign is used to compile source ownership, never to sample cargo.
"""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence, Set
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Never, cast

from pydantic import BaseModel

from document_ocr.atomic import (
    ArtifactReadError,
    AtomicConflictError,
    atomic_publish_json,
    read_regular_file_bytes,
)
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated import Runner, digest
from document_ocr.synthesis.curated_auxiliary import validate_identifier_privacy
from document_ocr.synthesis.curated_campaign import Campaign
from document_ocr.synthesis.curated_equipment_inventory import (
    compile_equipment_inventory,
    rebase_equipment_inventory_headers,
)
from document_ocr.synthesis.curated_private_contacts import validate_private_person_coverage
from document_ocr.synthesis.curated_scenarios import (
    FilePin,
    SourceCapabilities,
    _pin,
)
from document_ocr.synthesis.independent_accounting import (
    MIXED_PACKING_SYSTEM,
    MixedPackingPlan,
    MixedPackingRequest,
    mixed_packing_facts,
    mixed_packing_prompt,
)
from document_ocr.synthesis.independent_campaign import (
    IndependentCampaign,
    source_equipment_readiness,
)
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
)
from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates
from document_ocr.synthesis.independent_packages import (
    PACKAGE_CHOICE_SYSTEM,
    PackageChoice,
    PackageChoiceRequest,
    equipment_choice_brief,
    package_choice_output_type,
    package_choice_prompt,
    package_coverage,
    select_package_type,
    validate_package_choice,
)
from document_ocr.synthesis.independent_presentation import (
    load_distinct_actor_reviews,
    load_locality_reviews,
)
from document_ocr.synthesis.independent_relationships import (
    PRIVATE_PACKING_SYSTEM,
    PrivatePackingPlan,
    PrivatePackingRequest,
    private_packing_facts,
    private_packing_output_type,
    private_packing_prompt,
    validate_private_packing_plan,
)
from document_ocr.synthesis.independent_sampling import IndependentSampler, sample_equipment
from document_ocr.synthesis.independent_wording import (
    Certificate,
    load_certificates,
    prepare_blueprint,
)
from document_ocr.synthesis.tank_equipment import TANK_TYPES

type JsonObject = dict[str, Any]
type NativeJob = tuple[
    str, type[BaseModel], str, str, Callable[[JsonObject], BaseModel], JsonObject
]

_AUTHORITIES = frozenset(
    {
        "registry",
        "numeric_certificates",
        "wording_certificates",
        "geography_certificates",
        "locality_owner_reviews",
        "distinct_actor_reviews",
    }
)
_REGIMES = {
    "ambient": "ambient",
    "vehicle": "ambient",
    "dg_vehicle": "dg",
    "dg_chemical": "dg",
    "frozen": "frozen",
    "chilled": "chilled",
}
_TYPE_PATH = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeCategory"


class _SourceOwnershipCampaign(Campaign):
    """Ordinary source compiler with observed-cargo fitting disabled."""

    sampling_mode = "independent"

    def plan(self, sid: str, variant: int) -> Never:
        raise RuntimeError("source preparation may compile blueprints, never sample observed cargo")


@dataclass(frozen=True)
class _SourceContext:
    source: _SourceOwnershipCampaign
    sampler: IndependentSampler
    numeric: IndependentNumericCertificates
    certificates: dict[str, JsonObject]
    wording: dict[str, Certificate]
    geography: IndependentGeographyGate
    pins: dict[str, str | None]


@dataclass(frozen=True)
class _CompiledSource:
    capability: SourceCapabilities
    regime: str
    dg_requirements: dict[str, bool | int] | None
    relations: tuple[str, ...]
    fixed_mix: bool
    container_count: int
    equipment_domain: list[dict[str, str]] | None
    equipment_slots: Sequence[Mapping[str, str]] | None


class _SourceNotReady(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("sampling_mode") != "independent" or set(config["independent"]) != _AUTHORITIES:
        raise ValueError(
            "preparation needs independent mode and exactly six unprepared authorities"
        )


def _validate_mixed_scope(config: Mapping[str, Any], mixed_source_ids: Collection[str]) -> set[str]:
    mixed = set(mixed_source_ids)
    if len(mixed) != len(mixed_source_ids) or not mixed <= set(config["source_ids"]):
        raise ValueError("mixed source scope must be distinct and selected")
    return mixed


def _output_directory(root: Path, config: Mapping[str, Any]) -> Path:
    output = (root / cast(str, config["output"])).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("preparation output must be a specific directory inside the project")
    return output


def _load_source_context(root: Path, config: dict[str, Any]) -> _SourceContext:
    """Validate global authority files outside per-source readiness handling."""
    independent = config["independent"]
    # Each loader validates its own FilePins and schema once. A corrupt global
    # authority is fatal, never hundreds of quiet source holds.
    source = _SourceOwnershipCampaign(root, config)
    sampler = IndependentSampler.from_config(root, independent["registry"])
    numeric_pin = FilePin.model_validate(independent["numeric_certificates"])
    numeric = IndependentNumericCertificates.from_path(
        _pin(root, numeric_pin), expected_sha256=numeric_pin.sha256
    )
    wording_pin = FilePin.model_validate(independent["wording_certificates"])
    wording = load_certificates(_pin(root, wording_pin), expected_sha256=wording_pin.sha256)
    geography_pin = FilePin.model_validate(independent["geography_certificates"])
    geography = IndependentGeographyGate(
        load_independent_geography_certificates(
            read_regular_file_bytes(_pin(root, geography_pin)), expected_sha256=geography_pin.sha256
        )
    )
    for name, loader in (
        ("locality_owner_reviews", load_locality_reviews),
        ("distinct_actor_reviews", load_distinct_actor_reviews),
    ):
        pin = FilePin.model_validate(independent[name])
        loader(read_regular_file_bytes(_pin(root, pin)), expected_sha256=pin.sha256)
    certificates = numeric.certificates()
    evidence_pins = {}
    for certificate in certificates.values():
        for hold in certificate.get("sourceReadinessHolds", []):
            evidence = FilePin.model_validate(hold["evidence"])
            evidence_pins[str(_pin(root, evidence))] = evidence.sha256
    pins = _source_pins(root, config, source.source_ids)
    pins.update(evidence_pins)
    return _SourceContext(
        source,
        sampler,
        numeric,
        certificates,
        wording,
        geography,
        pins,
    )


def _compile_source(context: _SourceContext, sid: str, mixed: Set[str]) -> _CompiledSource:
    """One source-only prerequisite path shared by inventory and preparation."""
    if sid not in context.certificates:
        raise _SourceNotReady(
            "missing_numeric_certificate", "source lacks a complete reviewed numeric certificate"
        )
    held = context.certificates[sid].get("holdReasons", [])
    if held:
        raise _SourceNotReady("explicit_numeric_hold", "; ".join(held))
    readiness = context.numeric.source_readiness_holds(sid)
    if readiness:
        raise _SourceNotReady(
            "source_readiness_hold",
            "; ".join(f"{hold.category}: {hold.reason}" for hold in readiness),
        )
    ordinary = context.source.blueprint(sid)
    validate_private_person_coverage(ordinary)
    validate_identifier_privacy(ordinary)
    bp = context.numeric.apply_blueprint(ordinary, sid)
    if sid in context.wording:
        bp = prepare_blueprint(bp, context.wording[sid])
    context.geography.bind_blueprint(ordinary, bp)
    goods = bp.target["documentPatch"]["goodsItemDetails"]
    if len(goods) != 1 or len(goods[0]["numberAndTypeOfPackages"]) != 1:
        raise _SourceNotReady(
            "unsupported_cardinality",
            "independent preparation requires one goods and one package row",
        )
    if not any(_TYPE_PATH in region.target_paths for region in bp.regions):
        raise _SourceNotReady(
            "unowned_package_declaration",
            "package-type variation lacks a source-owned printed declaration",
        )
    relations = context.numeric.allowed_private_relations(sid)
    if not relations:
        raise _SourceNotReady(
            "unsupported_private_presentation",
            "source has no certified private packing presentation",
        )
    if sid in mixed:
        equipment = compile_equipment_inventory(bp, allow_uniform_aggregate=True)
        if equipment.uniform_aggregate is not None or equipment.row_count < 2:
            raise _SourceNotReady(
                "unsupported_mixed_equipment",
                "requested mixed equipment lacks independent row ownership",
            )
    rows = bp.target["documentPatch"].get("containerInformation", [])
    if len(rows) > 1 and all({"sizeCategory", "typeCategory"} <= row.keys() for row in rows):
        # The real campaign always compiles these inventories, including uniform
        # requests. Do the identical source-only check before paid native calls.
        compile_equipment_inventory(
            rebase_equipment_inventory_headers(bp), allow_uniform_aggregate=True
        )
    cap = context.source.capabilities[sid]
    fixed_mix = cap.equipment_mix is not None and len(cap.equipment_mix.members) > 1
    regime = _REGIMES[cap.family]
    domain, _, slots = source_equipment_readiness(bp, context.source.physical_support, cap, regime)
    return _CompiledSource(
        cap,
        regime,
        _dg_requirements(goods[0], regime),
        relations,
        fixed_mix,
        len(rows),
        domain,
        slots,
    )


def _source_readiness(
    context: _SourceContext, sid: str, mixed: Set[str]
) -> tuple[_CompiledSource | None, dict[str, str] | None]:
    try:
        return _compile_source(context, sid, mixed), None
    except (ArtifactReadError, AtomicConflictError, json.JSONDecodeError):
        raise
    except (ValueError, KeyError) as error:
        return None, {
            "category": error.category
            if isinstance(error, _SourceNotReady)
            else "source_compile_hold",
            "reason": f"{type(error).__name__}: {error}",
        }


def inventory_sources(root: Path, config: dict[str, Any]) -> JsonObject:
    """Inspect every selected source without cargo draws or provider calls.

    The ordinary source compiler may initialize the configured output ledger
    directory. This function writes no reports, samples, sources or receipts.
    Ready means source prerequisites only, not native or physical certification.
    """
    root, config = Path(root).resolve(), copy.deepcopy(config)
    _validate_config(config)
    _output_directory(root, config)
    mixed = _validate_mixed_scope(
        config, config.get("independent_preparation", {}).get("mixed_source_ids", ())
    )
    context = _load_source_context(root, config)
    inventory = {}
    for sid in context.source.source_ids:
        compiled, failure = _source_readiness(context, sid, mixed)
        row = {
            "sourceDocumentId": sid,
            "variants": context.source.variant_counts[sid],
            "status": "ready_source" if compiled is not None else "held_source",
            "category": "source_prerequisites_pass" if failure is None else failure["category"],
            "reason": None if failure is None else failure["reason"],
        }
        if compiled is not None:
            row.update(regime=compiled.regime, sourceRelations=list(compiled.relations))
        inventory[sid] = row
    _verify_pins(context.pins)
    return {
        "inventory": inventory,
        "selectedSources": len(inventory),
        "plannedSlots": sum(row["variants"] for row in inventory.values()),
        "readySources": sum(row["status"] == "ready_source" for row in inventory.values()),
        "readySlots": sum(
            row["variants"] for row in inventory.values() if row["status"] == "ready_source"
        ),
        "counts": dict(Counter(row["category"] for row in inventory.values())),
        "nativeCalls": 0,
        "cargoDraws": 0,
        "datasetMutation": False,
        "outputDirectoryMayBeInitialized": True,
        "inputPins": context.pins,
        "scope": "Source prerequisites only; no native choices, physical plans or admissions",
    }


def _load_pin(root: Path, value: Mapping[str, Any]) -> JsonObject:
    pin = FilePin.model_validate(value)
    return cast(JsonObject, json.loads(read_regular_file_bytes(_pin(root, pin))))


def _publish(root: Path, path: Path, value: Any) -> dict[str, str]:
    atomic_publish_json(path, value)
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def _source_pins(
    root: Path, config: Mapping[str, Any], source_ids: Sequence[str]
) -> dict[str, str | None]:
    paths = {
        root / config["dataset"] / "train.jsonl",
        root / config["dataset"] / "validation.jsonl",
        root / config["ownership"],
        root / config["auxiliary"],
        root / config["task_constraints"],
    }
    for sid in source_ids:
        paths.add(root / config["source_contracts"] / sid / "contract.json")
        paths.add(root / config["historical_catalog"] / sid / "template.json")
    for name, pin in config["independent"].items():
        values = pin.values() if name == "registry" else (pin,)
        for value in values:
            paths.add(_pin(root, FilePin.model_validate(value)))
    return {str(path): sha256_file(path) if path.exists() else None for path in paths}


def _verify_pins(pins: Mapping[str, str | None]) -> None:
    for name, expected in pins.items():
        path = Path(name)
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            raise ValueError("preparation authority changed during execution: " + name)


def _dg_requirements(goods: Mapping[str, Any], regime: str) -> dict[str, bool | int] | None:
    if regime != "dg":
        return None
    declarations = goods.get("dangerousGoods", [])
    if len(declarations) != 1:
        raise ValueError("joined DG preparation requires exactly one declared DG identity")
    fields = declarations[0]
    result: dict[str, bool | int] = {
        "flashPoint": "flashPoint" in fields,
        "packingGroup": "packingGroupCategory" in fields,
    }
    if "subsidiaryHazardCategories" in fields:
        result["subsidiaryCount"] = len(fields["subsidiaryHazardCategories"])
    return result


def _package_validator(request: PackageChoiceRequest) -> Callable[[JsonObject], PackageChoice]:
    def validate(value: JsonObject) -> PackageChoice:
        return validate_package_choice(request, value)

    return validate


def _private_validator(
    request: PrivatePackingRequest,
) -> Callable[[JsonObject], PrivatePackingPlan]:
    def validate(value: JsonObject) -> PrivatePackingPlan:
        return validate_private_packing_plan(request, value)

    return validate


async def _native_batch(
    runner: Runner, jobs: Mapping[str, NativeJob], directory: Path
) -> dict[str, JsonObject]:
    """Every scheduled request completes and journals even when another fails."""

    async def one(key: str, job: NativeJob) -> tuple[str, JsonObject]:
        stage, model, system, prompt, validate, native_request = job
        packet = {
            "stage": stage,
            "identity": key,
            "schema": model.model_json_schema(),
            "system": system,
            "prompt": prompt,
            "provider": runner.config["provider"],
            "request": native_request,
        }
        packet_sha = digest(packet)
        packet_pin = _publish(
            runner.root, runner.output / "preparation-packets" / (packet_sha + ".json"), packet
        )
        started, output, error = time.perf_counter(), None, None
        try:
            value = await runner.call(stage, key, model, system, prompt)
            output = value.model_dump(mode="json")
            validate(output)
        except (ArtifactReadError, AtomicConflictError, json.JSONDecodeError):
            # Local corruption is not a provider hold. The outer barrier still
            # waits for every other attempt before surfacing the fatal failure.
            raise
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        call_key = digest(
            [stage, key, model.model_json_schema(), system, prompt, runner.config["provider"]]
        )
        receipts = [
            {"path": str(path.relative_to(runner.root)), "sha256": sha256_file(path)}
            for path in sorted((runner.output / "calls").glob(f"{stage}-{call_key}*.json"))
        ]
        if output is not None and not receipts:
            raise ValueError("native output has no canonical immutable Runner receipt")
        row = {
            "output": output,
            "error": error,
            "packetSha256": packet_sha,
            "packet": packet_pin,
            "callReceipts": receipts,
            "seconds": time.perf_counter() - started,
        }
        atomic_publish_json(directory / "results" / (digest(key) + ".json"), row)
        return key, row

    outcomes = await asyncio.gather(
        *(one(key, job) for key, job in jobs.items()), return_exceptions=True
    )
    failures = {
        key: f"{type(value).__name__}: {value}"
        for key, value in zip(jobs, outcomes, strict=True)
        if isinstance(value, BaseException)
    }
    if failures:
        raise RuntimeError(
            f"native batch failed after all scheduled requests completed: {failures}"
        )
    return dict(cast(list[tuple[str, JsonObject]], outcomes))


def _runnable_config(
    root: Path,
    config: dict[str, Any],
    groups: Mapping[str, int],
    requests: Mapping[str, JsonObject],
    plans: Mapping[str, JsonObject],
    output: Path,
) -> JsonObject | None:
    if not groups:
        return None
    sources = [sid for sid in config["source_ids"] if sid in groups]
    ready_requests = {
        key: value for key, value in requests.items() if value["sourceDocumentId"] in groups
    }
    ready_plans = {key: plans[key] for key in ready_requests}
    identity = digest([sources, ready_requests, ready_plans, config["independent"]])
    directory = output / "prepared" / identity
    result = copy.deepcopy(config)
    result.pop("independent_preparation", None)
    result["source_ids"] = sources
    result["capabilities"] = {sid: config["capabilities"][sid] for sid in sources}
    result["template_sampling"] = {"samples": sum(groups.values()), "source_counts": dict(groups)}
    result["independent"]["requests"] = _publish(root, directory / "requests.json", ready_requests)
    result["independent"]["private_plans"] = _publish(
        root, directory / "private-plans.json", ready_plans
    )
    for name in ("wording_certificates", "locality_owner_reviews", "distinct_actor_reviews"):
        value = _load_pin(root, config["independent"][name])
        if name == "wording_certificates":
            value["sources"] = {
                sid: item for sid, item in value["sources"].items() if sid in groups
            }
        else:
            value = {sid: item for sid, item in value.items() if sid in groups}
        result["independent"][name] = _publish(root, directory / (name + ".json"), value)
    # Existing equipment policy is a source scope, not permission to reassign
    # the omitted groups' policies to another source during filtering.
    inventory = result["sampling"].get("equipment_inventory")
    if inventory is not None:
        inventory["source_ids"] = [sid for sid in inventory["source_ids"] if sid in groups]
        if not inventory["source_ids"]:
            result["sampling"].pop("equipment_inventory")
    return result


async def prepare_campaign(
    root: Path,
    config: JsonObject,
    *,
    package_targets: Mapping[str, int],
    mixed_source_ids: Collection[str] = (),
    mixed_probability: float = 0.5,
    accepted_counts: Mapping[str, int] | None = None,
    unit_profile_source_ids: Collection[str] | None = None,
    retry_rate_limited: bool = False,
    retry_invalid_output: bool = False,
    retry_failed_receipt_sha256s: Collection[str] = (),
) -> JsonObject:
    """Prepare new synthesis only; never audit, rewrite or regenerate existing data.

    Source-group atomicity deliberately trades some usable sibling proposals for
    a normal runnable publication contract. The complete inventory preserves
    those siblings as blocked_by_source_group, never as failed or replacements.
    """
    root = await asyncio.to_thread(Path(root).resolve)
    if type(retry_rate_limited) is not bool or type(retry_invalid_output) is not bool:
        raise ValueError("native retry permissions must be explicit booleans")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in retry_failed_receipt_sha256s
    ):
        raise ValueError("failed receipt retry requires exact SHA-256 values")
    config = copy.deepcopy(config)
    _validate_config(config)
    accepted_counts = {} if accepted_counts is None else dict(accepted_counts)
    package_coverage(package_targets, accepted_counts)
    mixed = _validate_mixed_scope(config, mixed_source_ids)
    profile_sources = frozenset(
        config["source_ids"] if unit_profile_source_ids is None else unit_profile_source_ids
    )
    if not profile_sources <= set(config["source_ids"]):
        raise ValueError("unit-profile sources must belong to the selected campaign")
    if (
        type(mixed_probability) not in {int, float}
        or not math.isfinite(mixed_probability)
        or not 0 <= mixed_probability <= 1
    ):
        raise ValueError("mixed_probability must be a finite probability")
    output = _output_directory(root, config)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "native-stage.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another native preparation stage owns this output ledger") from error
        return await _prepare_locked(
            root,
            config,
            package_targets,
            mixed,
            mixed_probability,
            accepted_counts,
            output,
            retry_rate_limited=retry_rate_limited,
            retry_invalid_output=retry_invalid_output,
            retry_failed_receipt_sha256s=frozenset(retry_failed_receipt_sha256s),
            unit_profile_source_ids=profile_sources,
        )


async def _prepare_locked(
    root: Path,
    config: JsonObject,
    targets: Mapping[str, int],
    mixed: Set[str],
    probability: float,
    accepted: Mapping[str, int],
    output: Path,
    *,
    retry_rate_limited: bool,
    retry_invalid_output: bool,
    retry_failed_receipt_sha256s: frozenset[str],
    unit_profile_source_ids: frozenset[str],
) -> JsonObject:
    directory = output / "preparation" / str(time.time_ns())
    authorities = _load_source_context(root, config)
    source, sampler, numeric, pins = (
        authorities.source,
        authorities.sampler,
        authorities.numeric,
        authorities.pins,
    )
    source.calls.retry_rate_limited = retry_rate_limited
    source.calls.retry_invalid_output = retry_invalid_output
    source.calls.retry_failed_receipt_sha256s = retry_failed_receipt_sha256s
    required = {
        sid
        for sid in source.source_ids
        if sid in authorities.certificates
        and authorities.certificates[sid].get("requiredPhysicalAuthority") is not None
    }
    if not required <= unit_profile_source_ids:
        raise ValueError(
            "source requires native unit-profile preparation: " + str(sorted(required))
        )
    atomic_publish_json(directory / "authority.json", {"config": config, "inputPins": pins})
    inventory: dict[str, JsonObject] = {}
    requests: dict[str, JsonObject] = {}
    native_requests: dict[str, PackageChoiceRequest] = {}
    jobs: dict[str, NativeJob] = {}
    proposed: Counter[str] = Counter()
    spent_before = source.calls.spent
    for sid in source.source_ids:
        count = source.variant_counts[sid]
        compiled, failure = _source_readiness(authorities, sid, mixed)
        error = None if failure is None else failure["reason"]
        if compiled is not None:
            cap, regime = compiled.capability, compiled.regime
            relations, fixed_mix = compiled.relations, compiled.fixed_mix
            requirements = compiled.dg_requirements
        excluded: list[str] = []
        for variant in range(1, count + 1):
            key = f"{sid}:{variant}"
            inventory[key] = {
                "sourceDocumentId": sid,
                "variant": variant,
                "status": "held_source",
                "reason": error,
            }
            if failure is not None:
                inventory[key]["readinessCategory"] = failure["category"]
            if error is not None:
                continue
            assert compiled is not None
            try:
                draw = int(digest([config["seed"], key, "equipment-mode"]), 16) / 2**256
                request = sampler.prepare_request(
                    source_document_id=sid,
                    variant=variant,
                    regime=regime,
                    identity_count=cap.identity_count,
                    seed=config["seed"],
                    equipment_mode="mixed"
                    if fixed_mix or (sid in mixed and draw < probability)
                    else "uniform",
                    dg_requirements=requirements,
                    excluded_identifiers=excluded,
                    cargo_presentation=authorities.certificates[sid].get("cargoPresentation"),
                )
                sampler.validate_request(request)
                request["equipmentSlots"] = sample_equipment(
                    regime=regime,
                    equipment_mode=request["equipmentMode"],
                    seed=config["seed"],
                    draw_key=key + ":equipment",
                    container_count=compiled.container_count,
                    equipment_domain=compiled.equipment_domain,
                    equipment_slots=compiled.equipment_slots,
                )
                excluded.extend(
                    [request["dangerousGoods"]["recordId"]]
                    if regime == "dg"
                    else [g["hs6"] for g in request["goods"]]
                )
                context: JsonObject = {
                    "goods": request["goods"],
                    "regime": regime,
                    "equipment": equipment_choice_brief(request["equipmentSlots"]),
                    "form": request["goodsForm"],
                    "dangerousGoods": request["dangerousGoods"],
                }
                if any(row["typeCategory"] in TANK_TYPES for row in request["equipmentSlots"]):
                    context["installedTank"] = True
                if regime == "dg":
                    context["packingContext"] = (
                        "Registry physical state: "
                        + sampler.hmt_states[request["dangerousGoods"]["recordId"]]
                    )
                if relations == ("MIXED",):
                    context["packingContext"] = (
                        (context.get("packingContext", "") + "\n")
                        + "Source presentation counts supports plus loose packages as generic "
                        "PACKAGE_PACKAGE. A later semantic planner selects compatible support "
                        "and loose-package types; no source cargo or ratios are retained."
                    )
                candidates = [
                    category
                    for category, target in targets.items()
                    if target > accepted.get(category, 0) + proposed[category]
                ]
                presentation = request.get("cargoPresentation")
                if presentation is not None:
                    context["allowedPackageTypes"] = presentation["packageTypes"]
                    candidates = [c for c in candidates if c in presentation["packageTypes"]]
                if relations == ("MIXED",):
                    context["requiredCategory"] = "PACKAGE_PACKAGE"
                elif candidates:
                    candidate = min(
                        candidates,
                        key=lambda c: (
                            -(targets[c] - accepted.get(c, 0) - proposed[c]),
                            digest([config["seed"], key, c]),
                        ),
                    )
                    context["requiredCategory"] = candidate
                    proposed[candidate] += 1
                package_request = PackageChoiceRequest.model_validate(context)
                prompt = package_choice_prompt(package_request, hs_codes=sampler.phrases)
                native_requests[key], requests[key] = package_request, request
                jobs[key] = (
                    "independent-package-choice",
                    package_choice_output_type(package_request),
                    PACKAGE_CHOICE_SYSTEM,
                    prompt,
                    _package_validator(package_request),
                    package_request.model_dump(mode="json"),
                )
                inventory[key] = {
                    "sourceDocumentId": sid,
                    "variant": variant,
                    "status": "goods_prepared",
                    "sourceRelations": list(relations),
                }
            except Exception as exception:
                inventory[key].update(
                    status="held_goods_preparation",
                    reason=f"{type(exception).__name__}: {exception}",
                )
    _verify_pins(pins)
    atomic_publish_json(directory / "goods-requests.json", requests)
    choices = await _native_batch(source.calls, jobs, directory / "choices")
    _verify_pins(pins)
    reserved: Counter[str] = Counter()
    used: defaultdict[str, set[str]] = defaultdict(set)
    plans: dict[str, JsonObject] = {}
    private_jobs: dict[str, NativeJob] = {}
    private_requests: dict[str, PrivatePackingRequest | MixedPackingRequest] = {}
    for key, result in choices.items():
        row, request = inventory[key], requests[key]
        if result["error"] is not None:
            row.update(status="held_native_choice", reason=result["error"])
            continue
        answer = validate_package_choice(native_requests[key], result["output"])
        if answer.unknown:
            row.update(
                status="held_unknown_package", reason="native choice explicitly returned UNKNOWN"
            )
            continue
        alternatives = answer.package_types
        if row["sourceRelations"] == ["MIXED"]:
            alternatives = [c for c in alternatives if c == "PACKAGE_PACKAGE"]
            if not alternatives:
                row.update(
                    status="held_unknown_package",
                    reason="Generic mixed accounting is not plausible for sampled goods",
                )
                continue
        category = select_package_type(
            alternatives,
            targets=targets,
            accepted=accepted,
            reserved=reserved,
            used=used[row["sourceDocumentId"]],
            seed=config["seed"],
            key=key,
        )
        request.update(
            selectedCategory=category,
            countedUnitContext=answer.counting_contexts.get(category),
            planningStage="native_package_choice_accepted",
            nativeChoicePacketSha256=result["packetSha256"],
            nativeChoiceEvidence=_evidence(native_requests[key], result),
        )
        reserved[category] += 1
        used[row["sourceDocumentId"]].add(category)
        row.update(status="package_selected", selectedCategory=category)
        if row["sourceRelations"] == ["MIXED"]:
            mixed_request = MixedPackingRequest.model_validate(
                {
                    **native_requests[key].model_dump(exclude={"requiredCategory"}),
                    "countedCategory": category,
                }
            )
            private_requests[key] = mixed_request
            private_jobs[key] = (
                "independent-mixed-accounting",
                MixedPackingPlan,
                MIXED_PACKING_SYSTEM,
                mixed_packing_prompt(mixed_request, hs_codes=sampler.phrases),
                MixedPackingPlan.model_validate,
                mixed_request.model_dump(mode="json"),
            )
            continue
        if row["sourceRelations"] == ["NONE"]:
            plans[key] = {
                "privatePackingRelation": "NONE",
                "relatedPackageType": None,
                "origin": "reviewed_no_private_relation",
                "sourceCertificateSha256": digest(numeric.certificate(row["sourceDocumentId"])),
            }
            row["status"] = "ready"
            continue
        private_request = PrivatePackingRequest.model_validate(
            {
                **native_requests[key].model_dump(exclude={"requiredCategory"}),
                "countedCategory": category,
                "allowedRelations": row["sourceRelations"],
            }
        )
        private_requests[key] = private_request
        private_jobs[key] = (
            "independent-private-packing",
            private_packing_output_type(private_request),
            PRIVATE_PACKING_SYSTEM,
            private_packing_prompt(private_request, hs_codes=sampler.phrases),
            _private_validator(private_request),
            private_request.model_dump(mode="json"),
        )
    private_results = await _native_batch(source.calls, private_jobs, directory / "private")
    _verify_pins(pins)
    for key, result in private_results.items():
        row = inventory[key]
        if result["error"] is not None:
            row.update(status="held_native_private_plan", reason=result["error"])
            continue
        is_mixed = isinstance(private_requests[key], MixedPackingRequest)
        plan = (
            MixedPackingPlan.model_validate(result["output"])
            if is_mixed
            else validate_private_packing_plan(private_requests[key], result["output"])
        )
        if plan.unknown:
            row.update(
                status="held_unknown_private_plan",
                reason="private plan explicitly returned UNKNOWN",
            )
            continue
        plans[key] = {
            **(mixed_packing_facts(plan) if is_mixed else private_packing_facts(plan)),
            "origin": "native_mixed_accounting" if is_mixed else "native_typed_private_plan",
            "nativePlanPacketSha256": result["packetSha256"],
            "nativePlanEvidence": _evidence(private_requests[key], result),
        }
        row["status"] = "ready"
    groups = _complete_groups(source.variant_counts, inventory)
    runnable = _runnable_config(root, config, groups, requests, plans, output)
    if runnable is not None and (profile_scope := set(groups) & unit_profile_source_ids):
        from document_ocr.synthesis.independent_unit_preparation import prepare_unit_profiles

        profile_report = await prepare_unit_profiles(
            root,
            runnable,
            source_ids=profile_scope,
            retry_rate_limited=retry_rate_limited,
            retry_invalid_output=retry_invalid_output,
            retry_failed_receipt_sha256s=retry_failed_receipt_sha256s,
        )
        source.calls.spent = Decimal(profile_report["spentAfterUsd"])
        for key, row in profile_report["inventory"].items():
            if row["status"] != "ready":
                inventory[key].update(row)
        runnable = profile_report["config"]
        if runnable is not None:
            requests.update(_load_pin(root, runnable["independent"]["requests"]))
        groups = _complete_groups(source.variant_counts, inventory)
    if runnable is not None:
        campaign = IndependentCampaign(root, runnable)
        for sid in tuple(groups):
            for variant in range(1, groups[sid] + 1):
                key = f"{sid}:{variant}"
                try:
                    campaign.plan(sid, variant)
                except Exception as exception:
                    inventory[key].update(
                        status="held_physical_plan",
                        reason=f"{type(exception).__name__}: {exception}",
                    )
        groups = _complete_groups(source.variant_counts, inventory)
        runnable = _runnable_config(root, config, groups, requests, plans, output)
    _verify_pins(pins)
    config_pin = (
        None if runnable is None else _publish(root, directory / "prepared-config.json", runnable)
    )
    result = {
        "config": runnable,
        "preparedConfig": config_pin,
        "inventory": inventory,
        "counts": dict(Counter(row["status"] for row in inventory.values())),
        "plannedSlots": len(inventory),
        "readySources": len(groups),
        "readySlots": sum(groups.values()),
        "provisionalReservations": dict(reserved),
        "acceptedFinalCounts": dict(accepted),
        "readyReservations": dict(
            Counter(
                requests[key]["selectedCategory"]
                for key, row in inventory.items()
                if row["status"] == "ready"
            )
        ),
        "proposedEquipmentModeCounts": dict(Counter(r["equipmentMode"] for r in requests.values())),
        "readyEquipmentModeCounts": dict(
            Counter(
                requests[key]["equipmentMode"]
                for key, row in inventory.items()
                if row["status"] == "ready"
            )
        ),
        "coverage": package_coverage(targets, accepted),
        "registrySupport": sampler.support(),
        "spentBeforeUsd": str(spent_before),
        "spentAfterUsd": str(source.calls.spent),
        "newGenerationAdmissions": 0,
        "datasetMutation": False,
        "sourceGroupAtomicity": (
            "No renumbering/replacement; successful siblings of held slots "
            "remain blocked_by_source_group"
        ),
    }
    result["summaryPath"] = str((directory / "summary.json").relative_to(root))
    atomic_publish_json(directory / "summary.json", result)
    return result


def _evidence(request: BaseModel, result: Mapping[str, Any]) -> JsonObject:
    return {
        "request": request.model_dump(mode="json"),
        "output": copy.deepcopy(result["output"]),
        "packet": result["packet"],
        "packetSha256": result["packetSha256"],
        "callReceipts": result["callReceipts"],
    }


def _complete_groups(counts: Mapping[str, int], inventory: dict[str, JsonObject]) -> dict[str, int]:
    groups: dict[str, int] = {}
    for sid, count in counts.items():
        keys = [f"{sid}:{variant}" for variant in range(1, count + 1)]
        if all(inventory[key]["status"] == "ready" for key in keys):
            groups[sid] = count
        else:
            for key in keys:
                if inventory[key]["status"] == "ready":
                    inventory[key]["status"] = "blocked_by_source_group"
    return groups
