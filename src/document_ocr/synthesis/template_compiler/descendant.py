from __future__ import annotations

import asyncio
import difflib
import json
import re
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from .mixed_inventory import MixedInventory

import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model
from pydantic_ai import Agent, ModelProfile, NativeOutput, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v5 import (
    ContainerSizeCategory,
    ContainerTypeCategory,
)
from document_ocr.synthesis.container_semantics import (
    canonical_equipment_surface,
    review_source_equipment_surface,
)
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_container_number,
    generate_from_surface_pattern,
    surface_pattern,
    validate_container_number,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    load_provider_key,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.package_registry import package_category_surface_present
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import CustomsProgramRegistry
from document_ocr.synthesis.raw_text_template import (
    CompiledRawTextTemplate,
    TemplateRenderProof,
    TemplateSlot,
    printed_topology_mismatches,
    render_compiled_template,
    validate_slot_replacements,
)
from document_ocr.synthesis.rendering import (
    _numeric_interpretations as numeric_surface_interpretations,
)
from document_ocr.synthesis.rendering import render_number_surface
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_V5_TASK_ADAPTER
from document_ocr.synthesis.usage_receipt import LinguisticUsageReceipt
from document_ocr.training.address_projection import project_training_target
from document_ocr.training.reviewed_package_projection import (
    ReviewedPackageContract,
    load_reviewed_package_contract,
    project_reviewed_package_target,
)
from document_ocr.training.tasks import get_training_task

from . import (
    contact_values,
    count_aliases,
    entity_identifiers,
    geographic_context,
    labelled_context,
    lexical_partitions,
)
from .coherence import (
    coherence_dependency_paths,
    inclusive_range_cardinalities,
    validate_render_coherence,
    whole_inclusive_range_surface,
)
from .customs_presentation import CustomsPresentation, compile_neutral_customs
from .dangerous_goods_realization import DangerousGoodsFact, render_facts, validate_un_references
from .descendant_models import (
    BindingRoute,
    DescendantCaseResult,
    DescendantConfig,
    PinnedCommittedRun,
    PreparedTargetReceipt,
    ResidualReplayReceipt,
    ResidualStageReceipt,
)
from .fixed_vocabulary import fixed_context
from .generation_contract import (
    INVALID_TEXT_CONTROL,
    leaves,
    party_owned_surfaces,
    require_complete_variation,
    validate_compiled_party_contract,
    validate_rendered_party_boundaries,
    validate_repeated_agent_party_pages,
    validate_seal_realization,
    validate_unbound_lexical_surfaces,
)
from .host import validate_compiled_location_payment_grounding
from .latest_target import latest_target_from_source
from .models import (
    AuxiliaryEntity,
    AuxiliaryEntityMember,
    BindingRealization,
    CertifiedSemanticTemplate,
    CoherenceConstraint,
    OpenRouterProviderConfig,
    ProviderConfig,
    SemanticBinding,
    SourceBindingRelationship,
)
from .numeric_auxiliary import PreparedNumeric, numeric_bindings, render_prepared, surface_quantum
from .pipeline import project_root_from_config, resolve_input
from .range_generation import plan_ranges, render_composite_range
from .realization_contract import (
    effective_realization_template,
    fixed_projection_ranges,
    projected_auxiliary_values,
    shared_projection_ranges,
    token_projection_intervals,
)
from .semantic_plan import (
    disposition_by_binding,
    entity_members_by_binding,
    resolve_geographic_members,
    validate_auxiliary_render,
)
from .synthetic_values import DeterministicValueFactory, GeoProfile

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_MODEL_PATH = Path(__file__).with_name("descendant_models.py").resolve(strict=True)
_SEMANTIC_PLAN_PATH = Path(__file__).with_name("semantic_plan.py").resolve(strict=True)
_SYNTHETIC_VALUES_PATH = Path(__file__).with_name("synthetic_values.py").resolve(strict=True)
_COHERENCE_PATH = Path(__file__).with_name("coherence.py").resolve(strict=True)
_LATEST_TARGET_PATH = Path(__file__).with_name("latest_target.py").resolve(strict=True)
_STRICT_DYNAMIC = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")
_NUMBER = re.compile(r"[-+]?[0-9][0-9., '\u00a0]*")
_TEMPERATURE_SURFACE = re.compile(
    r"(?i)^(?P<leading>\s*)(?P<sign>[+-]?)(?P<integer>[0-9]+)"
    r"(?:(?P<separator>[.,])(?P<fraction>[0-9]+))?(?P<number_gap>\s*)"
    r"(?P<degree>°?)(?P<unit_gap>\s*)(?P<unit>C|F|CELSIUS|FAHRENHEIT)"
    r"(?P<trailing>\s*)$"
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
    "%m.%d.%Y",
    "%d.%b.%Y",
    "%d.%B.%Y",
    "%b.%d.%Y",
    "%B.%d.%Y",
    "%b-%d-%Y",
    "%B-%d-%Y",
    "%b. %d,%Y",
    "%B. %d,%Y",
    "%b. %d, %Y",
    "%B. %d, %Y",
    "%b. %d. %Y",
    "%B. %d. %Y",
    "%m/%d/%Y %I:%M:%S %p",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%y",
    "%m-%d-%y",
    "%d.%m.%y",
    "%m.%d.%y",
    "%Y %b %d",
    "%Y %B %d",
    "%Y-%b-%d",
    "%Y-%B-%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d %b, %Y",
    "%d %B, %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%b-%y",
    "%d %b %y",
    "%b %d %y",
    "%d/%b/%y",
    "%d/%b/%Y",
    "%b/%d/%y",
    "%b/%d/%Y",
    "%b.%d,%Y",
    "%B.%d,%Y",
)
_SENSITIVE_AUXILIARY_KINDS = frozenset(
    {
        "organization",
        "person",
        "address",
        "contact_name",
        "email",
        "phone",
        "url_or_domain",
        "identifier",
    }
)
_NUMBER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
    }
)
_EQUIPMENT_TYPE_CODES = {
    "GENERAL_PURPOSE": "GP",
    "VENTILATED_GENERAL_PURPOSE": "VH",
    "DRY_BULK": "BU",
    "NAMED_CARGO": "SN",
    "REFRIGERATED": "RE",
    "REFRIGERATED_AND_HEATED": "RT",
    "SELF_POWERED_REFRIGERATED": "RS",
    "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT": "HR",
    "INSULATED": "HI",
    "OPEN_TOP": "UT",
    "PLATFORM": "PL",
    "PLATFORM_FIXED": "PF",
    "PLATFORM_COLLAPSIBLE": "PC",
    "PLATFORM_COMPLETE_SUPERSTRUCTURE": "PS",
    "PLATFORM_NAMED_CARGO": "PT",
    "PRESSURIZED_TANK": "KL",
    "DRY_HOPPER_TANK": "NH",
    "DRY_REAR_DISCHARGE_TANK": "NN",
    "AIR_SURFACE": "AS",
}


@dataclass(frozen=True, slots=True)
class PreparedCase:
    document_id: str
    source_document_id: str
    source: bytes
    source_target: dict[str, Any]
    topology_reference_target: dict[str, Any]
    target: dict[str, Any]
    template: CertifiedSemanticTemplate
    target_receipt: PreparedTargetReceipt
    auxiliary_values: Mapping[str, str] = field(default_factory=dict)
    numeric_auxiliary: Mapping[str, PreparedNumeric] = field(default_factory=dict)
    customs_presentation: CustomsPresentation | None = None
    dangerous_goods_facts: tuple[DangerousGoodsFact, ...] = ()
    equipment_tare_values: Mapping[str, Decimal] = field(default_factory=dict)


def _equipment_tare_payload(values: Mapping[str, Decimal]) -> dict[str, str]:
    """Serialize the independent physical sampler values for immutable receipts."""

    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, Decimal)
        or not value.is_finite()
        or value <= 0
        for key, value in values.items()
    ):
        raise ValueError("equipment tare values must be positive finite Decimals by binding key")
    return {key: str(value) for key, value in values.items()}


def _published_equipment_tares(
    row: Mapping[str, Any], numbers: Mapping[str, PreparedNumeric]
) -> dict[str, Decimal]:
    """Restore private physical context, never silently substitute source tares."""
    expected = {
        key for key, value in numbers.items() if value.contract.mode == "sampled_equipment_tare"
    }
    payload = row.get("equipmentTareValues", {})
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("published equipment tares must exactly cover sampled tare contracts")
    if any(not isinstance(value, str) for value in payload.values()):
        raise ValueError("published equipment tares must use exact decimal strings")
    values = {key: Decimal(value) for key, value in payload.items()}
    _equipment_tare_payload(values)
    return values


def _require_frozen_target(case: PreparedCase) -> None:
    receipt = case.target_receipt
    digest = sha256_bytes(canonical_json_bytes(case.target))
    if (
        digest != receipt.proposed_target_sha256
        or digest != receipt.prepared_target_sha256
        or receipt.compatibility_adaptations
    ):
        raise ValueError(
            f"synthetic target changed after generation: {case.document_id}; "
            "rendering and publication require the exact completed target"
        )
    if sha256_bytes(canonical_json_bytes(case.auxiliary_values)) != receipt.auxiliary_values_sha256:
        raise ValueError("auxiliary semantic context changed after generation")
    numeric = {key: value.model_dump(mode="json") for key, value in case.numeric_auxiliary.items()}
    if sha256_bytes(canonical_json_bytes(numeric)) != receipt.numeric_auxiliary_sha256:
        raise ValueError("numeric auxiliary context changed after generation")
    tare_payload = _equipment_tare_payload(case.equipment_tare_values)
    if sha256_bytes(canonical_json_bytes(tare_payload)) != receipt.equipment_tare_values_sha256:
        raise ValueError("sampled equipment tare context changed after generation")
    customs_hash = (
        case.customs_presentation.sha256
        if case.customs_presentation is not None
        else sha256_bytes(canonical_json_bytes([]))
    )
    if customs_hash != receipt.customs_presentation_sha256:
        raise ValueError("customs presentation contract changed after preparation")
    facts = [fact.model_dump(mode="json") for fact in case.dangerous_goods_facts]
    if sha256_bytes(canonical_json_bytes(facts)) != receipt.dangerous_goods_facts_sha256:
        raise ValueError("dangerous goods registry facts changed after preparation")
    old_groups = case.topology_reference_target["documentPatch"].get("cargoGroups", [])
    new_groups = case.target["documentPatch"].get("cargoGroups", [])
    if not facts and any(
        old.get("dangerousGoods") != new.get("dangerousGoods")
        for old, new in zip(old_groups, new_groups, strict=True)
    ):
        raise ValueError("changed dangerous goods require complete registry realization facts")
    labelled_context.validate_final(
        SimpleNamespace(template=case.template, target=case.source_target), case.target
    )


@dataclass(frozen=True, slots=True)
class BindingOutput:
    replacements: Mapping[str, str]
    canonical_value: JsonValue


@dataclass(frozen=True, slots=True)
class EntityGeographyConstraint:
    calling_code_width: int | None = None
    country_name_width: int | None = None
    postal_code_pattern: str | None = None
    fixed_country_code: str | None = None
    fixed_country_bindings: frozenset[str] = frozenset()


def _country_code_surface_style(source: str, *, country_surfaces: frozenset[str]) -> str:
    compact = _alphanumeric(source)
    normalized = compact.casefold()
    if normalized in country_surfaces:
        return "country_name"
    if re.fullmatch(r"\s*\+\s*[0-9]+\s*", source):
        return "calling_plus"
    if re.fullmatch(r"\s*00[0-9]+\s*", source):
        return "calling_00"
    if compact.isalpha() and len(compact) == 2:
        return "iso2"
    if compact.isalpha() and len(compact) == 3:
        return "iso3"
    if compact.isalpha() and len(compact) == 5:
        return "unlocode"
    raise ValueError(f"unsupported country-code surface grammar: {source!r}")


def _entity_country_code_context(
    template: CertifiedSemanticTemplate,
    country_codes: Mapping[str, str],
    source: bytes | None = None,
) -> tuple[dict[str, str], dict[str, EntityGeographyConstraint]]:
    bindings = {binding.logical_key: binding for binding in template.bindings}
    styles: dict[str, str] = {}
    constraints: dict[str, EntityGeographyConstraint] = {}
    plan = resolve_geographic_members(
        template.auxiliary_semantic_plan, template.bindings, country_codes
    )
    for entity in plan.entities:
        pinned_members = {
            member.logical_key: code
            for member in entity.members
            if member.field == "country" and source is not None
            if (
                code := geographic_context.pinned_country(
                    bindings[member.logical_key], template, source, country_codes
                )
            )
            is not None
        }
        pinned_countries = set(pinned_members.values())
        if source is not None:
            pinned_countries.update(
                geographic_context.address_country_context(entity, template, source, country_codes)
            )
        if len(pinned_countries) > 1:
            raise ValueError("auxiliary entity has incompatible fixed country contexts")
        country_surfaces = frozenset(
            _alphanumeric(slot.source_text).casefold()
            for member in entity.members
            if member.field == "country"
            for slot in bindings[member.logical_key].occurrences
        )
        calling_widths: set[int] = set()
        country_name_widths: set[int] = set()
        postal_patterns = {
            surface_pattern(_alphanumeric(slot.source_text))
            for member in entity.members
            if member.field == "postal_code"
            for slot in bindings[member.logical_key].occurrences
            if slot.render_policy == "opaque_identifier"
        }
        if len(postal_patterns) > 1:
            raise ValueError(
                f"auxiliary entity has incompatible postal-code shapes: {entity.entity_id}"
            )
        for member in entity.members:
            if member.field != "country_code":
                continue
            binding = bindings[member.logical_key]
            member_styles = {
                _country_code_surface_style(
                    slot.source_text,
                    country_surfaces=country_surfaces,
                )
                for slot in binding.occurrences
            }
            if len(member_styles) != 1:
                raise ValueError(
                    f"country-code binding mixes surface grammars: {member.logical_key}"
                )
            style = member_styles.pop()
            styles[member.logical_key] = style
            for slot in binding.occurrences:
                compact = _alphanumeric(slot.source_text)
                if style == "calling_plus":
                    calling_widths.add(len(compact))
                elif style == "calling_00":
                    calling_widths.add(len(compact) - 2)
                elif style == "country_name":
                    country_name_widths.add(len(compact))
        if len(calling_widths) > 1 or len(country_name_widths) > 1:
            raise ValueError(
                f"auxiliary entity has incompatible country-code widths: {entity.entity_id}"
            )
        constraints[entity.entity_id] = EntityGeographyConstraint(
            calling_code_width=next(iter(calling_widths), None),
            country_name_width=next(iter(country_name_widths), None),
            postal_code_pattern=next(iter(postal_patterns), None),
            fixed_country_code=next(iter(pinned_countries), None),
            fixed_country_bindings=frozenset(pinned_members),
        )
    return styles, constraints


def _preserve_unpadded_day(source: str, rendered: str, *, old: date, new: date) -> str:
    if old.day >= 10:
        return rendered
    unpadded = re.search(rf"(?<![0-9]){old.day}(?![0-9])", source)
    padded = re.search(rf"(?<![0-9])0{old.day}(?![0-9])", source)
    if unpadded is None or padded is not None:
        return rendered
    return re.sub(
        rf"(?<![0-9])0{new.day}(?![0-9])",
        str(new.day),
        rendered,
        count=1,
    )


def _render_date_surface(raw: str, old_iso: str, new_iso: str) -> str:
    """Render every named/numeric date format accepted by the template compiler."""

    old = date.fromisoformat(old_iso)
    new = date.fromisoformat(new_iso)
    if old == new:
        return raw
    compact = raw.strip()
    prefix_length = len(raw) - len(raw.lstrip())
    suffix_length = len(raw) - len(raw.rstrip())
    matching_formats = []
    for date_format in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(compact, date_format).date()
        except ValueError:
            continue
        if parsed == old:
            matching_formats.append(date_format)
    if not matching_formats:
        raise ValueError(f"unsupported certified date surface: {raw!r}")

    # Equivalent numeric day/month formats can parse the same ambiguous source. Their
    # output is identical for a uniformly shifted target only when the target day/month
    # coincide. Refuse ambiguity rather than silently selecting a locale convention.
    rendered_candidates = {
        _preserve_unpadded_day(
            compact,
            _case_like(compact, new.strftime(date_format)),
            old=old,
            new=new,
        )
        for date_format in matching_formats
    }
    if len(rendered_candidates) != 1:
        raise ValueError(f"ambiguous certified date surface: {raw!r}")
    rendered = rendered_candidates.pop()
    suffix = raw[len(raw) - suffix_length :] if suffix_length else ""
    return raw[:prefix_length] + rendered + suffix


@dataclass(frozen=True, slots=True)
class RenderPlan:
    routes: tuple[BindingRoute, ...]
    deterministic_outputs: Mapping[str, BindingOutput]
    residual_bindings: tuple[SemanticBinding, ...]
    independent_phone_countries: Mapping[str, str]


def load_descendant_config(path: Path) -> DescendantConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return DescendantConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


def _validate_committed_run(
    project_root: Path,
    configured: PinnedCommittedRun,
    *,
    workers: int = 1,
) -> Path:
    root = (project_root / configured.path).resolve(strict=True)
    if project_root not in root.parents or root.is_symlink() or not root.is_dir():
        raise ValueError(f"configured run is not a regular project directory: {configured.path}")
    commit_path = root / "_COMMIT.json"
    if sha256_file(commit_path) != configured.commit_sha256:
        raise ValueError(f"committed-run receipt differs: {configured.path}")
    run = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
        validation_workers=workers,
    )
    if not run.completed:
        raise ValueError(f"configured run is not committed: {configured.path}")
    return root


def _read_jsonl(path: Path, *, records: int | None) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_regular_file_bytes(path).splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
        rows.append(row)
    if records is not None and len(rows) != records:
        raise ValueError(f"{path}: expected {records} rows, found {len(rows)}")
    return tuple(rows)


def _path_parts(path: str) -> tuple[str | int, ...]:
    if not path.startswith("documentPatch"):
        raise ValueError(f"target path is outside documentPatch: {path}")
    parts: list[str | int] = []
    cursor = 0
    for match in re.finditer(r"(?:^|\.)([A-Za-z][A-Za-z0-9]*)|\[([0-9]+)\]", path):
        if match.start() != cursor:
            raise ValueError(f"invalid target path syntax: {path}")
        if match.group(1) is not None:
            parts.append(match.group(1))
        else:
            parts.append(int(cast(str, match.group(2))))
        cursor = match.end()
    if cursor != len(path) or not parts:
        raise ValueError(f"invalid target path syntax: {path}")
    return tuple(parts)


def _latest_schema_path(path: str) -> str | None:
    """Map the two relation-v3 DG paths that moved in relation-v4/v5."""

    if ".dangerousGoods[" not in path:
        return None
    if path.endswith(".subsidiaryHazardCategory"):
        return path.removesuffix(".subsidiaryHazardCategory") + (".subsidiaryHazardCategories[0]")
    marker = ".flashPoint.packingGroupCategory"
    if path.endswith(marker):
        return path.removesuffix(marker) + ".packingGroupCategory"
    return None


def _resolve_exact_path(target: Mapping[str, Any], path: str) -> Any:
    value: Any = target
    for part in _path_parts(path):
        if isinstance(part, int):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"target path expects a list at {path}")
            try:
                value = value[part]
            except IndexError as error:
                raise ValueError(f"target path index is absent: {path}") from error
        else:
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"target path is absent: {path}")
            value = value[part]
    return value


def _resolve_path(target: Mapping[str, Any], path: str) -> Any:
    try:
        return _resolve_exact_path(target, path)
    except ValueError:
        alias = _latest_schema_path(path)
        if alias is None or target.get("schemaVersion") != "5.0.0-experimental":
            raise
        return _resolve_exact_path(target, alias)


def _path_exists(target: Mapping[str, Any], path: str) -> bool:
    try:
        _resolve_path(target, path)
    except ValueError:
        return False
    return True


def _flatten_leaves(value: Any, path: str = "") -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "schemaVersion" and not path:
                continue
            child_path = f"{path}.{key}" if path else str(key)
            output.update(_flatten_leaves(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(_flatten_leaves(child, f"{path}[{index}]"))
    else:
        output[path] = cast(JsonValue, value)
    return output


def _validate_canonical_target(target: dict[str, Any]) -> None:
    task_by_version = {
        "3.0.0-experimental": "bill_of_lading_relation_explicit_v3",
        "4.0.0-experimental": "bill_of_lading_relation_explicit_v4",
        "5.0.0-experimental": "bill_of_lading_relation_explicit_v5",
    }
    version_name = target.get("schemaVersion")
    if version_name not in task_by_version:
        raise ValueError(f"unsupported target schema version: {version_name!r}")
    if get_training_task(task_by_version[version_name]).canonicalize(target) != target:
        raise ValueError("target differs from its canonical task representation")


def _require_source_carrier(*, source_target: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    """The producer owns its target; a renderer must not repair its carrier."""
    source_parties = cast(Mapping[str, Any], source_target["documentPatch"]).get("parties")
    target_parties = cast(Mapping[str, Any], target["documentPatch"]).get("parties")
    if not isinstance(source_parties, Mapping) or not isinstance(target_parties, Mapping):
        raise ValueError("carrier-bound target lacks a parties object")
    source_carrier = source_parties.get("carrier")
    if not isinstance(source_carrier, Mapping) or not source_carrier.get("name"):
        raise ValueError("carrier-bound source label lacks a carrier object")
    if source_carrier != target_parties.get("carrier"):
        raise ValueError("synthetic target changes the template-bound carrier; target not modified")


def _case_like(source: str, value: str) -> str:
    letters = [character for character in source if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return value.upper()
    if letters and all(character.islower() for character in letters):
        return value.lower()
    return value


def _line_edges(line: str) -> tuple[str, str, str]:
    leading = re.match(r"^[ \t]*", line)
    trailing = re.search(r"[ \t]*$", line)
    assert leading is not None and trailing is not None
    return leading.group(0), line[leading.end() : trailing.start()], trailing.group(0)


def _allocate_words(words: Sequence[str], weights: Sequence[int]) -> tuple[tuple[str, ...], ...]:
    if not weights:
        raise ValueError("line-layout weights are empty")
    output: list[tuple[str, ...]] = []
    cursor = 0
    remaining_weight = sum(weights)
    for index, weight in enumerate(weights):
        remaining_lines = len(weights) - index - 1
        remaining_words = len(words) - cursor
        if index == len(weights) - 1:
            take = remaining_words
        elif remaining_words <= remaining_lines:
            take = 0
        elif remaining_weight <= 0:
            take = 1
        else:
            take = round(remaining_words * weight / remaining_weight)
            take = max(1 if weight else 0, take)
            take = min(take, remaining_words - remaining_lines)
        output.append(tuple(words[cursor : cursor + take]))
        cursor += take
        remaining_weight -= weight
    if cursor != len(words):
        raise ValueError("line-layout allocation did not consume every word")
    return tuple(output)


def _layout_like_source(source: str, candidate: str) -> str:
    endings = tuple(match.group(0) for match in re.finditer(r"\r\n|\r|\n", source))
    source_lines = re.split(r"\r\n|\r|\n", source)
    # A numeric-only source line has no case evidence. Apply the whole slot's
    # case before distributing words so new text on that line cannot lose it.
    words = tuple(_case_like(source, candidate).split())
    if not words:
        raise ValueError("replacement content is empty")
    weights = tuple(max(1, len(_TOKEN.findall(_line_edges(line)[1]))) for line in source_lines)
    allocated = _allocate_words(words, weights)
    rendered_lines: list[str] = []
    for source_line, line_words in zip(source_lines, allocated, strict=True):
        leading, _body, trailing = _line_edges(source_line)
        body = " ".join(line_words)
        rendered_lines.append(leading + _case_like(source_line, body) + trailing)
    output = rendered_lines[0]
    for ending, line in zip(endings, rendered_lines[1:], strict=True):
        output += ending + line
    return output


def _alphanumeric(value: str) -> str:
    return "".join(character for character in value if character.isalnum())


def _shape_alphanumeric_like_source(source: str, candidate: str) -> str:
    characters = tuple(character for character in candidate if character.isalnum())
    expected = sum(character.isalnum() for character in source)
    if len(characters) != expected:
        raise ValueError(
            f"replacement has {len(characters)} alphanumerics but source shape requires {expected}"
        )
    output: list[str] = []
    cursor = 0
    for source_character in source:
        if not source_character.isalnum():
            output.append(source_character)
            continue
        candidate_character = characters[cursor]
        if source_character.isupper():
            candidate_character = candidate_character.upper()
        elif source_character.islower():
            candidate_character = candidate_character.lower()
        output.append(candidate_character)
        cursor += 1
    return "".join(output)


def _unique_subsequence_indices(canonical: str, observed: str) -> tuple[int, ...]:
    """Return the sole certified deletion projection from canonical to observed."""

    canonical_chars = tuple(character.casefold() for character in _alphanumeric(canonical))
    observed_chars = tuple(character.casefold() for character in _alphanumeric(observed))
    if not observed_chars or len(observed_chars) > len(canonical_chars):
        raise ValueError("identifier occurrence is not a deletion projection of its target")

    # Count embeddings, capped at two: only a unique mapping is safe to transfer.
    counts = [[0] * (len(observed_chars) + 1) for _ in range(len(canonical_chars) + 1)]
    for source_index in range(len(canonical_chars) + 1):
        counts[source_index][len(observed_chars)] = 1
    for source_index in range(len(canonical_chars) - 1, -1, -1):
        for observed_index in range(len(observed_chars) - 1, -1, -1):
            count = counts[source_index + 1][observed_index]
            if canonical_chars[source_index] == observed_chars[observed_index]:
                count += counts[source_index + 1][observed_index + 1]
            counts[source_index][observed_index] = min(2, count)
    if counts[0][0] != 1:
        raise ValueError("identifier deletion projection is absent or ambiguous")

    indices: list[int] = []
    source_index = 0
    observed_index = 0
    while observed_index < len(observed_chars):
        use_count = 0
        if canonical_chars[source_index] == observed_chars[observed_index]:
            use_count = counts[source_index + 1][observed_index + 1]
        skip_count = counts[source_index + 1][observed_index]
        if use_count == 1 and skip_count == 0:
            indices.append(source_index)
            observed_index += 1
        elif use_count == 0 and skip_count == 1:
            pass
        else:  # The total-count proof above makes this unreachable unless the DP is changed.
            raise ValueError("identifier deletion projection cannot be reconstructed uniquely")
        source_index += 1
    return tuple(indices)


def _project_identifier_occurrence(
    *, source_canonical: str, source_surface: str, target_canonical: str
) -> str:
    source_value = _alphanumeric(source_canonical)
    target_value = _alphanumeric(target_canonical)
    observed_value = _alphanumeric(source_surface)
    if len(source_value) != len(target_value):
        raise ValueError("identifier target changes certified canonical width")

    def same_character_class(left: str, right: str) -> bool:
        return left.isdigit() == right.isdigit() and left.isalpha() == right.isalpha()

    def noise_character(character: str, index: int) -> str:
        alphabet = (
            "0123456789"
            if character.isdigit()
            else "abcdefghijklmnopqrstuvwxyz"
            if character.islower()
            else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        digest = sha256_bytes(
            (
                source_canonical
                + "\x00"
                + source_surface
                + "\x00"
                + target_canonical
                + "\x00"
                + str(index)
            ).encode()
        )
        selected = int(digest[:16], 16) % len(alphabet)
        if alphabet[selected].casefold() == character.casefold():
            selected = (selected + 1) % len(alphabet)
        return alphabet[selected]

    if len(observed_value) == len(source_value):
        return "".join(
            target_character
            if source_character.casefold() == observed_character.casefold()
            and same_character_class(target_character, observed_character)
            else noise_character(observed_character, index)
            for index, (source_character, observed_character, target_character) in enumerate(
                zip(source_value, observed_value, target_value, strict=True)
            )
        )

    if len(observed_value) < len(source_value):
        contiguous = tuple(
            match.start()
            for match in re.finditer(re.escape(observed_value.casefold()), source_value.casefold())
        )
        indices = (
            tuple(range(contiguous[0], contiguous[0] + len(observed_value)))
            if len(contiguous) == 1
            else _unique_subsequence_indices(source_value, observed_value)
        )
        return "".join(target_value[index] for index in indices)

    contiguous = tuple(
        match.start()
        for match in re.finditer(re.escape(source_value.casefold()), observed_value.casefold())
    )
    if len(contiguous) == 1:
        start = contiguous[0]
        source_by_observed = {
            start + source_index: source_index for source_index in range(len(source_value))
        }
    else:
        source_positions = _unique_subsequence_indices(observed_value, source_value)
        source_by_observed = {
            observed_index: source_index
            for source_index, observed_index in enumerate(source_positions)
        }
    literal_prefix_length = min(source_by_observed) if source_by_observed else 0
    literal_prefix = observed_value[:literal_prefix_length].casefold()
    preserve_prefix = literal_prefix in {"no", "nr", "ref"}
    output: list[str] = []
    for observed_index, observed_character in enumerate(observed_value):
        source_index = source_by_observed.get(observed_index)
        if source_index is None:
            output.append(
                observed_character
                if preserve_prefix and observed_index < literal_prefix_length
                else noise_character(observed_character, observed_index)
            )
            continue
        target_character = target_value[source_index]
        output.append(
            target_character
            if same_character_class(target_character, observed_character)
            else noise_character(observed_character, observed_index)
        )
    return "".join(output)


def _scalar_surface(value: Any) -> str:
    if (
        isinstance(value, bool)
        or value is None
        or not isinstance(value, (str, int, float, Decimal))
    ):
        raise ValueError(f"target value is not a renderable scalar: {value!r}")
    return str(value)


def _token_spans(value: str) -> tuple[tuple[str, int, int], ...]:
    if not value.isascii():
        # The compiler tokenizes casefolded text. Unicode casefolding can create
        # ASCII tokens (İ -> i + combining dot), so retain an explicit offset map.
        folded = value.casefold()
        offsets = tuple(i for i, char in enumerate(value) for _ in char.casefold())
        return tuple(
            (match.group(0), offsets[match.start()], offsets[match.end() - 1] + 1)
            for match in _TOKEN.finditer(folded)
        )
    return tuple(
        (match.group(0).casefold(), match.start(), match.end()) for match in _TOKEN.finditer(value)
    )


def _package_surface(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("PACKAGE_"):
        raise ValueError(f"invalid package category: {value!r}")
    return value.removeprefix("PACKAGE_").replace("_", " ")


def _semantic_equipment_value(target: Mapping[str, Any], path: str) -> dict[str, str] | None:
    match = re.fullmatch(r"documentPatch\.containers\[([0-9]+)\]\.typeDescription", path)
    if match is None:
        return None
    containers = cast(Mapping[str, Any], target["documentPatch"]).get("containers")
    if not isinstance(containers, Sequence):
        return None
    index = int(match.group(1))
    if index >= len(containers) or not isinstance(containers[index], Mapping):
        return None
    container = cast(Mapping[str, Any], containers[index])
    size = container.get("sizeCategory")
    kind = container.get("typeCategory")
    if not isinstance(size, str) or not isinstance(kind, str):
        return None
    return {"sizeCategory": size, "typeCategory": kind}


def _binding_target_value(target: Mapping[str, Any], path: str) -> JsonValue:
    try:
        return cast(JsonValue, _resolve_path(target, path))
    except ValueError:
        equipment = _semantic_equipment_value(target, path)
        if equipment is None:
            raise
        return cast(JsonValue, equipment)


def _render_signed_temperature_word_surface(source: str, old: Any, new: Any) -> str:
    """Render the sign word with the number, never as a stale literal prefix."""
    match = re.fullmatch(
        r"(?P<word>PLUS|MINUS)(?P<space>\s+)(?P<magnitude>\d+(?:[.,]\d+)?)",
        source,
        re.I,
    )
    if match is None:
        raise ValueError("certified signed temperature surface changed")
    before, after = Decimal(str(old)), Decimal(str(new))
    observed = Decimal(match["magnitude"].replace(",", "."))
    if match["word"].upper() == "MINUS":
        observed = -observed
    if not before.is_finite() or not after.is_finite() or observed != before:
        raise ValueError("signed temperature surface disagrees with its certified source")
    word = "MINUS" if after < 0 else "PLUS"
    if match["word"].islower():
        word = word.lower()
    elif match["word"].istitle():
        word = word.title()
    magnitude = render_number_surface(match["magnitude"], abs(before), abs(after))
    if magnitude.startswith(("-", "+")):
        raise ValueError("signed temperature magnitude has an extra sign")
    return word + match["space"] + magnitude


def _render_whole_slot(
    *, slot: TemplateSlot, binding: SemanticBinding, old_value: Any, new_value: Any
) -> str:
    realization = next(row for row in binding.realization.slots if row.slot_id == slot.slot_id)
    if binding.realization.adapter == "date":
        return _render_certified_date_surface(
            slot.source_text, cast(str, old_value), cast(str, new_value)
        )
    if binding.realization.adapter == "numeric":
        return render_number_surface(
            slot.source_text,
            cast(int | float, old_value),
            cast(int | float, new_value),
        )
    if binding.realization.adapter == "signed_temperature_word":
        return _render_signed_temperature_word_surface(slot.source_text, old_value, new_value)
    if binding.realization.adapter == "package_category":
        if not isinstance(new_value, str):
            raise ValueError("package category target is not text")
        return _package_candidate(slot.source_text, new_value)
    if binding.realization.adapter == "measurement_unit":
        if old_value != new_value:
            raise ValueError(
                "measurement_unit changed outside this template's deterministic vocabulary"
            )
        return slot.source_text
    target_surface = _scalar_surface(new_value)
    prefix = realization.literal_prefix
    suffix = realization.literal_suffix
    if not slot.source_text.startswith(prefix) or not slot.source_text.endswith(suffix):
        raise ValueError(f"compiled literal frame no longer matches {slot.slot_id}")
    end = len(slot.source_text) - len(suffix) if suffix else len(slot.source_text)
    source_core = slot.source_text[len(prefix) : end]
    if binding.realization.adapter == "opaque_identifier":
        candidate = _shape_alphanumeric_like_source(source_core, target_surface)
    else:
        candidate = _layout_like_source(source_core, target_surface)
    return prefix + candidate + suffix


def _partition_target_surface(value: str, slots: Sequence[TemplateSlot]) -> tuple[str, ...]:
    chunks = tuple(value.split())
    # An explicitly opaque numeric suffix is one fixed-shape component, not a
    # proportional share of the address words. Weighting it by the old address
    # length could put "States 12345" into a five-digit postal slot.
    if (
        len(slots) > 1
        and slots[-1].render_policy == "opaque_identifier"
        and re.fullmatch(r"[0-9]+", slots[-1].source_text)
        and all(slot.render_policy != "opaque_identifier" for slot in slots[:-1])
    ):
        if not chunks or not re.fullmatch(
            r"[0-9]{" + str(len(slots[-1].source_text)) + "}", chunks[-1]
        ):
            raise ValueError("target does not end with its certified numeric component shape")
        prefix = _partition_target_surface(" ".join(chunks[:-1]), slots[:-1])
        return (*prefix, chunks[-1])
    weights = tuple(max(1, len(_TOKEN.findall(slot.source_text))) for slot in slots)
    return tuple(" ".join(words) for words in _allocate_words(chunks, weights))


def _render_target_binding(
    binding: SemanticBinding,
    target: Mapping[str, Any],
    *,
    auxiliary_values: Mapping[str, str] | None = None,
    source: bytes | None = None,
    template: CertifiedSemanticTemplate | None = None,
) -> BindingOutput:
    target_values = tuple(_binding_target_value(target, path) for path in binding.target_paths)
    if not target_values:
        raise ValueError("target binding has no target values")
    if binding.realization.adapter == "temperature_instruction":
        from .temperature_prose import (
            composite_instruction_contract,
            render_composite_instruction,
        )

        if len(binding.occurrences) != 1:
            raise ValueError("temperature instruction needs one exact source span")
        slot = binding.occurrences[0]
        composite_source_values = {
            row.target_path: row.source_value for row in binding.realization.target_values
        }
        descendant_values = dict(zip(binding.target_paths, target_values, strict=True))
        contract = composite_instruction_contract(
            binding.target_paths, slot.source_text, composite_source_values
        )
        if contract is None:
            raise ValueError("temperature instruction lost its compiled source contract")
        output = render_composite_instruction(
            slot.source_text, binding.target_paths, composite_source_values, descendant_values
        )
        return BindingOutput(
            replacements={slot.slot_id: output}, canonical_value=descendant_values[contract[0]]
        )
    if len({canonical_json_bytes(value) for value in target_values}) != 1:
        raise ValueError("one deterministic binding has unequal descendant target values")
    new_value = target_values[0]
    source_values = tuple(row.source_value for row in binding.realization.target_values)
    old_value = source_values[0]
    slots = binding.occurrences
    if source_values and all(
        canonical_json_bytes(value) == canonical_json_bytes(new_value) for value in source_values
    ):
        # The compiler already proved these exact source spans realize this canonical
        # value. Keeping them byte-for-byte avoids degrading document-native enum,
        # punctuation, abbreviation, and segmentation styles when the target is unchanged.
        return BindingOutput(
            replacements={slot.slot_id: slot.source_text for slot in slots},
            canonical_value=new_value,
        )
    if isinstance(binding.realization, BindingRealization) and any(
        slot.repeat_group_index is not None for slot in binding.realization.slots
    ):
        groups: dict[int, list[int]] = {}
        for index, plan in enumerate(binding.realization.slots):
            if plan.repeat_group_index is None:
                raise ValueError("repeated binding contains an ungrouped slot")
            groups.setdefault(plan.repeat_group_index, []).append(index)
        grouped_replacements: dict[str, str] = {}
        for indexes in groups.values():
            members = tuple(binding.occurrences[index] for index in indexes)
            if (
                binding.realization.mode == "token_projected_surface"
                and binding.value_kind == "address"
                and len(members) == 2
                and "POSTAL CODE" in members[1].source_text.upper()
            ):
                source_postcode = re.search(r"([0-9]+)\s*$", members[1].source_text)
                if source_postcode is None:
                    raise ValueError("printed postcode slot has no numeric source code")
                proposed = _scalar_surface(new_value)
                match = re.fullmatch(
                    rf"(.*?)([0-9]{{{len(source_postcode[1])}}})\s*",
                    proposed,
                    flags=re.S,
                )
                if match is not None and match[1].rstrip(" ,"):
                    grouped_replacements[members[0].slot_id] = _layout_like_source(
                        members[0].source_text, match[1].rstrip(" ,")
                    )
                    grouped_replacements[members[1].slot_id] = (
                        members[1].source_text[: source_postcode.start(1)] + match[2]
                    )
                else:
                    # An alphanumeric or differently shaped postal address is
                    # printed whole; the source's numeric-only caption is not
                    # asserted for a code it cannot represent.
                    grouped_replacements[members[0].slot_id] = _layout_like_source(
                        members[0].source_text, proposed
                    )
                    grouped_replacements[members[1].slot_id] = ""
                continue
            realization = binding.realization.model_copy(
                update={
                    "slots": tuple(
                        binding.realization.slots[index].model_copy(
                            update={"repeat_group_index": None}
                        )
                        for index in indexes
                    )
                }
            )
            member = binding.model_copy(
                update={
                    "occurrences": tuple(binding.occurrences[index] for index in indexes),
                    "realization": realization,
                }
            )
            member_output = _render_target_binding(
                member,
                target,
                auxiliary_values=auxiliary_values,
                source=source,
                template=template,
            )
            grouped_replacements.update(member_output.replacements)
        if set(grouped_replacements) != {slot.slot_id for slot in slots}:
            raise ValueError("repeated binding does not render every source slot once")
        if (
            binding.value_kind == "address"
            and binding.group_kind == "party"
            and template is not None
        ):
            name_ends = {
                slot.byte_end
                for other in template.bindings
                if other.group_key == binding.group_key
                and any(path.endswith(".name") for path in other.target_paths)
                for slot in other.occurrences
            }
            for indexes in groups.values():
                anchor = binding.occurrences[indexes[0]]
                if anchor.byte_start in name_ends:
                    grouped_replacements[anchor.slot_id] = (
                        "\n" + grouped_replacements[anchor.slot_id]
                    )
        return BindingOutput(replacements=grouped_replacements, canonical_value=new_value)
    cargo_partition = lexical_partitions.partition(binding)
    if cargo_partition is not None:
        if not isinstance(new_value, str):
            raise ValueError("cargo fragment target is not text")
        return BindingOutput(
            replacements=lexical_partitions.render_parts(
                cargo_partition, new_value, auxiliary_values or {}, source=source
            ),
            canonical_value=new_value,
        )
    mode = binding.realization.mode
    replacements: dict[str, str] = {}
    if mode in {"single_surface", "repeated_surface"}:
        for slot in slots:
            replacements[slot.slot_id] = _render_whole_slot(
                slot=slot,
                binding=binding,
                old_value=old_value,
                new_value=new_value,
            )
            if binding.value_kind == "phone" and source is not None:
                replacements[slot.slot_id] = contact_values.phone_inside_literal_prefix(
                    source=source,
                    byte_start=slot.byte_start,
                    rendered=replacements[slot.slot_id],
                )
    elif mode == "segmented_surface":
        from .realization_contract import character_partition_frames

        character_frames = character_partition_frames(binding)
        if character_frames is not None:
            proposed = _scalar_surface(new_value)
            if any(c.isspace() for c in proposed) or len(proposed) < len(character_frames):
                raise ValueError("segmented scalar requires enough non-whitespace characters")
            pieces = _allocate_words(
                tuple(proposed), tuple(len(core) for _, core, _ in character_frames)
            )
            for slot, (character_prefix, core, character_suffix), piece in zip(
                slots, character_frames, pieces, strict=True
            ):
                replacements[slot.slot_id] = (
                    character_prefix + _case_like(core, "".join(piece)) + character_suffix
                )
            return BindingOutput(replacements=replacements, canonical_value=new_value)
        if binding.realization.adapter == "opaque_identifier":
            original = "".join(_alphanumeric(slot.source_text) for slot in slots)
            proposed = _alphanumeric(_scalar_surface(new_value))
            if original.casefold() != _alphanumeric(_scalar_surface(old_value)).casefold() or len(
                original
            ) != len(proposed):
                raise ValueError(
                    "segmented identifier does not preserve its proven character partition"
                )
            offset = 0
            for slot in slots:
                width = len(_alphanumeric(slot.source_text))
                replacements[slot.slot_id] = _shape_alphanumeric_like_source(
                    slot.source_text, proposed[offset : offset + width]
                )
                offset += width
            return BindingOutput(replacements=replacements, canonical_value=new_value)
        surfaces = _partition_target_surface(_scalar_surface(new_value), slots)
        for slot, surface in zip(slots, surfaces, strict=True):
            replacements[slot.slot_id] = _layout_like_source(slot.source_text, surface)
    elif mode == "token_projected_surface":
        target_surface = _scalar_surface(new_value)
        target_tokens = _token_spans(target_surface)
        intervals = token_projection_intervals(binding)
        if intervals is not None:
            boundaries = sorted({index for interval in intervals for index in interval})
            source_size = len(_token_spans(str(binding.realization.target_values[0].source_value)))
            normalized = tuple(row[0] for row in target_tokens)
            anchors = [(0, 0, 0, 0)]
            cursor = 0
            for start, end, fixed in shared_projection_ranges(binding, template, target):
                matches = [
                    i
                    for i in range(cursor, len(normalized) - len(fixed) + 1)
                    if normalized[i : i + len(fixed)] == fixed
                    and (start != 0 or i == 0)
                    and (end != source_size or i + len(fixed) == len(normalized))
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "target violates unique token-prefix, token-suffix "
                        "or literal-gap constraint"
                    )
                position = matches[0]
                anchors.append((start, end, position, position + len(fixed)))
                cursor = position + len(fixed)
            anchors.append((source_size, source_size, len(normalized), len(normalized)))
            new_boundaries = {}
            for start, end, position, stop in anchors:
                for boundary in boundaries:
                    if boundary == start:
                        new_boundaries[boundary] = position
                    elif boundary == end:
                        new_boundaries[boundary] = stop
                    elif start < boundary < end:
                        if stop - position != end - start:
                            raise ValueError("a mutable slot divides a resized shared context")
                        new_boundaries[boundary] = position + boundary - start
            for prior, following in pairwise(anchors):
                left, right = prior[1], following[0]
                new_left, new_right = prior[3], following[2]
                if left == right:
                    if new_left != new_right:
                        raise ValueError(
                            "unowned text was inserted outside projected mutable segments"
                        )
                    continue
                region = sorted({left, right, *[b for b in boundaries if left < b < right]})
                weights = tuple(b - a for a, b in pairwise(region))
                if new_right - new_left < len(weights):
                    raise ValueError("target has fewer tokens than its printed partition")
                pieces = _allocate_words(tuple(str(i) for i in range(new_left, new_right)), weights)
                new_boundaries[left] = new_left
                current = new_left
                for boundary, piece in zip(region[1:], pieces, strict=True):
                    current += len(piece)
                    new_boundaries[boundary] = current
            for slot, (start, end) in zip(slots, intervals, strict=True):
                left, right = new_boundaries[start], new_boundaries[end]
                begin = target_tokens[left][1]
                stop = target_tokens[right - 1][2]
                replacements[slot.slot_id] = _layout_like_source(
                    slot.source_text, target_surface[begin:stop]
                )
            return BindingOutput(replacements=replacements, canonical_value=new_value)
        normalized_tokens = tuple(row[0] for row in target_tokens)
        for slot, slot_plan in zip(slots, binding.realization.slots, strict=True):
            prefix = tuple(value.casefold() for value in slot_plan.required_target_prefix_tokens)
            suffix = tuple(value.casefold() for value in slot_plan.required_target_suffix_tokens)
            if normalized_tokens[: len(prefix)] != prefix:
                raise ValueError(f"target violates token-prefix constraint for {slot.slot_id}")
            if suffix and normalized_tokens[-len(suffix) :] != suffix:
                raise ValueError(f"target violates token-suffix constraint for {slot.slot_id}")
            start = len(prefix)
            stop = len(target_tokens) - len(suffix) if suffix else len(target_tokens)
            if start >= stop:
                raise ValueError(f"target projection is empty for {slot.slot_id}")
            projected = target_surface[target_tokens[start][1] : target_tokens[stop - 1][2]]
            replacements[slot.slot_id] = _layout_like_source(slot.source_text, projected)
    elif mode == "normalized_projected_surface":
        target_surface = _scalar_surface(new_value)
        target_normalized = _alphanumeric(target_surface).casefold()
        for slot, slot_plan in zip(slots, binding.realization.slots, strict=True):
            normalized_prefix = slot_plan.required_target_prefix_normalized.casefold()
            normalized_suffix = slot_plan.required_target_suffix_normalized.casefold()
            if not target_normalized.startswith(normalized_prefix) or (
                normalized_suffix and not target_normalized.endswith(normalized_suffix)
            ):
                raise ValueError(f"target violates normalized projection for {slot.slot_id}")
            stop = (
                len(target_normalized) - len(normalized_suffix)
                if normalized_suffix
                else len(target_normalized)
            )
            projected = target_normalized[len(normalized_prefix) : stop]
            if not projected:
                raise ValueError(f"normalized target projection is empty for {slot.slot_id}")
            replacements[slot.slot_id] = _shape_alphanumeric_like_source(
                slot.source_text, projected
            )
    else:
        raise ValueError(f"binding is not a deterministic target surface: {mode}")
    return BindingOutput(replacements=replacements, canonical_value=new_value)


def _parse_auxiliary_date(value: str) -> date:
    compact = value.strip()
    parsed = []
    for date_format in _DATE_FORMATS:
        try:
            candidate = datetime.strptime(compact, date_format).date()
        except ValueError:
            continue
        if candidate not in parsed:
            parsed.append(candidate)
    if len(parsed) != 1:
        raise ValueError(f"source-only date is not uniquely parseable: {value!r}")
    return parsed[0]


def _source_container_identifier(binding: SemanticBinding) -> str | None:
    tokens = set(re.findall(r"[a-z]+", binding.logical_key.lower()))
    source = _alphanumeric(binding.occurrences[0].source_text).upper()
    if (
        not binding.target_paths
        and "container" in tokens
        and not tokens & {"seal", "seals"}
        and re.fullmatch(r"[A-Z]{3}[UJZ][0-9]{7}", source)
    ):
        return source
    return None


def _render_direct_auxiliary(
    binding: SemanticBinding, stream: DeterministicStream
) -> BindingOutput:
    if _explicit_unknown_placeholder(binding):
        return _preserved_source_output(binding)
    first = binding.occurrences[0]
    policy = first.render_policy
    if binding.value_kind == "date" or policy == "date_surface":
        return _fixed_date_auxiliary(binding)
    if binding.value_kind == "email":
        return _render_text_candidate(
            binding, contact_values.mailbox(stream.derive(binding.logical_key))
        )
    if binding.value_kind == "phone":
        region = contact_values.source_phone_country(first.source_text)
        return _render_text_candidate(
            binding, contact_values.phone(stream.derive(binding.logical_key), country_code=region)
        )
    source_container = _source_container_identifier(binding)
    if source_container is not None:
        return _render_text_candidate(
            binding,
            generate_container_number(
                owner_and_category=source_container[:4],
                stream=stream.derive(binding.logical_key),
                excluded={source_container},
            ),
        )
    source_seal = (
        binding.value_kind == "equipment"
        and "seal" in set(re.findall(r"[a-z]+", binding.logical_key.lower()))
        and any(c.isdigit() for c in first.source_text)
    )
    if policy == "opaque_identifier" and (binding.value_kind == "identifier" or source_seal):
        generated = generate_from_surface_pattern(
            pattern=surface_pattern(first.source_text),
            stream=stream.derive(binding.logical_key),
            additional_excluded=first.source_text,
        )
        return _render_text_candidate(binding, generated)
    if policy == "numeric_surface":
        raise ValueError(
            "numeric auxiliary requires a proven scenario dependency; "
            "digit randomization is forbidden"
        )
    raise ValueError(f"no deterministic auxiliary renderer for {policy}")


def _render_text_candidate(binding: SemanticBinding, candidate: str) -> BindingOutput:
    if not candidate.strip():
        raise ValueError("typed auxiliary candidate is empty")
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        if binding.value_kind == "email":
            replacement = contact_values.render_mailbox(slot.source_text, candidate)
        elif slot.render_policy in {"opaque_identifier", "numeric_surface"}:
            canonical_source = _alphanumeric(binding.occurrences[0].source_text)
            observed = _alphanumeric(slot.source_text)
            value = _alphanumeric(candidate)
            # Repeated source-only identifiers may include a printed alphabetic
            # designator (EIN/REF/etc.). Prove its exact embedding in the source;
            # it is neither part of the new identifier nor random OCR noise.
            matches = tuple(re.finditer(re.escape(canonical_source), observed, re.I))
            if (
                len(value) == len(canonical_source)
                and len(matches) == 1
                and all(
                    c.isalpha()
                    for c in observed[: matches[0].start()] + observed[matches[0].end() :]
                )
            ):
                match = matches[0]
                value = observed[: match.start()] + value + observed[match.end() :]
            replacement = _shape_alphanumeric_like_source(slot.source_text, value)
        else:
            replacement = _layout_like_source(slot.source_text, candidate)
        replacements[slot.slot_id] = replacement
    return BindingOutput(replacements=replacements, canonical_value=candidate.strip())


def _fixed_date_auxiliary(binding: SemanticBinding) -> BindingOutput:
    """Retain the declared timeline; never invent an unrelated source-only date."""
    if binding.target_paths:
        raise ValueError("target-backed dates must use their target realization")
    return BindingOutput(
        replacements={slot.slot_id: slot.source_text for slot in binding.occurrences},
        canonical_value=binding.occurrences[0].source_text,
    )


def _render_generated_auxiliary(
    binding: SemanticBinding,
    *,
    stream: DeterministicStream,
    values: DeterministicValueFactory,
) -> BindingOutput:
    """Execute the compiler's provider-free ``generated_auxiliary`` promise."""

    try:
        return _render_direct_auxiliary(binding, stream)
    except ValueError:
        pass

    typed = values.textual(binding)
    if typed is not None:
        return _render_text_candidate(binding, typed)

    first = binding.occurrences[0]
    if binding.value_kind == "identifier" and first.render_policy == "opaque_identifier":
        generated = generate_from_surface_pattern(
            pattern=surface_pattern(first.source_text),
            stream=stream.derive(binding.logical_key),
            additional_excluded=first.source_text,
        )
        return _render_text_candidate(binding, generated)

    raise ValueError(
        f"no typed deterministic auxiliary generator for {binding.value_kind}: "
        f"{binding.logical_key}; source-copy substitution is forbidden"
    )


def _render_entity_auxiliary(
    binding: SemanticBinding,
    *,
    entity_member: tuple[AuxiliaryEntity, AuxiliaryEntityMember],
    country_code_style: str | None,
    geography_constraint: EntityGeographyConstraint,
    stream: DeterministicStream,
    values: DeterministicValueFactory,
    country_codes: Mapping[str, str],
    prepared_auxiliary: Mapping[str, str] | None = None,
) -> BindingOutput:
    entity, member = entity_member
    if _explicit_unknown_placeholder(binding):
        return _preserved_source_output(binding)
    if member.field == "registration_type":
        return _preserved_source_output(binding)
    if geography_constraint.fixed_country_code and member.field in {"country", "country_code"}:
        if binding.logical_key not in geography_constraint.fixed_country_bindings and any(
            country_codes.get(_alphanumeric(s.source_text).casefold())
            != geography_constraint.fixed_country_code
            for s in binding.occurrences
        ):
            raise ValueError("country surface conflicts with its immutable geographic context")
        output = _preserved_source_output(binding)
        return BindingOutput(
            replacements=output.replacements,
            canonical_value=(
                geography_constraint.fixed_country_code
                if member.field == "country_code"
                else output.canonical_value
            ),
        )
    if prepared_auxiliary is not None and binding.logical_key in prepared_auxiliary:
        return _render_text_candidate(binding, prepared_auxiliary[binding.logical_key])
    typed = values.entity_textual(
        entity=entity,
        member=member,
        country_codes=country_codes,
        calling_code_width=geography_constraint.calling_code_width,
        country_name_width=geography_constraint.country_name_width,
        postal_code_pattern=geography_constraint.postal_code_pattern,
        fixed_country_code=geography_constraint.fixed_country_code,
    )
    if typed is not None:
        if member.field == "country_code":
            if country_code_style is None:
                raise ValueError(
                    f"country-code member has no surface grammar: {binding.logical_key}"
                )
            geography: GeoProfile | None = None
            if entity.relationship == "independent":
                geography = values.geography_for_identity(
                    entity.entity_id,
                    calling_code_width=geography_constraint.calling_code_width,
                    country_name_width=geography_constraint.country_name_width,
                    postal_code_pattern=geography_constraint.postal_code_pattern,
                    fixed_country_code=geography_constraint.fixed_country_code,
                )
                if geography.country_code != typed:
                    raise ValueError("entity country code differs from its constrained geography")
            replacements: dict[str, str] = {}
            for slot in binding.occurrences:
                if country_code_style == "iso2":
                    candidate = typed
                elif geography is None and (
                    country_codes.get(_alphanumeric(slot.source_text).casefold()) == typed
                ):
                    # A target-linked country retains its certified source spelling
                    # when that spelling explicitly resolves to the same country.
                    # The canonical receipt remains ISO2, not a five-letter port code.
                    candidate = _alphanumeric(slot.source_text)
                elif country_code_style == "iso3" and geography is not None:
                    candidate = geography.alpha3
                elif country_code_style == "country_name" and geography is not None:
                    candidate = geography.country
                elif country_code_style == "unlocode" and geography is not None:
                    candidate = geography.locode
                elif country_code_style in {"calling_plus", "calling_00"} and geography is not None:
                    candidate = (
                        "00" + geography.calling_code
                        if country_code_style == "calling_00"
                        else geography.calling_code
                    )
                else:
                    raise ValueError(
                        "country-code style cannot be rendered from its entity: "
                        + binding.logical_key
                    )
                replacements[slot.slot_id] = _shape_alphanumeric_like_source(
                    slot.source_text,
                    candidate,
                )
            return BindingOutput(replacements=replacements, canonical_value=typed)
        return _render_text_candidate(binding, typed)
    if entity.relationship == "same_as_target_party" and member.field in {
        "name",
        "address",
        "city",
        "region",
        "postal_code",
        "country",
        "country_code",
        "phone_extension",
        "other",
    }:
        raise ValueError(
            f"target-linked auxiliary context is unresolved: {binding.logical_key}; "
            "a missing generated facet must not be replaced with its source surface"
        )
    if binding.value_kind in {"identifier", "email", "phone"}:
        return _render_direct_auxiliary(binding, stream)
    if entity.relationship == "same_as_target_party":
        raise ValueError(
            f"target party does not expose auxiliary field {member.field}: {binding.logical_key}"
        )
    return _render_generated_auxiliary(binding, stream=stream, values=values)


_ENGLISH_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_FRENCH_MONTHS = (
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
)
_FRENCH_MONTH_ALIASES = {
    "janv": 1,
    "janvier": 1,
    "fev": 2,
    "fevr": 2,
    "fevrier": 2,
    "mars": 3,
    "avr": 4,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "juillet": 7,
    "aout": 8,
    "sept": 9,
    "septembre": 9,
    "oct": 10,
    "octobre": 10,
    "nov": 11,
    "novembre": 11,
    "dec": 12,
    "decembre": 12,
}
_PACKAGE_SURFACES = {
    "PACKAGE_BAG": ("BAG", "BAGS"),
    "PACKAGE_BARREL": ("BBL", "BARREL", "BARRELS"),
    "PACKAGE_BOX": ("BOX", "BOXES", "BX", "BXS"),
    "PACKAGE_CARTON": ("CTN", "CTNS", "CARTON", "CARTONS"),
    "PACKAGE_CASE": ("CASE", "CASES"),
    "PACKAGE_CRATE": ("CRT", "CRATE", "CRATES"),
    "PACKAGE_DRUM": ("DRM", "DRUM", "DRUMS"),
    "PACKAGE_PACKAGE": ("PKG", "PKGS", "PACKAGE", "PACKAGES"),
    "PACKAGE_PALLET": ("PLT", "PLTS", "PALLET", "PALLETS"),
    "PACKAGE_ROLL": ("ROLL", "ROLLS"),
}


def _ascii_word(value: str) -> str:
    return (
        value.casefold()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("û", "u")
        .replace("ù", "u")
    )


def _render_certified_date_surface(raw: str, old_iso: str, new_iso: str) -> str:
    """Render an agent-labelled date from its certified canonical source value and grammar."""

    try:
        formatted = _render_date_surface(raw, old_iso, new_iso)
        if re.findall(r"\r\n|\r|\n", formatted) == re.findall(r"\r\n|\r|\n", raw):
            return formatted
    except ValueError:
        pass
    old = date.fromisoformat(old_iso)
    new = date.fromisoformat(new_iso)
    if old not in _date_candidates(raw):
        raise ValueError(f"date surface does not express its certified source value: {raw!r}")

    compact = raw.strip()
    token_matches = tuple(re.finditer(r"[0-9]+|[A-Za-zÀ-ÿ]+", compact))
    tokens = tuple(match.group(0) for match in token_matches)
    word_indices = tuple(index for index, token in enumerate(tokens) if token[0].isalpha())
    numeric_indices = tuple(index for index, token in enumerate(tokens) if token[0].isdigit())
    replacements: dict[int, str] = {}
    if len(word_indices) == 1 and len(numeric_indices) == 2:
        month_index = word_indices[0]
        source_month = _ascii_word(tokens[month_index])
        french_month = _FRENCH_MONTH_ALIASES.get(source_month)
        english_month = next(
            (
                index
                for index, name in enumerate(_ENGLISH_MONTHS, start=1)
                if source_month in {name.casefold(), name[:3].casefold()}
            ),
            None,
        )
        if french_month == old.month:
            full = len(source_month) > 4 or source_month in {"mars", "mai", "juin", "aout"}
            rendered_month = _FRENCH_MONTHS[new.month - 1]
            if not full:
                rendered_month = rendered_month[:4].rstrip("r")
        elif english_month == old.month:
            rendered_month = _ENGLISH_MONTHS[new.month - 1]
            if len(tokens[month_index]) <= 3:
                rendered_month = rendered_month[:3]
        else:
            raise ValueError(f"unrecognized certified month token: {tokens[month_index]!r}")
        replacements[month_index] = _case_like(tokens[month_index], rendered_month)
        for index in numeric_indices:
            token = tokens[index]
            number = int(token)
            if len(token) == 4 and number == old.year:
                replacements[index] = f"{new.year:04d}"
            elif len(token) == 2 and number == old.year % 100:
                replacements[index] = f"{new.year % 100:02d}"
            elif number == old.day:
                replacements[index] = f"{new.day:0{len(token)}d}"
            else:
                raise ValueError(f"date number has no certified role: {token!r}")
    elif len(word_indices) == 0 and len(numeric_indices) == 3:
        unassigned = set(numeric_indices)
        year_candidates = tuple(
            index
            for index in numeric_indices
            if int(tokens[index]) in {old.year, old.year % 100} and len(tokens[index]) in {2, 4}
        )
        if len(year_candidates) != 1:
            raise ValueError(f"numeric date has no unique year position: {raw!r}")
        year_index = year_candidates[0]
        replacements[year_index] = (
            f"{new.year:04d}" if len(tokens[year_index]) == 4 else f"{new.year % 100:02d}"
        )
        unassigned.remove(year_index)
        day_candidates = tuple(index for index in unassigned if int(tokens[index]) == old.day)
        month_candidates = tuple(index for index in unassigned if int(tokens[index]) == old.month)
        if old.day == old.month and day_candidates == month_candidates and len(day_candidates) == 2:
            if new.day != new.month:
                raise ValueError(
                    f"numeric date has no certified day/month ordering: {raw!r}; "
                    "locale guessing is forbidden"
                )
            # Either ordering produces the same new date; no locale inference is needed.
            day_candidates = (min(day_candidates),)
            month_candidates = (max(month_candidates),)
        if (
            len(day_candidates) != 1
            or len(month_candidates) != 1
            or day_candidates == month_candidates
        ):
            raise ValueError(f"numeric date has ambiguous day/month positions: {raw!r}")
        day_index = day_candidates[0]
        month_index = month_candidates[0]
        replacements[day_index] = f"{new.day:0{len(tokens[day_index])}d}"
        replacements[month_index] = f"{new.month:0{len(tokens[month_index])}d}"
    else:
        raise ValueError(f"date grammar is unsupported: {raw!r}")

    pieces: list[str] = []
    cursor = 0
    for index, match in enumerate(token_matches):
        pieces.append(compact[cursor : match.start()])
        pieces.append(replacements.get(index, match.group(0)))
        cursor = match.end()
    pieces.append(compact[cursor:])
    rendered = "".join(pieces)
    if new not in _date_candidates(rendered):
        raise ValueError(f"rendered date does not round-trip: {rendered!r}")
    prefix = raw[: len(raw) - len(raw.lstrip())]
    suffix = raw[len(raw.rstrip()) :]
    return prefix + rendered + suffix


def _package_candidate(source: str, value: str) -> str:
    # Abbreviations are optional presentation choices, not the package vocabulary.
    # Every validated category has a complete readable surface, including material
    # qualifiers (e.g. FIBRE DRUM), which must never be dropped to fit an old alias.
    words = _package_surface(value).split()
    if len(words) > 1 and words[0] in {
        "BAG",
        "BARREL",
        "BOX",
        "CASE",
        "CRATE",
        "DRUM",
        "PALLET",
        "SACK",
    }:
        words = words[1:] + words[:1]
    readable = " ".join(words)
    candidates = (*_PACKAGE_SURFACES.get(value, ()), readable)
    optional_plural = re.fullmatch(
        r"(?P<leading>\s*)(?P<noun>[A-Za-z]+)\((?P<suffix>ES|S)\)(?P<trailing>\s*)",
        source,
        re.IGNORECASE,
    )
    if optional_plural is not None:
        noun = readable
        plural_suffix = (
            "ES"
            if noun.casefold().endswith(("s", "x", "z", "ch", "sh"))
            else "IES"
            if noun.casefold().endswith("y") and len(noun) > 1
            else "S"
        )
        rendered_noun = noun[:-1] if plural_suffix == "IES" else noun
        return (
            optional_plural.group("leading")
            + _case_like(optional_plural.group("noun"), rendered_noun)
            + "("
            + _case_like(optional_plural.group("suffix"), plural_suffix)
            + ")"
            + optional_plural.group("trailing")
        )
    source_length = len(_alphanumeric(source))
    candidate = min(
        candidates,
        key=lambda row: (abs(len(_alphanumeric(row)) - source_length), len(row), row),
    )
    return _layout_like_source(source, candidate)


def _slots_are_repeated_realizations(binding: SemanticBinding) -> bool:
    if len(binding.occurrences) < 2:
        return False
    normalized = tuple(_normalized_semantic(slot.source_text) for slot in binding.occurrences)
    if any(not value for value in normalized):
        return False
    return (
        min(
            difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()
            for index, left in enumerate(normalized)
            for right in normalized[index + 1 :]
        )
        >= 0.62
    )


def _partition_semantic_surface(value: str, slots: Sequence[TemplateSlot]) -> tuple[str, ...]:
    """Split a composite surface on observed semantic boundaries before token weighting."""

    if len(slots) < 2:
        return (value,)
    first_source = slots[0].source_text.strip()
    if (
        first_source
        and len(_normalized_semantic(first_source)) >= 4
        and value.casefold().startswith(first_source.casefold())
    ):
        remainder = value[len(first_source) :].lstrip()
        if remainder:
            return (
                value[: len(first_source)],
                *_partition_semantic_surface(remainder, slots[1:]),
            )

    clauses = tuple(row.strip() for row in re.findall(r"[^,]+(?:,|$)", value) if row.strip())
    if len(clauses) >= len(slots):
        weights = tuple(max(1, len(_TOKEN.findall(slot.source_text))) for slot in slots)
        remaining_clauses = len(clauses)
        remaining_weight = sum(weights)
        groups: list[str] = []
        cursor = 0
        for index, weight in enumerate(weights):
            remaining_slots = len(slots) - index
            if remaining_slots == 1:
                take = remaining_clauses
            else:
                proportional = round(remaining_clauses * weight / remaining_weight)
                take = max(1, min(proportional, remaining_clauses - remaining_slots + 1))
            groups.append(" ".join(clauses[cursor : cursor + take]))
            cursor += take
            remaining_clauses -= take
            remaining_weight -= weight
        return tuple(groups)
    return _partition_target_surface(value, slots)


def _composite_target_surface(binding: SemanticBinding, values: Sequence[JsonValue]) -> str:
    strings = tuple(
        value for value in values if isinstance(value, str) and not value.startswith("PACKAGE_")
    )
    if not strings:
        raise ValueError("composite target has no textual surface")
    longest = max(strings, key=lambda value: len(_normalized_semantic(value)))
    if all(
        _string_semantics_match(value, longest)
        if isinstance(value, str)
        else isinstance(value, (int, float))
        and not isinstance(value, bool)
        and Decimal(str(value)) in _numeric_surface_values(longest)
        for value in values
    ):
        return longest
    if any(isinstance(value, str) and value.startswith("PACKAGE_") for value in values):
        raise ValueError("composite package surface needs a human-readable semantic composition")
    if binding.value_kind == "package" and any("PACK" in value.upper() for value in strings):
        return longest
    if binding.value_kind in {"address", "location", "other_text"}:
        unique = tuple(dict.fromkeys(value.strip() for value in strings if value.strip()))
        return ", ".join(unique)
    raise ValueError("multiple target values have no deterministic textual composition")


def _package_summary_output(
    binding: SemanticBinding, source_target: Mapping[str, Any], target: Mapping[str, Any]
) -> BindingOutput | None:
    text_paths = [
        p
        for p in binding.target_paths
        if re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.additionalInformation\[\d+\]", p)
    ]
    if len(text_paths) != 1 or len(binding.target_paths) == 1:
        return None
    old = _resolve_path(source_target, text_paths[0])
    new = _resolve_path(target, text_paths[0])
    values = [_binding_target_value(target, p) for p in binding.target_paths]
    categories = []
    for path, value in zip(binding.target_paths, values, strict=True):
        if path == text_paths[0]:
            continue
        if isinstance(value, str):
            if not _string_semantics_match(value, new):
                return None
        elif re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", path):
            if not _composite_package_quantity_matches(binding, target, path, new):
                return None
            package = _resolve_path(target, path.removesuffix(".quantity"))
            original = _resolve_path(source_target, path.removesuffix(".quantity"))
            for key in ("typeCategory", "typeDescription"):
                if key in package and package.get(key) == original.get(key):
                    categories.append(package[key])
        elif isinstance(value, int) and path.endswith(".packageQuantity"):
            if not re.search(rf"(?<![A-Za-z0-9]){value}(?![0-9])", new):
                return None
        else:
            return None
    replacements = {}
    for slot in binding.occurrences:
        if old.casefold() == slot.source_text.casefold():
            surface = new
        elif old.casefold().startswith(slot.source_text.casefold()):
            suffix = old[len(slot.source_text) :]
            if (
                not suffix.strip()
                or not any(
                    _string_semantics_match(category, suffix.strip()) for category in categories
                )
                or not new.casefold().endswith(suffix.casefold())
            ):
                return None
            # The category has its own printed slot; do not duplicate it inside
            # the preceding quantity/description slot.
            surface = new[: -len(suffix)]
        else:
            return None
        replacements[slot.slot_id] = _layout_like_source(slot.source_text, surface)
    return BindingOutput(replacements=replacements, canonical_value=values)


def _annotated_cargo_output(
    binding: SemanticBinding, source_target: Mapping[str, Any], target: Mapping[str, Any]
) -> BindingOutput | None:
    """Print complete item rows whose parenthetical annotations interrupt the label.

    Both the source and generated description must equal the ordered item names
    with their separately labelled annotations removed. This is not loose token
    subsequence matching and cannot hide missing/reordered product identities.
    """
    if binding.value_kind != "cargo_text" or len(binding.occurrences) != 1:
        return None
    descriptions = [p for p in binding.target_paths if p.endswith(".description")]
    rows = [p for p in binding.target_paths if re.search(r"\.additionalInformation\[\d+\]$", p)]
    if len(descriptions) != 1 or len(rows) < 2 or len(rows) + 1 != len(binding.target_paths):
        return None
    slot = binding.occurrences[0]
    lines = re.split(r"\r\n|\r|\n", slot.source_text)
    if len(lines) != len(rows):
        return None
    for line, path in zip(lines, rows, strict=True):
        if _normalized_semantic(line) != _normalized_semantic(_resolve_path(source_target, path)):
            return None
    for document in (source_target, target):
        names = " ".join(
            re.sub(r"\([^()]*\)", "", _resolve_path(document, p)).strip() for p in rows
        )
        if _normalized_semantic(names) != _normalized_semantic(
            _resolve_path(document, descriptions[0])
        ):
            return None
    endings = re.findall(r"\r\n|\r|\n", slot.source_text)
    rendered = [
        _layout_like_source(line, _resolve_path(target, path))
        for line, path in zip(lines, rows, strict=True)
    ]
    surface = rendered[0] + "".join(a + b for a, b in zip(endings, rendered[1:], strict=True))
    return BindingOutput(
        replacements={slot.slot_id: surface},
        canonical_value=[_binding_target_value(target, p) for p in binding.target_paths],
    )


def _party_country_annotation_output(
    binding: SemanticBinding, source_target: Mapping[str, Any], target: Mapping[str, Any]
) -> BindingOutput | None:
    """Compose frozen party facts only when the source proves a country annotation.

    Parenthesis closure may belong to the literal outside the slot. Preserve that
    ownership instead of inventing another name or duplicating a closing bracket.
    """
    names = [
        p
        for p in binding.target_paths
        if p.startswith("documentPatch.parties.") and p.endswith(".name")
    ]
    if len(binding.target_paths) != 2 or len(names) != 1:
        return None
    name_path = names[0]
    country_path = name_path.removesuffix(".name") + ".country"
    if country_path not in binding.target_paths:
        return None
    old_name, old_country = (_resolve_path(source_target, p) for p in (name_path, country_path))
    name, country = (_resolve_path(target, p) for p in (name_path, country_path))
    if not all(isinstance(v, str) and v.strip() for v in (old_name, old_country, name, country)):
        raise ValueError("party annotation requires complete textual name and country facts")
    source_meanings = {_normalized_semantic(old_name), _normalized_semantic(old_name + old_country)}
    suffix = re.search(r"\s*\(\s*" + re.escape(country) + r"\s*\)[,.;]?\s*$", name, re.I)
    core = name[: suffix.start()].rstrip() if suffix else name
    if not core:
        raise ValueError("party annotation has no organization name")
    replacements = {}
    for slot in binding.occurrences:
        match = re.fullmatch(
            r"(?s)(.+?)\s*\(\s*" + re.escape(old_country) + r"(?P<ending>\)?[,.;]?)",
            slot.source_text,
            re.I,
        )
        if match is None or _normalized_semantic(slot.source_text) not in source_meanings:
            return None
        candidate = core + " (" + country + match["ending"]
        replacements[slot.slot_id] = _layout_like_source(slot.source_text, candidate)
    return BindingOutput(
        replacements=replacements,
        canonical_value=[_binding_target_value(target, p) for p in binding.target_paths],
    )


def _country_occurrence_surface(
    source_text: str, country: str, country_codes: Mapping[str, str]
) -> str:
    """Keep a printed country code compact and a printed country name complete."""
    code = country_codes.get(_normalized_semantic(country))
    if code is None:
        raise ValueError("synthetic country is absent from the pinned registry")
    compact = _alphanumeric(source_text)
    if len(compact) in {2, 3} and compact.isascii() and compact.isalpha():
        # Registry aliases are all exact country identities. Prefer the shortest
        # lexicographic registered spelling of the same printed width; in the
        # pinned ISO registry these include ISO2/ISO3, e.g. UAE -> GBR, CN -> IN.
        choices = sorted(
            alias.upper()
            for alias, owner in country_codes.items()
            if owner == code and len(alias) == len(compact) and alias.isascii() and alias.isalpha()
        )
        if not choices:
            raise ValueError("country has no registered alias matching the printed code width")
        return _shape_alphanumeric_like_source(source_text, choices[0])
    return _layout_like_source(source_text, country)


def _render_agent_target_binding(
    binding: SemanticBinding,
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    source: bytes | None = None,
    country_codes: Mapping[str, str] | None = None,
) -> BindingOutput:
    """Execute agent-labelled mappings whose target-to-surface transform is fully typed."""

    values = tuple(_binding_target_value(target, path) for path in binding.target_paths)
    if not values:
        raise ValueError("agent target binding has no target values")
    source_values = tuple(row.source_value for row in binding.realization.target_values)
    if (
        len(binding.occurrences) > 1
        and all(path.endswith(".country") for path in binding.target_paths)
        and len(set(map(canonical_json_bytes, values))) == 1
        and isinstance(values[0], str)
    ):
        if country_codes is None:
            raise ValueError("multiple country surfaces require the pinned country registry")
        original_codes = {
            country_codes.get(_normalized_semantic(str(value))) for value in source_values
        }
        observed_codes = {
            country_codes.get(_normalized_semantic(slot.source_text))
            for slot in binding.occurrences
        }
        if (
            len(original_codes) == 1
            and None not in original_codes
            and observed_codes == original_codes
        ):
            if country_codes.get(_normalized_semantic(values[0])) is None:
                raise ValueError("synthetic country is absent from the pinned registry")
            # Each source alias is a COMPLETE country fact, not one segment.
            return BindingOutput(
                replacements={
                    slot.slot_id: _country_occurrence_surface(
                        slot.source_text, values[0], country_codes
                    )
                    for slot in binding.occurrences
                },
                canonical_value=values[0],
            )
    summary = _package_summary_output(binding, source_target, target)
    if summary is not None:
        return summary
    annotated = _annotated_cargo_output(binding, source_target, target)
    if annotated is not None:
        return annotated
    party_annotation = _party_country_annotation_output(binding, source_target, target)
    if party_annotation is not None:
        return party_annotation
    if len(binding.target_paths) == 2:
        quantity_paths = [
            p
            for p in binding.target_paths
            if re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", p)
        ]
        if len(quantity_paths) == 1:
            path = quantity_paths[0]
            category_path = path.removesuffix("quantity") + "typeCategory"
            if category_path in binding.target_paths:
                category = _resolve_path(target, category_path)
                if category == _resolve_path(source_target, category_path):
                    old, new = _resolve_path(source_target, path), _resolve_path(target, path)
                    package_replacements = {}
                    for slot in binding.occurrences:
                        match = re.fullmatch(
                            r"\s*([0-9][0-9,]*)\s*([A-Za-z ]+)\s*", slot.source_text
                        )
                        if match is None or not _string_semantics_match(category, match.group(2)):
                            break
                        package_replacements[slot.slot_id] = render_number_surface(
                            slot.source_text, old, new
                        )
                    if len(package_replacements) == len(binding.occurrences):
                        return BindingOutput(
                            replacements=package_replacements, canonical_value=list(values)
                        )
    if binding.value_kind == "date" and len(values) == len(source_values) == 1:
        old, new = source_values[0], values[0]
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError("date target values are not ISO strings")
        return BindingOutput(
            replacements={
                slot.slot_id: _render_certified_date_surface(slot.source_text, old, new)
                for slot in binding.occurrences
            },
            canonical_value=new,
        )
    if (
        (
            binding.value_kind == "identifier"
            or all(path.endswith(".containerNumber") for path in binding.target_paths)
        )
        and len(set(map(canonical_json_bytes, values))) == 1
        and len(set(map(canonical_json_bytes, source_values))) == 1
        and isinstance(values[0], str)
        and isinstance(source_values[0], str)
    ):
        source_canonical = source_values[0]
        target_canonical = values[0]
        source_target_value = _binding_target_value(source_target, binding.target_paths[0])
        if source_target_value != source_canonical:
            raise ValueError("compiled identifier snapshot differs from the source target")
        return BindingOutput(
            replacements={
                slot.slot_id: _shape_alphanumeric_like_source(
                    slot.source_text,
                    _project_identifier_occurrence(
                        source_canonical=source_canonical,
                        source_surface=slot.source_text,
                        target_canonical=target_canonical,
                    ),
                )
                for slot in binding.occurrences
            },
            canonical_value=target_canonical,
        )
    if binding.value_kind == "package" and len(values) == 1 and isinstance(values[0], str):
        return BindingOutput(
            replacements={
                slot.slot_id: _package_candidate(slot.source_text, values[0])
                for slot in binding.occurrences
            },
            canonical_value=values[0],
        )
    if binding.value_kind == "phone" and len(values) == 1 and isinstance(values[0], str):
        replacements: dict[str, str] = {}
        for slot in binding.occurrences:
            match = re.fullmatch(
                r"(?P<phone>\+?[0-9][0-9 ()+./-]*[0-9])(?P<suffix>.*)",
                slot.source_text,
                flags=re.DOTALL,
            )
            if match is None or len(re.sub(r"[^0-9]", "", match.group("phone"))) < 7:
                raise ValueError("phone surface has no uniquely typed numeric prefix")
            replacements[slot.slot_id] = _layout_like_source(
                slot.source_text,
                values[0] + match.group("suffix"),
            )
            if source is not None:
                replacements[slot.slot_id] = contact_values.phone_inside_literal_prefix(
                    source=source,
                    byte_start=slot.byte_start,
                    rendered=replacements[slot.slot_id],
                )
        return BindingOutput(replacements=replacements, canonical_value=values[0])
    if _is_equipment_receipt_binding(binding):
        return _render_equipment_receipt_binding(
            binding,
            source_target=source_target,
            target=target,
        )
    if all(
        isinstance(value, Mapping) and set(value) == {"sizeCategory", "typeCategory"}
        for value in values
    ):
        if len({canonical_json_bytes(value) for value in values}) == 1:
            return _render_semantic_equipment_binding(binding, target)
        descriptions = tuple(
            _equipment_description(cast(Mapping[str, Any], value), apostrophe=True)
            for value in values
        )
        candidate = " / ".join(descriptions)
        return BindingOutput(
            replacements={
                slot.slot_id: _layout_like_source(slot.source_text, candidate)
                for slot in binding.occurrences
            },
            canonical_value=cast(JsonValue, list(values)),
        )
    if binding.value_kind == "operational_text":
        raise ValueError(f"{binding.value_kind} may contain an unowned contextual fragment")

    candidate = (
        _scalar_surface(values[0])
        if len(values) == 1
        else _composite_target_surface(binding, values)
    )
    if (
        len(binding.occurrences) == 1
        or binding.value_kind in {"commercial_text", "legal_text"}
        or _slots_are_repeated_realizations(binding)
    ):
        replacements = {
            slot.slot_id: _layout_like_source(slot.source_text, candidate)
            for slot in binding.occurrences
        }
    else:
        surfaces = _partition_semantic_surface(candidate, binding.occurrences)
        replacements = {
            slot.slot_id: _layout_like_source(slot.source_text, surface)
            for slot, surface in zip(binding.occurrences, surfaces, strict=True)
        }
    canonical: JsonValue = cast(JsonValue, values[0] if len(values) == 1 else list(values))
    return BindingOutput(replacements=replacements, canonical_value=canonical)


def _direct_auxiliary_route(binding: SemanticBinding) -> tuple[bool, str]:
    """Prove that a source-only field has an audited deterministic generator."""

    slots = binding.occurrences
    first = slots[0]
    if binding.value_kind == "email":
        if any(slot.render_policy != "natural_text" for slot in slots):
            return False, "email requires a lexical mailbox envelope, not identifier punctuation"
        return True, "typed valid mailbox generator is available"
    if binding.value_kind == "phone":
        try:
            contact_values.source_phone_country(first.source_text)
        except ValueError as error:
            return False, str(error)
        if any(s.render_policy != "natural_text" for s in slots):
            return False, "phone requires a typed lexical envelope"
        return True, "country-valid international phone generator is available"
    if first.render_policy == "date_surface":
        try:
            parsed_date = _parse_auxiliary_date(first.source_text)
            for slot in slots:
                if _parse_auxiliary_date(slot.source_text) != parsed_date:
                    return False, "repeated date surfaces do not encode one source date"
                _render_date_surface(
                    slot.source_text,
                    parsed_date.isoformat(),
                    date(2031, 9, 23).isoformat(),
                )
        except ValueError as error:
            return False, f"date surface lacks one unambiguous render style: {error}"
        return True, "typed format-preserving date generator is available"
    if first.render_policy == "numeric_surface":
        return False, "numeric auxiliaries require an explicit dependency contract"
    if first.render_policy == "opaque_identifier":
        if binding.value_kind != "identifier":
            return False, "opaque surface is a semantic vocabulary value, not an identifier"
        if not any(character.isdigit() for character in first.source_text):
            return False, "alphabetic-only surface may be a semantic code or descriptor"
        words = {token.casefold() for token in _TOKEN.findall(first.source_text)}
        if words & _NUMBER_WORDS:
            return False, "identifier surface contains a coupled number-word expression"
        length = len(_alphanumeric(first.source_text))
        if any(len(_alphanumeric(slot.source_text)) != length for slot in slots):
            return False, "repeated identifier surfaces have unequal alphanumeric lengths"
        return True, "typed format-preserving opaque identifier generator is available"
    return False, "no audited typed generator exists for this source-only surface"


def _stable_source_vocabulary(binding: SemanticBinding) -> bool:
    from .numeric_auxiliary import measurement_unit_binding

    if measurement_unit_binding(binding):
        return True
    if binding.target_paths:
        return False
    tokens = frozenset(
        token for token in re.split(r"[^a-z0-9]+", binding.logical_key.casefold()) if token
    )
    return bool(
        {"registration", "type"} <= tokens
        or {"original", "count"} <= tokens
        or {"seal", "status"} <= tokens
    )


def _explicit_unknown_placeholder(binding: SemanticBinding) -> bool:
    if (
        binding.value_kind == "location"
        and binding.group_kind == "route"
        and not binding.target_paths
    ):
        return bool(binding.occurrences) and all(
            re.fullmatch(r"X{3,}", slot.source_text.strip(), re.I) is not None
            for slot in binding.occurrences
        )
    if binding.value_kind == "email" and not binding.target_paths:
        return bool(binding.occurrences) and all(
            " ".join(slot.source_text.upper().split())
            in {"NA", "N/A", "NIL", "NONE", "NOT PROVIDED", "NOT AVAILABLE", "-", "--", "---"}
            for slot in binding.occurrences
        )
    return bool(
        not binding.target_paths
        and binding.value_kind == "identifier"
        and binding.occurrences
        and all(
            slot.source_text.strip() and not _alphanumeric(slot.source_text)
            for slot in binding.occurrences
        )
    )


def _direct_unbound_identifier(binding: SemanticBinding) -> bool:
    if (
        binding.target_paths
        or binding.source_relationships
        or binding.value_kind != "identifier"
        or any(slot.render_policy != "opaque_identifier" for slot in binding.occurrences)
    ):
        return False
    first = binding.occurrences[0]
    if not any(character.isdigit() for character in first.source_text):
        return False
    pattern = first.format_envelope.exact_surface_pattern
    return pattern is not None and all(
        slot.format_envelope.exact_surface_pattern == pattern for slot in binding.occurrences
    )


def _active_identifier_relationships(
    template: CertifiedSemanticTemplate,
) -> dict[str, tuple[SourceBindingRelationship, ...]]:
    """Return relationships whose embedded identifier meets the compiler's evidence floor."""

    bindings = {binding.logical_key: binding for binding in template.bindings}
    return {
        binding.logical_key: tuple(
            relationship
            for relationship in binding.source_relationships
            if min(
                len(_alphanumeric(binding.occurrences[0].source_text)),
                len(
                    _alphanumeric(
                        bindings[relationship.dependency_binding].occurrences[0].source_text
                    )
                ),
            )
            >= 6
        )
        for binding in template.bindings
    }


def _solve_identifier_relationships(
    *,
    template: CertifiedSemanticTemplate,
    outputs: dict[str, BindingOutput],
    stream: DeterministicStream,
    equivalence_groups: tuple[tuple[str, ...], ...] = (),
    fixed_prefixes: Mapping[str, str] | None = None,
) -> None:
    """Solve exact identifier-containment constraints at character-position level."""

    bindings = {binding.logical_key: binding for binding in template.bindings}
    active_relationships = _active_identifier_relationships(template)
    related_keys = {
        key
        for binding in template.bindings
        for key in (
            binding.logical_key,
            *(row.dependency_binding for row in active_relationships[binding.logical_key]),
        )
        if active_relationships[binding.logical_key]
    }
    related_keys.update(key for keys in equivalence_groups for key in keys)
    related_keys.update(fixed_prefixes or {})
    if not related_keys:
        return
    for key in related_keys:
        binding = bindings[key]
        if binding.value_kind != "identifier" or any(
            slot.render_policy != "opaque_identifier" for slot in binding.occurrences
        ):
            raise ValueError(f"identifier relationship has a non-opaque endpoint: {key}")
        if key not in outputs:
            raise ValueError(f"identifier relationship endpoint has no initial output: {key}")

    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    source_values = {
        key: _alphanumeric(bindings[key].occurrences[0].source_text) for key in related_keys
    }
    for key, value in source_values.items():
        if not value or any(
            len(_alphanumeric(slot.source_text)) != len(value) for slot in bindings[key].occurrences
        ):
            raise ValueError(f"identifier relationship changes width across occurrences: {key}")
        for index in range(len(value)):
            find((key, index))

    for binding in template.bindings:
        for relationship in active_relationships[binding.logical_key]:
            if relationship.relationship == "embeds_exact_source_identifier":
                container_key = binding.logical_key
                embedded_key = relationship.dependency_binding
            else:
                container_key = relationship.dependency_binding
                embedded_key = binding.logical_key
            container = source_values[container_key]
            embedded = source_values[embedded_key]
            offsets = tuple(
                match.start()
                for match in re.finditer(re.escape(embedded.casefold()), container.casefold())
            )
            if len(offsets) != 1:
                raise ValueError(
                    "identifier relationship lacks one exact source position: "
                    f"{container_key} -> {embedded_key}"
                )
            for embedded_index in range(len(embedded)):
                union(
                    (container_key, offsets[0] + embedded_index),
                    (embedded_key, embedded_index),
                )

    for keys in equivalence_groups:
        anchor = keys[0]
        for key in keys[1:]:
            if source_values[key].casefold() != source_values[anchor].casefold():
                raise ValueError("equivalent identifier contract lacks equal source evidence")
            for index in range(len(source_values[anchor])):
                union((anchor, index), (key, index))

    fixed: dict[tuple[str, int], str] = {}
    for key, prefix in (fixed_prefixes or {}).items():
        for index, character in enumerate(prefix):
            root = find((key, index))
            if fixed.setdefault(root, character).casefold() != character.casefold():
                raise ValueError("identifier country prefixes conflict in one component")
    for key in related_keys:
        binding = bindings[key]
        if not binding.target_paths and _source_container_identifier(binding) is None:
            continue
        rendered = _alphanumeric(outputs[key].replacements[binding.occurrences[0].slot_id])
        if len(rendered) != len(source_values[key]):
            raise ValueError(f"target identifier changes relationship width: {key}")
        for index, character in enumerate(rendered):
            root = find((key, index))
            existing = fixed.setdefault(root, character)
            if existing.casefold() != character.casefold():
                raise ValueError("synthetic target values conflict inside an identifier component")

    generated: dict[tuple[str, int], str] = {}
    alphabet_by_root: dict[tuple[str, int], str] = {}
    source_characters_by_root: dict[tuple[str, int], tuple[str, ...]] = {}
    roots = sorted({find(node) for node in parent})
    for ordinal, root in enumerate(roots):
        members = tuple(node for node in parent if find(node) == root)
        source_characters = tuple(source_values[key][index] for key, index in members)
        if all(character.isdigit() for character in source_characters):
            alphabet = "0123456789"
        elif all(character.isalpha() for character in source_characters):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        else:
            raise ValueError("identifier relationship joins incompatible character classes")
        alphabet_by_root[root] = alphabet
        source_characters_by_root[root] = source_characters
        generated[root] = fixed.get(
            root,
            alphabet[
                stream.derive("relationship:" + "|".join(sorted(related_keys))).randbelow(
                    len(alphabet), counter=ordinal
                )
            ],
        )

    # Random generation may legitimately draw every original character, especially when a
    # target-bound identifier fixes most components.  Resolve that collision by changing one
    # free connected component, preserving every containment relationship by construction.
    for key in sorted(related_keys):
        binding = bindings[key]
        if binding.target_paths:
            continue
        candidate = "".join(
            generated[find((key, index))] for index in range(len(source_values[key]))
        )
        if candidate.casefold() != source_values[key].casefold():
            continue
        mutable_roots = tuple(
            sorted(
                {
                    find((key, index))
                    for index in range(len(source_values[key]))
                    if find((key, index)) not in fixed
                }
            )
        )
        if not mutable_roots:
            # The source-only surface can be a strict projection of an unchanged target-bound
            # identifier (for example, a printed 9-digit classification prefix of a 12-digit
            # HS code).  Every component is then fixed by the structured target; preserving the
            # source projection is required for consistency and is not a generator collision.
            continue
        selected = mutable_roots[
            stream.derive("relationship-novelty-root:" + key).randbelow(len(mutable_roots))
        ]
        source_character_set = {
            character.casefold() for character in source_characters_by_root[selected]
        }
        alternatives = tuple(
            character
            for character in alphabet_by_root[selected]
            if character.casefold() not in source_character_set
        )
        if not alternatives:
            raise ValueError(f"identifier relationship has no source-distinct character: {key}")
        generated[selected] = alternatives[
            stream.derive("relationship-novelty-value:" + key).randbelow(len(alternatives))
        ]

    for key in sorted(related_keys):
        binding = bindings[key]
        if binding.target_paths:
            continue
        candidate = "".join(
            generated[find((key, index))] for index in range(len(source_values[key]))
        )
        if candidate.casefold() == source_values[key].casefold() and any(
            find((key, index)) not in fixed for index in range(len(source_values[key]))
        ):
            raise RuntimeError(
                f"identifier relationship novelty repair failed for source identifier: {key}"
            )
        outputs[key] = BindingOutput(
            replacements={
                slot.slot_id: _shape_alphanumeric_like_source(slot.source_text, candidate)
                for slot in binding.occurrences
            },
            canonical_value=candidate,
        )


def _semantic_equipment_binding_value(
    binding: SemanticBinding, target: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    if not binding.target_paths:
        return None
    values = tuple(_binding_target_value(target, path) for path in binding.target_paths)
    if not all(
        isinstance(value, Mapping) and set(value) == {"sizeCategory", "typeCategory"}
        for value in values
    ):
        return None
    first = cast(Mapping[str, Any], values[0])
    if any(canonical_json_bytes(value) != canonical_json_bytes(first) for value in values[1:]):
        return None
    return first


def _has_semantic_equipment_values(binding: SemanticBinding, target: Mapping[str, Any]) -> bool:
    return any(
        isinstance(value, Mapping) and set(value) == {"sizeCategory", "typeCategory"}
        for value in (_binding_target_value(target, path) for path in binding.target_paths)
    )


def _equipment_surface_candidates(value: Mapping[str, Any], source: str) -> tuple[str, ...]:
    from document_ocr.synthesis.container_semantics import iso_equipment_surface

    size = value.get("sizeCategory")
    equipment_type = value.get("typeCategory")
    if not isinstance(size, str) or not isinstance(equipment_type, str):
        raise ValueError("semantic equipment value lacks size/type categories")
    if size.startswith("TWENTY_"):
        length = "20"
    elif size.startswith("FORTY_FIVE_"):
        length = "45"
    elif size.startswith("FORTY_"):
        length = "40"
    else:
        raise ValueError(f"unsupported semantic equipment size: {size}")
    try:
        type_code = _EQUIPMENT_TYPE_CODES[equipment_type]
    except KeyError as error:
        raise ValueError(f"unsupported semantic equipment type: {equipment_type}") from error

    type_words = equipment_type.replace("_", " ")
    values = [
        f"{length}{type_code}",
        f"{type_code}{length}",
        f"{length} {type_code}",
        f"{length}FT {type_code}",
        f"{length} FT {type_code}",
        f"{length} {type_words}",
        canonical_equipment_surface(
            cast(ContainerSizeCategory, size), cast(ContainerTypeCategory, equipment_type)
        ),
    ]
    iso = iso_equipment_surface(
        cast(ContainerSizeCategory, size), cast(ContainerTypeCategory, equipment_type), source
    )
    if iso is not None:
        values.insert(0, iso)
    high_cube = "HIGH_CUBE" in size
    if equipment_type == "GENERAL_PURPOSE":
        values.extend(
            (
                f"{length} DRY",
                f"{length} DRY GP",
                f"{length} DRY GP TYPE",
                f"{length} GP GENERAL",
                f"{length} GENERAL PURPOSE",
            )
        )
        if high_cube:
            values.extend(
                (
                    f"{length}HC",
                    f"{length}HQ",
                    f"{length} HC",
                    f"{length} HQ",
                    f"{length} HIGH CUBE",
                    f"{length} HI CUBE",
                    f"{length} DRY HIGH CUBE",
                    f"{length} DRY HI CUBE",
                    f"{length} DRY 9 6",
                )
            )
    elif high_cube:
        values.extend(
            (
                f"{length} HC {type_code}",
                f"{length} HIGH CUBE {type_words}",
                f"{length} 9 6 {type_words}",
            )
        )
    if equipment_type == "REFRIGERATED":
        values.extend((f"{length}RF", f"{length} RF", f"{length} REEFER"))
        height = "9" if high_cube else "8"
        values.append(f"{length} {height} 6 REEFER")
        values.append(f"{length} REEF {height} 6")
        if high_cube:
            values.append(f"{length} HIGH CUBE REEFER")

    values.extend(
        candidate + " CONTAINER" for candidate in tuple(values) if "CONTAINER" not in candidate
    )
    operational = re.search(r"\s+(?:FCL/FCL|FCL/LCL|LCL/FCL|LCL/LCL)\s*$", source, re.I)
    if operational is not None:
        # Freight handling is independent of equipment category, not disposable
        # decoration when a source combines the two in one type-description slot.
        values = [candidate + operational[0] for candidate in values]

    count = re.match(r"\s*([1-9][0-9]*)\s*[xX]", source)
    if count is not None:
        prefix = count.group(1) + "X"
        decorated: list[str] = []
        for candidate in values:
            decorated.extend(
                (
                    prefix + candidate,
                    prefix + candidate + " CONTAINER",
                    prefix + candidate + " CONTAINER ONLY",
                )
            )
        values = decorated + values
    return tuple(dict.fromkeys(values))


def _render_semantic_equipment_binding(
    binding: SemanticBinding, target: Mapping[str, Any]
) -> BindingOutput:
    if binding.realization.mode == "normalized_projected_surface":
        from .equipment_projection import validate_source_surfaces

        validate_source_surfaces(binding)
    value = _semantic_equipment_binding_value(binding, target)
    if value is None:
        raise ValueError("binding does not resolve to one semantic equipment value")
    observed_temperature = _binding_observed_temperature(binding, target)
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        if _equipment_semantics_match(
            value, slot.source_text, observed_temperature=observed_temperature
        ):
            replacements[slot.slot_id] = slot.source_text
            continue
        source_token_lengths = tuple(len(token) for token in _TOKEN.findall(slot.source_text))
        rendered = None
        candidates = _equipment_surface_candidates(value, slot.source_text)
        for candidate in candidates:
            if tuple(len(token) for token in _TOKEN.findall(candidate)) != source_token_lengths:
                continue
            shaped = _shape_alphanumeric_like_source(slot.source_text, candidate)
            exact_pattern = slot.format_envelope.exact_surface_pattern
            if exact_pattern is not None and surface_pattern(shaped) != exact_pattern:
                continue
            if _equipment_semantics_match(value, shaped):
                rendered = shaped
                break
        if rendered is None and slot.render_policy != "opaque_identifier":
            for candidate in candidates:
                shaped = _layout_like_source(slot.source_text, candidate)
                if _equipment_semantics_match(value, shaped):
                    rendered = shaped
                    break
        if rendered is None:
            raise ValueError(f"no semantic equipment wording fits certified slot {slot.slot_id}")
        replacements[slot.slot_id] = rendered
    return BindingOutput(replacements=replacements, canonical_value=cast(JsonValue, dict(value)))


def _equipment_code(value: Mapping[str, Any]) -> str:
    size = value.get("sizeCategory")
    equipment_type = value.get("typeCategory")
    if not isinstance(size, str) or not isinstance(equipment_type, str):
        raise ValueError("semantic equipment value lacks size/type categories")
    if equipment_type == "GENERAL_PURPOSE":
        return "HC" if "HIGH_CUBE" in size else "GP"
    try:
        return _EQUIPMENT_TYPE_CODES[equipment_type]
    except KeyError as error:
        raise ValueError(f"unsupported semantic equipment type: {equipment_type}") from error


def _equipment_length(value: Mapping[str, Any]) -> str:
    size = value.get("sizeCategory")
    if not isinstance(size, str):
        raise ValueError("semantic equipment value lacks a size category")
    if size.startswith("TWENTY_"):
        return "20"
    if size.startswith("FORTY_FIVE_"):
        return "45"
    if size.startswith("FORTY_"):
        return "40"
    raise ValueError(f"unsupported semantic equipment size: {size}")


def _equipment_description(value: Mapping[str, Any], *, apostrophe: bool) -> str:
    separator = "' " if apostrophe else " "
    return _equipment_length(value) + separator + _equipment_code(value)


def _receipt_equipment_surface(value: Mapping[str, Any], source: str) -> str:
    # Receipt aliases are equipment spellings, not arbitrary identifiers. An
    # unchanged semantic pair must preserve its actual source spelling (40H,
    # 40HQ, etc.), and ISO 6346 type codes must retain the ISO representation.
    if _equipment_semantics_match(value, source):
        return source
    # A compact HC receipt asserts height only. Preserve its exact carrier
    # spelling when the new unit remains high-cube, regardless of cargo type.
    height_only = re.fullmatch(r"(20|40|45)\s*['\u2019`]?\s*HC", source, re.I)
    if (
        height_only
        and value["sizeCategory"]
        == {
            "20": "TWENTY_FOOT_HIGH_CUBE",
            "40": "FORTY_FOOT_HIGH_CUBE",
            "45": "FORTY_FIVE_FOOT_HIGH_CUBE",
        }[height_only[1]]
    ):
        return source
    from document_ocr.synthesis.container_semantics import iso_equipment_surface

    iso = iso_equipment_surface(
        cast(ContainerSizeCategory, value["sizeCategory"]),
        cast(ContainerTypeCategory, value["typeCategory"]),
        source,
    )
    if iso is not None:
        return iso
    length_only = re.fullmatch(r"(?:20|40|45)(\s*(?:['\u2019`]|FT\.?|FEET|FOOT)?)", source, re.I)
    if length_only:
        return _equipment_length(value) + length_only[1]
    compact = re.fullmatch(r"(?i)(?:20|40|45)(?P<gap>['\u2019`]?\s*)[A-Z]{1,2}", source)
    candidate = (
        _equipment_length(value) + (compact["gap"] if compact else "") + _equipment_code(value)
    )
    if _equipment_semantics_match(value, candidate):
        return candidate
    return canonical_equipment_surface(
        cast(ContainerSizeCategory, value["sizeCategory"]),
        cast(ContainerTypeCategory, value["typeCategory"]),
    )


def _is_equipment_receipt_binding(binding: SemanticBinding) -> bool:
    if binding.derivation == "equipment_receipt":
        return True
    if binding.value_kind not in {"equipment", "operational_text"}:
        return False
    return any(
        path == "documentPatch.containers"
        or re.fullmatch(r"documentPatch\.containers\[[0-9]+\]", path) is not None
        for path in binding.target_paths
    )


def _mixed_inventory_for_case(case: PreparedCase) -> MixedInventory | None:
    from . import mixed_inventory

    return mixed_inventory.compile_bindings(case.source, case.source_target, case.template.bindings)


def _render_equipment_receipt_binding(
    binding: SemanticBinding,
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    aggregate_inventory: MixedInventory | None = None,
) -> BindingOutput:
    aggregate_member = (
        aggregate_inventory is not None and binding.logical_key in aggregate_inventory.binding_keys
    )
    if not binding.target_paths and not binding.dependency_paths and not aggregate_member:
        raise ValueError(
            "equipment receipt has no target dependency; source-copy substitution is forbidden"
        )
    from .equipment_receipts import owned_inventory, validate_source_receipt

    dependency_paths = (
        ("documentPatch.containers",)
        if aggregate_member
        else (*binding.target_paths, *binding.dependency_paths)
    )
    source_containers = owned_inventory(source_target, dependency_paths)
    target_containers = owned_inventory(target, dependency_paths)
    if aggregate_member:
        assert aggregate_inventory is not None
        evidence = dict(aggregate_inventory.binding_evidence)[binding.logical_key]
        if evidence != tuple(
            (slot.source_text, slot.byte_start, slot.byte_end) for slot in binding.occurrences
        ):
            raise ValueError("aggregate receipt binding differs from its proved source")
        if (
            binding.target_paths
            or not (
                binding.dependency_paths == ("documentPatch.containers",)
                or (
                    binding.derivation == "same_as_binding"
                    and len(binding.dependency_bindings) == 1
                    and binding.dependency_bindings[0] in aggregate_inventory.binding_keys
                )
            )
            or tuple(row["containerNumber"] for row in source_containers)
            != aggregate_inventory.source_numbers
            or len(target_containers) != aggregate_inventory.container_count
            or len({row.get("containerNumber") for row in target_containers})
            != len(target_containers)
            or any(
                {"typeDescription", "sizeCategory", "typeCategory"} & row.keys()
                for row in (*source_containers, *target_containers)
            )
        ):
            raise ValueError("aggregate receipt changed its proved inventory or label visibility")
        # This public target intentionally has no type-to-ID labels. The whole
        # source multiset was proved together, and the sampler separately proves
        # its private assignment; retaining that multiset is not source fallback.
        return BindingOutput(
            replacements={slot.slot_id: slot.source_text for slot in binding.occurrences},
            canonical_value=cast(JsonValue, list(target_containers)),
        )
    # A source can print only the receipt count, or an exact unclassified type
    # description (e.g. 1 X 20'). Neither requires inventing a v5 size/type pair.
    # Keep this path strict: a conflicting HR/RQ surface does not match it.
    descriptions = {
        row.get("typeDescription") for row in source_containers if isinstance(row, Mapping)
    }
    target_descriptions = {
        row.get("typeDescription") for row in target_containers if isinstance(row, Mapping)
    }
    projected: dict[str, str] = {}
    for slot in binding.occurrences:
        match = re.fullmatch(r"(?P<count>\s*0*\d+)(?P<body>.*)", slot.source_text, re.DOTALL)
        if match is None or int(match.group("count")) != len(source_containers):
            break
        body = match.group("body")
        count_only = (
            re.fullmatch(
                r"(?i)\s*(?:X|(?:CONTAINER|CNTR|CTNR|CONTR|CONT\.?)(?:S|\(S\))?)?\s*", body
            )
            is not None
        )
        description = next(iter(descriptions)) if len(descriptions) == 1 else None
        exact_description = (
            isinstance(description, str)
            and descriptions == target_descriptions
            and _normalized_semantic(
                re.sub(r"(?i)\bCONTAINERS?\b", "", re.sub(r"^\s*[xX]\s*", "", body))
            )
            == _normalized_semantic(description)
        )
        if not count_only and not exact_description:
            break
        count = render_number_surface(
            match.group("count"), len(source_containers), len(target_containers)
        )
        projected[slot.slot_id] = count + body
    if len(projected) == len(binding.occurrences):
        for slot in binding.occurrences:
            validate_source_receipt(
                slot.source_text, source_containers, number_words=_number_to_words
            )
            validate_source_receipt(
                projected[slot.slot_id], target_containers, number_words=_number_to_words
            )
        return BindingOutput(
            replacements=projected, canonical_value=cast(JsonValue, list(target_containers))
        )
    from .equipment_receipts import project_receipt

    replacements = {
        slot.slot_id: _layout_like_source(
            slot.source_text,
            project_receipt(
                slot.source_text,
                source_containers,
                target_containers,
                format_equipment=_receipt_equipment_surface,
                number_words=_number_to_words,
            ),
        )
        for slot in binding.occurrences
    }
    return BindingOutput(
        replacements=replacements,
        canonical_value=cast(JsonValue, list(target_containers)),
    )


def _preserved_source_output(binding: SemanticBinding) -> BindingOutput:
    return BindingOutput(
        replacements={slot.slot_id: slot.source_text for slot in binding.occurrences},
        canonical_value=binding.occurrences[0].source_text.strip(),
    )


_TYPOGRAPHIC_QUOTES = str.maketrans("\u2019\u2018\u201c\u201d", "''\"\"")


def _typographic_text_equal(left: JsonValue, right: JsonValue) -> bool:
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and left.translate(_TYPOGRAPHIC_QUOTES) == right.translate(_TYPOGRAPHIC_QUOTES)
    )


def _unchanged_target_output(
    binding: SemanticBinding, target: Mapping[str, Any]
) -> BindingOutput | None:
    """Return the compiler-certified source surfaces only when every target value is unchanged."""

    snapshots = {row.target_path: row.source_value for row in binding.realization.target_values}
    if not binding.target_paths or set(snapshots) != set(binding.target_paths):
        return None
    values = tuple(_binding_target_value(target, path) for path in binding.target_paths)
    if any(
        canonical_json_bytes(value) != canonical_json_bytes(snapshots[path])
        and not _typographic_text_equal(value, snapshots[path])
        for path, value in zip(binding.target_paths, values, strict=True)
    ):
        return None
    canonical: JsonValue = cast(JsonValue, values[0] if len(values) == 1 else list(values))
    return BindingOutput(
        replacements={slot.slot_id: slot.source_text for slot in binding.occurrences},
        canonical_value=canonical,
    )


def _binding_dependencies_unchanged(
    binding: SemanticBinding,
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    constraints: Sequence[CoherenceConstraint],
) -> bool:
    paths = tuple(
        dict.fromkeys(
            (
                *binding.dependency_paths,
                *(
                    path
                    for constraint in constraints
                    if binding.logical_key in constraint.member_logical_keys
                    for path in coherence_dependency_paths(constraint)
                ),
            )
        )
    )
    return all(
        canonical_json_bytes(_binding_target_value(source_target, path))
        == canonical_json_bytes(_binding_target_value(target, path))
        for path in paths
    )


def _binding_route(
    binding: SemanticBinding, target: Mapping[str, Any]
) -> tuple[Literal["deterministic", "agent"], str]:
    if _stable_source_vocabulary(binding):
        return "deterministic", "stable non-identifying document vocabulary is preserved"
    if _explicit_unknown_placeholder(binding):
        return "deterministic", "explicit source placeholder is preserved without invented identity"
    if _direct_unbound_identifier(binding):
        return "deterministic", "unbound identifier has a complete exact surface pattern"
    if _semantic_equipment_binding_value(binding, target) is not None:
        try:
            _render_semantic_equipment_binding(binding, target)
        except ValueError as error:
            return "agent", f"semantic equipment requires residual wording: {error}"
        return "deterministic", "semantic equipment fits a certified document-native wording"
    if binding.realization.requires_agent:
        return "agent", "compiled template explicitly requires residual semantic assistance"
    if any(
        _semantic_equipment_value(target, path) is not None and not _path_exists(target, path)
        for path in binding.target_paths
    ):
        return (
            "agent",
            "v5 semantic equipment categories replace the compiled legacy description surface",
        )
    if binding.realization.mode == "generated_auxiliary":
        supported, reason = _direct_auxiliary_route(binding)
        return ("deterministic", reason) if supported else ("agent", reason)
    if (
        binding.realization.mode == "deterministic_derivation"
        and binding.derivation == "equipment_receipt"
    ):
        return "agent", "equipment receipt wording requires template-specific semantic realization"
    return "deterministic", "compiled deterministic realization is supported by the host"


def _build_initial_plan(
    case: PreparedCase, *, seed: int, country_codes: Mapping[str, str]
) -> RenderPlan:
    _require_frozen_target(case)
    dg_outputs = _dangerous_goods_outputs(case)
    if conflicts := geographic_context.source_entity_conflicts(case.template, country_codes):
        raise ValueError("source auxiliary geography requires review: " + "; ".join(conflicts))
    from .package_prose import validate_package_masses

    validate_package_masses(case.template, case.topology_reference_target, case.target)
    if {b.logical_key for b in numeric_bindings(case.template)} != set(case.numeric_auxiliary):
        raise ValueError("numeric dependency contracts do not cover unowned shipment facts")
    missing_package_contracts = {
        binding.logical_key
        for binding in numeric_bindings(case.template)
        if binding.value_kind == "package" and binding.logical_key not in case.numeric_auxiliary
    }
    if missing_package_contracts:
        raise ValueError(
            "source-only package quantities require numeric contracts: "
            + ", ".join(sorted(missing_package_contracts))
        )
    numeric_outputs = (
        render_prepared(
            numeric_bindings(case.template),
            case.numeric_auxiliary,
            source_target=case.topology_reference_target,
            target=case.target,
            source_template=case.template,
            equipment_tare_values=case.equipment_tare_values,
        )
        if case.numeric_auxiliary
        else {}
    )
    # Coherence is a pre-routing safety gate. A stale target-backed range or a compiler-marked
    # review case must fail before residual planning can initialize a provider request.
    validate_render_coherence(
        bindings=case.template.bindings,
        constraints=case.template.coherence_constraints,
        source_target=case.source_target,
        target=case.target,
        outputs=None,
    )
    range_outputs = plan_ranges(case.template, case.source_target, case.target).auxiliary_surfaces
    coherence_members_with_dependencies = {
        logical_key
        for constraint in case.template.coherence_constraints
        if coherence_dependency_paths(constraint)
        for logical_key in constraint.member_logical_keys
    }
    changed_coherence_members = {
        logical_key
        for constraint in case.template.coherence_constraints
        if any(
            _resolve_path(case.source_target, path) != _resolve_path(case.target, path)
            for path in coherence_dependency_paths(constraint)
        )
        for logical_key in constraint.member_logical_keys
    }
    projected_context = projected_auxiliary_values(case.template, case.target)
    routes: list[BindingRoute] = []
    outputs: dict[str, BindingOutput] = {}
    residual: list[SemanticBinding] = []
    stream = DeterministicStream(seed, "carrier-bound-descendant-render-v1", case.document_id)
    values = DeterministicValueFactory(
        seed=seed,
        document_id=case.document_id,
        target=case.target,
    )
    auxiliary_dispositions = disposition_by_binding(case.template.auxiliary_semantic_plan)
    auxiliary_plan = resolve_geographic_members(
        case.template.auxiliary_semantic_plan, case.template.bindings, country_codes
    )
    auxiliary_entities = entity_members_by_binding(auxiliary_plan)
    country_code_styles, entity_geography_constraints = _entity_country_code_context(
        case.template, country_codes, case.source
    )
    bindings_by_key = {candidate.logical_key: candidate for candidate in case.template.bindings}
    shape_sensitive_relationship_dependencies = {
        dependency
        for candidate in case.template.bindings
        for dependency in candidate.dependency_bindings
        if bindings_by_key[dependency].value_kind == "phone"
        if candidate.value_kind != "phone"
        or any(
            _alphanumeric(s.source_text)
            != _alphanumeric(bindings_by_key[dependency].occurrences[0].source_text)
            or s.render_policy == "opaque_identifier"
            for s in candidate.occurrences
        )
    }
    identifier_relationship_members = {
        key
        for owner, relationships in _active_identifier_relationships(case.template).items()
        for relationship in relationships
        for key in (owner, relationship.dependency_binding)
    }
    for binding in case.template.bindings:
        route, reason = _binding_route(binding, case.target)
        eager_output: BindingOutput | None = None
        auxiliary_disposition = auxiliary_dispositions.get(binding.logical_key)
        unchanged_output = _unchanged_target_output(binding, case.target)
        if unchanged_output is not None and not _binding_dependencies_unchanged(
            binding,
            source_target=case.source_target,
            target=case.target,
            constraints=case.template.coherence_constraints,
        ):
            unchanged_output = None
        measurement_output = _render_target_measurement(binding, case)
        composite_range = render_composite_range(
            binding, case.source_target, case.target, case.source
        )
        if binding.logical_key in dg_outputs:
            eager_output = dg_outputs[binding.logical_key]
            route = "deterministic"
            reason = "exact dangerous goods fact from the complete sampled regulatory tuple"
        elif composite_range is not None:
            eager_output = BindingOutput(
                replacements=composite_range,
                canonical_value=[
                    _binding_target_value(case.target, p) for p in binding.target_paths
                ],
            )
            route = "deterministic"
            reason = "source-byte-proven caption and range express typed package facts"
        elif measurement_output is not None:
            eager_output = measurement_output
            route = "deterministic"
            reason = "each measurement occurrence has source-proven units and arithmetic"
        elif (
            binding.logical_key in projected_context
            and binding.logical_key not in case.numeric_auxiliary
        ):
            eager_output = _render_text_candidate(binding, projected_context[binding.logical_key])
            route = "deterministic"
            reason = "auxiliary context derived from the completed target projection contract"
        elif binding.logical_key in range_outputs:
            surfaces = dict(range_outputs[binding.logical_key])
            eager_output = BindingOutput(
                replacements=surfaces, canonical_value=next(iter(surfaces.values()))
            )
            route = "deterministic"
            reason = "source-proven package range cardinality contract"
        elif binding.logical_key in case.numeric_auxiliary:
            prepared_numeric = case.numeric_auxiliary[binding.logical_key]
            eager_output = BindingOutput(
                replacements=numeric_outputs[binding.logical_key],
                canonical_value=prepared_numeric.value,
            )
            route = "deterministic"
            reason = "source-proven numeric scenario dependency"
        elif (
            not binding.target_paths
            and binding.derivation is None
            and (
                binding.value_kind == "date"
                or all(slot.render_policy == "date_surface" for slot in binding.occurrences)
            )
        ):
            eager_output = _fixed_date_auxiliary(binding)
            route = "deterministic"
            reason = (
                "complete source chronology is fixed scenario context, not independently sampled"
            )
        elif unchanged_output is not None:
            # The compiler already certified that these exact source slots express the target
            # snapshots. Preserve them for every realization mode when no dependency changed.
            eager_output = unchanged_output
            route = "deterministic"
            reason = "target value is unchanged from the compiler-certified source surface"
        elif (
            fixed_context(
                binding,
                case.topology_reference_target
                if binding.value_kind == "equipment"
                else case.source_target,
                case.target,
            )
            and binding.logical_key not in changed_coherence_members
        ):
            eager_output = _preserved_source_output(binding)
            route = "deterministic"
            reason = (
                "source-only location belongs to the complete unchanged route scenario"
                if binding.group_kind == "route" and binding.value_kind == "location"
                else "typed source-only equipment or independent feeder context is unchanged"
                if binding.value_kind == "equipment"
                else "complete source surface is closed, non-identifying template vocabulary"
            )
        elif (
            binding.logical_key in coherence_members_with_dependencies
            and binding.logical_key not in changed_coherence_members
        ):
            eager_output = _preserved_source_output(binding)
            route = "deterministic"
            reason = "coherence dependencies are unchanged from the certified source surface"
        elif auxiliary_disposition is not None and auxiliary_disposition.disposition in {
            "composite_number",
            "document_sequence",
            "stable_vocabulary",
        }:
            eager_output = _preserved_source_output(binding)
            route = "deterministic"
            reason = (
                "compiled auxiliary semantic plan preserves the atomic "
                f"{auxiliary_disposition.disposition} fact"
            )
        elif auxiliary_disposition is not None and (
            auxiliary_disposition.disposition == "entity_member"
        ):
            try:
                candidate_output = (
                    _render_direct_auxiliary(binding, stream)
                    if binding.logical_key
                    in (shape_sensitive_relationship_dependencies | identifier_relationship_members)
                    and not binding.target_paths
                    else _render_entity_auxiliary(
                        binding,
                        entity_member=auxiliary_entities[binding.logical_key],
                        country_code_style=country_code_styles.get(binding.logical_key),
                        geography_constraint=entity_geography_constraints[
                            auxiliary_entities[binding.logical_key][0].entity_id
                        ],
                        stream=stream,
                        values=values,
                        country_codes=country_codes,
                        prepared_auxiliary=case.auxiliary_values,
                    )
                )
                _validate_binding_format(
                    source=case.source,
                    template=case.template.byte_template,
                    output=candidate_output,
                )
            except ValueError as error:
                route = "agent"
                reason = f"canonical auxiliary entity execution is unavailable: {error}"
            else:
                eager_output = candidate_output
                route = "deterministic"
                reason = (
                    "shape-sensitive relationship dependency rendered deterministically"
                    if binding.logical_key
                    in (shape_sensitive_relationship_dependencies | identifier_relationship_members)
                    and not binding.target_paths
                    else "canonical auxiliary entity field rendered deterministically"
                )
        elif lexical_partitions.partition(binding) is not None:
            eager_output = _render_target_binding(
                binding,
                case.target,
                auxiliary_values=case.auxiliary_values,
                source=case.source,
                template=case.template,
            )
            route = "deterministic"
            reason = "source-proven cargo fragment ownership and exact target assembly"
        elif (
            binding.realization.mode == "deterministic_derivation"
            and binding.derivation == "equipment_receipt"
        ):
            try:
                candidate_output = _render_equipment_receipt_binding(
                    binding,
                    source_target=case.source_target,
                    target=case.target,
                    aggregate_inventory=_mixed_inventory_for_case(case),
                )
                _validate_binding_format(
                    source=case.source,
                    template=case.template.byte_template,
                    output=candidate_output,
                )
            except ValueError as error:
                raise ValueError(
                    "unproven equipment receipt requires contract review: "
                    f"{binding.logical_key}: {error}"
                ) from error
            else:
                eager_output = candidate_output
                route = "deterministic"
                reason = "typed equipment receipt renderer executed successfully"
        elif binding.target_paths and (
            binding.realization.requires_agent
            or _has_semantic_equipment_values(binding, case.target)
        ):
            try:
                candidate_output = _render_agent_target_binding(
                    binding,
                    source_target=case.source_target,
                    target=case.target,
                    source=case.source,
                    country_codes=country_codes,
                )
                _validate_binding_format(
                    source=case.source,
                    template=case.template.byte_template,
                    output=candidate_output,
                )
            except ValueError as error:
                route = "agent"
                reason = f"typed target execution is unavailable: {error}"
            else:
                eager_output = candidate_output
                route = "deterministic"
                reason = "typed target-to-surface renderer executed successfully"
        elif binding.logical_key not in changed_coherence_members and (
            (
                binding.realization.mode == "generated_auxiliary"
                and not _stable_source_vocabulary(binding)
            )
            or (
                route == "agent" and binding.realization.requires_agent and not binding.target_paths
            )
        ):
            try:
                candidate_output = _render_generated_auxiliary(
                    binding,
                    stream=stream,
                    values=values,
                )
                _validate_binding_format(
                    source=case.source,
                    template=case.template.byte_template,
                    output=candidate_output,
                )
            except ValueError as error:
                route = "agent"
                reason = f"typed deterministic execution is unavailable: {error}"
            else:
                eager_output = candidate_output
                route = "deterministic"
                reason = "typed deterministic auxiliary generator executed successfully"
        routes.append(
            BindingRoute.model_validate(
                {
                    "binding_id": binding.binding_id,
                    "logical_key": binding.logical_key,
                    "compiled_mode": binding.realization.mode,
                    "runtime_route": route,
                    "route_reason": reason,
                    "slot_ids": tuple(slot.slot_id for slot in binding.occurrences),
                }
            )
        )
        if route == "agent":
            if (
                binding.value_kind == "equipment"
                and not binding.target_paths
                and not binding.derivation
                and any(s.render_policy == "opaque_identifier" for s in binding.occurrences)
            ):
                raise ValueError(
                    "unproven opaque equipment requires contract review: " + binding.logical_key
                )
            residual.append(binding)
            continue
        if eager_output is not None:
            outputs[binding.logical_key] = eager_output
        elif (
            _stable_source_vocabulary(binding)
            or _explicit_unknown_placeholder(binding)
            or binding.realization.mode == "static"
        ):
            outputs[binding.logical_key] = _preserved_source_output(binding)
        elif _direct_unbound_identifier(binding):
            outputs[binding.logical_key] = _render_direct_auxiliary(binding, stream)
        elif _semantic_equipment_binding_value(binding, case.target) is not None:
            outputs[binding.logical_key] = _render_semantic_equipment_binding(binding, case.target)
        elif binding.realization.mode in {
            "single_surface",
            "repeated_surface",
            "segmented_surface",
            "token_projected_surface",
            "normalized_projected_surface",
        }:
            outputs[binding.logical_key] = _render_target_binding(
                binding,
                case.target,
                auxiliary_values=case.auxiliary_values,
                source=case.source,
                template=case.template,
            )
        elif binding.realization.mode == "generated_auxiliary":
            outputs[binding.logical_key] = _render_generated_auxiliary(
                binding,
                stream=stream,
                values=values,
            )
        elif binding.realization.mode == "deterministic_derivation":
            continue
        else:
            raise ValueError(f"unsupported realization mode: {binding.realization.mode}")
    entity_groups, country_prefixes = entity_identifiers.contracts(
        case.template, case.source_target, case.target, country_codes
    )
    _solve_identifier_relationships(
        template=case.template,
        outputs=outputs,
        stream=stream,
        equivalence_groups=entity_groups,
        fixed_prefixes=country_prefixes,
    )
    _plan_derivations(
        case=case,
        outputs=outputs,
        routes=routes,
        residual=residual,
        country_codes=country_codes,
    )
    for output in outputs.values():
        _validate_binding_format(
            source=case.source,
            template=case.template.byte_template,
            output=output,
        )
    phone_countries = {}
    for entity in auxiliary_plan.entities:
        members = [m for m in entity.members if m.field == "phone"]
        if entity.relationship != "independent" or not members:
            continue
        constraint = entity_geography_constraints[entity.entity_id]
        geo = values.geography_for_identity(
            entity.entity_id,
            calling_code_width=constraint.calling_code_width,
            country_name_width=constraint.country_name_width,
            postal_code_pattern=constraint.postal_code_pattern,
            fixed_country_code=constraint.fixed_country_code,
        )
        phone_countries.update({m.logical_key: geo.country_code for m in members})
    return RenderPlan(
        routes=tuple(routes),
        deterministic_outputs=outputs,
        residual_bindings=tuple(residual),
        independent_phone_countries=phone_countries,
    )


def _validate_binding_format(
    *, source: bytes, template: CompiledRawTextTemplate, output: BindingOutput
) -> None:
    if len(source) != template.source_size_bytes or sha256_bytes(source) != template.source_sha256:
        raise ValueError("template source payload differs from the pinned source")
    validate_slot_replacements(template=template, replacements=output.replacements)


def _party_exposes_auxiliary_field(party: Mapping[str, Any], field: str) -> bool:
    direct = {
        "name": "name",
        "address": "address",
        "city": "city",
        "country": "country",
    }
    if field in direct:
        value = party.get(direct[field])
        return isinstance(value, str) and bool(value.strip())
    if field == "country_code":
        value = party.get("country")
        return isinstance(value, str) and bool(value.strip())
    contacts = party.get("contactDetails")
    if isinstance(contacts, Mapping):
        if field == "contact_name":
            return bool(contacts.get("contactName"))
        if field in {"phone", "email"}:
            return bool(contacts.get("phoneNumbers" if field == "phone" else "emailAddresses"))
    return False


def _validate_unrepresented_party_facets(
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    template: CertifiedSemanticTemplate,
    auxiliary_values: Mapping[str, str] | None = None,
    pending_auxiliary_keys: frozenset[str] = frozenset(),
) -> None:
    """Missing identity context is unresolved work, never permission to undo synthesis."""
    contextual_fields = {
        "name",
        "address",
        "city",
        "region",
        "postal_code",
        "country",
        "country_code",
        "phone_extension",
        "other",
    }
    known_auxiliary = {
        member.logical_key
        for entity in template.auxiliary_semantic_plan.entities
        for member in entity.members
    }
    from .route_derivations import DERIVATIONS as route_derivations

    known_auxiliary.update(
        b.logical_key for b in template.bindings if b.derivation in route_derivations
    )
    if pending_auxiliary_keys - known_auxiliary:
        raise ValueError("pending linguistic context contains unknown auxiliary bindings")
    projected_context = projected_auxiliary_values(template, target) if auxiliary_values else {}
    for entity in template.auxiliary_semantic_plan.entities:
        party_path = entity.target_party_path
        if party_path is not None and auxiliary_values:
            party = _resolve_path(target, party_path)
            for member in entity.members:
                if (
                    member.logical_key in auxiliary_values
                    and _party_exposes_auxiliary_field(party, member.field)
                    and auxiliary_values[member.logical_key]
                    != projected_context.get(member.logical_key)
                ):
                    raise ValueError(
                        f"auxiliary context cannot override target field: {member.logical_key}"
                    )
        if party_path is None:
            continue
        party = _resolve_path(target, party_path)
        if not isinstance(party, Mapping):
            raise ValueError(f"auxiliary target party is not an object: {party_path}")
        missing = tuple(
            member
            for member in entity.members
            if member.field in contextual_fields
            and not _party_exposes_auxiliary_field(party, member.field)
            and (auxiliary_values is None or member.logical_key not in auxiliary_values)
            and member.logical_key not in pending_auxiliary_keys
        )
        if missing and party != _resolve_path(source_target, party_path):
            keys = ", ".join(sorted(member.logical_key for member in missing))
            raise ValueError(
                f"synthetic party has unresolved auxiliary identity context: {party_path}: "
                f"{keys}; source-party restoration is forbidden"
            )


def _validate_target_compatibility(
    *,
    source: bytes,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    template: CertifiedSemanticTemplate,
    auxiliary_values: Mapping[str, str] | None = None,
    pending_auxiliary_keys: frozenset[str] = frozenset(),
) -> None:
    """Check the frozen target without replacing any proposed fact by source data."""
    _validate_unrepresented_party_facets(
        source_target=source_target,
        target=target,
        template=template,
        auxiliary_values=auxiliary_values,
        pending_auxiliary_keys=pending_auxiliary_keys,
    )
    deterministic_modes = {
        "single_surface",
        "repeated_surface",
        "segmented_surface",
        "token_projected_surface",
        "normalized_projected_surface",
    }
    for binding in template.bindings:
        partition = lexical_partitions.partition(binding)
        if binding.realization.mode not in deterministic_modes and partition is None:
            continue
        if auxiliary_values is None and partition is not None:
            # Preliminary structured proposals have not generated language yet.
            # The completed target must pass with its explicit fragment receipt.
            continue
        try:
            if _has_semantic_equipment_values(binding, target):
                output = _render_agent_target_binding(
                    binding, source_target=source_target, target=target
                )
            else:
                output = _render_target_binding(
                    binding,
                    target,
                    auxiliary_values=auxiliary_values,
                    source=source,
                    template=template,
                )
            _validate_binding_format(source=source, template=template.byte_template, output=output)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"synthetic target is incompatible with binding {binding.logical_key} "
                f"at {binding.target_paths}: {error}; target not modified"
            ) from error


def _case_files(template_root: Path, document_id: str) -> tuple[bytes, dict[str, Any], bytes]:
    root = (template_root / "cases" / document_id).resolve(strict=True)
    if template_root not in root.parents or root.is_symlink() or not root.is_dir():
        raise ValueError(f"template case is not a regular child: {document_id}")
    source_path = root / "source.txt"
    label_path = root / "source-label.json"
    template_path = root / "template.json"
    for path in (source_path, label_path, template_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"template case input is not a regular file: {path}")
    source = read_regular_file_bytes(source_path)
    source_target = json.loads(read_regular_file_bytes(label_path))
    if not isinstance(source_target, dict):
        raise ValueError(f"source label is not an object: {document_id}")
    return source, source_target, read_regular_file_bytes(template_path)


def _load_cases(*, project_root: Path, config: DescendantConfig) -> tuple[PreparedCase, ...]:
    template_root = _validate_committed_run(
        project_root, config.inputs.template_run, workers=config.workflow.max_concurrent_requests
    )
    customs_registry = None
    customs_cache: dict[str, CustomsPresentation] = {}
    if config.inputs.customs_program_registry is not None:
        customs_pin = config.inputs.customs_program_registry
        customs_path = resolve_input(project_root, customs_pin.path)
        if sha256_file(customs_path) != customs_pin.sha256:
            raise ValueError("customs program registry hash differs")
        customs_registry = CustomsProgramRegistry.model_validate_json(customs_path.read_bytes())
    country_path = resolve_input(project_root, config.inputs.iso3166_snapshot.path)
    if sha256_file(country_path) != config.inputs.iso3166_snapshot.sha256:
        raise ValueError("ISO-3166 snapshot hash differs")

    # The compiler selection can contain fail-closed source-integrity or semantic-review cases.
    # Only its committed certified catalog is renderable; selecting from the original manifest
    # would turn an intentional review outcome into a missing-file failure downstream.
    catalog_path = template_root / "catalog.jsonl"
    complete_catalog = _read_jsonl(catalog_path, records=None)
    if any(row.get("certified") is not True for row in complete_catalog):
        raise ValueError("template catalog contains a non-certified row")
    complete_document_ids = tuple(row.get("documentId") for row in complete_catalog)
    if any(not isinstance(value, str) for value in complete_document_ids):
        raise ValueError("template catalog contains an invalid document ID")
    if len(set(complete_document_ids)) != len(complete_document_ids):
        raise ValueError("template catalog document IDs are not unique")

    sample_plan = config.inputs.sample_plan
    sample_plan_run = config.inputs.sample_plan_run
    target_run = config.inputs.synthetic_target_run
    synthetic_targets = config.inputs.synthetic_targets
    if target_run is None or synthetic_targets is None:
        raise ValueError(
            "complete synthetic targets are required; source-copy substitution is forbidden"
        )
    target_root = _validate_committed_run(
        project_root, target_run, workers=config.workflow.max_concurrent_requests
    )
    target_path = resolve_input(project_root, synthetic_targets.path)
    if target_root not in target_path.parents:
        raise ValueError("synthetic-target file is outside its pinned committed run")
    if sha256_file(target_path) != synthetic_targets.sha256:
        raise ValueError("synthetic-target file hash differs")
    targets: dict[str, tuple[str, dict[str, Any]]] = {}
    auxiliary_by_sample: dict[str, dict[str, str]] = {}
    numeric_by_sample: dict[str, dict[str, PreparedNumeric]] = {}
    tares_by_sample: dict[str, dict[str, Decimal]] = {}
    dg_by_sample: dict[str, tuple[DangerousGoodsFact, ...]] = {}
    for row in _read_jsonl(target_path, records=synthetic_targets.records):
        sample_id = row.get("sampleId") if sample_plan is not None else row.get("baseDocumentId")
        source_id = row.get("sourceDocumentId") if sample_plan is not None else sample_id
        target = row.get("target")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(source_id, str):
            raise ValueError("synthetic-target row lacks its sample/source identity")
        if not isinstance(target, dict):
            raise ValueError(f"synthetic-target row lacks a complete target: {sample_id}")
        if sample_id in targets:
            raise ValueError(f"duplicate synthetic target for {sample_id}")
        if sha256_bytes(canonical_json_bytes(target)) != row.get("targetSha256"):
            raise ValueError(f"synthetic-target row hash differs: {sample_id}")
        targets[sample_id] = (source_id, target)
        auxiliary = row.get("auxiliaryValues", {})
        if not isinstance(auxiliary, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or not v.strip()
            for k, v in auxiliary.items()
        ):
            raise ValueError(f"invalid auxiliary semantic context: {sample_id}")
        auxiliary_by_sample[sample_id] = auxiliary
        numeric_by_sample[sample_id] = {
            key: PreparedNumeric.model_validate_json(canonical_json_bytes(value), strict=True)
            for key, value in row.get("numericAuxiliary", {}).items()
        }
        tares_by_sample[sample_id] = _published_equipment_tares(row, numeric_by_sample[sample_id])
        dg_by_sample[sample_id] = tuple(
            DangerousGoodsFact.model_validate_json(canonical_json_bytes(value), strict=True)
            for value in row.get("dangerousGoodsFacts", [])
        )

    selected: list[tuple[str, str]] = []
    if sample_plan is None:
        if len(complete_catalog) < config.workflow.documents:
            raise ValueError("template catalog has fewer certified rows than the requested cohort")
        selected = [
            (document_id, document_id)
            for document_id in cast(
                tuple[str, ...], complete_document_ids[: config.workflow.documents]
            )
        ]
    else:
        assert sample_plan_run is not None
        plan_root = _validate_committed_run(
            project_root, sample_plan_run, workers=config.workflow.max_concurrent_requests
        )
        plan_path = resolve_input(project_root, sample_plan.path)
        if plan_root not in plan_path.parents:
            raise ValueError("sample-plan file is outside its pinned committed run")
        if sha256_file(plan_path) != sample_plan.sha256:
            raise ValueError("sample-plan file hash differs")
        plan_rows = _read_jsonl(plan_path, records=sample_plan.records)
        if len(plan_rows) != config.workflow.documents:
            raise ValueError("sample-plan row count differs from workflow.documents")

        plan_by_sample: dict[str, tuple[str, int]] = {}
        for row in plan_rows:
            sample_id = row.get("sampleId")
            source_document_id = row.get("sourceDocumentId")
            variant_index = row.get("variantIndex")
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or not isinstance(source_document_id, str)
                or not source_document_id
                or not isinstance(variant_index, int)
                or isinstance(variant_index, bool)
                or variant_index < 0
            ):
                raise ValueError("sample-plan row has an invalid sample/source/variant identity")
            if (
                row.get("targetTask") != "bill_of_lading_relation_explicit_v5"
                or row.get("targetSchemaVersion") != config.workflow.target_schema_version
            ):
                raise ValueError(f"sample-plan target contract differs: {sample_id}")
            if row.get("targetGeneration") != config.workflow.target_generation:
                raise ValueError(f"sample-plan target generation differs: {sample_id}")
            if sample_id in plan_by_sample:
                raise ValueError(f"sample plan repeats sample identity: {sample_id}")
            plan_by_sample[sample_id] = (source_document_id, variant_index)
        source_variants = tuple(plan_by_sample.values())
        if len(source_variants) != len(set(source_variants)):
            raise ValueError("sample plan repeats a source-document/variant identity")
        selected = [
            (cast(str, row["sampleId"]), cast(str, row["sourceDocumentId"])) for row in plan_rows
        ]
        if set(targets) != {sample_id for sample_id, _ in selected}:
            raise ValueError("complete synthetic targets do not cover the sample plan exactly")

    planned_inputs: list[tuple[str, str, dict[str, Any], str]] = []
    for sample_id, source_id in selected:
        if sample_id not in targets:
            raise ValueError(
                f"missing completed synthetic target: {sample_id}; "
                "source-copy substitution is forbidden"
            )
        actual_source_id, target = targets[sample_id]
        if actual_source_id != source_id:
            raise ValueError(
                f"synthetic-target source identity differs from sample plan: {sample_id}"
            )
        planned_inputs.append((sample_id, source_id, target, "complete_synthetic_target"))

    catalog_document_ids = frozenset(cast(tuple[str, ...], complete_document_ids))
    unknown_sources = sorted(
        {source_document_id for _sample_id, source_document_id, _target, _origin in planned_inputs}
        - catalog_document_ids
    )
    if unknown_sources:
        raise ValueError(f"sample plan references unknown template documents: {unknown_sources}")

    prepared: list[PreparedCase] = []
    case_cache: dict[str, tuple[bytes, dict[str, Any], CertifiedSemanticTemplate]] = {}
    latest_target_cache: dict[str, dict[str, Any]] = {}
    for sample_id, source_document_id, planned_target, target_origin in planned_inputs:
        cached = case_cache.get(source_document_id)
        if cached is None:
            source, source_target, template_bytes = _case_files(template_root, source_document_id)
            template = CertifiedSemanticTemplate.model_validate_json(template_bytes, strict=True)
            if template.document_id != source_document_id or template.source_sha256 != sha256_bytes(
                source
            ):
                raise ValueError(f"template/source identity differs: {source_document_id}")
            _validate_canonical_target(source_target)
            validate_compiled_location_payment_grounding(
                source_target=source_target, template=template
            )
            validate_compiled_party_contract(
                raw=source,
                source_target=source_target,
                bindings=template.bindings,
                entities=template.auxiliary_semantic_plan.entities,
            )
            template = effective_realization_template(template)
            cached = (source, source_target, template)
            case_cache[source_document_id] = cached
        source, source_target, template = cached
        customs_presentation = None
        if customs_registry is not None:
            if source_document_id not in customs_cache:
                customs_cache[source_document_id] = compile_neutral_customs(
                    source=source,
                    template=template.byte_template,
                    registry=customs_registry,
                    static_slot_ids=frozenset(
                        slot.slot_id
                        for binding in template.bindings
                        if binding.realization.mode == "static" and binding.group_kind != "carrier"
                        for slot in binding.occurrences
                    ),
                )
            customs_presentation = customs_cache[source_document_id]
        comparison_source_target = latest_target_cache.get(source_document_id)
        if comparison_source_target is None:
            comparison_source_target = latest_target_from_source(source_target)
            latest_target_cache[source_document_id] = comparison_source_target
        target = deepcopy(planned_target)
        auxiliary_values = auxiliary_by_sample[sample_id]
        numeric_auxiliary = numeric_by_sample[sample_id]
        equipment_tare_values = tares_by_sample[sample_id]
        dangerous_goods_facts = dg_by_sample[sample_id]
        known_auxiliary = {
            member.logical_key
            for entity in template.auxiliary_semantic_plan.entities
            for member in entity.members
        }
        projection_values = projected_auxiliary_values(template, target)
        known_auxiliary.update(projection_values)
        known_auxiliary.update(lexical_partitions.known_keys(template))
        from .route_derivations import DERIVATIONS as route_derivations

        route_keys = {b.logical_key for b in template.bindings if b.derivation in route_derivations}
        known_auxiliary.update(route_keys)
        from .transport_derivations import DERIVATIONS as transport_derivations

        transport_keys = {
            b.logical_key for b in template.bindings if b.derivation in transport_derivations
        }
        known_auxiliary.update(transport_keys)
        from .cargo_identity_derivations import DERIVATIONS as cargo_identity_derivations

        commodity_keys = {
            b.logical_key for b in template.bindings if b.derivation in cargo_identity_derivations
        }
        known_auxiliary.update(commodity_keys)
        if commodity_keys - auxiliary_values.keys():
            raise ValueError(f"missing pinned commodity identity values: {sample_id}")
        if transport_keys - auxiliary_values.keys():
            raise ValueError(f"missing pinned independent vessel values: {sample_id}")
        if route_keys - auxiliary_values.keys():
            raise ValueError(f"missing pinned source-only route values: {sample_id}")
        if set(auxiliary_values) - known_auxiliary:
            raise ValueError(f"unknown auxiliary semantic keys: {sample_id}")
        if any(auxiliary_values.get(key) != value for key, value in projection_values.items()):
            raise ValueError(f"target-derived auxiliary context receipt differs: {sample_id}")
        if target.get("schemaVersion") != config.workflow.target_schema_version:
            raise ValueError(f"prepared target is not latest-schema: {sample_id}")
        _validate_canonical_target(target)
        validate_seal_realization(target=target, bindings=template.bindings)
        proposed_sha = sha256_bytes(canonical_json_bytes(target))
        _require_source_carrier(source_target=source_target, target=target)
        require_complete_variation(comparison_source_target, target, bindings=template.bindings)
        _validate_target_compatibility(
            source=source,
            source_target=source_target,
            target=target,
            template=template,
            auxiliary_values=auxiliary_values,
        )
        validate_render_coherence(
            bindings=template.bindings,
            constraints=template.coherence_constraints,
            source_target=source_target,
            target=target,
            outputs=None,
        )
        mismatches = printed_topology_mismatches(comparison_source_target, target)
        if mismatches:
            details = ", ".join(f"{row.path}:{row.value_kind}" for row in mismatches)
            raise ValueError(f"prepared target changes printed topology for {sample_id}: {details}")
        source_leaves = _flatten_leaves(comparison_source_target)
        target_leaves = _flatten_leaves(target)
        changed = {
            path
            for path in source_leaves.keys() | target_leaves.keys()
            if source_leaves.get(path) != target_leaves.get(path)
        }
        carrier_prefix = "documentPatch.parties.carrier"
        carrier_source = {
            path: value for path, value in source_leaves.items() if path.startswith(carrier_prefix)
        }
        carrier_target = {
            path: value for path, value in target_leaves.items() if path.startswith(carrier_prefix)
        }
        if not carrier_source or carrier_source != carrier_target:
            raise ValueError(f"prepared target does not preserve the complete carrier: {sample_id}")
        target_sha = sha256_bytes(canonical_json_bytes(target))
        synthetic_id = (
            "syn_tpl_"
            + sha256_bytes(
                canonical_json_bytes(
                    [
                        sample_id,
                        source_document_id,
                        template.source_sha256,
                        target_sha,
                        config.workflow.controlled_target_seed,
                    ]
                )
            )[:40]
        )
        receipt = PreparedTargetReceipt.model_validate(
            {
                "schema_version": 1,
                "document_id": sample_id,
                "source_document_id": source_document_id,
                "target_origin": target_origin,
                "source_schema_version": source_target["schemaVersion"],
                "target_schema_version": target["schemaVersion"],
                "fixed_carrier_name": template.carrier.canonical_name,
                "source_target_sha256": sha256_bytes(canonical_json_bytes(source_target)),
                "proposed_target_sha256": proposed_sha,
                "prepared_target_sha256": target_sha,
                "auxiliary_values_sha256": sha256_bytes(canonical_json_bytes(auxiliary_values)),
                "customs_presentation_sha256": (
                    customs_presentation.sha256
                    if customs_presentation is not None
                    else sha256_bytes(canonical_json_bytes([]))
                ),
                "numeric_auxiliary_sha256": sha256_bytes(
                    canonical_json_bytes(
                        {k: v.model_dump(mode="json") for k, v in numeric_auxiliary.items()}
                    )
                ),
                "dangerous_goods_facts_sha256": sha256_bytes(
                    canonical_json_bytes([v.model_dump(mode="json") for v in dangerous_goods_facts])
                ),
                "equipment_tare_values_sha256": sha256_bytes(
                    canonical_json_bytes(_equipment_tare_payload(equipment_tare_values))
                ),
                "synthetic_document_id": synthetic_id,
                "topology_mismatch_count": 0,
                "target_leaf_count": len(target_leaves),
                "changed_target_leaf_count": len(changed),
                "carrier_leaf_count": len(carrier_source),
                "carrier_changed_leaf_count": 0,
                "compatibility_adaptations": (),
                "training_eligible": False,
            }
        )
        prepared.append(
            PreparedCase(
                document_id=sample_id,
                source_document_id=source_document_id,
                source=source,
                source_target=source_target,
                topology_reference_target=comparison_source_target,
                target=target,
                template=template,
                target_receipt=receipt,
                auxiliary_values=auxiliary_values,
                numeric_auxiliary=numeric_auxiliary,
                equipment_tare_values=equipment_tare_values,
                customs_presentation=customs_presentation,
                dangerous_goods_facts=dangerous_goods_facts,
            )
        )
    return tuple(prepared)


def _provider_settings(provider: ProviderConfig) -> ModelSettings:
    if isinstance(provider, OpenRouterProviderConfig):
        openrouter_settings: OpenRouterModelSettings = {
            "max_tokens": provider.max_output_tokens,
            "timeout": provider.request_timeout_seconds,
            "openrouter_reasoning": {"effort": provider.reasoning_effort},
            "openrouter_provider": cast(
                Any,
                {
                    "only": list(provider.provider_only),
                    "order": list(provider.provider_only),
                    "require_parameters": provider.require_parameters,
                    "data_collection": provider.data_collection,
                    "allow_fallbacks": provider.allow_fallbacks,
                    "max_price": {
                        "prompt": float(provider.max_prompt_price_usd_per_million),
                        "completion": float(provider.max_completion_price_usd_per_million),
                    },
                },
            ),
        }
        return cast(ModelSettings, openrouter_settings)
    openai_settings: OpenAIResponsesModelSettings = {
        "max_tokens": provider.max_output_tokens,
        "timeout": provider.request_timeout_seconds,
        "openai_reasoning_effort": provider.reasoning_effort,
        "openai_reasoning_mode": "standard",
        "openai_reasoning_context": "current_turn",
        "openai_store": provider.store_responses,
    }
    return cast(ModelSettings, openai_settings)


def _provider_model(
    *, project_root: Path, environment_file: str, provider: ProviderConfig
) -> Model:
    key = load_provider_key(project_root, environment_file, provider.api_key_env)
    client = AsyncOpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1"
        if isinstance(provider, OpenRouterProviderConfig)
        else None,
        max_retries=provider.transport_max_retries,
        timeout=provider.request_timeout_seconds,
        default_headers={"X-Title": "DocumentParsing"}
        if isinstance(provider, OpenRouterProviderConfig)
        else None,
    )
    if isinstance(provider, OpenRouterProviderConfig):
        return OpenRouterModel(
            provider.model,
            provider=OpenRouterProvider(openai_client=client),
            profile=(
                ModelProfile(supports_json_schema_output=True)
                if provider.native_structured_output_profile == "provider_verified"
                else None
            ),
        )
    return OpenAIResponsesModel(provider.model, provider=OpenAIProvider(openai_client=client))


def _empty_usage() -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt.model_validate(
        {
            "requests": 0,
            "providerResponseIds": (),
            "finishReasons": (),
            "inputTokens": 0,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "outputTokens": 0,
            "reasoningTokens": 0,
            "visibleOutputTokens": 0,
            "providerTokenAccountingAnomaly": False,
            "estimatedCostUsd": Decimal(0),
            "providerReportedCostUsd": None,
            "downstreamProviders": (),
        }
    )


def _stable_boilerplate(binding: SemanticBinding) -> bool:
    return binding.value_kind in {
        "operational_text",
        "legal_text",
        "dangerous_goods",
        "package",
        "other_text",
    } and binding.group_kind not in {"party", "customs"}


def _residual_payload(*, case: PreparedCase, plan: RenderPlan) -> dict[str, Any]:
    known = plan.deterministic_outputs
    active_relationships = _active_identifier_relationships(case.template)
    rows = []
    for binding in plan.residual_bindings:
        rows.append(
            {
                "bindingId": binding.binding_id,
                "logicalKey": binding.logical_key,
                "compiledMode": binding.realization.mode,
                "valueKind": binding.value_kind,
                "groupKind": binding.group_kind,
                "groupKey": binding.group_key,
                "targetPaths": binding.target_paths,
                "targetValues": tuple(
                    {"path": path, "value": _binding_target_value(case.target, path)}
                    for path in binding.target_paths
                ),
                "derivation": binding.derivation,
                "dependencyPaths": binding.dependency_paths,
                "dependencyBindings": tuple(
                    {
                        "logicalKey": key,
                        "knownValue": known[key].canonical_value if key in known else None,
                    }
                    for key in binding.dependency_bindings
                ),
                "sourceRelationships": tuple(
                    {
                        **row.model_dump(mode="json"),
                        "dependencyKnownValue": (
                            known[row.dependency_binding].canonical_value
                            if row.dependency_binding in known
                            else None
                        ),
                    }
                    for row in active_relationships[binding.logical_key]
                ),
                "coherenceConstraints": tuple(
                    {
                        "contract": constraint.model_dump(mode="json"),
                        "sourceDependencyValues": tuple(
                            {
                                "path": path,
                                "value": _binding_target_value(case.source_target, path),
                            }
                            for path in coherence_dependency_paths(constraint)
                        ),
                        "targetDependencyValues": tuple(
                            {
                                "path": path,
                                "value": _binding_target_value(case.target, path),
                            }
                            for path in coherence_dependency_paths(constraint)
                        ),
                    }
                    for constraint in case.template.coherence_constraints
                    if binding.logical_key in constraint.member_logical_keys
                ),
                "stableNonIdentifyingBoilerplate": _stable_boilerplate(binding),
                "bindingRationale": binding.rationale,
                "slots": tuple(
                    {
                        "slotId": slot.slot_id,
                        "sourceText": slot.source_text,
                        "renderPolicy": slot.render_policy,
                        "caseProfile": slot.format_envelope.case_profile,
                        "lineCount": len(slot.format_envelope.newline_sequence) + 1,
                        "sourceLineTokenCounts": tuple(
                            len(_TOKEN.findall(line)) for line in slot.source_text.splitlines()
                        ),
                        "alphanumericCount": len(_alphanumeric(slot.source_text)),
                        "exactOpaquePattern": slot.format_envelope.exact_surface_pattern,
                    }
                    for slot in binding.occurrences
                ),
            }
        )
    return {
        "schemaVersion": 1,
        "documentId": case.document_id,
        "fixedCarrier": case.template.carrier.model_dump(mode="json"),
        "syntheticTarget": case.target,
        "auxiliaryValues": dict(case.auxiliary_values),
        "numericAuxiliary": {
            k: v.model_dump(mode="json") for k, v in case.numeric_auxiliary.items()
        },
        "residualBindings": tuple(rows),
    }


def _residual_output_type(bindings: Sequence[SemanticBinding]) -> type[BaseModel]:
    slots = tuple(slot for binding in bindings for slot in binding.occurrences)
    # A primary value kind does not narrow a composite binding's contract.
    # Address/city/country/phone surfaces must express every owned target path;
    # constraining those to phone syntax makes correct rendering impossible.
    phone_slots = {
        s.slot_id
        for b in bindings
        if b.value_kind == "phone"
        and all(
            re.fullmatch(r"documentPatch\.parties\..+\.contactDetails\.phoneNumbers\[\d+\]", path)
            for path in (*b.target_paths, *b.dependency_paths)
        )
        for s in b.occurrences
    }
    if not slots:
        raise ValueError("cannot build a residual schema without slots")
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("residual output schema contains duplicate slot IDs")

    def text_pattern(source: str) -> str:
        # A semantic value is not another serialized JSON string. Literal slash-n
        # appeared when models tried to encode the host-owned line layout.
        slash = r"\\" if "\\" not in source else ""
        forbidden = r"\x00-\x1f\x7f-\x9f" + slash
        return rf"^[^\s{forbidden}](?:[^{forbidden}]*[^\s{forbidden}])?$"

    fields: dict[str, Any] = {
        slot.slot_id: (
            str,
            Field(
                min_length=1,
                pattern=(
                    r"^\+[0-9][0-9 ().-]*[0-9]$"
                    if slot.slot_id in phone_slots
                    else text_pattern(slot.source_text)
                ),
                description=(
                    f"Semantic replacement content for {slot.semantic_role}; "
                    f"source render policy is {slot.render_policy}."
                    " Use spaces between semantic words. The host owns all line breaks; "
                    "do not encode or escape newlines, tabs, or control separators."
                    + (
                        " Return one valid international phone starting with +, consistent with "
                        "the owner's country; no extensions or alternative numbers."
                        if slot.slot_id in phone_slots
                        else ""
                    )
                ),
            ),
        )
        for slot in slots
    }
    name = (
        "DescendantResidual_"
        + sha256_bytes(canonical_json_bytes(tuple(field for field in fields)))[:16]
    )
    return cast(
        type[BaseModel],
        create_model(name, __config__=_STRICT_DYNAMIC, __module__=__name__, **fields),
    )


def _not_required_stage(*, document_id: str, system_prompt_sha256: str) -> ResidualStageReceipt:
    return ResidualStageReceipt.model_validate(
        {
            "schema_version": 1,
            "document_id": document_id,
            "status": "not_required",
            "started_at": None,
            "completed_at": None,
            "duration_seconds": 0.0,
            "system_prompt_sha256": system_prompt_sha256,
            "user_prompt_sha256": None,
            "output_schema_sha256": None,
            "output": None,
            "error_type": None,
            "error_message": None,
            "messages": [],
            "usage": _empty_usage().model_dump(mode="json"),
        }
    )


async def _call_residual_agent(
    *,
    model: Model,
    provider: ProviderConfig,
    system_prompt: str,
    case: PreparedCase,
    plan: RenderPlan,
    limiter: asyncio.Semaphore,
) -> tuple[Mapping[str, str] | None, ResidualStageReceipt]:
    if not plan.residual_bindings:
        return {}, _not_required_stage(
            document_id=case.document_id,
            system_prompt_sha256=sha256_bytes(system_prompt.encode()),
        )
    output_type = _residual_output_type(plan.residual_bindings)
    payload = _residual_payload(case=case, plan=plan)
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    schema = output_type.model_json_schema(mode="validation")
    agent = Agent[Any, Any](
        model,
        output_type=NativeOutput(
            output_type,
            name="carrier_bound_descendant_residuals",
            description="Return every residual slot value and no other content.",
            strict=True,
        ),
        system_prompt=system_prompt,
        model_settings=_provider_settings(provider),
        retries=0,
        name="carrier-bound-descendant-renderer",
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    output: Mapping[str, str] | None = None
    error_type: str | None = None
    error_message: str | None = None
    captured: list[Any]
    async with limiter:
        with capture_run_messages() as messages:
            try:
                result = await agent.run(
                    user_prompt,
                    usage_limits=UsageLimits(
                        request_limit=1,
                        output_tokens_limit=provider.max_output_tokens,
                    ),
                )
                output = cast(Mapping[str, str], result.output.model_dump(mode="python"))
            except Exception as error:  # exact error is retained in the immutable receipt
                error_type = type(error).__name__
                error_message = str(error)
            captured = list(messages)
    completed_at = datetime.now(UTC)
    responses = tuple(row for row in captured if isinstance(row, ModelResponse))
    usage = (
        usage_receipt(
            responses,
            cast(Any, provider.pricing),
            require_provider_cost=isinstance(provider, OpenRouterProviderConfig),
        )
        if responses
        else _empty_usage()
    )
    receipt = ResidualStageReceipt.model_validate(
        {
            "schema_version": 1,
            "document_id": case.document_id,
            "status": "success" if output is not None else "provider_error",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": time.perf_counter() - started,
            "system_prompt_sha256": sha256_bytes(system_prompt.encode()),
            "user_prompt_sha256": sha256_bytes(user_prompt.encode()),
            "output_schema_sha256": sha256_bytes(canonical_json_bytes(schema)),
            "output": dict(output) if output is not None else None,
            "error_type": error_type,
            "error_message": error_message,
            "messages": model_messages(cast(Sequence[Any], captured)),
            "usage": usage.model_dump(mode="json"),
        }
    )
    return output, receipt


def _postprocess_residual_outputs(
    *,
    case: PreparedCase,
    plan: RenderPlan,
    raw_output: Mapping[str, str],
) -> dict[str, BindingOutput]:
    expected = {slot.slot_id for binding in plan.residual_bindings for slot in binding.occurrences}
    if set(raw_output) != expected:
        missing = sorted(expected - set(raw_output))
        extra = sorted(set(raw_output) - expected)
        raise ValueError(f"residual output keys differ; missing={missing}, extra={extra}")
    outputs: dict[str, BindingOutput] = {}
    for binding in plan.residual_bindings:
        replacements: dict[str, str] = {}
        for slot in binding.occurrences:
            candidate = raw_output[slot.slot_id]
            if INVALID_TEXT_CONTROL.search(candidate):
                raise ValueError(
                    f"residual slot {slot.slot_id} contains an invalid control character; "
                    "return ordinary printable text with spaces, not control separators"
                )
            if "\\" not in slot.source_text and "\\" in candidate:
                raise ValueError(
                    f"residual slot {slot.slot_id} introduced a backslash/encoded separator; "
                    "return plain semantic words separated by spaces; the host owns line layout"
                )
            if slot.render_policy == "opaque_identifier":
                replacement = _shape_alphanumeric_like_source(slot.source_text, candidate)
            else:
                replacement = _layout_like_source(slot.source_text, candidate)
            replacements[slot.slot_id] = replacement
        if (
            not binding.target_paths
            and binding.value_kind in _SENSITIVE_AUXILIARY_KINDS
            and not _stable_source_vocabulary(binding)
        ):
            for slot in binding.occurrences:
                if (
                    _alphanumeric(replacements[slot.slot_id]).casefold()
                    == _alphanumeric(slot.source_text).casefold()
                ):
                    raise ValueError(
                        f"residual agent copied sensitive source auxiliary: {binding.logical_key}"
                    )
        if binding.target_paths:
            values = tuple(
                _binding_target_value(case.target, path) for path in binding.target_paths
            )
            canonical: JsonValue = cast(JsonValue, values[0] if len(values) == 1 else list(values))
        else:
            canonical = replacements[binding.occurrences[0].slot_id].strip()
        output = BindingOutput(replacements=replacements, canonical_value=canonical)
        _validate_binding_format(
            source=case.source, template=case.template.byte_template, output=output
        )
        outputs[binding.logical_key] = output
    return outputs


def _reconcile_identifier_relationships(
    *, template: CertifiedSemanticTemplate, outputs: dict[str, BindingOutput]
) -> None:
    """Project embedded source identifiers from their authoritative container surface."""

    bindings = {binding.logical_key: binding for binding in template.bindings}
    active_relationships = _active_identifier_relationships(template)
    parents_by_child: dict[str, set[str]] = {}
    for binding in template.bindings:
        for relationship in active_relationships[binding.logical_key]:
            if relationship.relationship == "embeds_exact_source_identifier":
                parent_key = binding.logical_key
                child_key = relationship.dependency_binding
            else:
                parent_key = relationship.dependency_binding
                child_key = binding.logical_key
            parents_by_child.setdefault(child_key, set()).add(parent_key)

    for child_key, parent_keys in sorted(parents_by_child.items()):
        child = bindings[child_key]
        child_source = _alphanumeric(child.occurrences[0].source_text)
        if not child_source:
            raise ValueError(f"embedded identifier has no alphanumeric source: {child_key}")
        candidates: set[str] = set()
        for parent_key in sorted(parent_keys):
            parent = bindings[parent_key]
            parent_source = _alphanumeric(parent.occurrences[0].source_text)
            offsets = tuple(
                match.start()
                for match in re.finditer(
                    re.escape(child_source.casefold()), parent_source.casefold()
                )
            )
            if len(offsets) != 1:
                raise ValueError(
                    f"source identifier relationship is not uniquely positioned: "
                    f"{parent_key} -> {child_key}"
                )
            parent_rendered = _alphanumeric(
                outputs[parent_key].replacements[parent.occurrences[0].slot_id]
            )
            if len(parent_rendered) != len(parent_source):
                raise ValueError(f"rendered relationship container changes length: {parent_key}")
            start = offsets[0]
            candidates.add(parent_rendered[start : start + len(child_source)])
        if len(candidates) != 1:
            raise ValueError(
                f"multiple relationship containers imply different values for {child_key}"
            )
        value = candidates.pop()
        if child.target_paths:
            existing = _alphanumeric(outputs[child_key].replacements[child.occurrences[0].slot_id])
            if existing.casefold() != value.casefold():
                raise ValueError(
                    f"embedded target binding conflicts with its container: {child_key}"
                )
            continue
        replacement = {
            slot.slot_id: _shape_alphanumeric_like_source(slot.source_text, value)
            for slot in child.occurrences
        }
        outputs[child_key] = BindingOutput(replacements=replacement, canonical_value=value)


def _country_code_map(path: Path) -> dict[str, str]:
    payload = json.loads(read_regular_file_bytes(path))
    rows = payload.get("3166-1") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("ISO-3166 snapshot lacks the 3166-1 registry")

    def normalized(value: str) -> str:
        return "".join(character for character in value.casefold() if character.isalnum())

    output: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("alpha_2"), str):
            raise ValueError("ISO-3166 registry row is malformed")
        code = cast(str, row["alpha_2"])
        for key in ("name", "official_name", "alpha_2", "alpha_3", "numeric"):
            value = row.get(key)
            if isinstance(value, str):
                existing = output.setdefault(normalized(value), code)
                if existing != code:
                    raise ValueError(f"ambiguous ISO-3166 name: {value}")
    aliases = {
        "uk": "GB",
        "greatbritain": "GB",
        "usa": "US",
        "unitedstates": "US",
        "uae": "AE",
        "prchina": "CN",
        "peoplesrepublicchina": "CN",
        "southkorea": "KR",
        "korea": "KR",
        "northkorea": "KP",
        "taiwan": "TW",
        "turkiye": "TR",
        "russia": "RU",
        "czechrepublic": "CZ",
        "ivorycoast": "CI",
        "democraticrepublicofthecongo": "CD",
        "hongkongchina": "HK",
    }
    for name, code in aliases.items():
        output.setdefault(name, code)
    return output


def _numeric_interpretations(value: str) -> tuple[Decimal, ...]:
    matches = tuple(match.group(0).strip() for match in _NUMBER.finditer(value))
    if len(matches) != 1:
        return ()
    return tuple(
        dict.fromkeys(
            parsed
            for parsed, _decimal, _grouping, _places in numeric_surface_interpretations(matches[0])
            if parsed.is_finite()
        )
    )


def _numeric_value(value: JsonValue) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"value is not numeric: {value!r}")
    if isinstance(value, (int, float, Decimal)):
        parsed = Decimal(str(value))
        if parsed.is_finite():
            return parsed
    if isinstance(value, str):
        candidates = _numeric_interpretations(value)
        if len(candidates) == 1:
            return candidates[0]
        word_value = _number_word_value(value)
        if word_value is not None:
            return Decimal(word_value)
    raise ValueError(f"value is not uniquely numeric: {value!r}")


def _first_surface_number(value: str) -> Decimal:
    candidates = _numeric_interpretations(value)
    if not candidates:
        word_value = _number_word_value(value)
        if word_value is None:
            raise ValueError(f"surface has no numeric interpretation: {value!r}")
        return Decimal(word_value)
    integers = tuple(candidate for candidate in candidates if candidate == candidate.to_integral())
    if len(integers) == 1:
        return integers[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"surface number is ambiguous: {value!r}")


def _number_to_words(value: int) -> str:
    if value < 0 or value >= 1_000_000_000:
        raise ValueError("number-to-words supports integers from zero through 999,999,999")
    small = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

    def below_thousand(number: int) -> list[str]:
        words: list[str] = []
        if number >= 100:
            words.extend((small[number // 100], "hundred"))
            number %= 100
        if number >= 20:
            words.append(tens[number // 10])
            if number % 10:
                words.append(small[number % 10])
        elif number:
            words.append(small[number])
        return words

    if value == 0:
        return small[0]
    words: list[str] = []
    for divisor, label in ((1_000_000, "million"), (1_000, "thousand")):
        if value >= divisor:
            words.extend(below_thousand(value // divisor))
            words.append(label)
            value %= divisor
    words.extend(below_thousand(value))
    return " ".join(words)


def _number_word_value(value: str) -> int | None:
    """Parse only complete canonical English number-word surfaces.

    Requiring a round trip through ``_number_to_words`` prevents incidental prose that happens
    to contain number words from being treated as a numeric field.
    """

    if any(character.isdigit() for character in value):
        return None
    tokens = tuple(re.findall(r"[a-z]+", value.casefold()))
    if not tokens:
        return None
    if "and" in tokens:
        # British conjunctions may bridge a hundreds/thousands group to its
        # remainder; an arbitrary conjunction is not a numeric dependency.
        for index, token in enumerate(tokens):
            if token == "and" and (
                index == 0
                or index == len(tokens) - 1
                or tokens[index - 1] not in {"hundred", "thousand", "million"}
            ):
                return None
        tokens = tuple(token for token in tokens if token != "and")
    small = {
        word: number
        for number, word in enumerate(
            (
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            )
        )
    }
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    total = 0
    group = 0
    for token in tokens:
        if token in small:
            group += small[token]
        elif token in tens:
            group += tens[token]
        elif token == "hundred" and 0 < group < 10:
            group *= 100
        elif token in {"thousand", "million"} and group:
            scale = 1_000 if token == "thousand" else 1_000_000
            total += group * scale
            group = 0
        else:
            return None
    parsed = total + group
    canonical_tokens = tuple(_number_to_words(parsed).split())
    return parsed if canonical_tokens == tokens else None


def _number_word_phrase(value: str) -> tuple[int, int, int]:
    tokens = tuple(re.finditer(r"[A-Za-z]+", value))
    candidates: list[tuple[int, int, int, int]] = []
    for start in range(len(tokens)):
        for stop in range(start + 1, len(tokens) + 1):
            parsed = _number_word_value(" ".join(match.group(0) for match in tokens[start:stop]))
            if parsed is not None:
                candidates.append(
                    (stop - start, tokens[start].start(), tokens[stop - 1].end(), parsed)
                )
    if not candidates:
        raise ValueError(f"surface has no complete number-word phrase: {value!r}")
    width = max(row[0] for row in candidates)
    longest = tuple(row for row in candidates if row[0] == width)
    if len(longest) != 1:
        raise ValueError(f"surface has multiple number-word phrases: {value!r}")
    _token_count, start, end, parsed = longest[0]
    return parsed, start, end


def _pluralize_number_word_noun(value: str, *, number: int) -> str:
    match = re.match(
        r"(?i)^(?P<gap>\s*)(?P<noun>container|package|pallet|carton|crate|drum|"
        r"bundle|case|piece|roll|sack|bag|box)(?P<suffix>\(s\)|es|s)?(?![A-Za-z])",
        value,
    )
    if match is None or match.group("suffix") == "(s)":
        return value
    noun = match.group("noun")
    suffix = match.group("suffix") or ""
    if number == 1:
        replacement = noun
    elif suffix:
        replacement = noun + suffix
    else:
        ending = "es" if noun.casefold() == "box" else "s"
        replacement = noun + (ending.upper() if noun.isupper() else ending)
    return match.group("gap") + replacement + value[match.end() :]


def _numeric_surface_values(value: str) -> tuple[Decimal, ...]:
    # A composite surface can contain several numeric facts. Digits inside an
    # identifier are not numeric evidence (e.g. TTNU6462679 is not a temperature).
    values = list(
        dict.fromkeys(
            parsed
            for match in re.finditer(
                r"(?<![A-Za-z0-9])[-+]?[0-9]+(?:[.,][0-9]+)*(?![A-Za-z0-9])", value
            )
            for parsed in _numeric_interpretations(match.group())
        )
    )
    word_value = _number_word_value(value)
    if word_value is not None and Decimal(word_value) not in values:
        values.append(Decimal(word_value))
    return tuple(values)


def _dependency_count(target: Mapping[str, Any], path: str) -> int:
    value = _resolve_path(target, path)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, Mapping):
        return 1
    raise ValueError(f"count derivation dependency is not a collection/object: {path}")


def _package_quantities(target: Mapping[str, Any]) -> tuple[int, ...]:
    patch = target.get("documentPatch")
    packages = patch.get("cargoPackages") if isinstance(patch, Mapping) else None
    if not isinstance(packages, Sequence):
        return ()
    output = []
    for package in packages:
        quantity = package.get("quantity") if isinstance(package, Mapping) else None
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            continue
        output.append(quantity)
    return tuple(output)


def _container_package_count_value(
    *, source_target: Mapping[str, Any], target: Mapping[str, Any], source_surface: str
) -> tuple[int, int]:
    source_patch = cast(Mapping[str, Any], source_target["documentPatch"])
    target_patch = cast(Mapping[str, Any], target["documentPatch"])
    source_candidates = {
        "containers": len(cast(Sequence[Any], source_patch.get("containers") or [])),
        "packages": len(cast(Sequence[Any], source_patch.get("cargoPackages") or [])),
        "package_quantity": sum(_package_quantities(source_target)),
    }
    target_candidates = {
        "containers": len(cast(Sequence[Any], target_patch.get("containers") or [])),
        "packages": len(cast(Sequence[Any], target_patch.get("cargoPackages") or [])),
        "package_quantity": sum(_package_quantities(target)),
    }
    for index, (source_quantity, target_quantity) in enumerate(
        zip(_package_quantities(source_target), _package_quantities(target), strict=True)
    ):
        key = f"package_quantity:{index}"
        source_candidates[key] = source_quantity
        target_candidates[key] = target_quantity
    observed = int(_first_surface_number(source_surface))
    matched = tuple(key for key, value in source_candidates.items() if value == observed)
    if not matched:
        raise ValueError("container/package receipt matches no source count")
    generated = {target_candidates[key] for key in matched}
    if len(generated) != 1:
        raise ValueError("container/package receipt is ambiguous after target generation")
    return observed, next(iter(generated))


def _numeric_leaves(value: Any, *, names: frozenset[str] | None = None) -> tuple[Decimal, ...]:
    def collect(current: Any, *, selected: bool) -> list[Decimal]:
        output: list[Decimal] = []
        if isinstance(current, Mapping):
            for key, child in current.items():
                output.extend(
                    collect(
                        child,
                        selected=selected or names is None or key in names,
                    )
                )
        elif isinstance(current, list):
            for child in current:
                output.extend(collect(child, selected=selected))
        elif selected or names is None:
            output.append(_numeric_value(cast(JsonValue, current)))
        return output

    root_is_scalar = not isinstance(value, (Mapping, list))
    return tuple(collect(value, selected=root_is_scalar))


def _derivation_numeric_values(
    *,
    binding: SemanticBinding,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    bindings: Mapping[str, SemanticBinding],
    outputs: Mapping[str, BindingOutput],
    numeric_auxiliary: Mapping[str, PreparedNumeric] | None = None,
) -> tuple[Decimal, Decimal]:
    derivation = binding.derivation
    if derivation in {"container_count", "package_count"}:
        if binding.dependency_paths:

            def count_dependencies(document: Mapping[str, Any]) -> Decimal:
                collection = "containers" if derivation == "container_count" else "cargoPackages"
                roots = [
                    re.match(rf"^(documentPatch\.{collection}\[\d+\])(?:\.|$)", path)
                    for path in binding.dependency_paths
                ]
                if all(match is not None for match in roots):
                    for match in roots:
                        if match is not None:
                            _resolve_path(document, match.group(1))
                    return Decimal(len({match.group(1) for match in roots if match is not None}))
                values = [_resolve_path(document, path) for path in binding.dependency_paths]
                if len(values) == 1 and isinstance(values[0], (Mapping, list)):
                    return Decimal(_dependency_count(document, binding.dependency_paths[0]))
                field = "containerNumber" if derivation == "container_count" else "packageId"
                if all(path.endswith("." + field) for path in binding.dependency_paths):
                    return Decimal(len({canonical_json_bytes(value) for value in values}))
                raise ValueError("count dependencies do not identify distinct typed entities")

            return count_dependencies(source_target), count_dependencies(target)
        if binding.dependency_bindings and not binding.dependency_paths:
            count = Decimal(len(binding.dependency_bindings))
            return count, count
        raise ValueError(f"count derivation lacks one countable dependency: {binding.logical_key}")
    if derivation == "container_package_count":
        old, new = _container_package_count_value(
            source_target=source_target,
            target=target,
            source_surface=binding.occurrences[0].source_text,
        )
        return Decimal(old), Decimal(new)
    filters = {
        "sum_package_quantity": frozenset({"quantity", "packageQuantity"}),
        "sum_gross_weight": frozenset({"value"}),
        "sum_net_weight": frozenset({"value"}),
        "sum_tare_weight": frozenset({"value", "tareWeight"}),
        "sum_volume": frozenset({"value"}),
        "sum_decimal_values": None,
    }
    if derivation not in filters:
        raise ValueError(f"derivation is not a path-based numeric sum: {derivation}")
    source_values: list[Decimal] = []
    target_values: list[Decimal] = []
    if (
        derivation in {"sum_gross_weight", "sum_net_weight", "sum_volume"}
        and binding.dependency_paths
    ):
        old_values = _measurement_dependency_values(binding, source_target)
        new_values = _measurement_dependency_values(binding, target)
        if not old_values or old_values.keys() != new_values.keys():
            raise ValueError("typed measurement dependencies are empty or changed topology")
        source_values.extend(old_values.values())
        target_values.extend(new_values.values())
    for path in () if source_values else binding.dependency_paths:
        names = filters[derivation]
        if path.endswith(".unit") and names == frozenset({"value"}):
            if _resolve_path(source_target, path) != _resolve_path(target, path):
                raise ValueError("measurement derivation units changed without a conversion")
            continue
        if names is not None and path.rsplit(".", 1)[-1] in names:
            names = None
        source_values.extend(_numeric_leaves(_resolve_path(source_target, path), names=names))
        target_values.extend(_numeric_leaves(_resolve_path(target, path), names=names))
    for key in binding.dependency_bindings:
        dependency = bindings[key]
        if dependency.target_paths and all(
            any(
                path == declared
                or path.startswith(declared + ".")
                or path.startswith(declared + "[")
                for declared in binding.dependency_paths
            )
            for path in dependency.target_paths
        ):
            continue
        if binding.dependency_paths and dependency.target_paths:
            shared = {
                path
                for path in dependency.target_paths
                if any(
                    path == declared
                    or path.startswith(declared + ".")
                    or path.startswith(declared + "[")
                    for declared in binding.dependency_paths
                )
            }
            if shared:
                representative = sorted(shared)[0]
                if all(
                    _resolve_path(document, path) == _resolve_path(document, representative)
                    for document in (source_target, target)
                    for path in dependency.target_paths
                ):
                    # A compiled scalar can also own equal allocation aliases.
                    # Count that scalar once only after proving both equalities.
                    continue
            raise ValueError(
                "numeric binding dependencies are not covered by declared target paths"
            )
        if numeric_auxiliary is not None and key in numeric_auxiliary:
            source_values.append(Decimal(numeric_auxiliary[key].contract.source_value))
            target_values.append(Decimal(numeric_auxiliary[key].value))
        elif len(dependency.target_paths) == 1 and isinstance(
            _resolve_path(source_target, dependency.target_paths[0]), (int, float)
        ):
            source_values.append(
                Decimal(str(_resolve_path(source_target, dependency.target_paths[0])))
            )
            target_values.append(Decimal(str(_resolve_path(target, dependency.target_paths[0]))))
        else:
            source_values.append(_numeric_value(dependency.occurrences[0].source_text))
            target_values.append(_numeric_value(outputs[key].canonical_value))
    if not source_values or not target_values:
        raise ValueError(f"numeric derivation has no values: {binding.logical_key}")
    return sum(source_values, Decimal(0)), sum(target_values, Decimal(0))


def _render_proven_numeric_derivation(
    binding: SemanticBinding,
    *,
    source_value: Decimal,
    target_value: Decimal,
    source_uncertainty: Decimal = Decimal(0),
    occurrences: Sequence[TemplateSlot] | None = None,
) -> BindingOutput:
    """Source arithmetic disambiguates punctuation; never infer an arbitrary ratio."""
    replacements = {}
    canonical = None
    for slot in binding.occurrences if occurrences is None else occurrences:
        if source_value == int(source_value) and target_value == int(target_value):
            alias = count_aliases.render_pair(
                slot.source_text, int(source_value), int(target_value)
            )
            if alias is not None:
                replacements[slot.slot_id] = _layout_like_source(slot.source_text, alias)
                canonical = target_value
                continue
        candidates = []
        for old in _numeric_interpretations(slot.source_text):
            try:
                step = surface_quantum(slot.source_text, old)
            except ValueError:
                continue
            if source_value.quantize(step, rounding=ROUND_HALF_UP) == old or (
                source_uncertainty > 0 and abs(source_value - old) <= step / 2 + source_uncertainty
            ):
                candidates.append((old, step))
        if len(candidates) == 1:
            old, step = candidates[0]
            new = target_value.quantize(step, rounding=ROUND_HALF_UP)
            replacements[slot.slot_id] = render_number_surface(slot.source_text, old, new)
            if canonical is None:
                canonical = new
        elif (
            not candidates
            and not any(c.isdigit() for c in slot.source_text)
            and source_value == source_value.to_integral_value()
        ):
            number, start, end = _number_word_phrase(slot.source_text)
            if Decimal(number) != source_value or target_value != target_value.to_integral_value():
                raise ValueError("number-word surface disagrees with the declared derivation")
            replacement = (
                slot.source_text[:start]
                + _case_like(slot.source_text[start:end], _number_to_words(int(target_value)))
                + _pluralize_number_word_noun(slot.source_text[end:], number=int(target_value))
            )
            replacements[slot.slot_id] = _layout_like_source(slot.source_text, replacement)
            if canonical is None:
                canonical = target_value
        else:
            raise ValueError(
                "source does not prove the declared arithmetic or its printed precision: "
                f"{binding.logical_key}"
            )
    if canonical is None:
        raise ValueError("numeric derivation has no printed occurrences")
    return BindingOutput(replacements=replacements, canonical_value=float(canonical))


def _measurement_dependency_values(
    binding: SemanticBinding, document: Mapping[str, Any]
) -> dict[str, Decimal]:
    names = {
        "sum_gross_weight": "grossWeight",
        "sum_net_weight": "netWeight",
        "sum_volume": "volume",
    }
    if binding.derivation not in names:
        raise ValueError("derivation is not a typed measurement sum")
    field_name = names[binding.derivation]
    selected = {}
    for path in binding.dependency_paths:
        for leaf_path, value in leaves(_resolve_path(document, path), path).items():
            if leaf_path.endswith("." + field_name + ".value"):
                selected[leaf_path] = _numeric_value(value)
    return selected


_MEASUREMENT_UNIT_PATTERNS = tuple(
    (unit, re.compile(r"(?<![A-Za-z])" + pattern + r"(?![A-Za-z])", re.I))
    for unit, pattern in (
        ("kilogram", r"(?:kg|kgs|kgm|kilograms?)"),
        ("metric_tonne", r"(?:mt|mts|metric[ \t]+tonnes?|tonnes?)"),
        ("pound", r"(?:lb|lbs|lbr|pounds?)"),
        ("cubic_metre", r"(?:cbm|mtq|m3|m³|cubic[ \t]+met(?:er|re)s?)"),
        ("cubic_foot", r"(?:cbf|cft|ft3|ft³|ftq|cubic[ \t]+feet)"),
    )
)
_MEASUREMENT_ADJACENT = re.compile(r"^[ \t]*[A-Za-z³0-9]+")


def _validate_description_volume_units(target: Mapping[str, Any], rendered: str) -> None:
    """Reject one-group documents that reuse a CBM number under an imperial unit."""
    if "cu. ft." not in rendered.casefold() and "ftq" not in rendered.casefold():
        return
    groups = target.get("documentPatch", {}).get("cargoGroups", ())
    if len(groups) != 1:
        return
    description = groups[0].get("description", "")
    if not isinstance(description, str) or "VOLUME" not in description.upper():
        return
    metric = {
        Decimal(match.group(1).replace(",", ""))
        for match in re.finditer(
            r"\bVOLUME\s+([0-9][0-9,.]*)\s*(?:CBM|M³|CUBIC\s+METRES?)\b",
            description,
            re.IGNORECASE,
        )
    }
    if not metric:
        return
    imperial = {
        Decimal(match.group(1).replace(",", ""))
        for match in re.finditer(
            r"\b([0-9][0-9,.]*)\s*(?:cu\.\s*ft\.|FTQ)(?=$|[^A-Za-z0-9])",
            rendered,
            re.IGNORECASE,
        )
    }
    if metric & imperial:
        raise ValueError(
            "the same cargo-volume number is stated as cubic metres in the "
            "target description and cubic feet in the rendered document"
        )


def _derivation_measurement_factor(
    binding: SemanticBinding,
    case: PreparedCase,
    *,
    value_paths: Sequence[str] = (),
    occurrences: Sequence[TemplateSlot] | None = None,
) -> tuple[Decimal, Decimal]:
    return _measurement_factor_for_source(
        binding,
        source=case.source,
        source_target=case.source_target,
        target=case.target,
        template=case.template,
        value_paths=value_paths,
        occurrences=occurrences,
    )


def _measurement_factor_for_source(
    binding: SemanticBinding,
    *,
    source: bytes,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    template: CertifiedSemanticTemplate,
    value_paths: Sequence[str] = (),
    occurrences: Sequence[TemplateSlot] | None = None,
) -> tuple[Decimal, Decimal]:
    """Use explicit typed input units and adjacent printed units, never a fitted ratio."""
    from .measurement_columns import ordered_column_unit

    if not value_paths and binding.derivation not in {
        "sum_gross_weight",
        "sum_net_weight",
        "sum_volume",
    }:
        return Decimal(1), Decimal(0)
    units = set()
    numeric_values = (
        {path: _numeric_value(_resolve_path(source_target, path)) for path in value_paths}
        if value_paths
        else _measurement_dependency_values(binding, source_target)
    )
    for path in numeric_values:
        unit_path = path.removesuffix(".value") + ".unit"
        old = _resolve_path(source_target, unit_path)
        if _resolve_path(target, unit_path) != old:
            raise ValueError("derived measurement cannot change its source unit contract")
        units.add(old)
    if not units:
        return Decimal(1), Decimal(0)
    if len(units) != 1:
        raise ValueError("derived measurement sum mixes source units")
    source_unit = units.pop()
    printed_units = set()
    for slot in binding.occurrences if occurrences is None else occurrences:
        column_unit = ordered_column_unit(
            source, byte_end=slot.byte_end, surface=slot.source_text
        )
        suffix = source[slot.byte_end :].split(b"\n", 1)[0].decode()
        # Only the immediately adjacent unit is context; subsequent columns
        # or prose cannot supply a conversion for this numeric field.
        adjacent = _MEASUREMENT_ADJACENT.match(suffix)
        text = column_unit or slot.source_text + (adjacent[0] if adjacent is not None else "")
        for unit, pattern in _MEASUREMENT_UNIT_PATTERNS:
            if pattern.search(text):
                printed_units.add(unit)
    if not printed_units:
        return Decimal(1), Decimal(0)
    if len(printed_units) != 1:
        raise ValueError("derived measurement occurrences use differing units")
    printed = printed_units.pop()
    masses = {"kilogram": Decimal(1), "metric_tonne": Decimal(1000), "pound": Decimal("0.45359237")}
    volumes = {"cubic_metre": Decimal(1), "cubic_foot": Decimal("0.028316846592")}
    if source_unit in masses and printed in masses:
        factor = masses[source_unit] / masses[printed]
    elif source_unit in volumes and printed in volumes:
        factor = volumes[source_unit] / volumes[printed]
    else:
        raise ValueError("derived measurement units are dimensionally incompatible")
    uncertainty = Decimal(0)
    if factor != 1:
        for path, value in numeric_values.items():
            steps = []
            for member in template.bindings:
                if path not in member.target_paths:
                    continue
                for slot in member.occurrences:
                    try:
                        steps.append(surface_quantum(slot.source_text, value))
                    except ValueError:
                        continue
            if not steps:
                raise ValueError("unit conversion lacks proven input-side printed precision")
            uncertainty += min(steps) * factor / 2
    return factor, uncertainty


def _render_target_measurement(
    binding: SemanticBinding, case: PreparedCase
) -> BindingOutput | None:
    """Render every typed numeric occurrence in its own proved printed unit."""
    paths = tuple(
        path
        for path in binding.target_paths
        if re.search(r"\.(grossWeight|netWeight|volume)\.value$", path)
    )
    if not paths or set(paths) != set(binding.target_paths):
        return None
    old_values = {_numeric_value(_resolve_path(case.source_target, path)) for path in paths}
    new_values = {_numeric_value(_resolve_path(case.target, path)) for path in paths}
    if len(old_values) != 1 or len(new_values) != 1:
        raise ValueError("measurement binding does not identify one repeated numeric value")
    old, new = old_values.pop(), new_values.pop()
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        factor, uncertainty = _derivation_measurement_factor(
            binding, case, value_paths=paths, occurrences=(slot,)
        )
        output = _render_proven_numeric_derivation(
            binding,
            source_value=old * factor,
            target_value=new * factor,
            source_uncertainty=uncertainty,
            occurrences=(slot,),
        )
        replacements.update(output.replacements)
    return BindingOutput(replacements=replacements, canonical_value=float(new))


def _render_inclusive_range_cardinality(
    binding: SemanticBinding,
    *,
    target: Mapping[str, Any],
    outputs: Mapping[str, BindingOutput],
) -> BindingOutput:
    if len(binding.dependency_paths) == 1 and not binding.dependency_bindings:
        value = _resolve_path(target, binding.dependency_paths[0])
    elif len(binding.dependency_bindings) == 1 and not binding.dependency_paths:
        # An outer packing level may be printed without an extraction quantity.
        # Its already prepared numeric owner, not the inner carton target, owns
        # this range. The compiler proves the source cardinality against it.
        value = outputs[binding.dependency_bindings[0]].canonical_value
    else:
        raise ValueError("inclusive-range derivation requires exactly one quantity owner")
    numeric = _numeric_value(value)
    if numeric != numeric.to_integral_value() or numeric <= 0:
        raise ValueError("inclusive-range derivation quantity is not a positive integer")
    quantity = int(numeric)
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        match = whole_inclusive_range_surface(slot.source_text)
        if match is None:
            raise ValueError("certified inclusive-range surface no longer matches its adapter")
        start = match.start
        old_end = match.end
        direction = 1 if old_end >= start else -1
        new_end = start + direction * (quantity - 1)
        rendered_end = render_number_surface(
            slot.source_text[match.end_start : match.end_end], old_end, new_end
        )
        replacements[slot.slot_id] = match.replace_end(slot.source_text, rendered_end)
    return BindingOutput(replacements=replacements, canonical_value=quantity)


def _render_temperature_setpoint(
    binding: SemanticBinding, *, target: Mapping[str, Any]
) -> BindingOutput:
    fields: dict[str, JsonValue] = {}
    for path in binding.dependency_paths:
        field = path.rsplit(".", 1)[-1]
        if field not in {"unit", "value"} or field in fields:
            raise ValueError("temperature-setpoint dependencies are not one unit/value pair")
        fields[field] = cast(JsonValue, _resolve_path(target, path))
    if set(fields) != {"unit", "value"}:
        raise ValueError("temperature-setpoint dependencies are not one unit/value pair")
    unit = fields["unit"]
    raw_value = fields["value"]
    if not isinstance(unit, str) or unit.casefold() not in {"celsius", "fahrenheit"}:
        raise ValueError("temperature-setpoint unit is unsupported")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError("temperature-setpoint value is not numeric")
    value = Decimal(str(raw_value))
    normalized_exponent = value.normalize().as_tuple().exponent
    if not isinstance(normalized_exponent, int):
        raise ValueError("temperature-setpoint value is not finite")
    required_precision = max(0, -normalized_exponent)
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        match = _TEMPERATURE_SURFACE.fullmatch(slot.source_text)
        if match is None:
            raise ValueError("certified temperature surface no longer matches its adapter")
        source_precision = len(match.group("fraction") or "")
        precision = max(source_precision, required_precision)
        number = f"{value:.{precision}f}"
        if value >= 0 and match.group("sign") == "+":
            number = "+" + number
        if match.group("separator") == ",":
            number = number.replace(".", ",")
        source_unit = match.group("unit")
        if len(source_unit) == 1:
            rendered_unit = "C" if unit.casefold() == "celsius" else "F"
        else:
            rendered_unit = "Celsius" if unit.casefold() == "celsius" else "Fahrenheit"
        rendered_unit = _case_like(source_unit, rendered_unit)
        replacements[slot.slot_id] = (
            match.group("leading")
            + number
            + match.group("number_gap")
            + match.group("degree")
            + match.group("unit_gap")
            + rendered_unit
            + match.group("trailing")
        )
    canonical_value: JsonValue = {
        "unit": unit,
        "value": int(value) if value == value.to_integral_value() else float(value),
    }
    return BindingOutput(replacements=replacements, canonical_value=canonical_value)


def _dependency_canonical(
    binding: SemanticBinding, outputs: Mapping[str, BindingOutput], target: Mapping[str, Any]
) -> JsonValue:
    values: list[JsonValue] = []
    if binding.dependency_paths:
        # A direct semantic path is more precise than a composite dependency binding. The latter
        # still establishes graph order, but its canonical value may contain an entire party or
        # address while the derivation names only that party's country.
        for path in binding.dependency_paths:
            values.append(cast(JsonValue, _resolve_path(target, path)))
    else:
        for key in binding.dependency_bindings:
            if key not in outputs:
                raise KeyError(key)
            values.append(outputs[key].canonical_value)
    if not values:
        raise ValueError(f"derived binding has no dependency value: {binding.logical_key}")
    first = values[0]
    if all(canonical_json_bytes(value) == canonical_json_bytes(first) for value in values[1:]):
        return first
    return cast(JsonValue, values)


def _render_same_value(binding: SemanticBinding, value: JsonValue) -> BindingOutput:
    surface = _scalar_surface(value)
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        if slot.render_policy == "opaque_identifier":
            replacement = _shape_alphanumeric_like_source(slot.source_text, surface)
        else:
            replacement = _layout_like_source(slot.source_text, surface)
        replacements[slot.slot_id] = replacement
    return BindingOutput(replacements=replacements, canonical_value=value)


def _render_same_as_binding(
    binding: SemanticBinding,
    dependency: SemanticBinding,
    dependency_output: BindingOutput,
) -> BindingOutput:
    """Project a proved source relationship instead of copying a dependency surface blindly."""

    if binding.value_kind == "date":
        dependency_dates = _date_candidates(dependency.occurrences[0].source_text)
        target_dates = (
            _date_candidates(dependency_output.canonical_value)
            if isinstance(dependency_output.canonical_value, str)
            else frozenset()
        )
        if len(dependency_dates) != 1 or len(target_dates) != 1:
            raise ValueError("same-as date dependency is not uniquely interpretable")
        old_date = next(iter(dependency_dates))
        new_date = next(iter(target_dates))
        replacements: dict[str, str] = {}
        for slot in binding.occurrences:
            if _date_candidates(slot.source_text) != frozenset({old_date}):
                raise ValueError("same-as date source differs from its dependency")
            replacements[slot.slot_id] = _render_date_surface(
                slot.source_text,
                old_date.isoformat(),
                new_date.isoformat(),
            )
        return BindingOutput(
            replacements=replacements,
            canonical_value=new_date.isoformat(),
        )

    if binding.value_kind == "phone" and all(
        _alphanumeric(s.source_text) == _alphanumeric(dependency.occurrences[0].source_text)
        and s.render_policy != "opaque_identifier"
        for s in binding.occurrences
    ):
        # A complete repeated phone is semantic equality, not a fixed-width ID.
        return _render_same_value(binding, dependency_output.canonical_value)
    if binding.value_kind in {"identifier", "phone"}:
        dependency_source = _alphanumeric(dependency.occurrences[0].source_text)
        dependency_target = _alphanumeric(_scalar_surface(dependency_output.canonical_value))
        if not dependency_source or len(dependency_source) != len(dependency_target):
            raise ValueError("same-as identifier dependency changes width")
        replacements = {}
        for slot in binding.occurrences:
            source = _alphanumeric(slot.source_text)
            dependency_inside_surface = tuple(
                match.start()
                for match in re.finditer(re.escape(dependency_source.casefold()), source.casefold())
            )
            surface_inside_dependency = tuple(
                match.start()
                for match in re.finditer(re.escape(source.casefold()), dependency_source.casefold())
            )
            if len(dependency_inside_surface) == 1:
                offset = dependency_inside_surface[0]
                candidate = (
                    source[:offset] + dependency_target + source[offset + len(dependency_source) :]
                )
            elif len(surface_inside_dependency) == 1:
                offset = surface_inside_dependency[0]
                candidate = dependency_target[offset : offset + len(source)]
            else:
                # A declared same-as edge may print a delimited component with
                # an alphabetic form prefix, rather than the whole reference.
                projections = []
                for token in re.finditer(r"[A-Za-z0-9]+", dependency.occurrences[0].source_text):
                    part = token.group()
                    if not any(c.isdigit() for c in part):
                        continue
                    positions = tuple(re.finditer(re.escape(part), source, flags=re.IGNORECASE))
                    if len(positions) != 1:
                        continue
                    hit = positions[0]
                    if any(c.isdigit() for c in source[: hit.start()] + source[hit.end() :]):
                        continue
                    offset = len(
                        _alphanumeric(dependency.occurrences[0].source_text[: token.start()])
                    )
                    projections.append(
                        source[: hit.start()]
                        + dependency_target[offset : offset + len(part)]
                        + source[hit.end() :]
                    )
                if len(projections) != 1:
                    raise ValueError("same-as identifier dependency is not uniquely embedded")
                candidate = projections[0]
            replacements[slot.slot_id] = _shape_alphanumeric_like_source(
                slot.source_text, candidate
            )
        return BindingOutput(
            replacements=replacements,
            canonical_value=replacements[binding.occurrences[0].slot_id],
        )

    if binding.value_kind == "location":
        # A compiler-declared same-as edge is the semantic evidence.  Geographic aliases can
        # differ lexically (for example TAIWAN vs TAIWAN, PROVINCE OF CHINA), so render every
        # projection from the one new canonical dependency instead of requiring source text
        # equality.
        return _render_same_value(binding, dependency_output.canonical_value)

    source_value = dependency.occurrences[0].source_text
    if binding.value_kind in {"integer", "decimal_measurement"}:
        return _render_proven_numeric_derivation(
            binding,
            source_value=_numeric_value(source_value),
            target_value=_numeric_value(dependency_output.canonical_value),
        )
    if any(
        _normalized_semantic(slot.source_text) != _normalized_semantic(source_value)
        for slot in binding.occurrences
    ):
        raise ValueError("same-as text source is not semantically identical to its dependency")
    return _render_same_value(binding, dependency_output.canonical_value)


def _number_to_words_value(
    binding: SemanticBinding,
    outputs: Mapping[str, BindingOutput],
    target: Mapping[str, Any],
) -> Decimal:
    if (
        len(binding.dependency_paths) == 1
        and not binding.dependency_bindings
        and isinstance(_resolve_path(target, binding.dependency_paths[0]), (Mapping, list))
    ):
        return Decimal(_dependency_count(target, binding.dependency_paths[0]))
    values: set[Decimal] = set()
    dependencies: list[JsonValue] = [
        outputs[key].canonical_value for key in binding.dependency_bindings
    ]
    dependencies.extend(
        cast(JsonValue, _resolve_path(target, path)) for path in binding.dependency_paths
    )
    for value in dependencies:
        values.add(_numeric_value(value))
    if len(values) != 1:
        raise ValueError(
            f"number-to-words dependency is not uniquely numeric: {binding.logical_key}"
        )
    return values.pop()


def _render_gross_minus_net_weight(binding: SemanticBinding, case: PreparedCase) -> BindingOutput:
    """Packaging mass from the same cargo group's explicit gross and net facts.

    This is not container tare. Each input converts independently into the
    explicitly printed output unit; mixed kg/tonne/pound inputs are supported.
    An unqualified number or a cross-group subtraction cannot prove this fact.
    """
    paths = binding.dependency_paths
    match = (
        re.fullmatch(r"(documentPatch\.cargoGroups\[\d+\])\.grossWeight\.value", paths[0])
        if paths
        else None
    )
    if (
        match is None
        or len(paths) != 2
        or paths[1] != match[1] + ".netWeight.value"
        or binding.target_paths
        or binding.dependency_bindings
        or binding.render_mode != "deterministic_derived"
        or binding.group_kind != "cargo"
        or binding.value_kind != "decimal_measurement"
    ):
        raise ValueError("gross-minus-net requires one ordered gross/net cargo owner")
    for slot in binding.occurrences:
        suffix = case.source[slot.byte_end :].split(b"\n", 1)[0].decode()
        adjacent = re.match(r"^[ \t]*[A-Za-z³0-9]+", suffix)
        text = slot.source_text + (adjacent[0] if adjacent else "")
        if not re.search(
            r"(?i)(?<![A-Za-z])(?:kgs?|kgm|kilograms?|mt|mts|"
            r"metric[ \t]+tonnes?|tonnes?|lbs?|pounds?)(?![A-Za-z])",
            text,
        ):
            raise ValueError("gross-minus-net requires an explicit printed mass unit")
    old = []
    new = []
    uncertainty = Decimal(0)
    for path in paths:
        if _resolve_path(case.source_target, path.removesuffix(".value") + ".unit") not in {
            "kilogram",
            "metric_tonne",
            "pound",
        }:
            raise ValueError("gross-minus-net input is not a mass measurement")
        factor, rounding = _derivation_measurement_factor(binding, case, value_paths=(path,))
        old.append(_numeric_value(_resolve_path(case.source_target, path)) * factor)
        new.append(_numeric_value(_resolve_path(case.target, path)) * factor)
        uncertainty += rounding
    if min(*old, *new) < 0 or old[0] < old[1] or new[0] < new[1]:
        raise ValueError("gross-minus-net requires nonnegative mass with gross at least net")
    return _render_proven_numeric_derivation(
        binding,
        source_value=old[0] - old[1],
        target_value=new[0] - new[1],
        source_uncertainty=uncertainty,
    )


def _render_one_derivation(
    *,
    binding: SemanticBinding,
    case: PreparedCase,
    outputs: Mapping[str, BindingOutput],
    bindings: Mapping[str, SemanticBinding],
    country_codes: Mapping[str, str],
) -> BindingOutput:
    derivation = binding.derivation
    from . import cargo_identity_derivations, dangerous_goods_realization, transport_derivations
    from .route_derivations import DERIVATIONS, endpoint

    if derivation == "gross_minus_net_weight":
        return _render_gross_minus_net_weight(binding, case)

    if derivation in dangerous_goods_realization.DERIVATIONS:
        outputs_dg = _dangerous_goods_outputs(case)
        if binding.logical_key not in outputs_dg:
            raise ValueError("source-only DG fact lacks its complete sampled registry tuple")
        return outputs_dg[binding.logical_key]

    if derivation in cargo_identity_derivations.DERIVATIONS:
        cargo_identity_derivations.owner(binding)
        if binding.logical_key not in case.auxiliary_values:
            raise ValueError("repeated commodity name lacks its pinned goods value")
        return _render_text_candidate(binding, case.auxiliary_values[binding.logical_key])

    if derivation in transport_derivations.DERIVATIONS:
        transport_derivations.validate(binding)
        if binding.logical_key not in case.auxiliary_values:
            raise ValueError("independent vessel lacks its pinned registry value")
        return _render_text_candidate(binding, case.auxiliary_values[binding.logical_key])

    if derivation in DERIVATIONS:
        endpoint(binding)
        if binding.logical_key not in case.auxiliary_values:
            raise ValueError("sampled-route surface lacks its pinned scenario value")
        return _render_text_candidate(binding, case.auxiliary_values[binding.logical_key])
    if derivation == "same_as_binding":
        if binding.value_kind == "equipment":
            aggregate_inventory = _mixed_inventory_for_case(case)
            if (
                aggregate_inventory is not None
                and binding.logical_key in aggregate_inventory.binding_keys
            ):
                return _render_equipment_receipt_binding(
                    binding,
                    source_target=case.source_target,
                    target=case.target,
                    aggregate_inventory=aggregate_inventory,
                )
        if (
            binding.value_kind == "package"
            and binding.dependency_paths
            and all(
                re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.typeCategory", path)
                for path in binding.dependency_paths
            )
        ):
            old_categories = {
                _resolve_path(case.source_target, p) for p in binding.dependency_paths
            }
            new_categories = {_resolve_path(case.target, p) for p in binding.dependency_paths}
            if len(old_categories) != 1 or len(new_categories) != 1:
                raise ValueError("shared package noun requires equal owned package categories")
            source_category, target_category = old_categories.pop(), new_categories.pop()
            if not all(
                _normalized_semantic(s.source_text) in _package_variants(source_category)
                for s in binding.occurrences
            ):
                raise ValueError("shared package noun contradicts its source category")
            return BindingOutput(
                replacements={
                    s.slot_id: _package_candidate(s.source_text, target_category)
                    for s in binding.occurrences
                },
                canonical_value=target_category,
            )
        if binding.dependency_paths:
            return _render_same_value(binding, _dependency_canonical(binding, outputs, case.target))
        if len(binding.dependency_bindings) == 1:
            key = binding.dependency_bindings[0]
            return _render_same_as_binding(binding, bindings[key], outputs[key])
        return _render_same_value(binding, _dependency_canonical(binding, outputs, case.target))
    if derivation == "country_code":
        country = _dependency_canonical(binding, outputs, case.target)
        if not isinstance(country, str):
            raise ValueError(f"country-code dependency is not text: {binding.logical_key}")
        normalized = "".join(character for character in country.casefold() if character.isalnum())
        code = country_codes.get(normalized)
        if code is None:
            raise ValueError(f"country is absent from the pinned ISO registry: {country!r}")
        return _render_same_value(binding, code)
    if derivation == "number_to_words":
        quantities = tuple(
            path
            for path in binding.dependency_paths
            if re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", path)
        )
        if (
            quantities
            and not binding.dependency_bindings
            and all(
                re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.(quantity|typeCategory)", path)
                for path in binding.dependency_paths
            )
        ):
            rendered = _render_proven_numeric_derivation(
                binding,
                source_value=sum(
                    (Decimal(str(_resolve_path(case.source_target, path))) for path in quantities),
                    Decimal(0),
                ),
                target_value=sum(
                    (Decimal(str(_resolve_path(case.target, path))) for path in quantities),
                    Decimal(0),
                ),
            )
            category_paths = tuple(
                p for p in binding.dependency_paths if p.endswith(".typeCategory")
            )
            if category_paths:
                old_kinds = {_resolve_path(case.source_target, p) for p in category_paths}
                new_kinds = {_resolve_path(case.target, p) for p in category_paths}
                if len(old_kinds) != 1 or len(new_kinds) != 1:
                    raise ValueError("number-word package frame has mixed package categories")
                old_kind, new_kind = old_kinds.pop(), new_kinds.pop()
                if old_kind != new_kind:
                    replacements = {}
                    for slot in binding.occurrences:
                        text = rendered.replacements[slot.slot_id]
                        _, _, count_end = _number_word_phrase(text)
                        words = tuple(re.finditer(r"[A-Za-z]+", text[count_end:]))
                        matches = [
                            (count_end + a.start(), count_end + b.end())
                            for i, a in enumerate(words)
                            for b in words[i:]
                            if (
                                _normalized_semantic(
                                    text[count_end + a.start() : count_end + b.end()]
                                )
                                in _package_variants(old_kind)
                                or (
                                    len(text[count_end + a.start() : count_end + b.end()].split())
                                    == len(old_kind.removeprefix("PACKAGE_").split("_"))
                                    and package_category_surface_present(
                                        text[count_end + a.start() : count_end + b.end()],
                                        old_kind,
                                    )
                                )
                            )
                        ]
                        matches = [
                            (a, b)
                            for a, b in matches
                            if not any(c <= a and b <= d and (a, b) != (c, d) for c, d in matches)
                        ]
                        if len(matches) != 1:
                            raise ValueError("number-word package noun has no unique source proof")
                        start, end = matches[0]
                        if text[end : end + 1] == ")" and text[start:end].count("(") > text[
                            start:end
                        ].count(")"):
                            end += 1
                        noun = _pluralize_number_word_noun(
                            _package_surface(new_kind),
                            number=int(_numeric_value(rendered.canonical_value)),
                        )
                        replacements[slot.slot_id] = (
                            text[:start] + _case_like(text[start:end], noun) + text[end:]
                        )
                    return BindingOutput(
                        replacements=replacements, canonical_value=rendered.canonical_value
                    )
            return rendered
        numeric = _number_to_words_value(binding, outputs, case.target)
        source_outputs = {
            key: BindingOutput(
                replacements={}, canonical_value=bindings[key].occurrences[0].source_text
            )
            for key in binding.dependency_bindings
        }
        source_numeric = _number_to_words_value(binding, source_outputs, case.source_target)
        return _render_proven_numeric_derivation(
            binding, source_value=source_numeric, target_value=numeric
        )
    if derivation == "sum_monetary_amounts":
        if binding.dependency_paths or not binding.dependency_bindings:
            raise ValueError("monetary sum requires explicit amount binding dependencies")
        new = sum(
            (_numeric_value(outputs[key].canonical_value) for key in binding.dependency_bindings),
            Decimal(0),
        )
        old = sum(
            (
                _numeric_value(bindings[key].occurrences[0].source_text)
                for key in binding.dependency_bindings
            ),
            Decimal(0),
        )
        return _render_proven_numeric_derivation(binding, source_value=old, target_value=new)
    if derivation == "temperature_setpoint":
        return _render_temperature_setpoint(binding, target=case.target)
    if derivation == "inclusive_range_cardinality":
        return _render_inclusive_range_cardinality(binding, target=case.target, outputs=outputs)
    if derivation == "equipment_receipt":
        return _render_equipment_receipt_binding(
            binding,
            source_target=case.source_target,
            target=case.target,
            aggregate_inventory=_mixed_inventory_for_case(case),
        )

    source_base, target_base = _derivation_numeric_values(
        binding=binding,
        source_target=case.source_target,
        target=case.target,
        bindings=bindings,
        outputs=outputs,
        numeric_auxiliary=case.numeric_auxiliary,
    )
    factor, uncertainty = _derivation_measurement_factor(binding, case)
    return _render_proven_numeric_derivation(
        binding,
        source_value=source_base * factor,
        target_value=target_base * factor,
        source_uncertainty=uncertainty,
    )


def _render_derivations(
    *,
    case: PreparedCase,
    outputs: dict[str, BindingOutput],
    country_codes: Mapping[str, str],
    residual_bindings: Sequence[SemanticBinding] = (),
) -> None:
    bindings = {binding.logical_key: binding for binding in case.template.bindings}
    # Residual responses cannot certify their own derived totals. Discard those
    # representations and recompute after their dependency outputs are available.
    for binding in residual_bindings:
        if binding.realization.mode == "deterministic_derivation":
            outputs.pop(binding.logical_key, None)
    pending = {
        binding.logical_key: binding
        for binding in case.template.bindings
        if binding.realization.mode == "deterministic_derivation"
        and binding.logical_key not in outputs
    }

    while pending:
        progressed = False
        for logical_key, binding in tuple(pending.items()):
            if any(key not in outputs for key in binding.dependency_bindings):
                continue
            output = _render_one_derivation(
                binding=binding,
                case=case,
                outputs=outputs,
                bindings=bindings,
                country_codes=country_codes,
            )
            _validate_binding_format(
                source=case.source, template=case.template.byte_template, output=output
            )
            outputs[logical_key] = output
            del pending[logical_key]
            progressed = True
        if not progressed:
            raise ValueError(
                "derived binding graph cannot be resolved: " + ", ".join(sorted(pending))
            )


def _plan_derivations(
    *,
    case: PreparedCase,
    outputs: dict[str, BindingOutput],
    routes: list[BindingRoute],
    residual: list[SemanticBinding],
    country_codes: Mapping[str, str],
) -> None:
    """Prove derivations before calls; defer only dependencies not yet generated."""

    bindings = {binding.logical_key: binding for binding in case.template.bindings}
    route_indexes = {route.logical_key: index for index, route in enumerate(routes)}
    residual_keys = {binding.logical_key for binding in residual}
    pending = {
        binding.logical_key: binding
        for binding in case.template.bindings
        if binding.realization.mode == "deterministic_derivation"
        and binding.logical_key not in outputs
        and binding.logical_key not in residual_keys
    }

    def route_to_agent(binding: SemanticBinding, reason: str) -> None:
        residual.append(binding)
        residual_keys.add(binding.logical_key)
        index = route_indexes[binding.logical_key]
        routes[index] = routes[index].model_copy(
            update={"runtime_route": "agent", "route_reason": reason}
        )

    def dependencies_preserve_source(binding: SemanticBinding) -> bool:
        from . import cargo_identity_derivations, transport_derivations
        from .route_derivations import DERIVATIONS

        if binding.derivation in (
            DERIVATIONS | transport_derivations.DERIVATIONS | cargo_identity_derivations.DERIVATIONS
        ):
            return False  # Its latent country/LOCODE is part of the pinned scenario too.
        for path in binding.dependency_paths:
            try:
                source_value = _binding_target_value(case.source_target, path)
                target_value = _binding_target_value(case.target, path)
            except (KeyError, ValueError):
                return False
            if canonical_json_bytes(source_value) != canonical_json_bytes(target_value):
                return False
        for dependency_key in binding.dependency_bindings:
            dependency = bindings[dependency_key]
            output = outputs.get(dependency_key)
            if output is None or any(
                output.replacements.get(slot.slot_id) != slot.source_text
                for slot in dependency.occurrences
            ):
                return False
        return bool(binding.dependency_paths or binding.dependency_bindings)

    while pending:
        progressed = False
        for logical_key, binding in tuple(pending.items()):
            missing = tuple(key for key in binding.dependency_bindings if key not in outputs)
            if missing:
                if any(key in residual_keys for key in missing):
                    route_to_agent(
                        binding,
                        "derivation depends on residual binding(s): " + ", ".join(missing),
                    )
                    del pending[logical_key]
                    progressed = True
                continue
            if dependencies_preserve_source(binding):
                output = _preserved_source_output(binding)
                _validate_binding_format(
                    source=case.source,
                    template=case.template.byte_template,
                    output=output,
                )
                outputs[logical_key] = output
                index = route_indexes[logical_key]
                routes[index] = routes[index].model_copy(
                    update={
                        "runtime_route": "deterministic",
                        "route_reason": (
                            "derivation dependencies preserve their compiler-certified "
                            "source surfaces"
                        ),
                    }
                )
                del pending[logical_key]
                progressed = True
                continue
            try:
                output = _render_one_derivation(
                    binding=binding,
                    case=case,
                    outputs=outputs,
                    bindings=bindings,
                    country_codes=country_codes,
                )
                _validate_binding_format(
                    source=case.source,
                    template=case.template.byte_template,
                    output=output,
                )
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"unproven derived binding requires contract review: {logical_key}: {error}"
                ) from error
            else:
                outputs[logical_key] = output
            del pending[logical_key]
            progressed = True
        if progressed:
            continue
        raise ValueError("unresolved derived dependency graph: " + ", ".join(sorted(pending)))


def _normalized_semantic(value: str) -> str:
    normalized = value.casefold().replace("³", "3").replace("²", "2")
    return "".join(character for character in normalized if character.isalnum())


def _english_word_date_candidate(value: str) -> date | None:
    match = re.fullmatch(
        r"(?i)\s*(?P<month>[A-Za-z]{3,9})\.?\s+"
        r"(?P<day>[A-Za-z]+(?:[- ]+[A-Za-z]+)*)\s*,?\s*(?P<year>[0-9]{4})\s*",
        value,
    )
    if match is None:
        return None
    month_names = {name.casefold(): index for index, name in enumerate(_ENGLISH_MONTHS, start=1)}
    month_names.update({name[:3].casefold(): index for name, index in month_names.items()})
    month = month_names.get(match.group("month").casefold())
    if month is None:
        return None
    ordinal_to_cardinal = {
        "first": "one",
        "second": "two",
        "third": "three",
        "fourth": "four",
        "fifth": "five",
        "sixth": "six",
        "seventh": "seven",
        "eighth": "eight",
        "ninth": "nine",
        "tenth": "ten",
        "eleventh": "eleven",
        "twelfth": "twelve",
        "thirteenth": "thirteen",
        "fourteenth": "fourteen",
        "fifteenth": "fifteen",
        "sixteenth": "sixteen",
        "seventeenth": "seventeen",
        "eighteenth": "eighteen",
        "nineteenth": "nineteen",
        "twentieth": "twenty",
        "thirtieth": "thirty",
    }
    day_tokens = re.findall(r"[A-Za-z]+", match.group("day").casefold())
    day_tokens[-1] = ordinal_to_cardinal.get(day_tokens[-1], day_tokens[-1])
    day = _number_word_value(" ".join(day_tokens))
    if day is None or not 1 <= day <= 31:
        return None
    try:
        return date(int(match.group("year")), month, day)
    except ValueError:
        return None


def _date_candidates(value: str) -> frozenset[date]:
    raw = value.strip()
    localized = re.sub(r"(?i)(?<=\d)(?:st|nd|rd|th)\b", "", raw)
    french_months = (
        (r"janv(?:ier)?", "Jan"),
        (r"f[ée]v(?:r(?:ier)?)?", "Feb"),
        (r"mars", "Mar"),
        (r"avr(?:il)?", "Apr"),
        (r"mai", "May"),
        (r"juin", "Jun"),
        (r"juil(?:let)?", "Jul"),
        (r"ao[uû]t", "Aug"),
        (r"sept(?:embre)?", "Sep"),
        (r"oct(?:obre)?", "Oct"),
        (r"nov(?:embre)?", "Nov"),
        (r"d[ée]c(?:embre)?", "Dec"),
    )
    for pattern, english in french_months:
        localized = re.sub(rf"(?i)\b(?:{pattern})\b", english, localized)
    month_period_normalized = re.sub(r"(?i)\b([A-Z]{3,9})\.(?=[0-9])", r"\1 ", localized)
    # Keep the original punctuation alongside normalized alternatives. Formats such
    # as 20.APR.2025 are valid surfaces in the source corpus and cannot be recovered
    # after removing the period that follows the month token.
    variants = {
        raw,
        localized,
        month_period_normalized,
        " ".join(localized.replace(",", " ").split()),
        " ".join(month_period_normalized.replace(",", " ").split()),
    }
    variants.update(
        suffix.strip()
        for candidate in tuple(variants)
        if "," in candidate
        for suffix in (candidate.split(",", 1)[1],)
        if suffix.strip()
    )
    output: set[date] = set()
    for variant in variants:
        word_date = _english_word_date_candidate(variant)
        if word_date is not None:
            output.add(word_date)
        for date_format in _DATE_FORMATS:
            try:
                output.add(datetime.strptime(variant, date_format).date())
            except ValueError:
                continue
    return frozenset(output)


def _package_variants(value: str) -> frozenset[str]:
    noun = value.removeprefix("PACKAGE_").replace("_", " ").casefold()
    abbreviations = {
        "bag": {"bag", "bags", "bg"},
        "box": {"box", "boxes", "bx"},
        "bundle": {"bundle", "bundles", "bdl"},
        "case": {"case", "cases", "cas"},
        "carton": {"carton", "cartons", "ctn"},
        "crate": {"crate", "crates", "crt"},
        "drum": {"drum", "drums", "drm"},
        "package": {"package", "packages", "pkg"},
        "pallet": {"pallet", "pallets", "plt"},
        "piece": {"piece", "pieces", "pcs", "pces"},
        "roll": {"roll", "rolls", "rol"},
        "sack": {"sack", "sacks", "sak"},
        "intermediate bulk container": {
            "intermediate bulk container",
            "intermediate bulk containers",
            "ibc",
            "ibcs",
        },
    }
    values = abbreviations.get(noun, {noun, noun + "s"})
    return frozenset(_normalized_semantic(item) for item in values)


def _string_semantics_match(value: str, rendered: str) -> bool:
    expected = _normalized_semantic(value)
    actual = _normalized_semantic(rendered)
    if not expected or not actual:
        return False
    if expected == "negotiable":
        # A substring of NON-NEGOTIABLE is not positive negotiability evidence.
        # The operative order wording also occurs as "TO THE ORDER OF".
        if re.search(r"\b(?:NON[ -]?NEGOTIABLE|NOT\s+NEGOTIABLE)\b", rendered, re.I):
            return False
        return bool(
            re.search(
                r"\bNEGOTIABLE\b|\bORIGINAL\s+BILL\s+OF\s+LADING\b|\bTO\s+(?:THE\s+)?ORDER\b",
                rendered,
                re.I,
            )
        )
    if expected in actual:
        return True
    if value.startswith("PACKAGE_"):
        return package_category_surface_present(rendered, value) or any(
            variant in actual for variant in _package_variants(value)
        )
    variants = {
        "nonnegotiable": {
            "nonnegotiable",
            "notnegotiable",
            "expressbilloflading",
            "seawaybill",
            "expressrelease",
        },
        "freightcollect": {"freightcollect", "collect", "destinationcollect"},
        "collect": {"collect", "destinationcollect"},
        "prepaid": {"prepaid", "freightprepaid"},
    }
    return any(candidate in actual for candidate in variants.get(expected, set()))


def _measurement_unit_semantics_match(value: str, rendered: str) -> bool:
    actual = _normalized_semantic(rendered)
    variants = {
        "kilogram": ("kg", "kgs", "kilo", "kilos", "kilogram", "kilograms", "kgm"),
        "metric_tonne": ("mt", "mts", "tonne", "tonnes", "metrictonne"),
        "pound": ("lb", "lbs", "pound", "pounds"),
        "cubic_metre": ("m3", "cbm", "cubicmetre", "cubicmeter"),
        "celsius": ("c", "degc", "celsius"),
    }
    return any(token in actual for token in variants.get(value, ()))


def _dangerous_goods_semantics_match(value: str, rendered: str) -> bool:
    hazard_classes = {
        "EXPLOSIVES": "1",
        "GASES": "2",
        "FLAMMABLE_LIQUIDS": "3",
        "FLAMMABLE_SOLIDS": "4",
        "OXIDIZING_SUBSTANCES_AND_ORGANIC_PEROXIDES": "5",
        "TOXIC_AND_INFECTIOUS_SUBSTANCES": "6",
        "RADIOACTIVE_MATERIAL": "7",
        "CORROSIVE_SUBSTANCES": "8",
        "MISCELLANEOUS_DANGEROUS_SUBSTANCES_AND_ARTICLES": "9",
    }
    actual = _normalized_semantic(rendered)
    expected = _normalized_semantic(value)
    if expected in actual:
        return True
    hazard_class = hazard_classes.get(value)
    if hazard_class is None:
        return False
    return re.search(rf"(?:class|cl){hazard_class}(?:[0-9])?", actual) is not None


def _binding_observed_temperature(binding: SemanticBinding, target: Mapping[str, Any]) -> bool:
    owners = tuple(
        path.removesuffix(".typeDescription")
        for path in binding.target_paths
        if re.fullmatch(r"documentPatch\.containers\[\d+\]\.typeDescription", path)
    )
    return bool(owners) and all(
        isinstance(_resolve_path(target, owner), Mapping)
        and "temperatureSetpoint" in _resolve_path(target, owner)
        for owner in owners
    )


def _equipment_semantics_match(
    value: Mapping[str, Any], rendered: str, *, observed_temperature: bool = False
) -> bool:
    size = value.get("sizeCategory")
    equipment_type = value.get("typeCategory")
    if not isinstance(size, str) or not isinstance(equipment_type, str):
        return False
    # The expected label is not source evidence. Passing its thermal class as
    # temperature_present made dry GP text validate as refrigerated equipment.
    reviewed = review_source_equipment_surface(rendered, temperature_present=observed_temperature)
    return reviewed.size_category == size and reviewed.type_category == equipment_type


def _equipment_receipt_semantics_match(value: Sequence[Any], rendered: str) -> bool:
    containers = tuple(row for row in value if isinstance(row, Mapping))
    if len(containers) != len(value) or not containers:
        return False
    actual = _normalized_semantic(rendered)
    count = len(containers)
    if not (
        re.search(rf"(?:^|[^0-9]){count}\s*[xX]", rendered) is not None
        or actual.startswith(_normalized_semantic(_number_to_words(count)))
    ):
        return False
    semantic_values = {
        canonical_json_bytes(
            {
                "sizeCategory": row.get("sizeCategory"),
                "typeCategory": row.get("typeCategory"),
            }
        ): {
            "sizeCategory": row.get("sizeCategory"),
            "typeCategory": row.get("typeCategory"),
        }
        for row in containers
    }
    return all(_equipment_semantics_match(item, rendered) for item in semantic_values.values())


def _typed_semantic_equipment_output_matches(
    *,
    binding: SemanticBinding,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    output: BindingOutput,
) -> bool:
    """Validate a compiler-certified projection against its typed renderer receipt."""

    if not _has_semantic_equipment_values(binding, target):
        return False
    try:
        expected = _render_agent_target_binding(
            binding,
            source_target=source_target,
            target=target,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return output.replacements == expected.replacements and canonical_json_bytes(
        output.canonical_value
    ) == canonical_json_bytes(expected.canonical_value)


def _interrupted_target_semantics_match(
    *,
    binding: SemanticBinding,
    target_value: str,
    rendered: str,
    template: CertifiedSemanticTemplate,
    outputs: Mapping[str, BindingOutput],
) -> bool:
    if len(binding.occurrences) < 2:
        return False
    start = min(slot.byte_start for slot in binding.occurrences)
    end = max(slot.byte_end for slot in binding.occurrences)
    owned_slots = {slot.slot_id for slot in binding.occurrences}
    interruptions: list[tuple[int, str]] = []
    for other in template.bindings:
        for slot in other.occurrences:
            if (
                slot.slot_id not in owned_slots
                and start <= slot.byte_start
                and slot.byte_end <= end
            ):
                interruptions.append(
                    (slot.byte_start, outputs[other.logical_key].replacements[slot.slot_id])
                )
    if not interruptions:
        return False
    expected = _normalized_semantic(target_value)
    for _offset, replacement in sorted(interruptions):
        normalized = _normalized_semantic(replacement)
        if normalized and expected.count(normalized) == 1:
            expected = expected.replace(normalized, "", 1)
    return bool(expected) and expected == _normalized_semantic(rendered)


def _composite_package_quantity_matches(
    binding: SemanticBinding | None, target: Mapping[str, Any], path: str, rendered: str
) -> bool:
    """Prove the quantity next to its package category in a composite summary.

    Presence of an unrelated number is insufficient. For example, `20 PKGS
    (12 IBC TANKS + 8 PLTS)` supports 12 IBCs, not 20 IBCs.
    """
    allocation_match = re.fullmatch(
        r"(documentPatch\.cargoAllocationGroups\[\d+\])\.allocations\[0\]\.packageQuantity",
        path,
    )
    if allocation_match and binding is not None:
        group = _resolve_path(target, allocation_match.group(1))
        if len(group["allocations"]) != 1 or len(group["packageIds"]) != 1:
            return False
        # Only a sole allocation and its sole package have a proven equal count.
        # A shipment total must not stand in for one of several container rows.
        package_paths = [
            candidate
            for candidate in binding.target_paths
            if re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", candidate)
            and _resolve_path(target, candidate.removesuffix(".quantity"))["packageId"]
            == group["packageIds"][0]
        ]
        if len(package_paths) != 1 or _resolve_path(target, package_paths[0]) != _resolve_path(
            target, path
        ):
            return False
        path = package_paths[0]
    elif not re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", path):
        return False
    package = _resolve_path(target, path.removesuffix(".quantity"))
    category = package.get("typeCategory")
    description = package.get("typeDescription")
    if category is None and not isinstance(description, str):
        return False
    quantity = Decimal(str(_resolve_path(target, path)))
    variants = (
        _package_variants(category) if category is not None else {_normalized_semantic(description)}
    )
    for match in re.finditer(
        r"(?<![A-Za-z0-9])([0-9]+(?:[,.'][0-9]+)*)\s*([A-Za-z][A-Za-z \t()/-]*)",
        " ".join(rendered.split()),
    ):
        words = match.group(2).split()
        if not (
            category is not None and package_category_surface_present(match.group(2), category)
        ) and not any(
            _normalized_semantic(" ".join(words[:end])) in variants
            or (category is not None and _string_semantics_match(category, " ".join(words[:end])))
            for end in range(1, len(words) + 1)
        ):
            continue
        if quantity in _numeric_interpretations(match.group(1)):
            return True
    return False


def _dangerous_goods_outputs(case: PreparedCase) -> dict[str, BindingOutput]:
    if not case.dangerous_goods_facts:
        return {}
    return render_facts(
        source=case.source,
        template=case.template,
        target=case.target,
        facts=case.dangerous_goods_facts,
    )


def _target_binding_semantics_valid(
    *,
    case: PreparedCase,
    outputs: Mapping[str, BindingOutput],
    country_codes: Mapping[str, str] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    dg_outputs = _dangerous_goods_outputs(case)
    deterministic_target_modes = {
        "single_surface",
        "repeated_surface",
        "segmented_surface",
        "token_projected_surface",
        "normalized_projected_surface",
    }
    for binding in case.template.bindings:
        if binding.logical_key in dg_outputs:
            if outputs[binding.logical_key] != dg_outputs[binding.logical_key]:
                failures.append(f"{binding.logical_key}:dangerous-goods-fact-differs")
            continue
        if not binding.target_paths:
            continue
        output = outputs[binding.logical_key]
        composite_range = render_composite_range(
            binding, case.source_target, case.target, case.source
        )
        if composite_range is not None:
            if output.replacements != composite_range:
                failures.append(f"{binding.logical_key}:composite-range-differs")
            continue
        measurement = _render_target_measurement(binding, case)
        if measurement is not None:
            if output.replacements != measurement.replacements:
                failures.append(f"{binding.logical_key}:measurement-occurrence-differs")
            continue
        unchanged = _unchanged_target_output(binding, case.target)
        if (
            unchanged is not None
            and _binding_dependencies_unchanged(
                binding,
                source_target=case.source_target,
                target=case.target,
                constraints=case.template.coherence_constraints,
            )
            and output.replacements == unchanged.replacements
        ):
            continue
        if _typed_semantic_equipment_output_matches(
            binding=binding,
            source_target=case.source_target,
            target=case.target,
            output=output,
        ):
            continue
        if lexical_partitions.partition(binding) is not None:
            expected = _render_target_binding(
                binding,
                case.target,
                auxiliary_values=case.auxiliary_values,
                source=case.source,
                template=case.template,
            )
            if output != expected:
                failures.append(f"{binding.logical_key}:cargo-fragment-rendering-differs")
            continue
        if binding.realization.requires_agent and binding.target_paths:
            try:
                expected = _render_agent_target_binding(
                    binding,
                    source_target=case.source_target,
                    target=case.target,
                    source=case.source,
                    country_codes=country_codes,
                )
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if output.replacements != expected.replacements or canonical_json_bytes(
                    output.canonical_value
                ) != canonical_json_bytes(expected.canonical_value):
                    failures.append(f"{binding.logical_key}:typed-rendering-differs")
                continue
        if (
            binding.realization.mode in deterministic_target_modes
            and not _has_semantic_equipment_values(binding, case.target)
        ):
            expected = _render_target_binding(
                binding,
                case.target,
                auxiliary_values=case.auxiliary_values,
                source=case.source,
                template=case.template,
            )
            if output.replacements != expected.replacements or canonical_json_bytes(
                output.canonical_value
            ) != canonical_json_bytes(expected.canonical_value):
                failures.append(f"{binding.logical_key}:deterministic-rendering-differs")
            continue
        rendered = " ".join(output.replacements[slot.slot_id] for slot in binding.occurrences)
        for path in binding.target_paths:
            target_value = _binding_target_value(case.target, path)
            if isinstance(target_value, str):
                if binding.value_kind == "date":
                    try:
                        expected_date = date.fromisoformat(target_value)
                    except ValueError:
                        if not _string_semantics_match(target_value, rendered):
                            failures.append(f"{binding.logical_key}:{path}:text-not-rendered")
                        continue
                    if not any(
                        expected_date in _date_candidates(value)
                        for value in (*output.replacements.values(), rendered)
                    ):
                        failures.append(f"{binding.logical_key}:{path}:date-not-rendered")
                elif path.endswith(".unit") and target_value in {
                    "kilogram",
                    "metric_tonne",
                    "pound",
                    "cubic_metre",
                    "celsius",
                }:
                    if not _measurement_unit_semantics_match(target_value, rendered):
                        failures.append(f"{binding.logical_key}:{path}:unit-not-rendered")
                elif path.endswith(".hazardCategory"):
                    if not _dangerous_goods_semantics_match(target_value, rendered):
                        failures.append(f"{binding.logical_key}:{path}:hazard-not-rendered")
                elif not _string_semantics_match(
                    target_value, rendered
                ) and not _interrupted_target_semantics_match(
                    binding=binding,
                    target_value=target_value,
                    rendered=rendered,
                    template=case.template,
                    outputs=outputs,
                ):
                    failures.append(f"{binding.logical_key}:{path}:text-not-rendered")
            elif isinstance(target_value, (int, float)) and not isinstance(target_value, bool):
                expected_number = Decimal(str(target_value))
                if binding.derivation == "inclusive_range_cardinality":
                    observed_cardinalities = {
                        cardinality
                        for value in output.replacements.values()
                        for cardinality in inclusive_range_cardinalities(value)
                    }
                    if (
                        expected_number != expected_number.to_integral_value()
                        or int(expected_number) not in observed_cardinalities
                    ):
                        failures.append(f"{binding.logical_key}:{path}:range-cardinality-mismatch")
                elif binding.derivation == "number_to_words":
                    try:
                        matches = all(
                            expected_number in _numeric_interpretations(value)
                            or Decimal(_number_word_phrase(value)[0]) == expected_number
                            for value in output.replacements.values()
                        )
                    except ValueError:
                        matches = False
                    if not matches:
                        failures.append(f"{binding.logical_key}:{path}:number-words-mismatch")
                elif _composite_package_quantity_matches(binding, case.target, path, rendered):
                    continue
                elif not any(
                    expected_number in _numeric_surface_values(value)
                    for value in output.replacements.values()
                ):
                    failures.append(f"{binding.logical_key}:{path}:number-not-rendered")
            elif isinstance(target_value, Mapping) and set(target_value) == {
                "sizeCategory",
                "typeCategory",
            }:
                if not _equipment_semantics_match(
                    target_value,
                    rendered,
                    observed_temperature=_binding_observed_temperature(binding, case.target),
                ):
                    failures.append(f"{binding.logical_key}:{path}:equipment-not-rendered")
            elif isinstance(target_value, (Mapping, list)):
                if (
                    binding.value_kind == "equipment"
                    and isinstance(target_value, list)
                    and _equipment_receipt_semantics_match(target_value, rendered)
                ):
                    continue
                # Other collection/object paths on physical surfaces are declared derivations;
                # their scalar receipts are checked by the derivation implementation and proof.
                if binding.realization.mode != "deterministic_derivation":
                    failures.append(f"{binding.logical_key}:{path}:unsupported-composite-target")
            else:
                failures.append(f"{binding.logical_key}:{path}:unsupported-target-scalar")
    return not failures, tuple(failures)


def _source_relationships_valid(
    *, template: CertifiedSemanticTemplate, outputs: Mapping[str, BindingOutput]
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    active_relationships = _active_identifier_relationships(template)
    for binding in template.bindings:
        relationships = active_relationships[binding.logical_key]
        if not relationships:
            continue
        current = _alphanumeric(
            outputs[binding.logical_key].replacements[binding.occurrences[0].slot_id]
        ).casefold()
        for relationship in relationships:
            dependency = next(
                row
                for row in template.bindings
                if row.logical_key == relationship.dependency_binding
            )
            dependency_value = _alphanumeric(
                outputs[dependency.logical_key].replacements[dependency.occurrences[0].slot_id]
            ).casefold()
            if relationship.relationship == "embeds_exact_source_identifier":
                valid = dependency_value in current
            else:
                valid = current in dependency_value
            if not valid:
                failures.append(
                    f"{binding.logical_key}:{relationship.relationship}:"
                    f"{relationship.dependency_binding}"
                )
    return not failures, tuple(failures)


def _carrier_unchanged(*, case: PreparedCase, outputs: Mapping[str, BindingOutput]) -> bool:
    source_carrier = _resolve_path(case.source_target, "documentPatch.parties.carrier")
    target_carrier = _resolve_path(case.target, "documentPatch.parties.carrier")
    if source_carrier != target_carrier:
        return False
    for binding in case.template.bindings:
        if binding.render_mode != "carrier_static":
            continue
        output = outputs[binding.logical_key]
        if any(
            output.replacements[slot.slot_id] != slot.source_text for slot in binding.occurrences
        ):
            return False
    return True


def _usage_values(
    stage: ResidualStageReceipt,
) -> tuple[int, int, int, int, Decimal, Decimal | None]:
    usage = cast(Mapping[str, Any], stage.usage)
    reported = usage.get("providerReportedCostUsd")
    return (
        int(usage["requests"]),
        int(usage["inputTokens"]),
        int(usage["outputTokens"]),
        int(usage["reasoningTokens"]),
        Decimal(str(usage["estimatedCostUsd"])),
        Decimal(str(reported)) if reported is not None else None,
    )


def _failed_result(
    *,
    case: PreparedCase,
    plan: RenderPlan,
    stage: ResidualStageReceipt,
    status: Literal["provider_error", "host_rejected"],
    error_type: str,
    error_message: str,
    duration: float,
) -> DescendantCaseResult:
    requests, input_tokens, output_tokens, reasoning_tokens, estimated, reported = _usage_values(
        stage
    )
    deterministic_bindings = sum(row.runtime_route == "deterministic" for row in plan.routes)
    deterministic_slots = sum(
        len(row.slot_ids) for row in plan.routes if row.runtime_route == "deterministic"
    )
    return DescendantCaseResult.model_validate(
        {
            "schema_version": 1,
            "document_id": case.document_id,
            "source_document_id": case.source_document_id,
            "synthetic_document_id": case.target_receipt.synthetic_document_id,
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
            "target_origin": case.target_receipt.target_origin,
            "template_binding_count": len(case.template.bindings),
            "template_slot_count": len(case.template.byte_template.slots),
            "deterministic_binding_count": deterministic_bindings,
            "agent_binding_count": len(case.template.bindings) - deterministic_bindings,
            "deterministic_slot_count": deterministic_slots,
            "agent_slot_count": len(case.template.byte_template.slots) - deterministic_slots,
            "changed_target_leaf_count": case.target_receipt.changed_target_leaf_count,
            "changed_slot_count": 0,
            "unchanged_static_slot_count": 0,
            "carrier_unchanged": False,
            "exact_topology": True,
            "every_slot_bound_once": False,
            "exact_literal_regions": False,
            "page_markers_unchanged": False,
            "line_endings_preserved": False,
            "format_envelopes_valid": False,
            "target_binding_semantics_valid": False,
            "source_relationships_valid": False,
            "semantic_coherence_valid": False,
            "output_sha256": None,
            "source_bytes": len(case.source),
            "output_bytes": 0,
            "render_duration_seconds": duration,
            "provider_requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "estimated_cost_usd": estimated,
            "provider_reported_cost_usd": reported,
        }
    )


@dataclass(frozen=True, slots=True)
class ExecutedCase:
    result: DescendantCaseResult
    stage: ResidualStageReceipt
    plan: RenderPlan
    slot_bindings: Mapping[str, str] | None
    binding_outputs: Mapping[str, BindingOutput] | None
    rendered: bytes | None
    proof: TemplateRenderProof | None
    semantic_failures: tuple[str, ...]
    relationship_failures: tuple[str, ...]


def _validate_replay_case_inputs(
    *, prefix: Path, case: PreparedCase, reuses_residual_output: bool
) -> None:
    if read_regular_file_bytes(prefix / "source.txt") != case.source:
        raise ValueError(f"replay source bytes differ for {case.document_id}")
    expected_json: tuple[tuple[str, Any], ...] = (("source-target.json", case.source_target),)
    if reuses_residual_output:
        # A residual response was conditioned on the complete prepared target and receipt. Exact
        # identity is mandatory whenever even one of its slots is reused. If the current plan is
        # fully deterministic, the old response is provenance-only and cannot influence output.
        expected_json += (
            ("target.json", case.target),
            ("target-receipt.json", case.target_receipt.model_dump(mode="json")),
        )
    for name, expected in expected_json:
        actual = json.loads(read_regular_file_bytes(prefix / name))
        if actual != expected:
            raise ValueError(f"replay {name} differs for {case.document_id}")


def _replay_stage_path(*, prefix: Path, document_id: str) -> Path:
    """Resolve one committed provider stage, including an explicitly proven replay chain."""

    provider_stage = prefix / "agent-stage.json"
    replayed_stage = prefix / "source-agent-stage.json"
    present = tuple(path for path in (provider_stage, replayed_stage) if path.is_file())
    if len(present) != 1:
        raise ValueError(
            f"replay must contain exactly one agent stage for {document_id}; found {len(present)}"
        )
    stage_path = present[0]
    if stage_path == replayed_stage:
        receipt_path = prefix / "replay-receipt.json"
        receipt = ResidualReplayReceipt.model_validate_json(read_regular_file_bytes(receipt_path))
        if receipt.document_id != document_id:
            raise ValueError(f"replay-chain receipt document differs for {document_id}")
        if sha256_file(stage_path) != receipt.source_agent_stage_sha256:
            raise ValueError(f"replay-chain agent-stage hash differs for {document_id}")
    return stage_path


def _load_replayed_stage(
    *,
    replay_root: Path,
    replay_commit_sha256: str,
    replay_transaction_sha256: str,
    expected_system_prompt_sha256: str,
    case: PreparedCase,
    plan: RenderPlan,
) -> tuple[Mapping[str, str], ResidualStageReceipt, ResidualReplayReceipt]:
    prefix = replay_root / "cases" / case.document_id
    expected_slots = {
        slot.slot_id for binding in plan.residual_bindings for slot in binding.occurrences
    }
    _validate_replay_case_inputs(
        prefix=prefix,
        case=case,
        reuses_residual_output=bool(expected_slots),
    )

    stage_path = _replay_stage_path(prefix=prefix, document_id=case.document_id)
    source_stage = ResidualStageReceipt.model_validate_json(read_regular_file_bytes(stage_path))
    if source_stage.document_id != case.document_id:
        raise ValueError(f"replay agent-stage document differs for {case.document_id}")
    if source_stage.system_prompt_sha256 != expected_system_prompt_sha256:
        raise ValueError(f"replay system prompt differs for {case.document_id}")

    if expected_slots:
        if source_stage.status not in {"success", "manual_review"} or not isinstance(
            source_stage.output, Mapping
        ):
            raise ValueError(f"replay has no successful residual output for {case.document_id}")
        raw_output = dict(source_stage.output)
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_output.items()
        ):
            raise ValueError(
                f"replay residual output is not string-to-string for {case.document_id}"
            )
        missing = expected_slots - set(raw_output)
        if missing:
            raise ValueError(
                f"replay residual output is missing current slots for {case.document_id}: "
                + ", ".join(sorted(missing))
            )
        projected = {key: cast(str, raw_output[key]) for key in expected_slots}
    else:
        if source_stage.status not in {"success", "not_required", "manual_review"}:
            raise ValueError(f"replay source stage failed for {case.document_id}")
        raw_output = dict(source_stage.output) if isinstance(source_stage.output, Mapping) else {}
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_output.items()
        ):
            raise ValueError(
                f"replay residual output is not string-to-string for {case.document_id}"
            )
        projected = {}

    requests, *_usage = _usage_values(source_stage)
    expected_requests = int(source_stage.status == "success")
    if source_stage.batch_provenance is not None:
        # The strict receipt validator proves the output projection and usage
        # allocation. Only the first member owns the shared physical request.
        expected_requests = int(source_stage.batch_provenance["member"] == "s0")
    if requests != expected_requests:
        raise ValueError(
            f"replay stage request count differs from its status for {case.document_id}"
        )
    dropped = tuple(sorted(set(raw_output) - expected_slots))
    receipt = ResidualReplayReceipt.model_validate(
        {
            "schema_version": 1,
            "document_id": case.document_id,
            "source_run_commit_sha256": replay_commit_sha256,
            "source_run_transaction_sha256": replay_transaction_sha256,
            "source_agent_stage_sha256": sha256_file(stage_path),
            "source_status": source_stage.status,
            "source_output_slot_count": len(raw_output),
            "replayed_output_slot_count": len(projected),
            "dropped_output_slot_count": len(dropped),
            "dropped_slot_ids": dropped,
            "new_provider_requests": 0,
        }
    )
    effective_stage = (
        source_stage
        if expected_slots
        else _not_required_stage(
            document_id=case.document_id,
            system_prompt_sha256=expected_system_prompt_sha256,
        )
    )
    return projected, effective_stage, receipt


def _validate_projected_context(
    template: CertifiedSemanticTemplate, outputs: Mapping[str, BindingOutput]
) -> None:
    """A fragment owned by another binding is not an immutable literal.

    Prove ownership from whole source occurrences; never infer it from arbitrary
    numbers or from proximity. Unowned literal regions remain byte-preserved by
    the template renderer. Shared mutable fragments need a dependency contract.
    """
    from .projected_context import declared_edges

    owners: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for binding in template.bindings:
        for slot in binding.occurrences:
            tokens = tuple(row[0] for row in _token_spans(slot.source_text))
            owners.setdefault(tokens, []).append((binding.logical_key, slot.slot_id))
    for binding in template.bindings:
        declared = {edge.owner_key: edge for edge in declared_edges(binding, template.bindings)}
        for start, end, fragment in fixed_projection_ranges(binding):
            scoped = {
                key for key, edge in declared.items() if (edge.start, edge.end) == (start, end)
            }
            for key, slot_id in owners.get(fragment, ()):
                if key == binding.logical_key:
                    continue
                if scoped and key not in scoped:
                    # Equal country text in another party/cargo is not the
                    # owner of this explicitly declared component.
                    continue
                actual = tuple(row[0] for row in _token_spans(outputs[key].replacements[slot_id]))
                if key in declared:
                    expected = tuple(
                        row[0]
                        for row in _token_spans(_scalar_surface(outputs[key].canonical_value))
                    )
                    edge = declared[key]
                    whole = tuple(
                        row[0]
                        for row in _token_spans(
                            _scalar_surface(outputs[binding.logical_key].canonical_value)
                        )
                    )
                    source_size = len(
                        _token_spans(
                            _scalar_surface(binding.realization.target_values[0].source_value)
                        )
                    )
                    matches = [
                        i
                        for i in range(len(whole) - len(expected) + 1)
                        if whole[i : i + len(expected)] == expected
                        and (edge.start != 0 or i == 0)
                        and (edge.end != source_size or i + len(expected) == len(whole))
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "projected target omits or contradicts its sampled context owner"
                        )
                else:
                    expected = fragment
                if actual != expected:
                    raise ValueError(
                        "projected target context changed in another binding: "
                        f"{binding.logical_key}: {fragment}, owner {key}; "
                        "cross-binding dependency requires compilation review"
                    )


def _materialize_case(
    *,
    case: PreparedCase,
    plan: RenderPlan,
    raw_output: Mapping[str, str],
    stage: ResidualStageReceipt,
    country_codes: Mapping[str, str],
) -> ExecutedCase:
    started = time.perf_counter()
    try:
        _require_frozen_target(case)
        if stage.manual_review is not None:
            stage.manual_review.validate_context(
                sample_id=case.document_id,
                source_document_id=case.source_document_id,
                source=case.source,
                template=case.template.model_dump(mode="json"),
                target=case.target,
                output=raw_output,
            )
        outputs = dict(plan.deterministic_outputs)
        outputs.update(_postprocess_residual_outputs(case=case, plan=plan, raw_output=raw_output))
        _reconcile_identifier_relationships(template=case.template, outputs=outputs)
        _render_derivations(
            case=case,
            outputs=outputs,
            country_codes=country_codes,
            residual_bindings=plan.residual_bindings,
        )
        if set(outputs) != {binding.logical_key for binding in case.template.bindings}:
            raise ValueError("binding outputs do not cover the semantic template exactly")
        for binding in case.template.bindings:
            if _source_container_identifier(binding) is not None and not validate_container_number(
                _alphanumeric(str(outputs[binding.logical_key].canonical_value)).upper()
            ):
                raise ValueError(
                    f"source-only container identifier is not ISO6346-valid: {binding.logical_key}"
                )
        for binding in case.template.bindings:
            country = plan.independent_phone_countries.get(binding.logical_key)
            if country is None or _explicit_unknown_placeholder(binding):
                continue
            output = outputs[binding.logical_key]
            if not isinstance(output.canonical_value, str):
                raise ValueError("independent phone must have a canonical string")
            contact_values.validate_phone(output.canonical_value, country_code=country)
            for replacement in output.replacements.values():
                contact_values.validate_phone(replacement, country_code=country)
                if _alphanumeric(replacement) != _alphanumeric(output.canonical_value):
                    raise ValueError("rendered independent phone differs from its canonical number")
        owned = entity_members_by_binding(case.template.auxiliary_semantic_plan)
        for binding in case.template.bindings:
            if (
                binding.value_kind != "phone"
                or binding.target_paths
                or binding.dependency_paths
                or binding.derivation
                or binding.group_kind == "carrier"
                or binding.logical_key in owned
                or _explicit_unknown_placeholder(binding)
            ):
                continue
            output = outputs[binding.logical_key]
            if not isinstance(output.canonical_value, str):
                raise ValueError("standalone phone requires a canonical string")
            contact_values.validate_phone(output.canonical_value)
            for replacement in output.replacements.values():
                contact_values.validate_phone(replacement)
                if _alphanumeric(replacement) != _alphanumeric(output.canonical_value):
                    raise ValueError("rendered standalone phone differs from its canonical number")
        for binding in case.template.bindings:
            if (
                binding.value_kind != "email"
                or binding.target_paths
                or _explicit_unknown_placeholder(binding)
            ):
                continue
            output = outputs[binding.logical_key]
            if not isinstance(output.canonical_value, str):
                raise ValueError("source-only email requires one canonical mailbox")
            contact_values.validate_mailbox(output.canonical_value)
            for replacement in output.replacements.values():
                mailbox = "".join(replacement.split())
                contact_values.validate_mailbox(mailbox)
                if mailbox.casefold() != output.canonical_value.casefold():
                    raise ValueError("rendered email differs from its canonical mailbox")
        validate_auxiliary_render(
            plan=case.template.auxiliary_semantic_plan,
            bindings=case.template.bindings,
            outputs=outputs,
            target=case.target,
            country_codes=country_codes,
            target_projection_values=projected_auxiliary_values(case.template, case.target),
        )
        entity_identifiers.validate(
            *entity_identifiers.contracts(
                case.template, case.source_target, case.target, country_codes
            ),
            outputs,
        )
        validate_render_coherence(
            bindings=case.template.bindings,
            constraints=case.template.coherence_constraints,
            source_target=case.source_target,
            target=case.target,
            outputs=outputs,
        )
        slot_bindings = {
            slot_id: replacement
            for output in outputs.values()
            for slot_id, replacement in output.replacements.items()
        }
        if len(slot_bindings) != len(case.template.byte_template.slots):
            raise ValueError("slot bindings do not cover every template slot exactly once")
        semantic_valid, semantic_failures = _target_binding_semantics_valid(
            case=case, outputs=outputs, country_codes=country_codes
        )
        if not semantic_valid:
            raise ValueError("target semantics failed: " + ", ".join(semantic_failures))
        relationships_valid, relationship_failures = _source_relationships_valid(
            template=case.template, outputs=outputs
        )
        if not relationships_valid:
            raise ValueError("source relationships failed: " + ", ".join(relationship_failures))
        if case.customs_presentation is not None:
            rendered, proof = case.customs_presentation.render(case.source, slot_bindings)
        else:
            rendered, proof = render_compiled_template(
                source=case.source,
                template=case.template.byte_template,
                bindings=slot_bindings,
            )
        validate_repeated_agent_party_pages(
            source=case.source,
            rendered=rendered,
            source_target=case.source_target,
            target=case.target,
            bindings=case.template.bindings,
        )
        validate_rendered_party_boundaries(case.target, rendered.decode("utf-8"))
        _validate_projected_context(case.template, outputs)
        if case.dangerous_goods_facts:
            validate_un_references(
                rendered.decode("utf-8"), {f.record.un_number for f in case.dangerous_goods_facts}
            )
        count_aliases.validate(rendered.decode("utf-8"))
        _validate_description_volume_units(case.target, rendered.decode("utf-8"))
        validate_seal_realization(
            target=case.target,
            bindings=case.template.bindings,
            slot_values=slot_bindings,
        )
        validate_unbound_lexical_surfaces(
            target=case.target,
            binding_paths={p for b in case.template.bindings for p in b.target_paths},
            rendered=rendered.decode("utf-8"),
            party_surfaces=party_owned_surfaces(
                case.template.bindings,
                slot_bindings,
                case.template.auxiliary_semantic_plan.entities,
            ),
            country_codes=country_codes,
        )
        if set(re.findall(rb"\bPACKAGE_[A-Z_]+\b", rendered)) - set(
            re.findall(rb"\bPACKAGE_[A-Z_]+\b", case.source)
        ):
            raise ValueError("training schema package enum leaked into document text")
        carrier_valid = _carrier_unchanged(case=case, outputs=outputs)
        if not carrier_valid:
            raise ValueError("carrier-bound label or carrier slot changed")
        if printed_topology_mismatches(case.topology_reference_target, case.target):
            raise ValueError("prepared target topology changed after preflight")
        requests, input_tokens, output_tokens, reasoning_tokens, estimated, reported = (
            _usage_values(stage)
        )
        deterministic_bindings = sum(row.runtime_route == "deterministic" for row in plan.routes)
        deterministic_slots = sum(
            len(row.slot_ids) for row in plan.routes if row.runtime_route == "deterministic"
        )
        changed_slots = sum(
            slot_bindings[slot.slot_id] != slot.source_text
            for slot in case.template.byte_template.slots
        )
        static_slots = sum(
            len(binding.occurrences)
            for binding in case.template.bindings
            if binding.realization.mode == "static"
        )
        result = DescendantCaseResult.model_validate(
            {
                "schema_version": 1,
                "document_id": case.document_id,
                "source_document_id": case.source_document_id,
                "synthetic_document_id": case.target_receipt.synthetic_document_id,
                "status": "passed",
                "error_type": None,
                "error_message": None,
                "target_origin": case.target_receipt.target_origin,
                "template_binding_count": len(case.template.bindings),
                "template_slot_count": len(case.template.byte_template.slots),
                "deterministic_binding_count": deterministic_bindings,
                "agent_binding_count": len(case.template.bindings) - deterministic_bindings,
                "deterministic_slot_count": deterministic_slots,
                "agent_slot_count": len(case.template.byte_template.slots) - deterministic_slots,
                "changed_target_leaf_count": case.target_receipt.changed_target_leaf_count,
                "changed_slot_count": changed_slots,
                "unchanged_static_slot_count": static_slots,
                "customs_caption_count": (
                    len(case.customs_presentation.replacements)
                    if case.customs_presentation is not None
                    else 0
                ),
                "carrier_unchanged": carrier_valid,
                "exact_topology": True,
                "every_slot_bound_once": proof.every_slot_bound_once,
                "exact_literal_regions": proof.exact_literal_regions,
                "page_markers_unchanged": proof.page_markers_unchanged,
                "line_endings_preserved": proof.line_endings_preserved,
                "format_envelopes_valid": proof.format_envelopes_valid,
                "target_binding_semantics_valid": semantic_valid,
                "source_relationships_valid": relationships_valid,
                "semantic_coherence_valid": True,
                "output_sha256": proof.output_sha256,
                "source_bytes": len(case.source),
                "output_bytes": len(rendered),
                "render_duration_seconds": time.perf_counter() - started,
                "provider_requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "estimated_cost_usd": estimated,
                "provider_reported_cost_usd": reported,
            }
        )
        return ExecutedCase(
            result,
            stage,
            plan,
            slot_bindings,
            outputs,
            rendered,
            proof,
            semantic_failures,
            relationship_failures,
        )
    except Exception as error:
        result = _failed_result(
            case=case,
            plan=plan,
            stage=stage,
            status="host_rejected",
            error_type=type(error).__name__,
            error_message=str(error),
            duration=time.perf_counter() - started,
        )
        return ExecutedCase(result, stage, plan, None, None, None, None, (), ())


async def _execute_case(
    *,
    case: PreparedCase,
    plan: RenderPlan,
    model: Model,
    provider: ProviderConfig,
    system_prompt: str,
    limiter: asyncio.Semaphore,
    country_codes: Mapping[str, str],
) -> ExecutedCase:
    raw_output, stage = await _call_residual_agent(
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        case=case,
        plan=plan,
        limiter=limiter,
    )
    if raw_output is None:
        result = _failed_result(
            case=case,
            plan=plan,
            stage=stage,
            status="provider_error",
            error_type=stage.error_type or "ProviderError",
            error_message=stage.error_message or "provider returned no residual output",
            duration=stage.duration_seconds,
        )
        return ExecutedCase(result, stage, plan, None, None, None, None, (), ())
    return _materialize_case(
        case=case,
        plan=plan,
        raw_output=raw_output,
        stage=stage,
        country_codes=country_codes,
    )


def _preflight_payload(
    cases: Sequence[PreparedCase],
    plans: Sequence[RenderPlan],
    *,
    country_codes: Mapping[str, str],
    system_prompt_sha256: str,
) -> dict[str, Any]:
    rows = []
    provider_free_failures: list[str] = []
    for case, plan in zip(cases, plans, strict=True):
        if not plan.residual_bindings:
            execution = _materialize_case(
                case=case,
                plan=plan,
                raw_output={},
                stage=_not_required_stage(
                    document_id=case.document_id,
                    system_prompt_sha256=system_prompt_sha256,
                ),
                country_codes=country_codes,
            )
            if execution.result.status != "passed":
                provider_free_failures.append(
                    f"{case.document_id} from {case.source_document_id}: "
                    f"{execution.result.error_type}: {execution.result.error_message}"
                )
        payload = _residual_payload(case=case, plan=plan)
        schema_bytes = (
            len(
                canonical_json_bytes(
                    _residual_output_type(plan.residual_bindings).model_json_schema()
                )
            )
            if plan.residual_bindings
            else 0
        )
        deterministic_bindings = sum(row.runtime_route == "deterministic" for row in plan.routes)
        deterministic_slots = sum(
            len(row.slot_ids) for row in plan.routes if row.runtime_route == "deterministic"
        )
        rows.append(
            {
                "documentId": case.document_id,
                "sourceDocumentId": case.source_document_id,
                "targetOrigin": case.target_receipt.target_origin,
                "targetSchemaVersion": case.target["schemaVersion"],
                "targetChanges": case.target_receipt.changed_target_leaf_count,
                "targetAdaptations": len(case.target_receipt.compatibility_adaptations),
                "bindings": len(case.template.bindings),
                "slots": len(case.template.byte_template.slots),
                "deterministicBindings": deterministic_bindings,
                "agentBindings": len(case.template.bindings) - deterministic_bindings,
                "deterministicSlots": deterministic_slots,
                "agentSlots": len(case.template.byte_template.slots) - deterministic_slots,
                "residualPromptBytes": len(canonical_json_bytes(payload)),
                "residualSchemaBytes": schema_bytes,
                "plannedProviderRequests": int(bool(plan.residual_bindings)),
            }
        )
    if provider_free_failures:
        raise ValueError(
            "provider-free preflight materialization failed for "
            f"{len(provider_free_failures)} document(s):\n" + "\n".join(provider_free_failures)
        )
    return {
        "schemaVersion": 1,
        "status": "ready",
        "documents": len(cases),
        "exactTopologyDocuments": len(cases),
        "carrierBoundDocuments": len(cases),
        "completeTargetDocuments": len(cases),
        "templateBindings": sum(len(case.template.bindings) for case in cases),
        "templateSlots": sum(len(case.template.byte_template.slots) for case in cases),
        "deterministicBindings": sum(row["deterministicBindings"] for row in rows),
        "agentBindings": sum(row["agentBindings"] for row in rows),
        "deterministicSlots": sum(row["deterministicSlots"] for row in rows),
        "agentSlots": sum(row["agentSlots"] for row in rows),
        "plannedProviderRequests": sum(row["plannedProviderRequests"] for row in rows),
        "providerFreeValidatedDocuments": sum(not plan.residual_bindings for plan in plans),
        "maximumProviderRequestsPerDocument": max(row["plannedProviderRequests"] for row in rows),
        "targetCompatibilityAdaptations": sum(row["targetAdaptations"] for row in rows),
        "residualPromptBytes": sum(row["residualPromptBytes"] for row in rows),
        "residualSchemaBytes": sum(row["residualSchemaBytes"] for row in rows),
        "rows": rows,
    }


def _build_render_plans(
    cases: Sequence[PreparedCase],
    *,
    seed: int,
    country_codes: Mapping[str, str],
) -> tuple[RenderPlan, ...]:
    plans: list[RenderPlan] = []
    for case in cases:
        try:
            plans.append(
                _build_initial_plan(
                    case,
                    seed=seed,
                    country_codes=country_codes,
                )
            )
        except ValueError as error:
            raise ValueError(
                f"render-plan preflight failed for sample {case.document_id} "
                f"from template {case.source_document_id}: {error}"
            ) from error
    return tuple(plans)


def preflight_descendants(config_path: Path) -> dict[str, Any]:
    project_root = project_root_from_config(config_path)
    config = load_descendant_config(config_path)
    prompt_path = resolve_input(project_root, config.prompts.residual_renderer.path)
    if sha256_file(prompt_path) != config.prompts.residual_renderer.sha256:
        raise ValueError("residual renderer prompt hash differs")
    cases = _load_cases(project_root=project_root, config=config)
    country_path = resolve_input(project_root, config.inputs.iso3166_snapshot.path)
    countries = _country_code_map(country_path)
    plans = _build_render_plans(
        cases,
        seed=config.workflow.controlled_target_seed,
        country_codes=countries,
    )
    payload = _preflight_payload(
        cases,
        plans,
        country_codes=countries,
        system_prompt_sha256=config.prompts.residual_renderer.sha256,
    )
    replay_pin = config.inputs.residual_replay_run
    if replay_pin is not None:
        replay_root = _validate_committed_run(
            project_root, replay_pin, workers=config.workflow.max_concurrent_requests
        )
        replay_failures: list[str] = []
        for case, plan in zip(cases, plans, strict=True):
            try:
                _load_replayed_stage(
                    replay_root=replay_root,
                    replay_commit_sha256=replay_pin.commit_sha256,
                    replay_transaction_sha256=replay_pin.transaction_sha256,
                    expected_system_prompt_sha256=config.prompts.residual_renderer.sha256,
                    case=case,
                    plan=plan,
                )
            except ValueError as error:
                replay_failures.append(f"{case.document_id}: {type(error).__name__}: {error}")
        if replay_failures:
            raise ValueError(
                f"replay validation failed for {len(replay_failures)} documents:\n"
                + "\n".join(replay_failures)
            )
        payload["replayValidatedDocuments"] = len(cases)
        payload["replayRunCommitSha256"] = replay_pin.commit_sha256
    else:
        payload["replayValidatedDocuments"] = 0
        payload["replayRunCommitSha256"] = None
    return payload


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _artifact_inventory(stage: StagedArtifactRun) -> tuple[str, ...]:
    return tuple(
        path.relative_to(stage.stage_root).as_posix()
        for path in sorted(stage.stage_root.rglob("*"))
        if path.is_file() and path.name not in {"_TRANSACTION.json", "_COMMIT.json"}
    )


def _unified_diff(source: bytes, rendered: bytes, document_id: str) -> bytes:
    import difflib

    before = source.decode("utf-8").splitlines(keepends=True)
    after = rendered.decode("utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before,
            after,
            fromfile=f"{document_id}/source.txt",
            tofile=f"{document_id}/rendered.txt",
        )
    ).encode("utf-8")


def _summary(
    results: Sequence[DescendantCaseResult],
    preflight: Mapping[str, Any],
    *,
    execution_mode: Literal["provider", "offline_replay"],
    training_records_published: bool,
) -> dict[str, Any]:
    total_estimated = sum((row.estimated_cost_usd for row in results), Decimal(0))
    reported_values = [
        row.provider_reported_cost_usd
        for row in results
        if row.provider_reported_cost_usd is not None
    ]
    total_reported = (
        sum(cast(Sequence[Decimal], reported_values), Decimal(0)) if reported_values else None
    )
    calls = sum(row.provider_requests for row in results)
    passed = sum(row.status == "passed" for row in results)
    deterministic_slots = sum(row.deterministic_slot_count for row in results)
    total_slots = sum(row.template_slot_count for row in results)
    changed_slots = sum(row.changed_slot_count for row in results)
    summary = {
        "schemaVersion": 1,
        "executionMode": execution_mode,
        "status": "passed" if passed == len(results) else "failed",
        "documents": len(results),
        "passedDocuments": passed,
        "providerErrorDocuments": sum(row.status == "provider_error" for row in results),
        "hostRejectedDocuments": sum(row.status == "host_rejected" for row in results),
        "providerRequests": calls,
        "providerRequestsPerDocument": calls / len(results),
        "maximumProviderRequestsPerDocument": max(row.provider_requests for row in results),
        "inputTokens": sum(row.input_tokens for row in results),
        "outputTokens": sum(row.output_tokens for row in results),
        "reasoningTokens": sum(row.reasoning_tokens for row in results),
        "estimatedCostUsd": str(total_estimated),
        "providerReportedCostUsd": str(total_reported) if total_reported is not None else None,
        "estimatedCostPerDocumentUsd": str(total_estimated / len(results)),
        "templateBindings": sum(row.template_binding_count for row in results),
        "templateSlots": total_slots,
        "deterministicBindings": sum(row.deterministic_binding_count for row in results),
        "agentBindings": sum(row.agent_binding_count for row in results),
        "deterministicSlots": deterministic_slots,
        "agentSlots": total_slots - deterministic_slots,
        "deterministicSlotFraction": deterministic_slots / total_slots,
        "changedSlots": changed_slots,
        "changedSlotFraction": changed_slots / total_slots,
        "carrierUnchangedDocuments": sum(row.carrier_unchanged for row in results),
        "exactTopologyDocuments": sum(row.exact_topology for row in results),
        "exactLiteralRegionDocuments": sum(row.exact_literal_regions for row in results),
        "pageMarkerPreservedDocuments": sum(row.page_markers_unchanged for row in results),
        "lineEndingPreservedDocuments": sum(row.line_endings_preserved for row in results),
        "formatValidDocuments": sum(row.format_envelopes_valid for row in results),
        "targetSemanticValidDocuments": sum(row.target_binding_semantics_valid for row in results),
        "sourceRelationshipValidDocuments": sum(row.source_relationships_valid for row in results),
        "semanticCoherenceValidDocuments": sum(row.semantic_coherence_valid for row in results),
        "targetCompatibilityAdaptations": preflight["targetCompatibilityAdaptations"],
        "trainingRecordsPublished": training_records_published,
        "trainingRecords": len(results) if training_records_published else 0,
    }
    summary.update(
        {
            "responseRequestsUsed": calls,
            "newProviderRequests": calls if execution_mode == "provider" else 0,
            "upstreamProviderRequests": calls if execution_mode == "offline_replay" else 0,
            "additionalEstimatedCostUsd": (
                str(total_estimated) if execution_mode == "provider" else "0"
            ),
            "upstreamEstimatedCostUsd": (
                str(total_estimated) if execution_mode == "offline_replay" else "0"
            ),
        }
    )
    return summary


def _report(summary: Mapping[str, Any]) -> str:
    if summary["executionMode"] == "offline_replay":
        request_line = (
            f"- Upstream responses reused: **{summary['responseRequestsUsed']}**; "
            f"new provider requests: **{summary['newProviderRequests']}**"
        )
        cost_line = (
            f"- Upstream provider cost: **${summary['upstreamEstimatedCostUsd']}**; "
            f"additional replay cost: **${summary['additionalEstimatedCostUsd']}**"
        )
    else:
        request_line = (
            f"- Provider requests: **{summary['providerRequests']}** "
            f"({summary['providerRequestsPerDocument']:.3f} per document)"
        )
        cost_line = f"- Estimated provider cost: **${summary['estimatedCostUsd']}**"
    return "\n".join(
        (
            "# Carrier-bound compiled-template descendant run",
            "",
            f"- Status: **{summary['status']}**",
            f"- Passed: **{summary['passedDocuments']}/{summary['documents']}**",
            request_line,
            "- Deterministic slots: "
            f"**{summary['deterministicSlots']}/{summary['templateSlots']}** "
            f"({summary['deterministicSlotFraction']:.2%})",
            f"- Changed slots: **{summary['changedSlots']}/{summary['templateSlots']}** "
            f"({summary['changedSlotFraction']:.2%})",
            cost_line,
            "- Carrier unchanged: "
            f"**{summary['carrierUnchangedDocuments']}/{summary['documents']}**",
            "- Exact literal regions: "
            f"**{summary['exactLiteralRegionDocuments']}/{summary['documents']}**",
            f"- Exact topology: **{summary['exactTopologyDocuments']}/{summary['documents']}**",
            "- Semantic coherence: "
            f"**{summary['semanticCoherenceValidDocuments']}/{summary['documents']}**",
            "- Training records published: "
            f"**{'yes' if summary['trainingRecordsPublished'] else 'no'}**",
            "",
        )
    )


def _training_dataset(
    *,
    cases: Sequence[PreparedCase],
    executions: Sequence[ExecutedCase],
    config: DescendantConfig,
    country_codes: Mapping[str, str],
    package_contract: ReviewedPackageContract | None = None,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Build the complete publishable dataset after every descendant passes."""

    records: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for case, execution in zip(cases, executions, strict=True):
        _require_frozen_target(case)
        require_complete_variation(
            case.topology_reference_target, case.target, bindings=case.template.bindings
        )
        if execution.result.status != "passed" or execution.rendered is None:
            raise ValueError("training publication requires every descendant to pass")
        synthetic_id = case.target_receipt.synthetic_document_id
        training_target, address_edits = project_training_target(
            case.target, country_codes=country_codes
        )
        training_target, package_edits = project_reviewed_package_target(
            source_document_id=case.source_document_id,
            source_target=case.source_target,
            target=training_target,
            contract=package_contract,
        )
        if address_edits or package_edits:
            canonical = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
                document_id=synthetic_id, target=training_target
            )
            if canonical != training_target:
                raise ValueError("address projection changed canonical target shape")
        records.append(
            {
                "documentId": synthetic_id,
                "sourceDocumentId": case.source_document_id,
                "samplePlanId": case.document_id,
                "splitGroupId": case.source_document_id,
                "joinedRawText": execution.rendered.decode("utf-8"),
                "joinedRawTextSha256": sha256_bytes(execution.rendered),
                "target": training_target,
            }
        )
        lineage.append(
            {
                "documentId": synthetic_id,
                "sourceDocumentId": case.source_document_id,
                "samplePlanId": case.document_id,
                "splitGroupId": case.source_document_id,
                "templateRunCommitSha256": config.inputs.template_run.commit_sha256,
                "templateRunTransactionSha256": config.inputs.template_run.transaction_sha256,
                "samplePlanRunCommitSha256": (
                    config.inputs.sample_plan_run.commit_sha256
                    if config.inputs.sample_plan_run is not None
                    else None
                ),
                "samplePlanRunTransactionSha256": (
                    config.inputs.sample_plan_run.transaction_sha256
                    if config.inputs.sample_plan_run is not None
                    else None
                ),
                "samplePlanSha256": (
                    config.inputs.sample_plan.sha256
                    if config.inputs.sample_plan is not None
                    else None
                ),
                "syntheticTargetRunCommitSha256": (
                    config.inputs.synthetic_target_run.commit_sha256
                    if config.inputs.synthetic_target_run is not None
                    else None
                ),
                "syntheticTargetRunTransactionSha256": (
                    config.inputs.synthetic_target_run.transaction_sha256
                    if config.inputs.synthetic_target_run is not None
                    else None
                ),
                "targetGeneration": config.workflow.target_generation,
                "templateSha256": sha256_bytes(
                    canonical_json_bytes(case.template.model_dump(mode="json"))
                ),
                "sourceTextSha256": sha256_bytes(case.source),
                "joinedRawTextSha256": sha256_bytes(execution.rendered),
                "preparedTargetSha256": case.target_receipt.prepared_target_sha256,
                "trainingTargetSha256": sha256_bytes(canonical_json_bytes(training_target)),
                "projectedAddressCount": len(address_edits),
                "projectedPackageGroupCount": sum(
                    bool(row["metadataPackageIds"])
                    for row in package_edits
                    if "metadataPackageIds" in row
                ),
                "taskPackageContractSha256": (
                    config.inputs.task_package_contract.sha256
                    if config.inputs.task_package_contract is not None
                    else None
                ),
                "targetReceiptSha256": sha256_bytes(
                    canonical_json_bytes(case.target_receipt.model_dump(mode="json"))
                ),
                "renderResultSha256": sha256_bytes(
                    canonical_json_bytes(execution.result.model_dump(mode="json"))
                ),
                "carrier": case.template.carrier.canonical_name,
            }
        )
    records_bytes = _jsonl_bytes(records)
    lineage_bytes = _jsonl_bytes(lineage)
    manifest = {
        "schemaVersion": 1,
        "records": len(records),
        "files": [
            {
                "kind": "training_records",
                "path": "records.jsonl",
                "records": len(records),
                "bytes": len(records_bytes),
                "sha256": sha256_bytes(records_bytes),
            },
            {
                "kind": "lineage",
                "path": "lineage.jsonl",
                "records": len(lineage),
                "bytes": len(lineage_bytes),
                "sha256": sha256_bytes(lineage_bytes),
            },
        ],
        "splitGroupField": "splitGroupId",
        "targetSchemaVersions": sorted({case.target["schemaVersion"] for case in cases}),
        "carrierBound": True,
    }
    return records_bytes, lineage_bytes, manifest


def _publish_prepared_case(
    stage: StagedArtifactRun,
    case: PreparedCase,
    plan: RenderPlan,
) -> None:
    prefix = f"cases/{case.document_id}"
    stage.publish_bytes(f"{prefix}/source.txt", case.source)
    stage.publish_json(f"{prefix}/source-target.json", case.source_target)
    stage.publish_json(f"{prefix}/target.json", case.target)
    stage.publish_json(f"{prefix}/auxiliary-values.json", case.auxiliary_values)
    if case.customs_presentation is not None:
        stage.publish_json(
            f"{prefix}/customs-presentation.json", case.customs_presentation.evidence
        )
        stage.publish_json(
            f"{prefix}/render-byte-template.json",
            case.customs_presentation.template.model_dump(mode="json"),
        )
    stage.publish_json(
        f"{prefix}/numeric-auxiliary.json",
        {k: v.model_dump(mode="json") for k, v in case.numeric_auxiliary.items()},
    )
    stage.publish_json(
        f"{prefix}/dangerous-goods-facts.json",
        [v.model_dump(mode="json") for v in case.dangerous_goods_facts],
    )
    stage.publish_json(f"{prefix}/target-receipt.json", case.target_receipt.model_dump(mode="json"))
    stage.publish_json(
        f"{prefix}/route-plan.json",
        [row.model_dump(mode="json") for row in plan.routes],
    )


def _publish_executed_case(
    stage: StagedArtifactRun,
    case: PreparedCase,
    execution: ExecutedCase,
    replay_receipt: ResidualReplayReceipt | None,
) -> None:
    prefix = f"cases/{case.document_id}"
    if replay_receipt is None:
        stage.publish_json(f"{prefix}/agent-stage.json", execution.stage.model_dump(mode="json"))
    else:
        stage.publish_json(
            f"{prefix}/source-agent-stage.json", execution.stage.model_dump(mode="json")
        )
        stage.publish_json(
            f"{prefix}/replay-receipt.json",
            replay_receipt.model_dump(mode="json"),
        )
    stage.publish_json(f"{prefix}/result.json", execution.result.model_dump(mode="json"))
    if execution.slot_bindings is not None:
        stage.publish_json(f"{prefix}/slot-bindings.json", execution.slot_bindings)
    if execution.binding_outputs is not None:
        stage.publish_json(
            f"{prefix}/binding-outputs.json",
            {
                key: {
                    "canonicalValue": value.canonical_value,
                    "replacements": dict(value.replacements),
                }
                for key, value in execution.binding_outputs.items()
            },
        )
    if execution.rendered is not None and execution.proof is not None:
        stage.publish_bytes(f"{prefix}/rendered.txt", execution.rendered)
        stage.publish_bytes(
            f"{prefix}/diff.patch",
            _unified_diff(case.source, execution.rendered, case.document_id),
        )
        stage.publish_json(f"{prefix}/render-proof.json", execution.proof.model_dump(mode="json"))


async def run_descendants(config_path: Path) -> Path:
    started = time.perf_counter()
    project_root = project_root_from_config(config_path)
    config = load_descendant_config(config_path)
    prompt_path = resolve_input(project_root, config.prompts.residual_renderer.path)
    if sha256_file(prompt_path) != config.prompts.residual_renderer.sha256:
        raise ValueError("residual renderer prompt hash differs")
    system_prompt = prompt_path.read_text(encoding="utf-8")
    package_pin = config.inputs.task_package_contract
    package_contract = (
        load_reviewed_package_contract(
            resolve_input(project_root, package_pin.path),
            expected_sha256=package_pin.sha256,
            catalog_commit_sha256=config.inputs.template_run.commit_sha256,
        )
        if package_pin is not None
        else None
    )
    cases = _load_cases(project_root=project_root, config=config)
    for case in cases:
        projected, _ = project_reviewed_package_target(
            source_document_id=case.source_document_id,
            source_target=case.source_target,
            target=case.target,
            contract=package_contract,
        )
        BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
            document_id=case.target_receipt.synthetic_document_id, target=projected
        )
    country_path = resolve_input(project_root, config.inputs.iso3166_snapshot.path)
    countries = _country_code_map(country_path)
    plans = _build_render_plans(
        cases,
        seed=config.workflow.controlled_target_seed,
        country_codes=countries,
    )
    preflight = _preflight_payload(
        cases,
        plans,
        country_codes=countries,
        system_prompt_sha256=config.prompts.residual_renderer.sha256,
    )
    replay_pin = config.inputs.residual_replay_run
    execution_mode: Literal["provider", "offline_replay"]
    replay_root: Path | None = None
    if replay_pin is None:
        execution_mode = "provider"
        if preflight["plannedProviderRequests"] and not config.workflow.provider_launch_authorized:
            raise ValueError("provider launch is disabled by descendant workflow configuration")
    else:
        execution_mode = "offline_replay"
        if config.workflow.provider_launch_authorized:
            raise ValueError("offline replay requires provider_launch_authorized=false")
        replay_root = _validate_committed_run(
            project_root, replay_pin, workers=config.workflow.max_concurrent_requests
        )

    transaction = {
        "schemaVersion": 1,
        "runName": config.run_name,
        "configSha256": sha256_file(config_path),
        "templateRunCommitSha256": config.inputs.template_run.commit_sha256,
        "samplePlanRunCommitSha256": (
            config.inputs.sample_plan_run.commit_sha256
            if config.inputs.sample_plan_run is not None
            else None
        ),
        "samplePlanRunTransactionSha256": (
            config.inputs.sample_plan_run.transaction_sha256
            if config.inputs.sample_plan_run is not None
            else None
        ),
        "samplePlanSha256": (
            config.inputs.sample_plan.sha256 if config.inputs.sample_plan is not None else None
        ),
        "syntheticTargetRunCommitSha256": (
            config.inputs.synthetic_target_run.commit_sha256
            if config.inputs.synthetic_target_run is not None
            else None
        ),
        "syntheticTargetsSha256": (
            config.inputs.synthetic_targets.sha256
            if config.inputs.synthetic_targets is not None
            else None
        ),
        "targetGeneration": config.workflow.target_generation,
        "iso3166Sha256": config.inputs.iso3166_snapshot.sha256,
        "taskPackageContractSha256": package_pin.sha256 if package_pin is not None else None,
        "customsProgramRegistrySha256": (
            config.inputs.customs_program_registry.sha256
            if config.inputs.customs_program_registry is not None
            else None
        ),
        "promptSha256": config.prompts.residual_renderer.sha256,
        "executionMode": execution_mode,
        "residualReplayRunCommitSha256": (
            replay_pin.commit_sha256 if replay_pin is not None else None
        ),
        "residualReplayRunTransactionSha256": (
            replay_pin.transaction_sha256 if replay_pin is not None else None
        ),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "modelsSha256": sha256_file(_MODEL_PATH),
        "runtimeComponentSha256": {
            path.name: sha256_file(path)
            for path in (
                _IMPLEMENTATION_PATH,
                _MODEL_PATH,
                _SEMANTIC_PLAN_PATH,
                _SYNTHETIC_VALUES_PATH,
                _COHERENCE_PATH,
                _LATEST_TARGET_PATH,
            )
        },
        "documentIds": [case.document_id for case in cases],
        "sourceDocumentIds": [case.source_document_id for case in cases],
        "preparedTargetSha256": [case.target_receipt.prepared_target_sha256 for case in cases],
        "runtime": {
            "providerKind": config.provider.kind,
            "model": config.provider.model,
            "reasoningEffort": config.provider.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
    }
    output_parent = Path(config.output_dir)
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
        validation_workers=config.workflow.max_concurrent_requests,
    )
    if stage.completed:
        return stage.final_root
    stage.recover_interrupted_temporary_files()
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    stage.publish_bytes("prompts/residual-renderer.md", read_regular_file_bytes(prompt_path))
    stage.publish_json("transaction.json", transaction)
    stage.publish_json("preflight.json", preflight)
    for start in range(0, len(cases), config.workflow.max_concurrent_requests):
        await asyncio.gather(
            *(
                asyncio.to_thread(_publish_prepared_case, stage, cases[index], plans[index])
                for index in range(
                    start, min(start + config.workflow.max_concurrent_requests, len(cases))
                )
            )
        )

    replay_receipts: tuple[ResidualReplayReceipt, ...] | None = None
    executions: Sequence[ExecutedCase]
    if replay_root is None:
        model = _provider_model(
            project_root=project_root,
            environment_file=config.environment_file,
            provider=config.provider,
        )
        limiter = asyncio.Semaphore(config.workflow.max_concurrent_requests)
        pending_executions: list[ExecutedCase | None] = [None] * len(cases)
        next_case = 0
        next_case_lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal next_case
            while True:
                async with next_case_lock:
                    if next_case >= len(cases):
                        return
                    index = next_case
                    next_case += 1
                pending_executions[index] = await _execute_case(
                    case=cases[index],
                    plan=plans[index],
                    model=model,
                    provider=config.provider,
                    system_prompt=system_prompt,
                    limiter=limiter,
                    country_codes=countries,
                )

        await asyncio.gather(
            *(worker() for _ in range(min(config.workflow.max_concurrent_requests, len(cases))))
        )
        if any(row is None for row in pending_executions):
            raise RuntimeError("provider worker pool did not execute every descendant case")
        executions = tuple(cast(ExecutedCase, row) for row in pending_executions)
    else:
        replay_rows: list[ResidualReplayReceipt] = []
        replayed_executions: list[ExecutedCase] = []
        assert replay_pin is not None
        for case, plan in zip(cases, plans, strict=True):
            raw_output, source_stage, replay_receipt = _load_replayed_stage(
                replay_root=replay_root,
                replay_commit_sha256=replay_pin.commit_sha256,
                replay_transaction_sha256=replay_pin.transaction_sha256,
                expected_system_prompt_sha256=config.prompts.residual_renderer.sha256,
                case=case,
                plan=plan,
            )
            replay_rows.append(replay_receipt)
            replayed_executions.append(
                _materialize_case(
                    case=case,
                    plan=plan,
                    raw_output=raw_output,
                    stage=source_stage,
                    country_codes=countries,
                )
            )
        replay_receipts = tuple(replay_rows)
        executions = tuple(replayed_executions)
    for start in range(0, len(cases), config.workflow.max_concurrent_requests):
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    _publish_executed_case,
                    stage,
                    cases[index],
                    executions[index],
                    replay_receipts[index] if replay_receipts is not None else None,
                )
                for index in range(
                    start, min(start + config.workflow.max_concurrent_requests, len(cases))
                )
            )
        )
    results = tuple(execution.result for execution in executions)
    stage.publish_bytes(
        "results.jsonl",
        _jsonl_bytes([row.model_dump(mode="json") for row in results]),
    )
    training_records_published = all(row.status == "passed" for row in results)
    if training_records_published:
        records_bytes, lineage_bytes, dataset_manifest = _training_dataset(
            cases=cases,
            executions=executions,
            config=config,
            country_codes=countries,
            package_contract=package_contract,
        )
        stage.publish_bytes("dataset/records.jsonl", records_bytes)
        stage.publish_bytes("dataset/lineage.jsonl", lineage_bytes)
        stage.publish_json("dataset/manifest.json", dataset_manifest)
    summary = _summary(
        results,
        preflight,
        execution_mode=execution_mode,
        training_records_published=training_records_published,
    )
    summary["wallTimeSeconds"] = time.perf_counter() - started
    stage.publish_json("summary.json", summary)
    stage.publish_bytes("REPORT.md", _report(summary).encode("utf-8"))
    stage.commit(
        expected_artifacts=_artifact_inventory(stage),
        metadata={
            "status": cast(JsonValue, summary["status"]),
            "documents": len(results),
            "passedDocuments": cast(int, summary["passedDocuments"]),
            "providerRequests": cast(int, summary["providerRequests"]),
            "newProviderRequests": cast(int, summary["newProviderRequests"]),
            "executionMode": execution_mode,
            "estimatedCostUsd": cast(str, summary["estimatedCostUsd"]),
            "trainingRecordsPublished": training_records_published,
        },
    )
    return stage.final_root
