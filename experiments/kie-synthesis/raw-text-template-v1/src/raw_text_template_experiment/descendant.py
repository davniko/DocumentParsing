from __future__ import annotations

import asyncio
import difflib
import json
import re
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_container_number,
    generate_from_surface_pattern,
    shift_document_dates,
    surface_pattern,
    validate_container_number,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    load_provider_key,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.raw_text_template import (
    CompiledRawTextTemplate,
    TemplateRenderProof,
    TemplateSlot,
    printed_topology_mismatches,
    render_compiled_template,
)
from document_ocr.synthesis.rendering import render_number_surface
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.usage_receipt import LinguisticUsageReceipt
from document_ocr.training.tasks import get_training_task
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, create_model
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from .coherence import (
    coherence_dependency_paths,
    inclusive_range_cardinalities,
    validate_render_coherence,
    whole_inclusive_range_surface,
)
from .descendant_models import (
    BindingRoute,
    DescendantCaseResult,
    DescendantConfig,
    PinnedCommittedRun,
    PreparedTargetReceipt,
    ResidualReplayReceipt,
    ResidualStageReceipt,
    TargetAdaptation,
)
from .models import (
    CertifiedSemanticTemplate,
    CoherenceConstraint,
    OpenRouterProviderConfig,
    ProviderConfig,
    SemanticBinding,
)
from .pipeline import project_root_from_config, resolve_input
from .synthetic_values import DeterministicValueFactory

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_MODEL_PATH = Path(__file__).with_name("descendant_models.py").resolve(strict=True)
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
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%y",
    "%m-%d-%y",
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
    source: bytes
    source_target: dict[str, Any]
    target: dict[str, Any]
    template: CertifiedSemanticTemplate
    target_receipt: PreparedTargetReceipt


@dataclass(frozen=True, slots=True)
class BindingOutput:
    replacements: Mapping[str, str]
    canonical_value: JsonValue


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


def load_descendant_config(path: Path) -> DescendantConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return DescendantConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


def _validate_committed_run(project_root: Path, configured: PinnedCommittedRun) -> Path:
    root = (project_root / configured.path).resolve(strict=True)
    if project_root not in root.parents or root.is_symlink() or not root.is_dir():
        raise ValueError(f"configured run is not a regular project directory: {configured.path}")
    commit_path = root / "_COMMIT.json"
    if sha256_file(commit_path) != configured.commit_sha256:
        raise ValueError(f"committed-run receipt differs: {configured.path}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    ).validate_committed_run()
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


def _resolve_path(target: Mapping[str, Any], path: str) -> Any:
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


def _path_exists(target: Mapping[str, Any], path: str) -> bool:
    try:
        _resolve_path(target, path)
    except ValueError:
        return False
    return True


def _set_path(target: dict[str, Any], path: str, replacement: Any) -> None:
    parts = _path_parts(path)
    value: Any = target
    for part in parts[:-1]:
        value = value[part]
    value[parts[-1]] = deepcopy(replacement)


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


def _restore_carrier(
    *, source_target: Mapping[str, Any], target: dict[str, Any]
) -> TargetAdaptation | None:
    source_parties = cast(Mapping[str, Any], source_target["documentPatch"]).get("parties")
    target_parties = cast(dict[str, Any], target["documentPatch"]).get("parties")
    if not isinstance(source_parties, Mapping) or not isinstance(target_parties, dict):
        raise ValueError("carrier-bound target lacks a parties object")
    source_carrier = source_parties.get("carrier")
    if not isinstance(source_carrier, Mapping) or not source_carrier.get("name"):
        raise ValueError("carrier-bound source label lacks a carrier object")
    proposed = deepcopy(target_parties.get("carrier"))
    if proposed == source_carrier:
        return None
    target_parties["carrier"] = deepcopy(dict(source_carrier))
    return TargetAdaptation.model_validate(
        {
            "target_path": "documentPatch.parties.carrier",
            "reason": "carrier-bound templates retain the complete source carrier object",
            "source_value": dict(source_carrier),
            "proposed_value": proposed,
            "adapted_value": dict(source_carrier),
        }
    )


def _safe_identifier_path(path: str) -> bool:
    return bool(
        path.endswith("billOfLadingNumber")
        or path.endswith("voyageNumber")
        or path.endswith("bookingNumber")
        or re.search(r"\.containerNumber$", path)
        or re.search(r"\.sealNumbers\[[0-9]+\]$", path)
        or re.search(r"\.forwardingAndExportReferences\[[0-9]+\]$", path)
    )


def _controlled_target(
    *,
    source_target: Mapping[str, Any],
    template: CertifiedSemanticTemplate,
    seed: int,
) -> dict[str, Any]:
    target = deepcopy(dict(source_target))
    stream = DeterministicStream(
        seed, "carrier-bound-descendant-controlled-v1", template.document_id
    )
    generated_by_source: dict[tuple[str, str], str] = {}
    paths = sorted(
        {
            path
            for binding in template.bindings
            if binding.value_kind == "identifier"
            for path in binding.target_paths
            if _safe_identifier_path(path)
        }
    )
    for path in paths:
        source_value = _resolve_path(source_target, path)
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(f"controlled identifier path is not a string: {path}")
        family = "container" if path.endswith("containerNumber") else "generic"
        key = (family, source_value)
        generated = generated_by_source.get(key)
        if generated is None:
            if family == "container" and validate_container_number(source_value):
                generated = generate_container_number(
                    owner_and_category=source_value[:4],
                    stream=stream.derive(path),
                    excluded={source_value},
                )
            else:
                generated = generate_from_surface_pattern(
                    pattern=surface_pattern(source_value),
                    stream=stream.derive(path),
                    additional_excluded=source_value,
                )
            generated_by_source[key] = generated
        _set_path(target, path, generated)

    source_patch = cast(Mapping[str, Any], source_target["documentPatch"])
    issue_raw = source_patch.get("issueDate")
    shipped_raw = source_patch.get("shippedOnBoardDate")
    issue = date.fromisoformat(issue_raw) if isinstance(issue_raw, str) else None
    shipped = date.fromisoformat(shipped_raw) if isinstance(shipped_raw, str) else None
    if issue is not None or shipped is not None:
        new_issue, new_shipped = shift_document_dates(
            issue_date=issue,
            shipped_on_board_date=shipped,
            minimum=date(2020, 1, 1),
            maximum=date(2030, 12, 31),
            stream=stream.derive("document-dates"),
        )
        target_patch = cast(dict[str, Any], target["documentPatch"])
        if new_issue is not None:
            target_patch["issueDate"] = new_issue.isoformat()
        if new_shipped is not None:
            target_patch["shippedOnBoardDate"] = new_shipped.isoformat()
    _validate_canonical_target(target)
    return target


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
    words = tuple(candidate.split())
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
        else:
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
    if len(observed_value) == len(source_value):
        return target_value
    indices = _unique_subsequence_indices(source_value, observed_value)
    return "".join(target_value[index] for index in indices)


def _scalar_surface(value: Any) -> str:
    if (
        isinstance(value, bool)
        or value is None
        or not isinstance(value, (str, int, float, Decimal))
    ):
        raise ValueError(f"target value is not a renderable scalar: {value!r}")
    return str(value)


def _token_spans(value: str) -> tuple[tuple[str, int, int], ...]:
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


def _render_whole_slot(
    *, slot: TemplateSlot, binding: SemanticBinding, old_value: Any, new_value: Any
) -> str:
    realization = next(row for row in binding.realization.slots if row.slot_id == slot.slot_id)
    if binding.realization.adapter == "date":
        return _render_date_surface(slot.source_text, cast(str, old_value), cast(str, new_value))
    if binding.realization.adapter == "numeric":
        return render_number_surface(
            slot.source_text,
            cast(int | float, old_value),
            cast(int | float, new_value),
        )
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
    weights = tuple(max(1, len(_TOKEN.findall(slot.source_text))) for slot in slots)
    return tuple(" ".join(words) for words in _allocate_words(chunks, weights))


def _render_target_binding(binding: SemanticBinding, target: Mapping[str, Any]) -> BindingOutput:
    target_values = tuple(_binding_target_value(target, path) for path in binding.target_paths)
    if not target_values:
        raise ValueError("target binding has no target values")
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
    elif mode == "segmented_surface":
        surfaces = _partition_target_surface(_scalar_surface(new_value), slots)
        for slot, surface in zip(slots, surfaces, strict=True):
            replacements[slot.slot_id] = _layout_like_source(slot.source_text, surface)
    elif mode == "token_projected_surface":
        target_surface = _scalar_surface(new_value)
        target_tokens = _token_spans(target_surface)
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


def _randomize_numeric_surface(source: str, stream: DeterministicStream) -> str:
    digits = [index for index, character in enumerate(source) if character.isdigit()]
    if not digits:
        raise ValueError("numeric auxiliary has no digits")
    output = list(source)
    for ordinal, index in enumerate(digits):
        source_digit = output[index]
        for attempt in range(20):
            candidate = str(stream.derive(str(ordinal)).randbelow(10, counter=attempt))
            if candidate != source_digit and (
                ordinal > 0 or source_digit == "0" or candidate != "0"
            ):
                output[index] = candidate
                break
        else:
            raise RuntimeError("failed to change numeric auxiliary digit")
    return "".join(output)


def _render_direct_auxiliary(
    binding: SemanticBinding, stream: DeterministicStream
) -> BindingOutput:
    first = binding.occurrences[0]
    policy = first.render_policy
    if policy == "opaque_identifier" or binding.value_kind in {"email", "phone"}:
        generated = generate_from_surface_pattern(
            pattern=surface_pattern(first.source_text),
            stream=stream.derive(binding.logical_key),
            additional_excluded=first.source_text,
        )
        canonical = _alphanumeric(generated)
        replacements = {
            slot.slot_id: _shape_alphanumeric_like_source(slot.source_text, canonical)
            for slot in binding.occurrences
        }
        return BindingOutput(replacements=replacements, canonical_value=generated)
    if policy == "numeric_surface":
        generated = _randomize_numeric_surface(
            first.source_text, stream.derive(binding.logical_key)
        )
        replacements = {
            slot.slot_id: _shape_alphanumeric_like_source(
                slot.source_text, _alphanumeric(generated)
            )
            for slot in binding.occurrences
        }
        return BindingOutput(replacements=replacements, canonical_value=generated.strip())
    if policy == "date_surface":
        old = _parse_auxiliary_date(first.source_text)
        offset = 31 + stream.derive(binding.logical_key).randbelow(334)
        new = old + timedelta(days=offset)
        replacements = {
            slot.slot_id: _render_date_surface(slot.source_text, old.isoformat(), new.isoformat())
            for slot in binding.occurrences
        }
        return BindingOutput(replacements=replacements, canonical_value=new.isoformat())
    raise ValueError(f"no deterministic auxiliary renderer for {policy}")


def _render_text_candidate(binding: SemanticBinding, candidate: str) -> BindingOutput:
    if not candidate.strip():
        raise ValueError("typed auxiliary candidate is empty")
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        if slot.render_policy in {"opaque_identifier", "numeric_surface"}:
            replacement = _shape_alphanumeric_like_source(slot.source_text, candidate)
        else:
            replacement = _layout_like_source(slot.source_text, candidate)
        replacements[slot.slot_id] = replacement
    return BindingOutput(replacements=replacements, canonical_value=candidate.strip())


def _render_pattern_date_auxiliary(
    binding: SemanticBinding, stream: DeterministicStream
) -> BindingOutput:
    """Render an arbitrary valid date when the source date itself is ambiguous or malformed.

    A source-only date has no canonical value to preserve.  Its separator/month-name grammar and
    field width are the reusable facts.  Numeric ambiguity (DD/MM versus MM/DD) therefore does not
    require an agent: both interpretations share the same observable surface grammar.
    """

    source = binding.occurrences[0].source_text
    chosen = date(2021, 1, 1) + timedelta(days=stream.derive(binding.logical_key).randbelow(3650))
    compact = source.strip()
    candidates: list[str] = []
    numeric_match = re.fullmatch(
        r"(?P<day>[0-9]{1,2})(?P<sep1>[/.-])(?P<month>[0-9]{1,2})"
        r"(?P<sep2>[/.-])(?P<year>[0-9]{2,4})",
        compact,
    )
    if numeric_match is not None:
        year_width = len(numeric_match.group("year"))
        candidates.append(
            f"{chosen.day:0{len(numeric_match.group('day'))}d}"
            f"{numeric_match.group('sep1')}"
            f"{chosen.month:0{len(numeric_match.group('month'))}d}"
            f"{numeric_match.group('sep2')}"
            f"{chosen.year % (10**year_width):0{year_width}d}"
        )
    month_match = re.fullmatch(
        r"(?P<day>[0-9]{1,2})(?P<sep>[-./ ]?)(?P<month>[A-Za-z]{3,9})"
        r"(?P=sep)(?P<year>[0-9]{2,4})",
        compact,
    )
    if month_match is not None:
        month = chosen.strftime("%B" if len(month_match.group("month")) > 3 else "%b")
        month = _case_like(month_match.group("month"), month)
        year_width = len(month_match.group("year"))
        candidates.append(
            f"{chosen.day:0{len(month_match.group('day'))}d}"
            f"{month_match.group('sep')}{month}{month_match.group('sep')}"
            f"{chosen.year % (10**year_width):0{year_width}d}"
        )
    if not candidates:
        raise ValueError(f"source-only date has no supported observable grammar: {source!r}")
    candidate = candidates[0]
    if len(_alphanumeric(candidate)) != len(_alphanumeric(compact)):
        raise ValueError("generated source-only date changes the certified alphanumeric width")
    return _render_text_candidate(binding, candidate)


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
    if binding.value_kind == "date":
        return _render_pattern_date_auxiliary(binding, stream)

    if binding.value_kind == "identifier" and first.render_policy == "opaque_identifier":
        generated = generate_from_surface_pattern(
            pattern=surface_pattern(first.source_text),
            stream=stream.derive(binding.logical_key),
            additional_excluded=first.source_text,
        )
        return _render_text_candidate(binding, generated)

    vocabulary_kinds = {
        "decimal_measurement",
        "equipment",
        "package",
        "dangerous_goods",
        "commercial_text",
        "legal_text",
        "operational_text",
        "other_text",
    }
    if binding.value_kind not in vocabulary_kinds:
        raise ValueError(f"no typed deterministic auxiliary generator for {binding.value_kind}")
    if binding.value_kind == "equipment" and binding.group_kind == "transport":
        raise ValueError("source-only transport equipment may be a vessel identity")

    # Units, movement modes, status words, and operational/legal clauses are controlled
    # vocabularies rather than private identities.  Preserve their semantic wording, while
    # deterministically changing any embedded shipment-specific number.
    if any(character.isdigit() for character in first.source_text):
        generated = _randomize_numeric_surface(
            first.source_text, stream.derive(binding.logical_key)
        )
        if all(
            len(_alphanumeric(slot.source_text)) == len(_alphanumeric(generated))
            for slot in binding.occurrences
        ):
            return _render_text_candidate(binding, generated)
        raise ValueError("repeated numeric vocabulary surfaces have incompatible widths")
    return _preserved_source_output(binding)


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
        return _render_date_surface(raw, old_iso, new_iso)
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
    candidates = _PACKAGE_SURFACES.get(value)
    if candidates is None:
        raise ValueError(f"unsupported package category: {value}")
    optional_plural = re.fullmatch(
        r"(?P<leading>\s*)(?P<noun>[A-Za-z]+)\((?P<suffix>ES|S)\)(?P<trailing>\s*)",
        source,
        re.IGNORECASE,
    )
    if optional_plural is not None:
        noun = value.removeprefix("PACKAGE_").replace("_", " ")
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
    strings = tuple(value for value in values if isinstance(value, str))
    if not strings:
        raise ValueError("composite target has no textual surface")
    longest = max(strings, key=lambda value: len(_normalized_semantic(value)))
    if all(
        not isinstance(value, str) or _normalized_semantic(value) in _normalized_semantic(longest)
        for value in values
    ):
        return longest
    if binding.value_kind == "package" and any("PACK" in value.upper() for value in strings):
        return longest
    if binding.value_kind in {"address", "location", "other_text"}:
        unique = tuple(dict.fromkeys(value.strip() for value in strings if value.strip()))
        return ", ".join(unique)
    raise ValueError("multiple target values have no deterministic textual composition")


def _render_agent_target_binding(
    binding: SemanticBinding,
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
) -> BindingOutput:
    """Execute agent-labelled mappings whose target-to-surface transform is fully typed."""

    values = tuple(_binding_target_value(target, path) for path in binding.target_paths)
    if not values:
        raise ValueError("agent target binding has no target values")
    source_values = tuple(row.source_value for row in binding.realization.target_values)
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
        binding.value_kind == "identifier"
        and len(values) == len(source_values) == 1
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
        return BindingOutput(replacements=replacements, canonical_value=values[0])
    if binding.value_kind == "equipment" and binding.target_paths == ("documentPatch.containers",):
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
    if binding.value_kind in {"email", "phone"}:
        if not _alphanumeric(first.source_text):
            return False, "contact surface has no replaceable alphanumeric positions"
        length = len(_alphanumeric(first.source_text))
        if any(len(_alphanumeric(slot.source_text)) != length for slot in slots):
            return False, "repeated contact surfaces have unequal alphanumeric lengths"
        return True, "typed format-preserving contact generator is available"
    if first.render_policy == "date_surface":
        try:
            parsed = _parse_auxiliary_date(first.source_text)
            for slot in slots:
                if _parse_auxiliary_date(slot.source_text) != parsed:
                    return False, "repeated date surfaces do not encode one source date"
                _render_date_surface(
                    slot.source_text,
                    parsed.isoformat(),
                    date(2031, 9, 23).isoformat(),
                )
        except ValueError as error:
            return False, f"date surface lacks one unambiguous render style: {error}"
        return True, "typed format-preserving date generator is available"
    if first.render_policy == "numeric_surface":
        if not any(character.isdigit() for character in first.source_text):
            return False, "numeric surface has no numeric digits"
        words = {token.casefold() for token in _TOKEN.findall(first.source_text)}
        if words & _NUMBER_WORDS:
            return False, "numeric surface contains a coupled number-word expression"
        return True, "typed format-preserving numeric generator is available"
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


def _solve_identifier_relationships(
    *,
    template: CertifiedSemanticTemplate,
    outputs: dict[str, BindingOutput],
    stream: DeterministicStream,
) -> None:
    """Solve exact identifier-containment constraints at character-position level."""

    bindings = {binding.logical_key: binding for binding in template.bindings}
    related_keys = {
        key
        for binding in template.bindings
        for key in (
            binding.logical_key,
            *(row.dependency_binding for row in binding.source_relationships),
        )
        if binding.source_relationships
    }
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
        for relationship in binding.source_relationships:
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

    fixed: dict[tuple[str, int], str] = {}
    for key in related_keys:
        binding = bindings[key]
        if not binding.target_paths:
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
        generated[root] = fixed.get(
            root,
            alphabet[
                stream.derive("relationship:" + "|".join(sorted(related_keys))).randbelow(
                    len(alphabet), counter=ordinal
                )
            ],
        )

    for key in sorted(related_keys):
        binding = bindings[key]
        if binding.target_paths:
            continue
        candidate = "".join(
            generated[find((key, index))] for index in range(len(source_values[key]))
        )
        if candidate.casefold() == source_values[key].casefold():
            raise ValueError(f"identifier relationship solver retained a source identifier: {key}")
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
    ]
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
        values.extend(
            (
                f"{length}RF",
                f"{length} RF",
                f"{length} REEFER",
                f"{length} HIGH CUBE REEFER",
                f"{length} 9 6 REEFER",
            )
        )

    values.extend(
        candidate + " CONTAINER" for candidate in tuple(values) if "CONTAINER" not in candidate
    )

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
    value = _semantic_equipment_binding_value(binding, target)
    if value is None:
        raise ValueError("binding does not resolve to one semantic equipment value")
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        if _equipment_semantics_match(value, slot.source_text):
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


def _render_equipment_receipt_binding(
    binding: SemanticBinding,
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
) -> BindingOutput:
    if not binding.target_paths:
        return _preserved_source_output(binding)
    if binding.target_paths != ("documentPatch.containers",):
        raise ValueError("equipment receipt does not depend on the complete container list")
    source_containers = _resolve_path(source_target, "documentPatch.containers")
    target_containers = _resolve_path(target, "documentPatch.containers")
    if (
        not isinstance(source_containers, Sequence)
        or isinstance(source_containers, (str, bytes))
        or not isinstance(target_containers, Sequence)
        or isinstance(target_containers, (str, bytes))
        or not target_containers
    ):
        raise ValueError("equipment receipt container dependencies are invalid")
    semantic_values = tuple(
        {
            "sizeCategory": row.get("sizeCategory"),
            "typeCategory": row.get("typeCategory"),
        }
        for row in target_containers
        if isinstance(row, Mapping)
        and isinstance(row.get("sizeCategory"), str)
        and isinstance(row.get("typeCategory"), str)
    )
    unique_semantics = {canonical_json_bytes(value): value for value in semantic_values}
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        rendered = slot.source_text
        old_count = len(source_containers)
        new_count = len(target_containers)
        if old_count != new_count:
            rendered, numeric_replacements = re.subn(
                rf"(?<![0-9]){old_count}(?![0-9])",
                str(new_count),
                rendered,
                count=1,
            )
            if not numeric_replacements:
                old_words = _number_to_words(old_count)
                if re.search(rf"(?i)\b{re.escape(old_words)}\b", rendered) is None:
                    raise ValueError("equipment receipt has no replaceable source count")
                rendered = re.sub(
                    rf"(?i)\b{re.escape(old_words)}\b",
                    _case_like(old_words, _number_to_words(new_count)),
                    rendered,
                    count=1,
                )
        if len(unique_semantics) == 1:
            semantic = next(iter(unique_semantics.values()))
            new_length = _equipment_length(semantic)
            size_match = re.search(r"(?<![0-9])(20|40|45)(?![0-9])", rendered)
            if size_match is not None:
                rendered = (
                    rendered[: size_match.start()] + new_length + rendered[size_match.end() :]
                )
            code_match = re.search(r"(?i)(?<![A-Z])(GP|HC|HQ|HR|RF|RE|DR|DV|DC)(?![A-Z])", rendered)
            if code_match is not None:
                code = _case_like(code_match.group(0), _equipment_code(semantic))
                rendered = rendered[: code_match.start()] + code + rendered[code_match.end() :]
        replacements[slot.slot_id] = rendered
    return BindingOutput(
        replacements=replacements,
        canonical_value=cast(JsonValue, list(target_containers)),
    )


def _preserved_source_output(binding: SemanticBinding) -> BindingOutput:
    return BindingOutput(
        replacements={slot.slot_id: slot.source_text for slot in binding.occurrences},
        canonical_value=binding.occurrences[0].source_text.strip(),
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
    # Coherence is a pre-routing safety gate. A stale target-backed range or a compiler-marked
    # review case must fail before residual planning can initialize a provider request.
    validate_render_coherence(
        bindings=case.template.bindings,
        constraints=case.template.coherence_constraints,
        source_target=case.source_target,
        target=case.target,
        outputs=None,
    )
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
    routes: list[BindingRoute] = []
    outputs: dict[str, BindingOutput] = {}
    residual: list[SemanticBinding] = []
    stream = DeterministicStream(seed, "carrier-bound-descendant-render-v1", case.document_id)
    values = DeterministicValueFactory(
        seed=seed,
        document_id=case.document_id,
        target=case.target,
    )
    for binding in case.template.bindings:
        route, reason = _binding_route(binding, case.target)
        eager_output: BindingOutput | None = None
        unchanged_output = _unchanged_target_output(binding, case.target)
        if unchanged_output is not None and not _binding_dependencies_unchanged(
            binding,
            source_target=case.source_target,
            target=case.target,
            constraints=case.template.coherence_constraints,
        ):
            unchanged_output = None
        if unchanged_output is not None:
            # The compiler already certified that these exact source slots express the target
            # snapshots. Preserve them for every realization mode when no dependency changed.
            eager_output = unchanged_output
            route = "deterministic"
            reason = "target value is unchanged from the compiler-certified source surface"
        elif (
            binding.logical_key in coherence_members_with_dependencies
            and binding.logical_key not in changed_coherence_members
        ):
            eager_output = _preserved_source_output(binding)
            route = "deterministic"
            reason = "coherence dependencies are unchanged from the certified source surface"
        elif (
            binding.realization.mode == "deterministic_derivation"
            and binding.derivation == "equipment_receipt"
        ):
            try:
                candidate_output = _render_equipment_receipt_binding(
                    binding,
                    source_target=case.source_target,
                    target=case.target,
                )
                _validate_binding_format(
                    source=case.source,
                    template=case.template.byte_template,
                    output=candidate_output,
                )
            except ValueError as error:
                route = "agent"
                reason = f"typed equipment receipt execution is unavailable: {error}"
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
            residual.append(binding)
            continue
        if eager_output is not None:
            outputs[binding.logical_key] = eager_output
        elif _stable_source_vocabulary(binding) or binding.realization.mode == "static":
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
            outputs[binding.logical_key] = _render_target_binding(binding, case.target)
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
    _solve_identifier_relationships(
        template=case.template,
        outputs=outputs,
        stream=stream,
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
    return RenderPlan(
        routes=tuple(routes),
        deterministic_outputs=outputs,
        residual_bindings=tuple(residual),
    )


def _validate_binding_format(
    *, source: bytes, template: CompiledRawTextTemplate, output: BindingOutput
) -> None:
    bindings = {slot.slot_id: slot.source_text for slot in template.slots}
    bindings.update(output.replacements)
    render_compiled_template(source=source, template=template, bindings=bindings)


def _adapt_target_compatibility(
    *,
    source: bytes,
    source_target: Mapping[str, Any],
    target: dict[str, Any],
    template: CertifiedSemanticTemplate,
) -> tuple[TargetAdaptation, ...]:
    adaptations: list[TargetAdaptation] = []
    deterministic_modes = {
        "single_surface",
        "repeated_surface",
        "segmented_surface",
        "token_projected_surface",
        "normalized_projected_surface",
    }
    for binding in template.bindings:
        if binding.realization.mode not in deterministic_modes:
            continue
        if any(
            not _path_exists(target, path) and _semantic_equipment_value(target, path) is not None
            for path in binding.target_paths
        ):
            # V5 removes the legacy printed description when semantic categories are present.
            # The grouped residual renderer receives those categories directly; the canonical
            # V5 target must not be contaminated with the mutually exclusive legacy field.
            continue
        try:
            output = _render_target_binding(binding, target)
            _validate_binding_format(source=source, template=template.byte_template, output=output)
        except (KeyError, TypeError, ValueError) as error:
            for snapshot in binding.realization.target_values:
                try:
                    proposed = _resolve_path(target, snapshot.target_path)
                except ValueError:
                    proposed = None
                if proposed == snapshot.source_value:
                    raise ValueError(
                        f"source value is not renderable under its certified plan: "
                        f"{binding.logical_key}: {error}"
                    ) from error
                _set_path(target, snapshot.target_path, snapshot.source_value)
                adaptations.append(
                    TargetAdaptation.model_validate(
                        {
                            "target_path": snapshot.target_path,
                            "reason": (
                                f"compiled descendant compatibility constraint: "
                                f"{type(error).__name__}: {error}"
                            ),
                            "source_value": snapshot.source_value,
                            "proposed_value": proposed,
                            "adapted_value": snapshot.source_value,
                        }
                    )
                )
            output = _render_target_binding(binding, target)
            _validate_binding_format(source=source, template=template.byte_template, output=output)
    return tuple(adaptations)


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
    template_root = _validate_committed_run(project_root, config.inputs.template_run)
    target_root = _validate_committed_run(project_root, config.inputs.synthetic_target_run)
    target_path = resolve_input(project_root, config.inputs.synthetic_targets.path)
    if target_root not in target_path.parents:
        raise ValueError("synthetic-target file is outside its pinned committed run")
    if sha256_file(target_path) != config.inputs.synthetic_targets.sha256:
        raise ValueError("synthetic-target file hash differs")
    country_path = resolve_input(project_root, config.inputs.iso3166_snapshot.path)
    if sha256_file(country_path) != config.inputs.iso3166_snapshot.sha256:
        raise ValueError("ISO-3166 snapshot hash differs")
    target_rows = _read_jsonl(target_path, records=config.inputs.synthetic_targets.records)
    targets: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        document_id = row.get("baseDocumentId")
        target = row.get("target")
        if not isinstance(document_id, str) or not isinstance(target, dict):
            raise ValueError("synthetic-target row lacks a document ID or target")
        if document_id in targets:
            raise ValueError(f"duplicate synthetic target for {document_id}")
        if sha256_bytes(canonical_json_bytes(target)) != row.get("targetSha256"):
            raise ValueError(f"synthetic-target row hash differs: {document_id}")
        targets[document_id] = target

    selection_path = template_root / "selection-manifest.json"
    selection = json.loads(read_regular_file_bytes(selection_path))
    rows = selection.get("rows") if isinstance(selection, Mapping) else None
    if not isinstance(rows, list) or len(rows) != config.workflow.documents:
        raise ValueError("template selection does not contain exactly 30 rows")
    document_ids = tuple(row.get("document_id") for row in rows)
    if any(not isinstance(value, str) for value in document_ids):
        raise ValueError("template selection contains an invalid document ID")
    if len(set(document_ids)) != config.workflow.documents:
        raise ValueError("template selection document IDs are not unique")

    prepared: list[PreparedCase] = []
    for document_id in cast(tuple[str, ...], document_ids):
        source, source_target, template_bytes = _case_files(template_root, document_id)
        template = CertifiedSemanticTemplate.model_validate_json(template_bytes, strict=True)
        if template.document_id != document_id or template.source_sha256 != sha256_bytes(source):
            raise ValueError(f"template/source identity differs: {document_id}")
        _validate_canonical_target(source_target)
        if document_id in targets:
            target_origin = "existing_linguistic_target_carrier_restored"
            proposed_target = deepcopy(targets[document_id])
        else:
            target_origin = "controlled_source_variant"
            proposed_target = _controlled_target(
                source_target=source_target,
                template=template,
                seed=config.workflow.controlled_target_seed,
            )
        proposed_sha = sha256_bytes(canonical_json_bytes(proposed_target))
        target = deepcopy(proposed_target)
        adaptations: list[TargetAdaptation] = []
        carrier_adaptation = _restore_carrier(source_target=source_target, target=target)
        if carrier_adaptation is not None:
            adaptations.append(carrier_adaptation)
        adaptations.extend(
            _adapt_target_compatibility(
                source=source,
                source_target=source_target,
                target=target,
                template=template,
            )
        )
        mismatches = printed_topology_mismatches(source_target, target)
        if mismatches:
            details = ", ".join(f"{row.path}:{row.value_kind}" for row in mismatches)
            raise ValueError(
                f"prepared target changes printed topology for {document_id}: {details}"
            )
        _validate_canonical_target(target)
        source_leaves = _flatten_leaves(source_target)
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
            raise ValueError(
                f"prepared target does not preserve the complete carrier: {document_id}"
            )
        target_sha = sha256_bytes(canonical_json_bytes(target))
        synthetic_id = (
            "syn_tpl_"
            + sha256_bytes(
                canonical_json_bytes(
                    [
                        document_id,
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
                "document_id": document_id,
                "target_origin": target_origin,
                "source_schema_version": source_target["schemaVersion"],
                "target_schema_version": target["schemaVersion"],
                "fixed_carrier_name": template.carrier.canonical_name,
                "source_target_sha256": sha256_bytes(canonical_json_bytes(source_target)),
                "proposed_target_sha256": proposed_sha,
                "prepared_target_sha256": target_sha,
                "synthetic_document_id": synthetic_id,
                "topology_mismatch_count": 0,
                "target_leaf_count": len(target_leaves),
                "changed_target_leaf_count": len(changed),
                "carrier_leaf_count": len(carrier_source),
                "carrier_changed_leaf_count": 0,
                "compatibility_adaptations": tuple(adaptations),
                "training_eligible": False,
            }
        )
        prepared.append(
            PreparedCase(
                document_id=document_id,
                source=source,
                source_target=source_target,
                target=target,
                template=template,
                target_receipt=receipt,
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
        return OpenRouterModel(provider.model, provider=OpenRouterProvider(openai_client=client))
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
                    for row in binding.source_relationships
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
        "residualBindings": tuple(rows),
    }


def _residual_output_type(bindings: Sequence[SemanticBinding]) -> type[BaseModel]:
    slots = tuple(slot for binding in bindings for slot in binding.occurrences)
    if not slots:
        raise ValueError("cannot build a residual schema without slots")
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("residual output schema contains duplicate slot IDs")
    fields: dict[str, Any] = {
        slot.slot_id: (
            str,
            Field(
                min_length=1,
                description=(
                    f"Semantic replacement content for {slot.semantic_role}; "
                    f"source render policy is {slot.render_policy}."
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
    parents_by_child: dict[str, set[str]] = {}
    for binding in template.bindings:
        for relationship in binding.source_relationships:
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
        "northkorea": "KP",
        "russia": "RU",
        "czechrepublic": "CZ",
        "ivorycoast": "CI",
    }
    for name, code in aliases.items():
        output.setdefault(name, code)
    return output


def _numeric_interpretations(value: str) -> tuple[Decimal, ...]:
    matches = tuple(match.group(0).strip() for match in _NUMBER.finditer(value))
    if len(matches) != 1:
        return ()
    compact = matches[0].replace(" ", "").replace("\u00a0", "").replace("'", "")
    variants = {compact}
    if "." in compact and "," in compact:
        decimal_separator = "." if compact.rfind(".") > compact.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        variants.add(compact.replace(grouping_separator, "").replace(decimal_separator, "."))
    elif compact.count(",") == 1:
        variants.update({compact.replace(",", "."), compact.replace(",", "")})
    elif compact.count(".") == 1:
        variants.update({compact, compact.replace(".", "")})
    output: list[Decimal] = []
    for variant in variants:
        try:
            parsed = Decimal(variant)
        except InvalidOperation:
            continue
        if parsed.is_finite() and parsed not in output:
            output.append(parsed)
    return tuple(output)


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
        if candidates:
            # Monetary and measurement surfaces conventionally use the interpretation with
            # the smallest absolute magnitude only when one separator can be decimal/grouping.
            # Prefer a fractional interpretation when it is available.
            fractional = tuple(
                candidate for candidate in candidates if candidate != candidate.to_integral()
            )
            if len(fractional) == 1:
                return fractional[0]
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


def _numeric_surface_values(value: str) -> tuple[Decimal, ...]:
    values = list(_numeric_interpretations(value))
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
) -> tuple[Decimal, Decimal]:
    derivation = binding.derivation
    if derivation in {"container_count", "package_count"}:
        if len(binding.dependency_paths) == 1:
            path = binding.dependency_paths[0]
            return Decimal(_dependency_count(source_target, path)), Decimal(
                _dependency_count(target, path)
            )
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
    for path in binding.dependency_paths:
        names = filters[derivation]
        if names is not None and path.rsplit(".", 1)[-1] in names:
            names = None
        source_values.extend(_numeric_leaves(_resolve_path(source_target, path), names=names))
        target_values.extend(_numeric_leaves(_resolve_path(target, path), names=names))
    if not binding.dependency_paths:
        for key in binding.dependency_bindings:
            dependency = bindings[key]
            source_values.append(_first_surface_number(dependency.occurrences[0].source_text))
            target_values.append(_numeric_value(outputs[key].canonical_value))
    if not source_values or not target_values:
        raise ValueError(f"numeric derivation has no values: {binding.logical_key}")
    return sum(source_values, Decimal(0)), sum(target_values, Decimal(0))


def _render_numeric_derived(
    binding: SemanticBinding, *, old: Decimal, new: Decimal
) -> BindingOutput:
    replacements: dict[str, str] = {}
    for slot in binding.occurrences:
        if old == new:
            replacement = slot.source_text
        else:
            replacement = render_number_surface(slot.source_text, old, new)
        replacements[slot.slot_id] = replacement
    canonical: JsonValue = int(new) if new == new.to_integral_value() else float(new)
    return BindingOutput(replacements=replacements, canonical_value=canonical)


def _render_inclusive_range_cardinality(
    binding: SemanticBinding, *, target: Mapping[str, Any]
) -> BindingOutput:
    if len(binding.dependency_paths) != 1 or binding.dependency_bindings:
        raise ValueError("inclusive-range derivation requires exactly one quantity path")
    quantity = _resolve_path(target, binding.dependency_paths[0])
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("inclusive-range derivation quantity is not a positive integer")
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
    for key in binding.dependency_bindings:
        if key not in outputs:
            raise KeyError(key)
        values.append(outputs[key].canonical_value)
    for path in binding.dependency_paths:
        values.append(cast(JsonValue, _resolve_path(target, path)))
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

    if binding.value_kind in {"identifier", "phone"}:
        dependency_source = _alphanumeric(dependency.occurrences[0].source_text)
        dependency_target = _alphanumeric(_scalar_surface(dependency_output.canonical_value))
        if not dependency_source or len(dependency_source) != len(dependency_target):
            raise ValueError("same-as identifier dependency changes width")
        replacements = {}
        for slot in binding.occurrences:
            source = _alphanumeric(slot.source_text)
            offsets = tuple(
                match.start()
                for match in re.finditer(re.escape(dependency_source.casefold()), source.casefold())
            )
            if len(offsets) != 1:
                raise ValueError("same-as identifier dependency is not uniquely embedded")
            offset = offsets[0]
            candidate = (
                source[:offset] + dependency_target + source[offset + len(dependency_source) :]
            )
            replacements[slot.slot_id] = _shape_alphanumeric_like_source(
                slot.source_text, candidate
            )
        return BindingOutput(
            replacements=replacements,
            canonical_value=replacements[binding.occurrences[0].slot_id],
        )

    source_value = dependency.occurrences[0].source_text
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
        try:
            values.add(_numeric_value(value))
        except ValueError:
            continue
    if len(values) != 1:
        raise ValueError(
            f"number-to-words dependency is not uniquely numeric: {binding.logical_key}"
        )
    return values.pop()


def _render_one_derivation(
    *,
    binding: SemanticBinding,
    case: PreparedCase,
    outputs: Mapping[str, BindingOutput],
    bindings: Mapping[str, SemanticBinding],
    country_codes: Mapping[str, str],
) -> BindingOutput:
    derivation = binding.derivation
    if derivation == "same_as_binding":
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
        numeric = _number_to_words_value(binding, outputs, case.target)
        if numeric != numeric.to_integral_value():
            raise ValueError(f"number-to-words dependency is not an integer: {binding.logical_key}")
        words = _number_to_words(int(numeric))
        return BindingOutput(
            replacements={
                slot.slot_id: _layout_like_source(slot.source_text, words)
                for slot in binding.occurrences
            },
            canonical_value=words,
        )
    if derivation == "sum_monetary_amounts":
        new = sum(
            (_numeric_value(outputs[key].canonical_value) for key in binding.dependency_bindings),
            Decimal(0),
        )
        old = _first_surface_number(binding.occurrences[0].source_text)
        return _render_numeric_derived(binding, old=old, new=new)
    if derivation == "temperature_setpoint":
        return _render_temperature_setpoint(binding, target=case.target)
    if derivation == "inclusive_range_cardinality":
        return _render_inclusive_range_cardinality(binding, target=case.target)
    if derivation == "equipment_receipt":
        raise ValueError("equipment receipt must be routed through the residual agent")

    source_base, target_base = _derivation_numeric_values(
        binding=binding,
        source_target=case.source_target,
        target=case.target,
        bindings=bindings,
        outputs=outputs,
    )
    try:
        source_surface = _first_surface_number(binding.occurrences[0].source_text)
    except ValueError:
        unit_paths = tuple(path for path in binding.dependency_paths if path.endswith(".unit"))
        if unit_paths and all(
            _resolve_path(case.source_target, path) == _resolve_path(case.target, path)
            for path in unit_paths
        ):
            return _preserved_source_output(binding)
        raise
    if source_base == 0:
        if target_base != 0:
            raise ValueError(
                f"numeric derivation cannot scale a zero source: {binding.logical_key}"
            )
        target_surface = source_surface
    else:
        target_surface = source_surface * target_base / source_base
    return _render_numeric_derived(binding, old=source_surface, new=target_surface)


def _render_derivations(
    *,
    case: PreparedCase,
    outputs: dict[str, BindingOutput],
    country_codes: Mapping[str, str],
) -> None:
    bindings = {binding.logical_key: binding for binding in case.template.bindings}
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
    """Execute derivations during preflight and explicitly route unsupported ones to one call."""

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
                route_to_agent(
                    binding,
                    f"typed deterministic derivation is unavailable: {type(error).__name__}: "
                    f"{error}",
                )
            else:
                outputs[logical_key] = output
            del pending[logical_key]
            progressed = True
        if progressed:
            continue
        for logical_key, binding in tuple(pending.items()):
            route_to_agent(
                binding,
                "derivation dependency graph cannot be resolved without residual assistance",
            )
            del pending[logical_key]


def _normalized_semantic(value: str) -> str:
    normalized = value.casefold().replace("³", "3").replace("²", "2")
    return "".join(character for character in normalized if character.isalnum())


def _date_candidates(value: str) -> frozenset[date]:
    raw = value.strip()
    localized = raw
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
    output: set[date] = set()
    for variant in variants:
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
        "piece": {"piece", "pieces", "pcs"},
        "roll": {"roll", "rolls", "rol"},
        "sack": {"sack", "sacks", "sak"},
    }
    values = abbreviations.get(noun, {noun, noun + "s"})
    return frozenset(_normalized_semantic(item) for item in values)


def _string_semantics_match(value: str, rendered: str) -> bool:
    expected = _normalized_semantic(value)
    actual = _normalized_semantic(rendered)
    if not expected or not actual:
        return False
    if expected in actual or actual in expected:
        return True
    if value.startswith("PACKAGE_"):
        return any(variant in actual for variant in _package_variants(value))
    variants = {
        "nonnegotiable": {
            "nonnegotiable",
            "notnegotiable",
            "expressbilloflading",
            "seawaybill",
            "expressrelease",
        },
        "negotiable": {"negotiable", "originalbilloflading", "toorder"},
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


def _equipment_semantics_match(value: Mapping[str, Any], rendered: str) -> bool:
    size = value.get("sizeCategory")
    equipment_type = value.get("typeCategory")
    if not isinstance(size, str) or not isinstance(equipment_type, str):
        return False
    actual = _normalized_semantic(rendered)
    length = (
        "20" if size.startswith("TWENTY_") else "45" if size.startswith("FORTY_FIVE_") else "40"
    )
    if length not in actual:
        return False
    if "HIGH_CUBE" in size and not any(
        token in actual for token in ("highcube", "hicube", "hc", "hq", "96")
    ):
        return False
    variants = {
        "GENERAL_PURPOSE": ("generalpurpose", "gp", "dry", "dv", "dc", "hc", "hq"),
        "VENTILATED_GENERAL_PURPOSE": ("ventilated", "vh"),
        "DRY_BULK": ("drybulk", "bulk", "bu"),
        "NAMED_CARGO": ("namedcargo", "sn"),
        "REFRIGERATED": ("refrigerated", "reefer", "rf", "re"),
        "REFRIGERATED_AND_HEATED": ("refrigeratedheated", "reefer", "rh", "rt"),
        "SELF_POWERED_REFRIGERATED": ("selfpowered", "reefer", "rs"),
        "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT": ("reefer", "removable", "hr"),
        "INSULATED": ("insulated", "hi"),
        "OPEN_TOP": ("opentop", "ot", "ut"),
        "PLATFORM": ("platform", "flat", "pl"),
        "PLATFORM_FIXED": ("platform", "flat", "pf"),
        "PLATFORM_COLLAPSIBLE": ("platform", "flat", "pc"),
        "PLATFORM_COMPLETE_SUPERSTRUCTURE": ("platform", "flat", "ps"),
        "PLATFORM_NAMED_CARGO": ("platform", "flat", "pt"),
        "PRESSURIZED_TANK": ("tank", "kl"),
        "DRY_HOPPER_TANK": ("hopper", "tank", "nh"),
        "DRY_REAR_DISCHARGE_TANK": ("tank", "rear", "nn"),
        "AIR_SURFACE": ("airsurface", "as"),
    }
    return any(token in actual for token in variants[equipment_type])


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


def _target_binding_semantics_valid(
    *, case: PreparedCase, outputs: Mapping[str, BindingOutput]
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    deterministic_target_modes = {
        "single_surface",
        "repeated_surface",
        "segmented_surface",
        "token_projected_surface",
        "normalized_projected_surface",
    }
    for binding in case.template.bindings:
        if not binding.target_paths:
            continue
        output = outputs[binding.logical_key]
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
        if binding.realization.requires_agent and binding.target_paths:
            try:
                expected = _render_agent_target_binding(
                    binding,
                    source_target=case.source_target,
                    target=case.target,
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
            expected = _render_target_binding(binding, case.target)
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
                        failures.append(f"{binding.logical_key}:{path}:invalid-target-date")
                        continue
                    if not any(
                        expected_date in _date_candidates(value)
                        for value in output.replacements.values()
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
                elif not any(
                    expected_number in _numeric_surface_values(value)
                    for value in output.replacements.values()
                ):
                    failures.append(f"{binding.logical_key}:{path}:number-not-rendered")
            elif isinstance(target_value, Mapping) and set(target_value) == {
                "sizeCategory",
                "typeCategory",
            }:
                if not _equipment_semantics_match(target_value, rendered):
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
    for binding in template.bindings:
        if not binding.source_relationships:
            continue
        current = _alphanumeric(
            outputs[binding.logical_key].replacements[binding.occurrences[0].slot_id]
        ).casefold()
        for relationship in binding.source_relationships:
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
    if read_regular_file_bytes(prefix / "source.txt") != case.source:
        raise ValueError(f"replay source bytes differ for {case.document_id}")
    expected_json = (
        ("source-target.json", case.source_target),
        ("target.json", case.target),
        ("target-receipt.json", case.target_receipt.model_dump(mode="json")),
    )
    for name, expected in expected_json:
        actual = json.loads(read_regular_file_bytes(prefix / name))
        if actual != expected:
            raise ValueError(f"replay {name} differs for {case.document_id}")

    stage_path = prefix / "agent-stage.json"
    source_stage = ResidualStageReceipt.model_validate_json(read_regular_file_bytes(stage_path))
    if source_stage.document_id != case.document_id:
        raise ValueError(f"replay agent-stage document differs for {case.document_id}")
    if source_stage.system_prompt_sha256 != expected_system_prompt_sha256:
        raise ValueError(f"replay system prompt differs for {case.document_id}")

    expected_slots = {
        slot.slot_id for binding in plan.residual_bindings for slot in binding.occurrences
    }
    if expected_slots:
        if source_stage.status != "success" or not isinstance(source_stage.output, Mapping):
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
        if source_stage.status not in {"success", "not_required"}:
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
    return projected, source_stage, receipt


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
        outputs = dict(plan.deterministic_outputs)
        outputs.update(_postprocess_residual_outputs(case=case, plan=plan, raw_output=raw_output))
        _reconcile_identifier_relationships(template=case.template, outputs=outputs)
        _render_derivations(case=case, outputs=outputs, country_codes=country_codes)
        if set(outputs) != {binding.logical_key for binding in case.template.bindings}:
            raise ValueError("binding outputs do not cover the semantic template exactly")
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
            case=case, outputs=outputs
        )
        if not semantic_valid:
            raise ValueError("target semantics failed: " + ", ".join(semantic_failures))
        relationships_valid, relationship_failures = _source_relationships_valid(
            template=case.template, outputs=outputs
        )
        if not relationships_valid:
            raise ValueError("source relationships failed: " + ", ".join(relationship_failures))
        rendered, proof = render_compiled_template(
            source=case.source,
            template=case.template.byte_template,
            bindings=slot_bindings,
        )
        carrier_valid = _carrier_unchanged(case=case, outputs=outputs)
        if not carrier_valid:
            raise ValueError("carrier-bound label or carrier slot changed")
        if printed_topology_mismatches(case.source_target, case.target):
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
                raise ValueError(
                    f"provider-free preflight materialization failed for {case.document_id}: "
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
    return {
        "schemaVersion": 1,
        "status": "ready",
        "documents": len(cases),
        "exactTopologyDocuments": len(cases),
        "carrierBoundDocuments": len(cases),
        "existingTargetDocuments": sum(
            case.target_receipt.target_origin == "existing_linguistic_target_carrier_restored"
            for case in cases
        ),
        "controlledTargetDocuments": sum(
            case.target_receipt.target_origin == "controlled_source_variant" for case in cases
        ),
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


def preflight_descendants(config_path: Path) -> dict[str, Any]:
    project_root = project_root_from_config(config_path)
    config = load_descendant_config(config_path)
    prompt_path = resolve_input(project_root, config.prompts.residual_renderer.path)
    if sha256_file(prompt_path) != config.prompts.residual_renderer.sha256:
        raise ValueError("residual renderer prompt hash differs")
    cases = _load_cases(project_root=project_root, config=config)
    country_path = resolve_input(project_root, config.inputs.iso3166_snapshot.path)
    countries = _country_code_map(country_path)
    plans = tuple(
        _build_initial_plan(
            case,
            seed=config.workflow.controlled_target_seed,
            country_codes=countries,
        )
        for case in cases
    )
    return _preflight_payload(
        cases,
        plans,
        country_codes=countries,
        system_prompt_sha256=config.prompts.residual_renderer.sha256,
    )


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
        "trainingRecordsPublished": False,
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
            "- Training records published: **no**",
            "",
        )
    )


async def run_descendants(config_path: Path) -> Path:
    started = time.perf_counter()
    project_root = project_root_from_config(config_path)
    config = load_descendant_config(config_path)
    prompt_path = resolve_input(project_root, config.prompts.residual_renderer.path)
    if sha256_file(prompt_path) != config.prompts.residual_renderer.sha256:
        raise ValueError("residual renderer prompt hash differs")
    system_prompt = prompt_path.read_text(encoding="utf-8")
    cases = _load_cases(project_root=project_root, config=config)
    country_path = resolve_input(project_root, config.inputs.iso3166_snapshot.path)
    countries = _country_code_map(country_path)
    plans = tuple(
        _build_initial_plan(
            case,
            seed=config.workflow.controlled_target_seed,
            country_codes=countries,
        )
        for case in cases
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
        replay_root = _validate_committed_run(project_root, replay_pin)

    transaction = {
        "schemaVersion": 1,
        "runName": config.run_name,
        "configSha256": sha256_file(config_path),
        "templateRunCommitSha256": config.inputs.template_run.commit_sha256,
        "syntheticTargetRunCommitSha256": config.inputs.synthetic_target_run.commit_sha256,
        "syntheticTargetsSha256": config.inputs.synthetic_targets.sha256,
        "iso3166Sha256": config.inputs.iso3166_snapshot.sha256,
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
        "documentIds": [case.document_id for case in cases],
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
    )
    if stage.completed:
        stage.validate_committed_run()
        return stage.final_root
    stage.recover_interrupted_temporary_files()
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    stage.publish_bytes("prompts/residual-renderer.md", read_regular_file_bytes(prompt_path))
    stage.publish_json("transaction.json", transaction)
    stage.publish_json("preflight.json", preflight)
    for case, plan in zip(cases, plans, strict=True):
        prefix = f"cases/{case.document_id}"
        stage.publish_bytes(f"{prefix}/source.txt", case.source)
        stage.publish_json(f"{prefix}/source-target.json", case.source_target)
        stage.publish_json(f"{prefix}/target.json", case.target)
        stage.publish_json(
            f"{prefix}/target-receipt.json", case.target_receipt.model_dump(mode="json")
        )
        stage.publish_json(
            f"{prefix}/route-plan.json",
            [row.model_dump(mode="json") for row in plan.routes],
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
        executions = await asyncio.gather(
            *(
                _execute_case(
                    case=case,
                    plan=plan,
                    model=model,
                    provider=config.provider,
                    system_prompt=system_prompt,
                    limiter=limiter,
                    country_codes=countries,
                )
                for case, plan in zip(cases, plans, strict=True)
            )
        )
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
    for index, (case, execution) in enumerate(zip(cases, executions, strict=True)):
        prefix = f"cases/{case.document_id}"
        if replay_receipts is None:
            stage.publish_json(
                f"{prefix}/agent-stage.json", execution.stage.model_dump(mode="json")
            )
        else:
            stage.publish_json(
                f"{prefix}/source-agent-stage.json", execution.stage.model_dump(mode="json")
            )
            stage.publish_json(
                f"{prefix}/replay-receipt.json",
                replay_receipts[index].model_dump(mode="json"),
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
            stage.publish_json(
                f"{prefix}/render-proof.json", execution.proof.model_dump(mode="json")
            )
    results = tuple(execution.result for execution in executions)
    stage.publish_bytes(
        "results.jsonl",
        _jsonl_bytes([row.model_dump(mode="json") for row in results]),
    )
    summary = _summary(results, preflight, execution_mode=execution_mode)
    summary["wallTimeSeconds"] = time.perf_counter() - started
    stage.publish_json("summary.json", summary)
    stage.publish_bytes("REPORT.md", _report(summary).encode("utf-8"))
    commit = stage.commit(
        expected_artifacts=_artifact_inventory(stage),
        metadata={
            "status": cast(JsonValue, summary["status"]),
            "documents": len(results),
            "passedDocuments": cast(int, summary["passedDocuments"]),
            "providerRequests": cast(int, summary["providerRequests"]),
            "newProviderRequests": cast(int, summary["newProviderRequests"]),
            "executionMode": execution_mode,
            "estimatedCostUsd": cast(str, summary["estimatedCostUsd"]),
            "trainingRecordsPublished": False,
        },
    )
    if not commit.created:
        stage.validate_committed_run()
    return stage.final_root
