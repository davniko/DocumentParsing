"""Current-target synthesis from explicitly owned source regions.

One contract drives identity replay, scenario rendering and publication.
Postal labels use complete owned surfaces, not historical address projection.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from math import gcd
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse

from document_ocr.labeling_agents.target_normalization import normalize_target_casing
from document_ocr.synthesis.curated_descriptions import (
    CompiledDescriptionBlocks,
    compile_description_blocks,
    project_rendered_descriptions,
    validate_description_regions,
)
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_container_number,
    generate_from_surface_pattern,
    surface_pattern,
    validate_container_number,
)
from document_ocr.synthesis.linguistic_probe_runtime import model_messages, usage_receipt
from document_ocr.synthesis.raw_text_template import (
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)
from document_ocr.synthesis.template_compiler.descendant import (
    _number_to_words,
    _number_word_phrase,
    _provider_model,
    _provider_settings,
)
from document_ocr.synthesis.template_compiler.models import OpenRouterProviderConfig
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CuratedSynthesisConfig(StrictModel):
    """Explicit input, scenario, provider and spend boundaries for a curated run."""

    dataset: str
    output: str
    environment_file: str
    task_constraints: str
    task_constraints_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    variants_per_source: int = Field(ge=1)
    concurrency: int = Field(ge=1, le=32)
    budget_usd: Decimal = Field(gt=0)
    provider: OpenRouterProviderConfig
    source_ids: list[str] = Field(min_length=1)


class Occurrence(StrictModel):
    """One exact source occurrence, counted from the beginning of the OCR."""

    text: str = Field(
        min_length=1, description="Exact OCR substring; include no headings or unrelated facts."
    )
    occurrence: int = Field(
        ge=1,
        description=(
            "One-based occurrence of this exact, case-sensitive substring in the complete OCR."
        ),
    )
    presentation: Literal["text", "number", "words"] = Field(
        description=(
            "Text retains its line structure; number retains printed "
            "precision/separators; words spells an "
            "integer."
        )
    )


class Variable(StrictModel):
    """One shared fact, possibly printed repeatedly or owning several label paths."""

    key: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique meaningful key, shared by repeated occurrences of the same fact.",
    )
    kind: Literal[
        "postal", "name", "product", "identifier", "container", "count", "mass", "volume"
    ] = Field(
        description=(
            "Generation responsibility: coherent lexical surface, "
            "deterministic ID, or coupled cargo "
            "arithmetic."
        )
    )
    value: str = Field(
        min_length=1,
        description=(
            "Source value. Numeric values use decimal point with no thousands "
            "separator. Text uses the whole owned value with physical "
            "newlines replaced by "
            "spaces."
        ),
    )
    meaning: str = Field(
        min_length=1,
        description=(
            "Concise ownership, semantic role and generation constraints. "
            "Postal: country/city context and component granularity. Product: "
            "classification and protected specifications. Product regions cover the main "
            "product passage or genuine continuation, separate from loading introductions "
            "and detached accounting/tracking text. Uncertain boundaries require review."
        ),
    )
    required_literals: list[str] = Field(
        description=(
            "Printed fixed anchors that replacement must retain: country "
            "spelling/code in postal regions, linked named sites, product "
            "chemical/brand/model/specification identity. Empty when none "
            "apply."
        )
    )
    occurrences: list[Occurrence] = Field(
        min_length=1,
        description=(
            "Every printed occurrence of this mutable fact, including "
            "attachment and auxiliary "
            "repetitions."
        ),
    )


class TargetBinding(StrictModel):
    """A current scalar target reconstructed from rendered variables."""

    path: str = Field(
        description=(
            "Existing dot/index scalar path beginning documentPatch., e.g. "
            "documentPatch.parties.shipper.addressLine."
        )
    )
    expression: str = Field(
        description=(
            "Source-faithful text with {key} substitutions. Numeric values "
            "use one variable or an explicit sum. Include all dependent "
            "labels."
        )
    )


class SourceContract(StrictModel):
    """Rebinding of current OCR/current labels; identity must replay without changes."""

    variables: list[Variable] = Field(
        min_length=1,
        description=(
            "Disjoint mutable regions. Repeated parties use one coherent "
            "variable, not independent addresses. Separate tax/contact data "
            "remain outside postal "
            "regions."
        ),
    )
    targets: list[TargetBinding] = Field(
        min_length=1,
        description=(
            "All current label scalars affected by the variables; unlisted labels stay fixed."
        ),
    )
    fixed_context: str = Field(
        description=(
            "Explicit limits on variation: commodity/HS/UN, thermal settings, "
            "equipment topology, countries, dates and any facts that cannot "
            "safely vary in this "
            "source."
        )
    )


class LexicalValue(StrictModel):
    """A complete coherent replacement for one contracted lexical region."""

    key: str = Field(description="Exact requested variable key.")
    value: str = Field(
        min_length=1,
        description=(
            "Single-line replacement. Postal includes every owned address "
            "component exactly once, without tax/contact/name/caption text. "
            "Product stays within the supplied classification and "
            "specifications."
        ),
    )


class LexicalBundle(StrictModel):
    """Joint generation for a shipment, with no independent repeated-party edits."""

    values: list[LexicalValue] = Field(
        min_length=1,
        description=(
            "Exactly one value per requested key; all co-referent occurrences "
            "are rendered by the "
            "host."
        ),
    )


class LexicalBatch(StrictModel):
    """Ordered variations of the same source-owned lexical regions."""

    variants: list[LexicalBundle] = Field(
        min_length=1, description="Requested number of distinct variations, in scenario order."
    )


class Finding(StrictModel):
    """A correctness defect or unresolved ownership question, not a style preference."""

    location: str = Field(description="Affected variable, target path or quoted OCR region.")
    problem: str = Field(
        description=(
            "Concrete defect and why, or competing ownership interpretations "
            "that remain unresolved."
        )
    )
    correction: str = Field(
        description=(
            "Specific source/policy-supported correction, or recommended review action "
            "when genuinely unresolved; no invented unprinted fact."
        )
    )


class Review(StrictModel):
    """Semantic review separate from deterministic replay checks."""

    findings: list[Finding] = Field(
        description=(
            "All concrete defects and unresolved ownership questions; "
            "empty only when neither remains."
        )
    )
    explanation: str = Field(
        min_length=40,
        description=(
            "At most three short sentences summarizing checks and any "
            "limitation. Findings carry specific defects; do not repeat all "
            "correct fields, tables or an "
            "essay."
        ),
    )


def digest(value: Any) -> str:
    data = (
        value
        if isinstance(value, bytes)
        else json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    )
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def flat(value: Any, path: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            p: x for k, v in value.items() for p, x in flat(v, f"{path}.{k}".lstrip(".")).items()
        }
    if isinstance(value, list):
        return {p: x for i, v in enumerate(value) for p, x in flat(v, f"{path}[{i}]").items()}
    return {path: value}


def assign(target: dict, path: str, value: Any) -> None:
    parts = re.findall(r"[^.\[\]]+", path)
    node: Any = target
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if isinstance(node, list):
        node[int(parts[-1])] = value
    else:
        if parts[-1] not in node:
            raise ValueError(f"target binding creates an unprinted field: {path}")
        node[parts[-1]] = value


def interpolate(expression: str, values: dict[str, str]) -> str:
    return re.sub(r"\{([a-z][a-z0-9_]*)\}", lambda m: values[m[1]], expression)


def contains_literal(text: str, literal: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", text, re.I) is not None


def locate(text: str, occurrence: Occurrence) -> tuple[int, int]:
    matches = list(re.finditer(re.escape(occurrence.text), text))
    if occurrence.occurrence > len(matches):
        raise ValueError(f"missing occurrence {occurrence.occurrence} of {occurrence.text!r}")
    match = matches[occurrence.occurrence - 1]
    return len(text[: match.start()].encode()), len(text[: match.end()].encode())


def _number_style(text: str, baseline: str, value: str) -> str:
    """Choose a numeric presentation only when it round-trips the observed value."""
    original, generated = Decimal(baseline), Decimal(value)
    candidates = []
    for decimal, thousands in ((".", ","), (",", "."), (".", " "), (",", " "), (".", "")):
        try:
            clean = text.replace(thousands, "") if thousands else text
            clean = clean.replace(decimal, ".")
            if Decimal(clean) != original:
                continue
        except Exception:
            continue
        precision = len(text.rsplit(decimal, 1)[1]) if decimal in text else 0
        if generated != generated.quantize(Decimal(1).scaleb(-precision)):
            continue
        grouped = bool(thousands and thousands in text)

        def formatted(
            number, precision=precision, grouped=grouped, decimal=decimal, thousands=thousands
        ):
            result = format(number, f",.{precision}f" if grouped else f".{precision}f")
            result = result.replace(",", "~").replace(".", decimal).replace("~", thousands)
            if text.startswith("0") and len(text) > 1 and not grouped and decimal not in text:
                result = result.zfill(len(text))
            return result

        if formatted(original) != text:
            continue
        rendered = formatted(generated)
        candidates.append(rendered)
    if not candidates:
        raise ValueError(
            f"number presentation cannot exactly represent {value}: {text!r} (baseline {baseline})"
        )
    if len(set(candidates)) != 1:
        raise ValueError(f"ambiguous numeric presentation: {text!r}")
    return candidates[0]


def layout_surface(source: str, value: str) -> str:
    """Reflow whole words, keeping closing punctuation with its preceding word.

    Approximate source line proportions without treating a standalone comma as
    a line-sized word. Short replacements may use fewer lines; text fidelity
    takes priority over artificial blank lines.
    """
    words = re.findall(r"\S+(?:[ \t]+[,.;:!?]+(?=\s|$))*", value)
    if not words:
        raise ValueError("cannot lay out empty text")
    lines = min(len(source.splitlines()), len(words))
    if lines <= 1:
        return " ".join(words)
    weights = [max(1, len(line)) for line in source.splitlines()[:lines]]
    prefix = [0]
    for word in words:
        prefix.append(prefix[-1] + len(word) + 1)
    total_weight = sum(weights)
    previous = consumed = 0
    output = []
    for line, weight in enumerate(weights[:-1]):
        consumed += weight
        desired = prefix[-1] * consumed / total_weight
        end = min(
            range(previous + 1, len(words) - (lines - line - 1) + 1),
            key=lambda index: abs(prefix[index] - desired),
        )
        output.append(" ".join(words[previous:end]))
        previous = end
    output.append(" ".join(words[previous:]))
    return "\n".join(output)


def surface(variable: Variable, occurrence: Occurrence, value: str) -> str:
    if value == variable.value:
        return occurrence.text
    if occurrence.presentation == "number":
        return _number_style(occurrence.text, variable.value, value)
    if occurrence.presentation == "words":
        number = Decimal(value)
        if number != int(number):
            raise ValueError("spelled quantity is not integral")
        words = _number_to_words(int(number))
        return (
            words.upper()
            if occurrence.text.isupper()
            else words.title()
            if occurrence.text.istitle()
            else words
        )
    if variable.kind in {"container", "identifier"}:
        # Keep source separators while replacing significant identifier characters.
        separators = r"[\s/\-]" if variable.kind == "container" else r"\s"
        old = re.sub(separators, "", occurrence.text)
        if old == variable.value and len(value) == len(old):
            it = iter(value)
            return "".join(c if re.fullmatch(separators, c) else next(it) for c in occurrence.text)
    return layout_surface(occurrence.text, value)


def compile_contract(
    row: dict,
    contract: SourceContract,
    *,
    description_blocks: CompiledDescriptionBlocks | None = None,
):
    keys = [v.key for v in contract.variables]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate variable keys")
    leaves = flat(row["target"])
    paths = [t.path for t in contract.targets]
    if len(paths) != len(set(paths)) or not set(paths) <= leaves.keys():
        raise ValueError("target paths must be distinct existing scalar leaves")
    used = {k for t in contract.targets for k in re.findall(r"\{([a-z][a-z0-9_]*)\}", t.expression)}
    if not used <= set(keys):
        raise ValueError(f"unknown expression variables: {used - set(keys)}")
    slots, associations = [], {}
    for variable in contract.variables:
        for literal in variable.required_literals:
            if not contains_literal(variable.value, literal):
                raise ValueError(
                    f"{variable.key}: fixed literal is absent from baseline: {literal!r}"
                )
        for occurrence in variable.occurrences:
            if occurrence.presentation == "number":
                _number_style(occurrence.text, variable.value, variable.value)
            elif occurrence.presentation == "words":
                number, _, _ = _number_word_phrase(occurrence.text)
                if Decimal(number) != Decimal(variable.value):
                    raise ValueError("spelled count differs from baseline")
            elif occurrence.presentation == "text":
                printed = " ".join(occurrence.text.split()).upper()
                expected = " ".join(variable.value.split()).upper()
                if variable.kind in {"identifier", "container"}:
                    separators = r"[\s/\-]" if variable.kind == "container" else r"\s"
                    printed, expected = (
                        re.sub(separators, "", printed),
                        re.sub(separators, "", expected),
                    )
                if printed != expected:
                    raise ValueError(
                        f"{variable.key}: printed value {printed!r} "
                        f"differs from baseline {expected!r}"
                    )
            start, end = locate(row["joinedRawText"], occurrence)
            slot_id = f"slot_{len(slots):04d}"
            slots.append(
                build_template_slot(
                    slot_id=slot_id,
                    byte_start=start,
                    byte_end=end,
                    source_text=occurrence.text,
                    target_paths=[
                        t.path for t in contract.targets if "{" + variable.key + "}" in t.expression
                    ],
                    semantic_role=variable.meaning,
                    evidence_origin="accepted_label_evidence"
                    if variable.key in used
                    else "audited_source_auxiliary",
                    render_policy="natural_text",
                )
            )
            associations[slot_id] = (variable, occurrence)
    template = compile_raw_text_template(
        document_id=row["documentId"], source=row["joinedRawText"].encode(), slots=slots
    )
    identity_contract = contract
    if description_blocks is not None:
        raw = row["joinedRawText"].encode()
        descriptions, _ = project_rendered_descriptions(description_blocks, raw, raw, [])
        if any(leaves.get(path) != value for path, value in descriptions.items()):
            raise ValueError("source description differs from reviewed block projection")
        # Lexical expressions still define generated wording dependencies; only
        # reviewed physical block membership defines a description target.
        identity_contract = contract.model_copy(
            update={"targets": [t for t in contract.targets if t.path not in descriptions]}
        )
    target = derive_target(
        row["target"], identity_contract, {v.key: v.value for v in contract.variables}
    )
    if target != row["target"]:
        differences = {p: [leaves.get(p), v] for p, v in flat(target).items() if leaves.get(p) != v}
        raise ValueError(f"current-label identity replay differs: {differences}")
    return template, associations


def derive_target(original: dict, contract: SourceContract, values: dict[str, str]) -> dict:
    target, leaves = deepcopy(original), flat(original)
    for binding in contract.targets:
        value: Any = " ".join(interpolate(binding.expression, values).split())
        old = leaves[binding.path]
        if isinstance(old, (int, float)) and not isinstance(old, bool):
            # Explicit sums of source-owned components are the only arithmetic
            # admitted by this contract; no eval or arbitrary expressions.
            number = sum((Decimal(term.strip()) for term in value.split("+")), Decimal(0))
            if isinstance(old, int) and number != int(number):
                raise ValueError(f"non-integral target quantity at {binding.path}")
            value = int(number) if isinstance(old, int) else float(number)
        assign(target, binding.path, value)
    target, _ = normalize_target_casing(target)
    canonical = get_training_task("bill_of_lading_extraction_v7_reduced").canonicalize(target)
    if canonical != target:
        raise ValueError("synthesis target is not already canonical under the current reduced task")
    return target


@dataclass(frozen=True)
class _DescriptionRegion:
    start: int
    end: int
    key: str
    kind: str
    target_paths: tuple[str, ...]


def render(
    row: dict,
    contract: SourceContract,
    values: dict[str, str],
    *,
    description_blocks: CompiledDescriptionBlocks | None = None,
) -> tuple[str, dict, dict]:
    description_paths = {
        f"documentPatch.goodsItemDetails[{index}].description"
        for index, goods in enumerate(
            row["target"].get("documentPatch", {}).get("goodsItemDetails", [])
        )
        if "description" in goods
    }
    if description_paths != (description_blocks.paths if description_blocks else set()):
        raise ValueError("description targets require complete reviewed description_blocks")
    if set(values) != {v.key for v in contract.variables}:
        raise ValueError("scenario variable set differs from contract")
    for var in contract.variables:
        value = values[var.key]
        if not value.strip() or "\n" in value or "\r" in value:
            raise ValueError(f"{var.key}: scenario requires a nonempty single-line value")
        for literal in var.required_literals:
            if not contains_literal(value, literal):
                raise ValueError(f"{var.key}: lost required literal {literal!r}")
            pattern = r"(?<!\w)" + re.escape(literal) + r"(?!\w)"
            if var.kind == "postal" and len(re.findall(pattern, value, re.I)) != len(
                re.findall(pattern, var.value, re.I)
            ):
                raise ValueError(f"{var.key}: repeated postal anchor {literal!r}")
        if var.kind == "product" and re.findall(r"\d+(?:[.,/]\d+)*", value) != re.findall(
            r"\d+(?:[.,/]\d+)*", var.value
        ):
            raise ValueError(
                f"{var.key}: product numeric specifications changed outside the numeric plan"
            )
        if var.kind == "container" and value != var.value and not validate_container_number(value):
            raise ValueError(f"{var.key}: invalid generated ISO6346 identifier")
    template, associations = compile_contract(row, contract, description_blocks=description_blocks)
    validate_description_regions(
        description_blocks,
        tuple(
            _DescriptionRegion(
                slot.byte_start,
                slot.byte_end,
                associations[slot.slot_id][0].key,
                associations[slot.slot_id][0].kind,
                tuple(slot.target_paths),
            )
            for slot in template.slots
        ),
    )
    bindings = {
        key: surface(var, occurrence, values[var.key])
        for key, (var, occurrence) in associations.items()
    }
    source_bytes = row["joinedRawText"].encode()
    for slot in template.slots:
        rendered = bindings[slot.slot_id]
        following = source_bytes[slot.byte_end : slot.byte_end + 1].decode(errors="ignore")
        if (
            rendered != slot.source_text
            and rendered[-1:] in {".", ",", ";", ":"}
            and following == rendered[-1:]
            and not slot.source_text.endswith(following)
        ):
            raise ValueError(f"{slot.slot_id}: duplicates a template-owned delimiter")
        var, occurrence = associations[slot.slot_id]
        if (
            occurrence.presentation == "text"
            and var.kind in {"postal", "name", "product"}
            and " ".join(rendered.split()).upper() != " ".join(values[var.key].split()).upper()
        ):
            raise ValueError(f"{var.key}: layout changed generated lexical content")
    payload, proof = render_compiled_template(
        source=row["joinedRawText"].encode(), template=template, bindings=bindings
    )
    # Independently replay the edits instead of accepting the proof object's flags.
    expected, cursor, edits = bytearray(), 0, []
    for slot in template.slots:
        expected.extend(source_bytes[cursor : slot.byte_start])
        expected.extend(bindings[slot.slot_id].encode())
        cursor = slot.byte_end
        if bindings[slot.slot_id] != slot.source_text:
            edits.append(
                {
                    "byteStart": slot.byte_start,
                    "byteEnd": slot.byte_end,
                    "before": slot.source_text,
                    "after": bindings[slot.slot_id],
                }
            )
    expected.extend(source_bytes[cursor:])
    if bytes(expected) != payload:
        raise ValueError("independent edit replay failed")
    descriptions, description_proof = project_rendered_descriptions(
        description_blocks, source_bytes, payload, edits
    )
    target_contract = contract.model_copy(
        update={"targets": [t for t in contract.targets if t.path not in descriptions]}
    )
    target = derive_target(row["target"], target_contract, values)
    for path, value in descriptions.items():
        assign(target, path, value)
    before = row["target"]["documentPatch"]
    after = target["documentPatch"]
    for old_goods, new_goods in zip(
        before.get("goodsItemDetails", []), after.get("goodsItemDetails", []), strict=True
    ):
        old_allocations = old_goods.get("splitGoodsPlacement", [])
        new_allocations = new_goods.get("splitGoodsPlacement", [])
        old_total = sum(
            p.get("packageQuantity", 0) for p in old_goods.get("numberAndTypeOfPackages", [])
        )
        new_total = sum(
            p.get("packageQuantity", 0) for p in new_goods.get("numberAndTypeOfPackages", [])
        )
        if old_allocations and all("packageQuantity" in p for p in old_allocations):
            old_sum = sum(p["packageQuantity"] for p in old_allocations)
            new_sum = sum(p["packageQuantity"] for p in new_allocations)
            if old_sum == old_total and new_sum != new_total:
                raise ValueError("complete source allocation no longer sums to package total")
            if old_sum <= old_total and new_sum > new_total:
                raise ValueError("partial allocation exceeds package total")
        # No numerical equality is used to infer an unprinted placement. Preserve
        # the exact source-supported membership and quantity-presence topology.
        if len(old_allocations) != len(new_allocations):
            raise ValueError("placement topology changed")
    receipt = proof.model_dump(mode="json")
    if description_proof is not None:
        receipt["descriptionProjection"] = description_proof
    return payload.decode(), target, receipt


def validate_candidate(
    row: dict,
    contract: SourceContract,
    candidate: dict,
    *,
    description_blocks: CompiledDescriptionBlocks | None = None,
) -> None:
    """Publication guard: recompute artifacts from the pinned source and scenario."""
    if candidate["sourceDocumentId"] != row["documentId"]:
        raise ValueError("candidate source identity differs")
    if candidate["contractSha256"] != digest(contract.model_dump(mode="json")):
        raise ValueError("candidate contract identity differs")
    expected_id = (
        "syn_v7_" + digest([row["documentId"], candidate["seed"], candidate["variant"]])[:24]
    )
    if candidate["documentId"] != expected_id:
        raise ValueError("candidate scenario identity differs")
    planned = scenario_values(
        contract, expected_id, candidate["seed"], candidate["variant"], candidate["variantCount"]
    )
    for variable in contract.variables:
        if variable.kind == "postal" and re.search(
            r"(?<!\d)0{5,6}(?!\d)", candidate["values"][variable.key]
        ):
            raise ValueError(f"{variable.key}: generated postal placeholder")
        if (
            variable.kind not in {"postal", "name", "product"}
            and candidate["values"][variable.key] != planned[variable.key]
        ):
            raise ValueError(f"host-owned scenario value changed: {variable.key}")
    expected_text, expected_target, expected_proof = render(
        row, contract, candidate["values"], description_blocks=description_blocks
    )
    if candidate["joinedRawText"] != expected_text or candidate["target"] != expected_target:
        raise ValueError("candidate differs from exact source/scenario replay")
    if (
        candidate["joinedRawTextSha256"] != digest(expected_text.encode())
        or candidate["proof"] != expected_proof
    ):
        raise ValueError("candidate hash/render proof differs")


def scenario_values(
    contract: SourceContract, sample_id: str, seed: int, variant: int, variant_count: int = 3
) -> dict[str, str]:
    """Preserve exact count/mass equations; downscale only on a representable lattice."""
    values = {v.key: v.value for v in contract.variables}
    if not 0 <= variant < variant_count:
        raise ValueError("variant is outside the requested scenario count")
    stream = DeterministicStream(seed, "curated-v7", sample_id)
    counts = [int(v.value) for v in contract.variables if v.kind == "count"]
    common = 0
    for count in counts:
        common = gcd(common, count)
    # Integer lattice, not a two-decimal percentage: each row is c/g shared
    # units. Sample the shared unit count while retaining all row/total ratios.
    # The 60-99% envelope explores lighter loads without enlarging source loads.
    lower, upper = (3 * common + 4) // 5, 99 * common // 100
    count_units = (
        lower + round(variant * (upper - lower) / max(1, variant_count - 1))
        if common > 1 and lower <= upper
        else common
    )
    # Every affected numeric surface must exactly represent the scenario. The
    # finite lattice search is explicit; a source incapable of mass variation
    # remains an identified capability failure, not a silent unchanged sample.
    ratios = [Decimal(n) / 100 for n in range(60, 100)]
    for dimension in ("mass", "volume"):
        numeric = [v for v in contract.variables if v.kind == dimension]
        if not numeric:
            continue
        feasible = []
        for ratio in ratios:
            proposal = {}
            try:
                for var in numeric:
                    value = Decimal(var.value) * ratio
                    if len(numeric) == 1:
                        # One independent printed total has no row-sum equation.
                        # Round at its finest shared observed presentation precision.
                        max_precision = max(len(o.text.rsplit(".", 1)[-1]) for o in var.occurrences)
                        for precision in range(max_precision, -1, -1):
                            rounded = value.quantize(Decimal(1).scaleb(-precision))
                            try:
                                for occ in var.occurrences:
                                    surface(var, occ, str(rounded))
                            except ValueError:
                                continue
                            if rounded <= 0:
                                raise ValueError("non-positive rounded measurement")
                            value = rounded
                            break
                        else:
                            raise ValueError("independent total cannot be represented")
                    for occ in var.occurrences:
                        surface(var, occ, str(value))
                    proposal[var.key] = str(value)
                feasible.append(proposal)
            except ValueError:
                continue
        if not feasible:
            raise ValueError(f"no exact downscaled {dimension} scenario")
        values.update(feasible[round(variant * (len(feasible) - 1) / max(1, variant_count - 1))])
    occupied = {v.value for v in contract.variables if v.kind in {"container", "identifier"}}
    for var in contract.variables:
        if var.kind == "count":
            values[var.key] = str(int(var.value) // common * count_units)
        elif var.kind == "container":
            values[var.key] = generate_container_number(
                owner_and_category=var.value[:4], stream=stream.derive(var.key), excluded=occupied
            )
            occupied.add(values[var.key])
        elif var.kind == "identifier":
            values[var.key] = generate_from_surface_pattern(
                pattern=surface_pattern(var.value),
                stream=stream.derive(var.key),
                additional_excluded=var.value,
            )
    return values


GENERATION_PROMPT = (
    "You generate bounded, coherent Bill of Lading training "
    "variations.\nReturn exactly the requested lexical keys, as "
    "single-line strings. The host owns numeric values, IDs and "
    "rendering.\nGenerate new plausible party names and complete "
    "postal components in the same country and source "
    "granularity.\nWhere several fragments form an address, generate "
    "them together; include each locality/country once across\nthe "
    "complete address. Keep separately printed country/locality "
    "fragments and named-site context coherent.\nPostal values contain "
    "address information only; other party fields remain untouched. "
    "Preserve source-style\npunctuation naturally, without adding "
    "captions, contacts, tax IDs or template markers.\nFor product "
    "wording use a genuine variation/synonym within the EXACT "
    "supplied product/HS/UN family;\npreserve all numeric "
    "specifications, brands/models/chemical identities that anchor "
    "that classification.\nAll source facts outside requested regions "
    "remain fixed. Do not create new goods accounting groups "
    "or\ncontradict frozen shipment quantities, equipment, "
    "temperature, packaging, customs or route "
    "context.\n"
)

REVIEW_PROMPT = (
    "You audit a source-bound synthetic Bill of Lading for KIE "
    "training.\nUse the OCR as the only evidence for labels. Check "
    "complete postal ownership/order/punctuation (line breaks\nbecome "
    "spaces), duplicates, stranded locality fragments, party identity "
    "consistency, goods/classification,\nquantities and units, "
    "repeated containers/seals, placement totals and all "
    "changed-label dependencies.\nThe fixed current V7 target policy "
    "is uppercase human text, full addressLine plus country, no "
    "separate city,\none source-supported goods accounting group, "
    "and reduced omitted fields. Descriptions copy complete reviewed "
    "main product-description blocks and genuine continuations in printed order, "
    "including embedded packing and quantity phrases. Exclude generic loading "
    "introductions and detached tracking/packing passages; separate Marks and "
    "accounting fields remain separate. "
    "Description expressions identify lexical dependencies; the final description "
    "is projected independently from the rendered blocks.\n"
    "Report concrete errors, missing changed dependencies, "
    "impossible/unjustified arithmetic or wrong ownership. Report unresolved "
    "description boundaries with competing interpretations and a recommended "
    "review action instead of guessing.\nDo not "
    "demand omitted carrier/marks/export-reference labels, registry "
    "enrichment, or cosmetic rewriting.\nUnchanged source "
    "imperfections are distinguished from newly introduced defects; "
    "identify both explicitly.\nAn empty findings list means this "
    "review found no defect, not a mathematical proof of every "
    "semantic fact.\n"
)

LEXICAL_REVIEW_PROMPT = (
    "Review new synthetic postal/company/product bundles for training. Each listed region "
    "has a source example and an ownership constraint. Judge the NEW values as plausible "
    "generated content, not as an extraction of the original address. Check full postal "
    "component coverage, coherent fragments, repeated city/country, tax/contact leakage, "
    "country compatibility, named-site dependencies, and product-family/specification "
    "preservation. The host verifies rendering, labels, quantities, IDs and unchanged text "
    "separately. Identify actual defects by variant/key and a specific correction. "
    "Postal deliverability is not required. Give a short substantive summary of your checks."
)


def lexical_review_input(contract: SourceContract, records: list[dict]) -> str:
    lexical = [v for v in contract.variables if v.kind in {"postal", "name", "product"}]
    return (
        "SOURCE-OWNED REGIONS AND MEANINGS\n"
        + "\n".join(
            f"{v.key} ({v.kind}): {v.value}\n{v.meaning}\n"
            f"Required anchors: {', '.join(v.required_literals)}"
            for v in lexical
        )
        + "\nFIXED CONTEXT\n"
        + contract.fixed_context
        + "\nNEW VARIATIONS\n"
        + "\n\n".join(
            f"VARIANT {i + 1}\n" + "\n".join(f"{v.key}: {r['values'][v.key]}" for v in lexical)
            for i, r in enumerate(records)
        )
    )


class Runner:
    def __init__(self, root: Path, config: dict):
        # YAML yields lists/numbers, whereas the nested provider contract uses
        # strict tuples/Decimals. Validate at its JSON wire boundary, as the
        # existing provider loader does; do not disable strict field validation.
        validated = CuratedSynthesisConfig.model_validate_json(json.dumps(config))
        if validated.provider.output_mode != "native":
            raise ValueError("curated synthesis requires native structured output")
        config = validated.model_dump(mode="json")
        self.root, self.config = root, config
        self.output = root / config["output"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider = OpenRouterProviderConfig.model_validate_json(json.dumps(config["provider"]))
        constraints_payload = (root / config["task_constraints"]).read_bytes()
        if digest(constraints_payload) != config["task_constraints_sha256"]:
            raise ValueError("task vocabulary hash differs from the configured authority")
        self.task = get_training_task("bill_of_lading_extraction_v7_reduced").bind_constraints(
            RelationExplicitTaskConstraints.model_validate_json(constraints_payload)
        )
        self.model = None
        self.limiter = asyncio.Semaphore(config["concurrency"])
        self.reserved = Decimal(0)
        self.retry_rate_limited = False
        self.retry_invalid_output = False
        self.spent = sum(
            (
                Decimal(str(json.loads(p.read_text())["costUsd"]))
                for p in (self.output / "calls").glob("*.json")
            ),
            Decimal(0),
        )

    async def call(
        self, stage: str, identity: str, output_type: type[BaseModel], system: str, prompt: str
    ):
        key = digest(
            [
                stage,
                identity,
                output_type.model_json_schema(),
                system,
                prompt,
                self.config["provider"],
            ]
        )
        path = self.output / "calls" / f"{stage}-{key}.json"
        attempt = 1
        while path.exists():
            receipt = json.loads(path.read_text())
            if receipt["output"] is None:
                retry_rejected = (
                    self.retry_rate_limited
                    and receipt["costStatus"] == "request_rejected"
                    and "status_code: 429" in receipt["error"]
                )
                retry_invalid = self.retry_invalid_output and str(receipt["error"]).startswith(
                    "UnexpectedModelBehavior:"
                )
                if retry_rejected or retry_invalid:
                    # Explicit resume permits one new attempt per invocation.
                    # Every failed attempt and its billed usage remain recorded.
                    attempt += 1
                    path = self.output / "calls" / f"{stage}-{key}-attempt{attempt}.json"
                    continue
                raise RuntimeError(f"cached failed call {path}: {receipt['error']}")
            return output_type.model_validate(receipt["output"])
        async with self.limiter:
            if self.model is None:
                self.model = _provider_model(
                    project_root=self.root,
                    environment_file=self.config["environment_file"],
                    provider=self.provider,
                )
            maximum = (
                Decimal(
                    len((system + prompt + json.dumps(output_type.model_json_schema())).encode())
                )
                * self.provider.max_prompt_price_usd_per_million
                + Decimal(self.provider.max_output_tokens)
                * self.provider.max_completion_price_usd_per_million
            ) / 1_000_000
            if self.spent + self.reserved + maximum > Decimal(str(self.config["budget_usd"])):
                raise RuntimeError("paid-request budget reservation exceeds configured cap")
            self.reserved += maximum
            agent = Agent(
                self.model,
                output_type=NativeOutput(output_type, strict=True),
                system_prompt=system,
                model_settings=_provider_settings(self.provider),
                retries=0,
            )
            started, output, error = time.perf_counter(), None, None
            rejected = False
            with capture_run_messages() as messages:
                try:
                    async with asyncio.timeout(self.provider.request_timeout_seconds):
                        result = await agent.run(prompt)
                    output = result.output
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}; cause: {exc.__cause__}"
                    rejected = isinstance(exc, ModelHTTPError) and exc.status_code in {
                        400,
                        401,
                        402,
                        403,
                        404,
                        422,
                        429,
                    }
            responses = [m for m in messages if isinstance(m, ModelResponse)]
            usage = (
                usage_receipt(responses, self.provider.pricing, require_provider_cost=True)
                if responses
                else None
            )
            # A disconnected/timed-out request may have been billed. Reserve its
            # full configured upper bound until provider accounting reconciles it.
            cost = (
                usage.providerReportedCostUsd
                if usage is not None
                else Decimal(0)
                if rejected
                else maximum
            )
            if cost is None:
                raise RuntimeError("provider did not return cost accounting")
            self.spent += cost
            self.reserved -= maximum
            save(
                path,
                {
                    "stage": stage,
                    "identity": identity,
                    "output": output.model_dump(mode="json") if output else None,
                    "error": error,
                    "usage": usage.model_dump(mode="json") if usage else None,
                    "costUsd": str(cost),
                    "costStatus": "provider_reported"
                    if usage
                    else "request_rejected"
                    if rejected
                    else "reserved_unknown",
                    "seconds": time.perf_counter() - started,
                    "promptSha256": digest(prompt),
                    "schemaSha256": digest(output_type.model_json_schema()),
                    "messages": model_messages(messages),
                },
            )
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "id": identity,
                        "status": "ok" if output else "error",
                        "costUsd": str(cost),
                        "totalCostUsd": str(self.spent),
                    }
                ),
                flush=True,
            )
            if output is None:
                raise RuntimeError(error)
            return output

    def _source_contract(
        self, row: dict
    ) -> tuple[SourceContract, CompiledDescriptionBlocks | None]:
        """Check the complete source contract before generation or publication."""
        directory = self.output / "sources" / row["documentId"]
        path = directory / "contract.json"
        if path.exists():
            envelope = json.loads(path.read_text())
            if envelope["sourceSha256"] != digest(row["joinedRawText"].encode()) or envelope[
                "targetSha256"
            ] != digest(row["target"]):
                raise ValueError("cached contract source/target identity changed")
            contract = SourceContract.model_validate(envelope["contract"])
            blocks = compile_description_blocks(
                row["joinedRawText"], row["target"], envelope.get("description_blocks")
            )
        else:
            raise ValueError(f"missing rebased source contract: {path}")
        baseline = {v.key: v.value for v in contract.variables}
        text, target, _ = render(row, contract, baseline, description_blocks=blocks)
        if text != row["joinedRawText"] or target != row["target"]:
            raise ValueError("identity round-trip differs")
        return contract, blocks

    async def source(self, row: dict, stage: str) -> dict:
        sid = row["documentId"]
        directory = self.output / "sources" / sid
        contract, blocks = self._source_contract(row)
        context = (
            "CURRENT TARGET SCALARS\n"
            + "\n".join(f"{p}: {v}" for p, v in flat(row["target"]).items())
            + "\n\nCOMPLETE OCR\n"
            + row["joinedRawText"]
        )
        if stage == "compile":
            return {"documentId": sid, "status": "compiled", "variables": len(contract.variables)}
        if stage == "review":
            review = await self.call(
                "contract-review",
                sid,
                Review,
                REVIEW_PROMPT,
                context + "\nMUTATION CONTRACT\n" + contract.model_dump_json(),
            )
            save(directory / "contract-review.json", review.model_dump(mode="json"))
            return {"documentId": sid, "status": "review" if review.findings else "reviewed"}
        results = []
        lexical = [v for v in contract.variables if v.kind in {"postal", "name", "product"}]
        request = (
            "SOURCE OCR (context only)\n"
            + row["joinedRawText"]
            + "\nFIXED CONTEXT\n"
            + contract.fixed_context
            + "\nREQUESTED LEXICAL REGIONS\n"
            + "\n".join(
                f"{v.key} ({v.kind}): {v.value}\nMeaning: {v.meaning}\n"
                f"Required literal anchors: {', '.join(v.required_literals)}"
                for v in lexical
            )
            + f"\nProduce {self.config['variants_per_source']} variations."
        )
        batch = await self.call("generate-batch", sid, LexicalBatch, GENERATION_PROMPT, request)
        if len(batch.variants) != self.config["variants_per_source"]:
            raise ValueError("wrong number of lexical variations")
        records = []
        for variant in range(self.config["variants_per_source"]):
            sample_id = "syn_v7_" + digest([sid, self.config["seed"], variant])[:24]
            values = scenario_values(
                contract,
                sample_id,
                self.config["seed"],
                variant,
                self.config["variants_per_source"],
            )
            bundle = batch.variants[variant]
            generated = {v.key: v.value for v in bundle.values}
            if len(generated) != len(bundle.values) or set(generated) != {v.key for v in lexical}:
                raise ValueError("lexical output keys differ")
            values.update(generated)
            text, target, proof = render(row, contract, values, description_blocks=blocks)
            result = {
                "documentId": sample_id,
                "sourceDocumentId": sid,
                "joinedRawText": text,
                "joinedRawTextSha256": digest(text.encode()),
                "target": target,
                "values": values,
                "proof": proof,
                "contractSha256": digest(contract.model_dump(mode="json")),
                "seed": self.config["seed"],
                "variant": variant,
                "variantCount": self.config["variants_per_source"],
            }
            records.append(result)
        audit, review_error = None, None
        try:
            audit = await self.call(
                "lexical-review",
                sid,
                Review,
                LEXICAL_REVIEW_PROMPT,
                lexical_review_input(contract, records),
            )
        except RuntimeError as exc:
            # Retain valid generated work, but never reinterpret API failure as
            # a clean review or authorize publication from it.
            review_error = str(exc)
        if audit:
            save(directory / "lexical-review.json", audit.model_dump(mode="json"))
        for result in records:
            result["review"] = audit.model_dump(mode="json") if audit else None
            result["reviewError"] = review_error
            result["status"] = "review" if audit is None or audit.findings else "candidate"
            validate_candidate(row, contract, result, description_blocks=blocks)
            save(self.output / "samples" / f"{result['documentId']}.json", result)
            results.append({"documentId": result["documentId"], "status": result["status"]})
        return {
            "documentId": sid,
            "status": "review" if any(r["status"] == "review" for r in results) else "generated",
            "samples": results,
        }

    async def run(self, stage: str, limit: int | None) -> dict:
        dataset = self.root / self.config["dataset"]
        train = {
            r["documentId"]: r
            for r in map(json.loads, (dataset / "train.jsonl").read_text().splitlines())
        }
        validation = list(map(json.loads, (dataset / "validation.jsonl").read_text().splitlines()))
        selected = self.config["source_ids"][:limit] if limit else self.config["source_ids"]
        if len(selected) != len(set(selected)) or not set(selected) <= train.keys():
            raise ValueError("selected sources must be distinct current training records")
        validation_text = {digest(r["joinedRawText"].encode()) for r in validation}
        for sid in selected:
            if digest(train[sid]["joinedRawText"].encode()) in validation_text:
                raise ValueError("selected source has validation-identical OCR")
        validation_bills = {
            r["target"]["documentPatch"].get("billOfLadingNumber", "").casefold()
            for r in validation
        } - {""}
        for sid in selected:
            bill = train[sid]["target"]["documentPatch"].get("billOfLadingNumber", "").casefold()
            if bill and bill in validation_bills:
                raise ValueError("selected source shares a validation bill identifier")
        if stage in {"validate", "publish"}:
            return self.validate_and_publish(train, selected, publish=stage == "publish")

        # Finish preflight for the complete requested source set before any
        # concurrent worker can spend on a partially ready campaign.
        for sid in selected:
            self._source_contract(train[sid])

        async def process(sid: str):
            try:
                return await self.source(train[sid], stage)
            except Exception as exc:
                return {
                    "documentId": sid,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        started = time.perf_counter()
        results = await asyncio.gather(*(process(sid) for sid in selected))
        summary = {
            "stage": stage,
            "results": results,
            "costUsd": str(self.spent),
            "seconds": time.perf_counter() - started,
            "trainSha256": digest((dataset / "train.jsonl").read_bytes()),
            "validationSha256": digest((dataset / "validation.jsonl").read_bytes()),
        }
        save(self.output / f"{stage}-summary.json", summary)
        return summary

    def validate_and_publish(self, train: dict, selected: list[str], *, publish: bool) -> dict:
        """Require exact expected coverage and explicit pilot review before export."""
        records, failures, seen = [], [], set()
        for sid in selected:
            directory = self.output / "sources" / sid
            contract, blocks = self._source_contract(train[sid])
            approval_path = directory / "adjudication.json"
            if publish and not approval_path.exists():
                failures.append(
                    {"source": sid, "error": "missing independent source/sample adjudication"}
                )
                continue
            approval = json.loads(approval_path.read_text()) if approval_path.exists() else None
            if (
                publish
                and approval
                and (approval.get("decision") != "approved" or not approval.get("rationale"))
            ):
                raise ValueError("publication requires an explicit reasoned approval")
            if approval and approval["contractSha256"] != digest(contract.model_dump(mode="json")):
                raise ValueError("adjudication belongs to a different contract")
            for variant in range(self.config["variants_per_source"]):
                sample_id = "syn_v7_" + digest([sid, self.config["seed"], variant])[:24]
                path = self.output / "samples" / f"{sample_id}.json"
                if not path.exists():
                    failures.append({"sample": sample_id, "error": "missing sample"})
                    continue
                record = json.loads(path.read_text())
                try:
                    validate_candidate(train[sid], contract, record, description_blocks=blocks)
                    if (
                        record["seed"] != self.config["seed"]
                        or record["variant"] != variant
                        or record["documentId"] != sample_id
                    ):
                        raise ValueError("publication scenario differs from configured run")
                    if self.task.canonicalize(record["target"]) != record["target"]:
                        raise ValueError("published target differs from frozen training contract")
                    if approval and approval["sampleSha256"].get(sample_id) != digest(record):
                        raise ValueError("sample changed after adjudication")
                    fingerprint = record["joinedRawTextSha256"]
                    if fingerprint in seen:
                        raise ValueError("duplicate generated OCR")
                    seen.add(fingerprint)
                    records.append(
                        {
                            k: record[k]
                            for k in (
                                "documentId",
                                "sourceDocumentId",
                                "joinedRawText",
                                "joinedRawTextSha256",
                                "target",
                            )
                        }
                    )
                except Exception as exc:
                    failures.append({"sample": sample_id, "error": str(exc)})
        report = {
            "expected": len(selected) * self.config["variants_per_source"],
            "valid": len(records),
            "failures": failures,
            "published": False,
        }
        if publish and not failures and len(records) == report["expected"]:
            path = self.output / "dataset.jsonl"
            payload = "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records
            )
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text(payload)
            temporary.replace(path)
            report.update(published=True, datasetSha256=digest(payload.encode()))
        save(self.output / "publication.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage", choices=("compile", "review", "generate", "validate", "publish"), required=True
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = args.project_root.resolve() / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    # One process owns this run's spend ledger and atomic publication at a time.
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = asyncio.run(
            Runner(args.project_root.resolve(), config).run(args.stage, args.limit)
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("failures") or any(
        r["status"] in {"error", "review"} for r in summary.get("results", [])
    ):
        sys.exit(5)


if __name__ == "__main__":
    main()
