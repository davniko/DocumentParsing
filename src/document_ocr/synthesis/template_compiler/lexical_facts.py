"""Host-owned commodity wording and contacts, before target acceptance.

The model owns organization/address language, not chemical identity, equipment
facts, telephone numbering, or item allocation. Split cargo descriptions require
a reviewed source-pinned recipe; a missing recipe is an error, never a fallback.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_from_surface_pattern,
    surface_pattern,
)

from . import (
    cargo_identifiers,
    contact_values,
    lexical_partitions,
    package_equations,
    package_prose,
    temperature_prose,
)
from . import complete_targets as targets
from . import descendant as render
from .cargo_scenarios import CargoScenario
from .models import CertifiedSemanticTemplate
from .route_projection import RouteProjection


class FragmentRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    role: Literal[
        "item",
        "identity",
        "reference",
        "labelled_reference",
        "detail",
        "brand",
        "literal",
        "source_locked_item",
    ]
    source: str
    identity_indices: tuple[int, ...] = ()
    identity_anchor: str | None = None


class ReferenceRecipe(BaseModel):
    """Reviewed semantic role for a source-pinned standalone mark/reference."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    role: Literal[
        "source_shaped_code",
        "source_shaped_code_list",
        "shipment_mark",
        "date_year_locked_code",
    ]
    source: str
    date_path: Literal["documentPatch.issueDate", "documentPatch.shippedOnBoardDate"] | None = None
    year_position: Literal["prefix", "suffix"] | None = None
    caption: Literal["INVOICE", "DU-E"] | None = None

    @model_validator(mode="after")
    def dated_fields_are_atomic(self) -> ReferenceRecipe:
        dated = self.role == "date_year_locked_code"
        if dated != all(
            value is not None for value in (self.date_path, self.year_position, self.caption)
        ):
            raise ValueError("dated export-reference role requires date, position, and caption")
        if not dated and any(
            value is not None for value in (self.date_path, self.year_position, self.caption)
        ):
            raise ValueError("non-dated reference cannot declare a date-year contract")
        if dated and (self.caption, self.year_position) not in {
            ("INVOICE", "suffix"),
            ("DU-E", "prefix"),
        }:
            raise ValueError("dated reference caption and year position disagree")
        return self


class AuxiliaryReferenceRecipe(BaseModel):
    """Source-proved immutable frame around a mutable source-only identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    role: Literal["source_frame_locked_identifier"]
    source: str
    fixed_prefix: str
    fixed_suffix: str
    required_date_year: int
    date_paths: tuple[
        Literal["documentPatch.issueDate", "documentPatch.shippedOnBoardDate"], ...
    ]

    @model_validator(mode="after")
    def has_mutable_source_middle(self) -> AuxiliaryReferenceRecipe:
        if (
            not self.fixed_prefix
            or not self.fixed_suffix
            or not self.source.startswith(self.fixed_prefix)
            or not self.source.endswith(self.fixed_suffix)
            or len(self.source) <= len(self.fixed_prefix) + len(self.fixed_suffix)
            or not self.date_paths
            or len(set(self.date_paths)) != len(self.date_paths)
            or self.required_date_year < 1
            or self.required_date_year > 9999
        ):
            raise ValueError("source-frame identifier lacks a proved mutable middle and dates")
        return self


class CargoLexicalContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_template_sha256: str
    lexical_contract_sha256: str
    fragments: dict[str, FragmentRecipe]
    references: dict[str, ReferenceRecipe] = Field(default_factory=dict)
    auxiliary_references: dict[str, AuxiliaryReferenceRecipe] = Field(default_factory=dict)


_MARK_PATH = re.compile(r"documentPatch\.cargoGroups\[\d+\]\.marksAndNumbers\[\d+\]")
_CARGO_NON_DESCRIPTION_PATH = re.compile(
    r"documentPatch\.cargoGroups\[(\d+)\]\."
    r"(?:marksAndNumbers|additionalInformation|handlingInstructions)\[\d+\]"
)
_HOST_PRODUCT_REFERENCE = re.compile(r"\bproduct reference\s+([A-F0-9]{12})\b", re.I)
_LABELLED_REFERENCE = re.compile(
    r"^(?P<prefix>[A-Za-z][A-Za-z ]{1,24}\s+NO\s*:\s*)"
    r"(?P<code>[A-Za-z0-9][A-Za-z0-9/-]{3,23})$"
)
_BRAND_FRAGMENT = re.compile(
    r"^(?P<caption>BRAND\s*:\s*)(?P<name>[A-Za-z][A-Za-z0-9 .&'-]{1,39})$", re.I
)
_BRAND_CONSONANTS = "BCDFGHJKLMNPRSTV"
_BRAND_VOWELS = "AEIOU"


def _word_surface(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", value.casefold()))


def _shared_description_surface(value: str, description: str) -> bool:
    left, right = _word_surface(value), _word_surface(description)
    if not left or not right:
        return False
    return f" {left} " in f" {right} " or f" {right} " in f" {left} "


def _contains_description(value: str, description: str) -> bool:
    mark, goods = _word_surface(value), _word_surface(description)
    return bool(goods and f" {goods} " in f" {mark} ")


def validate_generated_cargo_roles(
    *,
    fields: Sequence[Mapping[str, Any]],
    generated: Mapping[str, str],
    source_target: Mapping[str, Any],
    sampled_target: Mapping[str, Any],
) -> None:
    """Reject model text that copies host goods into an unrelated cargo role.

    Source-visible sharing is allowed. A source with distinct description and
    mark roles does not authorize a newly generated mark, qualifier, or handling
    instruction to duplicate the host's sampled product wording or identifier.
    """
    source_groups = source_target.get("documentPatch", {}).get("cargoGroups", ())
    sampled_groups = sampled_target.get("documentPatch", {}).get("cargoGroups", ())
    for field in fields:
        affected = [
            (path, int(match[1]))
            for path in field["paths"]
            if (match := _CARGO_NON_DESCRIPTION_PATH.fullmatch(path))
        ]
        if not affected:
            continue
        key = field["key"]
        if key not in generated:
            raise ValueError("cargo role validation lacks the generated field: " + key)
        value = targets.assemble_lexical_value(field, generated[key])
        if not isinstance(value, str):
            raise ValueError("generated cargo role is not textual: " + key)
        if len(source_groups) != len(sampled_groups):
            raise ValueError("source/sampled cargo group cardinality differs")
        for path, group_index in affected:
            if group_index >= len(source_groups):
                raise ValueError("cargo role group index differs: " + path)
            source_value = render._resolve_path(source_target, path)
            if not isinstance(source_value, str):
                raise ValueError("source cargo role is not textual: " + path)
            generated_references = {
                match.upper() for match in _HOST_PRODUCT_REFERENCE.findall(value)
            }
            for owner_index, (source_group, sampled_group) in enumerate(
                zip(source_groups, sampled_groups, strict=True)
            ):
                source_description = source_group.get("description")
                sampled_description = sampled_group.get("description")
                if not isinstance(source_description, str) or not isinstance(
                    sampled_description, str
                ):
                    raise ValueError("cargo role lacks its description owner: " + path)
                sampled_references = {
                    match.upper() for match in _HOST_PRODUCT_REFERENCE.findall(sampled_description)
                }
                copied_reference = bool(generated_references & sampled_references)
                copied_description = _contains_description(value, sampled_description)
                source_shared = _shared_description_surface(source_value, source_description)
                if (copied_reference or copied_description) and not source_shared:
                    raise ValueError(
                        "generated cargo role copies host-owned goods description without "
                        f"source-shared ownership: {path} from group {owner_index}"
                    )


def _validate_literal_fragment(field: Mapping[str, Any], recipe: FragmentRecipe) -> None:
    """Only a proved form caption may survive a change of commodity identity.

    A model-reviewed literal is not a compatibility contract for a retained
    product, grade, composition or technical specification. The currently
    supported split caption is PART followed by its printed number caption;
    every occurrence must prove that frame independently.
    """
    if recipe.role != "literal":
        return
    contexts = field.get("cargoFragment", {}).get("contexts", ())
    if (
        recipe.source.strip().casefold() == "part"
        and contexts
        and all(
            context.get("sourceSlot", "").strip().casefold() == "part"
            and re.match(r"\s+NO\s*\.:", context.get("after", ""), re.I)
            for context in contexts
        )
    ):
        return
    raise ValueError("retained cargo literal lacks a proved invariant caption: " + field["key"])


def _validate_labelled_reference_fragment(
    field: Mapping[str, Any], recipe: FragmentRecipe
) -> re.Match[str] | None:
    """A reviewed split product reference must own one complete printed slot."""
    if recipe.role != "labelled_reference":
        return None
    fragment = field.get("cargoFragment") or {}
    match = _LABELLED_REFERENCE.fullmatch(recipe.source)
    index = fragment.get("index", 0)
    parts = fragment.get("sourceParts") or ()
    if (
        match is None
        or recipe.identity_indices
        or not recipe.identity_anchor
        or not re.fullmatch(r"[A-Za-z]+", recipe.identity_anchor)
        or index < 1
        or index >= len(parts)
        or parts[index] != recipe.source
        or not re.search(
            r"\b" + re.escape(recipe.identity_anchor) + r"\b",
            " ".join(parts[:index]),
            re.I,
        )
        or len(fragment.get("targetPaths", ())) != 1
        or not re.fullmatch(
            r"documentPatch\.cargoGroups\[\d+\]\.description",
            fragment["targetPaths"][0],
        )
        or not fragment.get("contexts")
        or any(context.get("sourceSlot") != recipe.source for context in fragment["contexts"])
    ):
        raise ValueError("labelled cargo reference lacks a source-owned description slot")
    return match


def _validate_brand_fragment(
    field: Mapping[str, Any], recipe: FragmentRecipe
) -> re.Match[str] | None:
    """Admit only a separately printed BRAND fragment owned by one goods description."""
    if recipe.role != "brand":
        return None
    fragment = field.get("cargoFragment") or {}
    match = _BRAND_FRAGMENT.fullmatch(recipe.source)
    index = fragment.get("index")
    parts = fragment.get("sourceParts") or ()
    contexts = fragment.get("contexts") or ()
    if (
        match is None
        or recipe.identity_indices
        or recipe.identity_anchor is not None
        or not isinstance(index, int)
        or index < 1
        or index >= len(parts)
        or parts[index] != recipe.source
        or len(fragment.get("targetPaths", ())) != 1
        or not re.fullmatch(
            r"documentPatch\.cargoGroups\[\d+\]\.description",
            fragment["targetPaths"][0],
        )
        or not contexts
        or any(context.get("sourceSlot") != recipe.source for context in contexts)
    ):
        raise ValueError("brand fragment lacks one source-owned description slot")
    return match


def _validate_source_locked_item(field: Mapping[str, Any], recipe: FragmentRecipe) -> None:
    """Prove an immutable commodity fragment is one complete printed source item.

    This is an explicit reviewed contract, not a recovery from an incompatible
    model response. The source/template and lexical-plan hashes are checked by
    ``prepare`` before the host emits the exact source-owned item wording.
    """
    if recipe.role != "source_locked_item":
        return
    fragment = field.get("cargoFragment") or {}
    index = fragment.get("index")
    parts = fragment.get("sourceParts") or ()
    contexts = fragment.get("contexts") or ()
    if (
        recipe.identity_indices
        or recipe.identity_anchor is not None
        or not isinstance(index, int)
        or index < 0
        or index >= len(parts)
        or parts[index] != recipe.source
        or field.get("source") != recipe.source
        or len(fragment.get("targetPaths", ())) != 1
        or not re.fullmatch(
            r"documentPatch\.cargoGroups\[\d+\]\.description",
            fragment["targetPaths"][0],
        )
        or not contexts
        or any(
            _word_surface(context.get("sourceSlot", "")) != _word_surface(recipe.source)
            for context in contexts
        )
    ):
        raise ValueError("source-locked cargo item lacks a complete printed source fragment")


def _synthetic_brand(*, sample_id: str, key: str, seed: int) -> str:
    """Create a source-independent, pronounceable six-letter private brand token."""
    stream = DeterministicStream(seed, "cargo-brand-v1", sample_id).derive(key)
    return "".join(
        _BRAND_CONSONANTS[stream.randbelow(len(_BRAND_CONSONANTS), counter=pair * 2)]
        + _BRAND_VOWELS[stream.randbelow(len(_BRAND_VOWELS), counter=pair * 2 + 1)]
        for pair in range(3)
    )


def require_cargo_recipes(
    fields: Sequence[Mapping[str, Any]], contract: CargoLexicalContract | None
) -> None:
    """Reject missing source annotations before stochastic cargo draws.

    Candidate retries can solve physical incompatibility, but cannot supply a
    missing fragment owner or change its source-pinned recipe. This check uses
    only source contracts; wording length and goods identity checks remain in
    ``prepare`` because those really do depend on the sampled candidate.
    """
    for field in fields:
        recipe = contract.fragments.get(field["key"]) if contract is not None else None
        if recipe is not None and recipe.role == "brand":
            if recipe.source != field["source"]:
                raise ValueError("cargo fragment recipe source changed")
            _validate_brand_fragment(field, recipe)
        paths = field.get("cargoFragment", {}).get("targetPaths", field["paths"])
        if not any(
            re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.description", p) for p in paths
        ):
            continue
        if "hostAssembly" in field:
            raise ValueError("fixed commodity literal requires a goods-specific lexical contract")
        if "cargoFragment" not in field:
            continue
        key = field["key"]
        if contract is None or key not in contract.fragments:
            raise ValueError("split cargo description lacks a reviewed lexical recipe: " + key)
        if contract.fragments[key].source != field["source"]:
            raise ValueError("cargo fragment recipe source changed")
        _validate_literal_fragment(field, contract.fragments[key])
        _validate_labelled_reference_fragment(field, contract.fragments[key])
        _validate_brand_fragment(field, contract.fragments[key])
        _validate_source_locked_item(field, contract.fragments[key])


def validate_source_locked_items(
    fields: Sequence[Mapping[str, Any]],
    contract: CargoLexicalContract | None,
    auxiliary: Mapping[str, str],
    target: Mapping[str, Any],
) -> None:
    """Fail publication if an accepted cargo item differs from its pinned source role."""
    if contract is None:
        return
    for field in fields:
        recipe = contract.fragments.get(field["key"])
        if recipe is None or recipe.role != "source_locked_item":
            continue
        _validate_source_locked_item(field, recipe)
        if auxiliary.get(field["auxiliaryKey"]) != recipe.source:
            raise ValueError("source-locked cargo item changed after lexical generation")
        for path in field["cargoFragment"]["targetPaths"]:
            description = render._resolve_path(target, path)
            if not isinstance(description, str) or recipe.source not in description:
                raise ValueError("source-locked cargo item is absent from the accepted target")


def source_locked_description_paths(
    fields: Sequence[Mapping[str, Any]], contract: CargoLexicalContract | None
) -> frozenset[str]:
    """Exempt variation only when every printed fragment of a description is locked."""
    if contract is None:
        return frozenset()
    parts: dict[str, list[bool]] = {}
    for field in fields:
        fragment = field.get("cargoFragment")
        if fragment is None:
            continue
        paths = fragment.get("targetPaths") or ()
        recipe = contract.fragments.get(field["key"])
        for path in paths:
            if re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.description", path):
                parts.setdefault(path, []).append(
                    recipe is not None and recipe.role == "source_locked_item"
                )
    return frozenset(path for path, owned in parts.items() if owned and all(owned))


def validate_dated_references(
    fields: Sequence[Mapping[str, Any]],
    contract: CargoLexicalContract | None,
    target: Mapping[str, Any],
) -> None:
    """Keep accepted invoice/export-code year bytes tied to the structured date."""
    if contract is None:
        return
    for field in fields:
        recipe = contract.references.get(field["key"])
        if recipe is None or recipe.role != "date_year_locked_code":
            continue
        if recipe.date_path is None or recipe.year_position is None:
            raise ValueError("dated reference is missing its date-year contract")
        year = render._resolve_path(target, recipe.date_path)
        if not isinstance(year, str):
            raise ValueError("accepted dated reference has no structured date")
        expected = f"{date.fromisoformat(year).year % 100:02d}"
        for path in field["paths"]:
            value = render._resolve_path(target, path)
            if (
                not isinstance(value, str)
                or (
                    recipe.caption == "INVOICE" and re.fullmatch(r"[A-Z]\d{3}/\d{2}", value) is None
                )
                or (recipe.caption == "DU-E" and re.fullmatch(r"\d{2}BR\d{9}-\d", value) is None)
            ):
                raise ValueError("accepted dated export reference changed its source grammar")
            actual = value[:2] if recipe.year_position == "prefix" else value[-2:]
            if actual != expected:
                raise ValueError("accepted dated export reference contradicts its labelled year")
            if recipe.caption == "DU-E" and value[-1] != _due_check_digit(value[:2], value[4:-2]):
                raise ValueError("accepted DU-E reference has an invalid check digit")


def _due_check_digit(year: str, serial: str) -> str:
    """Brazil's published DU-E modulo-11 check digit for YYBR+9-digit serial.

    Siscomex Export FAQ, section 3.25, gives weights 12..2 and the 0/1
    remainder convention.
    """
    if re.fullmatch(r"\d{2}", year) is None or re.fullmatch(r"\d{9}", serial) is None:
        raise ValueError("DU-E year and serial do not have their published widths")
    weighted = sum(
        int(digit) * weight
        for digit, weight in zip(year + serial, range(12, 1, -1), strict=True)
    )
    remainder = weighted % 11
    return str(0 if remainder in (0, 1) else 11 - remainder)


def validate_source_locked_auxiliary(
    source: targets.SourceTemplate,
    contract: CargoLexicalContract | None,
    auxiliary: Mapping[str, str],
    target: Mapping[str, Any],
) -> None:
    """Prove each source-only identifier retains its printed immutable frame."""
    if contract is None or not contract.auxiliary_references:
        return
    if (
        contract.source_template_sha256 != source.original_template_sha256
        or contract.lexical_contract_sha256
        != sha256_bytes(canonical_json_bytes(targets.lexical_contract(source)))
    ):
        raise ValueError("source-frame identifier contract differs from its pinned source")
    validate_source_locked_auxiliary_receipt(
        source_target=source.target,
        template=source.template,
        auxiliary=auxiliary,
        target=target,
        recipes=contract.auxiliary_references,
    )


def validate_source_locked_auxiliary_receipt(
    *,
    source_target: Mapping[str, Any],
    template: CertifiedSemanticTemplate,
    auxiliary: Mapping[str, str],
    target: Mapping[str, Any],
    recipes: Mapping[str, AuxiliaryReferenceRecipe],
) -> None:
    """Reprove serialized source-only recipes at the publication boundary."""
    for key, recipe in recipes.items():
        bindings = [binding for binding in template.bindings if binding.logical_key == key]
        if (
            len(bindings) != 1
            or bindings[0].target_paths
            or bindings[0].value_kind != "identifier"
            or not bindings[0].occurrences
            or any(slot.source_text != recipe.source for slot in bindings[0].occurrences)
        ):
            raise ValueError("source-frame identifier does not own one source-only binding: " + key)
        for path in recipe.date_paths:
            source_date = render._resolve_path(source_target, path)
            sampled_date = render._resolve_path(target, path)
            if (
                not isinstance(source_date, str)
                or not isinstance(sampled_date, str)
                or date.fromisoformat(source_date).year != recipe.required_date_year
                or date.fromisoformat(sampled_date).year != recipe.required_date_year
            ):
                raise ValueError(
                    "source-frame identifier is not proved for sampled date year: " + key
                )
        old_middle = recipe.source[
            len(recipe.fixed_prefix) : -len(recipe.fixed_suffix)
        ]
        candidate = auxiliary.get(key)
        if (
            candidate is None
            or not candidate.startswith(recipe.fixed_prefix)
            or not candidate.endswith(recipe.fixed_suffix)
            or candidate == recipe.source
            or surface_pattern(candidate[len(recipe.fixed_prefix) : -len(recipe.fixed_suffix)])
            != surface_pattern(old_middle)
        ):
            raise ValueError("source-frame identifier changed its fixed source grammar: " + key)


def source_locked_auxiliary(
    source: targets.SourceTemplate,
    contract: CargoLexicalContract | None,
    target: Mapping[str, Any],
    *,
    sample_id: str,
    seed: int,
) -> dict[str, str]:
    """Generate only the source-proved mutable middle of a source-only identifier."""
    if contract is None or not contract.auxiliary_references:
        return {}
    generated = {}
    for key, recipe in contract.auxiliary_references.items():
        old_middle = recipe.source[
            len(recipe.fixed_prefix) : -len(recipe.fixed_suffix)
        ]
        middle = generate_from_surface_pattern(
            pattern=surface_pattern(old_middle),
            stream=DeterministicStream(seed, "source-frame-auxiliary-v1", sample_id).derive(key),
            additional_excluded=old_middle,
        )
        generated[key] = recipe.fixed_prefix + middle + recipe.fixed_suffix
    validate_source_locked_auxiliary(source, contract, generated, target)
    return generated


@dataclass(frozen=True)
class HostLexicalPlan:
    values: dict[str, str]
    evidence: dict[str, Any]

    def merge(self, generated: Mapping[str, str]) -> dict[str, str]:
        if self.values.keys() & generated.keys():
            raise ValueError("provider attempted to overwrite host-owned lexical facts")
        return {**generated, **self.values}

    def preview(
        self,
        source: targets.SourceTemplate,
        target: Mapping[str, Any],
        fields: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        result = deepcopy(dict(target))
        auxiliary = {}
        for field in fields:
            if field["key"] not in self.values:
                # The preview retains ungenerated prose, but declared context
                # is already sampled. Keep that context consistent for the
                # renderability check; this is never published linguistic output.
                for path in field["paths"] if field.get("sampledContext") else ():
                    value = render._resolve_path(result, path)
                    spans = render._token_spans(value)
                    tokens = tuple(token for token, _, _ in spans)
                    edits = []
                    for old, new in field["sampledContext"].items():
                        expected = tuple(token for token, _, _ in render._token_spans(old))
                        matches = [
                            i
                            for i in range(len(tokens) - len(expected) + 1)
                            if tokens[i : i + len(expected)] == expected
                        ]
                        if len(matches) != 1:
                            raise ValueError("preview context lacks unique declared occurrence")
                        i = matches[0]
                        edits.append((spans[i][1], spans[i + len(expected) - 1][2], new))
                    for start, end, text in sorted(edits, reverse=True):
                        value = value[:start] + text + value[end:]
                    targets._set(result, path, value)
                continue
            value = targets.assemble_lexical_value(field, self.values[field["key"]])
            for path in field["paths"]:
                targets._set(result, path, value)
            if "cargoFragment" in field:
                auxiliary[field["auxiliaryKey"]] = value
        if auxiliary:
            for path, value in lexical_partitions.assembled_targets(
                source.template, auxiliary
            ).items():
                targets._set(result, path, value)
        from . import labelled_context

        for path, value in labelled_context.generated_values(source, result).items():
            targets._set(result, path, value)
        return result


def _reference(sample_id: str, key: str, seed: int) -> str:
    return (
        DeterministicStream(seed, "cargo-commercial-reference-v1", sample_id)
        .derive(key)
        .bytes(counter=0, length=6)
        .hex()
        .upper()
    )


def _dated_reference(
    *,
    source: targets.SourceTemplate,
    target: Mapping[str, Any] | None,
    field: Mapping[str, Any],
    recipe: ReferenceRecipe,
    sample_id: str,
    seed: int,
) -> str:
    """Generate a source-proved invoice/export code coupled to its printed year."""
    if (
        target is None
        or recipe.date_path is None
        or recipe.year_position is None
        or recipe.caption is None
    ):
        raise ValueError("dated reference requires the proposed structured date")
    if len(field["paths"]) != 1 or not re.fullmatch(
        r"documentPatch\.forwardingAndExportReferences\[\d+\]", field["paths"][0]
    ):
        raise ValueError("dated export reference does not own a forwarding reference label")
    old = recipe.source
    if (recipe.caption == "INVOICE" and re.fullmatch(r"[A-Z]\d{3}/\d{2}", old) is None) or (
        recipe.caption == "DU-E" and re.fullmatch(r"\d{2}BR\d{9}-\d", old) is None
    ):
        raise ValueError("dated reference lacks its source-proven code grammar")
    bindings = [
        binding for binding in source.template.bindings if field["paths"][0] in binding.target_paths
    ]
    if len(bindings) != 1 or any(
        slot.source_text != old
        or not source.source[source.source.rfind(b"\n", 0, slot.byte_start) + 1 : slot.byte_start]
        .decode("utf-8")
        .rstrip()
        .endswith(recipe.caption + ":")
        for slot in bindings[0].occurrences
    ):
        raise ValueError("dated reference caption is not adjacent to its source binding")
    source_date = render._resolve_path(source.target, recipe.date_path)
    sampled_date = render._resolve_path(target, recipe.date_path)
    if not isinstance(source_date, str) or not isinstance(sampled_date, str):
        raise ValueError("dated reference year lacks a structured date")
    source_year = f"{date.fromisoformat(source_date).year % 100:02d}"
    sampled_year = f"{date.fromisoformat(sampled_date).year % 100:02d}"
    printed_year = old[:2] if recipe.year_position == "prefix" else old[-2:]
    if source_year != printed_year:
        raise ValueError("source reference year contradicts the labelled date")
    if recipe.caption == "DU-E" and old[-1] != _due_check_digit(old[:2], old[4:-2]):
        raise ValueError("source DU-E reference has an invalid published check digit")
    mutable_source = old[4:-2] if recipe.caption == "DU-E" else old[:-2]
    generated = generate_from_surface_pattern(
        pattern=surface_pattern(mutable_source),
        stream=DeterministicStream(seed, "dated-export-reference-v1", sample_id).derive(
            field["key"]
        ),
        additional_excluded=mutable_source,
    )
    candidate = (
        sampled_year + "BR" + generated + "-" + _due_check_digit(sampled_year, generated)
        if recipe.caption == "DU-E"
        else generated + sampled_year
    )
    if candidate == old:
        raise ValueError("dated reference did not change its source identifier")
    return candidate


def _labelled_reference_code(old: str, *, sample_id: str, key: str, seed: int) -> str:
    """Draw one unbiased HMAC byte stream for a reviewed short source-shaped code.

    The generic pattern generator derives a separate stream for every character.
    A labelled product code is bounded to 24 characters, so drawing them together
    avoids making deterministic reference generation a per-character hot path.
    """
    stream = DeterministicStream(seed, "cargo-labelled-reference-v1", sample_id).derive(key)
    alphabets = {
        "upper": string.ascii_uppercase,
        "lower": string.ascii_lowercase,
        "digit": string.digits,
    }
    for attempt in range(10_000):
        length = max(32, len(old) * 2)
        while True:
            raw = stream.bytes(counter=attempt * 4, length=length)
            cursor = 0
            output = []
            for character in old:
                if character in string.ascii_uppercase:
                    alphabet = alphabets["upper"]
                elif character in string.ascii_lowercase:
                    alphabet = alphabets["lower"]
                elif character in string.digits:
                    alphabet = alphabets["digit"]
                else:
                    output.append(character)
                    continue
                limit = 256 - 256 % len(alphabet)
                while cursor < len(raw) and raw[cursor] >= limit:
                    cursor += 1
                if cursor == len(raw):
                    length *= 2
                    break
                output.append(alphabet[raw[cursor] % len(alphabet)])
                cursor += 1
            else:
                candidate = "".join(output)
                if candidate != old:
                    return candidate
                break
    raise RuntimeError("labelled cargo reference space exhausted by source exclusion")


def _definitions(field: Mapping[str, Any], scenario: CargoScenario) -> list[str]:
    paths = field.get("cargoFragment", {}).get("targetPaths", field["paths"])
    indices = {
        int(match[1])
        for path in paths
        if (match := re.fullmatch(r"documentPatch\.cargoGroups\[(\d+)\]\.description", path))
    }
    groups = scenario.target["documentPatch"].get("cargoGroups", [])
    return [
        identity.get("properShippingName") or identity["requiredDescription"]
        for i in sorted(indices)
        for identity in scenario.identities[groups[i]["groupId"]]
    ]


def _package_mentions(source: targets.SourceTemplate, target: Mapping[str, Any], path: str) -> str:
    """Retain explicitly proved count/noun assertions, not arbitrary source prose."""
    old = render._resolve_path(source.target, path)
    clauses = []
    for mention in package_prose.mentions(source.target, path, old):
        quantities = [render._resolve_path(target, p) for p in mention.quantity_paths]
        if not mention.aggregate and len(set(quantities)) != 1:
            raise ValueError("cargo description has ambiguous package quantity ownership")
        quantity = sum(quantities) if mention.aggregate else quantities[0]
        kinds = {
            render._resolve_path(target, p.removesuffix("quantity") + "typeCategory")
            for p in mention.quantity_paths
        }
        if len(kinds) != 1:
            raise ValueError("one printed package count cannot express mixed package kinds")
        clauses.append(f"{quantity} {render._package_surface(kinds.pop())}")
    return "; ".join(dict.fromkeys(clauses))


def prepare(
    source: targets.SourceTemplate,
    fields: Sequence[Mapping[str, Any]],
    *,
    scenario: CargoScenario | None,
    projection: RouteProjection | None,
    contract: CargoLexicalContract | None,
    sample_id: str,
    seed: int,
    proposed_target: Mapping[str, Any] | None = None,
) -> HostLexicalPlan:
    if contract is not None and (
        contract.source_template_sha256 != source.original_template_sha256
        or contract.lexical_contract_sha256
        != sha256_bytes(canonical_json_bytes(targets.lexical_contract(source)))
    ):
        raise ValueError("cargo lexical contract differs from its pinned source")
    if contract is not None:
        field_keys = {field["key"] for field in fields}
        if not set(contract.references).issubset(field_keys):
            raise ValueError("reviewed reference contract names an absent field")
        if not set(contract.fragments).issubset(field_keys):
            raise ValueError("reviewed cargo fragment contract names an absent field")
    if contract is not None:
        require_cargo_recipes(fields, contract)
    values: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    references = (
        cargo_identifiers.generate_mark_references(
            source.target,
            source.template,
            DeterministicStream(seed, "host-mark-references-v1", sample_id),
            reviewed_lists={
                path: contract.references[field["key"]].source
                for field in fields
                if contract is not None
                and field["key"] in contract.references
                and contract.references[field["key"]].role == "source_shaped_code_list"
                for path in field["paths"]
            },
        )
        if scenario is not None
        else {}
    )
    ranges = (
        package_equations.mark_ranges(source.target, scenario.target)
        if scenario is not None
        else {}
    )
    overpacks = (
        package_equations.overpack_surfaces(source.target, scenario.target)
        if scenario is not None
        else {}
    )
    origin_marks = {}
    temperatures = (
        temperature_prose.generate(source.template, source.target, scenario.target)
        if scenario is not None
        else {}
    )
    if scenario is not None:
        for binding in source.template.bindings:
            origins = [
                p
                for p in binding.target_paths
                if re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.origin\.name", p)
            ]
            marks = [
                p
                for p in binding.target_paths
                if re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.marksAndNumbers\[\d+\]", p)
            ]
            if len(origins) != 1 or not marks:
                continue
            old = render._resolve_path(source.target, origins[0])
            new = render._resolve_path(scenario.target, origins[0])
            for path in marks:
                text = render._resolve_path(source.target, path)
                match = re.fullmatch(r"(?P<prefix>MADE\s+IN\s+)" + re.escape(old), text, re.I)
                if match:
                    origin_marks[path] = match["prefix"] + render._case_like(old, new)
    identifier_dependencies = {
        key
        for binding in source.template.bindings
        if binding.derivation == "same_as_binding" and binding.value_kind == "identifier"
        for key in binding.dependency_bindings
    }
    anonymous = scenario.receipt.get("anonymousEquipment") if scenario is not None else None
    for field in fields:
        key = field["key"]
        if contract is not None and key in contract.fragments:
            recipe = contract.fragments[key]
            if recipe.role == "source_locked_item":
                _validate_source_locked_item(field, recipe)
                values[key] = recipe.source
                evidence[key] = dict(kind="source_locked_cargo_item", source=recipe.source)
                continue
        if (
            anonymous is not None
            and field.get("auxiliaryKey") in anonymous["sourceEquipmentBindings"]
        ):
            values[key] = field["source"]
            evidence[key] = dict(kind="proved_retained_anonymous_inventory", **anonymous)
            continue
        if "cargoIdentityPath" in field:
            from .cargo_identity_derivations import description

            if scenario is None:
                raise ValueError("commodity alias generation requires the complete goods scenario")
            values[key] = description(field["cargoIdentityPath"], scenario)
            evidence[key] = dict(
                kind="registry_owned_commodity_alias", owner=field["cargoIdentityPath"]
            )
            continue
        if field["paths"] and all(p in temperatures for p in field["paths"]):
            generated = {temperatures[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("temperature prose has incompatible lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="compiled_temperature_dependency", paths=field["paths"])
            continue
        if contract is not None and key in contract.references:
            reference_recipe = contract.references[key]
            if (
                field["source"] != reference_recipe.source
                or not field["paths"]
                or "hostAssembly" in field
                or "cargoFragment" in field
                or not all(
                    re.fullmatch(
                        r"documentPatch\.(?:cargoGroups\[\d+\]\.marksAndNumbers|forwardingAndExportReferences)\[\d+\]",
                        path,
                    )
                    for path in field["paths"]
                )
            ):
                raise ValueError("reviewed reference has incompatible source/field ownership")
            if reference_recipe.role == "source_shaped_code":
                if not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9/.-]*", reference_recipe.source
                ) or not re.search(r"\d", reference_recipe.source):
                    raise ValueError("reviewed reference is not an opaque source-shaped code")
                value = generate_from_surface_pattern(
                    pattern=surface_pattern(reference_recipe.source),
                    stream=DeterministicStream(seed, "reviewed-reference-v1", sample_id).derive(
                        reference_recipe.source
                    ),
                    additional_excluded=reference_recipe.source,
                )
            elif reference_recipe.role == "date_year_locked_code":
                value = _dated_reference(
                    source=source,
                    target=proposed_target,
                    field=field,
                    recipe=reference_recipe,
                    sample_id=sample_id,
                    seed=seed,
                )
            elif reference_recipe.role == "source_shaped_code_list":
                candidates = {references[path] for path in field["paths"]}
                if len(candidates) != 1:
                    raise ValueError("reviewed reference list has inconsistent shared ownership")
                value = candidates.pop()
            else:
                value = "MARK-" + _reference(sample_id, key, seed)
            if len(re.findall(r"[A-Za-z0-9]+", value)) < max(
                (c.get("minimumWords", 1) for c in field["constraints"]), default=1
            ):
                raise ValueError("reviewed reference cannot fill its certified text segments")
            values[key] = value
            evidence[key] = dict(
                kind="reviewed_reference",
                role=reference_recipe.role,
                source=reference_recipe.source,
            )
            continue
        if field["paths"] and all(p in origin_marks for p in field["paths"]):
            generated = {origin_marks[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("manufacturing-origin mark has incompatible lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="compiled_manufacturing_origin", paths=field["paths"])
            continue
        if field["paths"] and all(p in overpacks for p in field["paths"]):
            generated = {overpacks[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("overpack surfaces have incompatible shared lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="source_proven_overpack_levels", paths=field["paths"])
            continue
        if field.get("hostRangeIdentity"):
            if "hostAssembly" not in field or not field["paths"]:
                raise ValueError("host shipping-mark identity lacks its proven range frame")
            values[key] = "MARK-" + _reference(sample_id, key, seed)
            evidence[key] = dict(
                kind="host_shipping_mark_and_range",
                paths=field["paths"],
                frame=field["hostAssembly"],
            )
            continue
        if field["paths"] and all(p in references for p in field["paths"]):
            generated = {references[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("mark references have incompatible shared lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="source_shaped_mark_reference", paths=field["paths"])
            continue
        if field["paths"] and all(p in ranges for p in field["paths"]):
            values_for_field = {ranges[p] for p in field["paths"]}
            if len(values_for_field) != 1 or "hostAssembly" in field:
                raise ValueError("package ranges have incompatible shared lexical ownership")
            values[key] = values_for_field.pop()
            evidence[key] = dict(kind="source_proven_carton_lot_partition", paths=field["paths"])
            continue
        if (
            field["paths"]
            and all(
                p.startswith("documentPatch.forwardingAndExportReferences[") for p in field["paths"]
            )
            and re.fullmatch(r"[A-Za-z0-9/.-]+", field["source"])
            and re.search(r"\d", field["source"])
            and "hostAssembly" not in field
            and not any(c.get("fixedLiteralTokens") for c in field["constraints"])
            and any(
                b.logical_key in identifier_dependencies
                and set(b.target_paths).intersection(field["paths"])
                for b in source.template.bindings
            )
        ):
            values[key] = generate_from_surface_pattern(
                pattern=surface_pattern(field["source"]),
                stream=DeterministicStream(seed, "dependent-reference-v1", sample_id).derive(key),
                additional_excluded=field["source"],
            )
            evidence[key] = dict(kind="source_shaped_dependent_reference", source=field["source"])
            continue
        phone_owners = [
            match[1]
            for path in field["paths"]
            if (
                match := re.fullmatch(
                    r"(documentPatch\.parties\.[^.]+)\.contactDetails\.phoneNumbers\[\d+\]", path
                )
            )
        ]
        if projection is not None and phone_owners:
            if len(phone_owners) != len(field["paths"]):
                raise ValueError("phone field shares ownership with a non-phone scalar")
            countries = {projection.party_geography[p].country_code for p in phone_owners}
            if len(countries) != 1:
                raise ValueError("shared phone field has inconsistent sampled countries")
            country = countries.pop()
            value = contact_values.phone(
                DeterministicStream(seed, "structured-party-phone-v1", sample_id).derive(key),
                country_code=country,
            )
            values[key] = value
            evidence[key] = dict(kind="country_valid_phone", country=country)
            continue
        if (
            field["paths"]
            and all(_MARK_PATH.fullmatch(path) for path in field["paths"])
            and field["source"].strip().casefold() == "n/m"
            and "hostAssembly" not in field
        ):
            # The printed mark explicitly declares no marks. It is independent
            # of the sampled product, so there is nothing linguistic to invent.
            values[key] = field["source"]
            evidence[key] = dict(kind="source_declared_no_marks", paths=field["paths"])
            continue
        if scenario is None or not (definitions := _definitions(field, scenario)):
            continue
        reference = _reference(sample_id, key, seed)
        if "hostAssembly" in field:
            raise ValueError("fixed commodity literal requires a goods-specific lexical contract")
        if "cargoFragment" in field:
            if contract is None or key not in contract.fragments:
                raise ValueError("split cargo description lacks a reviewed lexical recipe: " + key)
            recipe = contract.fragments[key]
            if recipe.source != field["source"]:
                raise ValueError("cargo fragment recipe source changed")
            _validate_literal_fragment(field, recipe)
            labelled_reference = _validate_labelled_reference_fragment(field, recipe)
            brand = _validate_brand_fragment(field, recipe)
            if recipe.role == "literal":
                value = recipe.source
            elif brand is not None:
                value = brand["caption"] + _synthetic_brand(sample_id=sample_id, key=key, seed=seed)
            elif labelled_reference is not None:
                if not any(
                    re.search(
                        r"\b" + re.escape(recipe.identity_anchor or "") + r"\b",
                        definition,
                        re.I,
                    )
                    for definition in definitions
                ):
                    raise ValueError(
                        "labelled cargo reference incompatible with sampled goods identity"
                    )
                old_code = labelled_reference["code"]
                new_code = _labelled_reference_code(
                    old_code, sample_id=sample_id, key=key, seed=seed
                )
                value = labelled_reference["prefix"] + new_code
            elif recipe.role == "reference":
                value = reference
            elif recipe.role == "detail":
                value = "Commercial product reference " + reference
            else:
                if not recipe.identity_indices or any(
                    i < 0 or i >= len(definitions) for i in recipe.identity_indices
                ):
                    raise ValueError("cargo item recipe does not address sampled identities")
                value = "; ".join(definitions[i] for i in recipe.identity_indices)
                if recipe.role == "item":
                    value += "; product reference " + reference
        else:
            value = "; ".join(definitions) + "; product reference " + reference
            packages = list(
                dict.fromkeys(
                    phrase
                    for path in field["paths"]
                    if (phrase := _package_mentions(source, scenario.target, path))
                )
            )
            if packages:
                value += "; " + "; ".join(packages)
        minimum = max((c.get("minimumWords", 1) for c in field["constraints"]), default=1)
        if len(re.findall(r"[A-Za-z0-9]+", value)) < minimum:
            raise ValueError("controlled cargo wording cannot fill the certified text segments")
        if boundary := field.get("requiredBoundaryPunctuation"):
            value = boundary["prefix"] + value + boundary["suffix"]
        values[key] = value
        evidence[key] = (
            dict(kind="source_shaped_labelled_cargo_reference", source=recipe.source)
            if "cargoFragment" in field and labelled_reference is not None
            else dict(kind="registry_owned_cargo", definitions=definitions)
        )
    return HostLexicalPlan(values, evidence)
