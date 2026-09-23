"""Byte-exact compiled templates for synthetic OCR rendering.

The compiler owns *where* changes may occur.  Renderers supply only replacement
values for those pre-approved slots; they never search or edit the document.
Every byte outside a slot is copied directly from the pinned source payload.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.generators import surface_pattern

NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SlotId = Annotated[str, StringConstraints(pattern=r"^slot_[0-9]{4}$")]
_SLOT_ID = re.compile(r"slot_[0-9]{4}\Z")
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_NEWLINE = re.compile(r"\r\n|\r|\n")
_PAGE_MARKER = re.compile(rb"(?m)^--- PAGE [1-9][0-9]* ---[ \t]*(?:\r?\n|$)")
_COLLECTION_INDEX = re.compile(r"\[[0-9]+\]")
_NON_PRINTED_SCHEMA_FIELDS = frozenset(
    {
        "schemaVersion",
        "groupId",
        "packageId",
        "packageIds",
        "coverage",
        "sizeCategory",
        "typeCategory",
        "typeDescription",
    }
)


class SlotFormatEnvelope(BaseModel):
    """Observable source formatting that a replacement must preserve.

    ``surface_pattern`` is required only for opaque identifiers where character
    class and punctuation are semantic formatting.  Natural-language slots use
    the less restrictive line/edge/case contract.
    """

    model_config = _STRICT

    newline_sequence: tuple[Literal["\n", "\r", "\r\n"], ...]
    leading_whitespace_by_line: tuple[str, ...]
    trailing_whitespace_by_line: tuple[str, ...]
    case_profile: Literal["upper", "lower", "mixed", "uncased"]
    exact_surface_pattern: NonEmptyText | None

    @model_validator(mode="after")
    def one_edge_pair_per_line(self) -> SlotFormatEnvelope:
        expected_lines = len(self.newline_sequence) + 1
        if len(self.leading_whitespace_by_line) != expected_lines:
            raise ValueError("format envelope has the wrong leading-whitespace arity")
        if len(self.trailing_whitespace_by_line) != expected_lines:
            raise ValueError("format envelope has the wrong trailing-whitespace arity")
        if any(set(value) - {" ", "\t"} for value in self.leading_whitespace_by_line):
            raise ValueError("leading whitespace may contain only spaces and tabs")
        if any(set(value) - {" ", "\t"} for value in self.trailing_whitespace_by_line):
            raise ValueError("trailing whitespace may contain only spaces and tabs")
        return self


class TemplateSlot(BaseModel):
    """One immutable, audited UTF-8 source span."""

    model_config = _STRICT

    slot_id: SlotId
    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(gt=0)]
    source_text: NonEmptyText
    source_text_sha256: Sha256
    target_paths: tuple[NonEmptyText, ...]
    semantic_role: NonEmptyText
    evidence_origin: Literal[
        "accepted_label_evidence",
        "audited_source_auxiliary",
        "derived_operational_fact",
        "host_verified_agent_proposal",
    ]
    render_policy: Literal[
        "natural_text",
        "opaque_identifier",
        "date_surface",
        "numeric_surface",
        "categorical_surface",
        "derived_surface",
    ]
    format_envelope: SlotFormatEnvelope

    @model_validator(mode="after")
    def span_and_identity_are_valid(self) -> TemplateSlot:
        # model_copy deliberately bypasses field validation. Nested model
        # instances still run this invariant at the compiler boundary.
        if not isinstance(self.slot_id, str) or _SLOT_ID.fullmatch(self.slot_id) is None:
            raise ValueError("template slot ID must use the four-digit slot namespace")
        if self.byte_end <= self.byte_start:
            raise ValueError("template slot end must be after its start")
        encoded = self.source_text.encode("utf-8")
        if self.byte_end - self.byte_start != len(encoded):
            raise ValueError("template slot byte span differs from encoded source text")
        if sha256_bytes(encoded) != self.source_text_sha256:
            raise ValueError("template slot source-text hash differs")
        if len(self.target_paths) != len(set(self.target_paths)):
            raise ValueError("template slot target paths must be unique")
        if self.render_policy == "opaque_identifier":
            if self.format_envelope.exact_surface_pattern is None:
                raise ValueError("opaque identifiers require an exact surface pattern")
        elif self.format_envelope.exact_surface_pattern is not None:
            raise ValueError("exact surface patterns are restricted to opaque identifiers")
        return self


class CompiledRawTextTemplate(BaseModel):
    """A source-bound sequence of disjoint mutable spans."""

    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    source_sha256: Sha256
    source_size_bytes: Annotated[int, Field(gt=0)]
    encoding: Literal["utf-8"]
    slots: tuple[TemplateSlot, ...]
    compiler: Literal["audited_byte_spans_v1"]

    @model_validator(mode="after")
    def slots_are_sorted_and_disjoint(self) -> CompiledRawTextTemplate:
        if tuple(sorted(self.slots, key=lambda item: item.byte_start)) != self.slots:
            raise ValueError("template slots must be sorted by byte offset")
        if len({slot.slot_id for slot in self.slots}) != len(self.slots):
            raise ValueError("template slot IDs must be unique")
        for slot in self.slots:
            if slot.byte_end > self.source_size_bytes:
                raise ValueError("template slot extends beyond the source payload")
        for left, right in pairwise(self.slots):
            if left.byte_end > right.byte_start:
                raise ValueError("template slots overlap")
        return self


class TemplateRenderProof(BaseModel):
    """Machine-checkable evidence for one byte-level render."""

    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    source_sha256: Sha256
    output_sha256: Sha256
    source_round_trip: bool
    every_slot_bound_once: Literal[True]
    exact_literal_regions: Literal[True]
    page_markers_unchanged: Literal[True]
    line_endings_preserved: Literal[True]
    format_envelopes_valid: Literal[True]
    slot_count: Annotated[int, Field(ge=0)]
    mutable_source_bytes: Annotated[int, Field(ge=0)]
    immutable_source_bytes: Annotated[int, Field(ge=0)]


class PrintedTopologyMismatch(BaseModel):
    """One source/target cardinality or optional-value shape difference.

    Model-facing relation identifiers and semantic category fields are excluded:
    they do not independently require a printed OCR slot.  Everything retained
    here changes the number or kind of values that the raw-text renderer must
    realize.
    """

    model_config = _STRICT

    path: NonEmptyText
    value_kind: NonEmptyText
    source_count: Annotated[int, Field(ge=0)]
    target_count: Annotated[int, Field(ge=0)]


def _printed_field_inventory(
    value: object, path: str = ""
) -> Counter[tuple[str, str]]:
    inventory: Counter[tuple[str, str]] = Counter()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _NON_PRINTED_SCHEMA_FIELDS:
                continue
            child_path = f"{path}.{key}" if path else str(key)
            inventory.update(_printed_field_inventory(child, child_path))
    elif isinstance(value, list):
        normalized = _COLLECTION_INDEX.sub("[]", path)
        inventory[(normalized + "#items", str(len(value)))] += 1
        for index, child in enumerate(value):
            inventory.update(_printed_field_inventory(child, f"{path}[{index}]"))
    else:
        normalized = _COLLECTION_INDEX.sub("[]", path)
        kind = "null" if value is None else type(value).__name__
        inventory[(normalized, kind)] += 1
    return inventory


def printed_topology_mismatches(
    source_target: Mapping[str, object], target: Mapping[str, object]
) -> tuple[PrintedTopologyMismatch, ...]:
    """Return every raw-text-bearing topology difference, sorted by path.

    A mismatch is a pairing error, not something the byte renderer may guess
    around.  Callers must re-pair/regenerate the target or invoke an explicitly
    audited structural-template compiler before rendering.
    """

    source = _printed_field_inventory(source_target)
    generated = _printed_field_inventory(target)
    keys = sorted(source.keys() | generated.keys())
    return tuple(
        PrintedTopologyMismatch(
            path=path,
            value_kind=value_kind,
            source_count=source[(path, value_kind)],
            target_count=generated[(path, value_kind)],
        )
        for path, value_kind in keys
        if source[(path, value_kind)] != generated[(path, value_kind)]
    )


def _case_profile(value: str) -> Literal["upper", "lower", "mixed", "uncased"]:
    letters = [character for character in value if character.isalpha()]
    if not letters:
        return "uncased"
    if all(character.isupper() for character in letters):
        return "upper"
    if all(character.islower() for character in letters):
        return "lower"
    return "mixed"


def _edge_whitespace(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lines = _NEWLINE.split(value)
    leading: list[str] = []
    trailing: list[str] = []
    for line in lines:
        leading_match = re.match(r"^[ \t]*", line)
        trailing_match = re.search(r"[ \t]*$", line)
        assert leading_match is not None and trailing_match is not None
        leading.append(leading_match.group(0))
        trailing.append(trailing_match.group(0))
    return tuple(leading), tuple(trailing)


def format_envelope(
    source_text: str, *, render_policy: str
) -> SlotFormatEnvelope:
    """Capture format metadata without retaining any additional source value."""

    if not source_text:
        raise ValueError("format envelope source must not be empty")
    leading, trailing = _edge_whitespace(source_text)
    pattern = surface_pattern(source_text) if render_policy == "opaque_identifier" else None
    return SlotFormatEnvelope.model_validate(
        {
            "newline_sequence": tuple(match.group(0) for match in _NEWLINE.finditer(source_text)),
            "leading_whitespace_by_line": leading,
            "trailing_whitespace_by_line": trailing,
            "case_profile": _case_profile(source_text),
            "exact_surface_pattern": pattern,
        },
        strict=True,
    )


def build_template_slot(
    *,
    slot_id: str,
    byte_start: int,
    byte_end: int,
    source_text: str,
    target_paths: Sequence[str],
    semantic_role: str,
    evidence_origin: str,
    render_policy: str,
) -> TemplateSlot:
    """Construct a slot while binding all derived metadata to its exact source bytes."""

    return TemplateSlot.model_validate(
        {
            "slot_id": slot_id,
            "byte_start": byte_start,
            "byte_end": byte_end,
            "source_text": source_text,
            "source_text_sha256": sha256_bytes(source_text.encode("utf-8")),
            "target_paths": tuple(target_paths),
            "semantic_role": semantic_role,
            "evidence_origin": evidence_origin,
            "render_policy": render_policy,
            "format_envelope": format_envelope(source_text, render_policy=render_policy),
        },
        strict=True,
    )


def compile_raw_text_template(
    *, document_id: str, source: bytes, slots: Sequence[TemplateSlot]
) -> CompiledRawTextTemplate:
    """Bind disjoint audited slots to a pinned UTF-8 source payload."""

    try:
        source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("raw OCR template must be valid UTF-8") from error
    ordered = tuple(sorted(slots, key=lambda item: item.byte_start))
    for slot in ordered:
        exact = slot.source_text.encode("utf-8")
        if source[slot.byte_start : slot.byte_end] != exact:
            raise ValueError(f"template slot {slot.slot_id} does not match the pinned source")
        try:
            source[: slot.byte_start].decode("utf-8", errors="strict")
            source[: slot.byte_end].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(f"template slot {slot.slot_id} splits a UTF-8 code point") from error
    template = CompiledRawTextTemplate.model_validate(
        {
            "schema_version": 1,
            "document_id": document_id,
            "source_sha256": sha256_bytes(source),
            "source_size_bytes": len(source),
            "encoding": "utf-8",
            "slots": ordered,
            "compiler": "audited_byte_spans_v1",
        },
        strict=True,
    )
    # Prove identity before the template can leave the compiler boundary.
    rendered, proof = render_compiled_template(
        source=source,
        template=template,
        bindings={slot.slot_id: slot.source_text for slot in ordered},
    )
    if rendered != source or not proof.source_round_trip:
        raise RuntimeError("compiled raw-text template failed its source round trip")
    return template


def _validate_format(slot: TemplateSlot, replacement: str) -> None:
    envelope = format_envelope(replacement, render_policy=slot.render_policy)
    expected = slot.format_envelope
    if envelope.newline_sequence != expected.newline_sequence:
        raise ValueError(f"slot {slot.slot_id} replacement changes line endings or line count")
    if envelope.leading_whitespace_by_line != expected.leading_whitespace_by_line:
        raise ValueError(f"slot {slot.slot_id} replacement changes leading whitespace")
    if envelope.trailing_whitespace_by_line != expected.trailing_whitespace_by_line:
        raise ValueError(f"slot {slot.slot_id} replacement changes trailing whitespace")
    if (
        expected.case_profile in {"upper", "lower"}
        and envelope.case_profile not in {expected.case_profile, "uncased"}
    ):
        raise ValueError(f"slot {slot.slot_id} replacement changes the source case profile")
    if (
        expected.exact_surface_pattern is not None
        and envelope.exact_surface_pattern != expected.exact_surface_pattern
    ):
        raise ValueError(f"slot {slot.slot_id} replacement changes identifier shape")


def validate_slot_replacements(
    *,
    template: CompiledRawTextTemplate,
    replacements: Mapping[str, str],
) -> None:
    """Validate a partial set of replacements against their compiled slot envelopes.

    Planning validates one semantic binding at a time.  Rendering the complete document for
    every binding repeats identical source, literal-region, and unchanged-slot work.  This
    focused contract rejects unknown slots and applies the exact same format validation only to
    the replacements under consideration; the final render still proves the complete template.
    """

    slots = {slot.slot_id: slot for slot in template.slots}
    unknown = sorted(set(replacements) - set(slots))
    if unknown:
        raise ValueError(f"replacement references unknown template slots: {unknown}")
    for slot_id, replacement in replacements.items():
        _validate_format(slots[slot_id], replacement)


def render_compiled_template(
    *,
    source: bytes,
    template: CompiledRawTextTemplate,
    bindings: Mapping[str, str],
    validate_format: bool = True,
) -> tuple[bytes, TemplateRenderProof]:
    """Render every slot exactly once and attest all untouched source bytes."""

    if len(source) != template.source_size_bytes or sha256_bytes(source) != template.source_sha256:
        raise ValueError("template source payload differs from the pinned source")
    expected_ids = {slot.slot_id for slot in template.slots}
    if set(bindings) != expected_ids:
        missing = sorted(expected_ids - set(bindings))
        extra = sorted(set(bindings) - expected_ids)
        raise ValueError(f"template bindings differ; missing={missing}, extra={extra}")

    output_parts: list[bytes] = []
    literal_hashes_before: list[str] = []
    literal_hashes_after: list[str] = []
    cursor = 0
    for slot in template.slots:
        literal = source[cursor : slot.byte_start]
        output_parts.append(literal)
        literal_hash = sha256_bytes(literal)
        literal_hashes_before.append(literal_hash)
        literal_hashes_after.append(literal_hash)
        replacement = bindings[slot.slot_id]
        if validate_format:
            _validate_format(slot, replacement)
        output_parts.append(replacement.encode("utf-8"))
        cursor = slot.byte_end
    final_literal = source[cursor:]
    output_parts.append(final_literal)
    final_hash = sha256_bytes(final_literal)
    literal_hashes_before.append(final_hash)
    literal_hashes_after.append(final_hash)
    output = b"".join(output_parts)

    source_newlines = tuple(match.group(0) for match in re.finditer(rb"\r\n|\r|\n", source))
    output_newlines = tuple(match.group(0) for match in re.finditer(rb"\r\n|\r|\n", output))
    source_markers = tuple(match.group(0) for match in _PAGE_MARKER.finditer(source))
    output_markers = tuple(match.group(0) for match in _PAGE_MARKER.finditer(output))
    mutable = sum(slot.byte_end - slot.byte_start for slot in template.slots)
    proof = TemplateRenderProof.model_validate(
        {
            "schema_version": 1,
            "document_id": template.document_id,
            "source_sha256": template.source_sha256,
            "output_sha256": sha256_bytes(output),
            "source_round_trip": output == source,
            "every_slot_bound_once": True,
            "exact_literal_regions": literal_hashes_before == literal_hashes_after,
            "page_markers_unchanged": source_markers == output_markers,
            "line_endings_preserved": source_newlines == output_newlines,
            "format_envelopes_valid": True,
            "slot_count": len(template.slots),
            "mutable_source_bytes": mutable,
            "immutable_source_bytes": len(source) - mutable,
        },
        strict=True,
    )
    return output, proof


def sentinel_bindings(template: CompiledRawTextTemplate) -> dict[str, str]:
    """Create deterministic non-secret probes that preserve whitespace and punctuation.

    Sentinel renders are for isolation tests, so callers disable semantic format validation.
    They prove that mutating every slot cannot alter any literal source region.
    """

    output: dict[str, str] = {}
    for slot_index, slot in enumerate(template.slots, start=1):
        digit = str(slot_index % 10)
        chars: list[str] = []
        for character in slot.source_text:
            if character.isupper():
                chars.append("X")
            elif character.islower():
                chars.append("x")
            elif character.isdigit():
                chars.append(digit)
            else:
                chars.append(character)
        candidate = "".join(chars)
        if candidate == slot.source_text:
            # A punctuation-only evidence span is still a valid mutable boundary.  Add a
            # non-newline sentinel inside that boundary for the isolation test; production
            # rendering continues to enforce the slot's normal format contract.
            insertion = len(candidate)
            while insertion > 0 and candidate[insertion - 1] in {"\r", "\n"}:
                insertion -= 1
            candidate = candidate[:insertion] + "X" + candidate[insertion:]
        output[slot.slot_id] = candidate
    return output
