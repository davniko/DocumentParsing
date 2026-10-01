"""Source-bound optional route-context clauses with exact render proofs.

The clause decision uses the sampled physical discharge country and printed
source ownership. It augments the byte template before rendering; it never
rewrites a completed OCR string or changes a training target.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.raw_text_template import (
    CompiledRawTextTemplate,
    TemplateRenderProof,
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)

from .customs_presentation import CustomsPresentation

_WORD = re.compile(r"[a-z0-9]+")
_LAW = re.compile(r" FREIGHT TAX IS FOR RECEIVER['\u2019]S ACCOUNT AS PER EGYPTIAN LAWS\.", re.I)
_ALEXANDRIA_TERMINAL = re.compile(
    r" Any transportation expss of containers at Alexandria terminal "
    r"to be collected fm receiver['\u2019]s prior releasing the cargo\.",
    re.I,
)
_FACILITY = re.compile(
    r" ALL LANDING, DISCHARGING AD RELOADING OPERATIONS TO BE EFFECTED BY THE "
    r"(?P<place>[^\n]+?) PORT CONTAINERS TERMINAL AT RISK AND EXPENSE OF THE MERCHANTS "
    r"\(INCLUDING WEEKENDS AND PUBLIC HOLIDAYS AND ALL OVERTIME\)\.",
    re.I,
)
_AGENT_HEADING = re.compile(r" IN EGYPT", re.I)
_AGENT_LINE = re.compile(r"(?im)^NAME AND FULL ADDRESS OF SHIPPING AGENT IN EGYPT\r?$")
_EGYPT = frozenset({"eg", "egypt", "egy", "arab republic of egypt"})


def _normal(value: str) -> str:
    return " ".join(_WORD.findall(value.casefold()))


def _same_port(left: str, right: str) -> bool:
    lhs, rhs = _normal(left), _normal(right)
    return bool(lhs and rhs and (f" {lhs} " in f" {rhs} " or f" {rhs} " in f" {lhs} "))


def _discharge(target: Mapping[str, Any]) -> tuple[str | None, str | None]:
    value = target.get("documentPatch", {}).get("route", {}).get("portOfDischarge", {})
    name, country = value.get("name"), value.get("country")
    return (
        name if isinstance(name, str) and name.strip() else None,
        country if isinstance(country, str) and country.strip() else None,
    )


def _sampled_country(
    *, target_country: str | None, sampled_country_code: str | None
) -> bool | None:
    label_egypt = _normal(target_country) in _EGYPT if target_country else None
    scenario_egypt = _normal(sampled_country_code) in _EGYPT if sampled_country_code else None
    if label_egypt is not None and scenario_egypt is not None and label_egypt != scenario_egypt:
        raise ValueError("sampled discharge country contradicts the target country")
    return scenario_egypt if scenario_egypt is not None else label_egypt


@dataclass(frozen=True, slots=True)
class RouteContextPresentation:
    template: CompiledRawTextTemplate
    replacements: Mapping[str, str]
    retired_slot_ids: frozenset[str]
    customs: CustomsPresentation | None
    evidence: tuple[dict[str, object], ...]

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.evidence))

    def binding_values(self, bindings: Mapping[str, str]) -> dict[str, str]:
        values = self.customs.binding_values(bindings) if self.customs else dict(bindings)
        for slot_id in self.retired_slot_ids:
            if slot_id not in values:
                raise ValueError(f"retired route-context slot is missing: {slot_id}")
            del values[slot_id]
        if set(values) & set(self.replacements):
            raise ValueError("route-context and value slot IDs overlap")
        return {**values, **self.replacements}

    def render(
        self, source: bytes, bindings: Mapping[str, str]
    ) -> tuple[bytes, TemplateRenderProof]:
        return render_compiled_template(
            source=source,
            template=self.template,
            bindings=self.binding_values(bindings),
        )


class RouteContextCache:
    """Bound repeated presentations to one synthesis run and source-object lifetime.

    A route projection can choose thousands of port names, but the byte-template
    extension depends only on whether the printed port changed and whether the
    sampled discharge country is Egypt. The source objects are kept in each
    entry, so identity keys cannot alias a later, unrelated object.
    """

    def __init__(self, *, max_entries: int = 2048) -> None:
        if max_entries < 1:
            raise ValueError("route-context cache capacity must be positive")
        self.max_entries = max_entries
        self._no_marker_sources: OrderedDict[int, bytes] = OrderedDict()
        self._entries: OrderedDict[
            tuple[int, int, int, str | None, bool, bool | None],
            tuple[
                bytes,
                CompiledRawTextTemplate,
                CustomsPresentation | None,
                RouteContextPresentation | None,
            ],
        ] = OrderedDict()

    def has_no_markers(self, source: bytes) -> bool:
        cached = self._no_marker_sources.get(id(source))
        if cached is not source:
            return False
        self._no_marker_sources.move_to_end(id(source))
        return True

    def remember_no_markers(self, source: bytes) -> None:
        self._no_marker_sources[id(source)] = source
        self._no_marker_sources.move_to_end(id(source))
        if len(self._no_marker_sources) > self.max_entries:
            self._no_marker_sources.popitem(last=False)

    def get(
        self,
        *,
        source: bytes,
        template: CompiledRawTextTemplate,
        customs: CustomsPresentation | None,
        source_port: str | None,
        changed_port: bool,
        sampled_egypt: bool | None,
    ) -> tuple[bool, RouteContextPresentation | None]:
        key = (id(source), id(template), id(customs), source_port, changed_port, sampled_egypt)
        entry = self._entries.get(key)
        if entry is None:
            return False, None
        if entry[0] is not source or entry[1] is not template or entry[2] is not customs:
            raise AssertionError("route-context cache object identity changed")
        self._entries.move_to_end(key)
        return True, entry[3]

    def put(
        self,
        *,
        source: bytes,
        template: CompiledRawTextTemplate,
        customs: CustomsPresentation | None,
        source_port: str | None,
        changed_port: bool,
        sampled_egypt: bool | None,
        presentation: RouteContextPresentation | None,
    ) -> None:
        key = (id(source), id(template), id(customs), source_port, changed_port, sampled_egypt)
        self._entries[key] = (source, template, customs, presentation)
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


def compile_route_context(
    *,
    source: bytes,
    template: CompiledRawTextTemplate,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    customs: CustomsPresentation | None = None,
    sampled_discharge_country_code: str | None = None,
    cache: RouteContextCache | None = None,
) -> RouteContextPresentation | None:
    """Compile only individually owned optional spans, failing on ambiguous overlap."""
    if cache is not None and cache.has_no_markers(source):
        return None
    upper_source = source.upper()
    if not any(
        marker in upper_source
        for marker in (
            b"EGYPTIAN LAWS",
            b"ALEXANDRIA TERMINAL",
            b"PORT CONTAINERS TERMINAL",
            b"SHIPPING AGENT IN EGYPT",
        )
    ):
        if cache is not None:
            cache.remember_no_markers(source)
        return None
    if sha256_bytes(source) != template.source_sha256:
        raise ValueError("route-context source differs from certified byte template")
    base = customs.template if customs else template
    if base.source_sha256 != template.source_sha256:
        raise ValueError("customs and route presentations have different sources")
    text = source.decode("utf-8")
    source_port, _ = _discharge(source_target)
    target_port, target_country = _discharge(target)
    sampled_egypt = _sampled_country(
        target_country=target_country, sampled_country_code=sampled_discharge_country_code
    )
    changed_port = bool(source_port and target_port and not _same_port(source_port, target_port))
    if cache is not None:
        hit, cached = cache.get(
            source=source,
            template=template,
            customs=customs,
            source_port=source_port,
            changed_port=changed_port,
            sampled_egypt=sampled_egypt,
        )
        if hit:
            return cached
    proposed: list[tuple[re.Match[str], str, str, bool]] = []

    source_law = tuple(_LAW.finditer(text))
    if source_law and changed_port and sampled_egypt is None:
        raise ValueError("Egyptian-law clause has no sampled discharge-country owner")
    if source_law and sampled_egypt is False:
        for match in source_law:
            before = text[: match.start()]
            if before.endswith("."):
                replacement = ""
            elif before.upper().endswith(("ACCOUNT", "FREE OUT")):
                replacement = "."
            else:
                raise ValueError("Egyptian-law clause lacks an independent sentence boundary")
            proposed.append((match, replacement, "omit_foreign_route_egyptian_law", False))

    if changed_port and source_port and target_port:
        if _same_port("Alexandria", source_port):
            for match in _ALEXANDRIA_TERMINAL.finditer(text):
                if not text[: match.start()].endswith("."):
                    raise ValueError("Alexandria-terminal sentence has no independent boundary")
                proposed.append((match, "", "omit_changed_port_alexandria_terminal", False))
        for match in _FACILITY.finditer(text):
            if not _same_port(match["place"], source_port):
                continue
            if not text[: match.start()].endswith("."):
                raise ValueError("port-terminal facility sentence has no independent boundary")
            proposed.append((match, "", "omit_unproved_changed_port_facility", True))

    agent_lines = tuple(_AGENT_LINE.finditer(text))
    if agent_lines:
        for line in agent_lines:
            suffix_match = _AGENT_HEADING.search(text, line.start(), line.end())
            if suffix_match is None:
                raise ValueError("Egypt-specific agent heading span is ambiguous")
            proposed.append((suffix_match, "", "neutralize_egypt_shipping_agent_heading", False))

    if not proposed:
        if cache is not None:
            cache.put(
                source=source,
                template=template,
                customs=customs,
                source_port=source_port,
                changed_port=changed_port,
                sampled_egypt=sampled_egypt,
                presentation=None,
            )
        return None
    proposed.sort(key=lambda item: item[0].start())
    slots = list(base.slots)
    next_id = max((int(slot.slot_id.split("_")[1]) for slot in slots), default=0) + 1
    retired: set[str] = set()
    replacements: dict[str, str] = {}
    evidence: list[dict[str, object]] = []
    previous_end = -1
    for match, replacement, reason, can_retire in proposed:
        if match.start() < previous_end:
            raise ValueError("optional route-context claims overlap")
        previous_end = match.end()
        start = len(text[: match.start()].encode())
        end = len(text[: match.end()].encode())
        overlaps = [slot for slot in slots if slot.byte_start < end and start < slot.byte_end]
        if overlaps:
            if not can_retire or any(
                slot.byte_start < start or slot.byte_end > end for slot in overlaps
            ):
                raise ValueError(f"optional route context overlaps value ownership: {reason}")
            remaining_discharge_slots = [
                slot
                for slot in slots
                if slot not in overlaps
                and any(
                    path == "documentPatch.route.portOfDischarge.name" for path in slot.target_paths
                )
            ]
            if not remaining_discharge_slots:
                raise ValueError("cannot omit the only printed discharge-port owner")
            if any(
                any(
                    path != "documentPatch.route.portOfDischarge.name" for path in slot.target_paths
                )
                for slot in overlaps
            ):
                raise ValueError("port-terminal clause overlaps non-route target data")
            retired.update(slot.slot_id for slot in overlaps)
        slot = build_template_slot(
            slot_id=f"slot_{next_id:04d}",
            byte_start=start,
            byte_end=end,
            source_text=match.group(),
            target_paths=(),
            semantic_role="optional_route_context:" + reason,
            evidence_origin="audited_source_auxiliary",
            render_policy="optional_literal",
        )
        slots.append(slot)
        replacements[slot.slot_id] = replacement
        evidence.append(
            {
                "slotId": slot.slot_id,
                "sourceSha256": sha256_bytes(source),
                "byteStart": start,
                "byteEnd": end,
                "source": match.group(),
                "replacement": replacement,
                "reason": reason,
                "sourcePort": source_port,
                "changedPort": changed_port,
                "sampledDischargeIsEgypt": sampled_egypt,
                "retiredSlotIds": sorted(slot.slot_id for slot in overlaps),
            }
        )
        next_id += 1
    expanded = compile_raw_text_template(
        document_id=template.document_id,
        source=source,
        slots=[slot for slot in slots if slot.slot_id not in retired],
    )
    result = RouteContextPresentation(
        expanded, replacements, frozenset(retired), customs, tuple(evidence)
    )
    # Prove the expanded render contract before accepting the extension.
    source_values = {slot.slot_id: slot.source_text for slot in template.slots}
    rendered, proof = result.render(source, source_values)
    if proof.output_sha256 != sha256_bytes(rendered):
        raise ValueError("route-context presentation output proof differs")
    if cache is not None:
        cache.put(
            source=source,
            template=template,
            customs=customs,
            source_port=source_port,
            changed_port=changed_port,
            sampled_egypt=sampled_egypt,
            presentation=result,
        )
    return result
