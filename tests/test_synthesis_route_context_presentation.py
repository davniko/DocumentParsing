"""Source-only route context is omitted within the certified render contract."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.raw_text_template import (
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)
from document_ocr.synthesis.template_compiler.customs_presentation import CustomsPresentation
from document_ocr.synthesis.template_compiler.route_context_presentation import (
    RouteContextCache,
    compile_route_context,
)
from document_ocr.synthesis.template_compiler.source_only_route_assertions import validate

SOURCE = (
    "Port of discharge ALEXANDRIA\n"
    "NAME AND FULL ADDRESS OF SHIPPING AGENT IN EGYPT\nALEXANDRIA\nAGENT CO\n"
    "Receiver's account. Freight Tax is for Receiver's account as per Egyptian Laws. "
    "All expenses separate. ALL LANDING, DISCHARGING AD RELOADING OPERATIONS TO BE "
    "EFFECTED BY THE ALEXANDRIA PORT CONTAINERS TERMINAL AT RISK AND EXPENSE OF THE "
    "MERCHANTS (INCLUDING WEEKENDS AND PUBLIC HOLIDAYS AND ALL OVERTIME). "
    "Other terms apply."
)
SOURCE_TARGET = {
    "documentPatch": {"route": {"portOfDischarge": {"name": "ALEXANDRIA", "country": "EGYPT"}}}
}


def fixture() -> tuple[bytes, object, dict[str, str]]:
    source = SOURCE.encode()
    starts = []
    cursor = 0
    while True:
        start = SOURCE.find("ALEXANDRIA", cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + len("ALEXANDRIA")
    assert len(starts) == 3
    slots = []
    # The agent's printed location is source-only, not a route slot.
    for slot_id, start in (("slot_0001", starts[0]), ("slot_0002", starts[2])):
        byte_start = len(SOURCE[:start].encode())
        slots.append(
            build_template_slot(
                slot_id=slot_id,
                byte_start=byte_start,
                byte_end=byte_start + len("ALEXANDRIA"),
                source_text="ALEXANDRIA",
                target_paths=["documentPatch.route.portOfDischarge.name"],
                semantic_role="route:portOfDischarge",
                evidence_origin="accepted_label_evidence",
                render_policy="natural_text",
            )
        )
    template = compile_raw_text_template(document_id="fixture", source=source, slots=slots)
    return source, template, {"slot_0001": "PORT PIRIE", "slot_0002": "PORT PIRIE"}


def test_foreign_route_omits_only_source_proven_clauses_and_preserves_render_proof() -> None:
    source, template, bindings = fixture()
    target = deepcopy(SOURCE_TARGET)
    target["documentPatch"]["route"]["portOfDischarge"] = {"name": "PORT PIRIE"}
    presentation = compile_route_context(
        source=source,
        template=template,
        source_target=SOURCE_TARGET,
        target=target,
        sampled_discharge_country_code="AU",
    )
    assert presentation is not None
    assert len(presentation.evidence) == 3
    assert presentation.retired_slot_ids == {"slot_0002"}
    rendered, proof = presentation.render(source, bindings)
    text = rendered.decode()
    assert "Port of discharge PORT PIRIE" in text
    assert "Freight Tax" not in text
    assert "PORT CONTAINERS TERMINAL" not in text
    assert "SHIPPING AGENT IN EGYPT" not in text
    assert "NAME AND FULL ADDRESS OF SHIPPING AGENT\nALEXANDRIA" in text
    assert proof.output_sha256 == sha256_bytes(rendered)
    assert proof.exact_literal_regions and proof.line_endings_preserved
    validate(
        source=source,
        rendered=rendered,
        source_target=SOURCE_TARGET,
        target=target,
        template=SimpleNamespace(bindings=()),
    )
    with pytest.raises(ValueError, match="optional literal"):
        bad = presentation.binding_values(bindings)
        bad[next(iter(presentation.replacements))] = "NEW LEGAL CLAIM"
        render_compiled_template(source=source, template=presentation.template, bindings=bad)


def test_unknown_discharge_country_fails_before_rendering_egyptian_law() -> None:
    source, template, _ = fixture()
    target = deepcopy(SOURCE_TARGET)
    target["documentPatch"]["route"]["portOfDischarge"] = {"name": "PORT PIRIE"}
    with pytest.raises(ValueError, match="no sampled discharge-country owner"):
        compile_route_context(
            source=source,
            template=template,
            source_target=SOURCE_TARGET,
            target=target,
        )
    with pytest.raises(ValueError, match="contradicts the target country"):
        egypt_label = deepcopy(target)
        egypt_label["documentPatch"]["route"]["portOfDischarge"]["country"] = "EGYPT"
        compile_route_context(
            source=source,
            template=template,
            source_target=SOURCE_TARGET,
            target=egypt_label,
            sampled_discharge_country_code="AU",
        )


def test_omission_cannot_remove_only_printed_discharge_owner() -> None:
    source, template, _ = fixture()
    only_facility_slot = compile_raw_text_template(
        document_id="fixture",
        source=source,
        slots=[template.slots[1]],
    )
    target = deepcopy(SOURCE_TARGET)
    target["documentPatch"]["route"]["portOfDischarge"] = {"name": "PORT PIRIE"}
    with pytest.raises(ValueError, match="only printed discharge-port owner"):
        compile_route_context(
            source=source,
            template=only_facility_slot,
            source_target=SOURCE_TARGET,
            target=target,
            sampled_discharge_country_code="AU",
        )


def test_complete_pipeline_preparation_freezes_route_presentation_receipt() -> None:
    from document_ocr.synthesis.template_compiler.complete_pipeline import _prepared

    source, template, _ = fixture()
    source_target = deepcopy(SOURCE_TARGET)
    source_target["schemaVersion"] = "5.0.0"
    source_target["documentPatch"]["parties"] = {"carrier": {"name": "Carrier Ltd."}}
    target = deepcopy(source_target)
    target["documentPatch"]["route"]["portOfDischarge"] = {"name": "PORT PIRIE"}
    original = SimpleNamespace(
        document_id="fixture",
        source=source,
        source_target=source_target,
        target=source_target,
        template=SimpleNamespace(
            byte_template=template,
            source_sha256=sha256_bytes(source),
            carrier=SimpleNamespace(canonical_name="Carrier Ltd."),
        ),
    )
    prepared = _prepared(
        original,
        target,
        {},
        sample_id="sample-1",
        seed=7,
        numeric_values={},
        sampled_discharge_country_code="AU",
    )
    assert prepared.route_context_presentation is not None
    assert (
        prepared.target_receipt.route_context_presentation_sha256
        == prepared.route_context_presentation.sha256
    )


def test_route_context_composes_with_existing_source_presentation() -> None:
    source, template, bindings = fixture()
    start = SOURCE.index("Other terms")
    customs_slot = build_template_slot(
        slot_id="slot_0003",
        byte_start=len(SOURCE[:start].encode()),
        byte_end=len(SOURCE[:start].encode()) + len("Other terms"),
        source_text="Other terms",
        target_paths=(),
        semantic_role="customs_caption:test",
        evidence_origin="audited_source_auxiliary",
        render_policy="natural_text",
    )
    expanded = compile_raw_text_template(
        document_id="fixture", source=source, slots=[*template.slots, customs_slot]
    )
    customs = CustomsPresentation(
        template=expanded,
        replacements={"slot_0003": "General terms"},
        evidence=({"source": "Other terms", "replacement": "General terms"},),
        retired_static_slots={},
    )
    target = deepcopy(SOURCE_TARGET)
    target["documentPatch"]["route"]["portOfDischarge"] = {"name": "PORT PIRIE"}
    presentation = compile_route_context(
        source=source,
        template=template,
        source_target=SOURCE_TARGET,
        target=target,
        customs=customs,
        sampled_discharge_country_code="AU",
    )
    assert presentation is not None
    rendered, proof = presentation.render(source, bindings)
    assert b"General terms apply." in rendered
    assert b"Freight Tax" not in rendered
    assert proof.output_sha256 == sha256_bytes(rendered)


def test_cache_reuses_only_equivalent_source_owned_route_branches() -> None:
    source, template, bindings = fixture()
    cache = RouteContextCache(max_entries=2)
    target = deepcopy(SOURCE_TARGET)
    target["documentPatch"]["route"]["portOfDischarge"] = {"name": "PORT PIRIE"}
    first = compile_route_context(
        source=source,
        template=template,
        source_target=SOURCE_TARGET,
        target=target,
        sampled_discharge_country_code="AU",
        cache=cache,
    )
    target["documentPatch"]["route"]["portOfDischarge"] = {"name": "GENOA"}
    second = compile_route_context(
        source=source,
        template=template,
        source_target=SOURCE_TARGET,
        target=target,
        sampled_discharge_country_code="IT",
        cache=cache,
    )
    assert first is second
    assert first is not None
    uncached = compile_route_context(
        source=source,
        template=template,
        source_target=SOURCE_TARGET,
        target=target,
        sampled_discharge_country_code="IT",
    )
    assert uncached is not None
    assert second.template == uncached.template
    assert second.replacements == uncached.replacements
    assert second.evidence == uncached.evidence
    rendered, proof = second.render(source, {slot_id: "GENOA" for slot_id in bindings})
    assert (rendered, proof) == uncached.render(source, {slot_id: "GENOA" for slot_id in bindings})
    assert b"Port of discharge GENOA" in rendered
    assert proof.output_sha256 == sha256_bytes(rendered)
    target["documentPatch"]["route"]["portOfDischarge"] = {
        "name": "PORT PIRIE",
        "country": "EGYPT",
    }
    with pytest.raises(ValueError, match="contradicts the target country"):
        compile_route_context(
            source=source,
            template=template,
            source_target=SOURCE_TARGET,
            target=target,
            sampled_discharge_country_code="AU",
            cache=cache,
        )
    egypt = compile_route_context(
        source=source,
        template=template,
        source_target=SOURCE_TARGET,
        target=target,
        sampled_discharge_country_code="EG",
        cache=cache,
    )
    assert egypt is not None and egypt is not first
    assert any("Egyptian Laws" in row["source"] for row in first.evidence)
    assert not any("Egyptian Laws" in row["source"] for row in egypt.evidence)
    reloaded = compile_raw_text_template(document_id="fixture", source=source, slots=template.slots)
    assert reloaded is not template
    new_object = compile_route_context(
        source=source,
        template=reloaded,
        source_target=SOURCE_TARGET,
        target=target,
        sampled_discharge_country_code="EG",
        cache=cache,
    )
    assert new_object is not egypt
    assert new_object is not None and new_object.template == egypt.template


def test_cache_no_marker_fast_path_is_source_bound_and_bounded() -> None:
    cache = RouteContextCache(max_entries=1)
    source = b"Port of discharge ALEXANDRIA\nOrdinary freight conditions."
    template = compile_raw_text_template(document_id="simple", source=source, slots=[])
    assert (
        compile_route_context(
            source=source,
            template=template,
            source_target=SOURCE_TARGET,
            target=SOURCE_TARGET,
            cache=cache,
        )
        is None
    )
    assert cache.has_no_markers(source)
    assert (
        compile_route_context(
            source=source,
            template=template,
            source_target=SOURCE_TARGET,
            target=SOURCE_TARGET,
            cache=cache,
        )
        is None
    )
    second = b"Port of discharge ALEXANDRIA\nAnother ordinary freight condition."
    second_template = compile_raw_text_template(document_id="simple-2", source=second, slots=[])
    assert (
        compile_route_context(
            source=second,
            template=second_template,
            source_target=SOURCE_TARGET,
            target=SOURCE_TARGET,
            cache=cache,
        )
        is None
    )
    assert not cache.has_no_markers(source)
    assert cache.has_no_markers(second)
