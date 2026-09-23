import json

import pytest

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import CustomsProgramRegistry
from document_ocr.synthesis.raw_text_template import build_template_slot, compile_raw_text_template
from document_ocr.synthesis.template_compiler.customs_presentation import compile_neutral_customs


def registry():
    return CustomsProgramRegistry.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "program_id": "egypt_advance_cargo_information",
                        "official_name": "ACI",
                        "authority": "NAFEZA",
                        "official_source_url": "https://www.nafeza.gov.eg/",
                        "jurisdiction_country_code": "EG",
                        "trade_direction": "import",
                        "source_surfaces": ["ACID NUMBER", "ACID NO", "ACID:", "ACID"],
                        "generic_replacement_surface": "CUSTOMS REFERENCE",
                    }
                ],
            }
        )
    )


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_customs_extension_preserves_values_unicode_and_all_repeated_captions(ending):
    source = ending.join(
        [
            "--- PAGE 1 ---",
            "Société",
            "ACID: 1234567890123456789",
            "ACID NUMBER",
            "1234567890123456789",
            "FATTY ACID: STEARIC ACID",
            "",
        ]
    ).encode()
    value = "1234567890123456789"
    slots = []
    cursor = 0
    for index in range(2):
        start = source.index(value.encode(), cursor)
        slots.append(
            build_template_slot(
                slot_id=f"slot_{index:04d}",
                byte_start=start,
                byte_end=start + len(value),
                source_text=value,
                target_paths=(),
                semantic_role="customs_reference",
                evidence_origin="audited_source_auxiliary",
                render_policy="opaque_identifier",
            )
        )
        cursor = start + len(value)
    template = compile_raw_text_template(document_id="source", source=source, slots=slots)
    plan = compile_neutral_customs(source=source, template=template, registry=registry())
    rendered, proof = plan.render(source, {s.slot_id: "9876543210987654321" for s in slots})
    assert rendered.count(b"CUSTOMS REFERENCE") == 2
    assert rendered.count(b"9876543210987654321") == 2
    assert b"FATTY ACID: STEARIC ACID" in rendered
    assert "Société".encode() in rendered
    assert proof.exact_literal_regions and proof.page_markers_unchanged
    assert proof.line_endings_preserved and proof.format_envelopes_valid
    assert template.source_sha256 == sha256_bytes(source)
    assert len(template.slots) == 2  # Original compilation is untouched.
    assert len(plan.evidence) == 2


def test_multiline_caption_keeps_layout():
    source = b"ACID\r\nNUMBER\r\n1234567890123456789\r\n"
    template = compile_raw_text_template(document_id="s", source=source, slots=[])
    plan = compile_neutral_customs(source=source, template=template, registry=registry())
    text, proof = plan.render(source, {})
    assert text == b"CUSTOMS\r\nREFERENCE\r\n1234567890123456789\r\n"
    assert proof.line_endings_preserved


def test_conflicting_value_ownership_fails_before_rendering():
    source = b"ACID NUMBER: 1234567890123456789\n"
    slot = build_template_slot(
        slot_id="slot_0000",
        byte_start=0,
        byte_end=4,
        source_text="ACID",
        target_paths=(),
        semantic_role="misclassified_cargo",
        evidence_origin="audited_source_auxiliary",
        render_policy="natural_text",
    )
    template = compile_raw_text_template(document_id="s", source=source, slots=[slot])
    with pytest.raises(ValueError, match="overlaps existing value ownership"):
        compile_neutral_customs(source=source, template=template, registry=registry())


def test_source_drift_rejected():
    template = compile_raw_text_template(document_id="s", source=b"original", slots=[])
    with pytest.raises(ValueError, match="source differs"):
        compile_neutral_customs(source=b"changed", template=template, registry=registry())


def test_house_customs_heading_cannot_consume_a_warehouse_name_on_previous_line():
    source = b"CARGO IN TRANSIT TO KABARI EFS BONDED WAR HOUSE\nACID: 2002706562024040110\n"
    start = source.index(b"KABARI")
    end = source.index(b"\n")
    slot = build_template_slot(
        slot_id="slot_0000",
        byte_start=start,
        byte_end=end,
        source_text=source[start:end].decode(),
        target_paths=("documentPatch.route.finalDestination.name",),
        semantic_role="destination",
        evidence_origin="audited_source_auxiliary",
        render_policy="natural_text",
    )
    data = registry().model_dump(mode="json")
    data["entries"][0]["source_surfaces"].insert(0, "HOUSE ACID:")
    customs = CustomsProgramRegistry.model_validate_json(json.dumps(data))
    template = compile_raw_text_template(document_id="s", source=source, slots=[slot])
    plan = compile_neutral_customs(source=source, template=template, registry=customs)
    text, _ = plan.render(source, {"slot_0000": "NEW BONDED WAREHOUSE"})
    assert b"NEW BONDED WAREHOUSE\nCUSTOMS REFERENCE: 2002706562024040110" in text


@pytest.mark.parametrize("caption", ["NO.", "number-"])
def test_number_caption_does_not_become_random_identifier_characters(caption):
    from document_ocr.synthesis.template_compiler import host

    text = caption + "5939523832023020082"
    raw = "ACID " + text
    draft = host.SpanDraft(
        draft_id="id",
        logical_key="customs:reference",
        render_mode="deterministic_auxiliary",
        value_kind="identifier",
        group_kind="document",
        group_key="document:customs",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=5,
        char_end=len(raw),
        source_text=text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="opaque_identifier",
        rationale="Printed reference.",
    )
    result = host.normalize_labeled_auxiliary_identifier_boundaries(raw=raw, drafts=(draft,))
    assert result[0].source_text == "5939523832023020082"
    host.validate_draft_source_alignment(raw=raw, drafts=result)
    assert host.normalize_labeled_auxiliary_identifier_boundaries(raw=raw, drafts=result) == result


def test_registered_caption_repair_does_not_freeze_target_values_or_legal_assertions():
    from dataclasses import replace

    from document_ocr.synthesis.template_compiler import host
    from document_ocr.synthesis.template_compiler.customs_presentation import (
        normalize_registered_caption_ownership,
    )

    raw = "ACID REF EGYPT: 1234567890123456789\nShipment bound to Egypt"
    start = raw.index("EGYPT")
    draft = host.SpanDraft(
        draft_id="country",
        logical_key="customs_country",
        render_mode="deterministic_auxiliary",
        value_kind="location",
        group_kind="document",
        group_key="document:customs",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + 5,
        source_text="EGYPT",
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Source.",
    )
    data = registry().model_dump(mode="json")
    data["entries"][0]["source_surfaces"] += ["ACID REF EGYPT", "Shipment bound to Egypt"]
    data["entries"][0]["source_surfaces"].sort(key=len, reverse=True)
    registry_value = CustomsProgramRegistry.model_validate_json(json.dumps(data))
    repaired = normalize_registered_caption_ownership(
        raw=raw, drafts=(draft,), registry=registry_value
    )
    assert repaired[0].render_mode == "literal_static"
    owned = replace(draft, target_paths=("documentPatch.parties.consignee.country",))
    assert normalize_registered_caption_ownership(
        raw=raw, drafts=(owned,), registry=registry_value
    ) == (owned,)
    start = raw.rindex("Egypt")
    assertion = replace(draft, char_start=start, char_end=start + 5, source_text="Egypt")
    assert normalize_registered_caption_ownership(
        raw=raw, drafts=(assertion,), registry=registry_value
    ) == (assertion,)


def test_registered_legal_caption_retires_only_redundant_route_country_occurrence():
    from dataclasses import replace

    from document_ocr.synthesis.template_compiler import host
    from document_ocr.synthesis.template_compiler.customs_presentation import (
        normalize_registered_caption_ownership,
    )

    caption = "Preliminary Customs registration number for shipment bound to Egypt"
    raw = "Port of discharge: ALEXANDRIA, Egypt\n" + caption + "\n"
    data = registry().model_dump(mode="json")
    data["entries"][0]["source_surfaces"].append(caption)
    data["entries"][0]["source_surfaces"].sort(key=len, reverse=True)
    reg = CustomsProgramRegistry.model_validate_json(json.dumps(data))
    start = raw.index("Egypt")
    field = host.SpanDraft(
        draft_id="port",
        logical_key="country",
        render_mode="target_binding",
        value_kind="location",
        group_kind="route",
        group_key="route:portOfDischarge",
        target_paths=("documentPatch.route.portOfDischarge.country",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + 5,
        source_text="Egypt",
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Printed country",
    )
    redundant = replace(
        field, draft_id="legal", char_start=raw.rindex("Egypt"), char_end=raw.rindex("Egypt") + 5
    )
    repaired = normalize_registered_caption_ownership(
        raw=raw, drafts=(field, redundant), registry=reg
    )
    assert repaired[0] == field
    assert repaired[1].render_mode == "literal_static" and repaired[1].target_paths == ()
    assert normalize_registered_caption_ownership(raw=raw, drafts=(redundant,), registry=reg) == (
        redundant,
    )
    # An entire registered caption (including punctuation) has no sampled value.
    whole = replace(
        redundant,
        logical_key="caption",
        render_mode="deterministic_auxiliary",
        target_paths=(),
        char_start=raw.index(caption),
        char_end=len(raw) - 1,
        source_text=caption,
    )
    result = normalize_registered_caption_ownership(raw=raw, drafts=(whole,), registry=reg)
    assert result[0].render_mode == "literal_static"


def test_parenthetical_importer_caption_does_not_claim_an_adjacent_party_country():
    source = b"EGYPT IMPORTER TAX ID: 123\nEGYPT IMPORTER (ACME) VAT NUMBER: 456\n"
    slot = build_template_slot(
        slot_id="slot_0000",
        byte_start=0,
        byte_end=5,
        source_text="EGYPT",
        target_paths=("documentPatch.parties.consignee.country",),
        semantic_role="party:consignee:0",
        evidence_origin="audited_source_auxiliary",
        render_policy="natural_text",
    )
    data = registry().model_dump(mode="json")
    data["schema_version"] = 3
    data["entries"][0]["party_surface_rewrites"] = [
        {
            "source_surface": "EGYPT IMPORTER (",
            "generic_replacement_surface": "CARGO IMPORTER (",
            "alternative_generic_replacement_surfaces": [],
            "target_party_role": "consignee",
        }
    ]
    plan = compile_neutral_customs(
        source=source,
        template=compile_raw_text_template(document_id="s", source=source, slots=[slot]),
        registry=CustomsProgramRegistry.model_validate_json(json.dumps(data)),
    )
    rendered, proof = plan.render(source, {slot.slot_id: "GERMANY"})
    assert rendered == b"GERMANY IMPORTER TAX ID: 123\nCARGO IMPORTER (ACME) VAT NUMBER: 456\n"
    assert proof.every_slot_bound_once and proof.exact_literal_regions
    assert len(plan.evidence) == 1


def test_static_caption_overlap_is_verified_and_removed_from_expanded_slot_ownership():
    source = b"ACID NUMBER: 1234567890123456789\n"
    slot = build_template_slot(
        slot_id="slot_0000",
        byte_start=5,
        byte_end=11,
        source_text="NUMBER",
        target_paths=(),
        semantic_role="static_caption",
        evidence_origin="audited_source_auxiliary",
        render_policy="natural_text",
    )
    template = compile_raw_text_template(document_id="s", source=source, slots=[slot])
    plan = compile_neutral_customs(
        source=source,
        template=template,
        registry=registry(),
        static_slot_ids=frozenset({slot.slot_id}),
    )
    rendered, proof = plan.render(source, {slot.slot_id: slot.source_text})
    assert rendered == b"CUSTOMS REFERENCE: 1234567890123456789\n"
    assert proof.every_slot_bound_once and proof.exact_literal_regions
    with pytest.raises(ValueError, match="unchanged static"):
        plan.render(source, {slot.slot_id: "VALUE"})
