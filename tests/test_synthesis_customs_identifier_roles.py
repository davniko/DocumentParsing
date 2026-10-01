from __future__ import annotations

from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    _source_binding_relationships,
    _template_slots,
    normalize_labeled_country_code_derivations,
    normalize_labeled_customs_identifier_pair,
    validate_draft_source_alignment,
)


def _draft(
    raw: str,
    surface: str,
    *,
    key: str,
    kind: str = "identifier",
    mode: str = "agent_residual",
    policy: str = "opaque_identifier",
) -> SpanDraft:
    start = raw.rindex(surface)
    return SpanDraft(
        draft_id=key,
        logical_key=key,
        render_mode=mode,
        value_kind=kind,
        group_kind="customs",
        group_key="customs:shipment",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(surface),
        source_text=surface,
        evidence_origin="host_verified_agent_proposal",
        render_policy=policy,
        rationale="Complete value under printed customs caption.",
    )


def test_printed_acid_and_vat_prefix_are_separate_deterministic_identifiers() -> None:
    raw = (
        "ACID NUMBER: 4157387252025032066\n"
        "EGYPTIAN IMPORTER VAT NUMBER: 415738725\n"
    )
    drafts = (
        _draft(raw, "4157387252025032066", key="agent:page2:acid_number"),
        _draft(raw, "415738725", key="agent:page2:importer_vat"),
    )

    result = normalize_labeled_customs_identifier_pair(raw=raw, drafts=drafts)

    validate_draft_source_alignment(raw=raw, drafts=result)
    assert len(result) == 2
    assert all(row.render_mode == "deterministic_auxiliary" for row in result)
    assert {row.logical_key for row in result} == {row.logical_key for row in drafts}
    slots = _template_slots(raw, result)
    grouped = {
        row.logical_key: ((row, slot),)
        for row, slot in zip(result, slots, strict=True)
    }
    relationships = _source_binding_relationships(grouped)
    assert relationships["agent:page2:acid_number"][0].dependency_binding == (
        "agent:page2:importer_vat"
    )
    assert relationships["agent:page2:acid_number"][0].source_suffix == "2025032066"


def test_customs_identifier_promotion_requires_exact_captions_and_prefix() -> None:
    raw = (
        "ACID NUMBER: 4157387252025032066\n"
        "EGYPTIAN IMPORTER VAT NUMBER: 998738725\n"
    )
    drafts = (
        _draft(raw, "4157387252025032066", key="agent:page2:acid_number"),
        _draft(raw, "998738725", key="agent:page2:importer_vat"),
    )
    assert normalize_labeled_customs_identifier_pair(raw=raw, drafts=drafts) == drafts

    wrong_caption = raw.replace("EGYPTIAN IMPORTER VAT NUMBER", "OTHER VAT NUMBER")
    rows = (
        _draft(wrong_caption, "4157387252025032066", key="agent:page2:acid_number"),
        _draft(wrong_caption, "998738725", key="agent:page2:importer_vat"),
    )
    assert normalize_labeled_customs_identifier_pair(raw=wrong_caption, drafts=rows) == rows


def test_wrapped_country_heading_derives_country_code_from_same_scope_country() -> None:
    raw = (
        "FOREIGN EXPORTER CO\n"
        "UNTRY: UNITED ARAB EMIRATES\n"
        "FOREIGN EXPORTER COUNTRY CODE: AE\n"
    )
    country = _draft(
        raw,
        "UNITED ARAB EMIRATES",
        key="agent:page2:foreign_exporter_country",
        kind="location",
        mode="deterministic_auxiliary",
        policy="natural_text",
    )
    code = _draft(
        raw,
        "AE",
        key="agent:page2:foreign_exporter_country_code",
        mode="deterministic_auxiliary",
    )

    result = normalize_labeled_country_code_derivations(raw=raw, drafts=(country, code))

    validate_draft_source_alignment(raw=raw, drafts=result)
    assert result[0] == country
    assert result[1].render_mode == "deterministic_derived"
    assert result[1].derivation == "country_code"
    assert result[1].dependency_bindings == (country.logical_key,)
    assert result[1].render_policy == "derived_surface"


def test_wrapped_country_heading_requires_exact_contiguous_caption() -> None:
    raw = (
        "FOREIGN EXPORTER CO\n"
        "SOME OTHER TEXT\n"
        "UNTRY: UNITED ARAB EMIRATES\n"
        "FOREIGN EXPORTER COUNTRY CODE: AE\n"
    )
    country = _draft(
        raw,
        "UNITED ARAB EMIRATES",
        key="agent:page2:foreign_exporter_country",
        kind="location",
        mode="deterministic_auxiliary",
        policy="natural_text",
    )
    code = _draft(
        raw,
        "AE",
        key="agent:page2:foreign_exporter_country_code",
        mode="deterministic_auxiliary",
    )
    assert normalize_labeled_country_code_derivations(raw=raw, drafts=(country, code)) == (
        country,
        code,
    )
