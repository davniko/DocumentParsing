from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from document_ocr.synthesis.raw_text_template import build_template_slot

from raw_text_template_experiment.host import (
    RequiredTargetCoBinding,
    SpanDraft,
    _repair_uniquely_bracketed_target_paths,
    anchor_drafts,
    anchor_summary,
    annotated_source,
    apply_anchor_overrides,
    apply_critic_patch,
    binding_realization,
    certify_template,
    inventory_binding_id,
    merge_drafts,
    normalize_compact_equipment_locality,
    normalize_country_code_locality,
    normalize_deterministic_draft_semantics,
    normalize_target_cobindings,
    reconcile_draft_overlaps,
    required_target_cobindings,
    resolve_agent_proposals,
    risk_candidates,
    target_path_relationship,
    uncovered_risks,
    validate_agent_proposal_paths,
    validate_binding_realizations,
    validate_carrier_assessment,
    validate_compact_equipment_locality,
    validate_target_binding_relationships,
)
from raw_text_template_experiment.models import (
    AgentBindingProposal,
    AnchorOverride,
    CarrierAssessment,
    CriticAgentOutput,
    CriticFinding,
    CriticOccurrenceRemoval,
)


def _fixture():
    body = (
        "CARRIER\n"
        "Acme Ocean Lines Ltd.\n"
        "B/L No. ACME123456\n"
        "VAT: 12345678\n"
        "GENERAL CLAUSE 14 APPLIES\n"
    )
    raw = "--- PAGE 1 ---\n" + body

    def anchor(raw_value: str, path: str, role: str, anchor_id: str):
        start = body.index(raw_value)
        return {
            "anchor_id": anchor_id,
            "document_id": "doc_fixture",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(raw_value),
            "raw_value": raw_value,
            "relation_target_path": path,
            "role_path": role,
            "surface_family": "text",
        }

    anchors = [
        anchor(
            "Acme Ocean Lines Ltd.",
            "documentPatch.parties.carrier.name",
            "documentPatch.parties.carrier.name",
            "anchor_carrier",
        ),
        anchor(
            "ACME123456",
            "documentPatch.billOfLadingNumber",
            "documentPatch.billOfLadingNumber",
            "anchor_bl",
        ),
    ]
    target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "billOfLadingNumber": "ACME123456",
            "parties": {"carrier": {"name": "Acme Ocean Lines Ltd."}},
        },
    }
    feature = {
        "document_type": "bill_of_lading",
        "template_proxy_id": "template_fixture",
        "page_count": 1,
        "ocr_lines": 6,
        "ocr_characters": len(raw),
        "container_count": 0,
        "seal_count": 0,
        "goods_group_count": 0,
        "package_fact_count": 0,
        "allocation_group_count": 0,
        "allocation_row_count": 0,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
        "additional_information_group_count": 0,
        "marks_group_count": 0,
        "target_leaf_paths": ["billOfLadingNumber", "parties.carrier.name"],
        "carrier_family": "ACME",
    }
    return raw, anchors, target, feature


def _proposal(text: str, line: str, *, key: str, mode: str = "deterministic_auxiliary"):
    return AgentBindingProposal.model_validate(
        {
            "logical_key": key,
            "render_mode": mode,
            "value_kind": "identifier",
            "group_kind": "customs",
            "group_key": "customs:vat",
            "target_paths": (),
            "derivation": None,
            "dependency_paths": (),
            "occurrences": (
                {
                    "line_start": line,
                    "line_end": line,
                    "source_text": text,
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Source-only VAT identifier requires deterministic regeneration.",
        }
    )


def test_annotated_source_preserves_text_inside_binding_boundaries() -> None:
    raw, anchor_rows, _target, _feature = _fixture()
    drafts = anchor_drafts(raw=raw, document_id="doc_fixture", anchors=anchor_rows)

    annotated = annotated_source(raw, drafts)

    assert "Acme Ocean Lines Ltd." in annotated
    assert "ACME123456" in annotated
    assert "⟦anchor:documentPatch.parties.carrier.name:carrier_static⟧" in annotated
    assert annotated.count("⟦/binding⟧") == len(drafts)


def test_agent_addition_covers_risk_and_certifies_exact_template() -> None:
    raw, anchor_rows, target, feature = _fixture()
    anchors = anchor_drafts(raw=raw, document_id="doc_fixture", anchors=anchor_rows)
    risks = risk_candidates(raw, anchors)
    assert {risk.source_text for risk in risks} >= {"12345678"}

    additions = resolve_agent_proposals(
        raw=raw, proposals=(_proposal("12345678", "L00005", key="vat_reference"),)
    )
    drafts = merge_drafts(anchors, additions)
    assert not [
        risk for risk in uncovered_risks(raw, risks, drafts) if risk.source_text == "12345678"
    ]

    assessment = CarrierAssessment.model_validate(
        {
            "canonical_name": "Acme Ocean Lines Ltd.",
            "aliases": (),
            "evidence_occurrences": (
                {
                    "line_start": "L00003",
                    "line_end": "L00003",
                    "source_text": "Acme Ocean Lines Ltd.",
                    "occurrence_index": 0,
                },
            ),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact accepted carrier evidence.",
        }
    )
    critic = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "All source-specific values are owned.",
        }
    )
    template = certify_template(
        raw=raw,
        document_id="doc_fixture",
        feature=feature,
        source_target=target,
        assessment=assessment,
        drafts=drafts,
        risks=risks,
        critic_outputs=(critic,),
    )
    assert template.certification.source_round_trip is True
    assert template.certification.sentinel_isolation is True
    assert template.certification.all_bindings_realization_planned is True
    assert template.carrier.canonical_name == "Acme Ocean Lines Ltd."
    assert {binding.render_mode for binding in template.bindings} >= {
        "carrier_static",
        "target_binding",
        "deterministic_auxiliary",
    }
    realization_modes = {binding.realization.mode for binding in template.bindings}
    assert realization_modes >= {"static", "single_surface", "generated_auxiliary"}


def test_unowned_risk_and_overlapping_agent_span_fail_closed() -> None:
    raw, anchor_rows, _target, _feature = _fixture()
    anchors = anchor_drafts(raw=raw, document_id="doc_fixture", anchors=anchor_rows)
    risks = risk_candidates(raw, anchors)
    assert any(risk.source_text == "12345678" for risk in uncovered_risks(raw, risks, anchors))

    overlapping = resolve_agent_proposals(
        raw=raw,
        proposals=(
            _proposal(
                "ACME123456",
                "L00004",
                key="bad_overlap",
            ),
            _proposal(
                "Acme Ocean Lines Ltd.",
                "L00003",
                key="second_bad_overlap",
            ),
        ),
    )
    with pytest.raises(ValueError, match="overlap") as captured:
        merge_drafts(anchors, overlapping)
    assert captured.value.args[0].count("agent_binding_") == 2


def test_risk_candidate_accepts_complete_union_of_value_and_unit_slots() -> None:
    body = "Weight: 23,720.000 kgs\n"
    raw = "--- PAGE 1 ---\n" + body
    value_start = body.index("23,720.000")
    unit_start = body.index("kgs")
    rows = (
        {
            "anchor_id": "anchor_weight_value",
            "document_id": "doc_measurement_union",
            "patchable": True,
            "page_number": 1,
            "page_start": value_start,
            "page_end": value_start + len("23,720.000"),
            "raw_value": "23,720.000",
            "target_value": 23720.0,
            "relation_target_path": "documentPatch.cargoGroups[0].grossWeight.value",
            "role_path": "documentPatch.cargoGroups[0].grossWeight.value",
            "surface_family": "decimal_measure",
        },
        {
            "anchor_id": "anchor_weight_unit",
            "document_id": "doc_measurement_union",
            "patchable": True,
            "page_number": 1,
            "page_start": unit_start,
            "page_end": unit_start + len("kgs"),
            "raw_value": "kgs",
            "target_value": "kilogram",
            "relation_target_path": "documentPatch.cargoGroups[0].grossWeight.unit",
            "role_path": "documentPatch.cargoGroups[0].grossWeight.unit",
            "surface_family": "text",
        },
    )
    drafts = anchor_drafts(raw=raw, document_id="doc_measurement_union", anchors=rows)
    risks = risk_candidates(raw, ())
    measurement = next(row for row in risks if row.source_text == "23,720.000 kgs")
    assert measurement not in uncovered_risks(raw, risks, drafts)
    value_only = tuple(row for row in drafts if row.source_text == "23,720.000")
    assert measurement in uncovered_risks(raw, risks, value_only)


def test_iso_equipment_risk_does_not_absorb_seal_caption() -> None:
    raw = "--- PAGE 1 ---\nSEAL 478617\nARKU 8486899\n"
    risks = risk_candidates(raw, ())
    assert [(row.kind, row.source_text) for row in risks] == [
        ("long_numeric_identifier", "478617"),
        ("equipment_identifier", "ARKU 8486899"),
    ]


def test_agent_target_semantics_survive_host_group_canonicalization() -> None:
    raw = "--- PAGE 1 ---\nWWW.HAYAT.COM.TR\nBARCELONA\n3077\n40'CONTAINER\n"

    def proposal(
        text: str,
        line: str,
        *,
        key: str,
        value_kind: str,
        target_paths: tuple[str, ...],
    ) -> AgentBindingProposal:
        return AgentBindingProposal.model_validate(
            {
                "logical_key": key,
                "render_mode": "target_binding",
                "value_kind": value_kind,
                "group_kind": "other",
                "group_key": "agent-supplied-group-is-not-authoritative",
                "target_paths": target_paths,
                "occurrences": (
                    {
                        "line_start": line,
                        "line_end": line,
                        "source_text": text,
                        "occurrence_index": 0,
                    },
                ),
                "rationale": "Fixture target binding.",
            }
        )

    drafts = resolve_agent_proposals(
        raw=raw,
        proposals=(
            proposal(
                "WWW.HAYAT.COM.TR",
                "L00002",
                key="shipper_website",
                value_kind="url_or_domain",
                target_paths=("documentPatch.parties.shipper.contactDetails.websiteUrls[0]",),
            ),
            proposal(
                "BARCELONA",
                "L00003",
                key="place_of_issue",
                value_kind="location",
                target_paths=("documentPatch.placeOfIssue.name",),
            ),
            proposal(
                "3077",
                "L00004",
                key="dangerous_goods_un_number",
                value_kind="dangerous_goods",
                target_paths=("documentPatch.cargoGroups[0].dangerousGoods[0].unNumber",),
            ),
            proposal(
                "40'CONTAINER",
                "L00005",
                key="equipment_summary",
                value_kind="equipment",
                target_paths=tuple(
                    f"documentPatch.containers[{index}].typeDescription" for index in range(4)
                ),
            ),
        ),
    )
    by_text = {draft.source_text: draft for draft in drafts}
    assert by_text["WWW.HAYAT.COM.TR"].value_kind == "url_or_domain"
    assert (by_text["BARCELONA"].group_kind, by_text["BARCELONA"].group_key) == (
        "document",
        "document",
    )
    assert (by_text["3077"].group_kind, by_text["3077"].group_key) == (
        "dangerous_goods",
        "dangerous_goods:0:0",
    )
    assert by_text["40'CONTAINER"].group_kind == "equipment"
    assert by_text["40'CONTAINER"].group_key == (
        "compound:container:0|container:1|container:2|container:3"
    )


def test_critic_patch_replaces_only_cited_binding_and_preserves_target_ownership() -> None:
    body = "Acme Ocean Lines Ltd.\nPORT OF LOADING: ALEXANDRIA\nPORT OF DISCHARGE: ALEXANDRIA\n"
    raw = "--- PAGE 1 ---\n" + body
    carrier = "Acme Ocean Lines Ltd."
    location = "ALEXANDRIA"
    first_location = body.index(location)
    target_paths = (
        "documentPatch.route.portOfLoading.name",
        "documentPatch.route.portOfDischarge.name",
    )

    def anchor(raw_value: str, start: int, path: str, anchor_id: str) -> dict[str, object]:
        return {
            "anchor_id": anchor_id,
            "document_id": "doc_local_patch",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(raw_value),
            "raw_value": raw_value,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_local_patch",
        anchors=(
            anchor(
                carrier,
                body.index(carrier),
                "documentPatch.parties.carrier.name",
                "carrier",
            ),
            *(
                anchor(location, first_location, path, f"route_{index}")
                for index, path in enumerate(target_paths)
            ),
        ),
    )
    route_binding = next(draft for draft in anchors if set(draft.target_paths) == set(target_paths))
    carrier_binding = next(draft for draft in anchors if draft is not route_binding)
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": carrier}},
            "route": {
                "portOfLoading": {"name": location},
                "portOfDischarge": {"name": location},
            },
        }
    }
    finding = CriticFinding.model_validate(
        {
            "finding_kind": "topology_or_grouping_error",
            "line_ids": ("L00003", "L00004"),
            "evidence": "The two route roles have separate printed occurrences.",
            "explanation": "Replace the shared first occurrence with role-specific bindings.",
        }
    )

    def route_proposal(path: str, line: str) -> AgentBindingProposal:
        return AgentBindingProposal.model_validate(
            {
                "logical_key": path,
                "render_mode": "target_binding",
                "value_kind": "location",
                "group_kind": "route",
                "group_key": path,
                "target_paths": (path,),
                "occurrences": (
                    {
                        "line_start": line,
                        "line_end": line,
                        "source_text": location,
                        "occurrence_index": 0,
                    },
                ),
                "rationale": "The route role has its own printed surface.",
            }
        )

    revised = apply_critic_patch(
        raw=raw,
        drafts=anchors,
        findings=(finding,),
        remove_inventory_binding_ids=(inventory_binding_id(route_binding.logical_key),),
        additional_bindings=(
            route_proposal(target_paths[0], "L00003"),
            route_proposal(target_paths[1], "L00004"),
        ),
        source_target=target,
    )
    assert carrier_binding in revised
    assert route_binding not in revised
    assert {draft.target_paths for draft in revised} >= {
        (target_paths[0],),
        (target_paths[1],),
    }

    with pytest.raises(ValueError, match="outside its cited findings"):
        apply_critic_patch(
            raw=raw,
            drafts=anchors,
            findings=(finding.model_copy(update={"line_ids": ("L00004",)}),),
            remove_inventory_binding_ids=(inventory_binding_id(route_binding.logical_key),),
            additional_bindings=(route_proposal(target_paths[0], "L00003"),),
            source_target=target,
        )

    repeated_repair = AgentBindingProposal.model_validate(
        {
            "logical_key": "route:shared_repeated_location",
            "render_mode": "target_binding",
            "value_kind": "location",
            "group_kind": "route",
            "group_key": "route:shared_repeated_location",
            "target_paths": target_paths,
            "occurrences": (
                {
                    "line_start": "L00003",
                    "line_end": "L00003",
                    "source_text": location,
                    "occurrence_index": 0,
                },
                {
                    "line_start": "L00004",
                    "line_end": "L00004",
                    "source_text": location,
                    "occurrence_index": 0,
                },
            ),
            "rationale": "A cited missed repeat extends the existing logical value group.",
        }
    )
    with pytest.raises(ValueError, match="independently mutable target facts"):
        apply_critic_patch(
            raw=raw,
            drafts=anchors,
            findings=(finding,),
            remove_inventory_binding_ids=(inventory_binding_id(route_binding.logical_key),),
            additional_bindings=(repeated_repair,),
            source_target=target,
        )


def test_critic_can_append_a_cited_occurrence_without_restating_existing_binding() -> None:
    body = "B/L: SAME123456\nCOPY: SAME123456\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "SAME123456"
    path = "documentPatch.billOfLadingNumber"
    target = {"documentPatch": {"billOfLadingNumber": value}}
    anchor = anchor_drafts(
        raw=raw,
        document_id="doc_append_repeat",
        anchors=(
            {
                "anchor_id": "anchor_original",
                "document_id": "doc_append_repeat",
                "patchable": True,
                "page_number": 1,
                "page_start": body.index(value),
                "page_end": body.index(value) + len(value),
                "raw_value": value,
                "target_value": value,
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "text",
            },
        ),
    )[0]
    finding = CriticFinding.model_validate(
        {
            "finding_kind": "unowned_repeated_fact",
            "line_ids": ("L00003",),
            "evidence": "The copy repeats the bill of lading number.",
            "explanation": "Append only the newly discovered physical occurrence.",
        }
    )
    append = AgentBindingProposal.model_validate(
        {
            "logical_key": anchor.logical_key,
            "render_mode": anchor.render_mode,
            "value_kind": anchor.value_kind,
            "group_kind": anchor.group_kind,
            "group_key": anchor.group_key,
            "target_paths": anchor.target_paths,
            "occurrences": (
                {
                    "line_start": "L00003",
                    "line_end": "L00003",
                    "source_text": value,
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Append the cited repeat without restating the accepted occurrence.",
        }
    )

    revised = apply_critic_patch(
        raw=raw,
        drafts=(anchor,),
        findings=(finding,),
        remove_inventory_binding_ids=(),
        additional_bindings=(append,),
        source_target=target,
    )

    assert len(revised) == 2
    assert {draft.logical_key for draft in revised} == {anchor.logical_key}
    assert {draft.source_text for draft in revised} == {value}

    removal_finding = CriticFinding.model_validate(
        {
            "finding_kind": "incorrect_semantic_owner",
            "line_ids": ("L00003",),
            "evidence": "The copy is not semantically part of this binding.",
            "explanation": "Remove only the cited physical occurrence.",
        }
    )
    partial_removal = CriticOccurrenceRemoval.model_validate(
        {
            "logical_key": anchor.logical_key,
            "occurrences": (
                {
                    "line_start": "L00003",
                    "line_end": "L00003",
                    "source_text": value,
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Preserve the logical contract and its first occurrence.",
        }
    )
    cleaned = apply_critic_patch(
        raw=raw,
        drafts=revised,
        findings=(removal_finding,),
        remove_inventory_binding_ids=(),
        additional_bindings=(),
        occurrence_removals=(partial_removal,),
        source_target=target,
    )

    assert len(cleaned) == 1
    assert cleaned[0].char_start == anchor.char_start
    with pytest.raises(ValueError, match="outside its cited findings"):
        apply_critic_patch(
            raw=raw,
            drafts=revised,
            findings=(removal_finding.model_copy(update={"line_ids": ("L00002",)}),),
            remove_inventory_binding_ids=(),
            additional_bindings=(),
            occurrence_removals=(partial_removal,),
            source_target=target,
        )


def test_critic_can_replace_item_receipt_with_same_span_collection_count() -> None:
    body = "1 CONT.\nCOPY 1 CONT.\n"
    raw = "--- PAGE 1 ---\n" + body
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSCU1234567",
                    "typeDescription": "40HQ",
                }
            ]
        }
    }
    starts = tuple(match.start() for match in re.finditer(r"1 CONT\.", raw))
    existing = tuple(
        SpanDraft(
            draft_id=f"old_{index}",
            logical_key="agent:equipment_receipt:container:0",
            render_mode="deterministic_derived",
            value_kind="equipment",
            group_kind="equipment",
            group_key="container:0",
            target_paths=("documentPatch.containers[0]",),
            derivation="equipment_receipt",
            dependency_paths=("documentPatch.containers[0].typeDescription",),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len("1 CONT."),
            source_text="1 CONT.",
            evidence_origin="derived_operational_fact",
            render_policy="derived_surface",
            rationale="Fixture item-scoped equipment receipt.",
        )
        for index, start in enumerate(starts)
    )
    finding = CriticFinding.model_validate(
        {
            "finding_kind": "missing_derivation",
            "line_ids": ("L00002", "L00003"),
            "evidence": "Both surfaces are container counts without equipment type text.",
            "explanation": "Replace the item receipt with a collection count derivation.",
        }
    )
    replacement = AgentBindingProposal.model_validate(
        {
            "logical_key": "agent:container_count:documentPatch.containers",
            "render_mode": "deterministic_derived",
            "value_kind": "integer",
            "group_kind": "equipment",
            "group_key": "equipment:all",
            "target_paths": ("documentPatch.containers",),
            "derivation": "container_count",
            "dependency_paths": ("documentPatch.containers",),
            "occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "1 CONT.",
                    "occurrence_index": 0,
                },
                {
                    "line_start": "L00003",
                    "line_end": "L00003",
                    "source_text": "1 CONT.",
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Both cited surfaces derive the collection count.",
        }
    )

    revised = apply_critic_patch(
        raw=raw,
        drafts=existing,
        findings=(finding,),
        remove_inventory_binding_ids=(inventory_binding_id(existing[0].logical_key),),
        additional_bindings=(replacement,),
        source_target=target,
    )

    assert len(revised) == 2
    assert {row.derivation for row in revised} == {"container_count"}
    assert {row.target_paths for row in revised} == {("documentPatch.containers",)}


def test_critic_rejects_functional_noop_across_repeated_occurrences() -> None:
    body = "B/L: SAME123456\nCOPY: SAME123456\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "SAME123456"
    path = "documentPatch.billOfLadingNumber"
    target = {"documentPatch": {"billOfLadingNumber": value}}
    offsets = (body.index(value), body.rindex(value))
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_atomic_repeat",
        anchors=tuple(
            {
                "anchor_id": f"anchor_{index}",
                "document_id": "doc_atomic_repeat",
                "patchable": True,
                "page_number": 1,
                "page_start": offset,
                "page_end": offset + len(value),
                "raw_value": value,
                "target_value": value,
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "text",
            }
            for index, offset in enumerate(offsets)
        ),
    )
    assert len(anchors) == 2
    assert len({draft.logical_key for draft in anchors}) == 1
    finding = CriticFinding.model_validate(
        {
            "finding_kind": "topology_or_grouping_error",
            "line_ids": ("L00002", "L00003"),
            "evidence": "Both copies need one corrected repeated binding.",
            "explanation": "Replace the complete prior logical inventory entry.",
        }
    )
    replacement = AgentBindingProposal.model_validate(
        {
            "logical_key": "corrected_bill_number",
            "render_mode": "target_binding",
            "value_kind": "identifier",
            "group_kind": "document",
            "group_key": "document",
            "target_paths": (path,),
            "occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": value,
                    "occurrence_index": 0,
                },
                {
                    "line_start": "L00003",
                    "line_end": "L00003",
                    "source_text": value,
                    "occurrence_index": 0,
                },
            ),
            "rationale": "One document identifier is printed twice.",
        }
    )
    with pytest.raises(ValueError, match="functional no-op"):
        apply_critic_patch(
            raw=raw,
            drafts=anchors,
            findings=(finding,),
            remove_inventory_binding_ids=(inventory_binding_id(anchors[0].logical_key),),
            additional_bindings=(replacement,),
            source_target=target,
        )


def test_agent_occurrence_uses_global_match_only_when_exact_quote_is_unique() -> None:
    raw, _anchors, _target, _feature = _fixture()
    proposal = _proposal("12345678", "L00004", key="wrong_line")
    resolved = resolve_agent_proposals(raw=raw, proposals=(proposal,))
    assert len(resolved) == 1
    assert raw[resolved[0].char_start : resolved[0].char_end] == "12345678"

    ambiguous_raw = "--- PAGE 1 ---\nVAT: 12345678\nVAT COPY: 12345678\nOTHER\n"
    ambiguous = _proposal("12345678", "L00004", key="ambiguous_wrong_line")
    with pytest.raises(ValueError, match="exact quote resolution failed"):
        resolve_agent_proposals(raw=ambiguous_raw, proposals=(ambiguous,))


def test_phone_typo_recovers_only_the_unique_typed_value_on_declared_line() -> None:
    raw = "--- PAGE 1 ---\nT: 202 2268 3858, F: 202 2268 3850\n"
    proposal = _proposal("202 22684 3850", "L00002", key="agent_fax").model_copy(
        update={
            "value_kind": "phone",
            "group_kind": "party",
            "group_key": "party:agent:0",
        }
    )

    resolved = resolve_agent_proposals(raw=raw, proposals=(proposal,))

    assert len(resolved) == 1
    assert resolved[0].source_text == "202 2268 3850"
    assert "one-character phone transcription error" in resolved[0].rationale

    ambiguous_raw = "--- PAGE 1 ---\nT: 202 2268 3858, F: 202 2268 3850\n"
    ambiguous = _proposal("202 2268 3859", "L00002", key="agent_contact").model_copy(
        update={
            "value_kind": "phone",
            "group_kind": "party",
            "group_key": "party:agent:0",
        }
    )
    with pytest.raises(ValueError, match="cannot be resolved unambiguously"):
        resolve_agent_proposals(raw=ambiguous_raw, proposals=(ambiguous,))


def test_exact_quote_rejection_batches_every_defect_with_authoritative_ranges() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "Unit 3208, 32/F, The Octagon, 6 Sha Tsui Road\n"
        "SIGNED OCEAN NETWORK EXPRESS (EUROPE)\n"
        "BY: LTD. GERMANY BRANCH\n"
    )
    wrong_address = _proposal(
        "Unit 3208, 32/F, The Octagon, 6 Tsui Road",
        "L00002",
        key="carrier_address_repeat",
    )
    wrong_multiline = AgentBindingProposal.model_validate(
        {
            **_proposal("placeholder", "L00003", key="carrier_signature").model_dump(mode="python"),
            "occurrences": (
                {
                    "line_start": "L00003",
                    "line_end": "L00004",
                    "source_text": ("OCEAN NETWORK EXPRESS (EUROPE)\nLTD. GERMANY BRANCH"),
                    "occurrence_index": 0,
                },
            ),
        }
    )
    with pytest.raises(ValueError, match="correct every listed occurrence") as captured:
        resolve_agent_proposals(raw=raw, proposals=(wrong_address, wrong_multiline))
    message = str(captured.value)
    assert "carrier_address_repeat" in message
    assert "carrier_signature" in message
    assert "declaredRangeText" in message
    assert "6 Sha Tsui Road" in message
    assert "SIGNED OCEAN NETWORK EXPRESS" in message
    assert "BY: LTD. GERMANY BRANCH" in message


def test_overlapping_exact_matches_use_occurrence_index() -> None:
    raw = "--- PAGE 1 ---\nPG III, (25C.C.C.)\n"
    unit = _proposal("C", "L00002", key="temperature_unit")
    method = AgentBindingProposal.model_validate(
        {
            **_proposal("C.C.", "L00002", key="flash_point_method").model_dump(mode="python"),
            "occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "C.C.",
                    "occurrence_index": 1,
                },
            ),
        }
    )
    drafts = resolve_agent_proposals(raw=raw, proposals=(unit, method))
    unit_draft = next(row for row in drafts if row.logical_key == "agent:temperature_unit")
    method_draft = next(row for row in drafts if row.logical_key == "agent:flash_point_method")
    assert method_draft.char_start == unit_draft.char_start + 2
    merge_drafts(drafts)


def test_semantic_negotiability_evidence_is_not_a_renderable_party_span() -> None:
    raw, anchor_rows, _target, _feature = _fixture()
    consignee = "Acme Ocean Lines Ltd."
    start = raw.split("--- PAGE 1 ---\n", 1)[1].index(consignee)
    anchor_rows.append(
        {
            "anchor_id": "anchor_semantic_negotiability",
            "document_id": "doc_fixture",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(consignee),
            "raw_value": consignee,
            "target_value": "non_negotiable",
            "relation_target_path": "documentPatch.negotiability",
            "role_path": "documentPatch.negotiability",
            "surface_family": "text",
        }
    )
    drafts = anchor_drafts(raw=raw, document_id="doc_fixture", anchors=anchor_rows)
    carrier = next(row for row in drafts if row.render_mode == "carrier_static")
    assert carrier.target_paths == ("documentPatch.parties.carrier.name",)


def test_country_is_location_and_url_risk_excludes_closing_delimiter() -> None:
    body = "Country: Egypt\nPolicy: [https://example.com/update]\n"
    raw = "--- PAGE 1 ---\n" + body
    country_start = body.index("Egypt")
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_fixture",
        anchors=(
            {
                "anchor_id": "anchor_country",
                "document_id": "doc_fixture",
                "patchable": True,
                "page_number": 1,
                "page_start": country_start,
                "page_end": country_start + len("Egypt"),
                "raw_value": "Egypt",
                "target_value": "Egypt",
                "relation_target_path": "documentPatch.parties.shipper.country",
                "role_path": "documentPatch.parties.shipper.country",
                "surface_family": "text",
            },
        ),
    )
    assert anchors[0].value_kind == "location"
    risks = risk_candidates(raw, anchors)
    assert any(row.source_text == "https://example.com/update" for row in risks)
    assert all(not row.source_text.endswith("]") for row in risks)


def test_agent_can_attach_repeated_occurrence_to_existing_anchor_key() -> None:
    raw, _anchor_rows, _target, _feature = _fixture()
    proposal = _proposal("12345678", "L00005", key="anchor:documentPatch.customs.vat")
    draft = resolve_agent_proposals(raw=raw, proposals=(proposal,))[0]
    assert draft.logical_key == "anchor:documentPatch.customs.vat"


def test_anchor_override_requires_complete_replacement_target_ownership() -> None:
    raw, anchor_rows, target, _feature = _fixture()
    anchors = anchor_drafts(raw=raw, document_id="doc_fixture", anchors=anchor_rows)
    bill = next(row for row in anchors if row.target_paths == ("documentPatch.billOfLadingNumber",))
    replacement = AgentBindingProposal.model_validate(
        {
            "logical_key": "document:bill_of_lading_number",
            "render_mode": "target_binding",
            "value_kind": "identifier",
            "group_kind": "document",
            "group_key": "document",
            "target_paths": ("documentPatch.billOfLadingNumber",),
            "derivation": None,
            "dependency_paths": (),
            "dependency_bindings": (),
            "occurrences": (
                {
                    "line_start": "L00004",
                    "line_end": "L00004",
                    "source_text": "ACME123456",
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Relocate a source-evidence anchor under explicit override.",
        }
    )
    validate_agent_proposal_paths(proposals=(replacement,), source_target=target)
    proposed = resolve_agent_proposals(raw=raw, proposals=(replacement,))
    assert proposed[0].logical_key == "anchor:documentPatch.billOfLadingNumber"
    assert proposed[0].value_kind == "identifier"
    override = AnchorOverride.model_validate(
        {"anchor_binding_id": bill.draft_id, "rationale": "Fixture relocation."}
    )
    merged = apply_anchor_overrides(
        raw=raw,
        source_target=target,
        anchors=anchors,
        proposed=proposed,
        overrides=(override,),
    )
    assert bill.draft_id not in {row.draft_id for row in merged}
    assert any(row.target_paths == bill.target_paths for row in merged)

    with pytest.raises(ValueError, match="lacks replacement target ownership") as captured:
        apply_anchor_overrides(
            raw=raw,
            source_target=target,
            anchors=anchors,
            proposed=(),
            overrides=(override,),
        )
    assert "L00004='ACME123456'" in str(captured.value)


def test_agent_target_paths_must_resolve_against_source_label() -> None:
    _raw, _anchors, target, _feature = _fixture()
    invalid = AgentBindingProposal.model_validate(
        {
            "logical_key": "missing",
            "render_mode": "target_binding",
            "value_kind": "identifier",
            "group_kind": "document",
            "group_key": "document",
            "target_paths": ("documentPatch.missingField",),
            "derivation": None,
            "dependency_paths": (),
            "dependency_bindings": (),
            "occurrences": (
                {
                    "line_start": "L00001",
                    "line_end": "L00001",
                    "source_text": "PAGE",
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Invalid fixture target path.",
        }
    )
    with pytest.raises(ValueError, match="does not exist"):
        validate_agent_proposal_paths(proposals=(invalid,), source_target=target)


def test_overlap_reconciliation_partitions_provable_natural_target_projection() -> None:
    raw = "--- PAGE 1 ---\nALSO NOTIFY: SAMSUNG SDS EGYPT\n"
    outer_text = "SAMSUNG SDS EGYPT"
    inner_text = "EGYPT"
    outer_start = raw.index(outer_text)
    inner_start = raw.index(inner_text)
    source_target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {"name": outer_text, "country": inner_text},
                ]
            }
        }
    }
    outer = SpanDraft(
        draft_id="outer",
        logical_key="anchor:documentPatch.parties.notifyParties[0].name",
        render_mode="target_binding",
        value_kind="organization",
        group_kind="party",
        group_key="party:notify:0",
        target_paths=("documentPatch.parties.notifyParties[0].name",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=outer_start,
        char_end=outer_start + len(outer_text),
        source_text=outer_text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Complete source party name.",
    )
    inner = replace(
        outer,
        draft_id="inner",
        logical_key="anchor:documentPatch.parties.notifyParties[0].country",
        value_kind="location",
        target_paths=("documentPatch.parties.notifyParties[0].country",),
        char_start=inner_start,
        char_end=inner_start + len(inner_text),
        source_text=inner_text,
        rationale="Country suffix has independent target ownership.",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(outer, inner),
        source_target=source_target,
    )

    assert [(row.logical_key, row.source_text) for row in reconciled] == [
        ("anchor:documentPatch.parties.notifyParties[0].name", "SAMSUNG SDS"),
        ("anchor:documentPatch.parties.notifyParties[0].country", "EGYPT"),
    ]
    validate_binding_realizations(
        raw=raw,
        drafts=reconciled,
        source_target=source_target,
    )


def test_overlap_reconciliation_transfers_only_provably_false_repeated_target_span() -> None:
    raw = "--- PAGE 1 ---\nCargo: ASSY OPEN CELL\nTax: 84899112\nCargo: ASSY OPEN CELL\n"
    source_target = {"documentPatch": {"cargoGroups": [{"description": "ASSY OPEN CELL"}]}}

    def cargo(source_text: str, occurrence: int) -> SpanDraft:
        offsets = tuple(match.start() for match in re.finditer(re.escape(source_text), raw))
        start = offsets[occurrence]
        return SpanDraft(
            draft_id=f"cargo_{occurrence}_{source_text}",
            logical_key="anchor:documentPatch.cargoGroups[0].description",
            render_mode="target_binding",
            value_kind="cargo_text",
            group_kind="cargo",
            group_key="cargo:0",
            target_paths=("documentPatch.cargoGroups[0].description",),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(source_text),
            source_text=source_text,
            evidence_origin="accepted_label_evidence",
            render_policy="natural_text",
            rationale="Accepted target evidence.",
        )

    valid_first = cargo("ASSY OPEN CELL", 0)
    valid_second = cargo("ASSY OPEN CELL", 1)
    false_target = cargo("84899112", 0)
    auxiliary = replace(
        false_target,
        draft_id="tax_auxiliary",
        logical_key="agent:notify_tax_id",
        render_mode="deterministic_auxiliary",
        value_kind="identifier",
        group_kind="party",
        group_key="party:notify:0",
        target_paths=(),
        evidence_origin="host_verified_agent_proposal",
        render_policy="opaque_identifier",
        rationale="Context identifies the exact tax value.",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(valid_first, false_target, auxiliary, valid_second),
        source_target=source_target,
    )

    assert false_target.draft_id not in {row.draft_id for row in reconciled}
    assert auxiliary in reconciled
    assert {valid_first.draft_id, valid_second.draft_id} <= {row.draft_id for row in reconciled}

    with pytest.raises(ValueError, match="template spans overlap"):
        reconcile_draft_overlaps(
            raw=raw,
            drafts=(false_target, auxiliary),
            source_target=source_target,
        )


def test_deterministic_semantic_normalization_stops_count_and_location_oscillation() -> None:
    raw = "--- PAGE 1 ---\n2 containers\nTAIWAN\nTAIWAN, PROVINCE OF CHINA\n"
    source_target = {
        "documentPatch": {
            "containers": [{"containerNumber": "A"}, {"containerNumber": "B"}],
            "cargoPackages": [{"quantity": 10}],
        }
    }
    count_start = raw.index("2 containers")
    count = SpanDraft(
        draft_id="count",
        logical_key="agent:container_receipt",
        render_mode="deterministic_derived",
        value_kind="integer",
        group_kind="equipment",
        group_key="equipment:all",
        target_paths=("documentPatch.containers",),
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.containers",),
        dependency_bindings=(),
        char_start=count_start,
        char_end=count_start + len("2 containers"),
        source_text="2 containers",
        evidence_origin="derived_operational_fact",
        render_policy="derived_surface",
        rationale="Compiler count classification.",
    )

    def location(source_text: str, occurrence: int) -> SpanDraft:
        offsets = tuple(match.start() for match in re.finditer(re.escape(source_text), raw))
        start = offsets[occurrence]
        return SpanDraft(
            draft_id=f"location_{occurrence}_{len(source_text)}",
            logical_key="agent:exporter_registration_country",
            render_mode="agent_residual",
            value_kind="location",
            group_kind="customs",
            group_key="customs:exporter",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(source_text),
            source_text=source_text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Compiler grouped source-only country variants.",
        )

    normalized = normalize_deterministic_draft_semantics(
        drafts=(
            count,
            location("TAIWAN", 0),
            location("TAIWAN, PROVINCE OF CHINA", 0),
        ),
        source_target=source_target,
    )

    normalized_count = next(row for row in normalized if row.draft_id == "count")
    assert normalized_count.derivation == "container_count"
    assert normalized_count.dependency_paths == ("documentPatch.containers",)
    locations = [row for row in normalized if row.value_kind == "location"]
    assert len({row.logical_key for row in locations}) == 2
    assert {row.render_mode for row in locations} == {"deterministic_auxiliary"}
    assert {row.group_key for row in locations} == {"customs:exporter"}
    validate_binding_realizations(
        raw=raw,
        drafts=normalized,
        source_target=source_target,
    )


def test_overlap_reconciliation_keeps_retained_anchor_over_redundant_extension() -> None:
    raw = "--- PAGE 1 ---\nAddress: MAIN STREET, 511300\n"
    anchor_text = "MAIN STREET"
    proposal_text = "MAIN STREET, 511300"
    start = raw.index(anchor_text)
    source_target = {
        "documentPatch": {
            "parties": {"shipper": {"address": proposal_text}},
        }
    }
    anchor = SpanDraft(
        draft_id="retained_anchor",
        logical_key="anchor:documentPatch.parties.shipper.address",
        render_mode="target_binding",
        value_kind="address",
        group_kind="party",
        group_key="party:shipper",
        target_paths=("documentPatch.parties.shipper.address",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(anchor_text),
        source_text=anchor_text,
        evidence_origin="accepted_label_evidence",
        render_policy="natural_text",
        rationale="Accepted target projection.",
    )
    redundant = replace(
        anchor,
        draft_id="redundant_agent_extension",
        char_end=start + len(proposal_text),
        source_text=proposal_text,
        evidence_origin="host_verified_agent_proposal",
        rationale="Unnecessary extension without an explicit anchor override.",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(anchor, redundant),
        source_target=source_target,
    )

    assert reconciled == (anchor,)


def test_unlabelled_decimal_table_is_not_misclassified_as_phone() -> None:
    raw = "--- PAGE 1 ---\n70.5560 169.00\n"
    risks = risk_candidates(raw, ())
    assert all(row.kind != "phone" for row in risks)


def test_real_wrong_duplicate_anchor_can_be_relocated_without_overlap() -> None:
    project_root = Path(__file__).resolve().parents[4]
    document_id = "doc_09585226e5552adad580b1d71b70fdf80b3aee7a85263a49dd3b8c82909c13b1"
    source_path = project_root / (
        "artifacts/kie-training/datasets/"
        "mpci-bl-combined2174-task-facing-package-categories-v1/records.jsonl"
    )
    anchor_path = project_root / (
        "artifacts/kie-synthesis/mpci-bl-combined2174-synthesis-preparation-v1/"
        "anchors/ocr-anchors.jsonl"
    )
    source = next(
        row
        for row in (json.loads(line) for line in source_path.open(encoding="utf-8"))
        if row["documentId"] == document_id
    )
    anchor_rows = [
        row
        for row in (json.loads(line) for line in anchor_path.open(encoding="utf-8"))
        if row["document_id"] == document_id
    ]
    raw = source["joinedRawText"]
    anchors = anchor_drafts(raw=raw, document_id=document_id, anchors=anchor_rows)
    defective = next(
        row
        for row in anchors
        if set(row.target_paths)
        == {
            "documentPatch.parties.shipper.city",
            "documentPatch.parties.shipper.name",
        }
    )
    proposals = (
        AgentBindingProposal.model_validate(
            {
                "logical_key": "party:shipper:0:name",
                "render_mode": "target_binding",
                "value_kind": "organization",
                "group_kind": "party",
                "group_key": "party:shipper:0",
                "target_paths": ("documentPatch.parties.shipper.name",),
                "derivation": None,
                "dependency_paths": (),
                "dependency_bindings": (),
                "occurrences": (
                    {
                        "line_start": "L00005",
                        "line_end": "L00005",
                        "source_text": "BEV GMBH - BERLINER ENGINEERING UND VERTRIEB",
                        "occurrence_index": 0,
                    },
                ),
                "rationale": "Own only the complete printed shipper name.",
            }
        ),
        AgentBindingProposal.model_validate(
            {
                "logical_key": "party:shipper:0:city",
                "render_mode": "target_binding",
                "value_kind": "location",
                "group_kind": "party",
                "group_key": "party:shipper:0",
                "target_paths": ("documentPatch.parties.shipper.city",),
                "derivation": None,
                "dependency_paths": (),
                "dependency_bindings": (),
                "occurrences": (
                    {
                        "line_start": "L00007",
                        "line_end": "L00007",
                        "source_text": "BERLIN",
                        "occurrence_index": 0,
                    },
                ),
                "rationale": "Relocate city from the substring in BERLINER to its city field.",
            }
        ),
    )
    validate_agent_proposal_paths(proposals=proposals, source_target=source["target"])
    replacements = resolve_agent_proposals(raw=raw, proposals=proposals)
    merged = apply_anchor_overrides(
        raw=raw,
        source_target=source["target"],
        anchors=anchors,
        proposed=replacements,
        overrides=(
            AnchorOverride.model_validate(
                {
                    "anchor_binding_id": defective.draft_id,
                    "rationale": "The accepted city evidence selected BERLIN inside BERLINER.",
                }
            ),
        ),
    )
    relocated = {
        path: row.source_text
        for row in merged
        for path in row.target_paths
        if path
        in {
            "documentPatch.parties.shipper.city",
            "documentPatch.parties.shipper.name",
        }
    }
    assert relocated == {
        "documentPatch.parties.shipper.city": "BERLIN",
        "documentPatch.parties.shipper.name": ("BEV GMBH - BERLINER ENGINEERING UND VERTRIEB"),
    }


def test_missing_source_label_carrier_requires_exact_bound_ocr_evidence() -> None:
    raw = "--- PAGE 1 ---\nSigned for Carrier: Blue Sea Lines Ltd.\n"
    proposal = AgentBindingProposal.model_validate(
        {
            "logical_key": "carrier:principal:name",
            "render_mode": "carrier_static",
            "value_kind": "organization",
            "group_kind": "carrier",
            "group_key": "carrier:principal",
            "target_paths": (),
            "derivation": None,
            "dependency_paths": (),
            "dependency_bindings": (),
            "occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "Blue Sea Lines Ltd.",
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Explicit printed carrier principal.",
        }
    )
    drafts = resolve_agent_proposals(raw=raw, proposals=(proposal,))
    assessment = CarrierAssessment.model_validate(
        {
            "canonical_name": "Blue Sea Lines Ltd.",
            "aliases": (),
            "evidence_occurrences": proposal.occurrences,
            "source": "ocr_resolved_missing_source_label",
            "rationale": "The signature explicitly identifies the carrier principal.",
        }
    )
    validate_carrier_assessment(
        assessment=assessment,
        expected=None,
        raw=raw,
        anchor_drafts_value=drafts,
    )

    wrong_source = assessment.model_copy(update={"source": "source_label_confirmed_by_ocr"})
    with pytest.raises(ValueError, match="assessment source"):
        validate_carrier_assessment(
            assessment=wrong_source,
            expected=None,
            raw=raw,
            anchor_drafts_value=drafts,
        )

    normalized_instead_of_exact = assessment.model_copy(
        update={"canonical_name": "blue sea lines ltd"}
    )
    with pytest.raises(ValueError, match="not copied exactly"):
        validate_carrier_assessment(
            assessment=normalized_instead_of_exact,
            expected=None,
            raw=raw,
            anchor_drafts_value=drafts,
        )

    hallucinated_alias = assessment.model_copy(update={"aliases": ("BSL",)})
    with pytest.raises(ValueError, match="alias is not copied exactly"):
        validate_carrier_assessment(
            assessment=hallucinated_alias,
            expected=None,
            raw=raw,
            anchor_drafts_value=drafts,
        )

    target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"parties": {}},
    }
    feature = {
        "document_type": "bill_of_lading",
        "template_proxy_id": "template_missing_carrier_fixture",
        "page_count": 1,
        "ocr_lines": 2,
        "ocr_characters": len(raw),
        "container_count": 0,
        "seal_count": 0,
        "goods_group_count": 0,
        "package_fact_count": 0,
        "allocation_group_count": 0,
        "allocation_row_count": 0,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
        "additional_information_group_count": 0,
        "marks_group_count": 0,
        "target_leaf_paths": [],
        "carrier_family": "<MISSING>",
    }
    critic = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "All source-specific values are owned.",
        }
    )
    template = certify_template(
        raw=raw,
        document_id="doc_missing_carrier_fixture",
        feature=feature,
        source_target=target,
        assessment=assessment,
        drafts=drafts,
        risks=risk_candidates(raw, ()),
        critic_outputs=(critic,),
    )
    assert template.carrier.family == "OCR_RESOLVED::Blue Sea Lines Ltd."
    assert template.certification.carrier_matches_source_label is None


def test_pinned_carrier_identity_accepts_different_exact_ocr_typography() -> None:
    body = "BLUE SEA LINES LLC\n"
    raw = "--- PAGE 1 ---\n" + body
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_carrier_typography",
        anchors=(
            {
                "anchor_id": "carrier",
                "document_id": "doc_carrier_typography",
                "patchable": True,
                "page_number": 1,
                "page_start": 0,
                "page_end": len(body.rstrip()),
                "raw_value": body.rstrip(),
                "target_value": "Blue Sea Lines Limited",
                "relation_target_path": "documentPatch.parties.carrier.name",
                "role_path": "documentPatch.parties.carrier.name",
                "surface_family": "text",
            },
        ),
    )
    assessment = CarrierAssessment.model_validate(
        {
            "canonical_name": "Blue Sea Lines Limited",
            "aliases": (),
            "evidence_occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "BLUE SEA LINES LLC",
                    "occurrence_index": 0,
                },
            ),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "The pinned carrier target owns this exact OCR typography.",
        }
    )
    validate_carrier_assessment(
        assessment=assessment,
        expected="Blue Sea Lines Limited",
        raw=raw,
        anchor_drafts_value=anchors,
    )


def test_one_physical_surface_can_encode_an_explicit_target_equality() -> None:
    body = "Container: MSKU7843379\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "MSKU7843379"
    start = body.index(value)
    paths = (
        "documentPatch.containers[0].containerNumber",
        "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
    )
    rows = tuple(
        {
            "anchor_id": f"anchor_{index}",
            "document_id": "doc_shared_surface",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(value),
            "raw_value": value,
            "target_value": value,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "iso6346_compact",
        }
        for index, path in enumerate(paths)
    )
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": value}],
            "cargoAllocationGroups": [{"allocations": [{"containerNumber": value}]}],
        }
    }
    drafts = anchor_drafts(raw=raw, document_id="doc_shared_surface", anchors=rows)
    assert len(drafts) == 1
    assert drafts[0].render_policy == "opaque_identifier"
    assert drafts[0].value_kind == "equipment"
    assert target_path_relationship(target, drafts[0].target_paths) == "shared_value_equality"
    assert anchor_summary(raw, drafts, target)[0]["targetRelationship"] == ("shared_value_equality")
    validate_target_binding_relationships(drafts=drafts, source_target=target)

    unequal = {
        "documentPatch": {
            "containers": [{"containerNumber": value}],
            "cargoAllocationGroups": [{"allocations": [{"containerNumber": "MSKU0000000"}]}],
        }
    }
    assert target_path_relationship(unequal, drafts[0].target_paths) == ("composite_target_surface")
    with pytest.raises(ValueError, match="combines unequal target values"):
        validate_target_binding_relationships(drafts=drafts, source_target=unequal)

    residual = replace(drafts[0], render_mode="agent_residual")
    validate_target_binding_relationships(drafts=(residual,), source_target=unequal)


def _realization_draft(
    *,
    target_paths: tuple[str, ...],
    value_kind: str,
    render_policy: str,
    render_mode: str = "target_binding",
) -> SpanDraft:
    return SpanDraft(
        draft_id="fixture_draft",
        logical_key="fixture:binding",
        render_mode=render_mode,
        value_kind=value_kind,
        group_kind="document",
        group_key="document",
        target_paths=target_paths,
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=0,
        char_end=1,
        source_text="fixture",
        evidence_origin="accepted_label_evidence",
        render_policy=render_policy,
        rationale="Realization fixture.",
    )


def _realization_slots(*surfaces: str, render_policy: str = "natural_text"):
    cursor = 0
    slots = []
    for index, surface in enumerate(surfaces, start=1):
        encoded_size = len(surface.encode("utf-8"))
        slots.append(
            build_template_slot(
                slot_id=f"slot_{index:04d}",
                byte_start=cursor,
                byte_end=cursor + encoded_size,
                source_text=surface,
                target_paths=("documentPatch.fixture",),
                semantic_role="document",
                evidence_origin="accepted_label_evidence",
                render_policy=render_policy,
            )
        )
        cursor += encoded_size
    return tuple(slots)


def test_realization_contract_covers_segments_repeats_frames_and_agent_gate() -> None:
    address_path = ("documentPatch.parties.shipper.address",)
    address = binding_realization(
        draft=_realization_draft(
            target_paths=address_path,
            value_kind="address",
            render_policy="natural_text",
        ),
        slots=_realization_slots("AJMAN INDUSTRIAL AREA", "5368", "Al Rashidiya 1"),
        source_target={
            "documentPatch": {
                "parties": {"shipper": {"address": "AJMAN INDUSTRIAL AREA 5368 Al Rashidiya 1"}}
            }
        },
    )
    assert address.mode == "segmented_surface"
    assert tuple(slot.source_token_count for slot in address.slots) == (3, 1, 3)
    assert tuple(slot.segment_index for slot in address.slots) == (0, 1, 2)

    package_path = ("documentPatch.cargoPackages[0].typeCategory",)
    package = binding_realization(
        draft=_realization_draft(
            target_paths=package_path,
            value_kind="package",
            render_policy="categorical_surface",
        ),
        slots=_realization_slots("CARTON", "CARTON", render_policy="categorical_surface"),
        source_target={"documentPatch": {"cargoPackages": [{"typeCategory": "PACKAGE_CARTON"}]}},
    )
    assert package.mode == "repeated_surface"
    assert package.adapter == "package_category"
    assert {slot.value_role for slot in package.slots} == {"repeat"}

    freight_path = ("documentPatch.freight.paymentArrangement",)
    freight = binding_realization(
        draft=_realization_draft(
            target_paths=freight_path,
            value_kind="commercial_text",
            render_policy="natural_text",
        ),
        slots=_realization_slots("FREIGHT PREPAID"),
        source_target={"documentPatch": {"freight": {"paymentArrangement": "prepaid"}}},
    )
    assert freight.mode == "single_surface"
    assert freight.slots[0].literal_prefix == "FREIGHT "
    assert freight.slots[0].literal_suffix == ""

    unit_path = ("documentPatch.cargoGroups[0].grossWeight.unit",)
    unit = binding_realization(
        draft=_realization_draft(
            target_paths=unit_path,
            value_kind="decimal_measurement",
            render_policy="natural_text",
        ),
        slots=_realization_slots("KGS"),
        source_target={"documentPatch": {"cargoGroups": [{"grossWeight": {"unit": "kilogram"}}]}},
    )
    assert unit.mode == "single_surface"
    assert unit.adapter == "measurement_unit"

    additional_information_path = ("documentPatch.cargoGroups[0].additionalInformation[0]",)
    projected = binding_realization(
        draft=_realization_draft(
            target_paths=additional_information_path,
            value_kind="cargo_text",
            render_policy="natural_text",
        ),
        slots=_realization_slots("MATERIAL 13672297", "13672297"),
        source_target={
            "documentPatch": {"cargoGroups": [{"additionalInformation": ["MATERIAL 13672297"]}]}
        },
    )
    assert projected.mode == "token_projected_surface"
    assert projected.requires_agent is False
    assert projected.slots[0].required_target_prefix_tokens == ()
    assert projected.slots[1].required_target_prefix_tokens == ("material",)
    assert all(slot.required_target_suffix_tokens == () for slot in projected.slots)

    equipment_path = ("documentPatch.containers[0].typeDescription",)
    equipment = binding_realization(
        draft=_realization_draft(
            target_paths=equipment_path,
            value_kind="equipment",
            render_policy="natural_text",
        ),
        slots=_realization_slots("40HQ", "1X40'HQ CONTAINER"),
        source_target={"documentPatch": {"containers": [{"typeDescription": "1X40'HQ CONTAINER"}]}},
    )
    assert equipment.mode == "normalized_projected_surface"
    assert equipment.adapter == "equipment_type"
    assert equipment.requires_agent is False
    assert equipment.slots[0].required_target_prefix_normalized == "1x"
    assert equipment.slots[0].required_target_suffix_normalized == "container"
    assert equipment.slots[1].required_target_prefix_normalized == ""
    assert equipment.slots[1].required_target_suffix_normalized == ""

    unsupported = binding_realization(
        draft=_realization_draft(
            target_paths=("documentPatch.transport.vesselName",),
            value_kind="other_text",
            render_policy="natural_text",
        ),
        slots=_realization_slots("UNRELATED PRINTED SURFACE"),
        source_target={"documentPatch": {"transport": {"vesselName": "EXPECTED VESSEL"}}},
    )
    assert unsupported.mode == "agent_required"
    assert unsupported.requires_agent is True


def test_one_to_one_package_quantity_requires_one_logical_owner() -> None:
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    package_path = "documentPatch.cargoPackages[0].quantity"
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "allocations": [
                        {
                            "containerNumber": None,
                            "packageId": "p1",
                            "packageQuantity": 96,
                        }
                    ],
                    "coverage": "one_to_one_package_allocations",
                    "groupId": "g1",
                    "packageIds": ["p1"],
                }
            ],
        }
    }
    allocation = _realization_draft(
        target_paths=(allocation_path,),
        value_kind="integer",
        render_policy="numeric_surface",
    )
    package = replace(
        allocation,
        draft_id="fixture_package",
        logical_key="fixture:package",
        target_paths=(package_path,),
        char_start=2,
        char_end=3,
    )
    with pytest.raises(ValueError, match="one_to_one_package_quantity split"):
        validate_target_binding_relationships(drafts=(allocation, package), source_target=target)

    normalized = normalize_target_cobindings(drafts=(allocation, package), source_target=target)
    assert len({draft.logical_key for draft in normalized}) == 1
    assert all(draft.target_paths == (allocation_path, package_path) for draft in normalized)
    validate_target_binding_relationships(drafts=normalized, source_target=target)

    combined = replace(
        allocation,
        target_paths=(allocation_path, package_path),
    )
    validate_target_binding_relationships(drafts=(combined,), source_target=target)


def test_exact_span_complementary_quantity_owners_join_before_overlap_rejection() -> None:
    raw = "96\n"
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    package_path = "documentPatch.cargoPackages[0].quantity"
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 96,
                }
            ],
            "cargoAllocationGroups": [
                {
                    "allocations": [
                        {
                            "packageId": "p1",
                            "packageQuantity": 96,
                        }
                    ],
                    "coverage": "one_to_one_package_allocations",
                    "groupId": "g1",
                    "packageIds": ["p1"],
                }
            ],
        }
    }
    allocation = replace(
        _realization_draft(
            target_paths=(allocation_path,),
            value_kind="integer",
            render_policy="numeric_surface",
        ),
        draft_id="anchor_binding_0001",
        logical_key="anchor:" + allocation_path,
        char_end=2,
        source_text="96",
    )
    package = replace(
        allocation,
        draft_id="agent_package_quantity",
        logical_key="anchor:" + package_path,
        target_paths=(package_path,),
        evidence_origin="host_verified_agent_proposal",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(allocation, package),
        source_target=target,
    )

    assert len(reconciled) == 1
    assert reconciled[0].target_paths == tuple(sorted((allocation_path, package_path)))
    validate_target_binding_relationships(drafts=reconciled, source_target=target)


def test_single_package_single_allocation_quantity_requires_one_logical_owner() -> None:
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    package_path = "documentPatch.cargoPackages[0].quantity"
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 20,
                    "typeCategory": "PACKAGE_PALLET",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "allocations": [
                        {
                            "containerNumber": "MRKU0811125",
                            "packageQuantity": 20,
                        }
                    ],
                    "coverage": "single_package_level",
                    "groupId": "g1",
                    "packageIds": ["p1"],
                }
            ],
            "containers": [{"containerNumber": "MRKU0811125"}],
        }
    }
    allocation = _realization_draft(
        target_paths=(allocation_path,),
        value_kind="integer",
        render_policy="numeric_surface",
    )
    package = replace(
        allocation,
        draft_id="fixture_single_package",
        logical_key="fixture:single_package",
        target_paths=(package_path,),
        char_start=2,
        char_end=3,
    )

    with pytest.raises(ValueError, match="single_package_single_allocation_quantity split"):
        validate_target_binding_relationships(drafts=(allocation, package), source_target=target)

    normalized = normalize_target_cobindings(drafts=(allocation, package), source_target=target)
    assert len({draft.logical_key for draft in normalized}) == 1
    assert all(draft.target_paths == (allocation_path, package_path) for draft in normalized)
    validate_target_binding_relationships(drafts=normalized, source_target=target)


def test_single_package_multiple_allocations_remain_independent_sum_components() -> None:
    allocation_paths = tuple(
        f"documentPatch.cargoAllocationGroups[0].allocations[{index}].packageQuantity"
        for index in range(2)
    )
    package_path = "documentPatch.cargoPackages[0].quantity"
    target = {
        "documentPatch": {
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 20}],
            "cargoAllocationGroups": [
                {
                    "allocations": [
                        {"containerNumber": "MSKU0000001", "packageQuantity": 8},
                        {"containerNumber": "MSKU0000002", "packageQuantity": 12},
                    ],
                    "coverage": "single_package_level",
                    "groupId": "g1",
                    "packageIds": ["p1"],
                }
            ],
            "containers": [
                {"containerNumber": "MSKU0000001"},
                {"containerNumber": "MSKU0000002"},
            ],
        }
    }
    owners = tuple(
        replace(
            _realization_draft(
                target_paths=(path,),
                value_kind="integer",
                render_policy="numeric_surface",
            ),
            draft_id=f"fixture_allocation_{index}",
            logical_key=f"fixture:allocation:{index}",
            char_start=index * 2,
            char_end=index * 2 + 1,
        )
        for index, path in enumerate(allocation_paths)
    )
    package = replace(
        owners[0],
        draft_id="fixture_package_total",
        logical_key="fixture:package_total",
        target_paths=(package_path,),
        char_start=4,
        char_end=5,
    )

    normalized = normalize_target_cobindings(drafts=(*owners, package), source_target=target)

    assert len({draft.logical_key for draft in normalized}) == 3
    assert required_target_cobindings(target) == (
        RequiredTargetCoBinding(
            relationship="allocation_container_identity",
            target_paths=(
                "documentPatch.containers[0].containerNumber",
                "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
            ),
        ),
        RequiredTargetCoBinding(
            relationship="allocation_container_identity",
            target_paths=(
                "documentPatch.containers[1].containerNumber",
                "documentPatch.cargoAllocationGroups[0].allocations[1].containerNumber",
            ),
        ),
    )


def test_normalization_does_not_bridge_independent_equal_quantity_rows() -> None:
    allocation_paths = tuple(
        f"documentPatch.cargoAllocationGroups[{index}].allocations[0].packageQuantity"
        for index in range(2)
    )
    package_paths = tuple(f"documentPatch.cargoPackages[{index}].quantity" for index in range(2))
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": f"g{index + 1}",
                    "packageId": f"p{index + 1}",
                    "quantity": 96,
                }
                for index in range(2)
            ],
            "cargoAllocationGroups": [
                {
                    "allocations": [
                        {
                            "packageId": f"p{index + 1}",
                            "packageQuantity": 96,
                        }
                    ],
                    "coverage": "one_to_one_package_allocations",
                    "groupId": f"g{index + 1}",
                    "packageIds": [f"p{index + 1}"],
                }
                for index in range(2)
            ],
        }
    }
    broad_package_owner = replace(
        _realization_draft(
            target_paths=package_paths,
            value_kind="integer",
            render_policy="numeric_surface",
        ),
        logical_key="fixture:all_equal_packages",
    )
    allocation_owners = tuple(
        replace(
            broad_package_owner,
            draft_id=f"fixture_allocation_{index}",
            logical_key=f"fixture:allocation:{index}",
            target_paths=(allocation_path,),
            char_start=2 + index * 2,
            char_end=3 + index * 2,
        )
        for index, allocation_path in enumerate(allocation_paths)
    )
    normalized = normalize_target_cobindings(
        drafts=(broad_package_owner, *allocation_owners), source_target=target
    )
    assert len({draft.logical_key for draft in normalized}) == 3
    with pytest.raises(ValueError, match="one_to_one_package_quantity split"):
        validate_target_binding_relationships(drafts=normalized, source_target=target)


def test_repeated_binding_cannot_aggregate_independently_mutable_rows() -> None:
    paths = (
        "documentPatch.cargoPackages[0].typeCategory",
        "documentPatch.cargoPackages[1].typeCategory",
    )
    target = {
        "documentPatch": {
            "cargoPackages": [
                {"packageId": "p1", "typeCategory": "PACKAGE_CARTON"},
                {"packageId": "p2", "typeCategory": "PACKAGE_CARTON"},
            ],
            "cargoAllocationGroups": [],
        }
    }
    first = _realization_draft(
        target_paths=paths,
        value_kind="package",
        render_policy="categorical_surface",
    )
    repeated = replace(
        first,
        draft_id="fixture_repeat",
        char_start=2,
        char_end=3,
    )
    with pytest.raises(ValueError, match="independently mutable target facts"):
        validate_target_binding_relationships(drafts=(first, repeated), source_target=target)

    validate_target_binding_relationships(
        drafts=(
            replace(first, render_mode="agent_residual"),
            replace(repeated, render_mode="agent_residual"),
        ),
        source_target=target,
    )

    # One actual summary occurrence may intentionally constrain both equal source fields.
    validate_target_binding_relationships(drafts=(first,), source_target=target)

    derived_first = replace(
        first,
        render_mode="deterministic_derived",
        derivation="sum_package_quantity",
        dependency_paths=paths,
    )
    validate_target_binding_relationships(
        drafts=(
            derived_first,
            replace(
                repeated,
                render_mode="deterministic_derived",
                derivation="sum_package_quantity",
                dependency_paths=paths,
            ),
        ),
        source_target=target,
    )


def test_occurrence_resolution_retains_source_unicode_case_variant() -> None:
    raw = "DFDS DFDS DENİZCILİK ve TAŞIMACILIK A.S."
    proposal = AgentBindingProposal.model_validate(
        {
            "logical_key": "carrier",
            "render_mode": "carrier_static",
            "value_kind": "organization",
            "group_kind": "party",
            "group_key": "party:carrier",
            "target_paths": ("documentPatch.parties.carrier.name",),
            "occurrences": (
                {
                    "line_start": "L00001",
                    "line_end": "L00001",
                    "source_text": "DFDS DFDS DENİZCİLİK ve TAŞIMACILIK A.S.",
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Exact selected carrier line with a copied Unicode case variant.",
        }
    )

    (draft,) = resolve_agent_proposals(raw=raw, proposals=(proposal,))

    assert draft.char_start == 0
    assert draft.char_end == len(raw)
    assert draft.source_text == raw

    changed_letter = proposal.model_copy(
        update={
            "occurrences": (
                proposal.occurrences[0].model_copy(update={"source_text": raw[:-1] + "X"}),
            )
        }
    )
    with pytest.raises(ValueError, match="exact quote resolution failed"):
        resolve_agent_proposals(raw=raw, proposals=(changed_letter,))


def test_occurrence_resolution_prunes_only_proven_redundant_duplicate() -> None:
    raw = "2405\nunrelated\n2405"
    proposal = AgentBindingProposal.model_validate(
        {
            "logical_key": "package_quantity",
            "render_mode": "deterministic_auxiliary",
            "value_kind": "integer",
            "group_kind": "cargo",
            "group_key": "cargo:package",
            "occurrences": (
                {
                    "line_start": "L00001",
                    "line_end": "L00001",
                    "source_text": "2405",
                },
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "2405",
                },
                {
                    "line_start": "L00003",
                    "line_end": "L00003",
                    "source_text": "2405",
                },
            ),
            "rationale": "Two real repeats plus one hallucinated duplicate declaration.",
        }
    )

    drafts = resolve_agent_proposals(raw=raw, proposals=(proposal,))

    assert tuple(draft.source_text for draft in drafts) == ("2405", "2405")
    assert tuple(draft.char_start for draft in drafts) == (0, 15)

    incomplete = proposal.model_copy(update={"occurrences": proposal.occurrences[:2]})
    with pytest.raises(ValueError, match="exact quote resolution failed"):
        resolve_agent_proposals(raw=raw, proposals=(incomplete,))


def test_realization_validation_rejects_mixed_equipment_and_false_determinism() -> None:
    raw = "--- PAGE 1 ---\nFCL / FCL\n/FCL/FCL /40HQ/\nREF-A1\nREF-B2\n"
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": "1X40'HQ CONTAINER"}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    movement_start = raw.index("FCL / FCL")
    row_start = raw.index("/FCL/FCL /40HQ/")
    mixed = SpanDraft(
        draft_id="mixed_1",
        logical_key="agent:movement",
        render_mode="deterministic_auxiliary",
        value_kind="operational_text",
        group_kind="transport",
        group_key="transport:movement",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=movement_start,
        char_end=movement_start + len("FCL / FCL"),
        source_text="FCL / FCL",
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Fixture movement.",
    )
    mixed_repeat = replace(
        mixed,
        draft_id="mixed_2",
        char_start=row_start,
        char_end=row_start + len("/FCL/FCL /40HQ/"),
        source_text="/FCL/FCL /40HQ/",
    )
    with pytest.raises(ValueError, match="declared deterministic_auxiliary"):
        validate_binding_realizations(raw=raw, drafts=(mixed, mixed_repeat), source_target=target)

    source_only_equipment = replace(
        mixed_repeat,
        logical_key="agent:mixed_row",
        render_mode="agent_residual",
    )
    with pytest.raises(ValueError, match="equipment-type token as source-only"):
        validate_binding_realizations(
            raw=raw, drafts=(source_only_equipment,), source_target=target
        )

    missing_target = {
        "documentPatch": {
            "containers": [
                {"typeDescription": "1X40'HQ CONTAINER"},
                {"containerNumber": "MSCU1234567"},
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    scoped_auxiliary = replace(
        mixed_repeat,
        logical_key="agent:equipment_type:1",
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:1",
        char_start=raw.index("40HQ"),
        char_end=raw.index("40HQ") + len("40HQ"),
        source_text="40HQ",
    )
    validate_binding_realizations(raw=raw, drafts=(scoped_auxiliary,), source_target=missing_target)

    equipment_start = raw.index("40HQ")
    target_bound = replace(
        mixed,
        draft_id="equipment",
        logical_key="anchor:documentPatch.containers[0].typeDescription",
        render_mode="target_binding",
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
        target_paths=("documentPatch.containers[0].typeDescription",),
        char_start=equipment_start,
        char_end=equipment_start + len("40HQ"),
        source_text="40HQ",
    )
    movement_only_start = raw.index("FCL/FCL", row_start)
    movement_only = replace(
        mixed,
        char_start=movement_only_start,
        char_end=movement_only_start + len("FCL/FCL"),
        source_text="FCL/FCL",
    )
    validate_binding_realizations(
        raw=raw,
        drafts=(mixed, movement_only, target_bound),
        source_target=target,
    )


def test_realization_validation_accepts_solver_owned_embedded_identifiers() -> None:
    raw = "--- PAGE 1 ---\nVAT 452568234\nACID 4525682342024020037\n"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    vat_start = raw.index("452568234")
    vat = SpanDraft(
        draft_id="vat",
        logical_key="agent:egypt_import_vat_number",
        render_mode="deterministic_auxiliary",
        value_kind="identifier",
        group_kind="customs",
        group_key="customs:egypt_import",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=vat_start,
        char_end=vat_start + len("452568234"),
        source_text="452568234",
        evidence_origin="host_verified_agent_proposal",
        render_policy="opaque_identifier",
        rationale="Fixture VAT.",
    )
    acid_start = raw.index("4525682342024020037")
    acid = replace(
        vat,
        draft_id="acid",
        logical_key="agent:acid_reference",
        group_key="customs:acid",
        char_start=acid_start,
        char_end=acid_start + len("4525682342024020037"),
        source_text="4525682342024020037",
        rationale="Fixture ACID.",
    )
    validate_binding_realizations(raw=raw, drafts=(vat, acid), source_target=target)

    validate_binding_realizations(
        raw=raw,
        drafts=(vat, replace(acid, render_mode="agent_residual")),
        source_target=target,
    )

    bol_raw = "--- PAGE 1 ---\n4114749850 OOLU4114749850\n"
    booking_start = bol_raw.index("4114749850")
    booking = replace(
        vat,
        draft_id="booking",
        logical_key="agent:booking_reference",
        group_kind="commercial",
        group_key="commercial:booking",
        char_start=booking_start,
        char_end=booking_start + len("4114749850"),
        source_text="4114749850",
    )
    bol_start = bol_raw.index("OOLU4114749850")
    bill_number = replace(
        vat,
        draft_id="bill_number",
        logical_key="anchor:documentPatch.billOfLadingNumber",
        render_mode="target_binding",
        group_kind="document",
        group_key="document",
        target_paths=("documentPatch.billOfLadingNumber",),
        char_start=bol_start,
        char_end=bol_start + len("OOLU4114749850"),
        source_text="OOLU4114749850",
    )
    bol_target = {
        "documentPatch": {
            "billOfLadingNumber": "OOLU4114749850",
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    validate_binding_realizations(
        raw=bol_raw,
        drafts=(booking, bill_number),
        source_target=bol_target,
    )


def test_country_code_derivation_rejects_caption_substring_but_allows_value() -> None:
    raw = "--- PAGE 1 ---\nSHIPPER COUNTRY CODE: DE\n"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def country_code(start: int, text: str = "DE") -> SpanDraft:
        return SpanDraft(
            draft_id="country_code",
            logical_key="agent:shipper_country_code",
            render_mode="deterministic_derived",
            value_kind="identifier",
            group_kind="party",
            group_key="party:shipper:0",
            target_paths=(),
            derivation="country_code",
            dependency_paths=(),
            dependency_bindings=("agent:shipper_country",),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="derived_operational_fact",
            render_policy="derived_surface",
            rationale="Fixture country code.",
        )

    with pytest.raises(ValueError, match="caption/word fragment"):
        validate_binding_realizations(
            raw=raw,
            drafts=(country_code(raw.index("DE")),),
            source_target=target,
        )
    validate_binding_realizations(
        raw=raw,
        drafts=(country_code(raw.rindex("DE")),),
        source_target=target,
    )

    concatenated = "--- PAGE 1 ---\nEXPORTER COUNTRY CODE:CNTRAINER LINES\n"
    validate_binding_realizations(
        raw=concatenated,
        drafts=(country_code(concatenated.index("CN"), text="CN"),),
        source_target=target,
    )


def test_compact_equipment_locality_normalizes_only_missing_neighbor_type() -> None:
    raw = (
        "--- PAGE 1 ---\nAAAU1234567\n40HQ\n"
        "CARGO DESCRIPTION SPACER\nBBBU1234567\n1X40'HQ CONTAINER\n"
    )
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAU1234567"},
                {
                    "containerNumber": "BBBU1234567",
                    "typeDescription": "1X40'HQ CONTAINER",
                },
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def target_draft(
        *, draft_id: str, logical_key: str, text: str, target_path: str, group_key: str
    ) -> SpanDraft:
        start = raw.index(text)
        return SpanDraft(
            draft_id=draft_id,
            logical_key=logical_key,
            render_mode="target_binding",
            value_kind="equipment",
            group_kind="equipment",
            group_key=group_key,
            target_paths=(target_path,),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="opaque_identifier",
            rationale="Equipment locality fixture.",
        )

    number_0 = target_draft(
        draft_id="number_0",
        logical_key="container:0:number",
        text="AAAU1234567",
        target_path="documentPatch.containers[0].containerNumber",
        group_key="container:0",
    )
    number_1 = target_draft(
        draft_id="number_1",
        logical_key="container:1:number",
        text="BBBU1234567",
        target_path="documentPatch.containers[1].containerNumber",
        group_key="container:1",
    )
    compact = target_draft(
        draft_id="compact",
        logical_key="container:1:type",
        text="40HQ",
        target_path="documentPatch.containers[1].typeDescription",
        group_key="container:1",
    )
    full = target_draft(
        draft_id="full",
        logical_key="container:1:type",
        text="1X40'HQ CONTAINER",
        target_path="documentPatch.containers[1].typeDescription",
        group_key="container:1",
    )
    drafts = (number_0, compact, number_1, full)
    with pytest.raises(ValueError, match="compact equipment locality violations"):
        validate_binding_realizations(raw=raw, drafts=drafts, source_target=target)

    normalized = normalize_compact_equipment_locality(raw=raw, drafts=drafts, source_target=target)
    relocated = next(row for row in normalized if row.source_text == "40HQ")
    assert relocated.render_mode == "deterministic_auxiliary"
    assert relocated.group_key == "container:0"
    assert relocated.target_paths == ()
    remaining_type = [
        row
        for row in normalized
        if row.target_paths == ("documentPatch.containers[1].typeDescription",)
    ]
    assert [row.source_text for row in remaining_type] == ["1X40'HQ CONTAINER"]
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_topology_repair_assigns_repeated_scalar_to_unique_entity_rows() -> None:
    raw = "--- PAGE 1 ---\nDESC A\nCODE 2401108590\nGRADE A\nDESC B\nCODE 2401108590\nGRADE B\n"
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"description": "DESC A", "hsCodes": ["2401108590"], "marksAndNumbers": "GRADE A"},
                {"description": "DESC B", "hsCodes": ["2401108590"], "marksAndNumbers": "GRADE B"},
            ],
        }
    }
    contexts = tuple(
        _test_draft(raw, text, "anchor:" + path, "target_binding", (path,))
        for text, path in (
            ("DESC A", "documentPatch.cargoGroups[0].description"),
            ("GRADE A", "documentPatch.cargoGroups[0].marksAndNumbers"),
            ("DESC B", "documentPatch.cargoGroups[1].description"),
            ("GRADE B", "documentPatch.cargoGroups[1].marksAndNumbers"),
        )
    )
    missing = (
        "documentPatch.cargoGroups[0].hsCodes[0]",
        "documentPatch.cargoGroups[1].hsCodes[0]",
    )

    repairs = _repair_uniquely_bracketed_target_paths(
        raw=raw, missing_paths=missing, occupied=contexts, source_target=target
    )

    assert [row.target_paths[0] for row in repairs] == list(missing)
    assert [row.char_start for row in repairs] == [
        raw.index("2401108590"),
        raw.rindex("2401108590"),
    ]


def test_near_ocr_address_variants_split_without_agent() -> None:
    raw = "--- PAGE 1 ---\nHELICOPOLIS, CAIRO\nHELIOPOLIS, CAIRO\n"
    first = replace(
        _test_draft(raw, "HELICOPOLIS, CAIRO", "agent:iss:address", "deterministic_auxiliary"),
        value_kind="address",
        group_kind="party",
        group_key="party:agent:0",
    )
    second = replace(
        _test_draft(raw, "HELIOPOLIS, CAIRO", "agent:iss:address", "deterministic_auxiliary"),
        value_kind="address",
        group_kind="party",
        group_key="party:agent:0",
    )
    target = {"documentPatch": {"containers": [], "cargoAllocationGroups": [], "cargoPackages": []}}

    normalized = normalize_deterministic_draft_semantics(
        drafts=(first, second), source_target=target
    )

    assert len({row.logical_key for row in normalized}) == 2
    assert {row.render_mode for row in normalized} == {"deterministic_auxiliary"}
    assert {row.group_key for row in normalized} == {"party:agent:0"}
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_exact_repeated_multiline_target_projection_splits_into_deterministic_occurrences() -> None:
    repeated = "13672297\n13672297"
    raw = "--- PAGE 1 ---\nMATERIAL\n" + repeated + "\n"
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
            "cargoGroups": [{"additionalInformation": ["MATERIAL 13672297"]}],
        }
    }
    draft = replace(
        _test_draft(raw, repeated, "anchor:" + path, "target_binding", (path,)),
        value_kind="cargo_text",
        group_kind="cargo",
        group_key="cargo:0",
        render_policy="natural_text",
    )

    normalized = normalize_deterministic_draft_semantics(drafts=(draft,), source_target=target)

    assert len(normalized) == 2
    assert {row.source_text for row in normalized} == {"13672297"}
    assert len({row.char_start for row in normalized}) == 2
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_exact_composite_party_surface_splits_into_independent_target_bindings() -> None:
    surface = "ZI la massane\n13210 Saint Remy De Provence Union"
    raw = "--- PAGE 1 ---\n" + surface + "\n"
    paths = (
        "documentPatch.parties.shipper.address",
        "documentPatch.parties.shipper.city",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
            "parties": {
                "shipper": {
                    "address": "ZI la massane 13210",
                    "city": "Saint Remy De Provence Union",
                }
            },
        }
    }
    draft = replace(
        _test_draft(raw, surface, "agent:shipper:address_city", "agent_residual", paths),
        value_kind="address",
        group_kind="party",
        group_key="party:shipper:0",
    )

    normalized = normalize_deterministic_draft_semantics(drafts=(draft,), source_target=target)

    assert {row.target_paths for row in normalized} == {(paths[0],), (paths[1],)}
    assert {row.source_text for row in normalized} == {
        "ZI la massane\n13210",
        "Saint Remy De Provence Union",
    }
    assert {row.render_mode for row in normalized} == {"target_binding"}
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_temperature_object_surface_expands_to_host_derived_leaf_pair() -> None:
    raw = "--- PAGE 1 ---\nSETPOINT +1,0 C\n"
    base = "documentPatch.containers[0].temperatureSetpoint"
    target = {
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    draft = replace(
        _test_draft(raw, "+1,0 C", "anchor:" + base, "target_binding", (base,)),
        value_kind="temperature",
        group_kind="equipment",
        group_key="container:0",
        render_policy="numeric_surface",
    )

    normalized = normalize_deterministic_draft_semantics(drafts=(draft,), source_target=target)

    assert len(normalized) == 1
    assert normalized[0].render_mode == "deterministic_derived"
    assert normalized[0].derivation == "temperature_setpoint"
    assert normalized[0].target_paths == (base + ".unit", base + ".value")
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_identifier_group_uses_one_canonical_opaque_format_policy() -> None:
    raw = "--- PAGE 1 ---\n3214.10.0010\nCOPY 3214.10.0010\n"
    path = "documentPatch.cargoGroups[0].hsCodes[0]"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
            "cargoGroups": [{"hsCodes": ["3214100010"]}],
        }
    }
    starts = tuple(match.start() for match in re.finditer(r"3214\.10\.0010", raw))
    drafts = tuple(
        replace(
            _test_draft(raw, "3214.10.0010", "anchor:" + path, "target_binding", (path,)),
            draft_id=f"hs_{index}",
            char_start=start,
            char_end=start + len("3214.10.0010"),
            value_kind="identifier",
            group_kind="cargo",
            group_key="cargo:0",
            render_policy="numeric_surface" if index == 0 else "opaque_identifier",
        )
        for index, start in enumerate(starts)
    )

    normalized = normalize_deterministic_draft_semantics(drafts=drafts, source_target=target)
    realization = binding_realization(
        draft=normalized[0],
        slots=_realization_slots("3214.10.0010", "3214.10.0010"),
        source_target=target,
    )

    assert {row.render_policy for row in normalized} == {"opaque_identifier"}
    assert realization.deterministic is True
    assert realization.mode == "repeated_surface"


def test_temperature_value_and_unit_cobinding_becomes_deterministic_derivation() -> None:
    raw = "--- PAGE 1 ---\nTEMP +1 C\nSETPOINT +1,0 C\n"
    paths = (
        "documentPatch.containers[0].temperatureSetpoint.unit",
        "documentPatch.containers[0].temperatureSetpoint.value",
    )
    target = {
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    drafts = tuple(
        replace(
            _test_draft(raw, text, "anchor:temperature", "target_binding"),
            value_kind="temperature",
            group_kind="equipment",
            group_key="container:0",
            target_paths=paths,
            render_policy="numeric_surface",
        )
        for text in ("+1 C", "+1,0 C")
    )

    normalized = normalize_deterministic_draft_semantics(drafts=drafts, source_target=target)

    assert {row.render_mode for row in normalized} == {"deterministic_derived"}
    assert {row.derivation for row in normalized} == {"temperature_setpoint"}
    assert {row.dependency_paths for row in normalized} == {paths}
    validate_target_binding_relationships(drafts=normalized, source_target=target)


def test_container_count_derivation_and_noun_surface_share_one_canonical_owner() -> None:
    raw = "--- PAGE 1 ---\nCOUNT 1\n1 CONTAINER(S)\n"
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "MSCU1234567"}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    bare = replace(
        _test_draft(raw, "1", "agent:bare_count", "deterministic_derived"),
        char_start=raw.index("COUNT 1") + len("COUNT "),
        char_end=raw.index("COUNT 1") + len("COUNT 1"),
        value_kind="integer",
        group_kind="equipment",
        group_key="equipment:container_count",
        target_paths=("documentPatch.containers",),
        derivation="container_count",
        dependency_paths=("documentPatch.containers",),
        render_policy="derived_surface",
    )
    noun = replace(
        _test_draft(raw, "1 CONTAINER(S)", "agent:noun_count", "deterministic_derived"),
        value_kind="equipment",
        group_kind="equipment",
        group_key="equipment:container_count",
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.containers",),
        render_policy="derived_surface",
    )

    normalized = normalize_deterministic_draft_semantics(drafts=(bare, noun), source_target=target)

    assert len(normalized) == 2
    assert {row.logical_key for row in normalized} == {
        "agent:container_count:documentPatch.containers"
    }
    assert {row.value_kind for row in normalized} == {"integer"}
    assert {row.derivation for row in normalized} == {"container_count"}
    assert {row.target_paths for row in normalized} == {("documentPatch.containers",)}
    validate_target_binding_relationships(drafts=normalized, source_target=target)


def test_country_code_locality_moves_caption_fragment_to_unique_labeled_value() -> None:
    raw = "--- PAGE 1 ---\nSHIPPER COUNTRY CODE: DE\n"
    caption_fragment = raw.index("DE")
    draft = replace(
        _test_draft(raw, "DE", "agent:shipper_country_code", "deterministic_derived"),
        char_start=caption_fragment,
        char_end=caption_fragment + 2,
        value_kind="identifier",
        group_kind="party",
        group_key="party:shipper:0",
        derivation="country_code",
        dependency_paths=("documentPatch.parties.shipper.country",),
        render_policy="derived_surface",
    )

    normalized = normalize_country_code_locality(raw=raw, drafts=(draft,))

    assert len(normalized) == 1
    assert normalized[0].char_start == raw.rindex("DE")
    assert normalized[0].source_text == "DE"


def test_topology_repair_rejects_one_source_span_claimed_by_two_entities() -> None:
    raw = "--- PAGE 1 ---\nA0 A1 2401108590 B1 B0\n"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {"description": "A0", "hsCodes": ["2401108590"], "marks": "B0"},
                {"description": "A1", "hsCodes": ["2401108590"], "marks": "B1"},
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def context(text: str, path: str) -> SpanDraft:
        return replace(
            _test_draft(raw, text, "anchor:" + path, "target_binding"),
            target_paths=(path,),
            group_kind="cargo",
            group_key=path.rsplit(".", 1)[0],
        )

    repairs = _repair_uniquely_bracketed_target_paths(
        raw=raw,
        missing_paths=(
            "documentPatch.cargoGroups[0].hsCodes[0]",
            "documentPatch.cargoGroups[1].hsCodes[0]",
        ),
        occupied=(
            context("A0", "documentPatch.cargoGroups[0].description"),
            context("B0", "documentPatch.cargoGroups[0].marks"),
            context("A1", "documentPatch.cargoGroups[1].description"),
            context("B1", "documentPatch.cargoGroups[1].marks"),
        ),
        source_target=target,
    )

    assert repairs == ()


def test_topology_repair_assigns_repeated_trailing_values_by_entity_context() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "DESC-A\nHS-A\nGRADE SAME\n"
        "DESC-B\nHS-B\nGRADE SAME\n"
        "DESC-C\nHS-C\nGRADE SAME\n"
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "description": f"DESC-{suffix}",
                    "hsCodes": [f"HS-{suffix}"],
                    "additionalInformation": ["GRADE SAME"],
                }
                for suffix in "ABC"
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def context(text: str, path: str) -> SpanDraft:
        return replace(
            _test_draft(raw, text, "anchor:" + path, "target_binding"),
            target_paths=(path,),
            group_kind="cargo",
            group_key=path.rsplit(".", 1)[0],
        )

    occupied = tuple(
        draft
        for index, suffix in enumerate("ABC")
        for draft in (
            context(
                f"DESC-{suffix}",
                f"documentPatch.cargoGroups[{index}].description",
            ),
            context(
                f"HS-{suffix}",
                f"documentPatch.cargoGroups[{index}].hsCodes[0]",
            ),
        )
    )
    missing = tuple(
        f"documentPatch.cargoGroups[{index}].additionalInformation[0]" for index in range(3)
    )

    repairs = _repair_uniquely_bracketed_target_paths(
        raw=raw,
        missing_paths=missing,
        occupied=occupied,
        source_target=target,
    )

    assert tuple(row.target_paths[0] for row in repairs) == missing
    assert tuple(row.char_start for row in repairs) == tuple(
        match.start() for match in re.finditer("GRADE SAME", raw)
    )


def test_topology_repair_uses_trusted_anchor_surface_for_ocr_whitespace() -> None:
    raw = "--- PAGE 1 ---\nDESCRIPTION\n2401108590\nGRADE\nEGRFMY4LR\n"
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "description": "DESCRIPTION",
                    "hsCodes": ["2401108590"],
                    "additionalInformation": ["GRADE EGRFMY4LR"],
                }
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    occupied = (
        replace(
            _test_draft(raw, "DESCRIPTION", "anchor:description", "target_binding"),
            target_paths=("documentPatch.cargoGroups[0].description",),
        ),
        replace(
            _test_draft(raw, "2401108590", "anchor:hs", "target_binding"),
            target_paths=("documentPatch.cargoGroups[0].hsCodes[0]",),
        ),
    )

    repairs = _repair_uniquely_bracketed_target_paths(
        raw=raw,
        missing_paths=(path,),
        occupied=occupied,
        source_target=target,
        source_hints={path: ("GRADE\nEGRFMY4LR",)},
    )

    assert len(repairs) == 1
    assert repairs[0].source_text == "GRADE\nEGRFMY4LR"


def test_compact_equipment_locality_remaps_same_line_target_permutation() -> None:
    raw = "--- PAGE 1 ---\nAAAU1234567 40HQ\nBBBU1234567 20GP\n"
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAU1234567", "typeDescription": "40HQ"},
                {"containerNumber": "BBBU1234567", "typeDescription": "20GP"},
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def row_draft(*, draft_id: str, text: str, target_path: str, occurrence: int = 0) -> SpanDraft:
        start = [match.start() for match in re.finditer(re.escape(text), raw)][occurrence]
        return SpanDraft(
            draft_id=draft_id,
            logical_key="anchor:" + target_path,
            render_mode="target_binding",
            value_kind="equipment" if target_path.endswith("typeDescription") else "identifier",
            group_kind="equipment",
            group_key="container:fixture",
            target_paths=(target_path,),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="opaque_identifier",
            rationale="Same-line locality fixture.",
        )

    drafts = (
        row_draft(
            draft_id="number_0",
            text="AAAU1234567",
            target_path="documentPatch.containers[0].containerNumber",
        ),
        row_draft(
            draft_id="number_1",
            text="BBBU1234567",
            target_path="documentPatch.containers[1].containerNumber",
        ),
        row_draft(
            draft_id="type_wrong_1",
            text="40HQ",
            target_path="documentPatch.containers[1].typeDescription",
        ),
        row_draft(
            draft_id="type_wrong_0",
            text="20GP",
            target_path="documentPatch.containers[0].typeDescription",
        ),
    )
    normalized = normalize_compact_equipment_locality(raw=raw, drafts=drafts, source_target=target)
    type_paths = {
        row.source_text: row.target_paths
        for row in normalized
        if row.source_text in {"40HQ", "20GP"}
    }
    assert type_paths == {
        "40HQ": ("documentPatch.containers[0].typeDescription",),
        "20GP": ("documentPatch.containers[1].typeDescription",),
    }
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_compact_equipment_locality_does_not_cross_container_list_into_cargo_block() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "AAAU1234567\nNET WEIGHT\n1000\n"
        "BBBU1234567\nNET WEIGHT\n2000\n\n"
        "2 x 40HR CONTAINER\n"
    )
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAU1234567", "typeDescription": "40HR"},
                {"containerNumber": "BBBU1234567", "typeDescription": "40HR"},
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def draft(draft_id: str, text: str, path: str) -> SpanDraft:
        start = raw.index(text)
        return SpanDraft(
            draft_id=draft_id,
            logical_key="anchor:" + path,
            render_mode="target_binding",
            value_kind="equipment" if path.endswith("typeDescription") else "identifier",
            group_kind="equipment",
            group_key="container:0",
            target_paths=(path,),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="opaque_identifier",
            rationale="Distant-list locality fixture.",
        )

    drafts = (
        draft("number_0", "AAAU1234567", "documentPatch.containers[0].containerNumber"),
        draft("number_1", "BBBU1234567", "documentPatch.containers[1].containerNumber"),
        draft("type_0", "40HR", "documentPatch.containers[0].typeDescription"),
    )
    normalized = normalize_compact_equipment_locality(raw=raw, drafts=drafts, source_target=target)
    compact = next(row for row in normalized if row.source_text == "40HR")
    assert compact.target_paths == ("documentPatch.containers[0].typeDescription",)
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_expanded_equipment_receipts_are_not_assigned_by_nearest_number() -> None:
    raw = "--- PAGE 1 ---\nAAAU1234567\nBBBU1234567\n1X40HIGH CUBE\n1X40HIGH CUBE\n"
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAU1234567", "typeDescription": "1X40HIGH CUBE"},
                {"containerNumber": "BBBU1234567", "typeDescription": "1X40HIGH CUBE"},
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def draft(
        *, draft_id: str, text: str, occurrence: int, target_path: str, group_key: str
    ) -> SpanDraft:
        start = raw.index(text) if occurrence == 0 else raw.index(text, raw.index(text) + 1)
        return SpanDraft(
            draft_id=draft_id,
            logical_key=f"agent:{draft_id}",
            render_mode="deterministic_derived",
            value_kind="equipment",
            group_kind="equipment",
            group_key=group_key,
            target_paths=(target_path,),
            derivation="equipment_receipt",
            dependency_paths=(target_path,),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="derived_operational_fact",
            render_policy="derived_surface",
            rationale="Ordered equipment receipt fixture.",
        )

    drafts = (
        replace(
            draft(
                draft_id="number_0",
                text="AAAU1234567",
                occurrence=0,
                target_path="documentPatch.containers[0].containerNumber",
                group_key="container:0",
            ),
            render_mode="target_binding",
            derivation=None,
            dependency_paths=(),
        ),
        replace(
            draft(
                draft_id="number_1",
                text="BBBU1234567",
                occurrence=0,
                target_path="documentPatch.containers[1].containerNumber",
                group_key="container:1",
            ),
            render_mode="target_binding",
            derivation=None,
            dependency_paths=(),
        ),
        draft(
            draft_id="receipt_0",
            text="1X40HIGH CUBE",
            occurrence=0,
            target_path="documentPatch.containers[0].typeDescription",
            group_key="container:0",
        ),
        draft(
            draft_id="receipt_1",
            text="1X40HIGH CUBE",
            occurrence=1,
            target_path="documentPatch.containers[1].typeDescription",
            group_key="container:1",
        ),
    )
    validate_compact_equipment_locality(raw=raw, drafts=drafts, source_target=target)


def test_anchor_value_kinds_distinguish_package_and_container_categories() -> None:
    body = "CARTON\n20 DRY 8'6\n"
    raw = "--- PAGE 1 ---\n" + body

    def row(value: str, path: str, anchor_id: str):
        start = body.index(value)
        return {
            "anchor_id": anchor_id,
            "document_id": "doc_categories",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(value),
            "raw_value": value,
            "target_value": value,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    drafts = anchor_drafts(
        raw=raw,
        document_id="doc_categories",
        anchors=(
            row("CARTON", "documentPatch.cargoPackages[0].typeCategory", "package"),
            row("20 DRY 8'6", "documentPatch.containers[0].typeDescription", "container"),
        ),
    )
    by_path = {draft.target_paths[0]: draft for draft in drafts}
    assert by_path["documentPatch.cargoPackages[0].typeCategory"].value_kind == "package"
    assert by_path["documentPatch.containers[0].typeDescription"].value_kind == "equipment"


def test_overlap_reconciliation_prefers_exact_fmc_carrier_owner() -> None:
    raw = "--- PAGE 1 ---\nFMC# 024085NF\n"
    start = raw.index("024085NF")
    carrier = SpanDraft(
        draft_id="carrier_fmc",
        logical_key="agent:carrier:fmc",
        render_mode="carrier_static",
        value_kind="identifier",
        group_kind="carrier",
        group_key="carrier:principal",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len("024085NF"),
        source_text="024085NF",
        evidence_origin="host_verified_agent_proposal",
        render_policy="opaque_identifier",
        rationale="Public carrier FMC identifier.",
    )
    auxiliary = replace(
        carrier,
        draft_id="shipment_fmc",
        logical_key="agent:shipment:fmc",
        render_mode="deterministic_auxiliary",
        group_kind="document",
        group_key="document",
        rationale="Conflicting duplicate proposal.",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw, drafts=(carrier, auxiliary), source_target={"documentPatch": {}}
    )

    assert reconciled == (carrier,)


def _test_draft(
    raw: str,
    text: str,
    logical_key: str,
    render_mode: str,
    target_paths: tuple[str, ...] = (),
    dependency_paths: tuple[str, ...] = (),
    group_key: str = "fixture",
    occurrence: int = 0,
) -> SpanDraft:
    starts = [match.start() for match in re.finditer(re.escape(text), raw)]
    start = starts[occurrence]
    return SpanDraft(
        draft_id=f"fixture_{start}_{len(text)}_{logical_key}",
        logical_key=logical_key,
        render_mode=render_mode,
        value_kind="equipment",
        group_kind="equipment",
        group_key=group_key,
        target_paths=target_paths,
        derivation="equipment_receipt" if render_mode == "deterministic_derived" else None,
        dependency_paths=dependency_paths,
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        evidence_origin="host_verified_agent_proposal",
        render_policy=(
            "derived_surface" if render_mode == "deterministic_derived" else "opaque_identifier"
        ),
        rationale="Test fixture.",
    )


def test_derived_surface_subsumes_contained_target_dependency() -> None:
    raw = "--- PAGE 1 ---\n1 X 20' STD CONTAINER\n"
    path = "documentPatch.containers[0].typeDescription"
    outer = _test_draft(
        raw,
        "1 X 20' STD CONTAINER",
        "agent:container:0:receipt",
        "deterministic_derived",
        ("documentPatch.containers[0]",),
        (path,),
        "container:0",
    )
    inner = _test_draft(raw, "20' STD CONTAINER", "anchor:" + path, "target_binding", (path,))
    inner = replace(
        inner, draft_id="anchor_binding_0001", evidence_origin="accepted_label_evidence"
    )
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": "20' STD CONTAINER"}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    reconciled = reconcile_draft_overlaps(raw=raw, drafts=(outer, inner), source_target=target)

    assert reconciled == (outer,)
    overridden = apply_anchor_overrides(
        raw=raw,
        source_target=target,
        anchors=(inner,),
        proposed=(outer,),
        overrides=(
            AnchorOverride(
                anchor_binding_id="anchor_binding_0001",
                rationale="The complete derived receipt owns the type dependency.",
            ),
        ),
    )
    assert overridden == (outer,)


def test_compact_equipment_auxiliary_joins_same_scope_target() -> None:
    raw = "--- PAGE 1 ---\n1X40'HQ CONTAINER\nCARGO\n40HQ\n"
    path = "documentPatch.containers[0].typeDescription"
    target_owner = _test_draft(
        raw,
        "1X40'HQ CONTAINER",
        "anchor:" + path,
        "target_binding",
        (path,),
        group_key="container:0",
    )
    auxiliary = _test_draft(
        raw, "40HQ", "agent:equipment_token", "deterministic_auxiliary", group_key="container:0"
    )
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": "1X40'HQ CONTAINER"}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_compact_equipment_locality(
        raw=raw, drafts=(target_owner, auxiliary), source_target=target
    )

    assert {row.logical_key for row in normalized} == {"anchor:" + path}
    assert {row.target_paths for row in normalized} == {(path,)}
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)
