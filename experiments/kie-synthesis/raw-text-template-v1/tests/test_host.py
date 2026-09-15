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
    _drop_exact_source_only_duplicates,
    _repair_uniquely_bracketed_target_paths,
    _resolve_exact_target_source_only_conflicts,
    anchor_drafts,
    anchor_summary,
    annotated_source,
    apply_anchor_overrides,
    apply_critic_patch,
    binding_realization,
    certify_template,
    inventory_binding_id,
    line_range_for_chars,
    line_spans,
    merge_drafts,
    normalize_auxiliary_identifier_boundaries,
    normalize_cargo_description_linked_package_prefixes,
    normalize_compact_equipment_locality,
    normalize_competing_auxiliary_outliers,
    normalize_country_code_locality,
    normalize_decimal_measurement_boundaries,
    normalize_deterministic_draft_semantics,
    normalize_labeled_shipper_and_receipt_locality,
    normalize_package_type_row_locality,
    normalize_pinned_carrier_assessment,
    normalize_relational_anchor_locality,
    normalize_segmented_party_address_targets,
    normalize_source_boundaries,
    normalize_structured_row_locality,
    normalize_target_cobindings,
    reconcile_draft_overlaps,
    refuted_semantic_only_target_paths,
    required_target_cobindings,
    resolve_agent_proposal_inventory,
    resolve_agent_proposals,
    risk_candidates,
    target_path_relationship,
    uncovered_risks,
    validate_agent_proposal_paths,
    validate_binding_realizations,
    validate_carrier_assessment,
    validate_compact_equipment_locality,
    validate_mutable_token_boundaries,
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


def test_selected_operational_and_type_values_are_high_recall_risks() -> None:
    raw = "--- PAGE 1 ---\nTYPE: VAT NUMBER\nSHIPPED ON BOARD VESSEL ONE 123A AT PORTX\n"

    risks = risk_candidates(raw, ())

    assert ("selected_text", "VAT NUMBER") in {(risk.kind, risk.source_text) for risk in risks}
    assert ("selected_text", "SHIPPED ON BOARD") in {
        (risk.kind, risk.source_text) for risk in risks
    }
    selected = tuple(risk for risk in risks if risk.kind == "selected_text")
    owned = tuple(
        SpanDraft(
            draft_id=f"selected_{index}",
            logical_key=f"agent:selected:{index}",
            render_mode="deterministic_auxiliary",
            value_kind="operational_text",
            group_kind="transport",
            group_key="transport:selected",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=len(raw[: risk.byte_start].encode("utf-8")),
            char_end=len(raw[: risk.byte_end].encode("utf-8")),
            source_text=risk.source_text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Fixture selected text ownership.",
        )
        for index, risk in enumerate(selected)
    )
    assert all(risk.kind != "selected_text" for risk in risk_candidates(raw, owned))


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
    body = (
        "Acme Ocean Lines Ltd.\n"
        "PORT OF LOADING: ALEXANDRIA\n"
        "PORT OF DISCHARGE: ALEXANDRIA\n"
        "UNRELATED COPY: ALEXANDRIA\n"
    )
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
            additional_bindings=(route_proposal(target_paths[0], "L00005"),),
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


def test_critic_split_can_preserve_uncited_exact_spans_from_removed_binding() -> None:
    value = "SAME"
    raw = "--- PAGE 1 ---\nFIRST: SAME\nSECOND: SAME\n"
    paths = (
        "documentPatch.route.portOfLoading.name",
        "documentPatch.route.portOfDischarge.name",
    )
    target = {
        "documentPatch": {
            "route": {
                "portOfLoading": {"name": value},
                "portOfDischarge": {"name": value},
            }
        }
    }
    lines = line_spans(raw)
    existing = tuple(
        SpanDraft(
            draft_id=f"aggregated_{index}",
            logical_key="agent:incorrectly_aggregated_route_roles",
            render_mode="target_binding",
            value_kind="location",
            group_kind="route",
            group_key="route:aggregated",
            target_paths=paths,
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=raw.index(value, lines[index + 1].char_start),
            char_end=raw.index(value, lines[index + 1].char_start) + len(value),
            source_text=value,
            evidence_origin="agent_proposed",
            render_policy="natural_text",
            rationale="Fixture with independently mutable route facts aggregated together.",
        )
        for index in range(2)
    )
    finding = CriticFinding.model_validate(
        {
            "finding_kind": "topology_or_grouping_error",
            "line_ids": ("L00003",),
            "evidence": "The second route role proves the aggregate must be split.",
            "explanation": "Split both existing physical occurrences into their exact roles.",
        }
    )

    def replacement(path: str, line_id: str) -> AgentBindingProposal:
        return AgentBindingProposal.model_validate(
            {
                "logical_key": "anchor:" + path,
                "render_mode": "target_binding",
                "value_kind": "location",
                "group_kind": "route",
                "group_key": path,
                "target_paths": (path,),
                "occurrences": (
                    {
                        "line_start": line_id,
                        "line_end": line_id,
                        "source_text": value,
                        "occurrence_index": 0,
                    },
                ),
                "rationale": "Preserve this existing span under its independent route owner.",
            }
        )

    revised = apply_critic_patch(
        raw=raw,
        drafts=existing,
        findings=(finding,),
        remove_inventory_binding_ids=(inventory_binding_id(existing[0].logical_key),),
        additional_bindings=(replacement(paths[0], "L00002"), replacement(paths[1], "L00003")),
        source_target=target,
    )

    assert {row.target_paths for row in revised} == {(paths[0],), (paths[1],)}
    carried = next(row for row in revised if row.target_paths == (paths[0],))
    assert "exact physical occurrence" in carried.rationale


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


def test_partial_proposal_inventory_preserves_valid_occurrences_and_reports_invalid_ones() -> None:
    raw = "--- PAGE 1 ---\nVALID VALUE\nOTHER ROW\n"
    proposal = _proposal("VALID VALUE", "L00002", key="mixed_occurrences").model_copy(
        update={
            "occurrences": (
                _proposal("VALID VALUE", "L00002", key="valid").occurrences[0],
                _proposal("MISSING VALUE", "L00003", key="invalid").occurrences[0],
            )
        }
    )

    resolved, rejected = resolve_agent_proposal_inventory(raw=raw, proposals=(proposal,))

    assert len(resolved) == 1
    assert resolved[0].source_text == "VALID VALUE"
    assert rejected == ("mixed_occurrences L00003-L00003",)


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


def test_original_bill_count_value_is_a_mandatory_selected_text_risk() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "Number of original Bills of Lading\n"
        "E / Express B/L\n"
        "SEAWAYBILL\n"
        "non negotiable\n"
    )

    risks = risk_candidates(raw, ())

    selected = tuple(
        row for row in risks if row.kind == "selected_text" and row.source_text == "E / Express B/L"
    )
    assert len(selected) == 1
    assert selected[0].line_id == "L00003"


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


def test_anchor_override_of_one_repeated_occurrence_retains_existing_target_ownership() -> None:
    body = "FIRST REF-123\nSECOND REF-123\n"
    raw = "--- PAGE 1 ---\n" + body
    target_path = "documentPatch.billOfLadingNumber"
    first_start = body.index("REF-123")
    second_start = body.index("REF-123", first_start + 1)
    anchor_rows = tuple(
        {
            "anchor_id": anchor_id,
            "document_id": "doc_repeated_anchor",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len("REF-123"),
            "raw_value": "REF-123",
            "target_value": "REF-123",
            "relation_target_path": target_path,
            "role_path": target_path,
            "surface_family": "identifier",
        }
        for anchor_id, start in (("anchor_first", first_start), ("anchor_second", second_start))
    )
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_repeated_anchor",
        anchors=anchor_rows,
    )
    first = min(anchors, key=lambda row: row.char_start)
    second = max(anchors, key=lambda row: row.char_start)

    revised = apply_anchor_overrides(
        raw=raw,
        source_target={"documentPatch": {"billOfLadingNumber": "REF-123"}},
        anchors=anchors,
        proposed=(),
        overrides=(
            AnchorOverride.model_validate(
                {
                    "anchor_binding_id": first.draft_id,
                    "rationale": "The first duplicate is not the role-specific occurrence.",
                }
            ),
        ),
    )

    assert revised == (second,)


def test_anchor_override_recovers_only_selected_freight_payment_token() -> None:
    body = "FREIGHT PREPAID ON BOARD\n"
    raw = "--- PAGE 1 ---\n" + body
    path = "documentPatch.freight.paymentArrangement"
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_freight_override",
        anchors=(
            {
                "anchor_id": "freight_payment",
                "document_id": "doc_freight_override",
                "patchable": True,
                "page_number": 1,
                "page_start": body.index("FREIGHT PREPAID"),
                "page_end": body.index("FREIGHT PREPAID") + len("FREIGHT PREPAID"),
                "raw_value": "FREIGHT PREPAID",
                "target_value": "prepaid",
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "text",
            },
        ),
    )
    revised = apply_anchor_overrides(
        raw=raw,
        source_target={"documentPatch": {"freight": {"paymentArrangement": "prepaid"}}},
        anchors=anchors,
        proposed=(),
        overrides=(
            AnchorOverride.model_validate(
                {
                    "anchor_binding_id": anchors[0].draft_id,
                    "rationale": "FREIGHT is a stable caption prefix.",
                }
            ),
        ),
    )

    assert len(revised) == 1
    assert revised[0].target_paths == (path,)
    assert revised[0].source_text == "PREPAID"
    assert revised[0].char_start == raw.index("PREPAID")


def test_anchor_override_does_not_choose_from_freight_option_vocabulary() -> None:
    body = "FREIGHT PREPAID COLLECT\n"
    raw = "--- PAGE 1 ---\n" + body
    path = "documentPatch.freight.paymentArrangement"
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_freight_options",
        anchors=(
            {
                "anchor_id": "freight_payment",
                "document_id": "doc_freight_options",
                "patchable": True,
                "page_number": 1,
                "page_start": 0,
                "page_end": len(body.rstrip()),
                "raw_value": body.rstrip(),
                "target_value": "prepaid",
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "text",
            },
        ),
    )

    with pytest.raises(ValueError, match="lacks replacement target ownership"):
        apply_anchor_overrides(
            raw=raw,
            source_target={"documentPatch": {"freight": {"paymentArrangement": "prepaid"}}},
            anchors=anchors,
            proposed=(),
            overrides=(
                AnchorOverride.model_validate(
                    {
                        "anchor_binding_id": anchors[0].draft_id,
                        "rationale": "This line is option vocabulary.",
                    }
                ),
            ),
        )


def test_exact_anchor_repeat_inherits_complete_retained_binding_contract() -> None:
    raw, anchor_rows, target, _feature = _fixture()
    raw += "SIGNED FOR Acme Ocean Lines Ltd.\n"
    anchors = anchor_drafts(raw=raw, document_id="doc_fixture", anchors=anchor_rows)
    carrier = next(
        row for row in anchors if row.target_paths == ("documentPatch.parties.carrier.name",)
    )
    proposal = AgentBindingProposal.model_validate(
        {
            "logical_key": carrier.logical_key,
            "render_mode": "target_binding",
            "value_kind": "organization",
            "group_kind": "party",
            "group_key": "party:carrier:0",
            "target_paths": carrier.target_paths,
            "derivation": None,
            "dependency_paths": (),
            "dependency_bindings": (),
            "occurrences": (
                {
                    "line_start": "L00007",
                    "line_end": "L00007",
                    "source_text": "Acme Ocean Lines Ltd.",
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Attach the signature copy to the accepted carrier binding.",
        }
    )
    proposed = resolve_agent_proposals(raw=raw, proposals=(proposal,))

    merged = apply_anchor_overrides(
        raw=raw,
        source_target=target,
        anchors=anchors,
        proposed=proposed,
        overrides=(),
    )

    carrier_occurrences = tuple(row for row in merged if row.logical_key == carrier.logical_key)
    assert len(carrier_occurrences) == 2
    for occurrence in carrier_occurrences:
        assert occurrence.render_mode == carrier.render_mode == "carrier_static"
        assert occurrence.value_kind == carrier.value_kind
        assert occurrence.group_kind == carrier.group_kind
        assert occurrence.group_key == carrier.group_key
        assert occurrence.target_paths == carrier.target_paths
        assert occurrence.derivation == carrier.derivation
        assert occurrence.dependency_paths == carrier.dependency_paths
        assert occurrence.dependency_bindings == carrier.dependency_bindings
        assert occurrence.render_policy == carrier.render_policy


def test_anchor_replacement_evidence_excludes_alphanumeric_substrings() -> None:
    body = "EGYPTIAN IMPORTER TAX ID\n"
    raw = "--- PAGE 1 ---\n" + body
    path = "documentPatch.parties.notifyParties[0].country"
    target = {"documentPatch": {"parties": {"notifyParties": [{"country": "EGYPT"}]}}}
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_embedded_country",
        anchors=(
            {
                "anchor_id": "embedded_country",
                "document_id": "doc_embedded_country",
                "patchable": True,
                "page_number": 1,
                "page_start": body.index("EGYPT"),
                "page_end": body.index("EGYPT") + len("EGYPT"),
                "raw_value": "EGYPT",
                "target_value": "EGYPT",
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "text",
            },
        ),
    )

    with pytest.raises(ValueError, match="unowned exact candidates: none"):
        apply_anchor_overrides(
            raw=raw,
            source_target=target,
            anchors=anchors,
            proposed=(),
            overrides=(
                AnchorOverride.model_validate(
                    {
                        "anchor_binding_id": anchors[0].draft_id,
                        "rationale": "Remove the embedded substring anchor.",
                    }
                ),
            ),
        )


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


def test_relational_anchor_locality_separates_repeated_cargo_rows() -> None:
    document_id = "doc_relational_rows"
    body = (
        "ABCU1234567 / SEAL-A 40HQ 96 CARTONS\n"
        "TOBACCO\n"
        "MATERIAL A11111\n"
        "BATCH A22222\n"
        "GRADE A33333\n"
        "COMMON 555555\n"
        "EFGU7654321 / SEAL-B 40HQ 96 CARTONS\n"
        "TOBACCO\n"
        "MATERIAL B11111\n"
        "BATCH B22222\n"
        "GRADE B33333\n"
        "COMMON 555555\n"
    )
    raw = "--- PAGE 1 ---\n" + body
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "ABCU1234567", "typeDescription": "40HQ"},
                {"containerNumber": "EFGU7654321", "typeDescription": "40HQ"},
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TOBACCO",
                    "additionalInformation": [
                        "MATERIAL A11111",
                        "BATCH A22222",
                        "GRADE A33333",
                        "COMMON 555555",
                    ],
                },
                {
                    "groupId": "g2",
                    "description": "TOBACCO",
                    "additionalInformation": [
                        "MATERIAL B11111",
                        "BATCH B22222",
                        "GRADE B33333",
                        "COMMON 555555",
                    ],
                },
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                },
                {
                    "groupId": "g2",
                    "packageId": "p2",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                },
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "one_to_one_package_allocations",
                    "allocations": [
                        {
                            "containerNumber": "ABCU1234567",
                            "packageId": "p1",
                            "packageQuantity": 96,
                        }
                    ],
                },
                {
                    "groupId": "g2",
                    "packageIds": ["p2"],
                    "coverage": "one_to_one_package_allocations",
                    "allocations": [
                        {
                            "containerNumber": "EFGU7654321",
                            "packageId": "p2",
                            "packageQuantity": 96,
                        }
                    ],
                },
            ],
        }
    }

    def row(
        anchor_id: str,
        surface: str,
        path: str,
        occurrence: int = 0,
        *,
        patchable: bool = True,
    ) -> dict[str, object]:
        starts = [match.start() for match in re.finditer(re.escape(surface), body)]
        start = starts[occurrence]
        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "patchable": patchable,
            "page_number": 1 if patchable else None,
            "page_start": start if patchable else None,
            "page_end": start + len(surface) if patchable else None,
            "raw_value": surface,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    anchors = [
        row("container_0", "ABCU1234567", "documentPatch.containers[0].containerNumber"),
        row("container_1", "EFGU7654321", "documentPatch.containers[1].containerNumber"),
        row("container_type_0", "40HQ", "documentPatch.containers[0].typeDescription"),
        # The preparation baseline also attaches this repeated container value to row zero.
        row("container_type_1", "40HQ", "documentPatch.containers[1].typeDescription"),
        row(
            "allocation_container_0",
            "ABCU1234567",
            "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
        ),
        row(
            "allocation_container_1",
            "EFGU7654321",
            "documentPatch.cargoAllocationGroups[1].allocations[0].containerNumber",
        ),
    ]
    for group_index, suffix in enumerate(("A", "B")):
        anchors.extend(
            (
                row(
                    f"material_{group_index}",
                    f"MATERIAL {suffix}11111",
                    f"documentPatch.cargoGroups[{group_index}].additionalInformation[0]",
                ),
                row(
                    f"batch_{group_index}",
                    f"BATCH {suffix}22222",
                    f"documentPatch.cargoGroups[{group_index}].additionalInformation[1]",
                ),
                row(
                    f"grade_{group_index}",
                    f"GRADE {suffix}33333",
                    f"documentPatch.cargoGroups[{group_index}].additionalInformation[2]",
                ),
                row(
                    f"common_{group_index}",
                    "COMMON 555555",
                    f"documentPatch.cargoGroups[{group_index}].additionalInformation[3]",
                    patchable=False,
                ),
                # The preparation baseline incorrectly attaches both rows to occurrence zero.
                row(
                    f"description_{group_index}",
                    "TOBACCO",
                    f"documentPatch.cargoGroups[{group_index}].description",
                ),
                row(
                    f"package_quantity_{group_index}",
                    "96",
                    f"documentPatch.cargoPackages[{group_index}].quantity",
                ),
                row(
                    f"allocation_quantity_{group_index}",
                    "96",
                    "documentPatch.cargoAllocationGroups"
                    f"[{group_index}].allocations[0].packageQuantity",
                ),
                row(
                    f"package_type_{group_index}",
                    "CARTONS",
                    f"documentPatch.cargoPackages[{group_index}].typeCategory",
                ),
            )
        )

    normalized, report = normalize_relational_anchor_locality(
        raw=raw,
        document_id=document_id,
        anchors=anchors,
        source_target=target,
    )

    assert report.orientation == "forward"
    assert report.forward_evidence == 6
    assert set(report.relocated_anchor_ids) == {
        "container_type_1",
        "description_1",
        "package_quantity_1",
        "allocation_quantity_1",
        "package_type_1",
    }
    assert report.suppressed_anchor_ids == ()
    assert len(report.recovered_topology_anchor_ids) == 2
    relocated = {row["anchor_id"]: row for row in normalized}
    assert relocated["description_1"]["page_start"] == body.rindex("TOBACCO")
    assert relocated["container_type_1"]["page_start"] == body.rindex("40HQ")
    assert relocated["package_quantity_1"]["page_start"] == body.rindex("96")
    assert relocated["package_type_1"]["page_start"] == body.rindex("CARTONS")

    drafts = normalize_target_cobindings(
        drafts=anchor_drafts(raw=raw, document_id=document_id, anchors=normalized),
        source_target=target,
    )
    assert not any(
        "cargoGroups[0]" in " ".join(draft.target_paths)
        and "cargoGroups[1]" in " ".join(draft.target_paths)
        for draft in drafts
    )
    second_description = next(
        draft
        for draft in drafts
        if draft.target_paths == ("documentPatch.cargoGroups[1].description",)
    )
    assert raw[: second_description.char_start].count("\n") + 1 == 9


def test_relational_cargo_blocks_recover_cross_page_column_continuation() -> None:
    document_id = "doc_relational_cross_page_blocks"
    page_one = (
        "ABCU1234567\n"
        "ALPHA TOBACCO\n"
        "ALPHA TOBACCO\n"
        "OMEGA LEAF\n"
        "MATERIAL A11111\n"
        "BATCH A22222\n"
        "CUSTOMS CDE 999999\n"
        "GRADE A33333\n"
        "\n"
        "EFGU7654321\n"
        "TO BE CONTINUED ON ATTACHED LIST\n"
    )
    page_two = (
        "IJKU1112223\n"
        "BETA TOBACCO\n"
        "MATERIAL B11111\n"
        "BATCH B22222\n"
        "CUSTOMS CDE 999999\n"
        "GRADE B33333\n"
        "\n"
        "GAMMA TOBACCO\n"
        "MATERIAL C11111\n"
        "BATCH C22222\n"
        "CUSTOMS CDE 999999\n"
        "GRADE C33333\n"
    )
    raw = f"--- PAGE 1 ---\n{page_one}\n--- PAGE 2 ---\n{page_two}"
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "ABCU1234567"},
                {"containerNumber": "EFGU7654321"},
                {"containerNumber": "IJKU1112223"},
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "ALPHA TOBACCO OMEGA LEAF",
                    "additionalInformation": [
                        "MATERIAL A11111",
                        "BATCH A22222",
                        "GRADE A33333",
                    ],
                    "hsCodes": ["999999"],
                },
                {
                    "groupId": "g2",
                    "description": "BETA TOBACCO",
                    "additionalInformation": [
                        "MATERIAL B11111",
                        "BATCH B22222",
                        "GRADE B33333",
                    ],
                    "hsCodes": ["999999"],
                },
                {
                    "groupId": "g3",
                    "description": "GAMMA TOBACCO",
                    "additionalInformation": [
                        "MATERIAL C11111",
                        "BATCH C22222",
                        "GRADE C33333",
                    ],
                    "hsCodes": ["999999"],
                },
            ],
            "cargoPackages": [],
            "cargoAllocationGroups": [
                {
                    "groupId": f"g{index + 1}",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": container_number}],
                }
                for index, container_number in enumerate(
                    ("ABCU1234567", "EFGU7654321", "IJKU1112223")
                )
            ],
        }
    }

    def row(
        anchor_id: str,
        surface: str,
        path: str,
        *,
        page_number: int,
        occurrence: int = 0,
    ) -> dict[str, object]:
        page = page_one if page_number == 1 else page_two
        starts = [match.start() for match in re.finditer(re.escape(surface), page)]
        start = starts[occurrence]
        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "patchable": True,
            "page_number": page_number,
            "page_start": start,
            "page_end": start + len(surface),
            "raw_value": surface,
            "target_value": surface,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    anchors: list[dict[str, object]] = []
    for index, (container_number, page_number) in enumerate(
        (("ABCU1234567", 1), ("EFGU7654321", 1), ("IJKU1112223", 2))
    ):
        anchors.extend(
            (
                row(
                    f"container_{index}",
                    container_number,
                    f"documentPatch.containers[{index}].containerNumber",
                    page_number=page_number,
                ),
                row(
                    f"allocation_container_{index}",
                    container_number,
                    f"documentPatch.cargoAllocationGroups[{index}].allocations[0].containerNumber",
                    page_number=page_number,
                ),
            )
        )
    for index, (prefix, page_number, hs_occurrence) in enumerate(
        (("A", 1, 0), ("B", 2, 0), ("C", 2, 1))
    ):
        description = (
            "ALPHA TOBACCO\nOMEGA LEAF",
            "BETA TOBACCO",
            "GAMMA TOBACCO",
        )[index]
        anchors.extend(
            (
                row(
                    f"description_{index}",
                    description,
                    f"documentPatch.cargoGroups[{index}].description",
                    page_number=page_number,
                ),
                row(
                    f"material_{index}",
                    f"MATERIAL {prefix}11111",
                    f"documentPatch.cargoGroups[{index}].additionalInformation[0]",
                    page_number=page_number,
                ),
                row(
                    f"batch_{index}",
                    f"BATCH {prefix}22222",
                    f"documentPatch.cargoGroups[{index}].additionalInformation[1]",
                    page_number=page_number,
                ),
                row(
                    f"grade_{index}",
                    f"GRADE {prefix}33333",
                    f"documentPatch.cargoGroups[{index}].additionalInformation[2]",
                    page_number=page_number,
                ),
                row(
                    f"hs_{index}",
                    "999999",
                    f"documentPatch.cargoGroups[{index}].hsCodes[0]",
                    page_number=page_number,
                    occurrence=hs_occurrence,
                ),
            )
        )

    normalized, report = normalize_relational_anchor_locality(
        raw=raw,
        document_id=document_id,
        anchors=anchors,
        source_target=target,
    )
    drafts = normalize_target_cobindings(
        drafts=anchor_drafts(raw=raw, document_id=document_id, anchors=normalized),
        source_target=target,
    )
    lines = line_spans(raw)

    assert report.orientation == "forward"
    assert report.cargo_block_assignments == (
        "cargoGroups[0]:L00001-L00009",
        "cargoGroups[1]:L00014-L00020",
        "cargoGroups[2]:L00022-L00026",
    )
    first_description = tuple(
        draft
        for draft in drafts
        if draft.target_paths == ("documentPatch.cargoGroups[0].description",)
    )
    assert tuple(
        line_range_for_chars(lines, draft.char_start, draft.char_end) for draft in first_description
    ) == (("L00003", "L00003"), ("L00004", "L00005"))
    for cargo_index, expected_line in ((1, "L00019"), (2, "L00025")):
        hs_drafts = tuple(
            draft
            for draft in drafts
            if draft.target_paths == (f"documentPatch.cargoGroups[{cargo_index}].hsCodes[0]",)
        )
        assert len(hs_drafts) == 1
        assert line_range_for_chars(
            lines,
            hs_drafts[0].char_start,
            hs_drafts[0].char_end,
        ) == (expected_line, expected_line)
    validate_target_binding_relationships(drafts=drafts, source_target=target)


def test_relational_anchor_locality_suppresses_independent_equal_facts() -> None:
    document_id = "doc_relational_ambiguity"
    body = "ABCU1234567\nDESCRIPTION UNIQUE\nMATERIAL UNIQUE\nBATCH UNIQUE\nCODE\n"
    raw = "--- PAGE 1 ---\n" + body
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "ABCU1234567"}],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "DESCRIPTION UNIQUE",
                    "additionalInformation": ["MATERIAL UNIQUE", "BATCH UNIQUE", "CODE", "CODE"],
                }
            ],
            "cargoPackages": [],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": "ABCU1234567"}],
                }
            ],
        }
    }

    def row(anchor_id: str, surface: str, path: str) -> dict[str, object]:
        start = body.index(surface)
        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(surface),
            "raw_value": surface,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    anchors = (
        row("container", "ABCU1234567", "documentPatch.containers[0].containerNumber"),
        row("description", "DESCRIPTION UNIQUE", "documentPatch.cargoGroups[0].description"),
        row("material", "MATERIAL UNIQUE", "documentPatch.cargoGroups[0].additionalInformation[0]"),
        row("batch", "BATCH UNIQUE", "documentPatch.cargoGroups[0].additionalInformation[1]"),
        row("code_0", "CODE", "documentPatch.cargoGroups[0].additionalInformation[2]"),
        row("code_1", "CODE", "documentPatch.cargoGroups[0].additionalInformation[3]"),
    )

    normalized, report = normalize_relational_anchor_locality(
        raw=raw,
        document_id=document_id,
        anchors=anchors,
        source_target=target,
    )

    assert report.orientation == "forward"
    assert set(report.suppressed_anchor_ids) == {"code_0", "code_1"}
    normalized_by_id = {row["anchor_id"]: row for row in normalized}
    assert normalized_by_id["code_0"]["patchable"] is False
    assert normalized_by_id["code_0"]["page_start"] is None
    assert normalized_by_id["code_1"]["patchable"] is False


def test_relational_anchor_locality_expands_all_proven_same_group_repeats() -> None:
    document_id = "doc_relational_repeats"
    body = (
        "ABCU1234567 96 CARTONS\n"
        "TOBACCO\n"
        "MATERIAL FIXED\n"
        "ORIGIN USA\n"
        "BATCH UNIQUE\n"
        "GRADE UNIQUE\n"
        "EFGU7654321 96 CARTONS\n"
        "TOBACCO\n"
        "MATERIAL FIXED\n"
    )
    raw = "--- PAGE 1 ---\n" + body
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "ABCU1234567"},
                {"containerNumber": "EFGU7654321"},
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TOBACCO",
                    "additionalInformation": [
                        "MATERIAL FIXED",
                        "ORIGIN USA",
                        "BATCH UNIQUE",
                        "GRADE UNIQUE",
                    ],
                }
            ],
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
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "one_to_one_package_allocations",
                    "allocations": [
                        {
                            "containerNumber": "ABCU1234567",
                            "packageId": "p1",
                            "packageQuantity": 96,
                        },
                        {
                            "containerNumber": "EFGU7654321",
                            "packageId": "p1",
                            "packageQuantity": 96,
                        },
                    ],
                }
            ],
        }
    }

    def row(anchor_id: str, surface: str, path: str, occurrence: int = 0) -> dict[str, object]:
        starts = [match.start() for match in re.finditer(re.escape(surface), body)]
        start = starts[occurrence]
        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(surface),
            "raw_value": surface,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    anchors = [
        row("container_0", "ABCU1234567", "documentPatch.containers[0].containerNumber"),
        row("container_1", "EFGU7654321", "documentPatch.containers[1].containerNumber"),
        row(
            "allocation_container_0",
            "ABCU1234567",
            "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
        ),
        row(
            "allocation_container_1",
            "EFGU7654321",
            "documentPatch.cargoAllocationGroups[0].allocations[1].containerNumber",
        ),
        row("description", "TOBACCO", "documentPatch.cargoGroups[0].description"),
        row(
            "material",
            "MATERIAL FIXED",
            "documentPatch.cargoGroups[0].additionalInformation[0]",
        ),
        row(
            "origin",
            "ORIGIN USA",
            "documentPatch.cargoGroups[0].additionalInformation[1]",
        ),
        row(
            "batch",
            "BATCH UNIQUE",
            "documentPatch.cargoGroups[0].additionalInformation[2]",
        ),
        row(
            "grade",
            "GRADE UNIQUE",
            "documentPatch.cargoGroups[0].additionalInformation[3]",
        ),
        row("package_quantity", "96", "documentPatch.cargoPackages[0].quantity"),
        row(
            "allocation_quantity_0",
            "96",
            "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
        ),
        row(
            "allocation_quantity_1",
            "96",
            "documentPatch.cargoAllocationGroups[0].allocations[1].packageQuantity",
        ),
        row("package_type", "CARTONS", "documentPatch.cargoPackages[0].typeCategory"),
    ]

    normalized, report = normalize_relational_anchor_locality(
        raw=raw,
        document_id=document_id,
        anchors=anchors,
        source_target=target,
    )

    assert report.orientation is None
    assert len(report.expanded_anchor_ids) == 2
    assert report.relocated_anchor_ids == ("allocation_quantity_1",)
    assert report.suppressed_anchor_ids == ()
    drafts = normalize_target_cobindings(
        drafts=anchor_drafts(raw=raw, document_id=document_id, anchors=normalized),
        source_target=target,
    )
    description_occurrences = [
        draft
        for draft in drafts
        if draft.target_paths == ("documentPatch.cargoGroups[0].description",)
    ]
    assert len(description_occurrences) == 2
    assert [draft.source_text for draft in description_occurrences] == ["TOBACCO", "TOBACCO"]


def test_relational_anchor_locality_preserves_container_field_without_group_pivot() -> None:
    document_id = "doc_container_without_allocation"
    body = "ABCU1234567 40HQ\n"
    raw = "--- PAGE 1 ---\n" + body
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "ABCU1234567", "typeDescription": "40HQ"}],
            "cargoGroups": [],
            "cargoPackages": [],
            "cargoAllocationGroups": [],
        }
    }

    def row(anchor_id: str, surface: str, path: str) -> dict[str, object]:
        start = body.index(surface)
        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(surface),
            "raw_value": surface,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    anchors = (
        row("container", "ABCU1234567", "documentPatch.containers[0].containerNumber"),
        row("type", "40HQ", "documentPatch.containers[0].typeDescription"),
    )
    normalized, report = normalize_relational_anchor_locality(
        raw=raw,
        document_id=document_id,
        anchors=anchors,
        source_target=target,
    )

    assert normalized == anchors
    assert report.pivot_count == 0
    assert report.relocated_anchor_ids == ()
    assert report.expanded_anchor_ids == ()
    assert report.suppressed_anchor_ids == ()


def test_relational_anchor_locality_recovers_unique_columnar_target_projections() -> None:
    document_id = "doc_relational_columnar"
    body = "ABCU1234567\nTOBACCO\nMATERIAL\n111111\n111111\n222222\n"
    raw = "--- PAGE 1 ---\n" + body
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "ABCU1234567"}],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TOBACCO",
                    "additionalInformation": ["MATERIAL 111111", "MATERIAL 222222"],
                }
            ],
            "cargoPackages": [],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": "ABCU1234567"}],
                }
            ],
        }
    }

    def row(
        anchor_id: str,
        surface: str,
        path: str,
        *,
        patchable: bool = True,
    ) -> dict[str, object]:
        start = body.find(surface)
        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "patchable": patchable,
            "page_number": 1 if patchable else None,
            "page_start": start if patchable else None,
            "page_end": start + len(surface) if patchable else None,
            "raw_value": surface,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    anchors = (
        row("container", "ABCU1234567", "documentPatch.containers[0].containerNumber"),
        row(
            "allocation_container",
            "ABCU1234567",
            "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
        ),
        row("description", "TOBACCO", "documentPatch.cargoGroups[0].description"),
        row(
            "material_0",
            "MATERIAL\n111111",
            "documentPatch.cargoGroups[0].additionalInformation[0]",
        ),
        row(
            "material_1",
            "MATERIAL 222222",
            "documentPatch.cargoGroups[0].additionalInformation[1]",
            patchable=False,
        ),
    )

    normalized, report = normalize_relational_anchor_locality(
        raw=raw,
        document_id=document_id,
        anchors=anchors,
        source_target=target,
    )

    assert len(report.recovered_topology_anchor_ids) == 2
    recovered = {
        row["relation_target_path"]: row["raw_value"]
        for row in normalized
        if row["anchor_id"] in report.recovered_topology_anchor_ids
    }
    assert recovered == {
        "documentPatch.cargoGroups[0].additionalInformation[0]": "111111",
        "documentPatch.cargoGroups[0].additionalInformation[1]": "222222",
    }


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


def test_pinned_carrier_assessment_uses_exact_target_owner_not_agent_pointer() -> None:
    raw = "--- PAGE 1 ---\nMATERIAL 13672297\nORIENT OVERSEAS CONTAINER\nLINE, AS CARRIER\n"
    path = "documentPatch.parties.carrier.name"
    owner = replace(
        _test_draft(
            raw,
            "ORIENT OVERSEAS CONTAINER\nLINE",
            "anchor:" + path,
            "carrier_static",
            (path,),
        ),
        value_kind="organization",
        group_kind="carrier",
        group_key="party:carrier:0",
        render_policy="natural_text",
    )
    mistaken = CarrierAssessment.model_validate(
        {
            "canonical_name": "ORIENT OVERSEAS CONTAINER LINE",
            "aliases": ("OOCL",),
            "evidence_occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "13672297",
                    "occurrence_index": 0,
                },
            ),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "The provider copied the wrong evidence handle.",
        }
    )

    normalized = normalize_pinned_carrier_assessment(
        assessment=mistaken,
        expected="ORIENT OVERSEAS CONTAINER LINE",
        raw=raw,
        drafts=(owner,),
    )

    assert normalized.aliases == ()
    assert normalized.evidence_occurrences[0].source_text == ("ORIENT OVERSEAS CONTAINER\nLINE")
    assert normalized.evidence_occurrences[0].line_start == "L00003"
    assert normalized.evidence_occurrences[0].line_end == "L00004"
    validate_carrier_assessment(
        assessment=normalized,
        expected="ORIENT OVERSEAS CONTAINER LINE",
        raw=raw,
        anchor_drafts_value=(owner,),
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
    with pytest.raises(ValueError, match="not the same complete composite"):
        validate_target_binding_relationships(
            drafts=(
                replace(first, render_mode="agent_residual"),
                replace(
                    repeated,
                    render_mode="agent_residual",
                    source_text="different composite",
                ),
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


def test_adjacent_agent_residual_segments_rejoin_across_literal_layout_only() -> None:
    raw = "FREE ZONE\nATEF EL SADAT ST .\nSHEBIN EL KOM 32111-22 EGYPT"
    paths = (
        "documentPatch.parties.consignee.address",
        "documentPatch.parties.consignee.city",
    )
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "address": "FREE ZONE ATEF EL SADAT ST 32111-22",
                    "city": "SHEBIN EL KOM",
                }
            }
        }
    }
    first_text = "FREE ZONE\nATEF EL SADAT ST"
    second_text = "SHEBIN EL KOM 32111-22"
    first = replace(
        _realization_draft(
            target_paths=paths,
            value_kind="address",
            render_policy="natural_text",
            render_mode="agent_residual",
        ),
        char_start=0,
        char_end=len(first_text),
        source_text=first_text,
    )
    second_start = raw.index(second_text)
    second = replace(
        first,
        draft_id="fixture_second_segment",
        char_start=second_start,
        char_end=second_start + len(second_text),
        source_text=second_text,
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(first, second),
        source_target=target,
    )

    assert len(reconciled) == 1
    assert reconciled[0].source_text == raw[: second_start + len(second_text)]
    assert reconciled[0].draft_id.startswith("reconciled_residual_")
    validate_target_binding_relationships(drafts=reconciled, source_target=target)

    separated_raw = "FIRST\nCAPTION\nSECOND"
    separated_first = replace(
        first,
        char_start=0,
        char_end=5,
        source_text="FIRST",
    )
    separated_second = replace(
        second,
        char_start=14,
        char_end=20,
        source_text="SECOND",
    )
    separated = reconcile_draft_overlaps(
        raw=separated_raw,
        drafts=(separated_first, separated_second),
        source_target=target,
    )
    assert len(separated) == 2
    with pytest.raises(ValueError, match="not the same complete composite"):
        validate_target_binding_relationships(drafts=separated, source_target=target)


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
        "CARGO DESCRIPTION SPACER\nBBBU1234567\n"
        "SECOND CARGO DESCRIPTION SPACER\n1X40'HQ CONTAINER\n"
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


def test_labeled_shipper_normalization_reconciles_unrelated_nested_party_fields() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "SHIPPER Name, Address, Phone\n"
        "RECTORSEAL LLC\n"
        "2601 SPENWICK DR.\n"
        "HOUSTON, TX 77055\n\n"
        "CONSIGNEE\n"
        "EGYPTIAN CHEMICALS EAMIC\n"
        "PLOT 63 B INDUSTRIAL ZONE A6\n"
        "10TH OF RAMADAN CITY\n"
        "AL SHARQIA, 44629\n"
    )
    shipper_address_path = "documentPatch.parties.shipper.address"
    shipper_city_path = "documentPatch.parties.shipper.city"
    consignee_address_path = "documentPatch.parties.consignee.address"
    consignee_city_path = "documentPatch.parties.consignee.city"

    def party_draft(text: str, path: str, value_kind: str, group_key: str) -> SpanDraft:
        return replace(
            _test_draft(raw, text, "anchor:" + path, "target_binding", (path,)),
            value_kind=value_kind,
            group_kind="party",
            group_key=group_key,
            render_policy="natural_text",
        )

    drafts = (
        party_draft("2601 SPENWICK DR", shipper_address_path, "address", "party:shipper:0"),
        party_draft("HOUSTON", shipper_city_path, "location", "party:shipper:0"),
        party_draft(
            "PLOT 63 B INDUSTRIAL ZONE A6\n10TH OF RAMADAN CITY\nAL SHARQIA, 44629",
            consignee_address_path,
            "address",
            "party:consignee:0",
        ),
        party_draft(
            "10TH OF RAMADAN CITY",
            consignee_city_path,
            "location",
            "party:consignee:0",
        ),
    )
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": "2601 SPENWICK DR. TX 77055",
                    "city": "HOUSTON",
                },
                "consignee": {
                    "address": "PLOT 63 B INDUSTRIAL ZONE A6 AL SHARQIA, 44629",
                    "city": "10TH OF RAMADAN CITY",
                },
            },
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_labeled_shipper_and_receipt_locality(
        raw=raw,
        drafts=drafts,
        source_target=target,
    )

    consignee_address = tuple(
        row.source_text
        for row in normalized
        if row.logical_key == "anchor:" + consignee_address_path
    )
    assert consignee_address == ("PLOT 63 B INDUSTRIAL ZONE A6", "AL SHARQIA, 44629")
    assert (
        normalize_labeled_shipper_and_receipt_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_compact_equipment_locality_discovers_receipt_projection_and_missing_target_auxiliary() -> (
    None
):
    raw = (
        "--- PAGE 1 ---\n"
        "AAAU1234567 /SEAL0\n"
        "96 CARTONS\n"
        "/FCL/FCL /40HQ/\n"
        "1X40'HQ CONTAINER\n"
        "96 CARTONS /FCL/FCL /40HQ/\n"
        "\n"
        "BBBU1234567 /SEAL1\n"
    )
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "AAAU1234567",
                    "typeDescription": "1X40'HQ CONTAINER",
                },
                {"containerNumber": "BBBU1234567"},
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    def target_draft(*, index: int, text: str) -> SpanDraft:
        start = raw.index(text)
        path = f"documentPatch.containers[{index}].containerNumber"
        return SpanDraft(
            draft_id=f"number_{index}",
            logical_key="anchor:" + path,
            render_mode="target_binding",
            value_kind="identifier",
            group_kind="equipment",
            group_key=f"container:{index}",
            target_paths=(path,),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="accepted_label_evidence",
            render_policy="opaque_identifier",
            rationale="Fixture container number.",
        )

    receipt_text = "1X40'HQ CONTAINER"
    receipt_start = raw.index(receipt_text)
    receipt = SpanDraft(
        draft_id="receipt_0",
        logical_key="agent:equipment_receipt_0",
        render_mode="deterministic_derived",
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
        target_paths=("documentPatch.containers[0]",),
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.containers[0].typeDescription",),
        dependency_bindings=(),
        char_start=receipt_start,
        char_end=receipt_start + len(receipt_text),
        source_text=receipt_text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="derived_surface",
        rationale="Fixture equipment receipt.",
    )
    initial = (
        target_draft(index=0, text="AAAU1234567"),
        receipt,
        target_draft(index=1, text="BBBU1234567"),
    )

    normalized = normalize_compact_equipment_locality(raw=raw, drafts=initial, source_target=target)
    compact_rows = [row for row in normalized if row.source_text == "40HQ"]

    assert [(row.logical_key, row.render_mode, row.group_key) for row in compact_rows] == [
        ("agent:equipment_receipt_0", "deterministic_derived", "container:0"),
        (
            "agent:host_localized_container_1_type_token",
            "deterministic_auxiliary",
            "container:1",
        ),
    ]
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)
    assert (
        normalize_compact_equipment_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )


def test_compact_equipment_locality_discovers_projection_adjacent_to_direct_type_owner() -> None:
    raw = "--- PAGE 1 ---\nAAAU1234567 /SEAL0\n96 CARTONS\n/FCL/FCL /40HQ/\n1X40'HQ CONTAINER\n"
    number_path = "documentPatch.containers[0].containerNumber"
    type_path = "documentPatch.containers[0].typeDescription"
    number = replace(
        _test_draft(raw, "AAAU1234567", "anchor:" + number_path, "target_binding", (number_path,)),
        value_kind="identifier",
        group_kind="equipment",
        group_key="container:0",
        render_policy="opaque_identifier",
    )
    direct = replace(
        _test_draft(
            raw,
            "1X40'HQ CONTAINER",
            "anchor:" + type_path,
            "target_binding",
            (type_path,),
        ),
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "AAAU1234567",
                    "typeDescription": "1X40'HQ CONTAINER",
                }
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_compact_equipment_locality(
        raw=raw, drafts=(number, direct), source_target=target
    )
    type_rows = tuple(row for row in normalized if row.logical_key == direct.logical_key)

    assert tuple(row.source_text for row in type_rows) == ("40HQ", "1X40'HQ CONTAINER")
    assert all(row.target_paths == (type_path,) for row in type_rows)
    assert (
        normalize_compact_equipment_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_compact_equipment_locality_does_not_cross_intervening_content_for_direct_owner() -> None:
    raw = "--- PAGE 1 ---\nAAAU1234567\n96 CARTONS\n/40HQ/\nUNRELATED CARGO\n1X40'HQ CONTAINER\n"
    number_path = "documentPatch.containers[0].containerNumber"
    type_path = "documentPatch.containers[0].typeDescription"
    number = replace(
        _test_draft(raw, "AAAU1234567", "anchor:" + number_path, "target_binding", (number_path,)),
        value_kind="identifier",
        group_kind="equipment",
        group_key="container:0",
        render_policy="opaque_identifier",
    )
    direct = replace(
        _test_draft(
            raw,
            "1X40'HQ CONTAINER",
            "anchor:" + type_path,
            "target_binding",
            (type_path,),
        ),
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "AAAU1234567",
                    "typeDescription": "1X40'HQ CONTAINER",
                }
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_compact_equipment_locality(
        raw=raw, drafts=(number, direct), source_target=target
    )

    assert tuple(row for row in normalized if row.source_text == "40HQ") == ()


def test_compact_equipment_locality_rejects_declared_group_against_adjacent_receipt() -> None:
    raw = "--- PAGE 1 ---\n/FCL/FCL /40HQ/\n1X40'HQ CONTAINER\n"
    target = {
        "documentPatch": {
            "containers": [
                {"typeDescription": "1X40'HQ CONTAINER"},
                {"typeDescription": "1X40'HQ CONTAINER"},
            ]
        }
    }
    receipt_start = raw.index("1X40'HQ CONTAINER")
    receipt = SpanDraft(
        draft_id="receipt_0",
        logical_key="agent:equipment_receipt_0",
        render_mode="deterministic_derived",
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
        target_paths=("documentPatch.containers[0]",),
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.containers[0].typeDescription",),
        dependency_bindings=(),
        char_start=receipt_start,
        char_end=receipt_start + len("1X40'HQ CONTAINER"),
        source_text="1X40'HQ CONTAINER",
        evidence_origin="host_verified_agent_proposal",
        render_policy="derived_surface",
        rationale="Fixture receipt.",
    )
    compact_start = raw.index("40HQ")
    wrong_group = replace(
        receipt,
        draft_id="wrong_group",
        logical_key="agent:equipment_compact_code_1",
        render_mode="deterministic_auxiliary",
        group_key="container:1",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        char_start=compact_start,
        char_end=compact_start + len("40HQ"),
        source_text="40HQ",
        render_policy="opaque_identifier",
    )

    with pytest.raises(ValueError, match=r"physically adjacent.*equipment_receipt_0"):
        normalize_compact_equipment_locality(
            raw=raw,
            drafts=(wrong_group, receipt),
            source_target=target,
        )


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


def test_distinct_source_only_dates_split_into_deterministic_bindings() -> None:
    raw = "--- PAGE 1 ---\nDATE CARGO RECEIVED\n1 MAR 2024\nDATE\n6 MAR 2024\n"
    drafts = tuple(
        replace(
            _test_draft(
                raw,
                source_text,
                "agent:document_date",
                "deterministic_auxiliary",
            ),
            value_kind="date",
            group_kind="document",
            group_key="document",
            render_policy="date_surface",
        )
        for source_text in ("1 MAR 2024", "6 MAR 2024")
    )
    target = {"documentPatch": {"containers": [], "cargoAllocationGroups": [], "cargoPackages": []}}

    normalized = normalize_deterministic_draft_semantics(drafts=drafts, source_target=target)

    assert len({row.logical_key for row in normalized}) == 2
    assert len({row.group_key for row in normalized}) == 2
    assert {row.render_mode for row in normalized} == {"deterministic_auxiliary"}
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_equivalent_source_only_date_formats_remain_one_residual_contract() -> None:
    raw = "--- PAGE 1 ---\n1 MAR 2024\n01 March 2024\n"
    drafts = tuple(
        replace(
            _test_draft(raw, source_text, "agent:document_date", "agent_residual"),
            value_kind="date",
            group_kind="document",
            group_key="document",
            render_policy="date_surface",
        )
        for source_text in ("1 MAR 2024", "01 March 2024")
    )
    target = {"documentPatch": {"containers": [], "cargoAllocationGroups": [], "cargoPackages": []}}

    normalized = normalize_deterministic_draft_semantics(drafts=drafts, source_target=target)

    assert {row.logical_key for row in normalized} == {"agent:document_date"}
    assert {row.render_mode for row in normalized} == {"agent_residual"}
    assert binding_realization(
        draft=normalized[0],
        slots=_realization_slots(*(row.source_text for row in normalized)),
        source_target=target,
    ).requires_agent


def test_exact_single_target_residual_becomes_direct_binding() -> None:
    raw = "--- PAGE 1 ---\nNON-NEGOTIABLE\n"
    path = "documentPatch.negotiability"
    residual = replace(
        _test_draft(raw, "NON-NEGOTIABLE", "agent:legal:negotiability", "agent_residual"),
        value_kind="legal_text",
        group_kind="legal",
        group_key="legal:negotiability",
        target_paths=(path,),
        render_policy="agent_generated",
    )
    target = {
        "documentPatch": {
            "negotiability": "NON_NEGOTIABLE",
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_deterministic_draft_semantics(drafts=(residual,), source_target=target)

    assert len(normalized) == 1
    assert normalized[0].logical_key == "anchor:" + path
    assert normalized[0].render_mode == "target_binding"
    assert normalized[0].target_paths == (path,)
    assert not binding_realization(
        draft=normalized[0],
        slots=_realization_slots(normalized[0].source_text),
        source_target=target,
    ).requires_agent


def test_exact_residual_occurrence_does_not_split_mixed_semantic_contract() -> None:
    raw = "--- PAGE 1 ---\nE / Express B/L\nnon negotiable\n"
    path = "documentPatch.negotiability"
    rows = tuple(
        replace(
            _test_draft(
                raw,
                source_text,
                "agent:negotiability:selected_status",
                "agent_residual",
            ),
            value_kind="commercial_text",
            group_kind="document",
            group_key="document",
            target_paths=(path,),
            render_policy="agent_generated",
        )
        for source_text in ("E / Express B/L", "non negotiable")
    )
    target = {
        "documentPatch": {
            "negotiability": "NON_NEGOTIABLE",
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_deterministic_draft_semantics(drafts=rows, source_target=target)

    assert {row.logical_key for row in normalized} == {rows[0].logical_key}
    assert {row.render_mode for row in normalized} == {"agent_residual"}
    assert tuple(row.source_text for row in normalized) == (
        "E / Express B/L",
        "non negotiable",
    )


def test_unsupported_dangerous_goods_class_mapping_is_bounded_residual() -> None:
    raw = "--- PAGE 1 ---\nUN 2857 CL 2.2\n"
    path = "documentPatch.cargoGroups[0].dangerousGoods[0].hazardCategory"
    draft = replace(
        _test_draft(raw, "CL 2.2", "anchor:" + path, "target_binding", (path,)),
        value_kind="dangerous_goods",
        group_kind="dangerous_goods",
        group_key="dangerous_goods:0:0",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [{"dangerousGoods": [{"hazardCategory": "GASES"}]}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_deterministic_draft_semantics(drafts=(draft,), source_target=target)

    assert normalized[0].render_mode == "agent_residual"
    assert normalized[0].target_paths == (path,)
    assert normalized[0].derivation is None
    assert normalized[0].dependency_paths == ()
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_exact_scalar_is_split_from_segmented_residual_occurrences() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "FREE ZONE - SHEBIN EL KOM\n"
        "ATEF EL SADAT ST .\n"
        "SHEBIN EL KOM 32111-22\n"
        "ALSO NOTIFY PARTY\n"
        "EGYPT\n"
    )
    prefix = "documentPatch.parties.notifyParties[0]"
    paths = (f"{prefix}.address", f"{prefix}.city", f"{prefix}.country")
    address_surface = "FREE ZONE - SHEBIN EL KOM\nATEF EL SADAT ST .\nSHEBIN EL KOM 32111-22"
    first = replace(
        _test_draft(
            raw,
            address_surface,
            "agent:party:notify:0:address_city_country",
            "agent_residual",
            paths,
        ),
        value_kind="address",
        group_kind="party",
        group_key="party:notify:0",
        render_policy="natural_text",
    )
    country = replace(
        _test_draft(
            raw,
            "EGYPT",
            "agent:party:notify:0:address_city_country",
            "agent_residual",
            paths,
        ),
        value_kind="address",
        group_kind="party",
        group_key="party:notify:0",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "parties": {
                "notifyParties": [
                    {
                        "address": "FREE ZONE - SHEBIN EL KOM ATEF EL SADAT ST . 32111-22",
                        "city": "SHEBIN EL KOM",
                        "country": "EGYPT",
                    }
                ]
            },
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_deterministic_draft_semantics(
        drafts=(first, country), source_target=target
    )

    by_surface = {draft.source_text: draft for draft in normalized}
    assert by_surface["EGYPT"].render_mode == "target_binding"
    assert by_surface["EGYPT"].target_paths == (f"{prefix}.country",)
    assert by_surface[address_surface].render_mode == "agent_residual"
    assert by_surface[address_surface].target_paths == (
        f"{prefix}.address",
        f"{prefix}.city",
    )
    validate_target_binding_relationships(drafts=normalized, source_target=target)
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_cross_key_near_ocr_locality_variants_use_one_typed_key_namespace() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "HELICOPOLIS , CAIRO , EGYPT\n"
        "HELIOPOLIS , CAIRO , EGYPT\n"
        "HELIOPOLIS , CAIRO , EGYPT\n"
    )
    first = replace(
        _test_draft(
            raw,
            "HELICOPOLIS , CAIRO , EGYPT",
            "agent:locality:helicopolis",
            "deterministic_auxiliary",
        ),
        value_kind="location",
        group_kind="party",
        group_key="party:agent:0",
    )
    repeated = tuple(
        replace(
            _test_draft(
                raw,
                "HELIOPOLIS , CAIRO , EGYPT",
                "agent:locality:heliopolis",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="location",
            group_kind="party",
            group_key="party:agent:0",
        )
        for index in range(2)
    )
    target = {"documentPatch": {"containers": [], "cargoAllocationGroups": [], "cargoPackages": []}}

    normalized = normalize_deterministic_draft_semantics(
        drafts=(first, *repeated), source_target=target
    )

    assert len(normalized) == 3
    assert {row.logical_key for row in normalized if row.source_text.startswith("HELIOPOLIS")} == {
        "agent:locality:helicopolis"
    }
    assert {row.logical_key for row in normalized if row.source_text.startswith("HELICOPOLIS")} == {
        "agent:locality:helicopolis:typed_variant:b4f01551bf6f"
    }
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_equal_source_only_localities_with_distinct_keys_are_not_merged() -> None:
    raw = "--- PAGE 1 ---\nSINGAPORE\nSINGAPORE\n"
    drafts = tuple(
        replace(
            _test_draft(
                raw,
                "SINGAPORE",
                key,
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="location",
            group_kind="party",
            group_key="party:agent:0",
        )
        for index, key in enumerate(("agent:city", "agent:country"))
    )
    target = {"documentPatch": {"containers": [], "cargoAllocationGroups": [], "cargoPackages": []}}

    normalized = normalize_deterministic_draft_semantics(drafts=drafts, source_target=target)

    assert {row.logical_key for row in normalized} == {"agent:city", "agent:country"}


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

    residuals = tuple(replace(row, render_mode="agent_residual") for row in drafts)
    normalized_residuals = normalize_deterministic_draft_semantics(
        drafts=residuals,
        source_target=target,
    )
    assert {row.render_mode for row in normalized_residuals} == {"deterministic_derived"}
    assert {row.derivation for row in normalized_residuals} == {"temperature_setpoint"}
    assert (
        normalize_deterministic_draft_semantics(
            drafts=normalized_residuals,
            source_target=target,
        )
        == normalized_residuals
    )


def test_reconcile_narrows_numeric_measurement_value_around_separate_unit() -> None:
    raw = "--- PAGE 1 ---\nTEMPERATURE TO BE SET AT +1,0 C\n"
    value_path = "documentPatch.containers[0].temperatureSetpoint.value"
    unit_path = "documentPatch.containers[0].temperatureSetpoint.unit"
    target = {
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    value = replace(
        _test_draft(raw, "+1,0 C", "anchor:" + value_path, "target_binding", (value_path,)),
        value_kind="temperature",
        group_kind="equipment",
        group_key="container:0",
        render_policy="numeric_surface",
    )
    unit = replace(
        _test_draft(raw, "C", "anchor:" + unit_path, "target_binding", (unit_path,)),
        char_start=raw.rindex("C"),
        char_end=raw.rindex("C") + 1,
        value_kind="temperature",
        group_kind="equipment",
        group_key="container:0",
        render_policy="numeric_surface",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(value, unit),
        source_target=target,
    )

    assert {row.source_text for row in normalized} == {"+1,0", "C"}
    assert len(normalized) == 2
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_reconcile_keeps_ambiguous_measurement_overlap_as_hard_error() -> None:
    raw = "--- PAGE 1 ---\nREFERENCE +1,0 C\n"
    target = {
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    unrelated = _test_draft(raw, "+1,0 C", "agent:reference", "deterministic_auxiliary")
    unit_path = "documentPatch.containers[0].temperatureSetpoint.unit"
    unit = replace(
        _test_draft(raw, "C", "anchor:" + unit_path, "target_binding", (unit_path,)),
        char_start=raw.rindex("C"),
        char_end=raw.rindex("C") + 1,
        value_kind="temperature",
        group_kind="equipment",
        group_key="container:0",
        render_policy="numeric_surface",
    )

    with pytest.raises(ValueError, match="template spans overlap"):
        reconcile_draft_overlaps(raw=raw, drafts=(unrelated, unit), source_target=target)


def test_reconcile_removes_only_redundant_nested_auxiliary_identifier_occurrence() -> None:
    raw = "--- PAGE 1 ---\nBE0475317024\nCH-02-BE0475317024\n"
    standalone_start = raw.index("BE0475317024")
    outer_start = raw.index("CH-02-BE0475317024")
    nested_start = outer_start + len("CH-02-")
    base = replace(
        _test_draft(raw, "BE0475317024", "agent:foreign_exporter", "deterministic_auxiliary"),
        draft_id="base_standalone",
        char_start=standalone_start,
        char_end=standalone_start + len("BE0475317024"),
        source_text="BE0475317024",
        value_kind="identifier",
    )
    nested = replace(
        base,
        draft_id="base_nested",
        char_start=nested_start,
        char_end=nested_start + len("BE0475317024"),
    )
    formatted = replace(
        _test_draft(
            raw,
            "CH-02-BE0475317024",
            "agent:export_registration",
            "deterministic_auxiliary",
        ),
        value_kind="identifier",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(base, nested, formatted),
        source_target={
            "documentPatch": {
                "containers": [],
                "cargoAllocationGroups": [],
                "cargoPackages": [],
            }
        },
    )

    assert {(row.logical_key, row.source_text) for row in normalized} == {
        ("agent:foreign_exporter", "BE0475317024"),
        ("agent:export_registration", "CH-02-BE0475317024"),
    }
    formatted_result = next(
        row for row in normalized if row.logical_key == "agent:export_registration"
    )
    assert "Host removed redundant nested auxiliary identifier occurrence" in (
        formatted_result.rationale
    )
    assert "agent:foreign_exporter" in formatted_result.rationale


def test_reconcile_does_not_remove_only_copy_of_nested_auxiliary_identifier() -> None:
    raw = "--- PAGE 1 ---\nCH-02-BE0475317024\n"
    nested = replace(
        _test_draft(raw, "BE0475317024", "agent:foreign_exporter", "deterministic_auxiliary"),
        value_kind="identifier",
    )
    formatted = replace(
        _test_draft(
            raw,
            "CH-02-BE0475317024",
            "agent:export_registration",
            "deterministic_auxiliary",
        ),
        value_kind="identifier",
    )

    with pytest.raises(ValueError, match="template spans overlap"):
        reconcile_draft_overlaps(
            raw=raw,
            drafts=(nested, formatted),
            source_target={
                "documentPatch": {
                    "containers": [],
                    "cargoAllocationGroups": [],
                    "cargoPackages": [],
                }
            },
        )


def test_reconcile_discards_non_token_auxiliary_fragment_inside_target_value() -> None:
    raw = "--- PAGE 1 ---\nBP 49 ROUTE DE THIL\n01122 MONTLUEL\n"
    path = "documentPatch.parties.shipper.address"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
            "parties": {"shipper": {"address": "BP 49 ROUTE DE THIL 01122"}},
        }
    }
    address = replace(
        _test_draft(
            raw,
            "BP 49 ROUTE DE THIL\n01122",
            "anchor:" + path,
            "target_binding",
            (path,),
        ),
        value_kind="address",
        group_kind="party",
        group_key="party:shipper:0",
        render_policy="natural_text",
    )
    fragment_start = raw.index("01122") + 1
    fragment = replace(
        _test_draft(raw, "1", "agent:freight_location", "deterministic_auxiliary"),
        char_start=fragment_start,
        char_end=fragment_start + 1,
        source_text="1",
        value_kind="location",
        group_kind="commercial",
        group_key="commercial:freight",
        render_policy="natural_text",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(address, fragment),
        source_target=target,
    )

    assert len(normalized) == 1
    assert normalized[0].logical_key == address.logical_key
    assert normalized[0].source_text == address.source_text
    assert "Host rejected malformed sub-token auxiliary proposal occurrence" in (
        normalized[0].rationale
    )
    assert "agent:freight_location" in normalized[0].rationale


def test_reconcile_keeps_token_bounded_auxiliary_as_separate_owned_value() -> None:
    raw = "--- PAGE 1 ---\nADDRESS BE 0475317024\n"
    path = "documentPatch.parties.shipper.address"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
            "parties": {"shipper": {"address": "ADDRESS BE 0475317024"}},
        }
    }
    address = replace(
        _test_draft(
            raw,
            "ADDRESS BE 0475317024",
            "anchor:" + path,
            "target_binding",
            (path,),
        ),
        value_kind="address",
        group_kind="party",
        group_key="party:shipper:0",
        render_policy="natural_text",
    )
    country = replace(
        _test_draft(raw, "BE", "agent:country_code", "deterministic_auxiliary"),
        value_kind="location",
        group_kind="party",
        group_key="party:shipper:0",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(address, country),
        source_target=target,
    )

    assert any(row.logical_key == "agent:country_code" for row in normalized)
    assert {row.source_text for row in normalized} == {"ADDRESS", "BE", "0475317024"}


def test_reconcile_discards_redundant_multitarget_occurrence_with_singleton_owners() -> None:
    raw = "--- PAGE 1 ---\nCONSIGNEE: EGYPT\nNOTIFY: EGYPT\n"
    consignee_path = "documentPatch.parties.consignee.country"
    notify_path = "documentPatch.parties.notifyParties[0].country"
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {"country": "EGYPT"},
                "notifyParties": [{"country": "EGYPT"}],
            }
        }
    }
    consignee = replace(
        _test_draft(
            raw,
            "EGYPT",
            "anchor:" + consignee_path,
            "target_binding",
            (consignee_path,),
        ),
        value_kind="location",
        group_kind="party",
        group_key="party:consignee:0",
        render_policy="natural_text",
    )
    notify = replace(
        _test_draft(
            raw,
            "EGYPT",
            "anchor:" + notify_path,
            "target_binding",
            (notify_path,),
            occurrence=1,
        ),
        value_kind="location",
        group_kind="party",
        group_key="party:notify:0",
        render_policy="natural_text",
    )
    redundant = replace(
        notify,
        draft_id="redundant_notify_equality",
        logical_key="agent:redundant_notify_equality",
        target_paths=(consignee_path, notify_path),
        group_key="party:consignee:0|party:notify:0",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(consignee, notify, redundant),
        source_target=target,
    )

    assert len(normalized) == 2
    assert {row.target_paths for row in normalized} == {(consignee_path,), (notify_path,)}
    assert "redundant multi-target" in next(
        row.rationale for row in normalized if row.target_paths == (notify_path,)
    )


def test_reconcile_discards_vessel_auxiliary_only_when_every_repeat_is_contained() -> None:
    raw = "--- PAGE 1 ---\nLONDON EXPRESS\nLONDON EXPRESS\n"
    vessel_path = "documentPatch.transport.vesselName"
    target = {"documentPatch": {"transport": {"vesselName": "LONDON EXPRESS"}}}
    vessels = tuple(
        replace(
            _test_draft(
                raw,
                "LONDON EXPRESS",
                "anchor:" + vessel_path,
                "target_binding",
                (vessel_path,),
                occurrence=index,
            ),
            draft_id=f"vessel_{index}",
            value_kind="equipment",
            group_kind="transport",
            group_key="transport:vesselName",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    suffixes = tuple(
        replace(
            _test_draft(
                raw,
                "EXPRESS",
                "agent:transport:vessel_suffix",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            draft_id=f"suffix_{index}",
            value_kind="operational_text",
            group_kind="transport",
            group_key="transport:vesselName",
            render_policy="natural_text",
        )
        for index in range(2)
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(*vessels, *suffixes),
        source_target=target,
    )

    assert len(normalized) == 2
    assert {row.logical_key for row in normalized} == {"anchor:" + vessel_path}
    assert all("same-scope auxiliary token" in row.rationale for row in normalized)


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


def test_repeated_source_binding_cannot_prove_independent_total_operands() -> None:
    raw = "--- PAGE 1 ---\n1542 PACKAGE\n3084 PACKAGE\n"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
        }
    }
    unsupported = replace(
        _test_draft(raw, "3084", "agent:package_total", "deterministic_derived"),
        value_kind="integer",
        group_kind="package",
        group_key="package:total",
        target_paths=(),
        derivation="sum_package_quantity",
        dependency_paths=(),
        dependency_bindings=("agent:package_quantity", "agent:package_quantity"),
        render_policy="derived_surface",
    )
    supported = replace(
        unsupported,
        draft_id="supported_distinct_total",
        logical_key="agent:distinct_total",
        dependency_bindings=("agent:left_quantity", "agent:right_quantity"),
    )

    normalized_unsupported = normalize_deterministic_draft_semantics(
        drafts=(unsupported,), source_target=target
    )[0]
    normalized_supported = normalize_deterministic_draft_semantics(
        drafts=(supported,), source_target=target
    )[0]

    assert normalized_unsupported.render_mode == "deterministic_auxiliary"
    assert normalized_unsupported.derivation is None
    assert normalized_unsupported.dependency_bindings == ()
    assert "does not prove independent summands" in normalized_unsupported.rationale
    assert normalized_supported.render_mode == "deterministic_derived"
    assert normalized_supported.derivation == "sum_package_quantity"


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


def test_topology_repair_assigns_repeated_multiline_values_by_entity_context() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "MARK-A\nSTEEL COILS\nHOT ROLLED\nHS-A\n"
        "MARK-B\nSTEEL COILS\nHOT ROLLED\nHS-B\n"
    )
    description = "STEEL COILS HOT ROLLED"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {"description": description, "hsCodes": [f"HS-{suffix}"], "marks": f"MARK-{suffix}"}
                for suffix in "AB"
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
        for index, suffix in enumerate("AB")
        for draft in (
            context(f"MARK-{suffix}", f"documentPatch.cargoGroups[{index}].marks"),
            context(f"HS-{suffix}", f"documentPatch.cargoGroups[{index}].hsCodes[0]"),
        )
    )
    missing = tuple(f"documentPatch.cargoGroups[{index}].description" for index in range(2))

    repairs = _repair_uniquely_bracketed_target_paths(
        raw=raw,
        missing_paths=missing,
        occupied=occupied,
        source_target=target,
    )

    assert tuple(row.target_paths[0] for row in repairs) == missing
    assert tuple(row.source_text for row in repairs) == (
        "STEEL COILS\nHOT ROLLED",
        "STEEL COILS\nHOT ROLLED",
    )


def test_topology_repair_recovers_distinctive_identifier_without_field_caption() -> None:
    raw = "--- PAGE 1 ---\nCARGO ROW\nMATERIAL 12345678\nEDEFLY2MU\n2401108590\n"
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "description": "CARGO ROW",
                    "additionalInformation": ["GRADE EDEFLY2MU"],
                    "hsCodes": ["2401108590"],
                }
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    occupied = (
        replace(
            _test_draft(raw, "MATERIAL 12345678", "anchor:material", "target_binding"),
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
    )

    assert len(repairs) == 1
    assert repairs[0].source_text == "EDEFLY2MU"
    assert repairs[0].target_paths == (path,)


def test_topology_repair_retains_contiguous_repeated_projections_of_one_fact() -> None:
    raw = "--- PAGE 1 ---\nCARGO ROW\n13672297\n13672297\n2401108590\n"
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "description": "CARGO ROW",
                    "additionalInformation": ["MATERIAL 13672297"],
                    "hsCodes": ["2401108590"],
                }
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    occupied = (
        replace(
            _test_draft(raw, "CARGO ROW", "anchor:description", "target_binding"),
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
    )

    assert len(repairs) == 2
    assert {row.target_paths for row in repairs} == {(path,)}
    assert [row.char_start for row in repairs] == [
        match.start() for match in re.finditer("13672297", raw)
    ]


def test_topology_repair_rejects_incompatible_shared_anchor_hint() -> None:
    raw = "--- PAGE 1 ---\nBP 49 ROUTE DE THIL\nAIR CONDITIONING CHILLERS\n"
    hazard_path = "documentPatch.cargoGroups[0].dangerousGoods[0].hazardCategory"
    target = {
        "documentPatch": {
            "containers": [],
            "cargoGroups": [
                {
                    "dangerousGoods": [{"hazardCategory": "GASES"}],
                    "description": "AIR CONDITIONING CHILLERS",
                }
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    occupied = (
        replace(
            _test_draft(
                raw,
                "AIR CONDITIONING CHILLERS",
                "anchor:description",
                "target_binding",
            ),
            target_paths=("documentPatch.cargoGroups[0].description",),
        ),
    )

    repairs = _repair_uniquely_bracketed_target_paths(
        raw=raw,
        missing_paths=(hazard_path,),
        occupied=occupied,
        source_target=target,
        source_hints={hazard_path: ("BP 49 ROUTE DE THIL",)},
    )

    assert repairs == ()


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


def test_compact_equipment_locality_counts_derived_receipt_as_stable_target_owner() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "AAAU1234567\n"
        "1X40HIGH CUBE\n"
        "BBBU1234567 40HQ\n"
    )
    number_paths = tuple(
        f"documentPatch.containers[{index}].containerNumber" for index in range(2)
    )
    type_path = "documentPatch.containers[0].typeDescription"
    container_path = "documentPatch.containers[0]"
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAU1234567", "typeDescription": "40HIGH CUBE"},
                {"containerNumber": "BBBU1234567"},
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    number_drafts = tuple(
        replace(
            _test_draft(
                raw,
                number,
                "anchor:" + number_paths[index],
                "target_binding",
                (number_paths[index],),
            ),
            value_kind="identifier",
            group_kind="equipment",
            group_key=f"container:{index}",
        )
        for index, number in enumerate(("AAAU1234567", "BBBU1234567"))
    )
    receipt = replace(
        _test_draft(
            raw,
            "1X40HIGH CUBE",
            "agent:container:0:receipt",
            "deterministic_derived",
            (container_path,),
            (type_path,),
            group_key="container:0",
        ),
        value_kind="equipment",
        group_kind="equipment",
    )
    misplaced = replace(
        _test_draft(
            raw,
            "40HQ",
            "anchor:" + type_path,
            "target_binding",
            (type_path,),
        ),
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
    )

    normalized = normalize_compact_equipment_locality(
        raw=raw,
        drafts=(*number_drafts, receipt, misplaced),
        source_target=target,
    )
    localized = next(row for row in normalized if row.source_text == "40HQ")

    assert localized.render_mode == "deterministic_auxiliary"
    assert localized.group_key == "container:1"
    assert localized.target_paths == ()
    assert any(row.draft_id == receipt.draft_id for row in normalized)


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


def test_auxiliary_identifier_boundary_expands_identical_hyphenated_tokens() -> None:
    raw = "--- PAGE 1 ---\nPOSTAL 32111-22\nPOSTAL 32111-22\n"
    drafts = tuple(
        replace(
            _test_draft(
                raw,
                "32111",
                "agent:postal_code",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="identifier",
            group_kind="party",
            group_key="party:postal",
        )
        for index in range(2)
    )

    normalized = normalize_auxiliary_identifier_boundaries(raw=raw, drafts=drafts)

    assert tuple(row.source_text for row in normalized) == ("32111-22", "32111-22")
    assert all("complete hyphenated identifier" in row.rationale for row in normalized)


def test_decimal_measurement_prefix_expands_to_complete_fractional_surface() -> None:
    raw = "--- PAGE 1 ---\nTOTAL 43956,000 KGM\n"
    draft = replace(
        _test_draft(
            raw,
            "43956",
            "agent:cargo:weight_43956",
            "deterministic_auxiliary",
        ),
        value_kind="decimal_measurement",
        group_kind="cargo",
        group_key="cargo:0",
    )

    (normalized,) = normalize_decimal_measurement_boundaries(raw=raw, drafts=(draft,))

    assert normalized.source_text == "43956,000"
    assert raw[normalized.char_start : normalized.char_end] == "43956,000"
    assert "fractional suffix" in normalized.rationale


def test_decimal_measurement_boundary_does_not_change_identifiers() -> None:
    raw = "--- PAGE 1 ---\nREFERENCE 43956,000\n"
    draft = replace(
        _test_draft(
            raw,
            "43956",
            "agent:reference",
            "deterministic_auxiliary",
        ),
        value_kind="identifier",
    )

    assert normalize_decimal_measurement_boundaries(raw=raw, drafts=(draft,)) == (draft,)


def test_competing_auxiliary_outliers_yield_to_one_complete_mutable_owner() -> None:
    raw = "--- PAGE 1 ---\nPO-123\nPO-123\nCITY NAME\n"
    repeated = tuple(
        replace(
            _test_draft(
                raw,
                "PO-123",
                "agent:purchase_order",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="identifier",
            group_kind="commercial",
            group_key="commercial:purchase_order",
        )
        for index in range(2)
    )
    outlier = replace(
        _test_draft(
            raw,
            "CITY NAME",
            "agent:purchase_order",
            "deterministic_auxiliary",
        ),
        value_kind="identifier",
        group_kind="commercial",
        group_key="commercial:purchase_order",
    )
    city_path = "documentPatch.parties.consignee.city"
    city = replace(
        _test_draft(raw, "CITY NAME", "anchor:" + city_path, "target_binding"),
        value_kind="location",
        group_kind="party",
        group_key="party:consignee:0",
        target_paths=(city_path,),
    )

    normalized = normalize_competing_auxiliary_outliers((*repeated, outlier, city))

    assert tuple(
        row.source_text for row in normalized if row.logical_key == "agent:purchase_order"
    ) == (
        "PO-123",
        "PO-123",
    )
    retained_city = next(row for row in normalized if row.logical_key == "anchor:" + city_path)
    assert "non-modal auxiliary duplicate" in retained_city.rationale


def test_mutable_token_boundary_rejects_quantity_inside_owned_container_number() -> None:
    raw = "--- PAGE 1 ---\nTGBU9219649 /702517\n"
    container_path = "documentPatch.containers[0].containerNumber"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    container = replace(
        _test_draft(raw, "TGBU9219649", "anchor:" + container_path, "target_binding"),
        value_kind="identifier",
        target_paths=(container_path,),
    )
    quantity = replace(
        _test_draft(raw, "96", "anchor:" + quantity_path, "target_binding"),
        value_kind="integer",
        target_paths=(quantity_path,),
        render_policy="numeric_surface",
    )

    with pytest.raises(ValueError, match="inside another owned alphanumeric token"):
        validate_mutable_token_boundaries(raw=raw, drafts=(container, quantity))


def test_anchor_override_retargets_unique_unmatched_projection_to_displaced_fact() -> None:
    raw = "--- PAGE 1 ---\nSTEEL COILS\n"
    path_0 = "documentPatch.cargoGroups[0].description"
    path_1 = "documentPatch.cargoGroups[1].description"
    source_target = {
        "documentPatch": {
            "cargoGroups": [
                {"description": "STEEL COILS"},
                {"description": "COPPER CATHODES"},
            ]
        }
    }
    accepted = replace(
        _test_draft(raw, "STEEL COILS", "anchor:" + path_0, "target_binding"),
        draft_id="anchor_binding_0001",
        value_kind="cargo_text",
        group_kind="cargo",
        group_key="cargo:0",
        target_paths=(path_0,),
        evidence_origin="accepted_label_evidence",
        render_policy="natural_text",
    )
    misplaced = replace(
        accepted,
        draft_id="agent_misplaced",
        logical_key="anchor:" + path_1,
        group_key="cargo:1",
        target_paths=(path_1,),
        evidence_origin="host_verified_agent_proposal",
    )

    revised = apply_anchor_overrides(
        raw=raw,
        source_target=source_target,
        anchors=(accepted,),
        proposed=(misplaced,),
        overrides=(
            AnchorOverride(
                anchor_binding_id="anchor_binding_0001",
                rationale="The model assigned this exact row to the adjacent cargo fact.",
            ),
        ),
    )

    assert len(revised) == 1
    assert revised[0].target_paths == (path_0,)
    assert revised[0].logical_key == "anchor:" + path_0
    assert "unique displaced target scalar" in revised[0].rationale


def test_anchor_override_expands_unique_complete_target_before_semantic_retargeting() -> None:
    raw = "--- PAGE 1 ---\nSEAL: 7025157\nOTHER: 702515\n"
    full_path = "documentPatch.containers[0].sealNumbers[0]"
    short_path = "documentPatch.containers[1].sealNumbers[0]"
    source_target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAU1234567", "sealNumbers": ["7025157"]},
                {"containerNumber": "BBBU1234567", "sealNumbers": ["702515"]},
            ]
        }
    }
    accepted = replace(
        _test_draft(raw, "7025157", "anchor:combined", "target_binding"),
        draft_id="anchor_binding_0001",
        value_kind="identifier",
        group_kind="equipment",
        group_key="container:0",
        target_paths=(full_path, short_path),
        evidence_origin="accepted_label_evidence",
        render_policy="opaque_identifier",
    )
    truncated = replace(
        _test_draft(raw, "702515", "anchor:" + full_path, "target_binding"),
        draft_id="agent_truncated",
        value_kind="identifier",
        group_kind="equipment",
        group_key="container:0",
        target_paths=(full_path,),
        evidence_origin="agent_proposed",
        render_policy="opaque_identifier",
    )
    correct_short = replace(
        _test_draft(
            raw,
            "702515",
            "anchor:" + short_path,
            "target_binding",
            occurrence=1,
        ),
        draft_id="agent_short",
        value_kind="identifier",
        group_kind="equipment",
        group_key="container:1",
        target_paths=(short_path,),
        evidence_origin="agent_proposed",
        render_policy="opaque_identifier",
    )

    revised = apply_anchor_overrides(
        raw=raw,
        source_target=source_target,
        anchors=(accepted,),
        proposed=(truncated, correct_short),
        overrides=(
            AnchorOverride(
                anchor_binding_id=accepted.draft_id,
                rationale="The equal-looking seal facts are independently mutable.",
            ),
        ),
    )

    by_path = {row.target_paths: row for row in revised}
    assert by_path[(full_path,)].source_text == "7025157"
    assert "unique enclosing exact target token" in by_path[(full_path,)].rationale
    assert by_path[(short_path,)].char_start == raw.rindex("702515")


def test_pinned_required_cobinding_wins_equal_valued_semantic_only_conflict() -> None:
    raw = "--- PAGE 1 ---\n96\n96\n"
    quantity_paths = {
        index: (
            f"documentPatch.cargoAllocationGroups[{index}].allocations[0].packageQuantity",
            f"documentPatch.cargoPackages[{index}].quantity",
        )
        for index in range(2)
    }
    source_target = {
        "documentPatch": {
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 96},
                {"groupId": "g2", "packageId": "p2", "quantity": 96},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1"],
                    "allocations": [{"packageId": "p1", "packageQuantity": 96}],
                },
                {
                    "groupId": "g2",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p2"],
                    "allocations": [{"packageId": "p2", "packageQuantity": 96}],
                },
            ],
        }
    }

    def quantity(index: int, occurrence: int, *, accepted: bool = False) -> SpanDraft:
        return replace(
            _test_draft(
                raw,
                "96",
                "anchor:" + "|".join(quantity_paths[index]),
                "target_binding",
                quantity_paths[index],
                occurrence=occurrence,
            ),
            draft_id=(
                "anchor_binding_0001" if accepted else f"proposal_{index}_{occurrence}"
            ),
            value_kind="integer",
            group_kind="package",
            group_key=f"package:{index}",
            evidence_origin=(
                "accepted_label_evidence" if accepted else "host_verified_agent_proposal"
            ),
            render_policy="numeric_surface",
        )

    accepted = quantity(1, 1, accepted=True)
    revised = apply_anchor_overrides(
        raw=raw,
        source_target=source_target,
        anchors=(accepted,),
        proposed=(quantity(0, 0), quantity(0, 1)),
        overrides=(
            AnchorOverride(
                anchor_binding_id=accepted.draft_id,
                rationale="Fixture incorrectly declares the pinned quantity unprinted.",
            ),
        ),
        semantic_only_target_paths=(quantity_paths[1][1],),
    )

    assert {(row.char_start, row.target_paths) for row in revised} == {
        (raw.index("96"), quantity_paths[0]),
        (raw.rindex("96"), quantity_paths[1]),
    }
    assert refuted_semantic_only_target_paths(revised) == frozenset(quantity_paths[1])
    retained = next(row for row in revised if row.target_paths == quantity_paths[1])
    assert "equal-valued competing target" in retained.rationale


def test_complete_conflicting_replacements_infer_anchor_override() -> None:
    raw = "--- PAGE 1 ---\nROLE A: PORTX\nROLE B: PORTX\n"
    paths = (
        "documentPatch.route.portOfLoading.name",
        "documentPatch.route.portOfDischarge.name",
    )
    target = {
        "documentPatch": {
            "route": {
                "portOfLoading": {"name": "PORTX"},
                "portOfDischarge": {"name": "PORTX"},
            }
        }
    }
    accepted = replace(
        _test_draft(raw, "PORTX", "anchor:shared", "target_binding"),
        draft_id="anchor_binding_0001",
        value_kind="location",
        group_kind="route",
        group_key="route:shared",
        target_paths=paths,
        evidence_origin="accepted_label_evidence",
        render_policy="natural_text",
    )

    def proposal(path: str, occurrence: int) -> SpanDraft:
        return replace(
            _test_draft(
                raw,
                "PORTX",
                "anchor:" + path,
                "target_binding",
                occurrence=occurrence,
            ),
            value_kind="location",
            group_kind="route",
            group_key=path,
            target_paths=(path,),
            evidence_origin="agent_proposed",
            render_policy="natural_text",
        )

    revised = apply_anchor_overrides(
        raw=raw,
        source_target=target,
        anchors=(accepted,),
        proposed=(proposal(paths[0], 0), proposal(paths[1], 1)),
        overrides=(),
    )

    assert {row.target_paths for row in revised} == {(paths[0],), (paths[1],)}
    assert all("inferred removal" in row.rationale for row in revised)


def test_incomplete_conflicting_replacement_does_not_infer_anchor_override() -> None:
    raw = "--- PAGE 1 ---\nROLE A: PORTX\nROLE B: PORTX\n"
    paths = (
        "documentPatch.route.portOfLoading.name",
        "documentPatch.route.portOfDischarge.name",
    )
    target = {
        "documentPatch": {
            "route": {
                "portOfLoading": {"name": "PORTX"},
                "portOfDischarge": {"name": "PORTX"},
            }
        }
    }
    accepted = replace(
        _test_draft(raw, "PORTX", "anchor:shared", "target_binding"),
        draft_id="anchor_binding_0001",
        value_kind="location",
        group_kind="route",
        group_key="route:shared",
        target_paths=paths,
        evidence_origin="accepted_label_evidence",
        render_policy="natural_text",
    )
    incomplete = replace(
        _test_draft(raw, "PORTX", "anchor:" + paths[0], "target_binding", occurrence=1),
        value_kind="location",
        group_kind="route",
        group_key=paths[0],
        target_paths=(paths[0],),
        evidence_origin="agent_proposed",
        render_policy="natural_text",
    )

    revised = apply_anchor_overrides(
        raw=raw,
        source_target=target,
        anchors=(accepted,),
        proposed=(incomplete,),
        overrides=(),
    )

    assert accepted in revised
    assert all("inferred removal" not in row.rationale for row in revised)


def test_competing_equal_target_occurrences_relocate_by_unique_row_topology() -> None:
    raw = "--- PAGE 1 ---\nCARGO A\nMATERIAL\n13668880\nCARGO B\nMATERIAL 13668880\n"
    paths = tuple(
        f"documentPatch.cargoGroups[{index}].additionalInformation[0]" for index in range(2)
    )
    description_paths = tuple(
        f"documentPatch.cargoGroups[{index}].description" for index in range(2)
    )
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"description": "CARGO A", "additionalInformation": ["MATERIAL 13668880"]},
                {"description": "CARGO B", "additionalInformation": ["MATERIAL 13668880"]},
            ]
        }
    }
    descriptions = tuple(
        replace(
            _test_draft(
                raw,
                f"CARGO {'A' if index == 0 else 'B'}",
                "anchor:" + description_paths[index],
                "target_binding",
            ),
            value_kind="cargo_text",
            group_kind="cargo",
            group_key=f"cargo:{index}",
            target_paths=(description_paths[index],),
            render_policy="natural_text",
        )
        for index in range(2)
    )
    wrong_shared_start = raw.index("MATERIAL 13668880")
    competing = tuple(
        SpanDraft(
            draft_id=f"wrong_{index}",
            logical_key="anchor:" + path,
            render_mode="target_binding",
            value_kind="cargo_text",
            group_kind="cargo",
            group_key=f"cargo:{index}",
            target_paths=(path,),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=wrong_shared_start,
            char_end=wrong_shared_start + len("MATERIAL 13668880"),
            source_text="MATERIAL 13668880",
            evidence_origin="agent_proposed",
            render_policy="natural_text",
            rationale="Fixture collision.",
        )
        for index, path in enumerate(paths)
    )

    revised = apply_anchor_overrides(
        raw=raw,
        source_target=target,
        anchors=(),
        proposed=(*descriptions, *competing),
        overrides=(),
    )

    by_path = {row.target_paths: row for row in revised}
    assert by_path[(paths[0],)].source_text == "MATERIAL\n13668880"
    assert by_path[(paths[1],)].source_text == "MATERIAL 13668880"
    assert "unique same-entity topology assignment" in by_path[(paths[0],)].rationale


def test_residual_edge_is_trimmed_when_exact_target_has_separate_owner() -> None:
    raw = "--- PAGE 1 ---\nSTREET 1 EGYPT\n"
    address_path = "documentPatch.parties.consignee.address"
    country_path = "documentPatch.parties.consignee.country"
    target = {
        "documentPatch": {"parties": {"consignee": {"address": "STREET 1", "country": "EGYPT"}}}
    }
    residual = replace(
        _test_draft(raw, "STREET 1 EGYPT", "agent:consignee:address", "agent_residual"),
        value_kind="address",
        group_kind="party",
        group_key="party:consignee:0",
        target_paths=(address_path, country_path),
        render_policy="natural_text",
    )
    country = replace(
        _test_draft(raw, "EGYPT", "anchor:" + country_path, "target_binding"),
        value_kind="location",
        group_kind="party",
        group_key="party:consignee:0",
        target_paths=(country_path,),
        render_policy="natural_text",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(residual, country),
        source_target=target,
    )

    by_key = {row.logical_key: row for row in reconciled}
    assert by_key[residual.logical_key].source_text == "STREET 1"
    assert by_key[residual.logical_key].target_paths == (address_path,)
    assert by_key[country.logical_key].source_text == "EGYPT"


def test_target_backed_residual_excludes_separated_trailing_footnote_marker() -> None:
    raw = "--- PAGE 1 ---\nSHEBIN EL KOM 32111-22 *\nS.A.E.\n"
    address_path = "documentPatch.parties.notifyParties[0].address"
    target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [{"address": "SHEBIN EL KOM 32111-22"}],
                "consignee": {"name": "S.A.E."},
            }
        }
    }
    residual = replace(
        _test_draft(
            raw,
            "SHEBIN EL KOM 32111-22 *",
            "agent:notify:address",
            "agent_residual",
        ),
        value_kind="address",
        group_kind="party",
        group_key="party:notify:0",
        target_paths=(address_path,),
        render_policy="natural_text",
    )
    attached_punctuation = replace(
        _test_draft(raw, "S.A.E.", "agent:consignee:name", "agent_residual"),
        value_kind="organization",
        group_kind="party",
        group_key="party:consignee:0",
        target_paths=("documentPatch.parties.consignee.name",),
        render_policy="natural_text",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(residual, attached_punctuation),
        source_target=target,
    )

    by_key = {row.logical_key: row for row in normalized}
    assert by_key[residual.logical_key].source_text == "SHEBIN EL KOM 32111-22"
    assert by_key[attached_punctuation.logical_key].source_text == "S.A.E."
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_targetless_residual_edge_is_trimmed_around_deterministic_auxiliary() -> None:
    raw = "--- PAGE 1 ---\nEGYPT IMPORT IS FREE OUT\n"
    target = {"documentPatch": {}}
    clause = replace(
        _test_draft(
            raw,
            "EGYPT IMPORT IS FREE OUT",
            "agent:operational:destination",
            "agent_residual",
        ),
        value_kind="operational_text",
        group_kind="transport",
        group_key="transport:destination_terms",
        render_policy="agent_generated",
    )
    country = replace(
        _test_draft(
            raw,
            "EGYPT",
            "agent:location:destination_clause_country",
            "deterministic_auxiliary",
        ),
        value_kind="location",
        group_kind="route",
        group_key="route:destination_clause",
        render_policy="natural_text",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(country, clause),
        source_target=target,
    )

    by_key = {row.logical_key: row for row in reconciled}
    assert by_key[country.logical_key].source_text == "EGYPT"
    assert by_key[clause.logical_key].source_text == "IMPORT IS FREE OUT"
    assert by_key[country.logical_key].char_end < by_key[clause.logical_key].char_start


def test_residual_isolates_exact_cobound_fact_and_consolidates_duplicate_contract() -> None:
    raw = "--- PAGE 1 ---\n96\n96 CARTONS SERBIA TOBACCO\n"
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    description_path = "documentPatch.cargoGroups[0].description"
    target = {
        "documentPatch": {
            "cargoAllocationGroups": [
                {
                    "groupId": "allocation-0",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["package-0"],
                    "allocations": [{"packageId": "package-0", "packageQuantity": 96}],
                }
            ],
            "cargoPackages": [
                {"packageId": "package-0", "groupId": "allocation-0", "quantity": 96}
            ],
            "cargoGroups": [{"description": "SERBIA TOBACCO"}],
        }
    }
    complete = replace(
        _test_draft(
            raw,
            "96 CARTONS SERBIA TOBACCO",
            "agent:quantity_and_description",
            "agent_residual",
        ),
        value_kind="package",
        group_kind="cargo",
        group_key="allocation:0:0",
        target_paths=(allocation_path, quantity_path, description_path),
        render_policy="agent_generated",
    )
    quantity = replace(
        _test_draft(raw, "96", complete.logical_key, "agent_residual"),
        value_kind=complete.value_kind,
        group_kind=complete.group_kind,
        group_key=complete.group_key,
        target_paths=complete.target_paths,
        render_policy=complete.render_policy,
    )
    duplicate_description = replace(
        complete,
        draft_id="duplicate_description",
        logical_key="agent:description",
        value_kind="cargo_text",
        group_kind="cargo",
        group_key="cargo:0",
        target_paths=(description_path,),
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(quantity, complete, duplicate_description),
        source_target=target,
    )

    assert len(reconciled) == 2
    by_mode = {row.render_mode: row for row in reconciled}
    assert by_mode["target_binding"].target_paths == (allocation_path, quantity_path)
    assert by_mode["target_binding"].source_text == "96"
    assert by_mode["agent_residual"].target_paths == (description_path,)
    assert by_mode["agent_residual"].source_text == "96 CARTONS SERBIA TOBACCO"


def test_package_type_locality_recovers_missing_row_and_reassigns_duplicate() -> None:
    raw = "--- PAGE 1 ---\nAAAU1234567 96 CARTONS\nBBBU1234567 96 CARTONS\n"
    quantity_paths = tuple(f"documentPatch.cargoPackages[{index}].quantity" for index in range(2))
    type_paths = tuple(f"documentPatch.cargoPackages[{index}].typeCategory" for index in range(2))
    target = {
        "documentPatch": {
            "cargoPackages": [
                {"quantity": 96, "typeCategory": "PACKAGE_CARTON"},
                {"quantity": 96, "typeCategory": "PACKAGE_CARTON"},
            ]
        }
    }
    quantities = tuple(
        replace(
            _test_draft(
                raw,
                "96",
                "anchor:" + quantity_paths[index],
                "target_binding",
                occurrence=index,
            ),
            value_kind="integer",
            group_kind="package",
            group_key=f"package:{index}",
            target_paths=(quantity_paths[index],),
            render_policy="numeric_surface",
        )
        for index in range(2)
    )
    misplaced = replace(
        _test_draft(raw, "CARTONS", "anchor:" + type_paths[0], "target_binding", occurrence=1),
        value_kind="package",
        group_kind="package",
        group_key="package:0",
        target_paths=(type_paths[0],),
        render_policy="categorical_surface",
    )
    correct_second = replace(
        misplaced,
        draft_id="correct_second",
        logical_key="anchor:" + type_paths[1],
        group_key="package:1",
        target_paths=(type_paths[1],),
    )

    normalized = normalize_package_type_row_locality(
        raw=raw,
        proposed=(misplaced, correct_second),
        contexts=quantities,
        candidate_paths=type_paths,
        source_target=target,
    )

    first_type_start = raw.index("CARTONS")
    second_type_start = raw.rindex("CARTONS")
    assert {
        (row.char_start, row.target_paths) for row in normalized if row.source_text == "CARTONS"
    } == {
        (first_type_start, (type_paths[0],)),
        (second_type_start, (type_paths[1],)),
    }


def test_package_type_locality_does_not_duplicate_path_owned_by_residual() -> None:
    raw = "--- PAGE 1 ---\nON 21 PALLETS\n"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    type_path = "documentPatch.cargoPackages[0].typeCategory"
    cargo_path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    target = {
        "documentPatch": {
            "cargoGroups": [{"additionalInformation": ["ON 21 PALLETS"]}],
            "cargoPackages": [
                {"packageId": "package-0", "quantity": 21, "typeCategory": "PACKAGE_PALLET"}
            ],
        }
    }
    residual = replace(
        _test_draft(raw, "ON 21 PALLETS", "agent:package_surface", "agent_residual"),
        value_kind="package",
        group_kind="package",
        group_key="package:0",
        target_paths=(cargo_path, quantity_path, type_path),
        render_policy="agent_generated",
    )

    normalized = normalize_package_type_row_locality(
        raw=raw,
        proposed=(residual,),
        contexts=(residual,),
        candidate_paths=(type_path,),
        source_target=target,
    )

    assert normalized == (residual,)


def test_package_type_locality_keeps_complete_inflected_target_owner() -> None:
    raw = "--- PAGE 1 ---\n2 Pallet(s)\n"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    type_path = "documentPatch.cargoPackages[0].typeCategory"
    quantity = replace(
        _test_draft(raw, "2", "anchor:" + quantity_path, "target_binding"),
        value_kind="integer",
        group_kind="package",
        group_key="package:0",
        target_paths=(quantity_path,),
        render_policy="numeric_surface",
    )
    complete = replace(
        _test_draft(raw, "Pallet(s)", "anchor:" + type_path, "target_binding"),
        value_kind="package",
        group_kind="package",
        group_key="package:0",
        target_paths=(type_path,),
        render_policy="categorical_surface",
    )
    target = {
        "documentPatch": {"cargoPackages": [{"quantity": 2, "typeCategory": "PACKAGE_PALLET"}]}
    }

    assert normalize_package_type_row_locality(
        raw=raw,
        proposed=(complete,),
        contexts=(quantity,),
        candidate_paths=(type_path,),
        source_target=target,
    ) == (complete,)


def test_exact_source_only_duplicate_yields_to_same_span_target_owner() -> None:
    raw = "--- PAGE 1 ---\n7100003132\n"
    path = "documentPatch.forwardingAndExportReferences[0]"
    owner = replace(
        _test_draft(raw, "7100003132", "anchor:" + path, "target_binding", (path,)),
        value_kind="identifier",
        group_kind="document",
        group_key="document",
        render_policy="natural_text",
    )
    duplicate = replace(
        _test_draft(
            raw,
            "7100003132",
            "agent:commercial_invoice_reference",
            "deterministic_auxiliary",
        ),
        value_kind="identifier",
        group_kind="commercial",
        group_key="commercial:invoice",
        render_policy="natural_text",
    )

    assert _drop_exact_source_only_duplicates(retained=(owner,), proposed=(duplicate,)) == ()
    assert _drop_exact_source_only_duplicates(
        retained=(owner,), proposed=(replace(duplicate, dependency_bindings=("agent:x",)),)
    ) == (replace(duplicate, dependency_bindings=("agent:x",)),)
    assert (
        _drop_exact_source_only_duplicates(
            retained=(owner,), proposed=(replace(duplicate, render_mode="literal_static"),)
        )
        == ()
    )


def test_incompatible_exact_target_duplicate_yields_to_typed_source_only_owner() -> None:
    raw = "--- PAGE 1 ---\n2601 SPENWICK DR\nBOOKING #: 64937668\n"
    path = "documentPatch.parties.shipper.address"
    valid = replace(
        _test_draft(raw, "2601 SPENWICK DR", "anchor:" + path, "target_binding", (path,)),
        value_kind="address",
        group_kind="party",
        group_key="party:shipper:0",
        render_policy="natural_text",
    )
    invalid = replace(
        _test_draft(raw, "64937668", "anchor:" + path, "target_binding", (path,)),
        value_kind="address",
        group_kind="party",
        group_key="party:shipper:0",
        render_policy="natural_text",
    )
    booking = replace(
        _test_draft(raw, "64937668", "agent:booking_reference", "deterministic_auxiliary"),
        value_kind="identifier",
        group_kind="commercial",
        group_key="commercial:booking",
        render_policy="opaque_identifier",
    )
    target = {"documentPatch": {"parties": {"shipper": {"address": "2601 SPENWICK DR"}}}}

    assert _resolve_exact_target_source_only_conflicts(
        drafts=(valid, invalid, booking), source_target=target
    ) == (valid, booking)


def test_structured_row_locality_recovers_every_repeated_package_category() -> None:
    raw = "--- PAGE 1 ---\nTCLU1610520\n96 CARTONS\n96 CARTONS /FCL/FCL/\n"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    type_path = "documentPatch.cargoPackages[0].typeCategory"
    target = {
        "documentPatch": {
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "allocations": [
                        {
                            "containerNumber": "TCLU1610520",
                            "packageId": "p1",
                            "packageQuantity": 96,
                        }
                    ],
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
            "containers": [{"containerNumber": "TCLU1610520"}],
        }
    }
    quantities = tuple(
        replace(
            _test_draft(raw, "96", "anchor:" + quantity_path, "target_binding", occurrence=index),
            value_kind="integer",
            group_kind="package",
            group_key="package:0",
            target_paths=(allocation_path, quantity_path),
            render_policy="numeric_surface",
        )
        for index in range(2)
    )

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=quantities,
        source_target=target,
    )

    categories = tuple(row for row in normalized if row.target_paths == (type_path,))
    assert tuple(row.source_text for row in categories) == ("CARTONS", "CARTONS")
    assert len({row.logical_key for row in categories}) == 1
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


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


def test_container_count_rejects_bare_equipment_multiplier() -> None:
    raw = "--- PAGE 1 ---\n1 X 20' STD CONTAINER\n"
    containers_path = "documentPatch.containers"
    draft = replace(
        _test_draft(
            raw,
            "1 X",
            "agent:container_count",
            "deterministic_derived",
            (containers_path,),
            (containers_path,),
        ),
        value_kind="integer",
        group_kind="equipment",
        group_key="equipment:all",
        derivation="container_count",
        render_policy="derived_surface",
    )
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": "20' STD CONTAINER"}],
            "cargoPackages": [],
        }
    }

    with pytest.raises(ValueError, match="bare equipment multiplier"):
        validate_binding_realizations(raw=raw, drafts=(draft,), source_target=target)


def test_equipment_receipt_expands_over_overlapping_type_dependency() -> None:
    raw = "--- PAGE 1 ---\n1X40HIGH CUBE SAID TO CONTAIN\n"
    container_path = "documentPatch.containers[0]"
    type_path = f"{container_path}.typeDescription"
    outer = replace(
        _test_draft(
            raw,
            "1X40HIGH",
            "agent:equipment_receipt:container:0",
            "deterministic_derived",
            (container_path,),
            (container_path, type_path),
            group_key="container:0",
        ),
        derivation="equipment_receipt",
    )
    inner = replace(
        _test_draft(
            raw,
            "40HIGH CUBE",
            "anchor:" + type_path,
            "target_binding",
            (type_path,),
            group_key="container:0",
        ),
        value_kind="equipment",
    )
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": "40HIGH CUBE"}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(outer, inner),
        source_target=target,
    )

    assert len(reconciled) == 1
    assert reconciled[0].logical_key == outer.logical_key
    assert reconciled[0].source_text == "1X40HIGH CUBE"
    assert reconciled[0].char_start == outer.char_start
    assert reconciled[0].char_end == inner.char_end


def test_carrier_static_aliases_keep_one_canonical_target_owner() -> None:
    raw = "--- PAGE 1 ---\nORIENT OVERSEAS CONTAINER LINE\nwww.oocl.com\n"
    path = "documentPatch.parties.carrier.name"
    canonical = replace(
        _test_draft(
            raw, "ORIENT OVERSEAS CONTAINER LINE", "carrier:name", "carrier_static", (path,)
        ),
        value_kind="organization",
        group_kind="carrier",
        group_key="carrier:principal",
        evidence_origin="accepted_label_evidence",
    )
    domain = replace(
        _test_draft(raw, "www.oocl.com", "carrier:domain", "carrier_static", (path,)),
        value_kind="url_or_domain",
        group_kind="carrier",
        group_key="carrier:domain",
    )
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "ORIENT OVERSEAS CONTAINER LINE"}},
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(canonical, domain),
        source_target=target,
    )
    by_key = {draft.logical_key: draft for draft in reconciled}

    assert by_key["carrier:name"].target_paths == (path,)
    assert by_key["carrier:domain"].target_paths == ()
    validate_target_binding_relationships(drafts=reconciled, source_target=target)


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


def test_structured_locality_completes_unique_party_address_suffix() -> None:
    raw = "--- PAGE 1 ---\nCONSIGNEE\nFREE ZONE\nSTREET\nCITY 32111-22 EGYPT\n"
    address_path = "documentPatch.parties.consignee.address"
    city_path = "documentPatch.parties.consignee.city"
    address = replace(
        _test_draft(
            raw,
            "FREE ZONE\nSTREET",
            "anchor:" + address_path,
            "target_binding",
            (address_path,),
            group_key="party:consignee:0",
        ),
        value_kind="address",
        group_kind="party",
        render_policy="natural_text",
    )
    city = replace(
        _test_draft(
            raw,
            "CITY",
            "anchor:" + city_path,
            "target_binding",
            (city_path,),
            group_key="party:consignee:0",
        ),
        value_kind="location",
        group_kind="party",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "address": "FREE ZONE STREET 32111-22",
                    "city": "CITY",
                }
            },
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw, drafts=(address, city), source_target=target
    )
    address_rows = tuple(row for row in normalized if row.logical_key == address.logical_key)

    assert tuple(row.source_text for row in address_rows) == (
        "FREE ZONE\nSTREET",
        "32111-22",
    )
    assert (
        binding_realization(
            draft=address_rows[0],
            slots=_realization_slots(*(row.source_text for row in address_rows)),
            source_target=target,
        ).mode
        == "segmented_surface"
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_leaves_ambiguous_party_address_suffix_for_review() -> None:
    raw = "--- PAGE 1 ---\nFREE ZONE\nCITY 32111-22 / 32111-22\n"
    address_path = "documentPatch.parties.consignee.address"
    city_path = "documentPatch.parties.consignee.city"
    address = replace(
        _test_draft(
            raw,
            "FREE ZONE",
            "anchor:" + address_path,
            "target_binding",
            (address_path,),
            group_key="party:consignee:0",
        ),
        value_kind="address",
        group_kind="party",
        render_policy="natural_text",
    )
    city = replace(
        _test_draft(
            raw,
            "CITY",
            "anchor:" + city_path,
            "target_binding",
            (city_path,),
            group_key="party:consignee:0",
        ),
        value_kind="location",
        group_kind="party",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "parties": {"consignee": {"address": "FREE ZONE 32111-22", "city": "CITY"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw, drafts=(address, city), source_target=target
    )

    assert tuple(row for row in normalized if row.logical_key == address.logical_key) == (address,)


def test_structured_locality_detaches_explicit_loading_terminal_repeat() -> None:
    raw = "--- PAGE 1 ---\nPORT OF LOADING\nANTWERP\nLOADING PIER/TERMINAL\nANTWERP\n"
    path = "documentPatch.route.portOfLoading.name"
    port_rows = tuple(
        replace(
            _test_draft(
                raw,
                "ANTWERP",
                "anchor:" + path,
                "target_binding",
                (path,),
                group_key="route:portOfLoading",
                occurrence=index,
            ),
            value_kind="location",
            group_kind="route",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "route": {"portOfLoading": {"name": "ANTWERP"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(raw=raw, drafts=port_rows, source_target=target)
    by_surface = {row.char_start: row for row in normalized}

    assert by_surface[port_rows[0].char_start].target_paths == (path,)
    terminal = by_surface[port_rows[1].char_start]
    assert terminal.render_mode == "deterministic_auxiliary"
    assert terminal.group_key == "route:loading_terminal"
    assert terminal.target_paths == ()
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_owns_unowned_explicit_loading_terminal_value() -> None:
    raw = "--- PAGE 1 ---\nLOADING PIER/TERMINAL\n  ANTWERP GATE 3  \n"
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(raw=raw, drafts=(), source_target=target)

    assert len(normalized) == 1
    terminal = normalized[0]
    assert terminal.source_text == "ANTWERP GATE 3"
    assert terminal.render_mode == "deterministic_auxiliary"
    assert terminal.value_kind == "location"
    assert terminal.group_key == "route:loading_terminal"
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_owns_exact_labeled_source_only_location() -> None:
    raw = "--- PAGE 1 ---\nFOREIGN EXPORTER COUNTRY:\nSWITZERLAND\n"
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(raw=raw, drafts=(), source_target=target)

    assert len(normalized) == 1
    location = normalized[0]
    assert location.logical_key == "agent:customs:foreign_exporter_country"
    assert location.render_mode == "deterministic_auxiliary"
    assert location.value_kind == "location"
    assert location.group_key == "customs:foreign_exporter"
    assert location.source_text == "SWITZERLAND"
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_owns_freight_payable_value_and_drops_page_number() -> None:
    raw = "--- PAGE 1 ---\nFreight payable at\nDEPARTURE\n"
    page_number = replace(
        _test_draft(raw, "1", "agent:freight_payable_location", "deterministic_auxiliary"),
        value_kind="commercial_text",
        group_kind="commercial",
        group_key="commercial:freight",
        render_policy="natural_text",
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(
        raw=raw, drafts=(page_number,), source_target=target
    )

    assert len(normalized) == 1
    payable = normalized[0]
    assert payable.logical_key == "agent:commercial:freight_payable_location"
    assert payable.source_text == "DEPARTURE"
    assert payable.render_mode == "deterministic_auxiliary"
    assert payable.value_kind == "commercial_text"
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )


def test_structured_locality_owns_captioned_and_inline_commercial_dates_and_charges() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "DATE CARGO RECEIVED\n"
        "1 MAR 2024\n"
        "AS PER P/I NO. 7100003132 DD. 15/03/202 INCOTERMS 2020\n"
        "ORIGIN PORT CHARGE PREPAID\n"
        "DESTINATION PORT CHARGE COLLECT\n"
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(raw=raw, drafts=(), source_target=target)
    by_key = {row.logical_key: row for row in normalized}

    assert by_key["agent:transport:cargo_received_date"].source_text == "1 MAR 2024"
    assert by_key["agent:commercial:proforma_invoice_date"].source_text == "15/03/202"
    assert by_key["agent:commercial:origin_port_charge_place"].source_text == "ORIGIN"
    assert by_key["agent:commercial:origin_port_charge_payment"].source_text == "PREPAID"
    assert by_key["agent:commercial:destination_port_charge_place"].source_text == "DESTINATION"
    assert by_key["agent:commercial:destination_port_charge_payment"].source_text == "COLLECT"
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_appends_exact_labeled_route_repeats() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "PORT OF DISCHARGE\nALEXANDRIA\n"
        "--- PAGE 2 ---\n"
        "PORT OF DISCHARGE\nALEXANDRIA\n"
        "ALEXANDRIA IN LEGAL TEXT\n"
    )
    path = "documentPatch.route.portOfDischarge.name"
    owner = replace(
        _test_draft(raw, "ALEXANDRIA", "anchor:" + path, "target_binding", (path,)),
        value_kind="location",
        group_kind="route",
        group_key="route:portOfDischarge",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "route": {"portOfDischarge": {"name": "ALEXANDRIA"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(raw=raw, drafts=(owner,), source_target=target)
    rows = tuple(row for row in normalized if row.logical_key == owner.logical_key)

    assert tuple(row.char_start for row in rows) == (
        raw.index("ALEXANDRIA"),
        raw.index("ALEXANDRIA", raw.index("--- PAGE 2 ---")),
    )
    assert all(row.char_start != raw.rindex("ALEXANDRIA") for row in rows)
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )


def test_structured_locality_splits_exhaustive_captioned_equal_route_facts() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "PORT OF DISCHARGE\nEL DEKHEILA\n"
        "PLACE OF DELIVERY\nEL DEKHEILA\n"
    )
    discharge_path = "documentPatch.route.portOfDischarge.name"
    delivery_path = "documentPatch.route.placeOfDelivery.name"
    target_paths = (delivery_path, discharge_path)
    shared_key = "anchor:" + "|".join(target_paths)
    rows = tuple(
        replace(
            _test_draft(
                raw,
                "EL DEKHEILA",
                shared_key,
                "target_binding",
                target_paths,
                occurrence=index,
            ),
            value_kind="location",
            group_kind="route",
            group_key="compound:route:placeOfDelivery|route:portOfDischarge",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "route": {
                "portOfDischarge": {"name": "EL DEKHEILA"},
                "placeOfDelivery": {"name": "EL DEKHEILA"},
            },
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=rows,
        source_target=target,
    )
    by_path = {row.target_paths: row for row in normalized}

    assert by_path[(discharge_path,)].char_start == raw.index("EL DEKHEILA")
    assert by_path[(delivery_path,)].char_start == raw.rindex("EL DEKHEILA")
    validate_target_binding_relationships(drafts=normalized, source_target=target)
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )


def test_structured_locality_does_not_guess_unlabeled_equal_route_fact() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "PORT OF DISCHARGE\nEL DEKHEILA\n"
        "UNLABELED COPY\nEL DEKHEILA\n"
    )
    discharge_path = "documentPatch.route.portOfDischarge.name"
    delivery_path = "documentPatch.route.placeOfDelivery.name"
    target_paths = (delivery_path, discharge_path)
    shared_key = "anchor:" + "|".join(target_paths)
    rows = tuple(
        replace(
            _test_draft(
                raw,
                "EL DEKHEILA",
                shared_key,
                "target_binding",
                target_paths,
                occurrence=index,
            ),
            value_kind="location",
            group_kind="route",
            group_key="compound:route:placeOfDelivery|route:portOfDischarge",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "route": {
                "portOfDischarge": {"name": "EL DEKHEILA"},
                "placeOfDelivery": {"name": "EL DEKHEILA"},
            },
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=rows,
        source_target=target,
    )

    assert normalized == rows
    with pytest.raises(
        ValueError,
        match="repeated binding aggregates independently mutable target facts",
    ):
        validate_target_binding_relationships(drafts=normalized, source_target=target)


def test_structured_locality_appends_complete_inline_transport_repeats() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "PORT OF LOADING\nRIJEKA\n"
        "VESSEL\nAL MURABBA\n"
        "VOYAGE\n521E\n"
        "PORT OF LOADING: RIJEKA\n"
        "VESSEL NAME: AL MURABBA VOYAGE: 521E\n"
    )
    port_path = "documentPatch.route.portOfLoading.name"
    vessel_path = "documentPatch.transport.vesselName"
    voyage_path = "documentPatch.transport.voyageNumber"

    def owner(text: str, path: str, value_kind: str) -> SpanDraft:
        group_kind = "route" if path == port_path else "transport"
        group_key = "route:portOfLoading" if path == port_path else "transport"
        return replace(
            _test_draft(
                raw,
                text,
                "anchor:" + path,
                "target_binding",
                (path,),
                group_key=group_key,
            ),
            value_kind=value_kind,
            group_kind=group_kind,
            render_policy="natural_text",
        )

    rows = (
        owner("RIJEKA", port_path, "location"),
        owner("AL MURABBA", vessel_path, "equipment"),
        owner("521E", voyage_path, "identifier"),
    )
    target = {
        "documentPatch": {
            "route": {"portOfLoading": {"name": "RIJEKA"}},
            "transport": {"vesselName": "AL MURABBA", "voyageNumber": "521E"},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=rows,
        source_target=target,
    )

    for row in rows:
        repeats = tuple(
            candidate
            for candidate in normalized
            if candidate.logical_key == row.logical_key
        )
        assert len(repeats) == 2
    validate_target_binding_relationships(drafts=normalized, source_target=target)
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )


def test_structured_locality_rejects_partial_inline_transport_frame() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "VESSEL\nAL MURABBA\n"
        "VOYAGE\n521E\n"
        "VESSEL NAME: AL MURABBA VOYAGE: 521E EXTRA\n"
    )
    vessel_path = "documentPatch.transport.vesselName"
    voyage_path = "documentPatch.transport.voyageNumber"
    vessel = replace(
        _test_draft(raw, "AL MURABBA", "anchor:" + vessel_path, "target_binding", (vessel_path,)),
        value_kind="equipment",
        group_kind="transport",
        group_key="transport",
        render_policy="natural_text",
    )
    voyage = replace(
        _test_draft(raw, "521E", "anchor:" + voyage_path, "target_binding", (voyage_path,)),
        value_kind="identifier",
        group_kind="transport",
        group_key="transport",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "transport": {"vesselName": "AL MURABBA", "voyageNumber": "521E"},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(vessel, voyage),
        source_target=target,
    )

    assert normalized == (vessel, voyage)


def test_structured_locality_appends_party_location_only_in_same_party_context() -> None:
    raw = "--- PAGE 1 ---\nBUSINESS EGYPT\nSENOURAS, FAYOUM, EGYPT\nUNRELATED EGYPT\n"
    name_path = "documentPatch.parties.consignee.name"
    city_path = "documentPatch.parties.consignee.city"
    country_path = "documentPatch.parties.consignee.country"
    name = replace(
        _test_draft(raw, "BUSINESS", "anchor:" + name_path, "target_binding", (name_path,)),
        value_kind="organization",
        group_kind="party",
        group_key="party:consignee:0",
        render_policy="natural_text",
    )
    city = replace(
        _test_draft(raw, "SENOURAS", "anchor:" + city_path, "target_binding", (city_path,)),
        value_kind="location",
        group_kind="party",
        group_key="party:consignee:0",
        render_policy="natural_text",
    )
    country = replace(
        _test_draft(
            raw,
            "EGYPT",
            "anchor:" + country_path,
            "target_binding",
            (country_path,),
        ),
        value_kind="location",
        group_kind="party",
        group_key="party:consignee:0",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": "BUSINESS EGYPT",
                    "city": "SENOURAS",
                    "country": "EGYPT",
                }
            },
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw, drafts=(name, city, country), source_target=target
    )
    country_rows = tuple(row for row in normalized if row.logical_key == country.logical_key)

    assert tuple(row.char_start for row in country_rows) == (
        raw.index("EGYPT"),
        raw.index("EGYPT", raw.index("SENOURAS")),
    )
    assert all(row.char_start != raw.rindex("EGYPT") for row in country_rows)


def test_structured_locality_leaves_partially_owned_labeled_source_location_for_review() -> None:
    raw = "--- PAGE 1 ---\nFOREIGN EXPORTER COUNTRY:\nNORTH MACEDONIA\n"
    existing = replace(
        _test_draft(raw, "NORTH", "agent:partial_country", "deterministic_auxiliary"),
        value_kind="location",
        group_kind="customs",
        group_key="customs:unknown",
        render_policy="natural_text",
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    assert normalize_structured_row_locality(raw=raw, drafts=(existing,), source_target=target) == (
        existing,
    )


def test_structured_locality_unifies_exact_canonical_carrier_repeats_only() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "ORIENT OVERSEAS CONTAINER\nLINE\n"
        "SIGNED OOCL (BENELUX) N.V.\n"
        "Published in OOCL's tariffs\n"
        "--- PAGE 2 ---\n"
        "ORIENT OVERSEAS CONTAINER\nLINE\n"
    )
    path = "documentPatch.parties.carrier.name"
    owner = replace(
        _test_draft(
            raw,
            "ORIENT OVERSEAS CONTAINER\nLINE",
            "anchor:" + path,
            "carrier_static",
            (path,),
        ),
        value_kind="organization",
        group_kind="carrier",
        group_key="party:carrier:0",
        render_policy="natural_text",
    )
    split_repeat = replace(
        _test_draft(
            raw,
            "ORIENT OVERSEAS CONTAINER\nLINE",
            "agent:carrier:split_repeat",
            "carrier_static",
            occurrence=1,
        ),
        value_kind="organization",
        group_kind="carrier",
        group_key="carrier:principal",
        render_policy="natural_text",
    )
    affiliate = replace(
        _test_draft(
            raw,
            "OOCL (BENELUX) N.V.",
            "agent:carrier:affiliate_signature",
            "carrier_static",
        ),
        value_kind="organization",
        group_kind="carrier",
        group_key="carrier:principal",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "ORIENT OVERSEAS CONTAINER LINE"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(owner, split_repeat, affiliate),
        source_target=target,
    )
    carrier_rows = tuple(row for row in normalized if row.logical_key == owner.logical_key)

    assert tuple(row.source_text for row in carrier_rows) == (
        "ORIENT OVERSEAS CONTAINER\nLINE",
        "ORIENT OVERSEAS CONTAINER\nLINE",
    )
    assert all(row.target_paths == (path,) for row in carrier_rows)
    assert split_repeat.logical_key not in {row.logical_key for row in normalized}
    assert affiliate in normalized
    initialism = next(
        row for row in normalized if row.logical_key == "agent:carrier:canonical_initialism:oocl"
    )
    assert initialism.source_text == "OOCL"
    assert initialism.render_mode == "carrier_static"
    assert initialism.char_start == raw.index("OOCL's")
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_target_binding_relationships(drafts=normalized, source_target=target)
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_does_not_infer_carrier_initialism_without_carrier_context() -> None:
    raw = "--- PAGE 1 ---\nORIENT OVERSEAS CONTAINER LINE\nCARGO OOCL CODE\n"
    path = "documentPatch.parties.carrier.name"
    owner = replace(
        _test_draft(
            raw,
            "ORIENT OVERSEAS CONTAINER LINE",
            "anchor:" + path,
            "carrier_static",
            (path,),
        ),
        value_kind="organization",
        group_kind="carrier",
        group_key="party:carrier:0",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "ORIENT OVERSEAS CONTAINER LINE"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    assert normalize_structured_row_locality(raw=raw, drafts=(owner,), source_target=target) == (
        owner,
    )


def test_structured_locality_leaves_overlapped_canonical_carrier_repeat_unchanged() -> None:
    raw = (
        "--- PAGE 1 ---\nORIENT OVERSEAS CONTAINER\nLINE\n"
        "--- PAGE 2 ---\nORIENT OVERSEAS CONTAINER\nLINE\n"
    )
    path = "documentPatch.parties.carrier.name"
    owner = replace(
        _test_draft(
            raw,
            "ORIENT OVERSEAS CONTAINER\nLINE",
            "anchor:" + path,
            "carrier_static",
            (path,),
        ),
        value_kind="organization",
        group_kind="carrier",
        group_key="party:carrier:0",
        render_policy="natural_text",
    )
    literal = replace(
        _test_draft(
            raw,
            "ORIENT OVERSEAS CONTAINER\nLINE",
            "agent:literal:carrier_repeat",
            "literal_static",
            occurrence=1,
        ),
        value_kind="legal_text",
        group_kind="document",
        group_key="document",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "ORIENT OVERSEAS CONTAINER LINE"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    assert normalize_structured_row_locality(
        raw=raw, drafts=(owner, literal), source_target=target
    ) == (owner, literal)


def test_structured_locality_owns_selected_operational_line_grammars() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "21 DAYS FREE DETENTION AT DESTINATION\n"
        "EGYPT IMPORT IS FREE OUT, ALL EXPENSES UNTIL RETURN OF EMPTY CONTAINER "
        "ON BOARD VESSEL ARE FOR RECEIVERS ACCOUNT.\n"
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(raw=raw, drafts=(), source_target=target)
    by_key = {row.logical_key: row for row in normalized}

    detention = by_key["agent:commercial:free_detention"]
    assert detention.render_mode == "deterministic_auxiliary"
    assert detention.source_text == "21 DAYS FREE DETENTION AT DESTINATION"
    instruction = by_key["agent:commercial:import_free_out_instruction"]
    assert instruction.render_mode == "agent_residual"
    assert instruction.source_text.startswith("EGYPT IMPORT IS FREE OUT")
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_does_not_guess_operational_prose_or_partial_ownership() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "ALLOW 21 DAYS FREE DETENTION AT DESTINATION IF AGREED\n"
        "EGYPT IMPORT IS FREE OUT, ALL EXPENSES ARE FOR RECEIVERS ACCOUNT.\n"
    )
    partial = replace(
        _test_draft(raw, "EGYPT", "agent:country", "deterministic_auxiliary"),
        value_kind="location",
        group_kind="route",
        group_key="route:destination",
        render_policy="natural_text",
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    assert normalize_structured_row_locality(raw=raw, drafts=(partial,), source_target=target) == (
        partial,
    )


def test_structured_locality_moves_freight_payment_from_option_header_to_selection() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "CODE TARIFF ITEM FREIGHTED AS RATE PREPAID COLLECT\n"
        "OCEAN FREIGHT PREPAID\n"
    )
    path = "documentPatch.freight.paymentArrangement"
    header = replace(
        _test_draft(raw, "PREPAID", "anchor:" + path, "target_binding", (path,)),
        value_kind="commercial_text",
        group_kind="commercial",
        group_key="commercial:freight",
        render_policy="categorical_surface",
    )
    target = {
        "documentPatch": {
            "freight": {"paymentArrangement": "prepaid"},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(raw=raw, drafts=(header,), source_target=target)

    assert len(normalized) == 2
    selected = next(row for row in normalized if row.render_mode == "target_binding")
    assert selected.logical_key == header.logical_key
    assert selected.source_text == "PREPAID"
    assert selected.char_start == raw.rindex("PREPAID")
    literal = next(row for row in normalized if row.render_mode == "literal_static")
    assert literal.char_start == raw.index("PREPAID")
    assert literal.target_paths == ()
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_narrows_freight_caption_from_direct_target_surface() -> None:
    raw = "--- PAGE 1 ---\nOCEAN FREIGHT PREPAID\n"
    path = "documentPatch.freight.paymentArrangement"
    row = replace(
        _test_draft(raw, "OCEAN FREIGHT PREPAID", "anchor:" + path, "target_binding", (path,)),
        value_kind="commercial_text",
        group_kind="commercial",
        group_key="commercial:freight",
        render_policy="categorical_surface",
    )
    target = {
        "documentPatch": {
            "freight": {"paymentArrangement": "prepaid"},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(raw=raw, drafts=(row,), source_target=target)

    assert len(normalized) == 1
    assert normalized[0].logical_key == row.logical_key
    assert normalized[0].source_text == "PREPAID"
    assert normalized[0].char_start == raw.index("PREPAID")
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_keeps_freight_option_header_without_selected_evidence() -> None:
    raw = "--- PAGE 1 ---\nCODE TARIFF ITEM FREIGHTED AS RATE PREPAID COLLECT\n"
    path = "documentPatch.freight.paymentArrangement"
    header = replace(
        _test_draft(raw, "PREPAID", "anchor:" + path, "target_binding", (path,)),
        value_kind="commercial_text",
        group_kind="commercial",
        group_key="commercial:freight",
        render_policy="categorical_surface",
    )
    target = {
        "documentPatch": {
            "freight": {"paymentArrangement": "prepaid"},
            "containers": [],
            "cargoPackages": [],
        }
    }

    assert normalize_structured_row_locality(raw=raw, drafts=(header,), source_target=target) == (
        header,
    )


def test_structured_locality_removes_unscoped_short_package_quantity_repeats() -> None:
    raw = "--- PAGE 1 ---\n2 PACKAGES\nCLAUSE 2\nREFERENCE 2\n"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    type_path = "documentPatch.cargoPackages[0].typeCategory"
    quantity_rows = tuple(
        replace(
            _test_draft(
                raw,
                "2",
                "anchor:" + quantity_path,
                "target_binding",
                (quantity_path,),
                occurrence=index,
            ),
            value_kind="integer",
            group_kind="package",
            group_key="package:0",
            render_policy="numeric_surface",
        )
        for index in range(3)
    )
    package_type = replace(
        _test_draft(raw, "PACKAGES", "anchor:" + type_path, "target_binding", (type_path,)),
        value_kind="package",
        group_kind="package",
        group_key="package:0",
        render_policy="categorical_surface",
    )
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 2,
                    "typeCategory": "PACKAGE",
                }
            ],
            "containers": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(*quantity_rows, package_type),
        source_target=target,
    )
    retained_quantities = tuple(row for row in normalized if quantity_path in row.target_paths)

    assert len(retained_quantities) == 1
    assert retained_quantities[0].char_start == raw.index("2 PACKAGES")
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_owns_labeled_booking_slac_and_country_clause() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "B/L NO. EGS2024045536\n"
        "BOOKING NO. SEA WAYBILL NO.\n"
        "ABC12345 EGS2024045536\n"
        "1 CONT. 40 REEFER CONTAINER SLAC*\n"
        "1 CONT. 40 REEFER CONTAINER SLAC*\n"
        "COUNTRY: EGYPT\n"
        "EGYPT CLAUSE\n"
    )
    bol_path = "documentPatch.billOfLadingNumber"
    country_path = "documentPatch.parties.shipper.country"
    bol = replace(
        _test_draft(raw, "EGS2024045536", "anchor:" + bol_path, "target_binding", (bol_path,)),
        value_kind="identifier",
        group_kind="document",
        group_key="document",
        render_policy="opaque_identifier",
    )
    country = replace(
        _test_draft(
            raw,
            "EGYPT",
            "anchor:" + country_path,
            "target_binding",
            (country_path,),
        ),
        value_kind="location",
        group_kind="party",
        group_key="party:shipper:0",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "billOfLadingNumber": "EGS2024045536",
            "parties": {"shipper": {"country": "EGYPT"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw, drafts=(bol, country), source_target=target
    )
    by_key: dict[str, list[SpanDraft]] = {}
    for row in normalized:
        by_key.setdefault(row.logical_key, []).append(row)

    assert [row.source_text for row in by_key["agent:commercial:booking_number"]] == ["ABC12345"]
    assert [row.source_text for row in by_key["agent:cargo:slac_qualifier"]] == [
        "SLAC",
        "SLAC",
    ]
    assert [row.source_text for row in by_key["agent:legal:country_clause_heading"]] == [
        "EGYPT CLAUSE"
    ]
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_joins_hash_framed_and_labeled_acid_identifiers() -> None:
    value = "5165256622025050151"
    raw = f"--- PAGE 1 ---\n#{value}#\nACID: {value}\nACID: {value}\n"
    rows = tuple(
        replace(
            _test_draft(
                raw,
                value,
                f"agent:reference:{index}",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="identifier",
            group_kind="document" if index == 0 else "customs",
            group_key="document:reference" if index == 0 else "customs:acid",
            render_policy="opaque_identifier",
        )
        for index in range(3)
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(raw=raw, drafts=rows, source_target=target)

    assert {row.logical_key for row in normalized} == {"agent:customs:acid_reference"}
    assert {row.group_key for row in normalized} == {"customs:acid"}
    assert tuple(row.source_text for row in normalized) == (value, value, value)
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_repairs_repeated_shipper_and_receipt_role_collisions() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "SHIPPER Name, Address, Phone\n"
        "RECTORSEAL LLC\n"
        "2601 SPENWICK DR.\n"
        "HOUSTON, TX 77055\n\n"
        "PLACE OF RECEIPT COMBINED TRANSPORT\n"
        "HOUSTON, TX 77055\n"
        "PLACE OF RECEIPT*COMBINED TRANSPORT\n"
        "HOUSTON, TX 77055\n"
    )
    address_path = "documentPatch.parties.shipper.address"
    city_path = "documentPatch.parties.shipper.city"
    receipt_path = "documentPatch.route.placeOfReceipt.name"
    address = tuple(
        replace(
            _test_draft(
                raw,
                text,
                "anchor:" + address_path,
                "target_binding",
                (address_path,),
            ),
            value_kind="address",
            group_kind="party",
            group_key="party:shipper:0",
            render_policy="natural_text",
        )
        for text in ("2601 SPENWICK DR", "77055")
    )
    city = replace(
        _test_draft(raw, "HOUSTON", "anchor:" + city_path, "target_binding", (city_path,)),
        value_kind="location",
        group_kind="party",
        group_key="party:shipper:0",
        render_policy="natural_text",
    )
    wrong_state = replace(
        _test_draft(raw, "TX", "anchor:" + receipt_path, "target_binding", (receipt_path,)),
        value_kind="location",
        group_kind="route",
        group_key="route:placeOfReceipt",
        render_policy="natural_text",
    )
    wrong_postal = replace(
        address[1],
        draft_id="wrong_route_postal",
        char_start=raw.rindex("77055"),
        char_end=raw.rindex("77055") + len("77055"),
        source_text="77055",
    )
    receipt = replace(
        _test_draft(
            raw,
            "HOUSTON, TX 77055",
            "anchor:" + receipt_path,
            "target_binding",
            (receipt_path,),
            occurrence=1,
        ),
        value_kind="location",
        group_kind="route",
        group_key="route:placeOfReceipt",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": "2601 SPENWICK DR. TX 77055",
                    "city": "HOUSTON",
                }
            },
            "route": {"placeOfReceipt": {"name": "HOUSTON, TX 77055"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(address[0], city, wrong_state, wrong_postal, receipt),
        source_target=target,
    )
    address_rows = tuple(row for row in normalized if row.logical_key == address[0].logical_key)
    receipt_rows = tuple(row for row in normalized if row.logical_key == receipt.logical_key)

    assert tuple(row.source_text for row in address_rows) == ("2601 SPENWICK DR", "TX 77055")
    assert tuple(row.source_text for row in receipt_rows) == (
        "HOUSTON, TX 77055",
        "HOUSTON, TX 77055",
    )
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_derives_country_code_and_owns_export_and_units() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "COUNTRY: UNITED STATES\n"
        "COUNTRY CODE: US\n"
        "THESE COMMODITIES WERE EXPORTED FROM THE UNITED STATES IN ACCORDANCE WITH RULES\n"
        "SEAL: APPLES AS PER KGM\n"
        "3084 PACKAGE KGM\n"
    )
    country = replace(
        _test_draft(
            raw,
            "UNITED STATES",
            "agent:shipment:country_of_origin",
            "deterministic_auxiliary",
        ),
        value_kind="location",
        group_kind="customs",
        group_key="customs:shipment",
        render_policy="natural_text",
    )
    code = replace(
        _test_draft(raw, "US", "agent:shipment:country_code", "deterministic_auxiliary"),
        value_kind="identifier",
        group_kind="customs",
        group_key="customs:shipment",
        render_policy="opaque_identifier",
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(
        raw=raw, drafts=(country, code), source_target=target
    )
    country_rows = tuple(row for row in normalized if row.logical_key == country.logical_key)
    code_row = next(row for row in normalized if row.logical_key == code.logical_key)
    unit_rows = tuple(
        row for row in normalized if row.logical_key == "agent:cargo:measurement_unit:kgm"
    )

    assert len(country_rows) == 2
    assert code_row.render_mode == "deterministic_derived"
    assert code_row.derivation == "country_code"
    assert code_row.dependency_bindings == (country.logical_key,)
    assert tuple(row.source_text for row in unit_rows) == ("KGM", "KGM")
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_joins_existing_detail_and_total_measurement_units() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "SEAL: FRESH APPLES AS PER KGM\n"
        "========================= 43956,000\n"
        "3084 PACKAGE KGM\n"
    )
    rows = tuple(
        replace(
            _test_draft(
                raw,
                "KGM",
                f"agent:weight:{scope}:unit",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="other_text",
            group_kind="cargo",
            group_key=f"weight:{scope}",
            render_policy="natural_text",
        )
        for index, scope in enumerate(("row", "total"))
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(raw=raw, drafts=rows, source_target=target)

    assert {row.logical_key for row in normalized} == {"agent:cargo:measurement_unit:kgm"}
    assert {row.group_key for row in normalized} == {"cargo:measurement_unit"}
    assert tuple(row.source_text for row in normalized) == ("KGM", "KGM")
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_derives_repeated_equipment_multipliers_and_original_marks() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "ORIGINAL BILL OF LADING\n"
        "1 X 20' STD CONTAINER STC:\n"
        "ORIGINAL\n"
        "--- PAGE 2 ---\n"
        "1 X 20' STD CONTAINER STC:\n"
        "ORIGINAL\n"
    )
    type_path = "documentPatch.containers[0].typeDescription"
    type_surface = "20' STD CONTAINER"
    type_starts = tuple(match.start() for match in re.finditer(re.escape(type_surface), raw))
    types = tuple(
        replace(
            _test_draft(raw, type_surface, "anchor:" + type_path, "target_binding"),
            draft_id=f"type_{index}",
            char_start=start,
            char_end=start + len(type_surface),
            source_text=type_surface,
            value_kind="equipment",
            group_kind="equipment",
            group_key="container:0",
            target_paths=(type_path,),
            render_policy="natural_text",
        )
        for index, start in enumerate(type_starts)
    )
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "HLBU1340015",
                    "typeDescription": type_surface,
                }
            ],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=types,
        source_target=target,
    )
    multipliers = tuple(
        row for row in normalized if row.logical_key == "agent:equipment:container_receipt:0"
    )
    original_marks = tuple(
        row for row in normalized if row.logical_key == "agent:document:original_mark"
    )

    assert tuple(row.source_text for row in multipliers) == (
        "1 X 20' STD CONTAINER",
        "1 X 20' STD CONTAINER",
    )
    assert {row.derivation for row in multipliers} == {"equipment_receipt"}
    assert {row.target_paths for row in multipliers} == {("documentPatch.containers[0]",)}
    assert {row.dependency_paths for row in multipliers} == {(type_path,)}
    assert tuple(row.source_text for row in original_marks) == ("ORIGINAL", "ORIGINAL")
    assert {row.render_mode for row in original_marks} == {"deterministic_auxiliary"}
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_expands_existing_type_only_equipment_receipt() -> None:
    raw = "--- PAGE 1 ---\n1 X 20' STD CONTAINER\n1 X 20' STD CONTAINER\n"
    container_path = "documentPatch.containers[0]"
    type_path = container_path + ".typeDescription"
    surface = "20' STD CONTAINER"
    receipts = tuple(
        replace(
            _test_draft(
                raw,
                surface,
                "agent:compiler_equipment_receipt",
                "deterministic_derived",
                (container_path,),
                (type_path,),
                occurrence=index,
            ),
            draft_id=f"compiler_receipt_{index}",
            value_kind="equipment",
            group_kind="equipment",
            group_key="container:0",
            derivation="equipment_receipt",
            render_policy="derived_surface",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": surface}],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=receipts,
        source_target=target,
    )

    assert tuple(row.source_text for row in normalized) == (
        "1 X 20' STD CONTAINER",
        "1 X 20' STD CONTAINER",
    )
    assert {row.logical_key for row in normalized} == {"agent:equipment:container_receipt:0"}
    assert {row.target_paths for row in normalized} == {(container_path,)}
    assert {row.dependency_paths for row in normalized} == {(type_path,)}
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )


def test_structured_locality_owns_abbreviated_counts_registration_type_and_package_nouns() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "1 CONT. 40'X9'6\" REEFER CONTAINER\n"
        "1542 PACKAGE\n"
        "FOREIGN EXPORTER REGISTRATION\n"
        "TYPE: VAT NUMBER\n"
        "--- PAGE 2 ---\n"
        "1 CONT. 40'X9'6\" REEFER CONTAINER\n"
        "1542 PACKAGE\n"
        "FOREIGN EXPORTER REGISTRATION\n"
        "TYPE: VAT NUMBER\n"
        "3084 PACKAGE KGM\n"
    )
    container_path = "documentPatch.containers[0]"
    type_path = container_path + ".typeDescription"
    type_surface = "40'X9'6\" REEFER CONTAINER"
    type_rows = tuple(
        replace(
            _test_draft(
                raw,
                type_surface,
                "anchor:" + type_path,
                "target_binding",
                (type_path,),
                occurrence=index,
            ),
            draft_id=f"type_{index}",
            value_kind="equipment",
            group_kind="equipment",
            group_key="container:0",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    quantities = tuple(
        replace(
            _test_draft(
                raw,
                "1542",
                "agent:package_quantity",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            draft_id=f"quantity_{index}",
            value_kind="integer",
            group_kind="package",
            group_key="package:source",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    total = replace(
        _test_draft(raw, "3084", "agent:package_total", "deterministic_derived"),
        value_kind="integer",
        group_kind="package",
        group_key="package:total",
        target_paths=(),
        derivation="sum_package_quantity",
        dependency_paths=(),
        dependency_bindings=("agent:package_quantity", "agent:package_quantity"),
        render_policy="derived_surface",
    )
    registration_types = tuple(
        replace(
            _test_draft(
                raw,
                "VAT NUMBER",
                "agent:literal:vat_number",
                "literal_static",
                occurrence=index,
            ),
            draft_id=f"registration_type_{index}",
            value_kind="legal_text",
            group_kind="customs",
            group_key="customs:registration_label",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": type_surface}],
            "cargoPackages": [],
        }
    }

    structured = normalize_structured_row_locality(
        raw=raw,
        drafts=(*type_rows, *quantities, total, *registration_types),
        source_target=target,
    )
    normalized = normalize_deterministic_draft_semantics(
        drafts=structured,
        source_target=target,
    )

    counts = tuple(row for row in normalized if row.derivation == "container_count")
    nouns = tuple(row for row in normalized if row.value_kind == "package")
    selected_types = tuple(
        row
        for row in normalized
        if row.logical_key == "agent:customs:foreign_exporter_registration_type"
    )
    normalized_total = next(row for row in normalized if row.logical_key == "agent:package_total")
    assert tuple(row.source_text for row in counts) == ("1 CONT.", "1 CONT.")
    assert tuple(row.source_text for row in nouns) == ("PACKAGE", "PACKAGE", "PACKAGE")
    assert tuple(row.source_text for row in selected_types) == ("VAT NUMBER", "VAT NUMBER")
    assert {row.render_mode for row in selected_types} == {"deterministic_auxiliary"}
    assert normalized_total.render_mode == "deterministic_auxiliary"
    assert normalized_total.derivation is None
    assert (
        normalize_deterministic_draft_semantics(
            drafts=normalize_structured_row_locality(
                raw=raw,
                drafts=normalized,
                source_target=target,
            ),
            source_target=target,
        )
        == normalized
    )


def test_structured_locality_relocates_unique_un_class_and_drops_digit_repeats() -> None:
    raw = "--- PAGE 1 ---\nUN 2857 CL 2.2\nNOTIFY 2: ACME\nin timely delivery (Clause 6.2.)\n"
    hazard_path = "documentPatch.cargoGroups[0].dangerousGoods[0].hazardCategory"
    class_start = raw.index("2.2")
    notify_start = raw.index("2: ACME")
    clause_start = raw.rindex("2")
    base = replace(
        _test_draft(raw, "2.2", "anchor:" + hazard_path, "target_binding", (hazard_path,)),
        draft_id="hazard_class_digit",
        char_start=class_start,
        char_end=class_start + 1,
        source_text="2",
        value_kind="dangerous_goods",
        group_kind="dangerous_goods",
        group_key="dangerous_goods:0:0",
        render_policy="natural_text",
    )
    drafts = (
        base,
        replace(
            base,
            draft_id="hazard_notify_digit",
            char_start=notify_start,
            char_end=notify_start + 1,
        ),
        replace(
            base,
            draft_id="hazard_clause_digit",
            char_start=clause_start,
            char_end=clause_start + 1,
        ),
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "cargoGroups": [
                {
                    "dangerousGoods": [
                        {
                            "hazardCategory": "GASES",
                            "unNumber": "2857",
                        }
                    ]
                }
            ],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=drafts,
        source_target=target,
    )
    hazard = tuple(row for row in normalized if hazard_path in row.target_paths)

    assert len(hazard) == 1
    assert hazard[0].source_text == "2.2"
    assert hazard[0].render_mode == "agent_residual"
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_owns_unique_un_labeled_description_suffix() -> None:
    description = "ENVIRONMENTALLY HAZARDOUS SUBSTANCE, SOLID, N.O.S."
    raw = f"--- PAGE 1 ---\nUN 3077 {description}\n"
    un_path = "documentPatch.cargoGroups[0].dangerousGoods[0].unNumber"
    un_owner = replace(
        _test_draft(raw, "3077", "anchor:" + un_path, "target_binding", (un_path,)),
        value_kind="identifier",
        group_kind="dangerous_goods",
        group_key="dangerous_goods:0:0",
        render_policy="opaque_identifier",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "cargoGroups": [{"dangerousGoods": [{"unNumber": "3077"}]}],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(un_owner,),
        source_target=target,
    )
    descriptions = tuple(
        row for row in normalized if row.logical_key == "agent:dangerous_goods:0:0:description"
    )

    assert len(descriptions) == 1
    assert descriptions[0].source_text == description
    assert descriptions[0].render_mode == "deterministic_auxiliary"
    assert descriptions[0].value_kind == "dangerous_goods"
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_does_not_misclassify_un_class_suffix_as_description() -> None:
    raw = "--- PAGE 1 ---\nUN 3077 CLASS 9\n"
    un_path = "documentPatch.cargoGroups[0].dangerousGoods[0].unNumber"
    un_owner = replace(
        _test_draft(raw, "3077", "anchor:" + un_path, "target_binding", (un_path,)),
        value_kind="identifier",
        group_kind="dangerous_goods",
        group_key="dangerous_goods:0:0",
        render_policy="opaque_identifier",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "cargoGroups": [{"dangerousGoods": [{"unNumber": "3077"}]}],
        }
    }

    assert normalize_structured_row_locality(
        raw=raw,
        drafts=(un_owner,),
        source_target=target,
    ) == (un_owner,)


def test_structured_locality_derives_labeled_code_from_same_party_country_target() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "SHIPPER\n"
        "ACME SP ZOO\n"
        "WARSZAWA POLAND\n"
        "COUNTRY CODE: PL\n"
        "COUNTRY CODE: PL\n"
    )
    country_path = "documentPatch.parties.shipper.country"
    country = replace(
        _test_draft(
            raw,
            "POLAND",
            "anchor:" + country_path,
            "target_binding",
            (country_path,),
        ),
        value_kind="location",
        group_kind="party",
        group_key="party:shipper:0",
        render_policy="natural_text",
    )
    codes = tuple(
        replace(
            _test_draft(
                raw,
                "PL",
                "agent:shipper_country_code",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="location",
            group_kind="party",
            group_key="party:shipper:0",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "parties": {"shipper": {"country": "POLAND"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw, drafts=(country, *codes), source_target=target
    )
    code_rows = tuple(row for row in normalized if row.logical_key == "agent:shipper_country_code")

    assert len(code_rows) == 2
    assert {row.render_mode for row in code_rows} == {"deterministic_derived"}
    assert {row.derivation for row in code_rows} == {"country_code"}
    assert {row.dependency_paths for row in code_rows} == {(country_path,)}
    assert {row.dependency_bindings for row in code_rows} == {()}
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_source_boundaries_trim_repeated_importer_name_captions() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "EGYPTIAN IMPORTER NAME: EGYPTION CHEMICALS EAMIC\n"
        "EGYPTIAN IMPORTER NAME: EGYPTION CHEMICALS EAMIC\n"
    )
    drafts = tuple(
        replace(
            _test_draft(
                raw,
                "EGYPTIAN IMPORTER NAME: EGYPTION CHEMICALS EAMIC",
                "agent:shipment_reference:importer_name",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="organization",
            group_kind="party",
            group_key="party:importer:0",
            render_policy="natural_text",
        )
        for index in range(2)
    )

    normalized = normalize_source_boundaries(raw=raw, drafts=drafts)

    assert tuple(row.source_text for row in normalized) == (
        "EGYPTION CHEMICALS EAMIC",
        "EGYPTION CHEMICALS EAMIC",
    )
    assert normalize_source_boundaries(raw=raw, drafts=normalized) == normalized


def test_structured_locality_appends_only_framed_movement_type_repeat() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "TYPE OF MOVEMENT (IF MIXED)\n"
        "FCL / FCL\n"
        "96 CARTONS /FCL/FCL/ /40HQ/\n"
        "96 CARTONS /FCL/FCL/ /40HQ/\n"
        "96 CARTONS /FCL/FCL/ /40HQ/\n"
        "NOTE /FCL/FCL/\n"
    )
    movement_key = "agent:movement_type"
    movement_rows = (
        replace(
            _test_draft(raw, "FCL / FCL", movement_key, "deterministic_auxiliary"),
            value_kind="operational_text",
            group_kind="transport",
            group_key="transport:movement_type",
            render_policy="natural_text",
        ),
        *tuple(
            replace(
                _test_draft(
                    raw,
                    "FCL/FCL",
                    movement_key,
                    "deterministic_auxiliary",
                    occurrence=index,
                ),
                value_kind="operational_text",
                group_kind="transport",
                group_key="transport:movement_type",
                render_policy="natural_text",
            )
            for index in range(2)
        ),
    )
    package_rows = tuple(
        replace(
            _test_draft(
                raw,
                "96 CARTONS",
                f"agent:package:{index}",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="package",
            group_kind="package",
            group_key=f"package:{index}",
            render_policy="natural_text",
        )
        for index in range(3)
    )
    equipment_rows = tuple(
        _test_draft(
            raw,
            "40HQ",
            f"agent:equipment:{index}",
            "deterministic_auxiliary",
            group_key=f"container:{index}",
            occurrence=index,
        )
        for index in range(3)
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(*movement_rows, *package_rows, *equipment_rows),
        source_target=target,
    )
    normalized_movement = tuple(row for row in normalized if row.logical_key == movement_key)
    note_start = raw.index("FCL/FCL", raw.index("NOTE"))

    assert len(normalized_movement) == 4
    assert {row.char_start for row in normalized_movement} == {
        movement_rows[0].char_start,
        *(
            raw.index("FCL/FCL", offset)
            for offset in (
                raw.index("96 CARTONS"),
                raw.index("96 CARTONS", raw.index("96 CARTONS") + 1),
                raw.rindex("96 CARTONS"),
            )
        ),
    }
    assert all(row.char_start != note_start for row in normalized_movement)
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )


def test_structured_locality_recovers_unowned_labeled_movement_value() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "TYPE OF MOVEMENT (IF MIXED)\n"
        "FCL / FCL\n"
        "96 CARTONS /FCL/FCL/ /40HQ/\n"
        "96 CARTONS /FCL/FCL/ /40HQ/\n"
        "96 CARTONS /FCL/FCL/ /40HQ/\n"
    )
    movement_key = "agent:movement_type"
    movement_rows = tuple(
        replace(
            _test_draft(
                raw,
                "FCL/FCL",
                movement_key,
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="operational_text",
            group_kind="transport",
            group_key="transport:movement_type",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    package_rows = tuple(
        replace(
            _test_draft(
                raw,
                "96 CARTONS",
                f"agent:package:{index}",
                "deterministic_auxiliary",
                occurrence=index,
            ),
            value_kind="package",
            group_kind="package",
            group_key=f"package:{index}",
            render_policy="natural_text",
        )
        for index in range(3)
    )
    equipment_rows = tuple(
        _test_draft(
            raw,
            "40HQ",
            f"agent:equipment:{index}",
            "deterministic_auxiliary",
            group_key=f"container:{index}",
            occurrence=index,
        )
        for index in range(3)
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(*movement_rows, *package_rows, *equipment_rows),
        source_target=target,
    )
    rows = tuple(row for row in normalized if row.logical_key == movement_key)

    assert len(rows) == 4
    assert {row.source_text for row in rows} == {"FCL / FCL", "FCL/FCL"}
    assert rows[0].char_start == raw.index("FCL / FCL")


def test_structured_locality_does_not_drop_sole_loading_port_target() -> None:
    raw = "--- PAGE 1 ---\nLOADING PIER/TERMINAL\nANTWERP\n"
    path = "documentPatch.route.portOfLoading.name"
    port = replace(
        _test_draft(
            raw,
            "ANTWERP",
            "anchor:" + path,
            "target_binding",
            (path,),
            group_key="route:portOfLoading",
        ),
        value_kind="location",
        group_kind="route",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "route": {"portOfLoading": {"name": "ANTWERP"}},
            "containers": [],
            "cargoPackages": [],
        }
    }

    assert normalize_structured_row_locality(raw=raw, drafts=(port,), source_target=target) == (
        port,
    )


def test_structured_locality_owns_populated_shipped_on_board_summary() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "DATE LADEN ON BOARD\n"
        "6 MAR 2024\n"
        "PORT OF LOADING\n"
        "ANTWERP\n"
        "SHIPPED ON BOARD SEASPAN NEW DELHI 011S AT ANTWERP ON 6 MAR 2024\n"
    )
    date_path = "documentPatch.shippedOnBoardDate"
    vessel_path = "documentPatch.transport.vesselName"
    voyage_path = "documentPatch.transport.voyageNumber"
    port_path = "documentPatch.route.portOfLoading.name"

    def target_row(
        text: str,
        path: str,
        *,
        occurrence: int = 0,
        value_kind: str = "location",
        group_kind: str = "transport",
        group_key: str = "transport",
    ) -> SpanDraft:
        return replace(
            _test_draft(
                raw,
                text,
                "anchor:" + path,
                "target_binding",
                (path,),
                group_key=group_key,
                occurrence=occurrence,
            ),
            value_kind=value_kind,
            group_kind=group_kind,
            render_policy="natural_text",
        )

    date_rows = tuple(
        target_row("6 MAR 2024", date_path, occurrence=index, value_kind="date")
        for index in range(2)
    )
    port = target_row(
        "ANTWERP",
        port_path,
        group_kind="route",
        group_key="route:portOfLoading",
    )
    vessel = target_row("SEASPAN NEW DELHI", vessel_path, value_kind="equipment")
    voyage = target_row("011S", voyage_path, value_kind="identifier")
    literal_status = replace(
        _test_draft(
            raw,
            "SHIPPED ON BOARD",
            "agent:literal:shipped_on_board",
            "literal_static",
        ),
        value_kind="legal_text",
        group_kind="document",
        group_key="document",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "shippedOnBoardDate": "2024-03-06",
            "route": {"portOfLoading": {"name": "ANTWERP"}},
            "transport": {"vesselName": "SEASPAN NEW DELHI", "voyageNumber": "011S"},
            "containers": [],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(*date_rows, port, vessel, voyage, literal_status),
        source_target=target,
    )
    status_rows = tuple(
        row for row in normalized if row.logical_key == "agent:operational:shipped_on_board_status"
    )
    normalized_ports = tuple(row for row in normalized if row.logical_key == port.logical_key)

    assert len(status_rows) == 1
    assert status_rows[0].render_mode == "deterministic_auxiliary"
    assert tuple(row.source_text for row in normalized_ports) == ("ANTWERP", "ANTWERP")
    assert literal_status not in normalized
    assert (
        normalize_structured_row_locality(raw=raw, drafts=normalized, source_target=target)
        == normalized
    )
    validate_target_binding_relationships(drafts=normalized, source_target=target)
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_keeps_unpopulated_shipped_on_board_caption_literal() -> None:
    raw = "--- PAGE 1 ---\nSHIPPED ON BOARD\n6 MAR 2024\n"
    literal = replace(
        _test_draft(
            raw,
            "SHIPPED ON BOARD",
            "agent:literal:shipped_on_board",
            "literal_static",
        ),
        value_kind="legal_text",
        group_kind="document",
        group_key="document",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "shippedOnBoardDate": "2024-03-06",
            "containers": [],
            "cargoPackages": [],
        }
    }

    assert normalize_structured_row_locality(raw=raw, drafts=(literal,), source_target=target) == (
        literal,
    )


def test_labeled_organization_typo_recovers_only_exact_declared_value() -> None:
    raw = "--- PAGE 1 ---\nEGYPTIAN IMPORTER NAME: EGYPTION CHEMICALS EAMIC\n"
    proposal = _proposal(
        "EGYPTIAN CHEMICALS EAMIC",
        "L00002",
        key="importer_name",
    ).model_copy(
        update={
            "value_kind": "organization",
            "group_kind": "party",
            "group_key": "party:importer:0",
        }
    )

    resolved = resolve_agent_proposals(raw=raw, proposals=(proposal,))

    assert len(resolved) == 1
    assert resolved[0].source_text == "EGYPTION CHEMICALS EAMIC"
    assert "labeled organization transcription error" in resolved[0].rationale

    invalid = proposal.model_copy(
        update={
            "occurrences": (
                proposal.occurrences[0].model_copy(update={"source_text": "ACME CHEMICALS EAMIC"}),
            )
        }
    )
    with pytest.raises(ValueError, match="cannot be resolved unambiguously"):
        resolve_agent_proposals(raw=raw, proposals=(invalid,))


def test_temperature_value_leaf_surface_expands_to_value_unit_derivation() -> None:
    raw = "--- PAGE 1 ---\nTEMPERATURE TO BE SET AT +1,0 C\n"
    base = "documentPatch.containers[0].temperatureSetpoint"
    value_path = base + ".value"
    target = {
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    draft = replace(
        _test_draft(raw, "+1,0 C", "anchor:" + value_path, "target_binding", (value_path,)),
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


def test_nested_single_token_auxiliary_yields_to_complete_mutable_surface() -> None:
    raw = "--- PAGE 1 ---\nPHONE +20 2 24558888\n"
    path = "documentPatch.parties.notifyParties[0].contactDetails.phoneNumbers[0]"
    phone = replace(
        _test_draft(raw, "+20 2 24558888", "anchor:" + path, "target_binding", (path,)),
        value_kind="phone",
        group_kind="party",
        group_key="party:notify:0",
        render_policy="natural_text",
    )
    single_start = raw.index("+20 2") + len("+20 ")
    single = replace(
        _test_draft(raw, "2", "agent:commercial:freight_payable_at", "deterministic_auxiliary"),
        draft_id="single_nested_auxiliary",
        char_start=single_start,
        char_end=single_start + 1,
        source_text="2",
        value_kind="commercial_text",
        group_kind="commercial",
        group_key="commercial:freight",
        render_policy="natural_text",
    )

    reconciled = reconcile_draft_overlaps(
        raw=raw,
        drafts=(phone, single),
        source_target={"documentPatch": {}},
    )
    assert len(reconciled) == 1
    assert reconciled[0].draft_id == phone.draft_id
    assert "rejected malformed sub-token auxiliary" in reconciled[0].rationale


def test_deterministic_auxiliary_clause_trims_owned_leading_auxiliary() -> None:
    raw = "--- PAGE 1 ---\nEGYPT IMPORT IS FREE OUT, ALL EXPENSES RECEIVERS ACCOUNT.\n"
    outer = replace(
        _test_draft(
            raw,
            "EGYPT IMPORT IS FREE OUT, ALL EXPENSES RECEIVERS ACCOUNT.",
            "agent:operational:import_instruction",
            "deterministic_auxiliary",
        ),
        value_kind="operational_text",
        group_kind="transport",
        group_key="transport:destination_terms",
        render_policy="natural_text",
    )
    country = replace(
        _test_draft(raw, "EGYPT", "agent:route:country", "deterministic_auxiliary"),
        value_kind="location",
        group_kind="route",
        group_key="route:destination_clause",
        render_policy="natural_text",
    )

    normalized = reconcile_draft_overlaps(
        raw=raw,
        drafts=(country, outer),
        source_target={"documentPatch": {}},
    )
    by_key = {row.logical_key: row for row in normalized}

    assert by_key[country.logical_key].source_text == "EGYPT"
    assert by_key[outer.logical_key].source_text.startswith("IMPORT IS FREE OUT")


def test_override_without_replacement_retains_accepted_required_cobinding() -> None:
    raw = "--- PAGE 1 ---\n96 CARTONS\n"
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    anchor = replace(
        _test_draft(
            raw,
            "96",
            "anchor:" + allocation_path + "|" + quantity_path,
            "target_binding",
            (allocation_path, quantity_path),
        ),
        draft_id="anchor_binding_0001",
        value_kind="integer",
        group_kind="package",
        group_key="package:0",
        evidence_origin="accepted_label_evidence",
        render_policy="numeric_surface",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1"],
                    "allocations": [{"packageId": "p1", "packageQuantity": 96}],
                }
            ],
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 96}],
        }
    }

    retained = apply_anchor_overrides(
        raw=raw,
        source_target=target,
        anchors=(anchor,),
        proposed=(),
        overrides=(
            AnchorOverride.model_validate(
                {
                    "anchor_binding_id": anchor.draft_id,
                    "rationale": "No replacement was supplied.",
                }
            ),
        ),
    )

    assert len(retained) == 1
    assert retained[0].draft_id == anchor.draft_id
    assert "retained this accepted required co-binding" in retained[0].rationale


def test_structured_locality_appends_quantity_beside_same_package_type() -> None:
    raw = "--- PAGE 1 ---\n96 CARTONS\n96 CARTONS /FCL/FCL/\n"
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    type_path = "documentPatch.cargoPackages[0].typeCategory"
    quantity = replace(
        _test_draft(raw, "96", "anchor:quantity", "target_binding", occurrence=0),
        value_kind="integer",
        group_kind="package",
        group_key="package:0",
        target_paths=(allocation_path, quantity_path),
        render_policy="numeric_surface",
    )
    types = tuple(
        replace(
            _test_draft(raw, "CARTONS", "anchor:" + type_path, "target_binding", occurrence=index),
            draft_id=f"package_type_{index}",
            value_kind="package",
            group_kind="package",
            group_key="package:0",
            target_paths=(type_path,),
            render_policy="categorical_surface",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1"],
                    "allocations": [{"packageId": "p1", "packageQuantity": 96}],
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(quantity, *types),
        source_target=target,
    )
    quantities = tuple(row for row in normalized if quantity_path in row.target_paths)

    assert tuple(row.source_text for row in quantities) == ("96", "96")
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )


def test_structured_locality_appends_party_city_beside_scoped_postal_identifier() -> None:
    raw = "--- PAGE 1 ---\nFREE ZONE - SHEBIN EL KOM\nSHEBIN EL KOM 32111-22\n"
    city_path = "documentPatch.parties.notifyParties[0].city"
    city = replace(
        _test_draft(raw, "SHEBIN EL KOM", "anchor:" + city_path, "target_binding", (city_path,)),
        value_kind="location",
        group_kind="party",
        group_key="party:notify:0",
        render_policy="natural_text",
    )
    postal = replace(
        _test_draft(raw, "32111-22", "agent:notify:postal", "deterministic_auxiliary"),
        value_kind="identifier",
        group_kind="party",
        group_key="party:notify:0",
        render_policy="opaque_identifier",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "parties": {"notifyParties": [{"city": "SHEBIN EL KOM"}]},
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(city, postal),
        source_target=target,
    )
    cities = tuple(row for row in normalized if row.target_paths == (city_path,))

    assert tuple(row.source_text for row in cities) == ("SHEBIN EL KOM", "SHEBIN EL KOM")


def test_structured_locality_separates_port_charge_from_freight_payment() -> None:
    raw = "--- PAGE 1 ---\nORIGIN PORT CHARGE PREPAID\nSEA FREIGHT PREPAID\n"
    path = "documentPatch.freight.paymentArrangement"
    payments = tuple(
        replace(
            _test_draft(
                raw, "PREPAID", "anchor:" + path, "target_binding", (path,), occurrence=index
            ),
            draft_id=f"freight_payment_{index}",
            value_kind="commercial_text",
            group_kind="commercial",
            group_key="commercial:freight",
            render_policy="natural_text",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "freight": {"paymentArrangement": "prepaid"},
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=payments,
        source_target=target,
    )
    target_rows = tuple(row for row in normalized if row.target_paths == (path,))
    port_rows = tuple(
        row
        for row in normalized
        if row.logical_key == "agent:commercial:origin_port_charge_payment"
    )

    assert len(target_rows) == 1
    assert target_rows[0].char_start == raw.rindex("PREPAID")
    assert len(port_rows) == 1
    assert port_rows[0].char_start == raw.index("PREPAID")
    assert port_rows[0].render_mode == "deterministic_auxiliary"


def test_structured_locality_canonicalizes_multitarget_type_only_equipment_receipt() -> None:
    raw = "--- PAGE 1 ---\n1 X 40' REEFER CONTAINER\n"
    container_path = "documentPatch.containers[0]"
    type_path = container_path + ".typeDescription"
    receipt = replace(
        _test_draft(
            raw,
            "40' REEFER CONTAINER",
            "agent:compiler:equipment_receipt",
            "deterministic_derived",
            (container_path, type_path),
            ("documentPatch.containers", container_path, type_path),
        ),
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
        render_policy="derived_surface",
    )
    target = {
        "documentPatch": {
            "containers": [{"typeDescription": "40' REEFER CONTAINER"}],
            "cargoPackages": [],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(receipt,),
        source_target=target,
    )

    assert len(normalized) == 1
    assert normalized[0].source_text == "1 X 40' REEFER CONTAINER"
    assert normalized[0].target_paths == (container_path,)
    assert normalized[0].dependency_paths == (type_path,)


def test_temperature_separate_value_and_unit_slots_remain_independent() -> None:
    raw = "--- PAGE 1 ---\nTEMP +1 C\nTEMP +1,0 C\n"
    base = "documentPatch.containers[0].temperatureSetpoint"
    value_path = base + ".value"
    unit_path = base + ".unit"
    values = tuple(
        replace(
            _test_draft(raw, text, "anchor:" + value_path, "target_binding"),
            draft_id=f"temperature_value_{index}",
            value_kind="temperature",
            group_kind="temperature",
            group_key="container:0",
            render_policy="numeric_surface",
        )
        for index, text in enumerate(("+1", "+1,0"))
    )
    units = tuple(
        replace(
            _test_draft(raw, "C", "anchor:" + unit_path, "target_binding", occurrence=index),
            draft_id=f"temperature_unit_{index}",
            value_kind="temperature",
            group_kind="temperature",
            group_key="container:0",
            target_paths=(unit_path,),
            render_policy="natural_text",
        )
        for index in range(2)
    )
    values = tuple(replace(row, target_paths=(value_path,)) for row in values)
    target = {
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}],
            "cargoPackages": [],
            "cargoAllocationGroups": [],
        }
    }

    normalized = normalize_deterministic_draft_semantics(
        drafts=(*values, *units),
        source_target=target,
    )

    assert {row.target_paths for row in normalized} == {(value_path,), (unit_path,)}
    assert {row.render_mode for row in normalized} == {"target_binding"}
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_drops_literal_slot_inside_synthetic_page_header() -> None:
    raw = "--- PAGE 1 ---\nBODY\n"
    marker = replace(
        _test_draft(raw, "1", "agent:literal:page_marker:1", "literal_static"),
        value_kind="identifier",
        group_kind="document",
        group_key="document",
        render_policy="opaque_identifier",
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=(marker,),
            source_target=target,
        )
        == ()
    )


def test_structured_locality_completes_adjacent_inline_address_target_tokens() -> None:
    raw = "--- PAGE 1 ---\nPLOT 63 B INDUSTRIAL ZONE A6\nAL SHARQIA, 44629\n"
    path = "documentPatch.parties.consignee.address"
    fragments = (
        replace(
            _test_draft(
                raw,
                "PLOT 63 B INDUSTRIAL ZONE A6",
                "anchor:" + path,
                "target_binding",
                (path,),
            ),
            draft_id="address_prefix",
            value_kind="address",
            group_kind="party",
            group_key="party:consignee:0",
            render_policy="natural_text",
        ),
        replace(
            _test_draft(raw, "44629", "anchor:" + path, "target_binding", (path,)),
            draft_id="address_postal",
            value_kind="address",
            group_kind="party",
            group_key="party:consignee:0",
            render_policy="natural_text",
        ),
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "parties": {"consignee": {"address": "PLOT 63 B INDUSTRIAL ZONE A6 AL SHARQIA, 44629"}},
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=fragments,
        source_target=target,
    )
    address_rows = tuple(row for row in normalized if row.target_paths == (path,))

    assert {row.source_text for row in address_rows} == {
        "PLOT 63 B INDUSTRIAL ZONE A6",
        "AL SHARQIA",
        "44629",
    }
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )


def test_structured_locality_owns_repeated_city_country_party_variant() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "TEL 202 2268 3858\n"
        "CAIRO, EG\n"
        "FAX 202 2268 3850\n"
        "--- PAGE 2 ---\n"
        "TEL 202 2268 3858\n"
        "CAIRO, EG\n"
        "FAX 202 2268 3850\n"
    )
    contexts = tuple(
        replace(
            _test_draft(
                raw,
                text,
                f"agent:local_agent:{kind}",
                "deterministic_auxiliary",
                occurrence=occurrence,
            ),
            draft_id=f"{kind}_{occurrence}",
            value_kind="phone",
            group_kind="party",
            group_key="party:local_agent:0",
            render_policy="natural_text",
        )
        for occurrence in range(2)
        for kind, text in (("phone", "202 2268 3858"), ("fax", "202 2268 3850"))
    )
    target = {"documentPatch": {"containers": [], "cargoPackages": []}}

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=contexts,
        source_target=target,
    )
    locations = tuple(row for row in normalized if row.source_text == "CAIRO, EG")

    assert len(locations) == 2
    assert len({row.logical_key for row in locations}) == 1
    assert {row.group_key for row in locations} == {"party:local_agent:0"}
    assert {row.render_mode for row in locations} == {"deterministic_auxiliary"}


def test_structured_locality_separates_generic_date_from_shipped_on_board_date() -> None:
    raw = "--- PAGE 1 ---\nDATE LADEN ON BOARD\n6 MAR 2024\nDATE\n6 MAR 2024\n"
    path = "documentPatch.shippedOnBoardDate"
    dates = tuple(
        replace(
            _test_draft(raw, "6 MAR 2024", "anchor:" + path, "target_binding", occurrence=index),
            draft_id=f"shipped_date_{index}",
            value_kind="date",
            group_kind="document",
            group_key="document:shipped_on_board",
            target_paths=(path,),
            render_policy="date_surface",
        )
        for index in range(2)
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "shippedOnBoardDate": "2024-03-06",
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=dates,
        source_target=target,
    )
    shipped = tuple(row for row in normalized if row.target_paths == (path,))
    generic = tuple(row for row in normalized if row.logical_key == "agent:document:generic_date")

    assert len(shipped) == 1
    assert shipped[0].char_start == raw.index("6 MAR 2024")
    assert len(generic) == 1
    assert generic[0].char_start == raw.rindex("6 MAR 2024")


def test_structured_locality_appends_exact_repeated_forwarding_reference() -> None:
    raw = (
        "--- PAGE 1 ---\nAES ITN X20240424508768\n"
        "--- PAGE 2 ---\nAES ITN X20240424508768\n"
        "--- PAGE 3 ---\nAES ITN X20240424508768\n"
    )
    path = "documentPatch.forwardingAndExportReferences[0]"
    owner = replace(
        _test_draft(
            raw,
            "X20240424508768",
            "anchor:" + path,
            "target_binding",
            (path,),
        ),
        value_kind="other_text",
        group_kind="document",
        group_key="document",
        render_policy="natural_text",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "forwardingAndExportReferences": ["X20240424508768"],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(owner,),
        source_target=target,
    )
    references = tuple(row for row in normalized if row.target_paths == (path,))

    assert len(references) == 3
    assert {row.source_text for row in references} == {"X20240424508768"}
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )


def test_structured_locality_owns_package_row_linked_by_container_type() -> None:
    raw = "--- PAGE 1 ---\n96 CARTONS /FCL/FCL /40HQ/\n"
    container_type_path = "documentPatch.containers[0].typeDescription"
    type_owner = replace(
        _test_draft(
            raw,
            "40HQ",
            "anchor:" + container_type_path,
            "target_binding",
            (container_type_path,),
        ),
        value_kind="equipment",
        group_kind="equipment",
        group_key="container:0",
        render_policy="natural_text",
    )
    allocation_quantity = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    package_quantity = "documentPatch.cargoPackages[0].quantity"
    package_type = "documentPatch.cargoPackages[0].typeCategory"
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "TCLU1610520", "typeDescription": "1X40'HQ CONTAINER"}
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1"],
                    "allocations": [
                        {
                            "containerNumber": "TCLU1610520",
                            "packageId": "p1",
                            "packageQuantity": 96,
                        }
                    ],
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(type_owner,),
        source_target=target,
    )

    assert any(
        row.source_text == "96" and row.target_paths == (allocation_quantity, package_quantity)
        for row in normalized
    )
    assert any(
        row.source_text == "CARTONS" and row.target_paths == (package_type,) for row in normalized
    )


def test_structured_locality_assigns_laden_on_board_caption_to_target_date() -> None:
    raw = "--- PAGE 1 ---\nDATE LADEN ON BOARD o\n6 MAR 2024\nSHIPPED ON BOARD\n6 MAR 2024\n"
    path = "documentPatch.shippedOnBoardDate"
    auxiliary = replace(
        _test_draft(raw, "6 MAR 2024", "agent:laden_date", "deterministic_auxiliary"),
        value_kind="date",
        group_kind="document",
        group_key="document",
        render_policy="date_surface",
    )
    target_owner = replace(
        _test_draft(
            raw,
            "6 MAR 2024",
            "anchor:" + path,
            "target_binding",
            (path,),
            occurrence=1,
        ),
        value_kind="date",
        group_kind="document",
        group_key="document:shipped_on_board",
        render_policy="date_surface",
    )
    target = {
        "documentPatch": {
            "containers": [],
            "cargoPackages": [],
            "shippedOnBoardDate": "2024-03-06",
        }
    }

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(auxiliary, target_owner),
        source_target=target,
    )
    dates = tuple(row for row in normalized if row.target_paths == (path,))

    assert len(dates) == 2
    assert auxiliary.draft_id not in {row.draft_id for row in normalized}
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )


def test_anchor_drafts_distinguish_pinned_and_host_inferred_relational_evidence() -> None:
    raw = "--- PAGE 1 ---\nPINNED\nINFERRED\n"
    body = raw.split("\n", 1)[1]

    def row(anchor_id: str, text: str, path: str) -> dict[str, object]:
        start = body.index(text)
        return {
            "anchor_id": anchor_id,
            "document_id": "doc_anchor_origin",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(text),
            "raw_value": text,
            "target_value": text,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    pinned_path = "documentPatch.cargoGroups[0].description"
    inferred_path = "documentPatch.cargoGroups[1].description"
    drafts = anchor_drafts(
        raw=raw,
        document_id="doc_anchor_origin",
        anchors=(
            row("anchor_original", "PINNED", pinned_path),
            row("anchor_relational_block_fixture", "INFERRED", inferred_path),
        ),
    )
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "PINNED"},
                {"groupId": "g2", "description": "INFERRED"},
            ]
        }
    }

    assert tuple(row.evidence_origin for row in drafts) == (
        "accepted_label_evidence",
        "host_verified_agent_proposal",
    )
    summaries = anchor_summary(raw, drafts, target)
    assert tuple(row["occurrences"][0]["evidenceOrigin"] for row in summaries) == (
        "accepted_label_evidence",
        "host_verified_agent_proposal",
    )


def test_cargo_description_prefix_reassigns_equal_package_values_by_group_id() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "96 CARTONS SERBIA TOBACCO\n"
        "96 CARTONS OTHER CARGO\n"
    )
    quantity_paths = {
        index: (
            f"documentPatch.cargoAllocationGroups[{index}].allocations[0].packageQuantity",
            f"documentPatch.cargoPackages[{index}].quantity",
        )
        for index in range(2)
    }
    type_paths = {
        index: f"documentPatch.cargoPackages[{index}].typeCategory" for index in range(2)
    }
    description_path = "documentPatch.cargoGroups[0].description"
    description = replace(
        _test_draft(
            raw,
            "SERBIA TOBACCO",
            "anchor:" + description_path,
            "target_binding",
            (description_path,),
        ),
        value_kind="cargo_text",
        group_kind="cargo",
        group_key="cargo:0",
        render_policy="natural_text",
    )

    def quantity(index: int, occurrence: int) -> SpanDraft:
        return replace(
            _test_draft(
                raw,
                "96",
                "anchor:" + "|".join(quantity_paths[index]),
                "target_binding",
                quantity_paths[index],
                occurrence=occurrence,
            ),
            value_kind="integer",
            group_kind="package",
            group_key=f"package:{index}",
            render_policy="numeric_surface",
        )

    def package_type(index: int, occurrence: int) -> SpanDraft:
        return replace(
            _test_draft(
                raw,
                "CARTONS",
                "anchor:" + type_paths[index],
                "target_binding",
                (type_paths[index],),
                occurrence=occurrence,
            ),
            value_kind="package",
            group_kind="package",
            group_key=f"package:{index}",
            render_policy="categorical_surface",
        )

    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "SERBIA TOBACCO"},
                {"groupId": "g2", "description": "OTHER CARGO"},
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                },
                {
                    "groupId": "g2",
                    "packageId": "p2",
                    "quantity": 96,
                    "typeCategory": "PACKAGE_CARTON",
                },
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1"],
                    "allocations": [{"packageId": "p1", "packageQuantity": 96}],
                },
                {
                    "groupId": "g2",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p2"],
                    "allocations": [{"packageId": "p2", "packageQuantity": 96}],
                },
            ],
        }
    }
    drafts = (
        description,
        quantity(1, 0),
        package_type(1, 0),
        quantity(1, 1),
        package_type(1, 1),
    )

    normalized = normalize_cargo_description_linked_package_prefixes(
        raw=raw,
        drafts=drafts,
        source_target=target,
    )
    first_quantity = raw.index("96")
    first_type = raw.index("CARTONS")
    owners = {
        (row.char_start, row.char_end): row.target_paths
        for row in normalized
        if row.char_start in {first_quantity, first_type}
    }

    assert owners == {
        (first_quantity, first_quantity + 2): quantity_paths[0],
        (first_type, first_type + len("CARTONS")): (type_paths[0],),
    }
    assert any(
        row.char_start > first_type and row.target_paths == quantity_paths[1]
        for row in normalized
    )
    assert (
        normalize_cargo_description_linked_package_prefixes(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )


def test_segmented_party_address_promotes_exact_role_scoped_target_projection() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "PLOT 63 B INDUSTRIAL ZONE A6\n"
        "10TH OF RAMADAN CITY\n"
        "AL SHARQIA, 44629\n"
        "PLOT 63 B INDUSTRIAL ZONE A6\n"
        "10TH OF RAMADAN CITY\n"
        "AL SHARQIA, 44629\n"
    )
    target_path = "documentPatch.parties.consignee.address"
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "address": "PLOT 63 B INDUSTRIAL ZONE A6 AL SHARQIA, 44629",
                    "city": "10TH OF RAMADAN CITY",
                }
            }
        }
    }
    fragments = tuple(
        replace(
            _test_draft(
                raw,
                text,
                "agent:consignee_address",
                "deterministic_auxiliary",
                occurrence=occurrence,
            ),
            value_kind="address",
            group_kind="party",
            group_key="party:consignee:0",
            render_policy="natural_text",
        )
        for occurrence in range(2)
        for text in ("PLOT 63 B INDUSTRIAL ZONE A6", "AL SHARQIA, 44629")
    )

    normalized = normalize_segmented_party_address_targets(
        drafts=fragments,
        source_target=target,
    )

    assert len(normalized) == 4
    assert {row.logical_key for row in normalized} == {"anchor:" + target_path}
    assert {row.render_mode for row in normalized} == {"target_binding"}
    assert {row.target_paths for row in normalized} == {(target_path,)}
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)
    assert (
        normalize_segmented_party_address_targets(
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )


def test_package_residual_normalizes_to_linked_package_count_derivation() -> None:
    raw = "--- PAGE 1 ---\n1 PARCEL\n"
    target_paths = (
        "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
        "documentPatch.cargoPackages[0].quantity",
        "documentPatch.cargoPackages[0].typeCategory",
    )
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 1,
                    "typeCategory": "PACKAGE_PARCEL",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1"],
                    "allocations": [{"packageId": "p1", "packageQuantity": 1}],
                }
            ],
        }
    }
    residual = replace(
        _test_draft(
            raw,
            "1 PARCEL",
            "agent:package:p1:quantity_and_type",
            "agent_residual",
            target_paths,
        ),
        value_kind="package",
        group_kind="package",
        group_key="package:0",
        render_policy="natural_text",
    )

    normalized = normalize_deterministic_draft_semantics(
        drafts=(residual,),
        source_target=target,
    )

    assert len(normalized) == 1
    assert normalized[0].render_mode == "deterministic_derived"
    assert normalized[0].derivation == "package_count"
    assert normalized[0].target_paths == target_paths
    assert normalized[0].dependency_paths == target_paths
    assert normalized[0].evidence_origin == "derived_operational_fact"
    validate_binding_realizations(raw=raw, drafts=normalized, source_target=target)


def test_structured_locality_expands_equipment_receipt_to_complete_target_type() -> None:
    raw = "--- PAGE 1 ---\n1X40HIGH CUBE SAID TO CONTAIN\n"
    container_path = "documentPatch.containers[0]"
    type_path = "documentPatch.containers[0].typeDescription"
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "AAAU1234567", "typeDescription": "40HIGH CUBE"}],
            "cargoAllocationGroups": [],
            "cargoPackages": [],
        }
    }
    partial = replace(
        _test_draft(
            raw,
            "1X40HIGH",
            "agent:container:0:equipment_receipt",
            "deterministic_derived",
            (container_path,),
            (type_path,),
            group_key="container:0",
        ),
        value_kind="equipment",
        group_kind="equipment",
    )

    normalized = normalize_structured_row_locality(
        raw=raw,
        drafts=(partial,),
        source_target=target,
    )

    assert len(normalized) == 1
    assert normalized[0].source_text == "1X40HIGH CUBE"
    assert normalized[0].char_end == raw.index(" SAID TO CONTAIN")
    assert "SAID TO CONTAIN" not in normalized[0].source_text
    assert (
        normalize_structured_row_locality(
            raw=raw,
            drafts=normalized,
            source_target=target,
        )
        == normalized
    )
