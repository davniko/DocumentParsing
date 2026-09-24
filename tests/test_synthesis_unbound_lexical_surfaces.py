from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.raw_text_template import (
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)
from document_ocr.synthesis.template_compiler import descendant
from document_ocr.synthesis.template_compiler.generation_contract import (
    require_complete_variation,
    validate_seal_realization,
)
from document_ocr.synthesis.template_compiler.generation_contract import (
    validate_unbound_lexical_surfaces as validate,
)
from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    normalize_source_seal_ownership,
    validate_source_seal_ownership,
)


def test_seal_ownership_is_leaf_level_container_local_and_covers_repeated_slots():
    target = {"documentPatch": {"containers": [{"sealNumbers": ["S001", "S002"]}]}}
    detached = NS(
        target_paths=("documentPatch.containers",),
        group_kind="equipment",
        group_key="container:0",
        occurrences=(NS(slot_id="count"),),
    )
    with pytest.raises(ValueError, match="exactly one leaf binding"):
        validate_seal_realization(
            target=target, bindings=(detached,), slot_values={"count": "ONE CONTAINER S001 S002"}
        )
    bindings = (
        NS(
            target_paths=("documentPatch.containers[0].sealNumbers[0]",),
            group_kind="equipment",
            group_key="container:0",
            occurrences=(NS(slot_id="seal0_a"), NS(slot_id="seal0_b")),
        ),
        NS(
            target_paths=("documentPatch.containers[0].sealNumbers[1]",),
            group_kind="equipment",
            group_key="container:0",
            occurrences=(NS(slot_id="seal1"),),
        ),
    )
    values = {"seal0_a": "S001", "seal0_b": "S-001", "seal1": "S002"}
    validate_seal_realization(target=target, bindings=bindings)
    validate_seal_realization(target=target, bindings=bindings, slot_values=values)
    validate_seal_realization(
        target=target,
        bindings=bindings,
        slot_values=values,
        rendered="CONTAINER 0 / S001 / S002",
    )
    with pytest.raises(ValueError, match="rendered document lacks target seal"):
        validate_seal_realization(
            target=target,
            bindings=bindings,
            slot_values=values,
            rendered="CONTAINER 0 / S001 / STALE",
        )
    with pytest.raises(ValueError, match="does not realize"):
        validate_seal_realization(
            target=target,
            bindings=bindings,
            slot_values={**values, "seal0_b": "OLD001"},
        )
    with pytest.raises(ValueError, match="does not realize"):
        validate_seal_realization(
            target=target,
            bindings=bindings,
            slot_values={**values, "seal1": "S003"},
            rendered="CONTAINER 0 / S001 / S002; CONTAINER 1 / S003",
        )
    with pytest.raises(ValueError, match="wrong container ownership"):
        validate_seal_realization(
            target=target,
            bindings=(NS(**{**bindings[0].__dict__, "group_key": "container:1"}), bindings[1]),
            slot_values=values,
        )
    segmented = NS(
        target_paths=("documentPatch.containers[0].sealNumbers[0]",),
        group_kind="equipment",
        group_key="container:0",
        realization=NS(mode="agent_required"),
        occurrences=(NS(slot_id="prefix"), NS(slot_id="suffix"), NS(slot_id="whole")),
    )
    segmented_target = {"documentPatch": {"containers": [{"sealNumbers": ["071760/SIF811"]}]}}
    segmented_values = {"prefix": "071760", "suffix": "SIF811", "whole": "071760/SIF811"}
    validate_seal_realization(
        target=segmented_target,
        bindings=(segmented,),
        slot_values=segmented_values,
        rendered="071760 / SIF811 / 071760/SIF811",
    )
    with pytest.raises(ValueError, match="does not realize"):
        validate_seal_realization(
            target=segmented_target,
            bindings=(segmented,),
            slot_values={**segmented_values, "suffix": "STALE"},
        )


def test_joined_container_seal_uses_literal_separator_outside_complete_seal_slot():
    source = b"ABCU12345607CM12345678/20GP"
    source_seal = "CM12345678"
    new_seal = "AB87654321"
    start = source.index(source_seal.encode())
    slot = build_template_slot(
        slot_id="slot_0001",
        byte_start=start,
        byte_end=start + len(source_seal),
        source_text=source_seal,
        target_paths=("documentPatch.containers[0].sealNumbers[0]",),
        semantic_role="container:0",
        evidence_origin="accepted_label_evidence",
        render_policy="opaque_identifier",
    )
    template = compile_raw_text_template(document_id="joined-seal", source=source, slots=(slot,))
    rendered, proof = render_compiled_template(
        source=source, template=template, bindings={"slot_0001": new_seal}
    )
    assert rendered == b"ABCU12345607AB87654321/20GP"
    assert proof.exact_literal_regions
    binding = NS(
        target_paths=("documentPatch.containers[0].sealNumbers[0]",),
        group_kind="equipment",
        group_key="container:0",
        realization=NS(mode="single_surface", adapter="opaque_identifier"),
        occurrences=(slot,),
    )
    target = {"documentPatch": {"containers": [{"sealNumbers": [new_seal]}]}}
    validate_seal_realization(
        target=target,
        bindings=(binding,),
        slot_values={"slot_0001": new_seal},
        rendered=rendered.decode(),
    )
    with pytest.raises(ValueError, match="does not realize"):
        validate_seal_realization(
            target=target,
            bindings=(binding,),
            slot_values={"slot_0001": "7" + new_seal},
        )


def test_compiler_promotes_only_unique_exact_container_local_seal_auxiliaries():
    raw = "ABCU1234567 / 0010496\nEFGU7654321 / 0010493"
    values = ("0010496", "0010493")
    starts = tuple(raw.index(value) for value in values)
    drafts = tuple(
        SpanDraft(
            draft_id=f"seal_{index}",
            logical_key=f"source_only_seal_{index}",
            render_mode="deterministic_auxiliary",
            value_kind="equipment",
            group_kind="equipment",
            group_key=f"container:{index}",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(values[index]),
            source_text=values[index],
            evidence_origin="host_verified_agent_proposal",
            render_policy="opaque_identifier",
            rationale="Source-only model proposal for a printed seal.",
        )
        for index, start in enumerate(starts)
    )
    target = {"documentPatch": {"containers": [{"sealNumbers": [value]} for value in values]}}
    with pytest.raises(ValueError, match="requires exactly one leaf binding"):
        validate_source_seal_ownership(drafts=drafts, source_target=target)
    promoted = normalize_source_seal_ownership(drafts=drafts, source_target=target)
    validate_source_seal_ownership(drafts=promoted, source_target=target)
    assert [draft.target_paths for draft in promoted] == [
        (f"documentPatch.containers[{index}].sealNumbers[0]",) for index in range(2)
    ]
    wrong_scope = (drafts[0], replace(drafts[1], group_key="container:0"))
    with pytest.raises(ValueError, match="requires exactly one leaf binding"):
        validate_source_seal_ownership(
            drafts=normalize_source_seal_ownership(drafts=wrong_scope, source_target=target),
            source_target=target,
        )


def test_unbound_new_party_cargo_and_second_phone_cannot_escape_binding_validation():
    target = {
        "documentPatch": {
            "parties": {
                "forwardingAgent": {
                    "name": "New Agency",
                    "contactDetails": {"phoneNumbers": ["123", "456"]},
                }
            },
            "cargoGroups": [{"description": "New chemical grade"}],
        }
    }
    with pytest.raises(ValueError, match="party target lacks role-owned printed evidence") as error:
        validate(
            target=target,
            binding_paths=set(),
            rendered="Old Agency 123 Old Grade",
            party_surfaces={"documentPatch.parties.forwardingAgent": ("Old Agency 123",)},
        )
    assert "phoneNumbers[1]" in str(error.value)
    owned = {"documentPatch.parties.forwardingAgent": ("NEW AGENCY 1-23, 4 56",)}
    with pytest.raises(ValueError, match="description"):
        validate(
            target=target,
            binding_paths=set(),
            rendered="NEW AGENCY 1-23, 4 56 Old Grade",
            party_surfaces=owned,
        )
    validate(
        target=target,
        binding_paths=set(),
        rendered="NEW AGENCY 1-23, 4 56 NEW CHEMICAL GRADE",
        party_surfaces=owned,
    )


def test_shared_party_and_direct_segmented_binding_use_their_existing_proofs():
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {"address": "23 Commercial Avenue East"},
                "consignee": {"name": "Shared Company"},
                "notifyParties": [{"name": "Shared Company"}],
            }
        }
    }
    validate(
        target=target,
        binding_paths={
            "documentPatch.parties.shipper.address",
            "documentPatch.parties.consignee.name",
        },
        rendered="23 Commercial / continued on next page / Avenue East SHARED COMPANY",
        party_surfaces={"documentPatch.parties.notifyParties[0]": ("SHARED COMPANY",)},
    )
    with pytest.raises(ValueError, match="notifyParties"):
        validate(
            target=target,
            binding_paths={
                "documentPatch.parties.shipper.address",
                "documentPatch.parties.consignee.name",
            },
            rendered="unrelated company",
            party_surfaces={"documentPatch.parties.consignee": ("SHARED COMPANY",)},
        )


@pytest.mark.parametrize(
    "control", [chr(i) for i in (*range(9), 11, 12, *range(14, 32), *range(127, 160))]
)
def test_controls_are_rejected_in_targets_and_even_fully_bound_documents(control):
    with pytest.raises(ValueError, match="invalid control"):
        require_complete_variation({}, {"documentPatch": {"vesselName": "A" + control + "B"}})
    with pytest.raises(ValueError, match="invalid control"):
        validate(
            target={},
            binding_paths=set(),
            rendered="SIGNED (INDIA)" + control + "BY: AGENT",
            party_surfaces={},
        )


def test_residual_nul_is_rejected_before_layout_or_identifier_shaping_can_hide_it():
    for policy in ("opaque_identifier", "free_text"):
        slot = NS(slot_id="slot", render_policy=policy)
        plan = NS(residual_bindings=(NS(occurrences=(slot,)),))
        with pytest.raises(ValueError, match="residual slot slot contains an invalid control"):
            descendant._postprocess_residual_outputs(
                case=NS(), plan=plan, raw_output={"slot": "A\x00B"}
            )


def test_native_residual_schema_forbids_controls_at_every_position():
    slot = NS(
        slot_id="slot",
        semantic_role="legal",
        render_policy="natural_text",
        source_text="SIGNED\nBY AGENT",
    )
    schema = descendant._residual_output_type((NS(occurrences=(slot,), value_kind="legal_text"),))
    controls = [chr(i) for i in (*range(32), *range(127, 160))]
    for invalid in [
        " A",
        "A ",
        *(s for c in controls for s in (c, "A" + c + "B", c + "A", "A" + c)),
    ]:
        with pytest.raises(ValidationError):
            schema.model_validate({"slot": invalid})
    assert (
        schema.model_validate({"slot": "SIGNED (INDIA) BY: AGENT"}).slot
        == "SIGNED (INDIA) BY: AGENT"
    )
    assert "pattern" in schema.model_json_schema()["properties"]["slot"]


@pytest.mark.parametrize(
    "escaped",
    [r"SIGNED\nBY AGENT", r"SIGNED\NBY AGENT", r"SIGNED\u0000BY AGENT", r"SIGNED\\nBY AGENT"],
)
def test_residual_must_not_encode_host_owned_line_breaks(escaped):
    slot = NS(
        slot_id="slot",
        semantic_role="legal",
        render_policy="natural_text",
        source_text="SIGNED\nBY AGENT",
    )
    binding = NS(occurrences=(slot,), value_kind="legal_text")
    schema = descendant._residual_output_type((binding,))
    with pytest.raises(ValidationError):
        schema.model_validate({"slot": escaped})
    with pytest.raises(ValueError, match="encoded separator"):
        descendant._postprocess_residual_outputs(
            case=NS(), plan=NS(residual_bindings=(binding,)), raw_output={"slot": escaped}
        )


@pytest.mark.parametrize("ownership", ["target_paths", "dependency_paths"])
def test_composite_phone_binding_can_express_its_complete_party_contract(ownership):
    party = "documentPatch.parties.notifyParties[0]"
    slot = NS(
        slot_id="slot",
        semantic_role="notify party",
        render_policy="natural_text",
        source_text="Old City, Old Address, Old Country PHONE:+20 12345678",
    )
    binding = NS(occurrences=(slot,), value_kind="phone", target_paths=(), dependency_paths=())
    setattr(
        binding,
        ownership,
        (
            party + ".address",
            party + ".city",
            party + ".country",
            party + ".contactDetails.phoneNumbers[0]",
        ),
    )
    value = "Plot 31, Musaffah, United Arab Emirates PHONE:+971 50 123 7569"
    assert (
        descendant._residual_output_type((binding,)).model_validate({"slot": value}).slot == value
    )


@pytest.mark.parametrize(
    "paths", [(), ("documentPatch.parties.shipper.contactDetails.phoneNumbers[0]",)]
)
def test_phone_only_bindings_retain_strict_phone_syntax(paths):
    slot = NS(
        slot_id="slot",
        semantic_role="phone",
        render_policy="natural_text",
        source_text="+20 12345678",
    )
    binding = NS(occurrences=(slot,), value_kind="phone", target_paths=paths, dependency_paths=())
    schema = descendant._residual_output_type((binding,))
    assert schema.model_validate({"slot": "+971 50 123 7569"}).slot == "+971 50 123 7569"
    with pytest.raises(ValidationError):
        schema.model_validate({"slot": "Musaffah PHONE:+971 50 123 7569"})
