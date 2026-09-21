from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler import descendant
from document_ocr.synthesis.template_compiler.generation_contract import require_complete_variation
from document_ocr.synthesis.template_compiler.generation_contract import (
    validate_unbound_lexical_surfaces as validate,
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
    with pytest.raises(ValueError, match="unbound target values absent") as error:
        validate(target=target, binding_paths=set(), rendered="Old Agency 123 Old Grade")
    assert "phoneNumbers[1]" in str(error.value)
    assert "description" in str(error.value)
    validate(target=target, binding_paths=set(), rendered="NEW AGENCY 1-23 4 56 NEW CHEMICAL GRADE")


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
    )
    with pytest.raises(ValueError, match="notifyParties"):
        validate(
            target=target,
            binding_paths={
                "documentPatch.parties.shipper.address",
                "documentPatch.parties.consignee.name",
            },
            rendered="unrelated company",
        )


@pytest.mark.parametrize(
    "control", [chr(i) for i in (*range(9), 11, 12, *range(14, 32), *range(127, 160))]
)
def test_controls_are_rejected_in_targets_and_even_fully_bound_documents(control):
    with pytest.raises(ValueError, match="invalid control"):
        require_complete_variation({}, {"documentPatch": {"vesselName": "A" + control + "B"}})
    with pytest.raises(ValueError, match="invalid control"):
        validate(target={}, binding_paths=set(), rendered="SIGNED (INDIA)" + control + "BY: AGENT")


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
