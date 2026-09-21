from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from document_ocr.synthesis.template_compiler.descendant import (
    _validate_unrepresented_party_facets,
)
from document_ocr.synthesis.template_compiler.models import (
    AuxiliaryBindingDisposition,
    AuxiliaryEntity,
    AuxiliaryEntityMember,
    AuxiliarySemanticPlan,
)
from document_ocr.synthesis.template_compiler.semantic_plan import (
    AuxiliarySemanticPlanReviewRequired,
    build_auxiliary_semantic_plan,
    resolve_geographic_members,
    validate_auxiliary_render,
)


def test_proven_name_suffix_is_validated_as_a_facet_not_as_the_whole_party():
    entity = SimpleNamespace(
        entity_id="test",
        target_party_path="documentPatch.parties.shipper",
        members=(SimpleNamespace(logical_key="suffix", field="name"),),
    )
    plan = SimpleNamespace(entities=(entity,), composite_numbers=(), document_sequences=())
    target = {
        "documentPatch": {"parties": {"shipper": {"name": "NEW EXPORT CO ON BEHALF OF PRINCIPAL"}}}
    }
    outputs = {"suffix": SimpleNamespace(canonical_value="ON BEHALF OF PRINCIPAL")}
    kwargs = dict(plan=plan, bindings=(), outputs=outputs, target=target, country_codes={})
    with pytest.raises(ValueError, match="name differs"):
        validate_auxiliary_render(**kwargs)
    validate_auxiliary_render(
        **kwargs, target_projection_values={"suffix": "ON BEHALF OF PRINCIPAL"}
    )
    with pytest.raises(ValueError, match="facet differs"):
        validate_auxiliary_render(**kwargs, target_projection_values={"suffix": "WRONG PRINCIPAL"})


def _binding(
    logical_key: str,
    source_text: str,
    *,
    value_kind: str = "other_text",
    group_kind: str = "document",
    group_key: str = "document:control",
    render_mode: str = "deterministic_auxiliary",
    dependency_bindings: tuple[str, ...] = (),
) -> Any:
    return SimpleNamespace(
        logical_key=logical_key,
        render_mode=render_mode,
        value_kind=value_kind,
        group_kind=group_kind,
        group_key=group_key,
        target_paths=(),
        dependency_paths=(),
        dependency_bindings=dependency_bindings,
        occurrences=(SimpleNamespace(source_text=source_text),),
        source_relationships=(),
    )


@pytest.mark.parametrize(
    "key,code,country,expected",
    [
        ("agent:foreign_exporter_code", "AE", "UNITED ARAB EMIRATES", "country_code"),
        ("agent:foreign_exporter_code", "ARE", "UNITED ARAB EMIRATES", "country_code"),
        ("agent:foreign_exporter_code", "DE", "UNITED ARAB EMIRATES", "registration_identifier"),
        ("agent:foreign_exporter_code", "AE123", "UNITED ARAB EMIRATES", "registration_identifier"),
        (
            "agent:foreign_exporter_registration_code",
            "AE",
            "UNITED ARAB EMIRATES",
            "registration_identifier",
        ),
        (
            "agent:foreign_exporter_tax_code",
            "AE",
            "UNITED ARAB EMIRATES",
            "registration_identifier",
        ),
        ("agent:foreign_exporter_code", "AE", "UNKNOWN", "registration_identifier"),
        ("agent:foreign_exporter_id", "AE", "UNITED ARAB EMIRATES", "registration_identifier"),
    ],
)
def test_country_code_resolution_requires_closed_same_entity_iso_evidence(
    key, code, country, expected
):
    entity = AuxiliaryEntity(
        entity_id="aux_entity_0123456789abcdef",
        role="exporter",
        relationship="independent",
        target_party_path=None,
        rationale="Explicit exporter fields.",
        members=(
            AuxiliaryEntityMember(logical_key="country", field="country"),
            AuxiliaryEntityMember(logical_key=key, field="registration_identifier"),
        ),
    )
    plan = AuxiliarySemanticPlan(
        schema_version=1,
        entities=(entity,),
        composite_numbers=(),
        document_sequences=(),
        dispositions=tuple(
            AuxiliaryBindingDisposition(
                logical_key=member.logical_key,
                disposition="entity_member",
                semantic_id=entity.entity_id,
                rationale="Same explicit exporter.",
            )
            for member in entity.members
        ),
    )
    bindings = (_binding("country", country), _binding(key, code))
    countries = {
        "unitedarabemirates": "AE",
        "ae": "AE",
        "are": "AE",
        "de": "DE",
        "canada": "CA",
        "ca": "CA",
    }
    resolved = resolve_geographic_members(plan, bindings, countries)
    assert resolved.entities[0].members[1].field == expected
    assert plan.entities[0].members[1].field == "registration_identifier"
    if expected != "country_code":
        assert resolved is plan
        return
    outputs = {
        "country": SimpleNamespace(canonical_value="CANADA"),
        key: SimpleNamespace(canonical_value="CA"),
    }
    validate_auxiliary_render(
        plan=plan, bindings=bindings, outputs=outputs, target={}, country_codes=countries
    )
    outputs[key].canonical_value = "XT"
    with pytest.raises(ValueError, match="country code conflicts"):
        validate_auxiliary_render(
            plan=plan, bindings=bindings, outputs=outputs, target={}, country_codes=countries
        )
    # A country in a different entity does not license reclassification.
    unrelated = entity.model_copy(
        update={"entity_id": "aux_entity_fedcba9876543210", "members": entity.members[:1]}
    )
    isolated_code = entity.model_copy(update={"members": entity.members[1:]})
    separated = plan.model_copy(update={"entities": (unrelated, isolated_code)})
    assert resolve_geographic_members(separated, bindings, countries) is separated


def test_plan_formalizes_word_digit_counts_and_bounded_sequences() -> None:
    bindings = (
        _binding("agent:legal:original_count", "ZERO (0)"),
        _binding("agent:document_copy_sequence_1", "1 Of Three"),
        _binding("agent:document_copy_sequence_2", "2 Of Three"),
    )

    plan = build_auxiliary_semantic_plan(
        raw="ZERO (0)\n1 Of Three\n2 Of Three\n",
        bindings=bindings,
        source_target={"documentPatch": {}},
    )

    assert tuple(row.value for row in plan.composite_numbers) == (0,)
    assert len(plan.document_sequences) == 1
    assert plan.document_sequences[0].total == 3
    assert tuple(row.index for row in plan.document_sequences[0].members) == (1, 2)


def test_plan_rejects_an_impossible_document_sequence() -> None:
    with pytest.raises(AuxiliarySemanticPlanReviewRequired, match="sequence is impossible"):
        build_auxiliary_semantic_plan(
            raw="8 Of Three\n",
            bindings=(_binding("agent:document_copy_sequence_8", "8 Of Three"),),
            source_target={"documentPatch": {}},
        )


def test_declared_dependency_remains_in_its_party_entity() -> None:
    dependency = _binding(
        "agent:fax_notify_0",
        "84899112",
        value_kind="phone",
        group_kind="party",
        group_key="party:notify:0",
        render_mode="agent_residual",
    )
    derived = _binding(
        "agent:customs_exporter_registration",
        "TW-02-84899112",
        value_kind="identifier",
        group_kind="customs",
        group_key="customs:exporter_registration",
        render_mode="deterministic_derived",
        dependency_bindings=(dependency.logical_key,),
    )

    plan = build_auxiliary_semantic_plan(
        raw="FAX:84899112\nEXPORTER REGISTRATION: TW-02-84899112\n",
        bindings=(dependency, derived),
        source_target={
            "documentPatch": {
                "parties": {"notifyParties": [{"name": "Example Notify", "country": "Taiwan"}]}
            }
        },
    )

    disposition = next(
        row for row in plan.dispositions if row.logical_key == dependency.logical_key
    )
    assert disposition.disposition == "entity_member"
    assert any(
        dependency.logical_key in {member.logical_key for member in entity.members}
        for entity in plan.entities
    )


def test_phone_extension_has_a_distinct_entity_field() -> None:
    binding = _binding(
        "agent:delivery_agent_extension",
        "EXT 112",
        value_kind="phone",
        group_kind="party",
        group_key="party:deliveryAgent:0",
    )

    plan = build_auxiliary_semantic_plan(
        raw="Delivery Agent\nEXT 112\n",
        bindings=(binding,),
        source_target={
            "documentPatch": {"parties": {"deliveryAgent": {"name": "Source Delivery Agent"}}}
        },
    )

    assert plan.entities[0].members[0].field == "phone_extension"


def test_unrepresented_party_context_rejects_without_restoring_the_source_party() -> None:
    plan = AuxiliarySemanticPlan.model_validate(
        {
            "schema_version": 1,
            "entities": (
                {
                    "entity_id": "aux_entity_0123456789abcdef",
                    "role": "shipper:0",
                    "relationship": "same_as_target_party",
                    "target_party_path": "documentPatch.parties.shipper",
                    "members": ({"logical_key": "agent:shipper_province", "field": "region"},),
                    "rationale": "The printed province belongs to the shipper.",
                },
            ),
            "composite_numbers": (),
            "document_sequences": (),
            "dispositions": (
                {
                    "logical_key": "agent:shipper_province",
                    "disposition": "entity_member",
                    "semantic_id": "aux_entity_0123456789abcdef",
                    "rationale": "Canonical shipper member.",
                },
            ),
        }
    )
    source_party = {"name": "Source Shipper", "country": "Canada"}
    target = {
        "documentPatch": {"parties": {"shipper": {"name": "Synthetic Shipper", "country": "Spain"}}}
    }

    with pytest.raises(ValueError, match="source-party restoration is forbidden"):
        _validate_unrepresented_party_facets(
            source_target={"documentPatch": {"parties": {"shipper": source_party}}},
            target=target,
            template=SimpleNamespace(auxiliary_semantic_plan=plan),
        )

    assert target["documentPatch"]["parties"]["shipper"] == {
        "name": "Synthetic Shipper",
        "country": "Spain",
    }
