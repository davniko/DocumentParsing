from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generators import DeterministicStream, validate_container_number
from document_ocr.synthesis.template_compiler import descendant as r
from document_ocr.synthesis.template_compiler import entity_identifiers
from document_ocr.synthesis.template_compiler.complete_pipeline import (
    ProviderCallError,
    _generation_schema,
    _provider_unavailable,
    _reusable_residual,
)
from document_ocr.synthesis.template_compiler.descendant_models import ResidualStageReceipt
from document_ocr.synthesis.template_compiler.fixed_vocabulary import fixed_context
from document_ocr.synthesis.template_compiler.models import AggregateRangeConstraint
from document_ocr.synthesis.template_compiler.package_prose import validate
from document_ocr.synthesis.template_compiler.realization_contract import token_projection_pattern
from document_ocr.synthesis.template_compiler.synthetic_values import DeterministicValueFactory


def binding(text, **updates):
    return NS(
        **{
            "logical_key": "agent:commercial:free_time",
            "target_paths": (),
            "dependency_paths": (),
            "dependency_bindings": (),
            "source_relationships": (),
            "derivation": None,
            "value_kind": "operational_text",
            "group_kind": "commercial",
            "occurrences": (NS(source_text=text),),
            **updates,
        }
    )


def test_identifier_prefix_is_proven_from_repeated_source_not_randomized():
    b = binding(
        "26-1691988",
        value_kind="identifier",
        occurrences=(
            NS(slot_id="a", source_text="26-1691988", render_policy="opaque_identifier"),
            NS(slot_id="b", source_text="EIN261691988", render_policy="opaque_identifier"),
        ),
    )
    result = r._render_text_candidate(b, "41-6829375")
    assert result.replacements == {"a": "41-6829375", "b": "EIN416829375"}
    with pytest.raises(ValueError, match="alphanumerics"):
        r._render_text_candidate(b, "41-682937")


def test_composite_numeric_validation_ignores_container_digits():
    assert Decimal(1) in r._numeric_surface_values("container TTNU6462679 at 1 degrees Celsius")
    assert Decimal(6462679) not in r._numeric_surface_values("container TTNU6462679")
    assert Decimal(-18) in r._numeric_surface_values("container TTNU6462679 at -18 degrees Celsius")


def test_source_only_container_uses_normative_generator_and_retains_printed_shape():
    b = binding(
        "TRHU 845 505-2",
        logical_key="agent:equipment:container:0:number",
        value_kind="identifier",
        occurrences=(
            NS(slot_id="c", source_text="TRHU 845 505-2", render_policy="opaque_identifier"),
        ),
    )
    result = r._render_direct_auxiliary(b, DeterministicStream(3, "test", "document"))
    assert validate_container_number(result.canonical_value)
    assert result.canonical_value.startswith("TRHU")
    assert result.canonical_value != "TRHU8455052"
    assert len(result.replacements["c"]) == len("TRHU 845 505-2")


def test_repair_schema_enforces_canonical_identifier_width():
    schema = _generation_schema(
        [{"key": "ref"}],
        [
            {
                "key": "ref",
                "obligations": [{"opaqueShapes": [{"source": "ABCD12345678"}]}],
            }
        ],
    )
    assert schema.model_validate({"ref": "WXYZ87654321"}).ref == "WXYZ87654321"
    with pytest.raises(ValidationError):
        schema.model_validate({"ref": "WXYZ8765432"})


def test_repair_schema_enforces_interior_literal_and_both_mutable_segments():
    b = NS(
        logical_key="address",
        realization=NS(
            mode="token_projected_surface",
            target_values=(NS(source_value="OLD STREET EGYPT OLD PARK"),),
            slots=(
                NS(
                    required_target_prefix_tokens=(),
                    required_target_suffix_tokens=("EGYPT", "OLD", "PARK"),
                ),
                NS(
                    required_target_prefix_tokens=("OLD", "STREET", "EGYPT"),
                    required_target_suffix_tokens=(),
                ),
            ),
        ),
    )
    pattern = token_projection_pattern(b)
    schema = _generation_schema(
        [{"key": "address"}], [{"key": "address", "obligations": [{"partitionPattern": pattern}]}]
    )
    assert schema.model_validate({"address": "Unit 14, Egypt, Red Sea Industrial Park"})
    for invalid in (
        "Unit 14, Red Sea Industrial Park, Egypt",
        "Egypt, Red Sea Industrial Park",
        "Unit 14, France, Red Sea Industrial Park",
    ):
        with pytest.raises(ValidationError):
            schema.model_validate({"address": invalid})


def test_equal_party_tax_identifiers_share_one_value_and_preserve_country_prefix():
    bindings = tuple(
        NS(
            logical_key=key,
            target_paths=(),
            value_kind="identifier",
            source_relationships=(),
            occurrences=(
                NS(source_text="DE123456789", render_policy="opaque_identifier", slot_id=key),
            ),
        )
        for key in ("tax1", "tax2")
    )
    entities = tuple(
        NS(target_party_path=path, members=(NS(field="tax_identifier", logical_key=key),))
        for key, path in zip(
            ("tax1", "tax2"),
            ("documentPatch.parties.shipper", "documentPatch.parties.consignee"),
            strict=True,
        )
    )
    template = NS(bindings=bindings, auxiliary_semantic_plan=NS(entities=entities))
    old_party = dict(name="Old Company", address="Old Street", city="Hamburg", country="Germany")
    new_party = {**old_party, "name": "New Company", "address": "New Street"}
    source = {"documentPatch": {"parties": {"shipper": old_party, "consignee": old_party}}}
    target = {"documentPatch": {"parties": {"shipper": new_party, "consignee": new_party}}}
    groups, prefixes = entity_identifiers.contracts(template, source, target, {"germany": "DE"})
    assert groups == (("tax1", "tax2"),)
    outputs = {
        key: r.BindingOutput(replacements={key: "XX987654321"}, canonical_value="XX987654321")
        for key in ("tax1", "tax2")
    }
    r._solve_identifier_relationships(
        template=template,
        outputs=outputs,
        stream=DeterministicStream(1, "test", "document"),
        equivalence_groups=groups,
        fixed_prefixes=prefixes,
    )
    entity_identifiers.validate(groups, prefixes, outputs)
    assert outputs["tax1"].canonical_value == outputs["tax2"].canonical_value
    assert outputs["tax1"].canonical_value.startswith("DE")
    assert outputs["tax1"].canonical_value != "DE123456789"
    target["documentPatch"]["parties"]["consignee"] = {**new_party, "name": "A Different Company"}
    assert not entity_identifiers.contracts(template, source, target, {"germany": "DE"})[0]


def test_independent_address_omits_separately_owned_locality_and_uses_postal_shape():
    factory = DeterministicValueFactory(seed=2, document_id="doc", target={"documentPatch": {}})
    entity = NS(
        entity_id="agent",
        target_party_path=None,
        members=tuple(NS(field=f) for f in ("address", "postal_code", "city", "country")),
    )
    values = {
        f: factory.entity_textual(
            entity=entity, member=NS(field=f), country_codes={}, postal_code_pattern="99999"
        )
        for f in ("address", "postal_code", "city", "country")
    }
    assert len(values["postal_code"]) == 5
    assert values["postal_code"].isdigit()
    assert values["postal_code"] not in values["address"]
    assert values["city"] not in values["address"]
    geo = factory.geography_for_identity("agent", postal_code_pattern="99999")
    assert (values["postal_code"], values["city"], values["country"]) == (
        geo.postal_code,
        geo.city,
        geo.country,
    )
    with pytest.raises(ValueError, match="no synthetic geography"):
        factory.geography_for_identity("agent", postal_code_pattern="99999999999999999999")


def test_fixed_freetime_requires_both_closed_grammar_and_semantic_role():
    target = {"documentPatch": {}}
    assert fixed_context(binding("21 days"), target, target)
    assert not fixed_context(binding("21 days FOR ACME"), target, target)
    assert not fixed_context(binding("21 days", logical_key="agent:cargo:quantity"), target, target)
    assert not fixed_context(binding("21 days", dependency_paths=("changed",)), target, target)


def test_dg_packing_group_retention_requires_unchanged_identities():
    source = {"documentPatch": {"cargoGroups": [{"hsCodes": ["290101"]}]}}
    target = deepcopy(source)
    b = binding(
        "PG III",
        logical_key="agent:dangerous_goods:packing_group",
        value_kind="dangerous_goods",
        group_kind="dangerous_goods",
    )
    assert fixed_context(b, source, target)
    target["documentPatch"]["cargoGroups"][0]["hsCodes"] = ["291101"]
    assert not fixed_context(b, source, target)
    assert not fixed_context(b, {"documentPatch": {}}, {"documentPatch": {}})


def test_new_generic_package_word_can_express_a_declared_lot_but_not_wrong_count():
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "CAR PARTS"}],
            "cargoPackages": [{"groupId": "g1", "quantity": 1, "typeCategory": "PACKAGE_LOT"}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoGroups"][0]["description"] = "Brake parts, one package lot"
    validate(source, target)
    target["documentPatch"]["cargoGroups"][0]["description"] = "Brake parts, two package lots"
    with pytest.raises(ValueError, match="matching structured fact"):
        validate(source, target)


def test_paid_residual_projection_keeps_provider_proof_and_rejects_changed_facts():
    target = {"documentPatch": {"name": "Generated Identity"}}
    facts = {"frozen": "receipt"}
    case = NS(
        document_id="sample",
        source_document_id="source",
        auxiliary_values={},
        numeric_auxiliary={},
        equipment_tare_values={},
        target_receipt=NS(model_dump=lambda **kw: facts),
    )
    plan = NS(residual_bindings=(NS(occurrences=(NS(slot_id="kept"),)),))
    stage = ResidualStageReceipt(
        schema_version=1,
        document_id="sample",
        status="success",
        started_at=datetime.now(UTC).isoformat(),
        completed_at=datetime.now(UTC).isoformat(),
        duration_seconds=0,
        system_prompt_sha256="a" * 64,
        user_prompt_sha256="b" * 64,
        output_schema_sha256="c" * 64,
        output={"kept": "New text", "now_deterministic": "Fixed label"},
        error_type=None,
        error_message=None,
        messages=[],
        usage=r._empty_usage().model_dump(mode="json"),
    )
    checkpoint = dict(
        sampleId="sample",
        sourceDocumentId="source",
        target=target,
        targetSha256=sha256_bytes(canonical_json_bytes(target)),
        targetReceipt=facts,
        auxiliaryValues={},
        numericAuxiliary={},
        agentStage=stage.model_dump(mode="json"),
        residualResponse=dict(
            output=stage.output, outputSha256=sha256_bytes(canonical_json_bytes(stage.output))
        ),
    )
    raw, proof, receipt = _reusable_residual(checkpoint, case, plan, "a" * 64)
    assert raw == {"kept": "New text"}
    assert proof.output == stage.output
    assert receipt["cacheReused"]
    checkpoint["targetReceipt"] = {"frozen": "different"}
    assert _reusable_residual(checkpoint, case, plan, "a" * 64) is None
    checkpoint["targetSha256"] = "0" * 64
    with pytest.raises(ValueError, match="target hash"):
        _reusable_residual(checkpoint, case, plan, "a" * 64)


def test_connection_failures_stop_admission_but_content_rejections_do_not():
    assert _provider_unavailable(ProviderCallError({"error": "Connection error."}))
    assert not _provider_unavailable(ValueError("wrong package count"))


@pytest.mark.parametrize(
    "failure_type", ["APIConnectionError", "AuthenticationError", "RateLimitError"]
)
@pytest.mark.parametrize("recovers", [True, False])
def test_connection_recovery_is_explicit_bounded_and_never_retries_account_errors(
    failure_type, recovers
):
    import asyncio

    from document_ocr.synthesis.template_compiler.complete_pipeline import _recover_connection_once

    calls = []
    failure = dict(
        error="Connection error.", requestSha256="a" * 64, errorCauses=[dict(type=failure_type)]
    )

    async def request():
        calls.append(1)
        if len(calls) == 1 or not recovers:
            raise ProviderCallError(failure)
        return dict(output={"value": "new"})

    if failure_type == "APIConnectionError" and recovers:
        result = asyncio.run(_recover_connection_once(request))
        assert result["output"] == {"value": "new"}
        assert result["transportRecovery"]["failureReceiptSha256"] == sha256_bytes(
            canonical_json_bytes(failure)
        )
    else:
        with pytest.raises(ProviderCallError):
            asyncio.run(_recover_connection_once(request))
    assert len(calls) == (2 if failure_type == "APIConnectionError" else 1)


def test_aggregate_range_subcounts_require_compiled_proof_and_local_attachment():
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "marksAndNumbers": ["NO. AB100-AB104; AB105-AB109"]}],
            "cargoPackages": [{"groupId": "g1", "quantity": 10, "typeCategory": "PACKAGE_CARTON"}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 8
    target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "NO. CD200-CD203 (4 CARTONS); CD204-CD207 (4 CARTONS)"
    ]
    proof = AggregateRangeConstraint(
        constraint_id="coherence_constraint_aaaaaaaaaaaaaaaa",
        candidate_fingerprint="a" * 64,
        kind="aggregate_inclusive_range_cardinality",
        member_logical_keys=("marks",),
        dependency_paths=("documentPatch.cargoPackages[0].quantity",),
        rationale="Source ranges cover the complete carton quantity.",
    )
    template = NS(
        coherence_constraints=(proof,), bindings=(NS(logical_key="marks", target_paths=(path,)),)
    )
    validate(source, target, template=template)
    with pytest.raises(ValueError, match="matching structured fact"):
        validate(source, target)
    for text in [
        "NO. CD200-CD203 (3 CARTONS); CD204-CD207 (4 CARTONS)",
        "NO. CD200-CD203; CD204-CD207; 4 CARTONS",
    ]:
        target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [text]
        with pytest.raises(ValueError, match="matching structured fact"):
            validate(source, target, template=template)
