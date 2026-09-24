from dataclasses import replace

import pytest

from document_ocr.synthesis.template_compiler import host


def _draft(key, text, start, *, role="consignee", address=False, target_paths=()):
    return host.SpanDraft(
        draft_id=key,
        logical_key=key,
        render_mode="target_binding" if address or target_paths else "deterministic_auxiliary",
        value_kind="address" if address else "identifier",
        group_kind="party",
        group_key=f"party:{role}:0",
        target_paths=(f"documentPatch.parties.{role}.address",) if address else target_paths,
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        evidence_origin="agent_proposal",
        render_policy="natural_text",
        rationale="Explicit role-owned field.",
    )


def test_postal_fragment_completes_only_its_own_role_address():
    street = _draft("address", "Port Road", 0, address=True)
    postcode = replace(
        _draft("consignee_postal_code", "21500", 30), render_policy="opaque_identifier"
    )
    foreign = _draft("shipper_postal_code", "21500", 50, role="shipper")
    city = replace(
        _draft("city", "Cairo", 20, target_paths=("documentPatch.parties.consignee.city",)),
        value_kind="location",
    )
    target = {
        "documentPatch": {"parties": {"consignee": {"address": "Port Road 21500", "city": "Cairo"}}}
    }
    result = host.normalize_owned_party_address_components(
        drafts=(street, postcode, foreign, city), source_target=target
    )
    by_id = {d.draft_id: d for d in result}
    assert by_id[postcode.draft_id].logical_key == street.logical_key
    assert by_id[postcode.draft_id].target_paths == street.target_paths
    assert by_id[postcode.draft_id].render_policy == street.render_policy
    assert by_id[foreign.draft_id] == foreign
    assert by_id[city.draft_id] == city
    assert (
        host.normalize_owned_party_address_components(drafts=result, source_target=target) == result
    )


def test_numeric_coincidence_and_referenced_postcodes_are_not_promoted():
    street = _draft("address", "Port Road", 0, address=True)
    vat = _draft("consignee_tax_id", "21500", 30)
    postal = _draft("consignee_postal_code", "21500", 40)
    dependent = replace(_draft("other", "Reference", 60), dependency_bindings=(postal.logical_key,))
    target = {"documentPatch": {"parties": {"consignee": {"address": "Port Road 21500"}}}}
    result = host.normalize_owned_party_address_components(
        drafts=(street, vat, postal, dependent), source_target=target
    )
    assert {d.draft_id: d for d in result} == {
        d.draft_id: d for d in (street, vat, postal, dependent)
    }


def test_competing_postal_owners_remain_unmodified():
    street = _draft("address", "Port Road", 0, address=True)
    first = _draft("postal_code_a", "21500", 30)
    second = _draft("postal_code_b", "21500", 40)
    target = {"documentPatch": {"parties": {"consignee": {"address": "Port Road 21500"}}}}
    result = host.normalize_owned_party_address_components(
        drafts=(street, first, second), source_target=target
    )
    assert {d.draft_id: d for d in result} == {d.draft_id: d for d in (street, first, second)}


def test_unbound_address_and_interleaved_postcode_form_one_exact_target():
    raw = "Port Tawfik Free Zone\nArea Suez Egypt\nALEXANDRIA 55698 Egypt"
    street = replace(
        _draft("agent:party:consignee:address", "Port Tawfik Free Zone\nArea Suez", 0),
        render_mode="agent_residual",
        value_kind="address",
    )
    postcode = _draft("agent:party:consignee:postal_code", "55698", raw.index("55698"))
    city = replace(
        _draft(
            "agent:party:consignee:city",
            "ALEXANDRIA",
            raw.index("ALEXANDRIA"),
            target_paths=("documentPatch.parties.consignee.city",),
        ),
        value_kind="location",
    )
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "address": "Port Tawfik Free Zone Area Suez 55698",
                    "city": "ALEXANDRIA",
                }
            }
        }
    }
    output = host.normalize_segmented_party_address_targets(
        drafts=(street, city, postcode), source_target=target
    )
    by_id = {draft.draft_id: draft for draft in output}
    assert by_id[street.draft_id].logical_key == by_id[postcode.draft_id].logical_key
    assert by_id[street.draft_id].target_paths == (
        "documentPatch.parties.consignee.address",
    )
    assert by_id[postcode.draft_id].value_kind == "address"
    assert by_id[postcode.draft_id].render_policy == "opaque_identifier"
    assert by_id[city.draft_id] == city
    host.validate_draft_source_alignment(raw=raw, drafts=output)


@pytest.mark.parametrize("prefix", ["NO.98, ", "2,2/"])
def test_multiline_address_completes_exact_prefix_on_first_line(prefix):
    text = "F,FUDU BUILDING,\n98 ARGYLE STREET"
    raw = "SHIPPER\n" + prefix + text + ",CAIRO\nOTHER PARTY"
    owner = _draft("address", text, raw.index(text), address=True)
    target = {"documentPatch": {"parties": {"consignee": {"address": prefix + text}}}}
    output = host.normalize_party_address_inline_segments(
        raw=raw, drafts=(owner,), source_target=target
    )
    assert len(output) == 2
    addition = next(d for d in output if d.draft_id != owner.draft_id)
    assert addition.source_text == prefix.rstrip(" ,/")
    assert addition.target_paths == owner.target_paths
    assert (
        host.normalize_party_address_inline_segments(raw=raw, drafts=output, source_target=target)
        == output
    )
    host.validate_draft_source_alignment(raw=raw, drafts=output)


def test_inline_address_does_not_consume_part_of_another_owned_surface():
    raw = "REF NO.98, Port Road\nCairo"
    owner = _draft("address", "Port Road", raw.index("Port Road"), address=True)
    other = _draft("reference", "REF NO.98", 0)
    target = {"documentPatch": {"parties": {"consignee": {"address": "NO.98, Port Road"}}}}
    output = host.normalize_party_address_inline_segments(
        raw=raw, drafts=(owner, other), source_target=target
    )
    assert {d.draft_id: d for d in output} == {d.draft_id: d for d in (owner, other)}
