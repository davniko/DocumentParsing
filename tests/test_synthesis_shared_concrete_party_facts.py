from copy import deepcopy
from dataclasses import replace

import pytest

from document_ocr.synthesis.template_compiler import host


def fixture():
    party = dict(name="SAMPLE TRADING", address="10 Sample Road NORTH DISTRICT", city="Harbor")
    source = {
        "documentPatch": {"parties": {"consignee": party, "notifyParties": [deepcopy(party)]}}
    }
    paths = (
        "documentPatch.parties.consignee.address",
        "documentPatch.parties.notifyParties[0].address",
    )
    raw = "10 Sample Road\n10 Sample Road\nNORTH DISTRICT"
    drafts = []
    start = 0
    for i, text in enumerate(raw.splitlines()):
        drafts.append(
            host.SpanDraft(
                draft_id=f"s{i}",
                logical_key="shared-address",
                render_mode="target_binding",
                value_kind="address",
                group_kind="party",
                group_key="shared-address",
                target_paths=paths,
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=start + len(text),
                source_text=text,
                evidence_origin="accepted_label_evidence",
                render_policy="natural_text",
                rationale="test",
            )
        )
        start += len(text) + 1
    return source, paths, raw, tuple(drafts)


def test_complete_identical_parties_can_own_repeated_segmented_shared_address():
    source, paths, raw, drafts = fixture()
    assert host.target_fact_components(source, paths) == (paths,)
    host.validate_target_binding_relationships(drafts=drafts, source_target=source)
    host.validate_binding_realizations(raw=raw, drafts=drafts, source_target=source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "OTHER TRADING"),
        ("city", "Other Harbor"),
        ("contactDetails", {"phoneNumbers": ["12345"]}),
        ("address", ""),
        ("name", ""),
    ],
)
def test_equal_address_field_does_not_merge_distinct_or_incomplete_parties(field, value):
    source, paths, _, drafts = fixture()
    if value == "":
        source["documentPatch"]["parties"]["consignee"][field] = value
    source["documentPatch"]["parties"]["notifyParties"][0][field] = value
    assert host.target_fact_components(source, paths) == tuple((p,) for p in paths)
    with pytest.raises(ValueError, match="independently mutable"):
        host.validate_repeated_binding_fact_topology(drafts=drafts, source_target=source)


def test_two_commercial_endpoints_and_carrier_are_not_merged_by_value_equality():
    source, _, _, drafts = fixture()
    parties = source["documentPatch"]["parties"]
    parties["shipper"] = deepcopy(parties["consignee"])
    parties["carrier"] = deepcopy(parties["consignee"])
    for roles in [
        ("shipper", "consignee"),
        ("carrier", "notifyParties[0]"),
        ("consignee", "notifyParties[0]"),
    ]:
        paths = tuple(f"documentPatch.parties.{r}.address" for r in roles)
        assert host.target_fact_components(source, paths) == tuple((p,) for p in paths)
        with pytest.raises(ValueError, match="independently mutable"):
            host.validate_repeated_binding_fact_topology(
                drafts=tuple(replace(d, target_paths=paths) for d in drafts), source_target=source
            )


def test_different_fields_of_same_identical_party_remain_independent():
    source, _, _, _ = fixture()
    for p in [
        source["documentPatch"]["parties"]["consignee"],
        source["documentPatch"]["parties"]["notifyParties"][0],
    ]:
        p["country"] = p["city"]
    paths = (
        "documentPatch.parties.consignee.city",
        "documentPatch.parties.notifyParties[0].country",
    )
    assert host.target_fact_components(source, paths) == tuple((p,) for p in paths)


def test_numeric_source_equality_never_becomes_party_identity():
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "grossWeight": {"value": 10.0, "unit": "kilogram"}},
                {"groupId": "g2", "grossWeight": {"value": 10.0, "unit": "kilogram"}},
            ]
        }
    }
    paths = tuple(f"documentPatch.cargoGroups[{i}].grossWeight.value" for i in range(2))
    assert host.target_fact_components(source, paths) == tuple((p,) for p in paths)
