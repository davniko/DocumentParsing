"""Reviewed fused name/postal boundaries retain exact source/replay ownership."""

from copy import deepcopy

import pytest

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_ownership import build_owned_blueprint
from document_ocr.synthesis.curated_templates import render_sampling_blueprint


def fixture(*, gap="", enabled=True, separator="\n"):
    raw = f"ÉDITÉ\nSHIPPER\nACME LTD{gap}12 ROAD\n"
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": "non_negotiable",
            "parties": {"shipper": {"name": "ACME LTD", "addressLine": "12 ROAD"}},
        },
    }
    row = {"documentId": "separator-source", "joinedRawText": raw, "target": target}
    historical = {
        "document_id": row["documentId"],
        "source_sha256": digest(raw.encode()),
        "bindings": [],
    }
    contract = SourceContract.model_validate(
        {
            "variables": [
                {
                    "key": "name",
                    "kind": "name",
                    "value": "ACME LTD",
                    "meaning": "shipper name",
                    "required_literals": [],
                    "occurrences": [{"text": "ACME LTD", "occurrence": 1, "presentation": "text"}],
                },
                {
                    "key": "postal",
                    "kind": "postal",
                    "value": "12 ROAD",
                    "meaning": "shipper postal",
                    "required_literals": [],
                    "occurrences": [{"text": "12 ROAD", "occurrence": 1, "presentation": "text"}],
                },
            ],
            "targets": [
                {"path": "documentPatch.parties.shipper.name", "expression": "{name}"},
                {"path": "documentPatch.parties.shipper.addressLine", "expression": "{postal}"},
            ],
            "fixed_context": "Reviewed fused party-field fixture",
        }
    )
    ownership = {}
    if enabled:
        ownership["party_field_separators"] = {
            "source_sha256": digest(raw.encode()),
            "boundaries": [
                {
                    "party_path": "documentPatch.parties.shipper",
                    "left": {"key": "name", "occurrence": 1},
                    "right": {"key": "postal", "occurrence": 1},
                    "separator": separator,
                }
            ],
        }
    return row, historical, contract, ownership


@pytest.mark.parametrize("separator", [" ", "\n"])
@pytest.mark.parametrize("changed", ["none", "name", "postal", "both"])
def test_reviewed_separator_preserves_labels_and_source_replay(separator, changed):
    row, historical, contract, ownership = fixture(separator=separator)
    blueprint = build_owned_blueprint(row, historical, contract, ownership)
    sampled = deepcopy(row["target"])
    values = {}
    if changed in {"name", "both"}:
        sampled["documentPatch"]["parties"]["shipper"]["name"] = "NEW CO"
        values["name"] = "NEW CO"
    if changed in {"postal", "both"}:
        sampled["documentPatch"]["parties"]["shipper"]["addressLine"] = "99 LANE"
        values["postal"] = "99 LANE"
    raw, target, proof = render_sampling_blueprint(blueprint, sampled, values)
    assert target == sampled
    party = sampled["documentPatch"]["parties"]["shipper"]
    joiner = "" if changed == "none" else separator
    assert raw == f"ÉDITÉ\nSHIPPER\n{party['name']}{joiner}{party['addressLine']}\n"
    replay = row["joinedRawText"].encode()
    for edit in reversed(proof["edits"]):
        assert replay[edit["byteStart"] : edit["byteEnd"]].decode() == edit["before"]
        replay = replay[: edit["byteStart"]] + edit["after"].encode() + replay[edit["byteEnd"] :]
    assert replay.decode() == raw
    if changed == "none":
        assert raw == row["joinedRawText"]
        assert not proof["edits"]
    else:
        postal_edit = next(edit for edit in proof["edits"] if edit["key"] == "postal")
        assert postal_edit["after"].startswith(separator)
    assert blueprint.source == row["joinedRawText"]


def test_undeclared_adjacent_fields_are_not_rewritten():
    row, historical, contract, ownership = fixture(enabled=False)
    blueprint = build_owned_blueprint(row, historical, contract, ownership)
    sampled = deepcopy(row["target"])
    sampled["documentPatch"]["parties"]["shipper"]["name"] = "NEW CO"
    raw, _, _ = render_sampling_blueprint(blueprint, sampled, {"name": "NEW CO"})
    assert raw == "ÉDITÉ\nSHIPPER\nNEW CO12 ROAD\n"


@pytest.mark.parametrize(
    "control",
    [
        "stale_hash",
        "cross_party",
        "unknown_owner",
        "wrong_occurrence",
        "bool_occurrence",
        "non_whitespace",
        "bad_separator_type",
        "wrong_kind",
        "duplicate",
        "unknown_option",
        "empty_boundaries",
        "null_declaration",
    ],
)
def test_bad_boundary_contracts_fail_during_source_compilation(control):
    row, historical, contract, ownership = fixture()
    declaration = ownership["party_field_separators"]
    boundary = declaration["boundaries"][0]
    if control == "stale_hash":
        declaration["source_sha256"] = "0" * 64
    elif control == "cross_party":
        boundary["party_path"] = "documentPatch.parties.consignee"
    elif control == "unknown_owner":
        boundary["right"]["key"] = "missing"
    elif control == "wrong_occurrence":
        boundary["right"]["occurrence"] = 2
    elif control == "bool_occurrence":
        boundary["right"]["occurrence"] = True
    elif control == "non_whitespace":
        boundary["separator"] = "; "
    elif control == "bad_separator_type":
        boundary["separator"] = []
    elif control == "wrong_kind":
        boundary["left"]["key"] = "postal"
    elif control == "duplicate":
        declaration["boundaries"].append(deepcopy(boundary))
    elif control == "unknown_option":
        declaration["guess"] = True
    elif control == "empty_boundaries":
        declaration["boundaries"] = []
    elif control == "null_declaration":
        ownership["party_field_separators"] = None
    with pytest.raises(ValueError, match="party-field"):
        build_owned_blueprint(row, historical, contract, ownership)


def test_already_separated_source_is_not_an_adjacent_boundary():
    row, historical, contract, ownership = fixture(gap="\n")
    with pytest.raises(ValueError, match="adjacent"):
        build_owned_blueprint(row, historical, contract, ownership)
