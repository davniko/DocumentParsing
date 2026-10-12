"""Exact reviewed context omissions must not erase active extraction evidence."""

from copy import deepcopy

import pytest

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_ownership import build_owned_blueprint, ownership_surfaces
from document_ocr.synthesis.curated_templates import render_sampling_blueprint

NAME = "documentPatch.parties.shipper.name"


def omission_fixture(raw="ÉVIDENCE\nACME LTD\n919 PIECES\n", *, occurrences=("919",)):
    row = dict(
        documentId="reviewed",
        joinedRawText=raw,
        target={
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": None,
                "parties": {"shipper": {"name": "ACME LTD"}},
            },
        },
    )
    contract = SourceContract(
        variables=[
            dict(
                key="company",
                kind="name",
                value="ACME LTD",
                meaning="Shipper",
                required_literals=[],
                occurrences=[dict(text="ACME LTD", occurrence=1, presentation="text")],
            )
        ],
        targets=[dict(path=NAME, expression="{company}")],
        fixed_context="Reviewed omission fixture.",
    )
    binding_occurrences = []
    offsets = {}
    for text in occurrences:
        position = raw.index(text, offsets.get(text, 0))
        offsets[text] = position + len(text)
        start = len(raw[:position].encode())
        binding_occurrences.append(
            dict(byte_start=start, byte_end=start + len(text.encode()), source_text=text)
        )
    historical = dict(
        document_id=row["documentId"],
        source_sha256=digest(raw.encode()),
        bindings=[
            dict(
                logical_key="old_inner_count",
                target_paths=["documentPatch.cargoPackages[1].quantity"],
                value_kind="integer",
                occurrences=binding_occurrences,
            )
        ],
    )
    decision = dict(
        key="reviewed:obsolete_inner_count",
        source_sha256=digest(raw.encode()),
        quote="919 PIECES",
        count=raw.count("919 PIECES"),
        reason="Separately reviewed obsolete non-target cargo tally.",
    )
    return row, historical, contract, {"omit": [decision]}


def test_exact_omission_preserves_labels_source_bytes_and_unrelated_text():
    args = omission_fixture()
    snapshot = deepcopy(args)
    row, _, _, declarations = args
    blueprint = build_owned_blueprint(*args)
    assert "old_inner_count" not in blueprint.historical_bindings
    key = declarations["omit"][0]["key"]
    assert next(region for region in blueprint.regions if region.key == key).allow_empty
    surfaces = ownership_surfaces(blueprint, row["target"], None)
    rendered, labels, proof = render_sampling_blueprint(
        blueprint,
        row["target"],
        {},
        surface_values=surfaces,
    )
    assert rendered == "ÉVIDENCE\nACME LTD\n\n"
    assert labels == row["target"]
    assert [(edit["before"], edit["after"]) for edit in proof["edits"]] == [("919 PIECES", "")]
    assert blueprint.source == row["joinedRawText"]
    assert args == snapshot


def test_repeated_whole_owner_can_be_retired_only_when_every_copy_is_reviewed():
    args = omission_fixture("ACME LTD\n919 PIECES\n919 PIECES\n", occurrences=("919", "919"))
    blueprint = build_owned_blueprint(*args)
    surfaces = ownership_surfaces(blueprint, args[0]["target"], None)
    assert list(surfaces.values()) == [["", ""]]
    rendered, labels, _ = render_sampling_blueprint(
        blueprint, args[0]["target"], {}, surface_values=surfaces
    )
    assert rendered == "ACME LTD\n\n\n"
    assert labels == args[0]["target"]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("source_sha256", "0" * 64, "hash differs"),
        ("count", 2, "count differs"),
        ("count", True, "positive occurrence"),
        ("count", 0, "positive occurrence"),
        ("quote", "absent", "count differs"),
        ("reason", " ", "nonempty text"),
        ("key", "old_inner_count", "duplicates"),
    ],
)
def test_omission_declaration_drift_is_an_error(field, value, error):
    row, history, contract, declarations = omission_fixture()
    declarations["omit"][0][field] = value
    with pytest.raises(ValueError, match=error):
        build_owned_blueprint(row, history, contract, declarations)


def test_omission_cannot_consume_active_company_or_partial_historical_region():
    row, history, contract, declarations = omission_fixture()
    declarations["omit"][0]["quote"] = "ACME LTD"
    with pytest.raises(ValueError, match="active or partly enclosed"):
        build_owned_blueprint(row, history, contract, declarations)
    row, history, contract, declarations = omission_fixture(occurrences=("919 PIECES",))
    declarations["omit"][0]["quote"] = "919"
    with pytest.raises(ValueError, match="active or partly enclosed"):
        build_owned_blueprint(row, history, contract, declarations)


def test_omission_cannot_retire_an_owner_repeated_outside_its_reviewed_spans():
    args = omission_fixture("ACME LTD\n919 PIECES\nOTHER 919\n", occurrences=("919", "919"))
    with pytest.raises(ValueError, match="occurrence outside"):
        build_owned_blueprint(*args)


def test_omission_cannot_retire_existing_measure_rendering_dependency():
    row, history, contract, declarations = omission_fixture()
    declarations["surfaces"] = {"old_inner_count": {"expression": "10"}}
    with pytest.raises(ValueError, match="active rendering dependency"):
        build_owned_blueprint(row, history, contract, declarations)


def test_omission_cannot_retire_a_target_parent_owner():
    row, history, contract, declarations = omission_fixture()
    history["bindings"][0]["target_paths"] = ["documentPatch.parties.shipper"]
    with pytest.raises(ValueError, match="active rendering dependency"):
        build_owned_blueprint(row, history, contract, declarations)


def test_overlapping_or_malformed_omission_declarations_are_rejected():
    row, history, contract, declarations = omission_fixture()
    duplicate = {**declarations["omit"][0], "key": "another_omission"}
    declarations["omit"].append(duplicate)
    with pytest.raises(ValueError, match="spans overlap"):
        build_owned_blueprint(row, history, contract, declarations)
    for malformed in ({}, {"omit": {}}, {"omit": [{"extra": True}]}):
        if not malformed:
            assert (
                build_owned_blueprint(row, history, contract, malformed).source
                == row["joinedRawText"]
            )
        else:
            with pytest.raises(ValueError, match="source omission"):
                build_owned_blueprint(row, history, contract, malformed)
