"""Reviewed identifier aliases must be explicit, scoped and replayable."""

from copy import deepcopy

import pytest
import yaml
from test_curated_auxiliary import scenario
from test_curated_templates import compile_sampling_blueprint, source_fixture

from document_ocr.synthesis.curated_auxiliary import (
    _render_recipe,
    _validate_recipe,
    augment_auxiliary_blueprint,
    auxiliary_surfaces,
    load_auxiliary_contract,
)
from document_ocr.synthesis.curated_templates import render_sampling_blueprint
from document_ocr.synthesis.generators import DeterministicStream


def render_identifier(original, *, canonical_identity=None, prefix="", seed=202610096):
    recipe = {"identifier": "registration", "prefix": prefix}
    if canonical_identity is not None:
        recipe["canonical_identity"] = canonical_identity
    _validate_recipe(recipe)
    return _render_recipe(
        recipe,
        scenario(),
        {"documentPatch": {}},
        DeterministicStream(seed, "identity-test", "source"),
        original,
        set(),
    )


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("643-668-997", "212880496"),
        ("643 668 997", "212880496"),
        ("643668977", "395405466"),
        ("81483", "496119"),
        ("CN91320282050282384L", "76331163051965853"),
    ],
)
def test_without_explicit_alias_original_deterministic_contract_is_unchanged(original, expected):
    assert render_identifier(original) == expected


@pytest.mark.parametrize("seed", [0, 1, 17, 202610096])
def test_reviewed_digit_aliases_share_canonical_seed_and_width(seed):
    expected = render_identifier("643668997", seed=seed)
    for original in ("643-668-997", "643 668 997", "643668977", "999"):
        assert render_identifier(original, canonical_identity="643668997", seed=seed) == expected
    assert len(expected) == 9
    assert render_identifier("643668977", seed=seed) != expected
    assert render_identifier("643668977", canonical_identity="81483", seed=seed) != expected
    assert (
        render_identifier("643668977", canonical_identity="643668997", prefix="VAT-", seed=seed)
        == "VAT-" + expected
    )


@pytest.mark.parametrize(
    "value", [None, "", " ", "643-668-997", "abc123", "ABC\n123", "É123", 643668997, True]
)
def test_invalid_canonical_identity_fails_contract_validation(value):
    with pytest.raises(ValueError, match="canonical identifier identity"):
        _validate_recipe({"identifier": "registration", "canonical_identity": value})


def test_canonical_identity_is_only_an_identifier_option():
    with pytest.raises(ValueError, match="unsupported auxiliary recipe options"):
        _validate_recipe({"text": "VAT", "canonical_identity": "643668997"})


def test_contract_loader_preserves_reviewed_identity_and_rejects_invalid_alias(tmp_path):
    path = tmp_path / "auxiliary.yaml"
    recipe = {"identifier": "registration", "canonical_identity": "643668997"}
    value = {
        "version": 1,
        "sources": {
            "sample": {
                "bindings": {"importer": recipe},
                "spans": [{"quote": "643668977", "count": 1, "value": recipe}],
            }
        },
    }
    path.write_text(yaml.safe_dump(value))
    assert load_auxiliary_contract(path) == value
    value["sources"]["sample"]["spans"][0]["value"] = {
        "identifier": "registration",
        "canonical_identity": "643 668 997",
    }
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="canonical identifier identity"):
        load_auxiliary_contract(path)


def test_owned_alias_render_replays_only_reviewed_importer_occurrences():
    row, contract, history = source_fixture()
    examples = {
        "consignee": "643-668-997",
        "notify": "643 668 997",
        "importer": "643668977",
        "exporter": "81483",
        "reference": "6436689772023090018",
    }
    for key, original in examples.items():
        prefix = key.upper() + " VAT: "
        offset = len(row["joinedRawText"].encode()) + len(prefix)
        row["joinedRawText"] += prefix + original + "\n"
        history["bindings"].append(
            {
                "logical_key": key,
                "target_paths": [],
                "value_kind": "identifier",
                "occurrences": [
                    {
                        "byte_start": offset,
                        "byte_end": offset + len(original),
                        "source_text": original,
                    }
                ],
            }
        )
    blueprint = compile_sampling_blueprint(row, history, contract)
    recipes = {key: {"identifier": "registration"} for key in examples}
    for key in ("consignee", "notify", "importer"):
        recipes[key]["canonical_identity"] = "643668997"
    declaration = {"sources": {"sample": {"bindings": recipes}}}
    unchanged = deepcopy(declaration)
    blueprint = augment_auxiliary_blueprint(blueprint, declaration)
    stream = DeterministicStream(202610096, "identity-test", "source")
    values, receipts = auxiliary_surfaces(blueprint, scenario(), row["target"], stream)
    assert values["consignee"] == values["notify"] == values["importer"] == "212880496"
    assert values["exporter"] == "496119" != values["importer"]
    assert values["reference"] == render_identifier(examples["reference"])
    rendered, labels, proof = render_sampling_blueprint(
        blueprint,
        row["target"],
        {},
        surface_values=values,
    )
    expected = row["joinedRawText"]
    for key, original in examples.items():
        expected = expected.replace(
            key.upper() + " VAT: " + original, key.upper() + " VAT: " + values[key]
        )
    assert rendered == expected
    assert labels == row["target"] and proof["unchangedBytesPreserved"]
    assert declaration == unchanged
    recorded = {r["key"]: r for r in receipts}
    for key in examples:
        assert recorded[key]["recipe"] == recipes[key]
        assert recorded[key]["before"] == [examples[key]]
        assert recorded[key]["after"] == [values[key]]
    assert auxiliary_surfaces(blueprint, scenario(), row["target"], stream) == (values, receipts)
