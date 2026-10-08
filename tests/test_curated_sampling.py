"""Instruction policy, exact quota feasibility and source-preserving synthesis."""

from copy import deepcopy
from itertools import product

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.synthesis.curated_sampling import (
    instruction_stratum,
    source_variant_counts,
    validate_instruction_inheritance,
)
from document_ocr.training.tasks import get_training_task


def label(neg=False, ref=False):
    return {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": "negotiable" if neg else "non_negotiable",
            "parties": {
                "consignee": {"name": "BANK"},
                "notifyParties": [
                    {"sameAs": "consignee"} if ref else {"name": "IMPORTER", "sameAs": None}
                ],
            },
        },
    }


@pytest.mark.parametrize("neg,ref", list(product((False, True), repeat=2)))
def test_decisions_round_trip_through_extraction_training_and_prompt(neg, ref):
    import json

    value = label(neg, ref)
    model = BillOfLadingExtractionV7Label.model_validate_json(json.dumps(value))
    assert model.canonical_target() == value
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    assert task.canonicalize(value) == value
    Draft202012Validator(json.loads(task.prompt_schema_json())).validate(value)
    wrong_role_null = deepcopy(value)
    wrong_role_null["documentPatch"]["parties"]["consignee"]["sameAs"] = None
    assert not Draft202012Validator(json.loads(task.prompt_schema_json())).is_valid(wrong_role_null)
    for path in ("negotiability", "sameAs"):
        broken = deepcopy(value)
        owner = (
            broken["documentPatch"]
            if path == "negotiability"
            else broken["documentPatch"]["parties"]["notifyParties"][0]
        )
        owner.pop(path)
        with pytest.raises(ValidationError):
            BillOfLadingExtractionV7Label.model_validate_json(json.dumps(broken))
    unknown = deepcopy(value)
    unknown["documentPatch"]["negotiability"] = None
    assert task.canonicalize(unknown) == unknown
    Draft202012Validator(json.loads(task.prompt_schema_json())).validate(unknown)


def test_quota_allocation_exhaustively_matches_feasible_small_integer_margins():
    keys = list(product((False, True), repeat=2))
    for mask in range(1, 16):
        rows = {str(i): {"target": label(*key)} for i, key in enumerate(keys) if mask & (1 << i)}
        available = {instruction_stratum(r["target"]) for r in rows.values()}
        for n in (1, 3, 7):
            for neg, ref in product(range(n + 1), repeat=2):
                feasible = any(
                    all(
                        count == 0 or key in available
                        for key, count in {
                            (True, True): joint,
                            (True, False): neg - joint,
                            (False, True): ref - joint,
                            (False, False): n - neg - ref + joint,
                        }.items()
                    )
                    for joint in range(max(0, neg + ref - n), min(neg, ref) + 1)
                )
                config = {
                    "seed": 42,
                    "source_ids": list(rows),
                    "variants_per_source": 2,
                    "template_sampling": {
                        "samples": n,
                        "negotiable_fraction": neg / n,
                        "notify_reference_fraction": ref / n,
                    },
                }
                if not feasible:
                    with pytest.raises(ValueError, match="unavailable"):
                        source_variant_counts(config, rows)
                    continue
                counts = source_variant_counts(config, rows)
                assert sum(counts.values()) == n
                assert (
                    sum(
                        c
                        for sid, c in counts.items()
                        if instruction_stratum(rows[sid]["target"])[0]
                    )
                    == neg
                )
                assert (
                    sum(
                        c
                        for sid, c in counts.items()
                        if instruction_stratum(rows[sid]["target"])[1]
                    )
                    == ref
                )
                assert counts == source_variant_counts(config, dict(reversed(list(rows.items()))))


def test_uniform_sampling_and_instruction_inheritance():
    rows = {"a": {"target": label(True, False)}, "b": {"target": label(False, True)}}
    config = {"seed": 7, "source_ids": list(rows), "variants_per_source": 3}
    assert source_variant_counts(config, rows) == {"a": 3, "b": 3}
    source = label(True, True)
    changed = deepcopy(source)
    changed["documentPatch"]["parties"]["consignee"]["name"] = "NEW BANK"
    validate_instruction_inheritance(source, changed)
    changed["documentPatch"]["negotiability"] = "non_negotiable"
    with pytest.raises(ValueError, match="negotiability differs"):
        validate_instruction_inheritance(source, changed)
    changed = deepcopy(source)
    changed["documentPatch"]["parties"]["notifyParties"] = [{"name": "BUYER", "sameAs": None}]
    with pytest.raises(ValueError, match="notify references differ"):
        validate_instruction_inheritance(source, changed)

    unknown = label()
    unknown["documentPatch"]["negotiability"] = None
    unknown["documentPatch"]["parties"].pop("consignee")
    assert instruction_stratum(unknown) == (False, False)
    validate_instruction_inheritance(unknown, deepcopy(unknown))
    for value in ("non_negotiable", "negotiable"):
        changed = deepcopy(unknown)
        changed["documentPatch"]["negotiability"] = value
        with pytest.raises(ValueError, match="negotiability differs"):
            validate_instruction_inheritance(unknown, changed)
    missing = deepcopy(unknown)
    missing["documentPatch"].pop("negotiability")
    with pytest.raises(ValueError, match="explicit source negotiability"):
        instruction_stratum(missing)
    with pytest.raises(ValueError, match="negotiability differs"):
        validate_instruction_inheritance(unknown, missing)
    rows["u"] = {"target": unknown}
    config["source_ids"].append("u")
    config["template_sampling"] = {"samples": 9, "negotiable_fraction": 1 / 3}
    counts = source_variant_counts(config, rows)
    assert counts == {"a": 3, "b": 3, "u": 3}
    assert rows["u"]["target"] == unknown


@pytest.mark.parametrize("axis", [None, "negotiable_fraction", "notify_reference_fraction"])
def test_optional_margins_preserve_source_diversity_and_exact_total(axis):
    rows = {
        str(i): {"target": label(*key)} for i, key in enumerate(product((False, True), repeat=2))
    }
    policy = {"samples": 7}
    if axis:
        policy[axis] = 0.5
    config = {
        "seed": 2,
        "source_ids": list(rows),
        "variants_per_source": 2,
        "template_sampling": policy,
    }
    counts = source_variant_counts(config, rows)
    assert set(counts) == set(rows)
    assert sum(counts.values()) == 7
    if axis:
        column = 0 if axis == "negotiable_fraction" else 1
        assert (
            sum(n for sid, n in counts.items() if instruction_stratum(rows[sid]["target"])[column])
            == 4
        )
    else:
        assert max(counts.values()) - min(counts.values()) <= 1


def test_unavailable_single_margin_fails_and_null_is_not_a_reference():
    rows = {"a": {"target": label(False, False)}}
    assert instruction_stratum(rows["a"]["target"]) == (False, False)
    config = {
        "seed": 1,
        "source_ids": ["a"],
        "variants_per_source": 2,
        "template_sampling": {"samples": 10, "notify_reference_fraction": 0.5},
    }
    with pytest.raises(ValueError, match="unavailable"):
        source_variant_counts(config, rows)
