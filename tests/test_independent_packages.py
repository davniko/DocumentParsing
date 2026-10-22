"""Production flat-choice contracts, native constraints and final-only coverage."""

from __future__ import annotations

import copy
import inspect
import itertools
import json
from collections import Counter

import pytest
from pydantic import ValidationError

from document_ocr.synthesis import independent_packages as packages
from document_ocr.synthesis import independent_relationships as relationships
from document_ocr.synthesis.independent_envelopes import PACKAGE_COUNTING_CONTEXTS
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES


def request(candidate="PACKAGE_BALE"):
    value = {
        "goods": [{"hs6": "520100", "description": "Cotton, not carded or combed"}],
        "regime": "ambient",
        "equipment": "Maritime general-purpose container",
        "form": "loose unprocessed cotton fibre",
    }
    if candidate is not None:
        value["requiredCategory"] = candidate
    return value


def answer(category="PACKAGE_BALE", *, context=None, compatible=True):
    return {
        "alternatives": []
        if category is None
        else [{"packageType": category, "declaredUnitContext": context}],
        "goodsForm": "loose cotton fibre",
        "candidateCompatible": compatible,
    }


def private_request(counted="PACKAGE_PALLET", allowed=None):
    value = request(None)
    value.update(
        countedCategory=counted,
        allowedRelations=allowed or ["NONE", "CONTAINS", "SUPPORTS", "ON_SUPPORT"],
    )
    return value


def private_plan(relation="SUPPORTS", related="PACKAGE_CARTON"):
    return {
        "privatePackingRelation": relation,
        "relatedPackageType": related,
        "reason": "One pallet per drum: this invented ratio must remain audit-only.",
    }


def select(alternatives, **changes):
    kwargs = {
        "targets": {"PACKAGE_BALE": 5, "PACKAGE_SET": 5, "PACKAGE_DRUM": 5},
        "accepted": {},
        "reserved": {},
        "used": set(),
        "seed": 17,
        "key": "first",
    }
    kwargs.update(changes)
    return packages.select_package_type(alternatives, **kwargs)


def test_vocabulary_is_exact48_without_analysis_or_artifact_imports():
    assert set(packages.PACKAGE_TYPES) == set(PACKAGE_ALIASES)
    assert len(packages.PACKAGE_TYPES) == 48
    for module in (packages, relationships):
        source = inspect.getsource(module)
        assert "docs/analysis" not in source
        assert "importlib" not in source
        assert ".read_text(" not in source
        assert ".read_bytes(" not in source


@pytest.mark.parametrize("category", packages.PACKAGE_TYPES)
def test_all48_complete_native_alternatives_preserve_exact_context(category):
    contexts = PACKAGE_COUNTING_CONTEXTS[category] or {None}
    for context in contexts:
        value = packages.validate_package_choice(
            request(category), answer(category, context=context)
        )
        assert value.package_types == (category,)
        assert dict(value.counting_contexts) == ({} if context is None else {category: context})
        assert not value.unknown
        schema = packages.package_choice_output_type(request(category)).model_json_schema()
        assert schema["properties"]["candidateCompatible"]["type"] == "boolean"


@pytest.mark.parametrize("category", sorted(k for k, v in PACKAGE_COUNTING_CONTEXTS.items() if v))
def test_sensitive_context_cannot_be_missing_or_detached_from_category(category):
    with pytest.raises(ValidationError):
        packages.validate_package_choice(request(category), answer(category))
    with pytest.raises(ValidationError):
        packages.validate_package_choice(request(category), answer(category, context="invented"))


def test_ordinary_category_does_not_accept_a_sensitive_context():
    with pytest.raises(ValidationError):
        packages.validate_package_choice(request(), answer(context="vehicle"))


@pytest.mark.parametrize("candidate", [None, "PACKAGE_BALE"])
def test_native_candidate_presence_is_request_conditioned_not_prompt_only(candidate):
    model = packages.package_choice_output_type(request(candidate))
    good = answer(compatible=None if candidate is None else True)
    assert model.model_validate(good)
    bad = {**good, "candidateCompatible": False if candidate is None else None}
    with pytest.raises(ValidationError):
        model.model_validate(bad)
    # The exact failure observed for an unforced tiles response is impossible
    # under this model's native JSON schema, not patched after response receipt.
    if candidate is None:
        assert model.model_json_schema()["properties"]["candidateCompatible"]["type"] == "null"


@pytest.mark.parametrize("compatible", [True, False])
def test_candidate_judgment_and_shortlist_cannot_disagree(compatible):
    value = answer("PACKAGE_SET" if compatible else "PACKAGE_BALE", compatible=compatible)
    with pytest.raises(ValueError, match="complete alternative disagree"):
        packages.validate_package_choice(request(), value)


def test_explicit_unknown_is_preserved_and_cannot_be_selected():
    result = packages.validate_package_choice(request(), answer(None, compatible=False))
    assert result.unknown and result.package_types == ()
    with pytest.raises(ValueError, match="nonempty"):
        select(result.package_types)
    with pytest.raises(ValueError, match="complete alternative disagree"):
        packages.validate_package_choice(request(), answer(None, compatible=True))


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "too_many", "bad_type", "empty_form"])
def test_native_output_failure_modes(mutation):
    value = answer()
    if mutation == "duplicate":
        value["alternatives"] *= 2
    elif mutation == "extra":
        value["packageQuantity"] = 12
    elif mutation == "too_many":
        value["alternatives"] = [
            {"packageType": category, "declaredUnitContext": None}
            for category in packages.PACKAGE_TYPES[:9]
        ]
    elif mutation == "bad_type":
        value["candidateCompatible"] = "true"
    else:
        value["goodsForm"] = "   "
    with pytest.raises(ValueError):
        packages.validate_package_choice(request(), value)


def test_factual_projection_excludes_generated_form_and_does_not_mutate_answer():
    value = packages.validate_package_choice(request(), answer())
    before = value.model_dump()
    facts = packages.package_choice_facts(value)
    assert facts == {"alternatives": before["alternatives"], "candidateCompatible": True}
    assert value.model_dump() == before
    with pytest.raises(TypeError):
        value.counting_contexts["PACKAGE_BALE"] = "invented"


def test_prompt_uses_all_supplied_goods_and_exact_caller_registry_membership():
    value = request()
    value["goods"].append({"hs6": "510121", "description": "Degreased shorn wool"})
    text = packages.package_choice_prompt(value, hs_codes={"520100", "510121"})
    assert "HS 520100: Cotton" in text and "HS 510121: Degreased shorn wool" in text
    assert all(category + ": " in text for category in packages.PACKAGE_TYPES)
    assert "Candidate package to judge independently: PACKAGE_BALE" in text
    with pytest.raises(ValueError, match="supplied HS registry"):
        packages.package_choice_prompt(value, hs_codes={"520100"})
    assert "No quotas are supplied" in packages.PACKAGE_CHOICE_SYSTEM
    assert "do not invent a" in packages.PACKAGE_CHOICE_SYSTEM
    assert "ordinary fluid or bulk goods into a SET" in packages.PACKAGE_CHOICE_SYSTEM


@pytest.mark.parametrize(
    "field,value",
    [
        ("goods", []),
        ("goods", [{"hs6": 520100, "description": "Cotton"}]),
        ("goods", [{"hs6": "5201", "description": "Cotton"}]),
        ("goods", [{"hs6": "520100", "description": " "}]),
        ("equipment", " "),
        ("regime", "invented"),
        ("requiredCategory", "SET"),
        ("sourceDocumentId", "donor-source"),
        ("form", ""),
        ("dangerousGoods", {"unNumber": "1230"}),
    ],
)
def test_malformed_or_source_bound_requests_fail_before_prompt(field, value):
    item = request()
    item[field] = value
    with pytest.raises(ValueError):
        packages.package_choice_prompt(item, hs_codes={"520100"})


def test_dg_requires_supplied_identity_and_disallows_nondg_liquid_context():
    value = request("PACKAGE_FLEXITANK")
    value["regime"] = "dg"
    with pytest.raises(ValueError, match="genuine DG identity"):
        packages.PackageChoiceRequest.model_validate(value)
    value["dangerousGoods"] = {"unNumber": "1230", "hazardClass": "3", "packingGroup": "II"}
    with pytest.raises(ValueError, match="non-DG"):
        packages.validate_package_choice(
            value, answer("PACKAGE_FLEXITANK", context="non_dg_liquid_system")
        )


def test_nested_dg_prompt_bytes_survive_sorted_json_persistence():
    value = request("PACKAGE_DRUM_STEEL")
    value.update(
        goods=[{"hs6": "290511", "description": "Methanol"}],
        regime="dg",
        dangerousGoods={
            "recordId": "hmt_" + "a" * 64,
            "unNumber": "1230",
            "properShippingName": "Methanol",
            "class": "3",
            "packingGroup": "II",
            "subsidiaryHazards": ["6.1"],
        },
    )
    original_order = list(value["dangerousGoods"])
    stored = json.loads(json.dumps(value, sort_keys=True))
    assert list(stored["dangerousGoods"]) != original_order
    assert packages.package_choice_prompt(
        value, hs_codes={"290511"}
    ) == packages.package_choice_prompt(stored, hs_codes={"290511"})
    private = {key: item for key, item in value.items() if key != "requiredCategory"}
    private.update(countedCategory="PACKAGE_DRUM_STEEL", allowedRelations=["NONE", "ON_SUPPORT"])
    persisted = json.loads(json.dumps(private, sort_keys=True))
    assert relationships.private_packing_prompt(
        private, hs_codes={"290511"}
    ) == relationships.private_packing_prompt(persisted, hs_codes={"290511"})


@pytest.mark.parametrize(
    "category,expected",
    [
        ("PACKAGE_BAG_PAPER_MULTI_WALL", "PAPER"),
        ("PACKAGE_BAG_PAPER", "PAPER"),
        ("PACKAGE_BAG_PLASTIC", "PLASTIC"),
        ("PACKAGE_BOX_FIBREBOARD", "FIBREBOARD"),
        ("PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY", "WOOD"),
        ("PACKAGE_CASE_WOODEN", "WOOD"),
        ("PACKAGE_CRATE_WOODEN", "WOOD"),
        ("PACKAGE_PALLET_WOODEN", "WOOD"),
        ("PACKAGE_DRUM_FIBRE", "FIBRE"),
        ("PACKAGE_DRUM_STEEL", "STEEL"),
        ("PACKAGE_DRUM_PLASTIC", "PLASTIC"),
        ("PACKAGE_DRUM", None),
        ("PACKAGE_BALE", None),
        ("PACKAGE_SET", None),
    ],
)
def test_material_authority_is_exact_category_not_cargo_or_substring(category, expected):
    assert packages.package_material(category) == expected
    with pytest.raises(ValueError):
        packages.package_material(category + "_UNKNOWN")


def test_deficit_policy_is_soft_approved_only_and_reservations_not_accepted():
    assert select(["PACKAGE_BALE", "PACKAGE_SET"], accepted={"PACKAGE_BALE": 5}) == "PACKAGE_SET"
    assert select(["PACKAGE_DRUM_STEEL"], targets={"PACKAGE_DRUM": 50}) == "PACKAGE_DRUM_STEEL"
    assert select(["PACKAGE_BALE", "PACKAGE_SET"], reserved={"PACKAGE_BALE": 5}) == "PACKAGE_SET"
    assert select(["PACKAGE_SET", "PACKAGE_BALE"], used={"PACKAGE_SET"}) == "PACKAGE_BALE"
    assert packages.realized_package_counts({}) == (Counter(), 0)


def test_selection_is_order_independent_and_deterministic_without_input_mutation():
    options = ["PACKAGE_BALE", "PACKAGE_SET", "PACKAGE_DRUM"]
    expected = select(options)
    for order in itertools.permutations(options):
        assert select(order) == expected
    assert options == ["PACKAGE_BALE", "PACKAGE_SET", "PACKAGE_DRUM"]


@pytest.mark.parametrize("mapping", ["targets", "accepted", "reserved"])
@pytest.mark.parametrize("count", [-1, True, 1.0, "1"])
def test_bad_counting_state_is_not_silently_defaulted(mapping, count):
    with pytest.raises(ValueError, match="nonnegative integers"):
        select(["PACKAGE_SET"], **{mapping: {"PACKAGE_SET": count}})


@pytest.mark.parametrize(
    "options,changes",
    [
        ([], {}),
        (["PACKAGE_SET", "PACKAGE_SET"], {}),
        (["UNKNOWN"], {}),
        (["PACKAGE_SET"], {"used": {"UNKNOWN"}}),
        (["PACKAGE_SET"], {"seed": True}),
        (["PACKAGE_SET"], {"key": ""}),
        (["PACKAGE_SET"], {"targets": {"UNKNOWN": 1}}),
    ],
)
def test_invalid_policy_inputs_are_explicit_errors(options, changes):
    with pytest.raises(ValueError):
        select(options, **changes)


def test_realized_counts_require_final_evidence_and_keep_specificity_shortfall():
    outcome = {
        "a": {
            "status": "accepted",
            "finalCategory": "PACKAGE_DRUM_STEEL",
            "candidateSha256": "a" * 64,
            "reviewEvidence": "complete-final-review/a",
        },
        "b": {"status": "hold", "reason": "material inconsistency"},
    }
    before = copy.deepcopy(outcome)
    accepted, held = packages.realized_package_counts(outcome)
    assert accepted == {"PACKAGE_DRUM_STEEL": 1} and held == 1
    assert packages.package_coverage({"PACKAGE_DRUM": 5}, accepted) == {
        "PACKAGE_DRUM": {"target": 5, "acceptedFinal": 0, "shortfall": 5, "surplus": 0}
    }
    assert (
        packages.package_coverage({"PACKAGE_DRUM_STEEL": 0}, accepted)["PACKAGE_DRUM_STEEL"][
            "surplus"
        ]
        == 1
    )
    assert outcome == before
    with pytest.raises(ValueError, match="cannot count twice"):
        packages.realized_package_counts({"a": outcome["a"], "b": outcome["a"]})


@pytest.mark.parametrize(
    "value",
    [
        {"status": "reserved"},
        {"status": "hold", "reason": ""},
        {"status": "accepted", "finalCategory": "PACKAGE_SET"},
        {
            "status": "accepted",
            "finalCategory": "PACKAGE_SET",
            "candidateSha256": "G" * 64,
            "reviewEvidence": "review",
        },
    ],
)
def test_nonfinal_or_unwitnessed_results_cannot_inflate_counts(value):
    with pytest.raises(ValueError):
        packages.realized_package_counts({"key": value})


@pytest.mark.parametrize("category", packages.PACKAGE_TYPES)
def test_private_none_supports_every_exact_flat_category(category):
    value = relationships.validate_private_packing_plan(
        private_request(category), private_plan("NONE", None)
    )
    assert relationships.private_packing_facts(value) == {
        "privatePackingRelation": "NONE",
        "relatedPackageType": None,
    }


@pytest.mark.parametrize(
    "platform", ["PACKAGE_PALLET", "PACKAGE_PALLET_WOODEN", "PACKAGE_SKID", "PACKAGE_RACK"]
)
def test_all_open_platform_forms_reject_containment_and_reversals(platform):
    with pytest.raises(ValidationError):
        relationships.validate_private_packing_plan(
            private_request(platform), private_plan("CONTAINS", "PACKAGE_BOX")
        )
    with pytest.raises(ValueError, match="reverses"):
        relationships.validate_private_packing_plan(
            private_request("PACKAGE_BAG"), private_plan("SUPPORTS", platform)
        )
    with pytest.raises(ValueError, match="reverses"):
        relationships.validate_private_packing_plan(
            private_request(platform), private_plan("ON_SUPPORT", "PACKAGE_BOX")
        )
    assert relationships.validate_private_packing_plan(
        private_request(platform), private_plan("SUPPORTS", "PACKAGE_BOX")
    )
    assert relationships.validate_private_packing_plan(
        private_request("PACKAGE_DRUM_STEEL"), private_plan("ON_SUPPORT", platform)
    )


def test_private_unknown_is_retained_but_never_operative_or_implicit_none():
    value = relationships.validate_private_packing_plan(
        private_request(), private_plan("UNKNOWN", None)
    )
    assert value.unknown
    with pytest.raises(ValueError, match="unresolved"):
        relationships.private_packing_facts(value)
    with pytest.raises(ValueError):
        relationships.validate_private_packing_plan(
            private_request(), private_plan("NONE", "PACKAGE_BOX")
        )


def test_source_capability_limits_native_schema_and_operational_validation():
    item = private_request("PACKAGE_CRATE", ["NONE", "CONTAINS"])
    model = relationships.private_packing_output_type(item)
    assert set(model.model_json_schema()["properties"]["privatePackingRelation"]["enum"]) == {
        "NONE",
        "CONTAINS",
        "UNKNOWN",
    }
    with pytest.raises(ValidationError):
        relationships.validate_private_packing_plan(
            item, private_plan("ON_SUPPORT", "PACKAGE_PALLET")
        )
    assert relationships.validate_private_packing_plan(
        item, private_plan("CONTAINS", "PACKAGE_BAG_PLASTIC")
    )


@pytest.mark.parametrize(
    "category", ["PACKAGE_PALLET", "PACKAGE_PALLET_WOODEN", "PACKAGE_SKID", "PACKAGE_RACK"]
)
def test_native_schema_does_not_offer_enclosure_to_open_supports(category):
    item = private_request(category)
    model = relationships.private_packing_output_type(item)
    allowed = model.model_json_schema()["properties"]["privatePackingRelation"]["enum"]
    assert "CONTAINS" not in allowed
    assert {"NONE", "SUPPORTS", "UNKNOWN"} <= set(allowed)
    assert relationships.validate_private_packing_plan(
        item, private_plan("SUPPORTS", "PACKAGE_CARTON")
    )


def test_private_prompt_preserves_all_context_and_reason_is_audit_only():
    item = private_request()
    text = relationships.private_packing_prompt(item, hs_codes={"520100"})
    assert "Declared counted package (unchangeable): PACKAGE_PALLET" in text
    assert "Equipment: Maritime general-purpose container" in text
    assert "alternatives=[]" not in text
    with pytest.raises(ValueError, match="supplied HS registry"):
        relationships.private_packing_prompt(item, hs_codes=set())
    value = relationships.validate_private_packing_plan(item, private_plan())
    assert "reason" not in relationships.private_packing_facts(value)
    assert "One pallet per drum" not in json.dumps(relationships.private_packing_facts(value))
    assert "One pallet per drum" in value.reason


@pytest.mark.parametrize("allowed", [[], ["NONE", "NONE"], ["UNKNOWN"], ["INVENTED"]])
def test_missing_or_unreviewed_capability_does_not_fallback(allowed):
    value = private_request()
    value["allowedRelations"] = allowed
    with pytest.raises(ValueError):
        relationships.private_packing_prompt(value, hs_codes={"520100"})
