from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import labelled_context as context


def fixture():
    base = "documentPatch.parties.consignee"
    mark = "documentPatch.cargoGroups[0].marksAndNumbers"
    handling = "documentPatch.cargoGroups[0].handlingInstructions[0]"
    port = "documentPatch.route.portOfDischarge.name"
    target = {
        "documentPatch": {
            "parties": {"consignee": {"name": "OLD COMPANY", "city": "CAIRO", "country": "EGYPT"}},
            "route": {"portOfDischarge": {"name": "ALEXANDRIA"}},
            "cargoGroups": [
                {
                    "marksAndNumbers": ["OLD COMPANY.", "CAIRO, EGYPT"],
                    "handlingInstructions": ["CARGO IN TRANSIT TO KABARI EFS BONDED WAREHOUSE"],
                }
            ],
        }
    }

    def binding(key, path, source, paths=(), keys=()):
        return NS(
            logical_key=key,
            target_paths=(path,),
            render_mode="target_binding",
            derivation=None,
            dependency_paths=paths,
            dependency_bindings=keys,
            realization=NS(mode="single_surface"),
            occurrences=(NS(source_text=source),),
        )

    bindings = [
        binding(field, base + "." + field, value)
        for field, value in target["documentPatch"]["parties"]["consignee"].items()
    ]
    bindings.extend(
        [
            binding("port", port, "ALEXANDRIA"),
            binding("name_mark", mark + "[0]", "OLD COMPANY.", (base + ".name",), ("name",)),
            binding(
                "place_mark",
                mark + "[1]",
                "CAIRO, EGYPT",
                (base + ".city", base + ".country"),
                ("city", "country"),
            ),
            binding(
                "warehouse",
                handling,
                "CARGO IN TRANSIT TO KABARI EFS\nBONDED WAREHOUSE",
                (port,),
                ("port",),
            ),
        ]
    )
    return NS(template=NS(bindings=bindings), target=target)


def test_party_and_destination_dependencies_follow_generated_owners_without_new_fields():
    source = fixture()
    target = deepcopy(source.target)
    target["documentPatch"]["parties"]["consignee"] = {
        "name": "Pacific Trade Ltd.",
        "city": "Phnom Penh",
        "country": "Cambodia",
    }
    target["documentPatch"]["route"]["portOfDischarge"]["name"] = "Hong Kong"
    old_target = deepcopy(target)
    generated = context.generated_values(source, target)
    assert generated == {
        "documentPatch.cargoGroups[0].marksAndNumbers[0]": "PACIFIC TRADE LTD.",
        "documentPatch.cargoGroups[0].marksAndNumbers[1]": "PHNOM PENH, CAMBODIA",
        "documentPatch.cargoGroups[0].handlingInstructions[0]": (
            "CARGO IN TRANSIT TO HONG KONG BONDED WAREHOUSE"
        ),
    }
    assert context.owned_paths(source) == frozenset(generated)
    assert target == old_target  # No hidden mutation or source-label enrichment.
    assert source.target["documentPatch"]["parties"]["consignee"]["country"] == "EGYPT"


def test_identical_marks_without_compiler_dependency_do_not_infer_party_ownership():
    source = fixture()
    for binding in source.template.bindings:
        binding.dependency_paths = ()
        binding.dependency_bindings = ()
    assert context.generated_values(source, source.target) == {}


@pytest.mark.parametrize("keys", [(), ("absent",), ("name", "country")])
def test_missing_or_wrong_owner_binding_is_not_a_source_fallback(keys):
    source = fixture()
    source.template.bindings[4].dependency_bindings = keys
    with pytest.raises(ValueError, match=r"owner bind|paths and declared"):
        context.owned_paths(source)


def test_source_name_must_prove_the_declared_party_relationship():
    source = fixture()
    source.target["documentPatch"]["parties"]["consignee"]["name"] = "ANOTHER COMPANY"
    with pytest.raises(ValueError, match="does not match its declared"):
        context.validate_source(source.template.bindings, source.target)


def test_extra_unowned_locality_words_are_not_silently_erased():
    source = fixture()
    source.target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"][1] = (
        "CAIRO, EGYPT VIA LONDON"
    )
    source.template.bindings[5].occurrences[0].source_text = "CAIRO, EGYPT VIA LONDON"
    with pytest.raises(ValueError, match="unowned semantic"):
        context.owned_paths(source)


def test_ambiguous_overlapping_city_country_components_require_review():
    source = fixture()
    source.target["documentPatch"]["parties"]["consignee"].update(
        city="Singapore", country="Singapore"
    )
    source.target["documentPatch"]["cargoGroups"][0]["marksAndNumbers"][1] = "Singapore, Singapore"
    source.template.bindings[5].occurrences[0].source_text = "Singapore, Singapore"
    with pytest.raises(ValueError, match="unique visible"):
        context.owned_paths(source)


def test_missing_sampled_owner_raises_instead_of_restoring_old_label():
    source = fixture()
    target = deepcopy(source.target)
    target["documentPatch"]["parties"]["consignee"]["city"] = ""
    with pytest.raises(ValueError, match="nonempty single-line"):
        context.generated_values(source, target)


def test_facility_contract_does_not_accept_unstructured_destination_prose():
    source = fixture()
    source.target["documentPatch"]["cargoGroups"][0]["handlingInstructions"][0] = (
        "DELIVER TO KABARI"
    )
    source.template.bindings[6].occurrences[0].source_text = "DELIVER TO KABARI"
    with pytest.raises(ValueError, match="transit/warehouse frame"):
        context.owned_paths(source)


def test_completion_rejects_independent_model_value_for_owned_mark(monkeypatch):
    from document_ocr.synthesis.template_compiler import complete_targets as targets

    source = fixture()
    monkeypatch.setattr(targets.lexical_partitions, "assembled_targets", lambda *args: {})
    request = {
        "key": "mark",
        "paths": ["documentPatch.cargoGroups[0].marksAndNumbers[0]"],
        "constraints": [],
    }
    with pytest.raises(ValueError, match="incorrectly requested from the model"):
        targets.complete_proposal(
            source, source.target, [request], {"mark": "INDEPENDENT COMPANY"}, sample_id="sample"
        )


def test_final_acceptance_rejects_stale_labels_without_mutating_target():
    from document_ocr.synthesis.template_compiler.complete_targets import _set

    source = fixture()
    target = deepcopy(source.target)
    target["documentPatch"]["parties"]["consignee"]["country"] = "Cambodia"
    original = deepcopy(target)
    with pytest.raises(ValueError, match="conflicts with its declared context owner"):
        context.validate_final(source, target)
    assert target == original
    for path, value in context.generated_values(source, target).items():
        _set(target, path, value)
    context.validate_final(source, target)
    target["documentPatch"]["cargoGroups"][0]["handlingInstructions"][0] = (
        "CARGO IN TRANSIT TO KABARI EFS BONDED WAREHOUSE"
    )
    with pytest.raises(ValueError, match="handlingInstructions"):
        context.validate_final(source, target)


def test_frozen_target_guard_rejects_stale_owned_marks_even_with_valid_checkpoint_hashes():
    from document_ocr.hashing import canonical_json_bytes, sha256_bytes
    from document_ocr.synthesis.template_compiler import descendant
    from document_ocr.synthesis.template_compiler.complete_targets import _set

    source = fixture()
    target = deepcopy(source.target)
    target["documentPatch"]["parties"]["consignee"]["country"] = "Cambodia"

    def digest(value):
        return sha256_bytes(canonical_json_bytes(value))

    receipt = NS(
        proposed_target_sha256=digest(target),
        prepared_target_sha256=digest(target),
        compatibility_adaptations=(),
        auxiliary_values_sha256=digest({}),
        numeric_auxiliary_sha256=digest({}),
        equipment_tare_values_sha256=digest({}),
        customs_presentation_sha256=digest([]),
        dangerous_goods_facts_sha256=digest([]),
    )
    case = NS(
        document_id="stale_mark_checkpoint",
        template=source.template,
        source_target=source.target,
        target=target,
        topology_reference_target=source.target,
        target_receipt=receipt,
        auxiliary_values={},
        numeric_auxiliary={},
        equipment_tare_values={},
        customs_presentation=None,
        dangerous_goods_facts=(),
    )
    with pytest.raises(ValueError, match="conflicts with its declared context owner"):
        descendant._require_frozen_target(case)
    for path, value in context.generated_values(source, target).items():
        _set(target, path, value)
    receipt.proposed_target_sha256 = receipt.prepared_target_sha256 = digest(target)
    descendant._require_frozen_target(case)
