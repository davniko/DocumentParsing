from datetime import date
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler.cargo_identifiers import (
    conditioned_requests,
    generate_mark_references,
    generate_references,
    labelled_references,
    validate,
    vehicle_ids,
)
from document_ocr.synthesis.template_compiler.complete_targets import propose_goods


def field():
    return dict(
        source="used Mercedes Benz C180 Fin.: WDD2040461A083953 Petrol - 1796 CM³",
        paths=["documentPatch.cargoGroups[0].description"],
        key="f1",
    )


def test_embedded_vehicle_serial_is_local_novel_and_preserves_classification_context():
    a = conditioned_requests([field()], sample_id="a", seed=42)[0]
    b = conditioned_requests([field()], sample_id="b", seed=42)[0]
    new = a["requiredVehicleIdentifiers"][0]
    assert len(new) == 17
    assert new[:11] == "WDD2040461A"
    assert new != "WDD2040461A083953"
    assert b["requiredVehicleIdentifiers"][0] != new
    assert conditioned_requests([field()], sample_id="a", seed=42)[0] == a
    validate(a, field()["source"].replace("WDD2040461A083953", new))
    with pytest.raises(ValueError, match="serial contract"):
        validate(a, "Used car FIN: INVALIDSHORT123")
    with pytest.raises(ValueError, match="serial contract"):
        validate(a, field()["source"])
    assert "requiredVehicleIdentifiers" not in field()


def test_vehicle_dependency_pins_tariff_not_source_cargo_description():
    target = {
        "documentPatch": {
            "issueDate": "2024-01-01",
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["870323"], "description": field()["source"]}
            ],
        }
    }
    registry = NS(global_codes=(), receipt=NS(snapshot_date=date(2026, 8, 31)))
    metadata = propose_goods(target, registry=registry, sample_id="a", seed=42)
    assert target["documentPatch"]["cargoGroups"][0]["hsCodes"] == ["870323"]
    assert "vehicle" in metadata["g1"][0]["reason"]


def test_unlabeled_numbers_are_not_assumed_to_be_vehicle_identifiers():
    assert not vehicle_ids("ORDER WDD2040461A083953")
    assert not vehicle_ids("FINANCIAL TOTAL 12345678901234567")


def test_labelled_reference_values_are_local_not_repeated_linguistic_fragments():
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "marksAndNumbers": [
                        "C/P.NO:BN96-51976A",
                        "ORD.NO:202512309850",
                        "C/P.NO:BN96-51976A",
                        "COTTON GOODS",
                    ]
                }
            ]
        }
    }
    old = labelled_references(target)
    template = NS(
        bindings=[NS(target_paths=[p], realization=NS(mode="repeated_surface")) for p in old]
    )
    new = generate_references(target, DeterministicStream(42, "references", "sample"), template)
    assert len(new) == 3
    assert all(new[p] != text for p, text in old.items())
    assert (
        new["documentPatch.cargoGroups[0].marksAndNumbers[0]"]
        == new["documentPatch.cargoGroups[0].marksAndNumbers[2]"]
    )
    assert all(text.startswith(("C/P.NO:", "ORD.NO:")) for text in new.values())


def test_labeled_mark_lists_preserve_roles_repetition_and_punctuation():
    marks = [
        "SI NO:Y25T0515",
        "P/O: 3089346273, 3090810834,",
        "P/O: 3090810834,",
        "PART NO.: 3903-001078, 3903-001100,",
        "P/NO.:BULK:5 CTNS,P1-P14",
        "BRAND C/NO. 1-200",
        "PART NO.: BOLTS",
    ]
    target = {"documentPatch": {"cargoGroups": [{"marksAndNumbers": marks}]}}
    template = NS(
        bindings=[
            NS(
                target_paths=[f"documentPatch.cargoGroups[0].marksAndNumbers[{i}]"],
                realization=NS(mode="single_surface"),
            )
            for i in range(len(marks))
        ]
    )
    result = generate_mark_references(target, template, DeterministicStream(42, "refs", "s"))
    values = list(result.values())
    assert len(values) == 4
    assert values[0].startswith("SI NO:") and values[0] != marks[0]
    assert values[1].split(", ")[1] == values[2].removeprefix("P/O: ")
    assert values[3].startswith("PART NO.: ") and values[3].endswith(",")
    assert all(len(value) == len(old) for value, old in zip(values, marks[:4], strict=True))
    assert (
        generate_mark_references(target, template, DeterministicStream(42, "refs", "s")) == result
    )


def test_reference_prefix_slot_does_not_grant_ownership_of_immutable_code():
    from document_ocr.synthesis.template_compiler.cargo_identifiers import fixed_references

    text = "C/P.NO:BN96-59930A"
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    target = {"documentPatch": {"cargoGroups": [{"marksAndNumbers": [text]}]}}
    binding = NS(
        target_paths=[path],
        logical_key="reference",
        realization=NS(
            mode="token_projected_surface",
            target_values=[NS(source_value=text)],
            slots=[
                NS(
                    required_target_prefix_tokens=(),
                    required_target_suffix_tokens=("p", "no", "bn96", "59930a"),
                )
            ],
        ),
    )
    template = NS(bindings=[binding])
    assert fixed_references(target, template) == {path}
    assert (
        generate_references(target, DeterministicStream(42, "test", "sample"), template)[path]
        == text
    )
    binding.realization.slots = [
        NS(required_target_prefix_tokens=("c", "p", "no", "bn96"), required_target_suffix_tokens=())
    ]
    new = generate_references(target, DeterministicStream(42, "test", "sample"), template)[path]
    assert new.startswith("C/P.NO:BN96-") and new != text


def test_reviewed_bare_lists_share_identifiers_with_captioned_lists():
    marks = ["PART NO.: 3903-001078,", "3903-001078,3903-001118,", "3903-001118"]
    paths = [f"documentPatch.cargoGroups[0].marksAndNumbers[{i}]" for i in range(3)]
    target = {"documentPatch": {"cargoGroups": [{"marksAndNumbers": marks}]}}
    template = NS(
        bindings=[NS(target_paths=[p], realization=NS(mode="single_surface")) for p in paths]
    )
    stream = DeterministicStream(42, "refs", "s")
    ordinary = generate_mark_references(target, template, stream)
    assert set(ordinary) == {paths[0]}
    result = generate_mark_references(
        target, template, stream, reviewed_lists=dict(zip(paths[1:], marks[1:], strict=True))
    )
    assert result[paths[0]] == ordinary[paths[0]]
    assert result[paths[0]].removeprefix("PART NO.: ") == result[paths[1]].split(",")[0] + ","
    assert result[paths[1]].split(",")[1] == result[paths[2]]
    assert all(
        result[p] != old and len(result[p]) == len(old) for p, old in zip(paths, marks, strict=True)
    )


@pytest.mark.parametrize(
    "text,reviewed,error",
    [
        ("A123", "A124", "differs from its source"),
        ("12 PALLETS", "12 PALLETS", "non-identifier content"),
        ("BOLTS", "BOLTS", "non-code word"),
    ],
)
def test_reviewed_reference_list_does_not_guess_its_role(text, reviewed, error):
    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    target = {"documentPatch": {"cargoGroups": [{"marksAndNumbers": [text]}]}}
    with pytest.raises(ValueError, match=error):
        generate_mark_references(
            target,
            NS(bindings=[]),
            DeterministicStream(42, "refs", "s"),
            reviewed_lists={path: reviewed},
        )


def test_reviewed_reference_path_must_exist():
    with pytest.raises(ValueError, match="path is absent"):
        generate_mark_references(
            {"documentPatch": {}},
            NS(bindings=[]),
            DeterministicStream(42, "refs", "s"),
            reviewed_lists={"missing": "A123"},
        )
