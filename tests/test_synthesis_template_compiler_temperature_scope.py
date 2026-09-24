"""Printed setpoint scope and signed prose must survive independent sampling."""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import temperature_prose
from document_ocr.synthesis.template_compiler.descendant import (
    _render_signed_temperature_word_surface,
)
from document_ocr.synthesis.template_compiler.host import (
    SpanDraft,
    global_shared_temperature_paths,
    repeated_cargo_temperature_contract,
    validate_compiled_global_shared_temperature_scope,
    validate_repeated_cargo_temperature_scope,
    validate_signed_temperature_word_scope,
)


def _draft(raw: str, text: str, path: str) -> SpanDraft:
    start = raw.index(text)
    return SpanDraft(
        draft_id=text,
        logical_key="setpoint",
        render_mode="target_binding",
        value_kind="temperature",
        group_kind="equipment",
        group_key="equipment:setpoint",
        target_paths=(path,),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        evidence_origin="accepted_label_evidence",
        render_policy="numeric_surface",
        rationale="Printed source setpoint.",
    )


def test_global_printed_setpoint_has_one_shared_direct_owner() -> None:
    raw = "CARGO\nSET TEMP: +5,5\nCARRYING TEMPERATURE: +5,5 DEGREES CELSIUS\n"
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1"}],
            "containers": [
                {"temperatureSetpoint": {"unit": "celsius", "value": 5.5}} for _ in range(3)
            ],
        }
    }
    paths = global_shared_temperature_paths(raw, target)
    assert len(paths) == 3
    numeric = tuple(
        replace(
            _draft(raw, "+5,5", paths[0]),
            draft_id=f"temp_{i}",
            char_start=match.start(),
            char_end=match.end(),
        )
        for i, match in enumerate(re.finditer(r"\+5,5", raw))
    )
    old = NS(
        target_paths=(paths[0],),
        target_relationship="single_target",
        realization=NS(deterministic=True, requires_agent=False),
        occurrences=tuple(NS(byte_start=d.char_start, byte_end=d.char_end) for d in numeric),
    )
    with pytest.raises(ValueError, match="global shared-temperature scope"):
        validate_compiled_global_shared_temperature_scope(
            raw=raw,
            source_target=target,
            template=NS(bindings=(old,), semantic_only_target_facts=()),
        )
    shared = NS(
        target_paths=paths,
        target_relationship="shared_value_equality",
        realization=NS(deterministic=True, requires_agent=False),
        occurrences=old.occurrences,
    )
    validate_compiled_global_shared_temperature_scope(
        raw=raw,
        source_target=target,
        template=NS(bindings=(shared,), semantic_only_target_facts=()),
    )
    unit_paths = tuple(path.removesuffix("value") + "unit" for path in paths)
    template = NS(
        bindings=(
            NS(target_relationship="shared_value_equality", target_paths=paths, derivation=None),
            NS(
                target_relationship="shared_value_equality",
                target_paths=unit_paths,
                derivation=None,
            ),
        )
    )
    assert temperature_prose.shared_setpoint_owners(template, target) == {
        i: (0, 1, 2) for i in range(3)
    }
    target["documentPatch"]["containers"][1]["temperatureSetpoint"]["value"] = 2
    with pytest.raises(ValueError, match="source setpoints disagree"):
        temperature_prose.shared_setpoint_owners(template, target)


def test_signed_temperature_word_is_one_mutable_surface() -> None:
    raw = "CARRYING TEMPERATURE OF PLUS 1 DEG' C"
    target = {
        "documentPatch": {
            "containers": [
                {},
                {"temperatureSetpoint": {"unit": "celsius", "value": 1}},
            ]
        }
    }
    path = "documentPatch.containers[1].temperatureSetpoint.value"
    with pytest.raises(ValueError, match="word and magnitude"):
        validate_signed_temperature_word_scope(
            raw=raw, source_target=target, drafts=(_draft(raw, "1", path),)
        )
    validate_signed_temperature_word_scope(
        raw=raw, source_target=target, drafts=(_draft(raw, "PLUS 1", path),)
    )
    assert _render_signed_temperature_word_surface("PLUS 1", 1, -22) == "MINUS 22"
    assert _render_signed_temperature_word_surface("PLUS 1", 1, 0) == "PLUS 0"
    assert _render_signed_temperature_word_surface("PLUS 1", 1, 3) == "PLUS 3"
    with pytest.raises(ValueError, match="disagrees"):
        _render_signed_temperature_word_surface("PLUS 1", -1, 3)


def _repeated_reefer_target() -> dict:
    return {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": f"g{i}",
                    "description": "Fresh apples",
                    "hsCodes": ["08081080"],
                    "handlingInstructions": ["VENT.: 20.0 CBM/H"],
                    "netWeight": {"unit": "kilogram", "value": 12000},
                    "volume": {"unit": "cubic_metre", "value": 45},
                }
                for i in range(1, 4)
            ],
            "cargoPackages": [
                {
                    "groupId": f"g{i}",
                    "packageId": f"p{i}",
                    "quantity": 1000,
                    "typeCategory": "PACKAGE_BOX",
                }
                for i in range(1, 4)
            ],
            "cargoAllocationGroups": [
                {"groupId": f"g{i}", "allocations": [{"containerNumber": f"ABCU000000{i}"}]}
                for i in range(1, 4)
            ],
            "containers": [
                {
                    "containerNumber": f"ABCU000000{i}",
                    "typeDescription": "40RQ",
                    "temperatureSetpoint": {"unit": "celsius", "value": 1},
                }
                for i in range(1, 4)
            ],
        }
    }


def test_one_global_setting_requires_complete_shared_reefer_and_goods_bindings() -> None:
    raw = "CARRYING TEMPERATURE OF PLUS 1 DEG' C"
    target = _repeated_reefer_target()
    contract = repeated_cargo_temperature_contract(raw, target)
    assert set(contract) == {"value", "unit", "description", "hs", "handling"}
    drafts = tuple(
        NS(target_paths=paths, logical_key=name, render_mode="target_binding")
        for name, paths in contract.items()
    )
    validate_repeated_cargo_temperature_scope(raw=raw, source_target=target, drafts=drafts)
    with pytest.raises(ValueError, match="shared printed description"):
        validate_repeated_cargo_temperature_scope(
            raw=raw, source_target=target, drafts=drafts[0:2] + drafts[3:]
        )
    target["documentPatch"]["containers"][0].pop("temperatureSetpoint")
    with pytest.raises(ValueError, match="incomplete container setpoint labels"):
        validate_repeated_cargo_temperature_scope(raw=raw, source_target=target, drafts=drafts)


def test_global_setting_proof_does_not_merge_distinct_cargo_or_allocations() -> None:
    raw = "CARRYING TEMPERATURE OF PLUS 1 DEG' C"
    target = _repeated_reefer_target()
    target["documentPatch"]["cargoGroups"][1]["description"] = "Different goods"
    assert repeated_cargo_temperature_contract(raw, target) == {}
    target = _repeated_reefer_target()
    target["documentPatch"]["cargoAllocationGroups"][1]["allocations"].append(
        {"containerNumber": "ABCU0000001"}
    )
    assert repeated_cargo_temperature_contract(raw, target) == {}
