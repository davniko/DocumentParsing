from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import raw_text_template_experiment.descendant as descendant
from raw_text_template_experiment.descendant import (
    _binding_target_value,
    _date_candidates,
    _direct_auxiliary_route,
    _load_replayed_stage,
    _numeric_leaves,
    _package_candidate,
    _partition_semantic_surface,
    _render_agent_target_binding,
    _render_certified_date_surface,
    _render_date_surface,
    _render_derivations,
    _render_target_binding,
    _residual_output_type,
    _resolve_path,
    _restore_carrier,
    _set_path,
    _solve_identifier_relationships,
    _unchanged_target_output,
)
from raw_text_template_experiment.descendant_models import ResidualStageReceipt


def _slot(source_text: str, *, policy: str = "natural_text", slot_id: str = "slot_1"):
    return SimpleNamespace(
        slot_id=slot_id,
        source_text=source_text,
        render_policy=policy,
        semantic_role="test semantic role",
    )


def _auxiliary(source_text: str, *, policy: str, value_kind: str = "identifier"):
    return SimpleNamespace(
        occurrences=(_slot(source_text, policy=policy),),
        value_kind=value_kind,
    )


def test_path_resolution_and_assignment_are_exact() -> None:
    target = {"documentPatch": {"containers": [{"containerNumber": "ABCD1234567"}]}}
    path = "documentPatch.containers[0].containerNumber"

    assert _resolve_path(target, path) == "ABCD1234567"
    _set_path(target, path, "WXYZ7654321")
    assert _resolve_path(target, path) == "WXYZ7654321"

    with pytest.raises(ValueError, match="outside documentPatch"):
        _resolve_path(target, "schemaVersion")
    with pytest.raises(ValueError, match="index is absent"):
        _resolve_path(target, "documentPatch.containers[1].containerNumber")


def test_carrier_restoration_replaces_the_complete_object() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "carrier": {
                    "name": "Source Line",
                    "address": "Source House",
                    "contactDetails": {"phoneNumbers": ["+1 555 0100"]},
                }
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "Generated Line", "address": "Generated House"}}
        }
    }

    receipt = _restore_carrier(source_target=source, target=target)

    assert receipt is not None
    assert receipt.target_path == "documentPatch.parties.carrier"
    assert (
        target["documentPatch"]["parties"]["carrier"]
        == source["documentPatch"]["parties"]["carrier"]
    )


def test_extended_date_renderer_preserves_hyphenated_named_month_style() -> None:
    assert _render_date_surface("10-MAR-2024", "2024-03-10", "2025-11-03") == "03-NOV-2025"


@pytest.mark.parametrize(
    ("source", "old", "new", "expected"),
    [
        ("30.MAY.2025", "2025-05-30", "2025-04-20", "20.APR.2025"),
        ("MAY/15/2024", "2024-05-15", "2024-03-29", "MAR/29/2024"),
        ("AUG.11,2024", "2024-08-11", "2027-10-21", "OCT.21,2027"),
        ("28 MARS 2024", "2024-03-28", "2029-01-07", "07 JANVIER 2029"),
    ],
)
def test_certified_date_renderer_uses_observable_grammar(
    source: str, old: str, new: str, expected: str
) -> None:
    assert _render_certified_date_surface(source, old, new) == expected


def test_package_candidate_uses_document_native_abbreviation_width() -> None:
    assert _package_candidate("CTN", "PACKAGE_PACKAGE") == "PKG"
    assert _package_candidate("CARTONS", "PACKAGE_PALLET") == "PALLETS"


def test_semantic_partition_prefers_exact_prefix_and_comma_clauses() -> None:
    cargo_slots = (_slot("PACKED IN", slot_id="one"), _slot("OLD BAGS", slot_id="two"))
    assert _partition_semantic_surface("PACKED IN 2367 NEW POLYPROPYLENE BAGS", cargo_slots) == (
        "PACKED IN",
        "2367 NEW POLYPROPYLENE BAGS",
    )

    address_slots = (
        _slot("LONG STREET ADDRESS", slot_id="one"),
        _slot("CITY", slot_id="two"),
        _slot("POSTCODE", slot_id="three"),
    )
    assert _partition_semantic_surface(
        "Plot 18, Chambal Industrial Estate, Baran Road", address_slots
    ) == ("Plot 18,", "Chambal Industrial Estate,", "Baran Road")


def test_phone_prefix_replacement_preserves_adjacent_context() -> None:
    binding = SimpleNamespace(
        value_kind="phone",
        target_paths=("documentPatch.phone",),
        occurrences=(_slot("+20 1228470604egy import***", slot_id="slot_phone"),),
        realization=SimpleNamespace(target_values=()),
    )

    output = _render_agent_target_binding(
        binding,
        source_target={"documentPatch": {}},
        target={"documentPatch": {"phone": "+33 7 82 14 63 29"}},
    )

    assert output.replacements == {"slot_phone": "+33 7 82 14 63 29egy import***"}
    assert output.canonical_value == "+33 7 82 14 63 29"


def test_identifier_relationship_solver_respects_fixed_target_position() -> None:
    marks = SimpleNamespace(
        logical_key="marks",
        value_kind="identifier",
        target_paths=(),
        occurrences=(_slot("2", policy="opaque_identifier", slot_id="slot_marks"),),
        source_relationships=(
            SimpleNamespace(
                dependency_binding="tax",
                relationship="is_embedded_in_exact_source_identifier",
            ),
            SimpleNamespace(
                dependency_binding="voyage",
                relationship="is_embedded_in_exact_source_identifier",
            ),
        ),
    )
    tax = SimpleNamespace(
        logical_key="tax",
        value_kind="identifier",
        target_paths=(),
        occurrences=(_slot("340072555", policy="opaque_identifier", slot_id="slot_tax"),),
        source_relationships=(
            SimpleNamespace(
                dependency_binding="acid",
                relationship="is_embedded_in_exact_source_identifier",
            ),
        ),
    )
    acid = SimpleNamespace(
        logical_key="acid",
        value_kind="identifier",
        target_paths=(),
        occurrences=(
            _slot("3400725552024020017", policy="opaque_identifier", slot_id="slot_acid"),
        ),
        source_relationships=(
            SimpleNamespace(
                dependency_binding="tax",
                relationship="embeds_exact_source_identifier",
            ),
        ),
    )
    voyage = SimpleNamespace(
        logical_key="voyage",
        value_kind="identifier",
        target_paths=("documentPatch.transport.voyageNumber",),
        occurrences=(_slot("ASA09E24", policy="opaque_identifier", slot_id="slot_voyage"),),
        source_relationships=(),
    )
    outputs = {
        "marks": descendant.BindingOutput({"slot_marks": "8"}, "8"),
        "tax": descendant.BindingOutput({"slot_tax": "111111111"}, "111111111"),
        "acid": descendant.BindingOutput(
            {"slot_acid": "2222222223333333333"}, "2222222223333333333"
        ),
        "voyage": descendant.BindingOutput({"slot_voyage": "EJS17Q76"}, "EJS17Q76"),
    }

    _solve_identifier_relationships(
        template=SimpleNamespace(bindings=(marks, tax, acid, voyage)),
        outputs=outputs,
        stream=descendant.DeterministicStream(7, "test", "doc"),
    )

    rendered_mark = outputs["marks"].replacements["slot_marks"]
    rendered_tax = outputs["tax"].replacements["slot_tax"]
    rendered_acid = outputs["acid"].replacements["slot_acid"]
    assert rendered_mark == "7"
    assert rendered_tax[5] == rendered_mark
    assert rendered_acid.startswith(rendered_tax)


def test_date_candidates_preserve_dotted_named_month_surface() -> None:
    assert _date_candidates("20.APR.2025") == frozenset({descendant.date(2025, 4, 20)})


def test_ambiguous_named_month_style_is_explicitly_routed() -> None:
    # MAY is both the abbreviated and full English month name. A different target month
    # would make those styles diverge, so the host must not guess which was intended.
    with pytest.raises(ValueError, match="ambiguous certified date surface"):
        _render_date_surface("05-MAY-2023", "2023-05-05", "2031-09-23")
    assert _render_date_surface("05-MAY-2023", "2023-05-05", "2023-05-05") == "05-MAY-2023"

    deterministic, reason = _direct_auxiliary_route(
        _auxiliary("05-MAY-2023", policy="date_surface", value_kind="date")
    )
    assert not deterministic
    assert "unambiguous render style" in reason


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        (_auxiliary("25.2KG", policy="numeric_surface"), True),
        (_auxiliary("Zero", policy="numeric_surface", value_kind="integer"), False),
        (_auxiliary("ABC123", policy="opaque_identifier"), True),
        (_auxiliary("VAT", policy="opaque_identifier"), False),
        (_auxiliary("40HC", policy="opaque_identifier", value_kind="equipment"), False),
    ],
)
def test_auxiliary_capability_gate_rejects_semantic_guessing(binding, expected: bool) -> None:
    deterministic, reason = _direct_auxiliary_route(binding)

    assert deterministic is expected
    assert reason


def test_residual_schema_requires_every_exact_slot_and_forbids_extras() -> None:
    binding = SimpleNamespace(
        occurrences=(
            _slot("ALPHA", slot_id="slot_alpha"),
            _slot("BETA", slot_id="slot_beta"),
        )
    )
    output_type = _residual_output_type((binding,))

    valid = output_type.model_validate({"slot_alpha": "one", "slot_beta": "two"})
    assert valid.model_dump() == {"slot_alpha": "one", "slot_beta": "two"}
    with pytest.raises(ValidationError):
        output_type.model_validate({"slot_alpha": "one"})
    with pytest.raises(ValidationError):
        output_type.model_validate({"slot_alpha": "one", "slot_beta": "two", "slot_gamma": "three"})


def test_v5_equipment_categories_replace_legacy_description_semantics() -> None:
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "ABCD1234567",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }

    assert _binding_target_value(target, "documentPatch.containers[0].typeDescription") == {
        "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
        "typeCategory": "GENERAL_PURPOSE",
    }


def test_unchanged_target_retains_compiled_document_native_surface() -> None:
    slot = _slot("NON-NEGOTIABLE", slot_id="slot_status")
    binding = SimpleNamespace(
        target_paths=("documentPatch.negotiability",),
        occurrences=(slot,),
        realization=SimpleNamespace(
            target_values=(
                SimpleNamespace(
                    target_path="documentPatch.negotiability",
                    source_value="non_negotiable",
                ),
            ),
        ),
    )

    output = _render_target_binding(
        binding,
        {"documentPatch": {"negotiability": "non_negotiable"}},
    )

    assert output.canonical_value == "non_negotiable"
    assert output.replacements == {"slot_status": "NON-NEGOTIABLE"}


def test_unchanged_agent_target_preserves_distinct_certified_values() -> None:
    binding = SimpleNamespace(
        target_paths=("documentPatch.address", "documentPatch.city"),
        occurrences=(_slot("OLD ADDRESS, OLD CITY", slot_id="slot_address"),),
        realization=SimpleNamespace(
            target_values=(
                SimpleNamespace(target_path="documentPatch.address", source_value="OLD ADDRESS"),
                SimpleNamespace(target_path="documentPatch.city", source_value="OLD CITY"),
            )
        ),
    )

    output = _unchanged_target_output(
        binding,
        {"documentPatch": {"address": "OLD ADDRESS", "city": "OLD CITY"}},
    )

    assert output is not None
    assert output.canonical_value == ["OLD ADDRESS", "OLD CITY"]
    assert output.replacements == {"slot_address": "OLD ADDRESS, OLD CITY"}


def test_scalar_numeric_leaf_survives_named_sum_filter() -> None:
    assert _numeric_leaves(7, names=frozenset({"quantity", "packageQuantity"})) == (7,)


def test_number_words_can_count_a_collection_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = SimpleNamespace(
        logical_key="agent:container_count_words",
        realization=SimpleNamespace(mode="deterministic_derivation"),
        derivation="number_to_words",
        dependency_bindings=(),
        dependency_paths=("documentPatch.containers",),
        occurrences=(_slot("ONE", slot_id="slot_count"),),
    )
    case = SimpleNamespace(
        template=SimpleNamespace(bindings=(binding,), byte_template=None),
        target={"documentPatch": {"containers": [{}, {}]}},
        source_target={"documentPatch": {"containers": [{}]}},
        source=b"",
    )
    monkeypatch.setattr(descendant, "_validate_binding_format", lambda **_kwargs: None)
    outputs = {}

    _render_derivations(case=case, outputs=outputs, country_codes={})

    assert outputs[binding.logical_key].canonical_value == "two"
    assert outputs[binding.logical_key].replacements == {"slot_count": "TWO"}


def test_quantity_sum_accepts_a_direct_scalar_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = SimpleNamespace(
        logical_key="agent:quantity_sum",
        realization=SimpleNamespace(mode="deterministic_derivation"),
        derivation="sum_package_quantity",
        dependency_bindings=(),
        dependency_paths=("documentPatch.cargoPackages[0].quantity",),
        occurrences=(_slot("3", policy="numeric_surface", slot_id="slot_quantity"),),
    )
    case = SimpleNamespace(
        template=SimpleNamespace(bindings=(binding,), byte_template=None),
        target={"documentPatch": {"cargoPackages": [{"quantity": 7}]}},
        source_target={"documentPatch": {"cargoPackages": [{"quantity": 3}]}},
        source=b"",
    )
    monkeypatch.setattr(descendant, "_validate_binding_format", lambda **_kwargs: None)
    outputs = {}

    _render_derivations(case=case, outputs=outputs, country_codes={})

    assert outputs[binding.logical_key].canonical_value == 7
    assert outputs[binding.logical_key].replacements == {"slot_quantity": "7"}


def test_temperature_setpoint_derivation_preserves_each_certified_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = SimpleNamespace(
        logical_key="anchor:container_temperature",
        realization=SimpleNamespace(mode="deterministic_derivation"),
        derivation="temperature_setpoint",
        dependency_bindings=(),
        dependency_paths=(
            "documentPatch.containers[0].temperatureSetpoint.unit",
            "documentPatch.containers[0].temperatureSetpoint.value",
        ),
        occurrences=(
            _slot("+1 C", policy="derived_surface", slot_id="slot_compact"),
            _slot("+1,0 C", policy="derived_surface", slot_id="slot_decimal"),
        ),
    )
    case = SimpleNamespace(
        template=SimpleNamespace(bindings=(binding,), byte_template=None),
        target={
            "documentPatch": {
                "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": -18.5}}]
            }
        },
        source_target={
            "documentPatch": {
                "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}]
            }
        },
        source=b"",
    )
    monkeypatch.setattr(descendant, "_validate_binding_format", lambda **_kwargs: None)
    outputs = {}

    _render_derivations(case=case, outputs=outputs, country_codes={})

    assert outputs[binding.logical_key].canonical_value == {
        "unit": "celsius",
        "value": -18.5,
    }
    assert outputs[binding.logical_key].replacements == {
        "slot_compact": "-18.5 C",
        "slot_decimal": "-18,5 C",
    }


def _replay_fixture(tmp_path, *, output: dict[str, str]):
    document_id = "doc_replay"
    prefix = tmp_path / "cases" / document_id
    prefix.mkdir(parents=True)
    source_target = {"documentPatch": {"billOfLadingNumber": "OLD1"}}
    target = {"documentPatch": {"billOfLadingNumber": "NEW2"}}
    target_receipt = {"document_id": document_id}
    (prefix / "source.txt").write_bytes(b"B/L OLD1\n")
    (prefix / "source-target.json").write_text(json.dumps(source_target), encoding="utf-8")
    (prefix / "target.json").write_text(json.dumps(target), encoding="utf-8")
    (prefix / "target-receipt.json").write_text(json.dumps(target_receipt), encoding="utf-8")
    stage = ResidualStageReceipt.model_validate(
        {
            "schema_version": 1,
            "document_id": document_id,
            "status": "success",
            "started_at": "2026-09-12T00:00:00Z",
            "completed_at": "2026-09-12T00:00:01Z",
            "duration_seconds": 1.0,
            "system_prompt_sha256": "a" * 64,
            "user_prompt_sha256": "b" * 64,
            "output_schema_sha256": "c" * 64,
            "output": output,
            "error_type": None,
            "error_message": None,
            "messages": [],
            "usage": {
                "requests": 1,
                "inputTokens": 10,
                "outputTokens": 5,
                "reasoningTokens": 3,
                "estimatedCostUsd": "0.001",
                "providerReportedCostUsd": None,
            },
        }
    )
    (prefix / "agent-stage.json").write_text(stage.model_dump_json(), encoding="utf-8")
    case = SimpleNamespace(
        document_id=document_id,
        source=b"B/L OLD1\n",
        source_target=source_target,
        target=target,
        target_receipt=SimpleNamespace(model_dump=lambda *, mode: target_receipt),
    )
    plan = SimpleNamespace(
        residual_bindings=(SimpleNamespace(occurrences=(_slot("OLD1", slot_id="slot_current"),)),)
    )
    return case, plan


def test_replay_projects_hardened_residual_schema_and_records_dropped_slots(tmp_path) -> None:
    case, plan = _replay_fixture(
        tmp_path,
        output={"slot_current": "NEW2", "slot_now_deterministic": "IGNORED"},
    )

    projected, source_stage, receipt = _load_replayed_stage(
        replay_root=tmp_path,
        replay_commit_sha256="d" * 64,
        replay_transaction_sha256="e" * 64,
        expected_system_prompt_sha256="a" * 64,
        case=case,
        plan=plan,
    )

    assert projected == {"slot_current": "NEW2"}
    assert source_stage.status == "success"
    assert receipt.replayed_output_slot_count == 1
    assert receipt.dropped_slot_ids == ("slot_now_deterministic",)
    assert receipt.new_provider_requests == 0


def test_replay_refuses_a_missing_current_residual_slot(tmp_path) -> None:
    case, plan = _replay_fixture(tmp_path, output={"slot_obsolete": "VALUE"})

    with pytest.raises(ValueError, match="missing current slots"):
        _load_replayed_stage(
            replay_root=tmp_path,
            replay_commit_sha256="d" * 64,
            replay_transaction_sha256="e" * 64,
            expected_system_prompt_sha256="a" * 64,
            case=case,
            plan=plan,
        )
