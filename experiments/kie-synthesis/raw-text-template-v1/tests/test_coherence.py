from __future__ import annotations

from types import SimpleNamespace

import pytest

from raw_text_template_experiment.coherence import (
    CoherenceContractError,
    CoherenceReviewRequired,
    coherence_review_candidates,
    inclusive_range_cardinalities,
    materialize_coherence_decisions,
    reconcile_coherence_constraints_after_binding_revision,
    validate_coherence_contracts,
    validate_render_coherence,
    whole_inclusive_range_surface,
)
from raw_text_template_experiment.host import SpanDraft
from raw_text_template_experiment.models import CoherenceCandidateDecision, CoherenceConstraint

MARK = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
QUANTITY = "documentPatch.cargoPackages[0].quantity"


def _target(*, quantity: int, mark: str = "PLT NO.1-7") -> dict[str, object]:
    return {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "marksAndNumbers": [mark]}],
            "cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": quantity}],
        }
    }


def _allocated_target(*, package_quantity: int, allocations: tuple[int, ...]) -> dict[str, object]:
    return {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1"}],
            "cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": package_quantity}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "single_package_level",
                    "packageIds": ["p1"],
                    "allocations": [
                        {"containerNumber": f"ABCU00000{index}", "packageQuantity": value}
                        for index, value in enumerate(allocations)
                    ],
                }
            ],
        }
    }


def _draft(
    raw: str,
    surface: str,
    *,
    logical_key: str = "agent:range",
    render_mode: str = "agent_residual",
    value_kind: str = "cargo_text",
    target_paths: tuple[str, ...] = (),
    occurrence: int = 0,
    group_kind: str = "cargo",
    group_key: str = "cargo:g1",
) -> SpanDraft:
    offsets: list[int] = []
    cursor = 0
    while (found := raw.find(surface, cursor)) >= 0:
        offsets.append(found)
        cursor = found + len(surface)
    start = offsets[occurrence]
    return SpanDraft(
        draft_id=f"draft_{logical_key}_{occurrence}",
        logical_key=logical_key,
        render_mode=render_mode,
        value_kind=value_kind,
        group_kind=group_kind,
        group_key=group_key,
        target_paths=target_paths,
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(surface),
        source_text=surface,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Fixture binding.",
    )


@pytest.mark.parametrize(
    ("surface", "cardinalities"),
    (("PACKAGE NO : 1\n- 18", (18,)), ("SERIAL 161-\n165", (5,))),
)
def test_inclusive_range_cardinalities_support_ocr_line_breaks(
    surface: str, cardinalities: tuple[int, ...]
) -> None:
    assert inclusive_range_cardinalities(surface) == cardinalities


def _candidate_rows(
    raw: str,
    bindings: tuple[SpanDraft, ...],
    target: dict[str, object],
    constraints: tuple[CoherenceConstraint, ...] = (),
) -> tuple[dict[str, object], ...]:
    return tuple(
        {"candidateId": f"review_candidate_{index:04d}", **row}
        for index, row in enumerate(
            coherence_review_candidates(
                raw=raw,
                bindings=bindings,
                source_target=target,
                constraints=constraints,
            ),
            start=1,
        )
    )


def _apply_all(
    rows: tuple[dict[str, object], ...],
    *,
    disposition: str = "apply_suggestion",
) -> tuple[CoherenceConstraint, ...]:
    return materialize_coherence_decisions(
        candidates=rows,
        constraints=(),
        decisions=tuple(
            CoherenceCandidateDecision.model_validate(
                {
                    "candidate_id": row["candidateId"],
                    "disposition": disposition,
                    "suggestion_index": 0 if disposition == "apply_suggestion" else None,
                    "rationale": "The source context resolves this arithmetic relationship.",
                }
            )
            for row in rows
        ),
    )


@pytest.mark.parametrize(
    ("surface", "expected"),
    (
        ("PACKAGE 1-7", (7,)),
        ("PLT NO.1-7", (7,)),
        ("RANGE 1 TO 7", (7,)),
        ("SEQ 7~1", (7,)),
        ("001-225 and 226-240", (225, 15)),
        ("36062 TO 36526", (465,)),
        ("SDW/1 TO SDW/80", (80,)),
        ("C/N0.1-36, P/N0.1-3", (36, 3)),
        ("1-2,880", (2880,)),
        ("PKG/NO.01-03", (3,)),
        ("SKU1-7X", ()),
    ),
)
def test_range_grammar_is_format_general(surface: str, expected: tuple[int, ...]) -> None:
    assert inclusive_range_cardinalities(surface) == expected


def test_complete_repeated_prefix_range_preserves_exact_endpoint_offsets() -> None:
    source = "SDW/1 TO SDW/80"
    surface = whole_inclusive_range_surface(source)

    assert surface is not None
    assert surface.repeated_prefix is True
    assert surface.cardinality == 80
    assert surface.replace_end(source, "23") == "SDW/1 TO SDW/23"
    assert whole_inclusive_range_surface("SDW/1 TO XYZ/80") is None


@pytest.mark.parametrize(
    "surface",
    ("PACKAGE 1-7", "PLT NO.1-7", "RANGE 1 TO 7", "SEQ 7~1"),
)
def test_range_candidate_is_topological_not_phrase_specific(surface: str) -> None:
    raw = f"--- PAGE 1 ---\n{surface}\n"
    bindings = (_draft(raw, surface),)
    candidates = _candidate_rows(raw, bindings, _target(quantity=7))

    assert len(candidates) == 1
    assert candidates[0]["kind"] == "cross_field_semantic_relation"
    assert candidates[0]["requiredRevision"] is True
    assert candidates[0]["details"]["suggestedContracts"] == (
        {
            "kind": "inclusive_range_cardinality",
            "memberLogicalKeys": ("agent:range",),
            "dependencyPaths": (QUANTITY,),
        },
    )


def test_repeated_printings_form_one_logical_expression() -> None:
    raw = "--- PAGE 1 ---\nPACKAGE 1-7\nPACKAGE 1-7\n"
    bindings = (
        _draft(raw, "PACKAGE 1-7", occurrence=0),
        _draft(raw, "PACKAGE 1-7", occurrence=1),
    )
    candidates = _candidate_rows(raw, bindings, _target(quantity=7))

    assert len(candidates) == 1
    assert candidates[0]["logicalKeys"] == ("agent:range",)
    assert candidates[0]["lineIds"] == ("L00002", "L00003")


def test_target_owned_constraint_accepts_changed_text_and_rejects_stale_text() -> None:
    raw = "--- PAGE 1 ---\nPLT NO.1-7\n"
    binding = _draft(
        raw,
        "PLT NO.1-7",
        logical_key="anchor:" + MARK,
        render_mode="target_binding",
        target_paths=(MARK,),
    )
    rows = _candidate_rows(raw, (binding,), _target(quantity=7))
    constraints = _apply_all(rows)
    validate_coherence_contracts(
        raw=raw,
        bindings=(binding,),
        source_target=_target(quantity=7),
        constraints=constraints,
    )

    validate_render_coherence(
        bindings=(binding,),
        constraints=constraints,
        source_target=_target(quantity=7),
        target=_target(quantity=9, mark="PLT NO.1-9"),
        outputs=None,
    )
    with pytest.raises(ValueError, match=r"stayed unchanged|cardinality"):
        validate_render_coherence(
            bindings=(binding,),
            constraints=constraints,
            source_target=_target(quantity=7),
            target=_target(quantity=9),
            outputs=None,
        )


def test_source_only_constraint_requires_a_final_rendered_surface() -> None:
    raw = "--- PAGE 1 ---\nPACKAGE 1-7\n"
    deterministic = _draft(
        raw,
        "PACKAGE 1-7",
        render_mode="deterministic_auxiliary",
        value_kind="operational_text",
        group_kind="package",
        group_key="package:detail:range",
    )
    deterministic_rows = _candidate_rows(raw, (deterministic,), _target(quantity=7))
    deterministic_constraints = _apply_all(deterministic_rows)
    with pytest.raises(CoherenceContractError, match="require agent_residual"):
        validate_coherence_contracts(
            raw=raw,
            bindings=(deterministic,),
            source_target=_target(quantity=7),
            constraints=deterministic_constraints,
        )

    binding = _draft(
        raw,
        "PACKAGE 1-7",
        render_mode="agent_residual",
        value_kind="operational_text",
        group_kind="package",
        group_key="package:detail:range",
    )
    rows = _candidate_rows(raw, (binding,), _target(quantity=7))
    constraints = _apply_all(rows)
    validate_coherence_contracts(
        raw=raw,
        bindings=(binding,),
        source_target=_target(quantity=7),
        constraints=constraints,
    )
    validate_render_coherence(
        bindings=(binding,),
        constraints=constraints,
        source_target=_target(quantity=7),
        target=_target(quantity=9, mark="PLT NO.1-9"),
        outputs={"agent:range": SimpleNamespace(replacements={"slot": "PACKAGE 1-9"})},
    )
    with pytest.raises(ValueError, match="cardinality"):
        validate_render_coherence(
            bindings=(binding,),
            constraints=constraints,
            source_target=_target(quantity=7),
            target=_target(quantity=9, mark="PLT NO.1-9"),
            outputs={"agent:range": SimpleNamespace(replacements={"slot": "PACKAGE 1-7"})},
        )


def test_independent_and_review_required_are_explicit_terminal_dispositions() -> None:
    raw = "--- PAGE 1 ---\nMODEL 1-7\n"
    binding = _draft(raw, "MODEL 1-7")
    rows = _candidate_rows(raw, (binding,), _target(quantity=7))
    independent = _apply_all(rows, disposition="reviewed_independent")
    validate_coherence_contracts(
        raw=raw,
        bindings=(binding,),
        source_target=_target(quantity=7),
        constraints=independent,
    )
    validate_render_coherence(
        bindings=(binding,),
        constraints=independent,
        source_target=_target(quantity=7),
        target=_target(quantity=9, mark="PLT NO.1-9"),
        outputs=None,
    )

    review = _apply_all(rows, disposition="review_required")
    with pytest.raises(CoherenceReviewRequired, match="resolves this arithmetic"):
        validate_coherence_contracts(
            raw=raw,
            bindings=(binding,),
            source_target=_target(quantity=7),
            constraints=review,
        )


def test_aggregate_constraint_can_span_bindings_or_multiple_ranges_in_one_binding() -> None:
    raw = "--- PAGE 1 ---\nRANGE 1-3\nRANGE 4-7\n"
    bindings = (
        _draft(raw, "RANGE 1-3", logical_key="agent:range_a"),
        _draft(raw, "RANGE 4-7", logical_key="agent:range_b"),
    )
    rows = _candidate_rows(raw, bindings, _target(quantity=7))
    assert len(rows) == 1
    assert rows[0]["logicalKeys"] == ("agent:range_a", "agent:range_b")
    assert _apply_all(rows)[0].kind == "aggregate_inclusive_range_cardinality"

    composite_raw = "--- PAGE 1 ---\nC/N0.1-36, P/N0.1-3\n"
    composite = (_draft(composite_raw, "C/N0.1-36, P/N0.1-3"),)
    composite_rows = _candidate_rows(composite_raw, composite, _target(quantity=39))
    assert len(composite_rows) == 1
    constraint = _apply_all(composite_rows)[0]
    assert constraint.kind == "aggregate_inclusive_range_cardinality"
    assert constraint.member_logical_keys == ("agent:range",)


def test_nested_allocation_subtotals_and_package_total_are_all_retained() -> None:
    raw = "--- PAGE 1 ---\n1-5\n1-10\n1-4\n1-35\n7-7\n"
    surfaces = ("1-5", "1-10", "1-4", "1-35", "7-7")
    bindings = tuple(
        _draft(raw, surface, logical_key=f"agent:range_{index}")
        for index, surface in enumerate(surfaces)
    )
    target = _allocated_target(package_quantity=55, allocations=(19, 36))
    rows = _candidate_rows(raw, bindings, target)

    assert len(rows) == 3
    assert sorted(row["details"]["structuredQuantities"][0]["value"] for row in rows) == [
        19,
        36,
        55,
    ]
    constraints = _apply_all(rows)
    validate_coherence_contracts(
        raw=raw,
        bindings=bindings,
        source_target=target,
        constraints=constraints,
    )
    assert len({key for row in constraints for key in row.member_logical_keys}) == 5


def test_equal_quantities_remain_path_distinct_suggestions() -> None:
    raw = "--- PAGE 1 ---\n1-52\n"
    binding = (_draft(raw, "1-52"),)
    target = _allocated_target(package_quantity=104, allocations=(52, 52))
    rows = _candidate_rows(raw, binding, target)

    assert len(rows) == 1
    suggestions = rows[0]["details"]["suggestedContracts"]
    assert len(suggestions) == 2
    assert suggestions[0]["dependencyPaths"] != suggestions[1]["dependencyPaths"]


def test_scope_fails_closed_when_multiple_groups_are_unresolved() -> None:
    raw = "--- PAGE 1 ---\n1-7\n"
    binding = (
        _draft(
            raw,
            "1-7",
            group_kind="package",
            group_key="package:detail:unknown",
        ),
    )
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1"}, {"groupId": "g2"}],
            "cargoPackages": [
                {"packageId": "p1", "groupId": "g1", "quantity": 7},
                {"packageId": "p2", "groupId": "g2", "quantity": 7},
            ],
        }
    }
    assert not _candidate_rows(raw, binding, target)


def test_invalid_declared_allocation_topology_is_rejected() -> None:
    raw = "--- PAGE 1 ---\n1-7\n"
    target = _allocated_target(package_quantity=7, allocations=(3, 3))
    with pytest.raises(ValueError, match="allocation total differs"):
        _candidate_rows(raw, (_draft(raw, "1-7"),), target)


def test_missing_or_unknown_decisions_fail_closed() -> None:
    raw = "--- PAGE 1 ---\n1-7\n"
    binding = (_draft(raw, "1-7"),)
    rows = _candidate_rows(raw, binding, _target(quantity=7))
    with pytest.raises(CoherenceContractError, match="omitted required"):
        materialize_coherence_decisions(candidates=rows, constraints=(), decisions=())
    with pytest.raises(CoherenceContractError, match="unknown candidates"):
        materialize_coherence_decisions(
            candidates=rows,
            constraints=(),
            decisions=(
                CoherenceCandidateDecision(
                    candidate_id="review_candidate_9999",
                    disposition="reviewed_independent",
                    rationale="Unknown fixture decision.",
                ),
            ),
        )


def test_binding_revision_invalidates_only_stale_relationship_decisions() -> None:
    raw = "--- PAGE 1 ---\n1-7\nREFERENCE ABC\n"
    original = _draft(raw, "1-7", logical_key="agent:old_range")
    rows = _candidate_rows(raw, (original,), _target(quantity=7))
    constraints = _apply_all(rows)

    renamed = _draft(raw, "1-7", logical_key="agent:revised_range")
    reconciled = reconcile_coherence_constraints_after_binding_revision(
        raw=raw,
        bindings=(renamed,),
        source_target=_target(quantity=7),
        constraints=constraints,
    )

    assert reconciled == ()
    validate_coherence_contracts(
        raw=raw,
        bindings=(renamed,),
        source_target=_target(quantity=7),
        constraints=reconciled,
        require_complete=False,
    )
    with pytest.raises(CoherenceContractError, match="lack dispositions"):
        validate_coherence_contracts(
            raw=raw,
            bindings=(renamed,),
            source_target=_target(quantity=7),
            constraints=reconciled,
        )


def test_unrelated_binding_revision_preserves_relationship_decision() -> None:
    raw = "--- PAGE 1 ---\n1-7\nREFERENCE ABC\n"
    range_binding = _draft(raw, "1-7")
    rows = _candidate_rows(raw, (range_binding,), _target(quantity=7))
    constraints = _apply_all(rows)
    unrelated = _draft(
        raw,
        "REFERENCE ABC",
        logical_key="agent:reference",
        render_mode="deterministic_auxiliary",
        value_kind="identifier",
        group_kind="customs",
        group_key="customs:reference",
    )

    reconciled = reconcile_coherence_constraints_after_binding_revision(
        raw=raw,
        bindings=(range_binding, unrelated),
        source_target=_target(quantity=7),
        constraints=constraints,
    )

    assert reconciled == constraints
    validate_coherence_contracts(
        raw=raw,
        bindings=(range_binding, unrelated),
        source_target=_target(quantity=7),
        constraints=reconciled,
    )
